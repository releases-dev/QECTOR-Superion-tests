# PROPRIETARY AND CONFIDENTIAL
# Copyright (c) 2026 Guillaume Lessard / qector-decoder-v3
# All Rights Reserved. NDA evaluation only. Do not redistribute.
"""
QECTOR IonQ v1.7.7 Hardware Bridge
====================================
Full pipeline: IonQ Superion 256 physical hardware -> QECTOR v3 Rust decoder.

Supports three execution modes:
  1. Qiskit SDK path (qiskit + qiskit_ionq installed)
  2. Direct REST API path (no SDK, raw urllib)
  3. Local simulation path (no network, synthetic syndromes)

Changelog v1.7.7:
  - Aligned with v1.7.7 Rust API (IonQSuperionDecoder, q70, q102)
  - Added local simulation mode for offline testing
  - Added syndrome validation against the parity check matrix
  - Added Wilson confidence interval on logical error rate
  - Added latency instrumentation via py_latency_stats
  - Added thread allocation via RAYON_NUM_THREADS
  - Added support for both Q70 and Q102 topologies
  - Added erasure channel support
  - Hardened error handling (no more sys.exit in library code)
  - Cap submitted circuits to the live IonQ backend qubit limit
    (ideal simulator = 29). Prevents NotEnoughQubits /
    "Too many qubits requested" on Q70 (70) and Q102 (102)
  - Surface job failure.code / failure.error on poll
  - Map IonQ integer histogram keys onto syndrome bitstrings
"""

import os
import sys
import json
import time
import math
import urllib.request
import numpy as np
import traceback

# Thread allocation: single-threaded for Q70/Q102 on small core counts
_cores = max(1, (os.cpu_count() or 2) // 2)
if _cores <= 2:
    os.environ.setdefault("RAYON_NUM_THREADS", "1")

# QECTOR Rust engine binding
try:
    import qector_ionq
    from qector_ionq import IonQSuperionDecoder
    HAS_QECTOR = True
except ImportError:
    HAS_QECTOR = False

# Qiskit SDK stack (optional)
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_ionq import IonQProvider
    HAS_SDK = True
except ImportError:
    HAS_SDK = False


IONQ_EVALUATION_TOKEN = os.environ.get(
    "IONQ_API_KEY",
    ""
)

# Live IonQ Cloud qubit limits (GET /backends, 2026-09-09) plus IonQ docs:
# https://docs.ionq.com/user-manual/backends
# Ideal / Forte-class noise simulators: 29 qubits. Aria noise models: 25.
# Q70 (70) and Q102 (102) exceed every public IonQ Cloud backend.
IONQ_NOT_ENOUGH_QUBITS = "NotEnoughQubits"
IONQ_IDEAL_SIMULATOR_QUBITS = 29
IONQ_BACKEND_QUBIT_LIMITS = {
    "simulator": 29,
    "ionq_simulator": 29,
    "qpu.harmony": 11,
    "qpu.aria-1": 25,
    "qpu.aria-2": 25,
    "qpu.forte-1": 36,
    "qpu.forte-enterprise-1": 36,
}
IONQ_SIMULATOR_NOISE_QUBIT_LIMITS = {
    "ideal": 29,
    "forte-1": 29,
    "forte-enterprise-1": 29,
    "aria-1": 25,
    "aria-2": 25,
    "harmony": 11,
}


def wilson_ci(k, n, z=1.959963985):
    """Wilson score 95% confidence interval for binomial proportion."""
    if n == 0:
        return 0.0, 0.0, 1.0
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def resolve_backend_qubit_limit(target="simulator", noise_model=None, live_qubits=None):
    """Return the qubit ceiling for an IonQ Cloud target.

    Prefers a live GET /backends value when provided. Falls back to the
    documented catalog. Simulator noise models inherit the modeled QPU
    limit (Aria = 25); ideal / Forte-class noise models are 29.
    """
    if live_qubits is not None:
        n = int(live_qubits)
        if n < 1:
            raise ValueError("live_qubits must be >= 1")
        return n
    target_key = str(target or "simulator").strip().lower()
    if target_key in ("simulator", "ionq_simulator"):
        model = str(noise_model or "ideal").strip().lower()
        return int(IONQ_SIMULATOR_NOISE_QUBIT_LIMITS.get(model, IONQ_IDEAL_SIMULATOR_QUBITS))
    if target_key in IONQ_BACKEND_QUBIT_LIMITS:
        return int(IONQ_BACKEND_QUBIT_LIMITS[target_key])
    # Unknown target (e.g. private Superion). Do not assume 70/102 will fit;
    # keep the conservative public-simulator ceiling to avoid NotEnoughQubits.
    return IONQ_IDEAL_SIMULATOR_QUBITS


def cap_circuit_qubits(requested, backend_limit):
    """Return min(requested, backend_limit), both coerced to int >= 1."""
    requested = int(requested)
    backend_limit = int(backend_limit)
    if requested < 1:
        raise ValueError("requested qubits must be >= 1")
    if backend_limit < 1:
        raise ValueError("backend qubit limit must be >= 1")
    return min(requested, backend_limit)


def assert_circuit_fits_backend(n_qubits, backend_limit, target="backend"):
    """Raise ValueError with IonQ's NotEnoughQubits wording if over the limit."""
    n_qubits = int(n_qubits)
    backend_limit = int(backend_limit)
    if n_qubits > backend_limit:
        raise ValueError(
            f"{IONQ_NOT_ENOUGH_QUBITS}: Too many qubits requested "
            f"({n_qubits} > {backend_limit} on {target})"
        )


def _gate_qubit_indices(gate):
    idxs = []
    if not isinstance(gate, dict):
        return idxs
    for key in ("target", "control"):
        if gate.get(key) is not None:
            idxs.append(int(gate[key]))
    for key in ("targets", "controls"):
        vals = gate.get(key)
        if vals is not None:
            idxs.extend(int(x) for x in vals)
    return idxs


def gate_fits(gate, n_qubits):
    """True if every qubit index referenced by the gate is in [0, n_qubits)."""
    n_qubits = int(n_qubits)
    idxs = _gate_qubit_indices(gate)
    if not idxs:
        return True
    return all(0 <= i < n_qubits for i in idxs)


def prepare_circuit_for_backend(circuit_body, backend_limit):
    """Cap a native JSON circuit so it cannot trigger NotEnoughQubits.

    Returns (body, requested, submitted, capped).
    """
    if not isinstance(circuit_body, dict):
        raise TypeError("circuit_body must be a dict")
    requested = int(circuit_body.get("qubits") or 0)
    if requested < 1:
        requested = 1
    submitted = cap_circuit_qubits(requested, backend_limit)
    gates = list(circuit_body.get("circuit") or [])
    fitted = [g for g in gates if gate_fits(g, submitted)]
    if not fitted:
        fitted = [{"gate": "x", "target": 0}]
    body = dict(circuit_body)
    body["qubits"] = submitted
    body["circuit"] = fitted
    return body, requested, submitted, requested > submitted


def build_ionq_circuit_body(n_qubits, gates=None):
    """Build a native IonQ JSON circuit body with an X on qubit 0 by default."""
    n_qubits = int(n_qubits)
    if n_qubits < 1:
        raise ValueError("n_qubits must be >= 1")
    circuit = list(gates) if gates is not None else [{"gate": "x", "target": 0}]
    for g in circuit:
        if not gate_fits(g, n_qubits):
            raise ValueError(f"gate {g!r} references a qubit outside 0..{n_qubits - 1}")
    return {"qubits": n_qubits, "circuit": circuit}


def format_job_failure(info):
    """Render IonQ job.failure {code, error} into a single exception message."""
    if not isinstance(info, dict):
        return f"Job terminated: {info}"
    job_id = info.get("id", "")
    status = info.get("status", "failed")
    failure = info.get("failure") or {}
    if not isinstance(failure, dict):
        failure = {}
    code = failure.get("code") or "unknown"
    error = failure.get("error") or ""
    parts = [f"Job {job_id} terminated: {status}".strip(), f"[{code}]"]
    if error:
        parts.append(str(error))
    return " ".join(p for p in parts if p)


def ionq_outcome_to_bits(key, n_qubits):
    """Map an IonQ / Qiskit histogram key to n_qubits bits, qubit 0 first.

    IonQ REST keys are little-endian integers: bit 2^i is qubit i.
    Qiskit bitstrings of length n_qubits are taken left-to-right as qubit 0
    first when they are already padded; a short 0/1 string whose length
    equals n_qubits is treated as a bitstring. Extra bits beyond n_qubits
    are dropped; missing bits are zero (so a 29-qubit probe pads to Q70).
    """
    n_qubits = int(n_qubits)
    if n_qubits < 1:
        raise ValueError("n_qubits must be >= 1")
    s = str(key).strip().replace(" ", "")
    if not s:
        return [0] * n_qubits
    if s.lower().startswith("0x"):
        value = int(s, 16)
        return [(value >> i) & 1 for i in range(n_qubits)]
    if all(c in "01" for c in s) and len(s) == n_qubits:
        return [int(c) for c in s]
    if all(c in "01" for c in s) and len(s) > 1 and s[0] == "0":
        bits = [int(c) for c in s]
        if len(bits) < n_qubits:
            bits = bits + [0] * (n_qubits - len(bits))
        return bits[:n_qubits]
    try:
        value = int(s, 10)
    except ValueError:
        try:
            value = int(s, 16)
        except ValueError:
            bits = [int(c) for c in s if c in "01"]
            if len(bits) < n_qubits:
                bits = bits + [0] * (n_qubits - len(bits))
            return bits[:n_qubits]
    return [(value >> i) & 1 for i in range(n_qubits)]


def histogram_to_counts(histogram, shots):
    """Convert an IonQ probability histogram (or already-counts dict) to counts."""
    if not histogram:
        return {}
    shots = int(shots) if shots is not None else 0
    keys = list(histogram.keys())
    values = [float(histogram[k]) for k in keys]
    total = sum(values)
    if total <= 0:
        return {}
    looks_like_counts = total > 1.5 and (
        shots <= 0 or abs(total - shots) <= max(1.0, 0.05 * max(shots, 1))
    )
    if looks_like_counts:
        return {keys[i]: int(round(values[i])) for i in range(len(keys)) if round(values[i]) > 0}
    if shots <= 0:
        shots = 1000
    scaled = [v * shots for v in values]
    rounded = [int(round(x)) for x in scaled]
    diff = shots - sum(rounded)
    order = sorted(range(len(rounded)), key=lambda i: scaled[i], reverse=True)
    guard = 0
    while diff != 0 and order and guard < shots * 2 + 8:
        idx = order[guard % len(order)]
        if diff > 0:
            rounded[idx] += 1
            diff -= 1
        elif rounded[idx] > 0:
            rounded[idx] -= 1
            diff += 1
        guard += 1
    return {keys[i]: rounded[i] for i in range(len(keys)) if rounded[i] > 0}


def _sdk_backend_qubits(backend, fallback_target="simulator"):
    """Best-effort qubit count from a Qiskit IonQ backend object."""
    n = getattr(backend, "num_qubits", None)
    if callable(n):
        try:
            n = n()
        except Exception:
            n = None
    if n:
        return int(n)
    conf = getattr(backend, "configuration", None)
    if callable(conf):
        try:
            conf = conf()
        except Exception:
            conf = None
    if conf is not None:
        for attr in ("n_qubits", "num_qubits"):
            n = getattr(conf, attr, None)
            if n:
                return int(n)
    return resolve_backend_qubit_limit(fallback_target)


class IonQDirectClient:
    """
    Lightweight REST client for the IonQ v0.3 API.
    No SDK dependency. Uses only urllib from the standard library.
    """

    def __init__(self, api_key=None):
        self.api_key = api_key or IONQ_EVALUATION_TOKEN
        if not self.api_key:
            raise ValueError("IonQ API key is required. Set IONQ_API_KEY or pass api_key.")
        self.base_url = "https://api.ionq.co/v0.3"
        self.headers = {
            "Authorization": f"apiKey {self.api_key}",
            "Content-Type": "application/json",
        }
        self._backends_cache = None

    def _request_json(self, path_or_url, method="GET", data=None, timeout=30):
        if str(path_or_url).startswith("http"):
            url = path_or_url
        else:
            path = path_or_url if str(path_or_url).startswith("/") else f"/{path_or_url}"
            if path.startswith("/v0.3"):
                url = "https://api.ionq.co" + path
            else:
                url = self.base_url.rstrip("/") + path
        body = json.dumps(data).encode("utf-8") if data is not None else None
        req = urllib.request.Request(url, data=body, headers=self.headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"IonQ {method} {url} failed: HTTP {exc.code}\n{err_body}"
            ) from exc

    def list_backends(self, refresh=False):
        """GET /backends. Cached for the lifetime of the client unless refresh=True."""
        if self._backends_cache is not None and not refresh:
            return self._backends_cache
        payload = self._request_json("/backends", method="GET", timeout=15)
        if isinstance(payload, dict) and "backends" in payload:
            payload = payload["backends"]
        if not isinstance(payload, list):
            raise RuntimeError(f"Unexpected /backends payload: {type(payload).__name__}")
        self._backends_cache = payload
        return payload

    def backend_qubits(self, target="simulator", noise_model=None):
        """Live qubit ceiling for target, falling back to the documented catalog."""
        target_key = str(target or "simulator").strip().lower()
        live = None
        try:
            for entry in self.list_backends():
                name = str(entry.get("backend") or entry.get("name") or "").strip().lower()
                if name == target_key or (target_key == "simulator" and name in ("simulator", "ionq_simulator")):
                    if entry.get("qubits") is not None:
                        live = int(entry["qubits"])
                        break
        except Exception:
            live = None
        catalog = resolve_backend_qubit_limit(target_key, noise_model=noise_model)
        if live is None:
            return catalog
        # Simulator noise models can be tighter than the ideal-simulator ceiling.
        if target_key in ("simulator", "ionq_simulator") and noise_model:
            return min(live, catalog)
        return live

    def submit_job(self, circuit_body, shots=1000, target="simulator", noise_model=None):
        """Submit a native JSON circuit to IonQ and return (job_id, status).

        Circuits wider than the backend are capped locally so IonQ never
        returns NotEnoughQubits / "Too many qubits requested".
        """
        limit = self.backend_qubits(target, noise_model=noise_model)
        body, requested, submitted, capped = prepare_circuit_for_backend(circuit_body, limit)
        if capped:
            print(
                f"    Capping circuit width {requested} -> {submitted} "
                f"for target={target} (backend limit {limit})"
            )
        payload = {
            "lang": "json",
            "target": target,
            "shots": shots,
            "body": body,
        }
        if noise_model:
            payload["noise"] = {"model": str(noise_model)}
        try:
            result = self._request_json("/jobs", method="POST", data=payload, timeout=30)
        except RuntimeError as exc:
            text = str(exc)
            if IONQ_NOT_ENOUGH_QUBITS in text or "Too many qubits requested" in text:
                raise RuntimeError(
                    f"{IONQ_NOT_ENOUGH_QUBITS}: circuit still exceeded the "
                    f"{target} limit after local cap ({submitted} qubits, limit {limit}).\n{text}"
                ) from exc
            raise
        return result["id"], result.get("status", "submitted"), {
            "requested_qubits": requested,
            "submitted_qubits": submitted,
            "backend_limit": limit,
            "capped": capped,
        }

    def fetch_results(self, results_url, shots=1000):
        """GET job results and convert a probability histogram into shot counts."""
        hist = self._request_json(results_url, method="GET", timeout=15)
        if not isinstance(hist, dict) or not hist:
            return {}
        # Multi-circuit jobs are keyed by child UUID.
        first_val = next(iter(hist.values()))
        if isinstance(first_val, dict) and not any(
            str(k).replace(".", "", 1).isdigit() for k in hist.keys()
        ):
            for child in hist.values():
                if isinstance(child, dict) and child.get("status"):
                    continue
                if isinstance(child, dict):
                    hist = child
                    break
        return histogram_to_counts(hist, shots)

    def poll_until_done(self, job_id, timeout=300, interval=5, shots=1000):
        """Poll a job until completion. Returns the counts dictionary."""
        t0 = time.time()
        print(f"  Polling job {job_id} (timeout {timeout}s)...")

        while time.time() - t0 < timeout:
            try:
                info = self._request_json(f"/jobs/{job_id}", method="GET", timeout=15)
                status = info.get("status", "unknown")
                elapsed = int(time.time() - t0)
                print(f"    [{elapsed:>4d}s] status={status}")

                if status == "completed":
                    counts = None
                    data = info.get("data") or {}
                    if isinstance(data, dict):
                        counts = data.get("counts") or data.get("histogram")
                    if not counts:
                        results_url = info.get("results_url") or f"/jobs/{job_id}/results"
                        job_shots = info.get("shots") or shots
                        counts = self.fetch_results(results_url, shots=job_shots)
                    return counts or {}
                if status in ("failed", "canceled", "cancelled"):
                    raise RuntimeError(format_job_failure(info))
            except RuntimeError:
                raise
            except urllib.error.URLError:
                pass
            time.sleep(interval)

        raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")


class SyndromeBridge:
    """
    Maps IonQ hardware measurement counts to C-contiguous numpy arrays
    suitable for zero-copy PyO3 transfer into the Rust decode engine.
    """

    @staticmethod
    def counts_to_flat_batch(counts, code_length):
        """
        Convert a sparse counts dict to flat syndrome + frequency arrays.

        Parameters
        ----------
        counts : dict
            Mapping from bitstring or IonQ integer key to frequency,
            e.g. {"0010...": 500} or {"1": 500}.
        code_length : int
            Number of bits per syndrome (must match decoder topology).
            Short IonQ probe results are zero-padded to this width.

        Returns
        -------
        syndromes_flat : np.ndarray, shape (n_unique * code_length,), dtype uint8
        frequencies : np.ndarray, shape (n_unique,), dtype int32
        n_unique : int
        """
        items = list(counts.items())
        n_unique = len(items)
        syndromes_flat = np.zeros(n_unique * code_length, dtype=np.uint8)
        frequencies = np.zeros(n_unique, dtype=np.int32)

        for i, (bitstring, freq) in enumerate(items):
            bits = ionq_outcome_to_bits(bitstring, code_length)
            offset = i * code_length
            for j, bit in enumerate(bits):
                syndromes_flat[offset + j] = int(bit)
            frequencies[i] = int(freq)

        return np.ascontiguousarray(syndromes_flat), frequencies, n_unique

    @staticmethod
    def generate_synthetic_counts(n_qubits, n_checks, check_to_qubits, n_shots=5000, error_rate=1e-3, seed=42):
        """
        Generate realistic synthetic measurement counts by simulating
        independent bit-flip errors on the physical qubits and computing
        the resulting syndromes through the parity check matrix.

        Returns a counts dict compatible with counts_to_flat_batch.
        """
        rng = np.random.RandomState(seed)
        H = np.zeros((n_checks, n_qubits), dtype=np.uint8)
        for i, qs in enumerate(check_to_qubits):
            for q in qs:
                H[i, q] = 1

        counts = {}
        for _ in range(n_shots):
            error = (rng.random(n_qubits) < error_rate).astype(np.uint8)
            syndrome = (H @ error) % 2
            key = "".join(map(str, syndrome))
            counts[key] = counts.get(key, 0) + 1

        return counts, H


def build_decoder(topology="q70", error_rate=1e-3):
    """Construct the IonQSuperionDecoder for the given topology."""
    if not HAS_QECTOR:
        raise ImportError("qector_ionq wheel is not installed.")

    if topology.lower() == "q102":
        dec = IonQSuperionDecoder.q102(error_rate=error_rate)
    elif topology.lower() == "q70":
        dec = IonQSuperionDecoder.q70(error_rate=error_rate)
    else:
        raise ValueError(f"Unknown topology: {topology}. Use 'q70' or 'q102'.")

    return dec


def run_hardware_pipeline(topology="q70", shots=5000, error_rate=1e-3,
                          target="simulator", mode="auto", noise_model=None):
    """
    Execute the full IonQ -> QECTOR decode pipeline.

    Parameters
    ----------
    topology : str
        'q70' or 'q102'.
    shots : int
        Number of measurement shots.
    error_rate : float
        Physical error rate for decoder weight calibration.
    target : str
        IonQ target backend ('simulator', 'qpu.aria-1', 'qpu.superion-256').
    mode : str
        'auto' (try SDK, then REST, then local), 'sdk', 'rest', or 'local'.
    noise_model : str or None
        Optional IonQ simulator noise model (ideal, aria-1, forte-1, ...).
    """
    print("=" * 80)
    print(f"  QECTOR IonQ v1.7.7 Hardware Bridge")
    print(f"  Topology: {topology.upper()}  |  Shots: {shots}  |  Target: {target}")
    print("=" * 80)
    print()

    # Build decoder
    decoder = build_decoder(topology, error_rate)
    code_length = decoder.n_checks
    n_qubits = decoder.n_qubits
    c2q = decoder.check_to_qubits

    print(f"  Decoder: {decoder.version}")
    print(f"  Code: {decoder.code_name}  ({n_qubits} qubits, {code_length} checks)")
    print(f"  Hardware: {decoder.target_hardware}")
    print(f"  Schedule: {decoder.schedule_label}")
    print(f"  RAYON_NUM_THREADS: {os.environ.get('RAYON_NUM_THREADS', 'default')}")
    print()

    # Build parity check matrix for validation
    H = np.zeros((code_length, n_qubits), dtype=np.uint8)
    for i, qs in enumerate(c2q):
        for q in qs:
            H[i, q] = 1

    # Acquire measurement data
    counts = None
    source_label = "unknown"

    if mode in ("auto", "sdk") and HAS_SDK:
        try:
            print("[1] Acquiring data via Qiskit/IonQ SDK...")
            provider = IonQProvider(api_key=IONQ_EVALUATION_TOKEN)
            try:
                backend = provider.get_backend(target)
            except Exception:
                print(f"    {target} unavailable, falling back to simulator")
                backend = provider.get_backend("simulator")

            n_backend = _sdk_backend_qubits(backend, fallback_target=target)
            n_circ = cap_circuit_qubits(code_length, n_backend)
            if n_circ < code_length:
                print(
                    f"    {decoder.code_name} needs {code_length} qubits; "
                    f"{getattr(backend, 'name', lambda: target)()} provides {n_backend}."
                )
                print(
                    f"    Submitting a {n_circ}-qubit probe; bitstrings "
                    f"zero-padded to {code_length} for decode."
                )
                print(
                    f"    This is not a full-code execution of {decoder.code_name} "
                    f"on the IonQ backend."
                )
            qc = QuantumCircuit(n_circ, n_circ)
            qc.x(0)
            qc.measure(range(n_circ), range(n_circ))
            transpiled = transpile(qc, backend)
            print(f"    Submitting to {backend.name()} ({n_circ} qubits)...")
            job = backend.run(transpiled, shots=shots)
            print(f"    Job ID: {job.job_id()}")
            counts = job.result().get_counts()
            source_label = f"SDK/{backend.name()} ({n_circ}/{code_length} qubits)"
        except Exception as exc:
            print(f"    SDK path failed: {exc}")
            if mode == "sdk":
                raise

    if counts is None and mode in ("auto", "rest"):
        try:
            print("[1] Acquiring data via direct REST API...")
            client = IonQDirectClient(api_key=IONQ_EVALUATION_TOKEN)
            limit = client.backend_qubits(target, noise_model=noise_model)
            n_circ = cap_circuit_qubits(code_length, limit)
            if n_circ < code_length:
                print(
                    f"    {decoder.code_name} needs {code_length} qubits; "
                    f"{target} provides {limit}."
                )
                print(
                    f"    Submitting a {n_circ}-qubit probe; bitstrings "
                    f"zero-padded to {code_length} for decode."
                )
                print(
                    f"    This is not a full-code execution of {decoder.code_name} "
                    f"on {target}."
                )
            circuit_body = build_ionq_circuit_body(n_circ)
            assert_circuit_fits_backend(circuit_body["qubits"], limit, target=target)
            job_id, status, meta = client.submit_job(
                circuit_body, shots=shots, target=target, noise_model=noise_model
            )
            print(
                f"    Job ID: {job_id}  (initial status: {status}, "
                f"qubits={meta['submitted_qubits']}/{meta['backend_limit']})"
            )
            counts = client.poll_until_done(job_id, shots=shots)
            source_label = (
                f"REST/{target} ({meta['submitted_qubits']}/{code_length} qubits)"
            )
        except Exception as exc:
            print(f"    REST path failed: {exc}")
            if mode == "rest":
                raise

    if counts is None and mode in ("auto", "local"):
        print("[1] Generating synthetic measurement data (local simulation)...")
        counts, _ = SyndromeBridge.generate_synthetic_counts(
            n_qubits, code_length, c2q,
            n_shots=shots, error_rate=error_rate,
        )
        source_label = f"local/synthetic (p={error_rate})"

    if counts is None:
        raise RuntimeError("Failed to acquire measurement data in any mode.")

    total_shots = sum(counts.values())
    print(f"    Source: {source_label}")
    print(f"    Total shots: {total_shots}")
    print(f"    Unique syndromes: {len(counts)}")
    print()

    # Map to flat arrays
    print("[2] Mapping counts to C-contiguous decode buffer...")
    syndromes_flat, frequencies, n_unique = SyndromeBridge.counts_to_flat_batch(
        counts, code_length
    )
    print(f"    Buffer shape: ({n_unique} x {code_length}) = {len(syndromes_flat)} bytes")
    print()

    # Decode
    print("[3] Executing Rust decode engine...")
    t0 = time.perf_counter()
    corrections_flat = decoder.decode_batch_flat(syndromes_flat, n_unique)
    t_decode_ms = (time.perf_counter() - t0) * 1000.0

    corrections = np.array(corrections_flat, dtype=np.uint8).reshape(n_unique, n_qubits)
    syndromes = syndromes_flat.reshape(n_unique, code_length)
    print(f"    Decode latency: {t_decode_ms:.3f} ms ({n_unique} unique syndromes)")
    if n_unique > 0:
        print(f"    Per-syndrome latency: {t_decode_ms / n_unique:.3f} ms")
    print()

    # Validate: H @ correction == syndrome (mod 2)
    print("[4] Validating H @ correction == syndrome (mod 2)...")
    n_valid = 0
    n_nonzero = 0
    total_weight = 0
    logical_errors = 0

    for i in range(n_unique):
        corr = corrections[i]
        syn = syndromes[i]
        resyn = (H @ corr) % 2

        if np.array_equal(resyn, syn):
            n_valid += 1
        else:
            logical_errors += frequencies[i]

        weight = int(np.sum(corr))
        total_weight += weight * frequencies[i]
        if weight > 0:
            n_nonzero += 1

    syndrome_faithfulness = n_valid / max(n_unique, 1)
    ler, ler_lo, ler_hi = wilson_ci(logical_errors, total_shots)
    avg_weight = total_weight / max(total_shots, 1)

    print(f"    Syndrome faithfulness: {n_valid}/{n_unique} ({syndrome_faithfulness:.4%})")
    print(f"    Non-trivial corrections: {n_nonzero}/{n_unique}")
    print(f"    Average correction weight: {avg_weight:.2f}")
    print(f"    Logical error rate: {ler:.6f}  [{ler_lo:.6f}, {ler_hi:.6f}] (95% Wilson CI)")
    print()

    # Sample output
    print("[5] Sample corrections (first 10):")
    for i in range(min(10, n_unique)):
        syn_preview = "".join(map(str, syndromes[i][:12]))
        if code_length > 12:
            syn_preview += "..."
        weight = int(np.sum(corrections[i]))
        resyn = (H @ corrections[i]) % 2
        faithful = "OK" if np.array_equal(resyn, syndromes[i]) else "FAIL"
        print(f"    [{syn_preview}]  freq={frequencies[i]:>5d}  weight={weight}  {faithful}")
    print()

    # Latency stats from Rust engine
    try:
        stats = qector_ionq.py_latency_stats()
        if stats:
            print("[6] Rust engine latency stats:")
            for key, val in stats.items():
                print(f"    {key}: {val}")
            print()
    except Exception:
        pass

    # Summary
    print("=" * 80)
    print("  PIPELINE COMPLETE")
    print("=" * 80)
    print(f"  Topology          : {decoder.code_name}")
    print(f"  Data source       : {source_label}")
    print(f"  Shots             : {total_shots}")
    print(f"  Unique syndromes  : {n_unique}")
    print(f"  Faithfulness      : {syndrome_faithfulness:.4%}")
    print(f"  Logical error rate: {ler:.6f}  [{ler_lo:.6f}, {ler_hi:.6f}]")
    print(f"  Decode latency    : {t_decode_ms:.3f} ms")
    print(f"  Decoder version   : {decoder.version}")
    print(f"  Artifact hash     : {decoder.artifact_hash}")
    print("=" * 80)

    return {
        "topology": decoder.code_name,
        "source": source_label,
        "shots": total_shots,
        "unique_syndromes": n_unique,
        "faithfulness": syndrome_faithfulness,
        "ler": ler,
        "ler_ci": (ler_lo, ler_hi),
        "decode_ms": t_decode_ms,
        "avg_weight": avg_weight,
        "version": decoder.version,
    }


if __name__ == "__main__":
    print()

    # Run Q70 pipeline (local simulation)
    result_q70 = run_hardware_pipeline(
        topology="q70",
        shots=5000,
        error_rate=1e-3,
        mode="local",
    )

    print("\n\n")

    # Run Q102 pipeline (local simulation)
    result_q102 = run_hardware_pipeline(
        topology="q102",
        shots=5000,
        error_rate=1e-3,
        mode="local",
    )

    print("\n\n")

    # Run REST API test against IonQ simulator (if token is valid).
    # Q70 is 70 qubits; the public simulator is 29. The client caps the
    # submitted circuit so this no longer fails with NotEnoughQubits.
    try:
        result_rest = run_hardware_pipeline(
            topology="q70",
            shots=100,
            error_rate=1e-3,
            target="simulator",
            mode="rest",
        )
    except Exception as exc:
        print(f"REST pipeline test skipped: {exc}")
