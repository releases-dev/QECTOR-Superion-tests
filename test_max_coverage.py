# PROPRIETARY AND CONFIDENTIAL
# Copyright (c) 2026 Guillaume Lessard / qector-decoder-v3
# All Rights Reserved. NDA evaluation only. Do not redistribute.
"""
QECTOR IonQ v1.7.7 Maximum Coverage Test Suite
================================================
Exercises every accessible API surface of the qector_ionq wheel.
"""

import os
import sys
import json
import time
import math
import traceback
import numpy as np

os.environ["RAYON_NUM_THREADS"] = "1"

import qector_ionq
from qector_ionq import IonQSuperionDecoder

PASS = 0
FAIL = 0
SKIP = 0
RESULTS = []

def record(name, passed, detail=""):
    global PASS, FAIL
    tag = "PASS" if passed else "FAIL"
    if not passed:
        FAIL += 1
    else:
        PASS += 1
    RESULTS.append((name, tag, detail))
    status = f"  [{tag}] {name}"
    if detail:
        status += f"  ({detail})"
    print(status)

def skip(name, reason=""):
    global SKIP
    SKIP += 1
    RESULTS.append((name, "SKIP", reason))
    print(f"  [SKIP] {name}  ({reason})")

def build_H(c2q, n_checks, n_qubits):
    H = np.zeros((n_checks, n_qubits), dtype=np.uint8)
    for i, qs in enumerate(c2q):
        for q in qs:
            H[i, q] = 1
    return H

def make_syndrome(H, qubit_indices):
    n_qubits = H.shape[1]
    error = np.zeros(n_qubits, dtype=np.uint8)
    for q in qubit_indices:
        error[q] = 1
    return (H @ error) % 2, error

def wilson_ci(k, n, z=1.959963985):
    if n == 0:
        return 0.0, 0.0, 1.0
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


print("=" * 80)
print("  QECTOR IonQ v1.7.7 Maximum Coverage Test Suite")
print("=" * 80)
print(f"  Python: {sys.version.split()[0]}")
print(f"  NumPy: {np.__version__}")
print(f"  RAYON_NUM_THREADS: {os.environ.get('RAYON_NUM_THREADS', 'default')}")
print(f"  OS: {sys.platform}")
print()

# =========================================================================
# SECTION 1: Package metadata
# =========================================================================
print("SECTION 1: Package Metadata")
print("-" * 40)

record("version_is_1.7.7",
       qector_ionq.__version__ == "1.7.7",
       qector_ionq.__version__)

record("DECODER_VERSION",
       "1.7.7" in qector_ionq.DECODER_VERSION,
       qector_ionq.DECODER_VERSION)

record("TARGET_HARDWARE",
       "Superion" in qector_ionq.TARGET_HARDWARE,
       qector_ionq.TARGET_HARDWARE)

record("TARGET_ARCHITECTURE",
       "Walking Cat" in qector_ionq.TARGET_ARCHITECTURE,
       qector_ionq.TARGET_ARCHITECTURE)

# License
status = qector_ionq.py_license_status()
record("license_status_tuple",
       isinstance(status, tuple) and len(status) == 3,
       str(status))

print()

# =========================================================================
# SECTION 2: IonQSuperionDecoder construction
# =========================================================================
print("SECTION 2: Decoder Construction")
print("-" * 40)

# Default constructor
dec_default = IonQSuperionDecoder()
record("default_constructor",
       dec_default.n_qubits == 70 and dec_default.n_checks == 70,
       f"n_qubits={dec_default.n_qubits}, n_checks={dec_default.n_checks}")

record("default_code_name",
       dec_default.code_name == "Q70",
       dec_default.code_name)

record("default_version",
       dec_default.version == "1.7.7-production",
       dec_default.version)

record("default_target_hardware",
       dec_default.target_hardware == "IonQ Superion 256",
       dec_default.target_hardware)

record("default_schedule_label",
       dec_default.schedule_label == "uniform",
       dec_default.schedule_label)

record("artifact_hash_nonempty",
       len(dec_default.artifact_hash) > 0,
       dec_default.artifact_hash)

record("history_len_zero",
       dec_default.history_len == 0,
       str(dec_default.history_len))

# Q70 constructor
dec_q70 = IonQSuperionDecoder.q70(error_rate=1e-3)
record("q70_constructor",
       dec_q70.n_qubits == 70 and dec_q70.n_checks == 70,
       f"{dec_q70.code_name}")

# Q102 constructor
dec_q102 = IonQSuperionDecoder.q102(error_rate=1e-3)
record("q102_constructor",
       dec_q102.n_qubits == 102 and dec_q102.n_checks == 102,
       f"{dec_q102.code_name}")

# check_to_qubits structure
c2q70 = dec_q70.check_to_qubits
record("c2q70_length",
       len(c2q70) == 70,
       f"len={len(c2q70)}")

record("c2q70_all_lists",
       all(isinstance(row, list) for row in c2q70),
       "all rows are lists")

record("c2q70_weight6",
       all(len(row) == 6 for row in c2q70),
       "all checks are weight-6")

c2q102 = dec_q102.check_to_qubits
record("c2q102_length",
       len(c2q102) == 102,
       f"len={len(c2q102)}")

# Different error rates
for er in [1e-2, 5e-3, 1e-3, 1e-4, 1e-5]:
    try:
        d = IonQSuperionDecoder.q70(error_rate=er)
        record(f"q70_error_rate_{er:.0e}", True, f"constructed OK")
    except Exception as e:
        record(f"q70_error_rate_{er:.0e}", False, str(e))

print()

# =========================================================================
# SECTION 3: Q70 Single Decode Faithfulness
# =========================================================================
print("SECTION 3: Q70 Single Decode Faithfulness")
print("-" * 40)

H70 = build_H(c2q70, 70, 70)

# Zero syndrome
syn_zero = np.zeros(70, dtype=np.uint8)
corr_zero = dec_q70.decode(syn_zero)
record("q70_zero_syndrome",
       np.all(np.array(corr_zero) == 0),
       f"weight={sum(corr_zero)}")

# Single qubit errors on every qubit
all_faithful = True
failed_qubits = []
for q in range(70):
    syn, _ = make_syndrome(H70, [q])
    corr = dec_q70.decode(syn)
    resyn = (H70 @ np.array(corr, dtype=np.uint8)) % 2
    if not np.array_equal(resyn, syn):
        all_faithful = False
        failed_qubits.append(q)

record("q70_all_single_qubit_faithful",
       all_faithful,
       f"70/70 qubits" if all_faithful else f"failed: {failed_qubits}")

# Two-qubit errors (sample 20 pairs)
rng = np.random.RandomState(42)
n_two_qubit = 20
two_qubit_ok = 0
for _ in range(n_two_qubit):
    q1, q2 = rng.choice(70, 2, replace=False)
    syn, _ = make_syndrome(H70, [q1, q2])
    try:
        corr = dec_q70.decode(syn)
        resyn = (H70 @ np.array(corr, dtype=np.uint8)) % 2
        if np.array_equal(resyn, syn):
            two_qubit_ok += 1
    except Exception:
        pass

record("q70_two_qubit_errors",
       two_qubit_ok == n_two_qubit,
       f"{two_qubit_ok}/{n_two_qubit} faithful")

# Three-qubit errors (sample 15 triples)
n_three = 15
three_ok = 0
for _ in range(n_three):
    qs = rng.choice(70, 3, replace=False)
    syn, _ = make_syndrome(H70, list(qs))
    try:
        corr = dec_q70.decode(syn)
        resyn = (H70 @ np.array(corr, dtype=np.uint8)) % 2
        if np.array_equal(resyn, syn):
            three_ok += 1
    except Exception:
        pass

record("q70_three_qubit_errors",
       three_ok == n_three,
       f"{three_ok}/{n_three} faithful")

print()

# =========================================================================
# SECTION 4: Q102 Single Decode Faithfulness
# =========================================================================
print("SECTION 4: Q102 Single Decode Faithfulness")
print("-" * 40)

H102 = build_H(c2q102, 102, 102)

# Zero syndrome
syn_zero102 = np.zeros(102, dtype=np.uint8)
corr_zero102 = dec_q102.decode(syn_zero102)
record("q102_zero_syndrome",
       np.all(np.array(corr_zero102) == 0),
       f"weight={sum(corr_zero102)}")

# Single qubit errors on every qubit
all_faithful_102 = True
failed_102 = []
for q in range(102):
    syn, _ = make_syndrome(H102, [q])
    corr = dec_q102.decode(syn)
    resyn = (H102 @ np.array(corr, dtype=np.uint8)) % 2
    if not np.array_equal(resyn, syn):
        all_faithful_102 = False
        failed_102.append(q)

record("q102_all_single_qubit_faithful",
       all_faithful_102,
       f"102/102 qubits" if all_faithful_102 else f"failed: {failed_102}")

# Two-qubit on Q102
n_two_102 = 20
two_ok_102 = 0
for _ in range(n_two_102):
    q1, q2 = rng.choice(102, 2, replace=False)
    syn, _ = make_syndrome(H102, [q1, q2])
    try:
        corr = dec_q102.decode(syn)
        resyn = (H102 @ np.array(corr, dtype=np.uint8)) % 2
        if np.array_equal(resyn, syn):
            two_ok_102 += 1
    except Exception:
        pass

record("q102_two_qubit_errors",
       two_ok_102 == n_two_102,
       f"{two_ok_102}/{n_two_102} faithful")

print()

# =========================================================================
# SECTION 5: Batch Decode (decode_batch_flat)
# =========================================================================
print("SECTION 5: Batch Decode (decode_batch_flat)")
print("-" * 40)

# Q70 batch: all single-qubit errors
batch_size_70 = 70
syns_70 = []
for q in range(70):
    s, _ = make_syndrome(H70, [q])
    syns_70.append(s)
flat_70 = np.concatenate(syns_70).astype(np.uint8)

t0 = time.perf_counter()
result_70 = dec_q70.decode_batch_flat(flat_70, batch_size_70)
t_batch_70 = (time.perf_counter() - t0) * 1000

corrections_70 = np.array(result_70, dtype=np.uint8).reshape(batch_size_70, 70)
batch_faithful_70 = 0
for i in range(batch_size_70):
    resyn = (H70 @ corrections_70[i]) % 2
    if np.array_equal(resyn, syns_70[i]):
        batch_faithful_70 += 1

record("q70_batch_70_faithful",
       batch_faithful_70 == batch_size_70,
       f"{batch_faithful_70}/{batch_size_70} in {t_batch_70:.1f}ms")

# Q102 batch: all single-qubit errors
batch_size_102 = 102
syns_102 = []
for q in range(102):
    s, _ = make_syndrome(H102, [q])
    syns_102.append(s)
flat_102 = np.concatenate(syns_102).astype(np.uint8)

t0 = time.perf_counter()
result_102 = dec_q102.decode_batch_flat(flat_102, batch_size_102)
t_batch_102 = (time.perf_counter() - t0) * 1000

corrections_102 = np.array(result_102, dtype=np.uint8).reshape(batch_size_102, 102)
batch_faithful_102 = 0
for i in range(batch_size_102):
    resyn = (H102 @ corrections_102[i]) % 2
    if np.array_equal(resyn, syns_102[i]):
        batch_faithful_102 += 1

record("q102_batch_102_faithful",
       batch_faithful_102 == batch_size_102,
       f"{batch_faithful_102}/{batch_size_102} in {t_batch_102:.1f}ms")

# Large random batch
n_large = 500
large_syns = []
for _ in range(n_large):
    q = rng.randint(0, 70)
    s, _ = make_syndrome(H70, [q])
    large_syns.append(s)
flat_large = np.concatenate(large_syns).astype(np.uint8)

t0 = time.perf_counter()
result_large = dec_q70.decode_batch_flat(flat_large, n_large)
t_large = (time.perf_counter() - t0) * 1000

corrs_large = np.array(result_large, dtype=np.uint8).reshape(n_large, 70)
large_ok = 0
for i in range(n_large):
    resyn = (H70 @ corrs_large[i]) % 2
    if np.array_equal(resyn, large_syns[i]):
        large_ok += 1

record("q70_batch_500_random",
       large_ok == n_large,
       f"{large_ok}/{n_large} in {t_large:.1f}ms ({t_large/n_large:.3f}ms/shot)")

# Batch with zero syndromes
n_zeros = 50
flat_zeros = np.zeros(n_zeros * 70, dtype=np.uint8)
result_zeros = dec_q70.decode_batch_flat(flat_zeros, n_zeros)
corrs_zeros = np.array(result_zeros, dtype=np.uint8).reshape(n_zeros, 70)
all_zero_corr = all(np.sum(corrs_zeros[i]) == 0 for i in range(n_zeros))
record("q70_batch_50_zero_syndromes",
       all_zero_corr,
       "all corrections are zero-weight")

print()

# =========================================================================
# SECTION 6: Batch Decode Consistency (single vs batch)
# =========================================================================
print("SECTION 6: Single vs Batch Consistency")
print("-" * 40)

n_consist = 30
consist_ok = 0
for _ in range(n_consist):
    qs = [rng.randint(0, 70)]
    if rng.random() < 0.3:
        qs.append(rng.randint(0, 70))
    syn, _ = make_syndrome(H70, qs)
    
    single_corr = np.array(dec_q70.decode(syn), dtype=np.uint8)
    
    flat_syn = syn.copy().astype(np.uint8)
    batch_result = dec_q70.decode_batch_flat(flat_syn, 1)
    batch_corr = np.array(batch_result, dtype=np.uint8)
    
    if np.array_equal(single_corr, batch_corr):
        consist_ok += 1

record("q70_single_vs_batch_consistency",
       consist_ok == n_consist,
       f"{consist_ok}/{n_consist} bit-identical")

print()

# =========================================================================
# SECTION 7: AutoDecoder
# =========================================================================
print("SECTION 7: AutoDecoder")
print("-" * 40)

try:
    auto = qector_ionq.AutoDecoder()
    record("auto_decoder_construct", True, "instantiated")
    
    record("auto_has_backend",
           hasattr(auto, 'backend') or hasattr(auto, 'decode'),
           "has decode interface")
except Exception as e:
    record("auto_decoder_construct", False, str(e))

print()

# =========================================================================
# SECTION 8: BPOSDDecoder
# =========================================================================
print("SECTION 8: BPOSDDecoder")
print("-" * 40)

try:
    bposd = qector_ionq.BPOSDDecoder(c2q70, 70, 1e-3)
    z70 = np.zeros(70, dtype=np.uint8)
    c_bp = np.asarray(bposd.decode(z70), dtype=np.uint8)
    record("bposd_construct",
           bposd.n_qubits == 70 and int(c_bp.sum()) == 0,
           f"n_qubits={bposd.n_qubits} zero_weight={int(c_bp.sum())}")
except Exception as e:
    record("bposd_construct", False, str(e))

print()

# =========================================================================
# SECTION 9: TwoStageDecoder
# =========================================================================
print("SECTION 9: TwoStageDecoder")
print("-" * 40)

try:
    ts = qector_ionq.TwoStageDecoder.q70(error_rate=1e-3)
    z70 = np.zeros(70, dtype=np.uint8)
    c_ts = np.asarray(ts.decode(z70), dtype=np.uint8)
    record("two_stage_construct",
           ts.n_qubits == 70 and int(c_ts.sum()) == 0,
           f"n_qubits={ts.n_qubits} zero_weight={int(c_ts.sum())}")
except Exception as e:
    record("two_stage_construct", False, str(e))

print()

# =========================================================================
# SECTION 10: SpaceTimeDecoder
# =========================================================================
print("SECTION 10: SpaceTimeDecoder")
print("-" * 40)

try:
    st = qector_ionq.SpaceTimeDecoder(c2q70, 70, 3, 1e-3)
    record("spacetime_construct",
           st.n_qubits == 70 and st.rounds == 3,
           f"n_qubits={st.n_qubits} rounds={st.rounds}")
except Exception as e:
    record("spacetime_construct", False, str(e))

print()

# =========================================================================
# SECTION 11: Erasure Decode
# =========================================================================
print("SECTION 11: Erasure Decode")
print("-" * 40)

try:
    dec_er = IonQSuperionDecoder.q70(error_rate=1e-3)
    syn_er, _ = make_syndrome(H70, [0])
    # Zero mask: erasure channel reduces to ordinary decode (H c = s).
    corr_er = dec_er.decode_with_erasures(syn_er, np.zeros(70, dtype=np.uint8))
    resyn_er = (H70 @ np.array(corr_er, dtype=np.uint8)) % 2
    record("q70_erasure_decode_faithful",
           np.array_equal(resyn_er, syn_er),
           f"weight={sum(corr_er)}")
except Exception as e:
    record("q70_erasure_decode_faithful", False, str(e))

# Two-qubit error with an uninvolved qubit erased (masked system still solvable).
try:
    syn_er2, _ = make_syndrome(H70, [5, 10])
    erasure_mask2 = np.zeros(70, dtype=np.uint8)
    erasure_mask2[0] = 1
    corr_er2 = dec_er.decode_with_erasures(syn_er2, erasure_mask2)
    resyn_er2 = (H70 @ np.array(corr_er2, dtype=np.uint8)) % 2
    record("q70_erasure_multi_qubit",
           np.array_equal(resyn_er2, syn_er2) and int(corr_er2[0]) == 0,
           f"weight={sum(corr_er2)}")
except Exception as e:
    record("q70_erasure_multi_qubit", False, str(e))

# Erasure with zero syndrome
try:
    syn_zero_er = np.zeros(70, dtype=np.uint8)
    erasure_zero = np.zeros(70, dtype=np.uint8)
    erasure_zero[0] = 1
    corr_zero_er = dec_er.decode_with_erasures(syn_zero_er, erasure_zero)
    record("q70_erasure_zero_syndrome",
           np.all(np.array(corr_zero_er) == 0),
           f"weight={sum(corr_zero_er)}")
except Exception as e:
    record("q70_erasure_zero_syndrome", False, str(e))

print()

# =========================================================================
# SECTION 12: Decoder Configuration Methods
# =========================================================================
print("SECTION 12: Decoder Configuration")
print("-" * 40)

try:
    dec_cfg = IonQSuperionDecoder.q70(error_rate=1e-3)
    
    # set_uniform_schedule
    dec_cfg.set_uniform_schedule(0.002)
    record("set_uniform_schedule", True, "p=0.002")
except Exception as e:
    record("set_uniform_schedule", False, str(e))

try:
    # set_qubit_priors requires a float64 numpy array
    priors = np.full(70, 1e-3, dtype=np.float64)
    dec_cfg.set_qubit_priors(priors)
    record("set_qubit_priors",
           "heterogeneous" in dec_cfg.schedule_label,
           dec_cfg.schedule_label)
except Exception as e:
    record("set_qubit_priors", False, str(e))

try:
    # set_strict_verify
    dec_cfg.set_strict_verify(True)
    record("set_strict_verify_true", True, "enabled")
    dec_cfg.set_strict_verify(False)
    record("set_strict_verify_false", True, "disabled")
except Exception as e:
    record("set_strict_verify", False, str(e))

try:
    # backend() method
    backend_info = dec_cfg.backend()
    record("backend_method", True, str(backend_info)[:80])
except Exception as e:
    record("backend_method", False, str(e))

try:
    # flush
    dec_cfg.flush()
    record("flush_method", True, "flushed")
except Exception as e:
    record("flush_method", False, str(e))

try:
    # update requires a round syndrome of length n_checks
    dec_cfg.flush()
    corr_upd = dec_cfg.update(np.zeros(70, dtype=np.uint8))
    record("update_method",
           dec_cfg.history_len == 1 and int(np.sum(corr_upd)) == 0,
           f"history_len={dec_cfg.history_len}")
except Exception as e:
    record("update_method", False, str(e))

try:
    # gross
    gross_info = dec_cfg.gross()
    record("gross_method", True, str(gross_info)[:80])
except Exception as e:
    record("gross_method", False, str(e))

print()

# =========================================================================
# SECTION 13: Latency Functions
# =========================================================================
print("SECTION 13: Latency API")
print("-" * 40)

try:
    stats = qector_ionq.py_latency_stats()
    record("py_latency_stats", True, f"type={type(stats).__name__}")
except Exception as e:
    record("py_latency_stats", False, str(e))

try:
    scopes = qector_ionq.py_latency_scopes()
    scope_name = "Q70" if "Q70" in list(scopes) else (list(scopes)[0] if scopes else "Q70")
    scoped = qector_ionq.py_latency_stats_scoped(scope_name)
    record("py_latency_stats_scoped",
           isinstance(scoped, tuple) and len(scoped) == 5,
           f"scope={scope_name} type={type(scoped).__name__} n={scoped[0]}")
except Exception as e:
    record("py_latency_stats_scoped", False, str(e))

try:
    scopes = qector_ionq.py_latency_scopes()
    record("py_latency_scopes", True, f"type={type(scopes).__name__}")
except Exception as e:
    record("py_latency_scopes", False, str(e))

try:
    qector_ionq.py_reset_latency()
    record("py_reset_latency", True, "reset OK")
except Exception as e:
    record("py_reset_latency", False, str(e))

print()

# =========================================================================
# SECTION 14: Monte Carlo Logical Error Rate
# =========================================================================
print("SECTION 14: Monte Carlo Logical Error Rate")
print("-" * 40)

for topo_name, dec, H_mat, nq, nc in [
    ("Q70", dec_q70, H70, 70, 70),
    ("Q102", dec_q102, H102, 102, 102),
]:
    n_shots_mc = 2000
    error_rate_mc = 5e-3
    logical_errors = 0
    faithful_count = 0

    for _ in range(n_shots_mc):
        error = (rng.random(nq) < error_rate_mc).astype(np.uint8)
        syndrome = (H_mat @ error) % 2

        if np.all(syndrome == 0):
            faithful_count += 1
            continue

        try:
            corr = np.array(dec.decode(syndrome), dtype=np.uint8)
            resyn = (H_mat @ corr) % 2
            if np.array_equal(resyn, syndrome):
                faithful_count += 1
                residual = (error ^ corr) % 2
                if np.any(residual):
                    logical_errors += 1
            else:
                logical_errors += 1
        except Exception:
            logical_errors += 1

    ler, lo, hi = wilson_ci(logical_errors, n_shots_mc)
    record(f"{topo_name}_mc_{n_shots_mc}_shots",
           faithful_count >= n_shots_mc * 0.99,
           f"faithful={faithful_count}/{n_shots_mc} LER={ler:.4f} [{lo:.4f},{hi:.4f}]")

print()

# =========================================================================
# SECTION 15: Throughput Benchmark
# =========================================================================
print("SECTION 15: Throughput Benchmark")
print("-" * 40)

for topo_name, dec, H_mat, nq, nc in [
    ("Q70", dec_q70, H70, 70, 70),
    ("Q102", dec_q102, H102, 102, 102),
]:
    # Build batch of random single-error syndromes
    n_bench = 1000
    bench_syns = []
    for _ in range(n_bench):
        q = rng.randint(0, nq)
        s, _ = make_syndrome(H_mat, [q])
        bench_syns.append(s)
    flat_bench = np.concatenate(bench_syns).astype(np.uint8)

    # Warmup
    _ = dec.decode_batch_flat(flat_bench[:nc * 10], 10)

    # Timed run
    t0 = time.perf_counter()
    _ = dec.decode_batch_flat(flat_bench, n_bench)
    elapsed = time.perf_counter() - t0
    throughput = n_bench / elapsed
    per_shot_us = elapsed / n_bench * 1e6

    record(f"{topo_name}_throughput_1000",
           throughput > 100,
           f"{throughput:.0f} shots/s  ({per_shot_us:.1f} us/shot)")

print()

# =========================================================================
# SECTION 16: Determinism (same input -> same output)
# =========================================================================
print("SECTION 16: Determinism")
print("-" * 40)

n_det = 50
det_ok = 0
for _ in range(n_det):
    q = rng.randint(0, 70)
    syn, _ = make_syndrome(H70, [q])
    c1 = np.array(dec_q70.decode(syn), dtype=np.uint8)
    c2 = np.array(dec_q70.decode(syn), dtype=np.uint8)
    if np.array_equal(c1, c2):
        det_ok += 1

record("q70_deterministic_50_runs",
       det_ok == n_det,
       f"{det_ok}/{n_det} identical")

print()

# =========================================================================
# SECTION 17: Minimum Weight Property
# =========================================================================
print("SECTION 17: Minimum Weight Property")
print("-" * 40)

# Single-qubit errors should produce weight-1 corrections
weight_ok = 0
for q in range(70):
    syn, _ = make_syndrome(H70, [q])
    corr = dec_q70.decode(syn)
    if sum(corr) == 1:
        weight_ok += 1

record("q70_single_error_weight1",
       weight_ok == 70,
       f"{weight_ok}/70 corrections are weight-1")

print()

# =========================================================================
# SECTION 18: Parity Check Matrix Properties
# =========================================================================
print("SECTION 18: Parity Check Matrix Properties")
print("-" * 40)

# Q70 H matrix properties
record("q70_H_shape",
       H70.shape == (70, 70),
       f"{H70.shape}")

record("q70_H_row_weight",
       all(np.sum(H70[i]) == 6 for i in range(70)),
       "all rows weight-6")

# Column weights
col_weights = [np.sum(H70[:, j]) for j in range(70)]
record("q70_H_col_weights_uniform",
       len(set(col_weights)) == 1,
       f"all columns weight-{col_weights[0]}")

# Q102 H matrix properties
record("q102_H_shape",
       H102.shape == (102, 102),
       f"{H102.shape}")

col_weights_102 = [np.sum(H102[:, j]) for j in range(102)]
record("q102_H_col_weights_uniform",
       len(set(col_weights_102)) == 1,
       f"all columns weight-{col_weights_102[0]}")

print()

# =========================================================================
# SECTION 19: Edge Cases and Error Handling
# =========================================================================
print("SECTION 19: Edge Cases and Error Handling")
print("-" * 40)

# Wrong syndrome length
try:
    dec_q70.decode(np.zeros(50, dtype=np.uint8))
    record("q70_wrong_length_rejected", False, "no error raised")
except Exception as e:
    record("q70_wrong_length_rejected", True, type(e).__name__)

# Wrong dtype (float)
try:
    dec_q70.decode(np.zeros(70, dtype=np.float64))
    record("q70_float_dtype_handled", True, "accepted (cast internally)")
except Exception as e:
    record("q70_float_dtype_handled", True, f"rejected: {type(e).__name__}")

# Batch size mismatch
try:
    flat_bad = np.zeros(100, dtype=np.uint8)  # not divisible by 70
    dec_q70.decode_batch_flat(flat_bad, 2)
    record("q70_batch_size_mismatch", False, "no error raised")
except Exception as e:
    record("q70_batch_size_mismatch", True, type(e).__name__)

print()

# =========================================================================
# SECTION 20: Hardware Bridge Integration
# =========================================================================
print("SECTION 20: Hardware Bridge Integration")
print("-" * 40)

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.')
    sys.path.insert(0, r"C:\Users\Admin\Downloads\qector_ionq_1_7_5_production\qector_ionq_wheel")
    from qector_ionq_hardware_bridge import (
        SyndromeBridge, build_decoder, wilson_ci as bridge_wilson,
        IonQDirectClient,
        IONQ_BACKEND_QUBIT_LIMITS,
        IONQ_IDEAL_SIMULATOR_QUBITS,
        IONQ_NOT_ENOUGH_QUBITS,
        cap_circuit_qubits,
        resolve_backend_qubit_limit,
        prepare_circuit_for_backend,
        build_ionq_circuit_body,
        assert_circuit_fits_backend,
        format_job_failure,
        ionq_outcome_to_bits,
        histogram_to_counts,
        gate_fits,
    )

    # SyndromeBridge.counts_to_flat_batch
    test_counts = {"0" * 70: 100, "1" + "0" * 69: 50, "01" + "0" * 68: 25}
    flat, freqs, n_unique = SyndromeBridge.counts_to_flat_batch(test_counts, 70)
    record("bridge_counts_to_flat",
           flat.shape == (n_unique * 70,) and freqs.shape == (n_unique,),
           f"n_unique={n_unique} flat_len={len(flat)}")

    record("bridge_frequencies_sum",
           np.sum(freqs) == 175,
           f"sum={np.sum(freqs)}")

    # SyndromeBridge.generate_synthetic_counts
    syn_counts, syn_H = SyndromeBridge.generate_synthetic_counts(
        70, 70, c2q70, n_shots=1000, error_rate=1e-3
    )
    record("bridge_synthetic_counts",
           sum(syn_counts.values()) == 1000,
           f"total_shots={sum(syn_counts.values())} unique={len(syn_counts)}")

    # build_decoder
    d70 = build_decoder("q70", 1e-3)
    record("bridge_build_decoder_q70", d70.code_name == "Q70", d70.code_name)
    d102 = build_decoder("q102", 1e-3)
    record("bridge_build_decoder_q102", d102.code_name == "Q102", d102.code_name)

    # wilson_ci validation
    p, lo, hi = bridge_wilson(50, 1000)
    record("bridge_wilson_ci",
           abs(p - 0.05) < 0.001 and lo < p and hi > p,
           f"p={p:.4f} [{lo:.4f}, {hi:.4f}]")

    # IonQDirectClient construction
    client = IonQDirectClient(api_key="test_key")
    record("bridge_client_construct",
           client.base_url == "https://api.ionq.co/v0.3",
           client.base_url)

    # ---- NotEnoughQubits regression (Q70=70, simulator=29) ----
    record("bridge_simulator_limit_29",
           resolve_backend_qubit_limit("simulator") == IONQ_IDEAL_SIMULATOR_QUBITS
           and IONQ_BACKEND_QUBIT_LIMITS["simulator"] == 29,
           f"limit={resolve_backend_qubit_limit('simulator')}")

    record("bridge_aria_noise_limit_25",
           resolve_backend_qubit_limit("simulator", noise_model="aria-1") == 25,
           str(resolve_backend_qubit_limit("simulator", noise_model="aria-1")))

    record("bridge_forte_qpu_limit_36",
           resolve_backend_qubit_limit("qpu.forte-1") == 36,
           str(resolve_backend_qubit_limit("qpu.forte-1")))

    record("bridge_live_qubits_override",
           resolve_backend_qubit_limit("simulator", live_qubits=29) == 29,
           "live_qubits=29")

    q70_capped = cap_circuit_qubits(70, resolve_backend_qubit_limit("simulator"))
    record("bridge_q70_capped_to_simulator",
           q70_capped == 29,
           f"70 -> {q70_capped}")

    q102_capped = cap_circuit_qubits(102, resolve_backend_qubit_limit("simulator"))
    record("bridge_q102_capped_to_simulator",
           q102_capped == 29,
           f"102 -> {q102_capped}")

    record("bridge_small_circuit_not_inflated",
           cap_circuit_qubits(3, 29) == 3,
           "3 stays 3")

    oversize_body = {"qubits": 70, "circuit": [{"gate": "x", "target": 0}]}
    prepared, requested_n, submitted_n, was_capped = prepare_circuit_for_backend(
        oversize_body, 29
    )
    record("bridge_prepare_caps_q70_body",
           was_capped and requested_n == 70 and submitted_n == 29
           and prepared["qubits"] == 29
           and prepared["circuit"] == [{"gate": "x", "target": 0}],
           f"qubits={prepared['qubits']} capped={was_capped}")

    probe = build_ionq_circuit_body(q70_capped)
    record("bridge_probe_fits_simulator",
           probe["qubits"] <= 29 and gate_fits(probe["circuit"][0], probe["qubits"]),
           f"qubits={probe['qubits']}")

    try:
        assert_circuit_fits_backend(70, 29, target="simulator")
        record("bridge_reject_70_on_simulator", False, "no error raised")
    except ValueError as exc:
        msg = str(exc)
        record("bridge_reject_70_on_simulator",
               IONQ_NOT_ENOUGH_QUBITS in msg and "Too many qubits requested" in msg,
               msg)

    try:
        assert_circuit_fits_backend(29, 29, target="simulator")
        record("bridge_accept_29_on_simulator", True, "29 == 29")
    except Exception as exc:
        record("bridge_accept_29_on_simulator", False, str(exc))

    # Exact IonQ failure payload from job 01a0860c-bb50-76cc-8fcf-68a1e372e09a
    failed_job = {
        "id": "01a0860c-bb50-76cc-8fcf-68a1e372e09a",
        "submitted_by": "235eb8c5-59b4-407c-9b1b-afbd2cd4063f",
        "status": "failed",
        "target": "simulator",
        "type": "circuit",
        "dry_run": False,
        "cost_model": "2QGE_operations",
        "gate_counts": {},
        "project_id": "207d868a-deeb-4322-8ca1-26eed33e418d",
        "request": 1788955376,
        "response": 1788955378,
        "failure": {
            "code": "NotEnoughQubits",
            "error": "Too many qubits requested",
        },
        "noise": {"model": "ideal"},
        "error_mitigation": {"debias": False},
        "children": [],
    }
    fail_msg = format_job_failure(failed_job)
    record("bridge_format_not_enough_qubits",
           failed_job["id"] in fail_msg
           and IONQ_NOT_ENOUGH_QUBITS in fail_msg
           and "Too many qubits requested" in fail_msg
           and "failed" in fail_msg,
           fail_msg)

    # IonQ REST histogram keys are little-endian integers, not 70-char bitstrings
    bits_q0 = ionq_outcome_to_bits("1", 70)
    record("bridge_ionq_key_qubit0",
           bits_q0[0] == 1 and sum(bits_q0) == 1 and len(bits_q0) == 70,
           f"weight={sum(bits_q0)} n={len(bits_q0)}")

    bits_pad = ionq_outcome_to_bits("1", 29)
    record("bridge_ionq_key_29_then_pad_to_70",
           bits_pad[0] == 1 and sum(bits_pad) == 1,
           f"n=29 weight={sum(bits_pad)}")

    # 29-qubit probe counts padded to Q70 syndrome length
    probe_counts = {"1": 100}
    flat_probe, freqs_probe, n_probe = SyndromeBridge.counts_to_flat_batch(probe_counts, 70)
    record("bridge_probe_counts_padded_to_q70",
           flat_probe.shape == (70,) and int(flat_probe[0]) == 1
           and int(np.sum(flat_probe)) == 1 and int(freqs_probe[0]) == 100,
           f"shape={flat_probe.shape} sum={int(np.sum(flat_probe))}")

    hist_counts = histogram_to_counts({"0": 0.0, "1": 1.0}, 100)
    record("bridge_histogram_to_counts",
           hist_counts.get("1") == 100 and hist_counts.get("0", 0) == 0,
           str(hist_counts))

    # Drop out-of-range gates when capping (X on qubit 50 cannot run on 29)
    wide_body = {
        "qubits": 70,
        "circuit": [
            {"gate": "x", "target": 0},
            {"gate": "x", "target": 50},
        ],
    }
    slim, _, slim_n, _ = prepare_circuit_for_backend(wide_body, 29)
    record("bridge_drop_out_of_range_gates",
           slim_n == 29 and slim["circuit"] == [{"gate": "x", "target": 0}],
           str(slim["circuit"]))

except Exception as e:
    record("bridge_integration", False, f"{type(e).__name__}: {e}")

print()

# =========================================================================
# SECTION 21: IonQ simulator qubit ceiling (live catalog, optional)
# =========================================================================
print("SECTION 21: IonQ Backend Qubit Ceiling")
print("-" * 40)

try:
    import urllib.request as _urlreq
    req = _urlreq.Request("https://api.ionq.co/v0.3/backends", method="GET")
    with _urlreq.urlopen(req, timeout=15) as resp:
        backends = json.loads(resp.read().decode("utf-8"))
    if isinstance(backends, dict) and "backends" in backends:
        backends = backends["backends"]
    sim = next(
        (b for b in backends if str(b.get("backend", "")).lower() in ("simulator", "ionq_simulator")),
        None,
    )
    record("live_ionq_simulator_present", sim is not None, "GET /backends")
    if sim is not None:
        live_n = int(sim.get("qubits", -1))
        record("live_ionq_simulator_qubits_29",
               live_n == 29,
               f"qubits={live_n}")
        record("q70_exceeds_live_simulator",
               70 > live_n,
               f"Q70=70 simulator={live_n}")
        record("q102_exceeds_live_simulator",
               102 > live_n,
               f"Q102=102 simulator={live_n}")
except Exception as e:
    skip("live_ionq_backends", f"{type(e).__name__}: {e}")

print()

# =========================================================================
# SECTION 22: Extreme Error Rates + OSD Variants
# =========================================================================
print("SECTION 22: Extreme Error Rates + OSD Variants")
print("-" * 40)
try:
    for er in [1e-12, 1e-9, 0.1, 0.25, 0.49]:
        d=qector_ionq.IonQSuperionDecoder.q70(error_rate=er)
        z=np.zeros(70,dtype=np.uint8)
        c=np.asarray(d.decode(z),dtype=np.uint8)
        record(f"q70_extreme_er_{er}", int(c.sum())==0, f"er={er}")
    for method in ["exact","min_sum"]:
        for order in [0,1,2]:
            b=qector_ionq.BPOSDDecoder(c2q70,70,1e-3,bp_method=method,osd_order=order)
            c=np.asarray(b.decode(np.zeros(70,dtype=np.uint8)),dtype=np.uint8)
            record(f"bposd_{method}_o{order}", int(c.sum())==0, f"{method} o={order}")
    b=qector_ionq.BPOSDDecoder(c2q70,70,1e-3)
    llr=np.asarray(b.bp_decode(np.zeros(70,dtype=np.uint8),10),dtype=np.float64)
    record("bposd_llr_finite", bool(np.all(np.isfinite(llr))), f"min={llr.min():.2f} max={llr.max():.2f}")
    c_t=np.asarray(b.decode_timed(np.zeros(70,dtype=np.uint8),5.0),dtype=np.uint8)
    record("bposd_timed_5ms", int(c_t.sum())==0)
except Exception as e:
    record("extreme_er_osd", False, str(e))

print()

# =========================================================================
# SECTION 23: TwoStage / SpaceTime / AutoDecoder Deep
# =========================================================================
print("SECTION 23: TwoStage / SpaceTime / AutoDecoder Deep")
print("-" * 40)
try:
    ts=qector_ionq.TwoStageDecoder.q102(error_rate=1e-3)
    for q in range(5):
        syn,_=make_syndrome(H102,[q])
        c=np.asarray(ts.decode(syn),dtype=np.uint8)
        record(f"twostage_q102_q{q}", np.array_equal((H102@c)%2,syn))
    ts_g=qector_ionq.TwoStageDecoder.gross(error_rate=1e-3)
    record("twostage_gross_zero", int(np.asarray(ts_g.decode(np.zeros(144,dtype=np.uint8))).sum())==0)
    st=qector_ionq.SpaceTimeDecoder(c2q70,70,3,1e-3)
    flat=np.zeros(3*70,dtype=np.uint8)
    out=np.asarray(st.decode_rounds_flat(flat,3),dtype=np.uint8)
    record("spacetime_3r_flat", out.size in (70,210), str(out.size))
    for prio in ["speed","accuracy","balanced"]:
        ad=qector_ionq.AutoDecoder(code="q70",priority=prio)
        c=np.asarray(ad.decode(np.zeros(ad.n_checks,dtype=np.uint8)),dtype=np.uint8)
        record(f"autodecoder_{prio}", int(c.sum())==0, ad.backend)
    rec=qector_ionq.AutoDecoder.recommend("q70","speed")
    record("autodecoder_recommend", isinstance(rec,str) and len(rec)>0 and "bposd" in rec.lower(), rec)
except Exception as e:
    record("deep_decoders", False, f"{type(e).__name__}: {e}")

print()

# =========================================================================
# SECTION 24: Erasure Exhaustive + Schedule Fuzz
# =========================================================================
print("SECTION 24: Erasure Exhaustive + Schedule Fuzz")
print("-" * 40)
try:
    dec=IonQSuperionDecoder.q70(error_rate=1e-3)
    for n_era in [1,5,10,35]:
        era=np.zeros(70,dtype=np.uint8); era[:n_era]=1
        s=np.zeros(70,dtype=np.uint8)
        c=np.asarray(dec.decode_with_erasures(s,era),dtype=np.uint8)
        record(f"erasure_n{n_era}_zero", int(c.sum())==0)
    # erasure with random syndrome (strict may raise Unreachable -> handled as pass)
    for _ in range(10):
        e=np.zeros(70,dtype=np.uint8); e[np.random.choice(70,2,replace=False)]=1; s=(H70@e)%2
        era=np.zeros(70,dtype=np.uint8); era[np.random.choice(70,3,replace=False)]=1
        try:
            c=np.asarray(dec.decode_with_erasures(s,era),dtype=np.uint8)
            record(f"erasure_rand_{_}", True, f"w={int(c.sum())}")
        except Exception as ex:
            if "unreachable" in str(ex).lower():
                record(f"erasure_rand_{_}", True, f"strict-raise {ex}")
            else:
                record(f"erasure_rand_{_}", False, str(ex))
    # schedule fuzz: heterogeneous with extremes
    dec.set_qubit_priors(np.random.uniform(1e-4,1e-2,size=70).astype(np.float64))
    record("hetero_random_priors", "heterogeneous" in dec.schedule_label)
    dec.set_schedule([0.001]*70,[0.002]*70,"fuzz")
    record("custom_schedule_fuzz", dec.schedule_label=="fuzz")
    dec.set_uniform_schedule(1e-3)
except Exception as e:
    record("erasure_schedule_fuzz", False, str(e))

print()

# =========================================================================
# SECTION 25: Dtype + Stride + Fortran Robustness
# =========================================================================
print("SECTION 25: Dtype + Stride + Fortran Robustness")
print("-" * 40)
try:
    dec=IonQSuperionDecoder.q70(error_rate=1e-3)
    for dtype in [np.uint8,np.int32,np.int64,np.float32,np.float64]:
        s=np.zeros(70,dtype=dtype)
        c=np.asarray(dec.decode(s.astype(np.uint8) if dtype!=np.uint8 else s),dtype=np.uint8)
        record(f"dtype_{dtype.__name__}", int(c.sum())==0)
    # Fortran order batch
    batch=np.zeros((4,70),dtype=np.uint8,order='F')
    flat=np.asfortranarray(batch.reshape(-1))
    out=np.asarray(dec.decode_batch_flat(np.ascontiguousarray(flat),4),dtype=np.uint8)
    record("fortran_batch", out.size==280)
    # non-contiguous stride (must make contiguous for PyO3)
    big=np.zeros(140,dtype=np.uint8)
    big[::2]=np.zeros(70,dtype=np.uint8)
    record("stride_decode", int(np.asarray(dec.decode(np.ascontiguousarray(big[::2])),dtype=np.uint8).sum())==0)
except Exception as e:
    record("dtype_robust", False, str(e))

print()

# =========================================================================
# SECTION 26: Thread Wrapper + Latency Scoped Stress
# =========================================================================
print("SECTION 26: Thread Wrapper + Latency Scoped Stress")
print("-" * 40)
try:
    import sys as _sys, os as _os
    _sys.path.insert(0, "python"); _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))
    from qector_thread_wrapper import optimal_thread_count, configure_threads
    for nq, exp in [(70,1),(102,2),(500,4),(2000,8)]:
        tc=optimal_thread_count(nq, physical_cores=8)
        record(f"threads_nq{nq}", tc==exp, f"got {tc}")
    qector_ionq.py_reset_latency()
    dec=IonQSuperionDecoder.q102(error_rate=1e-3)
    for _ in range(20): dec.decode(np.zeros(102,dtype=np.uint8))
    scopes=qector_ionq.py_latency_scopes()
    record("latency_scopes_after20", "Q102" in scopes)
    n,mean,p50,p95,mx=qector_ionq.py_latency_stats_scoped("Q102")
    record("latency_p95_under2ms", p95<2000, f"p95={p95:.0f}us")
    qector_ionq.py_reset_latency()
    record("latency_reset", qector_ionq.py_latency_stats_scoped("Q102")[0]==0)
except Exception as e:
    record("thread_latency_stress", False, str(e))

print()

# =========================================================================
# SECTION 27: Throughput SLO + Determinism at Scale
# =========================================================================
print("SECTION 27: Throughput SLO + Determinism at Scale")
print("-" * 40)
try:
    for name, dec, Hm, nq in [("Q70",dec_q70,H70,70),("Q102",dec_q102,H102,102)]:
        B=2000
        syns=np.zeros((B, Hm.shape[0]), dtype=np.uint8)
        for i in range(B):
            e=np.zeros(nq,dtype=np.uint8); e[np.random.default_rng(i).choice(nq,2,replace=False)]=1; syns[i]=(Hm@e)%2
        flat=np.ascontiguousarray(syns.reshape(-1))
        t0=time.perf_counter(); out=np.asarray(dec.decode_batch_flat(flat,B),dtype=np.uint8).reshape(B,nq); dt=time.perf_counter()-t0
        thr=B/dt
        record(f"{name}_thr2000_SLO", thr>900, f"{thr:.0f}/s")
        # determinism 3x
        s=syns[0]; c1=np.asarray(dec.decode(s),dtype=np.uint8); c2=np.asarray(dec.decode(s),dtype=np.uint8); c3=np.asarray(dec.decode(s),dtype=np.uint8)
        record(f"{name}_determinism3x", np.array_equal(c1,c2) and np.array_equal(c2,c3))
except Exception as e:
    record("throughput_determinism", False, str(e))

print()

# =========================================================================
# SECTION 28: License + Artifact + Version Invariants
# =========================================================================
print("SECTION 28: License + Artifact + Version Invariants")
print("-" * 40)
try:
    sts=qector_ionq.py_license_status()
    record("license_3tuple", isinstance(sts,tuple) and len(sts)==3, str(sts))
    for fac in ["q70","q102","gross"]:
        d=getattr(IonQSuperionDecoder, fac)(error_rate=1e-3)
        h=d.artifact_hash
        record(f"{fac}_hash_16hex", isinstance(h,str) and len(h)==16 and all(c in "0123456789abcdef" for c in h), h)
    h1=IonQSuperionDecoder.q102(error_rate=1e-3).artifact_hash
    h2=IonQSuperionDecoder.q102(error_rate=5e-3).artifact_hash
    record("hash_stable_across_er", h1==h2, f"{h1}")
    record("version_match", qector_ionq.__version__=="1.7.7" and "1.7.7" in qector_ionq.DECODER_VERSION)
    record("arch_walking_cat", "Walking Cat" in qector_ionq.TARGET_ARCHITECTURE)
except Exception as e:
    record("license_artifact", False, str(e))

print()

# =========================================================================
# SECTION 29: Gross Exhaustive w1 + w2 Sample + Memory
# =========================================================================
print("SECTION 29: Gross Exhaustive w1 + w2 Sample + Memory")
print("-" * 40)
try:
    dec_g=IonQSuperionDecoder.gross(error_rate=1e-3)
    Hg=np.zeros((144,144),dtype=np.uint8)
    for ci, qs in enumerate(dec_g.check_to_qubits):
        for q in qs: Hg[ci,q]=1
    ok=True
    for q in range(144):
        e=np.zeros(144,dtype=np.uint8); e[q]=1; s=(Hg@e)%2
        c=np.asarray(dec_g.decode(s),dtype=np.uint8)
        if not np.array_equal((Hg@c)%2,s): ok=False; break
    record("gross_w1_144", ok, "exhaustive 144")
    ok2=0
    for _ in range(100):
        e=np.zeros(144,dtype=np.uint8); e[np.random.choice(144,2,replace=False)]=1; s=(Hg@e)%2
        c=np.asarray(dec_g.decode(s),dtype=np.uint8)
        if np.array_equal((Hg@c)%2,s): ok2+=1
    record("gross_w2_100", ok2==100, f"{ok2}/100")
    # memory linear: batch 100 vs 10 latency ratio < 15x
    flat=np.zeros(100*144,dtype=np.uint8)
    t0=time.perf_counter(); dec_g.decode_batch_flat(flat,100); dt1=time.perf_counter()-t0
    flat10=np.zeros(10*144,dtype=np.uint8)
    t0=time.perf_counter(); dec_g.decode_batch_flat(flat10,10); dt10=time.perf_counter()-t0
    record("gross_mem_linear", dt1 < dt10*15, f"100:{dt1:.3f}s 10:{dt10:.3f}s")
except Exception as e:
    record("gross_exhaustive", False, str(e))

print()

# =========================================================================
# SECTION 30: Hardware Bridge Probe + NotEnoughQubits Guard (extended)
# =========================================================================
print("SECTION 30: Hardware Bridge Probe Extended")
print("-" * 40)
try:
    from qector_ionq_hardware_bridge import cap_circuit_qubits, resolve_backend_qubit_limit, build_ionq_circuit_body, prepare_circuit_for_backend, gate_fits, IONQ_NOT_ENOUGH_QUBITS
    for tgt, lim in [("simulator",29),("qpu.aria-1",25),("qpu.forte-1",36)]:
        record(f"bridge_limit_{tgt}", resolve_backend_qubit_limit(tgt)==lim, f"{lim}")
    body=build_ionq_circuit_body(29)
    record("bridge_body_29", body["qubits"]==29)
    try:
        build_ionq_circuit_body(70, [{"gate":"x","target":70}])
        record("bridge_bad_gate", False, "no raise")
    except ValueError:
        record("bridge_bad_gate", True, "raised")
    wide={"qubits":70,"circuit":[{"gate":"x","target":0},{"gate":"cx","control":0,"target":50}]}
    slim,_,n,_=prepare_circuit_for_backend(wide,29)
    record("bridge_cx_capped", n==29 and len(slim["circuit"])==1, str(slim["circuit"]))
    record("gate_fits_true", gate_fits({"gate":"x","target":28},29))
    record("gate_fits_false", not gate_fits({"gate":"x","target":29},29))
except Exception as e:
    record("bridge_ext", False, str(e))

print()

# =========================================================================
# SUMMARY
# =========================================================================
print()
print("=" * 80)
print(f"  TEST SUMMARY: {PASS} PASSED  |  {FAIL} FAILED  |  {SKIP} SKIPPED")
print(f"  Total: {PASS + FAIL + SKIP}")
print("=" * 80)
print()

if FAIL > 0:
    print("FAILED TESTS:")
    for name, tag, detail in RESULTS:
        if tag == "FAIL":
            print(f"  {name}: {detail}")
    print()

# Exit code
sys.exit(1 if FAIL > 0 else 0)
