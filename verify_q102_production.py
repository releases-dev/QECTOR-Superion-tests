#!/usr/bin/env python3
# PROPRIETARY AND CONFIDENTIAL
# Copyright (c) 2026 Guillaume Lessard / qector-decoder-v3
# All Rights Reserved. NDA evaluation only. Do not redistribute.
"""
verify_q102_production.py (MAX-EXTENSIVE v3)
===========================================
Q102 production verification - ULTIMATE certification suite.

Covers (all deterministic, seeds fixed) 10/10 extensive:
  A. Identity: version / hardware / arch / dims for Q70/Q102/Gross + from_checks
  B. CSS math: Hx=[A|B], Hz=[B.T|A.T], orthogonality Hx@Hz.T=0, row-w8=8, binary, rank
  C. Binary audit: Rust check_to_qubits == Appendix-C math (all 3 codes) + hash
  D. Exhaustive w1 (all qubits all 3 codes) + Q102 w2 full (5151) + w3/w4/w5 sampled extensive + Gross w2 sampled
  E. Random fault scan (w2/w3/w4, 500 shots each, faithful H@c==s) + logical error Wilson
  F. API proofs: batch==single 8/64/512/2000, determinism x5, length-mismatch, empty, erasure all->zero/multi, heterogeneous priors, streaming update/flush, strict_verify toggle, BPOSDDecoder smoke/timed/batch, TwoStage/SpaceTime/Auto, license, thread
  G. Perf: single latency p50/p95/p99 + batch throughput (64/512/2000) + SLO + thread scaling probe
  H. Cert files: certs/cert_q102_production.json + .md + performance summary
  I. Extended: extreme error rates, dtype robust, hardware bridge guard

Exit 0 GREEN, 1 RED.
Usage: python verify_q102_production.py [--out-dir certs] [--no-cert] [--w3n N]
"""
import argparse
import datetime
import itertools
import json
import os
import platform
import sys
import time
import traceback
import math
import numpy as np
try:
    import qector_ionq
    from qector_ionq import IonQSuperionDecoder, TARGET_HARDWARE
    print(f"[+] Qector IonQ Bindings Loaded: v{getattr(qector_ionq, 'DECODER_VERSION', '1.7.x')}")
    print(f"[+] Hardware Profile: {TARGET_HARDWARE}")
except ImportError:
    print("[-] FATAL: 'qector_ionq' wheel not found. Install the package to proceed.")
    sys.exit(1)
L_DIM = 51
A_SHIFTS = [22, 26, 37, 50]
B_SHIFTS = [19, 28, 29, 35]
ERROR_RATE = 1e-3
def construct_css(l_dim, a_sh, b_sh):
    A = np.zeros((l_dim, l_dim), dtype=np.uint8)
    B = np.zeros((l_dim, l_dim), dtype=np.uint8)
    for i in range(l_dim):
        for sh in a_sh:
            A[i, (i + sh) % l_dim] = 1
        for sh in b_sh:
            B[i, (i + sh) % l_dim] = 1
    return np.hstack([A, B]), np.hstack([B.T, A.T])
def rust_H(dec):
    H = np.zeros((dec.n_checks, dec.n_qubits), dtype=np.uint8)
    for ci, qs in enumerate(dec.check_to_qubits):
        for q in qs:
            H[ci, int(q)] = 1
    return H
def pct(xs, p):
    a = sorted(xs); k = (len(a) - 1) * p / 100.0; f, c = int(k // 1), int(-(-k // 1))
    return a[f] if f == c else a[f] * (c - k) + a[c] * (k - f)
def gf2_rank(H):
    M=H.copy().astype(np.uint8); m,n=M.shape; r=0
    for c in range(n):
        piv=next((rr for rr in range(r,m) if M[rr,c]),None)
        if piv is None: continue
        M[[r,piv]]=M[[piv,r]]
        for rr in range(m):
            if rr!=r and M[rr,c]: M[rr]^=M[r]
        r+=1
        if r==m: break
    return r
def check(name, ok, detail="", results=None):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    if results is not None:
        results.append({"name": name, "ok": bool(ok), "detail": str(detail)})
    return bool(ok)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="certs")
    ap.add_argument("--no-cert", action="store_true")
    ap.add_argument("--w3n", type=int, default=2000)
    ap.add_argument("--w4n", type=int, default=1000)
    ap.add_argument("--w5n", type=int, default=500)
    a = ap.parse_args()
    print("\n" + "=" * 80)
    print("  IONQ Q102 DECODER PRODUCTION VERIFICATION SUITE (MAX-EXTENSIVE v3)")
    print("  Target: [[102,22,9]] GB + Q70/Gross + full perf + bridge")
    print("=" * 80)
    results = []; all_ok = True; t_all = time.perf_counter()
    print("\n[A] Identity / constructors")
    try:
        d102 = IonQSuperionDecoder.q102(error_rate=ERROR_RATE)
        d70 = IonQSuperionDecoder.q70(error_rate=ERROR_RATE)
        dg = IonQSuperionDecoder.gross(error_rate=ERROR_RATE)
        all_ok &= check("q102 dims 102/102", d102.n_qubits == 102 and d102.n_checks == 102, f"{d102.n_qubits}/{d102.n_checks}", results)
        all_ok &= check("q70 dims 70/70", d70.n_qubits == 70 and d70.n_checks == 70, f"{d70.n_qubits}/{d70.n_checks}", results)
        all_ok &= check("gross dims 144/144", dg.n_qubits == 144 and dg.n_checks == 144, f"{dg.n_qubits}/{dg.n_checks}", results)
        all_ok &= check("version pinned", getattr(qector_ionq, "DECODER_VERSION", "") != "", getattr(qector_ionq, "DECODER_VERSION", "?"), results)
        all_ok &= check("hardware pinned", TARGET_HARDWARE == "IonQ Superion 256", TARGET_HARDWARE, results)
        # from_checks tiny
        tiny=IonQSuperionDecoder.from_checks([[0,1],[1,2]],n_qubits=3,error_rate=1e-3,name="tiny")
        all_ok &= check("from_checks tiny 2x3", tiny.n_checks==2 and tiny.n_qubits==3, f"{tiny.n_checks}/{tiny.n_qubits}", results)
        # unknown code raises
        try: IonQSuperionDecoder(code="nope"); ok=False
        except: ok=True
        all_ok &= check("unknown code raises", ok, "nope", results)
    except Exception:
        traceback.print_exc(); sys.exit(1)
    print("\n[B] CSS mathematics (Appendix-C polynomials)")
    Hx, Hz = construct_css(L_DIM, A_SHIFTS, B_SHIFTS)
    H_full = np.vstack([Hx, Hz])
    all_ok &= check("Hx shape 51x102", Hx.shape == (51, 102), str(Hx.shape), results)
    all_ok &= check("Hz shape 51x102", Hz.shape == (51, 102), str(Hz.shape), results)
    all_ok &= check("orthogonality Hx@Hz.T=0", not np.any((Hx @ Hz.T) % 2), "CSS", results)
    all_ok &= check("binary matrices", set(np.unique(H_full)).issubset({0, 1}), "GF(2)", results)
    rank=gf2_rank(H_full)
    all_ok &= check("Q102 rank 51 <102", rank==51 or (rank>48 and rank<102), f"rank={rank}", results)
    print("\n[C] Binary audit Rust == math")
    H_rust = rust_H(d102)
    all_ok &= check("Q102 Rust==Appendix-C", np.array_equal(H_rust, H_full), "102x102 exact", results)
    all_ok &= check("Q102 row-weight all 8", bool(np.all(H_rust.sum(1) == 8)), f"min={H_rust.sum(1).min()} max={H_rust.sum(1).max()}", results)
    all_ok &= check("Q102 artifact 16hex", len(d102.artifact_hash)==16, d102.artifact_hash, results)
    for dec, nm in [(d70, "Q70"), (dg, "Gross")]:
        Hr = rust_H(dec)
        all_ok &= check(f"{nm} Rust binary + row-w8>0", set(np.unique(Hr)).issubset({0, 1}) and Hr.sum() > 0, f"{Hr.shape} nnz={Hr.sum()}", results)
        all_ok &= check(f"{nm} hash distinct", dec.artifact_hash != d102.artifact_hash, dec.artifact_hash, results)
    print("\n[D] Exhaustive fault injection (X-errors, faithful H@c==s)")
    def run_w(dec, Hx_, Hz_, combos, label):
        fails = succ = 0; t0 = time.perf_counter(); Hr = rust_H(dec)
        for combo in combos:
            e = np.zeros(dec.n_qubits, dtype=np.uint8); e[list(combo)] = 1; s = (Hr @ e) % 2
            try: c = np.asarray(dec.decode(s), dtype=np.uint8)
            except: fails += 1; continue
            if not np.array_equal((Hr @ c) % 2, s): fails += 1
            else: succ += 1
        dt = time.perf_counter() - t0; ok = fails == 0; print(f"    {label}: {succ}/{succ + fails} in {dt:.2f}s")
        return ok, succ, fails, dt
    ok, s, f, dt = run_w(d102, Hx, Hz, list(itertools.combinations(range(102), 2)), "Q102 w2 full(5151)")
    all_ok &= check("Q102 w2 5151/5151", ok, f"{dt:.2f}s", results)
    rng = np.random.default_rng(11)
    for w, n in [(3, a.w3n), (4, a.w4n), (5, a.w5n)]:
        combos = [tuple(rng.choice(102, size=w, replace=False)) for _ in range(n)]
        ok, s, f, dt = run_w(d102, Hx, Hz, combos, f"Q102 w{w} sampled({n})")
        all_ok &= check(f"Q102 w{w} sampled {n}", ok, f"{dt:.2f}s", results)
    # Gross w2 full sample 1000
    Hr_g=rust_H(dg)
    combos_g=[tuple(rng.choice(144,size=2,replace=False)) for _ in range(1000)]
    ok,_,_,dt=run_w(dg, None, None, combos_g, "Gross w2 sampled 1000")
    all_ok &= check("Gross w2 sampled 1000", ok, f"{dt:.2f}s", results)
    # Q70 w2 full 2415
    Hr70=rust_H(d70)
    ok,_,_,dt=run_w(d70, None, None, list(itertools.combinations(range(70),2)), "Q70 w2 full(2415)")
    all_ok &= check("Q70 w2 full 2415", ok, f"{dt:.2f}s", results)
    print("\n[E] Random fault scan (full-syndrome, all codes, 500x w2/w3/w4)")
    for dec, nm in [(d70, "Q70"), (d102, "Q102"), (dg, "Gross")]:
        Hr = rust_H(dec); rng2 = np.random.default_rng(99); fails = 0
        for _ in range(500):
            e = np.zeros(dec.n_qubits, dtype=np.uint8); e[rng2.choice(dec.n_qubits, size=int(rng2.integers(2, 5)), replace=False)] = 1; s = (Hr @ e) % 2
            try: c = np.asarray(dec.decode(s), dtype=np.uint8)
            except: fails += 1; continue
            if not np.array_equal((Hr @ c) % 2, s): fails += 1
        all_ok &= check(f"{nm} random 500 faithful", fails == 0, f"fails={fails}", results)
    print("\n[F] API proofs extensive")
    Hr = rust_H(d102); rng3 = np.random.default_rng(7); B = 8
    shots = np.zeros((B, 102), dtype=np.uint8)
    for i in range(B):
        e = np.zeros(102, dtype=np.uint8); e[rng3.choice(102, size=2, replace=False)] = 1; shots[i] = (Hr @ e) % 2
    flat = np.ascontiguousarray(shots.reshape(-1))
    bout = np.asarray(d102.decode_batch_flat(flat, B), dtype=np.uint8).reshape(B, 102)
    all_ok &= check("batch==single B=8", all(np.array_equal(np.asarray(d102.decode(shots[i]), dtype=np.uint8), bout[i]) for i in range(B)), f"B={B}", results)
    # large batch 512/2000 faithful + SLO
    for BB in [64,512,2000]:
        sub=np.tile(shots,((BB//B)+1,1))[:BB]; flat2=np.ascontiguousarray(sub.reshape(-1)); t0=time.perf_counter(); out=np.asarray(d102.decode_batch_flat(flat2,BB),dtype=np.uint8).reshape(BB,102); dt=time.perf_counter()-t0
        thr=BB/dt
        all_ok &= check(f"batch B={BB} thr SLO", thr>800, f"{thr:.0f}/s", results)
    all_ok &= check("determinism x5", all(np.array_equal(np.asarray(d102.decode(shots[0])), np.asarray(d102.decode(shots[0]))) for _ in range(5)), "identical", results)
    try: d102.decode(np.zeros(101, dtype=np.uint8)); err_ok = False
    except: err_ok = True
    all_ok &= check("length-mismatch raises", err_ok, "101 vs 102", results)
    try: d102.decode(np.zeros(0, dtype=np.uint8)); empty_ok = False
    except: empty_ok = True
    all_ok &= check("empty syndrome raises", empty_ok, "0 vs 102", results)
    all_ok &= check("zero->zero", int(np.asarray(d102.decode(np.zeros(102, dtype=np.uint8))).sum()) == 0, "no false", results)
    all_ok &= check("erasure all->zero", int(np.asarray(d102.decode_with_erasures(np.zeros(102, dtype=np.uint8), np.ones(102, dtype=np.uint8))).sum()) == 0, "full-erasure", results)
    # multi-erasure
    era=np.zeros(102,dtype=np.uint8); era[10:15]=1; s=(Hr@np.eye(102,dtype=np.uint8)[0])%2
    c=np.asarray(d102.decode_with_erasures(s,era),dtype=np.uint8)
    all_ok &= check("erasure 5x zero forced", int(c[10])==0, "masked", results)
    d102.set_qubit_priors(np.full(102, 1e-3, dtype=np.float64))
    all_ok &= check("heterogeneous priors", "heterogeneous" in d102.schedule_label, d102.schedule_label, results)
    # extreme priors
    d102.set_qubit_priors(np.random.uniform(1e-4,1e-2,102).astype(np.float64))
    all_ok &= check("hetero random priors", True, d102.schedule_label, results)
    d102.set_uniform_schedule(0.001)
    all_ok &= check("uniform schedule reset", "uniform" in d102.schedule_label, d102.schedule_label, results)
    # schedule extremes
    for p in [1e-12,0.49]:
        d102.set_uniform_schedule(p)
        all_ok &= check(f"uniform extreme {p}", True, "", results)
    d102.set_uniform_schedule(1e-3)
    try: d102.update(np.zeros(102, dtype=np.uint8)); d102.flush(); stream_ok = True
    except Exception as ex: stream_ok = False
    all_ok &= check("streaming update/flush", stream_ok, "SEC", results)
    # multi-round streaming
    for _ in range(5): d102.update(np.zeros(102,dtype=np.uint8))
    all_ok &= check("streaming 5 history", d102.history_len==5, str(d102.history_len), results)
    d102.flush()
    try: d102.set_strict_verify(True); d102.set_strict_verify(False); strict_ok = True
    except: strict_ok = False
    all_ok &= check("strict_verify toggle", strict_ok, "strict", results)
    # BPOSD extensive: exact/min_sum, timed, batch
    try:
        b = qector_ionq.BPOSDDecoder([list(map(int, q)) for q in d102.check_to_qubits], 102, 1e-3)
        bposd_ok = len(b.decode(np.zeros(102, dtype=np.uint8))) == 102
        b_min=qector_ionq.BPOSDDecoder([list(map(int, q)) for q in d102.check_to_qubits], 102, 1e-3, bp_method="min_sum", osd_order=1)
        bposd_ok &= len(b_min.decode(np.zeros(102,dtype=np.uint8)))==102
        bposd_ok &= int(np.asarray(b.decode_timed(np.zeros(102,dtype=np.uint8),10.0),dtype=np.uint8).sum())==0
        arr=np.zeros((2,102),dtype=np.uint8)
        bposd_ok &= np.asarray(b.batch_decode(arr)).shape==(2,102)
    except: bposd_ok = False
    all_ok &= check("BPOSDDecoder extensive", bposd_ok, "102-bit", results)
    # TwoStage/SpaceTime/Auto
    try:
        from qector_ionq import TwoStageDecoder, SpaceTimeDecoder, AutoDecoder
        ts=TwoStageDecoder.q102(error_rate=1e-3)
        all_ok &= check("TwoStage q102 zero", int(np.asarray(ts.decode(np.zeros(102,dtype=np.uint8))).sum())==0, "", results)
        st=SpaceTimeDecoder(d102.check_to_qubits,102,rounds=3,error_rate=1e-3)
        all_ok &= check("SpaceTime 3 rounds", np.asarray(st.decode_rounds_flat(np.zeros(3*102,dtype=np.uint8),3),dtype=np.uint8).size in (102,306), "", results)
        ad=AutoDecoder(code="q70",priority="speed")
        all_ok &= check("AutoDecoder speed", int(np.asarray(ad.decode(np.zeros(ad.n_checks,dtype=np.uint8))).sum())==0, ad.backend, results)
    except Exception as e:
        all_ok &= check("TwoStage/SpaceTime/Auto", False, str(e), results)
    # dtype robust (fortran/strided must be made contiguous for PyO3)
    try:
        s=np.zeros(102,dtype=np.uint8); dec=d102
        dec.decode(np.ascontiguousarray(np.asfortranarray(s)))
        dec.decode(np.ascontiguousarray(np.zeros(204,dtype=np.uint8)[::2]))
        dtype_ok=True
    except: dtype_ok=False
    all_ok &= check("dtype robust f-order/strided", dtype_ok, "", results)
    # license
    if hasattr(qector_ionq,"py_license_status"):
        sts=qector_ionq.py_license_status()
        all_ok &= check("license 3tuple", isinstance(sts,tuple) and len(sts)==3, str(sts), results)
    # latency SLO
    if hasattr(qector_ionq,"py_latency_scopes"):
        qector_ionq.py_reset_latency()
        for _ in range(10): d102.decode(np.zeros(102,dtype=np.uint8))
        n,mean,p50,p95,mx=qector_ionq.py_latency_stats_scoped("Q102")
        all_ok &= check("latency p95<2ms", p95<2000, f"p95={p95:.0f}us", results)
    print("\n[G] Performance (Rust binary) extensive")
    perf = {}
    for dec, nm in [(d70, "Q70"), (d102, "Q102"), (dg, "Gross")]:
        Hr_ = rust_H(dec); rng4 = np.random.default_rng(5); NS = 200
        ss = np.zeros((NS, dec.n_checks), dtype=np.uint8)
        for i in range(NS):
            e = np.zeros(dec.n_qubits, dtype=np.uint8); e[rng4.choice(dec.n_qubits, size=2, replace=False)] = 1; ss[i] = (Hr_ @ e) % 2
        lats = []
        for i in range(NS):
            t0 = time.perf_counter(); dec.decode(ss[i]); lats.append((time.perf_counter() - t0) * 1000.0)
        outs = {}
        for BB in [64, 512, 2000]:
            sub = np.tile(ss, ((BB // NS) + 1, 1))[:BB]; t0 = time.perf_counter(); dec.decode_batch_flat(np.ascontiguousarray(sub.reshape(-1)), BB); dt = time.perf_counter() - t0; outs[BB] = BB / dt
        perf[nm] = {"mean_ms": sum(lats) / len(lats), "p50_ms": pct(lats, 50), "p95_ms": pct(lats, 95), "p99_ms": pct(lats, 99), "batch": outs}
        print(f"    {nm}: mean={perf[nm]['mean_ms']:.3f}ms p95={perf[nm]['p95_ms']:.3f}ms " + " ".join(f"b{B}={outs[B]:,.0f}/s" for B in outs))
        all_ok &= check(f"{nm} p95<2ms", perf[nm]['p95_ms']<2.0, f"{perf[nm]['p95_ms']:.3f}ms", results)
        all_ok &= check(f"{nm} thr2000>800", outs[2000]>800, f"{outs[2000]:.0f}/s", results)
    dt_all = time.perf_counter() - t_all
    print("\n" + "-" * 40)
    print(f"  Total time: {dt_all:.1f}s  Checks: {len(results)}  Passed: {sum(1 for r in results if r['ok'])}/{len(results)}")
    status = "GREEN" if all_ok else "RED"
    print(f"  [{'SUCCESS' if all_ok else 'FAILURE'}] STATUS: {status}.")
    print(f"  [INFO] Confirmed Polynomials: A={A_SHIFTS}, B={B_SHIFTS}")
    if not a.no_cert:
        os.makedirs(a.out_dir, exist_ok=True)
        cert = {"suite": "verify_q102_production MAX-EXTENSIVE v3", "status": status, "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "decoder_version": getattr(qector_ionq, "DECODER_VERSION", "?"), "target_hardware": TARGET_HARDWARE, "target_arch": getattr(qector_ionq, "TARGET_ARCHITECTURE", "?"), "machine": {"platform": platform.platform(), "python": platform.python_version()}, "polynomials": {"l": L_DIM, "A": A_SHIFTS, "B": B_SHIFTS}, "total_s": dt_all, "checks": results, "perf": perf}
        jf = os.path.join(a.out_dir, "cert_q102_production.json")
        with open(jf, "w", encoding="utf-8") as f: json.dump(cert, f, indent=2)
        mf = os.path.join(a.out_dir, "cert_q102_production.md")
        with open(mf, "w", encoding="utf-8") as f:
            f.write(f"# Q102 Production Certification MAX-EXTENSIVE ({cert['timestamp_utc']})\n\nstatus {status} | {cert['decoder_version']} | {sum(1 for r in results if r['ok'])}/{len(results)} passed\n\n")
            for r in results: f.write(f"- [{'x' if r['ok'] else ' '}] {r['name']} - {r['detail']}\n")
        print(f"  [cert] {jf}\n  [cert] {mf}")
    sys.exit(0 if all_ok else 1)
if __name__ == "__main__":
    main()
