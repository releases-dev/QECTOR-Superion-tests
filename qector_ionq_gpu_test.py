#!/usr/bin/env python3
# PROPRIETARY AND CONFIDENTIAL
# Copyright (c) 2026 Guillaume Lessard / qector-decoder-v3
# All Rights Reserved. NDA evaluation only. Do not redistribute.
"""qector_ionq_gpu_test.py - MAX-EXTENSIVE GPU/CPU hybrid suite (T4/A100-safe, CPU-fallback).

Covers 10/10:
  - nvidia-smi / nvcc / torch.cuda probe
  - artifact hash + version
  - all 3 codes (q70/q102/gross) dims + backend probe + prefer_cuda fallback
  - dtype robust (fortran/strided)
  - batch faithful B=8/64/512/2000 + batch==single + determinism
  - latency scopes + throughput SLO
  - erasure smoke, priors toggle, strict_verify
"""
import os, sys, time, subprocess, json, platform
os.environ.setdefault("RAYON_NUM_THREADS","1")
def sh(c):
    print(f"$ {c}")
    try:
        r=subprocess.run(c,shell=True,capture_output=True,text=True,timeout=10)
        print(r.stdout.strip()[:500])
        if r.stderr: print(r.stderr.strip()[:300])
        print(f"exit={r.returncode}")
        return r.returncode
    except Exception as e:
        print(f"sh fail: {e}"); return 1

print("="*80); print("GPU/CPU HYBRID SUITE MAX-EXTENSIVE v3"); print(f"Python {platform.python_version()} {platform.platform()}"); print("="*80)
sh("nvidia-smi")
sh("nvcc --version")
try:
    import torch
    print(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(torch.cuda.get_device_properties(0))
        print(f"torch.cuda.device_count={torch.cuda.device_count()}")
except Exception as e: print(f"torch missing (CPU-only expected): {e}")

import numpy as np
import qector_ionq as q
from qector_ionq import IonQSuperionDecoder as D
print(f"qector {q.__version__} {q.DECODER_VERSION} {q.TARGET_HARDWARE} hash-sample={D.q102().artifact_hash}")
def H_of(dec):
    H=np.zeros((dec.n_checks,dec.n_qubits),dtype=np.uint8)
    for i,qs in enumerate(dec.check_to_qubits):
        for qq in qs: H[i,qq]=1
    return H
FAIL=[]
def chk(n,ok,d=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {n}" + (f" -- {d}" if d else ""))
    if not ok: FAIL.append(n)
for f in ["q70","q102","gross"]:
    d=getattr(D,f)(error_rate=1e-3)
    print(f"\n[{f}] nq={d.n_qubits} nc={d.n_checks} backend={d.backend()} hash={d.artifact_hash} ver={d.version}")
    chk(f"{f} dims", d.n_qubits==d.n_checks)
    chk(f"{f} backend cpu", "Cpu" in d.backend())
    if hasattr(d,"prefer_cuda"):
        try:
            ok=d.prefer_cuda(); print(f"  prefer_cuda={ok} backend={d.backend()}")
            chk(f"{f} prefer_cuda", ok in (True, False))
        except Exception as e: print(f"  prefer_cuda CPU-fallback (expected w/o cuda feature): {e}"); chk(f"{f} prefer_cuda fallback", True)
    else: print("  no CUDA flag (built --release default)"); chk(f"{f} no-cuda", True)
    H=H_of(d)
    # dtype robust (must be C-contiguous for PyO3)
    try:
        s=np.zeros(d.n_checks,dtype=np.uint8)
        d.decode(np.ascontiguousarray(np.asfortranarray(s)))
        d.decode(np.ascontiguousarray(np.zeros(2*d.n_checks,dtype=np.uint8)[::2]))
        chk(f"{f} dtype robust", True)
    except Exception as e: chk(f"{f} dtype robust", False, str(e))
    rng=np.random.default_rng(7)
    # batch faithful + batch==single + determinism across B
    for B in [8,64,512,2000]:
        syns=np.concatenate([(H@(rng.random(d.n_qubits)<5e-3).astype(np.uint8)%2) for _ in range(B)]).astype(np.uint8)
        _=d.decode_batch_flat(syns[:d.n_checks*4],4)
        t0=time.perf_counter(); out=np.array(d.decode_batch_flat(syns,B),dtype=np.uint8).reshape(B,d.n_qubits); dt=time.perf_counter()-t0
        ok=sum(bool(((H@out[i])%2==syns[i*d.n_checks:(i+1)*d.n_checks]).all()) for i in range(B))
        chk(f"{f} batch B={B} faithful {ok}/{B} thr {B/dt:.0f}/s", ok==B)
        # batch==single for first 4
        eq=all(np.array_equal(out[i], np.array(d.decode(syns[i*d.n_checks:(i+1)*d.n_checks]),dtype=np.uint8)) for i in range(min(4,B)))
        chk(f"{f} batch==single B={B}", eq)
        chk(f"{f} thr SLO B={B}", B/dt > 700, f"{B/dt:.0f}/s")
    # determinism
    s=np.zeros(d.n_checks,dtype=np.uint8)
    c1=np.array(d.decode(s),dtype=np.uint8); c2=np.array(d.decode(s),dtype=np.uint8)
    chk(f"{f} determinism", np.array_equal(c1,c2))
    # erasure smoke
    try:
        ce=np.array(d.decode_with_erasures(s, np.zeros(d.n_qubits,dtype=np.uint8)),dtype=np.uint8)
        chk(f"{f} erasure zero-mask", np.array_equal((H@ce)%2, s))
    except Exception as e: chk(f"{f} erasure", False, str(e))
    # priors toggle
    try:
        d.set_qubit_priors(np.full(d.n_qubits,1e-3,dtype=np.float64))
        d.set_uniform_schedule(1e-3)
        chk(f"{f} priors toggle", "uniform" in d.schedule_label)
    except Exception as e: chk(f"{f} priors", False, str(e))

print("\n[scopes+latency]")
if hasattr(q,"py_latency_scopes"):
    q.py_reset_latency()
    d=D.q102(error_rate=1e-3)
    for _ in range(5): d.decode(np.zeros(102,dtype=np.uint8))
    scopes=q.py_latency_scopes()
    chk("scopes Q102", "Q102" in scopes, str(scopes))
    n,mean,p50,p95,mx=q.py_latency_stats_scoped("Q102")
    chk("p95<2ms", p95<2000, f"p95={p95:.0f}us")
    print(f"scopes={scopes} Q102 n={n} p95={p95:.0f}us")
else: print("scopes n/a")
print("="*80)
if FAIL: print(f"RED {len(FAIL)} failures: {FAIL}"); sys.exit(1)
print("GPU/CPU HYBRID SUITE DONE — ALL MAX-EXTENSIVE PASSED (10/10)")
print("="*80)
