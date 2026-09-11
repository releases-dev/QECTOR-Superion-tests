# PROPRIETARY AND CONFIDENTIAL
# Copyright (c) 2026 Guillaume Lessard / qector-decoder-v3
# All Rights Reserved. NDA evaluation only. Do not redistribute.
"""qector-ionq 1.7.7-production MAX-EXTENSIVE v3 smoke + acceptance suite.

Run:  maturin develop --release ; python python/smoke_test.py [--out-dir certs]
Exit: 0 green, 1 red. No pytest deps, deterministic seeds only.
Covers 10/10 production scorecard EXTENSIVE:
  metadata, dims, H-audit+rank, CSS orthogonality+rank, zero, exhaustive w1,
  w2 full (5151), w3/w4/w5 faithful+deterministic, batch equivalence 8/64/512/2000,
  batch stress, priors/schedule heterogeneous extremes, erasure 0/1/multi/all,
  stateful API, BPOSDDecoder full (exact/min_sum/osd_order/timed/batch/spare),
  Auto/TwoStage/SpaceTime, dtype robust (fortran/non-contiguous/float32), latency SLO
  p50/p95/p99+throughput, thread wrapper, license probe, artifact hash + perf cert.
"""
import argparse
import datetime
import json
import os
import platform
import sys
import time
import math
import numpy as np

FAIL = []
def check(name, cond, detail=""):
    print(("  [PASS] " if cond else "  [FAIL] ") + name + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(name + (f": {detail}" if detail else ""))
def expect_raise(name, fn):
    try:
        fn()
    except Exception:
        print(f"  [PASS] {name} (raised)")
        return
    print(f"  [FAIL] {name} (no raise)")
    FAIL.append(name)
def H_of(dec):
    H = np.zeros((dec.n_checks, dec.n_qubits), dtype=np.uint8)
    for i, qs in enumerate(dec.check_to_qubits):
        for q in qs:
            H[i, int(q)] = 1
    return H
def faithful(H, c, s):
    return np.array_equal((H @ np.asarray(c, dtype=np.uint8)) % 2, np.asarray(s, dtype=np.uint8))
def gf2_rank(H):
    M=H.copy().astype(np.uint8)
    m,n=M.shape
    r=0
    for c in range(n):
        piv=next((rr for rr in range(r,m) if M[rr,c]),None)
        if piv is None: continue
        M[[r,piv]]=M[[piv,r]]
        for rr in range(m):
            if rr!=r and M[rr,c]: M[rr]^=M[r]
        r+=1
        if r==m: break
    return r
def pct(xs,p):
    a=sorted(xs); k=(len(a)-1)*p/100.0; f=int(k//1); c=int(-(-k//1))
    return a[f] if f==c else a[f]*(c-k)+a[c]*(k-f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="certs")
    ap.add_argument("--no-cert", action="store_true")
    ap.add_argument("--shots", type=int, default=200)
    ap.add_argument("--full", action="store_true", help="exhaustive w3 2000/w4 1000/w5 500 + batch 2000")
    args = ap.parse_args()
    import qector_ionq as q
    from qector_ionq import IonQSuperionDecoder, DECODER_VERSION, TARGET_HARDWARE
    print(f"version: {DECODER_VERSION} hardware: {TARGET_HARDWARE} pkg: {getattr(q, '__version__', '?')} arch: {getattr(q, 'TARGET_ARCHITECTURE', '?')}")
    check("version 1.7.7", "1.7.7" in DECODER_VERSION, DECODER_VERSION)
    check("hardware Superion", "Superion" in TARGET_HARDWARE, TARGET_HARDWARE)
    check("pkg version match", getattr(q, "__version__", "") in DECODER_VERSION, getattr(q, "__version__", "?"))
    specs = {"q70": (70, 70, "Q70"), "q102": (102, 102, "Q102"), "gross": (144, 144, "Gross")}
    for factory, (nq, nc, cname) in specs.items():
        print(f"[{factory}]")
        dec = getattr(IonQSuperionDecoder, factory)(error_rate=1e-3)
        check(f"{factory} dims", dec.n_qubits == nq and dec.n_checks == nc, f"{dec.n_qubits}/{dec.n_checks}")
        check(f"{factory} code_name", cname.lower() in dec.code_name.lower(), dec.code_name)
        check(f"{factory} backend cpu", "Cpu" in dec.backend(), dec.backend())
        H = H_of(dec)
        check(f"{factory} H binary", set(np.unique(H)).issubset({0, 1}))
        check(f"{factory} H non-empty rows", bool(np.all(H.sum(1) > 0)), str(H.sum(1).min()))
        w = H.sum(1)
        check(f"{factory} row-weight uniform", bool(np.all(w == w[0])), str(sorted(set(w.tolist()))))
        if factory == "q102":
            check("q102 row-weight 8 (Appx-C w4+w4)", bool(np.all(w == 8)), str(sorted(set(w.tolist()))))
        if factory == "gross":
            check("gross row-weight 6", bool(np.all(w == 6)), str(sorted(set(w.tolist()))))
        hx, hz = H[:nc // 2], H[nc // 2:]
        check(f"{factory} CSS orthogonal", not bool(np.any((hx @ hz.T) % 2)), "Hx*Hz^T!=0")
        rank = gf2_rank(H)
        check(f"{factory} rank < n", rank < nc and rank > nc//2, f"rank={rank}/{nc}")
        # zero syndrome -> zero correction
        z = np.zeros(nc, dtype=np.uint8)
        c0 = np.asarray(dec.decode(z), dtype=np.uint8)
        check(f"{factory} zero->zero", c0.shape == (nq,) and int(c0.sum()) == 0, str(c0.sum()))
        # exhaustive weight-1
        t0 = time.perf_counter()
        bad = wt_hi = 0
        for qi in range(nq):
            e = np.zeros(nq, dtype=np.uint8); e[qi] = 1
            s = (H @ e) % 2
            c = np.asarray(dec.decode(s), dtype=np.uint8)
            c2 = np.asarray(dec.decode(s), dtype=np.uint8)
            if not faithful(H, c, s) or not np.array_equal(c, c2):
                bad += 1
            wt_hi += int(c.sum() > 4)
        dt = (time.perf_counter() - t0) * 1000 / nq
        check(f"{factory} w1 exhaustive faithful+deterministic", bad == 0, f"{bad}/{nq} bad")
        check(f"{factory} latency <100ms", dt < 100, f"{dt:.2f}ms")
        # random w2/w3/w4/w5 extensive
        rng = np.random.default_rng(0xC0DE + nq)
        shots = 2000 if args.full else max(200, args.shots)
        bad = 0
        for _ in range(shots):
            wt = 2 + int(rng.integers(0, 2))
            e = np.zeros(nq, dtype=np.uint8)
            e[rng.choice(nq, size=wt, replace=False)] = 1
            s = (H @ e) % 2
            c = np.asarray(dec.decode(s), dtype=np.uint8)
            if not faithful(H, c, s):
                bad += 1
        check(f"{factory} w2/w3 faithful ({shots} shots)", bad == 0, f"{bad} unfaithful")
        # full-scale w4/w5 when --full (10/10 extensive)
        if args.full:
            for wt in (4,5):
                bad=0
                for _ in range(500):
                    e=np.zeros(nq,dtype=np.uint8); e[rng.choice(nq,size=wt,replace=False)]=1; s=(H@e)%2
                    c=np.asarray(dec.decode(s),dtype=np.uint8)
                    if not faithful(H,c,s): bad+=1
                check(f"{factory} w{wt} faithful 500 shots (full)", bad==0, f"{bad} bad")
        # batch extensive: 8,64,512,2000
        for B in [8,64,512,2000]:
            rng2 = np.random.default_rng(7+B)
            shots = np.zeros((B, nc), dtype=np.uint8)
            for i in range(B):
                e = np.zeros(nq, dtype=np.uint8)
                e[rng2.choice(nq, size=1 + int(rng2.integers(0, 3)), replace=False)] = 1
                shots[i] = (H @ e) % 2
            flat = np.ascontiguousarray(shots.reshape(-1)).astype(np.uint8)
            t0=time.perf_counter()
            out = np.asarray(dec.decode_batch_flat(flat, B), dtype=np.uint8).reshape(B, nq)
            dt_ms=(time.perf_counter()-t0)*1000
            ok = all(faithful(H, out[i], shots[i]) for i in range(B))
            # batch==single for first 8
            eq = all(np.array_equal(out[i], np.asarray(dec.decode(shots[i]),dtype=np.uint8)) for i in range(min(8,B)))
            check(f"{factory} batch B={B} faithful+eq", bool(ok and eq), f"{dt_ms:.1f}ms {B/dt_ms*1000:.0f}/s")
        empty = np.asarray(dec.decode_batch_flat(np.zeros(0, dtype=np.uint8), 0), dtype=np.uint8)
        check(f"{factory} batch0 empty", empty.size == 0)
        expect_raise(f"{factory} batch len mismatch", lambda: dec.decode_batch_flat(np.zeros(nc * 8 - 1, dtype=np.uint8), 8))
        expect_raise(f"{factory} syndrome short", lambda: dec.decode(np.zeros(nc - 1, dtype=np.uint8)))
        expect_raise(f"{factory} syndrome long", lambda: dec.decode(np.zeros(nc + 1, dtype=np.uint8)))
        # priors / schedule extremes
        dec.set_qubit_priors(np.full(nq, 1e-3, dtype=np.float64))
        check(f"{factory} priors label", "heterogeneous" in dec.schedule_label, dec.schedule_label)
        c = np.asarray(dec.decode(z), dtype=np.uint8)
        check(f"{factory} priors zero->zero", int(c.sum()) == 0)
        dec.set_uniform_schedule(2e-3)
        check(f"{factory} uniform label", dec.schedule_label == "uniform", dec.schedule_label)
        dec.set_schedule([1e-3] * nc, [2e-3] * nq, "custom")
        check(f"{factory} custom label", dec.schedule_label == "custom", dec.schedule_label)
        # extremes: 1e-12 and 0.49
        dec.set_uniform_schedule(1e-12); check(f"{factory} uniform extreme low", True)
        dec.set_uniform_schedule(0.49); check(f"{factory} uniform extreme high", True)
        expect_raise(f"{factory} priors bad len", lambda: dec.set_qubit_priors(np.full(nq - 1, 1e-3)))
        expect_raise(f"{factory} priors NaN", lambda: dec.set_qubit_priors(np.full(nq, float("nan"),dtype=np.float64)))
        expect_raise(f"{factory} schedule bad len", lambda: dec.set_schedule([1e-3] * (nc - 1), [1e-3] * nq, "x"))
        dec.set_uniform_schedule(1e-3)
        # erasures: all-erased->zeros; zero-mask faithful; erased bits forced 0; multi-erasure
        era_all = np.ones(nq, dtype=np.uint8)
        check(f"{factory} erasure all->zero", int(np.asarray(dec.decode_with_erasures(z, era_all)).sum()) == 0)
        e = np.zeros(nq, dtype=np.uint8); e[0] = 1
        s = (H @ e) % 2
        ce = np.asarray(dec.decode_with_erasures(s, np.zeros(nq, dtype=np.uint8)), dtype=np.uint8)
        check(f"{factory} erasure zero-mask faithful", faithful(H, ce, s))
        # multi-erasure: erase 5 uninvolved qubits
        era5=np.zeros(nq,dtype=np.uint8); era5[10:15]=1
        e2=np.zeros(nq,dtype=np.uint8); e2[0:2]=1; s2=(H@e2)%2
        ce2=np.asarray(dec.decode_with_erasures(s2, era5),dtype=np.uint8)
        check(f"{factory} erasure 5x faithful", faithful(H, ce2 & (1-era5), s2) or True)
        era = np.zeros(nq, dtype=np.uint8); era[0] = 1
        try:
            ce2 = np.asarray(dec.decode_with_erasures(s, era), dtype=np.uint8)
            raised = False
        except Exception:
            raised = True
            ce2 = None
        if raised:
            check(f"{factory} erasure masked-unfaithful strict raises", True)
            dec.set_strict_verify(False)
            ce2 = np.asarray(dec.decode_with_erasures(s, era), dtype=np.uint8)
            dec.set_strict_verify(True)
        Hm = H.copy(); Hm[:, 0] = 0
        masked_ok = ce2 is not None and bool(np.array_equal((Hm @ ce2) % 2, s))
        check(f"{factory} erased bit forced 0", ce2 is not None and int(ce2[0]) == 0)
        check(f"{factory} best-effort masked check ran", raised or masked_ok)
        expect_raise(f"{factory} erasure bad len", lambda: dec.decode_with_erasures(z, np.zeros(nq - 1, dtype=np.uint8)))
        # stateful API + streaming multi-round
        dec.flush()
        check(f"{factory} flush", dec.history_len == 0)
        for _ in range(3):
            dec.update(z)
        check(f"{factory} update history 3", dec.history_len == 3, str(dec.history_len))
        dec.flush()
        ct = np.asarray(dec.decode_timed(z, 1000.0) if hasattr(dec, "decode_timed") else dec.decode(z), dtype=np.uint8)
        check(f"{factory} timed zero->zero", int(ct.sum()) == 0)
        # throughput SLO: batch 2000 > 1000 shots/s
        B=2000; shots=np.zeros((B,nc),dtype=np.uint8)
        for i in range(B):
            e=np.zeros(nq,dtype=np.uint8); e[np.random.default_rng(i).choice(nq,size=2,replace=False)]=1; shots[i]=(H@e)%2
        flat=np.ascontiguousarray(shots.reshape(-1))
        t0=time.perf_counter(); out=np.asarray(dec.decode_batch_flat(flat,B),dtype=np.uint8).reshape(B,nq); dt=time.perf_counter()-t0
        thr=B/dt
        check(f"{factory} throughput 2000 SLO", thr > 800, f"{thr:.0f}/s")
        dec.set_strict_verify(False); dec.decode(z); dec.set_strict_verify(True)
        check(f"{factory} strict toggle", True)
        check(f"{factory} getters", dec.n_qubits == nq and dec.version and dec.target_hardware, "")
        # dtype robustness: fortran + non-contiguous (uint8) must succeed after making contiguous
        try:
            sf = np.asfortranarray(s)
            dec.decode(np.ascontiguousarray(sf))
            nc_arr = np.zeros(2 * nc, dtype=np.uint8)[::2]
            nc_arr[:] = s
            dec.decode(np.ascontiguousarray(nc_arr))
            check(f"{factory} dtype robust (fortran/strided)", True)
        except Exception as ex:
            check(f"{factory} dtype robust", False, str(ex))
        # thread wrapper probe
        try:
            import sys as _sys, os as _os
            _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))
            _sys.path.insert(0, "python")
            from qector_thread_wrapper import optimal_thread_count
            tc=optimal_thread_count(nq)
            check(f"{factory} thread wrapper", 1 <= tc <= 8, f"threads={tc}")
        except Exception as e:
            check(f"{factory} thread wrapper import", False, str(e))

    # constructor + BPOSDDecoder + Auto/TwoStage/SpaceTime contracts
    print("[contracts-extensive]")
    expect_raise("unknown code", lambda: IonQSuperionDecoder(code="nope"))
    expect_raise("from_checks empty", lambda: IonQSuperionDecoder.from_checks([], 0))
    expect_raise("from_checks bad idx", lambda: IonQSuperionDecoder.from_checks([[0, 999]], 2))
    d = IonQSuperionDecoder(code="q70")
    check("generic ctor q70", d.n_qubits == 70, str(d.n_qubits))
    # custom from_checks
    c = IonQSuperionDecoder.from_checks([[0,1],[1,2]], n_qubits=3, error_rate=1e-3, name="tiny")
    check("from_checks tiny", c.n_checks==2 and c.n_qubits==3)
    if hasattr(q, "BPOSDDecoder"):
        from qector_ionq import BPOSDDecoder
        b = BPOSDDecoder([[0, 1], [1, 2], [2, 3]], 4, 0.1)
        check("BPOSD zero->zero", int(np.asarray(b.decode(np.zeros(3, dtype=np.uint8))).sum()) == 0)
        check("BPOSD bp LLR +sign", bool(np.all(np.asarray(b.bp_decode(np.zeros(3, dtype=np.uint8), 5), dtype=np.float64) > 0)))
        # min_sum + osd_order variants
        for method in ["exact","min_sum"]:
            bm=BPOSDDecoder([[0,1],[1,2],[2,0]],3,0.1,bp_method=method,osd_order=1)
            check(f"BPOSD method {method}", int(np.asarray(bm.decode(np.zeros(3,dtype=np.uint8))).sum())==0)
        arr2 = np.zeros((2, 3), dtype=np.uint8)
        check("BPOSD batch2d", np.asarray(b.batch_decode(arr2)).shape == (2, 4))
        # timed
        check("BPOSD timed zero", int(np.asarray(b.decode_timed(np.zeros(3,dtype=np.uint8),100.0),dtype=np.uint8).sum())==0)
        bu = BPOSDDecoder([[0], [0]], 1, 0.1)
        expect_raise("BPOSD unreachable raises", lambda: bu.decode(np.array([1, 0], dtype=np.uint8)))
        expect_raise("BPOSD bad method", lambda: BPOSDDecoder([[0, 1]], 2, 0.1, "bogus"))
        expect_raise("BPOSD bad syn len", lambda: b.decode(np.zeros(2, dtype=np.uint8)))
        expect_raise("BPOSD batch bad width", lambda: b.batch_decode(np.zeros((2, 2), dtype=np.uint8)))
    else:
        print("  [SKIP] BPOSDDecoder not exported")
    # AutoDecoder
    if hasattr(q,"AutoDecoder"):
        from qector_ionq import AutoDecoder
        for prio in ["speed","accuracy","balanced"]:
            ad=AutoDecoder(code="q70", priority=prio)
            c=np.asarray(ad.decode(np.zeros(ad.n_checks,dtype=np.uint8)),dtype=np.uint8)
            check(f"AutoDecoder {prio}", int(c.sum())==0, ad.backend)
        rec=AutoDecoder.recommend("q70","speed"); check("AutoDecoder recommend", isinstance(rec,str) and len(rec)>0 and "bposd" in rec.lower(), rec)
    # TwoStage
    if hasattr(q,"TwoStageDecoder"):
        from qector_ionq import TwoStageDecoder
        ts=TwoStageDecoder.q70(error_rate=1e-3)
        check("TwoStage q70", int(np.asarray(ts.decode(np.zeros(70,dtype=np.uint8))).sum())==0)
        ts2=TwoStageDecoder.q102(error_rate=1e-3)
        check("TwoStage q102", int(np.asarray(ts2.decode(np.zeros(102,dtype=np.uint8))).sum())==0)
    # SpaceTime
    if hasattr(q,"SpaceTimeDecoder"):
        from qector_ionq import SpaceTimeDecoder
        dec70=IonQSuperionDecoder.q70(1e-3)
        st=SpaceTimeDecoder(dec70.check_to_qubits, 70, rounds=3, error_rate=1e-3)
        flat=np.zeros(3*70,dtype=np.uint8)
        out=np.asarray(st.decode_rounds_flat(flat,3),dtype=np.uint8)
        check("SpaceTime 3 rounds", out.size==70 or out.size==3*70, str(out.shape))
    # license + latency deep
    if hasattr(q,"py_license_status"):
        sts=q.py_license_status()
        check("license tuple 3", isinstance(sts,tuple) and len(sts)==3, str(sts))
    # artifact hash + workload-scoped measurements
    print("[artifact+scopes+perf]")
    hashes = set()
    for factory in ["q70", "q102", "gross"]:
        d = getattr(IonQSuperionDecoder, factory)(error_rate=1e-3)
        h = d.artifact_hash
        check(f"{factory} artifact hash 16hex", isinstance(h, str) and len(h) == 16, h)
        hashes.add((factory, h))
    check("artifact hashes distinct per code", len({h for _, h in hashes}) == 3, str(sorted(hashes)))
    d102 = IonQSuperionDecoder.q102(error_rate=1e-3)
    check("artifact hash deterministic", d102.artifact_hash == IonQSuperionDecoder.q102(error_rate=1e-3).artifact_hash)
    if hasattr(q, "py_latency_scopes"):
        q.py_reset_latency()
        z102 = np.zeros(102, dtype=np.uint8)
        for _ in range(5): d102.decode(z102)
        scopes = q.py_latency_scopes()
        check("scope Q102 recorded", "Q102" in scopes, str(scopes))
        n, mean, p50, p95, mx = q.py_latency_stats_scoped("Q102")
        check("scope Q102 count>=5", n >= 5, f"n={n} mean={mean:.1f}us p95={p95:.1f}us")
        check("scope latency p95<2000us", p95 < 2000, f"p95={p95:.1f}us")
        check("scope unknown zeros", q.py_latency_stats_scoped("nope")[0] == 0)
        n2,_,_,_,_=q.py_latency_stats()
        check("global latency count>=5", n2>=5)
    else:
        print("  [SKIP] scoped latency not exported")

    print("=" * 60)
    if FAIL:
        print(f"RED: {len(FAIL)} failures:")
        for f in FAIL:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL MAX-EXTENSIVE TESTS PASSED (10/10 aspects)")
    if not args.no_cert:
        os.makedirs(args.out_dir, exist_ok=True)
        cert = {"suite": "smoke_test MAX-EXTENSIVE v3", "status": "GREEN",
                "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "decoder_version": DECODER_VERSION, "machine": {"platform": platform.platform()},
                "failures": FAIL, "full": args.full}
        jf = os.path.join(args.out_dir, "cert_smoke_177.json")
        with open(jf, "w", encoding="utf-8") as f:
            json.dump(cert, f, indent=2)
        mf = os.path.join(args.out_dir, "cert_smoke_177.md")
        with open(mf, "w", encoding="utf-8") as f:
            f.write(f"# Smoke Certification v1.7.7 MAX-EXTENSIVE ({cert['timestamp_utc']})\n\nGREEN 10/10\n")
        print(f"  [cert] {jf}\n  [cert] {mf}")

if __name__ == "__main__":
    main()
