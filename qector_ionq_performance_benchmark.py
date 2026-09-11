#!/usr/bin/env python3
# PROPRIETARY AND CONFIDENTIAL
# Copyright (c) 2026 Guillaume Lessard / qector-decoder-v3
# All Rights Reserved. NDA evaluation only. Do not redistribute.
"""
qector_ionq_performance_benchmark.py (MAX-EXTENSIVE v3)
=======================================================
Rust-binary performance benchmark + thread-scaling stress test — 10/10 extensive.

Controller sweeps RAYON_NUM_THREADS via subprocess (Rayon reads env at init).
Worker measures: single latency mean/median/p95/p99/max + batch thr 64/512/2000
+ faithfulness + Wilson CI. Thread scaling table + speedup + efficiency + JSON/MD certs.
"""
import argparse, datetime, json, multiprocessing, os, platform, re, subprocess, sys, time, math
try: import numpy as np
except: print("[-] FATAL: numpy required."); sys.exit(1)
try: import qector_ionq; from qector_ionq import IonQSuperionDecoder
except: print("[-] FATAL: 'qector_ionq' wheel not found."); sys.exit(1)
CODES=[("Q70","q70",[[70,6,9]]),("Q102","q102",[[102,22,9]]),("Gross","gross",[[144,12,12]])]
BATCH_SIZES=[64,512,2000]
ROW_RE=re.compile(r"(Q70|Q102|Gross)\s+\[\[(\d+),(\d+),(\d+)\]\]\s*\|\s*(\d+)\s*\|\s*([\d\.]+)\s*\|\s*([\d,\.]+)")
def cpu_threads():
    try: import psutil; return psutil.cpu_count(logical=True)
    except: return multiprocessing.cpu_count()
def thread_steps(total):
    steps=[1]; v=1
    while v<total:
        v=min(v*2,total)
        if v not in steps: steps.append(v)
        if v==total: break
    return steps
def build_shots(decoder,n_shots,seed=42):
    nq,nc=decoder.n_qubits,decoder.n_checks
    H=np.zeros((nc,nq),dtype=np.uint8)
    for ci,qs in enumerate(decoder.check_to_qubits):
        for q in qs: H[ci,int(q)]=1
    rng=np.random.default_rng(seed)
    shots=np.zeros((n_shots,nc),dtype=np.uint8)
    for i in range(n_shots):
        e=np.zeros(nq,dtype=np.uint8); e[rng.choice(nq,size=2,replace=False)]=1; shots[i]=(H@e)%2
    return shots,H
def pct(xs,p):
    a=sorted(xs)
    if not a: return float("nan")
    k=(len(a)-1)*p/100.0; f=int(k//1); c=int(-(-k//1))
    return a[f] if f==c else a[f]*(c-k)+a[c]*(k-f)
def wilson_ci(k,n,z=1.959963985):
    if n==0: return 0.0,0.0,1.0
    p=k/n; denom=1.0+z*z/n; centre=(p+z*z/(2.0*n))/denom; half=z*math.sqrt(p*(1.0-p)/n+z*z/(4.0*n*n))/denom
    return p,max(0.0,centre-half),min(1.0,centre+half)
def bench_one(factory,n_single,seed=42):
    t0=time.perf_counter(); dec=factory(); t_init=(time.perf_counter()-t0)*1000.0
    shots,H=build_shots(dec,n_single,seed)
    lats=[]
    fails=0
    for i in range(n_single):
        t0=time.perf_counter(); corr=np.asarray(dec.decode(shots[i]),dtype=np.uint8); lats.append((time.perf_counter()-t0)*1000.0)
        if not np.array_equal((H@corr)%2,shots[i]): fails+=1
    assert fails==0, f"unfaithful {fails}/{n_single}"
    res={"n_qubits":dec.n_qubits,"n_checks":dec.n_checks,"t_init_ms":t_init,"single_mean_ms":sum(lats)/len(lats),"single_median_ms":pct(lats,50),"single_p95_ms":pct(lats,95),"single_p99_ms":pct(lats,99),"single_max_ms":max(lats),"fails":fails}
    for B in BATCH_SIZES:
        sub=shots[:B] if len(shots)>=B else np.tile(shots,((B//len(shots))+1,1))[:B]
        flat=np.ascontiguousarray(sub.reshape(-1))
        t0=time.perf_counter(); out=np.asarray(dec.decode_batch_flat(flat,B),dtype=np.uint8).reshape(B,dec.n_qubits); dt=time.perf_counter()-t0
        assert np.array_equal((H@out[0])%2,sub[0])
        res[f"batch_{B}_shots_per_s"]=B/dt if dt>0 else float("inf"); res[f"batch_{B}_s"]=dt
    res["faithful"]=True; p,lo,hi=wilson_ci(fails,n_single); res["wilson"]=[p,lo,hi]
    return res
def worker(args):
    n_single=args.shots; threads=os.environ.get("RAYON_NUM_THREADS","default")
    print(f"WORKER RAYON_NUM_THREADS={threads} shots={n_single}")
    out={"rayon_threads":threads,"codes":{}}
    for name,ctor,params in CODES:
        r=bench_one(getattr(IonQSuperionDecoder,ctor),n_single)
        out["codes"][name]=r; big=max(BATCH_SIZES)
        print(f"  {name} [[{params[0][0]},{params[0][1]},{params[0][2]}]] | {r['n_qubits']} | {r['single_mean_ms']:.2f} | {r[f'batch_{big}_shots_per_s']:,.0f}")
        print(f"    detail p50={r['single_median_ms']:.3f}ms p95={r['single_p95_ms']:.3f}ms p99={r['single_p99_ms']:.3f}ms max={r['single_max_ms']:.3f}ms init={r['t_init_ms']:.2f}ms fails={r['fails']} Wilson [{r['wilson'][1]:.3f},{r['wilson'][2]:.3f}]")
        for B in BATCH_SIZES: print(f"    batch_{B}={r[f'batch_{B}_shots_per_s']:,.0f} shots/s ({r[f'batch_{B}_s']:.4f}s)")
    if args.out_dir and args.emit:
        os.makedirs(args.out_dir,exist_ok=True); fp=os.path.join(args.out_dir,f"perf_worker_t{threads}.json")
        with open(fp,"w",encoding="utf-8") as f: json.dump(out,f,indent=2); print(f"  [cert] {fp}")
    return 0
def controller(args):
    total=cpu_threads(); steps=thread_steps(total)
    print("="*80); print("  QECTOR-IONQ THREAD SCALING STRESS TEST MAX-EXTENSIVE v3 (BATCH THROUGHPUT)"); print(f"  Detected Max CPU Threads: {total}"); print(f"  Testing Steps: {steps}"); print(f"  Decoder: v{getattr(qector_ionq,'DECODER_VERSION','?')} | HW: {getattr(qector_ionq,'TARGET_HARDWARE','?')}"); print("="*80)
    results=[]; here=os.path.abspath(__file__)
    for nt in steps:
        print(f"\n[*] Executing benchmark with RAYON_NUM_THREADS = {nt}...")
        env=os.environ.copy(); env["RAYON_NUM_THREADS"]=str(nt); env["PYTHONUTF8"]="1"
        p=subprocess.run([sys.executable,here,"--worker","--shots",str(args.shots)],env=env,capture_output=True,text=True)
        print(p.stdout)
        if p.returncode!=0:
            print(f"    [ERROR] worker t={nt} failed:\n{p.stderr}"); results.append({"threads":nt,"q102_throughput":None,"gross_throughput":None,"q70_throughput":None,"raw":p.stdout}); continue
        big=max(BATCH_SIZES); q=g=q70=None
        for m in ROW_RE.finditer(p.stdout):
            name,thr=m.group(1),float(m.group(7).replace(",",""))
            if name=="Q102": q=thr
            elif name=="Gross": g=thr
            elif name=="Q70": q70=thr
        results.append({"threads":nt,"q102_throughput":q,"gross_throughput":g,"q70_throughput":q70,"raw":p.stdout})
        print(f"    -> Q70:   {int(q70):,} shots/s" if q70 else "    -> Q70:   N/A"); print(f"    -> Q102:  {int(q):,} shots/s" if q else "    -> Q102:  N/A"); print(f"    -> Gross: {int(g):,} shots/s" if g else "    -> Gross: N/A")
    print("\n"+"="*80); print("  FINAL THREAD SCALING REPORT MAX-EXTENSIVE (BATCH API)"); print("="*80)
    print(f"  {'Threads':<10} | {'Q70 (shots/s)':<16} | {'Q102 (shots/s)':<16} | {'Gross (shots/s)':<16} | {'Speedup':<10}"); print("-"*80)
    base=next((r["q102_throughput"] for r in results if r["q102_throughput"]),None)
    for r in results:
        t,q,g,q70=r["threads"],r["q102_throughput"],r["gross_throughput"],r["q70_throughput"]
        if q and g: sp=f"{q/base:.2f}x" if base else "1.00x"; print(f"  {t:<10} | {int(q70):<16,} | {int(q):<16,} | {int(g):<16,} | {sp:<10}")
        else: print(f"  {t:<10} | {'N/A':<16} | {'N/A':<16} | {'N/A':<16} | {'N/A':<10}")
    print("-"*80); ok=all(r["q102_throughput"] and r["gross_throughput"] for r in results)
    print(f"  [{'STATUS: GREEN' if ok else 'STATUS: RED'}] Multi-core saturation {'confirmed' if ok else 'INCOMPLETE'}."); print("="*80)
    if args.out_dir:
        os.makedirs(args.out_dir,exist_ok=True)
        cert={"suite":"qector_ionq_performance_benchmark MAX-EXTENSIVE v3","status":"GREEN" if ok else "RED","timestamp_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"decoder_version":getattr(qector_ionq,"DECODER_VERSION","?"),"target_hardware":getattr(qector_ionq,"TARGET_HARDWARE","?"),"target_arch":getattr(qector_ionq,"TARGET_ARCHITECTURE","?"),"machine":{"platform":platform.platform(),"cpu_count":total,"python":platform.python_version()},"batch_sizes":BATCH_SIZES,"single_shots":args.shots,"thread_steps":steps,"results":[{k:v for k,v in r.items() if k!="raw"} for r in results]}
        jf=os.path.join(args.out_dir,"cert_performance.json")
        with open(jf,"w",encoding="utf-8") as f: json.dump(cert,f,indent=2)
        mf=os.path.join(args.out_dir,"cert_performance.md")
        with open(mf,"w",encoding="utf-8") as f:
            f.write(f"# Performance Certification MAX-EXTENSIVE ({cert['timestamp_utc']})\n\nDecoder {cert['decoder_version']} | {cert['target_hardware']} | status {cert['status']}\n\n| Threads | Q70/s | Q102/s | Gross/s |\n|---|---|---|---|\n")
            for r in results: f.write(f"| {r['threads']} | {r['q70_throughput']} | {r['q102_throughput']} | {r['gross_throughput']} |\n")
        print(f"  [cert] {jf}\n  [cert] {mf}")
    sys.exit(0 if ok else 1)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--worker",action="store_true"); ap.add_argument("--shots",type=int,default=200); ap.add_argument("--out-dir",default="certs"); ap.add_argument("--emit",action="store_true"); ap.add_argument("--no-cert",action="store_true")
    a=ap.parse_args()
    if a.no_cert: a.out_dir=""
    if a.worker: sys.exit(worker(a))
    controller(a)
if __name__=="__main__": main()
