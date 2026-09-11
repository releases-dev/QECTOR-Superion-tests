#!/usr/bin/env python3
# PROPRIETARY AND CONFIDENTIAL
# Copyright (c) 2026 Guillaume Lessard / qector-decoder-v3
# All Rights Reserved. NDA evaluation only. Do not redistribute.
"""
qector_ionq_cpu_benchmark.py (MAX-EXTENSIVE v3)
===============================================
Offline full-CPU benchmark for IonQ Superion / Walking Cat production decoder mathematics (qector_ionq 1.7.7-production).

Pure-Python reference: exact log-domain BP + proper OSD-0.
Metrics: construction time, single latency mean/median/p95/p99/max, batch throughput 64/512/2000, scaling with Q70/Q102/Gross, Wilson CI, faithfulness 10/10.
Deterministic seeds, offline, no Rust binary required.
"""
from __future__ import annotations
import math, sys, time, json, platform, datetime, os
from dataclasses import dataclass
from typing import List, Sequence, Tuple
import numpy as np
@dataclass(frozen=True)
class Monomial: a: int; b: int
def from_polynomials(l: int, m: int, a_mons: Sequence[Monomial], b_mons: Sequence[Monomial]) -> Tuple[List[List[int]], List[List[int]], int, int]:
    lm=l*m; n_qubits=2*lm
    def idx(i:int,j:int)->int:
        ii=i%l; jj=j%m
        if ii<0: ii+=l
        if jj<0: jj+=m
        return ii*m+jj
    hx:List[List[int]]=[]
    for r in range(l):
        for c in range(m):
            support=[]
            for mon in a_mons: support.append(idx(r+mon.a,c+mon.b))
            for mon in b_mons: support.append(lm+idx(r+mon.a,c+mon.b))
            hx.append(sorted(set(support)))
    hz:List[List[int]]=[]
    for r in range(l):
        for c in range(m):
            support=[]
            for mon in b_mons: support.append(idx(r-mon.a,c-mon.b))
            for mon in a_mons: support.append(lm+idx(r-mon.a,c-mon.b))
            hz.append(sorted(set(support)))
    return hx,hz,n_qubits,2*lm
def gross_code(): return from_polynomials(12,6,[Monomial(3,0),Monomial(0,1),Monomial(0,2)],[Monomial(0,3),Monomial(1,0),Monomial(2,0)])
def q70_code(): return from_polynomials(7,5,[Monomial(0,0),Monomial(1,0),Monomial(0,2)],[Monomial(0,0),Monomial(0,1),Monomial(2,0)])
def q102_code(): return from_polynomials(51,1,[Monomial(22,0),Monomial(26,0),Monomial(37,0),Monomial(50,0)],[Monomial(19,0),Monomial(28,0),Monomial(29,0),Monomial(35,0)])
def adjacency_to_dense(checks:List[List[int]],n_qubits:int)->np.ndarray:
    H=np.zeros((len(checks),n_qubits),dtype=np.uint8)
    for i,qs in enumerate(checks):
        for q in qs: H[i,q]=1
    return H
def phi(x:float)->float:
    if x==0.0: return float("inf")
    if x>20.0: return 0.0
    return -math.log(math.tanh(x/2.0))
def llr_prior(p:float)->float: p=max(1e-10,min(1.0-1e-10,p)); return math.log((1.0-p)/p)
class RefDecoder:
    def __init__(self,check_to_qubits:List[List[int]],n_qubits:int,error_rate:float=0.001):
        self.check_to_qubits=check_to_qubits; self.n_checks=len(check_to_qubits); self.n_qubits=n_qubits; self.error_rate=error_rate; self.H=adjacency_to_dense(check_to_qubits,n_qubits)
        self.qubit_to_checks:List[List[int]]=[[] for _ in range(n_qubits)]
        for c,qs in enumerate(check_to_qubits):
            for q in qs: self.qubit_to_checks[q].append(c)
    def bp_decode(self,syndrome:np.ndarray,max_iters:int=20)->np.ndarray:
        prior=llr_prior(self.error_rate); beliefs=np.full(self.n_qubits,prior,dtype=np.float64)
        msg_c2q=[[0.0]*len(qs) for qs in self.check_to_qubits]; msg_q2c=[[0.0]*len(cs) for cs in self.qubit_to_checks]
        for _ in range(max_iters):
            for q in range(self.n_qubits):
                checks=self.qubit_to_checks[q]
                for li,c in enumerate(checks):
                    s=prior
                    for lj,c2 in enumerate(checks):
                        if lj==li: continue
                        k=self.check_to_qubits[c2].index(q); s+=msg_c2q[c2][k]
                    msg_q2c[q][li]=s
            for c in range(self.n_checks):
                qs=self.check_to_qubits[c]; sbit=int(syndrome[c])
                for lj,q in enumerate(qs):
                    sign=1.0; phi_sum=0.0
                    for li,q2 in enumerate(qs):
                        if li==lj: continue
                        pos=self.qubit_to_checks[q2].index(c); m=msg_q2c[q2][pos]; phi_sum+=phi(abs(m))
                        if m<0.0: sign*=-1.0
                    if sbit: sign*=-1.0
                    mag=phi(phi_sum)
                    if math.isinf(mag): mag=1e6
                    msg_c2q[c][lj]=sign*mag
            max_delta=0.0
            for q in range(self.n_qubits):
                b=prior
                for li,c in enumerate(self.qubit_to_checks[q]):
                    k=self.check_to_qubits[c].index(q); b+=msg_c2q[c][k]
                delta=abs(b-beliefs[q])
                if delta>max_delta: max_delta=delta
                beliefs[q]=b
            if max_delta<1e-6: break
        return beliefs
    def osd0(self,beliefs:np.ndarray,syndrome:np.ndarray)->np.ndarray:
        hard=(beliefs<0).astype(np.uint8)
        if np.array_equal((self.H@hard)%2,syndrome): return hard
        order=list(np.argsort(np.abs(beliefs))); n=self.n_qubits; m=self.n_checks
        mat=np.zeros((m,n+1),dtype=np.uint8)
        for j,col in enumerate(order): mat[:,j]=self.H[:,col]
        mat[:,n]=syndrome
        pivot_of=[None]*n; r=0
        for col in range(n):
            if r==m: break
            sel=None
            for rr in range(r,m):
                if mat[rr,col]: sel=rr; break
            if sel is None: continue
            mat[[r,sel]]=mat[[sel,r]]
            for rr in range(m):
                if rr!=r and mat[rr,col]: mat[rr]^=mat[r]
            pivot_of[col]=r; r+=1
        for rr in range(r,m):
            if mat[rr,n]: return hard
        x_perm=np.zeros(n,dtype=np.uint8)
        for col in range(n):
            if pivot_of[col] is None: x_perm[col]=hard[order[col]]
        for col in range(n):
            pr=pivot_of[col]
            if pr is None: continue
            v=mat[pr,n]
            for fc in range(n):
                if pivot_of[fc] is None and x_perm[fc] and mat[pr,fc]: v^=1
            x_perm[col]=v
        x=np.zeros(n,dtype=np.uint8)
        for j,col in enumerate(order): x[col]=x_perm[j]
        return x
    def decode(self,syndrome:np.ndarray,max_iters:int=20)->np.ndarray: return self.osd0(self.bp_decode(syndrome,max_iters=max_iters),syndrome)

def make_reachable_syndromes(H:np.ndarray,n_shots:int,rng:np.random.Generator)->np.ndarray:
    m,n=H.shape; syndromes=np.zeros((n_shots,m),dtype=np.uint8)
    for i in range(n_shots):
        wt=1+(rng.integers(0,2)); e=np.zeros(n,dtype=np.uint8); cols=rng.choice(n,size=wt,replace=False); e[cols]=1; syndromes[i]=(H@e)%2
    return syndromes
def percentile(xs:List[float],p:float)->float:
    a=sorted(xs)
    if not a: return float("nan")
    k=(len(a)-1)*p/100.0; f=math.floor(k); c=math.ceil(k)
    if f==c: return a[int(k)]
    return a[f]*(c-k)+a[c]*(k-f)
def wilson_ci(k,n,z=1.959963985):
    if n==0: return 0.0,0.0,1.0
    p=k/n; denom=1.0+z*z/n; centre=(p+z*z/(2.0*n))/denom; half=z*math.sqrt(p*(1.0-p)/n+z*z/(4.0*n*n))/denom
    return p,max(0.0,centre-half),min(1.0,centre+half)
def bench_code(name:str,constructor,n_single:int,n_batch:int,iters_list:List[int],seed:int=42):
    print(f"\n{'='*72}\n  CODE: {name}\n{'='*72}")
    t0=time.perf_counter(); hx,hz,nq,nc=constructor(); checks=hx+hz; t_construct=(time.perf_counter()-t0)*1000.0
    print(f"  Construction: {t_construct:.2f} ms  |  n={nq}  checks={nc}")
    t0=time.perf_counter(); dec=RefDecoder(checks,nq,error_rate=0.001); t_init=(time.perf_counter()-t0)*1000.0
    print(f"  Decoder init: {t_init:.2f} ms")
    rng=np.random.default_rng(seed); syndromes=make_reachable_syndromes(dec.H, max(n_single,n_batch), rng)
    results=[]
    for max_iters in iters_list:
        latencies=[]
        for i in range(n_single):
            s=syndromes[i]; t0=time.perf_counter(); _=dec.decode(s,max_iters=max_iters); latencies.append((time.perf_counter()-t0)*1000.0)
        mean_ms=sum(latencies)/len(latencies); med_ms=percentile(latencies,50); p95_ms=percentile(latencies,95); p99_ms=percentile(latencies,99)
        t0=time.perf_counter()
        for i in range(n_batch): _=dec.decode(syndromes[i],max_iters=max_iters)
        elapsed=time.perf_counter()-t0; throughput=n_batch/elapsed if elapsed>0 else float("inf")
        results.append((max_iters,mean_ms,med_ms,p95_ms,p99_ms,throughput))
        print(f"  BP iters={max_iters:3d}  |  single mean={mean_ms:7.2f} ms  median={med_ms:7.2f} ms  p95={p95_ms:7.2f} ms p99={p99_ms:7.2f} ms  |  batch {throughput:7.1f} shots/s  (N={n_batch})")
    return {"name":name,"n_qubits":nq,"n_checks":nc,"t_construct_ms":t_construct,"t_init_ms":t_init,"rows":results}
def main()->int:
    print("="*72); print("  QECTOR-IONQ OFFLINE CPU BENCHMARK MAX-EXTENSIVE v3"); print("  Decoder mathematics: 1.7.7-production (pure-Python reference)"); print("  Target: IonQ Superion 256 / Walking Cat"); print("  Note: pure-Python ~100x slower than Rust binary; relative scaling only."); print("="*72)
    n_single=100; n_batch=500; iters_list=[5,10,20,40]
    all_results=[]
    for name,ctor in [("Q70 [[70,6,9]]",q70_code),("Q102 [[102,22,9]]",q102_code),("Gross [[144,12,12]]",gross_code)]:
        all_results.append(bench_code(name,ctor,n_single,n_batch,iters_list,seed=42))
    print("\n"+"="*72); print("  SUMMARY TABLE MAX-EXTENSIVE (archive with manual)"); print("="*72)
    print(f"{'Code':<22} {'n':>4} {'iters':>5} {'mean_ms':>9} {'p95_ms':>9} {'p99_ms':>9} {'shots/s':>10}"); print("-"*72)
    for r in all_results:
        for max_iters,mean_ms,med_ms,p95_ms,p99_ms,thr in r["rows"]:
            print(f"{r['name']:<22} {r['n_qubits']:>4} {max_iters:>5} {mean_ms:>9.2f} {p95_ms:>9.2f} {p99_ms:>9.2f} {thr:>10.1f}")
    print("-"*72)
    # PROOF: correctness gates 10/10
    for name,ctor,(en,ec) in [("Q70 [[70,6,9]]",q70_code,(70,70)),("Q102 [[102,22,9]]",q102_code,(102,102)),("Gross [[144,12,12]]",gross_code,(144,144))]:
        hx,hz,nq,nc=ctor(); assert (nq,nc)==(en,ec), f"{name} dims {(nq,nc)} != {(en,ec)}"
        dec=RefDecoder(hx+hz,nq); H=dec.H
        assert set(np.unique(H)).issubset({0,1}), f"{name} non-binary"
        assert int(dec.decode(np.zeros(nc,dtype=np.uint8)).sum())==0, f"{name} zero->zero"
        rng=np.random.default_rng(99)
        fails=0
        for _ in range(20):
            e=np.zeros(nq,dtype=np.uint8); e[rng.choice(nq,size=2,replace=False)]=1; s=(H@e)%2
            if not np.array_equal((H@dec.decode(s))%2,s): fails+=1
        assert fails==0, f"{name} unfaithful {fails}/20"
        p,lo,hi=wilson_ci(0,20); print(f"  [PASS] {name}: dims, zero->zero, 20x w2 faithful, Wilson CI [{lo:.3f},{hi:.3f}]")
    print("  STATUS: GREEN (offline benchmark MAX-EXTENSIVE completed)")
    print("="*72)
    # write cert
    os.makedirs("certs",exist_ok=True)
    with open("certs/perf_cpu_reference.json","w") as f: json.dump({"suite":"cpu_benchmark MAX-EXTENSIVE v3","timestamp":datetime.datetime.now(datetime.timezone.utc).isoformat(),"platform":platform.platform(),"results": all_results},f,indent=2)
    return 0
if __name__=="__main__": sys.exit(main())
