# Qector IonQ Decoder v1.7.7 - Frozen Restricted Package

**Title:** Qector IonQ Superion 256 / Walking Cat Decoder v1.7.7-production (frozen wheels + source + certification)
**Version:** 1.7.7-production
**Authors:** Guillaume Lessard (qector-decoder-v3)
**Target hardware:** IonQ Superion 256
**Target architecture:** Walking Cat (arXiv:2604.19481)
**Access:** Restricted (NDA / evaluation only). Proprietary - LicenseRef-Proprietary. Do not redistribute.
**Build host:** Google Colab Linux x86_64 (manylinux_2_34) + Windows amd64
**Python:** >=3.9 (validated cp311/cp312/cp313), numpy>=1.24, maturin>=1.5,<2.0, PyO3 0.22

## Abstract
Frozen, artifact-hashed production decoder for IonQ Superion 256 Walking Cat
bivariate-bicycle codes (Q70, Q102, Gross-144). Exact log-domain belief
propagation + order-0 OSD with strict `H @ correction == syndrome` gating
(typed `Unreachable`, never silent vectors), erasure-channel OSD, heterogeneous
per-qubit priors, Rayon parallel batch decode (512-shot chunks, per-shot
scoped latency), zero-copy NumPy (`PyReadonlyArray1`), GF(2) word-packed
residuals. Optional `--features cuda` / `cascade` backends (gated, CPU default).

## Codes
- Q70: l=7,m=5, 70 qubits / 70 checks, weight-6 rows
- Q102: l=51,m=1, 102/102, Appendix-C shifts A=[22,26,37,50] B=[19,28,29,35], row-weight 8, Hx=[A|B] Hz=[B^T|A^T], Hx@Hz^T=0
- Gross-144: l=12,m=6, 144/144

## Contents (95 files, SHA256SUMS.txt)
- `wheels/`: qector_ionq-1.7.7-cp313-manylinux_2_34_x86_64.whl (Linux, 404860 B, maturin 1.15, CycloneDX SBOM) + qector_ionq-1.7.7-cp311-win_amd64.whl + 1.7.6/1.7.5 cp312 win (compat)
- `src/`: full Rust source (ionq_superion_decoder, bp_osd, gf2, bitpack, pool, metrics, license, auto/two-stage/space-time decoders, cuda_*, cascade/uf/blossom, gnn/mwpm scaffolding)
- `Cargo.toml`, `Cargo.lock`, `pyproject.toml`, `python/` (smoke_test, qector_ionq.pyi, qector_thread_wrapper, gen_keys)
- Verification: `test_max_coverage.py`, `verify_q102_production.py`, `qector_ionq_math_verification.py`, `qector_ionq_cpu_benchmark.py`, `qector_ionq_performance_benchmark.py`, `qector_ionq_gpu_test.py`, `qector_ionq_hardware_bridge.py` (env-only token), `qector_ionq_z3_verification.py`
- `certs/`: 10 GREEN certs (smoke, math 24 proofs, q102 31 checks, qector_test 23 checks, performance) + performance_summary.csv
- `Colab_Linux_Build_Suite.ipynb` + `Colab_Linux_Build_Suite_FULL.ipynb` (8-cell CPU+GPU matrix), `qector_ionq_colab_source.zip` (78-file Colab source)
- `qector_ionq_v1.7.7_certification (1).zip`, `performance_summary*.csv`, `build_wheel.sh`, `reinstall_and_test.bat`, `README.md`, `LICENSE-PROPRIETARY`, `LICENSE_KEYS.md` (8 NDA keys, 7-day EVAL01-07 + dev)

## Verification (all GREEN, Linux-6.6.122, py3.13)
- Math: phi involution err 2.5e-13, LLR(1e-3)=6.907, Q102 w2 full 5151/5151, Gross w2 10296/10296, Q70 w2 2415/2415, batch==single, determinism, zero->zero, erasure all->zero
- Q102 production: Rust==Appendix-C exact, CSS orthogonality, w3/w4/w5 sampled, 500-shot random faithful x3 codes, strict_verify toggle, BPOSD smoke, SEC update/flush
- Z3 oracle: Q70/Q102/Gross SAT, schema qector-z3-oracle-v1, sha256 1cd34d50d99bf2ab...

## Performance
- CPU (2-core Colab): Q70 7110 / Q102 3022 / Gross 3941 shots/s (1T); 6431/3291/4211 (2T); single p50 ~0.32ms Q70, ~0.70ms Q102
- GPU Tesla T4 batch-2000: Q70 62826 / Q102 34402 / Gross 28643 shots/s (~8-11x CPU)
- Thread policy: Q70/Q102 single-thread (RAYON_NUM_THREADS=1) to avoid contention

## API
`IonQSuperionDecoder.q70/q102/gross`, `decode`, `decode_batch_flat`, `decode_with_erasures`, `set_qubit_priors`/`set_uniform_schedule`/`set_schedule`, `update`/`flush`, `backend`, `artifact_hash` (FNV-1a 16hex), `BPOSDDecoder`, `TwoStageDecoder`, `SpaceTimeDecoder`, `AutoDecoder`, `py_latency_stats[_scoped]`, `py_license_status`

## Reproduce (Colab T4)
Upload `qector_ionq_colab_source.zip` to /content, run FULL notebook cells 0-7, or AIO cell in description. `maturin build --release --out target/wheels`, reinstall, run smoke + verify_q102 + gpu_test.

## License / Citation
Proprietary, evaluation under NDA. Wheels only to third parties. Cite: Lessard G., Qector IonQ v1.7.7-production, IonQ Superion 256 / Walking Cat, 2026-09-09. Contact via qector-decoder-v3 maintainer.
