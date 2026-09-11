# Qector Test Certification (2026-09-09T10:33:57.234769+00:00)

status GREEN

- [x] dims q102 102/102 - 102/102
- [x] dims q70 70/70 - 70/70
- [x] dims gross 144/144 - 144/144
- [x] CSS orthogonality - Hx@Hz.T=0
- [x] single-block H shape - (51, 102)
- [x] Binary audit Rust==Appendix-C math - 102x102 exact
- [x] Q102 row-weight 8 - nnz=816
- [x] Q70 binary + non-empty - (70, 70)
- [x] Gross binary + non-empty - (144, 144)
- [x] Q70 w1 70/70 - 0.09s
- [x] Q102 w1 102/102 - 0.18s
- [x] Gross w1 144/144 - 0.28s
- [x] Q102 w2 full 5151 - 5151/5151
- [x] Q102 w3 sampled 1500 - 1500/1500
- [x] Q102 w4 sampled 800 - 800/800
- [x] Batch==single determinism - B=8
- [x] Determinism x3 - identical
- [x] Length-mismatch raises - 101 vs 102
- [x] Zero->zero - clean
- [x] Erasure all->zero - fallback
- [x] Heterogeneous priors - heterogeneous_qubit_priors
- [x] Streaming update/flush - SEC
- [x] BPOSD smoke - 102-bit
