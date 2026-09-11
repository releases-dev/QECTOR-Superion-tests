//! =============================================================================
//! PROPRIETARY AND CONFIDENTIAL
//! Copyright (c) 2026 Guillaume Lessard / qector-decoder-v3
//! All Rights Reserved.
//!
//! This source code and the algorithms it implements are proprietary intellectual
//! property. Unauthorized copying, distribution, reverse engineering, or use
//! outside of an executed commercial license or NDA is strictly prohibited.
//!
//! Evaluation copies may be provided to IonQ engineering solely for integration
//! testing under NDA. Do not forward, commit to public repositories, or embed
//! in derivative products without written authorization.
//!
//! Contact: commercial licensing via the qector-decoder-v3 maintainer.
//! =============================================================================
//!
//! IonQ Superion 256 / Walking Cat - Production Decoder
//!
//! Version: 1.7.8-production
//! Target: IonQ Superion 256 + Walking Cat (arXiv:2604.19481)
//! Author: Guillaume Lessard - qector-decoder-v3

use crate::bp_osd::{BPOSDDecoder, BpMethod};
use crate::gf2;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::collections::VecDeque;

#[cfg(feature = "cuda")]
use crate::cuda_bp_osd::CUDABpOsdDecoder;
#[cfg(feature = "cascade")]
use crate::cascade_decoder::HybridCascadeDecoder;

// =============================================================================
// Metadata
// =============================================================================

pub const DECODER_VERSION: &str = "1.7.8-production";
pub const TARGET_HARDWARE: &str = "IonQ Superion 256";
pub const TARGET_ARCHITECTURE: &str = "Walking Cat (arXiv:2604.19481)";

// =============================================================================
// 1. Bivariate Bicycle constructors
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Monomial {
    pub a: i32,
    pub b: i32,
}

impl Monomial {
    #[inline]
    pub const fn new(a: i32, b: i32) -> Self {
        Self { a, b }
    }
}

#[derive(Clone, Debug)]
pub struct BivariateBicycle {
    pub l: usize,
    pub m: usize,
    pub a_monomials: Vec<Monomial>,
    pub b_monomials: Vec<Monomial>,
    pub hx_checks: Vec<Vec<u32>>,
    pub hz_checks: Vec<Vec<u32>>,
    pub n_qubits: usize,
    pub n_checks: usize,
}

impl BivariateBicycle {
    pub fn from_polynomials(
        l: usize,
        m: usize,
        a_mons: &[Monomial],
        b_mons: &[Monomial],
    ) -> Result<Self, String> {
        if l == 0 || m == 0 {
            return Err("l and m must be positive".into());
        }
        if a_mons.is_empty() || b_mons.is_empty() {
            return Err("A and B must each contain at least one monomial".into());
        }
        let lm = l * m;
        let n_qubits = 2 * lm;

        let idx = |i: i32, j: i32| -> usize {
            let ii = ((i.rem_euclid(l as i32)) as usize) % l;
            let jj = ((j.rem_euclid(m as i32)) as usize) % m;
            ii * m + jj
        };

        let mut hx_checks = Vec::with_capacity(lm);
        for r in 0..l as i32 {
            for c in 0..m as i32 {
                let mut support = Vec::with_capacity(a_mons.len() + b_mons.len());
                for mon in a_mons {
                    support.push(idx(r + mon.a, c + mon.b) as u32);
                }
                for mon in b_mons {
                    support.push((lm + idx(r + mon.a, c + mon.b)) as u32);
                }
                support.sort_unstable();
                support.dedup();
                hx_checks.push(support);
            }
        }

        let mut hz_checks = Vec::with_capacity(lm);
        for r in 0..l as i32 {
            for c in 0..m as i32 {
                let mut support = Vec::with_capacity(a_mons.len() + b_mons.len());
                for mon in b_mons {
                    support.push(idx(r - mon.a, c - mon.b) as u32);
                }
                for mon in a_mons {
                    support.push((lm + idx(r - mon.a, c - mon.b)) as u32);
                }
                support.sort_unstable();
                support.dedup();
                hz_checks.push(support);
            }
        }

        Ok(Self {
            l,
            m,
            a_monomials: a_mons.to_vec(),
            b_monomials: b_mons.to_vec(),
            hx_checks,
            hz_checks,
            n_qubits,
            n_checks: 2 * lm,
        })
    }

    pub fn check_to_qubits(&self) -> Vec<Vec<u32>> {
        let mut all = self.hx_checks.clone();
        all.extend_from_slice(&self.hz_checks);
        all
    }

    pub fn check_types(&self) -> Vec<bool> {
        let mut t = vec![false; self.hx_checks.len()];
        t.extend(std::iter::repeat_n(true, self.hz_checks.len()));
        t
    }
}

pub fn gross_code() -> Result<BivariateBicycle, String> {
    let a = [
        Monomial::new(3, 0),
        Monomial::new(0, 1),
        Monomial::new(0, 2),
    ];
    let b = [
        Monomial::new(0, 3),
        Monomial::new(1, 0),
        Monomial::new(2, 0),
    ];
    BivariateBicycle::from_polynomials(12, 6, &a, &b)
}

pub fn q70_code() -> Result<BivariateBicycle, String> {
    let a = [
        Monomial::new(0, 0),
        Monomial::new(1, 0),
        Monomial::new(0, 2),
    ];
    let b = [
        Monomial::new(0, 0),
        Monomial::new(0, 1),
        Monomial::new(2, 0),
    ];
    BivariateBicycle::from_polynomials(7, 5, &a, &b)
}

pub fn q102_code() -> Result<BivariateBicycle, String> {
    let a = [
        Monomial::new(22, 0),
        Monomial::new(26, 0),
        Monomial::new(37, 0),
        Monomial::new(50, 0),
    ];
    let b = [
        Monomial::new(19, 0),
        Monomial::new(28, 0),
        Monomial::new(29, 0),
        Monomial::new(35, 0),
    ];
    BivariateBicycle::from_polynomials(51, 1, &a, &b)
}

// =============================================================================
// 2. SEC schedule
// =============================================================================

#[derive(Clone, Debug)]
pub struct SecSchedule {
    pub p_meas: Vec<f64>,
    pub p_data: Vec<f64>,
    pub label: String,
}

impl SecSchedule {
    pub fn uniform(n_checks: usize, n_qubits: usize, p: f64) -> Self {
        let p = p.clamp(1e-12, 1.0 - 1e-12);
        Self {
            p_meas: vec![p; n_checks],
            p_data: vec![p; n_qubits],
            label: "uniform".into(),
        }
    }

    pub fn validate(&self, n_checks: usize, n_qubits: usize) -> Result<(), String> {
        if self.p_meas.len() != n_checks {
            return Err(format!(
                "p_meas len {} != n_checks {}",
                self.p_meas.len(),
                n_checks
            ));
        }
        if self.p_data.len() != n_qubits {
            return Err(format!(
                "p_data len {} != n_qubits {}",
                self.p_data.len(),
                n_qubits
            ));
        }
        for (i, &p) in self.p_meas.iter().enumerate() {
            if !(0.0..1.0).contains(&p) || !p.is_finite() {
                return Err(format!("p_meas[{i}]={p} invalid"));
            }
        }
        for (i, &p) in self.p_data.iter().enumerate() {
            if !(0.0..1.0).contains(&p) || !p.is_finite() {
                return Err(format!("p_data[{i}]={p} invalid"));
            }
        }
        Ok(())
    }

    pub fn mean_p_data(&self) -> f64 {
        if self.p_data.is_empty() {
            return 1e-3;
        }
        let s: f64 = self.p_data.iter().sum();
        (s / self.p_data.len() as f64).clamp(1e-12, 1.0 - 1e-12)
    }
}

// =============================================================================
// 3. Erasure-channel OSD (FIXED: full-length order)
// =============================================================================

fn erasure_channel_osd(
    check_to_qubits: &[Vec<u32>],
    n_qubits: usize,
    syndrome: &[u8],
    erasures: &[u8],
    beliefs: &[f64],
) -> Vec<u8> {
    debug_assert_eq!(erasures.len(), n_qubits);
    debug_assert_eq!(beliefs.len(), n_qubits);

    let mut active: Vec<usize> = (0..n_qubits).filter(|&q| erasures[q] == 0).collect();
    active.sort_unstable_by(|&a, &b| {
        beliefs[a]
            .abs()
            .total_cmp(&beliefs[b].abs())
            .then_with(|| a.cmp(&b))
    });

    let mut order = Vec::with_capacity(n_qubits);
    // Erased qubits should be preferred as pivots to absorb errors, so put them first.
    for q in 0..n_qubits {
        if erasures[q] != 0 {
            order.push(q);
        }
    }
    // Then active qubits, sorted by reliability (least reliable first to also act as pivots if needed).
    order.extend_from_slice(&active);
    debug_assert_eq!(order.len(), n_qubits);

    let mut free = vec![0u8; n_qubits];
    for &q in &active {
        free[q] = if beliefs[q] < 0.0 { 1 } else { 0 };
    }
    // Erased qubits have no belief, they can default to 0 as free variables (if they don't become pivots)

    let result = gf2::solve_selected_with_free(
        check_to_qubits,
        n_qubits,
        syndrome,
        &order,
        &free,
    );

    let mut out = match result {
        Some(x) => x,
        None => {
            let mut fallback = vec![0u8; n_qubits];
            for &q in &active {
                fallback[q] = if beliefs[q] < 0.0 { 1 } else { 0 };
            }
            fallback
        }
    };

    for (q, &e) in erasures.iter().enumerate() {
        if e != 0 {
            out[q] = 0;
        }
    }
    out
}

fn is_faithful(
    check_to_qubits: &[Vec<u32>],
    correction: &[u8],
    syndrome: &[u8],
    erasures: Option<&[u8]>,
) -> bool {
    if syndrome.len() != check_to_qubits.len() {
        return false;
    }
    // Lengths are validated by every caller; asserted (not branched) so the
    // inner parity fold stays flat and branch-free (auto-vectorized).
    debug_assert!(correction.len() >= check_to_qubits.iter().flatten().map(|&q| q as usize + 1).max().unwrap_or(0));
    if let Some(era) = erasures {
        debug_assert_eq!(era.len(), correction.len());
    }
    match erasures {
        None => {
            for (ci, qs) in check_to_qubits.iter().enumerate() {
                let mut p = 0u8;
                for &q in qs {
                    p ^= correction[q as usize] & 1;
                }
                if p != (syndrome[ci] & 1) {
                    return false;
                }
            }
            true
        }
        Some(era) => {
            for (ci, qs) in check_to_qubits.iter().enumerate() {
                let mut p = 0u8;
                for &q in qs {
                    let qi = q as usize;
                    // Masked parity: erased qubits contribute 0. Single
                    // predictable branch on the mask byte (not data-dependent
                    // on the correction), loop stays contiguous.
                    if era[qi] == 0 {
                        p ^= correction[qi] & 1;
                    }
                }
                if p != (syndrome[ci] & 1) {
                    return false;
                }
            }
            true
        }
    }
}

// =============================================================================
// 4. Errors
// =============================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IonQError {
    SyndromeLength { expected: usize, got: usize },
    ErasureLength { expected: usize, got: usize },
    Schedule(String),
    Construction(String),
    Unreachable,
}

impl std::fmt::Display for IonQError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SyndromeLength { expected, got } => {
                write!(f, "syndrome length {got} != n_checks {expected}")
            }
            Self::ErasureLength { expected, got } => {
                write!(f, "erasure mask length {got} != n_qubits {expected}")
            }
            Self::Schedule(s) => write!(f, "schedule: {s}"),
            Self::Construction(s) => write!(f, "construction: {s}"),
            Self::Unreachable => write!(f, "syndrome unreachable under H"),
        }
    }
}
impl std::error::Error for IonQError {}

// =============================================================================
// 5. Production decoder
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IonQBackend {
    CpuBpOsd,
    #[cfg(feature = "cuda")]
    CudaBpOsd,
    #[cfg(feature = "cascade")]
    CascadeBpOsd,
}

pub struct IonQSuperionDecoder {
    check_to_qubits: Vec<Vec<u32>>,
    n_qubits: usize,
    n_checks: usize,
    bp: BPOSDDecoder,
    schedule: SecSchedule,
    history: VecDeque<Vec<u8>>,
    history_cap: usize,
    strict_verify: bool,
    code_name: String,
    backend: IonQBackend,
    #[cfg(feature = "cuda")]
    cuda_bp: Option<CUDABpOsdDecoder>,
    #[cfg(feature = "cascade")]
    cascade: Option<HybridCascadeDecoder>,
}

impl IonQSuperionDecoder {
    pub fn from_checks(
        check_to_qubits: Vec<Vec<u32>>,
        n_qubits: Option<usize>,
        error_rate: f64,
        code_name: &str,
    ) -> Result<Self, IonQError> {
        if check_to_qubits.is_empty() {
            return Err(IonQError::Construction(
                "check_to_qubits must be non-empty".into(),
            ));
        }
        if check_to_qubits.iter().any(|c| c.is_empty()) {
            return Err(IonQError::Construction(
                "every check must be non-empty".into(),
            ));
        }
        let n_checks = check_to_qubits.len();
        let nq = n_qubits.unwrap_or_else(|| {
            check_to_qubits
                .iter()
                .flatten()
                .copied()
                .max()
                .map(|m| m as usize + 1)
                .unwrap_or(0)
        });
        let p = error_rate.clamp(1e-12, 1.0 - 1e-12);
        let bp = BPOSDDecoder::new_with_options(
            check_to_qubits.clone(),
            Some(nq),
            p,
            BpMethod::Exact,
            0,
        )
        .map_err(IonQError::Construction)?;
        Ok(Self {
            check_to_qubits,
            n_qubits: nq,
            n_checks,
            bp,
            schedule: SecSchedule::uniform(n_checks, nq, p),
            history: VecDeque::with_capacity(64),
            history_cap: 64,
            strict_verify: true,
            code_name: code_name.into(),
            backend: IonQBackend::CpuBpOsd,
            #[cfg(feature = "cuda")]
            cuda_bp: None,
            #[cfg(feature = "cascade")]
            cascade: None,
        })
    }

    pub fn from_bb(bb: &BivariateBicycle, error_rate: f64, name: &str) -> Result<Self, IonQError> {
        Self::from_checks(bb.check_to_qubits(), Some(bb.n_qubits), error_rate, name)
    }

    pub fn q70(error_rate: f64) -> Result<Self, IonQError> {
        let bb = q70_code().map_err(IonQError::Construction)?;
        Self::from_bb(&bb, error_rate, "Q70")
    }

    pub fn q102(error_rate: f64) -> Result<Self, IonQError> {
        let bb = q102_code().map_err(IonQError::Construction)?;
        Self::from_bb(&bb, error_rate, "Q102")
    }

    pub fn gross(error_rate: f64) -> Result<Self, IonQError> {
        let bb = gross_code().map_err(IonQError::Construction)?;
        Self::from_bb(&bb, error_rate, "Gross")
    }

    pub fn set_strict_verify(&mut self, on: bool) {
        self.strict_verify = on;
    }

    pub fn backend(&self) -> IonQBackend {
        self.backend
    }

    #[cfg(feature = "gnn")]
    pub fn gnn_edge_weights(
        &self,
        gnn: &crate::gnn_predecoder::GNNPredecoder,
        syndrome: &[u8],
    ) -> Result<Vec<f64>, IonQError> {
        if syndrome.len() != self.n_checks {
            return Err(IonQError::SyndromeLength {
                expected: self.n_checks,
                got: syndrome.len(),
            });
        }
        let graph = crate::gnn_graph::DetectorGraph::from_check_to_qubits(
            &self.check_to_qubits,
            syndrome,
            None,
            None,
            Some(self.schedule.p_data.clone()),
            Some(self.n_qubits),
        );
        Ok(gnn.forward(&graph))
    }

    #[cfg(feature = "grpc")]
    pub fn start_grpc(host: &str, port: u16) -> Result<crate::grpc_server::PyGrpcServerHandle, String> {
        crate::grpc_server::start_grpc_server(host, port)
            .map_err(|e| e.to_string())
    }

    #[cfg(feature = "cuda")]
    pub fn prefer_cuda(&mut self) -> Result<bool, IonQError> {
        match CUDABpOsdDecoder::new(
            self.check_to_qubits.clone(),
            Some(self.n_qubits),
            self.schedule.mean_p_data(),
        ) {
            Ok(cuda) => {
                self.cuda_bp = Some(cuda);
                self.backend = IonQBackend::CudaBpOsd;
                Ok(true)
            }
            Err(e) => Err(IonQError::Construction(format!("CUDA BP-OSD: {e}"))),
        }
    }

    #[cfg(feature = "cascade")]
    pub fn prefer_cascade(&mut self) -> Result<(), IonQError> {
        let c = HybridCascadeDecoder::new_with_bposd(
            self.check_to_qubits.clone(),
            Some(self.n_qubits),
            self.schedule.mean_p_data(),
            None,
        );
        self.cascade = Some(c.map_err(|e| IonQError::Construction(e))?);
        self.backend = IonQBackend::CascadeBpOsd;
        Ok(())
    }

    pub fn decode_batch_flat(&self, syndromes: &[u8], batch_size: usize) -> Result<Vec<u8>, IonQError> {
        if batch_size == 0 {
            return Ok(Vec::new());
        }
        let expect = batch_size
            .checked_mul(self.n_checks)
            .ok_or_else(|| IonQError::Construction("batch dims overflow".into()))?;
        if syndromes.len() != expect {
            return Err(IonQError::SyndromeLength {
                expected: batch_size * self.n_checks,
                got: syndromes.len(),
            });
        }

        #[cfg(feature = "cuda")]
        if self.backend == IonQBackend::CudaBpOsd {
            if let Some(ref cuda) = self.cuda_bp {
                return cuda
                    .batch_decode(syndromes, batch_size)
                    .map_err(|e| IonQError::Construction(e));
            }
        }

        #[cfg(feature = "cascade")]
        if self.backend == IonQBackend::CascadeBpOsd {
            if let Some(ref cascade) = self.cascade {
                return Ok(cascade.batch_decode(syndromes, batch_size));
            }
        }

        let nq = self.n_qubits;
        let nc = self.n_checks;
        let strict = self.strict_verify;
        let scope = self.code_name.clone();
        // v1.7.8: 512-shot chunks on the pinned pool (todo.md ?3). One BP
        // workspace per chunk (zero per-shot alloc), per-shot latency recorded
        // into the workload-scoped ring, bulletproof Hc == s gate per shot.
        let stride = nc
            .checked_mul(crate::pool::BATCH_CHUNK_SHOTS)
            .ok_or_else(|| IonQError::Construction("chunk stride overflow".into()))?;
        let parts: Result<Vec<(usize, Vec<u8>)>, IonQError> =
            crate::pool::batch_pool().install(|| {
                use rayon::prelude::*;
                syndromes
                    .par_chunks(stride)
                    .enumerate()
                    .map(|(ci, chunk)| {
                        let ns = chunk.len() / nc;
                        let cap = ns
                            .checked_mul(nq)
                            .ok_or_else(|| IonQError::Construction("chunk cap overflow".into()))?;
                        let mut ws = crate::bp_osd::DecoderWorkspace::new();
                        let mut out = Vec::with_capacity(cap);
                        for s in 0..ns {
                            let lo = s.wrapping_mul(nc);
                            let syn = &chunk[lo..lo.wrapping_add(nc)];
                            let t0 = std::time::Instant::now();
                            let beliefs = self.bp.bp_decode_ws(syn, 80, &mut ws).to_vec();
                            let corr = self.bp.osd_stage(&beliefs, syn);
                            crate::metrics::record_latency_scoped(
                                &scope,
                                t0.elapsed().as_secs_f64() * 1e6,
                            );
                            if strict && !self.bp.faithful(&corr, syn) {
                                return Err(IonQError::Unreachable);
                            }
                            out.extend_from_slice(&corr);
                        }
                        Ok((ci, out))
                    })
                    .collect()
            });
        let mut parts = parts?;
        parts.sort_by_key(|(ci, _)| *ci);
        let total = batch_size
            .checked_mul(nq)
            .ok_or_else(|| IonQError::Construction("total overflow".into()))?;
        let mut out = vec![0u8; total];
        let mut off = 0usize;
        for (_, chunk_out) in parts {
            let end = off.wrapping_add(chunk_out.len());
            out[off..end].copy_from_slice(&chunk_out);
            off = end;
        }
        debug_assert_eq!(off, total);
        Ok(out)
    }

    pub fn set_schedule(&mut self, schedule: SecSchedule) -> Result<(), IonQError> {
        schedule
            .validate(self.n_checks, self.n_qubits)
            .map_err(IonQError::Schedule)?;
        let mean_p = schedule.mean_p_data();
        self.bp = BPOSDDecoder::new_with_options(
            self.check_to_qubits.clone(),
            Some(self.n_qubits),
            mean_p,
            BpMethod::Exact,
            0,
        )
        .map_err(IonQError::Construction)?;
        self.schedule = schedule;
        Ok(())
    }

    pub fn schedule(&self) -> &SecSchedule {
        &self.schedule
    }

    pub fn set_qubit_priors(&mut self, p_data: Vec<f64>) -> Result<(), IonQError> {
        if p_data.len() != self.n_qubits {
            return Err(IonQError::Schedule(format!(
                "p_data len {} != n_qubits {}",
                p_data.len(),
                self.n_qubits
            )));
        }
        let mean = {
            let s: f64 = p_data.iter().sum();
            (s / p_data.len() as f64).clamp(1e-12, 1.0 - 1e-12)
        };
        let sched = SecSchedule {
            p_meas: vec![mean; self.n_checks],
            p_data: p_data.clone(),
            label: "heterogeneous_qubit_priors".into(),
        };
        self.set_schedule(sched)?;
        self.bp
            .set_qubit_priors(p_data)
            .map_err(IonQError::Construction)?;
        Ok(())
    }

    pub fn decode(&self, syndrome: &[u8]) -> Result<Vec<u8>, IonQError> {
        if syndrome.len() != self.n_checks {
            return Err(IonQError::SyndromeLength {
                expected: self.n_checks,
                got: syndrome.len(),
            });
        }
        let t0 = std::time::Instant::now();
        let corr = self.bp.decode(syndrome).map_err(IonQError::Construction)?;
        crate::metrics::record_latency_scoped(&self.code_name, t0.elapsed().as_secs_f64() * 1e6);
        // Defense in depth: bp.decode already gates Hc == s; re-check here so
        // EVERY correction leaving this facade is verified (bulletproof rule).
        if self.strict_verify && !self.bp.faithful(&corr, syndrome) {
            return Err(IonQError::Unreachable);
        }
        Ok(corr)
    }

    pub fn decode_with_erasures(
        &self,
        syndrome: &[u8],
        erasures: &[u8],
    ) -> Result<Vec<u8>, IonQError> {
        if syndrome.len() != self.n_checks {
            return Err(IonQError::SyndromeLength {
                expected: self.n_checks,
                got: syndrome.len(),
            });
        }
        if erasures.len() != self.n_qubits {
            return Err(IonQError::ErasureLength {
                expected: self.n_qubits,
                got: erasures.len(),
            });
        }
        let t0 = std::time::Instant::now();
        let beliefs = self.bp.bp_decode(syndrome, 80);
        let corr = erasure_channel_osd(
            &self.check_to_qubits,
            self.n_qubits,
            syndrome,
            erasures,
            &beliefs,
        );
        crate::metrics::record_latency_scoped(&self.code_name, t0.elapsed().as_secs_f64() * 1e6);
        // BULLETPROOF RULE: the masked system H_masked @ c == s is verified.
        // Strict mode (default) returns typed Unreachable instead of a silent
        // unfaithful vector; best-effort callers set_strict_verify(false).
        if self.strict_verify && !is_faithful(&self.check_to_qubits, &corr, syndrome, Some(erasures)) {
            return Err(IonQError::Unreachable);
        }
        Ok(corr)
    }

    /// Deterministic artifact fingerprint (FNV-1a hex) over the decode
    /// artifact: version + code id + dims + every parity-check adjacency
    /// entry. Certification measurements embed this hash, binding each
    /// result to the exact matrices that produced it (artifact-hashed).
    pub fn artifact_hash(&self) -> String {
        let mut h: u64 = 14695981039346656037;
        let mut mix = |bytes: &[u8]| {
            for &b in bytes {
                h ^= b as u64;
                h = h.wrapping_mul(1099511628211);
            }
        };
        mix(DECODER_VERSION.as_bytes());
        mix(self.code_name.as_bytes());
        mix(&(self.n_qubits as u64).to_le_bytes());
        mix(&(self.n_checks as u64).to_le_bytes());
        for qs in &self.check_to_qubits {
            for &q in qs {
                mix(&q.to_le_bytes());
            }
            mix(&[0xFF]);
        }
        format!("{h:016x}")
    }

    pub fn decode_batch(&self, syndromes: &[Vec<u8>]) -> Result<Vec<Vec<u8>>, IonQError> {
        let mut out = Vec::with_capacity(syndromes.len());
        for (i, s) in syndromes.iter().enumerate() {
            if s.len() != self.n_checks {
                return Err(IonQError::SyndromeLength {
                    expected: self.n_checks,
                    got: s.len(),
                });
            }
            let corr = self.bp.decode(s).map_err(IonQError::Construction)?;
            if self.strict_verify && !is_faithful(&self.check_to_qubits, &corr, s, None) {
                return Err(IonQError::Unreachable);
            }
            let _ = i;
            out.push(corr);
        }
        Ok(out)
    }

    pub fn decode_timed(&self, syndrome: &[u8], max_latency_ms: f64) -> Result<Vec<u8>, IonQError> {
        if syndrome.len() != self.n_checks {
            return Err(IonQError::SyndromeLength {
                expected: self.n_checks,
                got: syndrome.len(),
            });
        }
        let corr = if max_latency_ms > 0.0 {
            self.bp.decode_timed(syndrome, max_latency_ms).map_err(IonQError::Construction)?
        } else {
            self.bp.decode(syndrome).map_err(IonQError::Construction)?
        };
        if self.strict_verify && !is_faithful(&self.check_to_qubits, &corr, syndrome, None) {
            return Err(IonQError::Unreachable);
        }
        Ok(corr)
    }

    pub fn update(&mut self, round_syndrome: &[u8]) -> Result<Vec<u8>, IonQError> {
        if round_syndrome.len() != self.n_checks {
            return Err(IonQError::SyndromeLength {
                expected: self.n_checks,
                got: round_syndrome.len(),
            });
        }
        if self.history.len() == self.history_cap {
            self.history.pop_front();
        }
        self.history.push_back(round_syndrome.to_vec());
        self.decode(round_syndrome)
    }

    pub fn flush(&mut self) {
        self.history.clear();
    }

    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }
    pub fn n_checks(&self) -> usize {
        self.n_checks
    }
    pub fn history_len(&self) -> usize {
        self.history.len()
    }
    pub fn code_name(&self) -> &str {
        &self.code_name
    }
    pub fn version() -> &'static str {
        DECODER_VERSION
    }
    pub fn check_to_qubits(&self) -> &[Vec<u32>] {
        &self.check_to_qubits
    }
}

// =============================================================================
// 6. PyO3
// =============================================================================

#[pyclass(name = "IonQSuperionDecoder")]
pub struct PyIonQSuperionDecoder {
    inner: IonQSuperionDecoder,
}

#[pymethods]
impl PyIonQSuperionDecoder {
    #[new]
    #[pyo3(signature = (code="q70", error_rate=0.001))]
    fn new(code: &str, error_rate: f64) -> PyResult<Self> {
        let inner = match code.to_ascii_lowercase().as_str() {
            "q70" | "70" => IonQSuperionDecoder::q70(error_rate),
            "q102" | "102" => IonQSuperionDecoder::q102(error_rate),
            "gross" | "144" => IonQSuperionDecoder::gross(error_rate),
            other => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "unknown code '{other}': expected q70 | q102 | gross"
                )));
            }
        }
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (check_to_qubits, n_qubits=None, error_rate=0.001, name="custom"))]
    fn from_checks(
        check_to_qubits: Vec<Vec<u32>>,
        n_qubits: Option<usize>,
        error_rate: f64,
        name: &str,
    ) -> PyResult<Self> {
        let inner =
            IonQSuperionDecoder::from_checks(check_to_qubits, n_qubits, error_rate, name)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(Self { inner })
    }


    #[staticmethod]
    #[pyo3(signature = (error_rate=0.001))]
    fn q102(error_rate: f64) -> PyResult<Self> {
        let inner = IonQSuperionDecoder::q102(error_rate)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (error_rate=0.001))]
    fn gross(error_rate: f64) -> PyResult<Self> {
        let inner = IonQSuperionDecoder::gross(error_rate)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (error_rate=0.001))]
    fn q70(error_rate: f64) -> PyResult<Self> {
        let inner = IonQSuperionDecoder::q70(error_rate)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(Self { inner })
    }

    fn decode<'py>(
        &self,
        py: Python<'py>,
        syndrome: PyReadonlyArray1<u8>,
    ) -> PyResult<Bound<'py, PyArray1<u8>>> {
        let s = syndrome
            .as_slice()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{e}")))?;
        let c = self
            .inner
            .decode(s)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(c.into_pyarray_bound(py))
    }

    fn decode_with_erasures<'py>(
        &self,
        py: Python<'py>,
        syndrome: PyReadonlyArray1<u8>,
        erasures: PyReadonlyArray1<u8>,
    ) -> PyResult<Bound<'py, PyArray1<u8>>> {
        let s = syndrome
            .as_slice()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{e}")))?;
        let e = erasures
            .as_slice()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{e}")))?;
        let c = self
            .inner
            .decode_with_erasures(s, e)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(c.into_pyarray_bound(py))
    }

    fn update<'py>(
        &mut self,
        py: Python<'py>,
        round_syndrome: PyReadonlyArray1<u8>,
    ) -> PyResult<Bound<'py, PyArray1<u8>>> {
        let s = round_syndrome
            .as_slice()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{e}")))?;
        let c = self
            .inner
            .update(s)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(c.into_pyarray_bound(py))
    }

    fn set_uniform_schedule(&mut self, p: f64) -> PyResult<()> {
        let sched = SecSchedule::uniform(self.inner.n_checks, self.inner.n_qubits, p);
        self.inner
            .set_schedule(sched)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    fn set_schedule(&mut self, p_meas: Vec<f64>, p_data: Vec<f64>, label: &str) -> PyResult<()> {
        let sched = SecSchedule {
            p_meas,
            p_data,
            label: label.into(),
        };
        self.inner
            .set_schedule(sched)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    fn set_qubit_priors(&mut self, p_data: PyReadonlyArray1<f64>) -> PyResult<()> {
        let slice = p_data.as_slice().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid p_data: {e}"))
        })?;
        self.inner
            .set_qubit_priors(slice.to_vec())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    #[pyo3(signature = (syndromes, batch_size))]
    fn decode_batch_flat<'py>(
        &self,
        py: Python<'py>,
        syndromes: PyReadonlyArray1<u8>,
        batch_size: usize,
    ) -> PyResult<Bound<'py, PyArray1<u8>>> {
        let s = syndromes.as_slice().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid syndromes: {e}"))
        })?;
        let c = self
            .inner
            .decode_batch_flat(s, batch_size)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(c.into_pyarray_bound(py))
    }

    #[cfg(feature = "cuda")]
    fn prefer_cuda(&mut self) -> PyResult<bool> {
        self.inner
            .prefer_cuda()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    #[cfg(feature = "cascade")]
    fn prefer_cascade(&mut self) -> PyResult<()> {
        self.inner
            .prefer_cascade()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    fn backend(&self) -> String {
        format!("{:?}", self.inner.backend())
    }

    #[cfg(feature = "gnn")]
    fn gnn_edge_weights(
        &self,
        gnn: &crate::gnn_predecoder::PyGNNPredecoder,
        syndrome: PyReadonlyArray1<u8>,
    ) -> PyResult<Vec<f64>> {
        let s = syndrome.as_slice().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid syndrome: {e}"))
        })?;
        self.inner
            .gnn_edge_weights(&gnn.inner, s)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    #[cfg(feature = "grpc")]
    #[staticmethod]
    fn start_grpc(host: &str, port: u16) -> PyResult<crate::grpc_server::PyGrpcServerHandle> {
        crate::grpc_server::start_grpc_server(host, port).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
        })
    }

    fn set_strict_verify(&mut self, on: bool) {
        self.inner.set_strict_verify(on);
    }

    fn flush(&mut self) {
        self.inner.flush();
    }

    #[getter]
    fn n_qubits(&self) -> usize {
        self.inner.n_qubits()
    }
    #[getter]
    fn n_checks(&self) -> usize {
        self.inner.n_checks()
    }
    #[getter]
    fn history_len(&self) -> usize {
        self.inner.history_len()
    }
    #[getter]
    fn schedule_label(&self) -> String {
        self.inner.schedule().label.clone()
    }
    #[getter]
    fn code_name(&self) -> String {
        self.inner.code_name().to_string()
    }
    #[getter]
    fn artifact_hash(&self) -> String {
        self.inner.artifact_hash()
    }
    #[getter]
    fn version(&self) -> &'static str {
        DECODER_VERSION
    }
    #[getter]
    fn target_hardware(&self) -> &'static str {
        TARGET_HARDWARE
    }
    #[getter]
    fn check_to_qubits(&self) -> Vec<Vec<u32>> {
        self.inner.check_to_qubits().to_vec()
    }
}