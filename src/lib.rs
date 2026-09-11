// PROPRIETARY AND CONFIDENTIAL
// Copyright (c) 2026 Guillaume Lessard / qector-decoder-v3
// All Rights Reserved. NDA evaluation only. Do not redistribute.
//! qector_ionq - IonQ Superion 256 / Walking Cat production decoder wheel.
//!
//! PROPRIETARY AND CONFIDENTIAL - Guillaume Lessard / qector-decoder-v3
//! Evaluation under NDA only. Do not redistribute source.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::useless_conversion)]

pub mod gf2;
pub mod bp_osd;
pub mod pool;
pub mod ionq_superion_decoder;
pub mod bitpack;
pub mod two_stage_decoder;
pub mod space_time_decoder;
pub mod auto_decoder;
pub mod metrics;
pub mod license;

#[cfg(feature = "cuda")]
pub mod cuda_bp_osd;
#[cfg(feature = "cuda")]
pub mod cuda_runtime;
#[cfg(feature = "cascade")]
pub mod cascade_decoder;
#[cfg(feature = "cascade")]
pub mod uf_core;
#[cfg(feature = "cascade")]
pub mod fast_uf;
#[cfg(feature = "cascade")]
pub mod blossom;
#[cfg(feature = "cascade")]
pub mod sparse_blossom;
#[cfg(feature = "cascade")]
pub mod hybrid_decoder;

pub use ionq_superion_decoder::{
    DECODER_VERSION, TARGET_HARDWARE, TARGET_ARCHITECTURE, IonQSuperionDecoder,
    PyIonQSuperionDecoder, BivariateBicycle, Monomial, SecSchedule, IonQError, IonQBackend,
};
pub use bp_osd::{BPOSDDecoder, BpMethod, PyBPOSDDecoder};
pub use two_stage_decoder::{TwoStageDecoder, PyTwoStageDecoder};
pub use space_time_decoder::{SpaceTimeDecoder, PySpaceTimeDecoder};
pub use auto_decoder::{AutoDecoder, PyAutoDecoder};

use pyo3::prelude::*;

/// Python module: `import qector_ionq`
/// Symbol exported: `PyInit_qector_ionq` (required by CPython / maturin).
#[pymodule]
fn qector_ionq(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyIonQSuperionDecoder>()?;
    m.add_class::<PyBPOSDDecoder>()?;
    m.add_class::<PyTwoStageDecoder>()?;
    m.add_class::<PySpaceTimeDecoder>()?;
    m.add_class::<PyAutoDecoder>()?;
    m.add_function(wrap_pyfunction!(license::py_license_status, m)?)?;
    m.add_function(wrap_pyfunction!(py_latency_stats, m)?)?;
    m.add_function(wrap_pyfunction!(py_latency_stats_scoped, m)?)?;
    m.add_function(wrap_pyfunction!(py_latency_scopes, m)?)?;
    m.add_function(wrap_pyfunction!(py_reset_latency, m)?)?;
    m.add("DECODER_VERSION", DECODER_VERSION)?;
    m.add("TARGET_HARDWARE", TARGET_HARDWARE)?;
    m.add("TARGET_ARCHITECTURE", TARGET_ARCHITECTURE)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

#[pyfunction]
fn py_latency_stats() -> (u64, f64, f64, f64, f64) {
    metrics::latency_stats()
}

/// Workload-scoped stats: (count, mean_us, p50_us, p95_us, max_us) for one
/// scope (code name, e.g. "Q102"). Zeros when the scope is unknown.
#[pyfunction]
fn py_latency_stats_scoped(scope: &str) -> (u64, f64, f64, f64, f64) {
    metrics::latency_stats_scoped(scope)
}

/// Sorted list of workload scopes observed so far.
#[pyfunction]
fn py_latency_scopes() -> Vec<String> {
    metrics::latency_scopes()
}

#[pyfunction]
fn py_reset_latency() {
    metrics::reset_latency();
}

#[cfg(feature = "cascade")]
pub mod gnn_graph;
#[cfg(feature = "cascade")]
pub mod gnn_predecoder;
#[cfg(feature = "cascade")]
pub mod mwpm;
#[cfg(feature = "cascade")]
pub mod fusion_mwpm;
#[cfg(feature = "cascade")]
pub mod core;

#[cfg(feature = "cascade")]
pub mod gnn_layers;
#[cfg(feature = "cascade")]
pub mod safetensors_loader;
#[cfg(feature = "cascade")]
pub mod gnn_trainer;
