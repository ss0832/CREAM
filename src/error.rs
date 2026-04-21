//! Structured error type for all CREAM operations.
//!
//! Uses [`thiserror`] for derive-based `Display` / `Error` implementations.
//! All `unwrap()` calls are forbidden; callers receive `Result<_, CreamError>`.

use thiserror::Error;

/// The top-level error enum for the CREAM library.
///
/// Variants cover GPU lifecycle errors, out-of-memory, invalid user input,
/// EAM file parsing failures, shader compilation errors, Cell List errors,
/// and GPU buffer map errors.
#[derive(Error, Debug)]
#[allow(missing_docs)]
pub enum CreamError {
    /// The GPU device was lost or no suitable adapter was found.
    #[error("GPU device lost: {0}")]
    DeviceLost(String),

    /// Requested GPU allocation exceeds available or allowed memory.
    #[error("Out of GPU memory: requested {requested_mb} MB")]
    OutOfMemory { requested_mb: u64 },

    /// User-supplied data is logically invalid (e.g., mismatched lengths).
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Failure while parsing a `.eam.alloy` (or similar) potential file.
    #[error("EAM parse error at line {line}: {message}")]
    ParseError { line: usize, message: String },

    /// WGSL shader compilation or naga_oil composition failed.
    #[error("Shader compilation failed: {0}")]
    ShaderError(String),

    /// Cell List construction error.
    #[error("Cell list error: {0}")]
    CellListError(String),

    /// GPU buffer mapping (readback) failed.
    #[error("Buffer map error: {0}")]
    BufferMapError(String),
}

// ── PyO3 conversion (compiled only with the `python` feature) ──────────────
#[cfg(feature = "python")]
impl From<CreamError> for pyo3::PyErr {
    fn from(e: CreamError) -> pyo3::PyErr {
        pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
    }
}
