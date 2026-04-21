//! # CREAM — Compute-shader Rust EAM Atomistics
//!
//! GPU-accelerated molecular dynamics engine based on the Embedded Atom Method
//! (EAM) interatomic potential. Supports multi-element alloys, native GPU
//! (Vulkan/Metal/DX12) and WebGPU (browser) via the same Rust code.
//!
//! ## Quick start
//!
//! ```no_run
//! use cream::{
//!     engine::{ComputeEngine, ComputeResult},
//!     potential::{eam::EamPotential, NeighborStrategy},
//! };
//! use std::path::Path;
//!
//! # async fn demo() -> Result<(), Box<dyn std::error::Error>> {
//! let mut engine = ComputeEngine::new(NeighborStrategy::AllPairs).await?;
//! let potential  = EamPotential::from_file(Path::new("Cu.eam.alloy"))?;
//!
//! // positions: (x, y, z, 0) in Å — w component is GPU padding
//! let positions  = vec![[0.0f32, 0.0, 0.0, 0.0], [1.8, 1.8, 0.0, 0.0]];
//! let atom_types = vec![0u32, 0u32]; // both Cu
//!
//! // Orthorhombic cell [Å] — rows are lattice vectors a, b, c (f32 for GPU)
//! let cell: Option<[[f32; 3]; 3]> =
//!     Some([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]);
//! // For the f64 reference implementation use cream::ortho_cell(Lx, Ly, Lz).
//!
//! let result = engine.compute_sync(&positions, &atom_types, cell, &potential)?;
//! println!("Energy: {:.6} eV", result.energy);
//! # Ok(())
//! # }
//! ```

#![deny(unsafe_code)]
#![warn(missing_docs)]
// ── Stylistic-lint allow list ────────────────────────────────────────────────
//
// These lints are stylistic preferences that `-D warnings` turns into hard
// errors on newer clippy releases.  Disabling them crate-wide keeps the
// codebase portable across rust-toolchain versions without masking real bugs:
//
// * `type_complexity`  — tuple-heavy accumulator signatures in the CPU engine
//   and PyO3 return types;  `type` aliases are possible but reduce grep-ability
//   of the function signatures.
// * `too_many_arguments`— internal constructor `create_cell_list_buffers` has
//   10 parameters by design (each buffer shape is a separate concern).
// * `large_enum_variant`— `Backend::Gpu(ComputeEngine)` is ~4 KB and the
//   enum is instantiated once per `CreamEngine`.  Boxing would add an
//   unnecessary indirection to every force evaluation's backend dispatch.
// * `needless_range_loop`— several loops use the index for multi-array
//   coordination (positions[i] + forces[i] + types[i]); rewriting to
//   `iter().enumerate()` would hurt readability without measurable perf win.
// * `unnecessary_map_or` — flagged by clippy ≥ 1.83 for idioms that remain
//   compatible with rustc 1.75 (our MSRV declared in rust-toolchain.toml).
// * `doc_lazy_continuation`, `doc_overindented_list_items` — rustdoc style
//   drifts that we don't want to chase each clippy release.
// * `dead_code` on `NeighborList::starts` — read via Debug impl and by
//   downstream consumers of the struct.
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::large_enum_variant)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::unnecessary_map_or)]
#![allow(clippy::doc_lazy_continuation)]
#![allow(clippy::doc_overindented_list_items)]

pub mod engine;
pub mod error;
pub mod potential;
pub mod reference;

pub(crate) mod cell_list;
pub(crate) mod neighbor_list;

#[cfg(not(target_arch = "wasm32"))]
pub mod cpu_engine;

#[cfg(feature = "python")]
pub mod python;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

// Re-export most-used types at crate root
#[cfg(not(target_arch = "wasm32"))]
pub use cpu_engine::CpuEngine;
pub use engine::{ComputeEngine, ComputeResult};
pub use error::CreamError;
pub use potential::{eam::EamPotential, NeighborStrategy};
pub use reference::ortho_cell;
