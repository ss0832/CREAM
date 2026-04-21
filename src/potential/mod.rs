//! GPU potential trait and supporting types.
//!
//! [`GpuPotential`] is the central abstraction: it decouples the physical
//! model (EAM, ADP, …) from the GPU dispatch engine. The engine only calls
//! the trait methods and knows nothing about physics.
//!
//! # Neighbour strategy
//! [`NeighborStrategy`] selects between O(N²) AllPairs and O(N) Cell List.
//! The engine inspects this to decide whether to run the cell-list
//! construction passes before density/force computation.

use crate::error::CreamError;
use std::borrow::Cow;

pub mod eam;
pub mod spline;

// ── Buffer layout ────────────────────────────────────────────────────────────

/// Buffer sizing hints returned by [`GpuPotential::buffer_layout`].
///
/// The engine uses these values to allocate VRAM without knowing the physics.
pub struct BufferLayout {
    /// Bytes per atom for the Pass 1 → Pass 2 intermediate buffer (electron
    /// density). For EAM this is 4 (one `f32`). For ADP it would be 48.
    pub intermediate_stride_bytes: u64,

    /// Bytes per atom for the force output buffer.
    /// **Always 16**: GPU 16-byte alignment requires `vec4<f32>` (xyz + pad).
    pub output_stride_bytes: u64,
}

// ── Neighbour strategy ───────────────────────────────────────────────────────

/// Selects the neighbour-list algorithm used by the compute engine.
#[derive(Clone, Debug, PartialEq)]
pub enum NeighborStrategy {
    /// O(N²) all-pairs scan — no spatial data structures.
    ///
    /// Correct for any N, and the simplest possible reference implementation.
    /// Becomes prohibitively slow beyond a few thousand atoms.
    AllPairs,
    /// O(N) Cell List — Morton-ordered spatial index.
    ///
    /// `cell_size` should be at least the potential cutoff radius (Å).
    CellList {
        /// Cell edge length in Å; must be ≥ potential cutoff radius.
        cell_size: f32,
    },
}

// ── GPU buffer handles ────────────────────────────────────────────────────────

/// Uploaded potential tables, kept alive for the lifetime of the engine
/// (held in `ComputeEngine::table_cache`).
///
/// All tables are packed into a **single** Storage Buffer to stay within the
/// WebGPU limit of 8 `max_storage_buffers_per_shader_stage`.
#[allow(missing_docs)]
pub struct PotentialGpuBuffers {
    /// Flat array: `rho | embed | dembed | pair | drho | dpair` (all `f32`).
    pub tables_buf: wgpu::Buffer,
    /// [`TableLayout`](eam::TableLayout) uniform with offsets and strides.
    pub layout_buf: wgpu::Buffer,
}

// ── GpuPotential trait ────────────────────────────────────────────────────────

/// Contract every physics model must satisfy.
///
/// The engine holds a `&dyn GpuPotential` and never reaches inside the
/// concrete type — all it needs are buffer specs, shader sources, table
/// uploads, and a cache key.
///
/// Implementors must be `Send + Sync` so they can be shared across threads
/// (e.g., in Python via PyO3).
pub trait GpuPotential: Send + Sync {
    /// Buffer sizing hints used by the engine to allocate VRAM.
    fn buffer_layout(&self) -> BufferLayout;

    /// Number of distinct element species in this potential.
    fn n_elements(&self) -> usize;

    /// Cutoff radius in Å — the engine derives `cutoff_sq` from this.
    fn cutoff(&self) -> f32;

    /// Pass 1 WGSL source: electron density accumulation (AllPairs mode).
    fn pass1_shader(&self) -> Cow<'static, str>;

    /// Pass 2 WGSL source: embedding derivative, forces, per-atom energy
    /// (AllPairs mode).
    fn pass2_shader(&self) -> Cow<'static, str>;

    /// Pass 1 WGSL source for Cell List mode.
    ///
    /// Returns `None` if the potential does not support Cell List pass 1
    /// (the engine returns an error if [`NeighborStrategy::CellList`] is
    /// selected in that case).
    fn pass1_cellist_shader(&self) -> Option<Cow<'static, str>> {
        None
    }

    /// Pass 2 WGSL source for Cell List mode.
    fn pass2_cellist_shader(&self) -> Option<Cow<'static, str>> {
        None
    }

    /// Pass 1 WGSL source for the CPU-built neighbour-list backend.
    ///
    /// When the engine is configured with [`NeighborStrategy::CellList`] and
    /// the potential returns `Some(..)` here, the engine builds a CSR
    /// neighbour list on the CPU and dispatches this shader instead of the
    /// full GPU cell-list pipeline. The CPU-built path is kept as a
    /// fallback for the rare GPU configurations where the native cell-list
    /// pipeline produces driver-dependent numerical noise.
    fn pass1_neighlist_shader(&self) -> Option<Cow<'static, str>> {
        None
    }

    /// Pass 2 WGSL source for the CPU-built neighbour-list backend.
    fn pass2_neighlist_shader(&self) -> Option<Cow<'static, str>> {
        None
    }

    /// Upload all potential tables to the GPU and return the buffer handles.
    /// Called at most once per potential instance — the engine caches the result.
    fn upload_tables(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<PotentialGpuBuffers, CreamError>;

    /// Unique string identifying this potential instance.
    /// Used as the key in the engine's pipeline and table caches.
    fn cache_key(&self) -> String;
}
