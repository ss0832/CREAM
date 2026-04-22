//! GPU compute engine — multi-pass EAM with Cell List support.
//!
//! # Optimizations (v0.1.0-beta.9)
//!
//! ## Position reordering — Pass 0d (CellList only)
//! After pass0c (Morton scatter), a new pass copies `pos_buf`/`types_buf` into
//! `reordered_pos_buf`/`reordered_types_buf` in Morton order.  CellList passes 1
//! and 2 bind the reordered buffers to common.wgsl bindings 0/1 so that the
//! inner-loop reads `positions[c_s + si]` (sequential) instead of
//! `positions[sorted_atoms[si]]` (random gather).
//!
//! Force outputs are scatter-written back to original order (`forces_out[sorted_atoms[k]]`)
//! so the Verlet integrator and CPU readback remain unaffected.
//!
//! ## Workgroup cooperative tile loading (CellList passes 1/2)
//! All 64 WG threads compute their bounding box via workgroup atomics (one barrier
//! pair), then iterate the ±1-padded bounding box together.  For each cell, threads
//! cooperatively load 64-atom tiles into `var<workgroup>` (tile_pos/tile_type/
//! tile_rho).  Reads are coalesced; every thread shares the same cell sequence so
//! barriers are executed uniformly.
//!
//! TODO(beta.10): texture cache for `potential_tables` on non-WASM targets.
//!   Steps: (1) copy tables_buf → 1D R32Float texture in enc_a; (2) add
//!   `tex_cache: HashMap<String, (wgpu::Texture, wgpu::TextureView)>` to the
//!   engine; (3) write `common_native.wgsl` replacing binding 3 with
//!   `texture_1d<f32>` and `textureLoad` in `_lerp`; (4) gate the BGL/BG binding
//!   3 entry on `#[cfg(not(target_arch = "wasm32"))]`.
//!
//! # Optimizations (v0.1.0-beta.8)
//!
//! ## energy_per_atom eliminated
//! Pass 2 shaders perform an in-shader workgroup tree-reduction, producing
//! `ceil(N/64)` workgroup partial sums instead of N per-atom values.
//! This eliminates the N×4 B energy_per_atom readback entirely.
//!
//! ## CommandEncoder batching (7 submit → 2 submit)
//! CellList:  Enc-A (clear + pass0a + pass0b + copy_counts) → CPU prefix sum
//!            → Enc-B (pass0c + pass1 + pass2 + pass3a/b/c + [verlet] + readbacks)
//! AllPairs:  Single Enc-B.
//! GPU round-trips reduced from 7 poll(Wait) calls to 1 or 2.
//!
//! ## clear_buffer() for cell_counts_buf
//! Replaces `queue.write_buffer(&zeros_vec)` with `encoder.clear_buffer()`,
//! eliminating the per-frame heap allocation of a zero-filled Vec.
//!
//! ## 4-level tree reduction — unlimited N
//! Pass 2 does level-0: N atoms → W = ceil(N/64) workgroup partial sums.
//! Pass 3a: W        → ceil(W/64)        (level 1)
//! Pass 3b: ceil(W/64)  → ceil(W/4096)  (level 2)
//! Pass 3c: ceil(W/4096) → 1            (level 3, 1 workgroup)
//!
//! Exact for N ≤ 64^4 = 16_777_216 atoms (≈16M).  The old fallback to CPU
//! sum for N > 4096 is removed.  `energy_per_atom` is always empty in GPU
//! results.
//!
//! ## Verlet GPU integration kernel
//! `step_sync()` / `step()` dispatches a leapfrog Velocity Verlet shader after
//! pass 2.  Positions and velocities remain on the GPU between steps.

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
use std::{
    collections::hash_map::DefaultHasher,
    collections::HashMap,
    hash::{Hash, Hasher},
    sync::{
        atomic::{AtomicBool, AtomicU8, Ordering},
        Arc,
    },
};

use bytemuck::{Pod, Zeroable};
#[cfg(target_arch = "wasm32")]
use js_sys::Promise as JsPromise;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsValue;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::JsFuture;
use wgpu::util::DeviceExt;

use crate::{
    error::CreamError,
    potential::{GpuPotential, NeighborStrategy, PotentialGpuBuffers},
};

// ── Static BindGroupLayout entry slices ──────────────────────────────────────

const fn stor_r(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
const fn stor_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
const fn uni(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

// ── Pass 0 ────────────────────────────────────────────────────────────────────
static BGL_PASS0A: &[wgpu::BindGroupLayoutEntry] = &[stor_r(0), uni(2), stor_rw(5)];
static BGL_PASS0B: &[wgpu::BindGroupLayoutEntry] = &[stor_r(0), uni(2), stor_r(5), stor_rw(6)];
static BGL_PASS0C: &[wgpu::BindGroupLayoutEntry] =
    &[stor_r(0), uni(2), stor_r(5), stor_rw(6), stor_rw(7)];
/// Pass 0d: reorder positions / types into Morton order.
/// Uses bindings 0 (positions), 1 (atom_types), 2 (params),
/// 5 (sorted_atoms, read), 6 (reordered_positions, write),
/// 7 (reordered_types, write), 8 (cell_ids, read — stored in reordered W).
static BGL_PASS0D: &[wgpu::BindGroupLayoutEntry] = &[
    stor_r(0),
    stor_r(1),
    uni(2),
    stor_r(5),
    stor_rw(6),
    stor_rw(7),
    stor_r(8),
];

// ── Pass 1 ────────────────────────────────────────────────────────────────────
static BGL_PASS1_AP: &[wgpu::BindGroupLayoutEntry] =
    &[stor_r(0), stor_r(1), uni(2), stor_r(3), uni(4), stor_rw(5)];
/// CellList pass 1: binding 0/1 = reordered pos/types; no sorted_atoms needed
/// (self-exclusion is by Morton index equality); cell_start at binding 6;
/// debug_flags (atomic counters) at binding 10.
static BGL_PASS1_CL: &[wgpu::BindGroupLayoutEntry] = &[
    stor_r(0),
    stor_r(1),
    uni(2),
    stor_r(3),
    uni(4),
    stor_rw(5),  // densities (write, Morton order)
    stor_r(6),   // cell_start
    stor_rw(10), // debug_flags
];

// ── Pass 2 ───────────────────────────────────────────────────────────────────
//   binding  7 = wg_energy_out   (per-WG energy partial,  one f32  per WG)
//   binding 11 = wg_virial_out   (per-WG virial partials, six f32 per WG)
// The two reductions share the same 6-barrier tree inside the pass-2 shader
// (no extra barrier — see shader comments).  CPU sums the ceil(N/64)
// partials into six totals.
static BGL_PASS2_AP: &[wgpu::BindGroupLayoutEntry] = &[
    stor_r(0),
    stor_r(1),
    uni(2),
    stor_r(3),
    uni(4),
    stor_r(5),
    stor_rw(6),
    stor_rw(7),
    stor_rw(11), // wg_virial_out
];
static BGL_PASS2_CL: &[wgpu::BindGroupLayoutEntry] = &[
    stor_r(0),
    stor_r(1),
    uni(2),
    stor_r(3),
    uni(4),
    stor_r(5),
    stor_rw(6),
    stor_rw(7),
    stor_r(8),
    stor_r(9),
    stor_rw(10), // debug_flags
    stor_rw(11), // wg_virial_out
];

// ── Pass 1 / Pass 2 — CPU-built NeighborList mode ────────────────────────────
// Fallback path for the rare GPU configurations where the native GPU
// cell-list pipeline is unreliable. The CPU builds a CSR neighbour list and
// uploads it; the GPU just walks the list.
//   pass1_nl:
//     binding 5 = densities (write, original atom order)
//     binding 6 = nl_starts (CSR row pointers, length N+1)
//     binding 7 = nl_list   (flat neighbour indices)
//   pass2_nl:
//     binding 5 = densities_in
//     binding 6 = forces_out (direct write, no scatter)
//     binding 7 = wg_energy_out
//     binding 8 = nl_starts
//     binding 9 = nl_list
// Bindings 0 / 1 are the *original* pos_buf / types_buf — NO Morton reorder.
static BGL_PASS1_NL: &[wgpu::BindGroupLayoutEntry] = &[
    stor_r(0),
    stor_r(1),
    uni(2),
    stor_r(3),
    uni(4),
    stor_rw(5),  // densities
    stor_r(6),   // nl_starts
    stor_r(7),   // nl_list
    stor_rw(10), // debug_flags
];
static BGL_PASS2_NL: &[wgpu::BindGroupLayoutEntry] = &[
    stor_r(0),
    stor_r(1),
    uni(2),
    stor_r(3),
    uni(4),
    stor_r(5),   // densities_in
    stor_rw(6),  // forces_out
    stor_rw(7),  // wg_energy_out
    stor_r(8),   // nl_starts
    stor_r(9),   // nl_list
    stor_rw(10), // debug_flags
    stor_rw(11), // wg_virial_out
];

// ── Pass 3 tree-reduction ─────────────────────────────────────────────────────
static BGL_PASS3: &[wgpu::BindGroupLayoutEntry] = &[stor_r(0), stor_rw(1), uni(2)];

// ── Verlet integration ────────────────────────────────────────────────────────
static BGL_VERLET: &[wgpu::BindGroupLayoutEntry] = &[stor_rw(0), stor_rw(1), stor_r(2), uni(3)];

// ── f32 matrix helper ─────────────────────────────────────────────────────────

#[inline]
fn mat3_inv_f32(h: &[[f32; 3]; 3]) -> Option<[[f32; 3]; 3]> {
    let [[h00, h01, h02], [h10, h11, h12], [h20, h21, h22]] = *h;
    let det = h00 * (h11 * h22 - h12 * h21) - h01 * (h10 * h22 - h12 * h20)
        + h02 * (h10 * h21 - h11 * h20);
    if det.abs() < 1e-30 {
        return None;
    }
    let d = 1.0 / det;
    Some([
        [
            (h11 * h22 - h12 * h21) * d,
            -(h01 * h22 - h02 * h21) * d,
            (h01 * h12 - h02 * h11) * d,
        ],
        [
            -(h10 * h22 - h12 * h20) * d,
            (h00 * h22 - h02 * h20) * d,
            -(h00 * h12 - h02 * h10) * d,
        ],
        [
            (h10 * h21 - h11 * h20) * d,
            -(h00 * h21 - h01 * h20) * d,
            (h00 * h11 - h01 * h10) * d,
        ],
    ])
}

// ── Public result type ────────────────────────────────────────────────────────

/// Result of an EAM calculation.
///
/// Several fields are *CPU-only* by design — populating them would require
/// additional GPU→CPU readback buffers whose cost the GPU path deliberately
/// avoids.  Each CPU-only field is empty (`vec![]`) when the result came
/// from the GPU engine; see individual field docs for details.
#[derive(Debug, Clone)]
pub struct ComputeResult {
    /// Force on each atom [eV/Å]. Length equals atom count.
    pub forces: Vec<[f32; 3]>,
    /// Total potential energy [eV].
    pub energy: f32,
    /// Per-atom energy [eV].
    ///
    /// **CPU-only.**  Empty (`vec![]`) for GPU results — the N×4 B readback
    /// this would require dominated readback latency in profiling, so the
    /// GPU path returns only the global sum via `energy`.
    pub energy_per_atom: Vec<f32>,
    /// Virial stress tensor in Voigt notation `[σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy]`,
    /// units eV/Å³, sign convention `σ = −W/V` where `W = Σᵢ<ⱼ rᵢⱼ ⊗ Fᵢⱼ`.
    ///
    /// Populated by both CPU and GPU engines when the system is periodic.
    /// Zero (`[0.0; 6]`) for non-periodic (cluster) calculations where volume
    /// is undefined.  Divide by `160.21766` to convert to GPa.
    pub virial: [f64; 6],
    /// Per-atom virial tensor in Voigt notation, units **eV** (not divided by
    /// volume — raw pair-virial contribution assigned half-and-half to the
    /// two atoms of each pair, matching LAMMPS' `compute stress/atom`
    /// convention).
    ///
    /// Shape is `[N][6]`; `Σᵢ virial_per_atom[i][c]` equals the
    /// pre-normalisation raw virial `W_c` (i.e. `−virial[c] × V`).
    ///
    /// **CPU-only.**  Empty (`vec![]`) for GPU results.
    pub virial_per_atom: Vec<[f64; 6]>,
    /// EAM host electron density `ρᵢ = Σⱼ≠ᵢ fβ(rᵢⱼ)` at each atom site.
    ///
    /// Length `N`, units are dimensionless (same units as the `.eam.alloy`
    /// file).  Useful for diagnosing surface or defect environments.
    ///
    /// **CPU-only.**  Empty (`vec![]`) for GPU results — densities live
    /// only in GPU-resident buffers and are not read back.
    pub densities: Vec<f32>,
    /// EAM embedding energy `F_α(ρᵢ)` at each atom site [eV].
    ///
    /// `Σᵢ embedding_energies[i]` equals the embedding contribution to the
    /// total energy (the remainder is the pair sum `½ Σᵢ<ⱼ φ_αβ(rᵢⱼ)`).
    ///
    /// **CPU-only.**  Empty (`vec![]`) for GPU results.
    pub embedding_energies: Vec<f32>,
}

// ── CellList debug readback type ──────────────────────────────────────────────

/// Intermediate GPU state snapshot for post-mortem analysis of the CellList
/// pipeline.  Populated only when `compute_with_debug()` is called and a
/// CellList strategy is in use; the AllPairs path yields `None`.
///
/// The arrays are Morton-ordered where noted ("sorted" / "Morton slot k" =
/// index into `sorted_atoms`).  Cross-referencing example:
///
/// ```text
/// reordered_positions[k][0..3] == positions[sorted_atoms[k]]
/// reordered_positions[k][3]    == f32(cell_ids[sorted_atoms[k]])
/// densities[k]                 == rho for the Morton-k atom
/// ```
///
/// `debug_flags` decodes as documented on `CellListBuffers::debug_flags_buf`.
#[derive(Debug, Clone)]
pub struct CellListDebugReadback {
    /// Number of atoms (length of the original `positions` slice).
    pub n_atoms: usize,
    /// Cell-grid size (actual, non-padded): `(ncx, ncy, ncz)`.
    pub n_cells: (u32, u32, u32),
    /// Morton-padded grid (next power of two per axis).
    pub n_cells_pad: (u32, u32, u32),
    /// Size of the Morton code space (`cell_start.len() == n_morton + 1`).
    pub n_morton: usize,
    /// `cell_ids[i]` = Morton code assigned to atom `i` by pass0a.
    /// Length = `n_atoms`.
    pub cell_ids: Vec<u32>,
    /// `sorted_atoms[k]` = original index of the atom in Morton slot `k`.
    /// Must be a permutation of `0..n_atoms`. Length = `n_atoms`.
    pub sorted_atoms: Vec<u32>,
    /// Prefix sum built on the CPU after pass0b and re-uploaded to the GPU.
    /// `cell_start[m..m+1]` = slice of `sorted_atoms` belonging to Morton cell
    /// `m`.  Length = `n_morton + 1`; last element equals `n_atoms`.
    pub cell_start: Vec<u32>,
    /// Raw counts per Morton cell (from pass0b).  Length = `n_morton`.
    /// `cell_counts[m] == cell_start[m+1] - cell_start[m]`.
    pub cell_counts: Vec<u32>,
    /// Morton-ordered positions written by pass0d, 4-wide (xyz + cell ID as f32).
    /// Length = `n_atoms`.
    pub reordered_positions: Vec<[f32; 4]>,
    /// Morton-ordered types from pass0d. Length = `n_atoms`.
    pub reordered_types: Vec<u32>,
    /// Morton-ordered densities (`rho`) from pass1. Length = `n_atoms`.
    pub densities: Vec<f32>,
    /// 32-slot debug flag buffer written by pass1 / pass2 shaders.
    /// See `CellListBuffers::debug_flags_buf` for slot layout.
    pub debug_flags: [u32; 32],
    /// Coarse phase timing measured via `Instant` at submit / poll
    /// boundaries.  Populated only when `compute_with_debug()` is called on
    /// the CellList path; `None` for AllPairs or when the timing window
    /// cannot be constructed.  Times are GPU wall-clock as observed from
    /// the CPU side (i.e. include queue.submit overhead and device.poll
    /// blocking).
    pub timings: Option<SubmissionTimings>,
}

/// Coarse three-phase timing breakdown for a CellList frame.
///
/// Measured between explicit `queue.submit` / `device.poll(Maintain::Wait)`
/// boundaries — these are the only CPU-visible synchronisation points in
/// the pipeline.  Fine-grained per-pass timings would require wgpu
/// `TIMESTAMP_QUERY` infrastructure, which is a follow-up.
///
/// The three phases map to what `profile_gpu_v2` counters told us the
/// workload looks like, so this timing is directly comparable:
///   - Submission A = pass0a + pass0b + count readback  (neighbour build)
///   - CPU prefix   = prefix-sum on CPU + re-upload     (host round-trip)
///   - Submission B = pass0c+0d+pass1+pass2+pass3      (density + forces)
#[derive(Debug, Clone, Copy, Default)]
pub struct SubmissionTimings {
    /// Submission A wall-clock including clear + pass0a + pass0b + count
    /// readback copy.  Ends when `device.poll(Wait)` returns (counts are
    /// readable on CPU).
    pub submission_a_ms: f64,
    /// CPU prefix-sum + re-upload of `cell_start` / `write_offsets`.
    /// Ends when the upload enqueue returns (not synchronised until the
    /// next submit).
    pub cpu_prefix_ms: f64,
    /// Submission B wall-clock: pass0c + pass0d + pass1 + pass2 + pass3
    /// + readback copies.  Ends when `device.poll(Wait)` returns
    /// (forces and energy are readable on CPU).
    pub submission_b_ms: f64,
}

// ── GPU uniforms ──────────────────────────────────────────────────────────────

/// Must match `SimParams` in `common.wgsl` byte-for-byte (176 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuSimParams {
    n_atoms: u32,
    n_elem: u32,
    cutoff_sq: f32,
    min_dist_sq: f32,
    h0: [f32; 4],
    h1: [f32; 4],
    h2: [f32; 4],
    hinv_col0: [f32; 4],
    hinv_col1: [f32; 4],
    hinv_col2: [f32; 4],
    use_cell_list: u32,
    use_pbc: u32,
    _pad0: u32,
    _pad1: u32,
    n_cells_x: u32,
    n_cells_y: u32,
    n_cells_z: u32,
    n_cells_total: u32,
    cell_size: f32,
    n_cells_x_pad: u32,
    n_cells_y_pad: u32,
    n_cells_z_pad: u32,
    // Chunked dispatch: first Morton/atom index and count for this chunk.
    // For a full (non-chunked) dispatch: i_offset = 0, i_count = n_atoms.
    // i_offset is always a multiple of 64 (guaranteed by the Rust chunk loop).
    i_offset: u32,
    i_count: u32,
    _pad2: u32,
    _pad3: u32,
}

/// Uniform for the tree-reduction shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ReduceParams {
    count: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

/// Uniform for the Verlet integration shader.
/// Must match `VerletParams` in `verlet_integrate.wgsl` byte-for-byte.
#[allow(missing_docs)]
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct VerletParams {
    pub n_atoms: u32,
    /// Timestep [ps].
    pub dt: f32,
    /// 1 / (m_amu × 9648.5).  Pass 0.0 to disable integration.
    pub inv_mass: f32,
    pub _pad: u32,
}

// ── FrameBuffers ──────────────────────────────────────────────────────────────

/// Per-frame GPU buffers, cached while `n` is stable.
struct FrameBuffers {
    n: usize,
    /// True when `pos_buf` holds the latest Verlet-integrated positions.
    /// The next `run_passes()` call skips the CPU→GPU position upload.
    positions_current_on_gpu: bool,

    // Upload
    pos_buf: wgpu::Buffer,    // vec4<f32> × n
    types_buf: wgpu::Buffer,  // u32 × n
    params_buf: wgpu::Buffer, // GpuSimParams (176 B)

    // Compute intermediates
    density_buf: wgpu::Buffer, // f32 × n
    forces_buf: wgpu::Buffer,  // vec4<f32> × n

    // 3-level GPU energy reduction chain + CPU final sum
    //   pass2 (in-shader): N atoms → wg_energy_buf[ceil(N/64)]
    //   pass3a: wg_energy_buf → lvl1_buf[ceil(N/4096)]
    //   pass3b: lvl1_buf      → lvl2_buf[ceil(N/262144)]
    //   CPU:    Neumaier-compensated sum of lvl2_buf → scalar  (no upper N limit)
    //
    // pass3c (lvl2 → 1 workgroup) has been removed: it silently dropped
    // elements when lvl2_count > 64 (i.e. N > 16_777_216 = 64^4), causing
    // ~0.4 % energy under-estimation at N ≈ 23 M.  The CPU Neumaier sum
    // costs < 1 µs even for very large N and has no correctness limit.
    wg_energy_buf: wgpu::Buffer, // ceil(N/64)
    lvl1_buf: wgpu::Buffer,      // ceil(N/4096)
    lvl2_buf: wgpu::Buffer,      // ceil(N/262144)

    // Virial reduction — written by pass2 shader alongside wg_energy_buf.
    //
    // Each workgroup emits six f32 (`[Wxx, Wyy, Wzz, Wyz, Wxz, Wxy]`,
    // raw pair-virial partials, **not** yet divided by V).  The tree
    // reduction inside pass2 reuses the exact 6-barrier schedule already
    // walked for `wg_energy` — six extra workgroup-shared scalars and
    // six extra adds per step, zero additional `workgroupBarrier()` calls.
    //
    // Layout: flat `array<f32>` of length `6 × ceil(N/64)`.  Workgroup
    // `w` writes `wg_virial_buf[6*w .. 6*w + 6]`.
    //
    // We do not chain a GPU `pass3`-style reduction for virial because the
    // partial-sum count is tiny (≤ 262 144 for N ≤ 16.8 M) and the CPU
    // performs a Neumaier-compensated sum in O(N/64) — microseconds of
    // host work versus the overhead of six more reduction dispatches.
    wg_virial_buf: wgpu::Buffer,         // 4 × 6 × ceil(N/64) bytes
    rb_virial_partials: wgpu::Buffer,    // readback of wg_virial_buf

    reduce_params_a_buf: wgpu::Buffer, // ReduceParams for pass3a
    reduce_params_b_buf: wgpu::Buffer, // ReduceParams for pass3b

    // Verlet MD
    velocities_buf: wgpu::Buffer,    // vec4<f32> × n  (init 0)
    verlet_params_buf: wgpu::Buffer, // VerletParams (16 B)

    // Readback (forces + lvl2 energy partials only — no energy_per_atom)
    rb_forces: wgpu::Buffer, // N × 16 B
    rb_lvl2: wgpu::Buffer,   // ceil(N/262144) × 4 B  — CPU-summed to get total energy

    // BindGroups
    bg_pass0a: Option<wgpu::BindGroup>,
    bg_pass0b: Option<wgpu::BindGroup>,
    bg_pass0c: Option<wgpu::BindGroup>,
    /// Reorder pass bind group (CellList only).
    bg_pass0d: Option<wgpu::BindGroup>,
    bg_pass1: Option<(u64, wgpu::BindGroup)>,
    bg_pass2: Option<(u64, wgpu::BindGroup)>,
    bg_pass3a: Option<wgpu::BindGroup>,
    bg_pass3b: Option<wgpu::BindGroup>,
    bg_verlet: Option<wgpu::BindGroup>,

    cell_list: Option<CellListBuffers>,
    neighbor_list: Option<NeighborListBuffers>,
}

// ── CellListBuffers ───────────────────────────────────────────────────────────

#[allow(dead_code)]
struct CellListBuffers {
    n_atoms: usize,
    n_cells_x: u32,
    n_cells_y: u32,
    n_cells_z: u32,
    n_cells_total: u32,
    cell_size: f32,
    n_cells_x_pad: u32,
    n_cells_y_pad: u32,
    n_cells_z_pad: u32,
    n_morton: usize,
    cell_ids_buf: wgpu::Buffer,
    sorted_atoms_buf: wgpu::Buffer,
    cell_start_buf: wgpu::Buffer,
    cell_counts_buf: wgpu::Buffer,
    write_offsets_buf: wgpu::Buffer,
    rb_cell_counts: wgpu::Buffer,
    /// Positions reordered into Morton order by pass0d.
    reordered_pos_buf: wgpu::Buffer,
    /// Atom types reordered into Morton order by pass0d.
    reordered_types_buf: wgpu::Buffer,
    /// CellList diagnostic flag buffer — 32 × u32 slots.
    ///
    /// Layout (written by pass1_cellist / pass2_cellist when
    /// `params.use_cell_list == 1`):
    ///   [ 0]  pass1_cid_oob_count     — #(cid >= n_morton_capacity) in pass1
    ///   [ 1]  pass1_cs_oob_count      — #(cell_start[cid+1] > n_atoms) in pass1
    ///   [ 2]  pass1_cs_inverted_count — #(cell_start[cid+1] < cell_start[cid])
    ///   [ 3]  pass1_bb_empty_count    — #(bb_x0 > bb_x1) in pass1
    ///   [ 4]  pass1_atom_k_oob_count  — #(atom_k >= n_atoms) in pass1
    ///   [ 5]  pass1_nan_rho_count     — #(rho became NaN/Inf)
    ///   [ 6]  pass1_neighbor_count    — total pair-visit count (for throughput check)
    ///   [ 7]  pass1_cutoff_hit_count  — #(r_sq < cutoff_sq) pair hits
    ///   [ 8]  pass2_cid_oob_count     — same as [0] but for pass2
    ///   [ 9]  pass2_cs_oob_count      — same as [1] but for pass2
    ///   [10]  pass2_cs_inverted_count
    ///   [11]  pass2_bb_empty_count
    ///   [12]  pass2_atom_k_oob_count
    ///   [13]  pass2_nan_force_count
    ///   [14]  pass2_neighbor_count
    ///   [15]  pass2_cutoff_hit_count
    ///   [16]  pass2_sorted_atom_oob   — #(sorted_atoms[k] >= n_atoms on scatter)
    ///   [17..31] reserved
    ///
    /// Cleared to zero before each Submission-B run.  Read back on demand
    /// via `debug_readback_u32_slice` when CREAM_DIAG is set, or via the
    /// `compute_with_debug()` entry point exposed to Python.
    debug_flags_buf: wgpu::Buffer,
}

// ── NeighborListBuffers ───────────────────────────────────────────────────────
//
// GPU-side storage for the CPU-built CSR neighbour list consumed by
// `eam_pass1_neighlist.wgsl` / `eam_pass2_neighlist.wgsl`.  This path is a
// fallback to the native GPU cell-list pipeline.
//
// Layout:
//   nl_starts_buf  : array<u32> of length n_atoms + 1 (CSR row pointers)
//   nl_list_buf    : array<u32> of length = starts[n_atoms]
//                    Grown with ~25 % headroom to avoid per-step reallocation.
//   debug_flags_buf: 32 × u32, same slot semantics as CellListBuffers
//                    (see its doc for layout).
#[allow(dead_code)]
struct NeighborListBuffers {
    n_atoms: usize,
    /// Capacity (u32 entries) currently allocated for `nl_list_buf`.
    list_capacity: u32,
    nl_starts_buf: wgpu::Buffer,
    nl_list_buf: wgpu::Buffer,
    debug_flags_buf: wgpu::Buffer,
}

// ── ComputeEngine ─────────────────────────────────────────────────────────────

/// GPU compute engine.  Create once per process; reuse across MD steps.
pub struct ComputeEngine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pl_pass0a: Option<wgpu::ComputePipeline>,
    pl_pass0b: Option<wgpu::ComputePipeline>,
    pl_pass0c: Option<wgpu::ComputePipeline>,
    /// Reorder pass: scatter positions/types into Morton order (CellList only).
    pl_pass0d: Option<wgpu::ComputePipeline>,
    pl_pass3: Option<wgpu::ComputePipeline>,
    pl_verlet: Option<wgpu::ComputePipeline>,
    pipeline_cache: HashMap<u64, wgpu::ComputePipeline>,
    table_cache: HashMap<String, Arc<PotentialGpuBuffers>>,
    frame_cache: Option<FrameBuffers>,
    strategy: NeighborStrategy,
    /// Stashed coarse phase timings from the most recent `run_passes()`
    /// invocation.  Written by `run_passes` at submit / poll boundaries,
    /// read by `compute_with_debug` for inclusion in the debug readback.
    /// `None` on AllPairs (no Submission A) or before the first run.
    last_timings: Option<SubmissionTimings>,
}

type CellListInfo = (u32, u32, u32, u32, f32, u32, u32, u32);

impl ComputeEngine {
    /// Returns a reference to the active neighbour strategy.
    pub fn strategy(&self) -> &NeighborStrategy {
        &self.strategy
    }

    // ── Initialization ──────────────────────────────────────────────────────

    /// Initialize GPU adapter, device, and queue.
    pub async fn new(strategy: NeighborStrategy) -> Result<Self, CreamError> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .ok_or_else(|| CreamError::DeviceLost("No GPU adapter found".into()))?;

        // Use the adapter's actual hardware limits as the base so that
        // max_storage_buffer_binding_size is not artificially capped at the
        // 128 MB downlevel default.  We only override the storage-buffer
        // slot count to ensure CellList shaders (which need up to 13 with
        // the virial-reduction buffer at binding 11) work.
        let adapter_limits = adapter.limits();
        let required_limits = wgpu::Limits {
            max_storage_buffers_per_shader_stage: adapter_limits
                .max_storage_buffers_per_shader_stage
                .max(13),
            ..adapter_limits
        };

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_limits,
                    ..Default::default()
                },
                None,
            )
            .await
            .map_err(|e| CreamError::DeviceLost(e.to_string()))?;

        Ok(Self {
            device,
            queue,
            pl_pass0a: None,
            pl_pass0b: None,
            pl_pass0c: None,
            pl_pass0d: None,
            pl_pass3: None,
            pl_verlet: None,
            pipeline_cache: HashMap::new(),
            table_cache: HashMap::new(),
            frame_cache: None,
            strategy,
            last_timings: None,
        })
    }

    /// Initialize for WASM/WebGPU targets (degrades CellList→AllPairs if needed).
    #[cfg(target_arch = "wasm32")]
    pub async fn new_webgpu(mut strategy: NeighborStrategy) -> Result<Self, CreamError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .ok_or_else(|| CreamError::DeviceLost("No WebGPU adapter found".into()))?;

        // Query the adapter once; reuse for both the CellList check and limits.
        let adapter_limits = adapter.limits();
        let max_sb = adapter_limits.max_storage_buffers_per_shader_stage;
        if matches!(strategy, NeighborStrategy::CellList { .. }) && max_sb < 11 {
            log::warn!(
                "WebGPU adapter only guarantees {} storage buffers (need 11 for CellList \
                 with virial reduction). Falling back to AllPairs.",
                max_sb
            );
            strategy = NeighborStrategy::AllPairs;
        }

        // Use the adapter's actual hardware limits as the base so that
        // max_storage_buffer_binding_size is not artificially capped at the
        // 128 MB that downlevel_webgl2_defaults() imposes.  Most modern
        // browsers/drivers report 256 MB – 2 GB here.
        // Only cap the storage-buffer slot count to what the adapter supplies.
        let required_limits = wgpu::Limits {
            max_storage_buffers_per_shader_stage: 11u32.min(max_sb),
            ..adapter_limits
        };

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_limits,
                    ..Default::default()
                },
                None,
            )
            .await
            .map_err(|e| CreamError::DeviceLost(e.to_string()))?;

        Ok(Self {
            device,
            queue,
            pl_pass0a: None,
            pl_pass0b: None,
            pl_pass0c: None,
            pl_pass0d: None,
            pl_pass3: None,
            pl_verlet: None,
            pipeline_cache: HashMap::new(),
            table_cache: HashMap::new(),
            frame_cache: None,
            strategy,
            last_timings: None,
        })
    }

    // ── Sync wrappers ────────────────────────────────────────────────────────

    /// Synchronous wrapper for `compute()` (for PyO3 / CLI).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn compute_sync(
        &mut self,
        positions: &[[f32; 4]],
        atom_types: &[u32],
        cell: Option<[[f32; 3]; 3]>,
        potential: &dyn GpuPotential,
    ) -> Result<ComputeResult, CreamError> {
        pollster::block_on(self.compute(positions, atom_types, cell, potential))
    }

    /// Synchronous wrapper for `compute_with_debug()`.
    ///
    /// Runs the normal GPU pipeline and, in addition, reads back every
    /// intermediate CellList buffer so the caller can inspect them.
    /// Returns `(result, Some(debug))` when CellList was used, and
    /// `(result, None)` otherwise.
    ///
    /// This has a real performance cost (several extra GPU→CPU copies and
    /// `device.poll(Wait)` stalls) so it should only be called from tests
    /// and diagnostic tools, not the MD hot loop.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn compute_sync_with_debug(
        &mut self,
        positions: &[[f32; 4]],
        atom_types: &[u32],
        cell: Option<[[f32; 3]; 3]>,
        potential: &dyn GpuPotential,
    ) -> Result<(ComputeResult, Option<CellListDebugReadback>), CreamError> {
        pollster::block_on(self.compute_with_debug(positions, atom_types, cell, potential))
    }

    /// Run one MD step: compute forces+energy, then integrate with leapfrog
    /// Verlet.  Positions stay on the GPU between calls; the first call (or
    /// after an N change) uploads positions from the `positions` slice.
    ///
    /// # Parameters
    /// - `dt`:       timestep [ps]
    /// - `inv_mass`: 1 / (m_amu × 9648.5).  Pass `0.0` to skip integration.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn step_sync(
        &mut self,
        positions: &[[f32; 4]],
        atom_types: &[u32],
        cell: Option<[[f32; 3]; 3]>,
        potential: &dyn GpuPotential,
        dt: f32,
        inv_mass: f32,
    ) -> Result<ComputeResult, CreamError> {
        pollster::block_on(self.step(positions, atom_types, cell, potential, dt, inv_mass))
    }

    // ── Main calculation ─────────────────────────────────────────────────────

    /// Executes the EAM passes and returns forces and energy.
    /// `energy_per_atom` in the result is always empty for GPU results.
    pub async fn compute(
        &mut self,
        positions: &[[f32; 4]],
        atom_types: &[u32],
        cell: Option<[[f32; 3]; 3]>,
        potential: &dyn GpuPotential,
    ) -> Result<ComputeResult, CreamError> {
        if let Some(fb) = self.frame_cache.as_mut() {
            fb.positions_current_on_gpu = false;
        }
        self.run_passes(positions, atom_types, cell, potential, None)
            .await
    }

    /// Same as [`compute`] but also reads back every intermediate CellList
    /// buffer for post-mortem inspection.
    ///
    /// Returns `(result, Some(debug))` when a CellList strategy is active,
    /// and `(result, None)` otherwise (the AllPairs path has no intermediate
    /// CellList state).
    ///
    /// **Performance**: this performs several extra GPU→CPU copies and
    /// forces multiple `device.poll(Wait)` stalls.  Use only in tests or
    /// diagnostic tooling; do not call per MD step.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn compute_with_debug(
        &mut self,
        positions: &[[f32; 4]],
        atom_types: &[u32],
        cell: Option<[[f32; 3]; 3]>,
        potential: &dyn GpuPotential,
    ) -> Result<(ComputeResult, Option<CellListDebugReadback>), CreamError> {
        if let Some(fb) = self.frame_cache.as_mut() {
            fb.positions_current_on_gpu = false;
        }
        let result = self
            .run_passes(positions, atom_types, cell, potential, None)
            .await?;

        // Early-out for AllPairs: no cell-list state to expose.
        if !matches!(self.strategy, NeighborStrategy::CellList { .. }) || cell.is_none() {
            return Ok((result, None));
        }
        let fb = match self.frame_cache.as_ref() {
            Some(fb) => fb,
            None => return Ok((result, None)),
        };
        let cl = match fb.cell_list.as_ref() {
            Some(cl) => cl,
            None => return Ok((result, None)),
        };
        let n = positions.len();
        let n_morton = cl.n_morton;

        // Small closures that perform a single GPU→CPU copy+map round-trip.
        // They deliberately re-submit per call: debug readback is already
        // slow, and keeping them isolated avoids perturbing the hot path.
        let device = &self.device;
        let queue = &self.queue;

        let read_u32 = |buf: &wgpu::Buffer, count: usize| -> Vec<u32> {
            if count == 0 {
                return Vec::new();
            }
            let rb = Self::alloc_readback(device, 4 * count as u64);
            let mut enc = device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(buf, 0, &rb, 0, 4 * count as u64);
            queue.submit([enc.finish()]);
            device.poll(wgpu::Maintain::Wait);
            let slice = rb.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            device.poll(wgpu::Maintain::Wait);
            let v: Vec<u32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
            rb.unmap();
            v
        };
        let read_f32 = |buf: &wgpu::Buffer, count: usize| -> Vec<f32> {
            if count == 0 {
                return Vec::new();
            }
            let rb = Self::alloc_readback(device, 4 * count as u64);
            let mut enc = device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(buf, 0, &rb, 0, 4 * count as u64);
            queue.submit([enc.finish()]);
            device.poll(wgpu::Maintain::Wait);
            let slice = rb.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            device.poll(wgpu::Maintain::Wait);
            let v: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
            rb.unmap();
            v
        };
        let read_vec4 = |buf: &wgpu::Buffer, count: usize| -> Vec<[f32; 4]> {
            if count == 0 {
                return Vec::new();
            }
            let rb = Self::alloc_readback(device, 16 * count as u64);
            let mut enc = device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(buf, 0, &rb, 0, 16 * count as u64);
            queue.submit([enc.finish()]);
            device.poll(wgpu::Maintain::Wait);
            let slice = rb.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            device.poll(wgpu::Maintain::Wait);
            let v: Vec<[f32; 4]> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
            rb.unmap();
            v
        };

        let cell_ids = read_u32(&cl.cell_ids_buf, n);
        let sorted_atoms = read_u32(&cl.sorted_atoms_buf, n);
        let cell_start = read_u32(&cl.cell_start_buf, n_morton + 1);
        let cell_counts = read_u32(&cl.cell_counts_buf, n_morton);
        let reordered_positions = read_vec4(&cl.reordered_pos_buf, n);
        let reordered_types = read_u32(&cl.reordered_types_buf, n);
        let densities = read_f32(&fb.density_buf, n);
        let dbg_vec = read_u32(&cl.debug_flags_buf, 32);

        let mut debug_flags = [0u32; 32];
        for (i, &v) in dbg_vec.iter().take(32).enumerate() {
            debug_flags[i] = v;
        }

        let debug = CellListDebugReadback {
            n_atoms: n,
            n_cells: (cl.n_cells_x, cl.n_cells_y, cl.n_cells_z),
            n_cells_pad: (cl.n_cells_x_pad, cl.n_cells_y_pad, cl.n_cells_z_pad),
            n_morton,
            cell_ids,
            sorted_atoms,
            cell_start,
            cell_counts,
            reordered_positions,
            reordered_types,
            densities,
            debug_flags,
            timings: self.last_timings.take(),
        };
        Ok((result, Some(debug)))
    }

    /// MD step: compute forces then Verlet-integrate positions on GPU.
    pub async fn step(
        &mut self,
        positions: &[[f32; 4]],
        atom_types: &[u32],
        cell: Option<[[f32; 3]; 3]>,
        potential: &dyn GpuPotential,
        dt: f32,
        inv_mass: f32,
    ) -> Result<ComputeResult, CreamError> {
        self.run_passes(
            positions,
            atom_types,
            cell,
            potential,
            Some(VerletParams {
                n_atoms: positions.len() as u32,
                dt,
                inv_mass,
                _pad: 0,
            }),
        )
        .await
    }

    // ── Core dispatch ─────────────────────────────────────────────────────────

    async fn run_passes(
        &mut self,
        positions: &[[f32; 4]],
        atom_types: &[u32],
        cell: Option<[[f32; 3]; 3]>,
        potential: &dyn GpuPotential,
        verlet: Option<VerletParams>,
    ) -> Result<ComputeResult, CreamError> {
        let n = positions.len();
        if n == 0 {
            return Err(CreamError::InvalidInput("positions is empty".into()));
        }
        if atom_types.len() != n {
            return Err(CreamError::InvalidInput(format!(
                "atom_types length {} ≠ positions length {}",
                atom_types.len(),
                n
            )));
        }
        let n_elements = potential.n_elements();
        for (i, &t) in atom_types.iter().enumerate() {
            if (t as usize) >= n_elements {
                return Err(CreamError::InvalidInput(format!(
                    "atom_types[{i}]={t} out of range (n_elem={n_elements})"
                )));
            }
        }

        // ── Cell list geometry ──────────────────────────────────────────────
        let cl_info: Option<CellListInfo> = match (&self.strategy, cell) {
            (NeighborStrategy::CellList { cell_size }, Some(h)) => {
                let cs = *cell_size;
                if cs <= 0.0 || !cs.is_finite() {
                    return Err(CreamError::InvalidInput(format!(
                        "CellList cell_size must be positive and finite, got {cs}"
                    )));
                }
                let (nx, ny, nz) = Self::n_cells_from_dspacing(&h, cs);
                let nx_pad = nx.next_power_of_two();
                let ny_pad = ny.next_power_of_two();
                let nz_pad = nz.next_power_of_two();
                Some((nx, ny, nz, nx * ny * nz, cs, nx_pad, ny_pad, nz_pad))
            }
            (NeighborStrategy::CellList { .. }, None) => {
                return Err(CreamError::InvalidInput(
                    "CellList strategy requires periodic boundary conditions".into(),
                ));
            }
            _ => None,
        };

        // ── PBC setup ───────────────────────────────────────────────────────
        let (h, hinv, use_pbc) = match cell {
            None => {
                let id = [[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
                (id, id, 0u32)
            }
            Some(h) => {
                let h_inv = mat3_inv_f32(&h)
                    .ok_or_else(|| CreamError::InvalidInput("cell matrix is singular".into()))?;
                (h, h_inv, 1u32)
            }
        };

        // ── GpuSimParams ────────────────────────────────────────────────────
        let rc = potential.cutoff();
        let (ncx, ncy, ncz, nct, cs, nx_pad, ny_pad, nz_pad) =
            cl_info.unwrap_or((0, 0, 0, 0, 0.0, 0, 0, 0));
        let params_gpu = GpuSimParams {
            n_atoms: n as u32,
            n_elem: n_elements as u32,
            cutoff_sq: rc * rc,
            min_dist_sq: 1e-4,
            h0: [h[0][0], h[0][1], h[0][2], 0.0],
            h1: [h[1][0], h[1][1], h[1][2], 0.0],
            h2: [h[2][0], h[2][1], h[2][2], 0.0],
            hinv_col0: [hinv[0][0], hinv[1][0], hinv[2][0], 0.0],
            hinv_col1: [hinv[0][1], hinv[1][1], hinv[2][1], 0.0],
            hinv_col2: [hinv[0][2], hinv[1][2], hinv[2][2], 0.0],
            use_cell_list: u32::from(cl_info.is_some()),
            use_pbc,
            _pad0: 0,
            _pad1: 0,
            n_cells_x: ncx,
            n_cells_y: ncy,
            n_cells_z: ncz,
            n_cells_total: nct,
            cell_size: cs,
            n_cells_x_pad: nx_pad,
            n_cells_y_pad: ny_pad,
            n_cells_z_pad: nz_pad,
            // i_offset / i_count are set per-chunk in Submission B via
            // copy_buffer_to_buffer; zero-initialise here as the baseline.
            i_offset: 0,
            i_count: n as u32,
            _pad2: 0,
            _pad3: 0,
        };

        // ── Potential table upload ──────────────────────────────────────────
        let key = potential.cache_key();
        if !self.table_cache.contains_key(&key) {
            let bufs = potential.upload_tables(&self.device, &self.queue)?;
            self.table_cache.insert(key.clone(), Arc::new(bufs));
        }
        let table_bufs = Arc::clone(self.table_cache.get(&key).unwrap());
        let layout = potential.buffer_layout();

        // ── Frame buffer management ─────────────────────────────────────────
        let need_new_frame = self.frame_cache.as_ref().map_or(true, |f| f.n != n);
        if need_new_frame {
            self.frame_cache = Some(Self::create_frame_buffers(&self.device, n, &layout));
        }
        let mut fb = self.frame_cache.take().unwrap();

        // ── Upload per-frame data ───────────────────────────────────────────
        if !fb.positions_current_on_gpu {
            self.queue
                .write_buffer(&fb.pos_buf, 0, bytemuck::cast_slice(positions));
        }
        self.queue
            .write_buffer(&fb.types_buf, 0, bytemuck::cast_slice(atom_types));
        self.queue
            .write_buffer(&fb.params_buf, 0, bytemuck::bytes_of(&params_gpu));

        // ── Cell List: Submission A ─────────────────────────────────────────
        //
        // Per-phase wall-clock timing: record `Instant`s at phase boundaries
        // so `compute_with_debug` can report where time is spent.  Only
        // meaningful on the CellList path (AllPairs skips Submission A
        // entirely).  These are no-ops under `cfg(target_arch = "wasm32")`
        // because `std::time::Instant` is unreliable or unimplemented in
        // some WASM environments.
        #[cfg(not(target_arch = "wasm32"))]
        let t_phase_start = Instant::now();
        // Per-phase timing locals (valid only on the CellList path; `None` otherwise).
        #[cfg(not(target_arch = "wasm32"))]
        let mut t_after_a: Option<Instant> = None;
        #[cfg(not(target_arch = "wasm32"))]
        let mut t_after_cpu: Option<Instant> = None;

        if let Some((ncx2, ncy2, ncz2, nct2, cs2, nx_pad2, ny_pad2, nz_pad2)) = cl_info {
            // Reuse cell-list buffers only when the full per-axis grid matches.
            //
            // Comparing `n_cells_total` (the product) alone is NOT sufficient:
            // different (ncx, ncy, ncz) factorisations with the same product
            // produce different Morton layouts, padding, and buffer sizes.
            // Example: nct = 8 matches (2,2,2) and (1,2,4), but n_morton is
            // 8 vs 16 and the shader's bb loop iterates different ranges.
            // Reusing the wrong buffers causes silent OOB writes in cell_start
            // and miscomputed densities/forces.
            //
            // We also compare the padded axes because they drive Morton bit
            // widths; even if (ncx, ncy, ncz) match, a padding change would
            // still invalidate the buffers.
            let need_new_cl = fb.cell_list.as_ref().map_or(true, |cl| {
                cl.n_atoms != n
                    || cl.n_cells_total != nct2
                    || cl.n_cells_x != ncx2
                    || cl.n_cells_y != ncy2
                    || cl.n_cells_z != ncz2
                    || cl.n_cells_x_pad != nx_pad2
                    || cl.n_cells_y_pad != ny_pad2
                    || cl.n_cells_z_pad != nz_pad2
            });
            if need_new_cl {
                fb.cell_list = Some(Self::create_cell_list_buffers(
                    &self.device,
                    n,
                    ncx2,
                    ncy2,
                    ncz2,
                    nct2,
                    cs2,
                    nx_pad2,
                    ny_pad2,
                    nz_pad2,
                ));
                fb.bg_pass0a = None;
                fb.bg_pass0b = None;
                fb.bg_pass0c = None;
                fb.bg_pass0d = None;
                // Invalidate pass1/2 BGs: they reference CL buffers that were just recreated.
                fb.bg_pass1 = None;
                fb.bg_pass2 = None;
            }

            Self::ensure_fixed_pipeline(
                &self.device,
                &mut self.pl_pass0a,
                Self::PASS0A_SRC,
                BGL_PASS0A,
            );
            Self::ensure_fixed_pipeline(
                &self.device,
                &mut self.pl_pass0b,
                Self::PASS0B_SRC,
                BGL_PASS0B,
            );
            Self::ensure_fixed_pipeline(
                &self.device,
                &mut self.pl_pass0c,
                Self::PASS0C_SRC,
                BGL_PASS0C,
            );

            let cl = fb.cell_list.as_ref().unwrap();
            let n_morton = cl.n_morton;

            if fb.bg_pass0a.is_none() {
                let bgl = Self::make_bgl(&self.device, BGL_PASS0A);
                fb.bg_pass0a = Some(Self::make_bind_group(
                    &self.device,
                    &bgl,
                    &[(0, &fb.pos_buf), (2, &fb.params_buf), (5, &cl.cell_ids_buf)],
                ));
            }
            if fb.bg_pass0b.is_none() {
                let bgl = Self::make_bgl(&self.device, BGL_PASS0B);
                fb.bg_pass0b = Some(Self::make_bind_group(
                    &self.device,
                    &bgl,
                    &[
                        (0, &fb.pos_buf),
                        (2, &fb.params_buf),
                        (5, &cl.cell_ids_buf),
                        (6, &cl.cell_counts_buf),
                    ],
                ));
            }
            if fb.bg_pass0c.is_none() {
                let bgl = Self::make_bgl(&self.device, BGL_PASS0C);
                fb.bg_pass0c = Some(Self::make_bind_group(
                    &self.device,
                    &bgl,
                    &[
                        (0, &fb.pos_buf),
                        (2, &fb.params_buf),
                        (5, &cl.cell_ids_buf),
                        (6, &cl.write_offsets_buf),
                        (7, &cl.sorted_atoms_buf),
                    ],
                ));
            }

            // ── Pass 0d: reorder positions/types into Morton order ───────────
            Self::ensure_fixed_pipeline(
                &self.device,
                &mut self.pl_pass0d,
                Self::PASS0D_SRC,
                BGL_PASS0D,
            );
            if fb.bg_pass0d.is_none() {
                let bgl = Self::make_bgl(&self.device, BGL_PASS0D);
                fb.bg_pass0d = Some(Self::make_bind_group(
                    &self.device,
                    &bgl,
                    &[
                        (0, &fb.pos_buf),
                        (1, &fb.types_buf),
                        (2, &fb.params_buf),
                        (5, &cl.sorted_atoms_buf),
                        (6, &cl.reordered_pos_buf),
                        (7, &cl.reordered_types_buf),
                        (8, &cl.cell_ids_buf),
                    ],
                ));
            }

            // Submission A: clear + pass0a + pass0b + copy counts.
            //
            // Pass0a/0b must be chunk-dispatched for N exceeding
            // MAX_DISPATCH_ATOMS (= 65_535 × 64 = 4_194_240) because WebGPU
            // caps each dispatch dimension at 65 535 workgroups.  We use the
            // same per-chunk params-patching pattern as Submission B.
            //
            // For typical N ≪ 4M this loop runs exactly once (n_chunks == 1)
            // and is indistinguishable from the previous single-dispatch
            // code, so there is no overhead for small systems.
            const MAX_DISPATCH_ATOMS_A: usize = 65_535 * 64;
            let n_chunks_a = n.div_ceil(MAX_DISPATCH_ATOMS_A);
            let params_stride_a = std::mem::size_of::<GpuSimParams>() as u64;

            // Per-chunk params staging: only (i_offset, i_count) differ from params_gpu.
            let chunk_params_data_a: Vec<GpuSimParams> = (0..n_chunks_a)
                .map(|c| {
                    let i_off = (c * MAX_DISPATCH_ATOMS_A) as u32;
                    let i_cnt = (n - c * MAX_DISPATCH_ATOMS_A).min(MAX_DISPATCH_ATOMS_A) as u32;
                    GpuSimParams {
                        i_offset: i_off,
                        i_count: i_cnt,
                        ..params_gpu
                    }
                })
                .collect();
            let chunk_staging_a =
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("chunk_params_staging_a"),
                        contents: bytemuck::cast_slice(&chunk_params_data_a),
                        usage: wgpu::BufferUsages::COPY_SRC,
                    });

            {
                let mut enc_a = self.device.create_command_encoder(&Default::default());
                enc_a.clear_buffer(&cl.cell_counts_buf, 0, None);

                // Pass0a (Morton ID assignment) — chunk-dispatched.
                for c in 0..n_chunks_a {
                    let i_cnt = (n - c * MAX_DISPATCH_ATOMS_A).min(MAX_DISPATCH_ATOMS_A);
                    enc_a.copy_buffer_to_buffer(
                        &chunk_staging_a,
                        c as u64 * params_stride_a,
                        &fb.params_buf,
                        0,
                        params_stride_a,
                    );
                    Self::encode_dispatch(
                        &mut enc_a,
                        self.pl_pass0a.as_ref().unwrap(),
                        fb.bg_pass0a.as_ref().unwrap(),
                        i_cnt,
                    );
                }
                // Pass0b (atomic count per cell) — chunk-dispatched.
                for c in 0..n_chunks_a {
                    let i_cnt = (n - c * MAX_DISPATCH_ATOMS_A).min(MAX_DISPATCH_ATOMS_A);
                    enc_a.copy_buffer_to_buffer(
                        &chunk_staging_a,
                        c as u64 * params_stride_a,
                        &fb.params_buf,
                        0,
                        params_stride_a,
                    );
                    Self::encode_dispatch(
                        &mut enc_a,
                        self.pl_pass0b.as_ref().unwrap(),
                        fb.bg_pass0b.as_ref().unwrap(),
                        i_cnt,
                    );
                }
                enc_a.copy_buffer_to_buffer(
                    &cl.cell_counts_buf,
                    0,
                    &cl.rb_cell_counts,
                    0,
                    4 * n_morton as u64,
                );
                self.queue.submit([enc_a.finish()]);
            }

            // Poll 1: wait for Submission A.
            #[cfg(not(target_arch = "wasm32"))]
            self.device.poll(wgpu::Maintain::Wait);
            #[cfg(not(target_arch = "wasm32"))]
            {
                t_after_a = Some(Instant::now());
            }

            // CPU: map counts, prefix sum, re-upload.
            let cell_counts = {
                let slice = cl.rb_cell_counts.slice(..);
                let done = Arc::new(AtomicBool::new(false));
                let done2 = done.clone();
                slice.map_async(wgpu::MapMode::Read, move |_| {
                    done2.store(true, Ordering::Release);
                });
                #[cfg(not(target_arch = "wasm32"))]
                {
                    self.device.poll(wgpu::Maintain::Wait);
                }
                #[cfg(target_arch = "wasm32")]
                while !done.load(Ordering::Acquire) {
                    let p = JsPromise::resolve(&JsValue::UNDEFINED);
                    let _ = JsFuture::from(p).await;
                    self.device.poll(wgpu::Maintain::Poll);
                }
                let mapped = slice.get_mapped_range();
                let v: Vec<u32> = bytemuck::cast_slice(&mapped).to_vec();
                drop(mapped);
                cl.rb_cell_counts.unmap();
                v
            };

            let mut cell_start = vec![0u32; n_morton + 1];
            for m in 0..n_morton {
                cell_start[m + 1] = cell_start[m] + cell_counts[m];
            }
            // ALWAYS verify prefix sum — debug_assert is stripped in release.
            assert_eq!(
                cell_start[n_morton], n as u32,
                "CellList prefix sum mismatch: cell_start[{}]={} but n_atoms={}",
                n_morton, cell_start[n_morton], n
            );

            let cl = fb.cell_list.as_ref().unwrap();
            self.queue
                .write_buffer(&cl.cell_start_buf, 0, bytemuck::cast_slice(&cell_start));
            self.queue.write_buffer(
                &cl.write_offsets_buf,
                0,
                bytemuck::cast_slice(&cell_start[..n_morton]),
            );
            #[cfg(not(target_arch = "wasm32"))]
            {
                t_after_cpu = Some(Instant::now());
            }
        }

        // ── CPU-built neighbour list: build and upload ──────────────────────
        // When `NeighborStrategy::CellList` is active and the `cellist_gpu`
        // feature is NOT enabled, build a CSR neighbour list on the CPU and
        // upload it to the GPU.  The GPU pass0 construction shaders are
        // skipped; pass1/pass2 use the NeighborList shaders which iterate
        // directly over the flat CSR list.
        //
        // When `cellist_gpu` IS enabled, this block is omitted and the full
        // GPU cell-list pipeline (`pass0a..pass0d` + `pass1/pass2_cellist`)
        // runs instead, eliminating the CPU-side neighbour-list build at
        // the cost of a more complex GPU dispatch.
        #[cfg(not(feature = "cellist_gpu"))]
        if cl_info.is_some() {
            let h_mat = cell.unwrap(); // CellList requires Some(cell), checked above
            let nl = crate::neighbor_list::build(
                positions,
                Some(h_mat),
                potential.cutoff(),
                0.01 * potential.cutoff(), // skin for f32 boundary safety
            );
            let list_len = nl.list.len() as u32;
            let starts_len = nl.starts.len() as u32;

            // Allocate / grow GPU buffers as needed.
            let need_new = match &fb.neighbor_list {
                None => true,
                Some(nlb) => nlb.list_capacity < list_len || nlb.n_atoms != n,
            };
            if need_new {
                let starts_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("cream::nl_starts"),
                    size: (starts_len.max(1) as u64) * 4,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                // Grow with ~25 % headroom to avoid per-step reallocation.
                let cap = (list_len as f32 * 1.25) as u32 + 64;
                let list_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("cream::nl_list"),
                    size: (cap.max(1) as u64) * 4,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let dbg_buf = Self::alloc_storage(&self.device, 4 * 32);
                fb.neighbor_list = Some(NeighborListBuffers {
                    n_atoms: n,
                    list_capacity: cap,
                    nl_starts_buf: starts_buf,
                    nl_list_buf: list_buf,
                    debug_flags_buf: dbg_buf,
                });
                // Invalidate bind groups that reference NL buffers.
                fb.bg_pass1 = None;
                fb.bg_pass2 = None;
            }
            let nlb = fb.neighbor_list.as_ref().unwrap();
            self.queue
                .write_buffer(&nlb.nl_starts_buf, 0, bytemuck::cast_slice(&nl.starts));
            if !nl.list.is_empty() {
                self.queue
                    .write_buffer(&nlb.nl_list_buf, 0, bytemuck::cast_slice(&nl.list));
            }
        }

        // ── Build pass1/2 pipelines and BindGroups ──────────────────────────
        // Shader / layout selection depends on how the cell list (if any)
        // is being executed:
        //
        //   cl_info.is_none()                         → AllPairs path
        //                                              (pass1/pass2 _ap)
        //   cl_info.is_some() && !cellist_gpu feature → CPU-built CSR list
        //                                              (pass1/pass2 _nl)
        //   cl_info.is_some() &&  cellist_gpu feature → GPU cell-list path
        //                                              (pass1/pass2 _cl,
        //                                              reads the Morton-
        //                                              reordered buffers
        //                                              produced by pass0d)
        //
        // `cellist_gpu` is gated behind a cargo feature because the GPU
        // cell-list path requires the pass0a..pass0d construction pipeline
        // and carries more GPU state.  The default CPU-built neighbour-list
        // path works on any wgpu backend with only compute-storage support.
        let cellist_gpu = cfg!(feature = "cellist_gpu") && cl_info.is_some();
        let pass1_src = if cellist_gpu {
            potential.pass1_cellist_shader().ok_or_else(|| {
                CreamError::InvalidInput(
                    "Potential has no CellList pass1 shader (cellist_gpu feature)".into(),
                )
            })?
        } else if cl_info.is_some() {
            potential.pass1_neighlist_shader().ok_or_else(|| {
                CreamError::InvalidInput(
                    "Potential has no CPU-built NeighborList pass1 shader".into(),
                )
            })?
        } else {
            potential.pass1_shader()
        };
        let pass2_src = if cellist_gpu {
            potential.pass2_cellist_shader().ok_or_else(|| {
                CreamError::InvalidInput(
                    "Potential has no CellList pass2 shader (cellist_gpu feature)".into(),
                )
            })?
        } else if cl_info.is_some() {
            potential.pass2_neighlist_shader().ok_or_else(|| {
                CreamError::InvalidInput(
                    "Potential has no CPU-built NeighborList pass2 shader".into(),
                )
            })?
        } else {
            potential.pass2_shader()
        };

        // `pipeline_cache` is temporarily removed so we can borrow `self.device`
        // and `self.queue` without conflicting with `self.pipeline_cache`.
        let mut pipeline_cache = std::mem::take(&mut self.pipeline_cache);

        // ── Setup: insert pipelines and build BindGroups ────────────────────
        // `pipeline_cache` is mutated here; hash keys are saved so the later
        // dispatch section can look up immutable refs without conflicting
        // with further HashMap mutations.

        let h1 = Self::hash_str(&pass1_src);
        {
            let bgl_e1 = if cellist_gpu {
                BGL_PASS1_CL
            } else if cl_info.is_some() {
                BGL_PASS1_NL
            } else {
                BGL_PASS1_AP
            };
            pipeline_cache
                .entry(h1)
                .or_insert_with(|| Self::build_explicit_pipeline(&self.device, &pass1_src, bgl_e1));
            if fb.bg_pass1.as_ref().map(|(h, _)| *h) != Some(h1) {
                let bgl = Self::make_bgl(&self.device, bgl_e1);
                if cellist_gpu {
                    // CellList GPU: binding 0/1 are Morton-reordered pos/types
                    // (written by pass0d).  Binding 6 is cell_start.  Binding
                    // 10 is the debug_flags counter buffer.
                    let cl = fb.cell_list.as_ref().unwrap();
                    let slots: &[(u32, &wgpu::Buffer)] = &[
                        (0, &cl.reordered_pos_buf),
                        (1, &cl.reordered_types_buf),
                        (2, &fb.params_buf),
                        (3, &table_bufs.tables_buf),
                        (4, &table_bufs.layout_buf),
                        (5, &fb.density_buf),
                        (6, &cl.cell_start_buf),
                        (10, &cl.debug_flags_buf),
                    ];
                    let bg = Self::make_bind_group(&self.device, &bgl, slots);
                    fb.bg_pass1 = Some((h1, bg));
                } else if cl_info.is_some() {
                    // CPU-built NeighborList: original pos/types at 0/1; NL buffers at 6/7.
                    let nlb = fb.neighbor_list.as_ref().unwrap();
                    let slots: &[(u32, &wgpu::Buffer)] = &[
                        (0, &fb.pos_buf),
                        (1, &fb.types_buf),
                        (2, &fb.params_buf),
                        (3, &table_bufs.tables_buf),
                        (4, &table_bufs.layout_buf),
                        (5, &fb.density_buf),
                        (6, &nlb.nl_starts_buf),
                        (7, &nlb.nl_list_buf),
                        (10, &nlb.debug_flags_buf),
                    ];
                    let bg = Self::make_bind_group(&self.device, &bgl, slots);
                    fb.bg_pass1 = Some((h1, bg));
                } else {
                    // AllPairs: original pos/types buffers.
                    let slots: &[(u32, &wgpu::Buffer)] = &[
                        (0, &fb.pos_buf),
                        (1, &fb.types_buf),
                        (2, &fb.params_buf),
                        (3, &table_bufs.tables_buf),
                        (4, &table_bufs.layout_buf),
                        (5, &fb.density_buf),
                    ];
                    let bg = Self::make_bind_group(&self.device, &bgl, slots);
                    fb.bg_pass1 = Some((h1, bg));
                }
            }
        }

        let h2 = Self::hash_str(&pass2_src);
        {
            let bgl_e2 = if cellist_gpu {
                BGL_PASS2_CL
            } else if cl_info.is_some() {
                BGL_PASS2_NL
            } else {
                BGL_PASS2_AP
            };
            pipeline_cache
                .entry(h2)
                .or_insert_with(|| Self::build_explicit_pipeline(&self.device, &pass2_src, bgl_e2));
            if fb.bg_pass2.as_ref().map(|(h, _)| *h) != Some(h2) {
                let bgl = Self::make_bgl(&self.device, bgl_e2);
                if cellist_gpu {
                    // CellList GPU pass 2 layout:
                    //   0 = reordered_pos (Morton order, xyz + w=cell_id)
                    //   1 = reordered_types
                    //   5 = densities_in (Morton order, written by pass1_cl)
                    //   6 = forces_out  (scatter-written via sorted_atoms to
                    //                    original index → consumed by Verlet)
                    //   7 = wg_energy_out (partial sums per workgroup)
                    //   8 = sorted_atoms (k → original index j, for scatter)
                    //   9 = cell_start
                    //   10 = debug_flags
                    //   11 = wg_virial_out (6 × f32 per workgroup)
                    let cl = fb.cell_list.as_ref().unwrap();
                    let slots: &[(u32, &wgpu::Buffer)] = &[
                        (0, &cl.reordered_pos_buf),
                        (1, &cl.reordered_types_buf),
                        (2, &fb.params_buf),
                        (3, &table_bufs.tables_buf),
                        (4, &table_bufs.layout_buf),
                        (5, &fb.density_buf),
                        (6, &fb.forces_buf),
                        (7, &fb.wg_energy_buf),
                        (8, &cl.sorted_atoms_buf),
                        (9, &cl.cell_start_buf),
                        (10, &cl.debug_flags_buf),
                        (11, &fb.wg_virial_buf),
                    ];
                    let bg = Self::make_bind_group(&self.device, &bgl, slots);
                    fb.bg_pass2 = Some((h2, bg));
                } else if cl_info.is_some() {
                    // CPU-built NeighborList: original pos/types at 0/1; NL buffers at 8/9.
                    let nlb = fb.neighbor_list.as_ref().unwrap();
                    let slots: &[(u32, &wgpu::Buffer)] = &[
                        (0, &fb.pos_buf),
                        (1, &fb.types_buf),
                        (2, &fb.params_buf),
                        (3, &table_bufs.tables_buf),
                        (4, &table_bufs.layout_buf),
                        (5, &fb.density_buf),
                        (6, &fb.forces_buf),
                        (7, &fb.wg_energy_buf),
                        (8, &nlb.nl_starts_buf),
                        (9, &nlb.nl_list_buf),
                        (10, &nlb.debug_flags_buf),
                        (11, &fb.wg_virial_buf),
                    ];
                    let bg = Self::make_bind_group(&self.device, &bgl, slots);
                    fb.bg_pass2 = Some((h2, bg));
                } else {
                    // AllPairs: original pos/types buffers.
                    let slots: &[(u32, &wgpu::Buffer)] = &[
                        (0, &fb.pos_buf),
                        (1, &fb.types_buf),
                        (2, &fb.params_buf),
                        (3, &table_bufs.tables_buf),
                        (4, &table_bufs.layout_buf),
                        (5, &fb.density_buf),
                        (6, &fb.forces_buf),
                        (7, &fb.wg_energy_buf),
                        (11, &fb.wg_virial_buf),
                    ];
                    let bg = Self::make_bind_group(&self.device, &bgl, slots);
                    fb.bg_pass2 = Some((h2, bg));
                }
            }
        }

        // Pass 3 pipeline (shared for 3a/3b/3c)
        const REDUCE_SRC: &str = include_str!("shaders/eam_pass3_reduce.wgsl");
        Self::ensure_fixed_pipeline(&self.device, &mut self.pl_pass3, REDUCE_SRC, BGL_PASS3);

        // 3-level GPU reduction counts (CPU handles the final sum):
        //   Level 0 (in pass2): N    → num_wg
        //   Level 1 (pass3a):   num_wg → lvl1_count
        //   Level 2 (pass3b):   lvl1_count → lvl2_count  (CPU-summed)
        let num_wg = n.div_ceil(64); // number of wg partial sums from pass2
        let lvl1_count = num_wg.div_ceil(64);
        let lvl2_count = lvl1_count.div_ceil(64);

        let rp_a = ReduceParams {
            count: num_wg as u32,
            _p0: 0,
            _p1: 0,
            _p2: 0,
        };
        let rp_b = ReduceParams {
            count: lvl1_count as u32,
            _p0: 0,
            _p1: 0,
            _p2: 0,
        };
        self.queue
            .write_buffer(&fb.reduce_params_a_buf, 0, bytemuck::bytes_of(&rp_a));
        self.queue
            .write_buffer(&fb.reduce_params_b_buf, 0, bytemuck::bytes_of(&rp_b));

        if fb.bg_pass3a.is_none() {
            let bgl = Self::make_bgl(&self.device, BGL_PASS3);
            fb.bg_pass3a = Some(Self::make_bind_group(
                &self.device,
                &bgl,
                &[
                    (0, &fb.wg_energy_buf),
                    (1, &fb.lvl1_buf),
                    (2, &fb.reduce_params_a_buf),
                ],
            ));
        }
        if fb.bg_pass3b.is_none() {
            let bgl = Self::make_bgl(&self.device, BGL_PASS3);
            fb.bg_pass3b = Some(Self::make_bind_group(
                &self.device,
                &bgl,
                &[
                    (0, &fb.lvl1_buf),
                    (1, &fb.lvl2_buf),
                    (2, &fb.reduce_params_b_buf),
                ],
            ));
        }

        // Verlet BindGroup (built once per N)
        if verlet.is_some() {
            const VERLET_SRC: &str = include_str!("shaders/verlet_integrate.wgsl");
            Self::ensure_fixed_pipeline(&self.device, &mut self.pl_verlet, VERLET_SRC, BGL_VERLET);
            if fb.bg_verlet.is_none() {
                let bgl = Self::make_bgl(&self.device, BGL_VERLET);
                fb.bg_verlet = Some(Self::make_bind_group(
                    &self.device,
                    &bgl,
                    &[
                        (0, &fb.pos_buf),
                        (1, &fb.velocities_buf),
                        (2, &fb.forces_buf),
                        (3, &fb.verlet_params_buf),
                    ],
                ));
            }
            if let Some(vp) = verlet {
                self.queue
                    .write_buffer(&fb.verlet_params_buf, 0, bytemuck::bytes_of(&vp));
            }
        }

        // ── Submission B: pass0c + pass1/2/3 + [verlet] + readbacks ─────────
        //
        // The BindGroup-building setup above is complete.  No further
        // HashMap insertions happen, so immutable refs into `pipeline_cache`
        // are safe here.
        {
            let pl1 = pipeline_cache.get(&h1).unwrap();
            let pl2 = pipeline_cache.get(&h2).unwrap();

            let mut enc_b = self.device.create_command_encoder(&Default::default());

            // Clear debug-flag counters so each frame starts fresh.
            // CPU-built NL path: use the NL debug_flags_buf (CL buffers are not dispatched).
            if let Some(nlb) = fb.neighbor_list.as_ref() {
                enc_b.clear_buffer(&nlb.debug_flags_buf, 0, None);
            } else if let Some(cl) = fb.cell_list.as_ref() {
                enc_b.clear_buffer(&cl.debug_flags_buf, 0, None);
            }

            // ── Per-chunk params staging ──────────────────────────────────────
            //
            // Shared by pass1 and pass2 (pass0c/0d are skipped on the CPU-built NL path).
            // WebGPU guarantees at most 65 535 workgroups per dimension.
            // With 64 threads per workgroup, the per-chunk atom limit is
            // 65_535 × 64 = 4_194_240.  For typical MD system sizes (N ≪ 4M)
            // n_chunks == 1 and there is no overhead.
            //
            // Strategy: pre-build a COPY_SRC staging buffer that holds one
            // GpuSimParams per chunk (only i_offset / i_count differ).
            // Before each dispatch, copy_buffer_to_buffer patches params_buf
            // in-place.  All copies and dispatches stay in a single encoder
            // so there is only one queue.submit() for Submission B.
            //
            // Memory ordering: WebGPU automatically tracks the COPY_DST →
            // UNIFORM dependency within the same command buffer, so no
            // explicit pipeline barrier is required.
            //
            // All pass-1 chunks run to completion before any pass-2 chunk
            // starts, because encode_dispatch ends its compute pass and
            // begins a new one — inter-pass ordering is guaranteed by the
            // WebGPU spec.
            const MAX_DISPATCH_ATOMS: usize = 65_535 * 64; // 4_194_240
            let n_chunks = n.div_ceil(MAX_DISPATCH_ATOMS);
            let params_stride = std::mem::size_of::<GpuSimParams>() as u64;

            // Build per-chunk params and upload to a temporary staging buffer.
            let chunk_params_data: Vec<GpuSimParams> = (0..n_chunks)
                .map(|c| {
                    let i_off = (c * MAX_DISPATCH_ATOMS) as u32;
                    let i_cnt = (n - c * MAX_DISPATCH_ATOMS).min(MAX_DISPATCH_ATOMS) as u32;
                    GpuSimParams {
                        i_offset: i_off,
                        i_count: i_cnt,
                        ..params_gpu
                    }
                })
                .collect();
            let chunk_staging = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("chunk_params_staging"),
                    contents: bytemuck::cast_slice(&chunk_params_data),
                    usage: wgpu::BufferUsages::COPY_SRC,
                });

            // When `cellist_gpu` is active, pass0c (scatter atoms into
            // sorted_atoms[] using per-cell atomic write cursors) and pass0d
            // (reorder positions/types into Morton order, and embed cell IDs
            // in the W component of reordered_positions) run here in
            // Submission B.  They must dispatch before pass1_cellist reads
            // from `reordered_pos_buf` / `reordered_types_buf` / `cell_start_buf`.
            //
            // CPU-built NL path: pass0c / pass0d (Morton scatter + reorder) are skipped.
            // The CPU-built CSR neighbour list has already been uploaded above,
            // and the NL shaders read the original pos_buf / types_buf directly.
            //
            // ── TDR-safe dispatch strategy ────────────────────────────────────
            //
            // n_chunks == 1  (N ≤ 4_194_240, typical MD sizes):
            //   All passes stay in enc_b — single queue.submit(), no behaviour
            //   change vs. previous releases.
            //
            // n_chunks > 1  (N > 4_194_240):
            //   enc_b carries only the debug-flag clear and is flushed first.
            //   Every subsequent pass is dispatched one chunk at a time in its
            //   own CommandEncoder, immediately followed by device.poll(Wait).
            //   This caps each GPU submission to ≤ 4_194_240 atoms (~400 ms on
            //   the target hardware), well below the Windows TDR watchdog (2 s)
            //   for any N — including theoretically unbounded system sizes.
            //
            // Correctness of the per-chunk split:
            //   pass0c  — uses atomic write cursors; sequential chunk submission
            //             with poll(Wait) ensures each chunk sees the cursor
            //             state left by the previous one.  Total scatter result
            //             is identical to a single monolithic dispatch.
            //   pass0d  — reads sorted_atoms[] (fully committed after all
            //             pass0c polls).  Chunks write non-overlapping
            //             reordered_pos[i_offset .. i_offset+i_cnt] ranges;
            //             no write conflicts between chunks.
            //   pass1   — each chunk writes densities[i_offset .. i_offset+i_cnt]
            //             only.  No intra-pass write conflicts.  poll(Wait) after
            //             the last pass1 chunk ensures all density values are
            //             visible in GPU memory before any pass2 chunk starts.
            //   pass2   — each chunk writes forces[i_offset..] and
            //             wg_energy_buf[(i_offset/64)..]; non-overlapping ranges.
            //             poll(Wait) after each chunk bounds submission time.
            //   pass3   — tree-reduces the complete wg_energy_buf (fully written
            //             after all pass2 polls).  Result is bit-identical to
            //             the single-encoder path.
            //
            // Overhead: (3 × n_chunks + 1) extra poll(Wait) calls.
            // At N = 20 M (5 chunks) and ~2 ms per poll: ≈ 30 ms added to
            // a ~4 s computation — under 1 %.

            let bg1 = &fb.bg_pass1.as_ref().unwrap().1;
            let bg2 = &fb.bg_pass2.as_ref().unwrap().1;
            let virial_bytes = 4u64 * 6 * num_wg as u64;

            // Verlet is not chunked (it uses VerletParams, not GpuSimParams).
            // Guard here so the error fires regardless of which branch runs.
            if verlet.is_some() && n > MAX_DISPATCH_ATOMS {
                return Err(CreamError::InvalidInput(format!(
                    "Verlet integration currently supports at most {MAX_DISPATCH_ATOMS} \
                     atoms per step (N = {n} requested).  Split the simulation into \
                     sub-steps or run without Verlet (compute forces only)."
                )));
            }

            if n_chunks == 1 {
                // ── Single-encoder path (N ≤ 4_194_240) ─────────────────────
                if cellist_gpu {
                    // pass0c: scatter atoms into Morton cells
                    for c in 0..n_chunks {
                        let i_cnt = (n - c * MAX_DISPATCH_ATOMS).min(MAX_DISPATCH_ATOMS);
                        enc_b.copy_buffer_to_buffer(
                            &chunk_staging, c as u64 * params_stride,
                            &fb.params_buf, 0, params_stride,
                        );
                        Self::encode_dispatch(
                            &mut enc_b,
                            self.pl_pass0c.as_ref().unwrap(),
                            fb.bg_pass0c.as_ref().unwrap(),
                            i_cnt,
                        );
                    }
                    // pass0d: reorder positions/types into Morton order
                    for c in 0..n_chunks {
                        let i_cnt = (n - c * MAX_DISPATCH_ATOMS).min(MAX_DISPATCH_ATOMS);
                        enc_b.copy_buffer_to_buffer(
                            &chunk_staging, c as u64 * params_stride,
                            &fb.params_buf, 0, params_stride,
                        );
                        Self::encode_dispatch(
                            &mut enc_b,
                            self.pl_pass0d.as_ref().unwrap(),
                            fb.bg_pass0d.as_ref().unwrap(),
                            i_cnt,
                        );
                    }
                }

                // Pass 1 — density
                for c in 0..n_chunks {
                    let i_cnt = (n - c * MAX_DISPATCH_ATOMS).min(MAX_DISPATCH_ATOMS);
                    enc_b.copy_buffer_to_buffer(
                        &chunk_staging, c as u64 * params_stride,
                        &fb.params_buf, 0, params_stride,
                    );
                    Self::encode_dispatch(&mut enc_b, pl1, bg1, i_cnt);
                }

                // Pass 2 — forces + per-WG energy partials
                for c in 0..n_chunks {
                    let i_cnt = (n - c * MAX_DISPATCH_ATOMS).min(MAX_DISPATCH_ATOMS);
                    enc_b.copy_buffer_to_buffer(
                        &chunk_staging, c as u64 * params_stride,
                        &fb.params_buf, 0, params_stride,
                    );
                    Self::encode_dispatch(&mut enc_b, pl2, bg2, i_cnt);
                }

                // Pass 3a/b — GPU reduction; pass3c removed (CPU Neumaier sum).
                Self::encode_dispatch(
                    &mut enc_b, self.pl_pass3.as_ref().unwrap(),
                    fb.bg_pass3a.as_ref().unwrap(), num_wg,
                );
                Self::encode_dispatch(
                    &mut enc_b, self.pl_pass3.as_ref().unwrap(),
                    fb.bg_pass3b.as_ref().unwrap(), lvl1_count,
                );

                // Verlet integration (optional; N ≤ MAX_DISPATCH_ATOMS guaranteed above)
                if verlet.is_some() {
                    Self::encode_dispatch(
                        &mut enc_b,
                        self.pl_verlet.as_ref().unwrap(),
                        fb.bg_verlet.as_ref().unwrap(),
                        n,
                    );
                }

                // Readbacks: forces (N × 16 B) + lvl2 energy partials + virial partials.
                enc_b.copy_buffer_to_buffer(
                    &fb.forces_buf, 0, &fb.rb_forces, 0,
                    layout.output_stride_bytes * n as u64,
                );
                enc_b.copy_buffer_to_buffer(
                    &fb.lvl2_buf, 0, &fb.rb_lvl2, 0,
                    4 * lvl2_count as u64,
                );
                // Virial: 6 × f32 per workgroup — tiny vs. force readback.
                enc_b.copy_buffer_to_buffer(
                    &fb.wg_virial_buf, 0, &fb.rb_virial_partials, 0, virial_bytes,
                );

                self.queue.submit([enc_b.finish()]);

            } else {
                // ── Per-chunk TDR-safe path (N > 4_194_240) ──────────────────
                //
                // enc_b currently holds only the debug-flag clear; flush it now
                // so the GPU counter reset happens before any compute work.
                self.queue.submit([enc_b.finish()]);
                #[cfg(not(target_arch = "wasm32"))]
                self.device.poll(wgpu::Maintain::Wait);

                if cellist_gpu {
                    // pass0c — scatter atoms into Morton cells.
                    // Atomic write cursors in write_offsets_buf accumulate across
                    // submissions; chunks must therefore be strictly sequential.
                    for c in 0..n_chunks {
                        let i_cnt = (n - c * MAX_DISPATCH_ATOMS).min(MAX_DISPATCH_ATOMS);
                        let mut enc = self.device.create_command_encoder(&Default::default());
                        enc.copy_buffer_to_buffer(
                            &chunk_staging, c as u64 * params_stride,
                            &fb.params_buf, 0, params_stride,
                        );
                        Self::encode_dispatch(
                            &mut enc,
                            self.pl_pass0c.as_ref().unwrap(),
                            fb.bg_pass0c.as_ref().unwrap(),
                            i_cnt,
                        );
                        self.queue.submit([enc.finish()]);
                        #[cfg(not(target_arch = "wasm32"))]
                        self.device.poll(wgpu::Maintain::Wait);
                    }

                    // pass0d — reorder positions/types into Morton order.
                    // Reads sorted_atoms[] (complete after all pass0c polls above).
                    // Each chunk writes reordered_pos[i_offset..i_offset+i_cnt];
                    // ranges are non-overlapping so chunks are independent.
                    for c in 0..n_chunks {
                        let i_cnt = (n - c * MAX_DISPATCH_ATOMS).min(MAX_DISPATCH_ATOMS);
                        let mut enc = self.device.create_command_encoder(&Default::default());
                        enc.copy_buffer_to_buffer(
                            &chunk_staging, c as u64 * params_stride,
                            &fb.params_buf, 0, params_stride,
                        );
                        Self::encode_dispatch(
                            &mut enc,
                            self.pl_pass0d.as_ref().unwrap(),
                            fb.bg_pass0d.as_ref().unwrap(),
                            i_cnt,
                        );
                        self.queue.submit([enc.finish()]);
                        #[cfg(not(target_arch = "wasm32"))]
                        self.device.poll(wgpu::Maintain::Wait);
                    }
                }

                // pass1 — density accumulation.
                // Each chunk writes densities[i_offset..i_offset+i_cnt] only.
                // poll(Wait) after every chunk bounds GPU submission time.
                // poll(Wait) after the LAST chunk commits all density values to
                // GPU memory, satisfying the pass1→pass2 data dependency.
                for c in 0..n_chunks {
                    let i_cnt = (n - c * MAX_DISPATCH_ATOMS).min(MAX_DISPATCH_ATOMS);
                    let mut enc = self.device.create_command_encoder(&Default::default());
                    enc.copy_buffer_to_buffer(
                        &chunk_staging, c as u64 * params_stride,
                        &fb.params_buf, 0, params_stride,
                    );
                    Self::encode_dispatch(&mut enc, pl1, bg1, i_cnt);
                    self.queue.submit([enc.finish()]);
                    #[cfg(not(target_arch = "wasm32"))]
                    self.device.poll(wgpu::Maintain::Wait);
                }

                // pass2 — forces + per-WG energy partials.
                // Each chunk writes forces[i_offset..] and
                // wg_energy_buf[(i_offset/64)..]; non-overlapping ranges.
                for c in 0..n_chunks {
                    let i_cnt = (n - c * MAX_DISPATCH_ATOMS).min(MAX_DISPATCH_ATOMS);
                    let mut enc = self.device.create_command_encoder(&Default::default());
                    enc.copy_buffer_to_buffer(
                        &chunk_staging, c as u64 * params_stride,
                        &fb.params_buf, 0, params_stride,
                    );
                    Self::encode_dispatch(&mut enc, pl2, bg2, i_cnt);
                    self.queue.submit([enc.finish()]);
                    #[cfg(not(target_arch = "wasm32"))]
                    self.device.poll(wgpu::Maintain::Wait);
                }

                // pass3 + readbacks.
                // wg_energy_buf is fully written after all pass2 polls above.
                // This is the final encoder; it falls through to existing Poll 2.
                {
                    let mut enc = self.device.create_command_encoder(&Default::default());
                    // pass3a/b — GPU reduction; final sum is CPU Neumaier.
                    Self::encode_dispatch(
                        &mut enc, self.pl_pass3.as_ref().unwrap(),
                        fb.bg_pass3a.as_ref().unwrap(), num_wg,
                    );
                    Self::encode_dispatch(
                        &mut enc, self.pl_pass3.as_ref().unwrap(),
                        fb.bg_pass3b.as_ref().unwrap(), lvl1_count,
                    );
                    // Readbacks
                    enc.copy_buffer_to_buffer(
                        &fb.forces_buf, 0, &fb.rb_forces, 0,
                        layout.output_stride_bytes * n as u64,
                    );
                    enc.copy_buffer_to_buffer(
                        &fb.lvl2_buf, 0, &fb.rb_lvl2, 0,
                        4 * lvl2_count as u64,
                    );
                    enc.copy_buffer_to_buffer(
                        &fb.wg_virial_buf, 0, &fb.rb_virial_partials, 0, virial_bytes,
                    );
                    self.queue.submit([enc.finish()]);
                }
            }
        }
        self.pipeline_cache = pipeline_cache;

        // Poll 2: wait for Submission B.
        #[cfg(not(target_arch = "wasm32"))]
        self.device.poll(wgpu::Maintain::Wait);

        // Finalise phase timings (valid only if we took the CellList path).
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.last_timings = match (t_after_a, t_after_cpu) {
                (Some(t_a), Some(t_cpu)) => {
                    let t_b = Instant::now();
                    Some(SubmissionTimings {
                        submission_a_ms: (t_a - t_phase_start).as_secs_f64() * 1000.0,
                        cpu_prefix_ms: (t_cpu - t_a).as_secs_f64() * 1000.0,
                        submission_b_ms: (t_b - t_cpu).as_secs_f64() * 1000.0,
                    })
                }
                _ => None, // AllPairs or error before submission A finished
            };
        }

        // ── Map readback buffers ─────────────────────────────────────────────
        {
            // bit 0 = forces, bit 1 = lvl2 energy partials, bit 2 = virial partials
            let bits = Arc::new(AtomicU8::new(0));
            let b0 = bits.clone();
            fb.rb_forces
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |_| {
                    b0.fetch_or(0b001, Ordering::Release);
                });
            // Map only the populated portion: lvl2_count × 4 B.
            let b1 = bits.clone();
            fb.rb_lvl2
                .slice(..4 * lvl2_count as u64)
                .map_async(wgpu::MapMode::Read, move |_| {
                    b1.fetch_or(0b010, Ordering::Release);
                });
            // Map only the populated portion of the virial buffer
            // (`6 × num_wg × 4 B`).  The buffer's capacity is `ceil(N/64)`-
            // worth, so with a steady N there is never a size mismatch.
            let virial_bytes = 4u64 * 6 * num_wg as u64;
            let b2 = bits.clone();
            fb.rb_virial_partials
                .slice(..virial_bytes)
                .map_async(wgpu::MapMode::Read, move |_| {
                    b2.fetch_or(0b100, Ordering::Release);
                });

            #[cfg(not(target_arch = "wasm32"))]
            self.device.poll(wgpu::Maintain::Wait);

            #[cfg(target_arch = "wasm32")]
            while bits.load(Ordering::Acquire) != 0b111 {
                let p = JsPromise::resolve(&JsValue::UNDEFINED);
                let _ = JsFuture::from(p).await;
                self.device.poll(wgpu::Maintain::Poll);
            }
        }

        let forces = Self::read_forces(&fb.rb_forces, n)?;
        // Neumaier-compensated CPU sum of lvl2_count GPU partials.
        let energy = Self::read_and_sum_lvl2(&fb.rb_lvl2, lvl2_count)?;
        // Virial: read `6 × num_wg` f32 partials and compute `W = Σ partials`
        // with Neumaier-compensated summation per component (f64 target).
        // Then `σ = −W / V` for PBC systems; zeros for clusters.
        let virial = Self::read_and_finalize_virial(&fb.rb_virial_partials, num_wg, cell)?;

        // ── CREAM_DIAG: CellList intermediate readback ──────────────────────
        #[cfg(not(target_arch = "wasm32"))]
        if cl_info.is_some() && std::env::var("CREAM_DIAG").is_ok() {
            let cl = fb.cell_list.as_ref().unwrap();

            // Helper: one-shot GPU→CPU readback of a storage buffer.
            let readback_u32 = |buf: &wgpu::Buffer, count: usize| -> Vec<u32> {
                let rb = Self::alloc_readback(&self.device, 4 * count as u64);
                let mut enc = self.device.create_command_encoder(&Default::default());
                enc.copy_buffer_to_buffer(buf, 0, &rb, 0, 4 * count as u64);
                self.queue.submit([enc.finish()]);
                self.device.poll(wgpu::Maintain::Wait);
                let slice = rb.slice(..);
                slice.map_async(wgpu::MapMode::Read, |_| {});
                self.device.poll(wgpu::Maintain::Wait);
                let mapped = slice.get_mapped_range();
                let v: Vec<u32> = bytemuck::cast_slice(&mapped).to_vec();
                drop(mapped);
                rb.unmap();
                v
            };
            let readback_f32 = |buf: &wgpu::Buffer, count: usize| -> Vec<f32> {
                let rb = Self::alloc_readback(&self.device, 4 * count as u64);
                let mut enc = self.device.create_command_encoder(&Default::default());
                enc.copy_buffer_to_buffer(buf, 0, &rb, 0, 4 * count as u64);
                self.queue.submit([enc.finish()]);
                self.device.poll(wgpu::Maintain::Wait);
                let slice = rb.slice(..);
                slice.map_async(wgpu::MapMode::Read, |_| {});
                self.device.poll(wgpu::Maintain::Wait);
                let mapped = slice.get_mapped_range();
                let v: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
                drop(mapped);
                rb.unmap();
                v
            };
            let readback_vec4 = |buf: &wgpu::Buffer, count: usize| -> Vec<[f32; 4]> {
                let rb = Self::alloc_readback(&self.device, 16 * count as u64);
                let mut enc = self.device.create_command_encoder(&Default::default());
                enc.copy_buffer_to_buffer(buf, 0, &rb, 0, 16 * count as u64);
                self.queue.submit([enc.finish()]);
                self.device.poll(wgpu::Maintain::Wait);
                let slice = rb.slice(..);
                slice.map_async(wgpu::MapMode::Read, |_| {});
                self.device.poll(wgpu::Maintain::Wait);
                let mapped = slice.get_mapped_range();
                let v: Vec<[f32; 4]> = bytemuck::cast_slice(&mapped).to_vec();
                drop(mapped);
                rb.unmap();
                v
            };

            eprintln!("── CREAM_DIAG: CellList pipeline check (N={n}) ──");

            // ① sorted_atoms: must be a permutation of [0..N)
            let sa = readback_u32(&cl.sorted_atoms_buf, n);
            let mut seen = vec![false; n];
            let mut sa_ok = true;
            for (k, &orig) in sa.iter().enumerate() {
                if orig as usize >= n {
                    eprintln!("  [FAIL] sorted_atoms[{k}] = {orig} (out of range 0..{n})");
                    sa_ok = false;
                    break;
                }
                if seen[orig as usize] {
                    eprintln!(
                        "  [FAIL] sorted_atoms: duplicate original index {orig} at Morton slot {k}"
                    );
                    sa_ok = false;
                    break;
                }
                seen[orig as usize] = true;
            }
            if sa_ok && seen.iter().all(|&v| v) {
                eprintln!("  [OK]   sorted_atoms is a valid permutation of [0..{n})");
            } else if sa_ok {
                let missing: Vec<_> = seen
                    .iter()
                    .enumerate()
                    .filter(|(_, &v)| !v)
                    .map(|(i, _)| i)
                    .take(5)
                    .collect();
                eprintln!("  [FAIL] sorted_atoms: missing indices {:?}...", missing);
            }

            // ② reordered_positions: must match positions[sorted_atoms[k]]
            let rp = readback_vec4(&cl.reordered_pos_buf, n);
            let mut rp_max_diff: f32 = 0.0;
            let mut rp_fail = None;
            for k in 0..n {
                let orig = sa[k] as usize;
                if orig >= n {
                    break;
                }
                for c in 0..3 {
                    let diff = (rp[k][c] - positions[orig][c]).abs();
                    if diff > rp_max_diff {
                        rp_max_diff = diff;
                    }
                    if diff > 1e-6 && rp_fail.is_none() {
                        rp_fail = Some((k, orig, c, rp[k][c], positions[orig][c]));
                    }
                }
            }
            match rp_fail {
                None => eprintln!("  [OK]   reordered_positions match (max diff = {rp_max_diff:.2e})"),
                Some((k, orig, c, got, want)) => eprintln!(
                    "  [FAIL] reordered_positions[{k}][{c}] = {got:.6}, expected positions[{orig}][{c}] = {want:.6}"
                ),
            }

            // ③ densities: for a perfect crystal all should be equal (by symmetry)
            let dens = readback_f32(&fb.density_buf, n);
            let d_min = dens.iter().cloned().fold(f32::INFINITY, f32::min);
            let d_max = dens.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let d_mean = dens.iter().sum::<f32>() / n as f32;
            let d_spread = d_max - d_min;
            eprintln!(
                "  [INFO] densities: min={d_min:.6e} max={d_max:.6e} mean={d_mean:.6e} spread={d_spread:.2e}"
            );
            if d_spread > 1e-3 * d_mean.abs().max(1e-10) {
                // Find first outlier
                for (k, &d) in dens.iter().enumerate() {
                    if (d - d_mean).abs() > 0.01 * d_mean.abs().max(1e-10) {
                        let orig = sa[k] as usize;
                        eprintln!(
                            "  [WARN] density[{k}] = {d:.6e} (Morton), orig_atom={orig}, deviates from mean"
                        );
                        break;
                    }
                }
            }

            // ④ forces: show first few for inspection
            eprintln!("  [INFO] GPU forces (first 4 atoms, original order):");
            for i in 0..4.min(n) {
                eprintln!(
                    "    atom {i}: [{:.8e}, {:.8e}, {:.8e}]",
                    forces[i][0], forces[i][1], forces[i][2]
                );
            }

            // ⑤ cell_ids: every value must be < n_morton
            let cell_ids = readback_u32(&cl.cell_ids_buf, n);
            let n_morton = cl.n_morton;
            let mut cid_bad = 0usize;
            let mut cid_max = 0u32;
            for &cid in &cell_ids {
                if (cid as usize) >= n_morton {
                    cid_bad += 1;
                }
                if cid > cid_max {
                    cid_max = cid;
                }
            }
            if cid_bad == 0 {
                eprintln!("  [OK]   cell_ids all in [0, {n_morton}) (max observed = {cid_max})");
            } else {
                eprintln!("  [FAIL] cell_ids: {cid_bad} values ≥ n_morton={n_morton}");
            }

            // ⑥ cell_start: monotone non-decreasing and cell_start[n_morton] == n
            let cell_start = readback_u32(&cl.cell_start_buf, n_morton + 1);
            let mut cs_bad = 0usize;
            for m in 0..n_morton {
                if cell_start[m + 1] < cell_start[m] {
                    cs_bad += 1;
                }
            }
            if cs_bad == 0 && cell_start[n_morton] == n as u32 {
                eprintln!(
                    "  [OK]   cell_start monotone, terminal = {} (== n_atoms)",
                    cell_start[n_morton]
                );
            } else {
                eprintln!(
                    "  [FAIL] cell_start: {cs_bad} inversions, terminal = {} (expected {})",
                    cell_start[n_morton], n
                );
            }

            // ⑦ debug_flags: shader-side anomaly counters
            let dbg = readback_u32(&cl.debug_flags_buf, 32);
            let labels = [
                "p1_cid_oob",
                "p1_cs_oob",
                "p1_cs_inverted",
                "p1_bb_empty",
                "p1_atom_k_oob",
                "p1_nan_rho",
                "p1_neighbor_cnt",
                "p1_cutoff_hits",
                "p2_cid_oob",
                "p2_cs_oob",
                "p2_cs_inverted",
                "p2_bb_empty",
                "p2_atom_k_oob",
                "p2_nan_force",
                "p2_neighbor_cnt",
                "p2_cutoff_hits",
                "p2_sorted_atom_oob",
            ];
            let any_err = dbg[0]
                .max(dbg[1])
                .max(dbg[2])
                .max(dbg[4])
                .max(dbg[5])
                .max(dbg[8])
                .max(dbg[9])
                .max(dbg[10])
                .max(dbg[12])
                .max(dbg[13])
                .max(dbg[16])
                > 0;
            if any_err {
                eprintln!("  [FAIL] shader diagnostics report anomalies:");
            } else {
                eprintln!("  [OK]   shader diagnostics: no anomalies");
            }
            for (i, label) in labels.iter().enumerate() {
                if dbg[i] > 0 {
                    eprintln!("    dbg[{i:>2}] {label:<20} = {}", dbg[i]);
                }
            }

            eprintln!("── end CREAM_DIAG ──");
        }

        fb.positions_current_on_gpu = verlet.is_some();
        self.frame_cache = Some(fb);

        Ok(ComputeResult {
            forces,
            energy,
            energy_per_atom: vec![], // GPU path: always empty
            virial,                  // GPU path now computes this directly
            virial_per_atom: vec![], // CPU-only
            densities: vec![],       // CPU-only
            embedding_energies: vec![], // CPU-only
        })
    }

    // ── Buffer factories ─────────────────────────────────────────────────────

    fn create_frame_buffers(
        device: &wgpu::Device,
        n: usize,
        layout: &crate::potential::BufferLayout,
    ) -> FrameBuffers {
        // 3-level GPU reduction sizes (all ≥ 1); CPU handles the final sum.
        let num_wg = n.div_ceil(64).max(1);
        let lvl1 = num_wg.div_ceil(64).max(1);
        let lvl2 = lvl1.div_ceil(64).max(1);
        // Virial: six f32 per WG, flat layout.
        let virial_bytes = 4u64 * 6 * num_wg as u64;

        FrameBuffers {
            n,
            positions_current_on_gpu: false,
            pos_buf: Self::alloc_storage_init(device, &vec![[0f32; 4]; n]),
            types_buf: Self::alloc_storage_init_u32(device, &vec![0u32; n]),
            params_buf: Self::alloc_uniform_bytes(device, std::mem::size_of::<GpuSimParams>()),
            density_buf: Self::alloc_storage(device, 4 * n as u64),
            forces_buf: Self::alloc_storage(device, layout.output_stride_bytes * n as u64),
            wg_energy_buf: Self::alloc_storage(device, 4 * num_wg as u64),
            lvl1_buf: Self::alloc_storage(device, 4 * lvl1 as u64),
            lvl2_buf: Self::alloc_storage(device, 4 * lvl2 as u64),
            wg_virial_buf: Self::alloc_storage(device, virial_bytes),
            rb_virial_partials: Self::alloc_readback(device, virial_bytes),
            reduce_params_a_buf: Self::alloc_uniform_bytes(
                device,
                std::mem::size_of::<ReduceParams>(),
            ),
            reduce_params_b_buf: Self::alloc_uniform_bytes(
                device,
                std::mem::size_of::<ReduceParams>(),
            ),
            velocities_buf: Self::alloc_storage_init(device, &vec![[0f32; 4]; n]),
            verlet_params_buf: Self::alloc_uniform_bytes(
                device,
                std::mem::size_of::<VerletParams>(),
            ),
            rb_forces: Self::alloc_readback(device, layout.output_stride_bytes * n as u64),
            // lvl2 readback: sized to the maximum possible lvl2 count for this N.
            // CPU Neumaier sum reads exactly lvl2_count (≤ lvl2) elements each call.
            rb_lvl2: Self::alloc_readback(device, 4 * lvl2 as u64),
            bg_pass0a: None,
            bg_pass0b: None,
            bg_pass0c: None,
            bg_pass0d: None,
            bg_pass1: None,
            bg_pass2: None,
            bg_pass3a: None,
            bg_pass3b: None,
            bg_verlet: None,
            cell_list: None,
            neighbor_list: None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    // ── Morton-code helpers ──────────────────────────────────────────────────
    // These mirror the implementations in cpu_engine.rs and the WGSL shaders
    // (spread_bits / morton3).  Kept module-private; the single source of truth
    // for the *formula* is cpu_engine.rs — any change there must be reflected
    // here and in common.wgsl.
    //
    // Reference: Warren & Salmon (1993); Pharr & Humphreys (2004) §7.3.
    #[inline]
    fn morton_spread(v: u32) -> u32 {
        let mut x = v & 0x0000_03ff; // keep only 10 bits
        x = (x | (x << 16)) & 0x0300_00ff;
        x = (x | (x << 8)) & 0x0300_f00f;
        x = (x | (x << 4)) & 0x030c_30c3;
        x = (x | (x << 2)) & 0x0924_9249;
        x
    }

    /// Encode 3D cell coordinates → 30-bit Morton (Z-order) code.
    /// Each coordinate must fit in 10 bits (< 1024).
    /// Matches `morton3(cx, cy, cz)` in `common.wgsl`.
    #[inline]
    fn morton_encode(cx: u32, cy: u32, cz: u32) -> u32 {
        Self::morton_spread(cx) | (Self::morton_spread(cy) << 1) | (Self::morton_spread(cz) << 2)
    }

    /// Number of Morton-code slots needed for a padded grid of size
    /// `(nx_pad, ny_pad, nz_pad)`.
    ///
    /// **Not** equal to `nx_pad * ny_pad * nz_pad` for non-cubic grids.
    /// Example: nx_pad=2, ny_pad=2, nz_pad=8 → product=32, but
    /// `morton3(1,1,7) = 295`, so `n_morton = 296`.  Using the product
    /// would cause OOB writes into `cell_start` / `cell_counts`.
    ///
    /// Matches the formula in `cpu_engine.rs`.
    #[inline]
    fn n_morton_for_pad(nx_pad: u32, ny_pad: u32, nz_pad: u32) -> usize {
        Self::morton_encode(nx_pad - 1, ny_pad - 1, nz_pad - 1) as usize + 1
    }

    fn create_cell_list_buffers(
        device: &wgpu::Device,
        n: usize,
        ncx: u32,
        ncy: u32,
        ncz: u32,
        nct: u32,
        cs: f32,
        nx_pad: u32,
        ny_pad: u32,
        nz_pad: u32,
    ) -> CellListBuffers {
        // Use the Morton-code formula, not the product nx_pad*ny_pad*nz_pad.
        // For non-cubic padded grids the product underestimates the required
        // buffer size, causing OOB accesses in cell_start / cell_counts.
        // See Self::n_morton_for_pad for the full explanation.
        let n_morton = Self::n_morton_for_pad(nx_pad, ny_pad, nz_pad);
        CellListBuffers {
            n_atoms: n,
            n_cells_x: ncx,
            n_cells_y: ncy,
            n_cells_z: ncz,
            n_cells_total: nct,
            cell_size: cs,
            n_cells_x_pad: nx_pad,
            n_cells_y_pad: ny_pad,
            n_cells_z_pad: nz_pad,
            n_morton,
            cell_ids_buf: Self::alloc_storage(device, 4 * n as u64),
            sorted_atoms_buf: Self::alloc_storage(device, 4 * n as u64),
            cell_start_buf: Self::alloc_storage(device, 4 * (n_morton + 1) as u64),
            cell_counts_buf: Self::alloc_storage(device, 4 * n_morton as u64),
            write_offsets_buf: Self::alloc_storage(device, 4 * n_morton as u64),
            rb_cell_counts: Self::alloc_readback(device, 4 * n_morton as u64),
            reordered_pos_buf: Self::alloc_storage(device, 16 * n as u64),
            reordered_types_buf: Self::alloc_storage(device, 4 * n as u64),
            debug_flags_buf: Self::alloc_storage(device, 4 * 32),
        }
    }

    // ── Shader sources ───────────────────────────────────────────────────────

    const PASS0A_SRC: &'static str = concat!(
        include_str!("shaders/common.wgsl"),
        "\n",
        include_str!("shaders/cell_pass0a_assign.wgsl"),
    );
    const PASS0B_SRC: &'static str = concat!(
        include_str!("shaders/common.wgsl"),
        "\n",
        include_str!("shaders/cell_pass0b_sort.wgsl"),
    );
    const PASS0C_SRC: &'static str = concat!(
        include_str!("shaders/common.wgsl"),
        "\n",
        include_str!("shaders/cell_pass0c_range.wgsl"),
    );
    const PASS0D_SRC: &'static str = concat!(
        include_str!("shaders/common.wgsl"),
        "\n",
        include_str!("shaders/cell_pass0d_reorder.wgsl"),
    );

    fn n_cells_from_dspacing(h: &[[f32; 3]; 3], cell_size: f32) -> (u32, u32, u32) {
        let [a, b, c] = h;
        let bc = [
            b[1] * c[2] - b[2] * c[1],
            b[2] * c[0] - b[0] * c[2],
            b[0] * c[1] - b[1] * c[0],
        ];
        let ca = [
            c[1] * a[2] - c[2] * a[1],
            c[2] * a[0] - c[0] * a[2],
            c[0] * a[1] - c[1] * a[0],
        ];
        let ab = [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ];
        let norm = |v: [f32; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let dot3 = |u: [f32; 3], v: [f32; 3]| u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
        let vol = dot3(*a, bc).abs();
        let (bc_n, ca_n, ab_n) = (norm(bc), norm(ca), norm(ab));
        let d_a = if bc_n > 1e-12 { vol / bc_n } else { cell_size };
        let d_b = if ca_n > 1e-12 { vol / ca_n } else { cell_size };
        let d_c = if ab_n > 1e-12 { vol / ab_n } else { cell_size };
        let nx_raw = ((d_a / cell_size).floor() as u32).max(1);
        let ny_raw = ((d_b / cell_size).floor() as u32).max(1);
        let nz_raw = ((d_c / cell_size).floor() as u32).max(1);

        // ── Auto-power-of-two grid rounding ─────────────────────────────────
        //
        // The CellList fast path in `eam_pass{1,2}_cellist.wgsl` uses a
        // workgroup-wide bounding-box + periodic wrap stencil that is
        // correct and efficient ONLY on power-of-two grids.  On non-p2
        // grids the shader falls back to iterating all `ncx*ncy*ncz` real
        // cells per workgroup, which is O(ncx³) per workgroup and quickly
        // dominates runtime for `ncx ≥ 8`.
        //
        // Profiling data confirmed:
        //   • `visits/atom` ∝ N on non-p2 grids (stencil totally ineffective)
        //   • `cells/WG` = `ncx*ncy*ncz` (full grid) on non-p2
        //   • CellList only beats AllPairs on N=6912 (ncx=8, p2 naturally)
        //
        // Fix: round each axis DOWN to the nearest power of two.  This
        // preserves the invariant `cell_width ≥ cutoff` because rounding
        // DOWN makes cells LARGER (`cell_width = d_axis / nx_new` with
        // `nx_new ≤ nx_raw`).  Example for rep=20, a=3.615:
        //   L = 72.3, cell_size = 4.95 → nx_raw = floor(14.6) = 14
        //   Rounded down: nx = 8, new cell_width = 72.3/8 = 9.04 > 4.95 ✓
        //
        // Trade-off: larger cells mean more atoms per cell, which
        // increases the per-cell tile-load cost.  But this is dwarfed by
        // the O(N²) → O(N) reduction in pair-scan work when the 27-cell
        // stencil starts pruning effectively.
        //
        // Opt-out: set `CREAM_DISABLE_AUTO_P2=1` to restore the raw
        // floor-based grid (useful for regression testing and to verify
        // the fast path is actually being hit).
        let auto_p2 = !std::env::var("CREAM_DISABLE_AUTO_P2")
            .ok()
            .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));

        // The "round-DOWN to power-of-two" rule from earlier releases was
        // correct but wasteful.  Example, rep=75, cubic Cu (d_axis = 271.1 Å,
        // cutoff = 4.95 Å, cell_size ≈ 4.99 Å):
        //
        //   nx_raw   = floor(271.1 / 4.99) = 54
        //   round_DOWN(54) = 32  →  cell_width = 271.1 / 32 = 8.47 Å
        //   round_UP(54)   = 64  →  cell_width = 271.1 / 64 = 4.24 Å ← < cell_size ✗
        //
        // In this case the round-up result puts cells *smaller* than the
        // cutoff, which breaks the 27-cell stencil invariant.  So round-up
        // cannot be used blindly.
        //
        // But on orthogonal / cubic lattices where d_axis / cell_size is
        // nicely between two powers of two, round-down explodes the stencil
        // volume — the 27 cells around each atom cover (3 × 8.47)³ ≈ 16 000 Å³
        // of space, while the physical cutoff sphere is only ≈ 510 Å³ (~3%
        // effective).  On densely-packed systems this is a 30× speed hit.
        //
        // Strategy — "round UP when it's safe, fall back to round DOWN":
        //   1. nx_up = next_power_of_two(nx_raw)
        //   2. If d_axis / nx_up ≥ cell_size (i.e. cell_width still ≥ cutoff
        //      because cell_size ≥ cutoff), accept nx_up.
        //   3. Otherwise fall back to round_down.
        //
        // For rep=75 cubic the predicate  271.1/64 = 4.24  <  4.99  fails,
        // so we use round-down (32).  For rep=50 cubic (d_axis = 180.75,
        // nx_raw = 36, nx_up = 64, width = 180.75/64 = 2.82 < 4.99 → down
        // to 32, width = 5.65 ≥ 4.99 ✓).  For rep=100 cubic (d_axis = 361.5,
        // nx_raw = 72, nx_up = 128, width = 2.82 < 4.99 → down to 64,
        // width = 5.65 ≥ 4.99 ✓) — no regression.
        //
        // The "round UP is safe" predicate fires for systems where the
        // lattice-to-cutoff ratio is big enough that doubling the grid
        // resolution still leaves cells ≥ cutoff.  This is the regime in
        // which round-down wastes most stencil volume, so we gain the most
        // there.
        let round_down_p2 = |n: u32| -> u32 {
            if n == 0 {
                1
            } else {
                1u32 << (31 - n.leading_zeros())
            }
        };
        // `d_axis` is the perpendicular spacing for each axis — use it to
        // check whether the rounded-up grid respects `cell_width ≥ cell_size`.
        let round_p2_preserving_width = |n_raw: u32, d_axis: f32| -> u32 {
            let down = round_down_p2(n_raw);
            // If already a power of two, no work to do.
            if down == n_raw {
                return down;
            }
            let up = down.saturating_mul(2);
            if up == 0 {
                return down;
            }
            // Guard against pathological geometry where d_axis is smaller
            // than cell_size even at nx_raw — just stick with nx_raw floor.
            if d_axis <= 0.0 || cell_size <= 0.0 {
                return down;
            }
            let width_up = d_axis / (up as f32);
            if width_up >= cell_size {
                up
            } else {
                down
            }
        };

        let (nx, ny, nz) = if auto_p2 {
            (
                round_p2_preserving_width(nx_raw, d_a),
                round_p2_preserving_width(ny_raw, d_b),
                round_p2_preserving_width(nz_raw, d_c),
            )
        } else {
            (nx_raw, ny_raw, nz_raw)
        };

        (nx, ny, nz)
    }

    // ── GPU helpers ──────────────────────────────────────────────────────────

    fn make_bgl(
        device: &wgpu::Device,
        entries: &[wgpu::BindGroupLayoutEntry],
    ) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cream::bgl"),
            entries,
        })
    }

    fn build_explicit_pipeline(
        device: &wgpu::Device,
        shader_src: &str,
        bgl_entries: &[wgpu::BindGroupLayoutEntry],
    ) -> wgpu::ComputePipeline {
        let bgl = Self::make_bgl(device, bgl_entries);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cream::layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cream::shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });
        // Honour CREAM_ENABLE_DEBUG=1 as a pipeline override so the CellList
        // shaders activate their `dbg[*]` atomic counters.  Only pass the
        // override key when the shader actually declares it (pass1/pass2 for
        // CellList and NeighborList paths), because wgpu rejects unknown
        // override IDs as a validation error.
        let enable_debug = std::env::var("CREAM_ENABLE_DEBUG")
            .ok()
            .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
        let shader_has_debug_override = shader_src.contains("override ENABLE_DEBUG");
        let constants: std::collections::HashMap<String, f64> =
            if enable_debug && shader_has_debug_override {
                let mut m = std::collections::HashMap::new();
                m.insert("ENABLE_DEBUG".to_string(), 1.0);
                m
            } else {
                std::collections::HashMap::new()
            };
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cream::pipeline"),
            layout: Some(&layout),
            module: &module,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants,
                ..wgpu::PipelineCompilationOptions::default()
            },
            cache: None,
        })
    }

    fn make_bind_group(
        device: &wgpu::Device,
        bgl: &wgpu::BindGroupLayout,
        slots: &[(u32, &wgpu::Buffer)],
    ) -> wgpu::BindGroup {
        let entries: Vec<wgpu::BindGroupEntry<'_>> = slots
            .iter()
            .map(|(idx, buf)| wgpu::BindGroupEntry {
                binding: *idx,
                resource: buf.as_entire_binding(),
            })
            .collect();
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cream::bg"),
            layout: bgl,
            entries: &entries,
        })
    }

    fn ensure_fixed_pipeline(
        device: &wgpu::Device,
        slot: &mut Option<wgpu::ComputePipeline>,
        shader_src: &str,
        bgl_entries: &[wgpu::BindGroupLayoutEntry],
    ) {
        if slot.is_none() {
            *slot = Some(Self::build_explicit_pipeline(
                device,
                shader_src,
                bgl_entries,
            ));
        }
    }

    /// Encode a single compute dispatch into an existing `CommandEncoder`.
    /// All dispatches in Submission B share one encoder → one `queue.submit()`.
    fn encode_dispatch(
        enc: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        bg: &wgpu::BindGroup,
        n_dispatch: usize,
    ) {
        const WG: u32 = 64;
        let groups = (n_dispatch as u32).div_ceil(WG).max(1);
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }

    // ── Buffer allocation helpers ─────────────────────────────────────────────

    fn alloc_storage(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size.max(4),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn alloc_storage_init<T: Pod>(device: &wgpu::Device, data: &[T]) -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        })
    }

    fn alloc_storage_init_u32(device: &wgpu::Device, data: &[u32]) -> wgpu::Buffer {
        Self::alloc_storage_init(device, data)
    }

    fn alloc_uniform_bytes(device: &wgpu::Device, size: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (size as u64).max(16),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn alloc_readback(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size.max(4),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    }

    // ── Readback helpers ──────────────────────────────────────────────────────

    fn read_forces(buf: &wgpu::Buffer, n: usize) -> Result<Vec<[f32; 3]>, CreamError> {
        let slice = buf.slice(..);
        let mapped = slice.get_mapped_range();
        let raw: &[[f32; 4]] = bytemuck::cast_slice(&mapped);
        let forces = raw.iter().take(n).map(|v| [v[0], v[1], v[2]]).collect();
        drop(mapped);
        buf.unmap();
        Ok(forces)
    }

    /// Read `lvl2_count` f32 values from the lvl2 GPU reduction buffer and
    /// sum them with Neumaier-compensated summation (f64 accumulator).
    ///
    /// This replaces the former GPU pass3c (single-workgroup reduction) which
    /// silently dropped elements when `lvl2_count > 64` (N > 16_777_216),
    /// causing ~0.4 % energy under-estimation at N ≈ 23 M.
    ///
    /// `lvl2_count = ceil(ceil(N/64)/64)/64)` — fewer than 400 values for
    /// N ≤ 100 M — so the CPU cost is negligible (< 1 µs on modern hardware).
    ///
    /// Neumaier (improved Kahan) is chosen over plain summation because the
    /// lvl2 partials each represent ~262 K atoms and can span several orders
    /// of magnitude in heterogeneous systems, making catastrophic cancellation
    /// a real risk for near-equilibrium configurations.
    fn read_and_sum_lvl2(buf: &wgpu::Buffer, lvl2_count: usize) -> Result<f32, CreamError> {
        let nbytes = 4 * lvl2_count as u64;
        let slice = buf.slice(..nbytes);
        let mapped = slice.get_mapped_range();
        let partials: &[f32] = bytemuck::cast_slice(&mapped);

        // Neumaier-compensated sum with f64 accumulator.
        // Each partial is widened to f64 before accumulation so that the
        // compensation term itself does not lose low-order bits.
        let mut sum  = 0.0f64;
        let mut comp = 0.0f64;
        for &v in partials {
            let x = v as f64;
            let t = sum + x;
            // Capture the rounding error lost in `t`.
            if sum.abs() >= x.abs() {
                comp += (sum - t) + x;
            } else {
                comp += (x - t) + sum;
            }
            sum = t;
        }

        drop(mapped);
        buf.unmap();
        // Cast back to f32: the GPU pipeline is f32 throughout, so the result
        // cannot be more precise than f32 regardless of the accumulator width.
        Ok((sum + comp) as f32)
    }

    /// Read the `6 × num_wg` per-workgroup virial partials, sum them with
    /// Neumaier-compensated (improved Kahan) summation per component, then
    /// apply `σ_αβ = −W_αβ / V` for periodic systems (returns zeros for
    /// non-periodic ones).
    ///
    /// Sign convention and Voigt order match the CPU reference in
    /// `cpu_engine.rs`: `[σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy]`, eV/Å³.
    ///
    /// Per-thread Kahan compensation is already applied *inside* the pass-2
    /// shader during the pair-force loop, so the partials that arrive here
    /// have f32-accurate additions; the host-side Neumaier sum is the
    /// final defence against catastrophic cancellation across the ~N/64
    /// partials, which can matter for very large N at near-equilibrium
    /// lattices where W has significant cancellation between pairs.
    fn read_and_finalize_virial(
        buf: &wgpu::Buffer,
        num_wg: usize,
        cell: Option<[[f32; 3]; 3]>,
    ) -> Result<[f64; 6], CreamError> {
        let nbytes = 4u64 * 6 * num_wg as u64;
        let slice = buf.slice(..nbytes);
        let mapped = slice.get_mapped_range();
        let partials: &[f32] = bytemuck::cast_slice(&mapped);

        // Per-component Neumaier sum (f64 target).  For each of the six
        // virial components we walk the WG stride (6 partials per WG).
        let mut sums = [0.0f64; 6];
        let mut comps = [0.0f64; 6];
        for w in 0..num_wg {
            let base = 6 * w;
            for c in 0..6 {
                let x = partials[base + c] as f64;
                let t = sums[c] + x;
                if sums[c].abs() >= x.abs() {
                    comps[c] += (sums[c] - t) + x;
                } else {
                    comps[c] += (x - t) + sums[c];
                }
                sums[c] = t;
            }
        }
        let mut w_total = [0.0f64; 6];
        for c in 0..6 {
            w_total[c] = sums[c] + comps[c];
        }

        drop(mapped);
        buf.unmap();

        // σ = −W / V — only defined when the cell has positive volume.
        let virial: [f64; 6] = match cell {
            Some(h) => {
                let a = h[0];
                let b = h[1];
                let c = h[2];
                let vol = ((a[0] * (b[1] * c[2] - b[2] * c[1])
                    - a[1] * (b[0] * c[2] - b[2] * c[0])
                    + a[2] * (b[0] * c[1] - b[1] * c[0])) as f64)
                    .abs();
                if vol > 0.0 {
                    [
                        -w_total[0] / vol,
                        -w_total[1] / vol,
                        -w_total[2] / vol,
                        -w_total[3] / vol,
                        -w_total[4] / vol,
                        -w_total[5] / vol,
                    ]
                } else {
                    [0.0; 6]
                }
            }
            None => [0.0; 6],
        };
        Ok(virial)
    }

    fn hash_str(s: &str) -> u64 {
        let mut h = DefaultHasher::new();
        s.hash(&mut h);
        h.finish()
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::potential::eam::{synthetic_cu_alloy_src, EamPotential};
    use crate::reference::compute_eam_cpu;

    async fn try_engine() -> Option<ComputeEngine> {
        ComputeEngine::new(NeighborStrategy::AllPairs).await.ok()
    }

    fn cu4_pos4() -> Vec<[f32; 4]> {
        let a = 3.615_f32;
        vec![
            [0.0, 0.0, 0.0, 0.0],
            [a / 2.0, a / 2.0, 0.0, 0.0],
            [a / 2.0, 0.0, a / 2.0, 0.0],
            [0.0, a / 2.0, a / 2.0, 0.0],
        ]
    }
    fn cu4_types() -> Vec<u32> {
        vec![0u32; 4]
    }
    fn cell() -> Option<[[f32; 3]; 3]> {
        Some([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    }
    fn synth_pot() -> EamPotential {
        EamPotential::from_str(&synthetic_cu_alloy_src(100, 100, 0.05, 0.01, 4.5)).unwrap()
    }

    #[test]
    fn gpu_sim_params_size_is_176_bytes() {
        assert_eq!(
            std::mem::size_of::<GpuSimParams>(),
            176,
            "GpuSimParams must be 176 bytes to match SimParams in common.wgsl"
        );
    }

    #[ignore = "requires GPU"]
    #[test]
    fn gpu_engine_init() {
        pollster::block_on(async {
            let e = ComputeEngine::new(NeighborStrategy::AllPairs).await;
            assert!(e.is_ok(), "GPU engine init failed: {:?}", e.err());
        });
    }

    #[ignore = "requires GPU"]
    #[test]
    fn gpu_forces_finite() {
        pollster::block_on(async {
            let mut eng = match try_engine().await {
                Some(e) => e,
                None => return,
            };
            let pot = synth_pot();
            let res = eng
                .compute(&cu4_pos4(), &cu4_types(), cell(), &pot)
                .await
                .unwrap();
            assert_eq!(res.forces.len(), 4);
            for (i, f) in res.forces.iter().enumerate() {
                for &v in f {
                    assert!(v.is_finite(), "forces[{i}]={v} non-finite");
                }
            }
            assert!(res.energy.is_finite());
        });
    }

    #[ignore = "requires GPU"]
    #[test]
    fn gpu_vs_cpu_accuracy() {
        pollster::block_on(async {
            let mut eng = match try_engine().await {
                Some(e) => e,
                None => return,
            };
            let pot = synth_pot();
            let pos_f64: Vec<[f64; 3]> = cu4_pos4()
                .iter()
                .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
                .collect();
            let cell_f64 = cell().map(|h| {
                [
                    [h[0][0] as f64, h[0][1] as f64, h[0][2] as f64],
                    [h[1][0] as f64, h[1][1] as f64, h[1][2] as f64],
                    [h[2][0] as f64, h[2][1] as f64, h[2][2] as f64],
                ]
            });
            let cpu = compute_eam_cpu(&pot, &pos_f64, &cu4_types(), cell_f64);
            let gpu = eng
                .compute(&cu4_pos4(), &cu4_types(), cell(), &pot)
                .await
                .unwrap();
            let mut max_abs = 0.0_f64;
            for i in 0..4 {
                for c in 0..3 {
                    max_abs = max_abs.max((gpu.forces[i][c] as f64 - cpu.forces[i][c]).abs());
                }
            }
            let e_rel = ((gpu.energy as f64 - cpu.energy) / cpu.energy.abs().max(1e-10)).abs();
            assert!(max_abs < 1e-3, "max force error {max_abs:.2e} (limit 1e-3)");
            assert!(e_rel < 1e-5, "energy rel error {e_rel:.2e} (limit 1e-5)");
        });
    }

    #[ignore = "requires GPU"]
    #[test]
    fn gpu_frame_cache_reuse() {
        pollster::block_on(async {
            let mut eng = match try_engine().await {
                Some(e) => e,
                None => return,
            };
            let pot = synth_pot();
            let r1 = eng
                .compute(&cu4_pos4(), &cu4_types(), cell(), &pot)
                .await
                .unwrap();
            let r2 = eng
                .compute(&cu4_pos4(), &cu4_types(), cell(), &pot)
                .await
                .unwrap();
            let diff = (r1.energy - r2.energy).abs();
            assert!(
                diff < 1e-6,
                "energy differs across cached frames: {diff:.2e}"
            );
        });
    }

    // ── Morton Code unit tests ────────────────────────────────────────────────

    #[test]
    fn morton_encode_decode_roundtrip() {
        fn spread_bits(v: u32) -> u32 {
            let mut x = v & 0x3ff;
            x = (x | (x << 16)) & 0x030000ff;
            x = (x | (x << 8)) & 0x0300f00f;
            x = (x | (x << 4)) & 0x030c30c3;
            x = (x | (x << 2)) & 0x09249249;
            x
        }
        fn compact_bits(w: u32) -> u32 {
            let mut x = w & 0x09249249;
            x = (x | (x >> 2)) & 0x030c30c3;
            x = (x | (x >> 4)) & 0x0300f00f;
            x = (x | (x >> 8)) & 0x030000ff;
            x = (x | (x >> 16)) & 0x000003ff;
            x
        }
        fn morton3(cx: u32, cy: u32, cz: u32) -> u32 {
            spread_bits(cx) | (spread_bits(cy) << 1) | (spread_bits(cz) << 2)
        }
        for cx in 0u32..16 {
            for cy in 0u32..16 {
                for cz in 0u32..16 {
                    let m = morton3(cx, cy, cz);
                    assert_eq!(compact_bits(m), cx, "x mismatch ({cx},{cy},{cz})");
                    assert_eq!(compact_bits(m >> 1), cy, "y mismatch ({cx},{cy},{cz})");
                    assert_eq!(compact_bits(m >> 2), cz, "z mismatch ({cx},{cy},{cz})");
                }
            }
        }
    }

    #[test]
    fn morton_bit_interleaving() {
        fn spread_bits(v: u32) -> u32 {
            let mut x = v & 0x3ff;
            x = (x | (x << 16)) & 0x030000ff;
            x = (x | (x << 8)) & 0x0300f00f;
            x = (x | (x << 4)) & 0x030c30c3;
            x = (x | (x << 2)) & 0x09249249;
            x
        }
        fn morton3(cx: u32, cy: u32, cz: u32) -> u32 {
            spread_bits(cx) | (spread_bits(cy) << 1) | (spread_bits(cz) << 2)
        }
        assert_eq!(morton3(1, 0, 0), 0b001);
        assert_eq!(morton3(0, 1, 0), 0b010);
        assert_eq!(morton3(0, 0, 1), 0b100);
        assert_eq!(morton3(1, 1, 1), 0b111);
        assert_eq!(morton3(2, 0, 0), spread_bits(2));
    }

    #[test]
    fn morton_pad_size() {
        // Cubic case: product and Morton formula agree.
        assert_eq!(ComputeEngine::n_morton_for_pad(4, 4, 4), 64);
        assert_eq!(ComputeEngine::n_morton_for_pad(1, 1, 1), 1);

        // Non-cubic case: product UNDERESTIMATES — this is the bug the fix
        // addresses.  n_pad=[2,2,8] → product=32, but morton3(1,1,7)=295,
        // so n_morton must be 296.
        let (nx, ny, nz) = (2u32, 2u32, 8u32);
        let product = (nx * ny * nz) as usize; // 32 — WRONG
        let correct = ComputeEngine::n_morton_for_pad(nx, ny, nz); // 296
        assert!(
            correct > product,
            "non-cubic: Morton formula ({correct}) must exceed product ({product})"
        );
        assert_eq!(correct, 296);

        // A realistic asymmetric padded grid (nx_pad=16, ny_pad=8, nz_pad=16).
        // Product = 2048; Morton formula gives the true slot count.
        let nm = ComputeEngine::n_morton_for_pad(16, 8, 16);
        assert_eq!(nm, ComputeEngine::morton_encode(15, 7, 15) as usize + 1);
    }

    #[ignore = "requires GPU"]
    #[test]
    fn gpu_newton_third_law() {
        pollster::block_on(async {
            let mut eng = match try_engine().await {
                Some(e) => e,
                None => return,
            };
            let pot = synth_pot();
            let res = eng
                .compute(&cu4_pos4(), &cu4_types(), cell(), &pot)
                .await
                .unwrap();
            let sum: [f32; 3] = res
                .forces
                .iter()
                .fold([0.0_f32; 3], |a, f| [a[0] + f[0], a[1] + f[1], a[2] + f[2]]);
            for (ax, &s) in ['x', 'y', 'z'].iter().zip(&sum) {
                assert!(s.abs() < 1e-3, "sum F_{ax} = {s:.2e}");
            }
        });
    }

    /// CPU engine energy_per_atom is still populated (CPU path is unchanged).
    #[test]
    fn energy_per_atom_sum_equals_total_cpu_n108() {
        use crate::CpuEngine;
        let pot = synth_pot();
        let a = 3.615_f32;
        let rep = 3_usize;
        let basis: [[f32; 3]; 4] = [
            [0.0, 0.0, 0.0],
            [a / 2.0, a / 2.0, 0.0],
            [a / 2.0, 0.0, a / 2.0],
            [0.0, a / 2.0, a / 2.0],
        ];
        let mut pos4 = Vec::new();
        let mut types = Vec::new();
        for ix in 0..rep {
            for iy in 0..rep {
                for iz in 0..rep {
                    for b in &basis {
                        pos4.push([
                            b[0] + ix as f32 * a,
                            b[1] + iy as f32 * a,
                            b[2] + iz as f32 * a,
                            0.0,
                        ]);
                        types.push(0u32);
                    }
                }
            }
        }
        assert_eq!(pos4.len(), 108);
        let l = a * rep as f32;
        let cell = Some([[l, 0.0, 0.0], [0.0, l, 0.0], [0.0, 0.0, l]]);
        let eng = CpuEngine::new();
        let res = eng.compute_sync(&pos4, &types, cell, &pot).unwrap();
        let sum: f32 = res.energy_per_atom.iter().sum();
        let diff = (sum - res.energy).abs();
        assert!(
            diff < 1e-5,
            "energy_per_atom sum {sum:.6} ≠ energy {:.6} (diff {diff:.2e})",
            res.energy
        );
    }

    #[cfg_attr(miri, ignore = "dlopen (wgpu/Vulkan) is not supported under Miri")]
    #[test]
    fn celllist_invalid_cell_size_returns_error() {
        pollster::block_on(async {
            for bad_cs in [0.0_f32, -1.0, f32::NAN, f32::INFINITY] {
                let result =
                    ComputeEngine::new(NeighborStrategy::CellList { cell_size: bad_cs }).await;
                if let Ok(mut eng) = result {
                    let pot = synth_pot();
                    let err = eng
                        .compute(
                            &cu4_pos4(),
                            &cu4_types(),
                            Some([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]),
                            &pot,
                        )
                        .await;
                    assert!(err.is_err(), "cell_size={bad_cs} should return Err, got Ok");
                }
            }
        });
    }
}