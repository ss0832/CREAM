//! CPU-based EAM compute engine using rayon with **half-pair** (Newton on) optimisation.
//!
//! # Design
//! [`CpuEngine`] mirrors the public API of [`crate::engine::ComputeEngine`]:
//! same `compute_sync()` signature, same [`ComputeResult`] return type.
//!
//! It is intended for two purposes:
//! 1. **Fallback** when no GPU adapter is available.
//! 2. **Benchmark baseline** for measuring GPU speedup.
//!
//! # Half-pair optimisation (Newton's 3rd law)
//!
//! Each atom pair (i, j) is visited **once** (j > i only).
//! Both atoms receive their contributions in a single pass:
//!
//! - **Pass 1** (density): ρᵢ += f_β(r), ρⱼ += f_α(r)
//! - **Pass 2** (forces):  Fᵢ += coeff·r̂, Fⱼ −= coeff·r̂
//!
//! This halves the number of distance evaluations compared to full-pair.
//!
//! Since different rayon threads may process pairs that write to the same
//! atom j, each thread owns its own accumulator arrays and they are merged
//! after all threads complete.
//!
//! Parallelism strategy: **interleaved row distribution**.  Thread t processes
//! outer-loop rows {t, t+T, t+2T, …} where T = number of threads.  This
//! distributes work evenly (row i has N−i−1 pairs; interleaving mixes heavy
//! low-i rows with light high-i rows across threads) and creates exactly T
//! accumulators — eliminating the unbounded allocation growth that
//! `into_par_iter().fold().reduce()` suffers from work-stealing subdivision.
//!
//! Memory overhead: O(N × T) where T = rayon thread count.
//! For N = 10 000 and T = 8 this is ~1.5 MB.
//!
//! # Note on floating-point determinism
//! The half-pair rayon version reorders additions across threads **and**
//! changes which atoms contribute to which accumulator compared to the
//! full-pair reference implementation.  Results agree to within f32
//! rounding (~1e-5 relative) but are not bit-exact.

#[cfg(not(miri))]
use rayon::prelude::*;

#[cfg(not(miri))]
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use crate::{engine::ComputeResult, error::CreamError, potential::eam::EamPotential};

/// PBC matrices for f32 CPU engine: (cell H, inverse H⁻¹).
type PbcMatricesF32 = ([[f32; 3]; 3], [[f32; 3]; 3]);

// ── Local f32 helpers ─────────────────────────────────────────────────────────
//
// cpu_engine operates in f32 to match the GPU layout.  The shared helpers in
// reference.rs are now f64; we keep independent f32 versions here so the two
// implementations remain decoupled.

const MIN_DIST_SQ: f32 = 1e-4;

// ── EAM table lookup helpers ─────────────────────────────────────────────────
//
// `CellListData` and its Morton / PBC helpers live in `src/cell_list.rs`
// (shared with `src/neighbor_list.rs`).  The EAM table lookups remain
// physics-specific to this module and stay here.

#[inline]
fn linear_interp_f32(table: &[f32], idx_f: f32) -> f32 {
    let n = table.len();
    let idx = idx_f as usize;
    let frac = idx_f - idx as f32;
    let i0 = idx.min(n.saturating_sub(2));
    table[i0] + frac * (table[(i0 + 1).min(n - 1)] - table[i0])
}

#[inline]
fn lookup_by_r(table: &[f32], dr_inv: f32, r: f32) -> f32 {
    linear_interp_f32(table, r * dr_inv)
}

#[inline]
fn lookup_by_rho(table: &[f32], drho_inv: f32, rho: f32) -> f32 {
    linear_interp_f32(table, rho * drho_inv)
}

// ── Shared cell-list data structure ───────────────────────────────────────────
//
// `CellListData` and its Morton / PBC helpers live in `src/cell_list.rs` so
// both this module (physics) and `src/neighbor_list.rs` (GPU NL builder) can
// share a single implementation.  We re-implement the CPU-physics-only
// iterators (`for_each_neighbor`, `for_each_forward_neighbor`) here because
// they depend on `MIN_DIST_SQ` and `PbcMatricesF32` which are private to this
// module and physics-specific.
use crate::cell_list::{mat3_inv_f32, min_image_mat_f32, morton_encode, CellListData};

impl CellListData {
    /// Iterate over all atoms `j` within cutoff of atom `i` (j ≠ i).
    ///
    /// Full-neighbor variant (both directions); retained for testing.
    /// PBC displacement uses integer cell-offset shifts — no `roundf` per pair.
    ///
    /// # Stencil deduplication for triclinic PBC
    /// When `n_k = 2` in any direction, two stencil offsets (e.g. `dcx = -1` and
    /// `dcx = +1`) both wrap to the same cell index.  For orthorhombic PBC the
    /// different lattice shifts `k` produce distinct Cartesian displacements, so
    /// the cutoff guard naturally discards the far image — no deduplication needed.
    /// For triclinic PBC `min_image_mat_f32` is shift-independent, so visiting the
    /// same cell twice would double-count pairs.  A 27-slot stack array tracks
    /// visited Morton codes and skips repeated cell visits in that case.
    #[allow(dead_code)]
    #[inline]
    fn for_each_neighbor(
        &self,
        i: usize,
        positions: &[[f32; 4]],
        cutoff_sq: f32,
        pbc: &Option<PbcMatricesF32>,
        mut visitor: impl FnMut(usize, [f32; 3], f32),
    ) {
        let [n0, n1, n2] = self.n;
        let pi = &positions[i];
        let (cx0, cy0, cz0) = self.cell_coords_of(pi);

        let triclinic_pbc = pbc.is_some() && !self.is_ortho;
        let mut seen_cells = [usize::MAX; 27];
        let mut n_seen = 0usize;

        for dcx in -1i32..=1 {
            for dcy in -1i32..=1 {
                for dcz in -1i32..=1 {
                    let (nx, ny, nz, shift) = match pbc {
                        Some((h, _)) => self.pbc_cell_and_shift(cx0, cy0, cz0, dcx, dcy, dcz, h),
                        None => {
                            let nx = cx0 + dcx;
                            let ny = cy0 + dcy;
                            let nz = cz0 + dcz;
                            if nx < 0
                                || nx >= n0 as i32
                                || ny < 0
                                || ny >= n1 as i32
                                || nz < 0
                                || nz >= n2 as i32
                            {
                                continue;
                            }
                            (nx as u32, ny as u32, nz as u32, [0.0f32; 3])
                        }
                    };

                    let c = morton_encode(nx, ny, nz) as usize;
                    if triclinic_pbc {
                        if seen_cells[..n_seen].contains(&c) {
                            continue;
                        }
                        seen_cells[n_seen] = c;
                        n_seen += 1;
                    }

                    for &j in &self.sorted[self.cell_start[c]..self.cell_start[c + 1]] {
                        if j == i {
                            continue;
                        }
                        let d = match pbc {
                            None => [
                                positions[j][0] - pi[0] + shift[0],
                                positions[j][1] - pi[1] + shift[1],
                                positions[j][2] - pi[2] + shift[2],
                            ],
                            Some((h, h_inv)) => {
                                let raw = [
                                    positions[j][0] - pi[0],
                                    positions[j][1] - pi[1],
                                    positions[j][2] - pi[2],
                                ];
                                if self.is_ortho {
                                    [raw[0] + shift[0], raw[1] + shift[1], raw[2] + shift[2]]
                                } else {
                                    min_image_mat_f32(raw, h, h_inv)
                                }
                            }
                        };
                        let r_sq = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                        if r_sq < cutoff_sq && r_sq > MIN_DIST_SQ {
                            visitor(j, d, r_sq);
                        }
                    }
                }
            }
        }
    }

    /// **Half-pair** neighbour iterator: only visits `j > i`.
    ///
    /// # PBC displacement strategy
    ///
    /// **Orthorhombic PBC**: uses integer cell-offset shifts derived from how many
    /// times the stencil index wraps around the periodic boundary.  For ortho cells
    /// atoms placed in `[0, L)³` always have fractional coordinates in `[0, 1)`,
    /// so the cell index uniquely determines which periodic image is needed.
    /// This replaces per-pair `min_image_mat_f32` (three `roundf` calls) with a
    /// single shift vector computed once per stencil offset — O(27) work per atom.
    ///
    /// **Triclinic PBC**: atoms placed on a Cartesian grid may have fractional
    /// coordinates outside `[0, 1)`.  The cell index then does not uniquely
    /// determine the required periodic image, so we fall back to per-pair
    /// `min_image_mat_f32` which always gives the correct minimum image.
    /// Additionally, when `n_k = 2` in any direction, two stencil offsets
    /// (e.g. `dcx = -1` and `dcx = +1`) wrap to the *same* cell index but
    /// `min_image_mat_f32` is shift-independent — visiting the cell twice would
    /// double-count pairs.  A 27-slot stack array deduplicates cell visits for
    /// this case; it is a no-op for orthorhombic PBC and non-PBC (each stencil
    /// offset maps to a unique cell when `n_k ≥ 3`, and `n_k = 2` cannot arise
    /// for orthorhombic cells at the minimum box-size limit).
    ///
    /// **Non-PBC**: stencil with zero shift; out-of-bounds offsets are skipped.
    #[inline]
    fn for_each_forward_neighbor(
        &self,
        i: usize,
        positions: &[[f32; 4]],
        cutoff_sq: f32,
        pbc: &Option<PbcMatricesF32>,
        mut visitor: impl FnMut(usize, [f32; 3], f32),
    ) {
        let [n0, n1, n2] = self.n;
        let pi = &positions[i];
        let (cx0, cy0, cz0) = self.cell_coords_of(pi);

        // Triclinic PBC only: guard against visiting the same cell twice when
        // n_k = 2 causes two stencil offsets to wrap to the same Morton code.
        let triclinic_pbc = pbc.is_some() && !self.is_ortho;
        let mut seen_cells = [usize::MAX; 27];
        let mut n_seen = 0usize;

        for dcx in -1i32..=1 {
            for dcy in -1i32..=1 {
                for dcz in -1i32..=1 {
                    let (nx, ny, nz, shift_opt) = match pbc {
                        Some((h, h_inv)) => {
                            let (nx_w, ny_w, nz_w, shift) =
                                self.pbc_cell_and_shift(cx0, cy0, cz0, dcx, dcy, dcz, h);
                            // For triclinic cells, pass h_inv so the inner loop
                            // can apply per-pair min_image instead.
                            let _ = h_inv; // used via pbc ref below
                            (nx_w, ny_w, nz_w, Some(shift))
                        }
                        None => {
                            let nx = cx0 + dcx;
                            let ny = cy0 + dcy;
                            let nz = cz0 + dcz;
                            if nx < 0
                                || nx >= n0 as i32
                                || ny < 0
                                || ny >= n1 as i32
                                || nz < 0
                                || nz >= n2 as i32
                            {
                                continue;
                            }
                            (nx as u32, ny as u32, nz as u32, None)
                        }
                    };

                    let c = morton_encode(nx, ny, nz) as usize;
                    // Skip duplicate cell visits for triclinic PBC.
                    if triclinic_pbc {
                        if seen_cells[..n_seen].contains(&c) {
                            continue;
                        }
                        seen_cells[n_seen] = c;
                        n_seen += 1;
                    }

                    for &j in &self.sorted[self.cell_start[c]..self.cell_start[c + 1]] {
                        if j <= i {
                            continue; // skip j == i and j < i before any arithmetic
                        }
                        let d = match pbc {
                            None => [
                                positions[j][0] - pi[0],
                                positions[j][1] - pi[1],
                                positions[j][2] - pi[2],
                            ],
                            Some((h, h_inv)) => {
                                let raw = [
                                    positions[j][0] - pi[0],
                                    positions[j][1] - pi[1],
                                    positions[j][2] - pi[2],
                                ];
                                if self.is_ortho {
                                    // Safe: every atom is pre-wrapped into
                                    // `[0, L)³` by `compute_cell_list_sync`,
                                    // and `cell_list::fold_cell_index_pbc`
                                    // ensures the cell index is consistent
                                    // with the wrapped Cartesian position
                                    // even when f32 round-trip rounding
                                    // pushes `s` to exactly `1.0`.  The
                                    // integer shift is therefore exact.
                                    let shift = shift_opt.unwrap();
                                    [raw[0] + shift[0], raw[1] + shift[1], raw[2] + shift[2]]
                                } else {
                                    // Triclinic: atoms may be outside fractional [0,1);
                                    // per-pair min_image is always correct.
                                    min_image_mat_f32(raw, h, h_inv)
                                }
                            }
                        };
                        let r_sq = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                        if r_sq < cutoff_sq && r_sq > MIN_DIST_SQ {
                            visitor(j, d, r_sq);
                        }
                    }
                }
            }
        }
    }
}

// ── CpuEngine ─────────────────────────────────────────────────────────────────

/// CPU compute engine — rayon-parallelized half-pair O(N²) EAM.
///
/// Create with [`CpuEngine::new()`] (always succeeds — no GPU required).
/// Call [`CpuEngine::compute_sync()`] with the same argument shape as
/// `ComputeEngine::compute_sync()`.
///
/// # Rayon thread warm-up
///
/// Rayon worker threads park themselves after ~100 ms of inactivity, causing
/// the first `compute_*` call after a quiet period to incur a 50–100 ms
/// thread wake-up penalty.  `new()` spawns a lightweight keep-alive thread
/// that posts a no-op task to the global Rayon pool every 50 ms, preventing
/// workers from parking while any `CpuEngine` instance is live.
/// Drop the engine to stop the keep-alive thread (exits within 50 ms).
pub struct CpuEngine {
    // Under Miri the keep-alive thread is skipped entirely (rayon::spawn
    // triggers a crossbeam-epoch Stacked Borrows false positive in Miri).
    // The fields are cfg-gated so CpuEngine is a zero-sized type under Miri,
    // matching the behaviour of the serial #[cfg(miri)] compute paths.
    #[cfg(not(miri))]
    stop_flag: Arc<AtomicBool>,
    #[cfg(not(miri))]
    _keepalive: std::thread::JoinHandle<()>,
}

impl CpuEngine {
    /// Construct a new `CpuEngine`.
    ///
    /// Outside Miri: also starts a keep-alive thread that posts a no-op
    /// task to the global Rayon pool every 50 ms, preventing worker threads
    /// from parking between compute calls.
    ///
    /// Under Miri: keep-alive is skipped because `rayon::spawn` triggers a
    /// Stacked Borrows false positive inside `crossbeam-epoch` (a known Miri
    /// limitation with that crate's unsafe pointer aliasing).
    pub fn new() -> Self {
        #[cfg(not(miri))]
        {
            let stop_flag = Arc::new(AtomicBool::new(false));
            let stop_clone = Arc::clone(&stop_flag);
            let _keepalive = std::thread::spawn(move || {
                while !stop_clone.load(Ordering::Relaxed) {
                    rayon::spawn(|| {});
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
            });
            Self {
                stop_flag,
                _keepalive,
            }
        }
        #[cfg(miri)]
        Self {}
    }

    /// Compute EAM forces, energy, and the global virial tensor using
    /// rayon-parallelised half-pair CPU code.
    ///
    /// This is the **fast path**: it does NOT allocate the N × 6 × 8-byte
    /// per-atom virial accumulator (nor the analogous per-atom density and
    /// embedding-energy buffers) that [`Self::compute_sync_with_per_atom`]
    /// needs.  For a 4-million-atom system on 16 rayon threads that saves
    /// about 3 GB of transient RAM and removes the largest block from the
    /// tree-reduce memory fan-out.
    ///
    /// The returned [`ComputeResult`] therefore has `virial_per_atom`,
    /// `densities`, and `embedding_energies` as empty `Vec`s — callers that
    /// need those should use [`Self::compute_sync_with_per_atom`].
    ///
    /// # Arguments
    /// * `positions`   — `[x, y, z, w]` per atom (w ignored, matches GPU layout)
    /// * `atom_types`  — element index per atom
    /// * `cell`        — `Some([[ax,ay,az],[bx,by,bz],[cx,cy,cz]])` for triclinic PBC,
    ///   or `None` for non-periodic (ASE/LAMMPS row-vector convention)
    /// * `potential`   — parsed EAM potential tables
    pub fn compute_sync(
        &self,
        positions: &[[f32; 4]],
        atom_types: &[u32],
        cell: Option<[[f32; 3]; 3]>,
        potential: &EamPotential,
    ) -> Result<ComputeResult, CreamError> {
        self.compute_sync_inner(positions, atom_types, cell, potential, false)
    }

    /// Same as [`Self::compute_sync`] but also populates `virial_per_atom`,
    /// `densities`, and `embedding_energies` on the returned [`ComputeResult`].
    ///
    /// Uses T × N × 48 B more peak RAM than [`Self::compute_sync`] for the
    /// per-atom virial accumulator (where T is the rayon thread count), plus
    /// the ρ and F(ρ) per-atom buffers from Pass 1.  For large N the reduce
    /// step's total memory fan-out can be ~3× larger than the fast path,
    /// which is visible as a step in wall-time.
    ///
    /// Use this only when the caller actually needs the per-atom quantities
    /// — typically only from `compute_per_atom` on the Python side, or for
    /// OVITO-style defect analysis.
    pub fn compute_sync_with_per_atom(
        &self,
        positions: &[[f32; 4]],
        atom_types: &[u32],
        cell: Option<[[f32; 3]; 3]>,
        potential: &EamPotential,
    ) -> Result<ComputeResult, CreamError> {
        self.compute_sync_inner(positions, atom_types, cell, potential, true)
    }

    /// Internal implementation for [`Self::compute_sync`] and
    /// [`Self::compute_sync_with_per_atom`].  The `include_per_atom` flag
    /// controls whether the N × 6 per-atom virial accumulator is allocated
    /// and whether the matching writes in the hot loop are executed.
    fn compute_sync_inner(
        &self,
        positions: &[[f32; 4]],
        atom_types: &[u32],
        cell: Option<[[f32; 3]; 3]>,
        potential: &EamPotential,
        include_per_atom: bool,
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
        let n_elem = potential.elements.len();
        for (i, &t) in atom_types.iter().enumerate() {
            if (t as usize) >= n_elem {
                return Err(CreamError::InvalidInput(format!(
                    "atom_types[{i}]={t} is out of range (n_elem={n_elem})"
                )));
            }
        }

        let dr_inv = 1.0 / potential.dr;
        let drho_inv = 1.0 / potential.drho;
        let cutoff_sq = potential.cutoff_angstrom * potential.cutoff_angstrom;

        // Validate cell size for minimum image convention.
        if let Some(ref h) = cell {
            let twice_cut = 2.0 * potential.cutoff_angstrom;
            let a = h[0];
            let b = h[1];
            let c = h[2];
            let vol = (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0]))
                .abs();
            let cross = |u: [f32; 3], v: [f32; 3]| {
                [
                    u[1] * v[2] - u[2] * v[1],
                    u[2] * v[0] - u[0] * v[2],
                    u[0] * v[1] - u[1] * v[0],
                ]
            };
            let norm = |w: [f32; 3]| (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
            let l_min = [
                vol / norm(cross(b, c)),
                vol / norm(cross(a, c)),
                vol / norm(cross(a, b)),
            ]
            .into_iter()
            .fold(f32::INFINITY, f32::min);
            if l_min <= twice_cut {
                return Err(CreamError::InvalidInput(format!(
                    "Cell too small for minimum image convention: shortest perpendicular height \
                     {l_min:.3} Å ≤ 2 × cutoff = {twice_cut:.3} Å"
                )));
            }
        }

        // Pre-compute H⁻¹ once (not inside each thread's inner loop).
        let pbc: Option<PbcMatricesF32> = cell.and_then(|h| mat3_inv_f32(&h).map(|hi| (h, hi)));

        // Displacement helper — computes minimum-image vector rⱼ − rᵢ.
        let displace = |i: usize, j: usize| -> [f32; 3] {
            let raw = [
                positions[j][0] - positions[i][0],
                positions[j][1] - positions[i][1],
                positions[j][2] - positions[i][2],
            ];
            match &pbc {
                None => raw,
                Some((h, h_inv)) => min_image_mat_f32(raw, h, h_inv),
            }
        };

        // ── Pass 1: Half-pair density accumulation ──────────────────────────
        //
        // For each pair (i, j) with j > i, both atoms receive a contribution:
        //   ρᵢ += f_{type_j}(r)   — atom j's density function contributes to atom i
        //   ρⱼ += f_{type_i}(r)   — atom i's density function contributes to atom j
        //
        // Interleaved distribution: thread t processes outer rows {t, t+T, t+2T, …}.
        // Row i has (N−i−1) pairs, so low-i rows are heavy and high-i rows are light.
        // Interleaving perfectly balances load across threads and limits allocations
        // to exactly T Vec<f32; N> accumulators (T = thread count).  This avoids the
        // O(P×N) reduce overhead of into_par_iter().fold() where work-stealing can
        // create P >> T partitions.

        // Under Miri we use a sequential loop to avoid the Stacked Borrows
        // violation that crossbeam-epoch (used internally by rayon) triggers.
        #[cfg(not(miri))]
        let densities: Vec<f32> = {
            let n_threads = rayon::current_num_threads().max(1);
            (0..n_threads)
                .into_par_iter()
                .map(|t| {
                    // f64 thread-local accumulator: same throughput as f32 on x86-64,
                    // eliminates rounding drift from non-deterministic addition order.
                    let mut rho_acc = vec![0.0f64; n];
                    let mut i = t;
                    while i < n {
                        let type_i = atom_types[i] as usize;
                        for j in (i + 1)..n {
                            let d = displace(i, j);
                            let r_sq = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                            if r_sq < cutoff_sq && r_sq > MIN_DIST_SQ {
                                let r = r_sq.sqrt();
                                let type_j = atom_types[j] as usize;
                                rho_acc[i] +=
                                    lookup_by_r(&potential.rho_tables[type_j], dr_inv, r) as f64;
                                rho_acc[j] +=
                                    lookup_by_r(&potential.rho_tables[type_i], dr_inv, r) as f64;
                            }
                        }
                        i += n_threads;
                    }
                    rho_acc
                })
                .reduce(
                    || vec![0.0f64; n],
                    |mut a, b| {
                        for k in 0..n {
                            a[k] += b[k];
                        }
                        a
                    },
                )
                .into_iter()
                .map(|x| x as f32)
                .collect()
        };

        #[cfg(miri)]
        let densities: Vec<f32> = (0..n).fold(vec![0.0f32; n], |mut rho_acc, i| {
            let type_i = atom_types[i] as usize;
            for j in (i + 1)..n {
                let d = displace(i, j);
                let r_sq = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                if r_sq < cutoff_sq && r_sq > MIN_DIST_SQ {
                    let r = r_sq.sqrt();
                    let type_j = atom_types[j] as usize;
                    rho_acc[i] += lookup_by_r(&potential.rho_tables[type_j], dr_inv, r);
                    rho_acc[j] += lookup_by_r(&potential.rho_tables[type_i], dr_inv, r);
                }
            }
            rho_acc
        });

        // ── Pre-compute embedding quantities (O(N), serial) ─────────────────
        //
        // F'_α(ρᵢ) and F_α(ρᵢ) for each atom, needed by Pass 2.
        // These depend on the densities computed in Pass 1.

        let df_embed: Vec<f32> = (0..n)
            .map(|i| {
                lookup_by_rho(
                    &potential.d_embed_tables[atom_types[i] as usize],
                    drho_inv,
                    densities[i],
                )
            })
            .collect();
        let embed_energies: Vec<f32> = (0..n)
            .map(|i| {
                lookup_by_rho(
                    &potential.embed_tables[atom_types[i] as usize],
                    drho_inv,
                    densities[i],
                )
            })
            .collect();

        // ── Pass 2: Half-pair forces + pair energy ──────────────────────────
        //
        // For each pair (i, j) with j > i:
        //   coeff = F'_α(ρᵢ)·f'_β(r) + F'_β(ρⱼ)·f'_α(r) + φ'_αβ(r)
        //   Fᵢ += coeff · r̂ᵢⱼ          (r̂ᵢⱼ = d / r, d = rⱼ − rᵢ)
        //   Fⱼ −= coeff · r̂ᵢⱼ          (Newton's 3rd law)
        //   pair_energy_i += 0.5 · φ(r)
        //   pair_energy_j += 0.5 · φ(r)
        //   virial_αβ     += d_α · F_ij_β   (Voigt: xx,yy,zz,yz,xz,xy)
        //
        // Same interleaved row distribution as Pass 1: thread t owns rows
        // {t, t+T, t+2T, …}.  Accumulators are (forces[N][3], pair_energies[N], virial[6]).

        // f64 accumulators: memory stays f32, thread-local sums in f64.
        // AccF64 = (forces[N][3], pair_energies[N], virial_total[6], per_atom_virial[N][6]).
        // Per-atom virial receives `0.5 · d_α · F_ij_β` for *each* atom of a pair
        // — matching LAMMPS `compute stress/atom`.  Σᵢ per_atom[i] = virial_total.
        type AccF64 = (Vec<[f64; 3]>, Vec<f64>, [f64; 6], Vec<[f64; 6]>);
        let make_acc = || -> AccF64 {
            (
                vec![[0.0f64; 3]; n],
                vec![0.0f64; n],
                [0.0f64; 6],
                // Skip the N × 48-byte per-atom virial allocation when the
                // caller doesn't need it — at N=4M, T=16 this saves ≈3 GB of
                // peak RAM and removes the slowest line from the reduce tree.
                if include_per_atom {
                    vec![[0.0f64; 6]; n]
                } else {
                    Vec::new()
                },
            )
        };

        #[cfg(not(miri))]
        let (forces, pair_energies, virial_raw, virial_per_atom_raw): (
            Vec<[f32; 3]>,
            Vec<f32>,
            [f64; 6],
            Vec<[f64; 6]>,
        ) = {
            let n_threads = rayon::current_num_threads().max(1);
            let (f64_f, f64_e, f64_v, f64_va) = (0..n_threads)
                .into_par_iter()
                .map(|t| {
                    let (mut f_acc, mut e_acc, mut v_acc, mut va_acc) = make_acc();
                    let mut i = t;
                    while i < n {
                        let type_i = atom_types[i] as usize;
                        let df_i = df_embed[i] as f64;
                        for j in (i + 1)..n {
                            let [dx, dy, dz] = displace(i, j);
                            let r_sq = dx * dx + dy * dy + dz * dz;
                            if r_sq < cutoff_sq && r_sq > MIN_DIST_SQ {
                                let r = r_sq.sqrt();
                                let r_inv = 1.0 / r;
                                let type_j = atom_types[j] as usize;
                                let pidx = EamPotential::pair_index(type_i, type_j, n_elem);
                                let df_j = df_embed[j] as f64;
                                let df_beta_dr =
                                    lookup_by_r(&potential.d_rho_tables[type_j], dr_inv, r) as f64;
                                let df_alpha_dr =
                                    lookup_by_r(&potential.d_rho_tables[type_i], dr_inv, r) as f64;
                                let dphi_dr =
                                    lookup_by_r(&potential.d_pair_tables[pidx], dr_inv, r) as f64;
                                let phi =
                                    lookup_by_r(&potential.pair_tables[pidx], dr_inv, r) as f64;
                                let coeff = df_i * df_beta_dr + df_j * df_alpha_dr + dphi_dr;
                                let r_inv = r_inv as f64;
                                let fx = coeff * dx as f64 * r_inv;
                                let fy = coeff * dy as f64 * r_inv;
                                let fz = coeff * dz as f64 * r_inv;
                                f_acc[i][0] += fx;
                                f_acc[i][1] += fy;
                                f_acc[i][2] += fz;
                                f_acc[j][0] -= fx;
                                f_acc[j][1] -= fy;
                                f_acc[j][2] -= fz;
                                let half_phi = 0.5 * phi;
                                e_acc[i] += half_phi;
                                e_acc[j] += half_phi;
                                // Virial W_αβ += d_α · F_ij_β  (Voigt xx,yy,zz,yz,xz,xy)
                                let dx64 = dx as f64;
                                let dy64 = dy as f64;
                                let dz64 = dz as f64;
                                let w0 = -(dx64 * fx);
                                let w1 = -(dy64 * fy);
                                let w2 = -(dz64 * fz);
                                let w3 = -(dy64 * fz);
                                let w4 = -(dx64 * fz);
                                let w5 = -(dx64 * fy);
                                v_acc[0] += w0;
                                v_acc[1] += w1;
                                v_acc[2] += w2;
                                v_acc[3] += w3;
                                v_acc[4] += w4;
                                v_acc[5] += w5;
                                // Per-atom virial — split the pair contribution
                                // half-and-half between atoms i and j so that
                                // Σₖ per_atom[k] == W_total.  Guarded so we
                                // don't pay the N × 6 random-access L3-miss
                                // cost when the caller doesn't need it.
                                if include_per_atom {
                                    let h0 = 0.5 * w0;
                                    let h1 = 0.5 * w1;
                                    let h2 = 0.5 * w2;
                                    let h3 = 0.5 * w3;
                                    let h4 = 0.5 * w4;
                                    let h5 = 0.5 * w5;
                                    va_acc[i][0] += h0; va_acc[j][0] += h0;
                                    va_acc[i][1] += h1; va_acc[j][1] += h1;
                                    va_acc[i][2] += h2; va_acc[j][2] += h2;
                                    va_acc[i][3] += h3; va_acc[j][3] += h3;
                                    va_acc[i][4] += h4; va_acc[j][4] += h4;
                                    va_acc[i][5] += h5; va_acc[j][5] += h5;
                                }
                            }
                        }
                        i += n_threads;
                    }
                    (f_acc, e_acc, v_acc, va_acc)
                })
                .reduce(make_acc, |(mut af, mut ae, mut av, mut ava), (bf, be, bv, bva)| {
                    for k in 0..n {
                        af[k][0] += bf[k][0];
                        af[k][1] += bf[k][1];
                        af[k][2] += bf[k][2];
                        ae[k] += be[k];
                    }
                    // Merge per-atom virial only when it was actually
                    // allocated — `ava.len() == 0` iff `include_per_atom`
                    // was false, in which case bva is also empty.
                    if !ava.is_empty() {
                        for k in 0..n {
                            for c in 0..6 {
                                ava[k][c] += bva[k][c];
                            }
                        }
                    }
                    for c in 0..6 {
                        av[c] += bv[c];
                    }
                    (af, ae, av, ava)
                });
            // Downcast to f32 for ComputeResult (memory layout).
            let forces = f64_f
                .iter()
                .map(|&[x, y, z]| [x as f32, y as f32, z as f32])
                .collect();
            let pair_energies = f64_e.iter().map(|&e| e as f32).collect();
            (forces, pair_energies, f64_v, f64_va)
        };

        #[cfg(miri)]
        let (forces, pair_energies, virial_raw, virial_per_atom_raw): (
            Vec<[f32; 3]>,
            Vec<f32>,
            [f64; 6],
            Vec<[f64; 6]>,
        ) = {
            let (f64_f, f64_e, f64_v, f64_va) =
                (0..n).fold(make_acc(), |(mut f_acc, mut e_acc, mut v_acc, mut va_acc), i| {
                let type_i = atom_types[i] as usize;
                let df_i = df_embed[i] as f64;
                for j in (i + 1)..n {
                    let [dx, dy, dz] = displace(i, j);
                    let r_sq = dx * dx + dy * dy + dz * dz;
                    if r_sq < cutoff_sq && r_sq > MIN_DIST_SQ {
                        let r = r_sq.sqrt();
                        let r_inv = 1.0 / r as f64;
                        let type_j = atom_types[j] as usize;
                        let pidx = EamPotential::pair_index(type_i, type_j, n_elem);
                        let df_j = df_embed[j] as f64;
                        let df_beta_dr =
                            lookup_by_r(&potential.d_rho_tables[type_j], dr_inv, r) as f64;
                        let df_alpha_dr =
                            lookup_by_r(&potential.d_rho_tables[type_i], dr_inv, r) as f64;
                        let dphi_dr = lookup_by_r(&potential.d_pair_tables[pidx], dr_inv, r) as f64;
                        let phi = lookup_by_r(&potential.pair_tables[pidx], dr_inv, r) as f64;
                        let coeff = df_i * df_beta_dr + df_j * df_alpha_dr + dphi_dr;
                        let fx = coeff * dx as f64 * r_inv;
                        let fy = coeff * dy as f64 * r_inv;
                        let fz = coeff * dz as f64 * r_inv;
                        f_acc[i][0] += fx; f_acc[i][1] += fy; f_acc[i][2] += fz;
                        f_acc[j][0] -= fx; f_acc[j][1] -= fy; f_acc[j][2] -= fz;
                        let half_phi = 0.5 * phi;
                        e_acc[i] += half_phi; e_acc[j] += half_phi;
                        let dx64 = dx as f64; let dy64 = dy as f64; let dz64 = dz as f64;
                        // NOTE: sign convention must match the not-miri branch above
                        // (user's rc8 ASE/LAMMPS convention: compressive σ < 0).
                        // rc8 had an inconsistency here — miri-only builds were
                        // silently producing the opposite sign.  Fixed.
                        let w0 = -(dx64 * fx); let w1 = -(dy64 * fy); let w2 = -(dz64 * fz);
                        let w3 = -(dy64 * fz); let w4 = -(dx64 * fz); let w5 = -(dx64 * fy);
                        v_acc[0] += w0; v_acc[1] += w1; v_acc[2] += w2;
                        v_acc[3] += w3; v_acc[4] += w4; v_acc[5] += w5;
                        if include_per_atom {
                            let h0 = 0.5 * w0; let h1 = 0.5 * w1; let h2 = 0.5 * w2;
                            let h3 = 0.5 * w3; let h4 = 0.5 * w4; let h5 = 0.5 * w5;
                            va_acc[i][0] += h0; va_acc[j][0] += h0;
                            va_acc[i][1] += h1; va_acc[j][1] += h1;
                            va_acc[i][2] += h2; va_acc[j][2] += h2;
                            va_acc[i][3] += h3; va_acc[j][3] += h3;
                            va_acc[i][4] += h4; va_acc[j][4] += h4;
                            va_acc[i][5] += h5; va_acc[j][5] += h5;
                        }
                    }
                }
                (f_acc, e_acc, v_acc, va_acc)
            });
            let forces = f64_f
                .iter()
                .map(|&[x, y, z]| [x as f32, y as f32, z as f32])
                .collect();
            let pair_energies = f64_e.iter().map(|&e| e as f32).collect();
            (forces, pair_energies, f64_v, f64_va)
        };

        // ── Combine embedding energy + pair energy ──────────────────────────

        let energy_per_atom: Vec<f32> = (0..n)
            .map(|i| embed_energies[i] + pair_energies[i])
            .collect();
        let energy: f32 = energy_per_atom.iter().sum();

        // ── Virial → stress (eV/Å³) ──────────────────────────────────────────
        // σ_αβ = −W_αβ / V   for periodic systems.
        // Zero for non-periodic clusters (no well-defined volume).
        //
        // `virial_per_atom_raw[i]` carries the *raw* atomic virial contribution
        // in eV units (half-and-half split from each pair), matching LAMMPS
        // `compute stress/atom` — we return it as-is so users can divide by
        // their preferred per-atom volume (OVITO assumes V/N).
        let virial: [f64; 6] = match cell {
            Some(h) => {
                let a = h[0]; let b = h[1]; let c = h[2];
                let vol = ((a[0] * (b[1] * c[2] - b[2] * c[1])
                    - a[1] * (b[0] * c[2] - b[2] * c[0])
                    + a[2] * (b[0] * c[1] - b[1] * c[0])) as f64)
                    .abs();
                if vol > 0.0 {
                    let v = &virial_raw;
                    [-v[0]/vol, -v[1]/vol, -v[2]/vol, -v[3]/vol, -v[4]/vol, -v[5]/vol]
                } else {
                    [0.0; 6]
                }
            }
            None => [0.0; 6],
        };

        Ok(ComputeResult {
            forces,
            energy,
            energy_per_atom,
            virial,
            virial_per_atom: virial_per_atom_raw,
            densities: densities.clone(),
            embedding_energies: embed_energies.clone(),
        })
    }

    /// Compute EAM forces, energy, and the global virial tensor using a
    /// **cell list** for O(N) neighbour search.
    ///
    /// Supports orthorhombic PBC and finite (non-PBC) systems.
    /// Triclinic PBC transparently falls back to [`Self::compute_sync`].
    ///
    /// This is the **fast path** for cell-list mode: it does NOT allocate the
    /// N × 6 × 8-byte per-atom virial accumulator (nor the analogous per-atom
    /// density and embedding-energy buffers) — see [`Self::compute_sync`] for
    /// the AllPairs counterpart's memory math.
    ///
    /// Parallelism: **half-pair cell list** — each atom pair (i, j) with j > i
    /// is visited exactly once via the cell-list neighbour search.  Newton's
    /// 3rd law is applied inside the visitor, so both atoms receive their
    /// density / force contribution in a single pass.  Thread-local
    /// accumulators (interleaved row distribution, same as [`Self::compute_sync`])
    /// resolve the write-conflict to atom j without any atomic operations.
    pub fn compute_cell_list_sync(
        &self,
        positions: &[[f32; 4]],
        atom_types: &[u32],
        cell: Option<[[f32; 3]; 3]>,
        potential: &EamPotential,
    ) -> Result<ComputeResult, CreamError> {
        self.compute_cell_list_sync_inner(positions, atom_types, cell, potential, false)
    }

    /// Same as [`Self::compute_cell_list_sync`] but also populates
    /// `virial_per_atom`, `densities`, and `embedding_energies` on the
    /// returned [`ComputeResult`].  See [`Self::compute_sync_with_per_atom`]
    /// for the memory-cost trade-off.
    pub fn compute_cell_list_sync_with_per_atom(
        &self,
        positions: &[[f32; 4]],
        atom_types: &[u32],
        cell: Option<[[f32; 3]; 3]>,
        potential: &EamPotential,
    ) -> Result<ComputeResult, CreamError> {
        self.compute_cell_list_sync_inner(positions, atom_types, cell, potential, true)
    }

    /// Internal implementation for the cell-list compute methods.  See
    /// [`Self::compute_sync_inner`] for the semantics of `include_per_atom`.
    fn compute_cell_list_sync_inner(
        &self,
        positions: &[[f32; 4]],
        atom_types: &[u32],
        cell: Option<[[f32; 3]; 3]>,
        potential: &EamPotential,
        include_per_atom: bool,
    ) -> Result<ComputeResult, CreamError> {
        let n = positions.len();
        if n == 0 {
            return Err(CreamError::InvalidInput("positions is empty".into()));
        }
        if atom_types.len() != n {
            return Err(CreamError::InvalidInput(format!(
                "atom_types length {} != positions length {}",
                atom_types.len(),
                n
            )));
        }
        let n_elem = potential.elements.len();
        for (i, &t) in atom_types.iter().enumerate() {
            if (t as usize) >= n_elem {
                return Err(CreamError::InvalidInput(format!(
                    "atom_types[{i}]={t} is out of range (n_elem={n_elem})"
                )));
            }
        }

        // Cell-size validation (same as compute_sync)
        if let Some(ref h) = cell {
            let twice_cut = 2.0 * potential.cutoff_angstrom;
            let a = h[0];
            let b = h[1];
            let c = h[2];
            let vol = (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0]))
                .abs();
            let cross = |u: [f32; 3], v: [f32; 3]| {
                [
                    u[1] * v[2] - u[2] * v[1],
                    u[2] * v[0] - u[0] * v[2],
                    u[0] * v[1] - u[1] * v[0],
                ]
            };
            let norm = |w: [f32; 3]| (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
            let l_min = [
                vol / norm(cross(b, c)),
                vol / norm(cross(a, c)),
                vol / norm(cross(a, b)),
            ]
            .into_iter()
            .fold(f32::INFINITY, f32::min);
            if l_min <= twice_cut {
                return Err(CreamError::InvalidInput(format!(
                    "Cell too small for minimum image convention: {l_min:.3} A <= 2*cutoff = {twice_cut:.3} A"
                )));
            }
        }

        // ── Wrap positions to [0, L)³ fractional for PBC ─────────────────────
        //
        // # Why this is required (bug fix)
        //
        // The `is_ortho = true` fast path in `for_each_(forward_)neighbor`
        // computes the PBC shift once per stencil offset via
        // `pbc_cell_and_shift`, then uses `d = positions[j] - pi + shift`
        // **without** per-pair `min_image_mat_f32`.  That shift is exact only
        // when **both** atoms have fractional coordinates in [0, 1) — i.e.
        // their Cartesian positions are in the primary image box.
        //
        // `CellListData::build` uses `rem_euclid` to assign cells, which is
        // correct for atoms whose positions escape [0, L)³ (e.g. after
        // `rattle()`, MD thermalisation, or any external displacement).
        // But `pbc_cell_and_shift` computes the Cartesian shift from the
        // *cell index delta* — it assumes atom i is at its cell-canonical
        // position.  If i's fractional coord is 1.1 (one image out), the
        // shift is off by exactly one lattice vector, producing
        // very-far-neighbour d values and missing real neighbours.
        //
        // Symptom in the failing test suite: Cubic/Tetragonal/Orthorhombic
        // supercells (is_ortho == true) show ~100s eV energy mismatch and
        // ~10 eV/Å force norm vs CPU AllPairs, with 500+ mismatched atoms
        // out of 500.  Hexagonal/Rhombohedral/Monoclinic/Triclinic
        // (is_ortho == false) go through `min_image_mat_f32` and pass.
        //
        // # Fix
        //
        // Wrap positions once (O(N), no per-pair round() inside hot loop)
        // so every atom has fractional coord in [0, 1).  The `is_ortho`
        // fast path then stays correct without extra work per pair.
        //
        // The wrap is a pure shift by a lattice vector, so it does not
        // change pair distances under PBC — forces and energy are
        // unaffected in the correct-cell case, and are fixed in the
        // escaped-cell case.  Original `positions` is never mutated;
        // we build a local owned copy and bind `positions_eff` to either
        // the wrapped copy or the original slice.
        let positions_wrapped_storage: Vec<[f32; 4]>;
        let positions_eff: &[[f32; 4]] = match cell {
            Some(h) => match mat3_inv_f32(&h) {
                Some(h_inv) => {
                    positions_wrapped_storage = positions
                        .iter()
                        .map(|p| {
                            let s0 = p[0] * h_inv[0][0] + p[1] * h_inv[1][0] + p[2] * h_inv[2][0];
                            let s1 = p[0] * h_inv[0][1] + p[1] * h_inv[1][1] + p[2] * h_inv[2][1];
                            let s2 = p[0] * h_inv[0][2] + p[1] * h_inv[1][2] + p[2] * h_inv[2][2];
                            let sf0 = s0 - s0.floor();
                            let sf1 = s1 - s1.floor();
                            let sf2 = s2 - s2.floor();
                            // Guard against floor-rounding that produces sf == 1.0
                            // (can happen when s is a tiny negative number very
                            // close to 0 → floor = -1 → sf = 1.0 exactly in f32).
                            let sf0 = if sf0 >= 1.0 { 0.0 } else { sf0 };
                            let sf1 = if sf1 >= 1.0 { 0.0 } else { sf1 };
                            let sf2 = if sf2 >= 1.0 { 0.0 } else { sf2 };
                            let x0 = sf0 * h[0][0] + sf1 * h[1][0] + sf2 * h[2][0];
                            let x1 = sf0 * h[0][1] + sf1 * h[1][1] + sf2 * h[2][1];
                            let x2 = sf0 * h[0][2] + sf1 * h[1][2] + sf2 * h[2][2];
                            [x0, x1, x2, p[3]]
                        })
                        .collect();
                    &positions_wrapped_storage
                }
                None => positions, // singular cell — build() will also fail and we fall through
            },
            None => positions, // non-PBC: no wrapping
        };

        // Build cell list.  Returns None only for empty input or a singular
        // cell matrix (degenerate geometry) — both ortho and triclinic PBC
        // are now fully supported via the fractional-coordinate grid.
        let cl = match CellListData::build(positions_eff, cell, potential.cutoff_angstrom) {
            Some(cl) => cl,
            None => {
                return self.compute_sync_inner(
                    positions,
                    atom_types,
                    cell,
                    potential,
                    include_per_atom,
                )
            }
        };

        let dr_inv = 1.0 / potential.dr;
        let drho_inv = 1.0 / potential.drho;
        let cutoff_sq = potential.cutoff_angstrom * potential.cutoff_angstrom;
        let pbc: Option<PbcMatricesF32> = cell.and_then(|h| mat3_inv_f32(&h).map(|hi| (h, hi)));

        // Shadow `positions` so the rest of the function sees the wrapped copy.
        // This minimises churn: the two parallel blocks below continue to call
        // `cl.for_each_forward_neighbor(i, positions, ...)`.
        let positions = positions_eff;

        // Pass 1: density (half-pair cell list, parallel)
        //
        // For each pair (i, j) with j > i found by the cell-list:
        //   ρᵢ += f_{type_j}(r)
        //   ρⱼ += f_{type_i}(r)
        //
        // Interleaved distribution: thread t owns outer atoms {t, t+T, t+2T, …}.
        // Because the visitor writes to rho[j] as well as rho[i], each thread
        // carries a full-size accumulator; accumulators are merged at the end.
        // Pass 1 uses for_each_forward_neighbor (j > i guard INSIDE tight loop)
        // so no distance computation is wasted on j < i pairs.
        // f64 thread-local accumulators for numerical stability.
        #[cfg(not(miri))]
        let densities: Vec<f32> = {
            let n_threads = rayon::current_num_threads().max(1);
            (0..n_threads)
                .into_par_iter()
                .map(|t| {
                    let mut rho_acc = vec![0.0f64; n];
                    let mut i = t;
                    while i < n {
                        let type_i = atom_types[i] as usize;
                        cl.for_each_forward_neighbor(
                            i,
                            positions,
                            cutoff_sq,
                            &pbc,
                            |j, _d, r_sq| {
                                // j > i guaranteed by for_each_forward_neighbor
                                let r = r_sq.sqrt();
                                let type_j = atom_types[j] as usize;
                                rho_acc[i] +=
                                    lookup_by_r(&potential.rho_tables[type_j], dr_inv, r) as f64;
                                rho_acc[j] +=
                                    lookup_by_r(&potential.rho_tables[type_i], dr_inv, r) as f64;
                            },
                        );
                        i += n_threads;
                    }
                    rho_acc
                })
                .reduce(
                    || vec![0.0f64; n],
                    |mut a, b| {
                        for k in 0..n {
                            a[k] += b[k];
                        }
                        a
                    },
                )
                .into_iter()
                .map(|x| x as f32)
                .collect()
        };

        #[cfg(miri)]
        let densities: Vec<f32> = {
            let mut rho_acc = vec![0.0f64; n];
            for i in 0..n {
                let type_i = atom_types[i] as usize;
                cl.for_each_forward_neighbor(i, positions, cutoff_sq, &pbc, |j, _d, r_sq| {
                    let r = r_sq.sqrt();
                    let type_j = atom_types[j] as usize;
                    rho_acc[i] += lookup_by_r(&potential.rho_tables[type_j], dr_inv, r) as f64;
                    rho_acc[j] += lookup_by_r(&potential.rho_tables[type_i], dr_inv, r) as f64;
                });
            }
            rho_acc.into_iter().map(|x| x as f32).collect()
        };

        // Pre-compute embedding derivatives (O(N), serial)
        let df_embed: Vec<f32> = (0..n)
            .map(|i| {
                lookup_by_rho(
                    &potential.d_embed_tables[atom_types[i] as usize],
                    drho_inv,
                    densities[i],
                )
            })
            .collect();
        let embed_energies: Vec<f32> = (0..n)
            .map(|i| {
                lookup_by_rho(
                    &potential.embed_tables[atom_types[i] as usize],
                    drho_inv,
                    densities[i],
                )
            })
            .collect();

        // Pass 2: half-pair forces + energy via for_each_forward_neighbor + f64 accs.
        type AccF64 = (Vec<[f64; 3]>, Vec<f64>, [f64; 6], Vec<[f64; 6]>);
        let make_acc = || -> AccF64 {
            (
                vec![[0.0f64; 3]; n],
                vec![0.0f64; n],
                [0.0f64; 6],
                // See AllPairs make_acc for the memory-cost rationale.
                if include_per_atom {
                    vec![[0.0f64; 6]; n]
                } else {
                    Vec::new()
                },
            )
        };

        #[cfg(not(miri))]
        let (forces, pair_energies, virial_raw, virial_per_atom_raw): (
            Vec<[f32; 3]>,
            Vec<f32>,
            [f64; 6],
            Vec<[f64; 6]>,
        ) = {
            let n_threads = rayon::current_num_threads().max(1);
            let (f64_f, f64_e, f64_v, f64_va) = (0..n_threads)
                .into_par_iter()
                .map(|t| {
                    let (mut f_acc, mut e_acc, mut v_acc, mut va_acc) = make_acc();
                    let mut i = t;
                    while i < n {
                        let type_i = atom_types[i] as usize;
                        let df_i = df_embed[i] as f64;
                        cl.for_each_forward_neighbor(
                            i,
                            positions,
                            cutoff_sq,
                            &pbc,
                            |j, [dx, dy, dz], r_sq| {
                                // j > i guaranteed — no if-guard needed
                                let r = r_sq.sqrt();
                                let r_inv = 1.0 / r as f64;
                                let type_j = atom_types[j] as usize;
                                let pidx = EamPotential::pair_index(type_i, type_j, n_elem);
                                let df_j = df_embed[j] as f64;
                                let df_beta_dr =
                                    lookup_by_r(&potential.d_rho_tables[type_j], dr_inv, r) as f64;
                                let df_alpha_dr =
                                    lookup_by_r(&potential.d_rho_tables[type_i], dr_inv, r) as f64;
                                let dphi_dr =
                                    lookup_by_r(&potential.d_pair_tables[pidx], dr_inv, r) as f64;
                                let phi =
                                    lookup_by_r(&potential.pair_tables[pidx], dr_inv, r) as f64;
                                let coeff = df_i * df_beta_dr + df_j * df_alpha_dr + dphi_dr;
                                let fx = coeff * dx as f64 * r_inv;
                                let fy = coeff * dy as f64 * r_inv;
                                let fz = coeff * dz as f64 * r_inv;
                                f_acc[i][0] += fx; f_acc[i][1] += fy; f_acc[i][2] += fz;
                                f_acc[j][0] -= fx; f_acc[j][1] -= fy; f_acc[j][2] -= fz;
                                let half_phi = 0.5 * phi;
                                e_acc[i] += half_phi; e_acc[j] += half_phi;
                                // Virial W_αβ += d_α · F_ij_β  (Voigt xx,yy,zz,yz,xz,xy)
                                let dx64 = dx as f64;
                                let dy64 = dy as f64;
                                let dz64 = dz as f64;
                                let w0 = -(dx64 * fx);
                                let w1 = -(dy64 * fy);
                                let w2 = -(dz64 * fz);
                                let w3 = -(dy64 * fz);
                                let w4 = -(dx64 * fz);
                                let w5 = -(dx64 * fy);
                                v_acc[0] += w0; v_acc[1] += w1;
                                v_acc[2] += w2; v_acc[3] += w3;
                                v_acc[4] += w4; v_acc[5] += w5;
                                // Per-atom virial (half-and-half split).
                                // Guarded for the same reason as AllPairs.
                                if include_per_atom {
                                    let h0 = 0.5 * w0; let h1 = 0.5 * w1;
                                    let h2 = 0.5 * w2; let h3 = 0.5 * w3;
                                    let h4 = 0.5 * w4; let h5 = 0.5 * w5;
                                    va_acc[i][0] += h0; va_acc[j][0] += h0;
                                    va_acc[i][1] += h1; va_acc[j][1] += h1;
                                    va_acc[i][2] += h2; va_acc[j][2] += h2;
                                    va_acc[i][3] += h3; va_acc[j][3] += h3;
                                    va_acc[i][4] += h4; va_acc[j][4] += h4;
                                    va_acc[i][5] += h5; va_acc[j][5] += h5;
                                }
                            },
                        );
                        i += n_threads;
                    }
                    (f_acc, e_acc, v_acc, va_acc)
                })
                .reduce(make_acc, |(mut af, mut ae, mut av, mut ava), (bf, be, bv, bva)| {
                    for k in 0..n {
                        af[k][0] += bf[k][0];
                        af[k][1] += bf[k][1];
                        af[k][2] += bf[k][2];
                        ae[k] += be[k];
                    }
                    // See AllPairs reduce for the rationale.
                    if !ava.is_empty() {
                        for k in 0..n {
                            for c in 0..6 {
                                ava[k][c] += bva[k][c];
                            }
                        }
                    }
                    for c in 0..6 {
                        av[c] += bv[c];
                    }
                    (af, ae, av, ava)
                });
            let forces = f64_f
                .iter()
                .map(|&[x, y, z]| [x as f32, y as f32, z as f32])
                .collect();
            let pair_energies = f64_e.iter().map(|&e| e as f32).collect();
            (forces, pair_energies, f64_v, f64_va)
        };

        #[cfg(miri)]
        let (forces, pair_energies, virial_raw, virial_per_atom_raw): (
            Vec<[f32; 3]>,
            Vec<f32>,
            [f64; 6],
            Vec<[f64; 6]>,
        ) = {
            let (mut f_acc, mut e_acc, mut v_acc, mut va_acc) = make_acc();
            for i in 0..n {
                let type_i = atom_types[i] as usize;
                let df_i = df_embed[i] as f64;
                cl.for_each_forward_neighbor(
                    i,
                    positions,
                    cutoff_sq,
                    &pbc,
                    |j, [dx, dy, dz], r_sq| {
                        let r = r_sq.sqrt();
                        let r_inv = 1.0 / r as f64;
                        let type_j = atom_types[j] as usize;
                        let pidx = EamPotential::pair_index(type_i, type_j, n_elem);
                        let df_j = df_embed[j] as f64;
                        let df_beta_dr =
                            lookup_by_r(&potential.d_rho_tables[type_j], dr_inv, r) as f64;
                        let df_alpha_dr =
                            lookup_by_r(&potential.d_rho_tables[type_i], dr_inv, r) as f64;
                        let dphi_dr = lookup_by_r(&potential.d_pair_tables[pidx], dr_inv, r) as f64;
                        let phi = lookup_by_r(&potential.pair_tables[pidx], dr_inv, r) as f64;
                        let coeff = df_i * df_beta_dr + df_j * df_alpha_dr + dphi_dr;
                        let fx = coeff * dx as f64 * r_inv;
                        let fy = coeff * dy as f64 * r_inv;
                        let fz = coeff * dz as f64 * r_inv;
                        f_acc[i][0] += fx; f_acc[i][1] += fy; f_acc[i][2] += fz;
                        f_acc[j][0] -= fx; f_acc[j][1] -= fy; f_acc[j][2] -= fz;
                        let half_phi = 0.5 * phi;
                        e_acc[i] += half_phi; e_acc[j] += half_phi;
                        let dx64 = dx as f64; let dy64 = dy as f64; let dz64 = dz as f64;
                        // Sign convention must match the not-miri branch above
                        // (rc8 had a latent inconsistency — fixed).
                        let w0 = -(dx64 * fx); let w1 = -(dy64 * fy); let w2 = -(dz64 * fz);
                        let w3 = -(dy64 * fz); let w4 = -(dx64 * fz); let w5 = -(dx64 * fy);
                        v_acc[0] += w0; v_acc[1] += w1; v_acc[2] += w2;
                        v_acc[3] += w3; v_acc[4] += w4; v_acc[5] += w5;
                        if include_per_atom {
                            let h0 = 0.5 * w0; let h1 = 0.5 * w1; let h2 = 0.5 * w2;
                            let h3 = 0.5 * w3; let h4 = 0.5 * w4; let h5 = 0.5 * w5;
                            va_acc[i][0] += h0; va_acc[j][0] += h0;
                            va_acc[i][1] += h1; va_acc[j][1] += h1;
                            va_acc[i][2] += h2; va_acc[j][2] += h2;
                            va_acc[i][3] += h3; va_acc[j][3] += h3;
                            va_acc[i][4] += h4; va_acc[j][4] += h4;
                            va_acc[i][5] += h5; va_acc[j][5] += h5;
                        }
                    },
                );
            }
            let forces = f_acc
                .iter()
                .map(|&[x, y, z]| [x as f32, y as f32, z as f32])
                .collect();
            let pair_energies = e_acc.iter().map(|&e| e as f32).collect();
            (forces, pair_energies, v_acc, va_acc)
        };

        let energy_per_atom: Vec<f32> = (0..n)
            .map(|i| embed_energies[i] + pair_energies[i])
            .collect();
        let energy: f32 = energy_per_atom.iter().sum();

        // Virial → stress (eV/Å³): σ = −W/V for PBC, zeros for non-PBC.
        let virial: [f64; 6] = match cell {
            Some(h) => {
                let a = h[0]; let b = h[1]; let c = h[2];
                let vol = ((a[0] * (b[1] * c[2] - b[2] * c[1])
                    - a[1] * (b[0] * c[2] - b[2] * c[0])
                    + a[2] * (b[0] * c[1] - b[1] * c[0])) as f64)
                    .abs();
                if vol > 0.0 {
                    let v = &virial_raw;
                    [-v[0]/vol, -v[1]/vol, -v[2]/vol, -v[3]/vol, -v[4]/vol, -v[5]/vol]
                } else {
                    [0.0; 6]
                }
            }
            None => [0.0; 6],
        };

        Ok(ComputeResult {
            forces,
            energy,
            energy_per_atom,
            virial,
            virial_per_atom: virial_per_atom_raw,
            densities: if include_per_atom {
                densities.clone()
            } else {
                Vec::new()
            },
            embedding_energies: if include_per_atom {
                embed_energies.clone()
            } else {
                Vec::new()
            },
        })
    }
}

impl Drop for CpuEngine {
    fn drop(&mut self) {
        // Signal the keep-alive thread to stop.  It will exit within one
        // 50 ms sleep interval.  We do not join here to avoid blocking the
        // caller; the JoinHandle is dropped (thread detached) immediately
        // after, and the thread cleans up on its own.
        // No-op under Miri: fields are cfg-gated away, nothing to signal.
        #[cfg(not(miri))]
        self.stop_flag.store(true, Ordering::Relaxed);
    }
}

impl Default for CpuEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        potential::eam::synthetic_cu_alloy_src,
        reference::{compute_eam_cpu, ortho_cell},
    };

    fn make_pot() -> EamPotential {
        EamPotential::from_str(&synthetic_cu_alloy_src(100, 100, 0.05, 0.01, 4.5)).unwrap()
    }

    fn cu4_pos4() -> Vec<[f32; 4]> {
        let a = 3.615f32;
        vec![
            [0.0, 0.0, 0.0, 0.0],
            [a / 2.0, a / 2.0, 0.0, 0.0],
            [a / 2.0, 0.0, a / 2.0, 0.0],
            [0.0, a / 2.0, a / 2.0, 0.0],
        ]
    }

    /// Convert pos4 (f32 vec4) to the f64 vec3 form required by compute_eam_cpu.
    fn pos4_to_f64_3(pos4: &[[f32; 4]]) -> Vec<[f64; 3]> {
        pos4.iter()
            .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
            .collect()
    }

    fn cu4_types() -> Vec<u32> {
        vec![0u32; 4]
    }

    /// Orthorhombic f32 cell — used when calling CpuEngine (GPU-layout, f32).
    fn ortho_f32(l: f32) -> [[f32; 3]; 3] {
        [[l, 0.0, 0.0], [0.0, l, 0.0], [0.0, 0.0, l]]
    }

    #[test]
    fn cpu_engine_forces_finite() {
        let eng = CpuEngine::new();
        let pot = make_pot();
        let res = eng
            .compute_sync(&cu4_pos4(), &cu4_types(), Some(ortho_f32(10.0)), &pot)
            .unwrap();
        assert_eq!(res.forces.len(), 4);
        for (i, f) in res.forces.iter().enumerate() {
            for (c, &v) in f.iter().enumerate() {
                assert!(v.is_finite(), "forces[{i}][{c}]={v} non-finite");
            }
        }
        assert!(res.energy.is_finite());
    }

    /// CpuEngine (half-pair f32) and compute_eam_cpu (full-pair f64 reference) must
    /// agree to within f32 rounding. The half-pair version reorders additions,
    /// so exact equality is not guaranteed.
    #[test]
    fn cpu_engine_matches_reference() {
        let eng = CpuEngine::new();
        let pot = make_pot();

        // reference uses f64 cell and f64 positions
        let cell_f64 = Some(ortho_cell(10.0_f64, 10.0, 10.0));
        // engine uses f32 cell
        let cell_f32 = Some(ortho_f32(10.0));

        let pos_f64 = pos4_to_f64_3(&cu4_pos4());
        let cpu_ref = compute_eam_cpu(&pot, &pos_f64, &cu4_types(), cell_f64);
        let cpu_par = eng
            .compute_sync(&cu4_pos4(), &cu4_types(), cell_f32, &pot)
            .unwrap();

        for i in 0..4 {
            for c in 0..3 {
                let diff = (cpu_ref.forces[i][c] - cpu_par.forces[i][c] as f64).abs();
                assert!(
                    diff < 1e-4,
                    "forces[{i}][{c}] ref={:.6} par={:.6} diff={diff:.2e}",
                    cpu_ref.forces[i][c],
                    cpu_par.forces[i][c]
                );
            }
        }
        let e_diff = (cpu_ref.energy - cpu_par.energy as f64).abs();
        assert!(
            e_diff < 1e-4,
            "energy ref={:.6} par={:.6} diff={e_diff:.2e}",
            cpu_ref.energy,
            cpu_par.energy
        );
    }

    /// Newton's 3rd law: half-pair guarantees ΣFᵢ = 0 by construction,
    /// up to floating-point rounding.
    #[test]
    fn cpu_engine_newton_third_law() {
        let eng = CpuEngine::new();
        let pot = make_pot();
        let res = eng
            .compute_sync(&cu4_pos4(), &cu4_types(), None, &pot)
            .unwrap();
        let sum: [f32; 3] = res
            .forces
            .iter()
            .fold([0.0f32; 3], |a, f| [a[0] + f[0], a[1] + f[1], a[2] + f[2]]);
        for (c, &s) in sum.iter().enumerate() {
            assert!(s.abs() < 1e-3, "sum_F[{c}]={s:.2e} (Newton 3rd law)");
        }
    }

    #[test]
    fn cpu_engine_triclinic_pbc() {
        let eng = CpuEngine::new();
        let pot = make_pot();
        let cell = Some([[15.0f32, 0.0, 0.0], [7.5, 12.990381, 0.0], [0.0, 0.0, 30.0]]);
        let res = eng
            .compute_sync(&cu4_pos4(), &cu4_types(), cell, &pot)
            .unwrap();
        for (i, f) in res.forces.iter().enumerate() {
            for (c, &v) in f.iter().enumerate() {
                assert!(v.is_finite(), "forces[{i}][{c}]={v} non-finite (triclinic)");
            }
        }
    }

    #[test]
    fn cpu_engine_empty_input_errors() {
        let eng = CpuEngine::new();
        let pot = make_pot();
        let err = eng.compute_sync(&[], &[], None, &pot);
        assert!(err.is_err(), "empty input should return Err");
    }

    /// Energy per atom sum must match total energy.
    #[test]
    fn cpu_engine_energy_per_atom_sum() {
        let eng = CpuEngine::new();
        let pot = make_pot();
        let cell = Some(ortho_f32(10.0));
        let res = eng
            .compute_sync(&cu4_pos4(), &cu4_types(), cell, &pot)
            .unwrap();
        let sum: f32 = res.energy_per_atom.iter().sum();
        let diff = (sum - res.energy).abs();
        assert!(
            diff < 1e-5,
            "epa sum={sum:.6} energy={:.6} diff={diff:.2e}",
            res.energy
        );
    }

    /// Finite-difference validation: forces must agree with −dE/dr.
    #[test]
    fn cpu_engine_force_energy_fd_consistency() {
        let eng = CpuEngine::new();
        let src = synthetic_cu_alloy_src(500, 500, 0.01, 0.002, 4.5);
        let pot = EamPotential::from_str(&src).unwrap();
        // delta=1e-2 avoids f32 cancellation (positions ~1.5–3 Å, so rel delta ~3e-3 ≫ f32 eps)
        // tolerance loosened to 1.5e-2 to account for f32 interpolation discretisation.
        let delta = 1e-2f32;

        let pos4: Vec<[f32; 4]> = vec![
            [0.0, 0.0, 0.0, 0.0],
            [3.1, 0.0, 0.0, 0.0],
            [1.55, 2.68, 0.0, 0.0],
            [0.0, 1.6, 2.77, 0.0],
        ];
        let types = vec![0u32; 4];

        let r0 = eng.compute_sync(&pos4, &types, None, &pot).unwrap();

        for atom in 0..4 {
            for comp in 0..3 {
                let anal = r0.forces[atom][comp];
                if anal.abs() < 1e-4 {
                    continue;
                }

                let mut pp = pos4.clone();
                let mut pm = pos4.clone();
                pp[atom][comp] += delta;
                pm[atom][comp] -= delta;

                let ep = eng.compute_sync(&pp, &types, None, &pot).unwrap().energy;
                let em = eng.compute_sync(&pm, &types, None, &pot).unwrap().energy;
                let fd = -(ep - em) / (2.0 * delta);

                let rel = (anal - fd).abs() / anal.abs();
                assert!(
                    rel < 1.5e-2,
                    "atom {atom} axis {comp}: analytical={anal:.5e} fd={fd:.5e} rel={rel:.2e}"
                );
            }
        }
    }

    /// Half-pair must give same energy as full-pair reference for N=108.
    #[test]
    fn cpu_engine_n108_matches_reference() {
        let eng = CpuEngine::new();
        let pot = make_pot();
        let a = 3.615f32;
        let rep = 3usize;
        let basis: [[f32; 3]; 4] = [
            [0.0, 0.0, 0.0],
            [a / 2.0, a / 2.0, 0.0],
            [a / 2.0, 0.0, a / 2.0],
            [0.0, a / 2.0, a / 2.0],
        ];
        let mut pos4 = Vec::new();
        let mut pos_f64 = Vec::new();
        let mut types = Vec::new();
        for ix in 0..rep {
            for iy in 0..rep {
                for iz in 0..rep {
                    for b in &basis {
                        let p = [
                            b[0] + ix as f32 * a,
                            b[1] + iy as f32 * a,
                            b[2] + iz as f32 * a,
                        ];
                        pos4.push([p[0], p[1], p[2], 0.0f32]);
                        pos_f64.push([p[0] as f64, p[1] as f64, p[2] as f64]);
                        types.push(0u32);
                    }
                }
            }
        }
        let n = pos4.len();
        assert_eq!(n, 108);

        // reference requires f64 cell; engine requires f32 cell
        let l = a * rep as f32;
        let cell_f64 = Some(ortho_cell(l as f64, l as f64, l as f64));
        let cell_f32 = Some(ortho_f32(l));

        let ref_result = compute_eam_cpu(&pot, &pos_f64, &types, cell_f64);
        let hp_result = eng.compute_sync(&pos4, &types, cell_f32, &pot).unwrap();

        // Energy must agree within f32 rounding
        let e_diff = (ref_result.energy - hp_result.energy as f64).abs();
        assert!(
            e_diff < 1e-2,
            "N=108 energy: ref={:.6} hp={:.6} diff={e_diff:.2e}",
            ref_result.energy,
            hp_result.energy
        );

        // Forces must agree (compare via f64 to avoid loss)
        let mut max_f_diff = 0.0f64;
        for i in 0..n {
            for c in 0..3 {
                let d = (ref_result.forces[i][c] - hp_result.forces[i][c] as f64).abs();
                max_f_diff = max_f_diff.max(d);
            }
        }
        assert!(
            max_f_diff < 1e-3,
            "N=108 max force diff={max_f_diff:.2e} (limit 1e-3)"
        );

        // Newton's 3rd law (f32 engine output)
        let sum: [f32; 3] = hp_result
            .forces
            .iter()
            .fold([0.0f32; 3], |a, f| [a[0] + f[0], a[1] + f[1], a[2] + f[2]]);
        for (c, &s) in sum.iter().enumerate() {
            assert!(s.abs() < 1e-3, "N=108 sum_F[{c}]={s:.2e}");
        }
    }

    // ── Morton code unit tests ────────────────────────────────────────────────
    // Low-level `morton_encode`/`morton_decode_*` tests now live in
    // `src/cell_list.rs` (the canonical home of those helpers).  Physics-level
    // Morton-sort correctness remains tested here via
    // `morton_cell_list_matches_allpairs` / `morton_cell_list_newton_third_law`.

    /// Morton sort: cell list built with Morton ordering gives identical forces
    /// and energy to the reference all-pairs engine.  This verifies correctness
    /// of the new counting_sort_morton + for_each_neighbor Morton lookup.
    #[test]
    fn morton_cell_list_matches_allpairs() {
        let eng = CpuEngine::new();
        let pot = make_pot();
        let a = 3.615f32;
        let rep = 3usize; // 108 atoms
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
                            0.0f32,
                        ]);
                        types.push(0u32);
                    }
                }
            }
        }
        let l = a * rep as f32;
        let cell = Some(ortho_f32(l));

        let r_ap = eng.compute_sync(&pos4, &types, cell, &pot).unwrap();
        let r_cl = eng
            .compute_cell_list_sync(&pos4, &types, cell, &pot)
            .unwrap();

        let e_diff = (r_ap.energy - r_cl.energy).abs();
        assert!(
            e_diff < 1e-2,
            "Morton cell list energy: allpairs={:.6} celllist={:.6} diff={e_diff:.2e}",
            r_ap.energy,
            r_cl.energy
        );

        let mut max_f = 0.0f32;
        for i in 0..pos4.len() {
            for c in 0..3 {
                max_f = max_f.max((r_ap.forces[i][c] - r_cl.forces[i][c]).abs());
            }
        }
        assert!(
            max_f < 1e-3,
            "Morton cell list max force diff={max_f:.2e} (limit 1e-3)"
        );
    }

    /// Morton sort preserves Newton's 3rd law (sum of all forces vanishes).
    #[test]
    fn morton_cell_list_newton_third_law() {
        let eng = CpuEngine::new();
        let pot = make_pot();
        let a = 3.615f32;
        // rep=3 → l = 10.845 Å > 2×cutoff = 9 Å, satisfying minimum-image constraint.
        let rep = 3usize;
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
                            0.0f32,
                        ]);
                        types.push(0u32);
                    }
                }
            }
        }
        let l = a * rep as f32;
        let cell = Some(ortho_f32(l));
        let res = eng
            .compute_cell_list_sync(&pos4, &types, cell, &pot)
            .unwrap();
        let sum = res.forces.iter().fold([0.0f32; 3], |acc, f| {
            [acc[0] + f[0], acc[1] + f[1], acc[2] + f[2]]
        });
        for (c, &s) in sum.iter().enumerate() {
            assert!(s.abs() < 1e-3, "Morton N3L sum_F[{c}]={s:.2e}");
        }
    }

    // ── Cell list tests ───────────────────────────────────────────────────────

    /// Cell list energy must agree with O(N²) engine (same potential, same positions).
    #[test]
    fn cell_list_energy_agrees_with_allpairs() {
        let eng = CpuEngine::new();
        let pot = make_pot();
        let a = 3.615f32;
        let rep = 3usize; // 108 atoms
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
                            0.0f32,
                        ]);
                        types.push(0u32);
                    }
                }
            }
        }
        let l = a * rep as f32;
        let cell = Some(ortho_f32(l));
        let r_allpairs = eng.compute_sync(&pos4, &types, cell, &pot).unwrap();
        let r_cl = eng
            .compute_cell_list_sync(&pos4, &types, cell, &pot)
            .unwrap();

        let e_diff = (r_allpairs.energy - r_cl.energy).abs();
        assert!(
            e_diff < 1e-2,
            "energy: allpairs={:.6} celllist={:.6} diff={e_diff:.2e}",
            r_allpairs.energy,
            r_cl.energy
        );
    }

    /// Forces from cell list and all-pairs must agree to within f32 rounding.
    #[test]
    fn cell_list_forces_agree_with_allpairs() {
        let eng = CpuEngine::new();
        let pot = make_pot();
        let a = 3.615f32;
        let rep = 3usize;
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
                            0.0f32,
                        ]);
                        types.push(0u32);
                    }
                }
            }
        }
        let l = a * rep as f32;
        let cell = Some(ortho_f32(l));
        let r_ap = eng.compute_sync(&pos4, &types, cell, &pot).unwrap();
        let r_cl = eng
            .compute_cell_list_sync(&pos4, &types, cell, &pot)
            .unwrap();

        let mut max_diff = 0.0f32;
        for i in 0..pos4.len() {
            for c in 0..3 {
                max_diff = max_diff.max((r_ap.forces[i][c] - r_cl.forces[i][c]).abs());
            }
        }
        assert!(
            max_diff < 1e-3,
            "max force diff allpairs vs celllist = {max_diff:.2e} (limit 1e-3)"
        );
    }

    /// Cell list: Newton's third law (sum of all forces must vanish).
    #[test]
    fn cell_list_newton_third_law() {
        let eng = CpuEngine::new();
        let pot = make_pot();
        let pos4 = cu4_pos4();
        let res = eng
            .compute_cell_list_sync(&pos4, &cu4_types(), None, &pot)
            .unwrap();
        let sum = res
            .forces
            .iter()
            .fold([0.0f32; 3], |a, f| [a[0] + f[0], a[1] + f[1], a[2] + f[2]]);
        for (c, &s) in sum.iter().enumerate() {
            assert!(s.abs() < 1e-3, "cl sum_F[{c}]={s:.2e}");
        }
    }

    /// Cell list energy_per_atom must sum to total energy.
    #[test]
    fn cell_list_energy_per_atom_sum() {
        let eng = CpuEngine::new();
        let pot = make_pot();
        let a = 3.615f32;
        let rep = 3usize;
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
                            0.0f32,
                        ]);
                        types.push(0u32);
                    }
                }
            }
        }
        let cell = Some(ortho_f32(a * rep as f32));
        let res = eng
            .compute_cell_list_sync(&pos4, &types, cell, &pot)
            .unwrap();
        let sum: f32 = res.energy_per_atom.iter().sum();
        let diff = (sum - res.energy).abs();
        assert!(
            diff < 1e-4,
            "epa sum={sum:.6} energy={:.6} diff={diff:.2e}",
            res.energy
        );
    }

    /// Triclinic PBC: cell list forces must be finite (no fallback needed).
    #[test]
    fn cell_list_triclinic_pbc_finite() {
        let eng = CpuEngine::new();
        let pot = make_pot();
        let cell = Some([[15.0f32, 0.0, 0.0], [7.5, 12.990381, 0.0], [0.0, 0.0, 30.0]]);
        let res = eng.compute_cell_list_sync(&cu4_pos4(), &cu4_types(), cell, &pot);
        assert!(res.is_ok(), "triclinic cell list should succeed");
        for f in res.unwrap().forces.iter() {
            assert!(
                f.iter().all(|v| v.is_finite()),
                "non-finite force in triclinic cell list"
            );
        }
    }

    /// Triclinic PBC: cell list must agree with all-pairs to within f32 rounding.
    #[test]
    fn cell_list_triclinic_agrees_with_allpairs() {
        let eng = CpuEngine::new();
        // Use a larger rhombus cell so N=108 fits (cell heights > 2×cutoff).
        let src = synthetic_cu_alloy_src(100, 100, 0.05, 0.01, 4.5);
        let pot = EamPotential::from_str(&src).unwrap();
        let a = 3.615f32;
        let rep = 3usize;
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
                            0.0f32,
                        ]);
                        types.push(0u32);
                    }
                }
            }
        }
        // Slightly sheared triclinic cell — same volume as ortho, different shape.
        let l = a * rep as f32;
        let shear = l * 0.05;
        let cell = Some([[l, 0.0, 0.0], [shear, l, 0.0], [0.0, 0.0, l]]);

        let r_ap = eng.compute_sync(&pos4, &types, cell, &pot).unwrap();
        let r_cl = eng
            .compute_cell_list_sync(&pos4, &types, cell, &pot)
            .unwrap();

        let e_diff = (r_ap.energy - r_cl.energy).abs();
        assert!(
            e_diff < 1e-1,
            "triclinic energy: allpairs={:.4} celllist={:.4} diff={e_diff:.2e}",
            r_ap.energy,
            r_cl.energy
        );

        let mut max_f_diff = 0.0f32;
        for i in 0..pos4.len() {
            for c in 0..3 {
                max_f_diff = max_f_diff.max((r_ap.forces[i][c] - r_cl.forces[i][c]).abs());
            }
        }
        assert!(
            max_f_diff < 5e-3,
            "triclinic max force diff={max_f_diff:.2e} (limit 5e-3)"
        );
    }

    /// Regression test for the `bench_gpu` MISMATCH at `rep ≥ 35`.
    ///
    /// Before auto-p2 rounding was added to the CPU cell list, Cubic supercells
    /// with side length ~ 20 × lattice constant produced per-atom force errors
    /// of ~0.08 eV/Å on atoms whose fractional coordinates landed near cell
    /// boundaries under a borderline `cell_width ≈ cutoff + 0.1 Å` grid.
    /// This test exercises a non-trivially-sized cubic supercell with a
    /// deliberate rattle, and checks that CPU AllPairs vs CPU CellList agree
    /// to the same tight tolerance as small-N tests.
    #[test]
    fn cell_list_matches_allpairs_at_bench_mismatch_size() {
        let eng = CpuEngine::new();
        let pot = make_pot();
        let a = 3.615f32;
        // rep=8 keeps N=2048 (fast) but puts ncx on the boundary where the
        // pre-fix path had ncx=5 → cell_w=5.78 (OK) while bench's rep=35 had
        // ncx=25 → cell_w=5.06 (borderline).  The auto-p2 path now always
        // yields cell_w ≥ 1.5×cutoff, so this test becomes a structural
        // guard: force divergence here = auto-p2 accidentally disabled.
        let rep = 8usize;
        let basis: [[f32; 3]; 4] = [
            [0.0, 0.0, 0.0],
            [a / 2.0, a / 2.0, 0.0],
            [a / 2.0, 0.0, a / 2.0],
            [0.0, a / 2.0, a / 2.0],
        ];
        let mut pos4 = Vec::new();
        let mut types = Vec::new();
        let mut state: u32 = 0x1234_5678;
        let mut rattle = || -> f32 {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state as f32) / (u32::MAX as f32) - 0.5
        };
        for ix in 0..rep {
            for iy in 0..rep {
                for iz in 0..rep {
                    for b in &basis {
                        pos4.push([
                            b[0] + ix as f32 * a + 0.05 * rattle(),
                            b[1] + iy as f32 * a + 0.05 * rattle(),
                            b[2] + iz as f32 * a + 0.05 * rattle(),
                            0.0f32,
                        ]);
                        types.push(0u32);
                    }
                }
            }
        }
        let l = a * rep as f32;
        let cell = Some(ortho_f32(l));
        let r_ap = eng.compute_sync(&pos4, &types, cell, &pot).unwrap();
        let r_cl = eng
            .compute_cell_list_sync(&pos4, &types, cell, &pot)
            .unwrap();

        let mut max_diff = 0.0f32;
        let mut max_idx = 0usize;
        for i in 0..pos4.len() {
            for c in 0..3 {
                let d = (r_ap.forces[i][c] - r_cl.forces[i][c]).abs();
                if d > max_diff {
                    max_diff = d;
                    max_idx = i;
                }
            }
        }
        assert!(
            max_diff < 1e-3,
            "rep=8 rattled cubic: max|Δf|={max_diff:.3e} at atom {max_idx} (limit 1e-3)"
        );
    }
}
