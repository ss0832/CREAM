//! GPU per-pass workload profiling using the existing `dbg[*]` counters.
//!
//! This is a first-cut profiling tool that does NOT require modifying
//! `engine.rs`. It leverages the pair-visit / cutoff-hit / cell-walk
//! counters we already instrumented in `eam_pass{1,2}_cellist.wgsl`
//! (slots 17-24) to answer the key question: **for a given workload,
//! where is GPU time going?**
//!
//! What this tool measures
//! ───────────────────────
//! For each N, it:
//!   1.  Runs AllPairs (reference) + CellList (target) with wall-clock timing.
//!   2.  Reads back the `debug_flags[32]` counters for CellList.
//!   3.  Computes derived per-atom / per-WG metrics:
//!        - pair visits per atom          (avg "neighbour scan" cost)
//!        - cutoff hit ratio              (fraction of visits that survive)
//!        - real cells walked per WG      (spatial footprint of iteration)
//!        - pass1 vs pass2 ratio          (symmetry check)
//!   4.  Prints a wide diagnostic table.
//!
//! What this tool does NOT measure (yet)
//! ─────────────────────────────────────
//!   - True per-pass GPU time.  For that we'd need `TIMESTAMP_QUERY` which
//!     requires invasive changes to `run_passes()`.  This is planned as a
//!     follow-up if the counter data points at a specific pass as the
//!     bottleneck.
//!   - Pass0 (neighbour construction) timing.  Again, would need timestamps.
//!   - Memory bandwidth, atomic contention, WG occupancy.  Also timestamps.
//!
//! Usage
//! ─────
//!   $env:CREAM_ENABLE_DEBUG="1"         # required: turns on shader counters
//!   cargo run --release --features cellist_gpu --bin profile_gpu_v2
//!
//! Note on performance impact: `CREAM_ENABLE_DEBUG=1` makes every pair-visit
//! issue an atomicAdd to `dbg[17]`, which roughly doubles GPU time on
//! CellList.  The ABSOLUTE timings here are therefore NOT representative
//! of production speed — use `bench_gpu` for that.  The RATIOS between
//! passes / systems / sizes DO remain meaningful here because the overhead
//! is applied uniformly.

use cream::potential::eam::EamPotential;
use cream::{ComputeEngine, NeighborStrategy};
use std::time::Instant;

// ── Knobs ────────────────────────────────────────────────────────────────────

/// Base cutoff from the synthetic EAM potential.
const CUTOFF: f32 = 4.95;

/// cell_size / cutoff ratios to sweep.  Common production values:
///   0.5:  fine grid — more cells but fewer atoms/cell, stencil covers 5³
///         candidate cells instead of 3³ but each cell is tiny.
///   1.0:  minimal cells (each cell ≈ cutoff) — 3³ stencil; sometimes too
///         small in practice because single-atom cells lose coalescing.
///   1.1:  ASE/LAMMPS default-ish (small skin).
///   1.5:  moderate skin, used in some MD codes for Verlet list rebuild.
///   2.0:  large skin — cheaper neighbour rebuild, but the stencil covers
///         a (5×cell)³ region which is often worse than 1.0.
///
/// NOTE: when ratio < 1.0 the `cell_size` must still satisfy
/// `cell_width >= cutoff` in the engine; we handle that below.
const CELL_SIZE_RATIOS: &[f32] = &[1.0, 1.1, 1.5];

/// Warmup iters before measurement (pipeline / BG cache).
const WARMUP: usize = 3;
/// Iters for median wall-clock.
const MEASURE: usize = 8;

/// Atom counts to profile. Mix of power-of-2 and non-power-of-2 grid sizes
/// so we can compare fast-path vs fallback directly.
///
/// Grid ncx at cell_size=5.5 (see `engine_n_cells`):
///   rep=4   → N=256   → ncx=2  (p2)
///   rep=5   → N=500   → ncx=3  (non-p2, pad=4)
///   rep=6   → N=864   → ncx=3  (non-p2, pad=4)
///   rep=7   → N=1372  → ncx=4  (p2)
///   rep=8   → N=2048  → ncx=5  (non-p2, pad=8)
///   rep=10  → N=4000  → ncx=6  (non-p2, pad=8)
///   rep=12  → N=6912  → ncx=7  (non-p2, pad=8)
///   rep=14  → N=10976 → ncx=9  (non-p2, pad=16)
///   rep=16  → N=16384 → ncx=10 (non-p2, pad=16)
///   rep=20  → N=32000 → ncx=13 (non-p2, pad=16)
const REPS: &[usize] = &[4, 5, 6, 7, 8, 10, 12, 14, 16, 20];

// ── FCC supercell (same as bench_gpu.rs) ─────────────────────────────────────

fn fcc_supercell(rep: usize) -> (Vec<[f32; 4]>, Vec<u32>, [[f32; 3]; 3]) {
    let a = 3.615_f32;
    let basis: [[f32; 3]; 4] = [
        [0.0, 0.0, 0.0],
        [a / 2.0, a / 2.0, 0.0],
        [a / 2.0, 0.0, a / 2.0],
        [0.0, a / 2.0, a / 2.0],
    ];
    let n = rep * rep * rep * 4;
    let mut pos = Vec::with_capacity(n);
    let mut types = Vec::with_capacity(n);
    for ix in 0..rep {
        for iy in 0..rep {
            for iz in 0..rep {
                for b in &basis {
                    pos.push([
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
    let l = rep as f32 * a;
    let cell = [[l, 0.0, 0.0], [0.0, l, 0.0], [0.0, 0.0, l]];
    (pos, types, cell)
}

fn synth_cu_potential() -> EamPotential {
    use std::fmt::Write;
    let (nr, nrho) = (200u32, 200u32);
    let (dr, drho, cutoff) = (0.025_f32, 0.01_f32, 4.95_f32);
    let mut s = String::new();
    writeln!(s, "# Synthetic Cu diagnostic potential").unwrap();
    writeln!(s, "# diag").unwrap();
    writeln!(s, "# ok").unwrap();
    writeln!(s, "1 Cu").unwrap();
    writeln!(s, "{nrho} {drho} {nr} {dr} {cutoff}").unwrap();
    writeln!(s, "29 63.546 3.615 fcc").unwrap();
    for i in 0..nrho {
        let rho = i as f32 * drho;
        write!(s, "{:.8e} ", -(rho + 0.01_f32).sqrt()).unwrap();
        if (i + 1) % 5 == 0 {
            writeln!(s).unwrap();
        }
    }
    writeln!(s).unwrap();
    for i in 0..nr {
        let r = i as f32 * dr;
        let v = if r < cutoff {
            let t = cutoff - r;
            t * t / (cutoff * cutoff)
        } else {
            0.0
        };
        write!(s, "{v:.8e} ").unwrap();
        if (i + 1) % 5 == 0 {
            writeln!(s).unwrap();
        }
    }
    writeln!(s).unwrap();
    for i in 0..nr {
        let r = i as f32 * dr;
        let v = if r > 0.5 && r < cutoff {
            let t = cutoff - r;
            t * t * t / (r * cutoff * cutoff * cutoff)
        } else {
            0.0
        };
        write!(s, "{v:.8e} ").unwrap();
        if (i + 1) % 5 == 0 {
            writeln!(s).unwrap();
        }
    }
    writeln!(s).unwrap();
    EamPotential::from_str(&s).expect("synthetic potential parse failed")
}

// ── Workload metrics derivation ──────────────────────────────────────────────

/// Derived workload metrics per frame.
#[allow(dead_code)] // some fields are reserved for future analysis extensions
#[derive(Debug, Default, Clone)]
struct WorkloadMetrics {
    n_atoms: usize,
    cell_size: f32,
    grid: (u32, u32, u32),
    grid_pad: (u32, u32, u32),
    is_pow2: bool,
    n_wgs: usize,
    // pass1 counters
    p1_pair_visits: u64,
    p1_cutoff_hits: u64,
    p1_real_cells: u64,
    p1_pad_cells: u64,
    // pass2 counters
    p2_pair_visits: u64,
    p2_cutoff_hits: u64,
    p2_real_cells: u64,
    p2_pad_cells: u64,
    // wall-clock (Instant) in ms
    gpu_cl_ms: f64,
    gpu_ap_ms: f64,
    // ── 3-phase timing breakdown (CellList only) ────────────────────────
    // Measured from `Instant` boundaries inside `run_passes()`.  Populated
    // from the `dbg.timings` readback returned by `compute_with_debug()`.
    // Sum of the three typically ≈ `gpu_cl_ms` (slight overhead from the
    // extra readback in the debug path).
    /// Submission A: pass0a + pass0b + cell-counts readback copy.
    phase_a_ms: f64,
    /// CPU-side prefix sum + re-upload of cell_start / write_offsets.
    phase_cpu_ms: f64,
    /// Submission B: pass0c + pass0d + pass1 + pass2 + pass3.
    phase_b_ms: f64,
}

#[allow(dead_code)] // convenience accessors for future use
impl WorkloadMetrics {
    fn ap_equiv_pairs(&self) -> u64 {
        let n = self.n_atoms as u64;
        n * (n - 1)
    }
    fn visits_per_atom(&self) -> f64 {
        self.p1_pair_visits as f64 / self.n_atoms.max(1) as f64
    }
    fn pair_ratio(&self) -> f64 {
        self.p1_pair_visits as f64 / self.ap_equiv_pairs().max(1) as f64
    }
    fn cutoff_hit_ratio(&self) -> f64 {
        self.p1_cutoff_hits as f64 / self.p1_pair_visits.max(1) as f64
    }
    fn cells_per_wg(&self) -> f64 {
        (self.p1_real_cells + self.p1_pad_cells) as f64 / self.n_wgs.max(1) as f64
    }
    /// Theoretical best cells-per-WG if we had a perfect 27-cell stencil.
    /// For small grids, actual minimum is `min(27, ncx*ncy*ncz)`.
    fn cells_per_wg_ideal(&self) -> f64 {
        let total = (self.grid.0 * self.grid.1 * self.grid.2) as f64;
        27.0_f64.min(total)
    }
    /// How much extra cell-walk work we're doing vs ideal.
    fn cell_walk_overhead(&self) -> f64 {
        self.cells_per_wg() / self.cells_per_wg_ideal().max(1.0)
    }
    /// GPU-AP / GPU-CL speedup. > 1 means CL is faster.
    fn cl_speedup(&self) -> f64 {
        if self.gpu_cl_ms > 0.0 {
            self.gpu_ap_ms / self.gpu_cl_ms
        } else {
            f64::NAN
        }
    }
}

// ── Compute driver: run compute_with_debug once, extract counters ───────────

async fn profile_one(
    rep: usize,
    cell_size: f32,
    pot: &EamPotential,
) -> Result<WorkloadMetrics, Box<dyn std::error::Error>> {
    let (positions, types, cell) = fcc_supercell(rep);
    let n = positions.len();
    let n_wgs = n.div_ceil(64);

    // ── GPU CellList ─────────────────────────────────────────────────────
    let mut cl_engine = ComputeEngine::new(NeighborStrategy::CellList { cell_size }).await?;

    // Warmup
    for _ in 0..WARMUP {
        let _ = cl_engine
            .compute(&positions, &types, Some(cell), pot)
            .await?;
    }

    // Measure wall-clock median
    let mut cl_times = Vec::with_capacity(MEASURE);
    for _ in 0..MEASURE {
        let t0 = Instant::now();
        let _ = cl_engine
            .compute(&positions, &types, Some(cell), pot)
            .await?;
        cl_times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    cl_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let gpu_cl_ms = cl_times[cl_times.len() / 2];

    // Single debug readback (extracts dbg counters).  This resets counters
    // to zero at submission start, so the counter values reflect ONE frame.
    let (_, dbg_opt) = cl_engine
        .compute_with_debug(&positions, &types, Some(cell), pot)
        .await?;
    let dbg = dbg_opt.ok_or("CellList debug readback missing")?;

    // ── GPU AllPairs reference ───────────────────────────────────────────
    let mut ap_engine = ComputeEngine::new(NeighborStrategy::AllPairs).await?;
    for _ in 0..WARMUP {
        let _ = ap_engine
            .compute(&positions, &types, Some(cell), pot)
            .await?;
    }
    let mut ap_times = Vec::with_capacity(MEASURE);
    for _ in 0..MEASURE {
        let t0 = Instant::now();
        let _ = ap_engine
            .compute(&positions, &types, Some(cell), pot)
            .await?;
        ap_times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    ap_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let gpu_ap_ms = ap_times[ap_times.len() / 2];

    // ── Package ──────────────────────────────────────────────────────────
    let grid = dbg.n_cells;
    let grid_pad = dbg.n_cells_pad;
    let is_pow2 = grid == grid_pad;

    // Extract phase timings from `compute_with_debug`.
    // Populated by run_passes() when running the CellList path; `None` on
    // WASM or if the non-CellList path was somehow taken.
    let (phase_a_ms, phase_cpu_ms, phase_b_ms) = match dbg.timings {
        Some(t) => (t.submission_a_ms, t.cpu_prefix_ms, t.submission_b_ms),
        None => (0.0, 0.0, 0.0),
    };

    Ok(WorkloadMetrics {
        n_atoms: n,
        cell_size,
        grid,
        grid_pad,
        is_pow2,
        n_wgs,
        p1_pair_visits: dbg.debug_flags[17] as u64,
        p1_cutoff_hits: dbg.debug_flags[18] as u64,
        p1_real_cells: dbg.debug_flags[19] as u64,
        p1_pad_cells: dbg.debug_flags[20] as u64,
        p2_pair_visits: dbg.debug_flags[21] as u64,
        p2_cutoff_hits: dbg.debug_flags[22] as u64,
        p2_real_cells: dbg.debug_flags[23] as u64,
        p2_pad_cells: dbg.debug_flags[24] as u64,
        gpu_cl_ms,
        gpu_ap_ms,
        phase_a_ms,
        phase_cpu_ms,
        phase_b_ms,
    })
}

// ── Pretty-print ─────────────────────────────────────────────────────────────

fn print_header() {
    println!(
        "┌─────────┬────────┬──────────┬──────┬──────────┬──────────┬─────────┬───────────┬───────────┬──────────┬──────────┐"
    );
    println!(
        "│       N │   grid │  pad p2? │  WGs │  visits/ │  cutoff  │ cells/  │   ideal   │  overhead │  CL ms   │  AP ms   │"
    );
    println!(
        "│         │        │          │      │   atom   │   ratio  │   WG    │  cells/WG │   ratio   │ (median) │ (median) │"
    );
    println!(
        "├─────────┼────────┼──────────┼──────┼──────────┼──────────┼─────────┼───────────┼───────────┼──────────┼──────────┤"
    );
}

fn print_row(m: &WorkloadMetrics) {
    let grid_str = format!("{}×{}×{}", m.grid.0, m.grid.1, m.grid.2);
    let pad_str = format!("{}×{}×{}", m.grid_pad.0, m.grid_pad.1, m.grid_pad.2);
    let p2 = if m.is_pow2 { "p2" } else { "• " };

    println!(
        "│ {:7} │ {:>6} │ {:>6} {} │ {:>4} │ {:>8.1} │  {:>6.4} │ {:>7.2} │ {:>9.1} │ {:>9.2}× │ {:>8.3} │ {:>8.3} │",
        m.n_atoms,
        grid_str,
        pad_str,
        p2,
        m.n_wgs,
        m.visits_per_atom(),
        m.cutoff_hit_ratio(),
        m.cells_per_wg(),
        m.cells_per_wg_ideal(),
        m.cell_walk_overhead(),
        m.gpu_cl_ms,
        m.gpu_ap_ms,
    );
}

fn print_footer() {
    println!(
        "└─────────┴────────┴──────────┴──────┴──────────┴──────────┴─────────┴───────────┴───────────┴──────────┴──────────┘"
    );
}

fn print_legend() {
    println!();
    println!("Column legend:");
    println!("  grid         — actual real-grid cell counts (ncx × ncy × ncz)");
    println!("  pad p2?      — padded Morton grid; 'p2' if grid == pad (fast path eligible)");
    println!("  WGs          — number of 64-thread workgroups dispatched");
    println!("  visits/atom  — avg pairs scanned per atom in pass1 (before cutoff test)");
    println!("                 CL ≈ AP: CL walks every atom (no locality benefit)");
    println!("                 CL ≪ AP: stencil is pruning work effectively");
    println!("  cutoff ratio — fraction of scanned pairs that pass r < cutoff");
    println!("                 (low = dense system / big stencil; high = sparse / tight stencil)");
    println!("  cells/WG     — avg cells each WG walks (spatial iteration footprint)");
    println!("  ideal cells  — min(27, ncx*ncy*ncz): what a perfect stencil would walk");
    println!("  overhead     — cells/WG ÷ ideal; 1.0 = optimal, >1 = wasted iteration");
    println!();
    println!("Interpreting the results:");
    println!("  • If `visits/atom` scales like N → CL has no stencil benefit (need finer cells).");
    println!("  • If `overhead > 1.5×` for a size where CL is slower than CPU → stencil loop");
    println!("    is dominating; root cause is cell-walk inefficiency on non-p2 grids.");
    println!("  • If `cutoff ratio` is very low (<0.1) with high visits/atom → many wasted");
    println!("    pair-distance checks; stencil is too wide OR cell size is too large.");
    println!();
    println!("Timing caveat:");
    println!("  `CREAM_ENABLE_DEBUG=1` adds one `atomicAdd(&dbg[17], 1u)` per pair visit and");
    println!("  one per cutoff hit.  Production timings (no debug flag) are typically 2-3×");
    println!("  faster than the `CL ms` column.  The RATIOS and overhead metrics remain valid.");
}

// ── Summary analysis ─────────────────────────────────────────────────────────

fn print_analysis(all: &[WorkloadMetrics]) {
    println!();
    println!("══ Scaling analysis ══");

    // Check if visits/atom scales linearly with N (bad sign — means no stencil pruning).
    if all.len() >= 2 {
        let first = &all[0];
        let last = &all[all.len() - 1];
        let n_ratio = last.n_atoms as f64 / first.n_atoms as f64;
        let v_ratio = last.visits_per_atom() / first.visits_per_atom().max(1.0);
        let scaling_exponent = v_ratio.ln() / n_ratio.ln();
        println!(
            "  visits/atom scaling: {:.1}× over {:.1}× size  →  visits ∝ N^{:.2}",
            v_ratio, n_ratio, scaling_exponent
        );
        if scaling_exponent > 0.7 {
            println!("    ⚠  > 0.7: CellList is behaving ≈ O(N²); stencil is not pruning work.");
        } else if scaling_exponent < 0.2 {
            println!("    ✓  < 0.2: stencil pruning works — visits/atom ≈ constant (ideal).");
        } else {
            println!("    ~  partial pruning; stencil walks more than 27 cells on non-p2 grids.");
        }
    }

    // Identify where CL beats AP and where it loses.
    println!();
    println!("  CL speedup vs AllPairs:");
    for m in all {
        let sp = m.cl_speedup();
        let tag = if sp >= 1.0 { "✓" } else { "•" };
        println!(
            "    N={:6}  grid={}×{}×{} {:3}  CL={:7.3}ms  AP={:7.3}ms  {}{:.2}× ",
            m.n_atoms,
            m.grid.0,
            m.grid.1,
            m.grid.2,
            if m.is_pow2 { "p2" } else { "np" },
            m.gpu_cl_ms,
            m.gpu_ap_ms,
            tag,
            sp
        );
    }

    // Overhead summary
    println!();
    println!("  Cell-walk overhead (cells/WG ÷ ideal):");
    for m in all {
        let ov = m.cell_walk_overhead();
        let tag = if ov <= 1.1 {
            "✓"
        } else if ov <= 2.0 {
            "~"
        } else {
            "⚠"
        };
        println!(
            "    N={:6}  grid={}×{}×{} {:3}  cells/WG={:6.2}  ideal={:5.1}  {} overhead={:.2}×",
            m.n_atoms,
            m.grid.0,
            m.grid.1,
            m.grid.2,
            if m.is_pow2 { "p2" } else { "np" },
            m.cells_per_wg(),
            m.cells_per_wg_ideal(),
            tag,
            ov
        );
    }

    // p2 vs non-p2 break-out
    let p2: Vec<&WorkloadMetrics> = all.iter().filter(|m| m.is_pow2).collect();
    let np: Vec<&WorkloadMetrics> = all.iter().filter(|m| !m.is_pow2).collect();
    if !p2.is_empty() && !np.is_empty() {
        let p2_avg_ov: f64 =
            p2.iter().map(|m| m.cell_walk_overhead()).sum::<f64>() / p2.len() as f64;
        let np_avg_ov: f64 =
            np.iter().map(|m| m.cell_walk_overhead()).sum::<f64>() / np.len() as f64;
        println!();
        println!("  Average cell-walk overhead:");
        println!(
            "    power-of-2 grids:     {:.2}×  ({} samples)",
            p2_avg_ov,
            p2.len()
        );
        println!(
            "    non-power-of-2 grids: {:.2}×  ({} samples)",
            np_avg_ov,
            np.len()
        );
        if np_avg_ov > p2_avg_ov + 0.3 {
            println!("    ⇒  non-p2 grids pay a significant overhead.  Fix candidates:");
            println!("       (a) Engine-side auto-p2 grid (adjust cell_size)");
            println!("       (b) Better stencil iteration for non-p2");
        }
    }

    // Pass1 vs pass2 work
    let avg_p12_ratio: f64 = all
        .iter()
        .map(|m| m.p2_pair_visits as f64 / m.p1_pair_visits.max(1) as f64)
        .sum::<f64>()
        / all.len() as f64;
    println!();
    println!(
        "  Pass1 vs pass2 pair-visit ratio: avg = {:.3}",
        avg_p12_ratio
    );
    if (avg_p12_ratio - 1.0).abs() < 0.01 {
        println!("    ✓  pass2 walks same pairs as pass1; any timing asymmetry is per-pair cost.");
    } else {
        println!(
            "    ⚠  {:.1}% mismatch; pass2 iteration differs from pass1.",
            (avg_p12_ratio - 1.0) * 100.0
        );
    }

    // ── Per-phase wall-clock breakdown ──────────────────────────────────
    // Only display when populated (all.iter().any(|m| m.phase_a_ms > 0.0)).
    if all.iter().any(|m| m.phase_a_ms > 0.0) {
        println!();
        println!("  Phase breakdown (single frame via compute_with_debug):");
        println!(
            "    {:>6}  {:>8}  {:>8}  {:>8}  {:>6}  {:>6}  {:>6}",
            "N", "phase A", "phase CPU", "phase B", "A%", "CPU%", "B%"
        );
        println!(
            "    {:─>6}  {:─>8}  {:─>8}  {:─>8}  {:─>6}  {:─>6}  {:─>6}",
            "", "", "", "", "", "", ""
        );
        for m in all {
            let total = m.phase_a_ms + m.phase_cpu_ms + m.phase_b_ms;
            if total <= 0.0 {
                continue;
            }
            println!(
                "    {:>6}  {:>6.2}ms  {:>6.2}ms  {:>6.2}ms  {:>5.1}%  {:>5.1}%  {:>5.1}%",
                m.n_atoms,
                m.phase_a_ms,
                m.phase_cpu_ms,
                m.phase_b_ms,
                100.0 * m.phase_a_ms / total,
                100.0 * m.phase_cpu_ms / total,
                100.0 * m.phase_b_ms / total,
            );
        }
        // Where does time actually go? — compute averages
        let a_sum: f64 = all.iter().map(|m| m.phase_a_ms).sum();
        let c_sum: f64 = all.iter().map(|m| m.phase_cpu_ms).sum();
        let b_sum: f64 = all.iter().map(|m| m.phase_b_ms).sum();
        let total = a_sum + c_sum + b_sum;
        if total > 0.0 {
            println!();
            println!(
                "    totals:  A = {:.2}ms ({:.1}%), CPU = {:.2}ms ({:.1}%), B = {:.2}ms ({:.1}%)",
                a_sum,
                100.0 * a_sum / total,
                c_sum,
                100.0 * c_sum / total,
                b_sum,
                100.0 * b_sum / total,
            );
            let dom = if b_sum > a_sum + c_sum {
                "GPU density/force (pass1/pass2/pass3)"
            } else if a_sum > c_sum {
                "GPU neighbour build (pass0a/pass0b)"
            } else {
                "CPU prefix-sum round-trip"
            };
            println!("    ⇒ dominant phase across all sizes: {}", dom);
            println!(
                "      {}",
                if b_sum > (a_sum + c_sum) * 2.0 {
                    "(phase B >> A+CPU: shader compute is the bottleneck — \
                     optimise pass1/pass2 next)"
                } else if c_sum > (a_sum + b_sum) * 0.3 {
                    "(CPU prefix-sum is noticeable: consider GPU-native scan)"
                } else {
                    "(balanced; no single clear target)"
                }
            );
        }
    }
}

// ── main ────────────────────────────────────────────────────────────────────

fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Warn)
        .init();

    // Auto-enable shader debug counters.  Production timings should be
    // obtained from bench_gpu (no debug flag); this tool focuses on
    // WORKLOAD characterisation, not peak performance.
    std::env::set_var("CREAM_ENABLE_DEBUG", "1");

    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║   profile_gpu_v2 · CellList workload characterisation via dbg[*] counters    ║");
    println!("║                                                                              ║");
    println!("║   This tool identifies WHERE GPU CellList work is going (pair-scan cost,     ║");
    println!("║   cell-walk overhead, cutoff efficiency) and compares it against AllPairs    ║");
    println!("║   as a ceiling.  For true per-pass GPU time we'd need TIMESTAMP_QUERY, but   ║");
    println!("║   these metrics already localise most bottlenecks.                           ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!(
        "Config:  cutoff = {} Å   cell_size ratios swept = {:?}",
        CUTOFF, CELL_SIZE_RATIOS
    );
    println!(
        "         warmup = {} iters   measure = {} iters (median)",
        WARMUP, MEASURE
    );
    println!("         CREAM_ENABLE_DEBUG = 1  (dbg[*] counters enabled, ~2× slowdown)");
    println!();

    let pot = synth_cu_potential();

    // Outer loop: cell_size/cutoff ratio.  Inner loop: N.
    let mut all_by_ratio: Vec<(f32, Vec<WorkloadMetrics>)> =
        Vec::with_capacity(CELL_SIZE_RATIOS.len());
    for &ratio in CELL_SIZE_RATIOS {
        let cs = CUTOFF * ratio;
        println!("══════════════════════════════════════════════════════════════════════════════");
        println!(
            "  cell_size = {:.3} Å   (cell_size / cutoff = {:.2})",
            cs, ratio
        );
        println!("══════════════════════════════════════════════════════════════════════════════");

        let mut results = Vec::with_capacity(REPS.len());
        print_header();
        for &rep in REPS {
            match pollster::block_on(profile_one(rep, cs, &pot)) {
                Ok(m) => {
                    print_row(&m);
                    results.push(m);
                }
                Err(e) => {
                    let n = rep * rep * rep * 4;
                    eprintln!("N={} profiling failed: {}", n, e);
                }
            }
        }
        print_footer();
        print_analysis(&results);
        println!();
        all_by_ratio.push((ratio, results));
    }

    // Legend once at the end (no need to repeat per ratio).
    print_legend();

    // Cross-ratio summary
    println!("══════════════════════════════════════════════════════════════════════════════");
    println!("  Cross-ratio comparison (at each N, which cell_size was fastest?)");
    println!("══════════════════════════════════════════════════════════════════════════════");
    if !all_by_ratio.is_empty() {
        let n_reps = all_by_ratio[0].1.len();
        for row_i in 0..n_reps {
            let n_atoms = all_by_ratio[0].1.get(row_i).map(|m| m.n_atoms).unwrap_or(0);
            print!("  N = {:6}  ", n_atoms);
            let mut best_ms = f64::INFINITY;
            let mut best_ratio = 0.0_f32;
            for (ratio, results) in &all_by_ratio {
                if let Some(m) = results.get(row_i) {
                    let mark = if m.is_pow2 { "p2" } else { "np" };
                    print!("  cs/rc={:.2} {}: {:6.2} ms   ", ratio, mark, m.gpu_cl_ms);
                    if m.gpu_cl_ms < best_ms {
                        best_ms = m.gpu_cl_ms;
                        best_ratio = *ratio;
                    }
                }
            }
            println!("→ best = {:.2}", best_ratio);
        }
    }

    println!();
    println!("── Done ──────────────────────────────────────────────────────────────────────");
}
