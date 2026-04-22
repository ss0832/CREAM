//! GPU benchmark: AllPairs vs CellList vs CPU, warm-up + steady-state timing.
//!
//! Measures per-`compute()` wall time for N = 4 … 4000 atoms (FCC Cu supercells).
//! Reports three modes side-by-side:
//!
//!   * GPU AllPairs   — `ComputeEngine` with `NeighborStrategy::AllPairs`
//!   * GPU CellList   — `ComputeEngine` with `NeighborStrategy::CellList`
//!   * CPU CellList   — `CpuEngine::compute_cell_list_sync` (rayon parallel)
//!
//! Each row shows:
//!   1st-call  — pipeline compile + BindGroup creation (once per engine lifetime)
//!   steady    — median over `MEASURE_ITERS` calls with fully-cached BGs
//!
//! Environment:
//!   Software renderer (no physical GPU):
//!     XDG_RUNTIME_DIR=/tmp WGPU_BACKEND=vulkan \
//!       cargo run --bin bench_gpu --release 2>/dev/null
//!
//!   Physical GPU (Vulkan / Metal):
//!     cargo run --bin bench_gpu --release 2>/dev/null

use cream::potential::eam::EamPotential;
use cream::{ComputeEngine, CpuEngine, NeighborStrategy};
use std::time::{Duration, Instant};

// ── Knobs ─────────────────────────────────────────────────────────────────────

/// Cell size [Å] for both GPU CellList and CPU CellList.
/// Must exceed the potential cutoff (4.95 Å here).
const CELL_SIZE: f32 = 5.5;

/// Warm-up iterations before measurement (main table).
const WARM_ITERS: usize = 5;

/// Iterations used to compute the "steady" median in the main table.
const MEASURE_ITERS: usize = 12;

/// Iterations for the per-frame percentile breakdown section.
const PERCENTILE_ITERS: usize = 200;

/// Warm-up iterations for the percentile breakdown.
const PERCENTILE_WARM: usize = 30;

/// AllPairs is O(N²); skip above this threshold to avoid triggering the
/// Windows TDR watchdog (default 2-second GPU timeout).
const AP_MAX_ATOMS: usize = 50_000;

/// Tolerance for CPU vs GPU force equivalence checks.
const FORCE_TOLERANCE: f32 = 1e-5;

/// Tolerance for max |F_gpu − F_cpu| in the crystal scaling benchmark.
const CRYSTAL_TOLERANCE: f32 = 1e-3;

/// Tolerance for relative energy error in the crystal scaling benchmark:
///   |E_gpu − E_cpu| / |E_cpu|.max(1e-10)
/// GPU uses Neumaier/f64 accumulation; CPU uses plain f32 sum, so a small
/// asymmetry is expected at larger N — 1e-4 gives comfortable headroom.
const CRYSTAL_ENERGY_REL_TOL: f32 = 1e-4;

/// Upper bound on atom count for the crystal scaling section.
const CRYSTAL_MAX_ATOMS: usize = 5_000_000;

// ── FCC supercell builder ─────────────────────────────────────────────────────

/// Build an orthorhombic FCC supercell with `rep³ × 4` atoms.
///
/// Lattice constant a = 3.615 Å (Cu).
/// Returns `(positions [f32;4], atom_types, orthorhombic_cell)`.
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
    let l = a * rep as f32;
    let cell = [[l, 0.0, 0.0], [0.0, l, 0.0], [0.0, 0.0, l]];
    (pos, types, cell)
}

// ── Synthetic Cu potential ────────────────────────────────────────────────────

/// Build a lightweight synthetic Cu EAM potential (no `.alloy` file required).
fn synth_pot() -> EamPotential {
    use std::fmt::Write;
    let (nr, nrho) = (200u32, 200u32);
    let (dr, drho, cutoff) = (0.025_f32, 0.01_f32, 4.95_f32);
    let mut s = String::new();
    writeln!(s, "# Synthetic Cu benchmark potential").unwrap();
    writeln!(s, "# bench").unwrap();
    writeln!(s, "# ok").unwrap();
    writeln!(s, "1 Cu").unwrap();
    writeln!(s, "{nrho} {drho} {nr} {dr} {cutoff}").unwrap();
    writeln!(s, "29 63.546 3.615 fcc").unwrap();
    // F(rho) — embedding function
    for i in 0..nrho {
        let rho = i as f32 * drho;
        write!(s, "{:.8e} ", -(rho + 0.01_f32).sqrt()).unwrap();
        if (i + 1) % 5 == 0 {
            writeln!(s).unwrap();
        }
    }
    writeln!(s).unwrap();
    // f(r) — electron density
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
    // phi(r) — pair potential
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

// ── Validation Logic ──────────────────────────────────────────────────────────

/// CPU and GPU execution results verification. Panics if diff exceeds FORCE_TOLERANCE.
async fn validate_cpu_gpu_equivalence(rep: usize, pot: &EamPotential) {
    let (pos, types, cell) = fcc_supercell(rep);
    let n = pos.len();

    let cpu = CpuEngine::new();
    let cpu_res = cpu
        .compute_cell_list_sync(&pos, &types, Some(cell), pot)
        .expect("CPU compute failed during validation");

    let mut gpu = ComputeEngine::new(NeighborStrategy::CellList {
        cell_size: CELL_SIZE,
    })
    .await
    .expect("GPU engine init failed during validation");

    let gpu_res = gpu
        .compute(&pos, &types, Some(cell), pot)
        .await
        .expect("GPU compute failed during validation");

    for i in 0..n {
        for c in 0..3 {
            let diff = (cpu_res.forces[i][c] - gpu_res.forces[i][c]).abs();
            if diff > FORCE_TOLERANCE {
                panic!(
                    "VALIDATION FAILED: Atom {}, Component {} | CPU: {:.8e}, GPU: {:.8e}, Diff: {:.8e} (Limit: {:.8e})",
                    i, c, cpu_res.forces[i][c], gpu_res.forces[i][c], diff, FORCE_TOLERANCE
                );
            }
        }
    }
    println!(
        "  [OK] N={} Equivalence Verified (Max diff < {:.0e})",
        n, FORCE_TOLERANCE
    );
}

// ── Formatting ────────────────────────────────────────────────────────────────

fn fmt_dur(d: Duration) -> String {
    let us = d.as_secs_f64() * 1e6;
    if us < 1_000.0 {
        format!("{us:7.1} µs")
    } else {
        format!("{:7.2} ms", us / 1_000.0)
    }
}

// ── GPU benchmark for one strategy ───────────────────────────────────────────

/// Returns `(first_call, median_steady)`.
fn bench_gpu_one(
    engine: &mut ComputeEngine,
    pot: &EamPotential,
    pos: &[[f32; 4]],
    types: &[u32],
    cell: [[f32; 3]; 3],
) -> (Duration, Duration) {
    // 1st call: pipeline compile + BindGroup creation
    let t0 = Instant::now();
    engine
        .compute_sync(pos, types, Some(cell), pot)
        .expect("GPU compute failed");
    let first = t0.elapsed();

    for _ in 0..WARM_ITERS {
        engine
            .compute_sync(pos, types, Some(cell), pot)
            .expect("GPU compute failed");
    }

    let mut samples: Vec<Duration> = (0..MEASURE_ITERS)
        .map(|_| {
            let t = Instant::now();
            engine
                .compute_sync(pos, types, Some(cell), pot)
                .expect("GPU compute failed");
            t.elapsed()
        })
        .collect();
    samples.sort_unstable();
    (first, samples[samples.len() / 2])
}

// ── CPU CellList benchmark ────────────────────────────────────────────────────

/// Returns median steady-state duration, or `None` when the cell is too small
/// for the minimum-image convention (box side < 2 × cutoff).
fn bench_cpu_cl(
    cpu: &CpuEngine,
    pot: &EamPotential,
    pos: &[[f32; 4]],
    types: &[u32],
    cell: [[f32; 3]; 3],
) -> Option<Duration> {
    // Probe: a single call reveals whether the geometry is valid.
    cpu.compute_cell_list_sync(pos, types, Some(cell), pot)
        .ok()?;

    for _ in 0..WARM_ITERS {
        cpu.compute_cell_list_sync(pos, types, Some(cell), pot)
            .expect("CPU cell-list compute failed");
    }
    let mut samples: Vec<Duration> = (0..MEASURE_ITERS)
        .map(|_| {
            let t = Instant::now();
            cpu.compute_cell_list_sync(pos, types, Some(cell), pot)
                .expect("CPU cell-list compute failed");
            t.elapsed()
        })
        .collect();
    samples.sort_unstable();
    Some(samples[samples.len() / 2])
}

// ── Percentile row (GPU) ──────────────────────────────────────────────────────

fn print_percentile_row(
    label: &str,
    engine: &mut ComputeEngine,
    pot: &EamPotential,
    pos: &[[f32; 4]],
    types: &[u32],
    cell: [[f32; 3]; 3],
) {
    let t_first = {
        let t = Instant::now();
        engine
            .compute_sync(pos, types, Some(cell), pot)
            .expect("GPU compute failed");
        t.elapsed()
    };
    for _ in 0..PERCENTILE_WARM {
        engine
            .compute_sync(pos, types, Some(cell), pot)
            .expect("GPU compute failed");
    }
    let mut samples: Vec<Duration> = (0..PERCENTILE_ITERS)
        .map(|_| {
            let t = Instant::now();
            engine
                .compute_sync(pos, types, Some(cell), pot)
                .expect("GPU compute failed");
            t.elapsed()
        })
        .collect();

    let mean = samples.iter().sum::<Duration>() / samples.len() as u32;
    let variance = samples
        .iter()
        .map(|d| {
            let diff = d.as_secs_f64() - mean.as_secs_f64();
            diff * diff
        })
        .sum::<f64>()
        / samples.len() as f64;
    let stddev = Duration::from_secs_f64(variance.sqrt());

    samples.sort_unstable();
    let p50 = samples[samples.len() / 2];
    let p95 = samples[(samples.len() as f64 * 0.95) as usize];
    let p99 = samples[(samples.len() as f64 * 0.99) as usize];

    println!(
        "  {label:<10}  1st={first}  mean={mean}  σ={sd}  p50={p50}  p95={p95}  p99={p99}",
        first = fmt_dur(t_first),
        mean = fmt_dur(mean),
        sd = fmt_dur(stddev),
        p50 = fmt_dur(p50),
        p95 = fmt_dur(p95),
        p99 = fmt_dur(p99),
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// ── Crystal System Scaling Benchmark ─────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════════
//
// Mirrors the Python test_python_v2.py grid sweep:
//   • 7 crystal systems: Cubic, Tetragonal, Orthorhombic, Hexagonal,
//                        Rhombohedral, Monoclinic, Triclinic
//   • Supercell reps 4 … 14  →  N = 256 … 10 976 atoms
//   • Each FCC base cell is deformed by a 3×3 transformation matrix and
//     rattled (stdev=0.05 Å, seed=42) before compute.
//   • Reference: CPU CellList (steady-state median over MEASURE_ITERS)
//   • Target:    GPU CellList (same warm-up / measure protocol)
//   • Correctness: max |F_gpu − F_cpu| compared against CRYSTAL_TOLERANCE
//
// ── 3×3 matrix helpers ───────────────────────────────────────────────────────

/// Row-major 3×3 matrix multiply: C = A · B
fn mat3_mul(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut c = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

/// Apply 3×3 matrix `m` to a row-vector `v`:  result[j] = Σ_k v[k] · m[k][j]
fn apply_transform_to_pos(m: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        v[0] * m[0][0] + v[1] * m[1][0] + v[2] * m[2][0],
        v[0] * m[0][1] + v[1] * m[1][1] + v[2] * m[2][1],
        v[0] * m[0][2] + v[1] * m[1][2] + v[2] * m[2][2],
    ]
}

// ── Deterministic rattle (Box-Muller, LCG seed) ───────────────────────────────

/// LCG step — returns a float in [0, 1).
#[inline]
fn lcg_step(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 33) as f64) / (u64::MAX >> 33) as f64
}

/// Box-Muller: produce two standard-normal samples.
fn normal_pair(state: &mut u64) -> (f32, f32) {
    let u1 = lcg_step(state).max(1e-15);
    let u2 = lcg_step(state);
    let mag = (-2.0 * u1.ln()).sqrt() as f32;
    let angle = (2.0 * std::f64::consts::PI * u2) as f32;
    (mag * angle.cos(), mag * angle.sin())
}

/// Apply small Gaussian displacements in-place (stdev in Å).
/// Deterministic: seeded with `seed` (42 in the Python reference).
fn rattle(pos: &mut [[f32; 4]], stdev: f32, seed: u64) {
    let mut state = seed;
    for p in pos.iter_mut() {
        let (dx, dy) = normal_pair(&mut state);
        let (dz, _) = normal_pair(&mut state);
        p[0] += stdev * dx;
        p[1] += stdev * dy;
        p[2] += stdev * dz;
    }
}

// ── Crystal system definitions ────────────────────────────────────────────────

/// The seven Bravais crystal systems, parameterised as in test_python_v2.py.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CrystalSystem {
    Cubic,
    Tetragonal,
    Orthorhombic,
    Hexagonal,
    Rhombohedral,
    Monoclinic,
    Triclinic,
}

impl CrystalSystem {
    const ALL: &'static [Self] = &[
        Self::Cubic,
        Self::Tetragonal,
        Self::Orthorhombic,
        Self::Hexagonal,
        Self::Rhombohedral,
        Self::Monoclinic,
        Self::Triclinic,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::Cubic => "Cubic",
            Self::Tetragonal => "Tetragonal",
            Self::Orthorhombic => "Orthorhombic",
            Self::Hexagonal => "Hexagonal",
            Self::Rhombohedral => "Rhombohedral",
            Self::Monoclinic => "Monoclinic",
            Self::Triclinic => "Triclinic",
        }
    }

    /// Deformation matrix applied to the cubic FCC cell.
    ///
    /// Matches the NumPy arrays in `generate_crystal_system()` of the
    /// Python reference script.  The transformation is applied as:
    ///   new_cell = cell @ T   (ASE row-vector convention)
    ///   new_pos  = pos  @ T   (valid when base cell is cubic, i.e. cell = L·I)
    fn transform(self) -> [[f32; 3]; 3] {
        let s3 = (3.0f32).sqrt() / 2.0; // sin(60°) ≈ 0.866 025
        match self {
            Self::Cubic => [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            Self::Tetragonal => [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.2]],
            Self::Orthorhombic => [[1.0, 0.0, 0.0], [0.0, 1.1, 0.0], [0.0, 0.0, 1.2]],
            // Hexagonal: a₁=a, a₂=a(-½, √3/2, 0), c-axis stretched
            Self::Hexagonal => [[1.0, -0.5, 0.0], [0.0, s3, 0.0], [0.0, 0.0, 1.2]],
            Self::Rhombohedral => [[1.1, 0.1, 0.1], [0.1, 1.1, 0.1], [0.1, 0.1, 1.1]],
            Self::Monoclinic => [[1.0, 0.0, 0.2], [0.0, 1.1, 0.0], [0.0, 0.0, 1.2]],
            Self::Triclinic => [[1.00, 0.10, 0.20], [0.15, 1.10, 0.25], [0.05, 0.30, 1.20]],
        }
    }
}

// ── Crystal supercell builder ─────────────────────────────────────────────────

/// Build a deformed FCC supercell for the given crystal system.
///
/// 1. Start from an orthorhombic FCC supercell (rep³ × 4 atoms).
/// 2. Apply the system's deformation matrix (cell and atomic positions).
/// 3. Rattle positions (stdev=0.05 Å, seed=42) for realism.
///
/// Returns `(positions [f32;4], atom_types, lattice_cell [[f32;3];3])`.
fn crystal_supercell(
    system: CrystalSystem,
    rep: usize,
) -> (Vec<[f32; 4]>, Vec<u32>, [[f32; 3]; 3]) {
    let (mut pos, types, cell_cubic) = fcc_supercell(rep);
    let t = system.transform();

    // new_cell = cell_cubic @ T  (since cell_cubic = L·I, this is L·T)
    let new_cell = mat3_mul(cell_cubic, t);

    // new_pos[atom] = pos[atom] @ T
    for p in pos.iter_mut() {
        let xyz = apply_transform_to_pos(t, [p[0], p[1], p[2]]);
        p[0] = xyz[0];
        p[1] = xyz[1];
        p[2] = xyz[2];
    }

    // Deterministic rattle (stdev=0.05 Å, seed=42 matches Python reference)
    rattle(&mut pos, 0.05, 42);

    (pos, types, new_cell)
}

// ── Crystal scaling benchmark helpers ────────────────────────────────────────

/// Steady-state GPU-CL median + force/energy results for correctness comparison.
///
/// Returns `(first_call, median_steady, forces, energy)`.
/// Both `forces` and `energy` come from the first call (post-compile); the
/// computation is deterministic so these equal any subsequent steady-state
/// result for the same input.
fn crystal_gpu_cl(
    engine: &mut ComputeEngine,
    pot: &EamPotential,
    pos: &[[f32; 4]],
    types: &[u32],
    cell: [[f32; 3]; 3],
) -> (Duration, Duration, Vec<[f32; 3]>, f32) {
    // First call — pipeline compile
    let t0 = Instant::now();
    let first_res = engine
        .compute_sync(pos, types, Some(cell), pot)
        .expect("GPU CellList failed on first call");
    let first = t0.elapsed();

    // Warm-up
    for _ in 0..WARM_ITERS {
        engine
            .compute_sync(pos, types, Some(cell), pot)
            .expect("GPU CellList failed during warm-up");
    }

    // Measurement
    let mut samples: Vec<Duration> = (0..MEASURE_ITERS)
        .map(|_| {
            let t = Instant::now();
            engine
                .compute_sync(pos, types, Some(cell), pot)
                .expect("GPU CellList failed during measurement");
            t.elapsed()
        })
        .collect();
    samples.sort_unstable();

    (first, samples[samples.len() / 2], first_res.forces, first_res.energy)
}

// ── Main crystal benchmark driver ─────────────────────────────────────────────

/// Sweep all 7 crystal systems × supercell sizes (N ≤ CRYSTAL_MAX_ATOMS).
///
/// For each combination:
///   • Build a deformed+rattled FCC supercell
///   • Run CPU-CL as reference (validates geometry; measures steady-state time)
///   • Run GPU-CL (measures steady-state time; compares forces to CPU-CL)
///   • Print one table row
///
/// A final per-system summary is printed after the table.
fn bench_crystal_systems(pot: &EamPotential, cpu: &CpuEngine) {
    // rep³×4 atoms:  4→256  5→500  6→864  7→1372  8→2048  9→2916
    //               10→4000 11→5324 12→6912 13→8788 14→10976
    let reps: &[usize] = &[
        4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 25, 30, 35, 40, 50, 75, 100,
    ];

    let col_w = 110;
    let sep = "─".repeat(col_w);

    println!("\n{}", "═".repeat(col_w));
    println!(
        "  Crystal System Scaling Benchmark  \
         (7 systems × {} sizes, N ≤ {})",
        reps.len(),
        CRYSTAL_MAX_ATOMS
    );
    println!(
        "  Warm: {} iters | Measure: {} iters (median) | \
         Force tol: {:.0e} | Energy rel tol: {:.0e}",
        WARM_ITERS, MEASURE_ITERS, CRYSTAL_TOLERANCE, CRYSTAL_ENERGY_REL_TOL
    );
    println!("{}", "═".repeat(col_w));

    println!(
        "\n{:<14} {:>7}  {:>12}  {:>12}  {:>8}  {:>10}  {:>12}  {}",
        "System", "N", "CPU-CL med", "GPU-CL med", "Speedup", "MaxForceErr", "EnergyRelErr", "Status"
    );
    println!("{sep}");

    // Per-system summary data: (total_cases, passed, skipped)
    let mut summary: Vec<(&str, usize, usize, usize)> = Vec::new();

    for &system in CrystalSystem::ALL {
        let mut total = 0usize;
        let mut passed = 0usize;
        let mut skipped = 0usize;

        for &rep in reps {
            let n = rep * rep * rep * 4;
            if n > CRYSTAL_MAX_ATOMS {
                break;
            }
            total += 1;

            let (pos, types, cell) = crystal_supercell(system, rep);

            // ── CPU-CL reference ──────────────────────────────────────────
            // Captures forces AND energy for correctness comparison.
            let cpu_probe = cpu.compute_cell_list_sync(&pos, &types, Some(cell), pot);
            let (cpu_forces, cpu_energy) = match cpu_probe {
                Err(_) => {
                    // Box too small for minimum-image convention — skip this rep
                    println!(
                        "{:<14} {:>7}  {:>12}  {:>12}  {:>8}  {:>10}  {:>12}  SKIP (box too small)",
                        system.name(),
                        n,
                        "—", "—", "—", "—", "—"
                    );
                    skipped += 1;
                    continue;
                }
                Ok(r) => (r.forces, r.energy),
            };

            // Warm-up + steady median for CPU
            for _ in 0..WARM_ITERS {
                cpu.compute_cell_list_sync(&pos, &types, Some(cell), pot)
                    .expect("CPU cell-list failed during crystal warm-up");
            }
            let cpu_med = {
                let mut s: Vec<Duration> = (0..MEASURE_ITERS)
                    .map(|_| {
                        let t = Instant::now();
                        cpu.compute_cell_list_sync(&pos, &types, Some(cell), pot)
                            .expect("CPU cell-list failed during crystal measurement");
                        t.elapsed()
                    })
                    .collect();
                s.sort_unstable();
                s[s.len() / 2]
            };

            // ── GPU-CL target ─────────────────────────────────────────────
            let mut gpu_eng = pollster::block_on(ComputeEngine::new(NeighborStrategy::CellList {
                cell_size: CELL_SIZE,
            }))
            .expect("GPU CellList engine init failed");

            let (_gpu_first, gpu_med, gpu_forces, gpu_energy) =
                crystal_gpu_cl(&mut gpu_eng, pot, &pos, &types, cell);

            // ── Correctness check — forces ────────────────────────────────
            let max_err = cpu_forces
                .iter()
                .zip(gpu_forces.iter())
                .flat_map(|(cf, gf)| {
                    [
                        (cf[0] - gf[0]).abs(),
                        (cf[1] - gf[1]).abs(),
                        (cf[2] - gf[2]).abs(),
                    ]
                })
                .fold(0.0f32, f32::max);

            // ── Correctness check — energy ────────────────────────────────
            // Relative error: |ΔE| / |E_cpu|, guarded against zero denominator.
            // GPU uses Neumaier/f64; CPU uses plain f32 sequential sum, so a
            // small residual (~1–2 ULP × N) is expected for large N.
            let energy_rel_err =
                (gpu_energy - cpu_energy).abs() / cpu_energy.abs().max(1e-10_f32);

            let status = if max_err < CRYSTAL_TOLERANCE && energy_rel_err < CRYSTAL_ENERGY_REL_TOL
            {
                passed += 1;
                "PASS"
            } else {
                "MISMATCH"
            };

            let speedup = cpu_med.as_secs_f64() / gpu_med.as_secs_f64();

            println!(
                "{:<14} {:>7}  {:>12}  {:>12}  {:>7.2}x  {:>10.3e}  {:>12.3e}  {}",
                system.name(),
                n,
                fmt_dur(cpu_med),
                fmt_dur(gpu_med),
                speedup,
                max_err,
                energy_rel_err,
                status,
            );
        }

        println!("{sep}");
        summary.push((system.name(), total, passed, skipped));
    }

    // ── Per-system summary ────────────────────────────────────────────────────
    println!("\n{}", "═".repeat(col_w));
    println!("  Crystal Benchmark Summary");
    println!("{}", "═".repeat(col_w));
    println!(
        "{:<14}  {:>7}  {:>7}  {:>8}  {:>7}",
        "System", "Total", "PASS", "MISMATCH", "SKIP"
    );
    println!("{}", "─".repeat(55));

    let mut grand_total = 0usize;
    let mut grand_pass = 0usize;
    let mut grand_skip = 0usize;

    for (name, total, passed, skipped) in &summary {
        let mismatch = total - passed - skipped;
        println!(
            "{:<14}  {:>7}  {:>7}  {:>8}  {:>7}",
            name, total, passed, mismatch, skipped
        );
        grand_total += total;
        grand_pass += passed;
        grand_skip += skipped;
    }

    println!("{}", "─".repeat(55));
    let grand_mismatch = grand_total - grand_pass - grand_skip;
    println!(
        "{:<14}  {:>7}  {:>7}  {:>8}  {:>7}",
        "TOTAL", grand_total, grand_pass, grand_mismatch, grand_skip
    );
    println!("{}", "═".repeat(col_w));
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Warn)
        .init();

    let pot = synth_pot();
    let cpu = CpuEngine::new();

    // ── Verification Phase ────────────────────────────────────────────────────
    println!("── Verification (CPU vs GPU Equivalence) ──");
    pollster::block_on(async {
        validate_cpu_gpu_equivalence(5, &pot).await;
        validate_cpu_gpu_equivalence(20, &pot).await;
        validate_cpu_gpu_equivalence(50, &pot).await;
    });
    println!();

    // ── Crystal System Scaling Benchmark ──────────────────────────────────────
    bench_crystal_systems(&pot, &cpu);

    // rep³ × 4 atoms:
    //  1→4, 2→32, 3→108, 4→256, 5→500, 6→864,
    //  7→1372, 8→2048, 9→2916, 10→4000
    let reps: &[usize] = &[
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 75, 100, 110, 120, 130, 140, 150, 160,
        170, 180,
    ];

    // ── Main comparison table ─────────────────────────────────────────────────
    let sep = "─".repeat(110);
    println!(
        "\n{:<6}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}  {:>10}  {:>10}",
        "N",
        "GPU-AP 1st",
        "GPU-AP med",
        "GPU-CL 1st",
        "GPU-CL med",
        "CPU-CL med",
        "CL/AP",
        "CL/CPU",
    );
    println!(
        "{:<6}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}  {:>10}  {:>10}",
        "", "(1st call)", "(steady)", "(1st call)", "(steady)", "(steady)", "speedup", "speedup",
    );
    println!("{sep}");

    for &rep in reps {
        let (pos, types, cell) = fcc_supercell(rep);
        let n = pos.len();

        // GPU AllPairs — skipped above AP_MAX_ATOMS to avoid a Windows TDR reset.
        let ap_opt: Option<(Duration, Duration)> = if n <= AP_MAX_ATOMS {
            let mut eng = pollster::block_on(ComputeEngine::new(NeighborStrategy::AllPairs))
                .expect("AllPairs engine init failed");
            Some(bench_gpu_one(&mut eng, &pot, &pos, &types, cell))
        } else {
            None
        };

        // GPU CellList
        let mut eng_cl = pollster::block_on(ComputeEngine::new(NeighborStrategy::CellList {
            cell_size: CELL_SIZE,
        }))
        .expect("CellList engine init failed");
        let (cl_first, cl_med) = bench_gpu_one(&mut eng_cl, &pot, &pos, &types, cell);

        // CPU CellList (rayon) — returns None when box < 2×cutoff
        let cpu_opt = bench_cpu_cl(&cpu, &pot, &pos, &types, cell);

        let (ap_first_str, ap_med_str, cl_vs_ap_str) = match ap_opt {
            Some((f, m)) => (
                fmt_dur(f),
                fmt_dur(m),
                format!("{:>9.2}x", m.as_secs_f64() / cl_med.as_secs_f64()),
            ),
            None => ("N/A".to_owned(), "N/A".to_owned(), format!("{:>10}", "N/A")),
        };
        let cl_vs_cpu_str = match cpu_opt {
            Some(cpu_med) => format!("{:>9.2}x", cpu_med.as_secs_f64() / cl_med.as_secs_f64()),
            None => format!("{:>10}", "N/A"),
        };
        let cpu_med_str = match cpu_opt {
            Some(d) => fmt_dur(d),
            None => format!("{:>7}", "N/A"),
        };

        println!(
            "{n:<6}  {ap1:>12}  {aps:>12}  {cl1:>12}  {cls:>12}  {cpus:>12}  {sp1}  {sp2:>10}",
            ap1 = ap_first_str,
            aps = ap_med_str,
            cl1 = fmt_dur(cl_first),
            cls = fmt_dur(cl_med),
            cpus = cpu_med_str,
            sp1 = cl_vs_ap_str,
            sp2 = cl_vs_cpu_str,
        );
    }

    // ── Per-frame latency distribution at N=2048 ──────────────────────────────
    println!(
        "\n── Per-frame latency distribution at N=2048  ({PERCENTILE_ITERS} samples after {PERCENTILE_WARM} warm-up) ──"
    );
    let (pos_bd, types_bd, cell_bd) = fcc_supercell(8); // 2048 atoms
    assert_eq!(pos_bd.len(), 2048);

    let mut eng_ap_bd = pollster::block_on(ComputeEngine::new(NeighborStrategy::AllPairs))
        .expect("AllPairs engine init failed");
    print_percentile_row("GPU-AP", &mut eng_ap_bd, &pot, &pos_bd, &types_bd, cell_bd);

    let mut eng_cl_bd = pollster::block_on(ComputeEngine::new(NeighborStrategy::CellList {
        cell_size: CELL_SIZE,
    }))
    .expect("CellList engine init failed");
    print_percentile_row("GPU-CL", &mut eng_cl_bd, &pot, &pos_bd, &types_bd, cell_bd);

    // ── BindGroup cache: stable-N vs N-change overhead ────────────────────────
    println!("\n── BindGroup cache: stable N=2048 vs N-change 2048↔256 (AllPairs) ─────────");
    {
        let (pos_256, types_256, cell_256) = fcc_supercell(4); // 256 atoms
        let mut eng = pollster::block_on(ComputeEngine::new(NeighborStrategy::AllPairs))
            .expect("engine init failed");

        // warm up on 2048
        for _ in 0..8 {
            eng.compute_sync(&pos_bd, &types_bd, Some(cell_bd), &pot)
                .unwrap();
        }

        // stable N=2048 — median of 30 calls
        let mut stable_samples: Vec<Duration> = (0..30)
            .map(|_| {
                let t = Instant::now();
                eng.compute_sync(&pos_bd, &types_bd, Some(cell_bd), &pot)
                    .unwrap();
                t.elapsed()
            })
            .collect();
        stable_samples.sort_unstable();
        let stable_med = stable_samples[stable_samples.len() / 2];

        // N-change: 2048 → 256 → 2048 (buffer realloc + all BindGroups rebuilt)
        let mut change_samples: Vec<Duration> = (0..15)
            .flat_map(|_| {
                let t1 = Instant::now();
                eng.compute_sync(&pos_256, &types_256, Some(cell_256), &pot)
                    .unwrap();
                let d1 = t1.elapsed();
                let t2 = Instant::now();
                eng.compute_sync(&pos_bd, &types_bd, Some(cell_bd), &pot)
                    .unwrap();
                let d2 = t2.elapsed();
                [d1, d2]
            })
            .collect();
        change_samples.sort_unstable();
        let change_med = change_samples[change_samples.len() / 2];

        println!(
            "  stable  N=2048 median: {stable}  |  N-change 2048↔256 median: {change}  (overhead: {overhead:.1}x)",
            stable   = fmt_dur(stable_med),
            change   = fmt_dur(change_med),
            overhead = change_med.as_secs_f64() / stable_med.as_secs_f64(),
        );
    }

    println!(
        "\nDone.  (MEASURE_ITERS={MEASURE_ITERS}, median; on llvmpipe all times are CPU-bound)"
    );
}