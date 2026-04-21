//! Regression diagnostic for the cubic/tetragonal/orthorhombic CPU cell-list
//! mismatch that was observed on pre-release `bench_gpu` runs at rep ≥ 35.
//!
//! # What this tool does
//!
//! 1. Reproduces the historically failing supercells deterministically
//!    (Cubic rep=35 N=171500, Orthorhombic rep=75 N=1687500, Tetragonal
//!    rep=100 N=4000000) using the same Gaussian rattle that the benchmark
//!    uses.
//! 2. Compares GPU-CellList, CPU-CellList, GPU-AllPairs, and CPU-AllPairs
//!    on the same system so that any residual disagreement can be
//!    localised to a specific pipeline.
//! 3. For any atom whose CPU-CellList force disagrees with the GPU-AllPairs
//!    ground truth by more than `1e-3` eV/Å, prints its home cell,
//!    neighbour list, Cartesian stencil walk, and a ground-truth force
//!    computed from the brute-force min-image pair scan.
//!
//! On the current release this tool should report max `|Δf| ≲ 3e-4` eV/Å
//! across all four configurations — its primary role going forward is as a
//! regression guard.
//!
//! # Run
//! ```bash
//! cargo run --release --features cellist_gpu --bin diagnose_mismatch
//! ```

#![allow(clippy::too_many_arguments)]

use cream::potential::eam::EamPotential;
use cream::{ComputeEngine, CpuEngine, NeighborStrategy};

/// cell_size = cutoff (matching bench_gpu's default).
const CELL_SIZE: f32 = 4.95;

// ── Crystal system deformation (matches bench_gpu::crystal_supercell) ────────
//
// Bit-identical reproduction of the benchmark geometry is required or a
// historical MISMATCH would not recur.  These rules mirror the supercell
// builder in `src/bin/bench_gpu.rs`.

enum System {
    Cubic,
    Tetragonal,
    Orthorhombic,
}

impl System {
    fn name(&self) -> &'static str {
        match self {
            System::Cubic => "Cubic",
            System::Tetragonal => "Tetragonal",
            System::Orthorhombic => "Orthorhombic",
        }
    }

    /// Per-axis scaling (c/a, b/a).  Keep small so the stencil assumption
    /// (cell size ≥ cutoff) still holds after deformation.
    fn deformation(&self) -> [f32; 3] {
        match self {
            System::Cubic => [1.0, 1.0, 1.0],
            System::Tetragonal => [1.0, 1.0, 1.10],
            System::Orthorhombic => [1.0, 1.05, 1.10],
        }
    }
}

/// Build a deterministic FCC supercell with the specified deformation.
/// `rattle` (in Å) is added via a cheap per-atom LCG so the crystal has
/// non-zero forces — matches bench_gpu's rattle step.
fn build_supercell(
    system: &System,
    rep: usize,
    a: f32,
    rattle: f32,
) -> (Vec<[f32; 4]>, Vec<u32>, [[f32; 3]; 3]) {
    let scale = system.deformation();
    let ax = a * scale[0];
    let ay = a * scale[1];
    let az = a * scale[2];

    let basis: [[f32; 3]; 4] = [
        [0.0, 0.0, 0.0],
        [ax / 2.0, ay / 2.0, 0.0],
        [ax / 2.0, 0.0, az / 2.0],
        [0.0, ay / 2.0, az / 2.0],
    ];

    let n = rep * rep * rep * 4;
    let mut pos = Vec::with_capacity(n);
    let mut types = Vec::with_capacity(n);

    // Deterministic Gaussian rattle — EXACTLY matches bench_gpu::rattle()
    // with stdev=0.05 Å and seed=42.  This is critical: the MISMATCH is
    // triggered by Gaussian tails pushing atoms outside [0, L), which a
    // uniform rattle can't produce.
    let mut state: u64 = 42;
    let normal_pair = |state: &mut u64| -> (f32, f32) {
        let lcg = |s: &mut u64| -> f64 {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*s >> 33) as f64) / (1u64 << 31) as f64
        };
        let u1 = lcg(state).max(1e-10);
        let u2 = lcg(state);
        let mag = (-2.0 * u1.ln()).sqrt() as f32;
        let angle = (2.0 * std::f64::consts::PI * u2) as f32;
        (mag * angle.cos(), mag * angle.sin())
    };

    for ix in 0..rep {
        for iy in 0..rep {
            for iz in 0..rep {
                for b in &basis {
                    let x = b[0] + ix as f32 * ax;
                    let y = b[1] + iy as f32 * ay;
                    let z = b[2] + iz as f32 * az;
                    pos.push([x, y, z, 0.0]);
                    types.push(0u32);
                }
            }
        }
    }
    // Apply Gaussian rattle AFTER building lattice (separate step to match
    // bench_gpu's order).
    for p in pos.iter_mut() {
        let (dx, dy) = normal_pair(&mut state);
        let (dz, _) = normal_pair(&mut state);
        p[0] += rattle * dx;
        p[1] += rattle * dy;
        p[2] += rattle * dz;
    }

    let lx = ax * rep as f32;
    let ly = ay * rep as f32;
    let lz = az * rep as f32;
    let cell = [[lx, 0.0, 0.0], [0.0, ly, 0.0], [0.0, 0.0, lz]];
    (pos, types, cell)
}

// ── Synthetic Cu potential (matches bench_gpu's synth_cu_potential) ──────────

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

// ── Per-case diagnostic ──────────────────────────────────────────────────────

async fn diagnose_one(
    system: System,
    rep: usize,
    pot: &EamPotential,
) -> Result<(), Box<dyn std::error::Error>> {
    let a = 3.615_f32;
    let rattle = 0.05_f32;
    let (pos, types, cell) = build_supercell(&system, rep, a, rattle);
    let n = pos.len();

    println!();
    println!("════════════════════════════════════════════════════════════════════════");
    println!(
        "  {}  N = {}  (rep = {})  scale = {:?}",
        system.name(),
        n,
        rep,
        system.deformation()
    );
    println!("════════════════════════════════════════════════════════════════════════");

    // ── GPU AllPairs reference ──────────────────────────────────────────
    // For huge N this will be slow but it *is* the ground truth for the
    // GPU-side pair kernel.  Skip for N > 1 000 000.
    let ap_forces: Option<Vec<[f32; 3]>> = if n <= 1_000_000 {
        println!("[1/5] GPU-AllPairs reference …");
        let mut ap_engine = ComputeEngine::new(NeighborStrategy::AllPairs).await?;
        let ap_res = ap_engine.compute(&pos, &types, Some(cell), pot).await?;
        Some(ap_res.forces)
    } else {
        println!("[1/5] GPU-AllPairs skipped (N > 1e6)");
        None
    };

    // ── GPU CellList ────────────────────────────────────────────────────
    println!("[2/5] GPU-CellList …");
    let mut cl_engine = ComputeEngine::new(NeighborStrategy::CellList {
        cell_size: CELL_SIZE,
    })
    .await?;
    let cl_res = cl_engine.compute(&pos, &types, Some(cell), pot).await?;
    let gpu_cl_forces = &cl_res.forces;

    // ── CPU CellList ────────────────────────────────────────────────────
    // This is what bench_gpu uses as its reference.  We want to see
    // whether CPU-CL agrees with GPU-CL (and GPU-AP).
    println!("[3/5] CPU-CellList …");
    let cpu = CpuEngine::new();
    let cpu_res = cpu.compute_cell_list_sync(&pos, &types, Some(cell), pot)?;
    let cpu_cl_forces = &cpu_res.forces;

    // ── CPU AllPairs (second ground truth, for small N) ─────────────────
    // If CPU-AP agrees with GPU-AP and CPU-CL disagrees, the bug is purely
    // in the CPU cell list path (not in the CPU reference itself).
    let cpu_ap_forces: Option<Vec<[f32; 3]>> = if n <= 200_000 {
        println!("[4/5] CPU-AllPairs (ground truth check) …");
        let r = cpu.compute_sync(&pos, &types, Some(cell), pot)?;
        Some(r.forces)
    } else {
        println!("[4/5] CPU-AllPairs skipped (N > 200k)");
        None
    };

    // ── Diff analysis ───────────────────────────────────────────────────
    println!("[5/5] Diff analysis …");

    // Helper: component-wise max |a - b|.
    let max_diff = |a: &[[f32; 3]], b: &[[f32; 3]]| -> (f32, usize) {
        let mut m = 0.0f32;
        let mut mi = 0usize;
        for i in 0..a.len() {
            for c in 0..3 {
                let d = (a[i][c] - b[i][c]).abs();
                if d > m {
                    m = d;
                    mi = i;
                }
            }
        }
        (m, mi)
    };
    let count_above = |a: &[[f32; 3]], b: &[[f32; 3]], thr: f32| -> usize {
        let mut c = 0usize;
        for i in 0..a.len() {
            for k in 0..3 {
                if (a[i][k] - b[i][k]).abs() > thr {
                    c += 1;
                }
            }
        }
        c
    };
    let threshold = 1.0e-3_f32;

    println!();
    println!("Pairwise max |Δf| (eV/Å):");

    // GPU-CL vs GPU-AP
    if let Some(ap) = ap_forces.as_ref() {
        let (m, mi) = max_diff(ap, gpu_cl_forces);
        let above = count_above(ap, gpu_cl_forces, threshold);
        println!(
            "  GPU-CL vs GPU-AP:  max = {:.4e}  at atom {}  (#comp > {:.0e}: {} / {})",
            m,
            mi,
            threshold,
            above,
            3 * n
        );
    }

    // CPU-AP vs GPU-AP: sanity check, CPU-AP must agree with GPU-AP
    if let (Some(cpu_ap), Some(gpu_ap)) = (cpu_ap_forces.as_ref(), ap_forces.as_ref()) {
        let (m, mi) = max_diff(gpu_ap, cpu_ap);
        let above = count_above(gpu_ap, cpu_ap, threshold);
        let verdict = if m < threshold {
            "✓ OK"
        } else {
            "🚨 CPU-AP also wrong!"
        };
        println!(
            "  CPU-AP vs GPU-AP:  max = {:.4e}  at atom {}  (#comp > {:.0e}: {} / {})  {}",
            m,
            mi,
            threshold,
            above,
            3 * n,
            verdict
        );
    }

    // CPU-CL vs CPU-AP: key check, are they both correct on CPU?
    if let Some(cpu_ap) = cpu_ap_forces.as_ref() {
        let (m, mi) = max_diff(cpu_ap, cpu_cl_forces);
        let above = count_above(cpu_ap, cpu_cl_forces, threshold);
        let verdict = if m < threshold {
            "✓ OK"
        } else {
            "🚨 CPU-CL wrong vs CPU-AP"
        };
        println!(
            "  CPU-CL vs CPU-AP:  max = {:.4e}  at atom {}  (#comp > {:.0e}: {} / {})  {}",
            m,
            mi,
            threshold,
            above,
            3 * n,
            verdict
        );
    }

    // CPU-CL vs GPU-AP (the key comparison)
    if let Some(ap) = ap_forces.as_ref() {
        let (m, mi) = max_diff(ap, cpu_cl_forces);
        let above = count_above(ap, cpu_cl_forces, threshold);
        let verdict = if m < threshold { "✓ OK" } else { "🚨 BUG" };
        println!(
            "  CPU-CL vs GPU-AP:  max = {:.4e}  at atom {}  (#comp > {:.0e}: {} / {})  {}",
            m,
            mi,
            threshold,
            above,
            3 * n,
            verdict
        );

        if m > threshold {
            let p = &pos[mi];
            println!();
            println!("  🚨 CPU-CL is WRONG at atom {}:", mi);
            println!("    pos      = ({:.4}, {:.4}, {:.4})", p[0], p[1], p[2]);
            println!(
                "    GPU-AP f = ({:+.4e}, {:+.4e}, {:+.4e})  (ground truth)",
                ap[mi][0], ap[mi][1], ap[mi][2]
            );
            println!(
                "    CPU-CL f = ({:+.4e}, {:+.4e}, {:+.4e})  (bench reference)",
                cpu_cl_forces[mi][0], cpu_cl_forces[mi][1], cpu_cl_forces[mi][2]
            );
            println!(
                "    GPU-CL f = ({:+.4e}, {:+.4e}, {:+.4e})",
                gpu_cl_forces[mi][0], gpu_cl_forces[mi][1], gpu_cl_forces[mi][2]
            );

            // Grid info
            let lx = cell[0][0];
            let ly = cell[1][1];
            let lz = cell[2][2];
            let cutoff = 4.95_f32; // synth Cu cutoff
            let nx_cpu = (lx / cutoff).floor() as u32;
            let ny_cpu = (ly / cutoff).floor() as u32;
            let nz_cpu = (lz / cutoff).floor() as u32;
            let fx = (p[0] / lx).rem_euclid(1.0);
            let fy = (p[1] / ly).rem_euclid(1.0);
            let fz = (p[2] / lz).rem_euclid(1.0);
            let cx = ((fx * nx_cpu as f32) as u32).min(nx_cpu - 1);
            let cy = ((fy * ny_cpu as f32) as u32).min(ny_cpu - 1);
            let cz = ((fz * nz_cpu as f32) as u32).min(nz_cpu - 1);
            println!();
            println!("  CPU grid info:");
            println!(
                "    box = ({:.2}, {:.2}, {:.2})   cutoff = {}",
                lx, ly, lz, cutoff
            );
            println!(
                "    CPU ncx (cutoff-based, non-p2) = ({}, {}, {})",
                nx_cpu, ny_cpu, nz_cpu
            );
            println!("    atom {} home cell = ({}, {}, {})", mi, cx, cy, cz);
            let fracx = fx * nx_cpu as f32 - cx as f32;
            let fracy = fy * ny_cpu as f32 - cy as f32;
            let fracz = fz * nz_cpu as f32 - cz as f32;
            println!(
                "    frac within cell = ({:.3}, {:.3}, {:.3})  (0 or 1 = boundary)",
                fracx, fracy, fracz
            );
            // Is atom near box boundary?
            let at_box_edge_x = fx < 0.01 || fx > 0.99;
            let at_box_edge_y = fy < 0.01 || fy > 0.99;
            let at_box_edge_z = fz < 0.01 || fz > 0.99;
            if at_box_edge_x || at_box_edge_y || at_box_edge_z {
                println!(
                    "    ⚠ atom is near box boundary (fx={:.3}, fy={:.3}, fz={:.3})",
                    fx, fy, fz
                );
            }

            // ══════════════════════════════════════════════════════════════════
            // ── INJECTED FROM deep_debug_156728.rs ──────────────────────────
            //
            // Uses the LIVE buggy position set (the one that just produced
            // the CPU-CL mismatch) to do full ground-truth vs CL neighbour
            // enumeration, rho comparison, and manual force re-derivation.
            // No RNG sync games — we use the exact `pos` array in scope.
            // ══════════════════════════════════════════════════════════════════

            // Minimum-image distance for orthorhombic box (works for cubic,
            // tetragonal, orthorhombic; triclinic would need full min_image_mat).
            let min_image = |dx: f32, dy: f32, dz: f32| -> (f32, f32, f32) {
                let dx = dx - lx * (dx / lx).round();
                let dy = dy - ly * (dy / ly).round();
                let dz = dz - lz * (dz / lz).round();
                (dx, dy, dz)
            };

            let cutoff_sq = cutoff * cutoff;
            let t_pos = pos[mi];

            println!();
            println!(
                "  ── Ground-truth neighbour scan for atom {} ─────────────",
                mi
            );

            // Ground truth: brute-force O(N) scan using min-image
            let mut truth_neighbors: Vec<(usize, f32, [f32; 3])> = Vec::new();
            for j in 0..n {
                if j == mi {
                    continue;
                }
                let (dx, dy, dz) = min_image(
                    pos[j][0] - t_pos[0],
                    pos[j][1] - t_pos[1],
                    pos[j][2] - t_pos[2],
                );
                let r_sq = dx * dx + dy * dy + dz * dz;
                if r_sq > 0.0 && r_sq < cutoff_sq {
                    truth_neighbors.push((j, r_sq.sqrt(), [dx, dy, dz]));
                }
            }
            truth_neighbors.sort_by_key(|x| x.0);

            // Reimplement CPU-CL neighbour enumeration (matches
            // compute_cell_list_sync: wrap positions, assign cells at ncx_p2,
            // 27-stencil walk with PBC shifts).
            //
            // ncx from `CellListData::build` with the auto-power-of-two
            // rounding used by the engine:
            //   ncx_raw = floor(L / cutoff)
            //   ncx     = largest power of two ≤ ncx_raw
            let p2_down = |k: u32| -> u32 {
                if k == 0 {
                    1
                } else {
                    1u32 << (31 - k.leading_zeros())
                }
            };
            let ncx = p2_down(nx_cpu);
            let ncy = p2_down(ny_cpu);
            let ncz = p2_down(nz_cpu);
            println!(
                "    Actual CL grid (auto-p2):  ncx=({}, {}, {})",
                ncx, ncy, ncz
            );

            // Wrap all positions into [0, L)^3 (same as compute_cell_list_sync)
            let wrapped: Vec<[f32; 4]> = pos
                .iter()
                .map(|p| {
                    let sf0 = (p[0] / lx) - (p[0] / lx).floor();
                    let sf1 = (p[1] / ly) - (p[1] / ly).floor();
                    let sf2 = (p[2] / lz) - (p[2] / lz).floor();
                    let sf0 = if sf0 >= 1.0 { 0.0 } else { sf0 };
                    let sf1 = if sf1 >= 1.0 { 0.0 } else { sf1 };
                    let sf2 = if sf2 >= 1.0 { 0.0 } else { sf2 };
                    [sf0 * lx, sf1 * ly, sf2 * lz, 0.0]
                })
                .collect();
            let t_w = wrapped[mi];

            // Cell of target atom under the actual CL grid
            let cx0 = ((t_w[0] / lx) * ncx as f32).floor() as i32;
            let cy0 = ((t_w[1] / ly) * ncy as f32).floor() as i32;
            let cz0 = ((t_w[2] / lz) * ncz as f32).floor() as i32;
            println!(
                "    Wrapped target pos = ({:.4}, {:.4}, {:.4})",
                t_w[0], t_w[1], t_w[2]
            );
            println!("    Home cell (actual CL)  = ({}, {}, {})", cx0, cy0, cz0);

            // Assign each atom to its CL cell (same method as compute_cell_list_sync)
            let cells: Vec<(i32, i32, i32)> = (0..n)
                .map(|i| {
                    let w = wrapped[i];
                    (
                        ((w[0] / lx) * ncx as f32).floor() as i32,
                        ((w[1] / ly) * ncy as f32).floor() as i32,
                        ((w[2] / lz) * ncz as f32).floor() as i32,
                    )
                })
                .collect();

            // 27-stencil walk with PBC shifts
            let mut cl_neighbors: Vec<(usize, f32, [f32; 3], (i32, i32, i32))> = Vec::new();
            for dcx in -1..=1i32 {
                for dcy in -1..=1i32 {
                    for dcz in -1..=1i32 {
                        let nx_raw = cx0 + dcx;
                        let ny_raw = cy0 + dcy;
                        let nz_raw = cz0 + dcz;
                        let nx_w = nx_raw.rem_euclid(ncx as i32);
                        let ny_w = ny_raw.rem_euclid(ncy as i32);
                        let nz_w = nz_raw.rem_euclid(ncz as i32);
                        let kx = (nx_raw - nx_w) / ncx as i32;
                        let ky = (ny_raw - ny_w) / ncy as i32;
                        let kz = (nz_raw - nz_w) / ncz as i32;
                        let shift = [kx as f32 * lx, ky as f32 * ly, kz as f32 * lz];

                        for j in 0..n {
                            if j == mi {
                                continue;
                            }
                            let (jx, jy, jz) = cells[j];
                            if jx != nx_w || jy != ny_w || jz != nz_w {
                                continue;
                            }
                            let d = [
                                wrapped[j][0] - t_w[0] + shift[0],
                                wrapped[j][1] - t_w[1] + shift[1],
                                wrapped[j][2] - t_w[2] + shift[2],
                            ];
                            let r_sq = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                            if r_sq > 0.0 && r_sq < cutoff_sq {
                                cl_neighbors.push((j, r_sq.sqrt(), d, (dcx, dcy, dcz)));
                            }
                        }
                    }
                }
            }
            cl_neighbors.sort_by_key(|x| x.0);

            // Diff truth vs CL
            let truth_set: std::collections::HashSet<usize> =
                truth_neighbors.iter().map(|x| x.0).collect();
            let cl_counts: std::collections::HashMap<usize, Vec<(f32, [f32; 3], (i32, i32, i32))>> =
                cl_neighbors
                    .iter()
                    .fold(std::collections::HashMap::new(), |mut m, x| {
                        m.entry(x.0).or_default().push((x.1, x.2, x.3));
                        m
                    });

            let missing: Vec<usize> = truth_set
                .iter()
                .filter(|j| !cl_counts.contains_key(j))
                .copied()
                .collect();
            let extra: Vec<usize> = cl_counts
                .keys()
                .filter(|j| !truth_set.contains(j))
                .copied()
                .collect();
            let duplicated: Vec<(usize, usize)> = cl_counts
                .iter()
                .filter(|(_, v)| v.len() > 1)
                .map(|(&k, v)| (k, v.len()))
                .collect();

            println!(
                "    {} truth neighbours, {} CL neighbours",
                truth_neighbors.len(),
                cl_neighbors.len()
            );
            println!("    Missing from CL: {:?}", missing);
            println!("    Extra in CL:     {:?}", extra);
            println!("    Duplicated in CL: {:?}", duplicated);

            // Detail on missing/extra/duplicated neighbours
            if !missing.is_empty() {
                println!();
                println!("    🚨 MISSING neighbours (within cutoff but not in CL):");
                for j in missing.iter().take(10) {
                    let truth = truth_neighbors.iter().find(|x| x.0 == *j).unwrap();
                    println!("      j={:6}  r={:.4}  d=({:+.4}, {:+.4}, {:+.4})  pos_j=({:.4}, {:.4}, {:.4})",
                        j, truth.1, truth.2[0], truth.2[1], truth.2[2],
                        pos[*j][0], pos[*j][1], pos[*j][2]);
                    // Which cell does this atom occupy?  What stencil offset was needed?
                    let (jx, jy, jz) = cells[*j];
                    println!(
                        "         wrapped_pos_j = ({:.4}, {:.4}, {:.4})  cell_j = ({}, {}, {})",
                        wrapped[*j][0], wrapped[*j][1], wrapped[*j][2], jx, jy, jz
                    );
                    let req_dcx = (jx as i32 - cx0).rem_euclid(ncx as i32);
                    let req_dcy = (jy as i32 - cy0).rem_euclid(ncy as i32);
                    let req_dcz = (jz as i32 - cz0).rem_euclid(ncz as i32);
                    // Convert to -1..=1 representation
                    let to_signed = |r: i32, n: i32| if r > n / 2 { r - n } else { r };
                    println!(
                        "         needed stencil offset (signed): ({}, {}, {})",
                        to_signed(req_dcx, ncx as i32),
                        to_signed(req_dcy, ncy as i32),
                        to_signed(req_dcz, ncz as i32)
                    );
                }
                if missing.len() > 10 {
                    println!("      ... and {} more", missing.len() - 10);
                }
            }
            if !extra.is_empty() {
                println!();
                println!("    🚨 EXTRA neighbours (in CL but not within cutoff):");
                for j in extra.iter().take(10) {
                    let e = cl_counts.get(j).unwrap();
                    for (r, d, dc) in e {
                        println!("      j={:6}  r={:.4}  d=({:+.4}, {:+.4}, {:+.4})  via stencil=({}, {}, {})",
                            j, r, d[0], d[1], d[2], dc.0, dc.1, dc.2);
                    }
                }
            }
            if !duplicated.is_empty() {
                println!();
                println!("    🚨 DUPLICATED neighbours (CL visits same atom multiple times):");
                for (j, cnt) in &duplicated {
                    println!("      j={} counted {} times:", j, cnt);
                    if let Some(entries) = cl_counts.get(j) {
                        for (r, d, dc) in entries {
                            println!("        r={:.4}  d=({:+.4}, {:+.4}, {:+.4})  via stencil=({}, {}, {})",
                                r, d[0], d[1], d[2], dc.0, dc.1, dc.2);
                        }
                    }
                }
            }

            // ── rho comparison ─────────────────────────────────────────────
            // Build synth-Cu rho table (matches this binary's synth_cu_potential)
            let dr: f32 = 0.025;
            let dr_inv: f32 = 1.0 / dr;
            let nr = 200usize;
            let rho_table: Vec<f32> = (0..nr)
                .map(|i| {
                    let r = i as f32 * dr;
                    if r < cutoff {
                        let t = cutoff - r;
                        t * t / (cutoff * cutoff)
                    } else {
                        0.0
                    }
                })
                .collect();
            let lookup_f32 = |table: &[f32], idx_f: f32| -> f32 {
                let idx = idx_f as usize;
                let frac = idx_f - idx as f32;
                let i0 = idx.min(nr.saturating_sub(2));
                table[i0] + frac * (table[(i0 + 1).min(nr - 1)] - table[i0])
            };

            let mut rho_truth: f64 = 0.0;
            for (_j, r, _d) in &truth_neighbors {
                rho_truth += lookup_f32(&rho_table, r * dr_inv) as f64;
            }
            let mut rho_cl: f64 = 0.0;
            for (_j, r, _d, _dc) in &cl_neighbors {
                rho_cl += lookup_f32(&rho_table, r * dr_inv) as f64;
            }
            println!();
            println!("    ρ_target from truth neighbours = {:.6}", rho_truth);
            println!("    ρ_target from CL neighbours    = {:.6}", rho_cl);
            println!("    Δρ = {:.4e}", (rho_truth - rho_cl).abs());

            // ── Manual force from ground truth (f64 accum, for comparison) ──
            let drho: f32 = 0.01;
            let drho_inv: f32 = 1.0 / drho;
            let nrho = 200usize;
            let d_embed_table: Vec<f32> = (0..nrho)
                .map(|i| {
                    let rho = i as f32 * drho;
                    -0.5 / (rho + 0.01_f32).sqrt()
                })
                .collect();
            let pair_table: Vec<f32> = (0..nr)
                .map(|i| {
                    let r = i as f32 * dr;
                    if r > 0.5 && r < cutoff {
                        let t = cutoff - r;
                        t * t * t / (r * cutoff * cutoff * cutoff)
                    } else {
                        0.0
                    }
                })
                .collect();
            let d_pair_table: Vec<f32> = {
                let mut v = vec![0.0f32; nr];
                for i in 0..(nr - 1) {
                    v[i] = (pair_table[i + 1] - pair_table[i]) * dr_inv;
                }
                v[nr - 1] = v[nr - 2];
                v
            };
            let d_rho_table: Vec<f32> = {
                let mut v = vec![0.0f32; nr];
                for i in 0..(nr - 1) {
                    v[i] = (rho_table[i + 1] - rho_table[i]) * dr_inv;
                }
                v[nr - 1] = v[nr - 2];
                v
            };
            let lookup_rho_table = |table: &[f32], idx_f: f32| -> f32 {
                let idx = idx_f as usize;
                let frac = idx_f - idx as f32;
                let i0 = idx.min(nrho.saturating_sub(2));
                table[i0] + frac * (table[(i0 + 1).min(nrho - 1)] - table[i0])
            };

            // df_embed for target
            let df_i = lookup_rho_table(&d_embed_table, rho_truth as f32 * drho_inv) as f64;

            // Compute rho for each neighbour (brute force O(42 × N))
            let mut df_embed_neighbors = vec![0.0f32; truth_neighbors.len()];
            for (nb_idx, &(j, _, _)) in truth_neighbors.iter().enumerate() {
                let pj = pos[j];
                let mut rho_j: f64 = 0.0;
                for k in 0..n {
                    if k == j {
                        continue;
                    }
                    let (dx, dy, dz) =
                        min_image(pos[k][0] - pj[0], pos[k][1] - pj[1], pos[k][2] - pj[2]);
                    let r_sq = dx * dx + dy * dy + dz * dz;
                    if r_sq > 0.0 && r_sq < cutoff_sq {
                        rho_j += lookup_f32(&rho_table, r_sq.sqrt() * dr_inv) as f64;
                    }
                }
                df_embed_neighbors[nb_idx] =
                    lookup_rho_table(&d_embed_table, rho_j as f32 * drho_inv);
            }

            // Manual force (f64 accum) from the 42 truth neighbours
            let mut f_manual = [0.0f64; 3];
            for (nb_idx, (_j, r, d)) in truth_neighbors.iter().enumerate() {
                let r_inv = 1.0 / *r as f64;
                let df_beta_dr = lookup_f32(&d_rho_table, r * dr_inv) as f64;
                let df_alpha_dr = df_beta_dr; // same element
                let dphi_dr = lookup_f32(&d_pair_table, r * dr_inv) as f64;
                let df_j = df_embed_neighbors[nb_idx] as f64;
                let coeff = df_i * df_beta_dr + df_j * df_alpha_dr + dphi_dr;
                f_manual[0] += coeff * d[0] as f64 * r_inv;
                f_manual[1] += coeff * d[1] as f64 * r_inv;
                f_manual[2] += coeff * d[2] as f64 * r_inv;
            }
            println!();
            println!("    Manual force (42 truth pairs, f64 accum):");
            println!(
                "      ({:+.6e}, {:+.6e}, {:+.6e})",
                f_manual[0], f_manual[1], f_manual[2]
            );
            println!("    CPU-CL force (what cpu_engine produced):");
            println!(
                "      ({:+.6e}, {:+.6e}, {:+.6e})",
                cpu_cl_forces[mi][0], cpu_cl_forces[mi][1], cpu_cl_forces[mi][2]
            );
            println!("    GPU-AP force (ground truth):");
            println!(
                "      ({:+.6e}, {:+.6e}, {:+.6e})",
                ap[mi][0], ap[mi][1], ap[mi][2]
            );
            println!("    Δ(manual - CPU-CL):");
            println!(
                "      ({:+.6e}, {:+.6e}, {:+.6e})",
                f_manual[0] - cpu_cl_forces[mi][0] as f64,
                f_manual[1] - cpu_cl_forces[mi][1] as f64,
                f_manual[2] - cpu_cl_forces[mi][2] as f64
            );
            println!("    Δ(manual - GPU-AP):");
            println!(
                "      ({:+.6e}, {:+.6e}, {:+.6e})",
                f_manual[0] - ap[mi][0] as f64,
                f_manual[1] - ap[mi][1] as f64,
                f_manual[2] - ap[mi][2] as f64
            );

            // ══════════════════════════════════════════════════════════════════
            // ── END INJECTED BLOCK ───────────────────────────────────────────
            // ══════════════════════════════════════════════════════════════════
        }
    }

    // CPU-CL vs GPU-CL (reproduces bench_gpu's comparison)
    {
        let (m, mi) = max_diff(gpu_cl_forces, cpu_cl_forces);
        let above = count_above(gpu_cl_forces, cpu_cl_forces, threshold);
        let verdict = if m < threshold {
            "✓ OK"
        } else {
            "🚨 bench MISMATCH"
        };
        println!(
            "  CPU-CL vs GPU-CL:  max = {:.4e}  at atom {}  (#comp > {:.0e}: {} / {})  {}",
            m,
            mi,
            threshold,
            above,
            3 * n,
            verdict
        );
    }

    // Physical sanity: Σf must be near zero (Newton 3 + PBC)
    let mut sum = [0.0f64; 3];
    for f in gpu_cl_forces {
        for k in 0..3 {
            sum[k] += f[k] as f64;
        }
    }
    let drift = (sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]).sqrt();
    println!(
        "  GPU-CL Σf  = ({:+.2e}, {:+.2e}, {:+.2e})  |Σf| = {:.2e}",
        sum[0], sum[1], sum[2], drift
    );
    let mut sum = [0.0f64; 3];
    for f in cpu_cl_forces {
        for k in 0..3 {
            sum[k] += f[k] as f64;
        }
    }
    let drift_cpu = (sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]).sqrt();
    println!(
        "  CPU-CL Σf  = ({:+.2e}, {:+.2e}, {:+.2e})  |Σf| = {:.2e}",
        sum[0], sum[1], sum[2], drift_cpu
    );

    Ok(())
}

fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Warn)
        .init();

    println!();
    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║   diagnose_mismatch · investigate bench_gpu MISMATCH at rep ≥ 35       ║");
    println!("║                                                                        ║");
    println!("║   Compares GPU-CellList against GPU-AllPairs on the identical system.  ║");
    println!("║   If AP agrees with bench_gpu's CPU reference but CL disagrees, the    ║");
    println!("║   bug is in the CellList shaders.  Otherwise it's somewhere else.     ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝");

    // Use the same synthetic Cu potential as bench_gpu so we reproduce the
    // exact test conditions.
    let pot = synth_cu_potential();

    // The 3 MISMATCH cases from the 2026-04-18 log:
    let cases = [
        (System::Cubic, 35usize),
        (System::Orthorhombic, 75usize),
        (System::Tetragonal, 100usize),
    ];

    // Plus 1 "nearby" case that passes, as a sanity anchor.
    let anchor = (System::Cubic, 30usize); // N = 108000, max_err ≈ 3.0e-4

    println!();
    println!(
        "── Anchor case (expected PASS): {} rep={} N={} ──",
        anchor.0.name(),
        anchor.1,
        anchor.1.pow(3) * 4
    );
    if let Err(e) = pollster::block_on(diagnose_one(anchor.0, anchor.1, &pot)) {
        eprintln!("Anchor failed: {}", e);
    }

    for (sys, rep) in cases {
        if let Err(e) = pollster::block_on(diagnose_one(sys, rep, &pot)) {
            eprintln!("Case failed: {}", e);
        }
    }

    println!();
    println!("── Done ────────────────────────────────────────────────────────────────");
}
