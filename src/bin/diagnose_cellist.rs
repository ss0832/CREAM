//! CellList GPU vs AllPairs diagnostic tool (v2).
//!
//! # Purpose
//! Reproduce and localise the numerical mismatch between the GPU CellList
//! shaders (`eam_pass{1,2}_cellist.wgsl`) and the GPU AllPairs reference.
//!
//! # What's new in v2
//! v1 used `cell_size = 5.5 Å` which is larger than the cutoff (4.95 Å), and
//! this caused the CPU reference grid (sized by `cutoff`) to disagree with the
//! GPU grid (sized by `cell_size`) for some N, producing false-positive
//! pass-0 mismatches.  v2:
//!
//!   1. Uses `cell_size = cutoff` by default, matching the Python binding.
//!   2. Adds an optional deterministic rattle (~0.05 Å stdev) because a
//!      perfect FCC crystal has zero forces by symmetry — any numerical
//!      mistake has nothing to be measured against.
//!   3. Replaces the "density range" sanity check with an energy diff vs
//!      AllPairs.  Energy is a scalar summary that catches both density and
//!      force-accumulation bugs:
//!        E = Σᵢ [ F(ρᵢ) + ½ Σⱼ φ(r_ij) ]
//!      so any mistake in ρ propagates into E through both embedding term F
//!      and pair term φ.
//!   4. Scans multiple `cell_size / cutoff` ratios to test the hypothesis
//!      that the CellList shaders assume `cell_size ≈ cutoff` and break when
//!      the ratio grows.
//!
//! # Build and run
//! ```bash
//! cargo run --release --features cellist_gpu --bin diagnose_cellist
//! ```
//!
//! # Backend selection (Windows PowerShell)
//! ```powershell
//! $env:WGPU_BACKEND="dx12"
//! $env:WGPU_ADAPTER_NAME="NVIDIA"
//! cargo run --release --features cellist_gpu --bin diagnose_cellist
//! ```

#![allow(clippy::too_many_arguments)]

use cream::potential::eam::EamPotential;
use cream::potential::GpuPotential;
use cream::{ComputeEngine, NeighborStrategy};
use std::fmt::Write as _;

/// Force / pair-distance threshold for "this atom is mismatched" (eV / Å).
const NOISE_FLOOR: f32 = 1.0e-5;

/// Deterministic rattle amplitude used to break FCC symmetry.  Matches the
/// order of magnitude of ASE's `atoms.rattle(stdev=0.05)` default.
const RATTLE_AMPLITUDE: f32 = 0.05;

// ── FCC Cu supercell ─────────────────────────────────────────────────────────

fn fcc_supercell(rep: usize, rattle: bool) -> (Vec<[f32; 4]>, Vec<u32>, [[f32; 3]; 3]) {
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
    if rattle {
        // Deterministic PRNG: SplitMix64-style u32 hash → two f32 normals via
        // Box-Muller.  Seeding by atom index (not time) so re-runs give the
        // same perturbation.  stdev = RATTLE_AMPLITUDE.
        fn hash_u32(mut x: u32) -> u32 {
            x = x.wrapping_add(0x9e3779b9);
            x = (x ^ (x >> 16)).wrapping_mul(0x85ebca6b);
            x = (x ^ (x >> 13)).wrapping_mul(0xc2b2ae35);
            x ^ (x >> 16)
        }
        fn uniform01(seed: u32) -> f32 {
            // Map u32 → (0, 1) open interval so log() is safe.
            ((hash_u32(seed) >> 8) as f32 + 0.5) / ((1u64 << 24) as f32)
        }
        for (i, p) in pos.iter_mut().enumerate() {
            for c in 0..3 {
                let u1 = uniform01((i as u32) * 6 + (c as u32) * 2);
                let u2 = uniform01((i as u32) * 6 + (c as u32) * 2 + 1);
                // Box-Muller: z = sqrt(-2 ln u1) * cos(2π u2)
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                p[c] += RATTLE_AMPLITUDE * z;
            }
        }
    }
    let l = a * rep as f32;
    let cell = [[l, 0.0, 0.0], [0.0, l, 0.0], [0.0, 0.0, l]];
    (pos, types, cell)
}

fn synth_cu_potential() -> EamPotential {
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

// ── Adapter info ─────────────────────────────────────────────────────────────

async fn print_adapter_info() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapters: Vec<_> = instance.enumerate_adapters(wgpu::Backends::all());
    println!("── Available wgpu adapters ──");
    for (i, a) in adapters.iter().enumerate() {
        let info = a.get_info();
        println!(
            "  [{i}] {} ({:?}) via {:?}  — driver: {}",
            info.name, info.device_type, info.backend, info.driver_info
        );
    }
    println!();
}

// ── Force diff stats ─────────────────────────────────────────────────────────

fn force_diff_stats(f_a: &[[f32; 3]], f_b: &[[f32; 3]]) -> (f64, f32, usize) {
    assert_eq!(f_a.len(), f_b.len());
    let mut sum_sq = 0.0f64;
    let mut max_abs = 0.0f32;
    let mut worst_atom = 0usize;
    for i in 0..f_a.len() {
        for c in 0..3 {
            let d = f_a[i][c] - f_b[i][c];
            sum_sq += (d as f64) * (d as f64);
            let ad = d.abs();
            if ad > max_abs {
                max_abs = ad;
                worst_atom = i;
            }
        }
    }
    (sum_sq.sqrt(), max_abs, worst_atom)
}

fn count_mismatched_atoms(f_a: &[[f32; 3]], f_b: &[[f32; 3]], thresh: f32) -> usize {
    f_a.iter()
        .zip(f_b.iter())
        .filter(|(a, b)| (0..3).any(|c| (a[c] - b[c]).abs() > thresh))
        .count()
}

fn top_offenders_forces(
    f_cl: &[[f32; 3]],
    f_ap: &[[f32; 3]],
    cell_ids: &[u32],
    cell_start: &[u32],
    k_top: usize,
) -> Vec<(usize, f32, u32, u32)> {
    let mut stats: Vec<(usize, f32, u32, u32)> = Vec::with_capacity(f_cl.len());
    for i in 0..f_cl.len() {
        let mut max_c = 0.0f32;
        for c in 0..3 {
            max_c = max_c.max((f_cl[i][c] - f_ap[i][c]).abs());
        }
        if max_c > NOISE_FLOOR {
            let cid = cell_ids.get(i).copied().unwrap_or(u32::MAX);
            let cs = if (cid as usize) + 1 < cell_start.len() {
                cell_start[cid as usize + 1] - cell_start[cid as usize]
            } else {
                0
            };
            stats.push((i, max_c, cid, cs));
        }
    }
    stats.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    stats.truncate(k_top);
    stats
}

// ── Engine-matching grid calculation ─────────────────────────────────────────
//
// Must match `ComputeEngine::n_cells_from_dspacing` in `engine.rs` EXACTLY —
// otherwise the CPU reference will disagree with the GPU for reasons that have
// nothing to do with shader bugs.  This is a straight port of that function,
// including the auto-power-of-two rounding described in `engine.rs`.
fn engine_n_cells(h: &[[f32; 3]; 3], cell_size: f32) -> [u32; 3] {
    let [a, b, c] = *h;
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
    let vol = dot3(a, bc).abs();
    let (bc_n, ca_n, ab_n) = (norm(bc), norm(ca), norm(ab));
    let d_a = if bc_n > 1e-12 { vol / bc_n } else { cell_size };
    let d_b = if ca_n > 1e-12 { vol / ca_n } else { cell_size };
    let d_c = if ab_n > 1e-12 { vol / ab_n } else { cell_size };
    let nx_raw = ((d_a / cell_size).floor() as u32).max(1);
    let ny_raw = ((d_b / cell_size).floor() as u32).max(1);
    let nz_raw = ((d_c / cell_size).floor() as u32).max(1);

    // Mirror engine.rs::n_cells_from_dspacing auto-p2 rounding.
    let auto_p2 = !std::env::var("CREAM_DISABLE_AUTO_P2")
        .ok()
        .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
    let round_down_p2 = |n: u32| -> u32 {
        if n == 0 {
            1
        } else {
            1u32 << (31 - n.leading_zeros())
        }
    };

    if auto_p2 {
        [
            round_down_p2(nx_raw),
            round_down_p2(ny_raw),
            round_down_p2(nz_raw),
        ]
    } else {
        [nx_raw, ny_raw, nz_raw]
    }
}

// ── Pass-0 CPU reference ─────────────────────────────────────────────────────
//
// Returns what pass0a/0b should have built, using the SAME grid sizing rule as
// `ComputeEngine::n_cells_from_dspacing` (keyed on the caller-provided
// `cell_size`, not on `cutoff`).  Intra-cell order is not guaranteed on GPU,
// so `sorted_atoms` must be compared as a multiset per cell.
fn cpu_pass0_reference(
    positions: &[[f32; 4]],
    cell: [[f32; 3]; 3],
    cell_size: f32,
) -> (Vec<u32>, Vec<u32>, Vec<u32>, [u32; 3]) {
    let n_cells = engine_n_cells(&cell, cell_size);

    // h_inv (same formula as cell_list::mat3_inv_f32).
    let inv_det = {
        let h = cell;
        let det = h[0][0] * (h[1][1] * h[2][2] - h[1][2] * h[2][1])
            - h[0][1] * (h[1][0] * h[2][2] - h[1][2] * h[2][0])
            + h[0][2] * (h[1][0] * h[2][1] - h[1][1] * h[2][0]);
        assert!(det.abs() > 1e-30, "singular cell in diagnostic reference");
        1.0 / det
    };
    let h_inv = {
        let h = cell;
        let d = inv_det;
        [
            [
                (h[1][1] * h[2][2] - h[1][2] * h[2][1]) * d,
                -(h[0][1] * h[2][2] - h[0][2] * h[2][1]) * d,
                (h[0][1] * h[1][2] - h[0][2] * h[1][1]) * d,
            ],
            [
                -(h[1][0] * h[2][2] - h[1][2] * h[2][0]) * d,
                (h[0][0] * h[2][2] - h[0][2] * h[2][0]) * d,
                -(h[0][0] * h[1][2] - h[0][2] * h[1][0]) * d,
            ],
            [
                (h[1][0] * h[2][1] - h[1][1] * h[2][0]) * d,
                -(h[0][0] * h[2][1] - h[0][1] * h[2][0]) * d,
                (h[0][0] * h[1][1] - h[0][1] * h[1][0]) * d,
            ],
        ]
    };

    let n_pad = [
        (n_cells[0] as usize).next_power_of_two(),
        (n_cells[1] as usize).next_power_of_two(),
        (n_cells[2] as usize).next_power_of_two(),
    ];

    fn spread(v: u32) -> u32 {
        let mut x = v & 0x0000_03ff;
        x = (x | (x << 16)) & 0x0300_00ff;
        x = (x | (x << 8)) & 0x0300_f00f;
        x = (x | (x << 4)) & 0x030c_30c3;
        x = (x | (x << 2)) & 0x0924_9249;
        x
    }
    let morton =
        |cx: u32, cy: u32, cz: u32| -> u32 { spread(cx) | (spread(cy) << 1) | (spread(cz) << 2) };
    let n_morton = morton(
        (n_pad[0] - 1) as u32,
        (n_pad[1] - 1) as u32,
        (n_pad[2] - 1) as u32,
    ) as usize
        + 1;

    // Cell assignment — fractional coordinates, floor, rem_euclid, Morton code.
    let mut cell_ids = vec![0u32; positions.len()];
    for (i, p) in positions.iter().enumerate() {
        let s0 = p[0] * h_inv[0][0] + p[1] * h_inv[1][0] + p[2] * h_inv[2][0];
        let s1 = p[0] * h_inv[0][1] + p[1] * h_inv[1][1] + p[2] * h_inv[2][1];
        let s2 = p[0] * h_inv[0][2] + p[1] * h_inv[1][2] + p[2] * h_inv[2][2];
        let cx = ((s0 * n_cells[0] as f32).floor() as i32).rem_euclid(n_cells[0] as i32) as u32;
        let cy = ((s1 * n_cells[1] as f32).floor() as i32).rem_euclid(n_cells[1] as i32) as u32;
        let cz = ((s2 * n_cells[2] as f32).floor() as i32).rem_euclid(n_cells[2] as i32) as u32;
        cell_ids[i] = morton(cx, cy, cz);
    }

    let mut keyed: Vec<(u32, u32)> = cell_ids
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, i as u32))
        .collect();
    keyed.sort_by_key(|&(c, _)| c);
    let sorted_atoms: Vec<u32> = keyed.iter().map(|&(_, i)| i).collect();

    let mut cell_start = vec![0u32; n_morton + 1];
    for &(c, _) in &keyed {
        cell_start[(c as usize) + 1] += 1;
    }
    for m in 0..n_morton {
        cell_start[m + 1] += cell_start[m];
    }

    (cell_ids, sorted_atoms, cell_start, n_cells)
}

// ── Verdict enum ─────────────────────────────────────────────────────────────

#[derive(Copy, Clone, Debug)]
enum Verdict {
    Ok,
    Pass0Broken,
    Pass12Broken,
    Pass0AndPass12Broken,
}

impl Verdict {
    fn as_str(self) -> &'static str {
        match self {
            Verdict::Ok => "OK",
            Verdict::Pass0Broken => "Pass-0 build BROKEN",
            Verdict::Pass12Broken => "Pass-1/2 physics BROKEN",
            Verdict::Pass0AndPass12Broken => "Pass-0 + Pass-1/2 BROKEN",
        }
    }
}

// ── Core diagnostic for one (N, rattle, cell_size_ratio) ────────────────────

struct DiagOutcome {
    pass0_ok: bool,
    pass12_ok: bool,
    energy_diff: f32,
    force_max: f32,
    force_l2: f64,
    n_cells: [u32; 3],
    cid_mismatches: usize,
    rattle: bool,
    cell_size_ratio: f32,
    n: usize,
}

async fn diagnose(
    rep: usize,
    rattle: bool,
    cell_size_ratio: f32,
    pot: &EamPotential,
    verbose: bool,
) -> DiagOutcome {
    let cutoff = pot.cutoff();
    let cell_size = cutoff * cell_size_ratio;
    let (pos, types, cell) = fcc_supercell(rep, rattle);
    let n = pos.len();

    if verbose {
        println!("\n{}", "═".repeat(80));
        println!(
            "   N = {n}  (rep {rep}³ × 4)   rattle = {rattle}   cell_size / cutoff = {cell_size_ratio:.3}"
        );
        println!("   cutoff = {cutoff:.4} Å,  cell_size = {cell_size:.4} Å");
        println!("{}", "═".repeat(80));
    }

    // ── GPU AllPairs reference ───────────────────────────────────────────────
    let mut eng_ap = ComputeEngine::new(NeighborStrategy::AllPairs)
        .await
        .expect("AllPairs engine init");
    let ap = eng_ap
        .compute(&pos, &types, Some(cell), pot)
        .await
        .expect("AllPairs compute");

    // ── GPU CellList with debug readback ─────────────────────────────────────
    let mut eng_cl = ComputeEngine::new(NeighborStrategy::CellList { cell_size })
        .await
        .expect("CellList engine init");
    let (cl, dbg_opt) = eng_cl
        .compute_with_debug(&pos, &types, Some(cell), pot)
        .await
        .expect("CellList compute_with_debug");
    let dbg = dbg_opt.expect("CellList strategy should yield CellListDebugReadback");

    // ── 1. Pass-0 integrity ──────────────────────────────────────────────────
    let (cpu_cell_ids, cpu_sorted_atoms, cpu_cell_start, cpu_ncells) =
        cpu_pass0_reference(&pos, cell, cell_size);

    let grid_ok = [dbg.n_cells.0, dbg.n_cells.1, dbg.n_cells.2] == cpu_ncells
        && dbg.n_morton + 1 == cpu_cell_start.len();

    let cid_mismatches: Vec<(usize, u32, u32)> = (0..n)
        .filter_map(|i| {
            if dbg.cell_ids[i] != cpu_cell_ids[i] {
                Some((i, cpu_cell_ids[i], dbg.cell_ids[i]))
            } else {
                None
            }
        })
        .collect();

    let cs_mismatches: Vec<(usize, u32, u32)> = (0..dbg.cell_start.len().min(cpu_cell_start.len()))
        .filter_map(|m| {
            if dbg.cell_start[m] != cpu_cell_start[m] {
                Some((m, cpu_cell_start[m], dbg.cell_start[m]))
            } else {
                None
            }
        })
        .collect();

    let mut sa_cell_mismatches = 0usize;
    let mut first_sa_mismatch = None;
    for m in 0..cpu_cell_start.len().saturating_sub(1) {
        let s = cpu_cell_start[m] as usize;
        let e = cpu_cell_start[m + 1] as usize;
        if s == e {
            continue;
        }
        let mut cpu_slice: Vec<u32> = cpu_sorted_atoms[s..e].to_vec();
        let mut gpu_slice: Vec<u32> = if e <= dbg.sorted_atoms.len() {
            dbg.sorted_atoms[s..e].to_vec()
        } else {
            Vec::new()
        };
        cpu_slice.sort_unstable();
        gpu_slice.sort_unstable();
        if cpu_slice != gpu_slice {
            sa_cell_mismatches += 1;
            if first_sa_mismatch.is_none() {
                first_sa_mismatch = Some((m, cpu_slice, gpu_slice));
            }
        }
    }

    let mut rp_xyz_max_diff = 0.0f32;
    let mut rp_w_mismatch = 0usize;
    for k in 0..n {
        let j = dbg.sorted_atoms[k] as usize;
        let want_xyz = &pos[j];
        let got = &dbg.reordered_positions[k];
        for c in 0..3 {
            rp_xyz_max_diff = rp_xyz_max_diff.max((want_xyz[c] - got[c]).abs());
        }
        let want_w = dbg.cell_ids[j] as f32;
        if (got[3] - want_w).abs() > 0.5 {
            rp_w_mismatch += 1;
        }
    }

    let pass0_ok = grid_ok
        && cid_mismatches.is_empty()
        && cs_mismatches.is_empty()
        && sa_cell_mismatches == 0
        && rp_xyz_max_diff < 1e-6
        && rp_w_mismatch == 0;

    if verbose {
        println!(
            "Grid:        GPU n_cells = {:?}  (pad {:?})   CPU n_cells = {:?}",
            dbg.n_cells, dbg.n_cells_pad, cpu_ncells
        );
        println!(
            "             n_morton = {} (GPU) / {} (CPU)",
            dbg.n_morton,
            cpu_cell_start.len() - 1
        );
        println!(
            "cell_ids:    {} mismatches / {}   {}",
            cid_mismatches.len(),
            n,
            if cid_mismatches.is_empty() {
                "[OK]"
            } else {
                "[FAIL]"
            }
        );
        for (i, cpu_m, gpu_m) in cid_mismatches.iter().take(3) {
            println!(
                "   atom {i}: CPU morton={cpu_m}  ≠  GPU morton={gpu_m}  (pos {:?})",
                [pos[*i][0], pos[*i][1], pos[*i][2]]
            );
        }
        println!(
            "cell_start:  {} mismatches / {}   {}",
            cs_mismatches.len(),
            dbg.cell_start.len(),
            if cs_mismatches.is_empty() {
                "[OK]"
            } else {
                "[FAIL]"
            }
        );
        println!(
            "sorted_set:  {} cells differ / {}   {}",
            sa_cell_mismatches,
            cpu_cell_start.len().saturating_sub(1),
            if sa_cell_mismatches == 0 {
                "[OK]"
            } else {
                "[FAIL]"
            }
        );
        if let Some((m, cpu_s, gpu_s)) = first_sa_mismatch {
            println!("   first differing cell m={m}:");
            println!("     CPU: {cpu_s:?}");
            println!("     GPU: {gpu_s:?}");
        }
        println!(
            "reord.xyz:   max |Δ| = {:.3e}  {}",
            rp_xyz_max_diff,
            if rp_xyz_max_diff < 1e-6 {
                "[OK]"
            } else {
                "[FAIL]"
            }
        );
        println!(
            "reord.w:     {} mismatches / {}  {}",
            rp_w_mismatch,
            n,
            if rp_w_mismatch == 0 { "[OK]" } else { "[FAIL]" }
        );
    }

    // ── 2. Debug counters (enabled via CREAM_ENABLE_DEBUG) ───────────────────
    let dbg_vals = dbg.debug_flags;
    let dbg_nonzero = dbg_vals.iter().any(|&v| v != 0);
    if verbose {
        if dbg_nonzero {
            println!("Shader counters (ENABLE_DEBUG=true):");
            // Nonzero slots with human-readable labels.
            let labels: &[(usize, &str)] = &[
                (0, "pass1_cid_oob"),
                (1, "pass1_cs_oob"),
                (2, "pass1_cs_inverted"),
                (3, "pass1_bb_empty"),
                (4, "pass1_atom_k_oob"),
                (5, "pass1_nan_rho"),
                (6, "pass1_neigh_visits(lid0)"),
                (7, "pass1_cutoff_hits(lid0)"),
                (8, "pass2_cid_oob"),
                (9, "pass2_cs_oob"),
                (10, "pass2_cs_inverted"),
                (11, "pass2_bb_empty"),
                (12, "pass2_atom_k_oob"),
                (13, "pass2_nan_force"),
                (14, "pass2_neigh_visits(lid0)"),
                (15, "pass2_cutoff_hits(lid0)"),
                (16, "pass2_sorted_atom_oob"),
                (17, "pass1_all_pair_visits"),
                (18, "pass1_all_cutoff_hits"),
                (19, "pass1_real_cells_walked"),
                (20, "pass1_padding_cells_walked"),
                (21, "pass2_all_pair_visits"),
                (22, "pass2_all_cutoff_hits"),
                (23, "pass2_real_cells_walked"),
                (24, "pass2_padding_cells_walked"),
            ];
            for &(i, lbl) in labels {
                if dbg_vals[i] != 0 {
                    println!("   [{i:2}] {lbl:<32} = {:>12}", dbg_vals[i]);
                }
            }

            // ── Derived sanity checks ────────────────────────────────────────
            let p1_pair_visits = dbg_vals[17] as u64;
            let p1_cutoff_hits = dbg_vals[18] as u64;
            let p1_real_cells = dbg_vals[19] as u64;
            let p1_pad_cells = dbg_vals[20] as u64;
            let p2_pair_visits = dbg_vals[21] as u64;
            let p2_cutoff_hits = dbg_vals[22] as u64;
            let p2_real_cells = dbg_vals[23] as u64;
            let p2_pad_cells = dbg_vals[24] as u64;

            println!("Sanity checks:");
            println!(
                "   pass1 vs pass2 pair visits:   {} vs {}   {}",
                p1_pair_visits,
                p2_pair_visits,
                if p1_pair_visits == p2_pair_visits {
                    "[match]"
                } else {
                    "[DIFFER]"
                }
            );
            println!(
                "   pass1 vs pass2 cutoff hits:   {} vs {}   {}",
                p1_cutoff_hits,
                p2_cutoff_hits,
                if p1_cutoff_hits == p2_cutoff_hits {
                    "[match]"
                } else {
                    "[DIFFER]"
                }
            );
            println!(
                "   pass1 real/padding cells:     {} / {}",
                p1_real_cells, p1_pad_cells
            );
            println!(
                "   pass2 real/padding cells:     {} / {}",
                p2_real_cells, p2_pad_cells
            );

            // Expected comparison: AllPairs-equivalent pair count.  Each atom
            // scans n_atoms-1 candidates in AllPairs, so total visits = N*(N-1).
            // CellList should see at most that many (each pair visited at most
            // twice — once from each endpoint), but typically fewer since
            // the 27-stencil culls distant cells.
            let n_u = n as u64;
            let ap_equiv = n_u * (n_u - 1);
            let ratio = (p1_pair_visits as f64) / (ap_equiv as f64);
            println!(
                "   pair visits vs AllPairs-equiv ({}*(N-1)={}):  ratio = {:.4}",
                n_u, ap_equiv, ratio
            );
            if ratio > 1.0001 {
                println!("   → CL walks MORE pairs than AllPairs would.  Over-visit detected.");
            } else if ratio < 0.9999 {
                println!("   → CL walks FEWER pairs than AllPairs would.  Under-visit detected.");
            } else {
                println!("   → CL pair count matches AllPairs total (expected for small systems).");
            }

            // cells per WG.  For a power-of-2 grid, the 27-stencil hits at most
            // 27 real cells.  For non-p2 grids, the stencil might hit padding.
            let n_wgs = (n_u + 63) / 64;
            if n_wgs > 0 {
                let cells_per_wg = (p1_real_cells + p1_pad_cells) as f64 / n_wgs as f64;
                println!(
                    "   cells iterated per WG (avg): {:.2}   (real {:.2}, padding {:.2})",
                    cells_per_wg,
                    p1_real_cells as f64 / n_wgs as f64,
                    p1_pad_cells as f64 / n_wgs as f64,
                );
            }
        } else {
            println!("Shader counters:  all zero  (CREAM_ENABLE_DEBUG not honored? wgpu version?)");
        }
    }

    // ── 3. Energy and force comparison ───────────────────────────────────────
    let energy_diff = cl.energy - ap.energy;
    let energy_rel = if ap.energy.abs() > 1e-10 {
        (energy_diff / ap.energy).abs()
    } else {
        0.0
    };
    let (f_l2, f_max, worst) = force_diff_stats(&ap.forces, &cl.forces);
    let n_mismatched = count_mismatched_atoms(&ap.forces, &cl.forces, NOISE_FLOOR);

    let pass12_ok = f_max < NOISE_FLOOR;

    if verbose {
        println!(
            "Energy:      AP = {:+.4e}   CL = {:+.4e}   Δ = {:+.3e}  ({:.2e} rel)",
            ap.energy, cl.energy, energy_diff, energy_rel
        );
        println!(
            "Force diff:  L2 = {:.3e} eV/Å   max = {:.3e} eV/Å @ atom {}   mismatched atoms = {}/{}",
            f_l2, f_max, worst, n_mismatched, n
        );

        // top offenders only when pass0 looks good but physics is broken, so
        // we know cell_id / cell_start are meaningful.
        if pass0_ok && !pass12_ok {
            let top =
                top_offenders_forces(&cl.forces, &ap.forces, &dbg.cell_ids, &dbg.cell_start, 10);
            if !top.is_empty() {
                println!(
                    "Top offenders (|Δ| > {:.0e})  — {{atom, |Δ|, morton_cell, atoms_in_cell}}:",
                    NOISE_FLOOR
                );
                println!(
                    "  {:>6} {:>12} {:>10} {:>5}    |CL|        |AP|",
                    "atom", "|Δ|", "cell_id", "c_sz"
                );
                for (i, dmax, cid, csz) in top {
                    let f_cl = cl.forces[i];
                    let f_ap = ap.forces[i];
                    let cl_mag = (f_cl[0] * f_cl[0] + f_cl[1] * f_cl[1] + f_cl[2] * f_cl[2]).sqrt();
                    let ap_mag = (f_ap[0] * f_ap[0] + f_ap[1] * f_ap[1] + f_ap[2] * f_ap[2]).sqrt();
                    println!(
                        "  {:>6} {:>12.3e} {:>10} {:>5}  {:>10.4e}  {:>10.4e}",
                        i, dmax, cid, csz, cl_mag, ap_mag
                    );
                }
            }
        }

        let verdict = match (pass0_ok, pass12_ok) {
            (true, true) => Verdict::Ok,
            (true, false) => Verdict::Pass12Broken,
            (false, true) => Verdict::Pass0Broken,
            (false, false) => Verdict::Pass0AndPass12Broken,
        };
        println!("\nVerdict: {}", verdict.as_str());
        match verdict {
            Verdict::Pass12Broken => {
                println!("  → Cell list is correctly built; bug is in eam_pass{{1,2}}_cellist.wgsl")
            }
            Verdict::Pass0Broken | Verdict::Pass0AndPass12Broken => println!(
                "  → Cell list build (pass0a/0b/0c/0d) produces wrong output; fix that first."
            ),
            Verdict::Ok => {}
        }
    }

    DiagOutcome {
        pass0_ok,
        pass12_ok,
        energy_diff,
        force_max: f_max,
        force_l2: f_l2,
        n_cells: [dbg.n_cells.0, dbg.n_cells.1, dbg.n_cells.2],
        cid_mismatches: cid_mismatches.len(),
        rattle,
        cell_size_ratio,
        n,
    }
}

// ── Summary table printer ────────────────────────────────────────────────────

fn print_summary(outcomes: &[DiagOutcome]) {
    println!("\n{}", "═".repeat(104));
    println!("   SUMMARY");
    println!("{}", "═".repeat(104));
    println!(
        "{:>6} {:>7} {:>6} {:>12} {:>12} {:>10} {:>14} {:>8} {:>8}",
        "N", "rattle", "cs/rc", "n_cells", "cid_bad", "|ΔE|", "max|Δforce|", "pass-0", "pass-12"
    );
    println!("{}", "─".repeat(104));
    for o in outcomes {
        let nc = format!("{:?}", o.n_cells);
        println!(
            "{:>6} {:>7} {:>6.2} {:>12} {:>12} {:>10.3e} {:>14.3e} {:>8} {:>8}",
            o.n,
            if o.rattle { "yes" } else { "no" },
            o.cell_size_ratio,
            nc,
            o.cid_mismatches,
            o.energy_diff.abs(),
            o.force_max,
            if o.pass0_ok { "OK" } else { "FAIL" },
            if o.pass12_ok { "OK" } else { "FAIL" },
        );
    }
    println!("{}", "═".repeat(104));
    let _ = |o: &DiagOutcome| o.force_l2; // keep field used
}

fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Warn)
        .init();

    // Auto-enable the shader `ENABLE_DEBUG` pipeline override so we get real
    // counter data out of dbg[*].  Engine reads this env var in
    // `build_explicit_pipeline`.  Users running the binary directly don't
    // need to set it themselves.
    std::env::set_var("CREAM_ENABLE_DEBUG", "1");

    let pot = synth_cu_potential();

    println!(
        "\n╔══════════════════════════════════════════════════════════════════════════════╗\n\
           ║   CellList GPU diagnostic v2 · eam_pass{{1,2}}_cellist.wgsl verification      ║\n\
           ║                                                                              ║\n\
           ║   Compares GPU CellList against GPU AllPairs (same shader-side arithmetic). ║\n\
           ║   Sweeps (N, rattle, cell_size / cutoff) to localise the bug.               ║\n\
           ╚══════════════════════════════════════════════════════════════════════════════╝"
    );
    pollster::block_on(print_adapter_info());

    let mut all_outcomes: Vec<DiagOutcome> = Vec::new();

    // ── Primary sweep — focused on PASS/FAIL comparison ─────────────────────
    // N=256 (power-of-2 grid) vs N=500 (non-power-of-2) is the clearest
    // PASS/FAIL boundary.  We compare their shader counters to localise the
    // bug.  All runs are verbose so counter dumps appear.
    println!("\n### Primary sweep: PASS/FAIL comparison with dbg[*] instrumentation ###");
    let primary: &[(usize, bool)] = &[
        (3, false), // N=108  PASS   n_cells=(2,2,2)=n_pad
        (3, true),  // N=108  PASS
        (4, false), // N=256  PASS   n_cells=(2,2,2)=n_pad
        (4, true),  // N=256  PASS
        (5, false), // N=500  FAIL   n_cells=(3,3,3)≠n_pad(4,4,4)
        (5, true),  // N=500  FAIL
        (6, false), // N=864  PASS   n_cells=(4,4,4)=n_pad
        (6, true),  // N=864  PASS
        (7, false), // N=1372 FAIL   n_cells=(5,5,5)≠n_pad(8,8,8)
        (7, true),  // N=1372 FAIL
    ];
    for &(rep, rattle) in primary {
        let outcome = pollster::block_on(diagnose(rep, rattle, 1.0, &pot, true));
        all_outcomes.push(outcome);
    }

    // ── Final summary ────────────────────────────────────────────────────────
    print_summary(&all_outcomes);

    println!(
        "\n── Done ──────────────────────────────────────────────────────────────────────\n\
         Re-run on a specific backend / GPU (Windows PowerShell):\n  \
         $env:WGPU_BACKEND=\"dx12\";     $env:WGPU_ADAPTER_NAME=\"NVIDIA\"; cargo run --release --features cellist_gpu --bin diagnose_cellist\n  \
         $env:WGPU_BACKEND=\"vulkan\";   $env:WGPU_ADAPTER_NAME=\"NVIDIA\"; cargo run --release --features cellist_gpu --bin diagnose_cellist\n  \
         $env:WGPU_BACKEND=\"dx12\";     $env:WGPU_ADAPTER_NAME=\"Radeon\"; cargo run --release --features cellist_gpu --bin diagnose_cellist\n  \
         $env:WGPU_BACKEND=\"vulkan\";   $env:WGPU_ADAPTER_NAME=\"Radeon\"; cargo run --release --features cellist_gpu --bin diagnose_cellist\n"
    );
}
