//! Minimal reproducer for the CPU-CellList vs CPU-AllPairs bug at rep ≥ 35.
//!
//! Goal: find the smallest rep (= smallest N, fastest run) that triggers the
//! bug, so we can iterate on fixes in seconds, not minutes.
//!
//! We copy exactly the supercell + Gaussian rattle from `diagnose_mismatch.rs`.

#![allow(clippy::too_many_arguments)]

use cream::potential::eam::EamPotential;
use cream::CpuEngine;

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

fn build_supercell_cubic(
    rep: usize,
    rattle: f32,
    seed: u64,
) -> (Vec<[f32; 4]>, Vec<u32>, [[f32; 3]; 3]) {
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
    let mut state: u64 = seed;
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
                    let x = b[0] + ix as f32 * a;
                    let y = b[1] + iy as f32 * a;
                    let z = b[2] + iz as f32 * a;
                    pos.push([x, y, z, 0.0]);
                    types.push(0u32);
                }
            }
        }
    }
    for p in pos.iter_mut() {
        let (dx, dy) = normal_pair(&mut state);
        let (dz, _) = normal_pair(&mut state);
        p[0] += rattle * dx;
        p[1] += rattle * dy;
        p[2] += rattle * dz;
    }
    let l = a * rep as f32;
    let cell = [[l, 0.0, 0.0], [0.0, l, 0.0], [0.0, 0.0, l]];
    (pos, types, cell)
}

fn diff_stats(a: &[[f32; 3]], b: &[[f32; 3]]) -> (f32, usize) {
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
}

fn main() {
    let pot = synth_cu_potential();
    let cpu = CpuEngine::new();

    // Sweep rep sizes to find the smallest one that triggers the bug.
    // rep=8 → N=2048, rep=12 → N=6912, rep=16 → N=16384, rep=20 → N=32000, rep=24 → N=55296
    // rep=30 → N=108000 (anchor passes), rep=35 → N=171500 (fails)
    println!(
        "{:>6} {:>10} {:>10} {:>15}",
        "rep", "N", "ncx_p2", "max|Δf|"
    );
    for &rep in &[8usize, 12, 16, 20, 24, 28, 30, 32, 33, 34, 35] {
        let (pos, types, cell) = build_supercell_cubic(rep, 0.05, 42);
        let n = pos.len();
        let l = cell[0][0];
        let cutoff = pot.cutoff_angstrom;
        let ncx_raw = (l / cutoff).floor() as u32;
        let ncx_p2 = if ncx_raw == 0 {
            1
        } else {
            1u32 << (31 - ncx_raw.leading_zeros())
        };

        let r_ap = match cpu.compute_sync(&pos, &types, Some(cell), &pot) {
            Ok(r) => r,
            Err(e) => {
                println!("{rep:>6} {n:>10} {ncx_p2:>10} FAILED: {e:?}");
                continue;
            }
        };
        let r_cl = match cpu.compute_cell_list_sync(&pos, &types, Some(cell), &pot) {
            Ok(r) => r,
            Err(e) => {
                println!("{rep:>6} {n:>10} {ncx_p2:>10} FAILED: {e:?}");
                continue;
            }
        };
        let (m, mi) = diff_stats(&r_ap.forces, &r_cl.forces);
        let flag = if m > 1e-3 { "🚨" } else { "  " };
        println!("{rep:>6} {n:>10} {ncx_p2:>10} {m:>15.4e}  [atom {mi}] {flag}");
    }
}
