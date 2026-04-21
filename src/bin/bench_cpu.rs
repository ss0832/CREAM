//! Benchmark: rayon CpuEngine vs single-thread reference
use cream::{potential::eam::EamPotential, reference::compute_eam_cpu, CpuEngine};
use std::fmt::Write as _;
use std::time::{Duration, Instant};

/// Return type of `fcc_supercell`: (pos f32×4, pos f64×3, types, cell f32, cell f64).
type FccSupercell = (
    Vec<[f32; 4]>,
    Vec<[f64; 3]>,
    Vec<u32>,
    [[f32; 3]; 3],
    [[f64; 3]; 3],
);

#[inline]
fn ortho_cell(l: f32) -> [[f32; 3]; 3] {
    [[l, 0.0, 0.0], [0.0, l, 0.0], [0.0, 0.0, l]]
}

fn synthetic_pot_src(nr: u32, nrho: u32, dr: f32, drho: f32, cutoff: f32) -> String {
    let mut s = String::new();
    writeln!(s, "# Synthetic Cu").unwrap();
    writeln!(s, "# bench").unwrap();
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
        write!(s, "{:.8e} ", v).unwrap();
        if (i + 1) % 5 == 0 {
            writeln!(s).unwrap();
        }
    }
    writeln!(s).unwrap();
    for i in 0..nr {
        let r = i as f32 * dr;
        let v = if r < cutoff {
            let t = cutoff - r;
            r * t * t
        } else {
            0.0
        };
        write!(s, "{:.8e} ", v).unwrap();
        if (i + 1) % 5 == 0 {
            writeln!(s).unwrap();
        }
    }
    writeln!(s).unwrap();
    s
}

fn make_pot() -> EamPotential {
    EamPotential::from_str(&synthetic_pot_src(200, 200, 0.05, 0.01, 4.5)).unwrap()
}

fn fcc_supercell(n_rep: usize) -> FccSupercell {
    let a = 3.615f32;
    let basis = [
        [0.0f32, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ];
    let mut pos4 = Vec::new();
    let mut pos3_f64 = Vec::new();
    for ix in 0..n_rep {
        for iy in 0..n_rep {
            for iz in 0..n_rep {
                for b in &basis {
                    let x = (ix as f32 + b[0]) * a;
                    let y = (iy as f32 + b[1]) * a;
                    let z = (iz as f32 + b[2]) * a;
                    pos4.push([x, y, z, 0.0]);
                    pos3_f64.push([x as f64, y as f64, z as f64]);
                }
            }
        }
    }
    let l = n_rep as f32 * a;
    let types = vec![0u32; pos4.len()];
    let cell_f32 = ortho_cell(l);
    let cell_f64 = [
        [l as f64, 0.0, 0.0],
        [0.0, l as f64, 0.0],
        [0.0, 0.0, l as f64],
    ];
    (pos4, pos3_f64, types, cell_f32, cell_f64)
}

fn bench<F: Fn() -> f64>(label: &str, n: usize, warmup: usize, runs: usize, f: F) -> Duration {
    for _ in 0..warmup {
        let _ = f();
    }
    let mut times = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t0 = Instant::now();
        let _ = f();
        times.push(t0.elapsed());
    }
    times.sort();
    let median = times[runs / 2];
    let min = *times.first().unwrap();
    let mean: Duration = times.iter().sum::<Duration>() / runs as u32;
    println!(
        "  {label:<24} N={n:>5}  median={:>8.2}ms  min={:>8.2}ms  mean={:>8.2}ms",
        median.as_secs_f64() * 1e3,
        min.as_secs_f64() * 1e3,
        mean.as_secs_f64() * 1e3,
    );
    median
}

fn main() {
    let pot = make_pot();
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    println!("=== CREAM CPU Benchmark: reference / half-pair rayon / cell-list rayon ===");
    println!("Available HW threads : {n_threads}");
    println!();

    // n_rep=1 (3.615 Å) and n_rep=2 (7.23 Å) are both smaller than
    // 2×cutoff = 9.0 Å required by PBC, so they would panic in
    // reference::compute_eam_cpu.  Start from n_rep=3 (10.845 Å).
    // Larger sizes (n_rep>=7) skipped for reference (O(N²) too slow).
    let configs: &[(usize, usize, usize)] = &[
        (3, 2, 8),
        (4, 1, 6),
        (5, 1, 5),
        (6, 1, 4),
        (8, 1, 3),  // N=2048 — cell-list O(N) advantage becomes visible
        (10, 1, 3), // N=4000
    ];

    for &(n_rep, warmup, runs) in configs {
        let (pos4, pos3_f64, types, cell_f32, cell_f64) = fcc_supercell(n_rep);
        let n = pos4.len();
        println!("[n_rep={n_rep}  N={n}]");
        let cpu_eng = CpuEngine::new();

        let t_ref = if n <= 864 {
            bench("reference (serial)  ", n, warmup, runs, || {
                compute_eam_cpu(&pot, &pos3_f64, &types, Some(cell_f64)).energy
            })
        } else {
            println!(
                "  {:<24} N={:>5}  (skipped — O(N²) too slow)",
                "reference (serial)  ", n
            );
            std::time::Duration::MAX
        };
        let t_ray = bench("rayon half-pair     ", n, warmup, runs, || {
            cpu_eng
                .compute_sync(&pos4, &types, Some(cell_f32), &pot)
                .unwrap()
                .energy as f64
        });
        let t_cl = bench("rayon cell-list     ", n, warmup, runs, || {
            cpu_eng
                .compute_cell_list_sync(&pos4, &types, Some(cell_f32), &pot)
                .unwrap()
                .energy as f64
        });

        if t_ref != std::time::Duration::MAX {
            let speedup_hp = t_ref.as_secs_f64() / t_ray.as_secs_f64();
            let speedup_cl = t_ref.as_secs_f64() / t_cl.as_secs_f64();
            let cl_vs_hp = t_ray.as_secs_f64() / t_cl.as_secs_f64();
            println!("  => ref→half-pair: {speedup_hp:.2}x  ref→cell-list: {speedup_cl:.2}x  half-pair→cell-list: {cl_vs_hp:.2}x\n");
        } else {
            let cl_vs_hp = t_ray.as_secs_f64() / t_cl.as_secs_f64();
            println!("  => half-pair→cell-list: {cl_vs_hp:.2}x\n");
        }
    }
}
