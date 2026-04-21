//! Benchmark: Mishin Ni-Al 2009 potential — B2 NiAl and L1₂ Ni₃Al supercells.
//!
//! Measures reference (f64 serial), rayon half-pair, rayon cell-list, and
//! (with `cellist_gpu` feature) GPU CellList across a range of system sizes.
//! Results are directly comparable to the Cu benchmark in bench_cpu.rs.
use cream::{potential::eam::EamPotential, reference::compute_eam_cpu, CpuEngine};
#[cfg(feature = "cellist_gpu")]
use cream::{ComputeEngine, NeighborStrategy};
use std::{
    path::Path,
    time::{Duration, Instant},
};

/// `(pos_f32×4, pos_f64×3, atom_types, box_length_Å)`
type Supercell = (Vec<[f32; 4]>, Vec<[f64; 3]>, Vec<u32>, f64);

fn load_nial() -> EamPotential {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("Mishin-Ni-Al-2009_eam.alloy");
    EamPotential::from_file(&path).expect("failed to parse Mishin-Ni-Al-2009_eam.alloy")
}

fn ortho_f32(l: f32) -> [[f32; 3]; 3] {
    [[l, 0., 0.], [0., l, 0.], [0., 0., l]]
}
fn ortho_f64(l: f64) -> [[f64; 3]; 3] {
    [[l, 0., 0.], [0., l, 0.], [0., 0., l]]
}

// B2 NiAl: a=2.88 Å, 2 atoms per cell.
fn b2_supercell(a: f64, nx: usize) -> Supercell {
    let mut pos4 = Vec::new();
    let mut pos3 = Vec::new();
    let mut types = Vec::new();
    for ix in 0..nx {
        for iy in 0..nx {
            for iz in 0..nx {
                let (ox, oy, oz) = (ix as f64 * a, iy as f64 * a, iz as f64 * a);
                pos4.push([ox as f32, oy as f32, oz as f32, 0.]);
                pos3.push([ox, oy, oz]);
                types.push(0u32); // Ni
                pos4.push([
                    (ox + a / 2.) as f32,
                    (oy + a / 2.) as f32,
                    (oz + a / 2.) as f32,
                    0.,
                ]);
                pos3.push([ox + a / 2., oy + a / 2., oz + a / 2.]);
                types.push(1u32); // Al
            }
        }
    }
    (pos4, pos3, types, nx as f64 * a)
}

// L1₂ Ni₃Al: a=3.57 Å, 4 atoms per cell.
fn l12_supercell(a: f64, nx: usize) -> Supercell {
    let basis: [([f64; 3], u32); 4] = [
        ([0., 0., 0.], 1),         // Al
        ([a / 2., a / 2., 0.], 0), // Ni
        ([a / 2., 0., a / 2.], 0), // Ni
        ([0., a / 2., a / 2.], 0), // Ni
    ];
    let mut pos4 = Vec::new();
    let mut pos3 = Vec::new();
    let mut types = Vec::new();
    for ix in 0..nx {
        for iy in 0..nx {
            for iz in 0..nx {
                for (b, t) in &basis {
                    let p = [
                        b[0] + ix as f64 * a,
                        b[1] + iy as f64 * a,
                        b[2] + iz as f64 * a,
                    ];
                    pos4.push([p[0] as f32, p[1] as f32, p[2] as f32, 0.]);
                    pos3.push(p);
                    types.push(*t);
                }
            }
        }
    }
    (pos4, pos3, types, nx as f64 * a)
}

fn bench<F: Fn() -> f64>(label: &str, n: usize, warmup: usize, runs: usize, f: F) -> Duration {
    for _ in 0..warmup {
        let _ = f();
    }
    let mut times: Vec<Duration> = (0..runs)
        .map(|_| {
            let t = Instant::now();
            let _ = f();
            t.elapsed()
        })
        .collect();
    times.sort();
    let median = times[runs / 2];
    let min = *times.first().unwrap();
    let mean = times.iter().sum::<Duration>() / runs as u32;
    println!(
        "  {label:<26} N={n:>5}  median={:>8.2}ms  min={:>8.2}ms  mean={:>8.2}ms",
        median.as_secs_f64() * 1e3,
        min.as_secs_f64() * 1e3,
        mean.as_secs_f64() * 1e3
    );
    median
}

fn run_suite(
    label: &str,
    supercells: &[(usize, usize, usize, usize)], // (nx, warmup, runs, n_atoms)
    builder: &dyn Fn(usize) -> Supercell,
    pot: &EamPotential,
    eng: &CpuEngine,
) {
    println!("\n=== {label} ===");
    for &(nx, warmup, runs, _) in supercells {
        let (pos4, pos3, types, l) = builder(nx);
        let n = pos4.len();
        println!("[nx={nx}  N={n}  L={l:.2} Å]");
        let cell_f32 = Some(ortho_f32(l as f32));
        let cell_f64 = Some(ortho_f64(l));

        let t_ref = if n <= 1000 {
            bench("reference (serial)  ", n, warmup, runs, || {
                compute_eam_cpu(pot, &pos3, &types, cell_f64).energy
            })
        } else {
            println!(
                "  {:<26} N={:>5}  (skipped — O(N²) too slow)",
                "reference (serial)  ", n
            );
            Duration::MAX
        };
        let t_hp = bench("rayon half-pair     ", n, warmup, runs, || {
            eng.compute_sync(&pos4, &types, cell_f32, pot)
                .unwrap()
                .energy as f64
        });
        let t_cl = bench("rayon cell-list     ", n, warmup, runs, || {
            eng.compute_cell_list_sync(&pos4, &types, cell_f32, pot)
                .unwrap()
                .energy as f64
        });

        #[cfg(feature = "cellist_gpu")]
        let t_gpu_cl = {
            let mut gpu = pollster::block_on(ComputeEngine::new(NeighborStrategy::CellList {
                cell_size: pot.cutoff_angstrom,
            }))
            .expect("GPU init failed");
            // Warm up
            for _ in 0..warmup {
                let _ = gpu
                    .compute_sync(&pos4, &types, cell_f32, pot)
                    .expect("GPU compute failed (warmup)");
            }
            let mut times: Vec<Duration> = (0..runs)
                .map(|_| {
                    let t = Instant::now();
                    let _ = gpu
                        .compute_sync(&pos4, &types, cell_f32, pot)
                        .expect("GPU compute failed");
                    t.elapsed()
                })
                .collect();
            times.sort();
            let median = times[runs / 2];
            let min = *times.first().unwrap();
            let mean = times.iter().sum::<Duration>() / runs as u32;
            println!(
                "  {:<26} N={:>5}  median={:>8.2}ms  min={:>8.2}ms  mean={:>8.2}ms",
                "gpu cell-list       ",
                n,
                median.as_secs_f64() * 1e3,
                min.as_secs_f64() * 1e3,
                mean.as_secs_f64() * 1e3
            );
            median
        };

        if t_ref != Duration::MAX {
            println!(
                "  => ref→half-pair: {:.2}x  ref→cell-list: {:.2}x  half-pair→cell-list: {:.2}x",
                t_ref.as_secs_f64() / t_hp.as_secs_f64(),
                t_ref.as_secs_f64() / t_cl.as_secs_f64(),
                t_hp.as_secs_f64() / t_cl.as_secs_f64()
            );
        } else {
            println!(
                "  => half-pair→cell-list: {:.2}x",
                t_hp.as_secs_f64() / t_cl.as_secs_f64()
            );
        }
        #[cfg(feature = "cellist_gpu")]
        {
            println!(
                "  => cell-list(cpu)→gpu: {:.2}x",
                t_cl.as_secs_f64() / t_gpu_cl.as_secs_f64()
            );
        }
        println!();
    }
}

fn main() {
    let pot = load_nial();
    let eng = CpuEngine::new();
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    println!(
        "=== CREAM Ni-Al Benchmark (Mishin 2009, cutoff={:.3} Å) ===",
        pot.cutoff_angstrom
    );
    println!("Available HW threads: {n_threads}");
    println!("Elements: {:?}", pot.elements);

    // B2 NiAl: a=2.88 Å, 2 atoms/cell. L > 2*6.287=12.574 Å → nx≥5 (L=14.4 Å)
    // (nx, warmup, runs, n_atoms)
    let b2_configs: &[(usize, usize, usize, usize)] = &[
        (5, 1, 6, 250),   // 5³×2 = 250
        (6, 1, 5, 432),   // 6³×2 = 432
        (8, 1, 4, 1024),  // 8³×2 = 1024
        (10, 1, 3, 2000), // 10³×2 = 2000
        (13, 1, 3, 4394), // 13³×2 ≈ 4394
    ];
    run_suite(
        "B2 NiAl (a=2.88 Å, 50% Ni / 50% Al)",
        b2_configs,
        &|nx| b2_supercell(2.88, nx),
        &pot,
        &eng,
    );

    // L1₂ Ni₃Al: a=3.57 Å, 4 atoms/cell. nx≥4 → L=14.28 Å
    let l12_configs: &[(usize, usize, usize, usize)] = &[
        (4, 1, 6, 256),   // 4³×4 = 256
        (5, 1, 5, 500),   // 5³×4 = 500
        (6, 1, 4, 864),   // 6³×4 = 864
        (8, 1, 3, 2048),  // 8³×4 = 2048
        (10, 1, 3, 4000), // 10³×4 = 4000
    ];
    run_suite(
        "L1₂ Ni₃Al (a=3.57 Å, 75% Ni / 25% Al)",
        l12_configs,
        &|nx| l12_supercell(3.57, nx),
        &pot,
        &eng,
    );
}
