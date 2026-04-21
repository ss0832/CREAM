//! Ni-Al alloy potential diagnostic: verify 2-element EAM on GPU.
//!
//! Loads `Mishin-Ni-Al-2009_eam.alloy` (project root) and runs four
//! configurations:
//!   1. Pure Ni FCC (all types = 0)
//!   2. Pure Al FCC (all types = 1)
//!   3. B2 NiAl (50% Ni / 50% Al in CsCl structure)
//!   4. L1₂ Ni₃Al (75% Ni / 25% Al)
//!
//! For each configuration the tool runs:
//!   * CPU AllPairs reference (O(N²), small N only)
//!   * CPU CellList
//!   * GPU AllPairs
//!   * GPU CellList
//!
//! Reports per-pair force differences.  All four paths should agree to
//! ≤ 1e-3 eV/Å.
//!
//! # Build and run
//! ```bash
//! cargo run --release --features cellist_gpu --bin diagnose_nial
//! ```

#![allow(clippy::too_many_arguments)]

use cream::potential::eam::EamPotential;
use cream::{ComputeEngine, CpuEngine, NeighborStrategy};
use std::path::Path;

// ── Supercell builders ───────────────────────────────────────────────────────

/// Pure FCC of element `elem_type` with lattice constant `a`.
fn fcc_supercell(a: f32, rep: usize, elem_type: u32) -> (Vec<[f32; 4]>, Vec<u32>, [[f32; 3]; 3]) {
    let basis: [[f32; 3]; 4] = [
        [0.0, 0.0, 0.0],
        [a / 2.0, a / 2.0, 0.0],
        [a / 2.0, 0.0, a / 2.0],
        [0.0, a / 2.0, a / 2.0],
    ];
    let mut pos = Vec::new();
    let mut types = Vec::new();
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
                    types.push(elem_type);
                }
            }
        }
    }
    let l = a * rep as f32;
    let cell = [[l, 0.0, 0.0], [0.0, l, 0.0], [0.0, 0.0, l]];
    rattle(&mut pos, 0.05);
    (pos, types, cell)
}

/// B2 NiAl: CsCl structure, Ni at (0,0,0), Al at (½,½,½).
fn b2_supercell(a: f32, rep: usize) -> (Vec<[f32; 4]>, Vec<u32>, [[f32; 3]; 3]) {
    let mut pos = Vec::new();
    let mut types = Vec::new();
    for ix in 0..rep {
        for iy in 0..rep {
            for iz in 0..rep {
                let o = [ix as f32 * a, iy as f32 * a, iz as f32 * a];
                pos.push([o[0], o[1], o[2], 0.0]);
                types.push(0); // Ni
                pos.push([o[0] + a * 0.5, o[1] + a * 0.5, o[2] + a * 0.5, 0.0]);
                types.push(1); // Al
            }
        }
    }
    let l = a * rep as f32;
    let cell = [[l, 0.0, 0.0], [0.0, l, 0.0], [0.0, 0.0, l]];
    rattle(&mut pos, 0.05);
    (pos, types, cell)
}

/// L1₂ Ni₃Al: FCC with Al at corner, Ni at face centers.
fn l12_supercell(a: f32, rep: usize) -> (Vec<[f32; 4]>, Vec<u32>, [[f32; 3]; 3]) {
    let basis: [([f32; 3], u32); 4] = [
        ([0.0, 0.0, 0.0], 1),         // Al corner
        ([a * 0.5, a * 0.5, 0.0], 0), // Ni face
        ([a * 0.5, 0.0, a * 0.5], 0), // Ni face
        ([0.0, a * 0.5, a * 0.5], 0), // Ni face
    ];
    let mut pos = Vec::new();
    let mut types = Vec::new();
    for ix in 0..rep {
        for iy in 0..rep {
            for iz in 0..rep {
                for (b, t) in &basis {
                    pos.push([
                        b[0] + ix as f32 * a,
                        b[1] + iy as f32 * a,
                        b[2] + iz as f32 * a,
                        0.0,
                    ]);
                    types.push(*t);
                }
            }
        }
    }
    let l = a * rep as f32;
    let cell = [[l, 0.0, 0.0], [0.0, l, 0.0], [0.0, 0.0, l]];
    rattle(&mut pos, 0.05);
    (pos, types, cell)
}

/// Deterministic rattle (±0.05 Å) to break perfect symmetry.
fn rattle(pos: &mut [[f32; 4]], amp: f32) {
    let mut state: u32 = 0xDEAD_BEEF;
    let mut lcg = || -> f32 {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        (state as f32) / (u32::MAX as f32) - 0.5
    };
    for p in pos.iter_mut() {
        p[0] += amp * lcg();
        p[1] += amp * lcg();
        p[2] += amp * lcg();
    }
}

// ── Diagnostic ───────────────────────────────────────────────────────────────

async fn run_one(
    name: &str,
    pos: Vec<[f32; 4]>,
    types: Vec<u32>,
    cell: [[f32; 3]; 3],
    pot: &EamPotential,
) -> Result<(), Box<dyn std::error::Error>> {
    let n = pos.len();
    println!();
    println!("════════════════════════════════════════════════════════════════════════");
    println!("  {}  N = {}", name, n);
    println!("════════════════════════════════════════════════════════════════════════");

    // Element composition
    let mut n0 = 0usize;
    let mut n1 = 0usize;
    for &t in &types {
        if t == 0 {
            n0 += 1;
        } else {
            n1 += 1;
        }
    }
    println!(
        "  Composition: Ni = {} ({:.0}%)   Al = {} ({:.0}%)",
        n0,
        100.0 * n0 as f64 / n as f64,
        n1,
        100.0 * n1 as f64 / n as f64
    );

    // CPU AllPairs (ground truth for small N)
    let cpu = CpuEngine::new();
    let cpu_ap = if n <= 8000 {
        println!("[1/4] CPU AllPairs …");
        Some(cpu.compute_sync(&pos, &types, Some(cell), pot)?)
    } else {
        println!("[1/4] CPU AllPairs skipped (N > 8000)");
        None
    };

    // CPU CellList
    println!("[2/4] CPU CellList …");
    let cpu_cl = cpu.compute_cell_list_sync(&pos, &types, Some(cell), pot)?;

    // GPU AllPairs
    let gpu_ap = if n <= 500_000 {
        println!("[3/4] GPU AllPairs …");
        let mut eng = ComputeEngine::new(NeighborStrategy::AllPairs).await?;
        Some(eng.compute(&pos, &types, Some(cell), pot).await?)
    } else {
        println!("[3/4] GPU AllPairs skipped (N > 500k)");
        None
    };

    // GPU CellList
    println!("[4/4] GPU CellList …");
    let mut eng = ComputeEngine::new(NeighborStrategy::CellList {
        cell_size: pot.cutoff_angstrom,
    })
    .await?;
    let gpu_cl = eng.compute(&pos, &types, Some(cell), pot).await?;

    // Comparisons
    let max_diff = |a: &[[f32; 3]], b: &[[f32; 3]]| -> f32 {
        let mut m = 0.0f32;
        for i in 0..a.len() {
            for c in 0..3 {
                m = m.max((a[i][c] - b[i][c]).abs());
            }
        }
        m
    };

    println!();
    println!("Energy:");
    if let Some(ap) = cpu_ap.as_ref() {
        println!("  CPU AP: {:.6e} eV", ap.energy);
    }
    println!("  CPU CL: {:.6e} eV", cpu_cl.energy);
    if let Some(ap) = gpu_ap.as_ref() {
        println!("  GPU AP: {:.6e} eV", ap.energy);
    }
    println!("  GPU CL: {:.6e} eV", gpu_cl.energy);

    println!();
    println!("Max |Δf| (eV/Å):");
    if let Some(ap) = cpu_ap.as_ref() {
        println!(
            "  CPU-AP vs CPU-CL:  {:.4e}",
            max_diff(&ap.forces, &cpu_cl.forces)
        );
        if let Some(gap) = gpu_ap.as_ref() {
            println!(
                "  CPU-AP vs GPU-AP:  {:.4e}",
                max_diff(&ap.forces, &gap.forces)
            );
            println!(
                "  CPU-AP vs GPU-CL:  {:.4e}",
                max_diff(&ap.forces, &gpu_cl.forces)
            );
        }
    }
    if let Some(gap) = gpu_ap.as_ref() {
        println!(
            "  GPU-AP vs GPU-CL:  {:.4e}",
            max_diff(&gap.forces, &gpu_cl.forces)
        );
        println!(
            "  GPU-AP vs CPU-CL:  {:.4e}",
            max_diff(&gap.forces, &cpu_cl.forces)
        );
    }
    println!(
        "  CPU-CL vs GPU-CL:  {:.4e}",
        max_diff(&cpu_cl.forces, &gpu_cl.forces)
    );

    // Σf drift
    let sum_drift = |f: &[[f32; 3]]| -> f64 {
        let mut s = [0.0f64; 3];
        for v in f {
            for k in 0..3 {
                s[k] += v[k] as f64;
            }
        }
        (s[0].powi(2) + s[1].powi(2) + s[2].powi(2)).sqrt()
    };
    println!();
    println!("|Σf|:");
    if let Some(ap) = cpu_ap.as_ref() {
        println!("  CPU AP: {:.2e}", sum_drift(&ap.forces));
    }
    println!("  CPU CL: {:.2e}", sum_drift(&cpu_cl.forces));
    if let Some(ap) = gpu_ap.as_ref() {
        println!("  GPU AP: {:.2e}", sum_drift(&ap.forces));
    }
    println!("  GPU CL: {:.2e}", sum_drift(&gpu_cl.forces));

    Ok(())
}

fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Warn)
        .init();

    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("Mishin-Ni-Al-2009_eam.alloy");
    let pot = EamPotential::from_file(&path)
        .expect("failed to load Mishin-Ni-Al-2009_eam.alloy (place it in project root)");

    println!();
    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║   diagnose_nial · 2-element EAM alloy correctness test                 ║");
    println!("║                                                                        ║");
    println!("║   Potential: Mishin Ni-Al 2009                                         ║");
    println!("║   Elements: {:<58} ║", format!("{:?}", pot.elements));
    println!("║   Cutoff:   {:.3} Å {:<56} ║", pot.cutoff_angstrom, "");
    println!("╚════════════════════════════════════════════════════════════════════════╝");

    // Run 4 configurations at moderate size
    let cases: Vec<(
        &str,
        Box<dyn Fn() -> (Vec<[f32; 4]>, Vec<u32>, [[f32; 3]; 3])>,
    )> = vec![
        (
            "Pure Ni FCC (a=3.52 Å, rep=8, N=2048)",
            Box::new(|| fcc_supercell(3.52, 8, 0)),
        ),
        (
            "Pure Al FCC (a=4.05 Å, rep=8, N=2048)",
            Box::new(|| fcc_supercell(4.05, 8, 1)),
        ),
        (
            "B2 NiAl (a=2.88 Å, rep=10, N=2000)",
            Box::new(|| b2_supercell(2.88, 10)),
        ),
        (
            "L1₂ Ni₃Al (a=3.57 Å, rep=8, N=2048)",
            Box::new(|| l12_supercell(3.57, 8)),
        ),
    ];

    for (name, builder) in cases {
        let (pos, types, cell) = builder();
        if let Err(e) = pollster::block_on(run_one(name, pos, types, cell, &pot)) {
            eprintln!("Failed: {}", e);
        }
    }

    println!();
    println!("── Done ────────────────────────────────────────────────────────────────");
}
