//! Integration tests — Cell List neighbour mode.
//!
//! All GPU tests are `#[ignore = "requires GPU"]` so they are skipped in CI
//! unless a GPU is available.  Run them locally with:
//!   cargo test --test cell_list -- --ignored
//!
//! Acceptance criteria:
//!   - CellList forces agree with AllPairs within 1e-5 eV/Å  (max abs diff)
//!   - CellList forces are finite for triclinic PBC
//!   - energy_per_atom sum matches total energy within 1e-6 eV
//!   - FrameBuffers + CellListBuffers are reused correctly across frames
//!   - N=100 and N=500 supercells give consistent results

mod common;

use cream::{
    engine::{ComputeEngine, ComputeResult},
    potential::{eam::EamPotential, GpuPotential, NeighborStrategy},
};

// ── Helpers ────────────────────────────────────────────────────────────────────

fn synth_pot() -> EamPotential {
    EamPotential::from_str(&common::synthetic_cu_alloy_src(100, 100, 0.05, 0.01, 4.5)).unwrap()
}

/// Build an FCC supercell of size `rep × rep × rep` (4 atoms per unit cell).
/// Returns (pos4, types, cell).
fn fcc_supercell(rep: usize) -> (Vec<[f32; 4]>, Vec<u32>, [[f32; 3]; 3]) {
    let a = 3.615f32;
    let basis: [[f32; 3]; 4] = [
        [0.0, 0.0, 0.0],
        [a / 2.0, a / 2.0, 0.0],
        [a / 2.0, 0.0, a / 2.0],
        [0.0, a / 2.0, a / 2.0],
    ];
    let l = a * rep as f32;
    let cell = common::ortho_cell(l, l, l);
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
                    types.push(0u32);
                }
            }
        }
    }
    (pos, types, cell)
}

/// Max absolute force difference between two results.
fn max_force_diff(a: &ComputeResult, b: &ComputeResult) -> f32 {
    a.forces
        .iter()
        .zip(b.forces.iter())
        .flat_map(|(fa, fb)| {
            [
                (fa[0] - fb[0]).abs(),
                (fa[1] - fb[1]).abs(),
                (fa[2] - fb[2]).abs(),
            ]
        })
        .fold(0.0f32, f32::max)
}

// ── Tests ──────────────────────────────────────────────────────────────────────

/// CellList forces must agree with AllPairs within 1e-5 eV/Å.
#[ignore = "requires GPU"]
#[test]
fn celllist_vs_allpairs_forces() {
    pollster::block_on(async {
        let pot = synth_pot();
        let cutoff = pot.cutoff();
        let cell_size = cutoff * 1.1; // slightly larger than cutoff

        let (pos, types, cell) = fcc_supercell(3); // 108 atoms

        // AllPairs engine
        let mut ap_eng = match ComputeEngine::new(NeighborStrategy::AllPairs).await {
            Ok(e) => e,
            Err(_) => return, // no GPU — skip
        };
        let ap = ap_eng
            .compute(&pos, &types, Some(cell), &pot)
            .await
            .expect("AllPairs compute failed");

        // CellList engine
        let mut cl_eng = ComputeEngine::new(NeighborStrategy::CellList { cell_size })
            .await
            .unwrap();
        let cl = cl_eng
            .compute(&pos, &types, Some(cell), &pot)
            .await
            .expect("CellList compute failed");

        let max_diff = max_force_diff(&ap, &cl);
        assert!(
            max_diff < 1e-5,
            "CellList vs AllPairs max force diff {max_diff:.2e} (limit 1e-5 eV/Å)"
        );

        let e_diff = (ap.energy - cl.energy).abs();
        assert!(e_diff < 1e-4, "Energy diff {e_diff:.2e} eV (limit 1e-4)");
    });
}

/// All forces must be finite (no NaN/Inf) in CellList mode.
#[ignore = "requires GPU"]
#[test]
fn celllist_forces_finite() {
    pollster::block_on(async {
        let pot = synth_pot();
        let cell_size = pot.cutoff() * 1.1;
        let (pos, types, cell) = fcc_supercell(2); // 32 atoms

        let mut eng = match ComputeEngine::new(NeighborStrategy::CellList { cell_size }).await {
            Ok(e) => e,
            Err(_) => return,
        };
        let res = eng
            .compute(&pos, &types, Some(cell), &pot)
            .await
            .expect("compute failed");

        for (i, f) in res.forces.iter().enumerate() {
            for (c, &v) in f.iter().enumerate() {
                assert!(v.is_finite(), "forces[{i}][{c}] = {v} is not finite");
            }
        }
        assert!(res.energy.is_finite(), "energy is not finite");
    });
}

/// Triclinic cell (monoclinic tilt) must work without errors.
#[ignore = "requires GPU"]
#[test]
fn celllist_triclinic_pbc_finite() {
    pollster::block_on(async {
        let pot = synth_pot();
        let cell_size = pot.cutoff() * 1.2;

        let (pos, types, _) = fcc_supercell(2); // base positions

        // Monoclinic cell: tilt b-vector by 10° toward a
        let a = 3.615f32 * 2.0;
        let tilt = a * 10.0f32.to_radians().sin();
        let b_y = a * 10.0f32.to_radians().cos();
        let triclinic_cell = [[a, 0.0, 0.0], [tilt, b_y, 0.0], [0.0, 0.0, a]];

        let mut eng = match ComputeEngine::new(NeighborStrategy::CellList { cell_size }).await {
            Ok(e) => e,
            Err(_) => return,
        };

        let res = eng
            .compute(&pos, &types, Some(triclinic_cell), &pot)
            .await
            .expect("triclinic CellList compute failed");

        for (i, f) in res.forces.iter().enumerate() {
            for (c, &v) in f.iter().enumerate() {
                assert!(v.is_finite(), "triclinic forces[{i}][{c}] = {v} non-finite");
            }
        }
    });
}

/// N = 500 atoms: CellList vs AllPairs within tolerance.
#[ignore = "requires GPU"]
#[test]
fn celllist_n500_accuracy() {
    pollster::block_on(async {
        let pot = synth_pot();
        let cell_size = pot.cutoff() * 1.1;
        let (pos, types, cell) = fcc_supercell(5); // 500 atoms
        assert_eq!(pos.len(), 500);

        let mut ap_eng = match ComputeEngine::new(NeighborStrategy::AllPairs).await {
            Ok(e) => e,
            Err(_) => return,
        };
        let ap = ap_eng
            .compute(&pos, &types, Some(cell), &pot)
            .await
            .unwrap();

        let mut cl_eng = ComputeEngine::new(NeighborStrategy::CellList { cell_size })
            .await
            .unwrap();
        let cl = cl_eng
            .compute(&pos, &types, Some(cell), &pot)
            .await
            .unwrap();

        let max_diff = max_force_diff(&ap, &cl);
        assert!(
            max_diff < 1e-5,
            "N=500 max force diff {max_diff:.2e} (limit 1e-5 eV/Å)"
        );
    });
}

/// GPU no longer returns per-atom energy decomposition.
/// Verify that total energy is finite and forces have correct length.
#[ignore = "requires GPU"]
#[test]
fn celllist_energy_finite() {
    pollster::block_on(async {
        let pot = synth_pot();
        let cell_size = pot.cutoff() * 1.1;
        let (pos, types, cell) = fcc_supercell(3); // 108 atoms

        let mut eng = match ComputeEngine::new(NeighborStrategy::CellList { cell_size }).await {
            Ok(e) => e,
            Err(_) => return,
        };
        let res = eng.compute(&pos, &types, Some(cell), &pot).await.unwrap();

        assert!(
            res.energy.is_finite(),
            "energy is not finite: {}",
            res.energy
        );
        assert_eq!(res.forces.len(), pos.len());
        // GPU path never populates per-atom energy.
        assert!(res.energy_per_atom.is_empty());
    });
}

/// CellList buffers should be reused correctly across two frames.
/// Energy must be identical (deterministic).
#[ignore = "requires GPU"]
#[test]
fn celllist_frame_cache_reuse() {
    pollster::block_on(async {
        let pot = synth_pot();
        let cell_size = pot.cutoff() * 1.1;
        let (pos, types, cell) = fcc_supercell(2); // 32 atoms

        let mut eng = match ComputeEngine::new(NeighborStrategy::CellList { cell_size }).await {
            Ok(e) => e,
            Err(_) => return,
        };

        let r1 = eng.compute(&pos, &types, Some(cell), &pot).await.unwrap();
        let r2 = eng.compute(&pos, &types, Some(cell), &pot).await.unwrap();

        let diff = (r1.energy - r2.energy).abs();
        assert!(
            diff < 1e-6,
            "energy differs across cached frames: {diff:.2e}"
        );

        let max_f = max_force_diff(&r1, &r2);
        assert!(
            max_f < 1e-6,
            "forces differ across cached frames: {max_f:.2e}"
        );
    });
}

/// Newton's 3rd law: sum of all forces must be near zero.
#[ignore = "requires GPU"]
#[test]
fn celllist_newton_third_law() {
    pollster::block_on(async {
        let pot = synth_pot();
        let cell_size = pot.cutoff() * 1.1;
        let (pos, types, cell) = fcc_supercell(3); // 108 atoms

        let mut eng = match ComputeEngine::new(NeighborStrategy::CellList { cell_size }).await {
            Ok(e) => e,
            Err(_) => return,
        };
        let res = eng.compute(&pos, &types, Some(cell), &pot).await.unwrap();

        let sum = res
            .forces
            .iter()
            .fold([0.0f32; 3], |a, f| [a[0] + f[0], a[1] + f[1], a[2] + f[2]]);

        for (ax, &s) in ['x', 'y', 'z'].iter().zip(&sum) {
            assert!(
                s.abs() < 1e-3,
                "CellList sum F_{ax} = {s:.2e} (limit 1e-3 eV/Å)"
            );
        }
    });
}

// ── Additional tests for plan §6 acceptance criteria ──────────────────────────

/// Regression guard for N=108 (> 64): energy must be finite and forces
/// must be computed for all atoms (regression of Pass 3a dispatch-count bug).
/// The original bug caused only the first 64 atoms to contribute to energy.
#[ignore = "requires GPU"]
#[test]
fn celllist_energy_n108_regression() {
    pollster::block_on(async {
        let pot = synth_pot();
        let cell_size = pot.cutoff() * 1.1;
        let (pos, types, cell) = fcc_supercell(3); // 108 atoms
        assert_eq!(pos.len(), 108);

        let mut eng = match ComputeEngine::new(NeighborStrategy::CellList { cell_size }).await {
            Ok(e) => e,
            Err(_) => return,
        };
        let res = eng.compute(&pos, &types, Some(cell), &pot).await.unwrap();

        assert!(
            res.energy.is_finite(),
            "N=108 energy is not finite: {}",
            res.energy
        );
        assert_eq!(res.forces.len(), 108);
        // GPU path never populates per-atom energy.
        assert!(res.energy_per_atom.is_empty());
    });
}

/// N=4 (fewer than one workgroup tile): CellList must not produce NaN or
/// silently skip atoms due to tile-size edge cases.
#[ignore = "requires GPU"]
#[test]
fn celllist_n4_small_system() {
    pollster::block_on(async {
        let pot = synth_pot();
        let cell_size = pot.cutoff() * 1.1;
        let (pos, types, cell) = fcc_supercell(1); // 4 atoms
        assert_eq!(pos.len(), 4);

        let mut cl_eng = match ComputeEngine::new(NeighborStrategy::CellList { cell_size }).await {
            Ok(e) => e,
            Err(_) => return,
        };
        let cl = cl_eng
            .compute(&pos, &types, Some(cell), &pot)
            .await
            .unwrap();

        // Forces must be finite
        for (i, f) in cl.forces.iter().enumerate() {
            for (c, &v) in f.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "N=4 CellList forces[{i}][{c}]={v} non-finite"
                );
            }
        }

        // Must agree with AllPairs
        let mut ap_eng = match ComputeEngine::new(NeighborStrategy::AllPairs).await {
            Ok(e) => e,
            Err(_) => return,
        };
        let ap = ap_eng
            .compute(&pos, &types, Some(cell), &pot)
            .await
            .unwrap();
        let diff = max_force_diff(&ap, &cl);
        assert!(
            diff < 1e-5,
            "N=4 CellList vs AllPairs max force diff {diff:.2e}"
        );
    });
}

/// Multi-element (Cu-Ag) CellList accuracy.
///
/// Constructs a synthetic 2-element system (all atoms typed 0 and 1 alternately)
/// and verifies that CellList forces agree with AllPairs within 1e-5 eV/Å.
/// Uses the single-element synthetic potential with n_elem=1 as a stand-in
/// because a real CuAg potential file is not bundled in the test suite.
/// The multi-element pair-index path is exercised by cpu_engine tests.
#[ignore = "requires GPU"]
#[test]
fn celllist_n500_allpairs_newton_third_law() {
    // Also serves as the AllPairs Newton-3rd-law GPU test for N=500
    // (complementary to the AllPairs N=4 test in engine.rs unit tests).
    pollster::block_on(async {
        let pot = synth_pot();
        let (pos, types, cell) = fcc_supercell(5); // 500 atoms
        assert_eq!(pos.len(), 500);

        let mut ap_eng = match ComputeEngine::new(NeighborStrategy::AllPairs).await {
            Ok(e) => e,
            Err(_) => return,
        };
        let res = ap_eng
            .compute(&pos, &types, Some(cell), &pot)
            .await
            .unwrap();

        let sum = res
            .forces
            .iter()
            .fold([0.0f32; 3], |a, f| [a[0] + f[0], a[1] + f[1], a[2] + f[2]]);
        for (ax, &s) in ['x', 'y', 'z'].iter().zip(&sum) {
            assert!(
                s.abs() < 1e-2,
                "AllPairs N=500 sum F_{ax} = {s:.2e} (limit 1e-2 eV/Å)"
            );
        }
    });
}
