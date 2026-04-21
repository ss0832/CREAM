//! Test with the real Cu01_eam.alloy potential from Mishin (2001).

use cream::potential::eam::EamPotential;
use cream::reference::{compute_eam_cpu, ortho_cell};

fn load_cu01() -> EamPotential {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("Cu01_eam.alloy");
    EamPotential::from_file(&path).expect("failed to parse Cu01_eam.alloy")
}

/// N×N×N FCC supercell (f64 positions, for use with compute_eam_cpu).
/// For PBC with Cu01 (cutoff=5.507), need N≥4 (L=14.46 > 11.01).
fn cu_fcc_supercell(a: f64, nx: usize) -> (Vec<[f64; 3]>, Vec<u32>) {
    let basis = [
        [0.0, 0.0, 0.0],
        [a / 2.0, a / 2.0, 0.0],
        [a / 2.0, 0.0, a / 2.0],
        [0.0, a / 2.0, a / 2.0],
    ];
    let mut pos = Vec::new();
    for ix in 0..nx {
        for iy in 0..nx {
            for iz in 0..nx {
                let off = [ix as f64 * a, iy as f64 * a, iz as f64 * a];
                for b in &basis {
                    pos.push([b[0] + off[0], b[1] + off[1], b[2] + off[2]]);
                }
            }
        }
    }
    let n = pos.len();
    (pos, vec![0u32; n])
}

#[test]
fn parse_real_cu01() {
    let pot = load_cu01();
    assert_eq!(pot.elements, vec!["Cu"]);
    assert_eq!(pot.nr, 10000);
    assert_eq!(pot.nrho, 10000);
    assert!((pot.cutoff_angstrom - 5.506786).abs() < 1e-4);
}

#[test]
fn cpu_2atom_newton_no_pbc() {
    let pot = load_cu01();
    let res = compute_eam_cpu(&pot, &[[0.0f64, 0.0, 0.0], [2.5, 0.0, 0.0]], &[0, 0], None);
    let sum_x = res.forces[0][0] + res.forces[1][0];
    assert!(sum_x.abs() < 1e-6, "Newton X: {sum_x}");
}

#[test]
fn cpu_fcc_4x4x4_equilibrium_energy() {
    let pot = load_cu01();
    let a = 3.615f64;
    let (pos, types) = cu_fcc_supercell(a, 4);
    let l = 4.0 * a;
    let res = compute_eam_cpu(&pot, &pos, &types, Some(ortho_cell(l, l, l)));
    let e_per_atom = res.energy / pos.len() as f64;
    assert!(
        (e_per_atom - (-3.54_f64)).abs() < 0.01,
        "energy/atom = {e_per_atom:.6}, expected ~-3.54 eV"
    );
}

#[test]
fn cpu_fcc_4x4x4_forces_zero() {
    let pot = load_cu01();
    let a = 3.615f64;
    let (pos, types) = cu_fcc_supercell(a, 4);
    let l = 4.0 * a;
    let res = compute_eam_cpu(&pot, &pos, &types, Some(ortho_cell(l, l, l)));
    let max_f: f64 = res
        .forces
        .iter()
        .flat_map(|f| f.iter())
        .map(|v| v.abs())
        .fold(0.0f64, f64::max);
    assert!(max_f < 1e-4_f64, "max force = {max_f:.4e}");
}

#[test]
fn cpu_fcc_4x4x4_newton() {
    let pot = load_cu01();
    let a = 3.615f64;
    let (pos, types) = cu_fcc_supercell(a, 4);
    let l = 4.0 * a;
    let res = compute_eam_cpu(&pot, &pos, &types, Some(ortho_cell(l, l, l)));
    let sum: [f64; 3] = res
        .forces
        .iter()
        .fold([0.0f64; 3], |a, f| [a[0] + f[0], a[1] + f[1], a[2] + f[2]]);
    let rms = (sum.iter().map(|v| v * v).sum::<f64>() / 3.0).sqrt();
    assert!(rms < 1e-4_f64, "Newton RMS = {rms:.2e}");
}

#[test]
fn cpu_fcc_4x4x4_energy_sum() {
    let pot = load_cu01();
    let a = 3.615f64;
    let (pos, types) = cu_fcc_supercell(a, 4);
    let l = 4.0 * a;
    let res = compute_eam_cpu(&pot, &pos, &types, Some(ortho_cell(l, l, l)));
    let sum: f64 = res.energy_per_atom.iter().sum();
    assert!(
        (res.energy - sum).abs() < 1e-3_f64,
        "total={:.6} sum={:.6}",
        res.energy,
        sum
    );
}

#[test]
fn cpu_fcc_4x4x4_restoring_force() {
    let pot = load_cu01();
    let a = 3.615f64;
    let (mut pos, types) = cu_fcc_supercell(a, 4);
    pos[1][0] += 0.1;
    let l = 4.0 * a;
    let res = compute_eam_cpu(&pot, &pos, &types, Some(ortho_cell(l, l, l)));
    assert!(
        res.forces[1][0] < -0.1_f64,
        "expected restoring force, got {:.4}",
        res.forces[1][0]
    );
}

#[test]
#[should_panic(expected = "Cell too small")]
fn cpu_rejects_small_cell() {
    let pot = load_cu01();
    compute_eam_cpu(
        &pot,
        &[[0.0; 3], [1.0, 0.0, 0.0]],
        &[0, 0],
        Some(ortho_cell(3.615, 3.615, 3.615)),
    );
}
