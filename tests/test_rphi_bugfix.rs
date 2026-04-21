//! Regression tests for the r·φ(r) bug fix in synthetic_cu_alloy_src.
//!
//! BUG: `.eam.alloy` format stores r·φ(r), not φ(r). The synthetic test data
//! generators wrote φ(r) directly, so after the parser divided by r, the
//! stored pair_tables contained φ(r)/r instead of φ(r). This gave wrong
//! pair energies and forces in all tests using the synthetic potential.
//!
//! The fix changes all synthetic generators to write r·φ(r).

mod common;

use cream::potential::eam::EamPotential;
use cream::reference::compute_eam_cpu;

// ── Helper: generate WRONG (pre-fix) synthetic data for comparison ─────────

/// Old buggy version that writes phi(r) instead of r*phi(r).
fn buggy_synthetic_src(nr: u32, nrho: u32, dr: f32, drho: f32, cutoff: f32) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    writeln!(s, "# test\n# test\n# test").unwrap();
    writeln!(s, "1 Cu").unwrap();
    writeln!(s, "{nrho} {drho} {nr} {dr} {cutoff}").unwrap();
    writeln!(s, "29 63.546 3.615 fcc").unwrap();
    for i in 0..nrho {
        let rho = i as f32 * drho;
        write!(s, "{:.8e} ", -(rho + 0.01f32).sqrt()).unwrap();
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
    }
    writeln!(s).unwrap();
    // BUG: writes phi(r) = (cutoff-r)^2, but format expects r*phi(r)
    for i in 0..nr {
        let r = i as f32 * dr;
        let v = if r < cutoff {
            let t = cutoff - r;
            t * t
        } else {
            0.0
        };
        write!(s, "{:.8e} ", v).unwrap();
    }
    writeln!(s).unwrap();
    s
}

// ── Test 1: Verify pair table values are correct after fix ─────────────────

#[test]
fn pair_table_stores_phi_not_phi_over_r() {
    let src = common::synthetic_cu_alloy_src(100, 100, 0.05, 0.01, 4.5);
    let pot = EamPotential::from_str(&src).unwrap();

    // At r=2.0 Å (index 40), φ(r) = (4.5 - 2.0)² = 6.25
    // The parser divides r·φ(r) by r, recovering φ(r) = 6.25.
    let r = 2.0f32;
    let idx = (r / pot.dr) as usize;
    let actual = pot.pair_tables[0][idx];
    let expected_phi = (4.5 - r) * (4.5 - r); // 6.25

    assert!(
        (actual - expected_phi).abs() < 0.1,
        "pair_tables[0][{idx}] = {actual:.4}, expected φ({r}) = {expected_phi:.4}. \
         If you see ~{:.4} instead, the r·φ bug has regressed.",
        expected_phi / r
    );
}

#[test]
fn buggy_version_gives_phi_over_r() {
    let src = buggy_synthetic_src(100, 100, 0.05, 0.01, 4.5);
    let pot = EamPotential::from_str(&src).unwrap();

    let r = 2.0f32;
    let idx = (r / pot.dr) as usize;
    let actual = pot.pair_tables[0][idx];
    let expected_phi_over_r = (4.5 - r) * (4.5 - r) / r; // 3.125

    // Confirm the OLD behavior gives the wrong value φ/r
    assert!(
        (actual - expected_phi_over_r).abs() < 0.1,
        "buggy synthetic should yield φ/r = {expected_phi_over_r:.4}, got {actual:.4}"
    );
}

// ── Test 2: Force-energy consistency (finite difference) ───────────────────

#[test]
fn fixed_synthetic_force_energy_fd_consistency() {
    // Higher resolution synthetic for accurate FD test
    let src = common::synthetic_cu_alloy_src(500, 500, 0.01, 0.002, 4.5);
    let pot = EamPotential::from_str(&src).unwrap();

    let a = 3.615f64;
    let mut pos = vec![
        [0.0f64, 0.0, 0.0],
        [a / 2.0, a / 2.0, 0.0],
        [a / 2.0, 0.0, a / 2.0],
        [0.0, a / 2.0, a / 2.0],
    ];
    pos[0][0] += 0.1; // break symmetry
    let types = vec![0u32; 4];

    let res = compute_eam_cpu(&pot, &pos, &types, None);

    let h = 0.005f64;
    let mut max_err = 0.0f64;
    for atom in 0..4 {
        for comp in 0..3 {
            let mut pp = pos.clone();
            pp[atom][comp] += h;
            let mut pm = pos.clone();
            pm[atom][comp] -= h;
            let ep = compute_eam_cpu(&pot, &pp, &types, None).energy;
            let em = compute_eam_cpu(&pot, &pm, &types, None).energy;
            let fd = -(ep - em) / (2.0 * h);
            let err = (fd - res.forces[atom][comp]).abs();
            max_err = max_err.max(err);
        }
    }
    assert!(
        max_err < 0.005,
        "max FD error {max_err:.2e} — force/energy inconsistency in fixed synthetic potential"
    );
}

#[test]
fn buggy_synthetic_gives_wrong_pair_energy() {
    // The buggy version stores φ(r)/r instead of φ(r), so pair energies
    // are wrong even though forces/energy are internally consistent.
    let buggy = EamPotential::from_str(&buggy_synthetic_src(100, 100, 0.05, 0.01, 4.5)).unwrap();
    let fixed =
        EamPotential::from_str(&common::synthetic_cu_alloy_src(100, 100, 0.05, 0.01, 4.5)).unwrap();

    let a = 3.615f64;
    let pos = vec![
        [0.0f64, 0.0, 0.0],
        [a / 2.0, a / 2.0, 0.0],
        [a / 2.0, 0.0, a / 2.0],
        [0.0, a / 2.0, a / 2.0],
    ];
    let types = vec![0u32; 4];

    let e_buggy = compute_eam_cpu(&buggy, &pos, &types, None).energy;
    let e_fixed = compute_eam_cpu(&fixed, &pos, &types, None).energy;

    // Energies should differ significantly (wrong pair potential)
    let rel_diff = ((e_buggy - e_fixed) / e_fixed).abs();
    assert!(
        rel_diff > 0.01,
        "buggy and fixed should give different energies, but rel_diff = {rel_diff:.4e}"
    );
}

// ── Test 3: CuAg binary alloy FD consistency ──────────────────────────────

#[test]
fn cuag_binary_alloy_fd_consistency() {
    let src = common::synthetic_cuag_alloy_src(500, 500, 0.01, 0.002, 4.5);
    let pot = EamPotential::from_str(&src).unwrap();
    assert_eq!(pot.elements.len(), 2);

    let a = 3.615f64;
    let mut pos = vec![
        [0.0f64, 0.0, 0.0],
        [a / 2.0, a / 2.0, 0.0],
        [a / 2.0, 0.0, a / 2.0],
        [0.0, a / 2.0, a / 2.0],
    ];
    pos[0][0] += 0.08;
    // Mix atom types: Cu, Ag, Cu, Ag
    let types = vec![0u32, 1, 0, 1];

    let res = compute_eam_cpu(&pot, &pos, &types, None);

    let h = 0.005f64;
    let mut max_err = 0.0f64;
    for atom in 0..4 {
        for comp in 0..3 {
            let mut pp = pos.clone();
            pp[atom][comp] += h;
            let mut pm = pos.clone();
            pm[atom][comp] -= h;
            let ep = compute_eam_cpu(&pot, &pp, &types, None).energy;
            let em = compute_eam_cpu(&pot, &pm, &types, None).energy;
            let fd = -(ep - em) / (2.0 * h);
            let err = (fd - res.forces[atom][comp]).abs();
            max_err = max_err.max(err);
        }
    }
    assert!(
        max_err < 0.005,
        "CuAg binary FD error {max_err:.2e} — multi-element pair table may be wrong"
    );
}

// ── Test 4: Real Cu01 potential FD consistency ────────────────────────────

#[test]
fn real_cu01_force_energy_fd_consistency() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("Cu01_eam.alloy");
    let pot = EamPotential::from_file(&path).expect("Cu01_eam.alloy not found");

    let a = 3.615f64;
    let mut pos = vec![
        [0.0f64, 0.0, 0.0],
        [a / 2.0, a / 2.0, 0.0],
        [a / 2.0, 0.0, a / 2.0],
        [0.0, a / 2.0, a / 2.0],
    ];
    pos[0][0] += 0.1;
    let types = vec![0u32; 4];

    let res = compute_eam_cpu(&pot, &pos, &types, None);

    let h = 0.005f64;
    let mut max_err = 0.0f64;
    for atom in 0..4 {
        for comp in 0..3 {
            let mut pp = pos.clone();
            pp[atom][comp] += h;
            let mut pm = pos.clone();
            pm[atom][comp] -= h;
            let ep = compute_eam_cpu(&pot, &pp, &types, None).energy;
            let em = compute_eam_cpu(&pot, &pm, &types, None).energy;
            let fd = -(ep - em) / (2.0 * h);
            let err = (fd - res.forces[atom][comp]).abs();
            max_err = max_err.max(err);
        }
    }
    assert!(
        max_err < 0.001,
        "real Cu01 FD error {max_err:.2e} — force/energy inconsistency"
    );
}
