//! Integration tests: EAM file parser and CPU reference implementation.
//!
//! Runs on CPU only — no GPU required, always executed in CI.
//!
//! Run: `cargo test --test parser`

mod common;
use common::{cu_fcc_4atom_pos3, cu_fcc_4atom_types, pos3_to_f64, synthetic_cu_alloy_src};
use cream::{potential::eam::EamPotential, reference::compute_eam_cpu};

// ── Parser tests ────────────────────────────────────────────────────────────────

#[test]
fn parse_roundtrip_single_element() {
    let src = synthetic_cu_alloy_src(100, 100, 0.05, 0.01, 4.5);
    let pot = EamPotential::from_str(&src).expect("parse failed");
    assert_eq!(pot.elements, vec!["Cu"]);
    assert_eq!(pot.nr, 100);
    assert_eq!(pot.nrho, 100);
    assert!((pot.cutoff_angstrom - 4.5).abs() < 1e-5, "cutoff mismatch");
    assert!((pot.dr - 0.05).abs() < 1e-6, "dr mismatch");
    assert!((pot.drho - 0.01).abs() < 1e-6, "drho mismatch");
}

#[test]
fn parse_table_counts_single_element() {
    let src = synthetic_cu_alloy_src(50, 60, 0.1, 0.02, 5.0);
    let pot = EamPotential::from_str(&src).unwrap();
    // 1 element: embed × 1, rho × 1, pair × 1
    assert_eq!(pot.embed_tables.len(), 1);
    assert_eq!(pot.rho_tables.len(), 1);
    assert_eq!(pot.pair_tables.len(), 1);
    // derivative tables must also match counts
    assert_eq!(pot.d_embed_tables.len(), 1);
    assert_eq!(pot.d_rho_tables.len(), 1);
    assert_eq!(pot.d_pair_tables.len(), 1);
    // points per table
    assert_eq!(pot.embed_tables[0].len(), 60);
    assert_eq!(pot.rho_tables[0].len(), 50);
    assert_eq!(pot.pair_tables[0].len(), 50);
}

#[test]
fn parse_two_element_pair_count() {
    use std::fmt::Write;
    let (nr, nrho) = (50usize, 50usize);
    let (dr, drho, cutoff) = (0.1f32, 0.02f32, 4.5f32);
    let mut src = String::new();
    writeln!(src, "# line1").unwrap();
    writeln!(src, "# line2").unwrap();
    writeln!(src, "# line3").unwrap();
    writeln!(src, "2 Cu Ag").unwrap();
    writeln!(src, "{nrho} {drho} {nr} {dr} {cutoff}").unwrap();
    for _elem in 0..2 {
        writeln!(src, "29 63.5 3.6 fcc").unwrap();
        for i in 0..nrho {
            write!(src, "{:.4e} ", -(i as f32 * drho + 0.01).sqrt()).unwrap();
        }
        writeln!(src).unwrap();
        for i in 0..nr {
            let r = i as f32 * dr;
            write!(
                src,
                "{:.4e} ",
                if r < cutoff {
                    (cutoff - r).powi(2)
                } else {
                    0.0
                }
            )
            .unwrap();
        }
        writeln!(src).unwrap();
    }
    // pair tables: Cu-Cu, Cu-Ag, Ag-Ag (3 total)
    for _ in 0..3 {
        for i in 0..nr {
            let r = i as f32 * dr;
            write!(
                src,
                "{:.4e} ",
                if r < cutoff {
                    (cutoff - r).powi(2)
                } else {
                    0.0
                }
            )
            .unwrap();
        }
        writeln!(src).unwrap();
    }
    let pot = EamPotential::from_str(&src).unwrap();
    assert_eq!(pot.elements, vec!["Cu", "Ag"]);
    assert_eq!(pot.pair_tables.len(), 3, "n_elem*(n_elem+1)/2 = 3");
}

#[test]
fn parse_error_missing_element_name() {
    // Element count/name mismatch should produce ParseError
    let src = "# c\n# c\n# c\n2 Cu\n100 0.01 100 0.05 4.5\n";
    assert!(EamPotential::from_str(src).is_err());
}

// ── Flat buffer tests ───────────────────────────────────────────────────────────

#[test]
fn flat_buffer_size() {
    let src = synthetic_cu_alloy_src(20, 20, 0.2, 0.05, 4.0);
    let pot = EamPotential::from_str(&src).unwrap();
    let (buf, layout) = pot.build_flat_buffer();
    // 1 elem: rho(20) + embed(20) + dembed(20) + pair(20) + drho(20) + dpair(20) = 120
    assert_eq!(buf.len(), 120);
    assert_eq!(layout.nr, 20);
    assert_eq!(layout.nrho, 20);
    assert_eq!(layout.n_elem, 1);
    assert_eq!(layout.n_pairs, 1);
    assert!((layout.dr_inv - 1.0 / 0.2).abs() < 1e-4);
    assert!((layout.drho_inv - 1.0 / 0.05).abs() < 1e-3);
}

#[test]
fn flat_buffer_offsets_monotone() {
    let src = synthetic_cu_alloy_src(20, 20, 0.2, 0.05, 4.0);
    let pot = EamPotential::from_str(&src).unwrap();
    let (_, layout) = pot.build_flat_buffer();
    let offsets = [
        layout.rho_offset,
        layout.embed_offset,
        layout.dembed_offset,
        layout.pair_offset,
        layout.drho_tab_offset,
        layout.dpair_offset,
    ];
    for i in 0..offsets.len() - 1 {
        assert!(
            offsets[i] < offsets[i + 1],
            "offset[{i}]={} >= offset[{}]={} (should be strictly increasing)",
            offsets[i],
            i + 1,
            offsets[i + 1]
        );
    }
}

#[test]
fn table_layout_size_is_64_bytes() {
    use cream::potential::eam::TableLayout;
    assert_eq!(
        std::mem::size_of::<TableLayout>(),
        64,
        "TableLayout must be 64 bytes for WGSL uniform alignment"
    );
}

// ── CPU reference implementation tests ─────────────────────────────────────────

#[test]
fn cpu_forces_finite_cu4() {
    let pot = EamPotential::from_str(&synthetic_cu_alloy_src(100, 100, 0.05, 0.01, 4.5)).unwrap();
    let res = compute_eam_cpu(
        &pot,
        &pos3_to_f64(&cu_fcc_4atom_pos3()),
        &cu_fcc_4atom_types(),
        None,
    );
    for (i, f) in res.forces.iter().enumerate() {
        for (c, &v) in ['x', 'y', 'z'].iter().zip(f.iter()) {
            assert!(v.is_finite(), "forces[{i}].{c} = {v} is non-finite");
        }
    }
    assert!(res.energy.is_finite(), "energy is non-finite");
}

#[test]
fn cpu_energy_is_finite_cu4() {
    // Synthetic potential can be strongly repulsive, so sign is not checked; only finiteness.
    let pot = EamPotential::from_str(&synthetic_cu_alloy_src(100, 100, 0.05, 0.01, 4.5)).unwrap();
    let res = compute_eam_cpu(
        &pot,
        &pos3_to_f64(&cu_fcc_4atom_pos3()),
        &cu_fcc_4atom_types(),
        None,
    );
    assert!(
        res.energy.is_finite(),
        "energy = {} (should be finite)",
        res.energy
    );
}

/// Acceptance criterion: Newton's third law Σ Fᵢ ≈ 0 (< 1e-3 eV/Å).
#[test]
fn phase1_acceptance_newton_third_law() {
    let pot = EamPotential::from_str(&synthetic_cu_alloy_src(200, 200, 0.025, 0.005, 4.5)).unwrap();
    let res = compute_eam_cpu(
        &pot,
        &pos3_to_f64(&cu_fcc_4atom_pos3()),
        &cu_fcc_4atom_types(),
        None,
    );
    let sum: [f64; 3] = res
        .forces
        .iter()
        .fold([0.0f64; 3], |a, f| [a[0] + f[0], a[1] + f[1], a[2] + f[2]]);
    let rms = (sum.iter().map(|v| v * v).sum::<f64>() / 3.0).sqrt();
    assert!(
        rms < 1e-3,
        "Newton RMS = {rms:.2e} eV/A (limit 1e-3). sum = [{:.2e},{:.2e},{:.2e}]",
        sum[0],
        sum[1],
        sum[2]
    );
}

/// pair_index should be symmetric and correct.
#[test]
fn pair_index_symmetry_3elem() {
    let expected = [
        (0, 0, 0),
        (0, 1, 1),
        (0, 2, 2),
        (1, 1, 3),
        (1, 2, 4),
        (2, 2, 5),
    ];
    for (a, b, want) in expected {
        let got_ab = EamPotential::pair_index(a, b, 3);
        let got_ba = EamPotential::pair_index(b, a, 3);
        assert_eq!(got_ab, want, "pair_index({a},{b},3)");
        assert_eq!(got_ba, want, "pair_index({b},{a},3) symmetry");
    }
}

/// CPU reference energy should equal the sum of per-atom energies.
#[test]
fn cpu_energy_equals_per_atom_sum() {
    let pot = EamPotential::from_str(&synthetic_cu_alloy_src(100, 100, 0.05, 0.01, 4.5)).unwrap();
    let res = compute_eam_cpu(
        &pot,
        &pos3_to_f64(&cu_fcc_4atom_pos3()),
        &cu_fcc_4atom_types(),
        None,
    );
    let per_atom_sum: f64 = res.energy_per_atom.iter().sum();
    assert!(
        (res.energy - per_atom_sum).abs() < 1e-5,
        "energy={} per_atom_sum={} diff={}",
        res.energy,
        per_atom_sum,
        (res.energy - per_atom_sum).abs()
    );
}

// ── Multi-element (Cu-Ag alloy) tests ──────────────────────────────────────────

#[test]
fn cpu_cuag_2elem_forces_finite() {
    let src = common::synthetic_cuag_alloy_src(100, 100, 0.05, 0.01, 4.5);
    let pot = EamPotential::from_str(&src).unwrap();
    assert_eq!(pot.elements, vec!["Cu", "Ag"]);
    assert_eq!(
        pot.pair_tables.len(),
        3,
        "expected 3 pair tables: Cu-Cu, Cu-Ag, Ag-Ag"
    );

    // Cu2Ag2 mixed cluster
    let pos = [
        [0.0f64, 0.0, 0.0],
        [2.6, 0.0, 0.0],
        [1.3, 2.2, 0.0],
        [1.3, 0.8, 2.1],
    ];
    let types = [0u32, 1, 0, 1]; // Cu, Ag, Cu, Ag
    let res = compute_eam_cpu(&pot, &pos, &types, None);

    for (i, f) in res.forces.iter().enumerate() {
        for (c, &v) in ['x', 'y', 'z'].iter().zip(f.iter()) {
            assert!(v.is_finite(), "CuAg forces[{i}].{c} = {v} is non-finite");
        }
    }
    assert!(res.energy.is_finite(), "CuAg energy is non-finite");
}

/// Acceptance criterion: Newton's third law should hold for Cu-Ag binary systems.
#[test]
fn phase1_acceptance_cuag_newton_third_law() {
    let src = common::synthetic_cuag_alloy_src(200, 200, 0.025, 0.005, 4.5);
    let pot = EamPotential::from_str(&src).unwrap();
    let pos = [
        [0.0f64, 0.0, 0.0],
        [2.6, 0.0, 0.0],
        [1.3, 2.2, 0.0],
        [1.3, 0.8, 2.1],
        [0.0, 2.6, 0.0],
        [0.0, 0.0, 2.6],
        [2.6, 2.6, 0.0],
        [0.0, 2.6, 2.6],
    ];
    let types = [0u32, 1, 0, 1, 0, 1, 0, 1];
    let res = compute_eam_cpu(&pot, &pos, &types, None);

    let sum: [f64; 3] = res
        .forces
        .iter()
        .fold([0.0f64; 3], |a, f| [a[0] + f[0], a[1] + f[1], a[2] + f[2]]);
    let rms = (sum.iter().map(|v| v * v).sum::<f64>() / 3.0).sqrt();
    assert!(
        rms < 1e-3,
        "CuAg Newton RMS = {rms:.2e} eV/A (limit 1e-3). sum = [{:.2e},{:.2e},{:.2e}]",
        sum[0],
        sum[1],
        sum[2]
    );
}

// ── Additional coverage tests ─────────────────────────────────────────────────

/// numerical_deriv must not panic for single-point tables (n=1 guard).
#[test]
fn numerical_deriv_single_point_no_panic() {
    // We exercise the guard via a parse of a 1-point potential (nr=nrho=1).
    // The derivative should return [0.0] without panicking.
    use std::fmt::Write as _;
    let mut src = String::new();
    writeln!(src, "# test").unwrap();
    writeln!(src, "# test").unwrap();
    writeln!(src, "# test").unwrap();
    writeln!(src, "1 Cu").unwrap();
    writeln!(src, "1 0.01 1 0.05 4.5").unwrap();
    writeln!(src, "29 63.546 3.615 fcc").unwrap();
    writeln!(src, "-1.0").unwrap(); // F(rho): 1 point
    writeln!(src, "0.5").unwrap(); // f(r): 1 point
    writeln!(src, "1.0").unwrap(); // phi(r): 1 point
    let pot = EamPotential::from_str(&src).expect("parse failed");
    assert_eq!(
        pot.d_embed_tables[0],
        vec![0.0f32],
        "derivative of 1-point table should be [0]"
    );
    assert_eq!(pot.d_rho_tables[0], vec![0.0f32]);
    assert_eq!(pot.d_pair_tables[0], vec![0.0f32]);
}

/// Verify pair_index covers n_elem=4 (the n*(n+1)/2 = 10 case).
#[test]
fn pair_index_4elem_exhaustive() {
    let n = 4usize;
    let mut idx = 0usize;
    for lo in 0..n {
        for hi in lo..n {
            let got = EamPotential::pair_index(lo, hi, n);
            assert_eq!(got, idx, "pair_index({lo},{hi},{n}) expected {idx}");
            // symmetry
            assert_eq!(EamPotential::pair_index(hi, lo, n), idx);
            idx += 1;
        }
    }
    // n*(n+1)/2 = 10 distinct pairs
    assert_eq!(idx, 10);
}

/// build_flat_buffer for a 2-element potential must have correct sizes and offsets.
#[test]
fn flat_buffer_two_element_layout() {
    use cream::potential::eam::TableLayout;
    let src = common::synthetic_cuag_alloy_src(20, 30, 0.2, 0.05, 4.0);
    let pot = EamPotential::from_str(&src).unwrap();
    let (buf, layout) = pot.build_flat_buffer();

    let _nr = 20usize;
    let _nrho = 30usize;
    let _n_elem = 2usize;
    let _n_pairs = 3usize; // 2*(2+1)/2

    // Expected segment sizes (f32 count):
    // rho:    n_elem * nr   = 2*20 = 40
    // embed:  n_elem * nrho = 2*30 = 60
    // dembed: n_elem * nrho = 60
    // pair:   n_pairs * nr  = 3*20 = 60
    // drho:   n_elem * nr   = 40
    // dpair:  n_pairs * nr  = 60
    let expected_total = 40 + 60 + 60 + 60 + 40 + 60;
    assert_eq!(buf.len(), expected_total, "2-element flat buffer size");
    assert_eq!(layout.n_elem, 2);
    assert_eq!(layout.n_pairs, 3);
    assert_eq!(layout.nr, 20);
    assert_eq!(layout.nrho, 30);

    // Offsets must be strictly increasing
    let offsets = [
        layout.rho_offset,
        layout.embed_offset,
        layout.dembed_offset,
        layout.pair_offset,
        layout.drho_tab_offset,
        layout.dpair_offset,
    ];
    for i in 0..offsets.len() - 1 {
        assert!(
            offsets[i] < offsets[i + 1],
            "offset[{i}]={} >= offset[{}]={} (must be strictly increasing)",
            offsets[i],
            i + 1,
            offsets[i + 1]
        );
    }

    // TableLayout size must remain 64 bytes
    assert_eq!(std::mem::size_of::<TableLayout>(), 64);
}

/// build_flat_buffer for a 3-element potential: n_pairs = 6.
#[test]
fn flat_buffer_three_element_pair_count() {
    use std::fmt::Write as _;
    let (nr, nrho) = (10usize, 10usize);
    let (dr, drho, cutoff) = (0.5f32, 0.1f32, 4.5f32);
    let n_elem = 3usize;
    let n_pairs = n_elem * (n_elem + 1) / 2; // 6

    let mut src = String::new();
    writeln!(src, "# test3").unwrap();
    writeln!(src, "# test3").unwrap();
    writeln!(src, "# test3").unwrap();
    writeln!(src, "3 Cu Ag Au").unwrap();
    writeln!(src, "{nrho} {drho} {nr} {dr} {cutoff}").unwrap();
    for _ in 0..n_elem {
        writeln!(src, "29 63.5 3.6 fcc").unwrap();
        for i in 0..nrho {
            write!(src, "{:.4e} ", -(i as f32 * drho + 0.01).sqrt()).unwrap();
        }
        writeln!(src).unwrap();
        for i in 0..nr {
            let r = i as f32 * dr;
            write!(
                src,
                "{:.4e} ",
                if r < cutoff {
                    (cutoff - r).powi(2)
                } else {
                    0.0
                }
            )
            .unwrap();
        }
        writeln!(src).unwrap();
    }
    for _ in 0..n_pairs {
        for i in 0..nr {
            let r = i as f32 * dr;
            write!(
                src,
                "{:.4e} ",
                if r < cutoff {
                    (cutoff - r).powi(2)
                } else {
                    0.0
                }
            )
            .unwrap();
        }
        writeln!(src).unwrap();
    }

    let pot = EamPotential::from_str(&src).unwrap();
    assert_eq!(pot.elements.len(), 3);
    assert_eq!(pot.pair_tables.len(), 6, "n_elem=3 → 6 pair tables");
    let (buf, layout) = pot.build_flat_buffer();
    let expected = 3 * nr + 3 * nrho + 3 * nrho + 6 * nr + 3 * nr + 6 * nr;
    assert_eq!(buf.len(), expected, "3-element flat buffer total size");
    assert_eq!(layout.n_elem, 3);
    assert_eq!(layout.n_pairs, 6);
}

/// Parser must reject a file where the pair-table data is truncated.
#[test]
fn parse_error_truncated_pair_table() {
    use std::fmt::Write as _;
    let mut src = String::new();
    writeln!(src, "# c\n# c\n# c").unwrap();
    writeln!(src, "1 Cu").unwrap();
    writeln!(src, "10 0.01 10 0.05 4.5").unwrap();
    writeln!(src, "29 63.5 3.6 fcc").unwrap();
    for _ in 0..10 {
        write!(src, "0.0 ").unwrap();
    }
    writeln!(src).unwrap(); // embed
    for _ in 0..10 {
        write!(src, "0.0 ").unwrap();
    }
    writeln!(src).unwrap(); // rho
                            // Pair table should have 10 values — supply only 5 (truncation)
    for _ in 0..5 {
        write!(src, "0.0 ").unwrap();
    }
    assert!(
        EamPotential::from_str(&src).is_err(),
        "truncated pair table should error"
    );
}

/// atom_type out-of-range causes a clear panic in the reference implementation.
#[test]
#[should_panic(expected = "out of range")]
fn cpu_reference_atom_type_out_of_range_panics() {
    let src = synthetic_cu_alloy_src(100, 100, 0.05, 0.01, 4.5);
    let pot = EamPotential::from_str(&src).unwrap();
    // atom type 1 does not exist in a 1-element potential
    let pos = [[0.0f64, 0.0, 0.0], [2.5, 0.0, 0.0]];
    let types = [0u32, 1u32]; // type 1 is invalid for n_elem=1
    compute_eam_cpu(&pot, &pos, &types, None);
}
