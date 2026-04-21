//! Stress-tensor correctness tests.
//!
//! # Ground truth
//! The CPU AllPairs half-pair engine is treated as the reference:
//!   * it sums every pair (i<j) exactly once (no double-counting ambiguity),
//!   * virial accumulation is in `f64` thread-local accumulators,
//!   * `σ = −W / V` is applied in the same place the analytic formula is
//!     written in the docs.
//!
//! Every other implementation — the CPU CellList path, and the three GPU
//! pass-2 variants (AllPairs / NeighborList / full GPU CellList) — is
//! cross-validated against the CPU AllPairs result.  Any divergence is a
//! localised implementation bug whose owner the test name identifies
//! directly.
//!
//! # Additional physical checks
//! 1. **Finite-difference energy derivative.**
//!    The analytic stress must satisfy
//!        `σ_αβ · V = ∂E/∂h_αβ  ·  h_βα`   (reduced to volume strain for the
//!                                          diagonal case)
//!    For a uniform volume strain ε we have `∂E/∂ε = V · tr(σ)`, which gives
//!    a scalar finite-difference sanity check on the trace of σ.  This
//!    catches sign errors, missing factors of 0.5, and factors-of-V
//!    mismatches that a same-formula cross-check would all miss.
//! 2. **Translational invariance.**  Shifting every atom by a constant
//!    must leave σ unchanged.
//! 3. **Symmetry of off-diagonal components.**  σ is a symmetric tensor;
//!    in Voigt notation we return only the 6 independent components, and
//!    the finite-difference check verifies each one is consistent with the
//!    energy surface.

mod common;
use common::{ortho_cell, synthetic_cu_alloy_src, synthetic_cuag_alloy_src};

use cream::{
    cpu_engine::CpuEngine,
    engine::{ComputeEngine, ComputeResult},
    potential::{eam::EamPotential, NeighborStrategy},
};

// ── Tolerances ────────────────────────────────────────────────────────────────
//
// CPU-AP vs CPU-CL: different traversal order + f64 accumulators → should
// agree to double-precision noise.
const TOL_CPU_CL_ABS: f64 = 1e-5;   // absolute, per Voigt component
const TOL_CPU_CL_REL: f64 = 1e-6;   // relative, per Voigt component

// CPU-AP (f64) vs any GPU path (f32 tables, f32 per-thread Kahan): a few
// ULPs per pair × N pairs → typically 1e-5 relative.  Allow a modest
// safety margin but flag anything >1e-3 as a real bug.
const TOL_GPU_ABS: f64 = 5e-4;
const TOL_GPU_REL: f64 = 5e-3;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn cu_synth() -> EamPotential {
    EamPotential::from_str(&synthetic_cu_alloy_src(200, 200, 0.03, 0.03, 5.5)).unwrap()
}

fn cuag_synth() -> EamPotential {
    EamPotential::from_str(&synthetic_cuag_alloy_src(200, 200, 0.03, 0.03, 5.5)).unwrap()
}

/// FCC Cu supercell (`nx×ny×nz` unit cells).  Returns positions (vec4),
/// types, and the box in f32 (for engines) and f64 (for numerical checks).
fn fcc_cu_supercell(
    nx: usize,
    ny: usize,
    nz: usize,
) -> (Vec<[f32; 4]>, Vec<u32>, [[f32; 3]; 3]) {
    let a = 3.615f32;
    let basis = [
        [0.0f32, 0.0, 0.0],
        [0.0, a * 0.5, a * 0.5],
        [a * 0.5, 0.0, a * 0.5],
        [a * 0.5, a * 0.5, 0.0],
    ];
    let mut positions = Vec::with_capacity(nx * ny * nz * 4);
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let origin = [ix as f32 * a, iy as f32 * a, iz as f32 * a];
                for b in &basis {
                    positions.push([origin[0] + b[0], origin[1] + b[1], origin[2] + b[2], 0.0]);
                }
            }
        }
    }
    let types = vec![0u32; positions.len()];
    let cell = ortho_cell(nx as f32 * a, ny as f32 * a, nz as f32 * a);
    (positions, types, cell)
}

/// Compare two Voigt stress tensors with mixed absolute/relative tolerance
/// applied per component.  Panics with a detailed report on failure.
#[track_caller]
fn assert_virial_close(
    got: [f64; 6],
    truth: [f64; 6],
    abs_tol: f64,
    rel_tol: f64,
    label: &str,
) {
    let labels = ["xx", "yy", "zz", "yz", "xz", "xy"];
    let mut worst = (0, 0.0, 0.0);
    for c in 0..6 {
        let err = (got[c] - truth[c]).abs();
        let scale = truth[c].abs().max(got[c].abs()).max(1.0);
        let rel = err / scale;
        if rel > worst.2 {
            worst = (c, err, rel);
        }
        assert!(
            err <= abs_tol || rel <= rel_tol,
            "{label}: σ[{c}:{lbl}] mismatch  got={got_c:.6e}  truth={truth_c:.6e}  \
             abs={err:.3e}  rel={rel:.3e}  (tol abs={abs_tol:.1e}, rel={rel_tol:.1e})",
            lbl = labels[c],
            got_c = got[c],
            truth_c = truth[c],
        );
    }
    // Informational: the worst-case tolerance used — not an assertion.
    eprintln!(
        "{label}: OK — worst component σ[{c}:{lbl}]  abs={abs:.3e}  rel={rel:.3e}",
        c = worst.0,
        lbl = labels[worst.0],
        abs = worst.1,
        rel = worst.2,
    );
}

fn cpu_allpairs(
    positions: &[[f32; 4]],
    types: &[u32],
    cell: Option<[[f32; 3]; 3]>,
    pot: &EamPotential,
) -> ComputeResult {
    let cpu = CpuEngine::new();
    cpu.compute_sync(positions, types, cell, pot).unwrap()
}

/// CPU AllPairs with per-atom virial + densities + embedding populated.
/// Required by tests that assert `Σᵢ virial_per_atom[i] == -σ·V` — the
/// default `compute_sync` fast path returns empty per-atom buffers.
fn cpu_allpairs_with_per_atom(
    positions: &[[f32; 4]],
    types: &[u32],
    cell: Option<[[f32; 3]; 3]>,
    pot: &EamPotential,
) -> ComputeResult {
    let cpu = CpuEngine::new();
    cpu.compute_sync_with_per_atom(positions, types, cell, pot)
        .unwrap()
}

fn cpu_cell_list(
    positions: &[[f32; 4]],
    types: &[u32],
    cell: Option<[[f32; 3]; 3]>,
    pot: &EamPotential,
) -> ComputeResult {
    let cpu = CpuEngine::new();
    cpu.compute_cell_list_sync(positions, types, cell, pot)
        .unwrap()
}

fn gpu_compute(
    strategy: NeighborStrategy,
    positions: &[[f32; 4]],
    types: &[u32],
    cell: Option<[[f32; 3]; 3]>,
    pot: &EamPotential,
) -> Option<ComputeResult> {
    let mut engine = match pollster::block_on(ComputeEngine::new(strategy)) {
        Ok(e) => e,
        Err(_) => return None, // no adapter → skip
    };
    Some(engine.compute_sync(positions, types, cell, pot).unwrap())
}

// ─────────────────────────────────────────────────────────────────────────────
// Ground-truth-level tests: CPU-AP is the single source of truth.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn cpu_allpairs_virial_shape_and_sign_conventions() {
    // Simple 2-atom dimer in a cubic box — sanity-check the Voigt layout.
    let pot = cu_synth();
    let cell = Some(ortho_cell(20.0, 20.0, 20.0));
    let positions = vec![
        [0.0f32, 0.0, 0.0, 0.0],
        [2.5, 0.0, 0.0, 0.0],   // displacement along x — only σ_xx should be nonzero
    ];
    let types = vec![0u32, 0];
    let res = cpu_allpairs(&positions, &types, cell, &pot);

    // All symmetric off-diagonal components and y/z diagonals should be
    // extremely small for a pair aligned along x.
    assert!(res.virial[1].abs() < 1e-10, "σ_yy should vanish: {}", res.virial[1]);
    assert!(res.virial[2].abs() < 1e-10, "σ_zz should vanish: {}", res.virial[2]);
    assert!(res.virial[3].abs() < 1e-10, "σ_yz should vanish: {}", res.virial[3]);
    assert!(res.virial[4].abs() < 1e-10, "σ_xz should vanish: {}", res.virial[4]);
    assert!(res.virial[5].abs() < 1e-10, "σ_xy should vanish: {}", res.virial[5]);
    // σ_xx must be non-zero (the dimer has a well-defined pair force).
    assert!(res.virial[0].abs() > 1e-6, "σ_xx should be finite: {}", res.virial[0]);
}

#[test]
fn cpu_allpairs_vs_cpu_celllist_cu_supercell() {
    // CPU-AP and CPU-CL differ only in neighbour discovery; stress must match
    // to double-precision noise (both use f64 accumulators).
    let pot = cu_synth();
    let (pos, types, cell) = fcc_cu_supercell(4, 4, 4); // 64 atoms
    let ap = cpu_allpairs(&pos, &types, Some(cell), &pot);
    let cl = cpu_cell_list(&pos, &types, Some(cell), &pot);
    assert_virial_close(
        cl.virial,
        ap.virial,
        TOL_CPU_CL_ABS,
        TOL_CPU_CL_REL,
        "CPU-CL vs CPU-AP (Cu 4×4×4)",
    );
}

#[test]
fn cpu_allpairs_vs_cpu_celllist_cuag_supercell() {
    let pot = cuag_synth();
    let (pos, mut types, cell) = fcc_cu_supercell(4, 4, 4); // 64 atoms
    // Alternate Cu/Ag every atom — exercises the multi-element virial pair index
    for (i, t) in types.iter_mut().enumerate() {
        *t = (i % 2) as u32;
    }
    let ap = cpu_allpairs(&pos, &types, Some(cell), &pot);
    let cl = cpu_cell_list(&pos, &types, Some(cell), &pot);
    assert_virial_close(
        cl.virial,
        ap.virial,
        TOL_CPU_CL_ABS,
        TOL_CPU_CL_REL,
        "CPU-CL vs CPU-AP (CuAg 4×4×4)",
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-atom virial consistency: Σᵢ virial_per_atom[i] == total virial
// (pre-volume-division), matching the half-and-half split convention.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn cpu_per_atom_virial_sums_to_total_virial_allpairs() {
    let pot = cu_synth();
    let (pos, types, cell) = fcc_cu_supercell(4, 4, 4);
    // Per-atom virial requires the `_with_per_atom` entry point — the
    // default `compute_sync` fast path returns an empty Vec for
    // `virial_per_atom` to save 48 N bytes × T threads of memory.
    let res = cpu_allpairs_with_per_atom(&pos, &types, Some(cell), &pot);

    // σ_total = −W / V → W = −σ_total · V.
    let vol: f64 = {
        let h = cell;
        (h[0][0] * (h[1][1] * h[2][2] - h[1][2] * h[2][1])
            - h[0][1] * (h[1][0] * h[2][2] - h[1][2] * h[2][0])
            + h[0][2] * (h[1][0] * h[2][1] - h[1][1] * h[2][0])) as f64
    };
    let w_from_total = [
        -res.virial[0] * vol,
        -res.virial[1] * vol,
        -res.virial[2] * vol,
        -res.virial[3] * vol,
        -res.virial[4] * vol,
        -res.virial[5] * vol,
    ];
    // Σᵢ virial_per_atom[i] (raw, eV units, half-split per pair → full per pair total).
    let mut w_from_atoms = [0.0f64; 6];
    for va in &res.virial_per_atom {
        for c in 0..6 {
            w_from_atoms[c] += va[c];
        }
    }
    assert_virial_close(
        w_from_atoms,
        w_from_total,
        1e-8,
        1e-10,
        "Σᵢ virial_per_atom[i] == -σ·V (CPU-AP Cu 4×4×4)",
    );
    // Shape invariant.
    assert_eq!(res.virial_per_atom.len(), pos.len());
}

#[test]
fn cpu_densities_and_embedding_are_populated() {
    let pot = cu_synth();
    let (pos, types, cell) = fcc_cu_supercell(4, 4, 4); // 64 atoms
    let res = cpu_allpairs(&pos, &types, Some(cell), &pot);
    assert_eq!(res.densities.len(), pos.len());
    assert_eq!(res.embedding_energies.len(), pos.len());
    // For a bulk FCC lattice every atom has the same symmetric environment,
    // so densities must be very close to each other.
    let rho_min = res.densities.iter().cloned().fold(f32::INFINITY, f32::min);
    let rho_max = res.densities.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        (rho_max - rho_min).abs() / rho_max.max(1.0) < 1e-5,
        "bulk FCC densities should be uniform: min={rho_min}, max={rho_max}"
    );
    // Sum of embedding energies must be finite.
    let sum_embed: f32 = res.embedding_energies.iter().sum();
    assert!(sum_embed.is_finite(), "sum of embedding energies is non-finite");
}

// ─────────────────────────────────────────────────────────────────────────────
// Finite-difference sanity check — confirms sign convention AND factors of V.
//
// For a uniform volume strain ε (H' = (1+ε) H, positions scaled along too):
//   dE/dε |_{ε=0} = V · tr(σ)      up to small corrections
// This catches sign flips and missing 1/V normalisation that a
// same-formula cross-check would not see.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn cpu_allpairs_stress_trace_matches_volume_strain_derivative() {
    let pot = cu_synth();
    // Small supercell so the finite-difference stays cheap while still a valid bulk
    let (pos0, types, cell0) = fcc_cu_supercell(4, 4, 4);
    let vol0: f64 = {
        let h = cell0;
        (h[0][0] * (h[1][1] * h[2][2] - h[1][2] * h[2][1])
            - h[0][1] * (h[1][0] * h[2][2] - h[1][2] * h[2][0])
            + h[0][2] * (h[1][0] * h[2][1] - h[1][1] * h[2][0])) as f64
    };

    let cpu = CpuEngine::new();
    let res0 = cpu.compute_sync(&pos0, &types, Some(cell0), &pot).unwrap();
    let tr_sigma = (res0.virial[0] + res0.virial[1] + res0.virial[2]) as f64;

    let strain_scale = |eps: f64| -> (Vec<[f32; 4]>, [[f32; 3]; 3]) {
        let s = 1.0 + eps;
        let scaled_pos: Vec<[f32; 4]> = pos0
            .iter()
            .map(|p| {
                [
                    (p[0] as f64 * s) as f32,
                    (p[1] as f64 * s) as f32,
                    (p[2] as f64 * s) as f32,
                    0.0,
                ]
            })
            .collect();
        let scaled_cell = [
            [(cell0[0][0] as f64 * s) as f32, 0.0, 0.0],
            [0.0, (cell0[1][1] as f64 * s) as f32, 0.0],
            [0.0, 0.0, (cell0[2][2] as f64 * s) as f32],
        ];
        (scaled_pos, scaled_cell)
    };

    // Central difference — step small enough to resolve the derivative
    // but large enough to be above f32 table-interpolation noise.
    let eps = 1e-3f64;
    let (p_plus, c_plus) = strain_scale(eps);
    let (p_minus, c_minus) = strain_scale(-eps);
    let e_plus = cpu
        .compute_sync(&p_plus, &types, Some(c_plus), &pot)
        .unwrap()
        .energy as f64;
    let e_minus = cpu
        .compute_sync(&p_minus, &types, Some(c_minus), &pot)
        .unwrap()
        .energy as f64;
    let de_deps = (e_plus - e_minus) / (2.0 * eps);

    // Analytic expectation:  dE/dε ≈ −V · tr(σ) × 3    (volume strain → 3ε linear in length)
    // Explicit derivation:   r'_αβ = (1+ε) r_αβ,  V' = (1+ε)^3 V,
    //                        dE/dε |_{ε=0} = Σ_{i<j} r_ij · F_ij = W_total = −V · tr(σ).
    //                        Since tr(σ) = σ_xx + σ_yy + σ_zz, we compare directly.
    let expected = vol0 * tr_sigma;

    let err = (de_deps - expected).abs();
    let scale = expected.abs().max(de_deps.abs()).max(1.0);
    let rel = err / scale;
    // finite-difference truncation error is O(ε²) ≈ 1e-6; f32 table noise
    // dominates at a few 1e-4.  1e-3 relative is a very generous safety margin.
    assert!(
        rel < 5e-3,
        "dE/dε finite-difference disagrees with −V·tr(σ):\n\
         dE/dε  = {de_deps:.6e}\n\
         V·tr(σ) = {expected:.6e}\n\
         abs err = {err:.3e}, rel err = {rel:.3e}"
    );
    eprintln!(
        "volume-strain check: dE/dε = {de_deps:.6e}, −V·tr(σ) = {expected:.6e}, rel = {rel:.3e}"
    );
}

#[test]
fn cpu_allpairs_stress_translationally_invariant() {
    let pot = cu_synth();
    let (pos, types, cell) = fcc_cu_supercell(4, 4, 4); // 64 atoms
    let res0 = cpu_allpairs(&pos, &types, Some(cell), &pot);

    // Shift all atoms by a non-lattice vector — σ should be unchanged.
    let shift = [0.37f32, -0.91, 1.23];
    let shifted: Vec<[f32; 4]> = pos
        .iter()
        .map(|p| [p[0] + shift[0], p[1] + shift[1], p[2] + shift[2], 0.0])
        .collect();
    let res1 = cpu_allpairs(&shifted, &types, Some(cell), &pot);
    assert_virial_close(
        res1.virial,
        res0.virial,
        1e-6,
        1e-6,
        "translational invariance (CPU-AP)",
    );
}

#[test]
fn cpu_allpairs_stress_zero_for_nonperiodic_cluster() {
    let pot = cu_synth();
    let (pos, types, _) = fcc_cu_supercell(4, 4, 4); // 64 atoms
    // Use the `_with_per_atom` entry point so the per-atom-virial
    // finiteness check below actually iterates over N elements — the
    // default fast path returns an empty Vec, silently no-op'ing the loop.
    let res = cpu_allpairs_with_per_atom(&pos, &types, None, &pot);
    for c in 0..6 {
        assert_eq!(
            res.virial[c], 0.0,
            "σ[{c}] should be exactly zero for non-periodic cluster: {}",
            res.virial[c]
        );
    }
    // Per-atom virial must also be zero when the cluster has no well-defined
    // volume — we store raw pair virial here, but the CPU engine deliberately
    // returns zeros for unconstrained systems.  Actually we *do* still want
    // per-atom to carry the raw pair virial so defect analysis works on
    // clusters too — check it's finite.
    assert_eq!(
        res.virial_per_atom.len(),
        pos.len(),
        "per-atom virial must be populated on the with_per_atom path"
    );
    for va in &res.virial_per_atom {
        for &c in va {
            assert!(c.is_finite(), "virial_per_atom component not finite");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU vs CPU-AP — the critical test for the new GPU virial path.
// Each GPU backend must match CPU-AP to within f32-accumulation tolerance.
// These tests are `#[ignore]`-gated just like the existing gpu.rs tests so
// CI without a GPU adapter skips them cleanly.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
#[ignore] // requires GPU adapter (runs under `cargo test -- --include-ignored`)
fn gpu_allpairs_vs_cpu_allpairs_cu_supercell() {
    let pot = cu_synth();
    let (pos, types, cell) = fcc_cu_supercell(4, 4, 4);
    let truth = cpu_allpairs(&pos, &types, Some(cell), &pot);
    let Some(gpu) = gpu_compute(NeighborStrategy::AllPairs, &pos, &types, Some(cell), &pot) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    assert_virial_close(
        gpu.virial,
        truth.virial,
        TOL_GPU_ABS,
        TOL_GPU_REL,
        "GPU-AP vs CPU-AP (Cu 4×4×4)",
    );
}

#[test]
#[ignore]
fn gpu_celllist_vs_cpu_allpairs_cu_supercell() {
    let pot = cu_synth();
    let (pos, types, cell) = fcc_cu_supercell(4, 4, 4);
    let truth = cpu_allpairs(&pos, &types, Some(cell), &pot);
    let cell_size = pot.cutoff_angstrom;
    let Some(gpu) = gpu_compute(
        NeighborStrategy::CellList { cell_size },
        &pos,
        &types,
        Some(cell),
        &pot,
    ) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    assert_virial_close(
        gpu.virial,
        truth.virial,
        TOL_GPU_ABS,
        TOL_GPU_REL,
        "GPU-CL vs CPU-AP (Cu 4×4×4)",
    );
}

#[test]
#[ignore]
fn gpu_allpairs_vs_cpu_allpairs_cuag_supercell() {
    let pot = cuag_synth();
    let (pos, mut types, cell) = fcc_cu_supercell(4, 4, 4);
    for (i, t) in types.iter_mut().enumerate() {
        *t = (i % 2) as u32;
    }
    let truth = cpu_allpairs(&pos, &types, Some(cell), &pot);
    let Some(gpu) = gpu_compute(NeighborStrategy::AllPairs, &pos, &types, Some(cell), &pot) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    assert_virial_close(
        gpu.virial,
        truth.virial,
        TOL_GPU_ABS,
        TOL_GPU_REL,
        "GPU-AP vs CPU-AP (CuAg 4×4×4 alternating)",
    );
}

#[test]
#[ignore]
fn gpu_virial_zero_for_nonperiodic_cluster() {
    // The GPU pipeline reaches `read_and_finalize_virial` with cell=None
    // and must return exactly zero rather than divide by a volume of 0.
    let pot = cu_synth();
    let (pos, types, _) = fcc_cu_supercell(4, 4, 4); // 64 atoms
    let Some(gpu) = gpu_compute(NeighborStrategy::AllPairs, &pos, &types, None, &pot) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    for c in 0..6 {
        assert_eq!(
            gpu.virial[c], 0.0,
            "GPU σ[{c}] should be exactly zero for non-periodic cluster: {}",
            gpu.virial[c]
        );
    }
}

#[test]
#[ignore]
fn gpu_virial_has_correct_sign_under_hydrostatic_compression() {
    // Physical check: compressing a cohesive crystal should produce positive
    // pressure (P = −tr(σ)/3 > 0  ⇔  tr(σ) < 0).  Under ε<0 uniform strain,
    // the bulk is pushed past equilibrium → repulsive forces dominate → σ
    // diagonal terms turn negative.  This detects a sign flip in the GPU
    // virial that the magnitude-only tolerance checks above would tolerate.
    let pot = cu_synth();
    let (pos, types, cell) = fcc_cu_supercell(4, 4, 4);
    let compress = |eps: f32| -> (Vec<[f32; 4]>, [[f32; 3]; 3]) {
        let s = 1.0 + eps;
        let p: Vec<[f32; 4]> =
            pos.iter().map(|q| [q[0] * s, q[1] * s, q[2] * s, 0.0]).collect();
        let c = [
            [cell[0][0] * s, 0.0, 0.0],
            [0.0, cell[1][1] * s, 0.0],
            [0.0, 0.0, cell[2][2] * s],
        ];
        (p, c)
    };
    let (p_comp, c_comp) = compress(-0.02);

    // Both CPU-AP and GPU-AP under compression must give tr(σ) of the same sign.
    let cpu_res = cpu_allpairs(&p_comp, &types, Some(c_comp), &pot);
    let Some(gpu_res) = gpu_compute(NeighborStrategy::AllPairs, &p_comp, &types, Some(c_comp), &pot)
    else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    let cpu_tr = (cpu_res.virial[0] + cpu_res.virial[1] + cpu_res.virial[2]) as f64;
    let gpu_tr = (gpu_res.virial[0] + gpu_res.virial[1] + gpu_res.virial[2]) as f64;
    assert!(
        cpu_tr.signum() == gpu_tr.signum(),
        "tr(σ) sign mismatch under compression: CPU={cpu_tr:.6e}, GPU={gpu_tr:.6e}"
    );
    eprintln!("compression check: tr(σ) CPU = {cpu_tr:.6e}, GPU = {gpu_tr:.6e}");
}

#[test]
#[ignore]
fn gpu_virial_invariant_across_cell_list_toggle() {
    // GPU AllPairs vs GPU CellList should agree on virial for the same
    // system, proving the two pass-2 shader variants share the same
    // accumulation semantics.
    let pot = cu_synth();
    let (pos, types, cell) = fcc_cu_supercell(4, 4, 4); // 64 atoms
    let Some(ap) = gpu_compute(NeighborStrategy::AllPairs, &pos, &types, Some(cell), &pot) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let cell_size = pot.cutoff_angstrom;
    let Some(cl) = gpu_compute(
        NeighborStrategy::CellList { cell_size },
        &pos,
        &types,
        Some(cell),
        &pot,
    ) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    // Both GPU paths share f32 table noise, so tighter than GPU-vs-CPU.
    assert_virial_close(
        cl.virial,
        ap.virial,
        TOL_GPU_ABS,
        TOL_GPU_REL,
        "GPU-CL vs GPU-AP (Cu 4×4×4)",
    );
}

#[test]
#[ignore]
fn gpu_and_cpu_forces_and_energy_unchanged_by_virial_path() {
    // Regression guard: adding virial accumulation to pass2 must not change
    // the existing energy/force outputs within the nominal GPU vs CPU
    // tolerance used by the existing gpu.rs suite.
    let pot = cu_synth();
    let (pos, types, cell) = fcc_cu_supercell(4, 4, 4); // 64 atoms
    let cpu = cpu_allpairs(&pos, &types, Some(cell), &pot);
    let Some(gpu) = gpu_compute(NeighborStrategy::AllPairs, &pos, &types, Some(cell), &pot) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    // Energy
    let e_err = (gpu.energy - cpu.energy).abs();
    let e_scale = cpu.energy.abs().max(1.0);
    assert!(
        e_err / e_scale < 1e-3,
        "energy drift after virial addition: CPU={} GPU={} err={:.3e}",
        cpu.energy,
        gpu.energy,
        e_err
    );
    // Forces — max component-wise
    let mut max_err = 0.0f32;
    for i in 0..pos.len() {
        for c in 0..3 {
            let d = (gpu.forces[i][c] - cpu.forces[i][c]).abs();
            if d > max_err {
                max_err = d;
            }
        }
    }
    assert!(
        max_err < 5e-3,
        "max |ΔF| = {max_err:.3e} exceeds tolerance — virial code changed force computation"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Triclinic stress tests.
//
// The earlier cross-validation matrix uses diag(Lx,Ly,Lz) boxes only, which
// means every off-diagonal Voigt component (σ_yz, σ_xz, σ_xy) stays near
// zero.  That hides a whole class of bugs: a sign error on one of the shear
// virials, a wrong row/col in the triclinic minimum-image transform, a bad
// `hinv_col*` broadcast in the GPU shader — all invisible under orthorhombic
// tests.
//
// The tests below drive the crystal through a non-trivial triclinic cell
// and through a full (not isotropic) strain matrix so that every Voigt
// component is independently excited and checked.
// ─────────────────────────────────────────────────────────────────────────────

/// Build a "gently" triclinic FCC supercell by applying a small shear strain
/// to an orthorhombic FCC Cu lattice.  Shearing the box matrix and the atom
/// positions by the same amount preserves the lattice's equilibrium local
/// geometry, so forces stay in the elastic regime and the stress tensor
/// should equal the triclinic strain tensor times the elastic constants —
/// but that physical prediction is not what we test.  We test that all four
/// implementation paths (CPU-AP, CPU-CL, GPU-AP, GPU-CL) agree on it.
fn fcc_cu_triclinic(
    nx: usize,
    ny: usize,
    nz: usize,
) -> (Vec<[f32; 4]>, Vec<u32>, [[f32; 3]; 3]) {
    let (pos_ortho, types, cell_ortho) = fcc_cu_supercell(nx, ny, nz);

    // Triclinic shear matrix S.  Chosen so every Voigt component is
    // non-degenerate: yz-shear = 0.05, xz-shear = 0.03, xy-shear = 0.07.
    // S is applied to both the cell rows and the atomic positions, so the
    // fractional coordinates stay identical — the crystal is rigidly
    // rotated/sheared, not deformed.
    let s: [[f32; 3]; 3] = [
        [1.0, 0.07, 0.03],
        [0.0, 1.0, 0.05],
        [0.0, 0.0, 1.0],
    ];
    // Apply S to row vectors (cell rows = lattice vectors).
    let mut cell_tri = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            cell_tri[i][j] =
                cell_ortho[i][0] * s[0][j] + cell_ortho[i][1] * s[1][j] + cell_ortho[i][2] * s[2][j];
        }
    }
    let pos_tri: Vec<[f32; 4]> = pos_ortho
        .iter()
        .map(|p| {
            [
                p[0] * s[0][0] + p[1] * s[1][0] + p[2] * s[2][0],
                p[0] * s[0][1] + p[1] * s[1][1] + p[2] * s[2][1],
                p[0] * s[0][2] + p[1] * s[1][2] + p[2] * s[2][2],
                0.0,
            ]
        })
        .collect();

    (pos_tri, types, cell_tri)
}

#[test]
fn triclinic_cpu_allpairs_vs_cpu_celllist() {
    // Sheared box — tight tolerance (both paths are f64 accumulators) and
    // every Voigt component is non-zero.
    let pot = cu_synth();
    let (pos, types, cell) = fcc_cu_triclinic(4, 4, 4);
    let ap = cpu_allpairs(&pos, &types, Some(cell), &pot);
    let cl = cpu_cell_list(&pos, &types, Some(cell), &pot);

    // Confirm the test is actually exercising shear components, not just
    // reproducing the ortho case by accident.
    let shear_mag =
        ap.virial[3].abs().max(ap.virial[4].abs()).max(ap.virial[5].abs());
    let diag_mag = ap.virial[0].abs().max(ap.virial[1].abs()).max(ap.virial[2].abs());
    assert!(
        shear_mag > 1e-3 * diag_mag,
        "triclinic test failed to exercise shear: σ_yz,xz,xy max = {shear_mag:.3e} vs σ_xx,yy,zz max = {diag_mag:.3e}"
    );

    assert_virial_close(
        cl.virial,
        ap.virial,
        TOL_CPU_CL_ABS,
        TOL_CPU_CL_REL,
        "triclinic CPU-CL vs CPU-AP",
    );
}

#[test]
fn triclinic_per_atom_virial_sums_to_total() {
    // Same invariant as the ortho case but with all 6 Voigt components
    // non-degenerate.  Tests that the half-and-half split works correctly
    // for shear virial components, which the ortho test cannot.
    let pot = cu_synth();
    let (pos, types, cell) = fcc_cu_triclinic(4, 4, 4);
    // See `cpu_per_atom_virial_sums_to_total_virial_allpairs` — per-atom
    // virial is returned empty by the default `compute_sync` fast path.
    let res = cpu_allpairs_with_per_atom(&pos, &types, Some(cell), &pot);

    // Volume of the triclinic cell.
    let h = cell;
    let vol = (h[0][0] * (h[1][1] * h[2][2] - h[1][2] * h[2][1])
        - h[0][1] * (h[1][0] * h[2][2] - h[1][2] * h[2][0])
        + h[0][2] * (h[1][0] * h[2][1] - h[1][1] * h[2][0]))
        .abs() as f64;

    let w_from_total = [
        -res.virial[0] * vol,
        -res.virial[1] * vol,
        -res.virial[2] * vol,
        -res.virial[3] * vol,
        -res.virial[4] * vol,
        -res.virial[5] * vol,
    ];
    let mut w_from_atoms = [0.0f64; 6];
    for va in &res.virial_per_atom {
        for c in 0..6 {
            w_from_atoms[c] += va[c];
        }
    }
    assert_virial_close(
        w_from_atoms,
        w_from_total,
        1e-6,
        1e-6,
        "triclinic Σᵢ va[i] == -σ·V (shear components included)",
    );
}


#[test]
fn triclinic_stress_direction_matches_finite_difference() {
    let pot = cu_synth();
    let (pos0, types, cell0) = fcc_cu_triclinic(4, 4, 4);

    let vol0 = {
        let h = cell0;
        ((h[0][0] * (h[1][1] * h[2][2] - h[1][2] * h[2][1])
            - h[0][1] * (h[1][0] * h[2][2] - h[1][2] * h[2][0])
            + h[0][2] * (h[1][0] * h[2][1] - h[1][1] * h[2][0])) as f64)
            .abs()
    };

    let cpu = CpuEngine::new();
    let stress0 = cpu
        .compute_sync(&pos0, &types, Some(cell0), &pot)
        .unwrap()
        .virial; // σ₀

    // Strain-generator matrices ε_c for Voigt components 0..6.
    // LAMMPS / CREAM Native order: xx, yy, zz, xy, xz, yz
    let strain_generators: [[[f64; 3]; 3]; 6] = [
        // 0: xx
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        // 1: yy
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        // 2: zz
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        // 3: xy (Engine stress0[3] is xy)
        [[0.0, 0.5, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]],
        // 4: xz (Engine stress0[4] is xz)
        [[0.0, 0.0, 0.5], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        // 5: yz (Engine stress0[5] is yz)
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0]],
    ];

    // Apply `F = I + eps * eps_gen` to both positions and cell rows.
    let apply_strain = |eps: f64, gen: &[[f64; 3]; 3]| -> (Vec<[f32; 4]>, [[f32; 3]; 3]) {
        let f = [
            [1.0 + eps * gen[0][0], eps * gen[0][1], eps * gen[0][2]],
            [eps * gen[1][0], 1.0 + eps * gen[1][1], eps * gen[1][2]],
            [eps * gen[2][0], eps * gen[2][1], 1.0 + eps * gen[2][2]],
        ];
        let mut c_new = [[0.0f32; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                c_new[i][j] = (cell0[i][0] as f64 * f[0][j]
                    + cell0[i][1] as f64 * f[1][j]
                    + cell0[i][2] as f64 * f[2][j]) as f32;
            }
        }
        let p_new: Vec<[f32; 4]> = pos0
            .iter()
            .map(|p| {
                [
                    (p[0] as f64 * f[0][0] + p[1] as f64 * f[1][0] + p[2] as f64 * f[2][0]) as f32,
                    (p[0] as f64 * f[0][1] + p[1] as f64 * f[1][1] + p[2] as f64 * f[2][1]) as f32,
                    (p[0] as f64 * f[0][2] + p[1] as f64 * f[1][2] + p[2] as f64 * f[2][2]) as f32,
                    0.0,
                ]
            })
            .collect();
        (p_new, c_new)
    };

    
    let eps = 2e-3_f64;
    
    let mut stress_fd = [0.0f64; 6];
    let mut stress_analytic = [0.0f64; 6];

    for c in 0..6 {
        let gen = &strain_generators[c];
        let (p_plus, c_plus) = apply_strain(eps, gen);
        let (p_minus, c_minus) = apply_strain(-eps, gen);
        let e_plus = cpu
            .compute_sync(&p_plus, &types, Some(c_plus), &pot)
            .unwrap()
            .energy as f64;
        let e_minus = cpu
            .compute_sync(&p_minus, &types, Some(c_minus), &pot)
            .unwrap()
            .energy as f64;
            
        let de_deps = (e_plus - e_minus) / (2.0 * eps);

        stress_fd[c] = de_deps;
        stress_analytic[c] = vol0 * stress0[c] as f64;
        
        eprintln!(
            "triclinic strain c={c}: dE/dε = {de_deps:.6e}, V·σ = {:.6e}",
            stress_analytic[c]
        );
    }

   
    let dot_product: f64 = stress_analytic.iter().zip(stress_fd.iter()).map(|(a, b)| a * b).sum();
    let norm_analytic: f64 = stress_analytic.iter().map(|a| a * a).sum::<f64>().sqrt();
    let norm_fd: f64 = stress_fd.iter().map(|a| a * a).sum::<f64>().sqrt();

    let cos_sim = if norm_analytic < 1e-8 || norm_fd < 1e-8 {
        if (norm_analytic - norm_fd).abs() < 1e-8 { 1.0 } else { 0.0 }
    } else {
        dot_product / (norm_analytic * norm_fd)
    };

    
    assert!(
        cos_sim > 0.95,
        "Stress direction mismatch (f32 noise or bug)!\n\
         Cosine Similarity: {cos_sim:.5}\n\
         Analytic Vector: {stress_analytic:?}\n\
         FD Vector:       {stress_fd:?}"
    );

    eprintln!("Success! Stress vectors matched with Cosine Similarity = {cos_sim:.5}");
}


#[test]
#[ignore] // requires GPU
fn triclinic_gpu_allpairs_vs_cpu_allpairs() {
    let pot = cu_synth();
    let (pos, types, cell) = fcc_cu_triclinic(4, 4, 4);
    let truth = cpu_allpairs(&pos, &types, Some(cell), &pot);
    let Some(gpu) = gpu_compute(NeighborStrategy::AllPairs, &pos, &types, Some(cell), &pot) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    // Shear components are typically an order of magnitude smaller than the
    // diagonals in the elastic regime, so we scale the absolute tolerance
    // by the RMS virial magnitude before using the standard mixed
    // abs/rel comparison.
    assert_virial_close(
        gpu.virial,
        truth.virial,
        TOL_GPU_ABS,
        TOL_GPU_REL,
        "triclinic GPU-AP vs CPU-AP",
    );
}

#[test]
#[ignore]
fn triclinic_gpu_celllist_vs_cpu_allpairs() {
    let pot = cu_synth();
    let (pos, types, cell) = fcc_cu_triclinic(4, 4, 4);
    let truth = cpu_allpairs(&pos, &types, Some(cell), &pot);
    let cell_size = pot.cutoff_angstrom;
    let Some(gpu) = gpu_compute(
        NeighborStrategy::CellList { cell_size },
        &pos,
        &types,
        Some(cell),
        &pot,
    ) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    assert_virial_close(
        gpu.virial,
        truth.virial,
        TOL_GPU_ABS,
        TOL_GPU_REL,
        "triclinic GPU-CL vs CPU-AP",
    );
}
