//! CPU O(N²) EAM reference implementation (multi-element, triclinic-PBC-aware).
//!
//! # Periodic Boundary Conditions (PBC)
//! `cell: Option<[[f64; 3]; 3]>` — rows are lattice vectors a, b, c (ASE/LAMMPS convention):
//! ```text
//! h = [[ax, ay, az],   ← a
//!      [bx, by, bz],   ← b
//!      [cx, cy, cz]]   ← c
//! ```
//! Fractional coords: **s = d @ H⁻¹**,  Cartesian from fractional: **d = s @ H**.
//! For orthorhombic boxes use [`ortho_cell`].
//!
//! All arithmetic is performed in `f64`.  Potential tables remain stored as
//! `f32` in [`EamPotential`]; table values are widened to `f64` on read so
//! the interpolation benefits from higher precision without changing the
//! on-disk / GPU representation.

use crate::potential::eam::EamPotential;

/// PBC matrices: (cell H, inverse H⁻¹).
type PbcMatrices = ([[f64; 3]; 3], [[f64; 3]; 3]);

// ── Output type ────────────────────────────────────────────────────────────────

/// Result of CPU EAM computation (all quantities in `f64`).
#[derive(Debug, Clone)]
pub struct CpuResult {
    /// Force on each atom [eV/Å].
    pub forces: Vec<[f64; 3]>,
    /// Total potential energy [eV].
    pub energy: f64,
    /// Per-atom energy [eV].
    pub energy_per_atom: Vec<f64>,
    /// Electron density at each atom site.
    pub densities: Vec<f64>,
}

// ── Constants ──────────────────────────────────────────────────────────────────

pub(crate) const MIN_DIST_SQ: f64 = 1e-4;

// ── Table lookup (kept for backward compat; reference now uses splines) ─────

/// Linear interpolation into a `f32` table, computed in `f64`.
///
/// Retained as a crate-visible helper because `cpu_engine.rs` callers import
/// it; the reference implementation itself has switched to evaluating the
/// [`crate::potential::spline::CubicSpline`] companions on the potential
/// directly, which is strictly more accurate.
#[inline]
#[allow(dead_code)]
pub(crate) fn linear_interp(table: &[f32], idx_f: f64) -> f64 {
    let n = table.len();
    let idx = idx_f as usize;
    let frac = idx_f - idx as f64;
    let i0 = idx.min(n.saturating_sub(2));
    let v0 = table[i0] as f64;
    let v1 = table[(i0 + 1).min(n - 1)] as f64;
    v0 + frac * (v1 - v0)
}

#[inline]
#[allow(dead_code)]
pub(crate) fn lookup_by_r(table: &[f32], dr_inv: f64, r: f64) -> f64 {
    linear_interp(table, r * dr_inv)
}

#[inline]
#[allow(dead_code)]
pub(crate) fn lookup_by_rho(table: &[f32], drho_inv: f64, rho: f64) -> f64 {
    linear_interp(table, rho * drho_inv)
}

// ── 3×3 matrix helpers ─────────────────────────────────────────────────────────

/// Construct an orthorhombic cell matrix from box lengths.
/// Rows are lattice vectors: a=(Lx,0,0), b=(0,Ly,0), c=(0,0,Lz).
pub fn ortho_cell(lx: f64, ly: f64, lz: f64) -> [[f64; 3]; 3] {
    [[lx, 0.0, 0.0], [0.0, ly, 0.0], [0.0, 0.0, lz]]
}

/// Invert a 3×3 matrix (rows = lattice vectors).
/// Returns `None` if singular (|det| < 1e-30).
pub fn mat3_inv(h: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let [[h00, h01, h02], [h10, h11, h12], [h20, h21, h22]] = *h;

    let det = h00 * (h11 * h22 - h12 * h21) - h01 * (h10 * h22 - h12 * h20)
        + h02 * (h10 * h21 - h11 * h20);

    if det.abs() < 1e-30 {
        return None;
    }
    let d = 1.0 / det;

    Some([
        [
            (h11 * h22 - h12 * h21) * d,
            -(h01 * h22 - h02 * h21) * d,
            (h01 * h12 - h02 * h11) * d,
        ],
        [
            -(h10 * h22 - h12 * h20) * d,
            (h00 * h22 - h02 * h20) * d,
            -(h00 * h12 - h02 * h10) * d,
        ],
        [
            (h10 * h21 - h11 * h20) * d,
            -(h00 * h21 - h01 * h20) * d,
            (h00 * h11 - h01 * h10) * d,
        ],
    ])
}

/// Minimum image convention for a triclinic cell.
///
/// 1. `s = d @ H⁻¹`  (s[i] = Σⱼ d[j] * h_inv[j][i])
/// 2. `s -= round(s)`
/// 3. `d' = s @ H`  (d'[j] = Σᵢ s[i] * h[i][j])
#[inline]
pub fn min_image_mat(d: [f64; 3], h: &[[f64; 3]; 3], h_inv: &[[f64; 3]; 3]) -> [f64; 3] {
    let s = [
        d[0] * h_inv[0][0] + d[1] * h_inv[1][0] + d[2] * h_inv[2][0],
        d[0] * h_inv[0][1] + d[1] * h_inv[1][1] + d[2] * h_inv[2][1],
        d[0] * h_inv[0][2] + d[1] * h_inv[1][2] + d[2] * h_inv[2][2],
    ];
    let sf = [
        s[0] - s[0].round(),
        s[1] - s[1].round(),
        s[2] - s[2].round(),
    ];
    [
        sf[0] * h[0][0] + sf[1] * h[1][0] + sf[2] * h[2][0],
        sf[0] * h[0][1] + sf[1] * h[1][1] + sf[2] * h[2][1],
        sf[0] * h[0][2] + sf[1] * h[1][2] + sf[2] * h[2][2],
    ]
}

// ── Main compute function ──────────────────────────────────────────────────────

/// Computes EAM forces and energy on CPU (O(N²) all-pairs, multi-element).
///
/// * `cell` — `Some([[ax,ay,az],[bx,by,bz],[cx,cy,cz]])` for PBC, `None` for non-periodic.
///
/// All arithmetic is performed in `f64`.
pub fn compute_eam_cpu(
    potential: &EamPotential,
    positions: &[[f64; 3]],
    atom_types: &[u32],
    cell: Option<[[f64; 3]; 3]>,
) -> CpuResult {
    let n = positions.len();
    let n_elem = potential.elements.len();

    assert_eq!(
        atom_types.len(),
        n,
        "atom_types length {} ≠ positions length {}",
        atom_types.len(),
        n
    );
    for (i, &t) in atom_types.iter().enumerate() {
        assert!(
            (t as usize) < n_elem,
            "atom_types[{i}]={t} is out of range (n_elem={n_elem})"
        );
    }

    let cutoff_sq = (potential.cutoff_angstrom as f64).powi(2);

    // Shorthands — evaluate the f64 splines directly.  These replace the
    // old `lookup_by_r(potential.*_tables, …)` calls that consumed the f32
    // tables via linear interpolation.  Using the splines gives the
    // reference the full f64 accuracy of the tabulated potential, which
    // is what lets the CPU / GPU f32 paths be compared against a stable
    // ground truth independent of their own grid density.
    let rho_eval = |elem: usize, r: f64| -> f64 { potential.rho_splines[elem].eval(r) };
    let drho_eval = |elem: usize, r: f64| -> f64 { potential.rho_splines[elem].eval_deriv(r) };
    let embed_eval = |elem: usize, rho: f64| -> f64 { potential.embed_splines[elem].eval(rho) };
    let dembed_eval =
        |elem: usize, rho: f64| -> f64 { potential.embed_splines[elem].eval_deriv(rho) };
    let phi_eval = |pidx: usize, r: f64| -> f64 { potential.pair_splines[pidx].eval(r) };
    let dphi_eval = |pidx: usize, r: f64| -> f64 { potential.pair_splines[pidx].eval_deriv(r) };

    // Validate cell: minimum image requires L_min > 2 * cutoff.
    if let Some(ref h) = cell {
        let twice_cut = 2.0 * potential.cutoff_angstrom as f64;
        let a = h[0];
        let b = h[1];
        let c = h[2];
        let vol = (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0]))
            .abs();
        let cross_bc = [
            b[1] * c[2] - b[2] * c[1],
            b[2] * c[0] - b[0] * c[2],
            b[0] * c[1] - b[1] * c[0],
        ];
        let cross_ac = [
            a[1] * c[2] - a[2] * c[1],
            a[2] * c[0] - a[0] * c[2],
            a[0] * c[1] - a[1] * c[0],
        ];
        let cross_ab = [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ];
        let norm = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let l_min = (vol / norm(cross_bc))
            .min(vol / norm(cross_ac))
            .min(vol / norm(cross_ab));
        assert!(
            l_min > twice_cut,
            "Cell too small: shortest height {l_min:.3} Å < 2×cutoff = {twice_cut:.3} Å"
        );
    }

    let pbc: Option<PbcMatrices> = cell.and_then(|h| mat3_inv(&h).map(|hi| (h, hi)));

    let displace = |i: usize, j: usize| -> [f64; 3] {
        let raw = [
            positions[j][0] - positions[i][0],
            positions[j][1] - positions[i][1],
            positions[j][2] - positions[i][2],
        ];
        match &pbc {
            None => raw,
            Some((h, h_inv)) => min_image_mat(raw, h, h_inv),
        }
    };

    // ── Pass 1: ρᵢ = Σⱼ f_β(rᵢⱼ) ───────────────────────────────────────────
    let mut densities = vec![0.0_f64; n];
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        for j in 0..n {
            if j == i {
                continue;
            }
            let [dx, dy, dz] = displace(i, j);
            let r_sq = dx * dx + dy * dy + dz * dz;
            if r_sq < cutoff_sq && r_sq > MIN_DIST_SQ {
                let r = r_sq.sqrt();
                densities[i] += rho_eval(atom_types[j] as usize, r);
            }
        }
    }

    // ── Pass 2: forces + per-atom energy ─────────────────────────────────────
    let mut forces = vec![[0.0_f64; 3]; n];
    let mut energy_per_atom = vec![0.0_f64; n];

    for i in 0..n {
        let type_i = atom_types[i] as usize;
        let rho_i = densities[i];
        let df_i = dembed_eval(type_i, rho_i);
        let embed_e = embed_eval(type_i, rho_i);
        let mut pair_e = 0.0_f64;

        for j in 0..n {
            if j == i {
                continue;
            }
            let [dx, dy, dz] = displace(i, j);
            let r_sq = dx * dx + dy * dy + dz * dz;
            if r_sq < cutoff_sq && r_sq > MIN_DIST_SQ {
                let r = r_sq.sqrt();
                let r_inv = 1.0 / r;
                let type_j = atom_types[j] as usize;
                let pidx = EamPotential::pair_index(type_i, type_j, n_elem);

                let df_j = dembed_eval(type_j, densities[j]);
                let df_beta_dr = drho_eval(type_j, r);
                let df_alpha_dr = drho_eval(type_i, r);
                let dphi_dr = dphi_eval(pidx, r);
                let phi = phi_eval(pidx, r);

                let coeff = df_i * df_beta_dr + df_j * df_alpha_dr + dphi_dr;
                forces[i][0] += coeff * dx * r_inv;
                forces[i][1] += coeff * dy * r_inv;
                forces[i][2] += coeff * dz * r_inv;
                pair_e += 0.5 * phi;
            }
        }
        energy_per_atom[i] = embed_e + pair_e;
    }

    let energy: f64 = energy_per_atom.iter().sum();
    CpuResult {
        forces,
        energy,
        energy_per_atom,
        densities,
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::potential::eam::synthetic_cu_alloy_src;

    fn make_pot() -> EamPotential {
        EamPotential::from_str(&synthetic_cu_alloy_src(100, 100, 0.05, 0.01, 4.5)).unwrap()
    }

    fn cu4_fcc() -> (Vec<[f64; 3]>, Vec<u32>) {
        let a = 3.615_f64;
        (
            vec![
                [0.0, 0.0, 0.0],
                [a / 2.0, a / 2.0, 0.0],
                [a / 2.0, 0.0, a / 2.0],
                [0.0, a / 2.0, a / 2.0],
            ],
            vec![0u32; 4],
        )
    }

    #[test]
    fn forces_are_finite() {
        let pot = make_pot();
        let (pos, types) = cu4_fcc();
        let res = compute_eam_cpu(&pot, &pos, &types, None);
        for (i, f) in res.forces.iter().enumerate() {
            for (c, &v) in f.iter().enumerate() {
                assert!(v.is_finite(), "forces[{i}][{c}]={v} is non-finite");
            }
        }
        assert!(res.energy.is_finite());
    }

    #[test]
    fn newton_third_law() {
        let pot = make_pot();
        let pos = vec![
            [0.0_f64, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 1.7, 0.0],
            [0.5, 0.8, 1.4],
        ];
        let types = [0u32; 4];
        let res = compute_eam_cpu(&pot, &pos, &types, None);
        let sum: [f64; 3] = res
            .forces
            .iter()
            .fold([0.0_f64; 3], |a, f| [a[0] + f[0], a[1] + f[1], a[2] + f[2]]);
        for (c, &s) in sum.iter().enumerate() {
            assert!(s.abs() < 1e-3, "sum_F[{c}]={s:.2e} (Newton 3rd law)");
        }
    }

    #[test]
    fn densities_non_negative() {
        let pot = make_pot();
        let res = compute_eam_cpu(
            &pot,
            &[[0.0_f64, 0.0, 0.0], [2.5, 0.0, 0.0]],
            &[0u32; 2],
            None,
        );
        for (i, &rho) in res.densities.iter().enumerate() {
            assert!(rho >= 0.0, "density[{i}]={rho} < 0");
        }
    }

    // ── mat3_inv ───────────────────────────────────────────────────────────

    #[test]
    fn mat3_inv_identity() {
        let id = [[1.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let inv = mat3_inv(&id).unwrap();
        #[allow(clippy::needless_range_loop)]
        for i in 0..3 {
            for j in 0..3 {
                let exp = if i == j { 1.0_f64 } else { 0.0 };
                assert!(
                    (inv[i][j] - exp).abs() < 1e-12,
                    "inv[{i}][{j}]={} exp={exp}",
                    inv[i][j]
                );
            }
        }
    }

    #[test]
    fn mat3_inv_diagonal() {
        let h = [[2.0_f64, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 5.0]];
        let inv = mat3_inv(&h).unwrap();
        assert!((inv[0][0] - 0.5).abs() < 1e-12);
        assert!((inv[1][1] - 1.0 / 3.0).abs() < 1e-12);
        assert!((inv[2][2] - 0.2).abs() < 1e-12);
    }

    #[test]
    fn mat3_inv_roundtrip() {
        let h = [[3.0_f64, 1.0, 0.5], [0.0, 4.0, 1.0], [0.0, 0.0, 2.0]];
        let hi = mat3_inv(&h).unwrap();
        #[allow(clippy::needless_range_loop)]
        for i in 0..3 {
            for k in 0..3 {
                let dot: f64 = (0..3).map(|j| h[i][j] * hi[j][k]).sum();
                let exp = if i == k { 1.0_f64 } else { 0.0 };
                assert!(
                    (dot - exp).abs() < 1e-12,
                    "(H@H⁻¹)[{i}][{k}]={dot:.14} exp={exp}"
                );
            }
        }
    }

    #[test]
    fn mat3_inv_singular_returns_none() {
        let h = [[1.0_f64, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(mat3_inv(&h).is_none());
    }

    // ── min_image_mat ──────────────────────────────────────────────────────

    #[test]
    fn min_image_mat_ortho_matches_scalar() {
        let lx = 10.0_f64;
        let h = ortho_cell(lx, lx, lx);
        let hi = mat3_inv(&h).unwrap();
        for &d in &[1.0_f64, -1.0, 3.5, -4.9] {
            let result = min_image_mat([d, 0.0, 0.0], &h, &hi);
            let expected = d - lx * ((d / lx) + 0.5).floor();
            assert!(
                (result[0] - expected).abs() < 1e-12,
                "d={d}: got {}, expected {expected}",
                result[0]
            );
        }
    }

    #[test]
    fn min_image_mat_antisymmetry() {
        let h = ortho_cell(5.0, 7.0, 9.0);
        let hi = mat3_inv(&h).unwrap();
        for d in [[-3.0_f64, 2.1, -4.3], [1.5, -3.0, 3.8_f64]] {
            let v1 = min_image_mat(d, &h, &hi);
            let v2 = min_image_mat([-d[0], -d[1], -d[2]], &h, &hi);
            for c in 0..3 {
                assert!(
                    (v1[c] + v2[c]).abs() < 1e-12,
                    "antisymmetry failure axis {c}: {}+{} != 0",
                    v1[c],
                    v2[c]
                );
            }
        }
    }

    #[test]
    fn min_image_mat_triclinic_rhombus() {
        let h = [
            [5.0_f64, 0.0, 0.0],
            [2.5, 4.330_127_018_922_193, 0.0],
            [0.0, 0.0, 10.0],
        ];
        let hi = mat3_inv(&h).unwrap();
        let ra = min_image_mat([5.0, 0.0, 0.0], &h, &hi);
        #[allow(clippy::needless_range_loop)]
        for c in 0..3 {
            assert!(ra[c].abs() < 1e-10, "a→0 failed axis {c}: {}", ra[c]);
        }
        let rb = min_image_mat([2.5, 4.330_127_018_922_193, 0.0], &h, &hi);
        #[allow(clippy::needless_range_loop)]
        for c in 0..3 {
            assert!(rb[c].abs() < 1e-10, "b→0 failed axis {c}: {}", rb[c]);
        }
    }

    // ── PBC physics ────────────────────────────────────────────────────────

    #[test]
    fn pbc_energy_matches_non_pbc_when_no_wrapping() {
        let pot = make_pot();
        let (pos, types) = cu4_fcc();
        let r_pbc = compute_eam_cpu(&pot, &pos, &types, Some(ortho_cell(100.0, 100.0, 100.0)));
        let r_none = compute_eam_cpu(&pot, &pos, &types, None);
        assert!((r_pbc.energy - r_none.energy).abs() < 1e-8);
    }

    #[test]
    fn triclinic_vs_ortho_for_cubic_cell() {
        let pot = make_pot();
        let (pos, types) = cu4_fcc();
        let l = 10.0_f64;
        let r_o = compute_eam_cpu(&pot, &pos, &types, Some(ortho_cell(l, l, l)));
        let r_t = compute_eam_cpu(
            &pot,
            &pos,
            &types,
            Some([[l, 0.0, 0.0], [0.0, l, 0.0], [0.0, 0.0, l]]),
        );
        assert!(
            (r_o.energy - r_t.energy).abs() < 1e-10,
            "ortho vs triclinic: {:.10} vs {:.10}",
            r_o.energy,
            r_t.energy
        );
    }

    #[test]
    fn cu_fcc_4atom_acceptance() {
        let pot = make_pot();
        let (pos, types) = cu4_fcc();
        let res = compute_eam_cpu(&pot, &pos, &types, None);
        let sum: [f64; 3] = res
            .forces
            .iter()
            .fold([0.0_f64; 3], |a, f| [a[0] + f[0], a[1] + f[1], a[2] + f[2]]);
        let rms = (sum.iter().map(|v| v * v).sum::<f64>() / 3.0).sqrt();
        assert!(rms < 1e-3, "Newton RMS={rms:.2e} (limit 1e-3)");
    }
}
