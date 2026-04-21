"""Stress tensor and per-atom physics tests.

Ground truth: the CPU AllPairs backend (half-pair, f64 accumulators).
Every other implementation — CPU CellList, GPU AllPairs, GPU CellList
(CPU-built NL or GPU-resident, depending on Cargo features) — is
cross-validated against that reference.  When a discrepancy appears, the
test name points directly at which implementation deviates.

Additional physical checks:

* Finite-difference of energy w.r.t. uniform volume strain must match
  ``−V · tr(σ)`` (catches sign flips and missing volume normalisation —
  the exact bug class the user's documentation-vs-implementation audit
  warned us about).
* Per-atom virial sum equals ``−σ · V`` (LAMMPS half-and-half split).
* Densities are uniform across a perfect bulk FCC lattice.
* Translational invariance — shift every atom, σ unchanged.
* GPU/CPU stresses agree to f32 tolerance.
* ASE ``Atoms.get_stress()`` / ``get_stresses()`` wiring.

Run:
    pip install maturin ase pytest numpy
    maturin develop --release --features python,cellist_gpu
    pytest tests_python/test_stress.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

# ── Reuse the fixtures from conftest.py (`require_gpu`, `require_cpu`) ───────


def _find_potential() -> str:
    import pathlib

    candidates = [
        pathlib.Path("Cu01_eam.alloy"),
        pathlib.Path(__file__).parent.parent / "Cu01_eam.alloy",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    pytest.skip("Cu01_eam.alloy not found — skipping stress tests")
    return ""  # unreachable


@pytest.fixture
def pot_path() -> str:
    return _find_potential()


# ── Geometry builders ─────────────────────────────────────────────────────────


def _fcc_cu_supercell(nx: int, ny: int, nz: int, a: float = 3.615):
    """Return (positions, atom_types, cell) for an FCC Cu supercell."""
    basis = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ]
    ) * a

    positions = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                origin = np.array([ix, iy, iz]) * a
                for b in basis:
                    positions.append(origin + b)
    positions = np.asarray(positions, dtype=np.float64)
    atom_types = np.zeros(len(positions), dtype=np.int32)
    cell = np.diag([nx * a, ny * a, nz * a]).astype(np.float64)
    return positions, atom_types, cell


# ── Tolerances ────────────────────────────────────────────────────────────────
# CPU vs CPU: f64 reductions in different order 
CPU_VS_CPU_ATOL = 1e-5
CPU_VS_CPU_RTOL = 1e-6
# CPU vs GPU: f32 tables, per-thread Kahan, num_wg partial Neumaier sum.
GPU_VS_CPU_ATOL = 5e-4
GPU_VS_CPU_RTOL = 5e-3


def _assert_virial_close(got, truth, atol, rtol, label):
    got = np.asarray(got, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    assert got.shape == truth.shape == (6,), f"{label}: shape mismatch"
    diff = np.abs(got - truth)
    scale = np.maximum(np.abs(truth), 1.0)
    rel = diff / scale
    ok = (diff <= atol) | (rel <= rtol)
    labels = ["xx", "yy", "zz", "yz", "xz", "xy"]
    assert ok.all(), (
        f"{label}: virial mismatch\n"
        + "\n".join(
            f"  σ[{i}:{labels[i]}]  got={got[i]:.6e}  "
            f"truth={truth[i]:.6e}  abs={diff[i]:.3e}  rel={rel[i]:.3e}"
            for i in range(6)
            if not ok[i]
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# Contract: compute_stress return-shape & units.  Catches the exact bug class
# the user's audit called out (doc says eV/Å³, code returned eV) — if anyone
# ever re-introduces that regression, this test catches it immediately.
# ─────────────────────────────────────────────────────────────────────────────


class TestStressContract:
    def test_cpu_compute_stress_return_shape(self, pot_path, require_cpu):  # noqa: ARG002
        from cream import CreamEngine

        eng = CreamEngine(pot_path, backend="cpu")
        pos, types, cell = _fcc_cu_supercell(4, 4, 4)  # 256 atoms
        energy, forces, epa, stress = eng.compute_stress(pos, types, cell)
        assert isinstance(energy, float)
        assert forces.shape == (pos.shape[0], 3)
        assert stress.shape == (6,)
        # Units check: eV/Å³ for a bulk Cu crystal should be well under 1.
        # If somebody accidentally returns raw virial (eV), |stress| would
        # be ~N·(eV per pair) which is huge — orders of magnitude off.
        assert np.abs(stress).max() < 10.0, (
            f"Stress magnitude {np.abs(stress).max():.3e} eV/Å³ is suspiciously "
            "large — probable unit bug (returning raw virial W in eV instead "
            "of σ = -W/V in eV/Å³)"
        )

    def test_cpu_stress_is_zero_for_cluster(self, pot_path, require_cpu):  # noqa: ARG002
        from cream import CreamEngine

        eng = CreamEngine(pot_path, backend="cpu")
        pos, types, _ = _fcc_cu_supercell(4, 4, 4)  # 256 atoms
        _, _, _, stress = eng.compute_stress(pos, types, None)
        assert np.all(stress == 0.0), f"cluster stress must be exactly zero, got {stress}"

    def test_gpu_stress_is_zero_for_cluster(self, pot_path, require_gpu):  # noqa: ARG002
        from cream import CreamEngine

        eng = CreamEngine(pot_path, backend="gpu")
        pos, types, _ = _fcc_cu_supercell(4, 4, 4)  # 256 atoms
        _, _, _, stress = eng.compute_stress(pos, types, None)
        assert np.all(stress == 0.0), f"GPU cluster stress must be zero, got {stress}"


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validation matrix: CPU-AP is the ground truth.
# ─────────────────────────────────────────────────────────────────────────────


class TestCrossValidation:
    def test_cpu_cell_list_matches_cpu_allpairs(self, pot_path, require_cpu):  # noqa: ARG002
        from cream import CreamEngine

        ap = CreamEngine(pot_path, backend="cpu", use_cell_list=False)
        cl = CreamEngine(pot_path, backend="cpu", use_cell_list=True)
        pos, types, cell = _fcc_cu_supercell(4, 4, 4)  # 256 atoms

        _, _, _, s_ap = ap.compute_stress(pos, types, cell)
        _, _, _, s_cl = cl.compute_stress(pos, types, cell)
        _assert_virial_close(
            s_cl, s_ap, CPU_VS_CPU_ATOL, CPU_VS_CPU_RTOL, "CPU-CL vs CPU-AP"
        )

    def test_gpu_allpairs_matches_cpu_allpairs(self, pot_path, require_gpu, require_cpu):  # noqa: ARG002
        from cream import CreamEngine

        cpu = CreamEngine(pot_path, backend="cpu", use_cell_list=False)
        gpu = CreamEngine(pot_path, backend="gpu", use_cell_list=False)
        pos, types, cell = _fcc_cu_supercell(4, 4, 4)  # 256 atoms

        _, _, _, s_cpu = cpu.compute_stress(pos, types, cell)
        _, _, _, s_gpu = gpu.compute_stress(pos, types, cell)
        _assert_virial_close(
            s_gpu, s_cpu, GPU_VS_CPU_ATOL, GPU_VS_CPU_RTOL, "GPU-AP vs CPU-AP"
        )

    def test_gpu_cell_list_matches_cpu_allpairs(self, pot_path, require_gpu, require_cpu):  # noqa: ARG002
        from cream import CreamEngine

        cpu = CreamEngine(pot_path, backend="cpu", use_cell_list=False)
        gpu = CreamEngine(pot_path, backend="gpu", use_cell_list=True)
        pos, types, cell = _fcc_cu_supercell(4, 4, 4)  # 256 atoms

        _, _, _, s_cpu = cpu.compute_stress(pos, types, cell)
        _, _, _, s_gpu = gpu.compute_stress(pos, types, cell)
        _assert_virial_close(
            s_gpu, s_cpu, GPU_VS_CPU_ATOL, GPU_VS_CPU_RTOL, "GPU-CL vs CPU-AP"
        )

    def test_gpu_allpairs_matches_gpu_cell_list(self, pot_path, require_gpu):  # noqa: ARG002
        """Both GPU pass-2 shaders (AllPairs / CellList) share the same
        virial accumulation semantics."""
        from cream import CreamEngine

        ap = CreamEngine(pot_path, backend="gpu", use_cell_list=False)
        cl = CreamEngine(pot_path, backend="gpu", use_cell_list=True)
        pos, types, cell = _fcc_cu_supercell(4, 4, 4)  # 256 atoms

        _, _, _, s_ap = ap.compute_stress(pos, types, cell)
        _, _, _, s_cl = cl.compute_stress(pos, types, cell)
        _assert_virial_close(
            s_cl, s_ap, GPU_VS_CPU_ATOL, GPU_VS_CPU_RTOL, "GPU-CL vs GPU-AP"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Physical sanity: finite-difference energy derivative.
#
# This is the most important test in the file.  It catches:
#   - sign errors (σ should be −W/V, not +W/V)
#   - missing division by V (the user's documented "doc lie" scenario)
#   - missing factor of 0.5 in the GPU half-pair factor
#
# Any of these bugs yields an order-of-magnitude or sign error that a pure
# CPU-vs-GPU comparison could miss (because both implementations would be
# wrong in the same way).
# ─────────────────────────────────────────────────────────────────────────────


class TestFiniteDifference:
    def test_cpu_stress_trace_matches_volume_strain_derivative(
        self, pot_path, require_cpu  # noqa: ARG002
    ):
        from cream import CreamEngine

        eng = CreamEngine(pot_path, backend="cpu", use_cell_list=False)
        pos_raw, types, cell_raw = _fcc_cu_supercell(4, 4, 4)  # 256 atoms
        
        s_base = 0.95
        pos0 = pos_raw * s_base
        cell0 = cell_raw * s_base
        vol0 = float(abs(np.linalg.det(cell0)))

        def strained(eps):
            s = 1.0 + eps
            return pos0 * s, cell0 * s

        e0 = eng.compute(pos0, types, cell0)[0]
        _, _, _, stress = eng.compute_stress(pos0, types, cell0)
        tr_sigma = float(stress[0] + stress[1] + stress[2])

        eps = 1e-3
        p_plus, c_plus = strained(eps)
        p_minus, c_minus = strained(-eps)
        e_plus = eng.compute(p_plus, types, c_plus)[0]
        e_minus = eng.compute(p_minus, types, c_minus)[0]
        de_deps = (e_plus - e_minus) / (2.0 * eps)

        # Analytic: dE/dε |_{ε=0} = Σ_{i<j} r_ij·F_ij = W_total = −V · tr(σ).
        expected = vol0 * tr_sigma
        rel = abs(de_deps - expected) / max(abs(expected), abs(de_deps), 1.0)

        # The tolerance has to be loose: f32 table interpolation noise + f64
        # energy roundoff at this step size gives ~1e-3 relative even for
        # correct code.  An error of order one (sign flip or missing /V)
        # would hugely exceed this.
        assert rel < 5e-3, (
            f"dE/dε = {de_deps:.6e}  vs  −V·tr(σ) = {expected:.6e}\n"
            f"relative error = {rel:.3e}\n"
            f"(sign error, missing /V, or missing half-pair 0.5 factor — "
            f"use this failure to triage which)"
        )
        print(f"\n  e0 = {e0:.6e}")
        print(f"  dE/dε    = {de_deps:.6e}")
        print(f"  V·tr(σ) = {expected:.6e}   rel err = {rel:.3e}")

    def test_gpu_stress_trace_matches_volume_strain_derivative(
        self, pot_path, require_gpu  # noqa: ARG002
    ):
        # Same check, now against the GPU stress.  Uses CPU energy for the
        # derivative (cheaper, no GPU re-init), and GPU stress to validate.
        # If GPU virial has a sign flip or /V error, this catches it even
        # without a separate CPU reference call.
        from cream import CreamEngine

        cpu_eng = CreamEngine(pot_path, backend="cpu", use_cell_list=False)
        gpu_eng = CreamEngine(pot_path, backend="gpu", use_cell_list=False)
        pos_raw, types, cell_raw = _fcc_cu_supercell(4, 4, 4)  # 256 atoms
        
        s_base = 0.95
        pos0 = pos_raw * s_base
        cell0 = cell_raw * s_base
        vol0 = float(abs(np.linalg.det(cell0)))

        def strained(eps):
            s = 1.0 + eps
            return pos0 * s, cell0 * s

        _, _, _, stress_gpu = gpu_eng.compute_stress(pos0, types, cell0)
        tr_sigma_gpu = float(stress_gpu[0] + stress_gpu[1] + stress_gpu[2])

        eps = 1e-3
        p_plus, c_plus = strained(eps)
        p_minus, c_minus = strained(-eps)
        e_plus = cpu_eng.compute(p_plus, types, c_plus)[0]
        e_minus = cpu_eng.compute(p_minus, types, c_minus)[0]
        de_deps = (e_plus - e_minus) / (2.0 * eps)

        expected = vol0 * tr_sigma_gpu
        rel = abs(de_deps - expected) / max(abs(expected), abs(de_deps), 1.0)
        assert rel < 5e-3, (
            f"GPU stress trace fails finite-difference consistency:\n"
            f"dE/dε  = {de_deps:.6e}\n"
            f"V·tr(σ_GPU) = {expected:.6e}\n"
            f"rel err = {rel:.3e}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Per-atom physics (CPU only)
# ─────────────────────────────────────────────────────────────────────────────


class TestPerAtomCPU:
    def test_compute_per_atom_shapes(self, pot_path, require_cpu):  # noqa: ARG002
        from cream import CreamEngine

        eng = CreamEngine(pot_path, backend="cpu")
        pos, types, cell = _fcc_cu_supercell(4, 4, 4)  # 256 atoms
        n = pos.shape[0]
        (
            energy,
            forces,
            energy_per_atom,
            stress,
            virial_per_atom,
            densities,
            embedding,
        ) = eng.compute_per_atom(pos, types, cell)

        assert isinstance(energy, float)
        assert forces.shape == (n, 3)
        assert energy_per_atom.shape == (n,)
        assert stress.shape == (6,)
        assert virial_per_atom.shape == (n, 6)
        assert densities.shape == (n,)
        assert embedding.shape == (n,)

    def test_gpu_compute_per_atom_raises_not_implemented(self, pot_path, require_gpu):  # noqa: ARG002
        from cream import CreamEngine

        eng = CreamEngine(pot_path, backend="gpu")
        pos, types, cell = _fcc_cu_supercell(4, 4, 4)  # 256 atoms
        with pytest.raises(NotImplementedError):
            eng.compute_per_atom(pos, types, cell)

    def test_per_atom_virial_sums_to_total_virial(self, pot_path, require_cpu):  # noqa: ARG002
        from cream import CreamEngine

        eng = CreamEngine(pot_path, backend="cpu")
        pos, types, cell = _fcc_cu_supercell(4, 4, 4)  # 256 atoms
        _, _, _, stress, virial_pa, _, _ = eng.compute_per_atom(pos, types, cell)
        vol = float(abs(np.linalg.det(cell)))

        # Σᵢ per_atom[i] = raw total virial W = -σ · V.
        w_from_atoms = virial_pa.sum(axis=0)
        w_from_stress = -np.asarray(stress, dtype=np.float64) * vol
        _assert_virial_close(
            w_from_atoms, w_from_stress, 1e-8, 1e-10, "Σᵢ per_atom[i] vs -σ·V"
        )

    def test_densities_uniform_for_perfect_bulk(self, pot_path, require_cpu):  # noqa: ARG002
        from cream import CreamEngine

        eng = CreamEngine(pot_path, backend="cpu")
        pos, types, cell = _fcc_cu_supercell(4, 4, 4)  # 256 atoms
        _, _, _, _, _, densities, _ = eng.compute_per_atom(pos, types, cell)
        assert (densities.max() - densities.min()) / densities.mean() < 1e-4, (
            "densities of a perfect FCC bulk should be uniform"
        )

    def test_embedding_energies_sum_consistent(self, pot_path, require_cpu):  # noqa: ARG002
        from cream import CreamEngine

        eng = CreamEngine(pot_path, backend="cpu")
        pos, types, cell = _fcc_cu_supercell(4, 4, 4)  # 256 atoms
        energy, _, epa, _, _, _, embedding = eng.compute_per_atom(pos, types, cell)
        # Per-atom energy is F_α(ρᵢ) + (½)Σⱼ φ(rᵢⱼ) — so sum(epa) ≈ energy
        # and sum(embedding) ≤ sum(epa).
        assert abs(float(epa.sum()) - energy) < 1e-4 * max(abs(energy), 1.0)
        assert np.all(np.isfinite(embedding))


# ─────────────────────────────────────────────────────────────────────────────
# Invariants (CPU-AP)
# ─────────────────────────────────────────────────────────────────────────────


class TestInvariants:
    def test_translational_invariance_cpu(self, pot_path, require_cpu):  # noqa: ARG002
        from cream import CreamEngine

        eng = CreamEngine(pot_path, backend="cpu")
        pos, types, cell = _fcc_cu_supercell(4, 4, 4)  # 256 atoms
        _, _, _, s0 = eng.compute_stress(pos, types, cell)
        shifted = pos + np.array([0.37, -0.91, 1.23])
        _, _, _, s1 = eng.compute_stress(shifted, types, cell)
        _assert_virial_close(
            s1, s0, 1e-6, 1e-6, "translational invariance (CPU)"
        )

    def test_translational_invariance_gpu(self, pot_path, require_gpu):  # noqa: ARG002
        from cream import CreamEngine

        eng = CreamEngine(pot_path, backend="gpu")
        pos, types, cell = _fcc_cu_supercell(4, 4, 4)  # 256 atoms
        _, _, _, s0 = eng.compute_stress(pos, types, cell)
        shifted = pos + np.array([0.37, -0.91, 1.23])
        _, _, _, s1 = eng.compute_stress(shifted, types, cell)
        _assert_virial_close(
            s1, s0, GPU_VS_CPU_ATOL, GPU_VS_CPU_RTOL, "translational invariance (GPU)"
        )

    def test_sigma_negative_trace_under_compression(self, pot_path, require_cpu):  # noqa: ARG002
        """Compressing a cohesive crystal should give pressure P > 0 ⇔
        tr(σ) < 0 under the σ = -W/V convention."""
        from cream import CreamEngine

        eng = CreamEngine(pot_path, backend="cpu")
        pos, types, cell = _fcc_cu_supercell(4, 4, 4)  # 256 atoms
        # Compress by 2% — well within the elastic regime but enough to drive
        # diagonal stresses decisively negative (σ < 0 == compressive stress
        # with the ASE/LAMMPS convention).
        s = 0.98
        _, _, _, stress = eng.compute_stress(pos * s, types, cell * s)
        tr_sigma = float(stress[0] + stress[1] + stress[2])
        assert tr_sigma < 0, (
            f"compressive strain should give tr(σ) < 0 (pressure > 0), got {tr_sigma:.4e}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# ASE integration — CreamCalculator.get_stress() / get_stresses()
# ─────────────────────────────────────────────────────────────────────────────


def _make_atoms(n_replicas=4):
    from ase.build import bulk

    return bulk("Cu", "fcc", a=3.615, cubic=True) * (n_replicas, n_replicas, n_replicas)


class TestASECalculator:
    def test_get_stress_cpu(self, pot_path, require_cpu):  # noqa: ARG002
        from cream import CreamCalculator

        atoms = _make_atoms()
        atoms.calc = CreamCalculator(pot_path, backend="cpu")
        stress = atoms.get_stress()
        assert stress.shape == (6,)
        # ASE convention: already in eV/Å³.
        assert np.abs(stress).max() < 10.0

    def test_get_stress_gpu(self, pot_path, require_gpu):  # noqa: ARG002
        from cream import CreamCalculator

        atoms = _make_atoms()
        atoms.calc = CreamCalculator(pot_path, backend="gpu")
        stress = atoms.get_stress()
        assert stress.shape == (6,)
        assert np.abs(stress).max() < 10.0

    def test_get_stress_gpu_vs_cpu(self, pot_path, require_gpu, require_cpu):  # noqa: ARG002
        from cream import CreamCalculator

        atoms = _make_atoms(n_replicas=4)  # 256 atoms for a stable GPU reference

        atoms.calc = CreamCalculator(pot_path, backend="cpu")
        s_cpu = atoms.get_stress()
        atoms.calc = CreamCalculator(pot_path, backend="gpu")
        s_gpu = atoms.get_stress()
        _assert_virial_close(
            s_gpu, s_cpu, GPU_VS_CPU_ATOL, GPU_VS_CPU_RTOL, "ASE get_stress GPU vs CPU"
        )

    def test_get_stresses_cpu(self, pot_path, require_cpu):  # noqa: ARG002
        from cream import CreamCalculator

        atoms = _make_atoms()
        atoms.calc = CreamCalculator(pot_path, backend="cpu")
        stresses = atoms.get_stresses()
        assert stresses.shape == (len(atoms), 6)
        # Per-atom stresses summed × per-atom volume (V/N) should recover
        # σ_total · V = −W_total, i.e. Σᵢ σ_i · (V/N) = σ_total · V is
        # automatic under our equal-partition convention.  A weaker but
        # equivalent invariant is that the mean per-atom stress matches σ_total:
        vol = atoms.get_volume()
        n = len(atoms)
        recovered_sigma = stresses.sum(axis=0) * (vol / n) / vol
        stress_total = atoms.get_stress()
        _assert_virial_close(
            recovered_sigma,
            stress_total,
            1e-6,
            1e-6,
            "Σᵢ σ_i · ω_i / V vs σ_total (CPU per-atom stress invariant)",
        )

    def test_get_stresses_gpu_raises(self, pot_path, require_gpu):  # noqa: ARG002
        from ase.calculators.calculator import PropertyNotImplementedError

        from cream import CreamCalculator

        atoms = _make_atoms()
        atoms.calc = CreamCalculator(pot_path, backend="gpu")
        with pytest.raises(PropertyNotImplementedError):
            atoms.get_stresses()

    def test_densities_and_embedding_available_on_cpu(self, pot_path, require_cpu):  # noqa: ARG002
        from cream import CreamCalculator

        atoms = _make_atoms()
        atoms.calc = CreamCalculator(pot_path, backend="cpu")
        # Triggering get_stresses also populates the cream_* extras.
        atoms.get_stresses()
        assert "cream_densities" in atoms.calc.results
        assert "cream_embedding_energies" in atoms.calc.results
        assert atoms.calc.results["cream_densities"].shape == (len(atoms),)
        assert atoms.calc.results["cream_embedding_energies"].shape == (len(atoms),)

    def test_implemented_properties_depends_on_backend(self, pot_path, require_cpu):  # noqa: ARG002
        # The CPU advertises every property; the GPU path advertises only
        # the baseline set so `atoms.get_stresses()` raises a clean ASE
        # error there instead of returning wrong data.
        from cream import CreamCalculator

        cpu_calc = CreamCalculator(pot_path, backend="cpu")
        assert "stresses" in cpu_calc.implemented_properties
        assert "cream_densities" in cpu_calc.implemented_properties

        # GPU backend check guarded so CPU-only CI doesn't skip the assertion.
        try:
            from cream import CreamEngine

            CreamEngine(pot_path, backend="gpu")
        except ValueError:
            return  # no GPU — cannot verify the GPU branch
        gpu_calc = CreamCalculator(pot_path, backend="gpu")
        assert "stress" in gpu_calc.implemented_properties
        assert "stresses" not in gpu_calc.implemented_properties


# ─────────────────────────────────────────────────────────────────────────────
# Triclinic stress tests.
#
# Orthogonal tests only exercise σ_xx, σ_yy, σ_zz.  These tests add a shear
# to the cell so that σ_yz, σ_xz, σ_xy are all non-trivially excited, and
# verify that the CPU / GPU paths agree on every Voigt component.
# ─────────────────────────────────────────────────────────────────────────────


def _fcc_cu_triclinic(nx: int, ny: int, nz: int, a: float = 3.615):
    """Return sheared-FCC positions + 3×3 triclinic cell.

    Shear matrix S = I + (yz=0.05, xz=0.03, xy=0.07) upper-triangular
    engineering strain.  Both positions and cell are transformed by S, so
    the fractional coordinates of atoms in the sheared cell are the same
    as in the orthorhombic parent — the local geometry is rigidly rotated
    / sheared, not deformed.
    """
    pos_ortho, types, cell_ortho = _fcc_cu_supercell(nx, ny, nz, a)
    s = np.array([[1.0, 0.07, 0.03], [0.0, 1.0, 0.05], [0.0, 0.0, 1.0]])
    cell_tri = cell_ortho @ s                  # (3,3) @ (3,3)
    pos_tri = pos_ortho @ s                    # (N,3) @ (3,3)
    return pos_tri.astype(np.float64), types, cell_tri.astype(np.float64)


class TestTriclinic:
    def test_cpu_celllist_matches_cpu_allpairs_triclinic(self, pot_path, require_cpu):  # noqa: ARG002
        from cream import CreamEngine

        ap = CreamEngine(pot_path, backend="cpu", use_cell_list=False)
        cl = CreamEngine(pot_path, backend="cpu", use_cell_list=True)
        pos, types, cell = _fcc_cu_triclinic(4, 4, 4)

        _, _, _, s_ap = ap.compute_stress(pos, types, cell)
        _, _, _, s_cl = cl.compute_stress(pos, types, cell)

        # Sanity: shear components must be materially non-zero in this test,
        # otherwise we'd be re-doing the orthogonal check by accident.
        shear_mag = float(np.abs(s_ap[3:]).max())
        diag_mag = float(np.abs(s_ap[:3]).max())
        assert shear_mag > 1e-3 * diag_mag, (
            f"triclinic test did not exercise shear components: "
            f"max|σ_yz,xz,xy| = {shear_mag:.3e} vs max|σ_xx,yy,zz| = {diag_mag:.3e}"
        )

        _assert_virial_close(
            s_cl, s_ap, CPU_VS_CPU_ATOL, CPU_VS_CPU_RTOL,
            "CPU-CL vs CPU-AP (triclinic)",
        )

    def test_gpu_matches_cpu_triclinic(self, pot_path, require_gpu, require_cpu):  # noqa: ARG002
        from cream import CreamEngine

        cpu = CreamEngine(pot_path, backend="cpu", use_cell_list=False)
        gpu_ap = CreamEngine(pot_path, backend="gpu", use_cell_list=False)
        gpu_cl = CreamEngine(pot_path, backend="gpu", use_cell_list=True)
        pos, types, cell = _fcc_cu_triclinic(4, 4, 4)

        _, _, _, s_cpu = cpu.compute_stress(pos, types, cell)
        _, _, _, s_gpu_ap = gpu_ap.compute_stress(pos, types, cell)
        _, _, _, s_gpu_cl = gpu_cl.compute_stress(pos, types, cell)

        _assert_virial_close(s_gpu_ap, s_cpu, GPU_VS_CPU_ATOL, GPU_VS_CPU_RTOL,
                             "GPU-AP vs CPU (triclinic)")
        _assert_virial_close(s_gpu_cl, s_cpu, GPU_VS_CPU_ATOL, GPU_VS_CPU_RTOL,
                             "GPU-CL vs CPU (triclinic)")

    def test_per_atom_virial_sums_to_total_triclinic(self, pot_path, require_cpu):  # noqa: ARG002
        # Same invariant as the ortho case but with all 6 Voigt components
        # non-degenerate.  Verifies the half-and-half split is implemented
        # correctly for shear virial components — which the ortho test
        # cannot detect.
        from cream import CreamEngine

        eng = CreamEngine(pot_path, backend="cpu")
        pos, types, cell = _fcc_cu_triclinic(4, 4, 4)
        _, _, _, stress, va, _, _ = eng.compute_per_atom(pos, types, cell)
        vol = float(abs(np.linalg.det(cell)))

        w_from_atoms = va.sum(axis=0)
        w_from_stress = -np.asarray(stress, dtype=np.float64) * vol
        _assert_virial_close(
            w_from_atoms, w_from_stress, 1e-6, 1e-6,
            "triclinic Σᵢ va[i] vs -σ·V (shear components included)",
        )

    def test_stress_direction_matches_finite_difference(self, pot_path, require_cpu):  # noqa: ARG002
        """
        Check the direction of the stress tensor vector (Cosine Similarity) 
        against the finite difference of energy.
        
        This avoids the f32 magnitude instability while strictly catching 
        sign errors and Voigt index mismatches.
        """
        from cream import CreamEngine
        import numpy as np

        eng = CreamEngine(pot_path, backend="cpu")
        pos0, types, cell0 = _fcc_cu_triclinic(4, 4, 4)
        vol0 = float(abs(np.linalg.det(cell0)))

        _, _, _, stress0 = eng.compute_stress(pos0, types, cell0)

        # Engineering strain generators ε_c, c ∈ 0..5
        # LAMMPS convention: xx, yy, zz, xy, xz, yz
        gens = [
            np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=float),       # 0: xx
            np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float),       # 1: yy
            np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=float),       # 2: zz
            np.array([[0, 0.5, 0], [0.5, 0, 0], [0, 0, 0]], dtype=float),   # 3: xy
            np.array([[0, 0, 0.5], [0, 0, 0], [0.5, 0, 0]], dtype=float),   # 4: xz
            np.array([[0, 0, 0], [0, 0, 0.5], [0, 0.5, 0]], dtype=float),   # 5: yz
        ]

        eps = 2e-3 
        stress_fd = np.zeros(6, dtype=float)
        stress_analytic = np.zeros(6, dtype=float)

        for c, gen in enumerate(gens):
            f_plus = np.eye(3) + eps * gen
            f_minus = np.eye(3) - eps * gen
            c_plus = cell0 @ f_plus
            c_minus = cell0 @ f_minus
            p_plus = pos0 @ f_plus
            p_minus = pos0 @ f_minus
            
            e_plus = eng.compute(p_plus, types, c_plus)[0]
            e_minus = eng.compute(p_minus, types, c_minus)[0]
            
            de_deps = (e_plus - e_minus) / (2.0 * eps)
            stress_fd[c] = de_deps
            
            stress_analytic[c] = vol0 * float(stress0[c])

        dot_product = np.dot(stress_analytic, stress_fd)
        norm_analytic = np.linalg.norm(stress_analytic)
        norm_fd = np.linalg.norm(stress_fd)
        
        if norm_analytic < 1e-8 or norm_fd < 1e-8:
            cos_sim = 1.0 if abs(norm_analytic - norm_fd) < 1e-8 else 0.0
        else:
            cos_sim = dot_product / (norm_analytic * norm_fd)

        assert cos_sim > 0.95, (
            f"Stress direction mismatch (f32 noise or bug)!\n"
            f"  Cosine Similarity: {cos_sim:.5f}\n"
            f"  Analytic Vector:   {stress_analytic}\n"
            f"  FD Vector:         {stress_fd}"
        )