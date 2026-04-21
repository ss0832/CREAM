"""PyO3 interface tests for CREAM.

These tests verify the Python bindings work correctly, including
type conversion (f64→f32→f64), error handling, physical sanity, and
backend selection (``"gpu"`` / ``"cpu"``).

Requirements:
    pip install maturin pytest numpy
    maturin develop --features python

Run:
    pytest tests_python/test_pyo3.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

# ── Fixture: locate the Cu01_eam.alloy file ──────────────────────────────────


def _find_potential() -> str:
    """Find Cu01_eam.alloy relative to the project root."""
    import pathlib

    candidates = [
        pathlib.Path("Cu01_eam.alloy"),
        pathlib.Path(__file__).parent.parent / "Cu01_eam.alloy",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    pytest.skip("Cu01_eam.alloy not found — skipping real-potential tests")
    return ""  # unreachable


@pytest.fixture
def pot_path() -> str:
    return _find_potential()


# ── GPU engine fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def engine(pot_path: str, require_gpu):  # noqa: ARG001
    from cream import CreamEngine

    return CreamEngine(pot_path, use_cell_list=False, backend="gpu")


@pytest.fixture
def engine_cl(pot_path: str, require_gpu):  # noqa: ARG001
    from cream import CreamEngine

    return CreamEngine(pot_path, use_cell_list=True, backend="gpu")


# ── CPU engine fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def cpu_engine(pot_path: str, require_cpu):  # noqa: ARG001
    from cream import CreamEngine

    return CreamEngine(pot_path, use_cell_list=False, backend="cpu")


@pytest.fixture
def cpu_engine_cl(pot_path: str, require_cpu):  # noqa: ARG001
    from cream import CreamEngine

    return CreamEngine(pot_path, use_cell_list=True, backend="cpu")


# ── Shared geometry helpers ───────────────────────────────────────────────────


def _cu_fcc_4atom(cell_size: float = 10.0):
    """Cu FCC unit cell: 4 atoms at a=3.615 Å.

    Parameters
    ----------
    cell_size:
        Edge length of the (cubic) simulation box in Å.
        Must be > 2 × cutoff (~11.014 Å) for PBC with the CPU engine.
        Default 10.0 Å is fine for no-PBC calculations.
    """
    a = 3.615
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [a / 2, a / 2, 0.0],
            [a / 2, 0.0, a / 2],
            [0.0, a / 2, a / 2],
        ],
        dtype=np.float64,
    )
    atom_types = np.array([0, 0, 0, 0], dtype=np.int32)
    cell = np.diag([cell_size, cell_size, cell_size])
    return positions, atom_types, cell


def _cu_fcc_4x4x4():
    """4×4×4 orthorhombic supercell of Cu (256 atoms, a=3.615 Å).

    Cell dimensions 4*a ≈ 14.46 Å > 2×cutoff, so valid for CPU PBC.
    """
    a = 3.615
    base, _, _ = _cu_fcc_4atom()
    positions = []
    for ix in range(4):
        for iy in range(4):
            for iz in range(4):
                for b in base:
                    positions.append(b + [ix * a, iy * a, iz * a])
    positions = np.array(positions, dtype=np.float64)
    all_types = np.zeros(len(positions), dtype=np.int32)
    cell = np.diag([4 * a, 4 * a, 4 * a])
    return positions, all_types, cell


# ── Construction tests ────────────────────────────────────────────────────────


class TestConstruction:
    def test_create_engine(self, engine):
        assert engine is not None

    def test_create_engine_cell_list(self, engine_cl):
        assert engine_cl is not None

    def test_create_engine_custom_cell_size(self, pot_path: str, require_gpu):  # noqa: ARG002
        from cream import CreamEngine

        eng = CreamEngine(pot_path, use_cell_list=True, cell_size=6.0, backend="gpu")
        assert eng is not None

    def test_invalid_path_raises(self):
        from cream import CreamEngine

        with pytest.raises(ValueError):
            CreamEngine("nonexistent_file.eam.alloy")

    def test_gpu_backend_property(self, engine):
        assert engine.backend == "gpu"


# ── CPU Backend construction tests ────────────────────────────────────────────


class TestCpuBackendConstruction:
    """CPU backend should construct without a GPU adapter."""

    def test_create_cpu_engine(self, cpu_engine):
        assert cpu_engine is not None

    def test_cpu_backend_property(self, cpu_engine):
        assert cpu_engine.backend == "cpu"

    def test_create_cpu_engine_cell_list(self, cpu_engine_cl):
        assert cpu_engine_cl is not None

    def test_create_cpu_engine_cell_list_backend(self, cpu_engine_cl):
        assert cpu_engine_cl.backend == "cpu"

    def test_create_cpu_engine_custom_cell_size(self, pot_path: str, require_cpu):  # noqa: ARG002
        from cream import CreamEngine

        eng = CreamEngine(pot_path, use_cell_list=True, cell_size=6.0, backend="cpu")
        assert eng is not None
        assert eng.backend == "cpu"

    def test_invalid_backend_raises(self, pot_path: str):
        from cream import CreamEngine

        with pytest.raises(ValueError, match="backend"):
            CreamEngine(pot_path, backend="tpu")

    def test_invalid_path_cpu_raises(self):
        from cream import CreamEngine

        with pytest.raises(ValueError):
            CreamEngine("nonexistent_file.eam.alloy", backend="cpu")


# ── Getter tests ──────────────────────────────────────────────────────────────


class TestGetters:
    def test_elements(self, engine):
        elems = engine.elements
        assert isinstance(elems, list)
        assert len(elems) >= 1
        assert all(isinstance(e, str) for e in elems)

    def test_cutoff(self, engine):
        rc = engine.cutoff
        assert isinstance(rc, float)
        assert rc > 0.0

    def test_n_elements(self, engine):
        n = engine.n_elements
        assert isinstance(n, int)
        assert n == len(engine.elements)


class TestCpuGetters:
    def test_elements(self, cpu_engine):
        elems = cpu_engine.elements
        assert isinstance(elems, list)
        assert len(elems) >= 1
        assert all(isinstance(e, str) for e in elems)

    def test_cutoff(self, cpu_engine):
        rc = cpu_engine.cutoff
        assert isinstance(rc, float)
        assert rc > 0.0

    def test_n_elements(self, cpu_engine):
        n = cpu_engine.n_elements
        assert isinstance(n, int)
        assert n == len(cpu_engine.elements)


# ── Compute basic tests (GPU) ─────────────────────────────────────────────────


class TestCompute:
    def test_basic_compute(self, engine):
        pos, types, cell = _cu_fcc_4atom()
        energy, forces, epa = engine.compute(pos, types, cell)

        assert isinstance(energy, float)
        assert np.isfinite(energy)
        assert isinstance(forces, np.ndarray)
        assert forces.shape == (4, 3)
        assert forces.dtype == np.float64
        assert np.all(np.isfinite(forces))
        assert isinstance(epa, np.ndarray)
        assert epa.shape == (0,)

    def test_compute_no_pbc(self, engine):
        pos, types, _ = _cu_fcc_4atom()
        energy, forces, epa = engine.compute(pos, types, None)
        assert np.isfinite(energy)
        assert np.all(np.isfinite(forces))

    def test_compute_cell_list(self, engine_cl):
        pos, types, cell = _cu_fcc_4atom()
        energy, forces, epa = engine_cl.compute(pos, types, cell)
        assert np.isfinite(energy)
        assert np.all(np.isfinite(forces))

    def test_newton_third_law(self, engine):
        """Sum of all forces should be near zero."""
        pos, types, cell = _cu_fcc_4atom()
        _, forces, _ = engine.compute(pos, types, cell)
        force_sum = forces.sum(axis=0)
        np.testing.assert_allclose(force_sum, 0.0, atol=1e-2)

    def test_pbc_vs_no_pbc_large_cell(self, engine):
        """With a huge cell, PBC and no-PBC should agree."""
        pos, types, _ = _cu_fcc_4atom()
        cell_large = np.diag([100.0, 100.0, 100.0])
        e_pbc, f_pbc, _ = engine.compute(pos, types, cell_large)
        e_none, f_none, _ = engine.compute(pos, types, None)
        np.testing.assert_allclose(e_pbc, e_none, atol=1e-3)
        np.testing.assert_allclose(f_pbc, f_none, atol=1e-3)

    def test_deterministic(self, engine):
        """Two identical calls must give identical results."""
        pos, types, cell = _cu_fcc_4atom()
        e1, f1, _ = engine.compute(pos, types, cell)
        e2, f2, _ = engine.compute(pos, types, cell)
        assert e1 == e2
        np.testing.assert_array_equal(f1, f2)


# ── Compute tests (CPU backend) ───────────────────────────────────────────────
#
# NOTE on cell sizes: the CPU engine enforces the minimum image convention,
# requiring perpendicular cell height > 2 × cutoff ≈ 11.014 Å.
# PBC tests therefore use 15 Å cubic cells; no-PBC tests use None.


class TestCpuCompute:
    """CPU backend compute tests — run without GPU."""

    def test_basic_compute(self, cpu_engine):
        # Use a cell large enough for CPU PBC (> 2*cutoff ≈ 11.014 Å)
        pos, types, cell = _cu_fcc_4atom(cell_size=15.0)
        energy, forces, epa = cpu_engine.compute(pos, types, cell)

        assert isinstance(energy, float)
        assert np.isfinite(energy)
        assert forces.shape == (4, 3)
        assert forces.dtype == np.float64
        assert np.all(np.isfinite(forces))
        assert epa.shape == (4,)
        assert np.all(np.isfinite(epa))

    def test_compute_no_pbc(self, cpu_engine):
        pos, types, _ = _cu_fcc_4atom()
        energy, forces, epa = cpu_engine.compute(pos, types, None)
        assert np.isfinite(energy)
        assert np.all(np.isfinite(forces))

    def test_compute_cell_list(self, cpu_engine_cl):
        pos, types, cell = _cu_fcc_4atom(cell_size=15.0)
        energy, forces, epa = cpu_engine_cl.compute(pos, types, cell)
        assert np.isfinite(energy)
        assert np.all(np.isfinite(forces))

    def test_energy_per_atom_sum(self, cpu_engine):
        pos, types, cell = _cu_fcc_4atom(cell_size=15.0)
        energy, _forces, epa = cpu_engine.compute(pos, types, cell)
        np.testing.assert_allclose(epa.sum(), energy, atol=1e-3)

    def test_newton_third_law(self, cpu_engine):
        pos, types, cell = _cu_fcc_4atom(cell_size=15.0)
        _, forces, _ = cpu_engine.compute(pos, types, cell)
        np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=1e-2)

    def test_pbc_vs_no_pbc_large_cell(self, cpu_engine):
        pos, types, _ = _cu_fcc_4atom()
        cell_large = np.diag([100.0, 100.0, 100.0])
        e_pbc, f_pbc, _ = cpu_engine.compute(pos, types, cell_large)
        e_none, f_none, _ = cpu_engine.compute(pos, types, None)
        np.testing.assert_allclose(e_pbc, e_none, atol=1e-3)
        np.testing.assert_allclose(f_pbc, f_none, atol=1e-3)

    def test_deterministic(self, cpu_engine):
        pos, types, cell = _cu_fcc_4atom(cell_size=15.0)
        e1, f1, _ = cpu_engine.compute(pos, types, cell)
        e2, f2, _ = cpu_engine.compute(pos, types, cell)
        assert e1 == e2
        np.testing.assert_array_equal(f1, f2)

    def test_equilibrium_energy_per_atom(self, cpu_engine):
        """FCC Cu equilibrium energy should be ~-3.54 eV/atom for Mishin Cu01."""
        positions, all_types, cell = _cu_fcc_4x4x4()
        energy, _, _ = cpu_engine.compute(positions, all_types, cell)
        epa = energy / len(positions)
        assert -4.0 < epa < -3.0, f"energy/atom = {epa:.4f}, expected ~-3.54"

    def test_restoring_force(self, cpu_engine):
        """Displacing an atom in a PBC supercell should produce a restoring force."""
        # Use the full 4×4×4 supercell so atoms are at true equilibrium
        positions, all_types, cell = _cu_fcc_4x4x4()
        pos_disp = positions.copy()
        pos_disp[0, 0] += 0.1
        _, forces, _ = cpu_engine.compute(pos_disp, all_types, cell)
        assert forces[0, 0] < 0, f"Expected restoring force, got {forces[0, 0]:.4f}"

    def test_triclinic_cell(self, cpu_engine):
        """Triclinic cell (> 2×cutoff) should not crash and produce finite results."""
        pos, types, _ = _cu_fcc_4atom()
        # Use a large monoclinic cell to satisfy minimum image convention
        cell = np.array(
            [[15.0, 0.0, 0.0], [1.0, 15.0, 0.0], [0.0, 0.0, 15.0]],
            dtype=np.float64,
        )
        energy, forces, epa = cpu_engine.compute(pos, types, cell)
        assert np.isfinite(energy)
        assert np.all(np.isfinite(forces))


# ── Cross-backend consistency ─────────────────────────────────────────────────
#
# Use the 4×4×4 supercell (cell ≈ 14.46 Å > 11.014 Å) so both GPU and CPU
# accept the periodic boundary conditions.


class TestCrossBackendConsistency:
    """GPU and CPU backends should agree to within f32 rounding (~1e-3)."""

    def test_energy_agrees(self, engine, cpu_engine):
        positions, all_types, cell = _cu_fcc_4x4x4()
        e_gpu, _, _ = engine.compute(positions, all_types, cell)
        e_cpu, _, _ = cpu_engine.compute(positions, all_types, cell)
        np.testing.assert_allclose(e_gpu, e_cpu, rtol=1e-3, atol=1e-3)

    def test_forces_agree(self, engine, cpu_engine):
        positions, all_types, cell = _cu_fcc_4x4x4()
        _, f_gpu, _ = engine.compute(positions, all_types, cell)
        _, f_cpu, _ = cpu_engine.compute(positions, all_types, cell)
        np.testing.assert_allclose(f_gpu, f_cpu, rtol=1e-2, atol=1e-2)

    def test_cell_list_agrees_with_all_pairs_cpu(self, cpu_engine, cpu_engine_cl):
        """CPU cell-list and all-pairs should agree closely."""
        positions, all_types, cell = _cu_fcc_4x4x4()
        e_ap, f_ap, _ = cpu_engine.compute(positions, all_types, cell)
        e_cl, f_cl, _ = cpu_engine_cl.compute(positions, all_types, cell)
        np.testing.assert_allclose(e_ap, e_cl, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(f_ap, f_cl, rtol=1e-2, atol=1e-2)

    def test_gpu_cell_list_triclinic(self, engine_cl, cpu_engine_cl, pot_path: str):
        """GPU cell-list with triclinic cell must agree with CPU cell-list.

        Regression test for the BB optimisation bug that caused ~190 eV
        energy errors and ~37 eV/Å force errors for triclinic cells.
        Uses (6,6,6) primitive FCC supercell so d-spacing > 2×cutoff.
        """
        from ase.build import bulk

        # Primitive FCC: triclinic cell vectors, 125 atoms
        atoms = bulk("Cu", "fcc", a=3.615) * (6, 6, 6)
        pos = atoms.get_positions().astype(np.float64)
        cell = np.array(atoms.get_cell(), dtype=np.float64)
        types = np.zeros(len(atoms), dtype=np.int32)

        e_gpu, f_gpu, _ = engine_cl.compute(pos, types, cell)
        e_cpu, f_cpu, _ = cpu_engine_cl.compute(pos, types, cell)
        np.testing.assert_allclose(e_gpu, e_cpu, rtol=1e-3, atol=0.1)
        np.testing.assert_allclose(f_gpu, f_cpu, rtol=1e-2, atol=1e-2)


# ── Input validation tests ───────────────────────────────────────────────────


class TestInputValidation:
    def test_wrong_position_shape(self, engine):
        """positions must be (N, 3)."""
        pos = np.zeros((4, 2), dtype=np.float64)
        types = np.array([0, 0, 0, 0], dtype=np.int32)
        with pytest.raises(ValueError):
            engine.compute(pos, types, None)

    def test_mismatched_lengths(self, engine):
        """atom_types length must match positions rows."""
        pos = np.zeros((4, 3), dtype=np.float64)
        types = np.array([0, 0, 0], dtype=np.int32)  # length 3 != 4
        with pytest.raises(ValueError):
            engine.compute(pos, types, None)

    def test_negative_atom_type(self, engine):
        """Negative atom type indices should raise."""
        pos = np.zeros((2, 3), dtype=np.float64)
        pos[1, 0] = 2.5
        types = np.array([0, -1], dtype=np.int32)
        with pytest.raises(ValueError):
            engine.compute(pos, types, None)

    def test_out_of_range_atom_type(self, engine):
        """atom_type >= n_elements should raise."""
        pos = np.zeros((2, 3), dtype=np.float64)
        pos[1, 0] = 2.5
        types = np.array([0, 99], dtype=np.int32)
        with pytest.raises(ValueError):
            engine.compute(pos, types, None)

    def test_wrong_cell_shape(self, engine):
        """cell must be (3, 3)."""
        pos, types, _ = _cu_fcc_4atom()
        cell_bad = np.zeros((2, 3), dtype=np.float64)
        with pytest.raises(ValueError):
            engine.compute(pos, types, cell_bad)

    def test_empty_positions(self, engine):
        """Empty positions should raise."""
        pos = np.zeros((0, 3), dtype=np.float64)
        types = np.array([], dtype=np.int32)
        with pytest.raises(ValueError):
            engine.compute(pos, types, None)


class TestCpuInputValidation:
    """Same validation checks for the CPU backend."""

    def test_wrong_position_shape(self, cpu_engine):
        pos = np.zeros((4, 2), dtype=np.float64)
        types = np.array([0, 0, 0, 0], dtype=np.int32)
        with pytest.raises(ValueError):
            cpu_engine.compute(pos, types, None)

    def test_mismatched_lengths(self, cpu_engine):
        pos = np.zeros((4, 3), dtype=np.float64)
        types = np.array([0, 0, 0], dtype=np.int32)
        with pytest.raises(ValueError):
            cpu_engine.compute(pos, types, None)

    def test_negative_atom_type(self, cpu_engine):
        pos = np.zeros((2, 3), dtype=np.float64)
        pos[1, 0] = 2.5
        types = np.array([0, -1], dtype=np.int32)
        with pytest.raises(ValueError):
            cpu_engine.compute(pos, types, None)

    def test_out_of_range_atom_type(self, cpu_engine):
        pos = np.zeros((2, 3), dtype=np.float64)
        pos[1, 0] = 2.5
        types = np.array([0, 99], dtype=np.int32)
        with pytest.raises(ValueError):
            cpu_engine.compute(pos, types, None)

    def test_wrong_cell_shape(self, cpu_engine):
        pos, types, _ = _cu_fcc_4atom()
        cell_bad = np.zeros((2, 3), dtype=np.float64)
        with pytest.raises(ValueError):
            cpu_engine.compute(pos, types, cell_bad)

    def test_empty_positions(self, cpu_engine):
        pos = np.zeros((0, 3), dtype=np.float64)
        types = np.array([], dtype=np.int32)
        with pytest.raises(ValueError):
            cpu_engine.compute(pos, types, None)


# ── Physical sanity tests (GPU) ──────────────────────────────────────────────


class TestPhysics:
    def test_equilibrium_energy_per_atom(self, engine):
        """FCC Cu equilibrium energy should be ~-3.54 eV/atom for Mishin Cu01."""
        positions, all_types, cell = _cu_fcc_4x4x4()
        energy, _, _ = engine.compute(positions, all_types, cell)
        epa = energy / len(positions)
        assert -4.0 < epa < -3.0, f"energy/atom = {epa:.4f}, expected ~-3.54"

    def test_restoring_force(self, engine):
        """Displacing an atom in a PBC supercell should produce a restoring force."""
        positions, all_types, cell = _cu_fcc_4x4x4()
        pos_disp = positions.copy()
        pos_disp[0, 0] += 0.1  # displace atom 0 along x
        _, forces, _ = engine.compute(pos_disp, all_types, cell)
        assert forces[0, 0] < 0, f"Expected restoring force, got {forces[0, 0]:.4f}"

    def test_triclinic_cell(self, engine):
        """Triclinic cell should not crash and produce finite results."""
        pos, types, _ = _cu_fcc_4atom()
        # Monoclinic tilt
        cell = np.array(
            [[10.0, 0.0, 0.0], [1.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
            dtype=np.float64,
        )
        energy, forces, epa = engine.compute(pos, types, cell)
        assert np.isfinite(energy)
        assert np.all(np.isfinite(forces))


# ── ASE Calculator tests (optional, skipped if ASE not installed) ────────────


class TestASECalculator:
    @pytest.fixture(autouse=True)
    def _skip_if_no_ase(self):
        pytest.importorskip("ase")

    @pytest.fixture(autouse=True)
    def _skip_if_no_gpu(self, require_gpu):  # noqa: ARG002
        pass

    def test_cream_calculator_import(self):
        from cream import CreamCalculator

        assert CreamCalculator is not None

    def test_calculator_energy(self, pot_path: str):
        from ase.build import bulk
        from cream import CreamCalculator

        atoms = bulk("Cu", "fcc", a=3.615) * (4, 4, 4)
        atoms.calc = CreamCalculator(pot_path)
        energy = atoms.get_potential_energy()
        assert np.isfinite(energy)
        epa = energy / len(atoms)
        assert -4.0 < epa < -3.0, f"energy/atom = {epa:.4f}"

    def test_calculator_forces(self, pot_path: str):
        from ase.build import bulk
        from cream import CreamCalculator

        # Use the orthorhombic 4×4×4 supercell for well-defined equilibrium
        atoms = bulk("Cu", "fcc", a=3.615, cubic=True) * (4, 4, 4)
        atoms.calc = CreamCalculator(pot_path)
        forces = atoms.get_forces()
        assert forces.shape == (len(atoms), 3)
        assert np.all(np.isfinite(forces))
        # Equilibrium forces should be small (f32 precision → ~1e-1 tolerance)
        assert np.max(np.abs(forces)) < 0.5

    def test_calculator_energies_gpu_raises(
        self, pot_path: str, require_gpu  # noqa: ARG002
    ):
        """On the GPU backend, per-atom energies are deliberately NOT in the
        calculator's implemented_properties — the GPU path does not read back
        the per-atom embedding+pair breakdown (pass1 produces it, but the
        energy_per_atom buffer is not mapped to the host for throughput
        reasons).  ASE must therefore raise PropertyNotImplementedError
        rather than silently returning an empty array (which was the old
        behaviour — wrong, because downstream code can't tell "this
        calculator never computes energies" from "energies happen to be
        all-zero here").
        """
        from ase.build import bulk
        from ase.calculators.calculator import PropertyNotImplementedError

        from cream import CreamCalculator

        atoms = bulk("Cu", "fcc", a=3.615) * (4, 4, 4)
        atoms.calc = CreamCalculator(pot_path, backend="gpu")
        with pytest.raises(PropertyNotImplementedError):
            atoms.get_potential_energies()

    def test_calculator_energies_cpu(self, pot_path: str, require_cpu):  # noqa: ARG002
        """On the CPU backend, per-atom energies are produced natively by
        the half-pair kernel and must be returned with the right shape and
        a sum that matches the total energy.

        Uses ``cubic=True`` — same reasoning as ``test_calculator_forces``.
        The primitive FCC cell that ASE's default ``bulk()`` returns is
        rhombohedral, with perpendicular height ≈ a·√(2/3)·nrep ≈ 8.3 Å
        at (4,4,4) which is below 2·cutoff for real Cu01 (cutoff ≈ 4.95 Å).
        The conventional cubic cell has perpendicular height = a·nrep ≈
        14.5 Å which comfortably exceeds 2·cutoff.
        """
        from ase.build import bulk

        from cream import CreamCalculator

        atoms = bulk("Cu", "fcc", a=3.615, cubic=True) * (4, 4, 4)
        atoms.calc = CreamCalculator(pot_path, backend="cpu")
        total = atoms.get_potential_energy()
        energies = atoms.get_potential_energies()
        assert energies.shape == (len(atoms),)
        assert np.isfinite(energies).all()
        assert abs(float(energies.sum()) - total) < 1e-3 * max(abs(total), 1.0)

    def test_calculator_repr_gpu(self, pot_path: str):
        from cream import CreamCalculator

        calc = CreamCalculator(pot_path, backend="gpu")
        r = repr(calc)
        assert "CreamCalculator" in r
        assert "Cu" in r
        assert "gpu" in r

    def test_calculator_cell_list(self, pot_path: str):
        from ase.build import bulk
        from cream import CreamCalculator

        atoms = bulk("Cu", "fcc", a=3.615) * (4, 4, 4)
        atoms.calc = CreamCalculator(pot_path, use_cell_list=True)
        energy = atoms.get_potential_energy()
        assert np.isfinite(energy)

    def test_calculator_no_pbc(self, pot_path: str):
        from ase.build import bulk
        from cream import CreamCalculator

        atoms = bulk("Cu", "fcc", a=3.615)
        atoms.pbc = False
        atoms.calc = CreamCalculator(pot_path)
        energy = atoms.get_potential_energy()
        assert np.isfinite(energy)

    def test_unknown_element_raises(self, pot_path: str):
        from ase import Atoms
        from cream import CreamCalculator

        # Fe is not in a Cu-only potential
        atoms = Atoms("Fe", positions=[[0, 0, 0]], pbc=False)
        atoms.calc = CreamCalculator(pot_path)
        with pytest.raises(ValueError, match="not in the potential"):
            atoms.get_potential_energy()

    def test_calculator_backend_property_gpu(self, pot_path: str):
        from cream import CreamCalculator

        calc = CreamCalculator(pot_path, backend="gpu")
        assert calc.backend == "gpu"


# ── ASE Calculator tests (CPU backend) ───────────────────────────────────────
#
# NOTE on cells: ASE bulk("Cu", "fcc") uses the primitive FCC cell
# whose perpendicular height ≈ 2.556 Å per unit cell repetition.
# The CPU engine requires height > 2×cutoff ≈ 11.014 Å, so we use
# cubic=True to get the conventional 4-atom cell (a ≈ 3.615 Å each side)
# and tile at least 4× in each direction (4*3.615 = 14.46 Å > 11.014 Å).


class TestASECalculatorCpu:
    """ASE Calculator tests using the CPU backend — no GPU required."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_ase(self):
        pytest.importorskip("ase")

    @pytest.fixture(autouse=True)
    def _skip_if_no_cpu(self, require_cpu):  # noqa: ARG002
        pass

    def test_calculator_cpu_energy(self, pot_path: str):
        from ase.build import bulk
        from cream import CreamCalculator

        # cubic=True + (4,4,4) → orthorhombic cell ≈ 14.46 Å (> 2×cutoff)
        atoms = bulk("Cu", "fcc", a=3.615, cubic=True) * (4, 4, 4)
        atoms.calc = CreamCalculator(pot_path, backend="cpu")
        energy = atoms.get_potential_energy()
        assert np.isfinite(energy)
        epa = energy / len(atoms)
        assert -4.0 < epa < -3.0, f"energy/atom = {epa:.4f}"

    def test_calculator_cpu_forces(self, pot_path: str):
        from ase.build import bulk
        from cream import CreamCalculator

        atoms = bulk("Cu", "fcc", a=3.615, cubic=True) * (4, 4, 4)
        atoms.calc = CreamCalculator(pot_path, backend="cpu")
        forces = atoms.get_forces()
        assert forces.shape == (len(atoms), 3)
        assert np.all(np.isfinite(forces))
        # Equilibrium forces should be small (f32 precision → ~0.5 eV/Å tolerance)
        assert np.max(np.abs(forces)) < 0.5

    def test_calculator_cpu_energies(self, pot_path: str):
        from ase.build import bulk
        from cream import CreamCalculator

        atoms = bulk("Cu", "fcc", a=3.615, cubic=True) * (4, 4, 4)
        atoms.calc = CreamCalculator(pot_path, backend="cpu")
        energies = atoms.get_potential_energies()
        assert energies.shape == (len(atoms),)
        # f32 accumulation over 256 atoms → relax tolerance to 1e-2
        np.testing.assert_allclose(energies.sum(), atoms.get_potential_energy(), atol=1e-2)

    def test_calculator_cpu_cell_list(self, pot_path: str):
        from ase.build import bulk
        from cream import CreamCalculator

        atoms = bulk("Cu", "fcc", a=3.615, cubic=True) * (4, 4, 4)
        atoms.calc = CreamCalculator(pot_path, use_cell_list=True, backend="cpu")
        energy = atoms.get_potential_energy()
        assert np.isfinite(energy)

    def test_calculator_cpu_no_pbc(self, pot_path: str):
        from ase.build import bulk
        from cream import CreamCalculator

        atoms = bulk("Cu", "fcc", a=3.615)
        atoms.pbc = False
        atoms.calc = CreamCalculator(pot_path, backend="cpu")
        energy = atoms.get_potential_energy()
        assert np.isfinite(energy)

    def test_calculator_backend_property_cpu(self, pot_path: str):
        from cream import CreamCalculator

        calc = CreamCalculator(pot_path, backend="cpu")
        assert calc.backend == "cpu"

    def test_calculator_repr_cpu(self, pot_path: str):
        from cream import CreamCalculator

        calc = CreamCalculator(pot_path, backend="cpu")
        r = repr(calc)
        assert "CreamCalculator" in r
        assert "cpu" in r

    def test_invalid_backend_raises(self, pot_path: str):
        from cream import CreamCalculator

        with pytest.raises(ValueError, match="backend"):
            CreamCalculator(pot_path, backend="tpu")

    def test_unknown_element_raises_cpu(self, pot_path: str):
        from ase import Atoms
        from cream import CreamCalculator

        atoms = Atoms("Fe", positions=[[0, 0, 0]], pbc=False)
        atoms.calc = CreamCalculator(pot_path, backend="cpu")
        with pytest.raises(ValueError, match="not in the potential"):
            atoms.get_potential_energy()
