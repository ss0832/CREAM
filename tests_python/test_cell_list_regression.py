# SPDX-License-Identifier: MIT
"""Regression tests for the CellList path (CPU and GPU).

These tests capture the bug scenarios reported in the test_results.log /
bug_report_3.txt trio:

1.  **CPU CellList ``is_ortho`` stencil shift**
    After ``atoms.rattle()`` some Cu atoms end up with fractional
    coordinates slightly outside [0, 1).  The old CPU path took the
    ``is_ortho == true`` fast branch that computes the PBC shift from cell
    index deltas without per-pair ``round()``; that shift is only exact
    when every atom sits in the primary image.  The fix wraps positions
    in ``compute_cell_list_sync``.

    Symptom before fix: Cubic / Tetragonal / Orthorhombic supercells
    showed 100–500 eV energy mismatch and 6–20 eV/Å force-difference
    norms vs the CPU AllPairs reference.

2.  **GPU CellList vs CPU AllPairs**
    The GPU CellList pipeline must agree with CPU AllPairs for every
    crystal system.  The GPU path already uses per-pair ``min_image`` so
    it should never have been affected by the CPU ortho bug, but
    NVIDIA-specific failures remain under investigation; once a fix
    lands, these tests keep it fixed.

The GPU-dependent tests are skipped gracefully when no adapter is
available (CI without a GPU, for example).
"""

from __future__ import annotations

import itertools
import pathlib

import numpy as np
import pytest


# ── Shared fixtures (complement tests_python/conftest.py) ────────────────────

POT_PATH_CANDIDATES = [
    pathlib.Path("Cu01_eam.alloy"),
    pathlib.Path(__file__).parent.parent / "Cu01_eam.alloy",
]


def _find_potential() -> str:
    for p in POT_PATH_CANDIDATES:
        if p.exists():
            return str(p)
    pytest.skip("Cu01_eam.alloy not found next to the test file or cwd")


# Deformation matrices for the 7 Bravais lattice systems.  Applied to a
# Cu-fcc cubic supercell (lattice parameter 4.915 Å) via
# ``atoms.set_cell(cell @ M, scale_atoms=True)`` to emulate the geometry
# while keeping the FCC-Cu neighbourhood intact — this matches the user's
# failing test case exactly.
SYSTEMS: dict[str, np.ndarray] = {
    "cubic": np.eye(3),
    "tetragonal": np.diag([1.0, 1.0, 1.2]),
    "orthorhombic": np.diag([1.0, 1.1, 1.2]),
    "hexagonal": np.array(
        [[1.0, -0.5, 0.0], [0.0, np.sqrt(3) / 2, 0.0], [0.0, 0.0, 1.2]]
    ),
    "rhombohedral": np.array(
        [[1.1, 0.1, 0.1], [0.1, 1.1, 0.1], [0.1, 0.1, 1.1]]
    ),
    "monoclinic": np.array(
        [[1.0, 0.0, 0.2], [0.0, 1.1, 0.0], [0.0, 0.0, 1.2]]
    ),
    "triclinic": np.array(
        [[1.00, 0.10, 0.20], [0.15, 1.10, 0.25], [0.05, 0.30, 1.20]]
    ),
}

# Keep supercell sizes small-to-moderate so the full-matrix of tests stays
# under ~a few minutes on a laptop GPU.  The ``is_ortho`` bug reproduces
# from size=4 onward, so this range is sufficient for regression.
SIZES = [4, 5, 6]

# Force-error tolerance.  CPU AllPairs uses f64 accumulators and returns
# f32; GPU uses f32 throughout with Kahan compensation.  At 256–864 atoms
# the cross-backend drift is dominated by lookup-table interpolation error
# (~1e-5 per atom); we pick a slightly loose cap to be noise-robust.
FORCE_TOL_EV_PER_ANG = 1e-3
ENERGY_TOL_EV_PER_ATOM = 1e-4


def _build_atoms(system: str, size: int):
    """Build a rattled Cu-fcc supercell shaped to the given Bravais system."""
    from ase.build import bulk  # imported lazily so `pytest --collect-only` stays fast

    atoms = bulk("Cu", "fcc", a=4.915, cubic=True) * (size, size, size)
    cell = atoms.get_cell()
    new_cell = np.dot(np.asarray(cell), SYSTEMS[system])
    atoms.set_cell(new_cell, scale_atoms=True)
    # stdev=0.05 Å is large enough to push a handful of atoms just outside
    # the primary image box — the exact condition that tripped the
    # is_ortho fast path.
    atoms.rattle(stdev=0.05, seed=42)
    return atoms


def _run(atoms, backend: str, use_cell_list: bool):
    from cream import CreamCalculator

    atoms.calc = CreamCalculator(
        _find_potential(), use_cell_list=use_cell_list, backend=backend
    )
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    return float(energy), np.asarray(forces, dtype=np.float64)


# ── Parametrised regression matrix ───────────────────────────────────────────

_CASES = list(itertools.product(SYSTEMS.keys(), SIZES))


def _case_id(system: str, size: int) -> str:
    return f"{system}-{size}x{size}x{size}"


@pytest.mark.parametrize(
    ("system", "size"), _CASES, ids=[_case_id(s, n) for s, n in _CASES]
)
def test_cpu_cell_list_matches_cpu_allpairs(system: str, size: int) -> None:
    """CPU CellList must agree with CPU AllPairs across every Bravais system.

    This is the direct regression check for the ``is_ortho`` fast-path
    bug: before the fix, Cubic / Tetragonal / Orthorhombic cases failed
    with hundreds of mismatched atoms and multi-eV force deviations.
    """
    atoms = _build_atoms(system, size)
    n = len(atoms)

    e_ap, f_ap = _run(atoms, backend="cpu", use_cell_list=False)
    e_cl, f_cl = _run(atoms, backend="cpu", use_cell_list=True)

    assert np.all(np.isfinite(f_cl)), f"NaN/Inf in CPU CellList forces ({system})"
    max_abs = np.max(np.abs(f_cl - f_ap))
    assert max_abs < FORCE_TOL_EV_PER_ANG, (
        f"[{system} {size}^3, N={n}] CPU CellList disagrees with CPU AllPairs: "
        f"max |Δf| = {max_abs:.3e} eV/Å (tol {FORCE_TOL_EV_PER_ANG})"
    )
    assert abs(e_cl - e_ap) / n < ENERGY_TOL_EV_PER_ATOM, (
        f"[{system} {size}^3] energy mismatch: {e_cl - e_ap:+.3e} eV "
        f"({(e_cl - e_ap) / n:+.3e} eV/atom, tol {ENERGY_TOL_EV_PER_ATOM})"
    )


@pytest.mark.parametrize(
    ("system", "size"), _CASES, ids=[_case_id(s, n) for s, n in _CASES]
)
def test_gpu_cell_list_matches_cpu_allpairs(
    system: str, size: int, require_gpu: None
) -> None:
    """GPU CellList must agree with CPU AllPairs across every Bravais system.

    Requires ``require_gpu`` fixture (defined in conftest.py) to skip
    gracefully when no GPU adapter is available.
    """
    atoms = _build_atoms(system, size)
    n = len(atoms)

    e_ap, f_ap = _run(atoms, backend="cpu", use_cell_list=False)
    e_cl, f_cl = _run(atoms, backend="gpu", use_cell_list=True)

    assert np.all(np.isfinite(f_cl)), f"NaN/Inf in GPU CellList forces ({system})"
    max_abs = np.max(np.abs(f_cl - f_ap))
    assert max_abs < FORCE_TOL_EV_PER_ANG, (
        f"[{system} {size}^3, N={n}] GPU CellList disagrees with CPU AllPairs: "
        f"max |Δf| = {max_abs:.3e} eV/Å (tol {FORCE_TOL_EV_PER_ANG})"
    )
    assert abs(e_cl - e_ap) / n < ENERGY_TOL_EV_PER_ATOM, (
        f"[{system} {size}^3] energy mismatch: {e_cl - e_ap:+.3e} eV "
        f"({(e_cl - e_ap) / n:+.3e} eV/atom, tol {ENERGY_TOL_EV_PER_ATOM})"
    )


@pytest.mark.parametrize(
    ("system", "size"), _CASES, ids=[_case_id(s, n) for s, n in _CASES]
)
def test_gpu_allpairs_matches_cpu_allpairs(
    system: str, size: int, require_gpu: None
) -> None:
    """Sanity check: GPU AllPairs ≡ CPU AllPairs (no CellList involved).

    If this ever fails, the bug is in the GPU potential-evaluation code
    itself, not in the CellList pipeline.
    """
    atoms = _build_atoms(system, size)
    n = len(atoms)

    e_cpu, f_cpu = _run(atoms, backend="cpu", use_cell_list=False)
    e_gpu, f_gpu = _run(atoms, backend="gpu", use_cell_list=False)

    max_abs = np.max(np.abs(f_gpu - f_cpu))
    assert max_abs < FORCE_TOL_EV_PER_ANG, (
        f"[{system} {size}^3, N={n}] GPU AllPairs disagrees with CPU AllPairs: "
        f"max |Δf| = {max_abs:.3e} eV/Å"
    )
    assert abs(e_gpu - e_cpu) / n < ENERGY_TOL_EV_PER_ATOM


# ── GPU CellList structural-invariant checks (shader diagnostic flags) ───────


@pytest.mark.parametrize(
    ("system", "size"),
    list(itertools.product(SYSTEMS.keys(), [4, 6])),
    ids=[f"{s}-{n}x{n}x{n}" for s, n in itertools.product(SYSTEMS.keys(), [4, 6])],
)
def test_gpu_cell_list_debug_flags_clean(
    system: str, size: int, require_gpu: None
) -> None:
    """No shader-side diagnostic counter should fire under normal inputs.

    Uses the ``compute_with_debug`` entry point exposed by PyO3 to read
    back the ``debug_flags`` atomic buffer written by pass1_cellist and
    pass2_cellist.  Every "error-class" slot must be zero:

    *  cid OOB, cell_start OOB / inverted
    *  bb empty (non-empty chunks)
    *  atom_k OOB inside cell iteration
    *  NaN rho, NaN force
    *  sorted_atoms OOB on force scatter

    The throughput counters (neighbor_cnt, cutoff_hits) are allowed to
    be any positive value; they're included in the return dict for
    performance inspection, not correctness.
    """
    from ase.data import atomic_numbers
    from cream import CreamEngine

    atoms = _build_atoms(system, size)
    n = len(atoms)
    cell = np.asarray(atoms.get_cell(), dtype=np.float64)
    positions = atoms.get_positions().astype(np.float64)

    # Map Z → potential index.  The test potential is pure Cu so index 0.
    engine = CreamEngine(_find_potential(), use_cell_list=True, backend="gpu")
    lut = np.full(120, -1, dtype=np.int32)
    for i, sym in enumerate(engine.elements):
        lut[atomic_numbers[sym]] = i
    atom_types = lut[atoms.get_atomic_numbers()].astype(np.int32)

    energy, forces, _epa, debug = engine.compute_with_debug(
        positions, atom_types, cell
    )
    assert debug is not None, "debug dict must be returned for GPU + CellList"

    # 1. Sanity: the permutation is a bijection of [0, N).
    sa = np.asarray(debug["sorted_atoms"])
    assert sa.shape == (n,)
    assert sa.min() == 0 and sa.max() == n - 1
    assert np.unique(sa).size == n, "sorted_atoms is not a permutation"

    # 2. cell_start must be monotone non-decreasing and end at n.
    cs = np.asarray(debug["cell_start"])
    assert cs.shape == (debug["n_morton"] + 1,)
    assert np.all(np.diff(cs) >= 0), "cell_start not monotone"
    assert cs[-1] == n

    # 3. cell_ids must all be within Morton capacity.
    ci = np.asarray(debug["cell_ids"])
    assert ci.max() < debug["n_morton"], "cell_ids OOB"

    # 4. reordered_positions[k, :3] == positions[sorted_atoms[k], :3].
    rp = np.asarray(debug["reordered_positions"])
    expected = positions[sa]
    # f32 round-trip tolerance.
    assert np.max(np.abs(rp[:, :3] - expected)) < 1e-4, (
        "reordered_positions do not match positions[sorted_atoms]"
    )

    # 5. Shader-side diagnostic counters: every error slot must be zero.
    dbg = np.asarray(debug["debug_flags"])
    # Slot layout (see src/engine.rs::CellListBuffers::debug_flags_buf).
    error_slots = {
        0: "p1_cid_oob",
        1: "p1_cs_oob",
        2: "p1_cs_inverted",
        4: "p1_atom_k_oob",
        5: "p1_nan_rho",
        8: "p2_cid_oob",
        9: "p2_cs_oob",
        10: "p2_cs_inverted",
        12: "p2_atom_k_oob",
        13: "p2_nan_force",
        16: "p2_sorted_atom_oob",
    }
    errors = {name: int(dbg[idx]) for idx, name in error_slots.items() if dbg[idx] > 0}
    assert not errors, f"[{system} {size}^3] GPU shader diagnostics fired: {errors}"

    # 6. For a dense crystal, at least some neighbour pairs must be inside cutoff.
    assert int(dbg[7]) > 0, "pass1 saw zero cutoff hits — empty neighbour lists?"
    assert int(dbg[15]) > 0, "pass2 saw zero cutoff hits — empty neighbour lists?"
