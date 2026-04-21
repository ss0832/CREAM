# SPDX-License-Identifier: LGPL-3.0-or-later
# This file links against ASE (https://wiki.fysik.dtu.dk/ase/), which is
# distributed under the GNU Lesser General Public License v3 or later.
# The remainder of the CREAM project is MIT-licensed (see LICENSE).
"""ASE Calculator interface for CREAM.

Provides :class:`CreamCalculator`, an ASE ``Calculator`` subclass that
delegates energy/force evaluation to the CREAM Rust backend via
:class:`cream.CreamEngine`.

The backend can be selected explicitly:

- ``backend="gpu"`` (default) — wgpu GPU accelerated path.
- ``backend="cpu"`` — rayon-parallelised CPU path; always available even
  when no GPU adapter is present.

Usage::

    from ase.build import bulk
    from cream import CreamCalculator

    # GPU (default)
    atoms = bulk('Cu', 'fcc', a=3.615) * (5, 5, 5)
    atoms.calc = CreamCalculator('Cu.eam.alloy', use_cell_list=True)

    # CPU – no GPU required, exposes per-atom physics
    atoms.calc = CreamCalculator('Cu.eam.alloy', backend='cpu')

    energy = atoms.get_potential_energy()   # eV
    forces = atoms.get_forces()             # eV/Å, shape (N, 3)
    stress = atoms.get_stress()             # Voigt (6,) eV/Å³
    stresses = atoms.get_stresses()         # (N, 6)  — CPU backend only

Backend-dependent properties:

==============================  ===  ===
Property                        GPU  CPU
==============================  ===  ===
energy, forces, stress          yes  yes
energies (per-atom energy)      no   yes
stresses (per-atom virial/vol)  no   yes
cream_densities                 no   yes
cream_embedding_energies        no   yes
==============================  ===  ===

``implemented_properties`` is set at construction time to reflect the
active backend, so ``atoms.get_stresses()`` will raise a standard ASE
``PropertyNotImplementedError`` on the GPU path instead of silently
returning wrong data.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
from ase.calculators.calculator import Calculator, all_changes

from cream._cream import CreamEngine

# Valid backend identifiers exposed by the Rust layer.
_VALID_BACKENDS = frozenset({"gpu", "cpu"})

# Properties every CREAM backend provides.
_BASE_PROPERTIES = ("energy", "forces", "stress")
# Extra properties populated only by the CPU backend (per-atom physics).
_CPU_EXTRA_PROPERTIES = (
    "energies",
    "stresses",
    "cream_densities",
    "cream_embedding_energies",
)


class CreamCalculator(Calculator):
    """ASE Calculator backed by the CREAM engine.

    Parameters
    ----------
    potential_file : str or Path
        Path to a ``.eam.alloy`` potential file.
    use_cell_list : bool
        Use O(N) Cell List neighbour search instead of O(N²) AllPairs.
        Recommended for N > ~1000.
    cell_size : float or None
        Override Cell List edge length [Å].  Defaults to the potential
        cutoff radius.
    backend : str
        Compute backend: ``"gpu"`` (default) or ``"cpu"``.
        ``"cpu"`` is always available and requires no GPU adapter.
    label : str
        Calculator label (ASE convention).

    Notes
    -----
    CREAM requires either fully periodic (``pbc=True`` on all axes) or
    fully non-periodic (``pbc=False`` on all axes) boundary conditions.
    Partial PBC (e.g. slab geometries with ``pbc=[True, True, False]``)
    is not supported and will raise :class:`ValueError` at compute time.
    For slab calculations, use a large vacuum layer and set ``pbc=False``.

    **CPU backend -- Rayon thread warm-up**

    When using ``backend="cpu"``, Rayon worker threads park themselves after
    ~100 ms of inactivity.  ``CreamEngine`` keeps them warm automatically
    via an internal keep-alive thread (started at construction), suppressing
    latency spikes mid-loop as long as the calculator instance is alive.

    For the very first call after construction the Rayon thread pool may
    not yet be fully initialised.  If sub-millisecond first-call latency
    matters (e.g. benchmarking individual steps), add one explicit warm-up
    call before the timed section::

        calc = CreamCalculator("Cu.eam.alloy", backend="cpu",
                               use_cell_list=True)
        atoms.calc = calc
        atoms.get_potential_energy()   # warm-up: initialises Rayon pool
        # timed loop starts here
        for step in md_loop:
            atoms.get_potential_energy()
    """

    # Default fallback; the instance-level list in __init__ supersedes this.
    implemented_properties = list(_BASE_PROPERTIES) + list(_CPU_EXTRA_PROPERTIES)

    def __init__(
        self,
        potential_file: str | Path,
        *,
        use_cell_list: bool = False,
        cell_size: float | None = None,
        backend: str = "gpu",
        label: str = "cream",
        **kwargs: object,
    ) -> None:
        super().__init__(label=label, **kwargs)

        if backend not in _VALID_BACKENDS:
            raise ValueError(
                f"Unknown backend {backend!r}. Valid choices are {sorted(_VALID_BACKENDS)}."
            )

        self._pot_path = str(potential_file)
        self._use_cell_list = use_cell_list
        self._cell_size = cell_size
        self._backend = backend

        # Eagerly create the engine so init errors surface immediately.
        self._engine = CreamEngine(
            self._pot_path,
            use_cell_list=use_cell_list,
            cell_size=cell_size,
            backend=backend,
        )

        # Advertise only what this backend can actually produce.
        # ASE will raise PropertyNotImplementedError for anything else,
        # which is much safer than silently returning zeros or wrong data.
        if self._engine.backend == "cpu":
            self.implemented_properties = list(_BASE_PROPERTIES) + list(
                _CPU_EXTRA_PROPERTIES
            )
        else:
            self.implemented_properties = list(_BASE_PROPERTIES)

    # ── ASE Calculator protocol ───────────────────────────────────────────

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ) -> None:
        """Compute energy and forces for the given ``Atoms`` object.

        This method is called internally by ASE when you invoke
        ``atoms.get_potential_energy()`` or ``atoms.get_forces()``.

        Raises
        ------
        ValueError
            If partial periodic boundary conditions are detected.
            CREAM supports only fully-periodic or fully non-periodic
            systems.
        """
        super().calculate(atoms, properties, system_changes)

        positions = self.atoms.get_positions()  # (N, 3) float64
        atom_types = self._map_atom_types(self.atoms)  # (N,) int32

        # Validate and resolve PBC.
        #
        # CREAM has no partial-PBC mode: the Rust engine treats cell=Some(…)
        # as 3-D periodic and cell=None as fully non-periodic.  Passing a cell
        # for a partially-periodic system would silently apply wrong-axis PBC,
        # producing incorrect forces with no error raised.
        #
        # Policy: reject partial PBC eagerly.  Users who need slab geometries
        # should add a vacuum layer and set pbc=False.
        pbc = self.atoms.pbc  # bool array, shape (3,)
        if np.any(pbc) and not np.all(pbc):
            raise ValueError(
                f"CreamCalculator does not support partial periodic boundary "
                f"conditions (got pbc={list(pbc)}).  CREAM requires either "
                f"fully periodic (pbc=True) or fully non-periodic (pbc=False). "
                f"For slab geometries, add a vacuum layer and set pbc=False."
            )

        cell: np.ndarray | None = (
            np.array(self.atoms.get_cell(), dtype=np.float64) if np.all(pbc) else None
        )

        # Decide which engine call to dispatch to.
        #
        # Per-atom extras (stresses/densities/embedding) are CPU-only and
        # expensive to ask for when not needed; only call ``compute_per_atom``
        # if at least one CPU-only property was requested.  Otherwise use
        # ``compute_stress`` which is now GPU-native for the stress tensor.
        props = set(properties or ())
        need_per_atom = bool(props & set(_CPU_EXTRA_PROPERTIES))
        need_stress = "stress" in props or need_per_atom

        if need_per_atom:
            if self._engine.backend != "cpu":
                # Should not be reachable — implemented_properties guards this
                # in well-behaved ASE workflows — but defensive raise here
                # keeps the failure mode local and informative.
                from ase.calculators.calculator import PropertyNotImplementedError
                raise PropertyNotImplementedError(
                    "Per-atom stress / density / embedding energy require "
                    "backend='cpu'. Construct CreamCalculator with backend='cpu'."
                )
            (
                energy,
                forces,
                energy_per_atom,
                stress_voigt,
                virial_per_atom,
                densities,
                embedding_energies,
            ) = self._engine.compute_per_atom(positions, atom_types, cell)

            self.results["stress"] = np.asarray(stress_voigt, dtype=np.float64)

            # Per-atom stress: convert raw atomic virial (eV) to stress (eV/Å³).
            # Convention: equal-partition atomic volume Ω_i = V/N, so
            #   σ_atom_i = −virial_per_atom[i] / Ω_i  =  −N · virial_per_atom[i] / V.
            # This makes Σᵢ σ_atom_i · Ω_i = σ_total · V, i.e. the sum of
            # per-atom stresses equals the global stress (as expected).
            n_atoms = positions.shape[0]
            vol = float(abs(np.linalg.det(cell))) if cell is not None else 0.0
            if vol > 0.0:
                stresses = (-float(n_atoms) / vol) * np.asarray(
                    virial_per_atom, dtype=np.float64
                )
            else:
                # Non-periodic cluster — per-atom stress is undefined.
                stresses = np.zeros_like(np.asarray(virial_per_atom), dtype=np.float64)

            self.results["stresses"] = stresses
            # Raw atomic virial (eV units, LAMMPS convention) — exposed under
            # a namespaced key in case downstream tooling prefers it to the
            # derived per-atom stress.
            self.results["cream_atomic_virials"] = np.asarray(
                virial_per_atom, dtype=np.float64
            )
            self.results["cream_densities"] = np.asarray(densities, dtype=np.float64)
            self.results["cream_embedding_energies"] = np.asarray(
                embedding_energies, dtype=np.float64
            )

        elif need_stress:
            energy, forces, energy_per_atom, stress_voigt = self._engine.compute_stress(
                positions, atom_types, cell
            )
            self.results["stress"] = np.asarray(stress_voigt, dtype=np.float64)
        else:
            energy, forces, energy_per_atom = self._engine.compute(
                positions, atom_types, cell
            )

        self.results["energy"] = energy
        # Per-atom energy is populated only by the CPU backend (empty array
        # otherwise); we expose it unconditionally so existing consumers that
        # call `atoms.get_potential_energies()` on a CPU calculator keep
        # working, but ASE will reject the call on the GPU path via
        # implemented_properties.
        self.results["energies"] = energy_per_atom
        self.results["forces"] = forces

    # ── Helpers ────────────────────────────────────────────────────────────

    def _map_atom_types(self, atoms) -> npt.NDArray[np.int32]:
        """Map ASE atomic numbers to potential element indices.

        The mapping is built from the element list embedded in the
        ``.eam.alloy`` file (available as ``self._engine.elements``).
        For example, if the potential file declares ``Cu Ag``, then
        Cu → 0, Ag → 1.

        Uses a pre-built NumPy lookup table (shape (120,)) for O(1) per-species
        dispatch instead of a Python loop, giving ~50-100× speedup at large N.
        """
        from ase.data import atomic_numbers

        pot_elements = self._engine.elements  # e.g. ["Cu", "Ag"]

        # Build a lookup table indexed by atomic number (max Z ≈ 118 + margin).
        lut: np.ndarray = np.full(120, -1, dtype=np.int32)
        for i, sym in enumerate(pot_elements):
            lut[atomic_numbers[sym]] = i

        numbers = atoms.get_atomic_numbers()  # shape (N,), dtype int

        # Validate: any element absent from the potential maps to -1.
        types = cast(npt.NDArray[np.int32], lut[numbers])
        missing_mask = types < 0
        if np.any(missing_mask):
            from ase.data import chemical_symbols

            missing_syms = sorted({chemical_symbols[z] for z in numbers[missing_mask]})
            raise ValueError(
                f"Elements {missing_syms} are not in the potential file (available: {pot_elements})"
            )

        return types

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def backend(self) -> str:
        """Active compute backend (``"gpu"`` or ``"cpu"``)."""
        return self._engine.backend

    def __repr__(self) -> str:
        mode = "CellList" if self._use_cell_list else "AllPairs"
        elems = ",".join(self._engine.elements)
        return f"CreamCalculator({elems}, {mode}, backend={self._backend})"
