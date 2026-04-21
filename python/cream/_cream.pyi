"""Type stubs for the native cream._cream module (built by maturin/PyO3)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

class CreamEngine:
    """EAM force/energy calculator with selectable compute backend.

    Parameters
    ----------
    potential_file : str
        Path to a ``.eam.alloy`` potential file.
    use_cell_list : bool, optional
        If ``True``, use O(N) Cell List neighbour search (default ``False``).
    cell_size : float or None, optional
        Cell size for Cell List [Å]. Defaults to the potential cutoff.
    backend : str, optional
        Compute backend: ``"gpu"`` (default) or ``"cpu"``.
        Use ``"cpu"`` when no GPU adapter is available or for
        reproducible CPU-only execution.
    """

    def __init__(
        self,
        potential_file: str,
        *,
        use_cell_list: bool = False,
        cell_size: float | None = None,
        backend: str = "gpu",
    ) -> None: ...
    def compute(
        self,
        positions: npt.NDArray[np.float64],
        atom_types: npt.NDArray[np.int32],
        cell: npt.NDArray[np.float64] | None = None,
    ) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
    def compute_stress(
        self,
        positions: npt.NDArray[np.float64],
        atom_types: npt.NDArray[np.int32],
        cell: npt.NDArray[np.float64] | None = None,
    ) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Compute energy, forces, per-atom energy, and Voigt stress tensor.

        Both backends produce the stress tensor natively — the GPU path
        reduces six virial partials per workgroup alongside the energy
        partial with no extra barriers, so this is effectively as cheap
        as :py:meth:`compute` for the GPU backend.

        Returns
        -------
        energy : float
            Total potential energy [eV].
        forces : ndarray, shape (N, 3), float64
            Forces [eV/Å].
        energy_per_atom : ndarray, shape (N,), float64
            Per-atom energy [eV].  Empty for the GPU backend.
        stress : ndarray, shape (6,), float64
            Voigt stress ``[σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy]`` [eV/Å³].
            Multiply by 160.21766 to convert to GPa.
            For non-periodic systems (cell=None) this is always zero.
        """
        ...
    def compute_per_atom(
        self,
        positions: npt.NDArray[np.float64],
        atom_types: npt.NDArray[np.int32],
        cell: npt.NDArray[np.float64] | None = None,
    ) -> tuple[
        float,
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """Return per-atom physics (CPU backend only).

        In addition to everything :py:meth:`compute_stress` returns, this
        produces three CPU-only quantities:

        * **virial_per_atom** (N, 6) eV — raw pair-virial contribution
          assigned half-and-half per atom (LAMMPS ``compute stress/atom``
          convention).  Sum over atoms equals ``−stress × V``.
        * **densities** (N,) — EAM electron density :math:`\\rho_i`.
        * **embedding_energies** (N,) eV — :math:`F_\\alpha(\\rho_i)`.

        Returns
        -------
        energy : float
        forces : ndarray (N, 3)
        energy_per_atom : ndarray (N,)
        stress : ndarray (6,)
        virial_per_atom : ndarray (N, 6)
        densities : ndarray (N,)
        embedding_energies : ndarray (N,)

        Raises
        ------
        NotImplementedError
            If the engine was constructed with ``backend="gpu"``.
        """
        ...
    def compute_with_debug(
        self,
        positions: npt.NDArray[np.float64],
        atom_types: npt.NDArray[np.int32],
        cell: npt.NDArray[np.float64] | None = None,
    ) -> tuple[
        float,
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        dict[str, object] | None,
    ]:
        """Run the GPU pipeline and return every intermediate CellList buffer.

        Returns
        -------
        (energy, forces, energy_per_atom, debug)
            ``debug`` is ``None`` for the CPU backend or when CellList is
            inactive.  Otherwise it is a ``dict`` with keys:

            * ``n_atoms`` (int), ``n_cells`` (tuple), ``n_cells_pad`` (tuple),
              ``n_morton`` (int)
            * ``cell_ids``            (N,)          uint32
            * ``sorted_atoms``        (N,)          uint32
            * ``cell_start``          (n_morton+1,) uint32
            * ``cell_counts``         (n_morton,)   uint32
            * ``reordered_positions`` (N, 4)        float32
            * ``reordered_types``     (N,)          uint32
            * ``densities``           (N,)          float32
            * ``debug_flags``         (32,)         uint32

        Raises
        ------
        NotImplementedError
            If the engine was constructed with ``backend="cpu"``.
        """
        ...
    @property
    def elements(self) -> list[str]: ...
    @property
    def cutoff(self) -> float: ...
    @property
    def n_elements(self) -> int: ...
    @property
    def backend(self) -> str:
        """Active compute backend: ``"gpu"`` or ``"cpu"``."""
        ...
