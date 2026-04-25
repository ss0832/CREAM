#!/usr/bin/env python3
"""
LAMMPS-style EAM/alloy Python reference for small-system verification.

This module intentionally prioritizes clarity and LAMMPS behavioral fidelity over speed.
It implements the DYNAMO setfl / .eam.alloy subset used by LAMMPS pair_style eam/alloy.

Limitations:
- Verification reference only; O(N^2), no neighbor list.
- Supports DYNAMO setfl / .eam.alloy, not funcfl .eam and not eam/fs.
- No MPI/newton-pair bookkeeping; evaluates each unordered pair once.
- ASE stress is implemented as -virial / volume using pair virial contributions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import math
import numpy as np

try:
    from ase.calculators.calculator import Calculator, all_changes
except Exception:  # pragma: no cover - ASE optional for low-level use
    Calculator = object  # type: ignore
    all_changes = None  # type: ignore


def _parse_float(tok: str) -> float:
    """Parse LAMMPS-style numeric tokens, accepting Fortran D exponents."""
    return float(tok.replace("D", "E").replace("d", "e"))


def _tokens_as_floats(line: str) -> list[float]:
    return [_parse_float(tok) for tok in line.split()]


def _read_n_floats(lines: list[str], start: int, n: int) -> tuple[np.ndarray, int]:
    """Read n floating point tokens from consecutive text lines.

    This parser expects each setfl logical table block to end on a line boundary,
    which is how LAMMPS/NIST setfl files are normally written. This avoids having
    to push back extra tokens when moving from one table block to the next.
    """
    out: list[float] = []
    i = start
    while len(out) < n and i < len(lines):
        stripped = lines[i].strip()
        if stripped and not stripped.startswith("#"):
            out.extend(_tokens_as_floats(stripped))
        i += 1
    if len(out) < n:
        raise ValueError(f"Unexpected EOF while reading {n} floats; got {len(out)}")
    if len(out) > n:
        raise ValueError(
            f"Table block over-read by {len(out) - n} numeric tokens. "
            "Expected the setfl table block to end on a line boundary."
        )
    return np.asarray(out, dtype=np.float64), i


def _lammps_interpolate_coeffs(values_1indexed: np.ndarray, delta: float) -> np.ndarray:
    """Reproduce LAMMPS PairEAM::interpolate.

    For interval m, LAMMPS evaluates:
        value(p) = ((c3*p + c4)*p + c5)*p + c6
        deriv(x) = (c0*p + c1)*p + c2
    where p is the fractional coordinate in the interval.
    """
    f = np.asarray(values_1indexed, dtype=np.float64)
    n = len(f) - 1
    if n < 4:
        raise ValueError("LAMMPS-style spline requires at least 4 tabulated values")

    s = np.zeros((n + 1, 7), dtype=np.float64)
    for m in range(1, n + 1):
        s[m, 6] = f[m]

    s[1, 5] = s[2, 6] - s[1, 6]
    s[2, 5] = 0.5 * (s[3, 6] - s[1, 6])
    s[n - 1, 5] = 0.5 * (s[n, 6] - s[n - 2, 6])
    s[n, 5] = s[n, 6] - s[n - 1, 6]

    for m in range(3, n - 1):  # LAMMPS: m = 3; m <= n-2; m++
        s[m, 5] = ((s[m - 2, 6] - s[m + 2, 6]) + 8.0 * (s[m + 1, 6] - s[m - 1, 6])) / 12.0

    for m in range(1, n):
        s[m, 4] = 3.0 * (s[m + 1, 6] - s[m, 6]) - 2.0 * s[m, 5] - s[m + 1, 5]
        s[m, 3] = s[m, 5] + s[m + 1, 5] - 2.0 * (s[m + 1, 6] - s[m, 6])

    s[n, 4] = 0.0
    s[n, 3] = 0.0

    for m in range(1, n + 1):
        s[m, 2] = s[m, 5] / delta
        s[m, 1] = 2.0 * s[m, 4] / delta
        s[m, 0] = 3.0 * s[m, 3] / delta
    return s


def _eval_spline_r(coeff: np.ndarray, r: float, dr: float) -> tuple[float, float]:
    """Evaluate LAMMPS r-grid spline value and derivative wrt r."""
    n = coeff.shape[0] - 1
    p = r / dr + 1.0
    m = int(p)
    m = min(m, n - 1)
    if m < 1:
        m = 1
    p -= m
    p = min(p, 1.0)
    c = coeff[m]
    val = ((c[3] * p + c[4]) * p + c[5]) * p + c[6]
    der = (c[0] * p + c[1]) * p + c[2]
    return float(val), float(der)


def _eval_spline_rho_lammps(coeff: np.ndarray, rho: float, drho: float) -> tuple[float, float, float]:
    """Evaluate embedding spline using LAMMPS PairEAM::compute conventions."""
    n = coeff.shape[0] - 1
    rhomax = (n - 1) * drho
    p = rho / drho + 1.0
    m = int(p)
    m = max(1, min(m, n - 1))
    p -= m
    p = min(p, 1.0)
    c = coeff[m]
    val = ((c[3] * p + c[4]) * p + c[5]) * p + c[6]
    der = (c[0] * p + c[1]) * p + c[2]
    return float(val), float(der), float(rhomax)


@dataclass(frozen=True)
class SetflEAMAlloy:
    comments: tuple[str, str, str]
    elements: tuple[str, ...]
    masses: np.ndarray
    nrho: int
    drho: float
    nr: int
    dr: float
    cutoff: float
    frho: np.ndarray      # (nelements, nrho+1), 1-indexed
    rhor: np.ndarray      # (nelements, nr+1), 1-indexed; donor-element density
    z2r: np.ndarray       # (nelements, nelements, nr+1), symmetric, 1-indexed; z2r = r*phi(r)
    frho_spline: np.ndarray
    rhor_spline: np.ndarray
    z2r_spline: np.ndarray

    @classmethod
    def from_file(cls, path: str | Path) -> "SetflEAMAlloy":
        path = Path(path)
        raw_lines = path.read_text().splitlines()
        if len(raw_lines) < 5:
            raise ValueError(f"Not enough lines for setfl file: {path}")

        comments = tuple(raw_lines[:3])  # type: ignore[arg-type]
        elem_parts = raw_lines[3].split()
        if len(elem_parts) < 2:
            raise ValueError("Invalid setfl element line")
        nelements = int(elem_parts[0])
        elements = tuple(elem_parts[1 : 1 + nelements])
        if len(elements) != nelements:
            raise ValueError("Element count does not match setfl element line")

        grid = raw_lines[4].split()
        if len(grid) < 5:
            raise ValueError("Invalid setfl grid line")
        nrho = int(grid[0])
        drho = _parse_float(grid[1])
        nr = int(grid[2])
        dr = _parse_float(grid[3])
        cutoff = _parse_float(grid[4])
        if nrho <= 0 or nr <= 0 or dr <= 0.0 or drho <= 0.0:
            raise ValueError("Invalid setfl grid values")

        masses = np.zeros(nelements, dtype=np.float64)
        frho = np.zeros((nelements, nrho + 1), dtype=np.float64)
        rhor = np.zeros((nelements, nr + 1), dtype=np.float64)
        z2r = np.zeros((nelements, nelements, nr + 1), dtype=np.float64)

        line_idx = 5
        for e in range(nelements):
            while line_idx < len(raw_lines) and not raw_lines[line_idx].strip():
                line_idx += 1
            meta = raw_lines[line_idx].split()
            if len(meta) < 2:
                raise ValueError(f"Invalid element metadata line for element {e}: {raw_lines[line_idx]!r}")
            # LAMMPS ignores atomic number and consumes mass; lattice const/type are ignored.
            masses[e] = _parse_float(meta[1])
            line_idx += 1

            vals, line_idx = _read_n_floats(raw_lines, line_idx, nrho)
            frho[e, 1:] = vals
            vals, line_idx = _read_n_floats(raw_lines, line_idx, nr)
            rhor[e, 1:] = vals

        for i in range(nelements):
            for j in range(i + 1):
                vals, line_idx = _read_n_floats(raw_lines, line_idx, nr)
                z2r[i, j, 1:] = vals
                z2r[j, i, 1:] = vals

        frho_spline = np.stack([_lammps_interpolate_coeffs(frho[e], drho) for e in range(nelements)])
        rhor_spline = np.stack([_lammps_interpolate_coeffs(rhor[e], dr) for e in range(nelements)])
        z2r_spline = np.empty((nelements, nelements, nr + 1, 7), dtype=np.float64)
        for i in range(nelements):
            for j in range(nelements):
                z2r_spline[i, j] = _lammps_interpolate_coeffs(z2r[i, j], dr)

        return cls(
            comments=comments,
            elements=elements,
            masses=masses,
            nrho=nrho,
            drho=drho,
            nr=nr,
            dr=dr,
            cutoff=cutoff,
            frho=frho,
            rhor=rhor,
            z2r=z2r,
            frho_spline=frho_spline,
            rhor_spline=rhor_spline,
            z2r_spline=z2r_spline,
        )

    @property
    def rhomax(self) -> float:
        return (self.nrho - 1) * self.drho

    def element_index(self, symbol: str) -> int:
        try:
            return self.elements.index(symbol)
        except ValueError as exc:
            raise KeyError(f"Element {symbol!r} is not present in potential elements {self.elements}") from exc

    def compute(
        self,
        positions: np.ndarray,
        atom_types: np.ndarray,
        cell: np.ndarray | None = None,
        pbc: Sequence[bool] | bool = True,
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute EAM/alloy energy, forces, per-atom energies, virial, and rho."""
        x = np.asarray(positions, dtype=np.float64)
        types = np.asarray(atom_types, dtype=np.int64)
        if x.ndim != 2 or x.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        if types.shape != (x.shape[0],):
            raise ValueError("atom_types must have shape (N,)")
        if np.any(types < 0) or np.any(types >= len(self.elements)):
            raise ValueError("atom_types contain indices outside potential element range")

        n_atoms = x.shape[0]
        forces = np.zeros_like(x)
        per_atom = np.zeros(n_atoms, dtype=np.float64)
        rho = np.zeros(n_atoms, dtype=np.float64)
        fp = np.zeros(n_atoms, dtype=np.float64)
        virial = np.zeros((3, 3), dtype=np.float64)

        use_pbc = cell is not None and np.any(pbc)
        if cell is not None:
            h = np.asarray(cell, dtype=np.float64)
            if h.shape != (3, 3):
                raise ValueError("cell must be None or shape (3,3), with lattice vectors as rows")
            hinv = np.linalg.inv(h)
            if isinstance(pbc, bool):
                pbc_arr = np.array([pbc, pbc, pbc], dtype=bool)
            else:
                pbc_arr = np.asarray(pbc, dtype=bool)
                if pbc_arr.shape != (3,):
                    raise ValueError("pbc must be bool or length-3 bool sequence")
        else:
            h = None
            hinv = None
            pbc_arr = np.array([False, False, False], dtype=bool)

        def minimum_image(delta: np.ndarray) -> np.ndarray:
            if not use_pbc:
                return delta
            # ASE/CREAM/LAMMPS row-vector convention: Cartesian d = fractional s @ H.
            s = delta @ hinv  # type: ignore[operator]
            s[pbc_arr] -= np.rint(s[pbc_arr])
            return s @ h  # type: ignore[operator]

        pairs: list[tuple[int, int, np.ndarray, float]] = []
        cut2 = self.cutoff * self.cutoff

        # Pass 1: density accumulation.
        for i in range(n_atoms - 1):
            ti = int(types[i])
            for j in range(i + 1, n_atoms):
                tj = int(types[j])
                del_ij = minimum_image(x[i] - x[j])
                rsq = float(np.dot(del_ij, del_ij))
                if rsq >= cut2:
                    continue
                r = math.sqrt(rsq)
                if r == 0.0:
                    raise ValueError(f"Atoms {i} and {j} overlap exactly")

                # LAMMPS eam/alloy density at i receives contribution from donor j.
                rho_j_to_i, _ = _eval_spline_r(self.rhor_spline[tj], r, self.dr)
                rho_i_to_j, _ = _eval_spline_r(self.rhor_spline[ti], r, self.dr)
                rho[i] += rho_j_to_i
                rho[j] += rho_i_to_j
                pairs.append((i, j, del_ij, r))

        # Pass 2: embedding energy and derivative.
        total_energy = 0.0
        for i in range(n_atoms):
            ti = int(types[i])
            emb, fprime, rhomax = _eval_spline_rho_lammps(self.frho_spline[ti], float(rho[i]), self.drho)
            if rho[i] > rhomax:
                emb += fprime * (rho[i] - rhomax)
            fp[i] = fprime
            per_atom[i] += emb
            total_energy += emb

        # Pass 3: pair energy and forces.
        for i, j, del_ij, r in pairs:
            ti = int(types[i])
            tj = int(types[j])

            _val, rhoip = _eval_spline_r(self.rhor_spline[ti], r, self.dr)  # density at j due to i, derivative
            _val, rhojp = _eval_spline_r(self.rhor_spline[tj], r, self.dr)  # density at i due to j, derivative

            z2, z2p = _eval_spline_r(self.z2r_spline[ti, tj], r, self.dr)
            recip = 1.0 / r
            phi = z2 * recip
            phip = z2p * recip - phi * recip

            psip = fp[i] * rhojp + fp[j] * rhoip + phip
            fpair = -psip * recip
            fij = del_ij * fpair

            forces[i] += fij
            forces[j] -= fij
            virial += np.outer(del_ij, fij)

            total_energy += phi
            half_phi = 0.5 * phi
            per_atom[i] += half_phi
            per_atom[j] += half_phi

        return float(total_energy), forces, per_atom, virial, rho


class EAMAlloyReferenceCalculator(Calculator):
    """ASE Calculator wrapper around the LAMMPS-style Python reference."""

    implemented_properties = ["energy", "forces", "energies", "stress"]

    def __init__(self, potential: str | Path, element_order: Sequence[str] | None = None, **kwargs):
        if Calculator is object:
            raise ImportError("ASE is required to use EAMAlloyReferenceCalculator")
        super().__init__(**kwargs)
        self.potential = SetflEAMAlloy.from_file(potential)
        self.element_order = tuple(element_order) if element_order is not None else self.potential.elements
        for el in self.element_order:
            self.potential.element_index(el)

    def _types_from_atoms(self, atoms) -> np.ndarray:
        symbols = atoms.get_chemical_symbols()
        mapping = {el: self.potential.element_index(el) for el in self.element_order}
        try:
            return np.array([mapping[s] for s in symbols], dtype=np.int64)
        except KeyError as exc:
            raise KeyError(
                f"Atom symbol {exc.args[0]!r} not in element_order={self.element_order}; "
                f"potential elements={self.potential.elements}"
            ) from exc

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        atoms = self.atoms
        positions = atoms.get_positions()
        types = self._types_from_atoms(atoms)
        pbc = atoms.get_pbc()
        cell = atoms.get_cell().array if np.any(pbc) else None

        energy, forces, per_atom, virial, rho = self.potential.compute(
            positions=positions,
            atom_types=types,
            cell=cell,
            pbc=pbc,
        )
        self.results["energy"] = energy
        self.results["forces"] = forces
        self.results["energies"] = per_atom
        self.results["rho"] = rho
        self.results["virial"] = virial

        if cell is not None:
            volume = atoms.get_volume()
            if volume > 0.0:
                stress_tensor = -virial / volume
                self.results["stress"] = np.array(
                    [
                        stress_tensor[0, 0],
                        stress_tensor[1, 1],
                        stress_tensor[2, 2],
                        stress_tensor[1, 2],
                        stress_tensor[0, 2],
                        stress_tensor[0, 1],
                    ],
                    dtype=np.float64,
                )
            else:
                self.results["stress"] = np.zeros(6)
        else:
            self.results["stress"] = np.zeros(6)


def compute_from_ase_atoms(atoms, potential_path: str | Path, element_order: Sequence[str] | None = None):
    """Convenience function returning reference energy, forces, per-atom energy, virial, rho."""
    pot = SetflEAMAlloy.from_file(potential_path)
    order = tuple(element_order) if element_order is not None else pot.elements
    mapping = {el: pot.element_index(el) for el in order}
    atom_types = np.array([mapping[s] for s in atoms.get_chemical_symbols()], dtype=np.int64)
    cell = atoms.get_cell().array if np.any(atoms.get_pbc()) else None
    return pot.compute(atoms.get_positions(), atom_types, cell=cell, pbc=atoms.get_pbc())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Small smoke test for LAMMPS-style EAM/alloy reference.")
    parser.add_argument("potential", help="Path to .eam.alloy setfl potential, e.g. Cu01.eam.alloy")
    parser.add_argument("--element", default="Cu", help="Element symbol for ASE fcc smoke test")
    parser.add_argument("--a", type=float, default=3.615, help="FCC lattice constant in Angstrom")
    parser.add_argument("--repeat", type=int, default=2, help="FCC repetition in each direction")
    args = parser.parse_args()

    from ase.build import bulk

    atoms = bulk(args.element, "fcc", a=args.a) * (args.repeat, args.repeat, args.repeat)
    atoms.calc = EAMAlloyReferenceCalculator(args.potential)
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    print(f"n_atoms         : {len(atoms)}")
    print(f"energy          : {e:.12f} eV")
    print(f"energy_per_atom : {e/len(atoms):.12f} eV/atom")
    print(f"force_norm_max  : {np.linalg.norm(f, axis=1).max():.6e} eV/Angstrom")
    print(f"force_sum_norm  : {np.linalg.norm(f.sum(axis=0)):.6e} eV/Angstrom")
