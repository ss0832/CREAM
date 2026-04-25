#!/usr/bin/env python3
"""Compare CREAM's ASE calculator against the independent LAMMPS-style reference.

v5 changes:
- Keeps Cu/single-element cases from v4.
- Adds automatic binary-alloy test generation for 2+ element .eam.alloy files.
- For Mishin-Ni-Al-2009.eam.alloy, adds pure Ni/Al, B2-NiAl, L1_2-like,
  random alloy, antisite, vacancy, atom-order permutation, PBC-boundary,
  and species-resolved cutoff-pair cases.
- Treats this as an external reference comparison, not an internal backend test.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from ase import Atoms
from ase.build import bulk, fcc111

from eam_alloy_reference import EAMAlloyReferenceCalculator, SetflEAMAlloy

EV_PER_A3_TO_GPA = 160.21766208

# Rough lattice constants used only for verification geometry generation.
# They need not be fitted equilibrium values; they should produce reasonable,
# non-overlapping solids with cell sizes large enough for minimum-image PBC.
DEFAULT_FCC_A = {
    "Cu": 3.615,
    "Ni": 3.52,
    "Al": 4.05,
}
DEFAULT_B2_A = {
    frozenset(("Ni", "Al")): 2.887,
}
DEFAULT_L12_A = {
    frozenset(("Ni", "Al")): 3.57,
}


def _min_repeat_for_length(length: float, cutoff: float, margin: float = 1.05) -> int:
    if length <= 0.0:
        raise ValueError(f"cell length must be positive, got {length}")
    return max(1, int(math.ceil((2.0 * cutoff * margin) / length)))


def _min_bulk_repeat_for_cutoff(a: float, cutoff: float, margin: float = 1.05) -> int:
    return _min_repeat_for_length(a, cutoff, margin=margin)


def _min_slab_xy_repeat_for_cutoff(a: float, cutoff: float, margin: float = 1.05) -> int:
    lateral = a / math.sqrt(2.0)
    return _min_repeat_for_length(lateral, cutoff, margin=margin)


def _safe_periodic_box_length(cutoff: float, margin: float = 1.05) -> float:
    return 2.0 * cutoff * margin + 1.0


def _lammps_order_to_ase_order(v: np.ndarray) -> np.ndarray:
    """LAMMPS xx,yy,zz,xy,xz,yz -> ASE xx,yy,zz,yz,xz,xy."""
    v = np.asarray(v, dtype=np.float64)
    return np.array([v[0], v[1], v[2], v[5], v[4], v[3]], dtype=np.float64)


def _formula_label(symbols: Sequence[str]) -> str:
    """Compact stable label, e.g. ['Ni','Ni','Al'] -> Ni2Al."""
    out: list[str] = []
    i = 0
    while i < len(symbols):
        s = symbols[i]
        j = i + 1
        while j < len(symbols) and symbols[j] == s:
            j += 1
        n = j - i
        out.append(s if n == 1 else f"{s}{n}")
        i = j
    return "".join(out)


def _default_fcc_a(element: str, fallback: float) -> float:
    return float(DEFAULT_FCC_A.get(element, fallback))


def _default_b2_a(e0: str, e1: str, fallback: float) -> float:
    return float(DEFAULT_B2_A.get(frozenset((e0, e1)), fallback))


def _default_l12_a(e0: str, e1: str, fallback: float) -> float:
    return float(DEFAULT_L12_A.get(frozenset((e0, e1)), fallback))


def _make_b2(e0: str, e1: str, a: float) -> Atoms:
    return Atoms(
        symbols=[e0, e1],
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        cell=np.diag([a, a, a]),
        pbc=True,
    )


def _make_l12(a_species: str, b_species: str, a: float) -> Atoms:
    """Conventional L1_2 A3B cell.

    B at cube corner, A at face centers. This is a geometry/mapping test;
    it does not assert thermodynamic stability for every A/B choice.
    """
    return Atoms(
        symbols=[b_species, a_species, a_species, a_species],
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ],
        cell=np.diag([a, a, a]),
        pbc=True,
    )


def _symbols_set(atoms: Atoms, symbols: Sequence[str]) -> Atoms:
    atoms = atoms.copy()
    if len(symbols) != len(atoms):
        raise ValueError("symbol list length must match atoms length")
    atoms.set_chemical_symbols(list(symbols))
    return atoms


def _repeat_to_safe(atoms: Atoms, cutoff: float, margin: float = 1.05) -> Atoms:
    """Repeat a periodic unit cell until all active PBC heights are safely > 2*cutoff.

    This is exact for the orthorhombic/cubic cells generated here. For sheared
    variants, we generate the safe cubic/orthorhombic cell first and then apply
    the homogeneous deformation.
    """
    lengths = atoms.cell.lengths()
    reps = []
    for length, periodic in zip(lengths, atoms.get_pbc()):
        reps.append(_min_repeat_for_length(float(length), cutoff, margin=margin) if periodic else 1)
    return atoms * tuple(reps)


def make_elemental_cases(
    potential: SetflEAMAlloy,
    element: str = "Cu",
    a: float = 3.615,
    seed: int = 20260425,
    include_slab: bool = True,
):
    rng = np.random.default_rng(seed)
    cases: list[Atoms] = []
    cutoff = potential.cutoff
    rep = _min_bulk_repeat_for_cutoff(a, cutoff)
    slab_rep = _min_slab_xy_repeat_for_cutoff(a, cutoff)
    safe_l = _safe_periodic_box_length(cutoff)

    def add(name: str, atoms: Atoms):
        atoms.info["case_name"] = name
        cases.append(atoms)

    def fcc_supercell(scale: float = 1.0) -> Atoms:
        return bulk(element, "fcc", a=a * scale, cubic=True) * (rep, rep, rep)

    add(f"{element}_fcc_{rep}x{rep}x{rep}_equilibrium", fcc_supercell())

    atoms = fcc_supercell()
    atoms.positions += rng.normal(scale=0.01, size=atoms.positions.shape)
    atoms.wrap()
    add(f"{element}_fcc_{rep}x{rep}x{rep}_random_disp_0p01A", atoms)

    atoms = fcc_supercell()
    atoms.positions += rng.normal(scale=0.05, size=atoms.positions.shape)
    atoms.wrap()
    add(f"{element}_fcc_{rep}x{rep}x{rep}_random_disp_0p05A", atoms)

    add(f"{element}_fcc_{rep}x{rep}x{rep}_compressed_4pct", fcc_supercell(scale=0.96))
    add(f"{element}_fcc_{rep}x{rep}x{rep}_expanded_4pct", fcc_supercell(scale=1.04))

    atoms = fcc_supercell()
    cell = atoms.cell.array.copy()
    cell[0, 1] += 0.07 * cell[1, 1]
    atoms.set_cell(cell, scale_atoms=True)
    add(f"{element}_fcc_{rep}x{rep}x{rep}_shear_xy_7pct", atoms)

    atoms = fcc_supercell()
    del atoms[len(atoms) // 2]
    add(f"{element}_fcc_{rep}x{rep}x{rep}_single_vacancy", atoms)

    atoms = fcc_supercell()
    order = np.arange(len(atoms))
    rng.shuffle(order)
    atoms = atoms[order]
    add(f"{element}_fcc_{rep}x{rep}x{rep}_atom_order_permuted", atoms)

    nn = a / math.sqrt(2.0)
    atoms = Atoms(
        [element, element],
        positions=[[0.25, safe_l / 2.0, safe_l / 2.0], [safe_l - nn + 0.25, safe_l / 2.0, safe_l / 2.0]],
        cell=np.diag([safe_l, safe_l, safe_l]),
        pbc=True,
    )
    add(f"{element}_pbc_boundary_two_atoms_safe_box", atoms)

    add(f"{element}_cluster_dimer_nn_distance", Atoms([element, element], positions=[[0, 0, 0], [a / math.sqrt(2), 0, 0]], pbc=False))
    add(f"{element}_cluster_pair_just_below_cutoff", Atoms([element, element], positions=[[0, 0, 0], [0.999 * cutoff, 0, 0]], pbc=False))
    add(f"{element}_cluster_pair_just_above_cutoff", Atoms([element, element], positions=[[0, 0, 0], [1.001 * cutoff, 0, 0]], pbc=False))

    if include_slab:
        slab = fcc111(element, size=(slab_rep, slab_rep, 4), a=a, vacuum=cutoff + 3.0, periodic=True)
        add(f"{element}_fcc111_slab_{slab_rep}x{slab_rep}x4_vacuum", slab)

    return cases


def make_binary_cases(
    potential: SetflEAMAlloy,
    e0: str | None = None,
    e1: str | None = None,
    a0: float | None = None,
    a1: float | None = None,
    b2_a: float | None = None,
    l12_a: float | None = None,
    random_fcc_a: float | None = None,
    seed: int = 20260425,
):
    if len(potential.elements) < 2:
        raise ValueError("binary cases require a potential with at least two elements")
    e0 = e0 or potential.elements[0]
    e1 = e1 or potential.elements[1]
    potential.element_index(e0)
    potential.element_index(e1)

    rng = np.random.default_rng(seed)
    cutoff = potential.cutoff
    fallback_a = 3.70
    a0 = float(a0 if a0 is not None else _default_fcc_a(e0, fallback_a))
    a1 = float(a1 if a1 is not None else _default_fcc_a(e1, fallback_a))
    b2_a = float(b2_a if b2_a is not None else _default_b2_a(e0, e1, 0.78 * 0.5 * (a0 + a1)))
    l12_a = float(l12_a if l12_a is not None else _default_l12_a(e0, e1, 0.5 * (a0 + a1)))
    random_fcc_a = float(random_fcc_a if random_fcc_a is not None else 0.5 * (a0 + a1))

    cases: list[Atoms] = []

    def add(name: str, atoms: Atoms):
        atoms.info["case_name"] = name
        cases.append(atoms)

    # Pure-element cases inside the binary potential catch element-index and table-selection bugs.
    for el, aa in [(e0, a0), (e1, a1)]:
        unit = bulk(el, "fcc", a=aa, cubic=True)
        atoms = _repeat_to_safe(unit, cutoff)
        rep_label = "x".join(str(int(r)) for r in np.rint(atoms.cell.lengths() / unit.cell.lengths()).astype(int))
        add(f"{el}_pure_fcc_{rep_label}_binary_potential", atoms)

    # Ordered B2 AB and BA sublattice-swapped structures.
    b2 = _repeat_to_safe(_make_b2(e0, e1, b2_a), cutoff)
    add(f"B2_{e0}{e1}_ordered", b2)

    b2_swap = _repeat_to_safe(_make_b2(e1, e0, b2_a), cutoff)
    add(f"B2_{e1}{e0}_sublattice_swapped", b2_swap)

    atoms = b2.copy()
    atoms.positions += rng.normal(scale=0.01, size=atoms.positions.shape)
    atoms.wrap()
    add(f"B2_{e0}{e1}_random_disp_0p01A", atoms)

    atoms = b2.copy()
    cell = atoms.cell.array.copy()
    cell[0, 1] += 0.05 * cell[1, 1]
    atoms.set_cell(cell, scale_atoms=True)
    add(f"B2_{e0}{e1}_shear_xy_5pct", atoms)

    atoms = b2.copy()
    # Antisite: flip the first atom to the other species.
    syms = atoms.get_chemical_symbols()
    syms[0] = e1 if syms[0] == e0 else e0
    atoms.set_chemical_symbols(syms)
    add(f"B2_{e0}{e1}_single_antisite", atoms)

    atoms = b2.copy()
    del atoms[len(atoms) // 2]
    add(f"B2_{e0}{e1}_single_vacancy", atoms)

    # L1_2 A3B and B3A-like geometries.
    l12_a3b = _repeat_to_safe(_make_l12(e0, e1, l12_a), cutoff)
    add(f"L12_{e0}3{e1}_ordered", l12_a3b)

    l12_b3a = _repeat_to_safe(_make_l12(e1, e0, l12_a), cutoff)
    add(f"L12_{e1}3{e0}_ordered", l12_b3a)

    # Random 50/50 fcc alloy on a safe fcc supercell.
    unit = bulk(e0, "fcc", a=random_fcc_a, cubic=True)
    random_atoms = _repeat_to_safe(unit, cutoff)
    n = len(random_atoms)
    syms = np.array([e0] * n, dtype=object)
    idx = np.arange(n)
    rng.shuffle(idx)
    syms[idx[: n // 2]] = e1
    random_atoms.set_chemical_symbols(list(syms))
    add(f"random_{e0}50{e1}50_fcc_seed{seed}", random_atoms)

    atoms = random_atoms.copy()
    atoms.positions += rng.normal(scale=0.02, size=atoms.positions.shape)
    atoms.wrap()
    add(f"random_{e0}50{e1}50_fcc_random_disp_0p02A", atoms)

    atoms = random_atoms.copy()
    order = np.arange(len(atoms))
    rng.shuffle(order)
    atoms = atoms[order]
    add(f"random_{e0}50{e1}50_atom_order_permuted", atoms)

    # PBC minimum-image mixed pair across the boundary.
    safe_l = _safe_periodic_box_length(cutoff)
    mixed_pair_distance = min(b2_a, 0.45 * cutoff)
    atoms = Atoms(
        [e0, e1],
        positions=[[0.25, safe_l / 2.0, safe_l / 2.0], [safe_l - mixed_pair_distance + 0.25, safe_l / 2.0, safe_l / 2.0]],
        cell=np.diag([safe_l, safe_l, safe_l]),
        pbc=True,
    )
    add(f"{e0}{e1}_pbc_boundary_two_atoms_safe_box", atoms)

    # Species-resolved cutoff boundary pairs. These are non-periodic, so they do
    # not need a 2*cutoff-sized box.
    for s0, s1 in [(e0, e0), (e1, e1), (e0, e1), (e1, e0)]:
        label = f"{s0}{s1}"
        add(f"cluster_pair_{label}_just_below_cutoff", Atoms([s0, s1], positions=[[0, 0, 0], [0.999 * cutoff, 0, 0]], pbc=False))
        add(f"cluster_pair_{label}_just_above_cutoff", Atoms([s0, s1], positions=[[0, 0, 0], [1.001 * cutoff, 0, 0]], pbc=False))

    return cases


def make_cases(
    potential: SetflEAMAlloy,
    suite: str,
    element: str,
    a: float,
    binary_elements: Sequence[str] | None,
    a0: float | None,
    a1: float | None,
    b2_a: float | None,
    l12_a: float | None,
    random_fcc_a: float | None,
    seed: int,
    include_slab: bool,
) -> list[Atoms]:
    if suite == "auto":
        suite = "binary" if len(potential.elements) >= 2 else "elemental"

    cases: list[Atoms] = []
    if suite in ("elemental", "all"):
        cases.extend(make_elemental_cases(potential, element=element, a=a, seed=seed, include_slab=include_slab))
    if suite in ("binary", "all"):
        if len(potential.elements) < 2:
            raise ValueError("--suite binary/all requested, but potential contains fewer than two elements")
        if binary_elements:
            if len(binary_elements) != 2:
                raise ValueError("--binary-elements must provide exactly two symbols")
            e0, e1 = binary_elements
        else:
            e0, e1 = potential.elements[:2]
        cases.extend(
            make_binary_cases(
                potential,
                e0=e0,
                e1=e1,
                a0=a0,
                a1=a1,
                b2_a=b2_a,
                l12_a=l12_a,
                random_fcc_a=random_fcc_a,
                seed=seed,
            )
        )
    return cases


def _should_compare_stress(atoms: Atoms) -> bool:
    return bool(np.any(atoms.get_pbc())) and atoms.cell.rank == 3 and atoms.get_volume() > 0.0


def evaluate(atoms: Atoms, calc, compare_stress: bool = True):
    a = atoms.copy()
    a.calc = calc
    e = float(a.get_potential_energy())
    f = np.asarray(a.get_forces(), dtype=float)
    try:
        ep = np.asarray(a.get_potential_energies(), dtype=float)
    except Exception:
        ep = None

    stress = None
    stress_error = None
    if compare_stress and _should_compare_stress(a):
        try:
            stress = np.asarray(a.get_stress(voigt=True), dtype=float)
            if stress.shape != (6,):
                raise ValueError(f"ASE stress must have shape (6,), got {stress.shape}")
        except Exception as exc:
            stress_error = f"{type(exc).__name__}: {exc}"

    return e, f, ep, stress, stress_error


def _stress_metrics(stress_c: np.ndarray, stress_o: np.ndarray) -> dict:
    stress_c = np.asarray(stress_c, dtype=float)
    stress_o = np.asarray(stress_o, dtype=float)
    ds = stress_c - stress_o
    ds_flip = stress_c + stress_o
    ds_lammps_order = _lammps_order_to_ase_order(stress_c) - stress_o
    return {
        "stress_cream_eV_per_A3": [float(x) for x in stress_c],
        "stress_reference_eV_per_A3": [float(x) for x in stress_o],
        "max_abs_stress_error_eV_per_A3": float(np.max(np.abs(ds))),
        "rms_stress_error_eV_per_A3": float(np.sqrt(np.mean(ds * ds))),
        "max_abs_stress_error_GPa": float(np.max(np.abs(ds)) * EV_PER_A3_TO_GPA),
        "rms_stress_error_GPa": float(np.sqrt(np.mean(ds * ds)) * EV_PER_A3_TO_GPA),
        "diagnostic_sign_flipped_max_abs_stress_error_eV_per_A3": float(np.max(np.abs(ds_flip))),
        "diagnostic_lammps_order_max_abs_stress_error_eV_per_A3": float(np.max(np.abs(ds_lammps_order))),
    }


def _summarize(rows: list[dict]) -> dict:
    passed = [r for r in rows if r.get("passed")]
    failed = [r for r in rows if not r.get("passed")]
    numeric = [r for r in rows if "abs_energy_error_eV_per_atom" in r]
    stress_rows = [r for r in numeric if r.get("stress_checked")]
    return {
        "n_cases": len(rows),
        "n_passed": len(passed),
        "n_failed": len(failed),
        "max_abs_energy_error_eV_per_atom": max((r["abs_energy_error_eV_per_atom"] for r in numeric), default=None),
        "max_abs_force_error_eV_per_A": max((r["max_abs_force_error_eV_per_A"] for r in numeric), default=None),
        "max_rms_force_error_eV_per_A": max((r["rms_force_error_eV_per_A"] for r in numeric), default=None),
        "max_abs_stress_error_eV_per_A3": max((r.get("max_abs_stress_error_eV_per_A3", 0.0) for r in stress_rows), default=None),
        "max_abs_stress_error_GPa": max((r.get("max_abs_stress_error_GPa", 0.0) for r in stress_rows), default=None),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("potential", type=Path)
    parser.add_argument("--backend", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--use-cell-list", action="store_true")
    parser.add_argument("--suite", default="auto", choices=["auto", "elemental", "binary", "all"], help="auto: 1-element potentials use elemental, 2+ element potentials use binary")
    parser.add_argument("--element", default=None, help="element for elemental suite; default = first element in potential")
    parser.add_argument("--a", type=float, default=None, help="fcc lattice constant for elemental suite; default from element table or 3.615")
    parser.add_argument("--binary-elements", nargs=2, default=None, metavar=("E0", "E1"), help="two element symbols to use for binary suite; default = first two elements in setfl file")
    parser.add_argument("--a0", type=float, default=None, help="fcc lattice constant for binary element E0")
    parser.add_argument("--a1", type=float, default=None, help="fcc lattice constant for binary element E1")
    parser.add_argument("--b2-a", type=float, default=None, help="B2 lattice constant for binary ordered cases")
    parser.add_argument("--l12-a", type=float, default=None, help="L1_2 lattice constant for binary ordered cases")
    parser.add_argument("--random-fcc-a", type=float, default=None, help="fcc lattice constant for random binary alloy cases")
    parser.add_argument("--seed", type=int, default=20260425)
    parser.add_argument("--atol-energy-per-atom", type=float, default=1.0e-6)
    parser.add_argument("--atol-force-max", type=float, default=3.0e-4)
    parser.add_argument("--atol-force-rms", type=float, default=1.0e-4)
    parser.add_argument("--atol-stress-max", type=float, default=2.0e-3, help="Max stress error in eV/A^3. Default 2e-3 ~= 0.32 GPa.")
    parser.add_argument("--atol-stress-rms", type=float, default=1.0e-3, help="RMS stress error in eV/A^3. Default 1e-3 ~= 0.16 GPa.")
    parser.add_argument("--no-stress", action="store_true", help="skip stress comparison")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--no-slab", action="store_true", help="skip the elemental surface slab case")
    args = parser.parse_args()

    from cream import CreamCalculator

    potential = SetflEAMAlloy.from_file(args.potential)
    element = args.element or potential.elements[0]
    a = float(args.a if args.a is not None else _default_fcc_a(element, 3.615))
    suite_for_print = args.suite if args.suite != "auto" else ("binary" if len(potential.elements) >= 2 else "elemental")

    print(f"Potential elements : {', '.join(potential.elements)}")
    print(f"Potential cutoff   : {potential.cutoff:.12g} Angstrom")
    print(f"2 * cutoff         : {2.0 * potential.cutoff:.12g} Angstrom")
    print(f"Selected suite     : {suite_for_print}")
    if suite_for_print in ("elemental", "all"):
        rep = _min_bulk_repeat_for_cutoff(a, potential.cutoff)
        print(f"Elemental fcc      : {element}, a={a:.6g} A, repeat={rep}x{rep}x{rep}")
    if suite_for_print in ("binary", "all") and len(potential.elements) >= 2:
        be = args.binary_elements or list(potential.elements[:2])
        print(f"Binary elements    : {be[0]}, {be[1]}")
    print("External reference tolerances:")
    print(f"  energy/atom      : {args.atol_energy_per_atom:.3e} eV/atom")
    print(f"  force max        : {args.atol_force_max:.3e} eV/A")
    print(f"  force RMS        : {args.atol_force_rms:.3e} eV/A")
    if args.no_stress:
        print("  stress           : skipped")
    else:
        print(f"  stress max       : {args.atol_stress_max:.3e} eV/A^3 ({args.atol_stress_max * EV_PER_A3_TO_GPA:.3e} GPa)")
        print(f"  stress RMS       : {args.atol_stress_rms:.3e} eV/A^3 ({args.atol_stress_rms * EV_PER_A3_TO_GPA:.3e} GPa)")

    cases = make_cases(
        potential=potential,
        suite=args.suite,
        element=element,
        a=a,
        binary_elements=args.binary_elements,
        a0=args.a0,
        a1=args.a1,
        b2_a=args.b2_a,
        l12_a=args.l12_a,
        random_fcc_a=args.random_fcc_a,
        seed=args.seed,
        include_slab=not args.no_slab,
    )
    print(f"Generated cases    : {len(cases)}")

    rows: list[dict] = []
    ok_all = True
    for atoms in cases:
        name = atoms.info["case_name"]
        cream_calc = CreamCalculator(str(args.potential), backend=args.backend, use_cell_list=args.use_cell_list)
        reference_calc = EAMAlloyReferenceCalculator(str(args.potential))
        try:
            e_c, f_c, ep_c, s_c, s_err_c = evaluate(atoms, cream_calc, compare_stress=not args.no_stress)
            e_o, f_o, ep_o, s_o, s_err_o = evaluate(atoms, reference_calc, compare_stress=not args.no_stress)
        except Exception as exc:
            print(f"ERROR {name:52s} N={len(atoms):5d}: {type(exc).__name__}: {exc}")
            ok_all = False
            rows.append({"case": name, "n_atoms": len(atoms), "error": repr(exc), "passed": False})
            continue

        de = abs(e_c - e_o)
        de_pa = de / max(1, len(atoms))
        df = f_c - f_o
        df_max = float(np.max(np.abs(df)))
        df_rms = float(np.sqrt(np.mean(df * df)))

        stress_checked = False
        stress_ok = True
        stress_note = ""
        stress_data: dict = {}
        if not args.no_stress and _should_compare_stress(atoms):
            if s_err_c or s_err_o or s_c is None or s_o is None:
                stress_ok = False
                stress_note = f"stress unavailable: cream={s_err_c!r}, reference={s_err_o!r}"
            else:
                stress_checked = True
                stress_data = _stress_metrics(s_c, s_o)
                stress_ok = (
                    stress_data["max_abs_stress_error_eV_per_A3"] <= args.atol_stress_max
                    and stress_data["rms_stress_error_eV_per_A3"] <= args.atol_stress_rms
                )

        ok = (
            de_pa <= args.atol_energy_per_atom
            and df_max <= args.atol_force_max
            and df_rms <= args.atol_force_rms
            and stress_ok
        )
        ok_all = ok_all and ok
        row = {
            "case": name,
            "n_atoms": len(atoms),
            "symbols": sorted(set(atoms.get_chemical_symbols())),
            "pbc": [bool(x) for x in atoms.get_pbc()],
            "cell_lengths": [float(x) for x in atoms.cell.lengths()],
            "energy_cream_eV": e_c,
            "energy_reference_eV": e_o,
            "abs_energy_error_eV": de,
            "abs_energy_error_eV_per_atom": de_pa,
            "max_abs_force_error_eV_per_A": df_max,
            "rms_force_error_eV_per_A": df_rms,
            "force_sum_norm_cream": float(np.linalg.norm(f_c.sum(axis=0))),
            "force_sum_norm_reference": float(np.linalg.norm(f_o.sum(axis=0))),
            "stress_checked": stress_checked,
            "stress_note": stress_note,
            "passed": ok,
        }
        row.update(stress_data)
        rows.append(row)

        msg = (
            f"{'PASS' if ok else 'FAIL':4s} {name:52s} N={len(atoms):5d} "
            f"|dE|/N={de_pa:.3e} eV/atom  max|dF|={df_max:.3e} eV/A  rms|dF|={df_rms:.3e}"
        )
        if stress_checked:
            msg += (
                f"  max|dS|={stress_data['max_abs_stress_error_eV_per_A3']:.3e} eV/A^3"
                f" ({stress_data['max_abs_stress_error_GPa']:.3e} GPa)"
            )
        elif stress_note:
            msg += f"  STRESS-FAIL[{stress_note}]"
        print(msg)

    summary = _summarize(rows)
    print("Summary:")
    print(json.dumps(summary, indent=2))

    if args.json_out:
        payload = {
            "potential": str(args.potential),
            "potential_elements": list(potential.elements),
            "cutoff_A": potential.cutoff,
            "backend": args.backend,
            "use_cell_list": args.use_cell_list,
            "suite": suite_for_print,
            "tolerances": {
                "energy_per_atom_eV": args.atol_energy_per_atom,
                "force_max_eV_per_A": args.atol_force_max,
                "force_rms_eV_per_A": args.atol_force_rms,
                "stress_max_eV_per_A3": args.atol_stress_max,
                "stress_rms_eV_per_A3": args.atol_stress_rms,
            },
            "summary": summary,
            "rows": rows,
        }
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {args.json_out}")
    if not ok_all:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
