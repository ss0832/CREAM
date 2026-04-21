#!/usr/bin/env python3
"""
cream_quickcheck.py — CREAM sanity-check script
================================================

A lightweight script to verify that CREAM is correctly installed and can
load a potential file before running the full validation suite
(cu_md_simulation.py).  All checks complete in a few seconds.

Checks
------
  1. Potential file loading and metadata display
  2. Single-point energy / forces / stress calculation
  3. Force conservation check (ΣF ≈ 0)
  4. Cohesive energy comparison against experimental reference
  5. Determinism check (CPU backend only — identical input → identical output)

Usage
-----
  python cream_quickcheck.py --potential Cu01.eam.alloy
  python cream_quickcheck.py --potential Cu01.eam.alloy --backend cpu
  python cream_quickcheck.py --potential Mishin-Ni-Al-2009_eam.alloy --element Ni

Note
----
  Potential files (*.eam.alloy) must be obtained separately.
  For Cu, see the NIST Interatomic Potentials Repository:
  https://www.ctcms.nist.gov/potentials/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


# ── Reference values (NIST / experiment) ──────────────────────────────────────

# Per-element reference values: lattice parameter [Å], cohesive energy [eV/atom]
_REFERENCE: dict[str, dict[str, float]] = {
    "Cu": {"structure": "fcc", "a0": 3.615, "E_coh": -3.49},
    "Ni": {"structure": "fcc", "a0": 3.524, "E_coh": -4.44},
    "Al": {"structure": "fcc", "a0": 4.046, "E_coh": -3.36},
}


# ── Utilities ──────────────────────────────────────────────────────────────────

def _ok(msg: str) -> None:
    print(f"  [OK]   {msg}")

def _warn(msg: str) -> None:
    print(f"  [WARN] {msg}")

def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")

def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def _check(label: str, value: float, ref: float, tol_pct: float = 5.0) -> bool:
    diff_pct = 100.0 * (value - ref) / abs(ref)
    passed   = abs(diff_pct) <= tol_pct
    line     = f"{label}: {value:.4f}  (ref {ref:.4f},  Δ={diff_pct:+.2f}%)"
    (_ok if passed else _warn)(line)
    return passed


# ── Check body ─────────────────────────────────────────────────────────────────

def run_quickcheck(potential: str, backend: str, element: str) -> int:
    """Return 0 on success, 1 if any critical check fails."""
    from cream import CreamCalculator
    from ase.build import bulk

    pot_path = Path(potential)
    if not pot_path.exists():
        print(f"[ERROR] Potential file not found: {pot_path}")
        return 1

    ref = _REFERENCE.get(element)
    if ref is None:
        print(f"[WARN] No reference values registered for '{element}'. "
              "Energy comparison will be skipped.")

    failures = 0

    # ── 1. Potential loading ───────────────────────────────────────────────────
    _section("1. Potential loading")
    try:
        calc = CreamCalculator(str(pot_path), backend=backend, use_cell_list=False)
        eng  = calc._engine
        print(f"  File    : {pot_path.name}")
        print(f"  Elements: {eng.elements}")
        print(f"  Cutoff  : {eng.cutoff:.3f} Å")
        print(f"  Backend : {eng.backend}")
        _ok("Potential file loaded successfully")
    except Exception as exc:
        _fail(f"Failed to load potential: {exc}")
        return 1

    if element not in eng.elements:
        print(f"  [WARN] '{element}' not found in potential element list: {eng.elements}")

    # ── 2. Single-point calculation (energy / forces / stress) ────────────────
    _section("2. Single-point calculation")
    a0   = ref["a0"] if ref else 3.615
    size = 4   # 4³ × 4 = 256 atoms — lightweight but sufficient
    atoms = bulk(element, ref["structure"] if ref else "fcc",
                 a=a0, cubic=True) * (size, size, size)
    n     = len(atoms)
    print(f"  Supercell: {size}³ FCC = {n} atoms")

    try:
        atoms.calc = CreamCalculator(str(pot_path), backend=backend, use_cell_list=False)
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        stress = atoms.get_stress()                     # Voigt (6,) eV/Å³
        _ok(f"Energy  E = {energy:.4f} eV  ({energy/n:.4f} eV/atom)")
        _ok(f"Forces  shape = {forces.shape}")
        stress_gpa = stress * 160.218
        stress_str = "  ".join(f"{v:+.2f}" for v in stress_gpa)
        _ok(f"Stress  (Voigt, GPa) = [{stress_str}]")
    except Exception as exc:
        _fail(f"Single-point calculation failed: {exc}")
        failures += 1

    # ── 3. Force conservation (ΣF ≈ 0) ────────────────────────────────────────
    _section("3. Force conservation check  (ΣF ≈ 0)")
    f_sum = forces.sum(axis=0)
    f_max = float(np.abs(f_sum).max())
    print(f"  ΣF = [{f_sum[0]:.2e}, {f_sum[1]:.2e}, {f_sum[2]:.2e}] eV/Å")
    if f_max < 1e-3:
        _ok(f"|ΣF|_max = {f_max:.2e} eV/Å  (threshold 1e-3)")
    else:
        _warn(f"|ΣF|_max = {f_max:.2e} eV/Å — expected ~0 for a perfect crystal")
        failures += 1

    # ── 4. Cohesive energy comparison ─────────────────────────────────────────
    _section("4. Cohesive energy comparison")
    if ref:
        passed = _check(
            f"Cohesive energy [eV/atom]  ({element})",
            energy / n,
            ref["E_coh"],
            tol_pct=5.0,
        )
        if not passed:
            failures += 1
    else:
        print("  (No reference values — skipped)")

    # ── 5. Determinism check (CPU backend only) ────────────────────────────────
    _section("5. Determinism check")
    if backend == "cpu":
        atoms2 = bulk(element, ref["structure"] if ref else "fcc",
                      a=a0, cubic=True) * (size, size, size)
        atoms2.calc = CreamCalculator(str(pot_path), backend="cpu", use_cell_list=False)
        energy2 = atoms2.get_potential_energy()
        if energy == energy2:
            _ok(f"Identical input → identical output  (E = {energy2:.10f} eV)  bit-exact")
        else:
            diff = abs(energy - energy2)
            _warn(f"Energy difference = {diff:.2e} eV  (CPU backend should be deterministic)")
            failures += 1
    else:
        print("  Skipped: GPU floating-point accumulation order is non-deterministic.")
        print("  Run with --backend cpu to enable this check.")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    if failures == 0:
        print("  Result: all checks PASSED — ready to run the full suite.")
        print(f"  Next:   python cu_md_simulation.py --potential {pot_path}")
    else:
        print(f"  Result: {failures} check(s) returned WARN/FAIL.")
        print("  Review the messages above before proceeding.")
    print("=" * 60)

    return 0 if failures == 0 else 1


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CREAM sanity-check script — run this before the full suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--potential", "-p",
        required=True,
        help="Path to a *.eam.alloy potential file",
    )
    parser.add_argument(
        "--backend", "-b",
        default="gpu",
        choices=["gpu", "cpu"],
        help="Compute backend (default: gpu)",
    )
    parser.add_argument(
        "--element", "-e",
        default="Cu",
        help="Element symbol to check (default: Cu)",
    )
    args = parser.parse_args()

    for pkg, pip_name in [("cream", "cream-python"), ("ase", "ase"), ("numpy", "numpy")]:
        try:
            __import__(pkg)
        except ImportError:
            print(f"[ERROR] '{pkg}' not found.  Install with: pip install {pip_name}")
            sys.exit(1)

    sys.exit(run_quickcheck(args.potential, args.backend, args.element))


if __name__ == "__main__":
    main()
