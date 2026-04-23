#!/usr/bin/env python3
"""
cu_md_simulation.py — Copper physical-property validation via EAM-MD (CREAM)
===============================================================================

Runs a sequence of calculations that together verify whether the CREAM engine
reproduces known bulk copper properties with the NIST Cu EAM potential
(Cu01.eam.alloy or compatible):

  Static tests
  ------------
  1. Cohesive energy per atom at the ideal FCC lattice
  2. Equation-of-state scan (EV curve)
  3. Single-point force symmetry check

  Dynamic tests
  -------------
  4. NVT ensemble — Langevin thermostat, 300 K
  5. NPT ensemble — Nosé-Hoover (ASE NPT), 300 K, 0 Pa
  6. Temperature-series NPT (thermal expansion)
  7. NVT near melting point (1300 K)
  8. NPT Non-orthogonal (Triclinic) Cell relaxation
  9.  Elastic constants — C11, C12, C44 via Voigt finite-strain perturbation
                          + derived Zener anisotropy A and shear modulus G
  10. Vacancy formation energy — static, with per-atom energy perturbation
                          field around the defect (CPU backend)
  11. EAM per-atom fields — electron density ρᵢ and embedding energy F(ρᵢ)
                          distributions at 0 K and 300 K (CPU-only quantities)
  12. Self-diffusion coefficient — D(T) from NVE MSD Einstein relation at
                          multiple temperatures + Arrhenius fit → E_a
  13. Vibrational DOS — via velocity autocorrelation (VACF) FFT →
                          Debye frequency ν_D, Debye temperature Θ_D
  14. Intrinsic Stacking Fault Energy (ISFE) γ_sf — rigid [111] Shockley
                          partial shift (b_p = a₀/√6) + z-only FIRE relax
                          → γ_sf [mJ/m²], exp ~45 mJ/m² for Cu
  15. Isobaric heat capacity C_P — NVT potential-energy fluctuation formula
                          C_V = (3/2)Nk_B + Var(E_pot)/(k_B·T²)
                          C_P ≈ C_V + T·V_m·α²·B₀
                          compare Dulong-Petit 3R = 24.94 J/(mol·K)
  16. Helmholtz free energy A & entropy S — quasiharmonic Debye model
                          A_vib = 3Nk_BT·[ln(hν_D/k_BT) − 1]  (classical)
                          S_vib = 3Nk_B·[2 − ln(hν_D/k_BT)]
                          ν_D from TEST 13 (or exp fallback 7.2 THz)
                          compare S vs NIST-JANAF S°(298 K) = 33.15 J/(mol·K)
  17. Quasi-Harmonic Approximation (QHA) — quantum Debye model
                          E(V): Birch-Murnaghan EOS fit over ±4% volume scan
                          γ_G : Grüneisen constant from VDOS at 3 scaled volumes
                          θ_D(V) = θ_D(V₀)·(V₀/V)^γ_G (volume-dependent Debye temp)
                          F(V,T) = E_BM(V) + F_vib_quantum(θ_D(V), T)
                          Minimized over V → V_eq(T), S(T), α(T)
                          S includes ZPE → expected accuracy ~10–20% vs NIST-JANAF
                          compare S vs 33.15 J/(mol·K), α vs 3×16.6×10⁻⁶ K⁻¹

Usage
-----
  python cu_md_simulation.py --potential Cu01.eam.alloy --backend gpu
  python cu_md_simulation.py --potential Cu01.eam.alloy --backend cpu --size 5
  python cu_md_simulation.py --potential Cu01.eam.alloy --skip 6,7,8
  python cu_md_simulation.py --potential Cu01.eam.alloy --skip 9,10,11,12,13
  python cu_md_simulation.py --potential Cu01.eam.alloy --skip 12,13,14,15,16,17
"""

from __future__ import annotations

import argparse
import collections
import importlib
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from ase import units
from ase.build import bulk, fcc111
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet

# NumPy 2.0 renamed np.trapz → np.trapezoid; support both versions
_np_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


# ── Runtime dependency check ───────────────────────────────────────────────────

def _require(pkg: str, pip_name: Optional[str] = None) -> None:
    try:
        __import__(pkg)
    except ImportError:
        print(f"[ERROR] Required package '{pkg}' not found.  Install with:")
        print(f"        pip install {pip_name or pkg}")
        sys.exit(1)


# ── Constants ──────────────────────────────────────────────────────────────────

EV_PER_A3_TO_GPA = 160.21766    # 1 eV/Å³ in GPa
KB_EV            = 8.617333e-5  # Boltzmann constant [eV/K]
HBAR_EV_S        = 6.582119e-16 # ħ [eV·s]
# Additional constants for TEST 14 (stacking fault) and TEST 15 (heat capacity)
EV_PER_A2_TO_J_M2 = 16.021766  # 1 eV/Å² in J/m²  (= 1.602e-19 J / 1e-20 m²)
EV_PER_A2_TO_MJ_M2 = 16021.766 # 1 eV/Å² in mJ/m²
R_J_MOL_K        = 8.314462    # Universal gas constant [J/(mol·K)]
N_A              = 6.02214076e23 # Avogadro constant [/mol]
EV_TO_J          = 1.602176634e-19 # [J/eV]

# Experimental reference values for bulk Cu
# ──────────────────────────────────────────────────────────────────────────────
# All values are for pure Cu at or near ambient conditions unless noted.
# Each entry carries a short inline citation; full references are in the block
# comment below.  Keys must not be renamed — they are used by compare() calls.
# ──────────────────────────────────────────────────────────────────────────────
CU_EXP: dict[str, float] = {
    # ── Structural ────────────────────────────────────────────────────────────
    "a0_A":         3.615,    # Lattice parameter [Å]  RT X-ray
                              # [1] Kittel 8th ed. (2005) p.20
                              # [2] NIST SRM 1990a Cu powder standard (room temp)

    "E_coh_eV":    -3.49,    # Cohesive energy [eV/atom]
                              # [1] Kittel 8th ed. (2005) p.50 Table 1
                              # [3] CRC Handbook 97th ed. (2016–17), "Enthalpy of Sublimation"

    # ── Elastic ───────────────────────────────────────────────────────────────
    "B0_GPa":       140.0,   # Isothermal bulk modulus [GPa] at 298 K
                              # [4] Ledbetter & Naimon, J. Phys. Chem. Ref. Data 3, 897 (1974)
                              # [5] Simmons & Wang, "Single Crystal Elastic Constants" (1971)

    "C11_GPa":      168.4,   # Elastic constant C11 [GPa]
    "C12_GPa":      121.4,   # Elastic constant C12 [GPa]
    "C44_GPa":       75.4,   # Elastic constant C44 [GPa]
                              # [6] Neighbours & Alers, Phys. Rev. 111, 707–712 (1958)
                              #     ultrasonic pulse-echo at 298 K

    "A_zener":        3.21,  # Zener anisotropy A = 2C44/(C11−C12)  [dimensionless]
                              # Derived from [6]: 2×75.4/(168.4−121.4) = 3.208

    # ── Thermal ───────────────────────────────────────────────────────────────
    "alpha_1perK":  16.6e-6, # Linear thermal expansion coefficient α [K⁻¹] at 293 K
                              # [7] Touloukian et al., "Thermal Expansion: Metallic Elements
                              #     and Alloys", TPRC Vol. 12 (1975) p.4–39

    "T_melt_K":    1358.0,   # Melting point [K]
                              # [8] NIST WebBook, "Thermophysical Properties of Cu"
                              # [1] Kittel 8th ed. (2005) p.58

    # ── Defects ───────────────────────────────────────────────────────────────
    "E_vac_eV":       1.28,  # Monovacancy formation energy [eV]
                              # [9] Ehrhart, Landolt-Börnstein Vol.25 (1991) p.88
                              #     positron annihilation + resistivity quench consensus

    # ── Phonon / vibrational ──────────────────────────────────────────────────
    "nu_Debye_THz":   7.2,   # Debye cutoff frequency ν_D [THz]
                              # [1] Kittel 8th ed. (2005) Table 5.3
                              # [10] de Launay, Solid State Physics Vol.2 (1956) p.219

    "theta_D_K":    343.0,   # Debye temperature Θ_D [K]
                              # [1] Kittel 8th ed. (2005) Table 5.3 (value: 343 K)
                              # [11] Anderson, J. Phys. Chem. Solids 12, 41 (1959): 315–347 K

    # ── Diffusion ─────────────────────────────────────────────────────────────
    "Ea_diff_eV":     2.04,  # Solid-state self-diffusion activation energy [eV]
                              # [12] Maier, Mehrer & Rein, Phys. Rev. B 27, 2393 (1983)
                              #      tracer diffusion, valid 900–1350 K (solid)
                              # Note: liquid-Cu regime (TEST 12 temps) gives E_a ~ 0.45 eV

    # ── Stacking fault (TEST 14) ───────────────────────────────────────────────
    "gamma_sf_mJ_m2":  45.0, # Intrinsic stacking fault energy (ISFE) [mJ/m²]
                              # [13] Hirth & Lothe, "Theory of Dislocations" 2nd ed. (1982)
                              #      p.839  (widely cited consensus value)
                              # [14] Carter & Holmes, Philos. Mag. 35, 1161 (1977)
                              #      TEM weak-beam measurement: 41 ± 9 mJ/m²
                              # [15] Brandl, Bitzek, et al., Phys. Rev. B 76, 054124 (2007)
                              #      DFT-GGA: 40 mJ/m²
                              # EAM Mishin Cu: ~44 mJ/m²  [16]

    # ── Heat capacity (TEST 15) ───────────────────────────────────────────────
    "C_P_300K_J_mol_K": 24.44, # Isobaric heat capacity C_P at 300 K [J/(mol·K)]
                               # [17] Chase, NIST-JANAF Thermochem. Tables 4th ed. (1998)
                               #      p.698  (value at 298.15 K: 24.440 J/(mol·K))
                               # [8]  NIST WebBook "Cu heat capacity"
                               # Classical Dulong-Petit limit: 3R = 24.943 J/(mol·K)

    # ── Free energy / entropy (TEST 16) ──────────────────────────────────────
    "S_298K_J_mol_K":   33.15, # Standard molar entropy S° at 298.15 K [J/(mol·K)]
                               # [17] Chase, NIST-JANAF 4th ed. (1998) p.698
                               # [18] Barin, "Thermochemical Data of Pure Substances"
                               #      3rd ed. (1995) p.342: 33.164 J/(mol·K)
                               # Note: classical MD overestimates S by ~35–45% at T≈Θ_D
                               #       because quantum (ZPE) effects are absent
}

# ── Full reference list for CU_EXP ─────────────────────────────────────────────
# [1]  Kittel, C., "Introduction to Solid State Physics", 8th ed. (Wiley, 2005)
# [2]  NIST Standard Reference Material 1990a
# [3]  Lide, D.R. (ed.), CRC Handbook of Chemistry and Physics, 97th ed. (2016–17)
# [4]  Ledbetter, H.M. & Naimon, E.R., J. Phys. Chem. Ref. Data 3, 897 (1974)
# [5]  Simmons, G. & Wang, H., "Single Crystal Elastic Constants and Calculated
#      Aggregate Properties", 2nd ed. (MIT Press, 1971)
# [6]  Neighbours, J.R. & Alers, G.A., Phys. Rev. 111, 707–712 (1958)
# [7]  Touloukian, Y.S. et al., "Thermal Expansion: Metallic Elements and Alloys",
#      Thermophysical Properties of Matter Vol. 12 (IFI/Plenum, 1975)
# [8]  NIST WebBook, https://webbook.nist.gov/cgi/cbook.cgi?ID=C7440508
# [9]  Ehrhart, P., in Landolt-Börnstein NS III/25 "Atomic Defects in Metals"
#      (Springer, 1991) p.88 — Cu monovacancy formation energy
# [10] de Launay, J., Solid State Physics Vol.2 (Academic Press, 1956) p.219
# [11] Anderson, O.L., J. Phys. Chem. Solids 12, 41 (1959)
# [12] Maier, K., Mehrer, H. & Rein, G., Phys. Rev. B 27, 2393–2400 (1983)
# [13] Hirth, J.P. & Lothe, J., "Theory of Dislocations", 2nd ed. (Wiley, 1982) p.839
# [14] Carter, C.B. & Holmes, S.M., Philos. Mag. 35, 1161–1172 (1977)
# [15] Brandl, C., Bitzek, E., et al., Phys. Rev. B 76, 054124 (2007)
# [16] Mishin, Y., Mehl, M.J., et al., Phys. Rev. B 63, 224106 (2001)  ← Cu EAM pot.
# [17] Chase, M.W., NIST-JANAF Thermochemical Tables, 4th ed. (J. Phys. Chem. Ref.
#      Data Monograph No. 9, AIP, 1998)
# [18] Barin, I., "Thermochemical Data of Pure Substances", 3rd ed. (VCH, 1995)

# FCC unit cell (cubic=True) contains 4 atoms
_FCC_BASIS = 4

# Liquid Cu reference values for diffusion (EAM MD runs above T_melt are liquid)
# Solid-state self-diffusion via vacancy mechanism is inaccessible on MD timescales.
CU_EXP_LIQUID: dict[str, float] = {
    "D_1400K_m2s": 4.2e-9,   # Liquid Cu D at 1400 K [m²/s]  (Frohberg 1987)
    "Ea_liquid_eV": 0.45,     # Liquid-state activation energy [eV]
}


# ── Supercell helpers ──────────────────────────────────────────────────────────

def _n_reps(n_atoms: int, min_rep: int = 2, max_rep: int = 99) -> int:
    """Supercell replications per axis consistent with a target atom count.

    The FCC unit cell (cubic=True) has 4 atoms, so the supercell atom count is
    ``reps³ × 4``.  This function inverts that relation:

        reps = round( (n_atoms / 4)^(1/3) )

    Both ``min_rep`` and ``max_rep`` are clamped to avoid pathological sizes.

    Parameters
    ----------
    n_atoms : int   — target total atom count (e.g. from ``args.size ** 3 * 4``)
    min_rep : int   — minimum replications per axis (default 2)
    max_rep : int   — maximum replications per axis (default 99)
    """
    reps = round((n_atoms / _FCC_BASIS) ** (1.0 / 3.0))
    return max(min_rep, min(reps, max_rep))


def _make_supercell(n_atoms: int, min_rep: int = 2, max_rep: int = 99):
    """Return a CuFCC supercell with atom count closest to *n_atoms*."""
    r = _n_reps(n_atoms, min_rep, max_rep)
    return bulk("Cu", "fcc", a=CU_EXP["a0_A"], cubic=True) * (r, r, r)


# ── Printing helpers ───────────────────────────────────────────────────────────

def print_section(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}")


def print_subsection(title: str) -> None:
    print(f"\n  ── {title} ──")


def compare(
    label: str,
    value: float,
    experiment: float,
    unit: str = "",
    tol_pct: float = 5.0,
) -> str:
    diff_pct = 100.0 * (value - experiment) / abs(experiment)
    status = "PASS" if abs(diff_pct) <= tol_pct else "WARN"
    return (
        f"  {label:<40s}  calc={value:10.4f}  "
        f"exp={experiment:10.4f}  {unit}  "
        f"Δ={diff_pct:+.2f}%  [{status}]"
    )


def save_array(path: Path, header: str, data: np.ndarray) -> None:
    np.savetxt(path, data, header=header, fmt="%.6f", encoding="utf-8")
    print(f"  Saved: {path}")


# ── Radial distribution function (vectorized) ──────────────────────────────────

def compute_rdf(
    positions: np.ndarray,
    cell: np.ndarray,
    r_max: float = 8.0,
    n_bins: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute g(r) using a memory-efficient chunked approach.

    Previous vectorized approach (O(N^2) memory) was replaced to handle large
    systems (N > 10,000) without memory overflow.
    This version computes distances one atom at a time.

    Parameters
    ----------
    positions : (N, 3) float64  — Cartesian coordinates [Å]
    cell      : (3, 3) float64  — lattice row vectors [Å]
    r_max     : float           — maximum radius [Å]
    n_bins    : int             — number of histogram bins
    """
    n_atoms = len(positions)
    dr      = r_max / n_bins
    H_inv   = np.linalg.inv(cell)

    total_counts = np.zeros(n_bins, dtype=np.float64)

    for i in range(n_atoms):
        diff  = positions[i] - positions
        frac  = diff @ H_inv
        frac -= np.round(frac)
        cart  = frac @ cell
        r     = np.linalg.norm(cart, axis=-1)
        mask  = (r > 0.0) & (r < r_max)
        counts, _ = np.histogram(r[mask], bins=n_bins, range=(0.0, r_max))
        total_counts += counts

    volume    = abs(np.linalg.det(cell))
    rho       = n_atoms / volume
    r_lo      = np.arange(n_bins) * dr
    r_hi      = r_lo + dr
    shell_vol = (4.0 / 3.0) * np.pi * (r_hi**3 - r_lo**3)
    ideal     = rho * (n_atoms - 1) / n_atoms * shell_vol

    g         = total_counts / (n_atoms * ideal)
    r_centers = r_lo + dr / 2.0
    return r_centers, g


# ── Mean squared displacement ──────────────────────────────────────────────────

def msd_from_trajectory(traj: list[np.ndarray]) -> np.ndarray:
    """Compute MSD relative to the first frame."""
    ref = traj[0]
    return np.array([np.mean(np.sum((pos - ref) ** 2, axis=1)) for pos in traj])


# ── PBC-unwrapped MSD (incremental minimum-image accumulation) ────────────────

def msd_unwrapped_from_trajectory(
    traj: list[np.ndarray],
    cell: np.ndarray,
) -> np.ndarray:
    """MSD with PBC unwrapping via the minimum-image convention.

    At high temperatures atoms can cross periodic boundaries between frames,
    making the naive ``msd_from_trajectory`` (raw position differences) give
    a wrong, underestimated MSD.  This function accumulates the
    minimum-image-corrected displacement frame-by-frame so that long-range
    diffusion is captured correctly.

    Parameters
    ----------
    traj : list of (N, 3) arrays   — consecutive wrapped position snapshots
    cell : (3, 3) array            — lattice row vectors (assumed constant)
    """
    H_inv    = np.linalg.inv(cell)
    cumdisp  = np.zeros_like(traj[0])          # (N, 3) running unwrapped displacement
    prev_pos = traj[0].copy()
    msd      = np.zeros(len(traj))

    for i, pos in enumerate(traj):
        raw   = pos - prev_pos
        frac  = raw @ H_inv
        frac -= np.round(frac)                 # minimum-image fractional jump
        disp  = frac @ cell
        cumdisp  += disp
        prev_pos  = pos.copy()
        msd[i]    = float(np.mean(np.sum(cumdisp ** 2, axis=1)))

    return msd


# ── Birch-Murnaghan EOS ────────────────────────────────────────────────────────

def birch_murnaghan(
    V: np.ndarray,
    E0: float,
    V0: float,
    B0: float,
    B0p: float,
) -> np.ndarray:
    eta = (V0 / V) ** (2.0 / 3.0)
    return (
        E0
        + 9.0 * V0 * B0 / 16.0
        * (eta - 1.0) ** 2
        * (6.0 + B0p * (eta - 1.0) - 4.0 * eta)
    )


def fit_eos(volumes: np.ndarray, energies: np.ndarray, n_atoms: int) -> dict:
    from scipy.optimize import curve_fit

    i_min   = np.argmin(energies)
    popt, _ = curve_fit(
        birch_murnaghan,
        volumes,
        energies,
        p0=[energies[i_min], volumes[i_min], 1.0 / EV_PER_A3_TO_GPA, 4.0],
        maxfev=10_000,
    )
    E0, V0, B0_ev, B0p = popt
    B0_gpa = B0_ev * EV_PER_A3_TO_GPA
    a0     = (V0 / (n_atoms / _FCC_BASIS)) ** (1.0 / 3.0)
    return {
        "E0_eV_per_atom": E0 / n_atoms,
        "V0_A3":          V0,
        "a0_A":           a0,
        "B0_GPa":         B0_gpa,
        "B0p":            B0p,
    }


# ── NPT dynamics helper ────────────────────────────────────────────────────────

def _make_npt_dyn(
    atoms,
    timestep_fs: float,
    temperature_K: float,
    pressure_GPa: float,
    ttime_fs: float = 25.0,       # thermostat time constant [fs]
    tau_p_fs: float = 2000.0,     # barostat time constant [fs]
    B0_ref_GPa: float = 140.0,    # reference bulk modulus for pfactor [GPa]
):
    """Return an NPT dynamics object.

    Tries ASE NPT implementations in order of preference:
      1. ase.md.melchionna.MelchionnaNPT  (newer ASE)
      2. ase.md.npt.NPT                   (classic ASE)
      3. ase.md.nptberendsen.NPTBerendsen (fallback — not strict NpT)

    Parameters
    ----------
    ttime_fs    : thermostat time constant [fs] (Nosé-Hoover; default 25 fs)
    tau_p_fs    : barostat time constant [fs] (default 2000 fs = 2 ps).
                  Volume equilibration takes ≥ 5–10 × tau_p; the NPT equil_steps
                  in the callers are set accordingly.
    B0_ref_GPa  : reference bulk modulus [GPa] used in the pfactor expression
                  ``pfactor = tau_p² × B0``.  Default 140 GPa (Cu EAM).
    """
    dt            = timestep_fs * units.fs
    ttime         = ttime_fs * units.fs
    pfactor       = (tau_p_fs * units.fs) ** 2 * (B0_ref_GPa * units.GPa)
    ext_stress_ev = pressure_GPa / EV_PER_A3_TO_GPA

    for cls_path in ("ase.md.melchionna.MelchionnaNPT", "ase.md.npt.NPT"):
        module_name, cls_name = cls_path.rsplit(".", 1)
        try:
            mod = importlib.import_module(module_name)
            cls = getattr(mod, cls_name)
            return cls(
                atoms,
                timestep=dt,
                temperature_K=temperature_K,
                externalstress=ext_stress_ev,
                ttime=ttime,
                pfactor=pfactor,
            )
        except (ImportError, AttributeError, NotImplementedError):
            continue

    from ase.md.nptberendsen import NPTBerendsen
    print("  [INFO] Using NPTBerendsen (Berendsen coupling — not strict NpT)")
    return NPTBerendsen(
        atoms,
        timestep=dt,
        temperature_K=temperature_K,
        pressure_au=pressure_GPa * units.GPa,
        taut=ttime_fs * 20.0 * units.fs,            # Berendsen: taut ~ 20 × ttime
        taup=tau_p_fs * units.fs,
        compressibility_au=1.0 / (B0_ref_GPa * units.GPa),
    )


# ── TEST 1 & 2: Static energy + EV curve ──────────────────────────────────────

def run_static_tests(calc_factory, outdir: Path, size: int) -> dict:
    print_section("TEST 1 & 2  —  Static energy + Equation of State")

    print_subsection("Single-point at experimental lattice parameter")
    atoms   = bulk("Cu", "fcc", a=CU_EXP["a0_A"], cubic=True) * (size, size, size)
    n_atoms = len(atoms)
    print(f"  Supercell: {size}×{size}×{size} FCC  →  {n_atoms} atoms")

    atoms.calc = calc_factory(use_cell_list=(n_atoms > 500))
    e_sp = atoms.get_potential_energy()
    print(f"  E (single-point) = {e_sp:.6f} eV  ({e_sp/n_atoms:.6f} eV/atom)")
    print(compare("Cohesive energy [eV/atom]", e_sp / n_atoms, CU_EXP["E_coh_eV"],
                  "eV/atom", tol_pct=5.0))

    print_subsection("E-V curve (±8% volume range, 20 points)")
    a_values = CU_EXP["a0_A"] * np.linspace(0.94, 1.06, 20)
    volumes, energies = [], []
    for a in a_values:
        sc = bulk("Cu", "fcc", a=a, cubic=True) * (size, size, size)
        sc.calc = calc_factory(use_cell_list=(n_atoms > 500))
        energies.append(sc.get_potential_energy())
        volumes.append(sc.get_volume())

    volumes  = np.array(volumes)
    energies = np.array(energies)
    save_array(outdir / "ev_curve.txt",
               "volume_A3   energy_eV",
               np.column_stack([volumes, energies]))

    print_subsection("Birch-Murnaghan EOS fit")
    eos: dict = {}
    try:
        eos = fit_eos(volumes, energies, n_atoms)
        print(compare("Equilibrium lattice param [Å]",
                       eos["a0_A"], CU_EXP["a0_A"], "Å", tol_pct=1.0))
        print(compare("Cohesive energy [eV/atom]",
                       eos["E0_eV_per_atom"], CU_EXP["E_coh_eV"], "eV/atom", tol_pct=5.0))
        print(compare("Bulk modulus [GPa]",
                       eos["B0_GPa"], CU_EXP["B0_GPa"], "GPa", tol_pct=15.0))
        print(f"  B0' (pressure derivative)              = {eos['B0p']:.3f}")
    except Exception as exc:
        print(f"  [WARN] EOS fit failed: {exc}")

    return {
        "n_atoms":  n_atoms,
        "e_sp":     e_sp,
        "eos":      eos,
        "volumes":  volumes,
        "energies": energies,
    }


# ── TEST 3: Force symmetry ─────────────────────────────────────────────────────

def run_force_symmetry_test(calc_factory, size: int) -> None:
    print_section("TEST 3  —  Force symmetry (ΣF ≈ 0)")

    atoms   = bulk("Cu", "fcc", a=CU_EXP["a0_A"], cubic=True) * (size, size, size)
    n_atoms = len(atoms)
    atoms.calc = calc_factory(use_cell_list=(n_atoms > 500))
    forces  = atoms.get_forces()
    f_sum   = forces.sum(axis=0)
    f_max   = np.abs(f_sum).max()
    status  = "PASS" if f_max < 1e-3 else "FAIL"
    print(f"  |ΣF_x|={f_sum[0]:.3e}  |ΣF_y|={f_sum[1]:.3e}  |ΣF_z|={f_sum[2]:.3e}  eV/Å")
    print(f"  Max component of ΣF = {f_max:.3e} eV/Å  (tolerance < 1e-3)  [{status}]")


# ── TEST 4: NVT Langevin ───────────────────────────────────────────────────────

def run_nvt(
    calc_factory,
    outdir: Path,
    n_atoms: int,
    temperature_K: float = 300.0,
    equil_steps: int = 2_500,
    prod_steps: int = 5_000,
    timestep_fs: float = 2.0,
    log_interval: int = 50,
    quiet: bool = False,
    label: str = "300K",
    n_rdf_frames: int = 10,
    msd_max_frames: int = 500,
    friction_per_fs: float = 0.02,  # Langevin friction [1/fs]; 0.02 ≈ 50 fs coupling time
) -> dict:
    """NVT Langevin MD.

    Memory strategy
    ---------------
    - Log records are streamed line-by-line to disk inside the MD loop.
      No growing list is held in memory for the log.
    - RDF frames are held in a fixed-size ring buffer (``n_rdf_frames``).
      Only the most recent frames are kept regardless of run length.
    - MSD positions are subsampled so that at most ``msd_max_frames``
      frames are kept in memory at any time.
    - ``log_interval`` is auto-scaled upward for large systems so that the
      total number of I/O events stays below ~500 per run.
    """
    print_section(f"TEST 4  —  NVT Langevin  T={temperature_K:.0f} K")

    sc_size = _n_reps(n_atoms, min_rep=3)
    atoms   = bulk("Cu", "fcc", a=CU_EXP["a0_A"], cubic=True) * (sc_size, sc_size, sc_size)
    n       = len(atoms)

    n_records_target = 500
    auto_interval    = max(log_interval, prod_steps // n_records_target)
    if auto_interval > log_interval:
        print(f"  [INFO] log_interval auto-scaled {log_interval} → {auto_interval}"
              f" for {n}-atom system")
        log_interval = auto_interval

    print(f"  Supercell: {sc_size}³ = {n} atoms   T = {temperature_K:.0f} K")

    atoms.calc = calc_factory(use_cell_list=(n > 300))
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K,
                                 rng=np.random.default_rng(42))
    Stationary(atoms)

    dyn = Langevin(
        atoms,
        timestep=timestep_fs * units.fs,
        temperature_K=temperature_K,
        friction=friction_per_fs / units.fs,
        rng=np.random.default_rng(123),
    )

    print(f"  Equilibration: {equil_steps} steps × {timestep_fs} fs"
          f" = {equil_steps * timestep_fs / 1000:.1f} ps")
    dyn.run(equil_steps)

    print(f"  Production:    {prod_steps} steps × {timestep_fs} fs"
          f" = {prod_steps * timestep_fs / 1000:.1f} ps")

    tag      = f"nvt_{label}"
    log_path = outdir / f"{tag}_log.txt"

    temps: list[float] = []
    epots: list[float] = []

    rdf_buf: collections.deque[np.ndarray] = collections.deque(maxlen=n_rdf_frames)

    n_callbacks   = prod_steps // log_interval
    msd_stride    = max(1, n_callbacks // msd_max_frames)
    msd_frames:   list[np.ndarray] = []
    msd_times_fs: list[float]      = []
    callback_idx  = 0

    log_every = max(1, n_callbacks // 20)

    with open(log_path, "w", buffering=1) as log_file:
        log_file.write("# time_fs   T_K   Epot_eV_per_atom   Etot_eV_per_atom\n")

        def _record() -> None:
            nonlocal callback_idx
            T_inst = atoms.get_temperature()
            E_pot  = atoms.get_potential_energy() / n
            E_kin  = atoms.get_kinetic_energy() / n
            t_fs   = dyn.get_time() / units.fs

            temps.append(T_inst)
            epots.append(E_pot)

            log_file.write(f"{t_fs:.3f} {T_inst:.4f} {E_pot:.6f} {E_pot + E_kin:.6f}\n")

            rdf_buf.append(atoms.get_positions().copy())

            if callback_idx % msd_stride == 0:
                msd_frames.append(atoms.get_positions().copy())
                msd_times_fs.append(t_fs)

            if not quiet and callback_idx % log_every == 0:
                print(f"    step {callback_idx * log_interval:7d}"
                      f"  T={T_inst:6.1f} K  Epot={E_pot:.4f} eV/atom")

            callback_idx += 1

        dyn.attach(_record, interval=log_interval)
        dyn.run(prod_steps)

    print(f"  Saved: {log_path}")

    temps_arr = np.array(temps)
    epots_arr = np.array(epots)

    print_subsection("NVT production statistics")
    print(f"  Temperature : {temps_arr.mean():.2f} ± {temps_arr.std():.2f} K"
          f"  (target {temperature_K:.0f} K)")
    print(f"  Epot/atom   : {epots_arr.mean():.5f} ± {epots_arr.std():.5f} eV")
    print(f"  Log records : {len(temps_arr)}"
          f"  |  MSD frames: {len(msd_frames)}"
          f"  |  RDF frames buffered: {len(rdf_buf)}")

    print_subsection(f"Radial distribution function (last {len(rdf_buf)} frames average)")
    cell  = np.array(atoms.get_cell())
    g_rdf: np.ndarray | None = None
    r_rdf: np.ndarray | None = None
    for pos in rdf_buf:
        r, g = compute_rdf(pos, cell, r_max=8.0, n_bins=300)
        g_rdf = g if g_rdf is None else g_rdf + g
    r1_peak: float | None = None
    if g_rdf is not None:
        g_rdf   /= len(rdf_buf)
        r_rdf    = r
        peak_idx = np.argmax(g_rdf[g_rdf.size // 10:]) + g_rdf.size // 10
        r1_peak  = float(r_rdf[peak_idx])
        r1_exp   = CU_EXP["a0_A"] / np.sqrt(2.0)
        print(f"  1st RDF peak at r = {r1_peak:.3f} Å  (FCC NN expected: {r1_exp:.3f} Å)")
        save_array(outdir / f"{tag}_rdf.txt",
                   "r_A   g_r",
                   np.column_stack([r_rdf, g_rdf]))

    msd      = msd_from_trajectory(msd_frames)
    times_ps = np.array(msd_times_fs) / 1000.0
    print(f"  MSD at end of run: {msd[-1]:.3f} Å²  ({len(msd_frames)} frames sampled)")
    save_array(outdir / f"{tag}_msd.txt",
               "time_ps   msd_A2",
               np.column_stack([times_ps, msd]))

    return {
        "T_mean":    float(temps_arr.mean()),
        "T_std":     float(temps_arr.std()),
        "Epot_mean": float(epots_arr.mean()),
        "r1_rdf":    r1_peak,
        "msd_final": float(msd[-1]),
    }


# ── TEST 5: NPT Nosé-Hoover ────────────────────────────────────────────────────

def run_npt(
    calc_factory,
    outdir: Path,
    n_atoms: int,
    temperature_K: float = 300.0,
    pressure_GPa: float = 0.0,
    equil_steps: int = 2_500,
    prod_steps: int = 7_500,
    timestep_fs: float = 2.0,
    log_interval: int = 50,
    quiet: bool = False,
) -> dict:
    """NPT Nosé-Hoover MD.

    Memory strategy
    ---------------
    - Scalar thermodynamic quantities (T, P, V) are kept in lists for
      post-run statistics; their memory cost is negligible.
    - Log records are streamed line-by-line to disk — no growing tuple list.
    - ``log_interval`` is auto-scaled for large systems (same policy as NVT).
    """
    print_section(
        f"TEST 5  —  NPT Nosé-Hoover  T={temperature_K:.0f} K  P={pressure_GPa:.1f} GPa"
    )

    sc_size = max(2, round((n_atoms / _FCC_BASIS) ** (1.0 / 3.0)))
    atoms   = bulk("Cu", "fcc", a=CU_EXP["a0_A"], cubic=True) * (sc_size, sc_size, sc_size)
    n       = len(atoms)

    n_records_target = 500
    auto_interval    = max(log_interval, prod_steps // n_records_target)
    if auto_interval > log_interval:
        print(f"  [INFO] log_interval auto-scaled {log_interval} → {auto_interval}"
              f" for {n}-atom system")
        log_interval = auto_interval

    atoms.calc = calc_factory(use_cell_list=(n > 300))
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K,
                                 rng=np.random.default_rng(77))
    Stationary(atoms)

    dyn = _make_npt_dyn(atoms, timestep_fs, temperature_K, pressure_GPa)
    dyn.run(equil_steps)

    temps: list[float]     = []
    pressures: list[float] = []
    volumes: list[float]   = []

    log_path     = outdir / "npt_300K_log.txt"
    n_callbacks  = prod_steps // log_interval
    log_every    = max(1, n_callbacks // 20)
    callback_idx = 0

    with open(log_path, "w", buffering=1) as log_file:
        log_file.write("# time_fs   T_K   P_GPa   V_A3   Epot_eV_per_atom\n")

        def _record_npt() -> None:
            nonlocal callback_idx
            T_inst = atoms.get_temperature()
            stress = atoms.get_stress(include_ideal_gas=True)
            P_inst = -stress[:3].mean() * EV_PER_A3_TO_GPA
            V_inst = atoms.get_volume()
            E_pot  = atoms.get_potential_energy() / n
            t_fs   = dyn.get_time() / units.fs

            temps.append(T_inst)
            pressures.append(P_inst)
            volumes.append(V_inst)

            log_file.write(
                f"{t_fs:.3f} {T_inst:.4f} {P_inst:.4f} {V_inst:.4f} {E_pot:.6f}\n"
            )

            if not quiet and callback_idx % log_every == 0:
                print(f"    step {callback_idx * log_interval:7d}"
                      f"  T={T_inst:6.1f} K  P={P_inst:+.2f} GPa  V={V_inst:.2f} Å³")

            callback_idx += 1

        dyn.attach(_record_npt, interval=log_interval)
        dyn.run(prod_steps)

    print(f"  Saved: {log_path}")

    temps_np     = np.array(temps)
    pressures_np = np.array(pressures)
    volumes_np   = np.array(volumes)

    V_mean = float(volumes_np.mean())
    a_mean = (V_mean / n * _FCC_BASIS) ** (1.0 / 3.0)

    # ── Bulk modulus from NPT volume fluctuations ──────────────────────────────
    # B_T = k_B T <V> / Var(V)   [isothermal bulk modulus, NPT ensemble]
    #
    # Best practice: discard the first 20 % of production to remove any
    # residual volume drift from the equilibration-to-production transition.
    # Also check for remaining linear drift and warn if it is significant.
    #
    # NOTE on accuracy of B_fluct (why TEST 5 often gives B ≈ 0 or NaN)
    # ─────────────────────────────────────────────────────────────────────
    # The barostat introduces a "breathing" oscillation of the cell with period
    #
    #   τ_osc ≈ 2π · τ_p · √(B_ref / B_true)
    #
    # With τ_p = 2 ps and B ≈ 140 GPa this gives τ_osc ~ 10–15 ps for a
    # well-tuned barostat.  However, for a 16³ supercell (16 384 atoms) the
    # barostat mass is large and τ_osc can reach 50–100 ps.  If the production
    # run is shorter than ~5 × τ_osc the cell has not completed its damped
    # oscillation, dV is dominated by the oscillation, and B_fluct is wrong.
    #
    # How to get a reliable B_fluct in TEST 5:
    #   (a) Smaller cell (--size 4 to 8, i.e. 256–2048 atoms):
    #       Relative volume fluctuations δV/V ~ √(k_BT / B·V) scale as V^{-1/2},
    #       so a smaller cell gives larger relative fluctuations and a better
    #       signal-to-noise ratio.  4³ (256 atoms) is the practical lower bound.
    #   (b) Larger τ_p (softer barostat, e.g. --tau-p 5000):
    #       A slower barostat damps oscillations before production starts,
    #       provided equil_steps ≫ τ_osc (extend --equil-steps accordingly).
    #   (c) Much longer production (--prod-steps ≥ 500 000 at 2 fs = 1 ns):
    #       Ensures Var(V) converges to the thermodynamic value independent of
    #       initial drift.  This is the most reliable fix but also the slowest.
    #   (d) Use elastic constants from TEST 9 instead:
    #       B₀ = (C11 + 2·C12) / 3 from the Voigt finite-strain perturbation
    #       is far more accurate and does not require long NPT sampling.
    n_trim   = max(1, len(volumes_np) // 5)
    vol_stat = volumes_np[n_trim:]                # stationary window

    # Drift check: slope of V(t) over stationary window
    t_idx      = np.arange(len(vol_stat), dtype=float)
    drift_coef = float(np.polyfit(t_idx, vol_stat, 1)[0])   # Å³/step
    drift_pct  = abs(drift_coef) * len(vol_stat) / vol_stat.mean() * 100.0
    if drift_pct > 0.5:
        print(f"  [WARN] Volume drift {drift_pct:.2f} % detected over stationary window"
              f" — extend equil_steps (currently used {n_trim} discard frames).")

    dV2     = float(np.var(vol_stat))           # Å⁶
    V_stat  = float(vol_stat.mean())
    kBT_ev  = KB_EV * temperature_K             # eV

    if dV2 < 1e-30:
        print("  [WARN] Volume variance ≈ 0 — cell may not be fluctuating yet. "
              "Increase equil_steps or prod_steps.")
        B_fluct = float("nan")
    else:
        B_fluct = kBT_ev * V_stat / dV2 / EV_PER_A3_TO_GPA

    print_subsection("NPT production statistics")
    print(f"  Temperature  : {temps_np.mean():.2f} ± {temps_np.std():.2f} K")
    print(f"  Pressure     : {pressures_np.mean():.3f} ± {pressures_np.std():.3f} GPa")
    print(compare("NPT lattice parameter [Å]", a_mean, CU_EXP["a0_A"], "Å", tol_pct=2.0))
    print(f"  Bulk modulus from V-fluctuations ≈ {B_fluct:.1f} GPa")

    return {
        "T_mean":  float(temps_np.mean()),
        "P_mean":  float(pressures_np.mean()),
        "a_mean":  float(a_mean),
        "B_fluct": float(B_fluct),
    }


# ── TEST 6: Thermal expansion (temperature series) ────────────────────────────

def run_thermal_expansion(
    calc_factory,
    outdir: Path,
    n_atoms: int,
    temperatures: list[float],
    steps_per_T: int = 2_000,
    equil_steps: int = 1_000,
    timestep_fs: float = 2.0,
    quiet: bool = False,
    vol_record_interval: int = 10,   # how often to sample volume [steps]
) -> dict:
    print_section("TEST 6  —  Thermal expansion (temperature series)")

    sc_size  = max(2, round((n_atoms / _FCC_BASIS) ** (1.0 / 3.0)))
    a_values = []

    for T in temperatures:
        atoms = bulk("Cu", "fcc", a=CU_EXP["a0_A"], cubic=True) * (sc_size, sc_size, sc_size)
        n     = len(atoms)
        atoms.calc = calc_factory(use_cell_list=(n > 300))
        MaxwellBoltzmannDistribution(atoms, temperature_K=T,
                                     rng=np.random.default_rng(int(T)))
        Stationary(atoms)

        dyn = _make_npt_dyn(atoms, timestep_fs, T, 0.0)
        dyn.run(equil_steps)

        vol_samples: list[float] = []

        def _rec_vol() -> None:
            vol_samples.append(atoms.get_volume())

        dyn.attach(_rec_vol, interval=vol_record_interval)
        dyn.run(steps_per_T)

        V_mean = float(np.mean(vol_samples))
        a_T    = (V_mean / n * _FCC_BASIS) ** (1.0 / 3.0)
        a_values.append(a_T)
        if not quiet:
            print(f"  T = {T:5.0f} K   a = {a_T:.5f} Å")

    temps_arr = np.array(temperatures)
    a_arr     = np.array(a_values)

    coeffs = np.polyfit(temps_arr, a_arr, 1)
    da_dT  = float(coeffs[0])
    alpha  = da_dT / float(a_arr[0])

    print_subsection("Thermal expansion results")
    print(f"  da/dT = {da_dT * 1e5:.4f} × 10⁻⁵ Å/K")
    print(compare("Linear TEC α [×10⁻⁶ K⁻¹]",
                  alpha * 1e6, CU_EXP["alpha_1perK"] * 1e6, "×10⁻⁶ K⁻¹", tol_pct=40.0))

    save_array(outdir / "thermal_expansion.txt",
               "T_K   a_A",
               np.column_stack([temps_arr, a_arr]))

    return {"alpha_per_K": alpha, "a_T": list(zip(temperatures, a_values))}


# ── TEST 7: Near-melting NVT ───────────────────────────────────────────────────

def run_near_melting(
    calc_factory,
    outdir: Path,
    n_atoms: int,
    temperature_K: float = 1300.0,
    equil_steps: int = 1_000,
    prod_steps: int = 2_500,
    timestep_fs: float = 1.0,
    log_interval: int = 25,
    quiet: bool = False,
    friction_per_fs: float = 0.05,          # higher friction helps damp near-melting oscillations
    alpha_lin_expansion: float = 17e-6,     # linear TEC [1/K] for cell pre-scaling
) -> dict:
    print_section(f"TEST 7  —  Near-melting NVT  T={temperature_K:.0f} K")

    sc_size = _n_reps(n_atoms, min_rep=3)
    atoms   = bulk("Cu", "fcc", a=CU_EXP["a0_A"], cubic=True) * (sc_size, sc_size, sc_size)
    n       = len(atoms)
    atoms.calc = calc_factory(use_cell_list=(n > 300))

    alpha_approx = alpha_lin_expansion
    scale_factor = 1.0 + alpha_approx * (temperature_K - 300.0) / 3.0
    atoms.set_cell(atoms.get_cell() * scale_factor, scale_atoms=True)

    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K,
                                 rng=np.random.default_rng(99))
    Stationary(atoms)

    dyn = Langevin(
        atoms,
        timestep=timestep_fs * units.fs,
        temperature_K=temperature_K,
        friction=friction_per_fs / units.fs,
        rng=np.random.default_rng(999),
    )
    dyn.run(equil_steps)

    traj_pos: list[np.ndarray] = []
    temps: list[float] = []

    def _rec() -> None:
        traj_pos.append(atoms.get_positions().copy())
        temps.append(atoms.get_temperature())

    dyn.attach(_rec, interval=log_interval)
    dyn.run(prod_steps)

    msd      = msd_unwrapped_from_trajectory(traj_pos, np.array(atoms.get_cell()))
    times_ps = np.arange(len(msd)) * log_interval * timestep_fs / 1000.0
    T_mean   = float(np.mean(temps))

    print_subsection("Near-melting MSD")
    print(f"  Mean temperature : {T_mean:.1f} K")
    print(f"  MSD at end       : {msd[-1]:.3f} Å²")

    save_array(outdir / f"nvt_{int(temperature_K)}K_msd.txt",
               "time_ps   msd_A2",
               np.column_stack([times_ps, msd]))

    return {"T_mean": T_mean, "msd_final": float(msd[-1])}


# ── TEST 8: NPT Triclinic cell ─────────────────────────────────────────────────

def run_npt_triclinic(
    calc_factory,
    outdir: Path,
    size: int,
    temperature_K: float = 300.0,
    pressure_GPa: float = 0.0,
    equil_steps: int = 2_000,   # NVT pre-equilibration before shear is applied
    prod_steps: int = 5_000,
    timestep_fs: float = 2.0,
    shear_fraction: float = 0.02,  # shear as fraction of cell length (2 %)
                                   # previously hardcoded as 2.0 Å absolute
) -> dict:
    from ase.md.npt import NPT

    print_section(
        f"TEST 8  —  NPT Triclinic Cell  T={temperature_K:.0f} K  P={pressure_GPa:.1f} GPa"
    )

    atoms   = bulk("Cu", "fcc", a=CU_EXP["a0_A"], cubic=True) * (size, size, size)
    n_atoms = len(atoms)
    atoms.calc = calc_factory(use_cell_list=(n_atoms > 300))
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K,
                                 rng=np.random.default_rng(88))
    Stationary(atoms)

    # ── Step 1: brief NVT equilibration before applying the shear ─────────────
    # Applying a shear to a cold/non-equilibrated cell causes large initial
    # stresses that mislead the barostat.
    if equil_steps > 0:
        pre_dyn = Langevin(
            atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=temperature_K,
            friction=0.02 / units.fs,
            rng=np.random.default_rng(880),
        )
        pre_dyn.run(equil_steps)
        print(f"  NVT pre-equil done.  T_inst = {atoms.get_temperature():.1f} K")

    # ── Step 2: apply triclinic shear relative to cell lengths ────────────────
    cell       = atoms.get_cell().copy()
    shear_abs  = shear_fraction * float(np.linalg.norm(cell[0]))  # Å
    cell[0, 1] += shear_abs
    cell[1, 2] += shear_abs
    atoms.set_cell(cell, scale_atoms=True)

    init_angles = atoms.get_cell_lengths_and_angles()[3:]
    print(f"  Shear applied: {shear_fraction*100:.1f}% of |a₀| = {shear_abs:.3f} Å per off-diag")
    print(f"  Initial cell angles: {init_angles}")

    dyn = NPT(
        atoms,
        timestep=timestep_fs * units.fs,
        temperature_K=temperature_K,
        externalstress=pressure_GPa * units.GPa,
        ttime=25.0 * units.fs,                       # thermostat time constant
        pfactor=(2000.0 * units.fs) ** 2 * (140.0 * units.GPa),  # barostat pfactor
        mask=None,
    )

    def _print_progress() -> None:
        stress = atoms.get_stress(include_ideal_gas=True)
        P_avg  = -stress[:3].mean() * EV_PER_A3_TO_GPA
        angles = atoms.get_cell_lengths_and_angles()[3:]
        print(f"    step {dyn.get_number_of_steps():5d}"
              f"  T={atoms.get_temperature():5.1f} K"
              f"  P_avg={P_avg:6.2f} GPa"
              f"  angles={angles}")

    dyn.attach(_print_progress, interval=500)
    print(f"  Running {prod_steps * timestep_fs / 1000.0:.1f} ps relaxation …")
    dyn.run(prod_steps)

    final_angles = atoms.get_cell_lengths_and_angles()[3:]
    print(f"\n  Final cell angles: {final_angles}")

    success = all(abs(ang - 90.0) < 2.0 for ang in final_angles)
    print(f"  [{'PASS' if success else 'FAIL'}] Cell returned towards orthogonality.")

    return {
        "init_angles":  init_angles.tolist(),
        "final_angles": final_angles.tolist(),
        "success":      success,
    }


# ── TEST 9: Elastic constants ──────────────────────────────────────────────────

def run_elastic_constants(
    calc_factory,
    outdir: Path,
    size: int = 5,
    a0: float | None = None,
    strain_deltas: tuple[float, ...] = (-0.015, -0.010, -0.005, +0.005, +0.010, +0.015),
) -> dict:
    """
    Compute C11, C12, C44 via Voigt finite-strain perturbation.

    C_ij = dσ_i / dε_j  (linear regression over six strain amplitudes)

    Strains applied:
      Uniaxial ε_xx → σ_xx vs ε_xx slope = C11
                       σ_yy vs ε_xx slope = C12
      Shear    ε_yz → σ_yz vs ε_yz slope = C44

    Derived:
      B0  = (C11 + 2·C12) / 3    [Voigt bulk modulus — cross-check of TEST 2]
      G   = (C11 − C12 + 3·C44) / 5  [Voigt shear modulus]
      A_Z = 2·C44 / (C11 − C12)  [Zener anisotropy; isotropic crystal → 1]

    Parameters
    ----------
    a0 : float or None
        Lattice parameter to use for the undeformed reference cell [Å].
        **Pass the EAM equilibrium value** from the EOS fit (TEST 2) for best
        accuracy.  If None, falls back to the experimental value
        (CU_EXP["a0_A"] = 3.615 Å) which introduces a small systematic stress
        offset onto all elastic constants.
    strain_deltas : tuple of float
        Voigt strain amplitudes (dimensionless) to apply.  Default is ±0.5, ±1.0,
        ±1.5 % — appropriate for EAM Cu.
    """
    print_section("TEST 9  —  Elastic Constants  (C11, C12, C44)")

    if a0 is None:
        a0 = CU_EXP["a0_A"]
        print(f"  [INFO] a0 not provided — using experimental value {a0:.4f} Å.")
        print(f"         Pass a0=eos['a0_A'] from TEST 2 for higher accuracy.")
    else:
        print(f"  a0 = {a0:.5f} Å  (EAM equilibrium from EOS fit)")

    atoms0 = bulk("Cu", "fcc", a=a0, cubic=True) * (size, size, size)
    N      = len(atoms0)
    use_cl = N > 500
    print(f"  Supercell: {size}³ = {N} atoms")

    # Six strain amplitudes: ±0.5 %, ±1.0 %, ±1.5 %
    deltas = np.array(strain_deltas)

    # ── Uniaxial strain ε_xx ──────────────────────────────────────────────────
    print_subsection("Uniaxial strain scan (ε_xx = δ)")
    sxx, syy = [], []
    for δ in deltas:
        at = atoms0.copy()
        at.calc = calc_factory(use_cell_list=use_cl)
        F = np.eye(3)
        F[0, 0] += δ
        at.set_cell(atoms0.get_cell() @ F.T, scale_atoms=True)
        σ = at.get_stress()   # Voigt (6,): xx yy zz yz xz xy  [eV/Å³]
        sxx.append(σ[0] * EV_PER_A3_TO_GPA)
        syy.append(σ[1] * EV_PER_A3_TO_GPA)
        print(f"    δ={δ:+.3f}  σ_xx={sxx[-1]:+.4f} GPa  σ_yy={syy[-1]:+.4f} GPa")

    c11 = float(np.polyfit(deltas, sxx, 1)[0])
    c12 = float(np.polyfit(deltas, syy, 1)[0])

    # ── Shear strain ε_yz ─────────────────────────────────────────────────────
    print_subsection("Shear strain scan (ε_yz = δ)")
    syz = []
    for δ in deltas:
        at = atoms0.copy()
        at.calc = calc_factory(use_cell_list=use_cl)
        F = np.eye(3)
        F[1, 2] += δ / 2.0   # symmetric shear deformation
        F[2, 1] += δ / 2.0
        at.set_cell(atoms0.get_cell() @ F.T, scale_atoms=True)
        σ = at.get_stress()
        syz.append(σ[3] * EV_PER_A3_TO_GPA)   # Voigt index 3 = yz
        print(f"    δ={δ:+.3f}  σ_yz={syz[-1]:+.4f} GPa")

    c44 = float(np.polyfit(deltas, syz, 1)[0])

    B0    = (c11 + 2.0 * c12) / 3.0
    G     = (c11 - c12 + 3.0 * c44) / 5.0
    zener = 2.0 * c44 / (c11 - c12)

    print_subsection("Results")
    print(compare("C11  [GPa]", c11, CU_EXP["C11_GPa"], "GPa", tol_pct=10.0))
    print(compare("C12  [GPa]", c12, CU_EXP["C12_GPa"], "GPa", tol_pct=10.0))
    print(compare("C44  [GPa]", c44, CU_EXP["C44_GPa"], "GPa", tol_pct=10.0))
    print(compare("Zener anisotropy A = 2C44/(C11-C12)",
                  zener, CU_EXP["A_zener"], "", tol_pct=15.0))
    print(f"  {'B0 = (C11+2C12)/3':<40s}  calc={B0:10.4f}  exp={CU_EXP['B0_GPa']:10.4f}"
          "  GPa  (cross-check TEST 2)")
    print(f"  {'Voigt shear G':<40s}  calc={G:10.4f}  ref ~45.0  GPa")

    save_array(
        outdir / "elastic_strain_stress.txt",
        "delta  sigma_xx_GPa  sigma_yy_GPa  sigma_yz_GPa",
        np.column_stack([deltas, sxx, syy, syz]),
    )

    return {"C11": c11, "C12": c12, "C44": c44, "B0": B0, "G": G, "A_zener": zener}


# ── TEST 10: Vacancy formation energy ─────────────────────────────────────────

def run_vacancy_formation(
    calc_factory,
    pot_path: str,
    outdir: Path,
    size: int = 4,
    a0: float | None = None,
) -> dict:
    """
    E_vac = E(N-1, defect cell) − (N-1)/N · E(N, perfect cell)

    The CPU backend is used explicitly so that per-atom energies are available
    via get_potential_energies() (GPU backend does not provide this).  This
    enables mapping of the energy perturbation field around the vacancy,
    showing how far the defect's influence extends into the surrounding bulk.

    Convention: unrelaxed vacancy — ionic positions are not optimised.
    The unrelaxed value is typically ~10 % above the relaxed formation energy
    for EAM Cu potentials.

    Parameters
    ----------
    a0 : float or None
        Lattice parameter [Å].  Shell radii (1st, 2nd, 3rd NN) are derived as
        multiples of a0/√2 (FCC nearest-neighbour distance).  If None, uses
        CU_EXP["a0_A"].  Pass the EAM equilibrium value for accurate shells.
    """
    print_section("TEST 10  —  Vacancy Formation Energy  (CPU backend)")

    from cream import CreamCalculator

    if a0 is None:
        a0 = CU_EXP["a0_A"]

    # FCC nearest-neighbour distance and shell boundaries
    # 1NN: a/√2 ≈ 0.707a;  2NN: a ≈ 1.000a;  3NN: a√1.5 ≈ 1.225a
    # Shell cutoffs set at midpoints between shells:
    #   < 0.85a  → 1st shell,  0.85a–1.15a → 2nd,  1.15a–1.45a → 3rd
    r1_cut = 0.85 * a0
    r2_cut = 1.15 * a0
    r3_cut = 1.45 * a0

    atoms_perf = bulk("Cu", "fcc", a=a0, cubic=True) * (size, size, size)
    N          = len(atoms_perf)
    use_cl     = N > 500
    print(f"  Perfect cell: {size}³ = {N} atoms  [forcing CPU backend for per-atom energies]")

    calc_p = CreamCalculator(pot_path, use_cell_list=use_cl, backend="cpu")
    atoms_perf.calc = calc_p
    E_perf    = atoms_perf.get_potential_energy()
    epot_perf = atoms_perf.get_potential_energies()   # (N,) eV per atom

    print(f"  E_perf = {E_perf:.6f} eV  ({E_perf/N:.6f} eV/atom)")
    print(f"  Per-atom energy std-dev = {epot_perf.std():.3e} eV"
          "  (≈ 0 expected for perfect crystal)")

    # Remove atom nearest to cell centre → vacancy
    centre  = atoms_perf.get_cell().sum(axis=0) / 2.0
    dists_c = np.linalg.norm(atoms_perf.get_positions() - centre, axis=1)
    vac_idx = int(np.argmin(dists_c))
    pos_vac = atoms_perf.get_positions()[vac_idx].copy()

    atoms_vac = atoms_perf.copy()
    del atoms_vac[vac_idx]
    M = len(atoms_vac)   # N - 1

    calc_v = CreamCalculator(pot_path, use_cell_list=use_cl, backend="cpu")
    atoms_vac.calc = calc_v
    E_vac_cell = atoms_vac.get_potential_energy()
    epot_vac   = atoms_vac.get_potential_energies()   # (M,) eV per atom

    E_form = E_vac_cell - (M / N) * E_perf
    print(f"  Vacancy cell: {M} atoms  E = {E_vac_cell:.6f} eV")
    print(compare("E_vac (unrelaxed) [eV]", E_form, CU_EXP["E_vac_eV"],
                  "eV", tol_pct=20.0))
    print("  Note: unrelaxed EAM E_vac is typically ~10 % above the relaxed value.")

    # ── Per-atom energy perturbation field ────────────────────────────────────
    print_subsection("Per-atom energy perturbation around vacancy")
    bulk_e_ref = float(np.median(epot_vac))
    delta_E    = epot_vac - bulk_e_ref       # (M,) meV perturbations

    pos_rem    = atoms_vac.get_positions()
    dists_vac  = np.linalg.norm(pos_rem - pos_vac, axis=1)

    sh1 = dists_vac < r1_cut
    sh2 = (dists_vac >= r1_cut) & (dists_vac < r2_cut)
    sh3 = (dists_vac >= r2_cut) & (dists_vac < r3_cut)
    far = dists_vac >= r3_cut

    def _shell(mask: np.ndarray, label: str) -> None:
        if not mask.any():
            return
        dE = delta_E[mask] * 1000.0
        print(f"    {label:<40s}  n={mask.sum():3d}"
              f"  ΔE_mean={dE.mean():+7.2f} meV"
              f"  ΔE_max={dE.max():+7.2f} meV")

    _shell(sh1, f"1st shell  (r < {r1_cut:.2f} Å = 0.85a₀)")
    _shell(sh2, f"2nd shell  ({r1_cut:.2f}–{r2_cut:.2f} Å)")
    _shell(sh3, f"3rd shell  ({r2_cut:.2f}–{r3_cut:.2f} Å)")
    _shell(far, f"Bulk       (r ≥ {r3_cut:.2f} Å) ")

    save_array(
        outdir / "vacancy_perturbation.txt",
        "dist_to_vac_A  delta_E_meV",
        np.column_stack([dists_vac, delta_E * 1000.0]),
    )

    return {
        "E_form_eV":       float(E_form),
        "sh1_delta_E_meV": float(delta_E[sh1].mean() * 1000.0) if sh1.any() else 0.0,
    }


# ── TEST 11: EAM per-atom fields ───────────────────────────────────────────────

def run_eam_fields(
    calc_factory,
    pot_path: str,
    outdir: Path,
    size: int = 6,
    timestep_fs: float = 2.0,    # was hardcoded 2.0 fs
    nvt_steps: int = 2_000,      # was hardcoded 2000 steps (4 ps at 2 fs)
) -> dict:
    """
    Interrogate EAM intermediates exposed only on the CPU backend:

      cream_densities           — electron density ρᵢ at each atom
      cream_embedding_energies  — embedding contribution F(ρᵢ) [eV]

    Physics extracted
    -----------------
    (i)  Perfect crystal (0 K): ρᵢ should be a near-delta-function; any
         spread is numerical noise. ΣF(ρᵢ)/E_tot gives the embedding
         fraction of binding energy vs the pair-potential term.

    (ii) Thermal snapshot (300 K NVT): atomic vibrations broaden both
         distributions. The broadening ratio std(ρ)_300K / std(ρ)_0K
         characterises the sensitivity of ρᵢ to thermal disorder.

    (iii) Sorted F(ρᵢ) vs ρᵢ traces the embedding function F(ρ) curve
         as sampled by the MD trajectory.
    """
    print_section("TEST 11  —  EAM Per-atom Fields: ρᵢ and F(ρᵢ)  [CPU only]")

    from cream import CreamCalculator

    atoms  = bulk("Cu", "fcc", a=CU_EXP["a0_A"], cubic=True) * (size, size, size)
    N      = len(atoms)
    use_cl = N > 500
    print(f"  Supercell: {size}³ = {N} atoms  [forcing CPU backend]")

    calc = CreamCalculator(pot_path, use_cell_list=use_cl, backend="cpu")
    atoms.calc = calc
    atoms.calc.calculate(atoms, properties=["energy", "cream_densities", "cream_embedding_energies"])
    
    E_tot = atoms.get_potential_energy()
    Epot_0K = atoms.get_potential_energies().copy()
    rho_0K  = atoms.calc.results["cream_densities"].copy()
    Frho_0K = atoms.calc.results["cream_embedding_energies"].copy()
    
    emb_frac = float(Frho_0K.sum() / E_tot)

    print_subsection("Perfect crystal (0 K)")
    print(f"  E_tot = {E_tot:.6f} eV  ({E_tot/N:.6f} eV/atom)")
    print(f"  ρᵢ    : mean={rho_0K.mean():.8f}   std={rho_0K.std():.3e}"
          f"   range=[{rho_0K.min():.6f}, {rho_0K.max():.6f}]")
    print(f"  F(ρᵢ) : mean={Frho_0K.mean():.8f} eV  std={Frho_0K.std():.3e} eV"
          f"   range=[{Frho_0K.min():.6f}, {Frho_0K.max():.6f}] eV")
    print(f"  Embedding fraction ΣF(ρ)/E_tot = {emb_frac:.6f}")

    save_array(
        outdir / "eam_fields_0K.txt",
        "rho_i  F_rho_eV  Epot_i_eV",
        np.column_stack([rho_0K, Frho_0K, Epot_0K]),
    )

    print_subsection(f"Thermal snapshot ({nvt_steps} steps NVT @ 300 K"
                     f" = {nvt_steps * timestep_fs / 1000:.1f} ps)")
    MaxwellBoltzmannDistribution(atoms, temperature_K=300.0,
                                 rng=np.random.default_rng(7))
    Stationary(atoms)

    dyn = Langevin(
        atoms,
        timestep=timestep_fs * units.fs,
        temperature_K=300.0,
        friction=0.02 / units.fs,
        rng=np.random.default_rng(77),
    )
    dyn.run(nvt_steps)   # NVT equilibration (default 4 ps)

    atoms.calc.calculate(atoms, properties=["energy", "cream_densities", "cream_embedding_energies"])
    
    rho_T  = atoms.calc.results["cream_densities"].copy()
    Frho_T = atoms.calc.results["cream_embedding_energies"].copy()
    T_inst = atoms.get_temperature()

    std_ratio_rho  = rho_T.std()  / max(rho_0K.std(),  1e-8)
    std_ratio_Frho = Frho_T.std() / max(Frho_0K.std(), 1e-8)

    print(f"  T_inst = {T_inst:.1f} K")
    print(f"  ρᵢ    : mean={rho_T.mean():.8f}   std={rho_T.std():.6f}")
    print(f"  F(ρᵢ) : mean={Frho_T.mean():.8f} eV  std={Frho_T.std():.6f} eV")

    save_array(
        outdir / "eam_fields_300K.txt",
        "rho_i  F_rho_eV",
        np.column_stack([rho_T, Frho_T]),
    )

    counts, edges = np.histogram(rho_T, bins=100)
    centres = (edges[:-1] + edges[1:]) / 2.0
    save_array(
        outdir / "rho_histogram_300K.txt",
        "rho_centre  count",
        np.column_stack([centres, counts.astype(float)]),
    )

    sort_idx = np.argsort(rho_T)
    save_array(
        outdir / "embedding_function_trace_300K.txt",
        "rho_i  F_rho_eV  (sorted by rho — traces the F(rho) curve)",
        np.column_stack([rho_T[sort_idx], Frho_T[sort_idx]]),
    )
    print("  Saved embedding function trace (F vs ρ, sorted)")

    return {
        "rho_mean_0K":        float(rho_0K.mean()),
        "rho_std_300K":       float(rho_T.std()),
        "Frho_mean_0K":       float(Frho_0K.mean()),
        "embedding_fraction": emb_frac,
        "std_ratio_rho":      float(std_ratio_rho),
    }


# ── TEST 12: Self-diffusion coefficient ───────────────────────────────────────

def run_diffusion(
    calc_factory,
    outdir: Path,
    n_atoms: int,
    temperatures_K: tuple[float, ...] = (1400.0, 1500.0, 1600.0, 1700.0, 1800.0, 1900.0, 2000.0),
    npt_equil_steps: int = 15_000,
    nve_equil_steps: int = 10_000,
    prod_steps: int      = 100_000,
    timestep_fs: float   = 1.0,
    sample_interval: int = 20,
) -> dict:
    """Self-diffusion coefficient D via NVE Einstein relation.

    Protocol (MD best practice for EAM Cu)
    ----------------------------------------
    Cu solid-state self-diffusion (vacancy mechanism) has an activation energy
    of ~2 eV and a pre-exponential of ~6×10⁻⁶ m²/s; the resulting D is
    unmeasurably small (< 10⁻²⁰ m²/s) on MD timescales for T < T_melt.
    The correct approach for an EAM simulation is therefore:

      1. Run at liquid-state temperatures (T > T_melt_EAM ≈ 1300–1380 K).
         Liquid Cu has Ea ≈ 0.4–0.5 eV and D(1400 K) ≈ 4×10⁻⁹ m²/s,
         giving MSD ≈ 240 Å² over 100 ps — well within reach.

      2. **NPT pre-equilibration** to obtain the correct liquid density at
         each temperature.  A crude thermal expansion estimate fails badly
         for liquid Cu.  Using the wrong density biases D significantly
         (density directly controls the mean free path).

      3. Switch to NVE (velocity Verlet) for production to eliminate
         thermostat-induced artificial diffusion.

      4. Unwrap periodic-boundary jumps incrementally via the minimum-image
         convention; accumulate the displacement vector per atom.

      5. Fit D from the linear (diffusive) regime of MSD(t)/6t using only
         the last two-thirds of the trajectory (skipping the ballistic
         t→0 regime and any equilibration transient at the start).

    Comparison
    ----------
    The Arrhenius fit here samples the liquid regime.  The experimental
    reference E_a = 2.04 eV is for solid-state diffusion; this run will
    give E_a ≈ 0.4–0.5 eV (liquid Cu).  Both values are physically valid
    for their respective phases.
    """
    print_section("TEST 12  —  Self-Diffusion Coefficient  (NVE Einstein relation)")
    print("  NOTE: Running at T > T_melt (liquid Cu) where D is accessible on")
    print("        MD timescales.  Solid-state vacancy diffusion requires µs-ms")
    print("        simulations beyond the reach of direct MD.")

    # Smaller cell for speed — 8³ FCC = 2048 atoms is ample for liquid D.
    sc_size = _n_reps(n_atoms, min_rep=4, max_rep=8)
    atoms0  = bulk("Cu", "fcc", a=CU_EXP["a0_A"], cubic=True) * (sc_size, sc_size, sc_size)
    N       = len(atoms0)
    print(f"  Supercell : {sc_size}³ × {_FCC_BASIS} = {N} atoms   dt = {timestep_fs} fs")
    print(f"  Temperatures: {list(temperatures_K)} K")
    print(f"  NPT equil : {npt_equil_steps} steps = {npt_equil_steps * timestep_fs / 1000:.1f} ps")
    print(f"  NVE equil : {nve_equil_steps} steps = {nve_equil_steps * timestep_fs / 1000:.1f} ps")
    print(f"  NVE prod  : {prod_steps} steps = {prod_steps * timestep_fs / 1000:.1f} ps"
          f"  (sampled every {sample_interval} steps)")

    D_values:  list[float] = []
    inv_T_arr: list[float] = []

    for T in temperatures_K:
        print(f"\n  ── T = {T:.0f} K ─────────────────────────────────────")

        atoms = atoms0.copy()
        atoms.calc = calc_factory(use_cell_list=(N > 300))
        MaxwellBoltzmannDistribution(atoms, temperature_K=T,
                                     rng=np.random.default_rng(int(T)))
        Stationary(atoms)

        # ── Step 1: NPT equilibration to get correct liquid density ──────────
        # A thermal-expansion linear estimate is unreliable for liquid Cu.
        # We use ASE NPT with a short τ_p to converge the volume quickly.
        print(f"  NPT equilibration ({npt_equil_steps} steps) …")
        npt_dyn = _make_npt_dyn(atoms, timestep_fs, T, 0.0)
        npt_dyn.run(npt_equil_steps)
        V_npt = atoms.get_volume()
        a_liq = (V_npt / N * _FCC_BASIS) ** (1.0 / 3.0)
        T_npt = atoms.get_temperature()
        print(f"    After NPT: T={T_npt:.1f} K  V={V_npt:.2f} Å³  a_eff={a_liq:.4f} Å")

        # ── Step 2: Brief NVE equilibration to settle the microcanonical state
        print(f"  NVE equilibration ({nve_equil_steps} steps) …")
        nve = VelocityVerlet(atoms, timestep=timestep_fs * units.fs)
        nve.run(nve_equil_steps)

        # ── Step 3: NVE production with incremental MSD unwrapping ───────────
        cell_mat = np.array(atoms.get_cell())
        H_inv    = np.linalg.inv(cell_mat)

        prev_pos = atoms.get_positions().copy()
        cumdisp  = np.zeros_like(prev_pos)       # accumulated unwrapped displacement

        traj_msd: list[float] = []
        times_ps: list[float] = []
        step_idx = [0]                           # mutable counter for closure

        def _sample_msd() -> None:
            """Accumulate per-atom displacement under minimum-image convention."""
            nonlocal cumdisp          # FIX: augmented-assign (+=) makes Python treat
            #                         # cumdisp as local → UnboundLocalError without this.
            cur      = atoms.get_positions()
            raw      = cur - prev_pos
            frac     = raw @ H_inv
            frac    -= np.round(frac)            # minimum-image fractional jump
            disp     = frac @ cell_mat
            cumdisp += disp
            prev_pos[:] = cur                    # update in-place

            msd = float(np.mean(np.sum(cumdisp ** 2, axis=1)))
            traj_msd.append(msd)
            times_ps.append(step_idx[0] * sample_interval * timestep_fs / 1000.0)
            step_idx[0] += 1

        nve.attach(_sample_msd, interval=sample_interval)
        print(f"  NVE production ({prod_steps} steps) …")
        nve.run(prod_steps)

        msd_arr = np.array(traj_msd)
        t_arr   = np.array(times_ps)

        # ── Step 4: Fit D from diffusive regime (last 2/3 of trajectory) ─────
        # Skip ballistic (early) and any drift (late) regimes.
        fit_start = len(t_arr) // 3
        if fit_start < 5:
            slope_A2_ps = float("nan")
            D = float("nan")
        else:
            # D = lim MSD/(6t); slope of MSD vs t gives 6D
            slope_A2_ps = float(np.polyfit(t_arr[fit_start:], msd_arr[fit_start:], 1)[0])
            D = max(0.0, slope_A2_ps / 6.0 * 1e-8)  # Å²/ps → m²/s

        T_inst = atoms.get_temperature()
        print(f"  T_inst={T_inst:.1f} K  MSD_final={msd_arr[-1]:.2f} Å²"
              f"  slope={slope_A2_ps:.3f} Å²/ps  D={D:.3e} m²/s")

        D_values.append(D)
        inv_T_arr.append(1.0 / T)

        save_array(
            outdir / f"nve_msd_{int(T)}K.txt",
            "time_ps  msd_A2",
            np.column_stack([t_arr, msd_arr]),
        )

    D_np    = np.array(D_values)
    invT_np = np.array(inv_T_arr)
    valid   = np.isfinite(D_np) & (D_np > 0)

    Ea_eV = float("nan")
    D0    = float("nan")
    if valid.sum() >= 2:
        coeffs = np.polyfit(invT_np[valid], np.log(D_np[valid]), 1)
        Ea_eV  = float(-coeffs[0] * KB_EV)   # k_B has units eV/K
        D0     = float(np.exp(coeffs[1]))
        print_subsection("Arrhenius fit  (liquid Cu regime)")
        print(f"  ln D = ln D₀ − E_a / (k_B T)")
        print(f"  E_a = {Ea_eV:.3f} eV   D0 = {D0:.3e} m²/s")
        print(f"  [ref liquid Cu: E_a ≈ 0.45 eV  (solid-state exp: 2.04 eV — different regime)]")

    save_array(
        outdir / "diffusion_arrhenius.txt",
        "T_K  inv_T_K-1  D_m2s",
        np.column_stack([1.0 / invT_np, invT_np, D_np]),
    )

    return {
        "D_values":     D_values,
        "temperatures": list(temperatures_K),
        "Ea_eV":        Ea_eV,
        "D0_m2s":       D0,
    }


# ── TEST 13: Vibrational DOS via VACF ─────────────────────────────────────────

def run_vdos(
    calc_factory,
    outdir: Path,
    n_atoms: int,
    temperature_K: float = 300.0,
    equil_steps: int     = 5_000,
    prod_steps: int      = 20_000,
    timestep_fs: float   = 1.0,
    sample_interval: int = 5,
    max_sc_rep: int      = 8,
    friction_per_fs: float = 0.02,   # Langevin friction [1/fs] for NVT equil only
) -> dict:
    """Vibrational density of states g(ω) via velocity autocorrelation (VACF).

    Physics
    -------
        VACF(t) = <v(0)·v(t)> / <v(0)·v(0)>     (averaged over atoms & time origins)
        g(ω)   = |FT[VACF × Hann]|²             (windowed power spectrum)

    VACF via Wiener-Khinchin theorem (FFT-based, multiple time origins)
    -------------------------------------------------------------------
    The original single-time-origin estimator is noisy and biased.  The
    standard best practice uses *all possible time origins* simultaneously:

        VACF(τ) = (1 / (M − τ)) Σ_{t₀=0}^{M−1−τ}  ⟨v(t₀) · v(t₀+τ)⟩_atoms

    This is computed in O(M log M) via the Wiener-Khinchin theorem:
      1. Zero-pad velocity array to length 2M (avoids circular artefacts).
      2. FFT → power spectral density per degree of freedom.
      3. IFFT back → full autocorrelation.
      4. Divide by (M − τ) to correct for the decreasing number of pairs.
      5. Average over atoms and Cartesian directions; normalise by VACF(0).

    Memory
    ------
    The supercell is capped at ``max_sc_rep`` replications per axis.  Storing
    M × N × 3 float64 velocities for N = 8³×4 = 2048, M = prod/interval = 4000
    requires only ≈ 200 MB — safe on typical workstations.

    Parameters
    ----------
    max_sc_rep : int   — hard cap on replications per axis (default 8 → 2048 atoms)
    """
    print_section(f"TEST 13  —  Vibrational DOS via VACF  T = {temperature_K:.0f} K")

    # Cap supercell size to avoid memory issues with the velocity trajectory.
    sc_size      = _n_reps(n_atoms, min_rep=5, max_rep=max_sc_rep)
    atoms        = bulk("Cu", "fcc", a=CU_EXP["a0_A"], cubic=True) * (sc_size, sc_size, sc_size)
    N            = len(atoms)
    dt_sample_ps = sample_interval * timestep_fs / 1000.0  # ps between velocity frames
    n_frames     = prod_steps // sample_interval

    print(f"  Supercell : {sc_size}³ × {_FCC_BASIS} = {N} atoms   dt = {timestep_fs} fs")
    print(f"  NVT equil : {equil_steps} steps = {equil_steps * timestep_fs / 1000:.1f} ps")
    print(f"  NVE prod  : {prod_steps} steps = {prod_steps * timestep_fs / 1000:.1f} ps"
          f"  ({n_frames} VACF frames,  Δt_sample = {dt_sample_ps * 1000:.1f} fs)")

    mem_MB = n_frames * N * 3 * 8 / 1e6
    if mem_MB > 2_000:
        print(f"  [WARN] Estimated velocity storage {mem_MB:.0f} MB — consider reducing "
              f"prod_steps or increasing sample_interval.")

    atoms.calc = calc_factory(use_cell_list=(N > 300))
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K,
                                 rng=np.random.default_rng(42))
    Stationary(atoms)

    # ── Step 1: Langevin NVT equilibration ────────────────────────────────────
    nvt = Langevin(atoms,
                   timestep=timestep_fs * units.fs,
                   temperature_K=temperature_K,
                   friction=friction_per_fs / units.fs,
                   rng=np.random.default_rng(43))
    nvt.run(equil_steps)
    print(f"  NVT done.  T_inst = {atoms.get_temperature():.1f} K")

    # ── Step 2: NVE production — collect velocity trajectory ──────────────────
    nve      = VelocityVerlet(atoms, timestep=timestep_fs * units.fs)
    vel_traj: list[np.ndarray] = []

    def _sample_v() -> None:
        vel_traj.append(atoms.get_velocities().copy())   # (N, 3) Å/fs

    nve.attach(_sample_v, interval=sample_interval)
    nve.run(prod_steps)

    # ── Step 3: VACF via Wiener-Khinchin theorem (all time origins) ───────────
    vel      = np.stack(vel_traj, axis=0)   # (M, N, 3)
    M        = vel.shape[0]
    vel_flat = vel.reshape(M, N * 3)        # (M, K) where K = N*3

    # Zero-pad to next power of 2 ≥ 2M for linear (not circular) correlation
    M_pad = int(2 ** np.ceil(np.log2(2 * M)))
    Vf    = np.fft.rfft(vel_flat, n=M_pad, axis=0)   # (M_pad//2+1, K)

    # PSD: sum over degrees of freedom then IFFT → unnormalized autocorrelation
    psd_mean = np.mean(np.abs(Vf) ** 2, axis=1)      # (M_pad//2+1,)
    acf_raw  = np.fft.irfft(psd_mean, n=M_pad)[:M]   # (M,)

    # Correct for decreasing number of time-origin pairs at each lag τ
    n_pairs = M - np.arange(M, dtype=float)           # [M, M-1, ..., 1]
    acf_cor = acf_raw / n_pairs                        # (M,) normalised VACF (unnorm)

    if acf_cor[0] <= 0:
        print("  [WARN] VACF(0) = 0 — velocity data may be zero. Returning empty result.")
        return {"nu_peak_THz": float("nan"), "nu_debye_THz": float("nan"),
                "theta_D_K": float("nan"), "M_frames": M}

    vacf = acf_cor / acf_cor[0]   # VACF(0) = 1 by definition

    times_ps = np.arange(M) * dt_sample_ps
    save_array(
        outdir / f"vacf_{int(temperature_K)}K.txt",
        "time_ps  VACF_normalised",
        np.column_stack([times_ps, vacf]),
    )

    # ── Step 4: Power spectrum via Hann-windowed FFT ──────────────────────────
    window   = np.hanning(M)
    vdos_raw = np.abs(np.fft.rfft(vacf * window)) ** 2
    freqs    = np.fft.rfftfreq(M, d=dt_sample_ps)    # THz

    # Restrict to physically relevant window (Cu phonons < 10 THz)
    mask       = (freqs > 0.0) & (freqs <= 20.0)
    freqs_plot = freqs[mask]
    vdos_plot  = vdos_raw[mask]

    if freqs_plot.size == 0:
        print("  [WARN] No frequencies in [0, 20] THz — too few frames or large dt_sample.")
        return {"nu_peak_THz": float("nan"), "nu_debye_THz": float("nan"),
                "theta_D_K": float("nan"), "M_frames": M}

    area = float(_np_trapz(vdos_plot, freqs_plot))
    if area > 0:
        vdos_plot = vdos_plot / area   # normalise to unit area

    peak_idx = int(np.argmax(vdos_plot))
    nu_peak  = float(freqs_plot[peak_idx])

    # Debye frequency: frequency at which cumulative spectral weight reaches 99 %
    #
    # RATIONALE: The Debye frequency is the *cutoff* of the phonon spectrum —
    # the highest frequency at which phonon modes exist.  The 99th percentile of
    # the cumulative VDOS approximates this cutoff robustly without being
    # sensitive to the tail noise that a strict 100 % threshold would hit.
    #
    # Previously this threshold was 0.90 (90th percentile), which significantly
    # underestimates ν_D because most spectral weight concentrates in mid-band
    # peaks; the upper 10 % of the spectrum (the actual cutoff region) was
    # simply ignored.  For Cu EAM at 300 K the 90th percentile falls near
    # 5.5 THz, whereas the true phonon cutoff is ~7.5 THz (exp: 7.2 THz).
    # Changing to 0.99 corrects this systematic underestimate.
    _DEBYE_PERCENTILE = 0.99
    dnu     = np.gradient(freqs_plot)
    cumspec = np.cumsum(vdos_plot * dnu)
    cs_end  = float(cumspec[-1])
    if cs_end <= 0:
        print("  [WARN] Cumulative spectrum integrates to zero — VDOS is flat/empty.")
        nu_debye = nu_peak   # fallback
    else:
        cumspec  /= cs_end
        debye_idx = int(np.searchsorted(cumspec, _DEBYE_PERCENTILE))
        debye_idx  = min(debye_idx, len(freqs_plot) - 1)
        nu_debye   = float(freqs_plot[debye_idx])

    theta_D = HBAR_EV_S * (nu_debye * 1e12) * 2.0 * np.pi / KB_EV

    print_subsection("VACF → VDOS results")
    print(f"  VACF frames         : {M}"
          f"   spectral resolution: {freqs[1]:.4f} THz")
    print(f"  Peak VDOS frequency : {nu_peak:.3f} THz")
    print(compare("Debye frequency ν_D [THz]", nu_debye,
                  CU_EXP["nu_Debye_THz"], "THz", tol_pct=20.0))
    print(compare("Debye temperature Θ_D [K]", theta_D,
                  CU_EXP["theta_D_K"], "K", tol_pct=20.0))

    save_array(
        outdir / f"vdos_{int(temperature_K)}K.txt",
        "freq_THz  vdos_normalised",
        np.column_stack([freqs_plot, vdos_plot]),
    )

    return {
        "nu_peak_THz":  nu_peak,
        "nu_debye_THz": nu_debye,
        "theta_D_K":    theta_D,
        "M_frames":     M,
    }

# ── TEST 14: Intrinsic Stacking Fault Energy (ISFE) ───────────────────────────

def _mic_min_perp_widths(cell: np.ndarray) -> tuple[float, float, float]:
    """Return the three perpendicular widths of a parallelpiped cell.

    For each pair of lattice vectors (a_i, a_j), the perpendicular width in
    the direction of the third vector a_k is:

        w_k = Volume / |a_i × a_j|

    The Minimum Image Convention (MIC) is satisfied iff all three widths
    exceed 2 × r_cut.

    Parameters
    ----------
    cell : (3, 3) array — row vectors of the simulation cell [Å]

    Returns
    -------
    (w_a, w_b, w_c) : perpendicular widths along a, b, c [Å]
    """
    vol = abs(float(np.linalg.det(cell)))
    w_c = vol / float(np.linalg.norm(np.cross(cell[0], cell[1])))  # ⊥ to ab-plane
    w_b = vol / float(np.linalg.norm(np.cross(cell[0], cell[2])))  # ⊥ to ac-plane
    w_a = vol / float(np.linalg.norm(np.cross(cell[1], cell[2])))  # ⊥ to bc-plane
    return w_a, w_b, w_c


def _build_fcc111_bulk(nx: int, ny: int, n_layers: int, a0: float):
    """Build a fully 3D-periodic FCC [111] bulk supercell."""
    try:
        atoms = fcc111(
            "Cu",
            size=(nx, ny, n_layers),
            a=a0,
            orthogonal=False,
            periodic=True,      # 3D periodic — no vacuum
        )
    except TypeError:
        # Older ASE versions don't have the periodic keyword
        atoms = fcc111("Cu", size=(nx, ny, n_layers), a=a0, orthogonal=False)
        cell = atoms.get_cell()
        cell[2] = [0.0, 0.0, n_layers * a0 / np.sqrt(3.0)]
        atoms.set_cell(cell, scale_atoms=False)
    atoms.set_pbc([True, True, True])
    return atoms


def run_stacking_fault(
    calc_factory,
    outdir: Path,
    nx: int = 4,
    ny: int = 4,
    n_layers: int = 18,
    a0: Optional[float] = None,
) -> dict:
    """TEST 14 — Intrinsic Stacking Fault Energy (ISFE) γ_sf.

    Procedure (Tilted Cell Method)
    ---------
    1. Build a fully 3D periodic bulk FCC [111] supercell (no vacuum).
       n_layers MUST be a multiple of 3 to maintain ABC stacking periodicity.
    2. **MIC guard**: verify that all perpendicular cell widths exceed 2×r_cut.
       If the in-plane width is too small (common for nx=ny=4 with EAM Cu where
       the 60° rhombic cell gives only ~8.85 Å < 2×5.5 = 11 Å), nx and ny are
       automatically increased to the smallest integer that satisfies MIC.
       Without this guard the GPU/cell-list path silently introduces phantom
       neighbours, causing a spurious energy jump after relaxation.
    3. Compute E_perfect for the defect-free bulk cell.
    4. Apply a rigid Shockley partial shift b_p = (a₁+a₂)/3 to atoms in the
       top half of the cell.
    5. *Tilted Cell*: Add b_p to the cell's 3rd vector (C-vector) so that the
       Z-PBC seam is healed, leaving exactly ONE stacking fault at the mid-plane.
    6. γ_unrelaxed = (E_fault_unrelaxed − E_perfect) / A_fault.
    7. z-only FIRE relaxation → γ_relaxed.
       Physical sanity: E_relaxed must be ≤ E_unrelaxed; if it is not, the
       relaxation is discarded as unphysical (residual MIC artefact or
       optimiser instability) and γ_unrelaxed is reported instead.
    """
    print_section("TEST 14  —  Intrinsic Stacking Fault Energy (Tilted Cell)")

    if a0 is None:
        a0 = CU_EXP["a0_A"]

    # ── n_layers must be a multiple of 3 (ABC stacking) ───────────────────────
    if n_layers % 3 != 0:
        n_layers = (n_layers // 3) * 3
        if n_layers < 12:
            n_layers = 12
        print(f"  [INFO] Adjusted n_layers to {n_layers} (must be multiple of 3).")

    # ── Obtain EAM cutoff radius for MIC check ────────────────────────────────
    # A single probe call retrieves the potential cutoff without building atoms.
    _EAM_CUTOFF_FALLBACK = 5.5  # Å — conservative EAM Cu default
    try:
        _probe_engine = calc_factory()._engine
        r_cut = float(_probe_engine.cutoff)
    except Exception:
        r_cut = _EAM_CUTOFF_FALLBACK
        print(f"  [INFO] Could not read cutoff from engine; using fallback {r_cut} Å.")
    mic_required = 2.0 * r_cut   # MIC: all perpendicular widths > 2×r_cut

    # ── MIC guard: auto-scale nx / ny if in-plane width is too small ──────────
    # FCC [111] non-orthogonal cell: in-plane primitive vector = a0/√2 (NN dist).
    # The 2D cell is a 60° rhombus; its perpendicular height is
    #   h = n × (a0/√2) × sin(60°) = n × a0 × √6/4
    # We require h > mic_required for n = min(nx, ny).
    a_2d  = a0 / np.sqrt(2.0)                          # FCC [111] in-plane NN distance
    sin60 = np.sin(np.radians(60.0))                   # = √3/2
    h_per_rep = a_2d * sin60                           # perpendicular height per rep

    n_min_req = int(np.ceil(mic_required / h_per_rep))  # minimum reps to satisfy MIC
    if min(nx, ny) < n_min_req:
        print(
            f"  [WARN] MIC violation detected!\n"
            f"         In-plane perpendicular width at nx=ny={min(nx,ny)}: "
            f"{min(nx,ny)*h_per_rep:.2f} Å  <  2×r_cut = {mic_required:.2f} Å.\n"
            f"         GPU/cell-list paths may introduce phantom neighbours,\n"
            f"         causing spurious energy jumps after relaxation.\n"
            f"         Auto-scaling nx=ny: {min(nx,ny)} → {n_min_req}."
        )
        nx = ny = n_min_req

    # ── Build perfect bulk ────────────────────────────────────────────────────
    atoms_perfect = _build_fcc111_bulk(nx, ny, n_layers, a0)

    n_fault = len(atoms_perfect)
    cell    = atoms_perfect.get_cell().array

    # ── Verify MIC on the actual cell (catches residual issues in tilted cells) ─
    w_a, w_b, w_c = _mic_min_perp_widths(cell)
    mic_ok = all(w > mic_required for w in (w_a, w_b, w_c))
    if not mic_ok:
        print(
            f"  [WARN] Post-build MIC check FAILED for the perfect cell:\n"
            f"         w_a={w_a:.2f} Å  w_b={w_b:.2f} Å  w_c={w_c:.2f} Å  "
            f"(need > {mic_required:.2f} Å).\n"
            f"         Results may be unreliable; consider increasing n_layers."
        )
    else:
        print(
            f"  [INFO] MIC check passed: w_a={w_a:.2f} Å  w_b={w_b:.2f} Å  "
            f"w_c={w_c:.2f} Å  (need > {mic_required:.2f} Å, r_cut={r_cut:.2f} Å)."
        )

    # Fault area = |a₁ × a₂|  (in-plane cross product)
    A_fault = float(np.linalg.norm(np.cross(cell[0], cell[1])))

    # Shockley partial Burgers vector  b_p = (a₁/nx + a₂/ny) / 3
    b_partial    = (cell[0] / nx + cell[1] / ny) / 3.0
    b_partial[2] = 0.0

    print(f"  Supercell : {nx}×{ny}×{n_layers} [111] layers  → {n_fault} atoms")
    print(f"  Method    : Fully 3D Periodic Tilted Cell (No Vacuum)")
    print(f"  |b_p|     : {np.linalg.norm(b_partial):.4f} Å")
    print(f"  A_fault   : {A_fault:.2f} Å²")

    # ── Perfect crystal energy ────────────────────────────────────────────────
    atoms_perfect.calc = calc_factory(use_cell_list=(n_fault > 300))
    E_perf = float(atoms_perfect.get_potential_energy())
    print(f"  E_perfect : {E_perf:.4f} eV (bulk)  ({E_perf/n_fault:.4f} eV/atom)")

    # ── Rigid shift & Cell Tilt (unrelaxed fault) ─────────────────────────────
    atoms_fault = atoms_perfect.copy()

    z_positions = atoms_fault.positions[:, 2]
    z_mid       = float(z_positions.mean())

    # Shift top half by Shockley partial in xy
    mask = z_positions > z_mid
    atoms_fault.positions[mask, 0] += b_partial[0]
    atoms_fault.positions[mask, 1] += b_partial[1]

    # Tilt C-vector to heal PBC seam at z = L_z (Tilted Cell method)
    fault_cell = atoms_fault.get_cell().copy()
    fault_cell[2, 0] += b_partial[0]
    fault_cell[2, 1] += b_partial[1]
    atoms_fault.set_cell(fault_cell, scale_atoms=False)

    # ── Verify MIC on the tilted fault cell ───────────────────────────────────
    fc_arr = fault_cell.array if hasattr(fault_cell, "array") else np.array(fault_cell)
    fw_a, fw_b, fw_c = _mic_min_perp_widths(fc_arr)
    if not all(w > mic_required for w in (fw_a, fw_b, fw_c)):
        print(
            f"  [WARN] MIC check FAILED on tilted fault cell:\n"
            f"         w_a={fw_a:.2f} Å  w_b={fw_b:.2f} Å  w_c={fw_c:.2f} Å.\n"
            f"         Relaxation is disabled to prevent phantom-neighbour artefacts."
        )
        _tilt_mic_ok = False
    else:
        _tilt_mic_ok = True

    atoms_fault.calc = calc_factory(use_cell_list=(n_fault > 300))
    E_fault_unrelaxed = float(atoms_fault.get_potential_energy())

    EV_PER_A2_TO_MJ_M2 = 16021.766
    gamma_unrelaxed = (E_fault_unrelaxed - E_perf) / A_fault * EV_PER_A2_TO_MJ_M2

    # ── z-only FIRE relaxation ────────────────────────────────────────────────
    # Guard conditions:
    #   (a) both MIC checks passed — phantom neighbours would corrupt gradients
    #   (b) FixedLine constraint and FIRE are importable
    E_fault_relaxed = E_fault_unrelaxed
    gamma_relaxed   = gamma_unrelaxed
    relax_ok        = False

    if mic_ok and _tilt_mic_ok:
        try:
            from ase.constraints import FixedLine
            from ase.optimize import FIRE as _FIRE

            atoms_fault.set_constraint(
                FixedLine(list(range(n_fault)), [0.0, 0.0, 1.0])
            )
            opt_fault = _FIRE(atoms_fault, logfile=None)
            opt_fault.run(fmax=0.01, steps=300)

            E_relaxed_candidate = float(atoms_fault.get_potential_energy())

            # Physical sanity check: relaxation must lower (or not raise) energy.
            # If E_relaxed > E_unrelaxed the optimiser encountered phantom
            # neighbours or numerical instability — discard the result.
            _tol_ev = 1e-4  # allow ≤ 0.1 meV numerical noise
            if E_relaxed_candidate > E_fault_unrelaxed + _tol_ev:
                print(
                    f"  [WARN] z-relaxation raised energy by "
                    f"{(E_relaxed_candidate - E_fault_unrelaxed)*1000:.2f} meV "
                    f"— discarding as unphysical (phantom-neighbour artefact).\n"
                    f"         Reporting γ_sf (unrelaxed) instead."
                )
                # relax_ok remains False; E_fault_relaxed stays = unrelaxed
            else:
                E_fault_relaxed = E_relaxed_candidate
                gamma_relaxed   = (E_fault_relaxed - E_perf) / A_fault * EV_PER_A2_TO_MJ_M2
                relax_ok        = True

        except Exception as exc:
            print(f"  [WARN] Fault slab z-relaxation skipped ({exc}).")
    else:
        print("  [INFO] z-relaxation skipped: MIC not satisfied on fault cell.")

    print_subsection("Stacking fault energy results")
    print(f"  γ_sf (unrelaxed) = {gamma_unrelaxed:.2f} mJ/m²")
    if relax_ok:
        print(f"  γ_sf (z-relaxed) = {gamma_relaxed:.2f} mJ/m²")
        print(compare("ISFE γ_sf (z-relaxed) [mJ/m²]",
                      gamma_relaxed, CU_EXP["gamma_sf_mJ_m2"], "mJ/m²", tol_pct=15.0))
    else:
        print(compare("ISFE γ_sf (unrelaxed) [mJ/m²]",
                      gamma_unrelaxed, CU_EXP["gamma_sf_mJ_m2"], "mJ/m²", tol_pct=15.0))

    save_array(
        outdir / "stacking_fault_energies.txt",
        "label  E_eV  gamma_unrelaxed_mJ_m2  gamma_relaxed_mJ_m2",
        np.array([[0.0, E_perf,            0.0,             0.0],
                  [1.0, E_fault_unrelaxed, gamma_unrelaxed, gamma_unrelaxed],
                  [2.0, E_fault_relaxed,   gamma_unrelaxed, gamma_relaxed]]),
    )

    return {
        "gamma_sf_unrelaxed_mJ_m2": gamma_unrelaxed,
        "gamma_sf_relaxed_mJ_m2":   gamma_relaxed,
        "A_fault_A2":               A_fault,
        "E_perfect_eV":             E_perf,
        "E_relaxed_eV":             E_fault_relaxed,
        "relax_ok":                 relax_ok,
        "mic_ok":                   mic_ok and _tilt_mic_ok,
        "nx_used":                  nx,
        "ny_used":                  ny,
        "r_cut_A":                  r_cut,
    }
    
# ── TEST 15: Heat capacity C_V / C_P from NVT potential-energy fluctuations ───

def run_heat_capacity(
    calc_factory,
    outdir: Path,
    n_atoms: int,
    temperature_K: float = 300.0,
    pressure_GPa: float = 0.0,   # kept for API compat; used only for C_P correction
    equil_steps: int = 5_000,
    prod_steps: int = 50_000,
    timestep_fs: float = 2.0,
    sample_interval: int = 10,
    quiet: bool = False,
) -> dict:
    """TEST 15 — Heat capacity C_V / C_P from NVT potential-energy fluctuations.

    Theory
    ------
    In the canonical (NVT) ensemble the isochoric heat capacity satisfies
    (Allen & Tildesley §2.5; Lebowitz, Percus & Verlet 1967):

        C_V = (3/2)·N·k_B + Var(E_pot) / (k_B · T²)    [eV/K, whole cell]

    The first term is the exact equipartition kinetic contribution; the second
    captures anharmonic potential-energy fluctuations.  Converted to SI:

        C_V [J/(mol·K)] = C_V_cell [eV/K] × e × N_A / N_atoms

    C_P follows from the standard thermodynamic identity:
        C_P = C_V + T · V_m · α² · B₀

    Why NVT instead of NPT + Var(H)?
    ---------------------------------
    The NPT Var(H) formula is theoretically exact but requires the barostat to
    be fully converged.  With τ_baro ≈ 2 ps the barostat oscillation period is
    ~12.6 ps; only ~8 independent cycles fit in a 100 ps window, so Var(H) is
    inflated by ~18× (observed 608 eV² vs expected 33 eV²), giving a spurious
    C_P ≈ 460 J/(mol·K).  The NVT Var(E_pot) route avoids barostat coupling
    entirely and converges with O(10³) samples.

    Classical Dulong-Petit limit: C_V → 3R = 24.943 J/(mol·K) as T → ∞.
    Classical MD is always at the Dulong-Petit limit; quantum (ZPE) corrections
    that reduce exp C_P to 24.44 at 298 K are absent.  A ~+2% overshot is
    therefore expected and physically correct for this method.

    Experimental reference for Cu
    ──────────────────────────────
    C_P(298 K) = 24.44 J/(mol·K)  ≈ 2.94 R
      • Chase, NIST-JANAF Tables 4th ed. (1998)  [Ref 17]
      • NIST WebBook "Cu heat capacity"           [Ref 8]

    Parameters
    ----------
    sample_interval : record E_pot every this many steps (default 10)
    pressure_GPa    : used only for the C_P = C_V + T·Vm·α²·B₀ correction;
                      the MD run itself is NVT (no barostat)
    """
    print_section(
        f"TEST 15  —  Heat capacity C_P  T={temperature_K:.0f} K"
    )

    sc_size = max(2, round((n_atoms / _FCC_BASIS) ** (1.0 / 3.0)))
    atoms   = bulk("Cu", "fcc", a=CU_EXP["a0_A"], cubic=True) * (sc_size, sc_size, sc_size)
    n       = len(atoms)
    print(f"  Supercell  : {sc_size}³ × 4 = {n} atoms")
    print(f"  T_target   : {temperature_K:.0f} K   Ensemble : NVT (Langevin)")
    print(f"  Equil/Prod : {equil_steps} / {prod_steps} steps  (×{timestep_fs} fs)")

    atoms.calc = calc_factory(use_cell_list=(n > 300))
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K,
                                 rng=np.random.default_rng(42))
    Stationary(atoms)

    # NVT Langevin — no barostat, so Var(E_pot) reflects only thermal fluctuations
    dyn = Langevin(
        atoms,
        timestep=timestep_fs * units.fs,
        temperature_K=temperature_K,
        friction=0.01 / units.fs,   # 100 fs coupling — gentle, minimal T-bias
        rng=np.random.default_rng(43),
    )
    dyn.run(equil_steps)

    epots:    list[float] = []
    temps_cp: list[float] = []
    n_records = 0
    log_every = max(1, (prod_steps // sample_interval) // 20)

    def _record_E() -> None:
        nonlocal n_records
        E_pot  = atoms.get_potential_energy()
        T_inst = atoms.get_temperature()
        epots.append(E_pot)
        temps_cp.append(T_inst)
        n_records += 1
        if not quiet and n_records % log_every == 0:
            print(f"    sample {n_records:6d}  T={T_inst:.1f} K  Epot={E_pot:.4f} eV")

    dyn.attach(_record_E, interval=sample_interval)
    dyn.run(prod_steps)

    Ep_arr = np.array(epots)
    T_arr  = np.array(temps_cp)

    # Discard first 20 % for stationarity
    n_trim  = max(1, len(Ep_arr) // 5)
    Ep_stat = Ep_arr[n_trim:]
    T_mean  = float(T_arr[n_trim:].mean())

    Epot_var = float(np.var(Ep_stat))   # eV²

    # C_V [eV/K, whole cell] = (3/2)·N·k_B + Var(E_pot) / (k_B · T²)
    C_V_cell  = (1.5 * n * KB_EV) + Epot_var / (KB_EV * T_mean ** 2)
    C_V_J_mol = C_V_cell * EV_TO_J * N_A / n

    # C_P = C_V + T·V_m·α²·B₀   (standard thermodynamic correction)
    # α = 16.6×10⁻⁶ K⁻¹ (exp),  B₀ = 140 GPa,  V_m = M/ρ = 63.546e-3/8960 m³/mol
    alpha_K   = CU_EXP["alpha_1perK"]
    B0_Pa     = CU_EXP["B0_GPa"] * 1e9
    V_molar   = 63.546e-3 / 8960.0          # m³/mol
    dCP       = T_mean * V_molar * alpha_K ** 2 * B0_Pa   # J/(mol·K)
    C_P_J_mol = C_V_J_mol + dCP

    C_P_per_R = C_P_J_mol / R_J_MOL_K

    print_subsection("Heat capacity results")
    print(f"  Samples used   : {len(Ep_stat)}  (after 20% trim)")
    print(f"  T_mean         : {T_mean:.2f} K")
    print(f"  Var(E_pot)     : {Epot_var:.4e} eV²  (expected for 3R: "
          f"{1.5 * n * (KB_EV * T_mean)**2:.4e} eV²)")
    print(f"  C_V            : {C_V_J_mol:.4f} J/(mol·K)"
          f"  (Dulong-Petit 3R = {3*R_J_MOL_K:.4f})")
    print(f"  C_P-C_V corr.  : {dCP:.4f} J/(mol·K)")
    print(f"  C_P            : {C_P_J_mol:.4f} J/(mol·K)")
    print(f"  C_P / R        : {C_P_per_R:.4f}   (classical limit = 3.000)")
    print(compare("C_P [J/(mol·K)]",
                  C_P_J_mol, CU_EXP["C_P_300K_J_mol_K"], "J/(mol·K)", tol_pct=10.0))
    print(f"  [INFO] Classical MD lacks ZPE: C_P → 3R=24.94 at all T (≈+2% vs exp 24.44)")

    # Save E_pot time series
    times_ps = np.arange(len(Ep_arr)) * sample_interval * timestep_fs / 1000.0
    save_array(
        outdir / f"heat_capacity_{int(temperature_K)}K.txt",
        "time_ps   Epot_eV   T_K",
        np.column_stack([times_ps, Ep_arr, T_arr]),
    )

    return {
        "C_P_J_mol_K":  C_P_J_mol,
        "C_V_J_mol_K":  C_V_J_mol,
        "C_P_per_R":    C_P_per_R,
        "Epot_var_eV2": Epot_var,
        "T_mean_K":     T_mean,
        "n_samples":    len(Ep_stat),
    }


# ── TEST 16: Helmholtz free energy A & entropy S — quasiharmonic Debye model ──

def run_free_energy(
    calc_factory,
    outdir: Path,
    n_atoms: int,
    temperature_K: float = 300.0,
    equil_steps: int = 5_000,
    prod_steps: int = 30_000,
    timestep_fs: float = 2.0,
    sample_interval: int = 10,
    quiet: bool = False,
    nu_debye_THz: float | None = None,   # override: use TEST 13 result if passed
) -> dict:
    """TEST 16 — Helmholtz free energy A and entropy S via quasiharmonic Debye model.

    Method
    ------
    Combines a short NVT run (measures <U> = <E_pot> + <E_kin>) with the
    **classical Debye model** to compute vibrational thermodynamic functions:

    Vibrational free energy (classical Debye, T ≥ Θ_D limit):
        A_vib(T) = 3N·k_B·T · [ln(hν_D / k_B·T) − 1]        [eV, whole cell]

    Vibrational entropy:
        S_vib(T) = 3N·k_B · [2 − ln(hν_D / k_B·T)]           [eV/K]

    Cross-check: A_vib = <U_vib> − T·S_vib where <U_vib> = 3Nk_BT (equi-partition).

    Total Helmholtz free energy (relative to ideal lattice):
        A(T) = U₀ + A_vib(T)
    where U₀ = EAM cohesive energy (E_pot per atom × N, taken from NVT at T→0
    via the static single-point, or approximated as <E_pot(T)> − 3Nk_BT).

    ν_D is taken from the TEST 13 result (``nu_debye_THz`` argument) when
    available; otherwise falls back to the experimental value 7.2 THz.

    Known limitations
    -----------------
    *Classical* Debye model is used throughout, consistent with classical EAM MD.
    Quantum (ZPE) effects are absent, so:
      • S_vib is overestimated by ~35–45 % vs NIST-JANAF at T ≈ Θ_D (300 K).
      • The overestimate shrinks for T >> Θ_D.
    This is expected and documented — it is not a code defect.

    Experimental reference (NIST-JANAF 4th ed., Ref 17):
        S°(298.15 K) = 33.150 J/(mol·K)

    Parameters
    ----------
    nu_debye_THz : float or None
        Debye cutoff frequency [THz].  Pass ``results["vdos"]["nu_debye_THz"]``
        from TEST 13 to use the EAM-computed value.  Default: None → uses
        ``CU_EXP["nu_Debye_THz"]`` = 7.2 THz.
    """
    print_section(
        f"TEST 16  —  Free energy A & entropy S  T={temperature_K:.0f} K"
    )

    nu_D = nu_debye_THz if nu_debye_THz is not None else CU_EXP["nu_Debye_THz"]
    nu_D_source = "TEST 13 (EAM VDOS)" if nu_debye_THz is not None else "exp (7.2 THz)"
    print(f"  ν_D source : {nu_D_source}  →  ν_D = {nu_D:.3f} THz")

    # ── Short NVT run to measure <E_pot> and <E_kin> ──────────────────────────
    sc_size = max(2, round((n_atoms / _FCC_BASIS) ** (1.0 / 3.0)))
    atoms   = bulk("Cu", "fcc", a=CU_EXP["a0_A"], cubic=True) * (sc_size, sc_size, sc_size)
    n       = len(atoms)
    print(f"  Supercell  : {sc_size}³ × 4 = {n} atoms")
    print(f"  T_target   : {temperature_K:.0f} K   Ensemble : NVT (Langevin)")
    print(f"  Equil/Prod : {equil_steps} / {prod_steps} steps  (×{timestep_fs} fs)")

    atoms.calc = calc_factory(use_cell_list=(n > 300))
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K,
                                 rng=np.random.default_rng(44))
    Stationary(atoms)

    dyn = Langevin(
        atoms,
        timestep=timestep_fs * units.fs,
        temperature_K=temperature_K,
        friction=0.01 / units.fs,
        rng=np.random.default_rng(45),
    )
    dyn.run(equil_steps)

    epots: list[float] = []
    ekins: list[float] = []
    temps: list[float] = []
    n_rec = 0
    log_every = max(1, (prod_steps // sample_interval) // 15)

    def _record() -> None:
        nonlocal n_rec
        ep = atoms.get_potential_energy()
        ek = atoms.get_kinetic_energy()
        T_inst = atoms.get_temperature()
        epots.append(ep)
        ekins.append(ek)
        temps.append(T_inst)
        n_rec += 1
        if not quiet and n_rec % log_every == 0:
            print(f"    sample {n_rec:6d}  T={T_inst:.1f} K"
                  f"  Epot={ep:.4f} eV  Ekin={ek:.4f} eV")

    dyn.attach(_record, interval=sample_interval)
    dyn.run(prod_steps)

    # Discard first 20 %
    n_trim  = max(1, len(epots) // 5)
    Epot_mean = float(np.mean(epots[n_trim:]))  # eV
    Ekin_mean = float(np.mean(ekins[n_trim:]))  # eV  → should be ≈ (3/2)Nk_BT
    T_mean    = float(np.mean(temps[n_trim:]))  # K

    U_total   = Epot_mean + Ekin_mean           # <U> = <E_pot> + <E_kin>  [eV]

    # Equipartition check: Ekin should be ≈ (3/2)·N·k_B·T
    Ekin_equip = 1.5 * n * KB_EV * T_mean
    Ekin_ratio = Ekin_mean / Ekin_equip

    # ── Classical Debye vibrational free energy ───────────────────────────────
    # hν_D in eV
    hnu_D_eV = HBAR_EV_S * 2.0 * np.pi * nu_D * 1.0e12   # = h·ν_D

    kT = KB_EV * T_mean   # eV

    # x = hν_D / (k_B·T)  — dimensionless Debye ratio
    x = hnu_D_eV / kT
    print(f"\n  hν_D       : {hnu_D_eV*1000:.3f} meV   k_BT : {kT*1000:.3f} meV"
          f"   x = hν_D/k_BT = {x:.4f}"
          f"{'  (classical limit: x<<1)' if x < 0.5 else ''}")

    # A_vib [eV, whole cell] — classical Debye
    A_vib_eV = 3.0 * n * kT * (np.log(hnu_D_eV / kT) - 1.0)

    # S_vib [eV/K, whole cell] = (U_vib - A_vib)/T = 3Nk_B·[2 - ln(hν_D/k_BT)]
    S_vib_eV_K = 3.0 * n * KB_EV * (2.0 - np.log(hnu_D_eV / kT))

    # Cross-check: A_vib = U_vib - T·S_vib  where <U_vib> = 3Nk_BT (equi-partition)
    U_vib_equip  = 3.0 * n * kT                           # eV
    A_vib_check  = U_vib_equip - T_mean * S_vib_eV_K      # should equal A_vib_eV

    # Convert per-mole
    S_J_mol = S_vib_eV_K * EV_TO_J * N_A / n             # J/(mol·K)
    A_vib_J_mol = A_vib_eV * EV_TO_J * N_A / n           # J/mol

    # Total A = U₀ + A_vib   where U₀ = Epot at 0 K ≈ <E_pot(T)> − 3Nk_BT
    U0_eV      = Epot_mean - 3.0 * n * kT                # 0-K reference [eV]
    U0_eV_atom = U0_eV / n

    A_total_eV      = U0_eV + A_vib_eV                   # whole cell [eV]
    A_total_J_mol   = A_total_eV * EV_TO_J * N_A / n     # J/mol

    print_subsection("Free energy results")
    print(f"  Samples used      : {len(epots) - n_trim}  (after 20% trim)")
    print(f"  T_mean            : {T_mean:.2f} K")
    print(f"  <E_pot>           : {Epot_mean:.4f} eV")
    print(f"  <E_kin>           : {Ekin_mean:.4f} eV"
          f"  (equi-partition: {Ekin_equip:.4f} eV,"
          f" ratio {Ekin_ratio:.4f})")
    print(f"  U₀ (0-K E_pot)    : {U0_eV:.4f} eV"
          f"  ({U0_eV_atom:.4f} eV/atom,"
          f" exp E_coh = {CU_EXP['E_coh_eV']:.4f} eV/atom)")
    print()
    print(f"  A_vib(T)          : {A_vib_eV:.4f} eV  "
          f"({A_vib_J_mol/1000:.4f} kJ/mol)")
    print(f"  A_vib cross-check : {A_vib_check:.4f} eV"
          f"  (Δ = {abs(A_vib_eV - A_vib_check):.2e} eV)")
    print(f"  A_total           : {A_total_eV:.4f} eV"
          f"  ({A_total_J_mol/1000:.4f} kJ/mol)")
    print()
    print(compare("S_vib [J/(mol·K)]",
                  S_J_mol, CU_EXP["S_298K_J_mol_K"], "J/(mol·K)", tol_pct=50.0))
    print(f"  [INFO] Classical Debye overestimates S by ~35-45% at T≈Θ_D (ZPE absent)")
    print(f"         Expected overestimate at {T_mean:.0f} K: "
          f"{S_J_mol:.2f} vs exp {CU_EXP['S_298K_J_mol_K']:.2f} J/(mol·K)"
          f" → ratio {S_J_mol/CU_EXP['S_298K_J_mol_K']:.2f}×")

    # Save thermodynamic functions
    T_range  = np.linspace(100.0, 1200.0, 200)
    kT_range = KB_EV * T_range
    x_range  = hnu_D_eV / kT_range
    A_range  = 3.0 * n * kT_range * (np.log(hnu_D_eV / kT_range) - 1.0)
    S_range  = 3.0 * n * KB_EV * (2.0 - np.log(hnu_D_eV / kT_range))
    U_range  = 3.0 * n * kT_range   # classical: U_vib = 3Nk_BT

    A_range_J_mol = A_range * EV_TO_J * N_A / n
    S_range_J_mol = S_range * EV_TO_J * N_A / n

    save_array(
        outdir / "free_energy.txt",
        "T_K   A_vib_J_mol   S_vib_J_mol_K   U_vib_J_mol",
        np.column_stack([
            T_range,
            A_range_J_mol,
            S_range_J_mol,
            U_range * EV_TO_J * N_A / n,
        ]),
    )

    return {
        "T_mean_K":          T_mean,
        "nu_debye_THz":      nu_D,
        "hnu_D_eV":          hnu_D_eV,
        "x_debye":           x,
        "U0_eV_per_atom":    U0_eV_atom,
        "A_vib_J_mol":       A_vib_J_mol,
        "A_total_J_mol":     A_total_J_mol,
        "S_vib_J_mol_K":     S_J_mol,
        "S_exp_J_mol_K":     CU_EXP["S_298K_J_mol_K"],
        "S_ratio":           S_J_mol / CU_EXP["S_298K_J_mol_K"],
        "Ekin_equip_ratio":  Ekin_ratio,
        "n_samples":         len(epots) - n_trim,
    }




# ── TEST 17 helpers: quantum Debye integrals ──────────────────────────────────

def _debye_integral_F(u_max: float) -> float:
    """Return ∫₀^{u_max} u² ln(1 − e^{−u}) du  (always ≤ 0).

    This is the key integral in the quantum Debye Helmholtz free energy:

        F_vib / N = (9/8) k_B θ_D  +  3 k_B T (1/x³) ∫₀^x u² ln(1−e^{−u}) du

    where x = θ_D / T.  The integrand is smooth on (0, u_max]; u² → 0 guards
    the lower endpoint so no special treatment is needed.
    """
    from scipy.integrate import quad

    if u_max < 1e-6:
        return 0.0

    def _integrand(u):
        # u² ln(1-e^{-u}): both factors are well-defined for u > 0
        return u * u * np.log(1.0 - np.exp(-u))

    val, _ = quad(_integrand, 0.0, u_max, limit=200, epsabs=1e-12, epsrel=1e-10)
    return float(val)


def _debye_function_D3(x: float) -> float:
    """Debye function  D₃(x) = (3/x³) ∫₀^x t³ / (e^t − 1) dt.

    Limiting values:  D₃(0) = 1,  D₃(∞) → 0.
    Used in the Debye entropy formula S = N k_B [4 D₃(θ/T) − 3 ln(1−e^{−θ/T})].
    """
    from scipy.integrate import quad

    if x < 1e-6:
        return 1.0   # lim_{x→0} D₃(x) = 1

    def _integrand(t):
        if t < 1e-9:
            return t * t   # t³/(e^t−1) → t² as t → 0
        return t ** 3 / (np.exp(t) - 1.0)

    val, _ = quad(_integrand, 0.0, x, limit=200, epsabs=1e-12, epsrel=1e-10)
    return 3.0 / (x ** 3) * float(val)


def _quantum_debye_F_ev_per_atom(T: float, theta_D: float) -> float:
    """Quantum Debye Helmholtz free energy per atom [eV/atom].

    F_vib / N = (9/8) k_B θ_D
              + 3 k_B T × (1/x³) ∫₀^x u² ln(1 − e^{−u}) du

    where x = θ_D / T.  The first term is the zero-point energy (ZPE); the
    second term is the thermal excitation contribution (negative for T > 0).

    Parameters
    ----------
    T       : temperature [K]  (use T > 0; returns ZPE only for T < 1e-3 K)
    theta_D : Debye temperature [K]
    """
    zpe = (9.0 / 8.0) * KB_EV * theta_D    # eV/atom — zero-point energy

    if T < 1e-3:
        return zpe

    x         = theta_D / T
    integral  = _debye_integral_F(x)       # ∫₀^x u² ln(1-e^{-u}) du  (≤ 0)
    excit     = 3.0 * KB_EV * T * (3.0 / x ** 3) * integral   # eV/atom

    return zpe + excit


def _quantum_debye_S_ev_per_K_per_atom(T: float, theta_D: float) -> float:
    """Quantum Debye entropy per atom [eV / (K · atom)].

    S / N = k_B [ 4 D₃(θ_D / T) − 3 ln(1 − e^{−θ_D/T}) ]

    This is the analytic derivative −∂F/∂T|_V of ``_quantum_debye_F_ev_per_atom``.

    Parameters
    ----------
    T       : temperature [K]
    theta_D : Debye temperature [K]
    """
    if T < 1e-3:
        return 0.0   # S → 0 as T → 0  (third law)

    x  = theta_D / T
    d3 = _debye_function_D3(x)

    # Guard for very large x (deep quantum regime, T << θ_D):
    # ln(1-e^{-x}) → -e^{-x} → 0  which is numerically exact via np.log1p.
    if x > 700.0:
        log_term = 0.0
    else:
        log_term = np.log(1.0 - np.exp(-x))   # < 0

    return KB_EV * (4.0 * d3 - 3.0 * log_term)   # eV/(K·atom)


def _vdos_debye_freq_at_volume(
    atoms_template,
    calc_factory,
    volume_scale: float,
    equil_steps: int,
    prod_steps: int,
    timestep_fs: float,
    temperature_K: float,
    sample_interval: int = 4,
) -> float:
    """Run a short NVT→NVE simulation at a scaled volume and return ν_D [THz].

    ν_D is the 99th-percentile VDOS cutoff frequency, matching the convention
    used in ``run_vdos()`` (TEST 13).  The VACF is computed via the
    Wiener-Khinchin power-spectrum method (zero-padded FFT) identical to TEST 13.

    Parameters
    ----------
    atoms_template : ase.Atoms
        Equilibrium unit cell (e.g. 4-atom cubic FCC cell).
    calc_factory   : callable
        Returns a new calculator when called.
    volume_scale   : float
        Fractional volume factor: V = V₀ × volume_scale.
        The lattice constant scales as a = a₀ × volume_scale^{1/3}.
    equil_steps    : int   — NVT equilibration steps.
    prod_steps     : int   — NVE production steps for VACF collection.
    timestep_fs    : float — MD timestep [fs].
    temperature_K  : float — thermostat temperature [K].
    sample_interval: int   — velocity-sampling interval [steps].

    Returns
    -------
    float — ν_D [THz]; falls back to CU_EXP["nu_Debye_THz"] on failure.
    """
    # Build a supercell at the scaled lattice constant
    a_scaled = atoms_template.cell[0, 0] * volume_scale ** (1.0 / 3.0)
    sc_size  = max(2, round((108 / _FCC_BASIS) ** (1.0 / 3.0)))   # ≈ 3 → 108 atoms
    atoms    = bulk("Cu", "fcc", a=a_scaled, cubic=True) * (sc_size, sc_size, sc_size)
    N        = len(atoms)

    atoms.calc = calc_factory(use_cell_list=(N > 300))
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K,
                                 rng=np.random.default_rng(77))
    Stationary(atoms)

    # NVT equilibration (Langevin thermostat)
    nvt = Langevin(
        atoms,
        timestep=timestep_fs * units.fs,
        temperature_K=temperature_K,
        friction=0.01 / units.fs,
        rng=np.random.default_rng(78),
    )
    nvt.run(equil_steps)

    # NVE production — collect velocity snapshots
    nve      = VelocityVerlet(atoms, timestep=timestep_fs * units.fs)
    vel_traj: list[np.ndarray] = []

    def _sample_vel() -> None:
        vel_traj.append(atoms.get_velocities().copy())   # (N, 3) Å/fs

    nve.attach(_sample_vel, interval=sample_interval)
    nve.run(prod_steps)

    # VACF via Wiener-Khinchin (same method as TEST 13 / run_vdos)
    vel      = np.stack(vel_traj, axis=0)   # (M, N, 3)
    M        = vel.shape[0]
    vel_flat = vel.reshape(M, N * 3)

    M_pad  = int(2 ** np.ceil(np.log2(2 * M)))
    Vf     = np.fft.rfft(vel_flat, n=M_pad, axis=0)
    psd    = np.mean(np.abs(Vf) ** 2, axis=1)
    acf    = np.fft.irfft(psd, n=M_pad)[:M]
    n_pairs = M - np.arange(M, dtype=float)
    acf    /= n_pairs

    if acf[0] <= 0.0:
        print("    [WARN] VACF(0) ≤ 0 — falling back to exp ν_D.")
        return float(CU_EXP["nu_Debye_THz"])

    vacf   = acf / acf[0]
    dt_ps  = sample_interval * timestep_fs / 1000.0

    window   = np.hanning(M)
    vdos_raw = np.abs(np.fft.rfft(vacf * window)) ** 2
    freqs    = np.fft.rfftfreq(M, d=dt_ps)   # THz

    mask    = (freqs > 0.0) & (freqs <= 20.0)
    freqs_v = freqs[mask]
    vdos_v  = vdos_raw[mask]

    if freqs_v.size == 0:
        return float(CU_EXP["nu_Debye_THz"])

    area = float(_np_trapz(vdos_v, freqs_v))
    if area > 0.0:
        vdos_v = vdos_v / area

    # 99th-percentile cutoff (same as TEST 13)
    dnu   = np.gradient(freqs_v)
    cumsp = np.cumsum(vdos_v * dnu)
    cs_end = float(cumsp[-1])
    if cs_end <= 0.0:
        return float(CU_EXP["nu_Debye_THz"])

    cumsp /= cs_end
    idx    = min(int(np.searchsorted(cumsp, 0.99)), len(freqs_v) - 1)
    return float(freqs_v[idx])


# ── TEST 17: Quasi-Harmonic Approximation (QHA) ───────────────────────────────

def run_qha(
    calc_factory,
    outdir: Path,
    n_atoms: int,
    a0: float | None = None,
    nu_debye_THz: float | None = None,
    n_ev_points: int = 7,
    dV_frac: float = 0.04,
    n_gruneisen_vols: int = 3,
    gruneisen_equil_steps: int = 2_000,
    gruneisen_prod_steps: int = 6_000,
    gruneisen_timestep_fs: float = 1.0,
    gruneisen_temp_K: float = 300.0,
    T_lo: float = 50.0,
    T_hi: float = 1000.0,
    n_T_points: int = 20,
    target_T: float = 298.15,
) -> dict:
    """TEST 17 — Quasi-Harmonic Approximation (QHA).

    Combines a Birch-Murnaghan E(V) EOS fit with a quantum Debye vibrational
    free energy to compute temperature-dependent equilibrium volume, entropy,
    and volumetric thermal expansion coefficient.

    Method
    ------
    1. **E(V) scan** — Static energies at ``n_ev_points`` volumes spanning
       ±``dV_frac`` around ``a0``, fitted with the 3rd-order Birch-Murnaghan
       equation of state (same EOS used in TEST 2).

    2. **Grüneisen parameter γ_G** — Short NVT→NVE runs at ``n_gruneisen_vols``
       scaled volumes yield VDOS spectra via the velocity-autocorrelation
       function (VACF).  The Debye frequency ν_D(V) at each volume is the 99th-
       percentile spectral cutoff (identical convention to TEST 13 / run_vdos).
       A log-log linear fit gives:

           γ_G = − d ln(ν_D) / d ln(V)

       The volume-dependent Debye temperature then follows:

           θ_D(V) = θ_D(V₀) × (V₀ / V)^γ_G

       ν_D(V₀) is anchored to the TEST 13 result when available, otherwise the
       experimental 7.2 THz is used, to avoid systematic bias from the short
       Grüneisen VDOS runs.

    3. **Quantum Debye free energy** (per atom) —

           F_vib(V, T) = (9/8) k_B θ_D(V)                  [ZPE]
                       + 3 k_B T (1/x³) ∫₀^x u² ln(1−e^{−u}) du

       where x = θ_D(V) / T.  Unlike TEST 16 (classical Debye), ZPE and quantum
       Bose-Einstein statistics are included, so S is expected to agree with
       NIST-JANAF to within ~10–20 % rather than ~35–45 %.

    4. **Volume minimization** — F_total(V, T) = E_BM(V) + F_vib(V, T) is
       minimized over V at each temperature via ``scipy.minimize_scalar``.

    5. **Derived quantities** —

       * V_eq(T): equilibrium volume [Å³/atom] vs temperature.
       * S(T) = −∂F/∂T|_V: evaluated analytically via the quantum Debye entropy
         formula at V = V_eq(T).
       * α(T) = (1/V) dV/dT: volumetric thermal expansion [K⁻¹], computed by
         numerical differentiation of V_eq(T).

    Comparison targets
    ------------------
    * S°(298.15 K) = 33.15 J/(mol·K)       [NIST-JANAF 4th ed., Ref 17]
    * α_vol(293 K) = 3 × 16.6×10⁻⁶ K⁻¹   [Touloukian et al., Ref 7]

    Outputs saved
    -------------
    * qha_thermodynamics.txt — columns: T_K, V_eq_A3_per_atom, S_J_mol_K, alpha_per_K
    * qha_ev_scan.txt        — columns: V_pa_A3, E_pa_eV  (static EOS data)

    Parameters
    ----------
    calc_factory : callable
        Returns a new ASE calculator instance when called.
    outdir : Path
        Directory where output files are written.
    n_atoms : int
        Target atom count used to size the Grüneisen VDOS supercells.
    a0 : float or None
        Equilibrium lattice constant [Å].  Defaults to the EAM value from
        TEST 2 (``results["static"]["eos"]["a0_A"]``) or CU_EXP["a0_A"].
    nu_debye_THz : float or None
        Reference ν_D [THz] at V₀.  Pass ``results["vdos"]["nu_debye_THz"]``
        to use the EAM VDOS result from TEST 13.  Default: 7.2 THz (exp).
    n_ev_points : int
        Number of volume points in the static E(V) scan (default 7).
    dV_frac : float
        Half-range of the volume scan as a fraction of V₀ (default 0.04 = ±4%).
    n_gruneisen_vols : int
        Volumes at which VDOS is computed for the Grüneisen fit (default 3).
    gruneisen_equil_steps : int
        NVT equilibration steps per Grüneisen VDOS run (default 2 000).
    gruneisen_prod_steps : int
        NVE production steps per Grüneisen VDOS run (default 6 000).
    gruneisen_timestep_fs : float
        MD timestep [fs] for Grüneisen runs (default 1.0 fs).
    gruneisen_temp_K : float
        Temperature for Grüneisen VDOS runs [K] (default 300 K).
    T_lo, T_hi : float
        Temperature range [K] for QHA thermodynamic functions (default 50–1000).
    n_T_points : int
        Number of temperature grid points (default 20).
    target_T : float
        Temperature at which S and α are compared to experiment (default 298.15 K).

    Returns
    -------
    dict with keys
        V0_pa_A3, theta_D_V0_K, gamma_G,
        BM_V0_A3, BM_B0_GPa, BM_B0p, BM_E0_eV_per_atom,
        T_arr, V_eq_pa, S_arr_J_mol_K, alpha_arr_per_K,
        S_qha_J_mol_K, alpha_qha_per_K, alpha_exp_vol, S_exp_J_mol_K, T_eval_K,
        passed_S, passed_alpha, error_S_pct, error_alpha_pct
    """
    print_section(
        f"TEST 17  —  Quasi-Harmonic Approximation (QHA)"
        f"  T = {T_lo:.0f}–{T_hi:.0f} K"
    )

    from scipy.optimize import curve_fit, minimize_scalar

    # ── 0. Reference values ────────────────────────────────────────────────────
    a_ref    = a0 if a0 is not None else CU_EXP["a0_A"]
    V0_pa    = a_ref ** 3 / _FCC_BASIS    # volume per atom at reference a₀ [Å³]

    nu_D_ref = nu_debye_THz if nu_debye_THz is not None else CU_EXP["nu_Debye_THz"]
    nu_D_src = "TEST 13 (EAM VDOS)" if nu_debye_THz is not None else "exp (7.2 THz)"
    # Debye temperature anchor: θ_D = h ν_D / k_B   (using HBAR·2π = h)
    theta_D_ref = HBAR_EV_S * (nu_D_ref * 1.0e12) * 2.0 * np.pi / KB_EV   # K

    print(f"  a₀ used      : {a_ref:.4f} Å   →   V₀/atom = {V0_pa:.4f} Å³")
    print(f"  ν_D(V₀) src  : {nu_D_src}  →  {nu_D_ref:.3f} THz  "
          f"θ_D(V₀) = {theta_D_ref:.1f} K")

    # ── STEP 1: Static E(V) scan ───────────────────────────────────────────────
    print_subsection(
        f"Step 1 — Static E(V)  ({n_ev_points} pts, ±{dV_frac*100:.0f}% volume)"
    )

    # 3×3×3 = 108-atom supercell for static calculations; EAM E/atom is size-
    # independent for bulk, so a small cell keeps this step fast.
    sc_s   = 3
    n_sc_s = sc_s ** 3 * _FCC_BASIS    # 108 atoms

    # Lattice-constant fractions that span ±dV_frac in volume
    a_fracs = np.linspace(
        (1.0 - dV_frac) ** (1.0 / 3.0),
        (1.0 + dV_frac) ** (1.0 / 3.0),
        n_ev_points,
    )
    ev_V_pa = np.zeros(n_ev_points)    # Å³ per atom
    ev_E_pa = np.zeros(n_ev_points)    # eV per atom

    for i, af in enumerate(a_fracs):
        a_i   = a_ref * af
        sc    = bulk("Cu", "fcc", a=a_i, cubic=True) * (sc_s, sc_s, sc_s)
        sc.calc = calc_factory()
        ev_E_pa[i] = sc.get_potential_energy() / n_sc_s
        ev_V_pa[i] = sc.get_volume()           / n_sc_s
        print(f"  [{i+1:2d}/{n_ev_points}]  a = {a_i:.4f} Å   "
              f"V = {ev_V_pa[i]:.4f} Å³/atom   E = {ev_E_pa[i]:.6f} eV/atom")

    # Birch-Murnaghan fit (per atom)
    i_min  = int(np.argmin(ev_E_pa))
    p0_bm  = [ev_E_pa[i_min], ev_V_pa[i_min], 1.0 / EV_PER_A3_TO_GPA, 4.0]

    try:
        popt_bm, _ = curve_fit(
            birch_murnaghan, ev_V_pa, ev_E_pa, p0=p0_bm, maxfev=10_000
        )
        E0_pa, V0_pa_bm, B0_ev_pa, B0p_bm = popt_bm
        B0_gpa = B0_ev_pa * EV_PER_A3_TO_GPA
        a0_bm  = (V0_pa_bm * _FCC_BASIS) ** (1.0 / 3.0)
    except Exception as exc:
        print(f"  [WARN] BM fit failed ({exc}); using parabolic fallback.")
        coeffs   = np.polyfit(ev_V_pa, ev_E_pa, 2)
        V0_pa_bm = -coeffs[1] / (2.0 * coeffs[0])
        E0_pa    = float(np.polyval(coeffs, V0_pa_bm))
        B0_ev_pa = 2.0 * coeffs[0] * V0_pa_bm    # B ≈ 2a·V for E=aV²+bV+c
        B0p_bm   = 4.0
        B0_gpa   = B0_ev_pa * EV_PER_A3_TO_GPA
        a0_bm    = (V0_pa_bm * _FCC_BASIS) ** (1.0 / 3.0)
        popt_bm  = np.array([E0_pa, V0_pa_bm, B0_ev_pa, B0p_bm])

    print(f"\n  BM fit results:")
    print(f"    E₀   = {E0_pa:.6f} eV/atom  "
          f"(exp E_coh = {CU_EXP['E_coh_eV']:.3f} eV/atom)")
    print(f"    V₀   = {V0_pa_bm:.4f} Å³/atom  →  a₀ = {a0_bm:.4f} Å  "
          f"(input a₀ = {a_ref:.4f} Å)")
    print(f"    B₀   = {B0_gpa:.1f} GPa  (exp {CU_EXP['B0_GPa']:.0f} GPa)")
    print(f"    B₀′  = {B0p_bm:.3f}")
    print(compare("B₀ [GPa]", B0_gpa, CU_EXP["B0_GPa"], "GPa", tol_pct=10.0))

    # Volume search bounds for minimization [Å³/atom]
    V_lo_pa = ev_V_pa[0]
    V_hi_pa = ev_V_pa[-1]

    save_array(
        outdir / "qha_ev_scan.txt",
        "V_pa_A3   E_pa_eV",
        np.column_stack([ev_V_pa, ev_E_pa]),
    )

    # ── STEP 2: Grüneisen parameter γ_G ───────────────────────────────────────
    print_subsection(
        f"Step 2 — Grüneisen γ_G via VDOS  "
        f"({n_gruneisen_vols} vols, T = {gruneisen_temp_K:.0f} K)"
    )

    # Unit-cell template at the reference lattice constant
    uc_template = bulk("Cu", "fcc", a=a_ref, cubic=True)

    # Volume-scale factors covering ±dV_frac/2 for the Grüneisen fit.
    # Using half-range here keeps the VDOS volumes close to equilibrium where
    # the harmonic approximation is most accurate.
    grun_vscales = np.linspace(
        1.0 - dV_frac / 2.0,
        1.0 + dV_frac / 2.0,
        n_gruneisen_vols,
    )
    grun_V_pa  = V0_pa_bm * grun_vscales    # [Å³/atom]
    grun_nu_D  = np.zeros(n_gruneisen_vols) # [THz]

    for j, vs in enumerate(grun_vscales):
        V_j = grun_V_pa[j]
        print(f"  [{j+1}/{n_gruneisen_vols}]  V_scale = {vs:.4f}  "
              f"(V_pa = {V_j:.4f} Å³/atom)  running VDOS ...")
        grun_nu_D[j] = _vdos_debye_freq_at_volume(
            atoms_template  = uc_template,
            calc_factory    = calc_factory,
            volume_scale    = vs,
            equil_steps     = gruneisen_equil_steps,
            prod_steps      = gruneisen_prod_steps,
            timestep_fs     = gruneisen_timestep_fs,
            temperature_K   = gruneisen_temp_K,
        )
        theta_j = HBAR_EV_S * (grun_nu_D[j] * 1e12) * 2.0 * np.pi / KB_EV
        print(f"    ν_D = {grun_nu_D[j]:.3f} THz   θ_D = {theta_j:.1f} K")

    # Log-log linear fit: ln(ν_D) = −γ_G ln(V) + const
    # → slope = −γ_G  (Grüneisen: γ_G > 0 means ν_D decreases with expansion)
    log_V  = np.log(grun_V_pa)
    log_nu = np.log(grun_nu_D)
    grun_slope, _ = np.polyfit(log_V, log_nu, 1)
    gamma_G = -grun_slope    # positive for most metals (typically 1.5–2.5 for Cu)

    # Anchor θ_D(V₀) to the supplied reference (TEST 13 or exp), not to the
    # short Grüneisen VDOS average, to preserve accuracy of the entropy estimate.
    theta_D0 = theta_D_ref   # K, at V = V0_pa_bm

    def _theta_D(V_pa: float) -> float:
        """Grüneisen-scaled Debye temperature: θ_D(V) = θ_D₀ × (V₀/V)^γ_G."""
        return theta_D0 * (V0_pa_bm / V_pa) ** gamma_G

    print(f"\n  Grüneisen γ_G = {gamma_G:.4f}"
          f"  (literature ~1.7–2.0 for Cu)")
    if gamma_G > 2.5:
        print(
            f"  [WARN] γ_G = {gamma_G:.4f} > 2.5 — likely noise in the short Grüneisen VDOS runs.\n"
            f"         Increase --qha-grun-prod (≥15000) and --qha-n-grun-vols (≥5) for a\n"
            f"         tighter log-log slope.  α overestimation expected."
        )
    print(f"  θ_D anchor (V₀) = {theta_D0:.1f} K  from {nu_D_src}")

    # ── STEP 3: F(V, T) minimization → V_eq(T) ────────────────────────────────
    print_subsection(
        f"Step 3 — Minimizing F_total(V, T)  "
        f"({n_T_points} pts, T = {T_lo:.0f}–{T_hi:.0f} K)"
    )

    T_arr    = np.linspace(T_lo, T_hi, n_T_points)
    V_eq_pa  = np.zeros(n_T_points)   # Å³/atom

    for k, T in enumerate(T_arr):
        def _F_total(V):
            # Static (0-K) contribution [eV/atom] via BM EOS
            E_s = birch_murnaghan(V, *popt_bm)
            # Quantum Debye vibrational free energy [eV/atom]
            F_v = _quantum_debye_F_ev_per_atom(T, _theta_D(V))
            return E_s + F_v

        res = minimize_scalar(
            _F_total,
            bounds=(V_lo_pa, V_hi_pa),
            method="bounded",
            options={"xatol": 1e-8},
        )
        V_eq_pa[k] = res.x

    # ── STEP 4: S(T) and α(T) ─────────────────────────────────────────────────
    # Entropy S(T) = −∂F/∂T|_V evaluated at V = V_eq(T) with the analytical
    # Debye formula.  Because V_eq changes slowly with T, the isochoric
    # approximation introduces negligible error.
    S_ev_K = np.array([
        _quantum_debye_S_ev_per_K_per_atom(T, _theta_D(V_eq_pa[k]))
        for k, T in enumerate(T_arr)
    ])   # eV / (K · atom)

    S_arr_J_mol_K = S_ev_K * EV_TO_J * N_A   # J / (mol · K)

    # Volumetric thermal expansion α(T) = (1/V) dV/dT  [K⁻¹]
    dVdT      = np.gradient(V_eq_pa, T_arr)
    alpha_arr = dVdT / V_eq_pa   # K⁻¹

    # ── STEP 5: Comparison with experiment at target_T ─────────────────────────
    print_subsection(f"Step 4 — Results at T = {target_T:.2f} K")

    idx_T  = int(np.argmin(np.abs(T_arr - target_T)))
    T_eval = float(T_arr[idx_T])

    S_qha     = float(S_arr_J_mol_K[idx_T])
    alpha_qha = float(alpha_arr[idx_T])
    V_eq_ref  = float(V_eq_pa[idx_T])
    a_eq_ref  = (V_eq_ref * _FCC_BASIS) ** (1.0 / 3.0)

    # Experimental volumetric α = 3 × linear α (isotropic, small-strain limit)
    alpha_exp_vol = 3.0 * CU_EXP["alpha_1perK"]   # 49.8×10⁻⁶ K⁻¹
    S_exp         = CU_EXP["S_298K_J_mol_K"]       # 33.15 J/(mol·K)

    err_S     = 100.0 * abs(S_qha     - S_exp)         / abs(S_exp)
    err_alpha = 100.0 * abs(alpha_qha - alpha_exp_vol) / abs(alpha_exp_vol)

    # Tolerances — quantum Debye should achieve ~10–20% for S (vs ~35–45% for
    # classical TEST 16); α is harder because it depends on γ_G accuracy.
    passed_S     = err_S     < 15.0
    passed_alpha = err_alpha < 30.0

    print(compare("S_QHA [J/(mol·K)]",
                  S_qha, S_exp, "J/(mol·K)", tol_pct=15.0))
    print(compare("α_QHA [×10⁻⁶ K⁻¹]",
                  alpha_qha * 1e6, alpha_exp_vol * 1e6, "×10⁻⁶ K⁻¹", tol_pct=30.0))
    print(f"  V_eq({T_eval:.0f} K) = {V_eq_ref:.4f} Å³/atom  →  "
          f"a_eq = {a_eq_ref:.4f} Å  (BM V₀ = {V0_pa_bm:.4f} Å³/atom)")
    print(f"  θ_D({T_eval:.0f} K) = {_theta_D(V_eq_ref):.1f} K  "
          f"(anchor θ_D₀ = {theta_D0:.1f} K,  γ_G = {gamma_G:.4f})")
    print(f"  [INFO] QHA includes ZPE + quantum statistics → "
          f"S should be ~10–20% error vs exp  (TEST 16 classical: ~35–45%)")

    # Save full T-dependent thermodynamic functions
    save_array(
        outdir / "qha_thermodynamics.txt",
        "T_K   V_eq_A3_per_atom   S_J_mol_K   alpha_per_K",
        np.column_stack([T_arr, V_eq_pa, S_arr_J_mol_K, alpha_arr]),
    )

    return {
        # EOS fit
        "BM_E0_eV_per_atom" : float(E0_pa),
        "BM_V0_A3"          : float(V0_pa_bm),
        "BM_B0_GPa"         : float(B0_gpa),
        "BM_B0p"            : float(B0p_bm),
        # Grüneisen / Debye
        "V0_pa_A3"          : float(V0_pa_bm),
        "theta_D_V0_K"      : float(theta_D0),
        "gamma_G"           : float(gamma_G),
        # Temperature-dependent arrays
        "T_arr"             : T_arr,
        "V_eq_pa"           : V_eq_pa,
        "S_arr_J_mol_K"     : S_arr_J_mol_K,
        "alpha_arr_per_K"   : alpha_arr,
        # Scalar results at target_T
        "S_qha_J_mol_K"     : S_qha,
        "alpha_qha_per_K"   : alpha_qha,
        "alpha_exp_vol"     : alpha_exp_vol,
        "S_exp_J_mol_K"     : S_exp,
        "T_eval_K"          : T_eval,
        "passed_S"          : passed_S,
        "passed_alpha"      : passed_alpha,
        "error_S_pct"       : float(err_S),
        "error_alpha_pct"   : float(err_alpha),
    }


def write_summary(outdir: Path, results: dict, backend: str) -> None:
    lines = [
        "=" * 70,
        "  CREAM Cu EAM Simulation — Summary Report",
        f"  Backend : {backend}",
        f"  Output  : {outdir}",
        "=" * 70,
        "",
    ]

    if "static" in results:
        s   = results["static"]
        eos = s.get("eos", {})
        lines += [
            "  Static / EOS",
            (f"    Cohesive energy (sp)  : {s['e_sp']/s['n_atoms']:.4f} eV/atom"
             f"  (exp: {CU_EXP['E_coh_eV']:.3f} eV/atom)"),
        ]
        if eos:
            lines += [
                (f"    Lattice parameter a0  : {eos.get('a0_A', float('nan')):.4f} Å"
                 f"  (exp: {CU_EXP['a0_A']:.3f} Å)"),
                (f"    Bulk modulus B0       : {eos.get('B0_GPa', float('nan')):.1f} GPa"
                 f"  (exp: {CU_EXP['B0_GPa']:.0f} GPa)"),
            ]

    if "nvt_300K" in results:
        nv = results["nvt_300K"]
        lines += [
            "",
            "  NVT 300 K",
            f"    Temperature           : {nv['T_mean']:.2f} ± {nv['T_std']:.2f} K",
            f"    Epot/atom             : {nv['Epot_mean']:.4f} eV",
        ]
        if nv.get("r1_rdf"):
            r1_exp = CU_EXP["a0_A"] / np.sqrt(2.0)
            lines.append(
                f"    1st RDF peak          : {nv['r1_rdf']:.3f} Å  (FCC NN: {r1_exp:.3f} Å)"
            )
        lines.append(f"    Final MSD             : {nv['msd_final']:.3f} Å²")

    if "npt_300K" in results:
        p = results["npt_300K"]
        lines += [
            "",
            "  NPT 300 K, 0 GPa",
            f"    Temperature           : {p['T_mean']:.2f} K",
            f"    Pressure              : {p['P_mean']:.3f} GPa",
            (f"    Lattice parameter     : {p['a_mean']:.4f} Å"
             f"  (exp: {CU_EXP['a0_A']:.3f} Å)"),
            (f"    B0 (V-fluctuation)    : {p['B_fluct']:.1f} GPa"
             f"  (exp: {CU_EXP['B0_GPa']:.0f} GPa)"),
        ]

    if "thermal" in results:
        t = results["thermal"]
        lines += [
            "",
            "  Thermal expansion",
            (f"    α = {t['alpha_per_K']*1e6:.2f} × 10⁻⁶ K⁻¹  "
             f"(exp: {CU_EXP['alpha_1perK']*1e6:.1f} × 10⁻⁶ K⁻¹)"),
        ]

    if "near_melting" in results:
        m = results["near_melting"]
        lines += [
            "",
            "  Near-melting (1300 K)",
            f"    T_mean = {m['T_mean']:.1f} K   MSD = {m['msd_final']:.3f} Å²",
        ]

    if "npt_triclinic" in results:
        tr = results["npt_triclinic"]
        lines += [
            "",
            "  NPT Triclinic Relaxation",
            f"    Initial angles        : {tr['init_angles']}",
            f"    Final angles          : {tr['final_angles']}",
            f"    Orthogonal recovery   : {'PASS' if tr['success'] else 'FAIL'}",
        ]

    if "elastic" in results:
        e = results["elastic"]
        lines += [
            "",
            "  Elastic Constants (Voigt finite strain)",
            f"    C11 = {e['C11']:.1f} GPa  (exp {CU_EXP['C11_GPa']:.1f})",
            f"    C12 = {e['C12']:.1f} GPa  (exp {CU_EXP['C12_GPa']:.1f})",
            f"    C44 = {e['C44']:.1f} GPa  (exp {CU_EXP['C44_GPa']:.1f})",
            f"    B0  = {e['B0']:.1f} GPa   Zener A = {e['A_zener']:.4f}"
            f"  (exp {CU_EXP['A_zener']:.2f})",
        ]

    if "vacancy" in results:
        v = results["vacancy"]
        lines += [
            "",
            "  Vacancy Formation Energy",
            f"    E_vac (unrelaxed) = {v['E_form_eV']:.4f} eV"
            f"  (exp ~{CU_EXP['E_vac_eV']:.2f} eV)",
            f"    1st-shell ΔE      = {v['sh1_delta_E_meV']:.2f} meV/atom",
        ]

    if "eam_fields" in results:
        f = results["eam_fields"]
        lines += [
            "",
            "  EAM Per-atom Fields (CPU backend)",
            f"    <ρᵢ>  (0 K)      = {f['rho_mean_0K']:.8f}",
            f"    std(ρᵢ) (300 K)  = {f['rho_std_300K']:.6f}"
            f"  (×{f['std_ratio_rho']:.0f} vs 0 K)",
            f"    <F(ρᵢ)> (0 K)   = {f['Frho_mean_0K']:.6f} eV",
            f"    Embedding/E_tot  = {f['embedding_fraction']:.6f}",
        ]

    if "diffusion" in results:
        d = results["diffusion"]
        lines += ["", "  Self-Diffusion (NVE Einstein, liquid Cu regime)"]
        for T, D in zip(d["temperatures"], d["D_values"]):
            lines.append(f"    D({T:.0f} K) = {D:.3e} m²/s")
        lines.append(
            f"    Arrhenius: E_a = {d['Ea_eV']:.3f} eV"
            f"  D0 = {d['D0_m2s']:.3e} m²/s"
            f"  (liquid ref E_a ~ 0.45 eV  |  solid-state exp: {CU_EXP['Ea_diff_eV']:.2f} eV)"
        )

    if "vdos" in results:
        v = results["vdos"]
        lines += [
            "",
            "  Vibrational DOS (VACF + FFT)",
            f"    ν_peak  = {v['nu_peak_THz']:.3f} THz",
            f"    ν_Debye = {v['nu_debye_THz']:.3f} THz"
            f"  (exp ~{CU_EXP['nu_Debye_THz']:.1f} THz)",
            f"    Θ_D     = {v['theta_D_K']:.1f} K"
            f"  (exp {CU_EXP['theta_D_K']:.0f} K)",
        ]

    if "stacking_fault" in results:
        sf = results["stacking_fault"]
        lines += [
            "",
            "  Intrinsic Stacking Fault Energy (ISFE)",
            f"    γ_sf (unrelaxed)  = {sf['gamma_sf_unrelaxed_mJ_m2']:.1f} mJ/m²",
            (f"    γ_sf (z-relaxed) = {sf['gamma_sf_relaxed_mJ_m2']:.1f} mJ/m²"
             f"  (exp ~{CU_EXP['gamma_sf_mJ_m2']:.0f} mJ/m²)"),
            f"    Fault area        = {sf['A_fault_A2']:.1f} Å²",
            f"    z-relaxation OK   : {sf['relax_ok']}",
        ]

    if "heat_capacity" in results:
        cp = results["heat_capacity"]
        lines += [
            "",
            "  Heat Capacity C_P (NVT potential-energy fluctuations)",
            (f"    C_V = {cp['C_V_J_mol_K']:.4f} J/(mol·K)"
             f"  (Dulong-Petit 3R = {3*R_J_MOL_K:.4f} J/(mol·K))"),
            (f"    C_P = {cp['C_P_J_mol_K']:.4f} J/(mol·K)"
             f"  (exp {CU_EXP['C_P_300K_J_mol_K']:.2f}"
             f" | classical limit {3*R_J_MOL_K:.4f} J/(mol·K))"),
            f"    C_P / R           = {cp['C_P_per_R']:.4f}  (classical limit = 3.000)",
            f"    Var(E_pot)        = {cp['Epot_var_eV2']:.4e} eV²"
            f"  ({cp['n_samples']} samples)",
        ]

    if "free_energy" in results:
        fe = results["free_energy"]
        lines += [
            "",
            "  Free Energy A & Entropy S (quasiharmonic Debye, classical)",
            f"    ν_D used          = {fe['nu_debye_THz']:.3f} THz"
            f"  hν_D = {fe['hnu_D_eV']*1000:.2f} meV  x={fe['x_debye']:.3f}",
            f"    U₀/atom (0-K)     = {fe['U0_eV_per_atom']:.4f} eV/atom"
            f"  (exp E_coh = {CU_EXP['E_coh_eV']:.4f} eV/atom)",
            f"    A_vib({fe['T_mean_K']:.0f} K)      ="
            f" {fe['A_vib_J_mol']/1000:.4f} kJ/mol",
            f"    A_total           = {fe['A_total_J_mol']/1000:.4f} kJ/mol",
            (f"    S_vib (classical) = {fe['S_vib_J_mol_K']:.2f} J/(mol·K)"
             f"  (exp {CU_EXP['S_298K_J_mol_K']:.2f}"
             f" | ratio {fe['S_ratio']:.2f}× — expected: classical ~35-45% high)"),
            f"    E_kin equip ratio = {fe['Ekin_equip_ratio']:.4f}  (ideal = 1.000)",
        ]

    if "qha" in results:
        q = results["qha"]
        _s_status = "PASS" if q["passed_S"]     else "WARN"
        _a_status = "PASS" if q["passed_alpha"] else "WARN"
        lines += [
            "",
            "  Quasi-Harmonic Approximation — TEST 17",
            f"    BM EOS:  E₀ = {q['BM_E0_eV_per_atom']:.4f} eV/atom"
            f"   V₀ = {q['BM_V0_A3']:.4f} Å³/atom"
            f"   B₀ = {q['BM_B0_GPa']:.1f} GPa",
            f"    Grüneisen γ_G     = {q['gamma_G']:.4f}"
            f"  (literature ~1.7–2.0 for Cu)",
            f"    θ_D anchor (V₀)   = {q['theta_D_V0_K']:.1f} K",
            (f"    S_QHA ({q['T_eval_K']:.0f} K) = {q['S_qha_J_mol_K']:.2f} J/(mol·K)"
             f"  exp {q['S_exp_J_mol_K']:.2f}  Δ={q['error_S_pct']:.1f}%"
             f"  [{_s_status}]"),
            (f"    α_QHA ({q['T_eval_K']:.0f} K) = {q['alpha_qha_per_K']*1e6:.2f} ×10⁻⁶ K⁻¹"
             f"  exp {q['alpha_exp_vol']*1e6:.1f}  Δ={q['error_alpha_pct']:.1f}%"
             f"  [{_a_status}]"),
            f"    [INFO] Quantum Debye: expected S error ~10–20%"
            f" (vs ~35–45% for classical TEST 16)",
        ]

    lines += ["", "=" * 70]
    report = "\n".join(lines)
    print("\n" + report)

    path = outdir / "summary.txt"
    path.write_text(report + "\n", encoding="utf-8")
    print(f"\n  Full report saved to: {path}")


# ── Plots ──────────────────────────────────────────────────────────────────────

def _plot_results(outdir: Path, results: dict) -> None:
    import matplotlib.pyplot as plt

    # E-V curve
    ev_path = outdir / "ev_curve.txt"
    if ev_path.exists() and "static" in results:
        n    = results["static"]["n_atoms"]
        data = np.loadtxt(ev_path)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(data[:, 0] / n, data[:, 1] / n, "o-", color="steelblue", label="EAM")
        ax.set_xlabel("Volume per atom (Å³)")
        ax.set_ylabel("Energy per atom (eV)")
        ax.set_title("Cu E-V curve")
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / "ev_curve.png", dpi=150)
        plt.close(fig)
        print("  Saved: ev_curve.png")

    # RDF
    rdf_path = outdir / "nvt_300K_rdf.txt"
    if rdf_path.exists():
        data  = np.loadtxt(rdf_path)
        r1_exp = CU_EXP["a0_A"] / np.sqrt(2.0)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(data[:, 0], data[:, 1], color="steelblue")
        ax.axvline(r1_exp, ls="--", color="red", label=f"FCC NN = {r1_exp:.3f} Å")
        ax.set_xlabel("r (Å)")
        ax.set_ylabel("g(r)")
        ax.set_title("RDF — Cu NVT 300 K")
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / "rdf_300K.png", dpi=150)
        plt.close(fig)
        print("  Saved: rdf_300K.png")

    # NVT T and Epot trace
    nvt_path = outdir / "nvt_300K_log.txt"
    if nvt_path.exists():
        data = np.loadtxt(nvt_path)
        fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
        axes[0].plot(data[:, 0] / 1000, data[:, 1], lw=0.7, color="tomato")
        axes[0].axhline(300, ls="--", color="k", alpha=0.5, label="Target 300 K")
        axes[0].set_ylabel("T (K)")
        axes[0].legend(fontsize=8)
        axes[1].plot(data[:, 0] / 1000, data[:, 2], lw=0.7, color="steelblue")
        axes[1].set_ylabel("Epot / atom (eV)")
        axes[1].set_xlabel("Time (ps)")
        fig.suptitle("NVT 300 K — Cu")
        fig.tight_layout()
        fig.savefig(outdir / "nvt_300K_trace.png", dpi=150)
        plt.close(fig)
        print("  Saved: nvt_300K_trace.png")

    # Thermal expansion
    te_path = outdir / "thermal_expansion.txt"
    if te_path.exists():
        data   = np.loadtxt(te_path)
        coeffs = np.polyfit(data[:, 0], data[:, 1], 1)
        T_fit  = np.linspace(data[0, 0], data[-1, 0], 100)
        alpha_fit = coeffs[0] / data[0, 1] * 1e6
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(data[:, 0], data[:, 1], "o", color="steelblue", label="NPT")
        ax.plot(T_fit, np.polyval(coeffs, T_fit), "--", color="tomato",
                label=f"fit α={alpha_fit:.2f}×10⁻⁶ K⁻¹")
        ax.set_xlabel("T (K)")
        ax.set_ylabel("a (Å)")
        ax.set_title("Thermal expansion — Cu")
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / "thermal_expansion.png", dpi=150)
        plt.close(fig)
        print("  Saved: thermal_expansion.png")

    # Elastic strain-stress curves
    es_path = outdir / "elastic_strain_stress.txt"
    if es_path.exists():
        data = np.loadtxt(es_path)
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].plot(data[:, 0] * 100, data[:, 1], "o-",
                     color="steelblue", label="σ_xx (→ C11)")
        axes[0].plot(data[:, 0] * 100, data[:, 2], "s--",
                     color="tomato",   label="σ_yy (→ C12)")
        axes[0].set_xlabel("Uniaxial strain δ (%)")
        axes[0].set_ylabel("Stress (GPa)")
        axes[0].legend(fontsize=8)
        axes[0].set_title("Uniaxial strain scan")
        axes[1].plot(data[:, 0] * 100, data[:, 3], "o-",
                     color="mediumseagreen", label="σ_yz (→ C44)")
        axes[1].set_xlabel("Shear strain δ (%)")
        axes[1].set_ylabel("Stress (GPa)")
        axes[1].legend(fontsize=8)
        axes[1].set_title("Shear strain scan")
        fig.tight_layout()
        fig.savefig(outdir / "elastic_strain_stress.png", dpi=150)
        plt.close(fig)
        print("  Saved: elastic_strain_stress.png")

    # Vacancy perturbation field
    vac_path = outdir / "vacancy_perturbation.txt"
    if vac_path.exists():
        data = np.loadtxt(vac_path)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(data[:, 0], data[:, 1], s=3, alpha=0.5, color="steelblue")
        ax.axhline(0, ls="--", color="gray", lw=0.8)
        ax.set_xlabel("Distance from vacancy (Å)")
        ax.set_ylabel("ΔE per atom (meV)")
        ax.set_title("Per-atom energy perturbation — Cu vacancy")
        fig.tight_layout()
        fig.savefig(outdir / "vacancy_perturbation.png", dpi=150)
        plt.close(fig)
        print("  Saved: vacancy_perturbation.png")

    # EAM ρᵢ histogram
    rho_path = outdir / "rho_histogram_300K.txt"
    if rho_path.exists():
        data = np.loadtxt(rho_path)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(data[:, 0], data[:, 1],
               width=data[1, 0] - data[0, 0],
               color="mediumseagreen", alpha=0.85, edgecolor="none")
        ax.set_xlabel("Electron density ρᵢ")
        ax.set_ylabel("Atom count")
        ax.set_title("EAM electron density distribution — Cu 300 K")
        fig.tight_layout()
        fig.savefig(outdir / "eam_rho_hist.png", dpi=150)
        plt.close(fig)
        print("  Saved: eam_rho_hist.png")

    # Self-diffusion MSD multi-temperature
    msd_files = sorted(outdir.glob("nve_msd_*K.txt"))
    if msd_files:
        fig, ax = plt.subplots(figsize=(7, 4))
        cmap = plt.cm.plasma
        for i, f in enumerate(msd_files):
            T_label = f.stem.replace("nve_msd_", "").replace("K", " K")
            d = np.loadtxt(f)
            ax.plot(d[:, 0], d[:, 1],
                    label=T_label,
                    color=cmap(i / max(len(msd_files) - 1, 1)),
                    lw=1.0)
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("MSD (Å²)")
        ax.set_title("NVE MSD — Cu self-diffusion")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / "msd_multi_T.png", dpi=150)
        plt.close(fig)
        print("  Saved: msd_multi_T.png")

    # Arrhenius plot
    arr_path = outdir / "diffusion_arrhenius.txt"
    if arr_path.exists() and "diffusion" in results:
        data  = np.loadtxt(arr_path)
        D_vals = data[:, 2]
        valid  = np.isfinite(D_vals) & (D_vals > 0)
        if valid.sum() >= 2:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(data[valid, 1] * 1000, np.log10(D_vals[valid]),
                    "o-", color="tomato")
            ax.set_xlabel("1000/T (K⁻¹)")
            ax.set_ylabel("log₁₀ D (m²/s)")
            ax.set_title("Arrhenius plot — Cu self-diffusion")
            fig.tight_layout()
            fig.savefig(outdir / "arrhenius.png", dpi=150)
            plt.close(fig)
            print("  Saved: arrhenius.png")

    # VDOS spectrum
    for vf in sorted(outdir.glob("vdos_*K.txt")):
        data    = np.loadtxt(vf)
        T_label = vf.stem.replace("vdos_", "").replace("K", " K")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(data[:, 0], data[:, 1], color="steelblue", lw=1.2)
        if "vdos" in results:
            nu_d = results["vdos"]["nu_debye_THz"]
            ax.axvline(nu_d, ls="--", color="tomato",
                       label=f"ν_D (calc) = {nu_d:.2f} THz")
            ax.axvline(CU_EXP["nu_Debye_THz"], ls=":", color="gray",
                       label=f"ν_D (exp)  = {CU_EXP['nu_Debye_THz']:.1f} THz")
        ax.set_xlabel("Frequency (THz)")
        ax.set_ylabel("VDOS (normalised)")
        ax.set_title(f"Vibrational DOS — Cu NVE VACF  {T_label}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / f"vdos_{T_label.replace(' ', '')}.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: vdos_{T_label.replace(' ', '')}.png")

    # Stacking fault energy bar chart (TEST 14)
    sf_path = outdir / "stacking_fault_energies.txt"
    if sf_path.exists() and "stacking_fault" in results:
        sf = results["stacking_fault"]
        labels = ["Unrelaxed", "z-Relaxed", "Experiment\n(exp ~45)"]
        vals   = [sf["gamma_sf_unrelaxed_mJ_m2"],
                  sf["gamma_sf_relaxed_mJ_m2"],
                  CU_EXP["gamma_sf_mJ_m2"]]
        colors = ["steelblue", "mediumseagreen", "tomato"]
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(labels, vals, color=colors, alpha=0.85, edgecolor="k", linewidth=0.5)
        ax.set_ylabel("γ_sf (mJ/m²)")
        ax.set_title("Cu Intrinsic Stacking Fault Energy — TEST 14")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2.0, val + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, max(vals) * 1.25)
        fig.tight_layout()
        fig.savefig(outdir / "stacking_fault_energy.png", dpi=150)
        plt.close(fig)
        print("  Saved: stacking_fault_energy.png")

    # E_pot time series (TEST 15 — NVT heat capacity)
    for cp_file in sorted(outdir.glob("heat_capacity_*K.txt")):
        data    = np.loadtxt(cp_file)
        T_label = cp_file.stem.replace("heat_capacity_", "").replace("K", " K")
        fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
        axes[0].plot(data[:, 0], data[:, 1], lw=0.6, color="steelblue", label="E_pot (eV)")
        axes[0].set_ylabel("Potential energy E_pot (eV)")
        axes[0].legend(fontsize=8)
        axes[1].plot(data[:, 0], data[:, 2], lw=0.6, color="tomato", label="T (K)")
        axes[1].axhline(300, ls="--", color="k", alpha=0.5, lw=0.8)
        axes[1].set_ylabel("T (K)")
        axes[1].set_xlabel("Time (ps)")
        axes[1].legend(fontsize=8)
        fig.suptitle(f"NVT E_pot trace — Cu {T_label}  (TEST 15)")
        fig.tight_layout()
        fig.savefig(outdir / f"heat_capacity_{T_label.replace(' ', '')}.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: heat_capacity_{T_label.replace(' ', '')}.png")

    # Free energy thermodynamic functions (TEST 16)
    fe_file = outdir / "free_energy.txt"
    if fe_file.exists():
        data = np.loadtxt(fe_file)
        fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
        axes[0].plot(data[:, 0], data[:, 1] / 1000, lw=1.2, color="steelblue",
                     label="A_vib (kJ/mol)")
        axes[0].plot(data[:, 0], data[:, 3] / 1000, lw=1.2, color="tomato",
                     label="U_vib (kJ/mol)", ls="--")
        axes[0].set_ylabel("Energy (kJ/mol)")
        axes[0].legend(fontsize=8)
        axes[1].plot(data[:, 0], data[:, 2], lw=1.2, color="seagreen",
                     label="S_vib (J/mol·K)")
        axes[1].axhline(CU_EXP["S_298K_J_mol_K"], ls="--", color="k", alpha=0.6, lw=0.8,
                        label=f"exp S°(298K) = {CU_EXP['S_298K_J_mol_K']:.2f} J/(mol·K)")
        axes[1].set_ylabel("S_vib (J/mol·K)")
        axes[1].set_xlabel("T (K)")
        axes[1].legend(fontsize=8)
        fig.suptitle("Quasiharmonic Debye — Cu free energy & entropy  (TEST 16)")
        fig.tight_layout()
        fig.savefig(outdir / "free_energy.png", dpi=150)
        plt.close(fig)
        print("  Saved: free_energy.png")

    # QHA thermodynamic functions (TEST 17)
    qha_file = outdir / "qha_thermodynamics.txt"
    if qha_file.exists():
        data = np.loadtxt(qha_file)
        T      = data[:, 0]
        V_eq   = data[:, 1]
        S_data = data[:, 2]
        alpha  = data[:, 3]

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle("TEST 17 — Quasi-Harmonic Approximation (QHA)  Cu EAM",
                     fontsize=11)

        # Panel (a): equilibrium volume V_eq(T)
        V0_ref = V_eq[0]   # low-T reference
        axes[0].plot(T, V_eq / V0_ref, lw=1.5, color="steelblue")
        axes[0].axhline(1.0, ls="--", color="gray", lw=0.8, label="V₀ (low T)")
        axes[0].set_xlabel("Temperature (K)")
        axes[0].set_ylabel("V_eq / V(T_lo)")
        axes[0].set_title("Equilibrium volume  V_eq(T)")
        axes[0].legend(fontsize=8)

        # Panel (b): entropy S(T) vs exp
        S_exp_ref = CU_EXP["S_298K_J_mol_K"]
        axes[1].plot(T, S_data, lw=1.5, color="seagreen", label="QHA (quantum Debye)")
        axes[1].axhline(S_exp_ref, ls="--", color="k", alpha=0.7,
                        label=f"NIST-JANAF {S_exp_ref:.2f} J/(mol·K)")
        axes[1].axvline(298.15, ls=":", color="gray", lw=0.8)
        if "qha" in results:
            q = results["qha"]
            axes[1].scatter([q["T_eval_K"]], [q["S_qha_J_mol_K"]],
                            color="tomato", zorder=5,
                            label=f"Calc {q['S_qha_J_mol_K']:.2f}  Δ={q['error_S_pct']:.1f}%")
        axes[1].set_xlabel("Temperature (K)")
        axes[1].set_ylabel("S (J mol⁻¹ K⁻¹)")
        axes[1].set_title("Entropy  S(T)")
        axes[1].legend(fontsize=7)

        # Panel (c): thermal expansion α(T) vs exp
        alpha_exp = 3.0 * CU_EXP["alpha_1perK"]
        axes[2].plot(T, alpha * 1e6, lw=1.5, color="tomato", label="QHA")
        axes[2].axhline(alpha_exp * 1e6, ls="--", color="k", alpha=0.7,
                        label=f"exp (vol) {alpha_exp*1e6:.1f} ×10⁻⁶ K⁻¹")
        axes[2].axvline(298.15, ls=":", color="gray", lw=0.8)
        axes[2].set_xlabel("Temperature (K)")
        axes[2].set_ylabel("α (×10⁻⁶ K⁻¹)")
        axes[2].set_title("Volumetric thermal expansion  α(T)")
        axes[2].legend(fontsize=7)

        # Also overplot QHA vs classical (TEST 16) entropy if both exist
        if fe_file.exists():
            fe_data = np.loadtxt(fe_file)
            axes[1].plot(fe_data[:, 0], fe_data[:, 2], lw=1.0, ls="-.",
                         color="mediumpurple", alpha=0.7,
                         label="Classical Debye (TEST 16)")
            axes[1].legend(fontsize=7)

        fig.tight_layout()
        fig.savefig(outdir / "qha_thermodynamics.png", dpi=150)
        plt.close(fig)
        print("  Saved: qha_thermodynamics.png")

        # Second figure: BM EOS fit (TEST 17 Step 1)
        ev_file = outdir / "qha_ev_scan.txt"
        if ev_file.exists():
            ev_data = np.loadtxt(ev_file)
            V_fit   = np.linspace(ev_data[0, 0], ev_data[-1, 0], 200)
            if "qha" in results:
                q      = results["qha"]
                E_fit  = birch_murnaghan(
                    V_fit,
                    q["BM_E0_eV_per_atom"], q["BM_V0_A3"],
                    q["BM_B0_GPa"] / EV_PER_A3_TO_GPA, q["BM_B0p"],
                )
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                ax2.scatter(ev_data[:, 0], ev_data[:, 1], s=40, color="steelblue",
                            zorder=5, label="EAM static")
                ax2.plot(V_fit, E_fit, "-", color="tomato", lw=1.5,
                         label=(f"BM fit  B₀={q['BM_B0_GPa']:.0f} GPa"
                                f"  B₀′={q['BM_B0p']:.2f}"))
                ax2.set_xlabel("Volume per atom (Å³)")
                ax2.set_ylabel("Energy per atom (eV)")
                ax2.set_title("TEST 17 — BM EOS fit  (E-V scan)")
                ax2.legend(fontsize=8)
                fig2.tight_layout()
                fig2.savefig(outdir / "qha_ev_curve.png", dpi=150)
                plt.close(fig2)
                print("  Saved: qha_ev_curve.png")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CREAM Cu EAM simulation suite — static, MD, elastic, defect, VDOS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--potential", "-p",
        required=True,
        help="Path to Cu01.eam.alloy (or any compatible Cu .eam.alloy file)",
    )
    parser.add_argument(
        "--backend", "-b",
        default="gpu",
        choices=["gpu", "cpu"],
        help="Compute backend (default: gpu)",
    )
    parser.add_argument(
        "--size", "-s",
        type=int,
        default=16,
        help="Supercell replicas per axis (default: 16 → 16384 atoms for FCC)",
    )
    parser.add_argument(
        "--skip",
        default="",
        help="Comma-separated test numbers to skip, e.g. --skip 6,7,8  or  --skip 9,10,11,12,13,14,15,16,17",
    )
    parser.add_argument(
        "--outdir",
        default="cream_results",
        help="Output directory for data files (default: ./cream_results)",
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=2.0,
        help="MD timestep in fs (default: 2.0)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-step MD logging",
    )
    parser.add_argument(
        "--friction",
        type=float,
        default=0.02,
        help="Langevin friction coefficient [1/fs] (default: 0.02 → 50 fs coupling)",
    )
    parser.add_argument(
        "--equil-steps",
        type=int,
        default=None,
        dest="equil_steps",
        help="Override equilibration steps for tests 4/5/6/7/8 "
             "(default: per-test heuristic scaled by --timestep)",
    )
    parser.add_argument(
        "--prod-steps",
        type=int,
        default=None,
        dest="prod_steps",
        help="Override production steps for tests 4/5 "
             "(default: per-test heuristic scaled by --timestep)",
    )
    parser.add_argument(
        "--nvt-temp",
        type=float,
        default=1300.0,
        dest="nvt_temp",
        help="Temperature for TEST 7 near-melting NVT run [K] (default: 1300.0)",
    )
    parser.add_argument(
        "--diff-temps",
        type=str,
        default="1400,1500,1600,1700,1800,1900,2000",
        dest="diff_temps",
        help="Comma-separated temperatures for TEST 12 diffusion run [K] "
             "(default: 1400,1500,1600,1700,1800,1900,2000 — liquid Cu regime)",
    )
    parser.add_argument(
        "--vdos-temp",
        type=float,
        default=300.0,
        dest="vdos_temp",
        help="Temperature for TEST 13 VDOS run [K] (default: 300.0)",
    )
    parser.add_argument(
        "--ttime",
        type=float,
        default=25.0,
        help="Nosé-Hoover thermostat time constant [fs] (default: 25.0)",
    )
    parser.add_argument(
        "--tau-p",
        type=float,
        default=2000.0,
        dest="tau_p",
        help="NPT barostat time constant [fs] (default: 2000.0 = 2 ps)",
    )

    # ── TEST 17: QHA parameters ────────────────────────────────────────────────
    parser.add_argument(
        "--qha-t-lo",
        type=float,
        default=50.0,
        dest="qha_t_lo",
        help="QHA lower temperature bound [K] (default: 50.0)",
    )
    parser.add_argument(
        "--qha-t-hi",
        type=float,
        default=1000.0,
        dest="qha_t_hi",
        help="QHA upper temperature bound [K] (default: 1000.0)",
    )
    parser.add_argument(
        "--qha-n-t",
        type=int,
        default=30,
        dest="qha_n_t",
        help="Number of temperature grid points for QHA (default: 30)",
    )
    parser.add_argument(
        "--qha-target-t",
        type=float,
        default=298.15,
        dest="qha_target_t",
        help="Temperature at which S and α are compared to experiment [K] "
             "(default: 298.15)",
    )
    parser.add_argument(
        "--qha-n-ev",
        type=int,
        default=15,
        dest="qha_n_ev",
        help="Number of volume points in the QHA static E(V) scan (default: 15)",
    )
    parser.add_argument(
        "--qha-dv-frac",
        type=float,
        default=0.04,
        dest="qha_dv_frac",
        help="Half-range of QHA volume scan as fraction of V₀ (default: 0.04 = ±4%%)",
    )
    parser.add_argument(
        "--qha-n-grun-vols",
        type=int,
        default=10,
        dest="qha_n_grun_vols",
        help="Number of volumes for Grüneisen γ_G fit; ≥5 strongly recommended "
             "for a well-constrained log-log slope (default: 10)",
    )
    parser.add_argument(
        "--qha-grun-equil",
        type=int,
        default=5_000,
        dest="qha_grun_equil",
        help="NVT equilibration steps per Grüneisen VDOS run (default: 5000)",
    )
    parser.add_argument(
        "--qha-grun-prod",
        type=int,
        default=25_000,
        dest="qha_grun_prod",
        help="NVE production steps per Grüneisen VDOS run; longer runs give a "
             "more reliable ν_D and tighter γ_G (default: 25000)",
    )
    parser.add_argument(
        "--qha-grun-dt",
        type=float,
        default=1.0,
        dest="qha_grun_dt",
        help="MD timestep [fs] for Grüneisen VDOS runs (default: 1.0)",
    )
    parser.add_argument(
        "--qha-grun-temp",
        type=float,
        default=300.0,
        dest="qha_grun_temp",
        help="Temperature [K] for Grüneisen VDOS runs (default: 300.0)",
    )

    args = parser.parse_args()

    _require("cream", "cream-python")
    _require("ase", "ase")
    _require("numpy", "numpy")
    _require("scipy", "scipy")

    from cream import CreamCalculator

    pot_path = Path(args.potential)
    if not pot_path.exists():
        print(f"[ERROR] Potential file not found: {pot_path}")
        sys.exit(1)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    skip = {int(x) for x in args.skip.split(",") if x.strip()}
    dt   = args.timestep

    # ── Derived / overridable run parameters ──────────────────────────────────
    _friction = args.friction
    _ttime    = args.ttime
    _tau_p    = args.tau_p
    _diff_temps = tuple(float(t) for t in args.diff_temps.split(",") if t.strip())

    # equil/prod step helpers — honour --equil-steps / --prod-steps overrides
    def _equil(default_steps_at_2fs: int) -> int:
        """Scale default equil step count to current timestep, then apply override."""
        scaled = int(default_steps_at_2fs * 2.0 / dt)
        return args.equil_steps if args.equil_steps is not None else scaled

    def _prod(default_steps_at_2fs: int) -> int:
        """Scale default prod step count to current timestep, then apply override."""
        scaled = int(default_steps_at_2fs * 2.0 / dt)
        return args.prod_steps if args.prod_steps is not None else scaled

    n_atoms = args.size ** 3 * _FCC_BASIS

    def calc_factory(use_cell_list: bool = False) -> "CreamCalculator":
        try:
            return CreamCalculator(
                str(pot_path),
                use_cell_list=use_cell_list,
                backend=args.backend,
            )
        except ValueError as exc:
            if "GPU" in str(exc):
                # Explicit user-visible fallback — never hidden by the library.
                print("  [WARN] GPU unavailable — falling back to CPU for this call.")
                return CreamCalculator(
                    str(pot_path),
                    use_cell_list=use_cell_list,
                    backend="cpu",
                )
            raise

    # Tests 10 and 11 need to spin up a CPU-backend calculator independently
    # of the main backend choice (per-atom quantities are CPU-only).
    # Attaching pot_path as a function attribute passes it without changing
    # the factory signature used by all other tests.
    calc_factory._pot_path = str(pot_path)

    eng = calc_factory()._engine
    print_section("CREAM Cu EAM Simulation Suite  (v2)")
    print(f"  Potential  : {pot_path}")
    print(f"  Elements   : {eng.elements}")
    print(f"  Cutoff     : {eng.cutoff:.3f} Å")
    print(f"  Backend    : {eng.backend}")
    print(f"  Supercell  : {args.size}³ × {_FCC_BASIS} = {n_atoms} atoms (FCC)")
    print(f"  Timestep   : {dt} fs")
    print(f"  Friction   : {_friction:.4f} /fs  (Langevin, τ_fric = {1/_friction:.0f} fs)")
    print(f"  τ_thermo   : {_ttime:.0f} fs  (Nosé-Hoover)")
    print(f"  τ_baro     : {_tau_p:.0f} fs  (NPT barostat)")
    print(f"  Diff temps : {list(_diff_temps)} K  (TEST 12)")
    print(f"  VDOS temp  : {args.vdos_temp:.0f} K  (TEST 13)")
    print(f"  NVT temp   : {args.nvt_temp:.0f} K  (TEST 7)")
    print(f"  Output     : {outdir}/")
    print(f"  Skipping   : {skip or '(none)'}")
    if "Cu" not in eng.elements:
        print("[WARN] 'Cu' not in potential elements — results may not match Cu experiment.")

    results: dict = {}

    # ── Tests 1–8: original suite ──────────────────────────────────────────────
    if 1 not in skip and 2 not in skip:
        results["static"] = run_static_tests(calc_factory, outdir, args.size)

    if 3 not in skip:
        run_force_symmetry_test(calc_factory, args.size)

    if 4 not in skip:
        results["nvt_300K"] = run_nvt(
            calc_factory, outdir, n_atoms,
            temperature_K=300.0,
            equil_steps=_equil(10_000),
            prod_steps=_prod(5_000),
            timestep_fs=dt,
            quiet=args.quiet,
            label="300K",
            friction_per_fs=_friction,
        )

    if 5 not in skip:
        results["npt_300K"] = run_npt(
            calc_factory, outdir, n_atoms,
            temperature_K=300.0,
            pressure_GPa=0.0,
            equil_steps=_equil(10_000),   # ≥10×τ_barostat (τ_p=2 ps → 20 ps min)
            prod_steps=_prod(75_000),
            timestep_fs=dt,
            quiet=args.quiet,
        )

    if 6 not in skip:
        results["thermal"] = run_thermal_expansion(
            calc_factory, outdir, n_atoms,
            temperatures=[100.0, 200.0, 300.0, 400.0, 500.0, 600.0,
                          700.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0, 1300.0],
            steps_per_T=int(20_000 * 2.0 / dt),
            equil_steps=_equil(5_000),   # ≥5×τ_baro = 10 ps at default τ_p=2 ps
            timestep_fs=dt,
            quiet=args.quiet,
        )

    if 7 not in skip:
        results["near_melting"] = run_near_melting(
            calc_factory, outdir, n_atoms,
            temperature_K=args.nvt_temp,      # CLI: --nvt-temp
            equil_steps=_equil(1_000),
            prod_steps=_prod(2_500),
            timestep_fs=min(dt, 1.0),
            quiet=args.quiet,
            friction_per_fs=_friction,
        )

    if 8 not in skip:
        results["npt_triclinic"] = run_npt_triclinic(
            calc_factory, outdir,
            size=8,
            temperature_K=300.0,
            pressure_GPa=0.0,
            equil_steps=_equil(2_000),    # NVT pre-equil before shear
            prod_steps=_prod(50_000),
            timestep_fs=dt,
            shear_fraction=0.02,
        )

    # ── Tests 9–13: advanced suite ─────────────────────────────────────────────
    # If TEST 1&2 ran, use the EAM equilibrium a0 for elastic/vacancy tests.
    _eam_a0 = (
        results["static"]["eos"].get("a0_A", CU_EXP["a0_A"])
        if "static" in results and results["static"].get("eos")
        else CU_EXP["a0_A"]
    )

    if 9 not in skip:
        results["elastic"] = run_elastic_constants(
            calc_factory, outdir, size=5,
            a0=_eam_a0,          # EAM equilibrium a0 from TEST 2 EOS fit
        )

    if 10 not in skip:
        results["vacancy"] = run_vacancy_formation(
            calc_factory, calc_factory._pot_path, outdir, size=4,
            a0=_eam_a0,          # shell radii relative to EAM a0
        )

    if 11 not in skip:
        results["eam_fields"] = run_eam_fields(
            calc_factory, calc_factory._pot_path, outdir, size=6,
            timestep_fs=min(dt, 2.0),
            nvt_steps=max(2_000, int(4_000 * 2.0 / dt)),
        )

    if 12 not in skip:
        results["diffusion"] = run_diffusion(
            calc_factory, outdir, n_atoms,
            temperatures_K=_diff_temps,           # CLI: --diff-temps
            npt_equil_steps=20_000,
            nve_equil_steps=10_000,
            prod_steps=100_000,
            timestep_fs=min(dt, 1.0),
        )

    if 13 not in skip:
        results["vdos"] = run_vdos(
            calc_factory, outdir, n_atoms,
            temperature_K=args.vdos_temp,          # CLI: --vdos-temp
            equil_steps=5_000,
            prod_steps=20_000,
            timestep_fs=min(dt, 1.0),
            friction_per_fs=_friction,
        )

    # ── Tests 14–15: stacking fault + heat capacity ────────────────────────────
    if 14 not in skip:
        results["stacking_fault"] = run_stacking_fault(
            calc_factory, outdir,
            nx=4, ny=4, n_layers=18,
            a0=_eam_a0,          # EAM equilibrium a0 from TEST 2 EOS if available
        )

    if 15 not in skip:
        results["heat_capacity"] = run_heat_capacity(
            calc_factory, outdir, n_atoms,
            temperature_K=300.0,
            pressure_GPa=0.0,
            equil_steps=_equil(5_000),
            prod_steps=_prod(50_000),
            timestep_fs=dt,
            quiet=args.quiet,
        )

    if 16 not in skip:
        # Pass ν_D from TEST 13 when available; falls back to exp 7.2 THz otherwise
        _nu_D = (
            results["vdos"]["nu_debye_THz"]
            if "vdos" in results and results["vdos"].get("nu_debye_THz")
            else None
        )
        results["free_energy"] = run_free_energy(
            calc_factory, outdir, n_atoms,
            temperature_K=300.0,
            equil_steps=_equil(5_000),
            prod_steps=_prod(30_000),
            timestep_fs=dt,
            quiet=args.quiet,
            nu_debye_THz=_nu_D,
        )

    if 17 not in skip:
        # Pass ν_D from TEST 13 and a0 from TEST 2 when available
        _nu_D_qha = (
            results["vdos"]["nu_debye_THz"]
            if "vdos" in results and results["vdos"].get("nu_debye_THz")
            else None
        )
        results["qha"] = run_qha(
            calc_factory, outdir, n_atoms,
            a0=_eam_a0,                                  # EAM a0 from TEST 2, or exp
            nu_debye_THz=_nu_D_qha,                      # TEST 13 ν_D, or exp 7.2 THz
            n_ev_points=args.qha_n_ev,
            dV_frac=args.qha_dv_frac,
            n_gruneisen_vols=args.qha_n_grun_vols,       # ≥5 for stable γ_G fit
            gruneisen_equil_steps=args.qha_grun_equil,
            gruneisen_prod_steps=args.qha_grun_prod,     # default 15000 for stable ν_D
            gruneisen_timestep_fs=args.qha_grun_dt,
            gruneisen_temp_K=args.qha_grun_temp,
            T_lo=args.qha_t_lo,
            T_hi=args.qha_t_hi,
            n_T_points=args.qha_n_t,
            target_T=args.qha_target_t,
        )

    write_summary(outdir, results, args.backend)

    try:
        import matplotlib
        matplotlib.use("Agg")
        _plot_results(outdir, results)
    except ImportError:
        print("\n  [INFO] matplotlib not installed — skipping plots.")


if __name__ == "__main__":
    main()