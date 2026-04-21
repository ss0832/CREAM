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

Usage
-----
  python cu_md_simulation.py --potential Cu01.eam.alloy --backend gpu
  python cu_md_simulation.py --potential Cu01.eam.alloy --backend cpu --size 5
  python cu_md_simulation.py --potential Cu01.eam.alloy --skip 6,7,8
  python cu_md_simulation.py --potential Cu01.eam.alloy --skip 9,10,11,12,13
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
from ase.build import bulk
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet


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

# Experimental reference values for bulk Cu
CU_EXP: dict[str, float] = {
    "a0_A":         3.615,    # Lattice parameter [Å]
    "E_coh_eV":    -3.49,    # Cohesive energy [eV/atom]
    "B0_GPa":       140.0,   # Bulk modulus [GPa]
    "alpha_1perK":  16.6e-6, # Linear thermal expansion coefficient [1/K]
    "T_melt_K":    1358.0,   # Melting point [K]
    # Elastic constants
    "C11_GPa":      168.4,   # [GPa]
    "C12_GPa":      121.4,
    "C44_GPa":       75.4,
    "A_zener":        3.21,  # Zener anisotropy = 2C44/(C11-C12)
    # Defect
    "E_vac_eV":       1.28,  # Vacancy formation energy [eV]
    # Phonon / diffusion
    "nu_Debye_THz":   7.2,   # Debye frequency [THz]
    "theta_D_K":    343.0,   # Debye temperature [K]
    "Ea_diff_eV":     2.04,  # Self-diffusion activation energy [eV]
}

# FCC unit cell (cubic=True) contains 4 atoms
_FCC_BASIS = 4


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

def _make_npt_dyn(atoms, timestep_fs: float, temperature_K: float, pressure_GPa: float):
    """Return an NPT dynamics object.

    Tries ASE NPT implementations in order of preference:
      1. ase.md.melchionna.MelchionnaNPT  (newer ASE)
      2. ase.md.npt.NPT                   (classic ASE)
      3. ase.md.nptberendsen.NPTBerendsen (fallback — not strict NpT)
    """
    dt            = timestep_fs * units.fs
    ttime         = 25.0 * units.fs
    pfactor       = (2000.0 * units.fs) ** 2 * (140.0 * units.GPa)
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
        taut=500.0 * units.fs,
        taup=1000.0 * units.fs,
        compressibility_au=1.0 / (140.0 * units.GPa),
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

    sc_size = max(3, round(n_atoms ** (1.0 / 3.0)))
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
        friction=0.02 / units.fs,
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

    dV2     = float(np.var(volumes_np))
    kBT_ev  = KB_EV * temperature_K
    B_fluct = kBT_ev * V_mean / (dV2 + 1e-30) / EV_PER_A3_TO_GPA

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

        dyn.attach(_rec_vol, interval=10)
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
) -> dict:
    print_section(f"TEST 7  —  Near-melting NVT  T={temperature_K:.0f} K")

    sc_size = max(3, round(n_atoms ** (1.0 / 3.0)))
    atoms   = bulk("Cu", "fcc", a=CU_EXP["a0_A"], cubic=True) * (sc_size, sc_size, sc_size)
    n       = len(atoms)
    atoms.calc = calc_factory(use_cell_list=(n > 300))

    alpha_approx = 17e-6
    scale_factor = 1.0 + alpha_approx * (temperature_K - 300.0) / 3.0
    atoms.set_cell(atoms.get_cell() * scale_factor, scale_atoms=True)

    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K,
                                 rng=np.random.default_rng(99))
    Stationary(atoms)

    dyn = Langevin(
        atoms,
        timestep=timestep_fs * units.fs,
        temperature_K=temperature_K,
        friction=0.05 / units.fs,
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

    msd      = msd_from_trajectory(traj_pos)
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
    prod_steps: int = 5_000,
    timestep_fs: float = 2.0,
) -> dict:
    from ase.md.npt import NPT

    print_section(
        f"TEST 8  —  NPT Triclinic Cell  T={temperature_K:.0f} K  P={pressure_GPa:.1f} GPa"
    )

    atoms   = bulk("Cu", "fcc", a=CU_EXP["a0_A"], cubic=True) * (size, size, size)
    n_atoms = len(atoms)

    cell       = atoms.get_cell().copy()
    cell[0, 1] += 2.0
    cell[1, 2] += 2.0
    atoms.set_cell(cell, scale_atoms=True)

    atoms.calc = calc_factory(use_cell_list=(n_atoms > 300))
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K,
                                 rng=np.random.default_rng(88))
    Stationary(atoms)

    init_angles = atoms.get_cell_lengths_and_angles()[3:]
    print(f"  Initial cell angles: {init_angles}")

    dyn = NPT(
        atoms,
        timestep=timestep_fs * units.fs,
        temperature_K=temperature_K,
        externalstress=pressure_GPa * units.GPa,
        ttime=25.0 * units.fs,
        pfactor=(75.0 * units.fs) ** 2 * (140.0 * units.GPa),
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
    """
    print_section("TEST 9  —  Elastic Constants  (C11, C12, C44)")

    atoms0 = bulk("Cu", "fcc", a=CU_EXP["a0_A"], cubic=True) * (size, size, size)
    N      = len(atoms0)
    use_cl = N > 500
    print(f"  Supercell: {size}³ = {N} atoms")

    # Six strain amplitudes: ±0.5 %, ±1.0 %, ±1.5 %
    deltas = np.array([-0.015, -0.010, -0.005, +0.005, +0.010, +0.015])

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
    """
    print_section("TEST 10  —  Vacancy Formation Energy  (CPU backend)")

    from cream import CreamCalculator

    atoms_perf = bulk("Cu", "fcc", a=CU_EXP["a0_A"], cubic=True) * (size, size, size)
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

    sh1 = dists_vac < 3.2
    sh2 = (dists_vac >= 3.2) & (dists_vac < 4.6)
    sh3 = (dists_vac >= 4.6) & (dists_vac < 6.0)
    far = dists_vac >= 6.0

    def _shell(mask: np.ndarray, label: str) -> None:
        if not mask.any():
            return
        dE = delta_E[mask] * 1000.0
        print(f"    {label:<30s}  n={mask.sum():3d}"
              f"  ΔE_mean={dE.mean():+7.2f} meV"
              f"  ΔE_max={dE.max():+7.2f} meV")

    _shell(sh1, "1st shell  (r < 3.2 Å)")
    _shell(sh2, "2nd shell  (3.2–4.6 Å)")
    _shell(sh3, "3rd shell  (4.6–6.0 Å)")
    _shell(far, "Bulk       (r ≥ 6.0 Å) ")

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

    print_subsection("Thermal snapshot (4 ps NVT @ 300 K)")
    MaxwellBoltzmannDistribution(atoms, temperature_K=300.0,
                                 rng=np.random.default_rng(7))
    Stationary(atoms)

    dyn = Langevin(
        atoms,
        timestep=2.0 * units.fs,
        temperature_K=300.0,
        friction=0.02 / units.fs,
        rng=np.random.default_rng(77),
    )
    dyn.run(2_000)   # 4 ps equilibration

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
    temperatures_K: tuple[float, ...] = (700.0, 900.0, 1100.0, 1300.0),
    equil_steps: int = 2_000,
    prod_steps: int  = 10_000,
    timestep_fs: float = 1.0,
    sample_interval: int = 20,
) -> dict:
    """
    Self-diffusion coefficient D from the Einstein relation:

        D = lim_{t→∞}  MSD(t) / (6t)

    Protocol
    --------
    1. FCC supercell scaled to thermal equilibrium volume.
    2. Short Langevin NVT equilibration to reach target temperature.
    3. Switch to NVE (Velocity Verlet) to remove thermostat artefacts.
    4. Collect unwrapped positions (minimum-image incremental tracking).
    5. Fit D from linear regression on the second half of the MSD.

    Unwrapping
    ----------
    At each sample step the per-atom displacement is evaluated under the
    minimum-image convention and accumulated into a cumulative displacement
    vector, avoiding jump artefacts at periodic boundaries.

    Arrhenius analysis
    ------------------
    ln D = ln D₀ − E_a / (k_B T)
    A linear fit to ln D vs 1/T gives E_a and the pre-exponential D₀.
    """
    print_section("TEST 12  —  Self-Diffusion Coefficient  (NVE Einstein relation)")

    sc_size = max(3, round(n_atoms ** (1.0 / 3.0)))
    atoms0  = bulk("Cu", "fcc", a=CU_EXP["a0_A"], cubic=True) * (sc_size, sc_size, sc_size)
    N       = len(atoms0)
    print(f"  Supercell: {sc_size}³ = {N} atoms   dt = {timestep_fs} fs")
    print(f"  Temperatures: {list(temperatures_K)} K")
    print(f"  NVE production: {prod_steps} steps"
          f" = {prod_steps * timestep_fs / 1000:.1f} ps"
          f"  (sampled every {sample_interval} steps)")

    D_values:  list[float] = []
    inv_T_arr: list[float] = []

    for T in temperatures_K:
        atoms = atoms0.copy()

        alpha = 17.0e-6
        scale = 1.0 + alpha * (T - 300.0) / 3.0
        atoms.set_cell(atoms0.get_cell() * scale, scale_atoms=True)

        atoms.calc = calc_factory(use_cell_list=(N > 300))
        MaxwellBoltzmannDistribution(atoms, temperature_K=T,
                                     rng=np.random.default_rng(int(T)))
        Stationary(atoms)

        nvt = Langevin(atoms,
                       timestep=timestep_fs * units.fs,
                       temperature_K=T,
                       friction=0.05 / units.fs,
                       rng=np.random.default_rng(int(T) + 1))
        nvt.run(equil_steps)

        nve      = VelocityVerlet(atoms, timestep=timestep_fs * units.fs)
        cell_mat = np.array(atoms.get_cell())
        H_inv    = np.linalg.inv(cell_mat)

        prev_pos = atoms.get_positions().copy()
        cumdisp  = np.zeros_like(prev_pos)

        traj_msd: list[float] = []
        times_ps: list[float] = []
        step_counter = [0]

        def _sample(
            _prev=prev_pos, _cum=cumdisp, _hi=H_inv, _cell=cell_mat,
        ) -> None:
            cur  = atoms.get_positions()
            raw  = cur - _prev
            frac = raw @ _hi
            frac -= np.round(frac)
            disp = frac @ _cell
            _cum += disp
            _prev[:] = cur
            msd = float(np.mean(np.sum(_cum ** 2, axis=1)))
            traj_msd.append(msd)
            times_ps.append(
                step_counter[0] * sample_interval * timestep_fs / 1000.0
            )
            step_counter[0] += 1

        nve.attach(_sample, interval=sample_interval)
        nve.run(prod_steps)

        msd_arr = np.array(traj_msd)
        t_arr   = np.array(times_ps)

        half = len(t_arr) // 2
        if half > 4:
            slope_A2_ps = float(np.polyfit(t_arr[half:], msd_arr[half:], 1)[0])
            D = slope_A2_ps / 6.0 * 1e-8   # Å²/ps → m²/s
        else:
            slope_A2_ps = float("nan")
            D = float("nan")

        T_inst = atoms.get_temperature()
        print(f"  T={T:.0f} K (inst={T_inst:.1f} K)"
              f"  MSD_final={msd_arr[-1]:.2f} Å²"
              f"  slope={slope_A2_ps:.3f} Å²/ps"
              f"  D={D:.3e} m²/s")

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
        Ea_eV  = float(-coeffs[0] * KB_EV)
        D0     = float(np.exp(coeffs[1]))
        print_subsection("Arrhenius fit")
        print(f"  ln D = ln D₀ − E_a / (k_B T)")
        print(f"  E_a = {Ea_eV:.3f} eV   D0 = {D0:.3e} m²/s")
        print(compare("Activation energy E_a [eV]", Ea_eV,
                      CU_EXP["Ea_diff_eV"], "eV", tol_pct=30.0))

    save_array(
        outdir / "diffusion_arrhenius.txt",
        "T_K  inv_T_K-1  D_m2s",
        np.column_stack([1.0 / invT_np, invT_np, D_np]),
    )

    return {
        "D_values":    D_values,
        "temperatures": list(temperatures_K),
        "Ea_eV":       Ea_eV,
        "D0_m2s":      D0,
    }


# ── TEST 13: Vibrational DOS via VACF ─────────────────────────────────────────

def run_vdos(
    calc_factory,
    outdir: Path,
    n_atoms: int,
    temperature_K: float = 300.0,
    equil_steps: int    = 2_000,
    prod_steps: int     = 20_000,
    timestep_fs: float  = 1.0,
    sample_interval: int = 5,
) -> dict:
    """
    Vibrational density of states g(ω) via velocity autocorrelation (VACF):

        VACF(t) = <v(0)·v(t)> / <v(0)·v(0)>
        g(ω)    = |FT[VACF(t)]|²   (power spectrum)

    Physical observables
    --------------------
    ν_peak  — dominant phonon branch frequency [THz]
    ν_D     — Debye frequency (90 % cumulative spectral weight) [THz]
    Θ_D     = ħω_D / k_B  [K]

    Protocol
    --------
    1. Short Langevin NVT equilibration (removes startup transients).
    2. NVE production — velocities sampled every sample_interval steps.
       NVE avoids thermostat artefacts on the velocity spectrum.
    3. Hann window applied before FFT to suppress spectral leakage.
    4. Spectrum trimmed to [0, 20] THz (Cu phonons live below ~8 THz).
    """
    print_section(f"TEST 13  —  Vibrational DOS via VACF  T = {temperature_K:.0f} K")

    sc_size      = max(3, round(n_atoms ** (1.0 / 3.0)))
    atoms        = bulk("Cu", "fcc", a=CU_EXP["a0_A"], cubic=True) * (sc_size, sc_size, sc_size)
    N            = len(atoms)
    dt_sample_ps = sample_interval * timestep_fs / 1000.0
    print(f"  Supercell: {sc_size}³ = {N} atoms   dt = {timestep_fs} fs")

    atoms.calc = calc_factory(use_cell_list=(N > 300))
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K,
                                 rng=np.random.default_rng(42))
    Stationary(atoms)

    nvt = Langevin(atoms,
                   timestep=timestep_fs * units.fs,
                   temperature_K=temperature_K,
                   friction=0.02 / units.fs,
                   rng=np.random.default_rng(43))
    nvt.run(equil_steps)
    print(f"  NVT equilibration: {equil_steps} steps"
          f" = {equil_steps * timestep_fs / 1000:.1f} ps  done.")

    nve      = VelocityVerlet(atoms, timestep=timestep_fs * units.fs)
    vel_traj: list[np.ndarray] = []

    def _sample_v() -> None:
        vel_traj.append(atoms.get_velocities().copy())   # (N, 3) Å/fs

    nve.attach(_sample_v, interval=sample_interval)
    n_frames = prod_steps // sample_interval
    print(f"  NVE production:    {prod_steps} steps"
          f" = {prod_steps * timestep_fs / 1000:.1f} ps"
          f"  ({n_frames} VACF frames)")
    nve.run(prod_steps)

    # ── VACF ──────────────────────────────────────────────────────────────────
    vel  = np.stack(vel_traj, axis=0)   # (M, N, 3)
    M    = vel.shape[0]
    v0   = vel[0]
    norm = float(np.mean(v0 * v0))
    vacf = np.array([float(np.mean(vel[t] * v0)) / norm for t in range(M)])

    times_ps = np.arange(M) * dt_sample_ps
    save_array(
        outdir / f"vacf_{int(temperature_K)}K.txt",
        "time_ps  VACF_normalised",
        np.column_stack([times_ps, vacf]),
    )

    # ── Power spectrum via windowed FFT ───────────────────────────────────────
    window   = np.hanning(M)
    vdos_raw = np.abs(np.fft.rfft(vacf * window)) ** 2
    freqs    = np.fft.rfftfreq(M, d=dt_sample_ps)   # THz

    mask       = (freqs > 0.0) & (freqs <= 20.0)
    freqs_plot = freqs[mask]
    vdos_plot  = vdos_raw[mask]

    area = np.trapz(vdos_plot, freqs_plot)
    if area > 0:
        vdos_plot /= area

    peak_idx = int(np.argmax(vdos_plot))
    nu_peak  = float(freqs_plot[peak_idx])

    dnu     = np.gradient(freqs_plot)
    cumspec = np.cumsum(vdos_plot * dnu)
    cumspec /= cumspec[-1]
    debye_idx = int(np.searchsorted(cumspec, 0.90))
    nu_debye  = float(freqs_plot[min(debye_idx, len(freqs_plot) - 1)])

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


# ── Summary report ─────────────────────────────────────────────────────────────

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
        lines += ["", "  Self-Diffusion (NVE Einstein)"]
        for T, D in zip(d["temperatures"], d["D_values"]):
            lines.append(f"    D({T:.0f} K) = {D:.3e} m²/s")
        lines.append(
            f"    Arrhenius: E_a = {d['Ea_eV']:.3f} eV"
            f"  D0 = {d['D0_m2s']:.3e} m²/s"
            f"  (exp E_a ~ {CU_EXP['Ea_diff_eV']:.2f} eV)"
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
        help="Comma-separated test numbers to skip, e.g. --skip 6,7,8  or  --skip 9,10,11,12,13",
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
    print(f"  Potential : {pot_path}")
    print(f"  Elements  : {eng.elements}")
    print(f"  Cutoff    : {eng.cutoff:.3f} Å")
    print(f"  Backend   : {eng.backend}")
    print(f"  Supercell : {args.size}³ × {_FCC_BASIS} = {n_atoms} atoms (FCC)")
    print(f"  Timestep  : {dt} fs")
    print(f"  Output    : {outdir}/")
    print(f"  Skipping  : {skip or '(none)'}")
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
            equil_steps=int(10_000 * 2.0 / dt),
            prod_steps=int(5_000 * 2.0 / dt),
            timestep_fs=dt,
            quiet=args.quiet,
            label="300K",
        )

    if 5 not in skip:
        results["npt_300K"] = run_npt(
            calc_factory, outdir, n_atoms,
            temperature_K=300.0,
            pressure_GPa=0.0,
            equil_steps=int(2_500 * 2.0 / dt),
            prod_steps=int(75_000 * 2.0 / dt),
            timestep_fs=dt,
            quiet=args.quiet,
        )

    if 6 not in skip:
        results["thermal"] = run_thermal_expansion(
            calc_factory, outdir, n_atoms,
            temperatures=[100.0, 200.0, 300.0, 400.0, 500.0, 600.0,
                          700.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0, 1300.0],
            steps_per_T=int(20_000 * 2.0 / dt),
            equil_steps=int(1_000 * 2.0 / dt),
            timestep_fs=dt,
            quiet=args.quiet,
        )

    if 7 not in skip:
        results["near_melting"] = run_near_melting(
            calc_factory, outdir, n_atoms,
            temperature_K=1300.0,
            equil_steps=int(1_000 * 1.0 / dt),
            prod_steps=int(2_500 * 1.0 / dt),
            timestep_fs=min(dt, 1.0),
            quiet=args.quiet,
        )

    if 8 not in skip:
        results["npt_triclinic"] = run_npt_triclinic(
            calc_factory, outdir,
            size=8,
            temperature_K=300.0,
            pressure_GPa=0.0,
            prod_steps=int(50_000 * 2.0 / dt),
            timestep_fs=dt,
        )

    # ── Tests 9–13: advanced suite ─────────────────────────────────────────────
    if 9 not in skip:
        results["elastic"] = run_elastic_constants(
            calc_factory, outdir, size=5,
        )

    if 10 not in skip:
        results["vacancy"] = run_vacancy_formation(
            calc_factory, calc_factory._pot_path, outdir, size=4,
        )

    if 11 not in skip:
        results["eam_fields"] = run_eam_fields(
            calc_factory, calc_factory._pot_path, outdir, size=6,
        )

    if 12 not in skip:
        results["diffusion"] = run_diffusion(
            calc_factory, outdir, n_atoms,
            temperatures_K=(700.0, 900.0, 1100.0, 1300.0),
            equil_steps=2_000,
            prod_steps=10_000,
            timestep_fs=min(dt, 1.0),
        )

    if 13 not in skip:
        results["vdos"] = run_vdos(
            calc_factory, outdir, n_atoms,
            temperature_K=300.0,
            equil_steps=2_000,
            prod_steps=20_000,
            timestep_fs=min(dt, 1.0),
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
