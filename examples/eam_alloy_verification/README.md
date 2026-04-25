# EAM/alloy verification examples for CREAM

This directory contains an independent Python reference implementation for LAMMPS-style `.eam.alloy` / setfl EAM potentials and scripts that compare CREAM's ASE Calculator output against that reference implementation.

The purpose is **external reference verification**. It is separate from CREAM's internal backend-consistency tests. Internal tests should check CPU/GPU and all-pairs/cell-list consistency inside CREAM. This example checks whether CREAM agrees with an independently written Python evaluator for the same `.eam.alloy` energy, force, and stress calculations.

## Files

| File | Purpose |
|---|---|
| `eam_alloy_reference.py` | Independent `.eam.alloy` / setfl parser and EAM evaluator; exposes `EAMAlloyReferenceCalculator` as an ASE Calculator. |
| `compare_against_reference.py` | Generates deterministic Cu or binary-alloy test cases and compares CREAM against the reference implementation. |
| `run_all_backend_matrix.py` | Runs `compare_against_reference.py` over `cpu_allpairs`, `cpu_celllist`, `gpu_allpairs`, and `gpu_celllist`. |
| `VERIFICATION_RESULTS.md` | Human-readable summary of the current verification results. |
| `results/*.json` | Sanitized summary JSON files. Raw command logs are intentionally not included. |
| `potentials/README.md` | Potential-file download and placement notes. |

## What is compared

The same ASE `Atoms` object is evaluated with CREAM and with the Python reference implementation.

```text
energy per atom = |E_CREAM - E_reference| / N
force max error = max_iα |F_CREAM(i,α) - F_reference(i,α)|
force RMS error = RMS over all force components
stress max error = max Voigt-component error
```

The implementation targets LAMMPS-style `eam/alloy` / setfl files. It does not implement funcfl `.eam` or `.eam.fs`.

## Default tolerances

```text
Cu elemental tests:
  energy/atom <= 1.0e-6 eV/atom

Binary alloy tests:
  energy/atom <= 2.0e-6 eV/atom

All tests:
  force max   <= 3.0e-4 eV/A
  force RMS   <= 1.0e-4 eV/A
  stress max  <= 2.0e-3 eV/A^3  ~= 0.320 GPa
  stress RMS  <= 1.0e-3 eV/A^3  ~= 0.160 GPa
```

The binary alloy energy tolerance is slightly looser because binary EAM/alloy tests exercise additional species mapping, donor-density, and cross-pair table paths.

## Reproduction

Elemental Cu:

```powershell
python .\compare_against_reference.py .\Cu01.eam.alloy --backend cpu --suite elemental
```

Binary Ni-Al:

```powershell
python .\compare_against_reference.py .\Mishin-Ni-Al-2009.eam.alloy `
  --backend cpu `
  --suite binary `
  --binary-elements Ni Al `
  --atol-energy-per-atom 2e-6
```

Full backend matrix:

```powershell
python .\run_all_backend_matrix.py .\Cu01.eam.alloy .\Mishin-Ni-Al-2009.eam.alloy
```

Arguments after `--` are forwarded to `compare_against_reference.py`:

```powershell
python .\run_all_backend_matrix.py .\Mishin-Ni-Al-2009.eam.alloy -- --suite binary --binary-elements Ni Al
```

## Backend coverage note

The current GPU cell-list backend is intended for periodic systems. Non-periodic two-atom cluster cases are evaluated in all-pairs modes and skipped in cell-list modes.

## Notes

- This is not a direct LAMMPS executable comparison.
- It is an independent Python reference implementation following LAMMPS-style setfl/EAM-alloy conventions.
- Raw logs are not committed because they can contain local absolute paths. Only sanitized JSON summaries and human-readable result summaries are included.
