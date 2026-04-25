# CREAM EAM/alloy external reference verification results

This report summarizes verification against an independent Python `.eam.alloy` reference implementation and CREAM's ASE Calculator interface.

The result should be interpreted as **independent LAMMPS-style setfl/EAM-alloy reference comparison**, not direct bitwise comparison with the LAMMPS executable.

Raw logs are intentionally omitted because they can expose local usernames and absolute directory paths. The included JSON files contain only sanitized aggregate result data.

## Potentials

| Potential | Suite | Purpose |
|---|---|---|
| `Cu01.eam.alloy` | elemental | Elemental Cu tests: fcc bulk, strain, vacancy, slab, PBC boundary, cutoff boundary, non-periodic cluster cases |
| `Mishin-Ni-Al-2009.eam.alloy` | binary | Ni-Al alloy tests: pure Ni/Al under binary potential, B2-NiAl, L1_2-like structures, random alloy, antisite, vacancy, atom-order permutation, hetero-pair cutoff cases |

## Cu01.eam.alloy summary

The Cu suite contains 13 deterministic cases. The potential cutoff is `5.506786 A`, so periodic fcc cases use a `4x4x4` conventional-cell supercell.

| Mode | Evaluated | Passed | Failed | Skipped | Max energy error | Max force error | Max force RMS | Max stress error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cpu_allpairs` | 13 | 13 | 0 | 0 | `7.80e-7 eV/atom` | `1.76e-4 eV/A` | `5.42e-5 eV/A` | `5.98e-3 GPa` |
| `cpu_celllist` | 10 | 10 | 0 | 3 | `6.61e-7 eV/atom` | `1.77e-4 eV/A` | `5.44e-5 eV/A` | `5.99e-3 GPa` |
| `gpu_allpairs` | 13 | 13 | 0 | 0 | `7.80e-7 eV/atom` | `1.76e-4 eV/A` | `5.42e-5 eV/A` | `5.99e-3 GPa` |
| `gpu_celllist` | 10 | 10 | 0 | 3 | `7.80e-7 eV/atom` | `1.76e-4 eV/A` | `5.42e-5 eV/A` | `5.99e-3 GPa` |

The skipped cases are non-periodic two-atom cluster tests in cell-list mode. They are covered by all-pairs modes.

## Ni-Al binary summary

The Ni-Al suite contains 22 deterministic cases. With `--atol-energy-per-atom 2e-6`, all evaluated CPU all-pairs cases passed.

| Mode | Evaluated | Passed | Failed | Skipped | Max energy error | Max force error | Max force RMS | Max stress error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cpu_allpairs` | 22 | 22 | 0 | 0 | `1.102e-6 eV/atom` | `1.785e-4 eV/A` | `5.206e-5 eV/A` | `7.17e-3 GPa` |

The largest energy error occurred for `L12_Al3Ni_ordered`:

```text
|dE|/N  = 1.102e-6 eV/atom
max|dF| = 9.347e-6 eV/A
rms|dF| = 3.661e-6 eV/A
max|dS| = 1.894e-5 eV/A^3 = 3.035e-3 GPa
```

The force and stress errors for that case are well below the thresholds.

## Interpretation

The verification covers elemental and binary EAM/alloy paths, including:

```text
periodic fcc bulk
random displacements
homogeneous strain
shear strain
vacancy
surface slab
PBC-boundary cases
cutoff-boundary cases
ordered B2 and L1_2-like alloy structures
antisite defects
random Ni-Al alloy environments
atom-order permutation
Ni-Ni / Al-Al / Ni-Al / Al-Ni pair tests near cutoff
```

The results support the claim that CREAM agrees with an independent LAMMPS-style `.eam.alloy` reference implementation for the tested energy, force, and stress paths within the stated tolerances.
