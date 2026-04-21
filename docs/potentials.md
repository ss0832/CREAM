# Obtaining EAM potential files

CREAM reads the standard LAMMPS `setfl` format, commonly named with the
`.eam.alloy` extension. No potential files are bundled with the public
release; you download the one you need from the
**NIST Interatomic Potentials Repository**:

<https://www.ctcms.nist.gov/potentials/>

The NIST repository is the authoritative source: every file is reviewed,
versioned, and annotated with the reference publication and a set of
validation plots (cohesive energy, lattice parameter, elastic constants).

## Common starting points

| Element set | File name                                     | NIST search                    |
|-------------|-----------------------------------------------|--------------------------------|
| Cu          | `Cu01.eam.alloy`                              | NIST Cu single-element page    |
| Al          | `Al99.eam.alloy`, `Mishin-Al-Y-2006.eam.alloy` | NIST Al or Al-alloy pages      |
| Ni-Al       | `Mishin-Ni-Al-2009_eam.alloy`                 | NIST Ni-Al binary page         |
| Fe-Ni       | `Bonny-Fe-Ni-2009.eam.alloy`                  | NIST Fe-Ni binary page         |

Exact file names evolve; always refer to the live NIST pages for the
current link.

## Expectations on the file

- **Format**: LAMMPS setfl. Single-element (`.eam`) variants from some
  repositories are **not** supported; only the `.eam.alloy` multi-element
  setfl layout is parsed.
- **Element order**: the order elements appear on the header line of the
  `.eam.alloy` file is the order CREAM assigns `atom_types` indices.
  `atom_types[i] = 0` refers to the first element in that list.
- **Cutoff and grid**: read from the file header. CREAM makes no
  assumption beyond what the file declares.
- **Units**: all EAM potentials in NIST's repository are in LAMMPS
  `metal` units — distances in Å, energies in eV. CREAM's outputs are
  in the same units (forces in eV/Å, energies in eV).

## Using the file

Pass the path directly to the calculator or engine:

```python
from cream import CreamCalculator
atoms.calc = CreamCalculator("Mishin-Ni-Al-2009_eam.alloy", use_cell_list=True)
```

```rust
let pot = cream::potential::eam::EamPotential::from_file(
    std::path::Path::new("Mishin-Ni-Al-2009_eam.alloy"),
)?;
```

The element order in your `Atoms` object (Python) or `atom_types` array
(Rust) must match the file's header order. If the file lists `"Ni Al"`
then `atom_types[i] = 0` means Ni and `atom_types[i] = 1` means Al.

## License / citation

NIST potential files are typically released under the author's chosen
terms and carry a **citation requirement** for the corresponding
publication. Consult the NIST page for each potential before
redistributing or publishing results. CREAM does not redistribute any
potential file.
