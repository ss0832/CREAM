# CREAM — Compute-shader Rust EAM Atomistics

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19708613.svg)](https://doi.org/10.5281/zenodo.19708613)

GPU-accelerated molecular-dynamics force engine based on the Embedded Atom
Method (EAM). CREAM takes atomic coordinates, element types, and an EAM
potential, and returns energies and forces. It supports multi-element alloys
(`.eam.alloy`) and runs on native GPU (Vulkan / Metal / DX12) from a single
Rust codebase.

*These codes were developed using LLMs, mainly Claude.

## What it is — and is not

CREAM is a pure force engine. It does not advance time, visualize trajectories,
integrate the equations of motion, or apply thermostats. Integration and
analysis happen in the host application; CREAM supplies the per-step forces
and energies the integrator needs. The recommended integration point for
atomistic-simulation users is the ASE `Calculator` interface, included in the
Python package.

## Feature matrix

| Capability                                             | Status               |
|--------------------------------------------------------|----------------------|
| Single-element and multi-element `.eam.alloy` parsing  | Supported            |
| Orthorhombic PBC                                       | Supported            |
| Triclinic PBC                                          | Supported            |
| Non-periodic (cluster) calculations                    | Supported            |
| O(N²) AllPairs GPU backend                             | Supported            |
| O(N) Cell List with CPU-built neighbour list           | Supported            |
| O(N) Cell List fully on GPU (`cellist_gpu` feature)    | Supported            |
| CPU-only engine for reference and GPU-less hosts       | Supported            |
| Python / ASE bindings                                  | Supported            |

## Getting an EAM potential

CREAM consumes standard LAMMPS `setfl` (`.eam.alloy`) potential files. The
canonical source is the **NIST Interatomic Potentials Repository**:

- <https://www.ctcms.nist.gov/potentials/>

Download the `.eam.alloy` file for your element(s) from NIST and pass its
path to CREAM. Common starting points:

| Potential file                  | Elements | NIST page                                  |
|---------------------------------|----------|--------------------------------------------|
| `Cu01.eam.alloy`                | Cu       | NIST Cu single-element potentials          |
| `Mishin-Ni-Al-2009.eam.alloy`   | Ni, Al   | NIST Ni-Al binary alloy potentials         |

No potential files are bundled with the public release. Make sure the file
you download matches the element order you pass to the calculator or
engine (CREAM's `atom_types` indices are 0-based into that order).

## Quick start — Python + ASE

```bash
pip install eam-cream-python
```

CREAM is distributed on PyPI as a **source distribution** (sdist) built
with `maturin sdist`. A `pip install` therefore compiles the Rust code
on your machine the first time; expect two or three minutes for the
initial build while `wgpu` and its transitive dependencies finish
compiling. Subsequent installs into the same Python environment reuse
the compiled cache.

What you need on your machine:

- A **Rust toolchain** (stable). The easiest install is
  [rustup](https://rustup.rs); any Rust ≥ 1.75 works.
- A C compiler that Rust's `cc` crate can find (`cc`, `clang`, or MSVC
  on Windows). Rustup's default install bundles what you need.
- **ASE** is installed automatically as a runtime dependency.

The build is configured (via `pyproject.toml`) to compile the GPU
cell-list pipeline by default (`--features python,cellist_gpu`), so
you get a wheel with the full GPU backend without having to set
anything on the command line.

### GPU backend (default)

This is the path you want whenever a GPU is available. It drives the
Rust + `wgpu` engine, the four-pass compute pipeline, and the Morton-
ordered cell list, all on the GPU.

```python
from ase.build import bulk
from cream import CreamCalculator

atoms = bulk("Cu", "fcc", a=3.615) * (10, 10, 10)  # 4 000 atoms

# `backend="gpu"` is the default; the argument is shown here for clarity.
# `use_cell_list=True` selects the O(N) Morton cell-list pipeline.
# Without it, AllPairs (O(N²)) is used — fine for N ≲ 2 000, too slow above.
atoms.calc = CreamCalculator(
    "Cu01.eam.alloy",
    backend="gpu",
    use_cell_list=True,
)

energy = atoms.get_potential_energy()     # eV          (GPU)
forces = atoms.get_forces()               # eV/Å, (N,3) (GPU)
per_atom = atoms.get_potential_energies() # eV, (N,)    (GPU)

# Verify which backend is actually active.
print(atoms.calc.backend)   # → "gpu"
```

If no GPU adapter is available on the host, constructing a
`CreamCalculator` with `backend="gpu"` raises a `ValueError` with
a message starting `GPU device lost:`. The fail-fast behaviour is
deliberate — a silent fallback to CPU in the middle of an MD
production run would hide 10–100× performance regressions. Users
running on GPU-less hosts should select the CPU backend explicitly
(see below).

### Check what the engine loaded

`CreamCalculator` exposes the parsed potential metadata directly. This
is the recommended way to confirm element order before building your
`Atoms` object.

```python
calc = CreamCalculator("Mishin-Ni-Al-2009_eam.alloy", use_cell_list=True)

# Low-level engine is always accessible.
eng = calc._engine
print(eng.elements)     # → ["Ni", "Al"]  — atom_types indexing order
print(eng.n_elements)   # → 2
print(eng.cutoff)       # → 6.5   (Å, from the file header)
print(eng.backend)      # → "gpu"
```

### CPU backend — no GPU required

Use the CPU backend on machines without a GPU, in CI runners, or when
you need bit-for-bit reproducibility across reruns (the CPU path is
deterministic; the GPU path is subject to floating-point accumulation
order inside workgroups).

```python
atoms.calc = CreamCalculator(
    "Cu01.eam.alloy",
    backend="cpu",
    use_cell_list=True,
)

# Same call surface — only the compute device changes.
e = atoms.get_potential_energy()
```

The CPU backend is rayon-parallel and uses the same Morton cell list as
the GPU path when `use_cell_list=True`. It scales well to several
million atoms on a modern workstation CPU; it is several × slower than
the GPU path on the same hardware for the same system.

### Low-level engine — no ASE required

If you are writing your own integrator, or wrapping CREAM from a
non-ASE workflow, call `CreamEngine` directly. It takes raw NumPy
arrays and returns energy, forces, and per-atom energies.

```python
import numpy as np
from cream import CreamEngine

# Build the engine once — pipelines and potential tables are cached.
engine = CreamEngine(
    "Cu01.eam.alloy",
    use_cell_list=True,
    backend="gpu",     # "gpu" (default) or "cpu"
)

# Inputs:
#   positions  : (N, 3) float64, Cartesian coordinates in Å
#   atom_types : (N,)   int32,   indices into engine.elements
#   cell       : (3, 3) float64, lattice vectors as rows (ASE/LAMMPS
#                convention), or None for a non-periodic cluster
positions  = np.array([[0.0, 0.0, 0.0],
                       [1.8, 1.8, 0.0]], dtype=np.float64)
atom_types = np.array([0, 0], dtype=np.int32)
cell       = np.diag([10.0, 10.0, 10.0])

energy, forces, energy_per_atom = engine.compute(
    positions, atom_types, cell,
)
# energy          : float, eV
# forces          : (N, 3) float64, eV/Å
# energy_per_atom : (N,)   float64, eV (sum ≈ energy)
```

The engine releases Python's GIL during `compute()`, so driving many
engines from threads for ensemble sampling is safe and scales with the
number of GPUs and CPU cores available.

### Debug access to cell-list internals (GPU backend only)

For profiling or bug investigations the engine exposes
`compute_with_debug`, which returns every intermediate buffer of the
cell-list pipeline:

```python
energy, forces, energy_per_atom, debug = engine.compute_with_debug(
    positions, atom_types, cell,
)
# debug is None for backend="cpu" or when use_cell_list is False.
# Otherwise a dict with keys:
#   n_atoms, n_cells, n_cells_pad, n_morton
#   cell_ids              (N,)          uint32
#   sorted_atoms          (N,)          uint32
#   cell_start            (n_morton+1,) uint32
#   cell_counts           (n_morton,)   uint32
#   reordered_positions   (N, 4)        float32
#   reordered_types       (N,)          uint32
#   densities             (N,)          float32
#   debug_flags           (32,)         uint32
```

### Choosing backend and neighbour strategy

| System size          | Recommended settings                                           |
|----------------------|----------------------------------------------------------------|
| N ≲ 2 000            | `backend="gpu"`, `use_cell_list=False` (AllPairs)              |
| 2 000 ≲ N ≲ 10⁴       | `backend="gpu"`, `use_cell_list=True`                          |
| 10⁴ ≲ N              | `backend="gpu"`, `use_cell_list=True` (mandatory for speed)    |
| any N, no GPU host   | `backend="cpu"`, `use_cell_list=True`                          |
| debugging / CI       | `backend="cpu"` for determinism                                |

Multi-element alloys work the same way — supply the corresponding
`.eam.alloy` file and make sure the element order in the file matches
the order of species in your `Atoms` object.

## Quick start — Rust

```rust
use cream::{
    engine::ComputeEngine,
    potential::{eam::EamPotential, NeighborStrategy},
    ortho_cell,
};
use std::path::Path;

let mut engine = pollster::block_on(
    ComputeEngine::new(NeighborStrategy::AllPairs)
)?;
let potential = EamPotential::from_file(Path::new("Cu01.eam.alloy"))?;

let positions  = vec![[0.0f32, 0.0, 0.0, 0.0], [1.8, 1.8, 0.0, 0.0]];
let atom_types = vec![0u32, 0];
let cell       = Some(ortho_cell(10.0, 10.0, 10.0));

let result = engine.compute_sync(&positions, &atom_types, cell, &potential)?;
println!("Energy: {:.6} eV", result.energy);
```

## How the pipeline is organised

CREAM dispatches four GPU compute passes per force evaluation. With the cell
list enabled, three additional short passes build the neighbour structure first:

| Pass                        | Role                                                       |
|-----------------------------|------------------------------------------------------------|
| `cell_pass0a_assign`        | Assign each atom a Morton (Z-order) cell code              |
| `cell_pass0b_sort`          | Count atoms per cell via atomic increment                  |
| CPU prefix sum              | Compute `cell_start[]` boundaries over Morton space        |
| `cell_pass0c_range`         | Scatter atoms into sorted order                            |
| `eam_pass1_*`               | Electron density `ρᵢ = Σⱼ fβ(r)` (Kahan-compensated)       |
| `eam_pass2_*`               | Embedding derivative, forces, and per-atom pair energies   |
| `eam_pass3_reduce`          | Tree-reduction of the energy (falls back to CPU for N ≥ 4096) |

Morton-code ordering improves GPU L2 cache hit rates in passes 1 and 2
because spatially adjacent cells map to adjacent indices in the sorted buffer.

All EAM tables (`f_α(r)`, `F_α(ρ)`, `F'_α(ρ)`, `φ_αβ(r)`, `f'_α(r)`,
`φ'_αβ(r)`) are packed into a single storage buffer so that every shader
stage stays within WebGPU's eight-binding limit.

Detailed architecture documentation lives in [`docs/`](docs/) and renders on
[Read the Docs](https://cream.readthedocs.io/).

## Periodic boundary conditions

CREAM supports both orthorhombic and fully triclinic cells. The cell matrix
`H` stores lattice vectors as rows (the ASE / LAMMPS convention):

```
H = [[ax, ay, az],   ← a
     [bx, by, bz],   ← b
     [cx, cy, cz]]   ← c
```

The minimum-image convention is applied in fractional coordinates
(`s = d · H⁻¹ − round(s)`, then back to Cartesian). Passing `None` (Rust) or
`cell=None` (Python) selects the non-periodic cluster path.

## Build from source

### Prerequisites

| Tool     | Version                            |
|----------|------------------------------------|
| Rust     | stable (see `rust-toolchain.toml`) |
| Python   | ≥ 3.9                              |
| maturin  | ≥ 1.5, < 2.0                       |
| ASE      | ≥ 3.22 (installed automatically with the Python package) |

### Editable install with Python bindings

```bash
git clone https://github.com/ss0832/cream.git
cd cream
maturin develop --release --features python,cellist_gpu
```

`--release` is recommended even during development: the CPU engine is ~10×
slower in debug mode and some tests time out without it. The
`cellist_gpu` feature is baked into the default wheel configuration
(`pyproject.toml` → `[tool.maturin].features`); you only need to name it
explicitly when invoking `maturin develop` by hand.

### Running the test suite

```bash
# Python / ASE tests (no GPU needed)
pytest tests_python/ -v

# Rust unit and integration tests (CPU-only)
cargo test --release

# Include GPU tests (requires a GPU or software rasterizer such as llvmpipe / WARP)
cargo test --release -- --include-ignored
```

### Code quality

```bash
# Rust
cargo fmt                                                  # format
cargo clippy --features python,cellist_gpu -- -D warnings  # lint (zero warnings)

# Python
ruff check python/ tests_python/ --fix
ruff format python/ tests_python/
mypy python/cream/ --ignore-missing-imports --no-site-packages
```

### Building distributable wheels

```bash
maturin build --release   # binary wheel for the current platform
maturin sdist              # source distribution
twine upload target/wheels/cream_python-*.tar.gz
```

`maturin build` picks up the feature list from `pyproject.toml`
(`python,cellist_gpu`), so no `--features` flag is needed at release
time.

The package name on PyPI is **`eam-cream-python`**. Install with
`pip install eam-cream-python`; import as `from cream import ...`.

## Benchmarking and diagnostics

CREAM ships with a suite of diagnostic binaries under `src/bin/`. They are
not part of the library's public API but can be run directly to reproduce
published performance numbers or to investigate a suspected regression.

| Binary              | What it does                                                                    |
|---------------------|---------------------------------------------------------------------------------|
| `bench_cpu`         | CPU engine throughput across a range of atom counts                             |
| `bench_gpu`         | GPU vs CPU wall-clock sweep over 7 crystal systems × 23 sizes (up to ~10⁷ atoms) |
| `bench_nial`        | Ni-Al alloy potential benchmark (2-element EAM)                                 |
| `repro_cpu_cl`      | Focused CPU AllPairs vs CPU CellList agreement sweep (minimal regression probe) |
| `diagnose_cellist`  | Per-case comparison of GPU and CPU cell-list paths with shader counter dumps    |
| `diagnose_mismatch` | Regression guard for historical cubic / tetragonal / orthorhombic MISMATCH cases |
| `diagnose_nial`     | 4-way comparison (CPU-AP / CPU-CL / GPU-AP / GPU-CL) on Mishin Ni-Al 2009       |
| `profile_gpu_v2`    | GPU workload metrics and per-phase wall-clock breakdown                         |

Typical invocation:

```bash
cargo run --release --bin bench_cpu
cargo run --release --features cellist_gpu --bin bench_gpu
```

Benchmarks that require a 2-element EAM potential (`bench_nial`,
`diagnose_nial`) expect `Mishin-Ni-Al-2009_eam.alloy` at the project root —
download it from NIST first (see **Getting an EAM potential** above).

### Known limits

- `bench_gpu` at N ≳ 10⁷ can exhaust host RAM because the harness retains
  force and reference arrays in double precision for multiple backends
  simultaneously. This is a benchmark-harness limitation, not an engine
  limit; the engine itself has been exercised up to N ≈ 1.6 × 10⁷ atoms.
- The fully GPU-resident cell-list path (`--features cellist_gpu`)
  subdivides its atom-count dispatch at `N = 4 194 240` (the WebGPU
  per-dimension workgroup limit × 64). At that boundary there is a small
  non-monotone step in wall-clock time; see `docs/known_limits.md`.

## Roadmap

The v0.1.0 release freezes the current public API and data structures.
Work planned for subsequent releases:

- Browser / WebAssembly entry point — a WebGPU build is implemented in the
  source tree but has not yet been validated end-to-end on the current
  shader set, so it is not advertised as a supported path for v0.1.0.
- Command-line front-end (`cream-cli`) — present in the source tree but
  similarly not yet validated against the current pipeline; it will be
  re-documented once that work completes.

## Project layout

```
cream/
├── src/
│   ├── lib.rs              # Crate root, re-exports
│   ├── error.rs            # CreamError (thiserror)
│   ├── engine.rs           # ComputeEngine — GPU 4-pass dispatch
│   ├── cpu_engine.rs       # CpuEngine — rayon-parallel CPU fallback
│   ├── cell_list.rs        # Morton-ordered cell list (shared CPU/GPU helpers)
│   ├── neighbor_list.rs    # CSR neighbour list used by the default CellList path
│   ├── reference.rs        # O(N²) f64 CPU reference with PBC helpers
│   ├── python.rs           # PyO3 bindings (feature = "python")
│   ├── wasm.rs             # WASM bindings (experimental, not advertised)
│   ├── potential/
│   │   ├── mod.rs          # GpuPotential trait, NeighborStrategy
│   │   └── eam.rs          # EamPotential + .eam.alloy parser
│   ├── shaders/            # WGSL shaders (common + pass0/1/2/3 variants)
│   └── bin/                # Benchmarks and diagnostic tools
├── cream-cli/              # CLI frontend (experimental, not advertised)
├── python/cream/           # ASE Calculator wrapper
├── examples/               # Small example programs
├── tests/                  # Rust integration tests
├── tests_python/           # Python / PyO3 tests
├── docs/                   # Sphinx + MyST documentation source
├── pyproject.toml          # maturin build config
├── Cargo.toml
└── rust-toolchain.toml
```

## References

### ASE

- A. H. Larsen, J. J. Mortensen, J. Blomqvist, I. E. Castelli, R. Christensen, M. Dułak, J. Friis, M. N. Groves, B. Hammer, C. Hargus, E. D. Hermes, P. C. Jennings, P. B. Jensen, J. Kermode, J. R. Kitchin, E. L. Kolsbjerg, J. Kubal, K. Kaasbjerg, S. Lysgaard, J. B. Maronsson, T. Maxson, T. Olsen, L. Pastewka, A. Peterson, C. Rostgaard, J. Schiøtz, O. Schütt, M. Strange, K. S. Thygesen, T. Vegge, L. Vilhelmsen, M. Walter, Z. Zeng, and K. W. Jacobsen,  
  "The atomic simulation environment — a Python library for working with atoms,"  
  *J. Phys.: Condens. Matter* **29**, 273002 (2017).  
  DOI: [10.1088/1361-648X/aa680e](https://doi.org/10.1088/1361-648X/aa680e)

- S. R. Bahn and K. W. Jacobsen,  
  "An object-oriented scripting interface to a legacy electronic structure code,"  
  *Comput. Sci. Eng.* **4**, 56–66 (2002).  
  DOI: [10.1109/5992.998641](https://doi.org/10.1109/5992.998641)

### Embedded Atom Method (EAM)

- M. S. Daw and M. I. Baskes,  
  "Embedded-atom method: Derivation and application to impurities, surfaces, and other defects in metals,"  
  *Phys. Rev. B* **29**, 6443–6453 (1984).  
  DOI: [10.1103/PhysRevB.29.6443](https://doi.org/10.1103/PhysRevB.29.6443)

- S. M. Foiles, M. I. Baskes, and M. S. Daw,  
  "Embedded-atom-method functions for the fcc metals Cu, Ag, Au, Ni, Pd, Pt, and their alloys,"  
  *Phys. Rev. B* **33**, 7983–7991 (1986).  
  DOI: [10.1103/PhysRevB.33.7983](https://doi.org/10.1103/PhysRevB.33.7983)

- M. S. Daw, S. M. Foiles, and M. I. Baskes,  
  "The embedded-atom method: a review of theory and applications,"  
  *Mater. Sci. Rep.* **9**, 251–310 (1993).  
  DOI: [10.1016/0920-2307(93)90001-U](https://doi.org/10.1016/0920-2307(93)90001-U)

### EAM potential files used in examples and benchmarks

- Y. Mishin, M. J. Mehl, and D. A. Papaconstantopoulos,  
  "Embedded-atom potential for B2-NiAl,"  
  *Phys. Rev. B* **65**, 224114 (2002).  
  DOI: [10.1103/PhysRevB.65.224114](https://doi.org/10.1103/PhysRevB.65.224114)

---

## License

CREAM is **dual-licensed**. The Rust engine, PyO3 bindings, WGSL shaders,
and documentation are MIT-licensed. The ASE `Calculator` subclass in
`python/cream/calculator.py` is LGPL-3.0-or-later because it derives from
ASE (`ase.calculators.calculator.Calculator`), which itself is LGPL-3.0.

| Component                                              | License                           |
|--------------------------------------------------------|-----------------------------------|
| Core Rust library (`src/`)                             | [MIT](LICENSE)                    |
| Python bindings (`src/python.rs`)                      | [MIT](LICENSE)                    |
| WGSL shaders (`src/shaders/`)                          | [MIT](LICENSE)                    |
| Documentation (`docs/`, `README.md`)                   | [MIT](LICENSE)                    |
| ASE Calculator wrapper (`python/cream/calculator.py`)  | [LGPL-3.0-or-later](LICENSE.LGPL) |

The effect of this dual licensing on the distributed Python package
(`cream-python`) is that the binary wheel contains MIT-licensed compiled
code **plus** an LGPL-3.0-or-later Python file. If you redistribute or
link against `cream-python`, you must comply with the LGPL-3.0-or-later
terms for the `calculator.py` component — in particular, any modifications
to `calculator.py` must be made available under LGPL-3.0-or-later, and
users of your redistributed package must be able to relink against a
modified `calculator.py`. The LGPL does **not** require your own
application code to be open-sourced.

See [LICENSE](LICENSE) and [LICENSE.LGPL](LICENSE.LGPL) for the full
license texts.
