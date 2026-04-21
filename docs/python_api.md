# Python API

The `cream-python` package exposes two objects:

- **`cream.CreamCalculator`** — an ASE `Calculator` subclass, the
  recommended entry point for most simulation workflows.
- **`cream.CreamEngine`** — the low-level Rust-backed engine, for when
  you are not using ASE or you want direct access to the compute call.

Both run on the GPU by default (`backend="gpu"`) and both accept a
`backend="cpu"` fallback for machines without a usable GPU.

## `CreamCalculator` — ASE `Calculator`

```python
from cream import CreamCalculator

calc = CreamCalculator(
    potential_file,        # str | pathlib.Path — .eam.alloy file
    *,
    use_cell_list=False,   # True → O(N) Morton cell list; False → O(N²)
    cell_size=None,        # Cell edge length (Å); None → potential cutoff
    backend="gpu",         # "gpu" (default) or "cpu"
    label="cream",
)
```

### Typical use

```python
from ase.build import bulk
from cream import CreamCalculator

atoms = bulk("Cu", "fcc", a=3.615) * (10, 10, 10)   # 4 000 atoms

atoms.calc = CreamCalculator(
    "Cu01.eam.alloy",
    backend="gpu",         # explicit; GPU is the default
    use_cell_list=True,    # required for N ≳ a few thousand
)

energy   = atoms.get_potential_energy()       # eV          (GPU)
forces   = atoms.get_forces()                 # eV/Å, (N,3) (GPU)
stress   = atoms.get_stress()                 # eV/Å³, Voigt (6,) (GPU)
per_atom = atoms.get_potential_energies()     # eV, (N,)    (CPU only)
stresses = atoms.get_stresses()               # eV/Å³, (N,6) (CPU only)
```

### Backend selection

```python
# GPU (default) — the recommended path when a GPU is available.
atoms.calc = CreamCalculator("Cu01.eam.alloy", backend="gpu",
                             use_cell_list=True)

# CPU fallback — always available, no GPU needed.
atoms.calc = CreamCalculator("Cu01.eam.alloy", backend="cpu",
                             use_cell_list=True)

# Inspect what the calculator is actually using.
print(atoms.calc.backend)   # "gpu" or "cpu"
```

`backend="cpu"` picks the rayon-parallel CPU engine in the Rust crate.
It is **deterministic** across reruns (the GPU path is not, because of
floating-point accumulation order inside workgroups), so it is what
you want for CI runs and for reproducibility-sensitive workflows.

### Backend-dependent properties

Not all properties are available on every backend. `implemented_properties` is
set at construction time to reflect the active backend, so requesting an
unavailable property raises ASE's standard `PropertyNotImplementedError` rather
than returning incorrect data silently.

| Property                    | GPU | CPU |
|-----------------------------|-----|-----|
| `energy`                    | yes | yes |
| `forces`                    | yes | yes |
| `stress`                    | yes | yes |
| `energies` (per-atom)       | no  | yes |
| `stresses` (per-atom)       | no  | yes |
| `cream_densities`           | no  | yes |
| `cream_embedding_energies`  | no  | yes |

The CPU-only per-atom properties are accessed via the ASE results dict after a
compute call:

```python
atoms.calc = CreamCalculator("Cu01.eam.alloy", backend="cpu",
                             use_cell_list=True)

energy   = atoms.get_potential_energy()   # eV
forces   = atoms.get_forces()             # eV/Å, (N, 3)
stress   = atoms.get_stress()             # Voigt (6,) eV/Å³

# CPU-only extras
energies  = atoms.get_potential_energies()  # eV, (N,) per-atom energy
stresses  = atoms.get_stresses()            # eV/Å³, (N, 6) per-atom stress

densities          = atoms.calc.results["cream_densities"]           # (N,) float64
embedding_energies = atoms.calc.results["cream_embedding_energies"]  # (N,) float64
```

The per-atom stress follows the equal-partition convention:
`σ_i = −N · virial_i / V`, so summing over all atoms recovers the global stress.
The raw atomic virial (eV, LAMMPS convention) is also stored under
`atoms.calc.results["cream_atomic_virials"]` for downstream tools that prefer it.

### When no GPU is available

Constructing a `CreamCalculator` with `backend="gpu"` on a host with
no usable GPU adapter raises `ValueError("GPU device lost: …")`
immediately. CREAM never silently falls back to CPU — a hidden
fallback mid-simulation would mask 10–100× performance regressions
that are impossible to recover after-the-fact in an MD production run.

For GPU-less hosts and for CI, request the CPU backend explicitly:

```python
try:
    atoms.calc = CreamCalculator("Cu01.eam.alloy", backend="gpu",
                                 use_cell_list=True)
except ValueError as exc:
    if "GPU" in str(exc):
        # Graceful fallback under the application's control, not the library's.
        atoms.calc = CreamCalculator("Cu01.eam.alloy", backend="cpu",
                                     use_cell_list=True)
    else:
        raise
```

This pattern — **explicit** in the user's code, not hidden in the
library — preserves the observability of the backend choice while
still letting one codebase run on mixed GPU/CPU hosts.

### Introspecting the loaded potential

The `CreamCalculator` holds a `CreamEngine` internally, and the engine
exposes the potential metadata after parsing:

```python
calc = CreamCalculator("Mishin-Ni-Al-2009_eam.alloy", use_cell_list=True)

eng = calc._engine              # cream.CreamEngine
print(eng.elements)             # ['Ni', 'Al']
print(eng.n_elements)           # 2
print(eng.cutoff)               # 6.5 — Å, from the file header
print(eng.backend)              # 'gpu'
```

The order of `eng.elements` defines how ASE atomic numbers are mapped
to CREAM's internal `atom_types` indices. For the file above,
`atom_types[i] = 0` means Ni and `atom_types[i] = 1` means Al.

### Periodic boundary conditions

- Fully periodic: `atoms.pbc = True` (equivalently `[True, True, True]`).
  The engine reads `atoms.get_cell()` and applies the minimum-image
  convention on every pair.
- Fully non-periodic: `atoms.pbc = False`. The engine treats the system
  as a finite cluster.
- **Partially periodic** (e.g. `[True, True, False]` for slab geometries)
  is **rejected** at compute time with `ValueError`. For slab
  geometries, add a vacuum layer and set `pbc=False`.

### CPU backend — thread warm-up

The rayon thread pool that powers the CPU backend parks its worker
threads after ~100 ms of inactivity. `CreamCalculator` keeps them
warm automatically via an internal keep-alive thread, so mid-loop
latency spikes do not appear during MD production runs.

For the very first call, the pool may not yet be fully spun up. If
sub-millisecond first-call latency matters (benchmarking a single
step, for instance), add one explicit warm-up call before the timed
section:

```python
calc = CreamCalculator("Cu01.eam.alloy", backend="cpu",
                       use_cell_list=True)
atoms.calc = calc
atoms.get_potential_energy()   # warm-up — discarded
# Timed section starts here:
for step in md_loop:
    atoms.get_potential_energy()
```

## `CreamEngine` — low-level engine

Use this when you do not need ASE — for custom integrators, for
embedding in non-ASE pipelines, or for ensemble sampling where you
drive many engines from threads.

```python
from cream import CreamEngine

engine = CreamEngine(
    potential_file,        # str — .eam.alloy file
    *,
    use_cell_list=False,   # bool
    cell_size=None,        # float | None
    backend="gpu",         # "gpu" | "cpu"
)
```

### `compute`

```python
energy, forces, energy_per_atom = engine.compute(
    positions,    # (N, 3) float64, Å, Cartesian
    atom_types,   # (N,)   int32,   indices into engine.elements
    cell,         # (3, 3) float64 | None — lattice rows or None
)
```

The return tuple:

- `energy` — Python `float`, eV.
- `forces` — NumPy `(N, 3)` float64, eV/Å.
- `energy_per_atom` — NumPy `(N,)` float64, eV. The sum equals
  `energy` up to f32-to-f64 rounding.

The cell matrix follows the ASE / LAMMPS convention (lattice vectors as
rows). Pass `cell=None` for a non-periodic cluster.

`compute()` releases the GIL for the full duration of the GPU or CPU
compute, so it is safe to call from multiple Python threads in
parallel.

### `compute_stress`

Like `compute`, but also returns the stress tensor. Available on both backends.

```python
energy, forces, energy_per_atom, stress_voigt = engine.compute_stress(
    positions,    # (N, 3) float64, Å, Cartesian
    atom_types,   # (N,)   int32
    cell,         # (3, 3) float64 | None
)
```

The return tuple:

- `energy` — Python `float`, eV.
- `forces` — NumPy `(N, 3)` float64, eV/Å.
- `energy_per_atom` — NumPy `(N,)` float64, eV.
- `stress_voigt` — NumPy `(6,)` float64, eV/Å³, Voigt order
  (xx, yy, zz, yz, xz, xy).

### `compute_per_atom`

CPU backend only. Returns all per-atom physical quantities in a single call.
Raises `NotImplementedError` when called on a GPU-backend engine.

```python
(
    energy,
    forces,
    energy_per_atom,
    stress_voigt,
    virial_per_atom,
    densities,
    embedding_energies,
) = engine.compute_per_atom(
    positions,    # (N, 3) float64, Å, Cartesian
    atom_types,   # (N,)   int32
    cell,         # (3, 3) float64 | None
)
```

The return tuple:

- `energy` — Python `float`, eV.
- `forces` — NumPy `(N, 3)` float64, eV/Å.
- `energy_per_atom` — NumPy `(N,)` float64, eV.
- `stress_voigt` — NumPy `(6,)` float64, eV/Å³, Voigt order.
- `virial_per_atom` — NumPy `(N, 6)` float64, eV. Raw atomic virial in the
  LAMMPS convention (not divided by atomic volume).
- `densities` — NumPy `(N,)` float64. EAM electron density at each atom.
- `embedding_energies` — NumPy `(N,)` float64, eV. EAM embedding energy at
  each atom.

### `compute_with_debug` — GPU-only introspection `compute_with_debug` entry point
that returns every intermediate buffer of the cell-list pipeline. Use
it for profiling, shader diagnosis, or bug reports.

```python
energy, forces, energy_per_atom, debug = engine.compute_with_debug(
    positions, atom_types, cell,
)

# debug is None when:
#   - backend == "cpu", or
#   - use_cell_list is False.
# Otherwise it is a dict with keys:
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

Calling `compute_with_debug` on a CPU-backend engine raises
`NotImplementedError`.

### Properties

```python
engine.elements      # list[str], e.g. ['Cu']          — element order
engine.n_elements    # int                            — same as len(elements)
engine.cutoff        # float, Å                       — from the file header
engine.backend       # 'gpu' or 'cpu'                 — active backend
```

## Choosing backend and neighbour strategy

| System size          | Recommended settings                                           |
|----------------------|----------------------------------------------------------------|
| N ≲ 2 000            | `backend="gpu"`, `use_cell_list=False` (AllPairs)              |
| 2 000 ≲ N ≲ 10⁴       | `backend="gpu"`, `use_cell_list=True`                          |
| 10⁴ ≲ N              | `backend="gpu"`, `use_cell_list=True` (mandatory for speed)    |
| any N, no GPU host   | `backend="cpu"`, `use_cell_list=True`                          |
| CI / determinism     | `backend="cpu"` (deterministic accumulation order)             |

The CPU backend is several × slower than the GPU backend at the same
N on the same hardware; at N = 10⁶ it is a well-defined baseline, but
GPU is what you want for production runs.
