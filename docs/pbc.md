# Periodic boundary conditions

CREAM supports orthorhombic and fully triclinic cells as well as
non-periodic (cluster) calculations.

## Cell matrix convention

The cell matrix `H` stores lattice vectors as **rows**, the ASE and
LAMMPS convention:

```text
H = [[ax, ay, az],   ← a
     [bx, by, bz],   ← b
     [cx, cy, cz]]   ← c
```

Consequently, fractional coordinates are computed from Cartesian
coordinates via

```text
s = x · H⁻¹
```

where `x` is a row vector. This matches the convention used by ASE's
`atoms.cell` and is compatible with LAMMPS' restart files.

## Orthorhombic

When `H` is diagonal the cell is orthorhombic. A convenience constructor
is provided:

```rust
let cell = cream::ortho_cell(10.0, 10.0, 10.0);   // f64, for the reference impl
let cell = [[10.0f32, 0.0, 0.0],                   // f32, for ComputeEngine
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0]];
```

## Triclinic

Any non-singular 3×3 matrix with a positive determinant is accepted. The
engine checks the shortest perpendicular height against `2 × cutoff` and
returns `CreamError::InvalidInput` if the minimum-image convention would
be violated.

## Minimum-image convention

The minimum-image displacement between two atoms is computed in
fractional space:

```text
d   = xⱼ − xᵢ
s   = d · H⁻¹
s' = s − round(s)    # round-to-nearest, ties to even
d' = s' · H          # back to Cartesian
```

The rounding step picks the image of `j` that lies within the primary
image box centred on `i`. This formulation is correct for any valid
cell geometry, orthorhombic or triclinic, as long as the cell's shortest
perpendicular height is at least `2 × cutoff`.

For orthorhombic cells, a faster integer-shift fast path is used: the
cell-list stencil computes how many full periodic images a neighbour
crosses, and adds the corresponding lattice vector to the raw Cartesian
difference. This avoids three `round()` calls per pair at the cost of
some f32-boundary subtlety (see [Design notes](design.md)).

## Non-periodic (cluster)

Pass `None` (Rust) or `cell=None` (Python) to compute forces for a
finite cluster. The cell list, if used, is a plain Cartesian
bounding-box grid without PBC wrap. The engine adds a `cutoff / 2` skin
around the atom positions to ensure boundary atoms do not fall outside
the grid.

## Wrapping behaviour

CREAM does **not** mutate input positions. Atoms that fall outside the
primary image box are still accepted and give the correct forces via
minimum-image. Internally, when the cell-list path is selected, CREAM
creates a wrapped copy of the positions so that the cell-assignment step
and the Cartesian stencil walk stay in sync. The user-visible
`positions` array is never modified.
