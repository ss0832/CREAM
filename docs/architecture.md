# Architecture

CREAM is a four-pass GPU compute pipeline wrapped in a small Rust
orchestrator. The orchestrator owns a [`wgpu`](https://wgpu.rs) device and
queue, maintains pipeline and buffer caches, and dispatches one shader
invocation per pass per force evaluation.

## Layered structure

```
┌──────────────────────────────────────────────────┐
│                Interface layer                    │
│     PyO3 / ASE        (v0.1.0 supported)          │
│     cream-cli, WASM   (experimental)              │
├──────────────────────────────────────────────────┤
│   ComputeEngine       (GPU, &mut self)            │
│   CpuEngine           (rayon fallback)            │
│   pipeline + table caches (Arc)                   │
├──────────────────────────────────────────────────┤
│   Pass 0: Cell list construction (Morton, O(N))   │
│   Pass 1: Electron density  ρᵢ = Σⱼ fβ(r)         │
│   Pass 2: Forces + per-atom energy                │
│   Pass 3: Energy reduction                        │
├──────────────────────────────────────────────────┤
│   GpuPotential trait  (LSP / OCP)                 │
│     └── EamPotential (.eam.alloy parser)          │
├──────────────────────────────────────────────────┤
│   wgpu  (Vulkan / Metal / DX12)                   │
└──────────────────────────────────────────────────┘
```

The key design principle is that the orchestrator knows nothing about
physics. It sees a `&dyn GpuPotential` trait object and asks it for:

- buffer sizes,
- number of elements and cutoff radius,
- the WGSL source of each shader variant it wants to dispatch,
- a stable cache key.

The trait object also knows how to upload its tables to the GPU. This is
what makes CREAM extensible: adding, say, Morse or Lennard-Jones would
require only a new trait impl and a few shader files, not an engine
rewrite.

## The four passes

| Pass        | What it computes                               | Output per atom  |
|-------------|------------------------------------------------|------------------|
| Pass 0      | Morton-ordered cell list (cell-list mode only) | sorted indices   |
| Pass 1      | Electron density `ρᵢ = Σⱼ≠ᵢ fβ(rᵢⱼ)`            | `ρᵢ` (f32)       |
| Pass 2      | `Fᵢ`, per-atom pair energy                     | force vec + `eᵢ` |
| Pass 3      | Sum of `eᵢ` over all atoms                     | total energy     |

Passes 1 and 2 together form the EAM force computation. Splitting them
is mandatory: the force on atom `i` depends on the embedding derivative
`F'(ρᵢ)` **and** `F'(ρⱼ)` for every neighbour `j`, so all densities must
be known before any forces are computed.

### Pass 0 — cell-list construction

When `NeighborStrategy::CellList` is selected, three short compute passes
build the spatial index:

1. **`cell_pass0a_assign`** — each atom computes its fractional cell
   coordinates and packs them into a 30-bit Morton (Z-order) code.
2. **`cell_pass0b_sort`** — atoms increment an atomic per-cell counter,
   producing cell occupancy.
3. **CPU prefix sum** — occupancy is read back and scanned to produce
   `cell_start[]` offsets. This is done on the CPU because `wgpu` does
   not expose a standard GPU scan; N ≤ 10⁷ makes the readback negligible.
4. **`cell_pass0c_range`** — atoms write their index into the sorted
   buffer at the position given by an atomic write cursor.

Morton-code ordering is chosen so that spatially adjacent cells map to
adjacent indices in the sorted buffer. On desktop GPUs this gives a
measurable L2 cache win in passes 1 and 2, which iterate over a
27-cell stencil per atom.

### Passes 1 and 2 — density and forces

Passes 1 and 2 come in three variants:

- **AllPairs** — pure O(N²) pair scan. Correct for any cell geometry and
  used as the reference implementation.
- **CellList (CPU-built NL)** — the CPU builds a CSR neighbour list from
  the Morton-ordered cell list and uploads it; the GPU shader simply
  walks the CSR rows. This is the default cell-list path.
- **CellList (GPU-resident)** — the shader reads the Morton-reordered
  atom buffer produced by pass 0d and walks a 27-cell stencil directly.
  Enabled by the `cellist_gpu` feature flag.

All three variants compute Pass 1 with Kahan-compensated summation and
shared-memory tiling; Pass 2 is full-pair (Newton off) because WGSL does
not support `atomicAdd<f32>`.

See [Backends](backends.md) for how to choose between them.

### Pass 3 — energy reduction

Pass 3 is a two-dispatch workgroup tree-reduction: `N → ⌈N / 64⌉ → 1`.
For `N ≥ 4096` the second dispatch is skipped and the CPU sums the ⌈N/64⌉
partial sums directly. This avoids a deep reduction tree while staying
within WebGPU's 8-binding-per-stage limit.

## Buffer layout

CREAM packs every EAM table into a **single** storage buffer so that the
shader bindings stay within WebGPU's limit of eight per stage. The flat
layout is:

```
[ f_α(r) | F_α(ρ) | F'_α(ρ) | φ_αβ(r) | f'_α(r) | φ'_αβ(r) ]
```

A `TableLayout` uniform (64 bytes) describes offsets, table lengths, and
the `dr` / `dρ` spacings. All tables are `f32`.

The shader bindings for every pass share indices 0 – 4:

| Binding | Buffer                                                       |
|---------|--------------------------------------------------------------|
| 0       | `positions : array<vec4<f32>>` — xyz + w padding             |
| 1       | `atom_types : array<u32>`                                     |
| 2       | `params : SimParams` (uniform)                                |
| 3       | `potential_tables : array<f32>` — the packed table buffer     |
| 4       | `tbl : TableLayout` (uniform)                                 |

Pass-specific buffers (density, forces, energies, cell list) start at
binding 5.

## Caching

`ComputeEngine` caches two kinds of resources across calls:

- **Pipelines** — keyed by the hash of the WGSL source. Two successive
  calls with the same potential therefore reuse the same
  `wgpu::ComputePipeline`.
- **Potential tables** — keyed by `GpuPotential::cache_key()`. The
  uploaded GPU buffers live as long as the engine does.

Per-frame atom buffers (positions, types, forces, densities, cell list)
are kept in `FrameBuffers` and grown in place only when `N` or the cell
geometry changes. A steady-state simulation loop therefore allocates no
new GPU memory after the first few calls.
