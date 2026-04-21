# Neighbour-list backends

CREAM exposes three paths for computing the pair list that Pass 1 and
Pass 2 iterate over. All three produce the same forces and energy to
within f32 rounding; they differ in the amount of GPU state they carry
and in the system size at which they become efficient.

## `NeighborStrategy::AllPairs`

The simplest path. Every pair `(i, j)` with `i ≠ j` is visited, the
minimum-image distance is computed, and the cutoff is checked inside the
shader. Cost is O(N²).

Use it when:

- you are debugging and want the most auditable reference,
- your system is small enough that O(N²) is still fast (N ≲ 5000 on a
  modern discrete GPU),
- or the cell list would not help because the cutoff is a significant
  fraction of the box side length.

AllPairs has no cell-list state, no Morton codes, no extra passes. It is
the canonical correctness baseline.

## `NeighborStrategy::CellList` (default)

The default cell-list path builds a Morton-ordered cell list plus a CSR
neighbour list on the CPU, uploads both to the GPU, and dispatches
specialised `pass1` / `pass2` shaders that simply walk the CSR rows. The
GPU sees an already-pruned, already-indexed neighbour list.

This is what you want for production runs up to a few million atoms on
any `wgpu` backend. It is portable, well tested, and keeps the GPU
pipeline small.

The cost breakdown per step is roughly:

- CPU: Morton assignment + CSR build — O(N) with a small constant.
- GPU upload: two `u32` arrays of lengths `N + 1` and `~N × k_avg`, where
  `k_avg` is the average number of neighbours (typically 30 – 60 in
  condensed matter).
- GPU compute: two dispatches that do a flat CSR walk.

## `NeighborStrategy::CellList` with `--features cellist_gpu`

When the `cellist_gpu` feature is enabled, the engine switches to a
fully GPU-resident cell-list pipeline:

1. `cell_pass0a_assign` — Morton code per atom.
2. `cell_pass0b_sort` — atomic occupancy count.
3. CPU prefix sum + re-upload of `cell_start[]`.
4. `cell_pass0c_range` — scatter into sorted order.
5. `cell_pass0d_reorder` — copy positions and types into the sorted
   buffer so passes 1 and 2 get coalesced memory access.
6. `eam_pass1_cellist` — density with a 27-cell stencil walk.
7. `eam_pass2_cellist` — forces and per-atom energy, same stencil.

No CPU-side neighbour list is built. For very large systems (N ≳ 10⁶)
this saves tens of milliseconds per step that would otherwise be spent
on the CPU building CSR rows.

The trade-off is more GPU state (six extra buffers, seven extra passes)
and a narrower range of verified configurations. The GPU-resident path
is an opt-in feature for that reason, not because it is incorrect —
it has been validated to bit-identical agreement with GPU-AllPairs on
all seven crystal systems for sizes up to N = 4 × 10⁶.

## Choosing

| System size      | Recommendation                                            |
|------------------|-----------------------------------------------------------|
| N ≤ 5 × 10³      | `AllPairs` is fine; the cell list has no speedup yet      |
| 5 × 10³ < N ≤ 10⁶ | Default `CellList` (CPU-built NL)                         |
| N > 10⁶          | `--features cellist_gpu` for the fully-GPU path           |

When in doubt, start with the default and switch only if you have
measured a CPU-side neighbour-list bottleneck.

## Choosing `cell_size`

The `cell_size` field of `NeighborStrategy::CellList` must be at least
the potential cutoff radius. Larger values reduce the number of grid
cells and shift work from the 27-cell stencil walk into the per-cell
pair check; smaller values do the opposite.

A good default, and the one used by the CLI, is

```text
cell_size == potential.cutoff
```

The engine internally rounds the per-axis cell count down to the nearest
power of two so that the fast-path stencil walk in the shaders remains
efficient (see `ComputeEngine::n_cells_from_dspacing` in `engine.rs`).
To disable that rounding for debugging, set the environment variable
`CREAM_DISABLE_AUTO_P2=1` before launching the process.
