# Design notes

This page collects the design decisions that shaped CREAM. It exists to
answer the "why not X?" questions that users and contributors tend to
ask; nothing here is required reading for use of the library.

## Why a single flat table buffer

The obvious layout for the EAM tables is one buffer per table: one for
`f_α(r)`, one for `F_α(ρ)`, and so on. That would give six bindings
per shader stage, which, combined with the per-atom position, type,
params, density, and forces buffers, exceeds WebGPU's limit of eight
`max_storage_buffers_per_shader_stage`.

Packing every table into a single storage buffer plus a small
`TableLayout` uniform that describes the offsets leaves the binding
count at five (per-atom + tables + uniform + one pass-specific). This
is important for two reasons:

1. The engine can always add one or two more per-atom outputs (for
   example, virial tensors or per-atom stresses) without running out of
   bindings.
2. It works on every `wgpu` backend, including WebGPU in the browser,
   without requiring the `storage-resource-binding-array` extension.

## Why Morton (Z-order) cell codes

Alternative orderings for the sorted atom buffer include:

- **Lexicographic** (x-major, then y, then z) — trivial to compute but
  spatially distant in y and z.
- **Hilbert** — better locality than Morton in theory, significantly
  more expensive to encode on a GPU.
- **Morton** (chosen) — preserves spatial locality well enough for the
  27-cell stencil walk, and encodes in a constant number of bit-twiddle
  operations on the GPU.

Pass 1 and Pass 2 iterate over a 3×3×3 cell stencil per atom. With
Morton ordering, the 27 cells being visited map to a small, mostly
contiguous range of indices in the sorted buffer, which keeps L2 hit
rates high. Profiling on RTX 4070 showed Morton ordering giving a
measurable speedup versus lexicographic at `N = 10⁵`.

## Why rounding the cell count down to a power of two

The cell-list fast path in `eam_pass{1,2}_cellist.wgsl` uses a
workgroup-wide bounding-box plus a periodic-wrap stencil that is only
correct and efficient on power-of-two grids. On non-power-of-two grids
the shaders fall back to iterating all `ncx × ncy × ncz` cells per
workgroup, which quickly dominates runtime for `ncx ≥ 8`.

Rounding each axis **down** to the nearest power of two preserves the
invariant `cell_width ≥ cutoff` (rounding down only makes cells larger)
while keeping the fast path active. The trade-off — more atoms per
cell — is dwarfed by the O(N²) → O(N) reduction in pair-scan work
as long as the 27-cell stencil can prune effectively.

To disable this rounding (useful only for regression testing the fast
path), set `CREAM_DISABLE_AUTO_P2=1`.

## Why CPU-built CSR is the default cell-list path

The fully GPU-resident cell-list pipeline is faster for large systems
but carries six extra GPU buffers and seven extra passes of state. The
CPU-built path is smaller, easier to reason about, and runs on every
`wgpu` backend without any additional assumptions about atomic
operations or workgroup barrier semantics.

For v0.1.0 the CPU-built path is the default; the GPU-resident path is
available through the `cellist_gpu` feature flag for users who need
the largest systems.

## Why the ortho fast path in the CPU cell list

The CPU cell list's hot inner loop (`for_each_forward_neighbor` in
`src/cpu_engine.rs`) has to compute the minimum-image displacement
between the central atom and each neighbour. For orthorhombic cells
this can be done in two very different ways:

1. **Per-pair `min_image`** — for each pair, map the raw Cartesian
   difference into fractional space, apply `s - round(s)`, and map back.
   This is six FMAs and three `round()` operations per pair.
2. **Stencil integer shift** — precompute the shift vector once per
   27-cell stencil offset from how many periodic images the cell index
   wraps around, then just add that shift to the raw Cartesian
   difference. Three FMAs per pair.

Method 2 is substantially faster per pair, but it assumes the atom's
cell index is consistent with its wrapped Cartesian position — a
property that requires care when f32 round-trip rounding is involved
(see below).

## Why `fold_cell_index_pbc` exists

When a wrapped fractional coordinate `sf = 1 - ulp` is multiplied by
the box side length and then divided by it again through a re-computed
`h_inv`, the result can round to exactly `1.0` in f32. The natural
cell-assignment expression `(floor(s * n) as i32).rem_euclid(n)` then
returns `0`, which places the atom in the cell at the **opposite**
end of the box from where it physically sits.

The ortho fast path, which derives its shift from the cell-index
delta, then produces a shift off by exactly one lattice vector for
every pair involving that atom. The resulting force error can be as
large as tens of meV/Å, concentrated in a small number of boundary
atoms.

`src/cell_list.rs::fold_cell_index_pbc` fixes this by recognising the
`raw == n` case as an f32 round-trip artifact and folding to `n - 1`
instead of zero. For every other input it behaves exactly like
`rem_euclid` — so the fix is invisible to normal periodic-image
handling.

A regression test (`tests::cell_coords_of_handles_f32_boundary_roundtrip`
in `src/cell_list.rs`) exercises the exact failing configuration from
the bench.

## Why `unsafe` is forbidden

`src/lib.rs` declares `#![deny(unsafe_code)]`. The engine's
correctness-critical hot loops use only safe Rust and `wgpu`'s safe
interface; there is no `transmute`, no manual SIMD, no pointer casts.
Parallel reduction safety is provided by `rayon`'s work-stealing API,
not by raw atomics.

Should a future optimisation require `unsafe` (for example, to avoid a
bounds check in a tight inner loop that `rustc` cannot prove safe),
the plan is to isolate it in a small module with `#[allow(unsafe_code)]`
at the module level and a safety argument in prose, not to relax the
crate-wide ban.

## Why half-pair (Newton's third law) on CPU but full-pair on GPU

The CPU engine visits each pair `(i, j)` with `j > i` exactly once and
writes the force contribution to both atoms, halving the distance
computations. This is safe because each rayon thread owns its own
accumulator array; the accumulators are merged after all threads
complete.

The GPU engine cannot do this because WGSL does not support
`atomicAdd<f32>`. Emulating `atomicAdd` with an `atomicCompareExchange`
loop on `u32` and bit-reinterpretation is possible but slow and
introduces non-determinism. Instead, the GPU visits every pair twice
(once from each side) and writes to each atom's force slot without
contention. The 2× extra pair work is more than offset by the lack of
atomic contention.
