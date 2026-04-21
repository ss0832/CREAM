# Known limits

The design trade-offs and hard limits that CREAM users should know about
for v0.1.0.

## Engine limits

### Maximum atom count per dispatch

The fully GPU-resident cell-list path (`--features cellist_gpu`) has a
hard upper bound of `65_535 × 64 = 4_194_240` atoms per single
`dispatch`. Beyond this limit, the passes that build the cell list
(`cell_pass0a_assign` and `cell_pass0b_sort`) are chunked. The engine
handles chunking transparently, but there is a measurable non-monotone
step in wall-clock time at the boundary: the first few chunks flush
pipeline state that a single dispatch would have kept hot. For systems
in the 4.2 × 10⁶ – 10 × 10⁶ atom range the cost is a few percent of
total frame time. Above 10 × 10⁶ atoms the GPU cell-list path continues
to work but performance trends are less smooth than below 4 × 10⁶.

The Verlet integrator support uses a separate uniform buffer that has
not yet been chunked; the engine returns an explicit error for
Verlet-mode dispatches above `4 194 240` atoms.

### Numerical precision

- All GPU storage is `f32`. The CPU engine also uses `f32` for storage
  but f64 for thread-local accumulators. Per-atom force errors versus
  the f64 reference implementation are typically below `5 × 10⁻⁴ eV/Å`
  on single-element systems and below `1 × 10⁻³ eV/Å` on multi-element
  alloys; these are dominated by the accumulation order of the parallel
  reduction and by the f32 table interpolation.

- The total energy is computed via a tree reduction that falls back to
  CPU summation for `N ≥ 4096`. Agreement with the f64 reference is
  typically `|ΔE| / |E| ≲ 1 × 10⁻⁵`.

- Kahan-compensated summation is used in Pass 1 (density) to keep
  per-atom `ρᵢ` stable when the number of contributing neighbours grows
  into the hundreds.

### Minimum cell size

The minimum-image convention requires the shortest perpendicular cell
height to be at least `2 × cutoff`. The engine enforces this
precondition and returns `CreamError::InvalidInput` when it is violated.

### Cell-list stencil

The cell-list stencil is fixed at ±1 cells in each direction (27 cells
total). This assumes each cell is at least one cutoff wide. The engine
rounds the per-axis cell count down to the nearest power of two to keep
the fast-path stencil walk in the shaders efficient; rounding **down**
only makes cells larger, so the invariant `cell_width ≥ cutoff` is
preserved.

## Benchmark-harness limits (not engine limits)

### `bench_gpu` memory

The `bench_gpu` binary retains force and reference arrays in `f64` for
multiple backends so it can cross-validate results. At `N ≳ 10⁷` the
host memory footprint of these arrays alone can exceed what a laptop
with 16 GB of RAM can sustain, and the benchmark will abort with a
`memory allocation of N bytes failed` message. This is a benchmark
harness limitation, not an engine limit; the engine itself has been
exercised up to `N ≈ 1.6 × 10⁷` atoms. If you need to benchmark larger
systems, call `ComputeEngine::compute_sync` directly from a minimal
harness that retains only the buffers you care about.

### Table size

The EAM tables are uploaded as a single flat buffer. For typical
potentials (`nr = 200 – 2000`, `nrho = 200 – 2000`, ≤ 3 elements) the
buffer is a few hundred kilobytes. Pathological potentials with
tens of thousands of grid points and many elements may exceed the
default wgpu storage-buffer binding size on some backends; in that
case, splitting the tables would require changes to the binding layout.

## Platform limits

### WASM / WebGPU

A WebAssembly build is present in the source tree but is not advertised
as a supported target for v0.1.0; end-to-end validation on the current
shader set is scheduled for a subsequent release. The 30-bit Morton
code (`ncx × ncy × ncz ≤ 2³⁰`) is a hard limit of the shader format on
any backend, including future browser builds.

### GPU backends

- Tested on NVIDIA RTX (DX12 / Vulkan) and AMD Radeon (DX12 / Vulkan).
- The engine does not request any optional wgpu features. Any backend
  that supports compute shaders with `Storage` binding should work.
