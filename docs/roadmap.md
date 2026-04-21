# Roadmap

The v0.1.0 release freezes the public Rust API, the PyO3 / ASE bindings,
and the WGSL shader interface. Work planned for subsequent releases is
listed here for transparency; it is not a schedule.


## Medium term — feature completion

### WebAssembly / WebGPU build

A WASM entry point exists in the source tree (`src/wasm.rs`) and
targets the WebGPU backend of `wgpu`. It has not yet been validated
end-to-end against the current shader set, so it is not advertised as
a supported path for v0.1.0. Validation will consist of running the
same correctness sweep as `cargo test --release -- --include-ignored`
inside a headless Chromium via `wasm-bindgen-test`, plus an
interactive demo for the project page.

### Command-line front-end (`cream-cli`)

The CLI source exists in `cream-cli/` and builds against the current
library, but the v0.1.0 release does not advertise it because the
argument surface has not been reviewed since the API freeze. Work for
a subsequent release:

- Audit the `clap` argument definitions against the final public Rust
  API.
- Add integration tests that exercise each subcommand end-to-end.
- Document the full CLI in a dedicated doc page.

## Longer term — performance and physics

### Subgroup / warp intrinsics in the density pass

Pass 1 currently accumulates the density per atom in a straightforward
shared-memory reduction. On backends that expose the WGSL `subgroup`
intrinsics (Vulkan 1.1+, D3D12 with SM 6.0+) a warp-level reduction
would save a barrier and a shared-memory round-trip per 64 atoms.

### Neighbour-list rebuild avoidance

The engine rebuilds the cell list on every call. For long MD runs
where atoms move slowly this is wasteful; a skin-margin scheme that
rebuilds only every N steps (or when the maximum displacement exceeds
the skin) would give a measurable speedup. The neighbour-list
infrastructure (`src/neighbor_list.rs`) already supports a skin
parameter; the piece that is missing is the host-side state machine
that decides when to rebuild.
