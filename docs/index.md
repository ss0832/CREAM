# CREAM documentation

CREAM is a GPU-accelerated molecular-dynamics force engine based on the
Embedded Atom Method. This site is the reference documentation for the
Rust crate, the Python package, the CLI, and the WebGPU build.

- **Source code**: <https://github.com/ss0832/cream>
- **Python package**: `pip install cream-python`
- **Rust crate**: `cargo add cream`

If you are new to CREAM, start with [Architecture](architecture.md) to
see how the engine is organised, then [Backends](backends.md) to
choose the right neighbour strategy for your system size, then
[PBC](pbc.md) for the cell conventions.

For Python users specifically, the **[Python API](python_api.md)** page
documents `CreamCalculator`, `CreamEngine`, how to select the GPU or
CPU backend, and how to introspect the cell-list pipeline for
debugging.

The design rationale — why certain algorithms were chosen, what
trade-offs they make, which paths were considered and rejected — is
gathered in [Design notes](design.md). It is not required reading but
it answers most "why not X?" questions.

```{toctree}
:maxdepth: 2
:caption: Reference

architecture
backends
pbc
potentials
python_api
api
known_limits
design
roadmap
```
