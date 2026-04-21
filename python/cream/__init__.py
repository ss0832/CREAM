"""CREAM — Compute-shader Rust EAM Atomistics.

GPU-accelerated (or CPU-parallel) EAM molecular dynamics engine accessible
from Python / ASE.

Quick start::

    from ase.build import bulk
    from cream import CreamCalculator

    # GPU backend (default)
    atoms = bulk('Cu', 'fcc', a=3.615) * (3, 3, 3)
    atoms.calc = CreamCalculator('Cu.eam.alloy')
    print(atoms.get_potential_energy())

    # CPU backend – no GPU required
    atoms.calc = CreamCalculator('Cu.eam.alloy', backend='cpu')
    print(atoms.get_potential_energy())
"""

from cream._cream import CreamEngine  # noqa: F401
from cream.calculator import CreamCalculator  # noqa: F401

__version__ = "0.1.0"
__all__ = ["CreamEngine", "CreamCalculator"]
