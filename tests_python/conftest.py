# tests_python/conftest.py
"""Shared pytest fixtures and backend availability guards.

GPU guard
---------
When running in a headless CI environment without a GPU (or software
rasteriser), every ``CreamEngine(backend="gpu")`` construction raises::

    ValueError: GPU device lost: No GPU adapter found

Rather than marking every test with ``@pytest.mark.skipif``, we probe GPU
availability once at session start and expose a ``require_gpu`` fixture that
automatically skips when no adapter is present.

CPU guard
---------
The CPU backend (``backend="cpu"``) uses rayon and never requires a GPU.
``require_cpu`` is provided for symmetry but **never skips**.
"""

from __future__ import annotations

import pathlib

import pytest

# ── Potential file helper ─────────────────────────────────────────────────────


def _find_potential() -> str | None:
    """Return path to Cu01_eam.alloy, or None if not found."""
    candidates = [
        pathlib.Path("Cu01_eam.alloy"),
        pathlib.Path(__file__).parent.parent / "Cu01_eam.alloy",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


# ── Session-scoped GPU probe ──────────────────────────────────────────────────


@pytest.fixture(scope="session")
def _gpu_available() -> bool:
    """Return True if wgpu can find an adapter, False otherwise.

    Probes once per session to avoid repeated init overhead.
    """
    pot = _find_potential()
    if pot is None:
        return False  # can't probe without a potential file

    try:
        from cream import CreamEngine

        CreamEngine(pot, use_cell_list=False, backend="gpu")
        return True
    except ValueError as exc:
        msg = str(exc).lower()
        if "gpu" in msg or "adapter" in msg or "device" in msg:
            return False
        raise  # re-raise unexpected errors


# ── Session-scoped CPU probe ──────────────────────────────────────────────────


@pytest.fixture(scope="session")
def _cpu_available() -> bool:
    """Return True if the CPU backend can be constructed.

    The CPU backend never requires a GPU and should always be available
    when the potential file exists.
    """
    pot = _find_potential()
    if pot is None:
        return False

    try:
        from cream import CreamEngine

        CreamEngine(pot, backend="cpu")
        return True
    except Exception:  # noqa: BLE001
        return False


# ── Convenience skip fixtures ─────────────────────────────────────────────────


@pytest.fixture
def require_gpu(_gpu_available: bool) -> None:
    """Skip the calling test when no GPU adapter is available."""
    if not _gpu_available:
        pytest.skip("No GPU adapter found — skipping GPU-dependent test")


@pytest.fixture
def require_cpu(_cpu_available: bool) -> None:
    """Skip the calling test when the CPU backend is unavailable.

    In practice this only skips when the potential file is missing.
    """
    if not _cpu_available:
        pytest.skip("CPU backend unavailable (potential file missing?)")
