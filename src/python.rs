//! PyO3 bindings for CREAM.
//!
//! Exposes [`CreamEngine`] to Python, which wraps either [`ComputeEngine`]
//! (GPU) or [`CpuEngine`] (CPU) and [`EamPotential`] behind a single object.
//! All coordinate arrays are accepted as NumPy `float64` (the ASE default)
//! and converted to `f32` internally.
//!
//! # Backend selection
//!
//! Pass `backend="gpu"` (default) to use the wgpu GPU backend, or
//! `backend="cpu"` to use the rayon-parallelised CPU backend.
//!
//! ```python
//! import numpy as np
//! from cream._cream import CreamEngine
//!
//! # GPU (default)
//! engine_gpu = CreamEngine("Cu.eam.alloy", backend="gpu")
//!
//! # CPU – always available, no GPU required
//! engine_cpu = CreamEngine("Cu.eam.alloy", backend="cpu")
//!
//! positions = np.array([[0.0, 0.0, 0.0], [1.8, 1.8, 0.0]], dtype=np.float64)
//! atom_types = np.array([0, 0], dtype=np.int32)
//! cell = np.diag([10.0, 10.0, 10.0])
//! energy, forces, energy_per_atom = engine_cpu.compute(positions, atom_types, cell)
//! ```

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{
    cpu_engine::CpuEngine,
    engine::ComputeEngine,
    potential::{eam::EamPotential, NeighborStrategy},
};

// ── Return type of PyCreamEngine::compute ────────────────────────────────────
type ComputeResult<'py> = (f64, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>);

// ── Internal backend discriminant ────────────────────────────────────────────

/// Holds either a GPU engine or a CPU engine.
///
/// `ComputeEngine::compute_sync` requires `&mut self` (manages GPU buffers),
/// while `CpuEngine::compute_sync` only needs `&self`; using `&mut` in the
/// match arm is fine for both.
enum Backend {
    Gpu(ComputeEngine),
    Cpu(CpuEngine),
}

// ── PyO3 class ────────────────────────────────────────────────────────────────

/// EAM force/energy calculator with selectable compute backend.
///
/// Parameters
/// ----------
/// potential_file : str
///     Path to a `.eam.alloy` potential file.
/// use_cell_list : bool, optional
///     If ``True``, use O(N) Cell List neighbour search (default ``False``).
/// cell_size : float or None, optional
///     Cell size for Cell List [Å]. Defaults to the potential cutoff.
/// backend : str, optional
///     Compute backend: ``"gpu"`` (default) or ``"cpu"``.
///     Use ``"cpu"`` when no GPU adapter is available or for reproducible
///     CPU-only execution.
#[pyclass(name = "CreamEngine")]
pub struct PyCreamEngine {
    backend: Backend,
    potential: EamPotential,
    /// Stored so that CPU dispatch can choose compute_sync vs
    /// compute_cell_list_sync.
    use_cell_list: bool,
    /// Cached string tag returned by the `backend` property.
    backend_name: &'static str,
}

#[pymethods]
impl PyCreamEngine {
    /// Create a new CREAM engine.
    #[new]
    #[pyo3(signature = (potential_file, *, use_cell_list=false, cell_size=None, backend="gpu"))]
    fn new(
        potential_file: &str,
        use_cell_list: bool,
        cell_size: Option<f32>,
        backend: &str,
    ) -> PyResult<Self> {
        // ── Load potential (shared by all backends) ───────────────────────
        let potential = EamPotential::from_file(std::path::Path::new(potential_file))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        // ── Neighbour strategy ────────────────────────────────────────────
        // GPU path uses this directly; CPU path uses `use_cell_list` to
        // dispatch the right method at compute time.
        let strategy = if use_cell_list {
            let cs = cell_size.unwrap_or(potential.cutoff_angstrom);
            NeighborStrategy::CellList { cell_size: cs }
        } else {
            NeighborStrategy::AllPairs
        };

        // ── Construct chosen backend ──────────────────────────────────────
        let (engine_backend, backend_name) = match backend {
            "gpu" => {
                let engine = pollster::block_on(ComputeEngine::new(strategy))
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                (Backend::Gpu(engine), "gpu")
            }
            "cpu" => (Backend::Cpu(CpuEngine::new()), "cpu"),
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown backend {other:?}. Valid choices are \"gpu\" and \"cpu\"."
                )))
            }
        };

        Ok(Self {
            backend: engine_backend,
            potential,
            use_cell_list,
            backend_name,
        })
    }

    /// Compute EAM energy and forces.
    ///
    /// Parameters
    /// ----------
    /// positions : ndarray, shape (N, 3), dtype float64
    ///     Cartesian coordinates [Å].
    /// atom_types : ndarray, shape (N,), dtype int32
    ///     Element index per atom (0-based, matching the potential file order).
    /// cell : ndarray, shape (3, 3), dtype float64, optional
    ///     Row-vector cell matrix ``[[ax,ay,az],[bx,by,bz],[cx,cy,cz]]``.
    ///     ``None`` for non-periodic (cluster) calculations.
    ///
    /// Returns
    /// -------
    /// energy : float
    ///     Total potential energy [eV].
    /// forces : ndarray, shape (N, 3), dtype float64
    ///     Force on each atom [eV/Å].
    /// energy_per_atom : ndarray, shape (N,), dtype float64
    ///     Per-atom energy [eV].
    #[pyo3(signature = (positions, atom_types, cell=None))]
    fn compute<'py>(
        &mut self,
        py: Python<'py>,
        positions: PyReadonlyArray2<'py, f64>,
        atom_types: PyReadonlyArray1<'py, i32>,
        cell: Option<PyReadonlyArray2<'py, f64>>,
    ) -> PyResult<ComputeResult<'py>> {
        let pos = positions.as_array();
        let types_arr = atom_types.as_array();

        let n = pos.nrows();
        if pos.ncols() != 3 {
            return Err(PyValueError::new_err(format!(
                "positions must be (N, 3), got (N, {})",
                pos.ncols()
            )));
        }
        if n == 0 {
            return Err(PyValueError::new_err("positions must not be empty"));
        }
        if types_arr.len() != n {
            return Err(PyValueError::new_err(format!(
                "atom_types length {} ≠ positions rows {}",
                types_arr.len(),
                n
            )));
        }

        // Validate atom type range before converting (gives PyValueError, not PyRuntimeError)
        let n_elements = self.potential.elements.len();
        for (i, &t) in types_arr.iter().enumerate() {
            if t < 0 {
                return Err(PyValueError::new_err(format!("negative atom type: {t}")));
            }
            if (t as usize) >= n_elements {
                return Err(PyValueError::new_err(format!(
                    "atom_types[{i}]={t} out of range (n_elem={n_elements})"
                )));
            }
        }

        // f64 → f32 positions (vec4 with w=0 padding)
        let positions4: Vec<[f32; 4]> = (0..n)
            .map(|i| {
                [
                    pos[[i, 0]] as f32,
                    pos[[i, 1]] as f32,
                    pos[[i, 2]] as f32,
                    0.0,
                ]
            })
            .collect();

        // i32 → u32 atom types (range already validated above)
        let atom_types_u32: Vec<u32> = types_arr.iter().map(|&t| t as u32).collect();

        // Cell: f64 (3,3) → Option<[[f32;3];3]>
        let cell_f32: Option<[[f32; 3]; 3]> = match cell {
            Some(c) => {
                let c = c.as_array();
                if c.nrows() != 3 || c.ncols() != 3 {
                    return Err(PyValueError::new_err("cell must be (3, 3)"));
                }
                Some([
                    [c[[0, 0]] as f32, c[[0, 1]] as f32, c[[0, 2]] as f32],
                    [c[[1, 0]] as f32, c[[1, 1]] as f32, c[[1, 2]] as f32],
                    [c[[2, 0]] as f32, c[[2, 1]] as f32, c[[2, 2]] as f32],
                ])
            }
            None => None,
        };

        // Dispatch to the chosen backend (releases GIL during compute)
        let use_cell_list = self.use_cell_list;
        let result = py
            .allow_threads(|| match &mut self.backend {
                Backend::Gpu(engine) => {
                    engine.compute_sync(&positions4, &atom_types_u32, cell_f32, &self.potential)
                }
                Backend::Cpu(engine) => {
                    if use_cell_list {
                        engine.compute_cell_list_sync(
                            &positions4,
                            &atom_types_u32,
                            cell_f32,
                            &self.potential,
                        )
                    } else {
                        engine.compute_sync(&positions4, &atom_types_u32, cell_f32, &self.potential)
                    }
                }
            })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // f32 → f64 for forces.
        // Use a flat Vec<f64> + from_owned_array_bound instead of from_vec2_bound,
        // which allocates N inner Vec objects (O(N) heap pressure).
        let forces_flat: Vec<f64> = result
            .forces
            .iter()
            .flat_map(|f| [f[0] as f64, f[1] as f64, f[2] as f64])
            .collect();
        // SAFETY: flat_forces has exactly n*3 elements, layout matches (n, 3).
        let forces_nd = numpy::ndarray::Array2::from_shape_vec((n, 3), forces_flat)
            .expect("forces shape invariant violated");
        let forces_arr = PyArray2::from_owned_array_bound(py, forces_nd);

        // Per-atom energy: f32 → f64
        let epa: Vec<f64> = result.energy_per_atom.iter().map(|&e| e as f64).collect();
        let epa_arr = PyArray1::from_vec_bound(py, epa);

        Ok((result.energy as f64, forces_arr, epa_arr))
    }

    // ── Getters ───────────────────────────────────────────────────────────────

    /// Element symbols in this potential, in order.
    #[getter]
    fn elements(&self) -> Vec<String> {
        self.potential.elements.clone()
    }

    /// Cutoff radius [Å].
    #[getter]
    fn cutoff(&self) -> f32 {
        self.potential.cutoff_angstrom
    }

    /// Number of element species.
    #[getter]
    fn n_elements(&self) -> usize {
        self.potential.elements.len()
    }

    /// Active compute backend: ``"gpu"`` or ``"cpu"``.
    #[getter]
    fn backend(&self) -> &str {
        self.backend_name
    }

    /// Compute EAM energy, forces, per-atom energy **and** the virial stress tensor.
    ///
    /// The stress tensor is computed via the pair-wise virial:
    ///
    /// .. math::
    ///
    ///     \\sigma_{\\alpha\\beta} = -\\frac{1}{V} \\sum_{i<j} r^{\\alpha}_{ij} F^{\\beta}_{ij}
    ///
    /// where the sum runs over all pairs within the cutoff, and
    /// :math:`r^{\\alpha}_{ij}` is the :math:`\\alpha`-component of the
    /// minimum-image displacement vector :math:`\\mathbf{r}_j - \\mathbf{r}_i`.
    ///
    /// Parameters
    /// ----------
    /// positions : ndarray, shape (N, 3), dtype float64
    ///     Cartesian coordinates [Å].
    /// atom_types : ndarray, shape (N,), dtype int32
    ///     Element index per atom.
    /// cell : ndarray, shape (3, 3), dtype float64, optional
    ///     Row-vector cell matrix. ``None`` for non-periodic calculations
    ///     (stress returned as zeros in that case).
    ///
    /// Returns
    /// -------
    /// energy : float
    ///     Total potential energy [eV].
    /// forces : ndarray, shape (N, 3), dtype float64
    ///     Forces [eV/Å].
    /// energy_per_atom : ndarray, shape (N,), dtype float64
    ///     Per-atom energy [eV].  Empty for the GPU backend.
    /// stress : ndarray, shape (6,), dtype float64
    ///     Voigt stress tensor ``[σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy]`` [eV/Å³].
    ///     Multiply by ``160.21766`` to convert to GPa.
    ///
    /// Notes
    /// -----
    /// Both backends produce the stress tensor in a single compute pass:
    /// the GPU reduces six virial partials per workgroup alongside the
    /// energy partial, with zero additional barriers (the reduction tree
    /// is shared), and the Rust host performs the final Neumaier-compensated
    /// sum-and-divide-by-volume step.  ``compute_stress`` has effectively
    /// the same cost as ``compute`` for the GPU backend.
    #[pyo3(signature = (positions, atom_types, cell=None))]
    fn compute_stress<'py>(
        &mut self,
        py: Python<'py>,
        positions: PyReadonlyArray2<'py, f64>,
        atom_types: PyReadonlyArray1<'py, i32>,
        cell: Option<PyReadonlyArray2<'py, f64>>,
    ) -> PyResult<(
        f64,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    )> {
        // ── Input validation (same as compute) ───────────────────────────────
        let pos = positions.as_array();
        let types_arr = atom_types.as_array();
        let n = pos.nrows();
        if pos.ncols() != 3 {
            return Err(PyValueError::new_err(format!(
                "positions must be (N, 3), got (N, {})",
                pos.ncols()
            )));
        }
        if n == 0 {
            return Err(PyValueError::new_err("positions must not be empty"));
        }
        if types_arr.len() != n {
            return Err(PyValueError::new_err(format!(
                "atom_types length {} ≠ positions rows {}",
                types_arr.len(),
                n
            )));
        }
        let n_elements = self.potential.elements.len();
        for (i, &t) in types_arr.iter().enumerate() {
            if t < 0 {
                return Err(PyValueError::new_err(format!("negative atom type: {t}")));
            }
            if (t as usize) >= n_elements {
                return Err(PyValueError::new_err(format!(
                    "atom_types[{i}]={t} out of range (n_elem={n_elements})"
                )));
            }
        }

        // ── Convert inputs ────────────────────────────────────────────────────
        let positions4: Vec<[f32; 4]> = (0..n)
            .map(|i| [pos[[i, 0]] as f32, pos[[i, 1]] as f32, pos[[i, 2]] as f32, 0.0])
            .collect();
        let atom_types_u32: Vec<u32> = types_arr.iter().map(|&t| t as u32).collect();
        let cell_f32: Option<[[f32; 3]; 3]> = match &cell {
            Some(c) => {
                let c = c.as_array();
                if c.nrows() != 3 || c.ncols() != 3 {
                    return Err(PyValueError::new_err("cell must be (3, 3)"));
                }
                Some([
                    [c[[0, 0]] as f32, c[[0, 1]] as f32, c[[0, 2]] as f32],
                    [c[[1, 0]] as f32, c[[1, 1]] as f32, c[[1, 2]] as f32],
                    [c[[2, 0]] as f32, c[[2, 1]] as f32, c[[2, 2]] as f32],
                ])
            }
            None => None,
        };

        // ── Compute forces + energy + stress in one shot ─────────────────────
        // Both backends now populate `result.virial` directly:
        //   • CPU: half-pair pair-loop, σ = −W/V computed inline in cpu_engine.rs.
        //   • GPU: per-WG virial tree-reduction in pass2 + Neumaier CPU sum
        //          + σ = −W/V in engine.rs::read_and_finalize_virial.
        let use_cell_list = self.use_cell_list;
        let result = py
            .allow_threads(|| match &mut self.backend {
                Backend::Gpu(engine) => {
                    engine.compute_sync(&positions4, &atom_types_u32, cell_f32, &self.potential)
                }
                Backend::Cpu(engine) => {
                    if use_cell_list {
                        engine.compute_cell_list_sync(
                            &positions4,
                            &atom_types_u32,
                            cell_f32,
                            &self.potential,
                        )
                    } else {
                        engine.compute_sync(&positions4, &atom_types_u32, cell_f32, &self.potential)
                    }
                }
            })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // ── Package results ───────────────────────────────────────────────────
        let forces_flat: Vec<f64> = result
            .forces
            .iter()
            .flat_map(|f| [f[0] as f64, f[1] as f64, f[2] as f64])
            .collect();
        let forces_nd = numpy::ndarray::Array2::from_shape_vec((n, 3), forces_flat)
            .expect("forces shape invariant");
        let forces_arr = PyArray2::from_owned_array_bound(py, forces_nd);
        let epa: Vec<f64> = result.energy_per_atom.iter().map(|&e| e as f64).collect();
        let epa_arr = PyArray1::from_vec_bound(py, epa);
        let stress_arr = PyArray1::from_vec_bound(py, result.virial.to_vec());

        Ok((result.energy as f64, forces_arr, epa_arr, stress_arr))
    }

    /// Compute energy, forces, per-atom stress, electron densities, and
    /// embedding energies (CPU backend only).
    ///
    /// Returns everything ``compute_stress`` returns, plus three CPU-only
    /// physics quantities that are not exposed by the GPU path:
    ///
    /// * **Per-atom virial** (shape ``(N, 6)``, units **eV**) — raw
    ///   pair-virial contribution accumulated per atom, split half-and-half
    ///   between the two atoms of each pair.  Matches LAMMPS'
    ///   ``compute stress/atom`` convention so the values drop straight
    ///   into OVITO.  Row-sum equals ``−stress × V`` up to f64 rounding.
    /// * **Electron densities** (shape ``(N,)``) — :math:`\\rho_i`, useful
    ///   for diagnosing surface/defect environments.
    /// * **Embedding energies** (shape ``(N,)``, eV) — :math:`F_\\alpha(\\rho_i)`.
    ///
    /// Returns
    /// -------
    /// energy : float
    /// forces : ndarray (N, 3)
    /// energy_per_atom : ndarray (N,)
    /// stress : ndarray (6,)
    /// virial_per_atom : ndarray (N, 6)
    /// densities : ndarray (N,)
    /// embedding_energies : ndarray (N,)
    ///
    /// Raises
    /// ------
    /// NotImplementedError
    ///     If called on a GPU-backed engine.  Per-atom virial and per-atom
    ///     density/embedding would require N × (6+1+1) extra GPU readbacks
    ///     — the GPU path is deliberately stripped of those for MD
    ///     hot-loop throughput.  Use ``backend="cpu"`` for this method.
    #[pyo3(signature = (positions, atom_types, cell=None))]
    #[allow(clippy::type_complexity)]
    fn compute_per_atom<'py>(
        &mut self,
        py: Python<'py>,
        positions: PyReadonlyArray2<'py, f64>,
        atom_types: PyReadonlyArray1<'py, i32>,
        cell: Option<PyReadonlyArray2<'py, f64>>,
    ) -> PyResult<(
        f64,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    )> {
        use pyo3::exceptions::PyNotImplementedError;

        // ── Input validation ─────────────────────────────────────────────────
        let pos = positions.as_array();
        let types_arr = atom_types.as_array();
        let n = pos.nrows();
        if pos.ncols() != 3 {
            return Err(PyValueError::new_err(format!(
                "positions must be (N, 3), got (N, {})",
                pos.ncols()
            )));
        }
        if n == 0 {
            return Err(PyValueError::new_err("positions must not be empty"));
        }
        if types_arr.len() != n {
            return Err(PyValueError::new_err(format!(
                "atom_types length {} ≠ positions rows {}",
                types_arr.len(),
                n
            )));
        }
        let n_elements = self.potential.elements.len();
        for (i, &t) in types_arr.iter().enumerate() {
            if t < 0 {
                return Err(PyValueError::new_err(format!("negative atom type: {t}")));
            }
            if (t as usize) >= n_elements {
                return Err(PyValueError::new_err(format!(
                    "atom_types[{i}]={t} out of range (n_elem={n_elements})"
                )));
            }
        }

        // ── Only the CPU backend populates per-atom virial/density/embed ─────
        let cpu = match &mut self.backend {
            Backend::Cpu(cpu) => cpu,
            Backend::Gpu(_) => {
                return Err(PyNotImplementedError::new_err(
                    "compute_per_atom requires backend=\"cpu\" — the GPU path \
                     does not read back per-atom virial, density, or embedding \
                     energy for throughput reasons",
                ));
            }
        };

        // ── Convert inputs ───────────────────────────────────────────────────
        let positions4: Vec<[f32; 4]> = (0..n)
            .map(|i| [pos[[i, 0]] as f32, pos[[i, 1]] as f32, pos[[i, 2]] as f32, 0.0])
            .collect();
        let atom_types_u32: Vec<u32> = types_arr.iter().map(|&t| t as u32).collect();
        let cell_f32: Option<[[f32; 3]; 3]> = match &cell {
            Some(c) => {
                let c = c.as_array();
                if c.nrows() != 3 || c.ncols() != 3 {
                    return Err(PyValueError::new_err("cell must be (3, 3)"));
                }
                Some([
                    [c[[0, 0]] as f32, c[[0, 1]] as f32, c[[0, 2]] as f32],
                    [c[[1, 0]] as f32, c[[1, 1]] as f32, c[[1, 2]] as f32],
                    [c[[2, 0]] as f32, c[[2, 1]] as f32, c[[2, 2]] as f32],
                ])
            }
            None => None,
        };

        // ── Dispatch ─────────────────────────────────────────────────────────
        // `compute_per_atom` is the ONE Python entry point that actually
        // needs virial_per_atom / densities / embedding_energies, so it is
        // the only caller using the `_with_per_atom` variant.  Every other
        // caller (`compute`, `compute_stress`, the Rust tests, the bins) goes
        // through the plain `compute_sync` / `compute_cell_list_sync` and
        // therefore skips the N × 6 × 8-byte per-atom virial allocation plus
        // the N-size densities / embedding clones.
        let use_cell_list = self.use_cell_list;
        let result = py
            .allow_threads(|| {
                if use_cell_list {
                    cpu.compute_cell_list_sync_with_per_atom(
                        &positions4,
                        &atom_types_u32,
                        cell_f32,
                        &self.potential,
                    )
                } else {
                    cpu.compute_sync_with_per_atom(
                        &positions4,
                        &atom_types_u32,
                        cell_f32,
                        &self.potential,
                    )
                }
            })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // ── Package results ──────────────────────────────────────────────────
        let forces_flat: Vec<f64> = result
            .forces
            .iter()
            .flat_map(|f| [f[0] as f64, f[1] as f64, f[2] as f64])
            .collect();
        let forces_nd = numpy::ndarray::Array2::from_shape_vec((n, 3), forces_flat)
            .expect("forces shape invariant");
        let forces_arr = PyArray2::from_owned_array_bound(py, forces_nd);

        let epa: Vec<f64> = result.energy_per_atom.iter().map(|&e| e as f64).collect();
        let epa_arr = PyArray1::from_vec_bound(py, epa);

        let stress_arr = PyArray1::from_vec_bound(py, result.virial.to_vec());

        // Per-atom virial — (N, 6) f64.
        let va_flat: Vec<f64> = result
            .virial_per_atom
            .iter()
            .flat_map(|v| [v[0], v[1], v[2], v[3], v[4], v[5]])
            .collect();
        let va_nd = numpy::ndarray::Array2::from_shape_vec((n, 6), va_flat)
            .expect("virial_per_atom shape invariant");
        let va_arr = PyArray2::from_owned_array_bound(py, va_nd);

        let dens: Vec<f64> = result.densities.iter().map(|&d| d as f64).collect();
        let dens_arr = PyArray1::from_vec_bound(py, dens);

        let embed: Vec<f64> = result.embedding_energies.iter().map(|&e| e as f64).collect();
        let embed_arr = PyArray1::from_vec_bound(py, embed);

        Ok((
            result.energy as f64,
            forces_arr,
            epa_arr,
            stress_arr,
            va_arr,
            dens_arr,
            embed_arr,
        ))
    }

    /// Run the GPU pipeline and also return every intermediate CellList buffer.
    ///
    /// Returns
    /// -------
    /// (energy, forces, energy_per_atom, debug)
    ///     Where ``debug`` is ``None`` for the CPU backend or when CellList
    ///     is inactive, and otherwise a ``dict`` with keys:
    ///
    ///     * ``n_atoms``, ``n_cells`` (tuple), ``n_cells_pad`` (tuple),
    ///       ``n_morton`` (int)
    ///     * ``cell_ids``            (N,)        uint32  — pass0a output
    ///     * ``sorted_atoms``        (N,)        uint32  — pass0c output
    ///     * ``cell_start``          (n_morton+1,) uint32 — CPU prefix sum
    ///     * ``cell_counts``         (n_morton,) uint32  — pass0b output
    ///     * ``reordered_positions`` (N, 4)      float32 — pass0d (w = cid)
    ///     * ``reordered_types``     (N,)        uint32  — pass0d output
    ///     * ``densities``           (N,)        float32 — pass1 (Morton order)
    ///     * ``debug_flags``         (32,)       uint32  — shader counters
    ///
    /// This is **slow** (extra GPU→CPU copies) and only intended for tests /
    /// diagnosis.  Do not call inside an MD hot loop.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If the underlying ``compute_sync_with_debug`` fails.
    /// NotImplementedError
    ///     If ``backend="cpu"`` (CPU path has no GPU intermediate state;
    ///     use ``backend="gpu"`` instead).
    #[pyo3(signature = (positions, atom_types, cell=None))]
    fn compute_with_debug<'py>(
        &mut self,
        py: Python<'py>,
        positions: PyReadonlyArray2<'py, f64>,
        atom_types: PyReadonlyArray1<'py, i32>,
        cell: Option<PyReadonlyArray2<'py, f64>>,
    ) -> PyResult<(
        f64,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray1<f64>>,
        Option<Bound<'py, pyo3::types::PyDict>>,
    )> {
        use pyo3::exceptions::PyNotImplementedError;
        use pyo3::types::PyDict;

        let pos = positions.as_array();
        let types_arr = atom_types.as_array();
        let n = pos.nrows();
        if pos.ncols() != 3 {
            return Err(PyValueError::new_err(format!(
                "positions must be (N, 3), got (N, {})",
                pos.ncols()
            )));
        }
        if n == 0 {
            return Err(PyValueError::new_err("positions must not be empty"));
        }
        if types_arr.len() != n {
            return Err(PyValueError::new_err(format!(
                "atom_types length {} ≠ positions rows {}",
                types_arr.len(),
                n
            )));
        }
        let n_elements = self.potential.elements.len();
        for (i, &t) in types_arr.iter().enumerate() {
            if t < 0 {
                return Err(PyValueError::new_err(format!("negative atom type: {t}")));
            }
            if (t as usize) >= n_elements {
                return Err(PyValueError::new_err(format!(
                    "atom_types[{i}]={t} out of range (n_elem={n_elements})"
                )));
            }
        }

        let positions4: Vec<[f32; 4]> = (0..n)
            .map(|i| {
                [
                    pos[[i, 0]] as f32,
                    pos[[i, 1]] as f32,
                    pos[[i, 2]] as f32,
                    0.0,
                ]
            })
            .collect();
        let atom_types_u32: Vec<u32> = types_arr.iter().map(|&t| t as u32).collect();
        let cell_f32: Option<[[f32; 3]; 3]> = match cell {
            Some(c) => {
                let c = c.as_array();
                if c.nrows() != 3 || c.ncols() != 3 {
                    return Err(PyValueError::new_err("cell must be (3, 3)"));
                }
                Some([
                    [c[[0, 0]] as f32, c[[0, 1]] as f32, c[[0, 2]] as f32],
                    [c[[1, 0]] as f32, c[[1, 1]] as f32, c[[1, 2]] as f32],
                    [c[[2, 0]] as f32, c[[2, 1]] as f32, c[[2, 2]] as f32],
                ])
            }
            None => None,
        };

        let (result, debug_opt) = py.allow_threads(|| match &mut self.backend {
            Backend::Gpu(engine) => engine
                .compute_sync_with_debug(&positions4, &atom_types_u32, cell_f32, &self.potential)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string())),
            Backend::Cpu(_) => Err(PyNotImplementedError::new_err(
                "compute_with_debug requires backend=\"gpu\" — CPU has no GPU intermediate state",
            )),
        })?;

        // Convert ComputeResult → NumPy (mirrors compute()).
        let forces_flat: Vec<f64> = result
            .forces
            .iter()
            .flat_map(|f| [f[0] as f64, f[1] as f64, f[2] as f64])
            .collect();
        let forces_nd = numpy::ndarray::Array2::from_shape_vec((n, 3), forces_flat)
            .expect("forces shape invariant violated");
        let forces_arr = PyArray2::from_owned_array_bound(py, forces_nd);
        let epa: Vec<f64> = result.energy_per_atom.iter().map(|&e| e as f64).collect();
        let epa_arr = PyArray1::from_vec_bound(py, epa);

        // Convert CellListDebugReadback → dict of NumPy arrays.
        let dbg_dict: Option<Bound<'py, PyDict>> = match debug_opt {
            None => None,
            Some(d) => {
                let py_d = PyDict::new_bound(py);
                py_d.set_item("n_atoms", d.n_atoms)?;
                py_d.set_item("n_cells", (d.n_cells.0, d.n_cells.1, d.n_cells.2))?;
                py_d.set_item(
                    "n_cells_pad",
                    (d.n_cells_pad.0, d.n_cells_pad.1, d.n_cells_pad.2),
                )?;
                py_d.set_item("n_morton", d.n_morton)?;
                py_d.set_item("cell_ids", PyArray1::from_vec_bound(py, d.cell_ids))?;
                py_d.set_item("sorted_atoms", PyArray1::from_vec_bound(py, d.sorted_atoms))?;
                py_d.set_item("cell_start", PyArray1::from_vec_bound(py, d.cell_start))?;
                py_d.set_item("cell_counts", PyArray1::from_vec_bound(py, d.cell_counts))?;
                // reordered_positions: (N, 4) float32
                let rp_flat: Vec<f32> = d
                    .reordered_positions
                    .iter()
                    .flat_map(|p| [p[0], p[1], p[2], p[3]])
                    .collect();
                let rp_nd = numpy::ndarray::Array2::from_shape_vec((d.n_atoms, 4), rp_flat)
                    .expect("reordered_positions shape invariant violated");
                py_d.set_item(
                    "reordered_positions",
                    PyArray2::from_owned_array_bound(py, rp_nd),
                )?;
                py_d.set_item(
                    "reordered_types",
                    PyArray1::from_vec_bound(py, d.reordered_types),
                )?;
                py_d.set_item("densities", PyArray1::from_vec_bound(py, d.densities))?;
                py_d.set_item(
                    "debug_flags",
                    PyArray1::from_vec_bound(py, d.debug_flags.to_vec()),
                )?;
                Some(py_d)
            }
        };

        Ok((result.energy as f64, forces_arr, epa_arr, dbg_dict))
    }
}

/// Register the `_cream` native module.
#[pymodule]
pub fn _cream(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCreamEngine>()?;
    Ok(())
}
