//! WebAssembly bindings for CREAM.
//!
//! Exposes [`CreamWasm`] to JavaScript via `wasm-bindgen`.  The interface is
//! intentionally thin: the browser passes raw typed arrays and receives a plain
//! JS object back, keeping the coupling to zero.
//!
//! # Design decisions
//!
//! ## No `pollster::block_on`
//! WASM has no native threads, so blocking the event loop is forbidden.
//! Every public async method on `CreamWasm` is compiled by `wasm-bindgen`
//! (>= 0.2.87) into a JavaScript `Promise` automatically.
//!
//! ## Potential loaded from bytes, not a path
//! Browsers cannot open arbitrary file-system paths.  Callers supply the
//! `.eam.alloy` file contents as a `Uint8Array`; `EamPotential::from_str`
//! handles parsing.
//!
//! ## Flat array I/O
//! Positions are packed as `[x0,y0,z0, x1,y1,z1, ...]` (stride 3).
//! `wasm-bindgen` maps Rust `Vec<f64>` / `Vec<u32>` directly to
//! `Float64Array` / `Uint32Array` in JS.
//!
//! ## Result object shape
//! ```json
//! {
//!   "energy":        -3.54,
//!   "forces":        [-0.1, 0.0, 0.1, ...],
//!   "energyPerAtom": [-1.77, -1.77, ...]
//! }
//! ```
//!
//! ## WebGPU limit handling
//! `ComputeEngine::new_webgpu` queries the adapter's actual
//! `max_storage_buffers_per_shader_stage` at runtime.  If it cannot supply
//! the 10 buffers required by Cell List pass 2, the strategy is silently
//! downgraded to AllPairs (see `engine.rs`).
//!
//! ## WebWorker usage (recommended for large N)
//! ```js
//! // worker.js
//! import init, { CreamWasm } from './cream.js';
//! await init();
//! const engine = await CreamWasm.new(potBytes, false, undefined);
//! self.onmessage = async ({ data: { positions, atomTypes, cell } }) => {
//!     const result = await engine.compute(positions, atomTypes, cell);
//!     self.postMessage(result);
//! };
//! ```

#![cfg(target_arch = "wasm32")]

use js_sys::Error as JsError;
use serde::Serialize;
use wasm_bindgen::prelude::*;

use crate::{
    engine::ComputeEngine,
    error::CreamError,
    potential::{eam::EamPotential, NeighborStrategy},
};

// ── JS-serialisable result ────────────────────────────────────────────────────

/// Serialised via `serde-wasm-bindgen` into a plain JS object with camelCase keys.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ComputeOutput {
    energy: f64,
    forces: Vec<f64>,
    energy_per_atom: Vec<f64>,
}

// ── CreamWasm ─────────────────────────────────────────────────────────────────

/// GPU-accelerated EAM engine for the browser (WebGPU backend).
///
/// All expensive methods return a `Promise` and must be `await`-ed in JS.
///
/// # JavaScript example
/// ```js
/// import init, { CreamWasm } from './cream.js';
/// await init();
///
/// const resp     = await fetch('Cu.eam.alloy');
/// const potBytes = new Uint8Array(await resp.arrayBuffer());
///
/// // Construct (returns Promise<CreamWasm>)
/// const engine = await CreamWasm.new(potBytes, false, undefined);
/// console.log('elements:', engine.elements);  // ['Cu']
/// console.log('cutoff:', engine.cutoff);      // e.g. 6.5
///
/// // Two-atom cluster (no PBC)
/// const positions = new Float64Array([0, 0, 0,  1.8, 1.8, 0]);
/// const types     = new Uint32Array([0, 0]);
/// const { energy, forces, energyPerAtom } =
///     await engine.compute(positions, types, undefined);
/// ```
#[wasm_bindgen]
pub struct CreamWasm {
    engine: ComputeEngine,
    potential: EamPotential,
}

#[wasm_bindgen]
impl CreamWasm {
    // ── Constructor ───────────────────────────────────────────────────────────

    /// Parse the potential and initialise the WebGPU engine.
    ///
    /// * `potential_bytes` – contents of a `.eam.alloy` file as `Uint8Array`.
    /// * `use_cell_list`   – `true` for O(N) Cell List; `false` for O(N²)
    ///   AllPairs.  Auto-downgraded to AllPairs when the WebGPU adapter
    ///   cannot provide the 10 storage buffers needed by Cell List pass 2.
    /// * `cell_size`       – Cell List edge length [Å]; `undefined` uses
    ///   the potential cutoff radius.
    #[wasm_bindgen(constructor)]
    pub async fn new(
        potential_bytes: Vec<u8>,
        use_cell_list: bool,
        cell_size: Option<f32>,
    ) -> Result<CreamWasm, JsValue> {
        let src = std::str::from_utf8(&potential_bytes)
            .map_err(|e| js_err(&format!("potential_bytes is not valid UTF-8: {e}")))?;

        let potential = EamPotential::from_str(src).map_err(js_cream)?;

        let strategy = if use_cell_list {
            NeighborStrategy::CellList {
                cell_size: cell_size.unwrap_or(potential.cutoff_angstrom),
            }
        } else {
            NeighborStrategy::AllPairs
        };

        let engine = ComputeEngine::new_webgpu(strategy)
            .await
            .map_err(js_cream)?;

        Ok(Self { engine, potential })
    }

    // ── Compute ───────────────────────────────────────────────────────────────

    /// Compute EAM energy and forces.
    ///
    /// * `positions`  – `Float64Array` of length `N*3`, row-major
    ///   `[x0,y0,z0, x1,y1,z1, ...]` in Å.
    /// * `atom_types` – `Uint32Array` of length `N`; 0-based indices into the
    ///   potential file's element list.
    /// * `cell`       – `Float64Array` of length 9 (row-major 3x3 cell matrix
    ///   `[ax,ay,az, bx,by,bz, cx,cy,cz]`) in Å, or `undefined` / `null`
    ///   for non-periodic calculations.
    ///
    /// Resolves with `{ energy: number, forces: number[], energyPerAtom: number[] }`.
    #[wasm_bindgen]
    pub async fn compute(
        &mut self,
        positions: Vec<f64>,
        atom_types: Vec<u32>,
        cell: Option<Vec<f64>>,
    ) -> Result<JsValue, JsValue> {
        let n = validate_inputs(&positions, &atom_types, cell.as_deref())?;

        // f64 → f32 vec4 (w=0 padding required by GPU layout)
        let positions4: Vec<[f32; 4]> = (0..n)
            .map(|i| {
                [
                    positions[i * 3] as f32,
                    positions[i * 3 + 1] as f32,
                    positions[i * 3 + 2] as f32,
                    0.0_f32,
                ]
            })
            .collect();

        // Optional cell matrix: flat 9-element row-major → [[f32;3];3]
        let cell_f32: Option<[[f32; 3]; 3]> = cell.as_deref().map(|c| {
            [
                [c[0] as f32, c[1] as f32, c[2] as f32],
                [c[3] as f32, c[4] as f32, c[5] as f32],
                [c[6] as f32, c[7] as f32, c[8] as f32],
            ]
        });

        let result = self
            .engine
            .compute(&positions4, &atom_types, cell_f32, &self.potential)
            .await
            .map_err(js_cream)?;

        // Pack forces as flat f64 vec (row-major, stride 3)
        let forces_flat: Vec<f64> = result
            .forces
            .iter()
            .flat_map(|f| [f[0] as f64, f[1] as f64, f[2] as f64])
            .collect();

        let output = ComputeOutput {
            energy: result.energy as f64,
            forces: forces_flat,
            energy_per_atom: vec![], // GPU path no longer computes per-atom decomposition
        };

        serde_wasm_bindgen::to_value(&output).map_err(|e| js_err(&e.to_string()))
    }

    // ── Properties ────────────────────────────────────────────────────────────

    /// Element symbols in the potential file order, e.g. `["Cu"]` or `["Cu","Ag"]`.
    #[wasm_bindgen(getter)]
    pub fn elements(&self) -> Vec<JsValue> {
        self.potential
            .elements
            .iter()
            .map(|s| JsValue::from_str(s))
            .collect()
    }

    /// Cutoff radius [Å].
    #[wasm_bindgen(getter)]
    pub fn cutoff(&self) -> f32 {
        self.potential.cutoff_angstrom
    }

    /// Number of element species.
    #[wasm_bindgen(getter, js_name = "nElements")]
    pub fn n_elements(&self) -> usize {
        self.potential.elements.len()
    }

    /// Whether Cell List mode is active.
    ///
    /// May be `false` even when `use_cell_list=true` was passed to the
    /// constructor, if the WebGPU adapter could not supply enough storage
    /// buffers and the strategy was automatically downgraded to AllPairs.
    #[wasm_bindgen(getter, js_name = "usesCellList")]
    pub fn uses_cell_list(&self) -> bool {
        matches!(self.engine.strategy(), NeighborStrategy::CellList { .. })
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Validate that input arrays have consistent lengths and shapes.
/// Returns `n_atoms` on success.
fn validate_inputs(
    positions: &[f64],
    atom_types: &[u32],
    cell: Option<&[f64]>,
) -> Result<usize, JsValue> {
    if positions.len() % 3 != 0 {
        return Err(js_err(&format!(
            "positions length {} is not a multiple of 3 (expected N*3)",
            positions.len()
        )));
    }
    let n = positions.len() / 3;

    if atom_types.len() != n {
        return Err(js_err(&format!(
            "atom_types length {} does not match positions atom count {}",
            atom_types.len(),
            n
        )));
    }

    if let Some(c) = cell {
        if c.len() != 9 {
            return Err(js_err(&format!(
                "cell must be a Float64Array of length 9 (3x3 row-major), got {}",
                c.len()
            )));
        }
    }

    Ok(n)
}

/// Wrap `CreamError` as a JS `Error` for idiomatic `Promise.catch()` / `try/catch`.
fn js_cream(e: CreamError) -> JsValue {
    JsError::new(&e.to_string()).into()
}

/// Wrap a plain string as a JS `Error`.
fn js_err(msg: &str) -> JsValue {
    JsError::new(msg).into()
}
