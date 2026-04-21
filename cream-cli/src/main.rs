//! cream-cli — UNIX-philosophy command-line frontend for CREAM.
//!
//! # Usage
//!
//! ```bash
//! # Single-element, JSON output
//! cream calc --pot Cu.eam.alloy --in cu_fcc.xyz
//!
//! # Multi-element alloy
//! cream calc --pot CuAg.eam.alloy --elem Cu Ag --in cuag.xyz
//!
//! # Read from stdin, write to stdout
//! cat input.xyz | cream calc --pot Cu.eam.alloy --in -
//!
//! # Explicit output file
//! cream calc --pot Cu.eam.alloy --in input.xyz --out result.json
//!
//! # Use Cell List neighbour search
//! cream calc --pot Cu.eam.alloy --in input.xyz --cell-list
//!
//! # No periodic boundary conditions (clusters / molecules)
//! cream calc --pot Cu.eam.alloy --in cluster.xyz --no-pbc
//! ```
//!
//! # XYZ format expected
//! ```text
//! <N>
//! Lattice="ax ay az bx by bz cx cy cz" Properties=species:S:1:pos:R:3
//! Cu  x  y  z
//! Cu  x  y  z
//! ...
//! ```
//!
//! # JSON output schema
//! ```json
//! { "energy_eV": -3.52,
//!   "forces_eV_per_AA": [[fx,fy,fz], ...],
//!   "energy_per_atom_eV": [...] }
//! ```

use clap::{Parser, Subcommand};
use serde::Serialize;
use std::{
    io::{self, BufRead},
    path::{Path, PathBuf},
};

use cream::{
    engine::ComputeEngine,
    potential::{eam::EamPotential, NeighborStrategy},
};

/// Orthorhombic cell matrix (f32) — used locally in CLI and tests.
#[cfg(test)]
fn ortho_cell_f32(lx: f32, ly: f32, lz: f32) -> [[f32; 3]; 3] {
    [[lx, 0.0, 0.0], [0.0, ly, 0.0], [0.0, 0.0, lz]]
}

/// Return type of [`parse_xyz`]: (positions vec4, atom types, optional cell matrix).
type XyzParseResult =
    Result<(Vec<[f32; 4]>, Vec<u32>, Option<[[f32; 3]; 3]>), Box<dyn std::error::Error>>;

// ── CLI definition ────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name    = "cream",
    version = "0.1.0",
    author  = "ss0832",
    about   = "Compute-shader Rust EAM Atomistics — GPU-accelerated force/energy calculator",
    long_about = None,
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compute EAM forces and energy for a given structure.
    Calc {
        /// Path to the EAM potential file (.eam.alloy).
        #[arg(long, short = 'p')]
        pot: PathBuf,

        /// Input XYZ file.  Use `-` to read from stdin.
        #[arg(long, short = 'i')]
        r#in: String,

        /// Output file.  Defaults to stdout.
        #[arg(long, short = 'o')]
        out: Option<PathBuf>,

        /// Element symbols, in the same order as the potential file.
        /// Required for multi-element potentials.
        /// Example: --elem Cu Ag
        #[arg(long, num_args = 1.., value_name = "ELEM")]
        elem: Option<Vec<String>>,

        /// Output format.
        #[arg(long, default_value = "json")]
        format: OutputFormat,

        /// Use the O(N) Cell List neighbour algorithm instead of O(N²) AllPairs.
        #[arg(long)]
        cell_list: bool,

        /// Cell size for the Cell List [Å]. Defaults to the potential cutoff.
        #[arg(long)]
        cell_size: Option<f32>,

        /// Disable periodic boundary conditions (for clusters / molecules).
        #[arg(long)]
        no_pbc: bool,
    },
}

#[derive(Clone, clap::ValueEnum)]
enum OutputFormat {
    Json,
    Xyz,
}

// ── Output JSON schema ────────────────────────────────────────────────────────

#[derive(Serialize)]
struct OutputJson {
    energy_ev: f32,
    forces_ev_per_aa: Vec<[f32; 3]>,
    /// Always empty for GPU engine results; populated by CPU engine only.
    energy_per_atom_ev: Vec<f32>,
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() {
    env_logger::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Calc {
            pot,
            r#in,
            out,
            elem,
            format,
            cell_list,
            cell_size,
            no_pbc,
        } => {
            if let Err(e) = run_calc(
                &pot,
                &r#in,
                out.as_deref(),
                elem.as_deref(),
                format,
                cell_list,
                cell_size,
                no_pbc,
            ) {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_calc(
    pot_path: &Path,
    input: &str,
    output: Option<&Path>,
    elem: Option<&[String]>,
    format: OutputFormat,
    cell_list: bool,
    cell_size: Option<f32>,
    no_pbc: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // ── Load potential ────────────────────────────────────────────────────────
    let potential = EamPotential::from_file(pot_path)?;

    // Validate user-supplied element list against the potential file.
    if let Some(elems) = elem {
        if elems != &potential.elements[..] {
            return Err(format!(
                "--elem {:?} doesn't match potential elements {:?}",
                elems, potential.elements,
            )
            .into());
        }
    }

    // ── Parse input XYZ ───────────────────────────────────────────────────────
    let xyz_src = if input == "-" {
        let stdin = io::stdin();
        stdin
            .lock()
            .lines()
            .collect::<Result<Vec<_>, _>>()?
            .join("\n")
    } else {
        std::fs::read_to_string(input)?
    };

    let (positions4, atom_types, cell) = parse_xyz(&xyz_src, &potential.elements)?;

    // If --no-pbc, discard cell info
    let cell = if no_pbc { None } else { cell };

    // ── Build engine ──────────────────────────────────────────────────────────
    let strategy = if cell_list {
        let cs = cell_size.unwrap_or(potential.cutoff_angstrom);
        NeighborStrategy::CellList { cell_size: cs }
    } else {
        NeighborStrategy::AllPairs
    };

    let mut engine = pollster::block_on(ComputeEngine::new(strategy))?;

    // ── Compute ───────────────────────────────────────────────────────────────
    let result = engine.compute_sync(&positions4, &atom_types, cell, &potential)?;

    // ── Output ────────────────────────────────────────────────────────────────
    let text = match format {
        OutputFormat::Json => {
            let out_json = OutputJson {
                energy_ev: result.energy,
                forces_ev_per_aa: result.forces.clone(),
                energy_per_atom_ev: result.energy_per_atom.clone(), // empty for GPU
            };
            serde_json::to_string_pretty(&out_json)?
        }
        OutputFormat::Xyz => format_xyz_output(
            &positions4,
            &result.forces,
            &potential.elements,
            &atom_types,
            cell.as_ref(),
        ),
    };

    match output {
        Some(p) => std::fs::write(p, text)?,
        None => println!("{text}"),
    }

    Ok(())
}

// ── XYZ parser ────────────────────────────────────────────────────────────────

/// Parse an extended-XYZ file.
/// Returns `(positions_vec4, atom_types, cell)`.
///
/// Supports both full 3×3 Lattice and shorthand diagonal forms:
///   `Lattice="ax ay az bx by bz cx cy cz"` → full triclinic cell
///   `Lattice="Lx Ly Lz"`                   → orthorhombic shorthand
fn parse_xyz(src: &str, elements: &[String]) -> XyzParseResult {
    let mut lines = src.lines();

    // Line 1: atom count
    let n: usize = lines
        .next()
        .ok_or("XYZ: missing atom count")?
        .trim()
        .parse()?;

    // Line 2: comment — look for Lattice="..."
    let comment = lines.next().ok_or("XYZ: missing comment line")?;
    let cell = parse_lattice(comment);

    let mut positions = Vec::with_capacity(n);
    let mut types = Vec::with_capacity(n);

    for (li, line) in lines.take(n).enumerate() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 4 {
            return Err(format!(
                "XYZ line {}: expected 'ELEM x y z', got {:?}",
                li + 3,
                parts
            )
            .into());
        }
        let sym = parts[0];
        let x: f32 = parts[1]
            .parse()
            .map_err(|e| format!("line {}: x: {e}", li + 3))?;
        let y: f32 = parts[2]
            .parse()
            .map_err(|e| format!("line {}: y: {e}", li + 3))?;
        let z: f32 = parts[3]
            .parse()
            .map_err(|e| format!("line {}: z: {e}", li + 3))?;

        let tidx = elements
            .iter()
            .position(|e| e == sym)
            .ok_or_else(|| format!("element '{sym}' not in potential elements {:?}", elements))?;

        positions.push([x, y, z, 0.0]);
        types.push(tidx as u32);
    }

    if positions.len() != n {
        return Err(format!("XYZ: header says {n} atoms but found {}", positions.len()).into());
    }

    Ok((positions, types, cell))
}

/// Extract cell matrix from `Lattice="ax ay az bx by bz cx cy cz"`.
/// Returns `Some([[ax,ay,az],[bx,by,bz],[cx,cy,cz]])` for 9-value form,
/// `Some(ortho_cell_f32(Lx,Ly,Lz))` for 3-value shorthand, or `None`.
fn parse_lattice(comment: &str) -> Option<[[f32; 3]; 3]> {
    let start = comment.find("Lattice=\"")?;
    let rest = &comment[start + 9..];
    let end = rest.find('"')?;
    let nums: Vec<f32> = rest[..end]
        .split_whitespace()
        .filter_map(|s| s.parse().ok())
        .collect();
    if nums.len() >= 9 {
        Some([
            [nums[0], nums[1], nums[2]],
            [nums[3], nums[4], nums[5]],
            [nums[6], nums[7], nums[8]],
        ])
    } else if nums.len() >= 3 {
        let (lx, ly, lz) = (nums[0], nums[1], nums[2]);
        Some([[lx, 0.0, 0.0], [0.0, ly, 0.0], [0.0, 0.0, lz]])
    } else {
        None
    }
}

/// Format results back as extended XYZ with forces.
fn format_xyz_output(
    positions: &[[f32; 4]],
    forces: &[[f32; 3]],
    elements: &[String],
    types: &[u32],
    cell: Option<&[[f32; 3]; 3]>,
) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    writeln!(s, "{}", positions.len()).unwrap();

    // Comment line with Lattice and Properties
    let lattice_str = match cell {
        Some(h) => format!(
            "Lattice=\"{:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}\" ",
            h[0][0], h[0][1], h[0][2], h[1][0], h[1][1], h[1][2], h[2][0], h[2][1], h[2][2],
        ),
        None => String::new(),
    };
    writeln!(s, "{lattice_str}Properties=species:S:1:pos:R:3:forces:R:3").unwrap();

    for (i, pos) in positions.iter().enumerate() {
        let sym = &elements[types[i] as usize];
        let f = forces[i];
        writeln!(
            s,
            "{sym} {:.6} {:.6} {:.6}  {:.6} {:.6} {:.6}",
            pos[0], pos[1], pos[2], f[0], f[1], f[2],
        )
        .unwrap();
    }
    s
}

// ── CLI-level tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_lattice_full_matrix() {
        let comment = r#"Lattice="10.0 0.0 0.0  0.0 12.0 0.0  0.0 0.0 8.0" Properties=species:S:1"#;
        let cell = parse_lattice(comment).unwrap();
        assert!((cell[0][0] - 10.0).abs() < 1e-5);
        assert!((cell[1][1] - 12.0).abs() < 1e-5);
        assert!((cell[2][2] - 8.0).abs() < 1e-5);
        assert!((cell[0][1]).abs() < 1e-5);
    }

    #[test]
    fn parse_lattice_triclinic() {
        let comment = r#"Lattice="10.0 0.0 0.0  1.0 12.0 0.0  0.5 0.5 8.0""#;
        let cell = parse_lattice(comment).unwrap();
        assert!((cell[0][0] - 10.0).abs() < 1e-5);
        assert!((cell[1][0] - 1.0).abs() < 1e-5);
        assert!((cell[2][2] - 8.0).abs() < 1e-5);
    }

    #[test]
    fn parse_lattice_shorthand() {
        let comment = r#"Lattice="10 12 8""#;
        let cell = parse_lattice(comment).unwrap();
        assert!((cell[0][0] - 10.0).abs() < 1e-5);
        assert!((cell[1][1] - 12.0).abs() < 1e-5);
        assert!((cell[2][2] - 8.0).abs() < 1e-5);
    }

    #[test]
    fn parse_lattice_missing() {
        assert!(parse_lattice("just a comment").is_none());
    }

    #[test]
    fn parse_xyz_4atom() {
        let xyz = "\
4
Lattice=\"10 0 0 0 10 0 0 0 10\"
Cu 0.0 0.0 0.0
Cu 1.8 1.8 0.0
Cu 1.8 0.0 1.8
Cu 0.0 1.8 1.8
";
        let (pos, types, cell) = parse_xyz(xyz, &["Cu".to_string()]).unwrap();
        assert_eq!(pos.len(), 4);
        assert_eq!(types, vec![0u32; 4]);
        let c = cell.unwrap();
        assert!((c[0][0] - 10.0).abs() < 1e-4);
    }

    #[test]
    fn parse_xyz_no_lattice() {
        let xyz = "1\ncomment\nCu 0.0 0.0 0.0\n";
        let (_pos, _types, cell) = parse_xyz(xyz, &["Cu".to_string()]).unwrap();
        assert!(cell.is_none());
    }

    #[test]
    fn parse_xyz_unknown_element_errors() {
        let xyz = "1\n\nFe 0.0 0.0 0.0\n";
        assert!(parse_xyz(xyz, &["Cu".to_string()]).is_err());
    }

    #[test]
    fn format_xyz_roundtrip() {
        let positions = vec![[0.0f32, 0.0, 0.0, 0.0], [1.8, 0.0, 0.0, 0.0]];
        let forces = vec![[0.1f32, 0.0, 0.0], [-0.1, 0.0, 0.0]];
        let elements = vec!["Cu".to_string()];
        let types = vec![0u32, 0];
        let cell = Some(ortho_cell_f32(10.0, 10.0, 10.0));
        let out = format_xyz_output(&positions, &forces, &elements, &types, cell.as_ref());
        assert!(out.starts_with("2\n"));
        assert!(out.contains("Lattice="));
        assert!(out.contains("Cu"));
    }
}
