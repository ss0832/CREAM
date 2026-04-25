#!/usr/bin/env python3
"""Run CREAM-vs-reference comparison over the 2x2 backend matrix.

Runs, for each potential file:
  1. cpu_allpairs  : backend=cpu, use_cell_list=False
  2. cpu_celllist  : backend=cpu, use_cell_list=True
  3. gpu_allpairs  : backend=gpu, use_cell_list=False
  4. gpu_celllist  : backend=gpu, use_cell_list=True

Each run writes a JSON result and a plain-text log. A combined summary.json and
summary.md are also generated.

Example:
  python run_all_backend_matrix.py Cu01.eam.alloy Mishin-Ni-Al-2009.eam.alloy --out-dir reference_matrix_results

Extra arguments after `--` are forwarded to compare_against_reference.py, e.g.:
  python run_all_backend_matrix.py Mishin-Ni-Al-2009.eam.alloy -- --suite binary --binary-elements Ni Al
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Mode:
    name: str
    backend: str
    use_cell_list: bool


DEFAULT_MODES = [
    Mode("cpu_allpairs", "cpu", False),
    Mode("cpu_celllist", "cpu", True),
    Mode("gpu_allpairs", "gpu", False),
    Mode("gpu_celllist", "gpu", True),
]


def _safe_stem(path: Path) -> str:
    stem = path.name
    for suffix in (".eam.alloy", ".alloy", ".eam"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in stem)


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _fmt_float(x: Any, fmt: str = ".3e") -> str:
    if x is None:
        return "—"
    try:
        return format(float(x), fmt)
    except Exception:
        return str(x)


def _summary_row_from_payload(potential: Path, mode: Mode, returncode: int, elapsed_s: float, payload: dict[str, Any] | None, log_path: Path, json_path: Path) -> dict[str, Any]:
    summary = payload.get("summary", {}) if payload else {}
    return {
        "potential": str(potential),
        "potential_name": potential.name,
        "mode": mode.name,
        "backend": mode.backend,
        "use_cell_list": mode.use_cell_list,
        "returncode": returncode,
        "elapsed_s": elapsed_s,
        "n_cases": summary.get("n_cases"),
        "n_passed": summary.get("n_passed"),
        "n_failed": summary.get("n_failed"),
        "max_abs_energy_error_eV_per_atom": summary.get("max_abs_energy_error_eV_per_atom"),
        "max_abs_force_error_eV_per_A": summary.get("max_abs_force_error_eV_per_A"),
        "max_rms_force_error_eV_per_A": summary.get("max_rms_force_error_eV_per_A"),
        "max_abs_stress_error_eV_per_A3": summary.get("max_abs_stress_error_eV_per_A3"),
        "max_abs_stress_error_GPa": summary.get("max_abs_stress_error_GPa"),
        "json": str(json_path),
        "log": str(log_path),
    }


def _write_summary_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    lines.append("# CREAM external-reference backend matrix summary")
    lines.append("")
    lines.append("This report summarizes `compare_against_reference.py` over CPU/GPU and all-pairs/cell-list modes.")
    lines.append("")
    by_potential: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_potential.setdefault(row["potential_name"], []).append(row)

    for pot_name, pot_rows in by_potential.items():
        lines.append(f"## {pot_name}")
        lines.append("")
        lines.append("| Mode | Return | Cases | Passed | Failed | max abs dE/N (eV/atom) | max abs dF (eV/A) | max RMS dF (eV/A) | max abs dS (GPa) | Log | JSON |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
        for row in pot_rows:
            status = "PASS" if row.get("returncode") == 0 and row.get("n_failed") in (0, None) else "FAIL"
            log_name = Path(row["log"]).name
            json_name = Path(row["json"]).name
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["mode"],
                        f"{status} ({row.get('returncode')})",
                        str(row.get("n_cases") if row.get("n_cases") is not None else "—"),
                        str(row.get("n_passed") if row.get("n_passed") is not None else "—"),
                        str(row.get("n_failed") if row.get("n_failed") is not None else "—"),
                        _fmt_float(row.get("max_abs_energy_error_eV_per_atom")),
                        _fmt_float(row.get("max_abs_force_error_eV_per_A")),
                        _fmt_float(row.get("max_rms_force_error_eV_per_A")),
                        _fmt_float(row.get("max_abs_stress_error_GPa")),
                        log_name,
                        json_name,
                    ]
                )
                + " |"
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    # Split runner arguments from compare_against_reference.py arguments explicitly.
    # Do not use argparse.REMAINDER here: with a variable-length positional
    # (`potentials`, nargs="+"), REMAINDER can greedily capture runner options
    # such as --out-dir and forward them to compare_against_reference.py.
    argv = sys.argv[1:]
    if "--" in argv:
        sep = argv.index("--")
        runner_argv = argv[:sep]
        forwarded = argv[sep + 1 :]
    else:
        runner_argv = argv
        forwarded = []

    parser = argparse.ArgumentParser(
        description="Run CREAM-vs-reference comparison over cpu/gpu x allpairs/celllist modes. Arguments after -- are forwarded to compare_against_reference.py."
    )
    parser.add_argument("potentials", nargs="+", type=Path, help="One or more .eam.alloy potential files")
    parser.add_argument("--out-dir", type=Path, default=Path("reference_matrix_results"))
    parser.add_argument("--compare-script", type=Path, default=Path(__file__).with_name("compare_against_reference.py"))
    parser.add_argument("--modes", nargs="*", default=[m.name for m in DEFAULT_MODES], choices=[m.name for m in DEFAULT_MODES])
    parser.add_argument("--continue-on-error", action="store_true", help="Run remaining modes even if one mode fails")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to launch compare_against_reference.py")
    args = parser.parse_args(runner_argv)

    mode_map = {m.name: m for m in DEFAULT_MODES}
    modes = [mode_map[name] for name in args.modes]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    overall_ok = True

    for potential in args.potentials:
        potential = potential.resolve()
        if not potential.exists():
            raise FileNotFoundError(potential)
        pot_dir = args.out_dir / _safe_stem(potential)
        pot_dir.mkdir(parents=True, exist_ok=True)

        for mode in modes:
            json_path = pot_dir / f"{mode.name}.json"
            log_path = pot_dir / f"{mode.name}.log"
            cmd = [
                args.python,
                str(args.compare_script.resolve()),
                str(potential),
                "--backend",
                mode.backend,
                "--json-out",
                str(json_path),
            ]
            if mode.use_cell_list:
                cmd.append("--use-cell-list")
            cmd.extend(forwarded)

            print("=" * 88)
            print(f"Potential : {potential.name}")
            print(f"Mode      : {mode.name}")
            print("Command   : " + " ".join(cmd))
            print("=" * 88)

            t0 = time.perf_counter()
            proc = subprocess.run(cmd, cwd=str(args.compare_script.resolve().parent), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            elapsed = time.perf_counter() - t0
            log_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
            print(proc.stdout)
            print(f"[{potential.name} / {mode.name}] returncode={proc.returncode}, elapsed={elapsed:.2f}s")
            print(f"log : {log_path}")
            print(f"json: {json_path}")

            payload = _load_json(json_path) if json_path.exists() else None
            row = _summary_row_from_payload(potential, mode, proc.returncode, elapsed, payload, log_path, json_path)
            rows.append(row)

            if proc.returncode != 0:
                overall_ok = False
                if not args.continue_on_error:
                    summary_payload = {"rows": rows, "overall_ok": False}
                    (args.out_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
                    _write_summary_md(args.out_dir / "summary.md", rows)
                    raise SystemExit(proc.returncode)

    summary_payload = {"rows": rows, "overall_ok": overall_ok}
    (args.out_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    _write_summary_md(args.out_dir / "summary.md", rows)
    print("=" * 88)
    print(f"Wrote summary JSON: {args.out_dir / 'summary.json'}")
    print(f"Wrote summary MD  : {args.out_dir / 'summary.md'}")
    if not overall_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
