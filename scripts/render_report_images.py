#!/usr/bin/env python3
"""
Generate report images using the headless visualization GUI.
Iterates over boundary conditions and initialization modes as specified in the report.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

def load_config_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines(keepends=True)

def write_config(path: Path, lines: list[str], boundary: str, init_mode: str) -> None:
    """Updates the config lines with specific boundary and init modes."""
    in_grid = False
    boundary_set = False
    init_set = False
    updated = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_grid = stripped.lower() == "[grid]"

        if in_grid and stripped.startswith("boundary_mode"):
            updated.append(f"boundary_mode = {boundary}\n")
            boundary_set = True
            continue
        if in_grid and stripped.startswith("init_mode"):
            updated.append(f"init_mode = {init_mode}\n")
            init_set = True
            continue

        updated.append(line)

    if not boundary_set:
        updated.append(f"\n[grid]\nboundary_mode = {boundary}\n")
    if not init_set:
        updated.append(f"init_mode = {init_mode}\n")

    path.write_text("".join(updated), encoding="utf-8")

def run_headless_render(gui_exe: Path, db_path: Path, cfg_path: Path, backend: str, output_name: Path):
    """Launches the GUI in headless mode, runs simulation, and exports images."""
    cmd = [
        str(gui_exe),
        "--headless",
        "-platform", "offscreen",
        "--database", str(db_path),
        "--config", str(cfg_path),
        "--backend", backend,
        "--width", "1200",
        "--height", "900"
    ]

    print(f"Running: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    run_id = None
    try:
        # 1. Wait for session start
        for line in process.stdout:
            print(f"  [GUI] {line.strip()}")
            if "CLI_SESSION_STARTED" in line:
                # Extract run_id (expected format: CLI_SESSION_STARTED run_id=XYZ)
                parts = line.strip().split("run_id=")
                if len(parts) > 1:
                    run_id = parts[1]
                break
        
        if not run_id:
            print("  Error: Failed to get run_id from GUI")
            process.terminate()
            return False

        # 2. Wait for simulation completion
        for line in process.stdout:
            print(f"  [GUI] {line.strip()}")
            if "CLI_SIMULATION_FINISHED" in line:
                break
        
        # 3. Request Export
        print(f"  Requesting export to: {output_name}")
        process.stdin.write(f"EXPORT_VISUALS {output_name}\n")
        process.stdin.flush()

        # 4. Wait for export confirmation then quit
        for line in process.stdout:
            print(f"  [GUI] {line.strip()}")
            if "CLI_VISUALS_EXPORTED" in line:
                process.stdin.write("QUIT\n")
                process.stdin.flush()
                break

        process.wait(timeout=10)
        return True

    except Exception as e:
        print(f"  Process Error: {e}")
        process.kill()
        return False

def main():
    parser = argparse.ArgumentParser(description="Batch render report images.")
    parser.add_argument("--gui", default="build/visualization_gui/salr_gui", help="Path to GUI executable")
    parser.add_argument("--database", default="database", help="Path to database directory")
    parser.add_argument("--config", default="configs/default.cfg", help="Base config file")
    parser.add_argument("--output", default="docs/report/SALR3YUCUDA/src", help="Output directory for images")
    parser.add_argument("--backend", default="cuda", choices=["cpu", "cuda"], help="Compute backend")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    gui_exe = (root / args.gui).resolve()
    db_path = (root / args.database).resolve()
    base_cfg = (root / args.config).resolve()
    out_dir = (root / args.output).resolve()
    workdir = (root / "output/report_configs").resolve()

    out_dir.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)

    # Scenarios defined in the report 
    scenarios = [
        {"name": "rep_pbc", "boundary": "PBC", "init": "sinusoids"},
        {"name": "rep_w2",  "boundary": "W2",  "init": "sinusoids"},
        {"name": "rep_w4",  "boundary": "W4",  "init": "sinusoids"},
    ]

    config_lines = load_config_lines(base_cfg)

    for sc in scenarios:
        print(f"\n>>> Generating: {sc['name']} ({sc['boundary']}, {sc['init']})")
        
        cfg_path = workdir / f"{sc['name']}.cfg"
        write_config(cfg_path, config_lines, sc['boundary'], sc['init'])
        
        # The GUI will append _scatter.png and _heatmap.png to this base path
        output_base = out_dir / sc['name']
        
        success = run_headless_render(gui_exe, db_path, cfg_path, args.backend, output_base)
        if not success:
            print(f"!!! Failed to generate {sc['name']}")

    print("\nBatch processing complete.")

if __name__ == "__main__":
    main()