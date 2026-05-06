#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

def load_config_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines(keepends=True)

def write_config(path: Path, lines: list[str], boundary: str, init_mode: str) -> None:
    in_grid = False
    boundary_set = False
    init_set = False
    updated = []

    for line in lines:
        # Strip inline comments to prevent Qt QSettings parsing errors
        if '#' in line and not line.lstrip().startswith('#'):
            line = line.split('#')[0].rstrip() + '\n'

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

def run_headless_render(gui_exe: Path, db_path: Path, cfg_path: Path, backend: str, output_base: Path):
    # Ensure the solver is in the PATH or the same dir as GUI
    # We pass the absolute path to the database to avoid ambiguity
    cmd = [
        str(gui_exe),
        "--headless",
        "-platform", "offscreen",
        "--database", str(db_path.absolute()),
        "--config", str(cfg_path.absolute()),
        "--backend", backend,
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

    session_started = False
    run_id = None

    try:
        for line in process.stdout:
            clean_line = line.strip()
            print(f"  [GUI] {clean_line}")

            if "CLI_SESSION_STARTED" in clean_line:
                session_started = True
                # Try to parse run_id
                if "run_id=" in clean_line:
                    run_id = clean_line.split("run_id=")[1].split()[0]

            if "CLI_SESSION_FINISHED" in clean_line:
                if "run_id=starting" in clean_line or "converged=false" in clean_line:
                    # If it finished without converging or starting, it likely crashed
                    if not session_started:
                        print("  Error: Solver failed to start (Check if salr_dft_cuda_db exists in build folder)")
                        process.kill()
                        return False
                
                # If we get here, simulation is "done" (either converged or hit max iter)
                print(f"  Simulation complete. Exporting to {output_base}...")
                process.stdin.write(f"EXPORT_VISUALS {output_base.absolute()}\n")
                process.stdin.flush()

            if "CLI_VISUALS_EXPORTED" in clean_line:
                print("  Export successful. Quitting.")
                process.stdin.write("QUIT\n")
                process.stdin.flush()
                break

        process.wait(timeout=5)
        return True
    except Exception as e:
        print(f"  Exception: {e}")
        process.kill()
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="cuda")
    args = parser.parse_args()

    # Setup paths relative to script location
    root = Path(__file__).resolve().parents[1]
    gui_exe = root / "build/visualization_gui/salr_gui"
    db_path = root / "database"
    base_cfg = root / "configs/default.cfg"
    out_dir = root / "docs/report/SALR3YUCUDA/src"
    workdir = root / "output/report_configs"

    out_dir.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)

    scenarios = [
        {"name": "rep_pbc", "boundary": "PBC", "init": "sinusoids"},
        {"name": "rep_w2",  "boundary": "W2",  "init": "sinusoids"},
        {"name": "rep_w4",  "boundary": "W4",  "init": "sinusoids"},
    ]

    lines = load_config_lines(base_cfg)

    for sc in scenarios:
        print(f"\n>>> Case: {sc['name']}")
        cfg_path = workdir / f"{sc['name']}.cfg"
        write_config(cfg_path, lines, sc['boundary'], sc['init'])
        
        output_base = out_dir / sc['name']
        run_headless_render(gui_exe, db_path, cfg_path, args.backend, output_base)

if __name__ == "__main__":
    main()