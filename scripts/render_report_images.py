#!/usr/bin/env python3
"""
Generate report images using the headless visualization GUI.
Iterates over boundary conditions as specified in the report and formats
the exported images to exactly match the LaTeX \\includegraphics commands.
"""

import argparse
import subprocess
import sys
import time
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
    export_requested = False
    export_successful = False

    try:
        for line in process.stdout:
            clean_line = line.strip()
            print(f"  [GUI] {clean_line}")

            if "Created run:" in clean_line or "CLI_SESSION_STARTED" in clean_line:
                session_started = True

            if "CLI_SESSION_FINISHED" in clean_line:
                if not session_started:
                    print("  Error: Solver failed to start.")
                    process.kill()
                    return False

            # Wait for the snapshot to ACTUALLY load into the OpenGL widget before exporting
            if "CLI_SNAPSHOT_LOADED" in clean_line and not export_requested:
                print(f"  Snapshot loaded. Requesting export to {output_base}...")
                process.stdin.write(f"EXPORT_VISUALS {output_base.absolute()}\n")
                process.stdin.flush()
                export_requested = True

            # DO NOT break until we get confirmation that the files are actually on disk
            if "CLI_VISUALS_EXPORTED" in clean_line:
                print("  Export confirmed by GUI. Quitting...")
                export_successful = True
                process.stdin.write("QUIT\n")
                process.stdin.flush()
                break
            
            if "CLI_ERROR" in clean_line and "Failed to export" in clean_line:
                print("  GUI reported failure to export.")
                process.stdin.write("QUIT\n")
                process.stdin.flush()
                break

        process.wait(timeout=10)
        
        if not export_successful:
             print("  -> ERROR: Process exited before export could complete.")
             return False

        # --- POST-PROCESSING FOR LATEX ---
        # The GUI automatically saves files as base_heatmap.png and base_scatter.png.
        # Your LaTeX report expects just base.png (e.g., rep_pbc.png) for the heatmap.
        heatmap_path = output_base.parent / f"{output_base.name}_heatmap.png"
        final_path = output_base.parent / f"{output_base.name}.png"
        
        # Add a tiny sleep just in case the OS hasn't fully flushed the file to the directory visible state
        time.sleep(0.5) 

        if heatmap_path.exists():
            # Rename the heatmap so it drops the "_heatmap" suffix
            heatmap_path.replace(final_path)
            print(f"  -> SUCCESS: Created LaTeX-ready image at '{final_path}'")
            return True
        else:
            print(f"  -> ERROR: Expected heatmap at {heatmap_path.name} was not found on disk!")
            return False

    except Exception as e:
        print(f"  Exception occurred: {e}")
        process.kill()
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="cuda", choices=["cuda", "cpu"], help="Solver backend")
    args = parser.parse_args()

    # Absolute paths mapped to the repository root
    root = Path(__file__).resolve().parents[1]
    gui_exe = root / "build/visualization_gui/salr_gui"
    db_path = root / "database"
    base_cfg = root / "configs/default.cfg"
    
    # Target directory strictly specified for the LaTeX report
    out_dir = root / "docs/report/SALR3YUCUDA/src"
    workdir = root / "output/report_configs"

    out_dir.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)

    # Scenarios defined exactly by the LaTeX document includes
    scenarios = [
        {"name": "rep_pbc", "boundary": "PBC", "init": "sinusoids"},
        {"name": "rep_w2",  "boundary": "W2",  "init": "sinusoids"},
        {"name": "rep_w4",  "boundary": "W4",  "init": "sinusoids"},
    ]

    lines = load_config_lines(base_cfg)

    for sc in scenarios:
        print(f"\n{'='*50}")
        print(f">>> Generating Report Image: {sc['name']}.png")
        print(f"{'='*50}")
        
        cfg_path = workdir / f"{sc['name']}.cfg"
        write_config(cfg_path, lines, sc['boundary'], sc['init'])
        
        # Base path without extension. GUI adds _heatmap.png, which we then strip.
        output_base = out_dir / sc['name']
        
        success = run_headless_render(gui_exe, db_path, cfg_path, args.backend, output_base)
        
        if not success:
            print(f"!!! Failed to generate visuals for {sc['name']}")

    print("\nBatch generation complete! Your images are ready in 'docs/report/SALR3YUCUDA/src/'")

if __name__ == "__main__":
    main()