#!/usr/bin/env python3
"""
Generate report images using the headless visualization GUI.
Iterates over boundary conditions as specified in the report and formats
the exported images to exactly match the LaTeX \\includegraphics commands.
"""

import argparse
import select
import subprocess
import sys
import time
from pathlib import Path

def load_config_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines(keepends=True)

def write_config(path: Path, lines: list[str], overrides: dict[str, dict[str, str]]) -> None:
    current_section = None
    pending = {section: set(values.keys()) for section, values in overrides.items()}
    updated = []

    for line in lines:
        if '#' in line and not line.lstrip().startswith('#'):
            line = line.split('#')[0].rstrip() + '\n'

        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current_section = stripped.strip("[]").lower()

        if current_section in overrides and "=" in stripped and not stripped.startswith("#"):
            key = stripped.split("=", 1)[0].strip()
            if key in overrides[current_section]:
                updated.append(f"{key} = {overrides[current_section][key]}\n")
                pending[current_section].discard(key)
                continue

        updated.append(line)

    for section, keys in pending.items():
        if not keys:
            continue
        updated.append(f"\n[{section}]\n")
        for key in sorted(keys):
            updated.append(f"{key} = {overrides[section][key]}\n")

    path.write_text("".join(updated), encoding="utf-8")

def run_headless_render(gui_exe: Path, db_path: Path, cfg_path: Path, backend: str,
                        output_base: Path, width: int, height: int, platform: str, timeout: int) -> bool:
    cmd = [
        str(gui_exe),
        "--headless",
        "--database", str(db_path.absolute()),
        "--config", str(cfg_path.absolute()),
        "--backend", backend,
        "--width", str(width),
        "--height", str(height),
    ]
    if platform:
        cmd.extend(["-platform", platform])

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
    finished = False
    snapshot_requested = False
    export_requested = False
    export_successful = False
    started_at = time.time()

    try:
        while True:
            if time.time() - started_at > timeout:
                print("  Timeout exceeded, terminating GUI process.")
                process.kill()
                return False

            if process.poll() is not None:
                break

            ready, _, _ = select.select([process.stdout], [], [], 0.5)
            if not ready:
                continue

            line = process.stdout.readline()
            if not line:
                continue

            clean_line = line.strip()
            print(f"  [GUI] {clean_line}")

            if "CLI_SESSION_STARTED" in clean_line:
                session_started = True

            if "CLI_SESSION_FINISHED" in clean_line:
                finished = True
                if not snapshot_requested and process.stdin:
                    process.stdin.write("LOAD_SNAPSHOT latest\n")
                    process.stdin.flush()
                    snapshot_requested = True

            if "CLI_SNAPSHOT_LOADED" in clean_line and not export_requested:
                print(f"  Snapshot loaded. Requesting export to {output_base}...")
                if process.stdin:
                    process.stdin.write(f"EXPORT_VISUALS {output_base.absolute()}\n")
                    process.stdin.flush()
                    export_requested = True

            if "CLI_EXPORT_DONE" in clean_line:
                export_successful = True
                if process.stdin:
                    process.stdin.write("QUIT\n")
                    process.stdin.flush()
                break

            if "CLI_ERROR" in clean_line:
                print("  GUI reported an error.")
                if process.stdin:
                    process.stdin.write("QUIT\n")
                    process.stdin.flush()
                break

        process.wait(timeout=10)

        if not session_started:
            print("  -> ERROR: Session never started.")
            return False
        if not finished:
            print("  -> ERROR: Session did not finish.")
            return False
        if not export_successful:
            print("  -> ERROR: Export did not complete.")
            return False

        combined_path = output_base
        scatter_path = output_base.parent / f"{output_base.stem}_scatter.png"
        heatmap_path = output_base.parent / f"{output_base.stem}_heatmap.png"

        time.sleep(0.5)

        missing = [p.name for p in (combined_path, scatter_path, heatmap_path) if not p.exists()]
        if missing:
            print(f"  -> ERROR: Missing output files: {', '.join(missing)}")
            return False

        print(f"  -> SUCCESS: Created {combined_path.name}, {scatter_path.name}, {heatmap_path.name}")
        return True

    except Exception as exc:
        print(f"  Exception occurred: {exc}")
        process.kill()
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="cuda", choices=["cuda", "cpu"], help="Solver backend")
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--platform", default="", help="Qt platform plugin (omit for default)")
    parser.add_argument("--timeout", type=int, default=7200, help="Timeout per case (seconds)")
    parser.add_argument("--max-iterations", type=int, default=0, help="Override max_iterations if > 0")
    parser.add_argument("--save-every", type=int, default=0, help="Override save_every if > 0")
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

    scenarios = [
        {"name": "pbc-random", "boundary": "PBC", "init": "random", "grid": (160, 160, 0.1, 0.1), "physics": (8.0, 0.2, 0.2)},
        {"name": "pbc-sinusoidal", "boundary": "PBC", "init": "sinusoids", "grid": (160, 160, 0.1, 0.1), "physics": (8.0, 0.2, 0.2)},
        {"name": "w2-random", "boundary": "W2", "init": "random", "grid": (160, 160, 0.1, 0.1), "physics": (8.0, 0.2, 0.2)},
        {"name": "w2-sinusoidal", "boundary": "W2", "init": "sinusoids", "grid": (160, 160, 0.1, 0.1), "physics": (8.0, 0.2, 0.2)},
        {"name": "w4-random", "boundary": "W4", "init": "random", "grid": (160, 160, 0.1, 0.1), "physics": (8.0, 0.2, 0.2)},
        {"name": "w4-sinusoidal", "boundary": "W4", "init": "sinusoids", "grid": (160, 160, 0.1, 0.1), "physics": (8.0, 0.2, 0.2)},
        {"name": "rep_pbc", "boundary": "PBC", "init": "sinusoids", "grid": (80, 80, 0.2, 0.2), "physics": (2.9, 0.4, 0.2)},
        {"name": "rep_w2", "boundary": "W2", "init": "sinusoids", "grid": (160, 160, 0.1, 0.1), "physics": (2.9, 0.4, 0.2)},
        {"name": "rep_w4", "boundary": "W4", "init": "sinusoids", "grid": (160, 160, 0.1, 0.1), "physics": (2.9, 0.4, 0.2)},
    ]

    lines = load_config_lines(base_cfg)

    for sc in scenarios:
        print(f"\n{'='*50}")
        print(f">>> Generating Report Image: {sc['name']}.png")
        print(f"{'='*50}")

        cfg_path = workdir / f"{sc['name']}.cfg"
        nx, ny, dx, dy = sc["grid"]
        temp, rho1, rho2 = sc["physics"]

        overrides = {
            "grid": {
                "boundary_mode": sc["boundary"],
                "init_mode": sc["init"],
                "nx": str(nx),
                "ny": str(ny),
                "dx": str(dx),
                "dy": str(dy),
            },
            "physics": {
                "temperature": str(temp),
                "rho1": str(rho1),
                "rho2": str(rho2),
            },
        }

        if args.max_iterations > 0:
            overrides.setdefault("solver", {})["max_iterations"] = str(args.max_iterations)
        if args.save_every > 0:
            overrides.setdefault("output", {})["save_every"] = str(args.save_every)

        write_config(cfg_path, lines, overrides)

        output_base = out_dir / f"{sc['name']}.png"

        success = run_headless_render(
            gui_exe,
            db_path,
            cfg_path,
            args.backend,
            output_base,
            args.width,
            args.height,
            args.platform,
            args.timeout,
        )
        
        if not success:
            print(f"!!! Failed to generate visuals for {sc['name']}")

    print("\nBatch generation complete! Your images are ready in 'docs/report/SALR3YUCUDA/src/'")

if __name__ == "__main__":
    main()