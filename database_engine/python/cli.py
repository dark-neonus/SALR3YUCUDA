#!/usr/bin/env python3
"""
SALR Database CLI - Command-line interface for database management.

Usage:
    python -m salr_db.cli list [--temp-min X] [--temp-max Y] [--limit N]
    python -m salr_db.cli info <run_id>
    python -m salr_db.cli nickname <run_id> <name>
    python -m salr_db.cli delete <run_id> [--force]
    python -m salr_db.cli export <run_id> [--iteration N] <output.csv>
    python -m salr_db.cli snapshots <run_id>
"""

import argparse
import sys
from pathlib import Path

from .salr_db import Database, HAS_H5PY


def cmd_list(args):
    """List all runs matching filter criteria."""
    db = Database(args.data_root)

    try:
        runs = db.list_runs(
            temp_min=args.temp_min,
            temp_max=args.temp_max,
            rho1_min=args.rho1_min,
            rho1_max=args.rho1_max,
            limit=args.limit
        )
    except FileNotFoundError:
        print(f"Database not found at {args.data_root}/runs.db")
        return 1

    if not runs:
        print("No runs found.")
        return 0

    # Header
    print(f"{'Run ID':<40} {'Nickname':<15} {'T':<8} {'rho1':<8} {'Snaps':<6} {'Conv':<5}")
    print("-" * 90)

    for r in runs:
        nickname = r.nickname[:14] if r.nickname else "-"
        conv = "Yes" if r.converged else "No"
        print(f"{r.run_id:<40} {nickname:<15} {r.temperature:<8.3f} "
              f"{r.rho1_bulk:<8.3f} {r.snapshot_count:<6} {conv:<5}")

    print(f"\nTotal: {len(runs)} run(s)")
    return 0


def cmd_info(args):
    """Show detailed information about a run."""
    db = Database(args.data_root)

    info = db.get_run_info(args.run_id)
    if not info:
        print(f"Run not found: {args.run_id}")
        return 1

    print(f"Run ID:        {info.run_id}")
    print(f"Nickname:      {info.nickname or '(none)'}")
    print(f"Created:       {info.created_at}")
    print(f"Temperature:   {info.temperature}")
    print(f"rho1 (bulk):   {info.rho1_bulk}")
    print(f"rho2 (bulk):   {info.rho2_bulk}")
    print(f"Grid size:     {info.nx} x {info.ny}")
    print(f"Boundary:      {info.boundary_mode}")
    print(f"Snapshots:     {info.snapshot_count}")
    print(f"Final error:   {info.final_error if info.final_error else '(not set)'}")
    print(f"Converged:     {'Yes' if info.converged else 'No'}")

    # List snapshots if run directory exists
    try:
        run = db.open_run(args.run_id)
        snapshots = run.list_snapshots()
        if snapshots:
            print(f"\nSnapshots: {', '.join(map(str, snapshots[:10]))}"
                  + (f"... ({len(snapshots)} total)" if len(snapshots) > 10 else ""))
    except FileNotFoundError:
        print("\nWarning: Run directory not found")

    return 0


def cmd_nickname(args):
    """Set or clear a run's nickname."""
    db = Database(args.data_root)

    nickname = args.nickname if args.nickname != "none" else None

    try:
        db.set_nickname(args.run_id, nickname)
        if nickname:
            print(f"Set nickname for {args.run_id}: {nickname}")
        else:
            print(f"Cleared nickname for {args.run_id}")
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


def cmd_delete(args):
    """Delete a run (directory + registry entry)."""
    db = Database(args.data_root)

    # Check if run exists
    info = db.get_run_info(args.run_id)
    if not info:
        print(f"Run not found: {args.run_id}")
        return 1

    # Confirm deletion
    if not args.force:
        print(f"About to delete run: {args.run_id}")
        print(f"  Nickname: {info.nickname or '(none)'}")
        print(f"  Snapshots: {info.snapshot_count}")
        confirm = input("Are you sure? [y/N] ")
        if confirm.lower() != 'y':
            print("Cancelled.")
            return 0

    try:
        db.delete_run(args.run_id)
        print(f"Deleted: {args.run_id}")
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


def cmd_export(args):
    """Export a snapshot to CSV format."""
    if not HAS_H5PY:
        print("Error: h5py is required for export. Install with: pip install h5py")
        return 1

    import numpy as np

    db = Database(args.data_root)

    try:
        run = db.open_run(args.run_id)
    except FileNotFoundError:
        print(f"Run not found: {args.run_id}")
        return 1

    try:
        rho1, rho2, meta = run.load_snapshot(args.iteration)
    except FileNotFoundError as e:
        print(f"Error loading snapshot: {e}")
        return 1

    # Export as CSV
    ny, nx = rho1.shape
    output = Path(args.output)

    with open(output, 'w') as f:
        f.write(f"# SALR Export: {args.run_id}\n")
        f.write(f"# iteration={meta.iteration}, temperature={meta.temperature}\n")
        f.write(f"# rho1_bulk={meta.rho1_bulk}, rho2_bulk={meta.rho2_bulk}\n")
        f.write("x,y,rho1,rho2\n")

        for iy in range(ny):
            for ix in range(nx):
                x = (ix + 0.5) * meta.dx
                y = (iy + 0.5) * meta.dy
                f.write(f"{x},{y},{rho1[iy, ix]},{rho2[iy, ix]}\n")

    print(f"Exported iteration {meta.iteration} to {output}")
    print(f"  Grid: {nx} x {ny}")
    print(f"  Temperature: {meta.temperature}")
    return 0


def cmd_snapshots(args):
    """List all snapshots in a run."""
    db = Database(args.data_root)

    try:
        run = db.open_run(args.run_id)
    except FileNotFoundError:
        print(f"Run not found: {args.run_id}")
        return 1

    snapshots = run.list_snapshots()

    if not snapshots:
        print("No snapshots found.")
        return 0

    print(f"Run: {args.run_id}")
    print(f"Snapshots ({len(snapshots)} total):")

    for i, iter_num in enumerate(snapshots):
        if HAS_H5PY:
            try:
                meta = run.load_metadata(iter_num)
                print(f"  {iter_num:6d}  error={meta.current_error:.3e}")
            except Exception:
                print(f"  {iter_num:6d}")
        else:
            print(f"  {iter_num:6d}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="SALR Database CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-root", "-d",
        default="./data",
        help="Path to data directory (default: ./data)"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # list command
    p_list = subparsers.add_parser("list", help="List runs")
    p_list.add_argument("--temp-min", type=float, help="Minimum temperature")
    p_list.add_argument("--temp-max", type=float, help="Maximum temperature")
    p_list.add_argument("--rho1-min", type=float, help="Minimum rho1")
    p_list.add_argument("--rho1-max", type=float, help="Maximum rho1")
    p_list.add_argument("--limit", "-n", type=int, help="Limit results")
    p_list.set_defaults(func=cmd_list)

    # info command
    p_info = subparsers.add_parser("info", help="Show run details")
    p_info.add_argument("run_id", help="Run ID")
    p_info.set_defaults(func=cmd_info)

    # nickname command
    p_nick = subparsers.add_parser("nickname", help="Set run nickname")
    p_nick.add_argument("run_id", help="Run ID")
    p_nick.add_argument("nickname", help="New nickname (use 'none' to clear)")
    p_nick.set_defaults(func=cmd_nickname)

    # delete command
    p_del = subparsers.add_parser("delete", help="Delete a run")
    p_del.add_argument("run_id", help="Run ID")
    p_del.add_argument("--force", "-f", action="store_true",
                       help="Skip confirmation")
    p_del.set_defaults(func=cmd_delete)

    # export command
    p_exp = subparsers.add_parser("export", help="Export snapshot to CSV")
    p_exp.add_argument("run_id", help="Run ID")
    p_exp.add_argument("output", help="Output CSV file")
    p_exp.add_argument("--iteration", "-i", type=int, default=-1,
                       help="Snapshot iteration (-1 for latest)")
    p_exp.set_defaults(func=cmd_export)

    # snapshots command
    p_snaps = subparsers.add_parser("snapshots", help="List snapshots in a run")
    p_snaps.add_argument("run_id", help="Run ID")
    p_snaps.set_defaults(func=cmd_snapshots)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
