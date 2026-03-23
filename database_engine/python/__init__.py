"""
salr_db - Python bindings for SALR database engine

This package provides tools for managing SALR DFT simulation data:
- HDF5 snapshot storage and retrieval
- SQLite registry for run indexing
- Command-line interface for data management

Quick Start:
    from salr_db import Database

    db = Database("./data")
    runs = db.list_runs(temp_min=2.0, temp_max=3.0)

    for run in runs:
        print(f"{run.run_id}: T={run.temperature}")

CLI Usage:
    python -m salr_db.cli list
    python -m salr_db.cli info <run_id>
    python -m salr_db.cli export <run_id> output.csv
"""

from .salr_db import (
    Database,
    Run,
    RunSummary,
    SnapshotMeta,
    HAS_H5PY
)

__version__ = "1.0.0"
__all__ = ["Database", "Run", "RunSummary", "SnapshotMeta", "HAS_H5PY"]
