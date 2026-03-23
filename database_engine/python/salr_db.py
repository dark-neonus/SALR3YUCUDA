"""
salr_db - Python bindings for SALR database engine

This module provides a high-level interface for:
- Listing and filtering simulation runs
- Loading HDF5 snapshots
- Managing run metadata (nicknames, deletion)
- Extracting partial data via hyperslabs

Usage:
    from salr_db import Database, Run

    db = Database("./data")

    # List runs
    for run in db.list_runs(temp_min=2.0, temp_max=3.0):
        print(f"{run.run_id}: {run.nickname or 'unnamed'}")

    # Load snapshot
    run = db.open_run("session_20260323_143052_a1b2c3d4")
    rho1, rho2, meta = run.load_snapshot(iteration=1000)

    # Extract slice
    cross_section = run.extract_slice("rho1", y_start=80, y_count=1)
"""

import sqlite3
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("Warning: h5py not installed. HDF5 operations will not work.")


@dataclass
class RunSummary:
    """Summary information for a simulation run."""
    run_id: str
    nickname: Optional[str]
    created_at: str
    temperature: float
    rho1_bulk: float
    rho2_bulk: float
    nx: int
    ny: int
    boundary_mode: str
    snapshot_count: int
    final_error: Optional[float]
    converged: bool


@dataclass
class SnapshotMeta:
    """Metadata from an HDF5 snapshot."""
    iteration: int
    current_error: float
    delta_error: float
    temperature: float
    rho1_bulk: float
    rho2_bulk: float
    nx: int
    ny: int
    Lx: float
    Ly: float
    dx: float
    dy: float
    boundary_mode: str
    xi1: float
    xi2: float
    cutoff_radius: float
    created_at: str


class Run:
    """Handle to an open simulation run."""

    def __init__(self, run_path: Path, run_id: str):
        """
        Initialize run handle.

        Args:
            run_path: Path to the run directory
            run_id: Run identifier string
        """
        self.path = run_path
        self.run_id = run_id

    def list_snapshots(self) -> List[int]:
        """
        List all snapshot iterations in this run.

        Returns:
            Sorted list of iteration numbers
        """
        snapshots = sorted(self.path.glob("snapshot_*.h5"))
        iterations = []
        for s in snapshots:
            try:
                iter_num = int(s.stem.split("_")[1])
                iterations.append(iter_num)
            except (IndexError, ValueError):
                continue
        return sorted(iterations)

    def load_snapshot(
        self,
        iteration: int = -1
    ) -> Tuple[np.ndarray, np.ndarray, SnapshotMeta]:
        """
        Load density arrays and metadata from a snapshot.

        Args:
            iteration: Iteration number to load (-1 for latest)

        Returns:
            Tuple of (rho1, rho2, metadata)

        Raises:
            FileNotFoundError: If no snapshots found
            ImportError: If h5py not installed
        """
        if not HAS_H5PY:
            raise ImportError("h5py is required to load HDF5 snapshots")

        if iteration < 0:
            iterations = self.list_snapshots()
            if not iterations:
                raise FileNotFoundError(f"No snapshots found in {self.path}")
            iteration = max(iterations)

        filepath = self.path / f"snapshot_{iteration:06d}.h5"

        if not filepath.exists():
            raise FileNotFoundError(f"Snapshot not found: {filepath}")

        with h5py.File(filepath, "r") as f:
            rho1 = f["rho1"][:]
            rho2 = f["rho2"][:]

            meta = SnapshotMeta(
                iteration=int(f.attrs.get("iteration", 0)),
                current_error=float(f.attrs.get("current_error", 0.0)),
                delta_error=float(f.attrs.get("delta_error", -1.0)),
                temperature=float(f.attrs.get("temperature", 0.0)),
                rho1_bulk=float(f.attrs.get("rho1_bulk", 0.0)),
                rho2_bulk=float(f.attrs.get("rho2_bulk", 0.0)),
                nx=int(f.attrs.get("nx", 0)),
                ny=int(f.attrs.get("ny", 0)),
                Lx=float(f.attrs.get("Lx", 0.0)),
                Ly=float(f.attrs.get("Ly", 0.0)),
                dx=float(f.attrs.get("dx", 0.0)),
                dy=float(f.attrs.get("dy", 0.0)),
                boundary_mode=f.attrs.get("boundary_mode", b"PBC").decode()
                    if isinstance(f.attrs.get("boundary_mode", "PBC"), bytes)
                    else str(f.attrs.get("boundary_mode", "PBC")),
                xi1=float(f.attrs.get("xi1", 0.0)),
                xi2=float(f.attrs.get("xi2", 0.0)),
                cutoff_radius=float(f.attrs.get("cutoff_radius", 0.0)),
                created_at=f.attrs.get("created_at", b"").decode()
                    if isinstance(f.attrs.get("created_at", ""), bytes)
                    else str(f.attrs.get("created_at", ""))
            )

        return rho1, rho2, meta

    def load_metadata(self, iteration: int = -1) -> SnapshotMeta:
        """
        Load only metadata from a snapshot (fast, no data loading).

        Args:
            iteration: Iteration number (-1 for latest)

        Returns:
            Snapshot metadata
        """
        if not HAS_H5PY:
            raise ImportError("h5py is required to read HDF5 metadata")

        if iteration < 0:
            iterations = self.list_snapshots()
            if not iterations:
                raise FileNotFoundError(f"No snapshots found in {self.path}")
            iteration = max(iterations)

        filepath = self.path / f"snapshot_{iteration:06d}.h5"

        with h5py.File(filepath, "r") as f:
            meta = SnapshotMeta(
                iteration=int(f.attrs.get("iteration", 0)),
                current_error=float(f.attrs.get("current_error", 0.0)),
                delta_error=float(f.attrs.get("delta_error", -1.0)),
                temperature=float(f.attrs.get("temperature", 0.0)),
                rho1_bulk=float(f.attrs.get("rho1_bulk", 0.0)),
                rho2_bulk=float(f.attrs.get("rho2_bulk", 0.0)),
                nx=int(f.attrs.get("nx", 0)),
                ny=int(f.attrs.get("ny", 0)),
                Lx=float(f.attrs.get("Lx", 0.0)),
                Ly=float(f.attrs.get("Ly", 0.0)),
                dx=float(f.attrs.get("dx", 0.0)),
                dy=float(f.attrs.get("dy", 0.0)),
                boundary_mode=f.attrs.get("boundary_mode", b"PBC").decode()
                    if isinstance(f.attrs.get("boundary_mode", "PBC"), bytes)
                    else str(f.attrs.get("boundary_mode", "PBC")),
                xi1=float(f.attrs.get("xi1", 0.0)),
                xi2=float(f.attrs.get("xi2", 0.0)),
                cutoff_radius=float(f.attrs.get("cutoff_radius", 0.0)),
                created_at=f.attrs.get("created_at", b"").decode()
                    if isinstance(f.attrs.get("created_at", ""), bytes)
                    else str(f.attrs.get("created_at", ""))
            )

        return meta

    def extract_slice(
        self,
        dataset: str,
        x_start: int = 0,
        x_count: Optional[int] = None,
        y_start: int = 0,
        y_count: Optional[int] = None,
        iteration: int = -1
    ) -> np.ndarray:
        """
        Extract a rectangular slice from a dataset.

        Args:
            dataset: Dataset name ("rho1" or "rho2")
            x_start: Starting X index
            x_count: Number of X elements (None for all)
            y_start: Starting Y index
            y_count: Number of Y elements (None for all)
            iteration: Snapshot iteration (-1 for latest)

        Returns:
            NumPy array containing the slice
        """
        if not HAS_H5PY:
            raise ImportError("h5py is required for hyperslab extraction")

        if iteration < 0:
            iteration = max(self.list_snapshots())

        filepath = self.path / f"snapshot_{iteration:06d}.h5"

        with h5py.File(filepath, "r") as f:
            dset = f[dataset]
            ny, nx = dset.shape

            if x_count is None:
                x_count = nx - x_start
            if y_count is None:
                y_count = ny - y_start

            return dset[y_start:y_start+y_count, x_start:x_start+x_count]

    def __repr__(self) -> str:
        return f"Run(id='{self.run_id}', path='{self.path}')"


class Database:
    """Main database interface for SALR simulations."""

    def __init__(self, data_root: Union[str, Path]):
        """
        Initialize database connection.

        Args:
            data_root: Path to the data directory
        """
        self.data_root = Path(data_root)
        self.db_path = self.data_root / "runs.db"

        # Create data directory if needed
        if not self.data_root.exists():
            self.data_root.mkdir(parents=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def list_runs(
        self,
        temp_min: Optional[float] = None,
        temp_max: Optional[float] = None,
        rho1_min: Optional[float] = None,
        rho1_max: Optional[float] = None,
        nickname_contains: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[RunSummary]:
        """
        List runs matching filter criteria.

        Args:
            temp_min: Minimum temperature
            temp_max: Maximum temperature
            rho1_min: Minimum rho1 bulk density
            rho1_max: Maximum rho1 bulk density
            nickname_contains: Filter by nickname substring
            limit: Maximum number of results

        Returns:
            List of RunSummary objects
        """
        conn = self._get_connection()

        query = """
            SELECT run_id, nickname, created_at, temperature, rho1_bulk,
                   rho2_bulk, nx, ny, boundary_mode, snapshot_count,
                   final_error, converged
            FROM runs WHERE 1=1
        """
        params = []

        if temp_min is not None:
            query += " AND temperature >= ?"
            params.append(temp_min)
        if temp_max is not None:
            query += " AND temperature <= ?"
            params.append(temp_max)
        if rho1_min is not None:
            query += " AND rho1_bulk >= ?"
            params.append(rho1_min)
        if rho1_max is not None:
            query += " AND rho1_bulk <= ?"
            params.append(rho1_max)
        if nickname_contains:
            query += " AND nickname LIKE ?"
            params.append(f"%{nickname_contains}%")

        query += " ORDER BY created_at DESC"

        if limit:
            query += f" LIMIT {limit}"

        cursor = conn.execute(query, params)
        results = []

        for row in cursor:
            results.append(RunSummary(
                run_id=row["run_id"],
                nickname=row["nickname"],
                created_at=row["created_at"],
                temperature=row["temperature"],
                rho1_bulk=row["rho1_bulk"],
                rho2_bulk=row["rho2_bulk"],
                nx=row["nx"],
                ny=row["ny"],
                boundary_mode=row["boundary_mode"],
                snapshot_count=row["snapshot_count"] or 0,
                final_error=row["final_error"],
                converged=bool(row["converged"])
            ))

        conn.close()
        return results

    def open_run(self, run_id: str) -> Run:
        """
        Open a run by ID.

        Args:
            run_id: Run identifier

        Returns:
            Run handle

        Raises:
            FileNotFoundError: If run not found
        """
        run_path = self.data_root / run_id
        if not run_path.exists():
            raise FileNotFoundError(f"Run not found: {run_id}")
        return Run(run_path, run_id)

    def set_nickname(self, run_id: str, nickname: Optional[str]) -> None:
        """
        Set or clear the nickname for a run.

        Args:
            run_id: Run identifier
            nickname: New nickname (or None to clear)
        """
        conn = self._get_connection()
        conn.execute(
            "UPDATE runs SET nickname = ? WHERE run_id = ?",
            (nickname, run_id)
        )
        conn.commit()
        conn.close()

    def delete_run(self, run_id: str, force: bool = False) -> None:
        """
        Delete a run atomically (directory + registry).

        Args:
            run_id: Run identifier
            force: If True, skip confirmation

        Raises:
            ValueError: If run not found
        """
        run_path = self.data_root / run_id

        conn = self._get_connection()
        try:
            conn.execute("BEGIN EXCLUSIVE")

            # Delete directory
            if run_path.exists():
                shutil.rmtree(run_path)

            # Delete registry entry
            conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def get_run_info(self, run_id: str) -> Optional[RunSummary]:
        """
        Get detailed information about a run.

        Args:
            run_id: Run identifier

        Returns:
            RunSummary or None if not found
        """
        conn = self._get_connection()

        cursor = conn.execute("""
            SELECT run_id, nickname, created_at, temperature, rho1_bulk,
                   rho2_bulk, nx, ny, boundary_mode, snapshot_count,
                   final_error, converged
            FROM runs WHERE run_id = ?
        """, (run_id,))

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return RunSummary(
            run_id=row["run_id"],
            nickname=row["nickname"],
            created_at=row["created_at"],
            temperature=row["temperature"],
            rho1_bulk=row["rho1_bulk"],
            rho2_bulk=row["rho2_bulk"],
            nx=row["nx"],
            ny=row["ny"],
            boundary_mode=row["boundary_mode"],
            snapshot_count=row["snapshot_count"] or 0,
            final_error=row["final_error"],
            converged=bool(row["converged"])
        )

    def __repr__(self) -> str:
        return f"Database(root='{self.data_root}')"
