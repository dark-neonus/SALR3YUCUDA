# SALR Database Engine

A high-performance storage engine for SALR DFT simulation data with HDF5 snapshots and SQLite registry.

## Overview

The database engine provides:
- **HDF5 Snapshots**: Compressed binary storage with embedded metadata
- **SQLite Registry**: Fast indexing and filtering of simulation runs
- **State Resumption**: Continue simulations from checkpoints
- **Python Bindings**: Easy data access for analysis and visualization

---

## Terminology

This documentation uses precise terminology to avoid confusion:

| Term | Definition | Example |
|------|------------|---------|
| **Snapshot** | A single HDF5 file containing density fields and metadata at one iteration | `snapshot_001000.h5` |
| **Session** | A directory containing all snapshots from one simulation run | `session_20260323_143052_a1b2c3d4/` |
| **Registry** | SQLite database indexing all sessions | `runs.db` |

### Visual Structure

```
database/                           # Data root directory
├── runs.db                         # Registry (SQLite)
└── session_20260323_143052_a1b2c3d4/   # Session directory
    ├── snapshot_000000.h5          # Initial snapshot
    ├── snapshot_001000.h5          # Snapshot at iteration 1000
    ├── snapshot_002000.h5          # Snapshot at iteration 2000
    └── snapshot_003500.h5          # Final snapshot (converged)
```

---

## Quick Start

### C API

```c
#include "database_engine/include/db_engine.h"

int main() {
    // Initialize database
    db_init("./database");

    // Create new session
    DbRun *run;
    db_run_create(&cfg, &run);
    printf("Session ID: %s\n", db_run_get_id(run));

    // Save snapshots during iteration
    db_snapshot_save(run, rho1, rho2, iter, error, delta_error, &cfg);

    // Close session
    db_run_close(run);
    db_close();
}
```

### Python API

```python
from database_engine.python.salr_db import Database

# Open database
db = Database("./database")

# List sessions
for run in db.list_runs(temp_min=2.5, temp_max=3.5):
    print(f"{run.run_id}: T={run.temperature}")

# Load snapshot
run = db.open_run("session_20260323_143052_a1b2c3d4")
rho1, rho2, meta = run.load_snapshot(iteration=1000)

# Extract a slice (efficient partial read)
cross_section = run.extract_slice("rho1", y_start=64, y_count=1)
```

---

## C API Reference

### Lifecycle Functions

#### `db_init()`
Initialize the database engine. Must be called before any other functions.

```c
DbError db_init(const char *data_root);
```
- **data_root**: Path to the data directory (e.g., `"./database"`)
- **Returns**: `DB_OK` on success
- Creates the directory and registry if they don't exist

#### `db_close()`
Close the database and release all resources.

```c
void db_close(void);
```

#### `db_is_initialized()`
Check if the database is initialized.

```c
int db_is_initialized(void);
```
- **Returns**: 1 if initialized, 0 otherwise

---

### Session Management

#### `db_run_create()`
Create a new simulation session.

```c
DbError db_run_create(const struct SimConfig *cfg, DbRun **run_out);
```
- **cfg**: Simulation configuration
- **run_out**: Output handle (caller must call `db_run_close()`)
- Session ID is auto-generated: `session_{YYYYMMDD_HHMMSS}_{hash8}`

#### `db_run_open()`
Open an existing session by ID.

```c
DbError db_run_open(const char *run_id, DbRun **run_out);
```
- **run_id**: Session identifier
- **Returns**: `DB_ERR_NOT_FOUND` if session doesn't exist

#### `db_run_close()`
Close a session handle and free resources.

```c
void db_run_close(DbRun *run);
```

#### `db_run_get_id()` / `db_run_get_path()`
Get session ID or directory path.

```c
const char *db_run_get_id(const DbRun *run);
const char *db_run_get_path(const DbRun *run);
```
- Returned strings are valid until `db_run_close()`

---

### Snapshot Operations

#### `db_snapshot_save()`
Save density fields to an HDF5 snapshot.

```c
DbError db_snapshot_save(DbRun *run,
                         const double *rho1,
                         const double *rho2,
                         int iteration,
                         double error,
                         double delta_error,
                         const struct SimConfig *cfg);
```
- **rho1, rho2**: Density arrays (nx * ny, row-major order)
- **iteration**: Current Picard iteration number
- **error**: Current L2 convergence error
- **delta_error**: Error change from previous iteration (`|err_prev - err|`), use -1.0 for initial snapshot
- Creates file: `snapshot_{iteration:06d}.h5`

#### `db_snapshot_load()`
Load density fields from a snapshot.

```c
DbError db_snapshot_load(const DbRun *run,
                         int snapshot_iter,
                         double *rho1,
                         double *rho2,
                         SnapshotMeta *meta_out);
```
- **snapshot_iter**: Iteration to load (-1 for latest snapshot)
- **rho1, rho2**: Pre-allocated output arrays
- **meta_out**: Optional metadata output (can be NULL)

#### `db_snapshot_list()`
List all snapshot iterations in a session.

```c
DbError db_snapshot_list(const DbRun *run, int **iters_out, int *count_out);
```
- **iters_out**: Output array of iteration numbers (caller frees with `free()`)
- **count_out**: Number of snapshots

#### `db_snapshot_extract_slice()`
Read partial data without loading full arrays (hyperslab extraction).

```c
DbError db_snapshot_extract_slice(const DbRun *run,
                                  int snapshot_iter,
                                  const char *dataset,
                                  int x_start, int x_count,
                                  int y_start, int y_count,
                                  double *data_out);
```
- **dataset**: `"rho1"` or `"rho2"`
- **data_out**: Pre-allocated array (x_count * y_count)

---

### Registry Operations

#### `db_registry_list()`
Query sessions with optional filters.

```c
DbError db_registry_list(double temp_min, double temp_max,
                         double rho1_min, double rho1_max,
                         RunSummary **runs_out, int *count_out);
```
- Use 0 to disable a filter (e.g., `temp_min=0, temp_max=0` returns all temperatures)
- **runs_out**: Array of `RunSummary` (caller frees)

#### `db_registry_set_nickname()`
Set or clear a session nickname.

```c
DbError db_registry_set_nickname(const char *run_id, const char *nickname);
```
- **nickname**: New nickname, or NULL to clear

#### `db_registry_delete_run()`
Atomically delete a session (both directory and registry entry).

```c
DbError db_registry_delete_run(const char *run_id);
```
- **Warning**: This is destructive and cannot be undone

#### `db_registry_update_status()`
Update session completion status (called when solver finishes).

```c
DbError db_registry_update_status(const char *run_id,
                                  int snapshot_count,
                                  double final_error,
                                  int converged);
```

---

### State Resumption

#### `db_resume_state()`
Resume solver from a checkpoint with grid validation.

```c
DbError db_resume_state(const char *run_id,
                        int snapshot_iter,
                        const struct SimConfig *cfg,
                        double *rho1,
                        double *rho2,
                        int *start_iter_out);
```
- **snapshot_iter**: Iteration to resume from (-1 for latest)
- **cfg**: Current config (validates grid dimensions match)
- **start_iter_out**: Iteration number to continue from
- **Returns**: `DB_ERR_MISMATCH` if grid dimensions differ

---

### Export Utilities

#### `db_export_ascii()`
Export snapshot to ASCII format compatible with visualization scripts.

```c
DbError db_export_ascii(const DbRun *run,
                        int snapshot_iter,
                        const char *output_path);
```
- Format: `x y rho1 rho2` columns, blank lines between rows

---

## Python API Reference

### Database Class

```python
class Database:
    def __init__(self, data_root: str): ...
    def list_runs(self, temp_min=None, temp_max=None,
                  rho1_min=None, rho1_max=None,
                  nickname_contains=None, limit=None) -> List[RunSummary]: ...
    def open_run(self, run_id: str) -> Run: ...
    def set_nickname(self, run_id: str, nickname: Optional[str]): ...
    def delete_run(self, run_id: str, force=False): ...
    def get_run_info(self, run_id: str) -> Optional[RunSummary]: ...
```

### Run Class

```python
class Run:
    def list_snapshots(self) -> List[int]: ...
    def load_snapshot(self, iteration=-1) -> Tuple[np.ndarray, np.ndarray, SnapshotMeta]: ...
    def load_metadata(self, iteration=-1) -> SnapshotMeta: ...
    def extract_slice(self, dataset: str, x_start=0, x_count=None,
                      y_start=0, y_count=None, iteration=-1) -> np.ndarray: ...
```

### Data Classes

```python
@dataclass
class RunSummary:
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
    iteration: int
    current_error: float
    delta_error: float       # Error change from previous iteration
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
```

---

## HDF5 Snapshot Format

Each snapshot file contains:

### Datasets

| Dataset | Type | Shape | Compression |
|---------|------|-------|-------------|
| `/rho1` | float64 | (ny, nx) | GZIP level 5, chunked 64x64 |
| `/rho2` | float64 | (ny, nx) | GZIP level 5, chunked 64x64 |

### Root Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `iteration` | int32 | Picard iteration number |
| `current_error` | float64 | L2 convergence error |
| `delta_error` | float64 | Error change from previous iteration |
| `temperature` | float64 | Reduced temperature T |
| `rho1_bulk` | float64 | Bulk density species 1 |
| `rho2_bulk` | float64 | Bulk density species 2 |
| `nx`, `ny` | int32 | Grid dimensions |
| `Lx`, `Ly` | float64 | Physical domain size |
| `dx`, `dy` | float64 | Grid spacing |
| `boundary_mode` | string | "PBC", "W2", or "W4" |
| `xi1`, `xi2` | float64 | Picard mixing coefficients |
| `cutoff_radius` | float64 | Potential cutoff |
| `created_at` | string | ISO8601 timestamp |

### Example: Reading with h5py

```python
import h5py

with h5py.File("snapshot_001000.h5", "r") as f:
    rho1 = f["rho1"][:]
    rho2 = f["rho2"][:]
    print(f"Iteration: {f.attrs['iteration']}")
    print(f"Error: {f.attrs['current_error']}")
    print(f"Delta Error: {f.attrs['delta_error']}")
```

---

## SQLite Registry Schema

```sql
CREATE TABLE runs (
    run_id         TEXT PRIMARY KEY,        -- e.g., "session_20260323_143052_a1b2c3d4"
    nickname       TEXT DEFAULT NULL,       -- User-defined name
    created_at     TEXT NOT NULL,           -- ISO8601 timestamp
    temperature    REAL NOT NULL,           -- Reduced temperature
    rho1_bulk      REAL NOT NULL,           -- Bulk density species 1
    rho2_bulk      REAL NOT NULL,           -- Bulk density species 2
    nx             INTEGER NOT NULL,        -- Grid size X
    ny             INTEGER NOT NULL,        -- Grid size Y
    boundary_mode  TEXT NOT NULL,           -- "PBC", "W2", "W4"
    config_hash    TEXT NOT NULL,           -- Config fingerprint
    snapshot_count INTEGER DEFAULT 0,       -- Number of snapshots
    final_error    REAL DEFAULT NULL,       -- Final convergence error
    converged      INTEGER DEFAULT 0        -- 1 if converged
);

CREATE INDEX idx_runs_temp ON runs(temperature);
CREATE INDEX idx_runs_rho ON runs(rho1_bulk, rho2_bulk);
```

---

## Error Codes

### DbError (db_engine.h)

| Code | Value | Description |
|------|-------|-------------|
| `DB_OK` | 0 | Success |
| `DB_ERR_INIT` | -1 | Initialization error |
| `DB_ERR_IO` | -2 | File I/O error |
| `DB_ERR_HDF5` | -3 | HDF5 library error |
| `DB_ERR_SQLITE` | -4 | SQLite error |
| `DB_ERR_NOT_FOUND` | -5 | Session or snapshot not found |
| `DB_ERR_INVALID` | -6 | Invalid parameter or state |
| `DB_ERR_MISMATCH` | -7 | Grid size mismatch on resume |
| `DB_ERR_ALLOC` | -8 | Memory allocation failure |

### HDF5Error (hdf5_io.h)

| Code | Value | Description |
|------|-------|-------------|
| `HDF5_OK` | 0 | Success |
| `HDF5_ERR_FILE` | -1 | File open/create error |
| `HDF5_ERR_DATASET` | -2 | Dataset operations error |
| `HDF5_ERR_ATTR` | -3 | Attribute operations error |
| `HDF5_ERR_SPACE` | -4 | Dataspace error |
| `HDF5_ERR_MISMATCH` | -5 | Grid size mismatch |
| `HDF5_ERR_ALLOC` | -6 | Memory allocation error |

### RegistryError (registry.h)

| Code | Value | Description |
|------|-------|-------------|
| `REG_OK` | 0 | Success |
| `REG_ERR_OPEN` | -1 | Failed to open database |
| `REG_ERR_EXEC` | -2 | SQL execution error |
| `REG_ERR_PREPARE` | -3 | SQL prepare error |
| `REG_ERR_NOT_FOUND` | -4 | Session not found |
| `REG_ERR_ALLOC` | -5 | Memory allocation error |
| `REG_ERR_BUSY` | -6 | Database is busy/locked |

---

## Building & Dependencies

### System Dependencies

```bash
# Ubuntu/Debian
sudo apt install libhdf5-dev libsqlite3-dev

# macOS
brew install hdf5 sqlite

# Build with CMake
mkdir build && cd build
cmake .. -DUSE_DB_ENGINE=ON
make
```

### Python Dependencies

```bash
pip install h5py numpy
```

---

## Examples

### Creating a Session and Saving Snapshots (C)

```c
#include "database_engine/include/db_engine.h"

void simulation_with_database(SimConfig *cfg) {
    // Initialize
    db_init("./database");

    // Create session
    DbRun *run;
    db_run_create(cfg, &run);
    printf("Session: %s\n", db_run_get_id(run));

    // Initial snapshot
    db_snapshot_save(run, rho1, rho2, 0, 1.0, -1.0, cfg);

    // Iteration loop
    double prev_err = 1.0;
    for (int iter = 1; iter <= cfg->solver.max_iterations; iter++) {
        // ... solver iteration ...
        double err = compute_error();
        double delta_err = fabs(prev_err - err);

        // Save periodic snapshot
        if (iter % cfg->save_every == 0) {
            db_snapshot_save(run, rho1, rho2, iter, err, delta_err, cfg);
        }

        // Check convergence
        if (err < cfg->solver.tolerance) {
            db_snapshot_save(run, rho1, rho2, iter, err, delta_err, cfg);
            db_registry_update_status(db_run_get_id(run),
                                      snapshot_count, err, 1);
            break;
        }
        prev_err = err;
    }

    db_run_close(run);
    db_close();
}
```

### Resuming a Session (C)

```c
DbRun *run;
int start_iter;

// Resume from latest snapshot
DbError err = db_resume_state(run_id, -1, cfg, rho1, rho2, &start_iter);
if (err == DB_ERR_MISMATCH) {
    fprintf(stderr, "Error: Grid dimensions don't match!\n");
    return;
}

db_run_open(run_id, &run);
printf("Resuming from iteration %d\n", start_iter);

// Continue simulation from start_iter...
```

### Analyzing Data (Python)

```python
import numpy as np
import matplotlib.pyplot as plt
from database_engine.python.salr_db import Database

db = Database("./database")

# Find converged sessions at temperature 2.9
runs = db.list_runs(temp_min=2.85, temp_max=2.95)
converged = [r for r in runs if r.converged]

for run_info in converged:
    run = db.open_run(run_info.run_id)
    rho1, rho2, meta = run.load_snapshot()

    print(f"{run_info.run_id}:")
    print(f"  Final error: {meta.current_error:.2e}")
    print(f"  Delta error: {meta.delta_error:.2e}")
    print(f"  Iterations:  {meta.iteration}")

    # Plot density
    plt.figure()
    plt.imshow(rho1, cmap='viridis')
    plt.colorbar(label='rho1')
    plt.title(f"T={meta.temperature}")
    plt.savefig(f"{run_info.run_id}_rho1.png")
```

### CLI Usage

```bash
# List all sessions
python -m database_engine.python.cli list

# Filter by temperature
python -m database_engine.python.cli list --temp-min 2.5 --temp-max 3.0

# Set nickname
python -m database_engine.python.cli nickname session_20260323_143052_a1b2c3d4 "Control Run"

# Delete session
python -m database_engine.python.cli delete session_20260323_143052_a1b2c3d4 --force

# Export to ASCII
python -m database_engine.python.cli export session_20260323_143052_a1b2c3d4 1000 output.dat
```

---

## Internal Architecture

```
database_engine/
├── include/
│   ├── db_engine.h      # Public C API
│   ├── hdf5_io.h        # HDF5 operations (internal)
│   ├── registry.h       # SQLite operations (internal)
│   └── db_utils.h       # Utility functions
├── src/
│   ├── db_engine.c      # Main API implementation
│   ├── hdf5_io.c        # HDF5 read/write with compression
│   ├── registry.c       # SQLite CRUD with WAL mode
│   └── db_utils.c       # Hash, timestamp, path utilities
├── python/
│   ├── __init__.py
│   ├── salr_db.py       # Python wrapper (h5py + sqlite3)
│   └── cli.py           # Command-line interface
└── tests/
    ├── test_hdf5.c      # HDF5 unit tests
    ├── test_registry.c  # SQLite unit tests
    └── test_integration.c  # Full workflow tests
```

---

## License

Part of the SALR3YUCUDA project. See main repository for license information.
