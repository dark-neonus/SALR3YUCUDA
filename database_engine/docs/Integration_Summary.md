# Database Engine Integration Summary

## Overview

The SALR DFT solver now has full database engine integration using HDF5 for efficient binary storage and SQLite for run metadata management. All simulation results are saved to `project_root/database/`.

## Architecture

### Database Location
- **Root directory**: `$PROJECT_ROOT/database/`
- **Registry**: `database/registry.db` (SQLite database)
- **Run directories**: `database/run_YYYYMMDD_HHMMSS_HASH/`
- **Snapshots**: `run_*/snapshot_NNNNNN.h5` (HDF5 format)

### Integration Points

#### 1. C Source Files
- **`src/main.c`**:
  - Initializes database with `db_init()`
  - Creates new runs with `db_run_create()`
  - Resumes from checkpoints with `db_resume_state()`
  - Uses `SALR_DB_PATH` environment variable for database location

- **`src/cpu/solver_cpu.c`**:
  - Function: `solver_run_binary_db()`
  - Saves HDF5 snapshots every `save_every` iterations
  - Calls `db_snapshot_save()` to store density fields

- **`src/cuda/solver_cuda.cu`**:
  - Function: `solver_run_binary_db()`
  - GPU-accelerated solver with HDF5 snapshot support
  - Identical interface to CPU version

#### 2. Build System (CMakeLists.txt)
- **Option**: `ENABLE_DB_ENGINE` (default: ON)
- **Dependencies**: HDF5, SQLite3
- **Executables created**:
  - `salr_dft_db` - CPU version with database
  - `salr_dft_cuda_db` - CUDA version with database
  - `salr_dft` - CPU version without database
  - `salr_dft_cuda` - CUDA version without database

#### 3. Run Scripts

**`run.sh`** (CPU version):
```bash
./run.sh              # Run with database (default)
./run.sh sim          # Same as above
./run.sh sim-nodb     # Run without database
./run.sh resume <id>  # Resume from checkpoint
./run.sh list         # List all runs in database
```

**`run_cuda.sh`** (CUDA version):
```bash
./run_cuda.sh              # Run CUDA with database
./run_cuda.sh sim          # Same as above
./run_cuda.sh resume <id>  # Resume from checkpoint
./run_cuda.sh list         # List all runs
```

Both scripts:
- Create `$PROJECT_ROOT/database/` directory
- Export `SALR_DB_PATH` environment variable
- Use database-enabled executables by default
- Support resume functionality

## Data Flow

### New Simulation Run
```
1. run.sh/run_cuda.sh
   ↓ sets SALR_DB_PATH=$PROJECT_ROOT/database
   ↓ creates database directory
   ↓
2. main.c
   ↓ db_init(SALR_DB_PATH)
   ↓ db_run_create(cfg, &run)
   ↓ generates unique run_id: run_YYYYMMDD_HHMMSS_HASH
   ↓
3. solver_run_binary_db()
   ↓ iteration loop
   ↓ every save_every iterations:
   ↓   db_snapshot_save(run, rho1, rho2, iter, error, cfg)
   ↓
4. Database structure created:
   database/
   ├── registry.db (SQLite)
   └── session_20260323_143052_a1b2c3d4/
       ├── snapshot_000000.h5
       ├── snapshot_001000.h5
       ├── snapshot_002000.h5
       └── ...
```

### Resume from Checkpoint
```
1. run.sh resume session_20260323_143052_a1b2c3d4 [iteration]
   ↓
2. main.c
   ↓ db_resume_state(run_id, iteration, cfg, rho1, rho2, &start_iter)
   ↓ validates grid size matches
   ↓ loads density arrays from HDF5
   ↓ db_run_open(run_id, &run)
   ↓
3. solver_run_binary_db(rho1, rho2, cfg, run, start_iter)
   ↓ continues from start_iter
   ↓ appends new snapshots to same run directory
```

## Snapshot Format (HDF5)

Each `snapshot_NNNNNN.h5` file contains:

### Datasets
- `/density/rho1` - Species 1 density field (double, nx×ny)
- `/density/rho2` - Species 2 density field (double, nx×ny)

### Metadata (attributes)
- `iteration` - Picard iteration number
- `error` - L2 convergence error
- `timestamp` - Unix timestamp
- `nx`, `ny` - Grid dimensions
- `dx`, `dy` - Grid spacing
- `temperature` - System temperature
- `rho1_bulk`, `rho2_bulk` - Bulk densities

### Compression
- Datasets use GZIP compression (level 6)
- Typical compression ratio: 3-5×

## Registry Database (SQLite)

**Table: `runs`**
```sql
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    nickname TEXT,
    created_at TEXT,
    nx INTEGER,
    ny INTEGER,
    temperature REAL,
    rho1_bulk REAL,
    rho2_bulk REAL,
    snapshot_count INTEGER,
    final_error REAL,
    converged INTEGER
);
```

**Query examples**:
```bash
# List all runs
sqlite3 database/registry.db "SELECT run_id, created_at, temperature, converged FROM runs;"

# Find converged runs at T=0.1
sqlite3 database/registry.db "SELECT * FROM runs WHERE temperature=0.1 AND converged=1;"
```

## Command Line Usage

### Basic workflow
```bash
# Build with database support
./run.sh build

# Run simulation (saves to database/)
./run.sh sim

# Output:
#   Database initialized at: /mnt/D/ucu/acs/SALR3YUCUDA/database
#   Created run: session_20260323_143052_a1b2c3d4
#   Run directory: /mnt/D/ucu/acs/SALR3YUCUDA/database/session_20260323_143052_a1b2c3d4
#   ...
#   Converged at iteration 15000
#   HDF5 snapshots saved to: /mnt/D/ucu/acs/SALR3YUCUDA/database/

# List all runs
./run.sh list

# Resume from latest snapshot
./run.sh resume session_20260323_143052_a1b2c3d4

# Resume from specific iteration
./run.sh resume session_20260323_143052_a1b2c3d4 10000
```

### CUDA version
```bash
./run_cuda.sh sim                               # Run on GPU with database
./run_cuda.sh resume session_20260323_143052_a1b2c3d4  # Resume on GPU
```

## API Functions Used

### Initialization
- `db_init(const char *data_root)` - Initialize database engine

### Run Management
- `db_run_create(SimConfig *cfg, DbRun **run)` - Create new run
- `db_run_open(const char *run_id, DbRun **run)` - Open existing run
- `db_run_close(DbRun *run)` - Close run handle

### Snapshots
- `db_snapshot_save(run, rho1, rho2, iter, error, cfg)` - Save snapshot
- `db_snapshot_load(run, iter, rho1, rho2, meta)` - Load snapshot
- `db_snapshot_list(run, &iters, &count)` - List all snapshots

### Registry
- `db_registry_list(...)` - Query runs with filters
- `db_registry_update_status(run_id, count, error, converged)` - Update status
- `db_registry_set_nickname(run_id, nickname)` - Set friendly name

### State Resumption
- `db_resume_state(run_id, iter, cfg, rho1, rho2, &start_iter)` - Resume from checkpoint

## File Structure

```
project_root/
├── database/                          # All simulation data
│   ├── registry.db                   # SQLite run registry
│   ├── session_20260323_143052_a1b2c3d4/
│   │   ├── snapshot_000000.h5
│   │   ├── snapshot_001000.h5
│   │   └── ...
│   └── session_20260323_150230_b2c3d4e5/
│       └── ...
├── database_engine/                   # Database engine source
│   ├── include/
│   │   ├── db_engine.h               # Main API
│   │   ├── hdf5_io.h                 # HDF5 operations
│   │   ├── registry.h                # SQLite registry
│   │   └── db_utils.h                # Utilities
│   └── src/
│       ├── db_engine.c
│       ├── hdf5_io.c
│       ├── registry.c
│       └── db_utils.c
├── src/
│   ├── main.c                        # Integration point (db_init, create/resume)
│   ├── cpu/solver_cpu.c              # solver_run_binary_db()
│   └── cuda/solver_cuda.cu           # solver_run_binary_db()
├── run.sh                            # CPU run script (sets SALR_DB_PATH)
├── run_cuda.sh                       # CUDA run script (sets SALR_DB_PATH)
└── scripts/
    └── verify_db_integration.sh      # Verification script
```

## Environment Variables

- `SALR_DB_PATH`: Database root directory (default: `./database`)
  - Set by run scripts to `$PROJECT_ROOT/database`
  - Can be overridden by user
  - Example: `export SALR_DB_PATH=/data/salr_results`

## Verification

Run the verification script to check integration status:
```bash
./scripts/maintenance/verify_db_integration.sh
```

This checks:
- HDF5 and SQLite3 libraries installed
- Database-enabled executables built
- Source files have integration code
- Database structure exists
- Registry contains runs

## Benefits

1. **Efficient storage**: HDF5 binary format with compression
2. **Fast access**: Direct array slicing without loading full file
3. **Metadata**: All simulation parameters stored with data
4. **Queryable**: SQLite registry for finding runs by parameters
5. **Resumable**: Continue interrupted simulations from any snapshot
6. **Organized**: All data in structured `database/` directory
7. **Portable**: Standard HDF5/SQLite formats, cross-platform

## Changes Made

1. **`src/main.c`**:
   - Added `SALR_DB_PATH` environment variable support
   - Database path now configurable (default: `./database`)

2. **`run.sh`**:
   - Exports `SALR_DB_PATH=$PROJECT_ROOT/database`
   - Ensures database directory creation
   - Uses `salr_dft_db` executable by default

3. **`run_cuda.sh`**:
   - Exports `SALR_DB_PATH=$PROJECT_ROOT/database`
   - Ensures database directory creation
   - Uses `salr_dft_cuda_db` executable by default

4. **`scripts/maintenance/verify_db_integration.sh`**:
   - New verification script
   - Checks all integration points
   - Reports database status

## Migration from Old Output Format

The database system runs **alongside** the old ASCII output:
- ASCII files still saved to `output/data/` for compatibility
- HDF5 snapshots saved to `database/` for efficiency
- Both contain the same data
- Can disable ASCII output by modifying solvers if desired

## Next Steps

1. Build the project with database support:
   ```bash
   ./run.sh build
   ```

2. Run a simulation:
   ```bash
   ./run.sh sim
   ```

3. Check the database:
   ```bash
   ls -la database/
   ./run.sh list
   ```

4. Test resumption:
   ```bash
   # Get run_id from list command
   ./run.sh resume run_YYYYMMDD_HHMMSS_HASH
   ```

## Troubleshooting

**Error: HDF5 not found**
```bash
sudo apt install libhdf5-dev
```

**Error: SQLite3 not found**
```bash
sudo apt install libsqlite3-dev
```

**Database path issues**
- Ensure `SALR_DB_PATH` is exported before running executable
- Run scripts handle this automatically
- For manual execution: `export SALR_DB_PATH=/path/to/database`

**Grid mismatch on resume**
- Cannot resume if grid size (nx, ny) changed
- Start a new run instead
