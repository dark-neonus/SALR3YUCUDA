# SYSTEM INSTRUCTION: I/O & Data Architecture for SALR3YUCUDA

## 1. Core Objective
Implement a scalable, high-performance, and GUI-friendly data serialization architecture for a C/CUDA-based Density Functional Theory (DFT) solver. The architecture must support easy state resumption, prevent data overwriting, provide O(1) fast-indexing for GUI filtering, and include robust database management features (deletion, annotation, and extraction) while complying with standard version control limits (e.g., GitHub's 100MB file limit).

## 2. Architectural Philosophy
* **Atomic Snapshots:** Every saved `.h5` state must be 100% self-contained, including physical density arrays and thermodynamic parameters as HDF5 Attributes.
* **Centralized Discovery (The Registry):** All runs must be logged in a single Master Registry (`runs_registry.csv`).
* **Data Lifecycle Management:** The system must support safe deletion (preventing orphaned files) and user annotations (nicknames) without breaking the linkage between the registry and the binary files.
* **Targeted Extraction:** Reading data must be surgical. The system should allow extracting specific slices, arrays, or metadata without loading entire simulation histories into memory.

## 3. Directory Structure & Fast Indexing Specification
Enforce the following hierarchical structure:

```text
/project_root
└── /data
    ├── runs_registry.csv                       <-- MASTER INDEX (Appended by C solver, edited by GUI)
    ├── /session_{YYYYMMDD_HHMMSS}_{ShortHash}      <-- Unique ID prevents overlap
    │   ├── snapshot_00000.h5
    │   └── snapshot_00010.h5
    └── /session_{YYYYMMDD_HHMMSS}_{ShortHash}
        └── ...
```

### The Master Registry (`runs_registry.csv`)
This acts as the high-speed search index and annotation layer.
* **Schema:** `Run_ID, Nickname, Timestamp, Temperature, Rho1_Bulk, Rho2_Bulk, Nx, Ny, Boundary_Mode`
* **Nicknaming:** The `Nickname` field defaults to `None` or an empty string upon C solver initialization. The user (via GUI or CLI) can modify this field to assign human-readable labels (e.g., "Stripes_PBC_Run1"). 
* **Rule:** The `Run_ID` acts as the immutable primary key. The `Nickname` is mutable metadata.

## 4. Database Management & Operations Protocol
The GUI and helper scripts must implement the following operations strictly as defined below:

### A. Deletion Protocol (Safe Erase)
To delete a simulation record, a two-step atomic operation must occur to prevent orphaned data:
1.  **File System:** Recursively delete the target directory `/data/[Run_ID]`.
2.  **Registry:** Remove the corresponding row containing `[Run_ID]` from `runs_registry.csv`.
*Constraint:* Never remove the registry row without deleting the folder, and vice versa.

### B. Record Annotation (Nicknaming)
To rename or tag a run:
1.  Locate the row by `Run_ID` in `runs_registry.csv`.
2.  Update the `Nickname` column.
*Constraint:* Do NOT rename the actual directory `/data/[Run_ID]`. The directory name must remain the original timestamped/hashed ID to maintain strict linkage and prevent path resolution errors.

### C. Data Extraction Protocol
To extract data for external analysis or plotting without loading the entire dataset:
1.  **Metadata Extraction:** Read attributes directly from the root of the `.h5` file using `H5Aread` (C) or `.attrs` (h5py/Python).
2.  **Partial Array Extraction (Slicing):** Use HDF5 hyperslabs to read specific 1D projections (e.g., a cross-section of the density profile) or specific regions of the $N_x \times N_y$ grid.
3.  **Export Format:** Provide an export utility that converts a target `.h5` snapshot into standard formats (CSV or raw text) for users who need to import the data into MATLAB, Gnuplot, or Excel.

## 5. HDF5 File Specification (.h5)
The snapshot must act as the single source of truth for the physical state.

* **Datasets (Binary Arrays):**
  * `/rho1` : (Double) 2D Array of size Nx by Ny.
  * `/rho2` : (Double) 2D Array of size Nx by Ny.
  * `/preview_rho` : (Optional, Unsigned Char) Downsampled 2D array for instant GUI rendering.
* **Attributes (Metadata):**
  * `iteration` : (Int) Current Picard iteration step.
  * `current_error` : (Double) Root-mean-square error.
  * `temperature` : (Double) System temperature (T).
  * `rho1_bulk`, `rho2_bulk` : (Double) Bulk densities.
  * `Nx`, `Ny` : (Int) Grid dimensions.

## 6. System Constraints & Compliance
* **GitHub Limits:** Output strictly one file per iteration snapshot (e.g., `snapshot_00010.h5`) to keep file sizes well below the 100MB limit.
* **Compression:** Enable HDF5 GZIP/deflate filters (level 4 to 6) and data shuffling.
* **State Resumption Execution:** The C solver accepts a specific `.h5` file path via CLI. It bypasses noise generation, loads `rho1` and `rho2`, validates grid size attributes against the current config, and resumes the Picard loop.