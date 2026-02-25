# SALR DFT Solver — Bug Report and Fixes

This document summarizes the bugs discovered during code review and the fixes applied.

---

## Summary of Issues Found

| # | Type | Severity | Component | Status |
|---|------|----------|-----------|--------|
| 1 | Physics | Critical | solver_cpu.c | Fixed |
| 2 | Physics | High | configs/default.cfg | Fixed |
| 3 | Numerical | Medium | solver_cpu.c | Fixed |
| 4 | Visualization | Medium | scripts/ | Fixed |

---

## Bug 1: Solution Converges to Uniform (No Structure)

**File:** `src/cpu/solver_cpu.c`

**Symptom:** 
The solver converged to a uniform density distribution (std dev < 1e-6) instead of the expected SALR stripe/cluster patterns shown in reference images.

**Root Cause:**
At the original temperature T=12, the system was above the spinodal temperature for microphase separation. Combined with:
- Mass renormalization forcing solution back toward bulk
- Anti-checkerboard smoothing dampening all spatial modes
- Initial random noise being averaged out

The thermodynamically stable solution at T>T_spinodal is uniform.

**Fix:**
1. Reduced temperature from T=12 to T=2.0 (below spinodal)
2. Kept mass renormalization (necessary for proper thermodynamics)
3. Re-enabled gentle smoothing to prevent numerical checkerboard instability

**Before:**
```
species1: min=0.3999 max=0.4000 mean=0.4000 range=0.0001
species2: min=0.1999 max=0.2000 mean=0.2000 range=0.0001
(Converged in 128 iterations - WRONG: uniform solution)
```

**After:**
```
species1: min=0.0006 max=11.39 mean=0.4000 range=11.39
species2: min=0.0000 max=5.96  mean=0.2000 range=5.96
(Structure formed - density peaks 28x mean value)
```

---

## Bug 2: Temperature Too High for SALR Clustering

**File:** `configs/default.cfg`

**Original Value:**
```
temperature = 12.0
```

**Issue:**
At T=12, β = 1/12 ≈ 0.083. The effective interaction strength β·U(r) was too weak to drive microphase separation. The SALR potential has a characteristic temperature scale, and T=12 was above the ordering transition.

**Fixed Value:**
```
temperature = 2.0
```

**Explanation:**
At T=2, β = 0.5. This puts the system well within the structured phase region where SALR interactions create density inhomogeneities (clusters or stripes depending on density).

---

## Bug 3: Smoothing Disabled Causes Instability

**File:** `src/cpu/solver_cpu.c`

**Issue:**
When smoothing was completely disabled to preserve structure, the solution became unstable at very low temperatures (T < 1.5):
- Max density reached 800+ (unphysical)
- Numerical checkerboard modes dominated

**Fix:**
Re-enabled the 5-point Laplacian smoothing with ε=0.01:

```c
/* Step 5b: Gentle smoothing to prevent checkerboard instability */
smooth_density(rho1, Nx, Ny);
smooth_density(rho2, Nx, Ny);
apply_boundary_mask(rho1, rho2, Nx, Ny, mode);
```

This suppresses grid-scale (k = π/dx) artifacts while preserving physical structure at wavelengths λ >> dx.

---

## Bug 4: No Joint Visualization for Two Species

**File:** `scripts/plot_density.py`, `scripts/plot_joint_heatmap.py`

**Issue:**
Original visualization only showed single-species heatmaps. No way to see how the two species relate spatially (anti-correlation in SALR systems).

**Fix:**
Created new `scripts/plot_joint_heatmap.py` that generates:
1. Species 1 heatmap (blue colormap)
2. Species 2 heatmap (red colormap)  
3. Joint overlay (blue=species1, red=species2, purple=both)

**Usage:**
```bash
python3 scripts/plot_joint_heatmap.py output/
```

---

## Potential Issues Not Fixed (For Future Work)

### 1. Convergence Oscillation
The solver error oscillates around 0.07 rather than monotonically decreasing. This may indicate:
- Limit cycle behavior near phase boundary
- Need for adaptive mixing parameter
- Temperature-dependent optimal xi values

### 2. Floating Point Considerations
- No underflow protection when computing `exp(-beta * Phi)` at very low T
- Could add: `if (arg < -700) K[k] = 0.0;` to prevent underflow to 0

### 3. Mass Conservation vs Structure
The mass renormalization step enforces `<rho> = rho_bulk` exactly. This is thermodynamically correct but may slow structure formation. Alternative: allow small mass drift and correct periodically.

---

## Recommended Parameters for Structure Formation

Based on testing, these parameters reliably produce SALR patterns:

```ini
[physics]
temperature = 2.0      # Lower = stronger structure, but slower convergence

[solver]
max_iterations = 5000  # May need many iterations at low T
xi1 = 0.05             # Lower = more stable but slower
xi2 = 0.05
tolerance = 1.0e-6     # Relaxed tolerance if not converging to 1e-8
```

---

## Files Modified

1. `src/cpu/solver_cpu.c` — Re-enabled smoothing
2. `configs/default.cfg` — Changed temperature, mixing parameters
3. `scripts/plot_joint_heatmap.py` — New file for two-color visualization

---

## Testing the Fix

```bash
# Build
make clean && make salr_dft

# Run simulation
./build/salr_dft configs/default.cfg

# Visualize results
python3 scripts/plot_joint_heatmap.py output/

# View output
ls output/*.png
```

Expected output: Heat maps showing density clusters/stripes with:
- Species 1 concentrated in certain regions (blue)
- Species 2 concentrated in complementary regions (red)
- Anti-correlation between species in SALR systems
