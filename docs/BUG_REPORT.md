# Bug Report and Fixes

This document summarises the issues discovered during development and the corrections applied to the CPU DFT solver.

---

## Summary

| # | Type | Severity | Component | Status |
|---|------|----------|-----------|--------|
| 1 | Physics | Critical | solver_cpu.c | Fixed |
| 2 | Physics | High | configs/default.cfg | Fixed |
| 3 | Numerical | Medium | solver_cpu.c | Fixed |
| 4 | Visualisation | Medium | scripts/ | Fixed |

---

## Bug 1: Uniform Solution Instead of Structured Phase

**Component:** `src/cpu/solver_cpu.c`

**Symptom:** The solver converged to a spatially uniform density distribution regardless of initial conditions.

**Cause:** At the original temperature T = 12, the system is above the spinodal temperature for microphase separation. The thermodynamically stable solution at this temperature is uniform.

**Fix:**
1. Reduced the default temperature from T = 12 to T = 2.0 (below the spinodal).
2. Retained chemical-potential renormalisation (required for correct thermodynamics).
3. Kept anti-checkerboard smoothing active to suppress grid-scale numerical artefacts.

---

## Bug 2: Default Temperature Too High

**Component:** `configs/default.cfg`

**Original value:** `temperature = 12.0`

At T = 12, the inverse temperature is $\beta = 1/12 \approx 0.083$, yielding an effective interaction strength too weak to drive microphase separation.

**Fixed value:** `temperature = 2.0`

At T = 2.0, $\beta = 0.5$, placing the system within the structured-phase region of the parameter space.

---

## Bug 3: Checkerboard Instability Without Smoothing

**Component:** `src/cpu/solver_cpu.c`

**Symptom:** At low temperatures (T < 1.5) without smoothing, the solution exhibited unphysical checkerboard oscillations and divergent density peaks.

**Fix:** Re-enabled the 5-point Laplacian smoothing with $\varepsilon_s = 0.01$. This suppresses the grid-Nyquist mode while preserving physical structure at wavelengths much larger than the grid spacing. See `src/cpu/MATH.md` (section 11) for the spectral analysis.

---

## Bug 4: No Joint Species Visualisation

**Component:** `scripts/`

**Issue:** Only individual species heatmaps were available. The spatial anti-correlation between species, characteristic of SALR systems, could not be inspected in a single figure.

**Fix:** Added `scripts/plot_joint_heatmap.py`, which produces a three-panel figure (species 1, species 2, and a combined overlay).

---

## Known Limitations

1. **Convergence oscillation.** The solver error may oscillate near the phase boundary rather than decreasing monotonically. Adaptive mixing parameters or Anderson acceleration could improve this.
2. **No underflow protection.** At very low temperatures, `exp(-beta * Phi)` may underflow. A guard (`if (arg < -700) K[k] = 0.0`) would prevent silent zeroing.
3. **Mass conservation vs. convergence rate.** The per-iteration mass renormalisation is thermodynamically correct but may slow structure formation. Periodic (rather than per-step) renormalisation is a possible alternative.

---

## Recommended Parameters for Structure Formation

```ini
[physics]
temperature = 2.0

[solver]
max_iterations = 5000
xi1 = 0.05
xi2 = 0.05
tolerance = 1.0e-6
```
