# Annotation for project report

## General Context

### Insights

#### Results depend heavily on the starting distribution
The final density distribution depends strongly on the initial condition.
For PBC runs, a random start tends to produce large chaotic structures, while a sinusoidal start more often converges to diagonal stripe patterns.

#### Strange patterns at low temperatures
At low temperatures, the density field can develop small point-like spikes that appear gradually as the system evolves. We are still not sure whether this is a physical effect or a numerical artifact.

#### Direct iteration accumulates error; normalization is necessary
When we implemented the direct integration algorithm, the solution often drifted toward near-zero uniform density or produced inf and NaN values on parts of the grid. Adding normalization and Laplace smoothing fixed most of these failures, but the result is sensitive to coefficients such as the Laplace epsilon.

## Directories and files

### requirements
Contains files connected to the project scope and report requirements.

#### Course_Requirements.md
Contains the report and project requirements. It is verbose, but still useful for understanding the expected structure and general direction of the report.

#### Project_Description.md and Project_Description.pdf
Contain the original project description, its scope, motivation, and general goals. These are valid sources for the introduction and motivation sections.

### sources
Contains external sources that were actually read and used for reference.

#### royallHuntingMermaidsSoftMatter2018.md
Markdown version of the Royall et al. article. It is not a direct match to our implementation, but it is a strong source for the motivation, general SALR context, and the physical interpretation of the potential shape.

### SALR3YUCUDA
Directory with report files.

##### SALR3YUCUDA.tex
Main LaTeX report file.

##### src
Directory with figures, plots, and supporting images used in the report.

###### Mermaid_Chart.png
SALR pair-potential illustration used in the introduction. It shows the attractive head and repulsive tail of a mermaid-style interaction and is reproduced from the Royall article. Taken from royallHuntingMermaidsSoftMatter2018.

###### PBC_2D.png
2D heatmap for the periodic-boundary-condition case. It shows stripe-like microphase separation for the chosen parameter set.
Runned for parameter_set_1:
```
# ── Computational grid ──────────────────────────────────────────────────────
[grid]
dx  = 0.1                   # discretisation step along x
dy  = 0.1                   # discretisation step along y
nx  = 160                   # grid nodes along x
ny  = 160                   # grid nodes along y
# Lx and Ly are derived: Lx = dx * nx, Ly = dy * ny
boundary_mode = PBC     # PBC | W2 (walls in x) | W4 (walls on all sides)
init_mode = sinusoids      # random | sinusoids | trivial

# ── Physics ─────────────────────────────────────────────────────────────────
[physics]
temperature    = 8.0     # environment temperature, T
rho1           = 0.4         # average density of species 1
rho2           = 0.2         # average density of species 2
cutoff_radius  = 8.0         # interaction cutoff radius, r_c

# ── Interaction matrix — 3Y (Yukawa) potential parameters ───────────────────
[interaction]
A_11_1 =  150.6561472
A_11_2 = -122.6130119
A_11_3 =   27.11810649
a_11_1 =    1.92325375
a_11_2 =    1.261150
a_11_3 =    0.756690

A_12_1 =  413.5305236
A_12_2 = -275.6870157
A_12_3 =    0.0
a_12_1 =    3.405465108
a_12_2 =    3.0
a_12_3 =    1.0

A_22_1 =  413.5305236
A_22_2 = -275.6870157
A_22_3 =    0.0
a_22_1 =    3.405465108
a_22_2 =    3.0
a_22_3 =    1.0

# ── Picard iteration solver ─────────────────────────────────────────────────
[solver]
max_iterations         = 50000
tolerance              = 1.0e-8      # convergence accuracy, epsilon
xi1                    = 0.02       # mixing coefficient for species 1
xi2                    = 0.02       # mixing coefficient for species 2
error_change_threshold = 1.0e-10     # when error change < this, apply damping
xi_damping_factor      = 1.0         # multiply xi by this when error stabilizes

# ── Output ──────────────────────────────────────────────────────────────────
[output]
output_dir = output/
save_every = 100             # save density profile every N iterations
```

###### PBC_3D.png
3D scatter view of the same periodic-boundary-condition run. Yellow points represent species 1 and blue points represent species 2.
Runned for parameter_set_1

###### W2_2D.png
2D heatmap for the two-wall boundary condition. The image shows depletion near the walls and an ordered pattern aligned with the confinement geometry.
Runned for parameter_set_1 with boundare conditions W2.

###### W2_3D.png
3D scatter view of the two-wall boundary case. The density is concentrated between the walls and forms layered structures along the free direction.
Runned for parameter_set_1 with boundare conditions W2.

###### W4_2D.png
2D heatmap for the four-wall confinement case. The field is depleted near all boundaries and the dense region remains mostly in the center.
Runned for parameter_set_1 with boundare conditions W4.

###### W4_3D.png
3D scatter view of the four-wall boundary case. The result shows a compact central density structure with strong boundary depletion and corner effects.
Runned for parameter_set_1 with boundare conditions W4.

###### performance_real_params.svg
Performance comparison figure for a realistic parameter set. It contains CPU(OpenMP) and CUDA runtime comparison, iterations to stop, final error versus tolerance, and convergence trajectories. The recorded test setup was:

```text
Grid: 320 x 320 (dx=0.1, dy=0.1)
T=8.0, rho1=0.2, rho2=0.2, rc=8.0
boundary=PBC, init=sinusoids, max_iter=50000, tol=1.0e-8
CPU: 13th Gen Intel(R) Core(TM) i7-13700H
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
OpenMP threads (CPU run): 20
```

###### speedup_analysis.svg
Speedup-analysis figure showing how CUDA compares to CPU baselines across different thread counts. It includes CUDA speedup over CPU, CPU parallel scaling, and comparisons against 1, 2, 4, 8, 12, 16, and 20 OpenMP threads.

###### placeholder.png
Minimal placeholder image. It is used as a stand-in asset when a final GUI screenshot or another illustration is not yet available.