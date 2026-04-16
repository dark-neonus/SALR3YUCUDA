# Annotation for project report

## General Context

### Insights

#### Results depends heavily on starting distribution
Final density distribution depends heavilu on starting distribution.
This is well seen on example of PBC when starting random distribution produce big chaotic structures and starting with sinusoids produce diagonal stripes.

#### Strange patterns at low temperatures
At low temperatues(not starting immidiatelly, but appearing slowly proportionally to temperature) density distribution start showing point increases of distribution of species across all field, producing weird dotted results. For now we not sure if it is right behaviour or bug.

#### Direct approach collect error, normalization is needed, it indluence result a lot
When we implemented integration and general algorythm, all our results after some time converged to uniform density of around 0 or produce infinity and NaN at some points of grid. We introduced normalization and Laplace smoothing to fix this problems and they helped. When changed koefficients, for example Laplace epsilon, results differ a lot. 

## Directories and files

### requirements
Contain files connected to project scope and report requirements

#### Course_Requirements.md
Contain info about requirements for project and report. Contain a lot of water but also usefull info about general idea of report.

#### Project_Description.md and Project_Description.pdf
Contain general info about projects, its scope, motivations and ways. It is first description of project given to us. You can cite this file, including in motivation or intro.

### sources
Contain sources that we really have read for reference

#### royallHuntingMermaidsSoftMatter2018.md
Markdown version of article with same name. Not related too much to our project, but still describe same topic, same motivation. It is very good for referencing to for motivation, general concepts and practical use of theory we trying to calculate.

### SALR3YUCUDA
Directory with report files

##### SALR3YUCUDA.tex
LaTeX report file. Main report file.

##### src
Directory that contain figures, images, plots etc that can or shoudl be used in report.

###### performance_real_params.svg
SALR DFT Performance Analysis. Contain plots:
- CPU(OpenMP) and CUDA Runtime Conparasion
- CPU(OpenMP) and CUDA Iterations to Stop(same amount, show possible identity in convergence(identity should be, but this plot just dont prove it 100%))
- CPU(OpenMP) and CUDA Final Error vs Tolerance(same)
- CPU(OpenMP) CPU 20 threads and CUDA Convergence Trajectory(same)

Parameters of machine that have runned this tests:
```
Grid: 320 x 320 (dx=0.1, dy=0.1)
T=8.0, rho1=0.2, rho2=0.2, rc=8.0
boundary=PBC, init=sinusoids, max_iter=50000, tol=1.0e-8
CPU: 13th Gen Intel(R) Core(TM) i7-13700H
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
OpenMP threads (CPU run): 20
```

###### PBC_2D.png and PBC_3D.png
Heatmap and Scatterplot showing results of program run at parameters:
parameter_set_1.cfg
```cfg
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

# TODO: Add random seed as parameter
```
Yellow represent species1, blue represent species2.