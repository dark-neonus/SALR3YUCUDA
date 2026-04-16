# Annotation for project report

## General Context

## Directories and files

### requirements
Contain files connected to project scope and report requirements

#### Course_Requirements.md
Contain info about requirements for project and report. Contain a lot of water but also usefull info about general idea of report.

#### Project_Description.md and Project_Description.pdf
Contain general info about projects, its scope, motivations and ways. It is first description of project given to us.

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