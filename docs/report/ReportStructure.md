# GPU-Accelerated Density Functional Theory Solver for Binary SALR Colloids (SALR3YUCUDA)

## Abstract
General sum up of our report and what we have done

## 1. Introduction
Explanation of basic concepts like colloids, SALR interaction, SALR systems, "mermaid" potential.

Mention about real experiments and their results in topic of colloidal systems with "Mermaid" interaction.

Introduce DFT and its computation demands.

Introduce project and general overview ovew what we are doing

Images:
 - SALR pair potential plot (Mermaid Chart) from article

### 1.1 Motivation
Explain why binary colloidal mixtures with SARL interation are interesting fo researchers.

Introduce other approaches to model this systems.

Comparison of DFT advantages over particle-resolved simulations like MD and MC.

List of benefits: direct equilibrium access, clean boundary conditions, parameter scalability.

## 2. Theoretical Background
### 2.1 SALR Interactions
Short-range attraction and long-range repulsion origins in colloidal systems and microphase separation patterns.

### 2.2 Triple Yukawa Potential
Description of the pair interaction model for species i and j.

List of formulas:
 - U_ij(r) triple Yukawa potential equation

List of definitions:
 - amplitude coefficients
 - inverse screening lengths
 - cutoff radius

### 2.3 Mean-Field Density Functional Theory
Derivation of equations for equilibrium density profiles and interaction fields.

List of formulas:
 - Euler-Lagrange equations for species 1 and 2
 - Interaction field convolution integral

List of definitions:
 - inverse temperature
 - bulk density
 - bulk field values

### 2.4 Picard Iteration
Iterative solution method for nonlinear integral equations and convergence monitoring.

List of formulas:
 - Picard mixing equation
 - Root-mean-square convergence criterion

List of definitions:
 - mixing parameter

### 2.5 Boundary Conditions
Explanation of spatial constraints supported by the solver.

List of definitions:
 - PBC (Periodic Boundary Conditions)
 - W2 (Two Walls)
 - W4 (Four Walls)

## 3. Implementation
### 3.1 Discretization
Transition from continuous grid to discrete nodes.

List of formulas:
 - Discrete convolution sum

### 3.2 Algorithm Overview
Step-by-step logic of the execution pipeline from initialization to output.

### 3.3 CPU Parallelization
Details on OpenMP usage for potential tables, convolution, and mixing.

### 3.4 CUDA Implementation
GPU acceleration via specialized kernels and memory optimization.

List of terminology:
 - coalesced memory access
 - constant memory
 - thread blocks

### 3.5 GUI and Session Database
Frameworks used for user interface and data storage.

List of terminology:
 - Qt 6
 - SQLite
 - HDF5

## 4. Results on our machines
### 4.1 Simulation Parameters
Constants and settings used for the benchmarks.

Table with all parameters presented in config which will be used for all benchmarks and visualization.

### 4.2 Periodic Boundary Conditions (PBC)

#### 4.2.1 Random starting distribution
Analysis of stripe formation on a periodic domain.

Images:
 - 2D heatmap of PBC density
 - 3D scatter view of PBC density

#### 4.2.2 Sinusoidal starting distribution
Analysis of stripe formation on a periodic domain.

Images:
 - 2D heatmap of PBC density
 - 3D scatter view of PBC density

### 4.3 Two-Wall Confinement (W2)

#### 4.3.1 Random starting distribution
Analysis of density depletion and wall-parallel modulation.

Images:
 - 2D heatmap of W2 density
 - 3D scatter view of W2 density

#### 4.3.2 Sinusoidal starting distribution
Analysis of density depletion and wall-parallel modulation.

Images:
 - 2D heatmap of W2 density
 - 3D scatter view of W2 density

### 4.4 Four-Wall Confinement (W4)

#### 4.4.1 Random starting distribution
Analysis of boundary layers and central concentration.

Images:
 - 2D heatmap of W4 density
 - 3D scatter view of W4 density

#### 4.4.2 Sinusoidal starting distribution
Analysis of boundary layers and central concentration.

Images:
 - 2D heatmap of W4 density
 - 3D scatter view of W4 density

### 4.5 Performance on our machines
#### 4.5.1 Description of metrics we will measure
Why we choosen specific metrics.

List of terminology:
- Execution time(just for clarity)
- Speedup

#### 4.5.2 Description of parameters and hardware used for performance test

Table of parameter given to our programm at benchmark.

CPU info of machine we were using(list specific values and explain what this parameters represent and mean): 
- CPU frequency
- Number of cores
- Cores type
- All levels cache size
- RAM size and type

GPU info(list specific values and explain what this parameters represent and mean):
- Model
- Architecture
- Cores numbers
- Clock Speed
- VRAM Capacity, Memory Type, Bus Width
- Memory bandwidth'
- Throughput in TFLOPS
- Presented cache levels and their capacity

#### 4.5.3 Methodology description

Explanation of how did we runned this benchmark, under what conditions, how many times each. How we calculate values.

Description and list of starting conditions:

run test:
test(num_of_threads, grid_size) 

for all combination of parameters:

num_of_threads: {1, 2, 4, 8, 12, 16, 20, GPU}

grid_size: {16, 32, 64, 128, 164}

#### 4.5.4 Benchmark results

Analysis of received results

Comparation of GPU and CPU results and metrics(iterations till stop of algorithm, finall error, error as a function of time for CPU and CUDA) to showcase identity of results.

Images:
- Execution time as a function of number of threads and GPU line for comparasion
- Speedup of CUDA over CPU(different num of threads) as a function from grid size
- Speedup of multiple threads and CUDA over single thread as a function from grid size

## 5. Results on cluster

...

Same sections and information as in section 4 but for cluster benchmarks

...


## 6. Discussion
Synthesis of solver behavior, physical interpretation of patterns, and performance gains.

## 7. Conclusion
Summary of project achievements and goals for future development.

List of project vectors:
 - 3D domain extension
 - Multi-GPU scaling

## Acknowledgment
Acknowledgment

## References
1. C. P. Royall, “Hunting mermaids in real space: known knowns, known
unknowns and unknown unknowns,” Soft Matter, vol. 14, no. 20,
pp. 4020–4028, 2018.
2. Y. Liu and Y. Xi, “Colloidal systems with a short-range attraction
and long-range repulsion: Phase diagrams, structures, and dynamics,”
Current Opinion in Colloid & Interface Science, vol. 39, pp. 123–136,
2019
3. J.-P. Hansen and I. R. McDonald, Theory of Simple Liquids: With
Applications to Soft Matter, 4th ed. Academic Press, 2013.
