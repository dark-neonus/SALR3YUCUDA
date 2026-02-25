# Density Distribution in Particle Monolayers with Competing Interactions

This document summarizes the research model for calculating spatial density distributions in two-component mixtures. [cite_start]The study focuses on systems where particles exhibit "competing" interactions (Short-Range Attraction, Long-Range Repulsion - SALR)[cite: 1].

---

## 1. Physical Model: The 3Y Potential
[cite_start]The system consists of two particle species (Component 1 and 2) interacting via a **Triple Yukawa (3Y) potential**[cite: 1]:

$$U_{ij}(r) = \sum_{m=1}^{3} A_{ij}^{(m)} \frac{e^{-\alpha_{ij}^{(m)}r}}{r}$$

### Key Parameters:
* [cite_start]**$A_{ij}^{(m)}$**: Energy parameters defining the strength of interaction[cite: 1].
* [cite_start]**$\alpha_{ij}^{(m)}$**: Damping parameters determining the range of interaction[cite: 1].
* [cite_start]**Interaction Matrix**: Specific values for $A_{11}$, $A_{12}$, and $A_{22}$ (and their respective $\alpha$ values) are provided to simulate specific physical behaviors, such as cluster or stripe formation[cite: 1].

---

## 2. Theoretical Framework
[cite_start]The goal is to find the spatial density distributions $\rho_1(\mathbf{r})$ and $\rho_2(\mathbf{r})$ by minimizing the **Grand Thermodynamic Potential** ($\Omega$) in the **Mean Field Approximation (MFA)**[cite: 2].

### The Grand Potential Equation:
[cite_start]The potential $\Omega$ is a functional of the densities and includes terms for[cite: 2]:
* Ideal gas contributions (entropy).
* Chemical potentials ($\mu_1, \mu_2$).
* Interaction energies between particles of the same and different types.

### Euler-Lagrange Equations:
[cite_start]Minimization leads to the following expressions for local density[cite: 1, 3]:
$$\rho_1(\mathbf{r}) = \rho_{1,b} \exp[-\beta (\Phi_{11}(\mathbf{r}) + \Phi_{12}(\mathbf{r}) - \Phi_{11,b} - \Phi_{12,b})]$$
$$\rho_2(\mathbf{r}) = \rho_{2,b} \exp[-\beta (\Phi_{21}(\mathbf{r}) + \Phi_{22}(\mathbf{r}) - \Phi_{21,b} - \Phi_{22,b})]$$

Where:
* [cite_start]**$\rho_{i,b}$**: Bulk density of component $i$[cite: 2, 3].
* [cite_start]**$\Phi_{ij}(\mathbf{r})$**: Energy contributions from interactions in an inhomogeneous system[cite: 3].
* [cite_start]**$\Phi_{ij,b}$**: Energy contributions in a homogeneous (bulk) system[cite: 3].

---

## 3. Boundary Conditions
[cite_start]The model is tested under three distinct spatial constraints[cite: 1]:

| Type | Description | Mathematical Constraint |
| :--- | :--- | :--- |
| **PBC** | Periodic Boundary Conditions | $\rho(x_i \pm L_x, y_j \pm L_y) = \rho(x_i, y_j)$ |
| **W2** | Two Parallel Walls | $\rho = 0$ if $x < 0$ or $x \geq L_x$; Periodic in Y |
| **W4** | Square Cage | $\rho = 0$ if $x, y < 0$ or $x, y \geq L_{x,y}$ |

---

## 4. Numerical Task: What Needs to be Done
[cite_start]To solve for the density distributions, an iterative numerical approach must be implemented[cite: 4, 5]:

### A. Space Discretization
* [cite_start]Divide the 2D area ($L_x \times L_y$) into a grid of $N_x \times N_y$ nodes[cite: 5].
* [cite_start]Current parameters: $16.0 \times 16.0$ area with a step $\Delta x, \Delta y = 0.2$, resulting in a $80 \times 80$ grid ($6400$ nodes)[cite: 5].

### B. Picard Iterative Method
[cite_start]Since the density at a point depends on the density of all surrounding points, the solution is reached through successive approximations[cite: 4]:
1.  **Calculate New Guess**: Compute the current density based on the potential integrals.
2.  **Mixing**: Update the density for the next step ($t+1$) using a mixing parameter $\xi$ (typically $0.2$) to prevent divergence:
    $$\rho^{(t+1)} = \xi K[\rho^{(t)}] + (1 - \xi)\rho^{(t)}$$
3.  [cite_start]**Potential Cutoff**: For efficiency, only calculate interactions within a radius $r_c = 8.0$[cite: 5].

### C. Convergence Criteria
[cite_start]The iterations continue until the root-mean-square deviation between two consecutive steps is sufficiently small[cite: 5]:
$$\sqrt{\frac{1}{N_x N_y} \sum_{i,j} [\rho^{(t+1)}(x_i, y_j) - \rho^{(t)}(x_i, y_j)]^2} < \epsilon = 10^{-8}$$

---

## 5. Required Output
[cite_start]The final result should produce **Spatial Density Maps**, visualizing how Component 1 (often forming the main structure) and Component 2 (often acting as a solvent or secondary layer) arrange themselves in the 2D plane[cite: 1].