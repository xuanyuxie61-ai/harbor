# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# Project: Stochastic Vibration and Reliability of Perforated Plates  

This project builds a computational pipeline for random vibration analysis and structural reliability assessment of a perforated thin plate with stochastic Young’s modulus. The material uncertainty is propagated using Polynomial Chaos Expansion (PCE) and optimal sampling, while the dynamic response is treated in the frequency domain with special handling of resonance singularities. Advanced post‑processing includes FORM/SORM reliability analysis, elliptic integral evaluations for nonlinear effects, and quadrature rules for finite element integration.  

The repository contains a main driver script and several supporting modules; only `main.py` is preserved for the benchmark reproduction task. The agent must re‑create all other `.py` files based on the following description.

## Module Inventory  

### `main.py`  
The master script that orchestrates the entire workflow. It:  
- Generates two meshes (a polyiamond‑hexagon mesh and a perforated rectangular mesh).  
- Defines deterministic and stochastic material properties, sets up a Karhunen‑Loève (KL) expansion for the Young’s modulus field.  
- Assembles global stiffness and mass matrices for plane‑stress triangular elements, applies boundary conditions.  
- Solves a modal problem to extract natural frequencies and mode shapes.  
- Verifies tridiagonal solvers (Thomas, Conjugate Gradient, Jacobi) on a beam‑like system.  
- Computes random vibration response using modal superposition and integrates the power spectral density, including a Cauchy Principal Value check near resonance.  
- Performs PCE‑based uncertainty quantification by projecting CVT‑sampled responses onto Legendre‑chaos bases.  
- Maps stochastic samples onto a sphere and computes directional sensitivity.  
- Runs FORM (both HL‑RF and golden‑section search variants) and applies a SORM curvature correction; also evaluates first‑passage failure probability.  
- Computes several special functions and elliptic integrals (gamma, chi‑square, Rayleigh, elliptic K/E, nonlinear oscillation period, hole stress concentration).  
- Validates wandzura triangle quadrature, witherden hexahedron quadrature, and triangulation‑level integration.  
- Recovers nodal stresses from a static solution and exports structural matrices in Matrix Market format.  
- Constructs and checks a magic square test matrix.  

The agent’s implementation of the missing modules must match the interface expected by `main.py`.

### `cvt_sampling.py`  
Implements Centroidal Voronoi Tessellation (CVT) for optimal low‑discrepancy sampling in a hypercube or in a radially adapted manner.  
- **Core functions** compute discrete CVT energy (mean squared distance to nearest generator), perform one Lloyd iteration (centroid update with optional density weighting), and generate a set of CVT generator points via repeated Lloyd steps with random sample batches.  
- An *adaptive* function generates generators concentrated near a spherical shell (radius = reliability index) by introducing a radial weighting factor.  
- A quadrature weight estimation routine counts samples falling into each Voronoi region by Monte Carlo to assign integration weights to the generators.  

### `dynamic_integrator.py`  
Contains frequency‑domain tools for random vibration of structures.  
- Provides a Cauchy Principal Value (CPV) integrator for near‑singular integrands (handles the compensation of the singular term using even‑order Gauss‑Legendre rules).  
- Defines a generic frequency response function (SDOF or modal‑superposition MDOF) and computes response power spectral density (PSD) as `|H|² S_x`.  
- Integrates response PSD with special treatment of the resonant peak for lightly damped systems, returning mean‑square response and zero‑crossing frequency.  
- Computes first‑passage failure probability using Poisson and Vanmarcke approximations based on up‑crossing rate.  
- Performs full modal superposition analysis for a given stiffness/mass system, including mass‑normalization of mode shapes, and can convert displacement PSD to stress PSD via a simplified strain‑displacement matrix.  

### `elliptic_module.py`  
Wraps elliptic integrals and Jacobi elliptic functions (using `scipy.special`) and adds higher‑level structural applications.  
- Complete elliptic integrals of the first and second kind (parameter *m*, not modulus).  
- Jacobi elliptic functions sn, cn, dn.  
- Approximate large‑deflection cantilever beam shape via bisection for the tip angle and numerical integration along the arc length using Jacobi functions.  
- Period of a Duffing oscillator (hardening/softening) expressed through the complete elliptic integral.  
- Stress concentration factor and tangential stress distribution around an elliptical hole using the classical Inglis solution.  

### `fem_quadrature.py`  
Finite element integration and matrix assembly helpers.  
- **Wandzura rules** for triangular elements: predefined symmetric quadrature points and weights for degrees 1, 2, 5; function to map reference triangle coordinates to physical triangles and to integrate over a triangle.  
- **Witherden rules** for hexahedra: analogous integration over a 3D brick with predefined points/weights.  
- **Triangulation‑level integration** that sums nodal values weighted by element areas using a simple average assumption.  
- **Face‑to‑node averaging** to recover smooth nodal stresses from per‑element values.  
- **T3 shape functions** and derivatives (linear triangles).  
- **Matrix assembly** for 2D plane‑stress triangular elements: computes constant‑strain B‑matrix, builds element stiffness via `B^T D B` and consistent mass matrix, assembles global matrices from element contributions.  

### `matrix_exporter.py`  
Handles NIST Matrix Market format I/O.  
- Converts a dense matrix to coordinate format (1‑based indices) respecting symmetry flags.  
- Exports a matrix to file with appropriate header and optional comment.  
- Provides a convenience function to export stiffness, mass, and optional damping matrices.  
- Reads back a Matrix Market file into a dense numpy array, supporting both coordinate and array formats.  

### `mesh_generator.py`  
Constructs triangular meshes and special test matrices.  
- **Polyiamond hexagon mesh**: generates nodes on a triangular lattice bounded by a hexagon of given order, connects them into up/down triangles, identifies boundary nodes.  
- **Perforated plate mesh**: creates a structured rectangular grid, removes nodes inside circular holes, forms triangles per quad and marks boundary.  
- **Magic square**: constructs magic squares (Siamese method for odd orders, doubly‑even method, simplified singly‑even method).  
- **Test stiffness matrix**: builds a symmetric positive definite matrix either from a magic square (`M^T M` + diagonal) or as a 1D bar stiffness.  
- **Nodal stress recovery** (averaging of element stresses) and a function to compute per‑element stresses (plane stress, constant‑strain triangle) and von Mises equivalent.  

### `pce_expansion.py`  
Polynomial Chaos based on normalized Legendre polynomials for uniform variables.  
- Functions to evaluate orthonormal Legendre polynomials via recurrence.  
- Precomputation of 1‑D product integrals (including a weight `x^e`) for stochastic Galerkin projections.  
- Generation of multi‑index sets for total‑degree truncation.  
- Evaluation of multidimensional PCE basis at given sample points.  
- Assembly of the stochastic Galerkin system: builds block‑structured stiffness matrix coupling PCE indices using deterministic stiffness and first‑order expansion of the Young’s modulus, exploiting orthogonality and the Legendre product table.  
- Computation of PCE mean and variance from coefficients.  
- 1‑D Karhunen‑Loève expansion: analytical eigenvalues and (co)sine‑based eigenfunctions for an exponential covariance kernel.  

### `reliability_optimizer.py`  
Numerical reliability methods and special functions.  
- Lanczos‑based log‑gamma and gamma functions; chi‑square PDF, Rayleigh PDF, and normal CDF (using `erf`).  
- **Golden‑section search** for univariate function minimisation.  
- **Armijo backtracking line search**.  
- **FORM via HL‑RF iteration**: iteratively updates the design point using gradient information of the limit‑state function in standard normal space.  
- **FORM with golden‑section search**: scans radial directions, uses golden section to find the distance to the limit surface, and selects the closest point.  
- **SORM (Breitung correction)**: computes principal curvatures of the
