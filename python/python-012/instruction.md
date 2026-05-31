# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# Project Description: Topological Insulator Surface State Transport Simulation

## Overview
This project computes quantum transport properties of magnetically doped topological insulator (TI) surface states. The central physics is the massive Dirac Hamiltonian with hexagonal warping, Berry curvature, disorder scattering, and linear response theory. The entry point is `main.py`, which orchestrates a full simulation pipeline, printing results and writing output files. All other Python modules provide the computational building blocks.

After evaluation, `main.py` will be retained, and the remaining modules must be reimplemented from scratch based on this description.

## Module Descriptions

### dirac_surface.py
Defines the `DiracSurfaceHamiltonian` class. It encapsulates the low‑energy effective Hamiltonian for TI surface states:  
H(k) = ℏ v_F (k_x σ_y − k_y σ_x) + Δ σ_z + optional hexagonal warping term.  
Internally it uses Pauli matrices and provides methods to compute eigenvalues, eigenvectors, spin texture, and velocity operators. A utility function `effective_mass_tensor` returns the inverse effective mass tensor near the Dirac point.

### berry_curvature.py
Contains `BerryCurvatureCalculator`. It depends on `DiracSurfaceHamiltonian` and computes:
- Numerical Berry curvature via finite differences of eigenvectors.
- Analytical Berry curvature for the massive Dirac model (upper/lower band).
- Chern number by integrating Berry curvature over a momentum grid.
- Anomalous Hall conductivity in unit of e²/h, optionally with temperature and Fermi‑Dirac statistics.
- Berry phase along a closed 1D k‑space path.

### disorder_scattering.py
Defines `DisorderScattering`, which depends on `DiracSurfaceHamiltonian`. It models impurity scattering for TI surface states. Capabilities include:
- Density of states (analytical formula).
- Spin overlap factor between two momentum states.
- Born approximation scattering rate, transport scattering time, mean free path, and diffusivity.
- Retarded self‑energy in the Born approximation.
- Self‑consistent T‑matrix evaluation.
- Skew scattering rate from spin‑momentum locking.

### fermi_surface.py
Holds the `FermiSurface` class that describes Fermi surface geometry for TI surface states. It computes Fermi wavevector, Fermi velocity, Fermi surface area/circumference, density of states, carrier density, cyclotron mass, and cyclotron frequency. It can sample momenta uniformly on circular or ellipsoidal Fermi surfaces, generate warped Fermi surfaces (hexagonal warping), and produce chord‑length distributions for scattering problems.

### geometry_utils.py
Provides `SampleGeometry` to define sample shapes (square, hexagon, circle, or a complex “tortoise” boundary). It generates polygon vertices, traces a boundary from a direction word encoding, tests point‑in‑polygon, creates a grid of points inside the shape, and computes edge length and area.

### io_manager.py
The `IOManager` class handles text‑based input/output. It writes simulation parameter headers, matrices, vectors, and zone‑structured data (similar to Tecplot format). It can also parse variable definitions and read zone data. Results are saved to a user‑specified output directory.

### kubo_conductivity.py
Implements `KuboConductivity` (depends on `DiracSurfaceHamiltonian` and `DisorderScattering`). It calculates transport coefficients using Kubo–Greenwood / Boltzmann approaches:
- DC longitudinal conductivity (Drude‑Boltzmann and semiclassical forms).
- Intrinsic anomalous Hall conductivity from Berry curvature.
- Skew scattering and side‑jump contributions to the Hall effect.
- Total Hall conductivity and spin Hall conductivity.
- Thermoelectric coefficients (Seebeck and Lorenz number) via the Mott formula.

### tight_binding_surface.py
Contains `Tight‑BindingSurface`, a finite‑size lattice model that discretises the Dirac Hamiltonian on a 2D square lattice. It builds the Hamiltonian with nearest‑neighbour hopping and an onsite gap, diagonalises it, computes local density of states, identifies edge states, builds current operators, and calculates finite‑size conductivity using the Kubo formula.

### spectral_integrator.py
Provides several integration tools:
- `LatticeIntegrator`: implements lattice rules, including Fibonacci lattice integration for periodic functions on a torus.
- `JacobiQuadrature`: Gauss‑Jacobi quadrature for integrals with algebraic endpoint singularities.
- `MonteCarloIntegrator`: uniform sampling integration over a disk, hypersphere positive octant, or 2D Brillouin zone.

### nonlinear_solver.py
The `NonlinearSolver` class offers root‑finding methods: Snyder’s bracketing method, bisection, and secant. It includes higher‑level functions to find the Fermi level from a target carrier density, compute a self‑consistent scattering time, and locate T‑matrix poles.

### utils_special.py
Collects special mathematical functions:
- `TrigammaFunction`: evaluates the trigamma function ψ′(x) and provides a derivative of the Fermi‑Dirac integral.
- `CarlsonEllipticIntegrals`: symmetric elliptic integrals R_F and R_D, plus ellipsoid surface area computation.
- `Interpolation2D`: inverse‑distance weighting and radial basis function interpolation for scattered 2D data.

### main.py (preserved)
Orchestrates the full simulation. It imports all modules, sets material parameters, and runs a sequence of sections:
1. Construct the Dirac Hamiltonian.
2. Analyse Fermi surface geometry.
3. Compute Berry curvature, Chern number, and Berry phase.
4. Calculate disorder scattering rates and self‑energy.
5. Evaluate transport coefficients via the Kubo formula.
6. Verify results using a tight‑binding lattice model.
7. Test spectral integration methods.
8. Perform self‑consistent nonlinear solutions.
9. Evaluate special functions.
10. Analyse sample geometry.
11. Write results to disk.

The script prints a summary of key physical quantities to stdout and saves structured output files.

## Implementation Guidelines
- All physical constants (ℏ, e, k_B, etc.) should be defined or imported from standard libraries.
- Modules may depend on each other as indicated; circular imports should be avoided.
- Numerical integrations over momentum space typically use uniform grids, finite differences, or the provided integration classes.
- The tight‑binding model must handle both open and periodic boundary conditions, and correctly incorporate spin‑1/2 degrees of freedom.
- The self‑consistent T‑matrix solver should compute G₀(E) by momentum integration and apply the scalar T‑matrix formula.
- An analytical Berry curvature formula for the massive Dirac model (without warping) should be implemented; numerical finite‑difference evaluation should also be available.
- The I/O manager writes plain text files; no visualisation libraries are required.
- The special functions (trigamma, elliptic integrals) should be implemented from scratch or using efficient numerical algorithms described in the respective project references.
