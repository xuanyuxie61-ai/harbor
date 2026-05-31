# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# Project Description: Drip-Line Nuclear Structure and Decay Dynamics Suite

This project provides a computational pipeline for studying the structure, reactions, and decay of neutron‑drip‑line nuclei. It combines mean‑field potential construction, radial Schrödinger‑equation solvers, self‑consistent pairing (HFB‑BCS), mass‑surface interpolation, stochastic Langevin dynamics, β‑decay statistics, reaction cross sections, three‑dimensional density meshes, and high‑dimensional quadrature. The main entry point (`main.py`) orchestrates all these calculations for a representative nucleus (e.g. ²⁸O) and prints results to stdout.

## File Inventory and Responsibilities

* **`constants.py`**  
  Physical constants (ħc, nucleon masses, …), conversion factors, liquid‑drop model coefficients, Woods–Saxon default parameters, pairing strength, and numerical defaults. Also provides two utility functions: `reduced_mass` and `hbar2_over_2m`.

* **`nuclear_potential.py`**  
  Construction of deformed optical potentials. Defines real spherical harmonics (Y₂₀, Y₃₀, Y₄₀), a deformed radius function, Woods–Saxon form factor and its derivative, spin‑orbit potential, and Coulomb potential. Provides functions to assemble the total neutron/proton potentials (`build_neutron_potential`, `build_proton_potential`) for a given deformation.

* **`radial_solver.py`**  
  Solves the radial Schrödinger equation for a given orbital angular momentum. Contains Gauss–Lobatto node/weight routines, Vandermonde‑based quadrature weights, a Lagrange differentiation matrix, and a stable finite‑difference (Numerov‑like) solver (`solve_radial_schroedinger`) that returns bound‑state energies and wave functions. Also includes helpers for radial matrix elements and kinetic‑energy matrix elements.

* **`hfb_selfconsistent.py`**  
  Hartree‑Fock‑Bogoliubov (HFB) solver using the BCS approximation. Implements BCS occupation amplitudes, a conjugate‑gradient linear solver for symmetric positive‑definite matrices, and a self‑consistent pairing loop (`solve_hfb_bcs`). Also provides functions to build the one‑body density matrix and the pairing tensor.

* **`mass_surface.py`**  
  Multidimensional interpolation of nuclear masses. Contains the liquid‑drop model binding energy, the atomic mass from LDM, a shell‑correction formula from single‑particle spectra, and a class `NuclearMassSurface` that performs radial‑basis‑function interpolation of mass residuals. Methods include point evaluation, separation energy, drip‑line location, and a curvature estimator. Also includes a Genz oscillatory test function.

* **`decay_statistics.py`**  
  Statistical modelling of decay chains. Computes the non‑central Beta CDF (via a series expansion) and approximate PDF. Provides a Monte‑Carlo discrete‑chain simulation using inverse‑CDF sampling. Also contains functions for β‑decay Q‑value, an approximate half‑life formula, and a Bayesian credible interval for neutron‑drip existence using the non‑central Beta distribution.

* **`stochastic_dynamics.py`**  
  Overdamped Langevin dynamics of nucleons. Implements a single Euler–Maruyama step, full trajectory generation, ensemble averaging with mean‑squared displacement (MSD) analysis, diffusion‑coefficient extraction, nuclear temperature from the Fermi‑gas model, and an evaporative decay rate (Weisskopf estimate).

* **`reaction_phasespace.py`**  
  Reaction cross sections and phase‑space integrals. Computes unit‑disk monomial integrals (exact formula), a Gaussian disk integral, a peripheral transfer probability and its total cross section, a Coulomb breakup cross section in the equivalent‑photon approximation, and a normalised angular‑momentum coupling weight.

* **`density_mesh.py`**  
  Generation and manipulation of 3‑D tetrahedral meshes for nuclear density distributions. Provides icosahedron vertices/faces, a recursive spherical triangulation, a tetrahedral spherical‑shell mesh builder, tetrahedron volume, mesh integration of a density function, a deformed Fermi density, RMS‑radius extraction, and an XML mesh writer.

* **`quadrature_engine.py`**  
  High‑dimensional integration tools. Stores hard‑coded Fekete quadrature rules on the reference triangle (degrees 1–7), functions to integrate over a physical triangle, sparse‑grid Clenshaw–Curtis quadrature (Smolyak construction) for hyper‑cubes, and a dedicated function to integrate a deformation probability density over a (β₂,β₃) rectangle.

* **`main.py`**  
  Main entry point. Sets up a target nucleus (Z, N, deformation parameters), and sequentially calls the other modules to:
  1. Build neutron and proton mean‑field potentials.
  2. Solve the radial Schrödinger equation for a few angular momenta.
  3. Run self‑consistent HFB‑BCS for neutrons and protons.
  4. Build a local mass surface from
