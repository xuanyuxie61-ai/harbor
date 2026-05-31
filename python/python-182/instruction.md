# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# Project Description: Bayesian Hierarchical Calibration of a Spatially-Coupled Reaction-Diffusion System

This project implements a full Bayesian inference pipeline for a physical model that combines FitzHugh‑Nagumo reaction kinetics with a Helmholtz wave field on an annular domain. The inference is performed on synthetic spatial observations using a circular Gaussian Markov random field (GMRF) prior, polynomial surrogate acceleration, and adaptive MCMC with rotation-invariant proposals. Model evidence is estimated via Monte Carlo quadrature over several canonical domains.

Only `main.py` will be retained; all other modules must be re‑implemented based on this description.

---

## File Overview

### main.py
Entry point that invokes the complete pipeline and prints posterior summaries. It calls a single function from the inference engine and displays the results.

### inference_engine.py
High‑level orchestration module. It contains:

- A function to generate synthetic observations on a circular arrangement of sensors. This uses:
  - The Helmholtz exact solution on a disk (from `forward_models`).
  - The FitzHugh‑Nagumo stationary voltage (from `forward_models`), coupled through a simple finite‑element basis interpolation on a reference triangle.
  - A given spatial field of stimulus contributions.
- A prior density (`log_prior`) over nine parameters: four scalar physics parameters (with truncated Gaussian priors), four spatial coefficients (with a circular GMRF prior), and a log‑transformed noise scale (Jeffreys prior).
- A likelihood function (`log_likelihood`) that computes a Gaussian log‑likelihood using either the exact forward model or a fast polynomial surrogate, depending on parameter values.
- A posterior function that combines prior and likelihood.
- A main routine (`run_bayesian_inference`) that:
  1. Generates synthetic data.
  2. Builds the GMRF precision matrix and computes its covariance via a special periodic tridiagonal solver (`periodic_solver`).
  3. Builds a polynomial surrogate for the FitzHugh‑Nagumo model (`surrogate`).
  4. Initialises the MCMC chain using Latin hypercube samples (`utils`).
  5. Runs an adaptive MCMC sampler (`mcmc_sampler`) that uses random‑walk proposals, a separate Gibbs‑like update for the noise parameter, and periodic rotation proposals to preserve symmetry.
  6. After burn‑in, estimates marginal evidence on three canonical domains (line, square, triangle) using Monte Carlo quadrature (`bayesian_quadrature`).

### mcmc_sampler.py
Provides an adaptive MCMC sampler function. The sampler iterates over:
- A block random‑walk Metropolis‑Hastings update on the continuous parameters (excluding the log‑scale noise).
- A separate one‑dimensional proposal for the log‑scale noise (approximating a Gibbs step).
- Periodic “unicycle” rotation proposals that cyclically permute the four spatial coefficients to enforce rotational invariance on the circular domain.

All random numbers are drawn from a custom PRNG object. The function returns the full chain, log‑posterior trace, and an acceptance rate estimate.

### bayesian_quadrature.py
Monte Carlo integration utilities over:
- **1D interval:** uniform random or golden‑ratio ergodic (low‑discrepancy) sampling; exact monomial integrals.
- **Unit square:** uniform random sampling.
- **Reference triangle (0,0)–(1,0)–(0,1):** uniform random sampling via Dirichlet/‑exponential spacings; exact monomial integrals.

Wrappers `integrate_1d`, `integrate_square`, `integrate_triangle` accept a callable integrand, sample size, method, and a random generator. The triangle integrator automatically multiplies by the triangle area (0.5).

### forward_models.py
Contains two physical forward models:

1. **FitzHugh‑Nagumo (FHN) system:**
   - Right‑hand side function for the ODE (dv/dt, dw/dt) depending on four parameters (a, b, c, d).
   - A classical forward Euler integrator that given a callable RHS function returns time and state trajectories.
   - A convenience function `fhn_stationary_voltage` that integrates the system from rest and returns the final membrane voltage v, used as a steady‑state surrogate for the reaction part of the model.

2. **Helmholtz equation on a disk:**
   - A function `besselzero` that computes the first k positive zeros of the Bessel functions J_n or Y_n using Halley’s method with analytic initial guesses; higher zeros are obtained by linear extrapolation.
   - A function `helmholtz_exact` that evaluates the exact standing‑wave solution in Cartesian coordinates given disk radius, mode indices, angular coefficients, and a radial coefficient.

### periodic_solver.py
Implements direct solvers for periodic tridiagonal (circulant) systems stored in a specialised R83P format (3×n array including wrap‑around entries). The factorization (`r83p_fa`) uses a Schur‑complement approach on the leading non‑periodic (n‑1)×(n‑1) principal submatrix, solved by a non‑periodic tridiagonal factor‑solve pair (`r83_np_fa`, `r83_np_sl`). The solve routine (`r83p_sl`) supports both A and A^T systems. These routines are essential for efficiently computing the covariance matrix of the circular GMRF prior.

### spatial_domain.py
Spatial utilities:
- `annulus_grid` – structured polar grid in an annulus.
- `annulus_grid_fibonacci` – near‑uniform Fibonacci (golden‑angle) spiral points in an annulus.
- `fem_basis_2d` – evaluates a single Lagrange basis function of arbitrary degree on the reference triangle (0,0), (1,0), (0,1). The function uses product‑type formulas to enforce the Kronecker‑delta property at equally spaced nodes.
- Functions to compute exact mean and variance of the distance between two random points on a unit circle, and a Monte Carlo estimator for the same statistics.

### surrogate.py
Builds a 1D polynomial surrogate for the FHN stationary voltage as a function of the stimulus parameter d, for fixed (a, b). It uses a least‑squares polynomial fit (with SVD‑thresholded pseudoinverse for robustness) over a set of training points obtained by evaluating the full forward model. The construction returns a dictionary containing the polynomial coefficients and validity bounds. The `surrogate_predict` function evaluates the surrogate with input clamping.

### prng_asa183.py
Provides two pseudorandom number generators used throughout the project:

- **WichmannHill** (Algorithm AS 183) – combines three LCGs with a cycle length of ~6.95e12. Provides methods to generate single or arrays of uniform [0,1] variates and standard normal variates via the Box‑Muller transform.
- **LEcuyer** – a 32‑bit combined generator with extremely long cycle (~2.3e18), offering the same interface (uniform, uniforms, normal, normals).

Both classes are used as random number sources for MCMC, quadrature, and data generation.

### utils.py
Miscellaneous tools:
- **Latin hypercube edge sampling** – produces stratified samples in the unit hypercube using random permutations.
- **Julian day ↔ NYT publication calendar** – converts a Julian Ephemeris Date to a (volume, issue) pair for The New York Times (including historical corrections). Also provides the inverse conversion via bisection.
- **Permutation generation** – lexicographic successor (`perm_lex_next`) and a function to generate the next unicycle (single n‑cycle) permutation starting with 1 (`unicycle_next`).
- **Circle distance utilities** – uniform sampling on the unit circle, chord distance computation, and a Monte Carlo estimator for mean/variance of the distance between two random circle points.

---

## Core Algorithms & Module Interactions

The pipeline in `inference_engine.py` ties everything together. The GMRF prior is constructed as a periodic tridiagonal precision matrix Q; its exact covariance is obtained by solving Q Σ_{·,j} = e_j for each column using the `r83p` solver. This covariance trace is routinely compared against a direct NumPy inversion for verification.

The polynomial surrogate (in `surrogate.py`) is trained offline around the true parameters to accelerate likelihood evaluations
