# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# Quantum Error Correction Threshold Analysis — Project Description

This project implements a numerical pipeline for estimating the threshold of Kitaev’s surface code under spatially correlated, temporally non‑Markovian Pauli noise. The main entry point is `main.py`, which orchestrates the full workflow: code construction, noise modelling, open‑system dynamics, syndrome decoding, threshold edge detection, sparse‑grid integration, rare‑event sampling and finite‑size scaling.

The remaining Python files must be re‑implemented by the agent. The following sections describe each file’s role, its main classes and functions, and their expected behaviour. No implementation details or exact constants are given; the descriptions provide the architectural knowledge needed to write the missing code.

---

## 1. `utils.py` – Utility Functions for Quantum Information

This module provides foundational mathematical and quantum‑information routines that are used everywhere else.

**Key functions (exact signatures may vary):**

- `pauli_operators()` – returns the four single‑qubit Pauli matrices (I, X, Y, Z) as complex arrays.
- `depolarizing_channel(p, n_qubits=1)` – builds the superoperator matrix for an n‑qubit depolarizing channel.
- `von_neumann_entropy(rho)` – computes the von Neumann entropy of a density matrix `rho` (with numerical clipping).
- `fidelity(rho, sigma)` – computes Uhlmann fidelity between two density matrices.
- `chop_array(arr, tol=1e-14)` – sets near‑zero entries of a complex/real array to zero.
- `symplectic_inner_product(a, b)` – symplectic inner product over GF(2) for binary vectors representing Pauli operators.
- `hamming_weight(v)` – Hamming weight of a binary vector.
- `binary_gaussian_elimination(M)` – reduced row echelon form of a binary matrix over GF(2); returns the RREF, rank, and pivot columns.
- `stabilizer_centralizer(S)` – computes a basis for the centralizer of a binary symplectic stabilizer matrix.

All functions must be numerically stable and support both real and complex inputs where appropriate.

---

## 2. `stabilizer_surface_code.py` – Surface Code Construction and Analysis

Implements Kitaev’s surface code on an \( L\times L \) lattice with either toric (periodic) or planar (open) boundary conditions.

**Main class: `SurfaceCode`**

Constructor accepts `L` and `boundary` (string `"toric"` or planar). Internally it builds:

- Attributes: `n_qubits`, `n_stabilizers`, `n_logical`, `distance`, and the stabilizer matrices `Hx` and `Hz` (binary).
- The edge indexing scheme and stabilizer assignments (star/plaquette operators) must be consistent with standard surface code construction.

**Methods:**

- `get_parity_check_matrix()` – returns a combined binary symplectic matrix of shape \((m, 2n)\) (X‑stabilizers in the first half, Z‑stabilizers in the second).
- `convert_to_crs(H)` – converts a dense binary matrix to Compressed Row Storage (values, column indices, row pointers).
- `sparse_parity_check(which)` – returns a scipy CSR matrix for `'x'` or `'z'` stabilizers.
- `boundary_word_topology()` – returns a string encoding the cyclic boundary structure.
- `box_distance_logical_operators()` – returns a dict with minimum box distances for logical operators (e.g., `dx`, `dz`, `mean_box`).
- `syndrome_of_error(error_vec)` – computes the syndrome of a binary error vector (length \(2n\)) mod 2.
- `logical_error_indicator(recovery, error)` – returns a binary array indicating which logical qubits are flipped after recovery.
- `compute_code_distance_brute_force()` – for small codes, enumerates logical operators to compute the exact code distance.

The class must handle both toric and planar layout correctly, including the counts of qubits, stabilizers, and logical qubits.

---

## 3. `noise_correlation.py` – Correlated and Non‑Markovian Noise Models

Provides spatially correlated Pauli error models and temporally correlated non‑Markovian extensions.

**Class `CorrelatedPauliNoise`**

Constructor takes: `n_qubits`, `base_rate`, `sigma`, `correlation_length`, `nu`, and an optional random generator `rng`.

Internally it builds a covariance matrix using a Matérn‑type correlation function (exponential for \(\nu=0.5\), general Matérn otherwise). The covariance is ensured to be positive semidefinite.

**Methods:**

- `sample_rates_cholesky()` – returns an array of per‑qubit error probabilities via Cholesky factorisation of the covariance, clipped to \([0,1]\).
- `sample_rates_eigen()` – same via eigenvalue decomposition.
- `sample_rates_fft(n_periodic=None)` – same via circulant embedding and FFT.
- `sample_error_instance(rates, error_type)` – generates a binary error vector (X and Z parts) for a given per‑qubit rate and error model (`"depolarizing"`, `"bitflip"`, `"phaseflip"`).
- `covariance_to_correlation()` – converts the covariance matrix to a correlation matrix.
- `generate_brc_like_data(n_samples)` – returns synthetic noise data of shape `(n_samples, n_qubits)` with Gaussian perturbations.

**Class `NonMarkovianNoise`** (inherits from `CorrelatedPauliNoise`)

Additional parameter: `memory_lambda`. Implements a quasi‑Markov temporal memory: at each time step the error is either refreshed or copied from the previous step with probability \(\lambda\). 

**Method:**

- `sample_temporal_sequence(n_steps, error_type)` – returns an array of shape `(n_steps, 2*n_qubits)`.

---

## 4. `lindblad_dynamics.py` – Open‑System Dynamics and DG Solver

Contains routines for Lindblad master equation simulation and a Discontinuous Galerkin solver for error probability density on a 1D chain.

**Functions:**

- `lindbladian_superoperator(H, jump_ops, hbar=1.0)` – builds the vectorised Liouvillian superoperator for a given Hamiltonian and a list of jump operators, returning a square complex matrix of size \(d^2 \times d^2\).
- `forward_euler_rho(rho0, L, t_final, n_steps)` – solves \(d\vec{\rho}/dt = L\vec{\rho}\) with forward Euler and re‑normalises trace each step.
- `exact_lindblad_evolution(rho0, L, t)` – uses matrix exponential to compute the exact evolution at time \(t\).

**Class `DGLindbladSolver`**

Represents a nodal Discontinuous Galerkin spectral solver on a 1D domain with Legendre‑Gauss‑Lobatto nodes. 

Constructor parameters: `n_elements`, `poly_order`, `domain=(0,1)`. Internally it builds:
- global node arrays and the local differentiation matrix,
- geometric factors (Jacobian, metric),
- surface lift operators.

**Methods:**

- `rhs(u, v, D, gamma)` – computes the right‑hand side of the advection‑diffusion‑reaction PDE \(\partial_t u = -v\,\partial_x u + D\,\partial_{xx} u + \gamma(1-2u)\) with upwind advection and central diffusion fluxes.
- `evolve(u0, t_final, n_steps, v, D, gamma)` – time‑steps using a low‑storage explicit Runge‑Kutta method (5 stages, 4th order), clipping the solution to \([0,1]\).

---

## 5. `syndrome_decoder.py` – Syndrome Decoders for Surface Codes

Implements several decoding algorithms that recover the error from a syndrome.

**Class `MWPMBruteDecoder`**

Given a binary parity‑check matrix `H`, its `decode(syndrome)` method enumerates all error patterns (using Gray code) and returns the one with minimum Hamming weight that matches the
