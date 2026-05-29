"""
Benchmark Problem Suite for Sparse Linear Algebra Optimization.

Integrates:
  - 702_logistic_ode: time-dependent reaction-diffusion test
  - 422_feynman_kac_1d: stochastic PDE reference solution
  - 171_chirikov_iteration: chaotic sensitivity matrix generation
  - 751_menger_sponge_chaos: hierarchical self-similar sparse matrices
  - 136_candy_count: structured circulant/Toeplitz sparse patterns

Scientific formulas:
  1. Logistic reaction-diffusion:
       du/dt = D * nabla^2 u + r * u * (1 - u/K)
     Steady-state: -D * nabla^2 u = r * u * (1 - u/K)
     Linearized about u=0: -D * nabla^2 u - r * u = f

  2. Feynman-Kac reference:
       (1/2) * d^2U/dX^2 - V(X)*U = 0,  U(boundary)=1
       V(X) = 2*(X/a^2)^2 + 1/a^2
       Exact solution: U(X) = exp((X/a)^2 - 1)

  3. Chirikov standard map Jacobian:
       J = [[1, k*cos(x)], [1, 1+k*cos(x)]]
     Ensemble average Jacobian is used as test matrix.

  4. Menger sponge hierarchical matrix:
     Self-similar structure analogous to H-matrices.
     Block structure: A = [A_sub, B; C, A_sub] with recursive pattern.

  5. Candy-count circulant pattern:
     For n x n matrix with block size b:
     A_{i,j} = 1 if mod(i+j, b) == 0, else 0.
"""

import numpy as np
import math


def logistic_reaction_diffusion_matrix(n, D=1.0, r=1.0, K=1.0, domain=1.0):
    """
    Generate the stiffness matrix for the linearized logistic reaction-diffusion
    equation on a 1D uniform mesh:
      -D * u'' - r * u = f
    with Dirichlet boundaries u(0)=u(L)=0.

    Finite difference discretization (n interior points):
      h = L / (n+1)
      -D/h^2 * (u_{i-1} - 2*u_i + u_{i+1}) - r * u_i = f_i

    This yields a tridiagonal SPD matrix.
    """
    h = domain / (n + 1)
    main_diag = 2.0 * D / (h * h) - r
    off_diag = -D / (h * h)

    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = main_diag
        if i > 0:
            A[i, i - 1] = off_diag
        if i < n - 1:
            A[i, i + 1] = off_diag
    return A


def feynman_kac_potential(a, x):
    """
    Potential function from seed 422_feynman_kac_1d:
      V(X) = 2 * (X/a^2)^2 + 1/a^2
    """
    return 2.0 * (x / (a * a)) ** 2 + 1.0 / (a * a)


def feynman_kac_exact(a, x):
    """
    Exact solution: U(X) = exp((X/a)^2 - 1)
    """
    return np.exp((x / a) ** 2 - 1.0)


def feynman_kac_stochastic_solve(a=2.0, h=0.01, n_paths=5000, n_x=21):
    """
    Monte-Carlo solver for the 1D Feynman-Kac problem.
    Returns grid points and estimated solution values.
    Based on seed 422_feynman_kac_1d.
    """
    x_grid = np.linspace(-a, a, n_x)
    u_est = np.zeros(n_x)

    rng = np.random.default_rng(42)
    for ix, X0 in enumerate(x_grid):
        total = 0.0
        for _ in range(n_paths):
            X = X0
            integral_V = 0.0
            while abs(X) < a:
                # Euler-Maruyama step: dX = dW, dW ~ N(0, h)
                dW = rng.normal(0.0, math.sqrt(h))
                V_old = feynman_kac_potential(a, X)
                X_new = X + dW
                # Trapezoidal rule for integral
                V_new = feynman_kac_potential(a, X_new)
                integral_V += 0.5 * (V_old + V_new) * h
                X = X_new
            total += math.exp(-integral_V)
        u_est[ix] = total / n_paths

    return x_grid, u_est


def chirikov_map_jacobian_ensemble(n_ensemble=100, k=0.55, n_steps=50, seed=42):
    """
    Generate an ensemble-averaged Jacobian matrix from the Chirikov standard map.
    Map:  y' = y + k*sin(x)
          x' = x + y'
    Jacobian at step t:
          J_t = [[1, k*cos(x_t)], [1, 1+k*cos(x_t)]]
    Ensemble average over random initial conditions gives a 2x2 test matrix.
    For larger systems, we tile this structure.
    Based on seed 171_chirikov_iteration.
    """
    rng = np.random.default_rng(seed)
    J_sum = np.zeros((2, 2))
    for _ in range(n_ensemble):
        x = rng.random() * 2.0 * math.pi
        y = rng.random() * 2.0 * math.pi
        for _ in range(n_steps):
            J = np.array([[1.0, k * math.cos(x)],
                          [1.0, 1.0 + k * math.cos(x)]])
            J_sum += J
            y_new = y + k * math.sin(x)
            x_new = x + y_new
            x = x_new % (2.0 * math.pi)
            y = y_new % (2.0 * math.pi)
    J_avg = J_sum / n_ensemble
    return J_avg


def menger_sponge_hierarchical_matrix(level=3, epsilon=1e-3):
    """
    Construct a hierarchical sparse matrix inspired by the Menger sponge IFS.
    The matrix has a recursive block structure analogous to H-matrices:
      A^{(l)} = [ A^{(l-1)},  B^{(l-1)};
                  C^{(l-1)},  A^{(l-1)} ]
    where B and C are off-diagonal couplings with decaying magnitude.
    Based on seed 751_menger_sponge_chaos.

    For level l, matrix size n_l = 2^l.
    """
    if level == 0:
        return np.array([[1.0]])

    n_sub = 2 ** (level - 1)
    A_sub = menger_sponge_hierarchical_matrix(level - 1, epsilon)

    # Off-diagonal coupling decays with level
    scale = epsilon ** (level - 1)
    B = scale * np.random.default_rng(42 + level).random((n_sub, n_sub))
    C = scale * np.random.default_rng(43 + level).random((n_sub, n_sub))

    n = 2 * n_sub
    A = np.zeros((n, n))
    A[:n_sub, :n_sub] = A_sub
    A[:n_sub, n_sub:] = B
    A[n_sub:, :n_sub] = C
    A[n_sub:, n_sub:] = A_sub

    # Symmetrize for SPD property
    A = 0.5 * (A + A.T)
    # Add diagonal dominance
    for i in range(n):
        A[i, i] += 1.0 + np.sum(np.abs(A[i, :])) - abs(A[i, i])
    return A


def candy_circulant_matrix(n, block_size=5, value=1.0):
    """
    Generate a sparse matrix with circulant block pattern inspired by
    seed 136_candy_count.

    Pattern: A_{i,j} = value if mod(i+j, block_size) == 0, else small random.
    This produces a structured sparse pattern similar to block-cyclic
    distributions in parallel sparse linear algebra.
    """
    rng = np.random.default_rng(136)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if (i + j) % block_size == 0:
                A[i, j] = value
            else:
                A[i, j] = 0.01 * rng.random()
    # Make diagonally dominant
    for i in range(n):
        row_sum = np.sum(np.abs(A[i, :])) - abs(A[i, i])
        A[i, i] = 1.0 + row_sum
    return A


def generate_all_benchmark_matrices():
    """
    Generate a suite of benchmark matrices for the sparse solver framework.
    Returns dict of {name: (A, b)} pairs.
    """
    suite = {}

    # 1. Logistic reaction-diffusion (small)
    n1 = 49
    A1 = logistic_reaction_diffusion_matrix(n1, D=0.1, r=2.0, K=1.0)
    b1 = np.ones(n1)
    suite['logistic_reaction_diffusion'] = (A1, b1)

    # 2. Feynman-Kac deterministic finite-difference matrix
    a = 2.0
    n2 = 49
    h = 2.0 * a / (n2 + 1)
    x_grid = np.linspace(-a + h, a - h, n2)
    A2 = np.zeros((n2, n2))
    for i in range(n2):
        V_i = feynman_kac_potential(a, x_grid[i])
        A2[i, i] = 1.0 / (h * h) + V_i
        if i > 0:
            A2[i, i - 1] = -0.5 / (h * h)
        if i < n2 - 1:
            A2[i, i + 1] = -0.5 / (h * h)
    # Boundary conditions U(boundary)=1 contribute to RHS
    b2 = np.zeros(n2)
    b2[0] += 0.5 / (h * h)  # left boundary
    b2[-1] += 0.5 / (h * h)  # right boundary
    suite['feynman_kac_fd'] = (A2, b2)

    # 3. Chirikov ensemble-averaged tiled matrix
    J_avg = chirikov_map_jacobian_ensemble(n_ensemble=200, k=0.55, n_steps=30)
    n3 = 40
    A3 = np.zeros((n3, n3))
    for block in range(n3 // 2):
        base = 2 * block
        A3[base:base + 2, base:base + 2] = J_avg
    # Add coupling between blocks and diagonal dominance
    rng = np.random.default_rng(171)
    for i in range(n3):
        for j in range(n3):
            if i != j and abs(A3[i, j]) < 1e-15:
                A3[i, j] = 0.05 * rng.random()
    for i in range(n3):
        row_sum = np.sum(np.abs(A3[i, :])) - abs(A3[i, i])
        A3[i, i] = 2.0 + row_sum
    A3 = 0.5 * (A3 + A3.T)
    b3 = np.sin(np.linspace(0, 2 * math.pi, n3))
    suite['chirikov_tiled'] = (A3, b3)

    # 4. Menger sponge hierarchical matrix
    A4 = menger_sponge_hierarchical_matrix(level=4, epsilon=0.1)
    n4 = A4.shape[0]
    b4 = np.ones(n4)
    suite['menger_hierarchical'] = (A4, b4)

    # 5. Candy circulant pattern matrix
    A5 = candy_circulant_matrix(n=48, block_size=6, value=0.5)
    b5 = np.arange(48, dtype=float)
    suite['candy_circulant'] = (A5, b5)

    return suite
