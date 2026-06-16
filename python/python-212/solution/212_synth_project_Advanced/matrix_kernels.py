"""
matrix_kernels.py
=================
Structured matrix assembly for the KKT saddle-point system.

Matrices provided
-----------------
1. Hilbert matrix  H_{ij} = 1 / (i + j - 1)        (from 738_matrix_assemble_parfor)
   Used as a prototype ill-conditioned kernel for the background-error
   covariance regularisation.

2. Block KKT matrix assembly
   [ A    0    B^T  c  ] [ y ]   [ f     ]
   [ 0    A^T  -I   0  ] [ p ] = [ y_d   ]
   [ B   -I   -alpha*M 0 ] [ u ]   [ u_b_s ]
   [ c^T  0    0    0  ] [lam]   [ E_max ]

   where  c = quadrature weight vector for the integral constraint.

3. Observation operator  H_obs  (sparse sampling matrix)

4. Background-error covariance  B_cov  and  observation-error covariance R_cov
   (from 4D-Var / BayRad3D).

KKT role
--------
These kernels build the KKT system matrix that the active-set solver
works with.  The Hilbert block controls the regularisation strength and
the conditioning of the whole system.
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# Hilbert matrix  (from 738 / Burkardt)
# ---------------------------------------------------------------------------

def hilbert_matrix(m: int, n: int | None = None) -> np.ndarray:
    """Assemble the (m, n) Hilbert matrix  H_{ij} = 1/(i+j-1).

    The Hilbert matrix is the classical example of an ill-conditioned
    totally-positive matrix;  cond_2(H_n) ~ O(exp(3.5 n)).
    We use it as a structured building block for the background-error
    covariance  B_cov = scale * H + eps * I.
    """
    if n is None:
        n = m
    i = np.arange(1, m + 1, dtype=np.float64)[:, None]
    j = np.arange(1, n + 1, dtype=np.float64)[None, :]
    return 1.0 / (i + j - 1.0)


def hilbert_inverse(n: int) -> np.ndarray:
    """Exact inverse of the n x n Hilbert matrix via the closed-form formula.

    (H^{-1})_{ij} = (-1)^{i+j} (i+j-1) C(n+i-1, n-j) C(n+j-1, n-i) C(i+j-2, i-1)^2
    """
    from math import comb, factorial
    Hinv = np.zeros((n, n), dtype=np.float64)
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            sgn = (-1) ** (i + j)
            term1 = (i + j - 1)
            term2 = comb(n + i - 1, n - j)
            term3 = comb(n + j - 1, n - i)
            term4 = comb(i + j - 2, i - 1) ** 2
            Hinv[i - 1, j - 1] = sgn * term1 * term2 * term3 * term4
    return Hinv


# ---------------------------------------------------------------------------
# Background-error and observation-error covariance matrices
# ---------------------------------------------------------------------------

def build_background_covariance(
    n: int,
    scale: float,
    correlation_length: float = 0.2,
) -> np.ndarray:
    """Build a structured SPD background-error covariance matrix.

    B = scale * (Hilbert(n) / max(Hilbert) + eps * I)

    The Hilbert kernel provides off-diagonal correlation that mimics
    spatially-correlated background errors in 4D-Var.
    """
    H = hilbert_matrix(n)
    H_norm = np.max(H)
    if H_norm < 1.0e-30:
        H_norm = 1.0
    B = scale * (H / H_norm)
    # Add diagonal jitter for SPD guarantee
    eps = 1.0e-3 * scale
    B += eps * np.eye(n)
    return B


def build_observation_covariance(n_obs: int, obs_var: float = 1.0e-2) -> np.ndarray:
    """Diagonal observation-error covariance  R = obs_var * I.

    In the full 4D-Var cost function, R^{-1} weights the observation
    misfit  J_o = 0.5 * (H y - y_obs)^T R^{-1} (H y - y_obs).
    """
    return obs_var * np.eye(n_obs)


# ---------------------------------------------------------------------------
# Observation operator
# ---------------------------------------------------------------------------

def build_observation_operator(n_total: int, obs_indices: np.ndarray) -> np.ndarray:
    """Build a sparse observation operator  H_obs  (n_obs, n_total).

    H_obs[i, j] = 1  if j == obs_indices[i], else 0.

    This selects the state components that are observed.
    """
    obs_indices = np.asarray(obs_indices, dtype=np.int64).ravel()
    n_obs = obs_indices.size
    H = np.zeros((n_obs, n_total), dtype=np.float64)
    for i, j in enumerate(obs_indices):
        if 0 <= j < n_total:
            H[i, j] = 1.0
    return H


# ---------------------------------------------------------------------------
# Block KKT matrix assembly
# ---------------------------------------------------------------------------

def assemble_kkt_blocks(
    A: np.ndarray,
    M_diag: np.ndarray,
    B_cov_inv: np.ndarray,
    c_vec: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Assemble the full KKT saddle-point matrix.

    Block structure (3 * N^2 + 1) x (3 * N^2 + 1):

    K = [  A       0       0       c  ]
        [  0       A^T    -M_diag  0  ]
        [  0      -M_diag  H_reg   0  ]
        [  c^T     0       0       0  ]

    where  H_reg = alpha * M_diag + B_cov_inv  is the control Hessian block.

    The sign conventions follow the standard Lagrangian formulation:
        L = J(y,u) + p^T (f - A y + u) + lam * (c^T u - E_max)
    """
    n2 = A.shape[0]
    n_total = 3 * n2 + 1

    K = np.zeros((n_total, n_total), dtype=np.float64)

    # (1,1) block: A
    K[:n2, :n2] = A
    # (2,2) block: A^T
    K[n2: 2 * n2, n2: 2 * n2] = A.T
    # (2,3) block: -M_diag
    M_mat = np.diag(M_diag)
    K[n2: 2 * n2, 2 * n2: 3 * n2] = -M_mat
    # (3,2) block: -M_diag
    K[2 * n2: 3 * n2, n2: 2 * n2] = -M_mat
    # (3,3) block: H_reg = alpha * M_diag + B_cov_inv
    H_reg = alpha * np.diag(M_diag) + B_cov_inv
    K[2 * n2: 3 * n2, 2 * n2: 3 * n2] = H_reg
    # (1,4) and (4,1) blocks: c
    K[:n2, -1] = c_vec
    K[-1, :n2] = c_vec

    return K


def assemble_kkt_rhs(
    f_vec: np.ndarray,
    y_d_weighted: np.ndarray,
    u_b_weighted: np.ndarray,
    E_max: float,
    n2: int,
) -> np.ndarray:
    """Assemble the KKT right-hand side.

    rhs = [ f           ]
          [ y_d_weighted ]
          [ u_b_weighted ]
          [ E_max        ]
    """
    rhs = np.zeros(3 * n2 + 1, dtype=np.float64)
    rhs[:n2] = f_vec
    rhs[n2: 2 * n2] = y_d_weighted
    rhs[2 * n2: 3 * n2] = u_b_weighted
    rhs[-1] = E_max
    return rhs


# ---------------------------------------------------------------------------
# Quantum-inspired structured test matrix  (from project 1040)
# ---------------------------------------------------------------------------

def modular_multiplication_matrix(a: int, N: int) -> np.ndarray:
    """Unitary-like permutation matrix for  x -> (a*x) mod N.

    This is the core building block of Shor's algorithm (period finding).
    We use it to construct structured ill-conditioned test matrices for
    the KKT solver benchmarking.
    """
    dim = max(N, 2)
    U = np.zeros((dim, dim), dtype=np.float64)
    for i in range(N):
        j = (a * i) % N
        U[j, i] = 1.0
    # Pad remaining dimensions with identity
    for i in range(N, dim):
        U[i, i] = 1.0
    return U
