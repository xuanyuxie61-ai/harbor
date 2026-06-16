"""
kernel_methods.py
=================

Neural tangent kernel (NTK) methods for matrix-element interpolation
and K-factor estimation:
- Infinite-width NTK kernel for ReLU networks (from 1027_anonymousDB99)

Scientific context:
-------------------
NLO QCD corrections to the 2->3 cross-section are captured by the K-factor:

    K = sigma_NLO / sigma_LO

Computing K point-by-point in the 5-dimensional phase space is expensive.
We use a kernel regression approach based on the infinite-width NTK:

    K_NTK(x, x') = kappa_1(x . x' / (|x| |x'|)) * |x| |x'|

where kappa_1 is the ReLU activation kernel:
    kappa_1(u) = u * (1 - acos(u)/pi) + sqrt(1-u^2) / pi

This kernel arises from the arc-cosine kernel of Cho & Saul (2009) and
captures the angular structure of the matrix element naturally.

The kernel regression estimate of the K-factor at a new phase-space point x is:

    K_hat(x) = sum_i alpha_i * K_NTK(x, x_i)

where alpha = (K + lambda*I)^{-1} y are the regression coefficients
computed from training data (x_i, K_i).
"""

import math
from typing import Callable, List, Optional, Tuple


# ===========================================================================
# NTK kernel computation (from 1027_anonymousDB99_LatentSpaceDistillation)
# ===========================================================================

def _kappa_0(u: float) -> float:
    """Arc-cosine kernel of degree 0 (Cho & Saul 2009):
        kappa_0(u) = 1 - acos(u) / pi
    for u in [-1, 1].
    """
    u = max(-1.0 + 1e-10, min(1.0 - 1e-10, u))
    return 1.0 - math.acos(u) / math.pi


def _kappa_1(u: float) -> float:
    """Arc-cosine kernel of degree 1:
        kappa_1(u) = u * kappa_0(u) + sqrt(1 - u^2) / pi
    """
    u = max(-1.0 + 1e-10, min(1.0 - 1e-10, u))
    return u * _kappa_0(u) + math.sqrt(1.0 - u * u) / math.pi


def ntk_relu_kernel(x: List[float], z: List[float], depth: int = 2,
                    bias: float = 0.0) -> Tuple[float, float]:
    """
    Compute the infinite-width NTK kernel for ReLU networks at a single
    pair of points x, z on the unit sphere (or approximately so).

    The recursion is:
        S_0 = x . z + bias^2
        N_0 = S_0 + bias^2
        For k = 1, ..., depth-1:
            u = S_{k-1} / (|x| |z|)
            S_k = |x| |z| * kappa_1(u)
            N_k = N_{k-1} * kappa_0(u) + S_k + bias^2

    Returns (N, S) where N is the NTK kernel value and S is the
    NNGP kernel value.

    Reference: Lee et al., "Wide Neural Networks of Any Depth Evolve as
    Linear Models Under Gradient Descent", NeurIPS 2019.
    """
    eps = 1e-6
    norm_x = math.sqrt(sum(xi * xi for xi in x) + eps)
    norm_z = math.sqrt(sum(zi * zi for zi in z) + eps)
    dot = sum(xi * zi for xi, zi in zip(x, z))
    S = dot + bias * bias
    N = S + bias * bias
    for k in range(1, depth):
        u = dot / (norm_x * norm_z + eps)
        u = max(-1.0 + eps, min(1.0 - eps, u))
        S = norm_x * norm_z * _kappa_1(u)
        N = N * _kappa_0(u) + S + bias * bias
    return N, S


def ntk_kernel_matrix(X: List[List[float]], Z: Optional[List[List[float]]] = None,
                      depth: int = 2, bias: float = 0.0) -> List[List[float]]:
    """
    Compute the NTK kernel matrix between point sets X and Z.

    K[i, j] = NTK(X[i], Z[j])
    """
    if Z is None:
        Z = X
    n = len(X)
    m = len(Z)
    K = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            N, _ = ntk_relu_kernel(X[i], Z[j], depth, bias)
            K[i][j] = N
    return K


def solve_linear_system(A: List[List[float]], b: List[float],
                        reg: float = 1e-6) -> List[float]:
    """
    Solve (A + reg*I) x = b using Gaussian elimination with partial pivoting.

    Used for kernel regression: x = (K + lambda*I)^{-1} y.
    """
    n = len(A)
    if n != len(b):
        raise ValueError("solve_linear_system: dimension mismatch")
    # Augmented matrix
    M = [list(A[i]) + [b[i]] for i in range(n)]
    # Add regularization
    for i in range(n):
        M[i][i] += reg
    # Forward elimination with partial pivoting
    for col in range(n):
        # Find pivot
        max_val = abs(M[col][col])
        max_row = col
        for row in range(col + 1, n):
            if abs(M[row][col]) > max_val:
                max_val = abs(M[row][col])
                max_row = row
        M[col], M[max_row] = M[max_row], M[col]
        if abs(M[col][col]) < 1e-14:
            M[col][col] = 1e-14
        # Eliminate below
        for row in range(col + 1, n):
            factor = M[row][col] / M[col][col]
            for j in range(col, n + 1):
                M[row][j] -= factor * M[col][j]
    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = M[i][n]
        for j in range(i + 1, n):
            x[i] -= M[i][j] * x[j]
        x[i] /= M[i][i]
    return x


def kernel_regression_predict(x_train: List[List[float]],
                              y_train: List[float],
                              x_test: List[List[float]],
                              depth: int = 2, reg: float = 1e-4
                              ) -> List[float]:
    """
    Predict y_test using NTK kernel regression:

        alpha = (K_train + lambda*I)^{-1} y_train
        y_test[i] = sum_j alpha[j] * K(x_test[i], x_train[j])

    Parameters
    ----------
    x_train : list of lists
        Training input points.
    y_train : list of float
        Training targets (e.g., K-factors).
    x_test : list of lists
        Test input points.
    depth : int
        NTK depth parameter.
    reg : float
        Regularization parameter lambda.

    Returns
    -------
    list of float: predicted values at test points.
    """
    K_train = ntk_kernel_matrix(x_train, depth=depth)
    alpha = solve_linear_system(K_train, y_train, reg)
    K_test = ntk_kernel_matrix(x_test, x_train, depth=depth)
    y_pred = []
    for i in range(len(x_test)):
        pred = sum(K_test[i][j] * alpha[j] for j in range(len(x_train)))
        y_pred.append(pred)
    return y_pred


def k_factor_from_matrix_element(me_sq: float, me_sq_born: float) -> float:
    """
    Compute the K-factor from squared matrix elements:

        K = |M_NLO|^2 / |M_LO|^2

    with infrared safety cuts.
    """
    if me_sq_born < 1e-20:
        return 1.0
    return max(0.1, min(me_sq / me_sq_born, 10.0))
