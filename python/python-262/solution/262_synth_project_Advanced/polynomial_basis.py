"""
polynomial_basis.py
===================
High-order polynomial basis functions for spectral reconstruction in
the finite-difference discretization of the MHD induction equation.

Maps seed projects:
  - 990_r8poly: polynomial evaluation (Horner), Chebyshev coefficients,
    Lagrange interpolation, derivative/integral coefficient transforms
  - 777_monomial_value: multivariate monomial evaluation
  - 785_naca: polynomial thickness distribution (NACA 4-digit)

In the reconnection context, these polynomials serve three purposes:
  1. High-order reconstruction of B, J, v at cell interfaces (WENO-like)
  2. Compact finite difference coefficient computation
  3. Analytic representation of the current sheet profile

Key equations:
  - Lagrange basis: L_j(x) = prod_{k!=j} (x - x_k) / (x_j - x_k)
  - Chebyshev recurrence: T_0=1, T_1=x, T_{n+1} = 2x T_n - T_{n-1}
  - Derivative of Chebyshev expansion: c'_k via backward recurrence
  - NACA thickness: y_t(x) = 5t[c0 sqrt(x) + c1 x + c2 x^2 + c3 x^3 + c4 x^4]
"""

import numpy as np
from numpy.polynomial import chebyshev as ch


# ============================================================
# Lagrange Interpolation and Derivatives
# ============================================================

def lagrange_nodes_1d(n, a=-1.0, b=1.0):
    """
    Equally spaced interpolation nodes on [a, b].
    n: number of nodes (polynomial degree = n-1)
    """
    return np.linspace(a, b, n)


def chebyshev_nodes_1d(n, a=-1.0, b=1.0):
    """
    Chebyshev-Gauss-Lobatto nodes on [a, b]:
        x_k = -cos(pi * k / (n-1)),  k = 0, ..., n-1
    Mapped from [-1, 1] to [a, b].
    """
    k = np.arange(n)
    x_ref = -np.cos(np.pi * k / max(n - 1, 1))
    return 0.5 * (b - a) * x_ref + 0.5 * (a + b)


def lagrange_basis_values(x_eval, x_nodes):
    """
    Evaluate all Lagrange basis functions L_j(x) at points x_eval.

    Input:
        x_eval: array of evaluation points, shape (M,)
        x_nodes: array of n nodes, shape (n,)

    Output:
        L: shape (M, n), L[i, j] = L_j(x_eval[i])
    """
    n = len(x_nodes)
    M = len(x_eval)
    L = np.ones((M, n))
    for j in range(n):
        for k in range(n):
            if k != j:
                denom = x_nodes[j] - x_nodes[k]
                if abs(denom) < 1e-300:
                    denom = 1e-300  # prevent division by zero
                L[:, j] *= (x_eval - x_nodes[k]) / denom
    return L


def lagrange_derivative_matrix(x_nodes):
    """
    Compute the derivative matrix D for polynomial interpolation at nodes x_nodes.
    D[i, j] = L'_j(x_i), so that (df/dx)|_{x_i} ~ sum_j D[i,j] f(x_j).

    Uses the standard formula:
        For i != j:  D_{ij} = (w_j / w_i) / (x_i - x_j)
        For i == j:  D_{ii} = sum_{k!=i} 1 / (x_i - x_k)

    where w_j = prod_{k!=j} (x_j - x_k) are the barycentric weights.

    Output:
        D: shape (n, n), the spectral differentiation matrix
    """
    n = len(x_nodes)
    D = np.zeros((n, n))

    # Barycentric weights
    w = np.ones(n)
    for j in range(n):
        for k in range(n):
            if k != j:
                w[j] *= (x_nodes[j] - x_nodes[k])

    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = (w[j] / w[i]) / (x_nodes[i] - x_nodes[j])
            else:
                s = 0.0
                for k in range(n):
                    if k != i:
                        s += 1.0 / (x_nodes[i] - x_nodes[k])
                D[i, i] = s
    return D


# ============================================================
# Chebyshev Polynomial Tools
# ============================================================

def chebyshev_coefficients(n):
    """
    Return the coefficient array for the n-th Chebyshev polynomial T_n(x)
    in the power basis {1, x, x^2, ..., x^n}.

    Uses the recurrence:
        T_0 = 1, T_1 = x
        T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x)
    """
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return np.array([0.0, 1.0])

    T_prev2 = np.array([1.0])  # T_0
    T_prev1 = np.array([0.0, 1.0])  # T_1

    for _ in range(2, n + 1):
        # T_k = 2x * T_{k-1} - T_{k-2}
        # Multiply T_{k-1} by 2x: shift coefficients up by 1 and multiply by 2
        T_new = np.zeros(len(T_prev1) + 1)
        T_new[1:] += 2.0 * T_prev1
        # Subtract T_{k-2}
        T_new[:len(T_prev2)] -= T_prev2
        T_prev2 = T_prev1.copy()
        T_prev1 = T_new.copy()

    return T_prev1


def chebyshev_evaluate(coeffs_cheb, x):
    """
    Evaluate a Chebyshev expansion sum_k c_k T_k(x) using Clenshaw's algorithm.

    Input:
        coeffs_cheb: array of Chebyshev coefficients [c_0, c_1, ..., c_N]
        x: evaluation points (must be in [-1, 1])

    Output:
        values at x
    """
    N = len(coeffs_cheb) - 1
    if N < 0:
        return np.zeros_like(x)

    b_kp2 = np.zeros_like(x)
    b_kp1 = np.zeros_like(x)

    for k in range(N, 0, -1):
        b_k = coeffs_cheb[k] + 2.0 * x * b_kp1 - b_kp2
        b_kp2 = b_kp1
        b_kp1 = b_k

    return coeffs_cheb[0] + x * b_kp1 - b_kp2


def chebyshev_derivative_coefficients(coeffs_cheb):
    """
    Given Chebyshev coefficients c_k of f(x), compute coefficients c'_k
    of f'(x) using the backward recurrence:
        c'_N = 0, c'_{N-1} = 2N c_N
        c'_k = c'_{k+2} + 2(k+1) c_{k+1},  k = N-2, ..., 0
    with c'_0 halved at the end.
    """
    N = len(coeffs_cheb) - 1
    if N <= 0:
        return np.array([0.0])

    dc = np.zeros(N + 1)
    dc[N - 1] = 2.0 * N * coeffs_cheb[N]
    for k in range(N - 2, -1, -1):
        dc[k] = dc[k + 2] + 2.0 * (k + 1) * coeffs_cheb[k + 1]
    dc[0] *= 0.5
    return dc


# ============================================================
# Monomial Evaluation (maps to 777_monomial_value)
# ============================================================

def monomial_value_nd(exponents, x):
    """
    Evaluate a multivariate monomial:
        f(x) = prod_j x_j^{alpha_j}

    where exponents = [alpha_1, ..., alpha_d].

    Handles 0^0 = 1 by convention.

    Input:
        exponents: array of non-negative integers, shape (d,)
        x: evaluation points, shape (N, d)
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)
    N, d = x.shape
    value = np.ones(N)
    for j in range(d):
        if exponents[j] != 0:
            val = x[:, j]
            # Handle 0^0: if val==0 and exp==0, result should be 1
            with np.errstate(invalid='ignore', divide='ignore'):
                value *= np.where(
                    (np.abs(val) < 1e-300) & (exponents[j] == 0),
                    1.0,
                    np.power(np.abs(val) + 1e-300, exponents[j])
                    * np.sign(val) ** (exponents[j] % 2)
                )
    return value


# ============================================================
# NACA-inspired Current Sheet Profile (maps to 785_naca)
# ============================================================

def naca_thickness_profile(t_frac, x_cs, L_cs=1.0):
    """
    Generate a current sheet thickness profile inspired by the NACA
    4-digit symmetric airfoil thickness distribution:

        y(x) = 5 t * L * (a0 sqrt(x/L) + a1 (x/L) + a2 (x/L)^2
                            + a3 (x/L)^3 + a4 (x/L)^4)

    with standard NACA coefficients:
        a0 = 0.2969, a1 = -0.1260, a2 = -0.3516, a3 = 0.2843, a4 = -0.1015

    This provides a smooth, physically motivated profile for the current
    sheet boundary layer shape.

    Input:
        t_frac: maximum relative thickness (0 < t < 1)
        x_cs: positions along the sheet (0 <= x <= L_cs)
        L_cs: chord length (current sheet length)
    """
    xi = np.clip(x_cs / max(L_cs, 1e-30), 0.0, 1.0)
    y = 5.0 * t_frac * L_cs * (
        0.2969 * np.sqrt(xi)
        - 0.1260 * xi
        - 0.3516 * xi ** 2
        + 0.2843 * xi ** 3
        - 0.1015 * xi ** 4
    )
    return y


def naca_profile_derivative(t_frac, x_cs, L_cs=1.0):
    """
    Analytical derivative dy/dx of the NACA thickness profile.
    Used for computing the local slope of the current sheet boundary.
    """
    xi = np.clip(x_cs / max(L_cs, 1e-30), 1e-30, 1.0)
    dydx = 5.0 * t_frac * (
        0.2969 * 0.5 / np.sqrt(xi)
        - 0.1260
        - 2.0 * 0.3516 * xi
        + 3.0 * 0.2843 * xi ** 2
        - 4.0 * 0.1015 * xi ** 3
    ) / L_cs
    return dydx


# ============================================================
# Polynomial Reconstruction for High-Order FD
# ============================================================

def polynomial_reconstruct(stencil_values, x_nodes, x_target):
    """
    Reconstruct the function value at x_target using polynomial
    interpolation through the stencil points.

    This is the core of high-order finite difference methods: given
    function values at n stencil nodes, evaluate at an arbitrary point.

    Input:
        stencil_values: function values at stencil nodes, shape (n,)
        x_nodes: stencil node positions, shape (n,)
        x_target: target position(s)

    Output:
        reconstructed value(s)
    """
    x_t = np.atleast_1d(x_target)
    L = lagrange_basis_values(x_t, x_nodes)
    return L @ stencil_values


def compute_fd_weights(x_nodes, x_eval, derivative_order=0):
    """
    Compute finite difference weights for arbitrary node placement
    and derivative order using the Fornberg (1988) algorithm.

    Given nodes x_0, ..., x_n and evaluation point x_eval,
    compute weights w_j such that:
        f^(m)(x_eval) ~ sum_j w_j f(x_j)

    Reference:
        B. Fornberg, "Generation of Finite Difference Formulas on
        Arbitrarily Spaced Grids", Math. Comp. 51 (1988), 699-706.

    Input:
        x_nodes: array of n+1 node positions
        x_eval: evaluation point
        derivative_order: m (0=value, 1=first derivative, 2=second, ...)

    Output:
        weights: array of shape (n+1,)
    """
    n = len(x_nodes) - 1
    m = derivative_order

    # Fornberg's algorithm: c[j, k] stores the weight for node j,
    # derivative order k, using nodes 0..i
    c = np.zeros((n + 1, m + 1))
    c[0, 0] = 1.0
    c1_old = 1.0

    for i in range(1, n + 1):
        c2_new = 1.0
        for j in range(i):
            c3 = x_nodes[i] - x_nodes[j]
            c2_new *= c3
            if i <= m:
                c[i, i] = 0.0
            for k in range(min(i, m), 0, -1):
                c[i, k] = (c1_old * (
                    (x_nodes[i] - x_eval) * c[i - 1, k]
                    - k * c[i - 1, k - 1]
                )) / c2_new if abs(c2_new) > 1e-300 else 0.0
            c[i, 0] = c1_old * (x_nodes[i] - x_eval) * c[i - 1, 0] / (
                c2_new if abs(c2_new) > 1e-300 else 1e-300)

        for j in range(i):
            for k in range(min(i, m), 0, -1):
                c[j, k] = (
                    (x_eval - x_nodes[i]) * c[j, k]
                    + k * c[j, k - 1]
                ) / (c2_new if abs(c2_new) > 1e-300 else 1e-300)
            c[j, 0] = (x_eval - x_nodes[i]) * c[j, 0] / (
                c2_new if abs(c2_new) > 1e-300 else 1e-300)

        c1_old = c2_new

    return c[:, m]


# ============================================================
# High-Order Central Difference Stencils
# ============================================================

def central_fd4_weights():
    """
    4th-order central difference weights for 1st and 2nd derivatives.

    1st derivative (5-point stencil):
        f'_i ~ (-f_{i+2} + 8 f_{i+1} - 8 f_{i-1} + f_{i-2}) / (12 dx)

    2nd derivative (5-point stencil):
        f''_i ~ (-f_{i+2} + 16 f_{i+1} - 30 f_i + 16 f_{i-1} - f_{i-2}) / (12 dx^2)
    """
    d1 = np.array([1.0, -8.0, 0.0, 8.0, -1.0]) / 12.0
    d2 = np.array([-1.0, 16.0, -30.0, 16.0, -1.0]) / 12.0
    return d1, d2


def central_fd6_weights():
    """
    6th-order central difference weights (7-point stencil).

    1st derivative:
        f'_i ~ (f_{i+3} - 9 f_{i+2} + 45 f_{i+1} - 45 f_{i-1}
                + 9 f_{i-2} - f_{i-3}) / (60 dx)

    2nd derivative:
        f''_i ~ (2 f_{i+3} - 27 f_{i+2} + 270 f_{i+1} - 490 f_i
                 + 270 f_{i-1} - 27 f_{i-2} + 2 f_{i-3}) / (180 dx^2)
    """
    d1 = np.array([-1.0, 9.0, -45.0, 0.0, 45.0, -9.0, 1.0]) / 60.0
    d2 = np.array([2.0, -27.0, 270.0, -490.0, 270.0, -27.0, 2.0]) / 180.0
    return d1, d2


def compact_fd4_weights():
    """
    4th-order compact (Padé) finite difference for 1st derivative.

    Implicit scheme:
        (1/6) f'_{i-1} + (2/3) f'_i + (1/6) f'_{i+1}
            = (f_{i+1} - f_{i-1}) / (2 dx)

    This is a tridiagonal system: A f' = b
    where A has [1/6, 2/3, 1/6] on each row.
    """
    a_lower = 1.0 / 6.0
    a_diag = 2.0 / 3.0
    a_upper = 1.0 / 6.0
    rhs_left = -1.0 / 2.0
    rhs_center = 0.0
    rhs_right = 1.0 / 2.0
    return a_lower, a_diag, a_upper, rhs_left, rhs_center, rhs_right
