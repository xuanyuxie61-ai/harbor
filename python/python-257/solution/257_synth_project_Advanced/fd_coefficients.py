"""
fd_coefficients.py
==================
High-order finite-difference weight computation on arbitrary node sets.

The core algorithm is Fornberg's algorithm (1988, 1998), which computes
weights for FD approximations of derivatives of arbitrary order on
arbitrarily-spaced nodes.

Mathematical background
-----------------------
Given N+1 distinct nodes x_0, ..., x_N and a derivative order alpha,
the weights c_0, ..., c_N are determined by requiring:
    d^alpha f / dx^alpha (x_bar) ~ sum_{j=0}^{N} c_j f(x_j)
to be exact for polynomials of degree <= N.

This yields the linear system:
    sum_{j=0}^{N} c_j x_j^k = alpha! delta_{k,alpha},  k = 0, ..., N
which is a Vandermonde-type system.  Fornberg's recursion is the
standard O(N^2) method for computing the weights without
explicitly forming the Vandermonde matrix.

On the sphere, the nodes are not collinear but lie on a curved manifold.
We therefore use *local tangent-plane* approximations:
  - For each central node x_c, the neighbours are projected onto
    the tangent plane at x_c via the exponential map.
  - Fornberg weights are computed in the tangent-plane coordinates.
"""

from __future__ import annotations
import math
from typing import List, Tuple, Dict


# ---------------------------------------------------------------------------
# Fornberg's algorithm: 1D finite difference weights
# ---------------------------------------------------------------------------
def fornberg_weights(x_bar: float,
                     nodes: List[float],
                     max_deriv: int) -> List[List[float]]:
    """
    Fornberg (1988) algorithm.

    Parameters
    ----------
    x_bar      : evaluation point
    nodes      : list of N+1 node coordinates (distinct)
    max_deriv  : maximum derivative order desired

    Returns
    -------
    weights[m][j] : weight for node j, derivative order m
    """
    N = len(nodes) - 1
    M = max_deriv
    # 2D array weights[m][j]
    weights = [[0.0] * (N + 1) for _ in range(M + 1)]
    weights[0][0] = 1.0
    c1 = 1.0
    for i in range(1, N + 1):
        c2 = 1.0
        for j in range(i):
            c3 = nodes[i] - nodes[j]
            if abs(c3) < 1.0e-30:
                raise ValueError(f"Fornberg: nodes {i} and {j} coincide.")
            c2 *= c3
            if i <= M:
                weights[i][j] = 0.0
            for m in range(min(i, M), 0, -1):
                weights[m][j] = ((nodes[i] - x_bar) * weights[m][j] - m * weights[m - 1][j]) / c3
            weights[0][j] = (nodes[i] - x_bar) * weights[0][j] / c3
        for m in range(1, M + 1):
            weights[m][i] = c1 / c2 * (m * weights[m - 1][i - 1] - (nodes[i - 1] - x_bar) * weights[m][i - 1])
        weights[0][i] = -c1 / c2 * (nodes[i - 1] - x_bar) * weights[0][i - 1]
        c1 = c2
    return weights


# ---------------------------------------------------------------------------
# Central-difference weights on a uniform stencil
# ---------------------------------------------------------------------------
def central_fd_weights(order: int, deriv: int) -> List[float]:
    """
    Return weights for a central FD approximation of derivative `deriv`
    on a uniform 2*order+1 point stencil.
    E.g. central_fd_weights(2, 2) -> [1, -2, 1]  for the standard 2nd derivative.
    """
    n = 2 * order + 1
    nodes = list(range(-order, order + 1))
    w = fornberg_weights(0, [float(x) for x in nodes], deriv)
    return w[deriv]


# ---------------------------------------------------------------------------
# High-order compact (Pade-type) finite differences
# ---------------------------------------------------------------------------
def compact_fd_weights_1d(N: int, alpha: float = 1.0/3.0) -> Tuple[List[float], List[float]]:
    """
    Implicit / compact FD scheme for 1st derivative:
        alpha f'_{i-1} + f'_i + alpha f'_{i+1}
            = a (f_{i+1} - f_{i-1}) / (2h)
              + b (f_{i+2} - f_{i-2}) / (4h)
              + c (f_{i+3} - f_{i-3}) / (6h)
    With alpha = 1/3, a = 14/9, b = 1/9, c = 0  -> 4th-order accurate.
    Returns (a_weights, b_weights) for the RHS and LHS respectively.
    """
    if N < 4:
        raise ValueError("Compact FD requires N >= 4.")
    a = 14.0 / 9.0
    b = 1.0 / 9.0
    c = 0.0
    a_w = [-b / 2.0, -a / 2.0, 0.0, a / 2.0, b / 2.0]
    b_w = [0.0, alpha, 1.0, alpha, 0.0]
    return a_w, b_w


# ---------------------------------------------------------------------------
# Tangent-plane projection for spherical FD
# ---------------------------------------------------------------------------
def tangent_plane_project(theta_c: float, phi_c: float,
                           theta: float, phi: float) -> Tuple[float, float]:
    """
    Gnomonic projection of (theta, phi) onto the tangent plane at
    (theta_c, phi_c).  Returns (u, v) in radians.

    Mathematical form:
        x_c = (sin theta_c cos phi_c, sin theta_c sin phi_c, cos theta_c)
        x   = (sin theta   cos phi,   sin theta   sin phi,   cos theta  )
        n_hat = x_c  (outward normal)
        e_theta = d x_c / d theta_c,  e_phi = d x_c / d phi_c / sin theta_c
        u = arctan( x . e_phi / (x . n_hat) )
        v = arctan( x . e_theta / (x . n_hat) )
    """
    xc = math.sin(theta_c) * math.cos(phi_c)
    yc = math.sin(theta_c) * math.sin(phi_c)
    zc = math.cos(theta_c)
    x  = math.sin(theta)   * math.cos(phi)
    y  = math.sin(theta)   * math.sin(phi)
    z  = math.cos(theta)

    # Orthonormal basis on tangent plane
    e_theta = (math.cos(theta_c) * math.cos(phi_c),
               math.cos(theta_c) * math.sin(phi_c),
               -math.sin(theta_c))
    if abs(math.sin(theta_c)) < 1.0e-12:
        e_phi = (1.0, 0.0, 0.0)
    else:
        e_phi = (-math.sin(phi_c), math.cos(phi_c), 0.0)

    dot_n = x * xc + y * yc + z * zc
    if dot_n < 1.0e-30:
        return (0.0, 0.0)
    dot_et = x * e_theta[0] + y * e_theta[1] + z * e_theta[2]
    dot_ep = x * e_phi[0]   + y * e_phi[1]   + z * e_phi[2]
    return math.atan2(dot_ep, dot_n), math.atan2(dot_et, dot_n)


# ---------------------------------------------------------------------------
# High-order spherical FD operator at a single node
# ---------------------------------------------------------------------------
def spherical_fd_gradient(theta_c: float, phi_c: float,
                           nodes_thetaphi: List[Tuple[float, float]],
                           f_values: List[float]) -> Tuple[float, float]:
    """
    High-order FD gradient of a scalar field f at (theta_c, phi_c)
    using the given neighbour node values.

    Steps:
      1. Project all neighbour positions onto the tangent plane at (theta_c, phi_c).
      2. Build a local 2D polynomial fit   f(u,v) ~ sum a_{ij} u^i v^j.
      3. Solve for coefficients using least-squares.
      4. Return  df/dtheta, df/dphi  from the fitted polynomial.

    The polynomial fit uses terms up to total degree D, chosen so that
    the number of terms <= number of neighbours.
    """
    n_nbr = len(nodes_thetaphi)
    if n_nbr < 3:
        return 0.0, 0.0

    # 1) tangent-plane projection
    uv = [tangent_plane_project(theta_c, phi_c, th, ph)
          for (th, ph) in nodes_thetaphi]

    # 2) choose polynomial degree
    max_D = 1
    while (max_D + 1) * (max_D + 2) // 2 <= n_nbr:
        max_D += 1
    D = min(max_D, 4)   # cap at 4th order to avoid overfitting
    n_terms = (D + 1) * (D + 2) // 2

    # 3) build design matrix A  (n_nbr x n_terms)
    A = []
    for (u, v) in uv:
        row = []
        for p in range(D + 1):
            for q in range(D + 1 - p):
                row.append((u ** p) * (v ** q))
        A.append(row)

    # 4) solve via normal equations  (A^T A) a = A^T f
    ATA = [[0.0] * n_terms for _ in range(n_terms)]
    ATf = [0.0] * n_terms
    for i in range(n_terms):
        for j in range(n_terms):
            s = 0.0
            for k in range(n_nbr):
                s += A[k][i] * A[k][j]
            ATA[i][j] = s
        s = 0.0
        for k in range(n_nbr):
            s += A[k][i] * f_values[k]
        ATf[i] = s

    coeffs = _solve_linear_system(ATA, ATf)
    if coeffs is None:
        return 0.0, 0.0

    # 5) extract derivatives:  d/dtheta = d/dv (index p=0,q=1)
    #                          d/dphi   = d/du (index p=1,q=0)
    # The ordering is:  p=0,q=0; p=0,q=1; ... ; p=0,q=D; p=1,q=0; ...
    # Find indices:
    idx_p1q0 = -1
    idx_p0q1 = -1
    idx = 0
    for p in range(D + 1):
        for q in range(D + 1 - p):
            if p == 1 and q == 0:
                idx_p1q0 = idx
            if p == 0 and q == 1:
                idx_p0q1 = idx
            idx += 1

    df_dtheta = coeffs[idx_p0q1] if idx_p0q1 >= 0 else 0.0
    df_dphi   = coeffs[idx_p1q0] if idx_p1q0 >= 0 else 0.0

    return df_dtheta, df_dphi


# ---------------------------------------------------------------------------
# Utility: solve a small linear system by Gaussian elimination with pivoting
# ---------------------------------------------------------------------------
def _solve_linear_system(A: List[List[float]], b: List[float]) -> List[float] | None:
    n = len(b)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for k in range(n):
        # partial pivoting
        max_row = k
        max_val = abs(M[k][k])
        for i in range(k + 1, n):
            if abs(M[i][k]) > max_val:
                max_val = abs(M[i][k])
                max_row = i
        if max_val < 1.0e-30:
            return None  # singular
        M[k], M[max_row] = M[max_row], M[k]
        # eliminate below
        for i in range(k + 1, n):
            factor = M[i][k] / M[k][k]
            for j in range(k, n + 1):
                M[i][j] -= factor * M[k][j]
    # back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(M[i][i]) < 1.0e-30:
            return None
        s = M[i][n]
        for j in range(i + 1, n):
            s -= M[i][j] * x[j]
        x[i] = s / M[i][i]
    return x


# ---------------------------------------------------------------------------
# Multi-dimensional Fornberg weights (tensor product on sphere)
# ---------------------------------------------------------------------------
def sphere_laplacian_fd_2d(u: List[List[float]], dtheta: float, dphi: float,
                            theta: float) -> List[List[float]]:
    """
    4th-order FD approximation to the spherical Laplacian:
        Delta_S f = 1/sin(theta) d/dtheta(sin(theta) df/dtheta) + 1/sin^2(theta) d^2 f / dphi^2
    on a uniform (theta, phi) grid.
    Returns the array of Laplacian values at each grid point.
    """
    n_theta = len(u)
    n_phi = len(u[0])
    lap = [[0.0] * n_phi for _ in range(n_theta)]
    # 4th-order central weights for 2nd derivative: [-1/12, 4/3, -5/2, 4/3, -1/12]
    w2 = [-1.0 / 12.0, 4.0 / 3.0, -5.0 / 2.0, 4.0 / 3.0, -1.0 / 12.0]
    for i in range(2, n_theta - 2):
        s_theta = math.sin(theta + i * dtheta)
        s_theta_p = math.sin(theta + (i + 1) * dtheta)
        s_theta_m = math.sin(theta + (i - 1) * dtheta)
        if abs(s_theta) < 1.0e-30:
            continue
        for j in range(n_phi):
            # d^2 f / d phi^2
            d2f_dphi2 = 0.0
            for k in range(-2, 3):
                jj = (j + k) % n_phi
                d2f_dphi2 += w2[k + 2] * u[i][jj]
            d2f_dphi2 /= (dphi * dphi)

            # 1/sin d/dtheta (sin df/dtheta)
            df_theta_p = 0.0
            df_theta_m = 0.0
            for k in range(-2, 3):
                ii = i + k
                if 0 <= ii < n_theta:
                    df_theta_p += w2[k + 2] * u[ii][(j + 1) % n_phi]
                    df_theta_m += w2[k + 2] * u[ii][(j - 1) % n_phi]
            # simplified: use standard central difference for the radial part
            d_f_di = (u[i + 1][j] - u[i - 1][j]) / (2.0 * dtheta)
            cot_term = math.cos(theta + i * dtheta) / s_theta * d_f_di
            d2f_dtheta2 = 0.0
            for k in range(-2, 3):
                ii = i + k
                if 0 <= ii < n_theta:
                    d2f_dtheta2 += w2[k + 2] * u[ii][j]
            d2f_dtheta2 /= (dtheta * dtheta)
            lap[i][j] = d2f_dtheta2 + cot_term + d2f_dphi2 / (s_theta * s_theta)
    return lap


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Sanity checks:
    # 1) Fornberg weights for 2nd derivative on 5-point stencil
    w = fornberg_weights(0.0, [-2.0, -1.0, 0.0, 1.0, 2.0], 4)
    print("Fornberg weights for 2nd deriv, 5-pt stencil:", w[2])

    # 2) Central weights for 4th-order 2nd derivative
    print("Central 4th-order 2nd deriv:", central_fd_weights(2, 2))

    # 3) Tangent-plane projection of north pole -> (0,0)
    u, v = tangent_plane_project(0.0001, 0.0, 0.0001, 0.0)
    print("Tangent plane projection:", u, v)
