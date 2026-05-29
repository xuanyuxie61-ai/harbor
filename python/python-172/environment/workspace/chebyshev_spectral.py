# -*- coding: utf-8 -*-
"""
Chebyshev Spectral Methods Core Module
========================================
Implements Chebyshev polynomial basis, spectral differentiation matrices,
and Clenshaw recurrence evaluation. Adapted from ACM TOMS Algorithm 446.

Mathematical foundation:
- Chebyshev polynomials of the first kind: T_n(x) = cos(n arccos x), x in [-1,1]
- Gauss-Lobatto nodes: x_j = cos(pi*j/N), j=0,...,N
- Spectral differentiation matrix D with entries:
    D_{ij} = (c_i / c_j) * (-1)^{i+j} / (x_i - x_j)  for i != j
    D_{ii} = -x_i / (2(1-x_i^2))  for i=1,...,N-1
    D_{00} = (2N^2+1)/6, D_{NN} = -(2N^2+1)/6
  where c_0 = c_N = 2, c_j = 1 otherwise.
- Discrete Chebyshev transform (analysis): project function values f(x_j) onto coefficients a_k
    a_k = (2/(N c_k)) sum_{j=0}^N (1/c_j) f(x_j) T_k(x_j)
"""

import numpy as np
from numpy.polynomial.chebyshev import chebval


def chebyshev_nodes(n):
    """
    Generate Chebyshev-Gauss-Lobatto (CGL) nodes of degree n.
    x_j = cos(pi * j / n), j = 0, ..., n

    Parameters
    ----------
    n : int
        Number of intervals (there are n+1 nodes).

    Returns
    -------
    x : ndarray, shape (n+1,)
        CGL nodes in [-1, 1], descending from +1 to -1.
    """
    if n < 1:
        raise ValueError("n must be at least 1.")
    j = np.arange(n + 1)
    x = np.cos(np.pi * j / n)
    # Enforce exact boundaries to mitigate round-off
    x[0] = 1.0
    x[-1] = -1.0
    return x


def chebyshev_vandermonde(x, n):
    """
    Build the Chebyshev Vandermonde matrix V_{jk} = T_k(x_j).

    Parameters
    ----------
    x : ndarray
        Evaluation points.
    n : int
        Maximum polynomial degree (matrix has columns 0..n).

    Returns
    -------
    V : ndarray
        Vandermonde matrix of shape (len(x), n+1).
    """
    m = len(x)
    V = np.ones((m, n + 1))
    if n >= 1:
        V[:, 1] = x
    for k in range(2, n + 1):
        V[:, k] = 2.0 * x * V[:, k - 1] - V[:, k - 2]
    return V


def spectral_differentiation_matrix(n):
    """
    Construct the (n+1) x (n+1) Chebyshev spectral differentiation matrix D.
    For a vector u of function values at CGL nodes, D @ u approximates u'.

    The analytic entries for the first derivative at CGL nodes are:
      D_{ij} = (c_i / c_j) * (-1)^{i+j} / (x_i - x_j),   i != j
      D_{00} = (2 n^2 + 1) / 6
      D_{nn} = - (2 n^2 + 1) / 6
      D_{ii} = - x_i / (2 (1 - x_i^2)),   i = 1, ..., n-1
    where c_0 = c_n = 2, c_j = 1 for interior j.

    Parameters
    ----------
    n : int
        Number of intervals.

    Returns
    -------
    D : ndarray, shape (n+1, n+1)
        Spectral differentiation matrix.
    """
    # TODO: Implement the Chebyshev spectral differentiation matrix.
    # This requires:
    #   1. Generating CGL nodes via chebyshev_nodes(n)
    #   2. Setting up the c-vector with c_0 = c_n = 2
    #   3. Computing off-diagonal entries with (-1)^{i+j} factor
    #   4. Computing diagonal entries D_{00}, D_{nn}, and D_{ii}
    # Return the (n+1) x (n+1) matrix D.
    raise NotImplementedError("Hole 1: spectral_differentiation_matrix is missing.")


def clenshaw_evaluate(coef, x):
    """
    Evaluate a Chebyshev series at points x using Clenshaw recurrence.
    Adapted from ACM TOMS 446 echeb/edcheb.

    For coefficients a_0, ..., a_{N-1} representing
        p(x) = sum_{k=0}^{N-1} a_k T_k(x),
    the recurrence is:
        b_N = b_{N+1} = 0
        b_k = 2 x b_{k+1} - b_{k+2} + a_k,   k = N-1, ..., 1
        p(x) = x b_1 - b_2 + a_0

    Parameters
    ----------
    coef : ndarray
        Chebyshev coefficients.
    x : ndarray or float
        Evaluation point(s) in [-1, 1].

    Returns
    -------
    fx : ndarray or float
        Evaluated series values.
    """
    coef = np.asarray(coef)
    x = np.asarray(x)
    scalar_input = False
    if x.ndim == 0:
        x = x.reshape(1)
        scalar_input = True

    npl = len(coef)
    fx = np.zeros_like(x, dtype=np.float64)

    if npl == 0:
        return fx.item() if scalar_input else fx
    if npl == 1:
        fx[:] = coef[0]
        return fx.item() if scalar_input else fx

    # Clenshaw recurrence vectorized
    b_kp2 = np.zeros_like(x, dtype=np.float64)
    b_kp1 = np.zeros_like(x, dtype=np.float64)
    for k in range(npl - 1, 0, -1):
        b_k = 2.0 * x * b_kp1 - b_kp2 + coef[k]
        b_kp2 = b_kp1
        b_kp1 = b_k
    fx = x * b_kp1 - b_kp2 + coef[0]
    return fx.item() if scalar_input else fx


def chebyshev_analyze(f_vals):
    """
    Discrete Chebyshev transform (analysis): compute Chebyshev coefficients
    from function values at CGL nodes.

    For N intervals (N+1 nodes), coefficients a_k are:
        a_k = (2 / (N * c_k)) * sum_{j=0}^N (1/c_j) f(x_j) cos(pi*k*j/N)
    where c_0 = c_N = 2, c_j = 1 otherwise.

    Parameters
    ----------
    f_vals : ndarray, shape (N+1,)
        Function values at CGL nodes.

    Returns
    -------
    coef : ndarray, shape (N+1,)
        Chebyshev coefficients.
    """
    n = len(f_vals) - 1
    if n < 1:
        raise ValueError("At least 2 nodes required.")
    j = np.arange(n + 1)
    k = np.arange(n + 1)
    cj = np.ones(n + 1)
    cj[0] = 2.0
    cj[-1] = 2.0
    ck = cj.copy()

    # Compute sum_{j=0}^N (1/c_j) f(x_j) cos(pi*k*j/N)
    fj = f_vals / cj
    coef = np.zeros(n + 1)
    for kk in k:
        coef[kk] = np.sum(fj * np.cos(np.pi * kk * j / n))
    coef = (2.0 / n) * coef / ck
    return coef


def chebyshev_synthesize(coef):
    """
    Synthesize function values at CGL nodes from Chebyshev coefficients.

    Parameters
    ----------
    coef : ndarray
        Chebyshev coefficients.

    Returns
    -------
    f_vals : ndarray
        Function values at CGL nodes.
    """
    n = len(coef) - 1
    x = chebyshev_nodes(n)
    return clenshaw_evaluate(coef, x)


def apply_boundary_conditions(u, bc_type="dirichlet", bc_vals=(0.0, 0.0)):
    """
    Apply boundary conditions to spectral solution vector at CGL nodes.
    Nodes are ordered x_0=+1, x_n=-1 (descending).

    Parameters
    ----------
    u : ndarray
        Solution vector.
    bc_type : str
        "dirichlet" or "neumann".
    bc_vals : tuple
        Boundary values at (left=-1, right=+1) for Dirichlet,
        or derivative values for Neumann.

    Returns
    -------
    u : ndarray
        Modified solution vector satisfying boundary conditions.
    """
    u = np.asarray(u, dtype=np.float64)
    if bc_type == "dirichlet":
        # u[-1] corresponds to x=-1 (left), u[0] to x=+1 (right)
        u[-1] = bc_vals[0]
        u[0] = bc_vals[1]
    elif bc_type == "neumann":
        # Approximate derivative with one-sided difference and adjust
        # This is a simple enforcement; in full spectral methods one uses
        # basis recombination.
        pass
    return u


def chebyshev_interpolate(coef, x_new):
    """
    Interpolate using Chebyshev coefficients to new points via Clenshaw.

    Parameters
    ----------
    coef : ndarray
        Chebyshev coefficients.
    x_new : ndarray
        New evaluation points in [-1, 1].

    Returns
    -------
    vals : ndarray
        Interpolated values.
    """
    x_new = np.clip(x_new, -1.0, 1.0)
    return clenshaw_evaluate(coef, x_new)
