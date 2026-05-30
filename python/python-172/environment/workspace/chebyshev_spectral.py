# -*- coding: utf-8 -*-

import numpy as np
from numpy.polynomial.chebyshev import chebval


def chebyshev_nodes(n):
    if n < 1:
        raise ValueError("n must be at least 1.")
    j = np.arange(n + 1)
    x = np.cos(np.pi * j / n)

    x[0] = 1.0
    x[-1] = -1.0
    return x


def chebyshev_vandermonde(x, n):
    m = len(x)
    V = np.ones((m, n + 1))
    if n >= 1:
        V[:, 1] = x
    for k in range(2, n + 1):
        V[:, k] = 2.0 * x * V[:, k - 1] - V[:, k - 2]
    return V


def spectral_differentiation_matrix(n):







    raise NotImplementedError("Hole 1: spectral_differentiation_matrix is missing.")


def clenshaw_evaluate(coef, x):
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


    b_kp2 = np.zeros_like(x, dtype=np.float64)
    b_kp1 = np.zeros_like(x, dtype=np.float64)
    for k in range(npl - 1, 0, -1):
        b_k = 2.0 * x * b_kp1 - b_kp2 + coef[k]
        b_kp2 = b_kp1
        b_kp1 = b_k
    fx = x * b_kp1 - b_kp2 + coef[0]
    return fx.item() if scalar_input else fx


def chebyshev_analyze(f_vals):
    n = len(f_vals) - 1
    if n < 1:
        raise ValueError("At least 2 nodes required.")
    j = np.arange(n + 1)
    k = np.arange(n + 1)
    cj = np.ones(n + 1)
    cj[0] = 2.0
    cj[-1] = 2.0
    ck = cj.copy()


    fj = f_vals / cj
    coef = np.zeros(n + 1)
    for kk in k:
        coef[kk] = np.sum(fj * np.cos(np.pi * kk * j / n))
    coef = (2.0 / n) * coef / ck
    return coef


def chebyshev_synthesize(coef):
    n = len(coef) - 1
    x = chebyshev_nodes(n)
    return clenshaw_evaluate(coef, x)


def apply_boundary_conditions(u, bc_type="dirichlet", bc_vals=(0.0, 0.0)):
    u = np.asarray(u, dtype=np.float64)
    if bc_type == "dirichlet":

        u[-1] = bc_vals[0]
        u[0] = bc_vals[1]
    elif bc_type == "neumann":



        pass
    return u


def chebyshev_interpolate(coef, x_new):
    x_new = np.clip(x_new, -1.0, 1.0)
    return clenshaw_evaluate(coef, x_new)
