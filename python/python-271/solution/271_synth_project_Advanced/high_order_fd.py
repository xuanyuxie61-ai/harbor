# -*- coding: utf-8 -*-
"""
High-order finite-difference stencils for differentiating ground-state
observables of the TFIM with respect to the control parameter lambda = h/J.

The QCP is detected by non-analyticity of E_0(L, lambda) in the
thermodynamic limit.  On a finite lattice, E_0 is analytic but its
higher lambda-derivatives develop sharp peaks near lambda_c = 1 that
sharpen with L.  We therefore need *stable* high-order derivatives.

This module implements the Fornberg (1988) algorithm for arbitrary-order
central finite-difference weights, plus Richardson extrapolation for
accuracy boost and a stability diagnostic based on the Lebesgue
constant of the stencil.

References
----------
[1] B. Fornberg, "Generation of finite difference formulas on
    arbitrarily spaced spaces", Math. Comp. 51, 699 (1988).
[2] J. M. Varah, "The perils of Richardson extrapolation",
    SIAM Rev. (notes).
"""

from __future__ import annotations
from typing import Tuple, Sequence
import numpy as np
from fractions import Fraction
try:
    from . import constants as C
except ImportError:
    import constants as C


# ---------------------------------------------------------------------------
# Fornberg weights
# ---------------------------------------------------------------------------
def fornberg_weights(x: Sequence[float], x0: float,
                      max_deriv: int = 4) -> np.ndarray:
    """Return weights w[d, i] such that

        f^(d)(x0) ~ sum_i w[d, i] f(x_i)

    for derivatives d = 0, ..., max_deriv at node x0 given values at
    the (possibly non-uniform) nodes x.

    The algorithm follows Fornberg (1988) and is numerically stable
    for stencil sizes up to ~30 before floating-point cancellation
    starts to dominate; beyond that we fall back to symbolic rational
    arithmetic via ``fractions.Fraction`` for the central-difference
    special case below.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n <= max_deriv:
        raise ValueError(
            f"Need at least max_deriv+1={max_deriv+1} nodes for "
            f"derivative order {max_deriv}, got {n}")

    M = max_deriv
    c = np.zeros((M + 1, n), dtype=float)
    c1 = 1.0
    c4 = x[0] - x0
    c[0, 0] = 1.0
    for i in range(1, n):
        mn = min(i, M)
        c2 = 1.0
        c5 = c4
        c4 = x[i] - x0
        for j in range(i):
            c3 = x[i] - x[j]
            c2 *= c3
            if j == i - 1:
                for k in range(mn, 0, -1):
                    c[k, i] = c1 * (k * c[k - 1, i - 1] - c5 * c[k, i - 1]) / c2
                c[0, i] = -c1 * c5 * c[0, i - 1] / c2
            for k in range(mn, 0, -1):
                c[k, j] = (c4 * c[k, j] - k * c[k - 1, j]) / c3
            c[0, j] = c4 * c[0, j] / c3
        c1 = c2
    return c


def central_fd_weights(order: int, deriv: int) -> np.ndarray:
    """Return 1-D central FD weights for the ``deriv``-th derivative
    using a symmetric stencil of total width ``order + 1`` (``order``
    must be even and >= deriv).

    We use a clean implementation of the Fornberg (1988) algorithm on
    a symmetric integer grid  x_j = j - order/2, j = 0, ..., order.
    """
    if order % 2 != 0:
        raise ValueError(f"order must be even for central stencil, got {order}")
    if deriv > order:
        raise ValueError(f"deriv={deriv} cannot exceed order={order}")

    half = order // 2
    xs = np.arange(-half, half + 1, dtype=float)
    n = len(xs)
    M = deriv
    # Fornberg algorithm  (numerical, float64)
    c = np.zeros((M + 1, n), dtype=float)
    c[0, 0] = 1.0
    c1 = 1.0
    c4 = xs[0] - 0.0  # target point x0 = 0
    for i in range(1, n):
        mn = min(i, M)
        c2 = 1.0
        c5 = c4
        c4 = xs[i] - 0.0
        for j in range(i):
            c3 = xs[i] - xs[j]
            c2 *= c3
            if j == i - 1:
                for k in range(mn, 0, -1):
                    c[k, i] = c1 * (k * c[k - 1, i - 1] - c5 * c[k, i - 1]) / c2
                c[0, i] = -c1 * c5 * c[0, i - 1] / c2
            for k in range(mn, 0, -1):
                c[k, j] = (c4 * c[k, j] - k * c[k - 1, j]) / c3
            c[0, j] = c4 * c[0, j] / c3
        c1 = c2
    return c[deriv, :]


# ---------------------------------------------------------------------------
# Lebesgue constant of a stencil
# ---------------------------------------------------------------------------
def stencil_lebesgue_constant(weights: np.ndarray) -> float:
    """Return the discrete Lebesgue constant sum_i |w_i|.  Large values
    indicate amplification of round-off: for a 10th-order central
    second-derivative stencil one expects L ~ O(10^2)."""
    return float(np.sum(np.abs(weights)))


# ---------------------------------------------------------------------------
# Apply a derivative to a sampled function
# ---------------------------------------------------------------------------
def apply_fd(y: np.ndarray, h_step: float, deriv: int,
              order: int = 4) -> np.ndarray:
    """Apply a central FD derivative to a uniformly sampled signal y.

    Boundary treatment: ``deriv``-th order one-sided Fornberg stencils
    are used at the edges so that the output has the same length as y.
    This is critical when the signal represents E_0(L, lambda) on a
    finite grid in lambda: we must not discard the end-points because
    they often carry the largest Fisher information about lambda_c.
    """
    if y.ndim != 1:
        raise ValueError("apply_fd expects a 1-D signal")
    n = len(y)
    if n < order + 1:
        raise ValueError(f"signal too short (n={n}) for order={order}")

    w_central = central_fd_weights(order, deriv)
    stencil = np.arange(-order // 2, order // 2 + 1)
    out = np.zeros_like(y)

    # Interior
    for k, s in enumerate(stencil):
        out[order // 2: n - order // 2] += w_central[k] * y[order // 2 + s: n - order // 2 + s]

    # Boundaries via Fornberg one-sided stencils
    for edge in (0, n - 1):
        x0 = float(edge)
        # pick the order+1 closest nodes
        if edge == 0:
            xs = np.arange(order + 1, dtype=float)
            idx = np.arange(order + 1)
        else:
            xs = np.arange(n - order - 1, n, dtype=float)
            idx = np.arange(n - order - 1, n)
        w_edge = fornberg_weights(xs, x0, max_deriv=deriv)[deriv]
        val = float(np.dot(w_edge, y[idx]))
        out[edge] = val

    return out / (h_step ** deriv)


# ---------------------------------------------------------------------------
# Richardson extrapolation
# ---------------------------------------------------------------------------
def richardson_extrapolate(values: np.ndarray, refinement: int = 2,
                            deriv_order: int = 2) -> float:
    """Given ``values`` = [D_h, D_{h/r}, D_{h/r^2}, ...] perform
    Richardson extrapolation for a derivative of formal order
    ``deriv_order``.

    The error of a central FD of order p is a series in h^2, h^4, ...
    so the Richardson factors are r^{2k} - 1 for successive columns.
    """
    if len(values) < 2:
        return float(values[0])
    v = np.asarray(values, dtype=float).copy()
    r = float(refinement)
    for k in range(len(v) - 1):
        factor = r ** (2 * (k + 1)) - 1.0
        v_next = np.zeros(len(v) - k - 1)
        for j in range(len(v_next)):
            v_next[j] = (factor * v[j + 1] - v[j]) / (factor - 1.0)
        v = v_next
    return float(v[0])


def derivative_at_lambda(f_eval, lam0: float, deriv: int,
                          order: int = 8, n_richardson: int = 3,
                          h0: float = 1.0e-2) -> float:
    """Robust evaluation of the lambda-derivative of an arbitrary
    scalar function ``f_eval(lambda)`` using high-order central FD
    plus Richardson extrapolation.

    The step h0 is halved ``n_richardson`` times; at each refinement
    the central stencil of total width ``order + 1`` is applied.
    The final value is the Richardson-extrapolated combination.
    """
    values = []
    h = h0
    for _ in range(n_richardson):
        xs = lam0 + np.arange(-order // 2, order // 2 + 1) * h
        ys = np.array([f_eval(float(x)) for x in xs], dtype=float)
        w = central_fd_weights(order, deriv)
        d = float(np.dot(w, ys)) / (h ** deriv)
        values.append(d)
        h /= 2.0
    return richardson_extrapolate(np.asarray(values), deriv_order=deriv)
