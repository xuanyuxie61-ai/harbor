"""
lagrange_surrogate.py
=====================

Lagrange interpolation basis and tensor-product surrogate model used as
the LEVEL_1 fidelity in the multi-fidelity hierarchy.

Mathematical background
-----------------------
Given d-dimensional parameter space xi in [a, b]^d and nodes {x_j}_{j=1}^n
along each axis, the tensor-product Lagrange interpolant is:

    L(xi) = sum_{i in {1..n}^d} f(x_{i_1}, ..., x_{i_d}) * prod_{k=1}^d l_{i_k}(xi_k)

where l_j is the j-th Lagrange basis polynomial:

    l_j(x) = prod_{m != j} (x - x_m) / (x_j - x_m)

For stability we use Chebyshev nodes of the second kind:

    x_j = (a+b)/2 + (b-a)/2 * cos(pi * j / (n-1)),  j = 0, ..., n-1

This choice minimizes the Lebesgue constant and suppresses Runge's phenomenon
(Trefethen 2008, "Patient and Impatient Accuracy of Polynomial Interpolation").

We also compute the barycentric weights

    w_j = (-1)^j * delta_j,  delta_0 = delta_{n-1} = 1/2, delta_j = 1 o/w

so that the interpolant can be evaluated in O(n) per dimension via the
barycentric formula (Berrut & Trefethen 2004).
"""

from __future__ import annotations

import math
from typing import Callable, List, Tuple


# ----------------------------------------------------------------------
# Node generation.
# ----------------------------------------------------------------------
def chebyshev_nodes(a: float, b: float, n: int) -> List[float]:
    """Chebyshev nodes of the second kind on [a, b], n points."""
    if n < 1:
        raise ValueError("chebyshev_nodes: n must be >= 1.")
    if n == 1:
        return [0.5 * (a + b)]
    nodes: List[float] = []
    for j in range(n):
        theta = math.pi * j / (n - 1)
        x = 0.5 * (a + b) + 0.5 * (b - a) * math.cos(theta)
        nodes.append(x)
    return nodes


def tensor_grid(bounds: List[Tuple[float, float]],
                n_per_dim: int) -> List[List[float]]:
    """Construct a full tensor grid of d-dimensional nodes."""
    d = len(bounds)
    axes = [chebyshev_nodes(a, b, n_per_dim) for (a, b) in bounds]
    # Recursive Cartesian product.
    grid: List[List[float]] = [[]]
    for ax in axes:
        new_grid: List[List[float]] = []
        for g in grid:
            for x in ax:
                new_grid.append(g + [x])
        grid = new_grid
    return grid


# ----------------------------------------------------------------------
# Barycentric weights.
# ----------------------------------------------------------------------
def barycentric_weights(n: int) -> List[float]:
    """Barycentric weights w_j = (-1)^j * delta_j for Chebyshev 2nd kind."""
    w = [1.0] * n
    for j in range(n):
        w[j] = 1.0 if j % 2 == 0 else -1.0
        if j == 0 or j == n - 1:
            w[j] *= 0.5
    return w


# ----------------------------------------------------------------------
# 1-D Lagrange evaluation (naive and barycentric).
# ----------------------------------------------------------------------
def lagrange_basis_1d(xd: List[float], x: float) -> List[float]:
    """Evaluate the n Lagrange basis functions at a single point x.

    Returns lb[j] = l_j(x) for j = 0..n-1.
    """
    n = len(xd)
    lb = [1.0] * n
    for j in range(n):
        for m in range(n):
            if m == j:
                continue
            denom = xd[j] - xd[m]
            if abs(denom) < 1.0e-30:
                # Degenerate node placement: fall back to Kronecker delta.
                lb[j] = 1.0 if abs(x - xd[j]) < 1.0e-12 else 0.0
                break
            lb[j] *= (x - xd[m]) / denom
    return lb


def lagrange_value_1d(xd: List[float], yd: List[float], x: float) -> float:
    """Evaluate the 1-D Lagrange interpolant at x.

    L(x) = sum_{j} y_j l_j(x).
    """
    if len(xd) != len(yd):
        raise ValueError("lagrange_value_1d: xd and yd length mismatch.")
    lb = lagrange_basis_1d(xd, x)
    return sum(yd[j] * lb[j] for j in range(len(xd)))


def barycentric_eval_1d(xd: List[float], yd: List[float], x: float) -> float:
    """Barycentric formula for Lagrange interpolation at a single point.

        L(x) = [ sum_j w_j y_j / (x - x_j) ] / [ sum_j w_j / (x - x_j) ]

    with exact-match short-circuit when x coincides with a node.
    """
    n = len(xd)
    if n != len(yd):
        raise ValueError("barycentric_eval_1d: length mismatch.")
    w = barycentric_weights(n)
    # Exact match check.
    for j in range(n):
        if abs(x - xd[j]) < 1.0e-14 * max(1.0, abs(xd[j])):
            return yd[j]
    num = 0.0
    den = 0.0
    for j in range(n):
        t = w[j] / (x - xd[j])
        num += t * yd[j]
        den += t
    if abs(den) < 1.0e-30:
        return 0.0
    return num / den


# ----------------------------------------------------------------------
# Tensor-product Lagrange interpolation.
# ----------------------------------------------------------------------
def lagrange_tensor_eval(
    xi: List[float],
    nodes: List[List[float]],
    values: List[float],
) -> float:
    """Evaluate the tensor-product Lagrange interpolant at point xi.

    nodes : list of N grid points (each a d-dimensional list).
    values: list of N scalar function values, matching `nodes` order.

    Algorithm: dimension-by-dimension contraction.  Let d = len(xi).  We
    reshape `values` into a d-dimensional tensor of side n (assumed equal
    per axis), then contract each axis with the 1-D Lagrange basis at xi[k].
    """
    if not nodes or not values:
        return 0.0
    d = len(xi)
    n_total = len(nodes)
    # Infer n (number of nodes per axis) assuming uniform tensor.
    n_per = int(round(n_total ** (1.0 / d)))
    if n_per ** d != n_total:
        raise ValueError(
            "lagrange_tensor_eval: tensor-product grid requires n_total = n^d."
        )
    # Infer the 1-D axes from the nodes.
    axes: List[List[float]] = []
    for k in range(d):
        col = sorted({round(n[k], 12) for n in nodes})
        if len(col) != n_per:
            raise ValueError(
                f"lagrange_tensor_eval: axis {k} has {len(col)} distinct values,"
                f" expected {n_per}."
            )
        axes.append(sorted(col))

    # Map each flat index to its multi-index.
    flat = list(values)
    # Contract axis by axis.
    current = flat
    current_side = [n_per] * d
    for k in range(d):
        n_side = current_side[0]
        # 1-D Lagrange weights along axis k at xi[k].
        lb = lagrange_basis_1d(axes[k], xi[k])
        # Compute the contracted tensor: collapse axis 0 of current.
        n_remaining = len(current) // n_side
        new_current = [0.0] * n_remaining
        for r in range(n_remaining):
            s = 0.0
            for j in range(n_side):
                s += lb[j] * current[j * n_remaining + r]
            new_current[r] = s
        current = new_current
        current_side = current_side[1:]
    if not current:
        return 0.0
    return current[0]


# ----------------------------------------------------------------------
# Build calibration: sample a function on a tensor grid and store nodes/values.
# ----------------------------------------------------------------------
def build_lagrange_calibration(
    f: Callable[[List[float]], float],
    bounds: List[Tuple[float, float]],
    n_per_dim: int,
) -> dict:
    """Build a Lagrange tensor-product calibration dictionary.

    Returns {"bounds": bounds, "n_per_dim": n_per_dim,
             "nodes": [...], "values": [...]}.
    """
    nodes = tensor_grid(bounds, n_per_dim)
    values = [f(n) for n in nodes]
    return {
        "bounds": bounds,
        "n_per_dim": n_per_dim,
        "nodes": nodes,
        "values": values,
    }


# ----------------------------------------------------------------------
# Lebesgue constant estimate (diagnostic of interpolation stability).
# ----------------------------------------------------------------------
def lebesgue_constant_1d(n: int) -> float:
    """Approximate Lebesgue constant for Chebyshev-2 nodes of degree n-1.

    For Chebyshev nodes of the second kind:
        Lambda_n ~ (2/pi) log(n) + 0.573...   (Trefethen 2008)
    """
    if n < 2:
        return 1.0
    return (2.0 / math.pi) * math.log(n) + 0.5731


def tensor_lebesgue(d: int, n_per_dim: int) -> float:
    """Tensor-product Lebesgue constant: Lambda_d = Lambda_1^d."""
    return lebesgue_constant_1d(n_per_dim) ** d
