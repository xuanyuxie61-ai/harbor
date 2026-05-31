"""
multi_dim_quadrature.py
Multidimensional product quadrature rule construction.

Adapted from:
  - 919_product_rule: Tensor product of 1D quadrature rules

Role in synthesis:
  Constructs multi-dimensional quadrature rules for integrating over
  trait space (e.g., susceptibility, virulence) in structured population models.
"""

import numpy as np
from itertools import product


def r8vec_direct_product(factor_index: int, factor_order: int, factor_value: np.ndarray,
                         factor_num: int, point_num: int, x: np.ndarray) -> np.ndarray:
    """
    Fill a direct product array by repeating blocks.
    """
    x_out = x.copy()
    rep = 1
    skip = point_num // (factor_order * rep)
    for j in range(rep):
        for k in range(factor_order):
            start = j * factor_order * skip + k * skip
            x_out[start:start + skip, factor_index] = factor_value[k]
        rep *= factor_order
    return x_out


def r8vec_direct_product2(factor_index: int, factor_order: int, factor_weight: np.ndarray,
                          factor_num: int, point_num: int, w: np.ndarray) -> np.ndarray:
    """
    Fill a direct product weight array by multiplying weights.
    """
    w_out = w.copy()
    rep = 1
    skip = point_num // (factor_order * rep)
    for j in range(rep):
        for k in range(factor_order):
            start = j * factor_order * skip + k * skip
            w_out[start:start + skip] *= factor_weight[k]
        rep *= factor_order
    return w_out


def construct_product_rule(rules_1d: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct a multidimensional product quadrature rule.

    Parameters
    ----------
    rules_1d : list of (points, weights)
        Each tuple contains 1D quadrature points and weights.

    Returns
    -------
    points : ndarray, shape (N, d)
        Multidimensional quadrature points.
    weights : ndarray, shape (N,)
        Corresponding weights.
    """
    d = len(rules_1d)
    orders = [len(r[0]) for r in rules_1d]
    N = int(np.prod(orders))

    points = np.zeros((N, d))
    weights = np.ones(N)

    for dim, (pts, wts) in enumerate(rules_1d):
        points = r8vec_direct_product(dim, len(pts), pts, d, N, points)
        weights = r8vec_direct_product2(dim, len(pts), wts, d, N, weights)

    return points, weights


def integrate_trait_space(
    trait_func,
    trait_ranges: list[tuple[float, float]],
    orders: list[int]
) -> float:
    """
    Integrate a function over a multidimensional trait space using Gauss-Legendre
    product quadrature.

    Parameters
    ----------
    trait_func : callable
        Function f(traits) where traits is a 1D array.
    trait_ranges : list of (min, max)
        Integration bounds for each trait dimension.
    orders : list of int
        Quadrature order in each dimension.

    Returns
    -------
    integral : float
    """
    from numpy.polynomial.legendre import leggauss

    rules = []
    scales = []
    for (a, b), n in zip(trait_ranges, orders):
        xi, wi = leggauss(n)
        # Map from [-1, 1] to [a, b]
        xi_mapped = 0.5 * (b - a) * xi + 0.5 * (a + b)
        wi_mapped = 0.5 * (b - a) * wi
        rules.append((xi_mapped, wi_mapped))
        scales.append(b - a)

    points, weights = construct_product_rule(rules)
    total = 0.0
    for i in range(len(points)):
        total += weights[i] * trait_func(points[i])
    return float(total)
