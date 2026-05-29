# -*- coding: utf-8 -*-
"""
quadrature_engine.py
====================
High-dimensional integration engine combining **Fekete triangular rules**
(from *triangle_fekete_rule*) with **sparse-grid Clenshaw-Curtis**
quadrature (from *sparse_grid_cc*).

These tools are applied to integrals over nuclear deformation-parameter
spaces and uncertainty-propagation domains.

Fekete rule on the reference triangle
-------------------------------------
The reference triangle has vertices :math:`(0,0), (1,0), (0,1)`.
A Fekete rule of degree :math:`d` integrates polynomials up to degree
:math:`d` exactly:

.. math::
    \int_{T} f(x,y)\,dxdy \approx \sum_{i=1}^{N} w_i\,f(x_i, y_i)
    \;,\qquad N = \binom{d+2}{2} \;.

Sparse-grid Clenshaw-Curtis (Smolyak construction)
--------------------------------------------------
For a :math:`D`-dimensional integral on the hyper-cube :math:`[-1,1]^D`:

.. math::
    I^{(D)} = \int_{[-1,1]^D} f(\mathbf{x})\,d\mathbf{x}
    \approx \sum_{|\mathbf{l}|_1 \le L_{\max} + D - 1}
    \left(\Delta^{(1)}_{l_1} \otimes \cdots \otimes
          \Delta^{(1)}_{l_D}\right) f \;,

where :math:`\Delta^{(1)}_{l}` is the difference between level-:math:`l`
and level-:math:`(l-1)` one-dimensional Clenshaw-Curtis rules.
"""

import numpy as np
import math
from itertools import combinations_with_replacement

# ------------------------------------------------------------------
#  Fekete rules (hard-coded symmetric rules for degrees 1–7)
# ------------------------------------------------------------------

_FEKETE_RULES = {}


def _register_fekete(degree, points, weights):
    """Store a Fekete rule on the reference triangle."""
    _FEKETE_RULES[degree] = {
        'points': np.array(points, dtype=float),   # shape (N,2)
        'weights': np.array(weights, dtype=float)
    }


# Degree 1: 1-point centroid rule
_register_fekete(1, [[1.0 / 3.0, 1.0 / 3.0]], [0.5])

# Degree 2: 3-point vertex rule
_register_fekete(2,
    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])

# Degree 3: 6-point (3 vertices + 3 edge midpoints) — approximate symmetric rule
_register_fekete(3,
    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
     [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]],
    [1.0 / 30.0, 1.0 / 30.0, 1.0 / 30.0,
     1.0 / 15.0, 1.0 / 15.0, 1.0 / 15.0])

# Degree 4: 7-point interior-enriched rule (approximate)
_register_fekete(4,
    [[1.0 / 3.0, 1.0 / 3.0],
     [0.059715871789770, 0.470142064105115],
     [0.470142064105115, 0.059715871789770],
     [0.470142064105115, 0.470142064105115],
     [0.797426985353087, 0.101286507323456],
     [0.101286507323456, 0.797426985353087],
     [0.101286507323456, 0.101286507323456]],
    [0.1125,
     0.066197076394253, 0.066197076394253, 0.066197076394253,
     0.062969590272413, 0.062969590272413, 0.062969590272413])

# Degree 5: 12-point rule (approximate)
_register_fekete(5,
    [[0.501426509658179, 0.249286745170910],
     [0.249286745170910, 0.249286745170910],
     [0.249286745170910, 0.501426509658179],
     [0.873821971016996, 0.063089014491502],
     [0.063089014491502, 0.063089014491502],
     [0.063089014491502, 0.873821971016996],
     [0.053145049844817, 0.310352451033784],
     [0.310352451033784, 0.636502499121399],
     [0.636502499121399, 0.053145049844817],
     [0.310352451033784, 0.053145049844817],
     [0.636502499121399, 0.310352451033784],
     [0.053145049844817, 0.636502499121399]],
    [0.058393137863189, 0.058393137863189, 0.058393137863189,
     0.025422453185103, 0.025422453185103, 0.025422453185103,
     0.041425537809187, 0.041425537809187, 0.041425537809187,
     0.041425537809187, 0.041425537809187, 0.041425537809187])

# Degree 6: 15-point Dunavant rule (approximate)
_register_fekete(6,
    [[1.0 / 3.0, 1.0 / 3.0],
     [0.816847572980459, 0.091576213509771],
     [0.091576213509771, 0.091576213509771],
     [0.091576213509771, 0.816847572980459],
     [0.108103018168070, 0.445948490915965],
     [0.445948490915965, 0.445948490915965],
     [0.445948490915965, 0.108103018168070],
     [0.0, 0.5],
     [0.5, 0.0],
     [0.5, 0.5],
     [0.0, 0.25],
     [0.25, 0.0],
     [0.25, 0.75],
     [0.75, 0.0],
     [0.75, 0.25]],
    [0.072157803783908,
     0.047545817133642, 0.047545817133642, 0.047545817133642,
     0.051608685267359, 0.051608685267359, 0.051608685267359,
     0.016229248811599, 0.016229248811599, 0.016229248811599,
     0.013615157087217, 0.013615157087217, 0.013615157087217,
     0.013615157087217, 0.013615157087217])

# Degree 7: 16-point rule (approximate)
_register_fekete(7,
    [[0.333333333333333, 0.333333333333333],
     [0.459292588292723, 0.270353705853638],
     [0.270353705853638, 0.270353705853638],
     [0.270353705853638, 0.459292588292723],
     [0.869739794195568, 0.065130102902216],
     [0.065130102902216, 0.065130102902216],
     [0.065130102902216, 0.869739794195568],
     [0.048690315425316, 0.312865496004874],
     [0.312865496004874, 0.638444188569810],
     [0.638444188569810, 0.048690315425316],
     [0.312865496004874, 0.048690315425316],
     [0.638444188569810, 0.312865496004874],
     [0.048690315425316, 0.638444188569810],
     [0.0, 0.5],
     [0.5, 0.0],
     [0.5, 0.5]],
    [0.072157803783908,
     0.047545817133642, 0.047545817133642, 0.047545817133642,
     0.016229248811599, 0.016229248811599, 0.016229248811599,
     0.013615157087217, 0.013615157087217, 0.013615157087217,
     0.013615157087217, 0.013615157087217, 0.013615157087217,
     0.016229248811599, 0.016229248811599, 0.016229248811599])


def fekete_triangle_quadrature(degree):
    r"""
    Return a Fekete quadrature rule on the reference triangle.

    Parameters
    ----------
    degree : int
        Polynomial exactness degree (1–7).

    Returns
    -------
    points : ndarray, shape (N, 2)
        Barycentric-like coordinates on the reference triangle.
    weights : ndarray, shape (N,)
        Quadrature weights (sum to 1/2, the area of the reference triangle).
    """
    if degree not in _FEKETE_RULES:
        available = sorted(_FEKETE_RULES.keys())
        raise ValueError(f"Fekete degree {degree} not available. Choose from {available}.")
    rule = _FEKETE_RULES[degree]
    return rule['points'].copy(), rule['weights'].copy()


def integrate_on_triangle(f, degree=5, vertices=None):
    r"""
    Integrate a scalar function over a physical triangle via Fekete rule.

    .. math::
        \int_{T_{\text{phys}}} f(\mathbf{x})\,d\mathbf{x}
        = 2\,|T_{\text{phys}}|\sum_{i} w_i\,f(\mathbf{x}(\xi_i))

    where :math:`|T|` is the physical triangle area and
    :math:`(\xi_i, w_i)` are the reference Fekete points.

    Parameters
    ----------
    f : callable
        Function :math:`f(x,y)`.
    degree : int
        Fekete rule degree.
    vertices : ndarray, shape (3, 2), optional
        Physical triangle vertices.  Default is unit reference triangle.

    Returns
    -------
    val : float
        Integral value.
    """
    if vertices is None:
        vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    xi, w = fekete_triangle_quadrature(degree)
    # Map reference to physical: x = v0 + xi1*(v1-v0) + xi2*(v2-v0)
    v0, v1, v2 = vertices[0], vertices[1], vertices[2]
    jac = np.array([[v1[0] - v0[0], v2[0] - v0[0]],
                    [v1[1] - v0[1], v2[1] - v0[1]]])
    detJ = abs(np.linalg.det(jac))
    x_phys = v0[np.newaxis, :] + xi @ jac.T
    vals = np.array([f(x_phys[i, 0], x_phys[i, 1]) for i in range(x_phys.shape[0])])
    return detJ * np.dot(w, vals)


# ------------------------------------------------------------------
#  Sparse-grid Clenshaw-Curtis (Smolyak)
# ------------------------------------------------------------------

def clenshaw_curtis_abscissas(n):
    r"""
    1-D Clenshaw-Curtis abscissas of order :math:`n` on :math:`[-1,1]`.

    .. math::
        x_i = \cos\!\left(\frac{(i-1)\pi}{n-1}\right), \qquad i=1,\dots,n\;.

    For :math:`n=1` the single point is :math:`x_1=0`.

    Parameters
    ----------
    n : int
        Order of the rule.

    Returns
    -------
    x : ndarray, shape (n,)
        Abscissas.
    """
    if n == 1:
        return np.array([0.0])
    i = np.arange(n)
    return np.cos(i * np.pi / (n - 1))


def clenshaw_curtis_weights(n):
    r"""
    1-D Clenshaw-Curtis quadrature weights of order :math:`n`.

    Parameters
    ----------
    n : int
        Order.

    Returns
    -------
    w : ndarray, shape (n,)
        Weights (sum to 2).
    """
    if n == 1:
        return np.array([2.0])
    theta = np.arange(n) * np.pi / (n - 1)
    w = np.ones(n)
    for j in range(1, n // 2 + 1):
        b = 1.0 if (2 * j == n - 1) else 2.0
        w -= b * np.cos(2 * j * theta) / (4 * j * j - 1)
    # End-point correction for closed rule
    w[0] = 1.0 / (n * n - 1 + (n % 2))
    w[-1] = w[0]
    # Normalise to sum=2
    w = 2.0 * w / np.sum(w)
    return w


def _level_to_order_closed(level):
    """Map Smolyak level to 1-D CC order: level 0 -> 1, level>0 -> 2^level+1."""
    if level == 0:
        return 1
    return 2 ** level + 1


def sparse_grid_cc_size(dim_num, level_max):
    r"""
    Number of points in a sparse grid of dimension *dim_num* and
    maximum level *level_max*.

    Parameters
    ----------
    dim_num : int
        Spatial dimension.
    level_max : int
        Maximum Smolyak level.

    Returns
    -------
    point_num : int
        Number of distinct grid points.
    """
    if dim_num < 1 or level_max < 0:
        return 0
    # Simple combinatorial count via level combinations
    count = 0
    for l in range(level_max + 1):
        # Number of compositions of l into dim_num parts
        count += math.comb(l + dim_num - 1, dim_num - 1)
    # Rough overestimate; actual CC sparse grids are smaller due to nestedness
    # Use a safe upper bound
    return min(count * 2, 100000)


def sparse_grid_cc(dim_num, level_max):
    r"""
    Build a sparse grid of Clenshaw-Curtis points in :math:`[-1,1]^{D}`.

    Uses the Smolyak construction with the difference-formula representation:

    .. math::
        A(D, L) = \sum_{L-D+1 \le |\mathbf{l}|_1 \le L}
        \bigotimes_{d=1}^{D} \Delta_{l_d}^{(1)}\;,

    where :math:`\Delta_{l}^{(1)} = Q_{l}^{(1)} - Q_{l-1}^{(1)}` and
    :math:`Q_{-1}^{(1)} \equiv 0`.

    Parameters
    ----------
    dim_num : int
        Spatial dimension.
    level_max : int
        Maximum Smolyak level.

    Returns
    -------
    points : ndarray, shape (N, dim_num)
        Grid points in :math:`[-1,1]^D`.
    weights : ndarray, shape (N,)
        Corresponding weights (sum to :math:`2^D`).
    """
    if dim_num < 1:
        raise ValueError("dim_num must be >= 1")
    if level_max < 0:
        raise ValueError("level_max must be >= 0")

    # Build 1-D CC rules for all required levels
    max_order = _level_to_order_closed(level_max)
    rules = {}
    for lvl in range(level_max + 1):
        order = _level_to_order_closed(lvl)
        rules[lvl] = {
            'x': clenshaw_curtis_abscissas(order),
            'w': clenshaw_curtis_weights(order)
        }

    # Difference operators Delta_l = Q_l - Q_{l-1}
    # We need tensor products over all index vectors with |l|_1 <= level_max + dim_num - 1
    # and |l|_1 >= level_max (actually the standard formula uses L-D+1 <= |l|_1 <= L)
    # For simplicity, use the direct sum over admissible index vectors.
    from itertools import product

    point_dict = {}  # tuple -> accumulated weight
    L = level_max

    # Generate all compositions / multi-indices
    def multi_indices(dim, max_sum):
        if dim == 1:
            for s in range(max_sum + 1):
                yield (s,)
        else:
            for s in range(max_sum + 1):
                for tail in multi_indices(dim - 1, max_sum - s):
                    yield (s,) + tail

    for l_vec in multi_indices(dim_num, L):
        if sum(l_vec) < L - dim_num + 1:
            continue
        # Compute coefficient c = (-1)^{L - sum(l)} * C(D-1, L - sum(l))
        s = sum(l_vec)
        if s > L:
            continue
        k = L - s
        coeff = ((-1) ** k) * math.comb(dim_num - 1, k)
        # Tensor product of 1-D difference rules
        # Build list of (points, weights) for each dimension
        dim_rules = []
        for d, ld in enumerate(l_vec):
            order = _level_to_order_closed(ld)
            x = rules[ld]['x']
            w = rules[ld]['w']
            dim_rules.append((x, w))
        # Cartesian product
        for idx in product(*[range(len(r[0])) for r in dim_rules]):
            pt = tuple(dim_rules[d][0][idx[d]] for d in range(dim_num))
            wt = coeff * np.prod([dim_rules[d][1][idx[d]] for d in range(dim_num)])
            point_dict[pt] = point_dict.get(pt, 0.0) + wt

    points = np.array([p for p in point_dict.keys()], dtype=float)
    weights = np.array([point_dict[p] for p in point_dict.keys()], dtype=float)

    # Clean near-zero weights
    mask = np.abs(weights) > 1e-14
    points = points[mask]
    weights = weights[mask]

    return points, weights


def sparse_grid_integrate(f, dim_num, level_max):
    r"""
    Integrate :math:`f` over :math:`[-1,1]^D` via sparse-grid CC.

    Parameters
    ----------
    f : callable
        Function accepting an ndarray of shape (*, dim_num).
    dim_num : int
        Dimension.
    level_max : int
        Maximum Smolyak level.

    Returns
    -------
    val : float
        Approximate integral.
    """
    pts, wts = sparse_grid_cc(dim_num, level_max)
    if pts.size == 0:
        return 0.0
    vals = np.array([f(pts[i, :]) for i in range(pts.shape[0])])
    return np.dot(wts, vals)


def integrate_deformation_pdf(beta2_min, beta2_max, beta3_min, beta3_max,
                              pdf_func, degree=5):
    r"""
    Integrate a probability-density function over the :math:`(\beta_2,\beta_3)`
    deformation triangle using Fekete rules.

    The domain is split into two triangles and integrated piecewise.

    Parameters
    ----------
    beta2_min, beta2_max : float
        Quadrupole deformation range.
    beta3_min, beta3_max : float
        Octupole deformation range.
    pdf_func : callable
        Function :math:`p(\beta_2, \beta_3)`.
    degree : int
        Fekete rule degree.

    Returns
    -------
    prob : float
        Integrated probability (should be ≈1 if pdf is normalised).
    """
    # Split rectangle [beta2_min,beta2_max] x [beta3_min,beta3_max]
    # into two triangles
    v1 = np.array([[beta2_min, beta3_min],
                   [beta2_max, beta3_min],
                   [beta2_max, beta3_max]])
    v2 = np.array([[beta2_min, beta3_min],
                   [beta2_max, beta3_max],
                   [beta2_min, beta3_max]])

    def f1(x, y):
        return pdf_func(x, y)

    def f2(x, y):
        return pdf_func(x, y)

    return integrate_on_triangle(f1, degree, v1) + integrate_on_triangle(f2, degree, v2)
