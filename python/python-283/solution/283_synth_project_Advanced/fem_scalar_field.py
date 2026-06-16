"""
fem_scalar_field.py
===================
Numerical analysis (no visualization) of scalar fields on triangulated
meshes, ported from 414_fem2d_scalar_display.

In the perovskite defect-state context, we compute the electrostatic
potential phi(x, y) on a 2D triangular mesh of the perovskite active layer.
The scalar field may represent:
    - electrostatic potential (from Poisson solver)
    - electron or hole density (from drift-diffusion)
    - defect charge density rho_def
    - local recombination rate R_SRH

Analysis tasks (replacing the visualization of 414):
    (1) Read node coordinates and element connectivity
    (2) Compute min / max / mean / variance of the scalar field
    (3) Compute the L2 and H1 semi-norms of the field
    (4) Compute the gradient field ||grad phi|| (electric field magnitude)
    (5) Detect extrema (local maxima / minima) of the field
    (6) Compute contour-level statistics (fraction of area above threshold)

P1 (linear) finite element basis
---------------------------------
On a triangle T with vertices (x_1, y_1), (x_2, y_2), (x_3, y_3), the
P1 basis functions are:
    phi_i(x, y) = (a_i + b_i x + c_i y) / (2 A)
where A is the signed area of T and
    b_1 = y_2 - y_3,   c_1 = x_3 - x_2
    b_2 = y_3 - y_1,   c_2 = x_1 - x_3
    b_3 = y_1 - y_2,   c_3 = x_2 - x_1
The gradient of phi_i is constant on T: grad phi_i = (b_i, c_i) / (2A).
"""

from __future__ import annotations
import math
from typing import Tuple, List, Optional

import numpy as np
from numpy.typing import NDArray


# ============================================================================
# Triangle geometry
# ============================================================================
def triangle_area(p1: NDArray, p2: NDArray, p3: NDArray) -> float:
    """Signed area of triangle with vertices p1, p2, p3.
    A = 0.5 * ((x2-x1)(y3-y1) - (x3-x1)(y2-y1))."""
    return 0.5 * ((p2[0] - p1[0]) * (p3[1] - p1[1])
                  - (p3[0] - p1[0]) * (p2[1] - p1[1]))


def triangle_quality(p1: NDArray, p2: NDArray, p3: NDArray) -> float:
    """Compute the quality of a triangle as the ratio of inradius to
    circumradius: q = 2 r_in / r_circ in [0, 1].
    Equilateral triangle has q = 1."""
    a_ = np.linalg.norm(p2 - p3)
    b_ = np.linalg.norm(p3 - p1)
    c_ = np.linalg.norm(p1 - p2)
    s = 0.5 * (a_ + b_ + c_)
    A = abs(triangle_area(p1, p2, p3))
    if A < 1e-30:
        return 0.0
    r_in = A / s
    r_circ = (a_ * b_ * c_) / (4.0 * A)
    return float(2.0 * r_in / r_circ) if r_circ > 1e-30 else 0.0


# ============================================================================
# Scalar-field analysis on a triangulated mesh
# ============================================================================
def field_statistics(values: NDArray) -> dict:
    """Compute basic statistics of a scalar field on the nodes."""
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "L2_discrete": float(np.sqrt(np.mean(values ** 2))),
        "Linf": float(np.max(np.abs(values))),
    }


def field_gradient_magnitude(nodes: NDArray, elements: NDArray,
                             values: NDArray) -> NDArray:
    """Compute the magnitude of the gradient ||grad f|| on each element
    for a P1 scalar field f with nodal values.
    Returns an array of length n_elements.
    """
    n_elem = elements.shape[0]
    grad_mag = np.zeros(n_elem)
    for e in range(n_elem):
        i1, i2, i3 = elements[e]
        p1, p2, p3 = nodes[i1], nodes[i2], nodes[i3]
        f1, f2, f3 = values[i1], values[i2], values[i3]
        # Area
        A = triangle_area(p1, p2, p3)
        if abs(A) < 1e-30:
            continue
        # Gradient components
        b = np.array([p2[1] - p3[1], p3[1] - p1[1], p1[1] - p2[1]])
        c = np.array([p3[0] - p2[0], p1[0] - p3[0], p2[0] - p1[0]])
        grad_x = (b[0] * f1 + b[1] * f2 + b[2] * f3) / (2.0 * A)
        grad_y = (c[0] * f1 + c[1] * f2 + c[2] * f3) / (2.0 * A)
        grad_mag[e] = math.sqrt(grad_x ** 2 + grad_y ** 2)
    return grad_mag


def field_L2_norm(nodes: NDArray, elements: NDArray,
                  values: NDArray) -> float:
    """Compute the L2 norm of a P1 field on the triangulation.
    Uses the mass-matrix formula: Integral f^2 ~ sum_T (A_T / 12)
        * (f_1^2 + f_2^2 + f_3^2 + f_1 f_2 + f_2 f_3 + f_3 f_1)
    """
    L2_sq = 0.0
    for e in range(elements.shape[0]):
        i1, i2, i3 = elements[e]
        A = abs(triangle_area(nodes[i1], nodes[i2], nodes[i3]))
        f1, f2, f3 = values[i1], values[i2], values[i3]
        integral_T = (A / 12.0) * (f1 ** 2 + f2 ** 2 + f3 ** 2
                                   + f1 * f2 + f2 * f3 + f3 * f1)
        L2_sq += integral_T
    return math.sqrt(max(L2_sq, 0.0))


def field_H1_seminorm(nodes: NDArray, elements: NDArray,
                      values: NDArray) -> float:
    """Compute the H1 seminorm |f|_{H1} = ||grad f||_{L2}.
    Integral_T |grad f|^2 = A_T |grad f|^2 (grad is constant per element).
    """
    H1_sq = 0.0
    for e in range(elements.shape[0]):
        i1, i2, i3 = elements[e]
        p1, p2, p3 = nodes[i1], nodes[i2], nodes[i3]
        A = abs(triangle_area(p1, p2, p3))
        f1, f2, f3 = values[i1], values[i2], values[i3]
        b = np.array([p2[1] - p3[1], p3[1] - p1[1], p1[1] - p2[1]])
        c = np.array([p3[0] - p2[0], p1[0] - p3[0], p2[0] - p1[0]])
        grad_x = (b[0] * f1 + b[1] * f2 + b[2] * f3) / (2.0 * A + 1e-30)
        grad_y = (c[0] * f1 + c[1] * f2 + c[2] * f3) / (2.0 * A + 1e-30)
        H1_sq += A * (grad_x ** 2 + grad_y ** 2)
    return math.sqrt(max(H1_sq, 0.0))


# ============================================================================
# Contour-level statistics
# ============================================================================
def contour_coverage(nodes: NDArray, elements: NDArray,
                     values: NDArray, threshold: float) -> float:
    """Compute the fraction of the mesh area where f(x) >= threshold.
    Uses element-wise sampling at centroids."""
    total_area = 0.0
    above_area = 0.0
    for e in range(elements.shape[0]):
        i1, i2, i3 = elements[e]
        A = abs(triangle_area(nodes[i1], nodes[i2], nodes[i3]))
        total_area += A
        f_centroid = (values[i1] + values[i2] + values[i3]) / 3.0
        if f_centroid >= threshold:
            above_area += A
    return above_area / max(total_area, 1e-30)


# ============================================================================
# Local extrema detection
# ============================================================================
def find_local_extrema(nodes: NDArray, elements: NDArray,
                       values: NDArray) -> Tuple[List[int], List[int]]:
    """Find local maxima and minima of the scalar field.
    A node is a local maximum if its value exceeds all adjacent nodes.
    Returns lists of node indices."""
    n_nodes = nodes.shape[0]
    # Build adjacency
    adj: list = [set() for _ in range(n_nodes)]
    for e in range(elements.shape[0]):
        i1, i2, i3 = elements[e]
        adj[i1].add(i2); adj[i1].add(i3)
        adj[i2].add(i1); adj[i2].add(i3)
        adj[i3].add(i1); adj[i3].add(i2)
    maxima = []
    minima = []
    for i in range(n_nodes):
        if not adj[i]:
            continue
        neighbors = list(adj[i])
        vals = values[neighbors]
        if np.all(values[i] >= vals) and np.any(values[i] > vals):
            maxima.append(i)
        if np.all(values[i] <= vals) and np.any(values[i] < vals):
            minima.append(i)
    return maxima, minima


# ============================================================================
# Driver: analyze the 2D electrostatic potential
# ============================================================================
def analyze_2d_potential(nx: int = 16, ny: int = 16) -> dict:
    """Build a simple triangulation of [0,1]^2 with a synthetic potential
    field and run all analyses. Returns a dict of results."""
    # Build a structured triangular mesh (split each quad into 2 triangles)
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    xx, yy = np.meshgrid(xs, ys)
    nodes = np.column_stack([xx.ravel(), yy.ravel()])
    elements = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n0 = j * nx + i
            n1 = n0 + 1
            n2 = n0 + nx + 1
            n3 = n0 + nx
            elements.append([n0, n1, n2])
            elements.append([n0, n2, n3])
    elements = np.asarray(elements, dtype=int)
    # Synthetic potential: phi(x, y) = V_net * (1 - x) + 0.3 sin(2 pi x) cos(2 pi y)
    from perovskite_constants import V_NET
    values = V_NET * (1.0 - nodes[:, 0]) + 0.3 * np.sin(
        2.0 * math.pi * nodes[:, 0]) * np.cos(2.0 * math.pi * nodes[:, 1])
    # Run analyses
    stats = field_statistics(values)
    grad_mag = field_gradient_magnitude(nodes, elements, values)
    L2 = field_L2_norm(nodes, elements, values)
    H1 = field_H1_seminorm(nodes, elements, values)
    coverage = contour_coverage(nodes, elements, values,
                                threshold=0.5 * V_NET)
    maxima, minima = find_local_extrema(nodes, elements, values)
    # Triangle quality
    qualities = []
    for e in range(elements.shape[0]):
        i1, i2, i3 = elements[e]
        q = triangle_quality(nodes[i1], nodes[i2], nodes[i3])
        qualities.append(q)
    return {
        "n_nodes": nodes.shape[0],
        "n_elements": elements.shape[0],
        "field_stats": stats,
        "grad_mag_stats": field_statistics(grad_mag),
        "L2_norm": L2,
        "H1_seminorm": H1,
        "contour_coverage_half_Vbi": coverage,
        "n_local_maxima": len(maxima),
        "n_local_minima": len(minima),
        "triangle_quality_min": min(qualities),
        "triangle_quality_mean": float(np.mean(qualities)),
    }
