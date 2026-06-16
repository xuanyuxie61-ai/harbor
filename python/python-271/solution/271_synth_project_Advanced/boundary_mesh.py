# -*- coding: utf-8 -*-
"""
Boundary mesh construction and Boolean region predicates for the
finite-element discretisation of the Ginzburg-Landau functional
associated with the TFIM order parameter.

Merged from the ``109_boundary_word_right`` seed project, whose
boolean/word meshing utilities we adapt:

    boolean_to_string   -> region label encoding
    polygon_contains_point -> point-in-domain test for the
                              Ginzburg-Landau simulation box
    rectangle_nodes     -> structured boundary-node generator
    region_contains_tile -> tile acceptance for adaptive refinement

In the quantum-phase-transition context, the domain is the 2D
(space x imaginary-time) cylinder  (x, tau) in [0, L] x [0, beta]
with Neumann boundary conditions on all four sides.  The boundary
mesh provides:
  - Node coordinates (for FEM assembly)
  - Edge connectivity (for boundary integrals)
  - Tile membership (for adaptive h-refinement near the QCP)
"""

from __future__ import annotations
from typing import Tuple, List
import numpy as np
try:
    from . import constants as C
except ImportError:
    import constants as C


# ---------------------------------------------------------------------------
# Boolean region encoding
# ---------------------------------------------------------------------------
def region_code(kind: str) -> int:
    """Encode a boundary region as an integer.

    The encoding mirrors the boolean_to_string utility:
        'left'   -> 1
        'right'  -> 2
        'bottom' -> 3
        'top'    -> 4
        'interior' -> 0
    """
    return {"interior": 0, "left": 1, "right": 2,
             "bottom": 3, "top": 4}[kind]


def region_name(code: int) -> str:
    names = {0: "interior", 1: "left", 2: "right", 3: "bottom", 4: "top"}
    return names.get(code, "unknown")


# ---------------------------------------------------------------------------
# Rectangle nodes (structured grid)
# ---------------------------------------------------------------------------
def rectangle_nodes(nx: int, ny: int, Lx: float = 1.0,
                     Ly: float = 1.0) -> np.ndarray:
    """Return a (nx * ny, 2) array of node coordinates on a structured
    rectangular grid covering [0, Lx] x [0, Ly]."""
    if nx < 2 or ny < 2:
        raise ValueError("need nx, ny >= 2")
    xs = np.linspace(0.0, Lx, nx)
    ys = np.linspace(0.0, Ly, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel()])


def boundary_flags(nodes: np.ndarray, Lx: float, Ly: float,
                    tol: float = 1.0e-10) -> np.ndarray:
    """Return an integer array of length n_nodes with the region code
    of each node (multiple regions OR'd together if corner)."""
    x, y = nodes[:, 0], nodes[:, 1]
    on_left = np.abs(x) < tol
    on_right = np.abs(x - Lx) < tol
    on_bottom = np.abs(y) < tol
    on_top = np.abs(y - Ly) < tol
    code = np.zeros(len(nodes), dtype=int)
    code[on_left] |= 1
    code[on_right] |= 2
    code[on_bottom] |= 4
    code[on_top] |= 8
    return code


# ---------------------------------------------------------------------------
# Point-in-polygon (for curved boundaries)
# ---------------------------------------------------------------------------
def polygon_contains_point(polygon: np.ndarray,
                            point: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test.

    polygon : (N, 2) array of vertices in counter-clockwise order.
    point   : (2,) array.
    """
    x, y = point[0], point[1]
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + C.EPS_NUM) + xi):
            inside = not inside
        j = i
    return inside


# ---------------------------------------------------------------------------
# Tile acceptance for adaptive refinement
# ---------------------------------------------------------------------------
def tile_in_region(tile_center: np.ndarray, tile_half: float,
                    region_polygon: np.ndarray) -> bool:
    """Accept a square tile if it is fully inside the region.
    We test all four corners."""
    cx, cy = tile_center
    h = tile_half
    corners = np.array([[cx - h, cy - h], [cx + h, cy - h],
                         [cx + h, cy + h], [cx - h, cy + h]])
    return all(polygon_contains_point(region_polygon, c) for c in corners)


# ---------------------------------------------------------------------------
# Edge connectivity (structured grid)
# ---------------------------------------------------------------------------
def rectangle_edges(nx: int, ny: int) -> np.ndarray:
    """Return an (E, 2) array of node-index edges for a structured
    rectangular grid (triangulated by splitting each quad into 2
    triangles; we return the diagonal edges only for the FEM)."""
    edges = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a = i * ny + j
            b = (i + 1) * ny + j
            c = (i + 1) * ny + (j + 1)
            d = i * ny + (j + 1)
            edges.append([a, c])
            edges.append([a, d])
    return np.asarray(edges, dtype=int)


# ---------------------------------------------------------------------------
# Neumann boundary measure
# ---------------------------------------------------------------------------
def boundary_measure(Lx: float, Ly: float) -> float:
    """Total length of the boundary of the rectangle [0, Lx] x [0, Ly]."""
    return 2.0 * (Lx + Ly)


def neumann_test_function(nodes: np.ndarray, Lx: float, Ly: float) -> np.ndarray:
    """A test function satisfying Neumann BCs on all four sides:
        phi(x, y) = cos(pi x / Lx) * cos(pi y / Ly)
    whose normal derivative vanishes on the boundary by construction.
    Used to sanity-check FEM Neumann assembly."""
    x, y = nodes[:, 0], nodes[:, 1]
    return np.cos(C.PI * x / Lx) * np.cos(C.PI * y / Ly)
