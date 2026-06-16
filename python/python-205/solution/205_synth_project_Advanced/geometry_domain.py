"""
geometry_domain.py - Domain Geometry via Signed Distance Functions

This module defines complex spatial domains for the reaction-diffusion
system using signed distance functions (SDFs). The domain geometry
affects pattern formation (Turing patterns) and must be precisely
represented for accurate surrogate construction.

Mathematical Framework
----------------------
A signed distance function d(x) for domain Omega satisfies:
  d(x) < 0  if x in interior(Omega)
  d(x) = 0  if x on boundary(Omega)
  d(x) > 0  if x in exterior(Omega)
  |grad d| = 1 a.e. (Eikonal equation)

Boolean operations on SDFs:
  Union:        d_{A union B}(x) = min(d_A(x), d_B(x))
  Intersection: d_{A cap B}(x)   = max(d_A(x), d_B(x))
  Difference:   d_{A \ B}(x)     = max(d_A(x), -d_B(x))

Level-set evolution of domain boundaries follows:
  d(d)/dt + F |grad d| = 0
where F is the normal velocity.

References
----------
[1] Osher & Fedkiw, "Level Set Methods", Cambridge, 2003.
[2] Persson & Strang, "A Simple Mesh Generator in MATLAB", SIAM Rev., 2004.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List, Optional, Callable
import math


# ---------------------------------------------------------------------------
# Primitive SDFs
# ---------------------------------------------------------------------------

def sdf_circle(x: NDArray, y: NDArray, xc: float, yc: float,
               r: float) -> NDArray:
    """
    Signed distance to circle centered at (xc, yc) with radius r.

    d(x,y) = sqrt((x - xc)^2 + (y - yc)^2) - r

    Negative inside, positive outside.
    """
    return np.sqrt((x - xc) ** 2 + (y - yc) ** 2) - r


def sdf_rectangle(x: NDArray, y: NDArray, x1: float, x2: float,
                  y1: float, y2: float) -> NDArray:
    """
    Signed distance to rectangle [x1, x2] x [y1, y2].

    d(x,y) = -min(min(min(y - y1, y2 - y), x - x1), x2 - x)

    This is an approximation (not exact SDF near corners) but
    sufficient for domain membership testing.
    """
    dx = np.maximum(x1 - x, x - x2)
    dy = np.maximum(y1 - y, y - y2)
    # Inside: both dx < 0 and dy < 0
    # Use proper signed distance
    inside = np.minimum(-dx, -dy)
    outside = np.sqrt(np.maximum(dx, 0) ** 2 + np.maximum(dy, 0) ** 2)
    return np.where(inside > 0, inside, -outside)


def sdf_segment_distance(px: NDArray, py: NDArray,
                         ax: float, ay: float,
                         bx: float, by: float) -> NDArray:
    """
    Distance from points (px, py) to line segment (a, b).

    Projection parameter: t = dot(P-A, B-A) / dot(B-A, B-A)
    Clamped to [0, 1] for segment (not line).

    d = |P - (A + t*(B-A))|
    """
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay

    ab_sq = abx * abx + aby * aby
    if ab_sq < 1e-30:
        return np.sqrt(apx ** 2 + apy ** 2)

    t = np.clip((apx * abx + apy * aby) / ab_sq, 0.0, 1.0)
    proj_x = ax + t * abx
    proj_y = ay + t * aby

    return np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)


def sdf_polygon(x: NDArray, y: NDArray,
                vertices: NDArray) -> NDArray:
    """
    Signed distance to polygon defined by vertices.

    Uses minimum distance to edges with sign determined by
    point-in-polygon test (ray casting algorithm).

    Parameters
    ----------
    vertices : ndarray(nv, 2)
        Polygon vertices in order (first != last, automatically closed).
    """
    nv = len(vertices)
    if nv < 3:
        return np.full_like(x, float('inf'))

    # Compute distance to each edge
    min_dist = np.full_like(x, float('inf'))
    for i in range(nv):
        j = (i + 1) % nv
        d = sdf_segment_distance(x, y,
                                 vertices[i, 0], vertices[i, 1],
                                 vertices[j, 0], vertices[j, 1])
        min_dist = np.minimum(min_dist, d)

    # Sign: negative inside (ray casting)
    inside = _point_in_polygon(x, y, vertices)
    return np.where(inside, -min_dist, min_dist)


def _point_in_polygon(x: NDArray, y: NDArray,
                      vertices: NDArray) -> NDArray:
    """
    Ray-casting algorithm for point-in-polygon test.

    Casts a ray in the +x direction and counts crossings.
    Odd crossings = inside, even = outside.
    """
    nv = len(vertices)
    inside = np.zeros_like(x, dtype=bool)

    j = nv - 1
    for i in range(nv):
        xi, yi = vertices[i]
        xj, yj = vertices[j]

        cond1 = (yi > y) != (yj > y)
        slope = (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi
        cond2 = x < slope

        inside = inside ^ (cond1 & cond2)
        j = i

    return inside


# ---------------------------------------------------------------------------
# Boolean CSG operations
# ---------------------------------------------------------------------------

def sdf_union(d1: NDArray, d2: NDArray) -> NDArray:
    """Union: d = min(d1, d2)."""
    return np.minimum(d1, d2)


def sdf_intersection(d1: NDArray, d2: NDArray) -> NDArray:
    """Intersection: d = max(d1, d2)."""
    return np.maximum(d1, d2)


def sdf_difference(d1: NDArray, d2: NDArray) -> NDArray:
    """Difference A \\ B: d = max(d1, -d2)."""
    return np.maximum(d1, -d2)


# ---------------------------------------------------------------------------
# Domain parameterization for UQ
# ---------------------------------------------------------------------------

class DomainGeometry:
    """
    Parameterized domain geometry for reaction-diffusion simulations.

    The domain shape depends on uncertain parameters xi, which
    affect pattern formation. The surrogate must capture how
    domain perturbations influence QoIs.

    Parameters
    ----------
    domain_type : str
        Type of domain: 'rectangle', 'annulus', 'polygon_param'.
    params : dict
        Domain parameters (may be uncertain).
    """

    def __init__(self, domain_type: str = 'rectangle',
                 params: Optional[dict] = None):
        self.domain_type = domain_type
        self.params = params or {}
        self._setup_domain()

    def _setup_domain(self):
        """Configure domain based on type and parameters."""
        if self.domain_type == 'rectangle':
            self.x_range = (self.params.get('x_min', 0.0),
                            self.params.get('x_max', 1.0))
            self.y_range = (self.params.get('y_min', 0.0),
                            self.params.get('y_max', 1.0))
        elif self.domain_type == 'annulus':
            self.center = self.params.get('center', (0.5, 0.5))
            self.r_inner = self.params.get('r_inner', 0.1)
            self.r_outer = self.params.get('r_outer', 0.4)
        elif self.domain_type == 'polygon_param':
            self.n_vertices = self.params.get('n_vertices', 6)
            self.perturbation = self.params.get('perturbation', 0.0)

    def make_mesh(self, nx: int, ny: int) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Create computational mesh over the domain.

        Returns
        -------
        x, y : ndarray(ny, nx)
            Meshgrid coordinates.
        mask : ndarray(ny, nx)
            Boolean mask: True inside domain, False outside.
        """
        if self.domain_type == 'rectangle':
            x = np.linspace(self.x_range[0], self.x_range[1], nx)
            y = np.linspace(self.y_range[0], self.y_range[1], ny)
            X, Y = np.meshgrid(x, y)
            mask = np.ones_like(X, dtype=bool)
            return X, Y, mask

        elif self.domain_type == 'annulus':
            x = np.linspace(self.center[0] - self.r_outer,
                            self.center[0] + self.r_outer, nx)
            y = np.linspace(self.center[1] - self.r_outer,
                            self.center[1] + self.r_outer, ny)
            X, Y = np.meshgrid(x, y)
            d_outer = sdf_circle(X, Y, self.center[0], self.center[1],
                                 self.r_outer)
            d_inner = sdf_circle(X, Y, self.center[0], self.center[1],
                                 self.r_inner)
            d_domain = sdf_difference(d_outer, d_inner)
            mask = d_domain < 0
            return X, Y, mask

        elif self.domain_type == 'polygon_param':
            # Regular polygon with perturbed vertices
            angles = np.linspace(0, 2 * np.pi, self.n_vertices, endpoint=False)
            r_base = 0.4
            vx = r_base * np.cos(angles) + 0.5
            vy = r_base * np.sin(angles) + 0.5
            # Apply perturbation
            pert = self.perturbation * np.sin(3 * angles)
            vx += pert * np.cos(angles)
            vy += pert * np.sin(angles)
            vertices = np.column_stack([vx, vy])

            x = np.linspace(0.0, 1.0, nx)
            y = np.linspace(0.0, 1.0, ny)
            X, Y = np.meshgrid(x, y)
            d = sdf_polygon(X, Y, vertices)
            mask = d < 0
            return X, Y, mask

        return np.array([[]]), np.array([[]]), np.array([[]])

    def sdf_eval(self, x: NDArray, y: NDArray) -> NDArray:
        """Evaluate signed distance function at points (x, y)."""
        if self.domain_type == 'rectangle':
            d = sdf_rectangle(x, y, self.x_range[0], self.x_range[1],
                              self.y_range[0], self.y_range[1])
        elif self.domain_type == 'annulus':
            d_outer = sdf_circle(x, y, self.center[0], self.center[1],
                                 self.r_outer)
            d_inner = sdf_circle(x, y, self.center[0], self.center[1],
                                 self.r_inner)
            d = sdf_difference(d_outer, d_inner)
        elif self.domain_type == 'polygon_param':
            angles = np.linspace(0, 2 * np.pi, self.n_vertices, endpoint=False)
            r_base = 0.4
            vx = r_base * np.cos(angles) + 0.5
            vy = r_base * np.sin(angles) + 0.5
            pert = self.perturbation * np.sin(3 * angles)
            vx += pert * np.cos(angles)
            vy += pert * np.sin(angles)
            vertices = np.column_stack([vx, vy])
            d = sdf_polygon(x, y, vertices)
        else:
            d = np.full_like(x, 0.0)
        return d

    def domain_area_estimate(self, nx: int = 200, ny: int = 200) -> float:
        """Estimate domain area via pixel counting."""
        _, _, mask = self.make_mesh(nx, ny)
        total_pixels = mask.size
        interior_pixels = mask.sum()
        if self.domain_type == 'rectangle':
            total_area = ((self.x_range[1] - self.x_range[0]) *
                          (self.y_range[1] - self.y_range[0]))
        else:
            total_area = 1.0
        return total_area * interior_pixels / total_pixels
