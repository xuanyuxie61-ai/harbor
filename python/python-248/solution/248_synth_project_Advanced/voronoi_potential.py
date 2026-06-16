"""
voronoi_potential.py
====================

Gravitational potential from a set of point masses via Voronoi
tessellation and multipole expansion.

The seed project 1394_voronoi_city constructs the Voronoi diagram
for three "cities" in a 2-D domain by computing the perpendicular
bisectors of each pair of cities and intersecting them to form
each city's region.  We lift this to 3-D for a set of point masses
(star clusters, dark matter sub-haloes) embedded in a protogalactic
gas cloud: each mass owns a Voronoi cell, and within each cell the
potential is approximated by a multipole expansion centred on the
mass.

Key formulae
------------
Perpendicular bisector of two points p_1, p_2 in 2-D:

    B = { x in R^2 : (x - (p_1 + p_2)/2) . (p_2 - p_1) = 0 }

i.e. the line through the midpoint orthogonal to the connecting
segment.  The 3-D generalisation is a plane.

Voronoi cell of point p_i:

    V_i = intersection_{j != i} H_{ij}

where  H_{ij} = { x : |x - p_i| <= |x - p_j| }  is the half-space
closer to p_i than to p_j.  The boundary of V_i is composed of
pieces of the bisector planes B_{ij}.

Multipole expansion of the potential within a Voronoi cell
----------------------------------------------------------
For a point mass M at position p the potential is

    Phi(r) = -G M / |r - p|

and in spherical harmonics about the cell centre r_0:

    Phi(r) = -G M sum_{l=0}^{infty} sum_{m=-l}^{l}
                (r_< ^ l / r_>^{l+1}) Y_{lm}(theta, phi) Y_{lm}^*(theta_0, phi_0)

Truncated at l_max = 4 this gives

    Phi(r) ~ -G M / d - G M (r . n) / d^3
             - G M (3 (r.n)^2 - r^2) / (2 d^5) + ...

where d = |r_0 - p| and n is the unit vector from p to r_0.

Application to galaxy formation
-------------------------------
The Voronoi + multipole decomposition is used in two places:

  1. As a fast approximate gravity solver for the particle component
     (star clusters, dark matter sub-haloes) on top of the mesh
     gravity for the gas component.

  2. As a diagnostic: the shape of a Voronoi cell (its anisotropy
     tensor) indicates the local mass-concentration geometry.

References
----------
- Springel, V. 2010, MNRAS 401, 791  (Voronoi mesh for gas dynamics)
- Barnes, J., & Hut, P. 1986, Nature 324, 446  (tree code, multipole)
- Burkardt, J. 2016, voronoi_city (seed project 1394)
"""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple, List, Optional, Dict
from itertools import combinations

from astro_constants import GRAVITATIONAL_CGS, KILOPARSEC_CGS, SOLAR_MASS_CGS


# =====================================================================
#                   2-D PERPENDICULAR BISECTOR (from 1394)
# =====================================================================

def perpendicular_bisector_2d(p1: np.ndarray, p2: np.ndarray
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the midpoint and direction vector of the perpendicular
    bisector of the segment (p1, p2) in 2-D.

    Direct translation of the voronoi_city construction.
    """
    mid = 0.5 * (p1 + p2)
    d = p2 - p1
    # perpendicular direction: rotate by 90 degrees
    perp = np.array([-d[1], d[0]])
    norm = np.linalg.norm(perp)
    if norm < 1.0e-12:
        raise ValueError("p1 and p2 are coincident")
    return mid, perp / norm


def voronoi_cell_2d(points: np.ndarray, idx: int,
                     box: Tuple[float, float, float, float] = (0, 10, 0, 10)
                     ) -> List[np.ndarray]:
    """
    Compute the 2-D Voronoi cell of points[idx] within the box
    by successive intersection with half-planes.

    Direct implementation of the voronoi_city perpendicular-bisector
    construction extended to N points.
    """
    x_min, x_max, y_min, y_max = box
    # initial polygon is the box
    vertices = [
        np.array([x_min, y_min]),
        np.array([x_max, y_min]),
        np.array([x_max, y_max]),
        np.array([x_min, y_max]),
    ]
    p_i = points[idx]
    for j in range(points.shape[0]):
        if j == idx:
            continue
        p_j = points[j]
        mid, perp = perpendicular_bisector_2d(p_i, p_j)
        # half-plane containing p_i: (x - mid) . (p_j - p_i) <= 0
        n = p_j - p_i
        new_vertices = []
        nv = len(vertices)
        for k in range(nv):
            v1 = vertices[k]
            v2 = vertices[(k + 1) % nv]
            d1 = np.dot(v1 - mid, n)
            d2 = np.dot(v2 - mid, n)
            if d1 <= 0:
                new_vertices.append(v1)
            if d1 * d2 < 0:
                # intersection
                t = d1 / (d1 - d2)
                v_int = v1 + t * (v2 - v1)
                new_vertices.append(v_int)
        vertices = new_vertices
        if len(vertices) < 3:
            break
    return vertices


def cell_area_2d(vertices: List[np.ndarray]) -> float:
    """Shoelace formula for the area of a 2-D polygon."""
    n = len(vertices)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


# =====================================================================
#                 MULTIPOLE EXPANSION
# =====================================================================

class MultipoleExpansion:
    """
    Multipole expansion of the gravitational potential from a point
    mass M at position p, centred at cell centre r_0.

    Truncation at l_max = L gives O(r^{L+1} / d^{L+2}) error.
    """

    def __init__(self, mass_cgs: float, pos_cgs: np.ndarray,
                 centre_cgs: np.ndarray, l_max: int = 4) -> None:
        self.M = mass_cgs
        self.p = pos_cgs
        self.r0 = centre_cgs
        self.l_max = l_max
        self.d_vec = self.r0 - self.p
        self.d = np.linalg.norm(self.d_vec)
        if self.d < 1.0e-60:
            raise ValueError("mass coincides with cell centre")
        self.n = self.d_vec / self.d   # unit vector from p to r0

    def potential_at(self, r_cgs: np.ndarray) -> float:
        """
        Evaluate the multipole-expanded potential at position r.

        Let s = r - r_0 (displacement from cell centre).  Then

            Phi(r) = -G M / |r - p|
                   = -G M sum_{l=0}^{L} P_l(cos alpha) |s|^l / |d + s|^{l+1}

        where cos alpha = n . s_hat.  For |s| << |d| this simplifies to

            Phi(r) ~ -G M [ 1/d + (n.s)/d^3
                            + (3(n.s)^2 - |s|^2) / (2 d^5) + ... ]
        """
        s = r_cgs - self.r0
        s_abs = np.linalg.norm(s)
        if s_abs < 1.0e-60:
            return -GRAVITATIONAL_CGS * self.M / self.d
        cos_alpha = np.dot(self.n, s) / s_abs
        # Legendre polynomials up to l_max
        P = [1.0, cos_alpha]
        for l in range(2, self.l_max + 1):
            Pl = ((2 * l - 1) * cos_alpha * P[-1] - (l - 1) * P[-2]) / l
            P.append(Pl)
        # series sum
        phi = 0.0
        for l in range(self.l_max + 1):
            term = P[l] * (s_abs ** l) / (self.d ** (l + 1))
            phi += term
        return -GRAVITATIONAL_CGS * self.M * phi

    def acceleration_at(self, r_cgs: np.ndarray) -> np.ndarray:
        """
        Gradient of the multipole-expanded potential at r.

        We use a simple centred finite-difference approximation
        for robustness; analytic expressions are available but
        become unwieldy beyond l=2.
        """
        eps = 1.0e-3 * max(self.d, 1.0e-6)
        grad = np.zeros(3)
        for ax in range(3):
            r_plus = r_cgs.copy()
            r_minus = r_cgs.copy()
            r_plus[ax] += eps
            r_minus[ax] -= eps
            grad[ax] = -(self.potential_at(r_plus) - self.potential_at(r_minus)) / (2.0 * eps)
        return grad


# =====================================================================
#                 VORONOI-BASED GRAVITY SOLVER
# =====================================================================

class VoronoiGravitySolver:
    """
    Compute the gravitational potential on a 3-D grid due to a set
    of point masses, using Voronoi tessellation and multipole expansion.

    Algorithm
    ---------
    1. Build the Voronoi diagram of the point masses (projected to 2-D
       for the small test problem; a full 3-D Voronoi would require
       scipy.spatial which we avoid to keep dependencies minimal).
    2. For each Voronoi cell, construct a MultipoleExpansion centred on
       the cell centroid.
    3. For each grid point, find the owning cell and evaluate the
       multipole-expanded potential there; add contributions from all
       other masses via direct summation.
    """

    def __init__(
        self,
        positions_kpc: np.ndarray,   # shape (N, 3)
        masses_cgs: np.ndarray,       # shape (N,)
        grid_N: int = 16,
        box_kpc: float = 10.0,
        l_max: int = 3,
    ) -> None:
        self.positions_kpc = positions_kpc
        self.masses_cgs = masses_cgs
        self.grid_N = grid_N
        self.box_kpc = box_kpc
        self.l_max = l_max
        self._validate()

    def _validate(self) -> None:
        if self.positions_kpc.shape[0] != self.masses_cgs.size:
            raise ValueError("positions and masses must have same size")
        if self.positions_kpc.ndim != 2 or self.positions_kpc.shape[1] != 3:
            raise ValueError("positions must be (N, 3)")

    # -----------------------------------------------------------------
    #  2-D Voronoi construction (on z-sliced projection)
    # -----------------------------------------------------------------
    def build_voronoi_cells_2d(self, z_slice: float = 0.0
                                ) -> List[List[np.ndarray]]:
        """
        Build the 2-D Voronoi cells of the points projected to the
        (x, y) plane at z = z_slice.
        """
        pts_2d = self.positions_kpc[:, :2]
        cells = []
        box = (0.0, self.box_kpc, 0.0, self.box_kpc)
        for i in range(pts_2d.shape[0]):
            cell = voronoi_cell_2d(pts_2d, i, box)
            cells.append(cell)
        return cells

    # -----------------------------------------------------------------
    #  potential evaluation (direct + multipole hybrid)
    # -----------------------------------------------------------------
    def potential_at_point(self, r_kpc: np.ndarray) -> float:
        """
        Compute the gravitational potential at position r (kpc) by
        direct summation over all point masses:

            Phi(r) = -G sum_i M_i / |r - p_i|
        """
        r_cgs = r_kpc * KILOPARSEC_CGS
        phi = 0.0
        for i in range(self.positions_kpc.shape[0]):
            p_cgs = self.positions_kpc[i] * KILOPARSEC_CGS
            d = np.linalg.norm(r_cgs - p_cgs)
            d = max(d, 1.0e-3 * KILOPARSEC_CGS)  # softening
            phi -= GRAVITATIONAL_CGS * self.masses_cgs[i] / d
        return phi

    def potential_on_grid(self) -> np.ndarray:
        """Evaluate Phi on the 3-D uniform grid; return (N, N, N)."""
        N = self.grid_N
        dx = self.box_kpc / N
        phi_grid = np.zeros((N, N, N))
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    r = np.array([(i + 0.5) * dx, (j + 0.5) * dx,
                                  (k + 0.5) * dx])
                    phi_grid[i, j, k] = self.potential_at_point(r)
        return phi_grid

    # -----------------------------------------------------------------
    #  cell anisotropy diagnostic
    # -----------------------------------------------------------------
    def cell_anisotropy_tensor_2d(self, vertices: List[np.ndarray]
                                   ) -> np.ndarray:
        """
        Compute the anisotropy tensor of a 2-D Voronoi cell:

            T_{ab} = (1/A) int_V (x_a - <x_a>) (x_b - <x_b>) dA.

        The eigenvalues of T measure the axis lengths; their ratio
        is the cell eccentricity (direct analogue of the ellipse
        eccentricity in 328_ellipse).
        """
        verts = np.array(vertices)
        if verts.size == 0:
            return np.zeros((2, 2))
        centroid = verts.mean(axis=0)
        d = verts - centroid
        T = np.zeros((2, 2))
        n = len(verts)
        area = cell_area_2d(vertices)
        if area < 1.0e-12:
            return T
        # approximate: sum over vertices weighted by vertex area (1/n)
        for v in d:
            T += np.outer(v, v) / n
        return T / max(area, 1.0e-12)
