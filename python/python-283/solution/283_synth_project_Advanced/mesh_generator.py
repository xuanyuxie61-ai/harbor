"""
mesh_generator.py
=================
Mesh generation and quality control for 2D perovskite solar-cell defect
simulations. Combines three classical building blocks:

1. Q4 quadrilateral mesh construction    (from 953_quadrilateral_mesh)
2. Delaunay discrepancy check             (from 1335_triangulation_delaunay_discrepancy)
3. Lloyd's algorithm for CVT optimization (from 676_line_cvt_lloyd)

In the context of perovskite defect-state calculations, these meshes are used
to discretize the device cross-section (e.g. the MAPbI3 active layer between
ETL/perovskite/HTL interfaces). Defect clusters (e.g. iodine vacancy
aggregates, Pb-I anti-sites) are represented as point-like charge centers;
we wish to place them on a centroidal Voronoi tessellation so that each
defect "owns" a territory of equal weighted charge --- this is the physical
content of Lloyd's algorithm with a charge-density weight function.

Mathematical model
------------------
The CVT energy for generators {z_i}_{i=1}^N with weight function rho(x) is
    E({z_i}) = sum_{i=1}^N Integral_{V_i} rho(x) ||x - z_i||^2 dx
where V_i is the Voronoi cell of z_i. Lloyd's algorithm alternates:
    (1) Form the Voronoi partition V_i of the domain.
    (2) Update each generator to be the mass centroid of its cell:
        z_i^{new} = Integral_{V_i} x rho(x) dx / Integral_{V_i} rho(x) dx.
Convergence of this iteration is guaranteed to a critical point of E.

For a 1D defect line (e.g. a grain boundary at the ETL/perovskite interface),
we use a 1D Lloyd iteration with a defect-density weighting rho(x) obtained
from the SRH recombination rate.

Delaunay quality
----------------
For a triangulation T, the local Delaunay condition requires that for every
pair of adjacent triangles sharing edge e, the circumcircle of each triangle
does not contain the opposite vertex. The Delaunay discrepancy is:
    Delta_local = min_angle(T') - min_angle(T)
where T' is obtained by flipping e to the other diagonal of the quadrilateral
formed by T and T'. A positive max discrepancy indicates the mesh is NOT
Delaunay; for our CVT meshes we expect discrepancy <= O(1e-10).
"""

from __future__ import annotations
import math
from typing import Tuple, List, Optional

import numpy as np
from numpy.typing import NDArray


# ============================================================================
# 1. Q4 quadrilateral mesh (ported from 953)
# ============================================================================
def q4_mesh_unit_square(nx: int, ny: int) -> Tuple[NDArray, NDArray]:
    """Generate a structured Q4 mesh on [0,1]^2 with nx * ny elements.
    Returns:
        node_xy  : shape ( (nx+1)*(ny+1), 2 ) node coordinates
        elem_node: shape ( nx*ny, 4 ) node indices per element (CCW)
    """
    xs = np.linspace(0.0, 1.0, nx + 1)
    ys = np.linspace(0.0, 1.0, ny + 1)
    xx, yy = np.meshgrid(xs, ys)
    node_xy = np.column_stack([xx.ravel(), yy.ravel()])
    elem_node = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = n1 + (nx + 1)
            n3 = n0 + (nx + 1)
            elem_node.append([n0, n1, n2, n3])
    return node_xy, np.asarray(elem_node, dtype=int)


def area_quad(q: NDArray) -> float:
    """Signed area of a planar quadrilateral with vertices q[:,0..3].
    Uses the shoelace formula: A = 0.5 |sum_{i} (x_i y_{i+1} - x_{i+1} y_i)|.
    For non-convex or mildly warped Q4 elements, we split into two triangles.
    """
    x = q[0, :]
    y = q[1, :]
    # Split into triangles (0,1,2) and (0,2,3)
    a1 = 0.5 * abs(x[0] * y[1] - x[1] * y[0]
                   + x[1] * y[2] - x[2] * y[1]
                   + x[2] * y[0] - x[0] * y[2])
    a2 = 0.5 * abs(x[0] * y[2] - x[2] * y[0]
                   + x[2] * y[3] - x[3] * y[2]
                   + x[3] * y[0] - x[0] * y[3])
    return float(a1 + a2)


def area_q4_mesh(node_xy: NDArray, elem_node: NDArray) -> Tuple[NDArray, float]:
    """Compute per-element and total area of a Q4 mesh."""
    ne = elem_node.shape[0]
    elem_area = np.zeros(ne)
    for e in range(ne):
        q = node_xy[elem_node[e]].T  # (2, 4)
        elem_area[e] = area_quad(q)
    return elem_area, float(elem_area.sum())


def jacobian_q4(q: NDArray, xi: float, eta: float) -> Tuple[float, NDArray]:
    """Compute the Jacobian determinant and matrix of a Q4 element at local
    coords (xi, eta) in [-1,1]^2.

    Shape functions:
        N1 = 0.25 (1-xi)(1-eta)
        N2 = 0.25 (1+xi)(1-eta)
        N3 = 0.25 (1+xi)(1+eta)
        N4 = 0.25 (1-xi)(1+eta)
    dN/dxi, dN/deta follow by differentiation.

    The Jacobian matrix J = [dx/dxi, dy/dxi; dx/deta, dy/deta] and
    det J gives the local area scaling. A negative det J indicates an
    inverted element --- a fatal defect in any FEM mesh.
    """
    dN_dxi = 0.25 * np.array([
        [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
        [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]
    ])  # shape (2, 4)
    # q has shape (2, 4): q[axis, node]
    J = dN_dxi @ q.T  # shape (2, 2)
    detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
    return float(detJ), J


def mesh_quality_report(node_xy: NDArray, elem_node: NDArray) -> dict:
    """Return a quality report of the Q4 mesh:
    - min / max Jacobian determinant at element centers (xi=eta=0)
    - min interior angle
    - aspect ratio
    """
    ne = elem_node.shape[0]
    detJ_min = +math.inf
    detJ_max = -math.inf
    angle_min = +math.inf
    aspect_max = 0.0
    for e in range(ne):
        q = node_xy[elem_node[e]].T
        detJ, _ = jacobian_q4(q, 0.0, 0.0)
        detJ_min = min(detJ_min, detJ)
        detJ_max = max(detJ_max, detJ)
        # Angles at the four corners
        for i in range(4):
            p_prev = q[:, (i - 1) % 4]
            p_curr = q[:, i]
            p_next = q[:, (i + 1) % 4]
            v1 = p_prev - p_curr
            v2 = p_next - p_curr
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 * n2 < 1e-30:
                continue
            cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            angle_min = min(angle_min, math.degrees(math.acos(cos_a)))
        # Aspect ratio via edge lengths
        edges = []
        for i in range(4):
            a_ = q[:, i]
            b_ = q[:, (i + 1) % 4]
            edges.append(np.linalg.norm(a_ - b_))
        edges = np.array(edges)
        aspect = edges.max() / max(edges.min(), 1e-30)
        aspect_max = max(aspect_max, aspect)
    return {
        "n_elements": ne,
        "detJ_min": detJ_min,
        "detJ_max": detJ_max,
        "angle_min_deg": angle_min,
        "aspect_max": aspect_max,
        "total_area": float(area_q4_mesh(node_xy, elem_node)[1]),
    }


# ============================================================================
# 2. Triangulation Delaunay discrepancy (ported from 1335)
# ============================================================================
def delaunay_flip_discrepancy(tri: NDArray, pts: NDArray) -> float:
    """Compute the maximum Delaunay discrepancy of a triangulation.
    For each interior edge shared by two triangles, compute the angle
    improvement that would result from an edge flip. Returns the max
    discrepancy; a Delaunay mesh gives <= 0.

    tri : (n_tri, 3) node indices of each triangle
    pts : (n_pts, 2)  node coordinates
    """
    # Build edge-to-triangle adjacency
    edge_map: dict = {}
    for t_idx, (a, b, c) in enumerate(tri):
        for (u, v, w) in [(a, b, c), (b, c, a), (c, a, b)]:
            key = (min(u, v), max(u, v))
            edge_map.setdefault(key, []).append((t_idx, w))
    max_disc = -math.inf
    for key, adj in edge_map.items():
        if len(adj) != 2:
            continue
        (t1, opp1), (t2, opp2) = adj
        # Quadrilateral vertices: the two shared + two opposite
        u, v = key
        quad = [pts[u], pts[v], pts[opp1], pts[opp2]]
        # Original min angle of the two triangles
        tri1 = [pts[u], pts[v], pts[opp1]]
        tri2 = [pts[u], pts[v], pts[opp2]]
        a_orig = min(_min_angle_tri(tri1), _min_angle_tri(tri2))
        # Flipped: triangles (opp1, opp2, u) and (opp1, opp2, v)
        tri1f = [pts[opp1], pts[opp2], pts[u]]
        tri2f = [pts[opp1], pts[opp2], pts[v]]
        a_flip = min(_min_angle_tri(tri1f), _min_angle_tri(tri2f))
        disc = a_flip - a_orig
        max_disc = max(max_disc, disc)
    return max_disc


def _min_angle_tri(tri: list) -> float:
    """Return the minimum interior angle (radians) of a triangle."""
    p0, p1, p2 = np.asarray(tri[0]), np.asarray(tri[1]), np.asarray(tri[2])
    angles = []
    for a_, b_, c_ in [(p0, p1, p2), (p1, p2, p0), (p2, p0, p1)]:
        v1 = b_ - a_
        v2 = c_ - a_
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 * n2 < 1e-30:
            return 0.0
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        angles.append(math.acos(cos_a))
    return min(angles)


# ============================================================================
# 3. Lloyd's algorithm for CVT (ported from 676)
# ============================================================================
def line_cvt_lloyd(n: int, a: float, b: float, it_num: int,
                   density_fn=None,
                   x_init: Optional[NDArray] = None) -> NDArray:
    """Run Lloyd's algorithm in 1D to compute a centroidal Voronoi
    tessellation of [a, b] with n generators under a density function.

    density_fn : callable rho(x) >= 0  (defaults to rho = 1).
                 For perovskite defect modeling, use rho(x) = N_t(x)
                 (the defect-density profile from SRH recombination).

    Returns the final generator locations x of length n.
    """
    if density_fn is None:
        density_fn = lambda x: np.ones_like(x)
    if x_init is None:
        x = np.linspace(a + (b - a) / (2 * n),
                        b - (b - a) / (2 * n), n)
    else:
        x = np.copy(x_init).ravel()
        if len(x) != n:
            raise ValueError("x_init length must equal n")
    energies = []
    for _ in range(it_num):
        # Voronoi boundaries: midpoints between adjacent generators
        bounds = np.empty(n + 1)
        bounds[0] = a
        bounds[-1] = b
        bounds[1:-1] = 0.5 * (x[:-1] + x[1:])
        x_new = np.empty(n)
        energy = 0.0
        for i in range(n):
            lo, hi = bounds[i], bounds[i + 1]
            # Integrate rho and x*rho via Simpson's rule over [lo, hi]
            nq = 64
            qq = np.linspace(lo, hi, nq)
            rho_q = density_fn(qq)
            mass = np.trapz(rho_q, qq)
            if mass < 1e-30:
                x_new[i] = 0.5 * (lo + hi)
                continue
            x_new[i] = np.trapz(qq * rho_q, qq) / mass
            # CVT energy contribution
            dq = qq - x[i]
            energy += np.trapz(rho_q * dq * dq, qq)
        energies.append(energy)
        x = x_new
    return x


def line_cvt_energy_1d(x: NDArray, a: float, b: float,
                       density_fn=None) -> float:
    """Compute the CVT energy for 1D generator set x in [a, b]."""
    if density_fn is None:
        density_fn = lambda x: np.ones_like(x)
    n = len(x)
    bounds = np.empty(n + 1)
    bounds[0] = a
    bounds[-1] = b
    bounds[1:-1] = 0.5 * (x[:-1] + x[1:])
    energy = 0.0
    for i in range(n):
        lo, hi = bounds[i], bounds[i + 1]
        qq = np.linspace(lo, hi, 64)
        rho_q = density_fn(qq)
        dq = qq - x[i]
        energy += np.trapz(rho_q * dq * dq, qq)
    return energy


# ============================================================================
# Defect-aware mesh driver
# ============================================================================
def build_defect_mesh(nx: int = 16, ny: int = 8,
                      defect_centers: Optional[List[Tuple[float, float]]] = None,
                      defect_sigma: float = 0.05) -> dict:
    """Build a Q4 mesh of the perovskite active-layer cross-section, then
    refine a 1D CVT along the x-direction weighted by a Gaussian defect
    density centered at defect_centers.

    defect_centers : list of (x, y) defect cluster positions in [0,1]^2
    defect_sigma   : Gaussian width of each defect's charge cloud

    Returns a dict with keys:
        'node_xy', 'elem_node', 'quality', 'defect_cvt_x', 'delaunay_disc'
    """
    node_xy, elem_node = q4_mesh_unit_square(nx, ny)
    quality = mesh_quality_report(node_xy, elem_node)
    # Defect density as sum of Gaussians
    if defect_centers is None:
        defect_centers = [(0.3, 0.5), (0.7, 0.5)]

    def rho_def(x):
        s = 0.0
        for cx, _ in defect_centers:
            s += np.exp(-((x - cx) ** 2) / (2.0 * defect_sigma ** 2))
        return s + 1e-3  # background floor

    # Run Lloyd's algorithm along x with the defect density
    n_gen = max(8, nx // 2)
    cvt_x = line_cvt_lloyd(n_gen, 0.0, 1.0, it_num=30, density_fn=rho_def)
    # Build a simple Delaunay triangulation of the CVT points + boundary
    # to test the discrepancy check
    pts_2d = np.zeros((n_gen + 2, 2))
    pts_2d[:2] = [[0.0, 0.0], [1.0, 0.0]]
    pts_2d[2:, 0] = cvt_x
    pts_2d[2:, 1] = 0.5
    # Triangulation: connect consecutive triplets (simple fan)
    tris = []
    for i in range(len(pts_2d) - 2):
        tris.append([i, i + 1, i + 2])
    tris = np.asarray(tris, dtype=int)
    disc = delaunay_flip_discrepancy(tris, pts_2d)
    return {
        "node_xy": node_xy,
        "elem_node": elem_node,
        "quality": quality,
        "defect_cvt_x": cvt_x,
        "delaunay_disc": disc,
        "defect_centers": defect_centers,
    }
