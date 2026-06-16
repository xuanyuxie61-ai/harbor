"""
mesh_defect.py — 2D triangular mesh generation with defect-centered refinement
==============================================================================

To compute the *elastic* relaxation around a point defect we need a finite-
element mesh that is fine near the defect (where strain is large) and coarse
far away. This module implements a simple recursive refinement strategy:

  1. Build a coarse triangulation of the rectangular supercell by splitting
     each rectangle into two triangles.
  2. Mark for refinement every triangle whose centroid is within R_ref of
     the defect site.
  3. Refine marked triangles by connecting edge midpoints (red refinement).
  4. Repeat until no triangle is marked.
  5. Extract the boundary of the refined mesh using the algorithm from
     `1331_triangulation_boundary`.
  6. Project the scalar elastic energy density onto the mesh nodes and
     compute the elastic energy by 2-D FEM quadrature (inspired by
     `414_fem2d_scalar_display`).

Seed project integration:
  * 548_human_mesh2d/human_mesh2d.m: the overall strategy of generating
    a 2-D triangular mesh inside a polygonal domain with a maximum element
    size. We replace the MATLAB `mesh2d` toolbox by a simple structured
    initial mesh + adaptive refinement, which is enough for the defect
    problem where the domain is rectangular.
  * 1331_triangulation_boundary/boundary_edge_to_path.m: boundary
    extraction.
  * 414_fem2d_scalar_display/fem2d_scalar_display.m: scalar FEM field
    evaluation at nodes.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple, List

from crystal_lattice import extract_boundary


# -------------------------------------------------------------------------
# (1) Initial structured triangular mesh of a rectangle
# -------------------------------------------------------------------------
def structured_mesh(Lx: float, Ly: float, nx: int, ny: int
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (nodes, triangles) of a structured mesh of [0, Lx] × [0, Ly].

    Each rectangle is split along its diagonal into two triangles, giving
    2 nx ny triangles and (nx + 1)(ny + 1) nodes.
    """
    xs = np.linspace(0, Lx, nx + 1)
    ys = np.linspace(0, Ly, ny + 1)
    X, Y = np.meshgrid(xs, ys)
    nodes = np.column_stack([X.ravel(), Y.ravel()])
    triangles: List[Tuple[int, int, int]] = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = n0 + (nx + 1)
            n3 = n2 + 1
            triangles.append((n0, n1, n3))
            triangles.append((n0, n3, n2))
    return nodes, np.asarray(triangles, dtype=np.int64)


# -------------------------------------------------------------------------
# (2) Red refinement: split marked triangles into 4
# -------------------------------------------------------------------------
def red_refine_once(nodes: np.ndarray, triangles: np.ndarray,
                    marked: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Refine marked triangles by connecting edge midpoints.

    Each marked triangle (a, b, c) is split into 4 sub-triangles by adding
    the midpoints m_ab, m_bc, m_ca. New nodes are de-duplicated.
    """
    edge_midpoint: dict = {}
    new_nodes: List[np.ndarray] = list(nodes)

    def _midpoint(i: int, j: int) -> int:
        key = (min(i, j), max(i, j))
        if key in edge_midpoint:
            return edge_midpoint[key]
        mid = 0.5 * (nodes[i] + nodes[j])
        idx = len(new_nodes)
        new_nodes.append(mid)
        edge_midpoint[key] = idx
        return idx

    new_tris: List[Tuple[int, int, int]] = []
    for t_idx, tri in enumerate(triangles):
        if marked[t_idx]:
            a, b, c = tri
            mab = _midpoint(a, b)
            mbc = _midpoint(b, c)
            mca = _midpoint(c, a)
            new_tris.append((a, mab, mca))
            new_tris.append((mab, b, mbc))
            new_tris.append((mca, mbc, c))
            new_tris.append((mab, mbc, mca))
        else:
            new_tris.append(tuple(tri))
    new_nodes_arr = np.asarray(new_nodes, dtype=np.float64)
    new_tris_arr = np.asarray(new_tris, dtype=np.int64)
    return new_nodes_arr, new_tris_arr


# -------------------------------------------------------------------------
# (3) Adaptive refinement loop around a defect point
# -------------------------------------------------------------------------
def refine_around_defect(nodes: np.ndarray, triangles: np.ndarray,
                         defect_pt: np.ndarray, R_ref: float,
                         max_levels: int = 4
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Recursively refine triangles whose centroid is within R_ref of defect_pt."""
    for _ in range(max_levels):
        centroids = nodes[triangles].mean(axis=1)
        dist = np.linalg.norm(centroids - defect_pt, axis=1)
        marked = dist < R_ref
        if not marked.any():
            break
        # compute triangle sizes and shrink R_ref by factor sqrt(2)/2
        nodes, triangles = red_refine_once(nodes, triangles, marked)
        R_ref *= 0.6
    return nodes, triangles


# -------------------------------------------------------------------------
# (4) Boundary extraction (uses 1331_triangulation_boundary via import)
# -------------------------------------------------------------------------
def mesh_boundary(triangles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract the boundary edges and ordered path of a triangulation.

    Delegates to crystal_lattice.extract_boundary which is a direct port
    of Burkardt's boundary_edge_to_path.m.
    """
    return extract_boundary(triangles)


# -------------------------------------------------------------------------
# (5) FEM scalar field evaluation on the mesh (à la 414_fem2d_scalar_display)
# -------------------------------------------------------------------------
def evaluate_scalar_field(nodes: np.ndarray, triangles: np.ndarray,
                          nodal_values: np.ndarray,
                          query_points: np.ndarray) -> np.ndarray:
    """Evaluate a scalar field (given by nodal values) at query points.

    For each query point we find the enclosing triangle by brute force
    (sufficient for our small meshes) and interpolate with barycentric
    coordinates. Port of the display logic in `414_fem2d_scalar_display.m`.
    """
    out = np.zeros(query_points.shape[0])
    for k, q in enumerate(query_points):
        found = False
        for tri in triangles:
            a, b, c = nodes[tri[0]], nodes[tri[1]], nodes[tri[2]]
            # barycentric
            v0 = b - a
            v1 = c - a
            v2 = q - a
            d00 = np.dot(v0, v0)
            d01 = np.dot(v0, v1)
            d11 = np.dot(v1, v1)
            d20 = np.dot(v2, v0)
            d21 = np.dot(v2, v1)
            denom = d00 * d11 - d01 * d01
            if abs(denom) < 1e-30:
                continue
            v = (d11 * d20 - d01 * d21) / denom
            w = (d00 * d21 - d01 * d20) / denom
            u = 1.0 - v - w
            if u >= -1e-9 and v >= -1e-9 and w >= -1e-9:
                out[k] = (u * nodal_values[tri[0]]
                          + v * nodal_values[tri[1]]
                          + w * nodal_values[tri[2]])
                found = True
                break
        if not found:
            out[k] = 0.0
    return out


# -------------------------------------------------------------------------
# (6) Strain energy density on mesh nodes
# -------------------------------------------------------------------------
def strain_energy_on_mesh(nodes: np.ndarray, triangles: np.ndarray,
                          defect_pt: np.ndarray, G: float, nu: float,
                          misfit: float) -> np.ndarray:
    """Compute the Eshelby strain energy density at each node.

    For a circular inclusion of radius R centered at defect_pt, the strain
    energy density decays as 1/r⁴ outside the inclusion. We use
        w(r) = w_0 (R / r)^4   for r > R
        w(r) = w_0              for r ≤ R
    where w_0 = 2 G (1 + ν) / (1 − ν) · misfit².
    """
    w0 = 2.0 * G * (1.0 + nu) / (1.0 - nu) * misfit ** 2
    R = 0.3 * np.linalg.norm(nodes.max(axis=0) - nodes.min(axis=0))
    dists = np.linalg.norm(nodes - defect_pt, axis=1)
    w = np.where(dists <= R, w0, w0 * (R / np.maximum(dists, 1e-12)) ** 4)
    return w


# -------------------------------------------------------------------------
# (7) Build a refined defect mesh and return summary
# -------------------------------------------------------------------------
def build_defect_mesh(Lx: float, Ly: float, defect_pt: np.ndarray,
                      R_ref: float = 1.0, max_levels: int = 3
                      ) -> dict:
    """Build the full defect-centred mesh and return summary statistics."""
    nodes, triangles = structured_mesh(Lx, Ly, nx=8, ny=8)
    nodes, triangles = refine_around_defect(nodes, triangles,
                                            defect_pt, R_ref, max_levels)
    bedges, bpath = mesh_boundary(triangles)
    return {
        "n_nodes": nodes.shape[0],
        "n_triangles": triangles.shape[0],
        "n_boundary_edges": bedges.shape[0],
        "n_boundary_nodes": bpath.size,
        "nodes": nodes,
        "triangles": triangles,
        "boundary_path": bpath,
    }
