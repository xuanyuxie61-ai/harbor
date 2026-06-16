"""
spatial_ops.py
==============

Spatial I/O and geometric utilities ported from three Burkardt codes:

* ``triangulation_histogram`` — histogram a point cloud over a
  triangulation; checks whether the samples are uniformly distributed.
  In the Sobol pipeline we use this to diagnose the Saltelli sample
  distribution in the 2-D projection planes.

* ``grf_display`` — read an abstract graph from a ``.grf``-style
  text file (node list + edge list).  The graph represents the
  **sensitivity adjacency structure**: two parameters ``theta_i``,
  ``theta_j`` are linked iff their second-order Sobol index ``S_{ij}``
  exceeds a threshold.  The graph is the input to any subsequent
  community-detection or reduction step.

* ``ice_to_medit`` — originally a NetCDF-to-MEDIT converter; we
  repurpose the *coordinate transform* core to build curvilinear
  meshes on the disk.  In particular we provide:

  - ``disk_mesh_polar`` — structured polar mesh of the disk with
    configurable radial and angular resolution.
  - ``param_to_disk`` — map the unit hypercube ``[0, 1]^d`` to the
    physical parameter space of the reactor.
  - ``mesh_stats`` — compute area, minimum angle, aspect ratio.

These routines are pure I/O + geometry; they never call the forward
model.

References
----------
* J. Burkardt, ``triangulation_histogram``, ``grf_display``, and
  ``ice_to_medit`` MATLAB libraries.
* J. R. Shewchuk, *Triangle: Engineering a 2D quality mesh generator
  and Delaunay triangulator*, Lect. Notes Comput. Sci. 1143 (1996).
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


# =====================================================================
# Triangulation histogram
# =====================================================================
def triangle_area(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Signed area of the triangle ``(p1, p2, p3)`` in 2-D."""
    return 0.5 * ((p2[0] - p1[0]) * (p3[1] - p1[1])
                  - (p3[0] - p1[0]) * (p2[1] - p1[1]))


def point_in_triangle(pt: np.ndarray, p1: np.ndarray, p2: np.ndarray,
                      p3: np.ndarray) -> bool:
    """Barycentric inside-test for a 2-D point vs triangle."""
    a = triangle_area(p1, p2, p3)
    if abs(a) < 1.0e-14:
        return False
    a1 = triangle_area(pt, p2, p3) / a
    a2 = triangle_area(p1, pt, p3) / a
    a3 = triangle_area(p1, p2, pt) / a
    return (a1 >= -1.0e-12) and (a2 >= -1.0e-12) and (a3 >= -1.0e-12)


def triangulation_histogram(nodes: np.ndarray,
                            elements: np.ndarray,
                            samples: np.ndarray) -> dict:
    """Histogram ``samples`` over the 2-D triangulation.

    Parameters
    ----------
    nodes : (N_nodes, 2) ndarray
        Node coordinates.
    elements : (N_elems, 3) ndarray of ints
        Triangle vertex indices (0-based).
    samples : (N_samples, 2) ndarray
        Point cloud to histogram.

    Returns
    -------
    dict
        ``{'A': ..., 'A_total': ..., 'N_per_elem': ...,
        'N_total': ..., 'uniformity_score': ...}``.
    """
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError("triangulation_histogram: nodes must be (N, 2)")
    if elements.ndim != 2 or elements.shape[1] != 3:
        raise ValueError("triangulation_histogram: elements must be (M, 3)")
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError("triangulation_histogram: samples must be (S, 2)")
    n_elem = elements.shape[0]
    areas = np.array([abs(triangle_area(nodes[elements[k, 0]],
                                        nodes[elements[k, 1]],
                                        nodes[elements[k, 2]]))
                      for k in range(n_elem)])
    a_total = areas.sum()
    n_per_elem = np.zeros(n_elem, dtype=int)
    n_total = 0
    for s in samples:
        for k in range(n_elem):
            if point_in_triangle(s, nodes[elements[k, 0]],
                                 nodes[elements[k, 1]],
                                 nodes[elements[k, 2]]):
                n_per_elem[k] += 1
                n_total += 1
                break
    # Uniformity score: max |Ni/N - Ai/A|
    if n_total == 0 or a_total == 0.0:
        return dict(A=areas, A_total=a_total, N_per_elem=n_per_elem,
                    N_total=0, uniformity_score=1.0)
    frac_a = areas / a_total
    frac_n = n_per_elem / max(n_total, 1)
    unif = float(np.max(np.abs(frac_a - frac_n)))
    return dict(A=areas, A_total=a_total, N_per_elem=n_per_elem,
                N_total=n_total, uniformity_score=unif)


# =====================================================================
# GRF-style graph reader (in-memory variant)
# =====================================================================
def grf_read_from_string(text: str) -> dict:
    """Parse a mini GRF format::

        NODES <n>
        <id> <x> <y>
        ...
        EDGES <m>
        <i> <j>
        ...

    Returns ``{'nodes': ndarray (n, 2), 'edges': list of (i, j)}``.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    nodes = []
    edges = []
    i = 0
    while i < len(lines):
        tok = lines[i].split()
        if tok[0].upper() == "NODES":
            n = int(tok[1])
            for k in range(n):
                i += 1
                parts = lines[i].split()
                nodes.append([float(parts[1]), float(parts[2])])
        elif tok[0].upper() == "EDGES":
            m = int(tok[1])
            for k in range(m):
                i += 1
                parts = lines[i].split()
                edges.append((int(parts[0]), int(parts[1])))
        i += 1
    return dict(nodes=np.array(nodes, dtype=float), edges=edges)


def build_sensitivity_graph(S2: np.ndarray, threshold: float = 0.05
                            ) -> dict:
    """Build the Sobol second-order sensitivity graph.

    Parameters
    ----------
    S2 : (d, d) ndarray
        Second-order Sobol indices.
    threshold : float
        Edges are added for pairs ``|S2[i, j]| >= threshold``.

    Returns
    -------
    dict
        ``{'n_nodes': d, 'edges': list of (i, j), 'degree': ndarray,
        'grf_text': str}``.
    """
    if S2.ndim != 2 or S2.shape[0] != S2.shape[1]:
        raise ValueError("build_sensitivity_graph: S2 must be square")
    d = S2.shape[0]
    edges = []
    for i in range(d):
        for j in range(i + 1, d):
            if abs(S2[i, j]) >= threshold:
                edges.append((i, j))
    deg = np.zeros(d, dtype=int)
    for i, j in edges:
        deg[i] += 1
        deg[j] += 1
    # Build GRF text
    lines = [f"NODES {d}"]
    # Place nodes on a circle
    for k in range(d):
        ang = 2.0 * math.pi * k / d
        lines.append(f"{k} {math.cos(ang):.6f} {math.sin(ang):.6f}")
    lines.append(f"EDGES {len(edges)}")
    for i, j in edges:
        lines.append(f"{i} {j}")
    return dict(n_nodes=d, edges=edges, degree=deg,
                grf_text="\n".join(lines) + "\n")


# =====================================================================
# Disk mesh + parameter map (ice_to_medit lineage)
# =====================================================================
def disk_mesh_polar(R: float, Nr: int, Ntheta: int) -> dict:
    """Structured polar mesh of the disk of radius ``R``.

    Returns ``{'nodes': (N, 2), 'elements': (M, 4), 'areas': (M,)}``.
    Elements are quadrilaterals (r_i, r_{i+1}) x (theta_j, theta_{j+1}).
    """
    if R <= 0.0 or Nr < 1 or Ntheta < 3:
        raise ValueError("disk_mesh_polar: invalid parameters")
    r_edges = np.linspace(0.0, R, Nr + 1)
    th_edges = np.linspace(0.0, 2.0 * math.pi, Ntheta + 1)
    nodes = []
    for i in range(Nr + 1):
        for j in range(Ntheta):
            r = r_edges[i]
            th = th_edges[j]
            nodes.append((r * math.cos(th), r * math.sin(th)))
    nodes = np.array(nodes, dtype=float)
    elements = []
    for i in range(Nr):
        for j in range(Ntheta):
            jp = (j + 1) % Ntheta
            n00 = i * Ntheta + j
            n10 = (i + 1) * Ntheta + j
            n11 = (i + 1) * Ntheta + jp
            n01 = i * Ntheta + jp
            elements.append((n00, n10, n11, n01))
    elements = np.array(elements, dtype=int)
    # Areas (approximate via the shoelace formula)
    areas = np.zeros(elements.shape[0])
    for k, el in enumerate(elements):
        pts = nodes[el]
        # Shoelace for a quad
        x, y = pts[:, 0], pts[:, 1]
        areas[k] = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return dict(nodes=nodes, elements=elements, areas=areas,
                r_edges=r_edges, th_edges=th_edges)


def param_to_disk(theta: np.ndarray, R: float = 1.0) -> np.ndarray:
    """Map ``theta in [0, 1]^d`` to physical parameters of the reactor.

    Physical parameter dictionary (8-D)::

        K_chir  in [0.0, 2.0]       Chirikov stochasticity
        r_log   in [0.1, 3.0]       logistic growth rate
        k_cap   in [0.5, 5.0]       carrying capacity
        D_eff   in [1e-3, 1.0]      effective diffusivity
        R_disk  in [0.2, 2.0]       disk radius
        alpha_k in [0.0, 1.0]       competition coeff.
        beta_k  in [0.0, 1.0]       mutualism coeff.
        gamma_k in [0.0, 1.0]       mortality rate

    Returns an array of shape ``(d, )`` with the above physical values.
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    d = theta.size
    if d != 8:
        raise ValueError("param_to_disk: requires exactly 8-D input")
    # Unit cube -> physical ranges
    ranges = np.array([
        [0.0, 2.0],     # K_chir
        [0.1, 3.0],     # r_log
        [0.5, 5.0],     # k_cap
        [1.0e-3, 1.0],  # D_eff
        [0.2, 2.0],     # R_disk
        [0.0, 1.0],     # alpha_k
        [0.0, 1.0],     # beta_k
        [0.0, 1.0],     # gamma_k
    ])
    lo = ranges[:, 0]
    hi = ranges[:, 1]
    phys = lo + theta * (hi - lo)
    # Guard against numerical edge cases
    phys = np.maximum(phys, lo + 1.0e-12)
    phys = np.minimum(phys, hi - 1.0e-12)
    return phys


# =====================================================================
# Mesh statistics
# =====================================================================
def mesh_stats(nodes: np.ndarray, elements: np.ndarray) -> dict:
    """Compute basic quality metrics of a 2-D mesh."""
    if nodes.ndim != 2 or elements.ndim != 2:
        raise ValueError("mesh_stats: nodes and elements must be 2-D")
    n_nodes = nodes.shape[0]
    n_elems = elements.shape[0]
    # Approximate: for quad elements, split into two triangles
    areas = []
    for el in elements:
        pts = nodes[el]
        x, y = pts[:, 0], pts[:, 1]
        a = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        areas.append(a)
    areas = np.array(areas)
    return dict(n_nodes=n_nodes, n_elements=n_elems,
                total_area=float(areas.sum()),
                min_area=float(areas.min()),
                max_area=float(areas.max()),
                mean_area=float(areas.mean()))


# =====================================================================
if __name__ == "__main__":
    # Build a simple triangulation of [0,1]^2 (2 triangles)
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    elems = np.array([[0, 1, 2], [0, 2, 3]])
    rng = np.random.default_rng(0)
    samples = rng.random((50, 2))
    hist = triangulation_histogram(nodes, elems, samples)
    print("Triangulation histogram:", hist['N_total'],
          "uniformity:", hist['uniformity_score'])
    # Disk mesh
    mesh = disk_mesh_polar(1.0, 5, 8)
    print("Disk mesh:", mesh_stats(mesh['nodes'], mesh['elements']))
    # Parameter mapping
    phys = param_to_disk(np.full(8, 0.5))
    print("Physical params at theta = 0.5:", phys)
    # GRF / sensitivity graph
    S2 = np.random.default_rng(0).random((4, 4))
    S2 = (S2 + S2.T) / 2
    np.fill_diagonal(S2, 0.0)
    graph = build_sensitivity_graph(S2, threshold=0.3)
    print("Sensitivity graph:", graph['n_nodes'], "nodes,",
          len(graph['edges']), "edges")
