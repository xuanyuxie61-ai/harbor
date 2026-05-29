# -*- coding: utf-8 -*-
"""
================================================================================
Cortical Mesh Generation and Neighbor Analysis Module
================================================================================

This module generates a 2D triangulated mesh representing a cortical slice
and computes triangle neighbor relationships for finite-element analysis of
extracellular fields and synaptic connectivity patterns.

Mathematical Model:
-------------------
The cortical sheet is modeled as a planar domain Ω ⊂ ℝ² with a
Delaunay triangulation T = {T_1, T_2, ..., T_N}.

Each triangle T_k has vertices (v_{k1}, v_{k2}, v_{k3}) and edges:
    e_{k1} = (v_{k2}, v_{k3})   opposite v_{k1}
    e_{k2} = (v_{k1}, v_{k3})   opposite v_{k2}
    e_{k3} = (v_{k1}, v_{k2})   opposite v_{k3}

Two triangles are neighbors if they share an edge. The neighbor array
N[k, j] gives the index of the triangle sharing edge e_{kj} of T_k,
or -1 if the edge is on the boundary.

Mesh Quality Metrics:
---------------------
For each triangle with side lengths a, b, c:

    Area:     A = sqrt(s(s-a)(s-b)(s-c)),  s = (a+b+c)/2
    Inradius: r = A / s
    Circumradius: R = abc / (4A)
    Quality:  q = r / R = (b+c-a)(c+a-b)(a+b-c) / (abc)

A quality of q = 1 corresponds to an equilateral triangle.
A quality of q → 0 corresponds to a degenerate triangle.

================================================================================
"""

import numpy as np
from scipy.spatial import Delaunay
from typing import Tuple, Optional


def generate_cortical_boundary(
    n_vertices: int = 50,
    width: float = 1000.0,
    height: float = 800.0,
    noise_scale: float = 20.0,
    seed: int = 130,
) -> np.ndarray:
    """
    Generate a synthetic cortical slice boundary with irregular shape.

    The boundary is parameterized as a perturbed ellipse:

        x(θ) = (W/2)·cos(θ)·(1 + ε·n_x(θ))
        y(θ) = (H/2)·sin(θ)·(1 + ε·n_y(θ))

    where n_x, n_y are smoothed random perturbations modeling
    cortical folding (gyri and sulci).

    Parameters
    ----------
    n_vertices : int
        Number of boundary vertices.
    width : float
        Approximate width [μm].
    height : float
        Approximate height [μm].
    noise_scale : float
        Amplitude of boundary perturbation.
    seed : int
        Random seed.

    Returns
    -------
    boundary : np.ndarray
        Boundary vertices, shape (n_vertices, 2).
    """
    if n_vertices < 3:
        raise ValueError("n_vertices must be >= 3.")
    if width <= 0.0 or height <= 0.0:
        raise ValueError("width and height must be positive.")

    rng = np.random.default_rng(seed)
    theta = np.linspace(0.0, 2.0 * np.pi, n_vertices, endpoint=False)

    # Base ellipse
    x = 0.5 * width * np.cos(theta)
    y = 0.5 * height * np.sin(theta)

    # Add low-frequency perturbations to simulate cortical folding
    n_harmonics = 5
    for k in range(1, n_harmonics + 1):
        amp = noise_scale / k
        phase_x = rng.uniform(0.0, 2.0 * np.pi)
        phase_y = rng.uniform(0.0, 2.0 * np.pi)
        x += amp * np.cos(k * theta + phase_x)
        y += amp * np.sin(k * theta + phase_y)

    boundary = np.column_stack((x, y))
    return boundary


def generate_cortical_mesh(
    n_boundary: int = 50,
    n_interior: int = 200,
    width: float = 1000.0,
    height: float = 800.0,
    seed: int = 130,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a triangulated mesh of a cortical slice.

    Uses Delaunay triangulation on points sampled within the
    cortical boundary.

    Parameters
    ----------
    n_boundary : int
        Number of boundary vertices.
    n_interior : int
        Number of interior vertices.
    width : float
        Domain width [μm].
    height : float
        Domain height [μm].
    seed : int
        Random seed.

    Returns
    -------
    nodes : np.ndarray
        Node coordinates, shape (n_nodes, 2).
    elements : np.ndarray
        Triangle vertex indices (0-based), shape (n_triangles, 3).
    """
    if n_boundary < 3 or n_interior < 0:
        raise ValueError("Invalid vertex counts.")

    rng = np.random.default_rng(seed)

    # Generate boundary
    boundary = generate_cortical_boundary(n_boundary, width, height, seed=seed)

    # Generate interior points by rejection sampling within approx ellipse
    interior_points = []
    max_attempts = n_interior * 100
    attempts = 0

    while len(interior_points) < n_interior and attempts < max_attempts:
        px = rng.uniform(-width / 2.0, width / 2.0)
        py = rng.uniform(-height / 2.0, height / 2.0)

        # Simple inside test: check if point is inside boundary polygon
        # Using ray casting algorithm
        inside = False
        n = n_boundary
        j = n - 1
        for i in range(n):
            xi, yi = boundary[i]
            xj, yj = boundary[j]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-15) + xi):
                inside = not inside
            j = i

        if inside:
            interior_points.append([px, py])
        attempts += 1

    if len(interior_points) < n_interior:
        # Fill remaining with grid points
        nx = int(np.ceil(np.sqrt(n_interior - len(interior_points))))
        xs = np.linspace(-width / 2.0 * 0.8, width / 2.0 * 0.8, nx)
        ys = np.linspace(-height / 2.0 * 0.8, height / 2.0 * 0.8, nx)
        for xi in xs:
            for yi in ys:
                if len(interior_points) >= n_interior:
                    break
                interior_points.append([xi, yi])
            if len(interior_points) >= n_interior:
                break

    nodes = np.vstack([boundary, np.array(interior_points)])

    # Delaunay triangulation
    tri = Delaunay(nodes)
    elements = tri.simplices

    return nodes, elements


def compute_triangle_neighbors(
    elements: np.ndarray,
) -> np.ndarray:
    """
    Compute neighbor triangles for each triangle in a triangulation.

    Two triangles are neighbors if they share an edge. For each triangle
    and each of its 3 edges, we find the adjacent triangle.

    Algorithm:
    ----------
    1. For each edge (sorted vertex pair), record which triangles contain it.
    2. For each triangle edge, find the other triangle sharing that edge.
    3. Boundary edges have neighbor = -1.

    Parameters
    ----------
    elements : np.ndarray
        Triangle vertex indices, shape (n_triangles, 3).

    Returns
    -------
    neighbors : np.ndarray
        Neighbor indices, shape (n_triangles, 3).
        neighbors[t, j] = index of triangle sharing edge opposite vertex j
                          of triangle t, or -1 if boundary.
    """
    if elements.shape[1] != 3:
        raise ValueError("elements must have shape (n, 3).")

    n_tri = elements.shape[0]
    neighbors = np.full((n_tri, 3), -1, dtype=int)

    # Edge dictionary: (min_v, max_v) -> list of triangle indices
    edge_to_tri = {}

    for t in range(n_tri):
        v = elements[t]
        # Three edges: (v1,v2), (v2,v3), (v3,v1)
        edges = [
            (min(v[1], v[2]), max(v[1], v[2])),
            (min(v[2], v[0]), max(v[2], v[0])),
            (min(v[0], v[1]), max(v[0], v[1])),
        ]
        for e in edges:
            if e not in edge_to_tri:
                edge_to_tri[e] = []
            edge_to_tri[e].append(t)

    # Assign neighbors
    for t in range(n_tri):
        v = elements[t]
        edges = [
            (min(v[1], v[2]), max(v[1], v[2])),  # opposite v[0]
            (min(v[2], v[0]), max(v[2], v[0])),  # opposite v[1]
            (min(v[0], v[1]), max(v[0], v[1])),  # opposite v[2]
        ]
        for j, e in enumerate(edges):
            tri_list = edge_to_tri[e]
            if len(tri_list) == 2:
                # Two triangles share this edge
                neighbors[t, j] = tri_list[0] if tri_list[1] == t else tri_list[1]
            # else: boundary edge, stays -1

    return neighbors


def compute_mesh_quality(
    nodes: np.ndarray,
    elements: np.ndarray,
) -> np.ndarray:
    """
    Compute the quality metric for each triangle.

    For a triangle with vertices A, B, C:

        a = |B - C|,  b = |C - A|,  c = |A - B|
        s = (a + b + c) / 2
        Area = sqrt(s(s-a)(s-b)(s-c))

    Quality (normalized inradius-to-circumradius ratio):

        q = (b+c-a)(c+a-b)(a+b-c) / (a·b·c)

    q ∈ (0, 1], with q = 1 for equilateral triangles.

    Parameters
    ----------
    nodes : np.ndarray
        Node coordinates.
    elements : np.ndarray
        Triangle indices.

    Returns
    -------
    quality : np.ndarray
        Quality metric for each triangle.
    """
    n_tri = elements.shape[0]
    quality = np.zeros(n_tri)

    for t in range(n_tri):
        v = elements[t]
        A = nodes[v[0]]
        B = nodes[v[1]]
        C = nodes[v[2]]

        a = np.linalg.norm(B - C)
        b = np.linalg.norm(C - A)
        c = np.linalg.norm(A - B)

        # Check for degenerate triangle
        if a < 1e-15 or b < 1e-15 or c < 1e-15:
            quality[t] = 0.0
            continue

        # Quality formula
        num = (b + c - a) * (c + a - b) * (a + b - c)
        den = a * b * c
        if den < 1e-15:
            quality[t] = 0.0
        else:
            q = num / den
            quality[t] = max(0.0, min(1.0, q))

    return quality


def compute_element_areas(
    nodes: np.ndarray,
    elements: np.ndarray,
) -> np.ndarray:
    """
    Compute the area of each triangle using the cross product formula:

        Area = 0.5 · |(B-A) × (C-A)|

    Parameters
    ----------
    nodes : np.ndarray
        Node coordinates.
    elements : np.ndarray
        Triangle indices.

    Returns
    -------
    areas : np.ndarray
        Area of each triangle.
    """
    n_tri = elements.shape[0]
    areas = np.zeros(n_tri)

    for t in range(n_tri):
        v = elements[t]
        A = nodes[v[0]]
        B = nodes[v[1]]
        C = nodes[v[2]]

        # 2D cross product
        cross = (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])
        areas[t] = 0.5 * abs(cross)

    return areas


def simulate_cortical_mesh_analysis(
    n_boundary: int = 40,
    n_interior: int = 150,
) -> dict:
    """
    Generate a cortical mesh and compute comprehensive metrics.

    Parameters
    ----------
    n_boundary : int
        Boundary vertices.
    n_interior : int
        Interior vertices.

    Returns
    -------
    results : dict
        Mesh data and quality metrics.
    """
    nodes, elements = generate_cortical_mesh(n_boundary, n_interior)
    neighbors = compute_triangle_neighbors(elements)
    quality = compute_mesh_quality(nodes, elements)
    areas = compute_element_areas(nodes, elements)

    n_tri = elements.shape[0]
    boundary_edges = np.sum(neighbors == -1)

    return {
        "nodes": nodes,
        "elements": elements,
        "neighbors": neighbors,
        "quality": quality,
        "areas": areas,
        "n_nodes": nodes.shape[0],
        "n_triangles": n_tri,
        "n_boundary_edges": boundary_edges,
        "mean_quality": np.mean(quality),
        "min_quality": np.min(quality),
        "total_area": np.sum(areas),
    }


if __name__ == "__main__":
    results = simulate_cortical_mesh_analysis()
    print(f"Nodes: {results['n_nodes']}, Triangles: {results['n_triangles']}")
    print(f"Mean quality: {results['mean_quality']:.4f}")
    print(f"Min quality: {results['min_quality']:.4f}")
    print(f"Total area: {results['total_area']:.2f} μm²")
    print(f"Boundary edges: {results['n_boundary_edges']}")
