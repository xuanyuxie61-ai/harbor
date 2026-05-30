# -*- coding: utf-8 -*-

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
    if n_vertices < 3:
        raise ValueError("n_vertices must be >= 3.")
    if width <= 0.0 or height <= 0.0:
        raise ValueError("width and height must be positive.")

    rng = np.random.default_rng(seed)
    theta = np.linspace(0.0, 2.0 * np.pi, n_vertices, endpoint=False)


    x = 0.5 * width * np.cos(theta)
    y = 0.5 * height * np.sin(theta)


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
    if n_boundary < 3 or n_interior < 0:
        raise ValueError("Invalid vertex counts.")

    rng = np.random.default_rng(seed)


    boundary = generate_cortical_boundary(n_boundary, width, height, seed=seed)


    interior_points = []
    max_attempts = n_interior * 100
    attempts = 0

    while len(interior_points) < n_interior and attempts < max_attempts:
        px = rng.uniform(-width / 2.0, width / 2.0)
        py = rng.uniform(-height / 2.0, height / 2.0)



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


    tri = Delaunay(nodes)
    elements = tri.simplices

    return nodes, elements


def compute_triangle_neighbors(
    elements: np.ndarray,
) -> np.ndarray:
    if elements.shape[1] != 3:
        raise ValueError("elements must have shape (n, 3).")

    n_tri = elements.shape[0]
    neighbors = np.full((n_tri, 3), -1, dtype=int)


    edge_to_tri = {}

    for t in range(n_tri):
        v = elements[t]

        edges = [
            (min(v[1], v[2]), max(v[1], v[2])),
            (min(v[2], v[0]), max(v[2], v[0])),
            (min(v[0], v[1]), max(v[0], v[1])),
        ]
        for e in edges:
            if e not in edge_to_tri:
                edge_to_tri[e] = []
            edge_to_tri[e].append(t)


    for t in range(n_tri):
        v = elements[t]
        edges = [
            (min(v[1], v[2]), max(v[1], v[2])),
            (min(v[2], v[0]), max(v[2], v[0])),
            (min(v[0], v[1]), max(v[0], v[1])),
        ]
        for j, e in enumerate(edges):
            tri_list = edge_to_tri[e]
            if len(tri_list) == 2:

                neighbors[t, j] = tri_list[0] if tri_list[1] == t else tri_list[1]


    return neighbors


def compute_mesh_quality(
    nodes: np.ndarray,
    elements: np.ndarray,
) -> np.ndarray:
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


        if a < 1e-15 or b < 1e-15 or c < 1e-15:
            quality[t] = 0.0
            continue


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
    n_tri = elements.shape[0]
    areas = np.zeros(n_tri)

    for t in range(n_tri):
        v = elements[t]
        A = nodes[v[0]]
        B = nodes[v[1]]
        C = nodes[v[2]]


        cross = (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])
        areas[t] = 0.5 * abs(cross)

    return areas


def simulate_cortical_mesh_analysis(
    n_boundary: int = 40,
    n_interior: int = 150,
) -> dict:
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
