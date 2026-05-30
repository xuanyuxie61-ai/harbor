
import numpy as np


def triangle_refine(
    nodes: np.ndarray,
    triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    nodes = np.asarray(nodes, dtype=float)
    triangles = np.asarray(triangles, dtype=int)

    n_nodes = len(nodes)
    n_tri = len(triangles)


    edges = []
    edge_to_midpoint = {}
    edge_tri_pairs = []

    for t in range(n_tri):
        tri = triangles[t]
        tri_edges = []
        for e in range(3):
            n1 = tri[e]
            n2 = tri[(e + 1) % 3]
            key = tuple(sorted((n1, n2)))
            if key not in edge_to_midpoint:
                edge_to_midpoint[key] = n_nodes + len(edges)
                edges.append(key)
            tri_edges.append(edge_to_midpoint[key])
        edge_tri_pairs.append(tri_edges)


    new_nodes_list = [nodes]
    for e1, e2 in edges:
        mid = 0.5 * (nodes[e1] + nodes[e2])
        new_nodes_list.append(mid.reshape(1, -1))

    new_nodes = np.vstack(new_nodes_list)


    new_triangles_list = []
    for t in range(n_tri):
        tri = triangles[t]
        m01 = edge_to_midpoint[tuple(sorted((tri[0], tri[1])))]
        m12 = edge_to_midpoint[tuple(sorted((tri[1], tri[2])))]
        m20 = edge_to_midpoint[tuple(sorted((tri[2], tri[0])))]

        new_triangles_list.extend([
            [tri[0], m01, m20],
            [m01, tri[1], m12],
            [m20, m12, tri[2]],
            [m01, m12, m20],
        ])

    new_triangles = np.array(new_triangles_list, dtype=int)
    return new_nodes, new_triangles


def refine_mesh_multiple(
    nodes: np.ndarray,
    triangles: np.ndarray,
    n_refinements: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(n_refinements):
        nodes, triangles = triangle_refine(nodes, triangles)
    return nodes, triangles


def create_uniform_triangular_mesh(
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    nx: int, ny: int
) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    xv, yv = np.meshgrid(x, y, indexing='ij')
    nodes = np.column_stack([xv.ravel(), yv.ravel()])

    triangles = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            n00 = i * ny + j
            n10 = (i + 1) * ny + j
            n01 = i * ny + (j + 1)
            n11 = (i + 1) * ny + (j + 1)
            triangles.append([n00, n10, n01])
            triangles.append([n10, n11, n01])

    triangles = np.array(triangles, dtype=int)
    return nodes, triangles


def field_gradient_on_mesh(
    field: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    dx = x_coords[1] - x_coords[0]
    dy = y_coords[1] - y_coords[0]
    dfdx, dfdy = np.gradient(field, dx, dy)
    return dfdx, dfdy


def adaptive_refinement_indicator(
    field: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    threshold: float = 0.1
) -> np.ndarray:
    dfdx, dfdy = field_gradient_on_mesh(field, x_coords, y_coords)
    grad_mag = np.sqrt(dfdx ** 2 + dfdy ** 2)
    max_grad = np.max(grad_mag)
    if max_grad == 0:
        return np.zeros_like(field, dtype=bool)
    return grad_mag > threshold * max_grad
