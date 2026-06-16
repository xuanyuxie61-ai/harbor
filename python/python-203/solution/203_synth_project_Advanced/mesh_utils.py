"""
mesh_utils.py -- Mesh Processing Utilities
============================================
Provides mesh orientation checking/repair, mesh format conversion,
and triangulation utilities for the stochastic FEM mesh pipeline.

Operations:
1. Check and enforce positive (counter-clockwise) triangle orientation
2. Convert between FEM and MEDIT mesh formats
3. Generate simple 2D triangular meshes for stochastic domain decomposition

Seed references:
  - 1344_triangulation_orient: signed area test, vertex swapping for reorientation
  - 379_fem_to_medit: FEM-to-MEDIT format conversion, dimension detection
"""
import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# Triangle orientation
# ---------------------------------------------------------------------------
def triangle_area_2d(coords: np.ndarray) -> float:
    """
    Compute the signed area of a 2D triangle using the cross product formula:
        area = 0.5 * (x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))

    Positive area => counter-clockwise vertex ordering.
    Negative area => clockwise (needs reorientation).

    Parameters
    ----------
    coords : ndarray, shape (2, 3)
        Vertex coordinates: [[x1,x2,x3],[y1,y2,y3]].

    Returns
    -------
    area : float
        Signed area of the triangle.
    """
    x1, x2, x3 = coords[0, 0], coords[0, 1], coords[0, 2]
    y1, y2, y3 = coords[1, 0], coords[1, 1], coords[1, 2]
    return 0.5 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))


def orient_triangles(node_coords: np.ndarray,
                     element_nodes: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Check and enforce counter-clockwise orientation of all triangles.

    For each triangle with negative signed area, swap vertices 2 and 3
    (and for 6-node triangles, also swap mid-side nodes 4 and 6).

    Parameters
    ----------
    node_coords : ndarray, shape (n_nodes, 2)
        2D coordinates of all mesh nodes.
    element_nodes : ndarray, shape (n_elements, 3) or (n_elements, 6)
        Element connectivity (0-based or 1-based).

    Returns
    -------
    oriented_elements : ndarray
        Element connectivity with all triangles positively oriented.
    stats : dict
        'n_negative': number of triangles that were reoriented.
        'n_zero': number of degenerate (zero area) triangles.
        'n_total': total number of triangles.
    """
    elements = element_nodes.copy()
    n_elem, order = elements.shape

    # Detect 0-based indexing
    min_idx = np.min(elements)
    is_zero_based = (min_idx == 0)

    n_negative = 0
    n_zero = 0

    for e in range(n_elem):
        # Get vertex indices (convert to 0-based for indexing)
        if is_zero_based:
            v = elements[e, :3]
        else:
            v = elements[e, :3] - 1

        coords = np.zeros((2, 3))
        coords[0, :] = node_coords[v, 0]
        coords[1, :] = node_coords[v, 1]

        area = triangle_area_2d(coords)

        if area < -1e-14:
            # Swap vertices 2 and 3
            elements[e, 1], elements[e, 2] = elements[e, 2], elements[e, 1]
            if order == 6:
                # Also swap mid-side nodes
                elements[e, 3], elements[e, 5] = elements[e, 5], elements[e, 3]
            n_negative += 1
        elif abs(area) <= 1e-14:
            n_zero += 1

    stats = {
        'n_negative': n_negative,
        'n_zero': n_zero,
        'n_total': n_elem
    }

    return elements, stats


# ---------------------------------------------------------------------------
# Mesh generation for stochastic domain
# ---------------------------------------------------------------------------
def generate_stochastic_mesh(n_x: int, n_y: int, domain: Tuple = (0.0, 1.0, 0.0, 1.0)
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a simple structured triangular mesh on a rectangular domain.
    Each rectangle is split into 2 triangles (diagonal from bottom-left to top-right).

    Parameters
    ----------
    n_x, n_y : int
        Number of elements in x and y directions.
    domain : tuple
        (x_min, x_max, y_min, y_max).

    Returns
    -------
    nodes : ndarray, shape (n_nodes, 2)
        Node coordinates.
    elements : ndarray, shape (n_elements, 3)
        Triangle connectivity (0-based).
    boundary_mask : ndarray, shape (n_nodes,)
        1 if node is on the boundary, 0 if interior.
    """
    x_min, x_max, y_min, y_max = domain
    dx = (x_max - x_min) / n_x
    dy = (y_max - y_min) / n_y

    # Generate nodes
    nodes = []
    for j in range(n_y + 1):
        for i in range(n_x + 1):
            nodes.append([x_min + i * dx, y_min + j * dy])
    nodes = np.array(nodes)

    # Generate triangles (2 per quad cell)
    elements = []
    for j in range(n_y):
        for i in range(n_x):
            n0 = j * (n_x + 1) + i
            n1 = n0 + 1
            n2 = n0 + (n_x + 1)
            n3 = n2 + 1
            # Lower triangle
            elements.append([n0, n1, n2])
            # Upper triangle
            elements.append([n1, n3, n2])
    elements = np.array(elements)

    # Boundary mask
    boundary_mask = np.zeros(len(nodes), dtype=int)
    for idx, (x, y) in enumerate(nodes):
        if (abs(x - x_min) < 1e-10 or abs(x - x_max) < 1e-10 or
                abs(y - y_min) < 1e-10 or abs(y - y_max) < 1e-10):
            boundary_mask[idx] = 1

    return nodes, elements, boundary_mask


# ---------------------------------------------------------------------------
# FEM to MEDIT format conversion
# ---------------------------------------------------------------------------
def fem_to_medit(nodes: np.ndarray, elements: np.ndarray,
                 boundary_mask: np.ndarray = None) -> str:
    """
    Convert FEM mesh data to MEDIT .mesh format string.

    The MEDIT format:
        MeshVersionFormatted 2
        Dimension 2
        Vertices
          <n_vertices>
          x y label
        ...
        Triangles
          <n_triangles>
          v1 v2 v3 label
        ...
        End

    Parameters
    ----------
    nodes : ndarray, shape (n_nodes, dim)
        Node coordinates.
    elements : ndarray, shape (n_elem, 3)
        Triangle connectivity (0-based).
    boundary_mask : ndarray or None
        Node boundary labels.

    Returns
    -------
    mesh_str : str
        MEDIT format mesh string.
    """
    dim = nodes.shape[1]
    n_nodes = nodes.shape[0]
    n_elem = elements.shape[0]

    lines = []
    lines.append("MeshVersionFormatted 2")
    lines.append(f"Dimension {dim}")
    lines.append("")

    # Vertices
    lines.append("Vertices")
    lines.append(str(n_nodes))
    for i in range(n_nodes):
        label = 1 if (boundary_mask is not None and boundary_mask[i] > 0) else 0
        coords = " ".join(f"{nodes[i, d]:.10e}" for d in range(dim))
        lines.append(f"{coords} {label}")
    lines.append("")

    # Triangles
    lines.append("Triangles")
    lines.append(str(n_elem))
    for i in range(n_elem):
        # MEDIT uses 1-based indexing
        verts = " ".join(str(elements[i, j] + 1) for j in range(3))
        lines.append(f"{verts} 0")
    lines.append("")
    lines.append("End")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stochastic mesh refinement indicators
# ---------------------------------------------------------------------------
def compute_mesh_indicators(nodes: np.ndarray, elements: np.ndarray,
                            solution_field: np.ndarray) -> np.ndarray:
    """
    Compute element-wise error indicators for adaptive mesh refinement
    in the stochastic domain.

    Uses the gradient-based indicator:
        eta_K = h_K * ||grad(u_h)||_K
    where h_K is the element diameter and grad is computed from the
    piecewise-linear finite element solution.

    Parameters
    ----------
    nodes : ndarray, shape (n_nodes, 2)
    elements : ndarray, shape (n_elem, 3)
    solution_field : ndarray, shape (n_nodes,)
        FE solution at nodes.

    Returns
    -------
    indicators : ndarray, shape (n_elem,)
        Error indicator per element.
    """
    n_elem = elements.shape[0]
    indicators = np.zeros(n_elem)

    for e in range(n_elem):
        v = elements[e]
        coords = nodes[v]  # (3, 2)
        vals = solution_field[v]  # (3,)

        # Compute gradient using the element Jacobian
        # grad(u) = J^{-T} * grad_ref(u)
        # J = [[x2-x1, x3-x1], [y2-y1, y3-y1]]
        J = np.array([
            [coords[1, 0] - coords[0, 0], coords[2, 0] - coords[0, 0]],
            [coords[1, 1] - coords[0, 1], coords[2, 1] - coords[0, 1]]
        ])

        det_J = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if abs(det_J) < 1e-14:
            continue

        J_inv = np.array([
            [J[1, 1], -J[0, 1]],
            [-J[1, 0], J[0, 0]]
        ]) / det_J

        # Reference gradients of basis functions
        grad_ref = np.array([[-1, -1], [1, 0], [0, 1]]).T  # (2, 3)

        # Physical gradient
        grad_phys = J_inv @ grad_ref @ vals  # (2,)

        # Element diameter
        h = np.max([np.linalg.norm(coords[i] - coords[j])
                     for i in range(3) for j in range(i + 1, 3)])

        # Area
        area = 0.5 * abs(det_J)

        indicators[e] = h * np.linalg.norm(grad_phys) * np.sqrt(area)

    return indicators
