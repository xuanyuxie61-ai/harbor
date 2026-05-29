"""
tet_mesh_core.py
Tetrahedral mesh generation and volume integration for 3D molecular domains.

Derived from: 1235_tet_mesh_quad + 1241_tet_mesh_to_xml

This module constructs tetrahedral meshes around DNA and repair protein
complexes, computes exact tetrahedron volumes via the Cayley-Menger
determinant, and performs high-order quadrature for electrostatic
free-energy integrals over the molecular volume.

Key formulas:
  - Tetrahedron volume: V = |det([x2-x1, x3-x1, x4-x1])| / 6
  - Barycentric interpolation: phi(x) = sum_i w_i * phi_i,  sum_i w_i = 1
  - Integral over tet mesh: int_Omega f dV = sum_{tets} V_e * (f_1+f_2+f_3+f_4)/4
"""

import numpy as np


def tetrahedron_volume(p1, p2, p3, p4):
    """
    Compute the signed volume of a tetrahedron with vertices p1, p2, p3, p4.

    V = det([p2-p1, p3-p1, p4-p1]) / 6

    Parameters
    ----------
    p1, p2, p3, p4 : ndarray, shape (3,)
        Vertex coordinates.

    Returns
    -------
    volume : float
        Absolute volume (non-negative).
    """
    M = np.column_stack((p2 - p1, p3 - p1, p4 - p1))
    vol = np.linalg.det(M) / 6.0
    return abs(vol)


def cayley_menger_volume(p1, p2, p3, p4):
    """
    Compute tetrahedron volume via the Cayley-Menger determinant.

    For points in R^3, the squared volume is:

        288 * V^2 = det(B)

    where B is the 5x5 Cayley-Menger matrix:
        [ 0   1      1      1      1    ]
        [ 1   0   d12^2  d13^2  d14^2 ]
        [ 1 d21^2   0    d23^2  d24^2 ]
        [ 1 d31^2 d32^2    0    d34^2 ]
        [ 1 d41^2 d42^2 d43^2    0    ]

    Parameters
    ----------
    p1, p2, p3, p4 : ndarray, shape (3,)

    Returns
    -------
    volume : float
    """
    points = [p1, p2, p3, p4]
    B = np.zeros((5, 5))
    B[0, 1:] = 1.0
    B[1:, 0] = 1.0
    for i in range(4):
        for j in range(4):
            if i == j:
                B[i + 1, j + 1] = 0.0
            else:
                d2 = np.sum((points[i] - points[j]) ** 2)
                B[i + 1, j + 1] = d2
    det_B = np.linalg.det(B)
    if det_B < 0:
        # Numerical robustness: clamp to zero for near-degenerate tets
        det_B = 0.0
    vol = np.sqrt(det_B / 288.0)
    return vol


def generate_tet_mesh_box(nx=4, ny=4, nz=4, xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1)):
    """
    Generate a regular tetrahedral mesh inside a bounding box by
    subdividing each cubic cell into 6 tetrahedra.

    Parameters
    ----------
    nx, ny, nz : int
        Number of grid divisions along each axis.
    xlim, ylim, zlim : tuple of float
        Box bounds.

    Returns
    -------
    nodes : ndarray, shape (N, 3)
    elements : ndarray, shape (M, 4)
        Element node indices (0-based).
    """
    if nx < 2 or ny < 2 or nz < 2:
        raise ValueError("Grid divisions must be at least 2 in each dimension.")

    x = np.linspace(xlim[0], xlim[1], nx)
    y = np.linspace(ylim[0], ylim[1], ny)
    z = np.linspace(zlim[0], zlim[1], nz)

    # Build node list
    nodes = []
    node_index = {}
    idx = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                nodes.append([x[i], y[j], z[k]])
                node_index[(i, j, k)] = idx
                idx += 1
    nodes = np.array(nodes)

    # Subdivide each cube into 6 tetrahedra
    # Standard subdivision preserving orientation
    elements = []
    for k in range(nz - 1):
        for j in range(ny - 1):
            for i in range(nx - 1):
                v000 = node_index[(i, j, k)]
                v100 = node_index[(i + 1, j, k)]
                v010 = node_index[(i, j + 1, k)]
                v110 = node_index[(i + 1, j + 1, k)]
                v001 = node_index[(i, j, k + 1)]
                v101 = node_index[(i + 1, j, k + 1)]
                v011 = node_index[(i, j + 1, k + 1)]
                v111 = node_index[(i + 1, j + 1, k + 1)]

                # 6-tet decomposition with consistent orientation
                elements.append([v000, v100, v110, v111])
                elements.append([v000, v100, v111, v101])
                elements.append([v000, v101, v111, v001])
                elements.append([v000, v111, v011, v001])
                elements.append([v000, v011, v111, v010])
                elements.append([v000, v010, v111, v110])

    elements = np.array(elements, dtype=int)
    return nodes, elements


def integrate_over_tet_mesh(nodes, elements, nodal_values):
    """
    Integrate a scalar function over a tetrahedral mesh using
    linear (4-node) quadrature: average of vertices times volume.

    int_T f dV = V_T * (f_1 + f_2 + f_3 + f_4) / 4

    Parameters
    ----------
    nodes : ndarray, shape (N, 3)
    elements : ndarray, shape (M, 4)
    nodal_values : ndarray, shape (N,)

    Returns
    -------
    integral : float
    total_volume : float
    """
    if elements.shape[1] not in (4, 10):
        raise ValueError("Only 4-node or 10-node tetrahedra supported.")

    integral = 0.0
    total_volume = 0.0
    for e in range(elements.shape[0]):
        en = elements[e, :4]
        p = nodes[en]
        # Volume via determinant formula
        M = np.column_stack((p[1] - p[0], p[2] - p[0], p[3] - p[0]))
        vol = abs(np.linalg.det(M)) / 6.0
        if vol < 0:
            vol = 0.0
        avg_val = np.mean(nodal_values[en])
        integral += vol * avg_val
        total_volume += vol

    return integral, total_volume


def integrate_vector_over_tet_mesh(nodes, elements, nodal_values):
    """
    Integrate a vector-valued function over a tetrahedral mesh.

    Parameters
    ----------
    nodes : ndarray, shape (N, 3)
    elements : ndarray, shape (M, 4)
    nodal_values : ndarray, shape (N, D)

    Returns
    -------
    integral : ndarray, shape (D,)
    total_volume : float
    """
    if elements.shape[1] not in (4, 10):
        raise ValueError("Only 4-node or 10-node tetrahedra supported.")

    D = nodal_values.shape[1] if nodal_values.ndim > 1 else 1
    integral = np.zeros(D)
    total_volume = 0.0

    for e in range(elements.shape[0]):
        en = elements[e, :4]
        p = nodes[en]
        M = np.column_stack((p[1] - p[0], p[2] - p[0], p[3] - p[0]))
        vol = abs(np.linalg.det(M)) / 6.0
        if vol < 0:
            vol = 0.0
        if D == 1:
            avg_val = np.mean(nodal_values[en])
            integral[0] += vol * avg_val
        else:
            avg_val = np.mean(nodal_values[en, :], axis=0)
            integral += vol * avg_val
        total_volume += vol

    return integral, total_volume


def tet_mesh_gradients(nodes, elements, nodal_values):
    """
    Compute the piecewise-constant gradient of a scalar field on a tet mesh.

    For a linear tetrahedron with vertices x_i and values u_i:
        grad(u) = sum_i u_i * grad(lambda_i)
    where lambda_i are the barycentric coordinates. For a tet:
        grad(lambda_i) = (n_i * A_i) / (3 * V)
    with n_i the inward normal of face opposite vertex i, A_i its area.
    Equivalently, if X = [x2-x1, x3-x1, x4-x1], then:
        grad(u) = X^{-T} * [u2-u1, u3-u1, u4-u1]^T

    Parameters
    ----------
    nodes : ndarray, shape (N, 3)
    elements : ndarray, shape (M, 4)
    nodal_values : ndarray, shape (N,)

    Returns
    -------
    gradients : ndarray, shape (M, 3)
        Gradient vector in each tetrahedron.
    """
    M = elements.shape[0]
    gradients = np.zeros((M, 3))
    for e in range(M):
        en = elements[e, :4]
        p = nodes[en]
        X = np.column_stack((p[1] - p[0], p[2] - p[0], p[3] - p[0]))
        det_X = np.linalg.det(X)
        if abs(det_X) < 1e-14:
            gradients[e] = 0.0
            continue
        du = np.array([
            nodal_values[en[1]] - nodal_values[en[0]],
            nodal_values[en[2]] - nodal_values[en[0]],
            nodal_values[en[3]] - nodal_values[en[0]],
        ])
        try:
            grad = np.linalg.solve(X.T, du)
        except np.linalg.LinAlgError:
            grad = np.zeros(3)
        gradients[e] = grad
    return gradients
