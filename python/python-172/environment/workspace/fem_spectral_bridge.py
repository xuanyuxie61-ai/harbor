# -*- coding: utf-8 -*-
"""
Spectral-to-FEM Projection and Mesh Topology
=============================================
Bridges spectral solutions with finite element representations for
cross-validation and error estimation. Includes mesh topology utilities
inspired by tetrahedral neighbor computations.

Inspired by:
- fem1d_project: L2 projection onto FEM mesh using Gaussian quadrature
- tet_mesh_tet_neighbors: adjacency topology for mesh elements

Mathematical formulation:
- Given a spectral solution u_spectral on CGL nodes, we seek FEM coefficients
  u_fem on a linear P1 mesh such that:
      (phi_i, phi_j) u_fem_j = (phi_i, u_spectral)
  where (.,.) is the L2 inner product and phi_i are hat functions.
- The mass matrix M_{ij} = integral phi_i phi_j dx is assembled element-wise
  using 2-point Gaussian quadrature (nodes ±1/sqrt(3), weights 1).
- The right-hand side b_i = integral phi_i u_spectral dx uses exact
  integration of piecewise-linear products.
"""

import numpy as np


def gauss_legendre_2point():
    """
    Return nodes and weights for 2-point Gauss-Legendre quadrature on [-1,1].
    Nodes: ±1/sqrt(3), Weights: 1.
    """
    nodes = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
    weights = np.array([1.0, 1.0])
    return nodes, weights


def build_fem_mass_matrix(fem_nodes, elements):
    """
    Build the Galerkin mass matrix for P1 finite elements.

    Parameters
    ----------
    fem_nodes : ndarray
        FEM node coordinates.
    elements : ndarray, shape (n_elem, 2)
        Element connectivity (1D line segments).

    Returns
    -------
    M : ndarray
        Mass matrix.
    """
    n_nodes = len(fem_nodes)
    M = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    gp, gw = gauss_legendre_2point()

    for elem in elements:
        i, j = elem
        x_i, x_j = fem_nodes[i], fem_nodes[j]
        h = x_j - x_i
        if abs(h) < 1e-15:
            continue
        # Transform quadrature to element [x_i, x_j]
        x_gp = 0.5 * (x_j + x_i) + 0.5 * h * gp
        w_gp = 0.5 * h * gw

        for q in range(len(gp)):
            xq = x_gp[q]
            wq = w_gp[q]
            # P1 shape functions at xq
            phi_i = (x_j - xq) / h
            phi_j = (xq - x_i) / h
            M[i, i] += wq * phi_i * phi_i
            M[i, j] += wq * phi_i * phi_j
            M[j, i] += wq * phi_j * phi_i
            M[j, j] += wq * phi_j * phi_j
    return M


def fem_rhs_projection(fem_nodes, elements, u_func):
    """
    Build the right-hand side for L2 projection of a function onto P1 FEM.

    Parameters
    ----------
    fem_nodes : ndarray
    elements : ndarray
    u_func : callable
        Function to project.

    Returns
    -------
    b : ndarray
        Right-hand side vector.
    """
    n_nodes = len(fem_nodes)
    b = np.zeros(n_nodes, dtype=np.float64)
    gp, gw = gauss_legendre_2point()

    for elem in elements:
        i, j = elem
        x_i, x_j = fem_nodes[i], fem_nodes[j]
        h = x_j - x_i
        if abs(h) < 1e-15:
            continue
        x_gp = 0.5 * (x_j + x_i) + 0.5 * h * gp
        w_gp = 0.5 * h * gw

        for q in range(len(gp)):
            xq = x_gp[q]
            wq = w_gp[q]
            phi_i = (x_j - xq) / h
            phi_j = (xq - x_i) / h
            uq = u_func(xq)
            b[i] += wq * phi_i * uq
            b[j] += wq * phi_j * uq
    return b


def spectral_to_fem_projection(spectral_nodes, spectral_values,
                                fem_nodes, elements):
    """
    Project a spectral solution onto a P1 finite element mesh.

    Parameters
    ----------
    spectral_nodes : ndarray
        Spectral grid points.
    spectral_values : ndarray
        Solution values on spectral grid.
    fem_nodes : ndarray
        FEM node coordinates.
    elements : ndarray
        FEM element connectivity.

    Returns
    -------
    u_fem : ndarray
        FEM nodal coefficients.
    l2_error : float
        Estimated L2 projection error.
    """
    from scipy.interpolate import interp1d
    # Interpolate spectral solution to arbitrary points
    u_interp = interp1d(spectral_nodes, spectral_values, kind='cubic',
                        fill_value='extrapolate', bounds_error=False)

    M = build_fem_mass_matrix(fem_nodes, elements)
    b = fem_rhs_projection(fem_nodes, elements, u_interp)

    # Add small regularization for robustness
    M += 1e-12 * np.eye(len(fem_nodes))
    u_fem = np.linalg.solve(M, b)

    # Estimate L2 error by sampling
    x_test = np.linspace(fem_nodes[0], fem_nodes[-1], 200)
    u_spec_test = u_interp(x_test)
    u_fem_test = np.zeros_like(x_test)
    for elem in elements:
        i, j = elem
        mask = (x_test >= fem_nodes[i]) & (x_test <= fem_nodes[j])
        if np.any(mask):
            h = fem_nodes[j] - fem_nodes[i]
            phi_i = (fem_nodes[j] - x_test[mask]) / h
            phi_j = (x_test[mask] - fem_nodes[i]) / h
            u_fem_test[mask] = phi_i * u_fem[i] + phi_j * u_fem[j]

    l2_error = np.sqrt(np.trapezoid((u_spec_test - u_fem_test) ** 2, x_test))
    return u_fem, l2_error


def build_triangle_neighbors(triangles):
    """
    Compute triangle neighbor information (2D analog of tetrahedral neighbors).
    For each triangle and each edge, returns the index of the neighboring
    triangle sharing that edge, or -1 for boundary edges.

    Parameters
    ----------
    triangles : ndarray, shape (n_tri, 3)
        Triangle vertex indices.

    Returns
    -------
    neighbors : ndarray, shape (n_tri, 3)
        Neighbor indices.
    """
    triangles = np.asarray(triangles, dtype=int)
    n_tri = len(triangles)
    neighbors = np.full((n_tri, 3), -1, dtype=int)

    # Create edge signatures: sort vertex pairs and map to triangles
    edge_dict = {}
    for t in range(n_tri):
        verts = triangles[t]
        edges = [(verts[1], verts[2]), (verts[2], verts[0]), (verts[0], verts[1])]
        for e_idx, e in enumerate(edges):
            edge_key = tuple(sorted(e))
            if edge_key in edge_dict:
                # Found neighbor
                other_t, other_e = edge_dict[edge_key]
                neighbors[t, e_idx] = other_t
                neighbors[other_t, other_e] = t
            else:
                edge_dict[edge_key] = (t, e_idx)
    return neighbors


def build_1d_element_neighbors(n_nodes):
    """
    Build 1D element neighbor array (simplified topology).

    Parameters
    ----------
    n_nodes : int
        Number of nodes.

    Returns
    -------
    elements : ndarray
        Element connectivity.
    """
    elements = np.zeros((n_nodes - 1, 2), dtype=int)
    for i in range(n_nodes - 1):
        elements[i] = [i, i + 1]
    return elements
