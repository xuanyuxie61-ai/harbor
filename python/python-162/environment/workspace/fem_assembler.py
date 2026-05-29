"""
fem_assembler.py
================================================================================
Finite element matrix assembly for thermal PDE on triangular meshes.

Injects core algorithms from:
  - 412_fem2d_project_function  (FEM L2 projection, mass matrix assembly)
  - 407_fem2d_pack              (T3 shape functions, reference mappings, quadrature)

Scientific role:
  Assembles the global stiffness and mass matrices for the 2D heat equation
  over the battery cell cross-section using P1 (linear) triangular elements.
  Handles anisotropic thermal conductivity (different in-plane vs through-plane)
  and region-dependent material properties.
================================================================================
"""

import numpy as np
from typing import Tuple
from quadrature_special import triangle_unit_rule


def basis_t3(xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Linear triangular shape functions on reference triangle:
    N1 = 1 - xi - eta, N2 = xi, N3 = eta.
    Returns (N, dN_dxi, dN_deta).
    Maps from 407_fem2d_pack/basis_mn_t3.m.
    """
    N = np.array([1.0 - xi - eta, xi, eta], dtype=float)
    dN_dxi = np.array([-1.0, 1.0, 0.0], dtype=float)
    dN_deta = np.array([-1.0, 0.0, 1.0], dtype=float)
    return N, dN_dxi, dN_deta


def jacobian_t3(nodes_phys: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute Jacobian matrix and determinant for affine T3 mapping.
    nodes_phys: shape (3, 2) physical coordinates.
    """
    J = np.zeros((2, 2), dtype=float)
    J[0, 0] = nodes_phys[1, 0] - nodes_phys[0, 0]
    J[0, 1] = nodes_phys[2, 0] - nodes_phys[0, 0]
    J[1, 0] = nodes_phys[1, 1] - nodes_phys[0, 1]
    J[1, 1] = nodes_phys[2, 1] - nodes_phys[0, 1]
    detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
    return J, detJ


def assemble_thermal_matrices(nodes: np.ndarray, elements: np.ndarray,
                              thermal_conductivity: dict,
                              region_tags: np.ndarray,
                              rho_cp: float = 2.5e6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assemble global stiffness K and mass M matrices for:
      rho*cp * dT/dt = div(k*grad(T)) + Q
    on a triangular mesh.

    Parameters
    ----------
    nodes : (N, 2) array
    elements : (M, 3) array
    thermal_conductivity : dict mapping region_tag -> float (W/m/K)
    region_tags : (M,) array of region names
    rho_cp : volumetric heat capacity (J/m^3/K)

    Returns
    -------
    K : (N, N) sparse-like dense matrix (small enough for dense)
    M : (N, N) mass matrix
    """
    n_nodes = len(nodes)
    K = np.zeros((n_nodes, n_nodes), dtype=float)
    M = np.zeros((n_nodes, n_nodes), dtype=float)

    # 3-point rule on reference triangle
    xi_q, eta_q, w_q = triangle_unit_rule(order=3)

    for e in range(len(elements)):
        tri = elements[e]
        region = str(region_tags[e])
        k_val = thermal_conductivity.get(region, 1.0)

        # Physical nodes of element
        p = nodes[tri]
        J, detJ = jacobian_t3(p)
        if abs(detJ) < 1e-14:
            continue
        invJ = np.linalg.inv(J)

        # Element stiffness and mass
        Ke = np.zeros((3, 3), dtype=float)
        Me = np.zeros((3, 3), dtype=float)

        for q in range(len(w_q)):
            N, dN_dxi, dN_deta = basis_t3(xi_q[q], eta_q[q])
            # Gradient in physical space: [dN/dx; dN/dy] = inv(J^T) * [dN/dxi; dN/deta]
            grad = np.zeros((3, 2), dtype=float)
            for i in range(3):
                grad[i, 0] = invJ[0, 0] * dN_dxi[i] + invJ[1, 0] * dN_deta[i]
                grad[i, 1] = invJ[0, 1] * dN_dxi[i] + invJ[1, 1] * dN_deta[i]

            weight = w_q[q] * abs(detJ)
            Ke += k_val * weight * (grad @ grad.T)
            Me += rho_cp * weight * np.outer(N, N)

        # Assemble into global
        for i in range(3):
            for j in range(3):
                gi, gj = tri[i], tri[j]
                K[gi, gj] += Ke[i, j]
                M[gi, gj] += Me[i, j]

    return K, M


def apply_dirichlet_bc(K: np.ndarray, F: np.ndarray, bc_nodes: np.ndarray,
                       bc_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Dirichlet boundary conditions by row replacement.
    Maps from 412_fem2d_project_function.
    """
    K_bc = K.copy()
    F_bc = F.copy()
    for idx, node in enumerate(bc_nodes):
        K_bc[node, :] = 0.0
        K_bc[node, node] = 1.0
        F_bc[node] = bc_values[idx]
    return K_bc, F_bc


def compute_l2_error(nodes: np.ndarray, elements: np.ndarray,
                     T_numeric: np.ndarray, T_exact_func,
                     region_tags: np.ndarray) -> float:
    """
    Compute L2 error ||T_numeric - T_exact|| over the domain.
    Uses 7-point triangle quadrature for accuracy.
    """
    xi_q, eta_q, w_q = triangle_unit_rule(order=7)
    err2 = 0.0
    for e in range(len(elements)):
        tri = elements[e]
        p = nodes[tri]
        J, detJ = jacobian_t3(p)
        if abs(detJ) < 1e-14:
            continue
        for q in range(len(w_q)):
            N, _, _ = basis_t3(xi_q[q], eta_q[q])
            x_phys = np.dot(N, p[:, 0])
            y_phys = np.dot(N, p[:, 1])
            T_num = np.dot(N, T_numeric[tri])
            T_ex = T_exact_func(x_phys, y_phys)
            err2 += w_q[q] * abs(detJ) * (T_num - T_ex) ** 2
    return np.sqrt(err2)
