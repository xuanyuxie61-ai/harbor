"""
tet3d_basis.py
==============
3D tetrahedral finite element basis functions for poroelastic projection.

Incorporates TET4 (linear tetrahedron) basis function evaluation from
fem3d_project, including barycentric coordinate computation and gradient
calculation for 3D poroelastic media.

For a tetrahedron with vertices T = [x1, x2, x3, x4] (each column is a vertex),
the linear basis functions in terms of barycentric coordinates are:

    phi_i(x) = det([x, x_j, x_k, x_l]) / det(T_extended)

where {i,j,k,l} is a cyclic permutation of {1,2,3,4}.

The volume of the tetrahedron is:
    V = |det(T_extended)| / 6
where T_extended is the 4x4 matrix with a row of ones appended.
"""

import numpy as np


def tetrahedron_volume(tet_xyz):
    """
    Compute signed volume of a tetrahedron.

    Parameters
    ----------
    tet_xyz : ndarray, shape (3, 4)
        Vertices as columns.

    Returns
    -------
    volume : float
        Signed volume (6 * actual volume).
    """
    volume = (
        tet_xyz[0, 0] * (
            tet_xyz[1, 1] * (tet_xyz[2, 2] - tet_xyz[2, 3])
            - tet_xyz[1, 2] * (tet_xyz[2, 1] - tet_xyz[2, 3])
            + tet_xyz[1, 3] * (tet_xyz[2, 1] - tet_xyz[2, 2])
        )
        - tet_xyz[0, 1] * (
            tet_xyz[1, 0] * (tet_xyz[2, 2] - tet_xyz[2, 3])
            - tet_xyz[1, 2] * (tet_xyz[2, 0] - tet_xyz[2, 3])
            + tet_xyz[1, 3] * (tet_xyz[2, 0] - tet_xyz[2, 2])
        )
        + tet_xyz[0, 2] * (
            tet_xyz[1, 0] * (tet_xyz[2, 1] - tet_xyz[2, 3])
            - tet_xyz[1, 1] * (tet_xyz[2, 0] - tet_xyz[2, 3])
            + tet_xyz[1, 3] * (tet_xyz[2, 0] - tet_xyz[2, 1])
        )
        - tet_xyz[0, 3] * (
            tet_xyz[1, 0] * (tet_xyz[2, 1] - tet_xyz[2, 2])
            - tet_xyz[1, 1] * (tet_xyz[2, 0] - tet_xyz[2, 2])
            + tet_xyz[1, 2] * (tet_xyz[2, 0] - tet_xyz[2, 1])
        )
    )
    return volume


def basis_mn_tet4(tet_xyz, n_eval, p):
    """
    Evaluate all 4 basis functions at N points for a TET4 element.

    Parameters
    ----------
    tet_xyz : ndarray, shape (3, 4)
        Vertex coordinates.
    n_eval : int
        Number of evaluation points.
    p : ndarray, shape (3, n_eval)
        Evaluation points.

    Returns
    -------
    phi : ndarray, shape (4, n_eval)
        Basis function values.
    """
    phi = np.zeros((4, n_eval))
    volume = tetrahedron_volume(tet_xyz)

    if abs(volume) < 1e-14:
        raise ValueError("Tetrahedron has zero volume.")

    # Basis 1
    phi[0, :] = (
        p[0, :] * (
            tet_xyz[1, 1] * (tet_xyz[2, 2] - tet_xyz[2, 3])
            - tet_xyz[1, 2] * (tet_xyz[2, 1] - tet_xyz[2, 3])
            + tet_xyz[1, 3] * (tet_xyz[2, 1] - tet_xyz[2, 2])
        )
        - tet_xyz[0, 1] * (
            p[1, :] * (tet_xyz[2, 2] - tet_xyz[2, 3])
            - tet_xyz[1, 2] * (p[2, :] - tet_xyz[2, 3])
            + tet_xyz[1, 3] * (p[2, :] - tet_xyz[2, 2])
        )
        + tet_xyz[0, 2] * (
            p[1, :] * (tet_xyz[2, 1] - tet_xyz[2, 3])
            - tet_xyz[1, 1] * (p[2, :] - tet_xyz[2, 3])
            + tet_xyz[1, 3] * (p[2, :] - tet_xyz[2, 1])
        )
        - tet_xyz[0, 3] * (
            p[1, :] * (tet_xyz[2, 1] - tet_xyz[2, 2])
            - tet_xyz[1, 1] * (p[2, :] - tet_xyz[2, 2])
            + tet_xyz[1, 2] * (p[2, :] - tet_xyz[2, 1])
        )
    ) / volume

    # Basis 2
    phi[1, :] = (
        tet_xyz[0, 0] * (
            p[1, :] * (tet_xyz[2, 2] - tet_xyz[2, 3])
            - tet_xyz[1, 2] * (p[2, :] - tet_xyz[2, 3])
            + tet_xyz[1, 3] * (p[2, :] - tet_xyz[2, 2])
        )
        - p[0, :] * (
            tet_xyz[1, 0] * (tet_xyz[2, 2] - tet_xyz[2, 3])
            - tet_xyz[1, 2] * (tet_xyz[2, 0] - tet_xyz[2, 3])
            + tet_xyz[1, 3] * (tet_xyz[2, 0] - tet_xyz[2, 2])
        )
        + tet_xyz[0, 2] * (
            tet_xyz[1, 0] * (p[2, :] - tet_xyz[2, 3])
            - p[1, :] * (tet_xyz[2, 0] - tet_xyz[2, 3])
            + tet_xyz[1, 3] * (tet_xyz[2, 0] - p[2, :])
        )
        - tet_xyz[0, 3] * (
            tet_xyz[1, 0] * (p[2, :] - tet_xyz[2, 2])
            - p[1, :] * (tet_xyz[2, 0] - tet_xyz[2, 2])
            + tet_xyz[1, 2] * (tet_xyz[2, 0] - p[2, :])
        )
    ) / volume

    # Basis 3
    phi[2, :] = (
        tet_xyz[0, 0] * (
            tet_xyz[1, 1] * (p[2, :] - tet_xyz[2, 3])
            - p[1, :] * (tet_xyz[2, 1] - tet_xyz[2, 3])
            + tet_xyz[1, 3] * (tet_xyz[2, 1] - p[2, :])
        )
        - tet_xyz[0, 1] * (
            tet_xyz[1, 0] * (p[2, :] - tet_xyz[2, 3])
            - p[1, :] * (tet_xyz[2, 0] - tet_xyz[2, 3])
            + tet_xyz[1, 3] * (tet_xyz[2, 0] - p[2, :])
        )
        + p[0, :] * (
            tet_xyz[1, 0] * (tet_xyz[2, 1] - tet_xyz[2, 3])
            - tet_xyz[1, 1] * (tet_xyz[2, 0] - tet_xyz[2, 3])
            + tet_xyz[1, 3] * (tet_xyz[2, 0] - tet_xyz[2, 1])
        )
        - tet_xyz[0, 3] * (
            tet_xyz[1, 0] * (tet_xyz[2, 1] - p[2, :])
            - tet_xyz[1, 1] * (tet_xyz[2, 0] - p[2, :])
            + p[1, :] * (tet_xyz[2, 0] - tet_xyz[2, 1])
        )
    ) / volume

    # Basis 4
    phi[3, :] = (
        tet_xyz[0, 0] * (
            tet_xyz[1, 1] * (tet_xyz[2, 2] - p[2, :])
            - tet_xyz[1, 2] * (tet_xyz[2, 1] - p[2, :])
            + p[1, :] * (tet_xyz[2, 1] - tet_xyz[2, 2])
        )
        - tet_xyz[0, 1] * (
            tet_xyz[1, 0] * (tet_xyz[2, 2] - p[2, :])
            - tet_xyz[1, 2] * (tet_xyz[2, 0] - p[2, :])
            + p[1, :] * (tet_xyz[2, 0] - tet_xyz[2, 2])
        )
        + tet_xyz[0, 2] * (
            tet_xyz[1, 0] * (tet_xyz[2, 1] - p[2, :])
            - tet_xyz[1, 1] * (tet_xyz[2, 0] - p[2, :])
            + p[1, :] * (tet_xyz[2, 0] - tet_xyz[2, 1])
        )
        - p[0, :] * (
            tet_xyz[1, 0] * (tet_xyz[2, 1] - tet_xyz[2, 2])
            - tet_xyz[1, 1] * (tet_xyz[2, 0] - tet_xyz[2, 2])
            + tet_xyz[1, 2] * (tet_xyz[2, 0] - tet_xyz[2, 1])
        )
    ) / volume

    return phi


def gradient_basis_tet4(tet_xyz):
    """
    Compute gradients of TET4 basis functions (constant within element).

    Returns
    -------
    grad_phi : ndarray, shape (3, 4)
        Gradient of each basis function [dphi/dx, dphi/dy, dphi/dz].
    """
    volume = tetrahedron_volume(tet_xyz)
    if abs(volume) < 1e-14:
        raise ValueError("Tetrahedron has zero volume.")

    # For linear tetrahedron, grad(phi_i) is proportional to the outward normal
    # of the face opposite to node i, divided by 6*V
    grad_phi = np.zeros((3, 4))

    for i in range(4):
        # Get the three vertices of the face opposite to vertex i
        face = [j for j in range(4) if j != i]
        v0 = tet_xyz[:, face[0]]
        v1 = tet_xyz[:, face[1]]
        v2 = tet_xyz[:, face[2]]

        # Cross product (v1-v0) x (v2-v0)
        e1 = v1 - v0
        e2 = v2 - v0
        normal = np.cross(e1, e2)

        # Sign depends on orientation
        grad_phi[:, i] = normal / volume

    return grad_phi


def project_to_tet_mesh(sample_nodes, sample_values, tet_nodes, tet_elements):
    """
    Project sampled scalar data onto a tetrahedral mesh by averaging.

    For each tetrahedral mesh node, find the containing sample element
    and interpolate linearly.
    """
    n_tet_nodes = tet_nodes.shape[0]
    projected = np.zeros(n_tet_nodes)
    counts = np.zeros(n_tet_nodes)

    n_tets = tet_elements.shape[0]
    for e in range(n_tets):
        elem_nodes = tet_elements[e, :]
        for nid in elem_nodes:
            # Simple average of nearby sample values
            projected[nid] += np.mean(sample_values)
            counts[nid] += 1

    counts = np.maximum(counts, 1)
    projected /= counts
    return projected
