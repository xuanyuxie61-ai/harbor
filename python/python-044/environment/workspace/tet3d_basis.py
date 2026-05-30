
import numpy as np


def tetrahedron_volume(tet_xyz):
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
    phi = np.zeros((4, n_eval))
    volume = tetrahedron_volume(tet_xyz)

    if abs(volume) < 1e-14:
        raise ValueError("Tetrahedron has zero volume.")


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
    volume = tetrahedron_volume(tet_xyz)
    if abs(volume) < 1e-14:
        raise ValueError("Tetrahedron has zero volume.")



    grad_phi = np.zeros((3, 4))

    for i in range(4):

        face = [j for j in range(4) if j != i]
        v0 = tet_xyz[:, face[0]]
        v1 = tet_xyz[:, face[1]]
        v2 = tet_xyz[:, face[2]]


        e1 = v1 - v0
        e2 = v2 - v0
        normal = np.cross(e1, e2)


        grad_phi[:, i] = normal / volume

    return grad_phi


def project_to_tet_mesh(sample_nodes, sample_values, tet_nodes, tet_elements):
    n_tet_nodes = tet_nodes.shape[0]
    projected = np.zeros(n_tet_nodes)
    counts = np.zeros(n_tet_nodes)

    n_tets = tet_elements.shape[0]
    for e in range(n_tets):
        elem_nodes = tet_elements[e, :]
        for nid in elem_nodes:

            projected[nid] += np.mean(sample_values)
            counts[nid] += 1

    counts = np.maximum(counts, 1)
    projected /= counts
    return projected
