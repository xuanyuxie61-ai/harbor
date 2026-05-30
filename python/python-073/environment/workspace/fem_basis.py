# -*- coding: utf-8 -*-

import numpy as np


def tet4_basis(t, p):
    if p.ndim == 1:
        p = p.reshape(3, 1)
        squeeze = True
    else:
        squeeze = False
    n = p.shape[1]


    volume = (
        t[0, 0] * (t[1, 1] * (t[2, 2] - t[2, 3]) - t[1, 2] * (t[2, 1] - t[2, 3]) + t[1, 3] * (t[2, 1] - t[2, 2]))
        - t[0, 1] * (t[1, 0] * (t[2, 2] - t[2, 3]) - t[1, 2] * (t[2, 0] - t[2, 3]) + t[1, 3] * (t[2, 0] - t[2, 2]))
        + t[0, 2] * (t[1, 0] * (t[2, 1] - t[2, 3]) - t[1, 1] * (t[2, 0] - t[2, 3]) + t[1, 3] * (t[2, 0] - t[2, 1]))
        - t[0, 3] * (t[1, 0] * (t[2, 1] - t[2, 2]) - t[1, 1] * (t[2, 0] - t[2, 2]) + t[1, 2] * (t[2, 0] - t[2, 1]))
    )

    if abs(volume) < 1e-15:
        raise ValueError("fem_basis: 四面体体积为零，网格退化")

    phi = np.zeros((4, n))

    phi[0, :] = (
        p[0, :] * (t[1, 1] * (t[2, 2] - t[2, 3]) - t[1, 2] * (t[2, 1] - t[2, 3]) + t[1, 3] * (t[2, 1] - t[2, 2]))
        - t[0, 1] * (p[1, :] * (t[2, 2] - t[2, 3]) - t[1, 2] * (p[2, :] - t[2, 3]) + t[1, 3] * (p[2, :] - t[2, 2]))
        + t[0, 2] * (p[1, :] * (t[2, 1] - t[2, 3]) - t[1, 1] * (p[2, :] - t[2, 3]) + t[1, 3] * (p[2, :] - t[2, 1]))
        - t[0, 3] * (p[1, :] * (t[2, 1] - t[2, 2]) - t[1, 1] * (p[2, :] - t[2, 2]) + t[1, 2] * (p[2, :] - t[2, 1]))
    ) / volume


    phi[1, :] = (
        t[0, 0] * (p[1, :] * (t[2, 2] - t[2, 3]) - t[1, 2] * (p[2, :] - t[2, 3]) + t[1, 3] * (p[2, :] - t[2, 2]))
        - p[0, :] * (t[1, 0] * (t[2, 2] - t[2, 3]) - t[1, 2] * (t[2, 0] - t[2, 3]) + t[1, 3] * (t[2, 0] - t[2, 2]))
        + t[0, 2] * (t[1, 0] * (p[2, :] - t[2, 3]) - p[1, :] * (t[2, 0] - t[2, 3]) + t[1, 3] * (t[2, 0] - p[2, :]))
        - t[0, 3] * (t[1, 0] * (p[2, :] - t[2, 2]) - p[1, :] * (t[2, 0] - t[2, 2]) + t[1, 2] * (t[2, 0] - p[2, :]))
    ) / volume


    phi[2, :] = (
        t[0, 0] * (t[1, 1] * (p[2, :] - t[2, 3]) - p[1, :] * (t[2, 1] - t[2, 3]) + t[1, 3] * (t[2, 1] - p[2, :]))
        - t[0, 1] * (t[1, 0] * (p[2, :] - t[2, 3]) - p[1, :] * (t[2, 0] - t[2, 3]) + t[1, 3] * (t[2, 0] - p[2, :]))
        + p[0, :] * (t[1, 0] * (t[2, 1] - t[2, 3]) - t[1, 1] * (t[2, 0] - t[2, 3]) + t[1, 3] * (t[2, 0] - t[2, 1]))
        - t[0, 3] * (t[1, 0] * (t[2, 1] - p[2, :]) - t[1, 1] * (t[2, 0] - p[2, :]) + p[1, :] * (t[2, 0] - t[2, 1]))
    ) / volume


    phi[3, :] = (
        t[0, 0] * (t[1, 1] * (t[2, 2] - p[2, :]) - t[1, 2] * (t[2, 1] - p[2, :]) + p[1, :] * (t[2, 1] - t[2, 2]))
        - t[0, 1] * (t[1, 0] * (t[2, 2] - p[2, :]) - t[1, 2] * (t[2, 0] - p[2, :]) + p[1, :] * (t[2, 0] - t[2, 2]))
        + t[0, 2] * (t[1, 0] * (t[2, 1] - p[2, :]) - t[1, 1] * (t[2, 0] - p[2, :]) + p[1, :] * (t[2, 0] - t[2, 1]))
        - p[0, :] * (t[1, 0] * (t[2, 1] - t[2, 2]) - t[1, 1] * (t[2, 0] - t[2, 2]) + t[1, 2] * (t[2, 0] - t[2, 1]))
    ) / volume

    if squeeze:
        return phi[:, 0]
    return phi


def tetrahedron_volume(t):
    A = np.column_stack((t[:, 1] - t[:, 0], t[:, 2] - t[:, 0], t[:, 3] - t[:, 0]))
    vol = abs(np.linalg.det(A)) / 6.0
    return vol


def reference_to_physical_tet4(t, xi):
    if xi.ndim == 1:
        xi = xi.reshape(3, 1)
        squeeze = True
    else:
        squeeze = False

    n = xi.shape[1]
    phi = np.zeros((4, n))
    phi[0, :] = 1.0 - xi[0, :] - xi[1, :] - xi[2, :]
    phi[1, :] = xi[0, :]
    phi[2, :] = xi[1, :]
    phi[3, :] = xi[2, :]

    x_phys = t @ phi
    if squeeze:
        return x_phys[:, 0]
    return x_phys


def physical_to_reference_tet4(t, x):
    if x.ndim == 1:
        x = x.reshape(3, 1)
        squeeze = True
    else:
        squeeze = False

    J = np.column_stack((t[:, 1] - t[:, 0], t[:, 2] - t[:, 0], t[:, 3] - t[:, 0]))
    detJ = np.linalg.det(J)
    if abs(detJ) < 1e-14:
        raise ValueError("physical_to_reference_tet4: Jacobian 行列式接近零")

    rhs = x - t[:, 0][:, None]
    xi = np.linalg.solve(J, rhs)

    if squeeze:
        return xi[:, 0]
    return xi


def reference_tet4_sample(n):
    samples = np.random.rand(3, n)
    s1 = np.sort(samples, axis=0)
    xi = np.zeros((3, n))
    xi[0, :] = s1[0, :]
    xi[1, :] = s1[1, :] - s1[0, :]
    xi[2, :] = s1[2, :] - s1[1, :]
    return xi


def build_fem_mass_matrix(nodes, tetrahedra, rho=None):
    n_node = nodes.shape[0]
    n_elem = tetrahedra.shape[0]
    M = np.zeros((n_node, n_node))

    if rho is None:
        rho = 1.0
    if np.isscalar(rho):
        rho_vals = np.full(n_elem, rho)
    else:
        rho_vals = np.asarray(rho)

    for e in range(n_elem):
        idx = tetrahedra[e]
        t = nodes[idx].T
        vol = tetrahedron_volume(t)
        if vol < 1e-15:
            continue
        fac = rho_vals[e] * vol / 20.0
        for i in range(4):
            for j in range(4):
                ii = idx[i]
                jj = idx[j]
                add = fac * (2.0 if i == j else 1.0)
                M[ii, jj] += add
    return M


def build_fem_stiffness_matrix(nodes, tetrahedra, mu_field=None):
    n_node = nodes.shape[0]
    n_elem = tetrahedra.shape[0]
    K = np.zeros((n_node, n_node))

    if mu_field is None:
        mu_field = 1.0
    if np.isscalar(mu_field):
        mu_vals = np.full(n_elem, mu_field)
    else:
        mu_vals = np.asarray(mu_field)

    for e in range(n_elem):
        idx = tetrahedra[e]
        t = nodes[idx].T
        vol = tetrahedron_volume(t)
        if vol < 1e-15:
            continue


        J = np.column_stack((t[:, 1] - t[:, 0], t[:, 2] - t[:, 0], t[:, 3] - t[:, 0]))
        try:
            invJT = np.linalg.inv(J.T)
        except np.linalg.LinAlgError:
            continue


        grad_ref = np.array([[-1.0, -1.0, -1.0],
                             [ 1.0,  0.0,  0.0],
                             [ 0.0,  1.0,  0.0],
                             [ 0.0,  0.0,  1.0]])
        grad_phys = grad_ref @ invJT

        fac = mu_vals[e] * vol
        for i in range(4):
            for j in range(4):
                ii = idx[i]
                jj = idx[j]
                K[ii, jj] += fac * np.dot(grad_phys[i], grad_phys[j])
    return K
