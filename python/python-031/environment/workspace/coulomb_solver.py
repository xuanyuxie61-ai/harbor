# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

E_CHARGE = 1.43996448


def basis_mn_t3(t, p):
    p = np.atleast_2d(p)
    if p.shape[0] != 2:
        p = p.T
    n = p.shape[1]

    area = (t[0, 0] * (t[1, 1] - t[1, 2])
            + t[0, 1] * (t[1, 2] - t[1, 0])
            + t[0, 2] * (t[1, 0] - t[1, 1]))

    if abs(area) < 1e-15:
        raise ValueError("三角形面积为零")

    phi = np.zeros((3, n))
    dphidx = np.zeros((3, n))
    dphidy = np.zeros((3, n))

    phi[0, :] = ((t[0, 2] - t[0, 1]) * (p[1, :] - t[1, 1])
                 - (t[1, 2] - t[1, 1]) * (p[0, :] - t[0, 1]))
    dphidx[0, :] = -(t[1, 2] - t[1, 1])
    dphidy[0, :] = (t[0, 2] - t[0, 1])

    phi[1, :] = ((t[0, 0] - t[0, 2]) * (p[1, :] - t[1, 2])
                 - (t[1, 0] - t[1, 2]) * (p[0, :] - t[0, 2]))
    dphidx[1, :] = -(t[1, 0] - t[1, 2])
    dphidy[1, :] = (t[0, 0] - t[0, 2])

    phi[2, :] = ((t[0, 1] - t[0, 0]) * (p[1, :] - t[1, 0])
                 - (t[1, 1] - t[1, 0]) * (p[0, :] - t[0, 0]))
    dphidx[2, :] = -(t[1, 1] - t[1, 0])
    dphidy[2, :] = (t[0, 1] - t[0, 0])

    phi = phi / area
    dphidx = dphidx / area
    dphidy = dphidy / area

    return phi, dphidx, dphidy


def triangle_area(nodes):
    x1, y1 = nodes[0]
    x2, y2 = nodes[1]
    x3, y3 = nodes[2]
    return abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) / 2.0


def assemble_stiffness(nodes, elements, rho_p_func):
    n_nodes = nodes.shape[0]
    n_elements = elements.shape[0]

    row_ind = []
    col_ind = []
    data = []
    F = np.zeros(n_nodes)

    for elem in range(n_elements):
        idx = elements[elem]
        t = nodes[idx].T

        area = triangle_area(nodes[idx])
        if area < 1e-15:
            continue


        xq = np.mean(nodes[idx, 0])
        yq = np.mean(nodes[idx, 1])
        wq = 1.0 / 3.0

        phi, dphidx, dphidy = basis_mn_t3(t, np.array([[xq], [yq]]))
        phi = phi[:, 0]
        dphidx = dphidx[:, 0]
        dphidy = dphidy[:, 0]

        rho_val = rho_p_func(xq, yq)


        for i in range(3):
            ii = idx[i]
            F[ii] += area * wq * 4.0 * np.pi * E_CHARGE * rho_val * phi[i]
            for j in range(3):
                jj = idx[j]
                kij = area * wq * (dphidx[i] * dphidx[j] + dphidy[i] * dphidy[j])
                row_ind.append(ii)
                col_ind.append(jj)
                data.append(kij)

    K = csr_matrix((data, (row_ind, col_ind)), shape=(n_nodes, n_nodes))

    K = K + csr_matrix((1e-12 * np.ones(n_nodes), (np.arange(n_nodes), np.arange(n_nodes))),
                        shape=(n_nodes, n_nodes))
    return K, F


def apply_dirichlet_bc(K, F, bc_nodes, bc_values):
    K = K.tolil()
    for node, val in zip(bc_nodes, bc_values):
        K[node, :] = 0.0
        K[node, node] = 1.0
        F[node] = val
    return K.tocsr(), F


def solve_poisson_fem(nodes, elements, rho_p_func, bc_nodes=None, bc_values=None):
    K, F = assemble_stiffness(nodes, elements, rho_p_func)

    if bc_nodes is not None and len(bc_nodes) > 0:
        K, F = apply_dirichlet_bc(K, F, bc_nodes, bc_values)

    phi = spsolve(K, F)


    E_coulomb = 0.0
    for elem in range(elements.shape[0]):
        idx = elements[elem]
        area = triangle_area(nodes[idx])
        xq = np.mean(nodes[idx, 0])
        yq = np.mean(nodes[idx, 1])
        rho_val = rho_p_func(xq, yq)
        phi_avg = np.mean(phi[idx])
        E_coulomb += 0.5 * area * rho_val * phi_avg

    return phi, E_coulomb


def wigner_seitz_coulomb(density, proton_fraction, phase_id, u=None, n_r=50):
    from geometry_pasta import create_pasta_phase

    phase = create_pasta_phase(phase_id, density, proton_fraction, u)
    a = phase.a_WS


    theta = np.linspace(0, 2 * np.pi, n_r)
    r = np.linspace(0, a, n_r)
    R, Theta = np.meshgrid(r, theta)
    X = R.flatten() * np.cos(Theta.flatten())
    Y = R.flatten() * np.sin(Theta.flatten())
    nodes = np.column_stack((X, Y))



    n_nodes = len(nodes)

    elements = []
    for i in range(n_r - 1):
        for j in range(n_r - 1):
            n1 = i * n_r + j
            n2 = i * n_r + j + 1
            n3 = (i + 1) * n_r + j
            n4 = (i + 1) * n_r + j + 1
            elements.append([n1, n2, n3])
            elements.append([n2, n4, n3])
    elements = np.array(elements)


    rho_p_bulk = phase.rho_p
    rho_p_gas = phase.rho_p * 0.01

    if phase_id in [1, 2, 3]:
        def rho_p_func(x, y):
            r2 = x**2 + y**2
            if phase_id == 1:
                R_p = phase.R
                return rho_p_bulk if r2 <= R_p**2 else rho_p_gas
            elif phase_id == 2:
                R_p = phase.R
                return rho_p_bulk if np.sqrt(r2) <= R_p else rho_p_gas
            else:
                t = phase.t
                return rho_p_bulk if abs(x) <= t / 2 else rho_p_gas
    else:
        def rho_p_func(x, y):
            r2 = x**2 + y**2
            if phase_id == 4:
                R_p = phase.R
                return rho_p_gas if r2 <= R_p**2 else rho_p_bulk
            else:
                R_p = phase.R
                return rho_p_gas if r2 <= R_p**2 else rho_p_bulk


    boundary_mask = (R.flatten() >= a * 0.99)
    bc_nodes = np.where(boundary_mask)[0]
    bc_values = np.zeros(len(bc_nodes))

    try:
        phi, E_C = solve_poisson_fem(nodes, elements, rho_p_func, bc_nodes, bc_values)
    except Exception:

        E_C = analytical_coulomb(phase_id, density, proton_fraction, u)
        return E_C


    E_C_per_nucleon = E_C / (density * a**2)
    return E_C_per_nucleon


def analytical_coulomb(phase_id, density, proton_fraction, u=None):
    from geometry_pasta import create_pasta_phase
    phase = create_pasta_phase(phase_id, density, proton_fraction, u)
    R_WS = (3.0 / (4.0 * np.pi * density)) ** (1.0 / 3.0)
    f_C = phase.coulomb_factor()

    g_c = 5.0
    e_coul = (3.0 / 10.0) * E_CHARGE / R_WS * (proton_fraction)**2 * f_C * g_c
    return e_coul


if __name__ == '__main__':
    rho = 0.08
    x_p = 0.3
    for pid in [1, 2, 3]:
        e_c = analytical_coulomb(pid, rho, x_p)
        print(f"Phase {pid} analytical Coulomb energy: {e_c:.4f} MeV/nucleon")
