#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from parameter_manager import PhysicalConstants
from mesh_generator import StructuredMesh2D
from sparse_matrix_tools import DenseLUSolver


def mt_1d_analytic(resistivities, thicknesses, frequencies):
    mu0 = PhysicalConstants.MU_0
    n_layers = len(resistivities)
    n_freq = len(frequencies)
    Z_xy = np.zeros(n_freq, dtype=np.complex128)










    raise NotImplementedError("Hole 1: 一维层状介质MT正演递推公式待实现")


def mt_1d_analytic_cole_cole(resistivities, thicknesses, dispersion_list, frequencies):
    mu0 = PhysicalConstants.MU_0
    n_layers = len(resistivities)
    n_freq = len(frequencies)
    Z_xy = np.zeros(n_freq, dtype=np.complex128)

    for ifreq, f in enumerate(frequencies):
        omega = 2.0 * np.pi * f


        if dispersion_list[-1] is not None:
            sigma_star = dispersion_list[-1].complex_conductivity(omega)
        else:
            sigma_star = 1.0 / resistivities[-1]
        k_N = np.sqrt(1j * omega * mu0 * sigma_star)
        Z_top = 1j * omega * mu0 / k_N

        for ilayer in range(n_layers - 2, -1, -1):
            if dispersion_list[ilayer] is not None:
                sigma_star = dispersion_list[ilayer].complex_conductivity(omega)
            else:
                sigma_star = 1.0 / resistivities[ilayer]
            k = np.sqrt(1j * omega * mu0 * sigma_star)
            h = thicknesses[ilayer]
            ratio = k * Z_top / (1j * omega * mu0)
            inv_ratio = 1.0 / ratio
            if np.abs(inv_ratio) >= 0.999:
                inv_ratio = 0.999 * inv_ratio / np.abs(inv_ratio)
            arg = k * h + np.arctanh(inv_ratio)

            if np.abs(np.sinh(arg)) < 1e-15:
                arg = arg + 1e-15j
            Z_top = (1j * omega * mu0 / k) / np.tanh(arg)

        Z_xy[ifreq] = Z_top

    omega_all = 2.0 * np.pi * frequencies
    rho_a = np.abs(Z_xy) ** 2 / (omega_all * mu0)
    phi = np.angle(Z_xy, deg=True)
    return Z_xy, rho_a, phi


def mt_2d_te_fd(conductivity_map, mesh, frequency, boundary_value_func):
    mu0 = PhysicalConstants.MU_0
    omega = 2.0 * np.pi * frequency

    n_nodes = mesh.n_nodes


    if callable(conductivity_map):
        sigma_nodes = np.zeros(n_nodes, dtype=np.complex128)
        for idx in range(n_nodes):
            y, z = mesh.node_coords[idx]
            sigma_nodes[idx] = conductivity_map(y, z)
    else:
        sigma_nodes = np.asarray(conductivity_map, dtype=np.complex128)
        if len(sigma_nodes) != n_nodes:
            raise ValueError("电导率数组长度与节点数不匹配")




    dx2 = mesh.dx ** 2
    dy2 = mesh.dy ** 2



    n_int = len(mesh.interior_nodes)
    n_bnd = len(mesh.boundary_nodes)


    A = np.zeros((n_nodes, n_nodes), dtype=np.complex128)
    rhs = np.zeros(n_nodes, dtype=np.complex128)


    for idx in mesh.interior_nodes:
        i, j = mesh.inv_map[idx]
        neighbors = mesh.get_neighbors(idx)

        coeff = 0.0
        for nidx, direction in neighbors:
            if direction in ('E', 'W'):
                A[idx, nidx] += 1.0 / dx2
                coeff -= 1.0 / dx2
            else:
                A[idx, nidx] += 1.0 / dy2
                coeff -= 1.0 / dy2

        k2 = 1j * omega * mu0 * sigma_nodes[idx]
        A[idx, idx] = coeff + k2


    for idx in mesh.boundary_nodes:
        y, z = mesh.node_coords[idx]
        A[idx, idx] = 1.0
        rhs[idx] = boundary_value_func(y, z)


    solver = DenseLUSolver(A)
    info = solver.dgefa()
    if info != 0:

        E_x = np.linalg.lstsq(A, rhs, rcond=None)[0]
    else:
        E_x = solver.solve(rhs)



    H_y = np.zeros(n_nodes, dtype=np.complex128)
    for idx in range(n_nodes):
        i, j = mesh.inv_map[idx]
        nidx_s = mesh.get_node_index(i, j - 1)
        nidx_n = mesh.get_node_index(i, j + 1)

        if nidx_s >= 0 and nidx_n >= 0:
            dEdz = (E_x[nidx_n] - E_x[nidx_s]) / (2.0 * mesh.dy)
        elif nidx_n >= 0:
            dEdz = (E_x[nidx_n] - E_x[idx]) / mesh.dy
        elif nidx_s >= 0:
            dEdz = (E_x[idx] - E_x[nidx_s]) / mesh.dy
        else:
            dEdz = 0.0

        H_y[idx] = dEdz / (1j * omega * mu0)


    Z_xy = np.zeros(n_nodes, dtype=np.complex128)
    for idx in range(n_nodes):
        if abs(H_y[idx]) > 1e-20:
            Z_xy[idx] = E_x[idx] / H_y[idx]
        else:
            Z_xy[idx] = 0.0

    return E_x, H_y, Z_xy


def compute_apparent_resistivity_phase(Z, frequencies):
    mu0 = PhysicalConstants.MU_0
    omega = 2.0 * np.pi * frequencies
    rho_a = np.abs(Z) ** 2 / (omega * mu0)
    phi = np.angle(Z, deg=True)
    return rho_a, phi


def add_noise_to_mt_data(rho_a, phi, noise_level=0.05):
    rho_a = np.asarray(rho_a, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)

    rho_noise = rho_a * noise_level * np.random.randn(len(rho_a))

    phi_noise = np.random.randn(len(phi)) * noise_level * 5.0

    rho_a_noisy = np.maximum(rho_a + rho_noise, 0.1)
    phi_noisy = phi + phi_noise

    phi_noisy = np.clip(phi_noisy, -90.0, 90.0)

    return rho_a_noisy, phi_noisy


def thin_field_data(coords, field_values, thin_factor=2):
    coords = np.asarray(coords)
    field_values = np.asarray(field_values)
    n = len(coords)

    x_unique = np.unique(coords[:, 0])
    y_unique = np.unique(coords[:, 1])


    kept = []
    for i in range(n):
        xi = coords[i, 0]
        yi = coords[i, 1]
        ix = np.searchsorted(x_unique, xi)
        iy = np.searchsorted(y_unique, yi)
        if iy % thin_factor == thin_factor // 2 and ix % thin_factor == thin_factor // 2:
            kept.append(i)

    return coords[kept], field_values[kept]


if __name__ == "__main__":

    resistivities = np.array([100.0, 50.0, 10.0])
    thicknesses = np.array([500.0, 1000.0])
    frequencies = np.logspace(-2, 2, 20)
    Z, rho_a, phi = mt_1d_analytic(resistivities, thicknesses, frequencies)
    print("1D 解析正演结果 (前5个频率):")
    for i in range(min(5, len(frequencies))):
        print(f"  f={frequencies[i]:.4f} Hz, Z={Z[i]:.4e}, "
              f"ρ_a={rho_a[i]:.2f} Ω·m, φ={phi[i]:.2f}°")


    from mesh_generator import generate_rectangular_mesh
    mesh = generate_rectangular_mesh(0.0, 10000.0, 0.0, 5000.0, 21, 11)

    def sigma_map(y, z):
        if z < 500.0:
            return 0.01
        elif z < 2000.0:
            return 0.02
        else:
            return 0.1

    def bc_func(y, z):

        if abs(z) < 1.0:
            return 1.0

        return np.exp(-z / 1000.0)

    E, H, Z2d = mt_2d_te_fd(sigma_map, mesh, 10.0, bc_func)
    print(f"\n2D 正演: {len(E)} 节点, E_x 范围: [{np.min(np.abs(E)):.4e}, {np.max(np.abs(E)):.4e}]")
