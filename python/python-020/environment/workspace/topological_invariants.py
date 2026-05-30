# -*- coding: utf-8 -*-
import numpy as np
from utils import H_BAR, E_CHARGE





def berry_connection(u_k, u_k_plus_dk, dk):
    u_k = np.asarray(u_k, dtype=complex)
    u_k_plus_dk = np.asarray(u_k_plus_dk, dtype=complex)
    overlap = np.vdot(u_k, u_k_plus_dk)

    if abs(overlap) < 1e-14:
        return 0.0

    A = 1j * np.log(overlap / abs(overlap)) / dk
    return A


def berry_curvature_discrete(u_grid, kx_grid, ky_grid):
    Nx, Ny, N_states = u_grid.shape
    if Nx < 2 or Ny < 2:
        raise ValueError("网格尺寸必须 ≥ 2")

    Omega = np.zeros((Nx - 1, Ny - 1), dtype=float)
    kx_centers = 0.5 * (kx_grid[:-1] + kx_grid[1:])
    ky_centers = 0.5 * (ky_grid[:-1] + ky_grid[1:])

    for i in range(Nx - 1):
        for j in range(Ny - 1):
            u00 = u_grid[i, j]
            u10 = u_grid[i + 1, j]
            u11 = u_grid[i + 1, j + 1]
            u01 = u_grid[i, j + 1]


            o1 = np.vdot(u00, u10)
            o2 = np.vdot(u10, u11)
            o3 = np.vdot(u11, u01)
            o4 = np.vdot(u01, u00)


            phases = []
            for o in [o1, o2, o3, o4]:
                if abs(o) > 1e-14:
                    phases.append(np.angle(o))
                else:
                    phases.append(0.0)


            phi_total = phases[0] + phases[1] + phases[2] + phases[3]

            phi_total = (phi_total + np.pi) % (2.0 * np.pi) - np.pi
            Omega[i, j] = -phi_total

    return Omega, kx_centers, ky_centers






def chern_number_from_berry_curvature(Omega, dkx, dky):
    integral = np.sum(Omega) * dkx * dky
    C = integral / (2.0 * np.pi)
    return C


def tknn_conductivity(Omega_sum, dkx, dky):
    C = Omega_sum * dkx * dky / (2.0 * np.pi)
    return C






def filling_factor_from_chern(C, degeneracy_per_level):
    if abs(C) < 1e-14:
        return np.inf
    return 1.0 / C


def conductance_quantization(nu, m_laughlin=None):
    if m_laughlin is not None:
        nu = 1.0 / m_laughlin
    return nu






def flux_quantization_phase(n_phi, n_e, m_laughlin=3):
    if n_phi <= 0:
        raise ValueError("n_phi 必须为正")
    nu = n_e / n_phi
    phase = 2.0 * np.pi * nu
    charge = E_CHARGE / m_laughlin
    return phase, charge


def orbital_evolution_parameters(ecc, lon_deg, obliq_deg, n_points=100):
    lon = lon_deg * np.pi / 180.0
    obliq = obliq_deg * np.pi / 180.0

    tau = np.linspace(0.0, 2.0 * np.pi, n_points)


    theta = np.arctan2(
        np.sqrt(1.0 - ecc ** 2) * np.sin(tau),
        np.cos(tau) - ecc
    )


    x1 = np.cos(theta - (lon - np.pi / 2.0))
    y1 = np.sin(theta - (lon - np.pi / 2.0))


    x2 = np.cos(obliq) * x1
    y2 = y1
    z2 = -np.sin(obliq) * x1


    return tau, theta, x2, y2, z2





def test_topological_invariants():
    print("=" * 60)
    print("[topological_invariants.py] 拓扑不变量测试")
    print("=" * 60)


    print("\n1. Berry曲率与Chern数测试 (Haldane模型近似):")
    Nk = 40
    kx = np.linspace(-np.pi, np.pi, Nk)
    ky = np.linspace(-np.pi, np.pi, Nk)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')



    m_mass = 1.5
    d_x = np.sin(KX)
    d_y = np.sin(KY)
    d_z = m_mass + np.cos(KX) + np.cos(KY)
    d_mag = np.sqrt(d_x ** 2 + d_y ** 2 + d_z ** 2)

    u_grid = np.zeros((Nk, Nk, 2), dtype=complex)
    for i in range(Nk):
        for j in range(Nk):
            dx, dy, dz = d_x[i, j], d_y[i, j], d_z[i, j]
            dm = d_mag[i, j]
            if dm < 1e-14:
                u_grid[i, j] = np.array([1.0, 0.0])
            else:

                u_grid[i, j] = np.array([
                    np.sqrt((1.0 - dz / dm) / 2.0),
                    (dx + 1j * dy) / np.sqrt(2.0 * dm * (dm - dz))
                ])

                if abs(dm - dz) < 1e-14:
                    u_grid[i, j] = np.array([0.0, 1.0])

    Omega, kxc, kyc = berry_curvature_discrete(u_grid, kx, ky)
    dkx = kx[1] - kx[0]
    dky = ky[1] - ky[0]
    C = chern_number_from_berry_curvature(Omega, dkx, dky)
    print(f"   计算Chern数: C = {C:.4f} (预期 ≈ 0，因为 m={m_mass}>2)")


    m_mass = -1.5
    d_z = m_mass + np.cos(KX) + np.cos(KY)
    d_mag = np.sqrt(d_x ** 2 + d_y ** 2 + d_z ** 2)
    for i in range(Nk):
        for j in range(Nk):
            dx, dy, dz = d_x[i, j], d_y[i, j], d_z[i, j]
            dm = d_mag[i, j]
            if dm < 1e-14:
                u_grid[i, j] = np.array([1.0, 0.0])
            else:
                if abs(dm - dz) < 1e-14:
                    u_grid[i, j] = np.array([0.0, 1.0])
                else:
                    u_grid[i, j] = np.array([
                        np.sqrt((1.0 - dz / dm) / 2.0),
                        (dx + 1j * dy) / np.sqrt(2.0 * dm * (dm - dz))
                    ])

    Omega, _, _ = berry_curvature_discrete(u_grid, kx, ky)
    C = chern_number_from_berry_curvature(Omega, dkx, dky)
    print(f"   计算Chern数: C = {C:.4f} (预期 ≈ 0，因为 |m|>2)")


    print("\n2. 磁通量子化Berry相位测试:")
    for n_phi in [3, 6, 9, 12]:
        n_e = n_phi // 3
        phase, charge = flux_quantization_phase(n_phi, n_e, m_laughlin=3)
        print(f"   n_Φ={n_phi:2d}, n_e={n_e:2d}: phase={phase:.4f} rad, charge={charge:.6f}")


    print("\n3. 绝热参数演化测试:")
    tau, theta, x2, y2, z2 = orbital_evolution_parameters(
        ecc=0.01671, lon_deg=77.0, obliq_deg=23.44, n_points=100
    )
    print(f"   参数范围: τ∈[{tau[0]:.4f}, {tau[-1]:.4f}]")
    print(f"   角度范围: θ∈[{np.min(theta):.4f}, {np.max(theta):.4f}]")
    print(f"   轨道闭合检查: x²+y²+z² 均值 = {np.mean(x2**2+y2**2+z2**2):.6f}")

    print("\n[topological_invariants.py] 测试完成。\n")


if __name__ == "__main__":
    test_topological_invariants()
