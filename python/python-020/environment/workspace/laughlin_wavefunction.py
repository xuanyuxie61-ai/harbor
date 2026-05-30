# -*- coding: utf-8 -*-
import numpy as np
from utils import magnetic_length, safe_exp, safe_log





def laughlin_wavefunction(z, m, lB, return_log=False):
    z = np.asarray(z, dtype=complex)
    N = len(z)
    if N < 2:
        raise ValueError("Laughlin波函数至少需要2个电子")
    if m % 2 == 0:
        raise ValueError("Laughlin指数 m 必须为奇整数")
    if m < 1:
        raise ValueError("Laughlin指数 m 必须 ≥ 1")
    if lB <= 0:
        raise ValueError("磁长度 lB 必须为正")


    jastrow_log = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dz = z[i] - z[j]

            abs_dz = abs(dz)
            if abs_dz < 1e-15:
                abs_dz = 1e-15
            jastrow_log += m * (np.log(abs_dz) + 1j * np.angle(dz))


    rho_sum = np.sum(np.abs(z) ** 2) / (4.0 * lB * lB)

    log_psi = jastrow_log - rho_sum

    if return_log:
        return log_psi


    psi = np.exp(log_psi.real) * (np.cos(log_psi.imag) + 1j * np.sin(log_psi.imag))
    return psi


def laughlin_log_probability(z, m, lB):
    z = np.asarray(z, dtype=complex)
    N = len(z)
    log_prob = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            abs_dz = abs(z[i] - z[j])
            if abs_dz < 1e-15:
                abs_dz = 1e-15
            log_prob += 2.0 * m * np.log(abs_dz)
    log_prob -= np.sum(np.abs(z) ** 2) / (2.0 * lB * lB)
    return log_prob






def quasihole_wavefunction(z, z0, m, lB, return_log=False):
    z = np.asarray(z, dtype=complex)
    N = len(z)
    if N < 1:
        raise ValueError("至少需要1个电子坐标")


    qh_log = 0.0
    for j in range(N):
        dz = z[j] - z0
        abs_dz = abs(dz)
        if abs_dz < 1e-15:
            abs_dz = 1e-15
        qh_log += np.log(abs_dz) + 1j * np.angle(dz)

    base_log = laughlin_wavefunction(z, m, lB, return_log=True)
    log_psi = base_log + qh_log

    if return_log:
        return log_psi
    return np.exp(log_psi.real) * (np.cos(log_psi.imag) + 1j * np.sin(log_psi.imag))


def quasielectron_wavefunction(z, z0, m, lB, return_log=False):
    z = np.asarray(z, dtype=complex)
    N = len(z)
    if N < 1:
        raise ValueError("至少需要1个电子坐标")


    qe_log = 0.0
    for j in range(N):
        dz = np.conj(z[j]) - np.conj(z0)
        abs_dz = abs(dz)
        if abs_dz < 1e-15:
            abs_dz = 1e-15
        qe_log += np.log(abs_dz) + 1j * np.angle(dz)

    base_log = laughlin_wavefunction(z, m, lB, return_log=True)
    log_psi = base_log + qe_log

    if return_log:
        return log_psi
    return np.exp(log_psi.real) * (np.cos(log_psi.imag) + 1j * np.sin(log_psi.imag))






def pair_correlation_function(z, m, lB, r_bins=80, r_max=None):
    z = np.asarray(z, dtype=complex)
    N = len(z)
    if N < 2:
        raise ValueError("至少需要2个电子")


    distances = []
    for i in range(N):
        for j in range(i + 1, N):
            d = abs(z[i] - z[j])
            distances.append(d)
    distances = np.array(distances)

    if r_max is None:
        r_max = np.max(distances) * 1.2
    if r_max <= 0:
        r_max = 1.0


    g_r, r_edges = np.histogram(distances, bins=r_bins, range=(0.0, r_max))



    R_system = np.max(np.abs(z)) * 1.1
    area = np.pi * R_system ** 2
    n_density = N / area

    bin_widths = np.diff(r_edges)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])




    for i in range(len(g_r)):
        r_c = r_centers[i]
        dr = bin_widths[i]
        shell_area = 2.0 * np.pi * r_c * dr
        if shell_area < 1e-15:
            g_r[i] = 0.0
            continue

        norm = (N * (N - 1) / 2.0) * shell_area / area
        if norm < 1e-15:
            g_r[i] = 0.0
        else:
            g_r[i] = g_r[i] / norm

    return r_edges, g_r, r_centers


def structure_factor_s_q(z, m, lB, q_bins=60, q_max=None):
    z = np.asarray(z, dtype=complex)
    N = len(z)
    if N < 2:
        raise ValueError("至少需要2个电子")

    x = z.real
    y = z.imag

    if q_max is None:
        q_max = 10.0 / lB

    q_vals = np.linspace(0.01, q_max, q_bins)
    S_q = np.zeros_like(q_vals)

    for idx, q in enumerate(q_vals):

        n_angles = 36
        angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
        sq_ang = 0.0
        for theta in angles:
            qx = q * np.cos(theta)
            qy = q * np.sin(theta)
            phase = np.exp(-1j * (qx * x + qy * y))
            rho_q = np.sum(phase)
            sq_ang += np.abs(rho_q) ** 2
        S_q[idx] = sq_ang / (n_angles * N)

    return q_vals, S_q






def wavefunction_overlap(z_grid, psi1, psi2, dx, dy):
    psi1 = np.asarray(psi1, dtype=complex)
    psi2 = np.asarray(psi2, dtype=complex)
    if psi1.shape != psi2.shape:
        raise ValueError("两个波函数数组形状必须一致")
    return np.sum(np.conj(psi1) * psi2) * dx * dy





def test_laughlin_wavefunction():
    print("=" * 60)
    print("[laughlin_wavefunction.py] Laughlin波函数测试")
    print("=" * 60)

    B = 10.0
    m_star = 1.0
    lB = magnetic_length(B, m_star)
    m = 3


    N = 8
    np.random.seed(42)
    theta = np.random.uniform(0.0, 2.0 * np.pi, N)
    r = np.sqrt(np.random.uniform(0.0, 1.0, N)) * np.sqrt(2.0 * m * N) * lB * 0.5
    z = r * np.exp(1j * theta)

    print(f"\n物理参数:")
    print(f"  磁场 B = {B} T")
    print(f"  磁长度 l_B = {lB:.6f}")
    print(f"  Laughlin指数 m = {m} (填充因子 ν = 1/{m})")
    print(f"  电子数 N = {N}")


    log_psi = laughlin_wavefunction(z, m, lB, return_log=True)
    print(f"\nLaughlin波函数对数值: Re(logΨ) = {log_psi.real:.4f}, Im(logΨ) = {log_psi.imag:.4f}")


    r_edges, g_r, r_centers = pair_correlation_function(z, m, lB, r_bins=40)
    print(f"\n配对关联函数 g(r) 前5个值:")
    for i in range(min(5, len(r_centers))):
        print(f"  r = {r_centers[i]:.4f}, g(r) = {g_r[i]:.6f}")



    if len(g_r) > 2 and r_centers[1] > 0:
        slope_approx = np.log(g_r[2] + 1e-10) / np.log(r_centers[2] + 1e-10)
        print(f"  g(r) 小r近似幂次: {slope_approx:.2f} (理论值 ~ {2*m})")


    z0 = 0.5 * lB + 0.3j * lB
    log_psi_qh = quasihole_wavefunction(z, z0, m, lB, return_log=True)
    print(f"\n准空穴波函数 (z0={z0:.3f}):")
    print(f"  Re(logΨ_qh) = {log_psi_qh.real:.4f}")


    q_vals, S_q = structure_factor_s_q(z, m, lB, q_bins=30)
    print(f"\n结构因子 S(q) 前3个值:")
    for i in range(min(3, len(q_vals))):
        print(f"  q = {q_vals[i]:.4f}, S(q) = {S_q[i]:.6f}")

    print("\n[laughlin_wavefunction.py] 测试完成。\n")


if __name__ == "__main__":
    test_laughlin_wavefunction()
