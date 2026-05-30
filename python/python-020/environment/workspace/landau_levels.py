# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import genlaguerre, factorial
from utils import magnetic_length, cyclotron_frequency, landau_level_energy, H_BAR, E_CHARGE, gram_schmidt_qr





def landau_orbital_wavefunction(n, m, z, lB):
    if n < 0:
        raise ValueError("Landau能级指标 n 必须 ≥ 0")
    if m < -n:
        raise ValueError(f"角动量量子数 m 必须 ≥ -n = {-n}")
    if lB <= 0:
        raise ValueError("磁长度 lB 必须为正")



















    raise NotImplementedError("Landau 轨道波函数计算待实现")







def spectral_basis_1d(x, i, domain=(-1.0, 1.0)):
    a, b = domain
    x = np.asarray(x, dtype=float)
    if i < 1:
        raise ValueError("基函数指标 i 必须 ≥ 1")
    if not (a < b):
        raise ValueError("定义域必须满足 a < b")

    phi = (x ** (i - 1)) * (x - a) * (x - b)
    return phi


def spectral_basis_derivative_1d(x, i, domain=(-1.0, 1.0)):
    a, b = domain
    x = np.asarray(x, dtype=float)
    if i < 1:
        raise ValueError("基函数指标 i 必须 ≥ 1")

    if i == 1:
        dphi = (x - b) + (x - a)
    else:
        term1 = (i - 1) * (x ** (i - 2)) * (x - a) * (x - b)
        term2 = (x ** (i - 1)) * ((x - b) + (x - a))
        dphi = term1 + term2
    return dphi


def build_spectral_stiffness_matrix(N, domain=(-1.0, 1.0), nquad=100):
    a, b = domain

    from numpy.polynomial.legendre import leggauss
    xi, wi = leggauss(nquad)

    xq = 0.5 * (b - a) * xi + 0.5 * (b + a)
    wq = 0.5 * (b - a) * wi

    K = np.zeros((N, N), dtype=float)
    for k in range(1, N + 1):
        dphi_k = spectral_basis_derivative_1d(xq, k, domain)
        for i in range(1, N + 1):
            dphi_i = spectral_basis_derivative_1d(xq, i, domain)
            K[k - 1, i - 1] = np.sum(wq * dphi_k * dphi_i)
    return K






def local_basis_1d_lagrange(order, node_x, x):
    node_x = np.asarray(node_x, dtype=float)
    if len(node_x) != order:
        raise ValueError("node_x 长度必须与 order 一致")

    if len(np.unique(node_x)) != order:
        raise ValueError("Lagrange插值节点必须互异")

    x = np.atleast_1d(x)
    phi = np.ones((len(x), order), dtype=float)
    for j in range(order):
        for k in range(order):
            if k != j:
                denom = node_x[j] - node_x[k]
                if abs(denom) < 1e-14:
                    raise ValueError("插值节点过于接近，分母为零")
                phi[:, j] *= (x - node_x[k]) / denom
    if len(x) == 1:
        return phi[0, :]
    return phi


def local_fem_1d(order, node_x, node_v, sample_x):
    node_x = np.asarray(node_x, dtype=float)
    node_v = np.asarray(node_v, dtype=float)
    sample_x = np.atleast_1d(sample_x)
    phi = local_basis_1d_lagrange(order, node_x, sample_x)
    sample_v = phi @ node_v
    return sample_v






def landau_degeneracy(B, A, m_star=1.0):
    if B <= 0 or A <= 0:
        raise ValueError("B 和 A 必须为正")
    flux_quantum = 2.0 * np.pi * H_BAR / E_CHARGE
    return B * A / flux_quantum


def density_of_states_landau(E, B, m_star=1.0, gamma=0.01):
    E = np.asarray(E, dtype=float)
    omega_c = cyclotron_frequency(B, m_star)
    prefactor = E_CHARGE * B / (2.0 * np.pi * H_BAR)
    dos = np.zeros_like(E)
    n_max = int(np.max(E) / (H_BAR * omega_c) + 10)
    n_max = max(n_max, 20)
    for n in range(n_max):
        En = landau_level_energy(n, B, m_star)
        dos += (gamma / np.pi) / ((E - En) ** 2 + gamma ** 2)
    dos *= prefactor
    return dos





def test_landau_levels():
    print("=" * 60)
    print("[landau_levels.py] Landau能级与单粒子基函数测试")
    print("=" * 60)

    B = 10.0
    m_star = 1.0
    lB = magnetic_length(B, m_star)
    omega_c = cyclotron_frequency(B, m_star)

    print(f"\n物理参数:")
    print(f"  磁场 B = {B} T")
    print(f"  有效质量 m* = {m_star}")
    print(f"  磁长度 l_B = {lB:.6f}")
    print(f"  回旋频率 ω_c = {omega_c:.6f}")


    print(f"\n前5个Landau能级能量 E_n = ħω_c(n + 1/2):")
    for n in range(5):
        En = landau_level_energy(n, B, m_star)
        print(f"  n={n}: E_n = {En:.6f}")


    print(f"\n验证波函数正交归一性 (n,m)=(0,0),(0,1),(1,0),(1,1):")
    L = 5.0 * lB
    Ngrid = 80
    x = np.linspace(-L, L, Ngrid)
    y = np.linspace(-L, L, Ngrid)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    states = [(0, 0), (0, 1), (1, 0), (1, 1)]
    psis = []
    for n, m in states:
        psi = landau_orbital_wavefunction(n, m, Z, lB)
        psis.append(psi.flatten())
        norm = np.sum(np.abs(psi) ** 2) * dx * dy
        print(f"  ψ_({n},{m}) 范数 = {norm:.6f}")


    M = len(states)
    overlap = np.zeros((M, M), dtype=complex)
    for i in range(M):
        for j in range(M):
            overlap[i, j] = np.sum(np.conj(psis[i]) * psis[j]) * dx * dy
    print(f"\n重叠矩阵（应接近单位矩阵）:")
    for i in range(M):
        row = "  | " + " ".join([f"{overlap[i,j].real:8.5f}" for j in range(M)]) + " |"
        print(row)


    print(f"\n谱有限元刚度矩阵条件数测试:")
    for N in [4, 8, 12, 16]:
        K = build_spectral_stiffness_matrix(N, domain=(0.0, 1.0), nquad=200)
        cond = np.linalg.cond(K)
        print(f"  N={N:2d}: cond(K) = {cond:.4e}")


    print(f"\nLagrange基函数插值精度测试:")
    node_x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    node_v = np.sin(np.pi * node_x)
    test_x = np.linspace(0.0, 1.0, 101)
    u_h = local_fem_1d(len(node_x), node_x, node_v, test_x)
    u_exact = np.sin(np.pi * test_x)
    err = np.max(np.abs(u_h - u_exact))
    print(f"  对 sin(πx) 的五次Lagrange插值最大误差: {err:.6e}")

    print("\n[landau_levels.py] 测试完成。\n")


if __name__ == "__main__":
    test_landau_levels()
