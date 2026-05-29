# -*- coding: utf-8 -*-
"""
landau_levels.py
Landau能级与单粒子基函数

核心物理：
  二维电子气在垂直磁场 B 中的哈密顿量：
      H = (1/2m*) [ (p_x + eA_x)² + (p_y + eA_y)² ]

  采用对称规范 A = (B/2)(-y, x, 0)，引入复坐标 z = x + iy，
  则单粒子本征态可写为：
      ψ_{n,m}(z) = N_{n,m} z^m L_n^m(|z|²/2l_B²) exp(-|z|²/4l_B²)
  其中 L_n^m 为连带Laguerre多项式，n 为Landau能级指标，
  m 为角动量量子数（-n ≤ m ≤ ∞）。

  能级：E_n = ħω_c (n + 1/2)

本模块融合原项目：
  - 399_fem1d_spectral_numeric（谱有限元多项式基）
  - 388_fem1d_display（一维有限元Lagrange基函数）
"""
import numpy as np
from scipy.special import genlaguerre, factorial
from utils import magnetic_length, cyclotron_frequency, landau_level_energy, H_BAR, E_CHARGE, gram_schmidt_qr

# ============================================================================
# 1. 连带Laguerre多项式与Landau轨道波函数
# ============================================================================

def landau_orbital_wavefunction(n, m, z, lB):
    """
    计算对称规范下的Landau单粒子轨道波函数。

    公式：
        ψ_{n,m}(z) = N_{n,m} · (z/√2 l_B)^m · L_n^m(|z|²/2l_B²)
                     · exp(-|z|²/4l_B²)

    其中归一化常数：
        N_{n,m} = (-1)^n / √(2π·2^m·l_B²) · √[n! / (n+m)!]

    参数:
        n : int, Landau能级指标 (n ≥ 0)
        m : int, 角动量量子数 (m ≥ -n)
        z : array_like, 复坐标 z = x + iy (单位与lB一致)
        lB: float, 磁长度

    返回:
        psi : ndarray, 与z同形状的复数波函数值
    """
    if n < 0:
        raise ValueError("Landau能级指标 n 必须 ≥ 0")
    if m < -n:
        raise ValueError(f"角动量量子数 m 必须 ≥ -n = {-n}")
    if lB <= 0:
        raise ValueError("磁长度 lB 必须为正")

    # ========== HOLE 1 START ==========
    # TODO: 实现对称规范下的Landau单粒子轨道波函数计算
    #
    # 核心物理公式：
    #   ψ_{n,m}(z) = N_{n,m} · (z/√2 l_B)^m · L_n^m(|z|²/2l_B²)
    #                · exp(-|z|²/4l_B²)
    #
    # 其中归一化常数：
    #   N_{n,m} = (-1)^n / √(2π·2^m·l_B²) · √[n! / (n+m)!]
    #
    # 需处理：
    #   1. 连带Laguerre多项式 L_n^{(|m|)}(|z|²/2l_B²) 的求值
    #   2. 归一化常数（注意 m < 0 时利用 L_n^{(m)} 与 L_{n+m}^{(-m)} 的关系）
    #   3. 角动量因子 (z / √2 l_B)^m
    #   4. 高斯包络 exp(-|z|²/4l_B²)
    #   5. 四者相乘得到 ψ
    #
    # 提示：scipy.special.genlaguerre(n, alpha) 可用于求 Laguerre 多项式
    raise NotImplementedError("Landau 轨道波函数计算待实现")
    # ========== HOLE 1 END ==========


# ============================================================================
# 2. 一维谱有限元基函数（用于Landau能级径向方程的谱展开）
# ============================================================================

def spectral_basis_1d(x, i, domain=(-1.0, 1.0)):
    """
    一维权多项式谱基函数（融合原项目 399_fem1d_spectral_numeric）。

    定义在区间 [a, b] 上，带Dirichlet边界条件的基函数：
        φ_i(x) = x^{i-1} · (x - a)(x - b),   i = 1, 2, ..., N

    参数:
        x     : array_like, 空间坐标
        i     : int, 基函数指标（从1开始）
        domain: tuple, (a, b) 定义域

    返回:
        phi   : ndarray, 基函数值
    """
    a, b = domain
    x = np.asarray(x, dtype=float)
    if i < 1:
        raise ValueError("基函数指标 i 必须 ≥ 1")
    if not (a < b):
        raise ValueError("定义域必须满足 a < b")

    phi = (x ** (i - 1)) * (x - a) * (x - b)
    return phi


def spectral_basis_derivative_1d(x, i, domain=(-1.0, 1.0)):
    """
    谱基函数的一阶导数：
        φ_i'(x) = (i-1)·x^{i-2}·(x-a)(x-b) + x^{i-1}·[(x-b)+(x-a)]
    当 i = 1 时，简化为 φ_1'(x) = (x-b) + (x-a) = 2x - a - b
    """
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
    """
    构建谱有限元的刚度矩阵与载荷向量。

    对于径向方程 -(d²u/dr²) = f(r)，弱形式给出刚度矩阵：
        K_{ki} = ∫_a^b φ_k'(x) φ_i'(x) dx

    使用Gauss-Legendre数值积分计算。

    参数:
        N     : int, 基函数个数
        domain: tuple, (a, b)
        nquad : int, 积分点数

    返回:
        K     : (N, N) ndarray, 刚度矩阵
    """
    a, b = domain
    # Gauss-Legendre 积分点与权重
    from numpy.polynomial.legendre import leggauss
    xi, wi = leggauss(nquad)
    # 映射到 [a, b]
    xq = 0.5 * (b - a) * xi + 0.5 * (b + a)
    wq = 0.5 * (b - a) * wi

    K = np.zeros((N, N), dtype=float)
    for k in range(1, N + 1):
        dphi_k = spectral_basis_derivative_1d(xq, k, domain)
        for i in range(1, N + 1):
            dphi_i = spectral_basis_derivative_1d(xq, i, domain)
            K[k - 1, i - 1] = np.sum(wq * dphi_k * dphi_i)
    return K


# ============================================================================
# 3. 有限元Lagrange基函数（融合原项目 388_fem1d_display）
# ============================================================================

def local_basis_1d_lagrange(order, node_x, x):
    """
    局部Lagrange插值基函数（融合原项目 388_fem1d_display 的 local_basis_1d）。

    在单元内，基函数满足：
        φ_j(x_i) = δ_{ij}

    采用拉格朗日插值公式：
        φ_j(x) = ∏_{k≠j} (x - x_k) / (x_j - x_k)

    参数:
        order : int, 单元阶数（节点数）
        node_x: array_like, 单元节点坐标（长度=order）
        x     : float or array_like, 求值点

    返回:
        phi   : ndarray, shape (order,) 或 (len(x), order)
    """
    node_x = np.asarray(node_x, dtype=float)
    if len(node_x) != order:
        raise ValueError("node_x 长度必须与 order 一致")
    # 检查节点是否互异
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
    """
    局部有限元函数求值：
        u_h(x) = Σ_j v_j · φ_j(x)

    参数:
        order   : int, 单元阶数
        node_x  : array_like, 单元节点坐标
        node_v  : array_like, 节点处函数值
        sample_x: array_like, 求值点

    返回:
        sample_v: ndarray, 求值点处的函数值
    """
    node_x = np.asarray(node_x, dtype=float)
    node_v = np.asarray(node_v, dtype=float)
    sample_x = np.atleast_1d(sample_x)
    phi = local_basis_1d_lagrange(order, node_x, sample_x)
    sample_v = phi @ node_v
    return sample_v


# ============================================================================
# 4. Landau能级简并度与态密度
# ============================================================================

def landau_degeneracy(B, A, m_star=1.0):
    """
    单个Landau能级的简并度：
        N_Φ = BA / Φ_0 = eBA / (2πħ)
    其中 Φ_0 = h/e = 2πħ/e 为磁通量子。

    参数:
        B     : float, 磁场强度 (T)
        A     : float, 样品面积
        m_star: float, 有效质量

    返回:
        N_phi : float, 简并度
    """
    if B <= 0 or A <= 0:
        raise ValueError("B 和 A 必须为正")
    flux_quantum = 2.0 * np.pi * H_BAR / E_CHARGE
    return B * A / flux_quantum


def density_of_states_landau(E, B, m_star=1.0, gamma=0.01):
    """
    Landau能级的态密度（用Lorentzian展宽）：
        D(E) = (eB/h) Σ_n  (γ/π) / [(E - E_n)² + γ²]

    参数:
        E     : float or array_like, 能量
        B     : float, 磁场强度
        m_star: float, 有效质量
        gamma : float, 展宽参数（模拟无序效应）

    返回:
        dos   : ndarray, 态密度
    """
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


# ============================================================================
# 5. 完整测试接口
# ============================================================================
def test_landau_levels():
    """
    测试Landau能级系统：计算前几个能级的波函数并验证正交归一性。
    """
    print("=" * 60)
    print("[landau_levels.py] Landau能级与单粒子基函数测试")
    print("=" * 60)

    B = 10.0           # 磁场强度
    m_star = 1.0       # 有效质量
    lB = magnetic_length(B, m_star)
    omega_c = cyclotron_frequency(B, m_star)

    print(f"\n物理参数:")
    print(f"  磁场 B = {B} T")
    print(f"  有效质量 m* = {m_star}")
    print(f"  磁长度 l_B = {lB:.6f}")
    print(f"  回旋频率 ω_c = {omega_c:.6f}")

    # 测试能级
    print(f"\n前5个Landau能级能量 E_n = ħω_c(n + 1/2):")
    for n in range(5):
        En = landau_level_energy(n, B, m_star)
        print(f"  n={n}: E_n = {En:.6f}")

    # 测试波函数正交归一性
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

    # 重叠矩阵
    M = len(states)
    overlap = np.zeros((M, M), dtype=complex)
    for i in range(M):
        for j in range(M):
            overlap[i, j] = np.sum(np.conj(psis[i]) * psis[j]) * dx * dy
    print(f"\n重叠矩阵（应接近单位矩阵）:")
    for i in range(M):
        row = "  | " + " ".join([f"{overlap[i,j].real:8.5f}" for j in range(M)]) + " |"
        print(row)

    # 测试谱FEM刚度矩阵
    print(f"\n谱有限元刚度矩阵条件数测试:")
    for N in [4, 8, 12, 16]:
        K = build_spectral_stiffness_matrix(N, domain=(0.0, 1.0), nquad=200)
        cond = np.linalg.cond(K)
        print(f"  N={N:2d}: cond(K) = {cond:.4e}")

    # 测试Lagrange基函数
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
