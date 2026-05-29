# -*- coding: utf-8 -*-
"""
topological_invariants.py
拓扑不变量计算：Berry相位与Chern数

核心物理：
  在分数量子霍尔效应中，拓扑不变量表征了系统的全局性质。

  1. Berry相位：
     当参数 λ 沿闭合回路 C 绝热演化时，Berry相位为：
         γ_n(C) = i ∮_C ⟨u_n(λ)|∇_λ|u_n(λ)⟩ · dλ

     对于磁场中的二维电子气，参数空间为磁通量 (Φ_x, Φ_y)，
     Berry联络定义为：
         A_μ(λ) = i ⟨u_n(λ)|∂_{λ_μ}|u_n(λ)⟩

  2. Chern数（第一陈类）：
     在参数空间（即布里渊区）上，Chern数为Berry曲率的积分：
         C = (1/2π) ∫_{T²} d²k · Ω(k)

     其中Berry曲率为：
         Ω(k) = ∂_{k_x} A_y(k) - ∂_{k_y} A_x(k)
              = i [⟨∂_{k_x} u|∂_{k_y} u⟩ - ⟨∂_{k_y} u|∂_{k_x} u⟩]

     对于Landau能级，每个能级的Chern数精确为：
         C_n = 1    (对所有n)
     这反映了每个Landau能级贡献一个手性边缘模。

  3. 填充因子与Chern数的关系：
         σ_{xy} = (e²/h) · C
     这就是整数量子霍尔效应中的平台化电导。

  4. 对于分数量子霍尔态，使用TKNN公式：
         σ_{xy} = (e²/h) · (1/(2πi)) Σ_{n∈occ} ∫ d²k
                  [⟨∂_{k_x} u_n|∂_{k_y} u_n⟩ - ⟨∂_{k_y} u_n|∂_{k_x} u_n⟩]

本模块融合原项目：
  - 005_analemma（轨道角度参数化与周期性演化）
"""
import numpy as np
from utils import H_BAR, E_CHARGE

# ============================================================================
# 1. Berry联络与Berry曲率
# ============================================================================

def berry_connection(u_k, u_k_plus_dk, dk):
    """
    离散化的Berry联络：
        A_μ ≈ i ⟨u(k)|u(k+dk)⟩ / dk

    参数:
        u_k         : ndarray, |u(k)⟩ 波函数
        u_k_plus_dk : ndarray, |u(k+dk)⟩ 波函数
        dk          : float, 参数步长

    返回:
        A           : complex, Berry联络分量
    """
    u_k = np.asarray(u_k, dtype=complex)
    u_k_plus_dk = np.asarray(u_k_plus_dk, dtype=complex)
    overlap = np.vdot(u_k, u_k_plus_dk)
    # 保持相位连续性
    if abs(overlap) < 1e-14:
        return 0.0
    # Berry联络的离散近似
    A = 1j * np.log(overlap / abs(overlap)) / dk
    return A


def berry_curvature_discrete(u_grid, kx_grid, ky_grid):
    """
    离散化的Berry曲率计算（适用于二维参数空间格点）。

    采用四点plaquette公式：
        Ω_{ij} = -arg[⟨u_{i,j}|u_{i+1,j}⟩⟨u_{i+1,j}|u_{i+1,j+1}⟩
                        ⟨u_{i+1,j+1}|u_{i,j+1}⟩⟨u_{i,j+1}|u_{i,j}⟩]

    参数:
        u_grid   : ndarray, shape (Nx, Ny, N_states), 参数空间上的波函数
        kx_grid  : ndarray, shape (Nx,)
        ky_grid  : ndarray, shape (Ny,)

    返回:
        Omega    : ndarray, shape (Nx-1, Ny-1), Berry曲率格点
        kx_centers : ndarray
        ky_centers : ndarray
    """
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

            # 四个重叠积分
            o1 = np.vdot(u00, u10)
            o2 = np.vdot(u10, u11)
            o3 = np.vdot(u11, u01)
            o4 = np.vdot(u01, u00)

            # 处理零重叠
            phases = []
            for o in [o1, o2, o3, o4]:
                if abs(o) > 1e-14:
                    phases.append(np.angle(o))
                else:
                    phases.append(0.0)

            # Berry曲率 = - 总相位环绕
            phi_total = phases[0] + phases[1] + phases[2] + phases[3]
            # 规范化到 [-π, π]
            phi_total = (phi_total + np.pi) % (2.0 * np.pi) - np.pi
            Omega[i, j] = -phi_total

    return Omega, kx_centers, ky_centers


# ============================================================================
# 2. Chern数计算
# ============================================================================

def chern_number_from_berry_curvature(Omega, dkx, dky):
    """
    从Berry曲率计算Chern数：
        C = (1/2π) ∫ d²k Ω(k)
          ≈ (1/2π) Σ_{i,j} Ω_{ij} dkx dky

    参数:
        Omega : ndarray, Berry曲率格点
        dkx   : float, kx方向格点间距
        dky   : float, ky方向格点间距

    返回:
        C     : float, Chern数（应为接近整数的值）
    """
    integral = np.sum(Omega) * dkx * dky
    C = integral / (2.0 * np.pi)
    return C


def tknn_conductivity(Omega_sum, dkx, dky):
    """
    TKNN公式计算霍尔电导：
        σ_{xy} = (e²/h) · C

    其中 C 为Chern数。

    参数:
        Omega_sum : float, 所有占据态Berry曲率之和
        dkx, dky  : float, k空间格点间距

    返回:
        sigma_xy  : float, 霍尔电导 (单位 e²/h)
    """
    C = Omega_sum * dkx * dky / (2.0 * np.pi)
    return C  # 以 e²/h 为单位


# ============================================================================
# 3. 填充因子与拓扑序
# ============================================================================

def filling_factor_from_chern(C, degeneracy_per_level):
    """
    从Chern数和能级简并度推导填充因子：
        ν = N_e / (C · N_Φ)

    参数:
        C                    : float, Chern数
        degeneracy_per_level : float, 每个Landau能级的简并度

    返回:
        nu                   : float, 填充因子
    """
    if abs(C) < 1e-14:
        return np.inf
    return 1.0 / C


def conductance_quantization(nu, m_laughlin=None):
    """
    分数量子霍尔电导量子化：
        σ_{xy} = (e²/h) · ν

    对于Laughlin态 ν = 1/m：
        σ_{xy} = (e²/h) · (1/m)

    参数:
        nu        : float, 填充因子
        m_laughlin: int or None, Laughlin指数

    返回:
        sigma_xy  : float, 电导
    """
    if m_laughlin is not None:
        nu = 1.0 / m_laughlin
    return nu


# ============================================================================
# 4. 磁通量子化与轨道演化（融合原项目 005_analemma）
# ============================================================================

def flux_quantization_phase(n_phi, n_e, m_laughlin=3):
    """
    计算通过绝热插入磁通量子引起的Berry相位：
        γ = 2π · (n_e / n_Φ) = 2π ν

    对于Laughlin态，每插入一个磁通量子，系统产生一个准空穴：
        ΔQ = e/m

    参数:
        n_phi      : int, 磁通量子数
        n_e        : int, 电子数
        m_Laughlin : int, Laughlin指数

    返回:
        phase      : float, Berry相位
        charge     : float, 产生的分数电荷
    """
    if n_phi <= 0:
        raise ValueError("n_phi 必须为正")
    nu = n_e / n_phi
    phase = 2.0 * np.pi * nu
    charge = E_CHARGE / m_laughlin
    return phase, charge


def orbital_evolution_parameters(ecc, lon_deg, obliq_deg, n_points=100):
    """
    轨道参数演化（融合原项目 005_analemma 的角度参数化思想）。

    在量子霍尔系统中，用绝热参数替代天文参数：
        τ  →  磁通量参数
        ecc →  形变参数（椭圆度）
        lon →  相位偏移
        obliq→  倾斜角（层间耦合）

    参数:
        ecc      : float, 轨道偏心率（映射为形变参数）
        lon_deg  : float, 近日点经度（度）
        obliq_deg: float, 轨道倾角（度）
        n_points : int, 采样点数

    返回:
        tau      : ndarray, 参数数组
        theta    : ndarray, 绝热演化角度
    """
    lon = lon_deg * np.pi / 180.0
    obliq = obliq_deg * np.pi / 180.0

    tau = np.linspace(0.0, 2.0 * np.pi, n_points)

    # 真近点角（类比磁通量参数化）
    theta = np.arctan2(
        np.sqrt(1.0 - ecc ** 2) * np.sin(tau),
        np.cos(tau) - ecc
    )

    # 旋转操作（类比规范变换）
    x1 = np.cos(theta - (lon - np.pi / 2.0))
    y1 = np.sin(theta - (lon - np.pi / 2.0))

    # 倾角旋转（类比层间耦合）
    x2 = np.cos(obliq) * x1
    y2 = y1
    z2 = -np.sin(obliq) * x1

    # 返回参数化轨道
    return tau, theta, x2, y2, z2


# ============================================================================
# 5. 测试接口
# ============================================================================
def test_topological_invariants():
    """测试拓扑不变量模块。"""
    print("=" * 60)
    print("[topological_invariants.py] 拓扑不变量测试")
    print("=" * 60)

    # 测试Berry曲率与Chern数（简单模型：二维Dirac费米子）
    print("\n1. Berry曲率与Chern数测试 (Haldane模型近似):")
    Nk = 40
    kx = np.linspace(-np.pi, np.pi, Nk)
    ky = np.linspace(-np.pi, np.pi, Nk)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')

    # 构建简单的两能带模型波函数
    # H = d·σ, d = (sin kx, sin ky, m + cos kx + cos ky)
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
                # 低能带波函数
                u_grid[i, j] = np.array([
                    np.sqrt((1.0 - dz / dm) / 2.0),
                    (dx + 1j * dy) / np.sqrt(2.0 * dm * (dm - dz))
                ])
                # 处理 dz ≈ dm 的情况
                if abs(dm - dz) < 1e-14:
                    u_grid[i, j] = np.array([0.0, 1.0])

    Omega, kxc, kyc = berry_curvature_discrete(u_grid, kx, ky)
    dkx = kx[1] - kx[0]
    dky = ky[1] - ky[0]
    C = chern_number_from_berry_curvature(Omega, dkx, dky)
    print(f"   计算Chern数: C = {C:.4f} (预期 ≈ 0，因为 m={m_mass}>2)")

    # 改变质量符号
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

    # 测试磁通量子化
    print("\n2. 磁通量子化Berry相位测试:")
    for n_phi in [3, 6, 9, 12]:
        n_e = n_phi // 3
        phase, charge = flux_quantization_phase(n_phi, n_e, m_laughlin=3)
        print(f"   n_Φ={n_phi:2d}, n_e={n_e:2d}: phase={phase:.4f} rad, charge={charge:.6f}")

    # 测试轨道参数演化
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
