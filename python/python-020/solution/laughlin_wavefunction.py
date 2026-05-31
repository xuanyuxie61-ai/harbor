# -*- coding: utf-8 -*-
"""
laughlin_wavefunction.py
Laughlin多体波函数与准粒子激发

核心物理：
  Laughlin在1983年提出，对于填充因子 ν = 1/m（m为奇整数），
  N电子系统的基态波函数为：

      Ψ_m(z_1, ..., z_N) = ∏_{i<j} (z_i - z_j)^m · exp(-Σ_k |z_k|²/4l_B²)

  其中 z_j = x_j + i y_j 为第 j 个电子的复坐标，l_B = √(ħ/eB) 为磁长度。

  该波函数具有以下关键性质：
  1. 当任意两个电子位置重合时，波函数以 (z_i - z_j)^m 趋于零 → 强关联排斥
  2. 交换两个电子：Ψ_m(...,z_i,...,z_j,...) = (-1)^m Ψ_m(...,z_j,...,z_z,...)
     因 m 为奇数，满足费米子反对称性
  3. 最大密度 droplet 的半径 R ≈ √(2mN) l_B

  准空穴激发（quasihole）：在位置 z_0 插入一个准空穴：
      Ψ_m^{qh}(z_0) = ∏_j (z_j - z_0) · Ψ_m(z_1,...,z_N)

  准空穴携带分数电荷 e* = e/m。

本模块融合原项目：
  - 552_hyperball_distance（高维空间距离统计）
  - 561_hypercube_surface_distance（超立方体表面采样统计）
"""
import numpy as np
from utils import magnetic_length, safe_exp, safe_log

# ============================================================================
# 1. Laughlin波函数计算
# ============================================================================

def laughlin_wavefunction(z, m, lB, return_log=False):
    """
    计算Laughlin多体波函数 Ψ_m(z_1, ..., z_N)。

    公式：
        Ψ_m = ∏_{i<j} (z_i - z_j)^m · exp(-Σ_k |z_k|² / 4l_B²)

    对于大N，直接计算会数值溢出，因此提供 return_log=True 选项。

    参数:
        z          : ndarray, shape (N,), 复坐标数组
        m          : int, Laughlin指数（奇正整数）
        lB         : float, 磁长度
        return_log : bool, 是否返回对数波函数

    返回:
        psi 或 log_psi
    """
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

    # Jastrow 因子: ∏_{i<j} (z_i - z_j)^m
    jastrow_log = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dz = z[i] - z[j]
            # 处理 dz = 0 的边界情况（加入小量避免log(0)）
            abs_dz = abs(dz)
            if abs_dz < 1e-15:
                abs_dz = 1e-15
            jastrow_log += m * (np.log(abs_dz) + 1j * np.angle(dz))

    # 高斯包络
    rho_sum = np.sum(np.abs(z) ** 2) / (4.0 * lB * lB)

    log_psi = jastrow_log - rho_sum

    if return_log:
        return log_psi

    # 从对数恢复波函数，注意数值稳定性
    psi = np.exp(log_psi.real) * (np.cos(log_psi.imag) + 1j * np.sin(log_psi.imag))
    return psi


def laughlin_log_probability(z, m, lB):
    """
    计算 |Ψ_m|² 的对数（即概率密度的对数）：
        ln|Ψ|² = 2m Σ_{i<j} ln|z_i - z_j| - Σ_k |z_k|²/(2l_B²)

    这在Metropolis Monte Carlo采样中非常有用。
    """
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


# ============================================================================
# 2. 准空穴/准电子激发
# ============================================================================

def quasihole_wavefunction(z, z0, m, lB, return_log=False):
    """
    在位置 z0 插入一个准空穴后的波函数：
        Ψ_m^{qh}(z_0) = ∏_j (z_j - z_0) · Ψ_m(z_1,...,z_N)

    准空穴携带分数电荷 e* = e/m。

    参数:
        z   : ndarray, shape (N,), 电子复坐标
        z0  : complex, 准空穴位置
        m   : int, Laughlin指数
        lB  : float, 磁长度

    返回:
        psi 或 log_psi
    """
    z = np.asarray(z, dtype=complex)
    N = len(z)
    if N < 1:
        raise ValueError("至少需要1个电子坐标")

    # 准空穴因子
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
    """
    在位置 z0 插入一个准电子后的波函数。
    准电子通过在Laughlin波函数上作用复共轭导数算符构造：
        Ψ_m^{qe}(z_0) = ∏_j (2∂_{z_j^*} - z_0^*) · Ψ_m

    数值实现中采用近似形式（投影到最低Landau能级）：
        Ψ_m^{qe} ≈ ∏_j (z_j^* - 2∂_{z_j}) · Ψ_m

    这里采用简化的低能有效模型：
        Ψ_m^{qe}(z_0) ≈ ∏_j (z_j^* - z_0^*) · Ψ_m / (某种归一化)

    准电子携带分数电荷 e* = -e/m。
    """
    z = np.asarray(z, dtype=complex)
    N = len(z)
    if N < 1:
        raise ValueError("至少需要1个电子坐标")

    # 简化模型：准电子因子为复共轭
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


# ============================================================================
# 3. 配对关联函数（融合原项目 552_hyperball_distance, 561_hypercube_surface_distance）
# ============================================================================

def pair_correlation_function(z, m, lB, r_bins=80, r_max=None):
    """
    计算Laughlin态的配对关联函数 g(r)。

    定义：
        g(r) = (L² / N(N-1)) ⟨ Σ_{i≠j} δ(r - |r_i - r_j|) ⟩

    对于均匀各向同性系统，g(r) 仅依赖于相对距离 r = |z_i - z_j|。

    在圆盘几何中，Laughlin态的g(r)在小r处表现为：
        g(r) ~ r^{2m}  (r → 0)
    这反映了电子间的强关联排斥。

    参数:
        z      : ndarray, shape (N,), 电子复坐标
        m      : int, Laughlin指数
        lB     : float, 磁长度
        r_bins : int, 距离分箱数
        r_max  : float or None, 最大距离

    返回:
        r_edges : ndarray, 距离分箱边界
        g_r     : ndarray, 配对关联函数值
    """
    z = np.asarray(z, dtype=complex)
    N = len(z)
    if N < 2:
        raise ValueError("至少需要2个电子")

    # 计算所有配对距离
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

    # 构建直方图
    g_r, r_edges = np.histogram(distances, bins=r_bins, range=(0.0, r_max))

    # 归一化：考虑二维环形面积元 dA = 2πr dr
    # 总电子数密度 n = N / (π R²)，R 为系统半径
    R_system = np.max(np.abs(z)) * 1.1
    area = np.pi * R_system ** 2
    n_density = N / area

    bin_widths = np.diff(r_edges)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    # 归一化因子：理想均匀气体的配对关联函数
    # g(r) = (L²/N(N-1)) * (histogram_count / (2πr dr))
    # 对于均匀分布，g(r) = 1
    for i in range(len(g_r)):
        r_c = r_centers[i]
        dr = bin_widths[i]
        shell_area = 2.0 * np.pi * r_c * dr
        if shell_area < 1e-15:
            g_r[i] = 0.0
            continue
        # 归一化
        norm = (N * (N - 1) / 2.0) * shell_area / area
        if norm < 1e-15:
            g_r[i] = 0.0
        else:
            g_r[i] = g_r[i] / norm

    return r_edges, g_r, r_centers


def structure_factor_s_q(z, m, lB, q_bins=60, q_max=None):
    """
    计算结构因子 S(q)，即密度涨落的Fourier变换：
        S(q) = (1/N) ⟨ |Σ_j exp(-iq·r_j)|² ⟩
             = 1 + (1/N) Σ_{i≠j} exp(-iq·(r_i - r_j))

    参数:
        z      : ndarray, 电子复坐标
        m      : int, Laughlin指数
        lB     : float, 磁长度
        q_bins : int, 波矢分箱数
        q_max  : float or None

    返回:
        q_vals : ndarray, 波矢大小
        S_q    : ndarray, 结构因子
    """
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
        # 在各向同性系统中，对角度平均
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


# ============================================================================
# 4. 波函数重叠与内积
# ============================================================================

def wavefunction_overlap(z_grid, psi1, psi2, dx, dy):
    """
    计算两个波函数在离散格点上的重叠积分：
        ⟨ψ_1|ψ_2⟩ = ∫ ψ_1^*(r) ψ_2(r) d²r ≈ Σ_{i,j} ψ_1^*(r_{ij}) ψ_2(r_{ij}) dx dy

    参数:
        z_grid : ndarray, 复格点坐标
        psi1, psi2 : ndarray, 波函数值
        dx, dy : float, 格点间距

    返回:
        overlap : complex, 重叠积分
    """
    psi1 = np.asarray(psi1, dtype=complex)
    psi2 = np.asarray(psi2, dtype=complex)
    if psi1.shape != psi2.shape:
        raise ValueError("两个波函数数组形状必须一致")
    return np.sum(np.conj(psi1) * psi2) * dx * dy


# ============================================================================
# 5. 测试接口
# ============================================================================
def test_laughlin_wavefunction():
    """测试Laughlin波函数模块。"""
    print("=" * 60)
    print("[laughlin_wavefunction.py] Laughlin波函数测试")
    print("=" * 60)

    B = 10.0
    m_star = 1.0
    lB = magnetic_length(B, m_star)
    m = 3  # ν = 1/3

    # 生成N个电子位置（在圆盘内均匀分布作为初始猜测）
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

    # 计算Laughlin波函数
    log_psi = laughlin_wavefunction(z, m, lB, return_log=True)
    print(f"\nLaughlin波函数对数值: Re(logΨ) = {log_psi.real:.4f}, Im(logΨ) = {log_psi.imag:.4f}")

    # 计算配对关联函数
    r_edges, g_r, r_centers = pair_correlation_function(z, m, lB, r_bins=40)
    print(f"\n配对关联函数 g(r) 前5个值:")
    for i in range(min(5, len(r_centers))):
        print(f"  r = {r_centers[i]:.4f}, g(r) = {g_r[i]:.6f}")

    # 检验 g(r) ~ r^{2m} 在小r处的行为
    # 由于只有8个电子，统计涨落大，主要验证程序正确性
    if len(g_r) > 2 and r_centers[1] > 0:
        slope_approx = np.log(g_r[2] + 1e-10) / np.log(r_centers[2] + 1e-10)
        print(f"  g(r) 小r近似幂次: {slope_approx:.2f} (理论值 ~ {2*m})")

    # 准空穴测试
    z0 = 0.5 * lB + 0.3j * lB
    log_psi_qh = quasihole_wavefunction(z, z0, m, lB, return_log=True)
    print(f"\n准空穴波函数 (z0={z0:.3f}):")
    print(f"  Re(logΨ_qh) = {log_psi_qh.real:.4f}")

    # 结构因子
    q_vals, S_q = structure_factor_s_q(z, m, lB, q_bins=30)
    print(f"\n结构因子 S(q) 前3个值:")
    for i in range(min(3, len(q_vals))):
        print(f"  q = {q_vals[i]:.4f}, S(q) = {S_q[i]:.6f}")

    print("\n[laughlin_wavefunction.py] 测试完成。\n")


if __name__ == "__main__":
    test_laughlin_wavefunction()
