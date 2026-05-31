"""
谱分析与波数离散化约束模块

基于种子项目：
  - 155_change_diophantine：N 维丢番图方程求解

核心物理模型：
  1. 波数空间离散化约束：
     在谱方法或 Floquet-Bloch 分析中，周期性边界条件要求波数满足：
         k = 2π (n_x/L_x, n_y/L_y, n_z/L_z)
     其中 n_x, n_y, n_z 为整数。
     对于海洋平台 Bragg 共振问题，入射波与结构反射波满足：
         k_I = k_R + G
     其中 G 为倒格矢（reciprocal lattice vector），
     G = h_1 b_1 + h_2 b_2，h_i 为整数（丢番图约束）。

  2. 色散关系约束：
     深水：ω² = g|k|
     有限水深：ω² = g|k| tanh(|k|h)
     结合波数整数约束，频率也必须满足离散化关系。

  3. 丢番图方程（源自 155_change_diophantine）：
     给定整数系数 a = [a_1, ..., a_m] 和右端 b，
     求所有非负整数解 x 满足：
         a_1 x_1 + a_2 x_2 + ... + a_m x_m = b
     使用回溯法枚举所有解。

  4. 波数组合与频率匹配：
     对于平台立柱阵列（间距 L），Bragg 共振条件要求：
         2k cosθ = n · (2π/L),  n ∈ ℤ⁺
     这等价于一个一维丢番图约束。
"""

import numpy as np
from typing import List, Tuple, Optional
from utils import check_well_posed_diophantine, gcd_vector


# ======================================================================
# 1. 丢番图方程求解（源自 155_change_diophantine）
# ======================================================================

def diophantine_nd_nonnegative_solutions(
    a: np.ndarray, b: int
) -> List[np.ndarray]:
    """
    求解 a·x = b 的所有非负整数解。
    使用递归回溯法，按字典序枚举。
    返回解向量列表。
    """
    a = np.asarray(a, dtype=int)
    if not check_well_posed_diophantine(a, b):
        return []
    m = len(a)
    solutions = []
    _backtrack_diophantine(a, b, 0, np.zeros(m, dtype=int), solutions)
    return solutions


def _backtrack_diophantine(
    a: np.ndarray,
    remaining: int,
    idx: int,
    current: np.ndarray,
    solutions: List[np.ndarray],
):
    """递归回溯枚举丢番图方程的解。"""
    m = len(a)
    if idx == m - 1:
        if remaining % a[idx] == 0:
            current[idx] = remaining // a[idx]
            solutions.append(current.copy())
        return
    max_val = remaining // a[idx]
    for val in range(max_val, -1, -1):
        current[idx] = val
        _backtrack_diophantine(a, remaining - val * a[idx], idx + 1, current, solutions)


def diophantine_solution_count(a: np.ndarray, b: int) -> int:
    """计算 a·x = b 的非负整数解个数。"""
    return len(diophantine_nd_nonnegative_solutions(a, b))


# ======================================================================
# 2. 波数离散化约束
# ======================================================================

def wavenumber_discrete_constraint_bragg(
    wavelength: float,
    column_spacing: float,
    incidence_angle: float,
    max_order: int = 5,
) -> List[Tuple[int, float]]:
    """
    计算 Bragg 共振条件下的波数丢番图约束解。
    条件：2k cosθ = n · (2π/L)
    其中 L 为立柱间距，θ 为入射角。
    对于给定波长 λ，k = 2π/λ，检查哪些整数 n 满足条件。
    返回满足条件的 (n, 误差) 列表。
    """
    k = 2.0 * np.pi / wavelength
    cos_theta = np.cos(incidence_angle)
    if abs(cos_theta) < 1e-12:
        return []
    target = 2.0 * k * cos_theta
    solutions = []
    for n in range(1, max_order + 1):
        required = n * (2.0 * np.pi / column_spacing)
        error = abs(target - required)
        relative_error = error / abs(target) if abs(target) > 1e-12 else error
        if relative_error < 0.1:  # 10% 容差
            solutions.append((n, relative_error))
    return solutions


def wavenumber_discrete_constraint_floquet(
    domain_lengths: np.ndarray,
    max_modes: int = 3,
    omega: float = 1.0,
    h: float = 100.0,
) -> List[dict]:
    """
    Floquet-Bloch 波数离散化：
    在周期性域 [0,Lx]×[0,Ly] 中，允许波数：
        k = (h_1 · 2π/L_x, h_2 · 2π/L_y)
    结合色散关系 ω² = g|k| tanh(|k|h)，寻找整数对 (h_1, h_2)
    使得频率匹配。
    返回满足色散关系的模式列表。
    """
    domain_lengths = np.asarray(domain_lengths, dtype=float)
    if len(domain_lengths) < 2:
        raise ValueError("domain_lengths 至少包含两个元素")
    Lx, Ly = domain_lengths[0], domain_lengths[1]
    g = 9.80665
    omega2 = omega * omega
    modes = []
    for h1 in range(-max_modes, max_modes + 1):
        for h2 in range(-max_modes, max_modes + 1):
            if h1 == 0 and h2 == 0:
                continue
            kx = h1 * 2.0 * np.pi / Lx
            ky = h2 * 2.0 * np.pi / Ly
            k_mag = np.sqrt(kx ** 2 + ky ** 2)
            if k_mag < 1e-12:
                continue
            kh = k_mag * h
            if kh > 100:
                tanh_kh = 1.0
            else:
                tanh_kh = np.tanh(kh)
            omega_k = np.sqrt(g * k_mag * tanh_kh)
            rel_err = abs(omega_k - omega) / omega
            if rel_err < 0.15:
                modes.append(
                    {
                        "h1": h1,
                        "h2": h2,
                        "kx": kx,
                        "ky": ky,
                        "k_mag": k_mag,
                        "omega_computed": omega_k,
                        "relative_error": rel_err,
                    }
                )
    # 按误差排序
    modes.sort(key=lambda x: x["relative_error"])
    return modes


def generate_allowed_wavenumbers_diophantine(
    a_coeffs: np.ndarray,
    b_total: int,
    domain_scale: float = 100.0,
) -> np.ndarray:
    """
    使用丢番图方程的解生成允许的波数组合。
    设 a_coeffs 为频率/波数基底的整数权重，
    解 x_i 给出各基底模式的数量，总波数矢量为：
        k = Σ x_i · (2π / domain_scale) · e_i
    返回波数矢量数组。
    """
    solutions = diophantine_nd_nonnegative_solutions(a_coeffs, b_total)
    wavenumbers = []
    base_k = 2.0 * np.pi / domain_scale
    for sol in solutions:
        k_vec = np.zeros(len(a_coeffs))
        for i, xi in enumerate(sol):
            k_vec[i] = xi * base_k
        wavenumbers.append(k_vec)
    return np.array(wavenumbers)


# ======================================================================
# 3. 功率谱密度与响应谱分析
# ======================================================================

def response_spectrum_rao(
    omega: np.ndarray,
    omega_n: float,
    zeta: float,
    wave_spectrum: np.ndarray,
) -> np.ndarray:
    """
    计算单自由度系统响应谱（RAO² × S_η）。
    传递函数：
        |H(ω)|² = 1 / [ (1 - (ω/ω_n)²)² + (2ζ ω/ω_n)² ]
    响应谱：S_ξ(ω) = |H(ω)|² · S_η(ω)
    """
    omega = np.asarray(omega, dtype=float)
    wave_spectrum = np.asarray(wave_spectrum, dtype=float)
    if len(omega) != len(wave_spectrum):
        raise ValueError("omega 与 wave_spectrum 长度不一致")
    r = omega / omega_n
    r = np.where(r <= 0, 1e-12, r)
    H2 = 1.0 / ((1.0 - r ** 2) ** 2 + (2.0 * zeta * r) ** 2)
    return H2 * wave_spectrum


def significant_response_from_spectrum(
    response_spectrum: np.ndarray, omega: np.ndarray
) -> float:
    """
    由响应谱计算特征响应幅值：
        ξ_{1/3} = 2 √(m_0),  m_0 = ∫ S_ξ(ω) dω
    """
    if len(omega) < 2:
        return 0.0
    m0 = np.trapezoid(response_spectrum, omega)
    m0 = max(m0, 0.0)
    return 2.0 * np.sqrt(m0)


def spectral_moments(
    spectrum: np.ndarray, omega: np.ndarray, max_order: int = 4
) -> List[float]:
    """
    计算谱矩 m_n = ∫ ω^n S(ω) dω。
    返回 [m_0, m_1, ..., m_max_order]。
    """
    moments = []
    for n in range(max_order + 1):
        integrand = (omega ** n) * spectrum
        mn = np.trapezoid(integrand, omega)
        moments.append(float(mn))
    return moments


def spectral_bandwidth_params(
    spectrum: np.ndarray, omega: np.ndarray
) -> dict:
    """
    计算谱带宽参数：
      - 谱宽参数 ε = √(1 - m_2² / (m_0 m_4))
      - 平均周期 T_01 = 2π m_0 / m_1
      - 平均周期 T_02 = 2π √(m_0 / m_2)
    """
    moments = spectral_moments(spectrum, omega, max_order=4)
    m0, m1, m2, _, m4 = moments
    epsilon = 0.0
    if m0 > 1e-15 and m4 > 1e-15:
        epsilon = np.sqrt(max(0.0, 1.0 - (m2 ** 2) / (m0 * m4)))
    T01 = 2.0 * np.pi * m0 / m1 if m1 > 1e-15 else 0.0
    T02 = 2.0 * np.pi * np.sqrt(m0 / m2) if m2 > 1e-15 else 0.0
    return {
        "m0": m0,
        "m1": m1,
        "m2": m2,
        "m4": m4,
        "epsilon": epsilon,
        "T01": T01,
        "T02": T02,
    }


# ======================================================================
# 4. 波浪-结构相互作用的谱展开
# ======================================================================

def diffraction_transfer_function_diophantine(
    panel_ks: np.ndarray,
    incident_k: float,
    a_coeffs: np.ndarray,
    b_constraint: int,
) -> np.ndarray:
    """
    基于丢番图约束的绕射传递函数。
    将面板法得到的波数 panel_ks 与入射波数 incident_k 的差
    限制在丢番图方程解集中，筛选满足倒格矢约束的面板贡献。
    返回各面板的传递系数数组。
    """
    solutions = diophantine_nd_nonnegative_solutions(a_coeffs, b_constraint)
    if len(solutions) == 0:
        return np.ones(len(panel_ks))
    allowed_deltas = set()
    for sol in solutions:
        allowed_deltas.add(int(np.sum(sol)))
    transfer = np.zeros(len(panel_ks))
    for i, pk in enumerate(panel_ks):
        delta = abs(int(round(pk - incident_k)))
        if delta in allowed_deltas:
            transfer[i] = 1.0
        else:
            transfer[i] = 0.1  # 非共振模式贡献衰减
    return transfer
