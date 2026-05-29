"""
核裂变过程中的扩散系数与粘滞张量
===================================
融合原始项目:
  - 283_diffusion_pde: 扩散 PDE 参数与导数
  - 049_asa239: 正态累积分布函数 (alnorm)

科学背景:
---------
在核裂变的集体坐标描述中，内部自由度（核子运动）作为热浴
对集体运动产生耗散与涨落。根据涨落-耗散定理 (Fluctuation-Dissipation Theorem, FDT):

  γ_{ij} = (1/T) ∫_0^∞ dt ⟨ξ_i(t) ξ_j(0)⟩

其中 γ_{ij} 为粘滞张量，T 为核温度，ξ_i 为随机力。
Einstein 关系给出扩散系数:
  D_{ij} = T γ_{ij}

核温度由费米气体模型给出:
  T = √(E* / a)   [MeV]
其中 E* 为激发能，a = A/(8 MeV) 为能级密度参数。

粘滞系数的微观计算（基于线性响应理论）:
  γ_{ij} = (πℏ / T) Σ_{μ,ν} |⟨μ| ∂H/∂q_i |ν⟩|² n_μ(1-n_ν) δ(E_μ - E_ν)

宏观上，粘滞系数常采用 Wall-Formula 或 One-Body-Dissipation (OBD) 近似:
  γ_{wall} = (3/4) ρ v_F R₀² · (surface area ratio)
其中 ρ 为核物质密度，v_F 为费米速度。
"""

import numpy as np
from typing import Tuple

NUCLEON_DENSITY = 0.16
FERMI_VELOCITY = 4.5
FERMI_ENERGY = 38.0


def nuclear_temperature(excitation_energy: float, mass_number: int) -> float:
    """
    计算核温度 (MeV). 费米气体模型: T = √(E* / a).
    """
    if excitation_energy < 0:
        return 0.0
    if mass_number <= 0:
        raise ValueError("mass_number must be positive")
    a_level = mass_number / 8.0
    if excitation_energy < 1e-3:
        return excitation_energy / np.sqrt(a_level * excitation_energy + 1e-6)
    T = np.sqrt(excitation_energy / a_level)
    return float(T)


def wall_formula_viscosity(mass_number: int, beta2: float = 0.0) -> float:
    """
    Wall-Formula 粘滞系数. γ_wall = (3/4) ρ v_F R₀² B_s(β₂).
    """
    from collective_coordinates import nuclear_radius
    from potential_energy_surface import surface_area_ratio
    R0 = nuclear_radius(mass_number)
    Bs = surface_area_ratio(beta2)
    gamma = 0.75 * NUCLEON_DENSITY * FERMI_VELOCITY * (R0 ** 2) * Bs
    return float(gamma)


def one_body_dissipation(mass_number: int, charge_number: int, beta2: float, beta3: float = 0.0) -> np.ndarray:
    """
    计算集体坐标空间中的 One-Body-Dissipation 张量 (β₂, β₃) 子空间.
    """
    gamma_wall = wall_formula_viscosity(mass_number, beta2)
    f22 = 1.0 + 0.3 * beta2 + 0.1 * beta2 ** 2
    f33 = 0.5 + 0.2 * abs(beta3) + 0.05 * beta2 ** 2
    f23 = 0.1 * beta2 * beta3
    gamma = np.array([
        [gamma_wall * f22, gamma_wall * f23],
        [gamma_wall * f23, gamma_wall * f33],
    ])
    return gamma


def diffusion_tensor(excitation_energy: float, mass_number: int, charge_number: int,
                     beta2: float, beta3: float = 0.0) -> np.ndarray:
    """
    扩散张量 D = T · γ (Einstein 关系).
    """
    T = nuclear_temperature(excitation_energy, mass_number)
    gamma = one_body_dissipation(mass_number, charge_number, beta2, beta3)
    D = T * gamma
    return D


def diffusion_deriv_1d(t: float, u: np.ndarray, x: np.ndarray, mu: float,
                       source: np.ndarray = None) -> np.ndarray:
    """
    一维扩散方程右端项: du/dt = μ · u_xx + S(x).
    Neumann 零通量边界.
    """
    if len(u) < 3:
        raise ValueError("u must have at least 3 elements")
    if len(x) != len(u):
        raise ValueError("x and u must have the same length")
    n = len(u)
    dudt = np.zeros_like(u)
    for i in range(1, n - 1):
        dx_left = x[i] - x[i - 1]
        dx_right = x[i + 1] - x[i]
        dx_avg = 0.5 * (dx_left + dx_right)
        flux_right = mu * (u[i + 1] - u[i]) / dx_right
        flux_left = mu * (u[i] - u[i - 1]) / dx_left
        dudt[i] = (flux_right - flux_left) / dx_avg
    dudt[0] = dudt[1]
    dudt[-1] = dudt[-2]
    if source is not None:
        if len(source) != n:
            raise ValueError("source length mismatch")
        dudt = dudt + source
    return dudt


def alnorm(x: float, upper: bool = True) -> float:
    """
    标准正态分布累积密度函数 (Hill Algorithm AS 66).
    用于计算能量高于势垒的透射系数.
    """
    a1 = 5.75885480458
    a2 = 2.62433121679
    a3 = 5.92885724438
    b1 = -29.8213557807
    b2 = 48.6959930692
    c1 = -0.000000038052
    c2 = 0.000398064794
    c3 = -0.151679116635
    c4 = 4.8385912808
    c5 = 0.742380924027
    c6 = 3.99019417011
    con = 1.28
    d1 = 1.00000615302
    d2 = 1.98615381364
    d3 = 5.29330324926
    d4 = -15.1508972451
    d5 = 30.789933034
    ltone = 7.0
    p = 0.39894228044
    q = 0.39990348504
    r = 0.398942280385
    utzero = 18.66
    up = upper
    z = x
    if z < 0.0:
        up = not up
        z = -z
    if ltone < z and ((not up) or utzero < z):
        return 0.0 if up else 1.0
    y = 0.5 * z * z
    if z <= con:
        value = 0.5 - z * (p - q * y / (y + a1 + b1 / (y + a2 + b2 / (y + a3))))
    else:
        value = r * np.exp(-y) / (z + c1 + d1 / (z + c2 + d2 / (z + c3 + d3 / (z + c4 + d4 / (z + c5 + d5 / (z + c6))))))
    if not up:
        value = 1.0 - value
    return float(value)


def gammad(x: float, p: float) -> float:
    """
    不完全伽马函数比 P(X,P) = γ(x,p) / Γ(p) 的近似.
    在核物理中用于能级密度计算.
    """
    if x < 0.0 or p <= 0.0 or x == 0.0:
        return 0.0
    # 使用级数展开
    import math
    gln = math.lgamma(p)
    # 级数求和
    ap = p
    sum_val = 1.0 / p
    del_val = sum_val
    while True:
        ap += 1.0
        del_val *= x / ap
        sum_val += del_val
        if abs(del_val) < abs(sum_val) * 1e-7:
            break
        if ap > 10000:
            break
    result = sum_val * np.exp(-x + p * np.log(x) - gln)
    return float(np.clip(result, 0.0, 1.0))
