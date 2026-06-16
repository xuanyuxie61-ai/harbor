"""
physical_constants.py — 基本物理常数与无量纲化单位
============================================================

本项目采用原子单位制 (Hartree atomic units):
    ℏ = m_e = e = 1,  a_0 = 1 (Bohr radius)

在此单位下:
  - 能量单位: Hartree = e²/(4πε₀a₀) ≈ 27.211 eV
  - 长度单位: Bohr radius a₀ ≈ 0.529 Å
  - 磁场单位: ℏ/(e·a₀²) ≈ 1 (自然单位)
  - 磁长度: l_B = 1/√B

量子霍尔效应关键无量纲参数:
    ν = n_e·h/(eB)     — 填充因子 (filling factor)
    α = ℏω_c/(k_BT)   — 温度与回旋能量的比值
    ω_c = eB/m         — 回旋频率 (cyclotron frequency)

参考文献:
    [1] Girvin, S. M. & Prange, R. E. "The Quantum Hall Effect" (Springer, 1990)
    [2] Prange, R. E. & Girvin, S. M. (Eds.) Springer Series in Solid-State Sciences
"""

import numpy as np

# ──────────────────────────────────────────────────────────────
# 国际单位制基本常数 (CODATA 2018)
# ──────────────────────────────────────────────────────────────
HBAR = 1.054571817e-34       # ℏ  [J·s]
ELECTRON_MASS = 9.1093837015e-31   # m_e  [kg]
ELECTRON_CHARGE = 1.602176634e-19  # e    [C]
HBAR_EV = 6.582119569e-16   # ℏ  [eV·s]
BOHR_RADIUS = 5.29177210903e-11    # a₀   [m]
HARTREE_ENERGY = 4.3597447222071e-18  # E_h [J]
SPEED_OF_LIGHT = 299792458.0  # c   [m/s]
VACUUM_PERMITTIVITY = 8.8541878128e-12  # ε₀ [F/m]
VACUUM_PERMEABILITY = 1.25663706212e-6  # μ₀ [H/m]
FLUX_QUANTUM = 2.06783384846e-15  # Φ₀ = h/e [Wb]
BOLTZMANN = 1.380649e-23     # k_B [J/K]

# ──────────────────────────────────────────────────────────────
# 无量纲参数 (原子单位: ℏ = m_e = e = 1)
# ──────────────────────────────────────────────────────────────
def cyclotron_frequency(B: float) -> float:
    """回旋频率 ω_c = eB/m (原子单位: ω_c = B)

    Args:
        B: 磁感应强度 [原子单位]
    Returns:
        ω_c: 回旋频率 [原子单位]
    """
    return B


def magnetic_length(B: float) -> float:
    """磁长度 l_B = √(ℏ/(eB)) (原子单位: l_B = 1/√B)

    磁长度定义了朗道能级波函数的空间展宽:
        ψ_n(x,y) ∝ H_n(x/l_B) · exp(-x²/(2l_B²))

    Args:
        B: 磁感应强度 [原子单位]
    Returns:
        l_B: 磁长度 [原子单位]
    """
    if B <= 0:
        raise ValueError(f"磁场必须为正, 当前 B = {B}")
    return 1.0 / np.sqrt(B)


def landau_level_energy(n: int, B: float) -> float:
    """第 n 个朗道能级能量 (原子单位: E_n = B(n + 1/2))

    朗道能级是均匀磁场中二维电子气的精确解:
        E_n = ℏω_c(n + 1/2),  n = 0, 1, 2, ...

    每个能级的简并度: g = BA/(2π) = Φ/Φ₀
    其中 Φ = BA 是穿过面积 A 的总磁通量

    Args:
        n: 朗道能级指标 (n ≥ 0)
        B: 磁感应强度 [原子单位]
    Returns:
        E_n: 能级能量 [原子单位]
    """
    if n < 0:
        raise ValueError(f"朗道能级指标不能为负, 当前 n = {n}")
    return B * (n + 0.5)


def filling_factor(density: float, B: float) -> float:
    """填充因子 ν = 2π·n_e / B (原子单位)

    填充因子决定了量子霍尔态的性质:
      - 整数填充 ν = 1, 2, 3, ... → 整数量子霍尔效应 (IQHE)
      - 分数填充 ν = 1/3, 2/5, ... → 分数量子霍尔效应 (FQHE)

    Args:
        density: 电子面密度 n_e [原子单位]
        B: 磁感应强度 [原子单位]
    Returns:
        ν: 填充因子
    """
    if B <= 0:
        raise ValueError(f"磁场必须为正, 当前 B = {B}")
    return 2.0 * np.pi * density / B


def peierls_phase(A_line_integral: float) -> complex:
    """Peierls 相位因子: exp(i·∫A·dl / ℏ) (原子单位: exp(i·∫A·dl))

    Peierls 替换是格点模型中引入磁场的标准方法:
        t_{ij} → t_{ij} · exp(i·(e/ℏ)·∫_{r_i}^{r_j} A·dl)

    在原子单位下简化为: t_{ij} → t_{ij} · exp(i·∫A·dl)

    该相位的规范不变性确保了物理结果不依赖于规范选择.

    Args:
        A_line_integral: 矢量势的线积分 ∫A·dl
    Returns:
        exp(i·∫A·dl): Peierls 相位因子
    """
    return np.exp(1j * A_line_integral)


def dirac_notation_inner(psi: np.ndarray, phi: np.ndarray) -> complex:
    """计算量子态的内积 ⟨ψ|φ⟩ = Σ ψ*_i · φ_i

    Args:
        psi: 左矢 (bra) 波函数
        phi: 右矢 (ket) 波函数
    Returns:
        ⟨ψ|φ⟩: 内积
    """
    return np.vdot(psi, phi)


def expectation_value(operator: np.ndarray, state: np.ndarray) -> complex:
    """计算算符在给定态下的期望值 ⟨ψ|Ô|ψ⟩

    Args:
        operator: 算符矩阵 Ô
        state: 量子态 |ψ⟩
    Returns:
        ⟨ψ|Ô|ψ⟩: 期望值
    """
    return np.vdot(state, operator @ state)


def bohr_sommerfeld_quantization(action: float, n: int) -> float:
    """Bohr-Sommerfeld 量子化条件检验: |S - 2π(n+1/2)|

    在半经典近似下, 闭合轨道的相空间面积必须满足:
        ∮ p·dq = 2πℏ(n + γ),  γ = 1/2 (Maslov index)

    对于朗道能级, S = 2π·E/B, 所以 E = B(n + 1/2).

    Args:
        action: 相空间作用量 S
        n: 量子数
    Returns:
        量子化误差 |S - 2π(n+1/2)|
    """
    return abs(action - 2.0 * np.pi * (n + 0.5))
