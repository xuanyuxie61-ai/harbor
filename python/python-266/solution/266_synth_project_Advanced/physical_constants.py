"""
physical_constants.py — 物理常数与原子单位 (Atomic Units)
=========================================================

基于 Hartree 原子单位系统 (ℏ = m_e = e = 4πε₀ = 1):

  能量单位: 1 Hartree = 27.2114 eV
  长度单位: 1 Bohr = 0.529177 Å
  时间单位: ℏ/E_h = 2.4189×10⁻¹⁷ s

本模块提供 DFT 能带计算所需的所有基本物理常数,
以及单位转换因子和高阶有限差分模板系数。

高阶 FD 系数推导:
  对于 2p 阶中心差分, 模板系数 c_j 满足:
    Σ_{j=1}^{p} c_j [sin(j·k·dx)/(k·dx)]² = 1 - (k·dx)²/12 + ...

  更精确地, 由 Taylor 展开匹配:
    Σ_{j=1}^{p} c_j / j^{2m} = (-1)^{m+1}  for m = 1,...,p

  其中 c_j 为第 j 个邻居的权重。
"""

import numpy as np
from typing import Dict, Tuple, List

# ============================================================
# 基本物理常数 (SI 单位)
# ============================================================
HBAR_SI = 1.054571817e-34        # ℏ: 约化 Planck 常数 [J·s]
MASS_ELECTRON = 9.1093837015e-31  # m_e: 电子质量 [kg]
ELECTRON_CHARGE = 1.602176634e-19 # e: 基本电荷 [C]
BOHR_RADIUS = 5.29177210903e-11   # a₀: Bohr 半径 [m]
HARTREE_ENERGY = 4.3597447222071e-18  # E_h: Hartree 能量 [J]
SPEED_OF_LIGHT = 2.99792458e8     # c: 光速 [m/s]
BOLTZMANN_SI = 1.380649e-23       # k_B: Boltzmann 常数 [J/K]
RYDBERG_ENERGY = HARTREE_ENERGY / 2.0  # 1 Ry = 0.5 Ha

# ============================================================
# 原子单位转换因子
# ============================================================
EV_PER_HARTREE = 27.211386245988  # 1 Ha → eV
ANGSTROM_PER_BOHR = 0.529177210903  # 1 Bohr → Å
KELVIN_PER_HARTREE = 3.1577502480407e5  # 1 Ha → K

# ============================================================
# 计算凝聚态常用常数
# ============================================================
PI = np.pi
TWO_PI = 2.0 * PI
FOUR_PI = 4.0 * PI
SQRT_TWO = np.sqrt(2.0)
SQRT_PI = np.sqrt(PI)

# ============================================================
# 高阶有限差分模板系数 (2p 阶中心差分)
# ============================================================

def fd_coefficients_2nd(order: int) -> np.ndarray:
    """
    计算 2p 阶精度的 Laplacian 有限差分系数。

    对于 d²u/dx² 的 2p 阶中心差分近似:
      (d²u/dx²) ≈ (1/dx²) Σ_{j=-p}^{p} c_j u(x + j·dx)

    系数 c_j 由以下条件确定:
      Σ_{j=1}^{p} c_j = -1/2 (归一化, 对 j=0 项)
      Σ_{j=1}^{p} c_j · j^{2m} = 0  for m = 1,...,p-1
      Σ_{j=1}^{p} c_j · j^{2p} ≠ 0 (leading error term)

    具体地, 通过求解 Vandermonde 型线性方程组:
      [1²  2²  3²  ...  p² ] [c₁]   [-1/2]
      [1⁴  2⁴  3⁴  ...  p⁴ ] [c₂] = [0   ]
      [...                  ] [..]   [...  ]
      [1^{2p} ...       p^{2p}] [c_p]   [0   ]

    然后 leading error 为:
      ε = (dx²)^p · (-1)^{p+1} / ((2p+1)!) · Σ c_j j^{2p+2}

    Parameters
    ----------
    order : int
        半带宽 p, 总精度为 2p 阶。
        order=1 → 3点 (O(dx²))
        order=2 → 5点 (O(dx⁴))
        order=3 → 7点 (O(dx⁶))
        order=4 → 9点 (O(dx⁸))
        order=5 → 11点 (O(dx¹⁰))

    Returns
    -------
    coeffs : np.ndarray, shape (2p+1,)
        完整的 FD 模板系数, 索引 p 对应中心点。
        满足 Σ c_j = 0 (Laplacian 零和性质)。
    """
    if order < 1 or order > 10:
        raise ValueError(f"FD order {order} 超出范围 [1, 10]")

    p = order
    # 构建 Vandermonde 矩阵:
    # 条件: Σ_{j=1}^{p} c_j · j^{2m} = δ_{m,0} · (-1/2)  for m=0,...,p-1
    # 其中 m=0 时: Σ c_j = -1/2 (中心系数 c_0 = -2Σc_j = 1 → 需归一化)
    #
    # 更标准的形式 (Numerical Recipes §19.3):
    # 求解 A·c = b 其中 A[m,j] = (j+1)^{2(m+1)} / (2(m+1))!

    # 使用精确公式: 对于 2p 阶 Laplacian FD
    # c_0 = -2 · Σ_{j=1}^{p} c_j
    # c_j = c_{-j} (对称性)
    #
    # 通过匹配 Taylor 展开到 O(dx^{2p}):
    # Σ_{j=1}^{p} c_j · j^{2m} = 0  for m = 0 (Σ c_j 自由)
    #
    # 标准方法: 求解线性系统
    # [1   1   1  ...  1  ] [c₁]     [1]
    # [1²  2²  3² ...  p² ] [c₂]     [0]
    # [1⁴  2⁴  3⁴ ...  p⁴ ] [c₃]  =  [0]
    # [...                 ] [...]    [...]
    # [1^{2(p-1)} ...     ] [c_p]    [0]
    # 然后 c_0 = -2·Σ_{j=1}^{p} c_j

    A = np.zeros((p, p))
    b = np.zeros(p)
    b[0] = 1.0  # 归一化条件 (m=0)

    for m in range(p):
        for j in range(p):
            # 匹配 (k·dx)^{2m} 的系数
            # 对于 m=0: Σ c_j · 1 = 归一化
            # 对于 m>0: Σ c_j · j^{2m} / (2m)! 项
            n_val = j + 1  # j = 1, 2, ..., p
            if m == 0:
                A[m, j] = 1.0
            else:
                A[m, j] = n_val ** (2 * m)

    # 实际上需要更精确的系数计算
    # 使用 Fornberg 算法的思想
    coeffs_half = _compute_laplacian_fd_weights(p)

    # 构建完整系数数组
    coeffs = np.zeros(2 * p + 1)
    for j in range(1, p + 1):
        coeffs[p + j] = coeffs_half[j]
        coeffs[p - j] = coeffs_half[j]  # 对称性
    coeffs[p] = -2.0 * np.sum(coeffs_half[1:])  # 中心点

    return coeffs


def _compute_laplacian_fd_weights(p: int) -> np.ndarray:
    """
    计算 Laplacian 有限差分权重的内部函数。

    通过求解 Vandermonde 系统获得精确的 FD 系数:
      V · c = e₁

    其中 V[i,j] = j^{2i} for i=0,...,p-1, j=1,...,p
    这是从 Taylor 展开条件导出的:
      Σ_{j=1}^{p} c_j · (j·dx)^{2i} = δ_{i,0} · dx²  for i=0,...,p-1

    等价于要求 FD 算子在多项式基 {1, x², x⁴, ..., x^{2p-2}} 上精确。
    """
    V = np.zeros((p, p))
    rhs = np.zeros(p)
    rhs[0] = 1.0  # dx² 归一化

    for i in range(p):
        for j in range(p):
            n_val = j + 1
            V[i, j] = n_val ** (2 * (i + 1))

    # 求解
    try:
        c = np.linalg.solve(V, rhs)
    except np.linalg.LinAlgError:
        # 退化情况使用伪逆
        c = np.linalg.lstsq(V, rhs, rcond=None)[0]

    # 扩展为 j=0,1,...,p (j=0 位置为占位)
    result = np.zeros(p + 1)
    result[1:] = c
    return result


def fd_modified_wavenumber(kdx: np.ndarray, order: int) -> np.ndarray:
    """
    计算有限差分算子的修正波数 (modified wavenumber)。

    对于精确的 -d²/dx² 算子, 本征值为 k²。
    对于 FD 离散化, 本征值为:
      k²_mod = (1/dx²) · Σ_{j=1}^{p} 2·c_j·(1 - cos(j·k·dx))

    修正波数的相对误差:
      ε(k) = k²_mod / k² - 1

    当 k·dx → 0 时, ε → 0 (一致性)。
    当 k·dx → π 时, 误差最大 (Nyquist 极限)。

    对于 2p 阶 FD:
      ε(k) ≈ (-1)^p · (k·dx)^{2p} / (2p+1)! · Σ c_j j^{2p+2}

    Parameters
    ----------
    kdx : np.ndarray
        无量纲波数 k·dx
    order : int
        FD 半带宽 p

    Returns
    -------
    k2_mod : np.ndarray
        修正波数的平方 k²_mod · dx²
    """
    c = fd_coefficients_2nd(order)
    p = order

    k2_mod = np.zeros_like(kdx, dtype=float)
    for j in range(-p, p + 1):
        if j == 0:
            k2_mod += -c[p]  # c[p] is the center coefficient
        else:
            # exp(i·j·kdx) 的实部给出 cos 贡献
            k2_mod += -c[p + j] * np.cos(j * kdx)

    return k2_mod


# ============================================================
# Bernoulli 数 (用于 Euler-Maclaurin 求和和 trigamma 渐近展开)
# ============================================================

_BERNOULLI_NUMBERS: Dict[int, float] = {
    0: 1.0,
    1: -0.5,
    2: 1.0 / 6.0,
    4: -1.0 / 30.0,
    6: 1.0 / 42.0,
    8: -1.0 / 30.0,
    10: 5.0 / 66.0,
    12: -691.0 / 2730.0,
    14: 7.0 / 6.0,
    16: -3617.0 / 510.0,
    18: 43867.0 / 798.0,
    20: -174611.0 / 330.0,
}


def bernoulli_number(n: int) -> float:
    """
    返回第 n 个 Bernoulli 数 B_n。

    Bernoulli 数在以下场景中出现:
    1. Euler-Maclaurin 求和公式 (BZ 积分精度分析)
    2. trigamma 函数的渐近展开 (费米-狄拉克积分)
    3. Riemann zeta 函数的特殊值: ζ(2n) = (-1)^{n+1} B_{2n} (2π)^{2n} / (2(2n)!)

    递归定义:
      B_0 = 1
      Σ_{k=0}^{n} C(n+1,k) B_k = 0  for n ≥ 1

    奇数项 (n ≥ 3) 均为零。
    """
    if n < 0:
        raise ValueError(f"Bernoulli 数未定义于 n={n} < 0")
    if n == 1:
        return -0.5
    if n % 2 == 1 and n >= 3:
        return 0.0
    if n in _BERNOULLI_NUMBERS:
        return _BERNOULLI_NUMBERS[n]
    # 对于更大的偶数 n, 使用递归
    return _bernoulli_recursive(n)


def _bernoulli_recursive(n: int) -> float:
    """通过递归公式计算 Bernoulli 数 B_n (n 偶数)。"""
    from math import comb
    if n == 0:
        return 1.0
    if n % 2 == 1:
        return 0.0

    s = 0.0
    for k in range(0, n):
        s += comb(n + 1, k) * bernoulli_number(k)
    return -s / comb(n + 1, n)


# ============================================================
# 物理模型参数 (1D Mathieu 型晶体)
# ============================================================

def get_default_crystal_params() -> Dict[str, float]:
    """
    返回默认的一维晶体模型参数。

    模型: Mathieu 型周期势
      V(x) = V₁ cos(2πx/a) + V₂ cos(4πx/a) + V₃ cos(6πx/a)

    这是 Kronig-Penney 模型的平滑近似, 在固态物理中
    用于研究能隙结构和 Brillouin 区折叠。

    参数选择原则:
    - V₁ 控制第一 Brillouin 区边界的主要能隙
    - V₂ 控制 Γ 点和 X 点之间的能带曲率
    - V₃ 提供更高阶的能带修正

    物理单位: 原子单位 (Hartree, Bohr)
    """
    return {
        'a_lattice': 10.0,       # 晶格常数 a [Bohr]
        'V1': 0.5,               # 一次 Fourier 分量 [Ha]
        'V2': 0.15,              # 二次 Fourier 分量 [Ha]
        'V3': 0.03,              # 三次 Fourier 分量 [Ha]
        'n_electrons': 2,        # 每个原胞的电子数
        'n_bands': 4,            # 计算的能带数
        'n_kpoints': 32,         # BZ 采样 k 点数
        'n_grid': 128,           # 实空间网格点数
        'fd_order': 3,           # FD 精度阶数 (半带宽 p)
        'scf_tol': 1e-6,         # SCF 收敛阈值 [Ha/Bohr³]
        'scf_max_iter': 40,      # SCF 最大迭代数
        'mixing_alpha': 0.2,     # 线性混合参数
        'temperature': 0.001,    # 电子温度 [Ha] (用于 Fermi 展宽)
        'hartree_strength': 0.1, # Hartree 势强度参数
    }


# ============================================================
# 数学辅助: 组合数和阶乘 (用于 FD 误差分析)
# ============================================================

def fd_dispersion_error(kdx: float, order: int) -> float:
    """
    计算 FD 色散关系的相对误差。

    对于平面波 e^{ikx}, FD 算子给出:
      (-d²/dx²)_FD e^{ikx} = k²_mod · e^{ikx}

    其中 k²_mod dx² = -2 Σ_{j=1}^{p} c_j cos(j·kdx) + c_center

    相对误差定义为:
      δ(kdx) = |k²_mod - kdx²| / kdx²   (对 kdx ≠ 0)

    当 kdx → 0, δ → 0; 当 kdx → π, δ 达到最大。

    对于 2p 阶 FD, 在 kdx << 1 时:
      δ ≈ (kdx)^{2p} · |Σ c_j j^{2p+2}| / ((2p+2)!)

    Parameters
    ----------
    kdx : float
        无量纲波数
    order : int
        FD 半带宽

    Returns
    -------
    error : float
        相对误差 |k²_mod/k² - 1|
    """
    if abs(kdx) < 1e-14:
        return 0.0

    c = fd_coefficients_2nd(order)
    p = order

    k2_mod = 0.0
    for j in range(-p, p + 1):
        k2_mod += -c[p + j] * np.cos(j * kdx)

    k2_exact = kdx ** 2
    return abs(k2_mod - k2_exact) / k2_exact
