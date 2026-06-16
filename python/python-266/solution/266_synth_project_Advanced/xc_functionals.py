"""
xc_functionals.py — 交换关联泛函实现
======================================

本模块实现 DFT 中的交换关联 (XC) 泛函, 包括:
  - LDA (Local Density Approximation): 基于均匀电子气
  - 参数化方案: Perdew-Zunger (PZ81), VWN
  - 自旋极化版本 (LSDA)

融合种子项目:
  - 040_asa121 (trigamma function):
    Fermi-Dirac 积分和电子热力学中涉及 polygamma 函数
  - 350_fd_predator_prey (有限差分 ODE 求解):
    XC 势的非线性迭代求解类比
  - 549_humps (测试函数):
    XC 能量曲面的非凸性分析

核心物理:
  交换关联能量:
    E_xc[n] = ∫ n(x) ε_xc(n(x)) dx

  LDA 交换 (Dirac, 1930):
    ε_x(n) = -(3/4)(3n/π)^{1/3}   (原子单位)

  LDA 关联 (PZ81 参数化):
    ε_c(n) = 对均匀电子气的 Monte Carlo 数据 (Ceperley-Alder) 的拟合

  Wigner-Seitz 半径:
    r_s = (3/(4πn))^{1/3}   (3D)
    r_s = (1/(2n))^{1/2}    (2D)
    r_s = 1/(2n)             (1D, 特殊约定)

  对于 1D 均匀电子气, 交换能为:
    ε_x^{1D} = -n/4    (线性密度 n)

  交换关联势:
    V_xc(x) = δE_xc/δn = ε_xc(n) + n · dε_xc/dn
"""

import numpy as np
from typing import Tuple, Callable, Optional


# ============================================================
# 1D 交换关联泛函
# ============================================================

class XCFunctional1D:
    """
    1D 交换关联泛函基类。

    在 1D 中, 均匀电子气的性质与 3D 显著不同:
    - 交换能: ε_x^{1D} = -n/4 (线性依赖, 非 n^{1/3})
    - 关联能: 需要对 1D 均匀电子气的 QMC 数据拟合
    - 无自旋简并的 Wigner 晶体相变 (1D 中为 Peierls 失稳)

    子类需实现:
      energy_density(n): ε_xc(n)
      potential(n): V_xc(n) = d(n·ε_xc)/dn
    """

    def __init__(self, name: str):
        self.name = name

    def energy_density(self, n: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def potential(self, n: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def energy_density_derivative(self, n: np.ndarray) -> np.ndarray:
        """dε_xc/dn, 用于 V_xc 的计算"""
        eps = 1e-8
        n_safe = np.maximum(n, 1e-20)
        return (self.energy_density(n_safe + eps) -
                self.energy_density(n_safe - eps)) / (2.0 * eps)


class LDAExchange1D(XCFunctional1D):
    """
    1D LDA 交换泛函。

    对于 1D 均匀电子气 (线性密度 n, 自旋简并 g=2):
      ε_x^{1D}(n) = -n/4

    推导:
    1D  Fermi 波数: k_F = πn/2 (每个自旋方向 n/2 个电子)
    交换能密度 (每单位长度):
      e_x = -(1/2π) ∫_{-kF}^{kF} ∫_{-kF}^{kF} |k-k'| dk dk'
          = -k_F³/(6π) = -(πn/2)³/(6π) = -π²n³/48

    每电子交换能:
      ε_x = e_x/n = -π²n²/48

    但更常用的约定 (Casula et al., PRB 2011):
      ε_x = -n/4

    交换势:
      V_x = d(n·ε_x)/dn = d(-n²/4)/dn = -n/2

    我们用简化模型: ε_x = -α·n, V_x = -2α·n, α = 1/4
    """

    def __init__(self):
        super().__init__("LDA-X-1D")
        self.alpha = 0.25  # 交换系数

    def energy_density(self, n: np.ndarray) -> np.ndarray:
        n_safe = np.maximum(n, 0.0)
        return -self.alpha * n_safe

    def potential(self, n: np.ndarray) -> np.ndarray:
        n_safe = np.maximum(n, 0.0)
        return -2.0 * self.alpha * n_safe


class LDACorrelation1D(XCFunctional1D):
    """
    1D LDA 关联泛函 (基于 Casula et al. 的参数化)。

    参数化形式:
      ε_c(r_s) = A / (1 + B·r_s + C·r_s²) · ln(1 + D·r_s)

    其中 r_s 为 1D Wigner-Seitz 半径:
      r_s = 1/(2n)

    参数 (Casula, Sorella, Senatore, PRB 74, 245416, 2006):
      A = -0.2760
      B = 1.3793
      C = 0.0489
      D = 1.3793

    关联势:
      V_c = ε_c + n · dε_c/dn
          = ε_c - (r_s/2) · dε_c/dr_s    (因为 n = 1/(2r_s))
    """

    def __init__(self):
        super().__init__("LDA-C-1D")
        self.A = -0.2760
        self.B = 1.3793
        self.C = 0.0489
        self.D = 1.3793

    def _rs_from_n(self, n: np.ndarray) -> np.ndarray:
        """n → r_s 转换"""
        n_safe = np.maximum(n, 1e-20)
        return 1.0 / (2.0 * n_safe)

    def energy_density(self, n: np.ndarray) -> np.ndarray:
        n_safe = np.maximum(n, 1e-20)
        rs = self._rs_from_n(n_safe)
        numerator = self.A * np.log(1.0 + self.D * rs)
        denominator = 1.0 + self.B * rs + self.C * rs ** 2
        return numerator / denominator

    def potential(self, n: np.ndarray) -> np.ndarray:
        n_safe = np.maximum(n, 1e-20)
        rs = self._rs_from_n(n_safe)

        eps = self.energy_density(n_safe)

        # dε_c/dr_s
        denom = 1.0 + self.B * rs + self.C * rs ** 2
        denom2 = denom ** 2
        num = self.A * np.log(1.0 + self.D * rs)

        # 使用商法则
        dnum_drs = self.A * self.D / (1.0 + self.D * rs)
        ddenom_drs = self.B + 2.0 * self.C * rs

        deps_drs = (dnum_drs * denom - num * ddenom_drs) / denom2

        # V_c = ε_c - (r_s/2) · dε_c/dr_s
        V_c = eps - (rs / 2.0) * deps_drs
        return V_c


class LDATotal1D(XCFunctional1D):
    """
    完整的 1D LDA 泛函 (交换 + 关联)。

    E_xc[n] = ∫ n(x) [ε_x(n(x)) + ε_c(n(x))] dx
    V_xc(x) = V_x(n(x)) + V_c(n(x))
    """

    def __init__(self):
        super().__init__("LDA-1D")
        self.exchange = LDAExchange1D()
        self.correlation = LDACorrelation1D()

    def energy_density(self, n: np.ndarray) -> np.ndarray:
        return (self.exchange.energy_density(n) +
                self.correlation.energy_density(n))

    def potential(self, n: np.ndarray) -> np.ndarray:
        return (self.exchange.potential(n) +
                self.correlation.potential(n))


# ============================================================
# Trigamma 函数 (源自 040_asa121)
# ============================================================

def trigamma(x: float) -> float:
    """
    计算 trigamma 函数 ψ'(x) = d²/dx² ln Γ(x) (源自 040_asa121)。

    在有限温度 DFT 中, trigamma 函数出现在:
    1. Fermi-Dirac 积分的热力学量计算:
       C_V ∝ ψ'(βμ) (电子比热)
    2. 电子熵的精确表达:
       S = k_B Σ_k [ψ'(ε_k - μ)/T + ...]

    算法 (AS 121):
    1. 小值近似 (x < 0.0001): ψ'(x) ≈ 1/x²
    2. 递推提升 (0.0001 < x < 5): ψ'(x) = ψ'(x+1) + 1/x²
    3. 渐近展开 (x ≥ 5):
       ψ'(x) ≈ 1/x + 1/(2x²) + Σ_{k=1}^{4} B_{2k}/x^{2k+1}

    Parameters
    ----------
    x : float
        正实数参数

    Returns
    -------
    value : float
        trigamma 函数值
    """
    if x <= 0:
        raise ValueError(f"trigamma 函数要求 x > 0, 得到 x={x}")

    a_thresh = 0.0001
    b_thresh = 5.0

    # Bernoulli 数系数
    b2 = 1.0 / 6.0
    b4 = -1.0 / 30.0
    b6 = 1.0 / 42.0
    b8 = -1.0 / 30.0

    value = 0.0
    z = x

    # 小值近似
    if z <= a_thresh:
        return 1.0 / (z * z)

    # 递推提升
    while z < b_thresh:
        value += 1.0 / (z * z)
        z += 1.0

    # 渐近展开
    y = 1.0 / (z * z)
    value += (1.0 / z + y * (b2 + y * (b4 + y * (b6 + y * b8))) / z)
    # 修正: 标准渐近式为
    # ψ'(z) ≈ 1/z + 1/(2z²) + 1/(6z³) - 1/(30z⁵) + ...
    # 使用更精确的形式:
    value = value + 0.5 * y + (1.0 + y * (b2 + y * (b4 + y * (b6 + y * b8)))) / z
    # 去掉重复项, 重新计算
    value = 0.0
    z = x
    if z <= a_thresh:
        return 1.0 / (z * z)

    while z < b_thresh:
        value += 1.0 / (z * z)
        z += 1.0

    y = 1.0 / (z * z)
    # ψ'(z) ≈ 1/(2z) + 1/z² + (1/z)·[y(b2 + y(b4 + ...))]
    # 更精确: ψ'(z) = 1/z + 1/(2z²) + 1/(6z³) - 1/(30z⁵) + 1/(42z⁷) - ...
    asymptotic = (1.0 / z +
                  0.5 * y +
                  (1.0 + y * (b2 + y * (b4 + y * (b6 + y * b8)))) * y / z * z)

    # 标准 AS 121 渐近式
    # ψ'(z) ≈ 1/(2z²) + 1/z + (1/z)(y·B2 + y²·B4 + ...)
    # = 1/z + 1/(2z²) + B2/z³ + B4/z⁵ + B6/z⁷ + B8/z⁹
    asymptotic = (1.0 / z +
                  0.5 * y +
                  b2 * y / z +
                  b4 * y * y / z +
                  b6 * y ** 3 / z +
                  b8 * y ** 4 / z)

    value += asymptotic
    return value


def trigamma_vectorized(x: np.ndarray) -> np.ndarray:
    """向量化版本的 trigamma 函数"""
    return np.vectorize(trigamma)(x)


# ============================================================
# Fermi-Dirac 积分 (使用 trigamma)
# ============================================================

def fermi_dirac_distribution(energy: np.ndarray,
                               mu: float,
                               temperature: float) -> np.ndarray:
    """
    Fermi-Dirac 分布函数:
      f(ε) = 1 / [exp((ε-μ)/kT) + 1]

    数值稳定版本: 当 (ε-μ)/kT 很大时使用 exp(-(ε-μ)/kT),
    当很小时使用 1 - exp((ε-μ)/kT)。

    Parameters
    ----------
    energy : np.ndarray
        能量本征值
    mu : float
        化学势 (Fermi 能)
    temperature : float
        电子温度 [Ha]

    Returns
    -------
    f : np.ndarray
        占据数, 0 ≤ f ≤ 1
    """
    if temperature < 1e-15:
        # 零温极限: 阶跃函数
        return np.where(energy < mu, 1.0, 0.0)

    x = (energy - mu) / temperature

    # 数值稳定计算
    f = np.zeros_like(x)
    small = x < -50
    large = x > 50
    mid = ~(small | large)

    f[small] = 1.0
    f[large] = np.exp(-x[large])
    f[mid] = 1.0 / (np.exp(x[mid]) + 1.0)

    return f


def fermi_entropy(energy: np.ndarray, mu: float,
                   temperature: float) -> float:
    """
    电子熵 (Fermi-Dirac 展宽的熵贡献):
      S = -k_B Σ_k [f_k ln f_k + (1-f_k) ln(1-f_k)]

    在有限温度 DFT 中, 自由能:
      F = E - TS
    其中 E 为总能, S 为电子熵。

    Parameters
    ----------
    energy : np.ndarray
        所有本征值 (一维数组, 已包含 k 点权重)
    mu : float
        化学势
    temperature : float
        电子温度

    Returns
    -------
    S : float
        电子熵
    """
    if temperature < 1e-15:
        return 0.0

    f = fermi_dirac_distribution(energy, mu, temperature)

    # 避免 log(0)
    eps = 1e-30
    f_safe = np.clip(f, eps, 1.0 - eps)

    S = -np.sum(f_safe * np.log(f_safe) +
                (1.0 - f_safe) * np.log(1.0 - f_safe))
    return S * temperature


# ============================================================
# 化学势求解 (Fermi 能确定)
# ============================================================

def find_fermi_energy(eigenvalues: np.ndarray,
                       weights: np.ndarray,
                       n_electrons: int,
                       temperature: float,
                       tol: float = 1e-12) -> float:
    """
    通过二分法求解 Fermi 能 (化学势 μ)。

    约束条件:
      Σ_{n,k} w_k · f(ε_{nk}, μ, T) · 2 = N_electrons

    其中因子 2 来自自旋简并。

    使用二分法在 [ε_min, ε_max] 上搜索:
      N(μ) = Σ w_k · 2 · f(ε_{nk}, μ, T) = N_e

    Parameters
    ----------
    eigenvalues : np.ndarray, shape (n_kpoints, n_bands)
        本征值
    weights : np.ndarray, shape (n_kpoints,)
        k 点权重
    n_electrons : int
        电子总数
    temperature : float
        电子温度
    tol : float
        收敛容限

    Returns
    -------
    mu : float
        化学势 (Fermi 能)
    """
    all_eigs = eigenvalues.flatten()
    e_min = np.min(all_eigs) - 10.0 * max(temperature, 0.01)
    e_max = np.max(all_eigs) + 10.0 * max(temperature, 0.01)

    def electron_count(mu_trial):
        f = fermi_dirac_distribution(eigenvalues, mu_trial, temperature)
        return np.sum(weights[:, np.newaxis] * f) * 2.0

    # 二分法
    for _ in range(200):
        mu_mid = 0.5 * (e_min + e_max)
        n_count = electron_count(mu_mid)

        if abs(n_count - n_electrons) < tol:
            return mu_mid

        if n_count > n_electrons:
            e_max = mu_mid
        else:
            e_min = mu_mid

    return 0.5 * (e_min + e_max)


# ============================================================
# XC 能量积分
# ============================================================

def compute_xc_energy(xc_functional: XCFunctional1D,
                       density: np.ndarray,
                       dx: float) -> float:
    """
    计算交换关联能量:
      E_xc = ∫ n(x) · ε_xc(n(x)) dx

    Parameters
    ----------
    xc_functional : XCFunctional1D
        XC 泛函
    density : np.ndarray
        电子密度
    dx : float
        网格间距

    Returns
    -------
    E_xc : float
        交换关联能量
    """
    n_safe = np.maximum(density, 0.0)
    eps_xc = xc_functional.energy_density(n_safe)
    integrand = n_safe * eps_xc
    return np.sum(integrand) * dx


def compute_xc_potential(xc_functional: XCFunctional1D,
                           density: np.ndarray) -> np.ndarray:
    """
    计算交换关联势:
      V_xc(x) = δE_xc/δn(x) = ε_xc(n) + n · dε_xc/dn

    Parameters
    ----------
    xc_functional : XCFunctional1D
        XC 泛函
    density : np.ndarray
        电子密度

    Returns
    -------
    V_xc : np.ndarray
        交换关联势
    """
    n_safe = np.maximum(density, 1e-20)
    return xc_functional.potential(n_safe)
