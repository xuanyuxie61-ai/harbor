"""
quantum_objective.py
====================
量子光学目标函数 —— 基于光子态密度和介电函数的优化地景

融合种子项目:
  - 1242_ce335805_PhotonDosReference: SPhP 光子态密度, 介电函数, 频率积分
  - 234_cube_integrals: 单位立方体积分与采样

核心公式:
  1. LO 介电函数 (LST 关系): ε(ω) = ε_∞ (ω²_LO - ω²) / (ω²_TO - ω²)
  2. 表面极化激元 (SPhP) 色散: 1 + ε(ω) = 0 ⇒ ω_SPhP = √((ε_∞ ω²_LO + ω²_TO)/(ε_∞+1))
  3. 光子态密度: ρ(ω) = ω²/(π²c³) √ε(ω)^{3/2}
  4. Casimir 有效质量: m_eff ∝ ∫ dω ρ(ω,z) - ρ(ω,∞)
  5. 单胞积分: ∫_{[0,1]^d} Π x_i^{e_i} dx = Π 1/(e_i+1)
  6. 单调价值: Π x_i^{e_i} 在点 x 的求值
"""

import numpy as np
from typing import Callable, Tuple, Optional, Dict
import math


# ---------------------------------------------------------------------------
# 1. 介电函数 (源自 1242_ce335805_PhotonDosReference/SPhP/epsilonFunctions)
# ---------------------------------------------------------------------------

class DielectricModel:
    """Lorentz-Drude 介电模型.

    LO 介电函数 (单振子模型):
      ε(ω) = ε_∞ · (ω²_LO - ω² - iγω) / (ω²_TO - ω² - iγω)

    对无损 STiO₃ (γ→0):
      ε(ω) = ε_∞ · (ω²_LO - ω²) / (ω²_TO - ω²)

    参数:
      ω_TO: 横光学声子频率 (~7.92 THz for STO)
      ω_LO: 纵光学声子频率 (~32.04 THz for STO)
      ε_∞: 高频介电常数
      γ: 阻尼系数 (可选)
    """

    def __init__(self, wTO: float = 7.92e12, wLO: float = 32.04e12,
                 eps_inf: float = 1.0, gamma: float = 0.0):
        self.wTO = wTO
        self.wLO = wLO
        self.eps_inf = eps_inf
        self.gamma = gamma

    def epsilon(self, omega: complex) -> complex:
        """计算介电函数 ε(ω).
        源自 1242_ce335805/SPhP/epsilonFunctions.py.

        ε(ω) = ε_∞ · (ω²_LO - ω²) / (ω²_TO - ω²)  (无损)
        有损时: ε(ω) = ε_∞ · (ω²_LO - ω² - iγω) / (ω²_TO - ω² - iγω)
        """
        w2 = omega ** 2
        wTO2 = self.wTO ** 2
        wLO2 = self.wLO ** 2

        if self.gamma > 0:
            igw = 1j * self.gamma * omega
            return self.eps_inf * (wLO2 - w2 - igw) / (wTO2 - w2 - igw)
        else:
            denom = wTO2 - w2
            if abs(denom) < 1e-300:
                return complex(1e10, 0)  # 避免除零
            return self.eps_inf * (wLO2 - w2) / denom

    def epsilon_real(self, omega: float) -> float:
        """实数频率下的实介电常数."""
        return float(np.real(self.epsilon(complex(omega, 0))))

    def surface_mode_frequency(self) -> float:
        """表面极化激元 (SPhP) 频率.
        由 1 + ε(ω) = 0 解出:
          ω_SPhP = √((ε_∞ ω²_LO + ω²_TO) / (ε_∞ + 1))
        """
        return np.sqrt((self.eps_inf * self.wLO ** 2 + self.wTO ** 2)
                       / (self.eps_inf + 1.0))


# ---------------------------------------------------------------------------
# 2. 光子态密度 (源自 1242_ce335805)
# ---------------------------------------------------------------------------

def photon_dos_bulk(omega: float, eps_model: DielectricModel) -> float:
    """体光子态密度 (3D).
    ρ_bulk(ω) = ω²/(π²c³) · ε(ω)^{3/2}

    源自 1242_ce335805/SPhP/dosAsOfFreq.py.
    """
    c = 3e8  # 光速
    eps = eps_model.epsilon_real(omega)
    if eps <= 0:
        return 0.0
    return omega ** 2 / (np.pi ** 2 * c ** 3) * eps ** 1.5


def photon_dos_surface(z: float, omega: float,
                       eps_model: DielectricModel) -> float:
    """表面模式态密度 (距离表面 z).
    ρ_surf(ω, z) = 1/(8π) · (ω²_LO - ω²_TO)/ω²_LO · 1/z³

    近场近似 (z << λ):
    源自 1242_ce335805/SPhP/performFreqIntegral.py.
    """
    wLO = eps_model.wLO
    wTO = eps_model.wTO
    if abs(wLO) < 1e-300 or abs(z) < 1e-300:
        return 0.0
    return 1.0 / (8.0 * np.pi) * (wLO ** 2 - wTO ** 2) / wLO ** 2 / z ** 3


def casimir_effective_mass(z_arr: np.ndarray,
                           eps_model: DielectricModel,
                           w_max: float = 100e12) -> np.ndarray:
    """Casimir 有效质量 (源自 1242_ce335805/SPhP/performFreqIntegral.py).

    m_eff(z) = (3/2) · (4/3π) · α_fs · ℏ/(c²m_e) · ∫ dω [ρ(ω,z) - ρ(ω,∞)]

    其中 α_fs ≈ 1/137 为精细结构常数.

    返回各 z 值对应的有效质量.
    """
    c = 3e8
    alpha_fs = 1 / 137.036
    hbar = 1.0546e-34
    m_e = 9.1094e-31

    prefac = 1.5 * (4.0 / (3.0 * np.pi)) * alpha_fs * hbar / (c ** 2 * m_e)

    w_inf = eps_model.surface_mode_frequency()
    mass = np.zeros_like(z_arr)

    n_freq = 100
    w_arr = np.linspace(eps_model.wTO * 1.01, w_max, n_freq)
    dw = w_arr[1] - w_arr[0] if len(w_arr) > 1 else 1e12

    for iz, z in enumerate(z_arr):
        integrand = np.zeros(n_freq)
        for iw, w in enumerate(w_arr):
            rho_z = photon_dos_surface(z, w, eps_model)
            rho_inf = photon_dos_surface(np.inf, w, eps_model)
            integrand[iw] = prefac * (rho_z - rho_inf)
        mass[iz] = np.trapz(integrand, dx=dw)

    return mass


# ---------------------------------------------------------------------------
# 3. 量子光学优化目标
# ---------------------------------------------------------------------------

class QuantumOpticalObjective:
    """量子光学优化目标函数.

    组合多个物理贡献:
      V(x) = V_cavity(x) + V_phonon(x) + V_correlation(x)

    其中 x = [x₁, ..., x_d] 表示腔体参数 (如间距、频率调谐等).

    V_cavity: 腔光子模式贡献
    V_phonon: 声子贡献 (LO/TO 分裂)
    V_correlation: 关联修正
    """

    def __init__(self, dim: int = 5, seed: int = 42):
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self.eps_model = DielectricModel()

        # 随机参数 (定义优化地景的复杂度)
        self.a_coeffs = self.rng.standard_normal(dim)
        self.b_coeffs = self.rng.standard_normal(dim)
        self.center = self.rng.uniform(-1, 1, dim)
        self.scale = self.rng.uniform(0.5, 2.0, dim)

    def cavity_potential(self, x: np.ndarray) -> float:
        """腔光子模式势.
        V_cav(x) = Σ a_i · sin(2π x_i/L) · exp(-x_i²/(2σ²))
        模拟 Fabry-Pérot 腔的模式结构.
        """
        L = 5.0  # 腔长
        sigma = 1.5  # 模式宽度
        V = 0.0
        for i in range(self.dim):
            V += self.a_coeffs[i] * np.sin(2 * np.pi * x[i] / L) * np.exp(-x[i] ** 2 / (2 * sigma ** 2))
        return V

    def phonon_potential(self, x: np.ndarray) -> float:
        """声子势 (LO-TO 分裂).
        V_ph(x) = Σ b_i · (ω_LO² - ω_TO²) / (ω_i²(x) - ω_TO²)
        其中 ω_i(x) = ω_TO + |x_i| · (ω_LO - ω_TO)
        """
        wTO = self.eps_model.wTO
        wLO = self.eps_model.wLO
        dw2 = wLO ** 2 - wTO ** 2
        V = 0.0
        for i in range(self.dim):
            w_i = wTO + abs(x[i]) * (wLO - wTO) / 10.0
            denom = w_i ** 2 - wTO ** 2
            if abs(denom) < 1e-10:
                denom = 1e-10
            V += self.b_coeffs[i] * dw2 / denom
        return V * 1e-25  # 缩放到合理范围

    def correlation_potential(self, x: np.ndarray) -> float:
        """关联势 (多体效应).
        V_corr(x) = -Σ_i<j J_ij · exp(-|x_i-x_j|²/(2ρ²))
        J_ij 为耦合常数 (RKKY 型振荡).
        """
        rho = 1.0
        V = 0.0
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                J_ij = np.sin(2.0 * (i + 1) * (j + 1)) / (1.0 + abs(i - j))
                r2 = np.sum((x[i] - x[j]) ** 2)
                V -= J_ij * np.exp(-r2 / (2 * rho ** 2))
        return V

    def evaluate(self, x: np.ndarray) -> float:
        """总目标函数 V(x) = V_cav + V_ph + V_corr + 正则化."""
        x_shifted = (x - self.center) / self.scale
        V = (self.cavity_potential(x_shifted)
             + self.phonon_potential(x_shifted)
             + self.correlation_potential(x_shifted)
             + 0.1 * np.sum(x ** 2))  # 正则化
        return float(V)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """通过中心差分计算梯度 (数值)."""
        eps = 1e-6
        grad = np.zeros(self.dim)
        f0 = self.evaluate(x)
        for i in range(self.dim):
            x_plus = x.copy()
            x_plus[i] += eps
            grad[i] = (self.evaluate(x_plus) - f0) / eps
        return grad

    def hessian(self, x: np.ndarray) -> np.ndarray:
        """通过中心差分计算 Hessian (数值)."""
        eps = 1e-5
        n = self.dim
        H = np.zeros((n, n))
        f0 = self.evaluate(x)
        for i in range(n):
            for j in range(i, n):
                x_pp = x.copy(); x_pp[i] += eps; x_pp[j] += eps
                x_pm = x.copy(); x_pm[i] += eps; x_pm[j] -= eps
                x_mp = x.copy(); x_mp[i] -= eps; x_mp[j] += eps
                x_mm = x.copy(); x_mm[i] -= eps; x_mm[j] -= eps
                H[i, j] = (self.evaluate(x_pp) - self.evaluate(x_pm)
                           - self.evaluate(x_mp) + self.evaluate(x_mm)) / (4 * eps ** 2)
                H[j, i] = H[i, j]
        return H


# ---------------------------------------------------------------------------
# 4. 单位立方体积分 (源自 234_cube_integrals)
# ---------------------------------------------------------------------------

def monomial_value(m: int, e: np.ndarray, x: np.ndarray) -> float:
    """计算单项式 Π x_i^{e_i}.
    源自 234_cube_integrals/monomial_value.

    输入:
      m: 空间维数
      e: 指数向量 (m,)
      x: 求值点 (m,)
    """
    v = 1.0
    for i in range(m):
        if e[i] != 0:
            if abs(x[i]) < 1e-300 and e[i] < 0:
                return float('inf')
            v *= x[i] ** e[i]
    return v


def cube01_monomial_integral(e: np.ndarray) -> float:
    """单位立方体 [0,1]^d 上的单项式积分.
    源自 234_cube_integrals/cube01_monomial_integral.

    ∫_{[0,1]^d} Π x_i^{e_i} dx = Π ∫_0^1 x_i^{e_i} dx_i = Π 1/(e_i + 1)

    这是精确公式, 用于验证数值积分方法.
    """
    m = len(e)
    result = 1.0
    for i in range(m):
        if e[i] < -1:
            return float('inf')  # 发散
        result *= 1.0 / (e[i] + 1.0)
    return result


def cube01_sample(n: int, dim: int = 3,
                  rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """在 [0,1]^d 中均匀采样 n 个点.
    源自 234_cube_integrals/cube01_sample.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    return rng.uniform(0, 1, (n, dim))


def monte_carlo_integral(f: Callable, dim: int = 3,
                         n_samples: int = 10000,
                         rng: Optional[np.random.Generator] = None) -> Tuple[float, float]:
    """Monte Carlo 积分在 [0,1]^d 上.
    ∫ f(x) dx ≈ (1/N) Σ f(x_i), x_i ~ U([0,1]^d)

    误差: O(1/√N), 与维数无关 (相比确定性方法的 O(1/N^{2/d})).
    返回 (估计值, 标准误差).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    samples = rng.uniform(0, 1, (n_samples, dim))
    values = np.array([f(s) for s in samples])
    mean = np.mean(values)
    std_err = np.std(values) / np.sqrt(n_samples)
    return float(mean), float(std_err)


# ---------------------------------------------------------------------------
# 5. Rosenbrock 族测试函数 (含量子参数)
# ---------------------------------------------------------------------------

def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock 函数 (香蕉函数).
    f(x) = Σ_{i=1}^{d-1} [100(x_{i+1} - x_i²)² + (1 - x_i)²]
    全局极小: x* = (1,1,...,1), f(x*) = 0.
    条件数随维数指数增长, 是优化方法的经典测试.
    """
    n = len(x)
    val = 0.0
    for i in range(n - 1):
        val += 100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1.0 - x[i]) ** 2
    return val


def rosenbrock_grad(x: np.ndarray) -> np.ndarray:
    """Rosenbrock 梯度."""
    n = len(x)
    g = np.zeros(n)
    for i in range(n - 1):
        g[i] += -400.0 * x[i] * (x[i + 1] - x[i] ** 2) - 2.0 * (1.0 - x[i])
        g[i + 1] += 200.0 * (x[i + 1] - x[i] ** 2)
    return g


def rosenbrock_hessian(x: np.ndarray) -> np.ndarray:
    """Rosenbrock Hessian."""
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n - 1):
        H[i, i] += -400.0 * (x[i + 1] - 3.0 * x[i] ** 2) + 2.0
        H[i, i + 1] += -400.0 * x[i]
        H[i + 1, i] += -400.0 * x[i]
        H[i + 1, i + 1] += 200.0
    return H


def rastrigin(x: np.ndarray) -> float:
    """Rastrigin 函数 (多模态).
    f(x) = 10d + Σ [x_i² - 10 cos(2πx_i)]
    全局极小: x* = (0,...,0), f(x*) = 0.
    大量局部极小, 测试全局优化能力.
    """
    d = len(x)
    return 10.0 * d + np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x))


def rastrigin_grad(x: np.ndarray) -> np.ndarray:
    """Rastrigin 梯度."""
    return 2.0 * x + 20.0 * np.pi * np.sin(2.0 * np.pi * x)


def ackley(x: np.ndarray) -> float:
    """Ackley 函数.
    f(x) = -20 exp(-0.2 √(1/d Σ x_i²)) - exp(1/d Σ cos(2πx_i)) + 20 + e
    全局极小: x* = (0,...,0), f(x*) = 0.
    """
    d = len(x)
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(np.cos(2.0 * np.pi * x))
    return (-20.0 * np.exp(-0.2 * np.sqrt(sum1 / d))
            - np.exp(sum2 / d) + 20.0 + np.e)


def ackley_grad(x: np.ndarray) -> np.ndarray:
    """Ackley 梯度."""
    d = len(x)
    norm_x = np.sqrt(np.sum(x ** 2))
    if norm_x < 1e-300:
        return np.zeros(d)
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(np.cos(2.0 * np.pi * x))
    term1 = 20.0 * 0.2 * np.exp(-0.2 * np.sqrt(sum1 / d)) / (d * norm_x) * x
    term2 = 2.0 * np.pi * np.exp(sum2 / d) * np.sin(2.0 * np.pi * x) / d
    return term1 + term2
