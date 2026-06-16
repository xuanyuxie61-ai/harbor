# -*- coding: utf-8 -*-
"""
wave_particle_resonance.py
===========================
波-粒子共振动力学与混沌轨道模块

融合种子项目:
  517_henon_orbit    — Henon 映射轨道动力学
  321_dueling_idiots — Monte Carlo 模拟 / 二项式系数 / Brownian 运动

物理背景
--------
空间等离子体中的波-粒子相互作用通过共振条件实现能量交换:

1. Landau 共振 (无磁场):
     ω - k·v = 0
   粒子速度等于波相速度时，粒子持续感受波的加速/减速力。

2. Cyclotron 共振 (有磁场 B₀):
     ω - k·v_∥ = n·Ω_c
   其中 Ω_c = qB₀/m 为回旋频率，n 为谐波数。

3. 反常多普勒共振:
     ω - k·v_∥ = -|n|·Ω_c

Quasilinear 扩散理论
--------------------
波-粒子共振导致速度空间扩散:
  ∂f/∂t = ∂/∂v (D(v) ∂f/∂v)
其中扩散系数:
  D(v) = (π e² E₀² / m²) δ_D(ω - kv)  (单色波)
       = Σ_k |E_k|² δ_D(ω_k - kv)       (宽谱)

轨道混沌
--------
当多个波叠加时，相邻共振区重叠导致粒子轨道混沌
(Chirikov 判据): s = Δv_res / Δv_sep > 1
"""

import numpy as np
import math
from typing import Tuple, Optional


# ===== 共振条件 ============================================================

def landau_resonance_condition(omega: complex, k: float,
                                v: np.ndarray,
                                v_th: float) -> np.ndarray:
    """
    Landau 共振函数: R(v) = δ(ω - kv) 的高斯近似

    .. math::
        R(v) = \\frac{1}{\\sqrt{\\pi} \\Delta v}
               \\exp\\left(-\\frac{(v - v_\\phi)^2}{\\Delta v^2}\\right)

    其中 v_φ = Re(ω)/k 为相速度，Δv 为共振宽度。

    Parameters
    ----------
    omega : complex
        波频率 (实部为振荡频率, 虚部为阻尼率).
    k : float
        波数.
    v : np.ndarray
        速度网格.
    v_th : float
        热速度 (控制共振宽度).

    Returns
    -------
    np.ndarray
        共振函数值 R(v).
    """
    if abs(k) < 1.0e-300:
        return np.zeros_like(v)

    v_phase = omega.real / k
    gamma = abs(omega.imag) if omega.imag != 0 else 1.0e-10
    delta_v = max(gamma / abs(k), v_th * 0.01)

    x = (v - v_phase) / delta_v
    return np.exp(-x * x) / (math.sqrt(math.pi) * delta_v)


def cyclotron_resonance(omega: float, k_parallel: float,
                         v_parallel: np.ndarray,
                         omega_c: float, n: int = 1,
                         v_th: float = 0.1) -> np.ndarray:
    """
    回旋共振函数

    .. math::
        R_n(v_\\parallel) = \\delta(\\omega - k_\\parallel v_\\parallel - n\\Omega_c)

    n = ±1: 基频共振 (主加热机制)
    n = ±2, ±3, ...: 谐波共振 (较弱)

    Parameters
    ----------
    omega : float
        波频率.
    k_parallel : float
        平行波数.
    v_parallel : np.ndarray
        平行速度网格.
    omega_c : float
        回旋频率 |qB₀/m|.
    n : int
        谐波数.
    v_th : float
        热速度.
    """
    if abs(k_parallel) < 1.0e-300:
        return np.zeros_like(v_parallel)

    v_res = (omega - n * omega_c) / k_parallel
    delta_v = max(v_th * 0.05, 1.0e-10)
    x = (v_parallel - v_res) / delta_v
    return np.exp(-x * x) / (math.sqrt(math.pi) * delta_v)


def cyclotron_frequency(b_field: float, charge: float,
                         mass: float) -> float:
    """
    回旋频率 Ω_c = |q|B₀/m

    Parameters
    ----------
    b_field : float
        背景磁场强度 [T].
    charge : float
        粒子电荷 [C].
    mass : float
        粒子质量 [kg].

    Returns
    -------
    float
        回旋频率 [rad/s].
    """
    if abs(mass) < 1.0e-300:
        raise ValueError("cyclotron_frequency: mass must be nonzero")
    return abs(charge) * abs(b_field) / mass


# ===== Maxwellian 分布与导数 ==============================================

def maxwellian_1d(v: np.ndarray, n0: float, v_th: float) -> np.ndarray:
    """
    1D Maxwellian 速度分布

    .. math::
        f_0(v) = \\frac{n_0}{\\sqrt{2\\pi} v_{th}}
                 \\exp\\left(-\\frac{v^2}{2 v_{th}^2}\\right)
    """
    if v_th < 1.0e-300:
        raise ValueError("maxwellian_1d: v_th must be positive")
    return n0 / (math.sqrt(2.0 * math.pi) * v_th) * \
           np.exp(-0.5 * (v / v_th) ** 2)


def maxwellian_derivative(v: np.ndarray, n0: float,
                           v_th: float) -> np.ndarray:
    """
    Maxwellian 对速度的导数 (决定 Landau 阻尼率)

    .. math::
        \\frac{\\partial f_0}{\\partial v}
        = -\\frac{v}{v_{th}^2} f_0(v)
    """
    f0 = maxwellian_1d(v, n0, v_th)
    return -(v / (v_th * v_th)) * f0


# ===== Quasilinear 扩散 ====================================================

class QuasilinearDiffusion:
    """
    Quasilinear 扩散算子

    扩散方程:
      ∂f/∂t = ∂/∂v [D(v) ∂f/∂v]

    通量形式:
      Γ(v) = -D(v) ∂f/∂v

    扩散系数 (单色波):
      D(v) = (π q² E₀² / m²) · R(v)

    其中 R(v) 为共振函数。
    """

    def __init__(self, v_grid: np.ndarray, wave_number: float,
                 wave_amplitude: float, particle_mass: float = 1.0,
                 charge: float = 1.0):
        self.v = np.asarray(v_grid, dtype=np.float64)
        self.nv = len(self.v)
        self.dv = self.v[1] - self.v[0] if self.nv > 1 else 1.0
        self.k = wave_number
        self.E0 = wave_amplitude
        self.m = particle_mass
        self.q = charge

    def diffusion_coefficient(self, omega: complex) -> np.ndarray:
        """
        计算扩散系数 D(v)

        .. math::
            D(v) = \\frac{\\pi q^2 E_0^2}{m^2} R(v)
        """
        R = landau_resonance_condition(omega, self.k, self.v,
                                        v_th=abs(self.v[-1] - self.v[0]) / 8.0)
        coeff = math.pi * self.q ** 2 * self.E0 ** 2 / (self.m ** 2)
        return coeff * R

    def apply_operator(self, f: np.ndarray,
                        omega: complex) -> np.ndarray:
        """
        计算 quasilinear 扩散右端: ∂/∂v [D(v) ∂f/∂v]

        使用中心差分:
          [∂/∂v (D ∂f/∂v)]_i ≈ (D_{i+1/2}(f_{i+1}-f_i)
                                 - D_{i-1/2}(f_i-f_{i-1})) / Δv²
        """
        f = np.asarray(f, dtype=np.float64)
        D = self.diffusion_coefficient(omega)

        # 界面扩散系数 (算术平均)
        D_intf = 0.5 * (D[:-1] + D[1:])

        # 通量
        flux = np.zeros(self.nv - 1, dtype=np.float64)
        for j in range(self.nv - 1):
            flux[j] = -D_intf[j] * (f[j + 1] - f[j]) / self.dv

        # 散度
        result = np.zeros(self.nv, dtype=np.float64)
        for i in range(1, self.nv - 1):
            result[i] = (flux[i] - flux[i - 1]) / self.dv

        return result

    def resonance_energy_transfer(self, f: np.ndarray,
                                    omega: complex) -> float:
        """
        计算波-粒子能量交换率

        .. math::
            P = \\int v \\cdot \\frac{\\partial f}{\\partial t}\\Big|_{QL} dv
        """
        dfdt = self.apply_operator(f, omega)
        return float(np.sum(self.v * dfdt) * self.dv)

    def flattening_timescale(self, omega: complex) -> float:
        """
        估计共振区分布函数平坦化的特征时间

        τ_flat ≈ Δv² / D(v_res)
        """
        D = self.diffusion_coefficient(omega)
        D_max = np.max(D)
        if D_max < 1.0e-300:
            return float('inf')
        return self.dv ** 2 / D_max


# ===== Chirikov 重叠参数 ===================================================

def chirikov_overlap_parameter(
    wave_numbers: np.ndarray,
    amplitudes: np.ndarray,
    particle_mass: float = 1.0
) -> float:
    """
    Chirikov 重叠参数 s

    .. math::
        s = \\frac{\\Delta v_{res,1} + \\Delta v_{res,2}}{\\Delta v_{sep}}

    s > 1: 共振区重叠 → 全局混沌
    s < 1: 共振区分离 → 规则运动 + 局部混沌

    重叠判据决定了 quasilinear 理论的适用性:
      s >> 1: quasilinear 扩散有效
      s ~ 1: 需要考虑非线性轨道效应
      s << 1: 单波近似有效
    """
    if len(wave_numbers) < 2:
        return 0.0

    # 各波的捕获宽度
    trapping_widths = np.zeros(len(wave_numbers))
    for i in range(len(wave_numbers)):
        k = wave_numbers[i]
        E = amplitudes[i]
        if abs(k) > 1.0e-300:
            trapping_widths[i] = 2.0 * np.sqrt(
                abs(E / (k * particle_mass + 1e-300)))

    # 共振间隔
    v_phases = 1.0 / (wave_numbers + 1e-300)
    dv_sep = np.min(np.abs(np.diff(v_phases))) if len(v_phases) > 1 else 1.0

    # 相邻波的重叠
    max_overlap = 0.0
    for i in range(len(wave_numbers) - 1):
        s = (trapping_widths[i] + trapping_widths[i + 1]) / \
            max(dv_sep, 1.0e-300)
        max_overlap = max(max_overlap, s)

    return max_overlap


# ===== Henon 型映射 (等离子体波场中轨道) =================================

class PlasmaHenonMap:
    """
    Henon 型辛映射 — 描述粒子在周期性波场中的轨道

    .. math::
        x_{n+1} = x_n \\cos\\alpha - (p_n - x_n^2) \\sin\\alpha
        p_{n+1} = x_n \\sin\\alpha + (p_n - x_n^2) \\cos\\alpha

    参数 α 控制非线性强度:
      α ≈ 0: 近可积 (KAM 环面为主)
      α ~ π/4: 混合相空间 (混沌海 + 岛链)
      α ~ π/2: 强混沌

    在等离子体物理中，此映射描述粒子在驻波场中的逐周期演化,
    其中 (x, p) 对应 (相位, 动量)。
    """

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.cos_a = math.cos(alpha)
        self.sin_a = math.sin(alpha)

    def step(self, x: float, p: float) -> Tuple[float, float]:
        """执行一步映射."""
        x_new = x * self.cos_a - (p - x * x) * self.sin_a
        p_new = x * self.sin_a + (p - x * x) * self.cos_a
        return x_new, p_new

    def orbit(self, x0: float, p0: float, n_steps: int,
              escape_radius: float = 10.0) -> np.ndarray:
        """
        计算轨道 (x, p) 序列

        Returns
        -------
        np.ndarray, shape (n_valid, 2)
            轨道点 (x, p). 逃逸轨道被截断。
        """
        points = []
        x, p = x0, p0
        for _ in range(n_steps):
            if abs(x) > escape_radius or abs(p) > escape_radius:
                break
            points.append([x, p])
            x, p = self.step(x, p)
        return np.array(points) if points else np.zeros((0, 2))

    def is_bounded(self, x0: float, p0: float,
                    n_check: int = 1000) -> bool:
        """判断轨道是否有界 (不逃逸)."""
        x, p = x0, p0
        for _ in range(n_check):
            if abs(x) > 100.0 or abs(p) > 100.0:
                return False
            x, p = self.step(x, p)
        return True

    def lyapunov_exponent(self, x0: float, p0: float,
                           n_steps: int = 5000) -> float:
        """
        最大 Lyapunov 指数 — 表征轨道混沌程度

        λ > 0: 混沌 (相邻轨道指数分离)
        λ = 0: 规则运动
        λ < 0: 收敛到吸引子 (耗散系统)

        通过跟踪无穷小扰动向量的增长率计算:
          λ = lim_{N→∞} (1/N) Σ ln |δ_n| / |δ_0|
        """
        x, p = x0, p0
        dx, dp = 1.0e-8, 0.0
        total_log = 0.0
        n_valid = 0

        for _ in range(n_steps):
            if abs(x) > 100.0 or abs(p) > 100.0:
                break

            # Jacobian of the Henon map
            # dx_new = dx·cos_a - (dp - 2x·dx)·sin_a
            # dp_new = dx·sin_a + (dp - 2x·dx)·cos_a
            dx_new = dx * self.cos_a - (dp - 2.0 * x * dx) * self.sin_a
            dp_new = dx * self.sin_a + (dp - 2.0 * x * dx) * self.cos_a

            dx, dp = dx_new, dp_new
            norm = math.sqrt(dx * dx + dp * dp + 1e-300)
            total_log += math.log(norm)
            dx /= norm
            dp /= norm

            x, p = self.step(x, p)
            n_valid += 1

        return total_log / max(n_valid, 1)


# ===== Brownian 型碰撞算子 =================================================

class BrownianCollisionOperator:
    """
    Brownian 型速度空间扩散 (模拟粒子碰撞)

    .. math::
        \\frac{\\partial f}{\\partial t}
        = \\nu_{coll} \\frac{\\partial^2 f}{\\partial v^2}
        + \\nu_{coll} \\frac{\\partial}{\\partial v}(v f)

    第一项: 速度空间扩散 (随机碰撞)
    第二项: 摩擦拖曳 (热化)

    平衡分布: Maxwellian f_eq ∝ exp(-v²/(2v_th²))
    """

    def __init__(self, v_grid: np.ndarray, nu_coll: float,
                 v_th: float = 1.0):
        self.v = np.asarray(v_grid, dtype=np.float64)
        self.nv = len(self.v)
        self.dv = self.v[1] - self.v[0] if self.nv > 1 else 1.0
        self.nu = nu_coll
        self.v_th = v_th

    def apply(self, f: np.ndarray) -> np.ndarray:
        """
        计算碰撞算子: C[f] = ν ∂²f/∂v² + ν ∂(vf)/∂v
        """
        f = np.asarray(f, dtype=np.float64)

        # 二阶导数 (中心差分)
        d2f = np.zeros(self.nv, dtype=np.float64)
        for i in range(1, self.nv - 1):
            d2f[i] = (f[i - 1] - 2.0 * f[i] + f[i + 1]) / (self.dv ** 2)

        # 摩擦项: ∂(v·f)/∂v = f + v·∂f/∂v
        df = np.zeros(self.nv, dtype=np.float64)
        for i in range(1, self.nv - 1):
            df[i] = (f[i + 1] - f[i - 1]) / (2.0 * self.dv)
        friction = f + self.v * df

        return self.nu * (d2f + friction / (self.v_th ** 2))

    def equilibrium(self, n0: float = 1.0) -> np.ndarray:
        """平衡态 Maxwellian 分布."""
        return maxwellian_1d(self.v, n0, self.v_th)
