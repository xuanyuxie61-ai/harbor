# -*- coding: utf-8 -*-
"""
linear_buckling.py
圆柱壳线性屈曲特征值分析

融合种子项目:
  - 081_besselzero: Bessel 函数零点计算

科学背景:
  圆柱壳在轴向压力 N_x 作用下的经典线性屈曲方程:
    D ∇⁴w + (1/R) N_θ + N_x ∂²w/∂x² = 0
  其中 D = Et³/(12(1-ν²)) 为弯曲刚度。

  对于简支边界条件，假设屈曲模态:
    w(x,θ) = W sin(mπx/L) cos(nθ)
  代入控制方程可得特征方程:
    λ = (D/R²t) * [ (m²π²R²/L² + n²)² / (m²π²R²/L²) ] + [ Et / (D(m²π²R²/L² + n²)²) ]

  对于长壳 (Z > 1000)，最小屈曲载荷近似为:
    N_x,cr ≈ 0.605 Et² / (R √(1-ν²))

  Bessel 函数零点用于验证环向波数 n 的解析解:
    圆柱壳环向谐波对应 Bessel 方程 J_n'(x) = 0 或 J_n(x) = 0 的零点。
"""

import numpy as np
from scipy.special import jv, yv, jvp
from scipy.optimize import newton


def bessel_zero_halley(n: float, k: int, kind: int = 1, tol: float = 1e-14, max_iter: int = 100) -> float:
    """
    使用 Halley 迭代计算 Bessel 函数零点 (基于 081_besselzero)

    Halley 迭代公式:
      x_{k+1} = x_k - f(x_k) / [ f'(x_k) - f(x_k)f''(x_k) / (2f'(x_k)) ]

    对于 Bessel 函数 J_n(x):
      f(x)  = J_n(x)
      f'(x) = J_n'(x) = 0.5(J_{n-1}(x) - J_{n+1}(x))
      f''(x)= 0.25(J_{n-2}(x) - 2J_n(x) + J_{n+2}(x))

    Parameters
    ----------
    n : float
        Bessel 函数阶数
    k : int
        第 k 个正零点 (k >= 1)
    kind : int
        1 为 J_n (第一类), 2 为 Y_n (第二类)
    tol : float
        收敛容差
    max_iter : int
        最大迭代次数

    Returns
    -------
    x : float
        零点近似值
    """
    if kind not in (1, 2):
        raise ValueError("kind 必须为 1 或 2")
    # 初始猜测 (基于渐近展开)
    if k == 1:
        x0 = 2.0 * np.abs(n) + 1.857 * np.abs(n) ** 0.333 + 1.0
    else:
        # 高阶零点近似等距分布
        x0 = (k + 0.5 * np.abs(n) - 0.25) * np.pi
    x = x0
    for _ in range(max_iter):
        if kind == 1:
            f = jv(n, x)
            fp = jvp(n, x, 1)
            fpp = jvp(n, x, 2)
        else:
            f = yv(n, x)
            fp = jvp(n, x, 1)  # yv 的导数可用递推
            # yv' = 0.5*(Y_{n-1} - Y_{n+1})
            fpp = jvp(n, x, 2)
        denom = 2.0 * fp * fp - f * fpp
        if abs(denom) < 1e-20:
            break
        dx = 2.0 * f * fp / denom
        x_new = x - dx
        if abs(dx) < tol:
            return float(x_new)
        x = x_new
    return float(x)


def bessel_zeros_vector(n: float, k_max: int, kind: int = 1) -> np.ndarray:
    """
    计算前 k_max 个正零点
    """
    zeros = []
    for k in range(1, k_max + 1):
        z = bessel_zero_halley(n, k, kind)
        zeros.append(z)
    return np.array(zeros)


class LinearBucklingAnalyzer:
    """
    圆柱壳线性屈曲分析器
    """

    def __init__(self, geometry, material):
        self.geom = geometry
        self.mat = material
        self.D = self.mat.bending_rigidity(self.geom.t)
        self.C = self.mat.extensional_rigidity(self.geom.t)

    def analytical_buckling_load(self) -> float:
        """
        经典轴向屈曲临界载荷 (简支边界)

        N_x,cr = (1/√(3(1-ν²))) * (Et²/R)
              ≈ 0.605 Et²/R   (当 ν = 0.3)
        """
        E, nu = self.mat.E, self.mat.nu
        t, R = self.geom.t, self.geom.R
        Ncr = E * t ** 2 / (R * np.sqrt(3.0 * (1.0 - nu ** 2)))
        return float(Ncr)

    def buckling_modes_discrete(self, m_max: int = 10, n_max: int = 10) -> tuple:
        """
        离散搜索最小屈曲载荷对应的轴向半波数 m 和环向波数 n

        对于模态 w = W sin(mπx/L) cos(nθ):
          α = mπR/L
          β = n
          λ = (D/(R²t)) * [(α² + β²)² / α²] + (Et/D) * [α² / (α² + β²)²]

        Returns
        -------
        N_min, m_opt, n_opt, mode_table
        """
        R, L, t = self.geom.R, self.geom.L, self.geom.t
        D = self.D
        E = self.mat.E
        nu = self.mat.nu
        N_min = float('inf')
        m_opt = 1
        n_opt = 0
        modes = []
        for m in range(1, m_max + 1):
            alpha = m * np.pi * R / L
            for n in range(0, n_max + 1):
                beta = float(n)
                if alpha == 0:
                    continue
                term1 = (alpha ** 2 + beta ** 2) ** 2 / alpha ** 2
                term2 = alpha ** 2 / (alpha ** 2 + beta ** 2) ** 2
                # 修正的 Donnell 公式
                Nx = (D / (R ** 2 * t)) * term1 + (E * t / (1.0 - nu ** 2)) * term2
                modes.append((m, n, Nx))
                if Nx < N_min:
                    N_min = Nx
                    m_opt = m
                    n_opt = n
        return float(N_min), m_opt, n_opt, modes

    def bessel_verification(self, n_circumferential: int, n_zeros: int = 5) -> np.ndarray:
        """
        使用 Bessel 零点验证环向波数

        圆柱壳环向谐波满足 Bessel 方程特征条件:
          J_n'(kR) = 0  (自由边界)
          J_n(kR) = 0   (固支边界)

        临界波数 k_cr 与屈曲载荷关系:
          N_x,cr = D k_cr² + Et/(R² k_cr²)

        Returns
        -------
        zeros : ndarray
            Bessel 函数前 n_zeros 个零点
        """
        n = float(n_circumferential)
        zeros = bessel_zeros_vector(n, n_zeros, kind=1)
        return zeros

    def imperfection_sensitivity_koiter(self, imperfection_amplitude: float,
                                       imperfection_mode: int) -> float:
        """
        Koiter 初始后屈曲理论: 非线性屈曲载荷降低因子

        对于圆柱壳轴向压缩，经典 Koiter 公式:
          λ_s / λ_c = 1 - a * (δ/t)   (非对称屈曲)
        其中 a 为与模态相关的系数，通常 a ≈ 1.5√(3(1-ν²)) ≈ 2.4

        Parameters
        ----------
        imperfection_amplitude : float
            缺陷幅值 δ
        imperfection_mode : int
            缺陷波数

        Returns
        -------
        reduction_factor : float
            屈曲载荷降低因子
        """
        nu = self.mat.nu
        delta = imperfection_amplitude
        t = self.geom.t
        # Koiter 经典结果
        a = 1.5 * np.sqrt(3.0 * (1.0 - nu ** 2))
        ratio = max(0.0, 1.0 - a * (delta / t))
        return float(ratio)
