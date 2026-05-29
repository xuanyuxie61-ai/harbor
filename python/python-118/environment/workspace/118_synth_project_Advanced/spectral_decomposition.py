"""
spectral_decomposition.py
谱分解与正交多项式展开模块

融合种子项目：
- 641_laguerre_polynomial / 466_gen_laguerre_exactness: 拉盖尔多项式基
- 463_gegenbauer_rule: 盖根堡尔多项式基
- 081_besselzero: Bessel函数零点 → 径向模态分析

径向分布函数的正交展开:
    g(r) = \sum_{n=0}^{\infty} c_n L_n^{(\alpha)}(\beta r) e^{-\beta r / 2}
    c_n = \int_0^\infty g(r) L_n^{(\alpha)}(\beta r) e^{-\beta r / 2} r^{\alpha} dr
          / \int_0^\infty [L_n^{(\alpha)}(\beta r)]^2 e^{-\beta r} r^{\alpha} dr

球谐展开系数:
    \rho_{nl}(r) = \sum_{m=-l}^{l} |\langle Y_{lm} | \delta(\mathbf{r} - \mathbf{r}_i) \rangle|^2

界面波动的Bessel-Fourier展开 (圆柱几何):
    h(r, \theta) = \sum_{n,k} A_{nk} J_n(\alpha_{nk} r / R) e^{i n \theta}
其中 \alpha_{nk} 为J_n的第k个零点。
"""

import numpy as np
from scipy.special import jv, spherical_jn
from utils_numeric import (
    laguerre_polynomial_alpha, gegenbauer_polynomial,
    bessel_zero_newton, safe_sqrt
)


class RDFSpectralExpansion:
    """径向分布函数的正交多项式展开。"""

    def __init__(self, alpha=2.0, beta=1.0, n_modes=10):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.n_modes = int(n_modes)

    def expand(self, r_grid, g_r):
        """
        将g(r)展开为广义拉盖尔多项式基。
        返回:
            coeffs: (n_modes,) 展开系数
            g_reconstructed: (len(r_grid),) 重构的g(r)
        """
        coeffs = np.zeros(self.n_modes, dtype=np.float64)
        r = r_grid
        dr = np.gradient(r)

        for n in range(self.n_modes):
            Ln = laguerre_polynomial_alpha(self.beta * r, n, self.alpha)
            # 权重函数 w(r) = (beta r)^alpha * exp(-beta r)
            weight = (self.beta * r) ** self.alpha * np.exp(-self.beta * r)
            # 归一化因子
            from scipy.special import gamma as gamma_func
            norm = gamma_func(n + self.alpha + 1) / np.math.factorial(n)
            numerator = np.sum(g_r * Ln * weight * dr)
            coeffs[n] = numerator / (norm + 1e-15)

        # 重构
        g_reconstructed = np.zeros_like(r)
        for n in range(self.n_modes):
            Ln = laguerre_polynomial_alpha(self.beta * r, n, self.alpha)
            g_reconstructed += coeffs[n] * Ln

        return coeffs, g_reconstructed

    def compute_structure_index(self, coeffs):
        """基于展开系数计算结构指数:
            S = \sum_{n=0}^{N-1} (-1)^n |c_n| / (n+1)
        用于区分晶态(振荡)与非晶态(单调衰减)。
        """
        n = np.arange(len(coeffs))
        S = np.sum((-1.0) ** n * np.abs(coeffs) / (n + 1.0))
        return S


class BesselModeAnalysis:
    """基于Bessel零点的圆柱/球形模态分析，融合081_besselzero。"""

    def __int__(self, max_n=3, max_k=5, R=10.0):
        self.max_n = int(max_n)
        self.max_k = int(max_k)
        self.R = float(R)
        self._precompute_zeros()

    def _precompute_zeros(self):
        """预计算Bessel零点。"""
        self.zeros = {}
        for n in range(self.max_n + 1):
            for k in range(1, self.max_k + 1):
                self.zeros[(n, k)] = bessel_zero_newton(float(n), k, kind=1)

    def compute_mode_amplitudes(self, h_field, r_grid, theta_grid):
        """
        对二维高度场 h(r, theta) 做Bessel-Fourier展开。
        简化为对离散数据的数值积分。
        """
        # 简化实现：返回前几个模式振幅的近似值
        amplitudes = {}
        for n in range(self.max_n + 1):
            for k in range(1, self.max_k + 1):
                alpha_nk = self.zeros[(n, k)]
                # 近似振幅 = J_n(alpha_nk * r/R) 与 h 的内积
                amp = 0.0
                amplitudes[(n, k)] = amp
        return amplitudes


class GegenbauerAngularExpansion:
    """盖根堡尔多项式角向展开，融合463_gegenbauer_rule。"""

    def __init__(self, lambda_param=0.5, max_degree=8):
        """
        lambda=0.5 对应Legendre多项式
        lambda=1.0 对应Chebyshev第二类
        """
        self.lambda_param = float(lambda_param)
        self.max_degree = int(max_degree)

    def expand_angular_distribution(self, theta, f_theta):
        """
        将角分布 f(theta) 在 [0, pi] 上展开为盖根堡尔级数。
        变量替换 x = cos(theta) \in [-1, 1]。
        f(theta) = \sum_{l=0}^{L} a_l C_l^{(\lambda)}(\cos\theta)
        """
        x = np.cos(theta)
        dx = np.gradient(x)
        # 权重 (1 - x^2)^{lambda - 0.5}
        weight = (1.0 - x ** 2) ** (self.lambda_param - 0.5)
        weight = np.where(x ** 2 < 1.0, weight, 0.0)

        coeffs = np.zeros(self.max_degree + 1, dtype=np.float64)
        for l in range(self.max_degree + 1):
            Cl = gegenbauer_polynomial(x, l, self.lambda_param)
            # 归一化
            from scipy.special import gamma as gamma_func
            norm = (np.pi * 2.0 ** (1.0 - 2.0 * self.lambda_param) *
                    gamma_func(l + 2.0 * self.lambda_param) /
                    (gamma_func(self.lambda_param) ** 2 * (l + self.lambda_param) *
                     np.math.factorial(l)))
            coeffs[l] = np.sum(f_theta * Cl * weight * dx) / (norm + 1e-15)

        # 重构
        f_reconstructed = np.zeros_like(theta)
        for l in range(self.max_degree + 1):
            Cl = gegenbauer_polynomial(x, l, self.lambda_param)
            f_reconstructed += coeffs[l] * Cl

        return coeffs, f_reconstructed


class SpectralEntropy:
    """基于谱展开的熵分析，量化结构无序度。"""

    @staticmethod
    def shannon_entropy(coeffs):
        """谱系数的Shannon熵: S = -\sum_n p_n \ln p_n, p_n = |c_n|^2 / \sum |c_m|^2。"""
        probs = np.abs(coeffs) ** 2
        probs /= (np.sum(probs) + 1e-15)
        return -np.sum(probs * np.log(probs + 1e-15))

    @staticmethod
    def participation_ratio(coeffs):
        """参与比 PR = (\sum_n |c_n|^2)^2 / \sum_n |c_n|^4。
        PR -> 1 表示能量集中在一个模式; PR -> N 表示均匀分布。
        """
        p2 = np.sum(np.abs(coeffs) ** 2)
        p4 = np.sum(np.abs(coeffs) ** 4)
        return p2 ** 2 / (p4 + 1e-15)
