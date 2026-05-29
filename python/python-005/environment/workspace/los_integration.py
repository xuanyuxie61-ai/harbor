# -*- coding: utf-8 -*-
"""
los_integration.py
线-of-sight（LoS）快速数值积分模块

核心物理：
    CMB 各向异性的功率谱由线-of-sight 积分给出：
        C_l^{XY} = 4π ∫_0^∞ (dk/k) P_R(k) T_l^X(k) T_l^Y(k)
    其中 X,Y ∈ {T, E, B}。
    本模块实现四种高精度快速求积公式，用于上述径向积分：
        1. Gauss-Legendre 求积（最优多项式精度）
        2. Clenshaw-Curtis 求积（Chebyshev 节点，FFT 加速权重）
        3. Fejér 第一型求积（开区间，排除端点）
        4. Fejér 第二型求积（等距余弦节点）

融合种子项目 939_quad_fast_rule（Clenshaw-Curtis, Fejér, Gauss-Legendre 快速求积）。
"""

import numpy as np
from typing import Tuple
from utils import clip_to_unit, ensure_positive


# ---------------------------------------------------------------------------
# 辅助：Jacobi 矩阵特征值求 Gauss-Legendre 节点与权重（Golub-Welsch 算法）
# ---------------------------------------------------------------------------
def gauss_legendre_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    通过三对角 Jacobi 矩阵特征值计算 n 点 Gauss-Legendre 节点 x_i 和权重 w_i。
    矩阵 J 的元素：
        J_{i,i} = 0
        J_{i,i+1} = J_{i+1,i} = i / sqrt(4 i^2 - 1)
    特征值给出节点，(v_1)^2 给出权重。
    """
    if n < 1:
        raise ValueError("n 必须 ≥ 1")
    if n == 1:
        return np.array([0.0]), np.array([2.0])
    i = np.arange(1.0, n, dtype=float)
    beta = i / np.sqrt(4.0 * i ** 2 - 1.0)
    J = np.diag(beta, 1) + np.diag(beta, -1)
    eigvals, eigvecs = np.linalg.eigh(J)
    x = eigvals
    w = 2.0 * eigvecs[0, :] ** 2
    return x, w


# ---------------------------------------------------------------------------
# 快速 Fejér 权重（Waldvogel 2003，通过 IFFT 计算）
# ---------------------------------------------------------------------------
def fejer1_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fejér 第一型 n 点求积：开区间，节点在 cos[(2i-1)π/(2n)]。
    权重通过 IFFT 在 O(n log n) 内计算。
    """
    if n < 1:
        raise ValueError("n 必须 ≥ 1")
    i = np.arange(1, n + 1)
    x = np.cos((2.0 * i - 1.0) * np.pi / (2.0 * n))
    # Waldvogel 算法
    N = 2 * n
    v = np.zeros(N)
    v[0] = 2.0
    # 交替符号序列
    idx = np.arange(1, N, 2)
    v[idx] = 2.0 / (idx * (idx + 2))
    # IFFT
    v_tilde = np.fft.ifft(v).real
    w = 2.0 * v_tilde[:n]
    return x, w


def fejer2_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fejér 第二型 n 点求积：节点在 cos(iπ/n)，排除端点。
    """
    if n < 1:
        raise ValueError("n 必须 ≥ 1")
    i = np.arange(1, n + 1)
    x = np.cos(i * np.pi / (n + 1.0))
    # 权重公式
    k = np.arange(1, n + 1)
    w = np.zeros(n)
    for m in range(n):
        s = 0.0
        for j in range(1, (n + 1) // 2 + 1):
            s += np.sin((2.0 * j - 1.0) * k[m] * np.pi / (n + 1.0)) / (2.0 * j - 1.0)
        w[m] = 4.0 * np.sin(k[m] * np.pi / (n + 1.0)) * s / (n + 1.0)
    return x, w


# ---------------------------------------------------------------------------
# Clenshaw-Curtis 权重（直接余弦求和）
# ---------------------------------------------------------------------------
def clenshaw_curtis_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clenshaw-Curtis n+1 点求积（包含端点），节点在 cos(iπ/n)。
    权重通过离散余弦变换计算。
    """
    if n < 1:
        raise ValueError("n 必须 ≥ 1")
    i = np.arange(n + 1)
    x = np.cos(i * np.pi / n)
    # 权重
    w = np.zeros(n + 1)
    theta = i * np.pi / n
    # 使用直接公式
    v = np.ones(n - 1)
    if n % 2 == 0:
        w[0] = w[n] = 1.0 / (n ** 2 - 1.0)
        for k in range(1, n):
            s = 0.0
            for j in range(n // 2):
                s += np.cos(2.0 * j * theta[k]) / (4.0 * j ** 2 - 1.0)
            w[k] = 2.0 * (1.0 - s) / (n - 1.0)
            if k == 0 or k == n:
                w[k] *= 0.5
    else:
        w[0] = w[n] = 1.0 / (n ** 2)
        for k in range(1, n):
            s = 0.0
            for j in range((n - 1) // 2 + 1):
                s += np.cos(2.0 * j * theta[k]) / (4.0 * j ** 2 - 1.0)
            w[k] = 2.0 * s / n
    # 归一化修正
    w[0] *= 0.5
    w[n] *= 0.5
    return x, w


# ---------------------------------------------------------------------------
# 统一积分接口
# ---------------------------------------------------------------------------
class FastQuadrature:
    """快速高精度数值积分器，支持四种经典求积公式。"""

    RULES = ["gauss_legendre", "clenshaw_curtis", "fejer1", "fejer2"]

    def __init__(self, rule: str = "gauss_legendre", n: int = 64):
        """
        Parameters
        ----------
        rule : str
            求积规则名称。
        n : int
            求积点数。
        """
        if rule not in self.RULES:
            raise ValueError(f"不支持的求积规则: {rule}")
        self.rule = rule
        self.n = n
        self._precompute()

    def _precompute(self):
        """预计算节点和权重。"""
        if self.rule == "gauss_legendre":
            self.x_ref, self.w_ref = gauss_legendre_nodes_weights(self.n)
        elif self.rule == "clenshaw_curtis":
            self.x_ref, self.w_ref = clenshaw_curtis_nodes_weights(self.n)
        elif self.rule == "fejer1":
            self.x_ref, self.w_ref = fejer1_nodes_weights(self.n)
        elif self.rule == "fejer2":
            self.x_ref, self.w_ref = fejer2_nodes_weights(self.n)

    def integrate(self, f: callable, a: float, b: float) -> float:
        """
        在区间 [a, b] 上计算 ∫ f(x) dx。
        通过变量替换 x = 0.5(b-a)t + 0.5(a+b) 映射到 [-1,1]。
        """
        if b <= a:
            raise ValueError("积分上限必须大于下限")
        t = self.x_ref
        x = 0.5 * (b - a) * t + 0.5 * (a + b)
        fx = np.array([f(xi) for xi in x])
        return 0.5 * (b - a) * np.dot(self.w_ref, fx)


# ---------------------------------------------------------------------------
# 线-of-sight 积分专用封装
# ---------------------------------------------------------------------------
def los_integral_power_spectrum(l: int,
                                 transfer_l: callable,
                                 primordial_power: callable,
                                 k_min: float, k_max: float,
                                 n_quad: int = 128,
                                 rule: str = "gauss_legendre") -> float:
    """
    计算单极矩 C_l：
        C_l = 4π ∫_{k_min}^{k_max} (dk/k) P_R(k) [T_l(k)]^2
    其中 P_R(k) 为原初功率谱（近似尺度不变 P_R = A_s (k/k_p)^{n_s-1}）。
    """
    # TODO: 请补全 CMB 角功率谱线-of-sight 积分的实现
    # 提示：需要构造被积函数并调用 FastQuadrature 进行数值积分
    raise NotImplementedError("Hole_3: 请补全 los_integral_power_spectrum 的实现")


def compute_sachs_wolfe_integral(l: int, k: float, eta_grid: np.ndarray,
                                  Delta0_grid: np.ndarray,
                                  Phi_grid: np.ndarray) -> float:
    """
    Sachs-Wolfe 线-of-sight 积分：
        Θ_l^{SW}(k) = ∫_0^{η_0} dη g(η) [Δ_0(η)/4 + Φ(η)] j_l[k(η_0-η)]
    其中 g(η) = τ' e^{-τ} 为可见度函数。
    """
    eta0 = eta_grid[-1]
    # 可见度函数（Gaussian 近似）
    eta_rec = 280.0
    sigma_rec = 30.0
    g = np.exp(-0.5 * ((eta_grid - eta_rec) / sigma_rec) ** 2) / (sigma_rec * np.sqrt(2.0 * np.pi))
    source = g * (Delta0_grid / 4.0 + Phi_grid)
    arg = k * (eta0 - eta_grid)
    from utils import spherical_bessel_j
    jvals = np.array([spherical_bessel_j(l, a) for a in arg])
    integrand = source * jvals
    return np.trapezoid(integrand, eta_grid)
