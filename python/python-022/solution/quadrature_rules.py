"""
quadrature_rules.py
===================
高斯数值积分法则生成模块。

基于原项目 608_jacobi_rule（Gauss-Jacobi 求积法则）的核心算法，
采用 Golub-Welsch 特征值方法构造 Jacobi、Legendre、Chebyshev 等多种
高斯求积公式，用于 ICF 状态方程积分、聚变反应率积分及有限元弱形式积分。

物理应用：
1. 状态方程中电子简并压的 Fermi-Dirac 积分
2. 聚变反应截面在 Maxwellian 分布下的积分
3. Planck 与 Rosseland 平均不透明度的频率积分
4. 有限元刚度矩阵的单元积分
"""

import numpy as np
from typing import Tuple, Optional


class QuadratureRule:
    """通用求积法则容器。"""

    def __init__(self, x: np.ndarray, w: np.ndarray, a: float = -1.0, b: float = 1.0):
        self.x = np.array(x, dtype=float)
        self.w = np.array(w, dtype=float)
        self.a = a
        self.b = b
        self.n = len(x)

    def integrate(self, f) -> float:
        """对函数 f 在 [a, b] 上求积。"""
        return float(np.sum(self.w * f(self.x)))

    def scale_to(self, a_new: float, b_new: float) -> "QuadratureRule":
        """将求积公式线性缩放至新区间 [a_new, b_new]。"""
        if self.b <= self.a:
            raise ValueError("原始区间无效")
        scale = (b_new - a_new) / (self.b - self.a)
        shift = (a_new + b_new - (self.a + self.b) * scale) / 2.0
        x_new = self.x * scale + shift
        w_new = self.w * scale
        return QuadratureRule(x_new, w_new, a_new, b_new)


def jacobi_gw(n: int, alpha: float, beta: float) -> QuadratureRule:
    """
    Golub-Welsch 算法构造 Gauss-Jacobi 求积公式。

    权函数: w(x) = (1-x)^alpha * (1+x)^beta, x in [-1, 1]

    Jacobi 矩阵 J 为三对角对称矩阵：
        J_{ii} = (beta^2 - alpha^2) / ((2i+alpha+beta)(2i+alpha+beta+2))   (i>=1)
        J_{11} = (beta - alpha) / (alpha + beta + 2)
        J_{i,i+1} = sqrt( 4i(i+alpha)(i+beta)(i+alpha+beta)
                         / ((2i+alpha+beta)^2 - 1) / (2i+alpha+beta)^2 )

    节点为 J 的特征值，权重 w_j = mu0 * (v_{1,j})^2
    其中 mu0 = 2^(alpha+beta+1) * Gamma(alpha+1) * Gamma(beta+1) / Gamma(alpha+beta+2)
    """
    if n < 1:
        return QuadratureRule(np.array([]), np.array([]))
    if alpha <= -1.0 or beta <= -1.0:
        raise ValueError("Jacobi 参数必须满足 alpha > -1, beta > -1")

    ab = alpha + beta

    # 对角线
    diag = np.zeros(n)
    diag[0] = (beta - alpha) / (ab + 2.0)
    if n > 1:
        a2b2 = beta**2 - alpha**2
        for i in range(1, n):
            idx = i + 1  # 1-based index
            abi = 2.0 * idx + ab
            diag[i] = a2b2 / ((abi - 2.0) * abi)

    # 次对角线
    sub = np.zeros(n - 1)
    if n > 1:
        sub[0] = np.sqrt(4.0 * (1.0 + alpha) * (1.0 + beta)
                         / ((ab + 3.0) * (ab + 2.0)**2))
        for i in range(1, n - 1):
            idx = i + 1
            abi = 2.0 * idx + ab
            numer = 4.0 * idx * (idx + alpha) * (idx + beta) * (idx + ab)
            denom = (abi**2 - 1.0) * abi**2
            sub[i] = np.sqrt(numer / denom)

    # 构造对称三对角矩阵并求特征值
    J = np.diag(diag) + np.diag(sub, k=1) + np.diag(sub, k=-1)
    eigenvalues, eigenvectors = np.linalg.eigh(J)

    # 零阶矩
    log_mu0 = (ab + 1.0) * np.log(2.0) \
        + np.math.lgamma(alpha + 1.0) + np.math.lgamma(beta + 1.0) - np.math.lgamma(ab + 2.0)
    mu0 = np.exp(log_mu0)

    weights = mu0 * eigenvectors[0, :]**2
    nodes = eigenvalues

    return QuadratureRule(nodes, weights, -1.0, 1.0)


def legendre_gauss(n: int) -> QuadratureRule:
    """Gauss-Legendre 求积: alpha=beta=0。"""
    return jacobi_gw(n, 0.0, 0.0)


def chebyshev_gauss_first(n: int) -> QuadratureRule:
    """Chebyshev Type I: alpha=beta=-0.5。"""
    return jacobi_gw(n, -0.5, -0.5)


def compute_fermi_dirac_integral(k: int, eta: float, n_quad: int = 64) -> float:
    """
    计算 Fermi-Dirac 积分 F_k(eta) = integral_0^inf x^k / (exp(x-eta)+1) dx。

    通过变量替换 x = (1+t)/(1-t) 将 [0, inf) 映射到 [-1, 1]，
    使用权函数 (1-t)^(-2) 配合 Gauss-Legendre 求积。
    """
    if n_quad < 1:
        return 0.0

    quad = legendre_gauss(n_quad)

    def integrand(t):
        x = (1.0 + t) / (1.0 - t + 1.0e-15)
        dx_dt = 2.0 / (1.0 - t + 1.0e-15)**2
        # 防止溢出
        exp_arg = np.clip(x - eta, -700.0, 700.0)
        denom = np.exp(exp_arg) + 1.0
        return (x**k) / denom * dx_dt

    return quad.integrate(integrand)


def planck_mean_opacity_integral(T: float, n_quad: int = 32) -> float:
    """
    计算 Planck 平均不透明度的归一化频率积分。

    Planck 平均: kappa_P = integral_0^inf kappa_nu * B_nu dnu / integral B_nu dnu
    其中 B_nu 为 Planck 谱分布。

    采用无量纲变量 x = h*nu / (k_B*T)，则
    B_nu ~ x^3 / (exp(x)-1)
    """
    quad = legendre_gauss(n_quad)

    def integrand_weight(t):
        # 映射 x = (1+t)/(1-t)
        x = (1.0 + t) / (1.0 - t + 1.0e-15)
        dx_dt = 2.0 / (1.0 - t + 1.0e-15)**2
        exp_x = np.clip(x, 1.0e-10, 700.0)
        planck_weight = x**3 / (np.exp(exp_x) - 1.0)
        return planck_weight * dx_dt

    # 归一化常数
    norm = quad.integrate(integrand_weight)
    if norm < 1.0e-30:
        return 0.0
    return norm


def rosseland_mean_weight(n_quad: int = 32) -> float:
    """
    计算 Rosseland 平均的权重积分分母。
    integral (1/kappa) * dB/dT dnu
    采用无量纲形式，分母正比于 integral x^4 * exp(x) / (exp(x)-1)^2 dx
    """
    quad = legendre_gauss(n_quad)

    def integrand(t):
        x = (1.0 + t) / (1.0 - t + 1.0e-15)
        dx_dt = 2.0 / (1.0 - t + 1.0e-15)**2
        exp_x = np.clip(x, 1.0e-10, 700.0)
        weight = x**4 * np.exp(exp_x) / (np.exp(exp_x) - 1.0)**2
        return weight * dx_dt

    return quad.integrate(integrand)
