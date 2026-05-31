"""
quadrature_verify.py
数值积分规则精度验证模块

通过检验正交多项式基函数在各种积分规则上的精确度，
验证中子星物理计算中数值积分的可靠性。

原项目映射:
- 344_exactness  -> Hermite/Chebyshev/Laguerre积分精确度检验
"""

import numpy as np
import math
from typing import Callable, Tuple


# =============================================================================
# Hermite积分精确度 (源自 344_exactness)
# =============================================================================
def hermite_integral(p: int) -> float:
    """
    计算Hermite积分 ∫_{-∞}^{∞} x^p exp(-x^2) dx 的精确值。

    公式:
        I(p) = 0                         若 p 为奇数
        I(p) = (p-1)!! * √π / 2^{p/2}   若 p 为偶数

    其中 (p-1)!! = (p-1)(p-3)...1 为双阶乘。
    """
    if p < 0:
        raise ValueError("Exponent p must be non-negative.")
    if p % 2 == 1:
        return 0.0

    # 双阶乘
    double_fact = 1.0
    for k in range(p - 1, 0, -2):
        double_fact *= k

    return double_fact * math.sqrt(math.pi) / (2.0**(p / 2.0))


def hermite_quadrature_exactness(n: int, x: np.ndarray, w: np.ndarray,
                                 p_max: int = 10) -> np.ndarray:
    """
    检验Hermite积分规则的精确度。

    算法:
        对每个 p = 0, ..., p_max:
            计算 Q(f_p) = Σ_i w_i x_i^p
            计算精确值 I(f_p)
            相对误差 e_p = |Q - I| / |I|  (若 I ≠ 0)

    源自 344_exactness 中 hermite_exactness.m 的核心算法。

    Parameters
    ----------
    n : int
        积分节点数。
    x : np.ndarray
        节点坐标。
    w : np.ndarray
        权重。
    p_max : int
        最高检验阶数。

    Returns
    -------
    errors : np.ndarray
        各阶数的相对误差。
    """
    x = np.asarray(x).reshape(-1)
    w = np.asarray(w).reshape(-1)

    if x.size != n or w.size != n:
        raise ValueError("x and w must have length n.")

    errors = np.zeros(p_max + 1)

    for p in range(p_max + 1):
        exact = hermite_integral(p)
        values = x**p
        quad = np.dot(w, values)

        if abs(exact) < 1e-30:
            err = abs(quad - exact)
        else:
            err = abs((quad - exact) / exact)

        errors[p] = err

    return errors


# =============================================================================
# Chebyshev积分精确度
# =============================================================================
def chebyshev1_integral(p: int) -> float:
    """
    第一类Chebyshev积分: ∫_{-1}^{1} x^p / √(1-x^2) dx

    公式:
        p 为奇数 -> 0
        p 为偶数 -> π * (p-1)!! / p!!
    """
    if p < 0:
        raise ValueError("p must be non-negative.")
    if p % 2 == 1:
        return 0.0

    # (p-1)!! / p!!
    ratio = 1.0
    for k in range(2, p + 1, 2):
        ratio *= (k - 1.0) / k

    return math.pi * ratio


def chebyshev_quadrature_exactness(n: int, x: np.ndarray, w: np.ndarray,
                                   p_max: int = 10, kind: int = 1) -> np.ndarray:
    """
    检验Chebyshev积分规则的精确度。
    """
    x = np.asarray(x).reshape(-1)
    w = np.asarray(w).reshape(-1)
    errors = np.zeros(p_max + 1)

    for p in range(p_max + 1):
        if kind == 1:
            exact = chebyshev1_integral(p)
        else:
            exact = 0.0  # 第二类简化

        values = x**p
        quad = np.dot(w, values)

        if abs(exact) < 1e-30:
            err = abs(quad - exact)
        else:
            err = abs((quad - exact) / exact)

        errors[p] = err

    return errors


# =============================================================================
# Laguerre积分精确度
# =============================================================================
def laguerre_integral(p: int) -> float:
    """
    Laguerre积分: ∫_0^∞ x^p exp(-x) dx = p!
    """
    if p < 0:
        raise ValueError("p must be non-negative.")
    return math.factorial(p)


def laguerre_quadrature_exactness(n: int, x: np.ndarray, w: np.ndarray,
                                  p_max: int = 10) -> np.ndarray:
    """
    检验Laguerre积分规则的精确度。
    """
    x = np.asarray(x).reshape(-1)
    w = np.asarray(w).reshape(-1)
    errors = np.zeros(p_max + 1)

    for p in range(p_max + 1):
        exact = laguerre_integral(p)
        values = x**p
        quad = np.dot(w, values)

        if abs(exact) < 1e-30:
            err = abs(quad - exact)
        else:
            err = abs((quad - exact) / exact)

        errors[p] = err

    return errors


# =============================================================================
# Gauss-Hermite 节点和权重生成
# =============================================================================
def gauss_hermite_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 n 点 Gauss-Hermite 积分的节点和权重。

    方法: 用三对角Jacobi矩阵的特征值/特征向量求解。

    节点 x_i 是 Hermite 多项式 H_n(x) 的零点。
    权重 w_i = √π / (n^2 [H_{n-1}(x_i)]^2)

    Jacobi矩阵:
        J_{i,i+1} = J_{i+1,i} = √(i/2),  i = 0, ..., n-1
    """
    if n <= 0:
        raise ValueError("n must be positive.")

    # 构建对称三对角Jacobi矩阵
    diag = np.zeros(n)
    off_diag = np.zeros(n - 1)
    for i in range(n - 1):
        off_diag[i] = math.sqrt((i + 1) / 2.0)

    J = np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
    eigvals, eigvecs = np.linalg.eigh(J)

    x = eigvals
    # 权重 = μ_0 * (v_0)^2，其中 μ_0 = √π 为矩
    w = math.sqrt(math.pi) * eigvecs[0, :]**2

    return x, w


def gauss_laguerre_nodes_weights(n: int, alpha: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 n 点广义Gauss-Laguerre积分的节点和权重。

    Jacobi矩阵:
        J_{i,i} = 2i + 1 + α
        J_{i,i+1} = J_{i+1,i} = -√((i+1)(i+1+α))
    """
    if n <= 0:
        raise ValueError("n must be positive.")

    diag = np.zeros(n)
    off_diag = np.zeros(n - 1)
    for i in range(n):
        diag[i] = 2.0 * i + 1.0 + alpha
    for i in range(n - 1):
        off_diag[i] = -math.sqrt((i + 1.0) * (i + 1.0 + alpha))

    J = np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
    eigvals, eigvecs = np.linalg.eigh(J)

    x = eigvals
    mu0 = math.gamma(1.0 + alpha)
    w = mu0 * eigvecs[0, :]**2

    return x, w


# =============================================================================
# 物态方程数值积分验证
# =============================================================================
def verify_eos_integrals() -> dict:
    """
    使用高精度Gauss积分验证中子星物态方程中关键积分的数值精度。

    测试积分:
        1. ∫_0^∞ p^2 f(p) dp  (费米动量分布，用Laguerre变换)
        2. ∫_{-1}^{1} P_n(cosθ) V(θ) d(cosθ)  (Legendre展开系数)
    """
    results = {}

    # 测试1: Laguerre积分验证
    n = 16
    x_lag, w_lag = gauss_laguerre_nodes_weights(n)
    errors_lag = laguerre_quadrature_exactness(n, x_lag, w_lag, p_max=2 * n - 1)
    results['laguerre_max_error'] = float(np.max(errors_lag))

    # 测试2: Hermite积分验证
    x_her, w_her = gauss_hermite_nodes_weights(n)
    errors_her = hermite_quadrature_exactness(n, x_her, w_her, p_max=2 * n - 1)
    results['hermite_max_error'] = float(np.max(errors_her))

    # 测试3: 物态方程相关积分
    # 计算热力学量: ∫_0^∞ x^3 / (exp(x) + 1) dx = 7π^4 / 120
    # 用 Laguerre 节点做变量替换 x = t
    def fermi_dirac_integrand(t):
        return t**3 / (math.exp(t) + 1.0) * math.exp(t)

    quad_val = np.sum(w_lag * np.array([fermi_dirac_integrand(xi) for xi in x_lag]))
    exact_fd = 7.0 * math.pi**4 / 120.0
    results['fermi_dirac_relative_error'] = abs((quad_val - exact_fd) / exact_fd)

    return results
