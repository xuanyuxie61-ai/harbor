"""
vle_thermodynamics.py
=====================
汽液平衡(VLE)热力学计算模块。

本模块基于 Jacobi 多项式谱展开与 Laguerre 高斯求积，计算多组分非理想体系
的汽液平衡关系。核心包括：
1. Jacobi 多项式展开活度系数；
2. Laguerre-Gauss 求积计算无穷温度区间的相平衡积分；
3. Wilson 方程计算二元交互作用能参数；
4. Antoine 方程计算饱和蒸气压。

科学背景
--------
对于多组分体系，汽液平衡条件为：
    y_i * P * φ_i^V = x_i * γ_i * P_i^sat * φ_i^sat * exp[ V_i^L (P - P_i^sat) / (R T) ]

在低压下简化为：
    y_i * P = x_i * γ_i * P_i^sat

活度系数 γ_i 由 Wilson 模型给出：
    ln γ_i = 1 - ln( Σ_j x_j Λ_{ij} ) - Σ_k [ x_k Λ_{ki} / Σ_j x_j Λ_{kj} ]

其中 Wilson 参数：
    Λ_{ij} = (V_j / V_i) * exp[ - (λ_{ij} - λ_{ii}) / (R T) ]

Jacobi 多项式用于在 [-1,1] 区间上对组成依赖的函数进行谱展开：
    f(x) ≈ Σ_{n=0}^{N} a_n P_n^{(α,β)}(x)

Laguerre-Gauss 求积用于无穷温度区间的积分：
    ∫_0^∞ e^{-x} x^α f(x) dx ≈ Σ_{i=1}^{n} w_i f(x_i)
"""

import numpy as np
import math
from utils import ensure_positive, clip_with_warning, safe_divide


# ---------------------------------------------------------------------------
# Jacobi 多项式（源自项目 607_jacobi_polynomial）
# ---------------------------------------------------------------------------

def jacobi_polynomial(m, n, alpha, beta, x):
    """
    计算 Jacobi 多项式 P_k^{(α,β)}(x)，k=0..n。

    递推关系：
        P_0 = 1
        P_1 = [(2+α+β) x + (α-β)] / 2
        c1 P_n = (c3 + c2 x) P_{n-1} + c4 P_{n-2}

    Parameters
    ----------
    m : int
        求值点个数。
    n : int
        最高阶数。
    alpha, beta : float
        Jacobi 参数，必须 > -1。
    x : ndarray, shape (m,)
        求值点，通常在 [-1, 1]。

    Returns
    -------
    v : ndarray, shape (m, n+1)
        各阶 Jacobi 多项式值。
    """
    if alpha <= -1.0 or beta <= -1.0:
        raise ValueError("jacobi_polynomial: alpha and beta must be > -1")
    if n < 0:
        return np.empty((m, 0))

    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size != m:
        m = x.size

    v = np.ones((m, n + 1), dtype=float)
    if n == 0:
        return v

    v[:, 1] = (1.0 + 0.5 * (alpha + beta)) * x + 0.5 * (alpha - beta)

    for i in range(2, n + 1):
        c1 = 2.0 * i * (i + alpha + beta) * (2.0 * i - 2.0 + alpha + beta)
        c2 = (2.0 * i - 1.0 + alpha + beta) * (2.0 * i + alpha + beta) * (2.0 * i - 2.0 + alpha + beta)
        c3 = (2.0 * i - 1.0 + alpha + beta) * (alpha + beta) * (alpha - beta)
        c4 = -2.0 * (i - 1.0 + alpha) * (i - 1.0 + beta) * (2.0 * i + alpha + beta)
        v[:, i] = ((c3 + c2 * x) * v[:, i - 1] + c4 * v[:, i - 2]) / c1

    return v


# ---------------------------------------------------------------------------
# Laguerre-Gauss 求积（源自项目 640_laguerre_integrands）
# ---------------------------------------------------------------------------

def laguerre_root(x, norder, alpha, b, c):
    """
    使用 Newton 迭代寻找 Laguerre 多项式的根。
    """
    eps = 2.220446049250313e-16
    max_iter = 100
    for _ in range(max_iter):
        p1 = 1.0
        p2 = 0.0
        for j in range(1, norder + 1):
            p3 = p2
            p2 = p1
            p1 = (x - b[j - 1]) * p2 - c[j - 1] * p3
        dp1 = norder * (p1 - p2) / x if abs(x) > eps else norder * p1
        dx = p1 / dp1
        x = x - dx
        if abs(dx) < eps:
            break
    return x, dp1, p1


def laguerre_compute(norder, alpha=0.0):
    """
    计算 Gauss-Laguerre 求积的节点与权重。

    用于计算：
        ∫_0^∞ e^{-x} x^α f(x) dx ≈ Σ w_i f(x_i)

    Parameters
    ----------
    norder : int
        求积阶数，>=1。
    alpha : float
        指数参数，>=0。

    Returns
    -------
    xtab : ndarray, shape (norder,)
        节点。
    weight : ndarray, shape (norder,)
        权重。
    """
    if norder < 1:
        raise ValueError("laguerre_compute: norder must be >= 1")
    if alpha < 0.0:
        alpha = 0.0

    b = np.empty(norder, dtype=float)
    c = np.empty(norder, dtype=float)
    for i in range(1, norder + 1):
        b[i - 1] = alpha + 2.0 * i - 1.0
        c[i - 1] = (i - 1.0) * (alpha + i - 1.0)

    cc = math.gamma(alpha + 1.0) * np.prod(c[1:]) if norder > 1 else math.gamma(alpha + 1.0)

    xtab = np.empty(norder, dtype=float)
    weight = np.empty(norder, dtype=float)

    for i in range(1, norder + 1):
        if i == 1:
            x = (1.0 + alpha) * (3.0 + 0.92 * alpha) / (1.0 + 2.4 * norder + 1.8 * alpha)
        elif i == 2:
            x = xtab[0] + (15.0 + 6.25 * alpha) / (1.0 + 0.9 * alpha + 2.5 * norder)
        else:
            r1 = (1.0 + 2.55 * (i - 2)) / (1.9 * (i - 2))
            r2 = 1.26 * (i - 2) * alpha / (1.0 + 3.5 * (i - 2))
            ratio = (r1 + r2) / (1.0 + 0.3 * alpha)
            x = xtab[i - 2] + ratio * (xtab[i - 2] - xtab[i - 3])

        x, dp2, p1 = laguerre_root(x, norder, alpha, b, c)
        xtab[i - 1] = x
        weight[i - 1] = cc / dp2 / p1

    return xtab, weight


def laguerre_quadrature_integrate(f, norder=16, alpha=0.0, transform=None):
    """
    使用 Laguerre-Gauss 求积计算 ∫_0^∞ e^{-x} x^α f(x) dx。

    若提供 transform(x)，则计算 ∫_0^∞ e^{-x} x^α f(transform(x)) dx。
    """
    xtab, weight = laguerre_compute(norder, alpha)
    if transform is not None:
        vals = f(transform(xtab))
    else:
        vals = f(xtab)
    vals = np.asarray(vals, dtype=float)
    return float(np.sum(weight * vals))


# ---------------------------------------------------------------------------
# Antoine 方程与饱和蒸气压
# ---------------------------------------------------------------------------

def antoine_vapor_pressure(T, A, B, C):
    """
    Antoine 方程计算饱和蒸气压 [Pa]：
        log10(P_sat) = A - B / (T + C)
        P_sat = 10^{A - B/(T+C)} [mmHg] -> [Pa]

    Parameters
    ----------
    T : float
        温度 [°C]。
    A, B, C : float
        Antoine 常数。

    Returns
    -------
    P_sat : float
        饱和蒸气压 [Pa]。
    """
    T = float(T)
    P_mmHg = 10.0 ** (A - B / (T + C))
    P_pa = P_mmHg * 133.322
    return P_pa


# ---------------------------------------------------------------------------
# Wilson 方程计算活度系数
# ---------------------------------------------------------------------------

def wilson_parameters(V, Lambda_ij, T):
    """
    计算 Wilson 参数矩阵。

    Λ_{ij} = (V_j / V_i) * exp[ -Δλ_{ij} / (R T) ]

    Parameters
    ----------
    V : ndarray, shape (nc,)
        摩尔体积 [m^3/mol]。
    Lambda_ij : ndarray, shape (nc, nc)
        二元交互能参数 Δλ_{ij} / R [K]。
    T : float
        温度 [K]。

    Returns
    -------
    Lambda : ndarray, shape (nc, nc)
        Wilson 参数矩阵。
    """
    nc = len(V)
    T = ensure_positive(T, name="T")
    V = ensure_positive(V, name="V")
    Lambda = np.zeros((nc, nc), dtype=float)
    R = 8.314  # J/(mol K)
    for i in range(nc):
        for j in range(nc):
            if i == j:
                Lambda[i, j] = 1.0
            else:
                Lambda[i, j] = (V[j] / V[i]) * np.exp(-Lambda_ij[i, j] / T)
    return Lambda


def wilson_activity_coefficient(x, V, Lambda_ij, T):
    """
    使用 Wilson 方程计算活度系数。

    ln γ_i = 1 - ln( Σ_j x_j Λ_{ij} ) - Σ_k [ x_k Λ_{ki} / Σ_j x_j Λ_{kj} ]

    Parameters
    ----------
    x : ndarray, shape (nc,)
        液相摩尔分数。
    V : ndarray, shape (nc,)
        摩尔体积。
    Lambda_ij : ndarray, shape (nc, nc)
        二元交互能参数。
    T : float
        温度 [K]。

    Returns
    -------
    gamma : ndarray, shape (nc,)
        活度系数。
    """
    x = np.asarray(x, dtype=float)
    x = x / np.sum(x)
    nc = len(x)
    Lambda = wilson_parameters(V, Lambda_ij, T)

    gamma = np.zeros(nc, dtype=float)
    for i in range(nc):
        sum1 = np.sum(x * Lambda[i, :])
        sum1 = ensure_positive(sum1, name="sum1_wilson")
        term1 = -np.log(sum1)
        term2 = 0.0
        for k in range(nc):
            sum2 = np.sum(x * Lambda[k, :])
            sum2 = ensure_positive(sum2, name="sum2_wilson")
            term2 += x[k] * Lambda[k, i] / sum2
        gamma[i] = np.exp(1.0 + term1 - term2)

    return gamma


# ---------------------------------------------------------------------------
# 汽液平衡计算
# ---------------------------------------------------------------------------

def vle_flash_calculation(x, P_total, T, A_ant, B_ant, C_ant, V, Lambda_ij):
    """
    等温闪蒸计算：给定液相组成 x，计算汽相组成 y 和相对挥发度。

    低压简化模型：
        K_i = γ_i * P_i^sat / P_total
        y_i = K_i * x_i

    Parameters
    ----------
    x : ndarray
        液相摩尔分数。
    P_total : float
        总压 [Pa]。
    T : float
        温度 [K]，先转为 °C 用于 Antoine。
    A_ant, B_ant, C_ant : ndarray
        Antoine 常数。
    V : ndarray
        摩尔体积。
    Lambda_ij : ndarray
        Wilson 参数。

    Returns
    -------
    y : ndarray
        汽相摩尔分数（归一化）。
    K : ndarray
        相平衡常数 K_i。
    gamma : ndarray
        活度系数。
    """
    x = np.asarray(x, dtype=float)
    x = x / np.sum(x)
    nc = len(x)
    T_celsius = T - 273.15

    P_sat = np.array([antoine_vapor_pressure(T_celsius, A_ant[i], B_ant[i], C_ant[i]) for i in range(nc)])
    gamma = wilson_activity_coefficient(x, V, Lambda_ij, T)

    # TODO [Hole 1]: 实现低压 VLE 模型中相平衡常数 K_i 与汽相组成 y_i 的计算。
    # 科学背景：低压下 K_i = γ_i * P_i^sat / P_total，汽相组成 y_i = K_i * x_i。
    # 要求：
    #   1. 计算各组分的相平衡常数 K
    #   2. 由 K 和液相组成 x 计算汽相组成 y
    #   3. 对 y 进行归一化（保证和为1），并处理极端情况
    # 注意：此处的实现必须与 mass_transfer_dynamics.py 中 distillation_column_deriv
    #       对 alpha_rel 的使用方式保持一致（alpha_rel 由 K 导出）。
    K = None  # 待修复
    y = None  # 待修复
    # --------------------------------------------------
    return y, K, gamma


def vle_relative_volatility(K):
    """
    计算相对挥发度 α_i = K_i / K_ref，以 K 最大值组分为参考。
    """
    K = np.asarray(K, dtype=float)
    K_ref = np.max(K)
    K_ref = max(K_ref, 1e-15)
    return K / K_ref


# ---------------------------------------------------------------------------
# Jacobi 谱展开活度系数随组成变化
# ---------------------------------------------------------------------------

def activity_coefficient_spectral_expansion(x_range, nc, alpha_jac=0.0, beta_jac=0.0, n_modes=8):
    """
    使用 Jacobi 多项式在组成区间 [-1,1] 上构造活度系数的谱表示。
    这里将组成 x ∈ [0,1] 线性映射到 ξ ∈ [-1,1]：
        ξ = 2x - 1

    返回一组基函数的系数（示例用正弦型模拟真实活度系数变化）。
    """
    x_range = np.asarray(x_range, dtype=float)
    xi = 2.0 * x_range - 1.0
    xi = clip_with_warning(xi, -1.0, 1.0, "xi")

    V_jac = jacobi_polynomial(len(xi), n_modes, alpha_jac, beta_jac, xi)
    return V_jac
