"""
measure.py — 概率测度与加权内积模块
=====================================
实现各类概率测度下的正交多项式三递推关系系数、
Gauss 求积节点/权重的计算，以及加权内积的数值求值。

核心公式:
  三递推关系:
    P_{n+1}(ξ) = (ξ - α_n) P_n(ξ) - β_n P_{n-1}(ξ)

  Gauss 求积:
    ∫ f(ξ) w(ξ) dξ ≈ Σ_{k=1}^{N_q} w_k f(ξ_k)

  加权内积:
    <f, g>_w = ∫ f(ξ) g(ξ) w(ξ) dξ

映射种子项目:
  - 033_asa076: 正态分布CDF和Owen T函数 → 概率计算基础
  - 068_ball_integrals: 高维积分 → 多维概率空间积分
"""

import numpy as np
from scipy import special, linalg
from typing import Tuple, List, Optional, Callable
from config import MeasureConfig


# ============================================================
#  三递推关系系数
# ============================================================
def recurrence_coefficients(measure: MeasureConfig, n: int
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算正交多项式的三递推关系系数 {α_k}, {β_k}。

    对经典测度有解析公式; 对一般测度使用 Stieltjes 过程 (离散化)。

    三递推:
      P_{k+1}(ξ) = (ξ - α_k) P_k(ξ) - β_k P_{k-1}(ξ)

    参数:
        measure: 概率测度配置
        n: 递推阶数 (返回 n 组系数)

    返回:
        alpha: shape (n,), 递推系数 α_0, ..., α_{n-1}
        beta:  shape (n,), 递推系数 β_0, ..., β_{n-1}
    """
    alpha = np.zeros(n)
    beta = np.zeros(n)

    if measure.measure_type == "gauss":
        # Hermite 多项式 (概率归一化)
        # α_k = μ (对标准正态 α_k = 0)
        # β_k = k * σ^2
        mu = measure.params["mu"]
        sigma = measure.params["sigma"]
        alpha[:] = mu
        beta[0] = 1.0  # P_0 = 1
        for k in range(1, n):
            beta[k] = k * sigma ** 2

    elif measure.measure_type == "uniform":
        # Legendre 多项式 (映射到 [a, b])
        a, b = measure.params["a"], measure.params["b"]
        mid = 0.5 * (a + b)
        half = 0.5 * (b - a)
        alpha[:] = mid
        beta[0] = 1.0
        for k in range(1, n):
            beta[k] = (half ** 2 * k ** 2) / (4 * k ** 2 - 1)

    elif measure.measure_type == "beta":
        # Jacobi 多项式 (映射到 [loc, loc+scale])
        a_jac = measure.params["alpha"] - 1.0  # Jacobi α 参数
        b_jac = measure.params["beta"] - 1.0   # Jacobi β 参数
        loc = measure.params["loc"]
        scale = measure.params["scale"]

        alpha[0] = loc + scale * (b_jac - a_jac) / (a_jac + b_jac + 2)
        beta[0] = 1.0

        for k in range(1, n):
            # Jacobi 递推系数
            a2 = a_jac ** 2
            b2 = b_jac ** 2
            denom = (2 * k + a_jac + b_jac) * (2 * k + a_jac + b_jac + 2)

            if k == 1:
                beta[k] = scale ** 2 * 4 * (a_jac + 1) * (b_jac + 1) / (
                    (a_jac + b_jac + 2) ** 2 * (a_jac + b_jac + 3))
            else:
                num = 4 * k * (k + a_jac) * (k + b_jac) * (
                    k + a_jac + b_jac)
                den = denom * (2 * k + a_jac + b_jac - 2) * (
                    2 * k + a_jac + b_jac - 1) if denom != 0 else 1.0
                # 防止除零
                if abs(den) < 1e-15:
                    beta[k] = 0.0
                else:
                    beta[k] = scale ** 2 * num / (den * (
                        2 * k + a_jac + b_jac) * (2 * k + a_jac + b_jac + 2)
                        / ((2 * k + a_jac + b_jac) * (
                            2 * k + a_jac + b_jac + 2)))
                    # 简化: 使用标准 Jacobi 递推
                    beta[k] = (4 * k * (k + a_jac) * (k + b_jac) *
                               (k + a_jac + b_jac) * scale ** 2 /
                               ((2 * k + a_jac + b_jac) ** 2 *
                                (2 * k + a_jac + b_jac + 1) *
                                (2 * k + a_jac + b_jac - 1)))

            alpha[k] = loc + scale * 0.5 * (
                (b_jac ** 2 - a2) /
                ((2 * k + a_jac + b_jac) * (2 * k + a_jac + b_jac + 2))
                if abs((2 * k + a_jac + b_jac) * (
                    2 * k + a_jac + b_jac + 2)) > 1e-15 else 0.0)
    else:
        raise ValueError(f"Unsupported measure type: {measure.measure_type}")

    return alpha, beta


# ============================================================
#  Gauss 求积节点与权重
# ============================================================
def gauss_quadrature(measure: MeasureConfig, n_points: int
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算给定测度下的 n 点 Gauss 求积规则。

    通过 Golub-Welsch 算法:
      构造 Jacobi 矩阵 J (对称三对角),
      J 的特征值 = 求积节点,
      J 的特征向量的第一行 = 求积权重的平方根。

    参数:
        measure: 概率测度
        n_points: 求积点数

    返回:
        nodes:   shape (n_points,), 求积节点
        weights: shape (n_points,), 求积权重 (Σw_k = 1 对概率测度)
    """
    if n_points <= 0:
        return np.array([]), np.array([])

    if n_points == 1:
        return np.array([measure.mean]), np.array([1.0])

    alpha, beta = recurrence_coefficients(measure, n_points)

    # 构造对称三对角 Jacobi 矩阵
    # J = diag(alpha) + diag(sqrt(beta[1:]), 1) + diag(sqrt(beta[1:]), -1)
    J = np.diag(alpha)
    for k in range(1, n_points):
        if beta[k] > 0:
            off = np.sqrt(beta[k])
            J[k, k - 1] = off
            J[k - 1, k] = off
        else:
            # β_k = 0 意味着正交多项式终止, 处理退化情况
            J[k, k - 1] = 0.0
            J[k - 1, k] = 0.0

    # 特征值分解
    eigenvalues, eigenvectors = linalg.eigh(J)

    nodes = eigenvalues
    # 权重 = β_0 * v_{0,k}^2, 其中 β_0 = ∫w(ξ)dξ
    # 对概率测度, β_0 = 1
    weights = beta[0] * eigenvectors[0, :] ** 2

    # 数值修正: 确保节点在支撑集内, 权重为正
    support = measure.support
    nodes = np.clip(nodes, support[0], support[1])
    weights = np.maximum(weights, 0.0)

    # 归一化权重使其和为 1 (概率测度)
    w_sum = np.sum(weights)
    if w_sum > 1e-15:
        weights /= w_sum

    return nodes, weights


# ============================================================
#  多维张量积求积
# ============================================================
def tensor_product_quadrature(measures: List[MeasureConfig],
                              n_points_per_dim: int
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """
    构造张量积求积规则。

    对 d 维随机向量 ξ = (ξ_1, ..., ξ_d), 每维 n 点:
      总点数 = n^d
      节点: 张量积 ξ_{i_1,...,i_d} = (ξ^{(1)}_{i_1}, ..., ξ^{(d)}_{i_d})
      权重: w_{i_1,...,i_d} = w^{(1)}_{i_1} × ... × w^{(d)}_{i_d}

    参数:
        measures: 各维度的测度列表
        n_points_per_dim: 每维求积点数

    返回:
        nodes:   shape (n^d, d), 多维求积节点
        weights: shape (n^d,), 多维求积权重
    """
    d = len(measures)

    # 逐维计算
    nodes_1d = []
    weights_1d = []
    for meas in measures:
        nd, wt = gauss_quadrature(meas, n_points_per_dim)
        nodes_1d.append(nd)
        weights_1d.append(wt)

    # 张量积
    grids = np.meshgrid(*nodes_1d, indexing='ij')
    nodes = np.column_stack([g.ravel() for g in grids])

    weight_grids = np.meshgrid(*weights_1d, indexing='ij')
    weights = np.ones(weight_grids[0].size)
    for wg in weight_grids:
        weights *= wg.ravel()

    return nodes, weights


# ============================================================
#  加权内积计算
# ============================================================
def weighted_inner_product(f_values: np.ndarray,
                           g_values: np.ndarray,
                           weights: np.ndarray) -> float:
    """
    计算离散加权内积。

    <f, g>_w = Σ_k w_k f(ξ_k) g(ξ_k)

    参数:
        f_values: f 在求积节点上的值, shape (n_q,)
        g_values: g 在求积节点上的值, shape (n_q,)
        weights:  求积权重, shape (n_q,)

    返回:
        加权内积值
    """
    return float(np.sum(weights * f_values * g_values))


# ============================================================
#  正交多项式在求积节点上的范数
# ============================================================
def polynomial_norm_sq(measure: MeasureConfig, degree: int) -> float:
    """
    计算正交多项式 P_n 的加权 L2 范数平方。

    h_n = <P_n, P_n>_w = ∫ P_n(ξ)^2 w(ξ) dξ

    对经典正交多项式:
      Hermite: h_n = n! σ^{2n}
      Legendre: h_n = (b-a)^{2n+1} / ((2n+1) C(2n,n))  [简化]
      一般: h_n = β_0 β_1 ... β_n

    参数:
        measure: 测度
        degree:  多项式阶数 n

    返回:
        h_n: 范数平方
    """
    _, beta = recurrence_coefficients(measure, degree + 1)
    h_n = beta[0]
    for k in range(1, degree + 1):
        h_n *= beta[k]
    return max(h_n, 1e-300)  # 防止零


# ============================================================
#  概率密度函数求值
# ============================================================
def pdf_eval(measure: MeasureConfig, xi: np.ndarray) -> np.ndarray:
    """
    计算概率密度函数 w(ξ) 在给定节点上的值。

    参数:
        measure: 测度配置
        xi: 求值点, shape (n,)

    返回:
        pdf_values: shape (n,)
    """
    if measure.measure_type == "gauss":
        mu = measure.params["mu"]
        sigma = measure.params["sigma"]
        z = (xi - mu) / sigma
        return np.exp(-0.5 * z ** 2) / (sigma * np.sqrt(2 * np.pi))

    elif measure.measure_type == "uniform":
        a, b = measure.params["a"], measure.params["b"]
        return np.where((xi >= a) & (xi <= b), 1.0 / (b - a), 0.0)

    elif measure.measure_type == "beta":
        alpha_p = measure.params["alpha"]
        beta_p = measure.params["beta"]
        loc = measure.params["loc"]
        scale = measure.params["scale"]
        x_norm = (xi - loc) / scale
        # Beta 分布 PDF
        log_pdf = (special.xlogy(alpha_p - 1, x_norm) +
                   special.xlogy(beta_p - 1, 1 - x_norm) -
                   special.betaln(alpha_p, beta_p))
        pdf = np.exp(log_pdf) / scale
        return np.where((xi >= loc) & (xi <= loc + scale), pdf, 0.0)

    raise ValueError(f"Unknown measure: {measure.measure_type}")


# ============================================================
#  累积分布函数 (映射 033_asa076 的 alnorm)
# ============================================================
def cdf_eval(measure: MeasureConfig, xi: np.ndarray) -> np.ndarray:
    """
    计算累积分布函数 Φ(ξ)。

    使用标准正态CDF (alnorm) 和 Beta 不完全函数。

    参数:
        measure: 测度配置
        xi: 求值点

    返回:
        cdf_values
    """
    if measure.measure_type == "gauss":
        mu = measure.params["mu"]
        sigma = measure.params["sigma"]
        z = (xi - mu) / sigma
        return 0.5 * (1.0 + special.erf(z / np.sqrt(2)))

    elif measure.measure_type == "uniform":
        a, b = measure.params["a"], measure.params["b"]
        return np.clip((xi - a) / (b - a), 0.0, 1.0)

    elif measure.measure_type == "beta":
        alpha_p = measure.params["alpha"]
        beta_p = measure.params["beta"]
        loc = measure.params["loc"]
        scale = measure.params["scale"]
        x_norm = np.clip((xi - loc) / scale, 0.0, 1.0)
        return special.betainc(alpha_p, beta_p, x_norm)

    raise ValueError(f"Unknown measure: {measure.measure_type}")


# ============================================================
#  Owen T 函数 (映射 033_asa076 的 tfn)
# ============================================================
def owen_t_function(h: float, a: float) -> float:
    """
    计算 Owen T 函数。

    T(h, a) = (1/(2π)) ∫_0^a exp(-h^2(1+t^2)/2) / (1+t^2) dt

    用于计算二元正态分布的累积概率, 在不确定性量化中用于
    计算失效概率和联合概率。

    使用 5 点 Gauss-Legendre 求积 (映射 asa076 方法)。

    参数:
        h: 参数 h
        a: 参数 a

    返回:
        T(h, a)
    """
    if abs(a) < 1e-15:
        return 0.0

    # 5 点 Gauss-Legendre 节点和权重 (在 [0, a] 上)
    nodes_gl, weights_gl = np.polynomial.legendre.leggauss(5)
    # 映射 [-1, 1] → [0, a]
    nodes = 0.5 * a * (nodes_gl + 1.0)
    weights = 0.5 * a * weights_gl

    # 被积函数: exp(-h^2(1+t^2)/2) / (1+t^2)
    integrand = np.exp(-0.5 * h ** 2 * (1.0 + nodes ** 2)) / (
        1.0 + nodes ** 2)

    T_val = np.sum(weights * integrand) / (2 * np.pi)

    return T_val


# ============================================================
#  失效概率计算 (映射 1180 的贝叶斯后验)
# ============================================================
def failure_probability(pce_coeffs: np.ndarray,
                        basis_values: np.ndarray,
                        threshold: float,
                        weights: np.ndarray) -> float:
    """
    计算失效概率 P(g(ξ) > threshold)。

    使用 PCE 代理模型:
      g(ξ) ≈ Σ_k c_k Ψ_k(ξ)

    P_f = ∫_{g(ξ)>threshold} w(ξ) dξ
        ≈ Σ_{i: g(ξ_i)>threshold} w_i

    参数:
        pce_coeffs:   PCE 系数, shape (P,)
        basis_values: 基函数在求积节点上的值, shape (n_q, P)
        threshold:    失效阈值
        weights:      求积权重, shape (n_q,)

    返回:
        P_f: 失效概率
    """
    # 计算 g(ξ) 在各节点的值
    g_values = basis_values @ pce_coeffs

    # 指示函数求和
    indicator = (g_values > threshold).astype(float)
    P_f = np.sum(weights * indicator)

    return float(np.clip(P_f, 0.0, 1.0))
