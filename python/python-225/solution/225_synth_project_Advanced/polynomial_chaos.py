# -*- coding: utf-8 -*-
"""
polynomial_chaos.py
===================

多项式混沌展开 (PCE) 模块, 用于量化暗物质天体物理参数的不确定性

对应种子项目 853_pce_legendre: 使用 Legendre 多项式基底的随机 Galerkin 组装。

科学公式
--------
多项式混沌展开:
  u(x, ξ) ≈ Σ_{|α|≤p} u_α(x) Ψ_α(ξ)

其中:
  ξ = (ξ_1, ..., ξ_N) 为随机参数向量
  Ψ_α(ξ) = Π_i P_{α_i}(ξ_i) 为多维 Legendre 多项式
  u_α(x) 为确定性系数

截断阶数 P 下的基函数个数:
  N_PCE = C(N + P, P) = (N + P)! / (N! P!)

Galerkin 投影:
  对随机 PDE: L[u(x, ξ)] = f(x, ξ)
  投影: E[L[u] Ψ_β] = E[f Ψ_β] 对所有 |β| ≤ P

Legendre 多项式递推:
  (n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x)
  P_0(x) = 1,  P_1(x) = x
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Callable
from itertools import combinations_with_replacement
from math import factorial


# ---------------------------------------------------------------------------
# 第一部分: Legendre 多项式
# ---------------------------------------------------------------------------

def legendre_poly(x: np.ndarray, n: int) -> np.ndarray:
    """
    计算 Legendre 多项式 P_n(x) (递推公式)。

    P_0(x) = 1
    P_1(x) = x
    (n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x)
    """
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x.copy()
    P_prev = np.ones_like(x)
    P_curr = x.copy()
    for k in range(1, n):
        P_next = ((2*k + 1) * x * P_curr - k * P_prev) / (k + 1)
        P_prev = P_curr
        P_curr = P_next
    return P_curr


def legendre_poly_derivative(x: np.ndarray, n: int) -> np.ndarray:
    """
    Legendre 多项式导数:
      P'_n(x) = n (x P_n(x) - P_{n-1}(x)) / (x² - 1)

    在 x = ±1 处使用 L'Hôpital:
      P'_n(1) = n(n+1)/2
      P'_n(-1) = (-1)^{n-1} n(n+1)/2
    """
    if n == 0:
        return np.zeros_like(x)
    Pn = legendre_poly(x, n)
    Pn_1 = legendre_poly(x, n - 1)
    denom = x * x - 1.0
    result = np.zeros_like(x)
    mask = np.abs(denom) > 1e-12
    result[mask] = n * (x[mask] * Pn[mask] - Pn_1[mask]) / denom[mask]
    # 端点
    if np.any(~mask):
        result[~mask] = np.where(
            x[~mask] > 0,
            n * (n + 1) / 2.0,
            (-1)**(n - 1) * n * (n + 1) / 2.0
        )
    return result


# ---------------------------------------------------------------------------
# 第二部分: 多项式混沌基
# ---------------------------------------------------------------------------

def pce_multi_index(N: int, P: int) -> List[Tuple[int, ...]]:
    """
    生成 N 维总阶数 ≤ P 的所有多重指标。

    对于 N=2, P=2:
      (0,0), (0,1), (0,2), (1,0), (1,1), (2,0)
    总数 = C(N+P, P) = (N+P)! / (N! P!)
    """
    indices = []
    for total_deg in range(P + 1):
        for combo in combinations_with_replacement(range(N), total_deg):
            alpha = [0] * N
            for idx in combo:
                alpha[idx] += 1
            indices.append(tuple(alpha))
    # 去重并排序
    indices = sorted(set(indices))
    return indices


def pce_basis_count(N: int, P: int) -> int:
    """PCE 基函数总数 = C(N+P, P)。"""
    return factorial(N + P) // (factorial(N) * factorial(P))


def evaluate_pce_basis(
    xi: np.ndarray,
    alpha: Tuple[int, ...],
) -> float:
    """
    计算多维 Legendre PCE 基函数在点 xi 的值:

    Ψ_α(ξ) = Π_i P_{α_i}(ξ_i)

    xi ∈ [-1, 1]^N
    """
    val = 1.0
    for i, a in enumerate(alpha):
        val *= legendre_poly(np.array([xi[i]]), a)[0]
    return val


def evaluate_pce_basis_array(
    xi_samples: np.ndarray,
    alpha: Tuple[int, ...],
) -> np.ndarray:
    """批量计算 PCE 基函数。"""
    N = len(alpha)
    result = np.ones(len(xi_samples))
    for i, a in enumerate(alpha):
        result *= legendre_poly(xi_samples[:, i], a)
    return result


# ---------------------------------------------------------------------------
# 第三部分: 随机 Galerkin 组装 (源自 853_pce_legendre)
# ---------------------------------------------------------------------------

def pce_galerkin_triple_product(
    alpha: Tuple[int, ...],
    beta: Tuple[int, ...],
    gamma: Tuple[int, ...],
) -> float:
    """
    计算三重乘积积分 (Galerkin 耦合系数):

    C_{αβγ} = E[Ψ_α Ψ_β Ψ_γ] = ∫ Ψ_α Ψ_β Ψ_γ ρ(ξ) dξ

    对于独立均匀分布 ξ_i ~ U(-1, 1):
      = Π_i ∫_{-1}^{1} P_{α_i}(ξ_i) P_{β_i}(ξ_i) P_{γ_i}(ξ_i) dξ_i / 2

    利用 Legendre 乘积积分公式:
      ∫_{-1}^{1} P_a P_b P_c dξ = 2 * (Gaunt 系数)
    """
    N = len(alpha)
    result = 1.0
    for i in range(N):
        a, b, c = alpha[i], beta[i], gamma[i]
        # 奇偶性筛选
        if (a + b + c) % 2 != 0:
            return 0.0
        # 三角不等式
        if a + b < c or a + c < b or b + c < a:
            return 0.0
        # 简化: 使用 Gauss-Legendre 求积
        from velocity_quadrature import gauss_legendre_nodes_weights
        nodes, weights = gauss_legendre_nodes_weights(max(a, b, c) + 5)
        Pa = legendre_poly(nodes, a)
        Pb = legendre_poly(nodes, b)
        Pc = legendre_poly(nodes, c)
        integral_i = np.sum(weights * Pa * Pb * Pc) / 2.0
        result *= integral_i
    return result


def assemble_stochastic_matrix(
    N_random: int,
    P_degree: int,
    A_deterministic: np.ndarray,
    kl_coefficients: Optional[List[np.ndarray]] = None,
) -> np.ndarray:
    """
    组装随机 Galerkin 矩阵 (源自 853_pce_legendre 核心算法)。

    对于随机 PDE:
      -∇·(a(x,ξ) ∇u) = f(x)
    其中 a(x,ξ) = a_0(x) + Σ_i a_i(x) ξ_i

    Galerkin 矩阵块:
      B_{(α,x),(β,y)} = Σ_i a_i(x,y) * C_{α,i,β}

    简化版本: 仅处理随机标量乘子
    """
    indices = pce_multi_index(N_random, P_degree)
    N_PCE = len(indices)
    n_spatial = A_deterministic.shape[0]
    N_total = n_spatial * N_PCE

    B = np.zeros((N_total, N_total))

    for i_idx, alpha in enumerate(indices):
        for j_idx, beta in enumerate(indices):
            # 计算 E[Ψ_α Ψ_β] (对角, 因为正交)
            ortho = 1.0
            for k in range(N_random):
                # ∫ P_{α_k} P_{β_k} dξ / 2 = δ_{α_k, β_k} / (2α_k + 1)
                if alpha[k] != beta[k]:
                    ortho = 0.0
                    break
                else:
                    ortho *= 1.0 / (2 * alpha[k] + 1)
            if ortho == 0:
                continue
            # 嵌入空间块
            B[
                i_idx*n_spatial:(i_idx+1)*n_spatial,
                j_idx*n_spatial:(j_idx+1)*n_spatial
            ] = A_deterministic * ortho

    return B


# ---------------------------------------------------------------------------
# 第四部分: DM 参数不确定性量化
# ---------------------------------------------------------------------------

def pce_recoil_uncertainty(
    rate_function: Callable,
    param_names: List[str],
    param_bounds: List[Tuple[float, float]],
    P_degree: int = 3,
    n_samples: int = 50,
    seed: int = 42,
) -> dict:
    """
    使用 PCE 量化反冲率对天体物理参数的敏感性。

    参数:
      rate_function(E_R, params) -> float
      param_names: ['v_0', 'rho_0', 'v_esc']
      param_bounds: [(200, 240), (0.2, 0.4), (500, 600)]

    返回:
      {mean, std, sensitivities, pce_coefficients}
    """
    N = len(param_names)
    indices = pce_multi_index(N, P_degree)
    N_PCE = len(indices)

    # 生成样本 (Latin Hypercube 风格)
    rng = np.random.default_rng(seed)
    xi_samples = rng.uniform(-1, 1, size=(n_samples, N))

    # 参数映射: [-1, 1] → 实际范围
    def map_params(xi):
        params = {}
        for i, (lo, hi) in enumerate(param_bounds):
            params[param_names[i]] = 0.5 * (hi - lo) * (xi[i] + 1) + lo
        return params

    # 评估率函数在样本点
    E_R_test = 5.0  # keV
    Y_samples = np.zeros(n_samples)
    for i in range(n_samples):
        params = map_params(xi_samples[i])
        try:
            Y_samples[i] = rate_function(E_R_test, params)
        except Exception:
            Y_samples[i] = 0.0

    # PCE 系数拟合 (最小二乘)
    Psi = np.zeros((n_samples, N_PCE))
    for j, alpha in enumerate(indices):
        Psi[:, j] = evaluate_pce_basis_array(xi_samples, alpha)

    # 求解 Psi @ c = Y
    # 使用最小二乘: c = (Psi^T Psi)^{-1} Psi^T Y
    try:
        coeffs, _, _, _ = np.linalg.lstsq(Psi, Y_samples, rcond=None)
    except np.linalg.LinAlgError:
        coeffs = np.zeros(N_PCE)

    # 统计量
    mean = coeffs[0]  # 0阶系数 = 均值
    variance = 0.0
    sensitivities = {name: 0.0 for name in param_names}
    for j, alpha in enumerate(indices):
        if j == 0:
            continue
        # 方差贡献: c_j² * E[Ψ_j²]
        norm_sq = 1.0
        for k in range(N):
            norm_sq *= 1.0 / (2 * alpha[k] + 1)
        variance += coeffs[j]**2 * norm_sq
        # Sobol 指数近似
        for k in range(N):
            if alpha[k] > 0:
                sensitivities[param_names[k]] += coeffs[j]**2 * norm_sq

    std = np.sqrt(max(variance, 0.0))

    return {
        'mean': float(mean),
        'std': float(std),
        'variance': float(variance),
        'pce_coefficients': coeffs,
        'multi_indices': indices,
        'sensitivities': sensitivities,
    }


# ---------------------------------------------------------------------------
# 第五部分: 多项式变换 (源自 158_change_polynomial)
# ---------------------------------------------------------------------------

def polynomial_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    多项式乘法 (源自 158_change_polynomial 中的 polynomial_multiply)。

    给定系数数组 a, b (a[i] 为 x^i 的系数),
    返回 c = a * b 的系数。
    """
    n = len(a)
    m = len(b)
    c = np.zeros(n + m - 1)
    for i in range(n):
        for j in range(m):
            c[i + j] += a[i] * b[j]
    return c


def change_polynomial_count(
    values: np.ndarray,
    target: int,
    coin_num: int,
) -> np.ndarray:
    """
    找零问题的多项式计数 (源自 158_change_polynomial 核心算法)。

    A(I) 表示使用 coin_num 枚硬币 (按顺序) 组成和 I 的序列数。

    应用于 DM 散射中的多体相空间计数:
      将末态粒子能量量子化为离散格点,
      计算能量守恒约束下的相空间体积。
    """
    a = np.zeros(target + 1, dtype=np.int64)
    if coin_num <= 0:
        return a
    for v in values:
        if 0 <= v <= target:
            a[v] += 1
    if coin_num == 1:
        return a
    p = a.copy()
    for _ in range(2, coin_num + 1):
        a = polynomial_multiply(a, p)
        if len(a) > target + 1:
            a = a[:target + 1]
    return a
