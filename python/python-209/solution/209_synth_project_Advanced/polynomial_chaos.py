"""
polynomial_chaos.py — 广义多项式混沌展开 (gPC)
================================================

核心方法:
  1. Diophantine 有界解枚举多指标集 (映射自 847_pariomino)
  2. 奇偶荷与对称变换 (映射自 847_pariomino)
  3. 正交多项式基: Hermite/Legendre/Laguerre
  4. PCE 系数 SVD 分析 (映射自 180_circle_map)

映射种子项目:
  - 847_pariomino: diophantine_nd_nonnegative_bounded, polyomino_charge,
                    pariomino_transform
  - 180_circle_map: SVD 椭圆映射分析 PCE 系数空间
"""

import numpy as np
from math import factorial


# ============================================================
# 第1部分: Diophantine 多指标枚举
# (映射自 847_pariomino: diophantine_nd_nonnegative_bounded)
# ============================================================

def diophantine_nd_nonnegative_bounded(a, b, m):
    """
    有界非负整数线性 Diophantine 方程:
        a₁x₁ + ... + aₙxₙ = b,  0 ≤ xᵢ ≤ mᵢ
    回溯搜索, 从最大可行值开始。
    PCE 应用: a=(1,...,1), b=p → 所有 |i|=p 的多指标。
    """
    a = np.asarray(a, dtype=int)
    m = np.asarray(m, dtype=int)
    n = len(a)
    solutions = []

    def backtrack(var_idx, residual, partial):
        if var_idx == n - 1:
            if a[var_idx] == 0:
                if residual == 0:
                    solutions.append(partial + [0])
                return
            if residual % a[var_idx] != 0:
                return
            x_last = residual // a[var_idx]
            if 0 <= x_last <= m[var_idx]:
                solutions.append(partial + [x_last])
            return
        max_val = min(m[var_idx], residual // max(a[var_idx], 1))
        for val in range(max_val, -1, -1):
            new_residual = residual - val * a[var_idx]
            if new_residual < 0:
                continue
            remaining_max = sum(a[k] * m[k] for k in range(var_idx + 1, n))
            if new_residual > remaining_max:
                continue
            backtrack(var_idx + 1, new_residual, partial + [val])

    backtrack(0, b, [])
    if solutions:
        solutions.sort()
    return np.array(solutions, dtype=int) if solutions else np.zeros((0, n), dtype=int)


def total_order_multiindices(d, p):
    """
    全阶截断多指标集:
        J_{p,d} = {α ∈ ℕ₀^d : |α| ≤ p}
    基函数数: C(p+d, d)
    """
    all_indices = [tuple([0] * d)]
    for total in range(1, p + 1):
        a = np.ones(d, dtype=int)
        m = np.full(d, total, dtype=int)
        sols = diophantine_nd_nonnegative_bounded(a, total, m)
        for row in sols:
            all_indices.append(tuple(row))
    return np.array(all_indices, dtype=int)


def hyperbolic_cross_multiindices(d, p, q_norm=0.5):
    """
    双曲交叉截断:
        HC(q) = {α ∈ ℕ₀^d : Π (1+αₖ)^q ≤ p+1}
    q=1: 全阶; q→0: 仅低阶交互
    """
    indices = []
    max_per = int((p + 1) ** (1.0 / max(q_norm, 0.01))) + 1

    def recurse(dim, prod, current):
        if dim == d:
            indices.append(tuple(current))
            return
        for val in range(max_per + 1):
            new_prod = prod * (1 + val) ** q_norm
            if new_prod > p + 1 + 1e-12:
                break
            recurse(dim + 1, new_prod, current + [val])

    recurse(0, 1.0, [])
    return np.array(indices, dtype=int)


# ============================================================
# 第2部分: 奇偶荷与对称性
# (映射自 847_pariomino: polyomino_charge, pariomino_transform)
# ============================================================

def multiindex_parity_charge(multi_index):
    """
    奇偶荷: charge(α) = (-1)^{|α|} = (-1)^{α₁+...+α_d}
    用于识别 PCE 基函数对称性, 加速 Galerkin 投影。
    """
    return 1 if sum(multi_index) % 2 == 0 else -1


def multiindex_transform(multi_index, permutation=None, sign_flip=None):
    """
    多指标对称变换 (映射自 pariomino_transform):
      - permutation: 维度置换
      - sign_flip:   维度反射
    用于检测随机参数置换对称性。
    """
    idx = list(multi_index)
    if permutation is not None:
        idx = [idx[permutation[k]] for k in range(len(idx))]
    if sign_flip is not None:
        for k in sign_flip:
            if k < len(idx):
                idx[k] = -idx[k]
    return tuple(idx)


# ============================================================
# 第3部分: 正交多项式基
# ============================================================

def hermite_polynomial(x, n):
    """
    概率论 Hermite 多项式:
        He₀=1, He₁=x, He_{n+1}=x·He_n - n·He_{n-1}
    正交: ∫ He_m He_n φ dx = n! δ_{mn}, φ=N(0,1)
    """
    x = np.asarray(x, dtype=np.float64)
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return x.copy()
    h2, h1 = np.ones_like(x), x.copy()
    for k in range(1, n):
        h_curr = x * h1 - k * h2
        h2, h1 = h1, h_curr
    return h1


def legendre_polynomial(x, n):
    """
    Legendre 多项式:
        P₀=1, P₁=x, (n+1)P_{n+1}=(2n+1)x·P_n - n·P_{n-1}
    正交: ∫_{-1}^1 P_m P_n dx = 2/(2n+1) δ_{mn}
    """
    x = np.asarray(x, dtype=np.float64)
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return x.copy()
    p2, p1 = np.ones_like(x), x.copy()
    for k in range(1, n):
        p_curr = ((2 * k + 1) * x * p1 - k * p2) / (k + 1)
        p2, p1 = p1, p_curr
    return p1


def laguerre_polynomial(x, n):
    """
    Laguerre 多项式:
        L₀=1, L₁=1-x, (n+1)L_{n+1}=(2n+1-x)L_n - n·L_{n-1}
    正交: ∫₀^∞ L_m L_n e^{-x} dx = δ_{mn}
    """
    x = np.asarray(x, dtype=np.float64)
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 1.0 - x
    l2, l1 = np.ones_like(x), 1.0 - x
    for k in range(1, n):
        l_curr = ((2 * k + 1 - x) * l1 - k * l2) / (k + 1)
        l2, l1 = l1, l_curr
    return l1


def evaluate_pce_basis(multi_index, xi, distribution='gauss'):
    """
    PCE 基函数: Ψ_α(ξ) = Π φ_{αₖ}(ξₖ)
    """
    xi = np.asarray(xi, dtype=np.float64)
    d = len(multi_index)
    val = 1.0
    for k in range(d):
        n_k = multi_index[k]
        if n_k == 0:
            continue
        xk = np.array([xi[k]])
        if distribution == 'gauss':
            val *= hermite_polynomial(xk, n_k)[0]
        elif distribution == 'uniform':
            val *= legendre_polynomial(xk, n_k)[0]
        elif distribution == 'gamma':
            val *= laguerre_polynomial(xk, n_k)[0]
        else:
            val *= hermite_polynomial(xk, n_k)[0]
    return val


def evaluate_pce_basis_batch(multi_indices, xi_batch, distribution='gauss'):
    """
    批量 PCE 基: Ψ[i,j] = Ψ_{α_i}(ξ_j)
    返回: (P, N)
    """
    P = len(multi_indices)
    N = len(xi_batch)
    Psi = np.zeros((P, N))
    for p_idx in range(P):
        for n_idx in range(N):
            Psi[p_idx, n_idx] = evaluate_pce_basis(
                multi_indices[p_idx], xi_batch[n_idx], distribution
            )
    return Psi


# ============================================================
# 第4部分: PCE 系数计算 (配点法)
# ============================================================

def compute_pce_coefficients_collocation(multi_indices, collocation_points,
                                         function_values, distribution='gauss'):
    """
    配点法 PCE: 最小二乘
        Ψ·c = u → c = (ΨΨᵀ)^{-1} Ψu
    返回: coefficients (P,), condition_number
    """
    Psi = evaluate_pce_basis_batch(multi_indices, collocation_points, distribution)
    # Psi: (P, N), 正规方程: (Psi·Psiᵀ)·c = Psi·u
    A = Psi @ Psi.T  # (P, P)
    rhs = Psi @ function_values  # (P,)
    cond = np.linalg.cond(A)
    reg = max(1e-12, 1e-14 * cond)
    A += reg * np.eye(len(A))
    coefficients = np.linalg.solve(A, rhs)
    return coefficients, cond


# ============================================================
# 第5部分: PCE 统计矩提取
# ============================================================

def pce_statistics(coefficients, multi_indices, distribution='gauss'):
    """
    PCE 直接提取统计矩:
        E[u]  = c₀
        Var[u] = Σ_{α≠0} c_α²·||Ψ_α||²
        Sobol 主效应: S_k = (Σ_{α: only α_k>0} c_α²||Ψ_α||²) / Var[u]
    """
    P = len(multi_indices)
    d = len(multi_indices[0])
    coefs = np.asarray(coefficients).ravel()[:P]
    mean = coefs[0] if P > 0 else 0.0
    variance = 0.0
    sobol = np.zeros(d)
    for p in range(1, P):
        alpha = multi_indices[p]
        if distribution == 'gauss':
            norm_sq = 1.0
            for k in range(d):
                norm_sq *= factorial(int(alpha[k]))
        elif distribution == 'uniform':
            norm_sq = 1.0
            for k in range(d):
                norm_sq *= 2.0 / (2 * int(alpha[k]) + 1)
        else:
            norm_sq = 1.0
        c_val = coefs[p] if p < len(coefs) else 0.0
        variance += c_val ** 2 * norm_sq
        nonzero_dims = [k for k in range(d) if alpha[k] > 0]
        if len(nonzero_dims) == 1:
            sobol[nonzero_dims[0]] += c_val ** 2 * norm_sq
    variance = max(variance, 1e-30)
    return mean, variance, np.sqrt(variance), sobol / variance


# ============================================================
# 第6部分: PCE 系数 SVD 分析
# (映射自 180_circle_map: SVD 椭圆映射)
# ============================================================

def pce_coefficient_svd(coefficients_matrix):
    """
    PCE 系数矩阵 SVD: C = UΣVᵀ
    椭圆长短轴比 σ₁/σ₂ → 系数空间各向异性
    (circle_map 的 SVD 椭圆映射思想)
    """
    U, S, Vt = np.linalg.svd(coefficients_matrix, full_matrices=False)
    if len(S) >= 2 and S[1] > 1e-15:
        aspect_ratio = S[0] / S[1]
    else:
        aspect_ratio = float('inf') if S[0] > 1e-15 else 1.0
    total_energy = np.sum(S ** 2)
    energy_top2 = np.sum(S[:min(2, len(S))] ** 2) / total_energy if total_energy > 1e-30 else 0.0
    return U, S, Vt, aspect_ratio, energy_top2
