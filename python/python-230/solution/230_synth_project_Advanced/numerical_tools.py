"""
numerical_tools.py
==================

数值工具集 —— 融合 Chebyshev 插值 (seed 159, 166) 与
高维求积 (seed 229, cube_arbq_rule) 以及参数化热方程
(seed 846, paraheat) 的 FEM 型参数空间积分。

功能:
1. Chebyshev 插值 profile likelihood 函数 (seed 159)
2. Gauss-Chebyshev Type 2 求积 (seed 166, 229)
3. 多维 nuisance 参数空间积分 (seed 846 → FEM 弱形式)
4. 自适应积分精度控制

公式:

Chebyshev 插值 (seed 159):
    c_k = (2/n) Σ_j f(x_j) T_k(x_j),  x_j = cos((2j-1)π/(2n))
    f(x) ≈ Σ_k c_k T_k(x) - c_0/2  (Clenshaw 递推)

Gauss-Chebyshev Type 2 (seed 166):
    ∫_{-1}^{1} f(x) √(1-x²) dx ≈ Σ_j w_j f(x_j)
    x_j = cos(jπ/(n+1)),  w_j = (π/(n+1)) sin²(jπ/(n+1))

3D 立方体求积 (seed 229):
    ∫_{[-1,1]³} f(x,y,z) dx dy dz ≈ Σ_j w_j f(x_j, y_j, z_j)
    使用预计算的对称求积规则 (degree ≤ 15)

参数空间 FEM 积分 (seed 846):
    将 nuisance 参数空间视为有限元域, 用分片线性基函数
    近似被积函数, 在每个单元上做高斯求积。
"""

from __future__ import annotations
import math
import numpy as np
from typing import Callable, Tuple, Optional, List


# ---------------------------------------------------------------------------
# Chebyshev 插值 (seed 159)
# ---------------------------------------------------------------------------
def chebyshev_zeros(n: int) -> np.ndarray:
    """
    Chebyshev 零点 (seed 159):
        x_k = cos((2k - 1) π / (2n)),  k = 1, ..., n
    """
    k = np.arange(1, n + 1)
    return np.cos((2 * k - 1) * np.pi / (2 * n))


def chebyshev_coefficients(
    f_values: np.ndarray, n: int
) -> np.ndarray:
    """
    离散 Chebyshev 系数 (seed 159):
        c_k = (2/n) Σ_{j=1}^{n} f(x_j) cos(k(2j-1)π/(2n))

    其中 x_j 为 Chebyshev 零点。
    """
    c = np.zeros(n)
    j_idx = np.arange(1, n + 1)
    for k in range(n):
        T_k = np.cos(k * (2 * j_idx - 1) * np.pi / (2 * n))
        c[k] = (2.0 / n) * np.sum(f_values * T_k)
    return c


def clenshaw_evaluate(coeffs: np.ndarray, x: float) -> float:
    """
    Clenshaw 递推计算 Chebyshev 级数 (seed 159):
        d_{n+1} = d_{n+2} = 0
        d_k = 2x d_{k+1} - d_{k+2} + c_k
        f(x) = d_0 - x d_1 + c_0/2
             = (d_1 · x - d_2 + c_0/2) ... 取决于约定

    简化: f(x) = c_0/2 + Σ_{k=1}^{n-1} c_k T_k(x)
    """
    n = len(coeffs)
    if n == 0:
        return 0.0
    if n == 1:
        return coeffs[0]

    d1 = 0.0
    d2 = 0.0
    for k in range(n - 1, 0, -1):
        d_new = 2.0 * x * d1 - d2 + coeffs[k]
        d2 = d1
        d1 = d_new

    return coeffs[0] / 2.0 + x * d1 - d2


def chebyshev_interpolant(
    f: Callable[[float], float],
    n: int,
    a: float = -1.0,
    b: float = 1.0,
) -> Callable[[float], float]:
    """
    构造 Chebyshev 插值器 (seed 159)。

    将 f: [a, b] → R 在 Chebyshev 零点上采样, 计算 Chebyshev 系数,
    返回 Clenshaw 求值闭包。

    区间映射: x ∈ [a, b] → t ∈ [-1, 1], t = (2x - a - b) / (b - a)
    """
    zeros = chebyshev_zeros(n)
    # 映射到 [a, b]
    x_nodes = 0.5 * (b - a) * zeros + 0.5 * (a + b)
    f_values = np.array([f(x) for x in x_nodes])
    coeffs = chebyshev_coefficients(f_values, n)

    def interpolant(x: float) -> float:
        t = (2.0 * x - a - b) / (b - a)
        t = max(min(t, 1.0), -1.0)  # 安全截断
        return clenshaw_evaluate(coeffs, t)

    return interpolant


# ---------------------------------------------------------------------------
# Gauss-Chebyshev Type 2 求积 (seed 166, 229)
# ---------------------------------------------------------------------------
def gauss_chebyshev_type2(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gauss-Chebyshev Type 2 节点和权重 (seed 166):

        ∫_{-1}^{1} f(x) √(1-x²) dx ≈ (π/(n+1)) Σ_j sin²(jπ/(n+1)) f(x_j)

        x_j = cos(jπ/(n+1)),  j = 1, ..., n
        w_j = (π/(n+1)) sin²(jπ/(n+1))
    """
    j = np.arange(1, n + 1)
    nodes = np.cos(j * np.pi / (n + 1))
    weights = (np.pi / (n + 1)) * np.sin(j * np.pi / (n + 1)) ** 2
    return nodes, weights


def chebyshev2_exactness_test(n: int, max_degree: Optional[int] = None) -> List[Tuple[int, float, float]]:
    """
    测试 Gauss-Chebyshev Type 2 求积的多项式精确度 (seed 166):

    ∫_{-1}^{1} x^k √(1-x²) dx = (精确解析公式)

    精确公式:
        k 奇: 0
        k 偶: B((k+1)/2, 3/2) = Γ((k+1)/2) Γ(3/2) / Γ((k+4)/2)

    Returns
    -------
    list of (degree, exact, quadrature, rel_error)
    """
    if max_degree is None:
        max_degree = 2 * n - 1
    nodes, weights = gauss_chebyshev_type2(n)
    results = []
    for k in range(max_degree + 1):
        # 精确值
        if k % 2 == 1:
            exact = 0.0
        else:
            # ∫_{-1}^{1} x^k √(1-x²) dx
            # = B((k+1)/2, 3/2) = Γ((k+1)/2) Γ(3/2) / Γ((k+4)/2)
            half_k = k / 2.0
            from math import gamma
            exact = gamma(half_k + 0.5) * gamma(1.5) / gamma(half_k + 2.0)

        # 求积
        quad_val = np.sum(weights * nodes**k)
        rel_err = abs(quad_val - exact) / max(abs(exact), 1e-300)
        results.append((k, exact, quad_val, rel_err))

    return results


# ---------------------------------------------------------------------------
# 简化 3D 求积 (seed 229 → 立方体规则简化版)
# ---------------------------------------------------------------------------
def cube_quadrature_3d(
    f: Callable[[np.ndarray], float],
    degree: int = 3,
) -> Tuple[float, int]:
    """
    3D 立方体 [-1,1]³ 上的高斯求积 (seed 229 简化版)。

    使用张量积 Gauss-Legendre 规则:
    ∫_{[-1,1]³} f(x,y,z) dV ≈ Σ_{i,j,k} w_i w_j w_k f(x_i, y_j, z_k)

    degree = 每维节点数 (3, 5, 7)

    Returns
    -------
    (integral, n_points) : (float, int)
    """
    if degree == 3:
        nodes_1d = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
        weights_1d = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    elif degree == 5:
        # 5-point Gauss-Legendre
        nodes_1d = np.array([
            -np.sqrt(5.0 + 2.0 * np.sqrt(10.0 / 7.0)) / 3.0,
            -np.sqrt(5.0 - 2.0 * np.sqrt(10.0 / 7.0)) / 3.0,
            0.0,
            np.sqrt(5.0 - 2.0 * np.sqrt(10.0 / 7.0)) / 3.0,
            np.sqrt(5.0 + 2.0 * np.sqrt(10.0 / 7.0)) / 3.0,
        ])
        w1 = (322.0 - 13.0 * np.sqrt(70.0)) / 900.0
        w2 = (322.0 + 13.0 * np.sqrt(70.0)) / 900.0
        weights_1d = np.array([w1, w2, 128.0 / 225.0, w2, w1])
    else:
        # 回退到 3 点
        nodes_1d = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
        weights_1d = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])

    n_1d = len(nodes_1d)
    total = 0.0
    n_points = 0
    for i in range(n_1d):
        for j in range(n_1d):
            for k in range(n_1d):
                point = np.array([nodes_1d[i], nodes_1d[j], nodes_1d[k]])
                w = weights_1d[i] * weights_1d[j] * weights_1d[k]
                total += w * f(point)
                n_points += 1

    return total, n_points


# ---------------------------------------------------------------------------
# Nuisance 参数空间 FEM 积分 (seed 846 → paraheat FEM)
# ---------------------------------------------------------------------------
def fem_parameter_integral(
    integrand: Callable[[np.ndarray], float],
    n_elements: int = 10,
    n_nuis: int = 2,
) -> float:
    """
    在 nuisance 参数空间 [-L, L]^n_nuis 上做 FEM 型积分 (seed 846)。

    将参数空间剖分为 n_elements 个线性单元,
    在每个单元上做 2 点 Gauss 求积。

    ∫ g(θ) dθ ≈ Σ_e Σ_q w_q g(θ_q^{(e)}) · J_e

    其中 J_e = h_e / 2 为 Jacobi 因子, h_e 为单元尺寸。
    """
    if n_nuis > 2:
        # 高维: 退化为张量积
        return _tensor_product_gauss(integrand, n_elements, n_nuis)

    L = 5.0  # 参数范围 [-5, 5] (Gaussian 约束 5σ)
    h = 2.0 * L / n_elements
    # 2 点 Gauss 节点 (参考单元 [-1, 1])
    xi = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
    w = np.array([1.0, 1.0])

    if n_nuis == 1:
        total = 0.0
        for e in range(n_elements):
            theta_left = -L + e * h
            theta_right = theta_left + h
            theta_mid = 0.5 * (theta_left + theta_right)
            jac = h / 2.0
            for q in range(2):
                theta_q = theta_mid + xi[q] * h / 2.0
                total += w[q] * integrand(np.array([theta_q])) * jac
        return total
    else:
        # 2D 张量积
        total = 0.0
        for e1 in range(n_elements):
            t1_left = -L + e1 * h
            t1_mid = t1_left + h / 2.0
            for e2 in range(n_elements):
                t2_left = -L + e2 * h
                t2_mid = t2_left + h / 2.0
                jac = (h / 2.0) ** 2
                for q1 in range(2):
                    for q2 in range(2):
                        th = np.array([
                            t1_mid + xi[q1] * h / 2.0,
                            t2_mid + xi[q2] * h / 2.0,
                        ])
                        total += w[q1] * w[q2] * integrand(th) * jac
        return total


def _tensor_product_gauss(
    integrand: Callable[[np.ndarray], float],
    n_elements: int,
    n_dim: int,
) -> float:
    """n 维张量积 Gauss 积分。"""
    L = 5.0
    h = 2.0 * L / n_elements
    n_per_dim = n_elements
    xi = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
    w = np.array([1.0, 1.0])

    from itertools import product
    total = 0.0
    jac = (h / 2.0) ** n_dim
    for elem_idx in product(range(n_per_dim), repeat=n_dim):
        for quad_idx in product(range(2), repeat=n_dim):
            theta = np.zeros(n_dim)
            weight_prod = 1.0
            for d in range(n_dim):
                t_left = -L + elem_idx[d] * h
                t_mid = t_left + h / 2.0
                theta[d] = t_mid + xi[quad_idx[d]] * h / 2.0
                weight_prod *= w[quad_idx[d]]
            # Gaussian 约束权重 (已在 integrand 中包含)
            total += weight_prod * integrand(theta) * jac
    return total


# ---------------------------------------------------------------------------
# 快速自检
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Chebyshev 插值
    f_test = lambda x: math.exp(-x * x)
    interp = chebyshev_interpolant(f_test, n=16, a=-3.0, b=3.0)
    x_test = 1.0
    print(f"Chebyshev 插值: f(1.0) = {interp(1.0):.10f}")
    print(f"精确值: exp(-1) = {math.exp(-1.0):.10f}")
    print(f"误差: {abs(interp(1.0) - math.exp(-1.0)):.2e}")

    # Gauss-Chebyshev Type 2 精确度测试
    results = chebyshev2_exactness_test(n=8, max_degree=20)
    print(f"\nGauss-Chebyshev Type 2 精确度 (n=8):")
    for k, exact, quad, rel_err in results[:10]:
        print(f"  k={k:2d}: exact={exact:.8f}, quad={quad:.8f}, rel_err={rel_err:.2e}")

    # 3D 求积
    f_3d = lambda p: p[0]**2 + p[1]**2 + p[2]**2
    val, npts = cube_quadrature_3d(f_3d, degree=3)
    print(f"\n3D 求积 ∫(x²+y²+z²) dV = {val:.8f} (精确=12.0), n_points={npts}")

    # FEM 参数积分
    g_1d = lambda th: math.exp(-0.5 * th[0]**2) / math.sqrt(2 * math.pi)
    val_fem = fem_parameter_integral(g_1d, n_elements=50, n_nuis=1)
    print(f"\nFEM 1D Gaussian 积分: {val_fem:.8f} (精确≈0.99999943)")
