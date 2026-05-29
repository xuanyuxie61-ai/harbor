"""
spectral_methods.py
移位勒让德多项式谱方法模块

融入种子项目:
  - 666_legendre_shifted_polynomial: 移位勒让德多项式 P_n^*(x) = P_n(2x-1)

功能:
  - 移位勒让德多项式计算（递推关系）
  - Gauss-Legendre 求积节点与权重
  - 谱时间离散化（用于蠕变演化方程）
  - 函数的谱投影与重构
"""

import numpy as np
from typing import Tuple, Optional


def shifted_legendre_polynomial(
    x: np.ndarray, n_max: int
) -> np.ndarray:
    """
    计算移位勒让德多项式 P_n^*(x) = P_n(2x-1) 在点 x 处的值。

    移位勒让德多项式定义在 [0, 1] 上，满足正交性:
        \\\\int_0^1 P_m^*(x) P_n^*(x) dx = \\frac{1}{2n+1} \\delta_{mn}

    递推关系:
        P_0^*(x) = 1
        P_1^*(x) = 2x - 1
        (n+1) P_{n+1}^*(x) = (2n+1)(2x-1) P_n^*(x) - n P_{n-1}^*(x)

    参数:
        x: 求值点数组，形状 (m,)
        n_max: 最高阶数

    返回:
        值数组，形状 (m, n_max+1)
    """
    x = np.asarray(x)
    m = x.shape[0]

    if n_max < 0:
        return np.zeros((m, 0))

    v = np.zeros((m, n_max + 1))
    v[:, 0] = 1.0

    if n_max < 1:
        return v

    v[:, 1] = 2.0 * x - 1.0

    for i in range(2, n_max + 1):
        v[:, i] = (
            (2 * i - 1) * (2.0 * x - 1.0) * v[:, i - 1]
            - (i - 1) * v[:, i - 2]
        ) / i

    return v


def shifted_legendre_derivative(
    x: np.ndarray, n_max: int
) -> np.ndarray:
    """
    计算移位勒让德多项式的导数 dP_n^*/dx。

    利用标准勒让德多项式的导数关系:
        \\frac{d}{dx} P_n^*(x) = 2 P_n'(2x-1)

    参数:
        x: 求值点数组
        n_max: 最高阶数

    返回:
        导数值数组，形状 (m, n_max+1)
    """
    x = np.asarray(x)
    m = x.shape[0]

    if n_max < 0:
        return np.zeros((m, 0))

    # 先计算在 t=2x-1 处的标准勒让德多项式及其导数
    t = 2.0 * x - 1.0
    v = np.zeros((m, n_max + 1))
    dv = np.zeros((m, n_max + 1))

    v[:, 0] = 1.0
    dv[:, 0] = 0.0

    if n_max >= 1:
        v[:, 1] = t
        dv[:, 1] = 1.0

    for i in range(2, n_max + 1):
        v[:, i] = ((2 * i - 1) * t * v[:, i - 1] - (i - 1) * v[:, i - 2]) / i
        dv[:, i] = ((2 * i - 1) * (v[:, i - 1] + t * dv[:, i - 1]) - (i - 1) * dv[:, i - 2]) / i

    # 链式法则: d/dx P_n^*(x) = d/dx P_n(2x-1) = 2 P_n'(2x-1)
    return 2.0 * dv


def gauss_legendre_nodes_weights(n: int, domain: Tuple[float, float] = (0.0, 1.0)) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 n 点 Gauss-Legendre 求积的节点和权重。

    标准 Gauss-Legendre 求积公式:
        \\\\int_{-1}^{1} f(x) dx \\approx \\\sum_{i=1}^n w_i f(x_i)

    其中 x_i 为 n 阶勒让德多项式 P_n(x) 的根，权重为:
        w_i = \\frac{2}{(1 - x_i^2) [P_n'(x_i)]^2}

    参数:
        n: 求积点数
        domain: 目标积分区间 (a, b)

    返回:
        (nodes, weights) 在目标区间上
    """
    if n < 1:
        return np.array([]), np.array([])

    # 使用 numpy 的 legendre 多项式功能
    # 计算标准区间 [-1, 1] 上的节点和权重
    nodes, weights = np.polynomial.legendre.leggauss(n)

    # 变换到目标区间 [a, b]
    a, b = domain
    nodes = 0.5 * (b - a) * nodes + 0.5 * (b + a)
    weights = 0.5 * (b - a) * weights

    return nodes, weights


def spectral_projection(
    f, n_modes: int, n_quad: int = None
) -> np.ndarray:
    """
    将函数 f 投影到移位勒让德基上，得到谱系数。

    对于函数 f(x) 在 [0, 1] 上，其谱展开为:
        f(x) = \\\sum_{n=0}^{N} c_n P_n^*(x)

    系数通过正交性得到:
        c_n = (2n+1) \\\\int_0^1 f(x) P_n^*(x) dx

    参数:
        f: 函数，接受数组返回数组
        n_modes: 模态数
        n_quad: 求积点数（默认 2*n_modes）

    返回:
        谱系数数组 c_n
    """
    if n_quad is None:
        n_quad = 2 * n_modes

    nodes, weights = gauss_legendre_nodes_weights(n_quad, domain=(0.0, 1.0))
    poly_vals = shifted_legendre_polynomial(nodes, n_modes)

    f_vals = f(nodes)
    coeffs = np.zeros(n_modes + 1)
    for n in range(n_modes + 1):
        coeffs[n] = (2 * n + 1) * np.sum(weights * f_vals * poly_vals[:, n])

    return coeffs


def spectral_reconstruct(
    coeffs: np.ndarray, x: np.ndarray
) -> np.ndarray:
    """
    由谱系数重构函数值。

        f(x) = \\\sum_{n=0}^{N} c_n P_n^*(x)

    参数:
        coeffs: 谱系数数组
        x: 求值点

    返回:
        函数值数组
    """
    n_max = len(coeffs) - 1
    poly_vals = shifted_legendre_polynomial(x, n_max)
    return poly_vals @ coeffs


def spectral_derivative_matrix(n: int, domain: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """
    构造移位勒让德多项式基下的谱微分矩阵。

    在配点 {x_j}_{j=0}^n 上，微分矩阵 D 满足:
        f'(x_i) = \\\sum_{j=0}^n D_{ij} f(x_j)

    参数:
        n: 多项式阶数
        domain: 区间

    返回:
        微分矩阵 (n+1, n+1)
    """
    # 使用 Gauss-Lobatto 点作为配点（包含端点）
    # 先计算标准 [-1,1] 上的 Gauss-Lobatto 点
    # 使用 Chebyshev 点作为近似
    x_cheb = np.cos(np.pi * np.arange(n + 1) / n)
    # 变换到 [0,1]
    a, b = domain
    x_nodes = 0.5 * (b - a) * (x_cheb + 1) + a

    # 构造微分矩阵（基于拉格朗日插值）
    D = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(n + 1):
            if i != j:
                # 拉格朗日基函数的导数
                prod = 1.0
                for k in range(n + 1):
                    if k != i and k != j:
                        prod *= (x_nodes[i] - x_nodes[k]) / (x_nodes[j] - x_nodes[k])
                D[i, j] = prod / (x_nodes[j] - x_nodes[i])
            else:
                # 对角元
                s = 0.0
                for k in range(n + 1):
                    if k != i:
                        s += 1.0 / (x_nodes[i] - x_nodes[k])
                D[i, i] = s

    return D


def integrate_with_legendre_expansion(
    coeffs: np.ndarray
) -> float:
    """
    利用谱系数计算定积分。

    由于 \\\\int_0^1 P_n^*(x) dx = 0 (n>0) 且 P_0^*(x)=1:
        \\\\int_0^1 f(x) dx = c_0

    参数:
        coeffs: 谱系数

    返回:
        积分值
    """
    return float(coeffs[0]) if len(coeffs) > 0 else 0.0


def convolution_legendre_kernel(
    kernel_func, n_modes: int, t: float
) -> np.ndarray:
    """
    计算核函数在移位勒让德基下的卷积矩阵。

    对于蠕变问题中的积分:
        \\\\int_0^t K(t-s) f(s) ds

    使用谱展开后，可转化为矩阵-向量乘积。

    参数:
        kernel_func: 核函数 K(tau)
        n_modes: 模态数
        t: 时间上限

    返回:
        卷积矩阵 (n_modes+1, n_modes+1)
    """
    n = n_modes + 1
    nodes, weights = gauss_legendre_nodes_weights(n, domain=(0.0, t))
    poly_vals = shifted_legendre_polynomial(nodes / t, n_modes)  # 缩放到 [0,1]

    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            # 积分 K(t-s) P_j^*(s/t) P_i^*(s/t) ds
            s = nodes
            tau = t - s
            K_vals = kernel_func(tau)
            C[i, j] = np.sum(weights * K_vals * poly_vals[:, i] * poly_vals[:, j])

    return C
