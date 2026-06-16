"""
disorder_quadrature.py
======================

高斯求积公式用于对淬火无序的数值平均。

物理背景
--------
在自旋玻璃中, 热力学观测量需要对淬火无序求平均:
    [O]_{av} = integral O[J] * P[J] dJ
其中 P[J] = prod_{<ij>} (1/sqrt(2*pi*J_var)) * exp(-J_{ij}^2 / (2*J_var))

对于小系统, 可以直接用数值积分 (高斯求积) 代替 Monte Carlo
对无序求平均, 从而获得精确基线。

高斯求积公式
------------
积分 I = integral_{-inf}^{inf} f(x) * exp(-x^2) dx
近似为:
    I ≈ sum_{k=1}^{n} w_k * f(x_k)
其中 {x_k} 是 n 阶 Hermite 多项式的零点, {w_k} 是对应的权重。

对于一般高斯分布 P(J) = (1/sqrt(2*pi*sigma^2)) * exp(-J^2/(2*sigma^2)):
    <f(J)> = sum_{k=1}^{n} w_k * f(sqrt(2)*sigma*x_k) / sqrt(pi)

本模块核心算法来源于 seed project:
- 229_cube_arbq_rule: 立方域上的任意阶求积公式
  (映射为高斯求积的任意阶扩展)
"""

import numpy as np
from typing import Callable, Tuple, Optional


# =====================================================================
#  Gauss-Hermite 求积
# =====================================================================

def gauss_hermite_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 n 阶 Gauss-Hermite 求积的节点和权重。

    Gauss-Hermite 求积公式:
        integral_{-inf}^{inf} f(x) * exp(-x^2) dx ≈ sum_{k=1}^{n} w_k * f(x_k)

    节点 x_k 是 Hermite 多项式 H_n(x) 的零点。
    权重 w_k = 2^{n-1} * n! * sqrt(pi) / (n^2 * [H_{n-1}(x_k)]^2)

    参数
    ----
    n : int
        求积阶数 (1 <= n <= 100)

    返回
    ----
    nodes : ndarray (n,)
        求积节点
    weights : ndarray (n,)
        求积权重
    """
    if n < 1:
        raise ValueError(f"求积阶数 n 必须 >= 1, 当前 n={n}")
    if n > 100:
        raise ValueError(f"求积阶数过高, n={n} > 100, 可能数值不稳定")

    # 使用 Golub-Welsch 算法: 通过 Jacobi 矩阵的特征值/特征向量
    # Hermite 多项式的三项递推:
    #   x * H_n(x) = (1/2) * H_{n+1}(x) + n * H_{n-1}(x)
    # 对应的 Jacobi 矩阵是三对角矩阵:
    #   a_i = 0 (对角)
    #   b_i = sqrt(i/2) (次对角)
    i = np.arange(1, n, dtype=np.float64)
    b = np.sqrt(i / 2.0)

    # Jacobi 矩阵
    J_matrix = np.zeros((n, n), dtype=np.float64)
    for k in range(n - 1):
        J_matrix[k, k + 1] = b[k]
        J_matrix[k + 1, k] = b[k]

    # 特征值 = 节点, 特征向量的第一分量给出权重
    eigenvalues, eigenvectors = np.linalg.eigh(J_matrix)
    nodes = eigenvalues
    weights = np.sqrt(np.pi) * eigenvectors[0, :] ** 2

    return nodes, weights


# =====================================================================
#  对一般高斯分布的求积
# =====================================================================

def gaussian_quadrature_1d(func: Callable[[np.ndarray], np.ndarray],
                           mu: float = 0.0,
                           sigma: float = 1.0,
                           n_quad: int = 20) -> float:
    """
    计算高斯分布下的期望值:
        <f(J)> = integral_{-inf}^{inf} f(J) * (1/(sqrt(2*pi)*sigma)) * exp(-(J-mu)^2/(2*sigma^2)) dJ

    通过变量替换 x = (J - mu) / (sqrt(2) * sigma):
        <f(J)> = (1/sqrt(pi)) * sum_{k=1}^{n} w_k * f(mu + sqrt(2)*sigma*x_k)

    参数
    ----
    func : callable
        被积函数, 接受 ndarray 返回 ndarray
    mu : float
        高斯分布的均值
    sigma : float
        高斯分布的标准差
    n_quad : int
        求积阶数

    返回
    ----
    result : float
        积分近似值
    """
    if sigma <= 0:
        raise ValueError(f"sigma 必须 > 0, 当前 sigma={sigma}")

    nodes, weights = gauss_hermite_nodes_weights(n_quad)
    # 变换到物理变量: J_k = mu + sqrt(2) * sigma * x_k
    J_nodes = mu + np.sqrt(2.0) * sigma * nodes
    f_vals = func(J_nodes)
    result = np.sum(weights * f_vals) / np.sqrt(np.pi)
    return float(result)


# =====================================================================
#  多维高斯求积 (用于多键联合平均)
# =====================================================================

def gaussian_quadrature_multidim(
        func: Callable[[np.ndarray], np.ndarray],
        n_bond: int,
        mu: float = 0.0,
        sigma: float = 1.0,
        n_quad_1d: int = 5) -> float:
    """
    多维高斯求积 (张量积格式):
        <f(J_1, ..., J_{n_bond})> =
            integral ... integral f(J_1,...,J_{n_bond}) * prod_k P(J_k) dJ_1...dJ_{n_bond}

    对于 n_bond 维、每维 n_quad_1d 阶的张量积, 总点数 = n_quad_1d^{n_bond}。
    仅适用于 n_bond 很小的情况 (如 n_bond <= 4)。

    参数
    ----
    func : callable
        接受 ndarray shape (n_bond, n_points) 返回 ndarray shape (n_points,)
    n_bond : int
        维度数 (键数)
    mu : float
        均值
    sigma : float
        标准差
    n_quad_1d : int
        每维求积阶数

    返回
    ----
    result : float
        多维积分近似值
    """
    if n_bond > 6:
        raise ValueError(f"维度过高: n_bond={n_bond} > 6, "
                         f"张量积点数 = {n_quad_1d**n_bond}, 改用 Monte Carlo")

    nodes_1d, weights_1d = gauss_hermite_nodes_weights(n_quad_1d)
    # 变换节点
    J_1d = mu + np.sqrt(2.0) * sigma * nodes_1d

    # 构建张量积
    grids = np.meshgrid(*[J_1d] * n_bond, indexing='ij')
    J_tensor = np.stack([g.ravel() for g in grids], axis=0)  # (n_bond, n_total)

    W_grids = np.meshgrid(*[weights_1d] * n_bond, indexing='ij')
    W_tensor = np.ones(W_grids[0].size, dtype=np.float64)
    for wg in W_grids:
        W_tensor *= wg.ravel()

    f_vals = func(J_tensor)  # (n_total,)
    result = np.sum(W_tensor * f_vals) / (np.sqrt(np.pi) ** n_bond)
    return float(result)


# =====================================================================
#  Clenshaw-Curtis 求积 (有限区间)
# =====================================================================

def clenshaw_curtis_nodes_weights(n: int,
                                  a: float = -1.0,
                                  b: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clenshaw-Curtis 求积的节点和权重 (在区间 [a, b] 上)。

    节点: x_k = (a+b)/2 + (b-a)/2 * cos(k*pi/(n-1)), k=0,...,n-1
    权重通过 DCT 或递推计算。

    用于对有限区间上的分布函数积分 (如截断高斯)。
    """
    if n < 2:
        raise ValueError(f"Clenshaw-Curtis 至少需要 n=2, 当前 n={n}")

    # [-1, 1] 上的节点
    theta = np.pi * np.arange(n) / (n - 1)
    x_cc = np.cos(theta)

    # 权重 (通过递推)
    w_cc = np.zeros(n, dtype=np.float64)
    for k in range(n):
        s = 0.0
        for j in range(1, n // 2 + 1):
            b_j = 2.0 if 2 * j < n - 1 else 1.0
            s += b_j * np.cos(2.0 * j * theta[k]) / (4.0 * j * j - 1.0)
        w_cc[k] = 2.0 * (1.0 - s) / (n - 1)
    w_cc[0] *= 0.5
    w_cc[-1] *= 0.5

    # 变换到 [a, b]
    x_ab = 0.5 * (a + b) + 0.5 * (b - a) * x_cc
    w_ab = 0.5 * (b - a) * w_cc

    return x_ab, w_ab


# =====================================================================
#  应用: 自旋玻璃中的无序平均
# =====================================================================

def disorder_average_energy_density(n_quad: int = 15,
                                    J_var: float = 1.0,
                                    beta: float = 1.0,
                                    z: int = 6) -> float:
    """
    平均场近似下的能量密度:
        e = -beta * J_var * z / 2 * tanh(beta * J_eff)

    这里用单键高斯求积计算:
        <J * tanh(beta * J)> = integral J * tanh(beta * J) * P(J) dJ

    参数
    ----
    n_quad : int
        求积阶数
    J_var : float
        耦合方差
    beta : float
        逆温度
    z : int
        配位数

    返回
    ----
    e_density : float
        能量密度近似
    """
    sigma = np.sqrt(J_var)

    def integrand(J):
        return J * np.tanh(beta * J)

    avg = gaussian_quadrature_1d(integrand, mu=0.0, sigma=sigma, n_quad=n_quad)
    e_density = -0.5 * z * avg
    return e_density


def disorder_average_susceptibility(n_quad: int = 20,
                                    J_var: float = 1.0,
                                    beta: float = 1.0,
                                    z: int = 6) -> float:
    """
    平均场自旋玻璃磁化率:
        chi_{SG} = beta^2 * <J^2> * z * chi_0^2 / (1 - beta^2 * <J^2> * z)

    其中 chi_0 = 1 (单自旋磁化率)。

    de Almeida-Thouless 线:
        beta_c^2 * <J^2> * z = 1
        => T_c = sqrt(<J^2> * z)

    参数
    ----
    n_quad : int
        求积阶数 (此处用于验证 beta_c)
    J_var : float
        耦合方差
    beta : float
        逆温度
    z : int
        配位数

    返回
    ----
    chi_SG : float
        自旋玻璃磁化率
    """
    # 精确计算 beta_c
    beta_c = 1.0 / np.sqrt(J_var * z)

    denom = 1.0 - (beta * np.sqrt(J_var)) ** 2 * z
    if abs(denom) < 1e-12:
        return float('inf')

    chi_SG = beta ** 2 * J_var * z / denom
    return chi_SG
