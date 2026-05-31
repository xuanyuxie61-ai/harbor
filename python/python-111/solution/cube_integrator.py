"""
三维立方体/构象空间高斯求积模块
基于 cube_felippa_rule 核心算法：一维 Gauss-Legendre 规则、张量积构造、
组合枚举测试积分精度。

在蛋白质折叠中的应用：
- 在构象空间子域计算局部配分函数
- 体积积分计算蛋白质占据体积
- 自由能微扰 (FEP) 中的 λ-依赖势能积分
- 周期性盒子中的 Ewald 实空间分割积分

数学基础:
    一维 Gauss-Legendre 规则精确积分 2n-1 次多项式:
        ∫_{-1}^{1} f(x) dx ≈ Σ w_i f(x_i)
    
    三维张量积规则:
        ∫∫∫ f(x,y,z) dxdydz ≈ Σ_i Σ_j Σ_k W_{ijk} f(x_i, y_j, z_k)
        W_{ijk} = w_i * w_j * w_k * (V/8)
    
    其中 V = (bx-ax)*(by-ay)*(bz-az) 为区域体积。
"""

import numpy as np
from typing import Tuple, List
from itertools import combinations_with_replacement


def line_unit_o05() -> Tuple[np.ndarray, np.ndarray]:
    """
    一维 5 点 Gauss-Legendre 规则 (精确积分 9 次多项式)。
    
    节点和权重为 [-1,1] 上标准值。
    
    Returns
    -------
    x : np.ndarray, shape (5,)
        积分节点。
    w : np.ndarray, shape (5,)
        积分权重。
    """
    x = np.array([
        -0.9061798459386640,
        -0.5384693101056831,
         0.0,
         0.5384693101056831,
         0.9061798459386640
    ])
    w = np.array([
        0.2369268850561891,
        0.4786286704993665,
        0.5688888888888889,
        0.4786286704993665,
        0.2369268850561891
    ])
    return x, w


def line_unit_o03() -> Tuple[np.ndarray, np.ndarray]:
    """
    一维 3 点 Gauss-Legendre 规则 (精确积分 5 次多项式)。
    
    Returns
    -------
    x : np.ndarray, shape (3,)
        积分节点。
    w : np.ndarray, shape (3,)
        积分权重。
    """
    x = np.array([-np.sqrt(3.0/5.0), 0.0, np.sqrt(3.0/5.0)])
    w = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])
    return x, w


def transform_interval(xi: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    将 [-1, 1] 上的节点映射到 [a, b]。
    
    变换公式:
        x = 0.5*(b-a)*xi + 0.5*(a+b)
    
    Parameters
    ----------
    xi : np.ndarray
        [-1, 1] 上的节点。
    a, b : float
        目标区间。
    
    Returns
    -------
    x : np.ndarray
        映射后的节点。
    """
    return 0.5 * (b - a) * xi + 0.5 * (a + b)


def cube_rule(ax: float, bx: float, ay: float, by: float, az: float, bz: float,
              order_1d: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    构造三维立方体区域上的张量积 Gauss-Legendre 求积规则。
    
    Parameters
    ----------
    ax, bx : float
        x 方向区间边界。
    ay, by : float
        y 方向区间边界。
    az, bz : float
        z 方向区间边界。
    order_1d : int
        一维规则点数，支持 3 或 5。
    
    Returns
    -------
    nodes : np.ndarray, shape (N, 3)
        三维求积节点。
    weights : np.ndarray, shape (N,)
        对应权重。
    """
    if order_1d == 3:
        xi, wi = line_unit_o03()
    elif order_1d == 5:
        xi, wi = line_unit_o05()
    else:
        raise ValueError("Only order_1d = 3 or 5 is supported")
    
    x_nodes = transform_interval(xi, ax, bx)
    y_nodes = transform_interval(xi, ay, by)
    z_nodes = transform_interval(xi, az, bz)
    
    volume = (bx - ax) * (by - ay) * (bz - az)
    scale = volume / 8.0
    
    # 张量积构造
    n = order_1d
    nodes = np.zeros((n ** 3, 3))
    weights = np.zeros(n ** 3)
    idx = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                nodes[idx, 0] = x_nodes[i]
                nodes[idx, 1] = y_nodes[j]
                nodes[idx, 2] = z_nodes[k]
                weights[idx] = wi[i] * wi[j] * wi[k] * scale
                idx += 1
    return nodes, weights


def cube_monomial_integral(ax: float, bx: float, ay: float, by: float,
                           az: float, bz: float, alpha: int, beta: int, gamma_exp: int) -> float:
    """
    计算单项式 x^α y^β z^γ 在立方体 [ax,bx]×[ay,by]×[az,bz] 上的解析积分。
    
    解析公式:
        I = [ (bx^{α+1} - ax^{α+1})/(α+1) ]
          * [ (by^{β+1} - ay^{β+1})/(β+1) ]
          * [ (bz^{γ+1} - az^{γ+1})/(γ+1) ]
    
    Parameters
    ----------
    ax, bx, ay, by, az, bz : float
        区间边界。
    alpha, beta, gamma_exp : int
        指数（非负）。
    
    Returns
    -------
    integral : float
        解析积分值。
    """
    if alpha < 0 or beta < 0 or gamma_exp < 0:
        raise ValueError("Exponents must be non-negative")
    
    def power_integral(a, b, p):
        if p == -1:
            return np.log(b / a) if a * b > 0 else 0.0
        return (b ** (p + 1) - a ** (p + 1)) / (p + 1)
    
    Ix = power_integral(ax, bx, alpha)
    Iy = power_integral(ay, by, beta)
    Iz = power_integral(az, bz, gamma_exp)
    return float(Ix * Iy * Iz)


def integrate_partition_function_subdomain(coords_min: np.ndarray, coords_max: np.ndarray,
                                           potential_func: callable,
                                           kT: float = 1.0,
                                           order_1d: int = 5) -> float:
    """
    在构象空间子域上计算局部配分函数（体积积分）。
    
    配分函数定义:
        Z = ∫_V exp( -U(r) / (k_B T) ) d^3r
    
    其中 U(r) 为势能函数，V 为构象空间子域。
    
    Parameters
    ----------
    coords_min : np.ndarray, shape (3,)
        子域最小坐标。
    coords_max : np.ndarray, shape (3,)
        子域最大坐标。
    potential_func : callable
        势能函数，输入 (N, 3) 返回 (N,)。
    kT : float
        热能量。
    order_1d : int
        一维积分阶数。
    
    Returns
    -------
    Z : float
        局部配分函数值。
    """
    ax, ay, az = coords_min
    bx, by, bz = coords_max
    nodes, weights = cube_rule(ax, bx, ay, by, az, bz, order_1d)
    energies = potential_func(nodes)
    boltzmann = np.exp(-energies / kT)
    Z = float(np.sum(boltzmann * weights))
    return Z


def comp_next(n: int, k: int) -> List[Tuple[int, ...]]:
    """
    生成整数 n 的所有 k 部分组合 (compositions)。
    
    组合定义: 满足 n1 + n2 + ... + nk = n 且 ni >= 0 的所有有序 k 元组。
    
    Parameters
    ----------
    n : int
        目标整数。
    k : int
        部分数。
    
    Returns
    -------
    compositions : list of tuples
        所有组合。
    """
    if k == 1:
        return [(n,)]
    result = []
    for i in range(n + 1):
        for tail in comp_next(n - i, k - 1):
            result.append((i,) + tail)
    return result


def test_cube_rule_precision(ax: float, bx: float, ay: float, by: float,
                             az: float, bz: float, max_degree: int = 4) -> dict:
    """
    测试三维张量积求积规则对单项式的精确度。
    
    对于 order_1d=3 (精确到5次)，max_degree 应 <= 4。
    对于 order_1d=5 (精确到9次)，max_degree 应 <= 8。
    
    Parameters
    ----------
    ax, bx, ay, by, az, bz : float
        测试区域。
    max_degree : int
        测试最高总次数。
    
    Returns
    -------
    errors : dict
        键为 (alpha, beta, gamma)，值为 |数值积分 - 解析积分|。
    """
    errors = {}
    for order_1d in [3, 5]:
        for total_deg in range(max_degree + 1):
            for comp in comp_next(total_deg, 3):
                alpha, beta, gamma_exp = comp
                nodes, weights = cube_rule(ax, bx, ay, by, az, bz, order_1d)
                fvals = (nodes[:, 0] ** alpha) * (nodes[:, 1] ** beta) * (nodes[:, 2] ** gamma_exp)
                num_int = np.sum(fvals * weights)
                ana_int = cube_monomial_integral(ax, bx, ay, by, az, bz, alpha, beta, gamma_exp)
                key = (order_1d, alpha, beta, gamma_exp)
                errors[key] = abs(num_int - ana_int)
    return errors
