"""
多维集体坐标空间的高斯数值积分
================================
融合原始项目:
  - 936_pyramid_rule: 金字塔区域上的高斯积分规则

科学背景:
---------
在核裂变统计模型中，需要计算多维相空间积分:
  Z = ∫ d⁵q · J(q) · exp( -V(q)/T )

其中 J(q) 为集体坐标空间的度量行列式（Jacobian）。
对于五维积分，采用张量积高斯求积规则:
  ∫ f(x) dx ≈ Σ_i w_i f(x_i)

Legendre-Gauss 求积用于有限区间 [a,b]:
  x_i = (b+a)/2 + (b-a)/2 · ξ_i
  w_i = (b-a)/2 · ω_i
其中 (ξ_i, ω_i) 为 [-1,1] 上的标准 Legendre-Gauss 节点与权重。

Jacobi-Gauss 求积用于带有权函数 (1-x)^α(1+x)^β 的积分，
在核物理中用于处理配对能隙坐标 Δ 的积分（Δ>0 带有权重）。

对于五维张量积，总节点数为 n_β₂ × n_β₃ × n_β₄ × n_β₅ × n_Δ。
本模块实现基于 Legendre-Jacobi 混合规则的多维高斯积分。
"""

import numpy as np
from typing import Tuple, Callable


def legendre_gauss_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 [-1,1] 上的 n 点 Legendre-Gauss 节点与权重.
    
    使用 numpy.polynomial.legendre.leggauss 获取精确节点.
    """
    if n < 1:
        raise ValueError("n must be positive")
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return nodes, weights


def jacobi_gauss_nodes_weights(n: int, alpha: float = 0.0, beta: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 [-1,1] 上权函数 w(x)=(1-x)^α(1+x)^β 的 Jacobi-Gauss 节点与权重.
    
    参数:
        n: 点数
        alpha, beta: Jacobi 参数 (>-1)
    """
    if n < 1:
        raise ValueError("n must be positive")
    if alpha <= -1.0 or beta <= -1.0:
        raise ValueError("alpha and beta must be > -1")
    nodes, weights = np.polynomial.legendre.leggauss(n)
    # 使用 numpy 的 jacobi 实现（通过转换）
    # numpy 不直接提供 jacobi，这里使用近似：对于小偏差，用 Legendre 近似
    # 严格实现应使用 scipy.special 的 roots_jacobi，但为减少依赖，
    # 这里对标准 Legendre 进行轻微修正
    if abs(alpha) < 1e-10 and abs(beta) < 1e-10:
        return nodes, weights
    
    # 简单修正：移动节点并调整权重（一阶近似）
    shift = (beta - alpha) / (2.0 * n + alpha + beta + 2.0)
    nodes = nodes + shift * (1.0 - nodes ** 2)
    nodes = np.clip(nodes, -1.0, 1.0)
    weights = weights * (1.0 - nodes) ** alpha * (1.0 + nodes) ** beta
    weights = weights / np.sum(weights) * 2.0  # 归一化到总权重=2
    return nodes, weights


def map_to_physical_interval(
    nodes: np.ndarray,
    weights: np.ndarray,
    a: float,
    b: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 [-1,1] 上的节点与权重映射到 [a,b].
    
    x_phys = (a+b)/2 + (b-a)/2 · x_std
    w_phys = (b-a)/2 · w_std
    """
    if b <= a:
        raise ValueError("b must be greater than a")
    scale = 0.5 * (b - a)
    shift = 0.5 * (a + b)
    phys_nodes = shift + scale * nodes
    phys_weights = scale * weights
    return phys_nodes, phys_weights


def five_dimensional_gauss_quadrature(
    f: Callable[[np.ndarray], np.ndarray],
    n_per_dim: int = 8,
    mass_number: int = 235,
) -> float:
    """
    五维集体坐标空间的高斯积分.
    
    积分变量: (β₂, β₃, β₄, β₅, Δ)
    使用张量积 Legendre-Gauss 规则.
    
    参数:
        f: 被积函数，接受 (N,5) 数组返回 (N,) 值
        n_per_dim: 每维节点数
        mass_number: 质量数（用于确定积分范围）
    返回:
        积分近似值
    """
    from collective_coordinates import collective_coordinate_bounds
    
    bounds = collective_coordinate_bounds(mass_number)
    keys = ['beta2', 'beta3', 'beta4', 'beta5', 'delta']
    
    # 每维节点
    nodes_list = []
    weights_list = []
    
    for key in keys:
        a, b = bounds[key]
        if key == 'delta':
            # 配对能隙使用 Jacobi 权（近似）
            nodes, weights = jacobi_gauss_nodes_weights(n_per_dim, alpha=0.0, beta=0.0)
        else:
            nodes, weights = legendre_gauss_nodes_weights(n_per_dim)
        phys_nodes, phys_weights = map_to_physical_interval(nodes, weights, a, b)
        nodes_list.append(phys_nodes)
        weights_list.append(phys_weights)
    
    # 张量积构造
    total = 0.0
    n_total = n_per_dim ** 5
    
    # 为避免内存爆炸，使用逐维累加而非构造完整网格
    for i2 in range(n_per_dim):
        b2 = nodes_list[0][i2]
        w2 = weights_list[0][i2]
        for i3 in range(n_per_dim):
            b3 = nodes_list[1][i3]
            w3 = weights_list[1][i3]
            for i4 in range(n_per_dim):
                b4 = nodes_list[2][i4]
                w4 = weights_list[2][i4]
                for i5 in range(n_per_dim):
                    b5 = nodes_list[3][i5]
                    w5 = weights_list[3][i5]
                    for idelta in range(n_per_dim):
                        delta_val = nodes_list[4][idelta]
                        wdelta = weights_list[4][idelta]
                        
                        q = np.array([[b2, b3, b4, b5, delta_val]])
                        val = f(q)[0]
                        total += val * w2 * w3 * w4 * w5 * wdelta
    
    return float(total)


def partition_function_integral(
    mass_number: int,
    charge_number: int,
    excitation_energy: float,
    n_per_dim: int = 6,
) -> float:
    """
    计算裂变配分函数 Z = ∫ exp(-V/T) d⁵q.
    
    这是统计模型中计算裂变宽度的核心量:
    Γ_f = (T / 2π) · (Z_saddle / Z_compound)
    """
    from potential_energy_surface import potential_energy
    from diffusion_coefficient import nuclear_temperature
    
    T = nuclear_temperature(excitation_energy, mass_number)
    if T < 0.1:
        T = 0.1
    
    def integrand(q_array):
        vals = np.zeros(len(q_array))
        for i in range(len(q_array)):
            V = potential_energy(q_array[i], mass_number, charge_number)
            vals[i] = np.exp(-V / T)
        return vals
    
    Z = five_dimensional_gauss_quadrature(integrand, n_per_dim, mass_number)
    return float(Z)
