"""
核裂变集体坐标空间定义与几何操作
=====================================
融合原始项目:
  - 1336_triangulation_display: 三角剖分拓扑结构思想
  - 192_closest_point_brute: 最近邻搜索算法

科学背景:
---------
重核裂变过程中，核形状由集体坐标 {q_λ} 描述。本模块定义五维集体坐标空间:
  q = (β₂, β₃, β₄, β₅, Δ) 
其中:
  β₂: 四极形变参数 (elongation),  控制核的总长度
  β₃: 八极形变参数 (mass asymmetry), 控制碎片质量不对称度
  β₄: 十六极形变参数 (necking), 控制颈部收缩
  β₅: 三十二极形变参数, 高阶形变修正
  Δ:  配对能隙, BCS理论中的超导能隙

核表面在球坐标系下展开为:
  R(θ,φ) = R₀ [ 1 + Σ_{λ=2}^{5} β_λ Y_{λ0}(θ,φ) ]
其中 Y_{λ0} 为球谐函数，R₀ = r₀ A^(1/3) 为等效球半径。

碎片质量数 A_L, A_H 与 β₃ 的关系:
  A_L = (A/2)(1 - c₃ β₃),  A_H = (A/2)(1 + c₃ β₃)
其中 c₃ = √(7/4π) ≈ 0.746，由八极矩与质量位移的几何关系导出。
"""

import numpy as np
from typing import Tuple, List

# 物理常数
R0_FACTOR = 1.2  # fm, 核半径参数 r₀
C3_ASYMMETRY = np.sqrt(7.0 / (4.0 * np.pi))  # 几何因子


def nuclear_radius(mass_number: int) -> float:
    """
    计算核等效球半径 R₀ = r₀ A^(1/3).
    
    参数:
        mass_number: 质量数 A
    返回:
        半径 (fm)
    """
    if mass_number <= 0:
        raise ValueError("mass_number must be positive")
    return R0_FACTOR * float(mass_number) ** (1.0 / 3.0)


def spherical_harmonic_y20(theta: np.ndarray) -> np.ndarray:
    """
    球谐函数 Y_{20}(θ,φ=0) = √(5/16π) (3cos²θ - 1).
    """
    return np.sqrt(5.0 / (16.0 * np.pi)) * (3.0 * np.cos(theta) ** 2 - 1.0)


def spherical_harmonic_y30(theta: np.ndarray) -> np.ndarray:
    """
    球谐函数 Y_{30}(θ,φ=0) = √(7/16π) (5cos³θ - 3cosθ).
    """
    c = np.cos(theta)
    return np.sqrt(7.0 / (16.0 * np.pi)) * (5.0 * c ** 3 - 3.0 * c)


def spherical_harmonic_y40(theta: np.ndarray) -> np.ndarray:
    """
    球谐函数 Y_{40}(θ,φ=0) = √(9/256π) (35cos⁴θ - 30cos²θ + 3).
    """
    c = np.cos(theta)
    return np.sqrt(9.0 / (256.0 * np.pi)) * (35.0 * c ** 4 - 30.0 * c ** 2 + 3.0)


def spherical_harmonic_y50(theta: np.ndarray) -> np.ndarray:
    """
    球谐函数 Y_{50}(θ,φ=0) = √(11/256π) (63cos⁵θ - 70cos³θ + 15cosθ).
    """
    c = np.cos(theta)
    return np.sqrt(11.0 / (256.0 * np.pi)) * (63.0 * c ** 5 - 70.0 * c ** 3 + 15.0 * c)


def nuclear_surface_profile(theta: np.ndarray, beta: np.ndarray, mass_number: int) -> np.ndarray:
    """
    计算核表面轮廓 R(θ).
    
    R(θ) = R₀ [ 1 + β₂ Y₂₀(θ) + β₃ Y₃₀(θ) + β₄ Y₄₀(θ) + β₅ Y₅₀(θ) ]
    
    参数:
        theta: 极角数组 (rad)
        beta: 形变参数 [β₂, β₃, β₄, β₅]
        mass_number: 质量数 A
    返回:
        R(θ) 数组 (fm)
    """
    if len(beta) < 4:
        raise ValueError("beta must have at least 4 elements")
    R0 = nuclear_radius(mass_number)
    # 数值鲁棒性：限制最大形变，防止 R < 0
    y2 = spherical_harmonic_y20(theta)
    y3 = spherical_harmonic_y30(theta)
    y4 = spherical_harmonic_y40(theta)
    y5 = spherical_harmonic_y50(theta)
    shape_factor = 1.0 + beta[0] * y2 + beta[1] * y3 + beta[2] * y4 + beta[3] * y5
    # 硬截断保证正性
    shape_factor = np.maximum(shape_factor, 0.1)
    return R0 * shape_factor


def mass_asymmetry_to_fragment_mass(beta3: float, mass_number: int) -> Tuple[float, float]:
    """
    将八极形变 β₃ 转换为轻、重碎片质量数.
    
    由八极矩定义 Q₃ = (3/(4π)) Z e R₀³ β₃，以及质量位移与 Q₃ 的线性关系:
    δA = A_L - A_H = -2 c₃ A β₃ / √(1 + (c₃ β₃)²)
    
    近似展开（小形变）:
    A_L ≈ (A/2)(1 - c₃ β₃)
    A_H ≈ (A/2)(1 + c₃ β₃)
    
    参数:
        beta3: 八极形变参数
        mass_number: 母核质量数 A
    返回:
        (A_light, A_heavy)
    """
    if mass_number <= 0:
        raise ValueError("mass_number must be positive")
    # TODO(Hole_1): 实现八极形变 β₃ 到碎片质量数 A_L, A_H 的转换。
    # 科学背景：由八极矩定义 Q₃ = (3/(4π)) Z e R₀³ β₃，质量位移与 Q₃ 线性相关。
    # 精确关系：A_L = (A/2)(1 - x/√(1+x²)), A_H = (A/2)(1 + x/√(1+x²)), x = c₃ β₃
    # 小形变近似：A_L ≈ (A/2)(1 - c₃ β₃), A_H ≈ (A/2)(1 + c₃ β₃)
    # 需考虑大形变下的边界保护（clip 到 [1, A-1]）。
    # 变量 C3_ASYMMETRY = √(7/4π) 已在模块顶部定义。
    raise NotImplementedError("Hole_1: mass_asymmetry_to_fragment_mass 待修复")
    return 0.0, 0.0  # 占位


def fragment_mass_to_asymmetry(A_light: float, mass_number: int) -> float:
    """
    由碎片质量反推八极形变 β₃.
    
    由 A_L = (A/2)(1 - x/√(1+x²)), x = c₃ β₃ 反解:
    x = (A - 2A_L) / (2 √(A_L (A - A_L)))
    """
    if A_light <= 0 or A_light >= mass_number:
        raise ValueError("A_light out of valid range")
    numerator = mass_number - 2.0 * A_light
    denominator = 2.0 * np.sqrt(A_light * (mass_number - A_light))
    if abs(denominator) < 1e-14:
        return 0.0
    x = numerator / denominator
    return x / C3_ASYMMETRY


def closest_point_brute(points: np.ndarray, target: np.ndarray) -> Tuple[int, float]:
    """
    暴力搜索 target 在点集 points 中的最近邻.
    
    源自 closest_point_brute.m 核心算法，用于在集体坐标空间中
    寻找与当前裂变构型最接近的参考构型。
    
    参数:
        points: (N, D) 点集
        target: (D,) 目标点
    返回:
        (最近邻索引, 欧氏距离)
    """
    if points.ndim != 2 or target.ndim != 1:
        raise ValueError("points must be 2D array and target must be 1D array")
    if points.shape[1] != target.shape[0]:
        raise ValueError("dimension mismatch between points and target")
    if len(points) == 0:
        raise ValueError("points array is empty")

    min_dist_sq = np.inf
    nearest_idx = -1
    for i in range(len(points)):
        dist_sq = np.sum((points[i] - target) ** 2)
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            nearest_idx = i
    return nearest_idx, np.sqrt(min_dist_sq)


def collective_coordinate_bounds(mass_number: int) -> dict:
    """
    返回集体坐标的物理合理范围.
    
    基于液滴模型与实验数据，不同质量数核的形变范围不同。
    """
    #  heavier nuclei allow larger deformation before scission
    scale = min(1.0, mass_number / 300.0)
    bounds = {
        'beta2': (-0.5 * scale, 2.5 * scale),   # 四极形变
        'beta3': (-1.2 * scale, 1.2 * scale),   # 八极形变 (mass asymmetry)
        'beta4': (-0.8 * scale, 0.8 * scale),   # 十六极形变
        'beta5': (-0.4 * scale, 0.4 * scale),   # 三十二极形变
        'delta': (0.0, 3.0),                     # 配对能隙 (MeV)
    }
    return bounds


def clip_to_physical_domain(q: np.ndarray, bounds: dict) -> np.ndarray:
    """
    将集体坐标裁剪到物理允许域.
    """
    keys = ['beta2', 'beta3', 'beta4', 'beta5', 'delta']
    q_clipped = q.copy()
    for i, key in enumerate(keys):
        if i < len(q):
            lo, hi = bounds[key]
            q_clipped[i] = np.clip(q_clipped[i], lo, hi)
    return q_clipped


def triangulate_configuration_space_1d(n_nodes: int, q_min: float, q_max: float) -> np.ndarray:
    """
    一维集体坐标空间的网格剖分 (基于 triangulation_display 的拓扑思想).
    
    生成均匀节点:
    q_i = q_min + i * (q_max - q_min) / n_nodes, i=0,...,n_nodes
    
    返回节点坐标数组.
    """
    if n_nodes < 2:
        raise ValueError("n_nodes must be at least 2")
    return np.linspace(q_min, q_max, n_nodes + 1)
