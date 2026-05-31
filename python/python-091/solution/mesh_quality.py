"""
声学有限元网格质量评估模块

基于种子项目 958_quality 的核心算法，为超声层析成像提供网格质量诊断。
融合多种几何与拓扑质量度量，确保声学Helmholtz方程有限元离散的几何可靠性。

核心公式:
- Q度量: Q = 4√3 · A / (a²+b²+c²)，A为三角形面积，a,b,c为边长
- Alpha度量: α = min(θ₁,θ₂,θ₃) / (π/3)，最小角标准化
- Gamma度量: γ = min(dᵢⱼ) / max(dᵢⱼ)，最近邻距离均匀性
- D度量: 基于Voronoi区域的偏张量行列式
"""

import numpy as np
from typing import List, Tuple, Dict


def triangle_area(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """计算三角形的有向面积（叉积法）。
    
    公式: A = 0.5 * | (x₂-x₁)(y₃-y₁) - (x₃-x₁)(y₂-y₁) |
    """
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))


def q_measure(triangles: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Q质量度量：内切圆半径与外接圆半径之比。
    
    对于三角形单元，Q = 4√3 · A / (a² + b² + c²)
    其中 A 为面积，a,b,c 为三条边长。
    Q ∈ [0, 1]，Q=1 时为等边三角形（最优）。
    
    参数:
        triangles: (N, 3) 三角形节点索引数组
        nodes: (M, 2) 节点坐标数组
    
    返回:
        q_values: (N,) 每个三角形的Q度量值
    """
    n_tri = triangles.shape[0]
    q_values = np.zeros(n_tri)
    
    for i in range(n_tri):
        idx = triangles[i]
        p1, p2, p3 = nodes[idx[0]], nodes[idx[1]], nodes[idx[2]]
        
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p1 - p3)
        
        area = triangle_area(p1, p2, p3)
        
        if area < 1e-14:
            q_values[i] = 0.0
            continue
        
        denom = a**2 + b**2 + c**2
        if denom < 1e-14:
            q_values[i] = 0.0
            continue
        
        q_values[i] = 4.0 * np.sqrt(3.0) * area / denom
    
    return q_values


def alpha_measure(triangles: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Alpha度量：最小内角的标准化值。
    
    公式: α = min(θ₁, θ₂, θ₃) / (π/3)
    其中 θᵢ 为三角形三个内角，π/3 为等边三角形内角。
    α ∈ [0, 1]，α=1 时为等边三角形。
    
    使用余弦定理计算内角:
    cos(θ₁) = (b² + c² - a²) / (2bc)
    """
    n_tri = triangles.shape[0]
    alpha_vals = np.zeros(n_tri)
    
    for i in range(n_tri):
        idx = triangles[i]
        p1, p2, p3 = nodes[idx[0]], nodes[idx[1]], nodes[idx[2]]
        
        a = np.linalg.norm(p2 - p3)
        b = np.linalg.norm(p1 - p3)
        c = np.linalg.norm(p1 - p2)
        
        # 边界检查：防止退化三角形
        if a < 1e-14 or b < 1e-14 or c < 1e-14:
            alpha_vals[i] = 0.0
            continue
        
        # 余弦定理计算三个角
        cos1 = (b**2 + c**2 - a**2) / (2.0 * b * c)
        cos2 = (a**2 + c**2 - b**2) / (2.0 * a * c)
        cos3 = (a**2 + b**2 - c**2) / (2.0 * a * b)
        
        # 数值鲁棒性：clamp到[-1,1]
        cos1 = np.clip(cos1, -1.0, 1.0)
        cos2 = np.clip(cos2, -1.0, 1.0)
        cos3 = np.clip(cos3, -1.0, 1.0)
        
        theta1 = np.arccos(cos1)
        theta2 = np.arccos(cos2)
        theta3 = np.arccos(cos3)
        
        min_angle = min(theta1, theta2, theta3)
        alpha_vals[i] = min_angle / (np.pi / 3.0)
    
    return alpha_vals


def gamma_measure(nodes: np.ndarray) -> float:
    """Gamma度量：点集最近邻距离的均匀性。
    
    公式: γ = min(dᵢ) / max(dᵢ)
    其中 dᵢ 为每个节点到其最近邻的距离。
    γ ∈ [0, 1]，γ接近1表示点集分布均匀。
    
    参数:
        nodes: (M, 2) 节点坐标数组
    
    返回:
        gamma: Gamma度量值
    """
    n_nodes = nodes.shape[0]
    if n_nodes <= 1:
        return 1.0
    
    min_distances = np.zeros(n_nodes)
    
    for i in range(n_nodes):
        dists = np.linalg.norm(nodes - nodes[i], axis=1)
        dists[i] = np.inf  # 排除自身
        min_distances[i] = np.min(dists)
    
    d_min = np.min(min_distances)
    d_max = np.max(min_distances)
    
    if d_max < 1e-14:
        return 1.0
    
    return d_min / d_max


def bandwidth_mesh(triangles: np.ndarray) -> int:
    """计算有限元网格对应刚度矩阵的半带宽。
    
    对于线性三角形单元，刚度矩阵的非零结构由共享节点的单元决定。
    半带宽 B = max|i - j|，其中 i,j 为共享同一单元的节点编号。
    
    参数:
        triangles: (N, 3) 三角形节点索引数组
    
    返回:
        bandwidth: 矩阵半带宽
    """
    bandwidth = 0
    for tri in triangles:
        i, j, k = tri
        local_bw = max(abs(i - j), abs(j - k), abs(i - k))
        bandwidth = max(bandwidth, local_bw)
    return bandwidth


def mesh_quality_report(triangles: np.ndarray, nodes: np.ndarray) -> Dict[str, float]:
    """生成完整的网格质量报告。
    
    返回包含多项质量指标的字典:
    - q_min, q_mean: Q度量的最小值和均值
    - alpha_min, alpha_mean: Alpha度量的最小值和均值
    - gamma: 全局Gamma度量
    - bandwidth: 刚度矩阵半带宽
    """
    q_vals = q_measure(triangles, nodes)
    alpha_vals = alpha_measure(triangles, nodes)
    gamma = gamma_measure(nodes)
    bw = bandwidth_mesh(triangles)
    
    report = {
        'q_min': float(np.min(q_vals)),
        'q_mean': float(np.mean(q_vals)),
        'alpha_min': float(np.min(alpha_vals)),
        'alpha_mean': float(np.mean(alpha_vals)),
        'gamma': float(gamma),
        'bandwidth': int(bw),
        'num_triangles': triangles.shape[0],
        'num_nodes': nodes.shape[0]
    }
    return report


def reject_poor_triangles(triangles: np.ndarray, nodes: np.ndarray,
                          q_threshold: float = 0.1,
                          alpha_threshold: float = 0.1) -> np.ndarray:
    """剔除质量过差的三角形单元。
    
    参数:
        triangles: 输入三角形数组
        nodes: 节点坐标
        q_threshold: Q度量阈值，低于此值的三角形被剔除
        alpha_threshold: Alpha度量阈值，低于此值的三角形被剔除
    
    返回:
        保留下来的三角形索引掩码
    """
    q_vals = q_measure(triangles, nodes)
    alpha_vals = alpha_measure(triangles, nodes)
    
    mask = (q_vals >= q_threshold) & (alpha_vals >= alpha_threshold)
    return mask
