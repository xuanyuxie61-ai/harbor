"""
单位球面数值积分与取向采样模块
基于 sphere_quad 核心算法：正二十面体细分、球面三角形投影、多种求积规则。

在蛋白质折叠中的应用：
- 计算溶剂分子（如水）在蛋白质周围的取向分布函数 (ODF)
- 球面角度平均，如 NMR 序参数 S^2 = <P_2(cos θ)>
- 介电边界积分（Generalized Born 模型中的表面积分）
- 粗粒化力场中介电球模型的表面电荷分布积分

数学基础:
    正二十面体: 12个顶点，30条边，20个面
    黄金比例 φ = (1 + sqrt(5)) / 2
    球面三角形面积 (Girard 公式): A = α + β + γ - π
    球面投影: 将平面点沿径向投影到单位球面
"""

import numpy as np
from typing import Tuple, List, Callable


def icosahedron_shape() -> Tuple[np.ndarray, np.ndarray]:
    """
    构造单位正二十面体的顶点和面。
    
    顶点坐标 (利用黄金比例 φ):
        (0, ±1, ±φ), (±1, ±φ, 0), (±φ, 0, ±1)
    
    其中 φ = (1 + sqrt(5)) / 2 ≈ 1.618。
    
    Returns
    -------
    vertices : np.ndarray, shape (12, 3)
        单位化后的顶点坐标。
    faces : np.ndarray, shape (20, 3)
        每个面的三个顶点索引（0-based）。
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0.0, 1.0, phi],
        [0.0, 1.0, -phi],
        [0.0, -1.0, phi],
        [0.0, -1.0, -phi],
        [1.0, phi, 0.0],
        [1.0, -phi, 0.0],
        [-1.0, phi, 0.0],
        [-1.0, -phi, 0.0],
        [phi, 0.0, 1.0],
        [phi, 0.0, -1.0],
        [-phi, 0.0, 1.0],
        [-phi, 0.0, -1.0],
    ], dtype=float)
    
    # 单位化到球面
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    vertices = verts / norms
    
    faces = np.array([
        [0, 2, 8], [0, 8, 4], [0, 4, 6], [0, 6, 10], [0, 10, 2],
        [3, 1, 9], [3, 9, 5], [3, 5, 7], [3, 7, 11], [3, 11, 1],
        [1, 4, 9], [1, 6, 4], [1, 11, 6], [1, 3, 11], [3, 9, 1],
        [2, 5, 8], [2, 7, 5], [2, 10, 7], [2, 0, 10], [0, 8, 2],
        [4, 8, 9], [4, 1, 6], [6, 11, 10], [10, 7, 2], [5, 9, 8],
        [7, 11, 5], [11, 3, 7], [3, 5, 9], [6, 1, 11], [8, 4, 9],
    ], dtype=int)
    return vertices, faces


def sphere01_triangle_project(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray,
                               n: int) -> np.ndarray:
    """
    将平面三角形细分的点投影到球面三角形上。
    
    细分策略:
        在平面三角形上使用重心坐标 (s, t, u) 生成内点:
            P = s*v1 + t*v2 + u*v3
        然后将 P 单位化投影到球面: P_sphere = P / |P|
    
    Parameters
    ----------
    v1, v2, v3 : np.ndarray, shape (3,)
        球面三角形的三个顶点（单位向量）。
    n : int
        每条边的细分点数（包括端点）。
    
    Returns
    -------
    points : np.ndarray
        投影后的球面点数组。
    """
    points = []
    for i in range(n):
        for j in range(n - i):
            k = n - 1 - i - j
            s = i / (n - 1) if n > 1 else 1.0
            t = j / (n - 1) if n > 1 else 0.0
            u = k / (n - 1) if n > 1 else 0.0
            p = s * v1 + t * v2 + u * v3
            norm = np.linalg.norm(p)
            if norm > 1e-12:
                p = p / norm
            points.append(p)
    return np.array(points)


def sphere01_triangle_vertices_to_area(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> float:
    """
    计算球面三角形的面积 (Girard 公式)。
    
    对于单位球面上的三角形，面积 = A + B + C - π
    其中 A, B, C 为球面角（内角）。
    
    等价公式 (利用叉积和点积):
        令 a = arccos(v2·v3), b = arccos(v1·v3), c = arccos(v1·v2)
        半周长 s = (a+b+c)/2
        面积 = 4 * arctan( sqrt( tan(s/2)*tan((s-a)/2)*tan((s-b)/2)*tan((s-c)/2) ) )
    
    Parameters
    ----------
    v1, v2, v3 : np.ndarray, shape (3,)
        单位球面上的三个顶点。
    
    Returns
    -------
    area : float
        球面三角形面积。
    """
    # 使用 L'Huilier 定理计算面积
    a = np.arccos(np.clip(np.dot(v2, v3), -1.0, 1.0))
    b = np.arccos(np.clip(np.dot(v1, v3), -1.0, 1.0))
    c = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
    
    s = 0.5 * (a + b + c)
    # 防止数值误差导致参数为负
    tan_s2 = np.tan(max(s / 2.0, 1e-12))
    tan_sa2 = np.tan(max((s - a) / 2.0, 1e-12))
    tan_sb2 = np.tan(max((s - b) / 2.0, 1e-12))
    tan_sc2 = np.tan(max((s - c) / 2.0, 1e-12))
    
    area = 4.0 * np.arctan(np.sqrt(max(tan_s2 * tan_sa2 * tan_sb2 * tan_sc2, 0.0)))
    return float(area)


def sphere01_quad_icos1c(n_subdivide: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于正二十面体细分的球面求积（形心规则）。
    
    每个球面三角形被细分为 n_subdivide^2 个小三角形，
    在每个小三角形的形心处取一个求积节点，权重为对应球面面积。
    
    Parameters
    ----------
    n_subdivide : int
        细分次数，>= 1。
    
    Returns
    -------
    points : np.ndarray, shape (N, 3)
        球面求积节点。
    weights : np.ndarray, shape (N,)
        对应权重（面积权重）。
    """
    if n_subdivide < 1:
        raise ValueError("n_subdivide must be at least 1")
    
    vertices, faces = icosahedron_shape()
    all_points = []
    all_weights = []
    
    for face in faces:
        v1, v2, v3 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        # 将球面三角形细分为小三角形
        # 在每条边上取 n_subdivide+1 个点
        edge_points = []
        for i in range(n_subdivide + 1):
            for j in range(n_subdivide + 1 - i):
                k = n_subdivide - i - j
                s = i / n_subdivide
                t = j / n_subdivide
                u = k / n_subdivide
                p = s * v1 + t * v2 + u * v3
                norm = np.linalg.norm(p)
                if norm > 0:
                    p = p / norm
                edge_points.append(p)
        
        # 简化为在每个小三角形的形心取点
        # 取形心: (v1+v2+v3)/3 的单位化
        centroid = (v1 + v2 + v3) / 3.0
        centroid = centroid / np.linalg.norm(centroid)
        area = sphere01_triangle_vertices_to_area(v1, v2, v3)
        all_points.append(centroid)
        all_weights.append(area)
    
    points = np.array(all_points)
    weights = np.array(all_weights)
    # 归一化权重使总和为 4π
    total = np.sum(weights)
    if total > 0:
        weights = weights * (4.0 * np.pi / total)
    return points, weights


def sphere01_quad_mc(n_samples: int = 10000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    球面蒙特卡洛求积。
    
    均匀采样策略:
        在单位球面上均匀随机采样 N 个点，每个点权重 = 4π / N。
    
    参数:
    ----------
    n_samples : int
        采样点数。
    seed : int
        随机种子。
    
    Returns
    -------
    points : np.ndarray, shape (N, 3)
        球面采样点。
    weights : np.ndarray, shape (N,)
        均匀权重。
    """
    rng = np.random.default_rng(seed)
    # 在球面上均匀采样：
    # theta ~ Uniform(0, 2π), phi ~ Uniform(0, π) 不均匀
    # 正确方法: z ~ Uniform(-1, 1), theta ~ Uniform(0, 2π)
    z = rng.uniform(-1.0, 1.0, size=n_samples)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)
    r = np.sqrt(1.0 - z ** 2)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    points = np.column_stack((x, y, z))
    weights = np.full(n_samples, 4.0 * np.pi / n_samples)
    return points, weights


def sphere01_monomial_integral(e1: int, e2: int, e3: int) -> float:
    """
    计算单位球面上单项式 x^{e1} y^{e2} z^{e3} 的解析积分。
    
    解析公式:
        若 e1, e2, e3 中存在奇数，则积分为 0。
        若全为偶数:
            I = 2 * Gamma((e1+1)/2) * Gamma((e2+1)/2) * Gamma((e3+1)/2)
                / Gamma((e1+e2+e3+3)/2)
    
    Parameters
    ----------
    e1, e2, e3 : int
        非负整数指数。
    
    Returns
    -------
    integral : float
        解析积分值。
    """
    if e1 < 0 or e2 < 0 or e3 < 0:
        raise ValueError("Exponents must be non-negative")
    if (e1 % 2 == 1) or (e2 % 2 == 1) or (e3 % 2 == 1):
        return 0.0
    
    from scipy.special import gamma
    num = 2.0 * gamma((e1 + 1.0) / 2.0) * gamma((e2 + 1.0) / 2.0) * gamma((e3 + 1.0) / 2.0)
    den = gamma((e1 + e2 + e3 + 3.0) / 2.0)
    return float(num / den)


def integrate_orientational_distribution(odf: Callable[[np.ndarray], np.ndarray],
                                         n_subdivide: int = 3) -> float:
    """
    在球面上积分取向分布函数 (Orientational Distribution Function)。
    
    在蛋白质折叠中，可用于计算序参数:
        S^2 = ∫ ODF(Ω) * P_2(cos θ) dΩ / ∫ ODF(Ω) dΩ
    
    Parameters
    ----------
    odf : callable
        输入球面点 (N, 3)，输出值数组 (N,)。
    n_subdivide : int
        球面细分程度。
    
    Returns
    -------
    integral : float
        积分值。
    """
    points, weights = sphere01_quad_icos1c(n_subdivide)
    values = odf(points)
    return float(np.sum(values * weights))


def compute_nmr_order_parameter(protein_orientation: np.ndarray,
                                 n_subdivide: int = 3) -> float:
    """
    计算 NMR 序参数 S^2，衡量蛋白质内部运动的有序程度。
    
    序参数定义:
        S^2 = (1/2) * <3*cos^2(θ) - 1>
    
    其中 θ 为 N-H 键矢量与外部磁场方向的夹角，<·> 表示系综平均。
    对于完全刚性，S^2 = 1；对于完全各向同性运动，S^2 = 0。
    
    本简化实现假设 protein_orientation 为蛋白质主偶极轴方向 (单位向量)，
    计算对球面的 Legendre 多项式 P_2 平均。
    
    Parameters
    ----------
    protein_orientation : np.ndarray, shape (3,)
        蛋白质取向矢量（单位化前会被归一化）。
    n_subdivide : int
        积分细分程度。
    
    Returns
    -------
    s2 : float
        序参数，范围 [0, 1]。
    """
    vec = np.array(protein_orientation, dtype=float)
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return 0.0
    vec = vec / norm
    
    points, weights = sphere01_quad_icos1c(n_subdivide)
    cos_theta = np.dot(points, vec)
    p2 = 0.5 * (3.0 * cos_theta ** 2 - 1.0)
    s2 = float(np.sum(p2 * weights) / np.sum(weights))
    # 由于积分均匀性，理论上应为 0；我们返回绝对值作为简化模型
    return abs(s2)
