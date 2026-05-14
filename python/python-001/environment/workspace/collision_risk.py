"""
collision_risk.py

基于 ball_distance (单位球内随机点距离统计) 与
neighbor_risk (邻接矩阵风险评估) 核心算法，
实现小行星表面附近航天器的碰撞风险评估与区域连通性分析。

科学背景：
近小行星任务（如 OSIRIS-REx, Hayabusa2）面临的核心风险之一是
航天器与不规则表面的碰撞。

风险评估模型：
1. 蒙特卡洛距离统计：
    在单位球内随机采样两点，其距离分布的期望为:
        E[D] = 36/35 ≈ 1.02857 (对于单位球)
    该统计用于评估小行星内部密度分布的各向异性。

2. 表面区域邻接图：
    将多面体表面三角面片作为节点，共享边的面片之间有边连接。
    通过邻接矩阵分析表面可达性与着陆区连通性。

3. 碰撞概率模型：
    P_collision ≈ Σ_{faces} A_face / (4πr²) * I( |r − r_face| < h_safe )
    其中 h_safe 为安全距离阈值，I 为示性函数。
"""

import numpy as np
from typing import Tuple, Optional


class CollisionRiskError(Exception):
    pass


def ball_unit_sample(n: int = 1, seed: Optional[int] = None) -> np.ndarray:
    """
    在单位球内均匀随机采样 n 个点。
    算法： rejection sampling 或球坐标变换。
    使用体积元 dV = r² sinθ dr dθ dφ 的变换：
        r = u1^{1/3},  θ = arccos(2u2 − 1),  φ = 2π u3
    """
    if seed is not None:
        np.random.seed(seed)
    points = np.zeros((n, 3))
    for i in range(n):
        u = np.random.rand(3)
        r = u[0] ** (1.0 / 3.0)
        theta = np.arccos(2.0 * u[1] - 1.0)
        phi = 2.0 * np.pi * u[2]
        points[i, 0] = r * np.sin(theta) * np.cos(phi)
        points[i, 1] = r * np.sin(theta) * np.sin(phi)
        points[i, 2] = r * np.cos(theta)
    return points


def ball_distance_stats(n_samples: int = 10000, seed: Optional[int] = None) -> Tuple[float, float]:
    """
    基于 ball_distance_stats.m 的蒙特卡洛统计。
    在单位球内随机取两点，计算距离 D = |P − Q| 的均值与方差。

    理论值（单位球）：
        E[D] = 36/35 ≈ 1.028571
        Var[D] 可由数值估计
    """
    if seed is not None:
        np.random.seed(seed)
    distances = np.zeros(n_samples)
    for i in range(n_samples):
        p = ball_unit_sample(1)
        q = ball_unit_sample(1)
        distances[i] = np.linalg.norm(p - q)

    mu = float(np.mean(distances))
    if n_samples > 1:
        var = float(np.sum((distances - mu) ** 2) / (n_samples - 1))
    else:
        var = 0.0
    return mu, var


def build_surface_adjacency_matrix(faces: np.ndarray, n_vertices: int) -> np.ndarray:
    """
    基于 neighbor_risk 的邻接矩阵思想，构建表面面片邻接矩阵。
    两个面片相邻当且仅当它们共享一条边。

    参数:
        faces: (n_faces, 3) 三角面片顶点索引
        n_vertices: 顶点总数

    返回:
        adj: (n_faces, n_faces) 邻接矩阵（对称，0/1）
    """
    n_faces = faces.shape[0]
    adj = np.zeros((n_faces, n_faces), dtype=int)

    # 构建边到面片的映射
    edge_to_faces = {}
    for fi in range(n_faces):
        for e in range(3):
            v1 = faces[fi, e]
            v2 = faces[fi, (e + 1) % 3]
            edge = tuple(sorted((int(v1), int(v2))))
            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append(fi)

    # 共享边的面片互为邻居
    for edge, face_list in edge_to_faces.items():
        if len(face_list) >= 2:
            for i in range(len(face_list)):
                for j in range(i + 1, len(face_list)):
                    f1 = face_list[i]
                    f2 = face_list[j]
                    adj[f1, f2] = 1
                    adj[f2, f1] = 1

    return adj


def compute_face_centroids(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    计算每个三角面片的质心。
    """
    n_faces = faces.shape[0]
    centroids = np.zeros((n_faces, 3))
    for i in range(n_faces):
        centroids[i] = np.mean(vertices[faces[i]], axis=0)
    return centroids


def compute_face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    计算每个三角面片的面积。
        A = 0.5 | (v2 − v1) × (v3 − v1) |
    """
    n_faces = faces.shape[0]
    areas = np.zeros(n_faces)
    for i in range(n_faces):
        v1 = vertices[faces[i, 0]]
        v2 = vertices[faces[i, 1]]
        v3 = vertices[faces[i, 2]]
        areas[i] = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))
    return areas


def collision_probability_surface(
    pos: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    safe_distance: float = 0.5,
    position_uncertainty: float = 0.1
) -> float:
    """
    计算航天器位置 pos 处与小行星表面的碰撞概率。

    模型：
        位置不确定性服从三维高斯分布 N(pos, σ²I)。
        对每个面片，计算高斯分布下该面片附近（距离 < safe_distance）的概率质量。
        使用面片面积加权。

    简化公式（点-面距离高斯近似）：
        P ≈ Σ_i (A_i / A_total) * Φ( (safe_distance − d_i) / σ )
    其中 Φ 为标准正态 CDF，d_i 为 pos 到面片 i 质心的距离。
    """
    centroids = compute_face_centroids(vertices, faces)
    areas = compute_face_areas(vertices, faces)
    total_area = np.sum(areas)
    if total_area < 1e-14:
        return 0.0

    p_total = 0.0
    for i in range(faces.shape[0]):
        d = np.linalg.norm(pos - centroids[i])
        # 标准正态累积分布的近似（误差函数）
        z = (safe_distance - d) / max(position_uncertainty, 1e-12)
        p_face = 0.5 * (1.0 + np.tanh(z / np.sqrt(2.0) * 0.8))  # 近似 Φ(z)
        p_total += (areas[i] / total_area) * p_face

    return min(p_total, 1.0)


def find_safe_hover_regions(
    vertices: np.ndarray,
    faces: np.ndarray,
    min_altitude: float = 1.0,
    n_samples: int = 500,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    在多面体表面上方搜索安全悬停区域。

    方法：
    1. 在包围球表面均匀采样候选悬停点
    2. 计算每个候选点的碰撞概率
    3. 返回碰撞概率低于阈值的区域

    返回:
        safe_points: (m, 3) 安全悬停点
        safe_probs: (m,) 对应的碰撞概率
    """
    if seed is not None:
        np.random.seed(seed)

    centroids = compute_face_centroids(vertices, faces)
    normals = np.zeros_like(centroids)
    for i in range(faces.shape[0]):
        v1 = vertices[faces[i, 0]]
        v2 = vertices[faces[i, 1]]
        v3 = vertices[faces[i, 2]]
        n_vec = np.cross(v2 - v1, v3 - v1)
        norm = np.linalg.norm(n_vec)
        if norm > 1e-14:
            normals[i] = n_vec / norm

    # 在每个面片法向方向上偏移 min_altitude 采样候选点
    candidates = centroids + min_altitude * normals

    safe_points = []
    safe_probs = []
    for i in range(min(n_samples, candidates.shape[0])):
        idx = i
        p_coll = collision_probability_surface(
            candidates[idx], vertices, faces,
            safe_distance=min_altitude * 0.5,
            position_uncertainty=min_altitude * 0.1
        )
        if p_coll < 0.1:
            safe_points.append(candidates[idx])
            safe_probs.append(p_coll)

    if len(safe_points) == 0:
        return np.zeros((0, 3)), np.zeros(0)
    return np.array(safe_points), np.array(safe_probs)


def region_connectivity_analysis(adj: np.ndarray) -> dict:
    """
    基于邻接矩阵分析表面区域的连通性。

    返回字典包含：
    - n_components: 连通分量数
    - component_sizes: 各分量大小
    - diameter_est: 图直径估计（最长最短路径）
    """
    n = adj.shape[0]
    visited = np.zeros(n, dtype=bool)
    component_sizes = []

    def bfs(start: int) -> Tuple[int, np.ndarray]:
        dist = -np.ones(n, dtype=int)
        dist[start] = 0
        queue = [start]
        while queue:
            u = queue.pop(0)
            for v in range(n):
                if adj[u, v] and dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        return int(np.max(dist[dist >= 0])), dist

    max_diameter = 0
    for i in range(n):
        if not visited[i]:
            _, dist = bfs(i)
            component = np.where(dist >= 0)[0]
            component_sizes.append(len(component))
            visited[component] = True
            # 在该分量内估计直径
            far_node = int(np.argmax(dist))
            d_far, _ = bfs(far_node)
            max_diameter = max(max_diameter, d_far)

    return {
        "n_components": len(component_sizes),
        "component_sizes": component_sizes,
        "diameter_est": max_diameter,
        "total_nodes": n
    }
