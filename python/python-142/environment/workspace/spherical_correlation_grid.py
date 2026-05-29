"""
spherical_correlation_grid.py
球面格点生成、球面 Voronoi 剖分与邻接关系计算
应用于信用风险中的地理-行业违约相关性空间建模

原项目映射: 1123_sphere_llt_grid, 1131_sphere_voronoi, 1397_voronoi_neighbors, 725_matlab_map
科学问题: 在全球化信用组合中，不同地理区域（国家/洲）的违约事件具有空间相关性。
将地球表面近似为单位球面 S^2，利用经纬度三角网格 (LLT) 生成离散区域中心，
再通过球面 Voronoi 剖分将全球划分为若干经济区域。
每个 Voronoi 单元代表一个信用风险区域，单元面积反映区域经济权重，
邻接关系用于建模区域间违约传染 (contagion) 的网络拓扑结构。

核心数学:
    - 球面坐标: x = sin(phi)*cos(theta), y = sin(phi)*sin(theta), z = cos(phi)
    - Girard 公式计算球面三角形面积: Area = (A+B+C-pi) * R^2
    - Voronoi 对偶: Delaunay 三角剖分的外接球心投影到球面
"""

import numpy as np
from typing import Tuple, List, Optional


def sphere_llt_grid_points(r: float, pc: np.ndarray, lat_num: int, long_num: int) -> np.ndarray:
    """
    在单位球面上生成 Latitude-Longitude-Triangle (LLT) 网格点

    Parameters:
        r: 球半径
        pc: 球心坐标 (3,)
        lat_num: 纬度圈数
        long_num: 经度分割数

    Returns:
        xyz: (n_points x 3) 坐标数组
    """
    pc = np.asarray(pc)
    n_points = 2 + lat_num * long_num
    xyz = np.zeros((n_points, 3), dtype=float)

    # 北极
    xyz[0, :] = pc + np.array([0.0, 0.0, r])
    idx = 1

    # 纬度圈
    for lat in range(1, lat_num + 1):
        phi = np.pi * lat / (lat_num + 1)
        for lon in range(long_num):
            theta = 2.0 * np.pi * lon / long_num
            xyz[idx, 0] = pc[0] + r * np.sin(phi) * np.cos(theta)
            xyz[idx, 1] = pc[1] + r * np.sin(phi) * np.sin(theta)
            xyz[idx, 2] = pc[2] + r * np.cos(phi)
            idx += 1

    # 南极
    xyz[idx, :] = pc + np.array([0.0, 0.0, -r])
    return xyz


def spherical_triangle_area(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, r: float = 1.0) -> float:
    """
    使用 Girard 公式计算球面三角形面积
    Area = (alpha + beta + gamma - pi) * r^2
    其中 alpha, beta, gamma 为球面三角形的内角

    球面余弦定理:
        cos(a) = (cos(alpha) - cos(b)*cos(c)) / (sin(b)*sin(c))
    但这里采用向量法计算角:
        在顶点 v1 处，边为 e12 和 e13 (切平面上的方向)
        cos(alpha) = <(v1 x v2), (v1 x v3)> / (|v1 x v2| * |v1 x v3|)
    """
    v1, v2, v3 = np.asarray(v1), np.asarray(v2), np.asarray(v3)
    # 归一化
    v1 = v1 / (np.linalg.norm(v1) + 1e-15)
    v2 = v2 / (np.linalg.norm(v2) + 1e-15)
    v3 = v3 / (np.linalg.norm(v3) + 1e-15)

    # 球面边长 (中心角)
    a = np.arccos(np.clip(np.dot(v2, v3), -1.0, 1.0))
    b = np.arccos(np.clip(np.dot(v1, v3), -1.0, 1.0))
    c = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

    # 使用 L'Huilier 定理计算球面过剩
    s = (a + b + c) / 2.0
    # 避免数值问题
    tan_E_4 = np.sqrt(
        np.maximum(0.0,
            np.tan(s / 2.0) *
            np.tan((s - a) / 2.0) *
            np.tan((s - b) / 2.0) *
            np.tan((s - c) / 2.0)
        )
    )
    E = 4.0 * np.arctan(tan_E_4)
    return E * r * r


def spherical_voronoi_areas(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    通过球面 Delaunay 三角剖分计算每个点的 Voronoi 单元面积

    算法:
        1. 使用 convhulln (凸包) 计算球面上的 Delaunay 三角剖分
        2. 每个三角形的面积按 Girard 公式计算
        3. 每个点 i 的 Voronoi 面积 = 1/3 * sum(相邻三角形的面积)
           (因为对偶关系: 每个 Delaunay 三角形贡献给其三个顶点)

    Parameters:
        xyz: 球面上的点 (n x 3)

    Returns:
        areas: 每个点的 Voronoi 面积 (n,)
        faces: Delaunay 三角形面 (m x 3)
    """
    try:
        from scipy.spatial import ConvexHull
    except ImportError:
        # 若无 scipy，使用近似方法
        return _approximate_voronoi_areas(xyz)

    n = xyz.shape[0]
    hull = ConvexHull(xyz)
    faces = hull.simplices

    areas = np.zeros(n, dtype=float)
    for face in faces:
        i, j, k = face
        area = spherical_triangle_area(xyz[i], xyz[j], xyz[k])
        areas[i] += area / 3.0
        areas[j] += area / 3.0
        areas[k] += area / 3.0

    return areas, faces


def _approximate_voronoi_areas(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    无 scipy 时的近似 Voronoi 面积计算
    通过将每个点的邻近区域投影到切平面并计算平面多边形面积
    """
    n = xyz.shape[0]
    areas = np.zeros(n, dtype=float)
    # 使用 k-近邻近似
    for i in range(n):
        # 找到角度最近的邻居 (简化: 用点积)
        dots = xyz @ xyz[i]
        # 排除自己
        dots[i] = -2.0
        # 取前 6 个近邻
        neighbors = np.argsort(-dots)[:6]
        # 计算邻居在切平面上的角度排序
        ni = xyz[i] / (np.linalg.norm(xyz[i]) + 1e-15)
        # 构造切平面坐标系
        if abs(ni[2]) < 0.9:
            u = np.cross(ni, np.array([0.0, 0.0, 1.0]))
        else:
            u = np.cross(ni, np.array([0.0, 1.0, 0.0]))
        u = u / (np.linalg.norm(u) + 1e-15)
        v = np.cross(ni, u)
        v = v / (np.linalg.norm(v) + 1e-15)

        angles = []
        for j in neighbors:
            pj = xyz[j] - np.dot(xyz[j], ni) * ni
            pj = pj / (np.linalg.norm(pj) + 1e-15)
            angle = np.arctan2(np.dot(pj, v), np.dot(pj, u))
            angles.append(angle)
        order = np.argsort(angles)
        # 计算平面多边形面积 (近似)
        poly_area = 0.0
        for k in range(len(order)):
            k1 = order[k]
            k2 = order[(k + 1) % len(order)]
            # 近似边长
            a = np.arccos(np.clip(dots[neighbors[k1]], -1.0, 1.0))
            b = np.arccos(np.clip(dots[neighbors[k2]], -1.0, 1.0))
            # 夹角
            p1 = xyz[neighbors[k1]]
            p2 = xyz[neighbors[k2]]
            gamma = np.arccos(np.clip(np.dot(p1, p2), -1.0, 1.0))
            poly_area += spherical_triangle_area(ni, p1, p2)
        areas[i] = poly_area / 2.0  # 粗略修正

    faces = np.array([])
    return areas, faces


def voronoi_neighbor_adjacency(xyz: np.ndarray, faces: Optional[np.ndarray] = None) -> np.ndarray:
    """
    基于 Delaunay 三角剖分计算 Voronoi 邻接矩阵
    两个点是邻居当且仅当它们共享一个 Delaunay 边 (即属于同一个三角形)

    数学意义: 在信用风险网络中，邻接矩阵定义了区域间违约传染的拓扑结构。
    邻接关系对应于 Voronoi 单元共享边界，意味着两区域在经济地理上相邻。

    Parameters:
        xyz: 球面上的点 (n x 3)
        faces: Delaunay 面 (m x 3)，若 None 则重新计算

    Returns:
        adj: 邻接矩阵 (n x n)，布尔型
    """
    n = xyz.shape[0]
    adj = np.zeros((n, n), dtype=bool)

    if faces is None or len(faces) == 0:
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(xyz)
            faces = hull.simplices
        except ImportError:
            # 使用 k-NN 近似
            for i in range(n):
                dots = xyz @ xyz[i]
                dots[i] = -2.0
                neighbors = np.argsort(-dots)[:6]
                adj[i, neighbors] = True
            return adj

    for face in faces:
        i, j, k = face
        adj[i, j] = True
        adj[j, i] = True
        adj[i, k] = True
        adj[k, i] = True
        adj[j, k] = True
        adj[k, j] = True

    return adj


def build_regional_default_correlation(
    n_regions: int = 20,
    base_correlation: float = 0.3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    构建基于球面 Voronoi 的区域违约相关性模型

    流程:
        1. 在单位球面上生成 LLT 网格点作为区域中心
        2. 计算球面 Voronoi 面积作为区域经济权重
        3. 计算邻接矩阵作为传染网络
        4. 构建空间相关性矩阵:
           Corr(i,j) = base_correlation * exp(-d(i,j)^2 / (2*sigma^2)) + delta_{ij}*(1-base_correlation)
           其中 d(i,j) 为球面上两点之间的中心角

    Parameters:
        n_regions: 区域数量 (将映射到合适的 lat_num, long_num)
        base_correlation: 基础相关性水平

    Returns:
        xyz: 区域中心坐标
        areas: 区域面积权重
        adj: 邻接矩阵
        corr: 相关性矩阵
    """
    # 根据 n_regions 估算 lat_num 和 long_num
    # n_points = 2 + lat_num * long_num
    # 取 lat_num ~ long_num ~ sqrt(n_regions)
    lat_num = max(2, int(np.sqrt(n_regions)))
    long_num = max(3, int((n_regions - 2) / lat_num))

    xyz = sphere_llt_grid_points(1.0, np.zeros(3), lat_num, long_num)
    n = xyz.shape[0]

    areas, faces = spherical_voronoi_areas(xyz)
    # 归一化面积
    areas = areas / (areas.sum() + 1e-15)

    adj = voronoi_neighbor_adjacency(xyz, faces if len(faces) > 0 else None)

    # TODO(Hole_2): 补全球面高斯核空间相关性矩阵的科学计算
    # 科学背景: 在全球化信用组合中，不同地理区域的违约事件具有空间相关性。
    # 将区域中心投影到单位球面 S^2 上，利用球面距离构建相关性矩阵:
    #   Corr(i, j) = base_correlation * exp( - d(i,j)^2 / (2 * sigma^2) )
    # 其中 d(i,j) 为球面上两点 xyz[i] 与 xyz[j] 之间的中心角 (大圆距离)。
    # 中心角可通过向量点积计算: d = arccos( clip( dot(xyz[i], xyz[j]), -1, 1 ) )
    # sigma^2 在此处取 0.25 (即分母中的 0.5 = 2*sigma^2)。
    # 对角线元素保持为 1 (即 delta_{ij} 项)。
    # 当前返回占位矩阵，会导致 main.py 中的总相关性融合结果不正确。
    corr = np.eye(n, dtype=float)  # PLACEHOLDER: 请根据上述公式实现正确计算

    # 确保正定性
    from utils import nearest_correlation_matrix
    corr = nearest_correlation_matrix(corr)

    return xyz, areas, adj, corr


def test_spherical_grid():
    """测试球面网格与 Voronoi 计算"""
    xyz = sphere_llt_grid_points(1.0, np.zeros(3), 3, 4)
    assert xyz.shape[0] == 2 + 3 * 4, "点数计算错误"

    areas, faces = spherical_voronoi_areas(xyz)
    assert np.all(areas >= 0), "面积存在负值"
    total_area = areas.sum()
    assert abs(total_area - 4 * np.pi) < 0.5, f"总面积偏离 4pi: {total_area}"

    adj = voronoi_neighbor_adjacency(xyz, faces if len(faces) > 0 else None)
    assert adj.shape == (xyz.shape[0], xyz.shape[0]), "邻接矩阵维度错误"

    xyz2, areas2, adj2, corr = build_regional_default_correlation(n_regions=12)
    assert corr.shape[0] == xyz2.shape[0], "相关性矩阵维度不匹配"
    assert np.allclose(np.diag(corr), 1.0, atol=1e-5), "对角线不为 1"
    print(f"spherical_correlation_grid test passed. n_regions={xyz2.shape[0]}")


if __name__ == "__main__":
    test_spherical_grid()
