"""
sphere_quad.py
==============
基于种子项目 1130_sphere_triangle_quad 的球面求积模块。
在单位球面的三角形区域上实现高阶数值积分，
用于物理信息 GAN 中三维湍流谱在球壳上的积分以及判别器的球谐分析。

核心数学：
  1. 球面三角形顶点归一化：
       v_i' = v_i / ||v_i||

  2. 球面边长（中心角）：
       a = arccos(v2·v3),  b = arccos(v3·v1),  c = arccos(v1·v2)

  3. L'Huilier 定理（球面盈量）：
       s = (a+b+c)/2
       tan(E/4) = √( tan(s/2)·tan((s-a)/2)·tan((s-b)/2)·tan((s-c)/2) )
       area = E · R²   （单位球面 R=1，area = E）

  4. 球面三角形的重心（centroid）：
       顶点向量之和归一化：
         c = (v1 + v2 + v3) / ||v1 + v2 + v3||

  5. 球面三角形上的 3点中点规则（icos1v）：
       取三边中点（单位化后）作为求积节点，权重各为 area/3。

  6. 球面三角形上的 7点规则（icos2v，含顶点与边中点）：
       3个顶点（权重 w_v）+ 3个边中点（权重 w_m）+ 1个重心（权重 w_c）
       总权重满足 3·w_v + 3·w_m + w_c = area。
"""

import numpy as np


def r8vec_normalize(v: np.ndarray) -> np.ndarray:
    """归一化向量。"""
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    if norm < 1e-15:
        return v
    return v / norm


def sphere01_distance_xyz(v1: np.ndarray, v2: np.ndarray) -> float:
    """单位球面上两点之间的中心角距离。"""
    v1 = r8vec_normalize(v1)
    v2 = r8vec_normalize(v2)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return float(np.arccos(dot))


def sphere01_triangle_vertices_to_angles(v1: np.ndarray, v2: np.ndarray,
                                         v3: np.ndarray) -> tuple:
    """
    由球面三角形顶点计算三条球面边长（中心角）。

    Returns
    -------
    a, b, c : float
        分别对边 v1, v2, v3 的球面边长。
    """
    v1 = r8vec_normalize(v1)
    v2 = r8vec_normalize(v2)
    v3 = r8vec_normalize(v3)
    a = sphere01_distance_xyz(v2, v3)
    b = sphere01_distance_xyz(v3, v1)
    c = sphere01_distance_xyz(v1, v2)
    return a, b, c


def sphere01_triangle_vertices_to_area(v1: np.ndarray, v2: np.ndarray,
                                       v3: np.ndarray) -> float:
    """
    计算单位球面三角形面积（L'Huilier 定理）。
    """
    a, b, c = sphere01_triangle_vertices_to_angles(v1, v2, v3)
    s = 0.5 * (a + b + c)
    # 边界处理
    if s > np.pi - 1e-12:
        return 2.0 * np.pi
    tan_s2 = np.tan(s * 0.5)
    tan_as = np.tan(max(0.0, (s - a) * 0.5))
    tan_bs = np.tan(max(0.0, (s - b) * 0.5))
    tan_cs = np.tan(max(0.0, (s - c) * 0.5))
    prod = tan_s2 * tan_as * tan_bs * tan_cs
    prod = max(prod, 0.0)
    E = 4.0 * np.arctan(np.sqrt(prod))
    return float(E)


def sphere01_triangle_vertices_to_centroid(v1: np.ndarray, v2: np.ndarray,
                                           v3: np.ndarray) -> np.ndarray:
    """球面三角形重心（顶点向量和的归一化）。"""
    v1 = r8vec_normalize(v1)
    v2 = r8vec_normalize(v2)
    v3 = r8vec_normalize(v3)
    c = v1 + v2 + v3
    return r8vec_normalize(c)


def sphere01_triangle_vertices_to_midpoints(v1: np.ndarray, v2: np.ndarray,
                                            v3: np.ndarray) -> tuple:
    """球面三角形三边中点（边向量和的归一化）。"""
    v1 = r8vec_normalize(v1)
    v2 = r8vec_normalize(v2)
    v3 = r8vec_normalize(v3)
    m12 = r8vec_normalize(v1 + v2)
    m23 = r8vec_normalize(v2 + v3)
    m31 = r8vec_normalize(v3 + v1)
    return m12, m23, m31


def sphere01_triangle_quad_03(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray,
                              f) -> float:
    """
    球面三角形上的 3点中点求积规则（icos1v 近似）。

    Parameters
    ----------
    v1, v2, v3 : np.ndarray, shape (3,)
        球面三角形顶点。
    f : callable
        被积函数 f(x) 其中 x 为 shape (3,) 向量。

    Returns
    -------
    result : float
        积分近似值。
    """
    area = sphere01_triangle_vertices_to_area(v1, v2, v3)
    m12, m23, m31 = sphere01_triangle_vertices_to_midpoints(v1, v2, v3)
    val = f(m12) + f(m23) + f(m31)
    return area * val / 3.0


def sphere01_triangle_quad_07(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray,
                              f) -> float:
    """
    球面三角形上的 7点求积规则（顶点 + 边中点 + 重心）。

    权重分配：
      顶点：w_v = area / 20
      边中点：w_m = area / 20
      重心：w_c = 9·area / 20
    满足 3·w_v + 3·w_m + w_c = area。
    """
    area = sphere01_triangle_vertices_to_area(v1, v2, v3)
    m12, m23, m31 = sphere01_triangle_vertices_to_midpoints(v1, v2, v3)
    c = sphere01_triangle_vertices_to_centroid(v1, v2, v3)
    w_v = area / 20.0
    w_m = area / 20.0
    w_c = 9.0 * area / 20.0
    result = (w_v * (f(v1) + f(v2) + f(v3))
              + w_m * (f(m12) + f(m23) + f(m31))
              + w_c * f(c))
    return float(result)


def icosahedron_faces() -> list:
    """
    返回单位球内接正二十面体的 20 个球面三角形面。
    每个面为顶点索引三元组，对应 vertices 数组。

    Returns
    -------
    faces : list of tuple
        索引三元组列表。
    vertices : np.ndarray, shape (12, 3)
        正二十面体顶点坐标。
    """
    phi = 0.5 * (1.0 + np.sqrt(5.0))
    vertices = np.array([
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
    # 归一化到单位球面
    vertices = vertices / np.linalg.norm(vertices[0])

    faces = [
        (0, 2, 8), (0, 8, 4), (0, 4, 6), (0, 6, 10), (0, 10, 2),
        (3, 1, 11), (3, 11, 7), (3, 7, 5), (3, 5, 9), (3, 9, 1),
        (1, 4, 6), (1, 6, 11), (1, 9, 4), (11, 6, 10), (11, 10, 7),
        (7, 10, 2), (7, 2, 5), (5, 2, 8), (5, 8, 9), (9, 8, 4),
    ]
    return faces, vertices


def integrate_on_sphere(f, rule: str = "icos1v") -> float:
    """
    在整个单位球面上数值积分函数 f。
    使用正二十面体剖分 + 球面三角形求积。

    Parameters
    ----------
    f : callable
        被积函数 f(x) 其中 x 为 shape (3,) 向量。
    rule : str
        "icos1v"（3点中点规则）或 "icos2v"（7点规则）。

    Returns
    -------
    result : float
        球面积分近似值（理论上应等于 4π·f_avg）。
    """
    faces, vertices = icosahedron_faces()
    total = 0.0
    for tri in faces:
        v1 = vertices[tri[0]]
        v2 = vertices[tri[1]]
        v3 = vertices[tri[2]]
        if rule == "icos1v":
            total += sphere01_triangle_quad_03(v1, v2, v3, f)
        else:
            total += sphere01_triangle_quad_07(v1, v2, v3, f)
    return float(total)
