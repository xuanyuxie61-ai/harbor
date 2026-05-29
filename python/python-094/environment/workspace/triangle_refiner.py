"""
triangle_refiner.py
===================
三角形自适应细分与网格细化。

融合种子项目：
  - 1314_triangle_refine : 三角形细分（centroids, vertices, quad）
  - 1346_triangulation_q2l : 二次三角剖分到线性转换

科学应用：
  在冲击波数值模拟中，自适应网格细化 (AMR) 对捕捉陡峭梯度至关重要。
  本模块实现基于物理量的三角形自适应细分，包括重心细分、中点细分、
  以及二次到线性单元的转换，确保在冲击波前沿区域达到所需的局部精度。
"""

import numpy as np


def triangle_refine_centroids(c, t):
    r"""
    递归细分三角形并返回所有子三角形的重心。

    原始算法来自 1314_triangle_refine/triangle_refine_centroids.m。
    每次将三角形分为 4 个小三角形，计算其重心。

    Parameters
    ----------
    c : int
        递归层数。
    t : np.ndarray, shape (3, 2)
        父三角形顶点。

    Returns
    -------
    np.ndarray, shape (4^c, 2)
        所有子三角形重心。
    """
    t = np.asarray(t, dtype=float)
    if c == 0:
        centroid = np.mean(t, axis=0)
        return centroid.reshape(1, 2)

    # 三条边中点
    m01 = 0.5 * (t[0] + t[1])
    m12 = 0.5 * (t[1] + t[2])
    m20 = 0.5 * (t[2] + t[0])

    # 4 个子三角形
    sub_triangles = [
        np.vstack([t[0], m01, m20]),
        np.vstack([m01, t[1], m12]),
        np.vstack([m20, m12, t[2]]),
        np.vstack([m01, m12, m20])
    ]

    centroids = []
    for sub in sub_triangles:
        centroids.append(triangle_refine_centroids(c - 1, sub))
    return np.vstack(centroids)


def triangle_refine_num(c):
    """
    c 层细分后的子三角形数量。

    .. math::
        N_{sub} = 4^c
    """
    return 4 ** int(c)


def triangle_refine_quad(c, t, f):
    r"""
    在细分后的三角形上计算积分近似。

    原始算法来自 1314_triangle_refine/triangle_refine_quad.m。

    .. math::
        \int_T f(x) dx \approx \frac{A(T)}{N_{sub}} \sum_{i=1}^{N_{sub}} f(c_i)

    Parameters
    ----------
    c : int
        细分层数。
    t : np.ndarray, shape (3, 2)
        三角形顶点。
    f : callable
        被积函数 f(xy) -> float，可接受 shape (N, 2) 输入返回 shape (N,) 输出。

    Returns
    -------
    float
        积分近似值。
    """
    from geometry_utils import triangle_area
    centroids = triangle_refine_centroids(c, t)
    n = triangle_refine_num(c)
    area = triangle_area(t)
    if n == 0:
        return 0.0
    f_vals = f(centroids)
    return np.sum(f_vals) * area / n


def triangulation_q2l_to_linear(triangle_node1):
    """
    将 6 节点二次三角形转换为 4 个 3 节点线性三角形。

    原始算法来自 1346_triangulation_q2l/triangulation_order6_to_order3.m。

    6 节点二次三角形：
        3
       / \
      5---4
     /     \
    1---2---0  (局部编号)

    转换为 4 个线性三角形：
    [0, 3, 5], [1, 4, 3], [2, 5, 4], [3, 4, 5]

    Parameters
    ----------
    triangle_node1 : np.ndarray, shape (6, n_tri)
        二次三角形节点索引（0-based）。

    Returns
    -------
    np.ndarray, shape (3, 4*n_tri)
        线性三角形节点索引。
    """
    triangle_node1 = np.asarray(triangle_node1, dtype=int)
    if triangle_node1.ndim != 2:
        raise ValueError("triangle_node1 must be 2D.")
    if triangle_node1.shape[0] != 6:
        raise ValueError("triangle_node1 must have 6 rows (quadratic triangles).")

    n_tri = triangle_node1.shape[1]
    triangle_node2 = np.zeros((3, 4 * n_tri), dtype=int)

    for tri1 in range(n_tri):
        n = triangle_node1[:, tri1]
        tri2 = tri1 * 4
        triangle_node2[:, tri2] = [n[0], n[3], n[5]]
        triangle_node2[:, tri2 + 1] = [n[1], n[4], n[3]]
        triangle_node2[:, tri2 + 2] = [n[2], n[5], n[4]]
        triangle_node2[:, tri2 + 3] = [n[3], n[4], n[5]]

    return triangle_node2


def adaptive_triangle_refine(triangles, nodes, indicator_func, threshold,
                              max_level=3, current_level=0):
    r"""
    基于指示函数的自适应三角形细分。

    指示函数 :math:`\eta(T)` 评估每个三角形是否需要细分。
    若 :math:`\eta(T) > \eta_{threshold}`，则将该三角形细分为 4 个子三角形。

    Parameters
    ----------
    triangles : list of np.ndarray
        每个元素 shape (3, 2) 或 (3,) 索引。
    nodes : np.ndarray, shape (N_nodes, 2)
        节点坐标。
    indicator_func : callable
        指示函数 indicator_func(centroid) -> float。
    threshold : float
        细分阈值。
    max_level : int
        最大细分层数。
    current_level : int
        当前层数。

    Returns
    -------
    list of np.ndarray
        细分后的三角形列表（顶点坐标）。
    list of int
        每个三角形的细分层数。
    """
    refined_triangles = []
    refined_levels = []

    for tri in triangles:
        tri = np.asarray(tri)
        if tri.ndim == 1 and nodes is not None:
            tri_coords = nodes[tri, :]
        else:
            tri_coords = tri

        centroid = np.mean(tri_coords, axis=0)
        indicator = indicator_func(centroid)

        if indicator > threshold and current_level < max_level:
            # 细分
            m01 = 0.5 * (tri_coords[0] + tri_coords[1])
            m12 = 0.5 * (tri_coords[1] + tri_coords[2])
            m20 = 0.5 * (tri_coords[2] + tri_coords[0])
            sub_tris = [
                np.vstack([tri_coords[0], m01, m20]),
                np.vstack([m01, tri_coords[1], m12]),
                np.vstack([m20, m12, tri_coords[2]]),
                np.vstack([m01, m12, m20])
            ]
            sub_results, sub_levels = adaptive_triangle_refine(
                sub_tris, None, indicator_func, threshold,
                max_level, current_level + 1)
            refined_triangles.extend(sub_results)
            refined_levels.extend(sub_levels)
        else:
            refined_triangles.append(tri_coords)
            refined_levels.append(current_level)

    return refined_triangles, refined_levels


class AdaptiveMeshRefinement:
    """
    自适应网格细化管理器，用于冲击波捕捉。
    """

    def __init__(self, base_nodes, base_triangles, max_level=3):
        """
        Parameters
        ----------
        base_nodes : np.ndarray, shape (N, 2)
            基础网格节点。
        base_triangles : np.ndarray, shape (M, 3)
            基础三角形索引。
        max_level : int
            最大细分层数。
        """
        self.nodes = np.asarray(base_nodes, dtype=float)
        self.triangles = np.asarray(base_triangles, dtype=int)
        self.max_level = int(max_level)

    def refine_by_gradient(self, field_values, gradient_threshold):
        r"""
        基于场量梯度的自适应细分。

        对每条三角形，估计局部梯度：
        .. math::
            \eta(T) = \frac{\max_{e \in \partial T} |\Delta f_e|}{h_T}

        Parameters
        ----------
        field_values : np.ndarray, shape (N_nodes,)
            节点上的场值。
        gradient_threshold : float
            梯度阈值。

        Returns
        -------
        list of np.ndarray
            细分后的三角形坐标列表。
        list of int
            对应细分层数。
        """
        field_values = np.asarray(field_values, dtype=float)
        triangles_list = [self.nodes[tri, :] for tri in self.triangles]

        def indicator(centroid):
            # 找最近的三角形并估计梯度
            # 简化：基于重心处的场值插值
            # 实际中可用更精细的梯度估计
            dists = np.sum((self.nodes - centroid) ** 2, axis=1)
            nearest = np.argpartition(dists, min(2, len(dists) - 1))[:3]
            vals = field_values[nearest]
            return np.std(vals) / (np.mean(dists[nearest]) ** 0.5 + 1e-10)

        refined, levels = adaptive_triangle_refine(
            triangles_list, None, indicator, gradient_threshold, self.max_level)
        return refined, levels

    def compute_refined_integral(self, f, base_indicator=None, threshold=1.0):
        r"""
        在自适应细分网格上计算积分。

        .. math::
            I = \sum_{T} \int_T f(x) dx

        Parameters
        ----------
        f : callable
            被积函数。
        base_indicator : callable or None
            基础指示函数。
        threshold : float
            细分阈值。

        Returns
        -------
        float
            积分值。
        int
            最终三角形数量。
        """
        triangles_list = [self.nodes[tri, :] for tri in self.triangles]

        if base_indicator is None:
            base_indicator = lambda c: 0.0

        refined, levels = adaptive_triangle_refine(
            triangles_list, None, base_indicator, threshold, self.max_level)

        total = 0.0
        for tri in refined:
            # 使用 2 层细分求积分
            total += triangle_refine_quad(2, tri, f)

        return total, len(refined)
