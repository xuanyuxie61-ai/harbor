r"""
causal_mesh_interpolator.py
================================================================================
基于有限元三角网格插值的因果场空间重构与区域判定

原项目映射:
- 425_ffmatlib — FreeFem++ 三角网格上的 PDE 数据插值
- 109_boundary_word_right — 多边形点包含判定（射线投射算法）

科学背景
--------
在地理流行病学、气候归因等因果推断场景中，观测站点往往呈不规则分布。
我们需要将离散观测点上的因果效应估计插值到连续空间域，以便：
1. 计算因果场的空间梯度（识别因果边界）
2. 判定某空间位置是否处于显著因果影响区域
3. 在三角形有限元网格上积分总因果效应

核心公式
--------
1. 三角单元上的线性形函数（P1 有限元）：
   对于三角形节点 $\mathbf{p}_1, \mathbf{p}_2, \mathbf{p}_3$，任意点 $\mathbf{x}$ 的
   重心坐标 $(\lambda_1, \lambda_2, \lambda_3)$ 满足：
   $$ \lambda_1 + \lambda_2 + \lambda_3 = 1, \quad
      \mathbf{x} = \lambda_1 \mathbf{p}_1 + \lambda_2 \mathbf{p}_2 + \lambda_3 \mathbf{p}_3 $$
   解为：
   $$ \lambda_1 = \frac{A_1}{A}, \quad \lambda_2 = \frac{A_2}{A}, \quad \lambda_3 = \frac{A_3}{A} $$
   其中 $A$ 为三角形总面积，$A_i$ 为对顶点子三角形面积。

2. 线性插值：
   $$ u(\mathbf{x}) = \lambda_1 u_1 + \lambda_2 u_2 + \lambda_3 u_3 $$

3. 三角形面积（行列式形式）：
   $$ A = \frac{1}{2}\left| (x_2-x_1)(y_3-y_1) - (x_3-x_1)(y_2-y_1) \right| $$

4. 点包含判定（射线投射算法，原项目 polygon_contains_point）：
   从查询点向右发射水平射线，计算与多边形边界的交点个数。
   若交点数为奇数，则点在内部。
r"""

import numpy as np
from typing import Tuple, List, Optional


def triangle_area(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    r"""
    计算三角形的有向面积。
    r"""
    return 0.5 * ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))


def barycentric_coordinates(x: np.ndarray,
                            p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Tuple[float, float, float]:
    r"""
    计算点 x 在三角形 (p1,p2,p3) 中的重心坐标。

    Returns
    -------
    l1, l2, l3 : float
        若均在 [0,1] 内且和为 1，则点在三角形内（含边界）。
    r"""
    A = triangle_area(p1, p2, p3)
    if abs(A) < 1e-14:
        return -1.0, -1.0, -1.0
    A1 = triangle_area(x, p2, p3)
    A2 = triangle_area(p1, x, p3)
    A3 = triangle_area(p1, p2, x)
    l1 = A1 / A
    l2 = A2 / A
    l3 = A3 / A
    return l1, l2, l3


def point_in_triangle(x: np.ndarray,
                      p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
                      tol: float = 1e-10) -> bool:
    r"""
    判断点是否在三角形内（含边界）。
    r"""
    l1, l2, l3 = barycentric_coordinates(x, p1, p2, p3)
    return (l1 >= -tol) and (l2 >= -tol) and (l3 >= -tol) and abs(l1 + l2 + l3 - 1.0) < 1e-8


def interpolate_on_triangle(x: np.ndarray,
                            p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
                            u1: float, u2: float, u3: float) -> float:
    r"""
    在三角形上做线性插值估计因果场值。
    r"""
    l1, l2, l3 = barycentric_coordinates(x, p1, p2, p3)
    if l1 < -1e-8 or l2 < -1e-8 or l3 < -1e-8:
        return np.nan
    return l1 * u1 + l2 * u2 + l3 * u3


def find_containing_triangle(x: np.ndarray,
                              points: np.ndarray,
                              triangles: np.ndarray) -> Tuple[int, np.ndarray]:
    r"""
    在三角网格中查找包含点 x 的三角形。

    Parameters
    ----------
    x : ndarray, shape (2,)
        查询点。
    points : ndarray, shape (n_points, 2)
        网格节点坐标。
    triangles : ndarray, shape (n_triangles, 3)
        三角形连通性（节点索引）。

    Returns
    -------
    tri_idx : int
        三角形索引，若未找到返回 -1。
    bary : ndarray, shape (3,)
        重心坐标。
    r"""
    for tri_idx in range(triangles.shape[0]):
        nodes = triangles[tri_idx]
        p1 = points[nodes[0]]
        p2 = points[nodes[1]]
        p3 = points[nodes[2]]
        if point_in_triangle(x, p1, p2, p3):
            l1, l2, l3 = barycentric_coordinates(x, p1, p2, p3)
            return tri_idx, np.array([l1, l2, l3])
    return -1, np.zeros(3)


def interpolate_mesh_field(points: np.ndarray,
                            triangles: np.ndarray,
                            field_values: np.ndarray,
                            query_points: np.ndarray) -> np.ndarray:
    r"""
    将定义在三角网格节点上的因果场插值到查询点集。

    Parameters
    ----------
    points : ndarray, shape (n_nodes, 2)
    triangles : ndarray, shape (n_triangles, 3)
    field_values : ndarray, shape (n_nodes,)
    query_points : ndarray, shape (n_queries, 2)

    Returns
    -------
    interp : ndarray, shape (n_queries,)
        插值结果，若某点不在任何三角形内则为 np.nan。
    r"""
    nq = query_points.shape[0]
    interp = np.full(nq, np.nan)
    for iq in range(nq):
        tri_idx, bary = find_containing_triangle(query_points[iq], points, triangles)
        if tri_idx >= 0:
            nodes = triangles[tri_idx]
            interp[iq] = bary[0] * field_values[nodes[0]] + \
                         bary[1] * field_values[nodes[1]] + \
                         bary[2] * field_values[nodes[2]]
    return interp


def polygon_contains_point(poly: np.ndarray, q: np.ndarray) -> bool:
    r"""
    射线投射算法判定二维点 q 是否在简单多边形 poly 内部。

    原项目 109_boundary_word_right 核心算法。
    处理边界情况（点恰好在边上）返回 True。

    Parameters
    ----------
    poly : ndarray, shape (n, 2)
        多边形顶点（按顺序）。
    q : ndarray, shape (2,)
        查询点。

    Returns
    -------
    inside : bool
    r"""
    n = poly.shape[0]
    inside = False
    x1, y1 = poly[n - 1]
    for i in range(n):
        x2, y2 = poly[i]
        # 检查是否在边上
        cross = (q[1] - y1) * (x2 - x1) - (y2 - y1) * (q[0] - x1)
        if abs(cross) < 1e-12:
            # 在直线上，检查是否在线段内
            if min(x1, x2) - 1e-12 <= q[0] <= max(x1, x2) + 1e-12 and \
               min(y1, y2) - 1e-12 <= q[1] <= max(y1, y2) + 1e-12:
                return True
        # 射线投射
        if ((y1 > q[1]) != (y2 > q[1])):
            xinters = (q[1] - y1) * (x2 - x1) / (y2 - y1) + x1
            if xinters > q[0]:
                inside = not inside
        x1, y1 = x2, y2
    return inside


def integrate_field_over_mesh(points: np.ndarray,
                               triangles: np.ndarray,
                               field_values: np.ndarray) -> float:
    r"""
    在三角网格上积分因果场（P1 有限元质量矩阵求和）。

    $$ \int_{\Omega} u(\mathbf{x})\,d\mathbf{x} \approx \sum_{T} A_T \cdot \frac{u_1+u_2+u_3}{3} $$
    r"""
    total = 0.0
    for tri in triangles:
        p1, p2, p3 = points[tri[0]], points[tri[1]], points[tri[2]]
        A = abs(triangle_area(p1, p2, p3))
        avg_val = (field_values[tri[0]] + field_values[tri[1]] + field_values[tri[2]]) / 3.0
        total += A * avg_val
    return total


def demo():
    r"""模块自测试。"""
    # 构造简单三角网格（单位正方形对角剖分）
    points = np.array([
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]
    ])
    triangles = np.array([
        [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]
    ])
    # 定义节点上的因果场值
    field = np.array([0.0, 1.0, 2.0, 1.0, 1.5])

    query = np.array([[0.6, 0.4], [0.2, 0.8], [1.5, 0.5]])
    interp = interpolate_mesh_field(points, triangles, field, query)
    print(f"[causal_mesh_interpolator] 插值结果: {interp}")

    total = integrate_field_over_mesh(points, triangles, field)
    print(f"[causal_mesh_interpolator] 网格上因果场积分: {total:.4f}")

    # 多边形包含测试
    poly = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    print(f"[causal_mesh_interpolator] 点(0.5,0.5)在多边形内? {polygon_contains_point(poly, np.array([0.5, 0.5]))}")
    return interp, total


if __name__ == "__main__":
    demo()
