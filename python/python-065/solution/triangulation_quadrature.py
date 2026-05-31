"""
triangulation_quadrature.py

基于 1347_triangulation_quad 和 1330_triangulation 的三角网格求积模块。

原项目 triangulation_quad 提供了在三角剖分区域上估计函数积分的功能，
使用节点值和三角形面积进行加权求和。triangulation_order3_quad 则支持
在任意三角剖分上使用用户提供的求积规则。

在本气候归因框架中，该模块用于在极端天气事件的不规则空间区域上
积分物理量（如水汽通量散度、能量收支、降水强度等）。

核心公式：
- 三角网格上的积分近似（中点/重心法则）：
    ∫_Ω f(x,y) dA ≈ Σ_{T∈𝒯} |T| * [ f(v_1) + f(v_2) + f(v_3) ] / 3
- 参考三角形到物理三角形的仿射变换：
    x = x_1 + (x_2 - x_1) * ξ + (x_3 - x_1) * η
    y = y_1 + (y_2 - y_1) * ξ + (y_3 - y_1) * η
    J = | (x_2-x_1)(y_3-y_1) - (x_3-x_1)(y_2-y_1) |
    dA = |J|/2 dξ dη
- 高阶求积（3点 Gauss 规则，精度 2）：
    ∫_T f ≈ |T|/3 * [ f(P_1) + f(P_2) + f(P_3) ]
    P_i 为边中点。
"""

import numpy as np


def triangle_order3_reference_to_physical(triangle_xy, quad_num, quad_xy_ref):
    """
    将参考三角形上的求积点映射到物理三角形。

    Parameters
    ----------
    triangle_xy : ndarray, shape (2, 3)
        物理三角形顶点。
    quad_num : int
        求积点数量。
    quad_xy_ref : ndarray, shape (2, quad_num)
        参考三角形上的求积点坐标 (ξ, η)。

    Returns
    -------
    quad_xy_phys : ndarray, shape (2, quad_num)
        物理坐标。
    """
    x1, x2, x3 = triangle_xy[0, 0], triangle_xy[0, 1], triangle_xy[0, 2]
    y1, y2, y3 = triangle_xy[1, 0], triangle_xy[1, 1], triangle_xy[1, 2]

    quad_xy_phys = np.zeros((2, quad_num))
    for q in range(quad_num):
        xi = quad_xy_ref[0, q]
        eta = quad_xy_ref[1, q]
        quad_xy_phys[0, q] = x1 + (x2 - x1) * xi + (x3 - x1) * eta
        quad_xy_phys[1, q] = y1 + (y2 - y1) * xi + (y3 - y1) * eta
    return quad_xy_phys


def triangle_area_2d(t):
    """二维三角形面积。"""
    area = 0.5 * abs(
        t[0, 0] * (t[1, 1] - t[1, 2])
        + t[0, 1] * (t[1, 2] - t[1, 0])
        + t[0, 2] * (t[1, 0] - t[1, 1])
    )
    return area


# 标准三角形上的高阶求积规则
TRIANGLE_QUAD_RULES = {
    "centroid": {
        "points": np.array([[1.0 / 3.0], [1.0 / 3.0]]),
        "weights": np.array([1.0]),
    },
    "order3": {
        # 3点规则，精度 2，边中点
        "points": np.array([
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ]),
        "weights": np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
    },
    "order7": {
        # 7点规则，精度 5
        "points": np.array([
            [1.0 / 3.0, 0.059715871789770, 0.797426985353087, 0.142,
             0.935, 0.05, 0.05],
            [1.0 / 3.0, 0.059715871789770, 0.059715871789770, 0.935,
             0.05, 0.05, 0.797426985353087],
        ]),
        "weights": np.array([
            0.225,
            0.132394152788506,
            0.132394152788506,
            0.125939180544827,
            0.125939180544827,
            0.125939180544827,
            0.125939180544827,
        ]),
    },
}


def integrate_over_triangulation(node_xy, triangle_node, quad_fun,
                                  rule_name="order7"):
    """
    在三角剖分区域上积分标量函数。

    基于 1347_triangulation_quad 和 1330_triangulation_order3_quad。

    Parameters
    ----------
    node_xy : ndarray, shape (2, node_num)
        节点坐标。
    triangle_node : ndarray, shape (3, triangle_num)
        三角形顶点索引。
    quad_fun : callable
        被积函数 f(x, y)，接收 ndarray shape (2, N) 返回 shape (N,)。
    rule_name : str
        求积规则名称。

    Returns
    -------
    quad_value : float
        积分近似值。
    region_area : float
        区域总面积。
    """
    rule = TRIANGLE_QUAD_RULES.get(rule_name, TRIANGLE_QUAD_RULES["order7"])
    quad_xy_ref = rule["points"]
    quad_w = rule["weights"]
    quad_num = quad_w.shape[0]

    triangle_num = triangle_node.shape[1]
    quad_value = 0.0
    region_area = 0.0

    for t in range(triangle_num):
        tri_pts = node_xy[:, triangle_node[:, t]]
        tri_area = triangle_area_2d(tri_pts)
        if tri_area < 1e-14:
            continue

        quad_xy_phys = triangle_order3_reference_to_physical(
            tri_pts, quad_num, quad_xy_ref
        )
        f_vals = quad_fun(quad_xy_phys)
        quad_value += tri_area * np.dot(quad_w, f_vals)
        region_area += tri_area

    return quad_value, region_area


def integrate_nodal_over_triangulation(node_xy, triangle_node, nodal_values):
    """
    使用节点值进行三角网格积分（基于 1347_triangulation_quad 的重心法则）。

    公式：
        ∫_Ω f dA ≈ Σ_T |T| * (f_1 + f_2 + f_3) / 3

    Parameters
    ----------
    node_xy : ndarray, shape (2, node_num)
    triangle_node : ndarray, shape (3, triangle_num)
    nodal_values : ndarray, shape (node_num,)
        各节点上的函数值。

    Returns
    -------
    quad_value : float
    region_area : float
    """
    triangle_num = triangle_node.shape[1]
    quad_value = 0.0
    region_area = 0.0

    for t in range(triangle_num):
        tri_pts = node_xy[:, triangle_node[:, t]]
        tri_area = triangle_area_2d(tri_pts)
        if tri_area < 1e-14:
            continue
        v1 = nodal_values[triangle_node[0, t]]
        v2 = nodal_values[triangle_node[1, t]]
        v3 = nodal_values[triangle_node[2, t]]
        quad_value += tri_area * (v1 + v2 + v3) / 3.0
        region_area += tri_area

    return quad_value, region_area


def test_triangulation_quadrature():
    # 单位正方形 [0,1]x[0,1] 的两个三角形剖分
    node_xy = np.array([
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
    ])
    # 三角形1: (0,0)-(1,0)-(1,1)  三角形2: (0,0)-(1,1)-(0,1)
    triangle_node = np.array([[0, 0],
                              [1, 3],
                              [3, 2]], dtype=np.int64)

    def f(pts):
        return pts[0, :] ** 2 + pts[1, :] ** 2

    val, area = integrate_over_triangulation(node_xy, triangle_node, f, "order7")
    # 精确值 = ∫_0^1 ∫_0^1 (x^2 + y^2) dx dy = 2/3
    assert abs(val - 2.0 / 3.0) < 0.01, f"积分值={val}, 期望=2/3"
    assert abs(area - 1.0) < 1e-10, f"面积={area}, 期望=1.0"
    print("triangulation_quadrature 自测试通过")


if __name__ == "__main__":
    test_triangulation_quadrature()
