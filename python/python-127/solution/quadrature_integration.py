"""
quadrature_integration.py
=========================
高精度三角形数值积分模块

基于种子项目:
  - 1325_triangle_witherden_rule: Witherden-Vincent 对称三角形求积规则

科学背景:
  有限元分析中，刚度矩阵和质量矩阵的组装需要计算
  三角形单元上的积分:
      ∫_T f(x,y) dA

  本模块实现高精度的对称求积规则，用于 FEM 刚度矩阵的精确积分。

  参考: Witherden, Vincent (2015), CAMWA 69:1232-1241.
"""

import numpy as np


# Witherden-Vincent 三角形求积规则数据 (precision 1-5)
# 规则定义在单位三角形: (0,0), (1,0), (0,1)
_RULES = {
    1: {
        'n': 1,
        'x': np.array([1.0/3.0]),
        'y': np.array([1.0/3.0]),
        'w': np.array([1.0]),
    },
    2: {
        'n': 3,
        'x': np.array([2.0/3.0, 1.0/6.0, 1.0/6.0]),
        'y': np.array([1.0/6.0, 2.0/3.0, 1.0/6.0]),
        'w': np.array([1.0/3.0, 1.0/3.0, 1.0/3.0]),
    },
    3: {
        'n': 6,
        'x': np.array([
            0.659027622374092, 0.231933368553031, 0.109039009072877,
            0.659027622374092, 0.231933368553031, 0.109039009072877
        ]),
        'y': np.array([
            0.231933368553031, 0.659027622374092, 0.231933368553031,
            0.109039009072877, 0.109039009072877, 0.659027622374092
        ]),
        'w': np.array([
            0.166666666666667, 0.166666666666667, 0.166666666666667,
            0.166666666666667, 0.166666666666667, 0.166666666666667
        ]),
    },
    4: {
        'n': 6,
        'x': np.array([
            0.816847572980459, 0.091576213509771, 0.091576213509771,
            0.108103018168070, 0.445948490915965, 0.445948490915965
        ]),
        'y': np.array([
            0.091576213509771, 0.816847572980459, 0.091576213509771,
            0.445948490915965, 0.108103018168070, 0.445948490915965
        ]),
        'w': np.array([
            0.109951743655322, 0.109951743655322, 0.109951743655322,
            0.223381589678011, 0.223381589678011, 0.223381589678011
        ]),
    },
    5: {
        'n': 7,
        'x': np.array([
            1.0/3.0,
            0.797426985353087, 0.101286507323456, 0.101286507323456,
            0.059715871789770, 0.470142064105115, 0.470142064105115
        ]),
        'y': np.array([
            1.0/3.0,
            0.101286507323456, 0.797426985353087, 0.101286507323456,
            0.470142064105115, 0.059715871789770, 0.470142064105115
        ]),
        'w': np.array([
            0.225000000000000,
            0.125939180544827, 0.125939180544827, 0.125939180544827,
            0.132394152788506, 0.132394152788506, 0.132394152788506
        ]),
    },
}


def triangle_quadrature_rule(precision):
    """
    获取三角形求积规则。

    Parameters
    ----------
    precision : int
        精度阶数 (1-5)

    Returns
    -------
    x, y, w : ndarray
        参考三角形上的求积点和权重
    """
    if precision not in _RULES:
        raise ValueError(f"precision 必须在 1-5 之间，当前 {precision}")
    rule = _RULES[precision]
    return rule['x'].copy(), rule['y'].copy(), rule['w'].copy()


def integrate_triangle(f, vertices, precision=4):
    """
    在任意三角形上积分函数 f(x,y)。

    Parameters
    ----------
    f : callable
        f(x, y) -> float
    vertices : ndarray, shape (3, 2)
        三角形顶点
    precision : int
        求积精度

    Returns
    -------
    integral : float
    """
    vertices = np.asarray(vertices, dtype=float)
    if vertices.shape != (3, 2):
        raise ValueError("vertices 必须为 (3, 2) 数组")

    x_ref, y_ref, w = triangle_quadrature_rule(precision)

    # 从参考三角形映射到实际三角形
    # P = P1 + x_ref*(P2-P1) + y_ref*(P3-P1)
    P1, P2, P3 = vertices
    detJ = abs((P2[0] - P1[0]) * (P3[1] - P1[1]) - (P3[0] - P1[0]) * (P2[1] - P1[1]))
    area_factor = detJ  # 参考三角形面积为 1/2，所以雅可比行列式 = 2*area

    P_x = P1[0] + x_ref * (P2[0] - P1[0]) + y_ref * (P3[0] - P1[0])
    P_y = P1[1] + x_ref * (P2[1] - P1[1]) + y_ref * (P3[1] - P1[1])

    integral = np.sum(w * f(P_x, P_y)) * area_factor * 0.5
    return integral


def integrate_over_mesh_elements(f, nodes, elements, precision=4):
    """
    在三角网格的所有单元上积分函数 f。

    Parameters
    ----------
    f : callable
    nodes : ndarray, shape (N, 2)
    elements : ndarray, shape (M, 3)
    precision : int

    Returns
    -------
    total_integral : float
    element_integrals : ndarray, shape (M,)
    """
    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)

    element_integrals = np.empty(elements.shape[0])
    for i, elem in enumerate(elements):
        verts = nodes[elem]
        element_integrals[i] = integrate_triangle(f, verts, precision)

    return np.sum(element_integrals), element_integrals


def triangle_monomial_integral(alpha, beta):
    """
    计算参考三角形上的单项式积分:
        ∫_T x^α y^β dxdy = α! β! / (α + β + 2)!

    Parameters
    ----------
    alpha, beta : int
        幂次

    Returns
    -------
    value : float
    """
    from math import factorial
    if alpha < 0 or beta < 0:
        raise ValueError("幂次必须非负")
    return factorial(alpha) * factorial(beta) / factorial(alpha + beta + 2)


def test_quadrature_precision(max_precision=5):
    """
    测试求积规则的精度。

    Returns
    -------
    errors : dict
        {precision: max_error}
    """
    errors = {}
    for p in range(1, max_precision + 1):
        max_err = 0.0
        x, y, w = triangle_quadrature_rule(p)
        # 测试 x^a y^b, a+b <= p
        for a in range(p + 1):
            for b in range(p + 1 - a):
                exact = triangle_monomial_integral(a, b)
                numerical = np.sum(w * (x**a) * (y**b)) * 0.5
                err = abs(exact - numerical)
                max_err = max(max_err, err)
        errors[p] = max_err
    return errors
