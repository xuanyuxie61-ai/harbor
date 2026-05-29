"""
quadrature_rules.py
高精度数值求积规则模块

实现参考三角形上的对称求积规则，支持从等边三角形到
标准参考三角形 (0,0)-(1,0)-(0,1) 的坐标转换。

核心数学：
    - 参考三角形 T_ref:
        顶点 v1=(0,0), v2=(1,0), v3=(0,1)
      
    - 等边三角形 T_eq:
        顶点可映射到 T_ref 通过仿射变换
    
    - 对称求积规则（基于 Xiao-Gimbutas 算法）:
        integral_{T} f(x,y) dx dy approx sum_{i=1}^{N} w_i * f(x_i, y_i)
      
      对于 d 次多项式精确，N 约为 (d+1)(d+2)/6（最优规则）。
    
    - 坐标变换:
        给定等边三角形顶点 vert1, vert2, vert3,
        映射到参考三角形后，节点为 (x_i, y_i, z_i=1-x_i-y_i)。
    
    - 雅可比行列式:
        |J| = 2 * Area(T_ref) = 1.0  （对标准参考三角形）
        
        一般三角形:
            |J| = 2 * Area(T_physical)
      
      因此积分换元:
        integral_{T_phy} f(x,y) dx dy = |J|/2 * integral_{T_ref} f(Xi(r,s)) dr ds
                                      = Area(T_phy) * sum_i w_i * f(x_i, y_i)
    
    - 常用低阶规则:
        * 1点规则（重心，精度1）:
            w=1/2, (1/3, 1/3)
        
        * 3点规则（边中点，精度2）:
            w=1/6, (1/2,0), (0,1/2), (1/2,1/2)
        
        * 4点规则（精度3）:
            w1=-9/32, (1/3,1/3)
            w2=25/96, (3/5,1/5), (1/5,3/5), (1/5,1/5)
        
        * 7点规则（精度5）:
            含重心、边点和内点
"""

import numpy as np
from typing import Tuple
from utils import compute_triangle_area


def triangle_quad_rule_degree1() -> Tuple[np.ndarray, np.ndarray]:
    """
    1点重心求积规则，精确到1次多项式。
    
    Returns
    -------
    weights : np.ndarray, shape (1,)
    points : np.ndarray, shape (1, 2)
    """
    weights = np.array([0.5])
    points = np.array([[1.0 / 3.0, 1.0 / 3.0]])
    return weights, points


def triangle_quad_rule_degree2() -> Tuple[np.ndarray, np.ndarray]:
    """
    3点边中点求积规则，精确到2次多项式。
    
    Returns
    -------
    weights : np.ndarray, shape (3,)
    points : np.ndarray, shape (3, 2)
    """
    w = 1.0 / 6.0
    weights = np.array([w, w, w])
    points = np.array([
        [0.5, 0.0],
        [0.5, 0.5],
        [0.0, 0.5]
    ])
    return weights, points


def triangle_quad_rule_degree3() -> Tuple[np.ndarray, np.ndarray]:
    """
    4点求积规则，精确到3次多项式。
    
    Returns
    -------
    weights : np.ndarray, shape (4,)
    points : np.ndarray, shape (4, 2)
    """
    w1 = -9.0 / 32.0
    w2 = 25.0 / 96.0
    weights = np.array([w1, w2, w2, w2])
    points = np.array([
        [1.0 / 3.0, 1.0 / 3.0],
        [3.0 / 5.0, 1.0 / 5.0],
        [1.0 / 5.0, 3.0 / 5.0],
        [1.0 / 5.0, 1.0 / 5.0]
    ])
    return weights, points


def triangle_quad_rule_degree5() -> Tuple[np.ndarray, np.ndarray]:
    """
    7点求积规则，精确到5次多项式（Stroud 规则）。
    
    Returns
    -------
    weights : np.ndarray, shape (7,)
    points : np.ndarray, shape (7, 2)
    """
    # 参数
    a1 = (6.0 + np.sqrt(15.0)) / 21.0
    a2 = (6.0 - np.sqrt(15.0)) / 21.0
    b1 = (4.0 + np.sqrt(15.0)) / 7.0  # 1 - 2*a1
    b2 = (4.0 - np.sqrt(15.0)) / 7.0  # 1 - 2*a2

    w1 = (155.0 - np.sqrt(15.0)) / 1200.0
    w2 = (155.0 + np.sqrt(15.0)) / 1200.0
    w3 = 9.0 / 40.0

    weights = np.array([
        w1, w1, w1,   # (a1, a1), (1-2a1, a1), (a1, 1-2a1)
        w2, w2, w2,   # (a2, a2), (1-2a2, a2), (a2, 1-2a2)
        w3            # (1/3, 1/3)
    ])

    points = np.array([
        [a1, a1],
        [b1, a1],
        [a1, b1],
        [a2, a2],
        [b2, a2],
        [a2, b2],
        [1.0 / 3.0, 1.0 / 3.0]
    ])
    return weights, points


def integrate_over_triangle(vertices: np.ndarray,
                            f: callable,
                            degree: int = 3) -> float:
    """
    在任意三角形上数值积分函数 f(x,y)。
    
    采用等参变换将物理三角形映射到参考三角形，
    然后使用求积规则:
        integral_{T} f(x,y) dx dy = |det(J)| * sum_i w_i * f(x_i, y_i)
    
    其中 (x_i, y_i) 为物理空间求积点，|det(J)| = 2 * Area(T)。
    
    Parameters
    ----------
    vertices : np.ndarray, shape (3, 2)
        三角形顶点
    f : callable
        被积函数 f(x,y) -> float 或 array
    degree : int
        求积规则精度（1, 2, 3, 5）
    
    Returns
    -------
    float
        积分值
    """
    vertices = np.asarray(vertices, dtype=float)
    if vertices.shape != (3, 2):
        raise ValueError("vertices must have shape (3, 2)")

    area = abs(compute_triangle_area(vertices[0], vertices[1], vertices[2]))
    if area < 1e-14:
        return 0.0

    if degree <= 1:
        weights, points_ref = triangle_quad_rule_degree1()
    elif degree == 2:
        weights, points_ref = triangle_quad_rule_degree2()
    elif degree == 3:
        weights, points_ref = triangle_quad_rule_degree3()
    else:
        weights, points_ref = triangle_quad_rule_degree5()

    # 参考到物理的仿射变换
    # x(r,s) = x1 + (x2-x1)*r + (x3-x1)*s
    # y(r,s) = y1 + (y2-y1)*r + (y3-y1)*s
    x1, y1 = vertices[0]
    dx2, dy2 = vertices[1] - vertices[0]
    dx3, dy3 = vertices[2] - vertices[0]

    total = 0.0
    for i in range(len(weights)):
        r, s = points_ref[i]
        x = x1 + dx2 * r + dx3 * s
        y = y1 + dy2 * r + dy3 * s
        total += weights[i] * float(f(x, y))

    # 雅可比行列式 = 2 * Area
    return 2.0 * area * total


def integrate_over_mesh(nodes: np.ndarray, triangles: np.ndarray,
                        f: callable, degree: int = 3) -> float:
    """
    在整个三角网格上积分函数 f。
    
    Parameters
    ----------
    nodes : np.ndarray, shape (n_nodes, 2)
    triangles : np.ndarray, shape (n_tri, 3)
        1-based 索引
    f : callable
    degree : int
    
    Returns
    -------
    float
        全局积分值
    """
    total = 0.0
    for e in range(triangles.shape[0]):
        verts = nodes[triangles[e] - 1]
        total += integrate_over_triangle(verts, f, degree)
    return total


def compute_moment_over_mesh(nodes: np.ndarray, triangles: np.ndarray,
                              order_x: int = 0, order_y: int = 0,
                              degree: int = 3) -> float:
    """
    计算网格上的矩量: integral x^{order_x} * y^{order_y} dx dy。
    
    Parameters
    ----------
    nodes : np.ndarray
    triangles : np.ndarray
    order_x, order_y : int
    degree : int
    
    Returns
    -------
    float
        矩量值
    """
    def f(x, y):
        return (x ** order_x) * (y ** order_y)
    return integrate_over_mesh(nodes, triangles, f, degree)
