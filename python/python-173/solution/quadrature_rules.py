"""
高阶数值积分规则模块

融合自:
- 1256_tetrahedron_witherden_rule: 四面体高斯求积规则
- 229_cube_arbq_rule: 立方体高斯求积规则

本模块为二维三角形单元提供高阶数值积分，支持 FEM 刚度矩阵与载荷向量的精确组装。
对于参考三角形 T_ref = {(xi, eta): xi >= 0, eta >= 0, xi + eta <= 1}，
采用 Duffy 变换将其映射到标准正方形 [-1,1]^2:
    xi  = (1 + s) * (1 - t) / 4
    eta = (1 + t) / 2
    |J| = (1 - t) / 8
"""

import numpy as np


class TriangleQuadrature:
    """
    三角形参考域上的高阶求积规则。
    积分公式：
        ∫∫_{T_ref} f(xi, eta) dxi deta ≈ Σ_i w_i * f(xi_i, eta_i)
    """

    _rules = {
        1: {
            'points': np.array([[1.0 / 3.0, 1.0 / 3.0]]),
            'weights': np.array([0.5])
        },
        2: {
            'points': np.array([
                [1.0 / 6.0, 1.0 / 6.0],
                [2.0 / 3.0, 1.0 / 6.0],
                [1.0 / 6.0, 2.0 / 3.0]
            ]),
            'weights': np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        },
        3: {
            'points': np.array([
                [1.0 / 3.0, 1.0 / 3.0],
                [0.6, 0.2],
                [0.2, 0.6],
                [0.2, 0.2]
            ]),
            'weights': np.array([
                -27.0 / 96.0,
                25.0 / 96.0,
                25.0 / 96.0,
                25.0 / 96.0
            ]) * 0.5
        },
        4: {
            'points': np.array([
                [0.108103018168070, 0.445948490915965],
                [0.445948490915965, 0.108103018168070],
                [0.445948490915965, 0.445948490915965],
                [0.816847572980459, 0.091576213509771],
                [0.091576213509771, 0.816847572980459],
                [0.091576213509771, 0.091576213509771]
            ]),
            'weights': np.array([
                0.223381589678011,
                0.223381589678011,
                0.223381589678011,
                0.109951743655322,
                0.109951743655322,
                0.109951743655322
            ]) * 0.5
        },
        5: {
            'points': np.array([
                [0.333333333333333, 0.333333333333333],
                [0.059715871789770, 0.470142064105115],
                [0.470142064105115, 0.059715871789770],
                [0.470142064105115, 0.470142064105115],
                [0.797426985353087, 0.101286507323456],
                [0.101286507323456, 0.797426985353087],
                [0.101286507323456, 0.101286507323456]
            ]),
            'weights': np.array([
                0.225000000000000,
                0.132394152788506,
                0.132394152788506,
                0.132394152788506,
                0.125939180544827,
                0.125939180544827,
                0.125939180544827
            ]) * 0.5
        }
    }

    def __init__(self, degree=3):
        """
        Parameters
        ----------
        degree : int
            期望的精确多项式次数 (1 <= degree <= 5)
        """
        if degree < 1 or degree > 5:
            raise ValueError(f"TriangleQuadrature: degree={degree} 不在支持范围 [1,5]")
        self.degree = degree
        data = self._rules.get(degree, self._rules[3])
        self.points = data['points'].copy()
        self.weights = data['weights'].copy()
        self.n_points = len(self.weights)

    def integrate(self, func):
        """
        在参考三角形上积分给定的标量/向量函数 func(xi, eta)。
        
        Parameters
        ----------
        func : callable
            func(xi, eta) -> scalar or array
        
        Returns
        -------
        result : float or ndarray
            积分值
        """
        result = 0.0
        for i in range(self.n_points):
            xi_i = self.points[i, 0]
            eta_i = self.points[i, 1]
            w_i = self.weights[i]
            result += w_i * func(xi_i, eta_i)
        return result

    def integrate_array(self, values):
        """
        对已经求值在积分点上的函数值进行加权求和。
        
        Parameters
        ----------
        values : ndarray, shape (n_points, ...)
            函数在积分点处的值
        
        Returns
        -------
        result : ndarray
            积分结果
        """
        if values.shape[0] != self.n_points:
            raise ValueError(
                f"integrate_array: values 第一维长度 {values.shape[0]} "
                f"不等于积分点数 {self.n_points}"
            )
        w = self.weights.reshape(-1, 1) if values.ndim > 1 else self.weights
        return np.sum(w * values, axis=0)


def integrate_over_physical_triangle(quad_rule, nodes, func_physical):
    """
    在物理三角形上积分，通过仿射变换将参考三角形映射到物理三角形。
    
    映射关系:
        x = x1 + (x2 - x1) * xi + (x3 - x1) * eta
        y = y1 + (y2 - y1) * xi + (y3 - y1) * eta
    
    Jacobian 行列式:
        |J| = |(x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)|
    
    Parameters
    ----------
    quad_rule : TriangleQuadrature
        参考三角形上的求积规则
    nodes : ndarray, shape (3, 2)
        物理三角形三个顶点的坐标 [ [x1,y1], [x2,y2], [x3,y3] ]
    func_physical : callable
        func_physical(x, y) -> scalar or array
    
    Returns
    -------
    result : float or ndarray
        物理三角形上的积分值
    """
    x1, y1 = nodes[0]
    x2, y2 = nodes[1]
    x3, y3 = nodes[2]

    J_det = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    if J_det < 1e-14:
        raise ValueError("integrate_over_physical_triangle: 退化三角形，面积为零")

    result = 0.0
    for i in range(quad_rule.n_points):
        xi = quad_rule.points[i, 0]
        eta = quad_rule.points[i, 1]
        x = x1 + (x2 - x1) * xi + (x3 - x1) * eta
        y = y1 + (y2 - y1) * xi + (y3 - y1) * eta
        result += quad_rule.weights[i] * func_physical(x, y)

    return result * J_det


def gauss_legendre_1d(n_points):
    """
    一维 Gauss-Legendre 求积节点与权重。
    
    在区间 [-1, 1] 上，∫ f(x) dx ≈ Σ w_i f(x_i)
    
    Parameters
    ----------
    n_points : int
        点数 (1 <= n_points <= 5)
    
    Returns
    -------
    x : ndarray
        节点
    w : ndarray
        权重
    """
    if n_points == 1:
        x = np.array([0.0])
        w = np.array([2.0])
    elif n_points == 2:
        x = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
        w = np.array([1.0, 1.0])
    elif n_points == 3:
        x = np.array([0.0, -np.sqrt(3.0 / 5.0), np.sqrt(3.0 / 5.0)])
        w = np.array([8.0 / 9.0, 5.0 / 9.0, 5.0 / 9.0])
    elif n_points == 4:
        x = np.array([
            -np.sqrt((3.0 + 2.0 * np.sqrt(6.0 / 5.0)) / 7.0),
            -np.sqrt((3.0 - 2.0 * np.sqrt(6.0 / 5.0)) / 7.0),
            np.sqrt((3.0 - 2.0 * np.sqrt(6.0 / 5.0)) / 7.0),
            np.sqrt((3.0 + 2.0 * np.sqrt(6.0 / 5.0)) / 7.0)
        ])
        w = np.array([
            (18.0 - np.sqrt(30.0)) / 36.0,
            (18.0 + np.sqrt(30.0)) / 36.0,
            (18.0 + np.sqrt(30.0)) / 36.0,
            (18.0 - np.sqrt(30.0)) / 36.0
        ])
    elif n_points == 5:
        x = np.array([
            0.0,
            -np.sqrt(5.0 + 2.0 * np.sqrt(10.0 / 7.0)) / 3.0,
            -np.sqrt(5.0 - 2.0 * np.sqrt(10.0 / 7.0)) / 3.0,
            np.sqrt(5.0 - 2.0 * np.sqrt(10.0 / 7.0)) / 3.0,
            np.sqrt(5.0 + 2.0 * np.sqrt(10.0 / 7.0)) / 3.0
        ])
        w = np.array([
            128.0 / 225.0,
            (322.0 + 13.0 * np.sqrt(70.0)) / 900.0,
            (322.0 - 13.0 * np.sqrt(70.0)) / 900.0,
            (322.0 - 13.0 * np.sqrt(70.0)) / 900.0,
            (322.0 + 13.0 * np.sqrt(70.0)) / 900.0
        ])
    else:
        raise ValueError(f"gauss_legendre_1d: n_points={n_points} 不在 [1,5] 范围")
    return x, w
