
import numpy as np


class TriangleQuadrature:

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
        if degree < 1 or degree > 5:
            raise ValueError(f"TriangleQuadrature: degree={degree} 不在支持范围 [1,5]")
        self.degree = degree
        data = self._rules.get(degree, self._rules[3])
        self.points = data['points'].copy()
        self.weights = data['weights'].copy()
        self.n_points = len(self.weights)

    def integrate(self, func):
        result = 0.0
        for i in range(self.n_points):
            xi_i = self.points[i, 0]
            eta_i = self.points[i, 1]
            w_i = self.weights[i]
            result += w_i * func(xi_i, eta_i)
        return result

    def integrate_array(self, values):
        if values.shape[0] != self.n_points:
            raise ValueError(
                f"integrate_array: values 第一维长度 {values.shape[0]} "
                f"不等于积分点数 {self.n_points}"
            )
        w = self.weights.reshape(-1, 1) if values.ndim > 1 else self.weights
        return np.sum(w * values, axis=0)


def integrate_over_physical_triangle(quad_rule, nodes, func_physical):
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
