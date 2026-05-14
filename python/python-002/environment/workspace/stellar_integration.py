# -*- coding: utf-8 -*-
"""
stellar_integration.py
基于 1319_triangle_symq_to_ref 与 684_line_ncc_rule 合成
恒星内部结构的数值积分引擎：Newton-Cotes 闭型求积与三角形高斯求积。
用于计算光度积分、质量积分、热力学量积分等。
"""

import numpy as np
from typing import Callable, Tuple, Optional


class StellarIntegrator:
    """
    恒星物理量数值积分器。
    支持：
      1) Newton-Cotes Closed (NCC) 等距求积 — 基于 684_line_ncc_rule
      2) 三角形参考域高斯求积 — 基于 1319_triangle_symq_to_ref
      3) 复合 Simpson 自适应积分
    """

    @staticmethod
    def newton_cotes_weights(n: int, a: float = 0.0, b: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算闭型 Newton-Cotes 求积规则的节点 x 与权重 w。
        区间 [a,b] 上 n 个等距节点（2 <= n <= 7）。
        使用解析系数表，避免数值不稳定。
        """
        if n < 2 or n > 7:
            raise ValueError("当前仅支持 2~7 点 Newton-Cotes 规则")
        x = np.linspace(a, b, n)
        h = (b - a) / (n - 1)
        # 解析权重系数 (Cotes 数)
        coeff_map = {
            2: np.array([1.0, 1.0]) / 2.0,
            3: np.array([1.0, 4.0, 1.0]) / 3.0,
            4: np.array([3.0, 9.0, 9.0, 3.0]) / 8.0,
            5: np.array([14.0, 64.0, 24.0, 64.0, 14.0]) / 45.0,
            6: np.array([95.0, 375.0, 250.0, 250.0, 375.0, 95.0]) / 288.0,
            7: np.array([41.0, 216.0, 27.0, 272.0, 27.0, 216.0, 41.0]) / 140.0,
        }
        w = coeff_map[n] * h * (n - 1)
        return x, w

    @staticmethod
    def integrate_ncc(f: Callable[[np.ndarray], np.ndarray], a: float, b: float,
                      n: int = 5) -> float:
        """使用 n 点 Newton-Cotes Closed 规则积分 f 在 [a,b] 上的值。"""
        x, w = StellarIntegrator.newton_cotes_weights(n, a, b)
        fx = f(x)
        return float(np.dot(w, fx))

    @staticmethod
    def triangle_gauss_rule(degree: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        参考三角形 (0,0),(1,0),(0,1) 上的对称高斯求积规则。
        基于 1319_triangle_symq_to_ref 思想，提供标准对称求积点。
        
        重心坐标: (x,y,z) 满足 x+y+z=1, x,y,z>=0
        节点数由精度阶数决定（degree 1~7 的常用规则）。
        """
        # 常用 Dunavant 规则
        if degree <= 1:
            # 1点，精度1
            nodes = np.array([[1.0 / 3.0, 1.0 / 3.0]])
            weights = np.array([0.5])
        elif degree <= 2:
            # 3点，精度2
            nodes = np.array([[2.0 / 3.0, 1.0 / 6.0],
                              [1.0 / 6.0, 2.0 / 3.0],
                              [1.0 / 6.0, 1.0 / 6.0]])
            weights = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        elif degree <= 3:
            # 4点，精度3 (含重心)
            nodes = np.array([[1.0 / 3.0, 1.0 / 3.0],
                              [0.6, 0.2],
                              [0.2, 0.6],
                              [0.2, 0.2]])
            weights = np.array([-27.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0])
        elif degree <= 4:
            # 6点，精度4
            a1 = 0.108103018168070
            b1 = 0.445948490915965
            a2 = 0.816847572980459
            b2 = 0.091576213509771
            nodes = np.array([[a1, b1], [b1, a1], [b1, b1],
                              [a2, b2], [b2, a2], [b2, b2]])
            w1 = 0.223381589678011
            w2 = 0.109951743655322
            weights = np.array([w1, w1, w1, w2, w2, w2])
        elif degree <= 5:
            # 7点，精度5
            a1 = 0.059715871789770
            b1 = 0.470142064105115
            a2 = 0.797426985353087
            b2 = 0.101286507323456
            nodes = np.array([[1.0 / 3.0, 1.0 / 3.0],
                              [a1, b1], [b1, a1], [b1, b1],
                              [a2, b2], [b2, a2], [b2, b2]])
            w1 = 0.225000000000000
            w2 = 0.132394152788506
            w3 = 0.125939180544827
            weights = np.array([w1, w2, w2, w2, w3, w3, w3])
        elif degree <= 6:
            # 12点，精度6
            a1 = 0.501426509658179
            b1 = 0.249286745170910
            a2 = 0.873821971016996
            b2 = 0.063089014491502
            a3 = 0.053145049844817
            b3 = 0.310352451033784
            c3 = 1.0 - a3 - b3
            nodes = np.array([
                [a1, b1], [b1, a1], [b1, b1],
                [a2, b2], [b2, a2], [b2, b2],
                [a3, b3], [b3, a3], [c3, a3],
                [a3, c3], [b3, c3], [c3, b3]
            ])
            w1 = 0.116786275726379
            w2 = 0.050844906370207
            w3 = 0.082851075618374
            weights = np.array([w1, w1, w1, w2, w2, w2, w3, w3, w3, w3, w3, w3])
        else:
            # 13点，精度7
            a1 = 0.479308067841920
            b1 = 0.260345966079040
            a2 = 0.869739794195568
            b2 = 0.065130102902216
            a3 = 0.048690315425316
            b3 = 0.312865496004874
            c3 = 1.0 - a3 - b3
            nodes = np.array([
                [1.0 / 3.0, 1.0 / 3.0],
                [a1, b1], [b1, a1], [b1, b1],
                [a2, b2], [b2, a2], [b2, b2],
                [a3, b3], [b3, a3], [c3, a3],
                [a3, c3], [b3, c3], [c3, b3]
            ])
            w1 = -0.149570044467671
            w2 = 0.175615257433204
            w3 = 0.053347235608838
            w4 = 0.077113760890257
            weights = np.array([w1, w2, w2, w2, w3, w3, w3, w4, w4, w4, w4, w4, w4])
        return nodes, weights * 0.5  # 面积归一化

    @staticmethod
    def integrate_triangle(f: Callable[[np.ndarray, np.ndarray], np.ndarray],
                           v1: Tuple[float, float],
                           v2: Tuple[float, float],
                           v3: Tuple[float, float],
                           degree: int = 5) -> float:
        """
        在由顶点 v1,v2,v3 定义的物理三角形上积分 f(x,y)。
        通过仿射变换映射到参考三角形求积。
        
        雅可比行列式: |J| = 2 * Area(T)
        Area(T) = 0.5 * |x2-x1  x3-x1|
                       |y2-y1  y3-y1|
        """
        v1 = np.array(v1, dtype=np.float64)
        v2 = np.array(v2, dtype=np.float64)
        v3 = np.array(v3, dtype=np.float64)
        # 面积
        area = 0.5 * abs((v2[0] - v1[0]) * (v3[1] - v1[1]) - (v3[0] - v1[0]) * (v2[1] - v1[1]))
        nodes_ref, weights_ref = StellarIntegrator.triangle_gauss_rule(degree)
        # 仿射变换: x = v1 + (v2-v1)*xi + (v3-v1)*eta
        x_phys = v1[0] + (v2[0] - v1[0]) * nodes_ref[:, 0] + (v3[0] - v1[0]) * nodes_ref[:, 1]
        y_phys = v1[1] + (v2[1] - v1[1]) * nodes_ref[:, 0] + (v3[1] - v1[1]) * nodes_ref[:, 1]
        fx = f(x_phys, y_phys)
        return float(np.sum(weights_ref * fx) * 2.0 * area)

    @staticmethod
    def integrate_shell(f: np.ndarray, mass: np.ndarray, method: str = 'simpson') -> float:
        """
        在质量坐标上积分场量 f(m)。
        使用复合 Simpson 或梯形法。
        """
        f = np.asarray(f, dtype=np.float64)
        mass = np.asarray(mass, dtype=np.float64)
        if len(f) != len(mass):
            raise ValueError("f 与 mass 长度必须相同")
        n = len(mass)
        if n < 2:
            return 0.0
        if method == 'trapezoidal':
            return np.trapz(f, mass)
        elif method == 'simpson':
            if n < 3:
                return np.trapz(f, mass)
            # 复合 Simpson (要求等距或近似等距)
            # 对非均匀网格，退化为梯形
            dm = np.diff(mass)
            if np.allclose(dm, dm[0], rtol=0.1):
                h = dm[0]
                result = f[0] + f[-1]
                result += 4.0 * np.sum(f[1:-1:2])
                result += 2.0 * np.sum(f[2:-1:2])
                return result * h / 3.0
            else:
                return np.trapz(f, mass)
        else:
            return np.trapz(f, mass)

    @staticmethod
    def moment_of_inertia(radius: np.ndarray, density: np.ndarray, dm: np.ndarray) -> float:
        """
        计算恒星转动惯量 [g cm^2]。
        I = (8π/3) ∫_0^R ρ(r) r^4 dr
        在质量坐标上: I = (2/3) ∫_0^M r^2 dm
        """
        r = np.asarray(radius, dtype=np.float64)
        dm_arr = np.asarray(dm, dtype=np.float64)
        integrand = (2.0 / 3.0) * r ** 2
        return float(np.trapz(integrand, np.cumsum(dm_arr)))

    @staticmethod
    def gravitational_binding_energy(mass: np.ndarray, radius: np.ndarray) -> float:
        """
        引力结合能：Ω = -∫_0^M G m / r dm
        """
        G = 6.67430e-8  # CGS
        m = np.asarray(mass, dtype=np.float64)
        r = np.asarray(radius, dtype=np.float64)
        r = np.maximum(r, 1e-3)
        integrand = -G * m / r
        return float(np.trapz(integrand, m))
