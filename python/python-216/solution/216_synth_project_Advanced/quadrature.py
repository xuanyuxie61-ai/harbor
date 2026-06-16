# -*- coding: utf-8 -*-
"""
quadrature.py
=============

三角形与一维高斯求积公式, 用于有限元 Helmholtz 装配与 SAA 目标积分.

融合的种子项目:
  - 1325_triangle_witherden_rule -> Witherden-Vincent 高精度对称求积
  - 1323_triangle_twb_rule       -> Taylor-Wingate-Bos (TWB) 求积

这些求积公式在 SAA 中的作用:
  1. 计算目标泛函 J = 0.5 * int |u|^2 dx  的数值积分
  2. 有限元质量 / 刚度矩阵的单元积分
  3. 为随机场在三角形单元上的平均值提供高精度近似
"""

import math
from typing import List, Tuple, Callable


class TriangleQuadrature:
    """三角形求积公式的统一接口.

    标准参考三角形: 顶点 (0,0), (1,0), (0,1).
    所有求积公式都返回 (x, y, w) 三元组, 满足:
        int_T f(x,y) dA ~= sum_i w_i f(x_i, y_i)
    权重之和应等于三角形面积 = 0.5.
    """

    @staticmethod
    def witherden_rule(strength: int
                       ) -> Tuple[List[float], List[float], List[float]]:
        """Witherden-Vincent 对称求积 (复刻 seed project 1325).

        参考: Witherden & Vincent, "On the identification of symmetric
        quadrature rules for finite element methods",
        Computers & Mathematics with Applications, 69 (2015), 1232-1241.

        Args:
            strength: 代数精度, 取 1, 2, 3, 5, 7 等
        """
        if strength <= 0:
            # 精度 0: 重心 (1/3, 1/3), 权 = 0.5
            return [1.0 / 3], [1.0 / 3], [0.5]

        if strength == 1:
            return [1.0 / 3], [1.0 / 3], [0.5]

        if strength == 2:
            # 3 点规则, 精度 2
            x = [1.0 / 6, 2.0 / 3, 1.0 / 6]
            y = [1.0 / 6, 1.0 / 6, 2.0 / 3]
            w = [1.0 / 6, 1.0 / 6, 1.0 / 6]
            return x, y, w

        if strength == 3 or strength == 4:
            # 4 点规则, 精度 3
            x = [1.0 / 3, 0.6, 0.2, 0.2]
            y = [1.0 / 3, 0.2, 0.6, 0.2]
            w = [-27.0 / 96, 25.0 / 96, 25.0 / 96, 25.0 / 96]
            return x, y, w

        if strength == 5:
            # 6 点规则, 精度 5 (Witherden-Vincent)
            a1 = 0.0915762135098
            a2 = 0.8168475729805
            b1 = 0.1081030181681
            b2 = 0.4459484909160
            w1 = 0.0549758718227
            w2 = 0.1116907948390
            x = [a1, a2, a1, b1, b2, b2]
            y = [a1, a1, a2, b1, b2, b1]
            w = [w1, w1, w1, w2, w2, w2]
            return x, y, w

        if strength >= 7:
            # 7 点规则, 精度 7 (近似)
            x = [1.0 / 3,
                 0.0597158717898, 0.4701420641051, 0.4701420641051,
                 0.7974269853531, 0.1012865073235, 0.1012865073235]
            y = [1.0 / 3,
                 0.0597158717898, 0.4701420641051, 0.4701420641051,
                 0.1012865073235, 0.7974269853531, 0.1012865073235]
            w = [0.1125,
                 0.0629695902724, 0.0629695902724, 0.0629695902724,
                 0.0661970763943, 0.0661970763943, 0.0661970763943]
            return x, y, w

        # fallback
        return [1.0 / 3], [1.0 / 3], [0.5]

    @staticmethod
    def twb_rule(strength: int
                 ) -> Tuple[List[float], List[float], List[float]]:
        """Taylor-Wingate-Bos 求积 (复刻 seed project 1323).

        参考: Taylor, Wingate & Bos, "Several new quadrature formulas
        for polynomial integration in the triangle", arXiv:math/0501496.

        TWB 的特点是在高精度下使用较少的点数.
        """
        if strength <= 1:
            return [1.0 / 3], [1.0 / 3], [0.5]

        if strength <= 3:
            # 4 点
            x = [1.0 / 3, 0.2, 0.6, 0.2]
            y = [1.0 / 3, 0.6, 0.2, 0.2]
            w = [-0.28125, 0.2604166666667,
                 0.2604166666667, 0.2604166666667]
            return x, y, w

        if strength <= 5:
            # 7 点 TWB
            a = 0.0597158717897705
            b = 0.797426985353087
            c = 0.470142064105115
            d = 0.101286507323456
            w1 = 0.225000000000000
            w2 = 0.132394152788506
            w3 = 0.125939180544827
            x = [1.0 / 3, a, a, b, c, d, c]
            y = [1.0 / 3, a, b, a, c, c, d]
            w = [w1 * 0.5, w2 * 0.5, w2 * 0.5, w2 * 0.5,
                 w3 * 0.5, w3 * 0.5, w3 * 0.5]
            return x, y, w

        # 默认 3 点
        return TriangleQuadrature.witherden_rule(2)


class GaussLegendre1D:
    """一维 Gauss-Legendre 求积, 用于一维积分与张量积构造."""

    @staticmethod
    def rule(n: int) -> Tuple[List[float], List[float]]:
        """n 点 Gauss-Legendre 在 [-1, 1] 上.
        对 n <= 5 用显式数据, 其余用渐近公式.
        """
        if n <= 1:
            return [0.0], [2.0]
        if n == 2:
            a = 1.0 / math.sqrt(3.0)
            return [-a, a], [1.0, 1.0]
        if n == 3:
            a = math.sqrt(3.0 / 5.0)
            return [-a, 0.0, a], [5.0 / 9, 8.0 / 9, 5.0 / 9]
        if n == 4:
            a1 = math.sqrt(3.0 / 7 - 2.0 / 7 * math.sqrt(6.0 / 5))
            a2 = math.sqrt(3.0 / 7 + 2.0 / 7 * math.sqrt(6.0 / 5))
            w1 = (18.0 + math.sqrt(30.0)) / 36.0
            w2 = (18.0 - math.sqrt(30.0)) / 36.0
            return [-a2, -a1, a1, a2], [w2, w1, w1, w2]
        if n == 5:
            a1 = math.sqrt(5.0 - 2 * math.sqrt(10.0 / 7.0)) / 3.0
            a2 = math.sqrt(5.0 + 2 * math.sqrt(10.0 / 7.0)) / 3.0
            w0 = 128.0 / 225.0
            w1 = (322.0 + 13.0 * math.sqrt(70.0)) / 900.0
            w2 = (322.0 - 13.0 * math.sqrt(70.0)) / 900.0
            return [-a2, -a1, 0.0, a1, a2], [w2, w1, w0, w1, w2]
        # n >= 6: 用 Newton 迭代求根 (简化版本)
        nodes = []
        weights = []
        for i in range(n):
            # 初始猜测 (Chebyshev 近似)
            x = math.cos(math.pi * (i + 0.75) / (n + 0.5))
            for _ in range(20):
                p0, p1 = 1.0, x
                for k in range(2, n + 1):
                    p2 = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
                    p0, p1 = p1, p2
                # p1 = P_n(x), 导数:
                dp = n * (x * p1 - p0) / (x * x - 1.0 + 1e-15)
                dx = -p1 / (dp + 1e-15)
                x += dx
                if abs(dx) < 1e-14:
                    break
            nodes.append(x)
            p0, p1 = 1.0, x
            for k in range(2, n + 1):
                p2 = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
                p0, p1 = p1, p2
            dp = n * (x * p1 - p0) / (x * x - 1.0 + 1e-15)
            weights.append(2.0 / ((1 - x * x) * dp * dp + 1e-15))
        return nodes, weights


def integrate_on_triangle(f: Callable, strength: int = 5,
                          rule_type: str = "witherden") -> float:
    """在参考三角形上积分函数 f(x, y).

    Args:
        f: 被积函数, 签名 f(x, y) -> float
        strength: 求积精度
        rule_type: "witherden" 或 "twb"

    Returns:
        积分近似值
    """
    if rule_type == "twb":
        x, y, w = TriangleQuadrature.twb_rule(strength)
    else:
        x, y, w = TriangleQuadrature.witherden_rule(strength)
    total = 0.0
    for xi, yi, wi in zip(x, y, w):
        total += wi * f(xi, yi)
    return total


def integrate_on_rectangle(f: Callable, a: float, b: float,
                           c: float, d: float,
                           n_x: int = 5, n_y: int = 5) -> float:
    """在矩形 [a,b] x [c,d] 上用张量积 Gauss-Legendre 积分.
    用于 SAA 目标的积分: int_Omega |u(x)|^2 dx.
    """
    nx_q, wx = GaussLegendre1D.rule(n_x)
    ny_q, wy = GaussLegendre1D.rule(n_y)
    # 映射到 [a, b] x [c, d]
    hx = (b - a) / 2.0
    hy = (d - c) / 2.0
    cx = (a + b) / 2.0
    cy = (c + d) / 2.0
    total = 0.0
    for i in range(n_x):
        for j in range(n_y):
            x_val = cx + hx * nx_q[i]
            y_val = cy + hy * ny_q[j]
            total += wx[i] * wy[j] * f(x_val, y_val)
    return total * hx * hy
