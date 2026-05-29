"""
quadrature_integrals.py
基于 527_hexagon_integrals 的多边形矩计算与 1313_triangle_quadrature_symmetry
的重心坐标求积规则，构建血凝级联反应网络中高维积分与 clot 几何性质计算工具。

科学背景：
    1. 纤维蛋白 clot 的微观结构可建模为多孔介质，其渗透率需要通过
       多边形/多面体上的积分计算。
    2. 反应速率常数依赖于温度，需要通过高斯-勒让德求积计算
       Arrhenius 积分：
           k(T) = A ∫_0^∞ exp(-Ea/(RT)) g(Ea) dEa

数学公式：
    1. 六边形区域上的矩：
       M_{p,q} = ∫_hex x^p y^q dx dy
       对于正六边形，奇数次矩为 0，偶数次矩可解析计算。

    2. 三角形上的重心坐标求积：
       ∫_T f(x,y) dx dy = |T| Σ_i w_i f(λ_1^{(i)}, λ_2^{(i)}, λ_3^{(i)})
       其中 λ_j 为重心坐标。

    3. 纤维蛋白网络孔隙率：
       φ = 1 - (V_fiber / V_total)
       V_fiber 通过六边形单元积分估算。
"""

import numpy as np


class HexagonQuadrature:
    """
    基于 527_hexagon_integrals 的正六边形数值积分。
    用于计算 clot 微观结构单元的几何性质。
    """

    def __init__(self, radius=1.0):
        """
        参数:
            radius : float, 正六边形外接圆半径
        """
        if radius <= 0:
            raise ValueError("radius 必须为正")
        self.R = radius
        # 正六边形顶点
        angles = np.linspace(0, 2 * np.pi, 7)[:-1]
        self.vertices_x = radius * np.cos(angles)
        self.vertices_y = radius * np.sin(angles)
        self.area = 3.0 * np.sqrt(3.0) / 2.0 * radius ** 2

    def monomial_integral(self, p, q):
        """
        计算 ∫_hex x^p y^q dx dy。
        解析公式：对于正六边形，若 p 或 q 为奇数，则积分为 0；
        否则可用矩方法递推计算。
        """
        if (p % 2 == 1) or (q % 2 == 1):
            return 0.0
        # 使用高斯-勒让德近似（7点规则）
        return self._gauss_hex_integral(p, q)

    def _gauss_hex_integral(self, p, q):
        """
        将六边形三角剖分为6个三角形，在每个三角形上使用3点高斯求积。
        """
        cx, cy = 0.0, 0.0
        total = 0.0
        for i in range(6):
            x1, y1 = cx, cy
            x2, y2 = self.vertices_x[i], self.vertices_y[i]
            x3, y3 = self.vertices_x[(i + 1) % 6], self.vertices_y[(i + 1) % 6]
            total += self._triangle_gauss3(x1, y1, x2, y2, x3, y3,
                                            lambda x, y: (x ** p) * (y ** q))
        return total

    @staticmethod
    def _triangle_gauss3(x1, y1, x2, y2, x3, y3, func):
        """
        3点高斯求积公式（精度2）：
            ∫_T f dA = |T| * (1/3) Σ_{i=1}^3 f(重心坐标 1/3 处)
        """
        area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        # 三个边中点
        pts = [
            ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
            ((x2 + x3) / 2.0, (y2 + y3) / 2.0),
            ((x3 + x1) / 2.0, (y3 + y1) / 2.0),
        ]
        s = sum(func(x, y) for x, y in pts) / 3.0
        return area * s

    def compute_clot_fiber_volume(self, fiber_radius_frac=0.15):
        """
        估算 clot 单元中纤维蛋白占据的体积分数。
        假设纤维以六边形顶点为中心呈圆柱状分布。

        体积分数:
            φ_fiber = 6 * (π r_fiber²) / Area_hex
        """
        r_fiber = self.R * fiber_radius_frac
        fiber_area = 6.0 * np.pi * r_fiber ** 2
        # 限制不超过总面积
        fiber_area = min(fiber_area, self.area * 0.95)
        porosity = 1.0 - fiber_area / self.area
        return porosity


class TriangleBarycentricQuadrature:
    """
    基于 1313_triangle_quadrature_symmetry 的重心坐标求积。
    用于 clot 表面三角形网格上的积分。
    """

    @staticmethod
    def xy_to_barycentric(xy):
        """
        将笛卡尔坐标 (x,y) 转换为重心坐标 (λ1, λ2, λ3)。
        对于参考三角形 (0,0), (1,0), (0,1)：
            λ1 = x, λ2 = y, λ3 = 1 - x - y
        """
        xy = np.asarray(xy, dtype=float)
        if xy.ndim == 1:
            xy = xy.reshape(1, -1)
        if xy.shape[1] != 2:
            raise ValueError("输入必须为 N×2 数组")
        n = xy.shape[0]
        bary = np.zeros((n, 3))
        bary[:, 0] = xy[:, 0]
        bary[:, 1] = xy[:, 1]
        bary[:, 2] = 1.0 - xy[:, 0] - xy[:, 1]
        return bary

    @staticmethod
    def integrate_on_triangle(func, vertices, order=3):
        """
        在任意三角形上使用对称求积规则积分。

        参数:
            func     : callable, 函数 f(x,y)
            vertices : ndarray, shape (3,2), 三角形顶点
            order    : int, 求积阶数 (1, 2, 3)

        返回:
            integral : float
        """
        v = np.asarray(vertices, dtype=float)
        if v.shape != (3, 2):
            raise ValueError("vertices 必须为 3×2 数组")

        # 计算面积
        area = 0.5 * abs(np.cross(v[1] - v[0], v[2] - v[0]))

        if order == 1:
            # 1点规则（重心）
            w = [1.0]
            pts_bary = np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]])
        elif order == 2:
            # 3点规则（边中点）
            w = [1.0 / 3.0] * 3
            pts_bary = np.array([
                [0.5, 0.5, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5]
            ])
        elif order == 3:
            # 4点规则（重心 + 三个对称点）
            w = [-27.0 / 48.0, 25.0 / 48.0, 25.0 / 48.0, 25.0 / 48.0]
            pts_bary = np.array([
                [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                [0.6, 0.2, 0.2],
                [0.2, 0.6, 0.2],
                [0.2, 0.2, 0.6]
            ])
        else:
            raise ValueError("order 必须为 1, 2, 3")

        total = 0.0
        for wi, bary in zip(w, pts_bary):
            x = bary[0] * v[0, 0] + bary[1] * v[1, 0] + bary[2] * v[2, 0]
            y = bary[0] * v[0, 1] + bary[1] * v[1, 1] + bary[2] * v[2, 1]
            total += wi * func(x, y)
        return area * total


def arrhenius_rate_integral(T, A_pre, Ea_mean, Ea_sigma, n_points=32):
    """
    使用高斯-埃尔米特求积计算考虑活化能分布的 Arrhenius 速率：
        k_eff(T) = A ∫ exp(-Ea/(RT)) * N(Ea_mean, Ea_sigma²) dEa
                 = A * exp(-Ea_mean/(RT)) * exp( (Ea_sigma/(RT))² / 2 )

    数值验证：与解析解比较。
    """
    if T <= 0:
        raise ValueError("温度 T 必须为正")
    R_gas = 8.314  # J/(mol·K)

    # 解析解 (对数正态期望)
    k_analytic = A_pre * np.exp(-Ea_mean / (R_gas * T)) * \
                 np.exp((Ea_sigma / (R_gas * T)) ** 2 / 2.0)

    # 高斯-埃尔米特数值积分
    # 被积函数: exp(-Ea/(RT)) * 1/(sqrt(2π)σ) exp(-(Ea-μ)²/(2σ²))
    # 令 Ea = μ + √2 σ x, 则 dEa = √2 σ dx
    # 积分变为: 1/√π ∫ exp(-(μ+√2 σ x)/(RT)) exp(-x²) dx
    from numpy.polynomial.hermite import hermgauss
    x, w = hermgauss(n_points)
    Ea_vals = Ea_mean + np.sqrt(2.0) * Ea_sigma * x
    integrands = np.exp(-Ea_vals / (R_gas * T))
    integral = np.sum(w * integrands) / np.sqrt(np.pi)
    k_numerical = A_pre * integral

    return k_analytic, k_numerical


if __name__ == "__main__":
    hex_q = HexagonQuadrature(radius=2.0)
    print(f"六边形面积 (解析): {hex_q.area:.6f}")
    print(f"六边形面积 (数值积分 x⁰y⁰): {hex_q.monomial_integral(0, 0):.6f}")
    print(f"孔隙率估算: {hex_q.compute_clot_fiber_volume():.4f}")

    tri_q = TriangleBarycentricQuadrature()
    verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    val = tri_q.integrate_on_triangle(lambda x, y: x * y, verts, order=3)
    print(f"参考三角形 ∫xy dxdy = {val:.6f} (精确值 = 1/24 = 0.041667)")

    k_a, k_n = arrhenius_rate_integral(T=310.0, A_pre=1e12, Ea_mean=5e4, Ea_sigma=3e3)
    print(f"Arrhenius 速率: 解析 = {k_a:.6e}, 数值 = {k_n:.6e}")
