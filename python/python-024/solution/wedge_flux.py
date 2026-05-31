r"""
wedge_flux.py
============
楔形区域（Wedge）上的精确积分与高斯求积验证模块。
在太阳耀斑磁重联中，电流片尖角（ Separator 线附近）
形成楔形几何结构，需要在该区域上进行精确的物理量积分。

核心数学模型
------------
楔形区域（3D 单位楔形）的定义:
    W = { (x, y, z) | 0 <= x, 0 <= y, x+y <= 1, -1 <= z <= 1 }

楔形体积:
    V_wedge = 1.0

楔形上的单项式积分:
    I(e1, e2, e3) = integral_W x^{e1} y^{e2} z^{e3} dV

解析公式:
    k = e1
    for i = 1 to e2:
        k = k + 1
        value = value * i / k
    k = k + 1; value = value / k
    k = k + 1; value = value / k

    if e3 为奇数: value = 0
    else: value = value * 2 / (e3 + 1)

在 MHD 中的应用:
    重联电流片尖角区域的磁通量计算:
        Phi = integral_W B . n dS

    能量沉积率:
        Q = integral_W eta J^2 dV

    当楔形被用于局部坐标变换时，需要验证求积规则的
    多项式精确度（degree of exactness）。

高斯求积验证:
    对于 N 阶高斯规则，应能精确积分总次数 <= N 的多项式。
    通过比较数值积分 quad = V * sum w_i f(x_i) 与精确值 exact，
    计算误差 |quad - exact|。

融入原项目:
- 1406_wedge_exactness: 楔形精确积分、多项式精确度验证
"""

import numpy as np
from typing import Tuple, List
from math import comb


class WedgeIntegrals:
    """
    单位楔形区域上的精确积分。
    """

    @staticmethod
    def volume() -> float:
        """单位楔形的体积。"""
        return 1.0

    @staticmethod
    def monomial_integral(exponents: Tuple[int, int, int]) -> float:
        """
        计算单项式 x^{e1} y^{e2} z^{e3} 在楔形上的精确积分。
        """
        e1, e2, e3 = exponents
        if e1 < 0 or e2 < 0:
            raise ValueError("e1, e2 必须非负")
        if e3 == -1:
            raise ValueError("e3 = -1 不合法")
        if e3 % 2 == 1:
            return 0.0

        value = 1.0
        k = e1
        for i in range(1, e2 + 1):
            k += 1
            value *= i / k
        k += 1
        value /= k
        k += 1
        value /= k
        value *= 2.0 / (e3 + 1)
        return value

    @staticmethod
    def gauss_legendre_wedge_7point() -> Tuple[np.ndarray, np.ndarray]:
        """
        7 点高斯求积规则（用于三角形 x 线段）。
        三角形部分: 3 阶 Stroud 规则（7 点中的 3 点在三角形上）
        这里构造一个简化的 7 点规则用于演示。
        """
        # 三角形部分的 3 点高斯规则（2 阶精确）
        x_tri = np.array([1.0/3.0, 1.0/5.0, 3.0/5.0])
        y_tri = np.array([1.0/3.0, 3.0/5.0, 1.0/5.0])
        w_tri = np.array([27.0/60.0, 25.0/60.0, 25.0/60.0]) * 0.5  # 三角形面积为 1/2

        # z 方向 2 点 Gauss-Legendre
        z_pts = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
        w_z = np.array([1.0, 1.0])

        # 张量积构造 6 点规则（三角形3点 x z方向2点）
        points = []
        weights = []
        for i in range(3):
            for j in range(2):
                points.append([x_tri[i], y_tri[i], z_pts[j]])
                weights.append(w_tri[i] * w_z[j])
        return np.array(points), np.array(weights)

    @staticmethod
    def quadrature_rule_exactness(degree_max: int = 5) -> List[dict]:
        """
        验证高斯求积规则的多项式精确度。
        遍历所有总次数 <= degree_max 的单项式，比较数值积分与精确值。
        """
        points, weights = WedgeIntegrals.gauss_legendre_wedge_7point()
        results = []
        for degree in range(degree_max + 1):
            # 生成所有 e1+e2+e3 = degree 的组合
            for e1 in range(degree + 1):
                for e2 in range(degree - e1 + 1):
                    e3 = degree - e1 - e2
                    # 数值积分
                    vals = (points[:, 0] ** e1) * (points[:, 1] ** e2) * (points[:, 2] ** e3)
                    quad = WedgeIntegrals.volume() * np.dot(weights, vals)
                    exact = WedgeIntegrals.monomial_integral((e1, e2, e3))
                    error = abs(quad - exact)
                    results.append({
                        'degree': degree,
                        'exponents': (e1, e2, e3),
                        'quadrature': quad,
                        'exact': exact,
                        'error': error
                    })
        return results

    @staticmethod
    def compute_magnetic_flux(B_func: callable,
                               points: np.ndarray,
                               weights: np.ndarray,
                               normal: np.ndarray = np.array([0.0, 0.0, 1.0])) -> float:
        """
        计算通过楔形表面的磁通量:
            Phi = integral B . n dS
        这里简化为体积分中的轴向分量。
        """
        normal = np.asarray(normal, dtype=float)
        normal = normal / (np.linalg.norm(normal) + 1e-15)
        B_vals = np.array([B_func(p) for p in points])
        dots = B_vals @ normal
        flux = np.dot(weights, dots)
        return flux

    @staticmethod
    def compute_joule_heating(eta: float,
                               J_func: callable,
                               points: np.ndarray,
                               weights: np.ndarray) -> float:
        """
        计算焦耳加热功率:
            Q = integral eta J^2 dV
        """
        J_vals = np.array([J_func(p) for p in points])
        J_sq = np.sum(J_vals ** 2, axis=1)
        Q = eta * np.dot(weights, J_sq)
        return Q


def demo_wedge():
    """
    演示楔形精确积分与求积规则验证。
    """
    print("\n[WedgeFlux] 演示: 楔形精确积分")

    # 1. 基本单项式积分
    test_cases = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 2), (2, 1, 0), (1, 1, 2)]
    for e in test_cases:
        val = WedgeIntegrals.monomial_integral(e)
        print(f"  integral x^{e[0]} y^{e[1]} z^{e[2]} dV = {val:.6f}")

    # 2. 求积规则精确度验证
    print("\n[WedgeFlux] 演示: 求积规则精确度")
    results = WedgeIntegrals.quadrature_rule_exactness(degree_max=4)
    max_err_by_degree = {}
    for r in results:
        d = r['degree']
        max_err_by_degree[d] = max(max_err_by_degree.get(d, 0.0), r['error'])
    for d in sorted(max_err_by_degree.keys()):
        print(f"  总次数 {d}: 最大误差 = {max_err_by_degree[d]:.3e}")

    # 3. 物理量积分
    print("\n[WedgeFlux] 演示: 磁通量与焦耳加热")
    pts, wts = WedgeIntegrals.gauss_legendre_wedge_7point()

    # 假设均匀磁场 B = (0, 0, 1)
    B_uniform = lambda p: np.array([0.0, 0.0, 1.0])
    flux = WedgeIntegrals.compute_magnetic_flux(B_uniform, pts, wts)
    print(f"  均匀 B_z=1 的磁通量: {flux:.6f} (理论值=1.0)")

    # 假设电流密度 J = (0, 0, p[0])，即随 x 线性变化
    J_linear = lambda p: np.array([0.0, 0.0, p[0]])
    Q = WedgeIntegrals.compute_joule_heating(eta=1.0, J_func=J_linear, points=pts, weights=wts)
    # 精确值: integral_W x^2 dV = 1/60
    exact_Q = 1.0 / 60.0
    print(f"  焦耳加热功率 (eta=1, J_z=x): {Q:.6f} (精确值={exact_Q:.6f})")


if __name__ == "__main__":
    demo_wedge()
