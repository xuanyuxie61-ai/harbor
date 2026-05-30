
import numpy as np


class HexagonQuadrature:

    def __init__(self, radius=1.0):
        if radius <= 0:
            raise ValueError("radius 必须为正")
        self.R = radius

        angles = np.linspace(0, 2 * np.pi, 7)[:-1]
        self.vertices_x = radius * np.cos(angles)
        self.vertices_y = radius * np.sin(angles)
        self.area = 3.0 * np.sqrt(3.0) / 2.0 * radius ** 2

    def monomial_integral(self, p, q):
        if (p % 2 == 1) or (q % 2 == 1):
            return 0.0

        return self._gauss_hex_integral(p, q)

    def _gauss_hex_integral(self, p, q):
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
        area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

        pts = [
            ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
            ((x2 + x3) / 2.0, (y2 + y3) / 2.0),
            ((x3 + x1) / 2.0, (y3 + y1) / 2.0),
        ]
        s = sum(func(x, y) for x, y in pts) / 3.0
        return area * s

    def compute_clot_fiber_volume(self, fiber_radius_frac=0.15):
        r_fiber = self.R * fiber_radius_frac
        fiber_area = 6.0 * np.pi * r_fiber ** 2

        fiber_area = min(fiber_area, self.area * 0.95)
        porosity = 1.0 - fiber_area / self.area
        return porosity


class TriangleBarycentricQuadrature:

    @staticmethod
    def xy_to_barycentric(xy):
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
        v = np.asarray(vertices, dtype=float)
        if v.shape != (3, 2):
            raise ValueError("vertices 必须为 3×2 数组")


        area = 0.5 * abs(np.cross(v[1] - v[0], v[2] - v[0]))

        if order == 1:

            w = [1.0]
            pts_bary = np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]])
        elif order == 2:

            w = [1.0 / 3.0] * 3
            pts_bary = np.array([
                [0.5, 0.5, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5]
            ])
        elif order == 3:

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
    if T <= 0:
        raise ValueError("温度 T 必须为正")
    R_gas = 8.314


    k_analytic = A_pre * np.exp(-Ea_mean / (R_gas * T)) * \
                 np.exp((Ea_sigma / (R_gas * T)) ** 2 / 2.0)





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
