
import numpy as np
from typing import Tuple, List
from math import comb


class WedgeIntegrals:

    @staticmethod
    def volume() -> float:
        return 1.0

    @staticmethod
    def monomial_integral(exponents: Tuple[int, int, int]) -> float:
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

        x_tri = np.array([1.0/3.0, 1.0/5.0, 3.0/5.0])
        y_tri = np.array([1.0/3.0, 3.0/5.0, 1.0/5.0])
        w_tri = np.array([27.0/60.0, 25.0/60.0, 25.0/60.0]) * 0.5


        z_pts = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
        w_z = np.array([1.0, 1.0])


        points = []
        weights = []
        for i in range(3):
            for j in range(2):
                points.append([x_tri[i], y_tri[i], z_pts[j]])
                weights.append(w_tri[i] * w_z[j])
        return np.array(points), np.array(weights)

    @staticmethod
    def quadrature_rule_exactness(degree_max: int = 5) -> List[dict]:
        points, weights = WedgeIntegrals.gauss_legendre_wedge_7point()
        results = []
        for degree in range(degree_max + 1):

            for e1 in range(degree + 1):
                for e2 in range(degree - e1 + 1):
                    e3 = degree - e1 - e2

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
        J_vals = np.array([J_func(p) for p in points])
        J_sq = np.sum(J_vals ** 2, axis=1)
        Q = eta * np.dot(weights, J_sq)
        return Q


def demo_wedge():
    print("\n[WedgeFlux] 演示: 楔形精确积分")


    test_cases = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 2), (2, 1, 0), (1, 1, 2)]
    for e in test_cases:
        val = WedgeIntegrals.monomial_integral(e)
        print(f"  integral x^{e[0]} y^{e[1]} z^{e[2]} dV = {val:.6f}")


    print("\n[WedgeFlux] 演示: 求积规则精确度")
    results = WedgeIntegrals.quadrature_rule_exactness(degree_max=4)
    max_err_by_degree = {}
    for r in results:
        d = r['degree']
        max_err_by_degree[d] = max(max_err_by_degree.get(d, 0.0), r['error'])
    for d in sorted(max_err_by_degree.keys()):
        print(f"  总次数 {d}: 最大误差 = {max_err_by_degree[d]:.3e}")


    print("\n[WedgeFlux] 演示: 磁通量与焦耳加热")
    pts, wts = WedgeIntegrals.gauss_legendre_wedge_7point()


    B_uniform = lambda p: np.array([0.0, 0.0, 1.0])
    flux = WedgeIntegrals.compute_magnetic_flux(B_uniform, pts, wts)
    print(f"  均匀 B_z=1 的磁通量: {flux:.6f} (理论值=1.0)")


    J_linear = lambda p: np.array([0.0, 0.0, p[0]])
    Q = WedgeIntegrals.compute_joule_heating(eta=1.0, J_func=J_linear, points=pts, weights=wts)

    exact_Q = 1.0 / 60.0
    print(f"  焦耳加热功率 (eta=1, J_z=x): {Q:.6f} (精确值={exact_Q:.6f})")


if __name__ == "__main__":
    demo_wedge()
