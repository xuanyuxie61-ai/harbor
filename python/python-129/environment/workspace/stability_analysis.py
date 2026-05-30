
import numpy as np


class StabilityAnalyzer:

    def __init__(self):
        pass

    @staticmethod
    def trapezoidal_stability_function(z):
        z = np.asarray(z, dtype=complex)
        denom = 1.0 - 0.5 * z

        denom = np.where(np.abs(denom) < 1e-15, 1e-15 + 0j, denom)
        return (1.0 + 0.5 * z) / denom

    @staticmethod
    def boundary_locus(method="trapezoidal", n_points=400):
        theta = np.linspace(0, 2 * np.pi, n_points)
        z_unit = np.exp(1j * theta)

        if method == "trapezoidal":



            z_boundary = 2.0 * (z_unit - 1.0) / (z_unit + 1.0 + 1e-30)
        elif method == "implicit_euler":

            z_boundary = 1.0 - 1.0 / (z_unit + 1e-30)
        elif method == "explicit_euler":

            z_boundary = z_unit - 1.0
        else:
            raise ValueError(f"未知方法: {method}")
        return z_boundary

    @staticmethod
    def is_stable(method, z):
        z = np.asarray(z, dtype=complex)
        if method == "trapezoidal":

            return np.real(z) <= 1e-12
        elif method == "implicit_euler":
            return np.real(z) <= 1e-12
        elif method == "explicit_euler":
            return np.abs(1.0 + z) <= 1.0 + 1e-12
        else:
            raise ValueError(f"未知方法: {method}")

    def analyze_jacobian_stiffness(self, J):



        pass

    def recommend_solver(self, J):
        eigs, stiff_ratio, h_euler, h_trap = self.analyze_jacobian_stiffness(J)
        print("=" * 60)
        print("ODE 系统刚性分析")
        print("=" * 60)
        print(f"特征值实部范围: [{np.min(np.real(eigs)):.4e}, {np.max(np.real(eigs)):.4e}]")
        print(f"刚性比 S = {stiff_ratio:.2e}")
        if stiff_ratio > 1e3:
            print("  -> 系统严重刚性，强烈建议使用隐式方法（梯形法/BDF）")
        elif stiff_ratio > 1e1:
            print("  -> 系统中度刚性，建议使用隐式方法")
        else:
            print("  -> 系统非刚性，显式方法可接受")
        print(f"显式Euler最大步长: {h_euler:.4e} s")
        print(f"梯形法建议步长: {h_trap:.4e} s")
        return stiff_ratio, h_trap


def demo_stability():
    analyzer = StabilityAnalyzer()


    np.random.seed(42)
    J = np.random.randn(12, 12) * 0.1

    J[0, 0] = -500.0
    J[1, 1] = -50.0
    J[4, 4] = -5.0
    J[5, 5] = -0.5
    J[6, 6] = -0.05

    J[4, 2] = 20.0
    J[5, 4] = 2.0

    stiff_ratio, h_trap = analyzer.recommend_solver(J)


    z = -1.0 + 2.0j
    R = analyzer.trapezoidal_stability_function(z)
    print(f"\n梯形法稳定性函数在 z={z}: |R(z)| = {abs(R):.4f}")
    print(f"是否稳定: {analyzer.is_stable('trapezoidal', z)}")

    return analyzer


if __name__ == "__main__":
    demo_stability()
