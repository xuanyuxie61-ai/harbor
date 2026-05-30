
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


class SemanticHeatDiffusion:

    def __init__(self, n: int = 41, a: float = 0.0, b: float = 1.0):
        if n < 3:
            raise ValueError(f"n must be at least 3, got {n}")
        if a >= b:
            raise ValueError(f"must have a < b, got a={a}, b={b}")

        self.n = int(n)
        self.a = float(a)
        self.b = float(b)
        self.dx = (b - a) / (n - 1)
        self.x = np.linspace(a, b, n)

    def solve(self, ua: float, ub: float,
              conductivity_func, source_func) -> np.ndarray:






        raise NotImplementedError("Hole 1: 热扩散求解器尚未实现")

    def compute_flux(self, U: np.ndarray, conductivity_func) -> np.ndarray:
        flux = np.zeros(self.n)
        for i in range(1, self.n - 1):
            dUdx = (U[i + 1] - U[i - 1]) / (2.0 * self.dx)
            flux[i] = -conductivity_func(self.x[i]) * dUdx


        flux[0] = -conductivity_func(self.x[0]) * (U[1] - U[0]) / self.dx
        flux[self.n - 1] = -conductivity_func(self.x[self.n - 1]) * (U[self.n - 1] - U[self.n - 2]) / self.dx

        return flux

    def solve_nonuniform_conductivity(self, ua: float, ub: float,
                                       K_values: np.ndarray, F_values: np.ndarray) -> np.ndarray:
        if len(K_values) != self.n or len(F_values) != self.n:
            raise ValueError("K_values and F_values must have length n")

        if np.any(K_values <= 0.0):
            raise ValueError("all conductivity values must be positive")


        data_main = np.zeros(self.n)
        data_lower = np.zeros(self.n - 1)
        data_upper = np.zeros(self.n - 1)
        rhs = np.zeros(self.n)

        data_main[0] = 1.0
        rhs[0] = ua

        for i in range(1, self.n - 1):
            k_left = 0.5 * (K_values[i - 1] + K_values[i])
            k_right = 0.5 * (K_values[i] + K_values[i + 1])

            data_lower[i - 1] = -k_left / (self.dx * self.dx)
            data_main[i] = (k_left + k_right) / (self.dx * self.dx)
            data_upper[i] = -k_right / (self.dx * self.dx)
            rhs[i] = F_values[i]

        data_main[self.n - 1] = 1.0
        rhs[self.n - 1] = ub

        A = csr_matrix(
            (np.concatenate([data_lower, data_main, data_upper]),
             (np.concatenate([np.arange(1, self.n), np.arange(self.n), np.arange(self.n - 1)]),
              np.concatenate([np.arange(self.n - 1), np.arange(self.n), np.arange(1, self.n)]))),
            shape=(self.n, self.n)
        )

        U = spsolve(A, rhs)
        return U


def demo():
    print("=" * 60)
    print("语义信息稳态热扩散演示")
    print("=" * 60)

    diffusion = SemanticHeatDiffusion(n=41, a=0.0, b=1.0)
    print(f"\n网格数: {diffusion.n}")
    print(f"空间步长: {diffusion.dx:.6f}")


    ua = 1.0
    ub = 0.0


    def conductivity(x):
        return 1.0 + 0.5 * np.sin(np.pi * x) ** 2


    def source(x):
        return 2.0 * np.exp(-((x - 0.3) ** 2) / 0.01)

    U = diffusion.solve(ua, ub, conductivity, source)
    flux = diffusion.compute_flux(U, conductivity)

    print(f"\n稳态解范围: [{U.min():.6f}, {U.max():.6f}]")
    print(f"热流范围: [{flux.min():.6f}, {flux.max():.6f}]")
    print(f"边界热流 (左): {flux[0]:.6f}")
    print(f"边界热流 (右): {flux[-1]:.6f}")


    total_source = np.trapezoid([source(xi) for xi in diffusion.x], diffusion.x)
    net_flux = flux[-1] - flux[0]
    print(f"\n质量守恒检查:")
    print(f"  总热源: {total_source:.6f}")
    print(f"  净热流: {net_flux:.6f}")
    print(f"  相对偏差: {abs(total_source - net_flux) / (abs(total_source) + 1e-15):.6e}")

    print("\n模块运行完成")
    return diffusion, U, flux


if __name__ == "__main__":
    demo()
