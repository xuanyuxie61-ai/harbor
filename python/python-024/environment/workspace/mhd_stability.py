
import numpy as np
from typing import Tuple, Optional
from scipy.linalg import eig


class HankelSolver:

    @staticmethod
    def build_hankel(n: int, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if len(x) < 2 * n - 1:
            x = np.append(x, np.zeros(2 * n - 1 - len(x)))
        A = np.zeros((n, n))
        for j in range(n):
            A[:, j] = x[j:j + n]
        return A

    @staticmethod
    def build_toeplitz(n: int, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if len(x) < 2 * n - 1:
            x = np.append(x, np.zeros(2 * n - 1 - len(x)))
        A = np.zeros((n, n))
        for i in range(n):
            A[i, :] = x[n - 1 - i:n - 1 - i + n]
        return A

    @staticmethod
    def inverse(n: int, x: np.ndarray) -> np.ndarray:
        A = HankelSolver.build_hankel(n, x)
        p_rhs = np.append(x[n:2 * n - 1], 0.0)
        u = np.linalg.solve(A, p_rhs)
        q_rhs = np.zeros(n)
        q_rhs[-1] = 1.0
        v = np.linalg.solve(A, q_rhs)


        w1 = np.append(v[1:], np.zeros(1))
        M1 = HankelSolver.build_hankel(n, w1)


        w2 = np.append(np.zeros(n - 1), u)
        M2 = HankelSolver.build_toeplitz(n, w2)


        z3 = np.zeros(n)
        z3[0] = -1.0
        w3 = np.append(u[1:], z3)
        M3 = HankelSolver.build_hankel(n, w3)


        w4 = np.append(np.zeros(n - 1), v)
        M4 = HankelSolver.build_toeplitz(n, w4)

        B = M1 @ M2 - M3 @ M4
        return B


class IntegerRREF:

    @staticmethod
    def _gcd_vec(v: np.ndarray) -> int:
        from math import gcd
        g = 0
        for val in v:
            g = gcd(g, int(abs(val)))
            if g == 1:
                break
        return g

    @staticmethod
    def rref(A_in: np.ndarray) -> Tuple[np.ndarray, int]:
        A = np.copy(A_in).astype(np.int64)
        m, n = A.shape
        lead = 0
        det = 1

        for r in range(m):
            if n <= lead:
                break

            i = r
            while A[i, lead] == 0:
                i += 1
                if m <= i:
                    i = r
                    lead += 1
                    if n <= lead:
                        lead = -1
                        break
            if lead < 0:
                break


            if i != r:
                A[[i, r], :] = A[[r, i], :]
                det = -det


            if A[r, lead] < 0:
                A[r, :] = -A[r, :]
                det = -det

            det *= int(A[r, lead])


            g = IntegerRREF._gcd_vec(A[r, :])
            if g > 1:
                A[r, :] = A[r, :] // g


            for i in range(m):
                if i != r and A[i, lead] != 0:
                    A[i, :] = A[r, lead] * A[i, :] - A[i, lead] * A[r, :]
                    g = IntegerRREF._gcd_vec(A[i, :])
                    if g > 1:
                        A[i, :] = A[i, :] // g

            lead += 1

        return A, det

    @staticmethod
    def verify_divergence_free(integer_incidence: np.ndarray,
                               b_field_dofs: np.ndarray) -> bool:
        C = np.asarray(integer_incidence, dtype=np.int64)
        B = np.asarray(b_field_dofs, dtype=np.int64)
        Cr, _ = IntegerRREF.rref(C)

        residual = Cr @ B
        return np.all(np.abs(residual) < 1e-10)


class MHDStabilityAnalyzer:

    def __init__(self,
                 B0: float = 1.0,
                 lambda_cs: float = 1.0,
                 eta: float = 1e-3,
                 nu: float = 1e-4,
                 kx: float = 0.5,
                 ny: int = 128,
                 y_max: float = 5.0):
        self.B0 = B0
        self.lambda_cs = lambda_cs
        self.eta = eta
        self.nu = nu
        self.kx = kx
        self.ny = ny
        self.y_max = y_max
        self.y = np.linspace(-y_max, y_max, ny)
        self.dy = self.y[1] - self.y[0]

    def _build_laplacian_1d(self) -> np.ndarray:



        raise NotImplementedError("Hole 2: 请实现一维离散拉普拉斯算子 L = d^2/dy^2 - k^2")

    def _equilibrium_B(self) -> np.ndarray:
        return self.B0 * np.tanh(self.y / self.lambda_cs)

    def _equilibrium_Bpp(self) -> np.ndarray:
        t = np.tanh(self.y / self.lambda_cs)
        s2 = 1.0 / np.cosh(self.y / self.lambda_cs) ** 2
        return -2.0 * self.B0 / (self.lambda_cs ** 2) * t * s2

    def tearing_mode_growth_rate(self) -> Tuple[np.ndarray, np.ndarray]:
        L = self._build_laplacian_1d()


        B = self._equilibrium_B()
        Bpp = self._equilibrium_Bpp()


        B_safe = np.where(np.abs(B) < 1e-10, 1e-10, B)
        drive = np.diag(-self.kx ** 2 * Bpp / B_safe)

        M = self.eta * L + drive

        M[0, :] = 0.0
        M[0, 0] = -1.0
        M[-1, :] = 0.0
        M[-1, -1] = -1.0

        eigenvalues, eigenvectors = eig(M)
        return eigenvalues, eigenvectors

    def analyze_stability(self) -> dict:
        gamma, modes = self.tearing_mode_growth_rate()
        real_parts = np.real(gamma)

        mask = np.abs(real_parts + 1.0) > 0.1
        if not np.any(mask):
            mask = np.ones_like(real_parts, dtype=bool)
        gamma_phys = gamma[mask]
        max_growth = np.max(np.real(gamma_phys))
        unstable = max_growth > 0

        return {
            'growth_rates': gamma,
            'max_growth_rate': max_growth,
            'unstable': unstable,
            'n_modes': len(gamma),
            'kx': self.kx,
            'eta': self.eta
        }


def demo_stability():
    print("\n[MHDStability] 演示: Tearing Mode 稳定性分析")


    n = 5

    np.random.seed(0)
    x = np.abs(np.random.randn(2 * n - 1)) + 0.1
    H = HankelSolver.build_hankel(n, x)
    try:
        H_inv = HankelSolver.inverse(n, x)
        I_approx = H @ H_inv
        err_inv = np.max(np.abs(I_approx - np.eye(n)))
        print(f"  Hankel 逆矩阵误差: {err_inv:.3e}")
    except np.linalg.LinAlgError:
        print("  Hankel 矩阵奇异，跳过逆矩阵测试")


    A_test = np.array([[1, 3, 0, 2, 6, 3, 1],
                       [-2, -6, 0, -2, -8, 3, 1],
                       [3, 9, 0, 0, 6, 6, 2],
                       [-1, -3, 0, 1, 0, 9, 3]], dtype=np.int64)
    Ar, det = IntegerRREF.rref(A_test)
    print(f"  IRREF 伪行列式: {det}")
    print(f"  IRREF 结果前三行主元列: {np.argmax(np.abs(Ar[0])), np.argmax(np.abs(Ar[1])), np.argmax(np.abs(Ar[2]))}")


    analyzer = MHDStabilityAnalyzer(kx=0.5, eta=1e-3, ny=64)
    result = analyzer.analyze_stability()
    print(f"  波数 kx={result['kx']}, 电阻率 eta={result['eta']:.0e}")
    print(f"  最大增长率: {result['max_growth_rate']:.4f}")
    print(f"  稳定性: {'不稳定' if result['unstable'] else '稳定'}")


if __name__ == "__main__":
    demo_stability()
