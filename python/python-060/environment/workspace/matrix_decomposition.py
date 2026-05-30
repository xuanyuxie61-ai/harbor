
import numpy as np
from typing import Tuple, Optional


class CholeskyDecomposition:

    def __init__(self, eta: float = 1e-12):
        self.eta = eta

    def decompose(self, A: np.ndarray) -> Tuple[np.ndarray, int, int]:
        n = A.shape[0]
        if n <= 0:
            return np.array([]), 0, 1

        if A.shape[0] != A.shape[1]:
            raise ValueError("A 必须是方阵")

        L = np.zeros((n, n))
        nullty = 0
        ifault = 0

        for icol in range(n):

            sum_sq = 0.0
            for irow in range(icol):
                sum_sq += L[icol, irow] ** 2

            diag_val = A[icol, icol] - sum_sq


            if diag_val < -self.eta * abs(A[icol, icol]):
                ifault = 2
                return L, nullty, ifault
            elif abs(diag_val) <= self.eta * abs(A[icol, icol]):
                L[icol, icol] = 0.0
                nullty += 1
            else:
                L[icol, icol] = np.sqrt(max(diag_val, 0.0))


            for jcol in range(icol + 1, n):
                sum_prod = 0.0
                for irow in range(icol):
                    sum_prod += L[jcol, irow] * L[icol, irow]

                if L[icol, icol] > self.eta:
                    L[jcol, icol] = (A[jcol, icol] - sum_prod) / L[icol, icol]
                else:
                    L[jcol, icol] = 0.0

        return L, nullty, ifault

    def solve(self, L: np.ndarray, b: np.ndarray) -> np.ndarray:
        n = L.shape[0]
        if len(b) != n:
            raise ValueError("b 长度与 L 不匹配")


        y = np.zeros(n)
        for i in range(n):
            if abs(L[i, i]) < 1e-30:
                y[i] = 0.0
            else:
                y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]


        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            if abs(L[i, i]) < 1e-30:
                x[i] = 0.0
            else:
                x[i] = (y[i] - np.dot(L[i + 1:, i], x[i + 1:])) / L[i, i]

        return x

    def inverse(self, L: np.ndarray) -> np.ndarray:
        n = L.shape[0]
        A_inv = np.zeros((n, n))

        for j in range(n):
            e_j = np.zeros(n)
            e_j[j] = 1.0
            A_inv[:, j] = self.solve(L, e_j)

        return A_inv

    def log_determinant(self, L: np.ndarray) -> float:
        diag = np.diag(L)
        if np.any(diag <= 0):
            return -np.inf
        return 2.0 * np.sum(np.log(diag))

    def condition_number_estimate(self, L: np.ndarray) -> float:
        diag = np.diag(L)
        diag = diag[diag > 1e-30]
        if len(diag) == 0:
            return np.inf
        return np.max(diag) / np.min(diag)


class CovarianceMatrixHandler:

    def __init__(self):
        self.cholesky = CholeskyDecomposition()

    def build_from_correlation(self, sigmas: np.ndarray,
                                correlation: np.ndarray) -> np.ndarray:
        n = len(sigmas)
        if correlation.shape != (n, n):
            raise ValueError("correlation 矩阵维度不匹配")

        Sigma = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                Sigma[i, j] = correlation[i, j] * sigmas[i] * sigmas[j]


        Sigma = 0.5 * (Sigma + Sigma.T)


        eigvals = np.linalg.eigvalsh(Sigma)
        if np.min(eigvals) < 1e-14:
            Sigma += (1e-12 - np.min(eigvals)) * np.eye(n)

        return Sigma

    def sample_multivariate_normal(self, mu: np.ndarray,
                                    Sigma: np.ndarray,
                                    n_samples: int = 1,
                                    seed: int = 42) -> np.ndarray:
        L, nullty, ifault = self.cholesky.decompose(Sigma)

        if ifault != 0:

            Sigma_reg = Sigma + 1e-10 * np.eye(Sigma.shape[0])
            L, _, _ = self.cholesky.decompose(Sigma_reg)

        rng = np.random.default_rng(seed)
        Z = rng.standard_normal((n_samples, len(mu)))
        X = mu + Z @ L.T
        return X

    def mahalanobis_distance(self, x: np.ndarray, mu: np.ndarray,
                              Sigma: np.ndarray) -> float:
        L, nullty, ifault = self.cholesky.decompose(Sigma)
        if ifault != 0:
            Sigma_reg = Sigma + 1e-10 * np.eye(Sigma.shape[0])
            L, _, _ = self.cholesky.decompose(Sigma_reg)

        diff = x - mu
        y = self.cholesky.solve(L, diff)
        return np.dot(y, y)


class PreconditionerBuilder:

    def __init__(self):
        self.cholesky = CholeskyDecomposition()

    def jacobi_preconditioner(self, A: np.ndarray) -> np.ndarray:
        diag = np.diag(A)
        diag = np.where(np.abs(diag) > 1e-30, diag, 1.0)
        return np.diag(1.0 / diag)

    def incomplete_cholesky(self, A: np.ndarray,
                             fill_level: int = 0) -> np.ndarray:
        n = A.shape[0]
        L = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1):
                if A[i, j] == 0 and fill_level == 0:
                    continue

                if i == j:
                    sum_sq = np.sum(L[i, :j] ** 2)
                    val = A[i, i] - sum_sq
                    if val > 1e-14:
                        L[i, j] = np.sqrt(val)
                else:
                    sum_prod = np.sum(L[i, :j] * L[j, :j])
                    if L[j, j] > 1e-30:
                        L[i, j] = (A[i, j] - sum_prod) / L[j, j]

        return L

    def ssor_preconditioner(self, A: np.ndarray,
                             omega: float = 1.0) -> np.ndarray:
        n = A.shape[0]
        D = np.diag(np.diag(A))
        L_strict = np.tril(A, -1)

        D_inv = np.diag(1.0 / (np.diag(A) + 1e-30))

        M = (D + omega * L_strict) @ D_inv @ (D + omega * L_strict.T)
        M = M / (omega * (2.0 - omega) + 1e-30)

        return M
