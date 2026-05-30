
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


class ConditionEstimator:

    @staticmethod
    def matrix_1norm(A: np.ndarray) -> float:
        return np.max(np.sum(np.abs(A), axis=0))

    @classmethod
    def hager_estimate(cls, A: np.ndarray, max_iter: int = 5, tol: float = 1.0e-8) -> float:
        n = A.shape[0]
        if A.shape[0] != A.shape[1]:
            raise ValueError("矩阵必须为方阵")

        anorm = cls.matrix_1norm(A)
        if anorm < 1.0e-30:
            return np.inf


        b = np.ones(n) / n
        c1 = 0.0
        i1 = -1

        for _ in range(max_iter):
            try:
                x = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                return np.inf

            c2 = np.sum(np.abs(x))
            b = np.sign(x)

            b[b == 0.0] = 1.0

            try:
                y = np.linalg.solve(A.T, b)
            except np.linalg.LinAlgError:
                return np.inf

            i2 = np.argmax(np.abs(y))

            if i1 >= 0:
                if i1 == i2 or c2 <= c1 * (1.0 + tol):
                    break

            i1 = i2
            c1 = c2
            b = np.zeros(n)
            b[i1] = 1.0


        if c2 < 1.0e-30:
            c2 = 1.0e-30

        return float(c2 * anorm)

    @classmethod
    def linpack_estimate(cls, A: np.ndarray) -> float:
        n = A.shape[0]
        anorm = cls.matrix_1norm(A)

        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            return np.inf

        ainv_norm = cls.matrix_1norm(A_inv)
        if ainv_norm < 1.0e-30:
            ainv_norm = 1.0e-30

        return float(anorm * ainv_norm)

    @classmethod
    def sparse_hager_estimate(cls, K: csr_matrix, max_iter: int = 5) -> float:
        n = K.shape[0]
        anorm = np.max(np.sum(np.abs(K.toarray()), axis=0))
        if anorm < 1.0e-30:
            return np.inf

        b = np.ones(n) / n
        c1 = 0.0
        i1 = -1

        for _ in range(max_iter):
            x = spsolve(K, b)
            if x is None:
                return np.inf
            c2 = np.sum(np.abs(x))
            b = np.sign(x)
            b[b == 0.0] = 1.0

            y = spsolve(K.T, b)
            if y is None:
                return np.inf
            i2 = np.argmax(np.abs(y))

            if i1 >= 0:
                if i1 == i2 or c2 <= c1:
                    break

            i1 = i2
            c1 = c2
            b = np.zeros(n)
            b[i1] = 1.0

        return float(c2 * anorm)


class FEMErrorEstimator:

    @staticmethod
    def max_element_size(nodes: np.ndarray, elements: np.ndarray) -> float:
        h_max = 0.0
        for e in range(elements.shape[0]):
            v = elements[e]
            for i in range(3):
                for j in range(i + 1, 3):
                    h = np.linalg.norm(nodes[v[i]] - nodes[v[j]])
                    h_max = max(h_max, h)
        return h_max

    @staticmethod
    def h1_error_estimate(h_max: float, u_h2_seminorm: float, C_interp: float = 1.0) -> float:
        return C_interp * h_max * u_h2_seminorm

    @staticmethod
    def peclet_number(velocity: float, h_max: float, diffusivity: float) -> float:
        if diffusivity <= 0.0:
            return np.inf
        return abs(velocity) * h_max / (2.0 * diffusivity)

    @staticmethod
    def check_stiffness_positive_definite(K_dense: np.ndarray) -> dict:
        if not np.allclose(K_dense, K_dense.T, atol=1.0e-10):
            return {
                "is_spd": False,
                "min_eig": np.nan,
                "max_eig": np.nan,
                "cond_2": np.inf,
                "symmetry_error": float(np.max(np.abs(K_dense - K_dense.T))),
            }

        eigvals = np.linalg.eigvalsh(K_dense)
        min_eig = np.min(eigvals)
        max_eig = np.max(eigvals)
        cond_2 = max_eig / (abs(min_eig) + 1.0e-30)

        return {
            "is_spd": min_eig > 0.0,
            "min_eig": float(min_eig),
            "max_eig": float(max_eig),
            "cond_2": float(cond_2),
        }


class NumericalRobustness:

    @staticmethod
    def safe_divide(a: np.ndarray, b: np.ndarray, eps: float = 1.0e-14) -> np.ndarray:
        b_safe = np.where(np.abs(b) < eps, eps * np.sign(b + eps), b)
        return a / b_safe

    @staticmethod
    def clip_gradient(grad: np.ndarray, max_norm: float = 1.0e6) -> np.ndarray:
        norm = np.linalg.norm(grad)
        if norm > max_norm:
            return grad * (max_norm / norm)
        return grad

    @staticmethod
    def check_convergence(residual_norm: float, tol: float, iter_idx: int, max_iter: int) -> bool:
        if residual_norm < tol:
            return True
        if iter_idx >= max_iter:
            return False
        if not np.isfinite(residual_norm):
            raise RuntimeError(f"迭代 {iter_idx}: 残差出现非有限值")
        return False

    @staticmethod
    def regularize_singular_matrix(A: np.ndarray, reg: float = 1.0e-12) -> np.ndarray:
        A_reg = A.copy()
        diag_idx = np.diag_indices_from(A_reg)
        A_reg[diag_idx] += reg
        return A_reg
