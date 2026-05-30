
import numpy as np
from typing import Tuple, Optional


def vandermonde_determinant(x: np.ndarray) -> float:
    n = len(x)
    if n == 0:
        return 1.0
    det = 1.0
    for i in range(1, n):
        for j in range(i):
            det *= (x[i] - x[j])
    return det


def chebyshev_grid(n: int) -> np.ndarray:
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return np.array([1.0])
    k = np.arange(n + 1)
    return np.cos(np.pi * k / n)


def chebyshev_differentiation_matrix(n: int) -> np.ndarray:
    if n < 1:
        raise ValueError("n must be at least 1")

    x = chebyshev_grid(n)
    c = np.ones(n + 1)
    c[0] = 2.0
    c[n] = 2.0
    c = c * ((-1.0) ** np.arange(n + 1))

    X = np.tile(x[:, np.newaxis], (1, n + 1))
    dX = X - X.T


    D = (c[:, np.newaxis] / c[np.newaxis, :]) / (dX + np.eye(n + 1))
    D = D - np.diag(np.diag(D))


    D = D - np.diag(D.sum(axis=1))

    return D


def plu_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square")

    L = np.eye(n)
    U = A.copy().astype(np.float64)
    P = np.eye(n)

    for k in range(n - 1):

        pivot_idx = k + np.argmax(np.abs(U[k:, k]))
        if abs(U[pivot_idx, k]) < 1e-15:
            continue


        if pivot_idx != k:
            U[[k, pivot_idx], :] = U[[pivot_idx, k], :]
            P[[k, pivot_idx], :] = P[[pivot_idx, k], :]
            if k > 0:
                L[[k, pivot_idx], :k] = L[[pivot_idx, k], :k]


        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]

    return P, L, U


def hager_condition_number_estimate(A: np.ndarray, max_iter: int = 5) -> float:
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square")

    anorm = np.linalg.norm(A, ord=1)


    b = np.ones(n) / n
    old_index = -1

    for _ in range(max_iter):
        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.inf

        c = np.sum(np.abs(x))
        b = np.sign(x)

        b[x == 0] = 1.0

        try:
            y = np.linalg.solve(A.T, b)
        except np.linalg.LinAlgError:
            return np.inf

        new_index = np.argmax(np.abs(y))
        if new_index == old_index or abs(np.abs(y[new_index]) - c) < 1e-10:
            break
        old_index = new_index
        b = np.zeros(n)
        b[new_index] = 1.0


    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.inf

    c_final = np.sum(np.abs(x))
    cond = c_final * anorm
    return cond


def sample_condition_estimate(A: np.ndarray, n_samples: int = 20) -> float:
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square")

    a_norm = 0.0
    ainv_norm = 0.0

    for _ in range(n_samples):

        x = np.random.randn(n)
        x = x / (np.linalg.norm(x) + 1e-15)

        ax = A @ x
        a_norm = max(a_norm, np.linalg.norm(ax, ord=1))

        try:
            ainv_x = np.linalg.solve(A, x)
            ainv_norm = max(ainv_norm, np.linalg.norm(ainv_x, ord=1))
        except np.linalg.LinAlgError:
            return np.inf

    if ainv_norm < 1e-15:
        return np.inf
    return a_norm * ainv_norm


class QuantumKernelMatrix:

    def __init__(self, kernel_func, data_points: np.ndarray):
        self.data_points = np.array(data_points, dtype=np.float64)
        self.n_samples = self.data_points.shape[0]
        self.kernel_func = kernel_func
        self._K: Optional[np.ndarray] = None
        self._K_inv: Optional[np.ndarray] = None

    def compute_kernel_matrix(self) -> np.ndarray:
        K = np.zeros((self.n_samples, self.n_samples))
        for i in range(self.n_samples):
            for j in range(i, self.n_samples):
                val = self.kernel_func(self.data_points[i], self.data_points[j])

                val = max(0.0, min(1.0, val))
                K[i, j] = val
                K[j, i] = val


        K = (K + K.T) / 2.0


        eigvals = np.linalg.eigvalsh(K)
        if np.min(eigvals) < -1e-10:
            K += (-np.min(eigvals) + 1e-10) * np.eye(self.n_samples)

        self._K = K
        return K

    def condition_number(self) -> float:
        if self._K is None:
            self.compute_kernel_matrix()

        K = self._K
        eigvals = np.linalg.eigvalsh(K)
        pos_eigvals = eigvals[eigvals > 1e-15]
        if len(pos_eigvals) == 0:
            return np.inf
        return np.max(eigvals) / np.min(pos_eigvals)

    def hager_cond_estimate(self) -> float:
        if self._K is None:
            self.compute_kernel_matrix()
        return hager_condition_number_estimate(self._K)

    def solve_kernel_system(self, y: np.ndarray, reg: float = 1e-6) -> np.ndarray:
        if self._K is None:
            self.compute_kernel_matrix()

        K_reg = self._K + reg * np.eye(self.n_samples)
        try:
            alpha = np.linalg.solve(K_reg, y)
        except np.linalg.LinAlgError:

            alpha = np.linalg.lstsq(K_reg, y, rcond=1e-10)[0]

        self._K_inv = alpha
        return alpha

    def kernel_target_alignment(self, y: np.ndarray) -> float:
        if self._K is None:
            self.compute_kernel_matrix()

        y = np.array(y, dtype=np.float64)
        Y = np.outer(y, y)

        k_norm = np.linalg.norm(self._K, "fro")
        y_norm = np.linalg.norm(Y, "fro")
        if k_norm < 1e-15 or y_norm < 1e-15:
            return 0.0

        inner = np.sum(self._K * Y)
        return inner / (k_norm * y_norm)


def quantum_kernel_with_vandermonde(
    x: np.ndarray,
    x_prime: np.ndarray,
    n_qubits: int = 4
) -> float:
    if len(x) != len(x_prime):
        raise ValueError("Input vectors must have same length")

    n = min(len(x), n_qubits)

    v_x = np.array([x[i] ** j for j in range(n) for i in range(n)], dtype=np.float64)
    v_xp = np.array([x_prime[i] ** j for j in range(n) for i in range(n)], dtype=np.float64)


    norm_x = np.linalg.norm(v_x)
    norm_xp = np.linalg.norm(v_xp)
    if norm_x < 1e-15 or norm_xp < 1e-15:
        return 0.0

    overlap = np.dot(v_x, v_xp) / (norm_x * norm_xp)
    return overlap ** 2
