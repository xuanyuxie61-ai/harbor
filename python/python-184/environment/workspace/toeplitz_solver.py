
import numpy as np


class ToeplitzSolver:

    def __init__(self, eps: float = 1e-14):
        self.eps = eps

    def _check_toeplitz(self, t: np.ndarray) -> None:
        if t.ndim != 1:
            raise ValueError("Toeplitz first column must be 1-D array.")
        if len(t) < 1:
            raise ValueError("Toeplitz first column must have length >= 1.")
        if t[0] <= self.eps:
            raise ValueError("Leading diagonal element t_0 must be positive.")

    def solve_yule_walker(self, autocorr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._check_toeplitz(autocorr)
        p = len(autocorr) - 1
        if p == 0:
            return np.array([]), np.array([])

        a = np.zeros(p)
        k = np.zeros(p)
        E = autocorr[0]

        for n in range(1, p + 1):










            raise NotImplementedError("Hole_1: 请实现 Levinson-Durbin 递推核心")

        return a, k

    def solve_toeplitz(self, t: np.ndarray, b: np.ndarray) -> np.ndarray:
        self._check_toeplitz(t)
        n = len(t)
        if b.shape != (n,):
            raise ValueError("b must have same length as t.")

        x = np.zeros(n)
        if n == 1:
            return np.array([b[0] / t[0]])


        x[0] = b[0] / t[0]
        E = t[0]


        y = np.zeros(n - 1)
        y[0] = -t[1] / t[0]

        for m in range(1, n):

            delta = np.dot(t[1:m + 1][::-1], x[:m]) if m > 0 else 0.0
            if m == 0:
                delta = 0.0
            alpha = (b[m] - delta) / E if abs(E) > self.eps else 0.0


            x_prev = x[:m].copy()
            x[:m] = x_prev + alpha * y[:m]
            x[m] = alpha

            if m == n - 1:
                break


            gamma = np.dot(t[1:m + 1][::-1], y[:m]) if m > 0 else 0.0
            beta = -(t[m + 1] + gamma) / E if abs(E) > self.eps else 0.0
            y_prev = y[:m].copy()
            y[:m] = y_prev + beta * y_prev[::-1]
            y[m] = beta


            E = E * (1.0 - beta ** 2)
            if E < self.eps:
                E = self.eps

        return x

    def schur_cohn_stability(self, reflection_coefs: np.ndarray) -> bool:
        if len(reflection_coefs) == 0:
            return True
        return bool(np.all(np.abs(reflection_coefs) < 1.0))

    def autocorr_to_toeplitz(self, autocorr: np.ndarray) -> np.ndarray:
        p = len(autocorr)
        i = np.arange(p)
        j = np.arange(p)
        return autocorr[np.abs(i[:, None] - j)]
