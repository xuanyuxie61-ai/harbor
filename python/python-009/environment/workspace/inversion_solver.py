
import numpy as np
from typing import Callable, Tuple, Optional, List


class BisectionRootFinder:

    def __init__(self, a: float, b: float, tol: float = 1e-10, max_iter: int = 100):
        if a >= b:
            raise ValueError(f"区间左端点必须小于右端点: a={a}, b={b}")
        self.a = a
        self.b = b
        self.tol = tol
        self.max_iter = max_iter
        self.iteration = 0
        self.state = 0
        self.fa = None
        self.fb = None
        self.x_mid = None

    def step(self, fx: Optional[float] = None) -> Tuple[float, bool]:
        if self.state == 0:
            self.x_mid = self.a
            self.state = 1
            return self.x_mid, False

        if self.state == 1:
            self.fa = fx
            self.x_mid = self.b
            self.state = 2
            return self.x_mid, False

        if self.state == 2:
            self.fb = fx
            if self.fa * self.fb > 0:
                raise ValueError(f"f(a)={self.fa} 和 f(b)={self.fb} 同号，无法二分")
            self.x_mid = 0.5 * (self.a + self.b)
            self.state = 3
            self.iteration = 1
            return self.x_mid, False


        if self.iteration >= self.max_iter:
            return self.x_mid, True

        if fx * self.fa > 0:
            self.a = self.x_mid
            self.fa = fx
        else:
            self.b = self.x_mid
            self.fb = fx

        if abs(self.b - self.a) < self.tol:
            return 0.5 * (self.a + self.b), True

        self.x_mid = 0.5 * (self.a + self.b)
        self.iteration += 1
        return self.x_mid, False

    def solve(self, func: Callable[[float], float]) -> float:
        x, _ = self.step(None)
        while True:
            fx = func(x)
            x, converged = self.step(fx)
            if converged:
                return x


class BroydenSolver:

    def __init__(self, atol: float = 1e-8, rtol: float = 1e-6,
                 maxit: int = 100, maxdim: int = 50):
        self.atol = atol
        self.rtol = rtol
        self.maxit = maxit
        self.maxdim = maxdim

    def solve(self, func: Callable[[np.ndarray], np.ndarray],
              x0: np.ndarray) -> Tuple[np.ndarray, int, float]:
        x = np.asarray(x0, dtype=np.float64).reshape(-1)
        n = x.shape[0]

        fc = func(x)
        fnrm = np.linalg.norm(fc) / max(np.sqrt(n), 1e-15)
        stop_tol = self.atol + self.rtol * fnrm


        J_inv = self._finite_difference_jacobian_inverse(func, x, fc)

        stp = np.zeros((n, self.maxdim), dtype=np.float64)
        stp[:, 0] = -J_inv @ fc
        stp_nrm = np.zeros(self.maxdim, dtype=np.float64)
        stp_nrm[0] = np.dot(stp[:, 0], stp[:, 0])

        nbroy = 0
        itc = 0

        while itc < self.maxit:
            nbroy += 1
            fnrmo = fnrm
            itc += 1

            x = x + stp[:, nbroy - 1]
            fc = func(x)
            fnrm = np.linalg.norm(fc) / max(np.sqrt(n), 1e-15)

            if fnrm <= stop_tol:
                return x, 0, fnrm

            if fnrmo <= fnrm:
                return x, 1, fnrm

            if nbroy < self.maxdim:
                z = -fc.copy()
                if nbroy > 1:
                    for kbr in range(nbroy - 1):
                        z = z + stp[:, kbr + 1] * np.dot(stp[:, kbr], z) / max(stp_nrm[kbr], 1e-30)

                zz = np.dot(stp[:, nbroy - 1], z)
                zz = zz / max(stp_nrm[nbroy - 1], 1e-30)
                stp[:, nbroy] = z / max(1.0 - zz, 1e-15)
                stp_nrm[nbroy] = np.dot(stp[:, nbroy], stp[:, nbroy])
            else:

                J_inv = self._finite_difference_jacobian_inverse(func, x, fc)
                stp[:, 0] = -J_inv @ fc
                stp_nrm[0] = np.dot(stp[:, 0], stp[:, 0])
                nbroy = 0

        return x, 1 if fnrm > stop_tol else 0, fnrm

    def _finite_difference_jacobian_inverse(self, func, x, fx, eps=1e-7):
        n = x.shape[0]
        J = np.zeros((n, n), dtype=np.float64)
        for j in range(n):
            xj = x.copy()
            h = eps * max(abs(xj[j]), 1.0)
            xj[j] += h
            fj = func(xj)
            J[:, j] = (fj - fx) / h

        try:
            J_inv = np.linalg.inv(J)
        except np.linalg.LinAlgError:
            J_inv = np.linalg.pinv(J)
        return J_inv


class LevenbergMarquardt:

    def __init__(self, max_iter: int = 100, tol: float = 1e-8,
                 lambda_init: float = 1e-3, lambda_up: float = 10.0,
                 lambda_down: float = 0.1):
        self.max_iter = max_iter
        self.tol = tol
        self.lambda_val = lambda_init
        self.lambda_up = lambda_up
        self.lambda_down = lambda_down

    def solve(self, residual_func: Callable[[np.ndarray], np.ndarray],
              jacobian_func: Callable[[np.ndarray], np.ndarray],
              x0: np.ndarray) -> Tuple[np.ndarray, int, float]:
        x = np.asarray(x0, dtype=np.float64).reshape(-1)

        for it in range(self.max_iter):
            r = residual_func(x)
            cost = 0.5 * np.dot(r, r)

            J = jacobian_func(x)
            JtJ = J.T @ J
            Jtr = J.T @ r


            damping = self.lambda_val * np.diag(np.diag(JtJ))

            try:
                delta = np.linalg.solve(JtJ + damping, -Jtr)
            except np.linalg.LinAlgError:
                delta = -Jtr / (np.diag(JtJ) + self.lambda_val + 1e-15)

            x_new = x + delta
            r_new = residual_func(x_new)
            cost_new = 0.5 * np.dot(r_new, r_new)

            if cost_new < cost:
                x = x_new
                self.lambda_val *= self.lambda_down
                if np.linalg.norm(delta) < self.tol:
                    return x, it + 1, cost_new
            else:
                self.lambda_val *= self.lambda_up
                if self.lambda_val > 1e15:
                    return x, it + 1, cost

        r = residual_func(x)
        return x, self.max_iter, 0.5 * np.dot(r, r)


class TikhonovRegularization:

    @staticmethod
    def first_order_difference_matrix(n: int) -> np.ndarray:
        L = np.zeros((n - 1, n), dtype=np.float64)
        for i in range(n - 1):
            L[i, i] = -1.0
            L[i, i + 1] = 1.0
        return L

    @staticmethod
    def second_order_difference_matrix(n: int) -> np.ndarray:
        L = np.zeros((n - 2, n), dtype=np.float64)
        for i in range(n - 2):
            L[i, i] = 1.0
            L[i, i + 1] = -2.0
            L[i, i + 2] = 1.0
        return L

    @staticmethod
    def solve_linear_tikhonov(A: np.ndarray, b: np.ndarray,
                               L: np.ndarray, alpha: float) -> np.ndarray:
        AtA = A.T @ A
        LtL = L.T @ L
        rhs = A.T @ b
        M = AtA + alpha * LtL
        try:
            x = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            x = np.linalg.lstsq(M, rhs, rcond=None)[0]
        return x
