
import numpy as np
from typing import Tuple, Optional, Callable


class ConjugateGradientSolver:

    def __init__(
        self,
        max_iter: int = 2000,
        tol: float = 1e-10,
        preconditioner: Optional[str] = "jacobi",
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.preconditioner = preconditioner

    def _apply_preconditioner(self, M_diag: np.ndarray, r: np.ndarray) -> np.ndarray:
        if self.preconditioner == "jacobi":

            diag_safe = np.where(np.abs(M_diag) > 1e-30, M_diag, 1.0)
            return r / diag_safe
        elif self.preconditioner == "identity":
            return r.copy()
        elif callable(self.preconditioner):
            return self.preconditioner(r)
        else:
            return r.copy()

    def solve(
        self,
        A_matvec: Callable[[np.ndarray], np.ndarray],
        b: np.ndarray,
        x0: Optional[np.ndarray] = None,
        M_diag: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, dict]:
        b = np.asarray(b, dtype=float)
        n = b.size

        if x0 is None:
            x = np.zeros(n)
        else:
            x = np.asarray(x0, dtype=float).copy()

        if M_diag is None:
            M_diag = np.ones(n)


        r = b - A_matvec(x)


        z = self._apply_preconditioner(M_diag, r)
        p = z.copy()

        rho = np.dot(r, z)
        b_norm = np.linalg.norm(b)
        if b_norm < 1e-30:
            b_norm = 1.0

        residual_history = []

        for k in range(self.max_iter):

            q = A_matvec(p)
            pdotq = np.dot(p, q)

            if abs(pdotq) < 1e-30:
                info = {
                    "iter": k,
                    "residual": np.linalg.norm(r),
                    "rel_residual": np.linalg.norm(r) / b_norm,
                    "converged": False,
                    "reason": "pdotq too small",
                }
                return x, info

            alpha = rho / pdotq
            x += alpha * p
            r -= alpha * q

            res_norm = np.linalg.norm(r)
            rel_res = res_norm / b_norm
            residual_history.append(rel_res)

            if rel_res < self.tol:
                info = {
                    "iter": k + 1,
                    "residual": res_norm,
                    "rel_residual": rel_res,
                    "converged": True,
                    "reason": "tolerance reached",
                }
                return x, info

            z = self._apply_preconditioner(M_diag, r)
            rho_new = np.dot(r, z)
            beta = rho_new / rho
            p = z + beta * p
            rho = rho_new

        info = {
            "iter": self.max_iter,
            "residual": np.linalg.norm(r),
            "rel_residual": np.linalg.norm(r) / b_norm,
            "converged": False,
            "reason": "max_iter reached",
        }
        return x, info

    def solve_dense(
        self,
        A: np.ndarray,
        b: np.ndarray,
        x0: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, dict]:
        A = np.asarray(A, dtype=float)
        M_diag = np.diag(A)

        def matvec(v):
            return A.dot(v)

        return self.solve(matvec, b, x0, M_diag)


class SparseMatrixOperator:

    def __init__(
        self,
        nr: int,
        ntheta: int,
        dr: float,
        dtheta: float,
        r_grid: np.ndarray,
        theta_grid: np.ndarray,
        dt: float,
        theta_implicit: float = 0.5,
        eta: float = 1.0,
    ):
        self.nr = nr
        self.ntheta = ntheta
        self.dr = dr
        self.dtheta = dtheta
        self.r_grid = r_grid
        self.theta_grid = theta_grid
        self.dt = dt
        self.theta_imp = theta_implicit
        self.eta = eta
        self.n_total = nr * ntheta


        self._build_coefficients()

    def _build_coefficients(self):
        self.c_r_center = -2.0 * self.eta / (self.dr ** 2)
        self.c_r_plus = self.eta / (self.dr ** 2)
        self.c_r_minus = self.eta / (self.dr ** 2)

        self.c_t_center = np.zeros((self.nr, self.ntheta))
        self.c_t_plus = np.zeros((self.nr, self.ntheta))
        self.c_t_minus = np.zeros((self.nr, self.ntheta))

        for i in range(self.nr):
            r = self.r_grid[i]
            r2 = r * r
            for j in range(self.ntheta):
                theta = self.theta_grid[j]

                self.c_t_center[i, j] = -2.0 * self.eta / (r2 * self.dtheta ** 2)
                self.c_t_plus[i, j] = self.eta / (r2 * self.dtheta ** 2)
                self.c_t_minus[i, j] = self.eta / (r2 * self.dtheta ** 2)

                if abs(theta) > 1e-10 and abs(theta - np.pi) > 1e-10:
                    cot_term = self.eta * np.cos(theta) / (r2 * np.sin(theta) * 2.0 * self.dtheta)
                    self.c_t_plus[i, j] -= cot_term
                    self.c_t_minus[i, j] += cot_term

    def matvec(self, v: np.ndarray) -> np.ndarray:
        v2d = v.reshape((self.nr, self.ntheta))
        result = np.zeros_like(v2d)

        for i in range(1, self.nr - 1):
            for j in range(1, self.ntheta - 1):
                val = v2d[i, j]

                val += self.theta_imp * self.dt * (
                    self.c_r_center * v2d[i, j]
                    + self.c_r_plus * v2d[i + 1, j]
                    + self.c_r_minus * v2d[i - 1, j]
                )

                val += self.theta_imp * self.dt * (
                    self.c_t_center[i, j] * v2d[i, j]
                    + self.c_t_plus[i, j] * v2d[i, j + 1]
                    + self.c_t_minus[i, j] * v2d[i, j - 1]
                )
                result[i, j] = val


        return result.reshape(-1)

    def apply_rhs(self, v: np.ndarray) -> np.ndarray:
        v2d = v.reshape((self.nr, self.ntheta))
        result = np.zeros_like(v2d)
        theta_exp = 1.0 - self.theta_imp

        for i in range(1, self.nr - 1):
            for j in range(1, self.ntheta - 1):
                val = v2d[i, j]
                val += theta_exp * self.dt * (
                    self.c_r_center * v2d[i, j]
                    + self.c_r_plus * v2d[i + 1, j]
                    + self.c_r_minus * v2d[i - 1, j]
                )
                val += theta_exp * self.dt * (
                    self.c_t_center[i, j] * v2d[i, j]
                    + self.c_t_plus[i, j] * v2d[i, j + 1]
                    + self.c_t_minus[i, j] * v2d[i, j - 1]
                )
                result[i, j] = val

        return result.reshape(-1)

    def diagonal(self) -> np.ndarray:
        diag = np.ones(self.n_total)
        for i in range(1, self.nr - 1):
            for j in range(1, self.ntheta - 1):
                idx = i * self.ntheta + j
                diag[idx] = 1.0 + self.theta_imp * self.dt * (
                    self.c_r_center + self.c_t_center[i, j]
                )
        return diag
