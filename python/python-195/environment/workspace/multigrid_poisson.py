
import numpy as np
from typing import Tuple, Optional, Callable
from utils import (
    gauss_seidel_sweep, restrict_fine_to_coarse,
    restrict_coarse_to_fine, is_power_of_two, EPSILON_MACHINE
)


class MultigridPoisson1D:
    def __init__(self, n: int, a: float, b: float,
                 ua: float, ub: float,
                 force_func: Callable[[np.ndarray], np.ndarray],
                 exact_func: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        if not is_power_of_two(n):
            raise ValueError(f"n must be a power of 2, got {n}")
        self.n = n
        self.a = a
        self.b = b
        self.ua = ua
        self.ub = ub
        self.force_func = force_func
        self.exact_func = exact_func
        self.x = np.linspace(a, b, n + 1)
        self.h = (b - a) / n

    def _build_rhs(self) -> np.ndarray:
        rhs = np.zeros(self.n + 1)
        rhs[0] = self.ua
        rhs[self.n] = self.ub
        interior_x = self.x[1:self.n]
        rhs[1:self.n] = (self.h ** 2) * self.force_func(interior_x)
        return rhs

    def jacobi_sweep(self, n_level: int, rhs: np.ndarray, u: np.ndarray,
                     omega: float = 1.0) -> np.ndarray:
        u_new = u.copy()
        for i in range(1, n_level):
            u_new[i] = (1.0 - omega) * u[i] + omega * 0.5 * (rhs[i] + u[i - 1] + u[i + 1])
        u_new[0] = self.ua
        u_new[n_level] = self.ub
        return u_new

    def _residual(self, n_level: int, rhs: np.ndarray, u: np.ndarray) -> np.ndarray:
        r = np.zeros_like(u)
        r[0] = 0.0
        r[n_level] = 0.0
        for i in range(1, n_level):
            r[i] = rhs[i] - (2.0 * u[i] - u[i - 1] - u[i + 1])
        return r

    def _v_cycle(self, n_level: int, rhs: np.ndarray, u: np.ndarray,
                 nu1: int = 2, nu2: int = 2, omega: float = 0.8) -> np.ndarray:
        if n_level <= 2:

            for _ in range(50):
                u = self.jacobi_sweep(n_level, rhs, u, omega=1.0)
            return u


        for _ in range(nu1):
            u = self.jacobi_sweep(n_level, rhs, u, omega=omega)


        res = self._residual(n_level, rhs, u)


        n_coarse = n_level // 2
        u_coarse = np.zeros(n_coarse + 1)
        rhs_coarse = np.zeros(n_coarse + 1)


        rhs_coarse[0] = res[0]
        rhs_coarse[n_coarse] = res[n_level]
        for j in range(1, n_coarse):
            fine_idx = 2 * j
            rhs_coarse[j] = (
                0.25 * res[fine_idx - 1]
                + 0.5 * res[fine_idx]
                + 0.25 * res[fine_idx + 1]
            )


        e_coarse = self._v_cycle(n_coarse, rhs_coarse, u_coarse, nu1, nu2, omega)


        e_fine = np.zeros(n_level + 1)
        for j in range(n_coarse):
            e_fine[2 * j] = e_coarse[j]
            e_fine[2 * j + 1] = 0.5 * (e_coarse[j] + e_coarse[j + 1])
        e_fine[n_level] = e_coarse[n_coarse]


        u = u + e_fine


        for _ in range(nu2):
            u = self.jacobi_sweep(n_level, rhs, u, omega=omega)

        return u

    def solve(self, tol: float = 1e-6, max_iter: int = 100,
              nu1: int = 2, nu2: int = 2, omega: float = 0.8) -> Tuple[np.ndarray, int]:
        rhs = self._build_rhs()
        u = np.zeros(self.n + 1)
        u[0] = self.ua
        u[self.n] = self.ub

        for it in range(max_iter):
            u_old = u.copy()
            u = self._v_cycle(self.n, rhs, u, nu1, nu2, omega)
            res = self._residual(self.n, rhs, u)
            res_norm = np.max(np.abs(res[1:self.n]))
            change = np.max(np.abs(u - u_old))

            if res_norm < tol or change < tol * 0.1:
                return u, it + 1

        print(f"[WARNING] Multigrid 1D: max_iter={max_iter} reached, res_norm={res_norm:.3e}")
        return u, max_iter


class MultigridPoisson2D:
    def __init__(self, nx: int, ny: int, Lx: float = 1.0, Ly: float = 1.0):
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.hx = Lx / nx
        self.hy = Ly / ny
        self.dx2 = self.hx ** 2
        self.dy2 = self.hy ** 2

    def apply_operator(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        Au = np.zeros_like(u)
        for i in range(1, self.nx):
            for j in range(1, self.ny):
                Au[i, j] = (
                    -(u[i + 1, j] + u[i - 1, j] - 2.0 * u[i, j]) / self.dx2
                    -(u[i, j + 1] + u[i, j - 1] - 2.0 * u[i, j]) / self.dy2
                )
        return Au

    def jacobi_sweep_2d(self, rhs: np.ndarray, u: np.ndarray,
                        omega: float = 0.8) -> np.ndarray:
        u_new = u.copy()
        denom = 2.0 / self.dx2 + 2.0 / self.dy2
        for i in range(1, self.nx):
            for j in range(1, self.ny):
                u_new[i, j] = (1.0 - omega) * u[i, j] + omega * (
                    rhs[i, j]
                    + (u[i + 1, j] + u[i - 1, j]) / self.dx2
                    + (u[i, j + 1] + u[i, j - 1]) / self.dy2
                ) / denom
        return u_new

    def restrict_2d(self, u_fine: np.ndarray) -> np.ndarray:
        nf_x, nf_y = u_fine.shape
        nc_x = (nf_x - 1) // 2 + 1
        nc_y = (nf_y - 1) // 2 + 1
        u_coarse = np.zeros((nc_x, nc_y))

        for ic in range(1, nc_x - 1):
            for jc in range(1, nc_y - 1):
                ifi, jfi = 2 * ic, 2 * jc
                u_coarse[ic, jc] = (
                    0.25 * u_fine[ifi, jfi]
                    + 0.125 * (u_fine[ifi - 1, jfi] + u_fine[ifi + 1, jfi]
                               + u_fine[ifi, jfi - 1] + u_fine[ifi, jfi + 1])
                    + 0.0625 * (u_fine[ifi - 1, jfi - 1] + u_fine[ifi + 1, jfi - 1]
                                + u_fine[ifi - 1, jfi + 1] + u_fine[ifi + 1, jfi + 1])
                )


        u_coarse[0, :] = u_fine[0, ::2]
        u_coarse[-1, :] = u_fine[-1, ::2]
        u_coarse[:, 0] = u_fine[::2, 0]
        u_coarse[:, -1] = u_fine[::2, -1]
        return u_coarse

    def prolong_2d(self, u_coarse: np.ndarray, shape_fine: Tuple[int, int]) -> np.ndarray:
        nc_x, nc_y = u_coarse.shape
        nf_x, nf_y = shape_fine
        u_fine = np.zeros((nf_x, nf_y))

        for ic in range(nc_x - 1):
            for jc in range(nc_y - 1):
                ifi, jfi = 2 * ic, 2 * jc
                if ifi >= nf_x or jfi >= nf_y:
                    continue
                u_fine[ifi, jfi] = u_coarse[ic, jc]
                if ifi + 2 < nf_x:
                    u_fine[ifi + 2, jfi] = u_coarse[ic + 1, jc]
                if jfi + 2 < nf_y:
                    u_fine[ifi, jfi + 2] = u_coarse[ic, jc + 1]
                if ifi + 2 < nf_x and jfi + 2 < nf_y:
                    u_fine[ifi + 2, jfi + 2] = u_coarse[ic + 1, jc + 1]
                if ifi + 1 < nf_x:
                    u_fine[ifi + 1, jfi] = 0.5 * (u_coarse[ic, jc] + u_coarse[ic + 1, jc])
                if jfi + 1 < nf_y:
                    u_fine[ifi, jfi + 1] = 0.5 * (u_coarse[ic, jc] + u_coarse[ic, jc + 1])
                if ifi + 1 < nf_x and jfi + 2 < nf_y:
                    u_fine[ifi + 1, jfi + 2] = 0.5 * (u_coarse[ic, jc + 1] + u_coarse[ic + 1, jc + 1])
                if ifi + 2 < nf_x and jfi + 1 < nf_y:
                    u_fine[ifi + 2, jfi + 1] = 0.5 * (u_coarse[ic + 1, jc] + u_coarse[ic + 1, jc + 1])
                if ifi + 1 < nf_x and jfi + 1 < nf_y:
                    u_fine[ifi + 1, jfi + 1] = 0.25 * (
                        u_coarse[ic, jc] + u_coarse[ic + 1, jc]
                        + u_coarse[ic, jc + 1] + u_coarse[ic + 1, jc + 1]
                    )


        u_fine[0, :] = u_fine[0, :]
        u_fine[-1, :] = u_fine[-1, :]
        u_fine[:, 0] = u_fine[:, 0]
        u_fine[:, -1] = u_fine[:, -1]
        return u_fine

    def _v_cycle_2d(self, u: np.ndarray, rhs: np.ndarray,
                    nu1: int = 2, nu2: int = 2, omega: float = 0.8,
                    min_size: int = 4) -> np.ndarray:
        nx, ny = u.shape
        nx -= 1
        ny -= 1

        if nx <= min_size or ny <= min_size:
            for _ in range(50):
                u = self.jacobi_sweep_2d(rhs, u, omega=1.0)
            return u


        for _ in range(nu1):
            u = self.jacobi_sweep_2d(rhs, u, omega=omega)


        res = rhs - self.apply_operator(u)


        rhs_coarse = self.restrict_2d(res)
        u_coarse = np.zeros_like(rhs_coarse)


        sub = MultigridPoisson2D(
            (rhs_coarse.shape[0] - 1), (rhs_coarse.shape[1] - 1),
            self.Lx, self.Ly
        )
        e_coarse = sub._v_cycle_2d(u_coarse, rhs_coarse, nu1, nu2, omega, min_size)


        e_fine = self.prolong_2d(e_coarse, u.shape)
        u = u + e_fine


        for _ in range(nu2):
            u = self.jacobi_sweep_2d(rhs, u, omega=omega)

        return u

    def solve(self, rhs: np.ndarray, u0: Optional[np.ndarray] = None,
              tol: float = 1e-6, max_iter: int = 50,
              nu1: int = 2, nu2: int = 2, omega: float = 0.8) -> Tuple[np.ndarray, int]:
        rhs = np.asarray(rhs, dtype=float)
        if u0 is None:
            u = np.zeros((self.nx + 1, self.ny + 1))
        else:
            u = u0.copy()

        for it in range(max_iter):
            u_old = u.copy()
            u = self._v_cycle_2d(u, rhs, nu1, nu2, omega)
            res = rhs - self.apply_operator(u)
            res_norm = np.max(np.abs(res[1:self.nx, 1:self.ny]))
            change = np.max(np.abs(u - u_old))
            if res_norm < tol and change < tol:
                return u, it + 1

        return u, max_iter
