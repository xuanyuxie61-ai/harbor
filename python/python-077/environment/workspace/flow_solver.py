
import numpy as np
from typing import Tuple, Optional, Callable
from numerical_utils import sor_solve, r8mat_fs, integrate_ode, langford_deriv


class FlowSolver:

    def __init__(self, nx: int = 40, ny: int = 40,
                 Lx: float = 5000.0, Ly: float = 5000.0,
                 rho: float = 1.225, nu: float = 1.5e-5):
        if nx <= 2 or ny <= 2:
            raise ValueError("网格数必须大于 2")
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.rho = rho
        self.nu = nu
        self.p = np.zeros((nx, ny))
        self.u = np.zeros((nx, ny))
        self.v = np.zeros((nx, ny))

    def _build_laplacian_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        nx, ny = self.nx, self.ny
        N = nx * ny
        A = np.zeros((N, N))
        b = np.zeros(N)

        idx = lambda i, j: j * nx + i

        cx = 1.0 / (self.dx ** 2)
        cy = 1.0 / (self.dy ** 2)
        cc = -2.0 * (cx + cy)

        for j in range(ny):
            for i in range(nx):
                k = idx(i, j)
                A[k, k] = cc


                if i > 0:
                    A[k, idx(i - 1, j)] = cx
                if i < nx - 1:
                    A[k, idx(i + 1, j)] = cx


                if j > 0:
                    A[k, idx(i, j - 1)] = cy
                if j < ny - 1:
                    A[k, idx(i, j + 1)] = cy


                b[k] = self._source_term(i, j)


        for i in range(nx):
            A[idx(i, 0), :] = 0.0
            A[idx(i, 0), idx(i, 0)] = 1.0
            b[idx(i, 0)] = 0.0
            A[idx(i, ny - 1), :] = 0.0
            A[idx(i, ny - 1), idx(i, ny - 1)] = 1.0
            b[idx(i, ny - 1)] = 0.0
        for j in range(ny):
            A[idx(0, j), :] = 0.0
            A[idx(0, j), idx(0, j)] = 1.0
            b[idx(0, j)] = 0.0
            A[idx(nx - 1, j), :] = 0.0
            A[idx(nx - 1, j), idx(nx - 1, j)] = 1.0
            b[idx(nx - 1, j)] = 0.0

        return A, b

    def _source_term(self, i: int, j: int) -> float:

        x = i * self.dx
        y = j * self.dy

        cx, cy = self.Lx / 2.0, self.Ly / 2.0
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        sigma = 200.0
        return -self.rho * np.exp(-r**2 / (2 * sigma**2)) * 0.01

    def solve_pressure_poisson_sor(self, omega: float = 1.8,
                                    tol: float = 1e-6,
                                    max_iter: int = 5000) -> np.ndarray:
        A, b = self._build_laplacian_matrix()
        N = self.nx * self.ny


        if N <= 400:
            p_vec = r8mat_fs(N, A.copy(), b.copy())
            self.p = p_vec.reshape((self.nx, self.ny))
            return self.p


        p_vec = np.zeros(N)
        p_vec, iters = sor_solve(A, b, w=omega, tol=tol, max_iter=max_iter)
        self.p = p_vec.reshape((self.nx, self.ny))
        return self.p

    def solve_pressure_poisson_direct(self) -> np.ndarray:
        A, b = self._build_laplacian_matrix()
        N = self.nx * self.ny
        p_vec = r8mat_fs(N, A.copy(), b.copy())
        self.p = p_vec.reshape((self.nx, self.ny))
        return self.p

    def velocity_correction(self) -> Tuple[np.ndarray, np.ndarray]:
        nx, ny = self.nx, self.ny
        dt = 1.0
        u_new = self.u.copy()
        v_new = self.v.copy()

        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                dpdx = (self.p[i + 1, j] - self.p[i - 1, j]) / (2.0 * self.dx)
                dpdy = (self.p[i, j + 1] - self.p[i, j - 1]) / (2.0 * self.dy)
                u_new[i, j] -= (dt / self.rho) * dpdx
                v_new[i, j] -= (dt / self.rho) * dpdy

        self.u = u_new
        self.v = v_new
        return self.u, self.v

    def set_inflow(self, u_inflow: float):
        self.u[:, 0] = u_inflow
        self.v[:, 0] = 0.0

    def compute_divergence(self) -> np.ndarray:
        nx, ny = self.nx, self.ny
        div = np.zeros((nx, ny))
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                dudx = (self.u[i + 1, j] - self.u[i - 1, j]) / (2.0 * self.dx)
                dvdy = (self.v[i, j + 1] - self.v[i, j - 1]) / (2.0 * self.dy)
                div[i, j] = dudx + dvdy
        return div

    def compute_vorticity(self) -> np.ndarray:
        nx, ny = self.nx, self.ny
        vort = np.zeros((nx, ny))
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                dvdx = (self.v[i + 1, j] - self.v[i - 1, j]) / (2.0 * self.dx)
                dudy = (self.u[i, j + 1] - self.u[i, j - 1]) / (2.0 * self.dy)
                vort[i, j] = dvdx - dudy
        return vort

    def turbulence_kinetic_energy(self) -> np.ndarray:
        nx, ny = self.nx, self.ny
        tke = np.zeros((nx, ny))
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                dudx = (self.u[i + 1, j] - self.u[i - 1, j]) / (2.0 * self.dx)
                dvdy = (self.v[i, j + 1] - self.v[i, j - 1]) / (2.0 * self.dy)

                tke[i, j] = 0.5 * (dudx**2 + dvdy**2) * self.dx**2
        return tke


class TurbulenceCascade:

    def __init__(self, beta_star: float = 0.09, beta: float = 0.075,
                 gamma: float = 0.553):
        self.beta_star = beta_star
        self.beta = beta
        self.gamma = gamma

    def rhs(self, t: float, y: np.ndarray, P_K: float) -> np.ndarray:
        K, omega = y

        K = max(K, 1e-10)
        omega = max(omega, 1e-10)


        dK = P_K - self.beta_star * K * omega

        domega = self.gamma * (omega / K) * P_K - self.beta * omega ** 2

        return np.array([dK, domega])

    def integrate(self, y0: np.ndarray, t_span: Tuple[float, float],
                  P_K: float, n_steps: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
        f = lambda t, y: self.rhs(t, y, P_K)
        return integrate_ode(f, y0, t_span, n_steps)
