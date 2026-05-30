
import numpy as np
from typing import Tuple


class PoissonBoltzmannSolver:

    def __init__(self, R_np: float = 2.5, R_max: float = 25.0,
                 z_np: float = +10.0, n_0: float = 0.1,
                 epsilon: float = 80.0 * 8.854e-12,
                 T: float = 300.0):
        self.R_np = float(R_np)
        self.R_max = float(R_max)
        self.z_np = float(z_np)
        self.n_0 = float(n_0)
        self.epsilon = float(epsilon)
        self.T = float(T)

        self.k_B = 1.380649e-23
        self.e_charge = 1.602176634e-19
        self.N_A = 6.02214076e23

        ionic_strength = n_0 * 1000.0 * self.N_A
        self.kappa_D = np.sqrt(2 * self.e_charge ** 2 * ionic_strength /
                               (self.epsilon * self.k_B * self.T))
        self.kappa_D_nm = self.kappa_D * 1e-9

    def _build_system(self, n_grid: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:










        raise NotImplementedError("HOLE 1: 请补全 _build_system 的离散化实现")

    def _u_to_phi(self, u: np.ndarray, r: np.ndarray) -> np.ndarray:
        phi = np.zeros_like(u)

        mask = r > 1e-12
        phi[mask] = u[mask] / r[mask]

        phi[~mask] = u[~mask] / (r[~mask] + 1e-12)
        return phi

    def solve_jacobi(self, n_grid: int = 257, it_max: int = 50000,
                     tol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray, int, float]:
        A, b, r = self._build_system(n_grid)
        u = np.zeros(n_grid, dtype=np.float64)
        u[0] = b[0]
        u[-1] = b[-1]
        diag = np.diag(A).copy()
        diag[diag == 0] = 1.0
        for it in range(1, it_max + 1):
            u_new = (b - A @ u + diag * u) / diag
            u_new[0] = b[0]
            u_new[-1] = b[-1]
            res = np.linalg.norm(A @ u_new - b) / np.sqrt(n_grid)
            u = u_new
            if res < tol:
                phi = self._u_to_phi(u, r)
                return phi, r, it, res
        phi = self._u_to_phi(u, r)
        return phi, r, it_max, res

    def solve_bicg(self, n_grid: int = 257, max_it: int = 2000,
                   tol: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, int, float]:
        A, b, r = self._build_system(n_grid)
        x = np.zeros(n_grid, dtype=np.float64)

        diag_A = np.diag(A).copy()
        diag_A[diag_A == 0] = 1.0
        M_inv = 1.0 / diag_A

        r_vec = b - A @ x
        r_tld = r_vec.copy()
        if np.linalg.norm(r_vec) < tol:
            phi = self._u_to_phi(x, r)
            return phi, r, 0, np.linalg.norm(r_vec)
        z = M_inv * r_vec
        z_tld = M_inv * r_tld
        p = z.copy()
        p_tld = z_tld.copy()
        rho_old = np.dot(z, r_tld)
        for it in range(1, max_it + 1):
            q = A @ p
            q_tld = A.T @ p_tld
            denom = np.dot(p_tld, q)
            if abs(denom) < 1e-30:
                break
            alpha = rho_old / denom
            x = x + alpha * p
            r_vec = r_vec - alpha * q
            r_tld = r_tld - alpha * q_tld
            residual = np.linalg.norm(r_vec)
            if residual < tol:
                phi = self._u_to_phi(x, r)
                return phi, r, it, residual
            z = M_inv * r_vec
            z_tld = M_inv * r_tld
            rho = np.dot(z, r_tld)
            if abs(rho_old) < 1e-30:
                break
            beta = rho / rho_old
            p = z + beta * p
            p_tld = z_tld + beta * p_tld
            rho_old = rho

        x[0] = b[0]
        x[-1] = b[-1]
        phi = self._u_to_phi(x, r)
        return phi, r, it, np.linalg.norm(b - A @ x)

    def debye_length(self) -> float:
        return 1.0 / self.kappa_D_nm if self.kappa_D_nm > 0 else float('inf')

    def electrostatic_force(self, n_grid: int = 257) -> float:
        A, b, r = self._build_system(n_grid)
        x = np.zeros(n_grid, dtype=np.float64)
        diag_A = np.diag(A).copy()
        diag_A[diag_A == 0] = 1.0
        M_inv = 1.0 / diag_A
        r_vec = b - A @ x
        r_tld = r_vec.copy()
        z = M_inv * r_vec
        z_tld = M_inv * r_tld
        p = z.copy()
        p_tld = z_tld.copy()
        rho_old = np.dot(z, r_tld)
        for it in range(1, 2000):
            q = A @ p
            q_tld = A.T @ p_tld
            denom = np.dot(p_tld, q)
            if abs(denom) < 1e-30:
                break
            alpha = rho_old / denom
            x = x + alpha * p
            r_vec = r_vec - alpha * q
            r_tld = r_tld - alpha * q_tld
            if np.linalg.norm(r_vec) < 1e-10:
                break
            z = M_inv * r_vec
            z_tld = M_inv * r_tld
            rho = np.dot(z, r_tld)
            if abs(rho_old) < 1e-30:
                break
            beta = rho / rho_old
            p = z + beta * p
            p_tld = z_tld + beta * p_tld
            rho_old = rho
        x[0] = b[0]
        x[-1] = b[-1]

        h = r[1] - r[0]
        dudr = (-3.0 * x[0] + 4.0 * x[1] - x[2]) / (2.0 * h)
        u0 = x[0]
        R = self.R_np


        dphi_dr = -(dudr * R - u0) / (R ** 2)
        return float(dphi_dr)
