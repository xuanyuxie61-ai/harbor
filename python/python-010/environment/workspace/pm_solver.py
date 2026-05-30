
import numpy as np
from typing import Tuple
from linalg_utils import SparseCRS, build_laplacian_1d, solve_tridiagonal


class PMSolver:

    def __init__(self, N: int, L: float, G: float = 4.30091e-9):
        self.N = N
        self.L = L
        self.dx = L / N
        self.G = G
        self.volume = L ** 3

        self._setup_green_function()

    def _setup_green_function(self):
        k_vec = 2.0 * np.pi * np.fft.fftfreq(self.N, d=self.dx)
        kx, ky, kz = np.meshgrid(k_vec, k_vec, k_vec, indexing="ij")
        self.k2 = kx ** 2 + ky ** 2 + kz ** 2

        self.green = np.zeros_like(self.k2)
        mask = self.k2 > 0.0
        self.green[mask] = -4.0 * np.pi * self.G / self.k2[mask]

    def cic_deposit(
        self, pos: np.ndarray, mass: np.ndarray
    ) -> np.ndarray:
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("pos 必须为 (N_p, 3) 数组")
        n_part = pos.shape[0]
        rho = np.zeros((self.N, self.N, self.N), dtype=float)


        xg = pos[:, 0] / self.dx
        yg = pos[:, 1] / self.dx
        zg = pos[:, 2] / self.dx

        i0 = np.floor(xg).astype(int) % self.N
        j0 = np.floor(yg).astype(int) % self.N
        k0 = np.floor(zg).astype(int) % self.N

        dx1 = xg - i0
        dy1 = yg - j0
        dz1 = zg - k0
        dx0 = 1.0 - dx1
        dy0 = 1.0 - dy1
        dz0 = 1.0 - dz1


        for di in [0, 1]:
            wx = dx0 if di == 0 else dx1
            ii = (i0 + di) % self.N
            for dj in [0, 1]:
                wy = dy0 if dj == 0 else dy1
                jj = (j0 + dj) % self.N
                for dk in [0, 1]:
                    wz = dz0 if dk == 0 else dz1
                    kk = (k0 + dk) % self.N
                    w = wx * wy * wz
                    np.add.at(rho, (ii, jj, kk), mass * w)


        rho /= self.dx ** 3
        return rho

    def compute_density_contrast(self, rho: np.ndarray, rho_mean: float) -> np.ndarray:
        if abs(rho_mean) < 1e-30:
            return np.zeros_like(rho)
        delta = (rho - rho_mean) / rho_mean
        return delta

    def solve_poisson_fft(self, delta: np.ndarray, a_scale: float = 1.0) -> np.ndarray:
        delta_k = np.fft.fftn(delta) / (self.N ** 3)
        phi_k = self.green * a_scale ** 2 * delta_k
        phi = np.fft.ifftn(phi_k).real * (self.N ** 3)
        return phi

    def compute_force_from_potential(self, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        gx = -(np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2.0 * self.dx)

        gy = -(np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2.0 * self.dx)

        gz = -(np.roll(phi, -1, axis=2) - np.roll(phi, 1, axis=2)) / (2.0 * self.dx)
        return gx, gy, gz

    def cic_interpolate_force(
        self, pos: np.ndarray, gx: np.ndarray, gy: np.ndarray, gz: np.ndarray
    ) -> np.ndarray:
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("pos 必须为 (N_p, 3)")
        n_part = pos.shape[0]
        acc = np.zeros((n_part, 3), dtype=float)

        xg = pos[:, 0] / self.dx
        yg = pos[:, 1] / self.dx
        zg = pos[:, 2] / self.dx

        i0 = np.floor(xg).astype(int) % self.N
        j0 = np.floor(yg).astype(int) % self.N
        k0 = np.floor(zg).astype(int) % self.N

        dx1 = xg - i0
        dy1 = yg - j0
        dz1 = zg - k0
        dx0 = 1.0 - dx1
        dy0 = 1.0 - dy1
        dz0 = 1.0 - dz1

        for di in [0, 1]:
            wx = dx0 if di == 0 else dx1
            ii = (i0 + di) % self.N
            for dj in [0, 1]:
                wy = dy0 if dj == 0 else dy1
                jj = (j0 + dj) % self.N
                for dk in [0, 1]:
                    wz = dz0 if dk == 0 else dz1
                    kk = (k0 + dk) % self.N
                    w = wx * wy * wz
                    acc[:, 0] += w * gx[ii, jj, kk]
                    acc[:, 1] += w * gy[ii, jj, kk]
                    acc[:, 2] += w * gz[ii, jj, kk]

        return acc

    def solve_poisson_sparse_direct(
        self, delta: np.ndarray, a_scale: float = 1.0
    ) -> np.ndarray:
        phi = np.zeros_like(delta)
        rho_bar_term = 4.0 * np.pi * self.G * a_scale ** 2
        n = self.N
        dx = self.dx


        a_tri = np.ones(n)
        b_tri = -2.0 * np.ones(n)
        c_tri = np.ones(n)



        for j in range(self.N):
            for k in range(self.N):
                rhs = rho_bar_term * delta[:, j, k] * dx ** 2

                rhs[0] = 0.0
                rhs[-1] = 0.0
                b_mod = b_tri.copy()
                b_mod[0] = 1.0
                b_mod[-1] = 1.0
                a_mod = a_tri.copy()
                a_mod[0] = 0.0
                c_mod = c_tri.copy()
                c_mod[-1] = 0.0
                try:
                    phi[:, j, k] = solve_tridiagonal(a_mod, b_mod, c_mod, rhs)
                except RuntimeError:
                    phi[:, j, k] = 0.0
        return phi

    def compute_gravity(
        self,
        pos: np.ndarray,
        mass: np.ndarray,
        rho_mean: float,
        a_scale: float = 1.0,
        use_fft: bool = True,
    ) -> np.ndarray:
        rho = self.cic_deposit(pos, mass)
        delta = self.compute_density_contrast(rho, rho_mean)
        if use_fft:
            phi = self.solve_poisson_fft(delta, a_scale)
        else:
            phi = self.solve_poisson_sparse_direct(delta, a_scale)
        gx, gy, gz = self.compute_force_from_potential(phi)
        acc = self.cic_interpolate_force(pos, gx, gy, gz)

        acc = acc / a_scale
        return acc


if __name__ == "__main__":
    N = 32
    L = 100.0
    solver = PMSolver(N, L)
    n_part = N ** 3
    pos = np.random.rand(n_part, 3) * L
    mass = np.ones(n_part) * 1e10
    rho_mean = n_part * 1e10 / (L ** 3)
    acc = solver.compute_gravity(pos, mass, rho_mean)
    print("加速度统计:")
    print(f"  mean = {acc.mean(axis=0)}")
    print(f"  std = {acc.std(axis=0)}")
