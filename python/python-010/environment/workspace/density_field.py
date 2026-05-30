
import numpy as np
from typing import Tuple


class AdvectionSolver:

    def __init__(self, nx: int, dx: float, c: float = 1.0):
        self.nx = nx
        self.dx = dx
        self.c = c

    def lax_wendroff_step_1d(self, u: np.ndarray, dt: float) -> np.ndarray:
        nx = self.nx
        if len(u) != nx:
            raise ValueError("u 长度与 nx 不符")

        courant = abs(self.c) * dt / self.dx
        if courant > 1.0:
            raise ValueError(f"CFL 条件破坏: {courant:.3f} > 1")

        c1 = 0.5 * self.c * dt / self.dx
        c2 = 0.5 * (self.c * dt / self.dx) ** 2


        im1 = np.roll(np.arange(nx), 1)
        ip1 = np.roll(np.arange(nx), -1)

        u_new = (
            u
            - c1 * (u[ip1] - u[im1])
            + c2 * (u[ip1] - 2.0 * u + u[im1])
        )
        return u_new

    def evolve_1d(
        self, u0: np.ndarray, t_final: float, n_steps: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        dt = t_final / n_steps
        t_arr = np.linspace(0.0, t_final, n_steps + 1)
        u_arr = np.zeros((n_steps + 1, self.nx))
        u_arr[0] = u0
        u = u0.copy()
        for i in range(n_steps):
            u = self.lax_wendroff_step_1d(u, dt)
            u_arr[i + 1] = u
        return t_arr, u_arr

    def lax_wendroff_step_3d_x(
        self, u: np.ndarray, dt: float
    ) -> np.ndarray:
        nx = u.shape[0]
        c1 = 0.5 * self.c * dt / self.dx
        c2 = 0.5 * (self.c * dt / self.dx) ** 2
        u_new = u.copy()
        for i in range(nx):
            im1 = (i - 1) % nx
            ip1 = (i + 1) % nx
            u_new[i, :, :] = (
                u[i, :, :]
                - c1 * (u[ip1, :, :] - u[im1, :, :])
                + c2 * (u[ip1, :, :] - 2.0 * u[i, :, :] + u[im1, :, :])
            )
        return u_new

    def lax_wendroff_step_3d_y(
        self, u: np.ndarray, dt: float
    ) -> np.ndarray:
        ny = u.shape[1]
        c1 = 0.5 * self.c * dt / self.dx
        c2 = 0.5 * (self.c * dt / self.dx) ** 2
        u_new = u.copy()
        for j in range(ny):
            jm1 = (j - 1) % ny
            jp1 = (j + 1) % ny
            u_new[:, j, :] = (
                u[:, j, :]
                - c1 * (u[:, jp1, :] - u[:, jm1, :])
                + c2 * (u[:, jp1, :] - 2.0 * u[:, j, :] + u[:, jm1, :])
            )
        return u_new

    def lax_wendroff_step_3d_z(
        self, u: np.ndarray, dt: float
    ) -> np.ndarray:
        nz = u.shape[2]
        c1 = 0.5 * self.c * dt / self.dx
        c2 = 0.5 * (self.c * dt / self.dx) ** 2
        u_new = u.copy()
        for k in range(nz):
            km1 = (k - 1) % nz
            kp1 = (k + 1) % nz
            u_new[:, :, k] = (
                u[:, :, k]
                - c1 * (u[:, :, kp1] - u[:, :, km1])
                + c2 * (u[:, :, kp1] - 2.0 * u[:, :, k] + u[:, :, km1])
            )
        return u_new

    def strang_split_3d(
        self, u: np.ndarray, dt: float
    ) -> np.ndarray:
        dt2 = dt * 0.5
        u = self.lax_wendroff_step_3d_x(u, dt2)
        u = self.lax_wendroff_step_3d_y(u, dt2)
        u = self.lax_wendroff_step_3d_z(u, dt)
        u = self.lax_wendroff_step_3d_y(u, dt2)
        u = self.lax_wendroff_step_3d_x(u, dt2)
        return u

    def evolve_3d_density_field(
        self,
        rho0: np.ndarray,
        t_final: float,
        n_steps: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        dt = t_final / n_steps
        t_arr = np.linspace(0.0, t_final, n_steps + 1)
        rho_arr = np.zeros((n_steps + 1,) + rho0.shape)
        rho_arr[0] = rho0
        rho = rho0.copy()
        for i in range(n_steps):
            rho = self.strang_split_3d(rho, dt)
            rho_arr[i + 1] = rho
        return t_arr, rho_arr


def test_mass_conservation() -> float:
    nx = 101
    L = 1.0
    dx = L / (nx - 1)
    x = np.linspace(0.0, L, nx)
    u0 = np.exp(-((x - 0.5) ** 2) / (2 * 0.02 ** 2))
    solver = AdvectionSolver(nx, dx, c=1.0)
    t_arr, u_arr = solver.evolve_1d(u0, t_final=1.0, n_steps=1000)
    mass0 = np.sum(u0) * dx
    mass_final = np.sum(u_arr[-1]) * dx
    return abs(mass_final - mass0) / mass0


if __name__ == "__main__":
    err = test_mass_conservation()
    print(f"一维质量守恒相对误差: {err:.4e}")


    nx = 16
    dx = 1.0
    rho0 = np.random.rand(nx, nx, nx)
    solver3d = AdvectionSolver(nx, dx, c=0.5)
    t_arr, rho_arr = solver3d.evolve_3d_density_field(rho0, t_final=0.1, n_steps=50)
    mass0 = rho0.sum()
    massf = rho_arr[-1].sum()
    print(f"三维质量守恒相对误差: {abs(massf - mass0) / mass0:.4e}")
