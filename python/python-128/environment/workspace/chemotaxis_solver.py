
import numpy as np
from special_math import tridiag_solve


class ChemotaxisSolver:

    def __init__(self,
                 nx: int = 32, ny: int = 32, nz: int = 16,
                 xlim: tuple = (-1.0, 1.0),
                 ylim: tuple = (-1.0, 1.0),
                 zlim: tuple = (-0.5, 0.5),
                 D: float = 0.01,
                 lambda_deg: float = 0.05,
                 Vmax: float = 1.0,
                 Km: float = 0.5):
        self.nx = max(3, int(nx))
        self.ny = max(3, int(ny))
        self.nz = max(3, int(nz))
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        self.D = float(D)
        self.lambda_deg = float(lambda_deg)
        self.Vmax = float(Vmax)
        self.Km = float(Km)

        self.x = np.linspace(xlim[0], xlim[1], self.nx)
        self.y = np.linspace(ylim[0], ylim[1], self.ny)
        self.z = np.linspace(zlim[0], zlim[1], self.nz)
        self.dx = (xlim[1] - xlim[0]) / (self.nx - 1)
        self.dy = (ylim[1] - ylim[0]) / (self.ny - 1)
        self.dz = (zlim[1] - zlim[0]) / (self.nz - 1)


        self.c = np.zeros((self.nx, self.ny, self.nz), dtype=float)

    def set_initial_condition(self, c0_func):
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    self.c[i, j, k] = c0_func(self.x[i], self.y[j], self.z[k])

    def _reaction_source(self, c):
        return self.Vmax * c / (self.Km + c + 1e-12)

    def _safe_dt(self, vx, vy, vz):
        dt_diff = min(self.dx ** 2, self.dy ** 2, self.dz ** 2) / (6.0 * self.D + 1e-15)
        vmax = max(np.max(np.abs(vx)), np.max(np.abs(vy)), np.max(np.abs(vz)))
        dt_adv = min(self.dx, self.dy, self.dz) / (vmax + 1e-15)
        return 0.3 * min(dt_diff, dt_adv)

    def _solve_1d_diffusion_implicit(self, u, dx, dt, dir_axis):

        raise NotImplementedError("Hole 1: _solve_1d_diffusion_implicit 尚未实现")


    def _advection_step_leapfrog_1d(self, u, v, dx, dt, dir_axis):

        u_new = np.copy(u)
        if dir_axis == 0:
            for i in range(1, u.shape[0] - 1):
                coeff = v[i] * dt / dx
                if coeff >= 0:
                    u_new[i, :, :] = u[i, :, :] - coeff * (u[i, :, :] - u[i - 1, :, :])
                else:
                    u_new[i, :, :] = u[i, :, :] - coeff * (u[i + 1, :, :] - u[i, :, :])
        elif dir_axis == 1:
            for j in range(1, u.shape[1] - 1):
                coeff = v[j] * dt / dx
                if coeff >= 0:
                    u_new[:, j, :] = u[:, j, :] - coeff * (u[:, j, :] - u[:, j - 1, :])
                else:
                    u_new[:, j, :] = u[:, j, :] - coeff * (u[:, j + 1, :] - u[:, j, :])
        else:
            for k in range(1, u.shape[2] - 1):
                coeff = v[k] * dt / dx
                if coeff >= 0:
                    u_new[:, :, k] = u[:, :, k] - coeff * (u[:, :, k] - u[:, :, k - 1])
                else:
                    u_new[:, :, k] = u[:, :, k] - coeff * (u[:, :, k + 1] - u[:, :, k])
        return u_new

    def step(self, vx, vy, vz, dt: float = None):
        if np.isscalar(vx):
            vx_arr = np.full(self.nx, float(vx))
        else:
            vx_arr = np.asarray(vx, dtype=float).reshape(self.nx)
        if np.isscalar(vy):
            vy_arr = np.full(self.ny, float(vy))
        else:
            vy_arr = np.asarray(vy, dtype=float).reshape(self.ny)
        if np.isscalar(vz):
            vz_arr = np.full(self.nz, float(vz))
        else:
            vz_arr = np.asarray(vz, dtype=float).reshape(self.nz)

        if dt is None:
            dt = self._safe_dt(vx_arr, vy_arr, vz_arr)
        dt = float(dt)
        if dt <= 1e-15:
            raise ValueError("chemotaxis_solver.step: dt 过小")

        c = self.c



        c = self._solve_1d_diffusion_implicit(c, self.dx, dt, 0)
        c = self._advection_step_leapfrog_1d(c, vx_arr, self.dx, dt, 0)


        c = self._solve_1d_diffusion_implicit(c, self.dy, dt, 1)
        c = self._advection_step_leapfrog_1d(c, vy_arr, self.dy, dt, 1)


        c = self._solve_1d_diffusion_implicit(c, self.dz, dt, 2)
        c = self._advection_step_leapfrog_1d(c, vz_arr, self.dz, dt, 2)


        source = self._reaction_source(c)

        c = (c + dt * source) / (1.0 + dt * self.lambda_deg)


        c = np.maximum(c, 0.0)

        self.c = c
        return dt

    def gradient(self):
        gx = np.zeros_like(self.c)
        gy = np.zeros_like(self.c)
        gz = np.zeros_like(self.c)


        gx[1:-1, :, :] = (self.c[2:, :, :] - self.c[:-2, :, :]) / (2.0 * self.dx)
        gx[0, :, :] = (self.c[1, :, :] - self.c[0, :, :]) / self.dx
        gx[-1, :, :] = (self.c[-1, :, :] - self.c[-2, :, :]) / self.dx


        gy[:, 1:-1, :] = (self.c[:, 2:, :] - self.c[:, :-2, :]) / (2.0 * self.dy)
        gy[:, 0, :] = (self.c[:, 1, :] - self.c[:, 0, :]) / self.dy
        gy[:, -1, :] = (self.c[:, -1, :] - self.c[:, -2, :]) / self.dy


        gz[:, :, 1:-1] = (self.c[:, :, 2:] - self.c[:, :, :-2]) / (2.0 * self.dz)
        gz[:, :, 0] = (self.c[:, :, 1] - self.c[:, :, 0]) / self.dz
        gz[:, :, -1] = (self.c[:, :, -1] - self.c[:, :, -2]) / self.dz

        return gx, gy, gz

    def total_mass(self):
        return np.sum(self.c) * self.dx * self.dy * self.dz
