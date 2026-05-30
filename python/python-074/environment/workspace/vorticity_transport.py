
import numpy as np


class VorticityTransportSolver:

    def __init__(self, nx, ny, lx, ly, nu, u_inf, dt, cylinder_params=None):
        if nx < 5 or ny < 5:
            raise ValueError("网格数 nx, ny 至少为 5。")
        if nu <= 0:
            raise ValueError("运动粘性系数 nu 必须为正。")
        if dt <= 0:
            raise ValueError("时间步长 dt 必须为正。")

        self.nx = nx
        self.ny = ny
        self.lx = lx
        self.ly = ly
        self.nu = nu
        self.u_inf = u_inf
        self.dt = dt

        self.dx = lx / (nx - 1)
        self.dy = ly / (ny - 1)


        cfl_x = u_inf * dt / self.dx
        cfl_y = u_inf * dt / self.dy
        diff_num_x = nu * dt / (self.dx ** 2)
        diff_num_y = nu * dt / (self.dy ** 2)

        if cfl_x > 1.0 or cfl_y > 1.0:

            pass
        if diff_num_x > 0.5 or diff_num_y > 0.5:

            pass


        self.omega = np.zeros((ny, nx))
        self.omega_old = np.zeros((ny, nx))
        self.omega_old2 = np.zeros((ny, nx))
        self.psi = np.zeros((ny, nx))
        self.u = np.zeros((ny, nx))
        self.v = np.zeros((ny, nx))


        if cylinder_params is None:
            cylinder_params = {'cx': lx * 0.25, 'cy': ly * 0.5, 'r': min(lx, ly) * 0.05}
        self.cx = cylinder_params['cx']
        self.cy = cylinder_params['cy']
        self.r_cyl = cylinder_params['r']
        if self.r_cyl <= 0:
            raise ValueError("圆柱半径必须为正。")


        self.x_grid = np.linspace(0.0, lx, nx)
        self.y_grid = np.linspace(0.0, ly, ny)
        self.X, self.Y = np.meshgrid(self.x_grid, self.y_grid)


        dist_sq = (self.X - self.cx) ** 2 + (self.Y - self.cy) ** 2
        self.solid_mask = dist_sq <= self.r_cyl ** 2
        self.fluid_mask = ~self.solid_mask


        self.interior_indices = []
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                if self.fluid_mask[j, i]:
                    self.interior_indices.append((j, i))


        for j in range(ny):
            self.psi[j, :] = u_inf * self.y_grid[j]
        self.psi[self.solid_mask] = 0.0

        self.u[:, :] = u_inf
        self.u[self.solid_mask] = 0.0

    def _is_inside_cylinder(self, x, y):
        return (x - self.cx) ** 2 + (y - self.cy) ** 2 <= self.r_cyl ** 2

    def apply_boundary_conditions(self):
        ny, nx = self.ny, self.nx


        self.omega[:, 0] = 0.0
        self.psi[:, 0] = self.u_inf * self.Y[:, 0]


        self.omega[:, nx - 1] = self.omega[:, nx - 2]
        self.psi[:, nx - 1] = self.psi[:, nx - 2]


        self.omega[0, :] = 0.0
        self.omega[ny - 1, :] = 0.0
        self.psi[0, :] = 0.0
        self.psi[ny - 1, :] = self.u_inf * self.ly



        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                if self.solid_mask[j, i]:
                    continue

                neighbors = [
                    (j - 1, i, self.dy),
                    (j + 1, i, self.dy),
                    (j, i - 1, self.dx),
                    (j, i + 1, self.dx),
                ]
                for nj, ni, dn in neighbors:
                    if 0 <= nj < ny and 0 <= ni < nx and self.solid_mask[nj, ni]:





                        wall_omega = -2.0 * self.psi[j, i] / (dn ** 2)
                        self.omega[nj, ni] = wall_omega


        self.omega[self.solid_mask] = 0.0
        self.psi[self.solid_mask] = 0.0

    def compute_velocity_from_psi(self):
        ny, nx = self.ny, self.nx
        dx, dy = self.dx, self.dy


        self.u[1:ny - 1, 1:nx - 1] = (
            self.psi[2:ny, 1:nx - 1] - self.psi[0:ny - 2, 1:nx - 1]
        ) / (2.0 * dy)
        self.v[1:ny - 1, 1:nx - 1] = -(
            self.psi[1:ny - 1, 2:nx] - self.psi[1:ny - 1, 0:nx - 2]
        ) / (2.0 * dx)


        self.u[:, 0] = self.u_inf
        self.u[:, nx - 1] = self.u[:, nx - 2]
        self.u[0, :] = 0.0
        self.u[ny - 1, :] = self.u_inf

        self.v[:, 0] = 0.0
        self.v[:, nx - 1] = self.v[:, nx - 2]
        self.v[0, :] = 0.0
        self.v[ny - 1, :] = 0.0


        self.u[self.solid_mask] = 0.0
        self.v[self.solid_mask] = 0.0

    def convective_term(self, omega_field):
        ny, nx = self.ny, self.nx
        dx, dy = self.dx, self.dy
        conv = np.zeros_like(omega_field)

        for j, i in self.interior_indices:
            u_ij = self.u[j, i]
            v_ij = self.v[j, i]


            if u_ij >= 0:
                dwdx = (omega_field[j, i] - omega_field[j, i - 1]) / dx
            else:
                dwdx = (omega_field[j, i + 1] - omega_field[j, i]) / dx


            if v_ij >= 0:
                dwdy = (omega_field[j, i] - omega_field[j - 1, i]) / dy
            else:
                dwdy = (omega_field[j + 1, i] - omega_field[j, i]) / dy

            conv[j, i] = -(u_ij * dwdx + v_ij * dwdy)

        return conv

    def diffusive_term(self, omega_field):
        ny, nx = self.ny, self.nx
        dx, dy = self.dx, self.dy
        diff = np.zeros_like(omega_field)

        for j, i in self.interior_indices:
            d2wdx2 = (
                omega_field[j, i + 1]
                - 2.0 * omega_field[j, i]
                + omega_field[j, i - 1]
            ) / (dx ** 2)
            d2wdy2 = (
                omega_field[j + 1, i]
                - 2.0 * omega_field[j, i]
                + omega_field[j - 1, i]
            ) / (dy ** 2)
            diff[j, i] = self.nu * (d2wdx2 + d2wdy2)

        return diff

    def time_step(self, step_count):

        c_n = self.convective_term(self.omega)
        d_n = self.diffusive_term(self.omega)

        if step_count == 0:

            rhs = self.omega + self.dt * (c_n + d_n)
        else:
            c_nm1 = self.convective_term(self.omega_old)
            d_nm1 = self.diffusive_term(self.omega_old)
            rhs = (
                self.omega
                + self.dt * (1.5 * c_n - 0.5 * c_nm1)
                + 0.5 * self.dt * d_n
                + 0.5 * self.dt * d_nm1
            )




        self.omega_old2 = self.omega_old.copy()
        self.omega_old = self.omega.copy()
        self.omega = rhs.copy()



        omega_new = self.omega.copy()
        for _ in range(3):
            for j, i in self.interior_indices:
                laplacian = (
                    self.omega[j, i + 1]
                    + self.omega[j, i - 1]
                    + self.omega[j + 1, i]
                    + self.omega[j - 1, i]
                )
                omega_new[j, i] = (
                    rhs[j, i]
                    + 0.5 * self.nu * self.dt / (self.dx ** 2) * laplacian
                ) / (
                    1.0
                    + self.nu * self.dt / (self.dx ** 2)
                    + self.nu * self.dt / (self.dy ** 2)
                )
            self.omega = omega_new.copy()



    def compute_force_coefficients(self):

        n_surf = 128
        theta = np.linspace(0.0, 2.0 * np.pi, n_surf, endpoint=False)
        x_surf = self.cx + self.r_cyl * np.cos(theta)
        y_surf = self.cy + self.r_cyl * np.sin(theta)

        omega_surf = np.zeros(n_surf)
        for k in range(n_surf):

            i = int(np.clip(np.round(x_surf[k] / self.dx), 0, self.nx - 1))
            j = int(np.clip(np.round(y_surf[k] / self.dy), 0, self.ny - 1))

            if self.solid_mask[j, i]:
                omega_surf[k] = self.omega[j, i]
            else:

                found = False
                for dj in range(-2, 3):
                    for di in range(-2, 3):
                        nj, ni = j + dj, i + di
                        if 0 <= nj < self.ny and 0 <= ni < self.nx:
                            if self.solid_mask[nj, ni]:
                                omega_surf[k] = self.omega[nj, ni]
                                found = True
                                break
                    if found:
                        break
                if not found:
                    omega_surf[k] = 0.0









        raise NotImplementedError("Hole 1: 壁面涡量积分公式尚未实现")

    def get_wake_profile(self, x_loc):
        i = int(np.clip(np.round(x_loc / self.dx), 0, self.nx - 1))
        profile = self.omega[:, i].copy()

        profile[self.solid_mask[:, i]] = np.nan
        return profile, self.y_grid.copy()
