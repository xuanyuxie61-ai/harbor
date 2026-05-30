
import numpy as np


class NavierStokesSolver:

    def __init__(self, nx, ny, dx, dy, dt, rho=1.0, mu=0.01,
                 surface_tension=0.1, epsilon=0.01):
        if nx < 3 or ny < 3:
            raise ValueError("网格维度必须至少为 3")
        if dx <= 0 or dy <= 0 or dt <= 0:
            raise ValueError("步长参数必须为正")

        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.rho = rho
        self.mu = mu
        self.surface_tension = surface_tension
        self.epsilon = epsilon


        self.nu = mu / rho
        self.inv_rho = 1.0 / rho

    def compute_surface_tension_force(self, phi):
        phi_clipped = np.clip(phi, -1.2, 1.2)


        grad_phi_x = np.zeros_like(phi)
        grad_phi_y = np.zeros_like(phi)

        grad_phi_x[1:-1, :] = (phi_clipped[2:, :] - phi_clipped[:-2, :]) / (2.0 * self.dx)
        grad_phi_y[:, 1:-1] = (phi_clipped[:, 2:] - phi_clipped[:, :-2]) / (2.0 * self.dy)


        dwdphi = phi_clipped ** 3 - phi_clipped


        coeff = (3.0 * self.surface_tension) / (2.0 * np.sqrt(2.0) * self.epsilon)

        Fx = coeff * dwdphi * grad_phi_x
        Fy = coeff * dwdphi * grad_phi_y

        return Fx, Fy

    def advection_diffusion_velocity(self, vx, vy):
        rhs_x = np.zeros_like(vx)
        rhs_y = np.zeros_like(vy)


        lap_vx = np.zeros_like(vx)
        lap_vy = np.zeros_like(vy)

        lap_vx[1:-1, 1:-1] = (
            (vx[2:, 1:-1] - 2.0 * vx[1:-1, 1:-1] + vx[:-2, 1:-1]) / (self.dx ** 2) +
            (vx[1:-1, 2:] - 2.0 * vx[1:-1, 1:-1] + vx[1:-1, :-2]) / (self.dy ** 2)
        )
        lap_vy[1:-1, 1:-1] = (
            (vy[2:, 1:-1] - 2.0 * vy[1:-1, 1:-1] + vy[:-2, 1:-1]) / (self.dx ** 2) +
            (vy[1:-1, 2:] - 2.0 * vy[1:-1, 1:-1] + vy[1:-1, :-2]) / (self.dy ** 2)
        )

        rhs_x += self.nu * lap_vx
        rhs_y += self.nu * lap_vy



        dvx_dx = np.zeros_like(vx)
        dvx_dy = np.zeros_like(vx)

        mask_pos = vx >= 0
        dvx_dx[1:-1, :][mask_pos[1:-1, :]] = (
            vx[1:-1, :][mask_pos[1:-1, :]] - vx[:-2, :][mask_pos[1:-1, :]]
        ) / self.dx
        mask_neg = vx < 0
        dvx_dx[1:-1, :][mask_neg[1:-1, :]] = (
            vx[2:, :][mask_neg[1:-1, :]] - vx[1:-1, :][mask_neg[1:-1, :]]
        ) / self.dx

        mask_pos = vy >= 0
        dvx_dy[:, 1:-1][mask_pos[:, 1:-1]] = (
            vx[:, 1:-1][mask_pos[:, 1:-1]] - vx[:, :-2][mask_pos[:, 1:-1]]
        ) / self.dy
        mask_neg = vy < 0
        dvx_dy[:, 1:-1][mask_neg[:, 1:-1]] = (
            vx[:, 2:][mask_neg[:, 1:-1]] - vx[:, 1:-1][mask_neg[:, 1:-1]]
        ) / self.dy

        rhs_x -= vx * dvx_dx + vy * dvx_dy


        dvy_dx = np.zeros_like(vy)
        dvy_dy = np.zeros_like(vy)

        mask_pos = vx >= 0
        dvy_dx[1:-1, :][mask_pos[1:-1, :]] = (
            vy[1:-1, :][mask_pos[1:-1, :]] - vy[:-2, :][mask_pos[1:-1, :]]
        ) / self.dx
        mask_neg = vx < 0
        dvy_dx[1:-1, :][mask_neg[1:-1, :]] = (
            vy[2:, :][mask_neg[1:-1, :]] - vy[1:-1, :][mask_neg[1:-1, :]]
        ) / self.dx

        mask_pos = vy >= 0
        dvy_dy[:, 1:-1][mask_pos[:, 1:-1]] = (
            vy[:, 1:-1][mask_pos[:, 1:-1]] - vy[:, :-2][mask_pos[:, 1:-1]]
        ) / self.dy
        mask_neg = vy < 0
        dvy_dy[:, 1:-1][mask_neg[:, 1:-1]] = (
            vy[:, 2:][mask_neg[:, 1:-1]] - vy[:, 1:-1][mask_neg[:, 1:-1]]
        ) / self.dy

        rhs_y -= vx * dvy_dx + vy * dvy_dy

        return rhs_x, rhs_y

    def solve_pressure_poisson_gs(self, div_vstar, max_iter=1000, tol=1e-6):
        p = np.zeros_like(div_vstar)
        rhs = (self.rho / self.dt) * div_vstar

        dx2 = self.dx ** 2
        dy2 = self.dy ** 2
        denom = 2.0 * (1.0 / dx2 + 1.0 / dy2)

        for it in range(max_iter):
            p_old = p.copy()


            for i in range(1, self.nx - 1):
                for j in range(1, self.ny - 1):
                    p[i, j] = (
                        (p[i + 1, j] + p[i - 1, j]) / dx2 +
                        (p[i, j + 1] + p[i, j - 1]) / dy2 -
                        rhs[i, j]
                    ) / denom


            p[0, :] = p[1, :]
            p[-1, :] = p[-2, :]
            p[:, 0] = p[:, 1]
            p[:, -1] = p[:, -2]


            p -= p.mean()


            diff = np.max(np.abs(p - p_old))
            if diff < tol:
                break

        return p

    def projection_step(self, vx_star, vy_star):

        div_vstar = np.zeros_like(vx_star)
        div_vstar[1:-1, 1:-1] = (
            (vx_star[2:, 1:-1] - vx_star[:-2, 1:-1]) / (2.0 * self.dx) +
            (vy_star[1:-1, 2:] - vy_star[1:-1, :-2]) / (2.0 * self.dy)
        )


        p = self.solve_pressure_poisson_gs(div_vstar)


        grad_p_x = np.zeros_like(p)
        grad_p_y = np.zeros_like(p)

        grad_p_x[1:-1, :] = (p[2:, :] - p[:-2, :]) / (2.0 * self.dx)
        grad_p_y[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (2.0 * self.dy)


        vx_new = vx_star - (self.dt / self.rho) * grad_p_x
        vy_new = vy_star - (self.dt / self.rho) * grad_p_y

        return vx_new, vy_new, p

    def time_step(self, vx, vy, phi):

        rhs_x, rhs_y = self.advection_diffusion_velocity(vx, vy)

        Fx, Fy = self.compute_surface_tension_force(phi)
        rhs_x += self.inv_rho * Fx
        rhs_y += self.inv_rho * Fy


        vx_star = vx + self.dt * rhs_x
        vy_star = vy + self.dt * rhs_y


        vx_star[0, :] = 0.0
        vx_star[-1, :] = 0.0
        vx_star[:, 0] = 0.0
        vx_star[:, -1] = 0.0
        vy_star[0, :] = 0.0
        vy_star[-1, :] = 0.0
        vy_star[:, 0] = 0.0
        vy_star[:, -1] = 0.0


        vx_new, vy_new, p = self.projection_step(vx_star, vy_star)

        return vx_new, vy_new, p
