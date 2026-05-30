
import numpy as np
from finite_difference import apply_laplacian_3d


class NernstPlanckSolver:
    def __init__(self, shape, dx, dy, dz, D_k=1.96e-9, D_na=1.33e-9,
                 z_k=1.0, z_na=1.0, T=300.0):
        self.shape = shape
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.D = {'K': D_k, 'Na': D_na}
        self.z = {'K': z_k, 'Na': z_na}
        kB = 1.380649e-23
        e_charge = 1.602176634e-19
        self.mu = {
            'K': D_k * z_k * e_charge / (kB * T),
            'Na': D_na * z_na * e_charge / (kB * T)
        }
        self.T = T

    def _diffusion_step(self, c, D, dt):
        lap_c = apply_laplacian_3d(c, self.dx, self.dy, self.dz)
        c_star = c + 0.5 * dt * D * lap_c

        c_star = np.maximum(c_star, 0.0)
        return c_star

    def _migration_step(self, c, mu, phi, dt):
        dphi_dx = np.gradient(phi, self.dx, axis=0)
        dphi_dy = np.gradient(phi, self.dy, axis=1)
        dphi_dz = np.gradient(phi, self.dz, axis=2)

        dc_dx = np.gradient(c, self.dx, axis=0)
        dc_dy = np.gradient(c, self.dy, axis=1)
        dc_dz = np.gradient(c, self.dz, axis=2)

        lap_phi = apply_laplacian_3d(phi, self.dx, self.dy, self.dz)

        div_c_grad_phi = dc_dx * dphi_dx + dc_dy * dphi_dy + dc_dz * dphi_dz + c * lap_phi
        c_new = c - 0.5 * dt * mu * div_c_grad_phi
        c_new = np.maximum(c_new, 0.0)
        return c_new

    def solve_step(self, c_k, c_na, phi, dt):







        raise NotImplementedError("Hole 3: 请实现 Strang 算子分裂 PNP 时间步")

    def steady_state_flux(self, c, phi, ion='K'):
        D = self.D[ion]
        mu = self.mu[ion]

        dc_dx = np.gradient(c, self.dx, axis=0)
        dc_dy = np.gradient(c, self.dy, axis=1)
        dc_dz = np.gradient(c, self.dz, axis=2)

        dphi_dx = np.gradient(phi, self.dx, axis=0)
        dphi_dy = np.gradient(phi, self.dy, axis=1)
        dphi_dz = np.gradient(phi, self.dz, axis=2)

        Jx = -D * dc_dx - mu * c * dphi_dx
        Jy = -D * dc_dy - mu * c * dphi_dy
        Jz = -D * dc_dz - mu * c * dphi_dz

        return Jx, Jy, Jz

    def permeability_coefficient(self, c_k, c_na, phi, channel_area=1.0e-18):
        Jx_k, Jy_k, Jz_k = self.steady_state_flux(c_k, phi, ion='K')
        Jx_na, Jy_na, Jz_na = self.steady_state_flux(c_na, phi, ion='Na')


        nz = self.shape[2]
        mid = nz // 2
        avg_Jz_k = np.mean(np.abs(Jz_k[:, :, mid]))
        avg_Jz_na = np.mean(np.abs(Jz_na[:, :, mid]))


        delta_c = 100.0
        P_k = avg_Jz_k / delta_c
        P_na = avg_Jz_na / delta_c

        selectivity = P_k / (P_na + 1e-30)
        return P_k, P_na, selectivity


def pnp_steady_state_iterator(shape, dx, dy, dz, phi_solver, np_solver,
                               c_k_init, c_na_init, max_iter=50, dt=1e-12):
    c_k = c_k_init.copy()
    c_na = c_na_init.copy()

    for it in range(max_iter):
        phi = phi_solver.solve(conc_k_bulk=np.mean(c_k), conc_na_bulk=np.mean(c_na))
        c_k_new, c_na_new = np_solver.solve_step(c_k, c_na, phi, dt)

        err_k = np.max(np.abs(c_k_new - c_k))
        err_na = np.max(np.abs(c_na_new - c_na))

        c_k = c_k_new
        c_na = c_na_new

        if max(err_k, err_na) < 1e-6:
            break

    return c_k, c_na, phi
