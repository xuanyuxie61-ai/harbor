
import numpy as np
from spectral_solver import SpectralDifferentiator, map_nodes_to_interval
from shock_physics import NonlinearAcousticsPhysics


class StrangSplittingSolver:

    def __init__(self, physics, dr, dtau, Nr, Ntau, r_max, tau_max,
                 diffraction=True, absorption=True, nonlinearity=True):
        self.physics = physics
        self.dr = float(dr)
        self.dtau = float(dtau)
        self.Nr = int(Nr)
        self.Ntau = int(Ntau)
        self.r_max = float(r_max)
        self.tau_max = float(tau_max)
        self.diffraction = diffraction
        self.absorption = absorption
        self.nonlinearity = nonlinearity


        self.r_grid = np.linspace(0.0, r_max, Nr)
        self.tau_grid = np.linspace(-tau_max, tau_max, Ntau)


        self.spec_tau = SpectralDifferentiator(Ntau, node_type='chebyshev_gauss_lobatto')
        self.tau_nodes, self.tau_jac = map_nodes_to_interval(
            self.spec_tau.nodes, -tau_max, tau_max)

        self.D_tau = self.spec_tau.differentiation_matrix
        self.D2_tau = self.spec_tau.second_derivative_matrix()


        self._check_cfl()

    def _check_cfl(self):
        c0 = self.physics.c0

        dz_diff = 0.5 * self.dr ** 2 * self.physics.k0


        p_est = max(self.physics.p0, 1.0)
        dz_nl = self.dtau * c0 / (self.physics.beta * p_est)
        self.dz_max = min(dz_diff, dz_nl)
        if self.dz_max <= 0.0 or not np.isfinite(self.dz_max):
            raise ValueError("CFL condition yields invalid dz_max.")

        self.dz_min = 1e-12

    def _step_diffraction(self, p, dz_half):
        if not self.diffraction:
            return p.copy()

        p_new = p.copy()
        coeff = 1.0 / (2.0 * self.physics.k0)

        for j_tau in range(self.Ntau):
            p_r = p[:, j_tau].copy()

            d2p = np.zeros(self.Nr, dtype=float)
            dp = np.zeros(self.Nr, dtype=float)

            if self.Nr >= 3:
                dp[1:-1] = (p_r[2:] - p_r[:-2]) / (2.0 * self.dr)
                d2p[1:-1] = (p_r[2:] - 2.0 * p_r[1:-1] + p_r[:-2]) / (self.dr ** 2)


            if self.Nr >= 3:
                d2p[0] = 2.0 * (p_r[1] - p_r[0]) / (self.dr ** 2)
                dp[0] = 0.0

                d2p[-1] = d2p[-2]
                dp[-1] = 0.0


            laplacian = d2p.copy()
            if self.Nr > 1:
                r_safe = self.r_grid.copy()
                r_safe[0] = r_safe[1]
                with np.errstate(divide='ignore', invalid='ignore'):
                    laplacian += dp / r_safe
                laplacian[0] = 2.0 * d2p[0]


            p_new[:, j_tau] = p_r + dz_half * coeff * laplacian

        return p_new

    def _step_nonlinear_absorption(self, p, dz):
        p_new = p.copy()
        beta = self.physics.beta
        rho0 = self.physics.rho0
        c0 = self.physics.c0
        alpha = self.physics.classical_absorption


        delta_eff = 2.0 * alpha * c0 ** 3 / self.physics.omega0 ** 2
        if delta_eff < 0.0:
            delta_eff = 0.0

        for i_r in range(self.Nr):
            p_tau = p[i_r, :].copy()


            p_tau_x = self.D_tau @ p_tau
            p_tau_xx = self.D2_tau @ p_tau

            rhs = np.zeros(self.Ntau, dtype=float)
            if self.nonlinearity:
                rhs += (beta / (rho0 * c0 ** 3)) * p_tau * p_tau_x
            if self.absorption:
                rhs += (delta_eff / (2.0 * c0 ** 3)) * p_tau_xx
                rhs -= alpha * p_tau


            def ode_rhs(v):
                v = np.asarray(v, dtype=float)
                vx = self.D_tau @ v
                vxx = self.D2_tau @ v
                r = np.zeros_like(v)
                if self.nonlinearity:
                    r += (beta / (rho0 * c0 ** 3)) * v * vx
                if self.absorption:
                    r += (delta_eff / (2.0 * c0 ** 3)) * vxx
                    r -= alpha * v
                return r

            k1 = ode_rhs(p_tau)
            k2 = ode_rhs(p_tau + 0.5 * dz * k1)
            k3 = ode_rhs(p_tau + 0.5 * dz * k2)
            k4 = ode_rhs(p_tau + dz * k3)
            p_new[i_r, :] = p_tau + (dz / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


            p_new[i_r, 0] = 0.0
            p_new[i_r, -1] = 0.0

        return p_new

    def propagate(self, p_initial, z_max, dz=None):
        p = np.asarray(p_initial, dtype=float)
        if p.shape != (self.Nr, self.Ntau):
            raise ValueError(f"p_initial shape {p.shape} does not match ({self.Nr}, {self.Ntau}).")

        if dz is None:
            dz = self.dz_max * 0.5
        dz = float(dz)
        if dz <= 0.0 or dz > self.dz_max:
            raise ValueError(f"dz={dz} violates CFL condition (max={self.dz_max}).")

        Nz_max = 5000
        Nz = int(np.ceil(z_max / dz))
        if Nz > Nz_max:
            Nz = Nz_max
            dz = z_max / Nz
        dz = z_max / Nz

        z_vec = np.linspace(0.0, z_max, Nz + 1)
        P_history = np.zeros((Nz + 1, self.Nr, self.Ntau), dtype=float)
        P_history[0, :, :] = p

        for step in range(Nz):

            p = self._step_diffraction(p, dz / 2.0)
            p = self._step_nonlinear_absorption(p, dz)
            p = self._step_diffraction(p, dz / 2.0)


            if np.any(~np.isfinite(p)):
                raise RuntimeError(f"Non-finite values detected at z={z_vec[step + 1]}")


            p_max_phys = 1e9
            p = np.clip(p, -p_max_phys, p_max_phys)

            P_history[step + 1, :, :] = p

        return P_history, z_vec


class FiniteVolumeShockCapturing:

    def __init__(self, Nx, x_min, x_max, nu):
        self.Nx = int(Nx)
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.nu = float(nu)
        self.dx = (x_max - x_min) / Nx
        self.x_faces = np.linspace(x_min, x_max, Nx + 1)
        self.x_centers = 0.5 * (self.x_faces[:-1] + self.x_faces[1:])

    def _godunov_flux(self, uL, uR):

        flux = np.where(
            uL <= uR,
            np.minimum(0.5 * uL ** 2, 0.5 * uR ** 2),
            np.maximum(0.5 * uL ** 2, 0.5 * uR ** 2)
        )

        shock_mask = (uL > uR)
        s = 0.5 * (uL + uR)
        flux = np.where(shock_mask & (s > 0), 0.5 * uL ** 2, flux)
        flux = np.where(shock_mask & (s <= 0), 0.5 * uR ** 2, flux)
        return flux

    def step(self, u, dt):
        u = np.asarray(u, dtype=float)
        if u.size != self.Nx:
            raise ValueError("u size mismatch.")


        cfl = dt / self.dx
        max_speed = np.max(np.abs(u))
        if max_speed > 0.0 and cfl * max_speed > 1.0:

            dt = 0.9 * self.dx / max_speed
            cfl = dt / self.dx


        uL = np.concatenate([[u[-1]], u])
        uR = np.concatenate([u, [u[0]]])


        F = self._godunov_flux(uL, uR)


        u_new = u - (dt / self.dx) * (F[1:] - F[:-1])


        if self.nu > 0.0:
            u_new += (self.nu * dt / self.dx ** 2) * (
                np.concatenate([u[1:], [u[0]]]) -
                2.0 * u +
                np.concatenate([[u[-1]], u[:-1]])
            )


        u_new[0] = 0.0
        u_new[-1] = 0.0

        return u_new

    def solve(self, u0, t_final, dt=None):
        u = np.asarray(u0, dtype=float)
        if u.size != self.Nx:
            raise ValueError("u0 size mismatch.")

        if dt is None:
            max_speed = max(np.max(np.abs(u)), 1.0)
            dt = 0.5 * self.dx / max_speed

        Nt = int(np.ceil(t_final / dt))
        dt = t_final / Nt

        U = np.zeros((Nt + 1, self.Nx), dtype=float)
        U[0, :] = u
        t_vec = np.linspace(0.0, t_final, Nt + 1)

        for n in range(Nt):
            u = self.step(u, dt)
            U[n + 1, :] = u

        return U, t_vec
