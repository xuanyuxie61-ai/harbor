
import numpy as np
from numpy.polynomial.legendre import leggauss


class NonlinearAcousticsPhysics:

    def __init__(self, medium='water', f0=1e6, p0=1e5, geometry='planar'):
        self.medium = medium
        self.f0 = float(f0)
        self.p0 = float(p0)
        self.geometry = geometry


        if medium == 'water':
            self.c0 = 1500.0
            self.rho0 = 1000.0
            self.nu = 1.0e-6
            self.beta = 3.5
            self.gamma = 7.0
            self.B_tait = 3.046e8
        elif medium == 'air':
            self.c0 = 343.0
            self.rho0 = 1.21
            self.nu = 1.5e-5
            self.beta = 1.2
            self.gamma = 1.4
            self.B_tait = 1.013e5
        else:
            raise ValueError(f"Unknown medium: {medium}")


        self.omega0 = 2.0 * np.pi * self.f0
        self.k0 = self.omega0 / self.c0
        self.lambda_wave = self.c0 / self.f0
        self.wavelength = self.lambda_wave
        self.acoustic_impedance = self.rho0 * self.c0


        self.M0 = self.p0 / (self.rho0 * self.c0 ** 2)
        if self.M0 <= 0.0 or self.M0 > 0.5:
            raise ValueError(f"Mach number {self.M0} out of valid range (0, 0.5]")


        self.classical_absorption = self._classical_absorption()


        self.shock_formation_distance = self._compute_shock_formation_distance()


        self.Goldberg_number = self._compute_goldberg_number()





    def _compute_shock_formation_distance(self):
        x_s = 1.0 / (self.beta * self.k0 * self.M0)
        if x_s <= 0.0 or not np.isfinite(x_s):
            raise ValueError("Shock formation distance computed as non-positive or non-finite.")
        return x_s

    def _compute_goldberg_number(self):
        alpha = self.classical_absorption
        if alpha <= 0.0:
            return np.inf
        Ng = 1.0 / (alpha * self.shock_formation_distance)
        return Ng

    def _classical_absorption(self):
        alpha = self.nu * self.omega0 ** 2 / (self.c0 ** 2)
        if alpha < 0.0:
            raise ValueError("Classical absorption coefficient negative.")
        return alpha

    def tait_equation(self, p):
        p = np.asarray(p)

        p_safe = np.where(p <= -self.B_tait, -0.999 * self.B_tait, p)
        rho = self.rho0 * ((p_safe / self.B_tait) + 1.0) ** (1.0 / self.gamma)
        return rho

    def nonlinear_wave_speed(self, p):
        return self.c0 + self.beta * p / (self.rho0 * self.c0)

    def burgers_rhs(self, u, x, nu_eff):
        if u.size < 3:
            raise ValueError("Velocity array too small for spatial derivative.")
        if x.size != u.size:
            raise ValueError("x and u must have same size.")


        dx = np.diff(x)
        if np.any(dx <= 0.0):
            raise ValueError("x coordinates must be strictly increasing.")


        du_dx = np.zeros_like(u)

        du_dx[1:-1] = (u[2:] - u[:-2]) / (x[2:] - x[:-2])

        du_dx[0] = (u[1] - u[0]) / (x[1] - x[0])
        du_dx[-1] = (u[-1] - u[-2]) / (x[-1] - x[-2])


        d2u_dx2 = np.zeros_like(u)
        d2u_dx2[1:-1] = 2.0 * (
            (u[2:] - u[1:-1]) / (x[2:] - x[1:-1]) -
            (u[1:-1] - u[:-2]) / (x[1:-1] - x[:-2])
        ) / (x[2:] - x[:-2])
        d2u_dx2[0] = d2u_dx2[1]
        d2u_dx2[-1] = d2u_dx2[-2]

        rhs = -u * du_dx + nu_eff * d2u_dx2
        return rhs

    def kzk_rhs(self, p, r_grid, z, diffraction=True, absorption=True):
        p = np.asarray(p, dtype=float)
        r_grid = np.asarray(r_grid, dtype=float)
        if p.size != r_grid.size:
            raise ValueError("p and r_grid must have same size.")
        if r_grid[0] != 0.0:
            raise ValueError("r_grid must start at 0 for axisymmetric geometry.")
        if np.any(np.diff(r_grid) <= 0.0):
            raise ValueError("r_grid must be strictly increasing.")

        Nr = p.size
        dp_dz = np.zeros(Nr, dtype=float)


        if diffraction:
            dp_dr = np.zeros(Nr, dtype=float)
            dp_dr[1:-1] = (p[2:] - p[:-2]) / (r_grid[2:] - r_grid[:-2])

            if Nr >= 3:
                dp_dr[0] = 0.0
                dp_dr[-1] = (p[-1] - p[-2]) / (r_grid[-1] - r_grid[-2])

            d2p_dr2 = np.zeros(Nr, dtype=float)
            d2p_dr2[1:-1] = 2.0 * (
                (p[2:] - p[1:-1]) / (r_grid[2:] - r_grid[1:-1]) -
                (p[1:-1] - p[:-2]) / (r_grid[1:-1] - r_grid[:-2])
            ) / (r_grid[2:] - r_grid[:-2])

            if Nr >= 3:
                d2p_dr2[0] = 2.0 * (p[1] - p[0]) / (r_grid[1] - r_grid[0]) ** 2
                d2p_dr2[-1] = d2p_dr2[-2]


            laplacian_r = d2p_dr2.copy()
            if Nr > 1:

                r_safe = r_grid.copy()
                r_safe[0] = r_safe[1]
                with np.errstate(divide='ignore', invalid='ignore'):
                    laplacian_r += dp_dr / r_safe
                laplacian_r[0] = 2.0 * d2p_dr2[0]


            diffraction_coeff = 1.0 / (2.0 * self.k0)
            dp_dz += diffraction_coeff * laplacian_r


        if absorption:
            alpha = self.classical_absorption


            dp_dz -= alpha * p


        return dp_dz

    def entropy_production_rate(self, u, x):
        if u.size < 3 or x.size < 3:
            return 0.0
        T0 = 300.0
        du_dx = np.zeros_like(u)
        du_dx[1:-1] = (u[2:] - u[:-2]) / (x[2:] - x[:-2])

        entropy_rate = np.sum(du_dx ** 2) * np.mean(np.diff(x)) * self.rho0 * self.nu / T0
        return float(entropy_rate)

    def shock_mach_number(self, u_post, u_pre=0.0):
        delta_u = u_post - u_pre
        Ms = 1.0 + (self.beta / 2.0) * (delta_u / self.c0)
        return Ms

    def spectral_cascade_energy(self, u_hat, k_vec):
        u_hat = np.asarray(u_hat)
        k_vec = np.asarray(k_vec)
        if u_hat.shape != k_vec.shape:
            raise ValueError("u_hat and k_vec must have same shape.")
        E = 0.5 * np.abs(u_hat) ** 2
        return E

    def validate_physical_state(self, u, p):
        if np.any(~np.isfinite(u)) or np.any(~np.isfinite(p)):
            return False
        if np.any(np.abs(u) > 10.0 * self.c0):
            return False
        if np.any(p < -self.B_tait * 0.999):
            return False
        return True
