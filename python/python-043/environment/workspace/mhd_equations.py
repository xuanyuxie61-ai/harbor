
import numpy as np
from utils import PHYSICAL_CONSTANTS, safe_div, clip_bounds


class CoreDynamoMHD:

    def __init__(self, nodes, r_icb=0.35, r_cmb=1.0,
                 eta=0.02, nu=1e-4, kappa=5e-5, alpha_g=1e-3,
                 Omega=1.0, Ra=1e6, alpha_effect=0.1):
        self.nodes = nodes
        self.n_nodes = len(nodes)
        self.r_icb = r_icb
        self.r_cmb = r_cmb
        self.eta = eta
        self.nu = nu
        self.kappa = kappa
        self.alpha_g = alpha_g
        self.Omega = Omega
        self.Ra = Ra
        self.alpha_effect = alpha_effect


        self.r = np.linalg.norm(nodes, axis=1)
        self.r_safe = np.where(self.r < 1e-15, 1e-15, self.r)
        self.theta = np.arccos(clip_bounds(nodes[:, 2] / self.r_safe, -1.0, 1.0))
        self.phi = np.arctan2(nodes[:, 1], nodes[:, 0])


        self.is_icb = np.abs(self.r - r_icb) < 0.02
        self.is_cmb = np.abs(self.r - r_cmb) < 0.02
        self.is_boundary = self.is_icb | self.is_cmb


        self.Ekman = nu / (2.0 * Omega * (r_cmb - r_icb) ** 2 + 1e-30)
        self.Rm_typical = 100.0


        self._build_radial_diff_matrices()

    def _build_radial_diff_matrices(self):

        r_unique, r_inv = np.unique(np.round(self.r, 8), return_inverse=True)
        n_shells = len(r_unique)


        Dr_shell = np.zeros((n_shells, n_shells))
        for i in range(1, n_shells - 1):
            dr_f = r_unique[i + 1] - r_unique[i]
            dr_b = r_unique[i] - r_unique[i - 1]
            if dr_f > 1e-15 and dr_b > 1e-15:
                Dr_shell[i, i + 1] = 1.0 / (dr_f + dr_b)
                Dr_shell[i, i - 1] = -1.0 / (dr_f + dr_b)

        if n_shells > 1:
            Dr_shell[0, 0] = -1.0 / (r_unique[1] - r_unique[0])
            Dr_shell[0, 1] = 1.0 / (r_unique[1] - r_unique[0])
            Dr_shell[-1, -1] = 1.0 / (r_unique[-1] - r_unique[-2])
            Dr_shell[-1, -2] = -1.0 / (r_unique[-1] - r_unique[-2])


        D2r_shell = np.zeros((n_shells, n_shells))
        for i in range(1, n_shells - 1):
            h = r_unique[i + 1] - r_unique[i - 1]
            hp = r_unique[i + 1] - r_unique[i]
            hm = r_unique[i] - r_unique[i - 1]
            if h > 1e-15 and hp > 1e-15 and hm > 1e-15:
                D2r_shell[i, i - 1] = 2.0 / (h * hm)
                D2r_shell[i, i] = -2.0 / (hp * hm)
                D2r_shell[i, i + 1] = 2.0 / (h * hp)
        D2r_shell[0, 0] = 1.0
        D2r_shell[-1, -1] = 1.0


        n = self.n_nodes
        self.Dr_orig = np.zeros((n, n))
        self.D2r_orig = np.zeros((n, n))
        for shell_i in range(n_shells):
            mask_i = (r_inv == shell_i)
            for shell_j in range(n_shells):
                mask_j = (r_inv == shell_j)
                if Dr_shell[shell_i, shell_j] != 0.0:
                    self.Dr_orig[np.ix_(mask_i, mask_j)] = Dr_shell[shell_i, shell_j] / np.sum(mask_j)
                if D2r_shell[shell_i, shell_j] != 0.0:
                    self.D2r_orig[np.ix_(mask_i, mask_j)] = D2r_shell[shell_i, shell_j] / np.sum(mask_j)

    def _compute_curl_A(self, A):
        B = np.zeros((self.n_nodes, 3))
        st = np.sin(self.theta)
        ct = np.cos(self.theta)
        sp = np.sin(self.phi)
        cp = np.cos(self.phi)

        dA_dr = self.Dr_orig @ A

        theta_idx = np.argsort(self.theta)
        inv_theta = np.argsort(theta_idx)
        theta_sorted = self.theta[theta_idx]
        A_sorted = A[theta_idx]
        dA_dtheta = np.zeros_like(A)
        for i in range(1, self.n_nodes - 1):
            dt = theta_sorted[i + 1] - theta_sorted[i - 1]
            if dt > 1e-15:
                dA_dtheta[inv_theta[i]] = (A_sorted[i + 1] - A_sorted[i - 1]) / dt

        sin_t_safe = np.where(st < 1e-15, 1e-15, st)
        B_r = 1.0 / (self.r_safe * sin_t_safe) * (st * A + dA_dtheta)
        B_theta = -dA_dr
        B_phi = self.alpha_effect * A / self.r_safe

        B[:, 0] = B_r * st * cp + B_theta * ct * cp - B_phi * sp
        B[:, 1] = B_r * st * sp + B_theta * ct * sp + B_phi * cp
        B[:, 2] = B_r * ct - B_theta * st

        return B

    def _compute_vorticity(self, u):
        omega = np.zeros((self.n_nodes, 3))
        for comp in range(3):
            du = self.Dr_orig @ u[:, comp]
            omega[:, (comp + 1) % 3] += du
            omega[:, (comp + 2) % 3] -= du
        return omega

    def _lorentz_force(self, B, J):
        mu0 = PHYSICAL_CONSTANTS["mu_0"]
        return np.cross(J, B) / mu0

    def _alpha_effect_term(self, omega_r):
        sin_t = np.sin(self.theta)
        radial_profile = (self.r - self.r_icb) * (self.r_cmb - self.r)
        radial_profile = np.where(radial_profile < 0, 0.0, radial_profile)
        sign_omega = np.sign(omega_r)
        sign_omega = np.where(sign_omega == 0, 1.0, sign_omega)
        return self.alpha_effect * (sin_t ** 2) * radial_profile * sign_omega

    def rhs(self, t, state):
        N = self.n_nodes
        A = state[0:N]
        u = np.column_stack((state[N:2*N], state[2*N:3*N], state[3*N:4*N]))
        T = state[4*N:5*N]


        B = self._compute_curl_A(A)


        mu0 = PHYSICAL_CONSTANTS["mu_0"]
        J = np.zeros_like(B)
        for comp in range(3):
            J[:, comp] = (self.Dr_orig @ B[:, comp]) / mu0


        omega = self._compute_vorticity(u)
        alpha = self._alpha_effect_term(omega[:, 0])


        u_cross_B = np.cross(u, B)
        dA_dt = u_cross_B[:, 2] + alpha * B[:, 2]


        dA_dr = self.Dr_orig @ A
        d2A_dr2 = self.D2r_orig @ A
        laplacian_A = d2A_dr2 + (2.0 / self.r_safe) * dA_dr
        dA_dt += self.eta * laplacian_A


        Omega_vec = np.array([0.0, 0.0, self.Omega])
        coriolis = -2.0 * np.cross(Omega_vec, u)
        F_L = self._lorentz_force(B, J)

        F_L = np.clip(F_L, -1e4, 1e4)
        F_L = np.nan_to_num(F_L, nan=0.0, posinf=1e4, neginf=-1e4)

        st = np.sin(self.theta)
        ct = np.cos(self.theta)
        sp = np.sin(self.phi)
        cp = np.cos(self.phi)
        e_r = np.column_stack((st * cp, st * sp, ct))
        buoyancy = self.Ra * T[:, None] * st[:, None] * e_r
        buoyancy = np.clip(buoyancy, -1e4, 1e4)

        viscous = np.zeros_like(u)
        for comp in range(3):
            d2u = self.D2r_orig @ u[:, comp]
            viscous[:, comp] = self.nu * d2u


        damping = -0.5 * u

        du_dt = coriolis + F_L + buoyancy + viscous + damping


        dT_dr = self.Dr_orig @ T
        advection = -u[:, 0] * dT_dr
        d2T = self.D2r_orig @ T
        diffusion = self.kappa * d2T
        Q_heat = 0.5 * np.exp(-self.r / 0.5)
        dT_dt = advection + diffusion + Q_heat


        dA_dt[self.is_boundary] = 0.0
        du_dt[self.is_boundary] = 0.0
        dT_dt[self.is_icb] = 0.0
        dT_dt[self.is_cmb] = -0.1 * T[self.is_cmb]


        rhs = np.concatenate([dA_dt, du_dt[:, 0], du_dt[:, 1], du_dt[:, 2], dT_dt])
        rhs = np.clip(rhs, -1e6, 1e6)
        rhs = np.nan_to_num(rhs, nan=0.0, posinf=1e6, neginf=-1e6)
        return rhs

    def get_dipole_moment(self, A):
        return np.sum(A * self.r * np.cos(self.theta)) / self.n_nodes

    def get_magnetic_energy(self, B):
        mu0 = PHYSICAL_CONSTANTS["mu_0"]
        return np.mean(np.sum(B ** 2, axis=1)) / (2.0 * mu0)

    def get_kinetic_energy(self, u):
        rho = PHYSICAL_CONSTANTS["rho_core"]
        return np.mean(np.sum(u ** 2, axis=1)) * rho / 2.0
