
import numpy as np


class NPZDEcosystem:

    def __init__(self, nx, nz, dx, dz, dt,
                 V_max=1.0, K_N=0.5, I_opt=50.0,
                 g_max=0.6, K_P=0.5, beta=0.75, gamma=0.3,
                 m_P=0.05, m_Z=0.05, g_zoo=0.1, r_D=0.05,
                 w_s=5.0, k_w=0.04, k_c=0.03,
                 kappa_bio=1.0e-5, I_0=200.0):
        if nx < 4 or nz < 4:
            raise ValueError("nx, nz >= 4")
        if dx <= 0 or dz <= 0 or dt <= 0:
            raise ValueError("空间步长与时间步长必须为正")

        self.nx = nx
        self.nz = nz
        self.dx = dx
        self.dz = dz
        self.dt = dt


        self.V_max = V_max
        self.K_N = K_N
        self.I_opt = I_opt
        self.g_max = g_max
        self.K_P = K_P
        self.beta = beta
        self.gamma = gamma
        self.m_P = m_P
        self.m_Z = m_Z
        self.g_zoo = g_zoo
        self.r_D = r_D
        self.w_s = w_s / 86400.0
        self.k_w = k_w
        self.k_c = k_c
        self.kappa_bio = kappa_bio
        self.I_0 = I_0


        self.N = np.ones((nx, nz)) * 5.0
        self.P = np.ones((nx, nz)) * 0.1
        self.Z = np.ones((nx, nz)) * 0.05
        self.D = np.ones((nx, nz)) * 0.05


        self.z_grid = np.linspace(0.0, (nz - 1) * dz, nz)


        self.N_old = None
        self.P_old = None
        self.Z_old = None
        self.D_old = None

    def light_profile(self, P_field):
        nx, nz = P_field.shape
        I = np.zeros((nx, nz))
        dz = self.dz


        for i in range(nx):
            cum_p = 0.0
            for k in range(nz):
                if k > 0:
                    cum_p += 0.5 * (P_field[i, k] + P_field[i, k - 1]) * dz
                atten = np.exp(-self.k_w * self.z_grid[k] - self.k_c * cum_p)
                I[i, k] = self.I_0 * atten
        return I

    def uptake_rate(self, N_field, P_field):


        raise NotImplementedError("Hole 1: 请实现 uptake_rate")

    def grazing_rate(self, P_field):
        g = self.g_max * P_field / (self.K_P + P_field)
        g = np.where(P_field >= 0.0, g, 0.0)
        return g / 86400.0

    def biological_tendency(self, N, P, Z, D):
        U = self.uptake_rate(N, P)
        G = self.grazing_rate(P)


        dNdt = -U * P + (self.gamma * self.g_zoo / 86400.0) * Z + (self.r_D / 86400.0) * D


        dPdt = U * P - (self.m_P / 86400.0) * P - G * Z


        dZdt = self.beta * G * Z - (self.m_Z / 86400.0) * Z - (self.g_zoo / 86400.0) * Z


        dDdt = (1.0 - self.beta) * G * Z + (self.m_P / 86400.0) * P + (self.m_Z / 86400.0) * Z - (self.r_D / 86400.0) * D


        dDdt_sinking = np.zeros_like(D)
        dDdt_sinking[:, 1:-1] = -self.w_s * (D[:, 2:] - D[:, :-2]) / (2 * self.dz)

        dDdt_sinking[:, 0] = -self.w_s * (D[:, 1] - D[:, 0]) / self.dz
        dDdt_sinking[:, -1] = -self.w_s * (D[:, -1] - D[:, -2]) / self.dz

        dDdt += dDdt_sinking

        return dNdt, dPdt, dZdt, dDdt

    def laplacian_bio(self, F):
        dx = self.dx
        dz = self.dz
        L = np.zeros_like(F)
        L[1:-1, 1:-1] = (
            (F[2:, 1:-1] - 2 * F[1:-1, 1:-1] + F[:-2, 1:-1]) / (dx ** 2) +
            (F[1:-1, 2:] - 2 * F[1:-1, 1:-1] + F[1:-1, :-2]) / (dz ** 2)
        )
        return L

    def advection_term(self, u, w, F):
        dx = self.dx
        dz = self.dz
        adv = np.zeros_like(F)

        adv[1:-1, 1:-1] = (
            u[1:-1, 1:-1] * (F[2:, 1:-1] - F[:-2, 1:-1]) / (2 * dx) +
            w[1:-1, 1:-1] * (F[1:-1, 2:] - F[1:-1, :-2]) / (2 * dz)
        )
        return adv

    def step(self, u, w):
        dt = self.dt

        bio_dt_max = 3600.0
        n_sub = max(1, int(np.ceil(dt / bio_dt_max)))
        dt_bio = dt / n_sub


        adv_N = self.advection_term(u, w, self.N)
        adv_P = self.advection_term(u, w, self.P)
        adv_Z = self.advection_term(u, w, self.Z)
        adv_D = self.advection_term(u, w, self.D)

        diff_N = self.kappa_bio * self.laplacian_bio(self.N)
        diff_P = self.kappa_bio * self.laplacian_bio(self.P)
        diff_Z = self.kappa_bio * self.laplacian_bio(self.Z)
        diff_D = self.kappa_bio * self.laplacian_bio(self.D)

        for _ in range(n_sub):
            bio_N, bio_P, bio_Z, bio_D = self.biological_tendency(
                self.N, self.P, self.Z, self.D)

            self.N += dt_bio * (-adv_N + diff_N + bio_N)
            self.P += dt_bio * (-adv_P + diff_P + bio_P)
            self.Z += dt_bio * (-adv_Z + diff_Z + bio_Z)
            self.D += dt_bio * (-adv_D + diff_D + bio_D)


            self.N = np.clip(self.N, 0.0, 20.0)
            self.P = np.clip(self.P, 0.0, 5.0)
            self.Z = np.clip(self.Z, 0.0, 3.0)
            self.D = np.clip(self.D, 0.0, 5.0)


            self.N = np.where(np.isfinite(self.N), self.N, 0.0)
            self.P = np.where(np.isfinite(self.P), self.P, 0.1)
            self.Z = np.where(np.isfinite(self.Z), self.Z, 0.05)
            self.D = np.where(np.isfinite(self.D), self.D, 0.05)


        self.N[0, :] = 5.0
        self.N[-1, :] = 2.0
        self.P[0, :] = 0.05
        self.P[-1, :] = 0.2
        self.Z[0, :] = 0.02
        self.Z[-1, :] = 0.08

    def total_nitrogen(self):
        return np.sum(self.N + self.P + self.Z + self.D) * self.dx * self.dz

    def primary_production(self):
        U = self.uptake_rate(self.N, self.P)
        return np.sum(U * self.P) * self.dx * self.dz
