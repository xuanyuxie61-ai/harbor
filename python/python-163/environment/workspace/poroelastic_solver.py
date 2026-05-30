
import numpy as np


class PoroelasticSolver2D:

    def __init__(self, nx, nz, dx, dz, E, nu, alpha, beta, rho_b, g):
        self.nx = nx
        self.nz = nz
        self.dx = dx
        self.dz = dz
        self.E = E
        self.nu = nu
        self.alpha = alpha
        self.beta = beta
        self.rho_b = rho_b
        self.g = g


        self.lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        self.mu = E / (2.0 * (1.0 + nu))
        self.K_T = E / (3.0 * (1.0 - 2.0 * nu))

    def solve_displacement(self, p, T, T0, num_iterations=100, tol=1.0e-8,
                           fixed_bottom=True, fixed_sides=True):
        nx, nz = self.nx, self.nz
        dx, dz = self.dx, self.dz
        lam, mu = self.lam, self.mu
        alpha, beta, K_T = self.alpha, self.beta, self.K_T
        rho_b_g = self.rho_b * self.g

        u_x = np.zeros((nx, nz), dtype=np.float64)
        u_z = np.zeros((nx, nz), dtype=np.float64)


        dp_dx = np.zeros_like(p)
        dp_dz = np.zeros_like(p)
        dT_dx = np.zeros_like(T)
        dT_dz = np.zeros_like(T)

        dp_dx[1:-1, :] = (p[2:, :] - p[:-2, :]) / (2.0 * dx)
        dp_dz[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (2.0 * dz)
        dT_dx[1:-1, :] = (T[2:, :] - T[:-2, :]) / (2.0 * dx)
        dT_dz[:, 1:-1] = (T[:, 2:] - T[:, :-2]) / (2.0 * dz)

        fx = alpha * dp_dx + beta * K_T * dT_dx
        fz = alpha * dp_dz + beta * K_T * dT_dz - rho_b_g


        coeff = 2.0 * (lam + 2.0 * mu) / dx**2 + 2.0 * mu / dz**2
        coeff_z = 2.0 * mu / dx**2 + 2.0 * (lam + 2.0 * mu) / dz**2

        for it in range(num_iterations):
            u_x_old = u_x.copy()
            u_z_old = u_z.copy()

            for i in range(1, nx - 1):
                for k in range(1, nz - 1):

                    rhs_x = (fx[i, k]
                             + (lam + 2.0 * mu) * (u_x[i - 1, k] + u_x[i + 1, k]) / dx**2
                             + mu * (u_x[i, k - 1] + u_x[i, k + 1]) / dz**2
                             + (lam + mu) * (u_z[i + 1, k + 1] - u_z[i + 1, k - 1]
                                             - u_z[i - 1, k + 1] + u_z[i - 1, k - 1])
                             / (4.0 * dx * dz))
                    u_x[i, k] = rhs_x / coeff


                    rhs_z = (fz[i, k]
                             + mu * (u_z[i - 1, k] + u_z[i + 1, k]) / dx**2
                             + (lam + 2.0 * mu) * (u_z[i, k - 1] + u_z[i, k + 1]) / dz**2
                             + (lam + mu) * (u_x[i + 1, k + 1] - u_x[i - 1, k + 1]
                                             - u_x[i + 1, k - 1] + u_x[i - 1, k - 1])
                             / (4.0 * dx * dz))
                    u_z[i, k] = rhs_z / coeff_z


            if fixed_bottom:
                u_x[:, 0] = 0.0
                u_z[:, 0] = 0.0
            else:

                u_z[:, 0] = u_z[:, 1]
                u_x[:, 0] = u_x[:, 1]


            u_x[:, -1] = u_x[:, -2]
            u_z[:, -1] = u_z[:, -2]

            if fixed_sides:
                u_x[0, :] = 0.0
                u_x[-1, :] = 0.0
            else:
                u_x[0, :] = u_x[1, :]
                u_x[-1, :] = u_x[-2, :]


            u_z[0, :] = u_z[1, :]
            u_z[-1, :] = u_z[-2, :]


            err_x = np.max(np.abs(u_x - u_x_old))
            err_z = np.max(np.abs(u_z - u_z_old))
            if max(err_x, err_z) < tol:
                break

        return u_x, u_z

    def compute_strain_stress(self, u_x, u_z, p, T, T0):
        nx, nz = self.nx, self.nz
        dx, dz = self.dx, self.dz
        lam, mu = self.lam, self.mu
        alpha, beta, K_T = self.alpha, self.beta, self.K_T

        exx = np.zeros((nx, nz))
        ezz = np.zeros((nx, nz))
        exz = np.zeros((nx, nz))

        exx[1:-1, :] = (u_x[2:, :] - u_x[:-2, :]) / (2.0 * dx)
        ezz[:, 1:-1] = (u_z[:, 2:] - u_z[:, :-2]) / (2.0 * dz)
        exz[1:-1, 1:-1] = 0.5 * ((u_x[1:-1, 2:] - u_x[1:-1, :-2]) / (2.0 * dz)
                                 + (u_z[2:, 1:-1] - u_z[:-2, 1:-1]) / (2.0 * dx))

        delta_T = T - T0
        sxx = (lam + 2.0 * mu) * exx + lam * ezz - alpha * p - beta * K_T * delta_T
        szz = lam * exx + (lam + 2.0 * mu) * ezz - alpha * p - beta * K_T * delta_T
        sxz = 2.0 * mu * exz

        return {"exx": exx, "ezz": ezz, "exz": exz}, {"sxx": sxx, "szz": szz, "sxz": sxz}

    def von_mises_stress(self, stress):
        sxx = stress["sxx"]
        szz = stress["szz"]
        sxz = stress["sxz"]
        svm = np.sqrt(sxx**2 + szz**2 - sxx * szz + 3.0 * sxz**2)
        return svm
