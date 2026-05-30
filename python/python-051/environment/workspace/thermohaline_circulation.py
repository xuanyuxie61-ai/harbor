
import numpy as np
from matrix_solvers import bicg_solve, create_poisson_stencil


class ThermohalineCirculation:

    def __init__(self, nx=64, nz=32, Lx=5.0e6, Lz=4.0e3,
                 rho0=1027.0, alpha=1.7e-4, beta=7.6e-4,
                 nu=10.0, kappa_T=1.0e-4, kappa_S=1.0e-4,
                 g=9.81, dt=86400.0):
        if nx < 4 or nz < 4:
            raise ValueError("网格数 nx, nz 必须 >= 4")
        if Lx <= 0 or Lz <= 0:
            raise ValueError("域尺寸必须为正")
        if dt <= 0:
            raise ValueError("时间步长必须为正")

        self.nx = nx
        self.nz = nz
        self.Lx = Lx
        self.Lz = Lz
        self.dx = Lx / (nx - 1)
        self.dz = Lz / (nz - 1)
        self.rho0 = rho0
        self.alpha = alpha
        self.beta = beta
        self.nu = nu
        self.kappa_T = kappa_T
        self.kappa_S = kappa_S
        self.g = g
        self.dt = dt


        self.psi = np.zeros((nx, nz))
        self.omega = np.zeros((nx, nz))
        self.T = np.zeros((nx, nz))
        self.S = np.zeros((nx, nz))


        self.poisson_A = create_poisson_stencil(nx, nz, self.dx, self.dz)


        self.T_ref = 0.0
        self.S_ref = 0.0


        self.omega_old = None
        self.T_old = None
        self.S_old = None

    def jacobian(self, A, B):
        nx, nz = A.shape
        if B.shape != A.shape:
            raise ValueError("A, B 形状必须相同")

        J = np.zeros_like(A)
        dx = self.dx
        dz = self.dz


        Ax = np.zeros_like(A)
        Az = np.zeros_like(A)
        Bx = np.zeros_like(B)
        Bz = np.zeros_like(B)

        Ax[1:-1, :] = (A[2:, :] - A[:-2, :]) / (2 * dx)
        Az[:, 1:-1] = (A[:, 2:] - A[:, :-2]) / (2 * dz)
        Bx[1:-1, :] = (B[2:, :] - B[:-2, :]) / (2 * dx)
        Bz[:, 1:-1] = (B[:, 2:] - B[:, :-2]) / (2 * dz)

        J = Ax * Bz - Az * Bx


        J[0, :] = 0.0
        J[-1, :] = 0.0
        J[:, 0] = 0.0
        J[:, -1] = 0.0
        return J

    def laplacian(self, F):
        nx, nz = F.shape
        dx = self.dx
        dz = self.dz
        L = np.zeros_like(F)

        L[1:-1, 1:-1] = (
            (F[2:, 1:-1] - 2 * F[1:-1, 1:-1] + F[:-2, 1:-1]) / (dx ** 2) +
            (F[1:-1, 2:] - 2 * F[1:-1, 1:-1] + F[1:-1, :-2]) / (dz ** 2)
        )
        return L

    def density_anomaly(self):
        return self.rho0 * (-self.alpha * self.T + self.beta * self.S)

    def baroclinic_term(self):


        raise NotImplementedError("Hole 4: 请实现 baroclinic_term")

    def apply_boundary_conditions(self):
        nx, nz = self.nx, self.nz


        self.psi[0, :] = 0.0
        self.psi[-1, :] = 0.0
        self.psi[:, 0] = 0.0
        self.psi[:, -1] = 0.0



        self.omega[0, :] = -(2.0 * self.psi[1, :] - 0.5 * self.psi[2, :]) / (self.dx ** 2)
        self.omega[-1, :] = -(2.0 * self.psi[-2, :] - 0.5 * self.psi[-3, :]) / (self.dx ** 2)
        self.omega[:, 0] = -(2.0 * self.psi[:, 1] - 0.5 * self.psi[:, 2]) / (self.dz ** 2)
        self.omega[:, -1] = -(2.0 * self.psi[:, -2] - 0.5 * self.psi[:, -3]) / (self.dz ** 2)


        relax_rate = 1.0 / (10.0 * 86400.0)

        self.T[0, :] = self.T_ref - 2.0
        self.S[0, :] = self.S_ref + 0.5

        self.T[-1, :] = self.T_ref + 2.0
        self.S[-1, :] = self.S_ref - 0.5


    def solve_streamfunction(self):
        rhs = -self.omega.flatten()
        psi_flat = bicg_solve(self.poisson_A, rhs, tol=1e-8, max_iter=500)


        if not np.all(np.isfinite(psi_flat)):

            try:
                psi_flat = np.linalg.solve(self.poisson_A, rhs)
            except np.linalg.LinAlgError:
                psi_flat = np.zeros_like(rhs)

        self.psi = psi_flat.reshape((self.nx, self.nz))

        self.psi[0, :] = 0.0
        self.psi[-1, :] = 0.0
        self.psi[:, 0] = 0.0
        self.psi[:, -1] = 0.0

    def step(self, forcing_T=None, forcing_S=None):
        nx, nz = self.nx, self.nz
        dt = self.dt


        self.apply_boundary_conditions()


        self.solve_streamfunction()


        J_omega = self.jacobian(self.psi, self.omega)
        J_T = self.jacobian(self.psi, self.T)
        J_S = self.jacobian(self.psi, self.S)

        L_omega = self.laplacian(self.omega)
        L_T = self.laplacian(self.T)
        L_S = self.laplacian(self.S)

        baroclinic = self.baroclinic_term()


        if self.omega_old is None:

            omega_rhs = -J_omega + self.nu * L_omega + baroclinic
            T_rhs = -J_T + self.kappa_T * L_T
            S_rhs = -J_S + self.kappa_S * L_S
        else:
            omega_rhs = -1.5 * J_omega + 0.5 * self.omega_old
            omega_rhs += self.nu * L_omega + baroclinic
            T_rhs = -1.5 * J_T + 0.5 * self.T_old
            T_rhs += self.kappa_T * L_T
            S_rhs = -1.5 * J_S + 0.5 * self.S_old
            S_rhs += self.kappa_S * L_S

        if forcing_T is not None:
            T_rhs += forcing_T
        if forcing_S is not None:
            S_rhs += forcing_S


        self.omega_old = J_omega.copy()
        self.T_old = J_T.copy()
        self.S_old = J_S.copy()

        self.omega += dt * omega_rhs
        self.T += dt * T_rhs
        self.S += dt * S_rhs


        self.omega = np.clip(self.omega, -1e-3, 1e-3)
        self.T = np.clip(self.T, -10.0, 10.0)
        self.S = np.clip(self.S, -5.0, 5.0)


        self.omega = np.where(np.isfinite(self.omega), self.omega, 0.0)
        self.T = np.where(np.isfinite(self.T), self.T, 0.0)
        self.S = np.where(np.isfinite(self.S), self.S, 0.0)


        self.apply_boundary_conditions()

    def get_velocity(self):
        u = np.zeros_like(self.psi)
        w = np.zeros_like(self.psi)
        dz = self.dz
        dx = self.dx

        u[:, 1:-1] = -(self.psi[:, 2:] - self.psi[:, :-2]) / (2 * dz)
        w[1:-1, :] = (self.psi[2:, :] - self.psi[:-2, :]) / (2 * dx)
        return u, w

    def get_overturning_streamfunction(self):
        return self.psi.copy()
