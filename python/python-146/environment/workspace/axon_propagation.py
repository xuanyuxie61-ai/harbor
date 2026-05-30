
import numpy as np
import math
from spike_neuron import HHNeuron


class JacobiPolynomial:

    @staticmethod
    def evaluate(x, alpha, beta, N):
        if alpha <= -1 or beta <= -1:
            raise ValueError("alpha and beta must be > -1")
        x = np.atleast_1d(x)
        if N < 0:
            return np.zeros_like(x)



        gamma0 = (2.0 ** (alpha + beta + 1.0)) / (alpha + beta + 1.0) * \
                 math.gamma(alpha + 1.0) * math.gamma(beta + 1.0) / math.gamma(alpha + beta + 1.0)
        P0 = np.ones_like(x) / np.sqrt(gamma0)
        if N == 0:
            return P0

        gamma1 = (alpha + 1.0) * (beta + 1.0) / (alpha + beta + 3.0) * gamma0
        P1 = ((alpha + beta + 2.0) * x / 2.0 + (alpha - beta) / 2.0) / np.sqrt(gamma1)
        if N == 1:
            return P1

        aold = 2.0 / (2.0 + alpha + beta) * np.sqrt(
            (alpha + 1.0) * (beta + 1.0) / (alpha + beta + 3.0)
        )

        PL_prev2 = P0
        PL_prev1 = P1
        for i in range(1, N):
            h1 = 2.0 * i + alpha + beta
            anew = 2.0 / (h1 + 2.0) * np.sqrt(
                (i + 1.0) * (i + 1.0 + alpha + beta) * (i + 1.0 + alpha) * (i + 1.0 + beta)
                / (h1 + 1.0) / (h1 + 3.0)
            )
            bnew = - (alpha ** 2 - beta ** 2) / h1 / (h1 + 2.0)
            PL_curr = (1.0 / anew) * (
                -aold * PL_prev2 + (x - bnew) * PL_prev1
            )
            aold = anew
            PL_prev2 = PL_prev1
            PL_prev1 = PL_curr
        return PL_curr

    @staticmethod
    def gauss_lobatto_nodes(N):
        if N < 0:
            raise ValueError("N must be non-negative.")


        k = np.arange(N + 1)
        nodes = np.cos(np.pi * k / N)

        nodes = nodes[::-1]
        return nodes


class DG1DNeuralCable:

    def __init__(self, xL, xR, K, Np, dt, epsilon=0.01):
        self.xL = xL
        self.xR = xR
        self.K = K
        self.Np = Np
        self.dt = dt
        self.epsilon = epsilon


        self.x_nodes = JacobiPolynomial.gauss_lobatto_nodes(Np - 1)

        self.dx = (xR - xL) / K
        self.x = np.zeros((Np, K))
        for k in range(K):
            x_center = xL + (k + 0.5) * self.dx
            self.x[:, k] = x_center + 0.5 * self.dx * self.x_nodes


        self.V = np.zeros((Np, Np))
        for j in range(Np):
            self.V[:, j] = JacobiPolynomial.evaluate(self.x_nodes, 0.0, 0.0, j)


        self.Dr = self._compute_derivative_matrix()
        self.Dx = (2.0 / self.dx) * self.Dr


        self.M = np.linalg.inv(self.V @ self.V.T)


        self.tau_m = 2.0
        self.lambda_e = 1.0
        self.D_coeff = 0.5

    def _compute_derivative_matrix(self):
        Np = self.Np
        Dr = np.zeros((Np, Np))
        for i in range(Np):
            for j in range(Np):
                if i != j:


                    pass


        for j in range(Np):

            lj = np.zeros(Np)
            lj[j] = 1.0

            coeffs = np.linalg.solve(self.V, lj)

            dcoeffs = np.zeros(Np)
            for p in range(1, Np):
                dcoeffs[p - 1] = p * coeffs[p]

            for i in range(Np):
                Dr[i, j] = np.sum(dcoeffs * self.V[i, :])
        return Dr

    def local_lax_friedrichs_flux(self, u_left, u_right):
        C = 1.0
        flux = 0.5 * (u_left + u_right) + 0.5 * C * (u_left - u_right)
        return flux

    def rhs(self, u, I_ion):
        Np, K = u.shape
        rhsu = np.zeros_like(u)


        for k in range(K):

            ux = self.Dx @ u[:, k]
            uxx = self.Dx @ ux
            rhsu[:, k] += self.D_coeff * uxx


            rhsu[:, k] += -(1.0 / self.lambda_e) * ux


            rhsu[:, k] += (-u[:, k] / self.tau_m + I_ion[:, k])


        for k in range(K):

            if k == 0:
                u_left_boundary = u[-1, k]
            else:
                u_left_boundary = u[-1, k - 1]
            u_right_boundary = u[0, k]

            flux_left = self.local_lax_friedrichs_flux(u_left_boundary, u_right_boundary)

            if k == K - 1:
                u_right_next = u[0, k]
            else:
                u_right_next = u[0, k + 1]
            flux_right = self.local_lax_friedrichs_flux(u[-1, k], u_right_next)


            rhsu[0, k] += self.epsilon * (flux_left - u[0, k]) / self.dx
            rhsu[-1, k] += self.epsilon * (flux_right - u[-1, k]) / self.dx

        return rhsu

    def step_rk4(self, u, I_ion):
        dt = self.dt

        rhs1 = self.rhs(u, I_ion)

        rhs2 = self.rhs(u + 0.5 * dt * rhs1, I_ion)

        rhs3 = self.rhs(u + 0.5 * dt * rhs2, I_ion)

        rhs4 = self.rhs(u + dt * rhs3, I_ion)
        u_new = u + dt / 6.0 * (rhs1 + 2.0 * rhs2 + 2.0 * rhs3 + rhs4)
        return u_new

    def simulate(self, u0, T_final, I_ion_func=None):
        n_steps = int(np.ceil(T_final / self.dt))
        u = u0.copy()
        history = [u.copy()]
        for step in range(n_steps):
            t = step * self.dt
            if I_ion_func is not None:
                I_ion = I_ion_func(t, self.x)
            else:
                I_ion = np.zeros_like(u)
            u = self.step_rk4(u, I_ion)

            u = np.clip(u, -100.0, 100.0)
            if step % max(1, n_steps // 100) == 0:
                history.append(u.copy())
        return u, history


class MHDNeuralCoupling:

    MU_0 = 4.0 * np.pi * 1e-7
    E_CHARGE = 1.602e-19

    @staticmethod
    def ionic_current_density(V, ion_concentrations):
        sigma_e = 1.5
        E_field = -np.gradient(V)
        J = sigma_e * E_field
        return J

    @staticmethod
    def magnetic_field_from_current(J, y_coord, mu0=MU_0):
        B = mu0 * J * y_coord
        return B

    @staticmethod
    def lorentz_force_modulation(J, B, ion_mobility=5e3):
        F = np.abs(J * B)
        correction = 1.0 / (1.0 + ion_mobility * F)
        return np.clip(correction, 0.1, 1.0)

    def compute_effective_conductivity(self, V, y_coord=5e-5):
        J = self.ionic_current_density(V, None)
        B = self.magnetic_field_from_current(J, y_coord)
        correction = self.lorentz_force_modulation(J, B)
        return correction


def demo_axon_propagation():
    cable = DG1DNeuralCable(xL=0.0, xR=10.0, K=20, Np=4, dt=0.001, epsilon=0.05)
    u0 = np.zeros((cable.Np, cable.K))

    for k in range(cable.K):
        for i in range(cable.Np):
            x = cable.x[i, k]
            u0[i, k] = 20.0 * np.exp(-((x - 1.0) ** 2) / 0.5)
    u_final, history = cable.simulate(u0, T_final=5.0)
    return u_final, cable.x


def demo_mhd_coupling():
    mhd = MHDNeuralCoupling()
    x = np.linspace(0, 10, 100)
    V = 20.0 * np.exp(-((x - 5.0) ** 2) / 1.0)
    correction = mhd.compute_effective_conductivity(V, y_coord=1e-6)
    return x, V, correction
