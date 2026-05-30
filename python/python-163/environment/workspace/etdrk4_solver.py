
import numpy as np


class ETDRK4Solver1D:

    def __init__(self, nx, L_domain, dt, kappa, advection_coeff=0.0, M=16):
        if nx % 2 != 0:
            raise ValueError("nx must be even for FFT-based solver.")
        self.nx = nx
        self.L_domain = float(L_domain)
        self.dt = float(dt)
        self.kappa = float(kappa)
        self.advection_coeff = float(advection_coeff)
        self.M = int(M)


        self.k = np.zeros(nx)
        self.k[:nx//2] = np.arange(0, nx//2)
        self.k[nx//2] = 0.0
        self.k[nx//2+1:] = np.arange(-nx//2 + 1, 0)
        self.k *= (2.0 * np.pi / L_domain)


        self.L = -kappa * self.k**2 - 1j * advection_coeff * self.k


        self._compute_etd_coefficients()

    def _compute_etd_coefficients(self):
        dt = self.dt
        L = self.L
        M = self.M

        E = np.exp(dt * L)
        E2 = np.exp(dt * L / 2.0)


        r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)


        LR = dt * L[:, np.newaxis] + r[np.newaxis, :]


        Q = dt * np.real(np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=1))
        f1 = dt * np.real(np.mean(
            (-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR**2)) / LR**3,
            axis=1))
        f2 = dt * np.real(np.mean(
            (2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR**3,
            axis=1))
        f3 = dt * np.real(np.mean(
            (-4.0 - 3.0 * LR - LR**2 + np.exp(LR) * (4.0 - LR)) / LR**3,
            axis=1))

        self.E = E
        self.E2 = E2
        self.Q = Q
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3


        self.N_prefactor = -0.5 * 1j * self.k

    def nonlinear_operator(self, v_hat):
        u = np.real(np.fft.ifft(v_hat))
        Nv = self.N_prefactor * np.fft.fft(u**2)
        return Nv

    def step(self, v_hat):
        Nv = self.nonlinear_operator(v_hat)
        a = self.E2 * v_hat + self.Q * Nv
        Na = self.nonlinear_operator(a)
        b = self.E2 * v_hat + self.Q * Na
        Nb = self.nonlinear_operator(b)
        c = self.E2 * a + self.Q * (2.0 * Nb - Nv)
        Nc = self.nonlinear_operator(c)

        v_hat_new = (self.E * v_hat
                     + self.f1 * Nv
                     + 2.0 * self.f2 * (Na + Nb)
                     + self.f3 * Nc)
        return v_hat_new

    def solve(self, u0, num_steps, save_interval=1):
        v_hat = np.fft.fft(u0)
        t_history = [0.0]
        u_history = [u0.copy()]

        for i in range(1, num_steps + 1):
            v_hat = self.step(v_hat)
            if i % save_interval == 0:
                u = np.real(np.fft.ifft(v_hat))
                t_history.append(i * self.dt)
                u_history.append(u.copy())

        return np.array(t_history), np.array(u_history)


class ETDRK4ThermalSolver:

    def __init__(self, nx, nz, Lx, Lz, dt, kappa, qx_flux, T_injection=323.15):
        self.nx = nx
        self.nz = nz
        self.Lx = Lx
        self.Lz = Lz
        self.dt = dt
        self.T_injection = float(T_injection)
        self.kappa = np.asarray(kappa, dtype=np.float64)
        if self.kappa.ndim == 0:
            self.kappa = np.full(nz, self.kappa)
        self.qx_flux = np.asarray(qx_flux, dtype=np.float64)
        self.dx = Lx / max(nx - 1, 1)
        self.dz = Lz / max(nz - 1, 1)


        dx, dz = self.dx, self.dz
        kappa_max = np.max(self.kappa)
        dt_diff_stable = 0.25 / (kappa_max * (1.0 / dx**2 + 1.0 / dz**2))
        if dt > dt_diff_stable:

            self.num_substeps = max(1, int(np.ceil(dt / dt_diff_stable)))
            self.dt_sub = dt / self.num_substeps
        else:
            self.num_substeps = 1
            self.dt_sub = dt


        qx_max = np.max(np.abs(self.qx_flux))
        if qx_max > 0:
            dt_adv_stable = 0.5 * dx / qx_max
            if self.dt_sub > dt_adv_stable:
                n_sub = int(np.ceil(self.dt_sub / dt_adv_stable))
                self.num_substeps *= n_sub
                self.dt_sub = dt / self.num_substeps

    def step(self, T):
        T = np.asarray(T, dtype=np.float64)
        if T.shape != (self.nx, self.nz):
            raise ValueError(f"T must have shape ({self.nx}, {self.nz}).")

        T_new = T.copy()
        dx, dz = self.dx, self.dz
        dt_sub = self.dt_sub
        nx, nz = self.nx, self.nz
        kappa = self.kappa
        qx = self.qx_flux

        for _ in range(self.num_substeps):
            T_old = T_new.copy()






            raise NotImplementedError("Diffusion-advection discretization is missing — scientific knowledge required.")

            T_new = T_old + dt_sub * (diff + adv)


            T_new[0, :] = self.T_injection
            T_new[-1, :] = T_new[-2, :]
            T_new[:, 0] = T_new[:, 1]
            T_new[:, -1] = T_new[:, -2]


            T_new = np.clip(T_new, 273.15, 1273.15)

        return T_new


def _solve_tridiagonal(lower, diag, upper, rhs):
    n = rhs.size
    c_prime = np.zeros(n)
    d_prime = np.zeros(n)

    c_prime[0] = upper[0] / diag[0]
    d_prime[0] = rhs[0] / diag[0]

    for i in range(1, n):
        denom = diag[i] - lower[i - 1] * c_prime[i - 1]
        if abs(denom) < 1.0e-30:
            denom = 1.0e-30
        c_prime[i] = upper[i] / denom if i < n - 1 else 0.0
        d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / denom

    x = np.zeros(n)
    x[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    return x
