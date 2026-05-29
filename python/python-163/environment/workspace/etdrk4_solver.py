"""
etdrk4_solver.py
================
Exponential Time Differencing Runge-Kutta 4 (ETD-RK4) solver for stiff
thermal advection-diffusion equations in the geothermal reservoir.

Incorporates algorithms from:
  - 630_kursiv_pde_etdrk4: ETD-RK4 time-stepping with contour integration

Mathematical formulation:
The thermal transport equation in Fourier space:

  \frac{\partial \hat{T}}{\partial t} = L \hat{T} + N(\hat{T})

where L is the linear (diffusion) operator and N is the nonlinear
(advection) operator. The ETD-RK4 scheme uses the integrating factor
approach:

  Step 1: N_1 = N(\hat{T}_n)
          A = e^{\Delta t L/2} \hat{T}_n + Q \cdot N_1

  Step 2: N_2 = N(A)
          B = e^{\Delta t L/2} \hat{T}_n + Q \cdot N_2

  Step 3: N_3 = N(B)
          C = e^{\Delta t L/2} A + Q \cdot (2 N_3 - N_1)

  Step 4: N_4 = N(C)

  Update: \hat{T}_{n+1} = e^{\Delta t L} \hat{T}_n
          + f_1 N_1 + 2 f_2 (N_2 + N_3) + f_3 N_4

The coefficients Q, f_1, f_2, f_3 are computed via contour integrals:

  Q = \Delta t \cdot \text{Re}\left[ \frac{1}{M} \sum_{m=1}^{M}
      \frac{e^{z_m/2} - 1}{z_m} \right]

  f_1 = \Delta t \cdot \text{Re}\left[ \frac{1}{M} \sum_{m=1}^{M}
        \frac{-4 - z_m + e^{z_m}(4 - 3 z_m + z_m^2)}{z_m^3} \right]

  f_2 = \Delta t \cdot \text{Re}\left[ \frac{1}{M} \sum_{m=1}^{M}
        \frac{2 + z_m + e^{z_m}(-2 + z_m)}{z_m^3} \right]

  f_3 = \Delta t \cdot \text{Re}\left[ \frac{1}{M} \sum_{m=1}^{M}
        \frac{-4 - 3 z_m - z_m^2 + e^{z_m}(4 - z_m)}{z_m^3} \right]

where z_m = \Delta t L + r_m and r_m are roots of unity on the complex plane.
"""

import numpy as np


class ETDRK4Solver1D:
    """
    ETD-RK4 solver for 1D periodic advection-diffusion equations using
    pseudo-spectral spatial discretization.
    """

    def __init__(self, nx, L_domain, dt, kappa, advection_coeff=0.0, M=16):
        """
        Parameters
        ----------
        nx : int
            Number of spatial grid points (must be even).
        L_domain : float
            Domain length.
        dt : float
            Time step.
        kappa : float
            Thermal diffusivity.
        advection_coeff : float
            Advection coefficient.
        M : int
            Number of contour points for ETD coefficients.
        """
        if nx % 2 != 0:
            raise ValueError("nx must be even for FFT-based solver.")
        self.nx = nx
        self.L_domain = float(L_domain)
        self.dt = float(dt)
        self.kappa = float(kappa)
        self.advection_coeff = float(advection_coeff)
        self.M = int(M)

        # Wave numbers
        self.k = np.zeros(nx)
        self.k[:nx//2] = np.arange(0, nx//2)
        self.k[nx//2] = 0.0
        self.k[nx//2+1:] = np.arange(-nx//2 + 1, 0)
        self.k *= (2.0 * np.pi / L_domain)

        # Linear operator in Fourier space: L = -kappa * k^2 - i * a * k
        self.L = -kappa * self.k**2 - 1j * advection_coeff * self.k

        # ETD coefficients via contour integration
        self._compute_etd_coefficients()

    def _compute_etd_coefficients(self):
        """Precompute E, E2, Q, f1, f2, f3 via contour integration."""
        dt = self.dt
        L = self.L
        M = self.M

        E = np.exp(dt * L)
        E2 = np.exp(dt * L / 2.0)

        # Roots of unity
        r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)

        # Row vector + column vector creates matrix
        LR = dt * L[:, np.newaxis] + r[np.newaxis, :]

        # Handle small LR to avoid division by zero
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

        # Nonlinear operator prefactor in Fourier space
        self.N_prefactor = -0.5 * 1j * self.k

    def nonlinear_operator(self, v_hat):
        """
        Compute nonlinear term in Fourier space.

        For advection-diffusion: N(v) = -0.5 * i * k * FFT(u^2)
        where u = IFFT(v).
        """
        u = np.real(np.fft.ifft(v_hat))
        Nv = self.N_prefactor * np.fft.fft(u**2)
        return Nv

    def step(self, v_hat):
        """
        Perform one ETD-RK4 time step.

        Parameters
        ----------
        v_hat : np.ndarray, shape (nx,)
            Fourier-space state.

        Returns
        -------
        v_hat_new : np.ndarray
            Updated Fourier-space state.
        """
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
        """
        Solve the PDE from initial condition u0.

        Parameters
        ----------
        u0 : np.ndarray, shape (nx,)
            Initial condition in physical space.
        num_steps : int
            Number of time steps.
        save_interval : int
            Save solution every `save_interval` steps.

        Returns
        -------
        t_history : np.ndarray
        u_history : np.ndarray, shape (nt, nx)
        """
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
    """
    2D thermal transport solver for a geothermal reservoir using
    operator splitting with explicit finite differences for diffusion
    and upwind advection, with proper boundary conditions.
    """

    def __init__(self, nx, nz, Lx, Lz, dt, kappa, qx_flux, T_injection=323.15):
        """
        Parameters
        ----------
        nx, nz : int
            Grid points in x and z.
        Lx, Lz : float
            Domain sizes.
        dt : float
            Time step.
        kappa : float or np.ndarray, shape (nz,)
            Thermal diffusivity (can vary with depth).
        qx_flux : np.ndarray, shape (nx, nz)
            Darcy flux in x-direction (m/s).
        T_injection : float
            Injection temperature for left Dirichlet BC.
        """
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

        # Stability check for explicit diffusion
        dx, dz = self.dx, self.dz
        kappa_max = np.max(self.kappa)
        dt_diff_stable = 0.25 / (kappa_max * (1.0 / dx**2 + 1.0 / dz**2))
        if dt > dt_diff_stable:
            # Limit effective dt for sub-stepping
            self.num_substeps = max(1, int(np.ceil(dt / dt_diff_stable)))
            self.dt_sub = dt / self.num_substeps
        else:
            self.num_substeps = 1
            self.dt_sub = dt

        # Stability check for advection
        qx_max = np.max(np.abs(self.qx_flux))
        if qx_max > 0:
            dt_adv_stable = 0.5 * dx / qx_max
            if self.dt_sub > dt_adv_stable:
                n_sub = int(np.ceil(self.dt_sub / dt_adv_stable))
                self.num_substeps *= n_sub
                self.dt_sub = dt / self.num_substeps

    def step(self, T):
        """
        Perform one time step using vectorized explicit finite differences.
        """
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

            # TODO: Implement diffusion and upwind advection terms.
            # Scientific knowledge required:
            #   1. Diffusion: \kappa \nabla^2 T (central finite differences)
            #   2. Advection: \mathbf{q}\cdot\nabla T (upwind scheme based on qx sign)
            # diff and adv must be np.ndarray of shape (nx, nz).
            raise NotImplementedError("Diffusion-advection discretization is missing — scientific knowledge required.")

            T_new = T_old + dt_sub * (diff + adv)

            # Boundary conditions
            T_new[0, :] = self.T_injection
            T_new[-1, :] = T_new[-2, :]
            T_new[:, 0] = T_new[:, 1]
            T_new[:, -1] = T_new[:, -2]

            # Physical bounds
            T_new = np.clip(T_new, 273.15, 1273.15)

        return T_new


def _solve_tridiagonal(lower, diag, upper, rhs):
    """Solve tridiagonal system using Thomas algorithm."""
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
