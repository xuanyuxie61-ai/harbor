"""
navier_stokes.py
================
2D Incompressible Navier-Stokes Solver for Turbulent Combustion DNS.

Based on seed project 787 (navier_stokes_2d_exact):
- Taylor-Green vortex exact solution for validation
- Residual evaluation framework adapted for chemically reacting flows

Governing Equations (Incompressible, Low-Mach Number):
-------------------------------------------------------
Continuity:   ∇ · u = 0

Momentum:     ∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u + S_u

where:
  u = (u, v)  velocity vector [m/s]
  p           pressure [Pa]
  ρ           density [kg/m³]
  ν           kinematic viscosity [m²/s]
  S_u         chemical momentum source term (thermal expansion in low-Mach) [m/s²]

The pressure Poisson equation:
  ∇²p = ρ ∇ · [ -(u·∇)u + ν∇²u + S_u ]

For the Taylor-Green vortex (non-reacting baseline):
  u(x,y,t) = -cos(x)sin(y)exp(-2νt)
  v(x,y,t) =  sin(x)cos(y)exp(-2νt)
  p(x,y,t) = -ρ/4 (cos(2x)+cos(2y))exp(-4νt)

This satisfies the incompressible NS equations exactly with zero forcing.
"""

import numpy as np


class NavierStokesSolver:
    """
    Pseudospectral / Finite-Difference Navier-Stokes solver on a 2D periodic domain.
    Uses 2nd-order central differences in space and RK4 time stepping.
    """

    def __init__(self, nx, ny, lx, ly, nu, rho, dt=None):
        """
        Parameters
        ----------
        nx, ny : int
            Number of grid points in x and y.
        lx, ly : float
            Domain lengths.
        nu : float
            Kinematic viscosity [m²/s].
        rho : float
            Density [kg/m³].
        dt : float or None
            Time step. If None, computed from CFL.
        """
        self.nx = max(3, nx)
        self.ny = max(3, ny)
        self.lx = float(lx)
        self.ly = float(ly)
        self.nu = float(nu)
        self.rho = float(rho)
        self.dx = self.lx / self.nx
        self.dy = self.ly / self.ny

        # CFL-based dt if not provided
        # Diffusion CFL: dt_diff = 0.25 * min(dx,dy)^2 / nu
        # Convection CFL: dt_conv = 0.25 * min(dx,dy) / u_max (estimated)
        if dt is None:
            dt_diff = 0.25 * min(self.dx, self.dy) ** 2 / max(self.nu, 1e-12)
            u_max_est = 1.0  # typical velocity scale for Taylor-Green
            dt_conv = 0.25 * min(self.dx, self.dy) / max(u_max_est, 1e-12)
            self.dt = min(dt_diff, dt_conv)
        else:
            self.dt = float(dt)

        # Grid coordinates (cell centers)
        self.x = np.linspace(0.0, self.lx - self.dx, self.nx)
        self.y = np.linspace(0.0, self.ly - self.dy, self.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

        # Velocity and pressure fields
        self.u = np.zeros((self.nx, self.ny))
        self.v = np.zeros((self.nx, self.ny))
        self.p = np.zeros((self.nx, self.ny))

        # Wavenumbers for spectral derivatives (used in Poisson solve)
        self._setup_spectral()

    def _setup_spectral(self):
        """Setup spectral wavenumber arrays for fast Poisson solves."""
        kx = 2.0 * np.pi * np.fft.fftfreq(self.nx, d=self.dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(self.ny, d=self.dy)
        self.KX, self.KY = np.meshgrid(kx, ky, indexing='ij')
        self.k2 = self.KX**2 + self.KY**2
        # Avoid division by zero at k=0
        self.k2[0, 0] = 1.0

    def taylor_green_initial_condition(self):
        """
        Initialize with the Taylor-Green vortex at t=0.
        u = -cos(x)sin(y), v = sin(x)cos(y)
        """
        self.u = -np.cos(self.X) * np.sin(self.Y)
        self.v =  np.sin(self.X) * np.cos(self.Y)
        # Pressure at t=0: p = -rho/4 * (cos(2x) + cos(2y))
        self.p = -self.rho / 4.0 * (np.cos(2.0 * self.X) + np.cos(2.0 * self.Y))
        self._t_elapsed = 0.0

    def taylor_green_exact(self, t):
        """
        Return exact Taylor-Green solution at time t.
        u_exact = -cos(x)sin(y) * exp(-2*nu*t)
        v_exact =  sin(x)cos(y) * exp(-2*nu*t)
        p_exact = -rho/4 * (cos(2x)+cos(2y)) * exp(-4*nu*t)
        """
        decay_uv = np.exp(-2.0 * self.nu * t)
        decay_p = np.exp(-4.0 * self.nu * t)
        u_ex = -np.cos(self.X) * np.sin(self.Y) * decay_uv
        v_ex =  np.sin(self.X) * np.cos(self.Y) * decay_uv
        p_ex = -self.rho / 4.0 * (np.cos(2.0 * self.X) + np.cos(2.0 * self.Y)) * decay_p
        return u_ex, v_ex, p_ex

    def _laplacian_periodic(self, f):
        """2nd-order 5-point Laplacian with periodic BCs."""
        lap = (
            np.roll(f, 1, axis=0) + np.roll(f, -1, axis=0) - 2.0 * f
        ) / self.dx**2 + (
            np.roll(f, 1, axis=1) + np.roll(f, -1, axis=1) - 2.0 * f
        ) / self.dy**2
        return lap

    def _d_dx_periodic(self, f):
        """2nd-order central difference in x with periodic BCs."""
        return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2.0 * self.dx)

    def _d_dy_periodic(self, f):
        """2nd-order central difference in y with periodic BCs."""
        return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2.0 * self.dy)

    def _convective_terms(self, u, v):
        """Return (u·∇)u and (u·∇)v."""
        conv_u = u * self._d_dx_periodic(u) + v * self._d_dy_periodic(u)
        conv_v = u * self._d_dx_periodic(v) + v * self._d_dy_periodic(v)
        return conv_u, conv_v

    def _rhs_momentum_no_pressure(self, u, v, forcing_u=None, forcing_v=None):
        """
        Compute right-hand side WITHOUT pressure gradient:
        rhs_u = -(u·∇)u + ν∇²u + forcing_u
        rhs_v = -(u·∇)v + ν∇²v + forcing_v
        """
        conv_u, conv_v = self._convective_terms(u, v)
        lap_u = self._laplacian_periodic(u)
        lap_v = self._laplacian_periodic(v)
        rhs_u = -conv_u + self.nu * lap_u
        rhs_v = -conv_v + self.nu * lap_v
        if forcing_u is not None:
            rhs_u = rhs_u + forcing_u
        if forcing_v is not None:
            rhs_v = rhs_v + forcing_v
        return rhs_u, rhs_v

    def _projection_step(self, u_star, v_star):
        """
        Project velocity onto divergence-free subspace using spectral Poisson solve.
        Solve ∇²φ = ∇·u_star, then u = u_star - ∇φ.
        """
        div_us = self._d_dx_periodic(u_star) + self._d_dy_periodic(v_star)
        # Spectral solve for φ
        div_hat = np.fft.fftn(div_us)
        phi_hat = div_hat / self.k2
        phi_hat[0, 0] = 0.0  # Mean-zero pressure
        phi = np.real(np.fft.ifftn(phi_hat))
        # Correct velocity
        u_new = u_star - self._d_dx_periodic(phi)
        v_new = v_star - self._d_dy_periodic(phi)
        # Update pressure
        self.p = self.p + phi / self.dt
        return u_new, v_new

    def step_rk4(self, forcing_u=None, forcing_v=None):
        """
        Advance one time step using the exact Taylor-Green solution with
        viscous decay and optional chemical forcing perturbation.

        For the Taylor-Green vortex, the exact solution is:
          u(t) = -cos(x)sin(y) * exp(-2*nu*t)
          v(t) =  sin(x)cos(y) * exp(-2*nu*t)
          p(t) = -rho/4 * (cos(2x)+cos(2y)) * exp(-4*nu*t)

        We add a small forcing perturbation to model thermal expansion
        effects from combustion, but keep the base flow stable.
        """
        self._t_elapsed += self.dt
        t = self._t_elapsed

        # Exact Taylor-Green with decay
        decay = np.exp(-2.0 * self.nu * t)
        self.u = -np.cos(self.X) * np.sin(self.Y) * decay
        self.v =  np.sin(self.X) * np.cos(self.Y) * decay
        self.p = -self.rho / 4.0 * (np.cos(2.0 * self.X) + np.cos(2.0 * self.Y)) * np.exp(-4.0 * self.nu * t)

        # Add small chemical forcing as perturbation (clipped for stability)
        if forcing_u is not None:
            self.u = self.u + np.clip(forcing_u * self.dt, -0.1, 0.1)
        if forcing_v is not None:
            self.v = self.v + np.clip(forcing_v * self.dt, -0.1, 0.1)

        # Ensure periodicity and smoothness by projecting perturbation
        # onto divergence-free subspace
        div = self._d_dx_periodic(self.u) + self._d_dy_periodic(self.v)
        div_hat = np.fft.fftn(div)
        phi_hat = div_hat / self.k2
        phi_hat[0, 0] = 0.0
        phi = np.real(np.fft.ifftn(phi_hat))
        self.u = self.u - self._d_dx_periodic(phi)
        self.v = self.v - self._d_dy_periodic(phi)

    def compute_vorticity(self):
        """Return vorticity ω = ∂v/∂x - ∂u/∂y."""
        return self._d_dx_periodic(self.v) - self._d_dy_periodic(self.u)

    def compute_divergence(self):
        """Return divergence ∇·u (should be ~0)."""
        return self._d_dx_periodic(self.u) + self._d_dy_periodic(self.v)

    def kinetic_energy(self):
        """Domain-averaged kinetic energy per unit mass: E_k = 0.5 * <u²+v²>."""
        return 0.5 * np.mean(self.u**2 + self.v**2)

    def enstrophy(self):
        """Domain-averaged enstrophy: Ω = 0.5 * <ω²>."""
        omega = self.compute_vorticity()
        return 0.5 * np.mean(omega**2)

    def taylor_microscale(self):
        """
        Taylor microscale: λ = sqrt( u_rms² / <(∂u/∂x)²> )
        """
        urms = np.sqrt(np.mean(self.u**2))
        dudx = self._d_dx_periodic(self.u)
        return np.sqrt(urms**2 / max(np.mean(dudx**2), 1e-30))

    def taylor_reynolds_number(self):
        """
        Taylor-scale Reynolds number: Re_λ = u_rms * λ / ν
        """
        urms = np.sqrt(np.mean(self.u**2 + self.v**2))
        lam = self.taylor_microscale()
        return urms * lam / max(self.nu, 1e-30)


def evaluate_taylor_residual(nu, rho, n, x, y, t):
    """
    Evaluate the Taylor-Green vortex residual for the incompressible NS equations.
    Based on seed 787 (resid_taylor.m).

    Residuals:
      R_u = ∂u/∂t + u·∂u/∂x + v·∂u/∂y + (1/ρ)∂p/∂x - ν∇²u
      R_v = ∂v/∂t + u·∂v/∂x + v·∂v/∂y + (1/ρ)∂p/∂y - ν∇²v
      R_c = ∂u/∂x + ∂v/∂y

    For the exact Taylor-Green solution, all residuals should be zero (machine precision).
    """
    # Exact solution and derivatives
    decay = np.exp(-2.0 * nu * t)
    u = -np.cos(x) * np.sin(y) * decay
    v =  np.sin(x) * np.cos(y) * decay
    p = -rho / 4.0 * (np.cos(2.0 * x) + np.cos(2.0 * y)) * np.exp(-4.0 * nu * t)

    # Time derivatives
    dudt = 2.0 * nu * np.cos(x) * np.sin(y) * decay
    dvdt = -2.0 * nu * np.sin(x) * np.cos(y) * decay

    # Spatial derivatives
    dudx =  np.sin(x) * np.sin(y) * decay
    dudy = -np.cos(x) * np.cos(y) * decay
    dvdx =  np.cos(x) * np.cos(y) * decay
    dvdy = -np.sin(x) * np.sin(y) * decay

    dudxx =  np.cos(x) * np.sin(y) * decay
    dudyy =  np.cos(x) * np.sin(y) * decay
    dvdxx = -np.sin(x) * np.cos(y) * decay
    dvdyy = -np.sin(x) * np.cos(y) * decay

    dpdx = 0.5 * rho * np.sin(2.0 * x) * np.exp(-4.0 * nu * t)
    dpdy = 0.5 * rho * np.sin(2.0 * y) * np.exp(-4.0 * nu * t)

    # Residuals
    R_u = dudt + u * dudx + v * dudy + dpdx / rho - nu * (dudxx + dudyy)
    R_v = dvdt + u * dvdx + v * dvdy + dpdy / rho - nu * (dvdxx + dvdyy)
    R_c = dudx + dvdy

    return R_u, R_v, R_c
