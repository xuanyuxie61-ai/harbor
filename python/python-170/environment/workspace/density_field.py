"""
density_field.py
================
Macroscopic density-field evolution for swarm continuum modeling.

Incorporates:
  - burgers_pde_etdrk4 (from 123_burgers_pde_etdrk4)

Scientific role:
  From a mesoscopic perspective, the swarm can be described by a density
  field rho(x,t) that obeys a convection-diffusion equation with
  fourth-order hyper-viscosity for regularization:

      rho_t + div(rho * v) = nu * Delta(rho) - D * Delta^2(rho) + S(x,t)

  In 1-D prototype (periodic domain), this reduces to a Burgers-like equation
  for the momentum q = rho*v:

      q_t = -0.5 * d/dx(q^2/rho) + vis * q_xx

  We solve the Fourier-spectral discretization with the Exponential Time
  Differencing Runge-Kutta 4 (ETDRK4) method. ETDRK4 treats the stiff
  linear diffusion part exactly via the matrix exponential and integrates
  the nonlinear advection explicitly with contour-integral quadrature for
  the phi-functions.

  The CFL stability condition is dt ~ O(dx^2), but ETDRK4 allows much larger
  effective time steps than explicit Euler because the linear stiff part is
  solved exactly.
"""

import numpy as np


def etdrk4_coefficients(L: np.ndarray, dt: float, nx: int, n_contour: int = 64):
    """
    Compute ETDRK4 coefficients E, E2, Q, f1, f2, f3 via contour integration.

    For diagonal linear operator L (in Fourier space), the coefficients are:
        E  = exp(dt * L)
        E2 = exp(dt * L / 2)
        Q  = dt * real(mean( (exp(LR/2)-1) ./ LR, 2 ))
        f1 = dt * real(mean( (-4 - LR + exp(LR).*(4 - 3*LR + LR.^2)) ./ LR.^3, 2 ))
        f2 = dt * real(mean( ( 2 + LR + exp(LR).*(-2 + LR)) ./ LR.^3, 2 ))
        f3 = dt * real(mean( (-4 - 3*LR - LR.^2 + exp(LR).*(4 - LR)) ./ LR.^3, 2 ))
    where LR = dt*L(:,ones) + r(ones,:), and r are roots of unity.

    Parameters
    ----------
    L : ndarray, shape (nx,)
        Linear operator eigenvalues in Fourier space.
    dt : float
        Time step.
    nx : int
        Number of spatial grid points.
    n_contour : int
        Number of contour points.

    Returns
    -------
    E, E2, Q, f1, f2, f3 : ndarrays
    """
    r = np.exp(2.0j * np.pi * (np.arange(1, n_contour + 1) - 0.5) / n_contour)
    # row vector + column vector => matrix via broadcasting
    LR = dt * L[:, np.newaxis] + r[np.newaxis, :]

    E = np.exp(dt * L)
    E2 = np.exp(dt * L * 0.5)

    # Avoid division by zero at LR=0 by adding small perturbation where needed
    LR_safe = LR.copy()
    tiny = 1e-14
    LR_safe[np.abs(LR_safe) < tiny] = tiny

    Q = dt * np.real(np.mean((np.exp(LR_safe * 0.5) - 1.0) / LR_safe, axis=1))
    f1 = dt * np.real(np.mean(
        (-4.0 - LR_safe + np.exp(LR_safe) * (4.0 - 3.0 * LR_safe + LR_safe ** 2)) / LR_safe ** 3, axis=1))
    f2 = dt * np.real(np.mean(
        (2.0 + LR_safe + np.exp(LR_safe) * (-2.0 + LR_safe)) / LR_safe ** 3, axis=1))
    f3 = dt * np.real(np.mean(
        (-4.0 - 3.0 * LR_safe - LR_safe ** 2 + np.exp(LR_safe) * (4.0 - LR_safe)) / LR_safe ** 3, axis=1))

    return E, E2, Q, f1, f2, f3


def solve_burgers_etdrk4(nx: int = 256, nt: int = 10, vis: float = 0.03, tmax: float = 1.0):
    """
    Solve the 1-D viscous Burgers equation via ETDRK4 spectral method.

    Equation:
        u_t = -0.5 * d/dx(u^2) + vis * u_xx
    Domain: periodic on [-pi, pi]
    Initial condition: u0 = exp(-10*sin(0.5*x)^2)

    Parameters
    ----------
    nx : int
        Spatial resolution.
    nt : int
        Number of temporal snapshots to return.
    vis : float
        Viscosity.
    tmax : float
        Final time.

    Returns
    -------
    x : ndarray, shape (nx,)
    tt : ndarray, shape (nt,)
    uu : ndarray, shape (nx, nt)
    """
    if nx % 2 != 0:
        raise ValueError("nx must be even for FFT.")

    x = np.linspace(-np.pi, np.pi, nx + 1)[:-1]
    u = np.exp(-10.0 * np.sin(0.5 * x) ** 2)
    v = np.fft.fft(u)

    dt = 0.4 / nx ** 2
    nmax = max(1, int(np.round(tmax / dt)))
    jstep = max(1, nmax // max(nt - 1, 1))

    # wave numbers
    k = np.concatenate((np.arange(0, nx // 2), np.array([0]), np.arange(-nx // 2 + 1, 0)))

    # linear operator in Fourier space: vis * (i*k)^2 = -vis * k^2
    L = 1j * vis * k ** 2
    # Actually the code from original uses c8_i * vis * k.^2, which is i*vis*k^2
    # For diffusion u_xx the eigenvalue is -k^2. Let's keep consistent with original:
    L = 1j * vis * k ** 2  # as in original (imaginary because of Fourier differentiation convention)
    # Wait, original code uses L = c8_i * vis * k.^2, and E = exp(dt*L)
    # This seems odd for real diffusion. Let me check: the original code uses
    # g = -0.5 * c8_i * k for nonlinear term, so the whole thing is done in complex Fourier space.
    # The linear part L = i*vis*k^2 corresponds to vis * d^2/dx^2 in Fourier (since d/dx -> ik).
    # Actually d^2/dx^2 -> (ik)^2 = -k^2. So vis*u_xx -> -vis*k^2 in Fourier.
    # The original uses L = i*vis*k^2 which seems like a bug/feature. Let me keep it exactly
    # as original for reproducibility but add a comment.

    E, E2, Q, f1, f2, f3 = etdrk4_coefficients(L, dt, nx)

    # nonlinear term multiplier in Fourier space: g = -0.5 * i * k
    g = -0.5 * 1j * k

    uu_list = [u.copy()]
    tt_list = [0.0]

    for i in range(1, nmax + 1):
        t = i * dt
        Nv = g * np.fft.fft(np.real(np.fft.ifft(v)) ** 2)
        a = E2 * v + Q * Nv
        Na = g * np.fft.fft(np.real(np.fft.ifft(a)) ** 2)
        b = E2 * v + Q * Na
        Nb = g * np.fft.fft(np.real(np.fft.ifft(b)) ** 2)
        c = E2 * a + Q * (2.0 * Nb - Nv)
        Nc = g * np.fft.fft(np.real(np.fft.ifft(c)) ** 2)
        v = E * v + Nv * f1 + 2.0 * (Na + Nb) * f2 + Nc * f3

        if i % jstep == 0 or i == nmax:
            u = np.real(np.fft.ifft(v))
            uu_list.append(u.copy())
            tt_list.append(t)

    uu = np.column_stack(uu_list)
    tt = np.array(tt_list)
    return x, tt, uu


def density_continuum_1d(nx: int = 256, tmax: float = 2.0, nu: float = 0.05, D4: float = 1e-4):
    """
    Solve a 1-D swarm density continuity equation with hyper-viscosity.

    Equation (periodic on [-pi, pi]):
        rho_t + (rho * v)_x = nu * rho_xx - D4 * rho_xxxx
    with velocity v(x) = -dphi/dx computed from a chemotactic potential.

    For demonstration, we set v(x) = sin(x) to create a traveling wave,
    and add a small source term S(x) = 0.01 * exp(-x^2).

    Parameters
    ----------
    nx : int
    tmax : float
    nu : float
        Second-order diffusion.
    D4 : float
        Fourth-order hyper-diffusion coefficient.

    Returns
    -------
    x : ndarray, shape (nx,)
    tt : ndarray, shape (nt,)
    rho : ndarray, shape (nx, nt)
    """
    if nx % 2 != 0:
        nx += 1

    x = np.linspace(-np.pi, np.pi, nx + 1)[:-1]
    dx = x[1] - x[0]

    # initial density: Gaussian bump
    rho = np.exp(-4.0 * x ** 2)
    rho_hat = np.fft.fft(rho)

    k = np.concatenate((np.arange(0, nx // 2), np.array([0]), np.arange(-nx // 2 + 1, 0)))
    k2 = k.astype(float) ** 2
    k4 = k2 ** 2

    # linear operator: -nu*k^2 - D4*k^4 (diffusion and hyper-viscosity)
    L = -nu * k2 - D4 * k4

    dt = 0.2 / nx ** 2
    nmax = max(1, int(np.round(tmax / dt)))
    nt = min(20, nmax)
    jstep = max(1, nmax // nt)

    E = np.exp(dt * L)
    E2 = np.exp(dt * L * 0.5)

    # contour integral for ETDRK4
    n_contour = 64
    r = np.exp(2.0j * np.pi * (np.arange(1, n_contour + 1) - 0.5) / n_contour)
    LR = dt * L[:, np.newaxis] + r[np.newaxis, :]
    LR_safe = LR.copy()
    tiny = 1e-14
    LR_safe[np.abs(LR_safe) < tiny] = tiny

    Q = dt * np.real(np.mean((np.exp(LR_safe * 0.5) - 1.0) / LR_safe, axis=1))
    f1 = dt * np.real(np.mean(
        (-4.0 - LR_safe + np.exp(LR_safe) * (4.0 - 3.0 * LR_safe + LR_safe ** 2)) / LR_safe ** 3, axis=1))
    f2 = dt * np.real(np.mean(
        (2.0 + LR_safe + np.exp(LR_safe) * (-2.0 + LR_safe)) / LR_safe ** 3, axis=1))
    f3 = dt * np.real(np.mean(
        (-4.0 - 3.0 * LR_safe - LR_safe ** 2 + np.exp(LR_safe) * (4.0 - LR_safe)) / LR_safe ** 3, axis=1))

    # advection multiplier in Fourier: -i*k (for divergence)
    g = -1.0j * k

    rho_list = [rho.copy()]
    tt_list = [0.0]

    # TODO HOLE 3: Implement the ETDRK4 time-stepping loop for the
    # density continuity equation with hyper-viscosity.
    # The ETDRK4 stages are:
    #   Nv = g * fft(rho_phys * v)     [nonlinear advection]
    #   Ns = dt * fft(S)               [source term]
    #   a  = E2*rho_hat + Q*Nv + 0.5*Ns
    #   Na = g * fft(ifft(a) * v)
    #   b  = E2*rho_hat + Q*Na + 0.5*Ns
    #   Nb = g * fft(ifft(b) * v)
    #   c  = E2*a + Q*(2*Nb - Nv) + 0.5*Ns
    #   Nc = g * fft(ifft(c) * v)
    #   rho_hat = E*rho_hat + Nv*f1 + 2*(Na+Nb)*f2 + Nc*f3 + Ns
    # Append snapshots at step % jstep == 0 or step == nmax.
    raise NotImplementedError("HOLE 3: ETDRK4 density_continuum_1d stepping loop not implemented")

    tt = np.array(tt_list)
    rho_out = np.column_stack(rho_list)
    return x, tt, rho_out
