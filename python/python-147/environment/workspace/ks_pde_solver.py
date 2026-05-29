"""
ks_pde_solver.py
================
Reference spectral solver for the Kuramoto-Sivashinsky (KS) PDE using
Exponential Time Differencing Runge-Kutta 4 (ETDRK4) in Fourier space.

The KS equation arises in plasma physics, flame front propagation, and
two-phase flow modelling:

    u_t + u * u_x + u_xx + u_xxxx = 0,   x \in [0, L],  t > 0

with periodic boundary conditions u(t, 0) = u(t, L).

Here L = 32*pi, a canonical domain size that exhibits spatiotemporal chaos.

In Fourier space, letting v = fft(u) and wavenumbers k, the linear operator is

    L = k^2 - k^4

and the nonlinear term becomes

    N(v) = -0.5 * i * k * fft( real(ifft(v))^2 )

The ETDRK4 scheme avoids the severe time-step restriction of explicit methods
by treating the linear part exactly via matrix exponentials and the nonlinear
part with a 4-stage RK integrator.  The contour-integral based coefficients
Q, f1, f2, f3 are precomputed via the roots-of-unity approach of Kassam &
Trefethen (2005).

Reference implementation adapted from seed project 630_kursiv_pde_etdrk4.
"""

import numpy as np


def solve_ks_etdrk4(nx=128, tmax=50.0, dt=0.25, n_snapshots=51):
    """
    Solve the Kuramoto-Sivashinsky equation using ETDRK4.

    Parameters
    ----------
    nx : int
        Number of spatial grid points (must be even).
    tmax : float
        Final simulation time.
    dt : float
        Time step size.
    n_snapshots : int
        Number of temporal snapshots to return (including t=0).

    Returns
    -------
    x : ndarray, shape (nx,)
        Spatial grid on [0, 32*pi).
    t : ndarray, shape (n_snapshots,)
        Temporal snapshot values.
    u : ndarray, shape (nx, n_snapshots)
        Solution field u(t, x) at the requested snapshots.
    k : ndarray, shape (nx,)
        Fourier wavenumbers.
    L : ndarray, shape (nx,)
        Linear operator L = k^2 - k^4.
    """
    if not isinstance(nx, int) or nx < 4:
        raise ValueError("nx must be an integer >= 4")
    if nx % 2 != 0:
        raise ValueError("nx must be even for FFT")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if tmax <= 0:
        raise ValueError("tmax must be positive")

    L_domain = 32.0 * np.pi
    x = L_domain * np.arange(nx) / nx

    # Initial condition: u(0,x) = cos(x/16) * (1 + sin(x/16))
    u = np.cos(x / 16.0) * (1.0 + np.sin(x / 16.0))
    v = np.fft.fft(u)

    # Wavenumbers: [0, 1, ..., nx/2-1, 0, -nx/2+1, ..., -1] / 16
    k = np.concatenate([
        np.arange(0, nx // 2),
        np.array([0.0]),
        np.arange(-nx // 2 + 1, 0)
    ]) / 16.0

    # Linear operator in Fourier space: L = k^2 - k^4
    L_op = k ** 2 - k ** 4

    # Exponential integrators
    E = np.exp(dt * L_op)
    E2 = np.exp(dt * L_op / 2.0)

    # Precompute ETDRK4 coefficients via roots of unity (Kassam-Trefethen)
    M = 16
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)

    # Row-vector + column-vector broadcasting creates matrix LR of size (nx, M)
    LR = dt * L_op[:, np.newaxis] + r[np.newaxis, :]

    # Avoid division by zero at LR=0 by the contour integral formulation
    Q = dt * np.real(np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=1))
    f1 = dt * np.real(np.mean(
        (-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR ** 2)) / LR ** 3, axis=1))
    f2 = dt * np.real(np.mean(
        (2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR ** 3, axis=1))
    f3 = dt * np.real(np.mean(
        (-4.0 - 3.0 * LR - LR ** 2 + np.exp(LR) * (4.0 - LR)) / LR ** 3, axis=1))

    # Nonlinear term prefactor in Fourier space: g = -0.5 * i * k
    g = -0.5j * k

    n_steps = int(np.round(tmax / dt))
    if n_steps < 1:
        n_steps = 1

    # Determine snapshot intervals
    snapshot_intervals = max(1, int(n_steps // (n_snapshots - 1)))
    actual_snapshots = min(n_snapshots, n_steps // snapshot_intervals + 1)

    u_storage = [u.copy()]
    t_storage = [0.0]

    # TODO (Hole 1): Implement the ETDRK4 time-stepping loop.
    # The KS equation in Fourier space is:
    #   v_t = L_op * v + N(v)
    # where N(v) = -0.5j * k * fft( real(ifft(v))^2 )
    # Use the precomputed coefficients E, E2, Q, f1, f2, f3.
    # For each step, compute the RK stages (Nv, Na, Nb, Nc) and update v.
    # Store snapshots at appropriate intervals.
    raise NotImplementedError("Hole 1: ETDRK4 time-stepping loop not implemented")

    # Ensure final snapshot is included
    if len(t_storage) < actual_snapshots:
        u = np.real(np.fft.ifft(v))
        u_storage.append(u.copy())
        t_storage.append(n_steps * dt)

    u_mat = np.column_stack(u_storage)
    t_vec = np.array(t_storage)

    return x, t_vec, u_mat, k, L_op


def ks_reference_residual(u, x, t, k):
    """
    Compute the PDE residual of a given field u(t,x) using spectral
    differentiation.

    For the KS equation:
        r = u_t + u * u_x + u_xx + u_xxxx

    We compute derivatives in Fourier space for spectral accuracy.

    Parameters
    ----------
    u : ndarray, shape (nx, nt)
        Solution field.
    x : ndarray, shape (nx,)
        Spatial grid.
    t : ndarray, shape (nt,)
        Temporal grid.
    k : ndarray, shape (nx,)
        Wavenumbers.

    Returns
    -------
    residual : ndarray, shape (nx, nt)
        PDE residual at each (x,t) point.
    """
    if u.ndim != 2:
        raise ValueError("u must be 2D array (nx, nt)")
    nx, nt = u.shape
    if len(x) != nx or len(k) != nx:
        raise ValueError("Dimension mismatch between u, x, and k")
    if nt < 2:
        raise ValueError("Need at least 2 time points for time derivative")

    # Time derivative via central differences (interior) and one-sided (boundaries)
    dt_vec = np.diff(t)
    if np.any(dt_vec <= 0):
        raise ValueError("t must be strictly increasing")

    u_t = np.zeros_like(u)
    # Forward difference for first column
    u_t[:, 0] = (u[:, 1] - u[:, 0]) / (t[1] - t[0])
    # Backward difference for last column
    u_t[:, -1] = (u[:, -1] - u[:, -2]) / (t[-1] - t[-2])
    # Central differences for interior
    for j in range(1, nt - 1):
        u_t[:, j] = (u[:, j + 1] - u[:, j - 1]) / (t[j + 1] - t[j - 1])

    # Spatial derivatives via FFT for spectral accuracy
    u_x = np.zeros_like(u)
    u_xx = np.zeros_like(u)
    u_xxxx = np.zeros_like(u)

    for j in range(nt):
        v = np.fft.fft(u[:, j])
        u_x[:, j] = np.real(np.fft.ifft(1j * k * v))
        u_xx[:, j] = np.real(np.fft.ifft((1j * k) ** 2 * v))
        u_xxxx[:, j] = np.real(np.fft.ifft((1j * k) ** 4 * v))

    # KS residual: u_t + u * u_x + u_xx + u_xxxx
    residual = u_t + u * u_x + u_xx + u_xxxx
    return residual
