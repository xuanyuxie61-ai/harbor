"""
spectral_ops.py
===============
Spectral differentiation operators and Fourier-based utilities.

The Kuramoto-Sivashinsky equation is solved most efficiently in Fourier space
using spectral methods.  This module provides:

  1. FFT-based spectral differentiation matrices
  2. Exponential Time Differencing (ETD) coefficient precomputation
  3. Wavenumber generation for periodic domains
  4. Spectral filtering for dealiasing

These operations underpin both the reference ETDRK4 solver and the spectral
accuracy checks for the PINN solution.

Mathematical background:
    For a periodic function u(x) on [0, L] with Fourier coefficients \hat{u}_k,
    the m-th derivative is:
        \partial_x^m u = ifft( (i*k)^m * fft(u) )
    where k = [0, 1, ..., N/2-1, 0, -N/2+1, ..., -1] * (2*pi/L).
"""

import numpy as np


def compute_wavenumbers(nx, L_domain):
    """
    Compute Fourier wavenumbers for a periodic domain [0, L].

    Parameters
    ----------
    nx : int
        Number of grid points (must be even).
    L_domain : float
        Domain length.

    Returns
    -------
    k : ndarray, shape (nx,)
        Wavenumbers.
    """
    if nx % 2 != 0:
        raise ValueError("nx must be even")
    k = np.concatenate([
        np.arange(0, nx // 2),
        np.array([0.0]),
        np.arange(-nx // 2 + 1, 0)
    ]) * (2.0 * np.pi / L_domain)
    return k


def spectral_derivative(u, k, order=1):
    """
    Compute the order-th spatial derivative of u via FFT.

    Parameters
    ----------
    u : ndarray, shape (nx,) or (nx, nt)
        Function values.
    k : ndarray, shape (nx,)
        Wavenumbers.
    order : int
        Derivative order (>= 0).

    Returns
    -------
    du : ndarray
        Derivative of same shape as u.
    """
    if order < 0:
        raise ValueError("order must be non-negative")
    if order == 0:
        return u.copy()

    factor = (1j * k) ** order
    if u.ndim == 1:
        u_hat = np.fft.fft(u)
        return np.real(np.fft.ifft(factor * u_hat))
    elif u.ndim == 2:
        du = np.zeros_like(u)
        for j in range(u.shape[1]):
            u_hat = np.fft.fft(u[:, j])
            du[:, j] = np.real(np.fft.ifft(factor * u_hat))
        return du
    else:
        raise ValueError("u must be 1D or 2D")


def etdrk4_coefficients(L_op, dt, M=16):
    """
    Precompute ETDRK4 scalar coefficients Q, f1, f2, f3.

    For the linear operator L_op (diagonal in Fourier space), the ETDRK4
    coefficients are computed via contour integrals around each eigenvalue:

        Q  = dt * mean( (exp(LR/2) - 1) / LR )
        f1 = dt * mean( (-4 - LR + exp(LR)*(4 - 3*LR + LR^2)) / LR^3 )
        f2 = dt * mean( (2 + LR + exp(LR)*(-2 + LR)) / LR^3 )
        f3 = dt * mean( (-4 - 3*LR - LR^2 + exp(LR)*(4 - LR)) / LR^3 )

    where LR = dt*L + r and r are roots of unity.

    Parameters
    ----------
    L_op : ndarray
        Linear operator eigenvalues (diagonal entries).
    dt : float
        Time step.
    M : int
        Number of roots of unity for contour integral.

    Returns
    -------
    E, E2, Q, f1, f2, f3 : ndarray
        ETDRK4 coefficients.
    """
    nx = len(L_op)
    E = np.exp(dt * L_op)
    E2 = np.exp(dt * L_op / 2.0)

    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = dt * L_op[:, np.newaxis] + r[np.newaxis, :]

    Q = dt * np.real(np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=1))
    f1 = dt * np.real(np.mean(
        (-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR ** 2)) / LR ** 3, axis=1))
    f2 = dt * np.real(np.mean(
        (2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR ** 3, axis=1))
    f3 = dt * np.real(np.mean(
        (-4.0 - 3.0 * LR - LR ** 2 + np.exp(LR) * (4.0 - LR)) / LR ** 3, axis=1))

    return E, E2, Q, f1, f2, f3


def dealias_2_3_rule(v_hat):
    """
    Apply the 2/3 dealiasing rule to Fourier coefficients.

    For quadratic nonlinearities, aliasing errors are eliminated by zeroing
    modes with |k| > (2/3) * k_max.
    """
    nx = len(v_hat)
    k_max = nx // 2
    cutoff = int(2.0 / 3.0 * k_max)
    v_filtered = v_hat.copy()
    v_filtered[cutoff:-cutoff] = 0.0
    return v_filtered


def compute_energy_spectrum(u, L_domain):
    """
    Compute the energy spectrum E(k) = 0.5 * |\hat{u}_k|^2.
    """
    nx = len(u)
    k = compute_wavenumbers(nx, L_domain)
    u_hat = np.fft.fft(u)
    E = 0.5 * np.abs(u_hat) ** 2
    return k, E


def kolmogorov_length_scale(u, L_domain):
    """
    Estimate the Kolmogorov dissipation length scale:
        \eta = (\nu^3 / \epsilon)^{1/4}

    For the KS equation, the effective viscosity is 1 and the dissipation rate
    is estimated from the enstrophy.
    """
    k = compute_wavenumbers(len(u), L_domain)
    u_x = spectral_derivative(u, k, order=1)
    u_xx = spectral_derivative(u, k, order=2)
    epsilon = np.mean(u_xx ** 2)
    eta = (1.0 / (epsilon + 1e-12)) ** 0.25
    return eta
