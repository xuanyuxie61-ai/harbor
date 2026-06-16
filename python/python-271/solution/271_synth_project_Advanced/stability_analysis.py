# -*- coding: utf-8 -*-
"""
Von Neumann stability analysis of the high-order finite-difference
schemes used to resolve the QCP of the TFIM.

The QCP is detected by the divergence of the correlation length
xi ~ |lambda - lambda_c|^{-nu} and the susceptibility
chi_F ~ |lambda - lambda_c|^{-mu}.  In finite-difference language this
means the *symbol* of the discrete operator must be positive definite
away from lambda_c and must develop a soft mode as lambda -> lambda_c.

We study two related linear problems:

  (A) The *imaginary-time* diffusion equation
        d psi / d tau  =  - H_fd psi
      which is the path-integral kernel.  Stability requires the
      amplification factor |g(k, dtau)| <= 1 for all Fourier modes k.

  (B) The *real-time* Schrodinger equation
        i d psi / d t  =  H_fd psi
      which is unitary; the discrete scheme preserves norm iff
      |g(k, dt)| == 1 exactly.

For each FD stencil we compute the Fourier symbol, the maximum stable
time step (CFL), and the leading truncation-error coefficient.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
try:
    from . import constants as C
except ImportError:
    import constants as C
try:
    from .high_order_fd import central_fd_weights, fornberg_weights



except ImportError:
    from high_order_fd import central_fd_weights, fornberg_weights


# ---------------------------------------------------------------------------
# Fourier symbol of a stencil
# ---------------------------------------------------------------------------
def fourier_symbol(weights: np.ndarray, stencil_offset: np.ndarray,
                    ks: np.ndarray) -> np.ndarray:
    """Given a stencil w[i] applied at grid points x0 + stencil_offset[i]*dx,
    the Fourier symbol is

        sigma(k dx) = sum_i w[i] exp( i k dx * stencil_offset[i] )

    for w approximating the d-th derivative, the symbol divided by
    (i k dx)^d should tend to 1 as k dx -> 0.
    """
    ks = np.asarray(ks, dtype=complex)
    symbol = np.zeros_like(ks, dtype=complex)
    for w, s in zip(weights, stencil_offset):
        symbol = symbol + w * np.exp(1j * ks * float(s))
    return symbol


def symbol_derivative_2(order: int, n_points: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """Return (theta, sigma(theta)) for the central FD approximation of
    d^2/dx^2 using a stencil of total width ``order + 1``.
    ``theta`` is the normalised wave-number k dx in [0, pi].
    """
    w = central_fd_weights(order, 2)
    offset = np.arange(-order // 2, order // 2 + 1)
    theta = np.linspace(0.0, C.PI, n_points)
    sigma = np.zeros_like(theta, dtype=complex)
    for wi, si in zip(w, offset):
        sigma += wi * np.exp(1j * theta * si)
    # Symbol for the 2nd derivative is real and negative.
    return theta, sigma.real


def dispersion_error(order: int) -> float:
    """Leading truncation-error coefficient of the central FD
    approximation to d^2/dx^2 of order ``order``.

    We expand  sigma(theta) = - theta^2 * (1 + c_{order} theta^{order} + ...)
    and return c_{order}.  For the standard 2nd-order stencil
    (order=2) we have c_2 = -1/12.
    """
    w = central_fd_weights(order, 2)
    offset = np.arange(-order // 2, order // 2 + 1)
    # Expand symbol(theta) in Taylor series by evaluating at small theta.
    thetas = np.array([1.0e-3, 5.0e-4, 2.5e-4])
    vals = np.zeros(3, dtype=complex)
    for idx, th in enumerate(thetas):
        for wi, si in zip(w, offset):
            vals[idx] += wi * np.exp(1j * th * si)
    # sigma(theta) = - theta^2 - c theta^{order+2} + ...
    # so (sigma + theta^2) / theta^{order+2} -> -c.
    coeffs = [-(vals[i] + thetas[i] ** 2) / (thetas[i] ** (order + 2))
              for i in range(3)]
    # Average to suppress higher-order contamination
    return float(np.mean(coeffs).real)


# ---------------------------------------------------------------------------
# Von Neumann amplification factor
# ---------------------------------------------------------------------------
def amplification_factor_explicit(sigma: np.ndarray, dtau: float) -> np.ndarray:
    """For the explicit Euler step  psi^{n+1} = psi^n + dtau * H_fd psi^n
    the amplification factor is  g = 1 + dtau * sigma(theta).
    """
    return 1.0 + dtau * sigma


def amplification_factor_crank_nicolson(sigma: np.ndarray,
                                         dtau: float) -> np.ndarray:
    """Crank-Nicolson:  g = (1 + dtau/2 * sigma) / (1 - dtau/2 * sigma).
    For a real non-positive sigma this has |g| = 1 identically
    (unconditionally stable, norm-preserving to machine precision).
    """
    num = 1.0 + 0.5 * dtau * sigma
    den = 1.0 - 0.5 * dtau * sigma
    den = np.where(np.abs(den) < C.EPS_NUM, C.EPS_NUM, den)
    return num / den


def amplification_factor_leapfrog(sigma: np.ndarray,
                                   dt: float) -> np.ndarray:
    """Leapfrog (real-time Schrodinger):  g solves
    g^2 - 2 i dt sigma g - 1 = 0
    so g = i dt sigma +/- sqrt(1 - (dt sigma)^2).
    |g| == 1 iff |dt sigma| <= 1  (CFL condition).
    """
    disc = 1.0 - (dt * sigma) ** 2 + 0j
    # Branch cut: choose sqrt with positive real part.
    sq = np.sqrt(disc + 0j)
    g1 = 1j * dt * sigma + sq
    return g1


def max_stable_dt_explicit(sigma: np.ndarray) -> float:
    """For explicit Euler stability we need |1 + dt sigma| <= 1 for all
    theta.  Since sigma(theta) is real and non-positive, the tightest
    constraint is dt * max |sigma| <= 2, i.e. dt_max = 2 / |sigma_max|.
    """
    smin = float(np.min(sigma.real))   # most negative
    if smin >= 0.0:
        return np.inf
    return 2.0 / abs(smin)


def max_stable_dt_leapfrog(sigma: np.ndarray) -> float:
    """Leapfrog stability: dt * max |sigma| <= 1."""
    smin = float(np.min(sigma.real))
    if smin >= 0.0:
        return np.inf
    return 1.0 / abs(smin)


# ---------------------------------------------------------------------------
# Spectral gap of the FD Laplacian  (used in stability criterion for
# the imaginary-time TFIM flow)
# ---------------------------------------------------------------------------
def fd_laplacian_eigenvalues(L: int, order: int, dx: float = 1.0) -> np.ndarray:
    """Return the L eigenvalues of the central FD approximation to
    d^2/dx^2 with Neumann boundary conditions on a uniform grid of
    spacing dx.

    Neumann BCs are enforced by ghost-point reflection at both ends
    (matches the finite-element Neumann implementation in
    ``fem_neumann.py``-style modules of the seed 377 project).
    """
    if L < 2:
        raise ValueError("need L >= 2 for a meaningful spectrum")
    w = central_fd_weights(order, 2)
    half = order // 2
    # Build full matrix with ghost reflection
    A = np.zeros((L, L), dtype=float)
    for i in range(L):
        for k, s in enumerate(range(-half, half + 1)):
            j = i + s
            # Reflect at boundaries 0 and L-1
            if j < 0:
                j = -j
            elif j >= L:
                j = 2 * (L - 1) - j
            if 0 <= j < L:
                A[i, j] += w[k]
    A /= dx ** 2
    A = 0.5 * (A + A.T)
    ev = np.linalg.eigvalsh(A)
    return np.sort(ev)


# ---------------------------------------------------------------------------
# Composite stability diagnostic
# ---------------------------------------------------------------------------
class StabilityReport:
    """Container for a full stability characterisation of a stencil.

    Attributes
    ----------
    order : int
        Stencil formal order.
    c_trunc : float
        Leading truncation coefficient.
    sigma_max : float
        Most-negative value of the Fourier symbol.
    dt_expl : float
        Max stable explicit-Euler step.
    dt_leap : float
        Max stable leapfrog step.
    lebesgue : float
        Stencil Lebesgue constant.
    well_posed : bool
        True iff sigma_max < 0  (dissipative symbol).
    """
    def __init__(self, order: int, c_trunc: float, sigma_max: float,
                  dt_expl: float, dt_leap: float, lebesgue: float):
        self.order = order
        self.c_trunc = c_trunc
        self.sigma_max = sigma_max
        self.dt_expl = dt_expl
        self.dt_leap = dt_leap
        self.lebesgue = lebesgue
        self.well_posed = sigma_max < -C.EPS_NUM

    def __repr__(self) -> str:
        return (f"StabilityReport(order={self.order}, c_trunc={self.c_trunc:+.4e}, "
                f"sigma_max={self.sigma_max:+.4e}, "
                f"dt_expl={self.dt_expl:.4e}, dt_leap={self.dt_leap:.4e}, "
                f"lebesgue={self.lebesgue:.3e}, well_posed={self.well_posed})")


def analyse_stencil(order: int) -> StabilityReport:
    """Return a StabilityReport for the central FD stencil of total width
    order+1 approximating d^2/dx^2."""
    w = central_fd_weights(order, 2)
    theta, sigma = symbol_derivative_2(order)
    c_trunc = dispersion_error(order)
    smin = float(np.min(sigma))
    dt_expl = max_stable_dt_explicit(sigma)
    dt_leap = max_stable_dt_leapfrog(sigma)
    lebesgue = float(np.sum(np.abs(w)))
    return StabilityReport(order, c_trunc, smin, dt_expl, dt_leap, lebesgue)
