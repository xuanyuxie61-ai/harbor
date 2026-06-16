"""
von_neumann_analysis.py
=======================

Von Neumann (Fourier-mode) stability analysis for the high-order FD +
WENO + SSP-RK3 scheme, extended with *Monte-Carlo sampling of random
perturbations* to estimate the probability of instability in the
presence of realistic noise.

The seed project 277_dice_simulation simulates the repeated tossing
and total-summing of many dice, producing a sum distribution that
converges to a Gaussian by the CLT.  We lift this idea: instead of
deterministic Fourier modes k, we sample *random* perturbation
spectra from a Gaussian ensemble and empirically estimate the
probability

    P(|g(k, dt)| > 1 + epsilon)

where g(k, dt) is the amplification factor of the scheme.  This gives
a *probabilistic stability region* that accounts for the stochastic
fluctuations present in turbulent astrophysical flows (McKee &
Ostriker 1977 ISM turbulence).

Key formulae
------------
For the advection equation  u_t + c u_x = 0  with c > 0 discretised
by the p-th order centred FD and forward Euler in time, the
amplification factor is

    g(k dx) = 1 - i (c dt / dx) * S_p(k dx)

where S_p is the symbol of the p-th order first-derivative operator:

    S_2(k dx) = sin(k dx)
    S_4(k dx) = (8 sin(k dx) - sin(2 k dx)) / 6
    S_6(k dx) = (45 sin(k dx) - 9 sin(2 k dx) + sin(3 k dx)) / 20

Stability requires |g| <= 1 for all k dx in [0, pi].

With SSP-RK3 the amplification factor of the full time integrator is

    g_RK3(z) = 1 + z + z^2/2 + z^3/6    where z = -i nu S_p(k dx),
                                          nu = c dt / dx.

The stability region of SSP-RK3 in the complex plane is bounded by

    |g_RK3(z)|^2 = (1 - y^2/2)^2 + (y - y^3/6)^2 <= 1

where z = x + i y.  Along the imaginary axis z = i y the boundary
is y_max ~ 2.51 (Shu 2003).

Monte-Carlo sampling
--------------------
Given N_trials random perturbation spectra (each a realisation of a
Gaussian random field with power spectrum P(k) ~ k^{-n}), we compute
the maximum amplification factor of the *non-linear* scheme
(including the WENO weights that depend on the data) and report

    P_unstable = #{trials with max |g| > 1 + eps} / N_trials.

This replaces the single deterministic amplification factor with a
probability distribution, in the spirit of the dice simulation.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple, List, Optional, Callable

from astro_constants import FD_ORDER_DEFAULT


# =====================================================================
#                 FD STENCIL SYMBOLS (Fourier analysis)
# =====================================================================

def fd_symbol_fd2(theta: np.ndarray) -> np.ndarray:
    """Symbol S_2(theta) of the 2nd-order centred first derivative."""
    return np.sin(theta)


def fd_symbol_fd4(theta: np.ndarray) -> np.ndarray:
    """Symbol S_4(theta) of the 4th-order centred first derivative."""
    return (8.0 * np.sin(theta) - np.sin(2.0 * theta)) / 6.0


def fd_symbol_fd6(theta: np.ndarray) -> np.ndarray:
    """Symbol S_6(theta) of the 6th-order centred first derivative."""
    return (45.0 * np.sin(theta) - 9.0 * np.sin(2.0 * theta)
            + np.sin(3.0 * theta)) / 20.0


def fd_symbol(theta: np.ndarray, order: int = FD_ORDER_DEFAULT) -> np.ndarray:
    if order == 2:
        return fd_symbol_fd2(theta)
    elif order == 4:
        return fd_symbol_fd4(theta)
    elif order == 6:
        return fd_symbol_fd6(theta)
    raise ValueError(f"unsupported order {order}")


# =====================================================================
#                   DETERMINISTIC AMPLIFICATION FACTORS
# =====================================================================

def amplification_forward_euler(nu: float, theta: np.ndarray,
                                 order: int = FD_ORDER_DEFAULT
                                 ) -> np.ndarray:
    """
    Amplification factor of forward Euler + p-th order centred FD
    for the linear advection equation  u_t + c u_x = 0:

        g(theta) = 1 - i nu S_p(theta)

    with nu = c dt / dx the CFL number.
    """
    S = fd_symbol(theta, order)
    return 1.0 - 1j * nu * S


def amplification_ssp_rk3(z: np.ndarray) -> np.ndarray:
    """
    Amplification factor of SSP-RK3 applied to u_t = L u:

        g(z) = 1 + z + z^2 / 2 + z^3 / 6

    (third-order Taylor polynomial of exp(z)).
    """
    return 1.0 + z + 0.5 * z * z + (1.0 / 6.0) * z * z * z


def amplification_advection_rk3(nu: float, theta: np.ndarray,
                                  order: int = FD_ORDER_DEFAULT
                                  ) -> np.ndarray:
    """
    Amplification factor of the full scheme
    SSP-RK3 + p-th order centred FD for u_t + c u_x = 0:

        z = -i nu S_p(theta)
        g = 1 + z + z^2/2 + z^3/6
    """
    S = fd_symbol(theta, order)
    z = -1j * nu * S
    return amplification_ssp_rk3(z)


# =====================================================================
#                DETERMINISTIC STABILITY REGION
# =====================================================================

def max_amplification_rk3(nu: float, order: int = FD_ORDER_DEFAULT,
                            n_theta: int = 2000) -> float:
    """
    Maximum |g(theta)| over theta in [0, pi] for the SSP-RK3 + FD_p
    scheme applied to u_t + c u_x = 0.
    """
    theta = np.linspace(0.0, math.pi, n_theta)
    g = amplification_advection_rk3(nu, theta, order)
    return float(np.max(np.abs(g)))


def stability_limit_rk3(order: int = FD_ORDER_DEFAULT,
                         n_scan: int = 200) -> float:
    """
    Find the maximum CFL number nu_max such that the scheme is
    stable (max |g| <= 1) for all theta.

    We do a bisection search on nu.
    """
    lo, hi = 0.0, 3.0
    for _ in range(n_scan):
        mid = 0.5 * (lo + hi)
        if max_amplification_rk3(mid, order) > 1.0 + 1.0e-12:
            hi = mid
        else:
            lo = mid
    return lo


# =====================================================================
#        MONTE-CARLO PERTURBATION SAMPLING (dice-simulation view)
# =====================================================================

def gaussian_random_spectrum_1d(n_cells: int, n_mode: float = 2.0,
                                 rng: Optional[np.random.Generator] = None
                                 ) -> np.ndarray:
    """
    Generate a realisation of a Gaussian random field on a periodic
    1-D grid with power spectrum  P(k) ~ |k|^{-n_mode}.

    The construction:

        u_hat(k) = sqrt(P(k)) * (xi_1(k) + i xi_2(k))

    with xi_1, xi_2 standard normal, then inverse FFT.

    (The "dice tossing" analogue: each realisation is a single
    "toss" of the turbulent spectrum.)
    """
    if rng is None:
        rng = np.random.default_rng(42)
    k = np.fft.fftfreq(n_cells, d=1.0 / n_cells)
    k[0] = 1.0  # avoid div-by-zero
    Pk = np.abs(k) ** (-n_mode)
    Pk[0] = 0.0
    xi1 = rng.standard_normal(n_cells)
    xi2 = rng.standard_normal(n_cells)
    u_hat = np.sqrt(Pk) * (xi1 + 1j * xi2)
    u = np.fft.ifft(u_hat).real
    u -= np.mean(u)
    return u


def mc_instability_probability(
    nu: float,
    order: int,
    n_cells: int = 64,
    n_mode: float = 2.0,
    n_trials: int = 200,
    epsilon: float = 1.0e-6,
    nonlinear_scheme: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Estimate the probability that a random turbulent perturbation
    grows under the scheme at CFL number nu.

    Two regimes:

      1. linear_scheme: we apply the linear amplification factor to
         each Fourier mode of the random spectrum and measure the
         maximum amplification across modes.

      2. nonlinear_scheme (optional): we apply the actual non-linear
         WENO + FD operator to the state vector and measure the
         ratio of post-step L2 norm to pre-step L2 norm.

    Returns (P_unstable, std_error) with std_error from binomial theory.
    """
    rng = np.random.default_rng(seed)
    theta = np.linspace(0.0, math.pi, 512)
    n_unstable = 0
    for _ in range(n_trials):
        u = gaussian_random_spectrum_1d(n_cells, n_mode, rng)
        if nonlinear_scheme is None:
            # linear amplification on spectrum
            u_hat = np.fft.fft(u)
            g_theta = amplification_advection_rk3(nu, theta, order)
            # project back
            u_new_hat = u_hat * np.interp(
                np.linspace(0.0, math.pi, n_cells),
                theta, g_theta
            )
            u_new = np.fft.ifft(u_new_hat).real
            ratio = (np.linalg.norm(u_new)
                     / max(np.linalg.norm(u), 1.0e-30))
        else:
            u_new = nonlinear_scheme(u)
            ratio = (np.linalg.norm(u_new)
                     / max(np.linalg.norm(u), 1.0e-30))
        if ratio > 1.0 + epsilon:
            n_unstable += 1
    p = n_unstable / n_trials
    err = math.sqrt(p * (1.0 - p) / n_trials) if n_trials > 0 else 0.0
    return (p, err)


# =====================================================================
#                  STABILITY REPORT GENERATOR
# =====================================================================

def stability_report(order: int = FD_ORDER_DEFAULT,
                      n_cfl: int = 30) -> str:
    """
    Print a stability report: scan CFL numbers, report max |g| and
    whether the scheme is stable.
    """
    nu_max = stability_limit_rk3(order)
    lines = [
        f"Von Neumann stability report for order-{order} FD + SSP-RK3:",
        f"  linear stability limit: nu_max = {nu_max:.5f}",
        "",
        "  nu        max|g|     status",
        "  ----      ------     ------",
    ]
    for nu in np.linspace(0.0, min(1.5 * nu_max, 2.5), n_cfl):
        gmax = max_amplification_rk3(nu, order)
        status = "STABLE" if gmax <= 1.0 + 1.0e-12 else "UNSTABLE"
        lines.append(f"  {nu:8.4f}  {gmax:10.6f}  {status}")
    return "\n".join(lines)
