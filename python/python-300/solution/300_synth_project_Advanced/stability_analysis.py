"""
stability_analysis.py
=====================
Von Neumann and matrix spectral stability analysis of the high-order
compact finite-difference discrete-ordinates transport operator.

The stability of the Pade (1,4,1) compact scheme for the steady SN
equation is governed by the amplification matrix  G  that relates the
angular flux at iteration k+1 to that at iteration k:

    psi^{k+1} = G psi^k + b

Convergence of source iteration requires  rho(G) < 1  where rho is the
spectral radius.  For the one-group slab problem with isotropic scattering
the amplification factor for Fourier mode exp(i k x) is

    g(k) = c / (1 + (mu k h)^2 / 12)      (modified wavenumber)

where c = Sigma_s / Sigma_t is the scattering ratio.  The scheme is
unconditionally stable for c < 1 (subcritical).

We also compute the *eigenmodes* of the discrete Laplacian on the blanket
mesh (analogous to the Chladni figures of a vibrating plate) to identify
the spatial structures that are most weakly damped.

Adapted from seed project:
    * 172_chladni_figures -> eigenmode analysis via sparse Laplacian
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple

import physics_constants as pc


# ---------------------------------------------------------------------------
# Von Neumann amplification factor
# ---------------------------------------------------------------------------
def von_neumann_factor(
    scattering_ratio: float,
    mu: float,
    sigma_t: float,
    dx: float,
    n_modes: int = 64,
) -> List[Tuple[float, complex]]:
    """Return the amplification factor g(k) for n_modes Fourier modes.

    For the compact Padé scheme the modified wavenumber is

        k_eff h = (3/2) sin(k h) / (1 + 0.5 cos(k h))

    and the amplification factor for one source-iteration step is

        g(k) = c * exp(-i k_eff h mu / sigma_t dx)
             / (1 + k_eff^2 mu^2 / (12 sigma_t^2))

    Parameters
    ----------
    scattering_ratio : c = Sigma_s / Sigma_t in [0, 1).
    mu : direction cosine of the SN ordinate.
    sigma_t : total cross section (cm^{-1}).
    dx : cell size (cm).
    n_modes : number of Fourier modes to sample.

    Returns
    -------
    List of (k_h, g) pairs where k_h = k * dx in [0, pi].
    """
    if not 0.0 <= scattering_ratio < 1.0:
        raise ValueError("scattering ratio must be in [0, 1)")
    if sigma_t <= 0.0 or dx <= 0.0:
        raise ValueError("sigma_t and dx must be positive")
    result: List[Tuple[float, complex]] = []
    for j in range(n_modes + 1):
        kh = math.pi * j / n_modes
        # Pade modified wavenumber
        sin_kh = math.sin(kh)
        cos_kh = math.cos(kh)
        denom = 1.0 + 0.5 * cos_kh
        if abs(denom) < pc.EPS_NUMERICAL:
            denom = pc.EPS_NUMERICAL
        k_eff_h = 1.5 * sin_kh / denom
        # amplification factor
        alpha = mu / (sigma_t * dx)
        num = scattering_ratio * complex(math.cos(k_eff_h * alpha),
                                          -math.sin(k_eff_h * alpha))
        den = 1.0 + (k_eff_h * alpha) ** 2 / 12.0
        g = num / den
        result.append((kh, g))
    return result


def spectral_radius(
    scattering_ratio: float,
    mu_values: List[float],
    sigma_t: float,
    dx: float,
    n_modes: int = 64,
) -> float:
    """Return max |g(k)| over all modes and directions."""
    rho = 0.0
    for mu in mu_values:
        for _, g in von_neumann_factor(scattering_ratio, mu, sigma_t, dx,
                                        n_modes):
            rho = max(rho, abs(g))
    return rho


# ---------------------------------------------------------------------------
# Matrix amplification operator (dense)
# ---------------------------------------------------------------------------
def build_amplification_matrix(
    n_cells: int,
    dx: float,
    mu: float,
    sigma_t: List[float],
    sigma_s: List[float],
) -> List[List[float]]:
    """Assemble the dense amplification matrix G for one SN direction.

    The one-group source-iteration operator for direction mu > 0 is

        psi^{k+1}_i = (1 / (alpha + Sigma_t_i))
                    * (alpha psi^{k+1}_{i-1} + Sigma_s_i phi^k)

    where alpha = mu / dx and phi^k = sum_m w_m psi^k_m.  For a single
    direction we drop the angular coupling and write

        G_ii = Sigma_s_i / (alpha + Sigma_t_i)
        G_{i,i-1} = alpha / (alpha + Sigma_t_i)

    The spectral radius of G determines convergence.
    """
    if n_cells < 2:
        raise ValueError("n_cells must be >= 2")
    G = [[0.0] * n_cells for _ in range(n_cells)]
    alpha = abs(mu) / dx
    for i in range(n_cells):
        st = max(sigma_t[i], pc.EPS_NUMERICAL)
        denom = alpha + st
        G[i][i] = sigma_s[i] / denom
        if i > 0:
            G[i][i - 1] = alpha / denom
    return G


def power_iteration(M: List[List[float]], n_iter: int = 200,
                     tol: float = 1.0e-10) -> Tuple[float, List[float]]:
    """Estimate the dominant eigenvalue of M by power iteration."""
    n = len(M)
    if n == 0:
        return 0.0, []
    v = [1.0 / math.sqrt(n)] * n
    lam = 0.0
    for _ in range(n_iter):
        w = [0.0] * n
        for i in range(n):
            for j in range(n):
                w[i] += M[i][j] * v[j]
        lam_new = sum(wi * vi for wi, vi in zip(w, v))
        norm = math.sqrt(sum(wi * wi for wi in w))
        if norm < pc.EPS_NUMERICAL:
            break
        v = [wi / norm for wi in w]
        if abs(lam_new - lam) < tol:
            lam = lam_new
            break
        lam = lam_new
    return lam, v


# ---------------------------------------------------------------------------
# Discrete Laplacian eigenmodes (Chladni-like)
# ---------------------------------------------------------------------------
def laplacian_eigenmodes_1d(
    n_cells: int, dx: float,
    n_modes: int = 5,
) -> List[Tuple[float, List[float]]]:
    """Compute the first n_modes eigenmodes of the 1-D discrete Laplacian.

    The negative Laplacian on a uniform mesh with Dirichlet BC has the
    well-known eigenvalues

        lambda_k = (4 / dx^2) sin^2(k pi / (2 (N+1))),   k = 1, ..., N

    and eigenvectors  v_k(i) = sin(i k pi / (N+1)).

    These are the "Chladni figures" of the blanket: the spatial patterns
    that are most weakly damped by diffusion.
    """
    if n_cells < 2:
        raise ValueError("n_cells must be >= 2")
    modes: List[Tuple[float, List[float]]] = []
    for k in range(1, n_modes + 1):
        lam = (4.0 / (dx * dx)) * math.sin(
            k * math.pi / (2.0 * (n_cells + 1))
        ) ** 2
        vec = [math.sin(i * k * math.pi / (n_cells + 1))
               for i in range(1, n_cells + 1)]
        modes.append((lam, vec))
    return modes


# ---------------------------------------------------------------------------
# Stability report
# ---------------------------------------------------------------------------
def full_stability_report(
    n_cells: int, dx: float, mu_values: List[float],
    sigma_t_mean: float, c_mean: float,
) -> Dict[str, float]:
    """Produce a compact stability report for the blanket calculation."""
    rho_vn = spectral_radius(c_mean, mu_values, sigma_t_mean, dx, n_modes=64)
    sigma_t_vec = [sigma_t_mean] * n_cells
    sigma_s_vec = [c_mean * sigma_t_mean] * n_cells
    for mu in mu_values[:2]:
        G = build_amplification_matrix(n_cells, dx, mu, sigma_t_vec, sigma_s_vec)
        lam, _ = power_iteration(G, n_iter=200)
    modes = laplacian_eigenmodes_1d(n_cells, dx, n_modes=3)
    return {
        "von_neumann_rho": rho_vn,
        "matrix_spectral_radius": lam,
        "first_eigenvalue": modes[0][0] if modes else 0.0,
        "critical_dx": dx,
        "stable": rho_vn < 1.0 and lam < 1.0,
    }
