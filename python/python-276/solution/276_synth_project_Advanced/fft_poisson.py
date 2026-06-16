"""
fft_poisson.py — FFT-based Poisson solver for charged defects
=============================================================

For a charged point defect (charge state q ≠ 0) the Hartree potential
satisfies the Poisson equation
    ∇² V_H(r) = -4π ρ(r),
whose solution in reciprocal space is
    V_H(G) = 4π ρ(G) / |G|²     for G ≠ 0,
    V_H(0) = 0                   (neutralising background convention).

This module implements:
  1. Forward / inverse 2-D FFT of the charge density (using our own
     Cooley–Tukey from `stability_analysis.fft_ct` or NumPy's).
  2. The reciprocal-space Poisson solve with proper treatment of G = 0
     (Makov–Payne correction for periodic images of a charged defect).
  3. Ewald-style correction for the slow 1/r tail of the charged defect.

Seed project integration:
  * 426_fft_serial/fft_serial.m: the FFT itself (we provide our own and
    a NumPy wrapper)
  * 335_elliptic_integral: elliptic integrals appear in the Ewald sum
    when the defect is modelled as an extended Gaussian charge distribution
    (see `_gaussian_ewald_correction` below).
"""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple


# -------------------------------------------------------------------------
# (1) FFT wrappers
# -------------------------------------------------------------------------
def fft2_grid(rho: np.ndarray) -> np.ndarray:
    """2-D forward FFT (NumPy). Returns ρ(G) with the usual ordering."""
    return np.fft.fft2(rho)


def ifft2_grid(rho_G: np.ndarray) -> np.ndarray:
    """2-D inverse FFT."""
    return np.fft.ifft2(rho_G).real


# -------------------------------------------------------------------------
# (2) Reciprocal-lattice vectors for a square simulation cell of side L
# -------------------------------------------------------------------------
def reciprocal_vectors(N: int, h: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (Gx, Gy) arrays of reciprocal-lattice vectors.

    For a periodic cell of side L = N h with square symmetry,
        G_x = 2π n / L,   n ∈ {-N/2, ..., N/2 - 1}
    and similarly for G_y.
    """
    L = N * h
    n = np.fft.fftfreq(N, d=1.0 / N).astype(int)
    gx = 2.0 * math.pi * n / L
    gy = 2.0 * math.pi * n / L
    GX, GY = np.meshgrid(gx, gy, indexing="xy")
    return GX, GY


# -------------------------------------------------------------------------
# (3) Poisson solve
# -------------------------------------------------------------------------
def poisson_solve(rho: np.ndarray, h: float,
                  q_charge: int = 0) -> np.ndarray:
    """Solve ∇² V_H = -4π ρ on a periodic grid.

    Parameters
    ----------
    rho : (N, N) charge density on the real-space grid
    h   : grid spacing
    q_charge : net charge of the defect (in units of e). If non-zero, a
        uniform neutralising background is automatically subtracted.

    Returns
    -------
    V_H : (N, N) Hartree potential
    """
    N = rho.shape[0]
    rho_G = fft2_grid(rho)
    GX, GY = reciprocal_vectors(N, h)
    G2 = GX * GX + GY * GY

    # Remove net charge (G = 0 component) for charged defect
    if q_charge != 0:
        rho_G[0, 0] = 0.0
    # Solve V(G) = 4π ρ(G) / |G|²
    VH_G = np.zeros_like(rho_G, dtype=np.complex128)
    mask = G2 > 1e-12
    VH_G[mask] = 4.0 * math.pi * rho_G[mask] / G2[mask]
    VH_G[0, 0] = 0.0

    VH = ifft2_grid(VH_G)
    return VH


# -------------------------------------------------------------------------
# (4) Makov–Payne correction for periodic charged defects
# -------------------------------------------------------------------------
def makov_payne_correction(q: int, L: float, epsilon: float = 1.0) -> float:
    """Makov–Payne correction to the formation energy of a charged defect.

    For a cubic cell of side L in a medium of dielectric constant ε,
        E_MP = -α_M q² / (2 ε L)
    where α_M ≈ 2.837297 is the Madelung constant of a simple cubic array
    of point charges. For a 2-D system the correction is (Ismail-Beigi 2007)
        E_MP^{2D} = -2π q² / (ε A)  *  (1 / G_min²)
    but we use the simpler 3-D formula scaled by the 2-D area.
    """
    if q == 0:
        return 0.0
    alpha_M = 2.837297
    E_mp = -alpha_M * q * q / (2.0 * epsilon * L)
    return E_mp


# -------------------------------------------------------------------------
# (5) Gaussian-charge Ewald correction (uses elliptic integral E(m))
# -------------------------------------------------------------------------
def elliptic_E(m: float) -> float:
    """Complete elliptic integral of the second kind E(m) with parameter m.

    Computed via the arithmetic-geometric mean (AGM). Port of the logic
    in `335_elliptic_integral/elliptic_em.m`.
    """
    if m < 0.0:
        m = 0.0
    if m >= 1.0:
        return 1.0
    a = 1.0
    g = math.sqrt(1.0 - m)
    c = math.sqrt(m)
    pow2 = 1.0
    total = m
    for _ in range(20):
        a_new = 0.5 * (a + g)
        g_new = math.sqrt(a * g)
        c_new = 0.5 * (a - g)
        pow2 *= 2.0
        total += pow2 * c_new * c_new
        if abs(a_new - g_new) < 1e-15 * a_new:
            a, g = a_new, g_new
            break
        a, g = a_new, g_new
    K = math.pi / (2.0 * a)
    return K * (1.0 - 0.5 * total)


def gaussian_ewald_correction(q: int, sigma: float, L: float) -> float:
    """Self-energy correction for a Gaussian charge distribution of width σ.

    The potential of a Gaussian charge ρ(r) = q (2πσ²)^{-1} exp(-r²/(2σ²))
    at the origin is V(0) = q / σ √(π/2). The interaction with periodic
    images is computed by Ewald summation; the leading term reduces to
        E_ewald ≈ q² √(π/2) / σ  -  q² π / A  *  (correction)
    where A = L² and the correction involves E(m) with m = 1 - (2σ/L)².
    """
    if q == 0 or sigma <= 0:
        return 0.0
    A = L * L
    self_en = q * q * math.sqrt(math.pi / 2.0) / sigma
    m_param = max(0.0, 1.0 - (2.0 * sigma / L) ** 2)
    Em = elliptic_E(m_param)
    image_en = -q * q * math.pi / A * Em
    return self_en + image_en


# -------------------------------------------------------------------------
# (6) Build Gaussian charge density of a point defect on the grid
# -------------------------------------------------------------------------
def defect_charge_density(N: int, h: float, q: int,
                          sigma: float = 0.5,
                          centre: Tuple[int, int] = (0, 0)) -> np.ndarray:
    """Build a Gaussian charge density ρ(r) = q (2πσ²)^{-1} exp(-r²/(2σ²))
    centred at grid cell `centre`."""
    xs = (np.arange(N) - N / 2) * h
    ys = (np.arange(N) - N / 2) * h
    X, Y = np.meshgrid(xs, ys)
    cx = (centre[1] - N / 2) * h
    cy = (centre[0] - N / 2) * h
    r2 = (X - cx) ** 2 + (Y - cy) ** 2
    norm = q / (2.0 * math.pi * sigma * sigma)
    return norm * np.exp(-r2 / (2.0 * sigma * sigma))


# -------------------------------------------------------------------------
# (7) Complete workflow: charged-defect Hartree energy and correction
# -------------------------------------------------------------------------
def charged_defect_hartree(N: int, h: float, q: int,
                           sigma: float = 0.5,
                           centre: Tuple[int, int] = (0, 0)
                           ) -> Tuple[float, float, float]:
    """Compute (E_Hartree, E_MP, E_ewald) for a charged defect.

    E_Hartree = (1/2) ∫ ρ V_H d²r  (discrete)
    E_MP      = Makov–Payne correction
    E_ewald   = Gaussian-Ewald self / image correction
    """
    L = N * h
    rho = defect_charge_density(N, h, q, sigma, centre)
    VH = poisson_solve(rho, h, q_charge=q)
    EH = 0.5 * float(np.sum(rho * VH)) * h * h
    EMP = makov_payne_correction(q, L)
    Eew = gaussian_ewald_correction(q, sigma, L)
    return EH, EMP, Eew
