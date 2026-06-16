# -*- coding: utf-8 -*-
"""
Centroidal Voronoi Tessellation (CVT) sampler for the 1D Brillouin zone
and circle-geometry tools for the Fermi surface.

Merged from:
  - ``256_cvt_corn_movie/cvt_disk``: CVT iteration with uniform or
    non-uniform sampling inside a disk.
  - ``185_circles/circles``: circle-geometry drawing (here repurposed
    as the Fermi-surface level-set construction).

For the 1D TFIM the reduced Brillouin zone is the interval
[-pi, pi] with endpoints identified.  We need a set of "representative"
k-points that is *uniform* with respect to a density rho(k) that
captures the Fermi velocity near the QCP.  CVT is the natural tool:

    min_{p_i} sum_i int_{V_i} rho(k) |k - p_i|^2 dk

The Lloyd iteration is
    p_i^{(n+1)} = int_{V_i} k rho(k) dk / int_{V_i} rho(k) dk

which we implement in 1-D with a non-uniform weight proportional to
1 / eps_k (the inverse single-particle gap) so that k-points are
pushed toward the gapless region near k = pi at the QCP.

The circle-geometry routine generates N equally spaced points on a
circle of radius R and is used to discretise the Fermi "surface"
(which in 1D is just two points {+kF, -kF}) as a limit of circles.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
try:
    from . import constants as C
except ImportError:
    import constants as C


# ---------------------------------------------------------------------------
# 1-D CVT for the Brillouin zone
# ---------------------------------------------------------------------------
def tfim_density(k: np.ndarray, J: float, h: float) -> np.ndarray:
    """Density weight for the k-space CVT:
        rho(k) propto 1 / eps_k
    where eps_k = 2 J sqrt(1 + lam^2 - 2 lam cos k) is the
    single-particle dispersion.  The divergence of 1/eps_k near the
    gapless mode at the QCP drives CVT points toward the important
    region.  We regularise with a floor proportional to 1/L.
    """
    lam = h / J if J != 0.0 else 0.0
    eps = 2.0 * abs(J) * np.sqrt(np.maximum(
        1.0 + lam * lam - 2.0 * lam * np.cos(k), 1.0e-8))
    return 1.0 / (eps + 1.0e-6)


def cvt_1d(n_points: int, n_samples: int = 4096,
            n_iter: int = 30,
            rho=None, seed: int = 0) -> np.ndarray:
    """Compute a 1-D CVT of [-pi, pi] with density ``rho`` (or uniform
    if None).  Returns the sorted generator locations.
    """
    rng = np.random.default_rng(seed)
    # Initial generator positions: roughly uniform in [-pi, pi]
    ps = np.sort(rng.uniform(-C.PI + 0.01, C.PI - 0.01, size=n_points))
    # Sample points for the Monte Carlo approximation of the Lloyd step
    xs = np.linspace(-C.PI, C.PI, n_samples)
    if rho is None:
        rs = np.ones_like(xs)
    else:
        rs = rho(xs)
    rs = np.maximum(rs, 0.0)
    dx = xs[1] - xs[0]

    for _ in range(n_iter):
        # Voronoi assignment: nearest generator
        # For 1-D sorted generators this is just midpoint-based.
        mids = 0.5 * (ps[:-1] + ps[1:])
        mids = np.concatenate([[-C.PI], mids, [C.PI]])
        for i in range(n_points):
            lo, hi = mids[i], mids[i + 1]
            mask = (xs >= lo) & (xs <= hi)
            weight = rs[mask]
            if weight.sum() < C.EPS_NUM:
                continue
            ps[i] = float(np.sum(xs[mask] * weight) / weight.sum())
    return np.sort(ps)


def cvt_energy(ps: np.ndarray, n_samples: int = 4096,
                rho=None) -> float:
    """Evaluate the CVT energy  E = sum_i int_{V_i} rho(k) |k - p_i|^2 dk."""
    ps = np.sort(ps)
    n = len(ps)
    xs = np.linspace(-C.PI, C.PI, n_samples)
    if rho is None:
        rs = np.ones_like(xs)
    else:
        rs = rho(xs)
    rs = np.maximum(rs, 0.0)
    dx = xs[1] - xs[0]
    mids = 0.5 * (ps[:-1] + ps[1:])
    mids = np.concatenate([[-C.PI], mids, [C.PI]])
    E = 0.0
    for i in range(n):
        lo, hi = mids[i], mids[i + 1]
        mask = (xs >= lo) & (xs <= hi)
        E += float(np.sum(rs[mask] * (xs[mask] - ps[i]) ** 2)) * dx
    return E


# ---------------------------------------------------------------------------
# Circle geometry  (Fermi-surface surrogate)
# ---------------------------------------------------------------------------
def circle_points(R: float, n_points: int = 100,
                   rotation_deg: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Return (x, y) arrays of n_points points on a circle of radius R
    centred at the origin, optionally rotated by rotation_deg.

    This mirrors the ``circles`` utility of the ``185_circles`` seed
    project, with ``n_points`` playing the role of the 'points' keyword.
    """
    theta = np.linspace(0.0, 2.0 * C.PI, n_points, endpoint=False)
    theta = theta + np.deg2rad(rotation_deg)
    x = R * np.cos(theta)
    y = R * np.sin(theta)
    return x, y


def fermi_circle(J: float, h: float, n_points: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """Discretise the 1D TFIM Fermi 'surface' as a circle of radius
    k_F in the (k_x, k_y) plane, with

        cos(k_F) = h / J   for |h/J| <= 1
        k_F = 0            otherwise.

    The circle is a surrogate that lets us visualise the QCP approach
    as a collapsing contour; it reduces to two points in the 1D limit.
    """
    lam = h / J if J != 0.0 else 0.0
    if abs(lam) >= 1.0:
        kF = 0.0
    else:
        kF = np.arccos(lam)
    return circle_points(R=kF, n_points=n_points)


# ---------------------------------------------------------------------------
# Radial growth  (finite-size scaling of the Fermi circle)
# ---------------------------------------------------------------------------
def fermi_radius_vs_L(Ls: np.ndarray, lam: float) -> np.ndarray:
    """Finite-size rounding of the Fermi radius:
        k_F(L) = k_F^inf + a / L
    where a ~ pi is the universal constant for open boundary conditions.
    """
    kF_inf = np.arccos(np.clip(lam, -1.0, 1.0)) if abs(lam) <= 1.0 else 0.0
    return kF_inf + C.PI / Ls.astype(float)
