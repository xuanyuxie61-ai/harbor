"""
phase_space_geometry.py
=======================

Phase-space geometry for multi-particle scattering:
- Fractal boundary generation for non-perturbative hadronization regions
  (from 446_fractal_coastline)
- Importance sampling in bounded annular momentum regions
  (from 010_annulus_monte_carlo)
- Monomial integration over the unit hypercube of Feynman-x fractions
  (from 234_cube_integrals)

Scientific context:
-------------------
For a 2->3 scattering process, the final-state phase space is bounded by:

    dPhi_3 = (2*pi)^4 * delta^4(P - sum pi) * prod d^3 p_i / ((2*pi)^3 * 2*E_i)

The non-perturbative hadronization boundary in (pT, eta) space has fractal
character (Mandelbrot, 1982). We model this using a recursive midpoint
perturbation of a closed polygon in 2D momentum space.

The annular region in transverse momentum:
    pT_min^2 <= pT_x^2 + pT_y^2 <= pT_max^2
is sampled via the Shirley warping technique (Graphics Gems III, 1992).

The Feynman-x fractions x_i lie in the unit hypercube [0,1]^n subject to
the momentum sum rule sum x_i = 1, giving integrals of the form:
    I = int_[0,1]^n x1^a1 * x2^a2 * ... * xn^an * delta(1-sum xi) d^n x
which evaluate analytically to a product of Beta functions.
"""

import math
import random
from typing import List, Tuple

from physics_constants import PI


# ===========================================================================
# Section 1: Fractal boundary (from 446_fractal_coastline)
# ===========================================================================

def coastline_perturb(points: List[Tuple[float, float]],
                      mu: float, rng: random.Random) -> List[Tuple[float, float]]:
    """
    Insert intermediate points in a closed polygonal curve with random
    perturbation (midpoint displacement algorithm).

    For a curve P of n points, produces Q of 2n points where:
        Q[2k]   = P[k]                            (original)
        Q[2k+1] = 0.5*(P[k] + P[k+1])
                  + w[k]*(P[k] + P[k+1])
                  - w[k]*(P[k-1] + P[k+2])        (perturbed midpoint)

    with w[k] = mu + mu^2 * N(0,1).

    Parameters
    ----------
    points : list of (x, y) tuples
        The closed polygonal curve (first point != last point).
    mu : float
        Perturbation strength, 0 <= mu <= 0.25 recommended.
    rng : random.Random
        Seeded RNG for reproducibility.

    Returns
    -------
    list of (x, y) tuples of length 2*n.
    """
    n = len(points)
    if n < 3:
        return list(points)
    sig = mu * mu
    q = []
    for k in range(n):
        pk = points[k]
        pk1 = points[(k + 1) % n]
        pkm1 = points[(k - 1) % n]
        pk2 = points[(k + 2) % n]
        w = mu + sig * rng.gauss(0.0, 1.0)
        # Perturbed midpoint in x and y
        for dim in range(2):
            avg = 0.5 * (pk[dim] + pk1[dim])
            perturb = (avg
                       + w * (pk[dim] + pk1[dim])
                       - w * (pkm1[dim] + pk2[dim]))
            # Store in interleaved form
        q.append(pk)
        # Compute the perturbed midpoint
        mx = (0.5 * (pk[0] + pk1[0])
              + w * (pk[0] + pk1[0])
              - w * (pkm1[0] + pk2[0]))
        my = (0.5 * (pk[1] + pk1[1])
              + w * (pk[1] + pk1[1])
              - w * (pkm1[1] + pk2[1]))
        q.append((mx, my))
    return q


def fractal_hadronization_boundary(mu: float, iterations: int,
                                   seed: int = 42) -> List[Tuple[float, float]]:
    """
    Generate a fractal boundary approximating the non-perturbative
    hadronization contour in (pT, rapidity) space.

    Starting from a regular hexagon in (pT, y) space with radius ~ 50 GeV,
    we apply `iterations` rounds of midpoint perturbation with strength mu.

    The resulting curve represents the boundary of the perturbative region:
    inside the curve = perturbative (alpha_s small),
    outside = non-perturbative (requires hadronization model).
    """
    rng = random.Random(seed)
    # Initial hexagon in (pT, y) space, pT in GeV, y dimensionless
    r_pt = 50.0  # GeV
    r_y = 3.0    # units of rapidity
    pts = []
    for k in range(6):
        theta = 2.0 * PI * k / 6.0
        pts.append((r_pt * math.cos(theta), r_y * math.sin(theta)))
    for _ in range(iterations):
        pts = coastline_perturb(pts, mu, rng)
    return pts


def fractal_dimension_estimate(points: List[Tuple[float, float]]) -> float:
    """
    Estimate the box-counting fractal dimension D of a closed curve.

    Uses the relation N(eps) ~ eps^(-D) where N(eps) is the number of
    boxes of size eps needed to cover the curve.

    For a smooth curve, D = 1. For a fractal (e.g., Koch curve), D > 1.
    The hadronization boundary is expected to have D ~ 1.1 - 1.3.

    We use multiple box sizes and fit the slope of log(N) vs log(1/eps).
    """
    if len(points) < 4:
        return 1.0
    # Get bounding box
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    L = max(x_max - x_min, y_max - y_min)
    if L < 1e-12:
        return 1.0
    # Count boxes at different scales
    eps_list = [L / (2 ** k) for k in range(1, 8)]
    counts = []
    for eps in eps_list:
        if eps < 1e-12:
            continue
        boxes = set()
        for x, y in points:
            bx = int((x - x_min) / eps)
            by = int((y - y_min) / eps)
            boxes.add((bx, by))
        counts.append(len(boxes))
    if len(counts) < 2:
        return 1.0
    # Fit log(N) = -D * log(eps) + const via least squares
    n_pts = len(counts)
    sum_lnx = sum(math.log(1.0 / eps) for eps in eps_list[:n_pts])
    sum_lny = sum(math.log(max(1, c)) for c in counts)
    sum_lnxy = sum(math.log(1.0 / eps_list[i]) * math.log(max(1, counts[i]))
                   for i in range(n_pts))
    sum_lnx2 = sum(math.log(1.0 / eps_list[i]) ** 2 for i in range(n_pts))
    denom = n_pts * sum_lnx2 - sum_lnx * sum_lnx
    if abs(denom) < 1e-12:
        return 1.0
    d = (n_pts * sum_lnxy - sum_lnx * sum_lny) / denom
    return max(1.0, min(d, 2.0))


# ===========================================================================
# Section 2: Annulus sampling (from 010_annulus_monte_carlo)
# ===========================================================================

def annulus_area(r1: float, r2: float) -> float:
    """
    Area of a circular annulus with inner radius r1, outer radius r2:

    A = pi * (r2 + r1) * (r2 - r1) = pi * (r2^2 - r1^2)
    """
    if r1 < 0.0:
        raise ValueError(f"annulus_area: inner radius r1={r1} < 0")
    if r2 < r1:
        raise ValueError(f"annulus_area: r2={r2} < r1={r1}")
    return PI * (r2 + r1) * (r2 - r1)


def annulus_sample(r1: float, r2: float, n: int,
                   rng: random.Random) -> List[Tuple[float, float]]:
    """
    Sample n points uniformly in an annulus r1^2 <= x^2 + y^2 <= r2^2.

    Uses the Shirley warping (Graphics Gems III, 1992):
        theta = U1 * 2*pi
        r = sqrt((1-U2)*r1^2 + U2*r2^2)

    This gives uniform density in the annular region.
    """
    if r1 < 0.0:
        raise ValueError(f"annulus_sample: r1={r1} < 0")
    if r2 < r1:
        raise ValueError(f"annulus_sample: r2={r2} < r1={r1}")
    if n < 0:
        raise ValueError(f"annulus_sample: n={n} < 0")
    pts = []
    r1sq = r1 * r1
    r2sq = r2 * r2
    for _ in range(n):
        u1 = rng.random()
        u2 = rng.random()
        theta = u1 * 2.0 * PI
        r = math.sqrt((1.0 - u2) * r1sq + u2 * r2sq)
        pts.append((r * math.cos(theta), r * math.sin(theta)))
    return pts


def sample_transverse_momenta(pt_min: float, pt_max: float, n: int,
                              rng: random.Random) -> List[Tuple[float, float]]:
    """
    Sample n transverse momentum vectors in the annular region:
        pt_min^2 <= px^2 + py^2 <= pt_max^2

    Physical interpretation: final-state parton transverse momenta
    bounded by detector acceptance (pt_min ~ 20 GeV) and kinematic
    ceiling (pt_max ~ sqrt(s)/2).
    """
    return annulus_sample(pt_min, pt_max, n, rng)


# ===========================================================================
# Section 3: Hypercube monomial integration (from 234_cube_integrals)
# ===========================================================================

def cube01_monomial_integral(exponents: List[int]) -> float:
    """
    Integral of the monomial x1^e1 * x2^e2 * ... * xn^en over the
    unit cube [0,1]^n:

        I = prod_{i=1}^n 1/(e_i + 1)

    In the phase-space context, e_i represent powers of Feynman-x
    moments in the PDF convolution integrals.
    """
    result = 1.0
    for e in exponents:
        if e < 0:
            raise ValueError(f"cube01_monomial_integral: negative exponent {e}")
        result /= (e + 1.0)
    return result


def cube01_sample(n: int, dim: int,
                  rng: random.Random) -> List[List[float]]:
    """
    Sample n points uniformly in the unit cube [0,1]^dim.

    Used for Monte Carlo estimation of multi-dimensional phase-space
    integrals over Feynman-x fractions.
    """
    if n < 0 or dim < 1:
        raise ValueError(f"cube01_sample: invalid n={n} or dim={dim}")
    return [[rng.random() for _ in range(dim)] for _ in range(n)]


def phase_space_volume_ndim(n_final: int, sqrts: float,
                            masses: List[float]) -> float:
    """
    Lorentz-invariant n-body phase-space volume (no mass factors):

    Phi_n(s) = (pi^(n-1) / (2^(n-1) * (n-1)!))
               * prod_{k=1}^{n-1} (1/k) * s^(n-2) * (1/sqrt(s))^(2n-4)

    This is the massless approximation. With masses, the actual volume
    is smaller by threshold factors.
    """
    if n_final < 2:
        return 0.0
    s = sqrts * sqrts
    if s <= 0.0:
        return 0.0
    # Check threshold
    msum = sum(masses) if masses else 0.0
    if sqrts < msum:
        return 0.0
    # Massless approximation
    n = n_final
    log_vol = ((n - 1) * math.log(PI)
               - (n - 1) * math.log(2.0)
               - sum(math.log(k + 1) for k in range(n - 1))
               + (n - 2) * math.log(s))
    return math.exp(log_vol)


def feynman_x_moment_integral(powers: List[int], n_partons: int) -> float:
    """
    Compute the moment integral over the (n-1)-simplex of momentum fractions:

        M(a1,...,an) = int_[0,1]^n x1^a1 * ... * xn^an * delta(1 - sum xi) d^n x

    This equals the product of Gamma functions:

        M = prod Gamma(ai+1) / Gamma(sum ai + n)
    """
    num = 1.0
    for a in powers:
        num *= math.gamma(a + 1.0)
    denom = math.gamma(sum(powers) + n_partons)
    return num / denom
