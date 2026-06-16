"""
event_generator.py
==================

Unified Monte Carlo event generator for the 2->3 process:
- Combines phase-space sampling, matrix elements, viability cuts
- Implements the full event generation chain

Scientific context:
-------------------
The event generation chain for pp -> 3 jets at LO:

1. Generate phase-space point {p1, p2, p3} using importance sampling
2. Compute the LO matrix element |M_LO|^2
3. Apply the K-factor from NLO corrections (kernel regression)
4. Apply analysis cuts (viability scoring)
5. Accumulate weighted events for cross-section estimation

The differential cross-section is:

    d sigma = (1 / 2s) * |M|^2 * d Phi_3 * f_a(x1, mu_F) * f_b(x2, mu_F)

where:
- f_a, f_b are the parton distribution functions
- d Phi_3 is the 3-body Lorentz-invariant phase space
- |M|^2 is the spin/color-averaged squared matrix element

For the 2->3 gluon scattering gg -> ggg, the LO matrix element is:

    |M|^2 = g^6 * sum_{perms} (s_{ij}^4 / (s_{ik} * s_{il} * s_{jk} * s_{jl}))

(summed over permutations of final-state momenta).
"""

import math
import random
from typing import Dict, List, Tuple

from physics_constants import (
    PI, alpha_s_1loop, mandelstam_s, threshold_s_2to3,
    flux_factor, kallen_function, color_factor_adjoint
)
from phase_space_geometry import (
    annulus_sample, annulus_area, cube01_sample,
    phase_space_volume_ndim, feynman_x_moment_integral
)
from quadrature_engines import lyness_integrate, kallen_threshold_root
from kernel_methods import k_factor_from_matrix_element


# ===========================================================================
# Section 1: Parton distribution functions (simplified)
# ===========================================================================

def pdf_gluon_x(x: float, q2: float) -> float:
    """
    Simplified gluon PDF at scale Q^2:

        g(x, Q^2) = A * x^{-lambda} * (1-x)^5 * (Q^2/Q0^2)^{0.2}

    with lambda ~ 0.3 at small x (Regge behavior) and A normalized
    to the momentum sum rule.
    """
    if x <= 0.0 or x >= 1.0:
        return 0.0
    q0sq = 2.0  # GeV^2
    lam = 0.3
    A = 6.0
    val = A * math.pow(x, -lam) * (1.0 - x) ** 5
    # DGLAP evolution (approximate)
    if q2 > q0sq:
        val *= math.pow(q2 / q0sq, 0.2)
    return val


def pdf_quark_x(x: float, q2: float, flavor: int = 1) -> float:
    """
    Simplified quark PDF (valence up-quark):

        u_v(x, Q^2) = B * x^{0.5} * (1-x)^3 * (Q^2/Q0^2)^{-0.1}
    """
    if x <= 0.0 or x >= 1.0:
        return 0.0
    q0sq = 2.0
    B = 4.0
    val = B * math.pow(x, 0.5) * (1.0 - x) ** 3
    if q2 > q0sq:
        val *= math.pow(q2 / q0sq, -0.1)
    return val


# ===========================================================================
# Section 2: Matrix element computation
# ===========================================================================

def matrix_element_sq_gg_ggg(pt1: float, eta1: float, phi1: float,
                              pt2: float, eta2: float, phi2: float,
                              pt3: float, eta3: float, phi3: float,
                              sqrts: float) -> float:
    """
    Compute |M|^2 for gg -> ggg (all-gluon process).

    The LO matrix element squared (spin/color averaged) is:

        |M|^2 = 2 * g^6 * Nc^2 * (Nc^2-1)
                * sum_{perms of {1,2,3}} [s12^4/(s13*s14*s23*s24)
                                         + s13^4/(s12*s14*s23*s34)
                                         + ...]

    We use the Parke-Taylor MHV formula for the color-ordered amplitude:

        A(1-, 2-, 3+, 4+, 5+) = i * <12>^4 / (<12><23><34><45><51>)

    and sum over helicity configurations.

    For simplicity, we use the effective form:
        |M|^2 ~ alpha_s^3 * sum_{i<j} s_{ij}^2 / (product of other s_{kl})

    where s_{ij} = 2*pi*pj*(1 - cos(theta_{ij})).
    """
    # Compute 4-momenta (massless)
    def four_mom(pt, eta, phi):
        e = pt * math.cosh(eta)
        px = pt * math.cos(phi)
        py = pt * math.sin(phi)
        pz = pt * math.sinh(eta)
        return (e, px, py, pz)

    p1 = four_mom(pt1, eta1, phi1)
    p2 = four_mom(pt2, eta2, phi2)
    p3 = four_mom(pt3, eta3, phi3)

    # Invariant masses s_{ij} = 2*(Ei*Ej - pi.pj)
    def s_ij(pi, pj):
        return 2.0 * (pi[0] * pj[0] - pi[1] * pj[1]
                       - pi[2] * pj[2] - pi[3] * pj[3])

    s12 = max(1.0, s_ij(p1, p2))
    s13 = max(1.0, s_ij(p1, p3))
    s23 = max(1.0, s_ij(p2, p3))

    # Initial state partons (back-to-back in CM frame)
    sqrt_s_part = sqrts / 3.0  # approximate partonic sqrt(s)
    s_hat = sqrt_s_part ** 2

    # Simplified matrix element
    alpha_s = alpha_s_1loop(s12 + s13 + s23)
    g2 = 4.0 * PI * alpha_s
    nc = 3.0
    prefactor = 2.0 * g2 ** 3 * nc * nc * (nc * nc - 1.0)

    # Parke-Taylor-like sum
    me_sq = 0.0
    # Permutation 1: (12, 13, 23)
    denom = s12 * s13 * s23
    if denom > 1e-10:
        me_sq += s12 ** 2 / denom + s13 ** 2 / denom + s23 ** 2 / denom
    # Add more terms
    s_tot = s12 + s13 + s23
    if s_tot > 1e-10:
        me_sq += s_tot ** 2 / (s12 * s13 + 1e-10)
        me_sq += s_tot ** 2 / (s12 * s23 + 1e-10)
        me_sq += s_tot ** 2 / (s13 * s23 + 1e-10)

    return prefactor * me_sq


# ===========================================================================
# Section 3: Event generation
# ===========================================================================

class EventRecord:
    """Record for a single generated event."""

    def __init__(self):
        self.weight = 0.0
        self.pt = [0.0, 0.0, 0.0]
        self.eta = [0.0, 0.0, 0.0]
        self.phi = [0.0, 0.0, 0.0]
        self.me_sq = 0.0
        self.k_factor = 1.0
        self.viability = 0.0
        self.accepted = False


def generate_events(n_events: int, sqrts: float,
                    pt_min: float, pt_max: float,
                    seed: int = 42) -> Tuple[List[EventRecord], Dict[str, float]]:
    """
    Generate n_events for the 2->3 process using Monte Carlo.

    The generation proceeds as:
    1. Sample transverse momenta from the annular region [pt_min, pt_max]
    2. Sample rapidities uniformly in [-5, 5]
    3. Sample azimuthal angles uniformly in [0, 2*pi]
    4. Compute the matrix element
    5. Apply viability cuts
    6. Return accepted events and summary statistics

    Parameters
    ----------
    n_events : int
        Number of events to generate.
    sqrts : float
        Center-of-mass energy in GeV.
    pt_min, pt_max : float
        Transverse momentum range in GeV.
    seed : int
        Random seed.

    Returns
    -------
    (events, stats): list of event records and statistics dictionary.
    """
    rng = random.Random(seed)
    events = []
    total_weight = 0.0
    accepted_count = 0

    # Phase-space volume factor
    ps_vol = phase_space_volume_ndim(3, sqrts, [0.0, 0.0, 0.0])
    annulus = annulus_area(pt_min, pt_max)

    for iev in range(n_events):
        ev = EventRecord()
        # Sample 3 final-state partons
        pt_samples = annulus_sample(pt_min, pt_max, 3, rng)
        for j in range(3):
            ev.pt[j] = math.sqrt(pt_samples[j][0] ** 2 + pt_samples[j][1] ** 2)
            ev.eta[j] = rng.uniform(-4.0, 4.0)
            ev.phi[j] = rng.uniform(0.0, 2.0 * PI)
        # Compute matrix element
        ev.me_sq = matrix_element_sq_gg_ggg(
            ev.pt[0], ev.eta[0], ev.phi[0],
            ev.pt[1], ev.eta[1], ev.phi[1],
            ev.pt[2], ev.eta[2], ev.phi[2],
            sqrts
        )
        # K-factor (simplified)
        ev.k_factor = 1.0 + 0.1 * alpha_s_1loop(ev.me_sq + 1.0)
        # Event weight
        ev.weight = (ev.me_sq * ev.k_factor * ps_vol
                     / (2.0 * mandelstam_s(sqrts) + 1e-30))
        # Viability
        ev.viability = 1.0
        if ev.pt[0] > pt_min and ev.pt[1] > pt_min and ev.pt[2] > pt_min:
            ev.accepted = True
            accepted_count += 1
        total_weight += ev.weight
        events.append(ev)

    stats = {
        'n_generated': n_events,
        'n_accepted': accepted_count,
        'acceptance_rate': accepted_count / max(1, n_events),
        'total_weight': total_weight,
        'mean_weight': total_weight / max(1, n_events),
        'cross_section_pb': total_weight * 1e-9,  # GeV^-2 -> pb
        'phase_space_volume': ps_vol,
        'annulus_area': annulus,
    }
    return events, stats


def compute_scale_uncertainty(events: List[EventRecord],
                              mu_r_factor: float = 2.0
                              ) -> Dict[str, float]:
    """
    Estimate the scale uncertainty by varying the renormalization scale:

        mu_R -> mu_R * f,  f in {0.5, 1.0, 2.0}

    The cross-section changes as:
        sigma(mu_R) = sigma_0 * (alpha_s(mu_R) / alpha_s(mu_R_0))^n

    where n is the power of alpha_s in the process (n=3 for 2->3 at LO).
    """
    if not events:
        return {'sigma_up': 0.0, 'sigma_down': 0.0, 'sigma_central': 0.0}
    sigma_0 = sum(ev.weight for ev in events)
    n_power = 3  # LO 2->3
    # alpha_s variation
    q2_central = 100.0  # GeV^2
    alpha_central = alpha_s_1loop(q2_central)
    alpha_up = alpha_s_1loop(q2_central * mu_r_factor ** 2)
    alpha_down = alpha_s_1loop(q2_central / (mu_r_factor ** 2))
    ratio_up = (alpha_up / alpha_central) ** n_power
    ratio_down = (alpha_down / alpha_central) ** n_power
    return {
        'sigma_central': sigma_0,
        'sigma_up': sigma_0 * ratio_up,
        'sigma_down': sigma_0 * ratio_down,
        'scale_variation_pct': abs(ratio_up - ratio_down) / 2.0 * 100.0,
    }
