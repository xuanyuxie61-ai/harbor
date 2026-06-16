"""
stability_analysis.py - Stability Analysis Under Parametric Uncertainty

This module implements stability analysis for the reaction-diffusion
system under parameter uncertainty, integrating:
  - Eigenvalue-based stability classification (from neuronal-stability)
  - Turing instability condition evaluation
  - Stability probability estimation via surrogate

Mathematical Framework
----------------------
### Linear Stability Analysis ###

For the reaction-diffusion system linearized about steady state (u*, v*):
  d/dt [delta_u, delta_v]^T = J [delta_u, delta_v]^T + D nabla^2 [delta_u, delta_v]^T

where J is the Jacobian of the reaction kinetics and D = diag(D_u, D_v).

Decomposing into spatial modes via Fourier: delta ~ exp(ikx + sigma*t):
  sigma(k) satisfies:
  det(sigma*I - J + k^2*D) = 0
  sigma^2 - tr(J - k^2*D)*sigma + det(J - k^2*D) = 0

### Stability Conditions ###

The steady state is stable to homogeneous perturbations if:
  tr(J) = f_u + g_v < 0
  det(J) = f_u*g_v - f_v*g_u > 0

Turing instability (diffusion-driven) occurs when there exists k != 0
such that Re(sigma(k)) > 0. This requires:
  1. tr(J) < 0 (stable to homogeneous perturbations)
  2. det(J) > 0 (stable to homogeneous perturbations)
  3. D_v*f_u + D_u*g_v > 0 (diffusion destabilization)
  4. (D_v*f_u + D_u*g_v)^2 > 4*D_u*D_v*det(J) (real eigenvalues)

### Eigenvalue Classification (from neuronal-stability) ###

For 2x2 matrix A with trace tau and determinant delta:
  Characteristic polynomial: lambda^2 - tau*lambda + delta = 0
  Discriminant: Delta = tau^2 - 4*delta
  Eigenvalues: lambda_{1,2} = (tau ± sqrt(Delta)) / 2

Classification:
  - Stable node:     tau < 0, delta > 0, Delta >= 0
  - Stable spiral:   tau < 0, delta > 0, Delta < 0
  - Unstable node:   tau > 0, delta > 0, Delta >= 0
  - Unstable spiral: tau > 0, delta > 0, Delta < 0
  - Saddle:          delta < 0
  - Center:          tau = 0, delta > 0

### Stability Probability ###

Given uncertain parameters xi, the probability of Turing instability is:
  P(Turing) = integral I(turing_conditions(xi)) f_X(xi) dxi

Estimated via Monte Carlo sampling from the surrogate.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Optional, Tuple, Any
import math
from polynomial_chaos import PolynomialChaosSurrogate


# ---------------------------------------------------------------------------
# Eigenvalue Analysis
# ---------------------------------------------------------------------------

def eigenvalue_classification(trace: float, det: float) -> Dict[str, Any]:
    """
    Classify eigenvalues of a 2x2 matrix from trace and determinant.

    Parameters
    ----------
    trace : float
        Trace of the matrix (sum of eigenvalues).
    det : float
        Determinant of the matrix (product of eigenvalues).

    Returns
    -------
    dict with:
        'eigenvalues': complex or real eigenvalues
        'discriminant': tau^2 - 4*delta
        'type': classification string
        'stable': bool
        'oscillatory': bool
    """
    discriminant = trace ** 2 - 4.0 * det

    if discriminant >= 0:
        sqrt_disc = math.sqrt(discriminant)
        lam1 = (trace + sqrt_disc) / 2.0
        lam2 = (trace - sqrt_disc) / 2.0
        eigenvalues = (lam1, lam2)
        oscillatory = False
    else:
        re = trace / 2.0
        im = math.sqrt(-discriminant) / 2.0
        eigenvalues = (complex(re, im), complex(re, -im))
        oscillatory = True

    stable = trace < 0 and det > 0

    if det < -1e-14:
        typ = 'saddle'
    elif abs(trace) < 1e-10 and det > 0:
        typ = 'center'
    elif stable and oscillatory:
        typ = 'stable_spiral'
    elif stable and not oscillatory:
        typ = 'stable_node'
    elif not stable and oscillatory:
        typ = 'unstable_spiral'
    elif not stable and not oscillatory and trace > 0:
        typ = 'unstable_node'
    else:
        typ = 'degenerate'

    return {
        'eigenvalues': eigenvalues,
        'discriminant': discriminant,
        'type': typ,
        'stable': stable,
        'oscillatory': oscillatory,
        'trace': trace,
        'determinant': det
    }


# ---------------------------------------------------------------------------
# Turing Stability Analysis
# ---------------------------------------------------------------------------

def turing_stability_analysis(f_u: float, f_v: float,
                              g_u: float, g_v: float,
                              D_u: float, D_v: float) -> Dict[str, Any]:
    """
    Full Turing stability analysis.

    Checks all conditions for Turing instability and computes
    the critical wavenumber and growth rate.

    Parameters
    ----------
    f_u, f_v, g_u, g_v : float
        Jacobian elements of reaction kinetics.
    D_u, D_v : float
        Diffusion coefficients.

    Returns
    -------
    dict with comprehensive stability information.
    """
    # Homogeneous stability
    trace = f_u + g_v
    det = f_u * g_v - f_v * g_u
    homogeneous = eigenvalue_classification(trace, det)

    # Turing conditions
    diff_trace = D_v * f_u + D_u * g_v
    turing_disc = diff_trace ** 2 - 4.0 * D_u * D_v * det

    turing_possible = (
        trace < 0 and
        det > 0 and
        diff_trace > 0 and
        turing_disc > 0 and
        D_u > 0 and D_v > 0
    )

    # Critical wavenumber
    if turing_possible and D_u * D_v > 1e-30:
        kc_sq = diff_trace / (2.0 * D_u * D_v)
        kc = math.sqrt(max(kc_sq, 0.0))
    else:
        kc = 0.0

    # Maximum growth rate
    if turing_possible:
        sigma_max = (diff_trace - math.sqrt(max(turing_disc, 0.0))) / 2.0
        sigma_max = max(sigma_max, 0.0)
    else:
        sigma_max = 0.0

    # Range of unstable wavenumbers
    if turing_possible and D_u * D_v > 1e-30:
        sqrt_disc = math.sqrt(max(turing_disc, 0.0))
        k_min_sq = (diff_trace - sqrt_disc) / (2.0 * D_u * D_v)
        k_max_sq = (diff_trace + sqrt_disc) / (2.0 * D_u * D_v)
        k_range = (math.sqrt(max(k_min_sq, 0.0)),
                   math.sqrt(max(k_max_sq, 0.0)))
    else:
        k_range = (0.0, 0.0)

    return {
        'homogeneous_stable': homogeneous['stable'],
        'homogeneous_type': homogeneous['type'],
        'trace': trace,
        'determinant': det,
        'diffusion_trace': diff_trace,
        'turing_discriminant': turing_disc,
        'turing_possible': turing_possible,
        'critical_wavenumber': kc,
        'critical_wavelength': 2.0 * math.pi / kc if kc > 1e-14 else float('inf'),
        'max_growth_rate': sigma_max,
        'unstable_k_range': k_range
    }


# ---------------------------------------------------------------------------
# Stability Under Uncertainty
# ---------------------------------------------------------------------------

def stability_probability(surrogate_trace: PolynomialChaosSurrogate,
                          surrogate_det: PolynomialChaosSurrogate,
                          surrogate_diff_trace: Optional[PolynomialChaosSurrogate],
                          n_samples: int = 5000,
                          lower: Optional[NDArray] = None,
                          upper: Optional[NDArray] = None,
                          seed: int = 42) -> Dict[str, Any]:
    """
    Estimate probability of stability/Turing conditions via MC sampling.

    For each MC sample:
    1. Evaluate surrogate for trace, determinant (and diff_trace)
    2. Check stability conditions
    3. Accumulate statistics

    Parameters
    ----------
    surrogate_trace : PolynomialChaosSurrogate
        Surrogate for trace(J).
    surrogate_det : PolynomialChaosSurrogate
        Surrogate for det(J).
    surrogate_diff_trace : PolynomialChaosSurrogate, optional
        Surrogate for D_v*f_u + D_u*g_v.
    n_samples : int
        Number of MC samples.
    lower, upper : ndarray, optional
        Parameter bounds.
    seed : int
        Random seed.

    Returns
    -------
    dict with stability probabilities and statistics.
    """
    d = surrogate_trace.d
    if lower is None:
        lower = -np.ones(d)
    if upper is None:
        upper = np.ones(d)

    rng = np.random.RandomState(seed)
    X = lower + rng.random((n_samples, d)) * (upper - lower)

    trace_vals = surrogate_trace.predict(X)
    det_vals = surrogate_det.predict(X)

    if surrogate_diff_trace is not None:
        diff_trace_vals = surrogate_diff_trace.predict(X)
    else:
        diff_trace_vals = np.zeros(n_samples)

    # Stability classification for each sample
    n_homogeneous_stable = 0
    n_turing = 0
    n_unstable = 0
    n_saddle = 0
    stability_types = []

    for i in range(n_samples):
        tr = trace_vals[i]
        dt = det_vals[i]
        dt_trace = diff_trace_vals[i]

        if dt < -1e-14:
            stability_types.append('saddle')
            n_saddle += 1
        elif tr < 0 and dt > 0:
            n_homogeneous_stable += 1
            # Check Turing
            if dt_trace > 0 and dt_trace ** 2 > 4.0 * 0.01 * 0.5 * dt:
                n_turing += 1
                stability_types.append('turing')
            else:
                stability_types.append('stable')
        elif tr > 0 and dt > 0:
            n_unstable += 1
            stability_types.append('unstable')
        else:
            stability_types.append('marginal')

    total = float(n_samples)
    return {
        'P_homogeneous_stable': n_homogeneous_stable / total,
        'P_turing': n_turing / total,
        'P_unstable': n_unstable / total,
        'P_saddle': n_saddle / total,
        'n_samples': n_samples,
        'trace_mean': float(np.mean(trace_vals)),
        'trace_std': float(np.std(trace_vals)),
        'det_mean': float(np.mean(det_vals)),
        'det_std': float(np.std(det_vals)),
        'P_trace_negative': float(np.sum(trace_vals < 0) / total),
        'P_det_positive': float(np.sum(det_vals > 0) / total)
    }


# ---------------------------------------------------------------------------
# Stability Map Computation
# ---------------------------------------------------------------------------

def compute_stability_map(trace_range: Tuple[float, float],
                          det_range: Tuple[float, float],
                          n_trace: int = 50,
                          n_det: int = 50) -> Dict[str, NDArray]:
    """
    Compute stability classification map in (trace, det) space.

    The discriminant parabola det = trace^2/4 separates nodes from spirals.
    The axes trace = 0 and det = 0 separate stable from unstable regions.

    Returns
    -------
    dict with meshgrid arrays and classification labels.
    """
    trace_vals = np.linspace(trace_range[0], trace_range[1], n_trace)
    det_vals = np.linspace(det_range[0], det_range[1], n_det)
    TR, DET = np.meshgrid(trace_vals, det_vals)

    classification = np.empty((n_det, n_trace), dtype=object)

    for i in range(n_det):
        for j in range(n_trace):
            result = eigenvalue_classification(TR[i, j], DET[i, j])
            classification[i, j] = result['type']

    # Discriminant parabola
    disc_parabola = trace_vals ** 2 / 4.0

    return {
        'trace_grid': TR,
        'det_grid': DET,
        'classification': classification,
        'discriminant_parabola': disc_parabola,
        'trace_values': trace_vals,
        'det_values': det_vals
    }
