"""
stability_analysis.py
=====================

Stability analysis and convergence checking for the numerical methods:
- Eigenvalue stability analysis
- Error norm computation
- Convergence order verification

Scientific context:
-------------------
The stability of the DGLAP evolution and parton shower depends on:
1. The eigenvalues of the anomalous dimension matrix (must have Re < 0
   for the evolution to be well-posed as t -> infinity)
2. The step size in the numerical integration (must satisfy the CFL
   condition for explicit methods)
3. The conditioning of the moment Hankel matrix (must be well-conditioned
   for the Cholesky factorization to be numerically stable)

The Courant-Friedrichs-Lewy (CFL) condition for the FD discretization:

    dt <= C * dx^2 / (2 * D)

where D is the diffusion coefficient and C ~ 0.5 is the safety factor.

For the implicit midpoint method, the stability region includes the
entire left half-plane (A-stable), but the fixed-point iteration for
the implicit stage requires:

    |h * df/dy| < 1

for convergence of the fixed-point iteration.
"""

import math
from typing import Dict, List, Tuple


# ===========================================================================
# Section 1: Eigenvalue stability analysis
# ===========================================================================

def matrix_eigenvalues_2x2(A: List[List[float]]) -> List[complex]:
    """
    Compute eigenvalues of a 2x2 matrix:

        lambda = (tr(A) +/- sqrt(tr(A)^2 - 4*det(A))) / 2

    Returns a list of 2 complex eigenvalues.
    """
    if len(A) != 2 or len(A[0]) != 2:
        raise ValueError("matrix_eigenvalues_2x2: not a 2x2 matrix")
    tr = A[0][0] + A[1][1]
    det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
    disc = tr * tr - 4.0 * det
    if disc >= 0.0:
        sq = math.sqrt(disc)
        return [complex((tr + sq) / 2.0, 0.0), complex((tr - sq) / 2.0, 0.0)]
    else:
        sq = math.sqrt(-disc)
        return [complex(tr / 2.0, sq / 2.0), complex(tr / 2.0, -sq / 2.0)]


def anomalous_dimension_matrix(n_moments: int, alpha_s: float
                               ) -> List[List[float]]:
    """
    Build the anomalous dimension matrix for the first n_moments Mellin
    moments of the DGLAP evolution.

    The LO anomalous dimensions for non-singlet evolution:

        gamma_N^{(0)} = -2*CF * (1/(N*(N+1)) - 2*sum_{j=2}^{N+1} 1/j + 1/2)

    For the singlet case, we have a 2x2 matrix:
        gamma = [[gamma_qq, gamma_qg], [gamma_gq, gamma_gg]]

    We construct the truncated moment-space matrix for stability analysis.
    """
    cf = 4.0 / 3.0
    mat = [[0.0] * n_moments for _ in range(n_moments)]
    for i in range(n_moments):
        n = i + 1  # moment index N = 1, 2, ...
        # Non-singlet anomalous dimension (simplified)
        harmonic = sum(1.0 / j for j in range(2, n + 2))
        gamma_ns = -2.0 * cf * (1.0 / (n * (n + 1.0)) - 2.0 * harmonic + 0.5)
        mat[i][i] = gamma_ns * alpha_s / (2.0 * math.pi)
        # Off-diagonal mixing (simplified)
        if i + 1 < n_moments:
            mat[i][i + 1] = -cf * alpha_s / (4.0 * math.pi * n)
            mat[i + 1][i] = -cf * alpha_s / (4.0 * math.pi * n)
    return mat


def check_stability(eigenvalues: List[complex],
                    evolution_direction: str = 'forward') -> Dict[str, float]:
    """
    Check the stability of the evolution based on eigenvalues.

    For DGLAP forward evolution in ln(Q^2), positive eigenvalues
    correspond to growth of moments with scale (physical, PDF evolution).
    For backward evolution, they would indicate instability.

    For a well-posed forward evolution:
    - All eigenvalues must have |Im(lambda)|/|Re(lambda)| bounded (no wild oscillation)
    - The stiffness ratio is max|Re| / min|Re| (large = stiff)

    Returns a dictionary with stability metrics.
    """
    real_parts = [ev.real for ev in eigenvalues]
    imag_parts = [ev.imag for ev in eigenvalues]
    max_re = max(real_parts) if real_parts else 0.0
    min_re = min(real_parts) if real_parts else 0.0
    max_abs = max(abs(ev) for ev in eigenvalues) if eigenvalues else 0.0
    min_abs_nonzero = min(
        (abs(ev) for ev in eigenvalues if abs(ev) > 1e-15),
        default=1.0
    )
    stiffness = max_abs / min_abs_nonzero if min_abs_nonzero > 1e-15 else float('inf')
    # For forward evolution (DGLAP), bounded eigenvalues = stable
    # Physical requirement: no exponentially growing imaginary parts
    max_im_ratio = max((abs(ev.imag) / (abs(ev.real) + 1e-15)
                        for ev in eigenvalues if abs(ev) > 1e-15),
                       default=0.0)
    stable = (max_abs < 100.0 and max_im_ratio < 10.0)
    return {
        'max_real_part': max_re,
        'min_real_part': min_re,
        'max_abs_eigenvalue': max_abs,
        'stiffness_ratio': stiffness,
        'max_im_re_ratio': max_im_ratio,
        'is_stable': stable,
    }


# ===========================================================================
# Section 2: Error norms and convergence
# ===========================================================================

def l2_norm(u: List[float], v: List[float]) -> float:
    """Compute the L2 norm of the difference: ||u - v||_2."""
    if len(u) != len(v):
        raise ValueError("l2_norm: vectors have different lengths")
    s = sum((ui - vi) ** 2 for ui, vi in zip(u, v))
    return math.sqrt(s / len(u)) if len(u) > 0 else 0.0


def linf_norm(u: List[float], v: List[float]) -> float:
    """Compute the L-infinity norm: ||u - v||_inf = max |u_i - v_i|."""
    if len(u) != len(v):
        raise ValueError("linf_norm: vectors have different lengths")
    if not u:
        return 0.0
    return max(abs(ui - vi) for ui, vi in zip(u, v))


def relative_error(approx: float, exact: float) -> float:
    """Compute the relative error |approx - exact| / |exact|."""
    if abs(exact) < 1e-15:
        return abs(approx - exact)
    return abs(approx - exact) / abs(exact)


def convergence_order(errors: List[float], spacings: List[float]
                      ) -> float:
    """
    Estimate the convergence order p from a sequence of errors and
    grid spacings:

        error ~ C * h^p

    Taking logs: log(error) = log(C) + p * log(h)

    We fit p using consecutive pairs:
        p = log(e_i / e_{i+1}) / log(h_i / h_{i+1})
    """
    if len(errors) < 2 or len(spacings) < 2:
        return 0.0
    orders = []
    for i in range(len(errors) - 1):
        if errors[i + 1] < 1e-15 or spacings[i + 1] < 1e-15:
            continue
        if errors[i] < 1e-15 or spacings[i] < 1e-15:
            continue
        p = math.log(errors[i] / errors[i + 1]) / math.log(spacings[i] / spacings[i + 1])
        orders.append(p)
    if not orders:
        return 0.0
    return sum(orders) / len(orders)


# ===========================================================================
# Section 3: CFL condition and step-size constraints
# ===========================================================================

def cfl_condition(dx: float, diffusion: float, safety: float = 0.5) -> float:
    """
    Compute the maximum stable time step for explicit FD diffusion:

        dt_max = safety * dx^2 / (2 * D)

    Parameters
    ----------
    dx : float
        Grid spacing.
    diffusion : float
        Diffusion coefficient D.
    safety : float
        Safety factor (default 0.5).

    Returns
    -------
    float: maximum stable dt.
    """
    if diffusion <= 0.0:
        return float('inf')
    return safety * dx * dx / (2.0 * diffusion)


def fixed_point_contraction_rate(L: float, h: float) -> float:
    """
    Estimate the contraction rate for the fixed-point iteration in
    the implicit midpoint method:

        rate = |h * L / 2|

    where L is the Lipschitz constant of f and h is the time step.
    For convergence, we need rate < 1.
    """
    return abs(h * L / 2.0)


# ===========================================================================
# Section 4: Condition number estimation
# ===========================================================================

def condition_number_estimate(A: List[List[float]]) -> float:
    """
    Estimate the condition number of matrix A using the ratio of
    max to min diagonal elements (rough estimate for diagonally
    dominant matrices).

    For a more accurate estimate, we would need SVD, but this suffices
    for checking numerical stability.
    """
    n = len(A)
    if n == 0:
        return 1.0
    diag = [abs(A[i][i]) for i in range(n)]
    max_d = max(diag) if diag else 1.0
    min_d = min(d for d in diag if d > 1e-15) if any(d > 1e-15 for d in diag) else 1e-15
    return max_d / min_d


def hankel_condition(moments: List[float], n: int) -> Dict[str, float]:
    """
    Compute the condition number of the Hankel moment matrix.

    A well-conditioned Hankel matrix has cond(H) < 1e6.
    Ill-conditioning (cond > 1e10) indicates that the moments
    are nearly linearly dependent and the resummation will be
    numerically unstable.
    """
    from hankel_moments import moment_hankel_matrix
    H = moment_hankel_matrix(moments, n)
    cond = condition_number_estimate(H)
    return {
        'condition_number': cond,
        'is_well_conditioned': cond < 1e6,
        'n_moments_used': n,
    }
