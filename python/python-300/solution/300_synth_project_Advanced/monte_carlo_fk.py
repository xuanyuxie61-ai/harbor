"""
monte_carlo_fk.py
=================
**Feynman-Kac** stochastic verification of the deterministic SN transport
solution, together with a **Monty-Hall-style conditional probability**
model for scattering-angle selection and a **basketball-trajectory**
analogue for neutron ballistics between collisions.

The Feynman-Kac formula expresses the solution of the steady transport
equation

    mu d psi / dx + Sigma_t psi = Q

as the expectation of a functional over stochastic paths:

    psi(x, mu) = E [ integral_0^tau exp(-int_0^s Sigma_t(X(r)) dr)
                     Q(X(s)) ds  |  X(0)=x, X'(0)=mu ]

where X(s) is a deterministic trajectory with velocity mu and tau is the
exit time from the domain.  In practice we sample N_walk random starting
positions and average the resulting path integrals.

Adapted from seed projects:
    * 424_feynman_kac_3d   -> path-integral verification
    * 779_monty_hall_simulation -> conditional-probability scattering
    * 073_basketball_dynamic -> parabolic-flight kinematics (ballistics)
"""

from __future__ import annotations
import math
from typing import Callable, Dict, List, Tuple

import physics_constants as pc


# ---------------------------------------------------------------------------
# Deterministic trajectory (basketball-dynamics analogue)
# ---------------------------------------------------------------------------
def ballistic_flight(
    x0: float, mu: float, v0: float,
    sigma_t_func: Callable[[float], float],
    max_time: float = 100.0,
    dt: float = 0.01,
    domain_L: float = 80.0,
) -> List[Tuple[float, float]]:
    """Trace a neutron trajectory x(t) under pure streaming + absorption.

    The trajectory follows  dx/dt = v0 * mu  with an exponential survival
    probability  exp(-Sigma_t dt)  at each step.  This is the *ballistic*
    phase of the neutron motion, analogous to the parabolic flight of a
    basketball between bounces.

    Returns a list of (t, x) pairs until the neutron exits [0, domain_L]
    or is absorbed.
    """
    path: List[Tuple[float, float]] = [(0.0, x0)]
    x = x0
    t = 0.0
    state = (int(x0 * 1e6) + 314159) & 0xFFFFFFFF
    while t < max_time:
        # survival check
        st = sigma_t_func(x)
        p_survive = math.exp(-st * v0 * abs(mu) * dt)
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        u = state / 0xFFFFFFFF
        if u > p_survive:
            break
        x += v0 * mu * dt
        t += dt
        path.append((t, x))
        if x < 0.0 or x > domain_L:
            break
    return path


# ---------------------------------------------------------------------------
# Feynman-Kac path integral estimator
# ---------------------------------------------------------------------------
def feynman_kac_estimator(
    x_eval: float,
    mu: float,
    sigma_t_func: Callable[[float], float],
    source_func: Callable[[float], float],
    n_walks: int = 500,
    domain_L: float = 80.0,
    v0: float = 1.0,
    seed: int = 271828,
) -> Tuple[float, float]:
    """Estimate psi(x_eval, mu) via the Feynman-Kac formula.

    For each random walk we:
      1. Start at x = x_eval moving with velocity mu.
      2. Advance in time steps dt, at each step:
         - accumulate the source contribution  Q(x) * dt * survival,
         - with probability 1 - exp(-Sigma_t v dt) the walk terminates
           (absorption).
      3. The walk also terminates if x exits [0, domain_L] (leakage).

    The estimator is the average over n_walks of the accumulated source.
    Returns (mean, std).
    """
    if n_walks < 1:
        return 0.0, 0.0
    dt = 0.05 * domain_L / 100.0   # adaptive time step
    state = seed & 0xFFFFFFFF
    values: List[float] = []
    for _ in range(n_walks):
        x = x_eval
        t = 0.0
        accumulated = 0.0
        survival = 1.0
        while t < 100.0:
            st = sigma_t_func(x)
            p_survive = math.exp(-st * v0 * abs(mu) * dt)
            state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
            u = state / 0xFFFFFFFF
            if u > p_survive:
                break
            # accumulate source
            accumulated += survival * source_func(x) * dt
            survival *= p_survive
            x += v0 * mu * dt
            t += dt
            if x < 0.0 or x > domain_L:
                break
        values.append(accumulated)
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1)
    return mean, math.sqrt(var)


# ---------------------------------------------------------------------------
# Monty-Hall conditional scattering model
# ---------------------------------------------------------------------------
def monty_hall_scattering_angle(
    mu_in: float,
    n_doors: int = 3,
    seed: int = 628318,
) -> float:
    """Choose a post-scattering direction cosine using a Monty-Hall-inspired
    conditional-probability model.

    The analogy:
      - The neutron "chooses" a door (scattering angle bin).
      - The "host" (physics) reveals one door that is *not* the correct
        scattering outcome (e.g. a backward scatter that violates energy
        conservation).
      - The neutron may "switch" to another door with probability p_switch.

    For n_doors = 3 and p_switch = 2/3 (the classic Monty Hall strategy),
    the expected probability of choosing the correct bin is 2/3.  We map
    this onto a continuous scattering angle by

        mu_out = mu_in * (1 - 2 * u_switch) + (1 - u_switch) * u_other

    where u_switch, u_other ~ U(0,1).
    """
    if n_doors < 2:
        raise ValueError("n_doors must be >= 2")
    state = seed & 0xFFFFFFFF
    state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
    u_switch = state / 0xFFFFFFFF
    state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
    u_other = state / 0xFFFFFFFF
    # classic Monty Hall: switch with probability (n-1)/n
    p_switch = (n_doors - 1.0) / n_doors
    if u_switch < p_switch:
        # switch: new angle drawn uniformly
        mu_out = 2.0 * u_other - 1.0
    else:
        # stay: keep original direction (forward-peaked)
        mu_out = mu_in
    return max(-1.0, min(1.0, mu_out))


# ---------------------------------------------------------------------------
# Conditional-probability scattering kernel
# ---------------------------------------------------------------------------
def conditional_scatter_kernel(
    mu_in: float,
    A_mass: float,
    n_samples: int = 100,
    seed: int = 161803,
) -> Tuple[float, float]:
    """Sample n_samples post-scattering mu_out for a neutron of mass 1
    scattering off a nucleus of mass A.

    The CM scattering angle is isotropic:  mu_cm ~ U(-1, 1).  The lab
    angle is

        mu_lab = (1 + A mu_cm) / sqrt(1 + A^2 + 2 A mu_cm)

    We also apply the Monty-Hall switching rule with probability p_switch
    to model the conditional rejection of kinematically forbidden angles.
    """
    if A_mass <= 0.0:
        raise ValueError("A_mass must be positive")
    state = seed & 0xFFFFFFFF
    mu_outs: List[float] = []
    for _ in range(n_samples):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        u1 = state / 0xFFFFFFFF
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        u2 = state / 0xFFFFFFFF
        mu_cm = 2.0 * u1 - 1.0
        denom = math.sqrt(1.0 + A_mass * A_mass + 2.0 * A_mass * mu_cm)
        if denom < pc.EPS_NUMERICAL:
            denom = pc.EPS_NUMERICAL
        mu_lab = (1.0 + A_mass * mu_cm) / denom
        # Monty-Hall conditional: reject if |mu_lab| > 1 (unphysical)
        if abs(mu_lab) > 1.0:
            # switch to isotropic
            mu_lab = 2.0 * u2 - 1.0
        mu_outs.append(max(-1.0, min(1.0, mu_lab)))
    mean = sum(mu_outs) / len(mu_outs)
    var = sum((v - mean) ** 2 for v in mu_outs) / max(len(mu_outs) - 1, 1)
    return mean, math.sqrt(var)


# ---------------------------------------------------------------------------
# Verification report
# ---------------------------------------------------------------------------
def fk_verification_report(
    x_eval: float,
    mu: float,
    sigma_t_func: Callable[[float], float],
    source_func: Callable[[float], float],
    phi_deterministic: float,
    n_walks: int = 500,
    domain_L: float = 80.0,
    seed: int = 141421,
) -> Dict[str, float]:
    """Compare the FK estimator to the deterministic solution."""
    psi_fk, sigma_fk = feynman_kac_estimator(
        x_eval, mu, sigma_t_func, source_func,
        n_walks=n_walks, domain_L=domain_L, seed=seed,
    )
    diff = abs(psi_fk - phi_deterministic)
    rel_err = diff / max(abs(phi_deterministic), pc.EPS_NUMERICAL)
    return {
        "x_eval": x_eval,
        "mu": mu,
        "phi_det": phi_deterministic,
        "psi_fk_mean": psi_fk,
        "psi_fk_std": sigma_fk,
        "abs_diff": diff,
        "rel_error": rel_err,
        "agreement": rel_err < 0.5,
    }
