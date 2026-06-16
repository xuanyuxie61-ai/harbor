"""
ode_chemistry.py
================

Chemical-kinetics ODE solver for the reduced CO/H2O/H2 gas-phase network
in the protoplanetary disk midplane.  This serves as LEVEL_2 (reduced ODE
fidelity) of the multi-fidelity hierarchy and also provides the time-
evolution sub-routine for the high-fidelity simulation.

Network (6 species, 8 reactions)
--------------------------------
We track the column-averaged number densities n_i (i = 0..5) of
  {H2, CO, H2O, CO2, CH4, OH}
subject to gas-phase reactions:
  R1:  H2 + CR   -> H2+ + e-                (cosmic-ray ionization)
  R2:  H2+ + CO  -> HCO+ + H                (fast ion-molecule)
  R3:  HCO+ + e- -> CO + H                  (dissociative recombination)
  R4:  H2O + CR  -> OH + H + e-
  R5:  OH + CO   -> CO2 + H
  R6:  CO + 3H   -> CH4 + H2O  (grain-surface effective)
  R7:  CH4 + OH  -> CH3 + H2O
  R8:  CO2 + H2  -> CO + H2O  (high-T reverse)

The ODE system is
    dn_i / dt = sum_k nu_{i,k} r_k(n, T)
with stoichiometric coefficients nu_{i,k} and rate coefficients
    r_k = k_k(T) * product_{j in reactants} n_j

Temperature dependence follows the Arrhenius / modified-Arrhenius form
    k_k(T) = alpha_k * (T / 300)^beta_k * exp(-gamma_k / T)

Integration uses a semi-implicit backward Euler method with Newton-Raphson
iteration for stability at high densities; the explicit RK4 method is also
available for verification.

Boundary / robustness features:
  - All densities are positivity-preserving (clamped to >= 0 after each step).
  - Temperature-dependent rates saturate outside [10 K, 3000 K].
  - A configurable relative tolerance controls adaptive sub-stepping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


# ----------------------------------------------------------------------
# Rate-coefficient database.
# ----------------------------------------------------------------------
@dataclass
class Reaction:
    """A single reaction with Arrhenius-modified rate coefficient."""
    label: str
    reactants: List[int]          # indices into the species vector
    products: List[int]
    alpha: float                  # pre-exponential factor [cgs]
    beta: float                   # temperature exponent
    gamma: float                  # activation temperature [K]
    stoich_react: List[float] = field(default_factory=list)  # per-reactant orders
    stoich_prod: List[float] = field(default_factory=list)   # per-product yields

    def rate_coeff(self, T: float) -> float:
        """k(T) = alpha (T/300)^beta exp(-gamma/T), clamped to [10, 3000] K."""
        Tc = min(max(T, 10.0), 3000.0)
        return self.alpha * math.pow(Tc / 300.0, self.beta) * math.exp(-self.gamma / Tc)


DEFAULT_REACTIONS: List[Reaction] = [
    Reaction("R1", [0], [0], alpha=1.0e-17, beta=0.0, gamma=0.0),
    Reaction("R2", [0, 1], [1], alpha=1.7e-9, beta=0.0, gamma=0.0),
    Reaction("R3", [1], [1], alpha=2.0e-7, beta=-0.5, gamma=0.0),
    Reaction("R4", [2], [5], alpha=5.0e-18, beta=0.0, gamma=0.0),
    Reaction("R5", [5, 1], [3], alpha=3.0e-13, beta=0.0, gamma=350.0),
    Reaction("R6", [1], [4, 2], alpha=1.0e-14, beta=0.5, gamma=600.0),
    Reaction("R7", [4, 5], [2], alpha=2.0e-11, beta=0.0, gamma=150.0),
    Reaction("R8", [3, 0], [1, 2], alpha=1.0e-12, beta=0.0, gamma=1500.0),
]


# ----------------------------------------------------------------------
# RHS evaluator.
# ----------------------------------------------------------------------
def rhs(
    n_vec: List[float],
    T: float,
    reactions: List[Reaction],
    n_species: int = 6,
) -> List[float]:
    """Evaluate dn_i/dt for the reduced chemical network.

    Parameters
    ----------
    n_vec : list of species number densities [cm^-3]
    T     : midplane temperature [K]
    reactions : list of Reaction objects
    n_species : number of species (default 6)
    """
    dn = [0.0] * n_species
    for rxn in reactions:
        # Rate = k(T) * prod_j n_j^{order_j}.
        r = rxn.rate_coeff(T)
        for idx, j in enumerate(rxn.reactants):
            order = rxn.stoich_react[idx] if idx < len(rxn.stoich_react) else 1.0
            nj = max(n_vec[j], 0.0)
            r *= math.pow(nj, order) if nj > 0.0 else 0.0
        # Consume reactants, produce products.
        for idx, j in enumerate(rxn.reactants):
            s = rxn.stoich_react[idx] if idx < len(rxn.stoich_react) else 1.0
            if j < n_species:
                dn[j] -= s * r
        for idx, j in enumerate(rxn.products):
            s = rxn.stoich_prod[idx] if idx < len(rxn.stoich_prod) else 1.0
            if j < n_species:
                dn[j] += s * r
    return dn


# ----------------------------------------------------------------------
# Explicit RK4 integrator.
# ----------------------------------------------------------------------
def rk4_step(
    n_vec: List[float],
    T: float,
    dt: float,
    reactions: List[Reaction],
    n_species: int = 6,
) -> List[float]:
    """Single RK4 step: y_{n+1} = y_n + dt/6 (k1 + 2k2 + 2k3 + k4)."""
    k1 = rhs(n_vec, T, reactions, n_species)
    y2 = [max(n_vec[i] + 0.5 * dt * k1[i], 0.0) for i in range(n_species)]
    k2 = rhs(y2, T, reactions, n_species)
    y3 = [max(n_vec[i] + 0.5 * dt * k2[i], 0.0) for i in range(n_species)]
    k3 = rhs(y3, T, reactions, n_species)
    y4 = [max(n_vec[i] + dt * k3[i], 0.0) for i in range(n_species)]
    k4 = rhs(y4, T, reactions, n_species)
    out = [0.0] * n_species
    for i in range(n_species):
        out[i] = max(
            n_vec[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]),
            0.0,
        )
    return out


# ----------------------------------------------------------------------
# Semi-implicit backward Euler with Newton iteration.
# ----------------------------------------------------------------------
def backward_euler_step(
    n_vec: List[float],
    T: float,
    dt: float,
    reactions: List[Reaction],
    n_species: int = 6,
    max_iter: int = 12,
    tol: float = 1.0e-6,
) -> List[float]:
    """Backward Euler step: y_{n+1} = y_n + dt * f(y_{n+1}).

    Solved by Newton iteration with a diagonal Jacobian approximation.
    Positivity-preserving by clamping after each Newton update.
    """
    y = list(n_vec)
    for _it in range(max_iter):
        f = rhs(y, T, reactions, n_species)
        # Approximate diagonal Jacobian by finite difference of each component.
        eps = 1.0e-8
        dy = [0.0] * n_species
        for i in range(n_species):
            yp = list(y)
            yp[i] += eps
            fp = rhs(yp, T, reactions, n_species)
            J_ii = (fp[i] - f[i]) / eps
            residual = y[i] - n_vec[i] - dt * f[i]
            denom = 1.0 - dt * J_ii
            if abs(denom) < 1.0e-30:
                dy[i] = 0.0
            else:
                dy[i] = -residual / denom
        # Damped update.
        damp = 1.0
        for i in range(n_species):
            y[i] = max(y[i] + damp * dy[i], 0.0)
        # Convergence check.
        res_norm = math.sqrt(sum((y[i] - n_vec[i] - dt * f[i]) ** 2 for i in range(n_species)))
        if res_norm < tol:
            break
    return y


# ----------------------------------------------------------------------
# Full time integrator with adaptive sub-stepping.
# ----------------------------------------------------------------------
def integrate_chemistry(
    n_vec_0: List[float],
    T: float,
    t_total: float,
    reactions: Optional[List[Reaction]] = None,
    n_species: int = 6,
    dt_init: float = 1.0e3,
    rtol: float = 1.0e-4,
    method: str = "rk4",
) -> Tuple[List[float], List[float]]:
    """Integrate the ODE system from t = 0 to t = t_total.

    Returns (n_vec_final, times) where times is the list of actual time
    steps taken.
    """
    if reactions is None:
        reactions = DEFAULT_REACTIONS
    if len(n_vec_0) != n_species:
        raise ValueError("integrate_chemistry: n_vec_0 length mismatch.")
    method = method.lower()
    if method not in ("rk4", "be"):
        raise ValueError("integrate_chemistry: unknown method.")
    step = (rk4_step if method == "rk4" else backward_euler_step)

    n_vec = list(n_vec_0)
    dt = dt_init
    t = 0.0
    times: List[float] = [0.0]
    while t < t_total:
        if t + dt > t_total:
            dt = t_total - t
        if dt <= 0.0:
            break
        n_new = step(n_vec, T, dt, reactions, n_species)
        # Adaptive sub-step control: if any species changed by > rtol of its
        # maximum, halve the step; if much less, grow it by sqrt(2).
        max_frac = 0.0
        for i in range(n_species):
            ref = max(abs(n_vec[i]), 1.0)
            frac = abs(n_new[i] - n_vec[i]) / ref
            if frac > max_frac:
                max_frac = frac
        if max_frac > rtol and dt > 1.0:
            dt *= 0.5
            continue
        if max_frac < 0.1 * rtol:
            dt *= math.sqrt(2.0)
        n_vec = n_new
        t += dt
        times.append(t)
    return n_vec, times


# ----------------------------------------------------------------------
# Exponential-decay unit test (following exp_ode from project 346).
# We integrate y' = alpha y using a custom RHS outside the network.
# ----------------------------------------------------------------------
def exp_rhs(n_vec: List[float], alpha: float) -> List[float]:
    """RHS for y' = alpha y (single species, linear growth/decay)."""
    return [alpha * n_vec[0]]


def exp_rk4_step(y: float, alpha: float, dt: float) -> float:
    """Single RK4 step for y' = alpha y."""
    k1 = alpha * y
    k2 = alpha * (y + 0.5 * dt * k1)
    k3 = alpha * (y + 0.5 * dt * k2)
    k4 = alpha * (y + dt * k3)
    return max(y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), 0.0)


def exp_ode_unit_test(alpha: float = -0.5, y0: float = 1.0,
                      tstop: float = 1.0, n_steps: int = 40) -> Dict[str, float]:
    """Solve y' = alpha y, y(0) = y0 with RK4 and compare to y0 exp(alpha t).

    This is a smoke test for the ODE integrator (adapted from exp_ode).
    """
    dt = tstop / n_steps
    y = y0
    for _ in range(n_steps):
        y = exp_rk4_step(y, alpha, dt)
    exact = y0 * math.exp(alpha * tstop)
    return {
        "numerical": y,
        "exact": exact,
        "rel_error": abs(y - exact) / max(abs(exact), 1.0e-30),
    }
