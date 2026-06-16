"""
defect_rate_equations.py
========================
Kinetic rate equations for defect formation, ion migration, and degradation
in MAPbI3 perovskite solar cells. The key equations modeled here:

1. Logistic defect-generation kinetics (from 702_logistic_ode):
       dN_t/dt = r * N_t * (1 - N_t / N_max)
   where N_max is the maximum defect density (lattice-site saturation)
   and r is a temperature-dependent rate r = r_0 exp(-E_a / kT).

2. Ion-migration pair interaction network (from 1023_covid-spread):
   Defects (iodine vacancies V_I) migrate through the lattice and interact
   pairwise; we track the state of each defect site as susceptible (S),
   occupied (O), or aggregated (A), analogous to the SIRD model.

3. Chemical-bond breaking during pyrolysis (from 1025_USTBifrt_Pyrolysis):
   MAPbI3 decomposes at elevated temperature:
       MAPbI3 -> PbI2 + CH3NH2 + HI
   The bond-breaking population follows first-order kinetics:
       d[Bond]/dt = -k_break [Bond]
   with Arrhenius rate k_break = A exp(-E_a/kT).

4. Matrix-chain optimal reaction pathway (from 739_matrix_chain_brute):
   When multiple defect species interact (e.g. V_I + I_i -> neutral pair,
   then pair + Pb_i -> complex), the order of reactions affects the total
   computational cost of tracking all combinations. We use Catalan-number
   enumeration to find the optimal ordering.

Mathematical model
------------------
For a single defect species with logistic saturation:
    dN_t/dt = r(T) N_t (1 - N_t/N_max) - gamma N_t  (loss term)
where r(T) = r_0 exp(-E_a/kT) is the Arrhenius generation rate and
gamma is the annealing rate. This is a Bernoulli ODE with analytical
solution (via u = 1/N_t):
    du/dt + (gamma - r) u = -r / N_max
with integrating factor exp((gamma - r)t).

For the pair-interaction network:
    dp_i/dt = beta sum_j A_{ij} S_i O_j - gamma p_i
where A is the adjacency matrix of the defect lattice and p_i is the
probability that site i hosts an active defect.
"""

from __future__ import annotations
import math
from typing import Tuple, List, Optional

import numpy as np
from numpy.typing import NDArray

from perovskite_constants import (
    K_B, T_K, E_CHARGE, DEFECT_DENSITY_DEFAULT, BAND_GAP_J
)


# ============================================================================
# Arrhenius rate constant
# ============================================================================
def arrhenius_rate(A0: float, E_a_J: float, T: float = T_K) -> float:
    """Compute the Arrhenius rate k = A_0 exp(-E_a / kT).
    A0      : pre-exponential factor [1/s]
    E_a_J   : activation energy [J]
    """
    return A0 * math.exp(-E_a_J / (K_B * T))


# ============================================================================
# 1. Logistic defect generation (from 702_logistic_ode)
# ============================================================================
def logistic_defect_deriv(t: float, N_t: float,
                          r: float, N_max: float,
                          gamma: float = 0.0) -> float:
    """Right-hand side of the logistic defect ODE:
    dN_t/dt = r N_t (1 - N_t/N_max) - gamma N_t

    Parameters
    ----------
    r     : intrinsic growth rate [1/s] (Arrhenius temperature-dependent)
    N_max : saturation defect density [1/m^3] (lattice-site limit)
    gamma : first-order annealing rate [1/s]
    """
    return r * N_t * (1.0 - N_t / N_max) - gamma * N_t


def logistic_defect_exact(t: float, N0: float,
                          r: float, N_max: float,
                          gamma: float = 0.0) -> float:
    """Closed-form solution of the logistic defect ODE.
    For gamma = 0:
        N_t(t) = N0 N_max / (N_max + (N0 - N_max) exp(-r t))
              = N_max / (1 + ((N_max - N0)/N0) exp(-r t))
    For gamma != 0, let r_eff = r - gamma; the ODE becomes
        dN/dt = r_eff N - (r/N_max) N^2
    and the solution is logistic with rate r_eff and carrying capacity
        K_eff = N_max * r_eff / r
    provided r_eff > 0. If r_eff <= 0, defects monotonically decay.
    """
    r_eff = r - gamma
    if r_eff <= 0.0:
        # Pure decay regime
        return N0 * math.exp(r_eff * t)
    K_eff = N_max * r_eff / r
    denom = 1.0 + ((K_eff - N0) / max(N0, 1e-30)) * math.exp(-r_eff * t)
    return K_eff / denom


def integrate_logistic(N0: float, r: float, N_max: float,
                       gamma: float, t_end: float, n_steps: int = 500
                       ) -> Tuple[NDArray, NDArray]:
    """Integrate the logistic defect ODE via RK4 and return (t, N_t)."""
    t_arr = np.linspace(0.0, t_end, n_steps)
    dt = t_end / (n_steps - 1)
    N_arr = np.zeros(n_steps)
    N_arr[0] = N0
    for i in range(n_steps - 1):
        t = t_arr[i]
        N = N_arr[i]
        k1 = logistic_defect_deriv(t, N, r, N_max, gamma)
        k2 = logistic_defect_deriv(t + 0.5 * dt, N + 0.5 * dt * k1,
                                   r, N_max, gamma)
        k3 = logistic_defect_deriv(t + 0.5 * dt, N + 0.5 * dt * k2,
                                   r, N_max, gamma)
        k4 = logistic_defect_deriv(t + dt, N + dt * k3, r, N_max, gamma)
        N_arr[i + 1] = N + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # Enforce non-negativity and saturation
        N_arr[i + 1] = max(0.0, min(N_arr[i + 1], N_max))
    return t_arr, N_arr


# ============================================================================
# 2. Ion-migration pair interaction network (from 1023)
# ============================================================================
class DefectNetwork:
    """Track defect state evolution on a 1D lattice of sites where defects
    migrate and interact pairwise. Inspired by the covid-spread simulation
    but adapted to defect kinetics.

    Each site i has state s_i in {0=empty, 1=occupied by V_I,
                                  2=aggregated (V_I)_2, 3=trapped}.
    Migration: a defect at site i can hop to an adjacent empty site j.
    Aggregation: two adjacent occupied sites form a divacancy.
    Trapping: a divacancy captures an electron and becomes a non-radiative
              recombination center.
    """

    def __init__(self, n_sites: int, p_seed: float = 0.2,
                 seed: int = 283):
        self.n_sites = n_sites
        self.rng = np.random.default_rng(seed)
        # Initial state: random seeding with V_I
        self.state = np.zeros(n_sites, dtype=int)
        mask = self.rng.random(n_sites) < p_seed
        self.state[mask] = 1
        # Transition rates [1/s]
        self.k_hop = 1.0e3        # hop rate (migration)
        self.k_agg = 1.0e2        # aggregation rate
        self.k_trap = 1.0e1       # trapping rate
        self.k_anneal = 1.0       # annealing (defect removal)

    def step(self, dt: float) -> dict:
        """Advance one kinetic Monte Carlo step of duration dt. Returns
        a dict of population counts."""
        n = self.n_sites
        for i in range(n):
            # Hop: site i occupied -> pick neighbor
            if self.state[i] == 1 and self.rng.random() < self.k_hop * dt:
                # Choose left or right neighbor
                j = i + (1 if self.rng.random() < 0.5 else -1)
                if 0 <= j < n and self.state[j] == 0:
                    self.state[j] = 1
                    self.state[i] = 0
            # Aggregation: two adjacent occupied -> divacancy
            if self.state[i] == 1 and i + 1 < n and self.state[i + 1] == 1:
                if self.rng.random() < self.k_agg * dt:
                    self.state[i] = 2
                    self.state[i + 1] = 0
            # Trapping: divacancy -> trap
            if self.state[i] == 2 and self.rng.random() < self.k_trap * dt:
                self.state[i] = 3
            # Annealing: trap -> empty
            if self.state[i] == 3 and self.rng.random() < self.k_anneal * dt:
                self.state[i] = 0
        counts = {
            "empty": int(np.sum(self.state == 0)),
            "V_I": int(np.sum(self.state == 1)),
            "divacancy": int(np.sum(self.state == 2)),
            "trap": int(np.sum(self.state == 3)),
        }
        return counts

    def run(self, n_steps: int, dt: float) -> List[dict]:
        history = []
        for _ in range(n_steps):
            history.append(self.step(dt))
        return history


# ============================================================================
# 3. Chemical bond breaking (from 1025 pyrolysis)
# ============================================================================
def bond_population_kinetics(bond0: float, k_break: float,
                             t_end: float, n_steps: int = 200
                             ) -> Tuple[NDArray, NDArray]:
    """First-order bond-breaking kinetics: d[Bond]/dt = -k [Bond].
    Solution: [Bond](t) = bond0 * exp(-k t).
    Returns (t, bond). Used to model MAPbI3 thermal decomposition."""
    t = np.linspace(0.0, t_end, n_steps)
    bond = bond0 * np.exp(-k_break * t)
    return t, bond


def pyrolysis_product_evolution(bond0: float, k_break: float,
                                t_end: float, n_steps: int = 200
                                ) -> Tuple[NDArray, NDArray, NDArray]:
    """Track MAPbI3 -> PbI2 + CH3NH2 + HI decomposition.
    Returns (t, MAPbI3_remaining, PbI2_formed)."""
    t, bond = bond_population_kinetics(bond0, k_break, t_end, n_steps)
    pbi2 = bond0 - bond  # stoichiometry: 1:1 conversion
    return t, bond, pbi2


# ============================================================================
# 4. Optimal reaction-pathway ordering (from 739)
# ============================================================================
def reaction_pathway_cost(dims: List[int]) -> Tuple[int, List[int]]:
    """Compute the optimal (minimum scalar-operation) ordering for a chain
    of defect-species interactions. Each 'matrix' in the chain corresponds
    to a reaction-step transition operator of size dims[i] x dims[i+1].

    We use the classical dynamic-programming (matrix chain multiplication)
    algorithm. The minimum cost equals the minimum number of scalar
    multiplications to evaluate the full chain of defect-reaction operators.

    This is used in the multi-species defect kinetics solver to determine
    the optimal order in which to compose transition operators for
    V_I, I_i, Pb_i, and their complexes.
    """
    n = len(dims) - 1
    if n < 1:
        return 0, []
    # DP table
    m = [[0] * n for _ in range(n)]
    s = [[0] * n for _ in range(n)]
    for chain_len in range(2, n + 1):
        for i in range(n - chain_len + 1):
            j = i + chain_len - 1
            m[i][j] = math.inf
            for k in range(i, j):
                cost = m[i][k] + m[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1]
                if cost < m[i][j]:
                    m[i][j] = cost
                    s[i][j] = k
    # Reconstruct the ordering
    order = []

    def build_order(i: int, j: int):
        if i == j:
            order.append(i)
            return
        k = s[i][j]
        build_order(i, k)
        build_order(k + 1, j)

    build_order(0, n - 1)
    return m[0][n - 1], order


def catalan_number(n: int) -> int:
    """Return the n-th Catalan number C_n = (2n)! / (n+1)! n!.
    Counts the number of distinct parenthesizations of a chain of n+1
    matrices (equivalently, n multiplications)."""
    return math.factorial(2 * n) // (math.factorial(n + 1) * math.factorial(n))


# ============================================================================
# Driver: full kinetic simulation
# ============================================================================
def run_defect_kinetics(t_end_s: float = 1e4,
                        n_time_steps: int = 500,
                        n_sites: int = 64,
                        temperature_K: float = T_K) -> dict:
    """Run the full defect-kinetics simulation. Returns a dict of results."""
    # Activation energies for iodine-vacancy formation and migration
    E_a_form_J = 0.58 * E_CHARGE  # 0.58 eV
    E_a_mig_J = 0.25 * E_CHARGE   # 0.25 eV
    r_form = arrhenius_rate(1e10, E_a_form_J, temperature_K)
    k_mig = arrhenius_rate(1e12, E_a_mig_J, temperature_K)
    N_max = 1e24  # lattice-site density ~ 1/m^3
    N0 = DEFECT_DENSITY_DEFAULT * 0.01  # start at 1% saturation
    # Logistic generation
    t_log, N_t_log = integrate_logistic(N0, r_form, N_max, gamma=0.0,
                                        t_end=t_end_s, n_steps=n_time_steps)
    # Network simulation
    net = DefectNetwork(n_sites=n_sites, p_seed=0.2, seed=283)
    net.k_hop = k_mig
    dt_net = t_end_s / n_time_steps
    net_history = net.run(n_time_steps, dt_net)
    # Pyrolysis (high-temperature test at 500 K)
    k_break_high_T = arrhenius_rate(1e13, 1.2 * E_CHARGE, 500.0)
    t_pyro, bond, pbi2 = pyrolysis_product_evolution(
        bond0=1.0, k_break=k_break_high_T, t_end=t_end_s, n_steps=n_time_steps)
    # Reaction pathway for 4-species defect chain
    # [V_I, I_i, Pb_i, (V_I-I_i complex)]
    dims = [32, 16, 8, 4, 2]
    cost, order = reaction_pathway_cost(dims)
    return {
        "t_log": t_log,
        "N_t_log": N_t_log,
        "r_form": r_form,
        "N_max": N_max,
        "network_history": net_history,
        "t_pyro": t_pyro,
        "bond_remaining": bond,
        "pbi2_formed": pbi2,
        "reaction_pathway_cost": cost,
        "reaction_order": order,
        "catalan_n4": catalan_number(4),
    }
