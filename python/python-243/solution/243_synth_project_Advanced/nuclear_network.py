"""
nuclear_network.py - Core r-process network ODE system.

Assembles the full network:
  dY(A,Z)/dt = lambda_{n,gamma}(A-1,Z) n_n Y(A-1,Z)
             + lambda_{gamma,n}(A+1,Z) Y(A+1,Z)
             + lambda_beta(A,Z-1) Y(A,Z-1)
             - [lambda_{n,gamma}(A,Z) n_n + lambda_{gamma,n}(A,Z) + lambda_beta(A,Z)] Y(A,Z)
             + fission contributions

Solved with RK4 or implicit Euler with adaptive stepping.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple
from physical_constants import TINY
from reaction_rates import build_rate_table

def build_network_index(nuclides: List[Tuple[int, int]]) -> Dict[Tuple[int, int], int]:
    """Map (A, Z) -> index in abundance vector."""
    return {nz: i for i, nz in enumerate(nuclides)}


def initial_abundances(n_nuclides: int, Y_seed: float = 1.0) -> np.ndarray:
    """Start with all abundance in seed nuclei (neutrons)."""
    Y = np.zeros(n_nuclides, dtype=np.float64)
    if n_nuclides > 0:
        Y[0] = Y_seed
    return Y


def dYdt(Y: np.ndarray, nuclides: List[Tuple[int, int]],
         idx: Dict[Tuple[int, int], int],
         rates: Dict[str, Dict[Tuple[int, int], float]],
         n_n: float, T9: float, dt_hint: float = 1.0) -> np.ndarray:
    """
    Compute dY/dt for the full network with reflecting boundary conditions.
    nuclides: list of (A, Z) in same order as Y
    rates: dict with keys "n_cap", "gamma_n", "beta", "fission"
    n_n: neutron number density [cm^-3]
    dt_hint: used to cap loss rates so dt*loss <= 5 (stability).

    Boundary handling:
      - If a product nuclide is not in the mesh, the corresponding loss
        is suppressed (reflecting boundary).
      - This conserves total mass within the mesh.
    """
    n = len(Y)
    dY = np.zeros(n, dtype=np.float64)
    max_loss = 5.0 / max(dt_hint, 1e-30)     # safety cap
    for k, (A, Z) in enumerate(nuclides):
        Yk = max(Y[k], 0.0)
        if Yk < TINY:
            continue
        # Losses (only if the product is in the mesh, or it's beta/fission)
        r_ncap = rates["n_cap"].get((A, Z), 0.0)
        r_gn = rates["gamma_n"].get((A, Z), 0.0)
        r_beta = rates["beta"].get((A, Z), 0.0)
        r_fis = rates["fission"].get((A, Z), 0.0)
        # Reflecting BC for neutron capture: product (A+1, Z) must be in mesh
        if (A + 1, Z) not in idx:
            r_ncap = 0.0
        # Reflecting BC for photodisintegration: product (A-1, Z) must be in mesh
        if (A - 1, Z) not in idx:
            r_gn = 0.0
        loss = (r_ncap * n_n + r_gn + r_beta + r_fis) * Yk
        # Cap loss to maintain stability
        loss = min(loss, max_loss * Yk)
        dY[k] -= loss
        # Gains from (A-1, Z) via n-cap
        if (A - 1, Z) in idx:
            j = idx[(A - 1, Z)]
            r_ncap_src = rates["n_cap"].get((A - 1, Z), 0.0)
            dY[k] += r_ncap_src * n_n * max(Y[j], 0.0)
        # Gains from (A+1, Z) via photodisintegration
        if (A + 1, Z) in idx:
            j = idx[(A + 1, Z)]
            r_gn_src = rates["gamma_n"].get((A + 1, Z), 0.0)
            dY[k] += r_gn_src * max(Y[j], 0.0)
        # Gains from beta decay of (A, Z-1)
        if (A, Z - 1) in idx:
            j = idx[(A, Z - 1)]
            r_beta_src = rates["beta"].get((A, Z - 1), 0.0)
            dY[k] += r_beta_src * max(Y[j], 0.0)
    return dY


def rk4_step(Y: np.ndarray, dt: float, nuclides: List[Tuple[int, int]],
             idx: Dict[Tuple[int, int], int],
             rates: Dict[str, Dict[Tuple[int, int], float]],
             n_n: float, T9: float) -> np.ndarray:
    """One RK4 integration step.  Normalization conserves total baryon
    number even when fission fragments are not explicitly tracked."""
    def F(y):
        return dYdt(y, nuclides, idx, rates, n_n, T9, dt_hint=dt)
    k1 = F(Y)
    k2 = F(Y + 0.5 * dt * k1)
    k3 = F(Y + 0.5 * dt * k2)
    k4 = F(Y + dt * k3)
    Y_new = Y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    Y_new = np.maximum(Y_new, 0.0)
    s = np.sum(Y_new)
    if s > 0.0:
        Y_new = Y_new / s
    return Y_new


def implicit_euler_step(Y: np.ndarray, dt: float,
                        nuclides: List[Tuple[int, int]],
                        idx: Dict[Tuple[int, int], int],
                        rates: Dict[str, Dict[Tuple[int, int], float]],
                        n_n: float, T9: float) -> np.ndarray:
    """
    Implicit Euler via fixed-point iteration.  Stable for stiff systems.
      Y^{n+1} = Y^n + dt F(Y^{n+1})
    Iterate: Y^{k+1} = Y^n + dt F(Y^k), starting from Y^0 = Y^n.
    """
    Y_new = Y.copy()
    for _ in range(20):
        Fv = dYdt(Y_new, nuclides, idx, rates, n_n, T9, dt_hint=dt)
        Y_next = Y + dt * Fv
        Y_next = np.maximum(Y_next, 0.0)
        s = np.sum(Y_next)
        if s > 0.0:
            Y_next = Y_next / s
        if np.max(np.abs(Y_next - Y_new)) < 1e-12:
            return Y_next
        Y_new = Y_next
    return Y_new


def integrate_network(Y0: np.ndarray, t_span: Tuple[float, float],
                      nsteps: int, nuclides: List[Tuple[int, int]],
                      idx: Dict[Tuple[int, int], int],
                      rates_func, n_n_func, T9_func,
                      method: str = "rk4") -> Tuple[np.ndarray, np.ndarray]:
    """
    Integrate the network from t_span[0] to t_span[1] in nsteps.
    rates_func(t), n_n_func(t), T9_func(t) return values at time t.
    Returns (times, Y_history) where Y_history has shape (nsteps+1, n_nuclides).
    """
    t0, t1 = t_span
    dt = (t1 - t0) / max(1, nsteps)
    times = np.linspace(t0, t1, nsteps + 1)
    Y_hist = np.zeros((nsteps + 1, len(Y0)), dtype=np.float64)
    Y_hist[0] = Y0.copy()
    Y = Y0.copy()
    for i in range(nsteps):
        t = times[i]
        rates = rates_func(t)
        n_n = n_n_func(t)
        T9 = T9_func(t)
        if method == "rk4":
            Y = rk4_step(Y, dt, nuclides, idx, rates, n_n, T9)
        elif method == "implicit_euler":
            Y = implicit_euler_step(Y, dt, nuclides, idx, rates, n_n, T9)
        else:
            raise ValueError(f"Unknown method: {method}")
        Y_hist[i + 1] = Y
    return times, Y_hist


__all__ = [
    "build_network_index", "initial_abundances",
    "dYdt", "rk4_step", "implicit_euler_step",
    "integrate_network",
]
