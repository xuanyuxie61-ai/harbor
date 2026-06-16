# -*- coding: utf-8 -*-
"""
turbulence_mcmc.py
------------------
Markov Chain Monte Carlo sampling of magnetic-turbulence spectral
parameters from synthetic cosmic-ray observations.

The turbulence power spectrum is parametrised as a composite
slab + 2D model (Matthaeus et al. 1995):

    P(k) = C_slab k^{-q_slab} exp(-k l_c_slab)
           + C_2D k_perp^{-q_2D} exp(-k_perp l_c_2D)

The parameter vector  theta = (q_slab, q_2D, l_c, delta_B / B)
enters the CR transport via the quasi-linear diffusion coefficient.
Given synthetic observations of  f(r, mu)  we form the likelihood

    log L(theta) = -(1/2 sigma^2) sum_i (f_model(theta; r_i) - f_obs_i)^2

and sample with a Metropolis-Hastings random walk whose proposal
is a swap-style perturbation of pairs of parameters (inspired by
the gerrychain swap-proposal module): at each step two parameters
are selected and perturbed jointly.
"""
from __future__ import annotations
import math
import numpy as np
from typing import Callable, Tuple

import cosmic_ray_physics as crp
import heliocentric_mesh as hm


# =====================================================================
# Turbulence spectrum
# =====================================================================
def composite_spectrum(k: np.ndarray, q_slab: float, q_2d: float,
                       l_c: float, slab_fraction: float = 0.2,
                       delta_B_B: float = 0.3) -> np.ndarray:
    """Return  P(k)  for a slab + 2D composite turbulence spectrum.

    P(k) = delta_B^2 * [ slab_fraction k^{-q_slab} exp(-k l_c)
                        + (1 - slab_fraction) k^{-q_2D} exp(-k l_c) ]
    """
    if np.any(k <= 0.0):
        k = np.where(k > 0.0, k, 1.0e-30)
    term1 = slab_fraction * np.power(k, -q_slab) * np.exp(-k * l_c)
    term2 = (1.0 - slab_fraction) * np.power(k, -q_2d) * np.exp(-k * l_c)
    return (delta_B_B ** 2) * (term1 + term2)


def effective_kappa_from_spectrum(R: float, B: float, v: float,
                                  q_slab: float, q_2d: float,
                                  l_c: float,
                                  delta_B_B: float) -> float:
    """Integrate  kappa_|| = (v / 3 B^2) int P(k) R(k) dk  using a
    simple trapezoidal rule.
    """
    k = np.logspace(-9, -3, 64)
    Pk = composite_spectrum(k, q_slab, q_2d, l_c,
                            slab_fraction=0.2, delta_B_B=delta_B_B)
    # resonance  k = Omega / (gamma v mu) ; average over mu with weight
    Omega = crp.q_e * B / crp.m_p
    gamma = crp.lorentz_factor(crp.rigidity_to_kinetic(R))
    k_res = Omega / (gamma * v * 0.5)  # mu = 0.5 typical
    # approximate R(k) = (B^2 / Pk) * (1 / k)  at resonance
    Pk_res = np.interp(k_res, k, Pk)
    if Pk_res <= 0.0:
        return 1.0e22
    return (v / 3.0) * (B * B / max(Pk_res, 1.0e-30)) / k_res


# =====================================================================
# Synthetic likelihood
# =====================================================================
def synthetic_obs_turbulence(r: np.ndarray, mu: np.ndarray, Ek: float,
                             theta_true: Tuple[float, float, float,
                                               float],
                             seed: int = 5) -> np.ndarray:
    """Generate synthetic observations using the true turbulence
    parameters.
    """
    import focused_transport_eq as fte
    q_slab, q_2d, l_c, delta_B_B = theta_true
    B0 = crp.B_FIELD_1AU

    def kfunc(rr):
        return effective_kappa_from_spectrum(
            crp.kinetic_to_rigidity(Ek), crp.parker_imf(rr)[2],
            crp.velocity_from_Ek(Ek), q_slab, q_2d, l_c, delta_B_B)

    op = fte.FocusedTransportOperator(r, mu, Ek,
                                      kappa_par_func=kfunc,
                                      delta_B_B=delta_B_B,
                                      stencil_order="upwind2")
    f = np.ones((r.size, mu.size))
    hm.apply_all_boundaries(f, np.ones(mu.size), np.zeros(mu.size))
    dt = op.cfl_timestep() * 0.5
    n_steps = min(200, max(int(1.0 / max(dt, 1.0)), 20))
    for _ in range(n_steps):
        dfdt = op(f)
        f = f + dt * dfdt
        hm.apply_all_boundaries(f, np.ones(mu.size), np.zeros(mu.size))
    f_obs = f[:, mu.size // 2].copy()
    rng = np.random.default_rng(seed)
    f_obs *= np.exp(0.02 * rng.standard_normal(f_obs.size))
    return f_obs


def log_likelihood(theta: np.ndarray, r: np.ndarray, mu: np.ndarray,
                   Ek: float, f_obs: np.ndarray) -> float:
    """Log-likelihood  -(1/2 sigma^2) sum (f_model - f_obs)^2."""
    import focused_transport_eq as fte
    q_slab, q_2d, l_c, delta_B_B = theta
    if not (1.0 < q_slab < 3.0 and 1.0 < q_2d < 3.0):
        return -1.0e30
    if not (1.0e6 < l_c < 1.0e10):
        return -1.0e30
    if not (0.05 < delta_B_B < 1.5):
        return -1.0e30

    def kfunc(rr):
        return effective_kappa_from_spectrum(
            crp.kinetic_to_rigidity(Ek), crp.parker_imf(rr)[2],
            crp.velocity_from_Ek(Ek), q_slab, q_2d, l_c, delta_B_B)

    op = fte.FocusedTransportOperator(r, mu, Ek, kappa_par_func=kfunc,
                                      delta_B_B=delta_B_B,
                                      stencil_order="upwind2")
    f = np.ones((r.size, mu.size))
    hm.apply_all_boundaries(f, np.ones(mu.size), np.zeros(mu.size))
    dt = op.cfl_timestep() * 0.5
    for _ in range(80):
        dfdt = op(f)
        f = f + dt * dfdt
        hm.apply_all_boundaries(f, np.ones(mu.size), np.zeros(mu.size))
    residual = f[:, mu.size // 2] - f_obs
    sigma2 = 1.0e-2
    return -0.5 * float(np.sum(residual ** 2)) / sigma2


# =====================================================================
# MCMC sampler with swap-pair proposals
# =====================================================================
def run_mcmc(theta0: np.ndarray, r: np.ndarray, mu: np.ndarray,
             Ek: float, f_obs: np.ndarray,
             n_steps: int = 200, proposal_scale: float = 0.05,
             seed: int = 11) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run a Metropolis-Hastings chain with swap-pair proposals.

    At each step we choose 2 of the 4 parameters and perturb them
    jointly (swap proposal), mimicking the strategy used in
    redistricting MCMC samplers.

    Returns (chain, log_like_chain, acceptance_rate).
    """
    rng = np.random.default_rng(seed)
    d = theta0.size
    chain = np.zeros((n_steps, d))
    logl_chain = np.zeros(n_steps)
    theta_cur = theta0.copy()
    ll_cur = log_likelihood(theta_cur, r, mu, Ek, f_obs)
    accepts = 0
    for t in range(n_steps):
        # choose pair
        i, j = rng.choice(d, size=2, replace=False)
        theta_prop = theta_cur.copy()
        theta_prop[i] *= math.exp(proposal_scale * rng.standard_normal())
        theta_prop[j] *= math.exp(proposal_scale * rng.standard_normal())
        ll_prop = log_likelihood(theta_prop, r, mu, Ek, f_obs)
        alpha = min(1.0, math.exp(ll_prop - ll_cur))
        if rng.random() < alpha:
            theta_cur = theta_prop
            ll_cur = ll_prop
            accepts += 1
        chain[t] = theta_cur
        logl_chain[t] = ll_cur
    return chain, logl_chain, accepts / max(n_steps, 1)


# =====================================================================
# Demo
# =====================================================================
def _demo() -> None:
    r = hm.logarithmic_radial_mesh(Nr=10)
    mu = hm.pitch_angle_mesh(Nmu=6)
    Ek = 1.0e9 * crp.q_e
    theta_true = (1.7, 1.7, 1.0e8, 0.3)
    f_obs = synthetic_obs_turbulence(r, mu, Ek, theta_true, seed=5)
    theta0 = np.array([1.6, 1.8, 5.0e7, 0.28])
    chain, ll, acc = run_mcmc(theta0, r, mu, Ek, f_obs,
                              n_steps=5, proposal_scale=0.05, seed=1)
    print(f"[turbulence_mcmc] MCMC acceptance: {acc:.2f}, "
          f"final logL: {ll[-1]:.3e}")


if __name__ == "__main__":
    _demo()
