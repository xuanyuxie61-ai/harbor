"""
mcmc_optimization.py — MCMC calibration of damage model parameters.

Seeds: 1283_nesting_OH_WI (MCMC redistricting with swap proposals),
       170_chinese_remainder_theorem (CRT for multi-scale synchronization).

Core idea
=========
We calibrate the damage model parameters (characteristic length ℓ_c,
softening exponent n_s, damage threshold κ_0) by matching simulation
outputs to reference observations using Markov Chain Monte Carlo (MCMC).

The posterior distribution is:
    p(θ | d_obs) ∝ p(d_obs | θ) * p(θ)

where θ = (ℓ_c, n_s, κ_0) and the likelihood is:
    p(d_obs | θ) ∝ exp( -½ Σ_i (d_sim_i(θ) - d_obs_i)² / σ_i² )

MCMC algorithm (Metropolis-Hastings, adapted from seed 1283):
  1. Start from initial parameter guess θ_0
  2. Propose θ' = θ + ε, where ε ~ N(0, step_size² I)
  3. Compute acceptance ratio: α = min(1, p(θ'|d) / p(θ|d))
  4. Accept with probability α
  5. Repeat for chain_length steps

The proposal mechanism is analogous to the "swap proposal" in
redistricting MCMC (seed 1283): instead of swapping nodes between
districts, we swap parameter values between different scales.

Multi-scale synchronization (CRT, seed 170):
  The damage model operates at multiple temporal scales:
    - Fast: elastic wave propagation (Δt ~ 10⁻⁷ s)
    - Medium: damage evolution (Δt ~ 10⁻⁵ s)
    - Slow: crack propagation (Δt ~ 10⁻³ s)

  To synchronize these scales, we use the Chinese Remainder Theorem:
  if the three time scales have periods T_1, T_2, T_3 that are
  pairwise coprime integers (in units of the smallest Δt), then
  the combined simulation has a fundamental period:
      T = T_1 * T_2 * T_3
  and the scales are exactly synchronized every T steps.

  This avoids accumulated round-off errors from floating-point
  time-step ratios.
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from config import SimulationConfig


# ===================================================================
# CRT-based multi-scale synchronization  (seed 170)
# ===================================================================

def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """Extended Euclidean algorithm.

    Returns (g, x, y) such that a*x + b*y = g = gcd(a, b).
    """
    if a == 0:
        return b, 0, 1
    g, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return g, x, y


def chinese_remainder_theorem(moduli: List[int],
                              remainders: List[int]) -> Tuple[int, int]:
    """Chinese Remainder Theorem reconstruction.

    Given pairwise coprime moduli m_i and remainders r_i, find the
    unique integer x modulo M = ∏m_i such that:
        x ≡ r_i (mod m_i)  for all i

    Algorithm (directly from seed 170_chinese_remainder_theorem):
        M = ∏ m_i
        x = Σ r_i * (M/m_i) * inv(M/m_i, m_i)   (mod M)

    where inv(a, m) is the modular inverse of a modulo m.

    Returns (x, M).
    """
    n = len(moduli)
    assert n == len(remainders)

    # Check pairwise coprimality
    for i in range(n):
        for j in range(i + 1, n):
            g, _, _ = extended_gcd(moduli[i], moduli[j])
            if g != 1:
                raise ValueError(
                    f"Moduli {moduli[i]} and {moduli[j]} are not coprime (gcd={g})"
                )

    M = 1
    for m in moduli:
        M *= m

    x = 0
    for i in range(n):
        Mi = M // moduli[i]
        g, yi, _ = extended_gcd(Mi, moduli[i])
        x += remainders[i] * Mi * yi

    x = x % M
    return x, M


def compute_multiscale_sync_periods(dt_fast: int,
                                    dt_medium: int,
                                    dt_slow: int) -> Dict:
    """Compute multi-scale synchronization parameters using CRT.

    The three time-step sizes (in integer units of the smallest step)
    must be pairwise coprime for exact CRT synchronization.

    We find the smallest pairwise-coprime integers ≥ the given values.
    """
    def next_coprime_pair(a: int, b: int) -> Tuple[int, int]:
        """Find smallest a' ≥ a, b' ≥ b with gcd(a',b') = 1."""
        a_p, b_p = a, b
        while True:
            g, _, _ = extended_gcd(a_p, b_p)
            if g == 1:
                return a_p, b_p
            b_p += 1
            if b_p > b + 100:
                a_p += 1
                b_p = b

    # Make pairwise coprime
    m1, m2 = next_coprime_pair(dt_fast, dt_medium)
    m2, m3 = next_coprime_pair(m2, dt_slow)
    m1, m2 = next_coprime_pair(m1, m2)

    sync_period, M_total = chinese_remainder_theorem([m1, m2, m3], [0, 0, 0])

    return {
        "fast_period": m1,
        "medium_period": m2,
        "slow_period": m3,
        "sync_period": M_total,
        "fast_steps_per_sync": M_total // m1,
        "medium_steps_per_sync": M_total // m2,
        "slow_steps_per_sync": M_total // m3,
    }


# ===================================================================
# Log-likelihood and prior
# ===================================================================

def gaussian_log_likelihood(simulated: np.ndarray,
                            observed: np.ndarray,
                            noise_std: float) -> float:
    """Gaussian log-likelihood:  ln p(d|θ) = -½ Σ (d_sim - d_obs)² / σ².

    Plus normalization constant (omitted for MCMC as it cancels).
    """
    residuals = simulated - observed
    n = len(residuals)
    log_lik = -0.5 * np.sum(residuals ** 2) / (noise_std ** 2)
    log_lik -= 0.5 * n * math.log(2.0 * math.pi * noise_std ** 2)
    return float(log_lik)


def log_prior_uniform(theta: np.ndarray,
                      lower_bounds: np.ndarray,
                      upper_bounds: np.ndarray) -> float:
    """Uniform prior:  ln p(θ) = 0 if θ ∈ [lo, hi], else -∞."""
    if np.any(theta < lower_bounds) or np.any(theta > upper_bounds):
        return -np.inf
    return 0.0


def log_prior_gaussian(theta: np.ndarray,
                       prior_mean: np.ndarray,
                       prior_std: np.ndarray) -> float:
    """Gaussian prior:  ln p(θ) = -½ Σ (θ_i - μ_i)² / σ_i²."""
    return float(-0.5 * np.sum(((theta - prior_mean) / prior_std) ** 2))


# ===================================================================
# Forward model (simplified damage simulation)
# ===================================================================

def simplified_damage_forward(applied_strain: np.ndarray,
                              kappa_0: float,
                              lc: float,
                              n_soft: float) -> np.ndarray:
    """Simplified 1D damage model for MCMC calibration.

    Under uniaxial strain ε:
        κ = ε
        D(κ) = 1 - (κ_0/κ) * exp(-n_soft*(κ - κ_0)/(κ_c - κ_0))  for κ > κ_0
        σ = (1-D) * E * ε

    Returns stress as function of applied strain.
    """
    E = 30.0e9  # Pa
    kappa_c = kappa_0 * 10.0

    damage = np.zeros_like(applied_strain)
    mask = applied_strain > kappa_0
    if np.any(mask):
        beta = n_soft / max(kappa_c - kappa_0, 1.0e-15)
        kappa_m = applied_strain[mask]
        damage[mask] = 1.0 - (kappa_0 / kappa_m) * np.exp(-beta * (kappa_m - kappa_0))
    damage = np.clip(damage, 0.0, 1.0 - 1.0e-8)

    stress = (1.0 - damage) * E * applied_strain

    return stress


# ===================================================================
# MCMC Metropolis-Hastings  (seed 1283)
# ===================================================================

def mcmc_calibration(observed_stress: np.ndarray,
                     applied_strain: np.ndarray,
                     cfg: SimulationConfig,
                     initial_params: Optional[np.ndarray] = None,
                     noise_std: float = 1.0e5) -> Dict:
    """Run MCMC to calibrate damage model parameters.

    Parameters to calibrate: θ = (κ_0, ℓ_c, n_s)
    Observations: stress-strain curve

    The proposal distribution is Gaussian:
        θ' = θ + N(0, Σ_proposal)

    where Σ_proposal = diag(step_sizes²).

    Adaptation: after burn-in, we adjust step sizes to target
    a 35% acceptance rate (optimal for Gaussian targets in high dim).

    Returns:
      chain: (n_samples, 3) array of parameter samples
      acceptance_rate: fraction of accepted proposals
      MAP_estimate: maximum a posteriori parameters
      credible_intervals: 95% credible intervals for each parameter
    """
    n_params = 3
    chain_length = cfg.numerical.mcmc_chain_length
    burnin = cfg.numerical.mcmc_burnin
    step_size = cfg.numerical.mcmc_step_size

    # Parameter bounds
    lower = np.array([1.0e-5, 0.01, 0.5])
    upper = np.array([1.0e-3, 0.20, 5.0])
    prior_mean = np.array([cfg.material.damage_threshold_strain,
                           cfg.material.characteristic_length,
                           cfg.material.softening_exponent])
    prior_std = np.array([5.0e-5, 0.02, 0.5])

    # Initial state
    if initial_params is None:
        theta = prior_mean.copy()
    else:
        theta = initial_params.copy()

    # Step sizes for each parameter (normalized to parameter scale)
    steps = step_size * (upper - lower)

    # Storage
    chain = np.zeros((chain_length, n_params))
    log_likelihoods = np.zeros(chain_length)

    # Initial log-posterior
    sim = simplified_damage_forward(applied_strain, theta[0], theta[1], theta[2])
    current_ll = gaussian_log_likelihood(sim, observed_stress, noise_std)
    current_lp = log_prior_gaussian(theta, prior_mean, prior_std)
    current_log_post = current_ll + current_lp

    n_accepted = 0
    rng = np.random.RandomState(42)

    for i in range(chain_length):
        # Propose
        theta_prop = theta + steps * rng.randn(n_params)

        # Check bounds
        if np.any(theta_prop < lower) or np.any(theta_prop > upper):
            chain[i] = theta
            log_likelihoods[i] = current_ll
            continue

        # Compute log-posterior at proposal
        sim_prop = simplified_damage_forward(applied_strain,
                                             theta_prop[0], theta_prop[1], theta_prop[2])
        prop_ll = gaussian_log_likelihood(sim_prop, observed_stress, noise_std)
        prop_lp = log_prior_gaussian(theta_prop, prior_mean, prior_std)
        prop_log_post = prop_ll + prop_lp

        # Accept/reject
        log_alpha = prop_log_post - current_log_post
        if math.log(max(rng.random(), 1.0e-300)) < log_alpha:
            theta = theta_prop
            current_ll = prop_ll
            current_log_post = prop_log_post
            n_accepted += 1

        chain[i] = theta
        log_likelihoods[i] = current_ll

        # Adaptive step size (during burn-in only)
        if i < burnin and (i + 1) % 50 == 0:
            recent_rate = n_accepted / max(i + 1, 1)
            if recent_rate < 0.2:
                steps *= 0.8
            elif recent_rate > 0.5:
                steps *= 1.2

    acceptance_rate = n_accepted / chain_length
    post_burnin = chain[burnin:]

    # MAP estimate
    map_idx = burnin + np.argmax(log_likelihoods[burnin:])
    map_estimate = chain[map_idx]

    # 95% credible intervals
    ci_lower = np.percentile(post_burnin, 2.5, axis=0)
    ci_upper = np.percentile(post_burnin, 97.5, axis=0)

    return {
        "chain": chain,
        "log_likelihoods": log_likelihoods,
        "acceptance_rate": float(acceptance_rate),
        "MAP_estimate": map_estimate,
        "credible_lower": ci_lower,
        "credible_upper": ci_upper,
        "posterior_mean": np.mean(post_burnin, axis=0),
        "posterior_std": np.std(post_burnin, axis=0),
        "param_names": ["kappa_0", "l_c", "n_soft"],
    }


# ===================================================================
# Synthetic observation generation
# ===================================================================

def generate_synthetic_observations(cfg: SimulationConfig,
                                    n_strain_points: int = 30,
                                    noise_level: float = 0.02
                                    ) -> Dict:
    """Generate synthetic stress-strain observations for calibration.

    Uses the default parameters with added Gaussian noise.
    """
    mat = cfg.material
    max_strain = cfg.loading.max_applied_strain
    applied_strain = np.linspace(0, max_strain, n_strain_points)

    # "True" stress
    true_stress = simplified_damage_forward(
        applied_strain,
        mat.damage_threshold_strain,
        mat.characteristic_length,
        mat.softening_exponent
    )

    # Add noise
    rng = np.random.RandomState(123)
    noise = noise_level * mat.tensile_strength * rng.randn(n_strain_points)
    observed_stress = true_stress + noise
    observed_stress = np.maximum(observed_stress, 0.0)

    return {
        "applied_strain": applied_strain,
        "observed_stress": observed_stress,
        "true_stress": true_stress,
        "noise_std": noise_level * mat.tensile_strength,
    }
