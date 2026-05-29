# -*- coding: utf-8 -*-
"""
================================================================================
Stochastic Synaptic Weight Evolution Module
================================================================================

This module models synaptic weight fluctuations using stochastic differential
equations (SDEs) inspired by the Black-Scholes framework, adapted for
neurobiological plasticity.

Biological Motivation:
----------------------
Synaptic weights are subject to continuous stochastic fluctuations due to:
1. Spontaneous vesicle release
2. Protein turnover noise
3. Thermal fluctuations of ion channels
4. Variable spine volume

These fluctuations can be modeled as multiplicative noise since the
relative magnitude of fluctuations scales with weight.

Mathematical Model:
-------------------
The synaptic weight W(t) follows a generalized geometric Brownian motion
with Hebbian drift and homeostatic mean-reversion:

    dW = [μ·W·(1 - W/W_max) - λ·(W - W_target)] dt + σ·W dB_t

where:
    μ     = Hebbian growth rate
    W_max = maximum weight (saturation)
    λ     = homeostatic mean-reversion rate
    W_target = target weight
    σ     = volatility (noise intensity)
    B_t   = standard Brownian motion

The term μ·W·(1 - W/W_max) is logistic growth modeling LTP saturation.
The term -λ·(W - W_target) is Ornstein-Uhlenbeck mean-reversion.

Fokker-Planck Equation:
-----------------------
The probability density p(w,t) satisfies:

    ∂p/∂t = -∂/∂w [A(w)·p] + (1/2)·∂²/∂w² [B(w)·p]

where:
    A(w) = μ·w·(1 - w/W_max) - λ·(w - W_target)   [drift]
    B(w) = σ²·w²                                     [diffusion coefficient]

Stationary Distribution:
------------------------
For the simplified model dW = λ·(W_target - W)dt + σ·W dB_t,
the stationary distribution is inverse-gamma-like.

The general stationary solution is:

    p_s(w) = C · exp(2·∫ A(w)/B(w) dw) / B(w)

Numerical Integration (Euler-Maruyama):
---------------------------------------
    W_{n+1} = W_n + A(W_n)·Δt + σ·W_n·√Δt·Z_n

where Z_n ~ N(0,1) are i.i.d. standard normals.

Black-Scholes Connection:
-------------------------
The classic Black-Scholes model for option pricing is:

    dS = μ·S·dt + σ·S·dB_t

with solution:
    S(t) = S_0 · exp((μ - σ²/2)·t + σ·B_t)

For synaptic weights, we add biological constraints (saturation,
mean-reversion) to prevent unphysical values.

================================================================================
"""

import numpy as np
from typing import Tuple, Optional


def sde_drift(
    w: np.ndarray,
    mu: float,
    w_max: float,
    lambda_homeo: float,
    w_target: float,
) -> np.ndarray:
    """
    Compute the drift term A(w) for the synaptic weight SDE.

    A(w) = μ·w·(1 - w/W_max) - λ·(w - W_target)

    Parameters
    ----------
    w : np.ndarray
        Current weights.
    mu : float
        Hebbian growth rate.
    w_max : float
        Maximum weight.
    lambda_homeo : float
        Homeostatic rate.
    w_target : float
        Target weight.

    Returns
    -------
    drift : np.ndarray
        Drift values.
    """
    if w_max <= 0.0:
        raise ValueError("w_max must be positive.")

    # Logistic growth term
    logistic = mu * w * (1.0 - w / w_max)

    # Mean-reversion term
    mean_rev = -lambda_homeo * (w - w_target)

    return logistic + mean_rev


def sde_diffusion(
    w: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """
    Compute the diffusion coefficient B(w) = σ²·w².

    Parameters
    ----------
    w : np.ndarray
        Current weights.
    sigma : float
        Volatility. Must be non-negative.

    Returns
    -------
    diff : np.ndarray
        Diffusion coefficients.
    """
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative.")
    return (sigma ** 2) * (w ** 2)


def euler_maruyama_step(
    w: np.ndarray,
    dt: float,
    mu: float,
    w_max: float,
    lambda_homeo: float,
    w_target: float,
    sigma: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Perform one Euler-Maruyama step for the synaptic weight SDE.

    W_{n+1} = W_n + A(W_n)·dt + σ·W_n·√dt·Z

    Parameters
    ----------
    w : np.ndarray
        Current weights.
    dt : float
        Time step. Must be positive.
    mu, w_max, lambda_homeo, w_target, sigma : float
        SDE parameters.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    w_new : np.ndarray
        Updated weights.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if rng is None:
        rng = np.random.default_rng()

    drift = sde_drift(w, mu, w_max, lambda_homeo, w_target)
    noise = sigma * w * np.sqrt(dt) * rng.standard_normal(w.shape)

    w_new = w + drift * dt + noise

    # Enforce physical bounds
    w_new = np.clip(w_new, 1e-6, w_max)

    return w_new


def simulate_stochastic_weights(
    n_synapses: int = 100,
    t_final: float = 100.0,
    dt: float = 0.01,
    w0: Optional[np.ndarray] = None,
    mu: float = 0.05,
    w_max: float = 1.0,
    lambda_homeo: float = 0.1,
    w_target: float = 0.5,
    sigma: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate stochastic evolution of synaptic weights.

    Parameters
    ----------
    n_synapses : int
        Number of synapses.
    t_final : float
        Final time.
    dt : float
        Time step.
    w0 : np.ndarray, optional
        Initial weights.
    mu : float
        Hebbian growth rate.
    w_max : float
        Maximum weight.
    lambda_homeo : float
        Homeostatic rate.
    w_target : float
        Target weight.
    sigma : float
        Volatility.
    seed : int
        Random seed.

    Returns
    -------
    t : np.ndarray
        Time points.
    w_history : np.ndarray
        Weight history, shape (n_steps+1, n_synapses).
    """
    if n_synapses < 1:
        raise ValueError("n_synapses must be >= 1.")
    if dt <= 0.0 or t_final <= 0.0:
        raise ValueError("dt and t_final must be positive.")

    rng = np.random.default_rng(seed)

    if w0 is None:
        w0 = rng.uniform(0.1, 0.9, n_synapses)
    else:
        w0 = np.asarray(w0).copy()

    n_steps = int(np.ceil(t_final / dt))
    t = np.linspace(0.0, t_final, n_steps + 1)
    w_history = np.zeros((n_steps + 1, n_synapses))
    w_history[0, :] = w0

    w = w0.copy()
    for step in range(n_steps):
        w = euler_maruyama_step(
            w, dt, mu, w_max, lambda_homeo, w_target, sigma, rng
        )
        w_history[step + 1, :] = w

    return t, w_history


def black_scholes_synaptic_option(
    w0: float,
    w_target: float,
    mu: float,
    sigma: float,
    tau: float,
) -> float:
    """
    Compute the "plasticity option value" inspired by Black-Scholes.

    The probability that a synaptic weight exceeds target at time τ:

        P(W_τ > W_target) = N(d₁)

    where:
        d₁ = [ln(w₀/W_target) + (μ + σ²/2)·τ] / (σ·√τ)
        d₂ = d₁ - σ·√τ

    and N(·) is the standard normal CDF.

    The expected value above target:
        E[max(W_τ - W_target, 0)] = w₀·N(d₁) - W_target·N(d₂)

    Parameters
    ----------
    w0 : float
        Initial weight.
    w_target : float
        Target weight (strike).
    mu : float
        Drift rate.
    sigma : float
        Volatility.
    tau : float
        Time to maturity.

    Returns
    -------
    option_value : float
        Expected plasticity "payoff".
    """
    if w0 <= 0.0 or w_target <= 0.0:
        raise ValueError("Weights must be positive.")
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative.")
    if tau <= 0.0:
        return max(w0 - w_target, 0.0)

    from scipy.stats import norm

    d1 = (np.log(w0 / w_target) + (mu + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    n1 = norm.cdf(d1)
    n2 = norm.cdf(d2)

    option_value = w0 * n1 - w_target * np.exp(-mu * tau) * n2
    return max(option_value, 0.0)


def compute_weight_statistics(
    w_history: np.ndarray,
    w_target: float = 0.5,
) -> dict:
    """
    Compute statistics of the weight distribution over time.

    Parameters
    ----------
    w_history : np.ndarray
        Weight history, shape (n_steps+1, n_synapses).
    w_target : float
        Target weight.

    Returns
    -------
    stats : dict
        Distribution statistics.
    """
    mean_w = np.mean(w_history, axis=1)
    std_w = np.std(w_history, axis=1)
    cv = std_w / (mean_w + 1e-15)

    # Fraction above target
    frac_above = np.mean(w_history > w_target, axis=1)

    # Entropy of weight distribution (binned)
    n_bins = 20
    entropies = []
    for step in range(w_history.shape[0]):
        hist, _ = np.histogram(w_history[step, :], bins=n_bins, range=(0.0, 1.0), density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log(hist + 1e-15)) * (1.0 / n_bins)
        entropies.append(entropy)

    return {
        "mean": mean_w,
        "std": std_w,
        "cv": cv,
        "fraction_above_target": frac_above,
        "entropy": np.array(entropies),
        "final_mean": mean_w[-1],
        "final_std": std_w[-1],
        "final_cv": cv[-1],
    }


def simulate_plasticity_option_portfolio(
    n_synapses: int = 50,
    tau: float = 10.0,
) -> dict:
    """
    Evaluate a portfolio of synaptic plasticity "options".

    Parameters
    ----------
    n_synapses : int
        Number of synapses.
    tau : float
        Time horizon.

    Returns
    -------
    results : dict
        Portfolio statistics.
    """
    rng = np.random.default_rng(130)

    w0 = rng.uniform(0.1, 0.9, n_synapses)
    mu = rng.uniform(0.01, 0.1, n_synapses)
    sigma = rng.uniform(0.1, 0.3, n_synapses)
    w_target = 0.5

    options = np.array([
        black_scholes_synaptic_option(w0[i], w_target, mu[i], sigma[i], tau)
        for i in range(n_synapses)
    ])

    return {
        "options": options,
        "total_value": np.sum(options),
        "mean_value": np.mean(options),
        "w0": w0,
        "mu": mu,
        "sigma": sigma,
    }


if __name__ == "__main__":
    t, w_hist = simulate_stochastic_weights(n_synapses=50, t_final=50.0)
    stats = compute_weight_statistics(w_hist)
    print(f"Final mean weight: {stats['final_mean']:.4f}")
    print(f"Final CV: {stats['final_cv']:.4f}")

    portfolio = simulate_plasticity_option_portfolio()
    print(f"Total portfolio value: {portfolio['total_value']:.4f}")
