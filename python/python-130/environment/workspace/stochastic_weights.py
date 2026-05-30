# -*- coding: utf-8 -*-

import numpy as np
from typing import Tuple, Optional


def sde_drift(
    w: np.ndarray,
    mu: float,
    w_max: float,
    lambda_homeo: float,
    w_target: float,
) -> np.ndarray:
    if w_max <= 0.0:
        raise ValueError("w_max must be positive.")


    logistic = mu * w * (1.0 - w / w_max)


    mean_rev = -lambda_homeo * (w - w_target)

    return logistic + mean_rev


def sde_diffusion(
    w: np.ndarray,
    sigma: float,
) -> np.ndarray:
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
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if rng is None:
        rng = np.random.default_rng()

    drift = sde_drift(w, mu, w_max, lambda_homeo, w_target)
    noise = sigma * w * np.sqrt(dt) * rng.standard_normal(w.shape)

    w_new = w + drift * dt + noise


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
    mean_w = np.mean(w_history, axis=1)
    std_w = np.std(w_history, axis=1)
    cv = std_w / (mean_w + 1e-15)


    frac_above = np.mean(w_history > w_target, axis=1)


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
