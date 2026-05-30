# -*- coding: utf-8 -*-

import numpy as np


def gelman_rubin_diagnostic(chains):
    n_chains, n_samples, n_params = chains.shape
    if n_samples < 2:
        return np.full(n_params, np.inf)


    chain_means = np.mean(chains, axis=1)

    overall_mean = np.mean(chain_means, axis=0)


    B = n_samples * np.var(chain_means, axis=0, ddof=1)


    W = np.mean(np.var(chains, axis=1, ddof=1), axis=0)


    var_estimate = ((n_samples - 1) / n_samples) * W + B / n_samples


    R_hat = np.sqrt(var_estimate / np.where(W < 1e-30, 1e-30, W))
    return R_hat


def log_prior_lognormal(theta, mu_ln, sigma_ln):
    theta = np.asarray(theta, dtype=float)
    if np.any(theta <= 0):
        return -np.inf
    log_pi = -0.5 * np.sum(((np.log(theta) - mu_ln) / sigma_ln) ** 2)
    log_pi -= np.sum(np.log(theta))

    return log_pi


def log_likelihood_gaussian(theta, model_func, data, sigma_noise):
    try:
        predictions = model_func(theta)
        predictions = np.asarray(predictions, dtype=float)
        residuals = data - predictions
        n = data.size
        log_L = -0.5 * n * np.log(2.0 * np.pi * sigma_noise ** 2)
        log_L -= 0.5 * np.sum(residuals ** 2) / (sigma_noise ** 2)
        return log_L
    except Exception:
        return -np.inf


def dream_mcmc(log_posterior, n_params, n_chains=3, n_generations=2000,
               bounds=None, init_scale=0.1, rng=None,
               gr_threshold=1.01, burnin_fraction=0.5,
               de_pairs=3, gamma_base=2.38 / np.sqrt(2)):
    if rng is None:
        rng = np.random.default_rng()
    if bounds is None:
        bounds = np.zeros((2, n_params))
        bounds[0, :] = 1e-12
        bounds[1, :] = 1e6

    n_chains = max(n_chains, 3)
    n_burnin = int(n_generations * burnin_fraction)
    n_keep = n_generations - n_burnin


    chains = np.zeros((n_chains, n_generations, n_params), dtype=float)
    logpost_values = np.zeros((n_chains, n_generations), dtype=float)

    for c in range(n_chains):

        log_low = np.log(bounds[0, :])
        log_high = np.log(bounds[1, :])
        chains[c, 0, :] = np.exp(log_low + rng.random(n_params) * (log_high - log_low))
        logpost_values[c, 0] = log_posterior(chains[c, 0, :])

    accepted = 0
    total_proposals = 0

    max_pairs = (n_chains - 1) // 2
    actual_pairs = min(de_pairs, max_pairs)
    for gen in range(1, n_generations):
        for c in range(n_chains):


            other_chains = [i for i in range(n_chains) if i != c]
            selected = rng.choice(other_chains, size=2 * actual_pairs, replace=False)

            diff_sum = np.zeros(n_params)
            for p in range(actual_pairs):
                a = chains[selected[2 * p], gen - 1, :]
                b = chains[selected[2 * p + 1], gen - 1, :]
                diff_sum += (a - b)


            gamma = gamma_base

            if rng.random() < 0.1:
                gamma = 1.0


            n_update = rng.integers(1, n_params + 1)
            update_dims = rng.choice(n_params, size=n_update, replace=False)

            proposal = chains[c, gen - 1, :].copy()
            eta = rng.uniform(-0.1, 0.1, n_update)
            epsilon = rng.normal(0, 1e-10, n_update)

            proposal[update_dims] += ((1.0 + eta) * gamma * diff_sum[update_dims] + epsilon)


            for j in range(n_params):
                if proposal[j] < bounds[0, j]:
                    proposal[j] = bounds[0, j] + (bounds[0, j] - proposal[j])
                    if proposal[j] < bounds[0, j]:
                        proposal[j] = bounds[0, j]
                elif proposal[j] > bounds[1, j]:
                    proposal[j] = bounds[1, j] - (proposal[j] - bounds[1, j])
                    if proposal[j] > bounds[1, j]:
                        proposal[j] = bounds[1, j]


            proposal = np.maximum(proposal, bounds[0, :])
            proposal = np.minimum(proposal, bounds[1, :])


            logpost_prop = log_posterior(proposal)
            logpost_curr = logpost_values[c, gen - 1]


            log_alpha = logpost_prop - logpost_curr
            if np.isnan(log_alpha):
                log_alpha = -np.inf

            if np.log(rng.random()) < log_alpha:
                chains[c, gen, :] = proposal
                logpost_values[c, gen] = logpost_prop
                accepted += 1
            else:
                chains[c, gen, :] = chains[c, gen - 1, :]
                logpost_values[c, gen] = logpost_curr
            total_proposals += 1


    samples = chains[:, n_burnin:, :]
    logpost_keep = logpost_values[:, n_burnin:]


    R_hat = gelman_rubin_diagnostic(samples)

    acceptance_rate = accepted / max(total_proposals, 1)

    return samples, logpost_keep, R_hat, acceptance_rate


def estimate_parameters_summary(samples):
    flat = samples.reshape(-1, samples.shape[-1])
    mean = np.mean(flat, axis=0)
    median = np.median(flat, axis=0)
    std = np.std(flat, axis=0, ddof=1)
    ci_95 = np.percentile(flat, [2.5, 97.5], axis=0)

    return {
        'mean': mean,
        'median': median,
        'std': std,
        'ci_lower': ci_95[0, :],
        'ci_upper': ci_95[1, :]
    }
