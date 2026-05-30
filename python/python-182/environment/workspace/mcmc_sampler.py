import numpy as np
import math


def run_adaptive_mcmc(log_posterior, init: np.ndarray, rng,
                      n_iter: int = 200, proposal_scale: float = 0.03,
                      rotation_period: int = 10):
    init = np.asarray(init, dtype=float)
    n_params = len(init)
    chain = np.empty((n_iter + 1, n_params), dtype=float)
    logpost = np.empty(n_iter + 1, dtype=float)
    chain[0, :] = init
    lp_current = log_posterior(init)
    if not np.isfinite(lp_current):
        raise ValueError("Initial state has non-finite log-posterior")
    logpost[0] = lp_current

    accepted = 0
    n_cont = n_params - 1

    for it in range(1, n_iter + 1):
        x = chain[it - 1, :].copy()


        x_prop = x.copy()
        for j in range(n_cont):
            x_prop[j] += proposal_scale * rng.normal(0.0, 1.0)

        lp_prop = log_posterior(x_prop)
        if np.isfinite(lp_prop):
            alpha = min(1.0, math.exp(lp_prop - lp_current))
            if rng.uniform() < alpha:
                x = x_prop
                lp_current = lp_prop
                accepted += 1





        j = n_cont
        x_prop_gibbs = x.copy()
        x_prop_gibbs[j] += 0.05 * rng.normal(0.0, 1.0)
        lp_prop_gibbs = log_posterior(x_prop_gibbs)
        if np.isfinite(lp_prop_gibbs):
            alpha_g = min(1.0, math.exp(lp_prop_gibbs - lp_current))
            if rng.uniform() < alpha_g:
                x = x_prop_gibbs
                lp_current = lp_prop_gibbs
                accepted += 1


        if rotation_period > 0 and it % rotation_period == 0:
            shift = int(rng.uniform() * 3) + 1
            x_rot = x.copy()
            c_rot = np.roll(x_rot[4:8], shift)
            x_rot[4:8] = c_rot
            lp_rot = log_posterior(x_rot)
            if np.isfinite(lp_rot):
                alpha_r = min(1.0, math.exp(lp_rot - lp_current))
                if rng.uniform() < alpha_r:
                    x = x_rot
                    lp_current = lp_rot
                    accepted += 1

        chain[it, :] = x
        logpost[it] = lp_current

    accept_rate = accepted / (n_iter * 2.0 + n_iter // rotation_period) if rotation_period > 0 else accepted / (n_iter * 2.0)
    return chain, logpost, accept_rate
