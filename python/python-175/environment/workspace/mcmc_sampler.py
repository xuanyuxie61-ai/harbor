
import numpy as np


def gelman_rubin_r_hat(chains):
    n_chains, n_samples, n_params = chains.shape
    if n_samples < 2:
        return np.full(n_params, np.inf)

    chain_means = np.mean(chains, axis=1)
    chain_vars = np.var(chains, axis=1, ddof=1)
    W = np.mean(chain_vars, axis=0)

    global_mean = np.mean(chain_means, axis=0)
    B = n_samples * np.var(chain_means, axis=0, ddof=1)

    var_hat = (n_samples - 1.0) / n_samples * W + B / n_samples
    r_hat = np.sqrt(var_hat / W)
    r_hat = np.where(np.isnan(r_hat), np.inf, r_hat)
    return r_hat


def dream_mcmc(log_posterior, bounds, n_chains=4, n_samples=2000, burn_in=500,
               delta_init=3, c=0.1, b_star=1e-6, adapt_cr=True):
    n_params = bounds.shape[0]
    total_iter = n_samples + burn_in


    chains = np.zeros((n_chains, total_iter, n_params))
    log_probs = np.zeros((n_chains, total_iter))
    for i in range(n_chains):
        chains[i, 0] = np.random.uniform(bounds[:, 0], bounds[:, 1])
        log_probs[i, 0] = log_posterior(chains[i, 0])


    cr_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    n_cr = len(cr_values)
    cr_counts = np.ones(n_cr)
    cr_accepts = np.ones(n_cr)
    cr_prob = cr_counts / np.sum(cr_counts)

    n_accept = np.zeros(n_chains)

    for t in range(total_iter - 1):
        for i in range(n_chains):

            if adapt_cr:
                cr_idx = np.random.choice(n_cr, p=cr_prob)
                cr = cr_values[cr_idx]
            else:
                cr = 1.0
                cr_idx = -1


            delta = np.random.randint(1, delta_init + 1)
            other = np.setdiff1d(np.arange(n_chains), [i])
            np.random.shuffle(other)
            a = other[:delta]
            b = other[delta:2 * delta]
            gamma = 2.38 / np.sqrt(2.0 * delta * n_params)
            e = np.random.uniform(1.0 - c, 1.0 + c, n_params)
            d_noise = np.random.normal(0.0, b_star, n_params)

            proposal = chains[i, t].copy()
            diff = np.sum(chains[a, t] - chains[b, t], axis=0)
            proposal += e * gamma * diff + d_noise


            if cr < 1.0:
                crossover_mask = np.random.rand(n_params) < cr
                if not np.any(crossover_mask):
                    crossover_mask[np.random.randint(n_params)] = True
                proposal = np.where(crossover_mask, proposal, chains[i, t])


            for p in range(n_params):
                if proposal[p] < bounds[p, 0]:
                    proposal[p] = 2.0 * bounds[p, 0] - proposal[p]
                elif proposal[p] > bounds[p, 1]:
                    proposal[p] = 2.0 * bounds[p, 1] - proposal[p]

                proposal[p] = np.clip(proposal[p], bounds[p, 0], bounds[p, 1])


            log_prop = log_posterior(proposal)
            log_alpha = log_prop - log_probs[i, t]
            if np.log(np.random.rand()) < log_alpha:
                chains[i, t + 1] = proposal
                log_probs[i, t + 1] = log_prop
                n_accept[i] += 1
                if adapt_cr and cr_idx >= 0:
                    cr_accepts[cr_idx] += 1
            else:
                chains[i, t + 1] = chains[i, t]
                log_probs[i, t + 1] = log_probs[i, t]

            if adapt_cr and cr_idx >= 0:
                cr_counts[cr_idx] += 1


        if adapt_cr and (t + 1) % 10 == 0:
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = cr_accepts / cr_counts
            ratio = np.where(np.isfinite(ratio), ratio, 1e-3)
            cr_prob = ratio / np.sum(ratio)
            cr_prob = np.where(cr_prob > 0, cr_prob, 1e-6)
            cr_prob /= np.sum(cr_prob)


    samples = chains[:, burn_in:, :]
    log_probs_out = log_probs[:, burn_in:]
    r_hat = gelman_rubin_r_hat(samples)
    info = {
        'acceptance_rate': np.mean(n_accept) / total_iter,
        'r_hat': r_hat,
        'converged': np.all(r_hat < 1.2)
    }
    return samples, log_probs_out, info


def test_mcmc_sampler():
    mean = np.array([0.5, -0.3])
    cov = np.array([[1.0, 0.3], [0.3, 0.5]])
    prec = np.linalg.inv(cov)
    log_post = lambda theta: -0.5 * (theta - mean) @ prec @ (theta - mean)
    bounds = np.array([[-3.0, 3.0], [-3.0, 3.0]])
    samples, logp, info = dream_mcmc(log_post, bounds, n_chains=4, n_samples=800, burn_in=200)
    combined = samples.reshape(-1, 2)
    est_mean = np.mean(combined, axis=0)
    assert np.allclose(est_mean, mean, atol=0.3), f"MCMC mean mismatch: {est_mean} vs {mean}"
    print("mcmc_sampler: all self-tests passed")


if __name__ == "__main__":
    test_mcmc_sampler()
