
import numpy as np
from utils_numerical import safe_divide


def build_markov_transition_matrix(n_states: int, move_range: int = 1,
                                   boundary_reflect: bool = True) -> np.ndarray:
    P = np.zeros((n_states, n_states))

    for i in range(n_states):
        neighbors = []
        weights = []

        for step in range(1, move_range + 1):
            if i - step >= 0:
                neighbors.append(i - step)
                weights.append(1.0)
            elif boundary_reflect:
                neighbors.append(step - i - 1)
                weights.append(1.0)

            if i + step < n_states:
                neighbors.append(i + step)
                weights.append(1.0)
            elif boundary_reflect:
                neighbors.append(2 * n_states - i - step - 1)
                weights.append(1.0)

        if len(neighbors) == 0:
            P[i, i] = 1.0
        else:
            total = sum(weights)
            for j, w in zip(neighbors, weights):
                P[i, j] += w / total

    return P


def metropolis_hastings_sampler(log_posterior, x0: np.ndarray, n_samples: int = 5000,
                                proposal_cov: np.ndarray = None,
                                burn_in: int = 1000, thin: int = 5) -> dict:
    dim = len(x0)
    if proposal_cov is None:
        proposal_cov = np.eye(dim) * 0.01

    total_iterations = burn_in + n_samples * thin
    chain = np.zeros((total_iterations, dim))
    chain[0, :] = x0

    log_p_current = log_posterior(x0)
    accepts = 0

    for t in range(1, total_iterations):

        proposal = np.random.multivariate_normal(chain[t - 1, :], proposal_cov)

        log_p_proposal = log_posterior(proposal)
        log_alpha = log_p_proposal - log_p_current

        if np.random.rand() < np.exp(min(log_alpha, 0.0)):
            chain[t, :] = proposal
            log_p_current = log_p_proposal
            accepts += 1
        else:
            chain[t, :] = chain[t - 1, :]


    samples = chain[burn_in::thin, :]


    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)
    acceptance_rate = accepts / total_iterations


    n_eff = len(samples)
    if n_eff > 100:
        mid = n_eff // 2
        var_first = np.var(samples[:mid, :], axis=0, ddof=1)
        var_second = np.var(samples[mid:, :], axis=0, ddof=1)
        W = 0.5 * (var_first + var_second)
        B = np.var([np.mean(samples[:mid, :], axis=0), np.mean(samples[mid:, :], axis=0)], axis=0, ddof=1)
        V_hat = (mid - 1) / mid * W + B
        r_hat = np.sqrt(V_hat / (W + 1e-14))
    else:
        r_hat = np.ones(dim)

    return {
        'samples': samples,
        'chain': chain,
        'mean': mean,
        'std': std,
        'acceptance_rate': acceptance_rate,
        'r_hat': r_hat,
        'n_samples': len(samples)
    }


def sample_turbulence_parameters(u_data: np.ndarray, v_data: np.ndarray,
                                 n_samples: int = 2000) -> dict:

    def log_posterior(theta):
        C_mu, sigma_k, sigma_eps = theta


        if not (0.05 <= C_mu <= 0.15 and 0.5 <= sigma_k <= 2.0 and 0.5 <= sigma_eps <= 2.0):
            return -np.inf


        k_obs = 0.5 * (np.var(u_data) + np.var(v_data))
        epsilon_obs = k_obs ** 1.5 / 0.1


        nu_t = C_mu * k_obs ** 2 / (epsilon_obs + 1e-14)
        k_pred = nu_t / (C_mu + 1e-14)


        sigma_noise = 0.1 * k_obs
        log_likelihood = -0.5 * ((k_obs - k_pred) / sigma_noise) ** 2


        log_prior = 0.0

        return log_likelihood + log_prior

    x0 = np.array([0.09, 1.0, 1.3])
    proposal_cov = np.diag([0.001, 0.05, 0.05])

    result = metropolis_hastings_sampler(
        log_posterior, x0, n_samples=n_samples,
        proposal_cov=proposal_cov, burn_in=500, thin=3
    )

    return result


def compute_markov_chain_stationary(P: np.ndarray, max_iter: int = 500, tol: float = 1e-12) -> np.ndarray:
    n = P.shape[0]
    pi = np.ones(n) / n

    for _ in range(max_iter):
        pi_new = pi @ P
        if np.linalg.norm(pi_new - pi, 1) < tol:
            break
        pi = pi_new

    return pi
