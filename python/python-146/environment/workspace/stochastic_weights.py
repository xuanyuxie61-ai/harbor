
import numpy as np


class LogNormalSynapse:

    def __init__(self, mu, sigma):
        if sigma <= 0:
            raise ValueError("sigma must be positive.")
        self.mu = mu
        self.sigma = sigma

    def pdf(self, w):
        w = np.atleast_1d(w)
        pdf_vals = np.zeros_like(w, dtype=float)
        mask = w > 0
        pdf_vals[mask] = (
            1.0 / (w[mask] * self.sigma * np.sqrt(2.0 * np.pi))
            * np.exp(-((np.log(w[mask]) - self.mu) ** 2) / (2.0 * self.sigma ** 2))
        )
        return pdf_vals

    def cdf(self, w):
        w = np.atleast_1d(w)
        z = (np.log(w) - self.mu) / self.sigma

        cdf_vals = 0.5 * (1.0 + np.erf(z / np.sqrt(2.0)))
        cdf_vals = np.where(w > 0, cdf_vals, 0.0)
        return cdf_vals

    def inv_cdf(self, p):
        p = np.atleast_1d(p)
        if np.any((p <= 0) | (p >= 1)):
            raise ValueError("p must be in (0, 1).")

        z = np.sqrt(2.0) * np.erfcinv(2.0 * (1.0 - p))
        w = np.exp(self.mu + self.sigma * z)
        return w

    def sample(self, size=None):

        z = np.random.normal(loc=self.mu, scale=self.sigma, size=size)
        return np.exp(z)

    def mean(self):
        return np.exp(self.mu + 0.5 * self.sigma ** 2)

    def variance(self):
        return (np.exp(self.sigma ** 2) - 1.0) * np.exp(2.0 * self.mu + self.sigma ** 2)

    def sample_mean_variance(self, n_samples=10000):
        samples = self.sample(size=n_samples)
        return np.mean(samples), np.var(samples, ddof=1)


class SynapticWeightSDE:

    def __init__(self, mu=0.0, sigma=0.5, theta=0.1, dt=0.01):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt

    def euler_maruyama_step(self, w_current):
        if w_current <= 0:
            w_current = 1e-6
        ln_w = np.log(w_current)
        dW = np.random.normal(0.0, np.sqrt(self.dt))
        ln_w_new = ln_w - self.theta * (ln_w - self.mu) * self.dt + self.sigma * dW
        w_new = np.exp(ln_w_new)
        return max(w_new, 1e-6)

    def simulate_trajectory(self, w0, T_total):
        n_steps = int(np.ceil(T_total / self.dt))
        trajectory = np.zeros(n_steps)
        w = max(w0, 1e-6)
        for k in range(n_steps):
            w = self.euler_maruyama_step(w)
            trajectory[k] = w
        return trajectory

    def steady_state_check(self, samples):
        ln_samples = np.log(samples[samples > 0])
        emp_mean = np.mean(ln_samples)
        emp_std = np.std(ln_samples)

        return abs(emp_mean - self.mu), abs(emp_std - self.sigma)


def normalize_weights_multiplicative(weights, target_sum=1.0):
    weights = np.asarray(weights, dtype=float)
    current_sum = np.sum(weights)
    if current_sum <= 0:
        return weights
    return weights * (target_sum / current_sum)


def demo_log_normal_weights():
    model = LogNormalSynapse(mu=-0.5, sigma=0.8)
    samples = model.sample(size=5000)
    theoretical_mean = model.mean()
    theoretical_var = model.variance()
    empirical_mean, empirical_var = model.sample_mean_variance(n_samples=5000)
    return {
        'theoretical_mean': theoretical_mean,
        'theoretical_var': theoretical_var,
        'empirical_mean': empirical_mean,
        'empirical_var': empirical_var,
        'samples': samples
    }


def demo_weight_sde():
    sde = SynapticWeightSDE(mu=0.0, sigma=0.3, theta=0.05, dt=0.01)
    traj = sde.simulate_trajectory(w0=1.0, T_total=50.0)
    return traj
