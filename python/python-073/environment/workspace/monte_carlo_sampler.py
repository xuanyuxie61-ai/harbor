# -*- coding: utf-8 -*-

import numpy as np
from math import sqrt, exp, pi, gamma as gamma_func


class HypersonicParameterSampler:

    def __init__(self, Ma_range=(5.0, 8.0), Re_range=(1e5, 1e7),
                 Tw_Te_range=(0.5, 2.0), Tu_range=(0.001, 0.02)):
        self.Ma_range = Ma_range
        self.Re_range = Re_range
        self.Tw_Te_range = Tw_Te_range
        self.Tu_range = Tu_range

    def lhs_sampling(self, n_samples):
        n_params = 4
        samples = np.zeros((n_samples, n_params))

        for p in range(n_params):
            perm = np.random.permutation(n_samples)
            u = (perm + np.random.rand(n_samples)) / n_samples
            if p == 0:
                samples[:, p] = self.Ma_range[0] + u * (self.Ma_range[1] - self.Ma_range[0])
            elif p == 1:
                log_min, log_max = np.log10(self.Re_range[0]), np.log10(self.Re_range[1])
                samples[:, p] = 10.0 ** (log_min + u * (log_max - log_min))
            elif p == 2:
                samples[:, p] = self.Tw_Te_range[0] + u * (self.Tw_Te_range[1] - self.Tw_Te_range[0])
            else:
                samples[:, p] = self.Tu_range[0] + u * (self.Tu_range[1] - self.Tu_range[0])

        return samples

    def hyperball_uniform_sample(self, m, n):
        X = np.random.exponential(scale=1.0, size=(n, m))
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        radii = np.random.rand(n, 1) ** (1.0 / m)
        samples = radii * X / np.maximum(norms, 1e-15)
        return samples

    def parameter_distance_stats(self, samples):
        n = samples.shape[0]
        if n < 2:
            return 0.0, 0.0


        mins = np.min(samples, axis=0)
        maxs = np.max(samples, axis=0)
        ranges = np.maximum(maxs - mins, 1e-15)
        normalized = (samples - mins) / ranges


        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                dists.append(np.linalg.norm(normalized[i] - normalized[j]))

        dists = np.array(dists)
        mu = np.mean(dists)
        var = np.var(dists, ddof=1) if len(dists) > 1 else 0.0
        return mu, var

    def sequential_optimal_sampling(self, n_total, n_skip=None):
        if n_skip is None:
            n_skip = max(1, int(n_total / exp(1)))


        candidates = self.lhs_sampling(n_total)



        center = np.array([6.5, 1e6, 1.0, 0.01])
        scales = np.array([1.0, 1e6, 0.5, 0.01])
        values = np.linalg.norm((candidates - center) / scales, axis=1)


        if n_skip >= n_total:
            best_idx = np.argmax(values)
            return {'best_idx': best_idx, 'best_value': values[best_idx], 'strategy': 'random'}

        skip_max = np.max(values[:n_skip])
        best_idx = n_total - 1
        for i in range(n_skip, n_total):
            if values[i] > skip_max:
                best_idx = i
                break

        success = values[best_idx] == np.max(values)
        return {
            'best_idx': best_idx,
            'best_value': values[best_idx],
            'global_max': np.max(values),
            'success': success,
            'strategy': 'optimal_stop'
        }

    def uncertainty_propagation(self, transition_model, n_samples=500):
        samples = self.lhs_sampling(n_samples)
        Re_t = np.zeros(n_samples)

        for i in range(n_samples):
            Ma, Re, Tw_Te, Tu = samples[i]
            try:
                Re_t[i] = transition_model(Ma, Re, Tw_Te, Tu)
            except Exception:
                Re_t[i] = np.nan

        valid = Re_t[~np.isnan(Re_t)]
        if len(valid) == 0:
            return {'mean': np.nan, 'std': np.nan, 'ci95': (np.nan, np.nan)}

        mean_val = np.mean(valid)
        std_val = np.std(valid, ddof=1)
        ci_low = np.percentile(valid, 2.5)
        ci_high = np.percentile(valid, 97.5)

        return {
            'mean': mean_val,
            'std': std_val,
            'ci95': (ci_low, ci_high),
            'samples': samples,
            'Re_t': Re_t
        }


def random_transition_model(Ma, Re, Tw_Te, Tu):
    C1 = 200.0
    C2 = 0.8
    C3 = 0.4
    C4 = 0.7

    Re_theta_t = C1 * (Ma ** (-C2)) * (Tw_Te ** C3) * (Tu ** (-C4))

    Re_xt = (Re_theta_t ** 2) * 2.5 + np.random.normal(0, 1e4)
    return max(Re_xt, 1e4)
