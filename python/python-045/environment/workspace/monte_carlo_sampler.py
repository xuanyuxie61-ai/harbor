#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def hypersphere01_sample(m, n):
    x = np.random.randn(m, n)
    norms = np.sqrt(np.sum(x ** 2, axis=0))
    norms[norms == 0.0] = 1.0
    x = x / norms
    return x


def hypersphere01_monomial_integral(m, e):
    from math import gamma
    e = np.asarray(e, dtype=np.int32)
    if np.any(e < 0):
        raise ValueError("指数必须非负")
    if np.any(e % 2 == 1):
        return 0.0

    num = 1.0
    for ei in e:
        num *= gamma((ei + 1) / 2.0)
    denom = gamma((m + np.sum(e)) / 2.0)
    return 2.0 * num / denom


def hypercube_surface_sample(n_points, d):
    p = np.random.rand(n_points, d)

    i = np.random.randint(0, d, size=n_points)

    s = np.random.randint(0, 2, size=n_points)
    for idx in range(n_points):
        p[idx, i[idx]] = float(s[idx])
    return p


def circle_unit_sample():
    theta = 2.0 * np.pi * np.random.rand()
    return np.array([np.cos(theta), np.sin(theta)])


def circle_distance_stats(n_samples):
    distances = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_samples):
        p = circle_unit_sample()
        q = circle_unit_sample()
        distances[i] = np.linalg.norm(p - q)
    mu = np.mean(distances)
    var = np.var(distances, ddof=1) if n_samples > 1 else 0.0
    return mu, var


def hypercube_surface_distance_stats(n_samples, d):
    p1 = hypercube_surface_sample(n_samples, d)
    p2 = hypercube_surface_sample(n_samples, d)
    distances = np.linalg.norm(p1 - p2, axis=1)
    mu = np.mean(distances)
    var = np.var(distances, ddof=1) if n_samples > 1 else 0.0
    return mu, var


class MetropolisHastingsSampler:

    def __init__(self, log_target, proposal_cov, bounds=None):
        self.log_target = log_target
        self.proposal_cov = np.asarray(proposal_cov, dtype=np.float64)
        self.dim = self.proposal_cov.shape[0]
        self.bounds = bounds

    def _proposal(self, current):
        proposal = np.random.multivariate_normal(current, self.proposal_cov)
        if self.bounds is not None:
            for i, (lb, ub) in enumerate(self.bounds):
                proposal[i] = np.clip(proposal[i], lb, ub)
        return proposal

    def sample(self, initial, n_samples, burn_in=1000, thinning=10):
        current = np.asarray(initial, dtype=np.float64)
        current_log = self.log_target(current)


        for _ in range(burn_in):
            proposal = self._proposal(current)
            prop_log = self.log_target(proposal)
            alpha = min(1.0, np.exp(prop_log - current_log))
            if np.random.rand() < alpha:
                current = proposal
                current_log = prop_log


        samples = np.zeros((n_samples, self.dim), dtype=np.float64)
        accepted = 0
        total = 0
        idx = 0
        step = 0

        while idx < n_samples:
            proposal = self._proposal(current)
            prop_log = self.log_target(proposal)
            alpha = min(1.0, np.exp(prop_log - current_log))
            total += 1
            if np.random.rand() < alpha:
                current = proposal
                current_log = prop_log
                accepted += 1

            step += 1
            if step % thinning == 0:
                samples[idx] = current
                idx += 1

        acceptance_rate = accepted / total if total > 0 else 0.0
        return samples, acceptance_rate


class AdaptiveCovarianceSampler:

    def __init__(self, log_target, initial_cov, bounds=None,
                 adapt_interval=100, target_rate=0.234):
        self.log_target = log_target
        self.cov = np.array(initial_cov, dtype=np.float64, copy=True)
        self.dim = self.cov.shape[0]
        self.bounds = bounds
        self.adapt_interval = adapt_interval
        self.target_rate = target_rate
        self.scale = 2.4 ** 2 / self.dim
        self._chain = []

    def sample(self, initial, n_samples, burn_in=1000):
        current = np.asarray(initial, dtype=np.float64)
        current_log = self.log_target(current)
        samples = np.zeros((n_samples, self.dim), dtype=np.float64)
        accepted = 0

        for step in range(-burn_in, n_samples):
            if step >= 0 and len(self._chain) > self.dim:

                emp_cov = np.cov(np.array(self._chain).T)

                emp_cov += 1e-8 * np.eye(self.dim)
                prop_cov = self.scale * emp_cov
            else:
                prop_cov = self.cov

            proposal = np.random.multivariate_normal(current, prop_cov)
            if self.bounds is not None:
                for i, (lb, ub) in enumerate(self.bounds):
                    proposal[i] = np.clip(proposal[i], lb, ub)

            prop_log = self.log_target(proposal)
            alpha = min(1.0, np.exp(prop_log - current_log))

            if np.random.rand() < alpha:
                current = proposal
                current_log = prop_log
                if step >= 0:
                    accepted += 1

            if step >= 0:
                samples[step] = current
                self._chain.append(current.copy())

        acceptance_rate = accepted / n_samples if n_samples > 0 else 0.0
        return samples, acceptance_rate


if __name__ == "__main__":

    x = hypersphere01_sample(3, 1000)
    norms = np.sqrt(np.sum(x ** 2, axis=0))
    print(f"超球面采样范数均值: {np.mean(norms):.6f} (应为 1.0)")

    mu, var = circle_distance_stats(10000)
    print(f"圆上距离统计: 均值={mu:.4f}, 方差={var:.4f} (理论均值≈1.2732)")

    mu_h, var_h = hypercube_surface_distance_stats(5000, 3)
    print(f"立方体表面距离: 均值={mu_h:.4f}, 方差={var_h:.4f}")


    def log_target(x):
        return -0.5 * np.sum(x ** 2)

    sampler = MetropolisHastingsSampler(log_target, 0.5 * np.eye(2))
    samples, rate = sampler.sample(np.zeros(2), 500, burn_in=200, thinning=5)
    print(f"MCMC 采样均值: {np.mean(samples, axis=0)}, 接受率: {rate:.3f}")
