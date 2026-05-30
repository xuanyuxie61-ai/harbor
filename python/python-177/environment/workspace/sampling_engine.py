# -*- coding: utf-8 -*-

import numpy as np


class LatinCenterSampler:

    @staticmethod
    def sample(dim_num, point_num):
        x = np.zeros((point_num, dim_num), dtype=np.float64)
        for j in range(dim_num):
            perm = np.random.permutation(point_num)
            for i in range(point_num):
                x[i, j] = (2.0 * perm[i] + 1.0) / (2.0 * point_num)
        return x

    @staticmethod
    def sample_scaled(dim_num, point_num, bounds):
        x_unit = LatinCenterSampler.sample(dim_num, point_num)
        x_scaled = np.zeros_like(x_unit)
        for j in range(dim_num):
            low, high = bounds[j]
            x_scaled[:, j] = low + x_unit[:, j] * (high - low)
        return x_scaled


class UncertaintyQuantification:

    def __init__(self, sampler=None):
        if sampler is None:
            sampler = LatinCenterSampler()
        self.sampler = sampler

    def estimate_mean_variance(self, func, dim, n_samples, bounds=None):
        if bounds is None:
            bounds = [(0.0, 1.0)] * dim
        samples = self.sampler.sample_scaled(dim, n_samples, bounds)
        vals = np.array([func(s) for s in samples])
        mean = np.mean(vals)
        var = np.var(vals, ddof=1)
        return mean, var

    def estimate_sensitivity_indices(self, func, dim, n_samples, bounds=None):
        if bounds is None:
            bounds = [(0.0, 1.0)] * dim

        A = self.sampler.sample_scaled(dim, n_samples, bounds)
        B = self.sampler.sample_scaled(dim, n_samples, bounds)

        fA = np.array([func(a) for a in A])
        fB = np.array([func(b) for b in B])

        var_f = np.var(np.concatenate([fA, fB]), ddof=1)
        if var_f < 1e-14:
            return np.zeros(dim)

        S1 = np.zeros(dim)
        for i in range(dim):
            C = B.copy()
            C[:, i] = A[:, i]
            fC = np.array([func(c) for c in C])
            S1[i] = np.mean(fB * (fC - fA)) / var_f

        S1 = np.clip(S1, 0.0, 1.0)
        return S1

    def monte_carlo_integral(self, func, dim, n_samples, bounds=None):
        if bounds is None:
            bounds = [(0.0, 1.0)] * dim
        samples = self.sampler.sample_scaled(dim, n_samples, bounds)
        vals = np.array([func(s) for s in samples])
        volume = 1.0
        for low, high in bounds:
            volume *= (high - low)
        return volume * np.mean(vals)
