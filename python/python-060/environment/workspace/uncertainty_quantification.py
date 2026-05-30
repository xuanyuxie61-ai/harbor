# -*- coding: utf-8 -*-

import numpy as np
from sparse_quadrature import sparse_grid_cc
from linear_solvers import cholesky_solve_dense
from utils import clip_positive


def build_covariance_matrix(n_param, correlation_length=0.5):
    i = np.arange(n_param)
    j = np.arange(n_param)
    I, J = np.meshgrid(i, j)
    Sigma = np.exp(-0.5 * ((I - J) / correlation_length) ** 2)
    return Sigma


def transform_to_physical(z, mu, L):
    z = np.asarray(z, dtype=float)
    mu = np.asarray(mu, dtype=float)
    return mu + L.dot(z)


class UQAnalyzer:

    def __init__(self, n_param=4, level_max=2):
        self.n_param = n_param
        self.level_max = level_max
        self.Sigma = build_covariance_matrix(n_param)

        self.L = np.linalg.cholesky(self.Sigma)
        self.mu = np.zeros(n_param)

        self.grid_points, self.grid_weights = sparse_grid_cc(n_param, level_max)

        from scipy.special import erfinv
        self.grid_points_z = np.sqrt(2.0) * erfinv(2.0 * self.grid_points - 1.0)

        valid = np.isfinite(self.grid_points_z).all(axis=1) & (np.abs(self.grid_weights) > 1e-16)
        self.grid_points_z = self.grid_points_z[valid]
        self.grid_weights = self.grid_weights[valid]

    def sample_parameters(self):
        samples = []
        for z in self.grid_points_z:
            xi = transform_to_physical(z, self.mu, self.L)
            samples.append(xi)
        return samples

    def estimate_moments(self, model_outputs):
        w = self.grid_weights
        y = np.asarray(model_outputs)

        w_sum = np.sum(w)
        if abs(w_sum) < 1e-15:
            w_sum = 1.0
        mean = np.tensordot(w, y, axes=([0], [0])) / w_sum
        diff = y - mean
        var = np.tensordot(w, diff ** 2, axes=([0], [0])) / w_sum
        var = np.maximum(var, 0.0)
        return mean, var, np.sqrt(var)

    def sobol_first_order(self, model_outputs, param_idx):
        y = np.asarray(model_outputs, dtype=float)
        w = self.grid_weights
        _, var_y, _ = self.estimate_moments(y)
        if var_y < 1e-20:
            return 0.0

        z_vals = self.grid_points_z[:, param_idx]

        h = 0.5
        n_q = len(y)
        cond_var = 0.0
        w_sum = np.sum(w)
        for j in range(n_q):
            kernel = np.exp(-0.5 * ((z_vals - z_vals[j]) / h) ** 2)
            kernel_w = kernel * w
            kw_sum = np.sum(kernel_w)
            if kw_sum > 1e-15:
                e_y = np.sum(kernel_w * y) / kw_sum
                cond_var += w[j] * (e_y ** 2)
        cond_var = cond_var / w_sum
        S_i = np.clip(cond_var / var_y, 0.0, 1.0)
        return float(S_i)


def monte_carlo_uncertainty(model_func, n_samples=500, n_param=4):
    Sigma = build_covariance_matrix(n_param)
    L = np.linalg.cholesky(Sigma)
    outputs = []
    for _ in range(n_samples):
        z = np.random.randn(n_param)
        xi = L.dot(z)
        out = model_func(xi)
        outputs.append(out)
    outputs = np.array(outputs)
    return float(np.mean(outputs)), float(np.std(outputs))
