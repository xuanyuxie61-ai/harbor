# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import solve_ivp
from growth_kinetics import size_dependent_growth
from nucleation_model import total_nucleation_rate


def newton_cotes_open_weights(n, a=0.0, b=1.0):
    if n <= 0:
        return np.array([]), np.array([])
    x = np.array([((n - i + 1) * a + i * b) / (n + 1) for i in range(1, n + 1)], dtype=float)


    w = np.zeros(n, dtype=float)
    for i in range(n):


        poly = np.array([1.0])
        denom = 1.0
        for j in range(n):
            if i == j:
                continue

            poly = np.convolve(poly, [1.0, -x[j]])
            denom *= (x[i] - x[j])


        antideriv = poly / np.arange(1, len(poly) + 1, dtype=float)

        val_b = np.polyval(antideriv[::-1], b)
        val_a = np.polyval(antideriv[::-1], a)
        w[i] = (val_b - val_a) / denom

    return x, w


def quadrature_integrate(func, a, b, n=64):
    x, w = newton_cotes_open_weights(n, a, b)
    if x.size == 0:
        return 0.0
    f_vals = func(x)
    f_vals = np.asarray(f_vals, dtype=float)
    return float(np.dot(w, f_vals))


class PopulationBalanceSolver:

    def __init__(self, params):
        self.p = params
        self.n_moments = 6

    def _rhs_moments(self, t, y, T_func):


















        raise NotImplementedError("Hole 2: _rhs_moments is not implemented.")

    def solve(self, t_span, T_func, y0=None, method='RK45', rtol=1e-6, atol=1e-9):
        n = self.n_moments
        if y0 is None:


            N0 = 1e10
            Lc = 50e-6
            sigma_L = 10e-6
            mu0 = np.zeros(n, dtype=float)
            for j in range(n):

                from math import comb, factorial
                s = 0.0
                for k in range(0, j // 2 + 1):
                    double_fact = 1.0
                    for m in range(1, k + 1):
                        double_fact *= (2.0 * m - 1.0)
                    s += comb(j, 2 * k) * double_fact * (sigma_L ** (2 * k)) * (Lc ** (j - 2 * k))
                mu0[j] = N0 * s
            y0 = np.zeros(n + 1, dtype=float)
            y0[:n] = mu0
            y0[n] = self.p['c0']

        y0 = np.asarray(y0, dtype=float)

        def rhs(t, y):
            return self._rhs_moments(t, y, T_func)

        sol = solve_ivp(rhs, t_span, y0, method=method,
                        dense_output=True, rtol=rtol, atol=atol,
                        max_step=(t_span[1] - t_span[0]) / 2000)
        return sol

    def get_moments_at_time(self, sol, t):
        t = float(t)
        t0, tf = sol.t[0], sol.t[-1]
        t = max(t0, min(t, tf))
        y = sol.sol(t)
        n = self.n_moments
        mu = y[:n]
        c = y[n]
        return mu, c

    def get_csd_parameters(self, sol, t):
        mu, c = self.get_moments_at_time(sol, t)
        N = mu[0]
        if N <= 0:
            return {'N': 0.0, 'mu_ln': 0.0, 'sigma_ln': 1.0, 'L_mean': 0.0, 'CV': 0.0}

        r1 = mu[1] / N
        r2 = mu[2] / N


        val = r2 / (r1 ** 2)
        val = max(val, 1.0 + 1e-10)
        sigma_ln_sq = np.log(val)
        sigma_ln = np.sqrt(sigma_ln_sq)
        mu_ln = np.log(r1) - 0.5 * sigma_ln_sq
        L_mean = r1
        CV = np.sqrt(np.exp(sigma_ln_sq) - 1.0)

        return {
            'N': N,
            'mu_ln': mu_ln,
            'sigma_ln': sigma_ln,
            'L_mean': L_mean,
            'CV': CV
        }
