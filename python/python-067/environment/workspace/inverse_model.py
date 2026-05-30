# -*- coding: utf-8 -*-

import numpy as np
from typing import Tuple, List, Callable, Optional
from scipy.special import erfc


class InverseModel:

    def __init__(self):
        self.rho = 1000.0
        self.g = 9.81
        self.mu = 1.0e-3

    @staticmethod
    def regula_falsi(f: Callable[[float], float],
                     a: float, b: float,
                     tol: float = 1.0e-10,
                     max_iter: int = 100) -> Tuple[float, int]:
        fa = f(a)
        fb = f(b)

        if fa * fb > 0:
            raise ValueError("f(a) 和 f(b) 必须异号")

        for it in range(max_iter):
            if abs(b - a) < tol:
                break


            if abs(fb - fa) < 1e-30:
                break
            c = (a * fb - b * fa) / (fb - fa)
            fc = f(c)

            if abs(fc) < tol:
                return c, it + 1

            if np.sign(fc) == np.sign(fa):
                a = c
                fa = fc
            else:
                b = c
                fb = fc

        return (a + b) / 2.0, max_iter

    def invert_permeability_from_travel_time(self,
                                              t_obs: float,
                                              L: float,
                                              i_hydraulic: float,
                                              n_e: float = 1.0,
                                              b_guess: float = 1.0e-4,
                                              b_min: float = 1.0e-6,
                                              b_max: float = 1.0e-2) -> dict:
        if t_obs <= 0 or L <= 0 or i_hydraulic <= 0:
            raise ValueError("物理参数必须为正")

        def travel_time_error(b):
            if b <= 0:
                return float('inf')
            K_eq = (self.rho * self.g * b ** 2) / (12.0 * self.mu)
            v = K_eq * i_hydraulic / n_e
            if v <= 0:
                return float('inf')
            t_sim = L / v
            return t_sim - t_obs


        f_min = travel_time_error(b_min)
        f_max = travel_time_error(b_max)

        if f_min * f_max > 0:

            if f_min > 0 and f_max > 0:

                b_max = b_max * 10.0
                f_max = travel_time_error(b_max)
            elif f_min < 0 and f_max < 0:

                b_min = max(b_min / 10.0, 1.0e-8)
                f_min = travel_time_error(b_min)

            if f_min * f_max > 0:

                b_opt = b_guess
                err = float('inf')
                for b_test in np.logspace(np.log10(b_min), np.log10(b_max), 100):
                    e = abs(travel_time_error(b_test))
                    if e < err:
                        err = e
                        b_opt = b_test
                K_opt = (self.rho * self.g * b_opt ** 2) / (12.0 * self.mu)
                return {
                    'aperture': b_opt,
                    'permeability': K_opt,
                    'iterations': -1,
                    'converged': False,
                    'method': 'grid_search'
                }

        b_inv, it = self.regula_falsi(travel_time_error, b_min, b_max)
        K_inv = (self.rho * self.g * b_inv ** 2) / (12.0 * self.mu)

        return {
            'aperture': b_inv,
            'permeability': K_inv,
            'iterations': it,
            'converged': it < 100,
            'method': 'regula_falsi'
        }

    def invert_dispersivity_from_breakthrough(self,
                                               times: np.ndarray,
                                               concentrations: np.ndarray,
                                               L: float,
                                               v: float,
                                               C0: float = 1.0) -> dict:
        if len(times) != len(concentrations):
            raise ValueError("times 和 concentrations 长度必须相同")

        def objective(D):
            if D <= 0:
                return 1e20
            C_sim = 0.5 * C0 * erfc((L - v * times) / (2.0 * np.sqrt(D * np.maximum(times, 1e-10))))
            return np.sum((concentrations - C_sim) ** 2)


        D_best = 1.0e-5
        obj_best = objective(D_best)

        for D_test in np.logspace(-8, -2, 100):
            obj = objective(D_test)
            if obj < obj_best:
                obj_best = obj
                D_best = D_test

        alpha_L = D_best / v if v > 0 else 0.0

        return {
            'dispersion_coefficient': D_best,
            'longitudinal_dispersivity': alpha_L,
            'objective_value': obj_best,
            'method': 'least_squares_grid'
        }

    @staticmethod
    def dirichlet_estimate_moments(samples: np.ndarray,
                                    alpha_min: float = 1.0e-5,
                                    max_iter: int = 100,
                                    tol: float = 1.0e-8) -> dict:
        from scipy.special import digamma, polygamma

        samples = np.asarray(samples)
        if samples.ndim != 2:
            raise ValueError("samples 必须为二维数组")

        n, k = samples.shape
        if n <= k:
            raise ValueError("样本数必须大于维度")


        row_sums = np.sum(samples, axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-3):
            raise ValueError("每行样本的和必须近似为 1")
        if np.any(samples <= 0):
            raise ValueError("所有样本分量必须为正")


        means = np.mean(samples, axis=0)
        vars_comp = np.var(samples, axis=0)


        s = np.sum(means * (1.0 - means) / (vars_comp + 1e-20))
        alpha0 = means * (s - 1.0)
        alpha0 = np.maximum(alpha0, alpha_min)


        log_x = np.mean(np.log(samples), axis=0)
        alpha = alpha0.copy()

        for it in range(max_iter):
            alpha_sum = np.sum(alpha)


            g = digamma(alpha_sum) - digamma(alpha) + log_x


            h_diag = -polygamma(1, alpha)
            h_off = polygamma(1, alpha_sum)


            z = h_diag
            b = h_off


            denom = 1.0 / z
            c = np.sum(denom)
            if c * b > -1.0:
                inv_h_times_g = g / z - denom * np.sum(g / z) / (1.0 / b + c)
            else:
                inv_h_times_g = g / z

            alpha_new = alpha - inv_h_times_g
            alpha_new = np.maximum(alpha_new, alpha_min)

            if np.max(np.abs(alpha_new - alpha)) < tol:
                alpha = alpha_new
                break

            alpha = alpha_new


        from scipy.special import gammaln
        ll = gammaln(np.sum(alpha)) - np.sum(gammaln(alpha)) + np.sum((alpha - 1.0) * log_x)

        return {
            'alpha': alpha,
            'log_likelihood': float(ll),
            'iterations': it + 1,
            'converged': it < max_iter - 1
        }

    def calibrate_dual_porosity(self,
                                 t_obs: np.ndarray,
                                 C_obs: np.ndarray,
                                 L: float,
                                 phi_m: float = 0.01,
                                 phi_im: float = 0.05) -> dict:
        if len(t_obs) != len(C_obs):
            raise ValueError("t_obs 和 C_obs 长度必须相同")



        t_early = t_obs[t_obs <= np.percentile(t_obs, 20)]
        C_early = C_obs[t_obs <= np.percentile(t_obs, 20)]
        t_late = t_obs[t_obs >= np.percentile(t_obs, 80)]
        C_late = C_obs[t_obs >= np.percentile(t_obs, 80)]


        if len(t_late) > 1 and np.all(C_late > 0):
            logC = np.log(C_late)
            dt = t_late - np.mean(t_late)
            slope = np.sum(dt * (logC - np.mean(logC))) / np.sum(dt ** 2)
            alpha_est = -slope * phi_im if slope < 0 else 1.0e-5
        else:
            alpha_est = 1.0e-5

        return {
            'mass_transfer_rate': max(alpha_est, 1.0e-8),
            'mobile_porosity': phi_m,
            'immobile_porosity': phi_im,
            'method': 'empirical_tail_fitting'
        }
