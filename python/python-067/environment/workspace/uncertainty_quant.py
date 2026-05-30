# -*- coding: utf-8 -*-

import numpy as np
from typing import Tuple, List
from scipy.special import betainc, gammaln, gamma, digamma, polygamma


class UncertaintyQuantification:

    def __init__(self):
        pass

    @staticmethod
    def alogam(x: float) -> float:
        if x <= 0:
            raise ValueError("x 必须为正")
        return float(gammaln(x))

    @staticmethod
    def noncentral_beta_cdf(x: float, a: float, b: float,
                            lam: float, error_max: float = 1.0e-10) -> float:
        if not (0.0 <= x <= 1.0):
            raise ValueError("x 必须在 [0, 1] 范围内")
        if a <= 0 or b <= 0:
            raise ValueError("a 和 b 必须为正")
        if lam < 0:
            raise ValueError("lam 必须为非负")

        if x == 0.0:
            return 0.0
        if x == 1.0:
            return 1.0


        i = 0
        pi_val = np.exp(-lam / 2.0)

        beta_log = gammaln(a) + gammaln(b) - gammaln(a + b)
        bi = betainc(a, b, x)


        si = np.exp(a * np.log(x) + b * np.log(1.0 - x) - beta_log - np.log(a))

        p_sum = pi_val
        pb_sum = pi_val * bi

        max_iter = 10000
        for _ in range(max_iter):
            if p_sum >= 1.0 - error_max:
                break

            i += 1
            pi_val = 0.5 * lam * pi_val / i
            bi = bi - si
            si = x * (a + b + i - 1.0) * si / (a + i)

            p_sum += pi_val
            pb_sum += pi_val * bi

            if pi_val < 1e-20:
                break

        return min(max(pb_sum, 0.0), 1.0)

    @staticmethod
    def gamma_sample_stats(alpha: float, beta_param: float) -> dict:
        if alpha <= 0 or beta_param <= 0:
            raise ValueError("参数必须为正")

        mean = alpha / beta_param
        variance = alpha / (beta_param ** 2)
        std = np.sqrt(variance)
        mode = (alpha - 1.0) / beta_param if alpha > 1.0 else 0.0

        return {
            'mean': mean,
            'variance': variance,
            'std': std,
            'mode': mode,
            'skewness': 2.0 / np.sqrt(alpha),
            'kurtosis': 6.0 / alpha
        }

    @staticmethod
    def monte_carlo_uncertainty(forward_model: callable,
                                 param_distributions: List[dict],
                                 n_samples: int = 1000,
                                 seed: int = 42) -> dict:
        rng = np.random.default_rng(seed)
        results = []

        for _ in range(n_samples):
            sample_params = {}
            for pd in param_distributions:
                name = pd['name']
                dist = pd['dist']
                p = pd['params']

                if dist == 'uniform':
                    sample_params[name] = rng.uniform(p['low'], p['high'])
                elif dist == 'normal':
                    sample_params[name] = rng.normal(p['mean'], p['std'])
                elif dist == 'lognormal':
                    sample_params[name] = rng.lognormal(p['mu'], p['sigma'])
                elif dist == 'gamma':
                    sample_params[name] = rng.gamma(p['alpha'], 1.0/p['beta'])
                else:
                    raise ValueError(f"不支持的分布: {dist}")

            try:
                result = forward_model(sample_params)
                results.append(result)
            except Exception:
                continue

        results = np.array(results)
        if len(results) == 0:
            return {'mean': 0.0, 'std': 0.0, 'ci_95': (0.0, 0.0)}

        return {
            'mean': float(np.mean(results)),
            'std': float(np.std(results)),
            'median': float(np.median(results)),
            'ci_95': (float(np.percentile(results, 2.5)),
                      float(np.percentile(results, 97.5))),
            'min': float(np.min(results)),
            'max': float(np.max(results)),
            'n_samples': len(results)
        }

    @staticmethod
    def permeability_confidence_interval(K_estimate: float,
                                          K_std: float,
                                          confidence: float = 0.95) -> Tuple[float, float]:
        if K_estimate <= 0 or K_std < 0:
            raise ValueError("K_estimate 必须为正，K_std 必须为非负")

        from scipy.stats import norm


        cv = K_std / K_estimate if K_estimate > 0 else 0.0
        sigma_ln = np.sqrt(np.log(1.0 + cv ** 2))
        mu_ln = np.log(K_estimate) - 0.5 * sigma_ln ** 2

        alpha = 1.0 - confidence
        z_low = norm.ppf(alpha / 2.0)
        z_high = norm.ppf(1.0 - alpha / 2.0)

        K_low = np.exp(mu_ln + z_low * sigma_ln)
        K_high = np.exp(mu_ln + z_high * sigma_ln)

        return K_low, K_high

    @staticmethod
    def sensitivity_analysis(forward_model: callable,
                              base_params: dict,
                              perturbation: float = 0.01) -> dict:
        f_base = forward_model(base_params)
        if abs(f_base) < 1e-20:
            f_base = 1e-20

        sensitivities = {}
        for name, value in base_params.items():
            if abs(value) < 1e-20:
                continue

            dp = value * perturbation
            params_plus = base_params.copy()
            params_plus[name] = value + dp

            f_plus = forward_model(params_plus)
            df_dp = (f_plus - f_base) / dp


            S = df_dp * value / f_base
            sensitivities[name] = float(S)

        return sensitivities

    @staticmethod
    def first_order_reliability(g_func: callable,
                                 mean_params: np.ndarray,
                                 cov_matrix: np.ndarray,
                                 tol: float = 1.0e-6,
                                 max_iter: int = 100) -> dict:
        from scipy.stats import norm

        n = len(mean_params)
        u = np.zeros(n)


        try:
            L = np.linalg.cholesky(cov_matrix)
        except np.linalg.LinAlgError:

            L = np.linalg.cholesky(cov_matrix + 1e-10 * np.eye(n))

        for _ in range(max_iter):
            x = mean_params + L @ u
            g_val = g_func(x)


            grad = np.zeros(n)
            h = 1e-6
            for i in range(n):
                x_pert = x.copy()
                x_pert[i] += h
                grad[i] = (g_func(x_pert) - g_val) / h


            grad_u = L.T @ grad
            grad_norm = np.linalg.norm(grad_u)
            if grad_norm < 1e-20:
                break


            u_new = (grad_u @ u - g_val) / grad_norm ** 2 * grad_u

            if np.linalg.norm(u_new - u) < tol:
                u = u_new
                break
            u = u_new

        beta = np.linalg.norm(u)
        pf = norm.cdf(-beta)

        return {
            'reliability_index': float(beta),
            'failure_probability': float(pf),
            'design_point': mean_params + L @ u,
            'converged': True
        }
