# -*- coding: utf-8 -*-

import numpy as np
from typing import Tuple, Optional


class UncertaintyQuantification:

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        self.seed = seed

    @staticmethod
    def box_muller_transform(n_samples: int, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
        n_pairs = (n_samples + 1) // 2
        u1 = np.random.uniform(1e-10, 1.0, n_pairs)
        u2 = np.random.uniform(0.0, 1.0, n_pairs)
        r = np.sqrt(-2.0 * np.log(u1))
        theta = 2.0 * np.pi * u2
        z1 = r * np.cos(theta)
        z2 = r * np.sin(theta)
        z = np.concatenate([z1, z2])[:n_samples]
        return mu + sigma * z

    @staticmethod
    def multivariate_normal_sample(n_samples: int, mean: np.ndarray,
                                   cov: np.ndarray, method: str = 'cholesky') -> np.ndarray:
        mean = np.asarray(mean, dtype=np.float64)
        cov = np.asarray(cov, dtype=np.float64)
        dim = len(mean)
        if cov.shape != (dim, dim):
            raise ValueError("协方差矩阵维度不匹配")


        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:

            cov = cov + np.eye(dim) * 1e-10
            L = np.linalg.cholesky(cov)


        if method == 'cholesky':
            Z = np.random.standard_normal((dim, n_samples))
        else:
            Z = np.array([UncertaintyQuantification.box_muller_transform(n_samples)
                          for _ in range(dim)])
        X = mean[:, np.newaxis] + L @ Z
        return X.T

    @staticmethod
    def salpeter_imf(m: np.ndarray, alpha: float = 2.35) -> np.ndarray:
        m = np.asarray(m, dtype=np.float64)
        m = np.clip(m, 0.08, 100.0)
        return m ** (-alpha)

    @staticmethod
    def kroupa_imf(m: np.ndarray) -> np.ndarray:
        m = np.asarray(m, dtype=np.float64)
        xi = np.zeros_like(m)
        mask1 = (m >= 0.01) & (m < 0.08)
        mask2 = (m >= 0.08) & (m < 0.5)
        mask3 = (m >= 0.5) & (m < 100.0)
        xi[mask1] = m[mask1] ** (-0.3)
        xi[mask2] = m[mask2] ** (-1.3)
        xi[mask3] = m[mask3] ** (-2.3)
        return xi

    def sample_stellar_masses(self, n_stars: int, m_min: float = 0.5,
                              m_max: float = 25.0, imf_type: str = 'kroupa') -> np.ndarray:
        masses = []
        max_attempts = n_stars * 1000
        attempts = 0
        if imf_type == 'salpeter':
            pdf = lambda m: self.salpeter_imf(m)
            pdf_max = pdf(np.array([m_min]))[0]
        else:
            pdf = lambda m: self.kroupa_imf(m)
            pdf_max = pdf(np.array([m_min]))[0]

        while len(masses) < n_stars and attempts < max_attempts:
            m_trial = np.random.uniform(m_min, m_max)
            u = np.random.uniform(0, pdf_max)
            if u <= pdf(m_trial):
                masses.append(m_trial)
            attempts += 1
        return np.array(masses, dtype=np.float64)

    def propagate_nuclear_uncertainty(self, base_params: np.ndarray,
                                      param_cov: np.ndarray,
                                      n_mc: int = 100,
                                      model_func: Optional[callable] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_params = len(base_params)
        samples = self.multivariate_normal_sample(n_mc, base_params, param_cov)
        outputs = np.zeros(n_mc, dtype=np.float64)

        if model_func is not None:
            for i in range(n_mc):
                try:
                    outputs[i] = model_func(samples[i])
                except Exception:
                    outputs[i] = np.nan
        else:

            for i in range(n_mc):
                outputs[i] = base_params[0] * (1.0 + 0.1 * (samples[i, 0] - base_params[0]))

        valid = ~np.isnan(outputs)
        if not np.any(valid):
            return samples, outputs, np.zeros(6)

        out_valid = outputs[valid]
        stats = np.array([
            np.mean(out_valid),
            np.std(out_valid),
            np.percentile(out_valid, 16),
            np.percentile(out_valid, 84),
            np.percentile(out_valid, 2.5),
            np.percentile(out_valid, 97.5)
        ])
        return samples, outputs, stats

    @staticmethod
    def noncentral_beta_mean(a: float, b: float, lam: float) -> float:
        if a <= 0 or b <= 0:
            return 0.5
        base = a / (a + b)
        if lam <= 1e-6:
            return base
        correction = 1.0 + lam * b / (2.0 * a * (a + b + 1.0))
        return min(base * correction, 1.0)
