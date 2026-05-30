
import numpy as np
from typing import Callable


class NumericalRobustness:

    EPS = np.finfo(float).eps
    REAL_MIN = np.finfo(float).tiny
    REAL_MAX = np.finfo(float).max

    @staticmethod
    def next_float(x: float) -> float:
        if np.isnan(x) or np.isinf(x):
            return x
        if x == 0.0:
            return NumericalRobustness.EPS
        if x > 0:
            if x >= NumericalRobustness.REAL_MAX:
                return np.inf
            return x / (1.0 - NumericalRobustness.EPS / 2.0)
        else:
            if x <= -NumericalRobustness.REAL_MAX:
                return -np.inf
            return x * (1.0 - NumericalRobustness.EPS / 2.0)

    @staticmethod
    def prev_float(x: float) -> float:
        if np.isnan(x) or np.isinf(x):
            return x
        if x == 0.0:
            return -NumericalRobustness.EPS
        if x > 0:
            if x <= NumericalRobustness.REAL_MIN:
                return 0.0
            return x * (1.0 - NumericalRobustness.EPS / 2.0)
        else:
            if x >= -NumericalRobustness.REAL_MIN:
                return 0.0
            return x / (1.0 - NumericalRobustness.EPS / 2.0)

    @staticmethod
    def regula_falsi(f: Callable[[float], float], a: float, b: float,
                     tol: float = 1e-10, max_iter: int = 100) -> float:
        fa = f(a)
        fb = f(b)
        if fa * fb > 0:
            raise ValueError("f(a) and f(b) must have opposite signs.")
        if abs(fa) < tol:
            return a
        if abs(fb) < tol:
            return b


        if a > b:
            a, b = b, a
            fa, fb = fb, fa

        for _ in range(max_iter):

            denom = fb - fa
            if abs(denom) < 1e-15:
                return (a + b) / 2.0

            c = (a * fb - b * fa) / denom

            if c <= a or c >= b:
                c = (a + b) / 2.0

            fc = f(c)
            if abs(fc) < tol or abs(b - a) < tol:
                return c

            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc


            if abs(fb) < 0.5 * abs(fa):
                fa *= 0.5

        return (a + b) / 2.0

    @staticmethod
    def threshold_by_quantile_root(scores: np.ndarray, target_fpr: float = 0.05) -> float:
        sorted_scores = np.sort(scores)
        n = len(sorted_scores)
        if n == 0:
            return 0.0

        def empirical_fpr(theta: float) -> float:
            return np.mean(scores > theta) - target_fpr

        a = float(sorted_scores[0]) - 1e-6
        b = float(sorted_scores[-1]) + 1e-6
        fa = empirical_fpr(a)
        fb = empirical_fpr(b)


        if fa * fb > 0:
            if abs(fa) < abs(fb):
                return a
            return b

        return NumericalRobustness.regula_falsi(empirical_fpr, a, b)

    @staticmethod
    def ball_monte_carlo_integral(f: Callable[[np.ndarray], float],
                                   dim: int = 3, n_samples: int = 10000) -> float:
        if dim < 1:
            raise ValueError("dim must be >= 1")


        from math import gamma, pi
        volume = pi ** (dim / 2.0) / gamma(dim / 2.0 + 1.0)


        samples = np.random.randn(n_samples, dim)
        norms = np.linalg.norm(samples, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        uniforms = np.random.rand(n_samples, 1) ** (1.0 / dim)
        points = uniforms * samples / norms

        vals = np.array([f(p) for p in points])
        return volume * np.mean(vals)

    @staticmethod
    def mahalanobis_ball_probability(center: np.ndarray, cov: np.ndarray,
                                      radius: float, dim: int, n_samples: int = 50000) -> float:
        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:

            cov_reg = cov + 1e-6 * np.eye(dim)
            L = np.linalg.cholesky(cov_reg)

        def transformed_density(x: np.ndarray) -> float:


            return np.linalg.det(L)

        return NumericalRobustness.ball_monte_carlo_integral(transformed_density, dim, n_samples)

    @staticmethod
    def condition_number_sensitivity(matrix: np.ndarray) -> dict:
        cond = np.linalg.cond(matrix)

        digits_lost = np.log10(cond)

        solvable = digits_lost < -np.log10(NumericalRobustness.EPS)
        return {
            "condition_number": float(cond),
            "digits_lost": float(digits_lost),
            "solvable_in_double_precision": bool(solvable),
            "recommended_regularization": float(cond * NumericalRobustness.EPS)
        }
