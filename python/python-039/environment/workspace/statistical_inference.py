
import numpy as np
from typing import Tuple, Optional
from scipy.special import betainc, gammaln


class NonCentralTDistribution:

    @staticmethod
    def cdf(t: float, df: float, delta: float = 0.0,
            max_iter: int = 200, tol: float = 1e-12) -> Tuple[float, int]:
        if df <= 0.0:
            return 0.0, 1

        ifault = 0
        t = float(t)
        df = float(df)
        delta = float(delta)


        if df > 1e6:

            from scipy.stats import norm
            prob = norm.cdf(t - delta)
            return prob, ifault



        x = df / (df + t ** 2)
        if t < 0.0:
            sign_t = -1.0
        else:
            sign_t = 1.0


        try:
            from scipy.stats import nct
            prob = nct.cdf(t, df, delta)
        except Exception:

            from scipy.stats import norm
            approx_mean = delta * np.sqrt(df / 2.0) * np.exp(
                gammaln((df - 1) / 2.0) - gammaln(df / 2.0)
            ) if df > 1 else delta
            approx_var = df * (1.0 + delta ** 2) / (df - 2.0) if df > 2 else 1.0
            prob = norm.cdf((t - approx_mean) / np.sqrt(approx_var))

        return prob, ifault

    @staticmethod
    def quantile_search(p: float, df: float, delta: float = 0.0,
                        a: float = -20.0, b: float = 20.0,
                        tol: float = 1e-6) -> Tuple[float, int]:
        if p <= 0.0:
            return a, 0
        if p >= 1.0:
            return b, 0

        fa = NonCentralTDistribution.cdf(a, df, delta)[0]
        fb = NonCentralTDistribution.cdf(b, df, delta)[0]


        if fa > p or fb < p:

            while fa > p:
                a *= 2.0
                fa = NonCentralTDistribution.cdf(a, df, delta)[0]
            while fb < p:
                b *= 2.0
                fb = NonCentralTDistribution.cdf(b, df, delta)[0]

        for it in range(1, 200):
            c = (a + b) / 2.0
            fc = NonCentralTDistribution.cdf(c, df, delta)[0]

            if abs(fc - p) < tol or (b - a) < tol:
                return c, it

            if (fa - p) * (fc - p) < 0.0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc

        return (a + b) / 2.0, 200


class BisectionInteger:

    @staticmethod
    def find_root(f, a: int, b: int, max_iter: int = 100) -> Tuple[int, int]:
        fa = f(a)
        fb = f(b)

        if fa == 0:
            return a, 0
        if fb == 0:
            return b, 0

        if fa * fb > 0:

            if abs(fa) < abs(fb):
                return a, 0
            else:
                return b, 0

        for it in range(1, max_iter + 1):
            c = (a + b) // 2
            if c == a or c == b:
                return c, it

            fc = f(c)
            if fc == 0:
                return c, it

            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc

        return (a + b) // 2, max_iter


class QGPStatisticalInference:

    def __init__(self):
        pass

    def centrality_significance(self, n_part_observed: float,
                                n_part_mean: float,
                                n_part_std: float,
                                df: float = 10.0) -> Tuple[float, float]:
        if n_part_std < 1e-15:
            return 0.0, 1.0
        t_stat = (n_part_observed - n_part_mean) / (n_part_std / np.sqrt(df))
        delta = 0.0
        prob, _ = NonCentralTDistribution.cdf(abs(t_stat), df, delta)
        p_value = 2.0 * (1.0 - prob)
        return float(t_stat), float(p_value)

    def v2_significance(self, v2_observed: float,
                        v2_stat_error: float,
                        v2_systematic: float = 0.0) -> Tuple[float, float]:
        total_err = np.sqrt(v2_stat_error ** 2 + v2_systematic ** 2)
        if total_err < 1e-15:
            return 0.0, 0.0
        t_stat = v2_observed / total_err

        from scipy.stats import norm
        p_value = 2.0 * (1.0 - norm.cdf(abs(t_stat)))
        significance = abs(t_stat)
        return float(t_stat), float(significance)

    def critical_temperature_confidence(self, T_measured: float,
                                        T_error: float,
                                        confidence: float = 0.95) -> Tuple[float, float]:
        if T_error < 0.0:
            T_error = abs(T_error)
        alpha = 1.0 - confidence
        df = 30.0


        t_lower, _ = NonCentralTDistribution.quantile_search(
            alpha / 2.0, df, delta=0.0
        )
        t_upper, _ = NonCentralTDistribution.quantile_search(
            1.0 - alpha / 2.0, df, delta=0.0
        )

        margin = t_upper * T_error
        T_lower = T_measured - margin
        T_upper = T_measured + margin
        return float(T_lower), float(T_upper)

    def find_critical_point(self, temperatures: np.ndarray,
                            susceptibility: np.ndarray) -> Tuple[float, float]:
        idx_max = np.argmax(susceptibility)
        chi_max = susceptibility[idx_max]
        T_c = temperatures[idx_max]


        if 0 < idx_max < len(temperatures) - 1:

            T0, T1, T2 = temperatures[idx_max - 1:idx_max + 2]
            C0, C1, C2 = susceptibility[idx_max - 1:idx_max + 2]

            A = np.array([
                [T0 ** 2, T0, 1.0],
                [T1 ** 2, T1, 1.0],
                [T2 ** 2, T2, 1.0]
            ])
            try:
                coeffs = np.linalg.solve(A, [C0, C1, C2])
                a, b, c = coeffs
                if abs(a) > 1e-15:
                    T_vertex = -b / (2.0 * a)
                    if T0 <= T_vertex <= T2:
                        T_c = T_vertex
                        chi_max = a * T_vertex ** 2 + b * T_vertex + c
            except np.linalg.LinAlgError:
                pass

        return float(T_c), float(chi_max)
