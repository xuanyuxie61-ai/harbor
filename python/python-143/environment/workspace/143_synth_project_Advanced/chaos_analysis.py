
import numpy as np
from typing import Tuple, Optional


class FractalDimension:

    @staticmethod
    def box_counting_dimension(x: np.ndarray, y: np.ndarray,
                                epsilons: Optional[np.ndarray] = None) -> float:
        if len(x) != len(y) or len(x) < 10:
            return 1.0

        if epsilons is None:

            log_eps = np.linspace(-3, 0, 20)
            epsilons = 10.0 ** log_eps

        counts = []
        valid_eps = []

        for eps in epsilons:
            if eps <= 0:
                continue

            x_bins = np.floor((x - np.min(x)) / eps).astype(int)
            y_bins = np.floor((y - np.min(y)) / eps).astype(int)
            boxes = set(zip(x_bins, y_bins))
            n_boxes = len(boxes)
            if n_boxes > 1:
                counts.append(n_boxes)
                valid_eps.append(eps)

        if len(counts) < 3:
            return 1.0

        counts = np.array(counts, dtype=float)
        valid_eps = np.array(valid_eps, dtype=float)


        log_n = np.log(counts)
        log_eps = np.log(1.0 / valid_eps)


        A = np.vstack([log_eps, np.ones(len(log_eps))]).T
        slope, intercept = np.linalg.lstsq(A, log_n, rcond=None)[0]
        return float(slope)


class LyapunovEstimator:

    @staticmethod
    def rosenstein_algorithm(data: np.ndarray,
                              embed_dim: int = 5,
                              tau: int = 1,
                              max_steps: int = 50) -> float:
        n = len(data)
        if n < embed_dim * tau + max_steps:
            return 0.0


        embedded = []
        for i in range(n - (embed_dim - 1) * tau):
            point = [data[i + j * tau] for j in range(embed_dim)]
            embedded.append(point)
        embedded = np.array(embedded)
        m = len(embedded)

        if m < 10:
            return 0.0


        nearest_dist = np.full(m, np.inf)
        nearest_idx = np.full(m, -1, dtype=int)

        for i in range(m):
            dists = np.linalg.norm(embedded - embedded[i], axis=1)
            dists[i] = np.inf
            nearest_idx[i] = np.argmin(dists)
            nearest_dist[i] = dists[nearest_idx[i]]


        k_vals = []
        y_vals = []

        for k in range(1, min(max_steps, m)):
            divergences = []
            for i in range(m - k):
                j = nearest_idx[i]
                if j + k < m:
                    d_new = np.linalg.norm(embedded[i + k] - embedded[j + k])
                    if d_new > 0 and nearest_dist[i] > 0:
                        divergences.append(np.log(d_new))

            if len(divergences) > 5:
                k_vals.append(k)
                y_vals.append(np.mean(divergences))

        if len(k_vals) < 5:
            return 0.0


        k_arr = np.array(k_vals, dtype=float)
        y_arr = np.array(y_vals, dtype=float)


        cutoff = len(k_arr) // 3
        if cutoff < 3:
            cutoff = len(k_arr)

        A = np.vstack([k_arr[:cutoff], np.ones(cutoff)]).T
        slope, _ = np.linalg.lstsq(A, y_arr[:cutoff], rcond=None)[0]


        dt = 1.0
        return float(slope / dt)


class HurstExponent:

    @staticmethod
    def rescaled_range(data: np.ndarray,
                        max_lag: Optional[int] = None) -> float:
        n = len(data)
        if n < 10:
            return 0.5

        if max_lag is None:
            max_lag = n // 4

        lags = []
        rs_values = []

        for lag in range(10, max_lag, max(1, max_lag // 20)):

            n_chunks = n // lag
            if n_chunks < 2:
                continue

            rs_chunks = []
            for i in range(n_chunks):
                chunk = data[i * lag:(i + 1) * lag]
                mean_c = np.mean(chunk)
                dev_cum = np.cumsum(chunk - mean_c)
                R = np.max(dev_cum) - np.min(dev_cum)
                S = np.std(chunk)
                if S > 1e-12:
                    rs_chunks.append(R / S)

            if len(rs_chunks) > 0:
                lags.append(lag)
                rs_values.append(np.mean(rs_chunks))

        if len(lags) < 3:
            return 0.5

        log_lags = np.log(lags)
        log_rs = np.log(rs_values)
        A = np.vstack([log_lags, np.ones(len(log_lags))]).T
        H, _ = np.linalg.lstsq(A, log_rs, rcond=None)[0]
        return float(np.clip(H, 0.0, 1.0))


class ChaosAnalyzer:

    def __init__(self):
        pass

    def analyze_price_path(self, prices: np.ndarray) -> dict:
        if len(prices) < 50:
            return {
                'box_dimension': 1.0,
                'lyapunov_max': 0.0,
                'hurst': 0.5,
                'returns_autocorr': 0.0,
            }


        t = np.arange(len(prices))

        prices_norm = (prices - np.min(prices)) / (np.max(prices) - np.min(prices) + 1e-12)
        t_norm = t / len(prices)

        d_box = FractalDimension.box_counting_dimension(t_norm, prices_norm)


        returns = np.diff(np.log(prices + 1e-12))
        lambda_max = LyapunovEstimator.rosenstein_algorithm(returns)
        H = HurstExponent.rescaled_range(returns)


        if len(returns) > 1 and np.std(returns) > 1e-12:
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            if np.isnan(autocorr):
                autocorr = 0.0
        else:
            autocorr = 0.0

        return {
            'box_dimension': d_box,
            'lyapunov_max': lambda_max,
            'hurst': H,
            'returns_autocorr': autocorr,
        }

    def regime_classification(self, metrics: dict) -> str:
        H = metrics.get('hurst', 0.5)
        lam = metrics.get('lyapunov_max', 0.0)
        d_box = metrics.get('box_dimension', 1.0)

        if lam > 0.01 and d_box > 1.3:
            if H > 0.55:
                return "混沌趋势 (Chaotic Trending)"
            elif H < 0.45:
                return "混沌均值回归 (Chaotic Mean-Reverting)"
            else:
                return "弱混沌 (Weak Chaos)"
        else:
            if H > 0.55:
                return "随机趋势 (Random Trending)"
            elif H < 0.45:
                return "随机均值回归 (Random Mean-Reverting)"
            else:
                return "有效市场 (Efficient Market)"
