
import numpy as np
from typing import Tuple, List, Dict
from collections import Counter


class CombinatorialPhysics:

    @staticmethod
    def stirling_numbers_second_kind(m: int, n: int) -> np.ndarray:
        if m < 0 or n < 0:
            return np.zeros((1, 1))
        s = np.zeros((m + 1, n + 1), dtype=np.int64)
        s[0, 0] = 1
        for i in range(1, m + 1):
            for j in range(1, min(i, n) + 1):
                s[i, j] = j * s[i - 1, j] + s[i - 1, j - 1]
        return s

    @staticmethod
    def bell_numbers(m: int) -> np.ndarray:
        s = CombinatorialPhysics.stirling_numbers_second_kind(m, m)
        bell = np.sum(s, axis=1)
        return bell

    @staticmethod
    def partition_function(n: int, max_k: int) -> np.ndarray:
        p = np.zeros((n + 1, max_k + 1), dtype=np.int64)
        p[0, 0] = 1
        for i in range(1, n + 1):
            for k in range(1, min(i, max_k) + 1):
                p[i, k] = p[i - 1, k - 1] + p[i - k, k]
        return p

    @staticmethod
    def subset_sum_count(weights: np.ndarray, target: float,
                         tolerance: float = 1e-6) -> int:
        n = len(weights)
        count = 0

        dp = {0.0: 1}
        for w in weights:
            new_dp = dict(dp)
            for s, c in dp.items():
                new_sum = s + w
                if abs(new_sum - target) <= tolerance:
                    count += c
                new_dp[new_sum] = new_dp.get(new_sum, 0) + c
            dp = new_dp

        return count


class HistogramAnalysis:

    def __init__(self, data: np.ndarray, n_bins: int = 50,
                 range_limits: Tuple[float, float] = (0.0, 1.0)):
        self.data = np.asarray(data)
        self.n_bins = max(2, n_bins)
        self.range_limits = range_limits
        self.counts, self.bin_edges = np.histogram(
            self.data, bins=self.n_bins, range=self.range_limits
        )
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        self.bin_width = self.bin_edges[1] - self.bin_edges[0]

    def probability_density(self) -> np.ndarray:
        n_total = np.sum(self.counts)
        if n_total == 0 or self.bin_width == 0:
            return np.zeros_like(self.counts, dtype=float)
        return self.counts.astype(float) / (n_total * self.bin_width)

    def cumulative_distribution(self) -> np.ndarray:
        pdf = self.probability_density()
        cdf = np.cumsum(pdf) * self.bin_width
        cdf = np.clip(cdf, 0.0, 1.0)
        return cdf

    def mean(self) -> float:
        return float(np.mean(self.data))

    def variance(self) -> float:
        return float(np.var(self.data, ddof=1))

    def skewness(self) -> float:
        mu = self.mean()
        std = np.sqrt(self.variance())
        if std < 1e-15:
            return 0.0
        gamma1 = np.mean((self.data - mu) ** 3) / (std ** 3)
        return float(gamma1)

    def kurtosis(self) -> float:
        mu = self.mean()
        var = self.variance()
        if var < 1e-15:
            return 0.0
        kappa = np.mean((self.data - mu) ** 4) / (var ** 2) - 3.0
        return float(kappa)

    def moments(self, max_order: int = 4) -> Dict[int, float]:
        mu = self.mean()
        result = {}
        for k in range(1, max_order + 1):
            result[k] = float(np.mean((self.data - mu) ** k))
        return result


class PartonCascade:

    def __init__(self, alpha_s: float = 0.3, q0: float = 1.0):
        self.alpha_s = alpha_s
        self.q0 = q0

    def splitting_probability(self, z: float, t: float,
                              splitting_type: str = 'gg') -> float:
        if z <= 0.0 or z >= 1.0:
            return 0.0
        if splitting_type == 'gg':
            ca = 3.0
            p = 2.0 * ca * (z / (1.0 - z + 1e-10) +
                            (1.0 - z) / (z + 1e-10) +
                            z * (1.0 - z))
        elif splitting_type == 'qg':
            cf = 4.0 / 3.0
            p = cf * ((1.0 + z ** 2) / (1.0 - z + 1e-10))
        else:
            p = 0.0

        return self.alpha_s * p / (2.0 * np.pi * max(t, 1e-10))

    def multiplicity_distribution(self, E_init: float,
                                  n_events: int = 1000) -> np.ndarray:
        multiplicities = []
        for _ in range(n_events):

            if E_init <= self.q0:
                multiplicities.append(1)
                continue
            n_avg = 2.0 * np.log(E_init / self.q0)
            n_avg = max(1.0, n_avg)

            n = np.random.poisson(n_avg)
            n = max(1, n)
            multiplicities.append(n)
        return np.array(multiplicities)

    def jet_energy_spectrum(self, energies: np.ndarray,
                            n_bins: int = 40) -> HistogramAnalysis:
        e_max = np.max(energies) * 1.1 if len(energies) > 0 else 10.0
        hist = HistogramAnalysis(energies, n_bins=n_bins,
                                  range_limits=(0.0, e_max))
        return hist

    def color_singlet_combinatorics(self, n_gluons: int) -> Dict[str, int]:
        if n_gluons < 0:
            n_gluons = 0
        bell = CombinatorialPhysics.bell_numbers(n_gluons)

        color_factor = 1 if n_gluons == 0 else 8 ** (n_gluons - 1)
        return {
            'n_gluons': n_gluons,
            'bell_number': int(bell[n_gluons]),
            'color_configurations': color_factor,
            'total_singlets_estimate': int(min(bell[n_gluons] * color_factor, 2**63 - 1))
        }
