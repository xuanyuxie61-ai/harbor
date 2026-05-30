
import math
from typing import List, Tuple, Dict


class EulerErrorAnalysis:

    def __init__(self, f, df_dt, df_dy, y0: float, t0: float, t_end: float):
        self.f = f
        self.df_dt = df_dt
        self.df_dy = df_dy
        self.y0 = y0
        self.t0 = t0
        self.t_end = t_end

    def local_truncation_error(self, h: float, t: float, y: float) -> float:
        d2y = self.df_dt(t, y) + self.f(t, y) * self.df_dy(t, y)
        return 0.5 * h * abs(d2y)

    def global_error_bound(self, h: float, L: float, M2: float) -> float:
        if L < 1e-15:
            return 0.5 * h * M2 * self.t_end
        return (h * M2 / (2.0 * L)) * (math.exp(L * self.t_end) - 1.0)

    def recommend_step(self, target_error: float, L: float, M2: float) -> float:
        if L < 1e-15:
            return 2.0 * target_error / (M2 * self.t_end + 1e-15)
        denom = (M2 / (2.0 * L)) * (math.exp(L * self.t_end) - 1.0)
        if denom < 1e-15:
            return 1.0
        return target_error / denom


class LogisticBifurcationAnalyzer:

    def __init__(self, r: float, x0: float = 0.5):
        self.r = r
        self.x0 = x0

    def fixed_points(self) -> List[float]:
        fps = [0.0]
        if self.r > 1.0:
            fps.append(1.0 - 1.0 / self.r)
        return fps

    def stability_of_fixed_point(self, x_star: float) -> float:
        return self.r * (1.0 - 2.0 * x_star)

    def lyapunov_exponent(self, n_iter: int = 5000, n_transient: int = 1000) -> float:
        x = self.x0

        for _ in range(n_transient):
            x = self.r * x * (1.0 - x)
            if x <= 0 or x >= 1:
                x = 0.5

        lam_sum = 0.0
        for _ in range(n_iter):
            x = self.r * x * (1.0 - x)
            if x <= 0 or x >= 1:
                x = 0.5
            deriv = abs(self.r * (1.0 - 2.0 * x))
            if deriv < 1e-15:
                deriv = 1e-15
            lam_sum += math.log(deriv)
        return lam_sum / n_iter

    def is_chaotic(self, threshold: float = 0.005) -> bool:
        return self.lyapunov_exponent() > threshold


class SIRStabilityAnalyzer:

    def __init__(self, alpha: float, beta: float, gamma: float, N: float):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.N = N

    def reproduction_number(self) -> float:
        if self.beta < 1e-15:
            return float('inf')
        return self.alpha / self.beta

    def endemic_equilibrium(self) -> Tuple[float, float, float]:
        r0 = self.reproduction_number()
        if r0 <= 1.0:
            return (self.N, 0.0, 0.0)
        S_star = self.N / r0
        I_star = self.gamma * (self.N - S_star) / (self.beta + self.gamma)
        R_star = max(self.N - S_star - I_star, 0.0)
        return (S_star, I_star, R_star)

    def jacobian_eigenvalues(self, S: float, I: float) -> Tuple[complex, complex, complex]:
        a11 = -self.alpha * I / self.N - self.gamma
        a12 = -self.alpha * S / self.N + self.gamma
        a13 = self.gamma
        a21 = self.alpha * I / self.N
        a22 = self.alpha * S / self.N - self.beta
        a23 = 0.0
        a31 = 0.0
        a32 = self.beta
        a33 = -self.gamma


        trace = a11 + a22 + a33


        det2 = a11 * a22 - a12 * a21
        disc = trace * trace - 4.0 * det2
        if disc >= 0:
            l1 = (-trace + math.sqrt(disc)) / 2.0
            l2 = (-trace - math.sqrt(disc)) / 2.0
        else:
            l1 = complex(-trace / 2.0, math.sqrt(-disc) / 2.0)
            l2 = complex(-trace / 2.0, -math.sqrt(-disc) / 2.0)
        l3 = a33
        return (l1, l2, l3)


class SortStabilityMonitor:

    def __init__(self, target_partitions: int, memory_size: int):
        self.P = target_partitions
        self.M = memory_size

    def partition_skew(self, partition_sizes: List[int]) -> float:
        if not partition_sizes:
            return 0.0
        mean_size = sum(partition_sizes) / len(partition_sizes)
        if mean_size < 1e-15:
            return 0.0
        max_size = max(partition_sizes)
        return max_size / mean_size - 1.0

    def run_length_cv(self, run_lengths: List[int]) -> float:
        if len(run_lengths) < 2:
            return 0.0
        mean_len = sum(run_lengths) / len(run_lengths)
        if mean_len < 1e-15:
            return 0.0
        var = sum((l - mean_len) ** 2 for l in run_lengths) / len(run_lengths)
        std = math.sqrt(var)
        return std / mean_len

    def overflow_risk(self, current_buffer: int, predicted_peak: float) -> float:
        if self.M < 1:
            return 1.0
        ratio = (predicted_peak - self.M) / self.M
        k = 5.0
        return 1.0 / (1.0 + math.exp(-k * ratio))

    def diagnose(self, partition_sizes: List[int], run_lengths: List[int],
                 predicted_peak: float) -> Dict[str, float]:
        return {
            "partition_skew": self.partition_skew(partition_sizes),
            "run_length_cv": self.run_length_cv(run_lengths),
            "overflow_risk": self.overflow_risk(0, predicted_peak),
            "stability_score": max(0.0, 1.0 - self.partition_skew(partition_sizes)
                                          - 0.5 * self.run_length_cv(run_lengths)
                                          - self.overflow_risk(0, predicted_peak))
        }
