
import numpy as np


def latin_random(dim_num: int, point_num: int, seed: int = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros((dim_num, point_num))
    for i in range(dim_num):
        perm = rng.permutation(point_num)
        for j in range(point_num):
            x[i, j] = (perm[j] + rng.random()) / point_num
    return x


class CheckpointStrategyOptimizer:

    def __init__(self, n_samples: int = 200):
        self.n_samples = n_samples

    def sample_parameters(self, seed: int = None):
        samples = latin_random(4, self.n_samples, seed).T
        samples[:, 0] = 10.0 ** (-5.0 + 3.0 * samples[:, 0])
        samples[:, 1] = 0.05 + 0.45 * samples[:, 1]
        samples[:, 2] = 0.1 + 9.9 * samples[:, 2]
        samples[:, 3] = 0.01 + 0.49 * samples[:, 3]
        return samples

    @staticmethod
    def objective(fault_rate: float, bw_ratio: float, state_gb: float,
                  compression_ratio: float, checkpoint_interval: float) -> float:
        B0 = 1.0
        T_overhead = (state_gb * compression_ratio) / (max(bw_ratio, 1.0e-6) * B0 * max(checkpoint_interval, 1.0e-6))
        T_wasted = fault_rate * checkpoint_interval * 0.5
        return T_overhead + T_wasted

    def optimize_interval(self, fault_rate: float, bw_ratio: float,
                          state_gb: float, compression_ratio: float,
                          interval_candidates: np.ndarray = None) -> tuple:
        if interval_candidates is None:
            interval_candidates = np.logspace(0.0, 4.0, 300)
        best_loss = float('inf')
        best_interval = interval_candidates[0]
        for dt in interval_candidates:
            loss = self.objective(fault_rate, bw_ratio, state_gb, compression_ratio, dt)
            if loss < best_loss:
                best_loss = loss
                best_interval = dt
        return best_interval, best_loss

    def robust_optimize(self, seed: int = None) -> dict:
        samples = self.sample_parameters(seed)
        optimal_intervals = []
        losses = []
        for i in range(self.n_samples):
            fr, bw, sg, cr = samples[i]
            dt, loss = self.optimize_interval(fr, bw, sg, cr)
            optimal_intervals.append(dt)
            losses.append(loss)
        optimal_intervals = np.array(optimal_intervals)
        losses = np.array(losses)
        return {
            "mean_interval": float(np.mean(optimal_intervals)),
            "std_interval": float(np.std(optimal_intervals)),
            "median_interval": float(np.median(optimal_intervals)),
            "mean_loss": float(np.mean(losses)),
            "worst_loss": float(np.max(losses)),
            "best_loss": float(np.min(losses)),
        }
