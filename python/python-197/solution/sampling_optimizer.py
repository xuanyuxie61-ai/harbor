"""
sampling_optimizer.py
================================================================================
高性能计算检查点容错：拉丁超立方采样与不确定性量化优化

融合原项目：
  - 652_latin_random (拉丁随机超立方采样)

科学角色：
  1) 使用拉丁超立方采样 (LHS) 在高维参数空间（故障率、带宽、状态规模）
     中高效抽取样本；
  2) 基于样本评估不同检查点策略的期望损失，寻找鲁棒最优策略；
  3) 计算各参数对目标函数的 Sobol-like 敏感性指标。
================================================================================
"""

import numpy as np


def latin_random(dim_num: int, point_num: int, seed: int = None) -> np.ndarray:
    """
    生成 dim_num 维、point_num 个点的拉丁随机超立方样本。
    返回形状 (dim_num, point_num) 的数组，值在 [0,1]。
    """
    rng = np.random.default_rng(seed)
    x = np.zeros((dim_num, point_num))
    for i in range(dim_num):
        perm = rng.permutation(point_num)
        for j in range(point_num):
            x[i, j] = (perm[j] + rng.random()) / point_num
    return x


class CheckpointStrategyOptimizer:
    """
    使用 LHS 评估检查点策略在不确定参数下的表现。
    """

    def __init__(self, n_samples: int = 200):
        self.n_samples = n_samples

    def sample_parameters(self, seed: int = None):
        """
        采样不确定参数:
            dim0: log10(故障率) in [-5, -2]
            dim1: 写入带宽比 (本地/内存) in [0.05, 0.5]
            dim2: 状态规模 (GB) in [0.1, 10.0]
            dim3: 压缩比 in [0.01, 0.5]
        返回 (n_samples, 4) 数组。
        """
        samples = latin_random(4, self.n_samples, seed).T
        samples[:, 0] = 10.0 ** (-5.0 + 3.0 * samples[:, 0])
        samples[:, 1] = 0.05 + 0.45 * samples[:, 1]
        samples[:, 2] = 0.1 + 9.9 * samples[:, 2]
        samples[:, 3] = 0.01 + 0.49 * samples[:, 3]
        return samples

    @staticmethod
    def objective(fault_rate: float, bw_ratio: float, state_gb: float,
                  compression_ratio: float, checkpoint_interval: float) -> float:
        """
        目标函数：单位时间期望损失 = 检查点开销密度 + 期望重启损失密度。
        公式:
            T_overhead = (state_gb * compression_ratio) / (bw_ratio * checkpoint_interval)
            T_wasted   = fault_rate * checkpoint_interval / 2
            return T_overhead + T_wasted
        最优解析解: T* = sqrt(2 * state_gb * compression_ratio / (bw_ratio * fault_rate))
        """
        B0 = 1.0
        T_overhead = (state_gb * compression_ratio) / (max(bw_ratio, 1.0e-6) * B0 * max(checkpoint_interval, 1.0e-6))
        T_wasted = fault_rate * checkpoint_interval * 0.5
        return T_overhead + T_wasted

    def optimize_interval(self, fault_rate: float, bw_ratio: float,
                          state_gb: float, compression_ratio: float,
                          interval_candidates: np.ndarray = None) -> tuple:
        """
        在候选间隔中寻找使目标函数最小的检查点间隔。
        返回 (best_interval, best_loss)。
        """
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
        """
        在参数不确定性下寻找鲁棒最优检查点间隔。
        返回统计摘要字典。
        """
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
