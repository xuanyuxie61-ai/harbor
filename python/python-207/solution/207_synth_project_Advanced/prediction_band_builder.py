"""
prediction_band_builder.py — 预测区间构建

科学背景
========
预测区间 (Prediction Interval, PI) 与置信区间 (CI) 的区别:
- CI 估计均值 μ(x) 的不确定性
- PI 预测新观测值 y_new(x) 的不确定性

对于新观测 y_new = μ(x) + ε,  ε ~ N(0, σ²):
    PI:  μ̂(x) ± t_{α/2,ν} · σ̂(x) · √(1 + 1/N)

√(1+1/N) 因子体现了 "预测新值" 的额外不确定性.

同时预测带 (Simultaneous Prediction Band):
    P( y_new(x) ∈ PB(x)  ∀x ) = 1 - α

需要更大的临界值 c_α^pred, 因为:
    c_α^pred = c_α^mean · √(1 + 1/N)

算法来源 (种子项目 809_nonlin_regula)
======================================
用 Regula Falsi 求解:
    F̂_pred(c) = (1/B) Σ_b 1{max_x |y_b(x) - μ̂(x)| / s_pred(x) ≤ c} = 1-α

在本项目中的角色
================
1. 构建点态预测区间
2. 构建同时预测带
3. 对比 CI 与 PI 的宽度差异

核心公式
========
1. 点态 PI:
   μ̂ ± t_{α/2,ν} · σ̂ · √(1 + 1/N)
2. 预测标准差:
   s_pred = σ̂ · √(1 + 1/N)
3. 同时 PB 临界值:
   c_pred = F̂_{M_pred}^{-1}(1-α)
"""

import numpy as np
from nonlinear_rootfinder import regula_falsi, fixed_point_iteration


class PredictionBandBuilder:
    """预测区间/带构建器."""

    def __init__(self, mc_solutions, alpha=0.05):
        """
        参数
        ----
        mc_solutions : ndarray, shape (n_mc, nt, nx)
        alpha : float
        """
        self.mc_solutions = mc_solutions
        self.alpha = alpha
        self.target_coverage = 1.0 - alpha
        self.n_mc = mc_solutions.shape[0]

        # 基本统计量
        self.mean_field = np.mean(mc_solutions, axis=0)
        self.std_field = np.std(mc_solutions, axis=0, ddof=1) if self.n_mc > 1 else np.zeros_like(mc_solutions[0])
        self.stderr_field = self.std_field / np.sqrt(self.n_mc)

        # 预测标准差: σ̂ · √(1 + 1/N), 带下限保护
        positive_std = self.std_field[self.std_field > 0]
        floor_val = 1.0e-6
        if len(positive_std) > 0:
            floor_val = max(1.0e-6, np.mean(positive_std) * 0.01)
        self.std_field = np.maximum(self.std_field, floor_val)
        self.pred_std = self.std_field * np.sqrt(1.0 + 1.0 / max(self.n_mc, 1))

    def t_critical_value(self):
        """t 分布临界值.

        t_{α/2, ν},  ν = N-1

        返回
        ----
        t_val : float
        """
        from scipy.stats import t as t_dist
        nu = max(self.n_mc - 1, 1)
        return t_dist.ppf(1.0 - self.alpha / 2.0, nu)

    def pointwise_prediction_interval(self):
        """点态预测区间.

        PI(x) = [μ̂(x) - t·s_pred(x),  μ̂(x) + t·s_pred(x)]

        返回
        ----
        pi_lower : ndarray
        pi_upper : ndarray
        t_val : float
        """
        t_val = self.t_critical_value()
        pi_lower = self.mean_field - t_val * self.pred_std
        pi_upper = self.mean_field + t_val * self.pred_std
        return pi_lower, pi_upper, t_val

    def compute_prediction_residuals(self):
        """计算预测残差.

        对于每个 MC 实现 m, 预测残差为:
            r_m(x) = (u_m(x) - μ̂_{-m}(x)) / s_pred(x)
        其中 μ̂_{-m} 为 leave-one-out 均值.

        简化: 使用全样本均值.

        返回
        ----
        pred_residuals : ndarray, shape (n_mc, ...)
        """
        pred_std_safe = np.maximum(self.pred_std, 1.0e-30)
        pred_residuals = (self.mc_solutions - self.mean_field[None, :, :]) / pred_std_safe[None, :, :]
        return pred_residuals

    def simultaneous_prediction_band(self, n_bootstrap=500, seed=42):
        """同时预测带.

        使用 Bootstrap 估计预测极大统计量的分布.

        参数
        ----
        n_bootstrap : int
        seed : int

        返回
        ----
        pb_lower : ndarray
        pb_upper : ndarray
        c_pred : float
        """
        pred_residuals = self.compute_prediction_residuals()

        # Bootstrap
        rng = np.random.default_rng(seed)
        boot_max = np.zeros(n_bootstrap)
        for b in range(n_bootstrap):
            idx = rng.choice(self.n_mc, size=self.n_mc, replace=True)
            resampled = pred_residuals[idx]
            boot_max[b] = np.max(np.abs(resampled))

        # 临界值
        c_pred = np.percentile(boot_max, self.target_coverage * 100)

        # Regula Falsi 精炼
        stats_sorted = np.sort(boot_max)
        n = len(stats_sorted)

        def empirical_cdf(c):
            return np.searchsorted(stats_sorted, c, side='right') / n

        def objective(c):
            return empirical_cdf(c) - self.target_coverage

        c_low = stats_sorted[0]
        c_high = stats_sorted[-1]

        if objective(c_low) <= 0 and objective(c_high) >= 0:
            result = regula_falsi(objective, c_low, c_high,
                                  tol=1.0 / n, max_iter=200)
            if result['converged']:
                c_pred = result['root']

        # 构建带
        pb_lower = self.mean_field - c_pred * self.pred_std
        pb_upper = self.mean_field + c_pred * self.pred_std

        return pb_lower, pb_upper, c_pred

    def compare_ci_pi_widths(self, ci_critical_value):
        """比较 CI 和 PI 的宽度.

        参数
        ----
        ci_critical_value : float

        返回
        ----
        info : dict
        """
        t_val = self.t_critical_value()

        ci_width = 2.0 * ci_critical_value * self.stderr_field
        pi_width = 2.0 * t_val * self.pred_std

        # 只在有显著宽度的位置计算比值
        valid = ci_width > 1.0e-10
        if np.any(valid):
            ratio = np.mean(pi_width[valid] / ci_width[valid])
        else:
            ratio = np.mean(pi_width) / max(np.mean(ci_width), 1e-30)

        return {
            'mean_ci_width': np.mean(ci_width),
            'mean_pi_width': np.mean(pi_width),
            'mean_width_ratio': ratio,
            'max_ci_width': np.max(ci_width),
            'max_pi_width': np.max(pi_width),
            't_critical': t_val,
            'ci_critical': ci_critical_value,
            'prediction_factor': np.sqrt(1.0 + 1.0 / max(self.n_mc, 1)),
        }
