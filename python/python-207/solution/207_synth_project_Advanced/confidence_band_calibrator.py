"""
confidence_band_calibrator.py — 置信带校准: 不动点迭代 + Bootstrap

科学背景
========
构建同时置信带 (Simultaneous Confidence Band, SCB):

    P( μ(x) ∈ [μ̂(x) ± c_α · SE(x)]  ∀x ) = 1 - α

关键挑战在于确定临界值 c_α, 使得覆盖率精确达到 1-α.

方法
====
1. ECH (Expected Euler Characteristic Heuristic):
   c_α ≈ √(2·ln(N_eff)) + correction
   其中 N_eff 为有效独立检验数

2. Bootstrap 校准:
   - 从 MC 样本中计算标准化残差的极大统计量 M_n
   - c_α = F̂_M^{-1}(1-α)
   - 用 Regula Falsi 精确定位分位数

3. 不动点迭代精炼:
   c_{k+1} = c_k + γ·(F̂_M(c_k) - (1-α))

算法来源 (种子项目 807_nonlin_fixed_point)
==========================================
不动点迭代: c_{k+1} = g(c_k)
其中 g(c) = c + γ·(F̂_M(c) - (1-α))

核心公式
========
1. ECH 近似:
   c_α ≈ √(-2·ln(1 - (1-α)^{1/N_eff}))
2. Bootstrap 分位数:
   c_α = min{c : F̂_M(c) ≥ 1-α}
3. 收敛准则:
   |c_{k+1} - c_k| < tol  或  |F̂_M(c_k) - (1-α)| < tol
"""

import numpy as np
from nonlinear_rootfinder import regula_falsi, fixed_point_iteration


class ConfidenceBandCalibrator:
    """置信带校准器.

    综合使用 ECH、Bootstrap、Regula Falsi 和不动点迭代
    来精确校准同时置信带的临界值.
    """

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

        # 检测 "有效" 节点 (方差大于全局均值的 1%):
        # 排除 Dirichlet 边界等方差为零的退化节点
        # 注意: std_field 可能为 (nt, nx) 或 (nx,), 需要在空间维度上判断
        if self.std_field.ndim >= 2:
            std_over_time = np.max(self.std_field, axis=tuple(range(self.std_field.ndim - 1)))
        else:
            std_over_time = self.std_field
        positive_std = std_over_time[std_over_time > 0]
        if len(positive_std) > 0:
            self.interior_floor = max(1.0e-6, np.mean(positive_std) * 0.01)
        else:
            self.interior_floor = 1.0e-6
        self.interior_mask = std_over_time > self.interior_floor

        # 标准化残差 (只在内部节点有意义)
        stderr_floor = max(1.0e-8, self.interior_floor / np.sqrt(max(self.n_mc, 2)))
        std_safe = np.maximum(self.stderr_field, stderr_floor)
        self.residuals = (mc_solutions - self.mean_field[None, :, :]) / std_safe[None, :, :]

        # 极大统计量: 仅对内部节点计算
        if self.residuals.ndim == 3:
            # shape: (n_mc, nt, nx), mask is (nx,)
            if self.interior_mask.ndim == 1 and self.interior_mask.shape[0] == self.residuals.shape[2]:
                masked_res = self.residuals[:, :, self.interior_mask]
            else:
                masked_res = self.residuals
            if masked_res.shape[-1] == 0:
                masked_res = self.residuals
        elif self.residuals.ndim == 2:
            if self.interior_mask.ndim == 1 and self.interior_mask.shape[0] == self.residuals.shape[1]:
                masked_res = self.residuals[:, self.interior_mask]
            else:
                masked_res = self.residuals
            if masked_res.shape[-1] == 0:
                masked_res = self.residuals
        else:
            masked_res = self.residuals

        self.max_stats = np.max(np.abs(masked_res), axis=tuple(range(1, masked_res.ndim)))
        self.n_interior = masked_res.shape[-1] if masked_res.ndim >= 2 else masked_res.shape[0]
        # 保存 masked 残差, 用于 bootstrap
        self.masked_residuals = masked_res

    def ech_critical_value(self):
        """Euler 特征启发式临界值.

        c_ECH = √(-2·ln(1 - (1-α)^{1/N_eff}))

        其中 N_eff 为有效独立检验数.

        返回
        ----
        c_ech : float
        n_eff : float
        """
        # 估计 N_eff: 基于相关长度
        # 简化: 使用空间点数 / 相关长度比
        if self.residuals.ndim == 3:
            n_spatial = self.residuals.shape[2]
        else:
            n_spatial = self.residuals.shape[1]

        # 基于零交叉率估计 N_eff
        n_eff = max(1, n_spatial / 3.0)

        # ECH 公式
        p_single = 1.0 - (1.0 - self.alpha) ** (1.0 / n_eff)
        if p_single <= 0 or p_single >= 1:
            c_ech = 2.0  # 退化
        else:
            from scipy.stats import norm
            c_ech = norm.ppf(1.0 - p_single / 2.0)

        return c_ech, n_eff

    def bootstrap_critical_value(self, n_bootstrap=500, seed=42):
        """Bootstrap 临界值.

        使用两种方法:
        1. MC max_stats 的经验分位数 (主要)
        2. Bootstrap 验证 (辅助)

        参数
        ----
        n_bootstrap : int
        seed : int

        返回
        ----
        c_boot : float
        boot_stats : ndarray
        """
        # 主要方法: 直接从 MC max_stats 的经验分布取分位数
        # 这是最直接且统计上有效的方法
        c_boot = np.percentile(self.max_stats, self.target_coverage * 100)

        # 辅助 Bootstrap: 用于估计临界值的标准误差
        rng = np.random.default_rng(seed)
        n_mc = len(self.max_stats)
        boot_medians = np.zeros(n_bootstrap)
        for b in range(n_bootstrap):
            idx = rng.choice(n_mc, size=n_mc, replace=True)
            resampled_max = self.max_stats[idx]
            boot_medians[b] = np.percentile(resampled_max, self.target_coverage * 100)

        # 返回: 主分位数 + bootstrap 样本 (用于后续 Regula Falsi)
        # 将 max_stats 自身作为 "bootstrap 样本" 使用
        return c_boot, self.max_stats.copy()

    def refine_with_regula_falsi(self, boot_max_stats):
        """用 Regula Falsi 精炼临界值 (种子 809).

        求解: F̂_M(c) - (1-α) = 0

        参数
        ----
        boot_max_stats : ndarray

        返回
        ----
        c_refined : float
        result : dict
        """
        stats_sorted = np.sort(boot_max_stats)
        n = len(stats_sorted)

        def empirical_cdf(c):
            return np.searchsorted(stats_sorted, c, side='right') / n

        def objective(c):
            return empirical_cdf(c) - self.target_coverage

        c_low = stats_sorted[0]
        c_high = stats_sorted[-1]

        if objective(c_low) > 0:
            return c_low, {'converged': True, 'message': '下界'}
        if objective(c_high) < 0:
            return c_high, {'converged': True, 'message': '上界'}

        result = regula_falsi(objective, c_low, c_high,
                              tol=1.0 / n, max_iter=300)
        return result['root'], result

    def refine_with_fixed_point(self, boot_max_stats, c_init=None):
        """用不动点迭代精炼临界值 (种子 807).

        c_{k+1} = c_k + γ·(F̂_M(c_k) - (1-α))

        参数
        ----
        boot_max_stats : ndarray
        c_init : float or None

        返回
        ----
        c_refined : float
        result : dict
        """
        stats_sorted = np.sort(boot_max_stats)
        n = len(stats_sorted)

        def empirical_cdf(c):
            return np.searchsorted(stats_sorted, c, side='right') / n

        if c_init is None:
            c_init = np.percentile(boot_max_stats, self.target_coverage * 100)

        gamma = np.std(boot_max_stats) * 0.5

        def g_map(c_arr):
            c_val = float(c_arr[0])
            return np.array([c_val + gamma * (empirical_cdf(c_val) - self.target_coverage)])

        result = fixed_point_iteration(
            g_map, np.array([c_init]),
            max_iter=200, tol=1.0 / n, damping=0.3
        )
        return float(result['x'][0]), result

    def calibrate(self, n_bootstrap=500, seed=42):
        """完整校准流程.

        返回
        ----
        result : dict
        """
        # Step 1: ECH 初始估计
        c_ech, n_eff = self.ech_critical_value()

        # Step 2: Bootstrap 临界值
        c_boot, boot_stats = self.bootstrap_critical_value(n_bootstrap, seed)

        # Step 3: Regula Falsi 精炼
        c_rf, rf_result = self.refine_with_regula_falsi(boot_stats)

        # Step 4: 不动点精炼
        c_fp, fp_result = self.refine_with_fixed_point(boot_stats, c_rf)

        # 选择最佳
        candidates = [
            ('ECH', c_ech),
            ('Bootstrap', c_boot),
            ('RegulaFalsi', c_rf),
            ('FixedPoint', c_fp),
        ]

        # 验证覆盖率
        stats_sorted = np.sort(boot_stats)
        n = len(stats_sorted)
        best_name = 'Bootstrap'
        best_c = c_boot
        best_err = abs(np.searchsorted(stats_sorted, c_boot) / n - self.target_coverage)

        for name, c in candidates:
            if np.isfinite(c):
                cov = np.searchsorted(stats_sorted, c) / n
                err = abs(cov - self.target_coverage)
                if err < best_err:
                    best_err = err
                    best_name = name
                    best_c = c

        return {
            'critical_value': best_c,
            'method': best_name,
            'coverage_error': best_err,
            'candidates': dict(candidates),
            'n_eff': n_eff,
            'bootstrap_stats': boot_stats,
            'regula_falsi_result': rf_result,
            'fixed_point_result': fp_result,
        }

    def build_confidence_band(self, critical_value=None):
        """构建同时置信带.

        CI(x) = [μ̂(x) - c·SE(x),  μ̂(x) + c·SE(x)]

        参数
        ----
        critical_value : float or None

        返回
        ----
        ci_lower : ndarray
        ci_upper : ndarray
        """
        if critical_value is None:
            cal = self.calibrate()
            critical_value = cal['critical_value']

        ci_lower = self.mean_field - critical_value * self.stderr_field
        ci_upper = self.mean_field + critical_value * self.stderr_field

        return ci_lower, ci_upper, critical_value
