"""
stability_analysis.py — 拟合稳定性分析与系统误差评估
=====================================================
融合种子项目:
  [121_brusselator_ode] : 参数敏感性 → 系统稳定性分析
  [1065_Azamat-Mukhamediya_SRPM-ST] : 交叉验证 → 稳定性评估

物理背景:
  格点 QCD 关联函数拟合中的稳定性分析至关重要:
  1. 拟合范围稳定性: 改变 (t_min, t_max) 观察结果变化
  2. 态数目稳定性: 增加激发态数目检查结果收敛性
  3. 先验依赖性: 贝叶斯拟合中先验宽度的影响
  4. 相关矩阵条件数: 高条件数导致拟合不稳定

核心公式:
  相关矩阵条件数:  kappa(C) = lambda_max / lambda_min
  Q 值:  Q = 1 - Gamma(chi2/2, ndof/2) / Gamma(ndof/2)
  稳定性指标:  S = max_{t_min} |m(t_min) - m(t_min+1)| / sigma_m
  Akaike 权重: w_k = exp(-AIC_k/2) / sum_j exp(-AIC_j/2)
  AIC = chi2 + 2*k - 2*ln(L) ≈ chi2 + 2*k  (k = 参数数目)
  BIC = chi2 + k*ln(N)
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from correlator_fitting import CorrelatorFitter


class StabilityAnalyzer:
    """关联函数拟合稳定性分析器.

    参数
    ----
    fitter : CorrelatorFitter
    """

    def __init__(self, fitter: CorrelatorFitter):
        self.fitter = fitter
        self.results_history = []

    # ------------------------------------------------------------------
    # 拟合范围稳定性 (源自 [121_brusselator_ode] 的参数敏感性)
    # ------------------------------------------------------------------
    def t_min_scan(self, t_min_start: int = 1,
                   t_min_end: Optional[int] = None,
                   n_states: int = 1,
                   n_restarts: int = 3,
                   seed: int = 42) -> Dict:
        """扫描 t_min 观察提取质量的稳定性.

        对 t_min = t_min_start, ..., t_min_end,
        执行 n_states 态拟合并记录结果.

        返回
        ----
        results : dict
            't_min_values', 'masses', 'chi2_dof', 'stability_score'
        """
        if t_min_end is None:
            t_min_end = self.fitter.N_t // 2

        t_min_values = list(range(t_min_start, min(t_min_end + 1,
                                                     self.fitter.N_t - 1)))
        all_masses = []
        all_chi2 = []

        for t_min in t_min_values:
            if n_states == 1:
                res = self.fitter.fit_single_state(
                    t_min=t_min, n_restarts=n_restarts, seed=seed)
                all_masses.append(res['E'])
                all_chi2.append(res['chi2_dof'])
            else:
                res = self.fitter.fit_multi_state(
                    n_states=n_states, t_min=t_min,
                    n_restarts=n_restarts, seed=seed)
                all_masses.append(res['masses'][0])
                all_chi2.append(res['chi2_dof'])

        all_masses = np.array(all_masses)
        all_chi2 = np.array(all_chi2)

        # 稳定性评分 (源自 [121] 的参数敏感性概念):
        # S = max |m(t+1) - m(t)| / mean(sigma_m)
        diffs = np.abs(np.diff(all_masses))
        mean_mass = np.mean(all_masses) if len(all_masses) > 0 else 1.0
        stability = np.max(diffs) / max(mean_mass * 0.1, 1e-10)

        return {
            't_min_values': t_min_values,
            'masses': all_masses,
            'chi2_dof': all_chi2,
            'stability_score': float(stability),
            'mean_mass': float(np.mean(all_masses)),
            'std_mass': float(np.std(all_masses))
        }

    # ------------------------------------------------------------------
    # 态数目稳定性
    # ------------------------------------------------------------------
    def n_state_scan(self, max_states: int = 4,
                     t_min: int = 2,
                     n_restarts: int = 3,
                     seed: int = 42) -> Dict:
        """扫描态数目观察基态质量收敛性.

        对 N = 1, 2, ..., max_states, 执行 N 态拟合.

        返回
        ----
        results : dict with 'n_states', 'ground_masses', 'chi2_dof', 'AIC'
        """
        n_values = list(range(1, max_states + 1))
        ground_masses = []
        chi2_dofs = []
        aics = []

        for n in n_values:
            res = self.fitter.fit_multi_state(
                n_states=n, t_min=t_min, n_restarts=n_restarts, seed=seed)
            ground_masses.append(res['masses'][0])
            chi2_dofs.append(res['chi2_dof'])
            # AIC = chi2 + 2*k
            k = 2 * n  # 参数数
            chi2_val = res['chi2']
            aics.append(chi2_val + 2 * k)

        # Akaike 权重 (模型选择)
        aics = np.array(aics)
        aic_min = np.min(aics)
        weights = np.exp(-0.5 * (aics - aic_min))
        weights /= np.sum(weights)

        # 模型平均质量
        mass_avg = np.average(ground_masses, weights=weights)

        return {
            'n_states': n_values,
            'ground_masses': ground_masses,
            'chi2_dof': chi2_dofs,
            'AIC': aics.tolist(),
            'akaike_weights': weights.tolist(),
            'model_averaged_mass': float(mass_avg)
        }

    # ------------------------------------------------------------------
    # 相关矩阵条件数分析 (源自 [995_r8sm] 的矩阵分析)
    # ------------------------------------------------------------------
    def correlation_matrix_analysis(self, t_min: int = 0,
                                    t_max: Optional[int] = None) -> Dict:
        """分析相关矩阵的条件数和特征值分布.

        kappa(C) = lambda_max / lambda_min

        高条件数 (>10^6) 表示拟合可能不稳定.
        源自 [995_r8sm] 中对 Sherman-Morrison 矩阵条件数的关注.

        返回
        ----
        results : dict
            'condition_number', 'eigenvalues', 'effective_ndof'
        """
        if t_max is None:
            t_max = self.fitter.N_t

        mask = (self.fitter.t_data >= t_min) & (self.fitter.t_data < t_max)
        cov = self.fitter.C_cov[np.ix_(mask, mask)]

        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(eigvals)[::-1]  # 降序

        # 条件数
        if eigvals[-1] > 0:
            cond = eigvals[0] / eigvals[-1]
        else:
            cond = np.inf

        # 有效自由度 (SVD 截断)
        threshold = eigvals[0] * 1e-10 if eigvals[0] > 0 else 1e-15
        eff_ndof = np.sum(eigvals > threshold)

        # 相关系数矩阵
        diag = np.sqrt(np.diag(cov))
        diag = np.maximum(diag, 1e-300)
        corr = cov / np.outer(diag, diag)

        return {
            'condition_number': float(cond),
            'eigenvalues': eigvals.tolist(),
            'effective_ndof': int(eff_ndof),
            'mean_correlation': float(np.mean(np.abs(corr - np.diag(np.diag(corr))))),
            'is_well_conditioned': cond < 1e6
        }

    # ------------------------------------------------------------------
    # 交叉验证稳定性 (源自 [1065] 的 K-fold CV)
    # ------------------------------------------------------------------
    def cross_validate_fit(self, n_folds: int = 5,
                           t_min: int = 1,
                           n_states: int = 1,
                           seed: int = 42) -> Dict:
        """K-fold 交叉验证拟合稳定性 (源自 [1065] 的 stratified CV).

        将时间片分为 K 个折叠, 每次留出 1 个折叠,
        用剩余数据拟合并预测留出点.

        返回
        ----
        results : dict
            'fold_masses', 'mean_mass', 'std_mass', 'cv_score'
        """
        rng = np.random.default_rng(seed)
        N = self.fitter.N_t
        t_fit_range = self.fitter.t_data[t_min:]
        n_fit = len(t_fit_range)

        if n_fit < n_folds + 2:
            n_folds = max(2, n_fit - 2)

        indices = np.arange(n_fit)
        rng.shuffle(indices)
        folds = np.array_split(indices, n_folds)

        fold_masses = []
        fold_predictions = []

        for fold_idx in range(n_folds):
            test_idx = folds[fold_idx]
            train_idx = np.concatenate([folds[j] for j in range(n_folds)
                                         if j != fold_idx])

            # 训练集
            t_train = self.fitter.t_data[t_min:][train_idx]
            C_train = self.fitter.C_data[t_min:][train_idx]
            C_err_train = np.sqrt(np.diag(self.fitter.C_cov))[t_min:][train_idx]

            fitter_train = CorrelatorFitter(
                t_train, C_train, C_errors=C_err_train)

            # 拟合
            if n_states == 1:
                res = fitter_train.fit_single_state(n_restarts=3, seed=seed)
                mass = res['E']
                # 在测试集上预测
                t_test = self.fitter.t_data[t_min:][test_idx]
                C_pred = CorrelatorFitter.single_exp_model(
                    t_test, res['A'], res['E'], self.fitter.Lt)
            else:
                res = fitter_train.fit_multi_state(
                    n_states=n_states, n_restarts=3, seed=seed)
                mass = res['masses'][0]
                t_test = self.fitter.t_data[t_min:][test_idx]
                C_pred = CorrelatorFitter.multi_exp_model(
                    t_test,
                    np.array([v for pair in zip(res['amplitudes'],
                                                  res['masses'])
                              for v in pair]),
                    n_states, self.fitter.Lt)

            fold_masses.append(mass)

            # 预测误差
            C_test = self.fitter.C_data[t_min:][test_idx]
            fold_predictions.append(np.mean((C_test - C_pred) ** 2))

        fold_masses = np.array(fold_masses)
        cv_score = np.mean(fold_predictions)

        return {
            'fold_masses': fold_masses.tolist(),
            'mean_mass': float(np.mean(fold_masses)),
            'std_mass': float(np.std(fold_masses)),
            'cv_mse': float(cv_score),
            'n_folds': n_folds
        }

    # ------------------------------------------------------------------
    # 综合稳定性报告
    # ------------------------------------------------------------------
    def comprehensive_stability_report(self, t_min_default: int = 2,
                                       seed: int = 42) -> Dict:
        """生成综合稳定性分析报告.

        包含:
        1. t_min 扫描
        2. 态数目扫描
        3. 条件数分析
        4. 交叉验证

        返回
        ----
        report : dict
            包含所有分析结果和总体稳定性评分.
        """
        # 1. t_min 扫描
        tmin_result = self.t_min_scan(t_min_start=1, n_states=1, seed=seed)

        # 2. 态数目扫描
        nstate_result = self.n_state_scan(max_states=3, t_min=t_min_default,
                                          seed=seed)

        # 3. 条件数
        cond_result = self.correlation_matrix_analysis()

        # 4. 交叉验证
        cv_result = self.cross_validate_fit(n_folds=3, t_min=t_min_default,
                                            seed=seed)

        # 综合稳定性评分 (0-1, 1=完全稳定)
        scores = []
        # t_min 稳定性: 相对标准差小则好
        rel_std = tmin_result['std_mass'] / max(tmin_result['mean_mass'], 1e-10)
        scores.append(max(0, 1.0 - rel_std))
        # 条件数: 条件数低则好
        cond = cond_result['condition_number']
        scores.append(max(0, 1.0 - min(np.log10(max(cond, 1)) / 10, 1)))
        # CV 一致性
        cv_std = cv_result['std_mass'] / max(cv_result['mean_mass'], 1e-10)
        scores.append(max(0, 1.0 - cv_std))

        overall = float(np.mean(scores))

        return {
            't_min_scan': tmin_result,
            'n_state_scan': nstate_result,
            'condition_analysis': cond_result,
            'cross_validation': cv_result,
            'overall_stability': overall,
            'is_stable': overall > 0.5
        }
