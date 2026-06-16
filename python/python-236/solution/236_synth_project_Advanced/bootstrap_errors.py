"""
bootstrap_errors.py — 统计误差分析: Bootstrap 与自助训练
=========================================================
融合种子项目:
  [1065_Azamat-Mukhamediya_SRPM-ST] : 自助训练 + 伪标签 → Bootstrap 误差
  [1158_shoh5301_Quick-MSD-Diffusivity-Calculator] : 集成预测 → Bootstrap 集成

物理背景:
  格点 QCD 中, 关联函数是从 Monte Carlo 生成的规范场配置上测量的.
  统计误差分析至关重要:
  - Bootstrap: 从 N 个配置中重采样 N 个 (有放回), 重复拟合
  - Jackknife: 删除 1 个配置, 从剩余 N-1 个拟合
  - 自相关: 连续配置间存在自相关, 有效统计量 N_eff < N

核心公式:
  Bootstrap:
  对 b = 1, ..., B:
    1. 从 {1,...,N} 中有放回抽取 N 个索引
    2. 计算 bootstrap 样本的均值
    3. 拟合得到 m_b
  误差: sigma_m = std({m_b})

  Jackknife:
  对 i = 1, ..., N:
    m_{-i} = 从除第 i 个外所有配置拟合的质量
  m_JK = (N*m_all - sum_i m_{-i}) / (N-1)
  sigma_m = sqrt((N-1)/N * sum_i (m_{-i} - m_JK)^2)

  自相关时间:
  tau_int = 1/2 + sum_{t=1}^{W} rho(t)
  N_eff = N / (2*tau_int)

  伪标签自训练 (源自 [1065]):
  1. 用有标签数据训练初始模型
  2. 对无标签数据生成伪标签
  3. 合并数据重新训练
  4. 迭代直到收敛
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Callable
from correlator_fitting import CorrelatorFitter


class BootstrapAnalyzer:
    """Bootstrap 误差分析器.

    参数
    ----
    C_ensemble : ndarray, shape (N_configs, N_t)
        各配置的关联函数数据.
    t_data : ndarray
        时间片.
    """

    def __init__(self, C_ensemble: np.ndarray, t_data: np.ndarray):
        self.C_ensemble = np.asarray(C_ensemble, dtype=np.float64)
        self.N_configs, self.N_t = self.C_ensemble.shape
        self.t_data = np.asarray(t_data)
        self.Lt = int(np.max(t_data)) + 1

        if self.N_configs < 2:
            raise ValueError(f"至少需要 2 个配置, 当前 {self.N_configs}")

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------
    def bootstrap_fit(self, n_bootstrap: int = 200,
                      t_min: int = 1,
                      seed: int = 42) -> Dict:
        """Bootstrap 拟合误差分析.

        参数
        ----
        n_bootstrap : int
            Bootstrap 重采样次数.
        t_min : int
            拟合起始时间.

        返回
        ----
        results : dict
            'masses': ndarray, 'mean_mass', 'std_mass',
            'confidence_interval': (lo, hi)
        """
        rng = np.random.default_rng(seed)
        masses = np.zeros(n_bootstrap)

        for b in range(n_bootstrap):
            # 重采样 (有放回)
            indices = rng.integers(0, self.N_configs, size=self.N_configs)
            C_boot = np.mean(self.C_ensemble[indices], axis=0)
            C_err = np.std(self.C_ensemble[indices], axis=0) / np.sqrt(self.N_configs)
            C_err = np.maximum(C_err, 1e-15)

            fitter = CorrelatorFitter(self.t_data, C_boot, C_errors=C_err)
            res = fitter.fit_single_state(t_min=t_min, n_restarts=2,
                                          seed=seed + b)
            masses[b] = res['E']

        mean_mass = float(np.mean(masses))
        std_mass = float(np.std(masses))
        ci_lo = float(np.percentile(masses, 2.5))
        ci_hi = float(np.percentile(masses, 97.5))

        return {
            'masses': masses,
            'mean_mass': mean_mass,
            'std_mass': std_mass,
            'confidence_interval': (ci_lo, ci_hi),
            'n_bootstrap': n_bootstrap
        }

    # ------------------------------------------------------------------
    # Jackknife
    # ------------------------------------------------------------------
    def jackknife_fit(self, t_min: int = 1,
                      seed: int = 42) -> Dict:
        """Jackknife 误差分析.

        删除-1 jackknife: 每次删除一个配置, 用剩余 N-1 个拟合.
        """
        masses_jk = np.zeros(self.N_configs)

        for i in range(self.N_configs):
            mask = np.ones(self.N_configs, dtype=bool)
            mask[i] = False
            C_jk = np.mean(self.C_ensemble[mask], axis=0)
            C_err = np.std(self.C_ensemble[mask], axis=0) / np.sqrt(self.N_configs - 1)
            C_err = np.maximum(C_err, 1e-15)

            fitter = CorrelatorFitter(self.t_data, C_jk, C_errors=C_err)
            res = fitter.fit_single_state(t_min=t_min, n_restarts=2,
                                          seed=seed + i)
            masses_jk[i] = res['E']

        # Jackknife 估计
        mass_all = np.mean(self.C_ensemble, axis=0)
        fitter_all = CorrelatorFitter(self.t_data, mass_all)
        res_all = fitter_all.fit_single_state(t_min=t_min, n_restarts=3, seed=seed)
        m_all = res_all['E']

        # Jackknife 质量
        m_jk = self.N_configs * m_all - (self.N_configs - 1) * np.mean(masses_jk)
        # Jackknife 误差
        sigma_jk = np.sqrt((self.N_configs - 1) / self.N_configs
                           * np.sum((masses_jk - np.mean(masses_jk)) ** 2))

        return {
            'jackknife_masses': masses_jk,
            'mass_jk': float(m_jk),
            'sigma_jk': float(sigma_jk),
            'mass_all': float(m_all)
        }

    # ------------------------------------------------------------------
    # 自相关分析
    # ------------------------------------------------------------------
    def autocorrelation_analysis(self, max_lag: Optional[int] = None
                                 ) -> Dict:
        """时间序列自相关分析.

        计算关联函数的自相关函数:
        Gamma(t) = <C(t_0) * C(t_0 + t)> - <C>^2

        积分自相关时间:
        tau_int = 1/2 + sum_{t=1}^{W} rho(t)
        其中 W 为自适应窗宽 (W ~ 10 * tau_int).

        返回
        ----
        results : dict
        """
        # 使用 C[t=2] 作为观测量
        t_idx = min(2, self.N_t - 1)
        series = self.C_ensemble[:, t_idx]
        N = len(series)

        if max_lag is None:
            max_lag = min(N // 3, 50)

        mean_s = np.mean(series)
        var_s = np.var(series)
        if var_s < 1e-300:
            return {'tau_int': 0.5, 'N_eff': N, 'rho': np.zeros(max_lag + 1)}

        # 自相关函数
        rho = np.zeros(max_lag + 1)
        for lag in range(max_lag + 1):
            if lag == 0:
                rho[lag] = 1.0
            else:
                c = np.mean((series[:N - lag] - mean_s)
                             * (series[lag:] - mean_s))
                rho[lag] = c / var_s

        # 积分自相关时间 (自适应窗)
        tau_int = 0.5
        for t in range(1, max_lag + 1):
            if rho[t] < 0.05:  # 窗宽判据
                break
            tau_int += rho[t]

        N_eff = N / (2.0 * max(tau_int, 0.5))

        return {
            'tau_int': float(tau_int),
            'N_eff': float(N_eff),
            'rho': rho.tolist(),
            'is_equilibrated': N_eff > 5
        }

    # ------------------------------------------------------------------
    # 伪标签自训练增强 (源自 [1065] 的 SRPM-ST)
    # ------------------------------------------------------------------
    def self_training_mass_enhancement(self,
                                       n_labeled: Optional[int] = None,
                                       n_rounds: int = 3,
                                       confidence_threshold: float = 0.7,
                                       seed: int = 42) -> Dict:
        """自助训练质量提取增强 (源自 [1065] 的半监督学习).

        思路: 将部分配置标记为"有标签" (高质量拟合),
        用其训练一个简单模型预测"无标签"配置的质量,
        将高置信度预测加入训练集.

        参数
        ----
        n_labeled : int
            初始有标签配置数.
        n_rounds : int
            自训练轮数.
        confidence_threshold : float
            伪标签置信度阈值.

        返回
        ----
        results : dict
        """
        rng = np.random.default_rng(seed)
        N = self.N_configs

        if n_labeled is None:
            n_labeled = max(N // 3, 3)

        # Step 1: 对有标签配置拟合质量
        perm = rng.permutation(N)
        labeled_idx = set(perm[:n_labeled].tolist())
        unlabeled_idx = set(perm[n_labeled:].tolist())

        labeled_masses = {}
        for idx in labeled_idx:
            C_conf = self.C_ensemble[idx]
            C_err = np.abs(C_conf) * 0.05 + 1e-15
            fitter = CorrelatorFitter(self.t_data, C_conf, C_errors=C_err)
            res = fitter.fit_single_state(n_restarts=2, seed=seed + idx)
            labeled_masses[idx] = res['E']

        # Step 2: 自训练循环
        for round_idx in range(n_rounds):
            if len(unlabeled_idx) == 0:
                break

            # 当前有标签数据的平均质量作为伪标签
            mean_mass = np.mean(list(labeled_masses.values()))
            std_mass = max(np.std(list(labeled_masses.values())), 0.01)

            # 对无标签配置生成伪标签
            pseudo_labels = {}
            for idx in list(unlabeled_idx):
                C_conf = self.C_ensemble[idx]
                C_err = np.abs(C_conf) * 0.05 + 1e-15
                fitter = CorrelatorFitter(self.t_data, C_conf, C_errors=C_err)
                res = fitter.fit_single_state(n_restarts=2,
                                              E_init=mean_mass,
                                              seed=seed + idx + 1000)
                # 置信度: 基于 chi2/dof
                confidence = max(0, 1.0 - res['chi2_dof'] / 10.0)
                if confidence >= confidence_threshold:
                    pseudo_labels[idx] = res['E']

            # 将高置信度伪标签加入训练集
            for idx, mass in pseudo_labels.items():
                labeled_masses[idx] = mass
                unlabeled_idx.discard(idx)

        all_masses = np.array(list(labeled_masses.values()))
        return {
            'final_labeled_count': len(labeled_masses),
            'pseudo_label_count': len(labeled_masses) - n_labeled,
            'mass_mean': float(np.mean(all_masses)),
            'mass_std': float(np.std(all_masses)),
            'n_rounds': n_rounds
        }

    # ------------------------------------------------------------------
    # 综合误差报告
    # ------------------------------------------------------------------
    def comprehensive_error_analysis(self, t_min: int = 1,
                                     seed: int = 42) -> Dict:
        """综合误差分析报告 (Bootstrap + Jackknife + 自相关)."""
        # Bootstrap
        boot = self.bootstrap_fit(n_bootstrap=100, t_min=t_min, seed=seed)

        # Jackknife
        jk = self.jackknife_fit(t_min=t_min, seed=seed)

        # 自相关
        ac = self.autocorrelation_analysis()

        return {
            'bootstrap': {
                'mean': boot['mean_mass'],
                'std': boot['std_mass'],
                'ci_95': boot['confidence_interval']
            },
            'jackknife': {
                'mass': jk['mass_jk'],
                'std': jk['sigma_jk']
            },
            'autocorrelation': {
                'tau_int': ac['tau_int'],
                'N_eff': ac['N_eff']
            },
            'corrected_error': boot['std_mass'] * np.sqrt(
                2.0 * max(ac['tau_int'], 0.5))
        }
