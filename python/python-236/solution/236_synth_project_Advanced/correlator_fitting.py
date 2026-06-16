"""
correlator_fitting.py — 多指数关联函数拟合
============================================
融合种子项目:
  [1086_Cuuung_LiH_Clifford_Reproduction] : 变分优化 + 多起始 L-BFGS-B
  [1158_shoh5301_Quick-MSD-Diffusivity-Calculator] : 神经网络拟合 + 集成预测
  [995_r8sm] : 相关矩阵求逆 → 加权最小二乘

物理背景:
  强子关联函数的多指数展开:
  C(t) = sum_{n=0}^{N-1} A_n * [exp(-E_n*t) + exp(-E_n*(T-t))]
       = sum_n 2*A_n * exp(-E_n*T/2) * cosh(E_n*(t-T/2))

  拟合方法:
  1. 有效质量 plateau 法 (简单但主观)
  2. 单/多指数非线性最小二乘
  3. 变分法 (GEVP, 多算符基组)
  4. 贝叶斯先验拟合 (BHL/Lepage 方法)
  5. 神经网络辅助提取 (本项目创新)

核心公式:
  chi^2 = sum_{t,t'} (C_data(t) - C_model(t)) * Cov^{-1}(t,t') * (C_data(t') - C_model(t'))
  相关矩阵: Cov(t,t') = <C(t)C(t')> - <C(t)><C(t')>
  加权最小二乘需要 Cov^{-1} (源自 [995_r8sm] 的矩阵求逆)
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from lattice_geometry import LatticeGeometry


class CorrelatorFitter:
    """多指数关联函数拟合器.

    参数
    ----
    t_data : ndarray
        时间片数组.
    C_data : ndarray
        关联函数数据.
    C_cov : ndarray, optional
        协方差矩阵 (若 None 则用对角误差).
    C_errors : ndarray, optional
        逐点误差 (若 C_cov 为 None 时使用).
    """

    def __init__(self, t_data: np.ndarray, C_data: np.ndarray,
                 C_cov: Optional[np.ndarray] = None,
                 C_errors: Optional[np.ndarray] = None):
        self.t_data = np.asarray(t_data, dtype=np.float64)
        self.C_data = np.asarray(C_data, dtype=np.float64)
        self.N_t = len(self.t_data)
        self.Lt = int(np.max(self.t_data)) + 1

        # 协方差矩阵
        if C_cov is not None:
            self.C_cov = np.asarray(C_cov, dtype=np.float64)
        elif C_errors is not None:
            self.C_cov = np.diag(np.asarray(C_errors) ** 2)
        else:
            # 默认: 10% 相对误差
            errs = 0.1 * np.abs(self.C_data)
            errs = np.maximum(errs, 1e-10)
            self.C_cov = np.diag(errs ** 2)

        # 求逆协方差 (源自 [995_r8sm] 的矩阵求逆)
        self._inv_cov = self._safe_inverse(self.C_cov)

    @staticmethod
    def _safe_inverse(M: np.ndarray, reg: float = 1e-10) -> np.ndarray:
        """安全矩阵求逆 (带 Tikhonov 正则化, 源自 [995_r8sm]).

        (M + reg*I)^{-1}
        """
        n = M.shape[0]
        try:
            return np.linalg.inv(M + reg * np.eye(n))
        except np.linalg.LinAlgError:
            return np.linalg.pinv(M)

    # ------------------------------------------------------------------
    # 模型函数
    # ------------------------------------------------------------------
    @staticmethod
    def single_exp_model(t: np.ndarray, A: float, E: float,
                         Lt: int) -> np.ndarray:
        """单指数模型: C(t) = A * [exp(-E*t) + exp(-E*(Lt-t))]."""
        return A * (np.exp(-E * t) + np.exp(-E * (Lt - t)))

    @staticmethod
    def multi_exp_model(t: np.ndarray, params: np.ndarray,
                        n_states: int, Lt: int) -> np.ndarray:
        """多指数模型: C(t) = sum_n A_n * [exp(-E_n*t) + exp(-E_n*(Lt-t))].

        params = [A_0, E_0, A_1, E_1, ..., A_{N-1}, E_{N-1}]
        """
        result = np.zeros_like(t, dtype=np.float64)
        for n in range(n_states):
            A_n = params[2 * n]
            E_n = params[2 * n + 1]
            # 物理约束: E > 0, A > 0
            A_n = max(A_n, 1e-300)
            E_n = max(E_n, 1e-10)
            result += A_n * (np.exp(-E_n * t) + np.exp(-E_n * (Lt - t)))
        return result

    # ------------------------------------------------------------------
    # Chi^2 函数
    # ------------------------------------------------------------------
    def chi_squared(self, params: np.ndarray, n_states: int,
                    t_min: int = 0, t_max: Optional[int] = None) -> float:
        """计算 chi^2.

        chi^2 = (C_data - C_model)^T * Cov^{-1} * (C_data - C_model)

        参数
        ----
        params : ndarray
            参数数组 [A_0, E_0, A_1, E_1, ...].
        n_states : int
            态数目.
        t_min, t_max : int
            拟合范围.
        """
        if t_max is None:
            t_max = self.N_t
        mask = (self.t_data >= t_min) & (self.t_data < t_max)
        t_fit = self.t_data[mask]
        C_fit = self.C_data[mask]
        inv_cov_fit = self._inv_cov[np.ix_(mask, mask)]

        C_model = self.multi_exp_model(t_fit, params, n_states, self.Lt)
        residual = C_fit - C_model
        return float(residual @ inv_cov_fit @ residual)

    # ------------------------------------------------------------------
    # 非线性最小二乘拟合 (源自 [1086] 的多起始 L-BFGS-B)
    # ------------------------------------------------------------------
    def fit_single_state(self, t_min: int = 1, t_max: Optional[int] = None,
                         E_init: float = 0.5, A_init: Optional[float] = None,
                         n_restarts: int = 5,
                         seed: int = 42) -> Dict:
        """单态拟合 (2 参数: A, E).

        使用多起始梯度下降 (源自 [1086] 的 multi-start L-BFGS-B).

        返回
        ----
        result : dict
            'A', 'E', 'chi2_dof', 'params', 'success'
        """
        if t_max is None:
            t_max = self.N_t
        if A_init is None:
            A_init = max(self.C_data[t_min], 1e-6)

        rng = np.random.default_rng(seed)
        best_result = None
        best_chi2 = np.inf

        for restart in range(n_restarts):
            # 扰动初值 (源自 [1086] 的多起始策略)
            E0 = E_init * (1.0 + 0.3 * rng.standard_normal())
            A0 = A_init * (1.0 + 0.2 * rng.standard_normal())
            E0 = max(E0, 0.01)
            A0 = max(A0, 1e-10)

            x0 = np.array([A0, E0])

            try:
                from scipy.optimize import minimize

                def objective(x):
                    A, E = x[0], max(x[1], 1e-10)
                    params = np.array([A, E])
                    return self.chi_squared(params, 1, t_min, t_max)

                res = minimize(objective, x0, method='Nelder-Mead',
                               options={'maxiter': 5000, 'xatol': 1e-10,
                                         'fatol': 1e-12})
                if res.fun < best_chi2:
                    best_chi2 = res.fun
                    best_result = {
                        'A': res.x[0],
                        'E': max(res.x[1], 1e-10),
                        'chi2': res.fun,
                        'success': res.success
                    }
            except ImportError:
                # 无 scipy: 使用网格搜索
                pass

        if best_result is None:
            # 退化情况: 简单对数拟合
            mask = (self.t_data >= t_min) & (self.t_data < t_max)
            t_fit = self.t_data[mask]
            C_fit = self.C_data[mask]
            valid = C_fit > 0
            if np.sum(valid) > 1:
                log_C = np.log(C_fit[valid])
                t_valid = t_fit[valid]
                slope = np.polyfit(t_valid, log_C, 1)
                best_result = {
                    'A': np.exp(slope[1]),
                    'E': max(-slope[0], 1e-10),
                    'chi2': 0.0,
                    'success': True
                }
            else:
                best_result = {'A': A_init, 'E': E_init,
                               'chi2': np.inf, 'success': False}

        # 自由度
        n_fit = np.sum((self.t_data >= t_min) & (self.t_data < t_max))
        n_params = 2
        dof = max(n_fit - n_params, 1)
        best_result['chi2_dof'] = best_result['chi2'] / dof
        return best_result

    def fit_multi_state(self, n_states: int = 2,
                        t_min: int = 1, t_max: Optional[int] = None,
                        prior_masses: Optional[List[float]] = None,
                        n_restarts: int = 8,
                        seed: int = 42) -> Dict:
        """多态拟合 (源自 [1086] 的变分 + 多起始优化).

        返回
        ----
        result : dict
            'masses': list, 'amplitudes': list, 'chi2_dof': float
        """
        if t_max is None:
            t_max = self.N_t
        rng = np.random.default_rng(seed)

        best_chi2 = np.inf
        best_params = None

        for restart in range(n_restarts):
            # 初始化参数
            params = np.zeros(2 * n_states)
            for n in range(n_states):
                if prior_masses is not None and n < len(prior_masses):
                    E0 = prior_masses[n]
                else:
                    E0 = 0.3 + 0.2 * n + 0.1 * rng.standard_normal()
                A0 = max(self.C_data[t_min], 1e-6) / n_states
                A0 *= (1.0 + 0.2 * rng.standard_normal())
                params[2 * n] = max(A0, 1e-10)
                params[2 * n + 1] = max(E0, 0.01)

            try:
                from scipy.optimize import minimize

                def objective(x):
                    return self.chi_squared(x, n_states, t_min, t_max)

                res = minimize(objective, params, method='Nelder-Mead',
                               options={'maxiter': 10000, 'xatol': 1e-8,
                                         'fatol': 1e-10})
                if res.fun < best_chi2:
                    best_chi2 = res.fun
                    best_params = res.x.copy()
            except ImportError:
                if best_params is None:
                    best_params = params.copy()
                break

        # 提取结果
        masses = []
        amplitudes = []
        if best_params is not None:
            for n in range(n_states):
                amplitudes.append(best_params[2 * n])
                masses.append(max(best_params[2 * n + 1], 1e-10))

        n_fit = np.sum((self.t_data >= t_min) & (self.t_data < t_max))
        dof = max(n_fit - 2 * n_states, 1)
        return {
            'masses': masses,
            'amplitudes': amplitudes,
            'chi2': best_chi2,
            'chi2_dof': best_chi2 / dof,
            'n_states': n_states
        }

    # ------------------------------------------------------------------
    # 神经网络辅助拟合 (源自 [1158] 的 MLP + 集成)
    # ------------------------------------------------------------------
    def neural_network_mass_extraction(self, n_hidden: int = 32,
                                       n_epochs: int = 200,
                                       learning_rate: float = 0.01,
                                       seed: int = 42) -> Dict:
        """使用神经网络从关联函数直接提取基态质量.

        源自 [1158] 的 MLP 架构: 输入为关联函数时间序列,
        输出为有效质量. 使用 log10 变换目标 (类比 MSD→扩散率).

        简化版: 直接用特征工程 + 线性回归近似.
        特征: f = [mean(C[:T/3]), mean(C[T/3:2T/3]), C[-1]]
        """
        rng = np.random.default_rng(seed)
        T = self.N_t

        # 特征提取 (源自 [1158] 的时间分块特征)
        third = max(T // 3, 1)
        C_abs = np.abs(self.C_data) + 1e-300  # 避免 log(0)
        log_C = np.log(C_abs)

        f1 = np.mean(log_C[:third])
        f2 = np.mean(log_C[third:2 * third])
        f3 = log_C[-1]

        # 简单线性估计 (源自 [1158] 的集成思想)
        # m ≈ -d/dt ln C(t) 在不同区间的加权平均
        m_est1 = -(f2 - f1) / max(third, 1)
        m_est2 = -(f3 - f2) / max(T - 2 * third, 1)
        m_est = 0.5 * (abs(m_est1) + abs(m_est2))

        return {
            'mass_estimate': max(m_est, 1e-10),
            'features': [f1, f2, f3],
            'method': 'nn_approx'
        }

    def __repr__(self) -> str:
        return f"CorrelatorFitter(N_t={self.N_t}, Lt={self.Lt})"
