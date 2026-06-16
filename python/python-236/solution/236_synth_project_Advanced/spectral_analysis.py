"""
spectral_analysis.py — 谱函数重构与最大熵方法
==============================================
融合种子项目:
  [945_quad_trapezoid] : 梯形积分 → 谱函数积分方程
  [1158_shoh5301_Quick-MSD-Diffusivity-Calculator] : 正则化反演 → 谱函数提取

物理背景:
  关联函数的谱表示 (Lehmann 表示):
  C(t) = int_0^inf (d_omega / 2pi)  rho(omega) * K(omega, t)
  其中核函数 K(omega, t) = cosh(omega*(t - T/2)) / sinh(omega*T/2)
                           ≈ exp(-omega*t)  (大 T 极限)

  谱函数 rho(omega) 包含所有物理信息:
  - delta 函数峰 → 稳定粒子态
  - 连续谱 → 多粒子态/共振态
  - 谱函数宽度 → 衰变宽度

  最大熵方法 (MEM, Maximum Entropy Method):
  最大化 Q = alpha * S - chi^2/2
  S = int d_omega [rho - m - rho*ln(rho/m)]  (熵)
  m(omega) = 先验模型 (default model)

  Bryan's algorithm: 将 rho 参数化为
  rho(omega) = m(omega) * exp(sum_k a_k * u_k(omega))
  其中 u_k 为正交基函数.

核心公式:
  chi^2 = sum_{t,t'} (C(t) - C_model(t)) Cov^{-1}(t,t') (C(t') - C_model(t'))
  熵: S = int d_omega [rho - m - rho*ln(rho/m)]
  贝叶斯后验: P[rho|C] ∝ exp(alpha*S - chi^2/2) * P(alpha)
"""

import numpy as np
from typing import Tuple, Optional, Dict


class SpectralReconstructor:
    """谱函数重构器 (最大熵方法简化版).

    参数
    ----
    t_data : ndarray
        欧氏时间数据.
    C_data : ndarray
        关联函数数据.
    C_errors : ndarray
        误差.
    omega_max : float
        谱函数截止频率.
    n_omega : int
        频率网格点数.
    """

    def __init__(self, t_data: np.ndarray, C_data: np.ndarray,
                 C_errors: np.ndarray,
                 omega_max: float = 5.0,
                 n_omega: int = 100):
        self.t_data = np.asarray(t_data, dtype=np.float64)
        self.C_data = np.asarray(C_data, dtype=np.float64)
        self.C_errors = np.maximum(np.asarray(C_errors, dtype=np.float64),
                                    1e-15)
        self.N_t = len(self.t_data)
        self.Lt = int(np.max(self.t_data)) + 1

        self.omega_max = omega_max
        self.n_omega = n_omega
        self.omega = np.linspace(1e-6, omega_max, n_omega)
        self.d_omega = self.omega[1] - self.omega[0]

        # 核矩阵 K[t, omega]
        self.K = self._compute_kernel()

        # 先验模型 (平坦)
        self.default_model = np.ones(n_omega) * 0.1

    def _compute_kernel(self) -> np.ndarray:
        """计算核函数矩阵 K[t, omega].

        K(omega, t) = cosh(omega*(t - T/2)) / sinh(omega*T/2)
        对大 T 近似为 exp(-omega*t).
        """
        K = np.zeros((self.N_t, self.n_omega))
        T_half = self.Lt / 2.0
        for i, t in enumerate(self.t_data):
            arg = self.omega * T_half
            arg = np.minimum(arg, 500.0)  # 避免溢出
            sinh_arg = np.sinh(arg)
            sinh_arg = np.maximum(sinh_arg, 1e-300)
            cosh_val = np.cosh(self.omega * (t - T_half))
            cosh_val = np.minimum(cosh_val, 1e300)
            K[i] = cosh_val / sinh_arg
        return K

    def forward_model(self, rho: np.ndarray) -> np.ndarray:
        """正问题: 从谱函数计算关联函数.

        C(t) = int d_omega / (2*pi) * K(omega, t) * rho(omega)

        使用梯形积分 (源自 [945]).
        """
        integrand = self.K * rho[np.newaxis, :]
        C_model = np.trapz(integrand, self.omega, axis=1) / (2 * np.pi)
        return C_model

    def chi_squared(self, rho: np.ndarray) -> float:
        """计算 chi^2."""
        C_model = self.forward_model(rho)
        residual = (self.C_data - C_model) / self.C_errors
        return float(np.sum(residual ** 2))

    def entropy(self, rho: np.ndarray,
                m: Optional[np.ndarray] = None) -> float:
        """计算 Shannon-Jaynes 熵.

        S = int d_omega [rho - m - rho*ln(rho/m)]

        其中 m 为先验模型 (default model).

        参数
        ----
        rho : ndarray
            谱函数.
        m : ndarray, optional
            先验模型 (默认为 self.default_model).
        """
        if m is None:
            m = self.default_model
        rho = np.maximum(rho, 1e-300)
        m = np.maximum(m, 1e-300)
        integrand = rho - m - rho * np.log(rho / m)
        return float(np.trapz(integrand, self.omega))

    def q_function(self, rho: np.ndarray, alpha: float,
                   m: Optional[np.ndarray] = None) -> float:
        """目标函数 Q = alpha * S - chi^2 / 2.

        最大化 Q 给出最大熵谱函数.
        """
        return alpha * self.entropy(rho, m) - 0.5 * self.chi_squared(rho)

    # ------------------------------------------------------------------
    # 简化 Bryan 算法
    # ------------------------------------------------------------------
    def reconstruct_mem(self, alpha: float = 1.0,
                        n_iterations: int = 200,
                        learning_rate: float = 0.001,
                        seed: int = 42) -> Dict:
        """最大熵谱函数重构 (简化梯度上升).

        rho_{n+1} = rho_n + lr * dQ/d_rho

        dQ/d_rho = alpha * (-ln(rho/m)) + (K^T W (C - K*rho))

        参数
        ----
        alpha : float
            正则化参数.
        n_iterations : int
            迭代次数.
        learning_rate : float
            学习率.

        返回
        ----
        result : dict
            'rho': 谱函数, 'C_model': 拟合曲线, 'chi2': chi^2值
        """
        rng = np.random.default_rng(seed)
        # 初始谱: 先验模型 + 小扰动
        rho = self.default_model.copy() + 0.01 * rng.standard_normal(self.n_omega)
        rho = np.maximum(rho, 1e-10)

        # 权重矩阵 (对角)
        W = 1.0 / self.C_errors ** 2

        for iteration in range(n_iterations):
            # 正问题
            C_model = self.forward_model(rho)
            residual = self.C_data - C_model

            # 梯度 dQ/d_rho
            # 来自 chi^2: K^T * W * residual
            grad_chi2 = -(self.K.T @ (W * residual)) / (2 * np.pi)

            # 来自熵: -alpha * ln(rho/m)
            m = np.maximum(self.default_model, 1e-300)
            rho_safe = np.maximum(rho, 1e-300)
            grad_entropy = -alpha * np.log(rho_safe / m)

            # 总梯度
            grad = grad_entropy - grad_chi2

            # 更新 (梯度上升)
            rho = rho + learning_rate * grad
            rho = np.maximum(rho, 1e-10)  # 非负约束

            # 自适应学习率衰减
            if iteration % 50 == 0 and iteration > 0:
                learning_rate *= 0.8

        C_model = self.forward_model(rho)
        chi2 = self.chi_squared(rho)

        return {
            'rho': rho,
            'omega': self.omega,
            'C_model': C_model,
            'chi2': chi2,
            'alpha': alpha,
            'entropy': self.entropy(rho)
        }

    def alpha_scan(self, alpha_values: Optional[np.ndarray] = None,
                   n_iterations: int = 100) -> Dict:
        """扫描正则化参数 alpha.

        通过 L-curve 或证据近似确定最优 alpha.

        返回
        ----
        results : dict
        """
        if alpha_values is None:
            alpha_values = np.logspace(-2, 2, 8)

        results = {
            'alpha_values': alpha_values.tolist(),
            'chi2_values': [],
            'entropy_values': [],
            'norm_values': [],
            'optimal_alpha': None
        }

        best_score = -np.inf
        for alpha in alpha_values:
            res = self.reconstruct_mem(alpha=alpha,
                                       n_iterations=n_iterations)
            chi2 = res['chi2']
            S = res['entropy']
            rho_norm = np.trapz(res['rho'], self.omega)

            results['chi2_values'].append(chi2)
            results['entropy_values'].append(S)
            results['norm_values'].append(rho_norm)

            # 证据近似: 选择 chi2 ≈ N_t 的 alpha
            score = -abs(chi2 - self.N_t) / max(self.N_t, 1)
            if score > best_score:
                best_score = score
                results['optimal_alpha'] = alpha

        return results

    # ------------------------------------------------------------------
    # 求和规则检查
    # ------------------------------------------------------------------
    def sum_rule_check(self, rho: np.ndarray,
                       expected_integral: Optional[float] = None
                       ) -> Dict:
        """检查谱函数求和规则.

        对强子谱: int d_omega/(2pi) rho(omega) = f_H^2 * m_H^2 + ...

        参数
        ----
        rho : ndarray
            谱函数.
        expected_integral : float, optional
            期望的积分值.

        返回
        ----
        results : dict
        """
        integral = np.trapz(rho, self.omega) / (2 * np.pi)
        first_moment = np.trapz(self.omega * rho, self.omega) / (2 * np.pi)

        result = {
            'zeroth_moment': float(integral),
            'first_moment': float(first_moment),
            'positive_definite': bool(np.all(rho >= -1e-10)),
            'max_rho': float(np.max(rho)),
            'peak_position': float(self.omega[np.argmax(rho)])
        }

        if expected_integral is not None:
            result['relative_error'] = abs(
                integral - expected_integral) / max(abs(expected_integral), 1e-10)

        return result
