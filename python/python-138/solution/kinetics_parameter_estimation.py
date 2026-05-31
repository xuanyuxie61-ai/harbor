"""
基于 QR 分解的反应动力学参数最小二乘估计
=============================================
利用正交三角化方法求解超定线性系统，估计微反应器内复杂反应网络的
动力学参数。

核心数学：
    给定实验数据 (C_i^{exp}, T_i, r_i^{exp})，i=1..m，拟合模型

        r^{model}(C,T;θ) = θ_1 exp(-θ_2/(R T)) C^{θ_3}

    对非线性模型做线性化处理（对数变换）：

        ln(r) ≈ ln(θ_1) - θ_2/(R T) + θ_3 ln(C)

    令 β = [ln(θ_1), θ_2, θ_3]^T，设计矩阵 X 的行向量为
        [1, -1/(R T_i), ln(C_i)]

    则线性最小二乘问题为：
        min || X β - y ||_2²

    通过 QR 分解 X = Q R 求解：
        R β = Q^T y

    其中 Q 为正交矩阵，R 为上三角矩阵。
"""

import numpy as np
from typing import Tuple, Optional


class KineticsParameterEstimator:
    """
    基于 QR 分解的最小二乘动力学参数估计器。
    """

    def __init__(self, R_gas: float = 8.314):
        self.R_gas = R_gas

    def _householder_reflection(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        计算 Householder 反射向量 v，使得 (I - 2 v v^T / (v^T v)) x = ||x|| e_1。
        返回 (v, beta) 其中 beta = 2 / (v^T v)。
        """
        n = len(x)
        if n == 0:
            return np.array([]), 0.0
        sigma = np.dot(x[1:], x[1:])
        v = np.zeros(n)
        v[0] = 1.0
        v[1:] = x[1:]
        if sigma == 0.0:
            return v, 0.0
        mu = np.sqrt(x[0] ** 2 + sigma)
        if x[0] <= 0.0:
            v[0] = x[0] - mu
        else:
            v[0] = -sigma / (x[0] + mu)
        beta = 2.0 / (v[0] ** 2 + sigma)
        return v, beta

    def qr_factorize(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        对 m×n 矩阵 A 进行 QR 分解（Householder 方法）。
        返回 (Q, R)，其中 Q 为 m×m 正交矩阵，R 为 m×n 上三角矩阵。
        """
        m, n = A.shape
        if m < n:
            raise ValueError("QR 分解要求行数不少于列数")
        R = A.copy().astype(float)
        Q = np.eye(m)
        for k in range(min(m - 1, n)):
            x = R[k:, k].copy()
            v, beta = self._householder_reflection(x)
            if beta == 0.0:
                continue
            # 应用 Householder 变换到 R
            for j in range(k, n):
                tau = beta * np.dot(v, R[k:, j])
                R[k:, j] -= tau * v
            # 累积 Q
            for j in range(m):
                tau = beta * np.dot(v, Q[k:, j])
                Q[k:, j] -= tau * v
        Q = Q.T
        return Q, R

    def solve_least_squares(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        使用 QR 分解求解最小二乘问题 min ||A x - b||_2。
        """
        m, n = A.shape
        if len(b) != m:
            raise ValueError("A 与 b 维度不匹配")
        Q, R = self.qr_factorize(A)
        # 取 R 的前 n 行形成上三角矩阵
        R_tri = R[:n, :n]
        Qtb = Q[:m, :n].T @ b
        # 回代求解
        x = self._back_substitution(R_tri, Qtb)
        return x

    def _back_substitution(self, U: np.ndarray, b: np.ndarray) -> np.ndarray:
        """上三角矩阵回代。"""
        n = len(b)
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            if abs(U[i, i]) < 1.0e-14:
                x[i] = 0.0
            else:
                x[i] = (b[i] - np.dot(U[i, i + 1 :], x[i + 1 :])) / U[i, i]
        return x

    def _predict_rate(
        self, C: np.ndarray, T: np.ndarray, params: np.ndarray
    ) -> np.ndarray:
        """由参数向量 [lnA, Ea, n] 预测反应速率。"""
        lnA, Ea, n = params
        A = np.exp(lnA)
        T_safe = np.maximum(T, 200.0)
        C_safe = np.maximum(C, 1.0e-12)
        rate = A * np.exp(-Ea / (self.R_gas * T_safe)) * (C_safe ** max(n, 0.0))
        return np.maximum(rate, 1.0e-20)

    def _gauss_newton_step(
        self,
        C: np.ndarray,
        T: np.ndarray,
        r_exp: np.ndarray,
        params: np.ndarray,
    ) -> np.ndarray:
        """
        Gauss-Newton 单步更新：
            params^{new} = params - (J^T J)^{-1} J^T f
        其中 f_i = r_pred_i - r_exp_i，J 为 Jacobian。
        """
        r_pred = self._predict_rate(C, T, params)
        f = r_pred - r_exp
        lnA, Ea, n = params
        A = np.exp(lnA)
        T_safe = np.maximum(T, 200.0)
        C_safe = np.maximum(C, 1.0e-12)
        exp_term = np.exp(-Ea / (self.R_gas * T_safe))

        # Jacobian 列: dr/d(lnA), dr/dEa, dr/dn
        dr_dlnA = r_pred  # dr/d(lnA) = A * exp(...) * C^n = r_pred
        dr_dEa = -r_pred / (self.R_gas * T_safe)
        dr_dn = r_pred * np.log(C_safe)

        J = np.column_stack([dr_dlnA, dr_dEa, dr_dn])
        # Levenberg-Marquardt: (J^T J + λ I) δ = -J^T f
        JTJ = J.T @ J
        JTf = J.T @ f
        # 添加阻尼
        lam_damp = 1.0e-3 * np.trace(JTJ) / 3.0
        H = JTJ + lam_damp * np.eye(3)
        try:
            delta = np.linalg.solve(H, -JTf)
        except np.linalg.LinAlgError:
            delta = -JTf / (np.diag(JTJ) + 1.0e-10)
        return delta

    def estimate_arrhenius_parameters(
        self,
        concentrations: np.ndarray,
        temperatures: np.ndarray,
        rates: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        """
        基于实验数据估计 Arrhenius 参数 (A, Ea, n)。

        采用 Levenberg-Marquardt / Gauss-Newton 非线性最小二乘
        直接在原尺度上优化：
            min || r_exp - A exp(-Ea/(R T)) C^n ||_2²

        输入:
            concentrations: 浓度数组 [mol/m³]
            temperatures:   温度数组 [K]
            rates:          反应速率数组 [mol/(m³·s)]

        返回:
            (A_est, Ea_est, n_est, residual_norm)
        """
        concentrations = np.asarray(concentrations, dtype=float)
        temperatures = np.asarray(temperatures, dtype=float)
        rates = np.asarray(rates, dtype=float)

        if len(concentrations) < 3:
            raise ValueError("至少需要 3 组数据才能估计 3 个参数")

        # 过滤非正数据
        valid = (concentrations > 0.0) & (temperatures > 0.0) & (rates > 0.0)
        if np.sum(valid) < 3:
            raise ValueError("有效数据点不足")

        C = concentrations[valid]
        T = temperatures[valid]
        r = rates[valid]

        # 初始猜测：对 n 在 [0, 2.5] 上格点搜索，对每个 n 做 A-Ea 线性回归
        best_res = float("inf")
        best_params = np.array([np.log(1.0e4), 10000.0, 1.0])
        for n_try in np.linspace(0.0, 2.5, 26):
            y_try = np.log(r) - n_try * np.log(C)
            X_try = np.column_stack([np.ones(len(y_try)), -1.0 / (self.R_gas * T)])
            try:
                beta_try, *_ = np.linalg.lstsq(X_try, y_try, rcond=None)
                lnA_try = beta_try[0]
                Ea_try = max(beta_try[1], 100.0)
                r_pred_try = np.exp(lnA_try) * np.exp(-Ea_try / (self.R_gas * T)) * (C ** n_try)
                res_try = np.linalg.norm(r - r_pred_try)
                if res_try < best_res:
                    best_res = res_try
                    best_params = np.array([lnA_try, Ea_try, n_try])
            except Exception:
                continue

        params = best_params

        # Gauss-Newton 迭代
        for it in range(100):
            delta = self._gauss_newton_step(C, T, r, params)
            # 线搜索
            alpha = 1.0
            current_res = np.linalg.norm(self._predict_rate(C, T, params) - r)
            for _ in range(10):
                params_new = params + alpha * delta
                # 边界保护
                params_new[1] = max(params_new[1], 500.0)
                params_new[2] = max(params_new[2], 0.0)
                new_res = np.linalg.norm(self._predict_rate(C, T, params_new) - r)
                if new_res < current_res:
                    break
                alpha *= 0.5
            else:
                break  # 线搜索失败

            if np.linalg.norm(alpha * delta) < 1.0e-8:
                break
            params = params_new

        lnA_est, Ea_est, n_est = params
        A_est = np.exp(lnA_est)
        A_est = max(A_est, 1.0e-6)
        Ea_est = max(Ea_est, 500.0)
        n_est = max(n_est, 0.0)

        r_pred = self._predict_rate(C, T, params)
        residual_norm = np.linalg.norm(r - r_pred)

        return A_est, Ea_est, n_est, residual_norm

    def compute_confidence_intervals(
        self,
        A: np.ndarray,
        b: np.ndarray,
        x_est: np.ndarray,
    ) -> np.ndarray:
        """
        基于 QR 分解的 R 矩阵计算参数估计的置信区间宽度。
        返回对角线元素的标准差估计 (covariance 的对角线)。
        """
        Q, R = self.qr_factorize(A)
        n = x_est.shape[0]
        R1 = R[:n, :n]
        # Covariance ≈ σ² (R^T R)^{-1}
        try:
            inv_R = np.linalg.inv(R1)
            cov = inv_R @ inv_R.T
            std_dev = np.sqrt(np.maximum(np.diag(cov), 0.0))
        except np.linalg.LinAlgError:
            std_dev = np.full(n, np.inf)
        return std_dev
