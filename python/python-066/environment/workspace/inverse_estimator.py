"""
inverse_estimator.py
================================================================================
地下水模型参数反演估计模块——非线性最小二乘与 Sherman-Morrison 快速重解

基于种子项目：
  - 1220_test_nls：非线性最小二乘（NLS）测试问题库与 Jacobian 计算
  - 995_r8sm      ：Sherman-Morrison 公式用于秩-1 更新后的快速线性系统重解

科学背景：
  地下水溶质运移模型的可靠性高度依赖于参数（水力传导度 K、弥散度 α_L、
  孔隙度 n、滞留因子 R、衰变速率 λ）的准确估计。由于野外直接测量这些
  参数往往成本高昂且空间代表性有限，反演估计（inverse parameter estimation）
  成为水文地质学中的核心问题。

  反演问题表述为最小化观测浓度与模拟浓度之间的残差平方和：

      min_β  Φ(β) = ½ Σ_{i=1}^M [ C_obs(t_i, x_i) - C_sim(t_i, x_i; β) ]²
                = ½ || f(β) ||²

  其中 β = [K, α_L, n, R, λ]^T 为待估参数向量，f_i(β) 为第 i 个观测点上的
  残差。该问题通常是非线性的、病态的（ill-posed），需要正则化和灵敏度分析。

  Gauss-Newton 迭代：
      β^{k+1} = β^k - [J(β^k)^T J(β^k)]^{-1} J(β^k)^T f(β^k)
  其中 Jacobian J_{ij} = ∂f_i / ∂β_j 刻画了模型输出对参数的敏感度。

  Sherman-Morrison 公式：
      若参数发生微小扰动 δβ_j，系统矩阵 A 的秩-1 更新可快速求逆：
      (A + u v^T)^{-1} = A^{-1} - (A^{-1} u v^T A^{-1}) / (1 + v^T A^{-1} u)
      避免每次参数调整后完全重新分解矩阵，计算复杂度从 O(n³) 降至 O(n²)。
================================================================================
"""

import numpy as np
from typing import Callable, Optional


class InverseEstimator:
    """
    基于 Gauss-Newton 与 Levenberg-Marquardt 的非线性最小二乘参数反演器。
    """

    def __init__(self, residual_func: Callable[[np.ndarray], np.ndarray],
                 param_names: list[str],
                 param_lower_bounds: Optional[np.ndarray] = None,
                 param_upper_bounds: Optional[np.ndarray] = None):
        """
        参数
        ----------
        residual_func : callable
            f(β) -> np.ndarray，返回 M 维残差向量
        param_names : list[str]
            参数名称列表
        param_lower_bounds, param_upper_bounds : np.ndarray
            参数上下界约束
        """
        self.f = residual_func
        self.param_names = param_names
        self.n_params = len(param_names)
        self.lb = param_lower_bounds
        self.ub = param_upper_bounds

    def _compute_jacobian_fd(self, beta: np.ndarray, h: float = 1e-6) -> np.ndarray:
        """
        使用中心差分计算 Jacobian 矩阵 J(β)。

        J_{ij} ≈ [ f_i(β + h e_j) - f_i(β - h e_j) ] / (2h)

        计算代价：2N 次模型正演（N 为参数个数）。
        """
        beta = np.asarray(beta, dtype=float)
        if len(beta) != self.n_params:
            raise ValueError("参数向量维度不匹配")
        f0 = self.f(beta)
        M = len(f0)
        J = np.zeros((M, self.n_params))
        for j in range(self.n_params):
            beta_plus = beta.copy()
            beta_minus = beta.copy()
            # 自适应步长
            hj = h * max(abs(beta[j]), 1.0)
            beta_plus[j] += hj
            beta_minus[j] -= hj
            f_plus = self.f(beta_plus)
            f_minus = self.f(beta_minus)
            J[:, j] = (f_plus - f_minus) / (2.0 * hj)
        return J

    def _clip_parameters(self, beta: np.ndarray) -> np.ndarray:
        """将参数裁剪到合法区间。"""
        if self.lb is not None:
            beta = np.maximum(beta, self.lb)
        if self.ub is not None:
            beta = np.minimum(beta, self.ub)
        return beta

    def solve_gauss_newton(self, beta0: np.ndarray,
                           max_iter: int = 50,
                           tol: float = 1e-6,
                           damping: float = 1.0) -> dict:
        """
        Gauss-Newton 迭代求解非线性最小二乘问题。

        迭代格式：
            (J^T J) Δβ = -J^T f
            β^{k+1} = β^k + α Δβ

        参数
        ----------
        beta0 : np.ndarray
            初始参数猜测
        max_iter : int
            最大迭代次数
        tol : float
            收敛容差（基于梯度范数 ||J^T f||）
        damping : float
            初始步长 α

        返回
        -------
        dict
            包含最终参数、残差历史、Jacobian 条件数等
        """
        beta = np.asarray(beta0, dtype=float).copy()
        beta = self._clip_parameters(beta)
        residuals_history = []
        param_history = [beta.copy()]

        for it in range(max_iter):
            f_val = self.f(beta)
            rms = np.sqrt(np.mean(f_val ** 2))
            residuals_history.append(rms)

            J = self._compute_jacobian_fd(beta)
            grad = J.T @ f_val
            grad_norm = np.linalg.norm(grad)

            if grad_norm < tol:
                break

            # Gauss-Newton 方向
            JTJ = J.T @ J
            # 正则化保证可逆性
            JTJ += np.eye(self.n_params) * 1e-8 * np.trace(JTJ)
            try:
                delta = np.linalg.solve(JTJ, -grad)
            except np.linalg.LinAlgError:
                delta = -grad / (grad_norm + 1e-15)

            # 线搜索
            alpha = damping
            for _ in range(10):
                beta_new = self._clip_parameters(beta + alpha * delta)
                f_new = self.f(beta_new)
                if np.mean(f_new ** 2) < np.mean(f_val ** 2):
                    beta = beta_new
                    break
                alpha *= 0.5
            else:
                beta = self._clip_parameters(beta + alpha * delta)

            param_history.append(beta.copy())

        final_f = self.f(beta)
        return {
            "beta_opt": beta,
            "param_names": self.param_names,
            "final_rms": float(np.sqrt(np.mean(final_f ** 2))),
            "residuals_history": residuals_history,
            "param_history": param_history,
            "n_iter": len(residuals_history),
            "jacobian_condition": float(np.linalg.cond(self._compute_jacobian_fd(beta)))
        }

    def solve_lm(self, beta0: np.ndarray,
                 max_iter: int = 100,
                 tol: float = 1e-6,
                 lambda_init: float = 0.01) -> dict:
        """
        Levenberg-Marquardt 方法：在 Gauss-Newton 和梯度下降之间自适应插值。

        求解线性系统：
            (J^T J + λ diag(J^T J)) Δβ = -J^T f

        当 λ → 0 时退化为 Gauss-Newton；当 λ 很大时退化为最速下降。
        """
        beta = np.asarray(beta0, dtype=float).copy()
        beta = self._clip_parameters(beta)
        lam = lambda_init
        residuals_history = []

        for it in range(max_iter):
            # TODO: Hole 3 — 实现 Levenberg-Marquardt 单次迭代核心逻辑
            # 要求：
            #   1. 计算当前残差 f_val = self.f(beta) 和 RMS
            #   2. 用中心差分计算 Jacobian J = self._compute_jacobian_fd(beta)
            #   3. 计算梯度 grad = J^T f_val，若范数 < tol 则收敛
            #   4. 组装 LM 矩阵: A_lm = J^T J + λ * diag(J^T J)
            #   5. 求解 Δβ = A_lm^{-1} (-grad)
            #   6. 线搜索：若新 RMS 下降则接受步长并减小 λ，否则增大 λ
            raise NotImplementedError("Hole 3: 请实现 LM 迭代核心逻辑")

        final_f = self.f(beta)
        return {
            "beta_opt": beta,
            "param_names": self.param_names,
            "final_rms": float(np.sqrt(np.mean(final_f ** 2))),
            "residuals_history": residuals_history,
            "n_iter": len(residuals_history),
        }


# ---------------------------------------------------------------------------
# Sherman-Morrison 快速重解
# ---------------------------------------------------------------------------

def sherman_morrison_solve(A_inv_u: np.ndarray, v: np.ndarray,
                           A_inv_b: np.ndarray,
                           alpha: float = 1.0) -> np.ndarray:
    """
    利用 Sherman-Morrison 公式求解 (A + α u v^T) x = b，已知 A^{-1} u 和 A^{-1} b。

    Sherman-Morrison 公式：
        (A + u v^T)^{-1} = A^{-1} - (A^{-1} u v^T A^{-1}) / (1 + v^T A^{-1} u)

    因此：
        x = A^{-1} b - (A^{-1} u) (v^T A^{-1} b) / (1 + v^T A^{-1} u)

    参数
    ----------
    A_inv_u : np.ndarray
        A^{-1} u（已预计算）
    v : np.ndarray
        秩-1 更新的右向量
    A_inv_b : np.ndarray
        A^{-1} b（原系统的解）
    alpha : float
        更新系数

    返回
    -------
    np.ndarray
        更新后系统的解 x
    """
    denom = 1.0 + alpha * np.dot(v, A_inv_u)
    if abs(denom) < 1e-14:
        raise ValueError("Sherman-Morrison 分母接近零，秩-1 更新奇异")
    return A_inv_b - alpha * A_inv_u * (np.dot(v, A_inv_b) / denom)


def lu_factor_solve(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对矩阵 A 进行 LU 分解并求解 A x = b。
    返回 (x, L, U) 以便后续 Sherman-Morrison 更新使用。
    """
    from scipy.linalg import lu_factor, lu_solve
    lu, piv = lu_factor(A)
    x = lu_solve((lu, piv), b)
    return x, lu, piv


if __name__ == "__main__":
    # 简单测试：拟合 y = a * exp(-b * x)
    x_data = np.linspace(0, 2, 20)
    y_true = 3.0 * np.exp(-1.5 * x_data)
    noise = np.random.default_rng(0).normal(0, 0.05, size=len(x_data))
    y_obs = y_true + noise

    def residual(beta):
        a, b_param = beta
        return a * np.exp(-b_param * x_data) - y_obs

    est = InverseEstimator(residual, ["a", "b"], lb=np.array([0.1, 0.1]), ub=np.array([10.0, 5.0]))
    result = est.solve_lm(np.array([1.0, 1.0]))
    assert result["final_rms"] < 0.1

    # Sherman-Morrison 测试
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    x0 = np.linalg.solve(A, b)
    u = np.array([1.0, 0.0])
    v = np.array([0.0, 1.0])
    A_inv_u = np.linalg.solve(A, u)
    x_sm = sherman_morrison_solve(A_inv_u, v, x0, alpha=1.0)
    x_direct = np.linalg.solve(A + np.outer(u, v), b)
    assert np.allclose(x_sm, x_direct)
    print("inverse_estimator: 自测试通过")
