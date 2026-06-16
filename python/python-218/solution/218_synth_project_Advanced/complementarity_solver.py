"""
complementarity_solver.py
=========================
互补问题 (NCP / Complementarity Problem) 的高阶求解器。

数学背景
--------
非线性互补问题 NCP(F): 求 x ∈ R^n 使得
    x ≥ 0,  F(x) ≥ 0,  x^T F(x) = 0.

等价于 VI(R^n_+, F). 本模块实现:
    1. 半光滑 Newton 法 (基于 Fischer-Burmeister NCP 函数)
    2. 投影 Gauss-Seidel 方法 (PGS)
    3. 内点势约化方法 (IPM / barrier method)
    4. 自适应 Levenberg-Marquardt 正则化

关键公式
--------
Fischer-Burmeister 函数:
    φ_FB(a, b) = sqrt(a^2 + b^2) - a - b

半光滑 Newton 步:
    V_k · Δx = -Φ_FB(x, F(x))
其中 V_k ∈ ∂_B Φ_FB(x, F(x)) 为 Bouligand 广义 Jacobian.

 merit 函数:
    Ψ(x) = (1/2) ||Φ_FB(x, F(x))||^2

作者: DA synthesis project
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass, field

from variational_inequality import VariationalInequalityProblem


@dataclass
class SolverResult:
    """求解器返回结果."""
    x: np.ndarray
    converged: bool
    iterations: int
    residual_history: list = field(default_factory=list)
    merit_history: list = field(default_factory=list)
    message: str = ""


class FischerBurmeisterFunction:
    """
    Fischer-Burmeister NCP 函数及其半光滑 Newton 的广义 Jacobian.

    φ(a, b) = sqrt(a^2 + b^2) - a - b

    性质:
        φ(a, b) = 0  ⟺  a ≥ 0, b ≥ 0, ab = 0

    偏导数 (当 (a,b) ≠ (0,0)):
        ∂φ/∂a = a / sqrt(a^2 + b^2) - 1
        ∂φ/∂b = b / sqrt(a^2 + b^2) - 1
    """

    @staticmethod
    def phi(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """逐点计算 FB 函数."""
        norm_ab = np.sqrt(a**2 + b**2)
        return norm_ab - a - b

    @staticmethod
    def merit(a: np.ndarray, b: np.ndarray) -> float:
        """ merit 函数 Ψ = 0.5 ||Φ||^2."""
        phi_vals = FischerBurmeisterFunction.phi(a, b)
        return 0.5 * np.dot(phi_vals, phi_vals)

    @staticmethod
    def generalized_jacobian(
        x: np.ndarray, Fx: np.ndarray, JF: np.ndarray
    ) -> np.ndarray:
        """
        计算 FB 函数的 Bouligand 广义 Jacobian V(x).

        设 θ = sqrt(x_i^2 + F_i^2), 则:
            V_ii = (x_i/θ_i - 1) + (F_i/θ_i - 1) · JF_ii   (对角部分)
            V_ij = (F_i/θ_i - 1) · JF_ij                    (非对角部分)

        当 θ_i < ε, 使用正则化 θ_i = max(θ_i, ε).
        """
        n = len(x)
        theta = np.sqrt(x**2 + Fx**2)
        eps_reg = 1e-14
        theta_reg = np.maximum(theta, eps_reg)

        # 系数向量
        da = x / theta_reg - 1.0  # ∂φ/∂a (不含 -1 的部分)
        db = Fx / theta_reg - 1.0  # ∂φ/∂b

        # V = diag(da) + diag(db) · JF
        V = np.diag(da) + db[:, np.newaxis] * JF
        return V


class SemiSmoothNewtonSolver:
    """
    半光滑 Newton 法求解 NCP(F).

    算法 (Qi-Sun 2004):
        初始化 x_0, β ∈ (0,1), σ ∈ (0,1)
        For k = 0, 1, ...
            1. 计算 r_k = Φ_FB(x_k, F(x_k))
            2. 计算广义 Jacobian V_k
            3. 解 Newton 方程: V_k d_k = -r_k
            4. 线搜索: m_k = 最大 m 使得
                   Ψ(x_k + β^m d_k) ≤ (1 - 2σβ^m) Ψ(x_k)
            5. x_{k+1} = x_k + β^{m_k} d_k
    """

    def __init__(
        self,
        max_iter: int = 500,
        tol: float = 1e-10,
        beta_line: float = 0.5,
        sigma: float = 1e-4,
        lm_reg: float = 1e-8,
        verbose: bool = False,
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.beta_line = beta_line
        self.sigma = sigma
        self.lm_reg = lm_reg
        self.verbose = verbose

    def solve(
        self,
        vi_problem: VariationalInequalityProblem,
        x0: np.ndarray,
    ) -> SolverResult:
        """执行半光滑 Newton 法."""
        x = x0.copy()
        n = vi_problem.n
        fb = FischerBurmeisterFunction

        residual_history = []
        merit_history = []
        converged = False
        message = "未收敛"

        for k in range(self.max_iter):
            Fx = vi_problem.evaluate_F(x)
            phi = fb.phi(x, Fx)
            merit = 0.5 * np.dot(phi, phi)

            residual_history.append(float(np.linalg.norm(phi)))
            merit_history.append(float(merit))

            # 收敛判定
            if np.linalg.norm(phi) < self.tol:
                converged = True
                message = f"半光滑 Newton 收敛于 {k} 次迭代"
                break

            # 计算广义 Jacobian
            JF = vi_problem.jacobian_F(x)
            V = fb.generalized_jacobian(x, Fx, JF)

            # Levenberg-Marquardt 正则化 (保证可解性)
            V_reg = V + self.lm_reg * np.eye(n)

            # Newton 方向
            try:
                d = np.linalg.solve(V_reg, -phi)
            except np.linalg.LinAlgError:
                # 矩阵奇异, 使用梯度方向 (最速下降)
                grad_merit = V.T @ phi
                d = -grad_merit / (np.linalg.norm(grad_merit) + 1e-16)

            # Armijo 线搜索
            alpha = 1.0
            for _ in range(40):
                x_trial = x + alpha * d
                Fx_trial = vi_problem.evaluate_F(x_trial)
                merit_trial = fb.merit(x_trial, Fx_trial)
                if merit_trial <= (1.0 - 2.0 * self.sigma * alpha) * merit:
                    break
                alpha *= self.beta_line
            else:
                # 线搜索失败, 使用梯度下降
                grad_merit = V.T @ phi
                d = -grad_merit
                alpha = min(1e-3, 1.0 / (k + 1))

            x = x + alpha * d

            # 边界保护: 强制非负
            x = np.maximum(x, 0.0)

            if self.verbose and k % 50 == 0:
                print(f"  [Newton iter {k}] ||Φ|| = {np.linalg.norm(phi):.4e}, "
                      f"merit = {merit:.4e}, α = {alpha:.4e}")

        if not converged:
            message = f"达到最大迭代次数 {self.max_iter}"

        return SolverResult(
            x=x,
            converged=converged,
            iterations=len(residual_history),
            residual_history=residual_history,
            merit_history=merit_history,
            message=message,
        )


class ProjectedGaussSeidelSolver:
    """
    投影 Gauss-Seidel (PGS) 方法.

    对于仿射 VI: F(x) = Mx + q, PGS 按分量迭代:
        x_i^{k+1} = max(0, x_i^k - ω/M_ii · (Mx^k + q)_i^{(partial)})

    其中 (·)^{(partial)} 表示使用已更新的分量.
    松弛参数 ω ∈ (0, 2) 控制收敛速度.

    收敛条件: M 为 P-矩阵且 ω ∈ (0, 2·min_i(M_ii)/ρ(D^{-1}(|L|+|U|))).
    """

    def __init__(
        self,
        max_iter: int = 5000,
        tol: float = 1e-8,
        omega: float = 1.0,
        verbose: bool = False,
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.omega = omega
        self.verbose = verbose

    def solve(
        self,
        vi_problem: VariationalInequalityProblem,
        x0: np.ndarray,
    ) -> SolverResult:
        """执行 PGS 方法."""
        x = x0.copy()
        n = vi_problem.n
        M = vi_problem.M
        q = vi_problem.q

        residual_history = []
        converged = False
        message = "未收敛"

        # 对角线元素
        diag_M = np.diag(M).copy()
        if np.any(np.abs(diag_M) < 1e-14):
            message = "PGS 要求对角元素非零, 对角线退化"
            return SolverResult(x=x, converged=False, iterations=0,
                                residual_history=[float('inf')], message=message)

        for k in range(self.max_iter):
            x_old = x.copy()

            # 逐分量更新 (Gauss-Seidel 顺序)
            for i in range(n):
                # 计算第 i 个方程的残差 (使用最新分量)
                r_i = (M[i, :] @ x + q[i])
                if vi_problem.nonlinear is not None:
                    r_i += vi_problem.nonlinear(x)[i] - (M[i, :] @ x)
                # 投影更新
                x[i] = max(0.0, x[i] - self.omega * r_i / diag_M[i])

            # 计算残差
            res = vi_problem.natural_residual(x)
            res_norm = float(np.linalg.norm(res))
            residual_history.append(res_norm)

            if res_norm < self.tol:
                converged = True
                message = f"PGS 收敛于 {k} 次迭代"
                break

            if self.verbose and k % 200 == 0:
                print(f"  [PGS iter {k}] ||r|| = {res_norm:.4e}")

        if not converged:
            message = f"PGS 达到最大迭代次数 {self.max_iter}"

        return SolverResult(
            x=x, converged=converged,
            iterations=len(residual_history),
            residual_history=residual_history,
            message=message,
        )


class InteriorPointSolver:
    """
    内点势约化方法 (Interior Point Method) 求解 NCP.

    核心思想: 将互补条件 xy = 0 松弛为 xy = μe, μ → 0.

    中心路径方程:
        F(x) - μ/x = 0    (μ > 0, 逐点除法)

    Newton 步:
        [∇F(x) + μ·diag(1/x^2)] Δx = -(F(x) - μ/x)

    μ 的更新策略: σ-规则
        μ_{k+1} = σ_k · (x_k^T F(x_k)) / n,  σ_k ∈ (0, 1)

    障碍参数 μ 控制从中心路径到互补解的趋近速度.
    """

    def __init__(
        self,
        max_iter: int = 500,
        tol: float = 1e-10,
        mu_init: float = 1.0,
        sigma: float = 0.2,
        verbose: bool = False,
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.mu_init = mu_init
        self.sigma = sigma
        self.verbose = verbose

    def solve(
        self,
        vi_problem: VariationalInequalityProblem,
        x0: np.ndarray,
    ) -> SolverResult:
        """执行内点法."""
        n = vi_problem.n
        x = np.maximum(x0.copy(), 1e-6)  # 保证严格内点

        mu = self.mu_init
        residual_history = []
        converged = False
        message = "未收敛"

        for k in range(self.max_iter):
            Fx = vi_problem.evaluate_F(x)

            # 互补残差
            comp_res = x * Fx
            mu_target = self.sigma * np.dot(x, Fx) / n
            mu = max(mu_target, 1e-14)

            # 中心路径方程: F(x) - μ/x = 0
            center_eq = Fx - mu / x

            # Newton 系统: [∇F + μ·diag(1/x^2)] Δx = -center_eq
            JF = vi_problem.jacobian_F(x)
            reg = mu / (x**2 + 1e-30)
            J_sys = JF + np.diag(reg)

            try:
                dx = np.linalg.solve(J_sys, -center_eq)
            except np.linalg.LinAlgError:
                dx = -center_eq / (np.diag(J_sys) + 1e-16)

            # 步长选择: 保证 x + α·dx > 0
            alpha = 1.0
            for _ in range(30):
                x_trial = x + alpha * dx
                if np.all(x_trial > 1e-12):
                    break
                alpha *= 0.5
            else:
                alpha = 1e-4

            x = x + alpha * dx
            x = np.maximum(x, 1e-12)  # 边界保护

            # 收敛判定
            res_norm = float(np.linalg.norm(comp_res))
            residual_history.append(res_norm)

            if res_norm < self.tol:
                converged = True
                message = f"内点法收敛于 {k} 次迭代, μ_final = {mu:.4e}"
                break

            if self.verbose and k % 50 == 0:
                print(f"  [IPM iter {k}] ||x·F(x)|| = {res_norm:.4e}, μ = {mu:.4e}")

        if not converged:
            message = f"内点法达到最大迭代次数 {self.max_iter}, μ = {mu:.4e}"

        return SolverResult(
            x=x, converged=converged,
            iterations=len(residual_history),
            residual_history=residual_history,
            message=message,
        )
