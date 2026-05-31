"""
inversion_solver.py
非线性大气参数反演求解模块。

融合原始项目：120_broyden（Broyden 拟牛顿法求解非线性方程组）
             097_bisection_rc（二分法求根，反向通信）

在系外行星大气光谱反演中，需要求解：
    min_θ χ²(θ) = min_θ Σ_i [ (F_obs(λ_i) - F_model(λ_i; θ))² / σ_i² ]

这是一个高度非线性的最小二乘优化问题，需结合多种数值方法求解。
"""

import numpy as np
from typing import Callable, Tuple, Optional, List


class BisectionRootFinder:
    """
    反向通信式二分法求根。

    融合 bisection_rc 的核心思想：
    在区间 [a, b] 内寻找 f(x) = 0 的根，要求 f(a) 和 f(b) 异号。

    收敛速度: 线性收敛，每次迭代区间减半。
        |x_n - x*| ≤ (b - a) / 2^n
    """

    def __init__(self, a: float, b: float, tol: float = 1e-10, max_iter: int = 100):
        if a >= b:
            raise ValueError(f"区间左端点必须小于右端点: a={a}, b={b}")
        self.a = a
        self.b = b
        self.tol = tol
        self.max_iter = max_iter
        self.iteration = 0
        self.state = 0  # 0: init, 1: need f(a), 2: need f(b), 3: need f(mid)
        self.fa = None
        self.fb = None
        self.x_mid = None

    def step(self, fx: Optional[float] = None) -> Tuple[float, bool]:
        """
        执行一步二分迭代。

        参数:
            fx: 当前点的函数值（首次调用应传入 None）

        返回:
            (x_next, converged): 下一个需要计算的点，以及是否收敛
        """
        if self.state == 0:
            self.x_mid = self.a
            self.state = 1
            return self.x_mid, False

        if self.state == 1:
            self.fa = fx
            self.x_mid = self.b
            self.state = 2
            return self.x_mid, False

        if self.state == 2:
            self.fb = fx
            if self.fa * self.fb > 0:
                raise ValueError(f"f(a)={self.fa} 和 f(b)={self.fb} 同号，无法二分")
            self.x_mid = 0.5 * (self.a + self.b)
            self.state = 3
            self.iteration = 1
            return self.x_mid, False

        # state == 3
        if self.iteration >= self.max_iter:
            return self.x_mid, True

        if fx * self.fa > 0:
            self.a = self.x_mid
            self.fa = fx
        else:
            self.b = self.x_mid
            self.fb = fx

        if abs(self.b - self.a) < self.tol:
            return 0.5 * (self.a + self.b), True

        self.x_mid = 0.5 * (self.a + self.b)
        self.iteration += 1
        return self.x_mid, False

    def solve(self, func: Callable[[float], float]) -> float:
        """便捷方法：直接求解。"""
        x, _ = self.step(None)
        while True:
            fx = func(x)
            x, converged = self.step(fx)
            if converged:
                return x


class BroydenSolver:
    """
    Broyden 拟牛顿法求解非线性方程组 F(x) = 0。

    融合 broyden 的核心算法：
    Broyden 方法通过低秩更新近似 Jacobian 矩阵的逆，
    避免了每次迭代重新计算 Jacobian，计算复杂度从 O(n³) 降至 O(n²)。

    更新公式（Broyden's good method）:
        J_{k+1} = J_k + (ΔF - J_k Δx) Δx^T / ||Δx||²
        x_{k+1} = x_k - J_{k+1}^{-1} F(x_k)

    或 Sherman-Morrison 公式直接更新逆：
        J_{k+1}^{-1} = J_k^{-1} + (Δx - J_k^{-1} ΔF) Δx^T J_k^{-1} / (Δx^T J_k^{-1} ΔF)
    """

    def __init__(self, atol: float = 1e-8, rtol: float = 1e-6,
                 maxit: int = 100, maxdim: int = 50):
        self.atol = atol
        self.rtol = rtol
        self.maxit = maxit
        self.maxdim = maxdim

    def solve(self, func: Callable[[np.ndarray], np.ndarray],
              x0: np.ndarray) -> Tuple[np.ndarray, int, float]:
        """
        求解 F(x) = 0。

        参数:
            func: 向量值函数 F(x)
            x0: 初始猜测

        返回:
            (x, ierr, residual_norm)
            ierr: 0 成功，1 失败
        """
        x = np.asarray(x0, dtype=np.float64).reshape(-1)
        n = x.shape[0]

        fc = func(x)
        fnrm = np.linalg.norm(fc) / max(np.sqrt(n), 1e-15)
        stop_tol = self.atol + self.rtol * fnrm

        # 有限差分初始化 Jacobian 逆
        J_inv = self._finite_difference_jacobian_inverse(func, x, fc)

        stp = np.zeros((n, self.maxdim), dtype=np.float64)
        stp[:, 0] = -J_inv @ fc
        stp_nrm = np.zeros(self.maxdim, dtype=np.float64)
        stp_nrm[0] = np.dot(stp[:, 0], stp[:, 0])

        nbroy = 0
        itc = 0

        while itc < self.maxit:
            nbroy += 1
            fnrmo = fnrm
            itc += 1

            x = x + stp[:, nbroy - 1]
            fc = func(x)
            fnrm = np.linalg.norm(fc) / max(np.sqrt(n), 1e-15)

            if fnrm <= stop_tol:
                return x, 0, fnrm

            if fnrmo <= fnrm:
                return x, 1, fnrm

            if nbroy < self.maxdim:
                z = -fc.copy()
                if nbroy > 1:
                    for kbr in range(nbroy - 1):
                        z = z + stp[:, kbr + 1] * np.dot(stp[:, kbr], z) / max(stp_nrm[kbr], 1e-30)

                zz = np.dot(stp[:, nbroy - 1], z)
                zz = zz / max(stp_nrm[nbroy - 1], 1e-30)
                stp[:, nbroy] = z / max(1.0 - zz, 1e-15)
                stp_nrm[nbroy] = np.dot(stp[:, nbroy], stp[:, nbroy])
            else:
                # 重启
                J_inv = self._finite_difference_jacobian_inverse(func, x, fc)
                stp[:, 0] = -J_inv @ fc
                stp_nrm[0] = np.dot(stp[:, 0], stp[:, 0])
                nbroy = 0

        return x, 1 if fnrm > stop_tol else 0, fnrm

    def _finite_difference_jacobian_inverse(self, func, x, fx, eps=1e-7):
        """有限差分近似 Jacobian 并求逆。"""
        n = x.shape[0]
        J = np.zeros((n, n), dtype=np.float64)
        for j in range(n):
            xj = x.copy()
            h = eps * max(abs(xj[j]), 1.0)
            xj[j] += h
            fj = func(xj)
            J[:, j] = (fj - fx) / h

        try:
            J_inv = np.linalg.inv(J)
        except np.linalg.LinAlgError:
            J_inv = np.linalg.pinv(J)
        return J_inv


class LevenbergMarquardt:
    """
    Levenberg-Marquardt 非线性最小二乘优化。

    求解:
        min_x Σ_i r_i(x)²

    其中 r_i(x) = [F_obs(λ_i) - F_model(λ_i; x)] / σ_i

    迭代公式:
        (J^T J + λ diag(J^T J)) Δx = J^T r
        x_{new} = x + Δx

    λ 是阻尼参数，在 Gauss-Newton (λ→0) 和梯度下降 (λ→∞) 间自适应调节。
    """

    def __init__(self, max_iter: int = 100, tol: float = 1e-8,
                 lambda_init: float = 1e-3, lambda_up: float = 10.0,
                 lambda_down: float = 0.1):
        self.max_iter = max_iter
        self.tol = tol
        self.lambda_val = lambda_init
        self.lambda_up = lambda_up
        self.lambda_down = lambda_down

    def solve(self, residual_func: Callable[[np.ndarray], np.ndarray],
              jacobian_func: Callable[[np.ndarray], np.ndarray],
              x0: np.ndarray) -> Tuple[np.ndarray, int, float]:
        """
        参数:
            residual_func: 返回残差向量 r(x)
            jacobian_func: 返回 Jacobian 矩阵 J(x) = ∂r/∂x
            x0: 初始参数

        返回:
            (x, iters, final_cost)
        """
        x = np.asarray(x0, dtype=np.float64).reshape(-1)

        for it in range(self.max_iter):
            r = residual_func(x)
            cost = 0.5 * np.dot(r, r)

            J = jacobian_func(x)
            JtJ = J.T @ J
            Jtr = J.T @ r

            # 阻尼矩阵
            damping = self.lambda_val * np.diag(np.diag(JtJ))

            try:
                delta = np.linalg.solve(JtJ + damping, -Jtr)
            except np.linalg.LinAlgError:
                delta = -Jtr / (np.diag(JtJ) + self.lambda_val + 1e-15)

            x_new = x + delta
            r_new = residual_func(x_new)
            cost_new = 0.5 * np.dot(r_new, r_new)

            if cost_new < cost:
                x = x_new
                self.lambda_val *= self.lambda_down
                if np.linalg.norm(delta) < self.tol:
                    return x, it + 1, cost_new
            else:
                self.lambda_val *= self.lambda_up
                if self.lambda_val > 1e15:
                    return x, it + 1, cost

        r = residual_func(x)
        return x, self.max_iter, 0.5 * np.dot(r, r)


class TikhonovRegularization:
    """
    Tikhonov 正则化用于病态反演问题。

    在光谱反演中，由于大气参数与观测光谱的关系通常是非唯一（ill-posed）的，
    需要正则化约束。

    优化目标:
        min_θ [ ||F_obs - F_model(θ)||² + α ||L θ||² ]

    其中 L 是正则化算子（如差分矩阵，惩罚参数剖面的剧烈变化）。
    """

    @staticmethod
    def first_order_difference_matrix(n: int) -> np.ndarray:
        """构造一阶差分正则化矩阵。"""
        L = np.zeros((n - 1, n), dtype=np.float64)
        for i in range(n - 1):
            L[i, i] = -1.0
            L[i, i + 1] = 1.0
        return L

    @staticmethod
    def second_order_difference_matrix(n: int) -> np.ndarray:
        """构造二阶差分正则化矩阵。"""
        L = np.zeros((n - 2, n), dtype=np.float64)
        for i in range(n - 2):
            L[i, i] = 1.0
            L[i, i + 1] = -2.0
            L[i, i + 2] = 1.0
        return L

    @staticmethod
    def solve_linear_tikhonov(A: np.ndarray, b: np.ndarray,
                               L: np.ndarray, alpha: float) -> np.ndarray:
        """
        求解线性 Tikhonov 正则化问题。

        正规方程:
            (A^T A + α L^T L) x = A^T b

        参数:
            A: 设计矩阵 (m, n)
            b: 观测向量 (m,)
            L: 正则化矩阵 (k, n)
            alpha: 正则化参数

        返回:
            x: 解向量 (n,)
        """
        AtA = A.T @ A
        LtL = L.T @ L
        rhs = A.T @ b
        M = AtA + alpha * LtL
        try:
            x = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            x = np.linalg.lstsq(M, rhs, rcond=None)[0]
        return x
