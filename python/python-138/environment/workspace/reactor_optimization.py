"""
微反应器操作条件优化 (基于测试优化问题与黄金分割搜索)
====================================================
对微反应器的操作参数（停留时间、壁温、入口浓度）进行优化，
以最大化目标函数（如转化率、选择性、能量效率）。

优化框架：
    1. 单参数优化：黄金分割搜索 (Golden Section Search)
       对单峰目标函数在区间 [a,b] 上寻找最优值：

           τ = (√5 - 1)/2 ≈ 0.618

           x1 = a + (1-τ)(b-a),  x2 = a + τ(b-a)
           比较 f(x1), f(x2)，缩小区间，直至 |b-a| < ε

    2. 多参数优化：采用测试函数库中的梯度/Hessian 信息，
       实现 Newton 型局部优化。

目标函数示例（能量效率最大化）：
    f(τ_res, T_w, C_0) = X · SE / (Q_cool + Q_heat)

    其中 X 为转化率，SE 为选择性，分母为外部能量输入。
"""

import numpy as np
from typing import Tuple, Callable, Optional, List


class ReactorOptimizer:
    """
    微反应器操作条件优化器。
    """

    def __init__(self):
        self.tau = (np.sqrt(5.0) - 1.0) / 2.0  # 黄金比例 ≈ 0.618

    def golden_section_search(
        self,
        f: Callable[[float], float],
        a: float,
        b: float,
        max_iter: int = 100,
        x_tol: float = 1.0e-8,
    ) -> Tuple[float, float, int, int]:
        """
        黄金分割搜索求解单峰函数最小值。

        输入:
            f: 目标函数
            a, b: 搜索区间
            max_iter: 最大迭代次数
            x_tol: 区间宽度容差

        返回:
            x_opt: 最优解
            f_opt: 最优值
            it:    迭代次数
            nf:    函数求值次数
        """
        if a >= b:
            raise ValueError("搜索区间必须满足 a < b")
        if max_iter < 1:
            raise ValueError("max_iter 至少为 1")

        x1 = a + (1.0 - self.tau) * (b - a)
        x2 = a + self.tau * (b - a)
        f1 = f(x1)
        f2 = f(x2)
        nf = 2

        for it in range(max_iter):
            if f1 < f2:
                b = x2
                x2 = x1
                f2 = f1
                x1 = a + (1.0 - self.tau) * (b - a)
                f1 = f(x1)
                nf += 1
            else:
                a = x1
                x1 = x2
                f1 = f2
                x2 = a + self.tau * (b - a)
                f2 = f(x2)
                nf += 1

            if abs(b - a) <= x_tol:
                break

        x_opt = 0.5 * (a + b)
        f_opt = f(x_opt)
        nf += 1
        return x_opt, f_opt, it + 1, nf

    def optimize_residence_time(
        self,
        objective: Callable[[float], float],
        tau_min: float = 0.1,
        tau_max: float = 100.0,
    ) -> Tuple[float, float]:
        """
        优化停留时间以最大化（或最小化）目标函数。
        """
        if tau_min <= 0.0:
            raise ValueError("停留时间下限必须为正")
        # 黄金分割默认求最小值，因此若目标为最大化则取负
        x_opt, f_opt, _, _ = self.golden_section_search(
            objective, tau_min, tau_max, max_iter=80, x_tol=1.0e-6
        )
        return x_opt, f_opt

    def gradient_descent_with_line_search(
        self,
        f: Callable[[np.ndarray], float],
        grad_f: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        max_iter: int = 200,
        tol: float = 1.0e-8,
        step_init: float = 1.0,
    ) -> Tuple[np.ndarray, float, int]:
        """
        带回溯线搜索的梯度下降法（多变量优化）。

        线搜索条件（Armijo）：
            f(x + α p) ≤ f(x) + c1 α ∇f(x)^T p

        其中 p = -∇f(x) 为下降方向，c1 = 1e-4。
        """
        x = x0.copy().astype(float)
        c1 = 1.0e-4
        for it in range(max_iter):
            g = grad_f(x)
            norm_g = np.linalg.norm(g)
            if norm_g < tol:
                return x, f(x), it

            p = -g / max(norm_g, 1.0e-12)
            alpha = step_init
            fx = f(x)
            # 回溯
            for _ in range(20):
                x_new = x + alpha * p
                if f(x_new) <= fx + c1 * alpha * np.dot(g, p):
                    break
                alpha *= 0.5
            x = x + alpha * p
        return x, f(x), max_iter

    def newton_method_with_hessian(
        self,
        f: Callable[[np.ndarray], float],
        grad_f: Callable[[np.ndarray], np.ndarray],
        hess_f: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        max_iter: int = 100,
        tol: float = 1.0e-10,
    ) -> Tuple[np.ndarray, float, int]:
        """
        使用精确 Hessian 的阻尼 Newton 法。

        搜索方向由 H p = -∇f 给出，通过 Cholesky 或 LU 求解。
        若 Hessian 不正定，则回退到梯度下降。
        """
        x = x0.copy().astype(float)
        for it in range(max_iter):
            g = grad_f(x)
            if np.linalg.norm(g) < tol:
                return x, f(x), it

            H = hess_f(x)
            # 尝试 Cholesky（H 正定）
            try:
                L = np.linalg.cholesky(H)
                # 解 L y = -g
                y = np.linalg.solve(L, -g)
                # 解 L^T p = y
                p = np.linalg.solve(L.T, y)
            except np.linalg.LinAlgError:
                # Hessian 不正定，回退梯度下降
                p = -g / max(np.linalg.norm(g), 1.0e-12)

            # 线搜索
            alpha = 1.0
            fx = f(x)
            c1 = 1.0e-4
            for _ in range(20):
                x_new = x + alpha * p
                if f(x_new) <= fx + c1 * alpha * np.dot(g, p):
                    break
                alpha *= 0.5
            x = x + alpha * p
        return x, f(x), max_iter

    def reactor_objective_rosenbrock_like(
        self, params: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        模拟微反应器多参数优化的 Rosenbrock 型测试函数：
            f(x) = Σ_{i=1}^{n-1} [ 100 (x_{i+1} - x_i²)² + (1 - x_i)² ]

        导数：
            ∂f/∂x_i = -400 x_i (x_{i+1} - x_i²) - 2(1 - x_i)
                      + 200 (x_i - x_{i-1}²)    (i>0)
        """
        x = params.flatten().astype(float)
        n = len(x)
        if n < 2:
            return x[0] ** 2, np.array([2.0 * x[0]]), np.array([[2.0]])

        f_val = 0.0
        for i in range(n - 1):
            f_val += 100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1.0 - x[i]) ** 2

        grad = np.zeros(n)
        grad[0] = -400.0 * x[0] * (x[1] - x[0] ** 2) - 2.0 * (1.0 - x[0])
        for i in range(1, n - 1):
            grad[i] = (
                -400.0 * x[i] * (x[i + 1] - x[i] ** 2)
                - 2.0 * (1.0 - x[i])
                + 200.0 * (x[i] - x[i - 1] ** 2)
            )
        grad[n - 1] = 200.0 * (x[n - 1] - x[n - 2] ** 2)

        H = np.zeros((n, n))
        H[0, 0] = -400.0 * (x[1] - 3.0 * x[0] ** 2) + 2.0
        H[0, 1] = -400.0 * x[0]
        for i in range(1, n - 1):
            H[i, i - 1] = -400.0 * x[i - 1]
            H[i, i] = 202.0 + 1200.0 * x[i] ** 2 - 400.0 * x[i + 1]
            H[i, i + 1] = -400.0 * x[i]
        H[n - 1, n - 2] = -400.0 * x[n - 2]
        H[n - 1, n - 1] = 200.0

        return f_val, grad, H

    def optimize_reactor_conditions(
        self, n_params: int = 4
    ) -> Tuple[np.ndarray, float]:
        """
        对 n_params 个操作参数进行优化演示。
        返回最优参数与目标值。
        """
        x0 = np.random.uniform(-1.0, 1.0, n_params)

        def f(x):
            fv, _, _ = self.reactor_objective_rosenbrock_like(x)
            return fv

        def g(x):
            _, gr, _ = self.reactor_objective_rosenbrock_like(x)
            return gr

        def h(x):
            _, _, he = self.reactor_objective_rosenbrock_like(x)
            return he

        x_opt, f_opt, it = self.newton_method_with_hessian(f, g, h, x0)
        return x_opt, f_opt
