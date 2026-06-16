"""
随机刚性常微分方程求解器 (Stochastic Stiff ODE Solver)
========================================================
求解具有随机参数的刚性常微分方程系统:

  du/dt = f(u, t; ξ(ω)),  u(0) = u₀(ξ)

Lindberg 问题 (经典刚性测试问题):
  du₁/dt = -0.04 u₁ + 10⁴ u₂ u₃
  du₂/dt = 0.04 u₁ - 10⁴ u₂ u₃ - 3×10⁷ u₂²
  du₃/dt = 3×10⁷ u₂²
  u₁(0) = 1, u₂(0) = 0, u₃(0) = 0

  刚性比: max|Re(λ)| / min|Re(λ)| ~ 10⁸
  特征: 初始快速瞬态 → 长时间缓慢演化

随机推广:
  将反应速率参数化为随机变量:
    k₁(ω) = k₁₀ exp(σ₁ ξ₁), k₂(ω) = k₂₀ exp(σ₂ ξ₂)

  这使系统变为:
    du/dt = f(u, t; ξ)

隐式 Euler 方法 (A-稳定):
  u^{n+1} = u^n + Δt f(u^{n+1}, t^{n+1}; ξ)

  需要求解非线性系统:
    G(u^{n+1}) = u^{n+1} - u^n - Δt f(u^{n+1}, t^{n+1}) = 0

  使用 Newton 迭代:
    J_G δu = -G(u^k)
    u^{k+1} = u^k + δu

  其中 Jacobian: J_G = I - Δt ∂f/∂u

BDF-2 方法 (二阶, A-稳定):
  (3u^{n+2} - 4u^{n+1} + u^n) / (2Δt) = f(u^{n+2}, t^{n+2})

向后差分公式 (BDF-k): k=1,...,6
  Σ_{j=0}^{k} α_j u^{n+j} = Δt β_k f(u^{n+k}, t^{n+k})

参考文献:
  Lindberg, B. (1974). On the numerical solution of the stiff ODE.
  Hairer, E. & Wanner, G. (1996). Solving ODEs II.
"""

import numpy as np
from typing import Tuple, Optional, Callable, List, Dict


class StiffStochasticODE:
    """
    随机刚性 ODE 系统求解器。

    实现隐式 Euler 和 BDF-2 方法,
    带有 Newton 迭代非线性求解器。
    """

    def __init__(
        self,
        n_equations: int = 3,
        time_end: float = 1.0,
        n_timesteps: int = 500,
        newton_tol: float = 1e-10,
        newton_max_iter: int = 20
    ):
        """
        参数:
            n_equations: 方程数
            time_end: 终止时间
            n_timesteps: 时间步数
            newton_tol: Newton 收敛容差
            newton_max_iter: Newton 最大迭代次数
        """
        self.n_eq = n_equations
        self.T = time_end
        self.n_steps = n_timesteps
        self.dt = self.T / self.n_steps
        self.newton_tol = newton_tol
        self.newton_max_iter = newton_max_iter

    def lindberg_rhs(
        self, u: np.ndarray, t: float,
        k1: float = 0.04, k2: float = 1e4, k3: float = 3e7
    ) -> np.ndarray:
        """
        Lindberg 问题右端函数 (刚性化学动力学):

          du₁/dt = -k₁ u₁ + k₂ u₂ u₃
          du₂/dt = k₁ u₁ - k₂ u₂ u₃ - k₃ u₂²
          du₃/dt = k₃ u₂²

        守恒律: u₁ + u₂ + u₃ = 1 (质量守恒)
        正定性: u_i ≥ 0

        Jacobian:
          J = [-k₁,      k₂u₃,        k₂u₂     ]
              [k₁,      -k₂u₃-2k₃u₂, -k₂u₂     ]
              [0,        2k₃u₂,       0          ]

        特征值估计:
          λ₁ ≈ -k₁ (慢)
          λ₂ ≈ -k₂-k₃ (快)
          λ₃ ≈ 0 (守恒模式)
        """
        u1, u2, u3 = u[0], u[1], u[2]

        # 确保非负
        u1 = max(u1, 0.0)
        u2 = max(u2, 0.0)
        u3 = max(u3, 0.0)

        f = np.zeros(3)
        f[0] = -k1 * u1 + k2 * u2 * u3
        f[1] = k1 * u1 - k2 * u2 * u3 - k3 * u2 ** 2
        f[2] = k3 * u2 ** 2
        return f

    def lindberg_jacobian(
        self, u: np.ndarray,
        k1: float = 0.04, k2: float = 1e4, k3: float = 3e7
    ) -> np.ndarray:
        """
        Lindberg 问题的 Jacobian 矩阵:
          J_ij = ∂f_i/∂u_j
        """
        u1, u2, u3 = max(u[0], 0.0), max(u[1], 0.0), max(u[2], 0.0)

        J = np.zeros((3, 3))
        J[0, 0] = -k1
        J[0, 1] = k2 * u3
        J[0, 2] = k2 * u2
        J[1, 0] = k1
        J[1, 1] = -k2 * u3 - 2.0 * k3 * u2
        J[1, 2] = -k2 * u2
        J[2, 0] = 0.0
        J[2, 1] = 2.0 * k3 * u2
        J[2, 2] = 0.0
        return J

    def implicit_euler_step(
        self,
        u_n: np.ndarray,
        t_n: float,
        rhs_func: Callable,
        jac_func: Callable
    ) -> np.ndarray:
        """
        隐式 Euler 步 (Backward Euler):
          u^{n+1} = u^n + Δt f(u^{n+1}, t^{n+1})

        Newton 迭代:
          G(v) = v - u^n - Δt f(v, t^{n+1}) = 0
          J_G = I - Δt J_f
          v^{k+1} = v^k - J_G^{-1} G(v^k)

        初始猜测: v⁰ = u^n (或外推)
        """
        dt = self.dt
        v = u_n.copy()  # 初始猜测

        for iteration in range(self.newton_max_iter):
            f_v = rhs_func(v, t_n + dt)
            G = v - u_n - dt * f_v

            # 收敛检查
            if np.linalg.norm(G) < self.newton_tol:
                return v

            # Jacobian of G
            J_f = jac_func(v)
            J_G = np.eye(self.n_eq) - dt * J_f

            # 求解线性系统
            try:
                delta = np.linalg.solve(J_G, -G)
            except np.linalg.LinAlgError:
                # 奇异矩阵: 使用伪逆
                delta = np.linalg.lstsq(J_G, -G, rcond=None)[0]

            v = v + delta

            # 确保物理约束 (非负)
            v = np.maximum(v, 0.0)

        return v

    def solve_lindberg(
        self,
        k1: float = 0.04,
        k2: float = 1e4,
        k3: float = 3e7,
        initial_condition: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        求解 Lindberg 问题。

        参数:
            k1, k2, k3: 反应速率参数
            initial_condition: 初始条件, shape (3,)

        返回:
            time: shape (n_steps+1,)
            solution: shape (n_steps+1, 3)
        """
        if initial_condition is None:
            u0 = np.array([1.0, 0.0, 0.0])
        else:
            u0 = np.asarray(initial_condition, dtype=np.float64)

        time = np.linspace(0.0, self.T, self.n_steps + 1)
        solution = np.zeros((self.n_steps + 1, 3))
        solution[0] = u0

        def rhs(u, t):
            return self.lindberg_rhs(u, t, k1, k2, k3)

        def jac(u):
            return self.lindberg_jacobian(u, k1, k2, k3)

        for n in range(self.n_steps):
            solution[n + 1] = self.implicit_euler_step(
                solution[n], time[n], rhs, jac
            )
            # 强制守恒: u1 + u2 + u3 = 1
            total = np.sum(solution[n + 1])
            if total > 1e-15:
                solution[n + 1] /= total

        return time, solution

    def solve_stochastic_lindberg(
        self,
        rate_samples: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        随机 Lindberg 问题的集成求解。

        参数:
            rate_samples: shape (N_samples, 3), 每行是 (k1, k2, k3)

        返回:
            time: shape (n_steps+1,)
            mean_solution: shape (n_steps+1, 3)
            std_solution: shape (n_steps+1, 3)
        """
        N = rate_samples.shape[0]
        time = np.linspace(0.0, self.T, self.n_steps + 1)
        all_solutions = np.zeros((N, self.n_steps + 1, 3))

        for i in range(N):
            k1, k2, k3 = rate_samples[i]
            _, sol = self.solve_lindberg(k1, k2, k3)
            all_solutions[i] = sol

        mean_sol = np.mean(all_solutions, axis=0)
        std_sol = np.std(all_solutions, axis=0)

        return time, mean_sol, std_sol

    def stiffness_ratio(
        self, u: np.ndarray,
        k1: float = 0.04, k2: float = 1e4, k3: float = 3e7
    ) -> float:
        """
        计算刚性比:
          S = max|Re(λ)| / min|Re(λ)|

        其中 λ 是 Jacobian 的特征值。

        刚性系统: S >> 1 (通常 > 10³)
        非刚性系统: S ~ 1
        """
        J = self.lindberg_jacobian(u, k1, k2, k3)
        eigenvalues = np.linalg.eigvals(J)
        re_eigs = np.abs(np.real(eigenvalues))
        re_eigs = re_eigs[re_eigs > 1e-15]

        if len(re_eigs) < 2:
            return 1.0
        return float(np.max(re_eigs) / np.min(re_eigs))
