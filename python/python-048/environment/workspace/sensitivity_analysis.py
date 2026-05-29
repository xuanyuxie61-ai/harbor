"""
sensitivity_analysis.py
裂缝扩展对初始条件的敏感性分析模块

原项目映射:
    1064_sensitive_ode -> 对初始条件敏感的常微分方程

水力压裂过程中，裂缝前缘的微小扰动可能在后续扩展中被指数放大，
导致裂缝网络几何形态的强烈不确定性。该现象数学上可由敏感型 ODE 描述。
本模块实现:
1. 敏感型 ODE 的数值积分（显式 Euler）；
2. 裂缝前缘扰动的 Lyapunov 指数估计；
3. 基于伴随方程的敏感性分析。

核心公式:
1. 敏感型 ODE 系统（Cook, 2013）:
   dy1/dt = y2
   dy2/dt = y1
   解析解: y1(t) = y1(0) cosh(t) + y2(0) sinh(t)
           y2(t) = y1(0) sinh(t) + y2(0) cosh(t)
   该系统的特征值为 ±1，扰动呈指数增长 e^t。

2. 将敏感型 ODE 类比到裂缝前缘扩展:
   设裂缝前缘位置为 r(t)，偏转角度为 θ(t)，
   在均匀应力场中，裂缝倾向于沿最大主应力方向扩展，
   但局部非均质性引入随机扰动:
   dθ/dt = κ (θ - θ_0) + ξ(t)
   其中 κ > 0 对应不稳定（敏感）扩展。

3. Lyapunov 指数估计:
   λ = lim_{t→∞} (1/t) ln( ||δy(t)|| / ||δy(0)|| )
   对于 dy/dt = A y，λ = max(Re(eig(A)))。

4. 伴随敏感性:
   对于前向方程 dy/dt = f(y,p)，伴随变量 λ 满足:
   dλ/dt = -(∂f/∂y)^T λ
   目标泛函 J = ∫ g(y,p) dt 对参数 p 的梯度:
   dJ/dp = ∫ [∂g/∂p + λ^T ∂f/∂p] dt
"""

import numpy as np
from typing import Tuple, Callable
from pore_pressure_solver import euler_integrate


def sensitive_deriv(t: float, y: np.ndarray) -> np.ndarray:
    """
    敏感型 ODE 的右端项。

    方程:
        dy1/dt = y2
        dy2/dt = y1
    """
    y = np.asarray(y, dtype=float)
    if y.size != 2:
        raise ValueError("状态向量维度必须为 2")
    dydt = np.zeros(2)
    dydt[0] = y[1]
    dydt[1] = y[0]
    return dydt


def sensitive_exact(t: float, y0: np.ndarray) -> np.ndarray:
    """
    敏感型 ODE 的精确解。

    公式:
        y(t) = [cosh(t)  sinh(t)] [y1(0)]
               [sinh(t)  cosh(t)] [y2(0)]
    """
    y0 = np.asarray(y0, dtype=float)
    return np.array([
        y0[0] * np.cosh(t) + y0[1] * np.sinh(t),
        y0[0] * np.sinh(t) + y0[1] * np.cosh(t)
    ])


def lyapunov_exponent_euler(y0: np.ndarray, delta_y0: np.ndarray,
                            tspan: Tuple[float, float], n_steps: int) -> float:
    """
    通过显式 Euler 积分估计敏感型 ODE 的最大 Lyapunov 指数。

    公式:
        λ ≈ (1/t) ln( ||δy(t)|| / ||δy(0)|| )
    """
    t, y_base = euler_integrate(sensitive_deriv, tspan, y0, n_steps)
    t, y_pert = euler_integrate(sensitive_deriv, tspan, y0 + delta_y0, n_steps)

    delta_y_final = y_pert[-1, :] - y_base[-1, :]
    norm_final = np.linalg.norm(delta_y_final)
    norm_init = np.linalg.norm(delta_y0)

    if norm_init < 1.0e-15 or norm_final < 1.0e-15:
        return 0.0
    T = tspan[1] - tspan[0]
    return np.log(norm_final / norm_init) / T


class FracturePropagationSensitivity:
    """
    裂缝扩展敏感性分析器。
    """

    def __init__(self, kappa: float = 1.0, sigma_noise: float = 0.05):
        """
        参数:
            kappa: 不稳定增长率。
            sigma_noise: 扰动强度。
        """
        self.kappa = kappa
        self.sigma_noise = sigma_noise

    def front_angle_ode(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        裂缝前缘偏转角度的敏感型演化方程。

        状态向量: [θ, dθ/dt]
        方程:
            dθ/dt = ω
            dω/dt = κ² (θ - θ_0) + ξ(t)
        简化确定性版本:
            dω/dt = κ² θ
        """
        state = np.asarray(state, dtype=float)
        if state.size != 2:
            raise ValueError("状态维度必须为 2")
        dstate = np.zeros(2)
        dstate[0] = state[1]
        dstate[1] = self.kappa ** 2 * state[0]
        return dstate

    def simulate_front(self, y0: np.ndarray, tspan: Tuple[float, float],
                       n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用 Euler 方法模拟裂缝前缘演化。
        """
        return euler_integrate(self.front_angle_ode, tspan, y0, n_steps)

    def adjoint_sensitivity(self, y0: np.ndarray, tspan: Tuple[float, float],
                            n_steps: int, parameter_index: int = 0) -> float:
        """
        使用离散伴随方法估计目标泛函对参数的敏感性。

        目标泛函: J = ∫_0^T (θ(t))² dt
        伴随方程（反向积分）:
            dλ_θ/dt = -∂H/∂θ = -2θ - κ² λ_ω
            dλ_ω/dt = -∂H/∂ω = -λ_θ
        其中 H = θ² + λ_θ ω + λ_ω κ² θ。
        """
        # 前向积分
        t_fwd, y_fwd = self.simulate_front(y0, tspan, n_steps)
        dt = t_fwd[1] - t_fwd[0]

        # 伴随变量初始化
        lambda_final = np.zeros(2)
        lambdas = np.zeros((n_steps + 1, 2))
        lambdas[-1, :] = lambda_final

        # 反向 Euler 积分伴随方程
        for k in range(n_steps, 0, -1):
            theta_k = y_fwd[k, 0]
            # 离散伴随更新
            # λ_k = λ_{k+1} + dt * [2θ_k + κ² λ_{k+1,1}, λ_{k+1,0}]^T
            # 注意符号: dλ/dt = -[2θ, 0]^T - A^T λ
            A = np.array([[0.0, self.kappa ** 2],
                          [1.0, 0.0]])
            rhs = -np.array([2.0 * theta_k, 0.0]) - A.T @ lambdas[k, :]
            lambdas[k - 1, :] = lambdas[k, :] + dt * rhs

        # 对 κ 的梯度
        dJ_dkappa = 0.0
        for k in range(n_steps + 1):
            dJ_dkappa += 2.0 * self.kappa * y_fwd[k, 0] * lambdas[k, 1] * dt

        return float(dJ_dkappa)
