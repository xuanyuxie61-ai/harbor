"""
delay_bistability.py
--------------------
延迟双稳态动力学模块 —— 映射自种子项目 1107_Gelens-Lab_cellcyclemodules
核心思想：求解带延迟的微分代数方程 (DDAE)，描述 SMB
色谱过程中的切换动力学与双稳态现象。

科学背景：
    SMB 色谱柱的切换操作引入离散延迟：
        du/dt = f(u(t), u(t - tau), w)
    其中 tau 为切换周期，w 为不确定性参数。
    双稳态模块 (Gelens Lab 2020)：
        dCdk1/dt = c - Cdk1 * Apc
        dApc/dt = eps * (Cdk1(t-tau)^n / (Cdk1(t-tau)^n + Xi^n) - Apc)
    其中 Xi = 1 + a * Apc * (Apc - 1) * (Apc - r).

    该系统的平衡点、Hopf 分岔、延迟诱导振荡直接影响
    SMB 过程的鲁棒性。本模块求解 DDAE 并分析双稳态。

数值方法：
    采用方法-of-lines + RK4 积分，延迟项通过线性插值近似。
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, Callable


# =============================================================================
# 延迟双稳态 ODE 系统
# =============================================================================
def delay_bist_2d_cubic(
    y: np.ndarray,
    y_hist: np.ndarray,
    t: float,
    tau: float,
    params: dict,
) -> np.ndarray:
    """
    延迟双稳态 2D 系统 (cubic 非线性)：
        dCdk1/dt = c - Cdk1 * Apc
        dApc/dt = eps * (Cdk1(t-tau)^n / (Cdk1(t-tau)^n + Xi^n) - Apc)
    其中 Xi = 1 + a * Apc * (Apc - 1) * (Apc - r).

    参数
    ----
    y : (2,) 当前状态 [Cdk1, Apc]
    y_hist : 延迟状态 (通过插值获取)
    t : 当前时间
    tau : 延迟
    params : 参数字典 {c, eps, n, a, r}
    """
    Cdk1, Apc = y[0], y[1]
    Cdk1_hist = y_hist[0]

    c = params.get("c", 0.5)
    eps = params.get("eps", 0.1)
    n = params.get("n", 4.0)
    a = params.get("a", 1.0)
    r = params.get("r", 0.5)

    # 双稳态非线性
    Xi = 1.0 + a * Apc * (Apc - 1.0) * (Apc - r)
    # 防止负数幂
    Cdk1_hist_n = max(Cdk1_hist, 1e-14) ** n
    Xi_n = max(abs(Xi), 1e-14) ** n

    f = np.zeros(2)
    f[0] = c - Cdk1 * Apc
    f[1] = eps * (Cdk1_hist_n / (Cdk1_hist_n + Xi_n) - Apc)
    return f


def delay_bist_2d_piecewise(
    y: np.ndarray,
    y_hist: np.ndarray,
    t: float,
    tau: float,
    params: dict,
) -> np.ndarray:
    """
    分段线性 Xi 的延迟双稳态系统。
    """
    Cdk1, Apc = y[0], y[1]
    Cdk1_hist = y_hist[0]

    c = params.get("c", 0.5)
    eps = params.get("eps", 0.1)
    n = params.get("n", 4.0)
    x_max = params.get("x_max", 0.3)
    x_min = params.get("x_min", 0.7)
    xi_max = params.get("xi_max", 2.0)
    xi_min = params.get("xi_min", 0.5)

    # 分段线性 Xi
    if Apc <= x_max:
        Xi = (xi_max - 1.0) / max(x_max, 1e-14) * Apc + 1.0
    elif Apc <= x_min:
        Xi = (xi_min - xi_max) / max(x_min - x_max, 1e-14) * (Apc - x_max) + xi_max
    else:
        Xi = (1.0 - xi_min) / max(1.0 - x_min, 1e-14) * (Apc - x_min) + xi_min

    Cdk1_hist_n = max(Cdk1_hist, 1e-14) ** n
    Xi_n = max(abs(Xi), 1e-14) ** n

    f = np.zeros(2)
    f[0] = c - Cdk1 * Apc
    f[1] = eps * (Cdk1_hist_n / (Cdk1_hist_n + Xi_n) - Apc)
    return f


# =============================================================================
# DDAE 求解器 (方法-of-lines + RK4)
# =============================================================================
class DDAESolver:
    """
    延迟微分代数方程求解器。
    采用 RK4 + 历史插值。
    """

    def __init__(
        self,
        rhs_fn: Callable,
        tau: float,
        params: dict,
        dim: int = 2,
    ):
        self.rhs_fn = rhs_fn
        self.tau = float(tau)
        self.params = params
        self.dim = dim

    def solve(
        self,
        y0: np.ndarray,
        T_final: float,
        history_fn: Optional[Callable] = None,
        dt: float = 0.01,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        求解 DDAE：
            dy/dt = f(y(t), y(t-tau), params)
        返回 (y_array, t_array).
        """
        y0 = np.asarray(y0, dtype=float).ravel()
        if y0.size != self.dim:
            raise ValueError("y0 维度不匹配")

        N = int(np.ceil(T_final / dt))
        t_arr = np.linspace(0.0, T_final, N + 1)
        y_arr = np.zeros((N + 1, self.dim))
        y_arr[0] = y0

        # 历史函数 (默认常数)
        if history_fn is None:
            history_fn = lambda t: y0.copy()

        # 存储历史
        history = [y0.copy()]
        t_history = [0.0]

        for i in range(N):
            t = t_arr[i]
            y = y_arr[i]

            # RK4 步
            def get_delay_state(t_eval):
                if t_eval <= 0.0:
                    return history_fn(t_eval)
                # 线性插值
                for k in range(len(t_history) - 1, 0, -1):
                    if t_history[k - 1] <= t_eval <= t_history[k]:
                        alpha = (t_eval - t_history[k - 1]) / max(
                            t_history[k] - t_history[k - 1], 1e-14
                        )
                        return (1.0 - alpha) * y_arr[k - 1] + alpha * y_arr[k]
                return y_arr[-1]

            # 延迟状态
            t_delay = t - self.tau
            y_delay = get_delay_state(t_delay)

            # RK4
            k1 = self.rhs_fn(y, y_delay, t, self.tau, self.params)
            k2 = self.rhs_fn(
                y + 0.5 * dt * k1,
                get_delay_state(t + 0.5 * dt - self.tau),
                t + 0.5 * dt,
                self.tau,
                self.params,
            )
            k3 = self.rhs_fn(
                y + 0.5 * dt * k2,
                get_delay_state(t + 0.5 * dt - self.tau),
                t + 0.5 * dt,
                self.tau,
                self.params,
            )
            k4 = self.rhs_fn(
                y + dt * k3,
                get_delay_state(t + dt - self.tau),
                t + dt,
                self.tau,
                self.params,
            )
            y_new = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            y_arr[i + 1] = y_new
            t_history.append(t_arr[i + 1])

        return y_arr, t_arr


# =============================================================================
# 平衡点与稳定性分析
# =============================================================================
def find_equilibrium(
    params: dict,
    y0: np.ndarray = None,
    max_iter: int = 1000,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, bool]:
    """
    寻找延迟系统的平衡点 (tau=0 时退化为 ODE)。
    牛顿迭代：y_{k+1} = y_k - J^{-1} f(y_k).
    """
    if y0 is None:
        y0 = np.array([0.5, 0.5])
    y = y0.copy()
    c = params.get("c", 0.5)
    eps = params.get("eps", 0.1)
    n = params.get("n", 4.0)
    a = params.get("a", 1.0)
    r = params.get("r", 0.5)

    for it in range(max_iter):
        # f(y)
        Cdk1, Apc = y[0], y[1]
        Xi = 1.0 + a * Apc * (Apc - 1.0) * (Apc - r)
        Xi_n = max(abs(Xi), 1e-14) ** n
        Cdk1_n = max(Cdk1, 1e-14) ** n
        f = np.zeros(2)
        f[0] = c - Cdk1 * Apc
        f[1] = eps * (Cdk1_n / (Cdk1_n + Xi_n) - Apc)

        if np.linalg.norm(f) < tol:
            return y, True

        # 数值 Jacobian
        h = 1e-7
        J = np.zeros((2, 2))
        for j in range(2):
            y_p = y.copy()
            y_m = y.copy()
            y_p[j] += h
            y_m[j] -= h
            # f(y_p)
            C_p, A_p = y_p[0], y_p[1]
            Xi_p = 1.0 + a * A_p * (A_p - 1.0) * (A_p - r)
            Xi_p_n = max(abs(Xi_p), 1e-14) ** n
            C_p_n = max(C_p, 1e-14) ** n
            f_p = np.array([c - C_p * A_p, eps * (C_p_n / (C_p_n + Xi_p_n) - A_p)])
            # f(y_m)
            C_m, A_m = y_m[0], y_m[1]
            Xi_m = 1.0 + a * A_m * (A_m - 1.0) * (A_m - r)
            Xi_m_n = max(abs(Xi_m), 1e-14) ** n
            C_m_n = max(C_m, 1e-14) ** n
            f_m = np.array([c - C_m * A_m, eps * (C_m_n / (C_m_n + Xi_m_n) - A_m)])
            J[:, j] = (f_p - f_m) / (2.0 * h)

        # 牛顿步
        try:
            dy = np.linalg.solve(J + 1e-12 * np.eye(2), -f)
        except np.linalg.LinAlgError:
            dy = -f
        y = y + dy
        # 边界保护
        y = np.maximum(y, 1e-14)

    return y, False


# ----------------------------------------------------------------------
# 自检
# ----------------------------------------------------------------------
if __name__ == "__main__":
    params = {"c": 0.5, "eps": 0.1, "n": 4.0, "a": 1.0, "r": 0.5}
    solver = DDAESolver(delay_bist_2d_cubic, tau=0.5, params=params, dim=2)
    y0 = np.array([0.5, 0.3])
    y_arr, t_arr = solver.solve(y0, T_final=5.0, dt=0.01)
    print(f"DDAE solve: t in [0, {t_arr[-1]:.2f}]")
    print(f"Final state: {y_arr[-1]}")

    # 平衡点
    y_eq, converged = find_equilibrium(params)
    print(f"Equilibrium: {y_eq}, converged={converged}")
