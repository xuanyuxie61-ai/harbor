"""
自适应隐式常微分方程积分器
融合来源: 765_midpoint_adaptive (自适应隐式中点法 + LTE 估计)

功能:
- 对多铁性 TDGL 方程进行自适应时间步进
- 隐式中点法 (theta=0.5) 具有二阶精度和良好稳定性
- 局部截断误差 (LTE) 估计驱动步长自适应
- 包含 Newton-Raphson 迭代求解非线性方程组

数学公式:
    y_{n+1} = y_n + τ f(t_{n+1/2}, (y_n + y_{n+1})/2)

局部截断误差估计 (基于嵌入法):
    LTE ≈ || y_{n+1} - ŷ_{n+1} ||
    其中 ŷ 为低阶估计。

步长更新:
    τ_{new} = κ * τ_n * (1 / ||LTE||)^{1/3}
    其中 κ = 0.85 为安全因子。
"""

import numpy as np
from typing import Callable, Tuple, Optional


class AdaptiveMidpointIntegrator:
    """
    自适应隐式中点法 ODE 积分器。
    """

    def __init__(self, reltol: float = 1e-6, abstol: float = 1e-8,
                 kappa: float = 0.85, theta: float = 0.5,
                 max_iter: int = 50, newton_tol: float = 1e-10):
        self.reltol = reltol
        self.abstol = abstol
        self.kappa = kappa
        self.theta = theta
        self.max_iter = max_iter
        self.newton_tol = newton_tol

    def _newton_solve(self, y_n: np.ndarray, tau: float,
                      f: Callable[[float, np.ndarray], np.ndarray],
                      t_mid: float) -> np.ndarray:
        """
        用 Newton-Raphson 迭代求解隐式中点方程:
            g(Y) = Y - y_n - τ f(t_mid, Y) = 0
        其中 Y = (y_n + y_{n+1})/2 为阶段值。
        """
        m = len(y_n)
        Y = y_n.copy()  # 初始猜测

        for _ in range(self.max_iter):
            f_val = f(t_mid, Y)
            # 数值雅可比 (有限差分)
            J = np.zeros((m, m))
            eps = 1e-8
            for j in range(m):
                Y_pert = Y.copy()
                Y_pert[j] += eps
                f_pert = f(t_mid, Y_pert)
                J[:, j] = (f_pert - f_val) / eps

            # g(Y) = Y - y_n - τ f(t_mid, Y)
            g = Y - y_n - tau * f_val
            if np.linalg.norm(g) < self.newton_tol:
                break

            # Newton 步: Y_new = Y - (I - τ J)^{-1} g
            A = np.eye(m) - tau * J
            try:
                delta = np.linalg.solve(A, g)
            except np.linalg.LinAlgError:
                # 若奇异，加正则化
                A += np.eye(m) * 1e-10
                delta = np.linalg.solve(A, g)
            Y = Y - delta

            # 边界处理: 防止发散
            if not np.all(np.isfinite(Y)):
                Y = y_n.copy()
                break

        return Y

    def _lte_estimate(self, y_prev2: np.ndarray, y_prev1: np.ndarray,
                      y_new: np.ndarray, tau_n: float, tau_nm1: float) -> float:
        """
        估计局部截断误差 (LTE)。
        对于隐式中点法 (二阶)，使用低阶前向 Euler 步作为参考:
            ŷ = y_prev1 + tau_n * f(t_prev1, y_prev1)
            LTE ≈ || y_new - ŷ || / scale
        源自 mad_lte 中利用嵌入法的思想。
        """
        if tau_nm1 <= 0:
            return 0.0
        # 低阶估计: 前向 Euler (一阶)
        y_euler = y_prev1 + tau_n * self._last_f_prev1
        scale = self.reltol * np.maximum(np.maximum(np.abs(y_new), np.abs(y_prev1)), self.abstol) + self.abstol
        # 局部截断误差与步长成正比 (|y_mid - y_euler| ~ O(tau^2), LTE ~ O(tau^3))
        lte = np.max(np.abs(y_new - y_euler) / scale) * tau_n
        if lte <= 0 or not np.isfinite(lte):
            lte = 1e-20
        return lte

    def integrate(self, f: Callable[[float, np.ndarray], np.ndarray],
                  t0: float, tmax: float, y0: np.ndarray,
                  tau0: float) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """
        主积分循环。

        参数:
            f: 右端函数 f(t, y)
            t0, tmax: 时间区间
            y0: 初始条件
            tau0: 初始时间步

        返回:
            t_arr: 时间序列
            y_arr: 解序列 (N, m)
            nstep: 成功步数
            n_rejected: 拒绝步数
        """
        m = len(y0)
        t_list = [t0]
        y_list = [y0.copy()]
        tau = tau0
        n_rejected = 0
        n_fsolve_fail = 0

        t = t0
        y = y0.copy()

        # 前两步用固定步长中点法启动
        y_prev2 = y0.copy()
        y_prev1 = y0.copy()
        self._last_f_prev1 = f(t0, y0)  # 保存前一步右端函数值，用于 LTE 估计

        count = 0
        while t < tmax and count < 100000:
            count += 1
            tau = min(tau, tmax - t)
            if tau <= 1e-30:
                break

            t_mid = t + 0.5 * tau
            try:
                Y = self._newton_solve(y, tau, f, t_mid)
            except Exception:
                n_fsolve_fail += 1
                tau *= 0.5
                n_rejected += 1
                continue

            y_new = y + tau * f(t_mid, Y)

            # LTE 估计（第三步起）
            if count >= 3:
                lte = self._lte_estimate(y_prev2, y_prev1, y_new, tau,
                                          t_list[-1] - t_list[-2] if len(t_list) > 1 else tau)
                tnmax = max(lte, 1e-20)

                if tnmax > 1.0:
                    # 拒绝步，减小步长
                    n_rejected += 1
                    factor = self.kappa * (1.0 / tnmax) ** (1.0 / 3.0)
                    factor = np.clip(factor, 0.02, 1.5)
                    tau *= factor
                    continue
                else:
                    # 接受步，更新步长
                    factor = self.kappa * (1.0 / tnmax) ** (1.0 / 3.0)
                    factor = np.clip(factor, 0.02, 1.5)
                    tau *= factor
            else:
                # 前两步固定步长
                pass

            # 更新历史
            y_prev2 = y_prev1.copy()
            y_prev1 = y.copy()
            y = y_new.copy()
            t += tau if count >= 3 else tau
            t_list.append(t)
            y_list.append(y.copy())
            self._last_f_prev1 = f(t, y)  # 更新右端函数值

        t_arr = np.array(t_list)
        y_arr = np.array(y_list)
        nstep = len(t_list) - 1
        return t_arr, y_arr, nstep, n_rejected


def tdgl_rhs(t: float, state: np.ndarray,
             gamma_P: float, gamma_M: float,
             dFdP_func: Callable[[np.ndarray], np.ndarray],
             dFdM_func: Callable[[np.ndarray], np.ndarray],
             noise_P: Optional[np.ndarray] = None,
             noise_M: Optional[np.ndarray] = None) -> np.ndarray:
    """
    构建 TDGL 方程右端:
        dP/dt = -Γ_P δF/δP + ξ_P
        dM/dt = -Γ_M δF/δM + ξ_M

    参数:
        state: [P_vec; M_vec] 展平向量
        gamma_P, gamma_M: 动力学系数
        dFdP_func, dFdM_func: 变分导数函数
        noise_P, noise_M: 可选热噪声

    返回:
        rhs: 时间导数展平向量
    """
    half = len(state) // 2
    P = state[:half]
    M = state[half:]

    dFdP = dFdP_func(P)
    dFdM = dFdM_func(M)

    dPdt = -gamma_P * dFdP
    dMdt = -gamma_M * dFdM

    if noise_P is not None:
        dPdt += noise_P
    if noise_M is not None:
        dMdt += noise_M

    return np.concatenate([dPdt, dMdt])
