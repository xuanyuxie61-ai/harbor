"""
time_integrator.py
==================
自适应时间积分器模块。

融合原项目 1038_rkf45（Runge-Kutta-Fehlberg 自适应步长 ODE 求解器）与
1039_rng_cliff（Cliff 随机数生成器）的核心思想，
为 ICF 内爆多物理耦合系统提供鲁棒的时间积分。

RKF45 算法（4阶/5阶嵌入 Runge-Kutta）:
    y_{n+1} = y_n + h * sum_{i=1}^6 b_i * k_i       (5阶)
    z_{n+1} = y_n + h * sum_{i=1}^6 b*_i * k_i      (4阶)
    误差估计: e = |y_{n+1} - z_{n+1}|
    步长调整: h_new = h * (tol / e)^(1/5)

Cliff RNG 用于在极端条件下（如时间步过小）引入受控数值扰动，
增强算法对刚性问题的鲁棒性。

物理系统 ODE:
    dy/dt = f(t, y)
    y = [r_nodes, u_nodes, e_cells, T_e_cells, T_i_cells]
"""

import numpy as np
from typing import Callable, Tuple
from icf_parameters import NP
from utils import clamp


# ========================================================================
# Cliff 随机数生成器（基于原项目 1039_rng_cliff）
# ========================================================================

class CliffRNG:
    """Cliff 随机数生成器：x_{n+1} = mod(-100 * ln(x_n), 1)。"""

    def __init__(self, seed: float = 0.314159265):
        self.x = clamp(seed, 1.0e-10, 1.0 - 1.0e-10)

    def next(self) -> float:
        """生成下一个 (0,1) 随机数。"""
        self.x = np.fmod(-100.0 * np.log(self.x), 1.0)
        if self.x <= 0.0 or self.x >= 1.0:
            self.x = 0.5
        return self.x

    def perturbation(self, magnitude: float = 1.0e-12) -> float:
        """生成受控扰动量。"""
        return magnitude * (2.0 * self.next() - 1.0)


# ========================================================================
# RKF45 嵌入式 Runge-Kutta（基于原项目 1038_rkf45）
# ========================================================================

class RKF45Integrator:
    """
    Runge-Kutta-Fehlberg 4(5) 阶自适应积分器。

    Butcher 表（经典 RKF45）：
      0     |
      1/4   | 1/4
      3/8   | 3/32        9/32
      12/13 | 1932/2197  -7200/2197   7296/2197
      1     | 439/216    -8            3680/513   -845/4104
      1/2   | -8/27       2           -3544/2565   1859/4104   -11/40
      ------+-----------------------------------------------------------
      b5    | 16/135      0            6656/12825  28561/56430  -9/50     2/55
      b4    | 25/216      0            1408/2565   2197/4104    -1/5      0
    """

    def __init__(self, reltol: float = NP.ADAPTIVE_TOL,
                 abstol: float = NP.ADAPTIVE_TOL * 1.0e-3):
        self.reltol = reltol
        self.abstol = abstol
        self.cliff = CliffRNG(seed=0.2718281828)
        self.nfe = 0  # 函数评估计数

    def _compute_ks(self, f: Callable, t: float, y: np.ndarray, h: float) -> Tuple[np.ndarray, ...]:
        """计算 6 个 RK 阶段。"""
        k1 = f(t, y)
        k2 = f(t + h / 4.0, y + h * k1 / 4.0)
        k3 = f(t + 3.0 * h / 8.0, y + h * (3.0 * k1 + 9.0 * k2) / 32.0)
        k4 = f(t + 12.0 * h / 13.0,
               y + h * (1932.0 * k1 - 7200.0 * k2 + 7296.0 * k3) / 2197.0)
        k5 = f(t + h,
               y + h * (439.0 * k1 / 216.0 - 8.0 * k2 + 3680.0 * k3 / 513.0
                        - 845.0 * k4 / 4104.0))
        k6 = f(t + h / 2.0,
               y + h * (-8.0 * k1 / 27.0 + 2.0 * k2 - 3544.0 * k3 / 2565.0
                        + 1859.0 * k4 / 4104.0 - 11.0 * k5 / 40.0))
        self.nfe += 6
        return k1, k2, k3, k4, k5, k6

    def step(self, f: Callable, t: float, y: np.ndarray, h: float) -> Tuple[np.ndarray, float, float, bool]:
        """
        尝试一个 RKF45 步。

        返回
        ----
        y_new, t_new, h_new, accepted
        """
        k1, k2, k3, k4, k5, k6 = self._compute_ks(f, t, y, h)

        # 5 阶解
        y5 = y + h * (16.0 * k1 / 135.0 + 6656.0 * k3 / 12825.0
                      + 28561.0 * k4 / 56430.0 - 9.0 * k5 / 50.0
                      + 2.0 * k6 / 55.0)

        # 4 阶解
        y4 = y + h * (25.0 * k1 / 216.0 + 1408.0 * k3 / 2565.0
                      + 2197.0 * k4 / 4104.0 - k5 / 5.0)

        # 误差估计
        e = np.abs(y5 - y4)
        scale = self.reltol * np.maximum(np.abs(y5), np.abs(y)) + self.abstol
        err = np.max(e / scale)

        if err <= 1.0 or h <= NP.MIN_DT * 2.0:
            # 接受步
            h_new = h * min(5.0, max(0.1, 0.9 * (1.0 / err)**0.2))
            h_new = clamp(h_new, NP.MIN_DT, NP.MAX_DT)
            # 加入受控数值扰动以增强刚性鲁棒性
            y5 = y5 + self.cliff.perturbation(magnitude=self.abstol * h)
            return y5, t + h, h_new, True
        else:
            # 拒绝步，减小步长
            h_new = h * max(0.1, 0.9 * (1.0 / err)**0.2)
            h_new = clamp(h_new, NP.MIN_DT, h)
            return y, t, h_new, False

    def integrate(self, f: Callable, t0: float, y0: np.ndarray,
                  t_end: float, h0: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        从 t0 积分到 t_end。

        返回
        ----
        t_history, y_history, dt_history
        """
        if h0 is None:
            h0 = NP.MAX_DT * 0.1

        t = t0
        y = np.array(y0, dtype=float)
        h = clamp(h0, NP.MIN_DT, NP.MAX_DT)

        t_list = [t]
        y_list = [y.copy()]
        dt_list = [h]

        max_steps = 100000
        step_count = 0

        while t < t_end and step_count < max_steps:
            h = min(h, t_end - t)
            y_new, t_new, h_new, accepted = self.step(f, t, y, h)

            if accepted:
                t = t_new
                y = y_new
                t_list.append(t)
                y_list.append(y.copy())

            h = h_new
            dt_list.append(h)
            step_count += 1

        return np.array(t_list), np.array(y_list), np.array(dt_list)


# ========================================================================
# 简化的显式 Euler（备用）
# ========================================================================

def explicit_euler_step(f: Callable, t: float, y: np.ndarray, h: float) -> np.ndarray:
    """显式 Euler 单步。"""
    return y + h * f(t, y)


def adaptive_euler(f: Callable, t0: float, y0: np.ndarray,
                   t_end: float, tol: float = 1.0e-6) -> Tuple[np.ndarray, np.ndarray]:
    """简单自适应 Euler（Richardson 外推）。"""
    t = t0
    y = np.array(y0, dtype=float)
    h = NP.MAX_DT * 0.1

    t_list = [t]
    y_list = [y.copy()]

    while t < t_end and len(t_list) < 50000:
        h = min(h, t_end - t)

        y1 = explicit_euler_step(f, t, y, h)
        y2a = explicit_euler_step(f, t, y, h / 2.0)
        y2 = explicit_euler_step(f, t + h / 2.0, y2a, h / 2.0)

        err = np.max(np.abs(y1 - y2))
        if err < tol or h <= NP.MIN_DT:
            y = y2
            t += h
            t_list.append(t)
            y_list.append(y.copy())
            h = min(h * 1.5, NP.MAX_DT)
        else:
            h = max(h * 0.5, NP.MIN_DT)

    return np.array(t_list), np.array(y_list)
