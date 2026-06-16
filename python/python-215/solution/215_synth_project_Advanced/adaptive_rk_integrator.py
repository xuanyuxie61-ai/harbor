#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
adaptive_rk_integrator.py — 自适应 Runge-Kutta 时间积分器

对应种子项目:
  - 972_r8but: Butcher 表 (Butcher tableau) 定义 RK 方法的系数
  - 839_ornstein_uhlenbeck: Euler-Maruyama 随机微分方程求解

核心数学公式:
  一般 RK 方法 (s 级):
      k_i = f(t_n + c_i h, y_n + h Σ_j a_{ij} k_j)
      y_{n+1} = y_n + h Σ_i b_i k_i

  Butcher 表:
      c | A
      ------
        | b^T

  经典 RK4 (4阶):
      0   |
      1/2 | 1/2
      1/2 | 0   1/2
      1   | 0   0    1
      ----|----------------
          | 1/6 1/3  1/3  1/6

  Dormand-Prince RK45 (自适应步长):
      基于局部截断误差估计 ε_{n+1} = |y_{n+1}^{(5)} - y_{n+1}^{(4)}|
      步长调整: h_{new} = h · min(max(0.9 (tol/ε)^{1/5}, 0.2), 5.0)

  Ornstein-Uhlenbeck SDE (Euler-Maruyama):
      dx = θ(μ - x) dt + σ dW
      x_{n+1} = x_n + θ(μ - x_n) dt + σ √dt · N(0,1)
"""

import numpy as np
from typing import Callable, Tuple, Optional


# ---------------------------------------------------------------------------
# Butcher 表定义
# ---------------------------------------------------------------------------
class ButcherTableau:
    """
    Butcher 表封装 — 定义 Runge-Kutta 方法的完整系数.

    属性:
        A  : (s, s) 矩阵
        b  : (s,) 权重向量 (高精度解)
        b_hat : (s,) 权重向量 (低精度解, 用于误差估计, 可选)
        c  : (s,) 节点向量
        order : 方法阶数
        name  : 方法名称
    """

    def __init__(self, A: np.ndarray, b: np.ndarray,
                 c: np.ndarray,
                 b_hat: Optional[np.ndarray] = None,
                 order: int = 4,
                 name: str = "RK"):
        self.A = np.asarray(A, dtype=np.float64)
        self.b = np.asarray(b, dtype=np.float64)
        self.c = np.asarray(c, dtype=np.float64)
        self.b_hat = np.asarray(b_hat, dtype=np.float64) if b_hat is not None else None
        self.order = order
        self.name = name
        self._validate()

    def _validate(self):
        s = self.A.shape[0]
        assert self.A.shape == (s, s), f"A 矩阵须为 ({s},{s})"
        assert self.b.shape == (s,), f"b 须为 ({s},)"
        assert self.c.shape == (s,), f"c 须为 ({s},)"
        if self.b_hat is not None:
            assert self.b_hat.shape == (s,)
        # c_i = Σ_j a_{ij} 一致性检查
        row_sums = self.A.sum(axis=1)
        if not np.allclose(row_sums, self.c, atol=1e-12):
            pass  # 某些 FSAL 方法可能不满足


def butchers_rk4() -> ButcherTableau:
    """经典四阶 Runge-Kutta."""
    A = np.array([
        [0, 0, 0, 0],
        [0.5, 0, 0, 0],
        [0, 0.5, 0, 0],
        [0, 0, 1.0, 0]
    ])
    b = np.array([1/6, 1/3, 1/3, 1/6])
    c = np.array([0, 0.5, 0.5, 1.0])
    return ButcherTableau(A, b, c, order=4, name="RK4")


def butchers_dormand_prince() -> ButcherTableau:
    """
    Dormand-Prince RK4(5) 对 — 自适应步长控制.

    高精度: 5 阶, 低精度: 4 阶.
    局部误差估计: ε = |y^{(5)} - y^{(4)}|
    """
    A = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
    ])
    # 5阶解权重
    b = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
    # 4阶解权重 (用于误差估计)
    b_hat = np.array([5179/57600, 0, 7571/16695, 393/640,
                      -92097/339200, 187/2100, 1/40])
    c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])
    return ButcherTableau(A, b, c, b_hat=b_hat, order=5,
                          name="Dormand-Prince RK45")


# ---------------------------------------------------------------------------
# 确定性 RK 求解器
# ---------------------------------------------------------------------------
def rk_step(f: Callable, t: float, y: np.ndarray,
            h: float, tab: ButcherTableau) -> np.ndarray:
    """
    单步 Runge-Kutta 推进.

    k_i = f(t + c_i h, y + h Σ_j a_{ij} k_j),  i = 1,...,s
    y_{n+1} = y + h Σ_i b_i k_i

    Parameters
    ----------
    f   : 右端函数 (t, y) → dy/dt
    t   : 当前时间
    y   : 当前状态
    h   : 步长
    tab : Butcher 表

    Returns
    -------
    y_new : ndarray — 下一步状态
    """
    s = tab.A.shape[0]
    k = np.zeros((s,) + y.shape, dtype=np.float64)
    for i in range(s):
        ti = t + tab.c[i] * h
        yi = y.copy()
        for j in range(i):
            yi = yi + h * tab.A[i, j] * k[j]
        k[i] = f(ti, yi)
    y_new = y.copy()
    for i in range(s):
        y_new = y_new + h * tab.b[i] * k[i]
    return y_new


def rk45_adaptive(f: Callable, t_span: Tuple[float, float],
                  y0: np.ndarray,
                  tol: float = 1e-6,
                  h_init: float = 0.01,
                  h_min: float = 1e-10,
                  h_max: float = 1.0,
                  max_steps: int = 100000
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dormand-Prince RK45 自适应步长积分.

    步长控制公式:
        ε_{n+1} = || y^{(5)} - y^{(4)} || / tol
        h_{new} = h · clamp(0.9 · ε^{-1/5}, 0.2, 5.0)

    Parameters
    ----------
    f       : 右端函数 (t, y) → dy/dt
    t_span  : (t0, tf)
    y0      : 初始状态
    tol     : 局部截断误差容限
    h_init  : 初始步长
    h_min   : 最小步长
    h_max   : 最大步长
    max_steps : 最大步数

    Returns
    -------
    (t_arr, y_arr) — 时间序列与状态序列
    """
    tab = butchers_dormand_prince()
    t = t_span[0]
    tf = t_span[1]
    y = np.array(y0, dtype=np.float64)
    h = min(h_init, tf - t)

    t_list = [t]
    y_list = [y.copy()]
    step = 0

    while t < tf - 1e-14 and step < max_steps:
        if t + h > tf:
            h = tf - t
        # 5阶解
        y5 = rk_step(f, t, y, h, tab)
        # 4阶解 (误差估计)
        s = tab.A.shape[0]
        k = np.zeros((s,) + y.shape, dtype=np.float64)
        for i in range(s):
            ti = t + tab.c[i] * h
            yi = y.copy()
            for j in range(i):
                yi = yi + h * tab.A[i, j] * k[j]
            k[i] = f(ti, yi)
        y4 = y.copy()
        for i in range(s):
            y4 = y4 + h * tab.b_hat[i] * k[i]

        # 误差估计
        err_vec = y5 - y4
        err = np.linalg.norm(err_vec)
        if err < 1e-16:
            err = 1e-16

        # 步长调整
        safety = 0.9
        factor = safety * (tol / err) ** 0.2
        factor = max(0.2, min(factor, 5.0))
        h_new = h * factor
        h_new = max(h_min, min(h_new, h_max))

        if err <= tol or h <= h_min * 1.01:
            # 接受此步
            t = t + h
            y = y5
            t_list.append(t)
            y_list.append(y.copy())
        # else: 拒绝, 不推进, 缩小步长重试

        h = h_new
        step += 1

    return np.array(t_list), np.array(y_list)


# ---------------------------------------------------------------------------
# Ornstein-Uhlenbeck SDE 求解器 (Euler-Maruyama)
# ---------------------------------------------------------------------------
def ornstein_uhlenbeck_em(theta: float, mu: float, sigma: float,
                          x0: float, tmax: float, n_steps: int,
                          seed: int = 42
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ornstein-Uhlenbeck SDE 的 Euler-Maruyama 求解.

    SDE: dx(t) = θ(μ - x(t)) dt + σ dW(t)

    Euler-Maruyama 离散:
        x_{n+1} = x_n + θ(μ - x_n) dt + σ √(dt) · ξ_n
        ξ_n ~ N(0, 1)

    解析均值: E[x(t)] = μ + (x_0 - μ) exp(-θ t)
    解析方差: Var[x(t)] = σ²/(2θ) · (1 - exp(-2θ t))

    Parameters
    ----------
    theta   : 回复速率 (> 0)
    mu      : 长期均值
    sigma   : 扩散系数 (≥ 0)
    x0      : 初始值
    tmax    : 终止时间
    n_steps : 时间步数
    seed    : 随机种子

    Returns
    -------
    (t_arr, x_arr) — 时间序列与路径
    """
    rng = np.random.default_rng(seed)
    dt = tmax / n_steps
    t_arr = np.linspace(0, tmax, n_steps + 1)
    x_arr = np.zeros(n_steps + 1)
    x_arr[0] = x0

    for n in range(n_steps):
        dW = rng.standard_normal() * np.sqrt(dt)
        x_arr[n + 1] = (x_arr[n]
                        + theta * (mu - x_arr[n]) * dt
                        + sigma * dW)

    return t_arr, x_arr


def ou_analytical_moments(theta: float, mu: float, sigma: float,
                          x0: float, t: float
                          ) -> Tuple[float, float]:
    """
    OU 过程的解析均值和方差.

    E[x(t)] = μ + (x_0 - μ) exp(-θ t)
    Var[x(t)] = σ²/(2θ) · (1 - exp(-2θ t))
    """
    mean = mu + (x0 - mu) * np.exp(-theta * t)
    if abs(theta) < 1e-15:
        var = sigma ** 2 * t
    else:
        var = sigma ** 2 / (2 * theta) * (1 - np.exp(-2 * theta * t))
    return mean, max(var, 0.0)
