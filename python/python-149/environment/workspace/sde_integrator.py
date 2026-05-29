"""
sde_integrator.py
随机微分方程数值积分器

融合种子项目:
  - 1063_sde: Euler-Maruyama, Milstein方法, 均方稳定性分析
  - 766_midpoint_explicit: 显式中点法思想 → 随机中点方法

科学背景:
  对于神经SDE:
      dX = f(X,u) dt + g(X) dW

  提供以下数值方法:
  1. Euler-Maruyama (强收敛阶 0.5):
      X_{j} = X_{j-1} + f(X_{j-1}) Δt + g(X_{j-1}) ΔW_j

  2. Milstein (强收敛阶 1.0):
      X_{j} = X_{j-1} + f(X_{j-1}) Δt + g(X_{j-1}) ΔW_j
                + 0.5 * g(X_{j-1}) g'(X_{j-1}) (ΔW_j^2 - Δt)

  3. 随机显式中点法 (Stochastic Explicit Midpoint, SEM):
      Y_m = X_{j-1} + 0.5 Δt f(X_{j-1}) + 0.5 g(X_{j-1}) ΔW_j
      X_j = X_{j-1} + Δt f(Y_m) + g(Y_m) ΔW_j

  均方稳定性条件:
      |1 + λΔt|^2 + |μ|^2 Δt < 1   (对于线性测试方程 dX = λX dt + μX dW)
"""

import numpy as np
from typing import Callable, Optional, Tuple


def euler_maruyama(
    f: Callable[[float, np.ndarray], np.ndarray],
    g: Callable[[float, np.ndarray], np.ndarray],
    tspan: Tuple[float, float],
    y0: np.ndarray,
    n_steps: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Euler-Maruyama方法求解多维SDE。

    Parameters
    ----------
    f : callable
        漂移系数 f(t, y) → ndarray
    g : callable
        扩散系数 g(t, y) → ndarray (对角噪声) 或矩阵 (一般噪声)
    tspan : (t0, tstop)
    y0 : ndarray
        初始条件
    n_steps : int
        时间步数
    rng : np.random.Generator or None

    Returns
    -------
    t : ndarray, shape (n_steps+1,)
    y : ndarray, shape (n_steps+1, dim)
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)

    t0, tstop = tspan
    dt = (tstop - t0) / n_steps
    if dt <= 0:
        raise ValueError("时间步长必须为正")

    dim = len(np.atleast_1d(y0))
    t = np.linspace(t0, tstop, n_steps + 1)
    y = np.zeros((n_steps + 1, dim))
    y[0, :] = np.atleast_1d(y0)

    sqrt_dt = np.sqrt(dt)

    for j in range(1, n_steps + 1):
        y_prev = y[j - 1, :]
        t_prev = t[j - 1]

        # 漂移项
        drift = np.atleast_1d(f(t_prev, y_prev))
        # 扩散项
        diffusion = np.atleast_1d(g(t_prev, y_prev))

        # Wiener增量
        dW = sqrt_dt * rng.standard_normal(dim)

        # 边界保护
        if not np.all(np.isfinite(drift)):
            drift = np.zeros(dim)
        if not np.all(np.isfinite(diffusion)):
            diffusion = np.zeros(dim)

        y[j, :] = y_prev + drift * dt + diffusion * dW

        # 硬边界约束：活动率不能为负或超过1
        y[j, :] = np.clip(y[j, :], -0.1, 1.1)

    return t, y


def milstein_method(
    f: Callable[[float, np.ndarray], np.ndarray],
    g: Callable[[float, np.ndarray], np.ndarray],
    dg: Callable[[float, np.ndarray], np.ndarray],
    tspan: Tuple[float, float],
    y0: np.ndarray,
    n_steps: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Milstein方法求解标量或多维对角SDE。

    对于第i维:
        X_i = X_{i,prev} + f_i Δt + g_i ΔW_i + 0.5 * g_i * dg_i * (ΔW_i^2 - Δt)

    其中 dg_i = ∂g_i/∂X_i。

    Parameters
    ----------
    dg : callable
        扩散项对状态的导数 dg(t, y) → ndarray
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)

    t0, tstop = tspan
    dt = (tstop - t0) / n_steps
    if dt <= 0:
        raise ValueError("时间步长必须为正")

    dim = len(np.atleast_1d(y0))
    t = np.linspace(t0, tstop, n_steps + 1)
    y = np.zeros((n_steps + 1, dim))
    y[0, :] = np.atleast_1d(y0)

    sqrt_dt = np.sqrt(dt)

    for j in range(1, n_steps + 1):
        y_prev = y[j - 1, :]
        t_prev = t[j - 1]

        drift = np.atleast_1d(f(t_prev, y_prev))
        diff = np.atleast_1d(g(t_prev, y_prev))
        diff_deriv = np.atleast_1d(dg(t_prev, y_prev))

        dW = sqrt_dt * rng.standard_normal(dim)

        if not np.all(np.isfinite(drift)):
            drift = np.zeros(dim)
        if not np.all(np.isfinite(diff)):
            diff = np.zeros(dim)
        if not np.all(np.isfinite(diff_deriv)):
            diff_deriv = np.zeros(dim)

        # TODO: Hole 2 — 补充Milstein方法的修正项与状态更新
        # 提示: 对于对角噪声SDE，Milstein修正项为 0.5 * g_i * g'_i * (ΔW_i^2 - Δt)
        #       状态更新: Y_j = Y_{j-1} + fΔt + gΔW + Milstein修正
        pass

    return t, y


def stochastic_explicit_midpoint(
    f: Callable[[float, np.ndarray], np.ndarray],
    g: Callable[[float, np.ndarray], np.ndarray],
    tspan: Tuple[float, float],
    y0: np.ndarray,
    n_steps: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    随机显式中点法 (Stochastic Explicit Midpoint):

        Y_m = Y_{j-1} + 0.5 Δt f(Y_{j-1}) + 0.5 g(Y_{j-1}) ΔW_j
        Y_j = Y_{j-1} + Δt f(Y_m) + g(Y_m) ΔW_j

    对于Stratonovich SDE具有更好的稳定性。
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)

    t0, tstop = tspan
    dt = (tstop - t0) / n_steps
    if dt <= 0:
        raise ValueError("时间步长必须为正")

    dim = len(np.atleast_1d(y0))
    t = np.linspace(t0, tstop, n_steps + 1)
    y = np.zeros((n_steps + 1, dim))
    y[0, :] = np.atleast_1d(y0)

    sqrt_dt = np.sqrt(dt)

    for j in range(1, n_steps + 1):
        y_prev = y[j - 1, :]
        t_prev = t[j - 1]

        drift = np.atleast_1d(f(t_prev, y_prev))
        diff = np.atleast_1d(g(t_prev, y_prev))
        dW = sqrt_dt * rng.standard_normal(dim)

        if not np.all(np.isfinite(drift)):
            drift = np.zeros(dim)
        if not np.all(np.isfinite(diff)):
            diff = np.zeros(dim)

        # 中点预测
        y_mid = y_prev + 0.5 * drift * dt + 0.5 * diff * dW
        y_mid = np.clip(y_mid, -0.1, 1.1)

        # 中点处的系数
        t_mid = t_prev + 0.5 * dt
        drift_mid = np.atleast_1d(f(t_mid, y_mid))
        diff_mid = np.atleast_1d(g(t_mid, y_mid))

        if not np.all(np.isfinite(drift_mid)):
            drift_mid = np.zeros(dim)
        if not np.all(np.isfinite(diff_mid)):
            diff_mid = np.zeros(dim)

        y[j, :] = y_prev + drift_mid * dt + diff_mid * dW
        y[j, :] = np.clip(y[j, :], -0.1, 1.1)

    return t, y


def generate_brownian_path(
    tspan: Tuple[float, float],
    n_steps: int,
    dim: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成离散化的布朗运动路径 W(t)。

        dW_j ~ N(0, dt)
        W_j = sum_{k=1}^j dW_k

    Returns
    -------
    t : ndarray
    W : ndarray, shape (n_steps+1, dim)
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)

    t0, tstop = tspan
    dt = (tstop - t0) / n_steps
    t = np.linspace(t0, tstop, n_steps + 1)

    dW = np.sqrt(dt) * rng.standard_normal((n_steps, dim))
    W = np.zeros((n_steps + 1, dim))
    W[1:, :] = np.cumsum(dW, axis=0)

    return t, W


def mean_square_stability_check(
    lambda_val: float,
    mu_val: float,
    dt: float,
) -> bool:
    """
    检查Euler-Maruyama方法对线性测试方程的均方稳定性:

        dX = λ X dt + μ X dW

    均方稳定条件:
        |1 + λΔt|^2 + |μ|^2 Δt < 1

    等价于:
        (1 + λΔt)^2 + μ^2 Δt < 1
        => 1 + 2λΔt + λ^2 Δt^2 + μ^2 Δt < 1
        => Δt (2λ + λ^2 Δt + μ^2) < 0

    对于 λ < 0, 稳定步长上限:
        Δt < -(2λ + μ^2) / λ^2

    Parameters
    ----------
    lambda_val : float
        漂移系数
    mu_val : float
        扩散系数
    dt : float
        时间步长

    Returns
    -------
    stable : bool
        是否均方稳定
    """
    lhs = (1.0 + lambda_val * dt) ** 2 + (mu_val ** 2) * dt
    return lhs < 1.0


def compute_strong_error(
    f: Callable,
    g: Callable,
    y0: float,
    tspan: Tuple[float, float],
    n_ref: int,
    n_coarse_list: list,
    n_paths: int = 500,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算Euler-Maruyama方法的强收敛误差:

        ε_strong(Δt) = E[ |X_L - X(T)| ]

    使用精细解作为参考真解，计算不同粗网格步长的末端误差。

    Returns
    -------
    dt_vals : ndarray
        各粗网格步长
    errors : ndarray
        各步长对应的强误差样本均值
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)

    t0, tstop = tspan
    dt_ref = (tstop - t0) / n_ref

    # 预生成所有路径的布朗增量（精细网格）
    dW_ref = np.sqrt(dt_ref) * rng.standard_normal((n_paths, n_ref))

    # 精细参考解
    X_ref = np.zeros(n_paths)
    for p in range(n_paths):
        xtemp = y0
        for j in range(n_ref):
            xtemp = xtemp + f(0.0, xtemp) * dt_ref + g(0.0, xtemp) * dW_ref[p, j]
        X_ref[p] = xtemp

    dt_vals = []
    errors = []

    for n_coarse in n_coarse_list:
        if n_ref % n_coarse != 0:
            continue
        ratio = n_ref // n_coarse
        dt_coarse = ratio * dt_ref

        X_coarse = np.zeros(n_paths)
        for p in range(n_paths):
            xtemp = y0
            for j in range(n_coarse):
                winc = np.sum(dW_ref[p, j * ratio:(j + 1) * ratio])
                xtemp = xtemp + f(0.0, xtemp) * dt_coarse + g(0.0, xtemp) * winc
            X_coarse[p] = xtemp

        err = np.mean(np.abs(X_coarse - X_ref))
        dt_vals.append(dt_coarse)
        errors.append(err)

    return np.array(dt_vals), np.array(errors)
