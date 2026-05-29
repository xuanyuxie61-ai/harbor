"""
hooke_jeeves.py
===============
基于种子项目 1266_toms178 的直接搜索优化模块。
实现 Hooke-Jeeves 模式搜索算法，用于物理信息 GAN 的超参数优化、
隐向量精修以及损失权重自适应调节。

核心数学：
  Hooke-Jeeves 直接搜索（Hooke & Jeeves, 1961）：
    给定目标函数 F: ℝⁿ → ℝ，初始点 x₀，步长参数 ρ ∈ (0,1)，
    收敛阈值 ε > 0，最大迭代次数 itermax。

    迭代过程：
      1. 探测移动（Exploratory Move）：
         沿每个坐标轴方向 ±δ_i 探测：
           若 F(x + δ_i·e_i) < F(x)，则 x ← x + δ_i·e_i，δ_i 保持符号；
           否则尝试 F(x - δ_i·e_i)，成功则更新，否则 δ_i 不变。
      2. 模式移动（Pattern Move）：
         若探测移动成功（F_new < F_old），则沿该方向外推：
           x_pattern = x_new + (x_new - x_old)
         并以 x_pattern 为基点继续探测。
      3. 步长收缩：
         若探测与模式均失败，则 δ ← ρ·δ。
         当 ||δ|| ≤ ε 时终止。

    步长控制：
      ρ^(iterations) ≈ ε   ⇒   iterations ≈ log(ε)/log(ρ)
"""

import numpy as np


def best_nearby(delta: np.ndarray, x: np.ndarray, fbefore: float,
                nvars: int, f, funevals: int) -> tuple:
    """
    沿每个坐标轴方向进行邻近搜索，寻找更优解。

    Parameters
    ----------
    delta : np.ndarray, shape (nvars,)
        各坐标轴步长。
    x : np.ndarray, shape (nvars,)
        当前基点。
    fbefore : float
        当前基点函数值。
    nvars : int
        维度。
    f : callable
        目标函数 f(x) → float。
    funevals : int
        已评估次数。

    Returns
    -------
    newf, newx, funevals : float, np.ndarray, int
    """
    newx = np.copy(x)
    newf = fbefore
    for i in range(nvars):
        # 正向探测
        z = np.copy(newx)
        z[i] += delta[i]
        fz = f(z)
        funevals += 1
        if fz < newf:
            newf = fz
            newx = z
        else:
            # 负向探测
            z = np.copy(newx)
            z[i] -= delta[i]
            fz = f(z)
            funevals += 1
            if fz < newf:
                newf = fz
                newx = z
    return newf, newx, funevals


def hooke_jeeves(nvars: int, startpt: np.ndarray, rho: float,
                 eps: float, itermax: int, f) -> tuple:
    """
    Hooke-Jeeves 模式搜索算法。

    Parameters
    ----------
    nvars : int
        维度。
    startpt : np.ndarray, shape (nvars,)
        初始点。
    rho : float
        步长收缩因子 (0,1)。
    eps : float
        收敛阈值。
    itermax : int
        最大迭代次数。
    f : callable
        目标函数 f(x) → float。

    Returns
    -------
    iters, endpt : int, np.ndarray
        迭代次数与最优点。
    """
    startpt = np.asarray(startpt, dtype=float)
    if startpt.shape[0] != nvars:
        raise ValueError("startpt 维度与 nvars 不匹配。")
    if not (0.0 < rho < 1.0):
        raise ValueError("rho 必须在 (0,1) 区间内。")

    newx = np.copy(startpt)
    xbefore = np.copy(startpt)
    delta = np.zeros(nvars)
    for i in range(nvars):
        if startpt[i] == 0.0:
            delta[i] = rho
        else:
            delta[i] = rho * abs(startpt[i])

    funevals = 0
    steplength = float(np.max(np.abs(delta)))
    iters = 0
    fbefore = f(newx)
    funevals += 1
    newf = fbefore

    while iters < itermax and eps < steplength:
        iters += 1
        newx = np.copy(xbefore)
        newf, newx, funevals = best_nearby(delta, newx, fbefore,
                                           nvars, f, funevals)

        keep = 1
        while newf < fbefore and keep == 1:
            for i in range(nvars):
                if newx[i] <= xbefore[i]:
                    delta[i] = -abs(delta[i])
                else:
                    delta[i] = abs(delta[i])
                tmp = xbefore[i]
                xbefore[i] = newx[i]
                newx[i] = newx[i] + newx[i] - tmp

            fbefore = newf
            newf, newx, funevals = best_nearby(delta, newx, fbefore,
                                               nvars, f, funevals)

            if fbefore <= newf:
                break

            keep = 0
            for i in range(nvars):
                if 0.5 * abs(delta[i]) < abs(newx[i] - xbefore[i]):
                    keep = 1
                    break

        if eps <= steplength and fbefore <= newf:
            steplength *= rho
            delta *= rho

    endpt = np.copy(xbefore)
    return iters, endpt


def optimize_gan_hyperparams(initial_params: np.ndarray,
                             loss_evaluator, rho: float = 0.85,
                             eps: float = 1e-4, itermax: int = 30) -> tuple:
    """
    使用 Hooke-Jeeves 优化 GAN 超参数（如学习率、物理损失权重等）。

    Parameters
    ----------
    initial_params : np.ndarray
        初始超参数向量（如 [lr_g, lr_d, lambda_phys, lambda_equiv]）。
    loss_evaluator : callable
        输入超参数向量，返回需要最小化的标量损失。
    rho, eps, itermax : Hooke-Jeeves 参数。

    Returns
    -------
    best_params : np.ndarray
        优化后的超参数。
    history : list
        每次迭代的损失值记录。
    """
    nvars = len(initial_params)
    history = []

    def wrapped_f(x):
        val = loss_evaluator(x)
        history.append(float(val))
        return float(val)

    iters, best = hooke_jeeves(nvars, initial_params, rho, eps, itermax, wrapped_f)
    return best, history
