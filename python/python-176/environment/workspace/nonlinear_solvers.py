"""
nonlinear_solvers.py
================================================================================
非线性方程求解与特殊函数模块

本模块融合以下种子项目的核心算法：
  - 644_lambert_w : Lambert W 函数的分段解析近似与 Halley 迭代精化

科学背景
--------
在最优控制问题中，非线性反应-扩散方程的稳态解或隐式时间步进
可能产生形如 y e^y = z 的超越方程，其解析解需要用 Lambert W 函数表示：
    y = W(z)

Lambert W 函数定义为 W(z) e^{W(z)} = z 的反函数，有无穷多分支。
在主分支 W_0 上，定义域 z ∈ [−1/e, +∞)，值域 W_0 ∈ [−1, +∞)。
在次分支 W_{−1} 上，定义域 z ∈ [−1/e, 0)，值域 W_{−1} ∈ (−∞, −1]。

在 PDE 约束最优控制中，Lambert W 函数出现在：
  - 某些非线性反应项的稳态解析解
  - 隐式时间离散中非线性方程的精确解
  - 边界层问题中的渐近分析

本模块实现了 Lambert W 函数的高精度近似，并辅以 Newton 迭代求解
一般非线性方程，用于处理最优控制中产生的非线性代数系统。

关键公式
--------
1. Lambert W 定义：
   W(z) · exp(W(z)) = z

2. 导数：
   W'(z) = W(z) / [z (1 + W(z))]   (z ≠ 0, W(z) ≠ −1)

3. 分支点附近展开（z = −1/e + ε）：
   W(z) ≈ −1 + √(2eε) − (2e/3) ε + ...

4. Halley 迭代精化（三阶收敛）：
   w_{n+1} = w_n − (w_n e^{w_n} − z) / [(w_n + 1) e^{w_n} − (w_n + 2)(w_n e^{w_n} − z) / (2w_n + 2)]

5. Newton 法求解 f(y) = 0：
   y_{n+1} = y_n − f(y_n) / f'(y_n)
"""

import numpy as np


def lambert_w_approx(x, branch=0):
    """
    Lambert W 函数的分段解析近似。
    融合 644_lambert_w 的核心算法思想：
    对分支点附近、中间区域和渐近区域分别使用不同的近似公式，
    然后辅以一次 Halley 迭代精化。

    参数
    ----
    x     : 输入值（标量或数组）
    branch: 0 = 主分支 W_0，非零 = 次分支 W_{-1}

    返回
    ----
    w : W(x) 的近似值
    """
    x = np.atleast_1d(x).astype(float)
    w = np.zeros_like(x)

    # 分支点
    em1 = -1.0 / np.e

    for idx in range(x.size):
        xv = x.flat[idx]

        if branch == 0:
            # 主分支 W_0
            if xv < em1:
                # 定义域外，返回 NaN
                w.flat[idx] = np.nan
                continue
            if xv < -0.323581708061267:  # 分支点附近
                # 级数展开
                p = np.sqrt(2.0 * (np.e * xv + 1.0))
                w.flat[idx] = -1.0 + p - p * p / 3.0 + 11.0 * p ** 3 / 72.0
            elif xv < 1.857183860207835:
                # 中间区域，有理近似
                w.flat[idx] = (0.665 * (1.0 + 0.0195 * xv) * np.log(1.0 + xv) +
                               0.04 * xv)
                if xv > 0:
                    w.flat[idx] = np.log(1.0 + xv) * (1.0 - np.log(1.0 + xv) / 3.0)
            else:
                # 大参数渐近：W(x) ≈ log(x) − log(log(x))
                lx = np.log(xv)
                llx = np.log(lx)
                w.flat[idx] = lx - llx + llx / lx
        else:
            # 次分支 W_{-1}
            if xv < em1 or xv > 0.0:
                w.flat[idx] = np.nan
                continue
            if xv < -0.323581708061267:
                p = -np.sqrt(2.0 * (np.e * xv + 1.0))
                w.flat[idx] = -1.0 + p - p * p / 3.0 + 11.0 * p ** 3 / 72.0
            else:
                # 对数近似
                lx = np.log(-xv)
                llx = np.log(-lx)
                w.flat[idx] = lx - llx + llx / lx

    # Halley 迭代精化一次（三阶收敛）
    for idx in range(x.size):
        if not np.isfinite(w.flat[idx]):
            continue
        wv = w.flat[idx]
        ew = np.exp(wv)
        we = wv * ew - x.flat[idx]
        g = (wv + 2.0) * we / (2.0 * wv + 2.0)
        denom = (wv + 1.0) * ew - g
        if abs(denom) > 1.0e-30:
            w.flat[idx] -= we / denom

    return w


def lambert_w_newton(x, branch=0, tol=1.0e-12, max_iter=50):
    """
    使用 Newton 迭代求解 W(x)，从 lambert_w_approx 的近似值出发。
    用于需要更高精度的场景。
    """
    x = float(x)
    w = float(lambert_w_approx(np.array([x]), branch)[0])
    if not np.isfinite(w):
        return w

    for _ in range(max_iter):
        ew = np.exp(w)
        f = w * ew - x
        df = ew * (w + 1.0)
        if abs(df) < 1.0e-30:
            break
        dw = f / df
        w -= dw
        if abs(dw) < tol:
            break
    return w


def newton_solve(f, df, x0, tol=1.0e-10, max_iter=50):
    """
    标准 Newton 法求解标量方程 f(x) = 0。

    参数
    ----
    f        : 目标函数
    df       : 导数函数
    x0       : 初始猜测
    tol      : 收敛容差
    max_iter : 最大迭代次数

    返回
    ----
    x     : 近似根
    conv  : 是否收敛
    iters : 实际迭代次数
    """
    x = float(x0)
    for k in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if abs(dfx) < 1.0e-30:
            return x, False, k
        dx = fx / dfx
        x -= dx
        if abs(dx) < tol:
            return x, True, k + 1
    return x, False, max_iter


def solve_nonlinear_reaction(y_old, dt, c, f_rhs, M=None):
    """
    求解隐式非线性反应项的局部方程：
        y + dt * c * y * exp(y) = rhs
    该方程出现在某些 Arrhenius 型反应动力学中。
    改写为：
        (y + dt*c*y*exp(y)) = rhs
    若 c=0 退化为线性。
    更一般地，对于方程 y + dt * c * y^3 = rhs，可用 Newton 法求解。
    """
    rhs = float(f_rhs)
    if abs(c) < 1.0e-15:
        return rhs

    # 求解 y + dt * c * y^3 = rhs
    def g(y):
        return y + dt * c * y ** 3 - rhs

    def dg(y):
        return 1.0 + 3.0 * dt * c * y ** 2

    x0 = rhs if abs(rhs) < 10.0 else np.cbrt(rhs / (dt * c))
    root, conv, iters = newton_solve(g, dg, x0)
    if not conv:
        # 回退到 Picard 迭代
        y = x0
        for _ in range(100):
            y_new = rhs / (1.0 + dt * c * y ** 2)
            if abs(y_new - y) < 1.0e-12:
                return y_new
            y = y_new
    return root


def nonlinear_rhs_cubic(y, c):
    """
    非线性反应项 R(y) = c · y³。
    导数 dR/dy = 3c · y²。
    """
    return c * y ** 3


def nonlinear_rhs_cubic_derivative(y, c):
    """非线性反应项的导数。"""
    return 3.0 * c * y ** 2
