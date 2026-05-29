"""
time_stepping.py
================
博士级时间积分器：Runge-Kutta 2/3 对与前向/后向 Euler 格式

本模块实现利率期限结构随机偏微分方程的时间离散化方法，包括：
  1. 显式 RK2、RK3 与嵌入式 RK23（带局部误差估计）
  2. 后向 Euler 隐式格式（用于抛物型 PDE 的稳定时间推进）

数学理论
--------
对于常微分方程组:
    y'(t) = f(t, y(t)),  y(t0) = y0

显式 RK23（Bogacki-Shampine 二阶/三阶对）:
    k1 = h * f(t_n, y_n)
    k2 = h * f(t_n + h, y_n + k1)
    k3 = h * f(t_n + h/2, y_n + (k1 + k2)/4)
    y_{n+1}^{(2)} = y_n + (k1 + k2)/2                (二阶)
    y_{n+1}^{(3)} = y_n + (k1 + k2 + 4 k3)/6         (三阶)
    e_{n+1} = y_{n+1}^{(3)} - y_{n+1}^{(2)}          (局部误差估计)

后向 Euler（用于热方程型 PDE）:
    (M + h A) u_{n+1} = M u_n + h f_{n+1}
    其中 M 为质量矩阵，A 为刚度矩阵，h 为时间步长。
    无条件稳定，适用于刚性问题。
"""

import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import spsolve


def rk2_step(yprime, t, y, dt):
    """
    RK2（改进 Euler）单步。

    公式:
        k1 = dt * f(t, y)
        k2 = dt * f(t + dt, y + k1)
        y_{new} = y + 0.5 * (k1 + k2)

    Parameters
    ----------
    yprime : callable
        导数函数 f(t, y)，返回 shape 与 y 相同的 ndarray。
    t : float
        当前时间。
    y : np.ndarray
        当前状态。
    dt : float
        时间步长，必须大于 0。

    Returns
    -------
    np.ndarray
        新状态。
    """
    if dt <= 0.0:
        raise ValueError("rk2_step: dt 必须大于 0")
    y = np.asarray(y, dtype=float)
    k1 = dt * np.asarray(yprime(t, y), dtype=float)
    k2 = dt * np.asarray(yprime(t + dt, y + k1), dtype=float)
    return y + 0.5 * (k1 + k2)


def rk3_step(yprime, t, y, dt):
    """
    RK3（Kutta 三阶）单步。

    公式:
        k1 = dt * f(t, y)
        k2 = dt * f(t + dt, y + k1)
        k3 = dt * f(t + dt/2, y + (k1 + k2)/4)
        y_{new} = y + (k1 + k2 + 4 k3) / 6

    Parameters
    ----------
    yprime : callable
        导数函数 f(t, y)。
    t : float
        当前时间。
    y : np.ndarray
        当前状态。
    dt : float
        时间步长，必须大于 0。

    Returns
    -------
    np.ndarray
        新状态。
    """
    if dt <= 0.0:
        raise ValueError("rk3_step: dt 必须大于 0")
    y = np.asarray(y, dtype=float)
    k1 = dt * np.asarray(yprime(t, y), dtype=float)
    k2 = dt * np.asarray(yprime(t + dt, y + k1), dtype=float)
    k3 = dt * np.asarray(yprime(t + 0.5 * dt, y + 0.25 * (k1 + k2)), dtype=float)
    return y + (k1 + k2 + 4.0 * k3) / 6.0


def rk23_integrate(yprime, tspan, y0, n_steps):
    """
    RK23 固定步长积分，返回解与局部误差估计。

    公式参见模块文档字符串。

    Parameters
    ----------
    yprime : callable
        导数函数 f(t, y)。
    tspan : tuple (t0, t1)
        时间区间。
    y0 : np.ndarray
        初始条件。
    n_steps : int
        步数，必须大于 0。

    Returns
    -------
    t : np.ndarray, shape (n_steps+1,)
        时间点。
    y : np.ndarray, shape (n_steps+1, m)
        状态轨迹。
    e : np.ndarray, shape (n_steps+1, m)
        局部误差估计（第一步为 0）。
    """
    if n_steps <= 0:
        raise ValueError("rk23_integrate: n_steps 必须大于 0")
    t0, t1 = tspan
    if t1 <= t0:
        raise ValueError("rk23_integrate: tspan 必须满足 t0 < t1")

    y0 = np.asarray(y0, dtype=float)
    m = y0.shape[0]
    dt = (t1 - t0) / n_steps

    t = np.zeros(n_steps + 1, dtype=float)
    y = np.zeros((n_steps + 1, m), dtype=float)
    e = np.zeros((n_steps + 1, m), dtype=float)

    t[0] = t0
    y[0, :] = y0
    e[0, :] = 0.0

    for i in range(n_steps):
        ti = t[i]
        yi = y[i, :]

        k1 = dt * np.asarray(yprime(ti, yi), dtype=float)
        k2 = dt * np.asarray(yprime(ti + dt, yi + k1), dtype=float)
        k3 = dt * np.asarray(yprime(ti + 0.5 * dt, yi + 0.25 * (k1 + k2)), dtype=float)

        y2 = yi + 0.5 * (k1 + k2)
        y3 = yi + (k1 + k2 + 4.0 * k3) / 6.0

        t[i + 1] = ti + dt
        y[i + 1, :] = y3
        e[i + 1, :] = y3 - y2

    return t, y, e


def backward_euler_step(A, M, u_old, dt, f_rhs, bc_indices=None, bc_values=None):
    """
    后向 Euler 单步求解抛物型 PDE。

    离散格式:
        (M + dt * A) * u_{n+1} = M * u_n + dt * f_rhs

    其中 A 为刚度矩阵（含扩散与对流项），M 为质量矩阵，
    f_rhs 为右端源项。

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        刚度矩阵，shape (N, N)。
    M : scipy.sparse.csr_matrix
        质量矩阵，shape (N, N)。
    u_old : np.ndarray, shape (N,)
        上一时间层的解。
    dt : float
        时间步长，必须大于 0。
    f_rhs : np.ndarray, shape (N,)
        右端源项。
    bc_indices : np.ndarray or None
        Dirichlet 边界节点索引。
    bc_values : np.ndarray or None
        Dirichlet 边界值。

    Returns
    -------
    np.ndarray, shape (N,)
        新时间层的解。
    """
    if dt <= 0.0:
        raise ValueError("backward_euler_step: dt 必须大于 0")
    u_old = np.asarray(u_old, dtype=float)
    f_rhs = np.asarray(f_rhs, dtype=float)
    N = u_old.shape[0]

    lhs = M + dt * A
    rhs = M @ u_old + dt * f_rhs

    # 施加 Dirichlet 边界条件
    if bc_indices is not None and bc_values is not None:
        bc_indices = np.asarray(bc_indices, dtype=int)
        bc_values = np.asarray(bc_values, dtype=float)
        for idx, val in zip(bc_indices, bc_values):
            if 0 <= idx < N:
                # 消去第 idx 行并置 1
                row_start = lhs.indptr[idx]
                row_end = lhs.indptr[idx + 1]
                lhs.data[row_start:row_end] = 0.0
                # 在 (idx, idx) 处置 1
                diag_found = False
                for j in range(row_start, row_end):
                    if lhs.indices[j] == idx:
                        lhs.data[j] = 1.0
                        diag_found = True
                        break
                if not diag_found:
                    # 若对角线不在稀疏结构中，转为 dense 处理（小规模回退）
                    lhs = lhs.todense()
                    lhs = np.array(lhs)
                    lhs[idx, :] = 0.0
                    lhs[idx, idx] = 1.0
                    rhs[idx] = val
                    u_new = np.linalg.solve(lhs, rhs)
                    return u_new
                rhs[idx] = val

    # 使用稀疏直接求解器
    if sp.isspmatrix(lhs):
        u_new = spsolve(lhs.tocsr(), rhs)
    else:
        u_new = np.linalg.solve(lhs, rhs)
    return u_new


def adaptive_rk23(yprime, tspan, y0, tol=1e-6, h_init=0.01, h_min=1e-6, h_max=1.0):
    """
    自适应步长 RK23 积分器。

    步长控制策略:
        h_new = h * min(5, max(0.1, 0.9 * (tol / ||e||)^{1/3}))

    Parameters
    ----------
    yprime : callable
        导数函数 f(t, y)。
    tspan : tuple (t0, t1)
        时间区间。
    y0 : np.ndarray
        初始条件。
    tol : float
        局部误差容限。
    h_init : float
        初始步长猜测。
    h_min : float
        最小允许步长。
    h_max : float
        最大允许步长。

    Returns
    -------
    t_list : list of float
        自适应时间点。
    y_array : np.ndarray
        状态轨迹。
    """
    t0, t1 = tspan
    if t1 <= t0:
        raise ValueError("adaptive_rk23: tspan 必须满足 t0 < t1")
    y0 = np.asarray(y0, dtype=float)
    t = t0
    y = y0.copy()
    h = h_init

    t_list = [t]
    y_list = [y.copy()]

    max_steps = 100000
    step = 0

    while t < t1 and step < max_steps:
        h = min(h, t1 - t)
        if h < h_min:
            raise RuntimeError(f"adaptive_rk23: 步长降至最小值以下 h={h}")

        k1 = h * np.asarray(yprime(t, y), dtype=float)
        k2 = h * np.asarray(yprime(t + h, y + k1), dtype=float)
        k3 = h * np.asarray(yprime(t + 0.5 * h, y + 0.25 * (k1 + k2)), dtype=float)

        y2 = y + 0.5 * (k1 + k2)
        y3 = y + (k1 + k2 + 4.0 * k3) / 6.0
        e = y3 - y2
        err_norm = np.linalg.norm(e) / max(1.0, np.linalg.norm(y3))

        if err_norm <= tol or h <= h_min * 1.1:
            t = t + h
            y = y3
            t_list.append(t)
            y_list.append(y.copy())
            step += 1
            # 接受步长，尝试放大
            factor = 0.9 * (tol / max(err_norm, 1e-15)) ** (1.0 / 3.0)
            factor = min(5.0, max(0.2, factor))
            h = min(factor * h, h_max)
        else:
            # 拒绝步长，缩小重试
            factor = 0.9 * (tol / max(err_norm, 1e-15)) ** (1.0 / 3.0)
            factor = max(0.1, factor)
            h = max(factor * h, h_min)

    y_array = np.array(y_list)
    return np.array(t_list), y_array
