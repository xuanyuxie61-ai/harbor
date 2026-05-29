"""
自适应 Runge-Kutta 时间积分器 (adaptive_rk.py)
===============================================
基于种子项目 1030_rk12_adapt 的嵌入式 RK(1,2) 对思想，
升级为更高阶的 RK45 (Dormand-Prince) 变体，用于地核发电机 MHD 方程的时间演化。

地核发电机的时间尺度跨越多个数量级：
  - 对流翻转时间: ~100 年
  - 磁扩散时间: ~10^4 年
  - 极性反转周期: ~10^5 年

因此自适应时间步进至关重要。

本模块提供：
  - RK12 (Euler-Heun) 自适应积分器（直接来自 rk12_adapt）
  - RK45 Dormand-Prince 自适应积分器
  - 用于 stiff 系统的隐式梯形法则（可选）
"""

import numpy as np
from typing import Callable, Tuple, List


# ---------------------------------------------------------------------------
# 1. RK12 自适应积分器（基于 rk12_adapt）
#    使用 Euler (1阶) 与 Heun (2阶) 的误差估计
# ---------------------------------------------------------------------------
def rk12_adaptive(f: Callable[[float, np.ndarray], np.ndarray],
                  tspan: Tuple[float, float],
                  y0: np.ndarray,
                  dt_init: float,
                  tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用嵌入式 RK(1,2) 对自适应积分 ODE y' = f(t, y)。

    参数:
      f        : 右端项函数 f(t, y) -> y'
      tspan    : (t0, t1)
      y0       : 初始条件
      dt_init  : 初始时间步
      tol      : 每步相对误差容限

    返回:
      t_array : 时间点
      y_array : 解轨迹
      e_array : 每步误差估计

    算法:
      k1 = dt * f(t_n, y_n)
      y1 = y_n + k1                          (Euler, order 1)
      k2 = dt * f(t_n + dt, y_n + k1)
      y2 = y_n + 0.5*k1 + 0.5*k2             (Heun, order 2)
      err  = ||y2 - y1||
      若 err > tol*dt : dt <- dt/2, 拒绝并重试
      若 err < tol*dt/16 : dt <- dt*2, 接受并增长
    """
    t0, t1 = tspan
    y = np.asarray(y0, dtype=float).copy()
    t = float(t0)
    dt = float(dt_init)

    ts = [t]
    ys = [y.copy()]
    es = [0.0]

    while t < t1:
        dt = min(dt, t1 - t)
        k1 = dt * f(t, y)
        y1 = y + k1
        k2 = dt * f(t + dt, y1)
        y2 = y + 0.5 * k1 + 0.5 * k2

        err = float(np.linalg.norm(y2 - y1))
        threshold = tol * abs(dt)

        if err > threshold and dt > 1e-15:
            dt *= 0.5
            continue  # 拒绝此步

        # 接受 Heun 解
        y = y2.copy()
        t += dt
        ts.append(t)
        ys.append(y.copy())
        es.append(err)

        if err < threshold / 16.0:
            dt *= 2.0

    t_array = np.array(ts, dtype=float)
    y_array = np.array(ys, dtype=float)
    e_array = np.array(es, dtype=float)
    return t_array, y_array, e_array


# ---------------------------------------------------------------------------
# 2. RK45 Dormand-Prince 自适应积分器
#    经典 4(5) 阶 embedded Runge-Kutta，局部截断误差 O(h^5)
# ---------------------------------------------------------------------------
def rk45_adaptive(f: Callable[[float, np.ndarray], np.ndarray],
                  tspan: Tuple[float, float],
                  y0: np.ndarray,
                  dt_init: float,
                  tol: float = 1e-8,
                  safety: float = 0.9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Dormand-Prince RK45 自适应积分器。

    Butcher 表 (Dormand-Prince):
      c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
      a 见代码内部
      b5 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
      b4 = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]
    """
    t0, t1 = tspan
    y = np.asarray(y0, dtype=float).copy()
    t = float(t0)
    dt = float(dt_init)

    ts = [t]
    ys = [y.copy()]
    es = [0.0]

    # Dormand-Prince 系数
    a2 = np.array([1.0 / 5.0])
    a3 = np.array([3.0 / 40.0, 9.0 / 40.0])
    a4 = np.array([44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0])
    a5 = np.array([19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0])
    a6 = np.array([9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0])
    a7 = np.array([35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0])

    b5 = a7  # 5阶解权重
    b4 = np.array([5179.0 / 57600.0, 0.0, 7571.0 / 16695.0, 393.0 / 640.0,
                   -92097.0 / 339200.0, 187.0 / 2100.0, 1.0 / 40.0])

    # TODO(Hole_2): 实现 Dormand-Prince RK45 自适应步进循环。
    # 要求:
    #   1. 使用已定义的 Butcher 系数 a2~a7, b5, b4 计算 7 个中间步 k1~k7
    #   2. 组合 5 阶解 y5 和 4 阶解 y4
    #   3. 估计局部截断误差 err，并基于 err 进行步长接受/拒绝及 dt 调整
    #   4. 将接受的步结果追加到 ts, ys, es 列表
    # 注意: 该 ODE 系统来自地核发电机感应方程，具有多尺度刚性特征。
    raise NotImplementedError("Hole_2: RK45 自适应步进循环待实现")

    t_array = np.array(ts, dtype=float)
    y_array = np.array(ys, dtype=float)
    e_array = np.array(es, dtype=float)
    return t_array, y_array, e_array


# ---------------------------------------------------------------------------
# 3. 隐式梯形法则（用于 stiff 子系统）
#    (y_{n+1} - y_n) / dt = 0.5 * (f(t_n, y_n) + f(t_{n+1}, y_{n+1}))
#    对于线性系统 y' = J y，可解析求解：
#    y_{n+1} = (I - 0.5*dt*J)^{-1} (I + 0.5*dt*J) y_n
# ---------------------------------------------------------------------------
def implicit_trapezoidal_linear(J: np.ndarray,
                                 y0: np.ndarray,
                                 dt: float,
                                 n_steps: int) -> np.ndarray:
    """
    对线性系统 y' = J y 使用隐式梯形法则推进 n_steps 步。

    公式:
      M = I - 0.5*dt*J
      N = I + 0.5*dt*J
      y_{n+1} = M^{-1} N y_n
    """
    J = np.asarray(J, dtype=float)
    y = np.asarray(y0, dtype=float).copy()
    n = y.size
    I = np.eye(n, dtype=float)
    M = I - 0.5 * dt * J
    N = I + 0.5 * dt * J
    # 求解 M y_{new} = N y
    for _ in range(n_steps):
        rhs = N @ y
        y_new = np.linalg.solve(M, rhs)
        y = y_new
    return y


# ---------------------------------------------------------------------------
# 4. 地核发电机专用：多尺度混合积分器
#    对非 stiff 部分用 RK45，对 stiff 扩散部分用隐式梯形
# ---------------------------------------------------------------------------
def hybrid_integrator(f_nonstiff: Callable[[float, np.ndarray], np.ndarray],
                      J_stiff: np.ndarray,
                      tspan: Tuple[float, float],
                      y0: np.ndarray,
                      dt_init: float,
                      stiff_fraction: float = 0.5,
                      tol: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """
    混合积分器：将状态分为非 stiff 和 stiff 两部分。
    这里假设前 n_stiff 个分量为 stiff（由扩散主导），其余为非 stiff。
    """
    t0, t1 = tspan
    y = np.asarray(y0, dtype=float).copy()
    n = y.size
    n_stiff = max(1, int(n * stiff_fraction))

    t = float(t0)
    dt = float(dt_init)
    ts = [t]
    ys = [y.copy()]

    I_full = np.eye(n, dtype=float)
    M = I_full - 0.5 * dt * J_stiff
    N = I_full + 0.5 * dt * J_stiff

    while t < t1:
        dt = min(dt, t1 - t)
        # stiff 部分隐式推进
        rhs = N @ y
        y = np.linalg.solve(M, rhs)
        # 非 stiff 部分显式 RK12 半步
        y = y + dt * f_nonstiff(t, y)
        t += dt
        ts.append(t)
        ys.append(y.copy())

    return np.array(ts, dtype=float), np.array(ys, dtype=float)


# ---------------------------------------------------------------------------
# 自测试
# ---------------------------------------------------------------------------
def _self_test():
    # 测试 RK12 对 y' = -y, y(0)=1
    f = lambda t, y: -y
    t, y, e = rk12_adaptive(f, (0.0, 1.0), np.array([1.0]), 0.1, tol=1e-5)
    assert abs(y[-1, 0] - np.exp(-1.0)) < 1e-3

    # 测试 RK45
    t, y, e = rk45_adaptive(f, (0.0, 1.0), np.array([1.0]), 0.1, tol=1e-8)
    assert abs(y[-1, 0] - np.exp(-1.0)) < 1e-6

    # 测试隐式梯形
    J = np.array([[-10.0]])
    y = implicit_trapezoidal_linear(J, np.array([1.0]), 0.01, 100)
    # y' = -10y, 精确解 y(1) = exp(-10)
    assert abs(y[0] - np.exp(-10.0)) < 1e-3

    print("adaptive_rk: self-test passed.")


if __name__ == "__main__":
    _self_test()
