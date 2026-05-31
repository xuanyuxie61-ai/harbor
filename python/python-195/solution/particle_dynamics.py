"""
particle_dynamics.py
粒子动力学推进模块

基于 Runge-Kutta-Fehlberg (RKF45) 自适应步长积分器，
求解粒子在混沌对流场（Rucklidge/Arneodo 系统）中的运动方程。

物理背景：
在双对流或等离子体湍流中，粒子轨迹由三维自治ODE系统描述。
采用 RKF45 (4阶/5阶 embedded Runge-Kutta) 进行自适应积分，
局部截断误差通过 4阶与5阶解的差估计，动态调整步长。

核心公式：
    - Rucklidge 系统:
        dx/dt = kappa * x - lambda * y - y*z
        dy/dt = x
        dz/dt = -z + y^2
    
    - Arneodo 系统:
        dx/dt = y
        dy/dt = z
        dz/dt = -alpha*x - beta*y - z + delta*x^3
    
    - RKF45 Butcher 表（经典系数）:
        k1 = h * f(t_n, y_n)
        k2 = h * f(t_n + h/4, y_n + k1/4)
        k3 = h * f(t_n + 3h/8, y_n + 3k1/32 + 9k2/32)
        k4 = h * f(t_n + 12h/13, y_n + 1932k1/2197 - 7200k2/2197 + 7296k3/2197)
        k5 = h * f(t_n + h, y_n + 439k1/216 - 8k2 + 3680k3/513 - 845k4/4104)
        k6 = h * f(t_n + h/2, y_n - 8k1/27 + 2k2 - 3544k3/2565 + 1859k4/4104 - 11k5/40)
    
    - 4阶解: y_{n+1} = y_n + 25k1/216 + 1408k3/2565 + 2197k4/4104 - k5/5
    - 5阶解: z_{n+1} = y_n + 16k1/135 + 6656k3/12825 + 28561k4/56430 - 9k5/50 + 2k6/55
    - 误差估计: err = |z_{n+1} - y_{n+1}| / h
    - 步长调整: h_new = h * min(5, max(0.1, 0.9*(tol/err)^(1/5)))
"""

import numpy as np
from typing import Callable, Tuple, Optional
from utils import check_bounds, EPSILON_MACHINE

# =============================================================================
# Rucklidge ODE 参数
# =============================================================================
RUCKLIDGE_KAPPA = 2.0
RUCKLIDGE_LAMBDA = 1.7

# =============================================================================
# Arneodo ODE 参数
# =============================================================================
ARNEODO_ALPHA = -5.5
ARNEODO_BETA = 3.5
ARNEODO_DELTA = -1.0


def rucklidge_deriv(t: float, xyz: np.ndarray) -> np.ndarray:
    """
    Rucklidge 混沌系统的右端项。
    
    源自双对流模型（Rucklidge, 1992, JFM 237:209-229）:
        dx/dt = kappa * x - lambda * y - y*z
        dy/dt = x
        dz/dt = -z + y^2
    
    Parameters
    ----------
    t : float
        时间（自治系统，不显含t）
    xyz : np.ndarray, shape (3,)
        状态变量 [x, y, z]
    
    Returns
    -------
    np.ndarray, shape (3,)
        时间导数 [dx/dt, dy/dt, dz/dt]
    """
    x, y, z = xyz[0], xyz[1], xyz[2]
    dxdt = RUCKLIDGE_KAPPA * x - RUCKLIDGE_LAMBDA * y - y * z
    dydt = x
    dzdt = -z + y ** 2
    return np.array([dxdt, dydt, dzdt], dtype=float)


def arneodo_deriv(t: float, xyz: np.ndarray) -> np.ndarray:
    """
    Arneodo 混沌系统的右端项。
    
    源自渐近混沌模型（Arneodo et al., 1985, Physica D 14:327-347）:
        dx/dt = y
        dy/dt = z
        dz/dt = -alpha*x - beta*y - z + delta*x^3
    
    Parameters
    ----------
    t : float
        时间
    xyz : np.ndarray, shape (3,)
        状态变量 [x, y, z]
    
    Returns
    -------
    np.ndarray, shape (3,)
        时间导数
    """
    x, y, z = xyz[0], xyz[1], xyz[2]
    dxdt = y
    dydt = z
    dzdt = -ARNEODO_ALPHA * x - ARNEODO_BETA * y - z + ARNEODO_DELTA * (x ** 3)
    return np.array([dxdt, dydt, dzdt], dtype=float)


def rkf45_step(f: Callable[[float, np.ndarray], np.ndarray],
               t: float, y: np.ndarray, h: float,
               relerr: float = 1e-6, abserr: float = 1e-9) -> Tuple[np.ndarray, float, float, bool]:
    """
    执行单个 RKF45 自适应步长。
    
    核心数学：
    使用 embedded RK 方法，同时计算 4阶和5阶近似，
    通过两者的差异估计局部截断误差，据此接受/拒绝步长并调整下一步步长。
    
    Parameters
    ----------
    f : callable
        右端函数 f(t, y) -> dy/dt
    t : float
        当前时间
    y : np.ndarray
        当前状态
    h : float
        尝试的步长
    relerr : float
        相对误差容限
    abserr : float
        绝对误差容限
    
    Returns
    -------
    y_new : np.ndarray
        新状态（若接受则为5阶解，否则返回旧状态）
    t_new : float
        新时间
    h_new : float
        建议的下一步步长
    accepted : bool
        当前步是否被接受
    """
    y = np.asarray(y, dtype=float)
    neqn = y.size

    # RKF45 经典系数
    a2, a3, a4, a5, a6 = 1.0 / 4.0, 3.0 / 8.0, 12.0 / 13.0, 1.0, 0.5
    b21 = 1.0 / 4.0
    b31, b32 = 3.0 / 32.0, 9.0 / 32.0
    b41, b42, b43 = 1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0
    b51, b52, b53, b54 = 439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0
    b61, b62, b63, b64, b65 = -8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0, -11.0 / 40.0
    c1, c3, c4, c5 = 25.0 / 216.0, 1408.0 / 2565.0, 2197.0 / 4104.0, -1.0 / 5.0
    d1, d3, d4, d5, d6 = 16.0 / 135.0, 6656.0 / 12825.0, 28561.0 / 56430.0, -9.0 / 50.0, 2.0 / 55.0

    # 计算6个阶段导数
    k1 = h * f(t, y)
    k2 = h * f(t + a2 * h, y + b21 * k1)
    k3 = h * f(t + a3 * h, y + b31 * k1 + b32 * k2)
    k4 = h * f(t + a4 * h, y + b41 * k1 + b42 * k2 + b43 * k3)
    k5 = h * f(t + a5 * h, y + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4)
    k6 = h * f(t + a6 * h, y + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5)

    # 4阶与5阶解
    y4 = y + c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5
    y5 = y + d1 * k1 + d3 * k3 + d4 * k4 + d5 * k5 + d6 * k6

    # 误差估计
    scale = 2.0 / relerr if relerr > 0 else 1.0
    ae = scale * abserr
    err_max = 0.0
    for i in range(neqn):
        et = abs(y[i]) + abs(y5[i]) + ae
        if et <= 0.0:
            # 解消失，无法做纯相对误差测试
            return y, t, h * 0.5, False
        ee = abs((-2090.0 * k1[i]
                  + (21970.0 * k3[i] - 15048.0 * k4[i])
                  + (22528.0 * k2[i] - 27360.0 * k5[i])))
        err_local = abs(h) * ee * scale / 752400.0
        err_ratio = err_local / et
        err_max = max(err_max, err_ratio)

    esttol = err_max

    if esttol <= 1.0:
        # 步长接受
        s = 5.0 if esttol <= 0.0001889568 else 0.9 / (esttol ** 0.2)
        h_new = s * abs(h)
        h_new = min(h_new, 5.0 * abs(h))
        h_new = max(h_new, 26.0 * EPSILON_MACHINE * max(abs(t), abs(h)))
        return y5, t + h, h_new if h > 0 else -h_new, True
    else:
        # 步长拒绝，减小步长重试
        s = 0.9 / (esttol ** 0.2) if esttol < 59049.0 else 0.1
        h_new = s * abs(h)
        h_new = max(h_new, 26.0 * EPSILON_MACHINE * max(abs(t), abs(h)))
        return y, t, h_new if h > 0 else -h_new, False


def integrate_trajectory(f: Callable, y0: np.ndarray, t_span: Tuple[float, float],
                         relerr: float = 1e-6, abserr: float = 1e-9,
                         max_steps: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 RKF45 积分粒子轨迹。
    
    Parameters
    ----------
    f : callable
        右端函数
    y0 : np.ndarray
        初始状态
    t_span : tuple(float, float)
        积分区间 [t0, t1]
    relerr, abserr : float
        误差容限
    max_steps : int
        最大步数
    
    Returns
    -------
    t_array : np.ndarray
        时间点数组
    y_array : np.ndarray
        状态数组，shape (n_steps, neqn)
    """
    y0 = np.asarray(y0, dtype=float)
    t0, t1 = t_span
    if t0 >= t1:
        raise ValueError("t_span must satisfy t0 < t1")

    t = t0
    y = y0.copy()
    h = (t1 - t0) * 0.01  # 初始步长猜测
    h = max(h, 26.0 * EPSILON_MACHINE * max(abs(t0), abs(t1 - t0)))

    t_list = [t]
    y_list = [y.copy()]
    nfe = 0

    for _ in range(max_steps):
        if abs(t1 - t) <= 26.0 * EPSILON_MACHINE * abs(t):
            break

        # 调整最后一步以精确到达 t1
        if abs(t1 - t) <= abs(h):
            h = t1 - t

        y_new, t_new, h_new, accepted = rkf45_step(f, t, y, h, relerr, abserr)
        nfe += 6

        if accepted:
            t = t_new
            y = y_new
            t_list.append(t)
            y_list.append(y.copy())
            h = h_new if t < t1 else -abs(h_new)
        else:
            h = h_new

        if nfe > 3000 * y0.size:
            print("[WARNING] RKF45: too many function evaluations, stopping early.")
            break
    else:
        print("[WARNING] RKF45: max_steps reached, integration may be incomplete.")

    return np.array(t_list), np.array(y_list)


def compute_particle_load_field(particles: np.ndarray, domain: Tuple[float, float, float, float],
                                nx: int, ny: int) -> np.ndarray:
    """
    将粒子分布沉积到二维笛卡尔网格上，计算每个网格单元的粒子数密度（负载）。
    
    采用 Cloud-In-Cell (CIC) 沉积方案:
        rho(x,y) = sum_p q_p * W(x - x_p) * W(y - y_p)
    其中 W 为双线性权重函数（一阶B-spline）。
    
    Parameters
    ----------
    particles : np.ndarray, shape (n_particles, 2)
        粒子二维位置
    domain : tuple(xmin, xmax, ymin, ymax)
        计算域
    nx, ny : int
        网格分辨率
    
    Returns
    -------
    np.ndarray, shape (nx, ny)
        网格上的粒子负载密度
    """
    particles = np.asarray(particles, dtype=float)
    xmin, xmax, ymin, ymax = domain
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny

    rho = np.zeros((nx, ny), dtype=float)

    for p in range(particles.shape[0]):
        x, y = particles[p, 0], particles[p, 1]
        # 边界截断
        x = max(xmin + 1e-12, min(xmax - 1e-12, x))
        y = max(ymin + 1e-12, min(ymax - 1e-12, y))

        ix = int((x - xmin) / dx)
        iy = int((y - ymin) / dy)
        ix = min(ix, nx - 1)
        iy = min(iy, ny - 1)

        # CIC 双线性权重
        wx = (x - xmin) / dx - ix
        wy = (y - ymin) / dy - iy
        wx = max(0.0, min(1.0, wx))
        wy = max(0.0, min(1.0, wy))

        # 沉积到相邻4个网格点
        ixp1 = min(ix + 1, nx - 1)
        iyp1 = min(iy + 1, ny - 1)

        rho[ix, iy] += (1.0 - wx) * (1.0 - wy)
        rho[ixp1, iy] += wx * (1.0 - wy)
        rho[ix, iyp1] += (1.0 - wx) * wy
        rho[ixp1, iyp1] += wx * wy

    # 归一化为密度
    cell_volume = dx * dy
    if cell_volume > 0:
        rho /= cell_volume

    return rho
