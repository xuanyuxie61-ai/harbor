"""
time_integrator.py
==================
时间积分器模块 (源自 1090_eanderson15 旋转足球轨迹分析中的 RK4)。

科学背景:
  材料相变动力学、离子扩散等问题都需要求解 ODE 系统:
    dy/dt = f(t, y)

  本模块实现多种时间积分方法:
    1. 显式 RK4 (经典 4 阶 Runge-Kutta)
    2. 自适应 RK45 (Dormand-Prince)
    3. 隐式 Backward Euler (对刚性问题)
    4. BDF2 (2阶向后差分, 对抛物型 PDE)

  RK4 (源自 1090 足球轨迹积分):
    k1 = f(t_n, y_n)
    k2 = f(t_n + h/2, y_n + h*k1/2)
    k3 = f(t_n + h/2, y_n + h*k2/2)
    k4 = f(t_n + h, y_n + h*k3)
    y_{n+1} = y_n + h/6 * (k1 + 2*k2 + 2*k3 + k4)

  局部截断误差: O(h^5)
  全局截断误差: O(h^4)

  Dormand-Prince RK45 (自适应):
    使用 6 阶和 5 阶两个估计的差作为误差估计:
    err = |y_5 - y_4*| ~ C * h^5
    步长控制: h_new = h * min(max_fac, max(min_fac, fac * (tol/err)^(1/5)))

  Backward Euler (隐式, A-稳定):
    y_{n+1} = y_n + h * f(t_{n+1}, y_{n+1})
    需解非线性方程 (Newton 迭代):
      G(y) = y - y_n - h*f(t_{n+1}, y) = 0
      J_G = I - h * df/dy

  BDF2 (2阶向后差分):
    (3*y_{n+1} - 4*y_n + y_{n-1}) / (2*h) = f(t_{n+1}, y_{n+1})
    A-稳定, L-稳定, 对刚性问题优秀

CFL 条件:
  对流: dt <= h / |v| (CFL number <= 1)
  扩散: dt <= h^2 / (2*d*alpha) (d: 空间维度)
"""

import numpy as np
from material_constants import SMALL_NUMBER


def rk4_step(f, t, y, h):
    """
    经典 RK4 单步 (源自 1090 足球轨迹的 RK4 积分)。

    应用于离子输运的相场动力学:
      d(phi)/dt = -M * delta F / delta phi

    参数:
        f: 右端函数 f(t, y) -> dydt
        t: 当前时间
        y: 当前状态 [N]
        h: 时间步长

    返回: y_{n+1}
    """
    k1 = f(t, y)
    k2 = f(t + 0.5*h, y + 0.5*h*k1)
    k3 = f(t + 0.5*h, y + 0.5*h*k2)
    k4 = f(t + h, y + h*k3)
    return y + (h / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)


def rk4_integrate(f, t_span, y0, n_steps):
    """
    RK4 积分整个时间区间。

    返回:
        t_array: [n_steps+1]
        y_array: [n_steps+1, len(y0)]
    """
    t0, tf = t_span
    h = (tf - t0) / n_steps
    t_arr = np.linspace(t0, tf, n_steps + 1)
    y_arr = np.zeros((n_steps + 1, len(y0)))
    y_arr[0] = y0.copy()
    for i in range(n_steps):
        y_arr[i+1] = rk4_step(f, t_arr[i], y_arr[i], h)
    return t_arr, y_arr


def rkf45_step(f, t, y, h, tol=1e-6):
    """
    Runge-Kutta-Fehlberg 45 自适应步 (简化版)。

    使用 RKF45 系数:
    4阶估计和5阶估计之差作为误差控制。

    返回:
        y_new, h_new, err, accepted
    """
    # RKF45 Butcher 表 (简化使用嵌入 RK4(3))
    k1 = h * f(t, y)
    k2 = h * f(t + h/4, y + k1/4)
    k3 = h * f(t + 3*h/8, y + 3*k1/32 + 9*k2/32)
    k4 = h * f(t + 12*h/13, y + 1932*k1/2197 - 7200*k2/2197 + 7296*k3/2197)
    k5 = h * f(t + h, y + 439*k1/216 - 8*k2 + 3680*k3/513 - 845*k4/4104)
    k6 = h * f(t + h/2, y - 8*k1/27 + 2*k2 - 3544*k3/2565 + 1859*k4/4104 - 11*k5/40)

    # 5阶估计
    y5 = y + 16*k1/135 + 6656*k3/12825 + 28561*k4/56430 - 9*k5/50 + 2*k6/55
    # 4阶估计
    y4 = y + 25*k1/216 + 1408*k3/2565 + 2197*k4/4104 - k5/5

    err_vec = y5 - y4
    err = np.linalg.norm(err_vec) / max(np.linalg.norm(y5), SMALL_NUMBER)

    if err <= tol:
        accepted = True
        if err > SMALL_NUMBER:
            h_new = 0.9 * h * (tol / err) ** 0.2
        else:
            h_new = h * 2.0
    else:
        accepted = False
        h_new = 0.9 * h * (tol / err) ** 0.25

    h_new = np.clip(h_new, 0.1 * h, 5.0 * h)
    return y5, h_new, err, accepted


def adaptive_integrate(f, t_span, y0, tol=1e-6, max_steps=10000, h_init=None):
    """
    自适应 RKF45 积分。

    返回:
        t_array, y_array, n_steps_taken, n_rejected
    """
    t0, tf = t_span
    t = t0
    y = np.asarray(y0, dtype=np.float64).copy()
    h = h_init if h_init is not None else (tf - t0) / 100.0

    t_list = [t]
    y_list = [y.copy()]
    n_accepted = 0
    n_rejected = 0

    while t < tf and n_accepted < max_steps:
        h = min(h, tf - t)
        y_new, h_new, err, accepted = rkf45_step(f, t, y, h, tol)
        if accepted:
            t += h
            y = y_new
            t_list.append(t)
            y_list.append(y.copy())
            n_accepted += 1
        else:
            n_rejected += 1
        h = h_new

    return np.array(t_list), np.array(y_list), n_accepted, n_rejected


def backward_euler_step(f, jac, t_new, y_n, h, newton_tol=1e-10, newton_max=20):
    """
    Backward Euler 隐式步 (Newton 迭代求解)。

    求解: G(y) = y - y_n - h * f(t_new, y) = 0
    Newton: y^{(k+1)} = y^{(k)} - J_G^{-1} * G(y^{(k)})
    J_G = I - h * J_f
    """
    y = y_n.copy()
    I = np.eye(len(y))
    for it in range(newton_max):
        G = y - y_n - h * f(t_new, y)
        J_f = jac(t_new, y)
        J_G = I - h * J_f
        delta = np.linalg.solve(J_G, -G)
        y = y + delta
        if np.linalg.norm(delta) < newton_tol:
            break
    return y


def bdf2_step(f, jac, t_new, y_n, y_nm1, h, newton_tol=1e-10, newton_max=20):
    """
    BDF2 步 (2阶向后差分):

    (3*y_{n+1} - 4*y_n + y_{n-1}) / (2*h) = f(t_{n+1}, y_{n+1})

    改写: G(y) = 3*y - 4*y_n + y_{n-1} - 2*h*f(t_{n+1}, y) = 0
    J_G = 3*I - 2*h*J_f
    """
    y = y_n.copy()
    I = np.eye(len(y))
    for it in range(newton_max):
        G = 3.0*y - 4.0*y_n + y_nm1 - 2.0*h * f(t_new, y)
        J_f = jac(t_new, y)
        J_G = 3.0*I - 2.0*h * J_f
        delta = np.linalg.solve(J_G, -G)
        y = y + delta
        if np.linalg.norm(delta) < newton_tol:
            break
    return y


def cfl_timestep_diffusion(alpha, h, dim=1, safety=0.4):
    """
    扩散 CFL 条件:
      dt <= safety * h^2 / (2^dim * alpha)
    """
    return safety * h**2 / (2**dim * max(alpha, SMALL_NUMBER))


def cfl_timestep_advection(v_max, h, safety=0.9):
    """
    对流 CFL 条件:
      dt <= safety * h / |v|_max
    """
    return safety * h / max(v_max, SMALL_NUMBER)


def phase_field_rhs(phi, M, kappa, f_prime, laplacian_op):
    """
    相场方程 (Allen-Cahn) 右端:

    dphi/dt = -M * (f'(phi) - kappa * laplacian(phi))

    用于模拟材料相分离、晶粒生长等。

    参数:
        phi: 相场 [N]
        M: 迁移率
        kappa: 梯度能系数
        f_prime: 双阱势导数函数
        laplacian_op: 离散 Laplace 算子 [N, N]
    """
    fp = f_prime(phi)
    lap_phi = laplacian_op.dot(phi)
    mu = fp - kappa * lap_phi
    return -M * mu


def double_well_potential(phi, A=1.0):
    """
    双阱势: f(phi) = A * phi^2 * (1-phi)^2
    导数: f'(phi) = A * (4*phi^3 - 6*phi^2 + 2*phi)
    """
    return A * phi**2 * (1.0 - phi)**2


def double_well_derivative(phi, A=1.0):
    """双阱势导数"""
    return A * (4.0*phi**3 - 6.0*phi**2 + 2.0*phi)


def spinodal_decomposition_energy(phi, kappa_grad, f_double_well):
    """
    Cahn-Hilliard 总自由能:

    F[phi] = integral(f(phi) + kappa/2 * |grad phi|^2 dx)

    离散:
      F ≈ sum_i [f(phi_i) + kappa/2 * ((phi_{i+1}-phi_i)/h)^2] * h
    """
    return np.sum(f_double_well(phi))
