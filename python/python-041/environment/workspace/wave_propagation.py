"""
 wave_propagation.py
 
 融合种子项目:
   - 1138_spring_double_ode: 双弹簧耦合系统 ODE
   - 216_control_bio_homework: RK4 时间积分与最优控制思想
   - 1160_standing_wave_exact: 一维波动方程精确解与残差计算
 
 科学应用:
   地震波在地下介质中的传播可以用波动方程描述:
     rho * d^2u/dt^2 = nabla * (mu * nabla u) + f
   
   在粘弹性介质中，应力-应变关系包含记忆效应，可模型化为耦合弹簧-阻尼系统。
   双弹簧系统对应于标准线性固体（Standard Linear Solid, SLS）模型:
     sigma + tau_sigma * dsigma/dt = M_R * (epsilon + tau_epsilon * depsilon/dt)
   
   本项目使用:
   1. 双弹簧 ODE 模型模拟粘弹性波传播的衰减和频散特性
   2. RK4 方法进行时间域正演模拟
   3. 驻波精确解验证数值算法的正确性
"""

import numpy as np


def spring_double_parameters(m1=3.0, m2=5.0, k1=1.0, k2=10.0,
                              t0=0.0, y0=None, tstop=50.0):
    """
    返回双弹簧耦合系统的物理参数。
    
    物理模型:
      弹簧1: 质量 m1, 刚度 k1, 连接固定支撑
      弹簧2: 质量 m2, 刚度 k2, 悬挂于弹簧1下方
    
    此系统可类比于粘弹性介质的本构关系：上层弹簧代表弹性响应，
    下层弹簧-质量系统代表粘滞弛豫机制。
    
    Parameters
    ----------
    m1, m2 : float
        两个质量块的质量。
    k1, k2 : float
        两个弹簧的刚度系数。
    t0 : float
        初始时间。
    y0 : ndarray, optional
        初始状态 [u1, v1, u2, v2]。
    tstop : float
        终止时间。
    
    Returns
    -------
    params : dict
        参数字典。
    """
    if y0 is None:
        y0 = np.array([0.0, 1.0, 0.0, 0.0])
    return {
        'm1': m1, 'm2': m2, 'k1': k1, 'k2': k2,
        't0': t0, 'y0': y0, 'tstop': tstop
    }


def spring_double_deriv(t, y, params):
    """
    双弹簧系统的右端项（时间导数）。
    
    状态变量: y = [u1, v1, u2, v2]
      u1: 质量1的位移
      v1: 质量1的速度
      u2: 质量2的位移
      v2: 质量2的速度
    
    运动方程:
      du1/dt = v1
      dv1/dt = (-k1*u1 + k2*(u2 - u1)) / m1
      du2/dt = v2
      dv2/dt = (-k2*(u2 - u1)) / m2
    
    该耦合 ODE 系统的特征频率为:
      omega_{1,2}^2 = 0.5 * [ (k1+k2)/m1 + k2/m2 ] 
                      +/- 0.5 * sqrt( [ (k1+k2)/m1 + k2/m2 ]^2 - 4*k1*k2/(m1*m2) )
    
    Parameters
    ----------
    t : float
        当前时间。
    y : ndarray, shape (4,)
        当前状态。
    params : dict
        系统参数。
    
    Returns
    -------
    dydt : ndarray, shape (4,)
        状态导数。
    """
    m1 = params['m1']
    m2 = params['m2']
    k1 = params['k1']
    k2 = params['k2']
    u1, v1, u2, v2 = y
    du1dt = v1
    dv1dt = (-k1 * u1 + k2 * (u2 - u1)) / m1
    du2dt = v2
    dv2dt = (-k2 * (u2 - u1)) / m2
    return np.array([du1dt, dv1dt, du2dt, dv2dt])


def rk4_integrate(dydt, tspan, y0, n, args=()):
    """
    四阶 Runge-Kutta (RK4) 时间积分。
    
    对于 ODE: dy/dt = f(t, y)
    RK4 格式:
      k1 = f(t_n, y_n)
      k2 = f(t_n + dt/2, y_n + dt*k1/2)
      k3 = f(t_n + dt/2, y_n + dt*k2/2)
      k4 = f(t_n + dt, y_n + dt*k3)
      y_{n+1} = y_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    局部截断误差: O(dt^5)
    全局误差: O(dt^4)
    
    Parameters
    ----------
    dydt : callable
        右端项函数 f(t, y, *args)。
    tspan : tuple
        (t0, tstop)。
    y0 : ndarray
        初始条件。
    n : int
        时间步数。
    args : tuple
        传递给 dydt 的额外参数。
    
    Returns
    -------
    t : ndarray, shape (n+1,)
        时间网格。
    y : ndarray, shape (n+1, len(y0))
        解。
    """
    y0 = np.asarray(y0, dtype=float).flatten()
    m = len(y0)
    t0, tstop = tspan
    dt = (tstop - t0) / n
    t = np.zeros(n + 1)
    y = np.zeros((n + 1, m))
    t[0] = t0
    y[0, :] = y0
    for i in range(n):
        k1 = dydt(t[i], y[i, :], *args)
        k2 = dydt(t[i] + dt / 2.0, y[i, :] + dt * k1 / 2.0, *args)
        k3 = dydt(t[i] + dt / 2.0, y[i, :] + dt * k2 / 2.0, *args)
        k4 = dydt(t[i] + dt, y[i, :] + dt * k3, *args)
        t[i + 1] = t[i] + dt
        y[i + 1, :] = y[i, :] + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return t, y


def standing_wave_exact(x, t, c=0.2):
    """
    一维波动方程的驻波精确解。
    
    方程:
      d^2u/dt^2 = c^2 * d^2u/dx^2
    
    精确解:
      u(x,t) = sin(x) * cos(c*t)
    
    各阶导数:
      ut  = -c * sin(x) * sin(c*t)
      utt = -c^2 * sin(x) * cos(c*t)
      ux  =  cos(x) * cos(c*t)
      uxx = -sin(x) * cos(c*t)
    
    Parameters
    ----------
    x, t : float or ndarray
        空间和时间坐标。
    c : float
        波速。
    
    Returns
    -------
    u, ut, utt, ux, uxx : float or ndarray
        解及其各阶导数。
    """
    u = np.sin(x) * np.cos(c * t)
    ut = -c * np.sin(x) * np.sin(c * t)
    utt = -c ** 2 * np.sin(x) * np.cos(c * t)
    ux = np.cos(x) * np.cos(c * t)
    uxx = -np.sin(x) * np.cos(c * t)
    return u, ut, utt, ux, uxx


def standing_wave_residual(x, t, c=0.2):
    """
    计算波动方程的残差:
      R = utt - c^2 * uxx
    
    对于精确解，R 理论上为零，可用于验证数值实现。
    
    Parameters
    ----------
    x, t : float or ndarray
    c : float
    
    Returns
    -------
    residual : float or ndarray
        残差值。
    """
    _, _, utt, _, uxx = standing_wave_exact(x, t, c)
    return utt - c ** 2 * uxx


def seismic_wave_rk4_1d(nx, dx, nt, dt, c, source_time_fn, source_pos,
                         boundary='absorbing'):
    """
    使用 RK4 方法求解一维声波方程的时间域正演模拟。
    
    将二阶波动方程转化为一阶系统:
      du/dt = v
      dv/dt = c^2 * d^2u/dx^2 + f(t)
    
    状态向量 y = [u_1, ..., u_nx, v_1, ..., v_nx]。
    
    Parameters
    ----------
    nx : int
        空间网格点数。
    dx : float
        空间步长。
    nt : int
        时间步数。
    dt : float
        时间步长。
    c : ndarray, shape (nx,)
        速度模型。
    source_time_fn : callable
        震源时间函数，输入时间，返回振幅。
    source_pos : int
        震源位置索引。
    boundary : str
        边界条件类型: 'absorbing' 或 'reflecting'。
    
    Returns
    -------
    u_history : ndarray, shape (nt+1, nx)
        每个时间步的波场快照。
    t : ndarray, shape (nt+1,)
        时间网格。
    """
    c = np.asarray(c, dtype=float)
    # 定义一阶系统的右端项
    def deriv(t, y):
        u = y[:nx]
        v = y[nx:]
        dudt = v.copy()
        dvdt = np.zeros(nx)
        # TODO: 实现空间二阶导数（中心差分）和边界条件
        # 科学知识：一维波动方程 d²u/dt² = c²(x)·d²u/dx² 的空间离散
        # 要求：
        #   1. 内部点使用中心差分离散二阶空间导数
        #   2. absorbing 边界使用一阶吸收边界条件 du/dt ± c·du/dx = 0
        #   3. reflecting 边界使用 Dirichlet 条件
        pass
        # 震源项
        if 0 <= source_pos < nx:
            dvdt[source_pos] += source_time_fn(t)
        return np.concatenate([dudt, dvdt])
    
    y0 = np.zeros(2 * nx)
    t, y = rk4_integrate(deriv, (0.0, nt * dt), y0, nt)
    u_history = y[:, :nx]
    return u_history, t


def test_standing_wave_convergence():
    """
    测试 RK4 波动方程求解器对驻波精确解的收敛性。
    
    Returns
    -------
    errors : list
        不同网格分辨率下的 L2 误差。
    dxs : list
        对应的空间步长。
    """
    c = 0.5
    t_final = 2.0 * np.pi / c
    errors = []
    dxs = []
    for nx in [41, 81, 161]:
        dx = 2.0 * np.pi / (nx - 1)
        x = np.linspace(0.0, 2.0 * np.pi, nx)
        # CFL 条件: dt <= dx / c_max
        dt = 0.5 * dx / c
        nt = int(np.ceil(t_final / dt))
        dt = t_final / nt
        c_arr = np.full(nx, c)
        # 零震源（自由振动）
        def source_fn(t):
            return 0.0
        u_hist, t = seismic_wave_rk4_1d(nx, dx, nt, dt, c_arr, source_fn, 0,
                                         boundary='reflecting')
        u_num = u_hist[-1, :]
        u_exact, _, _, _, _ = standing_wave_exact(x, t_final, c)
        # 初始条件匹配 sin(x)
        u_exact_init = np.sin(x)
        # 但由于初始条件设置，数值解可能有相位偏移，这里只检验残差行为
        err = np.sqrt(np.mean((u_num - u_exact_init * np.cos(c * t_final)) ** 2))
        errors.append(err)
        dxs.append(dx)
    return errors, dxs
