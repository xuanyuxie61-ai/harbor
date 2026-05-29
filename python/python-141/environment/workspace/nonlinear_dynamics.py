"""
非线性动力学与混沌分析模块
============================
融合种子项目:
  - 488_grazing_ode : 非线性ODE参数动力学
  - 318_dragon_chaos : 迭代函数系统与分形

在金融工程中，本模块用于：
1. 分析随机波动率参数的均值回归非线性动力学（受放牧ODE启发）
2. 研究资产收益与波动率的多重分形特征（受混沌IFS启发）
3. 求解Heston模型特征ODE（Riccati方程）

数学背景:
---------
Heston特征函数中的Riccati方程:
    dD/dτ = -½ u(u+i) + (κ - iρσu) D + ½ σ² D²
    dA/dτ = κθ D
    初值: D(0) = A(0) = 0

其中 u 为傅里叶变量，τ = T - t 为剩余到期时间。
解析解:
    D(u,τ) = (κ - iρσu - d) / σ² · (1 - e^{-dτ}) / (1 - g·e^{-dτ})
    d = √((iρσu - κ)² + σ²(u² + iu))
    g = (κ - iρσu - d) / (κ - iρσu + d)
    A(u,τ) = r·u·i·τ + (κθ/σ²)[(κ - iρσu - d)τ - 2 ln((1 - g·e^{-dτ})/(1 - g))]

放牧ODE（生态模型）映射到金融:
    将捕食者-食饵动力学映射为波动率-交易量耦合系统:
        dv/dt = r1·v·(1 - v/K) - c1·q·(1 - exp(-d1·v))
        dq/dt = -a·q + c2·q·(1 - exp(-d2·v))
    其中 v 为波动率，q 为订单流强度。

混沌IFS（迭代函数系统）:
    Barnsley蕨类/Dragon曲线生成映射到收益分布的多重分形测度:
        x_{n+1} = A_j · x_n + b_j   以概率 p_j
"""

import numpy as np
import cmath
from math import sqrt, log, exp, cos, sin, pi


# ========================================================================
# 488_grazing_ode : 非线性参数动力学
# ========================================================================

def grazing_parameters():
    """
    返回放牧生态系统的默认参数，映射到金融波动率-订单流动力学。
    """
    return {
        'a': 1.1,      # 订单流衰减率
        'c1': 1.2,     # 最大波动率消耗率
        'c2': 1.5,     # 订单流恢复率
        'd1': 0.001,   # 低波动率下的订单流敏感度
        'd2': 0.001,   # 低波动率下的市场深度敏感度
        'k': 3000.0,   # 最大未抑制波动率
        'r1': 0.8,     # 波动率固有增长率
        't0': 0.0,
        'y0': np.array([3000.0, 5.0]),
        'tstop': 100.0
    }


def volatility_orderflow_deriv(t, y, params):
    """
    波动率-订单流耦合ODE的右端项。

    方程组:
        du/dt = r1·u·(1 - u/k) - c1·v·(1 - exp(-d1·u))
        dv/dt = -a·v + c2·v·(1 - exp(-d2·u))

    参数:
    ------
    y = [u, v] : u为波动率水平，v为订单流强度
    """
    u, v = y[0], y[1]
    a = params['a']
    c1 = params['c1']
    c2 = params['c2']
    d1 = params['d1']
    d2 = params['d2']
    k = params['k']
    r1 = params['r1']

    dudt = r1 * u * (1.0 - u / k) - c1 * v * (1.0 - exp(-d1 * u))
    dvdt = -a * v + c2 * v * (1.0 - exp(-d2 * u))
    return np.array([dudt, dvdt], dtype=np.float64)


def rk4_integrate(deriv_func, y0, t_span, h=0.01, args=()):
    """
    四阶Runge-Kutta积分器。

    RK4公式:
        k1 = h·f(t_n, y_n)
        k2 = h·f(t_n + h/2, y_n + k1/2)
        k3 = h·f(t_n + h/2, y_n + k2/2)
        k4 = h·f(t_n + h, y_n + k3)
        y_{n+1} = y_n + (k1 + 2k2 + 2k3 + k4)/6
    """
    y0 = np.asarray(y0, dtype=np.float64)
    t0, tf = t_span
    if t0 >= tf:
        raise ValueError("t0必须小于tf")
    if h <= 0:
        raise ValueError("步长h必须为正")

    steps = int(np.ceil((tf - t0) / h))
    h = (tf - t0) / steps
    trajectory = np.zeros((steps + 1, len(y0)), dtype=np.float64)
    times = np.zeros(steps + 1, dtype=np.float64)
    trajectory[0] = y0
    times[0] = t0
    y = y0.copy()
    t = t0

    for i in range(steps):
        k1 = h * deriv_func(t, y, *args)
        k2 = h * deriv_func(t + 0.5 * h, y + 0.5 * k1, *args)
        k3 = h * deriv_func(t + 0.5 * h, y + 0.5 * k2, *args)
        k4 = h * deriv_func(t + h, y + k3, *args)
        y = y + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        t += h
        trajectory[i + 1] = y
        times[i + 1] = t

    return times, trajectory


def feller_dynamics_analysis(kappa, theta, sigma):
    """
    分析Heston模型Feller条件的非线性动力学稳定性。

    Feller条件: 2κθ ≥ σ²
    若违反，则波动率过程可在零边界处被吸收，导致数值困难。
    """
    feller_ratio = 2.0 * kappa * theta / (sigma * sigma)
    stable = feller_ratio >= 1.0
    absorption_prob = np.exp(-2.0 * kappa * theta / (sigma * sigma)) if sigma > 0 else 0.0
    return {
        'feller_ratio': feller_ratio,
        'feller_satisfied': stable,
        'absorption_probability_estimate': absorption_prob,
        'boundary_classification': 'entrance' if stable else 'regular'
    }


# ========================================================================
# 318_dragon_chaos : 迭代函数系统与分形
# ========================================================================

def iterated_function_system(n_iter, maps, probs, x0=None, dim=2):
    """
    通用迭代函数系统(IFS)。

    参数:
    ------
    n_iter : int, 迭代次数
    maps   : list of callable, 每个map接受x返回新x
    probs  : list of float, 每个map的选择概率（自动归一化）
    x0     : ndarray, 初始点，默认随机
    dim    : int, 空间维度

    返回:
    ------
    ndarray, 形状 (n_iter, dim)，迭代轨迹
    """
    if x0 is None:
        x0 = np.random.rand(dim)
    x0 = np.asarray(x0, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / np.sum(probs)

    trajectory = np.zeros((n_iter, dim), dtype=np.float64)
    x = x0.copy()
    rng = np.random.default_rng()

    for i in range(n_iter):
        j = rng.choice(len(maps), p=probs)
        x = maps[j](x)
        trajectory[i] = x

    return trajectory


def dragon_curve_ifs(n_iter=5000):
    """
    Levy Dragon曲线IFS（金融分形吸引子示例）。

    映射:
        A = [ -0.5,  0.5 ]
            [ -0.5, -0.5 ]
        b1 = [0, 0]^T,  b2 = [1, 0]^T
        x_{n+1} = A x_n + b_j  (j=1或2，各50%概率)
    """
    A = np.array([[-0.5, 0.5], [-0.5, -0.5]], dtype=np.float64)
    b1 = np.array([0.0, 0.0], dtype=np.float64)
    b2 = np.array([1.0, 0.0], dtype=np.float64)

    def map1(x):
        return A @ x + b1

    def map2(x):
        return A @ x + b2

    return iterated_function_system(n_iter, [map1, map2], [0.5, 0.5], dim=2)


def multifractal_spectrum(trajectory, q_values=None):
    """
    计算轨迹的多重分形谱（简化版）。

    使用盒计数法估计广义维数 D_q:
        D_q = lim_{ε→0} (1/(q-1)) · log(Σ_i p_i^q) / log(ε)

    参数:
    ------
    trajectory : ndarray, 形状 (N, d)
    q_values   : array, 矩阶数，默认 [-5, -2, -1, 0, 1, 2, 5]

    返回:
    ------
    dict, {q: D_q_estimate}
    """
    if q_values is None:
        q_values = [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0]
    traj = np.asarray(trajectory, dtype=np.float64)
    if traj.ndim != 2:
        raise ValueError("trajectory必须为二维数组")
    N, d = traj.shape

    # 归一化到单位超立方体
    mins = np.min(traj, axis=0)
    maxs = np.max(traj, axis=0)
    ranges = maxs - mins
    ranges[ranges < 1e-12] = 1.0
    traj_norm = (traj - mins) / ranges

    # 使用不同尺度的盒计数
    box_counts = [4, 8, 16, 32]
    results = {}
    for q in q_values:
        log_eps = []
        log_moments = []
        for boxes in box_counts:
            eps = 1.0 / boxes
            # 将点分配到盒中
            indices = np.floor(traj_norm * boxes).astype(np.int64)
            indices = np.clip(indices, 0, boxes - 1)
            # 计算每个盒的概率
            flat_idx = np.ravel_multi_index(indices.T, [boxes] * d)
            unique, counts = np.unique(flat_idx, return_counts=True)
            p = counts / N
            if q == 1.0:
                # Shannon熵
                moment = -np.sum(p * np.log(p + 1e-30))
            elif q == 0.0:
                # 盒数
                moment = len(unique)
            else:
                moment = np.sum(p ** q)
            log_eps.append(log(eps))
            if q == 0.0:
                log_moments.append(log(moment + 1e-30))
            elif q == 1.0:
                log_moments.append(moment)
            else:
                log_moments.append(log(moment + 1e-30))

        # 线性回归估计D_q
        if len(log_eps) >= 2:
            log_eps = np.array(log_eps)
            log_moments = np.array(log_moments)
            if q == 1.0:
                # D_1 = slope of moment vs log(eps)
                slope = np.polyfit(log_eps, log_moments, 1)[0]
                D_q = slope
            elif q == 0.0:
                slope = np.polyfit(log_eps, log_moments, 1)[0]
                D_q = -slope
            else:
                slope = np.polyfit(log_eps, log_moments, 1)[0]
                D_q = slope / (q - 1.0)
            results[q] = D_q
        else:
            results[q] = 0.0

    return results


# ========================================================================
# Heston Riccati ODE 求解
# ========================================================================

def heston_riccati_solution(u, tau, kappa, theta, sigma, rho, r):
    """
    解析求解Heston模型的Riccati方程。

    参数:
    ------
    u   : complex, 傅里叶变量
    tau : float, 剩余到期时间
    kappa, theta, sigma, rho, r : Heston模型参数

    返回:
    ------
    A, D : complex, 特征函数指数项
    """
    if tau < 0:
        raise ValueError("tau必须非负")
    if abs(sigma) < 1e-12:
        raise ValueError("sigma不能为零")

    # TODO: 实现Heston模型Riccati方程的解析解
    # 需要计算:
    #   d = sqrt((rho*sigma*u*i - kappa)^2 + sigma^2*(u*i + u^2))
    #   g = (kappa - rho*sigma*u*i - d) / (kappa - rho*sigma*u*i + d)
    #   D(u,tau) = (kappa - rho*sigma*u*i - d) / sigma^2 * (1 - exp(-d*tau)) / (1 - g*exp(-d*tau))
    #   A(u,tau) = r*u*i*tau + (kappa*theta/sigma^2) * [(kappa - rho*sigma*u*i - d)*tau
    #              - 2*ln((1 - g*exp(-d*tau))/(1 - g))]
    # 注意处理退化情况 |1 - g*exp(-d*tau)| < epsilon 或 |1 - g| < epsilon
    raise NotImplementedError("Hole_1: 需要实现Riccati方程解析解")


def heston_characteristic_function(u, S0, v0, T, r, kappa, theta, sigma, rho):
    """
    Heston模型对数价格的特征函数。

        φ(u) = exp( A(u,T) + D(u,T)·v0 + iu·ln(S0) )
    """
    A, D = heston_riccati_solution(u, T, kappa, theta, sigma, rho, r)
    phi = np.exp(A + D * v0 + 1j * u * log(S0))
    return phi
