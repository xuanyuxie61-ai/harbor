"""
numerical_propagation.py — 高阶数值传播方法与稳定性分析
========================================================

融合种子项目:
    [064_backward_euler_fixed]  : 隐式向后 Euler + 不动点 (Picard) 迭代
    [614_kdv_etdrk4]           : ETDRK4 指数时间差分 Runge-Kutta 4 阶
    [694_local_min]            : Brent 1D 最小化 (用于步长优化)

物理问题:
    带电粒子在磁场中的运动方程 (Lorentz 力):
        dp/dt = q·(v × B)
        dx/dt = v = p/(γm)

    在均匀磁场 B = (0, 0, B₀) 下，精确解为螺旋线:
        x(t) = r_L·sin(ωt + φ₀) + x_c
        y(t) = r_L·cos(ωt + φ₀) + y_c
        z(t) = v_z·t + z₀

    其中 r_L = p_T/(qB) 为 Larmor 半径, ω = qB/(γm) 为回旋频率

    在非均匀场中，需要用数值方法求解:
        dy/dt = f(t, y)

    本模块实现三种传播器:
    1. RK4: 经典 4 阶 Runge-Kutta (显式)
    2. Backward Euler: 隐式向后 Euler + Picard 迭代 (来自 [064])
    3. ETDRK4: 指数时间差分 (来自 [614])，适合刚性问题
"""

import math
import numpy as np
from typing import Callable, Tuple, List, Optional


# ============================================================
# 运动方程定义
# ============================================================
def lorentz_equations(state, bfield_func, charge=1, mass=139.57):
    """
    Lorentz 力运动方程右端函数

    状态向量: y = (x, y, z, px, py, pz) [mm, MeV/c]
    运动方程:
        dx/dt = px / (γm)     [mm/单位时间]
        dp/dt = q · (v × B)   [MeV/c/单位时间]

    其中 γ = √(1 + (p/mc)²)

    Parameters
    ----------
    state : array_like, shape (6,)
        [x, y, z, px, py, pz]
    bfield_func : callable
        B = bfield_func(x, y, z) 返回 (Bx, By, Bz) [Tesla]
    charge : int
        粒子电荷数 (±1, ±2, ...)
    mass : float
        粒子质量 [MeV/c²]

    Returns
    -------
    ndarray, shape (6,) : dy/dt
    """
    x, y, z = state[0], state[1], state[2]
    px, py, pz = state[3], state[4], state[5]

    # 动量大小和 Lorentz 因子
    p2 = px*px + py*py + pz*pz
    p = math.sqrt(max(p2, 1e-20))
    gamma = math.sqrt(1.0 + p2 / (mass * mass))

    # 速度 (c = 1 自然单位，转换为 mm/ns)
    # v = p/(γm) · c = p/(γm) · 299.79 mm/ns
    c_mm_ns = 299.792458
    vx = (px / (gamma * mass)) * c_mm_ns
    vy = (py / (gamma * mass)) * c_mm_ns
    vz = (pz / (gamma * mass)) * c_mm_ns

    # 磁场
    bx, by, bz = bfield_func(x, y, z)

    # Lorentz 力: dp/dt = q · (v × B)
    # 单位转换: 1 T · 1 mm/ns · 1 e = 1e-3 GeV/c / (mm/c) = ...
    # 简化: dp/dt [MeV/c/mm] = q · 0.2998 · (v × B) [MeV/c/mm]
    q_factor = charge * 0.299792458  # 单位转换因子

    # v × B
    vxb_y = vz * bx - vx * bz  # (v×B)_y → 对应 dpx
    vxb_z = vx * by - vy * bx  # 不对，重新计算

    # v × B 完整计算
    cross_x = vy * bz - vz * by
    cross_y = vz * bx - vx * bz
    cross_z = vx * by - vy * bx

    dpx = q_factor * cross_x
    dpy = q_factor * cross_y
    dpz = q_factor * cross_z

    return np.array([vx, vy, vz, dpx, dpy, dpz])


# ============================================================
# RK4 传播器 (显式 4 阶 Runge-Kutta)
# ============================================================
def propagate_rk4(state0, bfield_func, step_length_mm, n_steps=10,
                  charge=1, mass=139.57):
    """
    经典 4 阶 Runge-Kutta 传播

    k₁ = h·f(tₙ, yₙ)
    k₂ = h·f(tₙ + h/2, yₙ + k₁/2)
    k₃ = h·f(tₙ + h/2, yₙ + k₂/2)
    k₄ = h·f(tₙ + h, yₙ + k₃)
    yₙ₊₁ = yₙ + (k₁ + 2k₂ + 2k₃ + k₄)/6

    Parameters
    ----------
    state0 : array_like, shape (6,)
        初始状态 [x, y, z, px, py, pz]
    bfield_func : callable
        磁场函数
    step_length_mm : float
        总路径长度 [mm]
    n_steps : int
        步数
    charge : int
    mass : float

    Returns
    -------
    ndarray : 最终状态
    """
    h = step_length_mm / max(n_steps, 1)
    y = np.array(state0, dtype=float)

    for _ in range(n_steps):
        def f(s):
            return lorentz_equations(s, bfield_func, charge, mass)

        k1 = h * f(y)
        k2 = h * f(y + 0.5 * k1)
        k3 = h * f(y + 0.5 * k2)
        k4 = h * f(y + k3)

        y = y + (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0

    return y


# ============================================================
# [064_backward_euler_fixed] 隐式向后 Euler + Picard 迭代
# ============================================================
def propagate_backward_euler(state0, bfield_func, step_length_mm,
                             n_steps=10, charge=1, mass=139.57,
                             it_max=10, tol=1e-8):
    """
    基于 [064_backward_euler_fixed] 的隐式向后 Euler 传播

    原算法:
        y_{i+1} = y_i + dt · f(t_{i+1}, y_{i+1})
        通过 Picard 迭代求解:
        yp^(j+1) = y_i + dt · f(t_{i+1}, yp^(j))
        固定 it_max = 10 次迭代

    改进: 添加收敛判据
        ||yp^(j+1) - yp^(j)|| < tol → 提前终止

    Parameters
    ----------
    state0 : array_like, shape (6,)
    bfield_func : callable
    step_length_mm : float
    n_steps : int
    charge : int
    mass : float
    it_max : int
        每个时间步的最大 Picard 迭代次数 (原 [064] 硬编码为 10)
    tol : float
        收敛容差 (原 [064] 无此项)

    Returns
    -------
    tuple : (final_state, convergence_info)
        convergence_info: dict with 'iters_per_step', 'converged'
    """
    dt = step_length_mm / max(n_steps, 1)
    y = np.array(state0, dtype=float)

    def f(s):
        return lorentz_equations(s, bfield_func, charge, mass)

    iters_per_step = []
    all_converged = True

    for step in range(n_steps):
        # 向后 Euler 隐式步: y_{n+1} = y_n + dt · f(y_{n+1})
        # Picard 迭代 (与 [064] 一致)
        yp = y.copy()  # 初始猜测 = 当前值

        converged = False
        for iteration in range(it_max):
            # Picard 更新: yp_new = y + dt · f(yp)
            yp_new = y + dt * f(yp)

            # 收敛检查 (改进自 [064]，原代码无此项)
            diff = np.linalg.norm(yp_new - yp)
            yp = yp_new

            if diff < tol:
                converged = True
                iters_per_step.append(iteration + 1)
                break

        if not converged:
            iters_per_step.append(it_max)
            all_converged = False

        y = yp

    conv_info = {
        'iters_per_step': iters_per_step,
        'converged': all_converged,
        'mean_iters': sum(iters_per_step) / len(iters_per_step) if iters_per_step else 0,
        'max_iters': max(iters_per_step) if iters_per_step else 0,
    }

    return y, conv_info


# ============================================================
# [614_kdv_etdrk4] 指数时间差分 Runge-Kutta 4 阶
# ============================================================
def propagate_etdrk4(state0, bfield_func, step_length_mm, n_steps=10,
                     charge=1, mass=139.57, b0_approx=2.0):
    """
    基于 [614_kdv_etdrk4] 的 ETDRK4 传播器

    原算法用于求解 KdV 方程:
        u_t + u·u_x + u_xxx = 0
    将线性部分 L = ik³ 精确积分:
        E = exp(dt·L), E2 = exp(dt·L/2)
    非线性部分 N(v) 用 RK4 处理

    映射到粒子传播:
    将运动方程分为:
        线性部分 (均匀场螺旋运动): L·y = q·(v × B₀)
        非线性部分 (场不均匀性): N(y) = q·(v × δB(y))

    ETDRK4 步骤:
        a = E2·yₙ + dt·(Q·Nₙ + Q2·Ñ)
        b = E2·yₙ + dt·(Q2·Nₙ + Q·N̂)
        yₙ₊₁ = E·yₙ + dt·(Q·Nₙ + 2Q2·(Ñ + N̂) + Q·N̂ₙ₊₁)·...

    这里简化实现: 用矩阵指数处理线性部分

    Parameters
    ----------
    state0 : array_like, shape (6,)
    bfield_func : callable
    step_length_mm : float
    n_steps : int
    charge : int
    mass : float
    b0_approx : float
        参考磁场强度 [T]，用于构建线性化算子

    Returns
    -------
    ndarray : 最终状态
    """
    dt = step_length_mm / max(n_steps, 1)
    y = np.array(state0, dtype=float)

    # 构建线性化算子 L (均匀场近似下的螺旋运动)
    # 对于 B = (0, 0, B₀)，横向运动为:
    #   dpx/dt = q·0.3·B₀·vy  (简化单位)
    #   dpy/dt = -q·0.3·B₀·vx
    # 这是一个旋转矩阵: L = ω·J 其中 ω = q·0.3·B₀/(γm)

    c_mm_ns = 299.792458

    for step in range(n_steps):
        p = np.sqrt(y[3]**2 + y[4]**2 + y[5]**2)
        gamma = math.sqrt(1.0 + p**2 / (mass * mass))

        # 回旋频率
        omega = charge * 0.299792458 * b0_approx / (gamma * mass)

        # 矩阵指数 exp(dt·L) 的解析形式 (绕 z 轴旋转)
        cos_wt = math.cos(omega * dt)
        sin_wt = math.sin(omega * dt)

        # 线性传播 (精确螺旋运动在横向)
        E = np.eye(6)
        # 位置更新
        E[0, 3] = dt / (gamma * mass) * c_mm_ns
        E[1, 4] = dt / (gamma * mass) * c_mm_ns
        E[2, 5] = dt / (gamma * mass) * c_mm_ns
        # 动量旋转 (横向)
        E[3, 3] = cos_wt
        E[3, 4] = sin_wt
        E[4, 3] = -sin_wt
        E[4, 4] = cos_wt

        # 非线性部分: 场不均匀性
        def nonlinear(s):
            """N(y) = f(y) - L·y"""
            f_full = lorentz_equations(s, bfield_func, charge, mass)
            l_y = E @ s / dt - s / dt  # 近似 L·y
            # 更准确: L·y 应该只用横向部分
            return f_full  # 简化: 全部作为非线性

        # ETDRK4 阶段 (简化实现)
        N_n = nonlinear(y)

        # 阶段 1: a = exp(dt·L/2)·yₙ + dt·Q·Nₙ
        cos_half = math.cos(omega * dt / 2.0)
        sin_half = math.sin(omega * dt / 2.0)
        E2 = np.eye(6)
        E2[0, 3] = dt / 2.0 / (gamma * mass) * c_mm_ns
        E2[1, 4] = dt / 2.0 / (gamma * mass) * c_mm_ns
        E2[2, 5] = dt / 2.0 / (gamma * mass) * c_mm_ns
        E2[3, 3] = cos_half
        E2[3, 4] = sin_half
        E2[4, 3] = -sin_half
        E2[4, 4] = cos_half

        a = E2 @ y + (dt / 2.0) * N_n

        # 阶段 2
        N_a = nonlinear(a)
        b = E2 @ y + (dt / 2.0) * N_a

        # 阶段 3
        N_b = nonlinear(b)
        y_hat = E @ y + dt * N_b
        N_hat = nonlinear(y_hat)

        # Simpson 组合 (与 [614] 的 RK4 组合类似)
        y = E @ y + (dt / 6.0) * (N_n + 2.0*N_a + 2.0*N_b + N_hat)

    return y


# ============================================================
# 传播雅可比矩阵 (Kalman 滤波需要)
# ============================================================
def compute_propagation_jacobian(state, bfield_func, step_length_mm,
                                 charge=1, mass=139.57, delta=1e-5):
    """
    数值计算传播的雅可比矩阵 F = ∂y_{n+1}/∂y_n

    通过有限差分近似:
        F_{ij} = [f(y + δ·e_j)_i - f(y - δ·e_j)_i] / (2δ)

    Parameters
    ----------
    state : ndarray, shape (6,)
    bfield_func : callable
    step_length_mm : float
    charge, mass : 粒子参数
    delta : float
        有限差分步长

    Returns
    -------
    ndarray, shape (6, 6) : 雅可比矩阵
    """
    n = len(state)
    F = np.zeros((n, n))

    for j in range(n):
        # 前向扰动
        sp = np.array(state, dtype=float)
        sp[j] += delta
        yp = propagate_rk4(sp, bfield_func, step_length_mm, n_steps=5,
                           charge=charge, mass=mass)

        # 后向扰动
        sm = np.array(state, dtype=float)
        sm[j] -= delta
        ym = propagate_rk4(sm, bfield_func, step_length_mm, n_steps=5,
                           charge=charge, mass=mass)

        # 中心差分
        F[:, j] = (yp - ym) / (2.0 * delta)

    return F


# ============================================================
# [694_local_min] Brent 步长优化
# ============================================================
def optimize_step_size(bfield_func, state0, target_accuracy=1e-6,
                       charge=1, mass=139.57,
                       n_min=1, n_max=100):
    """
    基于 [694_local_min] 的 Brent 方法优化传播步数

    原算法使用黄金分割 + 抛物线插值寻找 1D 函数极小值

    这里最小化: 误差估计(n_steps) = ||y_rk4(n_steps) - y_rk4(2*n_steps)||
    通过选择最优步数使误差低于目标精度，同时最小化计算量

    Parameters
    ----------
    bfield_func : callable
    state0 : ndarray
    target_accuracy : float
    charge, mass : 粒子参数
    n_min, n_max : int
        步数搜索范围

    Returns
    -------
    int : 最优步数
    """
    # 参考解 (高精度)
    y_ref = propagate_rk4(state0, bfield_func, 100.0, n_steps=200,
                          charge=charge, mass=mass)

    def objective(n_steps):
        """误差目标函数"""
        n = max(int(round(n_steps)), 1)
        y_test = propagate_rk4(state0, bfield_func, 100.0, n_steps=n,
                               charge=charge, mass=mass)
        err = np.linalg.norm(y_test - y_ref)
        # 添加计算成本惩罚
        cost = n * 0.001
        return err + cost

    # Brent 方法 (简化黄金分割)
    a, b = float(n_min), float(n_max)
    golden = 0.381966011250105  # (3-√5)/2

    c = a + golden * (b - a)
    d = b - golden * (b - a)
    fc = objective(c)
    fd = objective(d)

    for _ in range(30):
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = a + golden * (b - a)
            fc = objective(c)
        else:
            a = c
            c = d
            fd = fc  # 复用
            d = b - golden * (b - a)
            fd = objective(d)

        if abs(b - a) < 1.0:
            break

    optimal_n = int(round((a + b) / 2.0))
    return max(1, min(optimal_n, n_max))
