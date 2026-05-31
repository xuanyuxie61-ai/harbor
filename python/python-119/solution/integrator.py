"""
integrator.py
分子动力学积分器模块

融合原项目:
- 1033_rk23: Runge-Kutta 2/3 阶 ODE 求解器，用于扩展系统热浴耦合
- 360_fd1d_heat_explicit: 显式格式的 CFL 稳定性条件思想用于时间步长约束

功能:
1. Velocity Verlet 积分器（主 MD 积分）
2. RK23 求解器用于 Nose-Hoover 热浴的扩展变量演化
3. 时间步长自适应与安全检查
"""

import numpy as np
from typing import Callable, Tuple, Optional


class VelocityVerletIntegrator:
    """
    Velocity Verlet 积分器。
    
    算法:
        1. v(t+dt/2) = v(t) + (dt/2m) F(t)
        2. r(t+dt) = r(t) + dt * v(t+dt/2)
        3. 计算 F(t+dt)
        4. v(t+dt) = v(t+dt/2) + (dt/2m) F(t+dt)
    
    优点:
        - 时间可逆
        - 辛结构保持
        - 二阶精度 O(dt^2)
    """
    
    def __init__(self, dt: float = 0.001):
        """
        参数:
            dt: 时间步长
        """
        if dt <= 0:
            raise ValueError("dt 必须 > 0")
        self.dt = dt
    
    def step(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        forces: np.ndarray,
        masses: np.ndarray,
        box: np.ndarray,
        force_func: Callable,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        执行一步 Velocity Verlet 积分。
        
        参数:
            positions: (N, 3) 当前位置
            velocities: (N, 3) 当前速度
            forces: (N, 3) 当前力
            masses: (N,) 质量
            box: (3,) 盒子尺寸
            force_func: 力计算函数 force_func(positions) -> forces
        
        返回:
            (new_positions, new_velocities, new_forces)
        """
        dt = self.dt
        N = positions.shape[0]
        
        # 质量倒数，避免重复除法
        inv_m = 1.0 / masses[:, np.newaxis]
        
        # 半步速度
        v_half = velocities + 0.5 * dt * forces * inv_m
        
        # 更新位置
        new_positions = positions + dt * v_half
        
        # 周期性边界
        new_positions = new_positions % box
        
        # 计算新力
        new_forces = force_func(new_positions)
        
        # 检查力的数值异常
        if np.any(~np.isfinite(new_forces)):
            # 回退: 保持原力
            new_forces = forces.copy()
        
        # 全步速度
        new_velocities = v_half + 0.5 * dt * new_forces * inv_m
        
        # 速度数值安全检查
        max_vel = 100.0 / dt  # 速度上限
        vel_norm = np.linalg.norm(new_velocities, axis=1)
        if np.any(vel_norm > max_vel):
            scale = np.where(vel_norm > max_vel, max_vel / vel_norm, 1.0)
            new_velocities = new_velocities * scale[:, np.newaxis]
        
        return new_positions, new_velocities, new_forces
    
    def cfl_constraint(self, positions: np.ndarray, velocities: np.ndarray, box: np.ndarray) -> bool:
        """
        检查 CFL-like 稳定性条件。
        
        条件:
            dt * max(|v|) < min(box) / 10
        
        参数:
            positions: (N, 3) 位置
            velocities: (N, 3) 速度
            box: (3,) 盒子尺寸
        
        返回:
            是否满足稳定性条件
        """
        max_vel = np.max(np.linalg.norm(velocities, axis=1))
        min_box = np.min(box)
        
        if max_vel < 1e-15:
            return True
        
        courant = self.dt * max_vel / min_box
        return courant < 0.1


def rk23_step(
    y: np.ndarray,
    t: float,
    dt: float,
    f: Callable[[float, np.ndarray], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    执行一步 RK23（Runge-Kutta 2/3 阶）积分。
    
    融合原项目 1033_rk23:
        Butcher 表（Heun-Embedded 方法）:
        k1 = dt * f(t, y)
        k2 = dt * f(t + dt, y + k1)
        k3 = dt * f(t + dt/2, y + k1/4 + k2/4)
        
        y2 = y + (k1 + k2) / 2       (二阶)
        y3 = y + (k1 + k2 + 4*k3) / 6 (三阶)
        e  = y3 - y2                  (误差估计)
    
    参数:
        y: 当前状态向量
        t: 当前时间
        dt: 时间步长
        f: 右端函数 f(t, y) -> dy/dt
    
    返回:
        (y_next, error_estimate, dt_suggested)
    """
    if dt <= 0:
        raise ValueError("rk23_step: dt 必须 > 0")
    
    k1 = dt * f(t, y)
    k2 = dt * f(t + dt, y + k1)
    k3 = dt * f(t + 0.5 * dt, y + 0.25 * k1 + 0.25 * k2)
    
    y2 = y + 0.5 * (k1 + k2)
    y3 = y + (k1 + k2 + 4.0 * k3) / 6.0
    
    error = np.abs(y3 - y2)
    
    # 步长建议（基于误差控制）
    tol = 1e-6
    max_err = np.max(error)
    if max_err < 1e-15:
        dt_suggested = 2.0 * dt
    else:
        dt_suggested = dt * min(2.0, max(0.25, 0.9 * (tol / max_err) ** (1.0 / 3.0)))
    
    return y3, error, dt_suggested


def rk23_integrate(
    y0: np.ndarray,
    tspan: Tuple[float, float],
    n_steps: int,
    f: Callable[[float, np.ndarray], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    RK23 积分器，固定步数。
    
    参数:
        y0: 初始状态
        tspan: (t0, t1) 时间区间
        n_steps: 步数
        f: 右端函数
    
    返回:
        (t_array, y_array, error_array)
    """
    t0, t1 = tspan
    dt = (t1 - t0) / n_steps
    m = len(y0)
    
    t_arr = np.zeros(n_steps + 1)
    y_arr = np.zeros((n_steps + 1, m))
    e_arr = np.zeros((n_steps + 1, m))
    
    t_arr[0] = t0
    y_arr[0, :] = y0
    e_arr[0, :] = 0.0
    
    for i in range(n_steps):
        y_next, err, _ = rk23_step(y_arr[i], t_arr[i], dt, f)
        t_arr[i + 1] = t_arr[i] + dt
        y_arr[i + 1, :] = y_next
        e_arr[i + 1, :] = err
    
    return t_arr, y_arr, e_arr


class NoseHooverIntegrator:
    """
    Nose-Hoover 恒温积分器。
    
    扩展系统哈密顿量:
        H_NH = Σ_i p_i^2/(2m_i s^2) + U(r) + p_s^2/(2Q) + g k_B T ln(s)
    
    其中:
        s: 热浴缩放因子
        p_s: 热浴动量
        Q: 热浴质量参数（热惯量）
        g = 3N - 3: 自由度
    
    运动方程:
        dr_i/dt = v_i
        dv_i/dt = F_i/m_i - ξ v_i
        dξ/dt = (2E_k - g k_B T) / Q
        其中 ξ = p_s / Q = ds/dt / s
    """
    
    def __init__(self, dt: float = 0.001, Q: float = 10.0):
        """
        参数:
            dt: 时间步长
            Q: Nose-Hoover 热浴质量参数
        """
        if dt <= 0 or Q <= 0:
            raise ValueError("dt 和 Q 必须 > 0")
        self.dt = dt
        self.Q = Q
        self.xi = 0.0  # 热浴变量初始值
    
    def step(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        forces: np.ndarray,
        masses: np.ndarray,
        box: np.ndarray,
        target_temperature: float,
        force_func: Callable,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        执行一步 Nose-Hoover 恒温积分。
        
        使用 RK23 求解热浴变量的耦合 ODE。
        
        参数:
            positions, velocities, forces, masses, box: 标准 MD 变量
            target_temperature: 目标温度 T_target
            force_func: 力计算函数
        
        返回:
            (new_positions, new_velocities, new_forces)
        """
        dt = self.dt
        N = positions.shape[0]
        ndof = 3 * N - 3
        inv_m = 1.0 / masses[:, np.newaxis]
        
        # 当前动能
        ek = 0.5 * np.sum(masses[:, np.newaxis] * velocities ** 2)
        
        # 使用 RK23 更新热浴变量 xi
        def bath_ode(t, xi_vec):
            xi_val = xi_vec[0]
            # dxi/dt = (2*E_k - g*k_B*T) / Q
            # 注意: 这里使用当前 E_k 近似
            return np.array([(2.0 * ek - ndof * target_temperature) / self.Q])
        
        xi_arr, _, _ = rk23_integrate(np.array([self.xi]), (0.0, dt), 1, bath_ode)
        self.xi = float(xi_arr[-1, 0])
        
        # Velocity Verlet 主步，加入热浴阻尼
        v_half = velocities + 0.5 * dt * forces * inv_m - 0.5 * dt * self.xi * velocities
        
        new_positions = positions + dt * v_half
        new_positions = new_positions % box
        
        new_forces = force_func(new_positions)
        if np.any(~np.isfinite(new_forces)):
            new_forces = forces.copy()
        
        new_velocities = v_half + 0.5 * dt * new_forces * inv_m - 0.5 * dt * self.xi * v_half
        
        # 速度数值安全
        max_vel = 100.0 / dt
        vel_norm = np.linalg.norm(new_velocities, axis=1)
        if np.any(vel_norm > max_vel):
            scale = np.where(vel_norm > max_vel, max_vel / vel_norm, 1.0)
            new_velocities = new_velocities * scale[:, np.newaxis]
        
        return new_positions, new_velocities, new_forces
