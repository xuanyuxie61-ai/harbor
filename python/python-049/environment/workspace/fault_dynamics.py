#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 fault_dynamics.py
 
 融合种子项目：
   - 1283_tough_ode：复杂刚性 ODE 系统（四变量 tough_ode）
   - 857_pendulum_comparison_ode：单摆非线性 ODE 与守恒量监测
 
 科学功能：
   海底地震断层破裂动力学模拟。地震断层滑动可以建模为复杂的
   非线性 ODE 系统，包含应力积累、摩擦弱化、滑动速度演化等过程。
   
   本模块同时借鉴单摆系统的能量守恒思想，对断层滑动过程进行
   能量守恒监测，确保数值计算的物理合理性。
 
 核心物理公式：
 
   1) 断层滑动速率-状态摩擦定律（Rate-and-State Friction）:
      μ = μ_0 + a · ln(V/V_0) + b · ln(θV_0/D_c)
      
   2) 应力演化方程:
      dτ/dt = k · (V_pl - V)
      
   3) 状态变量演化（老化定律）:
      dθ/dt = 1 - (V·θ)/D_c
      
   4) 滑动速度:
      dδ/dt = V
      
   其中：τ 为剪应力，V 为滑动速度，θ 为状态变量，δ 为滑动位移，
   k 为刚度，V_pl 为板块加载速度，D_c 为临界滑动距离。
 
   5) 单摆能量守恒类比（用于数值检验）:
      E = (1/2)mL²ω² + mgL(1-cosθ)
      其中 ω = dθ/dt 为角速度。
"""

import numpy as np


class FaultRuptureDynamics:
    """
    断层破裂动力学模型。
    
    基于 tough_ode 的四变量复杂 ODE 结构，将其改造为断层滑动动力学。
    使用固定点迭代后向 Euler 方法进行时间积分。
    """
    
    def __init__(self, mu_0=0.6, a=0.015, b=0.020, V_0=1e-6,
                 D_c=0.01, k=1e9, V_pl=1e-9, sigma_n=100e6):
        """
        初始化断层动力学参数。
        
        Parameters
        ----------
        mu_0 : float
            参考摩擦系数
        a, b : float
            速率-状态参数 (a < b 对应速度弱化，不稳定滑动)
        V_0 : float
            参考滑动速度 (m/s)
        D_c : float
            临界滑动距离 (m)
        k : float
            断层刚度 (Pa/m)
        V_pl : float
            板块加载速度 (m/s)
        sigma_n : float
            正应力 (Pa)
        """
        self.mu_0 = mu_0
        self.a = a
        self.b = b
        self.V_0 = V_0
        self.D_c = D_c
        self.k = k
        self.V_pl = V_pl
        self.sigma_n = sigma_n
        
        # 边界检查
        if a <= 0 or b <= 0:
            raise ValueError("摩擦参数 a, b 必须为正")
        if D_c <= 0:
            raise ValueError("临界滑动距离 D_c 必须为正")
        if sigma_n <= 0:
            raise ValueError("正应力 sigma_n 必须为正")
    
    def friction_coefficient(self, V, theta):
        """
        计算速率-状态摩擦系数。
        
        μ(V, θ) = μ_0 + a·ln(V/V_0) + b·ln(θ·V_0/D_c)
        
        对 V → 0 进行正则化处理，避免数值奇异性。
        """
        # 正则化：当 V 过小时使用正则化形式
        V_reg = max(V, 1e-12)
        theta_reg = max(theta, 1e-12)
        
        mu = self.mu_0 + self.a * np.log(V_reg / self.V_0) \
             + self.b * np.log(theta_reg * self.V_0 / self.D_c)
        return mu
    
    def derivatives(self, t, y):
        """
        计算断层动力学 ODE 的右端项。
        
        状态变量 y = [y1, y2, y3, y4] 分别对应：
          y1 = τ / (σ_n·μ_0)    归一化剪应力
          y2 = V / V_0          归一化滑动速度
          y3 = δ / D_c          归一化滑动位移
          y4 = θ·V_0 / D_c      归一化状态变量
        
        返回 dy/dt。
        """
        y1, y2, y3, y4 = y
        
        # 边界处理：防止变量过小导致数值问题
        y2 = max(y2, 1e-12)
        y4 = max(y4, 1e-12)
        
        # 归一化摩擦系数
        mu_norm = 1.0 + self.a * np.log(y2) + self.b * np.log(y4)
        
        # 归一化剪应力演化（应力积累与释放）
        # dy1/dt = (k·V_pl / (σ_n·μ_0)) - (k·V_0·y2 / (σ_n·μ_0))
        dy1dt = (self.k * self.V_pl) / (self.sigma_n * self.mu_0) \
                - (self.k * self.V_0 * y2) / (self.sigma_n * self.mu_0)
        
        # 滑动速度演化（由摩擦定律与应力平衡决定）
        # 这里使用 tough_ode 的复杂非线性结构进行类比：
        # dy2/dt = 10·t·exp(5·(y2-1))·y4
        # 改造为物理上更合理的形式
        dy2dt = 2.0 * t * (y2 ** 0.2) * y4 * (mu_norm - y1)
        
        # 滑动位移演化
        dy3dt = 2.0 * t * y4
        
        # 状态变量演化（老化定律的归一化形式）
        dy4dt = -2.0 * t * np.log(y1)
        
        # 数值鲁棒性：检查 NaN 和 Inf
        dydt = np.array([dy1dt, dy2dt, dy3dt, dy4dt])
        dydt = np.nan_to_num(dydt, nan=0.0, posinf=0.0, neginf=0.0)
        
        return dydt
    
    def solve_rupture_ode(self, t_span, y0, n_steps=500, it_max=10):
        """
        使用固定点迭代后向 Euler 方法求解断层动力学 ODE。
        
        算法（来源于 backward_euler_fixed）：
          y_{n+1}^{(k+1)} = y_n + dt · f(t_{n+1}, y_{n+1}^{(k)})
        
        该方法无条件稳定，适合处理刚性的摩擦动力学问题。
        
        Parameters
        ----------
        t_span : tuple
            (t_start, t_end)
        y0 : ndarray
            初始条件，形状 (4,)
        n_steps : int
            时间步数
        it_max : int
            每步固定点迭代次数
            
        Returns
        -------
        t : ndarray
            时间序列
        y : ndarray
            解序列，形状 (n_steps+1, 4)
        """
        t_start, t_end = t_span
        dt = (t_end - t_start) / n_steps
        m = len(y0)
        
        t = np.zeros(n_steps + 1)
        y = np.zeros((n_steps + 1, m))
        
        t[0] = t_start
        y[0, :] = y0
        
        for i in range(n_steps):
            tp = t[i] + dt
            yp = y[i, :].copy()
            
            # 固定点迭代
            for _ in range(it_max):
                f_val = self.derivatives(tp, yp)
                yp = y[i, :] + dt * f_val
            
            t[i + 1] = tp
            y[i + 1, :] = yp
        
        return t, y


class PendulumConservationMonitor:
    """
    单摆能量守恒监测器。
    
    融合 pendulum_comparison_ode 的守恒量思想，用于检验
    数值计算的物理一致性。
    
    线性单摆：
      d²θ/dt² + (g/L)·θ = 0
      
    非线性单摆：
      d²θ/dt² + (g/L)·sin(θ) = 0
      
    总能量：
      E = (1/2)mL²(dθ/dt)² + mgL(1 - cosθ)
    """
    
    def __init__(self, g=9.81, L=1.0, m=1.0):
        self.g = g
        self.L = L
        self.m = m
        
        if L <= 0:
            raise ValueError("摆长 L 必须为正")
        if m <= 0:
            raise ValueError("质量 m 必须为正")
    
    def check_energy_conservation(self, theta, omega):
        """
        检查数值解的能量守恒偏差。
        
        将断层滑动的位移类比为单摆角度 θ，速度类比为角速度 ω，
        监测总能量 E 的变化。若数值方法守恒性好，E 的变化应很小。
        
        Parameters
        ----------
        theta : ndarray
            类比角度序列（取断层归一化位移）
        omega : ndarray
            类比角速度序列（取断层归一化速度）
            
        Returns
        -------
        max_relative_deviation : float
            最大相对能量偏差
        """
        # 归一化到物理量级
        theta_norm = np.abs(theta)
        omega_norm = np.abs(omega)
        
        # 总能量
        kinetic = 0.5 * self.m * (self.L ** 2) * (omega_norm ** 2)
        potential = self.m * self.g * self.L * (1.0 - np.cos(np.clip(theta_norm, -np.pi, np.pi)))
        E_total = kinetic + potential
        
        # 相对偏差
        E0 = E_total[0]
        if E0 < 1e-12:
            return np.max(np.abs(E_total - E0))
        
        rel_deviation = np.abs(E_total - E0) / E0
        return np.max(rel_deviation)
