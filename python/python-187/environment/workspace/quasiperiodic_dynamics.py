#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quasiperiodic_dynamics.py
=========================

基于种子项目 959_quasiperiodic_ode 的准周期动力学系统。

科学背景
--------
用户偏好在时间上演化往往呈现多频率叠加的准周期特性，例如：
- 日常周期（24 小时）
- 周周期（7 天）
- 季节性周期（1 年）

这些不可公约频率导致系统产生准周期（quasiperiodic）行为，
而非简单周期。我们将其建模为四阶线性 ODE:

    d⁴y/dt⁴ + (π² + 1) d²y/dt² + π² y = 0

该方程的特征多项式为:
    r⁴ + (π² + 1) r² + π² = 0
    (r² + 1)(r² + π²) = 0

根为 r = ±i, ±iπ，对应频率 1 和 π（不可公约），
因此通解为准周期函数:

    y(t) = C₁ cos(t) + C₂ sin(t) + C₃ cos(π t) + C₄ sin(π t)

在推荐系统中，y(t) 可作为用户偏好随时间演化的调制因子。
"""

import numpy as np


class QuasiperiodicPreferenceDynamics:
    """
    准周期偏好动力学求解器。
    """
    
    def __init__(self, pi=np.pi):
        """
        初始化参数。
        
        物理参数:
            π : 圆周率，代表第二个不可公约频率的基频
        """
        self.pi = float(pi)
        # ODE 系数: y^{(4)} + (π² + 1) y'' + π² y = 0
        self.coeff = self.pi**2 + 1.0
        
    def exact_solution(self, t):
        """
        计算精确解析解。
        
        参数:
            t : array-like, 时间序列
            
        返回:
            y : ndarray, shape (len(t), 4)
                y[:,0] = p(t) = cos(t) + cos(π t)
                y[:,1] = q(t) = -sin(t) - π sin(π t)
                y[:,2] = r(t) = -cos(t) - π² cos(π t)
                y[:,3] = s(t) = sin(t) + π³ sin(π t)
                
        公式推导:
            由 y = cos(t) + cos(π t) 出发:
            y'  = -sin(t) - π sin(π t) = q(t)
            y'' = -cos(t) - π² cos(π t) = r(t)
            y'''=  sin(t) + π³ sin(π t) = s(t)
            y^{(4)} = cos(t) + π⁴ cos(π t)
                    = - (π²+1)(-cos(t)-π²cos(πt)) - π²(cos(t)+cos(πt))
                    = - (π²+1) y'' - π² y
        """
        t = np.asarray(t, dtype=float)
        if t.ndim == 0:
            t = np.array([t])
        
        p = np.cos(t) + np.cos(self.pi * t)
        q = -np.sin(t) - self.pi * np.sin(self.pi * t)
        r = -np.cos(t) - self.pi**2 * np.cos(self.pi * t)
        s = np.sin(t) + self.pi**3 * np.sin(self.pi * t)
        
        return np.column_stack([p, q, r, s])
    
    def deriv(self, t, y):
        """
        返回 ODE 的右端项 dy/dt。
        
        状态向量 y = [y, y', y'', y''']^T
        
        系统矩阵形式:
            dy/dt = A y
            
            A = [[0,     1,      0,      0],
                 [0,     0,      1,      0],
                 [0,     0,      0,      1],
                 [-π²,   0,   -(π²+1),   0]]
                 
        边界鲁棒性:
            - 输入形状自动调整
            - 空输入返回空数组
        """
        y = np.asarray(y, dtype=float)
        if y.size == 0:
            return np.array([])
        
        # 确保 y 是列向量形状 (4,) 或 (4, n_batch)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        n = y.shape[1]
        dydt = np.zeros_like(y)
        
        dydt[0, :] = y[1, :]
        dydt[1, :] = y[2, :]
        dydt[2, :] = y[3, :]
        dydt[3, :] = -(self.pi**2) * y[0, :] - self.coeff * y[2, :]
        
        return dydt.squeeze() if dydt.shape[1] == 1 else dydt
    
    def integrate_ode(self, t_eval, method='RK4'):
        """
        数值积分求解准周期 ODE。
        
        方法:
            RK4 (Runge-Kutta 四阶):
                k1 = h f(t_n, y_n)
                k2 = h f(t_n + h/2, y_n + k1/2)
                k3 = h f(t_n + h/2, y_n + k2/2)
                k4 = h f(t_n + h, y_n + k3)
                y_{n+1} = y_n + (k1 + 2k2 + 2k3 + k4) / 6
                
        局部截断误差: O(h⁵)
        全局误差: O(h⁴)
        
        初始条件:
            y(0) = [2, 0, -(1+π²), 0]^T
            对应 y(t) = cos(t) + cos(π t) 的初值
        """
        t_eval = np.asarray(t_eval, dtype=float)
        if t_eval.size == 0:
            return np.array([])
        
        # 初始条件
        y0 = np.array([2.0, 0.0, -(1.0 + self.pi**2), 0.0])
        
        if method.upper() == 'RK4':
            return self._rk4_integrate(t_eval, y0)
        else:
            raise ValueError(f"不支持的积分方法: {method}")
    
    def _rk4_integrate(self, t_eval, y0):
        """RK4 固定步长积分器，带边界保护。"""
        t_eval = np.sort(t_eval)
        n_steps = len(t_eval)
        y_out = np.zeros((n_steps, 4))
        y_out[0, :] = y0
        
        for i in range(1, n_steps):
            h = t_eval[i] - t_eval[i-1]
            if abs(h) < 1e-15:
                y_out[i, :] = y_out[i-1, :]
                continue
                
            y_prev = y_out[i-1, :].reshape(-1, 1)
            
            k1 = h * self.deriv(t_eval[i-1], y_prev)
            k2 = h * self.deriv(t_eval[i-1] + h/2.0, y_prev + k1.reshape(-1,1)/2.0)
            k3 = h * self.deriv(t_eval[i-1] + h/2.0, y_prev + k2.reshape(-1,1)/2.0)
            k4 = h * self.deriv(t_eval[i], y_prev + k3.reshape(-1,1))
            
            k1 = k1.flatten()
            k2 = k2.flatten()
            k3 = k3.flatten()
            k4 = k4.flatten()
            
            y_out[i, :] = y_prev.flatten() + (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0
        
        return y_out
    
    def temporal_modulation(self, t, base_preference=1.0, amplitude=0.05):
        """
        计算时间调制因子用于推荐评分演化。
        
        模型:
            pref(t) = base_preference × (1 + amplitude × y(t))
            
        边界条件:
            - 确保结果始终为正
            - 当振幅过大时自动截断
        """
        y = self.exact_solution(np.atleast_1d(t))[:, 0]
        modulation = base_preference * (1.0 + amplitude * y)
        # 数值鲁棒性: 不允许负偏好
        modulation = np.maximum(modulation, 0.1)
        return modulation
