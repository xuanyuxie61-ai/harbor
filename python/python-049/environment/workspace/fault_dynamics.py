#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class FaultRuptureDynamics:
    
    def __init__(self, mu_0=0.6, a=0.015, b=0.020, V_0=1e-6,
                 D_c=0.01, k=1e9, V_pl=1e-9, sigma_n=100e6):
        self.mu_0 = mu_0
        self.a = a
        self.b = b
        self.V_0 = V_0
        self.D_c = D_c
        self.k = k
        self.V_pl = V_pl
        self.sigma_n = sigma_n
        

        if a <= 0 or b <= 0:
            raise ValueError("摩擦参数 a, b 必须为正")
        if D_c <= 0:
            raise ValueError("临界滑动距离 D_c 必须为正")
        if sigma_n <= 0:
            raise ValueError("正应力 sigma_n 必须为正")
    
    def friction_coefficient(self, V, theta):

        V_reg = max(V, 1e-12)
        theta_reg = max(theta, 1e-12)
        
        mu = self.mu_0 + self.a * np.log(V_reg / self.V_0) \
             + self.b * np.log(theta_reg * self.V_0 / self.D_c)
        return mu
    
    def derivatives(self, t, y):
        y1, y2, y3, y4 = y
        

        y2 = max(y2, 1e-12)
        y4 = max(y4, 1e-12)
        

        mu_norm = 1.0 + self.a * np.log(y2) + self.b * np.log(y4)
        


        dy1dt = (self.k * self.V_pl) / (self.sigma_n * self.mu_0) \
                - (self.k * self.V_0 * y2) / (self.sigma_n * self.mu_0)
        




        dy2dt = 2.0 * t * (y2 ** 0.2) * y4 * (mu_norm - y1)
        

        dy3dt = 2.0 * t * y4
        

        dy4dt = -2.0 * t * np.log(y1)
        

        dydt = np.array([dy1dt, dy2dt, dy3dt, dy4dt])
        dydt = np.nan_to_num(dydt, nan=0.0, posinf=0.0, neginf=0.0)
        
        return dydt
    
    def solve_rupture_ode(self, t_span, y0, n_steps=500, it_max=10):
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
            

            for _ in range(it_max):
                f_val = self.derivatives(tp, yp)
                yp = y[i, :] + dt * f_val
            
            t[i + 1] = tp
            y[i + 1, :] = yp
        
        return t, y


class PendulumConservationMonitor:
    
    def __init__(self, g=9.81, L=1.0, m=1.0):
        self.g = g
        self.L = L
        self.m = m
        
        if L <= 0:
            raise ValueError("摆长 L 必须为正")
        if m <= 0:
            raise ValueError("质量 m 必须为正")
    
    def check_energy_conservation(self, theta, omega):

        theta_norm = np.abs(theta)
        omega_norm = np.abs(omega)
        

        kinetic = 0.5 * self.m * (self.L ** 2) * (omega_norm ** 2)
        potential = self.m * self.g * self.L * (1.0 - np.cos(np.clip(theta_norm, -np.pi, np.pi)))
        E_total = kinetic + potential
        

        E0 = E_total[0]
        if E0 < 1e-12:
            return np.max(np.abs(E_total - E0))
        
        rel_deviation = np.abs(E_total - E0) / E0
        return np.max(rel_deviation)
