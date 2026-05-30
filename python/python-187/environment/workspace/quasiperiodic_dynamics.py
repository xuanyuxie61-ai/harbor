#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class QuasiperiodicPreferenceDynamics:
    
    def __init__(self, pi=np.pi):
        self.pi = float(pi)

        self.coeff = self.pi**2 + 1.0
        
    def exact_solution(self, t):
        t = np.asarray(t, dtype=float)
        if t.ndim == 0:
            t = np.array([t])
        
        p = np.cos(t) + np.cos(self.pi * t)
        q = -np.sin(t) - self.pi * np.sin(self.pi * t)
        r = -np.cos(t) - self.pi**2 * np.cos(self.pi * t)
        s = np.sin(t) + self.pi**3 * np.sin(self.pi * t)
        
        return np.column_stack([p, q, r, s])
    
    def deriv(self, t, y):
        y = np.asarray(y, dtype=float)
        if y.size == 0:
            return np.array([])
        

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
        t_eval = np.asarray(t_eval, dtype=float)
        if t_eval.size == 0:
            return np.array([])
        

        y0 = np.array([2.0, 0.0, -(1.0 + self.pi**2), 0.0])
        
        if method.upper() == 'RK4':
            return self._rk4_integrate(t_eval, y0)
        else:
            raise ValueError(f"不支持的积分方法: {method}")
    
    def _rk4_integrate(self, t_eval, y0):
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
        y = self.exact_solution(np.atleast_1d(t))[:, 0]
        modulation = base_preference * (1.0 + amplitude * y)

        modulation = np.maximum(modulation, 0.1)
        return modulation
