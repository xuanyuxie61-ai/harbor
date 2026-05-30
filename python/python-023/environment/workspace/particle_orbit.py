#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def boris_push(x, v, q, m, B, E, dt):

    v_sq = np.sum(v**2)
    c = 2.99792458e8
    gamma = np.sqrt(1.0 + v_sq / c**2)
    

    v_minus = v + (q * dt / (2.0 * m)) * E
    

    t_vec = (q * dt / (2.0 * m * gamma)) * B
    t_sq = np.sum(t_vec**2)
    
    v_prime = v_minus + np.cross(v_minus, t_vec)
    

    denom = 1.0 + t_sq
    if denom < 1e-30:
        denom = 1e-30
    
    v_plus = v_minus + np.cross(v_prime, 2.0 * t_vec) / denom
    

    v_new = v_plus + (q * dt / (2.0 * m)) * E
    

    x_new = x + dt * v_new / gamma
    
    return x_new, v_new


def rk45_step(f, t, y, dt, args=()):
    a = [
        [0.0],
        [1.0/5.0],
        [3.0/40.0, 9.0/40.0],
        [44.0/45.0, -56.0/15.0, 32.0/9.0],
        [19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0],
        [9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0],
        [35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0]
    ]
    
    b4 = [35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0]
    b5 = [5179.0/57600.0, 0.0, 7571.0/16695.0, 393.0/640.0, -92097.0/339200.0, 187.0/2100.0, 1.0/40.0]
    
    c = [0.0, 1.0/5.0, 3.0/10.0, 4.0/5.0, 8.0/9.0, 1.0, 1.0]
    
    k = []
    k.append(np.array(f(t, y, *args)))
    
    for i in range(1, 7):
        ti = t + c[i] * dt
        yi = y.copy()
        for j in range(i):
            yi = yi + dt * a[i][j] * k[j]
        k.append(np.array(f(ti, yi, *args)))
    
    y4 = y + dt * sum(b4[i] * k[i] for i in range(7))
    y5 = y + dt * sum(b5[i] * k[i] for i in range(7))
    
    error = np.linalg.norm(y5 - y4)
    
    return y5, error


def lorentz_force(t, state, q, m, B_field_func, E_field_func):
    x = state[0:3]
    v = state[3:6]
    
    B = B_field_func(x, t)
    E = E_field_func(x, t)
    
    c = 2.99792458e8
    v_sq = np.sum(v**2)
    gamma = np.sqrt(1.0 + v_sq / c**2)
    
    dxdt = v / gamma
    dvdt = (q / m) * (E + np.cross(v, B))
    
    return np.concatenate([dxdt, dvdt])


def integrate_lorentz_orbits(x0, v0, B0_vec, params, t_span, n_steps):
    N = x0.shape[0]
    q_e = params['q_e']
    m_e = params['m_e']
    c = params['c']
    
    dt = t_span / n_steps
    

    def B_field(x, t):
        if x.ndim > 1:
            x = x.flatten()

        B = B0_vec.copy()

        B[0] += 0.05 * B0_vec[2] * np.sin(2 * np.pi * x[2] / 1e4)
        B[1] += 0.05 * B0_vec[2] * np.cos(2 * np.pi * x[2] / 1e4)
        return B
    
    def E_field(x, t):
        return np.zeros(3)
    
    orbits = np.zeros((N, n_steps + 1, 6))
    orbits[:, 0, 0:3] = x0
    orbits[:, 0, 3:6] = v0
    

    Omega_e = params['Omega_e']
    dt_max = 0.1 / Omega_e
    
    for i in range(N):
        state = np.concatenate([x0[i], v0[i]])
        
        for step in range(n_steps):

            x_curr = state[0:3]
            v_curr = state[3:6]
            
            B_curr = B_field(x_curr, step * dt)
            E_curr = E_field(x_curr, step * dt)
            

            n_sub = max(1, int(np.ceil(dt / dt_max)))
            dt_sub = dt / n_sub
            
            for _ in range(n_sub):
                x_curr, v_curr = boris_push(x_curr, v_curr, q_e, m_e, B_curr, E_curr, dt_sub)
            
            state = np.concatenate([x_curr, v_curr])
            orbits[i, step + 1] = state
    

    if N >= 2:
        separation = np.linalg.norm(orbits[0, :, 0:3] - orbits[1, :, 0:3], axis=1)

        separation = np.maximum(separation, 1e-30)
        times = np.linspace(0, t_span, n_steps + 1)

        valid = separation > 1e-20
        if np.sum(valid) > 10:
            lyap = np.polyfit(times[valid], np.log(separation[valid]), 1)[0]
            print(f"       最大李雅普诺夫指数 λ_max = {lyap:.4e} s⁻¹")
    
    return orbits
