#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class ShallowWaterSolver:
    
    def __init__(self, x, y, h_bathy, g=9.81, Cd=0.0025, rho=1025.0,
                 dt=30.0, n_steps=120):
        self.x = x
        self.y = y
        self.h_bathy = np.maximum(h_bathy, 10.0)
        self.g = g
        self.Cd = Cd
        self.rho = rho
        self.dt = dt
        self.n_steps = n_steps
        
        self.nx = len(x)
        self.ny = len(y)
        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]
        
        if self.dx <= 0 or self.dy <= 0:
            raise ValueError("网格间距必须为正")
        if np.any(self.h_bathy <= 0):
            raise ValueError("静水深必须为正")
        if dt <= 0:
            raise ValueError("时间步长必须为正")
        

        c_max = np.sqrt(g * np.max(h_bathy))
        dt_cfl = 0.5 * min(self.dx, self.dy) / c_max
        self.dt = min(dt, dt_cfl)

        total_time = dt * n_steps
        self.n_steps = max(int(total_time / self.dt), 1)
        self.dt = total_time / self.n_steps
        
        self.eta = np.zeros((self.ny, self.nx))
        self.u = np.zeros((self.ny, self.nx))
        self.v = np.zeros((self.ny, self.nx))
    
    def set_initial_condition(self, eta0, u0=None, v0=None):
        if eta0.shape != (self.ny, self.nx):
            raise ValueError(f"初始 eta 形状不匹配: {eta0.shape} vs {(self.ny, self.nx)}")
        self.eta = eta0.copy()
        if u0 is not None:
            self.u = u0.copy()
        if v0 is not None:
            self.v = v0.copy()
    
    def _divergence_flux(self, eta, u, v):
        H = self.h_bathy + eta
        H = np.maximum(H, 0.1)
        

        H_at_u = 0.5 * (H + np.roll(H, -1, axis=1))
        flux_x = H_at_u * u
        d_flux_x = (flux_x - np.roll(flux_x, 1, axis=1)) / self.dx
        

        H_at_v = 0.5 * (H + np.roll(H, -1, axis=0))
        flux_y = H_at_v * v
        d_flux_y = (flux_y - np.roll(flux_y, 1, axis=0)) / self.dy
        
        return d_flux_x + d_flux_y
    
    def _convection_u(self, u, v):

        du_dx = np.zeros_like(u)
        du_dy = np.zeros_like(u)
        

        for j in range(self.ny):
            for i in range(1, self.nx - 1):
                if u[j, i] >= 0:
                    du_dx[j, i] = (u[j, i] - u[j, i-1]) / self.dx
                else:
                    du_dx[j, i] = (u[j, i+1] - u[j, i]) / self.dx
        

        for j in range(1, self.ny - 1):
            for i in range(self.nx):
                v_avg = 0.25 * (v[j, i] + v[j, max(0, i-1)] + v[max(0, j-1), i] + v[max(0, j-1), max(0, i-1)])
                if v_avg >= 0:
                    du_dy[j, i] = (u[j, i] - u[max(0, j-1), i]) / self.dy
                else:
                    du_dy[j, i] = (u[min(self.ny-1, j+1), i] - u[j, i]) / self.dy
        
        v_at_u = np.zeros_like(u)
        for j in range(self.ny):
            for i in range(self.nx):
                jm = max(0, j-1)
                im = max(0, i-1)
                v_at_u[j, i] = 0.25 * (v[j, i] + v[j, im] + v[jm, i] + v[jm, im])
        
        return -(u * du_dx + v_at_u * du_dy)
    
    def _convection_v(self, u, v):
        dv_dx = np.zeros_like(v)
        dv_dy = np.zeros_like(v)
        

        for j in range(self.ny):
            for i in range(1, self.nx - 1):
                u_avg = 0.25 * (u[j, i] + u[j, min(self.nx-1, i+1)] + u[max(0, j-1), i] + u[max(0, j-1), min(self.nx-1, i+1)])
                if u_avg >= 0:
                    dv_dx[j, i] = (v[j, i] - v[j, i-1]) / self.dx
                else:
                    dv_dx[j, i] = (v[j, i+1] - v[j, i]) / self.dx
        

        for j in range(1, self.ny - 1):
            for i in range(self.nx):
                if v[j, i] >= 0:
                    dv_dy[j, i] = (v[j, i] - v[j-1, i]) / self.dy
                else:
                    dv_dy[j, i] = (v[j+1, i] - v[j, i]) / self.dy
        
        u_at_v = np.zeros_like(v)
        for j in range(self.ny):
            for i in range(self.nx):
                jm = max(0, j-1)
                im = max(0, i-1)
                u_at_v[j, i] = 0.25 * (u[j, i] + u[j, im] + u[jm, i] + u[jm, im])
        
        return -(u_at_v * dv_dx + v * dv_dy)
    
    def _friction_u(self, eta, u, v):
        H = self.h_bathy + eta
        H = np.maximum(H, 0.1)
        
        v_at_u = np.zeros_like(u)
        for j in range(self.ny):
            for i in range(self.nx):
                jm = max(0, j-1)
                im = max(0, i-1)
                v_at_u[j, i] = 0.25 * (v[j, i] + v[j, im] + v[jm, i] + v[jm, im])
        
        speed = np.sqrt(u**2 + v_at_u**2)
        speed = np.minimum(speed, 50.0)
        return -self.Cd * speed * u / H
    
    def _friction_v(self, eta, u, v):
        H = self.h_bathy + eta
        H = np.maximum(H, 0.1)
        
        u_at_v = np.zeros_like(v)
        for j in range(self.ny):
            for i in range(self.nx):
                jm = max(0, j-1)
                im = max(0, i-1)
                u_at_v[j, i] = 0.25 * (u[j, i] + u[j, im] + u[jm, i] + u[jm, im])
        
        speed = np.sqrt(u_at_v**2 + v**2)
        speed = np.minimum(speed, 50.0)
        return -self.Cd * speed * v / H
    
    def _pressure_gradient_x(self, eta):
        d_eta = np.zeros_like(eta)
        d_eta[:, :-1] = (eta[:, 1:] - eta[:, :-1]) / self.dx
        return -self.g * d_eta
    
    def _pressure_gradient_y(self, eta):
        d_eta = np.zeros_like(eta)
        d_eta[:-1, :] = (eta[1:, :] - eta[:-1, :]) / self.dy
        return -self.g * d_eta
    
    def _laplacian_eta(self, eta, H_bar):
        lap = np.zeros_like(eta)
        
        for j in range(1, self.ny - 1):
            for i in range(1, self.nx - 1):

                H_right = 0.5 * (H_bar[j, i] + H_bar[j, i+1])
                H_left = 0.5 * (H_bar[j, i] + H_bar[j, i-1])
                d_eta_right = (eta[j, i+1] - eta[j, i]) / self.dx
                d_eta_left = (eta[j, i] - eta[j, i-1]) / self.dx
                lap[j, i] += (H_right * d_eta_right - H_left * d_eta_left) / self.dx
                

                H_top = 0.5 * (H_bar[j, i] + H_bar[j+1, i])
                H_bottom = 0.5 * (H_bar[j, i] + H_bar[j-1, i])
                d_eta_top = (eta[j+1, i] - eta[j, i]) / self.dy
                d_eta_bottom = (eta[j, i] - eta[j-1, i]) / self.dy
                lap[j, i] += (H_top * d_eta_top - H_bottom * d_eta_bottom) / self.dy
        
        return lap
    
    def _apply_sponge_boundary(self, eta, u, v):
        sponge_width = min(5, self.nx // 10, self.ny // 10)
        if sponge_width < 1:
            return eta, u, v
        
        for i in range(sponge_width):
            factor = 0.5 * (1.0 - np.cos(np.pi * i / sponge_width))
            eta[:, i] *= factor
            u[:, i] *= factor
            v[:, i] *= factor
            eta[:, -1-i] *= factor
            u[:, -1-i] *= factor
            v[:, -1-i] *= factor
        
        for j in range(sponge_width):
            factor = 0.5 * (1.0 - np.cos(np.pi * j / sponge_width))
            eta[j, :] *= factor
            u[j, :] *= factor
            v[j, :] *= factor
            eta[-1-j, :] *= factor
            u[-1-j, :] *= factor
            v[-1-j, :] *= factor
        
        return eta, u, v
    
    def _time_step(self, eta_n, u_n, v_n):
        H_bar = self.h_bathy + eta_n
        H_bar = np.maximum(H_bar, 0.1)
        

        conv_u = self._convection_u(u_n, v_n)
        conv_v = self._convection_v(u_n, v_n)
        fric_u = self._friction_u(eta_n, u_n, v_n)
        fric_v = self._friction_v(eta_n, u_n, v_n)
        
        u_star = u_n + self.dt * (conv_u + fric_u)
        v_star = v_n + self.dt * (conv_v + fric_v)
        

        u_star = np.clip(u_star, -100.0, 100.0)
        v_star = np.clip(v_star, -100.0, 100.0)
        























        eta_p = eta_n.copy()
        u_new = u_n.copy()
        v_new = v_n.copy()
        

        eta_new = np.clip(eta_p, -50.0, 50.0)
        u_new = np.clip(u_new, -100.0, 100.0)
        v_new = np.clip(v_new, -100.0, 100.0)
        

        eta_new, u_new, v_new = self._apply_sponge_boundary(eta_new, u_new, v_new)
        
        return eta_new, u_new, v_new
    
    def solve(self, snapshot_interval=None):
        if snapshot_interval is None:
            snapshot_interval = max(1, self.n_steps // 12)
        
        t_snapshots = [0.0]
        eta_snapshots = [self.eta.copy()]
        u_snapshots = [self.u.copy()]
        v_snapshots = [self.v.copy()]
        
        eta_n = self.eta.copy()
        u_n = self.u.copy()
        v_n = self.v.copy()
        
        for step in range(1, self.n_steps + 1):
            eta_n, u_n, v_n = self._time_step(eta_n, u_n, v_n)
            

            if np.any(np.isnan(eta_n)) or np.any(np.isinf(eta_n)):
                print(f"  警告：step={step} 出现 NaN，重置为前一步")
                eta_n = eta_snapshots[-1].copy()
                u_n = u_snapshots[-1].copy()
                v_n = v_snapshots[-1].copy()
                continue
            
            if step % snapshot_interval == 0:
                t_snapshots.append(step * self.dt)
                eta_snapshots.append(eta_n.copy())
                u_snapshots.append(u_n.copy())
                v_snapshots.append(v_n.copy())
        

        if t_snapshots[-1] < self.n_steps * self.dt - 1e-12:
            t_snapshots.append(self.n_steps * self.dt)
            eta_snapshots.append(eta_n.copy())
            u_snapshots.append(u_n.copy())
            v_snapshots.append(v_n.copy())
        
        return t_snapshots, eta_snapshots, u_snapshots, v_snapshots
