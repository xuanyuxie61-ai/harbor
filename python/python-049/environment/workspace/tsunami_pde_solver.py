#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 tsunami_pde_solver.py
 
 融合种子项目：
   - 064_backward_euler_fixed：固定点迭代后向 Euler 时间积分
   - 857_pendulum_comparison_ode：非线性 ODE 守恒量监测思想
 
 科学功能：
   非线性浅水波方程（Saint-Venant 方程组）数值求解器。
   
   使用交错网格（Arakawa C-grid）有限差分离散空间，
   固定点迭代后向 Euler 隐式时间积分。
 
 核心物理公式：
 
   1) 非线性浅水波方程（二维）：
   
      连续性方程：
        ∂η/∂t + ∇·[(h + η) · u] = 0
        
      动量方程（x方向）：
        ∂u/∂t + u·∂u/∂x + v·∂u/∂y + g·∂η/∂x + τ_{bx}/(ρ(h+η)) = 0
        
      动量方程（y方向）：
        ∂v/∂t + u·∂v/∂x + v·∂v/∂y + g·∂η/∂y + τ_{by}/(ρ(h+η)) = 0
   
   2) 底摩擦（二次摩擦定律）：
      τ_{bx} = ρ · C_d · |u| · u
      τ_{by} = ρ · C_d · |u| · v
      
   3) 数值方法：
      - 空间：交错网格有限差分
      - 时间：线性化后向 Euler（分裂格式）
        
        第 1 步（动量预测，显式）：
          u* = u^n + dt · [-u^n·∂u^n/∂x - v^n·∂u^n/∂y - τ_{bx}^n/(ρH^n)]
          v* = v^n + dt · [-u^n·∂v^n/∂x - v^n·∂v^n/∂y - τ_{by}^n/(ρH^n)]
        
        第 2 步（压强修正，隐式/半隐式）：
          u^{n+1} = u* - dt·g·∂η^{n+1}/∂x
          v^{n+1} = v* - dt·g·∂η^{n+1}/∂y
        
        第 3 步（连续性方程，代入后得到 η 的椭圆方程）：
          η^{n+1} - dt²·g·∇·(H·∇η^{n+1}) = η^n - dt·∇·(H·u*)
        
        对椭圆方程使用固定点迭代求解（来源于 backward_euler_fixed）。
        
      - 该格式对线性波部分无条件稳定，非线性部分受 CFL 限制。
"""

import numpy as np


class ShallowWaterSolver:
    """
    非线性浅水波方程求解器。
    
    采用交错网格有限差分和分裂时间积分格式。
    """
    
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
        
        # 自适应时间步：取用户指定和 CFL 限制的较小值
        c_max = np.sqrt(g * np.max(h_bathy))
        dt_cfl = 0.5 * min(self.dx, self.dy) / c_max
        self.dt = min(dt, dt_cfl)
        # 保持总模拟时长不变
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
        """
        计算通量散度 ∇·[(h+η)·u]。
        """
        H = self.h_bathy + eta
        H = np.maximum(H, 0.1)
        
        # x 方向通量
        H_at_u = 0.5 * (H + np.roll(H, -1, axis=1))
        flux_x = H_at_u * u
        d_flux_x = (flux_x - np.roll(flux_x, 1, axis=1)) / self.dx
        
        # y 方向通量
        H_at_v = 0.5 * (H + np.roll(H, -1, axis=0))
        flux_y = H_at_v * v
        d_flux_y = (flux_y - np.roll(flux_y, 1, axis=0)) / self.dy
        
        return d_flux_x + d_flux_y
    
    def _convection_u(self, u, v):
        """
        x 方向动量的对流项（迎风格式）。
        """
        # 迎风格式
        du_dx = np.zeros_like(u)
        du_dy = np.zeros_like(u)
        
        # x 方向
        for j in range(self.ny):
            for i in range(1, self.nx - 1):
                if u[j, i] >= 0:
                    du_dx[j, i] = (u[j, i] - u[j, i-1]) / self.dx
                else:
                    du_dx[j, i] = (u[j, i+1] - u[j, i]) / self.dx
        
        # y 方向
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
        """
        y 方向动量的对流项（迎风格式）。
        """
        dv_dx = np.zeros_like(v)
        dv_dy = np.zeros_like(v)
        
        # x 方向
        for j in range(self.ny):
            for i in range(1, self.nx - 1):
                u_avg = 0.25 * (u[j, i] + u[j, min(self.nx-1, i+1)] + u[max(0, j-1), i] + u[max(0, j-1), min(self.nx-1, i+1)])
                if u_avg >= 0:
                    dv_dx[j, i] = (v[j, i] - v[j, i-1]) / self.dx
                else:
                    dv_dx[j, i] = (v[j, i+1] - v[j, i]) / self.dx
        
        # y 方向
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
        """
        x 方向底摩擦。
        """
        H = self.h_bathy + eta
        H = np.maximum(H, 0.1)
        
        v_at_u = np.zeros_like(u)
        for j in range(self.ny):
            for i in range(self.nx):
                jm = max(0, j-1)
                im = max(0, i-1)
                v_at_u[j, i] = 0.25 * (v[j, i] + v[j, im] + v[jm, i] + v[jm, im])
        
        speed = np.sqrt(u**2 + v_at_u**2)
        speed = np.minimum(speed, 50.0)  # 限制最大流速
        return -self.Cd * speed * u / H
    
    def _friction_v(self, eta, u, v):
        """
        y 方向底摩擦。
        """
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
        """
        x 方向压强梯度：-g·∂η/∂x。
        """
        d_eta = np.zeros_like(eta)
        d_eta[:, :-1] = (eta[:, 1:] - eta[:, :-1]) / self.dx
        return -self.g * d_eta
    
    def _pressure_gradient_y(self, eta):
        """
        y 方向压强梯度：-g·∂η/∂y。
        """
        d_eta = np.zeros_like(eta)
        d_eta[:-1, :] = (eta[1:, :] - eta[:-1, :]) / self.dy
        return -self.g * d_eta
    
    def _laplacian_eta(self, eta, H_bar):
        """
        计算 ∇·(H·∇η) 的离散形式。
        """
        lap = np.zeros_like(eta)
        
        for j in range(1, self.ny - 1):
            for i in range(1, self.nx - 1):
                # x 方向
                H_right = 0.5 * (H_bar[j, i] + H_bar[j, i+1])
                H_left = 0.5 * (H_bar[j, i] + H_bar[j, i-1])
                d_eta_right = (eta[j, i+1] - eta[j, i]) / self.dx
                d_eta_left = (eta[j, i] - eta[j, i-1]) / self.dx
                lap[j, i] += (H_right * d_eta_right - H_left * d_eta_left) / self.dx
                
                # y 方向
                H_top = 0.5 * (H_bar[j, i] + H_bar[j+1, i])
                H_bottom = 0.5 * (H_bar[j, i] + H_bar[j-1, i])
                d_eta_top = (eta[j+1, i] - eta[j, i]) / self.dy
                d_eta_bottom = (eta[j, i] - eta[j-1, i]) / self.dy
                lap[j, i] += (H_top * d_eta_top - H_bottom * d_eta_bottom) / self.dy
        
        return lap
    
    def _apply_sponge_boundary(self, eta, u, v):
        """
        应用 sponge 吸收边界。
        """
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
        """
        一个时间步（分裂格式 + 固定点迭代）。
        
        来源于 backward_euler_fixed 的固定点迭代思想：
          对椭圆方程 η^{n+1} - dt²·g·∇·(H·∇η^{n+1}) = RHS
          使用固定点迭代求解 η^{n+1}。
        """
        H_bar = self.h_bathy + eta_n
        H_bar = np.maximum(H_bar, 0.1)
        
        # 第 1 步：动量预测（显式处理对流和摩擦）
        conv_u = self._convection_u(u_n, v_n)
        conv_v = self._convection_v(u_n, v_n)
        fric_u = self._friction_u(eta_n, u_n, v_n)
        fric_v = self._friction_v(eta_n, u_n, v_n)
        
        u_star = u_n + self.dt * (conv_u + fric_u)
        v_star = v_n + self.dt * (conv_v + fric_v)
        
        # 限制预测速度
        u_star = np.clip(u_star, -100.0, 100.0)
        v_star = np.clip(v_star, -100.0, 100.0)
        
        # ============================================
        # [HOLE 1] 分裂格式：压强修正步 + 速度更新
        # ============================================
        # 此处被挖空，需要实现以下核心科学计算：
        # 
        # 1. 构造椭圆方程 RHS = η^n - dt·∇·(H·u*)
        #    提示：使用 self._divergence_flux(eta_n, u_star, v_star)
        #
        # 2. 固定点迭代求解 η^{n+1}：
        #    (I - dt²·g·L) η^{n+1} = RHS
        #    迭代格式：η^{k+1} = RHS + dt²·g·L(η^k)
        #    提示：使用 self._laplacian_eta(eta_p, H_bar)
        #
        # 3. 速度修正：
        #    u^{n+1} = u* + dt · pgx(η^{n+1})
        #    v^{n+1} = v* + dt · pgy(η^{n+1})
        #    提示：pgx = self._pressure_gradient_x(eta_p)
        #          注意 _pressure_gradient_x 返回的是 -g·∂η/∂x
        #
        # 物理约束：该分裂格式必须与能量守恒检验一致。
        # 若此处压强梯度处理方式改变，energy_quadrature.py
        # 中的能量公式也需同步调整。
        # ============================================
        eta_p = eta_n.copy()
        u_new = u_n.copy()
        v_new = v_n.copy()
        
        # 最终限制
        eta_new = np.clip(eta_p, -50.0, 50.0)
        u_new = np.clip(u_new, -100.0, 100.0)
        v_new = np.clip(v_new, -100.0, 100.0)
        
        # 应用边界
        eta_new, u_new, v_new = self._apply_sponge_boundary(eta_new, u_new, v_new)
        
        return eta_new, u_new, v_new
    
    def solve(self, snapshot_interval=None):
        """
        运行完整模拟。
        """
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
            
            # 数值稳定性检查
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
        
        # 确保最后一个状态被保存
        if t_snapshots[-1] < self.n_steps * self.dt - 1e-12:
            t_snapshots.append(self.n_steps * self.dt)
            eta_snapshots.append(eta_n.copy())
            u_snapshots.append(u_n.copy())
            v_snapshots.append(v_n.copy())
        
        return t_snapshots, eta_snapshots, u_snapshots, v_snapshots
