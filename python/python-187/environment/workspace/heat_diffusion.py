#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
heat_diffusion.py
=================

基于种子项目 403_fem2d_heat 的热传导方程推荐模型。

科学背景
--------
将用户-物品交互网络视为一个连续介质，评分信息在其中扩散。
该过程由热传导方程描述:

    ∂u/∂t - α ∇²u + k(x,y,t)·u = f(x,y,t)    在 Ω 内
    
    u(x,y,t) = g(x,y,t)                        在 ∂Ω 上 (Dirichlet)
    
    u(x,y,0) = h(x,y)                          初始条件

其中:
    u(x,y,t) : 在位置 (x,y) 时刻 t 的评分密度
    α        : 热扩散系数（信息传播速度）
    k        : 反应系数（信息衰减/增强）
    f        : 热源项（外部评分注入）
    
时间离散采用后向 Euler 格式（无条件稳定）:
    (u^{n+1} - u^n)/Δt - α ∇²u^{n+1} + k u^{n+1} = f^{n+1}
    
    ⇒ (I - α·Δt·L + Δt·K) u^{n+1} = u^n + Δt·f^{n+1}
    
其中 L 是离散拉普拉斯算子（图拉普拉斯）。

在推荐系统中，我们将用户和物品投影到二维网格上，
让已知评分作为 Dirichlet 边界条件，通过热扩散填充缺失值。
"""

import numpy as np


class HeatDiffusionRecommender:
    """
    基于 2D 热传导方程的评分信息扩散推荐器。
    """
    
    def __init__(self, alpha=0.05, n_steps=8):
        """
        参数:
            alpha   : 热扩散系数 α，控制信息传播速度
            n_steps : 时间步数 N
        """
        self.alpha = max(alpha, 1e-6)
        self.n_steps = max(n_steps, 1)
    
    def _build_laplacian(self, R_obs):
        """
        构建图拉普拉斯矩阵的网格近似。
        
        对于网格上的函数 u_{i,j}，五点差分格式:
            (∇²u)_{i,j} ≈ (u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}) / h²
            
        边界处理:
            - 使用 Neumann 边界（零通量）处理网格边缘
            - 已知评分点作为 Dirichlet 条件固定
        """
        n, m = R_obs.shape
        h = 1.0  # 网格间距
        
        # 创建拉普拉斯算子作用的函数
        def apply_laplacian(u):
            Lu = np.zeros_like(u)
            # 内部点
            Lu[1:-1, 1:-1] = (
                u[2:, 1:-1] + u[:-2, 1:-1] +
                u[1:-1, 2:] + u[1:-1, :-2] - 4.0 * u[1:-1, 1:-1]
            ) / (h ** 2)
            # Neumann 边界（镜像反射）
            Lu[0, 1:-1] = (u[1, 1:-1] + u[0, 2:] + u[0, :-2] - 3.0 * u[0, 1:-1]) / (h ** 2)
            Lu[-1, 1:-1] = (u[-2, 1:-1] + u[-1, 2:] + u[-1, :-2] - 3.0 * u[-1, 1:-1]) / (h ** 2)
            Lu[1:-1, 0] = (u[2:, 0] + u[:-2, 0] + u[1:-1, 1] - 3.0 * u[1:-1, 0]) / (h ** 2)
            Lu[1:-1, -1] = (u[2:, -1] + u[:-2, -1] + u[1:-1, -2] - 3.0 * u[1:-1, -1]) / (h ** 2)
            # 角点
            Lu[0, 0] = (u[1, 0] + u[0, 1] - 2.0 * u[0, 0]) / (h ** 2)
            Lu[0, -1] = (u[1, -1] + u[0, -2] - 2.0 * u[0, -1]) / (h ** 2)
            Lu[-1, 0] = (u[-2, 0] + u[-1, 1] - 2.0 * u[-1, 0]) / (h ** 2)
            Lu[-1, -1] = (u[-2, -1] + u[-1, -2] - 2.0 * u[-1, -1]) / (h ** 2)
            return Lu
        
        return apply_laplacian
    
    def _solve_linear_system(self, A_op, b, mask_known, u_init, max_iter=100, tol=1e-6):
        """
        使用 Jacobi 迭代求解线性系统 A u = b，并施加 Dirichlet 条件。
        
        迭代格式:
            u^{(k+1)} = D^{-1} (b - (A - D) u^{(k)})
            
        边界处理:
            - 已知评分点始终保持固定值
        """
        u = np.copy(u_init)
        # 固定已知值
        u[mask_known] = b[mask_known]
        
        for _ in range(max_iter):
            u_old = np.copy(u)
            # Jacobi 半步: u_new = u_old + (b - A u_old) / diag_approx
            # 对角近似: 对于 (I - αΔt L)，对角元约等于 1 + 4αΔt/h²
            residual = b - u_old + self.alpha * self._laplacian_op(u_old) * self.dt
            diag = 1.0 + 4.0 * self.alpha * self.dt
            u = u_old + residual / diag
            
            # 固定 Dirichlet 条件
            u[mask_known] = b[mask_known]
            
            if np.linalg.norm(u - u_old) < tol:
                break
        
        return u
    
    def diffuse(self, R_obs):
        """
        执行热扩散过程填充缺失评分。
        
        算法:
            1. 识别已知评分作为 Dirichlet 边界
            2. 初始化: u⁰ = 已知值 或 全局均值
            3. 对每个时间步:
               a. 构建右端项: f = u^n + Δt · source
               b. 求解: (I - αΔt L) u^{n+1} = f
               c. 施加 Dirichlet 条件
            4. 返回最终状态
            
        边界保护:
            - 全缺失矩阵直接返回
            - 结果截断在 [1, 5]
        """
        R = np.array(R_obs, dtype=float)
        n, m = R.shape
        
        if n == 0 or m == 0:
            return R
        
        mask_known = ~np.isnan(R)
        if not np.any(mask_known):
            return np.full_like(R, 3.0)
        
        global_mean = np.nanmean(R)
        if np.isnan(global_mean):
            global_mean = 3.0
        
        # 初始化
        u = np.full((n, m), global_mean, dtype=float)
        u[mask_known] = R[mask_known]
        
        # 时间步长（CFL 条件）
        self.dt = 0.5 / max(self.n_steps, 1)
        
        # 构建拉普拉斯算子
        self._laplacian_op = self._build_laplacian(R)
        
        # 时间迭代
        for step in range(self.n_steps):
            # 右端项: 包含源项（已知评分的强化）
            source = np.zeros_like(u)
            source[mask_known] = (R[mask_known] - u[mask_known]) * 0.5
            rhs = u + self.dt * source
            
            # Jacobi 迭代求解
            u_new = np.copy(u)
            for _ in range(50):
                u_old = np.copy(u_new)
                Lu = self._laplacian_op(u_old)
                # TODO(Hole 1): 实现热传导方程 Jacobi 迭代核心步骤
                # 物理模型: (I - α·Δt·L) u^{n+1} = u^n + Δt·f^{n+1}
                # 离散拉普拉斯算子 L 已通过 _build_laplacian 构建
                # 需要实现: u_new = rhs + α * dt * L(u_old)
                # 然后施加 Dirichlet 边界条件 (已知评分保持不变)
                # 最后检查收敛: ||u_new - u_old|| < tol
                u_new = u_old  # 占位，需要替换为正确实现
                u_new[mask_known] = R[mask_known]
                
                if np.linalg.norm(u_new - u_old) < 1e-5:
                    break
            
            u = u_new
        
        # 数值鲁棒性
        u = np.clip(u, 1.0, 5.0)
        return u
