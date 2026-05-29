# -*- coding: utf-8 -*-
"""
polygon_roughness.py
壁材料表面粗糙度演化模块
基于种子项目 883_polygon_average (多边形平均) 重构

本模块模拟等离子体轰击下壁材料表面微观形貌的演化，
使用多边形顶点平均算法模拟表面粗糙度的平滑化与再生长过程。
"""

import numpy as np


class PolygonRoughness:
    """
    表面粗糙度演化模型
    
    基于 polygon_average 的顶点平均思想:
        1. 表面顶点通过与其邻居平均而平滑
        2. 侵蚀移除材料导致顶点向内部移动
        3. 再沉积导致新材料的随机堆积
    
    演化方程:
        h^{new} = (1-alpha) * h + alpha * avg(neighbors) - erosion_rate*dt + redeposition*dt
    
    其中 h 为表面高度剖面。
    """

    def __init__(self, n_vertices=100):
        """
        Parameters:
            n_vertices: 表面轮廓采样点数
        """
        if n_vertices < 4:
            raise ValueError("n_vertices 必须至少为 4")
        self.n = n_vertices
        self.x = np.linspace(0.0, 1.0, n_vertices)
        self.h = np.zeros(n_vertices)

    def initialize_random_roughness(self, amplitude=1.0e-6, n_modes=10, seed=None):
        """
        初始化随机粗糙表面
        
        使用傅里叶级数合成:
            h(x) = sum_{k=1}^{n_modes} A_k * sin(2*pi*k*x + phi_k)
        
        Parameters:
            amplitude: 最大粗糙度幅值 [m]
            n_modes:   傅里叶模式数
            seed:      随机数种子
        """
        if seed is not None:
            np.random.seed(seed)

        self.h = np.zeros(self.n)
        for k in range(1, n_modes + 1):
            A_k = amplitude / k  # 幅值随模式数衰减
            phi_k = np.random.uniform(0.0, 2.0 * np.pi)
            self.h += A_k * np.sin(2.0 * np.pi * k * self.x + phi_k)

        # 归一化
        max_h = np.max(np.abs(self.h))
        if max_h > 1.0e-20:
            self.h = self.h / max_h * amplitude

        return self

    def averaging_step(self, alpha=0.5):
        """
        执行一次顶点平均平滑步骤（基于 polygon_average 的核心算法）
        
        算法:
            h2[i] = (h[i] + h[i+1]) / 2    （与右邻居平均）
            h2 = h2 - mean(h2)             （移去平均值）
            h2 = h2 / max(abs(h2))         （归一化）
        
        Parameters:
            alpha: 平均权重 (0 = 无平滑, 1 = 完全平均)
        """
        if alpha < 0.0 or alpha > 1.0:
            alpha = max(0.0, min(1.0, alpha))

        # 循环邻居平均
        h2 = np.zeros(self.n)
        for i in range(self.n):
            ip1 = (i + 1) % self.n
            h2[i] = (1.0 - alpha) * self.h[i] + alpha * 0.5 * (self.h[i] + self.h[ip1])

        # 减去均值（模拟材料守恒）
        h2 = h2 - np.mean(h2)

        # 尺度归一化
        max_h = np.max(np.abs(h2))
        if max_h > 1.0e-20:
            h2 = h2 / max_h * np.max(np.abs(self.h))

        self.h = h2
        return self

    def apply_erosion(self, erosion_rate, dt, material_removal_func=None):
        """
        应用侵蚀效应
        
        材料移除与局部曲率和入射离子通量成正比:
            dh/dt = -erosion_rate * (1 + kappa * |curvature|)
        
        Parameters:
            erosion_rate: 基准侵蚀速率 [m/s]
            dt:           时间步长 [s]
            material_removal_func: 可选的自定义材料移除函数
        """
        if erosion_rate < 0:
            erosion_rate = 0.0
        if dt < 0:
            dt = 0.0

        # 计算局部曲率（二阶导数近似）
        curvature = np.zeros(self.n)
        dx = self.x[1] - self.x[0]
        if dx > 1.0e-20:
            for i in range(1, self.n - 1):
                curvature[i] = (self.h[i+1] - 2*self.h[i] + self.h[i-1]) / (dx*dx)
            # 循环边界
            curvature[0] = (self.h[1] - 2*self.h[0] + self.h[-1]) / (dx*dx)
            curvature[-1] = (self.h[0] - 2*self.h[-1] + self.h[-2]) / (dx*dx)

        # 侵蚀量
        if material_removal_func is not None:
            dh = material_removal_func(self.x, self.h, curvature) * dt
        else:
            enhancement = 1.0 + 0.5 * np.abs(curvature) / (np.max(np.abs(curvature)) + 1.0e-20)
            dh = -erosion_rate * enhancement * dt

        self.h += dh
        return self

    def apply_redeposition(self, redeposition_rate, dt, stochastic=True):
        """
        应用再沉积效应（侵蚀原子返回表面）
        
        再沉积通常具有随机性，用噪声项模拟:
            dh/dt = redeposition_rate + stochastic_noise
        
        Parameters:
            redeposition_rate: 基准再沉积速率 [m/s]
            dt:                时间步长 [s]
            stochastic:        是否添加随机噪声
        """
        if redeposition_rate < 0:
            redeposition_rate = 0.0
        if dt < 0:
            dt = 0.0

        dh = redeposition_rate * dt * np.ones(self.n)

        if stochastic:
            noise_amplitude = 0.3 * redeposition_rate * dt
            dh += noise_amplitude * (2.0 * np.random.rand(self.n) - 1.0)

        self.h += dh
        return self

    def compute_roughness_parameters(self):
        """
        计算表面粗糙度统计参数
        
        Returns:
            Ra:  算术平均粗糙度 [m]
            Rq:  均方根粗糙度 [m]
            Rz:  最大高度 [m]
            skewness: 偏度
            kurtosis: 峰度
        """
        h_centered = self.h - np.mean(self.h)
        n = float(self.n)

        Ra = np.mean(np.abs(h_centered))
        Rq = np.sqrt(np.mean(h_centered**2))
        Rz = np.max(self.h) - np.min(self.h)

        if Rq > 1.0e-30:
            skewness = np.mean(h_centered**3) / (Rq**3)
            kurtosis = np.mean(h_centered**4) / (Rq**4)
        else:
            skewness = 0.0
            kurtosis = 3.0

        return {
            'Ra': Ra,
            'Rq': Rq,
            'Rz': Rz,
            'skewness': skewness,
            'kurtosis': kurtosis,
        }

    def evolve_surface(self, n_steps, erosion_rate, redeposition_rate, dt,
                       alpha_smooth=0.3, stochastic=True):
        """
        多步表面演化模拟
        
        Parameters:
            n_steps:            时间步数
            erosion_rate:       基准侵蚀速率 [m/s]
            redeposition_rate:  再沉积速率 [m/s]
            dt:                 时间步长 [s]
            alpha_smooth:       平滑系数
            stochastic:         是否启用随机再沉积
        
        Returns:
            history: 粗糙度参数演化历史
        """
        history = []
        for step in range(n_steps):
            self.averaging_step(alpha=alpha_smooth)
            self.apply_erosion(erosion_rate, dt)
            self.apply_redeposition(redeposition_rate, dt, stochastic=stochastic)

            stats = self.compute_roughness_parameters()
            stats['step'] = step
            history.append(stats)

        return history


def demo_roughness():
    """演示粗糙度演化"""
    surface = PolygonRoughness(n_vertices=128)
    surface.initialize_random_roughness(amplitude=1.0e-6, n_modes=15, seed=42)

    init_stats = surface.compute_roughness_parameters()
    print("初始粗糙度参数:")
    for k, v in init_stats.items():
        print(f"  {k}: {v:.3e}")

    # 演化
    history = surface.evolve_surface(
        n_steps=50,
        erosion_rate=1.0e-9,
        redeposition_rate=3.0e-10,
        dt=1.0e-3,
        alpha_smooth=0.2,
        stochastic=True
    )

    final_stats = surface.compute_roughness_parameters()
    print("\n最终粗糙度参数:")
    for k, v in final_stats.items():
        print(f"  {k}: {v:.3e}")

    return surface, history


if __name__ == "__main__":
    demo_roughness()
