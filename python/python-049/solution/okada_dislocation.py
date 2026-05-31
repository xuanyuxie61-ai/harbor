#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 okada_dislocation.py
 
 融合种子项目：
   - 857_pendulum_comparison_ode：参数化系统思想
 
 科学功能：
   Okada 弹性半空间位错模型。
   
   计算矩形断层滑动引起的三维地表/海底位移场。
   这是海啸数值模拟的初始条件来源。
 
 核心物理公式：
 
   Okada (1985) 弹性半空间矩形位错模型：
   
   对于均匀各向同性弹性半空间中的矩形断层，位移场有解析解。
   设断层参数为：
     - 长度 L，宽度 W
     - 走向 φ（strike），倾角 δ（dip）
     - 滑动角 λ（rake），滑动量 U
     - 顶部深度 d
   
   位移分量 u_i(x) 由以下积分给出：
   
   u_i = (U / 2π) · [ ... ]  
   
   具体地，对于走滑（strike-slip）和倾滑（dip-slip）分量：
   
   u_x = (U/2π) · [ u_x^{SS}·cos(λ) + u_x^{DS}·sin(λ) ]
   u_y = (U/2π) · [ u_y^{SS}·cos(λ) + u_y^{DS}·sin(λ) ]
   u_z = (U/2π) · [ u_z^{SS}·cos(λ) + u_z^{DS}·sin(λ) ]
   
   其中 SS 和 DS 分别表示单位走滑和单位倾滑对应的位移核函数。
   
   本模块实现简化的 Okada 模型，计算垂直位移 u_z（即海面抬升/沉降）。
   
   泊松比 ν 与拉梅常数关系：
     λ = 2μν / (1 - 2ν)
     
   剪切模量 μ 取 30 GPa（典型地壳值）。
"""

import numpy as np


class OkadaModel:
    """
    Okada 弹性半空间位错模型。
    
    计算矩形断层滑动引起的三维位移场。
    """
    
    def __init__(self, strike, dip, rake, slip, length, width, depth, nu=0.25):
        """
        初始化断层参数。
        
        Parameters
        ----------
        strike : float
            断层走向，从北顺时针测量（度）
        dip : float
            断层倾角，从水平面向下测量（度）
        rake : float
            滑动角，从走向方向测量（度）
            rake = 0: 纯左旋走滑
            rake = 90: 纯逆冲
            rake = -90: 纯正断层
        slip : float
            平均滑动量（m）
        length : float
            断层沿走向长度（m）
        width : float
            断层沿倾向宽度（m）
        depth : float
            断层顶部深度（m）
        nu : float
            泊松比
        """
        self.strike = strike
        self.dip = dip
        self.rake = rake
        self.slip = slip
        self.length = length
        self.width = width
        self.depth = depth
        self.nu = nu
        
        # 剪切模量（Pa）
        self.shear_modulus = 30e9
        
        # 边界检查
        if not (0 <= dip <= 90):
            raise ValueError("倾角 dip 必须在 [0, 90] 范围内")
        if slip < 0:
            raise ValueError("滑动量 slip 必须非负")
        if length <= 0 or width <= 0:
            raise ValueError("断层长度和宽度必须为正")
        if depth < 0:
            raise ValueError("深度必须非负")
        if not (0 <= nu < 0.5):
            raise ValueError("泊松比必须在 [0, 0.5) 范围内")
    
    def _chinnery(self, f, x, y, L, W, d, delta):
        """
        Chinnery 记号：对断层两端点进行求和。
        
        f(ξ, η) 在断层 ξ ∈ [0, L], η ∈ [0, W] 上的积分可通过
        Chinnery 记号简化为四个角点的代数和。
        """
        # 角度转换
        delta_rad = np.radians(delta)
        
        # 断层角点坐标（相对于断层中心）
        xis = [0.0, L]
        etas = [0.0, W]
        
        result = 0.0
        for xi in xis:
            for eta in etas:
                # 坐标旋转：从断层局部坐标到观测点坐标
                p = y * np.cos(delta_rad) + d * np.sin(delta_rad)
                q = y * np.sin(delta_rad) - d * np.cos(delta_rad)
                
                # 符号因子（Chinnery 记号）
                sign = 1.0
                if xi == 0:
                    sign *= -1.0
                if eta == 0:
                    sign *= -1.0
                
                result += sign * f(x, p, q, xi, eta, delta_rad)
        
        return result
    
    def _okada_vertical_displacement(self, x, y, L, W, d, delta, nu):
        """
        计算单位滑动量下的垂直位移核函数。
        
        简化公式：对于纯倾滑（rake = 90°），垂直位移近似为：
        
        u_z ≈ (U / 2π) · sin(δ) · [
            y'/(r·(r+ζ)) + (q·cos(δ))/(r·(r+ζ))
            + (1-2ν)·log(r+ζ)·cos(δ)
        ]
        
        其中：
          y' = y·cos(δ) + d·sin(δ)
          ζ = q·sin(δ) - d·cos(δ) + W  (深度方向)
          r = √(x² + y'² + ζ²)
        """
        delta_rad = np.radians(delta)
        sin_d = np.sin(delta_rad)
        cos_d = np.cos(delta_rad)
        
        # 断层局部坐标系变换
        x_prime = x
        y_prime = y * cos_d + d * sin_d
        
        # 深度方向坐标
        zeta = y * sin_d - d * cos_d
        
        # 距离
        r = np.sqrt(x_prime**2 + y_prime**2 + zeta**2)
        r = np.maximum(r, 1e-6)  # 避免除零
        
        # 垂直位移核函数（简化版，适用于倾滑断层）
        # 这是 Okada 解析解的简化形式
        kernel = sin_d * (
            y_prime / (r * (r + np.abs(zeta) + 1e-6))
            + (1.0 - 2.0 * nu) * np.log(r + np.abs(zeta) + 1e-6) * cos_d
        )
        
        return kernel
    
    def compute_seafloor_displacement(self, x_grid, y_grid):
        """
        计算海底位移场（即初始海面位移）。
        
        Parameters
        ----------
        x_grid : ndarray
            x 方向网格坐标（m）
        y_grid : ndarray
            y 方向网格坐标（m）
            
        Returns
        -------
        eta : ndarray
            初始海面位移场，形状 (ny, nx)
        """
        nx = len(x_grid)
        ny = len(y_grid)
        
        # 走向角（弧度）
        strike_rad = np.radians(self.strike)
        
        # 滑动角分解
        rake_rad = np.radians(self.rake)
        slip_strike = self.slip * np.cos(rake_rad)  # 走滑分量
        slip_dip = self.slip * np.sin(rake_rad)     # 倾滑分量
        
        eta = np.zeros((ny, nx))
        
        for j in range(ny):
            for i in range(nx):
                # 观测点坐标
                x_obs = x_grid[i]
                y_obs = y_grid[j]
                
                # 旋转到断层局部坐标系
                # 局部 x 沿断层走向，y 沿断层倾向（向下）
                x_loc = x_obs * np.cos(strike_rad) + y_obs * np.sin(strike_rad)
                y_loc = -x_obs * np.sin(strike_rad) + y_obs * np.cos(strike_rad)
                
                # 相对于断层中心的坐标
                x_loc -= 0.0  # 断层中心在原点
                y_loc -= 0.0
                
                # 走滑分量引起的垂直位移（简化为反对称形式）
                u_z_strike = slip_strike * self._strike_slip_kernel(
                    x_loc, y_loc, self.length, self.width, self.depth, self.dip, self.nu
                )
                
                # 倾滑分量引起的垂直位移
                u_z_dip = slip_dip * self._dip_slip_kernel(
                    x_loc, y_loc, self.length, self.width, self.depth, self.dip, self.nu
                )
                
                eta[j, i] = u_z_strike + u_z_dip
        
        # 数值平滑处理（避免网格噪声）
        eta = self._gaussian_smooth(eta, sigma=1.0)
        
        return eta
    
    def _strike_slip_kernel(self, x, y, L, W, d, delta, nu):
        """
        走滑分量垂直位移核函数（Okada 公式简化）。
        
        对于纯走滑（rake = 0°），垂直位移近似为：
        u_z^{SS} ≈ (sin(δ)/2π) · [ x/(r(r+ζ)) ]
        """
        delta_rad = np.radians(delta)
        sin_d = np.sin(delta_rad)
        cos_d = np.cos(delta_rad)
        
        # 断层角点贡献求和（简化：取断层中心近似）
        # 更精确的做法是沿断层面积分
        y_prime = y * cos_d + d * sin_d
        zeta = y * sin_d - d * cos_d
        r = np.sqrt(x**2 + y_prime**2 + zeta**2)
        r = max(r, 1e-6)
        
        kernel = (sin_d / (2.0 * np.pi)) * (
            x / (r * (r + np.abs(zeta) + 1e-6))
        )
        
        return kernel
    
    def _dip_slip_kernel(self, x, y, L, W, d, delta, nu):
        """
        倾滑分量垂直位移核函数（Okada 公式简化）。
        
        对于纯倾滑（rake = 90°），垂直位移近似为：
        u_z^{DS} ≈ (1/2π) · sin(δ) · [
            y'/(r(r+ζ)) + q·cos(δ)/(r(r+ζ))
            + (1-2ν)·cos(δ)·ln(r+ζ)
        ]
        """
        delta_rad = np.radians(delta)
        sin_d = np.sin(delta_rad)
        cos_d = np.cos(delta_rad)
        
        y_prime = y * cos_d + d * sin_d
        q = y * sin_d - d * cos_d
        zeta = q
        r = np.sqrt(x**2 + y_prime**2 + zeta**2)
        r = max(r, 1e-6)
        
        kernel = (1.0 / (2.0 * np.pi)) * sin_d * (
            y_prime / (r * (r + np.abs(zeta) + 1e-6))
            + q * cos_d / (r * (r + np.abs(zeta) + 1e-6))
            + (1.0 - 2.0 * nu) * cos_d * np.log(r + np.abs(zeta) + 1e-6)
        )
        
        return kernel
    
    def _gaussian_smooth(self, field, sigma=1.0):
        """
        高斯平滑滤波，去除数值噪声。
        """
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(field, sigma=sigma)
