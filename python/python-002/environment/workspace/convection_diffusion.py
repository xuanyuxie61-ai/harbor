# -*- coding: utf-8 -*-
"""
convection_diffusion.py
基于 486_gray_scott_movie 合成
恒星对流区化学元素扩散-反应耦合PDE求解器。
使用显式时间推进与9点 Laplacian 模板（周期性/反射边界）。
"""

import numpy as np
from typing import Tuple, Optional, Callable


class ConvectionDiffusion:
    """
    对流区元素混合的扩散-反应方程求解器。
    
    控制方程（基于 Gray-Scott 反应-扩散思想的推广）：
      ∂X_i/∂t = D_mix ∇² X_i + R_i(X, T, ρ) + v_conv · ∇X_i
    
    在球坐标一维近似下（仅径向）：
      ∂X_i/∂t = (1/r²) ∂/∂r (r² D_mix ∂X_i/∂r) + R_i
    
    其中 D_mix 是对流混合扩散系数，采用 Bohm-Vitense 混合长度理论：
      D_mix ≈ (1/3) v_conv l_mix
      v_conv ≈ sqrt( (∇ - ∇_ad) g H_P / (8 Γ1) )
      l_mix ≈ α_MLT H_P
      H_P = P / (ρ g) (压强标高)
    """

    def __init__(self, n_points: int = 256, r_min: float = 1e8, r_max: float = 7e10):
        self.n_points = n_points
        self.r_min = r_min
        self.r_max = r_max
        # 对数径向坐标网格（在对流边界加密）
        self.r = np.logspace(np.log10(r_min), np.log10(r_max), n_points)
        self.dr = np.diff(self.r)
        self.dr = np.append(self.dr, self.dr[-1])

    def laplacian_spherical_1d(self, f: np.ndarray, r: np.ndarray) -> np.ndarray:
        """
        球坐标一维 Laplacian：
          ∇²f = (1/r²) d/dr(r² df/dr)
        使用中心差分：
          df/dr|_i ≈ (f_{i+1} - f_{i-1}) / (2Δr)
          d²f/dr²|_i ≈ (f_{i+1} - 2f_i + f_{i-1}) / Δr²
        ∇²f_i ≈ (f_{i+1} - 2f_i + f_{i-1})/Δr² + (2/r_i)(f_{i+1}-f_{i-1})/(2Δr)
        """
        f = np.asarray(f, dtype=np.float64)
        n = len(f)
        lap = np.zeros(n, dtype=np.float64)

        # 内部点
        for i in range(1, n - 1):
            dr = 0.5 * (r[i + 1] - r[i - 1])
            if dr == 0:
                continue
            d2f = (f[i + 1] - 2.0 * f[i] + f[i - 1]) / (dr ** 2)
            df = (f[i + 1] - f[i - 1]) / (2.0 * dr)
            lap[i] = d2f + (2.0 / r[i]) * df

        # 边界条件：内边界反射 (df/dr = 0)，外边界反射
        lap[0] = lap[1]
        lap[-1] = lap[-2]
        return lap

    def convective_diffusivity(self, r: np.ndarray, rho: np.ndarray, T: np.ndarray,
                               P: np.ndarray, nabla: float, nabla_ad: float,
                               alpha_mlt: float = 1.5) -> np.ndarray:
        """
        计算 Bohm-Vitense 混合长度理论的混合扩散系数 [cm^2/s]。
        
        g = G m(r) / r²
        H_P = P / (ρ g)
        v_conv² = g l_mix (∇ - ∇_ad) / (8 Γ1)
        D_mix = (1/3) v_conv l_mix
        """
        G = 6.67430e-8
        # 估算 m(r) ≈ (4/3)π r³ ρ_avg
        rho_avg = np.maximum(rho, 1e-10)
        m_r = (4.0 / 3.0) * np.pi * r ** 3 * rho_avg
        g = G * m_r / r ** 2
        g = np.maximum(g, 1e-5)

        H_P = P / (rho * g)
        H_P = np.clip(H_P, 1e6, 1e14)

        l_mix = alpha_mlt * H_P
        Gamma1 = 5.0 / 3.0  # 近似
        delta_nabla = max(nabla - nabla_ad, 0.0)
        v_conv_sq = g * l_mix * delta_nabla / (8.0 * Gamma1)
        v_conv_sq = np.maximum(v_conv_sq, 0.0)
        v_conv = np.sqrt(v_conv_sq)

        D_mix = (1.0 / 3.0) * v_conv * l_mix
        D_mix = np.clip(D_mix, 0.0, 1e18)
        return D_mix

    def solve_diffusion_step(self, X: np.ndarray, D: np.ndarray,
                             R: np.ndarray, dt: float, r: np.ndarray) -> np.ndarray:
        """
        显式 Euler 求解一个时间步的扩散-反应方程：
          X_new = X + dt * (D * ∇²X + R)
        
        CFL条件：dt < dr² / (2 D_max)
        """
        X = np.asarray(X, dtype=np.float64)
        D_arr = np.asarray(D, dtype=np.float64)
        R_arr = np.asarray(R, dtype=np.float64)
        r_arr = np.asarray(r, dtype=np.float64)

        lap_X = self.laplacian_spherical_1d(X, r_arr)
        # 变系数扩散: D(r) * ∇²X + ∇D · ∇X
        # 简化处理：使用 D 的局部值
        dXdt = D_arr * lap_X + R_arr

        # CFL安全检查
        dr_min = np.min(np.diff(r_arr))
        D_max = np.max(D_arr)
        if D_max > 0:
            dt_cfl = 0.5 * dr_min ** 2 / D_max
            if dt > dt_cfl:
                dt = dt_cfl

        X_new = X + dt * dXdt
        X_new = np.clip(X_new, 1e-15, 1.0)
        return X_new

    def solve_gray_scott_like(self, U: np.ndarray, V: np.ndarray,
                              Du: float, Dv: float, F: float, k: float,
                              dt: float, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        基于 Gray-Scott 模型的双组分反应-扩散系统。
        用于模拟对流区中两种化学元素（如 C 和 N）的转换。
        
        ∂U/∂t = D_u ∇²U - UV² + F(1-U)
        ∂V/∂t = D_v ∇²V + UV² - (F+k)V
        
        U: 主元素丰度（如 12C）
        V: 次元素丰度（如 14N）
        F: 对流翻转率（feeding rate）
        k: 核反应速率系数
        """
        U = np.asarray(U, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        r_arr = np.asarray(r, dtype=np.float64)

        lap_U = self.laplacian_spherical_1d(U, r_arr)
        lap_V = self.laplacian_spherical_1d(V, r_arr)

        UV2 = U * V ** 2
        dUdt = Du * lap_U - UV2 + F * (1.0 - U)
        dVdt = Dv * lap_V + UV2 - (F + k) * V

        # CFL
        dr_min = np.min(np.diff(r_arr))
        dt_cfl = 0.5 * dr_min ** 2 / max(Du, Dv, 1e-10)
        if dt > dt_cfl:
            dt = dt_cfl

        U_new = U + dt * dUdt
        V_new = V + dt * dVdt
        U_new = np.clip(U_new, 0.0, 1.0)
        V_new = np.clip(V_new, 0.0, 1.0)
        return U_new, V_new

    def mixing_timescale(self, r: np.ndarray, D: np.ndarray) -> float:
        """
        对流混合时标：τ_mix ≈ H_P² / D_mix
        """
        r = np.asarray(r, dtype=np.float64)
        D = np.asarray(D, dtype=np.float64)
        H_P = np.max(r) - np.min(r)  # 粗略估计
        D_avg = np.mean(D[D > 0]) if np.any(D > 0) else 1e10
        return H_P ** 2 / D_avg if D_avg > 0 else 1e20
