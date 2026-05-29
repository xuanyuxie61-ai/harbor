"""
nuclear_geometry.py
重离子碰撞核几何与Woods-Saxon密度分布建模

基于种子项目:
- 1282_tortoise: 边界字追踪思想 → 核表面边界参数化
- 874_ply_to_tri_surface: 三角网格表面数据思想 → 核密度等值面离散化

物理模型:
1. Woods-Saxon 核密度分布:
   ρ(r) = ρ₀ / (1 + exp((r - R) / a))
   其中 R = r₀ A^(1/3), a ≈ 0.52 fm

2. Glauber模型重叠函数:
   T_A(x,y) = ∫ dz ρ_A(x,y,z)
   T_AB(b) = ∫ dxdy T_A(x,y) T_B(x-bx, y-by)

3. 碰撞几何参数:
   - 碰撞参数 b
   - 参与者核子数 N_part
   - 二叉碰撞数 N_coll
   - 偏心距 ε₂
"""

import numpy as np
from typing import Tuple, List


class NuclearGeometry:
    """
    重离子碰撞核几何模型类。
    """

    def __init__(self, mass_number_a: int = 197, mass_number_b: int = 197,
                 radius_param: float = 1.12, diffuseness: float = 0.54,
                 nucleon_cross_section: float = 4.2):
        """
        初始化核几何参数。

        Parameters
        ----------
        mass_number_a : int
            核A的质量数 (默认197 for Au)
        mass_number_b : int
            核B的质量数
        radius_param : float
            半径参数 r₀ [fm]
        diffuseness : float
            Woods-Saxon表面弥散参数 a [fm]
        nucleon_cross_section : float
            核子-核子非弹性散射截面 σ_nn [mb]
        """
        self.A = mass_number_a
        self.B = mass_number_b
        self.r0 = radius_param
        self.a = diffuseness
        self.sigma_nn = nucleon_cross_section  # mb
        self.R_A = self.r0 * (self.A ** (1.0 / 3.0))
        self.R_B = self.r0 * (self.B ** (1.0 / 3.0))
        # 最大密度 ρ₀，由归一化条件 ∫ρ(r)d³r = A 确定
        self.rho0 = self._compute_rho0(self.A, self.R_A, self.a)

    def _compute_rho0(self, A: int, R: float, a: float) -> float:
        """
        计算Woods-Saxon分布的归一化常数 ρ₀。
        近似: ρ₀ ≈ 3A / (4π R³ (1 + π² a² / R²))
        """
        correction = 1.0 + (np.pi ** 2) * (a ** 2) / (R ** 2)
        rho0 = 3.0 * A / (4.0 * np.pi * (R ** 3) * correction)
        return rho0

    def woods_saxon_density(self, r: np.ndarray) -> np.ndarray:
        """
        Woods-Saxon核密度分布。

        ρ(r) = ρ₀ / [1 + exp((r - R) / a)]

        Parameters
        ----------
        r : np.ndarray
            径向距离 [fm]

        Returns
        -------
        np.ndarray
            密度值 [核子/fm³]
        """
        r = np.asarray(r)
        # 边界处理：防止溢出
        exponent = (r - self.R_A) / self.a
        # 对极大值进行截断
        exponent = np.clip(exponent, -700, 700)
        density = self.rho0 / (1.0 + np.exp(exponent))
        return density

    def thickness_function(self, x: np.ndarray, y: np.ndarray,
                           nucleus: str = 'A') -> np.ndarray:
        """
        计算厚度函数 T(x,y) = ∫ dz ρ(x,y,z)。

        通过数值积分实现: T(x,y) = ∫_{-z_max}^{z_max} ρ(√(x²+y²+z²)) dz

        Parameters
        ----------
        x, y : np.ndarray
            横向坐标 [fm]
        nucleus : str
            'A' 或 'B'

        Returns
        -------
        np.ndarray
            厚度函数值 [核子/fm²]
        """
        if nucleus == 'A':
            R = self.R_A
        else:
            R = self.R_B

        x = np.asarray(x)
        y = np.asarray(y)
        s2 = x ** 2 + y ** 2
        # 积分上限：取 5 倍弥散参数 + 核半径
        z_max = R + 10.0 * self.a
        n_z = 200
        z_grid = np.linspace(-z_max, z_max, n_z)
        dz = z_grid[1] - z_grid[0]

        # 广播运算
        r = np.sqrt(s2[..., np.newaxis] + z_grid ** 2)
        if nucleus == 'A':
            rho_vals = self.woods_saxon_density(r.flatten()).reshape(r.shape)
        else:
            # 对B核使用相同参数（假设对称碰撞）
            rho_vals = self.woods_saxon_density(r.flatten()).reshape(r.shape)

        thickness = np.trapezoid(rho_vals, z_grid, axis=-1)
        return thickness

    def overlap_function(self, b: float, x_grid: np.ndarray,
                         y_grid: np.ndarray) -> np.ndarray:
        """
        计算碰撞参数为 b 时的重叠函数。

        T_AB(x,y; b) = T_A(x,y) · T_B(x - b, y)

        Parameters
        ----------
        b : float
            碰撞参数 [fm]
        x_grid, y_grid : np.ndarray
            横向网格

        Returns
        -------
        np.ndarray
            重叠函数分布
        """
        X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
        T_A = self.thickness_function(X, Y, 'A')
        T_B = self.thickness_function(X - b, Y, 'B')
        T_AB = T_A * T_B
        return T_AB

    def compute_npart_ncoll(self, b: float, x_grid: np.ndarray,
                            y_grid: np.ndarray) -> Tuple[float, float]:
        """
        计算参与者数 N_part 和碰撞数 N_coll（Glauber蒙特卡洛解析近似）。

        N_part(b) = A ∫ T_A(x,y) [1 - (1 - σ_nn T_B(x-b,y)/B)^B] dxdy
                  + B ∫ T_B(x-b,y) [1 - (1 - σ_nn T_A(x,y)/A)^A] dxdy

        N_coll(b) = σ_nn ∫ T_A(x,y) T_B(x-b,y) dxdy

        Parameters
        ----------
        b : float
            碰撞参数 [fm]
        x_grid, y_grid : np.ndarray
            横向网格

        Returns
        -------
        Tuple[float, float]
            (N_part, N_coll)
        """
        dx = x_grid[1] - x_grid[0]
        dy = y_grid[1] - y_grid[0]
        X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')

        T_A = self.thickness_function(X, Y, 'A')
        T_B = self.thickness_function(X - b, Y, 'B')

        # 转换 σ_nn 从 mb 到 fm²: 1 mb = 0.1 fm²
        sigma_fm2 = self.sigma_nn * 0.1

        # 参与者数
        term_A = 1.0 - (1.0 - sigma_fm2 * T_B / self.B) ** self.B
        term_B = 1.0 - (1.0 - sigma_fm2 * T_A / self.A) ** self.A

        N_part = (np.trapezoid(np.trapezoid(T_A * term_A, y_grid, axis=1),
                           x_grid, axis=0) +
                  np.trapezoid(np.trapezoid(T_B * term_B, y_grid, axis=1),
                           x_grid, axis=0))

        # 二叉碰撞数
        N_coll = sigma_fm2 * np.trapezoid(np.trapezoid(T_A * T_B, y_grid, axis=1),
                                      x_grid, axis=0)

        return float(N_part), float(N_coll)

    def eccentricity(self, b: float, x_grid: np.ndarray,
                     y_grid: np.ndarray) -> Tuple[float, float]:
        """
        计算重叠区域的二阶和四阶偏心距。

        ε₂ = √(⟨x² - y²⟩² + ⟨2xy⟩²) / ⟨r²⟩
        ε₄ = ⟨r⁴ cos(4φ)⟩ / ⟨r⁴⟩

        其中 ⟨·⟩ 表示以 T_AB 为权重的平均。

        Parameters
        ----------
        b : float
            碰撞参数 [fm]
        x_grid, y_grid : np.ndarray
            横向网格

        Returns
        -------
        Tuple[float, float]
            (ε₂, ε₄)
        """
        X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
        T_AB = self.overlap_function(b, x_grid, y_grid)

        dx = x_grid[1] - x_grid[0]
        dy = y_grid[1] - y_grid[0]
        dA = dx * dy

        total = np.sum(T_AB) * dA
        if total < 1e-15:
            return 0.0, 0.0

        # 二阶矩
        x2 = np.sum(X ** 2 * T_AB) * dA / total
        y2 = np.sum(Y ** 2 * T_AB) * dA / total
        xy = np.sum(X * Y * T_AB) * dA / total
        r2 = x2 + y2

        # 四阶矩
        r4 = np.sum((X ** 2 + Y ** 2) ** 2 * T_AB) * dA / total
        cos4phi = np.sum(((X ** 2 + Y ** 2) ** 2) * np.cos(4.0 * np.arctan2(Y, X)) * T_AB) * dA

        eps2 = np.sqrt((x2 - y2) ** 2 + 4.0 * xy ** 2) / r2 if r2 > 1e-15 else 0.0
        eps4 = cos4phi / (r4 * total) if r4 > 1e-15 else 0.0

        return float(eps2), float(eps4)

    def tortoise_boundary_word(self, n_segments: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        """
        基于tortoise边界字思想，生成核表面边界的多边形近似。

        将核表面按极角 φ 离散化为 n_segments 段，
        每段对应一个边界步长。

        Parameters
        ----------
        n_segments : int
            边界分段数

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            边界点坐标 (x, y)
        """
        phi = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)
        # 核表面近似半径
        r_surface = self.R_A + self.a * np.log(1.0 / 0.05 - 1.0)
        x = r_surface * np.cos(phi)
        y = r_surface * np.sin(phi)
        return x, y

    def participant_density_profile(self, b: float, x_grid: np.ndarray,
                                    y_grid: np.ndarray) -> np.ndarray:
        """
        计算参与者密度分布 ρ_part(x,y)。

        ρ_part(x,y) = T_A [1 - (1 - σ_nn T_B/B)^B]
                      + T_B [1 - (1 - σ_nn T_A/A)^A]

        Parameters
        ----------
        b : float
            碰撞参数 [fm]
        x_grid, y_grid : np.ndarray
            横向网格

        Returns
        -------
        np.ndarray
            参与者密度 [核子/fm²]
        """
        X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
        T_A = self.thickness_function(X, Y, 'A')
        T_B = self.thickness_function(X - b, Y, 'B')
        sigma_fm2 = self.sigma_nn * 0.1

        rho_part = (T_A * (1.0 - (1.0 - sigma_fm2 * T_B / self.B) ** self.B) +
                    T_B * (1.0 - (1.0 - sigma_fm2 * T_A / self.A) ** self.A))
        return rho_part
