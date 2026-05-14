"""
gravity_harmonics.py

基于 polynomial_conversion (Legendre/Chebyshev/Gegenbauer 多项式转换) 核心算法，
实现小行星外部引力场的球谐展开模型。

科学背景：
对于远离小行星的轨道器，引力势可用球谐级数展开：
    U(r,θ,λ) = GM/r [ 1 + Σ_{n=2}^{N_max} Σ_{m=0}^{n} (R_e/r)^n P̄_{nm}(sinθ)
                × ( C_{nm} cos(mλ) + S_{nm} sin(mλ) ) ]

其中:
- P̄_{nm} 为全归一化缔合 Legendre 函数
- C_{nm}, S_{nm} 为球谐系数（Stokes 系数）
- R_e 为参考半径
- N_max 为展开截断阶数

核心公式：
1. 引力加速度:
    g = ∇U = (∂U/∂r) e_r + (1/r)(∂U/∂θ) e_θ + (1/(r cosθ))(∂U/∂λ) e_λ

2. 径向分量:
    ∂U/∂r = −GM/r² [ 1 + Σ_{n,m} (n+1)(R_e/r)^n P̄_{nm}(C_{nm}cos mλ + S_{nm}sin mλ) ]

3. 纬度分量:
    (1/r) ∂U/∂θ = (GM/r²) Σ_{n,m} (R_e/r)^n dP̄_{nm}/dθ (C_{nm}cos mλ + S_{nm}sin mλ)
"""

import numpy as np
from typing import Tuple, Optional
from special_functions import associated_legendre_normalized, factorial_ratio


class GravityHarmonicsError(Exception):
    pass


def compute_stokes_coefficients_from_shape(
    vertices: np.ndarray,
    faces: np.ndarray,
    density: float,
    n_max: int = 8,
    r_ref: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    由多面体形状和均匀密度估算球谐系数 C_{nm} 和 S_{nm}。
    使用体积积分法：
        C_{nm} = (1/M) ∫_V ρ (r/R_e)^n P̄_{nm}(sinφ) cos(mλ) dV
        S_{nm} = (1/M) ∫_V ρ (r/R_e)^n P̄_{nm}(sinφ) sin(mλ) dV

    数值实现：将多面体拆分为四面体，对每个四面体使用高斯积分近似。
    """
    vol, com = _polyhedron_volume_com(vertices, faces)
    if vol <= 0.0:
        raise GravityHarmonicsError("多面体体积非正")

    mass = density * vol
    if r_ref is None:
        # 取最大质心距作为参考半径
        r_ref = float(np.max(np.linalg.norm(vertices - com, axis=1)))

    # 平移到质心坐标系
    v = vertices - com

    c_coeff = np.zeros((n_max + 1, n_max + 1))
    s_coeff = np.zeros((n_max + 1, n_max + 1))

    # 对每个四面体（原点+面三角形）进行积分
    for f in faces:
        v0, v1, v2 = v[f[0]], v[f[1]], v[f[2]]
        tet_vol = abs(np.dot(v0, np.cross(v1, v2))) / 6.0
        if tet_vol < 1e-16:
            continue
        # 四面体质心
        tet_com = (v0 + v1 + v2) / 4.0
        r = np.linalg.norm(tet_com)
        if r < 1e-14:
            continue
        sin_phi = tet_com[2] / r
        sin_phi = np.clip(sin_phi, -1.0, 1.0)
        lon = np.arctan2(tet_com[1], tet_com[0])

        for n in range(2, n_max + 1):
            radial_factor = (r / r_ref) ** n
            for m in range(0, n + 1):
                pnm = associated_legendre_normalized(n, m, sin_phi)
                c_coeff[n, m] += density * tet_vol * radial_factor * pnm * np.cos(m * lon)
                s_coeff[n, m] += density * tet_vol * radial_factor * pnm * np.sin(m * lon)

    c_coeff /= mass
    s_coeff /= mass

    return c_coeff, s_coeff, r_ref


def _polyhedron_volume_com(vertices: np.ndarray, faces: np.ndarray) -> Tuple[float, np.ndarray]:
    """内部辅助：计算多面体有向体积与质心。"""
    vol = 0.0
    com = np.zeros(3)
    for f in faces:
        v0 = vertices[f[0]]
        v1 = vertices[f[1]]
        v2 = vertices[f[2]]
        tet_vol = np.dot(v0, np.cross(v1, v2)) / 6.0
        vol += tet_vol
        com += tet_vol * (v0 + v1 + v2) / 4.0
    if abs(vol) < 1e-14:
        return 0.0, np.zeros(3)
    com /= vol
    return abs(vol), com


class SphericalHarmonicGravity:
    """
    球谐引力场模型，支持计算引力势、引力加速度与 Hessian 矩阵。
    """

    def __init__(
        self,
        gm: float,
        r_ref: float,
        c_coeff: np.ndarray,
        s_coeff: np.ndarray,
        n_max: int = 8
    ):
        """
        参数:
            gm: 引力常数 × 质量 (km³/s²)
            r_ref: 参考半径 (km)
            c_coeff: (n_max+1, n_max+1) C_{nm} 系数
            s_coeff: (n_max+1, n_max+1) S_{nm} 系数
            n_max: 最大阶数
        """
        self.gm = gm
        self.r_ref = r_ref
        self.n_max = min(n_max, c_coeff.shape[0] - 1)
        self.c = c_coeff
        self.s = s_coeff

    def potential(self, pos: np.ndarray) -> float:
        """
        计算引力势 U(r,θ,λ)。
        pos: 三维位置向量 (km)，假设坐标原点在质心。
        """
        r = np.linalg.norm(pos)
        if r < 1e-6:
            return -np.inf
        sin_phi = pos[2] / r
        sin_phi = np.clip(sin_phi, -1.0, 1.0)
        lon = np.arctan2(pos[1], pos[0])

        u = self.gm / r
        sum_term = 0.0
        for n in range(2, self.n_max + 1):
            radial = (self.r_ref / r) ** n
            for m in range(0, n + 1):
                pnm = associated_legendre_normalized(n, m, sin_phi)
                cml = np.cos(m * lon)
                sml = np.sin(m * lon)
                sum_term += radial * pnm * (self.c[n, m] * cml + self.s[n, m] * sml)
        return u * (1.0 + sum_term)

    def acceleration(self, pos: np.ndarray) -> np.ndarray:
        """
        计算引力加速度 a = ∇U。
        使用球坐标偏导数公式，再转换到笛卡尔坐标。
        """
        r = np.linalg.norm(pos)
        if r < 1e-6:
            return np.zeros(3)

        sin_phi = pos[2] / r
        sin_phi = np.clip(sin_phi, -1.0, 1.0)
        cos_phi = np.sqrt(max(0.0, 1.0 - sin_phi * sin_phi))
        lon = np.arctan2(pos[1], pos[0])

        dU_dr = 0.0
        dU_dphi = 0.0
        dU_dlon = 0.0

        for n in range(2, self.n_max + 1):
            radial = (self.r_ref / r) ** n
            for m in range(0, n + 1):
                pnm = associated_legendre_normalized(n, m, sin_phi)
                cml = np.cos(m * lon)
                sml = np.sin(m * lon)
                dU_dr += -(n + 1) * radial / r * pnm * (self.c[n, m] * cml + self.s[n, m] * sml)

                # dP/dphi 的近似数值导数（避免复杂的缔合 Legendre 导数递推）
                eps = 1e-8
                pnm_p = associated_legendre_normalized(n, m, sin_phi + eps)
                pnm_m = associated_legendre_normalized(n, m, sin_phi - eps)
                dpnm_dphi = (pnm_p - pnm_m) / (2.0 * eps)
                dU_dphi += radial * dpnm_dphi * (self.c[n, m] * cml + self.s[n, m] * sml)

                dU_dlon += radial * pnm * (-m * self.c[n, m] * sml + m * self.s[n, m] * cml)

        dU_dr = -self.gm / (r * r) * (1.0 + sum(
            (self.r_ref / r) ** n * sum(
                associated_legendre_normalized(n, m, sin_phi) *
                (self.c[n, m] * np.cos(m * lon) + self.s[n, m] * np.sin(m * lon))
                for m in range(0, n + 1)
            )
            for n in range(2, self.n_max + 1)
        )) if self.n_max >= 2 else -self.gm / (r * r)

        # 重新精确计算 dU_dr
        sum_dr = 0.0
        for n in range(2, self.n_max + 1):
            radial = (self.r_ref / r) ** n
            for m in range(0, n + 1):
                pnm = associated_legendre_normalized(n, m, sin_phi)
                cml = np.cos(m * lon)
                sml = np.sin(m * lon)
                sum_dr += -(n + 1) * radial / r * pnm * (self.c[n, m] * cml + self.s[n, m] * sml)
        dU_dr = -self.gm / (r * r) + self.gm * sum_dr

        # 球坐标到笛卡尔坐标的转换矩阵
        # e_r = (cosλ cosφ, sinλ cosφ, sinφ)
        # e_phi = (-cosλ sinφ, -sinλ sinφ, cosφ)
        # e_lon = (-sinλ, cosλ, 0)
        cos_lon = np.cos(lon)
        sin_lon = np.sin(lon)

        e_r = np.array([cos_lon * cos_phi, sin_lon * cos_phi, sin_phi])
        e_phi = np.array([-cos_lon * sin_phi, -sin_lon * sin_phi, cos_phi])
        e_lon = np.array([-sin_lon, cos_lon, 0.0])

        acc = dU_dr * e_r + (1.0 / r) * dU_dphi * e_phi + (1.0 / (r * max(cos_phi, 1e-12))) * dU_dlon * e_lon
        return acc

    def gradient_fd(self, pos: np.ndarray, h: float = 1e-5) -> np.ndarray:
        """
        数值有限差分验证引力加速度。
        """
        grad = np.zeros(3)
        for i in range(3):
            pos_p = pos.copy()
            pos_m = pos.copy()
            pos_p[i] += h
            pos_m[i] -= h
            grad[i] = (self.potential(pos_p) - self.potential(pos_m)) / (2.0 * h)
        return grad
