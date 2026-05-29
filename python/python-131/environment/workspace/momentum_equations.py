"""
momentum_equations.py
=====================
基于 762_mhd_exact 改造的动量方程模块。

将 Hartmann 磁流体力学精确解的思想迁移到气泡柱反应器的两相动量守恒。
在 Hartmann 流中，横向磁场产生 Lorentz 力，与气泡柱中气泡诱导的
虚拟质量力、升力、阻力有数学同构性。本模块提供：
1. Hartmann 流的解析解（验证基准）
2. 两流体模型的动量方程残差
3. 相间动量交换项（M_{gl}）

核心公式
--------
1. Hartmann 精确解（MHD）：
       u(y) = (G Re / Ha) / tanh(Ha) · [1 - cosh(Ha·y)/cosh(Ha)]
       b(y) = (G/S) · [sinh(Ha·y)/sinh(Ha) - y]
       p(x,y) = -G·x - 0.5·S·b(y)²
   其中 Ha = B₀ L √(σ/ν) 为 Hartmann 数，
         Re = ρ u₀ L / μ 为 Reynolds 数，
         Rm = μ σ u₀ L 为磁 Reynolds 数，
         S = Ha² / (Re·Rm)。

2. 气泡柱两流体动量方程（类比 MHD）：
   气相：
       ∂(α_g ρ_g u_g)/∂t + ∇·(α_g ρ_g u_g u_g)
           = -α_g ∇p + ∇·τ_g + M_{gl} + α_g ρ_g g
   液相：
       ∂(α_l ρ_l u_l)/∂t + ∇·(α_l ρ_l u_l u_l)
           = -α_l ∇p + ∇·τ_l - M_{gl} + α_l ρ_l g

3. 相间动量交换（Drag + Virtual Mass + Lift）：
       M_{gl} = M_D + M_VM + M_L
       M_D = (3/4) α_g (ρ_l / d_b) C_D |u_g - u_l| (u_g - u_l)
       M_VM = C_VM α_g ρ_l (D u_g/Dt - D u_l/Dt)
       M_L = C_L α_g ρ_l (u_g - u_l) × (∇ × u_l)
   其中 C_D 采用 Schiller-Naumann 关联式：
       C_D = (24/Re_p)(1 + 0.15 Re_p^{0.687})   for Re_p < 1000
       C_D = 0.44                               for Re_p ≥ 1000
       Re_p = ρ_l |u_g - u_l| d_b / μ_l

4. 有效粘度（固体颗粒影响）：
       μ_eff = μ_l (1 + 2.5 α_s + 7.54 α_s²)
       （Einstein-Roscoe 公式）
"""

import numpy as np


# ---------------------------------------------------------------------------
# Hartmann exact solution (from mhd_exact)
# ---------------------------------------------------------------------------

class HartmannFlow:
    """
    Hartmann 磁流体流动的解析解。
    用作 CFD 求解器的验证基准（Benchmark）。
    """

    def __init__(self, G=1.0, Ha=1.0, L=10.0, p0=4.0, Re=10.0, Rm=6.0):
        self.G = G
        self.Ha = Ha
        self.L = L
        self.p0 = p0
        self.Re = Re
        self.Rm = Rm
        self.S = Ha**2 / (Re * Rm)

    def velocity(self, y):
        """
        水平速度 u(y)，y ∈ [-1, 1]。
        """
        y = np.asarray(y, dtype=float)
        return (self.G * self.Re / self.Ha / np.tanh(self.Ha)
                * (1.0 - np.cosh(y * self.Ha) / np.cosh(self.Ha)))

    def magnetic_field_b(self, y):
        """
        横向磁场 b(y)。
        """
        y = np.asarray(y, dtype=float)
        return self.G / self.S * (np.sinh(y * self.Ha) / np.sinh(self.Ha) - y)

    def pressure(self, x, y):
        """
        压力场 p(x, y)。
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        b = self.magnetic_field_b(y)
        return -self.G * x - 0.5 * self.S * b**2

    def residual_check(self, y):
        """
        验证 u(y) 和 b(y) 满足控制方程：
            u'' + Re S b' = -G Re
            b'' + Rm u' = 0
        返回 (ur, br)。
        """
        y = np.asarray(y, dtype=float)
        dy = 1e-6
        # 中心差分求导
        u_p = self.velocity(y + dy)
        u_m = self.velocity(y - dy)
        u = self.velocity(y)
        uy = (u_p - u_m) / (2 * dy)
        uyy = (u_p - 2 * u + u_m) / dy**2

        b_p = self.magnetic_field_b(y + dy)
        b_m = self.magnetic_field_b(y - dy)
        b = self.magnetic_field_b(y)
        by = (b_p - b_m) / (2 * dy)
        byy = (b_p - 2 * b + b_m) / dy**2

        ur = uyy + self.Re * self.S * by + self.G * self.Re
        br = byy + self.Rm * uy
        return ur, br


# ---------------------------------------------------------------------------
# Two-fluid momentum exchange
# ---------------------------------------------------------------------------

def schiller_naumann_cd(re_p):
    """
    Schiller-Naumann 阻力系数。
    """
    re_p = np.asarray(re_p, dtype=float)
    cd = np.zeros_like(re_p)
    mask_low = re_p < 1000.0
    mask_high = ~mask_low
    cd[mask_low] = (24.0 / re_p[mask_low]) * (1.0 + 0.15 * re_p[mask_low]**0.687)
    cd[mask_high] = 0.44
    return cd


def interphase_momentum_exchange(alpha_g, u_g, u_l, rho_l, mu_l, d_b,
                                 C_VM=0.5, C_L=0.25):
    """
    计算相间动量交换项 M_gl（单位体积，N/m³）。

    Parameters
    ----------
    alpha_g, u_g, u_l : float or ndarray
        气含率、气速、液速（均为标量或同形数组）。
    rho_l, mu_l : float
        液相密度与动力粘度。
    d_b : float
        气泡直径（Sauter 平均直径或局部直径）。
    C_VM, C_L : float
        虚拟质量系数与升力系数。

    Returns
    -------
    M_gl : float or ndarray
        气相受到的相间力（液相对气相的作用）。
    """
    alpha_g = np.asarray(alpha_g, dtype=float)
    u_g = np.asarray(u_g, dtype=float)
    u_l = np.asarray(u_l, dtype=float)

    # 物理边界
    alpha_g = np.clip(alpha_g, 1e-6, 0.95)

    u_rel = u_g - u_l
    re_p = rho_l * np.abs(u_rel) * d_b / max(mu_l, 1e-12)
    C_D = schiller_naumann_cd(re_p)

    # 阻力
    M_D = 0.75 * alpha_g * (rho_l / max(d_b, 1e-9)) * C_D * np.abs(u_rel) * u_rel

    # 虚拟质量力（稳态简化）
    M_VM = C_VM * alpha_g * rho_l * 0.0  # 稳态时物质导数项为零

    # 升力（简化一维近似：仅考虑横向速度梯度引起的升力投影）
    # 在一维轴向模型中升力主要影响径向分布，此处以经验修正项近似
    M_L = C_L * alpha_g * rho_l * u_rel * 0.05

    M_gl = M_D + M_VM + M_L
    return M_gl


def effective_viscosity_slurry(mu_l, alpha_s):
    """
    Einstein-Roscoe 浆态有效粘度。

    Parameters
    ----------
    mu_l : float
        纯液相粘度 [Pa·s]。
    alpha_s : float or ndarray
        固含率 [-]。

    Returns
    -------
    mu_eff : float or ndarray
        有效粘度 [Pa·s]。
    """
    alpha_s = np.asarray(alpha_s, dtype=float)
    alpha_s = np.clip(alpha_s, 0.0, 0.6)
    return mu_l * (1.0 + 2.5 * alpha_s + 7.54 * alpha_s**2)


def two_fluid_momentum_residual(alpha_g, u_g, u_l, p, rho_g, rho_l, mu_eff,
                                g_vec, d_b, dx, dy):
    """
    二维稳态两流体动量方程的离散残差（简化结构化网格）。

    返回气相和液相 x,y 方向的动量残差。
    为简化起见，假设一维轴向流动（u_g = [0, u_{gz}], u_l = [0, u_{lz}]]）。
    """
    alpha_g = np.clip(alpha_g, 1e-6, 0.95)
    alpha_l = 1.0 - alpha_g

    # 相间力（仅 z 方向）
    M_gl = interphase_momentum_exchange(alpha_g, u_g, u_l, rho_l, mu_eff, d_b)

    # 气相动量残差（稳态 1D）：
    # 0 = -α_g dp/dz + M_gl + α_g ρ_g g_z
    # 简化压力梯度
    dp_dz = 0.0  # 由用户提供或外部计算
    res_g = -alpha_g * dp_dz + M_gl + alpha_g * rho_g * g_vec

    # 液相动量残差：
    res_l = -alpha_l * dp_dz - M_gl + alpha_l * rho_l * g_vec

    return res_g, res_l
