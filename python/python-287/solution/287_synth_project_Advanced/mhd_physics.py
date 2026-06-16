# -*- coding: utf-8 -*-
"""
mhd_physics.py
==============

Resistive MHD 方程与 Harris 电流片平衡模块.

来源种子项目:
  - 788_navier_stokes_3d_exact  -> 精确解验证思想 (ETHIER, Poiseuille, Burgers)
  - 861_pendulum_nonlinear_ode -> 非线性 ODE 右端项构造范式

物理模型:
  可压缩 resistive MHD 方程组 (归一化):
    ∂ρ/∂t + ∇·(ρv) = 0                               (质量守恒)
    ∂(ρv)/∂t + ∇·(ρvv + p*I - BB/μ₀ + B²/(2μ₀)*I) = 0  (动量守恒)
    ∂B/∂t - ∇×(v×B) = -∇×(η J)                       (感应方程)
    ∂e/∂t + ∇·((e+p)v - (v·B)B/μ₀) = η J²            (能量守恒)

  其中:
    ρ: 质量密度
    v: 速度场
    p: 热压强
    B: 磁场
    J = ∇×B/μ₀: 电流密度
    e = p/(γ-1) + ρv²/2 + B²/(2μ₀): 总能量密度
    η: 电阻率 (Spitzer 电阻率)
    γ = 5/3: 绝热指数

  Harris 电流片平衡:
    B_x(y) = B₀ tanh(y/L),   B_y = B_z = 0
    j_z(y) = -(B₀/(μ₀ L)) sech²(y/L)
    p(y) = p₀ - B₀²/(2μ₀) sech²(y/L)

  撕裂模扰动:
    对平衡量加上小扰动: f = f₀ + f₁ exp(i k x + γ t)
    在共振面 y_r 处满足 k·B₀(y_r) = 0 (即 B_x(y_r) = 0)
    对于 Harris 片, y_r = 0.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple

from plasma_constants import (
    B0, L_CS, RHO0, MU_0, ETA_SPITZER, J0, V_ALFVEN, TAU_ALFVEN
)


# -------------------------------------------------------------------
#  归一化单位
# -------------------------------------------------------------------
class MHDUnits:
    """归一化单位系统."""
    length = L_CS            # 特征长度
    magnetic = B0            # 特征磁场
    density = RHO0           # 特征密度
    velocity = V_ALFVEN      # 特征速度
    time = TAU_ALFVEN        # 特征时间
    current = B0 / (MU_0 * L_CS)  # 特征电流密度
    pressure = B0 * B0 / MU_0     # 特征压强
    resistivity = ETA_SPITZER     # 特征电阻率

    @classmethod
    def to_dimensionless_eta(cls) -> float:
        """归一化电阻率 eta_hat = eta / (mu_0 * v_A * L)."""
        return cls.resistivity / (MU_0 * cls.velocity * cls.length)

    @classmethod
    def lundquist_number(cls) -> float:
        """Lundquist 数 S = 1 / eta_hat."""
        return 1.0 / cls.to_dimensionless_eta()


# -------------------------------------------------------------------
#  Harris 电流片平衡
# -------------------------------------------------------------------
def harris_equilibrium_1d(
    y: np.ndarray,
    b0: float = None,
    l_cs: float = None,
    p0: float = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算一维 Harris 电流片平衡.

    参数:
        y: y 坐标数组
        b0:  渐近磁场强度 (默认 B0)
        l_cs: 电流片半宽 (默认 L_CS)
        p0:  边界压强 (默认 B0^2/(2 mu_0))

    返回:
        (Bx, jz, pressure): 磁场、电流密度、压强
    """
    if b0 is None:
        b0 = B0
    if l_cs is None:
        l_cs = L_CS
    if p0 is None:
        p0 = b0 * b0 / (2.0 * MU_0)

    arg = np.clip(y / l_cs, -20.0, 20.0)
    tanh_arg = np.tanh(arg)
    sech2_arg = 1.0 / np.cosh(arg) ** 2

    bx = b0 * tanh_arg
    jz = -(b0 / (MU_0 * l_cs)) * sech2_arg
    pressure = p0 - (b0 * b0 / (2.0 * MU_0)) * sech2_arg

    # 确保压强非负 (物理约束)
    pressure = np.maximum(pressure, 1.0e-15 * p0)

    return bx, jz, pressure


# -------------------------------------------------------------------
#  撕裂模本征函数初始扰动
# -------------------------------------------------------------------
def tearing_mode_perturbation(
    x: np.ndarray,
    y: np.ndarray,
    k_mode: float,
    psi_amp: float = 1.0e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    构造撕裂模扰动的磁通量函数 (vector potential A_z).

    在 Harris 电流片中加入 m=1 撕裂模扰动:
        psi_1(x, y) = psi_amp * cosh(y/delta)^{-2} * cos(k x)
    其中 delta 为内层宽度, 近似为 delta ~ L * S^{-1/3}.

    更简单的形式 (用于初始扰动):
        psi_1(x, y) = psi_amp * sech^2(y/L_cs) * cos(k x)
        这样扰动在共振面 y=0 处最大, 在边界处衰减.

    返回:
        (psi, bx_pert, by_pert):
            psi: 磁通量扰动 (ny, nx)
            bx_pert: 磁场 x 分量扰动 (ny, nx)
            by_pert: 磁场 y 分量扰动 (ny, nx)
    """
    X, Y = np.meshgrid(x, y)
    arg_y = np.clip(Y / L_CS, -20.0, 20.0)
    sech2_y = 1.0 / np.cosh(arg_y) ** 2

    psi = psi_amp * sech2_y * np.cos(k_mode * X)

    # B = ∇ × (psi * e_z)  =>  Bx = d(psi)/dy,  By = -d(psi)/dx
    # d(psi)/dy = psi_amp * (-2/L_cs) * sech^2(y/L) * tanh(y/L) * cos(kx)
    tanh_y = np.tanh(arg_y)
    bx_pert = psi_amp * (-2.0 / L_CS) * sech2_y * tanh_y * np.cos(k_mode * X)

    # d(psi)/dx = -psi_amp * sech^2(y/L) * k * sin(kx)
    by_pert = psi_amp * k_mode * sech2_y * np.sin(k_mode * X)

    return psi, bx_pert, by_pert


# -------------------------------------------------------------------
#  理想 MHD 稳定性参数: Delta' (tearing stability index)
# -------------------------------------------------------------------
def compute_delta_prime(
    y: np.ndarray,
    psi_hat: np.ndarray,
    k_mode: float,
) -> float:
    """
    计算撕裂模稳定性参数 Delta' (跳跃参数).

    Delta' 定义为磁通量扰动对数导数在共振面处的跳跃:
        Delta' = [psi_hat'(0+) - psi_hat'(0-)] / psi_hat(0)

    在外部理想 MHD 区域, psi_hat 满足:
        psi_hat'' - k^2 psi_hat = 0  (远离共振面)
    解为 psi_hat ~ exp(-k|y|), 因此 Delta' = -2k (对于 Harris 片).

    对于有限宽度电流片, 更精确的公式为:
        Delta' L = 2 * (1 - (kL)^2) / (kL)  (Copson, 1962)

    参数:
        y: y 坐标 (必须包含 y=0 点)
        psi_hat: 磁通量扰动幅度 (沿 y 的剖面)
        k_mode: 波数

    返回:
        Delta' (带符号, 正值表示不稳定)
    """
    # 找到 y=0 最近的索引
    idx0 = np.argmin(np.abs(y))
    if idx0 == 0 or idx0 == len(y) - 1:
        # y=0 在边界, 使用解析公式
        return 2.0 * (1.0 - (k_mode * L_CS) ** 2) / (k_mode * L_CS)

    # 数值计算: 中心差分
    dy_plus = y[idx0 + 1] - y[idx0]
    dy_minus = y[idx0] - y[idx0 - 1]
    psi0 = psi_hat[idx0]

    if abs(psi0) < 1.0e-30:
        return 0.0

    # 一阶导数
    dpsi_plus = (psi_hat[idx0 + 1] - psi0) / dy_plus
    dpsi_minus = (psi0 - psi_hat[idx0 - 1]) / dy_minus

    delta_prime = (dpsi_plus - dpsi_minus) / psi0
    return float(delta_prime)


# -------------------------------------------------------------------
#  磁 Reynolds 数和磁螺旋度
# -------------------------------------------------------------------
def magnetic_reynolds_number(
    v_rms: float,
    l_scale: float,
    eta: float = None,
) -> float:
    """
    磁 Reynolds 数 Rm = v * L / eta_hat.
    """
    if eta is None:
        eta = MHDUnits.to_dimensionless_eta()
    return v_rms * l_scale / eta


def magnetic_helicity(
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    bz: np.ndarray,
    volume_element: float,
) -> float:
    """
    磁螺旋度 K = integral A · B dV.
    在规范 ∇·A = 0 下, 磁螺旋度是守恒量.

    参数:
        ax, ay, az: 磁矢势分量
        bx, by, bz: 磁场分量
        volume_element: 单元体积

    返回:
        磁螺旋度标量
    """
    return float(np.sum((ax * bx + ay * by + az * bz) * volume_element))


def current_density_curl_b(
    bx: np.ndarray,
    by: np.ndarray,
    dz_dy: np.ndarray,
    dz_dx: np.ndarray,
    dx_dy: np.ndarray,
    dx_dz: np.ndarray,
    dy_dz: np.ndarray,
    dy_dx: np.ndarray,
    mu0: float = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从磁场计算电流密度 J = (1/mu_0) * ∇ × B.

    这里假设我们已经有所有偏导数 (由高阶 FD 计算).
    """
    if mu0 is None:
        mu0 = MU_0
    jx = (dz_dy - dy_dz) / mu0
    jy = (dx_dz - dz_dx) / mu0
    jz = (dy_dx - dx_dy) / mu0
    return jx, jy, jz


# -------------------------------------------------------------------
#  等离子体 beta 和 Alfven Mach 数
# -------------------------------------------------------------------
def plasma_beta_local(
    pressure: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    bz: np.ndarray,
    mu0: float = None,
) -> np.ndarray:
    """
    局部等离子体 beta = 2 mu_0 p / B^2.
    """
    if mu0 is None:
        mu0 = MU_0
    b2 = bx ** 2 + by ** 2 + bz ** 2
    b2 = np.maximum(b2, 1.0e-30)
    return 2.0 * mu0 * pressure / b2


def alfven_mach_number(
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    bz: np.ndarray,
    rho: np.ndarray,
    mu0: float = None,
) -> np.ndarray:
    """
    Alfven Mach 数 Ma = |v| / v_A, 其中 v_A = B / sqrt(mu_0 rho).
    """
    if mu0 is None:
        mu0 = MU_0
    v2 = vx ** 2 + vy ** 2 + vz ** 2
    b2 = bx ** 2 + by ** 2 + bz ** 2
    rho_safe = np.maximum(rho, 1.0e-30)
    v_a2 = b2 / (mu0 * rho_safe)
    v_a2 = np.maximum(v_a2, 1.0e-30)
    return np.sqrt(v2 / v_a2)
