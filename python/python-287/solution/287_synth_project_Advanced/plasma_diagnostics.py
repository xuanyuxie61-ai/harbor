# -*- coding: utf-8 -*-
"""
plasma_diagnostics.py
=====================

等离子体物理诊断模块.

物理背景:
  MHD 模拟需要一系列诊断量来量化不稳定性演化:

  1. 磁岛宽度 w:
     撕裂模形成磁岛, 岛宽度为:
       w = 4 * sqrt(|ψ_1| / (k B_x'(0)))
     其中 ψ_1 为磁通量扰动在共振面的幅值.

  2. 磁重构率 (reconnection rate):
     E_rec = η J_rec = η |∇ × B|_rec
     归一化重构率: R_rec = E_rec / (v_A B_0)

  3. 能量分解:
     - 磁能: W_B = ∫ B²/(2μ₀) dV
     - 动能: W_K = ∫ ρ v²/2 dV
     - 热能: W_T = ∫ p/(γ-1) dV

  4. 安全因子 q:
     q = (r/R₀) (B_φ/B_θ)
     共振面位于 q = m/n (有理面).
     对于撕裂模 m=1, 共振面在 q = 1.

  5. 电流片厚度 δ:
     非线性演化中电流片变薄:
       δ ~ L S^{-1/3} (线性)
       δ ~ L S^{-1/2} (非线性 Sweet-Parker)
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple

from plasma_constants import B0, L_CS, MU_0, RHO0, ETA_SPITZER


# -------------------------------------------------------------------
#  磁岛宽度
# -------------------------------------------------------------------
def magnetic_island_width(
    y: np.ndarray,
    psi_hat: np.ndarray,
    k_mode: float,
    bx0: np.ndarray,
) -> float:
    """
    计算磁岛宽度.

    在共振面附近, 磁通量:
        ψ(y) ≈ ψ(0) + (1/2) ψ''(0) y²
    岛宽度 (O-X 点距离):
        w = 4 sqrt(|ψ(0)| / |ψ''(0)|)

    或者从平衡磁场梯度:
        w = 4 sqrt(|ψ_1| / (k B_x'(0)))

    参数:
        y: y 坐标
        psi_hat: 磁通量扰动幅度剖面
        k_mode: 平行波数
        bx0: 平衡磁场 B_x(y)

    返回:
        岛宽度 w (归一化单位)
    """
    idx0 = np.argmin(np.abs(y))
    psi0 = psi_hat[idx0]

    if abs(psi0) < 1.0e-30:
        return 0.0

    # 方法 1: 从 ψ'' 计算
    if 1 <= idx0 < len(y) - 1:
        dy_m = y[idx0] - y[idx0 - 1]
        dy_p = y[idx0 + 1] - y[idx0]
        d2psi = 2.0 * (
            psi_hat[idx0 + 1] / (dy_p * (dy_m + dy_p))
            - psi_hat[idx0] / (dy_m * dy_p)
            + psi_hat[idx0 - 1] / (dy_m * (dy_m + dy_p))
        )
        if abs(d2psi) > 1.0e-30:
            w = 4.0 * np.sqrt(abs(psi0 / d2psi))
            return float(w)

    # 方法 2: 从平衡磁场梯度
    dbx_dy = np.gradient(bx0, y)
    dbx0 = dbx_dy[idx0]
    if abs(dbx0) < 1.0e-30:
        return 0.0
    w = 4.0 * np.sqrt(abs(psi0) / (k_mode * abs(dbx0)))
    return float(w)


# -------------------------------------------------------------------
#  磁重构率
# -------------------------------------------------------------------
def reconnection_rate(
    jz: np.ndarray,
    eta: float = None,
    b0: float = None,
    v_alfven: float = None,
) -> float:
    """
    计算归一化磁重构率 R_rec.

    R_rec = η |J_rec| / (v_A B_0)

    参数:
        jz: 电流密度 (在 X 点处)
        eta: 电阻率
        b0: 渐近磁场
        v_alfven: Alfven 速度

    返回:
        归一化重构率
    """
    if eta is None:
        eta = ETA_SPITZER
    if b0 is None:
        b0 = B0
    if v_alfven is None:
        from plasma_constants import V_ALFVEN
        v_alfven = V_ALFVEN

    # 最大电流密度 (X 点处)
    j_max = np.max(np.abs(jz))
    e_rec = eta * j_max
    return float(e_rec / (v_alfven * b0))


# -------------------------------------------------------------------
#  能量计算
# -------------------------------------------------------------------
def magnetic_energy_density(
    bx: np.ndarray,
    by: np.ndarray,
    bz: np.ndarray,
    mu0: float = None,
) -> np.ndarray:
    """磁能密度: B²/(2μ₀)."""
    if mu0 is None:
        mu0 = MU_0
    return (bx ** 2 + by ** 2 + bz ** 2) / (2.0 * mu0)


def kinetic_energy_density(
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    rho: np.ndarray,
) -> np.ndarray:
    """动能密度: ρ v²/2."""
    return 0.5 * rho * (vx ** 2 + vy ** 2 + vz ** 2)


def thermal_energy_density(
    pressure: np.ndarray,
    gamma: float = 5.0 / 3.0,
) -> np.ndarray:
    """热能密度: p/(γ-1)."""
    return pressure / (gamma - 1.0)


def total_energy(
    bx: np.ndarray,
    by: np.ndarray,
    bz: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    pressure: np.ndarray,
    rho: np.ndarray,
    area_element: np.ndarray,
    mu0: float = None,
    gamma: float = 5.0 / 3.0,
) -> Dict[str, float]:
    """
    计算总能量及其各成分.

    返回:
        dict: "magnetic", "kinetic", "thermal", "total"
    """
    w_b_density = magnetic_energy_density(bx, by, bz, mu0)
    w_k_density = kinetic_energy_density(vx, vy, vz, rho)
    w_t_density = thermal_energy_density(pressure, gamma)

    # 处理面积元与场尺寸不匹配 (area_element 为 cell, 场为 node)
    if area_element.shape != w_b_density.shape:
        # 简单策略: 取平均面积作为节点权重
        mean_area = float(np.mean(area_element))
        w_b = np.sum(w_b_density) * mean_area
        w_k = np.sum(w_k_density) * mean_area
        w_t = np.sum(w_t_density) * mean_area
    else:
        w_b = np.sum(w_b_density * area_element)
        w_k = np.sum(w_k_density * area_element)
        w_t = np.sum(w_t_density * area_element)

    return {
        "magnetic": float(w_b),
        "kinetic": float(w_k),
        "thermal": float(w_t),
        "total": float(w_b + w_k + w_t),
    }


# -------------------------------------------------------------------
#  安全因子 q
# -------------------------------------------------------------------
def safety_factor_profile(
    r: np.ndarray,
    b_theta: np.ndarray,
    b_phi: float,
    R0: float,
) -> np.ndarray:
    """
    计算安全因子剖面 q(r).

    q(r) = (r/R₀) (B_φ/B_θ)

    参数:
        r: 小半径数组
        b_theta: 极向磁场 B_θ(r)
        b_phi: 环向磁场 B_φ (常数近似)
        R0: 大半径

    返回:
        q(r) 数组
    """
    b_theta_safe = np.maximum(np.abs(b_theta), 1.0e-30)
    return (r / R0) * (b_phi / b_theta_safe)


def find_rational_surface(
    r: np.ndarray,
    q_profile: np.ndarray,
    m: int,
    n: int,
) -> float:
    """
    找到有理面 q = m/n 的位置.

    参数:
        r: 小半径数组
        q_profile: q(r) 剖面
        m, n: 极向/环向模数

    返回:
        有理面位置 r_res (若不存在则返回 -1)
    """
    q_target = float(m) / float(n)

    # 找 q 跨越 q_target 的位置
    for i in range(len(r) - 1):
        if (q_profile[i] - q_target) * (q_profile[i + 1] - q_target) <= 0:
            # 线性插值
            frac = (q_target - q_profile[i]) / (q_profile[i + 1] - q_profile[i])
            return float(r[i] + frac * (r[i + 1] - r[i]))

    return -1.0


# -------------------------------------------------------------------
#  电流片厚度
# -------------------------------------------------------------------
def current_sheet_thickness(
    y: np.ndarray,
    jz: np.ndarray,
) -> float:
    """
    估计电流片厚度 δ.

    δ 定义为电流密度从峰值下降到 1/e 的半宽度.
    """
    j_max = np.max(np.abs(jz))
    if j_max < 1.0e-30:
        return 0.0

    threshold = j_max / np.e

    # 找峰值位置
    idx_max = np.argmax(np.abs(jz))

    # 向右找阈值
    delta_right = 0.0
    for i in range(idx_max, len(y) - 1):
        if np.abs(jz[i]) < threshold:
            delta_right = y[i] - y[idx_max]
            break
    else:
        delta_right = y[-1] - y[idx_max]

    # 向左找阈值
    delta_left = 0.0
    for i in range(idx_max, 0, -1):
        if np.abs(jz[i]) < threshold:
            delta_left = y[idx_max] - y[i]
            break
    else:
        delta_left = y[idx_max] - y[0]

    return float((delta_left + delta_right) / 2.0)


# -------------------------------------------------------------------
#  诊断汇总
# -------------------------------------------------------------------
def diagnostic_summary(
    y: np.ndarray,
    bx0: np.ndarray,
    psi_hat: np.ndarray,
    jz: np.ndarray,
    k_mode: float,
) -> Dict[str, float]:
    """
    汇总所有诊断量.
    """
    w_island = magnetic_island_width(y, psi_hat, k_mode, bx0)
    R_rec = reconnection_rate(jz)
    delta_cs = current_sheet_thickness(y, jz)

    return {
        "island_width": w_island,
        "reconnection_rate": R_rec,
        "current_sheet_thickness": delta_cs,
        "max_jz": float(np.max(np.abs(jz))),
        "max_psi": float(np.max(np.abs(psi_hat))),
    }
