# -*- coding: utf-8 -*-
"""
stability_analyzer.py
=====================

线性稳定性分析模块.

物理背景:
  撕裂模线性稳定性分析基于本征值问题:
    ∂ψ₁/∂t = L ψ₁
  其中 L 为线性化 MHD 算子.

  对于 Harris 电流片, 在 Fourier 空间 (关于 x):
    γ ψ̂₁ = -i k B_x(y) ψ̂₁ + (η/μ₀) (d²/dy² - k²) ψ̂₁

  这是广义本征值问题, 可以通过离散化后求解矩阵本征值.

  Newcomb 判据:
    理想 MHD 稳定性由能量原理 δW 决定.
    若 δW > 0 对所有允许扰动, 则理想稳定.
    撕裂模在理想稳定但 resistive 不稳定的情况下出现.

  稳定性图 (stability diagram):
    在 (kL, S) 参数空间中绘制稳定性边界.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Dict

from plasma_constants import L_CS, B0, MU_0, LUNDQUIST_S
from high_order_fd import build_second_derivative_matrix


# -------------------------------------------------------------------
#  线性化感应算子
# -------------------------------------------------------------------
def build_induction_operator(
    y: np.ndarray,
    k_mode: float,
    bx0: np.ndarray,
    eta_hat: float,
    order: int = 4,
) -> np.ndarray:
    """
    构造线性化感应算子 L (在 y 方向离散).

    L = -i k B_x(y) + eta_hat * (d²/dy² - k²)

    参数:
        y: y 坐标 (ny,)
        k_mode: 平行波数
        bx0: 平衡磁场 B_x(y) (ny,)
        eta_hat: 归一化电阻率
        order: 差分阶数

    返回:
        ny x ny 复数矩阵 L
    """
    ny = y.size
    D2 = build_second_derivative_matrix(y, order=order, bc="dirichlet")

    # 对角项: -i k B_x(y) - eta_hat * k²
    diag = -1j * k_mode * bx0 - eta_hat * k_mode * k_mode

    L = eta_hat * D2 + np.diag(diag)
    return L


# -------------------------------------------------------------------
#  本征值求解
# -------------------------------------------------------------------
def solve_tearing_eigenvalues(
    y: np.ndarray,
    k_mode: float,
    bx0: np.ndarray,
    eta_hat: float,
    n_modes: int = 10,
    order: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    求解撕裂模本征值问题 L ψ = γ ψ.

    返回:
        (eigenvalues, eigenvectors):
            eigenvalues: 复数增长率 γ (按实部降序)
            eigenvectors: 对应本征函数 (ny, n_modes)
    """
    L = build_induction_operator(y, k_mode, bx0, eta_hat, order=order)

    # 全矩阵本征值求解 (小规模问题)
    eigenvalues, eigenvectors = np.linalg.eig(L)

    # 按实部降序排序 (最不稳定模式在前)
    idx = np.argsort(np.real(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx[:n_modes]]
    eigenvectors = eigenvectors[:, idx[:n_modes]]

    return eigenvalues, eigenvectors


# -------------------------------------------------------------------
#  Newcomb 能量原理 δW
# -------------------------------------------------------------------
def compute_delta_w(
    y: np.ndarray,
    xi: np.ndarray,
    bx0: np.ndarray,
    pressure0: np.ndarray,
    k_mode: float,
    dy: np.ndarray = None,
) -> float:
    """
    计算理想 MHD 能量原理的 δW.

    δW = (π/μ₀) ∫ dy [
        |F|² |ξ|² + |B₁|² - 2 μ₀ dp/dy |ξ|²
    ]

    其中:
        F = k · B₀ = k B_x(y)  (对撕模, k_y = 0)
        B₁ = i k B_x ξ + (B_x ξ)'

    对于撕裂模 (共振面处 F = 0), δW 主要由
    电流梯度项决定.

    参数:
        y: y 坐标
        xi: 位移本征函数 ξ(y)
        bx0: 平衡磁场 B_x(y)
        pressure0: 平衡压强 p(y)
        k_mode: 波数
        dy: 间距 (若为 None, 自动计算)

    返回:
        δW (标量, 正表示稳定)
    """
    ny = y.size
    if dy is None:
        dy = np.diff(y)

    # 数值微分: d(B_x)/dy, d(p)/dy
    dbx_dy = np.gradient(bx0, y)
    dp_dy = np.gradient(pressure0, y)

    # F = k B_x
    F = k_mode * bx0

    # B1 = i k B_x xi + d(B_x xi)/dy
    b1 = 1j * k_mode * bx0 * xi + np.gradient(bx0 * xi, y)

    # δW 被积函数
    integrand = (
        np.abs(F) ** 2 * np.abs(xi) ** 2
        + np.abs(b1) ** 2
        - 2.0 * MU_0 * dp_dy * np.abs(xi) ** 2
    )

    # 梯形积分
    delta_w = np.trapz(integrand, y)
    return float(np.real(delta_w)) / MU_0


# -------------------------------------------------------------------
#   tearing 稳定性参数 Delta'
# -------------------------------------------------------------------
def compute_tearing_delta_prime(
    y: np.ndarray,
    psi_hat: np.ndarray,
) -> float:
    """
    数值计算 tearing 稳定性参数 Delta'.

    Delta' = lim_{ε->0} [ψ̂'(ε) - ψ̂'(-ε)] / ψ̂(0)

    对于 Harris 片的外部理想 MHD 区域:
        ψ̂'' - k² ψ̂ = 0  =>  ψ̂ ~ exp(-k|y|)
        Delta' = -2k (稳定)

    但对于有限宽度电流片:
        Delta' L = 2 (1/(kL) - kL)
    """
    # 找到共振面 (y=0)
    idx0 = np.argmin(np.abs(y))
    psi0 = psi_hat[idx0]

    if abs(psi0) < 1.0e-30:
        return 0.0

    if idx0 == 0 or idx0 == len(y) - 1:
        # 边界: 使用解析近似
        return -2.0 / L_CS

    # 中心差分
    dy_plus = y[idx0 + 1] - y[idx0]
    dy_minus = y[idx0] - y[idx0 - 1]
    dpsi_plus = (psi_hat[idx0 + 1] - psi0) / dy_plus
    dpsi_minus = (psi0 - psi_hat[idx0 - 1]) / dy_minus

    return float((dpsi_plus - dpsi_minus) / psi0)


# -------------------------------------------------------------------
#  参数扫描: 稳定性图
# -------------------------------------------------------------------
def stability_scan(
    k_array: np.ndarray,
    eta_array: np.ndarray,
    y: np.ndarray,
    bx0: np.ndarray,
    order: int = 4,
) -> Dict[str, np.ndarray]:
    """
    在 (k, eta) 参数空间扫描最大增长率.

    返回:
        dict:
            "k_array": 波数数组
            "eta_array": 电阻率数组
            "gamma_max": 最大增长率 (len(k), len(eta))
            "omega_at_max": 对应频率
    """
    nk = k_array.size
    ne = eta_array.size
    gamma_max = np.zeros((nk, ne))
    omega_at_max = np.zeros((nk, ne))

    for ik, k in enumerate(k_array):
        for ie, eta in enumerate(eta_array):
            evals, evecs = solve_tearing_eigenvalues(
                y, k, bx0, eta, n_modes=5, order=order
            )
            # 最不稳定模式
            gamma_max[ik, ie] = np.real(evals[0])
            omega_at_max[ik, ie] = np.imag(evals[0])

    return {
        "k_array": k_array,
        "eta_array": eta_array,
        "gamma_max": gamma_max,
        "omega_at_max": omega_at_max,
    }


# -------------------------------------------------------------------
#  FKR 标度律验证
# -------------------------------------------------------------------
def fkr_scaling_test(
    y: np.ndarray,
    bx0: np.ndarray,
    eta_values: np.ndarray,
    k_mode: float,
) -> Dict[str, np.ndarray]:
    """
    验证 FKR 标度律: gamma ~ eta^{-3/5} (即 gamma ~ S^{3/5}).

    返回:
        dict: "eta", "gamma_numerical", "gamma_fkr", "ratio"
    """
    gamma_num = np.zeros_like(eta_values)
    gamma_fkr = np.zeros_like(eta_values)

    for i, eta in enumerate(eta_values):
        evals, _ = solve_tearing_eigenvalues(
            y, k_mode, bx0, eta, n_modes=3, order=4
        )
        gamma_num[i] = np.real(evals[0])

        # FKR 预测: gamma ~ S^{3/5} ~ eta^{-3/5}
        S = 1.0 / eta
        gamma_fkr[i] = 0.6 * (k_mode * L_CS) ** (2.0 / 3.0) * S ** 0.6 * eta

    ratio = np.where(
        gamma_fkr > 1.0e-30,
        gamma_num / gamma_fkr,
        0.0,
    )

    return {
        "eta": eta_values,
        "gamma_numerical": gamma_num,
        "gamma_fkr": gamma_fkr,
        "ratio": ratio,
    }
