# -*- coding: utf-8 -*-
"""
field_reconstructor.py
======================

三维磁场重构与上采样模块.

来源种子项目:
  - 1190_VitjanZ_3DSR -> 3D 超分辨率编码器-解码器结构

物理背景:
  托卡马克中的磁平衡重建通常需要结合有限数量的磁探针测量.
  我们的任务是从 2D 截面 (R, Z) 的磁通量分布重构 3D 磁场 B(R, φ, Z).

  方法:
    1. 2D 平衡: ψ(R, Z) 由 Grad-Shafranov 方程给出
    2. 环向展开: ψ(R, φ, Z) = sum_n ψ_n(R, Z) exp(i n φ)
    3. 上采样: 从粗网格到细网格 (类似 3DSR 的 encoder-decoder)

  简化模型 (用于验证):
    假设磁场主要由 m=0 (轴对称) 和 m=1 (撕裂模) 成分组成:
      B(R, φ, Z) ≈ B_0(R, Z) + B_1(R, Z) cos(φ - φ_0(R, Z))

  上采样网络:
    受 3DSR 启发, 我们使用多级插值:
      粗场 -> encoder (特征提取) -> decoder (重建) -> 细场
    简化为多线性插值 + 残差修正.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple

from mesh_generator import StructuredMesh2D, extend_to_3d


# -------------------------------------------------------------------
#  多级上采样 (Multi-level upsampling)
# -------------------------------------------------------------------
def bilinear_upsample_2d(
    field_coarse: np.ndarray,
    factor: int = 2,
) -> np.ndarray:
    """
    双线性上采样 2D 场.

    参数:
        field_coarse: 粗网格场 (ny, nx)
        factor: 上采样倍数

    返回:
        field_fine: 细网格场 (factor*ny, factor*nx)
    """
    ny, nx = field_coarse.shape
    ny_fine = factor * ny
    nx_fine = factor * nx

    field_fine = np.zeros((ny_fine, nx_fine))

    for j in range(ny_fine):
        for i in range(nx_fine):
            # 粗网格坐标
            y_c = j / factor
            x_c = i / factor

            j0 = min(int(y_c), ny - 1)
            i0 = min(int(x_c), nx - 1)
            j1 = min(j0 + 1, ny - 1)
            i1 = min(i0 + 1, nx - 1)

            wy = y_c - j0
            wx = x_c - i0

            field_fine[j, i] = (
                (1 - wy) * (1 - wx) * field_coarse[j0, i0]
                + (1 - wy) * wx * field_coarse[j0, i1]
                + wy * (1 - wx) * field_coarse[j1, i0]
                + wy * wx * field_coarse[j1, i1]
            )

    return field_fine


def multiscale_upsample(
    field_coarse: np.ndarray,
    target_shape: Tuple[int, int],
    n_levels: int = 3,
) -> np.ndarray:
    """
    多级上采样 (类似 3DSR 的 encoder-decoder 结构).

    策略:
      - 从最粗尺度开始, 每级上采样 2x
      - 每级使用双线性插值 + 简单平滑
    """
    ny_target, nx_target = target_shape
    ny_c, nx_c = field_coarse.shape

    # 计算需要的级数
    factor_y = ny_target / ny_c
    factor_x = nx_target / nx_c
    factor = max(factor_y, factor_x)
    actual_levels = max(1, int(np.ceil(np.log2(factor))))
    actual_levels = min(actual_levels, n_levels)

    field = field_coarse.copy()

    for level in range(actual_levels):
        ny_curr, nx_curr = field.shape
        # 上采样 2x
        field = bilinear_upsample_2d(field, factor=2)
        # 简单平滑 (5 点平均)
        field = _smooth_2d(field)

    # 裁剪到目标尺寸
    field = field[:ny_target, :nx_target]
    return field


def _smooth_2d(field: np.ndarray) -> np.ndarray:
    """简单 5 点平滑."""
    ny, nx = field.shape
    smoothed = field.copy()
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            smoothed[j, i] = (
                0.5 * field[j, i]
                + 0.125 * (
                    field[j - 1, i] + field[j + 1, i]
                    + field[j, i - 1] + field[j, i + 1]
                )
            )
    return smoothed


# -------------------------------------------------------------------
#  3D 磁场重构
# -------------------------------------------------------------------
def reconstruct_3d_field(
    psi_2d: np.ndarray,
    mesh2d: StructuredMesh2D,
    n_phi: int = 16,
    n_harmonics: int = 3,
    amplitudes: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从 2D 磁通量重构 3D 磁场.

    模型:
        ψ(R, φ, Z) = ψ_0(R, Z) + sum_{n=1}^{N} ψ_n(R, Z) cos(n φ)

    参数:
        psi_2d: 2D 磁通量 (ny, nx)
        mesh2d: 2D 网格
        n_phi: 环向分辨率
        n_harmonics: 谐波段数
        amplitudes: 各谐波幅度 (n_harmonics,), 默认递减

    返回:
        (B_R, B_phi, B_Z): 3D 磁场分量 (n_phi, ny, nx)
    """
    if amplitudes is None:
        amplitudes = 0.1 ** np.arange(1, n_harmonics + 1)

    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    B_R = np.zeros((n_phi, mesh2d.ny, mesh2d.nx))
    B_phi = np.zeros((n_phi, mesh2d.ny, mesh2d.nx))
    B_Z = np.zeros((n_phi, mesh2d.ny, mesh2d.nx))

    # 轴对称部分: B = ∇ψ × e_φ / R
    # B_R = -(1/R) dψ/dZ,  B_Z = (1/R) dψ/dR
    # 简化: 假设 R = R_0 (大半径常数)
    R0 = 1.0  # 归一化
    dpsi_dy = np.gradient(psi_2d, mesh2d.y, axis=0)
    dpsi_dx = np.gradient(psi_2d, mesh2d.x, axis=1)

    for k in range(n_phi):
        # 轴对称部分
        B_R[k] = -dpsi_dy / R0
        B_Z[k] = dpsi_dx / R0

        # 非轴对称扰动
        for n in range(n_harmonics):
            amp = amplitudes[n] if n < len(amplitudes) else 0.0
            cos_n_phi = np.cos((n + 1) * phi[k])
            B_phi[k] += amp * psi_2d * cos_n_phi

    return B_R, B_phi, B_Z


# -------------------------------------------------------------------
#  磁场散度检验
# -------------------------------------------------------------------
def check_divergence_free(
    B_R: np.ndarray,
    B_phi: np.ndarray,
    B_Z: np.ndarray,
    mesh2d: StructuredMesh2D,
    R0: float = 1.0,
) -> float:
    """
    检验重构磁场的散度 ∇·B.

    柱坐标:
        ∇·B = (1/R) d(R B_R)/dR + (1/R) d(B_φ)/dφ + d(B_Z)/dZ

    返回:
        max|∇·B| / max|B| (相对散度)
    """
    n_phi = B_R.shape[0]
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = phi[1] - phi[0] if n_phi > 1 else 1.0

    max_divB = 0.0
    max_B = 0.0

    for k in range(n_phi):
        # d(B_Z)/dZ
        dBZ_dZ = np.gradient(B_Z[k], mesh2d.y, axis=0)

        # (1/R) d(R B_R)/dR
        R = R0 + mesh2d.Y
        RB_R = R * B_R[k]
        dRB_R_dR = np.gradient(RB_R, mesh2d.x, axis=1)

        # (1/R) d(B_phi)/dphi
        if n_phi > 1:
            dBphi_dphi = np.zeros_like(B_phi[k])
            for j in range(mesh2d.ny):
                for i in range(mesh2d.nx):
                    # 中心差分 (周期)
                    kp = (k + 1) % n_phi
                    km = (k - 1) % n_phi
                    dBphi_dphi[j, i] = (
                        B_phi[kp, j, i] - B_phi[km, j, i]
                    ) / (2.0 * dphi)
        else:
            dBphi_dphi = np.zeros_like(B_phi[k])

        divB = dRB_R_dR / R + dBphi_dphi / R + dBZ_dZ
        max_divB = max(max_divB, np.max(np.abs(divB)))
        max_B = max(max_B, np.max(np.sqrt(
            B_R[k] ** 2 + B_phi[k] ** 2 + B_Z[k] ** 2
        )))

    return max_divB / max(max_B, 1.0e-30)


# -------------------------------------------------------------------
#  磁场能量
# -------------------------------------------------------------------
def magnetic_energy(
    B_R: np.ndarray,
    B_phi: np.ndarray,
    B_Z: np.ndarray,
    volume_element: float,
    mu0: float = None,
) -> float:
    """
    磁能 W_B = (1/(2μ₀)) ∫ B² dV.
    """
    if mu0 is None:
        from plasma_constants import MU_0 as mu0
    B2 = B_R ** 2 + B_phi ** 2 + B_Z ** 2
    return float(np.sum(B2) * volume_element / (2.0 * mu0))
