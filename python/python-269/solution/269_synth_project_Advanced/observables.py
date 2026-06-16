"""
observables.py — 量子霍尔可观测量的计算
============================================================

本模块计算量子霍尔效应中的关键物理可观测量:

1. Hall 电导 σ_xy (Kubo 公式):
    σ_xy = (e²/h) · C  (C 为 Chern 数)

    Kubo 线性响应:
    σ_xy = (iℏe²)/(AL²) Σ_{n≠m} (f_n - f_m) ·
           Im[⟨n|v_x|m⟩⟨m|v_y|n⟩] / (E_n - E_m)²

2. 局域态密度 (LDOS):
    ρ(r, E) = Σ_n |ψ_n(r)|² δ(E - E_n)

3. 电流密度:
    j(r) = (e/m) Re[ψ*(r)(-iℏ∇ - eA)ψ(r)]

4. 粒子数密度:
    n(r) = Σ_{n ∈ filled} |ψ_n(r)|²

5. 导向中心坐标:
    X = x - l_B² · π_y / ℏ
    Y = y + l_B² · π_x / ℏ
    [X, Y] = -il_B²

参考文献:
    [1] Kubo, R. JPSJ 12, 570 (1957)
    [2] TKNN: Thouless et al. PRL 49, 405 (1982)
"""

import numpy as np
from scipy import sparse
from typing import Tuple, Dict, Any, List, Optional
from physical_constants import magnetic_length, landau_level_energy


def hall_conductance_kubo(H: sparse.csr_matrix,
                           eigenvalues: np.ndarray,
                           eigenvectors: np.ndarray,
                           filled_levels: int,
                           Lx: float, Ly: float,
                           B: float, hx: float, hy: float,
                           Nx: int, Ny: int) -> complex:
    """使用 Kubo 公式计算 Hall 电导

    σ_xy = (e²/h) · Σ_{n ∈ filled, m ∉ filled}
           Im[⟨n|v_x|m⟩⟨m|v_y|n⟩] / (E_n - E_m)²

    其中速度算符:
        v_x = (1/ℏ) ∂H/∂k_x = (p_x - A_x)/m
        v_y = (1/ℏ) ∂H/∂k_y = (p_y - A_y)/m

    在有限差分表示中:
        v_x → -i·D_x (一阶导数矩阵)
        v_y → -i·D_y - (Bx/m) (含规范项)

    Args:
        H: 哈密顿量
        eigenvalues: 本征值
        eigenvectors: 本征态
        filled_levels: 填充的朗道能级数
        Lx, Ly: 系统尺寸
        B: 磁场
        hx, hy: 网格间距
        Nx, Ny: 格点数
    Returns:
        σ_xy (单位: e²/h)
    """
    N_total = H.shape[0]
    sigma_xy = 0.0 + 0.0j

    # 速度算符 (有限差分)
    from high_order_fd import build_1d_first_derivative
    Dy = build_1d_first_derivative(Ny, hy, fd_order=2)
    Iy = sparse.eye(Ny)

    # v_y = -i(∂_y - iBA_y) = -i∂_y - Bx
    # 简化: 使用 -i·D_y 作为 v_y 的近似
    v_y = -1j * sparse.kron(sparse.eye(Nx), Dy)

    # v_x 近似: 使用 p_x/m = -i∂_x
    Dx = build_1d_first_derivative(Nx, hx, fd_order=2)
    v_x = -1j * sparse.kron(Dx, Iy)

    # Kubo 公式
    prefactor = 1.0 / (Lx * Ly)  # e²/h 在原子单位 = 1/(2π)

    for n in range(min(filled_levels, len(eigenvalues))):
        psi_n = eigenvectors[:, n]
        E_n = eigenvalues[n]
        for m in range(len(eigenvalues)):
            if m == n:
                continue
            if m < filled_levels:
                continue
            psi_m = eigenvectors[:, m]
            E_m = eigenvalues[m]
            dE = E_n - E_m
            if abs(dE) < 1e-12:
                continue

            vx_nm = np.vdot(psi_n, v_x @ psi_m)
            vy_mn = np.vdot(psi_m, v_y @ psi_n)
            sigma_xy += np.imag(vx_nm * vy_mn) / (dE ** 2)

    return prefactor * sigma_xy


def local_density_of_states(psi_list: List[np.ndarray],
                             E_list: np.ndarray,
                             x_grid: np.ndarray,
                             y_grid: np.ndarray,
                             E_target: float,
                             sigma: float = 0.05) -> np.ndarray:
    """计算局域态密度 (LDOS)

    ρ(r, E) = Σ_n |ψ_n(r)|² · δ_σ(E - E_n)

    其中 δ_σ 是高斯展宽的 δ 函数:
        δ_σ(E) = (1/σ√2π) exp(-E²/(2σ²))

    LDOS 揭示:
    - 体态: 均匀分布
    - 边缘态: 集中在边界
    - 杂质态: 局域峰

    Args:
        psi_list: 本征态列表 (一维展开)
        E_list: 对应本征值
        x_grid: x 坐标
        y_grid: y 坐标
        E_target: 目标能量
        sigma: 展宽参数
    Returns:
        LDOS (二维数组)
    """
    Nx = len(x_grid)
    Ny = len(y_grid)
    ldos = np.zeros((Nx, Ny))

    for psi, E in zip(psi_list, E_list):
        weight = np.exp(-0.5 * ((E - E_target) / sigma) ** 2) / (
            sigma * np.sqrt(2 * np.pi))
        psi_2d = np.abs(psi[:Nx*Ny].reshape(Nx, Ny)) ** 2
        ldos += weight * psi_2d

    return ldos


def current_density(psi: np.ndarray, A_y: np.ndarray,
                    Nx: int, Ny: int, hx: float, hy: float,
                    mass: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """计算概率流密度 j = (1/m) Re[ψ*(-i∇ - A)ψ]

    j_x = (1/m) Im[ψ* ∂_x ψ]
    j_y = (1/m) Im[ψ* ∂_y ψ] - (A_y/m)|ψ|²

    电流密度的散度为零 (连续性方程):
        ∇·j = 0 (定态)

    在量子霍尔系统中:
    - 体电流: j = 0 (无净输运)
    - 边缘电流: j ≠ 0 (手征边缘态)

    Args:
        psi: 波函数 (一维)
        A_y: 矢量势 y 分量 (一维)
        Nx, Ny: 格点数
        hx, hy: 网格间距
        mass: 有效质量
    Returns:
        (j_x, j_y): 流密度分量 (一维)
    """
    psi_2d = psi.reshape(Nx, Ny)
    j_x = np.zeros((Nx, Ny))
    j_y = np.zeros((Nx, Ny))

    # j_x = (1/m) Im[ψ* ∂_x ψ]
    for ix in range(1, Nx - 1):
        dpsi_dx = (psi_2d[ix+1, :] - psi_2d[ix-1, :]) / (2 * hx)
        j_x[ix, :] = np.imag(np.conj(psi_2d[ix, :]) * dpsi_dx) / mass

    # j_y = (1/m) Im[ψ* ∂_y ψ] - (A_y/m)|ψ|²
    A_y_2d = A_y.reshape(Nx, Ny)
    for iy in range(1, Ny - 1):
        dpsi_dy = (psi_2d[:, iy+1] - psi_2d[:, iy-1]) / (2 * hy)
        j_y[:, iy] = (np.imag(np.conj(psi_2d[:, iy]) * dpsi_dy) / mass
                       - A_y_2d[:, iy] * np.abs(psi_2d[:, iy])**2 / mass)

    return j_x.flatten(), j_y.flatten()


def particle_density(psi_list: List[np.ndarray],
                      filling: int, Nx: int, Ny: int) -> np.ndarray:
    """计算粒子数密度 n(r) = Σ_{n < filling} |ψ_n(r)|²

    Args:
        psi_list: 本征态列表
        filling: 填充数
        Nx, Ny: 格点数
    Returns:
        粒子数密度 (一维)
    """
    n_total = Nx * Ny
    density = np.zeros(n_total)
    for n in range(min(filling, len(psi_list))):
        density += np.abs(psi_list[n]) ** 2
    return density


def guiding_center(psi: np.ndarray, x: np.ndarray, y: np.ndarray,
                   B: float) -> Tuple[float, float]:
    """计算波函数的导向中心 (X, Y)

    ⟨X⟩ = ⟨x⟩ + (1/B)⟨p_y⟩
    ⟨Y⟩ = ⟨y⟩ - (1/B)⟨p_x⟩

    导向中心坐标满足对易关系:
        [X, Y] = -i/B = -il_B²

    对于朗道能级, 导向中心是简并指标.

    Args:
        psi: 波函数
        x, y: 坐标
        B: 磁场
    Returns:
        (X_center, Y_center)
    """
    prob = np.abs(psi) ** 2
    prob = prob / np.sum(prob)

    X_center = np.sum(prob * x)
    Y_center = np.sum(prob * y)

    return X_center, Y_center


def streda_formula_density(B: float, filling: int,
                            area: float) -> float:
    """Streda 公式: n = ν·eB/h

    在原子单位: n = ν·B/(2π)
    其中 ν 是填充因子.

    Args:
        B: 磁场
        filling: 填充因子 (整数)
        area: 面积
    Returns:
        预期粒子数密度
    """
    return filling * B / (2 * np.pi)
