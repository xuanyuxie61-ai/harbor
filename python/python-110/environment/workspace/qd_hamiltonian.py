"""
qd_hamiltonian.py - 量子点哈密顿量构建模块

本模块基于原项目 087_biharmonic_exact 的高阶微分算子思想
与 508_hb_to_mm 的稀疏矩阵处理思想，构建量子点中电子与空穴的
有效质量薛定谔方程哈密顿量：

    H = - (hbar^2 / 2m*) nabla^2 + V_conf(r) + V_coulomb(r)

其中：
    - hbar: 约化普朗克常数
    - m*:   有效质量
    - V_conf: 量子点限制势（球形势阱近似）
    - V_coulomb: 电子-空穴库仑相互作用
"""

import numpy as np
from typing import Tuple, Dict
from utils import (
    validate_array_1d,
    validate_array_2d,
    build_sparse_hamiltonian_indices,
    spmatvec,
    tridiagonal_solve,
)


# 物理常数 (SI)
H_BAR = 1.054571817e-34       # J·s
M_ELECTRON = 9.10938356e-31   # kg
EV_TO_J = 1.602176634e-19     # J/eV


def effective_mass(material: str = "InAs") -> float:
    """
    返回常见半导体材料中电子的有效质量（以自由电子质量为单位）。
    
    参数:
        material: 材料名称，支持 "InAs", "GaAs", "InP"
    返回:
        m_star / m_e
    """
    table = {
        "InAs": 0.023,
        "GaAs": 0.067,
        "InP": 0.077,
    }
    return table.get(material, 0.067)


def spherical_confinement_potential(r: np.ndarray, R_dot: float, V0: float) -> np.ndarray:
    """
    球形势阱限制势：
    
        V(r) = 0          ,  r <= R_dot
        V(r) = V0         ,  r > R_dot
    
    参数:
        r:     径向坐标数组 (m)
        R_dot: 量子点半径 (m)
        V0:    势垒高度 (J)
    返回:
        V:     势能数组 (J)
    """
    r = validate_array_1d(r, "r")
    V = np.where(r <= R_dot, 0.0, V0)
    return V


def stark_field_potential(x: np.ndarray, F_field: float) -> np.ndarray:
    """
    外加电场（Stark 效应）引起的线性势：
    
        V_stark(x) = -e * F * x
    
    参数:
        x:       空间坐标 (m)
        F_field: 电场强度 (V/m)
    返回:
        V:       势能 (J)
    """
    x = validate_array_1d(x, "x")
    e_charge = 1.602176634e-19
    return -e_charge * F_field * x


def harmonic_confinement_potential(r: np.ndarray, hw: float) -> np.ndarray:
    """
    抛物线型限制势（常用于近似球形量子点）：
    
        V(r) = (1/2) m* omega^2 r^2 = (1/2) hbar omega (r / l_0)^2
    
    其中特征长度 l_0 = sqrt(hbar / (m* omega))。
    参数 hw 为 hbar * omega (J)。
    """
    r = validate_array_1d(r, "r")
    if hw <= 0:
        raise ValueError("hw must be positive")
    V = 0.5 * hw * (r ** 2)
    return V


def coulomb_potential_1d(x: np.ndarray, eps_r: float = 12.9) -> np.ndarray:
    """
    一维等效库仑势（用于电子-空穴对，激子）：
    
        V_c(x) = - e^2 / (4 pi epsilon_0 epsilon_r |x|)
    
    在 |x| -> 0 处做正则化：
        V_c(x) -> - e^2 / (4 pi epsilon_0 epsilon_r (|x| + a_B))
    
    其中 a_B 为有效玻尔半径。
    """
    x = validate_array_1d(x, "x")
    eps0 = 8.854187817e-12
    e_charge = 1.602176634e-19
    a_B_eff = 30.0e-9  # InAs 的有效玻尔半径约 30 nm
    abs_x = np.abs(x)
    V = - (e_charge ** 2) / (4.0 * np.pi * eps0 * eps_r * (abs_x + a_B_eff))
    return V


def build_kinetic_hamiltonian_1d(
    x_grid: np.ndarray, m_star_ratio: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    构造一维动能算符的离散化稀疏矩阵（有限差分法）：
    
        T_ij = - (hbar^2 / 2m*) * (delta_{i+1,j} - 2 delta_{i,j} + delta_{i-1,j}) / dx^2
    
    返回 COO 格式的 (rows, cols, data)。
    """
    x_grid = validate_array_1d(x_grid, "x_grid")
    n = x_grid.size
    if n < 3:
        raise ValueError("Grid must have at least 3 points")
    dx = float(x_grid[1] - x_grid[0])
    if abs(dx) < 1e-20:
        raise ValueError("Grid spacing dx is too small or non-uniform")
    m_star = m_star_ratio * M_ELECTRON
    coeff = (H_BAR ** 2) / (2.0 * m_star * (dx ** 2))
    rows, cols, data = build_sparse_hamiltonian_indices(n)
    # 重新缩放为实际物理系数
    data = data * coeff
    # 主对角线额外系数为 2，差分格式中系数为 -2 -> +2
    # build_sparse_hamiltonian_indices 已给出 (2, -1, -1)，乘以 coeff 即可
    return rows, cols, data


def add_potential_to_hamiltonian(
    rows: np.ndarray,
    cols: np.ndarray,
    data: np.ndarray,
    V: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将势能项 V(i) 以对角元形式加入 Hamiltonian。
    
    H_{ii} <- H_{ii} + V_i
    """
    V = validate_array_1d(V, "V")
    n = V.size
    new_rows = list(rows)
    new_cols = list(cols)
    new_data = list(data)
    for i in range(n):
        new_rows.append(i)
        new_cols.append(i)
        new_data.append(V[i])
    return (
        np.array(new_rows, dtype=int),
        np.array(new_cols, dtype=int),
        np.array(new_data, dtype=float),
    )


def sparse_to_dense(rows: np.ndarray, cols: np.ndarray, data: np.ndarray, n: int) -> np.ndarray:
    """将 COO 稀疏矩阵转换为稠密矩阵。"""
    A = np.zeros((n, n), dtype=float)
    for r, c, d in zip(rows, cols, data):
        if 0 <= r < n and 0 <= c < n:
            A[r, c] += d
    return A


def solve_eigenvalues_1d(
    x_grid: np.ndarray,
    m_star_ratio: float,
    potential_type: str = "spherical",
    **potential_params,
) -> Dict[str, np.ndarray]:
    """
    求解一维有效质量薛定谔方程的本征值与本征函数：
    
        [- (hbar^2 / 2m*) d^2/dx^2 + V(x)] psi_n(x) = E_n psi_n(x)
    
    参数:
        x_grid:        空间网格 (m)
        m_star_ratio:  有效质量 / 自由电子质量
        potential_type: "spherical" 或 "harmonic"
        potential_params: 额外势参数
    
    返回:
        dict with keys "energies_J", "energies_eV", "wavefunctions", "x_grid"
    """
    x_grid = validate_array_1d(x_grid, "x_grid")
    n = x_grid.size
    rows, cols, data = build_kinetic_hamiltonian_1d(x_grid, m_star_ratio)

    if potential_type == "spherical":
        R_dot = potential_params.get("R_dot", 5.0e-9)
        V0 = potential_params.get("V0", 0.5 * EV_TO_J)
        V = spherical_confinement_potential(x_grid, R_dot, V0)
    elif potential_type == "harmonic":
        hw = potential_params.get("hw", 0.05 * EV_TO_J)
        V = harmonic_confinement_potential(x_grid, hw)
    else:
        V = np.zeros_like(x_grid)

    rows, cols, data = add_potential_to_hamiltonian(rows, cols, data, V)
    H_dense = sparse_to_dense(rows, cols, data, n)

    # 对称化以消除数值误差
    H_dense = 0.5 * (H_dense + H_dense.T)

    # 强制 Hermitian 为实对称并去除极小虚部
    H_dense = np.real(H_dense)

    # 求解本征值问题（使用实对称矩阵特征值分解）
    eigvals, eigvecs = np.linalg.eigh(H_dense)

    # 按能量排序
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    return {
        "energies_J": eigvals,
        "energies_eV": eigvals / EV_TO_J,
        "wavefunctions": eigvecs,
        "x_grid": x_grid,
    }


def dipole_matrix_element_1d(
    psi_a: np.ndarray, psi_b: np.ndarray, x_grid: np.ndarray
) -> float:
    """
    计算一维电偶极矩矩阵元：
    
        d_{ab} = <psi_a | q x | psi_b> = q * integral psi_a*(x) x psi_b(x) dx
    
    采用梯形数值积分。
    """
    psi_a = validate_array_1d(psi_a, "psi_a")
    psi_b = validate_array_1d(psi_b, "psi_b")
    x_grid = validate_array_1d(x_grid, "x_grid")
    if not (psi_a.size == psi_b.size == x_grid.size):
        raise ValueError("Array sizes must match")
    dx = x_grid[1] - x_grid[0]
    integrand = psi_a * x_grid * psi_b
    d_ab = np.trapezoid(integrand, x_grid)
    # 对于离散化波函数，需要乘以 q (元电荷)
    e_charge = 1.602176634e-19
    return e_charge * d_ab


def exciton_binding_energy_1d(
    x_grid: np.ndarray,
    psi_e: np.ndarray,
    psi_h: np.ndarray,
    eps_r: float = 12.9,
) -> float:
    """
    估算一维等效激子结合能（Hartree 近似）：
    
        E_bind = - <psi_e psi_h | V_c(x_e - x_h) | psi_e psi_h>
              = - integral dx_e dx_h |psi_e(x_e)|^2 |psi_h(x_h)|^2 V_c(x_e - x_h)
    
    为简化计算，采用一维等效库仑势。
    """
    x_grid = validate_array_1d(x_grid, "x_grid")
    psi_e = validate_array_1d(psi_e, "psi_e")
    psi_h = validate_array_1d(psi_h, "psi_h")
    n = x_grid.size
    dx = float(x_grid[1] - x_grid[0])
    rho_e = np.abs(psi_e) ** 2
    rho_h = np.abs(psi_h) ** 2
    E_bind = 0.0
    for i in range(n):
        for j in range(n):
            dx_ij = x_grid[i] - x_grid[j]
            Vc = coulomb_potential_1d(np.array([dx_ij]), eps_r)[0]
            E_bind += rho_e[i] * rho_h[j] * Vc * (dx ** 2)
    return -E_bind  # 返回正值表示束缚能


def reduced_mass(m_e_star: float, m_h_star: float) -> float:
    """
    计算约化质量：
        mu* = (m_e* m_h*) / (m_e* + m_h*)
    """
    if m_e_star <= 0 or m_h_star <= 0:
        raise ValueError("Effective masses must be positive")
    return (m_e_star * m_h_star) / (m_e_star + m_h_star)
