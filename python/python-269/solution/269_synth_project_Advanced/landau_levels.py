"""
landau_levels.py — 朗道能级求解与分析
============================================================

均匀磁场中二维自由电子的精确解:

    H = (1/2m)(p - A)²   →   E_n = ℏω_c(n + 1/2)

在 Landau 规范 A = (0, Bx, 0) 下, 波函数可分离:
    ψ_{n,k}(x,y) = e^{iky} · φ_n(x - x_0)

其中:
    x_0 = -k/(eB)    — 导向中心 (guiding center)
    φ_n(x)           — 谐振子本征函数 (Hermite 函数)
    ω_c = eB/m       — 回旋频率

Hermite 函数:
    φ_n(ξ) = (1/√(2^n·n!·√π·l_B)) · H_n(ξ/l_B) · exp(-ξ²/(2l_B²))
    其中 H_n 是 Hermite 多项式, ξ = x - x_0

朗道能级的简并度:
    g = BA/(2π) = Φ/Φ₀  (每个单位面积的态密度)

本模块功能:
1. 计算解析朗道能级
2. 用有限差分数值求解
3. 比较数值与解析结果
4. 计算波函数重叠积分
5. 分析简并度

参考文献:
    [1] Landau, L. D. Z. Phys. 64, 629 (1930)
    [2] Girvin, S. M. "Introduction to the Fractional Quantum Hall Effect"
"""

import numpy as np
from scipy import sparse
from scipy.special import hermite, factorial
from typing import Tuple, List, Dict, Any
from physical_constants import landau_level_energy, magnetic_length
from hamiltonian_assemble import assemble_hamiltonian_landau, ground_state_energy


def analytical_landau_levels(B: float, n_max: int) -> np.ndarray:
    """计算前 n_max 个解析朗道能级

    E_n = B(n + 1/2) (原子单位: ℏ = m = e = 1)

    Args:
        B: 磁场强度
        n_max: 最大朗道能级指标
    Returns:
        能级数组 [E_0, E_1, ..., E_{n_max}]
    """
    return np.array([landau_level_energy(n, B) for n in range(n_max + 1)])


def landau_wavefunction(n: int, B: float, x: np.ndarray,
                        x0: float = 0.0) -> np.ndarray:
    """计算第 n 个朗道能级的波函数 (x 方向)

    φ_n(x) = N_n · H_n((x-x₀)/l_B) · exp(-(x-x₀)²/(2l_B²))

    归一化: N_n = 1/√(2^n · n! · √π · l_B)

    Args:
        n: 朗道能级指标
        B: 磁场强度
        x: 坐标数组
        x0: 导向中心位置
    Returns:
        归一化波函数
    """
    l_B = magnetic_length(B)
    xi = (x - x0) / l_B
    H_n = hermite(n)
    N_n = 1.0 / np.sqrt(2**n * factorial(n) * np.sqrt(np.pi) * l_B)
    return N_n * H_n(xi) * np.exp(-xi**2 / 2.0)


def numerical_landau_levels(Nx: int, Ny: int, Lx: float, Ly: float,
                            B: float, n_levels: int = 5,
                            fd_order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """数值求解朗道能级

    构建有限差分哈密顿量并求解最低若干本征值.

    Args:
        Nx, Ny: 格点数
        Lx, Ly: 系统尺寸
        B: 磁场强度
        n_levels: 要求的能级数
        fd_order: 有限差分精度
    Returns:
        (numerical_energies, analytical_energies)
    """
    H = assemble_hamiltonian_landau(Nx, Ny, Lx, Ly, B, fd_order=fd_order)
    evals, _ = ground_state_energy(H, num_states=n_levels)

    E_analytical = analytical_landau_levels(B, n_levels - 1)

    return evals[:n_levels], E_analytical[:n_levels]


def landau_level_comparison(B: float, Nx_list: List[int],
                            n_levels: int = 3) -> Dict[str, Any]:
    """系统比较不同网格密度下的数值精度

    收敛分析:
        E_h - E_exact = C · h^p + O(h^{p+1})
    其中 p 是有限差分精度阶数.

    Richardson 外推:
        E_exact ≈ (h₂^p · E_{h₁} - h₁^p · E_{h₂}) / (h₂^p - h₁^p)

    Args:
        B: 磁场强度
        Nx_list: 不同网格密度的 Nx 值列表
        n_levels: 比较的能级数
    Returns:
        比较结果字典
    """
    Lx = 10.0
    Ly = 10.0
    fd_order = 4

    E_exact = analytical_landau_levels(B, n_levels - 1)

    results = {
        'B': B,
        'exact': E_exact.tolist(),
        'nx_values': Nx_list,
        'energies': [],
        'errors': [],
    }

    for Nx in Nx_list:
        Ny = Nx  # 正方形网格
        try:
            E_num, _ = numerical_landau_levels(Nx, Ny, Lx, Ly, B,
                                                n_levels, fd_order)
            errors = [abs(E_num[i] - E_exact[i]) for i in range(n_levels)]
            results['energies'].append(E_num.tolist())
            results['errors'].append(errors)
        except Exception as e:
            results['energies'].append(None)
            results['errors'].append([float('inf')] * n_levels)

    return results


def guiding_center_overlap(psi1: np.ndarray, psi2: np.ndarray,
                           x: np.ndarray, hx: float) -> complex:
    """计算两个朗道态的导向中心重叠积分

    S_{12} = ∫ φ₁*(x) · φ₂(x) dx

    对于不同导向中心的朗道态:
        S = exp(-d²/(4l_B²)) · L_n(d²/(2l_B²))
    其中 d 是导向中心间距, L_n 是 Laguerre 多项式.

    Args:
        psi1, psi2: 波函数
        x: 坐标
        hx: 网格间距
    Returns:
        重叠积分
    """
    return np.trapz(np.conj(psi1) * psi2, x)


def degeneracy_count(B: float, area: float) -> int:
    """计算给定面积和磁场下的朗道能级简并度

    g = BA/(2π) = Φ/Φ₀

    每个朗道能级可容纳 g 个电子 (不考虑自旋).
    填充因子 ν = N_e/g 决定了量子霍尔态的性质.

    Args:
        B: 磁场强度
        area: 系统面积
    Returns:
        简并度 (整数)
    """
    return max(1, int(round(B * area / (2 * np.pi))))


def density_of_states_2d(B: float, E: np.ndarray,
                         sigma: float = 0.05) -> np.ndarray:
    """二维电子气在磁场中的态密度

    ρ(E) = (1/2πl_B²) Σ_n δ(E - E_n)

    用高斯展宽替代 δ 函数:
        δ(E - E_n) → (1/σ√2π) exp(-(E-E_n)²/(2σ²))

    Args:
        B: 磁场强度
        E: 能量网格
        sigma: 展宽参数
    Returns:
        态密度 ρ(E)
    """
    l_B = magnetic_length(B)
    prefactor = 1.0 / (2 * np.pi * l_B**2)
    rho = np.zeros_like(E)

    for n in range(50):  # 足够多的朗道能级
        E_n = landau_level_energy(n, B)
        if E_n > E[-1] + 5 * sigma:
            break
        rho += np.exp(-0.5 * ((E - E_n) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))

    return prefactor * rho
