"""
time_evolution.py — 边界态隐式时间演化
========================================

本模块实现拓扑绝缘体边界态波包的隐式时间演化,
用于研究边缘态在无序势中的传播和散射。

核心方法 (借鉴 061_b1g3 的隐式多步法):

1. **隐式中点法** (Implicit Midpoint):
    ψ^{n+1} = ψ^n - iΔt/ℏ · H · (ψ^n + ψ^{n+1})/2
    → (I + iΔtH/2ℏ) ψ^{n+1} = (I - iΔtH/2ℏ) ψ^n

2. **B1G3 三阶隐式多步法**:
    A₃ψ^{n+1} + A₂ψ^n + A₁ψ^{n-1} = (2Δt/√3) H ψ^{n+1/2}

    其中:
    A₃ = 1/2 + 1/√3
    A₂ = -2/√3
    A₁ = -1/2 + 1/√3

3. **Crank-Nicolson** (等价于隐式中点法对线性问题)

物理背景
--------
边界态波包传播:
    ψ(x, y, t=0) = 高斯波包 (局域在边缘, 具有初始动量 k₀)
    ψ(t) = exp(-iHt/ℏ) ψ(0)

观察:
- 波包沿边缘传播的速度 (群速度)
- 无序导致的背散射 (被时间反演对称性禁止)
- 波包在缺陷处的绕射

来源映射
--------
- 061_b1g3: B1G3 隐式多步 ODE 积分器
- 1099_amsontag: 大规模 ODE 系统并行求解
"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg
from typing import Tuple, Dict, Optional, Callable
from bhz_hamiltonian import BHZParameters, build_bhz_hamiltonian


def implicit_midpoint_step(H: sparse.csr_matrix, psi: np.ndarray,
                           dt: float) -> np.ndarray:
    """
    隐式中点法 (Crank-Nicolson) 单步推进。

    (I + iΔtH/2) ψ^{n+1} = (I - iΔtH/2) ψ^n

    由于 H 是 Hermitian, 此格式严格保幺正:
        ||ψ^{n+1}||² = ||ψ^n||²

    Parameters
    ----------
    H : csr_matrix
        哈密顿量 (ℏ=1 单位)
    psi : ndarray
        当前波函数
    dt : float
        时间步长

    Returns
    -------
    psi_new : ndarray
        新波函数
    """
    dim = H.shape[0]
    I_mat = sparse.eye(dim, format='csr', dtype=np.complex128)

    # 右端向量
    rhs_matrix = I_mat - 0.5j * dt * H
    rhs = rhs_matrix @ psi

    # 左端矩阵
    lhs_matrix = I_mat + 0.5j * dt * H

    # 求解线性系统
    psi_new = sparse_linalg.spsolve(lhs_matrix, rhs)

    return psi_new


def b1g3_step(H: sparse.csr_matrix, psi_curr: np.ndarray,
              psi_prev: np.ndarray, dt: float) -> np.ndarray:
    """
    B1G3 三阶隐式多步法单步推进。

    残差形式:
        A₃ψ^{n+1} + A₂ψ^n + A₁ψ^{n-1} = (2Δt/√3)(-iH)ψ_mid

    其中:
        A₃ = 1/2 + 1/√3
        A₂ = -2/√3
        A₁ = -1/2 + 1/√3
        ψ_mid = (ψ^{n+1} + ψ^n + ψ^{n-1}) / 3  (近似)

    简化实现: 将 B1G3 视为隐式格式,
    求解 (A₃ + i·2Δt/√3·H) ψ^{n+1} = -A₂ψ^n - A₁ψ^{n-1} + RHS_correction

    Parameters
    ----------
    H : csr_matrix
    psi_curr : ndarray
        ψ^n
    psi_prev : ndarray
        ψ^{n-1}
    dt : float

    Returns
    -------
    psi_next : ndarray
        ψ^{n+1}
    """
    sqrt3 = np.sqrt(3.0)
    A3 = 0.5 + 1.0 / sqrt3
    A2 = -2.0 / sqrt3
    A1 = -0.5 + 1.0 / sqrt3

    dim = H.shape[0]
    I_mat = sparse.eye(dim, format='csr', dtype=np.complex128)

    # 左端: A₃·I + i(2Δt/√3)·H
    # 简化: 使用 ψ_mid ≈ ψ^n 的近似
    lhs_matrix = A3 * I_mat + 1j * (2.0 * dt / sqrt3) * H

    # 右端: -A₂ψ^n - A₁ψ^{n-1}
    rhs = -A2 * psi_curr - A1 * psi_prev

    # 求解
    psi_next = sparse_linalg.spsolve(lhs_matrix, rhs)

    return psi_next


def backward_euler_step(H: sparse.csr_matrix, psi: np.ndarray,
                        dt: float) -> np.ndarray:
    """
    后向 Euler 单步 (一阶隐式, 无条件稳定但有数值耗散)。

    (I + iΔtH) ψ^{n+1} = ψ^n

    |g|² = 1/(1+(ωΔt)²) < 1 → 数值耗散

    Parameters
    ----------
    H : csr_matrix
    psi : ndarray
    dt : float

    Returns
    -------
    psi_new : ndarray
    """
    dim = H.shape[0]
    I_mat = sparse.eye(dim, format='csr', dtype=np.complex128)

    lhs = I_mat + 1j * dt * H
    psi_new = sparse_linalg.spsolve(lhs, psi)

    return psi_new


def create_edge_wavepacket(Nx: int, Ny: int, kx0: float,
                           sigma_x: float, sigma_y: float,
                           x0: float, y0: float,
                           hx: float, hy: float,
                           band: int = 0,
                           edge: str = 'bottom') -> np.ndarray:
    """
    创建局域在边缘的高斯波包初始态。

    ψ(x, y) = N × exp(-(x-x₀)²/(2σx²)) × exp(-(y-y₀)²/(2σy²)) × exp(ik₀x)
              × χ_band ⊗ χ_edge

    其中 χ_edge 将波包局域在特定边缘 (y=0 或 y=L)。

    Parameters
    ----------
    Nx, Ny : int
    kx0 : float
        初始动量
    sigma_x, sigma_y : float
        高斯宽度
    x0, y0 : float
        波包中心
    hx, hy : float
    band : int
        能带索引 (0=价带, 3=导带)
    edge : str
        'bottom' 或 'top'

    Returns
    -------
    psi : ndarray, shape (4*Nx*Ny,)
        归一化波函数
    """
    dim_total = 4 * Nx * Ny
    psi = np.zeros(dim_total, dtype=np.complex128)

    x = np.arange(Nx) * hx
    y = np.arange(Ny) * hy

    # 选择边缘 y 位置
    if edge == 'bottom':
        y_edge = y[0] + 2 * hy  # 靠近底边
    else:
        y_edge = y[-1] - 2 * hy  # 靠近顶边

    # 轨道×自旋索引 (band 决定哪个子带)
    # 4 个子带: 0=e↑, 1=h↑, 2=e↓, 3=h↓
    spin_orb_idx = band

    for ix in range(Nx):
        for iy in range(Ny):
            idx_2d = ix * Ny + iy
            idx_4d = spin_orb_idx * (Nx * Ny) + idx_2d

            # 高斯包络 × 平面波
            envelope_x = np.exp(-(x[ix] - x0) ** 2 / (2 * sigma_x ** 2))
            envelope_y = np.exp(-(y[iy] - y_edge) ** 2 / (2 * sigma_y ** 2))
            plane_wave = np.exp(1j * kx0 * x[ix])

            psi[idx_4d] = envelope_x * envelope_y * plane_wave

    # 归一化
    norm = np.sqrt(np.sum(np.abs(psi) ** 2))
    if norm > 1e-30:
        psi /= norm

    return psi


def wavepacket_evolution(H: sparse.csr_matrix, psi0: np.ndarray,
                         dt: float, n_steps: int,
                         method: str = 'crank_nicolson',
                         record_interval: int = 10) -> Dict:
    """
    波包时间演化模拟。

    Parameters
    ----------
    H : csr_matrix
    psi0 : ndarray
    dt : float
    n_steps : int
    method : str
        'crank_nicolson', 'b1g3', 'backward_euler'
    record_interval : int
        每隔多少步记录一次

    Returns
    -------
    result : dict
    """
    psi = psi0.copy()
    psi_prev = psi0.copy()  # 用于多步法

    times = [0.0]
    norms = [float(np.sum(np.abs(psi) ** 2))]
    energies = [float(np.real(psi.conj() @ H @ psi))]

    # 质心位置
    # (简化: 只追踪概率密度的一阶矩)

    for step in range(1, n_steps + 1):
        if method == 'crank_nicolson':
            psi = implicit_midpoint_step(H, psi, dt)
        elif method == 'b1g3':
            psi_new = b1g3_step(H, psi, psi_prev, dt)
            psi_prev = psi.copy()
            psi = psi_new
        elif method == 'backward_euler':
            psi = backward_euler_step(H, psi, dt)
        else:
            raise ValueError(f"未知方法: {method}")

        t = step * dt
        norm = float(np.sum(np.abs(psi) ** 2))
        energy = float(np.real(psi.conj() @ H @ psi))

        if step % record_interval == 0 or step == n_steps:
            times.append(t)
            norms.append(norm)
            energies.append(energy)

    return {
        'times': np.array(times),
        'norms': np.array(norms),
        'energies': np.array(energies),
        'final_psi': psi,
        'method': method,
        'dt': dt,
        'n_steps': n_steps,
        'norm_conservation': abs(norms[-1] - norms[0]),
        'energy_drift': abs(energies[-1] - energies[0])
    }


def time_evolution_report(result: Dict) -> str:
    """生成时间演化模拟报告。"""
    lines = []
    lines.append("=" * 60)
    lines.append("边界态波包时间演化报告")
    lines.append("=" * 60)
    lines.append(f"方法: {result['method']}")
    lines.append(f"时间步长: Δt = {result['dt']:.6e}")
    lines.append(f"总步数: {result['n_steps']}")
    lines.append(f"模拟时间: T = {result['times'][-1]:.6e}")

    lines.append(f"\n守恒量:")
    lines.append(f"  初始范数: {result['norms'][0]:.12f}")
    lines.append(f"  最终范数: {result['norms'][-1]:.12f}")
    lines.append(f"  范数偏差: {result['norm_conservation']:.6e}")
    lines.append(f"  初始能量: {result['energies'][0]:.10f} eV")
    lines.append(f"  最终能量: {result['energies'][-1]:.10f} eV")
    lines.append(f"  能量漂移: {result['energy_drift']:.6e} eV")

    # 稳定性判断
    if result['method'] == 'crank_nicolson':
        lines.append(f"\n  范数严格守恒: {'是' if result['norm_conservation'] < 1e-10 else '否'}")
        lines.append(f"  能量严格守恒: {'是' if result['energy_drift'] < 1e-10 else '否'}")
    elif result['method'] == 'backward_euler':
        lines.append(f"\n  范数衰减 (数值耗散): "
                     f"{result['norms'][-1]/result['norms'][0]:.10f}")

    return "\n".join(lines)
