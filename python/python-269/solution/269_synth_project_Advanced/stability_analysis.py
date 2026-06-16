"""
stability_analysis.py — 有限差分格式稳定性分析
============================================================

本模块分析有限差分格式的数值稳定性:

1. von Neumann 稳定性分析 (Fourier 方法):
   将数值解展开为 Fourier 模态: ψ_j^n = G^n · e^{ikjh}
   放大因子 G(k) 满足 |G(k)| ≤ 1 + CΔt (稳定性条件)

2. 矩阵稳定性分析:
   对于薛定谔方程 i∂_t ψ = Hψ, 时间演化算符 U = e^{-iHt}
   数值格式稳定的条件: ||U_num|| ≤ 1 + O(Δt)

3. CFL 条件:
   Δt < C · h^p / ||H||  (p 为空间精度阶数)

4. 能量守恒检验:
   对于哈密顿系统, 精确解满足 dE/dt = 0
   数值格式的能量漂移: |E(t) - E(0)|/|E(0)|

5. 网格收敛性:
   Richardson 外推估计实际收敛阶数

对于显式 Euler:
    G = 1 - iΔt·λ  →  |G|² = 1 + (Δt·λ)² > 1
    → 无条件不稳定!

对于 Crank-Nicolson:
    G = (1 - iΔt·λ/2)/(1 + iΔt·λ/2)  →  |G| = 1
    → 无条件稳定 (但不保辛)

对于 Suzuki-Trotter:
    U ≈ e^{-iTΔt/2} · e^{-iVΔt} · e^{-iTΔt/2} + O(Δt³)
    → 二阶精度, 保辛, 保酉

参考文献:
    [1] Strikwerda, J. C. "Finite Difference Schemes and PDEs" (SIAM, 2004)
    [2] Hairer, E. et al. "Geometric Numerical Integration" (Springer, 2006)
"""

import numpy as np
from scipy import sparse
from typing import Dict, Any, List, Tuple
from high_order_fd import build_1d_laplacian
from hamiltonian_assemble import assemble_hamiltonian_landau, ground_state_energy


def von_neumann_analysis(scheme: str, dx: float, dt: float,
                         B: float = 1.0) -> Dict[str, Any]:
    """von Neumann 稳定性分析

    分析不同时间积分格式的放大因子 G(k):

    1. 显式 Euler: G = 1 - iΔt·ω(k)
       |G|² = 1 + (Δt·ω)² > 1 → 不稳定

    2. 隐式 Euler: G = 1/(1 + iΔt·ω(k))
       |G| = 1/√(1 + (Δt·ω)²) < 1 → 稳定但耗散

    3. Crank-Nicolson: G = (1 - iΔt·ω/2)/(1 + iΔt·ω/2)
       |G| = 1 → 稳定且保模

    4. Suzuki-Trotter (二阶):
       U = e^{-iTΔt/2} · e^{-iVΔt} · e^{-iTΔt/2}
       ||U|| = 1 → 稳定且保酉

    其中 ω(k) 是空间离散算符的特征值:
        ω(k) = (2/h²)(1 - cos(kh))  [2阶 Laplacian]

    Args:
        scheme: 格式名称 ('euler_explicit', 'euler_implicit',
                'crank_nicolson', 'suzuki_trotter')
        dx: 空间步长
        dt: 时间步长
        B: 磁场强度
    Returns:
        稳定性分析结果
    """
    # 波矢网格
    k_max = np.pi / dx
    n_k = 1000
    k = np.linspace(-k_max, k_max, n_k)

    # 空间算符的特征值 (2阶 Laplacian)
    omega = (2.0 / dx**2) * (1.0 - np.cos(k * dx))
    # 加上磁场修正
    omega_total = omega + B**2 * dx**2 / 2.0

    results = {'scheme': scheme, 'k': k, 'omega': omega_total}

    if scheme == 'euler_explicit':
        G = 1.0 - 1j * dt * omega_total
        results['G'] = G
        results['G_modulus'] = np.abs(G)
        results['stable'] = np.all(np.abs(G) <= 1.0 + 1e-10)
        results['max_amplification'] = np.max(np.abs(G))
        results['cfl_condition'] = f"dt < 2/max(omega) = {2.0/np.max(omega_total):.6f}"

    elif scheme == 'euler_implicit':
        G = 1.0 / (1.0 + 1j * dt * omega_total)
        results['G'] = G
        results['G_modulus'] = np.abs(G)
        results['stable'] = True  # 无条件稳定
        results['max_amplification'] = np.max(np.abs(G))
        results['dissipative'] = np.max(np.abs(G)) < 1.0

    elif scheme == 'crank_nicolson':
        G = (1.0 - 0.5j * dt * omega_total) / (1.0 + 0.5j * dt * omega_total)
        results['G'] = G
        results['G_modulus'] = np.abs(G)
        results['stable'] = True
        results['unitary'] = np.allclose(np.abs(G), 1.0, atol=1e-10)
        results['max_amplification'] = np.max(np.abs(G))

    elif scheme == 'suzuki_trotter':
        # Suzuki-Trotter 保酉, |G| = 1
        results['G_modulus'] = np.ones_like(k)
        results['stable'] = True
        results['unitary'] = True
        results['order'] = 2
        results['error_scaling'] = "O(dt^3) per step, O(dt^2) global"

    else:
        raise ValueError(f"未知格式: {scheme}")

    return results


def energy_conservation_test(H: sparse.csr_matrix, psi0: np.ndarray,
                             dt: float, n_steps: int,
                             method: str = 'suzuki_trotter') -> Dict[str, Any]:
    """测试时间演化中的能量守恒

    对于精确的薛定谔方程演化:
        ⟨E⟩(t) = ⟨ψ(t)|H|ψ(t)⟩ = const

    数值格式的能量漂移:
        δE(t)/E(0) = |⟨E⟩(t) - ⟨E⟩(0)| / |⟨E⟩(0)|

    对于二阶格式: δE ~ O(dt²)
    对于四阶格式: δE ~ O(dt⁴)

    Args:
        H: 哈密顿量
        psi0: 初始波函数
        dt: 时间步长
        n_steps: 时间步数
        method: 时间积分方法
    Returns:
        能量守恒测试结果
    """
    from suzuki_trotter import evolve_step

    psi = psi0.copy()
    E0 = np.real(np.vdot(psi, H @ psi))
    energies = [E0]

    for step in range(n_steps):
        psi = evolve_step(H, psi, dt, method=method)
        E = np.real(np.vdot(psi, H @ psi))
        energies.append(E)

    energies = np.array(energies)
    drift = np.abs(energies - E0) / (abs(E0) + 1e-30)

    return {
        'method': method,
        'dt': dt,
        'n_steps': n_steps,
        'E0': E0,
        'E_final': energies[-1],
        'max_drift': np.max(drift),
        'mean_drift': np.mean(drift),
        'energies': energies,
        'stable': np.max(drift) < 0.01,
    }


def grid_convergence_study(Lx: float, Ly: float, B: float,
                           N_list: List[int],
                           target_level: int = 0) -> Dict[str, Any]:
    """网格收敛性研究

    计算不同网格密度下目标能级的误差, 并拟合收敛阶数.

    理论预期:
        E_h - E_exact = C · h^p
        log(E_h - E_exact) = log(C) + p · log(h)

    通过线性回归确定 p (实际收敛阶数).

    Args:
        Lx, Ly: 系统尺寸
        B: 磁场强度
        N_list: 不同网格密度的 N 值列表
        target_level: 目标朗道能级指标
    Returns:
        收敛性分析结果
    """
    from physical_constants import landau_level_energy

    E_exact = landau_level_energy(target_level, B)

    h_list = []
    errors = []

    for N in N_list:
        h = Lx / N
        h_list.append(h)
        try:
            H = assemble_hamiltonian_landau(N, N, Lx, Ly, B, fd_order=4)
            evals, _ = ground_state_energy(H, num_states=target_level + 1)
            E_num = evals[target_level]
            errors.append(abs(E_num - E_exact))
        except Exception:
            errors.append(float('inf'))

    h_arr = np.array(h_list)
    err_arr = np.array(errors)

    # 过滤掉无效数据
    valid = err_arr < 1e10
    h_valid = h_arr[valid]
    err_valid = err_arr[valid]

    # 拟合收敛阶数
    if len(h_valid) >= 2:
        log_h = np.log(h_valid)
        log_err = np.log(err_valid + 1e-30)
        # 线性回归: log_err = log_C + p * log_h
        p_fit, log_C = np.polyfit(log_h, log_err, 1)
        convergence_order = p_fit
    else:
        convergence_order = float('nan')

    return {
        'N_list': N_list,
        'h_list': h_list,
        'errors': errors,
        'E_exact': E_exact,
        'convergence_order': convergence_order,
        'expected_order': 4,  # 4th order FD
    }


def cfl_condition(H: sparse.csr_matrix, safety_factor: float = 0.9) -> float:
    """计算 CFL 稳定性条件的最大时间步长

    对于显式格式:
        dt_max = safety_factor · 2 / ρ(H)
    其中 ρ(H) = max|λ| 是谱半径.

    Args:
        H: 哈密顿量
        safety_factor: 安全因子 (0 < safety < 1)
    Returns:
        最大允许时间步长
    """
    from scipy.sparse.linalg import eigsh

    try:
        lambda_max = eigsh(H, k=1, which='LA',
                           return_eigenvectors=False)[0]
        return safety_factor * 2.0 / abs(lambda_max)
    except Exception:
        # 使用范数估计
        return safety_factor * 2.0 / sparse.linalg.norm(H, 'fro')


def condition_number_vs_h(Lx: float, B: float,
                          N_list: List[int]) -> List[float]:
    """分析条件数随网格间距的变化

    对于离散 Laplacian:
        κ(L_h) ~ O(1/h²) → 随网格加密而恶化

    对于磁场哈密顿量:
        κ(H_h) ~ O(1/h² + B²L²) → 强磁场加剧病态

    Args:
        Lx: 系统尺寸
        B: 磁场
        N_list: 网格密度列表
    Returns:
        条件数列表
    """
    from matrix_analyze_qhe import estimate_condition_number

    conditions = []
    for N in N_list:
        try:
            H = assemble_hamiltonian_landau(N, N, Lx, Lx, B, fd_order=2)
            cond = estimate_condition_number(H)
            conditions.append(cond)
        except Exception:
            conditions.append(float('inf'))

    return conditions
