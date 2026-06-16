"""
stability_analysis.py — 数值稳定性与收敛分析
==============================================

本模块执行径迹重建算法的数值稳定性分析:

    1. 向后 Euler 收敛性分析 (基于 [064])
    2. 传播方法对比: RK4 vs ETDRK4 vs Backward Euler
    3. 滤波放大矩阵谱分析 (von Neumann)
    4. 扰动传播与条件数估计
    5. 步长-精度权衡分析 (基于 [694] 的优化)

核心分析:

[向后 Euler 收敛性]
    Picard 迭代收敛条件:
        ||∂f/∂y|| · dt < 1
    对于磁场中的粒子:
        ||L|| ≈ ω = qB/(γm) < 1/dt

[滤波稳定性 — von Neumann 分析]
    放大矩阵: G = (I - KH) · F
    稳定条件: ρ(G) < 1 (谱半径)

[条件数]
    cond(C) = λ_max(C) / λ_min(C)
    大条件数 → 数值不稳定

[误差传播]
    δy_{n+1} = F_n · δy_n + ε_n
    ||δy_n|| ≤ Π ||F_k|| · ||δy_0|| + Σ ||ε_k||
"""

import math
import numpy as np
from typing import List, Dict, Tuple, Optional


# ============================================================
# [064] 向后 Euler 收敛性分析
# ============================================================
def analyze_backward_euler_convergence(omega_range, dt_range):
    """
    分析向后 Euler + Picard 迭代的收敛性

    对于线性测试方程 y' = λy:
        向后 Euler: y_{n+1} = y_n + dt · λ · y_{n+1}
        → y_{n+1} = y_n / (1 - dt·λ)

    Picard 迭代:
        yp^(j+1) = y_n + dt · λ · yp^(j)
        收敛条件: |dt · λ| < 1

    对于磁场问题:
        λ = iω (纯虚数, 回旋频率)
        |dt · λ| = dt · ω < 1

    Parameters
    ----------
    omega_range : list of float
        回旋频率范围 [1/mm]
    dt_range : list of float
        步长范围 [mm]

    Returns
    -------
    dict : {
        'convergence_map': list of list,  # 0/1 矩阵
        'critical_dt': dict,  # 每个 ω 的临界步长
        'convergence_rate': dict,  # 收敛速率
    }
    """
    n_omega = len(omega_range)
    n_dt = len(dt_range)

    convergence_map = [[0] * n_dt for _ in range(n_omega)]
    critical_dt = {}
    convergence_rate = {}

    for i, omega in enumerate(omega_range):
        crit_dt = None
        for j, dt in enumerate(dt_range):
            # Picard 收敛判据
            product = abs(dt * omega)
            if product < 1.0:
                convergence_map[i][j] = 1
                if crit_dt is None:
                    crit_dt = dt
            else:
                convergence_map[i][j] = 0

        critical_dt[omega] = crit_dt if crit_dt is not None else -1.0

        # 收敛速率 = |dt·ω| (越小越快)
        if omega > 1e-15:
            # 最小 dt 对应的速率
            min_dt = dt_range[0] if dt_range else 0.01
            rate = abs(min_dt * omega)
            convergence_rate[omega] = rate
        else:
            convergence_rate[omega] = 0.0

    return {
        'convergence_map': convergence_map,
        'omega_range': omega_range,
        'dt_range': dt_range,
        'critical_dt': critical_dt,
        'convergence_rate': convergence_rate,
    }


# ============================================================
# 传播方法对比
# ============================================================
def compare_propagation_methods(state0, bfield_func, step_lengths,
                                charge=1, mass=139.57):
    """
    对比三种传播方法的精度和稳定性

    1. RK4 (显式 4 阶 Runge-Kutta)
    2. Backward Euler (隐式 + Picard)
    3. ETDRK4 (指数时间差分)

    精度度量: 与参考解 (高精度 RK4) 的偏差
    稳定性度量: 能量守恒 ||p||² = const

    Parameters
    ----------
    state0 : ndarray, shape (6,)
    bfield_func : callable
    step_lengths : list of float
    charge, mass : 粒子参数

    Returns
    -------
    dict : {
        'method_errors': {method_name: [errors]},
        'energy_conservation': {method_name: [energy_errors]},
        'computation_cost': {method_name: float},
    }
    """
    from numerical_propagation import (
        propagate_rk4, propagate_backward_euler, propagate_etdrk4
    )

    # 参考解 (高精度 RK4)
    ref_state = propagate_rk4(state0, bfield_func, 100.0, n_steps=500,
                               charge=charge, mass=mass)
    p_ref = np.sqrt(ref_state[3]**2 + ref_state[4]**2 + ref_state[5]**2)

    results = {
        'method_errors': {'RK4': [], 'BackwardEuler': [], 'ETDRK4': []},
        'energy_conservation': {'RK4': [], 'BackwardEuler': [], 'ETDRK4': []},
    }

    # 初始能量
    p0 = np.sqrt(state0[3]**2 + state0[4]**2 + state0[5]**2)
    E0 = math.sqrt(p0**2 + mass**2)

    for step_len in step_lengths:
        n_steps = max(1, int(step_len))

        # RK4
        y_rk4 = propagate_rk4(state0, bfield_func, step_len, n_steps=n_steps,
                               charge=charge, mass=mass)
        err_rk4 = np.linalg.norm(y_rk4 - ref_state)
        p_rk4 = np.sqrt(y_rk4[3]**2 + y_rk4[4]**2 + y_rk4[5]**2)
        E_rk4 = math.sqrt(p_rk4**2 + mass**2)

        results['method_errors']['RK4'].append(err_rk4)
        results['energy_conservation']['RK4'].append(abs(E_rk4 - E0) / E0)

        # Backward Euler
        y_be, conv_info = propagate_backward_euler(
            state0, bfield_func, step_len, n_steps=n_steps,
            charge=charge, mass=mass, it_max=10)
        err_be = np.linalg.norm(y_be - ref_state)
        p_be = np.sqrt(y_be[3]**2 + y_be[4]**2 + y_be[5]**2)
        E_be = math.sqrt(p_be**2 + mass**2)

        results['method_errors']['BackwardEuler'].append(err_be)
        results['energy_conservation']['BackwardEuler'].append(abs(E_be - E0) / E0)

        # ETDRK4
        y_etd = propagate_etdrk4(state0, bfield_func, step_len, n_steps=n_steps,
                                  charge=charge, mass=mass)
        err_etd = np.linalg.norm(y_etd - ref_state)
        p_etd = np.sqrt(y_etd[3]**2 + y_etd[4]**2 + y_etd[5]**2)
        E_etd = math.sqrt(p_etd**2 + mass**2)

        results['method_errors']['ETDRK4'].append(err_etd)
        results['energy_conservation']['ETDRK4'].append(abs(E_etd - E0) / E0)

    results['step_lengths'] = step_lengths

    return results


# ============================================================
# 滤波放大矩阵谱分析
# ============================================================
def analyze_filter_amplification(filter_states, predicted_states, gains,
                                 jacobians=None):
    """
    分析 Kalman 滤波的放大矩阵谱特性

    放大矩阵: G_k = (I - K_k · H_k) · F_k
    稳定条件: ρ(G_k) < 1 (谱半径)

    条件数: cond(C_k) = λ_max / λ_min
    良好滤波: cond ~ 1 (各向同性协方差)
    病态滤波: cond >> 1 (某个方向高度约束)

    Parameters
    ----------
    filter_states : list of TrackState
    predicted_states : list of TrackState
    gains : list of ndarray
    jacobians : list of ndarray, optional

    Returns
    -------
    dict : {
        'spectral_radii': list,
        'condition_numbers': list,
        'trace_evolution': list,
        'is_stable': bool,
    }
    """
    spectral_radii = []
    condition_numbers = []
    trace_evolution = []

    for k, state in enumerate(filter_states):
        C = state.covariance

        # 条件数
        eigenvalues = np.linalg.eigvalsh(C)
        eigenvalues = np.maximum(eigenvalues, 1e-30)  # 防止零/负
        cond = max(eigenvalues) / max(min(eigenvalues), 1e-30)
        condition_numbers.append(float(cond))

        # 协方差迹 (总不确定性)
        trace_evolution.append(float(np.trace(C)))

        # 放大矩阵谱半径 (如果有雅可比)
        if jacobians and k < len(jacobians) and k < len(gains):
            F = jacobians[k]
            K = gains[k]
            H = np.zeros((2, 5))
            H[0, 3] = 1.0
            H[1, 4] = 1.0

            G = (np.eye(5) - K @ H) @ F
            eig_G = np.linalg.eigvals(G)
            rho = max(abs(eig_G))
            spectral_radii.append(float(rho))

    # 稳定性判断
    is_stable = all(r < 1.0 for r in spectral_radii) if spectral_radii else True

    return {
        'spectral_radii': spectral_radii,
        'condition_numbers': condition_numbers,
        'trace_evolution': trace_evolution,
        'is_stable': is_stable,
        'max_condition': max(condition_numbers) if condition_numbers else 0,
        'mean_spectral_radius': (sum(spectral_radii) / len(spectral_radii)
                                  if spectral_radii else 0),
    }


# ============================================================
# 扰动传播分析
# ============================================================
def perturbation_propagation(state, bfield_func, n_steps=20,
                             perturbation_scale=1e-4,
                             charge=1, mass=139.57):
    """
    分析初始扰动在传播中的放大

    对每个状态分量施加小扰动 δx_i,
    追踪传播后扰动的演化

    δy_n = Φ(n, 0) · δy_0
    其中 Φ 为状态转移矩阵

    放大因子: ||δy_n|| / ||δy_0||

    Parameters
    ----------
    state : ndarray, shape (6,)
    bfield_func : callable
    n_steps : int
    perturbation_scale : float
    charge, mass : float

    Returns
    -------
    dict : {
        'amplification_per_component': list,
        'max_amplification': float,
        'state_transfer_matrix': ndarray,
    }
    """
    from numerical_propagation import propagate_rk4

    # 参考传播
    y_ref = propagate_rk4(state, bfield_func, 100.0, n_steps=n_steps,
                           charge=charge, mass=mass)

    # 状态转移矩阵 (数值)
    n_dim = len(state)
    Phi = np.zeros((n_dim, n_dim))
    amplification = []

    for i in range(n_dim):
        # 正向扰动
        sp = np.array(state, dtype=float)
        sp[i] += perturbation_scale
        yp = propagate_rk4(sp, bfield_func, 100.0, n_steps=n_steps,
                            charge=charge, mass=mass)

        # 反向扰动
        sm = np.array(state, dtype=float)
        sm[i] -= perturbation_scale
        ym = propagate_rk4(sm, bfield_func, 100.0, n_steps=n_steps,
                            charge=charge, mass=mass)

        # 中心差分
        Phi[:, i] = (yp - ym) / (2.0 * perturbation_scale)

        # 该分量的放大因子
        delta_y = yp - y_ref
        amp = np.linalg.norm(delta_y) / perturbation_scale
        amplification.append(float(amp))

    # 矩阵条件数
    try:
        cond_Phi = float(np.linalg.cond(Phi))
    except (np.linalg.LinAlgError, ValueError):
        cond_Phi = float('inf')

    return {
        'amplification_per_component': amplification,
        'max_amplification': max(amplification),
        'mean_amplification': sum(amplification) / len(amplification),
        'state_transfer_matrix': Phi,
        'transfer_matrix_cond': cond_Phi,
    }


# ============================================================
# 综合稳定性报告
# ============================================================
def generate_stability_report(convergence_result, propagation_result,
                              filter_result, perturbation_result):
    """
    生成综合稳定性分析报告

    Parameters
    ----------
    convergence_result : dict (from analyze_backward_euler_convergence)
    propagation_result : dict (from compare_propagation_methods)
    filter_result : dict (from analyze_filter_amplification)
    perturbation_result : dict (from perturbation_propagation)

    Returns
    -------
    dict : 综合报告
    """
    report = {}

    # 1. 向后 Euler 收敛性
    crit_dts = convergence_result.get('critical_dt', {})
    n_convergent = sum(1 for v in crit_dts.values() if v > 0)
    n_total = len(crit_dts)
    report['backward_euler'] = {
        'convergence_fraction': n_convergent / max(n_total, 1),
        'critical_dt_range': (
            min(v for v in crit_dts.values() if v > 0) if any(v > 0 for v in crit_dts.values()) else 0,
            max(v for v in crit_dts.values() if v > 0) if any(v > 0 for v in crit_dts.values()) else 0,
        ),
    }

    # 2. 传播方法对比
    errors = propagation_result.get('method_errors', {})
    for method, err_list in errors.items():
        if err_list:
            report[f'propagation_{method}'] = {
                'mean_error': sum(err_list) / len(err_list),
                'max_error': max(err_list),
                'min_error': min(err_list),
            }

    # 3. 滤波稳定性
    report['filter_stability'] = {
        'is_stable': filter_result.get('is_stable', False),
        'max_condition_number': filter_result.get('max_condition', 0),
        'mean_spectral_radius': filter_result.get('mean_spectral_radius', 0),
    }

    # 4. 扰动放大
    report['perturbation'] = {
        'max_amplification': perturbation_result.get('max_amplification', 0),
        'mean_amplification': perturbation_result.get('mean_amplification', 0),
        'transfer_matrix_condition': perturbation_result.get('transfer_matrix_cond', 0),
    }

    # 5. 综合评级
    issues = []
    if not report['backward_euler']['convergence_fraction'] > 0.8:
        issues.append("向后 Euler 收敛率偏低")
    if report['filter_stability']['max_condition_number'] > 1e6:
        issues.append("滤波条件数过大")
    if report['perturbation']['max_amplification'] > 100:
        issues.append("扰动放大过高")

    report['overall_assessment'] = {
        'n_issues': len(issues),
        'issues': issues,
        'grade': 'A' if len(issues) == 0 else ('B' if len(issues) <= 1 else 'C'),
    }

    return report
