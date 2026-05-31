"""
================================================================================
诊断与收敛监测模块 (diagnostics_convergence.py)
================================================================================

在博士级CFD计算中，收敛性诊断与数值稳定性分析至关重要。
本模块提供：
  1. 残差历史监测与收敛阶估计
  2. 矩阵条件数监测（与linear_algebra_engine联动）
  3. 能量守恒误差分析
  4. 网格收敛性指标（GCI）

核心公式:
    Richardson外推收敛阶:
        p = log( (f_{2h} - f_{4h}) / (f_h - f_{2h}) ) / log(2)

    Grid Convergence Index (GCI):
        GCI = F_s · |ε| / (r^p - 1)

    其中 ε = (f_h - f_{2h}) / f_h，r = 2 为加密比，F_s = 1.25 为安全因子。
================================================================================
"""

import numpy as np
from utils_numerical import safe_divide


def estimate_convergence_order(residuals: list) -> dict:
    """
    从残差历史估计收敛阶

    假设残差按几何级数衰减: r_{k+1} = C · r_k^p
    则 log(r_{k+1}) = log(C) + p · log(r_k)

    通过线性回归估计收敛阶 p。
    """
    if len(residuals) < 10:
        return {'order': None, 'R_squared': 0.0}

    log_r = np.log(np.maximum(residuals, 1e-16))
    x = log_r[:-1]
    y = log_r[1:]

    # 线性回归
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)

    if ss_xx < 1e-14:
        return {'order': None, 'R_squared': 0.0}

    p = ss_xy / ss_xx
    r_sq = ss_xy ** 2 / (ss_xx * np.sum((y - y_mean) ** 2) + 1e-14)

    return {
        'order': float(p),
        'R_squared': float(r_sq),
        'asymptotic_rate': float(10 ** p) if p < 0 else None
    }


def compute_gci(fine: float, medium: float, coarse: float,
                r: float = 2.0, p: float = None, Fs: float = 1.25) -> dict:
    """
    计算网格收敛指标 (GCI, Roache 1994)

    参数:
        fine: 细网格解
        medium: 中等网格解
        coarse: 粗网格解
        r: 网格加密比
        p: 已知收敛阶（若为None则自动估计）
        Fs: 安全因子

    返回:
        dict 包含GCI、观察收敛阶、渐近范围
    """
    if p is None:
        # 从三组解估计收敛阶
        if abs(fine - medium) < 1e-14 or abs(medium - coarse) < 1e-14:
            p = 2.0  # 默认二阶
        else:
            p = np.log(abs(coarse - medium) / abs(medium - fine)) / np.log(r)
            p = np.clip(p, 0.5, 4.0)

    epsilon = safe_divide(fine - medium, fine)
    gci = Fs * abs(epsilon) / (r ** p - 1.0)

    # 渐近范围检查
    asymptotic_range = safe_divide(
        (r ** p - 1.0) * abs(coarse - medium),
        (r ** p * (r ** p - 1.0)) * abs(medium - fine) + 1e-14
    )

    return {
        'p_observed': float(p),
        'gci_fine_medium': float(gci),
        'epsilon': float(epsilon),
        'asymptotic_range': float(asymptotic_range),
        'mesh_acceptable': asymptotic_range > 0.8 and gci < 0.05
    }


def check_energy_conservation(Q_history: list, gamma: float = 1.4) -> dict:
    """
    检查能量守恒误差

    对于封闭系统，总能量应满足：

        d/dt ∫_Ω ρE dV = - ∮_∂Ω (ρE + p) u · n dS + viscous_dissipation

    数值能量误差:
        ε_E = |E^{n+1} - E^n| / (|E^n| · Δt)

    参数:
        Q_history: 守恒变量历史列表 [(ny,nx,4), ...]
        gamma: 比热比

    返回:
        dict 包含能量历史、守恒误差、相对漂移
    """
    total_energy = []
    for Q in Q_history:
        E_total = np.sum(Q[..., 3])  # ρE 的总和
        total_energy.append(float(E_total))

    energy = np.array(total_energy)
    if len(energy) < 2:
        return {'energy': energy, 'drift': 0.0, 'max_relative_error': 0.0}

    # 相对漂移
    E0 = energy[0]
    drift = (energy[-1] - E0) / (abs(E0) + 1e-14)

    # 每步相对变化
    rel_changes = np.abs(np.diff(energy)) / (np.abs(energy[:-1]) + 1e-14)
    max_rel_error = np.max(rel_changes) if len(rel_changes) > 0 else 0.0

    return {
        'energy': energy,
        'drift': float(drift),
        'max_relative_error': float(max_rel_error),
        'energy_conserved': abs(drift) < 0.01
    }


def compute_mass_flow_rate(Q: np.ndarray, y: np.ndarray, gamma: float = 1.4) -> dict:
    """
    计算质量流量守恒

    入口质量流量:  ṁ_in = ∫ ρu dy
    出口质量流量:  ṁ_out = ∫ ρu dy

    守恒要求: |ṁ_in - ṁ_out| / ṁ_in < tol
    """
    rho = Q[:, :, 0]
    u = safe_divide(Q[:, :, 1], rho)

    ny = len(y)
    dy = np.diff(y)
    dy = np.concatenate([dy, [dy[-1]]])

    # 入口 (i=0)
    m_dot_in = np.sum(rho[:, 0] * u[:, 0] * dy)

    # 出口 (i=-1)
    m_dot_out = np.sum(rho[:, -1] * u[:, -1] * dy)

    # 误差
    error = abs(m_dot_in - m_dot_out) / (abs(m_dot_in) + 1e-14)

    return {
        'mass_flow_in': float(m_dot_in),
        'mass_flow_out': float(m_dot_out),
        'relative_error': float(error),
        'mass_conserved': error < 0.05
    }


def monitor_cfl_stability(u: np.ndarray, v: np.ndarray, c: np.ndarray,
                          dx: float, dy: np.ndarray, dt: float,
                          gamma: float = 1.4) -> dict:
    """
    监测CFL数与稳定性

    CFL条件:
        CFL_x = (|u| + c) Δt / Δx
        CFL_y = (|v| + c) Δt / Δy

    显式格式要求 CFL ≤ 1（对流），粘性项要求 ν Δt / Δx² ≤ 0.5
    """
    cfl_x = (np.abs(u) + c) * dt / dx
    cfl_y = (np.abs(v) + c) * dt / dy[:, None]

    cfl_x_max = np.max(cfl_x)
    cfl_y_max = np.max(cfl_y)

    # 粘性CFL
    nu = 1.0 / 1000.0  # 简化
    cfl_visc = nu * dt / (dx ** 2)

    stable = (cfl_x_max < 1.0) and (cfl_y_max < 1.0) and (cfl_visc < 0.5)

    return {
        'cfl_x_max': float(cfl_x_max),
        'cfl_y_max': float(cfl_y_max),
        'cfl_viscous': float(cfl_visc),
        'stable': stable,
        'recommendation': 'reduce dt' if not stable else 'stable'
    }


def print_diagnostics_header():
    """打印诊断表头"""
    print("=" * 80)
    print(f"{'Step':>6} {'Time':>10} {'Resid':>12} {'CFL_x':>8} {'CFL_y':>8} {'Energy':>12} {'MassErr':>10}")
    print("-" * 80)


def print_diagnostics_row(step: int, time: float, residual: float,
                          cfl_x: float, cfl_y: float, energy: float, mass_err: float):
    """打印诊断行"""
    print(f"{step:>6} {time:>10.4f} {residual:>12.4e} {cfl_x:>8.3f} {cfl_y:>8.3f} {energy:>12.4e} {mass_err:>10.4e}")
