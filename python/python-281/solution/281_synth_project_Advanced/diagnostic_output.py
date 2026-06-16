"""
diagnostic_output.py
====================
诊断输出与结果验证模块。

核心内容:
  1. 浓度场统计分析
  2. 质量守恒验证
  3. 能量平衡检查
  4. 收敛性诊断
  5. 数值误差估计
  6. 文本格式诊断报告输出

无可视化内容, 所有输出为纯文本/数值。
"""

import math
import numpy as np
from electrode_constants import C_MAX, R_GAS, FARADAY, T_REF
from thermodynamic_models import (
    redlich_kister_ocv, thermodynamic_factor,
    concentration_dependent_D
)


def concentration_profile_stats(c_field, r_grid):
    """
    浓度场统计分析。

    Parameters
    ----------
    c_field : ndarray
        浓度分布
    r_grid : ndarray
        径向坐标

    Returns
    -------
    dict
        统计信息
    """
    N = len(c_field)
    x_soc = c_field / C_MAX

    # 加权平均 (球坐标体积权重 ∝ r²)
    weights = r_grid ** 2
    total_weight = np.sum(weights)
    if total_weight > 0:
        mean_soc = np.sum(x_soc * weights) / total_weight
    else:
        mean_soc = np.mean(x_soc)

    # 梯度
    if N > 1:
        dr = r_grid[1] - r_grid[0] if N > 1 else 1.0
        dc_dr = np.gradient(c_field, dr) if dr > 0 else np.zeros(N)
    else:
        dc_dr = np.zeros(N)

    return {
        'mean_soc': float(mean_soc),
        'surface_soc': float(x_soc[-1]) if N > 0 else 0.0,
        'center_soc': float(x_soc[0]) if N > 0 else 0.0,
        'soc_range': float(np.max(x_soc) - np.min(x_soc)),
        'max_gradient': float(np.max(np.abs(dc_dr))),
        'mean_gradient': float(np.mean(np.abs(dc_dr))),
        'concentration_variance': float(np.var(c_field))
    }


def mass_conservation_detailed(c_field, r_grid, h, c_initial=None):
    """
    详细的质量守恒检查。

    球坐标中总 Li 量:
    M = 4π ∫₀ᴿ c(r)*r² dr

    使用 Simpson 积分 (高精度)。

    Parameters
    ----------
    c_field : ndarray
        当前浓度场
    r_grid : ndarray
        径向坐标
    h : float
        网格间距
    c_initial : ndarray, optional
        初始浓度场

    Returns
    -------
    dict
        质量守恒信息
    """
    N = len(c_field)
    integrand = c_field * r_grid ** 2

    # Simpson 积分 (如果 N 为奇数)
    if N >= 3 and N % 2 == 1:
        # Simpson's rule
        total = integrand[0] + integrand[-1]
        for i in range(1, N - 1, 2):
            total += 4 * integrand[i]
        for i in range(2, N - 2, 2):
            total += 2 * integrand[i]
        total *= h / 3.0
    else:
        # 梯形法则
        total = np.trapz(integrand, dx=h)

    M_current = 4.0 * math.pi * total

    result = {
        'total_li_mol_m2': M_current,
        'mean_concentration': float(np.mean(c_field)),
        'max_concentration': float(np.max(c_field)),
        'min_concentration': float(np.min(c_field))
    }

    if c_initial is not None:
        integrand_init = c_initial * r_grid ** 2
        if N >= 3 and N % 2 == 1:
            total_init = integrand_init[0] + integrand_init[-1]
            for i in range(1, N - 1, 2):
                total_init += 4 * integrand_init[i]
            for i in range(2, N - 2, 2):
                total_init += 2 * integrand_init[i]
            total_init *= h / 3.0
        else:
            total_init = np.trapz(integrand_init, dx=h)

        M_initial = 4.0 * math.pi * total_init
        if abs(M_initial) > 1e-30:
            rel_change = abs(M_current - M_initial) / abs(M_initial)
        else:
            rel_change = 0.0

        result['initial_total_li'] = M_initial
        result['mass_change_absolute'] = M_current - M_initial
        result['mass_change_relative'] = rel_change
        result['mass_conserved'] = rel_change < 0.05

    return result


def energy_balance_check(c_field, r_grid, h, T=T_REF):
    """
    能量平衡检查。

    计算 Gibbs 自由能:
    G = ∫ [μ(c) * c + (1/2)*D*(∇c)²] dV

    其中 μ(c) 为化学势, 第二项为梯度能。

    Parameters
    ----------
    c_field : ndarray
        浓度场
    r_grid : ndarray
        径向坐标
    h : float
        网格间距
    T : float
        温度

    Returns
    -------
    dict
        能量信息
    """
    N = len(c_field)

    # 化学能: ∫ μ(c)*c * 4π*r² dr
    chem_energy_density = np.zeros(N)
    for i in range(N):
        x = max(1e-12, min(1.0 - 1e-12, c_field[i] / C_MAX))
        # μ ≈ F*U(x) + R*T*ln(x)
        mu = FARADAY * redlich_kister_ocv(x) + R_GAS * T * math.log(x)
        chem_energy_density[i] = mu * c_field[i]

    integrand_chem = chem_energy_density * r_grid ** 2
    G_chem = 4.0 * math.pi * np.trapz(integrand_chem, dx=h)

    # 梯度能: ∫ (1/2)*D*(∇c)² * 4π*r² dr
    grad_energy = 0.0
    if N > 1:
        dc_dr = np.gradient(c_field, h)
        D_field = np.array([concentration_dependent_D(c_field[i], T) for i in range(N)])
        grad_density = 0.5 * D_field * dc_dr ** 2
        integrand_grad = grad_density * r_grid ** 2
        G_grad = 4.0 * math.pi * np.trapz(integrand_grad, dx=h)
    else:
        G_grad = 0.0

    return {
        'chemical_energy_J_m2': G_chem,
        'gradient_energy_J_m2': G_grad,
        'total_gibbs_J_m2': G_chem + G_grad,
        'mean_chemical_potential_J_mol': float(np.mean(chem_energy_density / np.maximum(c_field, 1e-10))),
    }


def convergence_diagnostics(newton_history, time_history=None):
    """
    收敛性诊断。

    Parameters
    ----------
    newton_history : list of float
        Newton 残差历史
    time_history : list of float, optional
        时间步历史

    Returns
    -------
    dict
        收敛诊断信息
    """
    if not newton_history:
        return {'converged': False, 'message': 'No history available'}

    residuals = np.array(newton_history)
    n_iters = len(residuals)

    # 收敛阶估计
    if n_iters >= 3:
        # 连续残差比
        ratios = []
        for i in range(2, n_iters):
            if residuals[i-1] > 1e-30 and residuals[i-2] > 1e-30:
                r1 = residuals[i] / max(residuals[i-1], 1e-30)
                r0 = residuals[i-1] / max(residuals[i-2], 1e-30)
                if r0 > 1e-14 and r1 > 1e-14:
                    try:
                        order = math.log(r1) / math.log(r0)
                        if abs(order) < 100:  # 过滤异常值
                            ratios.append(order)
                    except (ValueError, ZeroDivisionError):
                        pass
        mean_order = float(np.mean(ratios)) if ratios else 0.0
    else:
        mean_order = 0.0

    result = {
        'n_iterations': n_iters,
        'initial_residual': float(residuals[0]) if n_iters > 0 else 0.0,
        'final_residual': float(residuals[-1]) if n_iters > 0 else 0.0,
        'reduction_factor': float(residuals[-1] / max(residuals[0], 1e-30)) if n_iters > 0 else 0.0,
        'convergence_order': float(mean_order),
        'converged': float(residuals[-1]) < 1e-8 if n_iters > 0 else False,
        'monotonic': bool(np.all(np.diff(residuals) <= 0)) if n_iters > 1 else True
    }

    if time_history is not None:
        result['mean_dt'] = float(np.mean(time_history))
        result['min_dt'] = float(np.min(time_history))
        result['max_dt'] = float(np.max(time_history))

    return result


def spatial_error_estimate(c_coarse, c_fine, method='richardson'):
    """
    空间离散误差估计。

    Richardson 外推:
    e ≈ (c_fine - c_coarse) / (2^p - 1)
    其中 p 为格式阶数

    Parameters
    ----------
    c_coarse : ndarray
        粗网格解
    c_fine : ndarray
        细网格解 (需插值到相同网格)
    method : str
        'richardson' 或 'difference'

    Returns
    -------
    dict
        误差估计
    """
    if len(c_coarse) != len(c_fine):
        # 简单截取
        n = min(len(c_coarse), len(c_fine))
        c_coarse = c_coarse[:n]
        c_fine = c_fine[:n]

    diff = c_fine - c_coarse
    max_diff = np.max(np.abs(diff))
    rms_diff = np.sqrt(np.mean(diff ** 2))

    result = {
        'max_absolute_difference': float(max_diff),
        'rms_difference': float(rms_diff),
        'relative_max': float(max_diff / max(np.max(np.abs(c_fine)), 1e-14)),
        'relative_rms': float(rms_diff / max(np.sqrt(np.mean(c_fine**2)), 1e-14))
    }

    if method == 'richardson':
        # 4阶格式: p=4, 2^4-1=15
        p = 4
        factor = 2 ** p - 1
        result['richardson_error_est'] = float(max_diff / factor)

    return result


def format_diagnostic_report(stats, mass_info, energy_info, convergence=None,
                               stability_info=None, grain_info=None):
    """
    格式化诊断报告 (纯文本输出)。

    Parameters
    ----------
    stats : dict
        浓度场统计
    mass_info : dict
        质量守恒信息
    energy_info : dict
        能量信息
    convergence : dict, optional
        收敛诊断
    stability_info : dict, optional
        稳定性信息
    grain_info : dict, optional
        晶粒网络信息

    Returns
    -------
    str
        格式化报告
    """
    lines = []
    lines.append("=" * 70)
    lines.append("电池电极离子扩散模拟 - 诊断报告")
    lines.append("=" * 70)
    lines.append("")

    # 浓度场统计
    lines.append("【浓度场统计】")
    lines.append(f"  平均 SOC:          {stats.get('mean_soc', 0):.6f}")
    lines.append(f"  表面 SOC:          {stats.get('surface_soc', 0):.6f}")
    lines.append(f"  中心 SOC:          {stats.get('center_soc', 0):.6f}")
    lines.append(f"  SOC 不均匀度:      {stats.get('soc_range', 0):.6f}")
    lines.append(f"  最大浓度梯度:      {stats.get('max_gradient', 0):.4e} mol/m⁴")
    lines.append(f"  浓度方差:          {stats.get('concentration_variance', 0):.4e}")
    lines.append("")

    # 质量守恒
    lines.append("【质量守恒】")
    lines.append(f"  总 Li 量:          {mass_info.get('total_li_mol_m2', 0):.6e} mol/m²")
    if 'mass_change_relative' in mass_info:
        lines.append(f"  质量变化率:        {mass_info['mass_change_relative']:.4e}")
        lines.append(f"  质量守恒:          {'通过' if mass_info.get('mass_conserved', False) else '警告'}")
    lines.append("")

    # 能量
    lines.append("【Gibbs 自由能】")
    lines.append(f"  化学能:            {energy_info.get('chemical_energy_J_m2', 0):.4e} J/m²")
    lines.append(f"  梯度能:            {energy_info.get('gradient_energy_J_m2', 0):.4e} J/m²")
    lines.append(f"  总 Gibbs 能:       {energy_info.get('total_gibbs_J_m2', 0):.4e} J/m²")
    lines.append("")

    # 收敛
    if convergence:
        lines.append("【收敛诊断】")
        lines.append(f"  Newton 迭代次数:   {convergence.get('n_iterations', 'N/A')}")
        lines.append(f"  初始残差:          {convergence.get('initial_residual', 0):.4e}")
        lines.append(f"  最终残差:          {convergence.get('final_residual', 0):.4e}")
        lines.append(f"  收敛阶:            {convergence.get('convergence_order', 0):.2f}")
        lines.append(f"  收敛:              {'是' if convergence.get('converged', False) else '否'}")
        lines.append("")

    # 稳定性
    if stability_info:
        lines.append("【稳定性分析】")
        for key, val in stability_info.items():
            if isinstance(val, float):
                lines.append(f"  {key}: {val:.6e}")
            else:
                lines.append(f"  {key}: {val}")
        lines.append("")

    # 晶粒网络
    if grain_info:
        lines.append("【多晶晶粒网络】")
        for key, val in grain_info.items():
            lines.append(f"  {key}: {val}")
        lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)
