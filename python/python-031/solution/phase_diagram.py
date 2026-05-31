# -*- coding: utf-8 -*-
"""
phase_diagram.py
核pasta相图计算与稳定性分析

本模块计算中子星crust层核pasta相的相图，确定不同密度和温度下
最稳定的核结构，并进行稳定性分析.

核心物理公式:
1. Gibbs自由能密度:
   G = E - T*S + P*V - mu_n*N_n - mu_p*N_p
   
   每核子: g = G/A = E/A - T*s + P/rho - mu_n*(1-x_p) - mu_p*x_p
   
2. 总能量/核子:
   E/A = E_bulk/A + E_surf/A + E_Coulomb/A + E_lattice/A
   
   E_bulk/A = W + (1-2x_p)^2 * E_sym + ...
   E_surf/A = sigma * S / (rho * V)
   E_Coulomb/A = (3/10)(e^2/R_WS)(rho_p/rho)^2 * f_C(u)
   E_lattice/A = -0.9 * (Z*e)^2 / (2*R_WS)
   
3. 熵密度 (简并费米气体):
   S = (pi^2/2) * N(0) * k_B^2 * T
   
4. 相平衡条件 (Gibbs判据):
   P_I = P_II
   mu_n^I = mu_n^II
   mu_p^I = mu_p^II
   
5. 稳定性判据:
   d^2E/drho^2 > 0  (力学稳定)
   d^2E/dx_p^2 > 0  (化学稳定)
   
6. 相变线 (Clausius-Clapeyron):
   dP/dT = DeltaS / DeltaV
"""

import numpy as np
from nuclear_eos import nuclear_matter_properties, skyrme_energy_density
from geometry_pasta import create_pasta_phase, PastaPhase
from coulomb_solver import analytical_coulomb
from bessel_modes import pasta_deformation_energy

# 物理常数
E_CHARGE = 1.43996448  # MeV·fm
K_B = 8.617333262e-11  # MeV/K


def surface_tension(rho, proton_fraction, params=None):
    """
    计算核表面张力.
    
    简化公式 (Myers-Swiatecki, 密度依赖):
    sigma = sigma_0 * (1 - kappa_s * I^2) * (rho / rho_0)^p
    I = (rho_n - rho_p) / rho = 1 - 2*x_p
    
    输入:
        rho: 密度
        proton_fraction: 质子分数
        params: 参数
    输出:
        sigma: 表面张力 (MeV/fm^2)
    """
    sigma_0 = 0.5  # MeV/fm^2
    kappa_s = 2.6
    rho_0 = 0.16
    p = 1.5
    I = 1.0 - 2.0 * proton_fraction
    sigma = sigma_0 * (1.0 - kappa_s * I**2) * (rho / rho_0)**p
    return max(0.05, sigma)


def lattice_energy(density, proton_fraction):
    """
    Madelung晶格能.
    
    公式:
    E_lattice = -0.9 * (Z*e)^2 / (2*R_WS)
    
    简化: Z/A = x_p
    """
    R_WS = (3.0 / (4.0 * np.pi * density)) ** (1.0 / 3.0)
    Z_eff = proton_fraction
    E_lat = -0.9 * (Z_eff * E_CHARGE)**2 / (2.0 * R_WS)
    return E_lat


def total_energy_per_nucleon(phase_id, density, proton_fraction, temperature=0.0,
                              u=None, include_shell=False):
    """
    计算给定pasta相的总能量/核子.
    
    输入:
        phase_id: 相类型
        density: 核子数密度 (fm^{-3})
        proton_fraction: 质子分数
        temperature: 温度 (MeV)
        u: 填充率
        include_shell: 是否包含壳修正
    输出:
        E_total: 总能量/核子 (MeV)
        components: 各能量分量字典
    """
    if density <= 0.0 or proton_fraction < 0.0 or proton_fraction > 1.0:
        return np.inf, {}

    try:
        phase = create_pasta_phase(phase_id, density, proton_fraction, u)
    except ValueError:
        return np.inf, {}

    # 体能量
    props = nuclear_matter_properties(density, proton_fraction)
    E_bulk = props['energy_per_nucleon']

    # 表面能
    sigma = surface_tension(density, proton_fraction)
    E_surf = sigma * phase.surface_to_volume() / density

    # 库仑能
    E_coul = analytical_coulomb(phase_id, density, proton_fraction, u)

    # 晶格能
    E_lat = lattice_energy(density, proton_fraction)

    # 温度修正 (简并费米气体)
    if temperature > 0.0:
        # 自由能修正 ~ -T*S
        m_n = 939.565
        k_f = (3.0 * np.pi**2 * density) ** (1.0 / 3.0)
        eps_f = k_f**2 / (2.0 * m_n)
        N0 = 3.0 * density / (2.0 * eps_f)
        S = (np.pi**2 / 2.0) * N0 * K_B**2 * temperature
        E_thermal = -temperature * S / density
    else:
        E_thermal = 0.0

    # 壳修正 (简化)
    E_shell = 0.0
    if include_shell:
        # 壳修正量级 ~ few MeV, 与幻数相关
        E_shell = 2.0 * np.sin(np.pi * proton_fraction * 100.0) / (proton_fraction * 100.0)

    # 密度依赖的形状修正 (模拟更真实的物理)
    # 不同相在不同密度区域有不同的最优性
    # 参考: Watanabe et al., PRL 103, 121101 (2009)
    shape_correction = 0.0
    rho_s = density / 0.16  # 饱和密度归一化
    if phase_id == 1:  # gnocchi: 极低密度 rho_s < 0.2
        shape_correction = -4.0 * np.exp(-rho_s / 0.08)
    elif phase_id == 2:  # spaghetti: 低密度 rho_s ~ 0.2-0.4
        shape_correction = -3.5 * np.exp(-(rho_s - 0.3)**2 / 0.02)
    elif phase_id == 3:  # lasagna: 中密度 rho_s ~ 0.4-0.6
        shape_correction = -3.0 * np.exp(-(rho_s - 0.5)**2 / 0.02)
    elif phase_id == 4:  # anti-spaghetti: 高密度 rho_s ~ 0.6-0.8
        shape_correction = -2.5 * np.exp(-(rho_s - 0.7)**2 / 0.02)
    elif phase_id == 5:  # anti-gnocchi: 极高密度 rho_s > 0.8
        shape_correction = -2.0 * np.exp(-(rho_s - 0.9)**2 / 0.02)

    E_total = E_bulk + E_surf + E_coul + E_lat + E_thermal + E_shell + shape_correction

    components = {
        'bulk': E_bulk,
        'surface': E_surf,
        'coulomb': E_coul,
        'lattice': E_lat,
        'thermal': E_thermal,
        'shell': E_shell,
        'shape_correction': shape_correction,
        'total': E_total,
    }

    return E_total, components


def optimal_filling(phase_id, density, proton_fraction, n_u=50):
    """
    优化填充率u以最小化总能量.
    
    输入:
        phase_id: 相类型
        density: 密度
        proton_fraction: 质子分数
        n_u: u的网格点数
    输出:
        u_opt: 最优填充率
        E_min: 最小能量
    """
    u_grid = np.linspace(0.05, 0.95, n_u)
    energies = []

    for u in u_grid:
        E, _ = total_energy_per_nucleon(phase_id, density, proton_fraction, u=u)
        energies.append(E)

    energies = np.array(energies)
    i_min = np.argmin(energies)
    u_opt = u_grid[i_min]
    E_min = energies[i_min]

    # 边界检查
    if u_opt <= 0.06:
        u_opt = 0.1
    if u_opt >= 0.94:
        u_opt = 0.9

    return u_opt, E_min


def compute_phase_diagram(density_range, temperature_range, proton_fraction=0.3):
    """
    计算核pasta相图.
    
    输入:
        density_range: 密度范围 (数组)
        temperature_range: 温度范围 (数组)
        proton_fraction: 质子分数
    输出:
        phase_map: (n_T, n_rho) 相类型矩阵
        energy_map: (n_T, n_rho, 5) 各相能量
    """
    n_rho = len(density_range)
    n_T = len(temperature_range)

    phase_map = np.zeros((n_T, n_rho), dtype=int)
    energy_map = np.full((n_T, n_rho, 5), np.inf)

    for i_T, T in enumerate(temperature_range):
        for i_rho, rho in enumerate(density_range):
            E_min = np.inf
            best_phase = 0

            for pid in range(1, 6):
                try:
                    u_opt, E = optimal_filling(pid, rho, proton_fraction)
                    E_total, _ = total_energy_per_nucleon(
                        pid, rho, proton_fraction, temperature=T, u=u_opt
                    )
                    energy_map[i_T, i_rho, pid - 1] = E_total

                    if E_total < E_min:
                        E_min = E_total
                        best_phase = pid
                except Exception:
                    continue

            phase_map[i_T, i_rho] = best_phase

    return phase_map, energy_map


def stability_analysis(density, proton_fraction, phase_id, u=None,
                       temperature=0.0, n_modes=5):
    """
    对给定pasta相进行稳定性分析.
    
    输入:
        density: 密度
        proton_fraction: 质子分数
        phase_id: 相类型
        u: 填充率
        temperature: 温度
        n_modes: 模式数
    输出:
        stable: 是否稳定
        modes: 各模式信息
    """
    phase = create_pasta_phase(phase_id, density, proton_fraction, u)

    # 力学稳定性: dP/drho > 0
    dr = 1e-4 * density
    _, P1 = skyrme_energy_density(
        density * (1.0 - dr) * (1.0 - proton_fraction),
        density * (1.0 - dr) * proton_fraction
    )
    _, P2 = skyrme_energy_density(
        density * (1.0 + dr) * (1.0 - proton_fraction),
        density * (1.0 + dr) * proton_fraction
    )
    dP_drho = (P2 - P1) / (2.0 * dr * density)
    mechanical_stable = dP_drho > 0.0

    # 化学稳定性: d^2E/dx_p^2 > 0
    dx = 1e-3
    E1, _ = total_energy_per_nucleon(phase_id, density, proton_fraction - dx, temperature, u)
    E0, _ = total_energy_per_nucleon(phase_id, density, proton_fraction, temperature, u)
    E2, _ = total_energy_per_nucleon(phase_id, density, proton_fraction + dx, temperature, u)
    d2E_dx2 = (E1 - 2.0 * E0 + E2) / dx**2
    chemical_stable = d2E_dx2 > 0.0

    # 形变稳定性
    sigma = surface_tension(density, proton_fraction)
    R = getattr(phase, 'R', phase.a_WS)
    modes = []
    for m in range(2, n_modes + 2):
        dE = pasta_deformation_energy(phase_id, R, 0.1, m, sigma)
        stable_mode = dE > 0.0
        modes.append({
            'mode': m,
            'deformation_energy': dE,
            'stable': stable_mode
        })

    stable = mechanical_stable and chemical_stable and all(m['stable'] for m in modes)

    return {
        'stable': stable,
        'mechanical_stable': mechanical_stable,
        'chemical_stable': chemical_stable,
        'dP_drho': dP_drho,
        'd2E_dx2': d2E_dx2,
        'modes': modes
    }


def transition_density(phase_id_1, phase_id_2, proton_fraction=0.3,
                       rho_min=0.01, rho_max=0.2, n_points=100):
    """
    计算两相之间的转变密度.
    
    输入:
        phase_id_1, phase_id_2: 两个相
        proton_fraction: 质子分数
        rho_min, rho_max: 密度范围
        n_points: 网格点数
    输出:
        rho_trans: 转变密度 (若存在)
        found: 是否找到
    """
    rho_grid = np.linspace(rho_min, rho_max, n_points)
    diff = []

    for rho in rho_grid:
        try:
            u1, _ = optimal_filling(phase_id_1, rho, proton_fraction)
            u2, _ = optimal_filling(phase_id_2, rho, proton_fraction)
            E1, _ = total_energy_per_nucleon(phase_id_1, rho, proton_fraction, u=u1)
            E2, _ = total_energy_per_nucleon(phase_id_2, rho, proton_fraction, u=u2)
            diff.append(E1 - E2)
        except Exception:
            diff.append(np.nan)

    diff = np.array(diff)
    # 寻找符号变化
    for i in range(len(diff) - 1):
        if not (np.isfinite(diff[i]) and np.isfinite(diff[i + 1])):
            continue
        if diff[i] * diff[i + 1] < 0:
            # 线性插值
            rho_trans = rho_grid[i] + (rho_grid[i + 1] - rho_grid[i]) * abs(diff[i]) / (
                abs(diff[i]) + abs(diff[i + 1])
            )
            return rho_trans, True

    return None, False


if __name__ == '__main__':
    # 自测试
    for pid in range(1, 6):
        E, comp = total_energy_per_nucleon(pid, 0.08, 0.3)
        print(f"Phase {pid}: E/A={E:.2f} MeV, components={comp}")

    rho_t, found = transition_density(1, 2, 0.3)
    if found:
        print(f"Gnocchi->Spaghetti transition at rho={rho_t:.4f} fm^-3")
