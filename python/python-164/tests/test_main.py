"""
main.py
PEM 燃料电池阴极催化剂层衰减多物理场耦合模拟系统

统一入口，零参数可运行。

本程序集成以下模块，完成从操作条件生成、电化学反应求解、
传质扩散模拟、颗粒熟化演化、碳腐蚀传播、ECSA 损失评估、
催化剂负载优化到形貌退化分析的全流程计算。

科学问题:
  能源系统：氢燃料电池催化剂衰减
  
  具体研究目标:
    建立 PEM 燃料电池阴极催化剂层 (CCL) 中 Pt 纳米颗粒
    电化学溶解-奥斯瓦尔德熟化-碳载体腐蚀多物理场耦合衰减模型，
    量化长期运行下的 ECSA 损失与性能衰减轨迹，
    并优化催化剂负载分布以延缓衰减。
"""

import numpy as np
import sys
import time

# 导入各模块
from ccl_grid import generate_ccl_parameter_grid, sample_operating_condition
from butler_volmer import (orr_kinetic_parameters, exchange_current_density,
                           solve_overpotential_muller, solve_overpotential_wdk)
from diffusion_solver import (effective_diffusivity, solve_diffusion_tridiagonal,
                               solve_diffusion_banded)
from ripening_model import (pt_dissolution_parameters, evolve_size_distribution,
                             disk_distance_stats_monte_carlo, critical_radius,
                             lsw_analytical_r3, moment_size_distribution)
from carbon_corrosion import (corrosion_current_density, corrosion_front_velocity,
                               solve_corrosion_propagation, structural_integrity_loss)
from ecsa_calculator import (ecsa_from_size_distribution, ecsa_loss_kinetics,
                              build_stability_jacobian, stability_analysis_max_eigenvalue,
                              voltage_loss_from_ecsa, total_ecsa_loss_model)
from catalyst_optimizer import (optimize_catalyst_loading, power_performance,
                                 catalyst_cost, sensitivity_analysis)
from sparse_assembler import SparseAssembler, harwell_boeing_metadata
from morphology_evolution import (box_counting_dimension, effective_surface_area_fractal,
                                   enumerate_catalyst_surface_states,
                                   pore_network_connectivity_map,
                                   morphology_degradation_index)

# 物理常数
FARADAY = 96485.33212
GAS_CONSTANT = 8.314462618


def run_simulation():
    """
    执行完整的催化剂衰减模拟流程。
    """
    print("=" * 70)
    print("PEM 燃料电池阴极催化剂层多物理场耦合衰减模拟系统")
    print("=" * 70)
    print()
    
    t_start = time.time()
    
    # =====================================================================
    # Step 1: 生成操作条件参数网格
    # =====================================================================
    print("[Step 1] 生成 CCL 操作条件参数网格 ...")
    param_grid = generate_ccl_parameter_grid()
    print(f"  生成 {param_grid['num_points']} 个采样点")
    
    # 选择典型操作条件 (索引 60 附近，中等负载)
    idx = min(60, param_grid['num_points'] - 1)
    cond = sample_operating_condition(param_grid, idx)
    T_op = cond['temperature_K']
    RH_op = cond['relative_humidity_pct']
    E_cell = cond['cell_potential_V']
    L_pt = cond['pt_loading_mg_cm2']
    S_c0 = cond['carbon_surface_area_m2_g']
    print(f"  典型条件: T={T_op:.2f} K, RH={RH_op:.1f}%, E={E_cell:.3f} V,")
    print(f"            L_Pt={L_pt:.3f} mg/cm^2, S_c={S_c0:.1f} m^2/g")
    print()
    
    # =====================================================================
    # Step 2: 求解电化学反应过电位 (Butler-Volmer)
    # =====================================================================
    print("[Step 2] 求解 ORR Butler-Volmer 电化学动力学 ...")
    orr_params = orr_kinetic_parameters(T_op)
    
    # 氧气浓度 (根据 RH 和温度估算)
    C_O2_bulk = 1.2 * (RH_op / 100.0) * (298.15 / T_op)
    j0 = exchange_current_density(T_op, C_O2_bulk,
                                   j0_ref=orr_params['j0_ref'],
                                   E_a=orr_params['E_a'])
    
    eta_muller = solve_overpotential_muller(
        E_cell, orr_params['E_eq'], orr_params['R_ct'], j0,
        orr_params['alpha_a'], orr_params['alpha_c'],
        orr_params['n'], T_op
    )
    
    eta_wdk = solve_overpotential_wdk(
        E_cell, orr_params['E_eq'], orr_params['R_ct'], j0,
        orr_params['alpha_a'], orr_params['alpha_c'],
        orr_params['n'], T_op
    )
    
    # 取两种方法的平均值作为最终过电位
    eta_overall = 0.5 * (eta_muller + eta_wdk)
    
    # 安全计算 Butler-Volmer 电流密度
    arg_pos = orr_params['alpha_a'] * orr_params['n'] * FARADAY * eta_overall / (GAS_CONSTANT * T_op)
    arg_neg = -orr_params['alpha_c'] * orr_params['n'] * FARADAY * eta_overall / (GAS_CONSTANT * T_op)
    arg_pos = np.clip(arg_pos, -350, 350)
    arg_neg = np.clip(arg_neg, -350, 350)
    
    j_cell = j0 * (np.exp(arg_pos) - np.exp(arg_neg))
    
    # 边界保护
    j_cell = float(np.clip(j_cell, -1e4, 1e4))
    
    print(f"  交换电流密度 j0 = {j0:.4e} A/m^2")
    print(f"  Muller 法过电位 = {eta_muller:.6f} V")
    print(f"  WDK 法过电位   = {eta_wdk:.6f} V")
    print(f"  综合过电位     = {eta_overall:.6f} V")
    print(f"  电池电流密度   = {j_cell:.4e} A/m^2")
    print()
    
    # =====================================================================
    # Step 3: 求解氧气在 CCL 中的扩散分布
    # =====================================================================
    print("[Step 3] 求解 CCL 传质扩散方程 ...")
    
    D_bulk = 2.1e-5  # O2 in N2, m^2/s
    epsilon_cl = 0.4
    D_eff = effective_diffusivity(D_bulk, epsilon_cl)
    
    # 反应速率常数 (与电流密度关联)
    k_rxn = max(abs(j_cell) / (4.0 * FARADAY * max(C_O2_bulk, 1e-6)), 0.1)
    
    L_ccl = 10e-6  # 10 um
    C_0 = C_O2_bulk
    N_grid = 51
    
    x_grid, C_profile = solve_diffusion_tridiagonal(D_eff, k_rxn, L_ccl, C_0, N_grid)
    _, C_profile_band = solve_diffusion_banded(D_eff, k_rxn, L_ccl, C_0, N_grid)
    
    # 验证两种求解器一致性
    diff_solver = np.max(np.abs(C_profile - C_profile_band))
    
    print(f"  有效扩散系数 D_eff = {D_eff:.4e} m^2/s")
    print(f"  反应速率常数 k_rxn = {k_rxn:.4e} 1/s")
    print(f"  CCL 厚度 = {L_ccl*1e6:.1f} um")
    print(f"  O2 浓度 (入口) = {C_profile[0]:.4f} mol/m^3")
    print(f"  O2 浓度 (膜侧) = {C_profile[-1]:.4f} mol/m^3")
    print(f"  循环约化与带状 LU 最大偏差 = {diff_solver:.4e}")
    print()
    
    # =====================================================================
    # Step 4: Pt 纳米颗粒溶解-熟化演化
    # =====================================================================
    print("[Step 4] 模拟 Pt 纳米颗粒 Ostwald 熟化 ...")
    
    pt_params = pt_dissolution_parameters()
    # 使用操作温度更新参数
    pt_params['T'] = T_op
    
    # 初始颗粒尺寸分布 (对数正态分布)
    # 现代 PEMFC 催化剂典型粒径 3-5 nm
    np.random.seed(42)
    r_mean = 4.0e-9  # 4.0 nm
    r_std = 0.8e-9
    n_particles = 20
    radii_initial = np.random.lognormal(np.log(r_mean), 0.15, n_particles)
    radii_initial = np.clip(radii_initial, 1.0e-9, 15e-9)
    
    # 体相 Pt^2+ 浓度
    # 设置临界半径约为 4 nm: C_bulk = C_sat_inf * exp(2*gamma*V_m/(4nm*R*T))
    # 这样大于 4 nm 的颗粒缓慢长大，小于 4 nm 的颗粒溶解
    exponent_rc = (2.0 * pt_params['gamma'] * pt_params['V_m']) \
                  / (4.0e-9 * GAS_CONSTANT * T_op)
    C_bulk_Pt = pt_params['C_sat_inf'] * np.exp(exponent_rc)
    
    rc = critical_radius(pt_params['gamma'], pt_params['V_m'], T_op,
                         C_bulk_Pt, pt_params['C_sat_inf'])
    
    # 演化 500 小时 (约 21 天) 以展示中长期衰减趋势
    dt_ripen = 3600.0  # 1 hour
    n_steps = 500
    radii_history = evolve_size_distribution(
        radii_initial, pt_params['D_Pt2'], pt_params['V_m'],
        pt_params['C_sat_inf'], C_bulk_Pt,
        pt_params['gamma'], T_op, dt_ripen, n_steps
    )
    
    # LSW 理论对比
    r0_mean = np.mean(radii_initial)
    r_lsw = lsw_analytical_r3(500.0 * 3600.0, r0_mean,
                               pt_params['gamma'], pt_params['D_Pt2'],
                               pt_params['V_m'], pt_params['C_sat_inf'], T_op)
    
    # 颗粒间距统计
    mean_dist, var_dist = disk_distance_stats_monte_carlo(radii_initial, radii_initial)
    
    print(f"  初始平均半径 = {r0_mean*1e9:.2f} nm")
    print(f"  500h后平均半径 = {np.mean(radii_history[-1])*1e9:.2f} nm")
    print(f"  LSW 理论预测  = {r_lsw*1e9:.2f} nm")
    print(f"  临界半径 r_c  = {rc*1e9:.4f} nm")
    print(f"  颗粒间距统计: mean={mean_dist:.4f}, var={var_dist:.6f}")
    print()
    
    # =====================================================================
    # Step 5: 碳腐蚀传播模拟
    # =====================================================================
    print("[Step 5] 模拟碳载体腐蚀传播 ...")
    
    nx = 51
    dx = L_ccl / (nx - 1)
    S_c_initial = np.ones(nx) * S_c0
    
    v_corr = corrosion_front_velocity(E_cell, T_op)
    k_corr = 1e-5
    theta_pore = epsilon_cl
    
    # 固定时间步长和步数，避免 v_corr 极小时的除零问题
    dt_corr = 360.0  # 6 分钟
    nt_corr = 100    # 总模拟 10 小时
    
    U_corrosion = solve_corrosion_propagation(
        S_c_initial, nx, nt_corr, dx, dt_corr,
        v_corr, k_corr, theta_pore, method='godunov'
    )
    
    integrity_loss = structural_integrity_loss(
        np.mean(U_corrosion[-1]), np.mean(S_c_initial)
    )
    
    print(f"  腐蚀前沿速度 = {v_corr:.4e} m/s")
    print(f"  模拟时间步数 = {nt_corr}")
    print(f"  初始比表面积 = {np.mean(S_c_initial):.2f} m^2/g")
    print(f"  最终比表面积 = {np.mean(U_corrosion[-1]):.2f} m^2/g")
    print(f"  结构完整性损失 = {integrity_loss*100:.2f}%")
    print()
    
    # =====================================================================
    # Step 6: ECSA 损失评估与稳定性分析
    # =====================================================================
    print("[Step 6] ECSA 损失评估与系统稳定性分析 ...")
    
    # 只统计半径大于 1 nm 的颗粒（小于此值的视为已溶解脱离）
    active_threshold = 1.0e-9
    radii_initial_active = radii_initial[radii_initial > active_threshold]
    radii_final_active = radii_history[-1][radii_history[-1] > active_threshold]
    
    ecsa_0 = ecsa_from_size_distribution(radii_initial_active)
    ecsa_24h = ecsa_from_size_distribution(radii_final_active)
    
    # 综合 ECSA 损失模型
    ecsa_model_24h = total_ecsa_loss_model(24.0, ecsa_0)
    
    # 电压损失
    if ecsa_24h > 0 and ecsa_0 > 0:
        ecsa_ratio = ecsa_24h / ecsa_0
    else:
        ecsa_ratio = 0.01
    
    ecsa_ratio = min(ecsa_ratio, 1.0)  # 上限 100%
    dV = voltage_loss_from_ecsa(ecsa_ratio)
    
    # 稳定性分析
    J_stab = build_stability_jacobian(
        3, [1e-5, 2e-5, 5e-6],
        np.array([[-1e-5, 2e-6, 0],
                  [3e-6, -2e-5, 1e-6],
                  [0, 4e-6, -5e-6]])
    )
    lambda_max, stability = stability_analysis_max_eigenvalue(J_stab)
    
    # 矩分析
    m1_init = moment_size_distribution(radii_initial, k=1)
    m2_init = moment_size_distribution(radii_initial, k=2)
    m1_final = moment_size_distribution(radii_history[-1], k=1)
    m2_final = moment_size_distribution(radii_history[-1], k=2)
    
    print(f"  初始 ECSA = {ecsa_0:.2f} m^2/g_Pt")
    print(f"  24h ECSA  = {ecsa_24h:.2f} m^2/g_Pt")
    print(f"  综合模型 ECSA = {ecsa_model_24h:.2f} m^2/g_Pt")
    print(f"  ECSA 保留率 = {ecsa_ratio*100:.2f}%")
    print(f"  电压损失 = {dV*1000:.2f} mV")
    print(f"  主导特征值 = {lambda_max:.4e}")
    print(f"  系统稳定性 = {stability}")
    print(f"  粒径一阶矩: 初始={m1_init*1e9:.3f} nm, 最终={m1_final*1e9:.3f} nm")
    print(f"  粒径二阶矩: 初始={m2_init*1e18:.3f} nm^2, 最终={m2_final*1e18:.3f} nm^2")
    print()
    
    # =====================================================================
    # Step 7: 催化剂负载优化
    # =====================================================================
    print("[Step 7] 催化剂负载优化 ...")
    
    L_opt, J_opt, info = optimize_catalyst_loading(
        w_cost=0.25, w_power=0.60, w_penalty=0.15,
        L_min=0.02, L_max=0.8
    )
    
    sens = sensitivity_analysis(L_opt)
    
    print(f"  最优 Pt 负载 = {L_opt:.4f} mg/cm^2")
    print(f"  最优目标函数值 = {J_opt:.6f}")
    print(f"  迭代次数 = {info['iterations']}")
    print(f"  优化后功率 = {info['power_at_opt']*100:.2f}%")
    print(f"  优化后成本 = {info['cost_at_opt']:.2f} USD")
    print(f"  敏感性 dJ/dL = {sens:.6f}")
    print()
    
    # =====================================================================
    # Step 8: 稀疏矩阵组装与验证
    # =====================================================================
    print("[Step 8] 离散化稀疏矩阵组装与对角占优性检验 ...")
    
    assembler = SparseAssembler(n_interior=N_grid, n_boundary=2)
    A_full, A_band = assembler.assemble_diffusion_reaction_matrix(
        D_eff, k_rxn, L_ccl / (N_grid - 1), bc_type='dirichlet_neumann'
    )
    
    is_dd, min_ratio = assembler.check_diagonal_dominance(A_full)
    meta = harwell_boeing_metadata(N_grid, N_grid, np.count_nonzero(A_full))
    
    print(f"  矩阵维度 = {A_full.shape}")
    print(f"  非零元素 = {meta['nnzero']}")
    print(f"  严格对角占优 = {is_dd}")
    print(f"  最小对角占优比 = {min_ratio:.4f}")
    
    # 耦合系统组装
    A_bb, ml, mu = assembler.assemble_coupled_system(
        D_eff, k_rxn, L_ccl / (N_grid - 1),
        orr_params['R_ct'], j0, orr_params['alpha_a'],
        orr_params['alpha_c'], orr_params['n'], T_op
    )
    print(f"  边界带状矩阵大小 = {len(A_bb)}")
    print()
    
    # =====================================================================
    # Step 9: 形貌退化分析
    # =====================================================================
    print("[Step 9] 催化剂层形貌退化分析 ...")
    
    # 测试分形维数计算
    theta_circ = np.linspace(0, 2 * np.pi, 100)
    x_circ = np.cos(theta_circ)
    y_circ = np.sin(theta_circ)
    D_f_circle = box_counting_dimension(x_circ, y_circ)
    
    # 形貌演化前后的有效比表面积
    A0 = ecsa_0
    A_eff_init = effective_surface_area_fractal(A0, 1e-6, 1e-9, 1.8)
    A_eff_deg = effective_surface_area_fractal(A0, 1e-6, 1e-9, 1.5)
    
    # 孔隙连通性
    conn_map, xr, yr = pore_network_connectivity_map(n_grid=32, max_iter=30)
    mean_conn = np.mean(conn_map)
    
    # 表面状态枚举 (4个吸附位点)
    surface_states = enumerate_catalyst_surface_states(4, max_states=16)
    
    # 形貌退化指数
    mdi = morphology_degradation_index(1.8, 1.5, 1.0 - mean_conn)
    
    print(f"  圆边界分形维数 = {D_f_circle:.4f} (理论 1.0)")
    print(f"  初始有效 ECSA = {A_eff_init:.2f} m^2/g_Pt")
    print(f"  退化后有效 ECSA = {A_eff_deg:.2f} m^2/g_Pt")
    print(f"  孔隙连通性均值 = {mean_conn:.4f}")
    print(f"  表面状态枚举数 = {len(surface_states)}")
    print(f"  形貌退化指数 MDI = {mdi:.4f}")
    print()
    
    # =====================================================================
    # 总结
    # =====================================================================
    t_elapsed = time.time() - t_start
    
    print("=" * 70)
    print("模拟结果总结")
    print("=" * 70)
    print(f"操作条件: T={T_op:.1f} K, E={E_cell:.3f} V, L_Pt={L_pt:.3f} mg/cm^2")
    print(f"电化学:   j0={j0:.3e} A/m^2, eta={eta_overall:.4f} V, j={j_cell:.3e} A/m^2")
    print(f"传质:     O2(膜侧)={C_profile[-1]:.4f} mol/m^3")
    print(f"熟化:     r_mean(0h)={r0_mean*1e9:.2f} nm -> r_mean(500h)={np.mean(radii_history[-1])*1e9:.2f} nm")
    print(f"碳腐蚀:   完整性损失={integrity_loss*100:.2f}%")
    print(f"ECSA:     {ecsa_0:.2f} -> {ecsa_24h:.2f} m^2/g_Pt (保留 {ecsa_ratio*100:.1f}%)")
    print(f"电压损失: {dV*1000:.2f} mV")
    print(f"稳定性:   lambda_max={lambda_max:.3e} ({stability})")
    print(f"优化:     L_Pt*={L_opt:.3f} mg/cm^2")
    print(f"形貌:     MDI={mdi:.3f}")
    print(f"计算时间: {t_elapsed:.3f} s")
    print("=" * 70)
    print("模拟正常结束。")
    print("=" * 70)
    
    return {
        'operating_condition': cond,
        'current_density': j_cell,
        'overpotential': eta_overall,
        'oxygen_profile': C_profile,
        'radii_history': radii_history,
        'corrosion_history': U_corrosion,
        'ecsa_initial': ecsa_0,
        'ecsa_final': ecsa_24h,
        'voltage_loss_mV': dV * 1000,
        'stability_eigenvalue': lambda_max,
        'stability_status': stability,
        'optimal_loading': L_opt,
        'morphology_degradation_index': mdi,
        'compute_time_s': t_elapsed
    }


if __name__ == "__main__":
    try:
        results = run_simulation()
        sys.exit(0)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# ================================================================
# 测试用例（60个，assert模式，涉及随机值均使用固定种子）
# ================================================================

# 补充导入 main.py 中未导入但测试用例需要的函数
from butler_volmer import butler_volmer_current
from ripening_model import kelvin_solubility, ripening_rate, gauss_legendre_integral_exactness
from diffusion_solver import r83_cr_fa, r83_cr_sl, thomas_algorithm
from carbon_corrosion import carbon_mass_loss_rate, numerical_flux_godunov
from ecsa_calculator import power_method_eigenvalue
from catalyst_optimizer import golden_section_search
from morphology_evolution import ubvec_next_gray, mandelbrot_like_escape_time
from ccl_grid import hypercube_grid

# ---- TC01: butler_volmer_current 返回标量且有限 ----
j_bv = butler_volmer_current(0.05, 1e-4, 0.5, 0.5, 4, 353.15)
assert isinstance(j_bv, float), '[TC01] butler_volmer_current 返回值非标量 FAILED'
assert np.isfinite(j_bv), '[TC01] butler_volmer_current 返回非有限值 FAILED'
assert j_bv > 0, '[TC01] butler_volmer_current 正过电位应返回正电流 FAILED'

# ---- TC02: butler_volmer_current 零过电位近似零电流 ----
j_zero = butler_volmer_current(0.0, 1e-4, 0.5, 0.5, 4, 353.15)
assert abs(j_zero) < 1e-6, '[TC02] butler_volmer_current 零过电位应接近零电流 FAILED'

# ---- TC03: exchange_current_density 返回正值 ----
j0_val = exchange_current_density(353.15, 1.2)
assert j0_val > 0, '[TC03] exchange_current_density 应返回正值 FAILED'
assert np.isfinite(j0_val), '[TC03] exchange_current_density 返回非有限值 FAILED'

# ---- TC04: exchange_current_density 温度升高交换电流增大 ----
j0_lo = exchange_current_density(333.15, 1.2)
j0_hi = exchange_current_density(353.15, 1.2)
assert j0_hi > j0_lo, '[TC04] 温度升高交换电流密度应增大 FAILED'

# ---- TC05: solve_overpotential_muller 返回有限值 ----
p = orr_kinetic_parameters()
eta_m = solve_overpotential_muller(0.7, p['E_eq'], p['R_ct'], p['j0_ref'],
                                    p['alpha_a'], p['alpha_c'], p['n'], 353.15)
assert np.isfinite(eta_m), '[TC05] Muller 法过电位应有限 FAILED'
assert abs(eta_m) < 2.0, '[TC05] Muller 法过电位应在 [-2, 2] 范围内 FAILED'

# ---- TC06: solve_overpotential_wdk 返回有限值 ----
eta_w = solve_overpotential_wdk(0.7, p['E_eq'], p['R_ct'], p['j0_ref'],
                                 p['alpha_a'], p['alpha_c'], p['n'], 353.15)
assert np.isfinite(eta_w), '[TC06] WDK 法过电位应有限 FAILED'
assert abs(eta_w) < 2.0, '[TC06] WDK 法过电位应在 [-2, 2] 范围内 FAILED'

# ---- TC07: orr_kinetic_parameters 返回正确结构的字典 ----
p2 = orr_kinetic_parameters(353.15)
assert 'alpha_a' in p2 and 'alpha_c' in p2 and 'n' in p2, '[TC07] orr_kinetic_parameters 缺少必要键 FAILED'
assert p2['n'] == 4, '[TC07] 电子转移数应为 4 FAILED'
assert 0 < p2['alpha_a'] < 1 and 0 < p2['alpha_c'] < 1, '[TC07] 传递系数应在 (0,1) 内 FAILED'

# ---- TC08: generate_ccl_parameter_grid 返回正确形状 ----
pg = generate_ccl_parameter_grid()
assert pg['num_points'] == int(np.prod(pg['ns'])), '[TC08] 网格点数与 ns 乘积不一致 FAILED'
assert pg['grid'].shape[0] == len(pg['names']), '[TC08] 网格维度与名称列表长度不一致 FAILED'

# ---- TC09: sample_operating_condition 返回正确字段 ----
cond = sample_operating_condition(pg, 0)
assert 'temperature_K' in cond, '[TC09] 操作条件缺少 temperature_K FAILED'
assert 333.15 <= cond['temperature_K'] <= 353.15, '[TC09] 温度超出范围 FAILED'

# ---- TC10: effective_diffusivity 返回正值且在合理范围 ----
D_eff = effective_diffusivity(2.1e-5, 0.4)
assert D_eff > 0, '[TC10] 有效扩散系数应大于零 FAILED'
assert D_eff < 2.1e-5, '[TC10] 有效扩散系数应小于体相扩散系数 FAILED'

# ---- TC11: effective_diffusivity 边界 epsilon=0 返回极小正值 ----
D_min = effective_diffusivity(2.1e-5, 0.0)
assert D_min >= 1e-15, '[TC11] epsilon=0 时有效扩散系数应不小于保护值 FAILED'

# ---- TC12: solve_diffusion_tridiagonal 返回正确的输出形状和技术 ----
x_grid, C_tri = solve_diffusion_tridiagonal(1e-9, 100.0, 10e-6, 1.2, N=51)
assert len(x_grid) == 51 and len(C_tri) == 51, '[TC12] 扩散解输出长度应为 51 FAILED'
assert abs(C_tri[0] - 1.2) < 1e-10, '[TC12] 左边界 Dirichlet C=C_0 未满足 FAILED'
assert C_tri[-1] >= 0, '[TC12] 膜侧浓度不应为负 FAILED'

# ---- TC13: solve_diffusion_banded 与三对角法一致 ----
_, C_band = solve_diffusion_banded(1e-9, 100.0, 10e-6, 1.2, N=51)
diff_max = np.max(np.abs(C_tri - C_band))
assert diff_max < 1e-6, '[TC13] 三对角法与带状 LU 分解结果不一致 FAILED'

# ---- TC14: thomas_algorithm 求解已知三对角系统 ----
# 系统: 2x0 - x1 = 1, -xi-1 + 2xi - xi+1 = 0, -xn-2 + 2xn-1 = 0
n_test = 10
lower_t = -np.ones(n_test - 1)
diag_t = 2.0 * np.ones(n_test)
upper_t = -np.ones(n_test - 1)
rhs_t = np.zeros(n_test)
rhs_t[0] = 1.0
x_thomas = thomas_algorithm(lower_t, diag_t, upper_t, rhs_t)
assert np.all(np.isfinite(x_thomas)), '[TC14] Thomas 算法应返回有限值 FAILED'
assert x_thomas[0] > 0, '[TC14] Thomas 算法解不合理 FAILED'

# ---- TC15: r83_cr_fa + r83_cr_sl 求解简单三对角系统 ----
n_cr = 5
a_cr = np.zeros((3, n_cr))
a_cr[0, :] = -1.0    # 上对角线
a_cr[1, :] = 2.0     # 对角线
a_cr[2, :] = -1.0    # 下对角线
b_cr = np.ones(n_cr)
a_cr_factored = r83_cr_fa(n_cr, a_cr)
x_cr = r83_cr_sl(n_cr, a_cr_factored, b_cr)
assert len(x_cr) == n_cr, '[TC15] 循环约化解长度应为 n FAILED'
assert np.all(np.isfinite(x_cr)), '[TC15] 循环约化解应有限 FAILED'

# ---- TC16: kelvin_solubility 随半径增大而减小（单调性） ----
C_r1 = kelvin_solubility(2e-9, 2.5, 9.09e-6, 353.15, 1e-6)
C_r2 = kelvin_solubility(5e-9, 2.5, 9.09e-6, 353.15, 1e-6)
assert C_r1 > C_r2, '[TC16] 曲率半径越小溶解度应越大 FAILED'

# ---- TC17: critical_radius 返回正值 ----
rc = critical_radius(2.5, 9.09e-6, 353.15, 2e-6, 1e-6)
assert rc > 0, '[TC17] 临界半径应大于零 FAILED'
assert np.isfinite(rc), '[TC17] 临界半径应有限 FAILED'

# ---- TC18: ripening_rate 正负与颗粒尺寸关系 ----
# 小于临界半径的颗粒溶解 (rate<0)，大于的熟化长大 (rate>0)
rc_val = critical_radius(2.5, 9.09e-6, 353.15, 2e-6, 1e-6)
rate_small = ripening_rate(rc_val * 0.5, 1e-12, 9.09e-6, 1e-6, 2e-6, 2.5, 353.15)
rate_large = ripening_rate(rc_val * 2.0, 1e-12, 9.09e-6, 1e-6, 2e-6, 2.5, 353.15)
assert rate_small <= 0, '[TC18] 小于临界半径的颗粒应溶解 (rate<=0) FAILED'
assert rate_large >= 0, '[TC18] 大于临界半径的颗粒应长大 (rate>=0) FAILED'

# ---- TC19: evolve_size_distribution 返回正确形状和无负值 ----
import numpy as np
np.random.seed(42)
radii_init = np.random.lognormal(np.log(4e-9), 0.15, 10)
radii_init = np.clip(radii_init, 1e-9, 15e-9)
hist = evolve_size_distribution(radii_init, 1e-12, 9.09e-6, 1e-6, 2e-6, 2.5, 353.15, 3600, 10)
assert hist.shape[0] == 11 and hist.shape[1] == 10, '[TC19] 演化历史形状错误 FAILED'
assert np.all(hist >= 0.5e-9), '[TC19] 颗粒半径不应低于物理下限 0.5 nm FAILED'

# ---- TC20: lsw_analytical_r3 返回正值且>初始半径 ----
r_lsw = lsw_analytical_r3(500 * 3600, 4e-9, 2.5, 1e-12, 9.09e-6, 1e-6, 353.15)
assert r_lsw > 4e-9, '[TC20] LSW 熟化后半径应大于初始半径 FAILED'
assert np.isfinite(r_lsw), '[TC20] LSW 预测半径应有限 FAILED'

# ---- TC21: disk_distance_stats_monte_carlo 可复现性 (固定种子) ----
np.random.seed(42)
mu1, var1 = disk_distance_stats_monte_carlo(radii_init, radii_init)
np.random.seed(42)
mu2, var2 = disk_distance_stats_monte_carlo(radii_init, radii_init)
assert abs(mu1 - mu2) < 1e-15, '[TC21] 固定随机种子应产生相同结果 FAILED'
assert mu1 >= 0, '[TC21] 平均距离不应为负 FAILED'

# ---- TC22: moment_size_distribution 一阶矩等于均值 ----
m1 = moment_size_distribution(radii_init, k=1)
assert abs(m1 - np.mean(radii_init)) < 1e-15, '[TC22] 一阶矩应等于算术均值 FAILED'

# ---- TC23: moment_size_distribution k=0 应返回 1 ----
m0 = moment_size_distribution(radii_init, k=0)
assert abs(m0 - 1.0) < 1e-15, '[TC23] 零阶矩应等于 1 FAILED'

# ---- TC24: gauss_legendre_integral_exactness 线性函数精确积分 ----
nodes = np.array([-0.577350269189626, 0.577350269189626])
weights = np.array([1.0, 1.0])
integral = gauss_legendre_integral_exactness(lambda t: 2.0 * t + 1.0, 2, weights, nodes, a=0, b=1)
assert abs(integral - 1.0) < 1e-12, '[TC24] 两点高斯积分应对线性函数精确 FAILED'

# ---- TC25: corrosion_current_density 非负 ----
j_corr = corrosion_current_density(1.0, T=353.15)
assert j_corr >= 0, '[TC25] 腐蚀电流密度不应为负 FAILED'
assert np.isfinite(j_corr), '[TC25] 腐蚀电流密度应有限 FAILED'

# ---- TC26: corrosion_current_density 低电位下为零 ----
j_corr_low = corrosion_current_density(0.1, E_corr_0=0.207)
assert j_corr_low == 0.0, '[TC26] 低于平衡电位时腐蚀电流应为零 FAILED'

# ---- TC27: corrosion_front_velocity 非负 ----
v_f = corrosion_front_velocity(0.8, 353.15)
assert v_f >= 0, '[TC27] 腐蚀前沿速度不应为负 FAILED'

# ---- TC28: solve_corrosion_propagation 返回正确形状且值非负 ----
nx_c = 21
dx_c = 10e-6 / (nx_c - 1)
u0_c = np.ones(nx_c) * 200.0
U_c = solve_corrosion_propagation(u0_c, nx_c, 50, dx_c, 360.0, 1e-12, 1e-5, 0.4, method='godunov')
assert U_c.shape == (51, nx_c), '[TC28] 腐蚀传播结果形状错误 FAILED'
assert np.all(U_c >= 0), '[TC28] 碳比表面积不应为负 FAILED'

# ---- TC29: solve_corrosion_propagation Lax-Wendroff 格式可运行 ----
U_lw = solve_corrosion_propagation(u0_c, nx_c, 50, dx_c, 360.0, 1e-12, 1e-5, 0.4, method='lax_wendroff')
assert U_lw.shape == (51, nx_c), '[TC29] Lax-Wendroff 格式应返回正确形状 FAILED'
assert np.all(np.isfinite(U_lw)), '[TC29] Lax-Wendroff 格式应产生有限值 FAILED'

# ---- TC30: structural_integrity_loss 在 [0, 1] 范围内 ----
loss = structural_integrity_loss(150.0, 200.0)
assert 0.0 <= loss <= 1.0, '[TC30] 结构完整性损失应在 [0, 1] 内 FAILED'
assert abs(loss - 0.25) < 1e-10, '[TC30] 150/200 损失应为 0.25 FAILED'

# ---- TC31: carbon_mass_loss_rate 返回负值（质量损失） ----
rate_mass = carbon_mass_loss_rate(10.0, 1.0)
assert rate_mass < 0, '[TC31] 碳质量损失速率应为负 FAILED'

# ---- TC32: ecsa_from_size_distribution 返回正值 ----
radii_ecsa = np.array([2e-9, 3e-9, 4e-9, 5e-9])
ecsa_val = ecsa_from_size_distribution(radii_ecsa)
assert ecsa_val > 0, '[TC32] ECSA 应为正值 FAILED'

# ---- TC33: ECSA 随粒径增大而减小 ----
radii_small = np.array([2e-9, 2.5e-9, 3e-9])
radii_large = np.array([4e-9, 5e-9, 6e-9])
ecsa_s = ecsa_from_size_distribution(radii_small)
ecsa_l = ecsa_from_size_distribution(radii_large)
assert ecsa_s > ecsa_l, '[TC33] 小粒径 ECSA 应大于大粒径 ECSA FAILED'

# ---- TC34: ecsa_loss_kinetics 单调衰减 ----
ecsa0 = 50.0
ecsa_t1 = ecsa_loss_kinetics(ecsa0, 100)
ecsa_t2 = ecsa_loss_kinetics(ecsa0, 200)
assert 0 <= ecsa_t2 < ecsa_t1 <= ecsa0, '[TC34] ECSA 应随时间单调衰减 FAILED'

# ---- TC35: voltage_loss_from_ecsa 单调性 ----
dV_50 = voltage_loss_from_ecsa(0.5)
dV_80 = voltage_loss_from_ecsa(0.8)
assert dV_50 > dV_80, '[TC35] ECSA 保留越少电压损失应越大 FAILED'

# ---- TC36: total_ecsa_loss_model 衰减行为 ----
ecsa_tot = total_ecsa_loss_model(100, 50.0)
assert 0 <= ecsa_tot <= 50.0, '[TC36] 综合 ECSA 应在 [0, 初始值] 内 FAILED'

# ---- TC37: build_stability_jacobian + stability_analysis_max_eigenvalue ----
J_s = build_stability_jacobian(3, [1e-4, 2e-4, 5e-5],
                                np.array([[-1e-4, 2e-5, 0],
                                          [3e-5, -2e-4, 1e-5],
                                          [0, 4e-5, -5e-5]]))
lam_s, stab_s = stability_analysis_max_eigenvalue(J_s)
assert stab_s in ('stable', 'unstable', 'critical'), '[TC37] 稳定性判断必须是 stable/unstable/critical 之一 FAILED'
assert np.isfinite(lam_s), '[TC37] 主导特征值应有限 FAILED'

# ---- TC38: power_method_eigenvalue 对角矩阵精确特征值 ----
A_diag = np.diag([3.0, 1.0, 2.0])
lam_pm, _, _ = power_method_eigenvalue(A_diag, it_max=100, tol=1e-12)
assert abs(lam_pm - 3.0) < 1e-6, '[TC38] 幂法对角矩阵最大特征值应为 3.0 FAILED'

# ---- TC39: golden_section_search 找到已知最小值 ----
def f_test(x):
    return (x - 0.3) ** 2
a_gs, b_gs, it_gs, x_opt_gs, f_opt_gs = golden_section_search(f_test, 0.0, 1.0, n_max=100, x_tol=1e-10)
assert abs(x_opt_gs - 0.3) < 1e-6, '[TC39] 黄金分割应找到 x*=0.3 FAILED'
assert abs(f_opt_gs) < 1e-10, '[TC39] 最小值应接近零 FAILED'
assert it_gs > 0, '[TC39] 黄金分割应至少迭代一次 FAILED'

# ---- TC40: power_performance 单调递增且有界 ----
P_01 = power_performance(0.01)
P_02 = power_performance(0.02)
assert 0 <= P_01 < P_02 <= 1.0, '[TC40] 功率应随负载单调递增且在 [0, 1] 内 FAILED'

# ---- TC41: catalyst_cost 线性关系 ----
cost1 = catalyst_cost(0.1, area_active=250.0, price_pt=50.0)
cost2 = catalyst_cost(0.2, area_active=250.0, price_pt=50.0)
assert abs(cost2 / cost1 - 2.0) < 1e-10, '[TC41] 催化剂成本应与负载成正比 FAILED'

# ---- TC42: optimize_catalyst_loading 返回合理区间内结果 ----
L_opt, J_opt, info = optimize_catalyst_loading(L_min=0.02, L_max=0.5)
assert 0.02 <= L_opt <= 0.5, '[TC42] 最优负载应在搜索区间内 FAILED'
assert info['iterations'] > 0, '[TC42] 优化应至少执行一次迭代 FAILED'

# ---- TC43: sensitivity_analysis 返回有限值 ----
sens = sensitivity_analysis(L_opt)
assert np.isfinite(sens), '[TC43] 敏感性分析应返回有限值 FAILED'

# ---- TC44: SparseAssembler 创建矩阵正确形状 ----
assembler = SparseAssembler(n_interior=10, n_boundary=2)
A_full, A_band = assembler.assemble_diffusion_reaction_matrix(1e-9, 100.0, 1e-7)
assert A_full.shape == (10, 10), '[TC44] 组装稠密矩阵形状应为 (10,10) FAILED'
assert A_band.shape[1] == 10, '[TC44] 带状矩阵列数应为 10 FAILED'

# ---- TC45: SparseAssembler 对角占优检查 ----
is_dd, ratio = assembler.check_diagonal_dominance(A_full)
assert isinstance(is_dd, bool), '[TC45] 对角占优判断应为布尔值 FAILED'
assert ratio > 0, '[TC45] 对角占优比应为正 FAILED'

# ---- TC46: harwell_boeing_metadata 返回正确结构 ----
meta = harwell_boeing_metadata(10, 10, 28)
assert meta['nrow'] == 10 and meta['ncol'] == 10, '[TC46] Harwell-Boeing 元数据维度不正确 FAILED'

# ---- TC47: box_counting_dimension 圆形约为 1.0 ----
theta_circ = np.linspace(0, 2 * np.pi, 200)
x_circ = np.cos(theta_circ)
y_circ = np.sin(theta_circ)
D_f_circ = box_counting_dimension(x_circ, y_circ)
assert 0.5 <= D_f_circ <= 1.5, '[TC47] 圆的分形维数应在 0.5-1.5 范围内 FAILED'

# ---- TC48: enumerate_catalyst_surface_states 输出形状正确 ----
states = enumerate_catalyst_surface_states(4, max_states=16)
assert states.shape == (16, 4), '[TC48] 4位点16状态枚举形状应为 (16,4) FAILED'
# 相邻状态汉明距离为1 (格雷码性质)
hamming = np.sum(np.abs(states[0] - states[1]))
assert hamming == 1, '[TC48] 格雷码相邻状态汉明距离应为 1 FAILED'

# ---- TC49: ubvec_next_gray 遍历所有状态 ----
t_vec = np.zeros(4, dtype=int)
all_vecs = [t_vec.copy()]
for _ in range(15):
    t_vec = ubvec_next_gray(t_vec)
    all_vecs.append(t_vec.copy())
# 检查是否遍历了16个不同状态
unique_vecs = set(tuple(v) for v in all_vecs)
assert len(unique_vecs) == 16, '[TC49] ubvec_next_gray 应遍历 16 个不同状态 FAILED'

# ---- TC50: morphology_degradation_index 在 [0, 1] 内 ----
mdi = morphology_degradation_index(1.8, 1.5, 0.2)
assert 0.0 <= mdi <= 1.0, '[TC50] 形貌退化指数应在 [0, 1] 内 FAILED'

# ---- TC51: pore_network_connectivity_map 返回有效连通性值 ----
conn_map, _, _ = pore_network_connectivity_map(n_grid=16, max_iter=30)
assert conn_map.shape == (16, 16), '[TC51] 连通性图形状应为 (16,16) FAILED'
assert np.all((conn_map >= 0) & (conn_map <= 1)), '[TC51] 连通性应在 [0, 1] 内 FAILED'

# ---- TC52: effective_surface_area_fractal 返回正值 ----
A_eff_f = effective_surface_area_fractal(50.0, 1e-6, 1e-9, 1.8)
assert A_eff_f > 0, '[TC52] 分形有效比表面积应为正 FAILED'
assert np.isfinite(A_eff_f), '[TC52] 分形有效比表面积应有限 FAILED'

# ---- TC53: mandelbrot_like_escape_time 在 Mandelbrot 集内返回 max_iter ----
escape_in = mandelbrot_like_escape_time(0.0, 0.0, max_iter=30)
assert escape_in == 30, '[TC53] z=0 应在 Mandelbrot 集内 (未逃逸) FAILED'
escape_out = mandelbrot_like_escape_time(2.0, 0.0, max_iter=30)
assert escape_out < 30, '[TC53] c=2 应快速逃逸 FAILED'

# ---- TC54: carbon_mass_loss_rate 零电流返回零 ----
rate_zero = carbon_mass_loss_rate(0.0, 1.0)
assert rate_zero == 0.0, '[TC54] 零腐蚀电流应返回零质量损失 FAILED'

# ---- TC55: pt_dissolution_parameters 返回正确字段字典 ----
pt_p = pt_dissolution_parameters()
assert 'gamma' in pt_p and 'D_Pt2' in pt_p and 'V_m' in pt_p, '[TC55] pt_dissolution_parameters 缺少必要字段 FAILED'
assert pt_p['gamma'] > 0 and pt_p['D_Pt2'] > 0, '[TC55] Pt 物理参数必须为正 FAILED'

# ---- TC56: solve_corrosion_propagation MacCormack 格式可运行 ----
U_mc = solve_corrosion_propagation(u0_c, nx_c, 50, dx_c, 360.0, 1e-12, 1e-5, 0.4, method='maccormack')
assert U_mc.shape == (51, nx_c), '[TC56] MacCormack 格式应返回正确形状 FAILED'
assert np.all(np.isfinite(U_mc)), '[TC56] MacCormack 格式应产生有限值 FAILED'

# ---- TC57: hypercube_grid 返回正确维度的网格 ----
ng = hypercube_grid(m=2, n=4, ns=[2, 2], a=[0.0, 0.0], b=[1.0, 1.0], c=[1, 1])
assert ng.shape == (2, 4), '[TC57] hypercube_grid 2D×2×2 应返回 (2,4) FAILED'
assert np.all((ng >= 0) & (ng <= 1)), '[TC57] 网格点应在 [0,1] 范围内 FAILED'

# ---- TC58: structural_integrity_loss 无损失时返回零 ----
loss_none = structural_integrity_loss(200.0, 200.0)
assert abs(loss_none) < 1e-15, '[TC58] 无碳损失时完整性损失应为零 FAILED'

# ---- TC59: numerical_flux_godunov 正速度取左侧通量 ----
flux_pos = numerical_flux_godunov(3.0, 5.0, v=1.0)
assert abs(flux_pos - 3.0) < 1e-15, '[TC59] v>0 时 Godunov 通量应取左侧值 FAILED'

# ---- TC60: numerical_flux_godunov 负速度取右侧通量 ----
flux_neg = numerical_flux_godunov(3.0, 5.0, v=-1.0)
assert abs(flux_neg - 5.0) < 1e-15, '[TC60] v<0 时 Godunov 通量应取右侧值 FAILED'

print('\n全部 60 个测试通过!\n')
