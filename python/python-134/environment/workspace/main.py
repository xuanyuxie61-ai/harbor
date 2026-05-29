#!/usr/bin/env python3
"""
main.py
PEM Fuel Cell Proton Exchange Membrane Water Management
化学工程：燃料电池质子交换膜水管理 — 统一入口

零参数运行，执行完整的耦合多物理场数值模拟流程：
  1. 化学计量学平衡验证
  2. 三维四面体网格生成与细化
  3. 电化学反应动力学计算（Butler-Volmer）
  4. 二维质子电势场求解（泊松方程 + Hermite 插值）
  5. 膜内水传输瞬态求解（隐式向后 Euler + stiff ODE）
  6. GDL 多孔介质液态水传输求解
  7. 催化层有效扩散系数蒙特卡洛估计
  8. 最优水含量测点 CVT 布置
  9. 带状矩阵线性代数性能测试（R8PBL + LINPACK + Hankel）
 10. 合成极化曲线实验数据生成
 11. 收敛性与残差分析
 12. 水含量测量不确定性量化（Hankel 协方差）
"""

import numpy as np
import sys
import os

# 将当前目录加入路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stoichiometry_balancer import balance_orr_stoichiometry, verify_stoichiometry_solution
from electrochemistry_kinetics import butler_volmer_kinetics, compute_exchange_current_density
from proton_potential_solver import solve_proton_potential, interpolate_proton_potential_hermite
from membrane_water_transport import solve_membrane_water_transport, water_content_wave_exact
from porous_gdl_transport import solve_gdl_saturation, porous_medium_exact
from mesh_generator import generate_pemfc_mesh, refine_mesh, compute_mesh_quality
from monte_carlo_clustering import estimate_effective_diffusivity_monte_carlo, estimate_water_cluster_distribution
from optimal_sampling import optimize_sensor_placement
from banded_linear_algebra import solve_banded_linear_system, hankel_covariance_factor
from synthetic_experiments import generate_polarization_curve, generate_impedance_spectrum
from convergence_analysis import compute_residuals, compute_mass_balance_error


def setup_physical_parameters():
    """设置 PEMFC 物理化学参数"""
    params = {
        'T': 353.15,          # 运行温度 [K]
        'P': 1.5,             # 运行压力 [atm]
        'R': 8.314,           # 通用气体常数 [J/(mol·K)]
        'F': 96485.0,         # 法拉第常数 [C/mol]
        'E_0': 1.229,         # 标准可逆电位 [V]
        'alpha_a': 0.5,       # 阳极传递系数
        'alpha_c': 0.5,       # 阴极传递系数
        'j_0_ref': 1.0e-3,    # 参考交换电流密度 [A/m²]
        'lambda_eq': 14.0,    # 平衡水含量
        'D_lambda_max': 2.5e-10,  # 最大水扩散系数 [m²/s]
        'n_drag': 2.5,        # 电渗拖拽系数
        't_membrane': 50e-6,  # 膜厚度 [m]
        't_gdl': 200e-6,      # GDL厚度 [m]
        'sigma_m_ref': 10.0,  # 参考膜电导率 [S/m]
        'epsilon_gdl': 0.4,   # GDL孔隙率
        'D_gdl_ref': 1.0e-6,  # GDL参考扩散系数 [m²/s]
        'm_porous': 2.5,      # 多孔介质方程指数
        'Nx': 81,             # 空间网格数
        'Nt': 500,            # 时间步数
        't_final': 10.0,      # 最终时间 [s]
        'N_mc': 10000,        # 蒙特卡洛样本数
        'N_sensors': 16,      # 传感器数量
    }
    return params


def main():
    print("=" * 70)
    print("PEM Fuel Cell Proton Exchange Membrane Water Management")
    print("博士级多物理场耦合数值模拟系统")
    print("=" * 70)

    # ---------------------------------------------------------------
    # 1. 物理参数与化学计量学初始化
    # ---------------------------------------------------------------
    params = setup_physical_parameters()
    print("\n[1] 物理参数与化学计量学初始化")
    print("    - 运行温度: {:.1f} K".format(params['T']))
    print("    - 运行压力: {:.2f} atm".format(params['P']))

    stoich = balance_orr_stoichiometry()
    print("    - ORR化学计量平衡验证: O₂ + {}H⁺ + {}e⁻ → {}H₂O".format(
        stoich['h_plus'], stoich['electrons'], stoich['water']))
    res_stoich = verify_stoichiometry_solution(stoich)
    print("    - 化学计量残差: O={}, H={}, e={}".format(
        res_stoich['r_o'], res_stoich['r_h'], res_stoich['r_e']))

    # ---------------------------------------------------------------
    # 2. 三维四面体网格生成与细化
    # ---------------------------------------------------------------
    print("\n[2] 生成燃料电池三维四面体网格")
    nodes, elements = generate_pemfc_mesh()
    quality_init = compute_mesh_quality(nodes, elements)
    print("    - 初始节点数: {}".format(quality_init['n_nodes']))
    print("    - 初始单元数: {}".format(quality_init['n_elements']))
    print("    - 初始最小体积: {:.4e}".format(quality_init['min_volume']))

    nodes_refined, elements_refined = refine_mesh(nodes, elements)
    quality_refined = compute_mesh_quality(nodes_refined, elements_refined)
    print("    - 细化后节点数: {}".format(quality_refined['n_nodes']))
    print("    - 细化后单元数: {}".format(quality_refined['n_elements']))
    print("    - 细化后最小体积: {:.4e}".format(quality_refined['min_volume']))

    # ---------------------------------------------------------------
    # 3. 电化学反应动力学计算
    # ---------------------------------------------------------------
    print("\n[3] 电化学反应动力学计算 (Butler-Volmer)")
    eta = np.linspace(-0.3, 0.3, 100)
    j_bv = butler_volmer_kinetics(eta, params)
    j0 = compute_exchange_current_density(params)
    print("    - 交换电流密度 j_0: {:.4e} A/m²".format(j0))
    print("    - 过电位范围: [{:.3f}, {:.3f}] V".format(eta.min(), eta.max()))
    print("    - 最大电流密度: {:.4e} A/m²".format(np.max(np.abs(j_bv))))

    # ---------------------------------------------------------------
    # 4. 质子电势场求解 (二维泊松方程 + Hermite 插值)
    # ---------------------------------------------------------------
    print("\n[4] 质子电势场求解 (二维泊松方程)")
    phi_m, x_grid, y_grid = solve_proton_potential(params)
    print("    - 网格尺寸: {} × {}".format(len(x_grid), len(y_grid)))
    print("    - 电势范围: [{:.6f}, {:.6f}] V".format(phi_m.min(), phi_m.max()))
    print("    - 电势均值: {:.6f} V".format(phi_m.mean()))

    # Hermite 插值验证
    xq, yq = 0.5, 0.5
    phi_interp = interpolate_proton_potential_hermite(phi_m, x_grid, y_grid, xq, yq)
    phi_exact_q = 2.0 * (1.0 + yq) / ((3.0 + xq) ** 2 + (1.0 + yq) ** 2)
    print("    - Hermite插值验证 @({:.1f},{:.1f}): 数值={:.6f}, 解析={:.6f}, 误差={:.2e}".format(
        xq, yq, phi_interp, phi_exact_q, abs(phi_interp - phi_exact_q)))

    # ---------------------------------------------------------------
    # 5. 膜内水传输瞬态求解 (隐式向后 Euler + stiff ODE)
    # ---------------------------------------------------------------
    print("\n[5] 膜内水传输求解 (扩散 + 电渗拖拽 + stiff ODE)")
    # TODO: Hole_3 待修复 —— 正确调用膜水传输求解器并处理返回结果
    # 需要与 membrane_water_transport.py 中的 solve_membrane_water_transport 接口一致，
    # 并保证 lambda_profile 的维度和格式满足后续 hankel_covariance_factor 与
    # compute_mass_balance_error 的调用要求。
    pass

    # ---------------------------------------------------------------
    # 6. GDL 多孔介质液态水传输求解
    # ---------------------------------------------------------------
    print("\n[6] 气体扩散层多孔介质传质求解")
    s_gdl, x_gdl = solve_gdl_saturation(params)
    print("    - GDL饱和度范围: [{:.4f}, {:.4f}]".format(s_gdl.min(), s_gdl.max()))
    print("    - GDL平均饱和度: {:.4f}".format(s_gdl.mean()))

    # Barenblatt 验证
    z_baren = np.linspace(-0.1, 0.1, 21)
    s_baren = porous_medium_exact(z_baren, 0.5, params['m_porous'], params)[0]
    s_baren = np.clip(s_baren / max(np.max(s_baren), 1e-12), 0.0, 1.0)
    print("    - Barenblatt自相似解平均饱和度: {:.4f}".format(s_baren.mean()))

    # ---------------------------------------------------------------
    # 7. 催化层有效扩散系数蒙特卡洛估计
    # ---------------------------------------------------------------
    print("\n[7] 蒙特卡洛估计催化层有效扩散系数")
    D_eff = estimate_effective_diffusivity_monte_carlo(
        nodes_refined, elements_refined, params, n_samples_per_tet=30)
    print("    - 有效扩散系数 D_eff: {:.4e} m²/s".format(D_eff))

    # 水团簇分布
    thresholds, fractions = estimate_water_cluster_distribution(
        nodes_refined, elements_refined, params, n_samples=2000)
    print("    - 水团簇阈值扫描点数: {}".format(len(thresholds)))
    print("    - λ>14 团簇占比: {:.3f}".format(fractions[12] if len(fractions) > 12 else 0.0))

    # ---------------------------------------------------------------
    # 8. 最优水含量测点 CVT 布置
    # ---------------------------------------------------------------
    print("\n[8] 最优水含量测点布置 (Centroidal Voronoi Tessellation)")
    sensors = optimize_sensor_placement(params, n_iter=20)
    print("    - 最优测点数量: {}".format(sensors.shape[0]))
    print("    - 测点平面范围 x: [{:.4f}, {:.4f}] m".format(sensors[:, 0].min(), sensors[:, 0].max()))
    print("    - 测点平面范围 y: [{:.4f}, {:.4f}] m".format(sensors[:, 1].min(), sensors[:, 1].max()))

    # ---------------------------------------------------------------
    # 9. 带状矩阵线性代数性能测试
    # ---------------------------------------------------------------
    print("\n[9] 结构化线性代数性能测试 (R8PBL + LINPACK + Hankel)")
    resid, t_solve = solve_banded_linear_system(params)
    print("    - 求解残差 (||Ax-b||/||b||): {:.2e}".format(resid))
    print("    - 求解耗时: {:.4f} ms".format(t_solve * 1000))

    # ---------------------------------------------------------------
    # 10. 合成极化曲线实验数据
    # ---------------------------------------------------------------
    print("\n[10] 合成极化曲线与阻抗谱实验数据")
    V_cell, I_cell = generate_polarization_curve(params)
    print("    - 开路电压: {:.3f} V".format(V_cell[0]))
    P = V_cell * I_cell
    idx_maxP = np.argmax(P)
    print("    - 最大功率密度: {:.3f} W/cm² @ {:.3f} A/cm²".format(
        P[idx_maxP], I_cell[idx_maxP]))

    freq, Z_real, Z_imag = generate_impedance_spectrum(params)
    print("    - EIS 频率范围: [{:.1e}, {:.1e}] Hz".format(freq.min(), freq.max()))
    print("    - EIS 最小实部阻抗: {:.4f} Ohm·cm²".format(np.min(Z_real)))

    # ---------------------------------------------------------------
    # 11. 收敛性与残差分析
    # ---------------------------------------------------------------
    print("\n[11] 收敛性与质量平衡分析")
    residuals = compute_residuals(phi_m, lambda_profile, s_gdl, params)
    print("    - 质子电势方程 L2 残差: {:.2e}".format(residuals['proton']))
    print("    - 水传输方程 L2 残差: {:.2e}".format(residuals['water']))
    print("    - 多孔介质方程 L2 残差: {:.2e}".format(residuals['porous']))

    j_profile = np.linspace(5000.0, 10000.0, len(lambda_profile))
    mb_err, lambda_total = compute_mass_balance_error(lambda_profile, j_profile, params)
    print("    - 稳态水质量平衡误差: {:.2e} mol/(m²·s)".format(mb_err))
    print("    - 膜内总水含量积分: {:.4e} mol/m²".format(lambda_total))

    # ---------------------------------------------------------------
    # 12. 水含量测量不确定性量化 (Hankel 协方差)
    # ---------------------------------------------------------------
    print("\n[12] 水含量测量不确定性量化 (Hankel 协方差)")
    R_cov = hankel_covariance_factor(lambda_profile)
    cov = R_cov @ R_cov.T
    # 安全计算条件数
    diag_cov = np.diag(cov)
    if np.all(np.isfinite(diag_cov)) and np.all(diag_cov > 0):
        cond_num = np.max(diag_cov) / max(np.min(diag_cov[diag_cov > 0]), 1e-15)
    else:
        cond_num = np.inf
    print("    - 协方差矩阵维度: {}×{}".format(R_cov.shape[0], R_cov.shape[0]))
    print("    - 协方差矩阵条件数: {:.2e}".format(cond_num))
    print("    - 协方差矩阵迹: {:.4e}".format(np.trace(cov)))

    # ---------------------------------------------------------------
    # 总结
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("模拟完成。所有 12 个科学计算模块运行正常，无报错。")
    print("=" * 70)


if __name__ == '__main__':
    main()
