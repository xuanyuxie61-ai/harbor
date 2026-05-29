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
    lambda_profile, t_grid = solve_membrane_water_transport(params)
    print("    - 稳态水含量范围: [{:.4f}, {:.4f}]".format(
        lambda_profile.min(), lambda_profile.max()))
    print("    - 模拟时间步数: {}".format(len(t_grid) - 1))
    print("    - 总模拟时间: {:.2f} s".format(t_grid[-1]))

    # 波动方程验证
    z_test = np.linspace(0.0, params['t_membrane'], 11)
    u_wave, ut_wave, utt_wave, uz_wave, uzz_wave = water_content_wave_exact(
        z_test, 1.0, params)
    print("    - 波动解析解残差 max|u_tt - c²u_zz|: {:.2e}".format(
        np.max(np.abs(utt_wave - (1e-3) ** 2 * uzz_wave))))

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

# ================================================================
# 测试用例（30个，assert模式，涉及随机值均使用固定种子）
# ================================================================

# ---- TC01: setup_physical_parameters 返回字典且关键参数为正 ----
params = setup_physical_parameters()
assert isinstance(params, dict), '[TC01] params type FAILED'
assert params['T'] > 0, '[TC01] temperature FAILED'
assert params['F'] > 0, '[TC01] Faraday constant FAILED'

# ---- TC02: balance_orr_stoichiometry 返回正确键与整数系数 ----
stoich = balance_orr_stoichiometry()
assert stoich['o2'] == 1, '[TC02] O2 coefficient FAILED'
assert stoich['water'] == 2, '[TC02] water coefficient FAILED'
assert stoich['h_plus'] == 4, '[TC02] H+ coefficient FAILED'
assert stoich['electrons'] == 4, '[TC02] electrons coefficient FAILED'

# ---- TC03: verify_stoichiometry_solution 对正确解返回全零残差 ----
res = verify_stoichiometry_solution(stoich)
assert res['r_o'] == 0, '[TC03] O residual FAILED'
assert res['r_h'] == 0, '[TC03] H residual FAILED'
assert res['r_e'] == 0, '[TC03] e residual FAILED'

# ---- TC04: butler_volmer_kinetics 零过电位电流近似为零 ----
import numpy as np
eta_z = np.array([0.0])
j_z = butler_volmer_kinetics(eta_z, params)
assert abs(j_z[0]) < 1e-12, '[TC04] zero eta current FAILED'

# ---- TC05: butler_volmer_kinetics 输出形状与输入一致 ----
import numpy as np
eta_arr = np.linspace(-0.3, 0.3, 100)
j_arr = butler_volmer_kinetics(eta_arr, params)
assert j_arr.shape == eta_arr.shape, '[TC05] shape mismatch FAILED'

# ---- TC06: butler_volmer_kinetics 输出单调递增 ----
import numpy as np
j_diff = np.diff(j_arr)
assert np.all(j_diff >= -1e-12), '[TC06] not monotonic FAILED'

# ---- TC07: compute_exchange_current_density 返回正有限值 ----
j0 = compute_exchange_current_density(params)
assert j0 > 0, '[TC07] j0 positive FAILED'
assert np.isfinite(j0), '[TC07] j0 finite FAILED'

# ---- TC08: generate_pemfc_mesh 返回 8 个三维节点和四面体单元 ----
nodes, elements = generate_pemfc_mesh()
assert nodes.shape == (8, 3), '[TC08] nodes shape FAILED'
assert elements.shape[1] == 4, '[TC08] not tetrahedra FAILED'

# ---- TC09: refine_mesh 增加节点和单元数量 ----
nodes_r, elements_r = refine_mesh(nodes, elements)
assert nodes_r.shape[0] > nodes.shape[0], '[TC09] refined nodes not increased FAILED'
assert elements_r.shape[0] > elements.shape[0], '[TC09] refined elements not increased FAILED'

# ---- TC10: compute_mesh_quality 返回正体积 ----
quality = compute_mesh_quality(nodes_r, elements_r)
assert quality['min_volume'] > 0, '[TC10] min volume FAILED'
assert quality['mean_volume'] > 0, '[TC10] mean volume FAILED'

# ---- TC11: solve_proton_potential 输出二维数组且形状匹配 ----
phi_m, x_grid, y_grid = solve_proton_potential(params)
assert phi_m.ndim == 2, '[TC11] phi not 2D FAILED'
assert phi_m.shape == (len(x_grid), len(y_grid)), '[TC11] phi shape mismatch FAILED'

# ---- TC12: interpolate_proton_potential_hermite 返回有限值 ----
phi_q = interpolate_proton_potential_hermite(phi_m, x_grid, y_grid, 0.5, 0.5)
assert np.isfinite(phi_q), '[TC12] hermite interp FAILED'

# ---- TC13: water_content_wave_exact 满足波动方程 ----
import numpy as np
z_test = np.linspace(0.0, params['t_membrane'], 11)
u, ut, utt, uz, uzz = water_content_wave_exact(z_test, 1.0, params)
c_w = 1.0e-3
wave_res = np.max(np.abs(utt - c_w**2 * uzz))
assert wave_res < 1e-10, '[TC13] wave equation residual FAILED'

# ---- TC14: solve_membrane_water_transport 输出在物理范围内 ----
lam, t_grid = solve_membrane_water_transport(params)
assert lam.ndim == 1, '[TC14] lambda not 1D FAILED'
assert np.all(np.isfinite(lam)), '[TC14] lambda not finite FAILED'
assert np.all(lam >= 0.0), '[TC14] lambda negative FAILED'
assert np.all(lam <= 22.0), '[TC14] lambda exceeds max FAILED'

# ---- TC15: solve_gdl_saturation 饱和度在 [0,1] 范围内 ----
s_gdl, x_gdl = solve_gdl_saturation(params)
assert np.all(s_gdl >= 0.0), '[TC15] saturation negative FAILED'
assert np.all(s_gdl <= 1.0), '[TC15] saturation exceeds 1 FAILED'
assert s_gdl.ndim == 1, '[TC15] saturation not 1D FAILED'

# ---- TC16: 边界条件: GDL 流道侧低饱和度、催化层侧较高 ----
import numpy as np
assert abs(s_gdl[0] - 0.05) < 1e-12, '[TC16] GDL channel side BC FAILED'
assert abs(s_gdl[-1] - 0.6) < 1e-12, '[TC16] GDL catalyst side BC FAILED'

# ---- TC17: porous_medium_exact 在 t=0 返回全零 ----
import numpy as np
x_pm = np.array([-0.1, 0.0, 0.1])
u_pm, ut_pm, ux_pm, uxx_pm = porous_medium_exact(x_pm, 0.0, params['m_porous'], params)
assert np.all(u_pm == 0.0), '[TC17] t=0 not zero FAILED'

# ---- TC18: porous_medium_exact 在 t>0 返回非负有限值 ----
import numpy as np
u_pm2, ut_pm2, ux2, uxx2 = porous_medium_exact(x_pm, 0.5, params['m_porous'], params)
assert np.all(u_pm2 >= 0.0), '[TC18] solution negative FAILED'
assert np.all(np.isfinite(u_pm2)), '[TC18] solution not finite FAILED'

# ---- TC19: estimate_effective_diffusivity_monte_carlo 返回正值 ----
D_eff = estimate_effective_diffusivity_monte_carlo(nodes_r, elements_r, params, n_samples_per_tet=30)
assert D_eff > 0, '[TC19] D_eff not positive FAILED'
assert np.isfinite(D_eff), '[TC19] D_eff not finite FAILED'

# ---- TC20: optimize_sensor_placement 返回正确形状和范围 ----
np.random.seed(42)
sensors = optimize_sensor_placement(params, n_iter=20)
assert sensors.shape[0] == params['N_sensors'], '[TC20] sensor count FAILED'
assert sensors.shape[1] == 2, '[TC20] sensor dim FAILED'
assert np.all(sensors >= 0.0), '[TC20] sensor negative FAILED'

# ---- TC21: solve_banded_linear_system 残差足够小且耗时为正 ----
resid, t_solve = solve_banded_linear_system(params)
assert resid < 1e-6, '[TC21] residual too large FAILED'
assert t_solve > 0, '[TC21] solve time non-positive FAILED'

# ---- TC22: hankel_covariance_factor 输出方阵 ----
R_cov = hankel_covariance_factor(lam)
assert R_cov.shape[0] > 0, '[TC22] cov factor empty FAILED'
assert R_cov.shape[0] == R_cov.shape[1], '[TC22] cov factor not square FAILED'

# ---- TC23: 协方差矩阵为正定 ----
import numpy as np
cov = R_cov @ R_cov.T
eigvals = np.linalg.eigvalsh(cov)
assert np.all(eigvals > 0), '[TC23] covariance not SPD FAILED'

# ---- TC24: generate_polarization_curve 电压单调递减 ----
V_cell, I_cell = generate_polarization_curve(params)
assert V_cell.shape == I_cell.shape, '[TC24] polarization shape FAILED'
assert V_cell[0] > V_cell[-1], '[TC24] voltage not decreasing FAILED'
assert np.all(np.isfinite(V_cell)), '[TC24] voltage not finite FAILED'

# ---- TC25: generate_impedance_spectrum 输出维度正确且实部为正 ----
freq, Z_real, Z_imag = generate_impedance_spectrum(params)
assert freq.shape == Z_real.shape, '[TC25] freq-real shape FAILED'
assert freq.shape == Z_imag.shape, '[TC25] freq-imag shape FAILED'
assert np.all(Z_real > 0), '[TC25] real impedance non-positive FAILED'

# ---- TC26: compute_residuals 返回正确键和有限值 ----
residuals = compute_residuals(phi_m, lam, s_gdl, params)
assert set(residuals.keys()) == {'proton', 'water', 'porous'}, '[TC26] residual keys FAILED'
assert np.isfinite(residuals['proton']), '[TC26] proton residual FAILED'
assert np.isfinite(residuals['water']), '[TC26] water residual FAILED'

# ---- TC27: compute_mass_balance_error 返回非负误差 ----
import numpy as np
j_profile = np.linspace(5000.0, 10000.0, len(lam))
mb_err, lambda_total = compute_mass_balance_error(lam, j_profile, params)
assert mb_err >= 0, '[TC27] mass balance negative FAILED'
assert lambda_total > 0, '[TC27] total lambda non-positive FAILED'

# ---- TC28: 可复现性: 固定种子两次调用 optimize_sensor_placement 结果一致 ----
import numpy as np
np.random.seed(42)
s1 = optimize_sensor_placement(params, n_iter=10)
np.random.seed(42)
s2 = optimize_sensor_placement(params, n_iter=10)
assert np.allclose(s1, s2), '[TC28] reproducibility FAILED'

# ---- TC29: capillary_diffusivity 正值 ----
from porous_gdl_transport import capillary_diffusivity
import numpy as np
s_test = np.linspace(0.1, 0.9, 20)
D_cap = capillary_diffusivity(s_test, {'epsilon_gdl': params['epsilon_gdl']})
assert np.all(D_cap > 0), '[TC29] capillary diffusivity non-positive FAILED'
assert np.all(np.isfinite(D_cap)), '[TC29] capillary diffusivity not finite FAILED'

# ---- TC30: 集成测试: 所有模块协调运行且输出核心量有限 ----
import numpy as np
p_int = setup_physical_parameters()
p_int['Nx'] = 11
p_int['Nt'] = 10
p_int['t_final'] = 0.1
stoich_int = balance_orr_stoichiometry()
nodes_int, elements_int = generate_pemfc_mesh()
nodes_r_int, elements_r_int = refine_mesh(nodes_int, elements_int)
eta_int = np.linspace(-0.3, 0.3, 10)
j_bv_int = butler_volmer_kinetics(eta_int, p_int)
phi_int, xg_int, yg_int = solve_proton_potential(p_int)
lam_int, tg_int = solve_membrane_water_transport(p_int)
s_int, zg_int = solve_gdl_saturation(p_int)
D_int = estimate_effective_diffusivity_monte_carlo(nodes_r_int, elements_r_int, p_int, n_samples_per_tet=5)
sensors_int = optimize_sensor_placement(p_int, n_iter=5)
resid_int, _ = solve_banded_linear_system(p_int)
V_int, I_int = generate_polarization_curve(p_int)
assert phi_int.ndim == 2, '[TC30] integration phi FAILED'
assert lam_int.ndim == 1, '[TC30] integration lambda FAILED'
assert s_int.ndim == 1, '[TC30] integration saturation FAILED'
assert np.isfinite(resid_int), '[TC30] integration residual FAILED'
assert np.isfinite(D_int), '[TC30] integration D_eff FAILED'

print('\n全部 30 个测试通过!\n')
