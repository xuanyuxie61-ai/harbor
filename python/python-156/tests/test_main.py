"""
main.py
=======
湍流燃烧火焰面模型的统一入口程序。

本程序执行完整的博士级科学计算流程：
1. 初始化物理参数与混合分数空间离散；
2. 使用 FEM/FDM 求解稳态火焰面温度与组分分布；
3. 进行点火/熄火极限分析（多项式求根）；
4. 生成湍流随机脉动场并修正标量耗散率；
5. 计算火焰前锋几何与曲率效应；
6. 3D 燃烧室网格生成与带宽分析；
7. 守恒律校验与误差分析；
8. 综合结果输出。

科学背景：
----------
基于 Peters (1986) 的稳态层流火焰面模型（SLFM），求解非预混湍流
燃烧中混合分数空间内的温度与组分分布。火焰面模型假设化学反应发生
在极薄的层流火焰面内，湍流的作用通过标量耗散率 χ(Z) 体现。

核心控制方程：
    ρ(Z) χ(Z)/2 * d²T/dZ² + ω̇_T = 0
    ρ(Z) χ(Z)/2 * d²Y_k/dZ² + ω̇_k = 0

边界条件：
    Z=0 (氧化剂侧): T=T_ox, Y_F=0, Y_O=Y_{O,0}
    Z=1 (燃料侧):   T=T_fuel, Y_F=Y_{F,0}, Y_O=0
"""

import numpy as np
import sys

# =====================================================================
# 导入所有科研模块
# =====================================================================
from flamelet_core import (
    scalar_dissipation_rate,
    density_mixture,
    mixture_molecular_weight,
    reaction_rate_one_step,
    flamelet_boundary_conditions,
    thermal_diffusivity_ref,
    Z_STOICHIOMETRIC,
    T_OXIDIZER,
    T_FUEL,
    ADIA_TEMP_STOIC,
)

from fem_thermal_solver import solve_fem_thermal
from fem_quadratic_solver import solve_fem_quadratic_species
from fd_scalar_solver import solve_fd_scalar_dissipation
from mesh3d_generator import generate_mesh_3d
from turbulent_random_field import generate_turbulent_velocity_fluctuation, scalar_dissipation_rate_fluctuation
from ignition_polynomial import analyze_ignition_extinction
from flame_front_shape import flame_front_surface_area, chicken_egg_shape
from arrhenius_kinetics import arrhenius_rate_constant, adiabatic_flame_temperature, integrate_progress_variable
from flame_instability import integrate_flame_instability, darrieus_landau_growth_rate
from elliptic_special import curved_flame_speed, markstein_length, flame_curvature_elliptic
from spatial_partition import domain_decomposition_1d, voronoi_area_2d, compute_partition_quality
from matrix_bandwidth import analyze_flamelet_bandwidth
from stoichiometric_polynomial import stoichiometric_paths, reaction_mechanism_complexity
from conservation_validator import validate_simulation
from exact_benchmark import manufactured_solution_temperature, gaussian_flamelet_solution, compute_errors


def print_section(title):
    """打印格式化章节标题。"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    """主程序入口，零参数可运行。"""

    np.random.seed(42)

    print_section("湍流燃烧火焰面模型数值模拟")
    print("科学领域: 燃烧科学 - 湍流燃烧火焰面模型 (SLFM)")
    print("问题难度: 博士级前沿科学计算")

    # =====================================================================
    # 1. 初始化参数与网格
    # =====================================================================
    print_section("1. 初始化物理参数与混合分数空间离散")

    n_nodes = 65  # 奇数，适配二次FEM
    Z_nodes = np.linspace(0.0, 1.0, n_nodes)
    chi_st = 2.0  # 化学计量标量耗散率，s⁻¹

    bc = flamelet_boundary_conditions()
    print(f"  节点数: {n_nodes}")
    print(f"  化学计量混合分数 Z_st = {bc['Z_st']:.4f}")
    print(f"  氧化剂温度 T_ox = {bc['T_left']:.1f} K")
    print(f"  燃料温度 T_fuel = {bc['T_right']:.1f} K")
    print(f"  绝热火焰温度 T_ad = {bc['T_ad']:.1f} K")
    print(f"  标量耗散率 χ_st = {chi_st:.2e} s⁻¹")

    # 初始猜测场
    # 温度：在 Z_st 附近给出高温点火核，峰值低于绝热温度
    T_peak = 1500.0
    T_init = T_OXIDIZER + (T_peak - T_OXIDIZER) * np.exp(
        -((Z_nodes - Z_STOICHIOMETRIC) ** 2) / (2.0 * 0.06 ** 2)
    )
    T_init = np.clip(T_init, T_OXIDIZER, 2200.0)

    # 燃料：在氧化剂侧几乎为 0，燃料侧接近 1
    Y_F_init = np.clip(Z_nodes ** 1.5, 0.0, 1.0)

    # 氧化剂：在燃料侧几乎为 0，氧化剂侧接近 0.232
    Y_O_init = np.clip(0.232 * (1.0 - Z_nodes ** 0.8), 0.0, 0.232)

    # =====================================================================
    # 2. FEM 求解温度场（线性单元）
    # =====================================================================
    print_section("2. FEM 求解稳态火焰面温度场 (线性单元)")

    T_fem, iter_thermal = solve_fem_thermal(
        n_nodes, Z_nodes, T_init, Y_F_init, Y_O_init, chi_st,
        tol=1.0e-10, max_iter=100
    )
    print(f"  非线性迭代收敛: {iter_thermal} 次")
    print(f"  温度范围: [{np.min(T_fem):.1f}, {np.max(T_fem):.1f}] K")
    print(f"  化学计量点温度: {np.interp(Z_STOICHIOMETRIC, Z_nodes, T_fem):.1f} K")

    # =====================================================================
    # 3. 二次 FEM 求解组分场
    # =====================================================================
    print_section("3. 二次 FEM 求解组分质量分数分布")

    Y_F_fem, iter_fuel = solve_fem_quadratic_species(
        n_nodes, Z_nodes, 'fuel', T_fem, chi_st, tol=1.0e-10, max_iter=50
    )
    Y_O_fem, iter_ox = solve_fem_quadratic_species(
        n_nodes, Z_nodes, 'oxidizer', T_fem, chi_st, tol=1.0e-10, max_iter=50
    )
    print(f"  燃料方程迭代: {iter_fuel} 次")
    print(f"  氧化剂方程迭代: {iter_ox} 次")
    print(f"  燃料质量分数范围: [{np.min(Y_F_fem):.4f}, {np.max(Y_F_fem):.4f}]")
    print(f"  氧化剂质量分数范围: [{np.min(Y_O_fem):.4f}, {np.max(Y_O_fem):.4f}]")

    # =====================================================================
    # 4. FD 求解标量耗散率修正方程
    # =====================================================================
    print_section("4. FDM 求解标量耗散率修正方程")

    chi_fd, iter_chi = solve_fd_scalar_dissipation(
        n_nodes, Z_nodes, chi_st, C_chi=2.0, omega_turb=100.0, k_turb=10.0
    )
    print(f"  迭代收敛: {iter_chi} 次")
    print(f"  标量耗散率范围: [{np.min(chi_fd):.4e}, {np.max(chi_fd):.4e}] s⁻¹")

    # =====================================================================
    # 5. 点火/熄火极限分析（多项式求根）
    # =====================================================================
    print_section("5. 点火与熄火极限分析 (WDK 多项式求根)")

    ignition_results = analyze_ignition_extinction(Ze=8.0, beta=10.0, sigma=0.135)
    print(f"  Zel'dovich 数 Ze = {ignition_results['Zel_dovich_number']:.2f}")
    print(f"  无量纲活化能 β = {ignition_results['activation_energy_beta']:.2f}")
    print(f"  临界 Damköhler 数 Da_cr = {ignition_results['critical_Damkohler']:.6e}")
    print(f"  点火极限 Da_ig = {ignition_results['ignition_limit']:.6e}")
    print(f"  熄火极限 Da_ext = {ignition_results['extinction_limit']:.6e}")
    print(f"  WDK 收敛: {ignition_results['converged']}")

    # =====================================================================
    # 6. 湍流随机脉动场生成
    # =====================================================================
    print_section("6. 湍流速度脉动场与标量耗散率脉动")

    u_fluct, turb_stats = generate_turbulent_velocity_fluctuation(
        n_nodes, k_turb=10.0, epsilon=100.0, integral_length=0.01, seed=314159265.0
    )
    chi_fluct = scalar_dissipation_rate_fluctuation(Z_nodes, chi_st, u_fluct)
    print(f"  湍流均方根速度 u_rms = {turb_stats['u_rms']:.4f} m/s")
    print(f"  Lagrangian 时间尺度 T_L = {turb_stats['T_L']:.4e} s")
    print(f"  标量耗散率脉动范围: [{np.min(chi_fluct):.4e}, {np.max(chi_fluct):.4e}] s⁻¹")

    # =====================================================================
    # 7. 火焰前锋几何与曲率效应
    # =====================================================================
    print_section("7. 火焰前锋几何与曲率效应分析")

    area, area_ratio = flame_front_surface_area(B=0.01, L=0.03, w=0.003, Ka=1.5)
    print(f"  火焰前锋表面积 A = {area:.6e} m²")
    print(f"  皱褶因子 A_T/A_L = {area_ratio:.4f}")

    S_L = 0.4
    Le = 1.0
    L_M = markstein_length(Le, alpha_diff=2.0e-5, S_L=S_L)
    curvature = 50.0  # m⁻¹
    S_n = curved_flame_speed(S_L, curvature, Le)
    print(f"  Markstein 长度 L_M = {L_M:.6e} m")
    print(f"  层流火焰速度 S_L = {S_L:.2f} m/s")
    print(f"  曲率修正火焰速度 S_n = {S_n:.4f} m/s")

    # =====================================================================
    # 8. 3D 燃烧室网格生成
    # =====================================================================
    print_section("8. 3D 圆柱燃烧室非结构网格生成")

    p_3d, t_3d = generate_mesh_3d(h0=0.015, R=0.05, H=0.20, iteration_max=15)
    print(f"  生成节点数: {len(p_3d)}")
    print(f"  生成四面体单元数: {len(t_3d)}")

    # =====================================================================
    # 9. 带宽分析
    # =====================================================================
    print_section("9. 刚度矩阵带宽分析")

    bw_linear = analyze_flamelet_bandwidth(n_nodes, fem_type='linear')
    bw_quadratic = analyze_flamelet_bandwidth(n_nodes, fem_type='quadratic')
    print(f"  线性 FEM 带宽: {bw_linear['total_bandwidth']}")
    print(f"  二次 FEM 带宽: {bw_quadratic['total_bandwidth']}")
    print(f"  线性 FEM 稀疏率: {bw_linear['sparse_ratio']:.4e}")
    print(f"  二次 FEM 稀疏率: {bw_quadratic['sparse_ratio']:.4e}")

    # =====================================================================
    # 10. 空间分区
    # =====================================================================
    print_section("10. 混合分数空间域分解与 Voronoi 分区")

    sub_idx, sub_bounds = domain_decomposition_1d(Z_nodes, n_subdomains=4)
    print(f"  子域数量: 4")
    for i, (idx, bound) in enumerate(zip(sub_idx, sub_bounds)):
        print(f"    子域 {i+1}: 索引 [{idx[0]}, {idx[1]}], Z ∈ [{bound[0]:.4f}, {bound[1]:.4f}]")

    # 二维 Voronoi 面积分析（简化示例）
    seeds_2d = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]])
    areas, lb = voronoi_area_2d(seeds_2d, bbox=((0, 1), (0, 1)), resolution=200)
    print(f"  Voronoi 负载均衡度: {lb:.4f}")

    # =====================================================================
    # 11. 化学计量多项式路径计数
    # =====================================================================
    print_section("11. 化学反应路径组合计数")

    reaction_steps = [1, 2, 3]
    target = 10
    max_steps = 8
    count, poly = stoichiometric_paths(reaction_steps, target, max_steps)
    print(f"  反应步长: {reaction_steps}")
    print(f"  目标变化: {target}")
    print(f"  最大步数: {max_steps}")
    print(f"  路径数: {count}")

    reactions = [
        {'name': 'R1', 'stoich_change': 1, 'rate': 1.0e5},
        {'name': 'R2', 'stoich_change': 2, 'rate': 5.0e4},
        {'name': 'R3', 'stoich_change': 3, 'rate': 2.0e4},
    ]
    complexity = reaction_mechanism_complexity(reactions, max_depth=6)
    print(f"  反应机理有效复杂度: {complexity['effective_complexity']:.4e}")

    # =====================================================================
    # 12. 火焰不稳定性分析
    # =====================================================================
    print_section("12. 火焰不稳定性动力学 (Darrieus-Landau)")

    t_arr, y_arr = integrate_flame_instability(
        (0.0, 0.05), y0=np.array([0.0, 0.0]), dt=1.0e-5,
        S_L=0.4, rho_u=1.2, rho_b=0.2, gamma_damp=50.0,
        omega_0=200.0, lam_thermal=100.0, mu_acoustic=500.0
    )
    print(f"  积分时间: 0.0 ~ {t_arr[-1]:.4f} s")
    print(f"  最大位移: {np.max(np.abs(y_arr[:, 0])):.6e} m")
    print(f"  最大速度: {np.max(np.abs(y_arr[:, 1])):.4f} m/s")

    k_wave = np.linspace(1.0, 500.0, 100)
    sigma_dl = darrieus_landau_growth_rate(k_wave, S_L=0.4, rho_u=1.2, rho_b=0.2)
    print(f"  DL 最大增长率: {np.max(sigma_dl):.2f} s⁻¹")

    # =====================================================================
    # 13. 精确解验证
    # =====================================================================
    print_section("13. 数值解验证 (Manufactured Solution)")

    T_exact_ms, _ = manufactured_solution_temperature(Z_nodes, T_ox=T_OXIDIZER, T_ad=ADIA_TEMP_STOIC)
    errors_ms = compute_errors(T_fem, T_exact_ms, Z_nodes)
    print(f"  L² 误差: {errors_ms['L2_error']:.4e}")
    print(f"  L∞ 误差: {errors_ms['Linf_error']:.4e}")
    print(f"  H¹ 半范数误差: {errors_ms['H1_semi_error']:.4e}")
    print(f"  相对 L² 误差: {errors_ms['relative_L2']:.4e}")

    # 高斯渐近解比较
    omega_max = 1.0e3
    T_gauss, sigma_flame = gaussian_flamelet_solution(
        Z_nodes, Z_STOICHIOMETRIC, T_OXIDIZER, ADIA_TEMP_STOIC, chi_st, omega_max
    )
    errors_gauss = compute_errors(T_fem, T_gauss, Z_nodes)
    print(f"  火焰厚度 σ = {sigma_flame:.6e}")
    print(f"  与高斯渐近解相对 L² 误差: {errors_gauss['relative_L2']:.4e}")

    # =====================================================================
    # 14. 守恒律校验
    # =====================================================================
    print_section("14. 质量/能量守恒校验")

    validation = validate_simulation(
        T_fem, Y_F_fem, Y_O_fem, Z_nodes,
        mass_tol=1.0e-2, energy_tol=3.0e-1
    )
    print(f"  质量守恒误差: {validation['mass_conservation_error']:.4e}")
    print(f"  质量守恒通过: {validation['mass_conservation_passed']}")
    print(f"  能量守恒误差: {validation['energy_conservation_error']:.4e}")
    print(f"  能量守恒通过: {validation['energy_conservation_passed']}")
    print(f"  温度范围: [{validation['temperature_range'][0]:.1f}, {validation['temperature_range'][1]:.1f}] K")
    print(f"  整体有效性: {validation['overall_valid']}")

    # =====================================================================
    # 15. 综合结果汇总
    # =====================================================================
    print_section("综合结果汇总")
    print(f"  火焰面峰值温度: {np.max(T_fem):.1f} K")
    print(f"  化学计量点温度: {np.interp(Z_STOICHIOMETRIC, Z_nodes, T_fem):.1f} K")
    print(f"  临界 Damköhler 数: {ignition_results['critical_Damkohler']:.6e}")
    print(f"  火焰皱褶因子: {area_ratio:.4f}")
    print(f"  曲率修正火焰速度: {S_n:.4f} m/s")
    print(f"  3D 网格节点数: {len(p_3d)}")
    print(f"  数值解相对误差: {errors_ms['relative_L2']:.4e}")
    print(f"  模拟验证状态: {'通过' if validation['overall_valid'] else '未通过'}")
    print("\n  === 湍流燃烧火焰面模型计算完成 ===\n")

    # 返回结果字典（供测试使用）
    return {
        'Z_nodes': Z_nodes,
        'T_fem': T_fem,
        'Y_F_fem': Y_F_fem,
        'Y_O_fem': Y_O_fem,
        'chi_fd': chi_fd,
        'ignition_results': ignition_results,
        'turb_stats': turb_stats,
        'area_ratio': area_ratio,
        'S_n': S_n,
        'p_3d': p_3d,
        't_3d': t_3d,
        'validation': validation,
        'errors_ms': errors_ms,
    }


if __name__ == "__main__":
    results = main()

# ================================================================
# 测试用例（28个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: main() 返回结果字典包含所有预期键 ----
assert isinstance(results, dict), '[TC01] 结果应为字典 FAILED'
expected_keys = ['Z_nodes', 'T_fem', 'Y_F_fem', 'Y_O_fem', 'chi_fd',
                 'ignition_results', 'turb_stats', 'area_ratio', 'S_n',
                 'p_3d', 't_3d', 'validation', 'errors_ms']
for key in expected_keys:
    assert key in results, f'[TC01] 缺少键 {key} FAILED'

# ---- TC02: Z_nodes 严格单调递增且范围 [0,1] ----
Z_nodes = results['Z_nodes']
assert np.all(np.diff(Z_nodes) > 0), '[TC02] Z_nodes 必须单调递增 FAILED'
assert abs(Z_nodes[0]) < 1e-12, '[TC02] Z_nodes[0] 应为 0 FAILED'
assert abs(Z_nodes[-1] - 1.0) < 1e-12, '[TC02] Z_nodes[-1] 应为 1 FAILED'

# ---- TC03: T_fem 所有值有限且在合理温度范围内 ----
T_fem = results['T_fem']
assert np.all(np.isfinite(T_fem)), '[TC03] T_fem 存在非有限值 FAILED'
assert np.all(T_fem >= 200.0), '[TC03] T_fem 存在低于 200K 的值 FAILED'
assert np.all(T_fem <= 3500.0), '[TC03] T_fem 存在高于 3500K 的值 FAILED'

# ---- TC04: Y_F_fem 所有值在 [0,1] 范围内 ----
Y_F_fem = results['Y_F_fem']
assert np.all(np.isfinite(Y_F_fem)), '[TC04] Y_F_fem 存在非有限值 FAILED'
assert np.all(Y_F_fem >= -1e-6), '[TC04] Y_F_fem 存在负值 FAILED'
assert np.all(Y_F_fem <= 1.0 + 1e-6), '[TC04] Y_F_fem 存在超过1的值 FAILED'

# ---- TC05: Y_O_fem 所有值在 [0,1] 范围内 ----
Y_O_fem = results['Y_O_fem']
assert np.all(np.isfinite(Y_O_fem)), '[TC05] Y_O_fem 存在非有限值 FAILED'
assert np.all(Y_O_fem >= -1e-6), '[TC05] Y_O_fem 存在负值 FAILED'
assert np.all(Y_O_fem <= 1.0 + 1e-6), '[TC05] Y_O_fem 存在超过1的值 FAILED'

# ---- TC06: chi_fd 所有值非负且有限 ----
chi_fd = results['chi_fd']
assert np.all(np.isfinite(chi_fd)), '[TC06] chi_fd 存在非有限值 FAILED'
assert np.all(chi_fd >= 0.0), '[TC06] chi_fd 存在负值 FAILED'

# ---- TC07: ignition_results 包含有效的正临界 Damköhler 数 ----
ig = results['ignition_results']
assert ig['critical_Damkohler'] > 0, '[TC07] 临界 Damköhler 数必须为正 FAILED'
assert ig['converged'], '[TC07] WDK 算法应收敛 FAILED'

# ---- TC08: validation 整体通过 ----
val = results['validation']
assert val['overall_valid'], '[TC08] 模拟验证应整体通过 FAILED'

# ---- TC09: 数值解相对 L² 误差小 ----
errs = results['errors_ms']
assert errs['relative_L2'] < 1.0, '[TC09] 相对 L² 误差应小于 1 FAILED'

# ---- TC10: area_ratio 为正数 ----
assert results['area_ratio'] > 0, '[TC10] 皱褶因子应为正数 FAILED'

# ---- TC11: S_n 为有限正值 ----
S_n = results['S_n']
assert np.isfinite(S_n) and S_n > 0, '[TC11] S_n 应为有限正值 FAILED'

# ---- TC12: 3D 网格有正数节点和单元 ----
assert len(results['p_3d']) > 0, '[TC12] 3D 网格节点数应为正 FAILED'
assert len(results['t_3d']) > 0, '[TC12] 3D 网格单元数应为正 FAILED'

# ---- TC13: scalar_dissipation_rate 在 Z_st 处等于 chi_st ----
chi_test = scalar_dissipation_rate(Z_STOICHIOMETRIC, 2.0)
assert abs(chi_test - 2.0) < 1e-6, '[TC13] χ(Z_st) 应等于 χ_st FAILED'

# ---- TC14: mixture_molecular_weight 在 Z=0 处返回氧化剂分子量 ----
W0 = mixture_molecular_weight(0.0)
assert abs(W0 - 28.97e-3) < 1e-6, '[TC14] W(0) 应等于 M_OX FAILED'

# ---- TC15: arrhenius_rate_constant 随温度单调递增 ----
k1 = arrhenius_rate_constant(500.0, A=1.0, Ea=5000.0)
k2 = arrhenius_rate_constant(1500.0, A=1.0, Ea=5000.0)
assert k2 > k1, '[TC15] Arrhenius 速率常数应随温度递增 FAILED'

# ---- TC16: adiabatic_flame_temperature 返回有限正值 ----
T_ad_test, phi_test = adiabatic_flame_temperature(0.05, 0.232, 300.0)
assert np.isfinite(T_ad_test) and T_ad_test > 300.0, '[TC16] 绝热火焰温度应为大于初温的有限值 FAILED'

# ---- TC17: markstein_length Le=1 时等于 δ_L ----
L_M_eq = markstein_length(1.0, alpha_diff=2.0e-5, S_L=0.4)
delta_L = 2.0e-5 / 0.4
assert abs(L_M_eq - delta_L) < 1e-12, '[TC17] Le=1 时 L_M 应等于 δ_L FAILED'

# ---- TC18: curved_flame_speed 在零曲率时等于层流速度 ----
S_n_zero = curved_flame_speed(0.4, 0.0, 1.0)
assert abs(S_n_zero - 0.4) < 1e-12, '[TC18] 零曲率时 S_n 应等于 S_L FAILED'

# ---- TC19: flame_front_surface_area 返回正面积和比率 ----
area_test, ratio_test = flame_front_surface_area(B=0.01, L=0.03, w=0.003, Ka=0.0)
assert area_test > 0, '[TC19] 火焰表面积应为正 FAILED'
assert ratio_test > 0, '[TC19] 皱褶因子应为正 FAILED'

# ---- TC20: Darrieus-Landau 增长率在 k=0 时为零 ----
sigma_dl_zero = darrieus_landau_growth_rate(0.0)
assert abs(sigma_dl_zero) < 1e-12, '[TC20] k=0 时 D-L 增长率应为零 FAILED'

# ---- TC21: poly_eval 对多项式 x^2 - 4 正确求值 ----
from ignition_polynomial import poly_eval, wdk_roots
c_test = np.array([-4.0, 0.0, 1.0], dtype=complex)
assert abs(poly_eval(c_test, 2.0)) < 1e-12, '[TC21] P(2)=0 for x²-4 FAILED'
assert abs(poly_eval(c_test, 0.0) + 4.0) < 1e-12, '[TC21] P(0)=-4 for x²-4 FAILED'

# ---- TC22: wdk_roots 对 x^2 - 4 找到 ±2 ----
roots_test, conv = wdk_roots(c_test)
assert conv, '[TC22] WDK 应对简单二次多项式收敛 FAILED'
found_p2 = any(abs(r - 2.0) < 1e-6 for r in roots_test)
found_m2 = any(abs(r + 2.0) < 1e-6 for r in roots_test)
assert found_p2 and found_m2, '[TC22] WDK 应找到根 ±2 FAILED'

# ---- TC23: compute_errors 返回正确键且零误差 ----
err_test = compute_errors(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]),
                          np.array([0.0, 0.5, 1.0]))
assert 'L2_error' in err_test, '[TC23] compute_errors 缺少 L2_error FAILED'
assert 'Linf_error' in err_test, '[TC23] compute_errors 缺少 Linf_error FAILED'
assert abs(err_test['L2_error']) < 1e-12, '[TC23] 相同输入的 L² 误差应为零 FAILED'

# ---- TC24: polynomial_multiply 正确计算卷积 ----
from stoichiometric_polynomial import polynomial_multiply
p_test = polynomial_multiply(np.array([1.0, 1.0]), np.array([1.0, 1.0]))
assert len(p_test) == 3, '[TC24] 乘积长度应为 3 FAILED'
assert abs(p_test[0] - 1.0) < 1e-12, '[TC24] 系数[0] 应为 1 FAILED'
assert abs(p_test[1] - 2.0) < 1e-12, '[TC24] 系数[1] 应为 2 FAILED'
assert abs(p_test[2] - 1.0) < 1e-12, '[TC24] 系数[2] 应为 1 FAILED'

# ---- TC25: chicken_egg_shape 在 x=0 处达到最大半径 B/2 ----
x_mid = np.array([0.0])
r_mid = chicken_egg_shape(0.01, 0.03, 0.0, x_mid)
assert abs(r_mid[0] - 0.005) < 1e-12, '[TC25] x=0 处 r 应等于 B/2 FAILED'

# ---- TC26: mass_conservation_checksum 验证总和为 1 ----
from conservation_validator import mass_conservation_checksum
Y_dict = {'fuel': np.array([0.0, 0.5, 1.0]), 'oxidizer': np.array([1.0, 0.5, 0.0])}
cs, cs_err = mass_conservation_checksum(Y_dict)
assert np.max(np.abs(cs - 1.0)) < 1e-12, '[TC26] 质量分数和应恒为 1 FAILED'

# ---- TC27: domain_decomposition_1d 正确分区 ----
Z_test = np.linspace(0.0, 1.0, 20)
sub_idx, sub_bounds = domain_decomposition_1d(Z_test, 4)
assert len(sub_idx) == 4, '[TC27] 应有 4 个子域 FAILED'
assert sub_idx[0][0] == 0, '[TC27] 第一个子域应从索引 0 开始 FAILED'
assert sub_idx[-1][1] == 20, '[TC27] 最后一个子域应到索引 20 FAILED'

# ---- TC28: flamelet_boundary_conditions 返回正确结构 ----
bc_test = flamelet_boundary_conditions()
assert 'T_left' in bc_test and 'T_right' in bc_test, '[TC28] 边界条件缺少温度键 FAILED'
assert abs(bc_test['T_ad'] - 2226.0) < 1.0, '[TC28] 绝热温度应约为 2226 K FAILED'

print('\n全部 28 个测试通过!\n')
