"""
main.py
=======
PROJECT 278: 计算材料 — 相图计算与 CALPHAD 建模
          高阶有限差分与稳定性分析 (小规模可复现实验)

统一入口: 零参数可运行。

Fe-C 二元合金体系的完整 CALPHAD 工作流:
  1. CVT 非均匀网格生成
  2. Gibbs 能曲面计算
  3. 分段常数相分析
  4. Newton-Maehly 相平衡求解
  5. 分岔检测与相界追踪
  6. 高阶紧致差分的 Cahn-Hilliard 模拟
  7. 刚性 ODE 积分与稳定性分析
  8. Lyapunov 指数与色散关系
  9. Allen-Cahn 相变动力学 (中点法)
  10. 高斯螺旋扫描搜索
  11. 蒙特卡罗不确定性传播
  12. MOEA/D 参数优化
  13. 统计学敏感性分析

每个模块源自一个种子项目, 详见 README_博士级合成说明.md。
"""

import sys
import time
import numpy as np

# ============================================================
# 导入所有子模块
# ============================================================
from calphad_fec_constants import (
    R_GAS, X_C_MIN, X_C_MAX, N_X_GRID,
    T_MIN, T_MAX, N_T_GRID,
    CH_MOBILITY_0, CH_KAPPA, CH_LX, CH_NX, CH_DT, CH_N_STEPS,
    MC_N_SAMPLES, MC_RANDOM_SEED,
)
from gibbs_energy_calphad import (
    gibbs_substitutional,
    chemical_potential_C,
    chemical_potential_Fe,
    second_derivative_G,
    gibbs_cementite,
)
from high_order_fd import (
    compact_first_derivative,
    compact_second_derivative,
    compact_fourth_derivative,
    von_neumann_stability_factor,
)
from newton_maehly_equilibrium import (
    newton_maehly_solve,
    common_tangent_search,
    lu_factor,
    lu_solve,
)
from backward_euler_calphad import (
    backward_euler_picard,
    backward_euler_adaptive,
    mobility_function,
)
from cvt_composition_mesh import (
    lloyd_cvt_1d,
    density_function,
)
from monte_carlo_hyperball import (
    sample_positive_hyperball,
    pairwise_distance_statistics,
    calphad_uncertainty_propagation,
)
from calphad_parameter_optimizer import (
    moead_calphad_optimize,
    pbi_scalarization,
    chebyshev_scalarization,
)
from lyapunov_phase_stability import (
    lyapunov_knn_estimate,
    linear_stability_dispersion,
    free_energy_lyapunov_functional,
    spinodal_boundary_search,
)
from bifurcation_detection import (
    detect_tangent_bifurcation,
    border_collision_map,
    jacobian_determinant_2phase,
)
from ode_midpoint_phasefield import (
    ode_midpoint_implicit,
    phase_transformation_trajectory,
    interpolation_phi,
    double_well_g,
)
from piecewise_constant_phase import (
    build_phase_indicator,
    pwc_expanded_coordinates,
    smoothed_gibbs_landscape,
    pwc_approximation_error,
)
from gaussian_spiral_scan import (
    thermal_spiral_scan,
    spiral_phase_boundary_trace,
    is_gaussian_prime,
)
from clinical_statistics_calphad import (
    welch_t_test,
    clopper_pearson_ci,
    wilcoxon_signed_rank_test,
    tipping_point_analysis,
    gamblers_ruin_parameter_survival,
)
from stiff_ode_cahn_hilliard import (
    stiff_ch_integrator,
    jacobian_eigenvalue_estimate,
)


def separator(title, char='=', width=70):
    """打印分隔线。"""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


# ============================================================
# Step 1: CVT 网格生成 (种子: 259_cvt_square_nonuniform)
# ============================================================
def step1_cvt_mesh():
    separator("Step 1: CVT 非均匀成分网格生成 (Lloyd 算法)")

    T_ref = 1000.0  # K
    cvt_result = lloyd_cvt_1d(T_ref, 'FCC',
                              n_generators=30, n_samples=1000,
                              max_iter=50)

    gen = cvt_result['generators']
    print(f"  CVT 生成点数: {len(gen)}")
    print(f"  Lloyd 迭代次数: {cvt_result['n_iter']}")
    print(f"  CVT 能量 (量化误差): {cvt_result['energy']:.6e}")
    print(f"  成分范围: [{gen[0]:.6e}, {gen[-1]:.6e}]")
    print(f"  前5个生成点: {gen[:5]}")
    print(f"  [OK] CVT 网格生成完成")
    return cvt_result


# ============================================================
# Step 2: Gibbs 能曲面计算
# ============================================================
def step2_gibbs_energy():
    separator("Step 2: 多相 Gibbs 能曲面计算")

    T_ref = 1000.0  # K
    x_grid = np.linspace(X_C_MIN, 0.20, N_X_GRID)

    phases = ['LIQUID', 'FCC', 'BCC']
    for phase in phases:
        G = gibbs_substitutional(x_grid, T_ref, phase)
        print(f"  {phase} @ {T_ref}K:")
        print(f"    G_min = {np.min(G):.2f} J/mol")
        print(f"    G_max = {np.max(G):.2f} J/mol")
        print(f"    G_mean = {np.mean(G):.2f} J/mol")

    # 渗碳体
    G_cem = gibbs_cementite(T_ref)
    print(f"  CEMENTITE @ {T_ref}K: G = {G_cem:.2f} J/mol")

    # d²G/dc² 计算
    d2G_fcc = second_derivative_G(x_grid, T_ref, 'FCC')
    n_neg = np.sum(d2G_fcc < 0)
    print(f"\n  FCC d²G/dc² < 0 的点数: {n_neg}/{N_X_GRID} (spinodal 区域)")
    if n_neg > 0:
        x_neg = x_grid[d2G_fcc < 0]
        print(f"  FCC spinodal 范围: [{x_neg[0]:.6f}, {x_neg[-1]:.6f}]")

    print(f"  [OK] Gibbs 能计算完成")
    return x_grid


# ============================================================
# Step 3: 分段常数相分析 (种子: 923_pwc_plot_1d)
# ============================================================
def step3_piecewise_phase():
    separator("Step 3: 分段常数相指示函数与 Gibbs 能景观")

    T_ref = 1000.0
    x_grid = np.linspace(X_C_MIN, 0.20, 200)
    phases = ['LIQUID', 'FCC', 'BCC']

    indicator = build_phase_indicator(x_grid, T_ref, phases)
    print(f"  温度: {T_ref} K")
    print(f"  相边界数: {len(indicator['boundaries'])}")
    for bd in indicator['boundaries']:
        print(f"    {bd['phase_left']} | {bd['phase_right']} @ x = {bd['x_boundary']:.6f}")

    # 扩展坐标 (PWC 绘图格式)
    xp, yp = pwc_expanded_coordinates(
        x_grid, indicator['phase_index'], indicator['phase_index']
    )
    print(f"  扩展坐标对数: {len(xp)}")

    # 光滑化 Gibbs 能
    smooth = smoothed_gibbs_landscape(x_grid, T_ref, phases,
                                       smoothing_width=0.01)
    print(f"  光滑化 beta: {smooth['beta']:.2f}")
    for phase in phases:
        frac = smooth['phase_fractions'][phase]
        print(f"    {phase} 平均分数: {np.mean(frac):.4f}")

    # 近似误差分析
    err_result = pwc_approximation_error(x_grid, T_ref, phases,
                                          n_refinements=3)
    print(f"  PWC 近似误差:")
    for i in range(len(err_result['n_grid'])):
        n = err_result['n_grid'][i]
        e = err_result['max_errors'][i]
        print(f"    N={n:4d}: max_error = {e:.2f} J/mol")

    print(f"  [OK] 分段常数相分析完成")


# ============================================================
# Step 4: Newton-Maehly 相平衡 (种子: 801_newton_maehly + 689_linpack_d)
# ============================================================
def step4_newton_equilibrium():
    separator("Step 4: Newton-Maehly 相平衡求解")

    T_values = [900.0, 1000.0, 1100.0, 1200.0, 1300.0]

    for T in T_values:
        # FCC/BCC 平衡
        result = newton_maehly_solve(T, 'FCC', 'BCC')
        status = "✓" if result['converged'] else "✗"
        print(f"  T={T:.0f}K FCC/BCC: {status} "
              f"x_a={result['x_C_alpha']:.6f}, x_b={result['x_C_beta']:.6f} "
              f"(iter={result['n_iter']}, res={result['residual']:.2e})")

    # LU 分解测试
    A = np.array([[4.0, 3.0], [6.0, 3.0]])
    b = np.array([10.0, 12.0])
    LU, piv = lu_factor(A)
    x_sol = lu_solve(LU, piv, b)
    err = np.max(np.abs(A @ x_sol - b))
    print(f"\n  LU 分解验证: ||Ax - b|| = {err:.2e}")

    print(f"  [OK] Newton-Maehly 求解完成")


# ============================================================
# Step 5: 分岔检测 (种子: 1085_border-collision-bifurcations)
# ============================================================
def step5_bifurcation():
    separator("Step 5: 相边界分岔检测")

    result = detect_tangent_bifurcation(
        (800.0, 1500.0), 'FCC', 'BCC', n_T=20
    )
    print(f"  相界类型: {result['phase_boundary_type']}")
    print(f"  分岔点数: {len(result['bifurcation_points'])}")
    for bp in result['bifurcation_points'][:5]:
        print(f"    T={bp['T_bifurcation']:.1f}K, type={bp['type']}")

    # Jacobian 行列式
    T_test = 1000.0
    detJ = jacobian_determinant_2phase(0.005, 0.02, T_test, 'FCC', 'BCC')
    print(f"\n  det(J) @ T={T_test}K: {detJ:.4e}")

    # 边界碰撞图
    bc_result = border_collision_map(1000.0, 'FCC', 'BCC',
                                      (X_C_MIN, 0.15), n_scan=100)
    print(f"  等能/等势点: {len(bc_result['equilibrium_points'])}")

    print(f"  [OK] 分岔检测完成")


# ============================================================
# Step 6: 高阶紧致差分 Cahn-Hilliard (种子: 064_backward_euler)
# ============================================================
def step6_cahn_hilliard():
    separator("Step 6: 高阶紧致差分 Cahn-Hilliard 模拟")

    T_sim = 900.0  # 低于 spinodal 温度
    N = 64
    h = CH_LX / N
    c0 = 0.05 + 0.01 * np.random.RandomState(42).randn(N)
    c0 = np.clip(c0, 1e-6, 0.3)

    # 紧致差分验证
    x_test = np.sin(2 * np.pi * np.arange(N) / N)
    df_exact = 2 * np.pi / N * np.cos(2 * np.pi * np.arange(N) / N)
    df_compact = compact_first_derivative(x_test, h)
    err_fd = np.max(np.abs(df_compact - df_exact))
    print(f"  紧致差分精度 (一阶): max_err = {err_fd:.4e}")

    # Backward Euler + Picard
    print(f"\n  Backward Euler 模拟 (T={T_sim}K, N={N}):")
    result = backward_euler_picard(
        c0, T_sim, CH_DT, min(CH_N_STEPS, 50), h,
        CH_MOBILITY_0, CH_KAPPA, 'FCC'
    )
    print(f"    收敛: {result['converged']}")
    print(f"    总 Picard 迭代: {result['n_iter_total']}")
    print(f"    初始能量: {result['energy_history'][0]:.4e}")
    print(f"    最终能量: {result['energy_history'][-1]:.4e}")
    print(f"    浓度范围: [{np.min(result['c_final']):.6f}, "
          f"{np.max(result['c_final']):.6f}]")

    print(f"  [OK] Cahn-Hilliard 模拟完成")


# ============================================================
# Step 7: 刚性 ODE 积分 (种子: 841_ozone_ode)
# ============================================================
def step7_stiff_ode():
    separator("Step 7: 刚性 ODE 积分器 (CH 方程)")

    N = 32
    h = CH_LX / N
    T_sim = 950.0
    c0 = 0.06 + 0.005 * np.random.RandomState(123).randn(N)
    c0 = np.clip(c0, 1e-6, 0.3)

    # 刚性检测
    eig = jacobian_eigenvalue_estimate(c0, T_sim, h, CH_MOBILITY_0,
                                        CH_KAPPA, 'FCC')
    print(f"  Jacobian 特征值估计:")
    print(f"    lambda_max = {eig['lambda_max_approx']:.4e}")
    print(f"    lambda_min = {eig['lambda_min_approx']:.4e}")
    print(f"    刚性比 = {eig['stiffness_ratio']:.2f}")
    print(f"    是否刚性: {eig['is_stiff']}")

    # 刚性积分
    result = stiff_ch_integrator(c0, T_sim, t_total=1e-3, h=h,
                                  M0=CH_MOBILITY_0, kappa=CH_KAPPA,
                                  phase='FCC', max_steps=50)
    print(f"\n  刚性积分结果:")
    print(f"    步数: {result['n_steps']}")
    print(f"    Newton 总迭代: {result['n_total_newton']}")
    print(f"    最终浓度范围: [{np.min(result['c_final']):.6f}, "
          f"{np.max(result['c_final']):.6f}]")

    print(f"  [OK] 刚性 ODE 积分完成")


# ============================================================
# Step 8: Lyapunov 稳定性分析 (种子: 1204_lle-chaos-demos)
# ============================================================
def step8_lyapunov():
    separator("Step 8: Lyapunov 指数与线性稳定性分析")

    T_sim = 900.0
    x0 = 0.05  # 测试成分

    # 色散关系
    disp = linear_stability_dispersion(T_sim, x0, 'FCC', CH_KAPPA,
                                        CH_MOBILITY_0, n_wavenum=32)
    print(f"  线性稳定性分析 (T={T_sim}K, x_C={x0}):")
    print(f"    d²G/dc² = {disp['d2G_dc2']:.4e} J/mol")
    print(f"    处于 spinodal: {disp['is_spinodal']}")
    print(f"    临界波数 k_c = {disp['k_critical']:.4e} rad/m")
    print(f"    最快增长波数 k_m = {disp['k_max_growth']:.4e} rad/m")
    print(f"    最大增长率 omega_m = {disp['omega_max']:.4e} s^-1")

    # Spinodal 搜索
    spinodal = spinodal_boundary_search(T_sim, 'FCC')
    print(f"\n  Spinodal 边界点:")
    for sp in spinodal:
        print(f"    x_C = {sp:.6f}")

    # LLE 估计 (从合成时间序列)
    rng = np.random.RandomState(278)
    t_series = 0.05 + 0.02 * np.exp(0.01 * np.arange(200)) \
        + 0.001 * rng.randn(200)
    lle_result = lyapunov_knn_estimate(t_series, embed_dim=4,
                                        time_lag=2, max_horizon=10)
    print(f"\n  Lyapunov 指数估计:")
    print(f"    LLE = {lle_result['lle']:.6f}")
    print(f"    不稳定: {lle_result['is_unstable']}")

    # 自由能泛函
    N = 64
    h = CH_LX / N
    c_test = 0.05 + 0.02 * np.sin(2 * np.pi * np.arange(N) / N)
    V = free_energy_lyapunov_functional(c_test, T_sim, h, CH_KAPPA, 'FCC')
    print(f"\n  自由能泛函 V[c] = {V:.4e} J/m")

    # von Neumann 稳定性
    k_test = np.logspace(3, 7, 10)
    k_eff_sq = von_neumann_stability_factor(k_test, h)
    print(f"\n  von Neumann 修正波数:")
    print(f"    k[0] = {k_test[0]:.2e}, k_eff² = {k_eff_sq[0]:.4e}")
    print(f"    k[-1] = {k_test[-1]:.2e}, k_eff² = {k_eff_sq[-1]:.4e}")

    print(f"  [OK] Lyapunov 分析完成")


# ============================================================
# Step 9: Allen-Cahn 相变动力学 (种子: 828_ode_midpoint)
# ============================================================
def step9_allen_cahn():
    separator("Step 9: Allen-Cahn 相变动力学 (中点法)")

    T_sim = 1050.0  # FCC/BCC 转变温度附近
    x_C = 0.01

    result = phase_transformation_trajectory(
        x_C, T_sim, L_kinetic=1e-2, W_barrier=3000.0,
        n_steps=100, phase_a='BCC', phase_b='FCC'
    )

    print(f"  BCC→FCC 相变 (T={T_sim}K, x_C={x_C}):")
    print(f"    收敛: {result['converged']}")
    print(f"    Picard 总迭代: {result['n_picard_total']}")
    print(f"    初始 eta: {result['y'][0]:.6f}")
    print(f"    最终 eta: {result['y'][-1]:.6f}")
    print(f"    时间范围: [{result['t'][0]:.2e}, {result['t'][-1]:.2e}]")

    # 插值函数验证
    eta_test = np.linspace(0, 1, 5)
    phi_vals = interpolation_phi(eta_test)
    g_vals = double_well_g(eta_test)
    print(f"\n  phi(eta) 验证:")
    for e, p, g in zip(eta_test, phi_vals, g_vals):
        print(f"    eta={e:.2f}: phi={p:.6f}, g={g:.6f}")

    print(f"  [OK] Allen-Cahn 动力学完成")


# ============================================================
# Step 10: 高斯螺旋扫描 (种子: 456_gaussian_prime_spiral)
# ============================================================
def step10_spiral_scan():
    separator("Step 10: 高斯整数螺旋扫描搜索")

    result = thermal_spiral_scan(
        (800.0, 1500.0), (X_C_MIN, 0.15),
        n_spiral=200, phase='FCC', scan_type='spinodal'
    )

    print(f"  螺旋路径长度: {len(result['spiral_path'])}")
    print(f"  发现的特征点: {result['n_features']}")
    print(f"  高斯素数命中: {result['n_prime_hits']}")

    # 验证高斯素数
    prime_examples = [(1, 1), (2, 1), (1, 2), (3, 0), (3, 2)]
    print(f"\n  高斯素数验证:")
    for a, b in prime_examples:
        print(f"    {a}+{b}i: prime={is_gaussian_prime(a, b)}")

    # 相边界追踪
    trace = spiral_phase_boundary_trace(1000.0, 0.005, 'FCC', 'BCC',
                                         max_steps=20, step_size=20.0)
    if trace['found']:
        print(f"\n  相边界追踪:")
        print(f"    追踪点数: {trace['n_points']}")
        print(f"    温度范围: [{trace['T_boundary'][0]:.0f}, "
              f"{trace['T_boundary'][-1]:.0f}] K")
        print(f"    x_alpha 范围: [{np.min(trace['x_alpha_boundary']):.6f}, "
              f"{np.max(trace['x_alpha_boundary']):.6f}]")

    print(f"  [OK] 螺旋扫描完成")


# ============================================================
# Step 11: Monte Carlo 不确定性 (种子: 555_hyperball + 450_gamblers_ruin)
# ============================================================
def step11_monte_carlo():
    separator("Step 11: 蒙特卡罗不确定性传播")

    # 超球面采样
    samples = sample_positive_hyperball(500, 6, seed=MC_RANDOM_SEED)
    print(f"  超球面采样: {samples.shape[0]} 点, {samples.shape[1]} 维")
    print(f"  样本范围: [{np.min(samples):.4f}, {np.max(samples):.4f}]")

    # 成对距离统计
    dist_stats = pairwise_distance_statistics(samples[:100])
    print(f"\n  成对距离统计:")
    print(f"    均值: {dist_stats['mean_distance']:.6f}")
    print(f"    标准差: {dist_stats['std_distance']:.6f}")
    print(f"    范围: [{dist_stats['min_distance']:.6f}, "
          f"{dist_stats['max_distance']:.6f}]")

    # 赌徒破产分析
    gr_result = gamblers_ruin_parameter_survival(
        1000.0, 'FCC', 'BCC',
        initial_stake=0.05, n_games=30, max_steps=30
    )
    print(f"\n  赌徒破产参数存活分析:")
    print(f"    平均存活步数: {gr_result['mean_survival']:.1f}")
    print(f"    破产概率: {gr_result['ruin_probability']:.4f}")

    print(f"  [OK] Monte Carlo 分析完成")


# ============================================================
# Step 12: MOEA/D 优化 (种子: 1219_MOEA-D)
# ============================================================
def step12_moea_optimizer():
    separator("Step 12: MOEA/D CALPHAD 参数优化")

    result = moead_calphad_optimize()

    print(f"  Pareto 前沿点数: {len(result['pareto_front'])}")
    if len(result['pareto_front']) > 0:
        print(f"  Pareto 前沿:")
        for i, pf in enumerate(result['pareto_front'][:5]):
            print(f"    [{i}] f1={pf[0]:.4e}, f2={pf[1]:.4e}")

    print(f"  最佳 theta: {result['best_theta']}")

    # 标量化测试
    f_test = np.array([0.01, 0.02])
    w_test = np.array([0.5, 0.5])
    z_star = np.array([0.0, 0.0])
    g_pbi = pbi_scalarization(f_test, w_test, z_star, theta=5.0)
    g_cheb = chebyshev_scalarization(f_test, w_test, z_star)
    print(f"\n  标量化测试:")
    print(f"    PBI: {g_pbi:.6f}")
    print(f"    Chebyshev: {g_cheb:.6f}")

    print(f"  [OK] MOEA/D 优化完成")


# ============================================================
# Step 13: 统计学分析 (种子: 1100_Exercise + 100_blood_pressure)
# ============================================================
def step13_statistics():
    separator("Step 13: 统计学敏感性分析")

    # 生成两组 "实验" 预测
    rng = np.random.RandomState(278)
    group1 = 0.005 + 0.002 * rng.randn(20)
    group2 = 0.008 + 0.003 * rng.randn(20)

    # Welch t 检验
    t_result = welch_t_test(group1, group2)
    print(f"  Welch t 检验:")
    print(f"    t 统计量: {t_result['t_statistic']:.4f}")
    print(f"    自由度: {t_result['df']:.1f}")
    print(f"    p 值: {t_result['p_value_approx']:.6f}")
    print(f"    显著 (α=0.05): {t_result['significant_005']}")

    # Clopper-Pearson 置信区间
    k, n = 15, 20
    ci = clopper_pearson_ci(k, n)
    print(f"\n  Clopper-Pearson 置信区间 ({k}/{n}):")
    print(f"    95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

    # Wilcoxon 检验
    diffs = group1 - group2
    w_result = wilcoxon_signed_rank_test(diffs)
    print(f"\n  Wilcoxon 符号秩检验:")
    print(f"    W+: {w_result['W_plus']:.1f}")
    print(f"    W-: {w_result['W_minus']:.1f}")
    print(f"    z 近似: {w_result['z_approx']:.4f}")
    print(f"    p 值: {w_result['p_value_approx']:.6f}")

    # Tipping-point 分析
    tp = tipping_point_analysis(1000.0, 'FCC', 'BCC',
                                 param_index=0, n_steps=10)
    print(f"\n  Tipping-point 分析 (L0_FCC):")
    print(f"    临界扰动: {tp['tipping_point']}")
    print(f"    收敛状态: {tp['convergence_status'][:5]}")

    print(f"  [OK] 统计学分析完成")


# ============================================================
# 主程序
# ============================================================
def main():
    """
    PROJECT 278: 计算材料 — Fe-C CALPHAD 建模与稳定性分析

    零参数入口: python main.py
    """
    print("=" * 70)
    print("  PROJECT 278: 计算材料 — 相图计算与 CALPHAD 建模")
    print("  高阶有限差分与稳定性分析 (小规模可复现实验)")
    print("=" * 70)
    print(f"\n  Fe-C 二元合金体系")
    print(f"  气体常数 R = {R_GAS} J/(mol·K)")
    print(f"  成分范围: [{X_C_MIN:.2e}, {X_C_MAX}]")
    print(f"  温度范围: [{T_MIN}, {T_MAX}] K")
    print(f"  随机种子: {MC_RANDOM_SEED}")

    t_start = time.time()

    # 执行所有步骤
    try:
        step1_cvt_mesh()
        step2_gibbs_energy()
        step3_piecewise_phase()
        step4_newton_equilibrium()
        step5_bifurcation()
        step6_cahn_hilliard()
        step7_stiff_ode()
        step8_lyapunov()
        step9_allen_cahn()
        step10_spiral_scan()
        step11_monte_carlo()
        step12_moea_optimizer()
        step13_statistics()
    except Exception as e:
        print(f"\n  [ERROR] 执行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    t_elapsed = time.time() - t_start

    separator("计算完成", '=')
    print(f"  总运行时间: {t_elapsed:.2f} 秒")
    print(f"  所有 13 个计算步骤均成功完成")
    print(f"  种子项目映射:")
    print(f"    689_linpack_d        → LU 分解 (Step 4)")
    print(f"    1085_border-bifurcations → 分岔检测 (Step 5)")
    print(f"    259_cvt_square       → CVT 网格 (Step 1)")
    print(f"    801_newton_maehly    → 相平衡 (Step 4)")
    print(f"    841_ozone_ode        → 刚性 ODE (Step 7)")
    print(f"    555_hyperball        → MC 采样 (Step 11)")
    print(f"    064_backward_euler   → 隐式积分 (Step 6)")
    print(f"    450_gamblers_ruin    → 参数存活 (Step 11)")
    print(f"    923_pwc_plot_1d      → 分段常数 (Step 3)")
    print(f"    1219_MOEA-D          → 参数优化 (Step 12)")
    print(f"    828_ode_midpoint     → 中点法 (Step 9)")
    print(f"    456_gaussian_prime   → 螺旋扫描 (Step 10)")
    print(f"    1204_lle-chaos       → Lyapunov (Step 8)")
    print(f"    100_blood_pressure   → 切换动力学 (Step 9)")
    print(f"    1100_Exercise        → 统计检验 (Step 13)")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
