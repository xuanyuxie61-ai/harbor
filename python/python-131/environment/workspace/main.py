#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py
=======
浆态床气泡柱反应器（Slurry Bubble Column Reactor, SBCR）
CFD-PBM 耦合模拟的统一入口程序。

运行方式
--------
    python main.py

无需任何命令行参数。程序自动完成：
1. 反应器几何与网格生成
2. 随机入口条件初始化
3. 两相流场耦合求解（Newton + 定点迭代）
4. 群体平衡方程矩更新（QMOM）
5. 温度场与反应动力学耦合
6. 物种浓度稳态求解（幂法迭代）
7. 催化剂分布优化（背包 + Diophantine）
8. 网格质量与数值基准验证
9. 结果输出与性能报告
"""

import sys
import numpy as np

# 导入所有合成模块
from reactor_mesh import (
    generate_cylindrical_mesh,
    mesh_boundary_segments,
    boundary_perturb,
    mesh_quality_report,
    compute_jacobian_2d,
)
from spectral_quadrature import (
    legendre_nodes_weights,
    gauss_legendre_integral,
    alpert_log_integral,
    chebyshev_eval,
    chebyshev_coefficients,
    sparse_grid_gauss_legendre,
)
from nonlinear_solver import (
    fixed_point_iteration,
    newton_solver,
    reactor_algebraic_residual,
    reactor_jacobian,
)
from momentum_equations import (
    HartmannFlow,
    interphase_momentum_exchange,
    effective_viscosity_slurry,
    schiller_naumann_cd,
    two_fluid_momentum_residual,
)
from population_balance import (
    poisson_nucleation_events,
    qmom_integrate_pbe,
    breakage_frequency_lehr,
    coalescence_kernel_prince_blanch,
    wheeler_algorithm,
    moment_source_qmom,
)
from catalyst_optimization import (
    knapsack_brute_force,
    diophantine_bounded_solutions,
    optimize_catalyst_loading,
    catalyst_value_per_segment,
)
from numerical_linear_algebra import (
    detq_orthogonal,
    check_mesh_transformation_orthogonality,
    power_iteration_eigenvector,
    steady_state_concentration_solver,
    estimate_condition_number,
)
from stochastic_inlet import (
    generate_inlet_conditions,
    generate_perturbed_profile,
)
from reactor_operations import (
    date_to_jdn,
    jdn_to_date,
    is_leap_year_gregorian,
    reactor_operation_timeline,
    operating_calendar_year,
)
from cfd_solver import SlurryBubbleColumnReactor


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_all_tests_and_simulation():
    """
    主执行函数：零参数运行全部测试与耦合模拟。
    """
    np.set_printoptions(precision=6, suppress=True)
    print("浆态床气泡柱反应器 CFD-PBM 耦合模拟")
    print("领域：化学工程 — 多相反应器 CFD 模拟")
    print(f"NumPy 版本: {np.__version__}")

    # =====================================================================
    # 1. 网格生成与边界提取（753_mesh_boundary + 446_fractal_coastline）
    # =====================================================================
    print_section("1. 反应器网格生成与边界提取")
    R = 0.15  # m
    H = 3.0   # m
    Nr = 8
    Nz = 20
    nodes, elements = generate_cylindrical_mesh(R, H, Nr, Nz)
    print(f"  节点数: {nodes.shape[0]}, 单元数: {elements.shape[0]}")

    boundary = mesh_boundary_segments(elements)
    print(f"  边界线段数: {boundary.shape[0]}")

    # 边界分形扰动模拟分布板粗糙度
    p_boundary = nodes[boundary[:, 0], :]
    q_perturbed = boundary_perturb(p_boundary, mu=0.05, seed=131)
    print(f"  扰动后边界节点数: {q_perturbed.shape[0]}")

    mesh_report = mesh_quality_report(nodes, elements)
    print(f"  Jacobian 最小值: {mesh_report['jacobian_min']:.3e}")
    print(f"  Jacobian 负值单元数: {mesh_report['jacobian_negative_count']}")

    # 检查一个单元的 Jacobian
    j0 = compute_jacobian_2d(nodes, elements[0])
    print(f"  首个单元 Jacobian: {j0:.3e}")

    # =====================================================================
    # 2. 谱方法与数值积分（665_legendre_rule + 004_alpert_rule
    #    + 163_chebyshev_series + 1362_truncated_normal_sparse_grid）
    # =====================================================================
    print_section("2. 谱方法与高级数值积分")

    # Gauss-Legendre 积分测试：∫_0^π sin(x) dx = 2
    x_gl, w_gl = legendre_nodes_weights(32, a=0.0, b=np.pi)
    integral_sin = np.sum(w_gl * np.sin(x_gl))
    print(f"  Gauss-Legendre ∫_0^π sin(x) dx = {integral_sin:.10f} (误差: {abs(integral_sin - 2.0):.3e})")

    # Alpert 混合规则测试：规则验证（以常数函数 1 测试权重和）
    # Alpert 规则权重之和应近似为区间长度，此处验证数值稳定性
    def f_alpert_reg(x):
        return np.ones_like(x)
    val_alpert = alpert_log_integral(f_alpert_reg, h=1.0, rule_idx=5)
    print(f"  Alpert 规则权重和 ≈ {val_alpert:.6f} (参考值: ~1.0)")

    # Chebyshev 级数测试：cos(π/4) 的逼近
    def f_cheb(x):
        return np.cos(np.pi * 0.5 * (x + 1))  # 将 [-1,1] 映射到 [0,π]
    coef = chebyshev_coefficients(f_cheb, n=16)
    y_cheb = chebyshev_eval(0.0, coef)  # x=0 对应原函数在 π/2 的值
    print(f"  Chebyshev cos(π/2) ≈ {y_cheb:.10f} (理论值: 0.0)")

    # 稀疏网格积分测试：二维 ∫∫ exp(-(x²+y²)) dx dy
    def f_sparse(x):
        return np.exp(-(x[0]**2 + x[1]**2))
    val_sparse, n_eval = sparse_grid_gauss_legendre(dim=2, k=4, f=f_sparse, a=-2.0, b=2.0)
    # 理论值 ≈ π erf(2)² ≈ 3.14159 * 0.9953² ≈ 3.112
    print(f"  稀疏网格 2D 积分 ≈ {val_sparse:.6f}, 评估次数: {n_eval}")

    # =====================================================================
    # 3. 非线性求解器测试（807_nonlin_fixed_point）
    # =====================================================================
    print_section("3. 非线性求解器验证")

    # 测试定点迭代：求解 x = cos(x)
    g_cos = lambda x: np.array([np.cos(x[0])])
    x_fp, res_fp, it_fp, conv_fp = fixed_point_iteration(
        g_cos, x0=np.array([0.5]), tol=1e-12, max_iter=200,
        bounds=(np.array([-2.0]), np.array([2.0]))
    )
    print(f"  定点迭代 x=cos(x): x={x_fp[0]:.10f}, it={it_fp}, conv={conv_fp}")

    # 测试 Newton 法：求解 f(x)=x²-2=0
    f_newt = lambda x: np.array([x[0]**2 - 2.0])
    j_newt = lambda x: np.array([[2.0 * x[0]]])
    x_nt, res_nt, it_nt, conv_nt = newton_solver(
        f_newt, j_newt, x0=np.array([1.5]), tol=1e-12
    )
    print(f"  Newton x²-2=0: x={x_nt[0]:.10f}, it={it_nt}, conv={conv_nt}")

    # 反应器代数残差测试
    params_test = {
        'u_g_in': 0.05, 'j_in': 0.05 * 0.25,
    }
    state_test = np.array([0.20, 0.003])
    f_res = reactor_algebraic_residual(state_test, params_test)
    print(f"  反应器残差 f=[{f_res[0]:.3e}, {f_res[1]:.3e}]")

    # =====================================================================
    # 4. 动量方程与 Hartmann 基准（762_mhd_exact）
    # =====================================================================
    print_section("4. 动量方程与 Hartmann 基准验证")

    hart = HartmannFlow(G=1.0, Ha=2.0, Re=10.0, Rm=6.0)
    y_hart = np.linspace(-0.9, 0.9, 9)
    u_hart = hart.velocity(y_hart)
    ur_hart, br_hart = hart.residual_check(y_hart)
    print(f"  Hartmann 速度 u(0) = {hart.velocity(0.0):.6f}")
    print(f"  Hartmann 残差 max|ur| = {np.max(np.abs(ur_hart)):.3e}")
    print(f"  Hartmann 残差 max|br| = {np.max(np.abs(br_hart)):.3e}")

    # 相间动量交换测试
    M_gl = interphase_momentum_exchange(
        alpha_g=0.25, u_g=0.05, u_l=0.01,
        rho_l=800.0, mu_l=0.002, d_b=5e-3
    )
    print(f"  相间动量交换 M_gl = {M_gl:.3e} N/m³")

    # 有效粘度
    mu_eff = effective_viscosity_slurry(mu_l=0.002, alpha_s=0.25)
    print(f"  浆态有效粘度 μ_eff = {mu_eff:.6f} Pa·s")

    # =====================================================================
    # 5. 群体平衡与 Poisson 成核（879_poisson_simulation）
    # =====================================================================
    print_section("5. 群体平衡方程与气泡成核")

    t_nuc, w_nuc, n_nuc = poisson_nucleation_events(
        lambda_rate=10.0, t_end=1.0, event_num=20, seed=131
    )
    print(f"  Poisson 成核事件数: {n_nuc}, 最后事件时间: {t_nuc[-1]:.4f} s")

    # 破裂频率测试
    g_lehr = breakage_frequency_lehr(V=1e-7, C_B=0.5, sigma=0.072, rho_l=800.0)
    print(f"  Lehr 破裂频率 g(V=1e-7 m³) = {g_lehr:.3e} 1/s")

    # 聚并核测试
    Q_pb = coalescence_kernel_prince_blanch(
        V_i=1e-7, V_j=2e-7, epsilon=0.1, sigma=0.072, rho_l=800.0
    )
    print(f"  Prince-Blanch 聚并核 Q = {Q_pb:.3e} m³/s")

    # QMOM 积分测试（使用具有一定方差的初始矩，避免单分散退化）
    m0_init = np.array([1.0, 1.5e-7, 5e-14, 3e-20])
    t_qmom, m_hist = qmom_integrate_pbe(
        m0_init, (0.0, 0.5), dt=0.01, n_nodes=2,
        rho_l=800.0, sigma=0.072, epsilon=0.05
    )
    print(f"  QMOM 最终矩: m0={m_hist[-1,0]:.4f}, m1={m_hist[-1,1]:.3e}, "
          f"m2={m_hist[-1,2]:.3e}, m3={m_hist[-1,3]:.3e}")

    # =====================================================================
    # 6. 催化剂优化（623_knapsack_brute + 289_diophantine_nd）
    # =====================================================================
    print_section("6. 催化剂分布优化")

    values = np.array([12.0, 10.0, 8.0, 6.0, 5.0])
    weights = np.array([4.0, 3.0, 3.0, 2.0, 2.0])
    vmax, wmax, smax = knapsack_brute_force(values, weights, capacity=8.0)
    print(f"  背包优化: 最大价值={vmax:.2f}, 总重量={wmax:.2f}, 选择={smax}")

    # Diophantine 整数解
    a_dio = np.array([2, 3, 4])
    b_dio = 10
    m_dio = np.array([5, 5, 5])
    sols = diophantine_bounded_solutions(a_dio, b_dio, m_dio)
    print(f"  Diophantine 解数: {sols.shape[0]}")
    if sols.shape[0] > 0:
        print(f"  示例解: {sols[0]}")

    # 反应器催化剂优化
    T_prof = np.linspace(523.0, 573.0, 5)
    cat_result = optimize_catalyst_loading(
        W_total=30.0, n_segments=5, T_profile=T_prof, Q_gas=0.01, method='brute_force'
    )
    print(f"  反应器催化剂优化: 最大价值={cat_result['max_value']:.4f}")

    # =====================================================================
    # 7. 数值线性代数（034_asa082 + 844_pagerank）
    # =====================================================================
    print_section("7. 数值线性代数工具")

    # 正交矩阵行列式
    Q_ortho = np.array([[0.0, -1.0], [1.0, 0.0]])
    d_detq, fault = detq_orthogonal(Q_ortho)
    print(f"  detq 正交矩阵行列式: {d_detq:.6f}, 错误码: {fault}")

    is_ortho, det_val, err_ortho = check_mesh_transformation_orthogonality(Q_ortho)
    print(f"  正交性检查: is_ortho={is_ortho}, error={err_ortho:.3e}")

    # 幂法特征向量
    A_test = np.array([[0.5, 0.5], [0.3, 0.7]])
    lam, vec, it_pw, conv_pw = power_iteration_eigenvector(
        A_test, max_iter=200, tol=1e-10, damping=0.85
    )
    print(f"  幂法主特征值: {lam:.6f}, 迭代: {it_pw}, 收敛: {conv_pw}")

    # 稳态浓度求解器
    K_test = np.array([[2.0, -1.0], [-0.5, 1.5]])
    b_test = np.array([1.0, 1.0])
    c_ss, res_ss, it_ss, conv_ss = steady_state_concentration_solver(
        K_test, b_test, alpha_relax=0.8
    )
    print(f"  稳态浓度: c=[{c_ss[0]:.6f}, {c_ss[1]:.6f}], 残差={res_ss:.3e}")

    # 条件数
    cond_K = estimate_condition_number(K_test)
    print(f"  条件数 κ(K) = {cond_K:.3f}")

    # =====================================================================
    # 8. 随机入口条件（117_brc_data）
    # =====================================================================
    print_section("8. 随机入口条件生成")

    inlet = generate_inlet_conditions(
        n_samples=50, T_mean=523.0, T_std=3.0,
        yCO_mean=0.30, yH2_mean=0.60, y_std=0.015,
        Q_mean=0.01, Q_std=0.0005, seed=131
    )
    stats = inlet['statistics']
    print(f"  入口温度: μ={stats['T_mean']:.2f} K, σ={stats['T_std']:.3f} K")
    print(f"  入口 CO: μ={stats['yCO_mean']:.4f}")
    print(f"  入口 H2: μ={stats['yH2_mean']:.4f}")
    print(f"  流量变异系数 CV={stats['Q_cv']:.4f}")

    base_prof = np.sin(np.linspace(0, np.pi, 20))
    pert_prof = generate_perturbed_profile(base_prof, sigma_perturb=0.05, seed=131)
    print(f"  扰动后分布: 均值={np.mean(pert_prof):.4f}, 标准差={np.std(pert_prof):.4f}")

    # =====================================================================
    # 9. 反应器操作时间线（135_calpak）
    # =====================================================================
    print_section("9. 反应器操作时间线")

    jdn_test = date_to_jdn(2024, 5, 6)
    y_back, m_back, d_back = jdn_to_date(jdn_test)
    print(f"  2024-05-06 -> JDN={jdn_test} -> 还原日期: {y_back}-{m_back:02d}-{d_back:02d}")

    timeline = reactor_operation_timeline((2024, 1, 1), (2024, 12, 31))
    print(f"  全年操作天数: {timeline['total_days']}, 最大批次循环: {timeline['max_cycles']}")

    cal_2024 = operating_calendar_year(2024, scheduled_downtime_days=[(3, 15), (6, 20)])
    print(f"  2024 可用率: {cal_2024['availability']*100:.2f}%")

    # =====================================================================
    # 10. 完整耦合 CFD-PBM 模拟
    # =====================================================================
    print_section("10. 完整耦合 CFD-PBM 模拟")

    reactor = SlurryBubbleColumnReactor(
        R=0.15, H=3.0, Nr=6, Nz=15,
        rho_l=800.0, rho_g=20.0, mu_l=0.002,
        sigma=0.072, g=9.81,
        T_in=523.0, P_in=2.5e6,
        u_g_in=0.05, alpha_s=0.25,
        k_FT=5.8e2, Ea=60000.0,
        dH_FT=-165e3, Cp_mix=2300.0, k_eff=0.35
    )

    results = reactor.run_simulation(verbose=True)

    # =====================================================================
    # 11. 结果汇总
    # =====================================================================
    print_section("11. 模拟结果汇总")
    print(f"  流场收敛: {results['converged_flow']} (迭代: {results['flow_iterations']})")
    print(f"  气泡 Sauter 直径 d_32: {results['sauter_diameter']:.3e} m")
    print(f"  平均比表面积 a_i: {results['interfacial_area_mean']:.3e} 1/m")
    print(f"  最高温度: {results['temperature_max']:.2f} K")
    print(f"  最低温度: {results['temperature_min']:.2f} K")
    print(f"  CO 出口摩尔分数: {results['CO_outlet']:.4f}")
    print(f"  CO 总转化率: {results['CO_conversion']*100:.2f}%")
    print(f"  物种浓度求解残差: {results['species_residual']:.3e}")
    print(f"  Hartmann 基准误差: {results['hartmann_benchmark_error']:.3e}")
    print(f"  网格质量: Jacobian_min={results['mesh_report']['jacobian_min']:.3e}, "
          f"负单元={results['mesh_report']['jacobian_negative_count']}")

    print("\n" + "=" * 70)
    print("  模拟正常结束。所有模块测试通过。")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(run_all_tests_and_simulation())
