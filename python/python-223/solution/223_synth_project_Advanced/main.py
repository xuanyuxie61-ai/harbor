"""
main.py — 计算高能物理: 喷注聚类与 jet substructure 分析
              高阶有限差分与稳定性分析 (小规模可复现实验)

统一入口, 零参数可运行.

融合 15 个种子项目的核心算法:
  849_partition_brute  → 动量分区暴力优化 (jet_fourvector.py)
  1233_eileenrmartin   → 互相关函数分析 (jet_substructure.py)
  964_r83p             → 四动量点集运算 (jet_fourvector.py)
  176_circle_arc_grid  → η-φ 圆弧网格 (jet_grid.py)
  843_padua            → Padua 正交节点 (jet_grid.py)
  793_nearest_neighbor → 最近邻聚类搜索 (jet_clustering.py)
  106_boundary_word    → 喷注边界词追踪 (jet_boundary.py)
  961_r8_scale         → 浮点邻域精度控制 (jet_fourvector.py, high_order_fd.py)
  1161_AmandaRosa      → 事件生成框架 (event_generator.py)
  208_conservation_ode → 守恒 ODE 验证 (conservation_laws.py)
  1131_cchrisgong      → 聚类树 merger 分析 (jet_substructure.py)
  1356_trig_interp     → 三角周期插值 (high_order_fd.py)
  192_closest_point    → 暴力最近点搜索 (jet_clustering.py)
  112_box_display      → 量热器网格区域 (jet_grid.py)
  1252_numerical-bypass→ 数值精度优化 (stability_analysis.py)

实验流程:
  1. 生成多喷注事件 (部分子簇射)
  2. 在 η-φ 量热器网格上沉积能量
  3. 使用 anti-kT 算法聚类喷注
  4. 计算 N-subjettiness, ECF, D₂ 等子结构观测量
  5. 高阶有限差分分析能量密度场
  6. Von Neumann 稳定性与 pT 阈值敏感性分析
  7. 守恒定律验证
  8. 汇总报告
"""

import sys
import os
import math
import numpy as np

# 确保模块路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jet_fourvector import (
    FourVector, random_fourvector, generate_multijet_event,
    partition_momentum_brute, mandelstam_s, mandelstam_t, mandelstam_u,
    mandelstam_check, r8_next, r8_previous, r8_bracket
)
from jet_grid import (
    CalorimeterGrid, padua_points, padua_weights, padua_to_jet_cone,
    circle_arc_grid, gauss_legendre_jet
)
from jet_clustering import (
    anti_kt, cambridge_aachen, kt_algorithm, brute_closest_pair,
    compute_distance_matrix, find_nearest_pair, recombine_E_scheme,
    ClusterTree
)
from jet_substructure import (
    n_subjettiness, n_subjettiness_ratio,
    energy_correlation_2, energy_correlation_3, D2_ratio,
    energy_density_profile, cross_correlation_2d,
    compute_mass_drop, pruning_mask
)
from jet_boundary import (
    BoundaryWord, circular_boundary_word, jet_area_monte_carlo,
    voronoi_cell_area, word_reflect, word_rotate
)
from high_order_fd import (
    fd_coefficients, fd_derivative_1d, fd_gradient_2d, fd_laplacian_2d,
    trig_interpolation, trig_interpolation_cardinal,
    richardson_extrapolation, fd_precision_scan, optimal_step_size
)
from stability_analysis import (
    von_neumann_amplification, cfl_condition, diffusion_stability_limit,
    cluster_matrix_condition_number, pT_threshold_sensitivity,
    clustering_stability_perturbation, conservation_drift_analysis,
    stability_report
)
from event_generator import (
    PartonShower, EventGenerator, pT_filter, eta_filter, isolation_filter,
    alpha_s, P_qq, P_gg
)
from conservation_laws import (
    total_four_momentum, invariant_mass, verify_momentum_conservation,
    compute_jet_observables, event_observables,
    momentum_sum_rule_check, simulate_conservation_ode
)


def separator(title: str):
    """打印分节标题."""
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


def main():
    """主函数: 零参数运行完整喷注分析流程."""
    print("╔" + "═" * 70 + "╗")
    print("║  计算高能物理: 喷注聚类与 jet substructure 分析                    ║")
    print("║  高阶有限差分与稳定性分析 — 小规模可复现实验                        ║")
    print("║  Computational HEP: Jet Clustering & Substructure                  ║")
    print("║  High-Order Finite Differences & Stability Analysis                 ║")
    print("╚" + "═" * 70 + "╝")

    rng_global = np.random.default_rng(42)

    # ════════════════════════════════════════════════════════════════════════
    # 阶段 1: 事件生成 (源自 1161 框架 + 208 ODE)
    # ════════════════════════════════════════════════════════════════════════
    separator("阶段 1: 多喷注事件生成")

    gen = EventGenerator(sqrt_s=13000.0, seed=42)
    event = gen._generate_multijet(n_jets=10)
    print(f"  生成 {len(event)} 个初始部分子")
    print(f"  质心系能量: √s = 13000 GeV")

    # 应用过滤器 (pT + η + 隔离)
    event_filtered = pT_filter(event, pT_min=15.0)
    event_filtered = eta_filter(event_filtered, eta_max=2.5)
    event_filtered = isolation_filter(event_filtered, delta_R_min=0.05)
    print(f"  过滤后: {len(event_filtered)} 个粒子 "
          f"(pT > 15 GeV, |η| < 2.5)")

    # 全局事件观测量
    evt_obs = event_observables(event_filtered)
    print(f"  H_T = {evt_obs['H_T']:.2f} GeV")
    print(f"  MET = {evt_obs['MET']:.4f} GeV")
    print(f"  不变质量 = {evt_obs['invariant_mass']:.2f} GeV")
    print(f"  球度 = {evt_obs['sphericity']:.4f}")
    print(f"  推力 = {evt_obs['thrust']:.4f}")

    # 部分子簇射模拟
    shower = PartonShower(Q_max=200.0, Q_min=1.0, seed=42)
    first_parton = event_filtered[0] if event_filtered else FourVector(100, 50, 50, 30)
    showered = shower.generate_shower(first_parton, max_depth=4)
    print(f"  部分子簇射: {first_parton.pT:.1f} GeV parton → "
          f"{len(showered)} particles, "
          f"{shower.n_splittings} splittings")
    print(f"  α_s(Q=200 GeV) = {alpha_s(200.0):.4f}")
    print(f"  α_s(Q=1 GeV)   = {alpha_s(1.0):.4f}")
    print(f"  P_qq(z=0.5) = {P_qq(0.5):.4f}, P_gg(z=0.5) = {P_gg(0.5):.4f}")

    # ════════════════════════════════════════════════════════════════════════
    # 阶段 2: 量热器网格与 Padua 节点
    # ════════════════════════════════════════════════════════════════════════
    separator("阶段 2: η-φ 量热器网格与能量沉积")

    grid = CalorimeterGrid(eta_min=-2.5, eta_max=2.5,
                           d_eta=0.2, d_phi=0.2)
    print(f"  网格: {grid.n_eta} × {grid.n_phi} = {grid.total_cells} cells")
    print(f"  Cell 尺寸: Δη={grid.d_eta}, Δφ={grid.d_phi:.4f}")

    for p in event_filtered:
        grid.deposit_energy(p.eta, p.phi, p.pT)
    total_E_dep = grid.get_total_energy()
    print(f"  总沉积能量: {total_E_dep:.2f} GeV")

    # Padua 节点
    for level in [2, 3, 4]:
        pts = padua_points(level)
        wts = padua_weights(level)
        print(f"  Padua level {level}: {len(pts)} nodes, "
              f"Σw = {np.sum(wts):.4f}")

    # 圆弧网格 (喷注边界采样)
    arc_pts = circle_arc_grid(n_arc=16, R=0.4, eta_c=0.0, phi_c=0.0,
                              n_radial=3)
    print(f"  圆弧网格: {len(arc_pts)} points "
          f"(16 angular × 3 radial)")

    # Gauss-Legendre 求积
    gl_nodes, gl_weights = gauss_legendre_jet(n_quad=5, R=0.4)
    print(f"  Gauss-Legendre 5点: nodes={gl_nodes[:3].round(4)}, ...")

    # ════════════════════════════════════════════════════════════════════════
    # 阶段 3: 喷注聚类 (anti-kT, C/A, kT)
    # ════════════════════════════════════════════════════════════════════════
    separator("阶段 3: 喷注聚类算法")

    R = 0.4
    result_ak = anti_kt(event_filtered, R=R, pT_min=5.0)
    result_ca = cambridge_aachen(event_filtered, R=R, pT_min=5.0)
    result_kt = kt_algorithm(event_filtered, R=R, pT_min=5.0)

    print(f"  Anti-kT (R={R}): {len(result_ak.jets)} jets, "
          f"{result_ak.n_steps} steps")
    print(f"  C/A     (R={R}): {len(result_ca.jets)} jets, "
          f"{result_ca.n_steps} steps")
    print(f"  kT      (R={R}): {len(result_kt.jets)} jets, "
          f"{result_kt.n_steps} steps")

    for i, jet in enumerate(result_ak.jets[:3]):
        print(f"    Jet {i+1}: pT={jet.pT:.2f} GeV, "
              f"η={jet.eta:.3f}, φ={jet.phi:.3f}, "
              f"m={jet.invariant_mass():.2f} GeV")

    # 暴力最近点 (源自 192)
    if len(event_filtered) >= 2:
        bi, bj, bdr = brute_closest_pair(event_filtered[:20])
        print(f"  暴力最近点对: ({bi},{bj}), ΔR={bdr:.4f}")

    # ════════════════════════════════════════════════════════════════════════
    # 阶段 4: 喷注子结构分析
    # ════════════════════════════════════════════════════════════════════════
    separator("阶段 4: Jet Substructure 观测量")

    if result_ak.jets:
        leading_jet = result_ak.jets[0]

        # 找出 leading jet 的 constituents
        jet_constituents = [p for p in event_filtered
                           if p.delta_R(leading_jet) < R]
        print(f"  Leading jet: pT={leading_jet.pT:.2f}, "
              f"{len(jet_constituents)} constituents")

        if len(jet_constituents) >= 2:
            # N-subjettiness
            axis1 = [leading_jet]
            axis2 = [leading_jet]  # 简化: 使用同一轴
            if len(jet_constituents) >= 2:
                # 两个最硬的 constituents 作为双轴
                sorted_c = sorted(jet_constituents, key=lambda c: -c.pT)
                axis2 = [sorted_c[0], sorted_c[1]]

            tau1 = n_subjettiness(jet_constituents, axis1, beta=1.0, R=R)
            tau2 = n_subjettiness(jet_constituents, axis2, beta=1.0, R=R)
            tau21 = tau2 / tau1 if tau1 > 1e-15 else 1.0

            print(f"  τ₁ = {tau1:.4f}")
            print(f"  τ₂ = {tau2:.4f}")
            print(f"  τ₂₁ = τ₂/τ₁ = {tau21:.4f}  "
                  f"(小值 → 2-prong)")

            # 能量关联函数
            e2 = energy_correlation_2(jet_constituents, alpha=1.0)
            e3 = energy_correlation_3(jet_constituents, alpha=1.0)
            d2 = D2_ratio(jet_constituents, alpha=1.0)
            print(f"  e₂(α=1) = {e2:.6f}")
            print(f"  e₃(α=1) = {e3:.8f}")
            print(f"  D₂ = {d2:.6f}  "
                  f"(quark~O(α_s), gluon~O(1))")

            # 能量密度分布
            r_centers, rho = energy_density_profile(
                jet_constituents, leading_jet, R=R, n_bins=10)
            print(f"  ρ(r) profile: {len(r_centers)} bins, "
                  f"peak = {np.max(rho):.4f}")

            # 互相关函数 (源自 1233)
            C = cross_correlation_2d(jet_constituents, leading_jet,
                                     R=R, n_bins_eta=5, n_bins_phi=5)
            print(f"  互相关 C(Δη,Δφ): shape={C.shape}, "
                  f"max={np.max(C):.6f}")

            # Soft Drop pruning
            mask = pruning_mask(jet_constituents, leading_jet,
                                z_cut=0.1, R_cut=0.2, R=R)
            n_kept = sum(mask)
            print(f"  Soft Drop: {n_kept}/{len(jet_constituents)} "
                  f"constituents kept (z_cut=0.1)")

    # ════════════════════════════════════════════════════════════════════════
    # 阶段 5: 喷注边界与面积
    # ════════════════════════════════════════════════════════════════════════
    separator("阶段 5: 喷注边界与面积分析")

    if result_ak.jets:
        leading_jet = result_ak.jets[0]
        jet_constituents = [p for p in event_filtered
                           if p.delta_R(leading_jet) < R]

        # 圆形边界词
        bw = circular_boundary_word(leading_jet.eta, leading_jet.phi,
                                    R=R, n_segments=16)
        print(f"  边界词: {len(bw.steps)} segments, "
              f"perimeter={bw.perimeter():.4f}")
        print(f"  边界面积: {bw.area():.4f}")
        closed = bw.is_closed(tol=0.5)
        print(f"  边界封闭: {closed}")

        # 反射对称性
        bw_reflected = word_reflect(bw, axis='eta')
        print(f"  η-反射后面积: {bw_reflected.area():.4f}")

        # Monte Carlo 面积
        if jet_constituents:
            area_mc = jet_area_monte_carlo(leading_jet, jet_constituents,
                                           R=R, n_samples=2000)
            print(f"  MC 喷注面积: {area_mc:.4f} "
                  f"(解析 πR²={math.pi * R ** 2:.4f})")

        # Voronoi cell 面积
        if jet_constituents and len(jet_constituents) <= 10:
            vor_areas = voronoi_cell_area(jet_constituents, leading_jet, R=R)
            print(f"  Voronoi cells: {len(vor_areas)} cells, "
                  f"total={sum(vor_areas):.4f}")

    # ════════════════════════════════════════════════════════════════════════
    # 阶段 6: 高阶有限差分分析
    # ════════════════════════════════════════════════════════════════════════
    separator("阶段 6: 高阶有限差分与精度分析")

    # FD 系数
    for order in [2, 4]:
        stencils, coeffs = fd_coefficients(1, order, 'central')
        print(f"  {order}阶精度一阶导数: stencil={stencils}, "
              f"coeffs={[f'{c:.4f}' for c in coeffs]}")

    # 一维 FD 测试 (能量密度径向分布的导数)
    test_x = np.linspace(0, 2 * math.pi, 50)
    test_y = np.sin(test_x) + 0.5 * np.cos(3 * test_x)
    dy_fd = fd_derivative_1d(test_y, test_x[1] - test_x[0],
                             derivative=1, accuracy=4, periodic=True)
    dy_exact = np.cos(test_x) - 1.5 * np.sin(3 * test_x)
    fd_error = np.max(np.abs(dy_fd - dy_exact))
    print(f"  1D FD 测试 (sin+cos): max error = {fd_error:.6e}")

    # 二维梯度 (能量密度场)
    if grid.total_cells > 4:
        df_deta, df_dphi = fd_gradient_2d(
            grid.energy_density, grid.d_eta, grid.d_phi,
            periodic_phi=True)
        print(f"  2D 梯度: |∇ε|_max = "
              f"{np.max(np.sqrt(df_deta**2 + df_dphi**2)):.6f}")
        lap = fd_laplacian_2d(grid.energy_density, grid.d_eta,
                              grid.d_phi, periodic_phi=True)
        print(f"  Laplacian: max|∇²ε| = {np.max(np.abs(lap)):.6f}")

    # 三角插值 (φ 方向)
    n_test = 12
    x_data = np.linspace(-math.pi, math.pi, n_test, endpoint=False)
    y_data = np.sin(2 * x_data) + 0.5 * np.cos(3 * x_data)
    x_eval = np.linspace(-math.pi, math.pi, 50)
    y_interp = trig_interpolation(x_data, y_data, x_eval)
    y_exact = np.sin(2 * x_eval) + 0.5 * np.cos(3 * x_eval)
    interp_error = np.max(np.abs(y_interp - y_exact))
    print(f"  三角插值 ({n_test} nodes): max error = {interp_error:.6e}")

    # Richardson 外推
    f_test = lambda x: math.sin(x)
    r_val, r_err = richardson_extrapolation(f_test, 1.0, 0.1,
                                            derivative=1, n_levels=4)
    exact_val = math.cos(1.0)
    print(f"  Richardson 外推: f'(1) ≈ {r_val:.10f}, "
          f"exact={exact_val:.10f}, error={abs(r_val - exact_val):.2e}")

    # 最佳步长
    h_opt = optimal_step_size(derivative=1, accuracy_order=2)
    print(f"  最佳步长 (2阶): h_opt = {h_opt:.2e}")
    h_opt4 = optimal_step_size(derivative=1, accuracy_order=4)
    print(f"  最佳步长 (4阶): h_opt = {h_opt4:.2e}")

    # 精度扫描
    h_scan, errors = fd_precision_scan(f_test, 1.0, derivative=1)
    best_idx = np.argmin(errors)
    print(f"  精度扫描: {len(h_scan)} points, "
          f"best h={h_scan[best_idx]:.2e}, "
          f"min error={errors[best_idx]:.2e}")

    # ════════════════════════════════════════════════════════════════════════
    # 阶段 7: 稳定性分析
    # ════════════════════════════════════════════════════════════════════════
    separator("阶段 7: 数值稳定性分析")

    # Von Neumann 分析
    k_dx = np.linspace(0, 2 * math.pi, 100)
    for scheme in ['FTCS', 'Lax-Friedrichs', 'Lax-Wendroff', 'Upwind',
                   'RK4-FD4']:
        G = von_neumann_amplification(scheme, k_dx, cfl=0.5)
        stable = np.all(G <= 1.0 + 1e-10)
        print(f"  {scheme:15s}: |G|_max = {np.max(G):.6f}, "
              f"stable={stable}")

    # CFL 条件
    max_v = max((p.pT for p in event_filtered), default=100.0)
    dt_cfl = cfl_condition(max_v, 0.1)
    print(f"  CFL: Δt_max = {dt_cfl:.6e} (v_max={max_v:.1f}, dx=0.1)")
    dt_diff = diffusion_stability_limit(0.1)
    print(f"  扩散稳定性: Δt_max = {dt_diff:.4f} (dx=0.1, ν=1)")

    # 距离矩阵条件数
    if len(event_filtered) >= 2:
        cond_info = cluster_matrix_condition_number(
            event_filtered[:15], R=R)
        print(f"  聚类矩阵条件数: κ = {cond_info['condition_number']:.4e}")
        print(f"    σ_max = {cond_info['sigma_max']:.4e}, "
              f"σ_min = {cond_info['sigma_min']:.4e}")

    # pT 阈值敏感性
    if len(event_filtered) >= 3:
        pT_vals, n_j, sens = pT_threshold_sensitivity(
            event_filtered, R=R, n_scan=20,
            pT_range=(5.0, 100.0))
        print(f"  pT 阈值扫描: {len(pT_vals)} points, "
              f"jets range [{n_j.min():.0f}, {n_j.max():.0f}]")

    # 微扰稳定性
    if len(event_filtered) >= 3:
        pert = clustering_stability_perturbation(
            event_filtered, R=R, n_perturbations=10,
            perturbation_scale=1e-5)
        print(f"  微扰稳定性 (δ={pert['perturbation_scale']:.1e}):")
        print(f"    ΔN_jet = {pert['mean_delta_n_jets']:.2f} "
              f"(max {pert['max_delta_n_jets']:.0f})")
        print(f"    ΔΣpT  = {pert['mean_delta_pT']:.4f} GeV "
              f"(max {pert['max_delta_pT']:.4f})")

    # 完整稳定性报告
    if len(event_filtered) >= 2:
        report = stability_report(event_filtered[:10], R=R)
        print(f"  稳定性报告汇总:")
        print(f"    条件数: {report['condition_number']['condition_number']:.4e}")
        print(f"    CFL Δt_max: {report['cfl_max_dt']:.6e}")

    # ════════════════════════════════════════════════════════════════════════
    # 阶段 8: 守恒定律验证
    # ════════════════════════════════════════════════════════════════════════
    separator("阶段 8: 守恒定律验证")

    # 聚类前后守恒
    if result_ak.jets:
        # 重组所有喷注 + 未聚类粒子
        all_after = list(result_ak.jets)
        cons_check = verify_momentum_conservation(event_filtered, all_after,
                                                  tolerance=0.5)
        print(f"  聚类守恒 (容差 50%):")
        print(f"    ΔE_rel = {cons_check['relative_error_E']:.4f}")
        print(f"    Δp_rel = {cons_check['relative_error_p']:.4f}")
        print(f"    passed = {cons_check['passed']}")

    # 动量求和规则
    if event_filtered:
        pT_fracs = [p.pT for p in event_filtered]
        msr = momentum_sum_rule_check(pT_fracs)
        print(f"  DGLAP 动量求和: Σ = {msr['sum']:.4f}, "
              f"deviation = {msr['deviation']:.4f}, "
              f"passed = {msr['passed']}")

    # 守恒 ODE (摆锤 + 刚性转子)
    ode_result = simulate_conservation_ode(
        np.array([0.5, 0.0]),
        t_span=(0, 10), n_steps=500, method='rk4'
    )
    print(f"  守恒 ODE (RK4, 500 steps):")
    print(f"    摆锤能量漂移: {ode_result['pendulum_energy_drift']:.2e}")
    print(f"    转子 |L|² 漂移: {ode_result['rigid_body_L_drift']:.2e}")
    print(f"    转子 E 漂移:  {ode_result['rigid_body_E_drift']:.2e}")

    # ════════════════════════════════════════════════════════════════════════
    # 阶段 9: Mandelstam 变量与分区优化
    # ════════════════════════════════════════════════════════════════════════
    separator("阶段 9: Mandelstam 变量与动量分区")

    if len(event_filtered) >= 4:
        p1, p2, p3, p4 = event_filtered[:4]
        s = mandelstam_s(p1, p2)
        t = mandelstam_t(p1, p3)
        u = mandelstam_u(p1, p4)
        m1, m2, m3, m4 = (p1.m, p2.m, p3.m, p4.m)
        check = mandelstam_check(s, t, u, m1, m2, m3, m4)
        print(f"  Mandelstam 变量:")
        print(f"    s = {s:.2f} GeV²")
        print(f"    t = {t:.2f} GeV²")
        print(f"    u = {u:.2f} GeV²")
        print(f"    s+t+u = {s+t+u:.2f}, Σm² = {m1**2+m2**2+m3**2+m4**2:.2f}")
        print(f"    |s+t+u - Σm²| = {check:.4f}")

    # 动量分区 (源自 849)
    if len(event_filtered) >= 4 and len(event_filtered) <= 15:
        part = partition_momentum_brute(event_filtered[:12], R=R)
        print(f"  动量分区 (暴力搜索):")
        print(f"    discrepancy = {part['discrepancy']:.4f} GeV")
        print(f"    cost = {part['cost']:.6f}")
        n0 = sum(1 for x in part['partition'] if x == 0)
        n1 = sum(1 for x in part['partition'] if x == 1)
        print(f"    划分: {n0} vs {n1} particles")

    # 浮点邻域测试 (源自 961)
    pT_test = 25.0
    bracket = r8_bracket(pT_test, n_eps=3)
    print(f"  浮点邻域 (pT={pT_test}):")
    print(f"    [{bracket[0]:.20f}, {bracket[1]:.20f}, {bracket[2]:.20f}]")
    print(f"    宽度: {bracket[2]-bracket[0]:.2e}")

    # ════════════════════════════════════════════════════════════════════════
    # 阶段 10: 喷注观测量汇总
    # ════════════════════════════════════════════════════════════════════════
    separator("阶段 10: 喷注观测量汇总")

    for i, jet in enumerate(result_ak.jets[:3]):
        constituents = [p for p in event_filtered if p.delta_R(jet) < R]
        obs = compute_jet_observables(jet, constituents)
        print(f"  Jet {i+1}:")
        print(f"    pT = {obs['pT']:.2f} GeV, η = {obs['eta']:.3f}, "
              f"φ = {obs['phi']:.3f}")
        print(f"    mass = {obs['mass']:.2f} GeV, "
              f"n_const = {obs['n_constituents']}")
        print(f"    width = {obs['width']:.4f}, "
              f"pT_balance = {obs['pT_balance']:.4f}")

    # ════════════════════════════════════════════════════════════════════════
    # 总结
    # ════════════════════════════════════════════════════════════════════════
    separator("合成项目总结")

    print("  本项目融合 15 个种子项目的核心算法:")
    print("    1.  849_partition_brute  → 暴力动量分区优化")
    print("    2.  1233_passive-DAS     → 互相关函数分析")
    print("    3.  964_r83p             → 四动量点集代数运算")
    print("    4.  176_circle_arc_grid  → η-φ 圆弧网格生成")
    print("    5.  843_padua            → Padua 正交插值节点")
    print("    6.  793_nearest_neighbor → 最近邻聚类搜索")
    print("    7.  106_boundary_word    → 喷注边界词追踪")
    print("    8.  961_r8_scale         → 浮点邻域精度控制")
    print("    9.  1161_AmandaRosa      → 事件生成框架")
    print("    10. 208_conservation_ode → 守恒 ODE 验证")
    print("    11. 1131_rockstar        → 聚类树 merger 分析")
    print("    12. 1356_trig_interp     → 三角周期插值")
    print("    13. 192_closest_point    → 暴力最近点搜索")
    print("    14. 112_box_display      → 量热器网格区域逻辑")
    print("    15. 1252_numerical-bypass→ 数值精度优化")
    print()
    print("  核心物理:")
    print("    - Anti-kT / C/A / kT 喷注聚类算法")
    print("    - N-subjettiness (τ_N), ECF (e2, e3), D₂ 子结构观测量")
    print("    - 高阶有限差分 (O(h⁴)) + Richardson 外推")
    print("    - Von Neumann 稳定性分析 + CFL 条件")
    print("    - DGLAP 部分子簇射 + Altarelli-Parisi splitting functions")
    print("    - 守恒 ODE 数值验证 (摆锤 + 刚性转子)")
    print()
    print("  文件结构:")
    for f in sorted(os.listdir(os.path.dirname(os.path.abspath(__file__)))):
        if f.endswith('.py'):
            size = os.path.getsize(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), f))
            print(f"    {f:30s} ({size:>6d} bytes)")

    print()
    print("  运行完成 ✓")


if __name__ == '__main__':
    main()
