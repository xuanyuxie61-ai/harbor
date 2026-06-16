# -*- coding: utf-8 -*-
"""
main.py
=======

PartonShowerHD 项目统一入口:
    高阶有限差分稳定性分析下的 Parton Shower 与强子化模型
    (小规模可复现实验)

本入口零参数运行, 完成以下完整流程:

1) 物理参数初始化 (constants)
2) DGLAP 分裂核多项式表示与投影 (splitting_kernels)
3) 有限差分算子构造与 Poisson 求解 (fd_operators)
4) Parton shower 守恒 ODE 演化 (parton_cascade)
5) 快度 Walsh 变换与扩散输运 (rapidity_transport)
6) 色流图 Dijkstra 与色 Laplacian (color_flow_graph)
7) Lund 弦强子化与通量管 Poisson (hadronization_string)
8) 量子数守恒 RREF 检查 (quantum_constraints)
9) 碎裂函数二次优化拟合 (fragmentation_optimizer)
10) von Neumann 稳定性与 Hankel-Cholesky (stability_analysis)
11) 相空间网格质量评估 (phase_space_quality)
12) Taylor-Green 流体矩基准验证 (fluid_moments)

=====================================================
"""

from __future__ import annotations
import math
import sys
import os

# 将项目目录加入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants import (
    C_F, C_A, T_R, N_C, N_F, PI,
    safe_alpha_s, alpha_s_one_loop, beta0_nf,
    M_PION, M_RHO, M_PROTON,
    Q2_MIN, Q2_MAX, A_LUND, B_LUND, SIGMA_STRING,
)
from splitting_kernels import (
    P_qq_lo, P_qg_lo, P_gq_lo, P_gg_lo,
    split_kernel_polynomial_projection,
    integral_P_qq, integral_P_gg,
    virtual_coefficient_qq, virtual_coefficient_gg,
    Poly1D,
)
from fd_operators import (
    diff_center_matrix, laplacian_1d_dd, laplacian_1d_nn,
    solve_fd1d_steady, solve_poisson_3d_cg,
    kronecker_sum,
)
from parton_cascade import (
    make_initial_state, evolve_shower_rk4,
    shower_conserved_quantity, background_cost,
    observation_cost, variational_assimilation_step,
)
from rapidity_transport import (
    fast_walsh_transform, inverse_walsh_transform,
    rapidity_density_initial, evolve_rapidity_diffusion,
    walsh_spectral_cutoff, walsh_energy_spectrum,
)
from color_flow_graph import (
    make_qqbar_event, ColorFlowGraph, ColorNode,
    dijkstra_min_distance, color_laplacian, fiedler_value,
    string_fragmentation_probability,
)
from hadronization_string import (
    lund_fragmentation_function, fragmentation_moment,
    solve_flux_tube_potential, flux_tube_energy_density,
    fragmentation_fixed_point_iteration,
    string_fragmentation_iterative,
)
from quantum_constraints import (
    check_quantum_conservation, i4mat_rref2,
    QUARK_QUANTUM_NUMBERS, HADRON_QUANTUM_NUMBERS,
)
from fragmentation_optimizer import (
    quadratic_interpolation_minimize, powell_direction_set,
    fragmentation_chi2, generate_mock_fragmentation_data,
    estimate_hessian,
)
from stability_analysis import (
    von_neumann_stability_check, critical_cfl_number,
    hankel_cholesky_upper, hankel_spd_check,
    laplacian_spectrum, condition_number_laplacian,
    dglap_max_dt_explicit,
)
from phase_space_quality import (
    phase_space_mesh_quality_2d, alpha_measure_triangle,
    beta_measure_triangle, gamma_measure_triangle,
)
from fluid_moments import (
    taylor_green_velocity, taylor_green_pressure,
    taylor_green_residual, energy_density_moment,
    pressure_moment, entropy_density_estimate,
    parton_mean_free_path, knudsen_number,
)


# ======================================================================
# 辅助打印
# ======================================================================
def section(title: str) -> None:
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def subsection(title: str) -> None:
    print(f"\n--- {title} ---")


# ======================================================================
# 主流程
# ======================================================================
def main() -> None:
    print()
    print("*" * 60)
    print("  PartonShowerHD v1.0")
    print("  高阶有限差分稳定性分析下的 Parton Shower 与强子化")
    print("  计算高能物理 · 博士级科学计算合成项目")
    print("*" * 60)

    # ==================================================================
    # Stage 1: 物理参数初始化
    # ==================================================================
    section("Stage 1: 物理参数初始化")

    print(f"  N_c = {N_C},  C_F = {C_F:.6f},  C_A = {C_A:.1f},  T_R = {T_R:.1f}")
    print(f"  beta_0(nf=5) = {beta0_nf(5):.6f}")

    q_values = [2.0, 5.0, 10.0, 50.0, 91.2, 500.0]
    print("\n  跑动耦合常数 alpha_s(mu):")
    for q in q_values:
        als = safe_alpha_s(q)
        als_1l = alpha_s_one_loop(q, 5)
        print(f"    mu = {q:6.1f} GeV:  alpha_s(2-loop) = {als:.6f},  (1-loop) = {als_1l:.6f}")

    # ==================================================================
    # Stage 2: DGLAP 分裂核多项式表示
    # ==================================================================
    section("Stage 2: DGLAP 分裂核多项式投影")

    for name in ['Pqq', 'Pqg', 'Pgq', 'Pgg']:
        poly = split_kernel_polynomial_projection(name, order=5)
        print(f"  {name}(z) ~ {poly}")

    subsection("数值积分")
    int_qq = integral_P_qq()
    int_gg = integral_P_gg()
    virt_qq = virtual_coefficient_qq()
    virt_gg = virtual_coefficient_gg(5)
    print(f"  int_0^1 P_qq dz = {int_qq:.6f}")
    print(f"  int_0^1 P_gg dz = {int_gg:.6f}")
    print(f"  virtual qq coeff = {virt_qq:.6f}")
    print(f"  virtual gg coeff = {virt_gg:.6f}")

    # ==================================================================
    # Stage 3: 有限差分算子与 Poisson 求解
    # ==================================================================
    section("Stage 3: 有限差分算子构造")

    n_fd = 6
    h_fd = 0.1
    D1_2 = diff_center_matrix(n_fd, h_fd, order=2)
    D1_4 = diff_center_matrix(n_fd, h_fd, order=4)
    print(f"  2阶中心差分 D1 (n={n_fd}, h={h_fd}):")
    for row in D1_2[:3]:
        print(f"    [{', '.join(f'{v:7.3f}' for v in row)}]")

    print(f"\n  4阶中心差分 D1 (n={n_fd}, h={h_fd}):")
    for row in D1_4[:3]:
        print(f"    [{', '.join(f'{v:8.4f}' for v in row)}]")

    L1dd = laplacian_1d_dd(n_fd, h_fd)
    print(f"\n  1D Laplacian (DD), n={n_fd}:")
    for row in L1dd:
        print(f"    [{', '.join(f'{v:7.3f}' for v in row)}]")

    subsection("一维稳态扩散方程")
    print("  -u''(x) = 1, u(0) = u(1) = 0, x in [0, 1]")
    x_grid, u_sol = solve_fd1d_steady(30, 0.0, 1.0, 0.0, 0.0,
                                        lambda x: 1.0, lambda x: 1.0)
    err = max(abs(u_sol[i] - 0.5 * x_grid[i] * (1.0 - x_grid[i]))
              for i in range(len(x_grid)))
    print(f"  最大误差 (vs 解析解): {err:.2e}")

    subsection("三维 Poisson 求解 (通量管)")
    U3d, X, Y, Z = solve_flux_tube_potential(n_grid=3, tube_radius=0.5, tube_length=2.0)
    max_phi = max(U3d[ix][iy][iz]
                  for ix in range(len(X))
                  for iy in range(len(Y))
                  for iz in range(len(Z)))
    hx = X[1] - X[0] if len(X) > 1 else 0.1
    hy = Y[1] - Y[0] if len(Y) > 1 else 0.1
    hz = Z[1] - Z[0] if len(Z) > 1 else 0.1
    E_flux = flux_tube_energy_density(U3d, X, Y, Z, hx, hy, hz)
    print(f"  3D Poisson (3x3x3 grid):")
    print(f"    Max potential Phi = {max_phi:.6f}")
    print(f"    Field energy E = {E_flux:.6f}")

    # ==================================================================
    # Stage 4: Parton Shower 演化与守恒律
    # ==================================================================
    section("Stage 4: Parton Shower ODE 演化")

    s0 = make_initial_state(nx=10, Q2=10.0)
    m0 = shower_conserved_quantity(s0)
    n0 = s0.parton_number()
    print(f"  初始状态: nx={s0.n}, momentum sum = {m0:.6f}, parton number = {n0:.4f}")

    trajectory = evolve_shower_rk4(s0, dt=0.05, n_steps=5)
    print(f"\n  RK4 演化 (dt=0.05, 5 步):")
    for k, sk in enumerate(trajectory):
        mk = shower_conserved_quantity(sk)
        nk = sk.parton_number()
        print(f"    Step {k}: <x> = {mk:.6f},  N = {nk:.4f},  "
              f"DeltaM = {abs(mk - m0):.2e}")

    subsection("变分数据同化 (简化 4D-Var)")
    obs_x = [0.2, 0.4, 0.6, 0.8]
    obs_val = [2.0, 1.5, 1.0, 0.5]
    B_inv = [1.0] * s0.n
    R_inv = 10.0
    cost_init = observation_cost(trajectory[-1], obs_x, obs_val, R_inv)
    print(f"  初始观测代价 J_o = {cost_init:.6f}")
    print(f"  (4D-Var 优化初始条件需 50 次迭代, 此处仅演示接口)")

    # ==================================================================
    # Stage 5: 快度 Walsh 变换与扩散
    # ==================================================================
    section("Stage 5: 快度 Walsh 变换输运")

    ny = 16
    y_grid = [-4.0 + 8.0 * i / (ny - 1) for i in range(ny)]
    rho0 = rapidity_density_initial(y_grid)
    print(f"  初始快度密度: {ny} 点, integral = {sum(rho0) * 8.0/ny:.4f}")

    # Walsh 变换
    spec = fast_walsh_transform(rho0)
    back = inverse_walsh_transform(spec)
    rt_err = max(abs(rho0[i] - back[i]) for i in range(ny))
    print(f"  Walsh 变换往返误差: {rt_err:.2e}")

    # Walsh 能量谱
    energy_spec = walsh_energy_spectrum(spec)
    total_E = sum(e for _, e in energy_spec)
    print(f"  Walsh 总能量: {total_E:.4f}")

    # 快度扩散
    rho_traj = evolve_rapidity_diffusion(rho0, diffusivity=0.5, dY=0.1, n_steps=3)
    print(f"\n  快度扩散 (D=0.5, 3 步):")
    for k, rho_k in enumerate(rho_traj):
        integral = sum(rho_k) * 8.0 / ny
        print(f"    Step {k}: integral = {integral:.6f}")

    # 谱截断
    spec_cut = walsh_spectral_cutoff(spec, max_sequency=4)
    rho_cut = inverse_walsh_transform(spec_cut)
    print(f"  谱截断 (seq<=4) 能量: {sum(w*w for w in spec_cut):.4f}")

    # ==================================================================
    # Stage 6: 色流图与 Dijkstra
    # ==================================================================
    section("Stage 6: 色流图与 Dijkstra 最短弦配置")

    graph = make_qqbar_event(n_gluons=4)
    print(f"  e+e- -> q qbar + 4 gluons: {graph.nv} 色荷节点")

    strings = graph.minimal_string_configuration()
    L_total = graph.total_string_length(strings)
    print(f"  最小弦配置: {len(strings)} 条弦")
    print(f"    弦对: {strings}")
    print(f"    总弦长 L = {L_total:.4f} (DeltaR)")

    w_prob = string_fragmentation_probability(L_total, len(strings))
    print(f"    Lund 碎裂概率 W ~ {w_prob:.4e}")

    L_color = color_laplacian(graph)
    fv = fiedler_value(L_color)
    print(f"  色 Laplacian Fiedler 值: {fv:.6f}")

    subsection("Dijkstra 最短路径")
    nv_dijk = 5
    ohd = [[10**9]*nv_dijk for _ in range(nv_dijk)]
    ohd[0][1] = 3; ohd[0][2] = 5
    ohd[1][2] = 2; ohd[1][3] = 6
    ohd[2][3] = 1; ohd[2][4] = 4
    ohd[3][4] = 2
    for i in range(nv_dijk):
        ohd[i][i] = 0
    mind = dijkstra_min_distance(nv_dijk, ohd)
    print(f"  Dijkstra (5 nodes): min distances from 0 = {mind}")

    # ==================================================================
    # Stage 7: 弦强子化
    # ==================================================================
    section("Stage 7: Lund 弦强子化")

    print("  Lund 碎裂函数 f(z):")
    for z in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        f = lund_fragmentation_function(z)
        print(f"    z = {z:.1f}: f(z) = {f:.6f}")

    subsection("碎裂函数矩")
    for n in range(5):
        mn = fragmentation_moment(n)
        print(f"    <z^{n}> = {mn:.6f}")

    subsection("自洽碎裂不动点迭代")
    z_grid_fp = [0.05 + 0.9 * i / 14 for i in range(15)]
    D0 = lambda z: lund_fragmentation_function(z)
    kernel = lambda y: 0.5 * (1.0 + (1.0-y)**2)
    D_fp = fragmentation_fixed_point_iteration(D0, kernel, z_grid_fp, max_iter=15)
    print(f"  收敛后 D[0] = {D_fp[0]:.4e}, D[-1] = {D_fp[-1]:.4e}")

    subsection("弦碎裂产生强子")
    hadrons = string_fragmentation_iterative(string_mass=15.0)
    print(f"  弦质量 15 GeV, 产生 {len(hadrons)} 个强子:")
    for h in hadrons[:5]:
        print(f"    rank {h['rank']}: {h['flavor']}, z={h['z']:.3f}")

    # ==================================================================
    # Stage 8: 量子数守恒
    # ==================================================================
    section("Stage 8: 量子数守恒 RREF 检查")

    test_cases = [
        (['u', 'ubar'], ['pi+', 'pi-'], "e+e- -> u ubar -> pi+ pi-"),
        (['u', 'd', 'd'], ['p', 'pi-'], "udd -> p pi- (baryon)"),
        (['s', 'sbar'], ['K+', 'K-'], "s sbar -> K+ K-"),
        (['u', 'sbar'], ['K+'], "u sbar -> K+ (strangeness)"),
    ]
    for init, final, desc in test_cases:
        ok, res = check_quantum_conservation(init, final)
        tag = "OK" if ok else "FAIL"
        print(f"  [{tag}] {desc}")
        if not ok:
            print(f"       residuals = {[f'{r:.3f}' for r in res]}")

    subsection("整数矩阵 RREF")
    A_test = [[2, 1, -1, 8],
              [-3, -1, 2, -11],
              [-2, 1, 2, -3]]
    Ar, rank = i4mat_rref2(3, 0, 3, A_test)
    print(f"  3x4 增广矩阵, rank = {rank}")
    for row in Ar:
        print(f"    {row}")

    # ==================================================================
    # Stage 9: 碎裂函数优化拟合
    # ==================================================================
    section("Stage 9: 碎裂函数二次优化拟合")

    z_data, d_data, d_sigma = generate_mock_fragmentation_data(
        n_points=8, true_a=0.3, true_b=0.8)
    print(f"  生成 {len(z_data)} 个模拟数据点 (true a=0.3, b=0.8)")

    def obj(params):
        return fragmentation_chi2(params, z_data, d_data, d_sigma)

    x_opt, it = powell_direction_set(obj, [0.5, 1.0], max_iter=15)
    chi2_min = obj(x_opt)
    print(f"  Powell 优化结果:")
    print(f"    a = {x_opt[0]:.4f} (true 0.3000)")
    print(f"    b = {x_opt[1]:.4f} (true 0.8000)")
    print(f"    chi^2_min = {chi2_min:.4f}, iterations = {it}")

    # Hessian
    H = estimate_hessian(obj, x_opt)
    print(f"  Hessian 对角元: H_aa = {H[0][0]:.2f}, H_bb = {H[1][1]:.2f}")

    # 一维二次插值
    f_1d = lambda a: obj([a, 0.8])
    a_min, it1d = quadratic_interpolation_minimize(f_1d, 0.1, 0.3, 0.8)
    print(f"\n  一维固定 b=0.8 下 a 的最优值: {a_min:.4f} ({it1d} 迭代)")

    # ==================================================================
    # Stage 10: 稳定性分析
    # ==================================================================
    section("Stage 10: Von Neumann 稳定性与 Hankel-Cholesky")

    print("  临界 CFL 数 (有限差分稳定性):")
    for order in [2, 4, 6]:
        r_crit = critical_cfl_number(order)
        stable, rho = von_neumann_stability_check(order, r_crit * 0.95)
        print(f"    差分阶 {order}: r_critical = {r_crit:.6f},  "
              f"at 0.95 r_crit: rho = {rho:.6f}, stable = {stable}")

    subsection("DGLAP 时间步长限制")
    als = safe_alpha_s(10.0)
    dt_max = dglap_max_dt_explicit(als)
    print(f"  alpha_s(Q=10 GeV) = {als:.6f}")
    print(f"  Explicit Euler 最大步长 dt_max = {dt_max:.6f}")

    subsection("Hankel-Cholesky 分解")
    n_hank = 5
    moments_test = [1.0 / (k + 1) for k in range(2 * n_hank - 1)]
    R_hank = hankel_cholesky_upper(n_hank, moments_test)
    print(f"  矩序列: M_k = 1/(k+1), n = {n_hank}")
    print(f"  Cholesky 对角元:")
    for i in range(n_hank):
        print(f"    R[{i},{i}] = {R_hank[i][i]:.6f}")

    is_spd, _ = hankel_spd_check(moments_test)
    print(f"  正定性: {is_spd}")

    subsection("Laplacian 谱与条件数")
    for n in [5, 10, 20]:
        h = 0.1
        eigs = laplacian_spectrum(n, h)
        cond = condition_number_laplacian(n, h)
        print(f"    n={n}: lambda_min = {eigs[0]:.2f}, lambda_max = {eigs[-1]:.2f}, "
              f"cond = {cond:.2f}")

    # ==================================================================
    # Stage 11: 相空间质量
    # ==================================================================
    section("Stage 11: 相空间网格质量评估")

    print("  三角形质量度量基准:")
    p1 = [0.0, 0.0]
    p2 = [1.0, 0.0]
    p3_eq = [0.5, math.sqrt(3)/2]
    p3_deg = [0.5, 0.001]
    print(f"    等边: alpha={alpha_measure_triangle(p1,p2,p3_eq):.4f}, "
          f"beta={beta_measure_triangle(p1,p2,p3_eq):.4f}, "
          f"gamma={gamma_measure_triangle(p1,p2,p3_eq):.4f}")
    print(f"    退化: alpha={alpha_measure_triangle(p1,p2,p3_deg):.4f}, "
          f"beta={beta_measure_triangle(p1,p2,p3_deg):.4f}")

    subsection("相空间网格质量 (x, Q^2)")
    x_grid_ps = [0.01 + 0.98 * i / 9 for i in range(10)]
    q2_grid_ps = [1.0 + 99.0 * i / 9 for i in range(10)]
    quality = phase_space_mesh_quality_2d(x_grid_ps, q2_grid_ps)
    print(f"  网格: 10x10 (x, Q^2)")
    print(f"    单元数: {quality['n_cells']}")
    print(f"    alpha_min = {quality['alpha_min']:.4f}")
    print(f"    beta_min  = {quality['beta_min']:.4f}")
    print(f"    gamma_min = {quality['gamma_min']:.4f}")
    print(f"    overall   = {quality['overall']:.4f}")

    # ==================================================================
    # Stage 12: Taylor-Green 流体基准
    # ==================================================================
    section("Stage 12: Taylor-Green 涡旋流体基准")

    nu = 0.01
    rho = 1.0
    t_test = 0.5
    x_test, y_test = 1.0, 1.0

    u, v = taylor_green_velocity(nu, rho, x_test, y_test, t_test)
    p_tg = taylor_green_pressure(nu, rho, x_test, y_test, t_test)
    print(f"  Taylor-Green 涡旋 (nu={nu}, t={t_test}):")
    print(f"    u = {u:.6f}, v = {v:.6f}")
    print(f"    p = {p_tg:.6f}")

    R_u, R_v = taylor_green_residual(nu, rho, x_test, y_test, t_test, h=0.001)
    print(f"  NS 残差: R_u = {R_u:.2e}, R_v = {R_v:.2e}")

    subsection("Parton 分布流体矩")
    e_moment = energy_density_moment(s0.fg, s0.fq, s0.x)
    p_moment = pressure_moment(s0.fg, s0.fq, s0.x)
    s_moment = entropy_density_estimate(e_moment, p_moment)
    print(f"  能量密度 e = {e_moment:.6f}")
    print(f"  压强 p = {p_moment:.6f}")
    print(f"  熵密度 s = {s_moment:.6f}")

    subsection("Knudsen 数")
    als_qgp = safe_alpha_s(5.0)
    T_qgp = 0.3  # GeV (RHIC 典型温度)
    lam = parton_mean_free_path(als_qgp, T_qgp)
    L_fm = 10.0
    Kn = knudsen_number(lam, L_fm)
    print(f"  alpha_s(5 GeV) = {als_qgp:.4f}, T = {T_qgp} GeV")
    print(f"  lambda_mfp = {lam:.4f} fm, L = {L_fm} fm")
    print(f"  Knudsen 数 Kn = {Kn:.4e}")
    if Kn < 0.1:
        print("  -> 流体近似适用")
    elif Kn < 1.0:
        print("  -> 中间耦合区, 需 viscous hydro")
    else:
        print("  -> 自由流区, 需完整 parton cascade")

    # ==================================================================
    # 总结
    # ==================================================================
    section("项目总结")
    print()
    print("  PartonShowerHD 项目成功运行!")
    print()
    print("  融合种子项目 (15 个):")
    print("    1. Variational-Data-Consistent-Assimilation  -> 4D-Var 优化 shower 初始条件")
    print("    2. walsh_transform                           -> 快度 Walsh 变换")
    print("    3. quality                                   -> 相空间网格质量度量")
    print("    4. i4mat_rref2                               -> 量子数守恒 RREF")
    print("    5. laplacian_matrix                          -> 色流图 Laplacian")
    print("    6. polynomial                                -> 分裂核多项式基")
    print("    7. conservation_ode                          -> Shower 守恒 ODE")
    print("    8. nonlin_fixed_point                        -> 自洽碎裂不动点")
    print("    9. fd1d_heat_steady                          -> 1D 稳态 parton 扩散")
    print("   10. opt_quadratic                             -> 碎裂函数二次优化")
    print("   11. fd3d_poisson                              -> 3D 通量管 Poisson")
    print("   12. navier_stokes_2d_exact                    -> Taylor-Green 流体基准")
    print("   13. diff_center                               -> 高阶中心差分")
    print("   14. dijkstra                                  -> 色弦最小配置")
    print("   15. hankel_cholesky                           -> 矩问题 Hankel 分解")
    print()
    print("  科学问题: 计算高能物理 - Parton shower 与强子化模型")
    print("              高阶有限差分与稳定性分析 (小规模可复现实验)")
    print()
    print("*" * 60)
    print("  运行完成, 无报错。")
    print("*" * 60)


if __name__ == "__main__":
    main()
