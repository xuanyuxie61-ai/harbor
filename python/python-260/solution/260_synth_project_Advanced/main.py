"""
============================================================================
main.py -- 暗能量状态方程约束: 高阶有限差分与稳定性分析
           统一入口 (零参数运行)

============================================================================
Project 260: 计算宇宙学 -- 暗能量状态方程约束
           高阶有限差分与稳定性分析 (小规模可复现实验)
============================================================================

科学问题
--------
暗能量是什么? 宇宙加速膨胀背后的驱动力是宇宙学常数 (w=-1)
还是动力学标量场 (w != -1)?

本项目通过以下流程约束暗能量状态方程参数 (w0, wa):
  1. CPL 参数化 + FLRW 背景宇宙学
  2. 高阶有限差分算子 (2/4/6/8 阶) + 紧致 (Padé) 格式
  3. von Neumann 稳定性分析 + CFL 条件
  4. 增长方程 BDF 多步法求解 (BDF1, BDF2, IMEX)
  5. CVT 优化巡天几何 + 蒙特卡洛体积积分
  6. Fisher 矩阵预测 + Gibbs 采样约束
  7. Sobol 全局灵敏度分析
  8. 综合验证: EdS/dS 极限, 收敛阶, 不稳定 ODE, 摄动 Kepler

核心物理公式
------------
(1) CPL EoS: w(a) = w0 + wa*(1-a)
(2) E^2(a) = Om/a^3 + Ode*a^{-3(1+w0+wa)}*exp(-3wa(1-a))
(3) Growth: D'' + (2+H'/H)D' - (3/2)Om(a)D = 0
(4) FD_p: f'(x) ~ sum c_j f(x_j) / h  (各阶系数)
(5) von Neumann: |G(z)| <= 1 for stability
(6) BDF2: (3/2)y_{n+1} - 2y_n + (1/2)y_{n-1} = dt f_{n+1}
(7) Fisher: F_{ab} = sum (1/sigma^2) dM/dp_a dM/dp_b
(8) Sobol: S_i = V[E(Y|X_i)] / V(Y)

运行方式
--------
    python main.py

无需任何参数, 自动完成全部分析流程.
============================================================================
"""

import math
import time
import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cosmo_constants as cc
from fd_operators import (
    fd1_matrix, fd2_matrix, uniform_grid, chebyshev_grid,
    compact_fd1_tridiag, compact_fd1_rhs, solve_tridiag,
    BlockToeplitzMatrix, make_laplacian_block_toeplitz,
    compute_mesh_bandwidth, lagrange_interpolate,
    gauss_legendre_integrate, gauss_legendre_nodes_weights,
    _fornberg_standard
)
from stability_analysis import (
    AMPLIFICATION_METHODS, modified_wavenumber_fd1,
    modified_wavenumber_fd2, modified_wavenumber_compact4,
    cfl_limit, growth_equation_eigenvalues, stiffness_ratio,
    integrate_kepler_rk4, kepler_perturbed_hamiltonian,
    kepler_angular_momentum, unstable_ode_test,
    von_neumann_report, dispersion_analysis
)
from growth_solver import BDFGrowthSolver, growth_coefficients
from monte_carlo_geom import (
    set_seed as mc_set_seed, triangle_area,
    monte_carlo_triangle_integral, disk_monomial_integral,
    monte_carlo_disk_integral, circle_monomial_integral,
    monte_carlo_circle_integral, stratified_disk_integral,
    effective_survey_volume, eisenstein_hu_power_spectrum
)
from cvt_mesh import CVTGenerator, voronoi_distance_field, voronoi_cell_area_estimate
from adaptive_mesh import (
    richardson_extrapolate, AdaptiveMesh1D, convergence_order_test,
    log_adaptive_grid
)
from fisher_forecast import (
    set_seed as fisher_set_seed, FisherMatrix, GibbsDarkEnergySampler,
    ParameterSweep, detect_chain_artifacts, bao_signal_model, dsc_filter
)
from sensitivity_analysis import (
    set_seed as sens_set_seed, SobolSensitivity, oat_sensitivity,
    elasticity_analysis
)


def section_header(title):
    width = 64
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def subsection(title):
    print(f"\n--- {title} ---")


# =====================================================================
#  Section 1: 宇宙学参数与背景演化
# =====================================================================

def run_section_1():
    section_header("1. CPL 暗能量状态方程与 FLRW 背景")

    print(f"\n基准模型 (LCDM):")
    print(f"  H0  = {cc.H0_KMS_MPC:.2f} km/s/Mpc")
    print(f"  Om  = {cc.OMEGA_M0:.4f}")
    print(f"  ODE = {cc.OMEGA_DE0:.4f}")
    print(f"  D_H = {cc.D_H_Mpc:.2f} Mpc")
    print(f"  t_H = {cc.t_H_Gyr:.2f} Gyr")

    print(f"\nCPL 模型 (w0=-0.9, wa=-0.3):")
    print(f"  w(z=0) = {cc.cpl_eos(1.0, -0.9, -0.3):.4f}")
    print(f"  w(z=1) = {cc.cpl_eos(0.5, -0.9, -0.3):.4f}")
    print(f"  Phantom crossing: {cc.is_phantom_crossing(-0.9, -0.3)}")
    print(f"  Phantom crossing a: {cc.phantom_crossing_scale_factor(-0.9, -0.3)}")

    subsection("FLRW 背景 (LCDM)")
    for z in [0.0, 0.5, 1.0, 2.0]:
        Hz = cc.hubble_z(z)
        dL = cc.luminosity_distance_mpc(z)
        print(f"  z={z:.1f}: H(z)={Hz:.2f} km/s/Mpc, d_L={dL:.1f} Mpc")

    print(f"\n  t(0) = {cc.age_of_universe_gyr(0.0):.2f} Gyr")
    print(f"  t(z=1) = {cc.age_of_universe_gyr(1.0):.2f} Gyr")
    print(f"  q(a=1) = {cc.deceleration_param(1.0):.4f}")


# =====================================================================
#  Section 2: 高阶有限差分算子
# =====================================================================

def run_section_2():
    section_header("2. 高阶有限差分算子与紧致格式")

    subsection("均匀网格 FD 精度测试")
    grid, h = uniform_grid(0.0, 1.0, 21)
    f = [x**3 for x in grid]  # f(x) = x^3
    df_exact = [3.0*x**2 for x in grid]
    d2f_exact = [6.0*x for x in grid]

    for order in [2, 4, 6, 8]:
        D1 = fd1_matrix(21, h, order=order)
        D1f = [sum(D1[i][j]*f[j] for j in range(21)) for i in range(21)]
        p = order // 2
        err = max(abs(D1f[i]-df_exact[i]) for i in range(p, 21-p))
        print(f"  FD1(order={order}): max err = {err:.4e}")

    for order in [2, 4, 6]:
        D2 = fd2_matrix(21, h, order=order)
        D2f = [sum(D2[i][j]*f[j] for j in range(21)) for i in range(21)]
        p = order // 2
        err = max(abs(D2f[i]-d2f_exact[i]) for i in range(p, 21-p))
        print(f"  FD2(order={order}): max err = {err:.4e}")

    subsection("紧致 (Padé) 格式")
    lo, di, up = compact_fd1_tridiag(21, h, compact_order=4)
    rhs = compact_fd1_rhs(21, h, f, compact_order=4)
    df_compact = solve_tridiag(lo, di, up, rhs)
    err_c = max(abs(df_compact[i]-df_exact[i]) for i in range(1, 20))
    print(f"  Compact-4: max err = {err_c:.4e}")

    subsection("修正波数 (色散关系)")
    for order in [2, 4, 6, 8]:
        keff_max = modified_wavenumber_fd1(math.pi, order)
        print(f"  FD{order}: k_eff(pi) = {keff_max:.4f} (exact = pi = {math.pi:.4f})")
    keff_c4 = modified_wavenumber_compact4(math.pi)
    print(f"  Compact-4: k_eff(pi) = {keff_c4:.4f}")

    subsection("块 Toeplitz 矩阵 (种子项目 971)")
    bt = make_laplacian_block_toeplitz(4, 3)
    x_test = [1.0] * bt.N
    y = bt.matvec(x_test)
    print(f"  块数 = {bt.L}, 总大小 = {bt.N}")
    print(f"  带宽 = {bt.bandwidth()}")
    print(f"  A*ones (前4): [{y[0]:.2f}, {y[1]:.2f}, {y[2]:.2f}, {y[3]:.2f}]")

    subsection("Fornberg 算法 (种子项目 395)")
    pts = [0, 1, 2, 3, 4]
    coeffs = _fornberg_standard([float(p) for p in pts], 2.0, 1)
    print(f"  5点一阶导 @ x=2: {coeffs}")

    subsection("Gauss-Legendre 积分 (种子项目 395)")
    for n in [2, 3, 5, 10]:
        val = gauss_legendre_integrate(lambda x: x**(2*n-2), -1.0, 1.0, n)
        exact = 2.0 / (2*n - 1)
        print(f"  GL({n}): int x^{2*n-2} = {val:.12f} (exact = {exact:.12f})")


# =====================================================================
#  Section 3: von Neumann 稳定性分析
# =====================================================================

def run_section_3():
    section_header("3. von Neumann 稳定性分析与 CFL 条件")

    subsection("放大因子测试")
    z_test = complex(-1.0, 0.0)
    print(f"  z = {z_test}:")
    for name in ['explicit_euler', 'implicit_euler', 'crank_nicolson', 'rk4']:
        func = AMPLIFICATION_METHODS[name]
        G = func(z_test)
        print(f"    {name:16s}: |G| = {abs(G):.6f}")

    subsection("增长方程特征值与刚性")
    for P, Q, label in [(2.0, -1.5, "EdS"),
                         (1.55, -0.47, "LCDM today"),
                         (2.0, 0.0, "de Sitter")]:
        l1, l2 = growth_equation_eigenvalues(P, Q)
        S = stiffness_ratio(P, Q)
        print(f"  {label}: P={P}, Q={Q}: "
              f"lam1={l1.real:.4f}, lam2={l2.real:.4f}, S={S:.2f}")

    subsection("CFL 条件")
    for method in ['explicit_euler', 'rk2', 'rk4']:
        for fd_order in [2, 4, 6]:
            info = cfl_limit(method, fd_order)
            print(f"  {method:16s} + FD{fd_order}: CFL = {info['cfl_number']:.4f}")

    subsection("不稳定 ODE 测试 (种子项目 1374)")
    for method in ['explicit_euler', 'implicit_euler', 'rk4']:
        res = unstable_ode_test(mu=0.1, t_stop=5.0, dt=0.02, method=method)
        print(f"  {method:16s}: err={res['max_abs_error']:.4e}, stable={res['is_stable']}")

    subsection("摄动 Kepler (种子项目 619)")
    state0 = [1.0, 0.0, 0.0, 1.0]
    traj = integrate_kepler_rk4(state0, t_end=6.28, dt=0.01, epsilon=0.001)
    H0 = kepler_perturbed_hamiltonian(state0, epsilon=0.001)
    Hf = kepler_perturbed_hamiltonian(traj[-1], epsilon=0.001)
    L0 = kepler_angular_momentum(state0)
    Lf = kepler_angular_momentum(traj[-1])
    print(f"  H: {H0:.6f} -> {Hf:.6f}, dH={abs(Hf-H0):.2e}")
    print(f"  L: {L0:.6f} -> {Lf:.6f}, dL={abs(Lf-L0):.2e}")


# =====================================================================
#  Section 4: 增长因子 BDF/IMEX 求解
# =====================================================================

def run_section_4():
    section_header("4. 增长因子 D(a) 多方法求解")

    subsection("BDF1 (隐式 Euler + Picard)")
    solver = BDFGrowthSolver(w0=-1.0, wa=0.0)
    solver.setup_grid(a_min=1e-4, n_points=100)
    D_bdf1 = solver.solve_bdf1()
    f_bdf1 = solver.compute_growth_rate()
    print(f"  D(a=1) = {D_bdf1[-1]:.6f}")
    print(f"  f(a=1) = {f_bdf1[-1]:.4f}")
    print(f"  f*sigma8(a=1) = {solver.f_sigma8()[-1]:.4f}")

    subsection("BDF2")
    D_bdf2 = solver.solve_bdf2()
    f_bdf2 = solver.compute_growth_rate()
    print(f"  D(a=1) = {D_bdf2[-1]:.6f}")
    print(f"  f(a=1) = {f_bdf2[-1]:.4f}")

    subsection("IMEX (Implicit-Explicit)")
    D_imex = solver.solve_imex()
    f_imex = solver.compute_growth_rate()
    print(f"  D(a=1) = {D_imex[-1]:.6f}")
    print(f"  f(a=1) = {f_imex[-1]:.4f}")

    subsection("FD 配置法 (4阶)")
    D_fd = solver.solve_fd_collocation(fd_order=4)
    Dp_fd = solver.D_prime
    f_fd = [Dp_fd[i]/D_fd[i] if abs(D_fd[i])>1e-30 else 1.0
            for i in range(len(D_fd))]
    print(f"  D(a=1) = {D_fd[-1]:.6f}")
    print(f"  f(a=1) = {f_fd[-1]:.4f}")

    subsection("CPL 模型 (w0=-0.9, wa=-0.3)")
    solver_cpl = BDFGrowthSolver(w0=-0.9, wa=-0.3)
    solver_cpl.setup_grid(a_min=1e-4, n_points=100)
    D_cpl = solver_cpl.solve_bdf1()
    f_cpl = solver_cpl.compute_growth_rate()
    print(f"  D(a=1) = {D_cpl[-1]:.6f}")
    print(f"  f(a=1) = {f_cpl[-1]:.4f}")

    subsection("物质主导极限检验")
    check = solver.matter_dominated_check()
    print(f"  max D error: {check['max_D_error']:.4e}")
    print(f"  max f error: {check['max_f_error']:.4e}")


# =====================================================================
#  Section 5: CVT 巡天与蒙特卡洛
# =====================================================================

def run_section_5():
    section_header("5. CVT 巡天几何与蒙特卡洛积分")
    mc_set_seed(42)

    subsection("圆盘精确积分 vs MC")
    for e1, e2 in [(0,0), (2,0), (2,2), (4,0)]:
        exact = disk_monomial_integral(e1, e2)
        def integrand(x, y, e1=e1, e2=e2):
            return x**e1 * y**e2
        mc, se = monte_carlo_disk_integral(integrand, n_samples=10000)
        print(f"  x^{e1}*y^{e2}: exact={exact:.6f}, MC={mc:.6f} +/- {se:.6f}")

    subsection("圆周精确积分")
    for e1, e2 in [(0,0), (2,0), (2,2)]:
        exact = circle_monomial_integral(e1, e2)
        print(f"  x^{e1}*y^{e2}: exact = {exact:.6f}")

    subsection("三角形 MC (种子项目 1312)")
    v1, v2, v3 = (0,0), (1,0), (0.5,1)
    area = triangle_area(v1, v2, v3)
    mc, se = monte_carlo_triangle_integral(lambda x,y: 1.0, v1, v2, v3, 5000)
    print(f"  area = {area:.4f}, MC = {mc:.6f} +/- {se:.6f}")

    subsection("CVT 巡天优化 (种子项目 238)")
    cvt = CVTGenerator(dim=2, domain_bounds=[(0,10),(0,10)])
    result = cvt.run(n_generators=12, max_iter=30, tol=1e-3, n_samples=2000)
    print(f"  converged: {result['converged']}, iter: {result['n_iterations']}")
    print(f"  final energy: {result['final_energy']:.6f}")

    subsection("Voronoi 距离场 (种子项目 1396)")
    if result['generators']:
        field = voronoi_distance_field(result['generators'], (0,10), (0,10), 15)
        max_d = max(max(row) for row in field)
        min_d = min(min(row) for row in field)
        print(f"  min dist: {min_d:.3f}, max dist: {max_d:.3f}")

    subsection("有效巡天体积")
    v_eff = effective_survey_volume(n_gal=1e-4, n_mc=200)
    print(f"  V_eff = {v_eff:.4e} Mpc^3")


# =====================================================================
#  Section 6: 自适应网格与收敛阶
# =====================================================================

def run_section_6():
    section_header("6. 自适应网格与收敛阶分析")

    subsection("Richardson 外推")
    I_h = 0.1999
    I_2h = 0.199
    I_ext, err = richardson_extrapolate(I_h, I_2h, order=2)
    print(f"  I_h={I_h}, I_2h={I_2h} => I_ext={I_ext:.6f}, err={err:.6f}")

    subsection("自适应网格加密")
    mesh = AdaptiveMesh1D(a_min=1e-4, a_max=1.0)
    mesh.initialize_uniform(20)
    sol = [a**0.5 for a in mesh.nodes]
    result = mesh.refine_by_gradient(sol, threshold=0.3, max_refine=3)
    print(f"  refined: {result['n_refined']}, total: {result['n_total']}, "
          f"max level: {result['max_level']}")

    subsection("对数自适应网格")
    grid = log_adaptive_grid(a_min=1e-4, a_max=1.0, n_points=30,
                              refinement_func=lambda a: cc.omega_m_a(a))
    print(f"  nodes: {len(grid)}, range: [{grid[0]:.6f}, {grid[-1]:.6f}]")

    subsection("分层采样方差缩减")
    strat, se_s = stratified_disk_integral(lambda x,y: x*x+y*y,
                                             n_strata=4, samples_per_stratum=2000)
    plain, se_p = monte_carlo_disk_integral(lambda x,y: x*x+y*y, n_samples=8000)
    exact = disk_monomial_integral(2,0) + disk_monomial_integral(0,2)
    print(f"  exact = {exact:.6f}")
    print(f"  MC plain: {plain:.6f} +/- {se_p:.6f}")
    print(f"  stratified: {strat:.6f} +/- {se_s:.6f}")


# =====================================================================
#  Section 7: Fisher/Gibbs 约束
# =====================================================================

def run_section_7():
    section_header("7. Fisher 矩阵与 Gibbs 采样约束")
    fisher_set_seed(42)

    subsection("Fisher 矩阵")
    fisher = FisherMatrix()
    F = fisher.compute_fisher(w0=-1.0, wa=0.0)
    print(f"  F = [[{F[0][0]:.2f}, {F[0][1]:.2f}], "
          f"[{F[1][0]:.2f}, {F[1][1]:.2f}]]")
    errors = fisher.marginalized_errors(F)
    print(f"  sigma_w0 = {errors['sigma_w0']:.4f}")
    print(f"  sigma_wa = {errors['sigma_wa']:.4f}")
    print(f"  FoM = {fisher.figure_of_merit(F):.2f}")

    subsection("Gibbs 采样")
    gibbs = GibbsDarkEnergySampler(sigma_w0=0.1, sigma_wa=0.5, correlation=-0.3)
    result = gibbs.run(n_steps=300, burn_in=50)
    print(f"  w0 = {result['w0_mean']:.4f} +/- {result['w0_std']:.4f}")
    print(f"  wa = {result['wa_mean']:.4f} +/- {result['wa_std']:.4f}")

    subsection("链扰动检测 (种子项目 1236)")
    artifacts = detect_chain_artifacts(result['chain_w0'])
    print(f"  jumps: {artifacts['n_jumps']}, clean: {artifacts['is_clean']}")

    subsection("BAO 信号 + DSC 滤波")
    k_test = [0.01 * i for i in range(1, 30)]
    bao_vals = [bao_signal_model(k) for k in k_test]
    smooth, osc = dsc_filter(bao_vals, sigma=2.0)
    print(f"  BAO range: [{min(bao_vals):.4f}, {max(bao_vals):.4f}]")
    print(f"  smooth range: [{min(smooth):.4f}, {max(smooth):.4f}]")
    print(f"  oscillatory range: [{min(osc):.4f}, {max(osc):.4f}]")


# =====================================================================
#  Section 8: 参数扫描与灵敏度
# =====================================================================

def run_section_8():
    section_header("8. 参数扫描与灵敏度分析")

    subsection("K-sweep (种子项目 1081)")
    sweep = ParameterSweep(n_w0=3, n_wa=3)
    sr = sweep.run_sweep()
    print(f"  total points: {sr['n_total']}, stable: {sr['n_stable']}")
    print(f"  best: w0={sr['best_point']['w0']:.2f}, wa={sr['best_point']['wa']:.2f}")
    imp = sweep.compute_feature_importance()
    print(f"  Delta_D(w0) = {imp['delta_D_w0']:.4f}")
    print(f"  Delta_D(wa) = {imp['delta_D_wa']:.4f}")
    print(f"  dominant: {imp['dominant_parameter']}")

    subsection("OAT 灵敏度")
    oat = oat_sensitivity()
    print(f"  base f = {oat['f_base']:.4f}")
    for i, name in enumerate(oat['param_names']):
        print(f"  S({name}) = {oat['sensitivities'][i]:.6f}")

    subsection("弹性分析")
    elast = elasticity_analysis()
    for i, name in enumerate(elast['param_names']):
        print(f"  E({name}) = {elast['elasticities'][i]:.6f}")

    subsection("Sobol 全局灵敏度")
    sens_set_seed(42)
    sobol = SobolSensitivity()
    result = sobol.compute_sobol(n_samples=20)
    for i, name in enumerate(result['param_names']):
        print(f"  S1({name}) = {result['S1'][i]:.4f}")
        print(f"  ST({name}) = {result['ST'][i]:.4f}")


# =====================================================================
#  Main
# =====================================================================

def main():
    print("=" * 64)
    print("  暗能量状态方程约束: 高阶有限差分与稳定性分析")
    print("  High-Order FD + von Neumann Stability for DE EoS")
    print("=" * 64)
    print(f"\n开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    t_start = time.time()

    mc_set_seed(42)
    random.seed(42)

    run_section_1()
    run_section_2()
    run_section_3()
    run_section_4()
    run_section_5()
    run_section_6()
    run_section_7()
    run_section_8()

    t_total = time.time() - t_start

    section_header("计算完成")
    print(f"\n  总耗时: {t_total:.2f} 秒")
    print(f"\n所有模块运行成功.")
    print("=" * 64)


if __name__ == '__main__':
    main()
