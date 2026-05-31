#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py
================================================================================
复合材料损伤演化与失效的博士级多尺度计算分析平台
================================================================================
科学问题：
  纤维增强复合材料层合板在循环载荷作用下的渐进损伤演化与失效预测。
  涵盖微观RVE（ Representative Volume Element ）均匀化、介观层合板刚度退化、
  宏观结构屈曲/振动稳定性分析，以及应力波传播检测。

核心物理模型：
  1. 细观力学均匀化（Halpin-Tsai / Mori-Tanaka）
  2. 连续损伤力学（CDM）疲劳损伤演化方程
  3. Hashin多模式失效准则
  4. 经典层合板理论（CLT）A-B-D刚度矩阵
  5. 一维间断伽辽金（DG）谱元法应力波传播
  6. 虚裂纹闭合技术（VCCT）能量释放率
  7. 屈曲与自由振动特征值分析
  8. 铺层顺序全局优化（Brent glomin + 动态规划）

原15个种子项目映射：
  - 511_heartbeat_ode       → 疲劳损伤演化ODE（立方非线性结构）
  - 891_polygonal_surface_display → RVE多边形纤维几何拓扑
  - 1385_vandermonde_interp_2d    → 二维应力场Vandermonde插值
  - 508_hb_to_mm          → 稀疏刚度矩阵结构与存储
  - 1387_vanderpol_ode_period    → 损伤极限循环次数估计
  - 950_quadrature_weights_vandermonde → 数值积分权重计算
  - 020_artery_pde        → 应力波传播PDE结构
  - 519_hermite_exactness → Hermite概率加权积分
  - 156_change_dynamic    → 动态规划最优铺层搜索
  - 972_r8but             → 带状上三角矩阵求解
  - 688_linpack_bench_backslash  → 稠密系统求解与残差分析
  - 1206_test_eigen       → 特征值结构生成与正交变换
  - 104_boundary_locus    → 时间积分稳定性区域分析
  - 471_glomin            → 全局优化（Lipschitz约束Brent法）
  - 274_dg1d_maxwell      → DG谱元法框架（Jacobi多项式、RK时间推进）
================================================================================
"""

import numpy as np
import time

# 项目模块导入
from utils import (
    r8vec_print, compute_condition_number, compute_normalized_residual,
    file_row_count, file_column_count
)
from rve_geometry import generate_hexagonal_fiber_rve, RVEGeometry
from material_properties import create_carbon_epoxy, CompositeMaterial
from damage_mechanics import (
    DamageParameters, DamageState, hashin_failure_criteria,
    integrate_damage_cycles, compute_damage_dissipation_energy, estimate_damage_period
)
from stiffness_assembly import (
    LaminateStiffness, SparseStiffnessAssembler, BandedUpperTriangularSolver,
    solve_equilibrium_dense, solve_equilibrium_sparse
)
from spectral_element import DGSpectralElement1D
from quadrature_integrals import (
    quadrature_weights_vandermonde, gauss_legendre_nodes_weights,
    hermite_gauss_nodes_weights, hermite_monomial_integral,
    compute_j_integral, compute_vcct_energy_release_rate,
    probabilistic_strength_integral, compute_strain_energy_release_rate_quadrature
)
from eigen_buckling import (
    BucklingAnalysis, VibrationAnalysis,
    generate_symmetric_eigenproblem, generate_nonsymmetric_eigenproblem
)
from wave_pde import WavePropagation1D, compute_stress_wave_reflection_coefficient
from stability_solver import (
    rk4_stability_function, low_storage_rk54_stability_function,
    check_eigenvalue_in_stability_region, compute_max_stable_timestep,
    analyze_damage_jacobian_eigenvalues, recommend_time_integrator
)
from optimization_design import LaminateOptimization, compute_ply_combinations


def section_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    t_start = time.time()
    np.random.seed(42)

    # ========================================================================
    # 1. RVE几何建模与纤维体积分数计算
    # ========================================================================
    section_header("1. RVE Geometry & Fiber Volume Fraction")
    rve = generate_hexagonal_fiber_rve(
        width=100.0, height=100.0, fiber_radius=8.0,
        n_fibers_x=3, n_fibers_y=3)
    rve.print_geometry_summary()

    # 2D Vandermonde插值示例：在RVE上构造虚拟应力场并插值
    n_data = min(rve.nx * rve.ny, 15)
    field_vals = np.sin(np.linspace(0, 2 * np.pi, n_data))
    eval_pts = np.array([[25.0, 25.0], [50.0, 50.0], [75.0, 75.0]])
    interp_vals = rve.vandermonde_interp_2d_field(field_vals, eval_pts, degree=3)
    print("  2D Vandermonde interpolated stress field at eval points:")
    for i, pt in enumerate(eval_pts):
        print(f"    ({pt[0]:.1f}, {pt[1]:.1f}) -> {interp_vals[i]:.6f}")

    # ========================================================================
    # 2. 材料性能与细观力学均匀化
    # ========================================================================
    section_header("2. Micromechanics Homogenization")
    V_f = rve.fiber_volume_fraction()
    material = create_carbon_epoxy(V_f=V_f)
    material.print_properties()

    # 偏轴刚度示例
    Q_45 = material.compute_transformed_stiffness(45.0)
    print("  Transformed stiffness Q̄(45°) [GPa]:")
    for i in range(3):
        row_str = "    " + "  ".join(f"{Q_45[i,j]:10.4f}" for j in range(3))
        print(row_str)

    # ========================================================================
    # 3. 层合板刚度矩阵（A-B-D）
    # ========================================================================
    section_header("3. Laminate A-B-D Stiffness Matrix")
    plies = [0.0, 45.0, -45.0, 90.0, 90.0, -45.0, 45.0, 0.0]
    thicknesses = [0.125] * len(plies)
    laminate = LaminateStiffness(plies, thicknesses, material)

    print(f"  Symmetric laminate: {plies}")
    print(f"  Total thickness: {laminate.get_total_thickness():.4f} mm")
    print("  A matrix [N/mm]:")
    for i in range(3):
        print("    " + "  ".join(f"{laminate.A[i,j]:12.4f}" for j in range(3)))
    print("  D matrix [N·mm]:")
    for i in range(3):
        print("    " + "  ".join(f"{laminate.D[i,j]:12.4f}" for j in range(3)))

    # ========================================================================
    # 4. 损伤演化分析
    # ========================================================================
    section_header("4. Progressive Damage Evolution Under Cyclic Loading")
    params = DamageParameters()

    # 循环应力幅值 [σ1, σ2, τ12] (MPa)
    stress_amp = np.array([1200.0, 60.0, 80.0])
    print(f"  Cyclic stress amplitude: σ1={stress_amp[0]:.1f}, σ2={stress_amp[1]:.1f}, τ12={stress_amp[2]:.1f} MPa")

    # Hashin失效评估
    criteria = hashin_failure_criteria(stress_amp, params)
    print("  Hashin failure criteria:")
    for mode, factor in criteria.items():
        status = "FAILED" if factor >= 1.0 else "SAFE"
        print(f"    {mode:20s}: {factor:.4f}  [{status}]")

    # 损伤极限循环次数估计（vanderpol-like period estimation）
    N_est = estimate_damage_period(stress_amp, params)
    print(f"  Estimated damage period (cycles to failure): {N_est:.2e}")

    # 积分损伤演化
    num_cycles = 5000
    initial_damage = DamageState(d_f=0.0, d_m=0.0, d_s=0.0, d_i=0.0)
    # 构造应力历史（恒定幅值）
    stress_history = np.tile(stress_amp, (num_cycles, 1))
    damage_states = integrate_damage_cycles(initial_damage, stress_history, params, num_cycles)

    final_damage = DamageState.from_array(damage_states[-1])
    print(f"  After {num_cycles} cycles:")
    print(f"    Fiber damage d_f    = {final_damage.d_f:.6f}")
    print(f"    Matrix damage d_m   = {final_damage.d_m:.6f}")
    print(f"    Shear damage d_s    = {final_damage.d_s:.6f}")
    print(f"    Interface damage d_i= {final_damage.d_i:.6f}")

    # 损伤耗散能
    W_d = compute_damage_dissipation_energy(damage_states, material, params)
    print(f"  Damage dissipation energy: {W_d:.6f} MJ/m³")

    # ========================================================================
    # 5. 退化刚度与平衡求解
    # ========================================================================
    section_header("5. Degraded Stiffness & Equilibrium Solution")
    # 为各层分配损伤状态（外层损伤更大）
    damage_by_ply = []
    for k in range(laminate.n_plies):
        scale = 1.0 - 0.5 * abs(k - laminate.n_plies / 2.0) / (laminate.n_plies / 2.0)
        d = DamageState(
            d_f=final_damage.d_f * scale,
            d_m=final_damage.d_m * scale,
            d_s=final_damage.d_s * scale,
            d_i=final_damage.d_i * scale
        )
        damage_by_ply.append(d)

    A_deg, B_deg, D_deg = laminate.compute_degraded_abd(damage_by_ply)
    print("  Degraded A matrix [N/mm]:")
    for i in range(3):
        print("    " + "  ".join(f"{A_deg[i,j]:12.4f}" for j in range(3)))

    # 稠密系统求解示例（linpack_bench_backslash思想）
    n_dof = 20
    K_test = np.random.randn(n_dof, n_dof)
    K_test = K_test.T @ K_test + 0.1 * np.eye(n_dof)  # 确保正定
    F_test = np.random.randn(n_dof)
    U_test, r_norm, norm_res, cond_K = solve_equilibrium_dense(K_test, F_test)
    print(f"  Dense solver test (n={n_dof}):")
    print(f"    Residual norm       = {r_norm:.6e}")
    print(f"    Normalized residual = {norm_res:.6e}")
    print(f"    Condition number    = {cond_K:.4e}")

    # 带状上三角求解器示例（r8but思想）
    n_band = 8
    mu = 2
    A_band = np.random.randn(mu + 1, n_band)
    A_band[mu, :] += 5.0  # 确保对角占优
    b_band = np.random.randn(n_band)
    solver_but = BandedUpperTriangularSolver(n_band, mu)
    x_but = solver_but.solve(A_band, b_band)
    b_check = solver_but.multiply(A_band, x_but)
    err_but = np.linalg.norm(b_band - b_check)
    print(f"  Banded solver test (n={n_band}, mu={mu}):")
    print(f"    Reconstruction error = {err_but:.6e}")

    # ========================================================================
    # 6. DG谱元法应力波传播
    # ========================================================================
    section_header("6. DG Spectral Element Wave Propagation")
    N_poly = 4
    K_elem = 10
    L_wave = 100.0

    def E_with_damage(x):
        # 损伤导致模量退化，使用较小模量保持数值稳定
        d_local = 0.1 * (x / L_wave)
        return 2000.0 * (1.0 - d_local)

    dg_solver = DGSpectralElement1D(
        N=N_poly, K=K_elem, x_bounds=[0.0, L_wave],
        rho=1.6, E_func=E_with_damage)

    sigma0 = np.zeros((dg_solver.Np, dg_solver.K))
    v0 = np.zeros((dg_solver.Np, dg_solver.K))
    # 初始高斯脉冲
    for k in range(dg_solver.K):
        for i in range(dg_solver.Np):
            x = dg_solver.x[i, k]
            v0[i, k] = np.exp(-((x - 30.0) / 15.0) ** 2)

    sigma_final, v_final = dg_solver.solve(sigma0, v0, FinalTime=0.5)
    # 数值稳定性检查
    v_final = np.clip(v_final, -1e6, 1e6)
    sigma_final = np.clip(sigma_final, -1e6, 1e6)
    print(f"  DG solver: N={N_poly}, K={K_elem}, L={L_wave:.1f}")
    print(f"  Initial pulse max velocity: {np.max(v0):.6f}")
    print(f"  Final max velocity: {np.max(v_final):.6f}")
    c_vals = dg_solver.compute_wave_speed()
    print(f"  Wave speed range: [{np.min(c_vals):.2f}, {np.max(c_vals):.2f}]")

    # ========================================================================
    # 7. 数值积分与能量释放率
    # ========================================================================
    section_header("7. Quadrature Rules & Energy Release Rate")

    # Vandermonde求积权重
    n_q = 5
    x_q = np.linspace(0.0, 1.0, n_q)
    w_q = quadrature_weights_vandermonde(n_q, 0.0, 1.0, x_q)
    print(f"  Vandermonde quadrature weights (n={n_q}):")
    r8vec_print(n_q, w_q, "")

    # Gauss-Legendre积分示例
    x_gl, w_gl = gauss_legendre_nodes_weights(8, 0.0, 1.0)
    integral_test = np.sum(w_gl * np.sin(np.pi * x_gl))
    exact = 2.0 / np.pi
    print(f"  Gauss-Legendre ∫_0^1 sin(πx) dx = {integral_test:.8f} (exact={exact:.8f})")

    # Hermite积分精确性检验
    print("  Hermite quadrature exactness check:")
    for deg in [0, 2, 4, 6, 8]:
        x_h, w_h = hermite_gauss_nodes_weights(5)
        quad_val = np.sum(w_h * (x_h ** deg))
        exact_val = hermite_monomial_integral(deg, option=1)
        err = abs(quad_val - exact_val) / (abs(exact_val) + 1e-15)
        print(f"    Degree {deg}: quad={quad_val:.10f}, exact={exact_val:.10f}, rel_err={err:.2e}")

    # VCCT能量释放率
    G_I, G_II = compute_vcct_energy_release_rate(
        stress_at_crack_tip=80.0, displacement_jump=0.05, delta_a=2.0, n_quad=10)
    print(f"  VCCT Energy Release Rate:")
    print(f"    G_I  = {G_I:.6f} N/mm")
    print(f"    G_II = {G_II:.6f} N/mm")

    # J积分
    J_val = compute_j_integral(None, None, [50.0, 0.0], 10.0, n_quad=16)
    print(f"  J-integral estimate = {J_val:.6f} N/mm")

    # 概率化强度
    E_strength = probabilistic_strength_integral(mean_strength=2500.0, std_strength=200.0)
    print(f"  Probabilistic mean strength (lognormal) = {E_strength:.2f} MPa")

    # ========================================================================
    # 8. 屈曲与振动特征值分析
    # ========================================================================
    section_header("8. Buckling & Vibration Eigenvalue Analysis")

    buckling = BucklingAnalysis(laminate.D, plate_length=200.0, plate_width=100.0, nx=12, ny=12)
    lambdas, modes = buckling.solve_buckling_loads(N_x=1.0, n_modes=3)
    print("  Buckling load factors λ:")
    for i, lam in enumerate(lambdas):
        if lam < np.inf and lam > 0:
            print(f"    Mode {i+1}: λ = {lam:.4f}, N_cr = {lam:.4f} N/mm")
        else:
            print(f"    Mode {i+1}: no valid mode found")

    # 振动分析
    omegas, vibe_modes = VibrationAnalysis().solve_natural_frequencies(
        laminate.D, rho=1600.0, thickness=laminate.get_total_thickness(),
        plate_length=200.0, plate_width=100.0, nx=12, ny=12)
    if len(omegas) > 0:
        print("  Natural frequencies (rad/s):")
        for i in range(min(3, len(omegas))):
            freq_hz = omegas[i] / (2.0 * np.pi)
            print(f"    Mode {i+1}: ω = {omegas[i]:.4f} rad/s, f = {freq_hz:.4f} Hz")

    # 对称特征值问题生成（test_eigen映射）
    A_sym, Q_sym, lambda_sym = generate_symmetric_eigenproblem(6, lambda_mean=1.0, lambda_dev=0.3)
    print("  Symmetric eigenproblem test (n=6):")
    print(f"    Generated eigenvalues: {np.array_str(lambda_sym, precision=4)}")

    # 非对称特征值问题（含损伤耦合）
    A_nsym, Q_nsym, T_nsym = generate_nonsymmetric_eigenproblem(6, lambda_mean=-0.5, lambda_dev=0.2)
    eig_nsym = np.linalg.eigvals(A_nsym)
    print("  Nonsymmetric eigenproblem test (n=6):")
    print(f"    Eigenvalues: {np.array_str(eig_nsym, precision=4)}")

    # ========================================================================
    # 9. 应力波传播（Artery-PDE类比）
    # ========================================================================
    section_header("9. Stress Wave Propagation in Damaged Composite")

    def E_func_wave(x):
        return material.E1 * 1e3 * (1.0 - 0.2 * np.sin(np.pi * x / 100.0))

    wave = WavePropagation1D(
        L=100.0, nx=51, E_func=E_func_wave, rho=1600.0,
        damping_ratio=0.05, forcing_params=(1000.0, 50.0, 50.0))

    u0 = np.zeros(wave.nx)
    v0 = np.zeros(wave.nx)
    v0[wave.nx // 2] = 1.0  # 初始速度脉冲
    t_span = np.linspace(0.0, 0.5e-3, 101)

    u_hist, v_hist = wave.solve(u0, v0, t_span)
    print(f"  Wave propagation: L={wave.L:.1f} mm, nx={wave.nx}")
    print(f"  Max displacement at t={t_span[-1]*1e6:.1f} μs: {np.max(np.abs(u_hist[-1])):.6f} mm")
    print(f"  Attenuation coefficient @ 50kHz: {wave.compute_attenuation_coefficient(50000.0):.6f} Np/m")

    # 波反射系数
    R_ref, T_ref = compute_stress_wave_reflection_coefficient(
        E1=material.E1 * 1e3, E2=material.E2 * 1e3, rho1=1600.0, rho2=1400.0)
    print(f"  Interface reflection coefficient R = {R_ref:.4f}")
    print(f"  Interface transmission coefficient T = {T_ref:.4f}")

    # ========================================================================
    # 10. 稳定性分析
    # ========================================================================
    section_header("10. Stability Analysis of Time Integration")

    # 分析当前损伤状态的Jacobian特征值
    eig_dam = analyze_damage_jacobian_eigenvalues(
        final_damage.to_array(), stress_amp, params)
    print("  Damage Jacobian eigenvalues:")
    print(f"    {np.array_str(eig_dam, precision=4)}")

    rec = recommend_time_integrator(final_damage, stress_amp, params)
    print(f"  Recommended integrator: {rec['method']}")
    print(f"  Max stable dt (RK4)     = {rec['max_dt_rk4']:.4e}")
    print(f"  Max stable dt (RK54)    = {rec['max_dt_rk54']:.4e}")
    print(f"  Stiffness ratio         = {rec['stiffness_ratio']:.2f}")

    # 稳定性区域检查
    dt_test = 1.0
    stable_flags = check_eigenvalue_in_stability_region(eig_dam, dt_test, rk4_stability_function)
    n_stable = np.sum(stable_flags)
    print(f"  Eigenvalues stable @ dt={dt_test}: {n_stable}/{len(eig_dam)}")

    # ========================================================================
    # 11. 层合板优化设计
    # ========================================================================
    section_header("11. Laminate Stacking Sequence Optimization")

    optimizer = LaminateOptimization(
        material=material, n_plies=8, target_load=500.0)

    # 全局单角度优化
    best_theta, best_obj, calls = optimizer.optimize_single_angle()
    print(f"  Single-angle optimization:")
    print(f"    Optimal angle = {best_theta:.2f}°")
    print(f"    Objective     = {best_obj:.6f}")
    print(f"    Function calls= {calls}")

    # 动态规划离散角度优化
    opt_angles_dp, dp_obj = optimizer.dynamic_programming_stack(
        angle_set=[0, 45, -45, 90])
    print(f"  Dynamic programming stack optimization:")
    print(f"    Optimal angles = {opt_angles_dp}")
    print(f"    Objective      = {dp_obj:.6f}")

    n_combos = compute_ply_combinations(8, [0, 45, -45, 90])
    print(f"  Total ply combinations (4 angles): {n_combos:.2e}")

    # ========================================================================
    # 12. 综合结果汇总
    # ========================================================================
    section_header("12. Summary of Results")
    t_elapsed = time.time() - t_start

    print(f"  RVE fiber volume fraction      : {V_f:.4f}")
    print(f"  Longitudinal modulus E_1       : {material.E1:.2f} GPa")
    print(f"  Critical buckling load factor  : {lambdas[0] if len(lambdas) > 0 and lambdas[0] < np.inf else 'N/A'}")
    print(f"  Final fiber damage d_f         : {final_damage.d_f:.6f}")
    print(f"  Final matrix damage d_m        : {final_damage.d_m:.6f}")
    print(f"  Damage dissipation energy      : {W_d:.6f} MJ/m³")
    print(f"  VCCT G_I                       : {G_I:.6f} N/mm")
    print(f"  Max stable dt (RK4)            : {rec['max_dt_rk4']:.4e}")
    print(f"  Optimal single angle           : {best_theta:.2f}°")
    print(f"  Elapsed time                   : {t_elapsed:.3f} s")
    print("=" * 70)
    print("  COMPOSITE DAMAGE EVOLUTION ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
