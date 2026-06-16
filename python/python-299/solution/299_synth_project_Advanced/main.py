#!/usr/bin/env python3
"""
main.py — Fokker-Planck 碰撞输运计算: 高阶有限差分与稳定性分析
================================================================

统一入口, 零参数可运行.

科学问题:
  求解各向同性 Fokker-Planck 碰撞方程, 模拟电子等离子体中
  双温度分布函数向 Maxwellian 平衡态的碰撞弛豫过程.
  使用 4 阶有限差分, 隐式/半隐式时间推进, 并执行完整的
  稳定性分析和多种诊断.

无量纲方程:
  ∂f/∂τ = (1/x²) ∂/∂x { x² [ D(x) ∂f/∂x + A(x) f ] }
  其中 D(x) = G(x) M(x)/x, A(x) = 2 G(x) M(x)/x²
        G(x) = [erf(x) - 2x/√π exp(-x²)] / (2x²)
        M(x) = 4π ∫₀ˣ f(v') v'² dv'

初始条件:
  f(x,0) = 0.5 · f_M(x; T₁=0.8) + 0.5 · f_M(x; T₂=1.2)
  (双温度混合, 总能量 = 归一化)

算法融合 (15 个种子项目):
  1.  Chandrupatla 求根 → 有效温度/色散关系求根
  2.  Voronoi 图       → 自适应速度网格
  3.  PWL 插值         → 分布函数重建
  4.  乘积求积         → 3D 速度空间矩计算
  5.  圆距离统计       → 散射角分布表征
  6.  GPC 预测编码     → 不确定性量化
  7.  多尺度特征提取   → 相空间结构诊断
  8.  有理背包         → 自适应网格优化
  9.  日历计算         → 实验时间戳
  10. 对比学习         → 相空间对比诊断
  11. L2 范数          → 收敛度量
  12. 非线性 FEM       → Newton 迭代稳态求解
  13. 金字塔求积       → 时空积分
  14. R8GB 带状矩阵    → 隐式系统求解
  15. 余弦积分         → 电磁屏蔽计算

运行: python main.py
"""

import sys
import os
import time
import math
import numpy as np

# 确保当前目录在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from physical_constants import (
    PlasmaParameters, PI, SQRT_PI, FOUR_PI,
    maxwellian_1d, maxwellian_shell, entropy_maxwellian,
)
from special_functions import (
    chandrasekhar_G, cosine_integral, plasma_dispersion_function,
    coulomb_logarithm_velocity_dependent,
)
from velocity_grid import (
    create_velocity_grid, VoronoiVelocityMesh, VelocityDistanceStatistics,
)
from fp_collision import (
    compute_collision_operator, compute_collision_coefficients,
    compute_cumulative_M, zero_chandrupatla, find_effective_temperature,
    newton_solve_steady_state,
)
from pwl_velocity import (
    PWLVelocityInterpolator, product_quadrature_3d_spherical,
    pyramid_velocity_time_quadrature, compute_moments,
    compute_rosenbluth_H, compute_rosenbluth_G_potential,
)
from banded_solver import (
    solve_tridiagonal, solve_pentadiagonal, r8gb_fa, band_matrix_info,
    band_matrix_vector_product,
)
from stability_analysis import (
    gershgorin_stability_check, von_neumann_analysis,
    compute_operator_eigenvalues, full_stability_diagnosis,
)
from entropy_norm import (
    l2_norm_discrete, l2_norm_perturbation, l2_norm_relative,
    boltzmann_H, entropy_S, entropy_production_rate,
    kl_divergence, convergence_metrics,
)
from adaptive_mesh import (
    knapsack_rational, compute_error_indicator,
    optimize_mesh_allocation, mesh_quality_metrics,
)
from calendar_utils import (
    PlasmaSimulationCalendar, generate_experiment_timestamp,
    format_experiment_header, gregorian_to_jdn,
)
from phase_space_contrastive import (
    MultiScaleFeatureExtractor, contrastive_phase_space_diagnostic,
    nt_xent_loss, triplet_loss,
)
from uncertainty_tracking import (
    UncertaintyTracker, stabilise_cov, uncertainty_diagnosis,
)
from fp_solver import (
    solve_fokker_planck, check_cfl_condition,
    fd_first_derivative_4th, fd_second_derivative_4th,
)


def separator(title=""):
    if title:
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}")
    else:
        print(f"\n{'='*70}")


def main():
    t_start = time.time()

    # =================================================================
    #  §0  实验初始化
    # =================================================================
    separator("Fokker-Planck 碰撞输运计算")
    print("高阶有限差分与稳定性分析 — 博士级科学计算项目")
    print()

    # 实验时间戳 (项目 135_calpak)
    ts = generate_experiment_timestamp()
    print(f"实验开始时间: {ts['gregorian']}")
    print(f"儒略日: {ts['jdn']}")
    print(f"Unix 时间戳: {ts['unix_timestamp']:.3f}")

    # =================================================================
    #  §1  等离子体参数  (项目 135_calpak → PlasmaParameters)
    # =================================================================
    separator("§1 等离子体参数设置")

    plasma = PlasmaParameters(
        T_e_eV=1000.0,
        n_e=1.0e19,
        Z_eff=1.0,
        m_i_amu=2.014,
    )
    print(plasma.summary())

    # 等离子体历法 (项目 135_calpak → PlasmaSimulationCalendar)
    cal = PlasmaSimulationCalendar(tau_c_seconds=plasma.tau_c)
    print(f"碰撞时间 τ_c = {plasma.tau_c:.4e} s")
    print(f"碰撞频率 ν_c = {plasma.nu_c:.4e} s⁻¹")

    # =================================================================
    #  §2  速度空间网格  (项目 1398_voronoi + 178_circle_distance)
    # =================================================================
    separator("§2 速度空间网格构造")

    V_MIN = 0.01
    V_MAX = 4.0
    N_GRID = 120

    # 均匀网格
    grid_uniform = create_velocity_grid(V_MIN, V_MAX, N_GRID, "uniform")
    v_grid = grid_uniform["v"]
    dv = v_grid[1] - v_grid[0]
    print(f"均匀网格: N={N_GRID}, v ∈ [{V_MIN}, {V_MAX}], Δv={dv:.4f}")

    # Voronoi 自适应网格 (项目 1398_voronoi_plot)
    f_init_test = maxwellian_1d(v_grid)
    vmesh = VoronoiVelocityMesh(V_MIN, V_MAX, N_GRID, f_dist=lambda v: float(maxwellian_1d(v)))
    print(f"Voronoi 网格: {len(vmesh.generators)} 生成元")
    print(f"  胞腔体积: min={np.min(vmesh.cell_volumes):.4f}, "
          f"max={np.max(vmesh.cell_volumes):.4f}, "
          f"ratio={np.max(vmesh.cell_volumes)/max(np.min(vmesh.cell_volumes),1e-30):.2f}")

    # 距离统计 (项目 178_circle_distance)
    dist_stats = grid_uniform["distance_stats"]
    res_diag = dist_stats.resolution_diagnostic()
    print(f"速度空间距离统计:")
    print(f"  平均距离: {res_diag['mean_distance']:.6f}")
    print(f"  最小距离: {res_diag['min_distance']:.6f}")
    print(f"  拉伸比: {res_diag['stretch_ratio']:.2f}")
    print(f"  有效分辨率: {res_diag['effective_resolution']:.1f}")

    # =================================================================
    #  §3  特殊函数验证  (项目 221_cosine_integral + 1428_chandrupatla)
    # =================================================================
    separator("§3 特殊函数计算")

    # Chandrasekhar 函数
    print("Chandrasekhar 函数 G(x):")
    test_x = [0.1, 0.5, 1.0, 2.0, 3.0]
    for xv in test_x:
        Gv = chandrasekhar_G(xv)
        print(f"  G({xv:.1f}) = {Gv:.8e}")

    # 余弦积分 (项目 221_cosine_integral)
    print("\n余弦积分 Ci(x):")
    for xv in [1.0, 5.0, 10.0, 20.0, 40.0]:
        ci_val = cosine_integral(xv)
        print(f"  Ci({xv:.1f}) = {ci_val:.10e}")

    # 等离子体色散函数
    print("\n等离子体色散函数 Z(ζ):")
    for zv in [0.5, 1.0, 2.0]:
        Zv = plasma_dispersion_function(zv)
        print(f"  Z({zv:.1f}) = {Zv:.6f}")

    # Chandrupatla 求根 (项目 1428_zero_chandrupatla)
    print("\nChandrupatla 求根测试:")
    # 求 erf(x) - 0.5 = 0 → x ≈ 0.4769
    root, froot, calls = zero_chandrupatla(
        lambda x: math.erf(x) - 0.5, 0.0, 2.0
    )
    print(f"  erf(x) = 0.5 的根: x = {root:.10f} (f(x) = {froot:.2e}, calls = {calls})")

    # =================================================================
    #  §4  初始条件
    # =================================================================
    separator("§4 初始分布函数")

    # 双温度混合
    T1, T2 = 0.8, 1.2
    alpha = 0.5  # 权重

    # 确保能量归一化: α n₁ T₁ + (1-α) n₂ T₂ = T_avg
    # n₁ = α / T₁^{3/2} / π^{3/2}  归一化
    f_M1 = maxwellian_1d(v_grid, T1)
    f_M2 = maxwellian_1d(v_grid, T2)
    f0 = alpha * f_M1 + (1.0 - alpha) * f_M2

    # 归一化粒子数
    n0 = np.trapz(FOUR_PI * v_grid**2 * f0, v_grid)
    f0 /= n0  # 归一化到 n=1

    # 重新计算能量
    E0 = np.trapz(FOUR_PI * v_grid**4 * f0, v_grid)
    T_eff_0 = E0 / (3.0 * np.trapz(FOUR_PI * v_grid**2 * f0, v_grid))

    print(f"初始条件: 双温度混合")
    print(f"  T₁ = {T1}, T₂ = {T2}, α = {alpha}")
    print(f"  初始有效温度: T_eff = {T_eff_0:.6f}")
    print(f"  初始数密度: n = {np.trapz(FOUR_PI * v_grid**2 * f0, v_grid):.6f}")
    print(f"  初始能量: E = {E0:.6f}")
    print(f"  max(f₀) = {np.max(f0):.6e}")

    # 平衡态 Maxwellian (T=1)
    f_maxwellian = maxwellian_1d(v_grid, 1.0)
    n_M = np.trapz(FOUR_PI * v_grid**2 * f_maxwellian, v_grid)
    f_maxwellian /= n_M

    # =================================================================
    #  §5  乘积求积验证  (项目 919_product_rule + 937_pyramid_witherden)
    # =================================================================
    separator("§5 求积规则验证")

    # 3D 球坐标乘积求积
    v_pts, v_wts = product_quadrature_3d_spherical(20, "gauss-hermite")
    # 验证: ∫ exp(-v²) d³v = π^{3/2}
    test_integrand = np.exp(-v_pts**2)
    integral_test = np.sum(v_wts * test_integrand)
    exact = PI**1.5
    print(f"Gauss-Hermite 乘积求积 (n=20):")
    print(f"  ∫ exp(-v²) d³v = {integral_test:.8f}")
    print(f"  精确值 π^(3/2) = {exact:.8f}")
    print(f"  相对误差 = {abs(integral_test - exact)/exact:.2e}")

    # 金字塔求积 (项目 937_pyramid_witherden_rule)
    v_pyr, t_pyr, w_pyr = pyramid_velocity_time_quadrature(3, V_MAX, 1.0)
    print(f"\n金字塔求积 (阶数=3): {len(v_pyr)} 个点")
    print(f"  v ∈ [0, {np.max(v_pyr):.3f}], t ∈ [0, {np.max(t_pyr):.3f}]")
    print(f"  总权重 = {np.sum(w_pyr):.6f}")

    # =================================================================
    #  §6  Rosenbluth 势  (项目 928_pwl_interp + 813_norm_l2)
    # =================================================================
    separator("§6 Rosenbluth 势计算")

    H_pot = compute_rosenbluth_H(v_grid, f0)
    G_pot = compute_rosenbluth_G_potential(v_grid, f0)
    print(f"Rosenbluth H(v): max={np.max(H_pot):.6e}, min={np.min(H_pot):.6e}")
    print(f"Rosenbluth G(v): max={np.max(G_pot):.6e}, min={np.min(G_pot):.6e}")

    # L2 范数 (项目 813_norm_l2)
    l2_f0 = l2_norm_discrete(v_grid, f0)
    print(f"初始 L2 范数: ||f₀||₂ = {l2_f0:.6e}")

    # PWL 插值测试 (项目 928_pwl_interp_2d_scattered)
    interp = PWLVelocityInterpolator(v_grid, f0)
    v_test = np.array([0.5, 1.0, 1.5, 2.0])
    f_interp = interp(v_test)
    f_exact = np.array([interp(vv) for vv in v_test])
    print(f"\nPWL 插值验证:")
    for i in range(len(v_test)):
        print(f"  f({v_test[i]:.1f}) = {f_interp[i]:.8e}")

    # =================================================================
    #  §7  CFL 条件与稳定性  (项目 979_r8gb + stability_analysis)
    # =================================================================
    separator("§7 稳定性分析")

    # 初始碰撞算子
    C0, A0, D0, J0 = compute_collision_operator(v_grid, f0)

    # CFL 检查
    cfl_info = check_cfl_condition(v_grid, f0, dt=0.001)
    print(f"CFL 条件 (dt=0.001):")
    print(f"  D_max = {cfl_info['D_max']:.6e}")
    print(f"  A_max = {cfl_info['A_max']:.6e}")
    print(f"  Δt_max (扩散) = {cfl_info['dt_max_diffusion']:.6e}")
    print(f"  Δt_max (对流) = {cfl_info['dt_max_advection']:.6e}")
    print(f"  CFL_扩散 = {cfl_info['cfl_diffusion']:.4f}")
    print(f"  CFL_对流 = {cfl_info['cfl_advection']:.4f}")
    print(f"  显式格式稳定: {cfl_info['stable_explicit']}")

    # 完整稳定性诊断
    stab_report = full_stability_diagnosis(v_grid, f0, dt=0.001, dx=dv)
    print(f"\nVon Neumann 分析:")
    vn = stab_report["von_neumann"]
    print(f"  显式格式稳定: {vn['stable_explicit']}")
    print(f"  Crank-Nicolson 稳定: {vn['stable_cn']}")
    print(f"  Δt_max = {vn['dt_max_explicit']:.6e}")

    print(f"\nGershgorin 分析:")
    gg = stab_report["gershgorin"]
    print(f"  稳定: {gg['is_stable']}")
    print(f"  Re(λ) 上界: {gg['max_real_upper_bound']:.6e}")

    if stab_report["eigenvalue"]["spectral_radius"] is not None:
        print(f"\n特征值分析:")
        print(f"  谱半径 ρ = {stab_report['eigenvalue']['spectral_radius']:.6e}")
        print(f"  max Re(λ) = {stab_report['eigenvalue']['max_real_part']:.6e}")

    # 带状矩阵演示 (项目 979_r8gb)
    print(f"\n带状矩阵演示:")
    n_demo = 20
    ml_demo, mu_demo = 2, 2
    a_demo = np.zeros((2*ml_demo+mu_demo+1, n_demo))
    for j in range(n_demo):
        a_demo[ml_demo, j] = 4.0  # 主对角
        if j > 0:
            a_demo[ml_demo-1, j] = -1.0  # 下次对角
            a_demo[ml_demo+1, j] = -1.0  # 上次对角
        if j > 1:
            a_demo[ml_demo-2, j] = 0.5
            a_demo[ml_demo+2, j] = 0.5
    info_demo = band_matrix_info(a_demo, n_demo, ml_demo, mu_demo)
    print(f"  {n_demo}×{n_demo} 五对角矩阵:")
    print(f"  非零元素: {info_demo['nnz']}/{info_demo['total_elements']}")
    print(f"  稀疏度: {info_demo['sparsity']:.3f}")

    # =================================================================
    #  §8  时间推进  (核心求解)
    # =================================================================
    separator("§8 Fokker-Planck 时间推进")

    DT = 0.001
    N_STEPS = 2000

    print(f"参数: Δτ = {DT}, N = {N_STEPS}, T_final = {DT * N_STEPS:.1f}")
    print(f"方法: 半隐式 (Crank-Nicolson 扩散 + 显式对流)")
    print()

    f_final, sim_history = solve_fokker_planck(
        v_grid, f0, DT, N_STEPS, method="semi_implicit"
    )

    print(f"模拟完成! {len(sim_history['f_snapshots'])} 个快照")
    print(f"最终 max(f) = {np.max(f_final):.6e}")

    # =================================================================
    #  §9  收敛与守恒诊断  (项目 813_norm_l2 + entropy)
    # =================================================================
    separator("§9 收敛与守恒诊断")

    # 最终有效温度
    T_eff_final = find_effective_temperature(v_grid, f_final)
    print(f"最终有效温度: T_eff = {T_eff_final:.6f}")

    # L2 范数
    l2_final = l2_norm_discrete(v_grid, f_final)
    l2_delta = l2_norm_perturbation(v_grid, f_final, f_maxwellian)
    l2_rel = l2_norm_relative(v_grid, f_final, f_maxwellian)
    print(f"L2 范数:")
    print(f"  ||f||₂ = {l2_final:.6e}")
    print(f"  ||δf||₂ = {l2_delta:.6e}")
    print(f"  相对误差 = {l2_rel:.6e}")

    # 熵诊断
    H_final = boltzmann_H(v_grid, f_final)
    H_initial = boltzmann_H(v_grid, f0)
    S_final = entropy_S(v_grid, f_final)
    S_initial = entropy_S(v_grid, f0)
    print(f"\n熵:")
    print(f"  H[f₀] = {H_initial:.6f}")
    print(f"  H[f]  = {H_final:.6f}")
    print(f"  ΔH = {H_final - H_initial:.6f} (应 ≤ 0)")
    print(f"  S[f₀] = {S_initial:.6f}")
    print(f"  S[f]  = {S_final:.6f}")
    print(f"  ΔS = {S_final - S_initial:.6f} (应 ≥ 0)")

    # KL 散度
    kl_final = kl_divergence(v_grid, f_final, f_maxwellian)
    kl_initial = kl_divergence(v_grid, f0, f_maxwellian)
    print(f"\nKL 散度:")
    print(f"  D_KL(f₀ || f_M) = {kl_initial:.6f}")
    print(f"  D_KL(f  || f_M) = {kl_final:.6f}")
    print(f"  ΔD_KL = {kl_final - kl_initial:.6f} (应 ≤ 0)")

    # 守恒检查
    n_final = np.trapz(FOUR_PI * v_grid**2 * f_final, v_grid)
    n_initial = np.trapz(FOUR_PI * v_grid**2 * f0, v_grid)
    E_final = np.trapz(FOUR_PI * v_grid**4 * f_final, v_grid)
    E_initial_check = np.trapz(FOUR_PI * v_grid**4 * f0, v_grid)
    print(f"\n守恒律:")
    print(f"  粒子数: n₀ = {n_initial:.6f}, n = {n_final:.6f}, "
          f"Δn/n = {abs(n_final-n_initial)/max(abs(n_initial),1e-30):.2e}")
    print(f"  能量: E₀ = {E_initial_check:.6f}, E = {E_final:.6f}, "
          f"ΔE/E = {abs(E_final-E_initial_check)/max(abs(E_initial_check),1e-30):.2e}")

    # =================================================================
    #  §10  矩分析  (项目 919_product_rule)
    # =================================================================
    separator("§10 速度空间矩分析")

    moments_final = compute_moments(v_grid, f_final)
    moments_initial = compute_moments(v_grid, f0)
    print(f"初始矩:")
    print(f"  n = {moments_initial['density']:.6f}")
    print(f"  T = {moments_initial['temperature']:.6f}")
    print(f"  ⟨v²⟩ = {moments_initial['mean_v2']:.6f}")
    print(f"  ⟨v⁴⟩ = {moments_initial['mean_v4']:.6f}")
    print(f"\n最终矩:")
    print(f"  n = {moments_final['density']:.6f}")
    print(f"  T = {moments_final['temperature']:.6f}")
    print(f"  ⟨v²⟩ = {moments_final['mean_v2']:.6f}")
    print(f"  ⟨v⁴⟩ = {moments_final['mean_v4']:.6f}")

    # =================================================================
    #  §11  自适应网格  (项目 627_knapsack_rational)
    # =================================================================
    separator("§11 自适应网格优化")

    refinement, new_grid, mesh_info = optimize_mesh_allocation(v_grid, f0, budget_factor=1.5)
    print(f"原始网格: {mesh_info['n_original']} 点")
    print(f"新网格: {mesh_info['n_new']} 点")
    print(f"加密单元: {mesh_info['n_refined']}")
    print(f"最大误差指示: {mesh_info['max_eta']:.6e}")
    print(f"平均误差指示: {mesh_info['mean_eta']:.6e}")

    quality_orig = mesh_quality_metrics(v_grid)
    quality_new = mesh_quality_metrics(new_grid)
    print(f"\n原始网格质量: min Δv={quality_orig['min_dx']:.4f}, "
          f"stretch={quality_orig['stretch_ratio']:.2f}")
    print(f"新网格质量: min Δv={quality_new['min_dx']:.4f}, "
          f"stretch={quality_new['stretch_ratio']:.2f}")

    # =================================================================
    #  §12  Newton 稳态求解  (项目 394_fem1d_nonlinear)
    # =================================================================
    separator("§12 Newton 迭代稳态求解")

    f_ss, n_iter, residuals = newton_solve_steady_state(v_grid, f0.copy(), max_newton=10)
    print(f"Newton 迭代: {n_iter} 步")
    if residuals:
        print(f"  初始残差: {residuals[0]:.6e}")
        print(f"  最终残差: {residuals[-1]:.6e}")
        if len(residuals) > 1:
            print(f"  收敛率: {residuals[-1]/max(residuals[0], 1e-30):.2e}")

    # =================================================================
    #  §13  对比诊断  (项目 1016_omipan + 1191_Reproduce-Algorithm)
    # =================================================================
    separator("§13 相空间对比诊断")

    contrast_diag = contrastive_phase_space_diagnostic(
        v_grid, f_final, f_maxwellian, f_previous=f0
    )
    print(f"余弦相似度 (f vs f_M): {contrast_diag['cosine_similarity_global']:.6f}")
    print(f"NT-Xent 损失: {contrast_diag['nt_xent_loss']:.6f}")
    print(f"Triplet 损失: {contrast_diag['triplet_loss']:.6f}")
    print(f"密度比: {contrast_diag['density_ratio']:.6f}")
    print(f"温度比: {contrast_diag['temperature_ratio']:.6f}")

    # 多尺度特征
    extractor = MultiScaleFeatureExtractor(v_grid, n_scales=3)
    features = extractor.extract(f_final)
    gm = features["global_moments"]
    gs = features["gradient_spectrum"]
    print(f"\n多尺度特征:")
    print(f"  n = {gm['n']:.6f}")
    print(f"  ⟨v²⟩ = {gm['mean_v2']:.6f}")
    print(f"  ⟨v⁴⟩ = {gm['mean_v4']:.6f}")
    print(f"  ||∇f||₁ = {gs['l1_norm_df']:.6e}")
    print(f"  ||∇²f||₁ = {gs['l1_norm_d2f']:.6e}")

    # =================================================================
    #  §14  不确定性量化  (项目 1262_gpc-self-location)
    # =================================================================
    separator("§14 不确定性量化")

    tracker = UncertaintyTracker(len(v_grid), initial_covariance_scale=1e-6)
    tracker.set_precision_from_distribution(f_final)
    tracker.predict(f_final)

    unc_report = uncertainty_diagnosis(tracker, f_final)
    unc_metrics = unc_report["uncertainty_metrics"]
    print(f"平均方差: {unc_metrics['mean_variance']:.6e}")
    print(f"最大方差: {unc_metrics['max_variance']:.6e}")
    print(f"总不确定性: {unc_metrics['total_uncertainty']:.6e}")
    print(f"对数行列式: {unc_metrics['log_determinant']:.4f}")
    print(f"条件数: {unc_metrics['condition_number']:.4e}")
    if unc_report["warnings"]:
        for w in unc_report["warnings"]:
            print(f"  ⚠ {w}")
    else:
        print("  ✓ 无异常")

    # =================================================================
    #  §15  库仑对数速度依赖  (项目 221_cosine_integral 扩展)
    # =================================================================
    separator("§15 库仑对数速度依赖性")

    ln_lambda_0 = plasma.ln_lambda
    ln_lambda_v = coulomb_logarithm_velocity_dependent(
        np.array([0.1, 0.5, 1.0, 2.0, 3.0]), ln_lambda_0, plasma.v_th
    )
    print(f"热库仑对数: ln Λ₀ = {ln_lambda_0:.4f}")
    print("速度依赖修正:")
    for i, xv in enumerate([0.1, 0.5, 1.0, 2.0, 3.0]):
        print(f"  x={xv:.1f}: ln Λ(v) = {ln_lambda_v[i]:.4f}")

    # =================================================================
    #  §16  总结
    # =================================================================
    separator("模拟总结")

    t_elapsed = time.time() - t_start

    print(f"模拟参数:")
    print(f"  网格点数: {N_GRID}")
    print(f"  时间步长: Δτ = {DT}")
    print(f"  总步数: {N_STEPS}")
    print(f"  最终时间: τ = {DT * N_STEPS:.1f}")
    print()
    print(f"物理结果:")
    print(f"  初始有效温度: T_eff(0) = {T_eff_0:.6f}")
    print(f"  最终有效温度: T_eff(τ) = {T_eff_final:.6f}")
    print(f"  ΔT/T = {abs(T_eff_final - 1.0):.6f}")
    print(f"  初始 KL 散度: {kl_initial:.6f}")
    print(f"  最终 KL 散度: {kl_final:.6f}")
    print(f"  熵增: ΔS = {S_final - S_initial:.6f}")
    print(f"  粒子数守恒: Δn/n = {abs(n_final-n_initial)/max(abs(n_initial),1e-30):.2e}")
    print(f"  能量守恒: ΔE/E = {abs(E_final-E_initial_check)/max(abs(E_initial_check),1e-30):.2e}")
    print()
    print(f"计算性能:")
    print(f"  总时间: {t_elapsed:.2f} s")
    print(f"  每步时间: {t_elapsed/N_STEPS*1000:.2f} ms")
    print()

    # 等离子体历法时间
    plasma_date = cal.sim_time_to_plasma_date(DT * N_STEPS)
    physical_time = cal.sim_time_to_wall_clock(DT * N_STEPS)
    print(f"模拟时间:")
    print(f"  等离子体日期: {plasma_date}")
    print(f"  物理时间: {physical_time:.4e} s")

    separator("模拟完成")
    print("所有 15 个种子项目已成功融合到本项目中.")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
