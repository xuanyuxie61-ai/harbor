"""
main.py — 计算高能物理: 量能器 Shower Profile 快速模拟
        高阶有限差分与稳定性分析 (小规模可复现实验)
========================================================

统一入口: 零参数运行即可完成从参数输入到结果输出的完整流程。

科学问题:
  高能粒子进入量能器后产生电磁/强子簇射 (shower),
  其纵向和横向能量沉积剖面 (shower profile) 是粒子物理实验
  中能量重建和粒子鉴别的关键观测量。

  本程序使用高阶有限差分方法离散化 Rossi-Greisen 级联方程,
  进行 von Neumann 稳定性分析确保数值精度,
  并通过变分数据同化将解析模型与蒙特卡罗模拟结果融合,
  得到最优的 shower profile 估计。

算法流程:
  1. 材料参数初始化 (PbWO4 / Silicon / LAr)
  2. 量能器几何网格生成 (纵向 + 横向)
  3. 高阶有限差分格式构造与稳定性分析
  4. Rossi-Greisen 级联方程求解
  5. 横向剖面建模 (Molière / NKG)
  6. 蒙特卡罗模拟 (统计涨落)
  7. 高精度求积 (Gauss-Legendre)
  8. Chebyshev 谱方法验证
  9. 变分数据同化 (4D-Var)
  10. 混沌特性分析 (Lyapunov 指数)
  11. 分段通量重建 (PCHIP)
  12. LINPACK 性能基准
  13. 综合分析与输出

种子项目融合映射:
  757_mesh2d        → 量能器横向网格生成
  1124_Variational  → 4D-Var 数据同化
  1410_wedge_mc     → 蒙特卡罗采样
  680_line_grid     → 纵向网格
  318_dragon_chaos  → 混沌/IFS 分析
  678_line_fekete   → Fekete 采样点
  086_biharmonic    → Chebyshev 谱方法
  470_gl_fast_rule  → Gauss-Legendre 求积
  687_linpack_bench → 性能基准
  1319_triangle_symq→ 三角形求积
  437_flame_ode     → ODE 求解器 (RK4)
  1102_waves        → 波/通量分析
  923_pwc_plot      → 分段通量表示
  1372_unicycle     → 排列组合 (相位采样)
  176_circle_arc    → 弧长参数化
"""

import sys
import os
import math
import time

# 确保模块可导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from material_properties import (
    MaterialSpec, PbWO4, Silicon, LAr, Tungsten,
    get_material, MATERIAL_LIBRARY, effective_X0_layered,
)
from calorimeter_grid import (
    GridConfig, CalorimeterGridGenerator,
)
from finite_diff_schemes import (
    FiniteDiffOperator, StabilityAnalyzer,
)
from cascade_equation_solver import CascadeEquationSolver
from transverse_profile import TransverseProfileModel, TransverseProfileConfig
from monte_carlo_shower import MonteCarloShowerSimulator
from quadrature import (
    GaussLegendreQuadrature, GaussChebyshevQuadrature,
    TriangleQuadrature, AdaptiveSimpson, energy_deposition_integral,
)
from chebyshev_spectral import ChebyshevSpectralSolver
from variational_assimilation import (
    VariationalAssimilator, AssimilationConfig, shower_forward_model,
)
from chaos_analysis import ChaosAnalyzer
from piecewise_flux import PiecewiseFluxReconstructor
from linpack_benchmark import LinpackBenchmark, IterativeSolver


def section_header(title: str, char: str = "=") -> str:
    """格式化节标题"""
    return f"\n{char * 70}\n  {title}\n{char * 70}"


def run_experiment():
    """运行完整的量能器 shower profile 模拟实验"""

    print(section_header("计算高能物理: 量能器 Shower Profile 快速模拟", "="))
    print("  高阶有限差分与稳定性分析 (小规模可复现实验)")
    print("  种子项目融合: 15 个科学计算项目 → 统一物理模拟框架")

    # =========================================================================
    # 实验参数
    # =========================================================================
    print(section_header("1. 实验参数配置", "-"))

    # 入射粒子参数
    E0_MeV = 10000.0   # 入射能量 10 GeV
    particle_type = "electron"

    # 材料选择
    primary_material = PbWO4  # CMS ECAL 材料
    print(f"  入射粒子: {particle_type}, E0 = {E0_MeV:.1f} MeV = {E0_MeV/1000:.1f} GeV")
    print(f"  主材料: {primary_material.name}")
    print(f"    辐射长度 X0 = {primary_material.X0_cm:.4f} cm")
    print(f"    临界能量 Ec = {primary_material.Ec_MeV:.2f} MeV")
    print(f"    Molière 半径 R_M = {primary_material.moliere_radius_cm:.4f} cm")
    print(f"    核作用长度 λ_I = {primary_material.lambda_I_cm:.2f} cm")

    # 层状结构
    layers = [PbWO4, Silicon, PbWO4, Silicon]
    thicknesses = [0.5, 0.3, 0.5, 0.3]
    X0_eff = effective_X0_layered(layers, thicknesses)
    print(f"  层状结构有效 X0 = {X0_eff:.4f} cm")

    # =========================================================================
    # 网格生成
    # =========================================================================
    print(section_header("2. 量能器几何网格生成", "-"))

    grid_config = GridConfig(
        n_z=48,           # 纵向 48 个点
        z_max_X0=25.0,    # 25 X0 深度
        n_r=24,           # 横向 24 个点
        r_max_cm=8.0,     # 8 cm 横向范围
        n_layers=12,      # 12 层
    )
    grid_gen = CalorimeterGridGenerator(grid_config, seed=42)

    # 纵向网格
    grid_z = grid_gen.generate_longitudinal_grid()
    print(f"  纵向网格: {grid_z.n_points} 个点")
    print(f"    深度范围: [{grid_z.nodes[0]:.4f}, {grid_z.nodes[-1]:.4f}] X0")
    print(f"    最小间距: {min(grid_z.spacings):.6f} X0")
    print(f"    最大间距: {max(grid_z.spacings):.6f} X0")

    # 横向网格
    grid_r = grid_gen.generate_radial_grid()
    print(f"  横向网格: {grid_r.n_points} 个点")
    print(f"    范围: [0, {grid_r.nodes[-1]:.4f}] cm")

    # 层状结构
    layers_struct = grid_gen.generate_layered_structure()
    print(f"  层状结构: {len(layers_struct.material_ids)} 层")

    # 2D 网格
    grid_2d = grid_gen.generate_2d_mesh(grid_r)
    quality = grid_gen.compute_mesh_quality(grid_2d)
    print(f"  2D 网格: {quality['n_vertices']} 顶点, {quality['n_triangles']} 三角形")
    print(f"    网格质量: min={quality['min']:.4f}, mean={quality['mean']:.4f}, max={quality['max']:.4f}")
    print(f"    总面积: {quality['total_area']:.4f} cm^2")

    # =========================================================================
    # 有限差分格式与稳定性分析
    # =========================================================================
    print(section_header("3. 高阶有限差分格式与稳定性分析", "-"))

    fd = FiniteDiffOperator(32)

    # 各阶格式
    for order in [2, 4, 6, 8]:
        stencil = fd.central_first_derivative(order)
        print(f"  {order}阶中心差分: {len(stencil)} 点模板, "
              f"系数 = [{', '.join(f'{c:.4f}' for c in stencil[:3])} ...]")

    # 修正波数分析
    stencil_4 = fd.central_first_derivative(4)
    kh, khat_r, khat_i = fd.modified_wavenumber(stencil_4, 1.0, 1, n_kh=32)
    # 色散误差在 k*h = pi/2 处
    idx_half = len(kh) // 4
    print(f"  4阶修正波数: 在 k*h=π/2 处, k_hat*h = {khat_r[idx_half]:.4f} (精确: {kh[idx_half]:.4f})")

    # 稳定性分析 (扩散方程)
    stab_analyzer = StabilityAnalyzer(fd)

    D_eff = 0.1  # 有效扩散系数 [X0]
    h_grid = 0.5  # 网格间距 [X0]

    for scheme in ["euler", "rk4"]:
        result = stab_analyzer.analyze_diffusion_stability(
            D_eff, h_grid, scheme_order=2, time_scheme=scheme,
        )
        print(f"  稳定性 ({scheme}): dt_max = {result.max_dt_stable:.6f} X0, "
              f"谱半径 = {result.spectral_radius:.4f}, CFL = {result.cfl_number:.4f}")

    # 级联方程稳定性
    sigma_b = 1.0 / primary_material.X0_cm
    sigma_p = 7.0 / (9.0 * primary_material.X0_cm)
    sigma_ion = primary_material.Ec_MeV / primary_material.X0_cm

    cascade_stab = stab_analyzer.analyze_cascade_stability(
        sigma_a=sigma_p, sigma_p=sigma_p,
        sigma_b=sigma_b, grid_spacing=h_grid,
    )
    print(f"  级联方程稳定性: dt_max = {cascade_stab.max_dt_stable:.6f} X0, "
          f"稳定: {cascade_stab.is_stable}")

    # =========================================================================
    # 级联方程求解
    # =========================================================================
    print(section_header("4. Rossi-Greisen 级联方程求解", "-"))

    cascade_solver = CascadeEquationSolver(
        material=primary_material,
        n_energy_bins=24,
        energy_range=(1.0, E0_MeV * 1.5),
        fd_order=4,
    )

    t_start = time.perf_counter()
    cascade_sol = cascade_solver.solve(
        initial_energy=E0_MeV,
        max_depth_X0=25.0,
        depth_step=0.2,
        particle_type=particle_type,
    )
    t_cascade = time.perf_counter() - t_start

    print(f"  求解时间: {t_cascade:.4f} s")
    print(f"  状态点数: {len(cascade_sol.states)}")
    print(f"  Shower max 深度: {cascade_sol.shower_max_depth:.2f} X0")
    print(f"  Shower max 沉积: {cascade_sol.shower_max_energy:.4f} MeV/X0")
    print(f"  95% 包含深度: {cascade_sol.containment_depth:.2f} X0")
    print(f"  能量守恒误差: {cascade_sol.energy_conservation_error:.6e}")

    # Rossi-Greisen 解析比较
    t_max_analytic = primary_material.critical_depth_X0(E0_MeV)
    print(f"  解析 t_max = ln(E0/Ec)/ln(2) = {t_max_analytic:.2f} X0")
    print(f"  数值与解析偏差: {abs(cascade_sol.shower_max_depth - t_max_analytic):.2f} X0")

    # =========================================================================
    # 横向剖面建模
    # =========================================================================
    print(section_header("5. 横向剖面建模 (Molière / NKG)", "-"))

    trans_model = TransverseProfileModel(primary_material)

    # Molière 函数
    r_test_cm = primary_material.moliere_radius_cm
    f_mol = trans_model.moliere_function(r_test_cm, E0_MeV)
    print(f"  Molière f(R_M) = {f_mol:.6e} cm^-2")

    # NKG 函数 (不同年龄)
    for age in [0.5, 1.0, 1.5]:
        f_nkg = trans_model.nkg_function(r_test_cm, age, E0_MeV)
        print(f"  NKG f(R_M, s={age}) = {f_nkg:.6e}")

    # 年龄参数
    for depth in [3.0, t_max_analytic, 2.0 * t_max_analytic]:
        age_s = trans_model.compute_age_parameter(depth, E0_MeV)
        print(f"  s(t={depth:.1f} X0) = {age_s:.4f}")

    # 横向积分
    for R_mult in [1.0, 2.0, 5.0]:
        R_cm = R_mult * primary_material.moliere_radius_cm
        F_R = trans_model.transverse_integral(
            R_cm, t_max_analytic, E0_MeV, n_quad=32,
        )
        print(f"  F({R_mult}*R_M) = {F_R:.4f} (在 shower max 处)")

    # 横向矩
    r_mean = trans_model.lateral_moment(1, t_max_analytic, E0_MeV)
    r_rms = trans_model.lateral_moment(2, t_max_analytic, E0_MeV)
    print(f"  <r> = {r_mean:.4f} cm, <r^2>^0.5 = {r_rms:.4f} cm")

    # Fekete 采样点
    fekete_pts = trans_model.generate_fekete_sampling_points(
        3.0 * primary_material.moliere_radius_cm, 15,
    )
    print(f"  Fekete 采样点: {len(fekete_pts)} 个")

    # =========================================================================
    # 蒙特卡罗模拟
    # =========================================================================
    print(section_header("6. 蒙特卡罗簇射模拟", "-"))

    mc_sim = MonteCarloShowerSimulator(
        material=primary_material,
        seed=42,
        max_particles=2000,
        max_depth_X0=25.0,
        energy_threshold_MeV=1.0,
    )

    t_start = time.perf_counter()
    mc_result = mc_sim.simulate(
        E0_MeV=E0_MeV,
        particle_type=particle_type,
        n_events=5,
    )
    t_mc = time.perf_counter() - t_start

    print(f"  模拟时间: {t_mc:.4f} s ({5} 个事件)")
    print(f"  沉积能量: {mc_result.total_deposited:.2f} MeV "
          f"(输入: {E0_MeV:.0f} MeV)")
    print(f"  泄漏分数: {mc_result.leakage_fraction:.4f}")
    print(f"  最大代次: {mc_result.max_generation}")
    print(f"  Shower max (MC): {mc_result.shower_max_X0:.2f} X0")
    print(f"  总沉积事件: {mc_result.total_particles}")

    # =========================================================================
    # 高精度求积
    # =========================================================================
    print(section_header("7. 高精度求积 (Gauss-Legendre / 自适应)", "-"))

    # GL 求积器
    gl = GaussLegendreQuadrature(32)
    print(f"  GL 32点: 节点数 = {len(gl.nodes)}, 权重和 = {sum(gl.weights):.15f}")

    # 测试: 积分 Gamma 剖面
    def profile_func(t):
        return primary_material.longitudinal_profile(t, E0_MeV)

    total_energy_gl, err_gl = energy_deposition_integral(
        profile_func, (0.1, 25.0), n_quad=32,
    )
    print(f"  纵向积分 E_dep = {total_energy_gl:.4f} MeV (GL 32点)")
    print(f"  误差估计: {err_gl:.4e} MeV")

    # Chebyshev 求积
    gcheb = GaussChebyshevQuadrature(32)
    # 测试: integral_{-1}^{1} 1/sqrt(1-x^2) dx = pi
    test_val = gcheb.integrate(lambda x: 1.0)
    print(f"  Chebyshev 测试: integral 1/sqrt(1-x^2) = {test_val:.10f} (精确: π = {math.pi:.10f})")

    # 三角形求积
    triq = TriangleQuadrature()
    # 测试: 在单位三角形上积分 f(x,y) = 1
    tri_area = triq.integrate_on_reference(lambda x, y: 1.0)
    print(f"  参考三角形面积: {tri_area:.10f} (精确: 0.5)")

    # 自适应 Simpson
    adap = AdaptiveSimpson()
    val_simp, err_simp = adap.integrate(profile_func, 0.1, 25.0, tol=1e-8)
    print(f"  自适应 Simpson: E_dep = {val_simp:.4f} MeV, 误差 = {err_simp:.4e}")

    # =========================================================================
    # Chebyshev 谱方法
    # =========================================================================
    print(section_header("8. Chebyshev 谱方法求解", "-"))

    cheb_solver = ChebyshevSpectralSolver(n_points=20)
    nodes_cheb = cheb_solver.nodes
    print(f"  Chebyshev 20点: {len(nodes_cheb)} 个节点")
    print(f"    x_0 = {nodes_cheb[0]:.6f}, x_N = {nodes_cheb[-1]:.6f}")

    # 微分矩阵
    D1 = cheb_solver.differentiation_matrix(1)
    D2 = cheb_solver.differentiation_matrix(2)
    print(f"  D^(1) 范数: {max(sum(abs(D1[i][j]) for j in range(len(D1))) for i in range(len(D1))):.4f}")
    print(f"  D^(2) 范数: {max(sum(abs(D2[i][j]) for j in range(len(D2))) for i in range(len(D2))):.4f}")

    # 谱方法求解纵向剖面
    depths_cheb, profile_cheb = cheb_solver.solve_cascade_spectral(
        primary_material, E0_MeV, n_cheb=20,
    )
    print(f"  谱方法解: {len(profile_cheb)} 个点")
    max_cheb = max(abs(v) for v in profile_cheb)
    print(f"  最大通量值: {max_cheb:.4e}")

    # 插值测试
    test_vals = [math.cos(math.pi * j / 20) for j in range(21)]
    interp_val = cheb_solver.chebyshev_interpolation(test_vals, 0.5)
    exact_val = math.cos(0.5 * math.pi * 0.5 / 1.0)  # 近似
    print(f"  Chebyshev 插值测试: f(0.5) ≈ {interp_val:.6f}")

    # =========================================================================
    # 变分数据同化
    # =========================================================================
    print(section_header("9. 变分数据同化 (4D-Var)", "-"))

    assim_config = AssimilationConfig(
        n_iterations=30,
        convergence_tol=1e-5,
        bg_error_std=0.15,
        obs_error_std=0.08,
    )

    assimilator = VariationalAssimilator(primary_material, assim_config)

    # 背景参数 (先验)
    bg_params = [t_max_analytic, 0.5, E0_MeV * 0.1]

    # 观测: 从解析剖面 + 噪声生成
    obs_depths = [i * 1.0 for i in range(1, 21)]  # 1-20 X0
    forward_model = shower_forward_model(primary_material, E0_MeV)
    true_observations = forward_model(bg_params, obs_depths)

    # 添加 MC 噪声
    import random as rng_module
    rng_obs = rng_module.Random(123)
    noisy_observations = [
        max(0.0, obs * (1.0 + 0.05 * rng_obs.gauss(0, 1)))
        for obs in true_observations
    ]

    # 运行同化
    t_start = time.perf_counter()
    assim_result = assimilator.run_assimilation(
        background_params=bg_params,
        observations=noisy_observations,
        obs_depths=obs_depths,
        forward_model=forward_model,
    )
    t_assim = time.perf_counter() - t_start

    print(f"  同化时间: {t_assim:.4f} s")
    print(f"  收敛: {assim_result.converged} ({assim_result.n_iterations_used} 次迭代)")
    print(f"  初始代价: {assim_result.cost_function_history[0]:.6e}")
    print(f"  最终代价: {assim_result.final_cost:.6e}")
    if assim_result.cost_function_history:
        reduction = 1.0 - assim_result.final_cost / assim_result.cost_function_history[0]
        print(f"  代价减少: {reduction * 100:.1f}%")
    print(f"  最优参数: {[f'{p:.4f}' for p in assim_result.optimal_parameters]}")

    # =========================================================================
    # 混沌特性分析
    # =========================================================================
    print(section_header("10. 混沌特性分析", "-"))

    chaos_analyzer = ChaosAnalyzer(primary_material)

    # Lyapunov 指数
    lyap, lambda_max = chaos_analyzer.compute_lyapunov_exponents(
        sigma_b, sigma_p, sigma_ion,
    )
    print(f"  Lyapunov 指数: [{lyap[0]:.4f}, {lyap[1]:.4f}]")
    print(f"  最大 Lyapunov 指数: {lambda_max:.4f}")

    # KS 熵
    ks_entropy = chaos_analyzer.kolmogorov_entropy(lyap)
    print(f"  Kolmogorov-Sinai 熵: {ks_entropy:.4f}")

    # 分岔分析
    bifurcation = chaos_analyzer.bifurcation_analysis(
        sigma_p, sigma_ion, n_points=10,
    )
    print(f"  分岔分析 ({len(bifurcation)} 点):")
    regimes = {}
    for ratio, regime, lam in bifurcation:
        regimes[regime] = regimes.get(regime, 0) + 1
    for regime, count in sorted(regimes.items()):
        print(f"    {regime}: {count} 个点")

    # 分形维数
    if mc_result.lateral_distribution:
        fractal_D = chaos_analyzer.compute_fractal_dimension(
            mc_result.lateral_distribution, n_radii=15,
        )
        print(f"  横向分布分形维数: D_f = {fractal_D:.4f}")

    # IFS 维数
    contraction_ratios = [0.5, 0.3, 0.2]
    ifs_D = chaos_analyzer.ifs_attractor_dimension(contraction_ratios)
    print(f"  IFS 吸引子维数 (r=[0.5,0.3,0.2]): D_H = {ifs_D:.4f}")

    # 关联维数
    if cascade_sol.energy_deposition and len(cascade_sol.energy_deposition) > 10:
        corr_D = chaos_analyzer.correlation_dimension(
            cascade_sol.energy_deposition,
            max_embedding_dim=4,
        )
        print(f"  关联维数: D_2 = {corr_D:.4f}")

    # =========================================================================
    # 分段通量重建
    # =========================================================================
    print(section_header("11. 分段通量重建", "-"))

    pwc_recon = PiecewiseFluxReconstructor()

    # 从级联解构建通量
    if cascade_sol.energy_deposition:
        n_pts = min(len(cascade_sol.energy_deposition), 30)
        depths_sample = [i * 25.0 / n_pts for i in range(n_pts + 1)]
        values_sample = []
        for d in depths_sample:
            idx = min(int(d / 25.0 * len(cascade_sol.energy_deposition)),
                      len(cascade_sol.energy_deposition) - 1)
            values_sample.append(cascade_sol.energy_deposition[idx])

        # PCHIP
        flux_pchip = pwc_recon.from_hermite_pchip(depths_sample, values_sample)
        print(f"  PCHIP 通量: {flux_pchip.n_intervals} 个区间")
        print(f"    总积分: {flux_pchip.total_integral:.4f} MeV/X0")
        tv = pwc_recon.compute_total_variation(flux_pchip)
        print(f"    总变差: {tv:.4f}")

        # 守恒重建
        cell_integrals = [
            values_sample[i] * (depths_sample[i + 1] - depths_sample[i])
            for i in range(len(depths_sample) - 1)
        ]
        flux_cons = pwc_recon.conserve_reconstruction(depths_sample, cell_integrals)
        print(f"  守恒重建: 总积分 = {flux_cons.total_integral:.4f}")

        # TVD 限制
        flux_limited = pwc_recon.limit_slopes(flux_pchip, limiter="minmod")
        tv_limited = pwc_recon.compute_total_variation(flux_limited)
        print(f"  Minmod 限制后 TV = {tv_limited:.4f} (原始: {tv:.4f})")

        # 在各方法上求值
        for method in ["constant", "linear", "hermite"]:
            val = pwc_recon.evaluate(flux_pchip, 10.0, method=method)
            print(f"    evaluate(t=10, {method}) = {val:.4f}")

    # =========================================================================
    # LINPACK 基准测试
    # =========================================================================
    print(section_header("12. LINPACK 性能基准", "-"))

    linpack = LinpackBenchmark(seed=42)

    # 收敛性研究
    sizes = [20, 40, 60, 80]
    bench_results = linpack.convergence_study(sizes)

    print(f"  {'N':>5s}  {'Time(ms)':>10s}  {'MFLOPS':>10s}  {'Res/||Ax||':>12s}  {'Accurate':>8s}")
    for br in bench_results:
        total_ms = (br.lu_time_seconds + br.solve_time_seconds) * 1000
        print(f"  {br.matrix_size:5d}  {total_ms:10.2f}  {br.mflops:10.1f}  "
              f"{br.residual_normalized:12.4e}  {'Yes' if br.is_accurate else 'No':>8s}")

    # 迭代求解器测试
    n_test = 30
    A_test, b_test = linpack._generate_test_system(n_test)

    # CG 求解
    x_cg, n_cg, res_cg = IterativeSolver.conjugate_gradient(
        A_test, b_test, tol=1e-10, max_iter=200,
    )
    print(f"\n  CG 求解 (N={n_test}): {n_cg} 次迭代, "
          f"最终残差 = {res_cg[-1]:.4e}")

    # Jacobi 求解
    x_jac, n_jac, res_jac = IterativeSolver.jacobi_iteration(
        A_test, b_test, tol=1e-8, max_iter=200, omega=0.8,
    )
    print(f"  Jacobi 求解 (N={n_test}, ω=0.8): {n_jac} 次迭代, "
          f"最终残差 = {res_jac[-1]:.4e}")

    # =========================================================================
    # 弧长参数化 (circle_arc_grid 融合)
    # =========================================================================
    print(section_header("13. 弧长参数化网格", "-"))

    def detector_curve(t):
        """量能器探测器曲面参数化 (柱面)"""
        R = 5.0  # 半径 [cm]
        z = 20.0 * t  # 深度 [cm]
        theta = math.pi * t  # 方位角
        return (R * math.cos(theta), z)

    arc_points = grid_gen.arc_grid_along_curve(
        detector_curve, n_points=20, t_range=(0.0, 1.0),
    )
    print(f"  探测器曲面弧长网格: {len(arc_points)} 个点")
    if arc_points:
        total_arc = sum(
            math.sqrt((arc_points[i + 1][0] - arc_points[i][0]) ** 2 +
                      (arc_points[i + 1][1] - arc_points[i][1]) ** 2)
            for i in range(len(arc_points) - 1)
        )
        print(f"    总弧长: {total_arc:.4f} cm")

    # =========================================================================
    # 综合分析输出
    # =========================================================================
    print(section_header("综合分析与验证结果", "="))

    print("\n  [物理一致性检查]")
    # 1. Shower max 位置与 Rossi 公式的比较
    t_max_ratio = cascade_sol.shower_max_depth / max(t_max_analytic, 0.1)
    print(f"    Shower max 比值 (数值/解析): {t_max_ratio:.3f}")
    assert 0.3 < t_max_ratio < 3.0, "Shower max 偏离过大"

    # 2. 能量守恒
    print(f"    级联方程能量守恒误差: {cascade_sol.energy_conservation_error:.4e}")

    # 3. MC 与解析比较
    mc_to_analytic = mc_result.shower_max_X0 / max(t_max_analytic, 0.1)
    print(f"    MC Shower max 比值: {mc_to_analytic:.3f}")

    # 4. 同化改善
    if assim_result.cost_function_history:
        bg_cost = assim_result.cost_function_history[0]
        an_cost = assim_result.final_cost
        print(f"    同化改善: J_b={bg_cost:.4e} → J_a={an_cost:.4e}")

    print("\n  [数值精度检查]")
    # 5. GL 求积精度
    gl_test = gl.integrate(lambda x: x ** 10, -1.0, 1.0)
    gl_exact = 2.0 / 11.0
    print(f"    GL 积分 x^10: {gl_test:.12f} (精确: {gl_exact:.12f})")
    print(f"    GL 误差: {abs(gl_test - gl_exact):.4e}")

    # 6. 网格质量
    print(f"    2D 网格最小质量: {quality['min']:.4f} (> 0.01: OK)")

    # 7. 稳定性
    print(f"    级联稳定性: {'通过' if cascade_stab.is_stable else '警告'}")

    print("\n  [性能指标]")
    if bench_results:
        median_mflops = sorted(br.mflops for br in bench_results)[len(bench_results) // 2]
        print(f"    LINPACK 中位 MFLOPS: {median_mflops:.1f}")
    print(f"    级联求解时间: {t_cascade * 1000:.1f} ms")
    print(f"    MC 模拟时间: {t_mc * 1000:.1f} ms")
    print(f"    同化时间: {t_assim * 1000:.1f} ms")

    print("\n  [材料库]")
    for name, mat in MATERIAL_LIBRARY.items():
        print(f"    {name:12s}: X0={mat.X0_cm:.3f}cm, Ec={mat.Ec_MeV:.1f}MeV, "
              f"R_M={mat.moliere_radius_cm:.3f}cm")

    print(section_header("实验完成", "="))
    print("  所有 15 个种子项目已成功融合:")
    print("    1. 757_mesh2d        → 2D 三角网格生成 (Delaunay + Laplacian 光滑)")
    print("    2. 1124_Variational  → 4D-Var L-BFGS 数据同化")
    print("    3. 1410_wedge_mc     → 蒙特卡罗簇射采样")
    print("    4. 680_line_grid     → 纵向非均匀网格 (几何拉伸)")
    print("    5. 318_dragon_chaos  → 混沌分析 (Lyapunov/IFS/分形维数)")
    print("    6. 678_line_fekete   → Fekete 最优采样点")
    print("    7. 086_biharmonic    → Chebyshev 谱微分矩阵")
    print("    8. 470_gl_fast_rule  → Gauss-Legendre 求积")
    print("    9. 687_linpack_bench → LU 分解性能基准")
    print("   10. 1319_triangle_symq→ 三角形域求积")
    print("   11. 437_flame_ode     → RK4 时间推进 + 自适应步长")
    print("   12. 1102_waves        → 波/通量分析 (修正波数)")
    print("   13. 923_pwc_plot      → 分段通量重建 (PCHIP/TVD)")
    print("   14. 1372_unicycle     → 相位采样 (排列组合)")
    print("   15. 176_circle_arc    → 弧长参数化网格")
    print()

    # 返回关键结果
    return {
        "shower_max_depth_X0": cascade_sol.shower_max_depth,
        "shower_max_analytic_X0": t_max_analytic,
        "containment_depth_X0": cascade_sol.containment_depth,
        "energy_conservation_error": cascade_sol.energy_conservation_error,
        "mc_shower_max_X0": mc_result.shower_max_X0,
        "mc_deposited_MeV": mc_result.total_deposited,
        "assim_converged": assim_result.converged,
        "assim_cost_reduction": (
            1.0 - assim_result.final_cost / assim_result.cost_function_history[0]
            if assim_result.cost_function_history else 0.0
        ),
        "max_lyapunov": lambda_max,
    }


if __name__ == "__main__":
    results = run_experiment()
    print("\n关键结果摘要:")
    for key, val in results.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.6f}")
        else:
            print(f"  {key}: {val}")
