"""
main.py — 全身 PBPK（Physiologically Based Pharmacokinetic）模型统一入口
==============================================================================
项目：PROJECT_126 博士级 Python 科研代码合成
领域：生物医学 — 药物代谢动力学 PBPK 模型

本程序实现一个面向前沿科学问题的全身多器官 PBPK 建模与仿真系统，
融合 15 个种子项目的核心算法，零参数可运行。

核心科学问题：
    基于多尺度随机微分方程、稀疏网格不确定性量化、刚性 ODE 系统与
    动态规划剂量优化的全身药物分布-代谢-毒性一体化建模。

运行方式：
    python main.py
"""

import numpy as np
import time

# ---------------------------------------------------------------------------
# 导入各模块
# ---------------------------------------------------------------------------
from pbpk_special_functions import (
    carlson_rf, carlson_rd, carlson_rc,
    jacobi_sncndn, gauss_agm, jacobi_theta,
    hyper_2f1, drug_protein_binding_fraction, effective_diffusion_coefficient
)
from pbpk_random import (
    r8vec_uniform_01_sorted_exponential, r8vec_uniform_01_sorted_product,
    normal_01_cdf_inv, r8vec_normal_01_sorted,
    sample_drug_arrival_times, sample_physiological_params
)
from pbpk_quadrature import (
    cc_abscissa, cc_weights, _generate_sparse_grid_direct,
    square_felippa_rule, square_monomial_integral,
    sparse_grid_integrate, integrate_organ_slice
)
from pbpk_diffusion import (
    laplacian_1d_dd, laplacian_1d_dn, laplacian_1d_nd,
    laplacian_1d_nn, laplacian_1d_pp,
    laplacian_dd_eigenvalues, laplacian_dd_eigenvectors,
    laplacian_apply, laplacian_3d_tensor,
    solve_steady_state_diffusion, solve_tissue_concentration_profile
)
from pbpk_stochastic import (
    random_walk_3d_step, inside_ellipsoid,
    feynman_kac_3d_monte_carlo, feynman_kac_1d,
    drug_absorption_probability_organ, organ_hitting_probability
)
from pbpk_ode_solver import (
    tough_deriv, tough_exact,
    rk4_step, implicit_trapezoidal_step, rosenbrock_step,
    solve_ode, PBPK_ODE_System, solve_pbpk_ode
)
from pbpk_geometry import (
    ellipsoid_tri_surface, mesh_surface_area, mesh_volume_tetrahedral,
    cvt_ellipsoid_lloyd, inside_ellipsoid_points,
    build_organ_geometries, compute_organ_distances, ORGAN_DEFS
)
from pbpk_polynomials import (
    rosenbrock, himmelblau, camel_back,
    binding_potential_surface, enzyme_kinetics_polynomial_substrate,
    multi_target_objective, sobol_g_function
)
from pbpk_interpolation import (
    runge_function, bernstein_example, oscillatory_function,
    piecewise_composite, lagrange_interpolate, chebyshev_nodes,
    piecewise_linear_interpolate,
    pd_effect_interpolate, build_pd_curve_from_hill, pharmacodynamic_response
)
from pbpk_optimization import (
    knapsack_01, optimize_dose_allocation,
    optimize_dosing_schedule, optimize_drug_combination
)
from pbpk_utils import (
    luhn_checksum, luhn_is_valid, validate_patient_id,
    concentration_binning, summary_statistics,
    safe_divide, safe_log, safe_exp, softplus, clip_concentration,
    PHYSIOLOGICAL_CONSTANTS, scale_by_body_weight
)


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def section_1_special_functions():
    """
    模块 1：特殊函数与药物-蛋白结合
    融合种子项目：327_elfun + 551_hyper_2f1
    """
    print_section("1. 特殊函数：Carlson 椭圆积分 / Jacobi 函数 / 超几何函数")

    # Carlson RF(1,2,0)
    rf_val = carlson_rf(1.0, 2.0, 0.0)
    print(f"  Carlson RF(1,2,0) = {rf_val:.10f}")

    # Jacobi sn/cn/dn
    sn, cn, dn = jacobi_sncndn(0.5, 0.5)
    verify = sn ** 2 + cn ** 2  # 应 ≈ 1
    print(f"  Jacobi sn(0.5|0.5)={sn:.10f}, cn={cn:.10f}, dn={dn:.10f}")
    print(f"  Identity check sn^2+cn^2 = {verify:.10f}")

    # AGM
    agm_val, iters = gauss_agm(1.0, np.sqrt(2.0))
    print(f"  AGM(1,√2) = {agm_val:.10f}  (iterations={iters})")

    # 超几何函数 2F1
    hf = hyper_2f1(1.0, 2.0, 3.0, 0.5)
    print(f"  2F1(1,2;3;0.5) = {hf:.10f}")

    # 药物-蛋白结合分数
    K_a = 1e5  # L/mol
    C_p = 1e-6  # mol/L
    fu = drug_protein_binding_fraction(K_a, C_p, n_sites=1)
    print(f"  Free fraction (K_a={K_a:.0e}, C_p={C_p:.0e}): fu = {fu:.6f}")

    # 非均质组织有效扩散系数（AGM 修正）
    D_parallel = 1e-9   # m^2/s
    D_perpendicular = 1e-10
    theta = np.pi / 4.0
    D_eff = effective_diffusion_coefficient(D_parallel, D_perpendicular, theta)
    print(f"  Effective diffusion (anisotropic, θ=π/4): D_eff = {D_eff:.3e} m^2/s")


def section_2_random_sampling():
    """
    模块 2：随机采样与顺序统计量
    融合种子项目：1007_random_sorted
    """
    print_section("2. 随机采样：均匀/正态顺序统计量 & 药物到达时间")

    n = 20
    u1 = r8vec_uniform_01_sorted_exponential(n)
    u2 = r8vec_uniform_01_sorted_product(n)
    print(f"  Uniform sorted (exp) first 5: {u1[:5]}")
    print(f"  Uniform sorted (prod) first 5: {u2[:5]}")

    z = r8vec_normal_01_sorted(n, method="exponential")
    print(f"  Normal sorted first 5: {z[:5]}")

    # 药物分子到达时间
    arrival = sample_drug_arrival_times(200, mean_interval=10.0, cv=0.3)
    print(f"  Drug arrival times (first 5): {arrival[:5]}")
    print(f"  Total absorption time: {arrival[-1]:.1f} s")

    # 生理参数 Latin-Hypercube 风格采样
    params = sample_physiological_params(
        100,
        np.array([70.0, 1.2, 0.05]),      # mean: BW, CO, fu
        np.array([0.15, 0.20, 0.30])      # CV
    )
    stats = summary_statistics(params[:, 0])
    print(f"  Physiological param sampling: mean BW={stats['mean']:.2f} kg, cv={stats['cv']:.3f}")


def section_3_quadrature():
    """
    模块 3：稀疏网格与二维求积
    融合种子项目：1103_sparse_grid_cc + 1144_square_felippa_rule
    """
    print_section("3. 高维求积：Smolyak 稀疏网格 CC & Felippa 乘积规则")

    # 验证 1D CC 权重和为 2
    w = cc_weights(3)
    print(f"  CC level 3 sum(weights) = {w.sum():.10f} (target=2.0)")

    # 2D 稀疏网格积分 ∫_{[-1,1]^2} x^2 y^2 dx dy = 16/81 * 9 ???
    # 实际 ∫_{-1}^1 x^2 dx = 2/3, 所以二维 = 4/9
    result = sparse_grid_integrate(lambda p: p[0]**2 * p[1]**2, 2, 3)
    print(f"  Sparse grid ∫ x^2 y^2 dA = {result:.10f} (exact = {4.0/9.0:.10f})")

    # 三维测试 ∫ 1 = 8
    result3 = sparse_grid_integrate(lambda p: 1.0, 3, 2)
    print(f"  Sparse grid ∫ 1 dV over [-1,1]^3 = {result3:.6f} (target=8.0)")

    # Felippa 规则在器官切片上积分
    xn, yn, wn = square_felippa_rule(0.0, 1.0, 0.0, 1.0, 5)
    integral = np.sum(wn)
    print(f"  Felippa 5-pt rule over unit square = {integral:.10f} (target=1.0)")

    # 器官切片药物浓度积分示例
    def conc_example(x, y):
        return 100.0 * np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.1)
    organ_integral = integrate_organ_slice(conc_example, (0.0, 1.0), (0.0, 1.0), order=5)
    print(f"  Organ slice concentration integral = {organ_integral:.4f} (mg·cm^2/L)")


def section_4_diffusion():
    """
    模块 4：Laplacian 离散化与组织扩散
    融合种子项目：648_laplacian_matrix
    """
    print_section("4. 组织扩散：有限差分 Laplacian & 稳态浓度分布")

    n = 30
    L = 0.01  # 10 mm 组织
    A_dd = laplacian_1d_dd(n, L)
    lam = laplacian_dd_eigenvalues(n, L)
    V = laplacian_dd_eigenvectors(n)
    # 验证特征分解
    err = np.linalg.norm(A_dd @ V - V @ np.diag(lam))
    print(f"  DD Laplacian eigen-decomposition error: {err:.4e}")

    # 验证 NN 矩阵奇异
    A_nn = laplacian_1d_nn(n, L)
    det_nn = np.linalg.det(A_nn)
    print(f"  NN Laplacian det (should be ~0): {det_nn:.4e}")

    # 矩阵自由应用
    u = np.sin(np.linspace(0, np.pi, n))
    Au_dd = laplacian_apply(u, bc_type="DD", L=L)
    Au_pp = laplacian_apply(u, bc_type="PP", L=L)
    print(f"  Matrix-free DD apply norm: {np.linalg.norm(Au_dd):.4e}")
    print(f"  Matrix-free PP apply norm: {np.linalg.norm(Au_pp):.4e}")

    # 3D Laplacian
    A3 = laplacian_3d_tensor(4, 4, 4, 0.01, 0.01, 0.01)
    print(f"  3D Laplacian shape: {A3.shape}, sparsity pattern: tridiagonal-block")

    # 稳态组织浓度
    x, C_exact, C_fd = solve_tissue_concentration_profile(
        50, 0.01, D_eff=1e-9, clearance_rate=0.01, influx=1.0
    )
    fd_err = np.max(np.abs(C_exact - C_fd))
    print(f"  Steady-state tissue profile FD max error: {fd_err:.4e}")
    print(f"  Concentration at tissue center: C_exact={C_exact[n//2]:.4f}, C_FD={C_fd[n//2]:.4f}")


def section_5_stochastic():
    """
    模块 5：Feynman-Kac 随机路径积分
    融合种子项目：424_feynman_kac_3d
    """
    print_section("5. 随机微分方程：Feynman-Kac 3D Monte Carlo")

    # 1D 验证
    L, V_const = 1.0, 2.0
    x0 = 0.5
    mc_val = feynman_kac_1d(x0, L, V_const, g_left=0.0, g_right=1.0,
                            h=0.0005, n_trajectories=10000)
    sqrt2V = np.sqrt(2.0 * V_const)
    A = (1.0 - 0.0 * np.cosh(sqrt2V * L)) / np.sinh(sqrt2V * L)
    B = 0.0
    exact = A * np.sinh(sqrt2V * x0) + B * np.cosh(sqrt2V * x0)
    print(f"  1D Feynman-Kac: MC={mc_val:.6f}, Exact={exact:.6f}, RelErr={abs(mc_val-exact)/max(abs(exact),1e-20):.4e}")

    # 3D 药物吸收概率（Feynman-Kac，非零边界条件）
    center = np.array([0.0, 0.0, 0.0])
    axes = np.array([0.05, 0.04, 0.03])
    a_ax, b_ax, c_ax = axes
    prob, se = feynman_kac_3d_monte_carlo(
        0.0, 0.0, 0.0, a_ax, b_ax, c_ax,
        potential=lambda p: 0.5,
        boundary_value=lambda p: np.exp((p[0]/a_ax)**2 + (p[1]/b_ax)**2 + (p[2]/c_ax)**2 - 1.0),
        h=0.001, n_trajectories=1000, max_steps=20000
    )
    print(f"  3D Feynman-Kac estimate: {prob:.6f} ± {se:.6f}")

    # Hitting probability（靶向效率）
    source = np.array([0.02, 0.0, 0.0])
    target = np.array([0.0, 0.0, 0.0])
    hit_prob = organ_hitting_probability(source, target, axes, D_eff=1e-8, n_trajectories=500)
    print(f"  Target hitting probability: {hit_prob:.6f}")


def section_6_ode_solver():
    """
    模块 6：刚性 ODE 系统与 PBPK 瞬态动力学
    融合种子项目：1283_tough_ode
    """
    print_section("6. 刚性 ODE：多 Compartment PBPK 瞬态动力学")

    # 验证 tough_ode（在短区间上使用 RK4 避免刚性累积误差）
    y0 = tough_exact(0.0)
    ts, ys = solve_ode(tough_deriv, (0.0, 0.01), y0, method="rk4", h_init=0.001)
    y_final_exact = tough_exact(ts[-1])
    err = np.linalg.norm(ys[-1] - y_final_exact)
    print(f"  Tough ODE (RK4, t=0.01) final error: {err:.4e}")

    # PBPK 求解（Rosenbrock 方法处理刚性）
    t_span = (0.0, 120.0)  # 2 小时
    ts2, ys2 = solve_pbpk_ode(t_span, method="rosenbrock", h_init=0.1)
    print(f"  PBPK solved in {len(ts2)} steps")

    organ_names = ["Arterial", "Liver", "Kidney", "Muscle", "Adipose", "Tumor", "Venous"]
    print(f"  {'Organ':<12} {'Cmax (mg/L)':<14} {'Tmax (min)':<12} {'C_final (mg/L)':<16}")
    for i, name in enumerate(organ_names):
        cmax = np.max(ys2[:, i])
        tmax = ts2[np.argmax(ys2[:, i])]
        cfinal = ys2[-1, i]
        print(f"  {name:<12} {cmax:<14.4f} {tmax:<12.1f} {cfinal:<16.4f}")

    # 计算 AUC（梯形法则）
    auc = np.trapezoid(ys2, ts2, axis=0)
    print(f"  AUC (mg·min/L): {auc}")


def section_7_geometry():
    """
    模块 7：器官几何与 CVT 采样
    融合种子项目：823_obj_to_tri_surface + 1378_usa_cvt_geo
    """
    print_section("7. 器官几何建模：三角网格 & CVT 采样")

    geoms = build_organ_geometries(n_cvt_points=100)
    print(f"  Organs modeled: {list(geoms.keys())}")
    print(f"  {'Organ':<12} {'Volume (L)':<14} {'Surface (m²)':<14} {'CVT points':<12}")
    for name, g in geoms.items():
        print(f"  {name:<12} {g['volume']:<14.6f} {g['surface_area']:<14.6f} {len(g['cvt_points']):<12}")

    # 器官间距离矩阵
    D = compute_organ_distances(geoms)
    names = list(geoms.keys())
    print(f"  Inter-organ distance matrix (first 3x3):")
    for i in range(min(3, len(names))):
        row = [f"{D[i,j]:.3f}" for j in range(min(3, len(names)))]
        print(f"    {names[i]:<8}: " + "  ".join(row))


def section_8_polynomials():
    """
    模块 8：多项式势能与多靶点目标
    融合种子项目：898_polynomials
    """
    print_section("8. 多项式模型：药物-受体势能 & 多靶点优化")

    # 经典 landscape
    x_rb = np.array([1.0, 1.0, 1.0])
    print(f"  Rosenbrock(1,1,1) = {rosenbrock(x_rb):.6f}")
    print(f"  Himmelblau(3,2) = {himmelblau(np.array([3.0, 2.0])):.6f}")

    # 药物-受体结合势能
    pot = binding_potential_surface(1e-6, 1e-9, 1e5, 1e-3, cooperativity=2)
    print(f"  Binding potential (n=2): {pot:.6e} kcal/mol-scale")

    # 酶动力学（竞争性抑制）
    v = enzyme_kinetics_polynomial_substrate(5.0, 10.0, 2.0, Ki=1.0, I=0.5)
    print(f"  Enzyme velocity (competitive inhibition): {v:.6f} mg/L/min")

    # Sobol G-function（敏感性分析）
    sg = sobol_g_function(np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
                          a=np.array([2.0, 4.0, 6.0, 8.0, 10.0]))
    print(f"  Sobol G-function (sensitivity): {sg:.6f}")


def section_9_interpolation():
    """
    模块 9：插值与浓度-效应关系
    融合种子项目：1213_test_interp_fun
    """
    print_section("9. 插值模型：药时曲线重构 & PD 响应")

    # Runge 函数 Chebyshev 插值测试
    nodes = chebyshev_nodes(12, -1, 1)
    vals = np.array([runge_function(xi) for xi in nodes])
    x_fine = np.linspace(-1, 1, 200)
    y_interp = lagrange_interpolate(nodes, vals, x_fine)
    y_exact = np.array([runge_function(xi) for xi in x_fine])
    max_err = np.max(np.abs(y_interp - y_exact))
    print(f"  Chebyshev Lagrange max error (Runge): {max_err:.4e}")

    # 构建 Hill PD 曲线
    C_nodes, E_nodes = build_pd_curve_from_hill(C50=1.0, Emax=100.0, n_hill=2.0, n_points=50)
    effect_1p5 = pd_effect_interpolate(1.5, C_nodes, E_nodes, method="linear")
    print(f"  PD effect at 1.5×C50: {effect_1p5:.2f}%")

    # 药效学响应
    resp = pharmacodynamic_response(C_plasma=2.0, C50=1.0, Emax=100.0,
                                     n_hill=2.0, baseline=10.0)
    print(f"  Pharmacodynamic response at C=2.0: {resp:.2f}%")

    # 振荡函数插值（快速 PK 变化）
    x_osc = np.linspace(0.01, 0.99, 20)
    y_osc = np.array([oscillatory_function(xi) for xi in x_osc])
    x_test = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    y_interp_lin = piecewise_linear_interpolate(x_osc, y_osc, x_test)
    y_exact_osc = np.array([oscillatory_function(xi) for xi in x_test])
    print(f"  Oscillatory function linear interp error: {np.max(np.abs(y_interp_lin - y_exact_osc)):.4e}")


def section_10_optimization():
    """
    模块 10：动态规划剂量优化
    融合种子项目：624_knapsack_dynamic
    """
    print_section("10. 动态规划：剂量分配 & 给药方案优化")

    # 0/1 背包验证
    w = np.array([2.0, 3.0, 4.0, 5.0])
    v = np.array([3.0, 4.0, 5.0, 6.0])
    max_val, sel = knapsack_01(w, v, 5.0)
    print(f"  Knapsack max value: {max_val:.2f}, selected items: {sel}")

    # 多器官剂量分配
    alloc, eff = optimize_dose_allocation(
        total_dose=500.0,
        organ_volumes=np.array([1.5, 0.3, 30.0, 10.0, 0.5]),
        organ_sensitivities=np.array([0.8, 0.6, 0.3, 0.2, 1.0]),
        organ_toxicities=np.array([0.05, 0.08, 0.02, 0.01, 0.10]),
        max_toxicity=50.0
    )
    print(f"  Optimal dose allocation (mg): {alloc}")
    print(f"  Total efficacy: {eff:.2f}")

    # 给药时间表
    sched, benefit = optimize_dosing_schedule(
        horizon_hours=72.0,
        dose_units=np.array([100.0, 200.0]),
        efficacy_values=np.array([10.0, 18.0]),
        toxicity_values=np.array([2.0, 5.0])
    )
    print(f"  72h dosing schedule (doses given): {sched.sum()} / {len(sched)}")
    print(f"  Net benefit: {benefit:.2f}")


def section_11_utils():
    """
    模块 11：数据校验与数值鲁棒性
    融合种子项目：704_luhn + 116_box_plot
    """
    print_section("11. 数据完整性 & 数值鲁棒性工具")

    # Luhn 校验
    patient_id = "45320194962"
    valid = validate_patient_id(patient_id)
    print(f"  Patient ID '{patient_id}' Luhn valid: {valid}")

    # 浓度分箱
    np.random.seed(42)
    fake_conc = np.random.lognormal(0.5, 0.8, 500)
    counts, edges, centers = concentration_binning(fake_conc, n_bins=8)
    print(f"  Concentration bin counts: {counts}")

    # 统计摘要
    stats = summary_statistics(fake_conc)
    print(f"  Conc stats: mean={stats['mean']:.3f}, median={stats['median']:.3f}, CV={stats['cv']:.3f}")

    # 安全运算
    print(f"  safe_divide(5,0) = {safe_divide(5.0, 0.0, 999.0)}")
    print(f"  softplus(10) = {softplus(10.0):.4f}")
    print(f"  clip_concentration(-1e-20) = {clip_concentration(-1e-20):.2e}")

    # 生理常数
    print(f"  Cardiac output (rest): {PHYSIOLOGICAL_CONSTANTS['CARDIAC_OUTPUT_REST']} L/min")
    print(f"  GFR standard: {PHYSIOLOGICAL_CONSTANTS['GFR_STANDARD']} L/min")
    print(f"  Scaled Q for 50kg patient: {scale_by_body_weight(5.0, 50.0):.3f} L/min")


def section_12_integrated_analysis():
    """
    模块 12：综合不确定性量化分析
    融合多个模块的协同计算
    """
    print_section("12. 综合不确定性量化：全身药物暴露 Monte Carlo")

    n_mc = 200
    # 采样生理参数
    param_samples = sample_physiological_params(
        n_mc,
        np.array([70.0, 5.0, 0.05]),    # BW, CO, fu
        np.array([0.15, 0.10, 0.20])
    )

    auc_liver_samples = []
    auc_tumor_samples = []

    for i in range(n_mc):
        bw, co, fu = param_samples[i]
        # 缩放 PBPK 参数
        # TODO: Hole 3 - 根据采样到的生理参数(bw, co, fu)正确缩放 PBPK 系统参数
        # 提示：需要使用 scale_by_body_weight 对 V 和 Q 进行异速生长缩放
        system = PBPK_ODE_System()
        raise NotImplementedError("Hole 3: Parameter scaling and integration loop not implemented")
        C0 = np.zeros(system.n_comp)
        # TODO: Hole 3 - 实现 Rosenbrock 积分循环以求解 PBPK ODE 系统
        # 需要正确调用 rosenbrock_step，并保证浓度非负，最后收集 ts 和 ys
        ts = []
        ys = []
        ys_arr = np.array(ys)
        ts_arr = np.array(ts)
        auc_liver = np.trapezoid(ys_arr[:, 1], ts_arr)
        auc_tumor = np.trapezoid(ys_arr[:, 5], ts_arr)
        auc_liver_samples.append(auc_liver)
        auc_tumor_samples.append(auc_tumor)

    auc_liver_samples = np.array(auc_liver_samples)
    auc_tumor_samples = np.array(auc_tumor_samples)

    stats_liver = summary_statistics(auc_liver_samples)
    stats_tumor = summary_statistics(auc_tumor_samples)

    print(f"  Monte Carlo samples: {n_mc}")
    print(f"  Liver AUC:   mean={stats_liver['mean']:.2f}, std={stats_liver['std']:.2f}, CV={stats_liver['cv']:.3f}")
    print(f"  Tumor AUC:   mean={stats_tumor['mean']:.2f}, std={stats_tumor['std']:.2f}, CV={stats_tumor['cv']:.3f}")
    print(f"  Tumor/Liver AUC ratio: {safe_divide(stats_tumor['mean'], stats_liver['mean']):.4f}")


def main():
    print("\n" + "#" * 70)
    print("#  PROJECT_126: 全身 PBPK 药物代谢动力学一体化建模系统")
    print("#  领域: 生物医学 — 药物代谢动力学 PBPK 模型")
    print("#  语言: Python  |  级别: 博士级前沿科学计算")
    print("#" * 70)

    start = time.time()

    section_1_special_functions()
    section_2_random_sampling()
    section_3_quadrature()
    section_4_diffusion()
    section_5_stochastic()
    section_6_ode_solver()
    section_7_geometry()
    section_8_polynomials()
    section_9_interpolation()
    section_10_optimization()
    section_11_utils()
    section_12_integrated_analysis()

    elapsed = time.time() - start
    print("\n" + "#" * 70)
    print(f"#  全部计算完成，总耗时: {elapsed:.2f} 秒")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
