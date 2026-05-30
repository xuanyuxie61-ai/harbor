
import numpy as np
import time




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
    print_section("1. 特殊函数：Carlson 椭圆积分 / Jacobi 函数 / 超几何函数")


    rf_val = carlson_rf(1.0, 2.0, 0.0)
    print(f"  Carlson RF(1,2,0) = {rf_val:.10f}")


    sn, cn, dn = jacobi_sncndn(0.5, 0.5)
    verify = sn ** 2 + cn ** 2
    print(f"  Jacobi sn(0.5|0.5)={sn:.10f}, cn={cn:.10f}, dn={dn:.10f}")
    print(f"  Identity check sn^2+cn^2 = {verify:.10f}")


    agm_val, iters = gauss_agm(1.0, np.sqrt(2.0))
    print(f"  AGM(1,√2) = {agm_val:.10f}  (iterations={iters})")


    hf = hyper_2f1(1.0, 2.0, 3.0, 0.5)
    print(f"  2F1(1,2;3;0.5) = {hf:.10f}")


    K_a = 1e5
    C_p = 1e-6
    fu = drug_protein_binding_fraction(K_a, C_p, n_sites=1)
    print(f"  Free fraction (K_a={K_a:.0e}, C_p={C_p:.0e}): fu = {fu:.6f}")


    D_parallel = 1e-9
    D_perpendicular = 1e-10
    theta = np.pi / 4.0
    D_eff = effective_diffusion_coefficient(D_parallel, D_perpendicular, theta)
    print(f"  Effective diffusion (anisotropic, θ=π/4): D_eff = {D_eff:.3e} m^2/s")


def section_2_random_sampling():
    print_section("2. 随机采样：均匀/正态顺序统计量 & 药物到达时间")

    n = 20
    u1 = r8vec_uniform_01_sorted_exponential(n)
    u2 = r8vec_uniform_01_sorted_product(n)
    print(f"  Uniform sorted (exp) first 5: {u1[:5]}")
    print(f"  Uniform sorted (prod) first 5: {u2[:5]}")

    z = r8vec_normal_01_sorted(n, method="exponential")
    print(f"  Normal sorted first 5: {z[:5]}")


    arrival = sample_drug_arrival_times(200, mean_interval=10.0, cv=0.3)
    print(f"  Drug arrival times (first 5): {arrival[:5]}")
    print(f"  Total absorption time: {arrival[-1]:.1f} s")


    params = sample_physiological_params(
        100,
        np.array([70.0, 1.2, 0.05]),
        np.array([0.15, 0.20, 0.30])
    )
    stats = summary_statistics(params[:, 0])
    print(f"  Physiological param sampling: mean BW={stats['mean']:.2f} kg, cv={stats['cv']:.3f}")


def section_3_quadrature():
    print_section("3. 高维求积：Smolyak 稀疏网格 CC & Felippa 乘积规则")


    w = cc_weights(3)
    print(f"  CC level 3 sum(weights) = {w.sum():.10f} (target=2.0)")



    result = sparse_grid_integrate(lambda p: p[0]**2 * p[1]**2, 2, 3)
    print(f"  Sparse grid ∫ x^2 y^2 dA = {result:.10f} (exact = {4.0/9.0:.10f})")


    result3 = sparse_grid_integrate(lambda p: 1.0, 3, 2)
    print(f"  Sparse grid ∫ 1 dV over [-1,1]^3 = {result3:.6f} (target=8.0)")


    xn, yn, wn = square_felippa_rule(0.0, 1.0, 0.0, 1.0, 5)
    integral = np.sum(wn)
    print(f"  Felippa 5-pt rule over unit square = {integral:.10f} (target=1.0)")


    def conc_example(x, y):
        return 100.0 * np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.1)
    organ_integral = integrate_organ_slice(conc_example, (0.0, 1.0), (0.0, 1.0), order=5)
    print(f"  Organ slice concentration integral = {organ_integral:.4f} (mg·cm^2/L)")


def section_4_diffusion():
    print_section("4. 组织扩散：有限差分 Laplacian & 稳态浓度分布")

    n = 30
    L = 0.01
    A_dd = laplacian_1d_dd(n, L)
    lam = laplacian_dd_eigenvalues(n, L)
    V = laplacian_dd_eigenvectors(n)

    err = np.linalg.norm(A_dd @ V - V @ np.diag(lam))
    print(f"  DD Laplacian eigen-decomposition error: {err:.4e}")


    A_nn = laplacian_1d_nn(n, L)
    det_nn = np.linalg.det(A_nn)
    print(f"  NN Laplacian det (should be ~0): {det_nn:.4e}")


    u = np.sin(np.linspace(0, np.pi, n))
    Au_dd = laplacian_apply(u, bc_type="DD", L=L)
    Au_pp = laplacian_apply(u, bc_type="PP", L=L)
    print(f"  Matrix-free DD apply norm: {np.linalg.norm(Au_dd):.4e}")
    print(f"  Matrix-free PP apply norm: {np.linalg.norm(Au_pp):.4e}")


    A3 = laplacian_3d_tensor(4, 4, 4, 0.01, 0.01, 0.01)
    print(f"  3D Laplacian shape: {A3.shape}, sparsity pattern: tridiagonal-block")


    x, C_exact, C_fd = solve_tissue_concentration_profile(
        50, 0.01, D_eff=1e-9, clearance_rate=0.01, influx=1.0
    )
    fd_err = np.max(np.abs(C_exact - C_fd))
    print(f"  Steady-state tissue profile FD max error: {fd_err:.4e}")
    print(f"  Concentration at tissue center: C_exact={C_exact[n//2]:.4f}, C_FD={C_fd[n//2]:.4f}")


def section_5_stochastic():
    print_section("5. 随机微分方程：Feynman-Kac 3D Monte Carlo")


    L, V_const = 1.0, 2.0
    x0 = 0.5
    mc_val = feynman_kac_1d(x0, L, V_const, g_left=0.0, g_right=1.0,
                            h=0.0005, n_trajectories=10000)
    sqrt2V = np.sqrt(2.0 * V_const)
    A = (1.0 - 0.0 * np.cosh(sqrt2V * L)) / np.sinh(sqrt2V * L)
    B = 0.0
    exact = A * np.sinh(sqrt2V * x0) + B * np.cosh(sqrt2V * x0)
    print(f"  1D Feynman-Kac: MC={mc_val:.6f}, Exact={exact:.6f}, RelErr={abs(mc_val-exact)/max(abs(exact),1e-20):.4e}")


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


    source = np.array([0.02, 0.0, 0.0])
    target = np.array([0.0, 0.0, 0.0])
    hit_prob = organ_hitting_probability(source, target, axes, D_eff=1e-8, n_trajectories=500)
    print(f"  Target hitting probability: {hit_prob:.6f}")


def section_6_ode_solver():
    print_section("6. 刚性 ODE：多 Compartment PBPK 瞬态动力学")


    y0 = tough_exact(0.0)
    ts, ys = solve_ode(tough_deriv, (0.0, 0.01), y0, method="rk4", h_init=0.001)
    y_final_exact = tough_exact(ts[-1])
    err = np.linalg.norm(ys[-1] - y_final_exact)
    print(f"  Tough ODE (RK4, t=0.01) final error: {err:.4e}")


    t_span = (0.0, 120.0)
    ts2, ys2 = solve_pbpk_ode(t_span, method="rosenbrock", h_init=0.1)
    print(f"  PBPK solved in {len(ts2)} steps")

    organ_names = ["Arterial", "Liver", "Kidney", "Muscle", "Adipose", "Tumor", "Venous"]
    print(f"  {'Organ':<12} {'Cmax (mg/L)':<14} {'Tmax (min)':<12} {'C_final (mg/L)':<16}")
    for i, name in enumerate(organ_names):
        cmax = np.max(ys2[:, i])
        tmax = ts2[np.argmax(ys2[:, i])]
        cfinal = ys2[-1, i]
        print(f"  {name:<12} {cmax:<14.4f} {tmax:<12.1f} {cfinal:<16.4f}")


    auc = np.trapezoid(ys2, ts2, axis=0)
    print(f"  AUC (mg·min/L): {auc}")


def section_7_geometry():
    print_section("7. 器官几何建模：三角网格 & CVT 采样")

    geoms = build_organ_geometries(n_cvt_points=100)
    print(f"  Organs modeled: {list(geoms.keys())}")
    print(f"  {'Organ':<12} {'Volume (L)':<14} {'Surface (m²)':<14} {'CVT points':<12}")
    for name, g in geoms.items():
        print(f"  {name:<12} {g['volume']:<14.6f} {g['surface_area']:<14.6f} {len(g['cvt_points']):<12}")


    D = compute_organ_distances(geoms)
    names = list(geoms.keys())
    print(f"  Inter-organ distance matrix (first 3x3):")
    for i in range(min(3, len(names))):
        row = [f"{D[i,j]:.3f}" for j in range(min(3, len(names)))]
        print(f"    {names[i]:<8}: " + "  ".join(row))


def section_8_polynomials():
    print_section("8. 多项式模型：药物-受体势能 & 多靶点优化")


    x_rb = np.array([1.0, 1.0, 1.0])
    print(f"  Rosenbrock(1,1,1) = {rosenbrock(x_rb):.6f}")
    print(f"  Himmelblau(3,2) = {himmelblau(np.array([3.0, 2.0])):.6f}")


    pot = binding_potential_surface(1e-6, 1e-9, 1e5, 1e-3, cooperativity=2)
    print(f"  Binding potential (n=2): {pot:.6e} kcal/mol-scale")


    v = enzyme_kinetics_polynomial_substrate(5.0, 10.0, 2.0, Ki=1.0, I=0.5)
    print(f"  Enzyme velocity (competitive inhibition): {v:.6f} mg/L/min")


    sg = sobol_g_function(np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
                          a=np.array([2.0, 4.0, 6.0, 8.0, 10.0]))
    print(f"  Sobol G-function (sensitivity): {sg:.6f}")


def section_9_interpolation():
    print_section("9. 插值模型：药时曲线重构 & PD 响应")


    nodes = chebyshev_nodes(12, -1, 1)
    vals = np.array([runge_function(xi) for xi in nodes])
    x_fine = np.linspace(-1, 1, 200)
    y_interp = lagrange_interpolate(nodes, vals, x_fine)
    y_exact = np.array([runge_function(xi) for xi in x_fine])
    max_err = np.max(np.abs(y_interp - y_exact))
    print(f"  Chebyshev Lagrange max error (Runge): {max_err:.4e}")


    C_nodes, E_nodes = build_pd_curve_from_hill(C50=1.0, Emax=100.0, n_hill=2.0, n_points=50)
    effect_1p5 = pd_effect_interpolate(1.5, C_nodes, E_nodes, method="linear")
    print(f"  PD effect at 1.5×C50: {effect_1p5:.2f}%")


    resp = pharmacodynamic_response(C_plasma=2.0, C50=1.0, Emax=100.0,
                                     n_hill=2.0, baseline=10.0)
    print(f"  Pharmacodynamic response at C=2.0: {resp:.2f}%")


    x_osc = np.linspace(0.01, 0.99, 20)
    y_osc = np.array([oscillatory_function(xi) for xi in x_osc])
    x_test = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    y_interp_lin = piecewise_linear_interpolate(x_osc, y_osc, x_test)
    y_exact_osc = np.array([oscillatory_function(xi) for xi in x_test])
    print(f"  Oscillatory function linear interp error: {np.max(np.abs(y_interp_lin - y_exact_osc)):.4e}")


def section_10_optimization():
    print_section("10. 动态规划：剂量分配 & 给药方案优化")


    w = np.array([2.0, 3.0, 4.0, 5.0])
    v = np.array([3.0, 4.0, 5.0, 6.0])
    max_val, sel = knapsack_01(w, v, 5.0)
    print(f"  Knapsack max value: {max_val:.2f}, selected items: {sel}")


    alloc, eff = optimize_dose_allocation(
        total_dose=500.0,
        organ_volumes=np.array([1.5, 0.3, 30.0, 10.0, 0.5]),
        organ_sensitivities=np.array([0.8, 0.6, 0.3, 0.2, 1.0]),
        organ_toxicities=np.array([0.05, 0.08, 0.02, 0.01, 0.10]),
        max_toxicity=50.0
    )
    print(f"  Optimal dose allocation (mg): {alloc}")
    print(f"  Total efficacy: {eff:.2f}")


    sched, benefit = optimize_dosing_schedule(
        horizon_hours=72.0,
        dose_units=np.array([100.0, 200.0]),
        efficacy_values=np.array([10.0, 18.0]),
        toxicity_values=np.array([2.0, 5.0])
    )
    print(f"  72h dosing schedule (doses given): {sched.sum()} / {len(sched)}")
    print(f"  Net benefit: {benefit:.2f}")


def section_11_utils():
    print_section("11. 数据完整性 & 数值鲁棒性工具")


    patient_id = "45320194962"
    valid = validate_patient_id(patient_id)
    print(f"  Patient ID '{patient_id}' Luhn valid: {valid}")


    np.random.seed(42)
    fake_conc = np.random.lognormal(0.5, 0.8, 500)
    counts, edges, centers = concentration_binning(fake_conc, n_bins=8)
    print(f"  Concentration bin counts: {counts}")


    stats = summary_statistics(fake_conc)
    print(f"  Conc stats: mean={stats['mean']:.3f}, median={stats['median']:.3f}, CV={stats['cv']:.3f}")


    print(f"  safe_divide(5,0) = {safe_divide(5.0, 0.0, 999.0)}")
    print(f"  softplus(10) = {softplus(10.0):.4f}")
    print(f"  clip_concentration(-1e-20) = {clip_concentration(-1e-20):.2e}")


    print(f"  Cardiac output (rest): {PHYSIOLOGICAL_CONSTANTS['CARDIAC_OUTPUT_REST']} L/min")
    print(f"  GFR standard: {PHYSIOLOGICAL_CONSTANTS['GFR_STANDARD']} L/min")
    print(f"  Scaled Q for 50kg patient: {scale_by_body_weight(5.0, 50.0):.3f} L/min")


def section_12_integrated_analysis():
    print_section("12. 综合不确定性量化：全身药物暴露 Monte Carlo")

    n_mc = 200

    param_samples = sample_physiological_params(
        n_mc,
        np.array([70.0, 5.0, 0.05]),
        np.array([0.15, 0.10, 0.20])
    )

    auc_liver_samples = []
    auc_tumor_samples = []

    for i in range(n_mc):
        bw, co, fu = param_samples[i]



        system = PBPK_ODE_System()
        raise NotImplementedError("Hole 3: Parameter scaling and integration loop not implemented")
        C0 = np.zeros(system.n_comp)


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
