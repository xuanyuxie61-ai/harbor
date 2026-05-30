
import numpy as np
import time




from parameters import (
    get_membrane_parameters,
    get_feed_composition,
    compute_permeability,
    compute_sorption_heat,
    get_critical_properties,
    validate_parameters,
    compute_dimensionless_numbers,
)
from utils import (
    van_t_hoff_correction,
    compute_regeneration_calendar,
)
from sparse_matrix import (
    ge_to_ccs,
    circulant_matrix_vector,
    build_periodic_fem_stiffness,
    solve_sparse_system,
)
from polynomial_spectral import (
    monomial_to_legendre_matrix,
    legendre_to_monomial_matrix,
    spectral_interpolate_legendre,
    map_domain_01_to_m1p1,
)
from membrane_geometry import (
    polygon_triangulate,
    generate_hollow_fiber_cross_section,
    integrate_flux_over_triangles,
)
from membrane_fem import (
    build_fem_mesh,
    solve_steady_state_diffusion_reaction,
    solve_transient_diffusion_reaction,
    compute_molar_flux,
    compute_separation_factor,
)
from mass_transfer_ode import (
    reaction_parameters,
    kepler_parameters,
    quasiperiodic_parameters,
    reaction_deriv,
    kepler_like_trajectory_deriv,
    quasiperiodic_forcing_deriv,
    coupled_membrane_reaction_ode,
    runge_function,
    runge_derivative,
    runge_second_derivative,
)
from time_integrator import (
    backward_euler_fixed,
    runge_kutta4,
    adaptive_rk45,
    conserved_quantity_kepler,
    conserved_quantity_quasiperiodic,
    compute_conservation_drift,
)
from nonlinear_solver import (
    broyden_solve,
    solve_permeation_nonlinear,
)
from pore_dynamics import (
    simulate_pore_network_flux,
    simulate_molecular_trajectory,
    knudsen_diffusivity,
    effective_diffusivity_support,
    runge_mesh_adaptation_nodes,
)
from cascade_network import (
    build_cascade_adjacency,
    adjacency_to_google_matrix,
    power_method_rank,
    compute_stage_cuts_from_rank,
    subset_sum_optimal_loading,
    cascade_mass_balance,
    cyclic_regeneration_schedule,
)


def print_banner():
    print("=" * 80)
    print("  博士级合成项目: 膜分离过程多尺度传质模型 (PROJECT_139)")
    print("  科学领域: 化学工程 — 膜分离过程传质模型")
    print("=" * 80)
    print()


def section(title):
    print("-" * 80)
    print(f"  [{title}]")
    print("-" * 80)


def run_parameter_initialization():
    section("1. 参数初始化与验证")
    params = get_membrane_parameters()
    validate_parameters(params)
    feed = get_feed_composition()
    permeability = compute_permeability(params)
    sorption_heat = compute_sorption_heat()
    critical = get_critical_properties()
    print(f"  膜厚度:           {params['membrane_thickness']:.3e} m")
    print(f"  纤维外径:         {params['fiber_outer_radius']:.3e} m")
    print(f"  进料组成 CO2:     {feed['CO2']:.2f}")
    print(f"  CO2 渗透系数:     {permeability['CO2']:.3e} mol·m/(m²·s·Pa)")
    print(f"  CH4 渗透系数:     {permeability['CH4']:.3e} mol·m/(m²·s·Pa)")
    print(f"  临界温度 CO2:     {critical['CO2']['Tc']:.2f} K")
    return params, feed, permeability, sorption_heat


def run_dimensionless_analysis(params):
    section("2. 无量纲数分析")
    dim_co2 = compute_dimensionless_numbers(params, species="CO2")
    dim_ch4 = compute_dimensionless_numbers(params, species="CH4")
    print(f"  CO2 Damkohler 数:   {dim_co2['Damkohler']:.3e}")
    print(f"  CO2 Thiele 模数:    {dim_co2['Thiele_modulus']:.3e}")
    print(f"  CH4 Damkohler 数:   {dim_ch4['Damkohler']:.3e}")
    print(f"  CH4 Thiele 模数:    {dim_ch4['Thiele_modulus']:.3e}")


def run_sparse_matrix_tests(params):
    section("3. 稀疏矩阵与循环矩阵运算")
    n = params["Nx"]
    dx = params["membrane_thickness"] / (n - 1)
    D = params["D_co2"]
    A_periodic = build_periodic_fem_stiffness(n, D, dx)

    ccs = ge_to_ccs(A_periodic)
    print(f"  周期 FEM 刚度矩阵维度: {A_periodic.shape}")
    print(f"  非零元素个数 (NZ):     {ccs.nz_num}")

    first_row = np.zeros(n)
    first_row[0] = 2.0
    first_row[1] = -1.0
    first_row[-1] = -1.0
    x_vec = np.ones(n)
    b_circ = circulant_matrix_vector(n, first_row, x_vec)
    print(f"  循环矩阵-向量乘积范数: {np.linalg.norm(b_circ):.6f}")


def run_spectral_methods(params):
    section("4. 正交多项式谱方法")
    n_poly = 8
    x_nodes = np.linspace(0.0, params["membrane_thickness"], params["Nx"])

    xi = map_domain_01_to_m1p1(x_nodes / params["membrane_thickness"])

    A_mono_to_leg = monomial_to_legendre_matrix(n_poly)
    A_leg_to_mono = legendre_to_monomial_matrix(n_poly)
    print(f"  Legendre 转换矩阵条件数: {np.linalg.cond(A_mono_to_leg):.3e}")

    x_mapped = map_domain_01_to_m1p1(np.linspace(0.0, 1.0, 100))
    coeffs_leg = np.zeros(n_poly + 1)
    coeffs_leg[0] = 1.0
    coeffs_leg[2] = -0.5
    val = spectral_interpolate_legendre(coeffs_leg, x_mapped)
    print(f"  谱插值在 0 处的值:       {val[50]:.6f}")


def run_geometry_triangulation(params):
    section("5. 膜几何剖分与面积通量积分")
    n_vert = 64
    x_poly, y_poly = generate_hollow_fiber_cross_section(
        n_vert, params["fiber_inner_radius"], params["fiber_outer_radius"]
    )
    triangles = polygon_triangulate(n_vert, x_poly, y_poly)

    r = np.sqrt(x_poly ** 2 + y_poly ** 2)
    flux = np.clip((params["fiber_outer_radius"] - r) / params["fiber_outer_radius"], 0.0, 1.0)
    total_flux = integrate_flux_over_triangles(triangles, x_poly, y_poly, flux)
    print(f"  三角形数量:       {len(triangles)}")
    print(f"  总积分通量:       {total_flux:.6e}")


def run_fem_steady_state(params, feed, permeability):
    section("6. 稳态 FEM 扩散-反应求解")
    L = params["membrane_thickness"]
    Nx = params["Nx"]
    c_feed_co2 = feed["CO2"] * params["pressure_feed"] / (params["R_gas"] * params["temperature"])
    c_perm_co2 = feed["CO2"] * params["pressure_permeate"] / (params["R_gas"] * params["temperature"])
    x, c_profile = solve_steady_state_diffusion_reaction(
        L, Nx, params["D_co2"], params["k_reaction"], c_feed_co2, c_perm_co2
    )
    J = compute_molar_flux(x, c_profile, params["D_co2"])
    alpha = compute_separation_factor(
        c_feed_co2, c_perm_co2,
        feed["CH4"] * params["pressure_feed"] / (params["R_gas"] * params["temperature"]),
        feed["CH4"] * params["pressure_permeate"] / (params["R_gas"] * params["temperature"])
    )
    print(f"  稳态膜通量 (CO2):     {np.mean(J):.6e} mol/(m²·s)")
    print(f"  理想分离因子 alpha:   {alpha:.3f}")
    return x, c_profile


def run_fem_transient(params, feed):
    section("7. 瞬态 FEM 扩散-反应求解 (Backward Euler)")
    L = params["membrane_thickness"]
    Nx = params["Nx"]
    Nt = min(params["Nt"], 200)
    c_feed_co2 = feed["CO2"] * params["pressure_feed"] / (params["R_gas"] * params["temperature"])
    c_perm_co2 = feed["CO2"] * params["pressure_permeate"] / (params["R_gas"] * params["temperature"])









    raise NotImplementedError("Hole 3: 请构建质量矩阵并调用修改后的瞬态求解器")
    print(f"  瞬态计算步数:          {Nt}")
    print(f"  最终时刻膜内浓度范围: [{profiles[-1].min():.6e}, {profiles[-1].max():.6e}]")


def run_ode_integrations():
    section("8. 耦合质量传递 ODE 积分")

    rp = reaction_parameters()
    f_react = lambda t, y: reaction_deriv(t, y, rp["k"], rp["K_co2"], rp["K_ch4"], rp["P_total"])
    t_r, y_r = adaptive_rk45(f_react, rp["tspan"], rp["y0"], atol=1e-10, rtol=1e-8)
    print(f"  表面反应 ODE 步数:     {len(t_r)}")
    print(f"  最终 CO2 表面浓度:     {y_r[-1, 0]:.6f}")


    qp = quasiperiodic_parameters()
    f_qp = lambda t, y: quasiperiodic_forcing_deriv(t, y, qp["omega1"])
    t_q, y_q = runge_kutta4(f_qp, qp["tspan"], qp["y0"], 2000)
    drift_qp = compute_conservation_drift(t_q, y_q, lambda y: conserved_quantity_quasiperiodic(y, qp["omega1"]))
    print(f"  准周期 ODE 守恒量漂移: {drift_qp:.6e}")


    t_k, y_k, drift_k = simulate_molecular_trajectory(5000)
    print(f"  类开普勒轨迹能量漂移:  {drift_k:.6e}")


def run_nonlinear_permeation(params, feed, permeability):
    section("9. 非线性渗透率方程求解 (Broyden 方法)")
    p_feed_co2 = feed["CO2"] * params["pressure_feed"]
    p_feed_ch4 = feed["CH4"] * params["pressure_feed"]
    sol, ierr = solve_permeation_nonlinear(
        p_feed_co2, p_feed_ch4,
        permeability["CO2"], permeability["CH4"],
        params["pressure_permeate"], params["membrane_thickness"],
        p_feed_co2 * 0.1, p_feed_ch4 * 0.05,
        T=params["temperature"]
    )
    print(f"  Broyden 收敛标志:      {ierr} (0=成功)")
    print(f"  渗透侧 CO2 分压:       {sol[0]:.6e} Pa")
    print(f"  渗透侧 CH4 分压:       {sol[1]:.6e} Pa")
    print(f"  级段切割率 theta:      {sol[2]:.4f}")


def run_pore_network(params, feed):
    section("10. 孔道网络传质模型")
    c_feed = {
        "CO2": feed["CO2"] * params["pressure_feed"] / (params["R_gas"] * params["temperature"]),
        "CH4": feed["CH4"] * params["pressure_feed"] / (params["R_gas"] * params["temperature"]),
    }
    c_perm = {
        "CO2": feed["CO2"] * params["pressure_permeate"] / (params["R_gas"] * params["temperature"]),
        "CH4": feed["CH4"] * params["pressure_permeate"] / (params["R_gas"] * params["temperature"]),
    }
    J_co2, J_ch4, radii = simulate_pore_network_flux(
        params["pore_count"], params["pore_mean_radius"], params["pore_std_radius"],
        params["module_length"], params["temperature"],
        44.01e-3, 16.04e-3,
        params["porosity"], params["tortuosity"],
        c_feed, c_perm
    )
    print(f"  平均孔径:              {np.mean(radii):.3e} m")
    print(f"  孔道网络 CO2 通量:     {J_co2:.6e} mol/(m²·s)")
    print(f"  孔道网络 CH4 通量:     {J_ch4:.6e} mol/(m²·s)")


def run_cascade_optimization(params, feed):
    section("11. 多级膜联优化与网络流分析")
    stages = params["stages"]
    A = build_cascade_adjacency(stages, recycle_ratio=0.15)
    G = adjacency_to_google_matrix(A, damping=0.15)
    rank = power_method_rank(G, max_iter=200, tol=1e-12)
    cuts = compute_stage_cuts_from_rank(rank, params["stage_cut_nominal"])
    print(f"  PageRank 分布:         {np.array2string(rank, precision=4, suppress_small=True)}")
    print(f"  优化后各级切割率:      {np.array2string(cuts, precision=4, suppress_small=True)}")


    module_capacities = np.array([50, 80, 120, 200, 350, 500], dtype=int)
    target_capacity = 980
    selected = subset_sum_optimal_loading(module_capacities, target_capacity)
    print(f"  目标产能:              {target_capacity}")
    print(f"  子集和最优模块组合:    {[int(x) for x in selected]}")


    perm_flows, ret_flows, perm_comps = cascade_mass_balance(
        feed_flow=1000.0, stage_cuts=cuts, rank_distribution=rank,
        feed_composition=feed
    )
    print(f"  各级渗透流量:          {np.array2string(perm_flows, precision=2, suppress_small=True)}")


    schedule = cyclic_regeneration_schedule(stages, cycle_days=30, start_day=1)
    print(f"  循环再生计划:          Stage {schedule[0]['stage']+1} -> Day {schedule[0]['next_regeneration']}")


def run_runge_mesh_adaptation(params):
    section("12. Runge 函数自适应网格生成")
    x_adapt = runge_mesh_adaptation_nodes(params["Nx"], params["membrane_thickness"])
    f_runge = runge_function(np.linspace(-1.0, 1.0, len(x_adapt)))
    df_runge = runge_derivative(np.linspace(-1.0, 1.0, len(x_adapt)))
    d2f_runge = runge_second_derivative(np.linspace(-1.0, 1.0, len(x_adapt)))
    print(f"  自适应节点数:          {len(x_adapt)}")
    print(f"  节点间距比 (max/min):  {np.max(np.diff(x_adapt)) / max(np.min(np.diff(x_adapt)), 1e-30):.3f}")
    print(f"  Runge 函数最大值:      {np.max(f_runge):.6f}")
    print(f"  Runge 二阶导最大值:    {np.max(np.abs(d2f_runge)):.6f}")


def main():
    print_banner()
    t_start = time.time()


    params, feed, permeability, sorption_heat = run_parameter_initialization()


    run_dimensionless_analysis(params)


    run_sparse_matrix_tests(params)


    run_spectral_methods(params)


    run_geometry_triangulation(params)


    x, c_profile = run_fem_steady_state(params, feed, permeability)


    run_fem_transient(params, feed)


    run_ode_integrations()


    run_nonlinear_permeation(params, feed, permeability)


    run_pore_network(params, feed)


    run_cascade_optimization(params, feed)


    run_runge_mesh_adaptation(params)

    t_elapsed = time.time() - t_start
    print()
    print("=" * 80)
    print(f"  全部计算流程完成，耗时: {t_elapsed:.3f} 秒")
    print("=" * 80)


if __name__ == "__main__":
    main()
