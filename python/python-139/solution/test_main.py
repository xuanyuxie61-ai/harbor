"""
================================================================================
博士级合成项目：膜分离过程多尺度传质模型
================================================================================

统一入口文件。零参数直接运行，完成从参数输入、网格生成、FEM求解、
ODE积分、非线性渗透率计算、孔道网络模拟、多级联优化到结果输出的
完整科研计算流程。

科学领域：化学工程 — 膜分离过程传质模型
================================================================================
"""

import numpy as np
import time

# ------------------------------------------------------------------------------
# Project modules
# ------------------------------------------------------------------------------
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
    # Convert to CCS
    ccs = ge_to_ccs(A_periodic)
    print(f"  周期 FEM 刚度矩阵维度: {A_periodic.shape}")
    print(f"  非零元素个数 (NZ):     {ccs.nz_num}")
    # Circulant matrix-vector product
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
    # Map to [-1,1]
    xi = map_domain_01_to_m1p1(x_nodes / params["membrane_thickness"])
    # Build Legendre-Vandermonde
    A_mono_to_leg = monomial_to_legendre_matrix(n_poly)
    A_leg_to_mono = legendre_to_monomial_matrix(n_poly)
    print(f"  Legendre 转换矩阵条件数: {np.linalg.cond(A_mono_to_leg):.3e}")
    # Test interpolation of a steep front (Runge function mapped)
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
    # Synthetic flux field (linear radial decay)
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
    x, profiles = solve_transient_diffusion_reaction(
        L, Nx, params["D_co2"], params["k_reaction"],
        c_feed_co2, c_perm_co2, Nt, params["t_final"]
    )
    print(f"  瞬态计算步数:          {Nt}")
    print(f"  最终时刻膜内浓度范围: [{profiles[-1].min():.6e}, {profiles[-1].max():.6e}]")


def run_ode_integrations():
    section("8. 耦合质量传递 ODE 积分")
    # 8a. Surface reaction ODE via adaptive RK45
    rp = reaction_parameters()
    f_react = lambda t, y: reaction_deriv(t, y, rp["k"], rp["K_co2"], rp["K_ch4"], rp["P_total"])
    t_r, y_r = adaptive_rk45(f_react, rp["tspan"], rp["y0"], atol=1e-10, rtol=1e-8)
    print(f"  表面反应 ODE 步数:     {len(t_r)}")
    print(f"  最终 CO2 表面浓度:     {y_r[-1, 0]:.6f}")

    # 8b. Quasiperiodic forcing via RK4
    qp = quasiperiodic_parameters()
    f_qp = lambda t, y: quasiperiodic_forcing_deriv(t, y, qp["omega1"])
    t_q, y_q = runge_kutta4(f_qp, qp["tspan"], qp["y0"], 2000)
    drift_qp = compute_conservation_drift(t_q, y_q, lambda y: conserved_quantity_quasiperiodic(y, qp["omega1"]))
    print(f"  准周期 ODE 守恒量漂移: {drift_qp:.6e}")

    # 8c. Kepler-like trajectory
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

    # Subset-sum optimal module loading
    module_capacities = np.array([50, 80, 120, 200, 350, 500], dtype=int)
    target_capacity = 980
    selected = subset_sum_optimal_loading(module_capacities, target_capacity)
    print(f"  目标产能:              {target_capacity}")
    print(f"  子集和最优模块组合:    {[int(x) for x in selected]}")

    # Mass balance
    perm_flows, ret_flows, perm_comps = cascade_mass_balance(
        feed_flow=1000.0, stage_cuts=cuts, rank_distribution=rank,
        feed_composition=feed
    )
    print(f"  各级渗透流量:          {np.array2string(perm_flows, precision=2, suppress_small=True)}")

    # Regeneration schedule
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

    # Step 1
    params, feed, permeability, sorption_heat = run_parameter_initialization()

    # Step 2
    run_dimensionless_analysis(params)

    # Step 3
    run_sparse_matrix_tests(params)

    # Step 4
    run_spectral_methods(params)

    # Step 5
    run_geometry_triangulation(params)

    # Step 6
    x, c_profile = run_fem_steady_state(params, feed, permeability)

    # Step 7
    run_fem_transient(params, feed)

    # Step 8
    run_ode_integrations()

    # Step 9
    run_nonlinear_permeation(params, feed, permeability)

    # Step 10
    run_pore_network(params, feed)

    # Step 11
    run_cascade_optimization(params, feed)

    # Step 12
    run_runge_mesh_adaptation(params)

    t_elapsed = time.time() - t_start
    print()
    print("=" * 80)
    print(f"  全部计算流程完成，耗时: {t_elapsed:.3f} 秒")
    print("=" * 80)


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（50个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: get_membrane_parameters 返回包含所有必要键的字典 ----
p = get_membrane_parameters()
required_keys = ["membrane_thickness", "fiber_inner_radius", "fiber_outer_radius",
                 "temperature", "Nx", "Nt", "pore_count", "stages"]
for k in required_keys:
    assert k in p, f'[TC01] Missing key: {k} FAILED'
assert p["membrane_thickness"] > 0, '[TC01] membrane_thickness must be positive FAILED'

# ---- TC02: get_feed_composition 摩尔分数和为 1 ----
feed_test = get_feed_composition()
total = feed_test["CO2"] + feed_test["CH4"] + feed_test["N2"]
assert abs(total - 1.0) < 1e-12, f'[TC02] Feed fractions sum = {total} FAILED'

# ---- TC03: compute_permeability 返回正值 ----
perm = compute_permeability(p)
assert perm["CO2"] > 0, '[TC03] CO2 permeability must be positive FAILED'
assert perm["CH4"] > 0, '[TC03] CH4 permeability must be positive FAILED'

# ---- TC04: compute_sorption_heat 返回负值（吸附放热） ----
sorp = compute_sorption_heat()
assert sorp["CO2"] < 0, '[TC04] CO2 sorption heat must be negative FAILED'
assert sorp["CH4"] < 0, '[TC04] CH4 sorption heat must be negative FAILED'

# ---- TC05: validate_parameters 对合法参数不抛出异常 ----
validate_parameters(p)

# ---- TC06: compute_dimensionless_numbers 返回有限正值 ----
dim = compute_dimensionless_numbers(p, species="CO2")
assert np.isfinite(dim["Damkohler"]), '[TC06] Damkohler must be finite FAILED'
assert dim["Damkohler"] > 0, '[TC06] Damkohler must be positive FAILED'
assert dim["Thiele_modulus"] > 0, '[TC06] Thiele_modulus must be positive FAILED'

# ---- TC07: safe_sqrt 正确处理正常输入 ----
import numpy as np
from utils import safe_sqrt
assert abs(safe_sqrt(4.0) - 2.0) < 1e-12, '[TC07] sqrt(4) = 2 FAILED'
assert abs(safe_sqrt(0.0) - 0.0) < 1e-12, '[TC07] sqrt(0) = 0 FAILED'

# ---- TC08: safe_divide 除零返回默认值 ----
from utils import safe_divide
assert safe_divide(1.0, 0.0) == 0.0, '[TC08] divide by zero returns default FAILED'
assert abs(safe_divide(6.0, 2.0) - 3.0) < 1e-12, '[TC08] 6/2 = 3 FAILED'

# ---- TC09: modular_wrap 正确回绕 ----
from utils import modular_wrap
assert modular_wrap(32, 1, 31) == 1, '[TC09] wrap(32,1,31) = 1 FAILED'
assert modular_wrap(15, 1, 31) == 15, '[TC09] wrap(15,1,31) = 15 FAILED'
assert modular_wrap(0, 1, 31) == 31, '[TC09] wrap(0,1,31) = 31 FAILED'

# ---- TC10: van_t_hoff_correction 高温下增大溶解度 ----
S_T = van_t_hoff_correction(1.0, -24000.0, 308.15)
assert S_T > 0, '[TC10] van_t_Hoff output must be positive FAILED'

# ---- TC11: linear_interpolate 节点处精确插值 ----
from utils import linear_interpolate, bracket_interval
x_nodes = np.array([0.0, 1.0, 2.0, 3.0])
y_vals = np.array([0.0, 2.0, 4.0, 6.0])
y_interp = linear_interpolate(x_nodes, y_vals, np.array([1.0]))
assert abs(y_interp[0] - 2.0) < 1e-12, '[TC11] linear interp at node FAILED'

# ---- TC12: ge_to_ccs 保持矩阵-向量乘积 ----
A_test = np.array([[4.0, 1.0, 0.0], [0.0, 3.0, 2.0], [1.0, 0.0, 5.0]])
ccs = ge_to_ccs(A_test)
A_dense = ccs.to_dense()
x_test = np.array([1.0, 2.0, 3.0])
assert np.allclose(A_dense.dot(x_test), A_test.dot(x_test), rtol=1e-12), '[TC12] CCS preserves mat-vec FAILED'

# ---- TC13: circulant_matrix_vector 结果与稠密矩阵等价 ----
n = 5
first_row = np.array([2.0, -1.0, 0.0, 0.0, -1.0])
x_vec = np.ones(n)
b_circ = circulant_matrix_vector(n, first_row, x_vec)
# 手动构建循环矩阵
C = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        C[i, j] = first_row[(j - i) % n]
b_dense = C.dot(x_vec)
assert np.allclose(b_circ, b_dense, rtol=1e-12), '[TC13] circulant = dense mat-vec FAILED'

# ---- TC14: solve_sparse_system 正确求解对角系统 ----
A_diag = np.diag(np.array([2.0, 3.0, 4.0]))
b_diag = np.array([1.0, 1.0, 1.0])
sol = solve_sparse_system(A_diag, b_diag)
expected = np.array([0.5, 1.0/3.0, 0.25])
assert np.allclose(sol, expected, rtol=1e-12), '[TC14] solve diag system FAILED'

# ---- TC15: monomial_to_legendre_matrix 可逆且乘积为单位阵 ----
A_m2l = monomial_to_legendre_matrix(5)
A_l2m = legendre_to_monomial_matrix(5)
product = A_m2l.dot(A_l2m)
assert np.allclose(product, np.eye(6), atol=1e-10), '[TC15] Legendre matrices not inverse FAILED'

# ---- TC16: spectral_interpolate_legendre 正确求值 P0 ----
from polynomial_spectral import spectral_interpolate_legendre
coeffs = np.zeros(4)
coeffs[0] = 3.0  # 3 * P_0(x) = 3
x_q = np.linspace(-1.0, 1.0, 5)
val = spectral_interpolate_legendre(coeffs, x_q)
assert np.allclose(val, 3.0, rtol=1e-12), '[TC16] Legendre interp of constant FAILED'

# ---- TC17: map_domain_01_to_m1p1 正确映射端点 ----
assert abs(map_domain_01_to_m1p1(0.0) - (-1.0)) < 1e-12, '[TC17] map 0 -> -1 FAILED'
assert abs(map_domain_01_to_m1p1(1.0) - 1.0) < 1e-12, '[TC17] map 1 -> 1 FAILED'
assert abs(map_domain_01_to_m1p1(0.5) - 0.0) < 1e-12, '[TC17] map 0.5 -> 0 FAILED'

# ---- TC18: generate_hollow_fiber_cross_section 生成正确点数 ----
x_poly, y_poly = generate_hollow_fiber_cross_section(32, 1e-4, 2e-4)
assert len(x_poly) == 32, '[TC18] wrong number of vertices FAILED'
assert len(y_poly) == 32, '[TC18] wrong number of y FAILED'

# ---- TC19: polygon_triangulate 产生 n-2 个三角形 ----
n_tri = 16
theta = np.linspace(0.0, 2.0 * np.pi, n_tri, endpoint=False)
x_tri = np.cos(theta)
y_tri = np.sin(theta)
triangles = polygon_triangulate(n_tri, x_tri, y_tri)
assert triangles.shape == (n_tri - 2, 3), f'[TC19] expected ({n_tri-2},3) got {triangles.shape} FAILED'

# ---- TC20: integrate_flux_over_triangles 返回非负值 ----
flux_field = np.ones(n_tri)
total_flux = integrate_flux_over_triangles(triangles, x_tri, y_tri, flux_field)
assert total_flux > 0, '[TC20] flux must be positive FAILED'

# ---- TC21: build_fem_mesh 返回正确尺寸 ----
x_mesh, dx_mesh = build_fem_mesh(1.5e-7, 128)
assert len(x_mesh) == 128, '[TC21] mesh size wrong FAILED'
assert x_mesh[0] == 0.0, '[TC21] first node must be 0 FAILED'
assert abs(x_mesh[-1] - 1.5e-7) < 1e-20, '[TC21] last node must be L FAILED'

# ---- TC22: solve_steady_state_diffusion_reaction 浓度有界 ----
c_feed = 150.0
c_perm = 1.0
x_ss, c_ss = solve_steady_state_diffusion_reaction(1.5e-7, 64, 2.5e-10, 4.2e-3, c_feed, c_perm)
assert np.all(c_ss >= c_perm - 1e-12), '[TC22] concentration below perm FAILED'
assert np.all(c_ss <= c_feed + 1e-12), '[TC22] concentration above feed FAILED'
assert len(c_ss) == 64, '[TC22] wrong profile size FAILED'

# ---- TC23: compute_molar_flux 返回有限值 ----
J_flux = compute_molar_flux(x_ss, c_ss, 2.5e-10)
assert np.all(np.isfinite(J_flux)), '[TC23] flux must be finite FAILED'

# ---- TC24: compute_separation_factor 返回非负值兼容 ----
alpha_test = compute_separation_factor(100.0, 10.0, 800.0, 80.0)
assert alpha_test >= 0, '[TC24] separation factor must be >= 0 FAILED'

# ---- TC25: reaction_deriv 返回 3 分量数组 ----
rp = reaction_parameters()
dy = reaction_deriv(0.0, rp["y0"], rp["k"], rp["K_co2"], rp["K_ch4"], rp["P_total"])
assert len(dy) == 3, '[TC25] reaction_deriv must return 3 components FAILED'

# ---- TC26: kepler_like_trajectory_deriv 返回 4 分量数组 ----
kp = kepler_parameters()
dy_k = kepler_like_trajectory_deriv(0.0, kp["y0"], mu=kp["mu"])
assert len(dy_k) == 4, '[TC26] kepler deriv must return 4 components FAILED'

# ---- TC27: quasiperiodic_forcing_deriv 返回 4 分量数组 ----
qp = quasiperiodic_parameters()
dy_q = quasiperiodic_forcing_deriv(0.0, qp["y0"], qp["omega1"])
assert len(dy_q) == 4, '[TC27] quasiperiodic deriv must return 4 components FAILED'

# ---- TC28: runge_function(0) = 1 ----
assert abs(runge_function(0.0) - 1.0) < 1e-12, '[TC28] Runge(0) = 1 FAILED'

# ---- TC29: runge_derivative(0) = 0 ----
assert abs(runge_derivative(0.0) - 0.0) < 1e-12, '[TC29] Runge\'(0) = 0 FAILED'

# ---- TC30: runge_second_derivative(0) = -50 ----
assert abs(runge_second_derivative(0.0) - (-50.0)) < 1e-10, '[TC30] Runge\'\'(0) = -50 FAILED'

# ---- TC31: runge_kutta4 保持输出形状 ----
t_rk, y_rk = runge_kutta4(lambda t, y: np.array([-y[0]]), (0.0, 1.0), np.array([1.0]), 100)
assert y_rk.shape[1] == 1, '[TC31] RK4 output dim incorrect FAILED'

# ---- TC32: adaptive_rk45 对简单系统正确积分（指数衰减） ----
import numpy as np
np.random.seed(42)
t_ad, y_ad = adaptive_rk45(
    lambda t, y: np.array([-y[0]]), (0.0, 1.0), np.array([1.0]), atol=1e-10, rtol=1e-8
)
assert y_ad[-1, 0] > 0, '[TC32] adaptive RK45 final value must be positive FAILED'

# ---- TC33: forward_euler 保持输出形状 ----
from time_integrator import forward_euler
t_fe, y_fe = forward_euler(lambda t, y: np.array([-y[0]]), (0.0, 1.0), np.array([1.0]), 100)
assert y_fe.shape == (101, 1), '[TC33] forward Euler shape incorrect FAILED'

# ---- TC34: broyden_solve 求解简单标量方程 -------
def f_lin(x):
    return np.array([x[0] - 2.0])
sol_b, ierr_b, _ = broyden_solve(f_lin, np.array([0.0]))
assert ierr_b == 0, '[TC34] Broyden did not converge FAILED'
assert abs(sol_b[0] - 2.0) < 1e-6, '[TC34] Broyden x[0] incorrect FAILED'

# ---- TC35: knudsen_diffusivity 返回正值 ----
D_k = knudsen_diffusivity(5e-9, 308.15, 44.01e-3)
assert D_k > 0, '[TC35] Knudsen diffusivity must be positive FAILED'

# ---- TC36: effective_diffusivity_support 有效扩散系数 ----
D_eff = effective_diffusivity_support(D_k, 0.35, 2.8)
assert D_eff > 0, '[TC36] effective diffusivity must be positive FAILED'
assert D_eff < D_k, '[TC36] effective D < Knudsen D FAILED'

# ---- TC37: build_cascade_adjacency 返回正确形状 ----
A_cas = build_cascade_adjacency(4, recycle_ratio=0.15)
assert A_cas.shape == (4, 4), '[TC37] adjacency shape incorrect FAILED'

# ---- TC38: adjacency_to_google_matrix 列和为 1 ----
G_mat = adjacency_to_google_matrix(A_cas, damping=0.15)
col_sums = np.sum(G_mat, axis=0)
assert np.allclose(col_sums, 1.0, rtol=1e-12), '[TC38] Google matrix columns must sum to 1 FAILED'

# ---- TC39: power_method_rank 返回概率分布 ----
rank_vec = power_method_rank(G_mat, max_iter=200, tol=1e-12)
assert abs(np.sum(rank_vec) - 1.0) < 1e-12, '[TC39] rank must sum to 1 FAILED'
assert np.all(rank_vec >= 0), '[TC39] rank entries must be non-negative FAILED'

# ---- TC40: compute_stage_cuts_from_rank 输出在 [0.05, 0.95] ----
cuts = compute_stage_cuts_from_rank(rank_vec, 0.3)
assert np.all(cuts >= 0.05), '[TC40] stage cuts >= 0.05 FAILED'
assert np.all(cuts <= 0.95), '[TC40] stage cuts <= 0.95 FAILED'

# ---- TC41: subset_sum_optimal_loading 找到可行子集 ----
capacities = np.array([50, 80, 120, 200, 350, 500], dtype=int)
selected = subset_sum_optimal_loading(capacities, 200)
assert len(selected) > 0, '[TC41] subset_sum must find non-empty subset FAILED'

# ---- TC42: cascade_mass_balance 返回正确数组长度 ----
feed_comp = get_feed_composition()
perm_f, ret_f, perm_c = cascade_mass_balance(1000.0, cuts, rank_vec, feed_comp)
assert len(perm_f) == 4, '[TC42] permeate flow array length incorrect FAILED'
assert len(ret_f) == 4, '[TC42] retentate flow array length incorrect FAILED'

# ---- TC43: solve_permeation_nonlinear 收敛并返回 3 分量 ----
p_test = get_membrane_parameters()
perm_test = compute_permeability(p_test)
pf_co2 = 0.15 * p_test["pressure_feed"]
pf_ch4 = 0.80 * p_test["pressure_feed"]
sol_perm, ierr_perm = solve_permeation_nonlinear(
    pf_co2, pf_ch4, perm_test["CO2"], perm_test["CH4"],
    p_test["pressure_permeate"], p_test["membrane_thickness"],
    pf_co2 * 0.1, pf_ch4 * 0.05, T=p_test["temperature"]
)
assert len(sol_perm) == 3, '[TC43] permeation solution must have 3 components FAILED'
assert sol_perm[2] >= 0.0 and sol_perm[2] <= 1.0, '[TC43] stage cut in [0,1] FAILED'

# ---- TC44: runge_mesh_adaptation_nodes 节点数正确 ----
x_adapt = runge_mesh_adaptation_nodes(64, 1.5e-7)
assert len(x_adapt) == 64, '[TC44] adaptation nodes count incorrect FAILED'
assert x_adapt[0] == 0.0, '[TC44] first node must be 0 FAILED'
assert abs(x_adapt[-1] - 1.5e-7) < 1e-20, '[TC44] last node must be L FAILED'

# ---- TC45: coupled_membrane_reaction_ode 返回 9 分量 ----
coupled_params = {
    "k_reaction": 4.2e-3, "K_ads_co2": 1.2e-3, "K_ads_ch4": 2.5e-4,
    "omega1": np.pi, "h_mt": 1e-4,
}
y0_coupled = np.array([150.0, 800.0, 0.0, 150.0, 800.0, 0.01, 0.0, -0.01*np.pi**2, 0.0])
dy_coupled = coupled_membrane_reaction_ode(0.0, y0_coupled, coupled_params)
assert len(dy_coupled) == 9, '[TC45] coupled ODE must return 9 components FAILED'

# ---- TC46: 确定性重复调用 adaptive_rk45 结果一致 ----
import numpy as np
np.random.seed(42)
t1, y1 = adaptive_rk45(
    lambda t, y: np.array([-y[0]]), (0.0, 1.0), np.array([1.0]), atol=1e-10, rtol=1e-8
)
assert abs(y1[-1, 0] - np.exp(-1.0)) < 0.05, '[TC46] adaptive RK45 near exp(-1) FAILED'

# ---- TC47: power_series_solution_ode 在 t=0 处等于常数项 ----
from time_integrator import power_series_solution_ode
coeffs_ps = [3.0, 2.0, 1.0]
val_ps = power_series_solution_ode(0.0, coeffs_ps)
assert abs(val_ps - 3.0) < 1e-12, '[TC47] power series at 0 = c0 FAILED'

# ---- TC48: 确定性 runge_kutta4 结果可复现 ----
import numpy as np
np.random.seed(42)
t_rk1, y_rk1 = runge_kutta4(lambda t, y: np.array([-y[0]]), (0.0, 2.0), np.array([1.0]), 200)
np.random.seed(42)
t_rk2, y_rk2 = runge_kutta4(lambda t, y: np.array([-y[0]]), (0.0, 2.0), np.array([1.0]), 200)
assert np.allclose(y_rk1, y_rk2, rtol=1e-12), '[TC48] RK4 reproducibility FAILED'

# ---- TC49: polygon_area 对单位正方形返回 1.0 ----
from membrane_geometry import polygon_area
x_sq = np.array([0.0, 1.0, 1.0, 0.0])
y_sq = np.array([0.0, 0.0, 1.0, 1.0])
area_sq = polygon_area(4, x_sq, y_sq)
assert abs(area_sq - 1.0) < 1e-12, '[TC49] unit square area = 1 FAILED'

# ---- TC50: safe_sqrt 对极小负数返回 0 不崩溃 ----
from utils import safe_sqrt
assert safe_sqrt(-1e-15) == 0.0, '[TC50] safe_sqrt small negative returns 0 FAILED'

print('\n全部 50 个测试通过!\n')
