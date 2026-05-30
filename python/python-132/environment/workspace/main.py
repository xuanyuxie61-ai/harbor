
import numpy as np
import time


from utils import thermo_factor_check, relative_change
from vle_thermodynamics import (
    jacobi_polynomial, laguerre_compute, laguerre_quadrature_integrate,
    antoine_vapor_pressure, wilson_activity_coefficient,
    vle_flash_calculation, vle_relative_volatility,
    activity_coefficient_spectral_expansion
)
from property_interpolation import (
    shepard_interp_1d, quad_trapezoid,
    interpolate_vle_data, integrate_mass_transfer_flux
)
from tray_geometry_mesh import (
    drectangle, reference_to_physical_q4, generate_tray_mesh,
    compute_local_efficiency_on_mesh, mesh_average_efficiency
)
from pressure_wave_dynamics import (
    fd1d_wave_solve, pressure_wave_in_column, pressure_stability_index
)
from mass_transfer_dynamics import (
    rk45_integrate, maxwell_stefan_diffusion,
    simulate_three_component_diffusion,
    langford_deriv, simulate_langford_mixing,
    lorenz96_deriv, simulate_lorenz96_convection,
    distillation_column_deriv, simulate_distillation_dynamics
)
from efficiency_optimizer import (
    fermat_optimize_trays, gilliland_correlation,
    estimate_N_from_R, reboiler_duty,
    total_cost_model, optimize_distillation_cost
)
from packing_simulation import (
    line_packing_simulation, packing_void_fraction,
    packing_efficiency_factor, ergun_pressure_drop,
    simulate_random_packing_column
)
from uncertainty_quantification import (
    hexagon01_sample, hexagon01_area,
    hexagon_monte_carlo_integrate,
    rcont_random_table, random_flow_distribution,
    sobol_first_order_index_mc, uncertainty_propagation_mc
)


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_vle_thermodynamics():
    print_section("1. 多组分汽液平衡(VLE)热力学计算")


    nc = 3
    T = 350.0
    T = thermo_factor_check(T)
    P_total = 101325.0


    A_ant = np.array([8.20417, 8.07131, 8.08097])
    B_ant = np.array([1642.89, 1730.63, 1582.91])
    C_ant = np.array([230.300, 233.426, 239.726])


    V = np.array([5.87e-5, 1.80e-5, 4.07e-5])


    Lambda_ij = np.array([
        [0.0, 155.21, 46.32],
        [292.51, 0.0, 289.19],
        [182.42, 201.21, 0.0]
    ])

    x_feed = np.array([0.4, 0.3, 0.3])

    print(f"  温度 T = {T:.2f} K")
    print(f"  总压 P = {P_total:.2f} Pa")
    print(f"  液相组成 x = {x_feed}")


    y, K, gamma = vle_flash_calculation(x_feed, P_total, T, A_ant, B_ant, C_ant, V, Lambda_ij)
    alpha_rel = vle_relative_volatility(K)

    print(f"  汽相组成 y = {y}")
    print(f"  相平衡常数 K = {K}")
    print(f"  活度系数 γ = {gamma}")
    print(f"  相对挥发度 α = {alpha_rel}")


    x_range = np.linspace(0.0, 1.0, 20)
    V_jac = activity_coefficient_spectral_expansion(x_range, nc, alpha_jac=0.0, beta_jac=0.0, n_modes=6)
    print(f"  Jacobi 谱展开矩阵 shape = {V_jac.shape}")


    def temp_integral(x):
        return np.exp(-0.001 * x) * np.sin(0.1 * x)

    xtab, weight = laguerre_compute(12, alpha=0.0)
    lag_result = laguerre_quadrature_integrate(temp_integral, norder=12, alpha=0.0)
    print(f"  Laguerre-Gauss 求积结果 = {lag_result:.6e}")

    return y, K, gamma, alpha_rel


def demo_property_interpolation():
    print_section("2. 物性插值与传质通量积分")


    z_data = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    T_data = np.array([373.0, 368.0, 362.0, 355.0, 348.0, 340.0])
    x_data = np.array([
        [0.1, 0.8, 0.1],
        [0.2, 0.7, 0.1],
        [0.3, 0.6, 0.1],
        [0.45, 0.45, 0.1],
        [0.6, 0.35, 0.05],
        [0.8, 0.18, 0.02]
    ])
    y_data = np.array([
        [0.3, 0.6, 0.1],
        [0.45, 0.45, 0.1],
        [0.55, 0.38, 0.07],
        [0.65, 0.30, 0.05],
        [0.78, 0.20, 0.02],
        [0.90, 0.09, 0.01]
    ])

    z_query = np.linspace(0.0, 10.0, 11)
    T_interp, x_interp, y_interp = interpolate_vle_data(
        z_data, T_data, x_data, y_data, z_query, p=2.0
    )

    print(f"  查询位置 z = {z_query}")
    print(f"  插值温度 T = {np.round(T_interp, 2)}")


    def mass_flux(z):
        return 0.5 * np.exp(-0.1 * z) * (1.0 + 0.05 * z)

    total_mt = integrate_mass_transfer_flux(z_query, mass_flux)
    print(f"  沿塔高总传质量 = {total_mt:.4e} mol/(m² s)")


    trap_result = quad_trapezoid(mass_flux, 0.0, 10.0, 20)
    print(f"  梯形积分验证 = {trap_result:.4e}")

    return total_mt


def demo_tray_geometry():
    print_section("3. 塔板几何网格与局部Murphree效率")

    tray_width = 1.5
    tray_height = 0.8
    nodes, elements, areas = generate_tray_mesh(tray_width, tray_height, nx=8, ny=5)

    print(f"  塔板尺寸: {tray_width} m × {tray_height} m")
    print(f"  节点数: {len(nodes)}, 单元数: {len(elements)}")


    test_points = np.array([[0.5, 0.4], [1.6, 0.9], [0.75, 0.4]])
    dists = drectangle(test_points, 0.0, tray_width, 0.0, tray_height)
    print(f"  有符号距离测试结果: {dists}")


    q4 = np.array([[0.0, tray_width, tray_width, 0.0],
                   [0.0, 0.0, tray_height, tray_height]])
    rs = np.array([[0.5], [0.5]])
    xy_mapped = reference_to_physical_q4(q4, 1, rs)
    print(f"  参考点 (0.5,0.5) 映射到物理坐标: {xy_mapped[:, 0]}")


    x_liq = np.array([0.5, 0.4, 0.1])
    y_vap = np.array([0.5, 0.3, 0.05])
    K_eq = np.array([1.2, 0.875, 0.5])
    E_local = compute_local_efficiency_on_mesh(nodes, elements, x_liq, y_vap, K_eq)
    E_avg = mesh_average_efficiency(nodes, elements, areas, E_local)
    print(f"  局部效率范围: [{E_local.min():.4f}, {E_local.max():.4f}]")
    print(f"  面积加权平均效率: {E_avg:.4f}")

    return E_avg


def demo_mass_transfer_dynamics(alpha_rel):
    print_section("4. 传质动力学与ODE系统")


    D_matrix = np.array([
        [1e-9, 1.2e-9, 0.8e-9],
        [1.2e-9, 1e-9, 1.1e-9],
        [0.8e-9, 1.1e-9, 1e-9]
    ])
    c_total = 50.0
    y0_diff = np.array([0.4, 0.3, 0.3, 0.0, 0.0, 0.0])
    t, y, e = simulate_three_component_diffusion(y0_diff, D_matrix, c_total, (0.0, 10.0), 100)
    print(f"  Maxwell-Stefan扩散: 初始 x={y0_diff[:3]}, 稳态 x≈{y[-1, :3]}")


    xyz0 = np.array([0.1, 0.1, 0.1])
    t_l, y_l, e_l = simulate_langford_mixing(xyz0, (0.0, 20.0), 200)
    print(f"  Langford混合: 终态 [x,y,z] = {y_l[-1, :]}")


    n_l96 = 20
    y0_l96 = np.ones(n_l96) * 0.5
    y0_l96[0] += 0.01
    t_96, y_96, e_96 = simulate_lorenz96_convection(y0_l96, (0.0, 10.0), 500, force=8.0)
    print(f"  Lorenz96对流: 终态均值={np.mean(y_96[-1, :]):.4f}, 方差={np.var(y_96[-1, :]):.4f}")


    n_trays = 10
    nc = 3
    F = np.zeros(n_trays)
    F[4] = 50.0
    z_feed = np.zeros((n_trays, nc))
    z_feed[4, :] = np.array([0.4, 0.3, 0.3])
    q_feed = np.zeros(n_trays)
    q_feed[4] = 0.5











    print(f"  精馏塔动态: 模拟 {n_trays} 块板, {nc} 个组分")
    print(f"  再沸器轻组分终态: {comp_profiles[-1, 0, :]}")
    print(f"  冷凝器轻组分终态: {comp_profiles[-1, -1, :]}")

    return comp_profiles


def demo_efficiency_optimizer():
    print_section("5. 塔板数与回流比耦合优化")

    N_min = 8
    R_min = 1.5
    D = 50.0
    q_cond = 5e5
    lambda_vap = 35000.0
    feed_rate = 100.0
    z_F, x_D, x_B = 0.4, 0.95, 0.05
    c_steam = 200.0
    t_op = 8000.0
    a_cap = 1e6
    b_cap = 5e4
    column_diameter = 1.2

    N_opt, R_opt, C_min, history = optimize_distillation_cost(
        N_min, R_min, D, q_cond, lambda_vap,
        feed_rate, z_F, x_D, x_B,
        c_steam, t_op, a_cap, b_cap, column_diameter
    )

    print(f"  最小理论塔板数 N_min = {N_min}")
    print(f"  最小回流比 R_min = {R_min}")
    print(f"  优化结果: N_opt = {N_opt}, R_opt = {R_opt:.3f}")
    print(f"  最小年度总成本 C_min = {C_min:.2e} CNY/year")


    residual = gilliland_correlation(R_opt, R_min, N_opt, N_min)
    print(f"  Gilliland 残差 = {residual:.4e}")

    Q_R = reboiler_duty(R_opt, D, q_cond, lambda_vap, feed_rate, z_F, x_D, x_B)
    print(f"  再沸器热负荷 Q_R = {Q_R:.2e} W")

    return N_opt, R_opt, C_min


def demo_packing_simulation():
    print_section("6. 填料塔随机堆积模拟")

    results = simulate_random_packing_column(
        column_diameter=1.0,
        packing_height=3.0,
        packing_diameter=0.05,
        packing_shape_factor=0.8,
        mu=1.8e-5,
        u=1.5,
        rho=2.5,
        n_runs=10
    )

    print(f"  模拟次数: {results['n_runs']}")
    print(f"  平均空隙率 ε = {results['epsilon_mean']:.4f} ± {results['epsilon_std']:.4f}")
    print(f"  平均效率因子 η = {results['eta_mean']:.4f} ± {results['eta_std']:.4f}")
    print(f"  平均压降 ΔP = {results['dP_mean']:.2f} ± {results['dP_std']:.2f} Pa")


    n_parked, density_obs, density_max, positions = line_packing_simulation(
        0.0, 5.0, 0.3, max_attempts=50000
    )
    print(f"  单次线段填充: 放置 {n_parked} 段, 密度 = {density_obs:.4f}")

    return results


def demo_uncertainty_quantification():
    print_section("7. 不确定性量化与敏感性分析")


    def model_in_hexagon(x, y):
        return np.exp(-(x**2 + y**2)) * (1.0 + 0.1 * x * y)

    hex_result = hexagon_monte_carlo_integrate(model_in_hexagon, n_samples=20000)
    print(f"  六边形蒙特卡洛积分 = {hex_result:.6f}")


    n_trays = 5
    nc = 3
    total_flows = np.array([20.0, 100.0, 120.0, 120.0, 20.0])
    component_totals = np.array([90.0, 110.0, 180.0])
    samples = random_flow_distribution(n_trays, nc, total_flows, component_totals, n_samples=3, seed=42)
    print(f"  随机流量分布样本数: {len(samples)}")
    print(f"  样本1各板总流量: {np.sum(samples[0], axis=1)}")


    def simple_model(params):
        return params['alpha'] * params['T']**2 + params['P'] * params['R']

    param_names = ['alpha', 'T', 'P', 'R']
    param_ranges = [(0.5, 2.0), (300.0, 400.0), (1e5, 2e5), (1.5, 5.0)]
    S1, VY = sobol_first_order_index_mc(simple_model, param_names, param_ranges, n_samples=1024)
    print(f"  Sobol 一阶敏感性指标:")
    for name, s in S1.items():
        print(f"    S_{name} = {s:.4f}")


    def cost_model(params):
        return params['reflux'] * 1e5 + params['efficiency'] * 2e6 + np.random.normal(0, 1e4)

    param_distributions = {
        'reflux': ('uniform', (1.5, 4.0)),
        'efficiency': ('normal', (0.75, 0.05))
    }
    mean, std, ci_95 = uncertainty_propagation_mc(cost_model, param_distributions, n_samples=3000)
    print(f"  成本不确定性: 均值={mean:.2e}, 标准差={std:.2e}")
    print(f"  95% 置信区间: [{ci_95[0]:.2e}, {ci_95[1]:.2e}]")

    return S1


def demo_pressure_wave():
    print_section("8. 塔内压力波动传播")

    column_height = 15.0
    c_sound = 85.0
    P_bottom = 120000.0
    P_top = 101325.0
    P_initial = 110000.0
    disturbance_z = 7.5
    disturbance_amp = 5000.0
    t_end = 0.5

    P_field, z_grid, t_grid, alpha = pressure_wave_in_column(
        column_height, c_sound, P_bottom, P_top, P_initial,
        disturbance_z, disturbance_amp, t_end, nz=60, nt=300
    )

    print(f"  塔高: {column_height} m, 声速: {c_sound} m/s")
    print(f"  CFL 数 α = {alpha:.4f}")
    print(f"  初始扰动位置: {disturbance_z} m, 幅值: {disturbance_amp} Pa")
    print(f"  压力场 shape: {P_field.shape}")
    print(f"  t=0 时压力范围: [{P_field[0, :].min():.1f}, {P_field[0, :].max():.1f}] Pa")
    print(f"  t={t_end:.2f}s 时压力范围: [{P_field[-1, :].min():.1f}, {P_field[-1, :].max():.1f}] Pa")

    stability_idx = pressure_stability_index(P_field)
    print(f"  压力稳定性指标: {stability_idx:.2e}")

    return P_field


def main():
    print("\n" + "#" * 70)
    print("#  精馏塔传质与能效优化 — 博士级科学计算平台")
    print("#  领域: 化学工程")
    print("#" * 70)

    np.random.seed(42)
    start_time = time.time()


    y_vle, K_vle, gamma_vle, alpha_rel = demo_vle_thermodynamics()


    total_mt = demo_property_interpolation()


    E_avg = demo_tray_geometry()


    comp_profiles = demo_mass_transfer_dynamics(alpha_rel)


    N_opt, R_opt, C_min = demo_efficiency_optimizer()


    packing_results = demo_packing_simulation()


    S1 = demo_uncertainty_quantification()


    P_field = demo_pressure_wave()

    elapsed = time.time() - start_time
    print("\n" + "#" * 70)
    print(f"#  所有计算完成，耗时 {elapsed:.3f} 秒")
    print("#" * 70)


    print_section("计算结果汇总")
    print(f"  汽液平衡: 相对挥发度 α = {alpha_rel}")
    print(f"  传质通量积分: {total_mt:.4e} mol/(m² s)")
    print(f"  塔板平均效率: {E_avg:.4f}")
    print(f"  优化塔板数: N_opt = {N_opt}, R_opt = {R_opt:.3f}")
    print(f"  填料空隙率: {packing_results['epsilon_mean']:.4f}")
    print(f"  压力稳定性: {pressure_stability_index(P_field):.2e}")
    print(f"\n  [INFO] 所有模块运行正常，无报错。")


if __name__ == "__main__":
    main()
