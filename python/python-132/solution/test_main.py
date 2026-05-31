"""
main.py
=======
精馏塔传质与能效优化 — 统一入口程序。

本程序零参数运行，完成以下博士级科学计算：
1. 多组分汽液平衡(VLE)热力学计算（Jacobi谱展开 + Laguerre-Gauss求积）
2. 传质动力学模拟（Maxwell-Stefan扩散 + RK45积分）
3. 塔板几何网格生成与局部效率计算（四边形Q4等参映射 + 有符号距离函数）
4. 物性插值与传质通量积分（Shepard插值 + 梯形法则）
5. 塔板数与回流比耦合优化（类费马分解搜索 + Gilliland关联）
6. 填料塔随机堆积模拟（线段停车问题 + Ergun压降方程）
7. 操作参数不确定性量化（六边形蒙特卡洛 + 随机列联表 + Sobol敏感性分析）
8. 塔内压力波动传播（一维波动方程有限差分）
9. 局部湍流混合与大尺度对流混沌分析（Langford + Lorenz96 ODE）

运行方式:
    python main.py
"""

import numpy as np
import time

# 导入各模块
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
    """演示汽液平衡热力学计算。"""
    print_section("1. 多组分汽液平衡(VLE)热力学计算")

    # 体系：乙醇-水-甲醇三元体系
    nc = 3
    T = 350.0  # K
    T = thermo_factor_check(T)
    P_total = 101325.0  # Pa

    # Antoine 常数（乙醇, 水, 甲醇）
    A_ant = np.array([8.20417, 8.07131, 8.08097])
    B_ant = np.array([1642.89, 1730.63, 1582.91])
    C_ant = np.array([230.300, 233.426, 239.726])

    # 摩尔体积 [m³/mol]（近似值）
    V = np.array([5.87e-5, 1.80e-5, 4.07e-5])

    # Wilson 参数 Δλ_ij / R [K]
    Lambda_ij = np.array([
        [0.0, 155.21, 46.32],
        [292.51, 0.0, 289.19],
        [182.42, 201.21, 0.0]
    ])

    x_feed = np.array([0.4, 0.3, 0.3])

    print(f"  温度 T = {T:.2f} K")
    print(f"  总压 P = {P_total:.2f} Pa")
    print(f"  液相组成 x = {x_feed}")

    # VLE 闪蒸计算
    y, K, gamma = vle_flash_calculation(x_feed, P_total, T, A_ant, B_ant, C_ant, V, Lambda_ij)
    alpha_rel = vle_relative_volatility(K)

    print(f"  汽相组成 y = {y}")
    print(f"  相平衡常数 K = {K}")
    print(f"  活度系数 γ = {gamma}")
    print(f"  相对挥发度 α = {alpha_rel}")

    # Jacobi 多项式谱展开
    x_range = np.linspace(0.0, 1.0, 20)
    V_jac = activity_coefficient_spectral_expansion(x_range, nc, alpha_jac=0.0, beta_jac=0.0, n_modes=6)
    print(f"  Jacobi 谱展开矩阵 shape = {V_jac.shape}")

    # Laguerre-Gauss 求积：计算温度相关积分
    def temp_integral(x):
        return np.exp(-0.001 * x) * np.sin(0.1 * x)

    xtab, weight = laguerre_compute(12, alpha=0.0)
    lag_result = laguerre_quadrature_integrate(temp_integral, norder=12, alpha=0.0)
    print(f"  Laguerre-Gauss 求积结果 = {lag_result:.6e}")

    return y, K, gamma, alpha_rel


def demo_property_interpolation():
    """演示物性插值与数值积分。"""
    print_section("2. 物性插值与传质通量积分")

    # 离散实验数据：沿塔高的温度与组成分布
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

    # 梯形积分：传质通量
    def mass_flux(z):
        return 0.5 * np.exp(-0.1 * z) * (1.0 + 0.05 * z)

    total_mt = integrate_mass_transfer_flux(z_query, mass_flux)
    print(f"  沿塔高总传质量 = {total_mt:.4e} mol/(m² s)")

    # 直接梯形积分验证
    trap_result = quad_trapezoid(mass_flux, 0.0, 10.0, 20)
    print(f"  梯形积分验证 = {trap_result:.4e}")

    return total_mt


def demo_tray_geometry():
    """演示塔板几何网格与局部效率。"""
    print_section("3. 塔板几何网格与局部Murphree效率")

    tray_width = 1.5  # m
    tray_height = 0.8  # m
    nodes, elements, areas = generate_tray_mesh(tray_width, tray_height, nx=8, ny=5)

    print(f"  塔板尺寸: {tray_width} m × {tray_height} m")
    print(f"  节点数: {len(nodes)}, 单元数: {len(elements)}")

    # 测试有符号距离函数
    test_points = np.array([[0.5, 0.4], [1.6, 0.9], [0.75, 0.4]])
    dists = drectangle(test_points, 0.0, tray_width, 0.0, tray_height)
    print(f"  有符号距离测试结果: {dists}")

    # 四边形映射测试
    q4 = np.array([[0.0, tray_width, tray_width, 0.0],
                   [0.0, 0.0, tray_height, tray_height]])
    rs = np.array([[0.5], [0.5]])
    xy_mapped = reference_to_physical_q4(q4, 1, rs)
    print(f"  参考点 (0.5,0.5) 映射到物理坐标: {xy_mapped[:, 0]}")

    # 局部效率计算
    x_liq = np.array([0.5, 0.4, 0.1])
    y_vap = np.array([0.5, 0.3, 0.05])  # 不等于平衡组成，使效率非零
    K_eq = np.array([1.2, 0.875, 0.5])
    E_local = compute_local_efficiency_on_mesh(nodes, elements, x_liq, y_vap, K_eq)
    E_avg = mesh_average_efficiency(nodes, elements, areas, E_local)
    print(f"  局部效率范围: [{E_local.min():.4f}, {E_local.max():.4f}]")
    print(f"  面积加权平均效率: {E_avg:.4f}")

    return E_avg


def demo_mass_transfer_dynamics(alpha_rel):
    """演示传质动力学模拟。"""
    print_section("4. 传质动力学与ODE系统")

    # 4.1 三组分Maxwell-Stefan扩散
    D_matrix = np.array([
        [1e-9, 1.2e-9, 0.8e-9],
        [1.2e-9, 1e-9, 1.1e-9],
        [0.8e-9, 1.1e-9, 1e-9]
    ])
    c_total = 50.0  # mol/m³
    y0_diff = np.array([0.4, 0.3, 0.3, 0.0, 0.0, 0.0])
    t, y, e = simulate_three_component_diffusion(y0_diff, D_matrix, c_total, (0.0, 10.0), 100)
    print(f"  Maxwell-Stefan扩散: 初始 x={y0_diff[:3]}, 稳态 x≈{y[-1, :3]}")

    # 4.2 Langford 局部混合
    xyz0 = np.array([0.1, 0.1, 0.1])
    t_l, y_l, e_l = simulate_langford_mixing(xyz0, (0.0, 20.0), 200)
    print(f"  Langford混合: 终态 [x,y,z] = {y_l[-1, :]}")

    # 4.3 Lorenz96 对流混沌
    n_l96 = 20
    y0_l96 = np.ones(n_l96) * 0.5
    y0_l96[0] += 0.01
    t_96, y_96, e_96 = simulate_lorenz96_convection(y0_l96, (0.0, 10.0), 500, force=8.0)
    print(f"  Lorenz96对流: 终态均值={np.mean(y_96[-1, :]):.4f}, 方差={np.var(y_96[-1, :]):.4f}")

    # 4.4 精馏塔动态物料平衡
    n_trays = 10
    nc = 3
    F = np.zeros(n_trays)
    F[4] = 50.0  # 第5块板进料 [mol/s]
    z_feed = np.zeros((n_trays, nc))
    z_feed[4, :] = np.array([0.4, 0.3, 0.3])
    q_feed = np.zeros(n_trays)
    q_feed[4] = 0.5
    # 恒摩尔流假设下的合理流量分布
    # 精馏段: L=80, V=120; 提馏段: L=105, V=95
    L = np.array([105.0, 105.0, 105.0, 105.0, 105.0,
                  80.0, 80.0, 80.0, 80.0, 80.0])
    V = np.array([95.0, 95.0, 95.0, 95.0, 95.0,
                  120.0, 120.0, 120.0, 120.0, 120.0])
    holdup = np.full(n_trays, 200.0)  # 增大持液量以提高稳定性
    tray_eff = np.full(n_trays, 0.7)

    x0 = np.tile(np.array([0.33, 0.33, 0.34]), n_trays)
    t_d, y_d, e_d, comp_profiles = simulate_distillation_dynamics(
        n_trays, nc, F, z_feed, q_feed, L, V, holdup,
        alpha_rel, tray_eff, x0, (0.0, 100.0), 2000
    )
    print(f"  精馏塔动态: 模拟 {n_trays} 块板, {nc} 个组分")
    print(f"  再沸器轻组分终态: {comp_profiles[-1, 0, :]}")
    print(f"  冷凝器轻组分终态: {comp_profiles[-1, -1, :]}")

    return comp_profiles


def demo_efficiency_optimizer():
    """演示能效优化。"""
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

    # Gilliland 验证
    residual = gilliland_correlation(R_opt, R_min, N_opt, N_min)
    print(f"  Gilliland 残差 = {residual:.4e}")

    Q_R = reboiler_duty(R_opt, D, q_cond, lambda_vap, feed_rate, z_F, x_D, x_B)
    print(f"  再沸器热负荷 Q_R = {Q_R:.2e} W")

    return N_opt, R_opt, C_min


def demo_packing_simulation():
    """演示填料塔随机堆积模拟。"""
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

    # 单次堆积演示
    n_parked, density_obs, density_max, positions = line_packing_simulation(
        0.0, 5.0, 0.3, max_attempts=50000
    )
    print(f"  单次线段填充: 放置 {n_parked} 段, 密度 = {density_obs:.4f}")

    return results


def demo_uncertainty_quantification():
    """演示不确定性量化。"""
    print_section("7. 不确定性量化与敏感性分析")

    # 7.1 六边形蒙特卡洛积分
    def model_in_hexagon(x, y):
        return np.exp(-(x**2 + y**2)) * (1.0 + 0.1 * x * y)

    hex_result = hexagon_monte_carlo_integrate(model_in_hexagon, n_samples=20000)
    print(f"  六边形蒙特卡洛积分 = {hex_result:.6f}")

    # 7.2 随机流量分布
    n_trays = 5
    nc = 3
    total_flows = np.array([20.0, 100.0, 120.0, 120.0, 20.0])
    component_totals = np.array([90.0, 110.0, 180.0])
    samples = random_flow_distribution(n_trays, nc, total_flows, component_totals, n_samples=3, seed=42)
    print(f"  随机流量分布样本数: {len(samples)}")
    print(f"  样本1各板总流量: {np.sum(samples[0], axis=1)}")

    # 7.3 Sobol 敏感性分析
    def simple_model(params):
        return params['alpha'] * params['T']**2 + params['P'] * params['R']

    param_names = ['alpha', 'T', 'P', 'R']
    param_ranges = [(0.5, 2.0), (300.0, 400.0), (1e5, 2e5), (1.5, 5.0)]
    S1, VY = sobol_first_order_index_mc(simple_model, param_names, param_ranges, n_samples=1024)
    print(f"  Sobol 一阶敏感性指标:")
    for name, s in S1.items():
        print(f"    S_{name} = {s:.4f}")

    # 7.4 不确定性传播
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
    """演示压力波动传播。"""
    print_section("8. 塔内压力波动传播")

    column_height = 15.0  # m
    c_sound = 85.0  # m/s (气相中声速，远低于空气)
    P_bottom = 120000.0  # Pa
    P_top = 101325.0  # Pa
    P_initial = 110000.0  # Pa
    disturbance_z = 7.5  # m
    disturbance_amp = 5000.0  # Pa
    t_end = 0.5  # s

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
    """主程序入口。"""
    print("\n" + "#" * 70)
    print("#  精馏塔传质与能效优化 — 博士级科学计算平台")
    print("#  领域: 化学工程")
    print("#" * 70)

    np.random.seed(42)
    start_time = time.time()

    # 1. VLE 热力学
    y_vle, K_vle, gamma_vle, alpha_rel = demo_vle_thermodynamics()

    # 2. 物性插值
    total_mt = demo_property_interpolation()

    # 3. 塔板几何
    E_avg = demo_tray_geometry()

    # 4. 传质动力学
    comp_profiles = demo_mass_transfer_dynamics(alpha_rel)

    # 5. 能效优化
    N_opt, R_opt, C_min = demo_efficiency_optimizer()

    # 6. 填料模拟
    packing_results = demo_packing_simulation()

    # 7. 不确定性量化
    S1 = demo_uncertainty_quantification()

    # 8. 压力波动
    P_field = demo_pressure_wave()

    elapsed = time.time() - start_time
    print("\n" + "#" * 70)
    print(f"#  所有计算完成，耗时 {elapsed:.3f} 秒")
    print("#" * 70)

    # 汇总输出
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

# ================================================================
# 测试用例（35个，assert模式，涉及随机值均使用固定种子）
# ================================================================

# ---- TC01: jacobi_polynomial 输出形状正确 ----
x_test = np.linspace(-1.0, 1.0, 10)
V = jacobi_polynomial(10, 5, 0.0, 0.0, x_test)
assert V.shape == (10, 6), '[TC01] jacobi_polynomial shape FAILED'
assert np.all(np.isfinite(V)), '[TC01] jacobi_polynomial finite values FAILED'

# ---- TC02: jacobi_polynomial P_0^(0,0)(x)=1 对所有 x ----
V02 = jacobi_polynomial(5, 0, 0.0, 0.0, np.array([-0.5, 0.0, 0.5]))
assert np.allclose(V02[:, 0], 1.0), '[TC02] jacobi P0 identity FAILED'

# ---- TC03: laguerre_compute 返回正确数量的节点和权重数组 ----
xtab, weight = laguerre_compute(10, alpha=0.0)
assert len(xtab) == 10, '[TC03] laguerre_compute node count FAILED'
assert len(weight) == 10, '[TC03] laguerre_compute weight count FAILED'
assert np.all(np.isfinite(xtab)), '[TC03] laguerre_compute nodes finite FAILED'
assert np.all(np.isfinite(weight)), '[TC03] laguerre_compute weights finite FAILED'

# ---- TC04: laguerre_quadrature_integrate 返回有限结果 ----
def f_test(x):
    return x
lag_result = laguerre_quadrature_integrate(f_test, norder=16, alpha=0.0)
assert np.isfinite(lag_result), '[TC04] laguerre_quadrature result finite FAILED'
assert not np.isnan(lag_result), '[TC04] laguerre_quadrature result not NaN FAILED'

# ---- TC05: antoine_vapor_pressure 返回正有限值 ----
Psat = antoine_vapor_pressure(80.0, 8.20417, 1642.89, 230.300)
assert Psat > 0, '[TC05] antoine_vapor_pressure positive FAILED'
assert np.isfinite(Psat), '[TC05] antoine_vapor_pressure finite FAILED'

# ---- TC06: wilson_activity_coefficient 返回正值且形状正确 ----
V_w = np.array([5.87e-5, 1.80e-5, 4.07e-5])
L_ij = np.array([[0.0, 155.21, 46.32], [292.51, 0.0, 289.19], [182.42, 201.21, 0.0]])
x_w = np.array([0.4, 0.3, 0.3])
gamma_w = wilson_activity_coefficient(x_w, V_w, L_ij, 350.0)
assert gamma_w.shape == (3,), '[TC06] wilson gamma shape FAILED'
assert np.all(gamma_w > 0), '[TC06] wilson gamma positive FAILED'
assert np.all(np.isfinite(gamma_w)), '[TC06] wilson gamma finite FAILED'

# ---- TC07: vle_flash_calculation 汽相组成归一化 ----
V_vle = np.array([5.87e-5, 1.80e-5, 4.07e-5])
L_vle = np.array([[0.0, 155.21, 46.32], [292.51, 0.0, 289.19], [182.42, 201.21, 0.0]])
A_vle = np.array([8.20417, 8.07131, 8.08097])
B_vle = np.array([1642.89, 1730.63, 1582.91])
C_vle = np.array([230.300, 233.426, 239.726])
y, K, gamma = vle_flash_calculation(np.array([0.4, 0.3, 0.3]), 101325.0, 350.0, A_vle, B_vle, C_vle, V_vle, L_vle)
assert np.abs(np.sum(y) - 1.0) < 1e-10, '[TC07] vle y sum FAILED'
assert np.all(y >= 0), '[TC07] vle y nonnegative FAILED'

# ---- TC08: vle_relative_volatility 最大值为 1.0 ----
alpha_rel = vle_relative_volatility(K)
assert np.abs(np.max(alpha_rel) - 1.0) < 1e-12, '[TC08] relative volatility max FAILED'

# ---- TC09: activity_coefficient_spectral_expansion 输出形状正确 ----
x_range = np.linspace(0.0, 1.0, 15)
V_jac09 = activity_coefficient_spectral_expansion(x_range, 3, alpha_jac=0.0, beta_jac=0.0, n_modes=5)
assert V_jac09.shape == (15, 6), '[TC09] spectral expansion shape FAILED'

# ---- TC10: shepard_interp_1d 在数据点处精确插值 ----
xd = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
yd = np.array([373.0, 368.0, 362.0, 355.0, 348.0, 340.0])
yi_exact = shepard_interp_1d(6, xd, yd, 2.0, 1, np.array([4.0]))
assert np.abs(yi_exact[0] - 362.0) < 1e-6, '[TC10] shepard exact match FAILED'

# ---- TC11: quad_trapezoid 积分 f(x)=x 从 0 到 1 得 0.5 ----
trap_result11 = quad_trapezoid(lambda x: x, 0.0, 1.0, 100)
assert np.abs(trap_result11 - 0.5) < 1e-10, '[TC11] trapezoid x integral FAILED'

# ---- TC12: shepard_interp_1d p=0 时返回等权平均 ----
yi_avg = shepard_interp_1d(3, np.array([0.0, 1.0, 2.0]), np.array([10.0, 20.0, 30.0]), 0.0, 1, np.array([0.5]))
assert np.abs(yi_avg[0] - 20.0) < 1e-10, '[TC12] shepard p=0 average FAILED'

# ---- TC13: drectangle 内部点返回负值 ----
test_pt_inside = np.array([[0.5, 0.4]])
d_inside = drectangle(test_pt_inside, 0.0, 1.5, 0.0, 0.8)
assert d_inside[0] < 0, '[TC13] drectangle inside FAILED'

# ---- TC14: drectangle 外部点返回正值 ----
test_pt_outside = np.array([[2.0, 0.5]])
d_outside = drectangle(test_pt_outside, 0.0, 1.5, 0.0, 0.8)
assert d_outside[0] > 0, '[TC14] drectangle outside FAILED'

# ---- TC15: reference_to_physical_q4 角点 (0,0) 映射到第一顶点 ----
q4 = np.array([[0.0, 1.5, 1.5, 0.0], [0.0, 0.0, 0.8, 0.8]])
rs_corner = np.array([[0.0], [0.0]])
xy_corner = reference_to_physical_q4(q4, 1, rs_corner)
assert np.allclose(xy_corner[:, 0], np.array([0.0, 0.0])), '[TC15] q4 corner mapping FAILED'

# ---- TC16: generate_tray_mesh 节点数和单元数正确 ----
nodes, elements, areas = generate_tray_mesh(1.5, 0.8, nx=8, ny=5)
assert nodes.shape == (54, 2), '[TC16] tray mesh node count FAILED'
assert len(elements) == 40, '[TC16] tray mesh element count FAILED'

# ---- TC17: quad_trapezoid 积分 sin(x) 0 到 pi 得 2.0 ----
trap_result17 = quad_trapezoid(lambda x: np.sin(x), 0.0, np.pi, 200)
assert np.abs(trap_result17 - 2.0) < 1e-4, '[TC17] trapezoid sin integral FAILED'

# ---- TC18: fd1d_wave_solve 输出形状正确且 alpha 有限 ----
def P_x1(t):
    return 120000.0
def P_x2(t):
    return 101325.0
def P_t1(z):
    return np.full_like(z, 110000.0)
def Pt_t1(z):
    return np.zeros_like(z)
P_field, alpha_wave = fd1d_wave_solve(20, 0.0, 15.0, 100, 0.0, 0.5, 85.0, P_x1, P_x2, P_t1, Pt_t1)
assert P_field.shape == (101, 21), '[TC18] wave solve shape FAILED'
assert np.isfinite(alpha_wave), '[TC18] wave alpha finite FAILED'
assert np.all(np.isfinite(P_field)), '[TC18] wave field finite FAILED'

# ---- TC19: rk45_integrate 指数衰减 ODE ----
def exp_decay(t, y):
    return -0.5 * y
t_rk, y_rk, e_rk = rk45_integrate(exp_decay, (0.0, 5.0), np.array([1.0]), 100)
assert len(t_rk) == 101, '[TC19] rk45 time length FAILED'
assert y_rk[-1, 0] > 0, '[TC19] rk45 decay positive FAILED'
assert y_rk[-1, 0] < y_rk[0, 0], '[TC19] rk45 monotonic decay FAILED'

# ---- TC20: langford_deriv 返回 3 维有限值 ----
import numpy as np
np.random.seed(42)
xyz0 = np.array([0.1, -0.2, 0.05])
deriv20 = langford_deriv(0.0, xyz0)
assert deriv20.shape == (3,), '[TC20] langford deriv shape FAILED'
assert np.all(np.isfinite(deriv20)), '[TC20] langford deriv finite FAILED'

# ---- TC21: lorenz96_deriv 输出形状匹配输入 ----
import numpy as np
np.random.seed(42)
y_l96_21 = np.random.randn(20) * 0.1 + 0.5
deriv_l96 = lorenz96_deriv(0.0, y_l96_21, n=20, force=8.0)
assert deriv_l96.shape == (20,), '[TC21] lorenz96 deriv shape FAILED'
assert np.all(np.isfinite(deriv_l96)), '[TC21] lorenz96 deriv finite FAILED'

# ---- TC22: gilliland_correlation 残差有限 ----
residual = gilliland_correlation(3.0, 1.5, 25, 8)
assert np.isfinite(residual), '[TC22] Gilliland residual finite FAILED'

# ---- TC23: reboiler_duty 返回正值 ----
Q_R = reboiler_duty(3.0, 50.0, 5e5, 35000.0, 100.0, 0.4, 0.95, 0.05)
assert Q_R > 0, '[TC23] reboiler duty positive FAILED'
assert np.isfinite(Q_R), '[TC23] reboiler duty finite FAILED'

# ---- TC24: estimate_N_from_R N > N_min ----
N_est = estimate_N_from_R(3.0, 1.5, 8)
assert N_est > 8, '[TC24] estimate N from R FAILED'

# ---- TC25: packing_void_fraction 在 [0.2, 0.98] 范围内 ----
eps = packing_void_fraction(1000, 0.05, 1.0, 3.0, 0.8)
assert 0.2 <= eps <= 0.98, '[TC25] void fraction range FAILED'

# ---- TC26: ergun_pressure_drop 返回正值 ----
dP_ergun = ergun_pressure_drop(0.7, 1.8e-5, 1.5, 2.5, 0.05, 3.0)
assert dP_ergun > 0, '[TC26] Ergun dP positive FAILED'
assert np.isfinite(dP_ergun), '[TC26] Ergun dP finite FAILED'

# ---- TC27: hexagon01_area 等于 3*sqrt(3)/2 ----
area_hex = hexagon01_area()
expected_area = 3.0 * np.sqrt(3.0) / 2.0
assert np.abs(area_hex - expected_area) < 1e-12, '[TC27] hexagon area FAILED'

# ---- TC28: hexagon_monte_carlo_integrate 常数函数积分 ----
import numpy as np
np.random.seed(42)
const_result = hexagon_monte_carlo_integrate(lambda x, y: 1.0, n_samples=10000)
assert np.abs(const_result - expected_area) / expected_area < 0.03, '[TC28] hexagon MC constant FAILED'

# ---- TC29: rcont_random_table 行和列和匹配 ----
nrowt = np.array([20, 100, 120, 120, 20])
ncolt = np.array([90, 110, 180])
mat_rcont = rcont_random_table(5, 3, nrowt, ncolt, seed=100)
assert np.allclose(np.sum(mat_rcont, axis=1), nrowt), '[TC29] rcont row sums FAILED'
assert np.sum(mat_rcont) > 0, '[TC29] rcont total positive FAILED'

# ---- TC30: interpolate_vle_data 输出形状正确 ----
z_data = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
T_data = np.array([373.0, 368.0, 362.0, 355.0, 348.0, 340.0])
x_data_30 = np.array([[0.1, 0.8, 0.1], [0.2, 0.7, 0.1], [0.3, 0.6, 0.1],
                      [0.45, 0.45, 0.1], [0.6, 0.35, 0.05], [0.8, 0.18, 0.02]])
y_data_30 = np.array([[0.3, 0.6, 0.1], [0.45, 0.45, 0.1], [0.55, 0.38, 0.07],
                      [0.65, 0.30, 0.05], [0.78, 0.20, 0.02], [0.90, 0.09, 0.01]])
z_query = np.linspace(0.0, 10.0, 5)
T_interp, x_interp, y_interp = interpolate_vle_data(z_data, T_data, x_data_30, y_data_30, z_query)
assert T_interp.shape == (5,), '[TC30] interp T shape FAILED'
assert x_interp.shape == (5, 3), '[TC30] interp x shape FAILED'
assert y_interp.shape == (5, 3), '[TC30] interp y shape FAILED'

# ---- TC31: integrate_mass_transfer_flux 返回有限值 ----
def mass_flux(z):
    return 0.5 * np.exp(-0.1 * z)
z_nodes = np.linspace(0.0, 10.0, 21)
total_mt = integrate_mass_transfer_flux(z_nodes, mass_flux)
assert total_mt > 0, '[TC31] mass transfer flux positive FAILED'
assert np.isfinite(total_mt), '[TC31] mass transfer flux finite FAILED'

# ---- TC32: compute_local_efficiency_on_mesh 效率在 [0,1] ----
nodes32, elements32, areas32 = generate_tray_mesh(1.5, 0.8, nx=5, ny=4)
x_liq32 = np.array([0.5, 0.4, 0.1])
y_vap32 = np.array([0.5, 0.3, 0.05])
K_eq32 = np.array([1.2, 0.875, 0.5])
E_local32 = compute_local_efficiency_on_mesh(nodes32, elements32, x_liq32, y_vap32, K_eq32)
assert np.all(E_local32 >= 0.0), '[TC32] local efficiency lower bound FAILED'
assert np.all(E_local32 <= 1.0), '[TC32] local efficiency upper bound FAILED'

# ---- TC33: mesh_average_efficiency 在 [0,1] ----
E_avg33 = mesh_average_efficiency(nodes32, elements32, areas32, E_local32)
assert 0.0 <= E_avg33 <= 1.0, '[TC33] avg efficiency range FAILED'

# ---- TC34: relative_change 对称性 ----
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.0, 2.0, 3.0])
rc = relative_change(a, b)
assert np.abs(rc) < 1e-12, '[TC34] relative_change identical FAILED'

# ---- TC35: thermo_factor_check 在区间内不变 ----
T_val = 400.0
T_checked = thermo_factor_check(T_val)
assert T_checked == 400.0, '[TC35] thermo factor in bounds FAILED'

print('\n全部 35 个测试通过!\n')
