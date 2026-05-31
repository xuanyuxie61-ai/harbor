"""
main.py
生态建模：渔业资源管理与最优捕捞策略 —— 统一入口

本程序综合调用所有子模块，完成从种群动态建模、空间生态分析、
最优捕捞策略计算、不确定性风险评估到生态系统状态转换模拟的
完整博士级科研计算流程。

运行方式：
    python main.py
无需任何命令行参数。
"""

import numpy as np
import time

# 导入所有科研模块
from utils import NumericalConfig
from recruitment_models import (
    sigmoid, sigmoid_derivative, beverton_holt, ricker_recruitment,
    sigmoid_allee_recruitment, recruitment_derivative
)
from vandermonde_interp import vandermonde_interp_1d, pvand, bidim_vandermonde_solve
from population_dynamics import (
    fem1d_bvp_quadratic, fem1d_nonlinear_picard_newton,
    solve_age_structured_steady_state, l2_error_quadratic
)
from spatial_ecology import (
    divergence_free_velocity, sphere_stereograph, sphere_stereograph_inverse,
    icosahedron_vertices, simulate_larval_dispersal
)
from optimal_harvest import (
    BrentOptimizer, schaefer_gordon_steady_state,
    find_optimal_effort, mpa_network_optimize, reconstruct_path
)
from uncertainty_quantification import (
    hammersley_sequence, monte_carlo_integral_1d, fishery_risk_assessment
)
from habitat_integration import (
    integrate_cube_domain, integrate_pyramid_domain,
    integrate_line_profile, estimate_total_biomass_cube
)
from stock_clustering import cluster_habitat_zones
from regime_shift import (
    simulate_regime_shift, fishery_forcing, compute_regime_shift_time,
    energy_functional
)


def print_section(title):
    """打印格式化章节标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_recruitment_models():
    """演示：鱼类种群补充量模型与 Sigmoid-Allee 效应"""
    print_section("1. 鱼类种群补充量模型（Sigmoid-Allee 修正）")

    S_values = np.linspace(0.0, 10.0, 21)
    alpha, beta = 2.0, 0.3
    S_crit = 1.5
    steepness = 8.0

    print("\n亲体量 S | Beverton-Holt | Ricker | Sigmoid-Allee")
    print("-" * 55)
    for S in S_values[::4]:
        R_bh = beverton_holt(S, alpha, beta)
        R_ricker = ricker_recruitment(S, alpha, beta)
        R_allee = sigmoid_allee_recruitment(S, alpha, beta, S_crit, steepness)
        print(f"  {S:6.2f}  |  {R_bh:10.4f}  | {R_ricker:8.4f} | {R_allee:12.4f}")

    # Sigmoid 高阶导数演示
    x_test = 0.5
    print(f"\nSigmoid 函数在 x={x_test} 处的各阶导数：")
    for n in range(1, 6):
        d = sigmoid_derivative(n, x_test)
        print(f"  s^({n})({x_test}) = {d:12.6e}")

    # 计算 Allee 模型的导数
    dR_dS = recruitment_derivative(3.0, alpha, beta, S_crit, steepness, 'allee')
    print(f"\nSigmoid-Allee 模型在 S=3.0 处的导数 dR/dS = {dR_dS:.6f}")
    print("  （导数用于种群稳定性分析和最优控制）")


def demo_vandermonde_interpolation():
    """演示：Vandermonde 快速求解与年龄-体长插值"""
    print_section("2. Vandermonde 快速算法：年龄-体长关系插值")

    # 构造一个已知精确多项式来验证 Vandermonde 插值
    # 精确多项式: p(x) = 10 + 20x - 15x^2 + 5x^3
    test_nodes = np.array([0.1, 0.25, 0.4, 0.6, 0.85])
    test_values = 10.0 + 20.0 * test_nodes - 15.0 * (test_nodes ** 2) + 5.0 * (test_nodes ** 3)
    eval_points = np.linspace(0.0, 1.0, 50)
    interp_values = vandermonde_interp_1d(test_nodes, test_values, eval_points)

    # 验证插值精度
    check_values = vandermonde_interp_1d(test_nodes, test_values, test_nodes)
    max_err = np.max(np.abs(check_values - test_values))
    exact_at_mid = 10.0 + 20.0 * 0.5 - 15.0 * 0.25 + 5.0 * 0.125
    print(f"\n验证 Vandermonde 插值（已知精确多项式）:")
    print(f"  插值节点数: {len(test_nodes)}")
    print(f"  插值回代最大误差: {max_err:.2e}")
    print(f"  x=0.5 处插值结果: {interp_values[25]:.4f} (精确值: {exact_at_mid:.4f})")
    print(f"  x=0.9 处插值结果: {interp_values[45]:.4f}")

    # 演示 Bjorck-Pereyra 算法求解
    n = 5
    alpha_nodes = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b_rhs = np.array([10.0, 25.0, 50.0, 85.0, 130.0])
    x_sol = pvand(n, alpha_nodes, b_rhs)
    print(f"\nBjorck-Pereyra 求解 Vandermonde 系统:")
    print(f"  解向量: {x_sol}")


def demo_population_dynamics():
    """演示：年龄结构种群稳态 FEM 求解"""
    print_section("3. 年龄结构种群稳态分布（二次 FEM 求解）")

    L_age = 20.0
    n_nodes = 41

    # 年龄相关自然死亡率（随年龄增加）
    def mortality_age(a):
        return 0.1 + 0.02 * a

    recruitment_rate = 1000.0
    diffusion_age = 0.5

    a_nodes, N_dist = solve_age_structured_steady_state(
        L_age, n_nodes, mortality_age, recruitment_rate, diffusion_age
    )

    total_population = np.trapezoid(N_dist, a_nodes)
    print(f"\n最大年龄: {L_age} 年")
    print(f"节点数: {n_nodes}")
    print(f"总种群数量（积分）: {total_population:.2f}")
    print(f"0 龄补充量密度: {N_dist[0]:.2f}")
    print(f"10 龄存活密度: {N_dist[n_nodes // 2]:.4f}")
    print(f"20 龄存活密度: {N_dist[-1]:.6f}")

    # 非线性 FEM 演示
    print("\n--- 非线性密度依赖种群分布 ---")
    x_nodes = np.linspace(0.0, 1.0, 21)

    def p_func(x):
        return 0.1 + 0.05 * x

    def q_func(x):
        return 1.0 + 0.5 * x

    def f_func(x):
        return 10.0 * np.exp(-2.0 * x)

    u_nonlinear, iters, resid = fem1d_nonlinear_picard_newton(
        21, x_nodes, p_func, q_func, f_func,
        nonlinear_coeff=0.5, max_iter=30, tol=1e-10
    )

    print(f"非线性 FEM 迭代次数: {iters}")
    print(f"最终残差范数: {resid:.2e}")
    print(f"x=0.5 处种群密度: {u_nonlinear[10]:.4f}")


def demo_spatial_ecology():
    """演示：海洋流场与鱼卵扩散模拟"""
    print_section("4. 海洋流场与仔稚鱼被动扩散模拟")

    # 生成无散度速度场
    nx, ny = 21, 21
    X_grid = np.linspace(0.0, 1.0, nx)
    Y_grid = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(X_grid, Y_grid, indexing='ij')

    C_param = 1.0
    U, V = divergence_free_velocity(nx * ny, X.flatten(), Y.flatten(), C_param)
    U = U.reshape(nx, ny)
    V = V.reshape(nx, ny)

    # 验证无散度条件
    div_max = np.max(np.abs(
        np.gradient(U, 1.0 / (nx - 1), axis=0) +
        np.gradient(V, 1.0 / (ny - 1), axis=1)
    ))
    print(f"\n无散度速度场验证:")
    max_speed = np.max(np.sqrt(U**2 + V**2))
    print(f"  最大速度幅值: {max_speed:.3f}")
    print(f"  最大散度数值: {div_max:.2e} (相对误差: {div_max/(max_speed+1e-10):.2e})")

    # 立体投影演示
    vertices = icosahedron_vertices()
    projected = sphere_stereograph(vertices)
    recovered = sphere_stereograph_inverse(projected)
    recon_err = np.max(np.linalg.norm(vertices - recovered, axis=1))
    print(f"\n球面立体投影:")
    print(f"  二十面体顶点数: {len(vertices)}")
    print(f"  逆投影重构误差: {recon_err:.2e}")

    # 鱼卵扩散模拟（简化版）
    print("\n--- 鱼卵被动扩散模拟 ---")
    C_final, times, C_history = simulate_larval_dispersal(
        nx=11, ny=11, Lx=1.0, Ly=1.0,
        C0_center=(0.3, 0.3), C0_sigma=0.1,
        U=U[::2, ::2], V=V[::2, ::2],
        D=0.005, T_total=0.5, dt=0.001,
        lambda_mortality=0.1
    )

    total_initial = np.sum(C_history[0])
    total_final = np.sum(C_final)
    print(f"  初始总浓度: {total_initial:.4f}")
    print(f"  最终总浓度（经扩散和死亡）: {total_final:.4f}")
    print(f"  浓度保留率: {100 * total_final / total_initial:.1f}%")


def demo_optimal_harvest():
    """演示：最优捕捞策略与 MPA 网络优化"""
    print_section("5. 最优捕捞策略与海洋保护区网络优化")

    # Schaefer-Gordon 模型参数
    r = 0.4      # 内禀增长率 (1/年)
    K = 1000.0   # 环境承载力 (吨)
    q = 0.01     # 可捕系数
    p = 500.0    # 单位鱼价 (元/吨)
    c = 2000.0   # 单位捕捞成本 (元/单位努力量)
    delta = 0.05 # 贴现率

    # 计算不同努力量下的稳态生物量和产量
    print("\nSchaefer-Gordon 模型稳态分析:")
    print("  E    |  B*(E)  |  Y(E)  |  Profit")
    print("-" * 45)
    for E in [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0]:
        B = schaefer_gordon_steady_state(E, r, K, q)
        Y = q * E * B
        profit = p * Y - c * E
        print(f"  {E:4.1f} | {B:7.1f} | {Y:6.1f} | {profit:8.1f}")

    # Brent 优化求解最优努力量
    E_opt, profit_opt = find_optimal_effort(r, K, q, p, c, delta, T=50.0)
    B_opt = schaefer_gordon_steady_state(E_opt, r, K, q)
    Y_opt = q * E_opt * B_opt

    print(f"\n--- Brent 优化结果 ---")
    print(f"  最优捕捞努力量 E* = {E_opt:.4f}")
    print(f"  稳态生物量 B*(E*) = {B_opt:.2f} 吨")
    print(f"  稳态产量 Y(E*) = {Y_opt:.2f} 吨/年")
    print(f"  最大贴现利润 Π* = {profit_opt:.2f} 元")

    # MSY 参考值
    E_msy = r / (2.0 * q)
    Y_msy = r * K / 4.0
    print(f"\n  MSY 参考值: E_MSY = {E_msy:.2f}, Y_MSY = {Y_msy:.2f}")

    # MPA 网络优化
    print("\n--- 海洋保护区网络连通性优化 ---")
    n_patches = 5
    # 构建连通性矩阵（负值表示生态收益/补贴）
    connectivity = np.array([
        [0.0, 2.5, 4.0, np.inf, 3.0],
        [2.5, 0.0, 1.5, 3.5, np.inf],
        [4.0, 1.5, 0.0, 2.0, 4.5],
        [np.inf, 3.5, 2.0, 0.0, 1.8],
        [3.0, np.inf, 4.5, 1.8, 0.0]
    ], dtype=float)

    dist, predecessor = mpa_network_optimize(n_patches, connectivity, source_patch=0)
    print("  从保护区 0 到各区的最优生态路径成本:")
    for i in range(n_patches):
        path = reconstruct_path(predecessor, i)
        print(f"    到保护区 {i}: 成本 = {dist[i]:.2f}, 路径 = {path}")


def demo_uncertainty_quantification():
    """演示：不确定性量化与风险评估"""
    print_section("6. 渔业资源评估不确定性量化（QMC）")

    # Hammersley 序列生成
    print("\nHammersley 低差异序列示例（3维，前5个点）:")
    points = hammersley_sequence(0, 4, m=3, n=1000)
    for i in range(min(5, len(points))):
        print(f"  点 {i}: [{points[i, 0]:.6f}, {points[i, 1]:.6f}, {points[i, 2]:.6f}]")

    # 1D 积分对比：MC vs QMC
    def test_func(x):
        return np.sin(2.0 * np.pi * x) + 0.5 * x ** 2

    exact_integral = 1.0 / 6.0  # 在 [0,1] 上的精确值: ∫sin(2πx)=0, ∫0.5x²=1/6
    mc_est = monte_carlo_integral_1d(test_func, 0.0, 1.0, 1000, method='mc')
    qmc_est = monte_carlo_integral_1d(test_func, 0.0, 1.0, 1000, method='qmc')

    print(f"\n一维积分对比 [0,1]: sin(2πx) + 0.5x²")
    print(f"  精确值: {exact_integral:.6f}")
    print(f"  MC 估计 (N=1000): {mc_est:.6f}, 误差 = {abs(mc_est - exact_integral):.2e}")
    print(f"  QMC估计 (N=1000): {qmc_est:.6f}, 误差 = {abs(qmc_est - exact_integral):.2e}")

    # 渔业风险评估
    print("\n--- 渔业参数不确定性风险评估 ---")
    r_dist = (0.4, 0.08)   # r ~ LogNormal(μ=0.4, σ=0.08)
    K_dist = (1000.0, 200.0)
    q_dist = (0.01, 0.002)
    E_fixed = 15.0

    risk = fishery_risk_assessment(
        r_dist, K_dist, q_dist, E_fixed,
        p=500.0, c=2000.0, delta=0.05,
        n_samples=2000, method='qmc'
    )

    print(f"  期望利润: {risk['expected_profit']:.2f} 元")
    print(f"  利润标准差: {risk['std_profit']:.2f} 元")
    print(f"  利润变异系数: {risk['profit_cv']:.4f}")
    print(f"  生物量低于安全限概率: {risk['prob_biomass_below_limit']:.4f}")
    print(f"  经济亏损概率: {risk['prob_negative_profit']:.4f}")
    print(f"  期望生物量: {risk['expected_biomass']:.2f} 吨")
    print(f"  生物量 5% 分位数: {risk['biomass_percentile_5']:.2f} 吨")
    print(f"  生物量 95% 分位数: {risk['biomass_percentile_95']:.2f} 吨")


def demo_habitat_integration():
    """演示：三维栖息地生物量积分"""
    print_section("7. 三维栖息地生物量体积积分")

    # 定义生物量密度函数：中心密集、边缘稀疏
    def density_func(pts):
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        r_sq = x ** 2 + y ** 2 + (0.5 * z) ** 2
        return 100.0 * np.exp(-r_sq / 4.0)

    # 立方体域积分
    cube_biomass = integrate_cube_domain(
        density_func, degree=5, scale=2.0, shift=np.array([0.0, 0.0, 0.0])
    )
    print(f"\n立方体域 [-2,2]³ 内总生物量: {cube_biomass:.4f} 吨")

    # 金字塔域积分
    pyramid_biomass = integrate_pyramid_domain(
        lambda pts: 50.0 * (1.0 - pts[:, 2]), degree=5
    )
    print(f"单位金字塔域内总生物量: {pyramid_biomass:.4f} 吨")

    # 深度剖面积分
    depth_integral = integrate_line_profile(
        lambda z: 20.0 * np.exp(0.1 * z), a=-100.0, b=0.0, order=5
    )
    print(f"深度剖面 [-100m, 0m] 生物量积分: {depth_integral:.4f} 吨")

    # 综合估算
    domain_bounds = ((-5.0, 5.0), (-5.0, 5.0), (-50.0, 0.0))
    total_bio = estimate_total_biomass_cube(
        lambda pts: 80.0 * np.exp(-(pts[:, 0] ** 2 + pts[:, 1] ** 2) / 10.0) *
                    np.exp(pts[:, 2] / 20.0),
        domain_bounds, degree=5
    )
    print(f"综合海域 (10×10×50 km³) 总生物量估算: {total_bio:.2f} 吨")


def demo_stock_clustering():
    """演示：栖息地快速聚类"""
    print_section("8. 渔业栖息地快速 K-Means 聚类")

    np.random.seed(42)
    n_stations = 200

    # 生成模拟环境数据：温度、盐度、深度、叶绿素
    # 模拟 4 个不同的生态区
    zone_centers = np.array([
        [18.0, 35.0, 50.0, 2.0],   # 近岸暖水区
        [12.0, 34.0, 200.0, 0.5],  # 外海冷水区
        [22.0, 36.0, 30.0, 3.5],   # 河口富营养区
        [15.0, 34.5, 120.0, 1.2]   # 过渡区
    ])

    env_features = []
    true_labels = []
    for i, center in enumerate(zone_centers):
        n_zone = n_stations // 4
        features = center + np.random.randn(n_zone, 4) * np.array([2.0, 0.5, 30.0, 0.5])
        env_features.append(features)
        true_labels.extend([i] * n_zone)

    env_features = np.vstack(env_features)
    true_labels = np.array(true_labels)

    zones, centers_norm, zone_stats = cluster_habitat_zones(
        n_stations, env_features, n_zones=4
    )

    print(f"\n聚类站点数: {n_stations}")
    print(f"分区数: 4")
    print(f"聚类惯性 (Inertia): {zone_stats['inertia']:.2f}")
    print(f"各区样本数: {zone_stats['n_points_per_zone']}")

    # 计算聚类纯度（与真实标签对比）
    from collections import Counter
    purity = 0.0
    for j in range(4):
        mask = zones == j
        if np.sum(mask) > 0:
            true_in_zone = true_labels[mask]
            most_common = Counter(true_in_zone).most_common(1)[0][1]
            purity += most_common
    purity /= len(true_labels)
    print(f"聚类纯度 (与真实分区对比): {purity:.4f}")


def demo_regime_shift():
    """演示：生态系统状态转换模拟"""
    print_section("9. Allen-Cahn 相场模型：渔业生态系统状态转换")

    nx = 101
    x_min, x_max = 0.0, 10.0
    nu = 0.1
    xi = 0.5
    T_total = 5.0
    dt = 0.001

    # 初始条件：高生物量态（u > 0）占主导
    x = np.linspace(x_min, x_max, nx)
    u_init = np.tanh((x - 3.0) / (np.sqrt(2.0) * xi))

    # 无强迫情况
    u_final, t_hist, u_hist = simulate_regime_shift(
        x_min, x_max, nx, u_init, nu, xi, T_total, dt,
        save_interval=500
    )

    energy_init = energy_functional(u_init, (x_max - x_min) / (nx - 1), nu, xi)
    energy_final = energy_functional(u_final, (x_max - x_min) / (nx - 1), nu, xi)

    print(f"\n无强迫情况:")
    print(f"  初始自由能: {energy_init:.4f}")
    print(f"  最终自由能: {energy_final:.4f}")
    print(f"  系统演化趋势: {'稳定' if energy_final <= energy_init else '不稳定'}")
    print(f"  高生物量区占比: {np.mean(u_final > 0):.2%}")

    # 有捕捞强迫的情况
    def forcing(t, u):
        return fishery_forcing(t, u, E_t=25.0, epsilon=0.3, q=0.01, K=1000.0)

    u_final_f, t_hist_f, u_hist_f = simulate_regime_shift(
        x_min, x_max, nx, u_init, nu, xi, T_total, dt,
        forcing_func=forcing, save_interval=500
    )

    energy_final_f = energy_functional(u_final_f, (x_max - x_min) / (nx - 1), nu, xi)

    print(f"\n有捕捞强迫 (E=25) 情况:")
    print(f"  最终自由能: {energy_final_f:.4f}")
    print(f"  高生物量区占比: {np.mean(u_final_f > 0):.2%}")

    shift_time = compute_regime_shift_time(u_hist_f, threshold=0.0)
    if shift_time is not None:
        print(f"  状态转换发生时间步: {shift_time}")
    else:
        print(f"  在模拟时间内未发生完全状态转换")


def main():
    """主函数：执行全部演示流程"""
    print("\n" + "#" * 70)
    print("#  生态建模：渔业资源管理与最优捕捞策略")
    print("#  博士级科研代码合成项目 —— 统一计算入口")
    print("#" * 70)
    print(f"\n运行时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数值精度 (EPS): {NumericalConfig.EPS:.2e}")

    t_start = time.time()

    demo_recruitment_models()
    demo_vandermonde_interpolation()
    demo_population_dynamics()
    demo_spatial_ecology()
    demo_optimal_harvest()
    demo_uncertainty_quantification()
    demo_habitat_integration()
    demo_stock_clustering()
    demo_regime_shift()

    t_elapsed = time.time() - t_start

    print("\n" + "#" * 70)
    print(f"#  全部计算完成，总耗时: {t_elapsed:.3f} 秒")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（27个，assert模式，涉及随机值均使用固定种子）
# ================================================================

# ---- TC01: sigmoid(0) 精确等于 0.5 ----
result = sigmoid(0.0)
assert abs(result - 0.5) < 1e-12, '[TC01] sigmoid(0) 应精确等于 0.5 FAILED'

# ---- TC02: sigmoid 极大值数值稳定性 ----
result = sigmoid(1000.0)
assert np.isfinite(result) and result > 0.999999, '[TC02] sigmoid 极大值应趋近于 1 且不溢出 FAILED'

# ---- TC03: sigmoid 对称性验证 s(-x) = 1 - s(x) ----
x_test = np.array([-2.0, -1.0, 0.5, 3.0])
left = sigmoid(-x_test)
right = 1.0 - sigmoid(x_test)
assert np.allclose(left, right, atol=1e-12), '[TC03] sigmoid 对称性 s(-x)=1-s(x) FAILED'

# ---- TC04: Beverton-Holt S=0 时补充量为 0 ----
R0 = beverton_holt(0.0, 2.0, 0.3)
assert abs(R0) < 1e-12, '[TC04] Beverton-Holt 在 S=0 时补充量应为 0 FAILED'

# ---- TC05: Beverton-Holt 大 S 渐近值趋近 alpha/beta ----
R_large = beverton_holt(1e6, 2.0, 0.3)
assert abs(R_large - 2.0 / 0.3) < 1e-4, '[TC05] Beverton-Holt 大 S 渐近值应为 alpha/beta FAILED'

# ---- TC06: Ricker 在 S=0 时补充量为 0 ----
R0_ricker = ricker_recruitment(0.0, 2.0, 0.3)
assert abs(R0_ricker) < 1e-12, '[TC06] Ricker 在 S=0 时补充量应为 0 FAILED'

# ---- TC07: Ricker 最优亲体量处取最大值解析验证 ----
alpha, beta = 2.0, 0.3
S_opt = 1.0 / beta
R_max = ricker_recruitment(S_opt, alpha, beta)
expected_max = alpha / (np.e * beta)
assert abs(R_max - expected_max) < 1e-10, '[TC07] Ricker 最优亲体量处最大值解析验证 FAILED'

# ---- TC08: Sigmoid-Allee 补充量非负性 ----
S_vals = np.linspace(0.0, 10.0, 11)
R_allee = sigmoid_allee_recruitment(S_vals, 2.0, 0.3, 1.5, 8.0)
assert np.all(R_allee >= 0.0), '[TC08] Sigmoid-Allee 补充量应始终非负 FAILED'

# ---- TC09: recruitment_derivative bh 模式解析验证 ----
S_test = 3.0
dR = recruitment_derivative(S_test, 2.0, 0.3, 1.5, 8.0, model_type='bh')
expected_bh = 2.0 / ((1.0 + 0.3 * S_test) ** 2)
assert abs(dR - expected_bh) < 1e-10, '[TC09] recruitment_derivative bh 模式解析验证 FAILED'

# ---- TC10: Vandermonde 插值在节点处精确回代 ----
nodes = np.array([0.1, 0.25, 0.4, 0.6, 0.85])
values = 10.0 + 20.0 * nodes - 15.0 * (nodes ** 2) + 5.0 * (nodes ** 3)
interp_at_nodes = vandermonde_interp_1d(nodes, values, nodes)
max_err = np.max(np.abs(interp_at_nodes - values))
assert max_err < 1e-10, '[TC10] Vandermonde 插值节点回代误差 FAILED'

# ---- TC11: pvand 求解 Vandermonde 系统验证 ----
n_v = 5
alpha_v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
c_true = np.array([5.0, 0.0, 5.0, 0.0, 0.0])
V = np.vander(alpha_v, N=n_v, increasing=True)
b_v = V.T @ c_true
x_sol = pvand(n_v, alpha_v, b_v)
residual = np.linalg.norm(V.T @ x_sol - b_v)
assert residual < 1e-10, '[TC11] pvand 求解 Vandermonde 系统残差 FAILED'

# ---- TC12: bidim_vandermonde_solve 输出尺寸与输入一致 ----
alpha_b = np.array([1.0, 2.0])
beta_b = np.array([3.0, 4.0])
b_b = np.array([1.0, 2.0, 3.0, 4.0])
x_b = bidim_vandermonde_solve(2, alpha_b, beta_b, b_b)
assert len(x_b) == 4, '[TC12] bidim_vandermonde_solve 输出长度应为 n*n FAILED'

# ---- TC13: sphere_stereograph_inverse 投影点 z 坐标为 1 ----
pts_plane = np.array([[1.0, 2.0, 1.0], [3.0, 4.0, 1.0]])
inv_pts = sphere_stereograph_inverse(pts_plane)
norm_inv = np.linalg.norm(inv_pts, axis=1)
assert np.allclose(norm_inv, 1.0, atol=1e-10), '[TC13] 逆投影点应位于单位球面 FAILED'

# ---- TC14: divergence_free_velocity 输出为有限值 ----
X_test = np.array([0.2, 0.5, 0.8])
Y_test = np.array([0.3, 0.6, 0.9])
U_test, V_test = divergence_free_velocity(3, X_test, Y_test, 1.0)
assert np.all(np.isfinite(U_test)) and np.all(np.isfinite(V_test)), '[TC14] 无散度速度场输出应全为有限值 FAILED'

# ---- TC15: integrate_cube_domain 常数函数精确积分 ----
cube_int = integrate_cube_domain(lambda pts: 2.0, degree=3, scale=1.0)
# 立方体 [-1,1]^3 体积为 8, 常数 2 积分应为 16
assert abs(cube_int - 16.0) < 1e-10, '[TC15] 立方体域常数函数积分应精确 FAILED'

# ---- TC16: integrate_pyramid_domain 常数函数非负 ----
pyr_int = integrate_pyramid_domain(lambda pts: 1.0, degree=3)
assert pyr_int > 0.0, '[TC16] 金字塔域常数正函数积分应大于 0 FAILED'

# ---- TC17: 二十面体顶点数为 12 且位于单位球面 ----
vertices = icosahedron_vertices()
norm_v = np.linalg.norm(vertices, axis=1)
assert len(vertices) == 12, '[TC17] 二十面体顶点数应为 12 FAILED'
assert np.allclose(norm_v, 1.0, atol=1e-10), '[TC17] 二十面体顶点应位于单位球面 FAILED'

# ---- TC18: 球面立体投影正反变换可逆 ----
proj = sphere_stereograph(vertices)
recovered = sphere_stereograph_inverse(proj)
recon_err = np.max(np.linalg.norm(vertices - recovered, axis=1))
assert recon_err < 1e-10, '[TC18] 球面立体投影正反变换可逆性 FAILED'

# ---- TC19: Schaefer-Gordon 稳态 E=0 时生物量等于承载力 ----
B0 = schaefer_gordon_steady_state(0.0, 0.4, 1000.0, 0.01)
assert abs(B0 - 1000.0) < 1e-10, '[TC19] Schaefer-Gordon E=0 时 B 应等于 K FAILED'

# ---- TC20: Schaefer-Gordon 过度捕捞时生物量为 0 ----
B_over = schaefer_gordon_steady_state(100.0, 0.4, 1000.0, 0.01)
assert abs(B_over) < 1e-10, '[TC20] Schaefer-Gordon 过度捕捞时 B 应为 0 FAILED'

# ---- TC21: 最短路径重构包含源点 ----
connectivity = np.array([
    [0.0, 2.5, 4.0, np.inf, 3.0],
    [2.5, 0.0, 1.5, 3.5, np.inf],
    [4.0, 1.5, 0.0, 2.0, 4.5],
    [np.inf, 3.5, 2.0, 0.0, 1.8],
    [3.0, np.inf, 4.5, 1.8, 0.0]
], dtype=float)
dist_path, predecessor = mpa_network_optimize(5, connectivity, source_patch=0)
path_to_0 = reconstruct_path(predecessor, 0)
assert path_to_0[0] == 0, '[TC21] 最短路径重构应包含源点 FAILED'

# ---- TC22: Allen-Cahn 能量泛函非负性 ----
x_ac = np.linspace(0.0, 10.0, 101)
u_ac = np.tanh((x_ac - 3.0) / (np.sqrt(2.0) * 0.5))
energy = energy_functional(u_ac, 10.0 / 100.0, 0.1, 0.5)
assert energy >= 0.0, '[TC22] Allen-Cahn 能量泛函应非负 FAILED'

# ---- TC23: fishery_forcing 输出尺寸与输入一致 ----
u_forcing = np.array([0.5, -0.3, 1.0, -1.0])
force = fishery_forcing(0.0, u_forcing, 25.0, 0.3, 0.01, 1000.0)
assert len(force) == len(u_forcing), '[TC23] fishery_forcing 输出尺寸应与输入一致 FAILED'

# ---- TC24: 一维蒙特卡洛积分常数函数精确验证 ----
np.random.seed(42)
mc_result = monte_carlo_integral_1d(lambda x: 3.0, 0.0, 1.0, 100, method='mc')
assert abs(mc_result - 3.0) < 1e-10, '[TC24] MC 积分常数函数应精确 FAILED'

# ---- TC25: Hammersley 序列输出尺寸正确 ----
pts = hammersley_sequence(0, 4, m=3, n=1000)
assert pts.shape == (5, 3), '[TC25] Hammersley 序列输出尺寸应为 (5, 3) FAILED'

# ---- TC26: 线积分常数函数精确验证 ----
line_int = integrate_line_profile(lambda z: 5.0, 0.0, 10.0, order=3)
assert abs(line_int - 50.0) < 1e-10, '[TC26] 线积分常数函数应精确 FAILED'

# ---- TC27: 栖息地聚类输出标签在合理范围内 ----
np.random.seed(42)
n_stations = 40
env_data = np.random.randn(n_stations, 4)
zones, _, zone_stats = cluster_habitat_zones(n_stations, env_data, n_zones=3)
assert np.all((zones >= 0) & (zones < 3)), '[TC27] 聚类标签应在 [0, n_zones) 范围内 FAILED'
assert zone_stats['inertia'] >= 0.0, '[TC27] 聚类惯性应非负 FAILED'

print('\n全部 27 个测试通过!\n')
