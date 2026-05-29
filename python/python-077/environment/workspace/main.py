"""
main.py
基于多物理场耦合的大型海上风电场微观选址与尾流效应数值模拟系统

统一入口，零参数可运行。

科学领域：计算流体力学 — 风力机尾流与风电场布局
"""

import numpy as np
import sys
from typing import List, Tuple

# 导入各模块
from numerical_utils import (
    trig_interpolant, pwl_approx_1d, alnorm, alnorm_array,
    r8mat_fs, sor_solve, integrate_ode, langford_deriv,
    weibull_pdf, weibull_cdf
)
from wake_model import WakeModel, WakeFarm
from wind_field import WindField
from turbine import WindTurbine, TurbineFarm
from terrain_model import TerrainProfile, FEM2DMesh
from flow_solver import FlowSolver, TurbulenceCascade
from layout_optimizer import LayoutOptimizer
from cable_routing import CableRouter
from uncertainty_quantification import UncertaintyQuantification


def print_section(title: str):
    """打印章节分隔符。"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_trig_interpolation():
    """
    演示：风向周期性分布的三角插值重构。
    融合项目 596_interp_trig。
    """
    print_section("[1] 风向周期性分布的三角插值重构")
    # 模拟风向数据（0-360度，周期性）
    n = 12
    xd = np.linspace(0, 360, n, endpoint=False)
    # 模拟风向频率分布（von Mises-like）
    mu = 270.0
    yd = np.exp(2.0 * np.cos(np.radians(xd - mu)))
    # 插值到更密网格
    xi = np.linspace(0, 360, 100)
    yi = trig_interpolant(xd, yd, xi)
    print(f"  原始节点数: {n}")
    print(f"  插值点数: {len(xi)}")
    print(f"  风向分布峰值角度: {xi[np.argmax(yi)]:.2f}°")
    print(f"  三角插值均方根值: {np.sqrt(np.mean(yi**2)):.4f}")


def demo_pwl_power_curve():
    """
    演示：风速-功率曲线的分段线性逼近。
    融合项目 925_pwl_approx_1d。
    """
    print_section("[2] 风速-功率曲线分段线性逼近")
    turbine = WindTurbine(D=126.0, rated_power=5.0)
    print(f"  转子直径: {turbine.D} m")
    print(f"  扫掠面积: {turbine.swept_area:.2f} m²")
    print(f"  额定功率: {turbine.rated_power} MW")
    print(f"  切入/额定/切出风速: {turbine.u_cut_in}/{turbine.u_rated}/{turbine.u_cut_out} m/s")
    # 测试功率曲线
    test_speeds = [3.0, 5.0, 8.0, 10.0, 12.0, 15.0, 25.0]
    for u in test_speeds:
        p = turbine.power(u)
        ct = turbine.thrust_coefficient(u)
        a = turbine.axial_induction_factor(u)
        print(f"  u={u:5.1f} m/s  →  P={p:6.3f} MW,  Ct={ct:.3f},  a={a:.3f}")


def demo_weibull_and_normal():
    """
    演示：Weibull分布与正态近似检验。
    融合项目 032_asa066。
    """
    print_section("[3] Weibull 风资源分布与正态近似检验")
    wf = WindField(A=10.0, k=2.0, mu_theta=270.0, kappa=2.0, turbulence_intensity=0.12)
    print(f"  Weibull 尺度参数 A: {wf.A} m/s")
    print(f"  Weibull 形状参数 k: {wf.k}")
    print(f"  平均风速: {wf.mean_wind_speed():.3f} m/s")
    print(f"  风速标准差: {wf.std_wind_speed():.3f} m/s")

    u_bins, weibull_cdf_vals, max_diff = wf.weibull_to_normal_test(n_bins=20)
    print(f"  Weibull vs Normal CDF 最大差异 (KS-like): {max_diff:.6f}")

    # 年风能密度
    e_density = wf.annual_energy_density()
    print(f"  年风能密度: {e_density:.2f} kWh/m²")

    # 测量噪声
    u_clean = np.array([8.0, 10.0, 12.0, 9.0, 11.0])
    u_noisy = wf.add_measurement_noise(u_clean, noise_level=0.05, noise_type='gaussian', seed=42)
    print(f"  原始风速: {u_clean}")
    print(f"  加噪风速: {np.round(u_noisy, 2)}")


def demo_wake_model():
    """
    演示：Jensen尾流模型与叠加。
    融合项目 468_geometry、1147_square_integrals。
    """
    print_section("[4] Jensen 尾流模型与叠加")
    # TODO: Hole 3-A — WakeModel 初始化已由推力系数 Ct 改为轴向诱导因子 a
    # 要求：使用 a = 0.293（对应 Ct ≈ 0.8 的额定工况）初始化 WakeModel
    raise NotImplementedError("Hole 3-A: 待修复")

    # 单尾流亏损
    x_positions = [2 * wm.D, 5 * wm.D, 10 * wm.D, 20 * wm.D]
    for x in x_positions:
        delta = wm.wake_deficit(x)
        Rw = wm.wake_radius(x)
        print(f"  x={x:6.0f} m  →  δ={delta:.4f},  R_w={Rw:.1f} m")

    # 扫掠面积平均亏损
    wfarm = WakeFarm(wm)
    turbines = [(0.0, 0.0), (500.0, 0.0), (1000.0, 0.0)]
    u0 = 10.0
    wind_dir = 0.0  # 正东方向吹来

    total_power = 0.0
    for i in range(len(turbines)):
        u_eff = wfarm.compute_effective_velocity(turbines, i, u0, wind_dir)
        print(f"  风机 {i+1}: u_eff = {u_eff:.3f} m/s (环境风速 {u0} m/s)")

    # 双尾流叠加测试
    d1, d2 = 0.15, 0.20
    combined = WakeModel.combine_deficits([d1, d2])
    print(f"  双尾流叠加测试: δ1={d1}, δ2={d2}  →  δ_total={combined:.4f}")


def demo_linear_solver():
    """
    演示：压力泊松方程求解（SOR + 直接求解）。
    融合项目 1098_solve、1099_sor。
    """
    print_section("[5] 压力泊松方程数值求解")
    solver = FlowSolver(nx=20, ny=20, Lx=2000.0, Ly=2000.0)
    solver.set_inflow(10.0)

    # SOR 求解
    p_sor = solver.solve_pressure_poisson_sor(omega=1.5, tol=1e-5, max_iter=2000)
    p_min, p_max = np.min(p_sor), np.max(p_sor)
    print(f"  SOR 求解网格: {solver.nx}×{solver.ny}")
    print(f"  压力场范围: [{p_min:.4f}, {p_max:.4f}] Pa")

    # 直接求解
    solver2 = FlowSolver(nx=10, ny=10, Lx=2000.0, Ly=2000.0)
    p_direct = solver2.solve_pressure_poisson_direct()
    p_min2, p_max2 = np.min(p_direct), np.max(p_direct)
    print(f"  直接求解网格: {solver2.nx}×{solver2.ny}")
    print(f"  压力场范围: [{p_min2:.4f}, {p_max2:.4f}] Pa")

    # 速度修正
    u_corr, v_corr = solver.velocity_correction()
    div = solver.compute_divergence()
    max_div = np.max(np.abs(div))
    print(f"  修正后最大速度散度: {max_div:.6e}")

    # 涡量
    vort = solver.compute_vorticity()
    print(f"  涡量范围: [{np.min(vort):.4f}, {np.max(vort):.4f}] 1/s")

    # TKE
    tke = solver.turbulence_kinetic_energy()
    print(f"  湍动能 TKE 均值: {np.mean(tke):.6f} m²/s²")


def demo_terrain_and_mesh():
    """
    演示：地形轮廓与FEM网格。
    融合项目 502_hand_data、406_fem2d_mesh_display、468_geometry。
    """
    print_section("[6] 海底地形轮廓与 FEM 网格")
    # 地形轮廓
    terrain = TerrainProfile()
    # 模拟海底地形（高斯隆起）
    x_terrain = np.linspace(0, 5000, 50)
    z_terrain = -30.0 + 10.0 * np.exp(-((x_terrain - 2500) / 800)**2)
    for x, z in zip(x_terrain, z_terrain):
        terrain.add_point(x, z)
    terrain.close_profile()

    area = terrain.polygon_area()
    print(f"  地形轮廓点数: {len(terrain.points)}")
    print(f"  地形多边形有向面积: {area:.2f} m²")

    # 地形高程与坡度
    x_test = [1000.0, 2500.0, 4000.0]
    for x in x_test:
        z = terrain.elevation_at(x)
        slope = terrain.slope_at(x)
        print(f"  x={x:.0f} m  →  z={z:.2f} m,  slope={slope:.6f}")

    # FEM 网格
    mesh = FEM2DMesh(xmin=0.0, xmax=5000.0, ymin=0.0, ymax=5000.0, nx=10, ny=10)
    print(f"  FEM 网格节点数: {mesh.n_nodes()}")
    print(f"  FEM 网格单元数: {mesh.n_elements()}")
    print(f"  单个单元面积示例: {mesh.element_area(0):.2f} m²")
    print(f"  计算域总面积: {mesh.total_domain_area():.2f} m²")

    # 点定位
    px, py = 2500.0, 2500.0
    elem_idx = mesh.find_element_containing(px, py)
    print(f"  点 ({px}, {py}) 所在单元: {elem_idx}")

    # 节点邻域
    neighbors = mesh.node_neighborhood(55, radius=800.0)
    print(f"  节点 55 在 800m 半径内邻居数: {len(neighbors)}")


def demo_cvt_layout():
    """
    演示：基于 CVT 的风机布局优化。
    融合项目 246_cvt_1d_sampling、793_nearest_neighbor、1180_subset_sum_brute。
    """
    print_section("[7] 基于 CVT 的风机布局优化")
    n_turbines = 8
    optimizer = LayoutOptimizer(
        n_turbines=n_turbines,
        domain=(0.0, 3000.0, 0.0, 3000.0),
        min_spacing=500.0,
        rated_power=5.0,
        max_grid_capacity=50.0
    )
    optimizer.initialize_grid()
    print(f"  初始布局最小间距: {optimizer.min_spacing():.1f} m")

    # 修复间距
    ok = optimizer.repair_spacing()
    print(f"  间距约束修复: {'成功' if ok else '失败'}")
    print(f"  修复后最小间距: {optimizer.min_spacing():.1f} m")

    # 最近邻分析
    nn_pairs = optimizer._nearest_neighbor_indices()
    print(f"  最近邻对示例 (风机 1 → 最近邻): 风机 {nn_pairs[0][1]+1}")

    # 容量组合优化
    capacities = [5.0] * n_turbines
    target_cap = 30.0
    found, chosen = optimizer.capacity_subset_optimization(capacities, target_cap)
    print(f"  容量组合优化: 目标 {target_cap} MW")
    print(f"  找到可行解: {'是' if found else '否'}")
    if found:
        total = sum(capacities[i] for i in chosen)
        print(f"  选中风机: {[i+1 for i in chosen]}, 总容量: {total:.1f} MW")

    # CVT 优化（简化目标函数）
    def simple_objective(pos):
        # 鼓励分散 + 靠近中心（简化示例）
        cx, cy = 1500.0, 1500.0
        spread = np.std(pos[:, 0]) + np.std(pos[:, 1])
        center_bias = -np.mean((pos[:, 0] - cx)**2 + (pos[:, 1] - cy)**2) * 0.0001
        return spread + center_bias

    optimized = optimizer.cvt_optimize(simple_objective, n_samples=2000,
                                       n_iterations=10, step_size=50.0)
    print(f"  CVT 优化后布局:")
    for i, (x, y) in enumerate(optimized):
        print(f"    风机 {i+1}: ({x:.1f}, {y:.1f}) m")
    print(f"  优化后最小间距: {optimizer.min_spacing():.1f} m")


def demo_cable_routing():
    """
    演示：电缆路由优化。
    融合项目 076_bellman_ford。
    """
    print_section("[8] 海上风电场电缆路由优化 (Bellman-Ford)")
    substation = (0.0, 0.0)
    turbines = [
        (800.0, 200.0), (1500.0, 300.0), (2200.0, 100.0),
        (1000.0, 800.0), (1800.0, 900.0), (2500.0, 700.0)
    ]
    router = CableRouter(substation, turbines, cable_cost_per_meter=500.0,
                         max_cable_length=3000.0)

    # Bellman-Ford 最短路径
    paths = router.find_paths(source=0)
    print(f"  升压站坐标: {substation}")
    print(f"  风机数量: {len(turbines)}")
    print("  Bellman-Ford 最短电缆路径:")
    for target, path in paths.items():
        length = sum(router._distance(path[k], path[k+1]) for k in range(len(path)-1))
        print(f"    风机 {target}: 路径 {path}, 长度 {length:.1f} m")

    total_cost = router.compute_total_cable_cost(paths)
    total_len, avg_len, max_len = router.compute_cable_length_stats(paths)
    print(f"  总电缆成本: {total_cost:,.0f} 元")
    print(f"  总电缆长度: {total_len:.1f} m")
    print(f"  平均路径长度: {avg_len:.1f} m")
    print(f"  最大路径长度: {max_len:.1f} m")

    # MST 优化
    mst_paths, mst_cost = router.optimize_routing_mst()
    print(f"  MST 优化后总成本: {mst_cost:,.0f} 元")


def demo_turbulence_ode():
    """
    演示：湍流能量级联 ODE 模型。
    融合项目 645_langford_ode。
    """
    print_section("[9] 湍流能量级联 ODE 模型")
    cascade = TurbulenceCascade(beta_star=0.09, beta=0.075, gamma=0.553)
    y0 = np.array([0.5, 1.0])  # [K0, ω0]
    P_K = 0.5  # 湍流产生率
    t, y = cascade.integrate(y0, (0.0, 20.0), P_K, n_steps=5000)

    print(f"  初始状态: K0={y0[0]:.3f} m²/s², ω0={y0[1]:.3f} 1/s")
    print(f"  终态: K={y[-1,0]:.4f} m²/s², ω={y[-1,1]:.4f} 1/s")
    print(f"  时间步数: {len(t)}")

    # Langford ODE 演示（混沌动力系统）
    t_lang, y_lang = integrate_ode(langford_deriv, np.array([0.1, 0.1, 0.1]),
                                   (0.0, 30.0), n_steps=3000)
    print(f"  Langford ODE 终态: x={y_lang[-1,0]:.4f}, y={y_lang[-1,1]:.4f}, z={y_lang[-1,2]:.4f}")


def demo_uncertainty():
    """
    演示：不确定性量化。
    融合项目 581_image_noise、1147_square_integrals、032_asa066。
    """
    print_section("[10] 风电场性能不确定性量化")
    uq = UncertaintyQuantification(n_mc_samples=500, seed=42)

    # 功率不确定性传播
    turbine = WindTurbine()
    u_nominal = 10.0
    sigma_u = 1.0
    mean_p, std_p = uq.propagate_speed_uncertainty(u_nominal, sigma_u, turbine.power)
    print(f"  名义风速: {u_nominal} m/s, 风速标准差: {sigma_u} m/s")
    print(f"  功率均值: {mean_p:.3f} MW, 功率标准差: {std_p:.3f} MW")

    # 置信区间
    ci_mean, ci_lo, ci_hi = uq.confidence_interval(
        np.random.normal(mean_p, std_p, 500), confidence=0.95
    )
    print(f"  功率 95% 置信区间: [{ci_lo:.3f}, {ci_hi:.3f}] MW")

    # KS 正态性检验
    samples = np.random.normal(10.0, 2.0, 200)
    d_stat, reject = uq.ks_test_normality(samples)
    print(f"  KS 正态性检验: D={d_stat:.4f}, 拒绝正态性: {'是' if reject else '否'}")

    # 积分不确定性
    wm = WakeModel()
    deficit_func = lambda x: wm.wake_deficit(x)
    mean_int, std_int = uq.integral_deficit_uncertainty(
        deficit_func, (wm.D * 2, wm.D * 10), sigma_x=20.0, n_points=50
    )
    print(f"  尾流亏损积分不确定性: 均值={mean_int:.4f}, 标准差={std_int:.6f}")

    # 灵敏度分析
    def dummy_aep_func(params):
        # params = [k_wake, Ct, D]
        k, ct, d = params
        # TODO: Hole 3-B — WakeModel 初始化参数已由 Ct 改为 a
        # 要求：根据当前的 ct 值计算出对应的轴向诱导因子 a，再传入 WakeModel
        raise NotImplementedError("Hole 3-B: 待修复")
        return 5000.0 * (1.0 - wm2.wake_deficit(d * 5))

    base_params = [0.05, 0.8, 126.0]
    base_aep = dummy_aep_func(base_params)
    sens = uq.sensitivity_analysis(
        base_aep,
        ['k_wake', 'Ct', 'D'],
        base_params,
        [0.01, 0.05, 5.0],
        dummy_aep_func,
        delta=0.01
    )
    print(f"  灵敏度分析 (基准 AEP={base_aep:.2f} MWh):")
    for name, vals in sens.items():
        print(f"    {name}: 灵敏度指数={vals['sensitivity_index']:.4f}, "
              f"不确定性贡献={vals['uncertainty_contribution']:.2f}")


def demo_integrated_simulation():
    """
    综合演示：完整风电场微观选址尾流耦合模拟。
    """
    print_section("[11] 综合风电场微观选址尾流耦合模拟")

    # 1. 风资源
    wf = WindField(A=10.0, k=2.2, mu_theta=270.0, kappa=1.5, turbulence_intensity=0.10)
    print(f"  风场: A={wf.A} m/s, k={wf.k}, 平均风速={wf.mean_wind_speed():.2f} m/s")

    # 2. 风机
    turbine = WindTurbine(D=126.0, hub_height=90.0, rated_power=5.0,
                          u_cut_in=3.0, u_rated=12.0, u_cut_out=25.0,
                          cp_max=0.45, ct_at_rated=0.8)
    print(f"  风机: D={turbine.D} m, P_rated={turbine.rated_power} MW")

    # 3. 尾流模型
    # TODO: Hole 3-C — WakeModel 初始化参数已由 Ct 改为 a
    # 要求：调用 turbine.axial_induction_factor(turbine.u_rated) 获取额定工况下的 a，
    #       然后传入 WakeModel 进行初始化
    raise NotImplementedError("Hole 3-C: 待修复")
    wake_farm = WakeFarm(wm)

    # 4. 布局
    layout = LayoutOptimizer(n_turbines=6,
                             domain=(0.0, 4000.0, 0.0, 4000.0),
                             min_spacing=500.0,
                             rated_power=5.0,
                             max_grid_capacity=50.0)
    layout.initialize_grid()
    layout.repair_spacing()
    turbines_pos = [(float(p[0]), float(p[1])) for p in layout.positions]
    print(f"  布局风机数: {len(turbines_pos)}")
    print(f"  最小间距: {layout.min_spacing():.1f} m")

    # 5. 多工况功率计算
    wind_speeds = [6.0, 8.0, 10.0, 12.0, 14.0]
    wind_dirs = [270.0, 260.0, 280.0, 270.0, 270.0]
    total_aep = 0.0
    hours_per_case = 8760.0 / len(wind_speeds)

    for u0, theta in zip(wind_speeds, wind_dirs):
        farm_power, powers = wake_farm.compute_farm_power(
            turbines_pos, u0, theta, turbine.power
        )
        case_aep = farm_power * hours_per_case
        total_aep += case_aep
        print(f"  工况 u0={u0:4.1f} m/s, θ={theta:5.1f}°  →  "
              f"场功率={farm_power:5.2f} MW, 案例AEP={case_aep:8.1f} MWh")

    print(f"  估算年总发电量 AEP: {total_aep:,.1f} MWh")

    # 6. 电缆路由
    substation = (2000.0, -500.0)
    router = CableRouter(substation, turbines_pos, cable_cost_per_meter=500.0)
    paths = router.find_paths()
    cable_cost = router.compute_total_cable_cost(paths)
    print(f"  电缆系统总成本: {cable_cost:,.0f} 元")

    # 7. 容量因子
    cf = turbine.capacity_factor(wf)
    print(f"  单风机容量因子: {cf:.3f}")

    # 8. 流场求解（简化）
    solver = FlowSolver(nx=15, ny=15, Lx=4000.0, Ly=4000.0)
    solver.set_inflow(wf.mean_wind_speed())
    p_field = solver.solve_pressure_poisson_direct()
    print(f"  流场压力范围: [{np.min(p_field):.4f}, {np.max(p_field):.4f}] Pa")


def main():
    """
    统一入口函数。零参数可运行。
    """
    print("\n" + "#" * 70)
    print("#  基于多物理场耦合的大型海上风电场微观选址与尾流效应数值模拟系统")
    print("#  科学领域: 计算流体力学 — 风力机尾流与风电场布局")
    print("#" * 70)

    demo_trig_interpolation()
    demo_pwl_power_curve()
    demo_weibull_and_normal()
    demo_wake_model()
    demo_linear_solver()
    demo_terrain_and_mesh()
    demo_cvt_layout()
    demo_cable_routing()
    demo_turbulence_ode()
    demo_uncertainty()
    demo_integrated_simulation()

    print("\n" + "#" * 70)
    print("#  所有演示模块运行完毕，无错误。")
    print("#" * 70 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
