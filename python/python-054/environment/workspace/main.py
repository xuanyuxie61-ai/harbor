
import numpy as np
import sys




from carbonate_chemistry import (
    solve_carbonate_system,
    equilibrium_constants,
    air_sea_co2_flux,
    batch_solve_carbonate,
)
from ocean_mesh import (
    generate_ocean_rectangle_mesh,
    generate_ocean_semicircle_mesh,
    mesh_bandwidth,
    compute_adjacency,
    compute_element_areas,
    compute_boundary_edges,
    sample_q4_mesh,
)
from vertical_profiles import (
    build_hermite_spline,
    evaluate_hermite_spline,
    integrate_hermite_spline,
    estimate_derivatives_central,
    compute_brunt_vaisala_frequency,
    mixed_layer_depth,
)
from carbon_transport import (
    euler_forward,
    autocatalytic_carbonate_deriv,
    vertical_carbon_transport_model,
    box_carbon_cycle_model,
    compute_anthropogenic_carbon_inventory,
)
from quadrature_engine import (
    cube_gauss_rule,
    integrate_over_cube,
    integrate_dic_inventory_cube,
    keast_rule,
    integrate_over_tetrahedron,
    tetrahedron_volume,
    hypersphere_uniform_sample,
    parameter_sensitivity_on_sphere,
)
from sensor_network import (
    cvt_on_disk,
    cvt_on_rectangle,
    deploy_sensor_network,
    plan_ocean_sampling_route,
)
from sparse_interpolation import (
    shepard_interp_nd,
    shepard_interp_3d_ocean,
    cross_validate_shepard,
    generate_sparse_ocean_observations,
)
from front_analysis import (
    news_gradient,
    sobel_gradient,
    detect_fronts_multi_field,
    front_statistics,
)
from sequestration_optimizer import (
    golden_section_search,
    optimize_sequestration_depth,
    multi_scenario_optimization,
)
from uncertainty_analysis import (
    hypersphere01_area,
    hypersphere01_sample,
    propagate_uncertainty,
    full_sensitivity_analysis,
    carbon_cycle_uncertainty_analysis,
)
from region_connectivity import (
    OceanRegionGraph,
    create_ocean_basin_graph,
    carbon_transport_path_analysis,
)


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_carbonate_chemistry():
    print_section("模块 1: 碳酸盐化学系统与 Muller 根查找")
    

    DIC = 2.0e-3
    TA = 2.3e-3
    T = 15.0
    S = 35.0
    
    print(f"输入: DIC={DIC*1e6:.1f} μmol/kg, TA={TA*1e6:.1f} μmol/kg, T={T}°C, S={S}")
    
    result = solve_carbonate_system(DIC, TA, T, S)
    
    print(f"  pH = {result['pH']:.4f}")
    print(f"  [H⁺] = {result['H']:.3e} mol/kg")
    print(f"  pCO₂ = {result['pCO2']:.1f} μatm")
    print(f"  [CO₃²⁻] = {result['CO3']*1e6:.1f} μmol/kg")
    print(f"  Ω_calcite = {result['Omega_calcite']:.3f}")
    print(f"  Ω_aragonite = {result['Omega_aragonite']:.3f}")
    print(f"  Muller 迭代次数: {result['iters']}")
    





    raise NotImplementedError("HOLE 3: 海-气通量调用与结果输出待补全")
    

    depths = np.array([0, 50, 100, 200, 500, 1000])
    DIC_profile = 2000.0 + 300.0 * (1.0 - np.exp(-depths / 800.0))
    TA_profile = DIC_profile + 100.0
    T_profile = 20.0 * np.exp(-depths / 200.0) + 2.0
    S_profile = np.full_like(depths, 35.0)
    
    batch_results = batch_solve_carbonate(DIC_profile, TA_profile, T_profile, S_profile, units='umolkg')
    print(f"\n  垂直剖面批量求解 ({len(depths)} 层):")
    for i, res in enumerate(batch_results):
        print(f"    z={depths[i]:4d}m: pH={res['pH']:.3f}, "
              f"Ω_arag={res['Omega_aragonite']:.3f}")


def demo_ocean_mesh():
    print_section("模块 2: 海洋 Q4 网格生成与带宽分析")
    
    Lx, Ly = 500.0, 500.0
    nx, ny = 20, 20
    
    node_xy, element_node, nx_out, ny_out = generate_ocean_rectangle_mesh(
        Lx, Ly, nx, ny)
    
    print(f"矩形网格: {nx}×{ny} 单元, {node_xy.shape[0]} 节点")
    
    bw = mesh_bandwidth(element_node, node_xy.shape[0])
    print(f"  稀疏矩阵带宽: ML={bw['ml']}, MU={bw['mu']}, M={bw['m']}")
    
    areas, total_area = compute_element_areas(node_xy, element_node)
    print(f"  总面积: {total_area:.1f} km² (期望: {Lx*Ly:.1f} km²)")
    
    boundary_edges, n_bound = compute_boundary_edges(element_node)
    print(f"  边界边数: {n_bound}")
    

    node_xy_semi, element_node_semi = generate_ocean_semicircle_mesh(
        R=200.0, nx=15, ny=20)
    print(f"\n半圆网格: {element_node_semi.shape[0]} 单元, {node_xy_semi.shape[0]} 节点")
    areas_semi, total_semi = compute_element_areas(node_xy_semi, element_node_semi)
    print(f"  半圆面积: {total_semi:.1f} km² (期望: {0.5*np.pi*200**2:.1f} km²)")
    

    samples = sample_q4_mesh(node_xy, element_node, n_samples=100)
    print(f"  面积加权采样 {len(samples)} 个点")


def demo_vertical_profiles():
    print_section("模块 3: Hermite 三次样条垂直剖面")
    
    z_nodes = np.array([0, 10, 20, 50, 100, 200, 500, 1000, 2000, 4000])
    T_nodes = np.array([25.0, 24.5, 23.0, 20.0, 15.0, 10.0, 6.0, 4.0, 3.0, 2.5])
    S_nodes = np.array([35.5, 35.5, 35.6, 35.7, 35.8, 34.9, 34.7, 34.6, 34.6, 34.6])
    
    dT = estimate_derivatives_central(z_nodes, T_nodes)
    dS = estimate_derivatives_central(z_nodes, S_nodes)
    
    T_spline = build_hermite_spline(z_nodes, T_nodes, dT)
    
    z_query = np.linspace(0, 4000, 41)
    T_interp, dTdz, _, _ = evaluate_hermite_spline(T_spline, z_query)
    
    print(f"构建温度样条: {len(z_nodes)} 节点, {len(z_query)} 查询点")
    print(f"  表层 T={T_interp[0]:.2f}°C, 深层 T={T_interp[-1]:.2f}°C")
    print(f"  表层 dT/dz={dTdz[0]:.4f} °C/m")
    

    N2, z_mid = compute_brunt_vaisala_frequency(z_nodes, T_nodes, S_nodes, lat=30.0)
    print(f"\nBrunt-Väisälä 频率 (N²):")
    for i in range(len(z_mid)):
        N = np.sqrt(max(0, N2[i]))
        if N > 1e-6:
            period_str = f"{2*np.pi/N/60:.1f} min"
        else:
            period_str = "inf min"
        print(f"  z={z_mid[i]:5.0f}m: N²={N2[i]:.2e} s⁻², N={N:.4f} s⁻¹ "
              f"(周期={period_str})")
    
    mld = mixed_layer_depth(z_nodes, T_nodes, threshold=0.5)
    print(f"\n  混合层深度 (MLD, ΔT=0.5°C): {mld:.1f} m")
    

    T_int = integrate_hermite_spline(T_spline, 0, 4000)
    print(f"  温度积分 ∫T(z)dz = {T_int:.1f} °C·m")


def demo_carbon_transport():
    print_section("模块 4: 碳输送与反应动力学")
    

    print("自催化碳酸盐动力学 (Gray-Scott 型):")
    y0 = np.array([500.0, 0.0, 0.0, 0.0])
    t, y = euler_forward(
        lambda t, s: autocatalytic_carbonate_deriv(t, s, alpha=0.002, beta=0.08, gamma=0.5),
        (0, 1000), y0, 1000
    )
    print(f"  初始: [CO₂*]={y0[0]:.1f}, [HCO₃⁻]={y0[1]:.1f}, "
          f"[CO₃²⁻]={y0[2]:.1f}, [CaCO₃]={y0[3]:.1f}")
    print(f"  t=1000天后: [CO₂*]={y[-1,0]:.1f}, [HCO₃⁻]={y[-1,1]:.1f}, "
          f"[CO₃²⁻]={y[-1,2]:.1f}, [CaCO₃]={y[-1,3]:.1f}")
    

    print("\n三箱碳循环模型 (大气-表层-深层):")
    y0_box = np.array([750.0, 800.0, 37100.0])
    t_box, y_box = box_carbon_cycle_model(
        (0, 200), y0_box, 200,
        k12=0.1, k21=0.05, k23=0.02, k32=0.01,
        F_anthro=8.0, buffer_factor=10.0
    )
    print(f"  初始: 大气={y0_box[0]:.1f}, 表层={y0_box[1]:.1f}, "
          f"深层={y0_box[2]:.1f} Pg C")
    print(f"  t=200年后: 大气={y_box[-1,0]:.1f}, 表层={y_box[-1,1]:.1f}, "
          f"深层={y_box[-1,2]:.1f} Pg C")
    print(f"  大气碳增量: {y_box[-1,0]-y0_box[0]:.1f} Pg C")
    

    print("\n一维垂向碳输送模型 (30天模拟):")
    z_grid = -np.linspace(0, 500, 11)
    DIC_init = 2000.0 + 200.0 * (1.0 - np.exp(np.abs(z_grid) / 800.0))
    T_prof = 20.0 * np.exp(np.abs(z_grid) / 200.0) + 2.0
    S_prof = np.full_like(z_grid, 35.0)
    
    DIC_hist, t_hist = vertical_carbon_transport_model(
        z_grid, DIC_init, T_prof, S_prof,
        dt_days=1.0, n_days=30, w=0.0, Kz=1e-4,
        mu_max=0.5, pCO2_atm=410.0
    )
    print(f"  初始表层 DIC: {DIC_hist[0,0]:.1f} μmol/kg")
    print(f"  30天后表层 DIC: {DIC_hist[-1,0]:.1f} μmol/kg")
    print(f"  变化: {DIC_hist[-1,0]-DIC_hist[0,0]:+.1f} μmol/kg")
    

    DIC_preindustrial = np.full_like(DIC_init, 1950.0)
    inventory = compute_anthropogenic_carbon_inventory(
        DIC_preindustrial, DIC_hist[-1], 1025.0, abs(z_grid[1]-z_grid[0])
    )
    print(f"  人为碳库存: {inventory:.2f} mol C/m²")


def demo_quadrature():
    print_section("模块 5: 三维碳库存数值积分")
    

    def test_func(x, y, z):
        return x * y * z + 1.0
    
    a = [0, 0, 0]
    b = [1, 1, 1]
    exact = 1.125
    approx = integrate_over_cube(test_func, a, b, order_1d=(3, 3, 3))
    print(f"立方体测试积分: 精确={exact:.6f}, 数值={approx:.6f}, 误差={abs(approx-exact):.2e}")
    

    def DIC_func(x, y, z):

        return 2.0e-3 + 3.0e-4 * (1.0 - np.exp(-z / 800.0)) + 1.0e-5 * np.sin(x / 100.0)
    
    def rho_func(x, y, z):
        return 1025.0
    
    inventory = integrate_dic_inventory_cube(
        DIC_func, [0, 0, 0], [100e3, 100e3, 4000.0], rho_func, order_1d=(3, 3, 3)
    )
    print(f"三维海域碳库存: {inventory:.3e} mol C")
    

    v = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    vol = tetrahedron_volume(v)
    print(f"\n参考四面体体积: {vol:.6f} (期望: 1/6={1/6:.6f})")
    
    def f_tet(x, y, z):
        return 1.0
    
    int_tet = integrate_over_tetrahedron(f_tet, v, rule_index=3)
    print(f"四面体常数积分: {int_tet:.6f} (期望: {1/6:.6f})")
    

    print(f"\n超球面采样测试 (S² 表面积):")
    for m in [2, 3, 4, 5]:
        area = hypersphere01_area(m)
        print(f"  S^{m-1} 表面积 = {area:.6f}")
    
    samples = hypersphere_uniform_sample(3, 100, seed=42)
    norms = np.linalg.norm(samples, axis=0)
    print(f"  100 个 S² 采样点范数均值: {np.mean(norms):.6f} (期望=1.0)")


def demo_sensor_network():
    print_section("模块 6: CVT 传感器布点与 TSP 采样路径")
    

    generators, types = cvt_on_disk(
        n_generators=15, r=200.0, n_samples=8000,
        n_iterations=40, density_power=0.5
    )
    interior = generators[types == 0]
    print(f"圆盘域 CVT: 15 个内部传感器 + {np.sum(types==1)} 个边界传感器")
    print(f"  内部传感器覆盖范围: 半径 200 km")
    

    if len(interior) >= 3:
        route = plan_ocean_sampling_route(interior, seed=42)
        print(f"  TSP 最优路径长度: {route['total_distance']:.1f} km")
        print(f"  路径经过 {len(route['path'])} 个站点")
    

    generators_rect, types_rect = cvt_on_rectangle(
        n_generators=12, Lx=300.0, Ly=200.0,
        n_samples=8000, n_iterations=40
    )
    interior_rect = generators_rect[types_rect == 0]
    print(f"\n矩形域 CVT: 12 个内部传感器")
    if len(interior_rect) >= 3:
        route_rect = plan_ocean_sampling_route(interior_rect, seed=42)
        print(f"  TSP 最优路径长度: {route_rect['total_distance']:.1f} km")


def demo_sparse_interpolation():
    print_section("模块 7: 稀疏海洋观测 Shepard 插值")
    

    obs = generate_sparse_ocean_observations(n_points=40, seed=42)
    print(f"生成 {len(obs['DIC'])} 个稀疏观测点")
    print(f"  DIC 范围: [{obs['DIC'].min():.1f}, {obs['DIC'].max():.1f}] μmol/kg")
    

    query_lons = np.linspace(obs['lons'].min(), obs['lons'].max(), 10)
    query_lats = np.linspace(obs['lats'].min(), obs['lats'].max(), 10)
    query_depths = np.full(100, 500.0)
    q_lons, q_lats = np.meshgrid(query_lons, query_lats)
    q_lons = q_lons.flatten()
    q_lats = q_lats.flatten()
    

    DIC_interp = shepard_interp_3d_ocean(
        obs['lons'], obs['lats'], obs['depths'], obs['DIC'],
        q_lons, q_lats, query_depths, p=2.0,
        lon_scale=10.0, lat_scale=10.0, depth_scale=1000.0
    )
    print(f"  插值到 {len(DIC_interp)} 个网格点")
    print(f"  插值 DIC 范围: [{DIC_interp.min():.1f}, {DIC_interp.max():.1f}] μmol/kg")
    

    data_coords = np.zeros((3, len(obs['DIC'])))
    data_coords[0, :] = obs['lons'] / 10.0
    data_coords[1, :] = obs['lats'] / 10.0
    data_coords[2, :] = obs['depths'] / 1000.0
    
    cv_result = cross_validate_shepard(data_coords, obs['DIC'],
                                        p_values=[1.0, 2.0, 3.0])
    print(f"  交叉验证最优 p = {cv_result['best_p']}")
    for p, rmse in cv_result['rmse_by_p'].items():
        print(f"    p={p}: RMSE={rmse:.2f}")


def demo_front_detection():
    print_section("模块 8: 海洋锋面 NEWS 梯度检测")
    

    ny, nx = 40, 40
    x = np.linspace(0, 500, nx)
    y = np.linspace(0, 500, ny)
    X, Y = np.meshgrid(x, y)
    

    T_field = 20.0 - 8.0 * (1.0 + np.tanh((Y - 250.0) / 30.0)) / 2.0
    T_field += np.random.normal(0, 0.3, (ny, nx))
    

    S_field = 35.0 + 1.5 * (1.0 + np.tanh((Y - 250.0) / 40.0)) / 2.0
    

    DIC_field = 2000.0 + 100.0 * (1.0 + np.tanh((Y - 250.0) / 35.0)) / 2.0
    
    print(f"生成 {ny}×{nx} 海洋场，模拟 y=250km 处锋面")
    

    grad_T = news_gradient(T_field)
    print(f"  T 场 NEWS 梯度均值: {grad_T.mean():.3f} °C/km")
    print(f"  T 场 NEWS 梯度最大: {grad_T.max():.3f} °C/km")
    

    fronts = detect_fronts_multi_field(
        {'T': T_field, 'S': S_field, 'DIC': DIC_field},
        weights={'T': 1.0, 'S': 0.5, 'DIC': 0.3},
        threshold_percentile=92
    )
    
    n_front_pixels = np.sum(fronts['front_mask'])
    print(f"  检测到锋面像素: {n_front_pixels} ({100*n_front_pixels/(ny*nx):.1f}%)")
    

    stats = front_statistics(fronts['front_mask'], T_field, dx=500/40, dy=500/40)
    print(f"  锋面长度: {stats['front_length_km']:.1f} km")
    print(f"  锋面平均梯度: {stats['front_mean_gradient']:.3f} °C/km")


def demo_sequestration():
    print_section("模块 9: 海洋碳封存深度优化 (黄金分割)")
    
    result = optimize_sequestration_depth(
        z_min=500, z_max=4000,
        K_z=1e-4, w=0.0, n_iter=80
    )
    
    print(f"最优封存深度: {result['optimal_depth_m']:.1f} m")
    print(f"  综合效率评分: {result['efficiency_score']:.4f}")
    print(f"  水合物稳定性: {result['hydrate_stability']:.4f}")
    print(f"  再暴露时间: {result['reexposure_time_years']:.1f} 年")
    print(f"  酸化影响指数: {result['acidification_impact']:.4f}")
    print(f"  优化迭代次数: {result['iterations']}")
    

    print(f"\n多情景封存对比:")
    scenarios = [
        {'name': '低扩散_静风', 'K_z': 5e-5, 'w': 0.0, 'injection_rate': 1.0},
        {'name': '高扩散_静风', 'K_z': 2e-4, 'w': 0.0, 'injection_rate': 1.0},
        {'name': '低扩散_上升流', 'K_z': 5e-5, 'w': 1e-7, 'injection_rate': 1.0},
    ]
    results = multi_scenario_optimization(scenarios)
    for r in results:
        print(f"  {r['scenario_name']}: 最优深度={r['optimal_depth_m']:.0f}m, "
              f"效率={r['efficiency_score']:.4f}, 再暴露={r['reexposure_time_years']:.0f}年")


def demo_uncertainty():
    print_section("模块 10: 碳循环参数不确定性量化 (超球面采样)")
    

    result = carbon_cycle_uncertainty_analysis(
        DIC_surf=2000.0, TA_surf=2300.0, T=15.0, S=35.0,
        n_samples=300, seed=42
    )
    
    pH_unc = result['pH_uncertainty']
    omega_unc = result['omega_uncertainty']
    
    print(f"pH 不确定性:")
    print(f"  均值={pH_unc['mean']:.4f}, 标准差={pH_unc['std']:.4f}")
    print(f"  范围=[{pH_unc['min']:.4f}, {pH_unc['max']:.4f}]")
    print(f"  变异系数 CV={pH_unc['cv']*100:.2f}%")
    
    print(f"\n文石饱和度 Ω 不确定性:")
    print(f"  均值={omega_unc['mean']:.4f}, 标准差={omega_unc['std']:.4f}")
    print(f"  范围=[{omega_unc['min']:.4f}, {omega_unc['max']:.4f}]")
    
    print(f"\nSobol 一阶敏感性指数 (pH):")
    for name, s in result['sobol_pH'].items():
        bar = "█" * int(s * 20)
        print(f"  {name:12s}: S₁={s:.4f} {bar}")


def demo_region_connectivity():
    print_section("模块 11: 海洋区域连通性图网络")
    
    graph = create_ocean_basin_graph(n_regions=10, basin_radius=300.0, seed=42)
    print(f"创建海洋盆地图: {graph.n_nodes} 个区域, {graph.n_edges} 条连通边")
    

    components = graph.connected_components()
    print(f"  连通分量数: {len(components)}")
    

    path_result = carbon_transport_path_analysis(graph, 0, 5)
    if path_result['path_exists']:
        print(f"  区域 0 → 区域 5 最短路径:")
        print(f"    路径长度(阻抗): {path_result['path_length']:.4f}")
        print(f"    经过区域: {' → '.join(str(n) for n in path_result['path_nodes'])}")
    

    grf_str = graph.to_grf_string()
    lines = grf_str.split("\n")
    print(f"  GRF 格式输出 ({len(lines)} 行):")
    for line in lines[:3]:
        print(f"    {line[:80]}...")
    

    graph2 = OceanRegionGraph.from_grf_string(grf_str)
    print(f"  解析验证: {graph2.n_nodes} 节点, {graph2.n_edges} 边")


def main():
    np.random.seed(42)
    
    print("\n" + "=" * 70)
    print("  海洋酸化与碳循环综合模拟系统 (PROJECT_54)")
    print("  Ocean Acidification and Carbon Cycle Synthesis Model")
    print("=" * 70)
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  NumPy:  {np.__version__}")
    
    try:
        demo_carbonate_chemistry()
        demo_ocean_mesh()
        demo_vertical_profiles()
        demo_carbon_transport()
        demo_quadrature()
        demo_sensor_network()
        demo_sparse_interpolation()
        demo_front_detection()
        demo_sequestration()
        demo_uncertainty()
        demo_region_connectivity()
        
        print("\n" + "=" * 70)
        print("  所有模块运行成功！")
        print("=" * 70)
        return 0
    
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
