# ---- TC01: equilibrium_constants 返回正数常数 ----
K = equilibrium_constants(15.0, 35.0)
assert K['K1'] > 0 and K['K2'] > 0 and K['Kw'] > 0, '[TC01] 平衡常数必须为正 FAILED'

# ---- TC02: solve_carbonate_system pH 在合理范围 ----
res = solve_carbonate_system(2.0e-3, 2.3e-3, 15.0, 35.0)
assert 7.0 <= res['pH'] <= 9.0, '[TC02] pH 超出合理范围 FAILED'

# ---- TC03: solve_carbonate_system 饱和度大于 0 ----
assert res['Omega_calcite'] > 0, '[TC03] 方解石饱和度必须为正 FAILED'
assert res['Omega_aragonite'] > 0, '[TC03] 文石饱和度必须为正 FAILED'

# ---- TC04: batch_solve_carbonate 输出长度匹配输入 ----
DIC_arr = np.array([2000.0, 2100.0])
TA_arr = np.array([2300.0, 2400.0])
T_arr = np.array([15.0, 10.0])
S_arr = np.array([35.0, 35.0])
batch_res = batch_solve_carbonate(DIC_arr, TA_arr, T_arr, S_arr, units='umolkg')
assert len(batch_res) == len(DIC_arr), '[TC04] 批量输出长度不匹配 FAILED'

# ---- TC05: air_sea_co2_flux 符号与 pCO2 差一致 ----
flux = air_sea_co2_flux(400.0, 410.0, 15.0, 35.0, u10=5.0)
assert flux < 0, '[TC05] 海洋 pCO2 低于大气时应为吸收(负) FAILED'

# ---- TC06: generate_ocean_rectangle_mesh 节点和单元数正确 ----
node_xy, element_node, nx_out, ny_out = generate_ocean_rectangle_mesh(100.0, 100.0, 5, 5)
assert node_xy.shape[0] == 36, '[TC06] 矩形网格节点数错误 FAILED'
assert element_node.shape[0] == 25, '[TC06] 矩形网格单元数错误 FAILED'

# ---- TC07: compute_element_areas 总面积等于域面积 ----
areas, total_area = compute_element_areas(node_xy, element_node)
assert abs(total_area - 10000.0) < 1.0, '[TC07] 矩形网格总面积错误 FAILED'

# ---- TC08: mesh_bandwidth 为非负整数 ----
bw = mesh_bandwidth(element_node, node_xy.shape[0])
assert bw['ml'] >= 0 and bw['mu'] >= 0 and bw['m'] > 0, '[TC08] 带宽非负 FAILED'

# ---- TC09: Hermite 样条精确重构节点值 ----
z_nodes = np.array([0.0, 10.0, 20.0])
f_nodes = np.array([25.0, 24.0, 22.0])
d_nodes = estimate_derivatives_central(z_nodes, f_nodes)
spline = build_hermite_spline(z_nodes, f_nodes, d_nodes)
f_eval, _, _, _ = evaluate_hermite_spline(spline, z_nodes)
assert np.allclose(f_eval, f_nodes), '[TC09] Hermite 样条节点重构失败 FAILED'

# ---- TC10: integrate_hermite_spline 线性函数积分精确 ----
z_nodes = np.array([0.0, 1.0, 2.0])
f_nodes = np.array([0.0, 1.0, 2.0])
d_nodes = np.ones_like(f_nodes)
spline = build_hermite_spline(z_nodes, f_nodes, d_nodes)
integral = integrate_hermite_spline(spline, 0, 2)
assert abs(integral - 2.0) < 1e-10, '[TC10] 线性函数积分不精确 FAILED'

# ---- TC11: mixed_layer_depth 在深度范围内 ----
z_nodes = np.array([0, 10, 20, 50, 100])
T_nodes = np.array([25.0, 24.8, 24.0, 20.0, 15.0])
mld = mixed_layer_depth(z_nodes, T_nodes, threshold=0.5)
assert z_nodes[0] <= mld <= z_nodes[-1], '[TC11] MLD 超出深度范围 FAILED'

# ---- TC12: euler_forward 输出时间序列长度正确 ----
def simple_dydt(t, y):
    return -0.1 * y
t, y = euler_forward(simple_dydt, (0, 10), np.array([1.0]), 100)
assert len(t) == 101 and y.shape == (101, 1), '[TC12] euler_forward 输出尺寸错误 FAILED'

# ---- TC13: autocatalytic_carbonate_deriv CO2 消耗率为负 ----
deriv = autocatalytic_carbonate_deriv(0, np.array([100.0, 50.0, 20.0, 10.0]))
assert deriv[0] < 0, '[TC13] CO2 消耗率应为负 FAILED'

# ---- TC14: compute_anthropogenic_carbon_inventory 解析验证 ----
DIC_pre = np.array([1950.0, 1950.0])
DIC_post = np.array([2000.0, 2000.0])
inventory = compute_anthropogenic_carbon_inventory(DIC_pre, DIC_post, 1025.0, 10.0)
expected = 2 * 50.0 * 1025.0 * 10.0 * 1e-6
assert abs(inventory - expected) < 1e-6, '[TC14] 人为碳库存计算错误 FAILED'

# ---- TC15: integrate_over_cube 多项式解析验证 ----
def test_func(x, y, z):
    return x * y * z + 1.0
approx = integrate_over_cube(test_func, [0, 0, 0], [1, 1, 1], order_1d=(3, 3, 3))
assert abs(approx - 1.125) < 1e-12, '[TC15] 立方体积分解析验证失败 FAILED'

# ---- TC16: tetrahedron_volume 参考值验证 ----
v = np.array([
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])
vol = tetrahedron_volume(v)
assert abs(vol - 1.0/6.0) < 1e-12, '[TC16] 四面体体积解析验证失败 FAILED'

# ---- TC17: hypersphere_uniform_sample 单位范数 ----
np.random.seed(42)
samples = hypersphere_uniform_sample(3, 100, seed=42)
norm_samples = np.linalg.norm(samples, axis=0)
assert np.allclose(norm_samples, 1.0), '[TC17] 超球面采样范数不为 1 FAILED'

# ---- TC18: plan_ocean_sampling_route 路径闭合 ----
np.random.seed(42)
coords = np.array([[0, 0], [1, 0], [0.5, 1]])
route = plan_ocean_sampling_route(coords, seed=42)
assert route['path'][0] == route['path'][-1], '[TC18] TSP 路径不闭合 FAILED'

# ---- TC19: shepard_interp_nd 查询点与数据点重合时精确 ----
data_coords = np.array([[0.0, 1.0, 2.0]])
data_values = np.array([10.0, 20.0, 30.0])
query = np.array([[1.0]])
result = shepard_interp_nd(data_coords, data_values, 2.0, query)
assert abs(result[0] - 20.0) < 1e-10, '[TC19] Shepard 精确插值失败 FAILED'

# ---- TC20: cross_validate_shepard 返回最优 p 在候选列表中 ----
np.random.seed(42)
dc = np.random.rand(2, 10)
dv = np.random.rand(10)
cv_res = cross_validate_shepard(dc, dv, p_values=[1.0, 2.0, 3.0])
assert cv_res['best_p'] in [1.0, 2.0, 3.0], '[TC20] 交叉验证最优 p 不在候选列表 FAILED'

# ---- TC21: news_gradient 常数场梯度为零 ----
field = np.ones((10, 10))
grad = news_gradient(field)
assert np.allclose(grad, 0.0), '[TC21] 常数场 NEWS 梯度不为零 FAILED'

# ---- TC22: detect_fronts_multi_field 输出尺寸与输入一致 ----
T_field = np.random.rand(20, 20)
S_field = np.random.rand(20, 20)
fronts = detect_fronts_multi_field({'T': T_field, 'S': S_field}, weights={'T': 1.0, 'S': 0.5}, threshold_percentile=90)
assert fronts['front_mask'].shape == T_field.shape, '[TC22] 锋面掩码尺寸不匹配 FAILED'

# ---- TC23: golden_section_search 最小值在区间内且精确 ----
f_quad = lambda x: (x - 3.0)**2
res = golden_section_search(f_quad, 0.0, 10.0, n_iterations=50)
assert 0.0 <= res['x_opt'] <= 10.0, '[TC23] 黄金分割最优解超出区间 FAILED'
assert abs(res['x_opt'] - 3.0) < 1e-3, '[TC23] 黄金分割最优解不精确 FAILED'

# ---- TC24: compute_boundary_edges 矩形网格边界边数合理 ----
node_xy, element_node, _, _ = generate_ocean_rectangle_mesh(100.0, 100.0, 5, 5)
boundary_edges, n_bound = compute_boundary_edges(element_node)
assert n_bound > 0, '[TC24] 边界边数必须为正 FAILED'

# ---- TC25: OceanRegionGraph 连通分量数不超过节点数 ----
np.random.seed(42)
graph = create_ocean_basin_graph(n_regions=8, basin_radius=300.0, seed=42)
comps = graph.connected_components()
assert len(comps) >= 1 and len(comps) <= graph.n_nodes, '[TC25] 连通分量数异常 FAILED'

# ---- TC26: carbon_transport_path_analysis 路径存在性 ----
path_res = carbon_transport_path_analysis(graph, 0, 1)
assert path_res['path_exists'] == True, '[TC26] 相邻节点间路径不应不存在 FAILED'

# ---- TC27: hypersphere01_area S^2 = 4π ----
area_s2 = hypersphere01_area(3)
assert abs(area_s2 - 4.0 * np.pi) < 1e-10, '[TC27] S^2 表面积不等于 4π FAILED'
