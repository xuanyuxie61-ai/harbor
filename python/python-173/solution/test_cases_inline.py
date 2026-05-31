from error_indicator import chebyshev_nodes, divided_differences, newton_interpolation, chebyshev_approximate_1d
from fem_solver import shape_functions_p1, assemble_fem_system
from cvt_mesh import compute_voronoi_cells, cvt_iterate, cvt_energy
from shepard_transfer import shepard_interp_nd
from time_integrator import backward_euler_step
from quadrature_rules import gauss_legendre_1d
from hex_boundary import axial_to_cartesian, cartesian_to_axial, hex_round, generate_hexagonal_lattice
from graph_mesh import adj_bandwidth

# ---- TC01: chebyshev_nodes 输出长度、边界、唯一性 ----
import numpy as np
x = chebyshev_nodes(0.0, 1.0, 7)
assert len(x) == 7, '[TC01] chebyshev_nodes 长度 FAILED'
assert np.all(x >= 0.0) and np.all(x <= 1.0), '[TC01] chebyshev_nodes 边界 FAILED'
assert len(np.unique(np.round(x, 10))) == 7, '[TC01] chebyshev_nodes 唯一性 FAILED'

# ---- TC02: divided_differences 线性数据高阶差商为零 ----
x2 = np.array([0.0, 1.0, 2.0, 3.0])
y2 = np.array([1.0, 3.0, 5.0, 7.0])
dd2 = divided_differences(x2, y2)
assert abs(dd2[-1]) < 1e-12, '[TC02] divided_differences 线性高阶差商 FAILED'

# ---- TC03: newton_interpolation 精确复现数据点 ----
x_nodes = np.array([0.0, 0.5, 1.0])
y_nodes = np.array([0.0, 0.25, 1.0])
dd3 = divided_differences(x_nodes, y_nodes)
y_eval = newton_interpolation(x_nodes, dd3, x_nodes)
assert np.allclose(y_eval, y_nodes, atol=1e-12), '[TC03] newton_interpolation 复现 FAILED'

# ---- TC04: chebyshev_approximate_1d 线性函数误差接近零 ----
def f_linear(x):
    return 2.0 * x + 1.0
max_err, dd4, x_cheb = chebyshev_approximate_1d(f_linear, -1.0, 1.0, 5)
assert max_err < 1e-10, '[TC04] chebyshev_approximate_1d 线性误差 FAILED'

# ---- TC05: shape_functions_p1 单位分解性 ----
phi, grad_phi = shape_functions_p1(0.3, 0.3)
assert abs(np.sum(phi) - 1.0) < 1e-12, '[TC05] shape_functions_p1 单位分解 FAILED'
assert np.all(phi >= -1e-12), '[TC05] shape_functions_p1 非负性 FAILED'

# ---- TC06: shape_functions_p1 节点处 delta 性质 ----
phi06, _ = shape_functions_p1(0.0, 0.0)
assert abs(phi06[0] - 1.0) < 1e-12, '[TC06] shape_functions_p1 delta 1 FAILED'
assert abs(phi06[1]) < 1e-12 and abs(phi06[2]) < 1e-12, '[TC06] shape_functions_p1 delta 0 FAILED'

# ---- TC07: triangle_area 直角三角形面积 ----
nodes07 = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
area07 = triangle_area(nodes07, [0, 1, 2])
assert abs(area07 - 0.5) < 1e-12, '[TC07] triangle_area FAILED'

# ---- TC08: TriangleQuadrature 常数积分 = 0.5 ----
quad08 = TriangleQuadrature(degree=2)
integral08 = quad08.integrate(lambda xi, eta: 1.0)
assert abs(integral08 - 0.5) < 1e-12, '[TC08] TriangleQuadrature 常数积分 FAILED'

# ---- TC09: gauss_legendre_1d 权重和 = 2 ----
x_g, w_g = gauss_legendre_1d(5)
assert abs(np.sum(w_g) - 2.0) < 1e-12, '[TC09] gauss_legendre_1d 权重和 FAILED'

# ---- TC10: compute_mass_matrix_lumped 输出正定性和形状 ----
nodes10 = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
tris10 = np.array([[0, 1, 2], [1, 3, 2]])
M10 = compute_mass_matrix_lumped(nodes10, tris10)
assert len(M10) == 4, '[TC10] mass matrix 形状 FAILED'
assert np.all(M10 > 0), '[TC10] mass matrix 正定性 FAILED'

# ---- TC11: axial_to_cartesian 往返一致性 ----
q0, r0 = 3, -2
x11, y11 = axial_to_cartesian(q0, r0, size=0.5)
q_back, r_back = cartesian_to_axial(x11, y11, size=0.5)
assert abs(q_back - q0) < 1e-10, '[TC11] axial 往返 q FAILED'
assert abs(r_back - r0) < 1e-10, '[TC11] axial 往返 r FAILED'

# ---- TC12: hex_round 输出整数类型 ----
np.random.seed(42)
q_rand = np.random.randn(10)
r_rand = np.random.randn(10)
q_int, r_int = hex_round(q_rand, r_rand)
assert np.issubdtype(q_int.dtype, np.integer), '[TC12] hex_round 整数类型 FAILED'

# ---- TC13: compute_voronoi_cells 输出形状和索引范围 ----
np.random.seed(42)
samples13 = np.random.rand(200, 2)
gens13 = np.random.rand(6, 2)
nearest13 = compute_voronoi_cells(samples13, gens13)
assert len(nearest13) == 200, '[TC13] voronoi cells 长度 FAILED'
assert np.all(nearest13 >= 0) and np.all(nearest13 < 6), '[TC13] voronoi cells 范围 FAILED'

# ---- TC14: cvt_energy 非负性 ----
np.random.seed(42)
samples14 = np.random.rand(200, 2)
gens14 = np.random.rand(8, 2)
energy14 = cvt_energy(gens14, samples14)
assert energy14 >= 0.0, '[TC14] cvt_energy 非负 FAILED'

# ---- TC15: shepard_interp_nd 数据点处精确复现 ----
xd15 = np.array([[0., 0.], [1., 0.], [0., 1.]])
zd15 = np.array([1.0, 2.0, 3.0])
zi15 = shepard_interp_nd(2, xd15, zd15, p=2.0, xi=xd15.copy())
assert np.allclose(zi15, zd15, atol=1e-12), '[TC15] shepard 精确复现 FAILED'

# ---- TC16: shepard_interp_nd p=0 简单平均 ----
zi16 = shepard_interp_nd(2, xd15, zd15, p=0.0, xi=np.array([[0.5, 0.5]]))
assert abs(zi16[0] - np.mean(zd15)) < 1e-12, '[TC16] shepard p=0 平均 FAILED'

# ---- TC17: build_mesh_adjacency 简单网格连通性 ----
nodes17 = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
tris17 = np.array([[0, 1, 2], [1, 3, 2]])
adj_list17, adj_row17, adj17 = build_mesh_adjacency(4, tris17)
assert len(adj_list17) == 4, '[TC17] 邻接表长度 FAILED'
assert 2 in adj_list17[0], '[TC17] 邻接 node0->node2 FAILED'

# ---- TC18: graph_is_connected 简单网格连通 ----
assert graph_is_connected(adj_list17) == True, '[TC18] 图连通 FAILED'

# ---- TC19: adj_bandwidth 正整数值 ----
bw19 = adj_bandwidth(4, adj_row17, adj17)
assert bw19 >= 1, '[TC19] 带宽正值 FAILED'
assert isinstance(bw19, (int, np.integer)), '[TC19] 带宽整数 FAILED'

# ---- TC20: compute_element_adjacency 邻居检测 ----
elem_neighbors20 = compute_element_adjacency(2, tris17)
assert len(elem_neighbors20) == 2, '[TC20] 单元邻接长度 FAILED'
assert 1 in elem_neighbors20[0], '[TC20] 单元邻接邻居 FAILED'

# ---- TC21: compute_mesh_quality 在 [0, 1] 内 ----
quality21 = compute_mesh_quality(nodes17, tris17)
assert np.all(quality21 >= 0.0), '[TC21] 网格质量 >= 0 FAILED'
assert np.all(quality21 <= 1.0 + 1e-12), '[TC21] 网格质量 <= 1 FAILED'

# ---- TC22: backward_euler_step 收敛性和输出形状 ----
def rhs22(u):
    return -u
u0_22 = np.array([1.0, 2.0, 3.0])
u_new22, conv22 = backward_euler_step(u0_22, 0.01, rhs22)
assert conv22 == True, '[TC22] backward_euler 收敛 FAILED'
assert len(u_new22) == 3, '[TC22] backward_euler 形状 FAILED'

# ---- TC23: compute_cfl_condition 正值输出 ----
def v_test23(x, y):
    return 0.5, 0.0
def D_test23(x, y):
    return 0.1
dt_max23, h_min23, pe_max23 = compute_cfl_condition(nodes17, tris17, v_test23, D_test23)
assert dt_max23 > 0, '[TC23] CFL dt 正值 FAILED'
assert h_min23 > 0, '[TC23] CFL h_min 正值 FAILED'

# ---- TC24: identify_boundary_edges 边界节点数 ----
b_nodes24, d_nodes24, n_edges24 = identify_boundary_edges(
    nodes17, tris17, ((0., 1.), (0., 1.))
)
assert len(b_nodes24) == 4, '[TC24] 边界节点数 FAILED'

# ---- TC25: compute_reaction_term 输出形状和有限值 ----
u_curr25 = np.array([0.1, 0.2, 0.3, 0.4])
def R_test25(u, x, y):
    return 5.0 * u * (1.0 - u)
b_R25 = compute_reaction_term(nodes17, tris17, u_curr25, R_test25)
assert len(b_R25) == 4, '[TC25] 反应项形状 FAILED'
assert np.all(np.isfinite(b_R25)), '[TC25] 反应项有限值 FAILED'

# ---- TC26: generate_cvt_mesh 可复现性 (内部 seed=42) ----
np.random.seed(42)
gens26a, ehist26a = generate_cvt_mesh(20, ((0., 1.), (0., 1.)), it_max=3, sample_multiplier=20)
np.random.seed(42)
gens26b, ehist26b = generate_cvt_mesh(20, ((0., 1.), (0., 1.)), it_max=3, sample_multiplier=20)
assert np.allclose(gens26a, gens26b, atol=1e-10), '[TC26] CVT 可复现 FAILED'

# ---- TC27: generate_hexagonal_lattice 非空二维输出 ----
points27, axial27 = generate_hexagonal_lattice(radius=2, size=1.0)
assert len(points27) > 0, '[TC27] 六边形格点非空 FAILED'
assert points27.shape[1] == 2, '[TC27] 六边形格点二维 FAILED'

# ---- TC28: assemble_fem_system 刚度矩阵对称性 ----
nodes28 = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
tris28 = np.array([[0, 1, 2], [1, 3, 2]])
def D_const28(x, y): return 1.0
def c_const28(x, y): return 0.0
def f_const28(x, y): return 1.0
A28, b28 = assemble_fem_system(nodes28, tris28, D_const28, c_const28, f_const28, quad_degree=3)
assert np.allclose(A28, A28.T, atol=1e-12), '[TC28] FEM 对称性 FAILED'

# ---- TC29: adaptive_time_stepping 到达终点时间 ----
def rhs29(u):
    return -0.5 * u
u0_29 = np.array([1.0])
t_hist29, u_hist29, dt_hist29 = adaptive_time_stepping(
    u0_29, (0.0, 0.05), 0.01, rhs29, scheme='bdf1', dt_min=1e-6, dt_max=0.1
)
assert len(t_hist29) > 1, '[TC29] 时间步进历史 FAILED'
assert t_hist29[-1] >= 0.049, '[TC29] 时间步进终点 FAILED'

# ---- TC30: cvt_iterate 迭代后能量非负 ----
np.random.seed(42)
gens30 = np.random.rand(12, 2)
samples30 = np.random.rand(300, 2)
gens30_new, diff30, energy30 = cvt_iterate(gens30, samples30)
assert energy30 >= 0.0, '[TC30] cvt_iterate 能量非负 FAILED'
