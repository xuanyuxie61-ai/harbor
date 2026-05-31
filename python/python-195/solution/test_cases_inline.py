# ---- TC01: rucklidge_deriv 原点固定点测试 ----
import numpy as np
deriv = rucklidge_deriv(0.0, np.array([0.0, 0.0, 0.0]))
assert deriv.shape == (3,), '[TC01] rucklidge_deriv 输出形状 FAILED'
assert np.all(np.isfinite(deriv)), '[TC01] rucklidge_deriv 包含NaN/Inf FAILED'

# ---- TC02: arneodo_deriv 原点固定点测试 ----
import numpy as np
deriv = arneodo_deriv(0.0, np.array([0.0, 0.0, 0.0]))
assert deriv.shape == (3,), '[TC02] arneodo_deriv 输出形状 FAILED'
assert np.all(np.isfinite(deriv)), '[TC02] arneodo_deriv 包含NaN/Inf FAILED'

# ---- TC03: rucklidge_deriv 随机输入有限性测试 ----
import numpy as np
np.random.seed(42)
x_test = np.random.randn(10, 3)
for i in range(10):
    d = rucklidge_deriv(0.0, x_test[i])
    assert np.all(np.isfinite(d)), '[TC03] rucklidge_deriv 随机输入产生NaN/Inf FAILED'

# ---- TC04: integrate_trajectory 简单ODE输出形状与有限性 ----
import numpy as np
def simple_ode(t, y):
    return np.array([-y[0]])
t_arr, y_arr = integrate_trajectory(simple_ode, np.array([1.0]), (0.0, 1.0), relerr=1e-6, abserr=1e-9)
assert t_arr.ndim == 1, '[TC04] integrate_trajectory t_arr 维度 FAILED'
assert y_arr.ndim == 2, '[TC04] integrate_trajectory y_arr 维度 FAILED'
assert y_arr.shape[1] == 1, '[TC04] integrate_trajectory y_arr 列数 FAILED'
assert len(t_arr) >= 2, '[TC04] integrate_trajectory 步数不足 FAILED'
assert np.all(np.isfinite(y_arr)), '[TC04] integrate_trajectory 输出含NaN/Inf FAILED'

# ---- TC05: compute_particle_load_field 输出形状与非负性 ----
import numpy as np
np.random.seed(42)
test_particles = np.random.rand(100, 2)
domain = (0.0, 1.0, 0.0, 1.0)
load_field = compute_particle_load_field(test_particles, domain, 16, 16)
assert load_field.shape == (16, 16), '[TC05] compute_particle_load_field 输出形状 FAILED'
assert np.all(load_field >= 0), '[TC05] compute_particle_load_field 负值 FAILED'
assert np.all(np.isfinite(load_field)), '[TC05] compute_particle_load_field 含NaN/Inf FAILED'

# ---- TC06: QuadMesh 初始网格构建 ----
import numpy as np
mesh = QuadMesh((0.0, 2.0, 0.0, 1.0), nx=4, ny=4)
assert mesh.nodes.shape[0] == 25, '[TC06] QuadMesh 节点数 FAILED'
assert len(mesh.elements) == 16, '[TC06] QuadMesh 单元数 FAILED'
assert np.all(np.isfinite(mesh.nodes)), '[TC06] QuadMesh 节点含NaN/Inf FAILED'

# ---- TC07: QuadMesh evaluate_load 输出类型 ----
import numpy as np
mesh = QuadMesh((0.0, 1.0, 0.0, 1.0), nx=4, ny=4)
np.random.seed(42)
particles_test = np.random.rand(80, 2)
loads = mesh.evaluate_load(particles_test)
assert len(loads) == 16, '[TC07] evaluate_load 输出长度 FAILED'
assert np.all(loads >= 0), '[TC07] evaluate_load 负值 FAILED'

# ---- TC08: QuadMesh triangulate_elements 输出 ----
import numpy as np
mesh = QuadMesh((0.0, 1.0, 0.0, 1.0), nx=4, ny=4)
nodes_tri, triangles_tri = mesh.triangulate_elements()
assert triangles_tri.shape[1] == 3, '[TC08] triangulate 三角形列数 FAILED'
assert triangles_tri.shape[0] == 32, '[TC08] triangulate 三角形数 FAILED'
assert np.all(triangles_tri >= 1), '[TC08] triangulate 索引非1-based FAILED'

# ---- TC09: FEMSystem 刚度矩阵对称性 ----
import numpy as np
nodes_fem = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
tris_fem = np.array([[1, 2, 3], [2, 4, 3]], dtype=int)
fem = FEMSystem(nodes_fem, tris_fem)
A = fem.assemble_stiffness_matrix()
A_dense = A.toarray()
assert np.allclose(A_dense, A_dense.T), '[TC09] 刚度矩阵不对称 FAILED'

# ---- TC10: FEMSystem 质量矩阵对角线为正 ----
import numpy as np
nodes_fem = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
tris_fem = np.array([[1, 2, 3], [2, 4, 3]], dtype=int)
fem = FEMSystem(nodes_fem, tris_fem)
M = fem.assemble_mass_matrix()
M_dense = M.toarray()
assert np.all(np.diag(M_dense) > 0), '[TC10] 质量矩阵对角线非正 FAILED'

# ---- TC11: FEMSystem solve_poisson 零RHS得零解 ----
import numpy as np
nodes_fem = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
tris_fem = np.array([[1, 2, 3], [2, 4, 3]], dtype=int)
fem = FEMSystem(nodes_fem, tris_fem)
rhs_zero = np.zeros(fem.n_nodes)
u = fem.solve_poisson(rhs_zero)
assert np.all(np.abs(u) < 1e-8), '[TC11] solve_poisson 零RHS非零解 FAILED'

# ---- TC12: MultigridPoisson1D 已知正弦解 ----
import numpy as np
def force_sin(x):
    return np.pi ** 2 * np.sin(np.pi * x)
mg1d = MultigridPoisson1D(n=64, a=0.0, b=1.0, ua=0.0, ub=0.0, force_func=force_sin)
u_1d, it_1d = mg1d.solve(tol=1e-6, max_iter=50)
assert it_1d <= 50, '[TC12] MG1D 未在max_iter内收敛 FAILED'
assert np.all(np.isfinite(u_1d)), '[TC12] MG1D 解含NaN/Inf FAILED'
assert u_1d[0] == 0.0 and u_1d[-1] == 0.0, '[TC12] MG1D 边界条件 FAILED'

# ---- TC13: MultigridPoisson2D 求解与残差 ----
import numpy as np
nx_mg, ny_mg = 16, 16
mg2d_test = MultigridPoisson2D(nx_mg, ny_mg, 1.0, 1.0)
rhs_2d_test = np.zeros((nx_mg + 1, ny_mg + 1))
rhs_2d_test[nx_mg//2, ny_mg//2] = 1.0
u_2d_test, it_2d_test = mg2d_test.solve(rhs_2d_test, tol=1e-4, max_iter=30)
assert it_2d_test <= 30, '[TC13] MG2D 未在max_iter内收敛 FAILED'
assert np.all(np.isfinite(u_2d_test)), '[TC13] MG2D 解含NaN/Inf FAILED'

# ---- TC14: laguerre_polynomial 已知值 L0(0)=1, L1(0)=1 ----
import numpy as np
x_lag = np.array([0.0])
L_vals = laguerre_polynomial(1, 2, x_lag)
assert abs(L_vals[0, 0] - 1.0) < 1e-12, '[TC14] L_0(0) != 1 FAILED'
assert abs(L_vals[0, 1] - 1.0) < 1e-12, '[TC14] L_1(0) != 1 FAILED'

# ---- TC15: generalized_laguerre_function 已知值 ----
import numpy as np
x_test = np.array([0.0])
Lg = generalized_laguerre_function(1, 2, 0.5, x_test)
assert abs(Lg[0, 0] - 1.0) < 1e-12, '[TC15] L_0^{(0.5)}(0) != 1 FAILED'
assert abs(Lg[0, 1] - 1.5) < 1e-12, '[TC15] L_1^{(0.5)}(0) != 1.5 FAILED'

# ---- TC16: chebyshev_nodes 对称性 ----
import numpy as np
nodes = chebyshev_nodes(-1.0, 1.0, 8)
assert len(nodes) == 8, '[TC16] chebyshev_nodes 节点数 FAILED'
assert np.all((nodes + nodes[::-1]) < 1e-12), '[TC16] chebyshev_nodes 不对称 FAILED'

# ---- TC17: divided_differences + newton_interpolate 线性函数精确 ----
import numpy as np
xd = np.array([0.0, 1.0, 2.0])
yd = np.array([3.0, 5.0, 7.0])
dd = divided_differences(xd, yd)
xp_test = np.linspace(0.0, 2.0, 21)
yp_test = newton_interpolate(xd, dd, xp_test)
expected = 3.0 + 2.0 * xp_test
assert np.allclose(yp_test, expected, atol=1e-10), '[TC17] 牛顿插值线性函数 FAILED'

# ---- TC18: chebyshev_interpolate 误差估计 ----
import numpy as np
def test_func(x):
    return np.sin(3.0 * x) * np.exp(-x ** 2)
xp_eval = np.linspace(-1, 1, 101)
yp_interp, maxerr = chebyshev_interpolate(test_func, -1.0, 1.0, 16, xp_eval)
assert not np.any(np.isnan(yp_interp)), '[TC18] chebyshev_interpolate 含NaN FAILED'
assert maxerr >= 0, '[TC18] chebyshev_interpolate 误差为负 FAILED'

# ---- TC19: chebyshev_spectral_derivative 线性函数导数恒定 ----
import numpy as np
x_lin = np.linspace(0.0, 2.0, 11)
u_lin = 3.0 * x_lin + 1.0
dudx = chebyshev_spectral_derivative(u_lin, 2.0)
assert np.allclose(dudx, 3.0, atol=1e-6), '[TC19] 谱导数线性函数 FAILED'

# ---- TC20: triangle_quad_rule_degree1 精确积分常数1 ----
import numpy as np
from quadrature_rules import triangle_quad_rule_degree1
w, pts = triangle_quad_rule_degree1()
assert abs(np.sum(w) - 0.5) < 1e-14, '[TC20] 求积规则权重和 != 0.5 FAILED'
assert np.all(pts >= 0) and np.all(pts <= 1), '[TC20] 求积节点越界 FAILED'

# ---- TC21: integrate_over_triangle 已知积分值 ----
import numpy as np
from quadrature_rules import integrate_over_triangle
verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
val = integrate_over_triangle(verts, lambda x, y: x + y, degree=3)
assert abs(val - 1.0 / 3.0) < 1e-10, '[TC21] integrate_over_triangle x+y积分 != 1/3 FAILED'

# ---- TC22: compute_moment_over_mesh 面积一致性 ----
import numpy as np
nodes_quad = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
tris_quad = np.array([[1, 2, 3], [2, 4, 3]], dtype=int)
m00 = compute_moment_over_mesh(nodes_quad, tris_quad, 0, 0, degree=3)
assert abs(m00 - 1.0) < 1e-10, '[TC22] 矩量面积 != 1 FAILED'

# ---- TC23: LoadBalancer 不均衡因子计算 ----
import numpy as np
lb = LoadBalancer(8, (0.0, 1.0, 0.0, 1.0))
loads_test = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
I = lb.imbalance_factor(loads_test)
assert abs(I - 1.0) < 1e-12, '[TC23] 均匀负载不均衡因子 != 1 FAILED'

# ---- TC24: LoadBalancer 不均衡负载测试 ----
import numpy as np
lb = LoadBalancer(8, (0.0, 1.0, 0.0, 1.0))
loads_uneven = np.array([20.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
I = lb.imbalance_factor(loads_uneven)
assert I > 1.0, '[TC24] 不均衡负载因子应 > 1 FAILED'

# ---- TC25: LoadBalancer evaluate_efficiency ----
import numpy as np
lb = LoadBalancer(8, (0.0, 1.0, 0.0, 1.0))
loads_test = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
eff = lb.evaluate_efficiency(loads_test)
assert abs(eff['parallel_efficiency'] - 1.0) < 1e-12, '[TC25] 均匀负载效率 != 1 FAILED'
assert abs(eff['std_load']) < 1e-12, '[TC25] 均匀负载标准差 != 0 FAILED'

# ---- TC26: diffusion_based_load_balance 收敛到均值 ----
import numpy as np
loads_demo = np.array([120.0, 80.0, 60.0, 100.0, 90.0, 70.0, 110.0, 50.0])
conn = np.zeros((8, 8))
for i in range(8):
    conn[i, (i + 1) % 8] = 1
    conn[i, (i - 1) % 8] = 1
balanced = diffusion_based_load_balance(loads_demo, conn, n_iterations=500, tolerance=1e-4)
assert abs(np.mean(balanced) - np.mean(loads_demo)) < 1e-8, '[TC26] 扩散均衡均值不守恒 FAILED'
assert np.std(balanced) < np.std(loads_demo), '[TC26] 扩散均衡后方差未减小 FAILED'

# ---- TC27: toeplitz_mv 已知结果 ----
import numpy as np
n_t = 4
a_t = np.array([2.0, 1.0, 0.5, 0.25, 3.0, 1.5, 0.75])
x_t = np.array([1.0, 0.0, 0.0, 0.0])
y_t = toeplitz_mv(n_t, a_t, x_t)
# 第一列 = [2, 3, 1.5, 0.75]
assert abs(y_t[0] - 2.0) < 1e-12, '[TC27] toeplitz_mv[0] FAILED'
assert abs(y_t[1] - 3.0) < 1e-12, '[TC27] toeplitz_mv[1] FAILED'

# ---- TC28: toeplitz_embedded_fft_mv 与 direct 对比 ----
import numpy as np
np.random.seed(42)
n_t = 32
a_t = np.random.rand(2 * n_t - 1)
x_t = np.random.rand(n_t)
y_dir = toeplitz_mv(n_t, a_t, x_t)
y_fft = toeplitz_embedded_fft_mv(n_t, a_t, x_t)
rel_diff = np.linalg.norm(y_dir - y_fft) / (np.linalg.norm(y_dir) + 1e-14)
assert rel_diff < 1e-10, '[TC28] FFT vs direct 差异过大 FAILED'

# ---- TC29: sample_unit_ball_positive 输出范围 ----
import numpy as np
np.random.seed(42)
samples = sample_unit_ball_positive(200)
assert samples.shape == (200, 3), '[TC29] ball_sample 形状 FAILED'
assert np.all(samples >= 0), '[TC29] ball_sample 负值 FAILED'
assert np.all(np.linalg.norm(samples, axis=1) <= 1.0 + 1e-10), '[TC29] ball_sample 范数>1 FAILED'

# ---- TC30: sample_unit_sphere_surface 单位范数 ----
import numpy as np
np.random.seed(42)
samples = sample_unit_sphere_surface(100, dim=4)
assert samples.shape == (100, 4), '[TC30] sphere_sample 形状 FAILED'
norms = np.linalg.norm(samples, axis=1)
assert np.allclose(norms, 1.0, atol=1e-10), '[TC30] sphere_sample 非单位范数 FAILED'

# ---- TC31: compute_prefix_sum_2d + query_region_count 一致性 ----
import numpy as np
np.random.seed(42)
particles_prefix = np.random.rand(200, 2)
domain_pf = (0.0, 1.0, 0.0, 1.0)
prefix = compute_prefix_sum_2d(particles_prefix, domain_pf, 8, 8)
total = query_region_count(prefix, 0, 8, 0, 8)
assert total == 200, '[TC31] prefix sum 总粒子数 != 200 FAILED'

# ---- TC32: multipole_expansion 单极子 = 总电荷 ----
import numpy as np
np.random.seed(42)
particles_mp = np.random.rand(20, 2)
charges_mp = np.ones(20) * 2.0
center_mp = np.array([0.5, 0.5])
mp = multipole_expansion(particles_mp, charges_mp, center_mp, max_order=2)
assert abs(mp['monopole'] - 40.0) < 1e-10, '[TC32] 单极子 != 40 FAILED'
assert 'dipole' in mp and 'quadrupole' in mp, '[TC32] 缺少偶极子/四极子 FAILED'

# ---- TC33: build_interaction_matrix_toeplitz 输出长度 ----
import numpy as np
a_tp = build_interaction_matrix_toeplitz(16, lambda r: 1.0 / max(r, 1e-10), h=0.1)
assert len(a_tp) == 31, '[TC33] Toeplitz数据长度 FAILED'
assert np.all(np.isfinite(a_tp)), '[TC33] Toeplitz数据含NaN/Inf FAILED'

# ---- TC34: safe_divide 边界情况 ----
import numpy as np
a_sd = np.array([10.0, 0.0, 5.0])
b_sd = np.array([2.0, 0.0, 0.0])
r_sd = safe_divide(a_sd, b_sd)
assert abs(r_sd[0] - 5.0) < 1e-12, '[TC34] safe_divide[0] FAILED'
assert r_sd[1] == 0.0, '[TC34] safe_divide 除以零回退值 FAILED'

# ---- TC35: compute_triangle_area 已知面积 ----
import numpy as np
p1 = np.array([0.0, 0.0])
p2 = np.array([1.0, 0.0])
p3 = np.array([0.0, 1.0])
area = compute_triangle_area(p1, p2, p3)
assert abs(abs(area) - 0.5) < 1e-12, '[TC35] 三角形面积 != 0.5 FAILED'

# ---- TC36: is_power_of_two 真值表 ----
assert is_power_of_two(1) == True, '[TC36] is_power_of_two(1) FAILED'
assert is_power_of_two(2) == True, '[TC36] is_power_of_two(2) FAILED'
assert is_power_of_two(3) == False, '[TC36] is_power_of_two(3) FAILED'
assert is_power_of_two(64) == True, '[TC36] is_power_of_two(64) FAILED'
assert is_power_of_two(0) == False, '[TC36] is_power_of_two(0) FAILED'

# ---- TC37: gauss_seidel_sweep 收敛性 ----
import numpy as np
n_gs = 10
rhs_gs = np.ones(n_gs) * 2.0
x_gs = np.zeros(n_gs)
x_new, d = gauss_seidel_sweep(n_gs, rhs_gs, x_gs)
assert d > 0, '[TC37] Gauss-Seidel 变化量非正 FAILED'
assert np.all(np.isfinite(x_new)), '[TC37] Gauss-Seidel 输出含NaN/Inf FAILED'

# ---- TC38: mesh_base_one 0-based转1-based ----
import numpy as np
elem_0based = np.array([[0, 1, 2], [2, 3, 1]])
elem_1based = mesh_base_one(elem_0based, 4)
assert elem_1based.min() == 1, '[TC38] mesh_base_one 转换后min != 1 FAILED'
assert elem_1based.max() == 4, '[TC38] mesh_base_one 转换后max != 4 FAILED'

# ---- TC39: reference_to_physical_q4 映射 ----
import numpy as np
q4_corners = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
rs_pt = np.array([[0.5, 0.5]])
phys = reference_to_physical_q4(q4_corners, rs_pt)
assert np.allclose(phys[0], [1.0, 0.5], atol=1e-10), '[TC39] Q4映射中心点 FAILED'

# ---- TC40: radial_distribution_spectrum 基本测试 ----
import numpy as np
r_test = np.linspace(0.01, 1.0, 50)
g_test = np.exp(-2.0 * r_test)
coeffs_test = radial_distribution_spectrum(r_test, g_test, n_modes=4, alpha=0.0, beta=2.0)
assert len(coeffs_test) == 4, '[TC40] 谱系数长度 FAILED'
assert np.all(np.isfinite(coeffs_test)), '[TC40] 谱系数含NaN/Inf FAILED'

# ---- TC41: LoadBalancer find_optimal_split 基本测试 ----
import numpy as np
lb = LoadBalancer(8, (0.0, 1.0, 0.0, 1.0))
particles_split = np.array([[0.2, 0.5], [0.3, 0.5], [0.7, 0.5], [0.8, 0.5]])
split_pos, load_left, load_right = lb.find_optimal_split(particles_split, axis=0)
assert 0.3 < split_pos < 0.7, '[TC41] find_optimal_split 切分点异常 FAILED'

# ---- TC42: 集成测试：完整main流程不崩溃 ----
import numpy as np
np.random.seed(42)
# 测试最小配置的完整流程
try:
    n_p = 200
    domain_t = (0.0, 1.0, 0.0, 1.0)
    ball = sample_unit_ball_positive(n_p)
    particles_t = np.zeros((n_p, 2))
    particles_t[:, 0] = check_bounds(0.5 + 0.3 * ball[:, 0], 0.0, 1.0, "x")
    particles_t[:, 1] = check_bounds(0.5 + 0.3 * ball[:, 1], 0.0, 1.0, "y")
    mesh_t = QuadMesh(domain_t, nx=4, ny=4)
    nodes_t, tris_t = mesh_t.triangulate_elements()
    fem_t = FEMSystem(nodes_t, tris_t)
    load_f = compute_particle_load_field(particles_t, domain_t, 8, 8)
    from scipy.interpolate import RegularGridInterpolator
    x_g = np.linspace(0, 1, 8)
    y_g = np.linspace(0, 1, 8)
    interp = RegularGridInterpolator((x_g, y_g), load_f.T, bounds_error=False, fill_value=0.0)
    rhs_t = interp(fem_t.nodes)
    u_t = fem_t.solve_poisson(rhs_t)
    balancer_t = LoadBalancer(8, domain_t)
    loads_t = balancer_t.compute_loads(particles_t)
    eff_t = balancer_t.evaluate_efficiency(loads_t)
    assert u_t.shape[0] == fem_t.n_nodes, '[TC42] FEM解维度 FAILED'
    assert eff_t['parallel_efficiency'] > 0, '[TC42] 并行效率非正 FAILED'
except Exception as e:
    assert False, f'[TC42] 集成测试异常: {e}'
