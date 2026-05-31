# ---- TC01: DualScalar 基本四则运算与幂运算 ----
a = DualScalar(2.0, 1.0)
b = DualScalar(3.0, 2.0)
c_add = a + b
assert abs(c_add.val - 5.0) < 1e-12, '[TC01] DualScalar addition val FAILED'
assert abs(c_add.der - 3.0) < 1e-12, '[TC01] DualScalar addition der FAILED'
c_mul = a * b
assert abs(c_mul.val - 6.0) < 1e-12, '[TC01] DualScalar multiplication val FAILED'
assert abs(c_mul.der - 7.0) < 1e-12, '[TC01] DualScalar multiplication der FAILED'
c_pow = a ** 2
assert abs(c_pow.val - 4.0) < 1e-12, '[TC01] DualScalar power val FAILED'
assert abs(c_pow.der - 4.0) < 1e-12, '[TC01] DualScalar power der FAILED'
c_sub = a - b
assert abs(c_sub.val + 1.0) < 1e-12, '[TC01] DualScalar subtraction val FAILED'
assert abs(c_sub.der + 1.0) < 1e-12, '[TC01] DualScalar subtraction der FAILED'
c_div = a / b
assert abs(c_div.val - 2.0 / 3.0) < 1e-12, '[TC01] DualScalar division val FAILED'

# ---- TC02: DualScalar 链式法则 sin*exp ----
from autodiff_core import dual_sin, dual_exp
x = DualScalar(1.0, 1.0)
f = dual_sin(x) * dual_exp(x)
expected_val = np.sin(1.0) * np.exp(1.0)
expected_der = np.exp(1.0) * (np.cos(1.0) + np.sin(1.0))
assert abs(f.val - expected_val) < 1e-12, '[TC02] Chain rule sin*exp value FAILED'
assert abs(f.der - expected_der) < 1e-12, '[TC02] Chain rule sin*exp derivative FAILED'

# ---- TC03: HyperDualScalar 二阶算术运算 ----
a = HyperDualScalar(2.0, 1.0, 0.0, 0.0)
b = HyperDualScalar(3.0, 0.0, 1.0, 0.0)
c = a * b
assert abs(c.f0 - 6.0) < 1e-12, '[TC03] HyperDual f0 FAILED'
assert abs(c.f1 - 3.0) < 1e-12, '[TC03] HyperDual f1 FAILED'
assert abs(c.f2 - 2.0) < 1e-12, '[TC03] HyperDual f2 FAILED'
assert abs(c.f12 - 1.0) < 1e-12, '[TC03] HyperDual f12 FAILED'

# ---- TC04: grad_scalar_func_ad 梯度精度 ----
def _f_grad(vars):
    x0, y0 = vars[0], vars[1]
    from autodiff_core import dual_sin, dual_exp
    return dual_sin(x0) * dual_exp(y0) + (x0 ** 2) * y0
x0 = np.array([1.0, 0.5])
grad = grad_scalar_func_ad(_f_grad, x0)
grad_exact = np.array([
    np.cos(x0[0]) * np.exp(x0[1]) + 2.0 * x0[0] * x0[1],
    np.sin(x0[0]) * np.exp(x0[1]) + x0[0] ** 2
])
assert np.max(np.abs(grad - grad_exact)) < 1e-10, '[TC04] Gradient accuracy FAILED'

# ---- TC05: directional_derivative_ad 方向导数 ----
direction = np.array([0.6, 0.8])
dd = directional_derivative_ad(_f_grad, x0, direction)
dd_exact = np.dot(grad_exact, direction)
assert abs(dd - dd_exact) < 1e-10, '[TC05] Directional derivative FAILED'

# ---- TC06: mixed_partial_hyperdual 混合偏导数 ----
def _f_hd(vars):
    from autodiff_core import hdual_sin, hdual_exp
    x0, y0 = vars[0], vars[1]
    return hdual_sin(x0) * hdual_exp(y0) + (x0 ** 2) * y0
mixed = mixed_partial_hyperdual(_f_hd, x0, 0, 1)
mixed_exact = np.cos(x0[0]) * np.exp(x0[1]) + 2.0 * x0[0]
assert abs(mixed - mixed_exact) < 1e-10, '[TC06] Mixed partial FAILED'

# ---- TC07: lennard_jones_potential 解析值验证 ----
pot0 = lennard_jones_potential(1.0, epsilon=1.0, sigma=1.0)
assert abs(pot0) < 1e-12, '[TC07] LJ at r=sigma should be 0 FAILED'
r_min = 2.0 ** (1.0 / 6.0)
pot_min = lennard_jones_potential(r_min, epsilon=2.0, sigma=1.5)
expected_min = -2.0  # minimum = -epsilon at r = sigma * 2^(1/6)
assert abs(pot_min - expected_min) < 1e-10, '[TC07] LJ potential minimum FAILED'

# ---- TC08: lennard_jones_force 与有限差分比较 ----
from potential_models import lennard_jones_force
r = 1.2
f_ana = lennard_jones_force(r, epsilon=1.0, sigma=1.0)
h = 1e-6
pot_plus = lennard_jones_potential(r + h)
pot_minus = lennard_jones_potential(r - h)
f_fd = -(pot_plus - pot_minus) / (2.0 * h)
assert abs(f_ana - f_fd) < 1e-5, '[TC08] LJ force vs finite difference FAILED'

# ---- TC09: gaussian_potential_2d 中心值及衰减 ----
val_center = gaussian_potential_2d(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, A=5.0, corr_matrix=np.eye(2))
assert abs(val_center - 5.0) < 1e-12, '[TC09] Gaussian at center FAILED'
val_far = gaussian_potential_2d(4.0, 0.0, 0.0, 0.0, 1.0, 1.0, A=1.0, corr_matrix=np.eye(2))
assert val_far < 1e-3, '[TC09] Gaussian decay at 4 sigma FAILED'

# ---- TC10: total_forces_lj 牛顿第三定律（合力为零） ----
from potential_models import total_forces_lj
np.random.seed(42)
pos = np.random.randn(5, 2) * 0.5
forces = total_forces_lj(pos, epsilon=1.0, sigma=1.0, rcut=3.0, box_size=10.0)
net_force = np.sum(forces, axis=0)
assert np.max(np.abs(net_force)) < 1e-12, '[TC10] Net force should be zero FAILED'
assert forces.shape == (5, 2), '[TC10] Force shape FAILED'

# ---- TC11: frobenius_number_2d 已知值 ----
g57 = frobenius_number_2d(5, 7)
assert g57 == 23, '[TC11] Frobenius g(5,7) should be 23 FAILED'
g35 = frobenius_number_2d(3, 5)
assert g35 == 7, '[TC11] Frobenius g(3,5) should be 7 FAILED'

# ---- TC12: diophantine_nd_nonnegative 解验证 ----
sols = diophantine_nd_nonnegative([3, 5, 7], 20)
assert len(sols) > 0, '[TC12] Should have at least one Diophantine solution FAILED'
for sol in sols:
    val = np.dot(sol, [3, 5, 7])
    assert val == 20, '[TC12] All solutions must satisfy the equation FAILED'

# ---- TC13: generate_hcp_lattice_2d 形状与间距 ----
hcp = generate_hcp_lattice_2d(3, 3, lattice_constant=1.2)
assert hcp.shape == (9, 2), '[TC13] HCP lattice shape FAILED'
d_nn = np.linalg.norm(hcp[0] - hcp[1])
assert abs(d_nn - 1.2) < 1e-10, '[TC13] HCP nearest neighbor distance FAILED'

# ---- TC14: coordination_number 已知值 ----
assert coordination_number("fcc") == 12, '[TC14] FCC coordination FAILED'
assert coordination_number("bcc") == 8, '[TC14] BCC coordination FAILED'
assert coordination_number("hcp") == 12, '[TC14] HCP coordination FAILED'
assert coordination_number("sc") == 6, '[TC14] SC coordination FAILED'
assert coordination_number("diamond") == 4, '[TC14] Diamond coordination FAILED'
assert coordination_number("hexagonal") == 6, '[TC14] Hexagonal coordination FAILED'
assert coordination_number("square") == 4, '[TC14] Square coordination FAILED'

# ---- TC15: latin_hypercube_sampling 形状、范围与可复现性 ----
np.random.seed(42)
pts_lhs = latin_hypercube_sampling(3, 20)
assert pts_lhs.shape == (20, 3), '[TC15] LHS shape FAILED'
assert np.all(pts_lhs >= 0.0) and np.all(pts_lhs <= 1.0), '[TC15] LHS range FAILED'
np.random.seed(42)
pts_lhs2 = latin_hypercube_sampling(3, 20)
assert np.allclose(pts_lhs, pts_lhs2), '[TC15] LHS reproducibility FAILED'

# ---- TC16: latin_center_sampling 形状、范围与可复现性 ----
np.random.seed(42)
pts_lc = latin_center_sampling(2, 16)
assert pts_lc.shape == (16, 2), '[TC16] Latin Center shape FAILED'
assert np.all(pts_lc >= 0.0) and np.all(pts_lc <= 1.0), '[TC16] Latin Center range FAILED'
np.random.seed(42)
pts_lc2 = latin_center_sampling(2, 16)
assert np.allclose(pts_lc, pts_lc2), '[TC16] Latin Center reproducibility FAILED'

# ---- TC17: sobol_like_sampling 形状与范围 ----
pts_sobol = sobol_like_sampling(2, 16)
assert pts_sobol.shape == (16, 2), '[TC17] Sobol-like shape FAILED'
assert np.all(pts_sobol >= 0.0) and np.all(pts_sobol <= 1.0), '[TC17] Sobol-like range FAILED'

# ---- TC18: set_partition_equivalence 正确划分 ----
n8 = 8
R8 = np.zeros((n8, n8))
for i in range(n8):
    for j in range(n8):
        if (i % 3) == (j % 3):
            R8[i, j] = 1.0
classes = set_partition_equivalence(n8, R8)
assert len(classes) == 3, '[TC18] Partition should have 3 classes FAILED'
for cls in classes:
    mods = set(x % 3 for x in cls)
    assert len(mods) == 1, '[TC18] Elements in same class must share mod 3 FAILED'

# ---- TC19: triangle_grid_points 点数与重心坐标范围 ----
verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
tri_pts = triangle_grid_points(4, verts)
n_expected = (4 + 1) * (4 + 2) // 2
assert len(tri_pts) == n_expected, '[TC19] Triangle grid point count FAILED'
assert np.all(tri_pts[:, 0] >= -1e-12) and np.all(tri_pts[:, 1] >= -1e-12), '[TC19] Triangle positivity FAILED'
assert np.all(tri_pts[:, 0] + tri_pts[:, 1] <= 1.0 + 1e-12), '[TC19] Triangle bounds x+y<=1 FAILED'

# ---- TC20: chebyshev_nodes 边界、个数与单调性 ----
n20 = 10
nodes = chebyshev_nodes(-2.0, 2.0, n20)
assert len(nodes) == n20, '[TC20] Chebyshev node count FAILED'
assert abs(nodes[0] - 2.0) < 1e-12, '[TC20] First node should be upper bound FAILED'
assert abs(nodes[-1] + 2.0) < 1e-12, '[TC20] Last node should be lower bound FAILED'
assert np.all(np.diff(nodes) <= 1e-12), '[TC20] Nodes not in descending order FAILED'

# ---- TC21: divided_differences + newton_interpolate 二次多项式精确重建 ----
def _poly2(x):
    return 1.0 + 2.0 * x + 3.0 * x * x
n21 = 5
xd21 = chebyshev_nodes(-1.0, 1.0, n21)
yd21 = _poly2(xd21)
dd21 = divided_differences(xd21, yd21)
xp_test = np.array([-0.5, 0.0, 0.5])
yp_interp = newton_interpolate(xd21, dd21, xp_test)
yp_exact = _poly2(xp_test)
assert np.max(np.abs(yp_interp - yp_exact)) < 1e-10, '[TC21] Newton interpolation of quadratic FAILED'

# ---- TC22: chebyshev_differentiation_matrix sin(pi*x) 求导精度 ----
n22 = 36
nodes22 = chebyshev_nodes(-1.0, 1.0, n22)
D = chebyshev_differentiation_matrix(n22)
f_vals = np.sin(np.pi * nodes22)
df_num = D @ f_vals
df_exact22 = np.pi * np.cos(np.pi * nodes22)
err22 = np.max(np.abs(df_num - df_exact22))
assert err22 < 1e-8, '[TC22] Chebyshev spectral derivative accuracy FAILED'

# ---- TC23: cvt_1d_nonuniform_python 输出形状、范围与单调性 ----
np.random.seed(42)
cvt_pts = cvt_1d_nonuniform_python(n_generators=8, density_type=0, n_steps=30, n_samples_per_step=500)
assert len(cvt_pts) == 8, '[TC23] CVT output size FAILED'
assert np.all(cvt_pts >= 0.0) and np.all(cvt_pts <= 1.0), '[TC23] CVT range FAILED'
assert np.all(np.diff(cvt_pts) > 0), '[TC23] CVT points not monotonic FAILED'

# ---- TC24: solve_biharmonic_fd1d 固支边界条件验证 ----
def _load_uniform(x):
    return 1.0
x_bh, u_bh = solve_biharmonic_fd1d(_load_uniform, n=33, xlim=(-1.0, 1.0), bc_displacement=(0.0, 0.0), bc_slope=(0.0, 0.0))
assert abs(u_bh[0]) < 1e-8, '[TC24] Left displacement BC FAILED'
assert abs(u_bh[-1]) < 1e-8, '[TC24] Right displacement BC FAILED'
assert len(u_bh) == len(x_bh), '[TC24] Solution size mismatch FAILED'
assert np.max(u_bh) > 0, '[TC24] Deflection should be positive for uniform load FAILED'

# ---- TC25: compute_strain_energy 非负性 ----
h25 = 0.01
x25 = np.arange(0.0, 1.0, h25)
u25 = np.sin(np.pi * x25) * 0.1
se = compute_strain_energy(u25, h25, young_modulus=2.0, moment_of_inertia=1.0)
assert se >= -1e-12, '[TC25] Strain energy must be non-negative FAILED'

# ---- TC26: TetrahedralMesh 体积计算 ----
mesh = TetrahedralMesh.generate_uniform_box_mesh(nx=2, ny=2, nz=2, xlim=(0.0, 1.0), ylim=(0.0, 1.0), zlim=(0.0, 1.0))
assert mesh.n_nodes > 0, '[TC26] Mesh node count FAILED'
assert mesh.n_elements > 0, '[TC26] Mesh element count FAILED'
_, vol = mesh.integrate_over_mesh(lambda x, y, z: 1.0, order=1)
assert abs(vol - 1.0) < 1e-10, '[TC26] Unit cube volume FAILED'

# ---- TC27: heat_capacity_cv 正值性与 Dulong-Petit 参考 ----
np.random.seed(42)
e27 = np.random.randn(1000) * 0.05 + 5.0
cv, cv_dp = heat_capacity_cv(e27, temperature=1.0, n_particles=36, dim=3)
assert cv > 0, '[TC27] CV should be positive FAILED'
assert cv_dp > 0, '[TC27] Dulong-Petit should be positive FAILED'

# ---- TC28: entropy_from_energy_distribution 非负性 ----
np.random.seed(42)
e28 = np.random.randn(500) * 0.1 + 3.0
ent = entropy_from_energy_distribution(e28, temperature=1.0, n_bins=15)
assert ent >= -1e-12, '[TC28] Entropy must be non-negative FAILED'

# ---- TC29: Timer 计时器正耗时 ----
t29 = Timer()
t29.start()
import time
time.sleep(0.005)
t29.stop()
assert t29.elapsed > 0, '[TC29] Timer elapsed must be positive FAILED'

# ---- TC30: HistogramStats 概率和为一 ----
np.random.seed(42)
data30 = np.random.randn(500)
hist30 = HistogramStats(data30, n_bins=12)
probs30 = hist30.probabilities
assert abs(np.sum(probs30) - 1.0) < 1e-12, '[TC30] Histogram probabilities must sum to 1 FAILED'

# ---- TC31: relative_error 基本行为 ----
err0 = relative_error(1.0, 1.0)
assert abs(err0) < 1e-15, '[TC31] Relative error of identical values should be 0 FAILED'
err1 = relative_error(1.2, 1.0)
assert abs(err1 - 0.2) < 1e-12, '[TC31] Relative error 1.2 vs 1.0 FAILED'

# ---- TC32: convergence_rate 几何序列收敛阶 ----
errors = [0.5 ** k for k in range(1, 6)]
rates = convergence_rate(errors)
assert len(rates) >= 2, '[TC32] Should have at least 2 rate estimates FAILED'
assert abs(rates[0] - 1.0) < 1e-10, '[TC32] Geometric convergence should be order 1 FAILED'

# ---- TC33: virial_stress_lj 应力张量对称性 ----
np.random.seed(42)
pos33 = np.random.randn(10, 3) * 0.3
stress = virial_stress_lj(pos33, epsilon=1.0, sigma=1.0, rcut=3.0, volume=1000.0, box_size=20.0)
assert stress.shape == (3, 3), '[TC33] Stress tensor shape FAILED'
assert np.max(np.abs(stress - stress.T)) < 1e-12, '[TC33] Stress tensor not symmetric FAILED'

# ---- TC34: MDEngine 初始化与短模拟 ----
np.random.seed(42)
md34 = MDEngine(n_particles=16, dim=2, mass=1.0, dt=0.001, box_size=5.0,
                epsilon=1.0, sigma=1.0, rcut=2.5, temperature=0.5, tau_thermostat=0.05)
md34.initialize_positions_lattice("square")
md34.initialize_velocities_maxwell_boltzmann()
assert md34.pos.shape == (16, 2), '[TC34] Position shape FAILED'
assert md34.vel.shape == (16, 2), '[TC34] Velocity shape FAILED'
res34 = md34.run(n_steps=30, equilibration_steps=10, apply_thermostat=True)
assert 'total_energy' in res34, '[TC34] Results missing total_energy FAILED'
assert len(res34['total_energy']) == 30, '[TC34] Energy history length FAILED'
assert np.all(np.isfinite(res34['total_energy'])), '[TC34] Energy contains non-finite values FAILED'
assert np.all(res34['temperature'] > 0), '[TC34] Temperature must be positive FAILED'

# ---- TC35: thermal_expansion_estimate 正值 ----
L = np.array([1.0, 1.002, 1.004, 1.006])
T = np.array([290.0, 300.0, 310.0, 320.0])
alpha = thermal_expansion_estimate(L, T)
assert alpha > 0, '[TC35] Thermal expansion must be positive FAILED'

# ---- TC36: voronoi_cell_area_hex 正值与一致性 ----
area1 = voronoi_cell_area_hex(1.0)
area2 = voronoi_cell_area_hex(2.0)
assert area1 > 0, '[TC36] Voronoi area must be positive FAILED'
assert abs(area2 / area1 - 4.0) < 1e-10, '[TC36] Area should scale with a^2 FAILED'

# ---- TC37: generate_square_lattice_2d 形状与间距 ----
sq = generate_square_lattice_2d(4, 4, lattice_constant=1.5)
assert sq.shape == (16, 2), '[TC37] Square lattice shape FAILED'
d_sq = np.linalg.norm(sq[0] - sq[1])
assert abs(d_sq - 1.5) < 1e-10, '[TC37] Square lattice spacing FAILED'

# ---- TC38: boundary_trace_hex 闭合六方词追踪 ----
word38 = "123456"
trace38 = boundary_trace_hex(word38)
assert len(trace38) > 0, '[TC38] Boundary trace empty FAILED'
# 六方词 "123456" 应回到原点附近
dist_to_origin = np.linalg.norm(trace38[-1] - trace38[0])
assert dist_to_origin < 1e-10, '[TC38] Hex word 123456 should close FAILED'

# ---- TC39: pram_boundary_word 返回类型 ----
w_pram, p_pram = pram_boundary_word()
assert isinstance(w_pram, str), '[TC39] PRAM word should be string FAILED'
assert len(w_pram) > 0, '[TC39] PRAM word non-empty FAILED'
assert isinstance(p_pram, np.ndarray), '[TC39] PRAM origin should be ndarray FAILED'

# ---- TC40: CVTOptimizer 2D 优化 ----
np.random.seed(42)
cvt40 = CVTOptimizer(dim=2, n_generators=9, domain=(np.array([0., 0.]), np.array([2., 2.])),
                     density_func=lambda x: 1.0, max_iter=20, tol=1e-6)
cvt40.initialize_generators("grid")
pts40 = cvt40.optimize(sample_multiplier=30)
assert pts40.shape == (9, 2), '[TC40] CVT optimizer output shape FAILED'
assert np.all(pts40 >= -1e-10) and np.all(pts40 <= 2.0 + 1e-10), '[TC40] CVT points out of domain FAILED'
# 均匀密度下，重心应接近域中心
center40 = np.mean(pts40, axis=0)
assert abs(center40[0] - 1.0) < 0.5, '[TC40] CVT center x near 1.0 FAILED'
assert abs(center40[1] - 1.0) < 0.5, '[TC40] CVT center y near 1.0 FAILED'

# ---- TC41: boundary_range_hex 范围计算 ----
imin, imax, jmin, jmax = boundary_range_hex("123")
assert imax >= imin, '[TC41] i range invalid FAILED'
assert jmax >= jmin, '[TC41] j range invalid FAILED'

# ---- TC42: benchmark_function 基准测试返回结构 ----
def _f_bench(x):
    return x * x
bm = benchmark_function(_f_bench, 3.0, n_runs=5)
assert 'mean_time' in bm, '[TC42] Benchmark missing mean_time FAILED'
assert 'min_time' in bm, '[TC42] Benchmark missing min_time FAILED'
assert 'max_time' in bm, '[TC42] Benchmark missing max_time FAILED'
assert 'result' in bm, '[TC42] Benchmark missing result FAILED'
assert abs(bm['result'] - 9.0) < 1e-12, '[TC42] Benchmark result mismatch FAILED'

# ---- TC43: is_symmetric_positive_definite 正定检测 ----
A_spd = np.array([[4., 1.], [1., 3.]])
assert is_symmetric_positive_definite(A_spd), '[TC43] SPD matrix detection FAILED'
A_not_spd = np.array([[1., 3.], [3., 1.]])
assert not is_symmetric_positive_definite(A_not_spd), '[TC43] Non-SPD matrix should be rejected FAILED'

# ---- TC44: condition_number_estimate 基本行为 ----
A44 = np.diag([1.0, 2.0, 4.0])
cond44 = condition_number_estimate(A44)
assert cond44 > 0, '[TC44] Condition number must be positive FAILED'
assert abs(cond44 - 4.0) < 1e-10, '[TC44] Condition number of diag(1,2,4) FAILED'

# ---- TC45: elastic_constants_strain_derivative 返回矩阵正定性 ----
np.random.seed(42)
pos45 = np.random.randn(20, 3) * 0.3
vol45 = 100.0
C = elastic_constants_strain_derivative(pos45, epsilon=1.0, sigma=1.0, rcut=3.0, volume=vol45, strain_perturbation=1e-4)
assert C.shape == (3, 3), '[TC45] Elastic constants shape FAILED'
assert np.all(np.isfinite(C)), '[TC45] Elastic constants finite FAILED'
