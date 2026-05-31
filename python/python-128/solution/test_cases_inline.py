
# ---- TC01: mesh_engine 四面体体积公式正确性 ----
nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
elements = np.array([[0, 1, 2, 3]])
mesh = TetrahedralMesh(nodes, elements)
vol = mesh.element_volume(0)
assert abs(vol - 1.0/6.0) < 1e-12, '[TC01] 四面体体积公式正确性 FAILED'

# ---- TC02: mesh_engine 均匀盒子网格体积 ----
mesh2 = generate_uniform_box_mesh(xlim=(0.0, 2.0), ylim=(0.0, 3.0), zlim=(0.0, 4.0), nx=3, ny=3, nz=3)
assert abs(mesh2.total_volume() - 24.0) < 0.1, '[TC02] 均匀盒子网格体积 FAILED'

# ---- TC03: mesh_engine 网格细化后体积守恒 ----
mesh3 = generate_uniform_box_mesh(xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), zlim=(-1.0, 1.0), nx=3, ny=3, nz=3)
vol_before = mesh3.total_volume()
mesh_ref = mesh3.refine_uniform()
vol_after = mesh_ref.total_volume()
assert abs(vol_after - vol_before) < 1e-9, '[TC03] 网格细化后体积守恒 FAILED'

# ---- TC04: mesh_engine 边界面提取 ----
bfaces = mesh_ref.compute_boundary_faces()
assert bfaces.shape[0] > 0, '[TC04] 边界面提取 FAILED'
assert bfaces.shape[1] == 3, '[TC04] 边界面提取 FAILED'

# ---- TC05: mesh_engine Gmsh 格式导出字符串非空 ----
gmsh_str = gmsh_format_string(mesh_ref)
assert len(gmsh_str) > 0, '[TC05] Gmsh 格式导出字符串非空 FAILED'
assert '$Nodes' in gmsh_str, '[TC05] Gmsh 格式导出字符串非空 FAILED'
assert '$Elements' in gmsh_str, '[TC05] Gmsh 格式导出字符串非空 FAILED'

# ---- TC06: chemotaxis_solver 初始条件与总质量 ----
solver = ChemotaxisSolver(nx=8, ny=8, nz=4, xlim=(0.0, 1.0), ylim=(0.0, 1.0), zlim=(0.0, 1.0), D=0.01)
solver.set_initial_condition(lambda x, y, z: 2.0)
mass = solver.total_mass()
expected_mass = 2.0 * solver.nx * solver.ny * solver.nz * solver.dx * solver.dy * solver.dz
assert abs(mass - expected_mass) < 1e-12, '[TC06] 初始条件与总质量 FAILED'

# ---- TC07: chemotaxis_solver 零速度场步进正性 ----
solver2 = ChemotaxisSolver(nx=8, ny=8, nz=4, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), zlim=(-0.5, 0.5), D=0.1)
solver2.set_initial_condition(lambda x, y, z: np.exp(-x**2 - y**2))
for _ in range(3):
    solver2.step(0.0, 0.0, 0.0, dt=0.01)
assert np.all(solver2.c >= -1e-12), '[TC07] 零速度场步进正性 FAILED'

# ---- TC08: chemotaxis_solver 常数场梯度为零 ----
solver3 = ChemotaxisSolver(nx=8, ny=8, nz=4, xlim=(0.0, 1.0), ylim=(0.0, 1.0), zlim=(0.0, 1.0), D=0.01)
solver3.set_initial_condition(lambda x, y, z: 5.0)
gx, gy, gz = solver3.gradient()
assert np.allclose(gx, 0.0), '[TC08] 常数场梯度为零 FAILED'
assert np.allclose(gy, 0.0), '[TC08] 常数场梯度为零 FAILED'
assert np.allclose(gz, 0.0), '[TC08] 常数场梯度为零 FAILED'

# ---- TC09: cell_dynamics 椭球体积解析公式 ----
v = ellipsoid_volume(2.0, 3.0, 4.0)
assert abs(v - (4.0/3.0)*np.pi*2.0*3.0*4.0) < 1e-12, '[TC09] 椭球体积解析公式 FAILED'

# ---- TC10: cell_dynamics 球体表面积 Rudolf 近似 ----
s = ellipsoid_surface_area_rudolf(1.0, 1.0, 1.0)
assert abs(s - 4.0*np.pi) < 0.01, '[TC10] 球体表面积 Rudolf 近似 FAILED'

# ---- TC11: cell_dynamics 椭球表面积椭圆积分公式 ----
from cell_dynamics import ellipsoid_surface_area_elliptic
s_ellip = ellipsoid_surface_area_elliptic(1.0, 1.0, 1.0)
assert abs(s_ellip - 4.0*np.pi) < 0.02, '[TC11] 椭球表面积椭圆积分公式 FAILED'

# ---- TC12: cell_dynamics CellAgent chemotaxis_velocity 零梯度 ----
from cell_dynamics import CellAgent
cell = CellAgent(np.array([0.0, 0.0, 0.0]))
v_cell = cell.chemotaxis_velocity(np.array([0.0, 0.0, 0.0]))
assert np.allclose(v_cell, 0.0), '[TC12] CellAgent chemotaxis_velocity 零梯度 FAILED'

# ---- TC13: cell_dynamics CellAgent 饱和迁移速度上界 ----
cell2 = CellAgent(np.array([0.0, 0.0, 0.0]))
grad = np.array([100.0, 0.0, 0.0])
v_sat = cell2.chemotaxis_velocity(grad, mu=1.0, gamma=0.5)
assert np.linalg.norm(v_sat) <= 2.0, '[TC13] CellAgent 饱和迁移速度上界 FAILED'

# ---- TC14: cell_dynamics CellPopulation 均值在域内 ----
np.random.seed(42)
pop = CellPopulation(n_cells=10, domain=((-1, 1), (-1, 1), (-0.5, 0.5)))
mean_pos = pop.compute_mean_position()
assert np.all(mean_pos >= np.array([-1.0, -1.0, -0.5])), '[TC14] CellPopulation 均值在域内 FAILED'
assert np.all(mean_pos <= np.array([1.0, 1.0, 0.5])), '[TC14] CellPopulation 均值在域内 FAILED'

# ---- TC15: cell_dynamics CellPopulation spread 非负 ----
spread = pop.compute_spread()
assert spread >= 0.0, '[TC15] CellPopulation spread 非负 FAILED'

# ---- TC16: monte_carlo_sampler cholesky_upper 单位矩阵 ----
from monte_carlo_sampler import cholesky_upper
U = cholesky_upper(np.eye(3))
assert np.allclose(U, np.eye(3)), '[TC16] cholesky_upper 单位矩阵 FAILED'

# ---- TC17: monte_carlo_sampler ellipsoid_volume_mc 球体体积 ----
A = np.eye(3)
vol_mc = ellipsoid_volume_mc(A, 1.0, 3)
assert abs(vol_mc - 4.0*np.pi/3.0) < 1e-9, '[TC17] ellipsoid_volume_mc 球体体积 FAILED'

# ---- TC18: monte_carlo_sampler solve_upper_triangular ----
from monte_carlo_sampler import solve_upper_triangular
U2 = np.array([[2.0, 1.0], [0.0, 3.0]])
b = np.array([4.0, 6.0])
x_sol = solve_upper_triangular(U2, b)
expected = np.array([1.0, 2.0])
assert np.allclose(x_sol, expected), '[TC18] solve_upper_triangular FAILED'

# ---- TC19: quadrature_rules prism_rule_order 权重和等于棱柱体积 (p=0,1,2,4) ----
from quadrature_rules import prism_rule_order
for p in [0, 1, 2, 4]:
    x, y, z, w = prism_rule_order(p)
    assert abs(np.sum(w) - 0.5) < 1e-12, '[TC19] prism_rule_order 权重和 (p=%d) FAILED' % p

# ---- TC20: quadrature_rules integrate_over_prism 常数函数 ----
from quadrature_rules import integrate_over_prism
val = integrate_over_prism(lambda x, y, z: 3.0, p=4)
assert abs(val - 1.5) < 1e-12, '[TC20] integrate_over_prism 常数函数 FAILED'

# ---- TC21: rom_analysis compute_pod_basis 简单矩阵能量 ----
from rom_analysis import compute_pod_basis
A2 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
U3, sigma, Vt, L, energy = compute_pod_basis(A2, energy_threshold=0.99)
assert L >= 1, '[TC21] compute_pod_basis 简单矩阵能量 FAILED'
assert energy[-1] >= 0.99, '[TC21] compute_pod_basis 简单矩阵能量 FAILED'

# ---- TC22: rom_analysis ChemotaxisROM 构造与重构一致性 ----
snapshots = [np.ones((4, 4, 4)) * float(i) for i in range(1, 4)]
rom = ChemotaxisROM()
rom.build_basis(snapshots, energy_threshold=0.99)
coeff = rom.project(snapshots[0])
recon = rom.reconstruct(coeff)
assert recon.shape == (4, 4, 4), '[TC22] ChemotaxisROM 重构形状 FAILED'

# ---- TC23: rom_analysis ChemotaxisROM relative_error 自洽 ----
snap_r = np.random.RandomState(42).rand(4, 4, 4)
rom2 = ChemotaxisROM()
rom2.build_basis([snap_r, snap_r*2, snap_r*3], energy_threshold=0.99)
err_r = rom2.relative_error(snap_r)
assert err_r >= 0.0, '[TC23] ChemotaxisROM relative_error 非负 FAILED'
assert err_r < 1.0, '[TC23] ChemotaxisROM relative_error 自洽 FAILED'

# ---- TC24: adaptive_grid trig_interpolant 数据节点精确恢复 ----
from adaptive_grid import trig_interpolant
xd = np.linspace(0.0, 2.0*np.pi, 8, endpoint=False)
yd = np.sin(xd)
xi = xd[2]
yi = trig_interpolant(xd, yd, xi)
assert abs(yi - yd[2]) < 1e-12, '[TC24] trig_interpolant 数据节点精确恢复 FAILED'

# ---- TC25: adaptive_grid trig_interpolant 正弦函数重构 ----
xd2 = np.linspace(0.0, 2.0*np.pi, 16, endpoint=False)
yd2 = np.sin(xd2)
xi2 = np.linspace(0.0, 2.0*np.pi, 100)
yi2 = trig_interpolant(xd2, yd2, xi2)
err_t = np.max(np.abs(yi2 - np.sin(xi2)))
assert err_t < 0.1, '[TC25] trig_interpolant 正弦函数重构 FAILED'

# ---- TC26: adaptive_grid AdaptiveChemotaxisSampler 采样点在域内 ----
np.random.seed(42)
adaptive = AdaptiveChemotaxisSampler(lambda pt: np.exp(-pt[0]**2 - pt[1]**2), domain=((-1, 1), (-1, 1)))
pts = adaptive.sample_adaptive(n_points=8, n_iter=5)
assert pts.shape == (8, 2), '[TC26] AdaptiveChemotaxisSampler 采样点形状 FAILED'
assert np.all(pts[:, 0] >= -1.0) and np.all(pts[:, 0] <= 1.0), '[TC26] AdaptiveChemotaxisSampler 采样点 x 范围 FAILED'
assert np.all(pts[:, 1] >= -1.0) and np.all(pts[:, 1] <= 1.0), '[TC26] AdaptiveChemotaxisSampler 采样点 y 范围 FAILED'

# ---- TC27: cell_cycle advance_cell_cycle 归一化保持 ----
phase = np.array([0.5, 0.2, 0.2, 0.1])
new_phase = advance_cell_cycle(phase, dt=0.1)
assert abs(new_phase.sum() - 1.0) < 1e-12, '[TC27] advance_cell_cycle 归一化保持 FAILED'

# ---- TC28: cell_cycle caesar_cycle_shift 循环 4 次回到原值 ----
phase2 = np.array([0.1, 0.2, 0.3, 0.4])
phase_temp = phase2.copy()
for _ in range(4):
    phase_temp = caesar_cycle_shift(phase_temp, k=1)
assert np.allclose(phase_temp, np.array([0.1, 0.2, 0.3, 0.4])), '[TC28] caesar_cycle_shift 循环 4 次 FAILED'

# ---- TC29: cell_cycle population_weighted_chemotaxis_sensitivity 范围 ----
phase3 = np.array([0.25, 0.25, 0.25, 0.25])
w = population_weighted_chemotaxis_sensitivity(phase3, w_max=1.0, w_min=0.1)
assert 0.1 <= w <= 1.0, '[TC29] population_weighted_chemotaxis_sensitivity 范围 FAILED'

# ---- TC30: cell_cycle chemotaxis_sensitivity_by_phase 值域范围 ----
from cell_cycle import chemotaxis_sensitivity_by_phase
w_g2 = chemotaxis_sensitivity_by_phase(2, w_max=1.0, w_min=0.1)
w_g1 = chemotaxis_sensitivity_by_phase(0, w_max=1.0, w_min=0.1)
assert abs(w_g2 - 1.0) < 1e-10, '[TC30] chemotaxis_sensitivity_by_phase G2=w_max FAILED'
assert abs(w_g1 - 0.1) < 1e-10, '[TC30] chemotaxis_sensitivity_by_phase G1=w_min FAILED'

# ---- TC31: special_math jacobi_elliptic 恒等式 sn²+cn²=1 ----
sn, cn, dn = jacobi_elliptic(0.8, 0.5)
assert abs(sn**2 + cn**2 - 1.0) < 1e-10, '[TC31] jacobi_elliptic 恒等式 FAILED'

# ---- TC32: special_math jacobi_elliptic m=0 退化到三角函数 ----
sn_m0, cn_m0, dn_m0 = jacobi_elliptic(0.5, 0.0)
assert abs(sn_m0 - np.sin(0.5)) < 1e-10, '[TC32] jacobi_elliptic m=0 退化到三角函数 FAILED'
assert abs(cn_m0 - np.cos(0.5)) < 1e-10, '[TC32] jacobi_elliptic m=0 退化到三角函数 FAILED'
assert abs(dn_m0 - 1.0) < 1e-10, '[TC32] jacobi_elliptic m=0 退化到三角函数 FAILED'

# ---- TC33: special_math cordic_sin_cos π/2 ----
c, s = cordic_sin_cos(np.pi/2.0, n=40)
assert abs(s - 1.0) < 1e-10, '[TC33] cordic_sin_cos π/2 FAILED'
assert abs(c - 0.0) < 1e-10, '[TC33] cordic_sin_cos π/2 FAILED'

# ---- TC34: special_math cordic_sin_cos π/6 ----
c2, s2 = cordic_sin_cos(np.pi/6.0, n=30)
assert abs(s2 - 0.5) < 1e-4, '[TC34] cordic_sin_cos π/6 FAILED'
assert abs(c2 - np.sqrt(3.0)/2.0) < 1e-4, '[TC34] cordic_sin_cos π/6 FAILED'

# ---- TC35: special_math tridiag_solve 单位三对角系统 ----
n = 5
a_t = np.zeros(n)
b_t = np.ones(n)
c_t = np.zeros(n)
f_t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
x_t = tridiag_solve(a_t, b_t, c_t, f_t)
assert np.allclose(x_t, f_t), '[TC35] tridiag_solve 单位三对角系统 FAILED'

# ---- TC36: special_math tridiag_solve 已知解析解 ----
n2 = 20
a_t2 = np.full(n2, -1.0, dtype=float)
b_t2 = np.full(n2, 2.0, dtype=float)
c_t2 = np.full(n2, -1.0, dtype=float)
f_t2 = np.ones(n2, dtype=float)
x_t2 = tridiag_solve(a_t2, b_t2, c_t2, f_t2)
res = np.zeros(n2)
res[0] = b_t2[0]*x_t2[0] + c_t2[0]*x_t2[1] - f_t2[0]
for i in range(1, n2-1):
    res[i] = a_t2[i]*x_t2[i-1] + b_t2[i]*x_t2[i] + c_t2[i]*x_t2[i+1] - f_t2[i]
res[-1] = a_t2[-1]*x_t2[-2] + b_t2[-1]*x_t2[-1] - f_t2[-1]
assert np.max(np.abs(res)) < 1e-10, '[TC36] tridiag_solve 已知解析解 FAILED'

# ---- TC37: special_math tridiag_solve_multi 多右端项 ----
from special_math import tridiag_solve_multi
n3 = 6
a_t3 = np.zeros(n3)
b_t3 = np.ones(n3)
c_t3 = np.zeros(n3)
F_t3 = np.column_stack([np.arange(1, n3+1, dtype=float), np.arange(2, n3+2, dtype=float)])
X_t3 = tridiag_solve_multi(a_t3, b_t3, c_t3, F_t3)
assert np.allclose(X_t3, F_t3), '[TC37] tridiag_solve_multi 多右端项 FAILED'

# ---- TC38: mesh_engine 1-based 索引自动检测 ----
nodes2 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
elements2 = np.array([[1, 2, 3, 4]])
mesh1b = TetrahedralMesh(nodes2, elements2)
assert mesh1b.elements[0, 0] == 0, '[TC38] 1-based 索引自动检测 FAILED'

# ---- TC39: mesh_engine 质心计算 ----
nodes_c = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
elements_c = np.array([[0, 1, 2, 3]])
mesh_c = TetrahedralMesh(nodes_c, elements_c)
centroids = mesh_c.compute_centroids()
assert np.allclose(centroids[0], np.array([0.5, 0.5, 0.5])), '[TC39] 质心计算 FAILED'

# ---- TC40: cell_dynamics CellAgent step 确定性更新 ----
np.random.seed(42)
cell3 = CellAgent(np.array([1.0, 2.0, 3.0]))
pos_before = cell3.position.copy()
cell3.step(np.array([0.1, 0.0, 0.0]), dt=0.1, ecm_density_func=None, mu=1.0, gamma=0.0, sigma=0.0)
assert not np.allclose(cell3.position, pos_before), '[TC40] CellAgent step 确定性更新 FAILED'

# ---- TC41: cell_dynamics 敏感性分析返回非负偏差 ----
np.random.seed(42)
pop_s = CellPopulation(n_cells=5, domain=((-1, 1), (-1, 1), (-0.5, 0.5)))
sens = pop_s.sensitivity_analysis(lambda p: np.zeros(3), dt=0.1, n_steps=5, eps=0.01)
assert np.all(sens >= 0.0), '[TC41] 敏感性分析返回非负偏差 FAILED'

# ---- TC42: cell_cycle cycle_transition_matrix 性质 ----
from cell_cycle import cycle_transition_matrix
P1 = cycle_transition_matrix(k=1)
assert P1.shape == (4, 4), '[TC42] cycle_transition_matrix 形状 FAILED'
assert np.allclose(P1.sum(axis=0), 1.0), '[TC42] cycle_transition_matrix 列和 FAILED'
assert np.allclose(P1.sum(axis=1), 1.0), '[TC42] cycle_transition_matrix 行和 FAILED'
