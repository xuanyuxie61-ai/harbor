
# ---- TC01: CylindricalShellGeometry 参数化表面输出形状为 (2,2,3) ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
theta_grid = np.array([[0.0, np.pi/2], [np.pi, 3*np.pi/2]])
x_grid = np.array([[0.0, 0.0], [0.5, 0.5]])
surf = geom_test.parametric_surface(theta_grid, x_grid)
assert surf.shape == (2, 2, 3), '[TC01] 参数化表面输出形状 FAILED'

# ---- TC02: CylindricalShellGeometry 测地距离对称性 ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
p1 = np.array([0.25, 0.0, 0.0])
p2 = np.array([0.0, 0.25, 0.5])
d12 = geom_test.geodesic_distance(p1, p2)
d21 = geom_test.geodesic_distance(p2, p1)
assert abs(d12 - d21) < 1e-12, '[TC02] 测地距离对称性 FAILED'

# ---- TC03: CylindricalShellGeometry 第一基本形式解析验证 ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
E, F, G = geom_test.first_fundamental_form()
assert abs(E - 0.0625) < 1e-12 and F == 0.0 and G == 1.0, '[TC03] 第一基本形式 FAILED'

# ---- TC04: CylindricalShellGeometry 主曲率解析验证 ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
k1, k2 = geom_test.principal_curvatures()
assert abs(k1 - 4.0) < 1e-12 and k2 == 0.0, '[TC04] 主曲率 FAILED'

# ---- TC05: CylindricalShellGeometry 中曲面面积解析验证 ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
A = geom_test.surface_area()
assert abs(A - 2.0 * np.pi * 0.25 * 0.50) < 1e-12, '[TC05] 中曲面面积 FAILED'

# ---- TC06: CylindricalShellGeometry Batdorf 参数为正有限值 ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
Z = geom_test.batdorf_parameter(70e9, 0.33)
assert Z > 0.0 and np.isfinite(Z), '[TC06] Batdorf 参数范围 FAILED'

# ---- TC07: ShellTriMesh 节点数和单元数正确 ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
mesh_test = ShellTriMesh(4, 3, geom_test)
assert mesh_test.n_nodes == 12 and mesh_test.n_elem == 16, '[TC07] 节点数和单元数 FAILED'

# ---- TC08: ShellTriMesh 单元面积均为正 ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
mesh_test = ShellTriMesh(4, 3, geom_test)
areas = [mesh_test.element_area(eid) for eid in range(mesh_test.n_elem)]
assert all(a > 0.0 for a in areas), '[TC08] 单元面积均为正 FAILED'

# ---- TC09: ShellTriMesh alpha_measure 范围在 [0,1] ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
mesh_test = ShellTriMesh(4, 3, geom_test)
amin, aave, aarea = mesh_test.alpha_measure()
assert 0.0 <= amin <= 1.0 and 0.0 <= aave <= 1.0 and 0.0 <= aarea <= 1.0, '[TC09] alpha_measure 范围 FAILED'

# ---- TC10: ShellTriMesh 边界节点数量 ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
mesh_test = ShellTriMesh(4, 3, geom_test)
bottom, top = mesh_test.get_boundary_nodes()
assert len(bottom) == 4 and len(top) == 4, '[TC10] 边界节点数量 FAILED'

# ---- TC11: ShellMaterial 拉伸刚度解析验证 ----
mat_test = ShellMaterial(70e9, 0.33, 2700.0)
C = mat_test.extensional_rigidity(0.001)
expected_C = 70e9 * 0.001 / (1.0 - 0.33**2)
assert abs(C - expected_C) < 1e-3, '[TC11] 拉伸刚度 FAILED'

# ---- TC12: ShellMaterial 弯曲刚度解析验证 ----
mat_test = ShellMaterial(70e9, 0.33, 2700.0)
D = mat_test.bending_rigidity(0.001)
expected_D = 70e9 * 0.001**3 / (12.0 * (1.0 - 0.33**2))
assert abs(D - expected_D) < 1e-9, '[TC12] 弯曲刚度 FAILED'

# ---- TC13: ShellMaterial 膜矩阵对称正定 ----
mat_test = ShellMaterial(70e9, 0.33, 2700.0)
Cm = mat_test.membrane_matrix(0.001)
assert np.allclose(Cm, Cm.T) and np.all(np.linalg.eigvalsh(Cm) > 0), '[TC13] 膜矩阵对称正定 FAILED'

# ---- TC14: ShellFEModel 线性刚度矩阵尺寸正确 ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
mesh_test = ShellTriMesh(4, 3, geom_test)
mat_test = ShellMaterial(70e9, 0.33, 2700.0)
fem_test = ShellFEModel(mesh_test, mat_test)
K = fem_test.assemble_linear_stiffness()
assert K.shape == (36, 36), '[TC14] 线性刚度矩阵尺寸 FAILED'

# ---- TC15: ShellFEModel 零位移内力为零 ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
mesh_test = ShellTriMesh(4, 3, geom_test)
mat_test = ShellMaterial(70e9, 0.33, 2700.0)
fem_test = ShellFEModel(mesh_test, mat_test)
u0 = np.zeros(fem_test.n_dof)
fint = fem_test.internal_force(u0)
assert np.linalg.norm(fint) < 1e-12, '[TC15] 零位移内力 FAILED'

# ---- TC16: LinearBucklingAnalyzer 解析屈曲载荷解析验证 ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
mat_test = ShellMaterial(70e9, 0.33, 2700.0)
buckling = LinearBucklingAnalyzer(geom_test, mat_test)
Ncr = buckling.analytical_buckling_load()
expected_Ncr = 70e9 * 0.001**2 / (0.25 * np.sqrt(3.0 * (1.0 - 0.33**2)))
assert abs(Ncr - expected_Ncr) < 1e-3, '[TC16] 解析屈曲载荷 FAILED'

# ---- TC17: LinearBucklingAnalyzer 离散搜索返回有限值 ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
mat_test = ShellMaterial(70e9, 0.33, 2700.0)
buckling = LinearBucklingAnalyzer(geom_test, mat_test)
N_min, m_opt, n_opt, modes = buckling.buckling_modes_discrete(m_max=5, n_max=5)
assert np.isfinite(N_min) and m_opt >= 1 and n_opt >= 0, '[TC17] 离散搜索 FAILED'

# ---- TC18: LinearBucklingAnalyzer Koiter 零缺陷敏感性为 1 ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
mat_test = ShellMaterial(70e9, 0.33, 2700.0)
buckling = LinearBucklingAnalyzer(geom_test, mat_test)
ratio = buckling.imperfection_sensitivity_koiter(0.0, 5)
assert abs(ratio - 1.0) < 1e-12, '[TC18] Koiter 零缺陷敏感性 FAILED'

# ---- TC19: bessel_zero_halley J0 第一个零点接近 2.4048 ----
from linear_buckling import bessel_zero_halley
z1 = bessel_zero_halley(0.0, 1, kind=1)
assert abs(z1 - 2.404825557685772) < 1e-10, '[TC19] Bessel J0 第一个零点 FAILED'

# ---- TC20: bessel_zeros_vector 返回单调递增向量 ----
zeros = bessel_zeros_vector(1.0, 4, kind=1)
assert len(zeros) == 4 and all(np.diff(zeros) > 0), '[TC20] Bessel 零点向量 FAILED'

# ---- TC21: DefectGenerator 单模态缺陷边界处为零 ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
defect_gen = DefectGenerator(geom_test, seed=42)
func = defect_gen.single_mode_defect(m=1, n=4, amplitude=0.001)
theta = np.array([0.0, np.pi/2, np.pi])
x = np.array([0.0, 0.25, 0.50])
w = func(theta, x)
assert abs(w[0]) < 1e-14, '[TC21] 单模态缺陷边界 FAILED'

# ---- TC22: DefectGenerator 缺陷统计返回正确键 ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
mesh_test = ShellTriMesh(4, 3, geom_test)
defect_gen = DefectGenerator(geom_test, seed=42)
func = defect_gen.single_mode_defect(m=1, n=2, amplitude=0.001)
stats = defect_gen.defect_statistics(mesh_test, func)
assert set(stats.keys()) == {'max_defect', 'rms_defect', 'defect_to_thickness', 'mean_defect'}, '[TC22] 缺陷统计键 FAILED'

# ---- TC23: sparse_matrix_io coo_to_mm 输出格式正确 ----
from scipy.sparse import coo_matrix
row = np.array([0, 1, 2])
col = np.array([1, 2, 0])
data = np.array([1.0, 2.0, 3.0])
coo = coo_matrix((data, (row, col)), shape=(3, 3))
mm_str = coo_to_mm(coo)
lines = mm_str.strip().splitlines()
assert lines[0].startswith('%%MatrixMarket') and '3 3 3' in lines[1], '[TC23] MM 格式输出 FAILED'

# ---- TC24: sparse_matrix_io coo_bandwidth 返回正数 ----
from scipy.sparse import coo_matrix
row = np.array([0, 1, 2, 2])
col = np.array([0, 1, 2, 0])
data = np.array([1.0, 2.0, 3.0, 4.0])
coo = coo_matrix((data, (row, col)), shape=(3, 3))
ml, mu, bw = coo_bandwidth(coo)
assert ml >= 0 and mu >= 0 and bw > 0, '[TC24] 带宽计算 FAILED'

# ---- TC25: sparse_matrix_io matrix_profile_reduction 返回排列 ----
from scipy.sparse import coo_matrix
row = np.array([0, 1, 2, 0])
col = np.array([0, 1, 2, 2])
data = np.array([1.0, 2.0, 3.0, 4.0])
coo = coo_matrix((data, (row, col)), shape=(3, 3))
perm = matrix_profile_reduction(coo)
assert len(perm) == 3 and set(perm) == {0, 1, 2}, '[TC25] RCM 排列 FAILED'

# ---- TC26: NewtonRaphsonSolver 零载荷立即收敛 ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
mesh_test = ShellTriMesh(4, 3, geom_test)
mat_test = ShellMaterial(70e9, 0.33, 2700.0)
fem_test = ShellFEModel(mesh_test, mat_test)
f_ext = np.zeros(fem_test.n_dof)
nr = NewtonRaphsonSolver(max_iter=5, tol_force=1e-6, tol_disp=1e-8)
res = nr.solve(fem_test, f_ext, lambda_load=0.0)
assert res['converged'] and res['iterations'] <= 1, '[TC26] Newton 零载荷收敛 FAILED'

# ---- TC27: StabilityAnalyzer Koiter 分岔分类对称稳定 ----
stab = StabilityAnalyzer(None)
path = [{'lambda': 0.0, 'max_disp': 0.0}, {'lambda': 0.1, 'max_disp': 1.0}, {'lambda': 0.2, 'max_disp': 2.0}]
cls = stab.koiter_bifurcation_class(path)
assert cls == 'symmetric-stable', '[TC27] Koiter 分岔分类 FAILED'

# ---- TC28: StabilityAnalyzer Lyapunov 短路径返回零 ----
stab = StabilityAnalyzer(None)
path = [{'lambda': 0.0, 'max_disp': 0.0}, {'lambda': 0.1, 'max_disp': 1.0}]
lyap = stab.lyapunov_exponent_discrete(path)
assert lyap == 0.0, '[TC28] Lyapunov 短路径 FAILED'

# ---- TC29: ArcLengthTracker Chirikov 短路径返回空数组 ----
tracker = ArcLengthTracker()
path_short = [{'lambda': 0.0, 'max_disp': 0.0}, {'lambda': 0.1, 'max_disp': 1.0}]
ind = tracker.chirikov_stability_indicator(path_short)
assert len(ind) == 0, '[TC29] Chirikov 短路径 FAILED'

# ---- TC30: ArcLengthTracker _compute_psi 返回非负有限值 ----
tracker = ArcLengthTracker()
psi = tracker._compute_psi(np.array([1.0, 2.0]), 0.5, 10.0)
assert psi >= 0.0 and np.isfinite(psi), '[TC30] psi 非负有限 FAILED'

# ---- TC31: DefectGenerator 蒙特卡洛多模态固定种子可复现 ----
np.random.seed(42)
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
defect_gen1 = DefectGenerator(geom_test, seed=42)
func1 = defect_gen1.monte_carlo_multi_mode(n_modes=3, amplitude_ratio=0.01)
np.random.seed(42)
geom_test2 = CylindricalShellGeometry(0.25, 0.50, 0.001)
defect_gen2 = DefectGenerator(geom_test2, seed=42)
func2 = defect_gen2.monte_carlo_multi_mode(n_modes=3, amplitude_ratio=0.01)
theta = np.array([0.5, 1.0])
x = np.array([0.1, 0.2])
w1 = func1(theta, x)
w2 = func2(theta, x)
assert np.allclose(w1, w2), '[TC31] MC 缺陷可复现性 FAILED'

# ---- TC32: ShellTriMesh 所有单元内角和为 pi ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
mesh_test = ShellTriMesh(4, 3, geom_test)
sum_errors = sum(abs(np.sum(mesh_test.element_angles(eid)) - np.pi) for eid in range(mesh_test.n_elem))
assert sum_errors < 1e-10, '[TC32] 单元内角和 FAILED'

# ---- TC33: PseudoTimeSolver 返回结果包含必要键 ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
mesh_test = ShellTriMesh(4, 3, geom_test)
mat_test = ShellMaterial(70e9, 0.33, 2700.0)
fem_test = ShellFEModel(mesh_test, mat_test)
f_ext = np.zeros(fem_test.n_dof)
pt = PseudoTimeSolver(damping_ratio=0.9, dt=0.01, max_steps=10)
res = pt.solve(fem_test, f_ext, lambda_load=0.0)
assert set(res.keys()) == {'disp', 'converged', 'steps', 'energy_history', 'residual_norm'}, '[TC33] 伪时间求解器返回键 FAILED'

# ---- TC34: CylindricalShellGeometry 边界排序返回正确长度 ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
mesh_test = ShellTriMesh(4, 3, geom_test)
sorted_idx = geom_test.boundary_sort(mesh_test.nodes)
assert len(sorted_idx) == 8, '[TC34] 边界排序长度 FAILED'

# ---- TC35: ShellFEModel 零位移几何刚度对称 ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
mesh_test = ShellTriMesh(4, 3, geom_test)
mat_test = ShellMaterial(70e9, 0.33, 2700.0)
fem_test = ShellFEModel(mesh_test, mat_test)
u0 = np.zeros(fem_test.n_dof)
Kg = fem_test.assemble_geometric_stiffness(u0)
Kg_dense = Kg.toarray()
assert np.allclose(Kg_dense, Kg_dense.T), '[TC35] 几何刚度对称性 FAILED'

# ---- TC36: LinearBucklingAnalyzer Bessel 验证返回单调零点 ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
mat_test = ShellMaterial(70e9, 0.33, 2700.0)
buckling = LinearBucklingAnalyzer(geom_test, mat_test)
zeros = buckling.bessel_verification(n_circumferential=3, n_zeros=5)
assert len(zeros) == 5 and all(z > 0 for z in zeros), '[TC36] Bessel 验证 FAILED'

# ---- TC37: ShellMaterial 弯曲矩阵对称正定 ----
mat_test = ShellMaterial(70e9, 0.33, 2700.0)
Cb = mat_test.bending_matrix(0.001)
assert np.allclose(Cb, Cb.T) and np.all(np.linalg.eigvalsh(Cb) > 0), '[TC37] 弯曲矩阵对称正定 FAILED'

# ---- TC38: CylindricalShellGeometry 测地距离同一点为零 ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
p = np.array([0.25, 0.0, 0.25])
d = geom_test.geodesic_distance(p, p)
assert abs(d) < 1e-12, '[TC38] 测地距离同一点 FAILED'

# ---- TC39: ArcLengthTracker track_path 返回结构正确 ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
mesh_test = ShellTriMesh(4, 3, geom_test)
mat_test = ShellMaterial(70e9, 0.33, 2700.0)
fem_test = ShellFEModel(mesh_test, mat_test)
f_ext = np.zeros(fem_test.n_dof)
tracker = ArcLengthTracker(initial_arc_length=0.01, min_arc_length=1e-5, max_arc_length=0.3, adaptivity=0.5)
result = tracker.track_path(fem_test, f_ext, n_steps=2, lambda_max=1.5)
assert set(result.keys()) == {'path', 'bifurcation_points', 'n_steps'}, '[TC39] 路径跟踪返回键 FAILED'

# ---- TC40: StabilityAnalyzer 能量势垒非负 ----
stab = StabilityAnalyzer(None)
path = [{'lambda': 0.0, 'max_disp': 0.0}, {'lambda': 0.2, 'max_disp': 2.0}, {'lambda': 0.1, 'max_disp': 1.0}]
eb = stab.energy_barrier(path)
assert eb >= 0.0, '[TC40] 能量势垒非负 FAILED'
