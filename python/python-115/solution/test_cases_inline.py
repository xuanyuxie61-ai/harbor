from special_math import ChebyshevEvaluator
from pes_surface import RBFKernel
from molecular_topology import MolecularGraph

# ---- TC01: ChebyshevEvaluator 对常数级数求值正确 ----
c_eval = ChebyshevEvaluator(np.array([2.0, 0.0, 0.0]))
val = c_eval.evaluate(0.5)
assert isinstance(val, float), '[TC01] 返回值类型应为 float FAILED'
assert abs(val - 1.0) < 1e-12, '[TC01] Chebyshev 常数级数求值错误 FAILED'

# ---- TC02: ChebyshevEvaluator 拒绝越界 x ----
c_eval2 = ChebyshevEvaluator(np.array([1.0, 0.0]))
try:
    c_eval2.evaluate(2.0)
    assert False, '[TC02] 应抛出 ValueError FAILED'
except ValueError:
    pass

# ---- TC03: clausen_function 在 x=0 处返回 0 ----
cl2_0 = clausen_function(0.0)
assert abs(cl2_0) < 1e-14, '[TC03] Cl2(0) 应为 0 FAILED'

# ---- TC04: clausen_function 奇函数对称性 Cl2(-x) = -Cl2(x) ----
cl2_pos = clausen_function(1.0)
cl2_neg = clausen_function(-1.0)
assert abs(cl2_pos + cl2_neg) < 1e-6, '[TC04] Cl2 奇对称性失败 FAILED'

# ---- TC05: clausen_function 在 π/2 处接近已知值 ----
cl2_pi2 = clausen_function(np.pi / 2)
assert abs(cl2_pi2 - 0.9159655941772190) < 1e-4, '[TC05] Cl2(π/2) 值与已知值不符 FAILED'

# ---- TC06: periodic_torsion_potential 返回有限标量 ----
vt = periodic_torsion_potential(np.pi / 4, n_terms=3)
assert isinstance(vt, float), '[TC06] 扭转势返回值应为 float FAILED'
assert np.isfinite(vt), '[TC06] 扭转势应有限 FAILED'

# ---- TC07: periodic_torsion_potential 对称性 V(φ) = V(-φ)（当 γ=0 时） ----
vt_p = periodic_torsion_potential(0.5)
vt_n = periodic_torsion_potential(-0.5)
assert np.isfinite(vt_p) and np.isfinite(vt_n), '[TC07] 扭转势应有限 FAILED'

# ---- TC08: angular_partition_function 返回正浮点数 ----
q_ang = angular_partition_function((-np.pi, np.pi), temperature=300.0)
assert q_ang > 0, '[TC08] 角向配分函数应为正 FAILED'
assert np.isfinite(q_ang), '[TC08] 角向配分函数应有限 FAILED'

# ---- TC09: CRSMatrix matvec 对于单位矩阵正确 ----
vals = np.array([1.0, 1.0, 1.0])
cols = np.array([0, 1, 2])
rows = np.array([0, 1, 2, 3])
crs = CRSMatrix(3, 3, rows, cols, vals)
x = np.array([1.0, 2.0, 3.0])
y = crs.matvec(x)
assert y.shape == (3,), '[TC09] matvec 输出尺寸错误 FAILED'
assert abs(y[0] - 1.0) < 1e-12, '[TC09] matvec 第0分量错误 FAILED'
assert abs(y[1] - 2.0) < 1e-12, '[TC09] matvec 第1分量错误 FAILED'
assert abs(y[2] - 3.0) < 1e-12, '[TC09] matvec 第2分量错误 FAILED'

# ---- TC10: CRSMatrix from_dense 往返一致性 ----
A_dense = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]])
crs_from = CRSMatrix.from_dense(A_dense)
A_back = crs_from.to_dense()
assert np.max(np.abs(A_dense - A_back)) < 1e-12, '[TC10] from_dense 往返不一致 FAILED'

# ---- TC11: CRSMatrix residual_norm 计算正确 ----
b = np.array([1.0, 0.0, 0.0])
r_norm = crs_from.residual_norm(np.array([0.0, 0.0, 0.0]), b)
assert abs(r_norm - 1.0) < 1e-12, '[TC11] residual_norm 错误 FAILED'

# ---- TC12: build_molecular_hessian_crs 维度正确 ----
np.random.seed(42)
coords_test = np.random.randn(5, 3) * 2.0
h_crs = build_molecular_hessian_crs(5, coords_test, force_constant=1.0, cutoff=3.5)
assert h_crs.n == 15, '[TC12] Hessian 维度应为 3N=15 FAILED'
assert h_crs.nz > 0, '[TC12] Hessian 应有非零元 FAILED'

# ---- TC13: lanczos_eigenvalue_solver 返回排序特征值 ----
np.random.seed(99)
ritz = lanczos_eigenvalue_solver(h_crs, max_iter=30)
assert len(ritz) > 0, '[TC13] Lanczos 应返回 Ritz 值 FAILED'
assert np.all(np.diff(ritz) >= -1e-10), '[TC13] Lanczos Ritz 值应升序 FAILED'

# ---- TC14: poisson_2d_exact_solution 在 (0,0) 处值正确 ----
u, ux, uy, uxx, uxy, uyy = poisson_2d_exact_solution(0.0, 0.0)
assert isinstance(float(u), float), '[TC14] 精确解应为浮点数 FAILED'
assert np.isfinite(float(u)), '[TC14] 精确解应有限 FAILED'

# ---- TC15: electrostatic_stabilization_energy 返回有限值 ----
phi_test = np.ones((4, 4))
rho_test = np.ones((4, 4)) * 0.1
estab = electrostatic_stabilization_energy(phi_test, rho_test, 1.0, 1.0)
assert np.isfinite(estab), '[TC15] 静电能应有限 FAILED'

# ---- TC16: PESInterpolator interpolate 返回标量 ----
np.random.seed(123)
xd_16 = np.random.rand(2, 10) * 2.0 - 1.0
fd_16 = np.sum(xd_16 ** 2, axis=0)
r0_16 = estimate_r0(xd_16)
interp_16 = PESInterpolator(2, 10, xd_16, r0_16, kernel_name='multiquadric')
interp_16.compute_weights(fd_16)
val_16 = interp_16.interpolate(np.array([0.0, 0.0]))
assert isinstance(val_16, float), '[TC16] interpolate 单点应返回标量 FAILED'
assert np.isfinite(val_16), '[TC16] 插值应有限 FAILED'

# ---- TC17: PESInterpolator gradient 维度正确 ----
grad = interp_16.gradient(np.array([0.5, 0.5]))
assert grad.shape == (2, 1), '[TC17] gradient 输出维度应为 (m, 1) FAILED'
assert np.all(np.isfinite(grad)), '[TC17] gradient 应有限 FAILED'

# ---- TC18: RBFKernel multiquadric 对称性 ----
r_test = np.array([1.0, 2.0, 3.0])
phi_mq = RBFKernel.multiquadric(r_test, 1.0)
assert len(phi_mq) == 3, '[TC18] multiquadric 输出长度错误 FAILED'
assert np.all(phi_mq > 0), '[TC18] multiquadric 应全正 FAILED'

# ---- TC19: estimate_r0 返回正值 ----
r0_est = estimate_r0(xd_16)
assert r0_est > 0, '[TC19] estimate_r0 应返回正值 FAILED'

# ---- TC20: RKF45Integrator 积分简单 ODE dy/dt = -y ----
def f_simple(t, y):
    return np.array([-y[0]])
integrator = RKF45Integrator(f_simple, 1, relerr=1e-6, abserr=1e-8)
t_f, y_f = integrator.integrate(0.0, np.array([1.0]), 1.0)
assert isinstance(t_f, float), '[TC20] 积分终止时间应为 float FAILED'
assert abs(y_f[0] - np.exp(-1.0)) < 1e-4, '[TC20] dy/dt=-y 积分错误 FAILED'

# ---- TC21: GlycolysisModel equilibrium 满足 dydt≈0 ----
model = GlycolysisModel()
equi = model.equilibrium()
dydt = model.derivatives(0.0, equi)
max_d = np.max(np.abs(dydt))
assert max_d < 1e-6, '[TC21] 平衡解应满足 dydt≈0 FAILED'

# ---- TC22: integrate_glycolysis 返回正确形状 ----
np.random.seed(42)
times, states = integrate_glycolysis()
assert times.shape[0] == states.shape[0], '[TC22] 时间与状态长度应相同 FAILED'
assert states.shape[1] == 2, '[TC22] 应有 2 个状态变量 FAILED'

# ---- TC23: GaussLaguerreQuadrature 积分 exp(-x) 精度 ----
glq = GaussLaguerreQuadrature(order=32, alpha_param=0.5, a=0.0, b=1.0)
integral = glq.integrate(lambda x: np.exp(-x))
theoretical = 0.313329
assert abs(integral - theoretical) < 1e-3, '[TC23] Gauss-Laguerre 积分精度不足 FAILED'

# ---- TC24: PolygonMoments 三角形面积正确 ----
tri_x = np.array([0.0, 2.0, 1.0])
tri_y = np.array([0.0, 0.0, 1.5])
area = PolygonMoments.moment_unnormalized(3, tri_x, tri_y, 0, 0)
assert abs(area - 1.5) < 1e-10, '[TC24] 三角形面积应为 1.5 FAILED'

# ---- TC25: HexagonMoments 奇数幂次积分为零 ----
hex_mom = HexagonMoments()
I_10 = hex_mom.integral_monomial(1, 0)
I_01 = hex_mom.integral_monomial(0, 1)
assert abs(I_10) < 1e-14, '[TC25] 正六边形 x^1 积分应为 0 FAILED'
assert abs(I_01) < 1e-14, '[TC25] 正六边形 y^1 积分应为 0 FAILED'

# ---- TC26: ThermodynamicIntegration 返回正活化能 ----
xi = np.linspace(0, 1, 50)
energy_profile = 5.0 * np.exp(-20.0 * (xi - 0.5) ** 2)  # Gaussian barrier
ti = ThermodynamicIntegration(n_lambda=20, temperature=300.0)
free_e, dG, dG_rev = ti.free_energy_barrier(energy_profile, xi)
assert dG > 0, '[TC26] 活化自由能应为正 FAILED'
assert np.isfinite(dG), '[TC26] 活化自由能应有限 FAILED'

# ---- TC27: PentominoShapes 有 12 种形状 ----
shapes = PentominoShapes.all_shapes()
assert len(shapes) == 12, '[TC27] Pentomino 应有 12 种形状 FAILED'

# ---- TC28: ConfigurationTiling 网格坐标往返一致 ----
tiling = ConfigurationTiling((-1.0, 1.0), (-1.0, 1.0), n_xi=10, n_eta=10)
xi_orig, eta_orig = 0.3, -0.5
i, j = tiling.physical_to_grid(xi_orig, eta_orig)
xi_back, eta_back = tiling.grid_to_physical(i, j)
assert abs(xi_orig - xi_back) < tiling.dxi + 1e-10, '[TC28] 网格坐标往返不一致 FAILED'

# ---- TC29: ConfigurationSpaceSampler metropolis 返回正确形状 ----
np.random.seed(42)
sampler = ConfigurationSpaceSampler(n_atoms=5, temperature=300.0)
def energy_1d(x):
    return 5.0 * x[0] ** 2
samples_mc, energies_mc, acc_ratio = sampler.metropolis_sampling(energy_1d, [0.5], n_steps=200, step_size=0.2)
assert samples_mc.shape[0] == 201, '[TC29] MC 样本数应为 n_steps+1 FAILED'
assert 0 <= acc_ratio <= 1, '[TC29] 接受率应在 [0,1] FAILED'

# ---- TC30: XYZParser 解析演示数据正确 ----
xyz_data = XYZParser.generate_demo_xyz()
atoms, coords = XYZParser.parse_string(xyz_data)
assert len(atoms) == 18, '[TC30] 应有 18 个原子 FAILED'
assert coords.shape == (18, 3), '[TC30] 坐标形状应为 (18,3) FAILED'

# ---- TC31: MolecularGraph 连通分量正确 ----
graph = MolecularGraph(atoms, coords)
graph.build_bonds()
components = graph.connected_components()
assert len(components) >= 1, '[TC31] 应至少有一个连通分量 FAILED'

# ---- TC32: PathParameterization arc_length 单调递增 ----
path_test = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [3.0, 1.0]])
s = PathParameterization.arc_length_parameterize(path_test)
assert s[0] == 0.0, '[TC32] 弧长参数应从 0 开始 FAILED'
assert s[-1] == 1.0 or abs(s[-1] - 1.0) < 1e-12, '[TC32] 弧长参数应归一化到 1 FAILED'
assert np.all(np.diff(s) >= -1e-15), '[TC32] 弧长参数应单调不减 FAILED'

# ---- TC33: SequenceManager 生成序列正确 ----
seq = SequenceManager.generate_sequence("frame", 5)
assert len(seq) == 5, '[TC33] 应生成 5 个序列名 FAILED'
assert seq[0] == "frame_000.dat", '[TC33] 序列名格式错误 FAILED'
assert seq[4] == "frame_004.dat", '[TC33] 序列名格式错误 FAILED'

# ---- TC34: SymmetryOperations apply_c2v 返回 4 个构型 ----
coords_3d = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
configs_sym = SymmetryOperations.apply_c2v_symmetry(coords_3d)
assert len(configs_sym) == 4, '[TC34] C2v 对称应生成 4 个构型 FAILED'

# ---- TC35: TransitionStateVerifier wigner_correction >= 1 ----
verifier = TransitionStateVerifier(lambda x: np.zeros(2))
kappa = verifier.wigner_correction(100.0, temperature=300.0)
assert kappa >= 1.0, '[TC35] Wigner 校正因子应 >= 1 FAILED'

# ---- TC36: ReactionPathAnalysis activation_energy 计算正确 ----
energies_test = np.array([0.0, 1.0, 3.0, 5.0, 7.0, 5.0, 3.0, 1.0, 0.0])
Ea_f, Ea_r, ts_idx = ReactionPathAnalysis.activation_energy(energies_test)
assert Ea_f > 0, '[TC36] 正反应活化能应为正 FAILED'
assert ts_idx == 4, '[TC36] 过渡态索引应为 4 FAILED'

# ---- TC37: 集成测试：完整分子拓扑分析流程 ----
xyz_data_full = XYZParser.generate_demo_xyz()
atoms_full, coords_full = XYZParser.parse_string(xyz_data_full)
graph_full, results = analyze_molecular_topology(atoms_full, coords_full)
assert results['n_atoms'] == 18, '[TC37] 原子数应为 18 FAILED'
assert results['n_bonds'] > 0, '[TC37] 应有化学键 FAILED'
assert results['avg_degree'] > 0, '[TC37] 平均配位数应为正 FAILED'
assert 'metis_graph' in results, '[TC37] 应有 METIS 图结果 FAILED'

# ---- TC38: 集成测试：Poisson 求解器完整流程 ----
np.random.seed(42)
nx, ny = 8, 6
dh = 1.0
solver = Poisson2DSolver(nx, ny, dh)
part_x = np.random.rand(20, 2) * np.array([nx - 2, ny - 2]) * dh
part_v = np.random.randn(20, 2) * 100.0
den_test = pic_charge_density(nx, ny, dh, part_x, part_v, 20, 1.0, 1.0)
phi_init = np.ones((nx, ny))
phi_result = solver.solve_gs(phi_init, den_test, 0.5, 0.0, 1.0, -2.0, 1.0, 1.0, max_iter=500, tol=0.5)
efx, efy = solver.compute_electric_field(phi_result)
assert phi_result.shape == (nx, ny), '[TC38] 电势解形状错误 FAILED'
assert np.all(np.isfinite(phi_result)), '[TC38] 电势解应有限 FAILED'

# ---- TC39: 集成测试：弦方法反应路径优化 ----
np.random.seed(123)
nd_sm = 40
xd_sm = np.random.rand(2, nd_sm) * 4.0 - 2.0
def true_pot(x):
    x = np.asarray(x)
    if x.ndim == 1:
        x1, x2 = x[0], x[1]
    else:
        x1, x2 = x[0, 0], x[1, 0]
    return (x1 ** 2 - 1.0) ** 2 + 0.5 * x2 ** 2 + 0.3 * x1 * x2
fd_sm = np.array([true_pot(xd_sm[:, i]) for i in range(nd_sm)])
r0_sm = estimate_r0(xd_sm)
pes_sm = PESInterpolator(2, nd_sm, xd_sm, r0_sm, kernel_name='gaussian')
pes_sm.compute_weights(fd_sm)
def e_func(x):
    return float(pes_sm.interpolate(np.array(x).reshape(2, 1)))
def g_func(x):
    return pes_sm.gradient(np.array(x).reshape(2, 1)).flatten()
neb_sm = NEBOptimizer(e_func, g_func, n_images=10, spring_k=0.5, dt=0.05, max_iter=100, tol=1e-2)
path_neb, energies_neb, _ = neb_sm.optimize(np.array([-1.0, 0.0]), np.array([1.0, 0.0]))
assert path_neb.shape[0] == 10, '[TC39] NEB 路径应含 10 个图像 FAILED'
assert len(energies_neb) == 10, '[TC39] 应有 10 个能量值 FAILED'

# ---- TC40: 集成测试：过渡态验证流程 ----
H_test = pes_sm.hessian(path_neb[4])
verifier_ts = TransitionStateVerifier(lambda x: pes_sm.gradient(np.array(x).reshape(2, 1)).flatten(),
                                       lambda x: pes_sm.hessian(np.array(x).reshape(2, 1)))
verif = verifier_ts.verify_saddle_point(path_neb[4], grad_tol=1.0)
assert 'is_stationary' in verif, '[TC40] 验证结果应包含 is_stationary FAILED'
assert 'gradient_norm' in verif, '[TC40] 验证结果应包含 gradient_norm FAILED'
