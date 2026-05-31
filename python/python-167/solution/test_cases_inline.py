# ---- TC01: Timer.elapsed 返回非负浮点数 ----
import numpy as np
import time
timer = Timer()
t_elapsed = timer.elapsed()
assert isinstance(t_elapsed, float), '[TC01] Timer.elapsed 应返回 float FAILED'
assert t_elapsed >= 0.0, '[TC01] 经过时间应为非负数 FAILED'

# ---- TC02: check_numerical_singularity 对单位矩阵返回 False ----
A_eye = np.eye(5)
assert not check_numerical_singularity(A_eye), '[TC02] 单位矩阵不应被判为奇异 FAILED'

# ---- TC03: check_numerical_singularity 检测近奇异矩阵 ----
A_sing = np.array([[1.0, 2.0], [2.0, 4.0]])
assert check_numerical_singularity(A_sing), '[TC03] 近奇异矩阵应被检测到 FAILED'

# ---- TC04: safe_divide 正常除法 ----
from utils import safe_divide
assert abs(safe_divide(6.0, 3.0) - 2.0) < 1e-14, '[TC04] safe_divide(6,3) 应等于 2 FAILED'

# ---- TC05: safe_divide 除零返回 fallback ----
assert safe_divide(1.0, 0.0, fallback=42.0) == 42.0, '[TC05] 除零应返回 fallback 值 FAILED'

# ---- TC06: robust_sqrt 正值平方根 ----
from utils import robust_sqrt
assert abs(robust_sqrt(4.0) - 2.0) < 1e-14, '[TC06] robust_sqrt(4) 应等于 2 FAILED'

# ---- TC07: robust_sqrt 负值返回 sqrt(eps) ----
assert robust_sqrt(-1.0) >= 0.0, '[TC07] robust_sqrt 应返回非负值 FAILED'

# ---- TC08: clip_to_bounds 裁剪到边界 ----
from utils import clip_to_bounds
val = np.array([0.5, 1.5, -0.1])
lower = np.array([0.0, 0.0, 0.0])
upper = np.array([1.0, 1.0, 1.0])
clipped = clip_to_bounds(val, lower, upper)
assert np.all(clipped >= lower) and np.all(clipped <= upper), '[TC08] 裁剪值应在边界内 FAILED'

# ---- TC09: finite_difference_jacobian 对 f(x)=x^2 计算导数 ----
from utils import finite_difference_jacobian
def f_sq(x):
    return np.array([x[0]**2, x[0]**3])
J_num = finite_difference_jacobian(f_sq, np.array([2.0]))
assert abs(J_num[0, 0] - 4.0) < 1e-3, '[TC09] f(x)=x^2 在 x=2 处导数应≈4 FAILED'
assert abs(J_num[1, 0] - 12.0) < 1e-3, '[TC09] f(x)=x^3 在 x=2 处导数应≈12 FAILED'

# ---- TC10: householder_reflection 正交性 H^T·H = I ----
from utils import householder_reflection
v = np.array([1.0, 2.0, 3.0])
H = householder_reflection(v)
HT_H = H.T @ H
assert np.allclose(HT_H, np.eye(3), atol=1e-14), '[TC10] Householder 矩阵应为正交 FAILED'

# ---- TC11: gershgorin_discs 对已知矩阵输出正确形状 ----
A_g = np.array([[4.0, 1.0, 1.0], [1.0, 4.0, 1.0], [1.0, 1.0, 4.0]])
centers, radii = gershgorin_discs(A_g)
assert len(centers) == 3 and len(radii) == 3, '[TC11] Gershgorin 圆盘输出维度应为 3 FAILED'
assert np.all(centers == 4.0), '[TC11] 圆盘中心应为对角元 4.0 FAILED'

# ---- TC12: CholeskySolver 分解与求解精度 ----
A_chol = np.array([[4.0, 2.0, 1.0], [2.0, 5.0, 2.0], [1.0, 2.0, 3.0]])
b_chol = np.array([1.0, 2.0, 3.0])
chol = CholeskySolver(eps=1e-13)
L = chol.decompose(A_chol)
assert np.allclose(L @ L.T, A_chol, atol=1e-12), '[TC12] Cholesky L·L^T 应等于 A FAILED'
x_chol = chol.solve(A_chol, b_chol)
assert np.linalg.norm(A_chol @ x_chol - b_chol) < 1e-10, '[TC12] Cholesky 求解残差应接近 0 FAILED'

# ---- TC13: BlockTridiagonalSolver 求解块三对角系统 ----
block_solver = BlockTridiagonalSolver(block_size=3)
B0 = np.diag([2.0, 2.0, 2.0])
B1 = np.diag([2.5, 2.5, 2.5])
B2 = np.diag([3.0, 3.0, 3.0])
lower = [np.eye(3) * 0.1, np.eye(3) * 0.1]
diag = [B0, B1, B2]
upper = [np.eye(3) * 0.1, np.eye(3) * 0.1]
rhs_bt = [np.ones(3), np.ones(3)*2, np.ones(3)*3]
x_bt = block_solver.solve(lower, diag, upper, rhs_bt)
assert len(x_bt) == 3, '[TC13] 块三对角求解应返回 3 个解块 FAILED'
assert all(xi.shape == (3,) for xi in x_bt), '[TC13] 每个解块形状应为 (3,) FAILED'

# ---- TC14: Radix2FFT ifft(fft(x)) ≈ x 可复现性 ----
np.random.seed(42)
fft_solver = Radix2FFT()
x_fft = np.random.rand(64)
X = fft_solver.fft(x_fft)
x_recovered = fft_solver.ifft(X)
assert np.allclose(x_recovered.real, x_fft, atol=1e-12), '[TC14] FFT/IFFT 往返应恢复原信号 FAILED'

# ---- TC15: VandermondeSolver 多项式插值 p(x)=x^2+1 ----
vand = VandermondeSolver()
x_nodes = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
b_vand = np.array([2.0, 5.0, 10.0, 17.0, 26.0])
coeffs, info = vand.solve(x_nodes, b_vand)
assert info == 0, '[TC15] Vandermonde 求解应成功 (info=0) FAILED'
p_eval = vand.evaluate(x_nodes, coeffs, np.array([2.5]))
assert abs(p_eval[0] - 7.25) < 1e-6, '[TC15] p(2.5) 应等于 7.25 FAILED'

# ---- TC16: MatrixMultiplyBenchmark 输出形状正确 ----
np.random.seed(42)
A_mm = np.random.rand(64, 48)
B_mm = np.random.rand(48, 32)
C_mm = MatrixMultiplyBenchmark.multiply(A_mm, B_mm)
assert C_mm.shape == (64, 32), '[TC16] C=A·B 形状应为 (64, 32) FAILED'

# ---- TC17: SupportPolygon.contains_point 点在凸包内 ----
foot_positions = np.array([[0.15, 0.10], [0.18, -0.08], [-0.05, 0.12], [-0.10, -0.05]])
support = SupportPolygon(foot_positions)
com = np.array([0.05, 0.02])
assert support.contains_point(com), '[TC17] COM(0.05,0.02) 应在支撑多边形内 FAILED'

# ---- TC18: StabilityMargin.zmp_position 输出为有限值 ----
stab = StabilityMargin(robot_mass=5.0)
com_3d = np.array([0.05, 0.02, 0.12])
a_com = np.array([0.0, 0.0, 0.0])
zmp = stab.zmp_position(com_3d, a_com, np.zeros(3), np.zeros((6, 3)), np.ones((6, 2)))
assert np.all(np.isfinite(zmp)), '[TC18] ZMP 位置应为有限值 FAILED'
assert zmp.shape == (2,), '[TC18] ZMP 形状应为 (2,) FAILED'

# ---- TC19: SupportGraphCentrality.pagerank 收敛且和≈1 ----
np.random.seed(42)
graph = SupportGraphCentrality(n_legs=6, alpha=0.85)
stance_state = np.array([1, 1, 0, 1, 0, 1])
coupling = np.ones((6, 6)) * 0.5 + np.eye(6) * 1.0
M = graph.build_transition_matrix(stance_state, coupling)
pr = graph.pagerank(M)
assert abs(np.sum(pr) - 1.0) < 1e-6, '[TC19] PageRank 得分之和应≈1 FAILED'

# ---- TC20: LogisticMap 确定性序列（固定种子时可复现） ----
from chaotic_search import LogisticMap
np.random.seed(42)
lm = LogisticMap(r=4.0, x0=0.3)
seq = lm.generate(100)
assert len(seq) == 100, '[TC20] LogisticMap 应生成 100 个值 FAILED'
assert np.all(seq >= 0.0) and np.all(seq <= 1.0), '[TC20] LogisticMap 输出应在 [0,1] 内 FAILED'

# ---- TC21: SerialLegKinematics 正运动学输出 4×4 齐次矩阵 ----
dh = np.array([[0.0, 0.0, 0.04, np.pi/2], [0.0, 0.0, 0.12, 0.0], [0.0, 0.0, 0.12, 0.0]])
limits = np.array([[-np.pi/4, np.pi/4], [-np.pi/6, np.pi/6], [-np.pi/3, np.pi/2*0.75]])
leg = SerialLegKinematics(dh, limits)
q_test = np.array([0.0, 0.0, 0.0])
T_ee, transforms = leg.forward_kinematics(q_test)
assert T_ee.shape == (4, 4), '[TC21] 正运动学输出应为 4×4 矩阵 FAILED'
assert abs(T_ee[3, 3] - 1.0) < 1e-14, '[TC21] 齐次矩阵右下角应为 1 FAILED'

# ---- TC22: SerialLegKinematics.jacobian 输出 3×3 矩阵 ----
J = leg.jacobian(q_test)
assert J.shape == (3, 3), '[TC22] Jacobian 应为 3×3 矩阵 FAILED'

# ---- TC23: JointLimitConstraint 测地距离为非负 ----
jlc = JointLimitConstraint(limits, safety_margin=0.05)
q_mid = np.array([0.0, 0.0, 0.0])
dist = jlc.geodesic_distance_to_limit(q_mid)
assert np.all(dist >= 0.0), '[TC23] 到限位的测地距离应为非负 FAILED'

# ---- TC24: FootContactGeometry 摩擦锥残留计算 ----
foot = FootContactGeometry(mu=0.8)
f_test = np.array([0.0, 0.0, 20.0])
n_test = np.array([0.0, 0.0, 1.0])
residual = foot.friction_cone_residual(f_test, n_test)
assert residual < 0, '[TC24] 纯法向力应在摩擦锥内（残留负值） FAILED'

# ---- TC25: CPGNetwork 相位和振幅提取 ----
cpg = CPGNetwork(n_osc=4, alpha=50.0, mu=1.0, omega=2.0*np.pi*1.0, coupling_strength=5.0)
state = np.array([1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0])
phase = cpg.extract_phase(state)
amp = cpg.extract_amplitude(state)
assert len(phase) == 4, '[TC25] 相位输出长度应为 4 FAILED'
assert len(amp) == 4, '[TC25] 振幅输出长度应为 4 FAILED'
assert np.all(amp >= 0.0), '[TC25] 振幅应为非负 FAILED'

# ---- TC26: CPGNetwork.rhs 输出维度正确 ----
state0 = np.zeros(8)
state0[0] = 1.0
rhs_out = cpg.rhs(0.0, state0)
assert rhs_out.shape == (8,), '[TC26] CPG rhs 输出应为 8 维 FAILED'

# ---- TC27: TSPBruteForce 对小规模点集求解 ----
candidates = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
tsp = TSPBruteForce()
best_perm, min_cost, avg_cost, max_cost = tsp.solve(candidates)
assert len(best_perm) == 3, '[TC27] TSP 最优排列应包含 3 个索引 FAILED'
assert min_cost > 0.0, '[TC27] 最短路径长度应为正值 FAILED'

# ---- TC28: PolynomialSwingTrajectory 边界条件满足 ----
traj = PolynomialSwingTrajectory()
T = 0.3
c = traj.fit_quintic(T=T, p_start=0.0, p_end=0.15, v_start=0.0, v_end=0.0, a_start=0.0, a_end=0.0)
p0, v0, a0 = traj.evaluate(c, 0.0)
pT, vT, aT = traj.evaluate(c, T)
assert abs(p0 - 0.0) < 1e-9, '[TC28] p(0) 应等于 0 FAILED'
assert abs(pT - 0.15) < 1e-9, '[TC28] p(T) 应等于 0.15 FAILED'
assert abs(v0 - 0.0) < 1e-9, '[TC28] v(0) 应等于 0 FAILED'
assert abs(vT - 0.0) < 1e-9, '[TC28] v(T) 应等于 0 FAILED'

# ---- TC29: FootfallPlanner 落点排序与轨迹生成 ----
np.random.seed(42)
cand_pts = np.array([[0.12, 0.08, 0.0], [0.15, 0.05, 0.0], [0.10, 0.10, 0.0]])
planner = FootfallPlanner(swing_height=0.05, swing_period=0.3)
perm, sorted_pts = planner.plan_footholds(cand_pts)
assert len(perm) == 3, '[TC29] 落点排列应包含 3 个索引 FAILED'
swing_traj = planner.generate_swing_trajectory(p_start=np.array([0.0,0.0,0.0]), p_end=np.array([0.15,0.05,0.0]), n_samples=10)
assert swing_traj.shape == (10, 3), '[TC29] 摆动轨迹形状应为 (10, 3) FAILED'

# ---- TC30: generate_sample_terrain 返回三角网格地形 ----
terrain = generate_sample_terrain()
assert len(terrain.vertices) > 0, '[TC30] 示例地形应有顶点 FAILED'
assert len(terrain.faces) > 0, '[TC30] 示例地形应有面片 FAILED'
assert hasattr(terrain, 'aabb_min') and hasattr(terrain, 'aabb_max'), '[TC30] 地形应有 AABB FAILED'

# ---- TC31: TriangulatedTerrain.query_height 对内部点返回有效高度 ----
z, normal, face_idx = terrain.query_height(0.0, 0.0)
assert isinstance(z, float), '[TC31] 查询高度应返回 float FAILED'
assert np.isfinite(z), '[TC31] 查询高度应为有限值 FAILED'

# ---- TC32: QuadrilateralTerrainPatch 双线性插值与法向量 ----
quad_nodes = np.array([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.1], [-0.5, 0.5, 0.1]])
quad = QuadrilateralTerrainPatch(quad_nodes)
p_mid, J_mid = quad.bilinear_interpolate(0.5, 0.5)
assert p_mid.shape == (3,), '[TC32] 插值点应为 3 维 FAILED'
assert J_mid.shape == (3, 2), '[TC32] Jacobian 形状应为 (3, 2) FAILED'
n_mid = quad.normal_at(0.5, 0.5)
assert abs(np.linalg.norm(n_mid) - 1.0) < 1e-9, '[TC32] 法向量应为单位向量 FAILED'

# ---- TC33: ChaoticSimulatedAnnealing 优化返回最优解 ----
from chaotic_search import ChaoticSimulatedAnnealing
np.random.seed(42)
csa = ChaoticSimulatedAnnealing(dim=2, bounds=np.array([[-5.0, 5.0], [-5.0, 5.0]]), T0=1.0, max_iter=100)
def objective(x):
    return x[0]**2 + x[1]**2
x_opt, f_opt = csa.optimize(objective)
assert np.all(np.isfinite(x_opt)), '[TC33] 优化解应为有限值 FAILED'
assert f_opt >= 0.0, '[TC33] f(x)=x^2+y^2 最优值应 ≥0 FAILED'

# ---- TC34: config_parser 解析默认机器人配置 ----
xml = build_default_robot_config()
config = parse_robot_config(xml)
links = extract_link_params(config)
assert len(links) > 0, '[TC34] 解析的 link 数量应 >0 FAILED'
assert 'base_link' in links, '[TC34] 应包含 base_link FAILED'
assert links['base_link']['mass'] > 0, '[TC34] base_link 质量应 >0 FAILED'

# ---- TC35: StanceSwingAutomaton 状态更新为 0/1 ----
np.random.seed(42)
ssa = StanceSwingAutomaton(n_legs=6)
phase = np.array([0.0, np.pi, -np.pi/2, np.pi/2, np.pi/4, -np.pi/4])
states = ssa.update(phase)
assert states.shape == (6,), '[TC35] 支撑状态应为 6 维 FAILED'
assert np.all((states == 0) | (states == 1)), '[TC35] 支撑状态应仅为 0 或 1 FAILED'

# ---- TC36: LegDynamics 正向动力学输出非 NaN ----
M_leg = np.diag([0.08, 0.12, 0.10])
C_leg = np.diag([0.5, 0.5, 0.5])
leg_dyn = LegDynamics(M_leg, C_leg, gravity=9.81)
q_leg = np.array([0.1, -0.1, 0.2])
dq_leg = np.array([0.0, 0.0, 0.0])
tau_leg = np.array([0.0, 0.0, 0.0])
f_contact = np.array([0.0, 0.0, 0.0])
J_leg = np.array([[-0.08, -0.12, -0.12], [0.15, 0.05, -0.03], [0.0, 0.12, 0.12]])
ddq = leg_dyn.dynamics(q_leg, dq_leg, tau_leg, f_contact, J_leg)
assert not np.any(np.isnan(ddq)), '[TC36] 正向动力学不应产生 NaN FAILED'

# ---- TC37: 集成测试 run_full_pipeline 无异常 ----
np.random.seed(42)
try:
    run_full_pipeline()
    pipeline_ok = True
except Exception as e:
    pipeline_ok = False
assert pipeline_ok, '[TC37] run_full_pipeline() 应无异常完成 FAILED'

# ---- TC38: BarnsleyFernIFS 采样输出在边界内 ----
from chaotic_search import BarnsleyFernIFS
np.random.seed(42)
ifs = BarnsleyFernIFS()
lower_b = np.array([0.4, 0.05, 0.02, 1.0, 0.5])
upper_b = np.array([1.5, 0.30, 0.10, 15.0, 5.0])
samples = ifs.sample(20, (lower_b, upper_b))
assert samples.shape == (20, 2), '[TC38] IFS 采样应为 (20, 2) FAILED'

# ---- TC39: SerialLegKinematics 逆运动学求解精度 ----
np.random.seed(42)
dh = np.array([[0.0, 0.0, 0.04, np.pi/2], [0.0, 0.0, 0.12, 0.0], [0.0, 0.0, 0.12, 0.0]])
limits = np.array([[-np.pi/4, np.pi/4], [-np.pi/6, np.pi/6], [-np.pi/3, np.pi/2*0.75]])
leg2 = SerialLegKinematics(dh, limits)
target = np.array([0.15, 0.05, -0.18])
q0 = np.array([0.1, -0.2, 0.3])
q_inv = leg2.inverse_kinematics_numerical(target, q0, max_iter=200)
T_check, _ = leg2.forward_kinematics(q_inv)
err_inv = np.linalg.norm(T_check[:3, 3] - target)
assert err_inv < 0.01, '[TC39] 逆运动学位置误差应小于 0.01 FAILED'

# ---- TC40: LinearStabilityConstraint COM 可行域 ----
foot_positions2 = np.array([[0.2, 0.1], [0.2, -0.1], [-0.1, 0.1], [-0.1, -0.1]])
support2 = SupportPolygon(foot_positions2)
lp = LinearStabilityConstraint()
A_cons, b_cons = lp.com_feasible_region(support2, margin=0.02)
assert A_cons.shape[1] == 2, '[TC40] 约束矩阵列数应为 2 FAILED'
com_test = np.array([0.0, 0.0])
assert np.all(A_cons @ com_test <= b_cons + 1e-9), '[TC40] COM(0,0) 应在可行域内 FAILED'
