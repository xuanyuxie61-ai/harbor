from utils import se2_exp, se2_log, is_positive_semidefinite, nearest_positive_semidefinite, mahalanobis_distance, robust_loss, format_matrix_latex

# ---- TC01: normalize_angle 应保持已归一化的角度不变 ----
assert abs(normalize_angle(1.5) - 1.5) < 1e-10, '[TC01] normalize_angle 已归一化角度不变 FAILED'

# ---- TC02: normalize_angle 应正确折叠超出范围的角度 ----
a = normalize_angle(np.pi + 0.5)
assert abs(a - (-np.pi + 0.5)) < 1e-10, '[TC02] normalize_angle 折叠角度 FAILED'

# ---- TC03: se2_exp 零向量应返回单位矩阵 ----
T_zero = se2_exp(np.array([0.0, 0.0, 0.0]))
assert np.allclose(T_zero, np.eye(3)), '[TC03] se2_exp 零向量->单位矩阵 FAILED'

# ---- TC04: se2_log 与 se2_exp 互逆 ----
import numpy as np
np.random.seed(42)
v_test = np.array([1.5, -0.8, 0.6])
T_test = se2_exp(v_test)
v_back = se2_log(T_test)
assert np.allclose(v_back, v_test, atol=1e-8), '[TC04] se2_exp/se2_log 互逆 FAILED'

# ---- TC05: compute_trajectory_ate 相同轨迹返回0 ----
import numpy as np
np.random.seed(42)
traj = np.random.randn(10, 3)
ate_same = compute_trajectory_ate(traj, traj)
assert abs(ate_same) < 1e-12, '[TC05] ATE 相同轨迹应为0 FAILED'

# ---- TC06: compute_trajectory_ate 已知偏移 ----
import numpy as np
gt = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
est = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0]])
ate = compute_trajectory_ate(est, gt)
assert abs(ate - np.sqrt(2.0)) < 1e-10, '[TC06] ATE 已知偏移 FAILED'

# ---- TC07: robust_loss 在阈值内应为 0.5*r^2 ----
r = robust_loss(0.5, huber_delta=1.0)
assert abs(r - 0.125) < 1e-12, '[TC07] robust_loss 阈值内 0.5*r^2 FAILED'

# ---- TC08: robust_loss 超出阈值应为线性 ----
r = robust_loss(2.0, huber_delta=1.0)
assert abs(r - 1.5) < 1e-12, '[TC08] robust_loss 超出阈值线性 FAILED'

# ---- TC09: is_positive_semidefinite 对单位矩阵返回True ----
assert is_positive_semidefinite(np.eye(4)), '[TC09] 单位矩阵应为半正定 FAILED'

# ---- TC10: is_positive_semidefinite 对负特征值矩阵返回False ----
M_bad = np.array([[1.0, 3.0], [3.0, 1.0]])
assert not is_positive_semidefinite(M_bad), '[TC10] 负特征值矩阵不应为半正定 FAILED'

# ---- TC11: nearest_positive_semidefinite 输出应为半正定 ----
import numpy as np
np.random.seed(42)
M_input = np.random.randn(4, 4)
M_input = M_input @ M_input.T
M_input[0, 0] = -1.0  # 破坏正定性
M_fixed = nearest_positive_semidefinite(M_input)
assert is_positive_semidefinite(M_fixed), '[TC11] nearest_psd 输出应为半正定 FAILED'

# ---- TC12: mahalanobis_distance 零向量自身距离为0 ----
mu = np.array([1.0, 2.0])
Sigma = np.eye(2)
d = mahalanobis_distance(mu, mu, Sigma)
assert abs(d) < 1e-12, '[TC12] 马氏距离自身应为0 FAILED'

# ---- TC13: chi2_confidence_interval 已知近似值 ----
chi2_val = chi2_confidence_interval(3, confidence=0.95)
assert chi2_val > 0.0, '[TC13] chi2 阈值应为正 FAILED'

# ---- TC14: DifferentialDriveRobot motion_model 直线运动 ----
import numpy as np
robot = DifferentialDriveRobot(x=0.0, y=0.0, theta=0.0, dt=1.0)
new_pose = robot.motion_model(v=1.0, w=0.0)
assert abs(new_pose[0] - 1.0) < 1e-10, '[TC14] 直线运动 x 应为1 FAILED'
assert abs(new_pose[1]) < 1e-10, '[TC14] 直线运动 y 应为0 FAILED'

# ---- TC15: DifferentialDriveRobot motion_model 旋转运动 ----
import numpy as np
robot2 = DifferentialDriveRobot(x=0.0, y=0.0, theta=0.0, dt=1.0)
new_pose2 = robot2.motion_model(v=0.0, w=np.pi/2)
assert abs(new_pose2[2] - np.pi/2) < 1e-6, '[TC15] 旋转运动 theta FAILED'

# ---- TC16: DifferentialDriveRobot propagate 固定种子可复现 ----
import numpy as np
np.random.seed(42)
robot3 = DifferentialDriveRobot(x=1.0, y=2.0, theta=0.5, sigma_v=0.05, sigma_w=0.02, dt=0.2)
p1, c1 = robot3.propagate(0.5, 0.1)
np.random.seed(42)
robot4 = DifferentialDriveRobot(x=1.0, y=2.0, theta=0.5, sigma_v=0.05, sigma_w=0.02, dt=0.2)
p2, c2 = robot4.propagate(0.5, 0.1)
assert np.allclose(p1, p2), '[TC16] propagate 固定种子可复现 FAILED'

# ---- TC17: DifferentialDriveRobot relative_transform 往返为零 ----
import numpy as np
robot5 = DifferentialDriveRobot(x=1.0, y=2.0, theta=0.3)
pose_target = np.array([2.0, 3.0, 0.8])
rel = robot5.relative_transform(pose_target)
# 将相对变换作用于当前位姿得到目标位姿
r_temp = DifferentialDriveRobot()
r_temp.pose = robot5.pose.copy()
r_temp2 = DifferentialDriveRobot()
r_temp2.pose = r_temp.motion_model(0.0, 0.0)  # dummy
Ti = r_temp.se2_to_matrix()
T_rel = np.array([
    [np.cos(rel[2]), -np.sin(rel[2]), rel[0]],
    [np.sin(rel[2]), np.cos(rel[2]), rel[1]],
    [0, 0, 1]
])
Tj = Ti @ T_rel
pose_recovered = np.array([Tj[0, 2], Tj[1, 2], np.arctan2(Tj[1, 0], Tj[0, 0])])
assert np.allclose(pose_recovered, pose_target, atol=1e-8), '[TC17] relative_transform 往返 FAILED'

# ---- TC18: Lidar2D _ray_circle_intersection 有交点 ----
dist = Lidar2D._ray_circle_intersection(0.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0)
assert dist is not None, '[TC18] 射线-圆应有交点 FAILED'
assert abs(dist - 2.0) < 1e-8, '[TC18] 射线-圆交点距离应为2 FAILED'

# ---- TC19: Lidar2D _ray_circle_intersection 无交点 ----
dist2 = Lidar2D._ray_circle_intersection(0.0, 0.0, 1.0, 0.0, 0.0, 3.0, 1.0)
assert dist2 is None, '[TC19] 射线-圆方向远离应无交点 FAILED'

# ---- TC20: Lidar2D _ray_segment_intersection 有交点 ----
dist3 = Lidar2D._ray_segment_intersection(0.0, 0.0, 1.0, 0.0, 2.0, -1.0, 2.0, 1.0)
assert dist3 is not None, '[TC20] 射线-线段应有交点 FAILED'
assert abs(dist3 - 2.0) < 1e-8, '[TC20] 射线-线段交点距离应为2 FAILED'

# ---- TC21: Lidar2D transform_points_to_local 与 to_world 互逆 ----
import numpy as np
np.random.seed(42)
pose_t = np.array([1.0, 2.0, 0.5])
pts_world = np.random.randn(5, 2)
pts_local = Lidar2D.transform_points_to_local(pts_world, pose_t)
pts_back = Lidar2D.transform_points_to_world(pts_local, pose_t)
assert np.allclose(pts_back, pts_world, atol=1e-8), '[TC21] 点云坐标变换互逆 FAILED'

# ---- TC22: PointCloudRegistration icp_2d 相同点云返回单位变换 ----
import numpy as np
np.random.seed(42)
src = np.random.randn(20, 2)
reg = PointCloudRegistration(max_iterations=10, tolerance=1e-6)
R, t, err = reg.icp_2d(src, src.copy())
assert np.allclose(R, np.eye(2), atol=1e-6), '[TC22] ICP 相同点云 R=I FAILED'
assert np.allclose(t, np.zeros(2), atol=1e-6), '[TC22] ICP 相同点云 t=0 FAILED'

# ---- TC23: R8GDMatrix 构造与矩阵向量乘法 ----
import numpy as np
np.random.seed(42)
n = 5
offsets = np.array([0, 1, -1], dtype=np.int64)
ndiag = len(offsets)
values = np.random.rand(n, ndiag)
r8gd = R8GDMatrix(n, ndiag, offsets, values)
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = r8gd.mv(x)
assert y.shape == (5,), '[TC23] R8GD mv 输出形状 FAILED'
assert np.all(np.isfinite(y)), '[TC23] R8GD mv 输出应为有限值 FAILED'

# ---- TC24: R8GDMatrix to_dense 与 mv 一致 ----
import numpy as np
np.random.seed(42)
n2 = 4
offsets2 = np.array([0, 1, -1, 2], dtype=np.int64)
ndiag2 = len(offsets2)
values2 = np.random.rand(n2, ndiag2)
r8gd2 = R8GDMatrix(n2, ndiag2, offsets2, values2)
A_dense = r8gd2.to_dense()
x2 = np.array([0.5, -1.2, 3.0, 2.1])
y_sparse = r8gd2.mv(x2)
y_dense = A_dense @ x2
assert np.allclose(y_sparse, y_dense, atol=1e-10), '[TC24] to_dense 与 mv 一致 FAILED'

# ---- TC25: R8GDMatrix mtv 与 mv 的转置关系 ----
import numpy as np
np.random.seed(42)
n3 = 4
offsets3 = np.array([0, 1, -1, 2], dtype=np.int64)
values3 = np.random.rand(n3, len(offsets3))
r8gd3 = R8GDMatrix(n3, len(offsets3), offsets3, values3)
x_a = np.array([1.0, 2.0, 3.0, 4.0])
x_b = np.array([5.0, 6.0, 7.0, 8.0])
mv_ab = np.dot(r8gd3.mv(x_a), x_b)
mtv_ba = np.dot(x_a, r8gd3.mtv(x_b))
assert abs(mv_ab - mtv_ba) < 1e-10, '[TC25] mv 与 mtv 转置关系 FAILED'

# ---- TC26: SORSolver solve 稠密矩阵求解 ----
import numpy as np
np.random.seed(42)
n_sor = 10
A_sor = np.random.randn(n_sor, n_sor) * 0.1
A_sor = A_sor @ A_sor.T + 3.0 * np.eye(n_sor)
x_true_sor = np.random.randn(n_sor)
b_sor = A_sor @ x_true_sor
sor = SORSolver(omega=1.5, max_iter=1000, tol=1e-10)
x_solved, res_norm, iters, conv = sor.solve(A_sor, b_sor)
assert conv, '[TC26] SOR 稠密求解应收敛 FAILED'
assert np.allclose(x_solved, x_true_sor, atol=1e-4), '[TC26] SOR 稠密解应接近真值 FAILED'

# ---- TC27: SORSolver solve_sparse R8GD矩阵求解 ----
import numpy as np
np.random.seed(42)
n_sp = 20
offsets_sp = np.array([0, 1, -1, 2, -2], dtype=np.int64)
ndiag_sp = len(offsets_sp)
values_sp = np.random.rand(n_sp, ndiag_sp) * 0.3
values_sp[:, 0] += 2.5
r8gd_sp = R8GDMatrix(n_sp, ndiag_sp, offsets_sp, values_sp)
x_true_sp = np.random.randn(n_sp)
b_sp = r8gd_sp.mv(x_true_sp)
sor_sp = SORSolver(omega=1.2, max_iter=2000, tol=1e-10)
x_solved_sp, res_sp, iters_sp, conv_sp = sor_sp.solve_sparse(r8gd_sp, b_sp)
assert conv_sp, '[TC27] SOR 稀疏求解应收敛 FAILED'
assert np.allclose(x_solved_sp, x_true_sp, atol=1e-4), '[TC27] SOR 稀疏解应接近真值 FAILED'

# ---- TC28: PoseGraph 添加顶点和边 ----
import numpy as np
graph = PoseGraph()
idx0 = graph.add_vertex(np.array([0.0, 0.0, 0.0]))
idx1 = graph.add_vertex(np.array([1.0, 0.0, 0.0]))
assert idx0 == 0 and idx1 == 1, '[TC28] PoseGraph 顶点索引 FAILED'
graph.add_edge(0, 1, np.array([1.0, 0.0, 0.0]), np.eye(3))
assert len(graph.edges) == 1, '[TC28] PoseGraph 边数 FAILED'

# ---- TC29: PoseGraph state_vector 往返 ----
import numpy as np
graph2 = PoseGraph()
np.random.seed(42)
for _ in range(4):
    graph2.add_vertex(np.random.randn(3))
vec = graph2.get_state_vector()
assert vec.shape == (12,), '[TC29] state_vector 维度 FAILED'
graph2.set_state_vector(vec)
vec2 = graph2.get_state_vector()
assert np.allclose(vec, vec2), '[TC29] state_vector 往返 FAILED'

# ---- TC30: GraphSLAMOptimizer _compute_error_and_jacobians ----
import numpy as np
xi = np.array([0.0, 0.0, 0.0])
xj = np.array([1.0, 0.0, 0.0])
z = np.array([1.0, 0.0, 0.0])
e, Ji, Jj = GraphSLAMOptimizer._compute_error_and_jacobians(xi, xj, z)
assert Ji.shape == (3, 3), '[TC30] Ji 形状 FAILED'
assert Jj.shape == (3, 3), '[TC30] Jj 形状 FAILED'
assert np.allclose(e, np.zeros(3), atol=1e-6), '[TC30] 无误差时e应接近0 FAILED'

# ---- TC31: ObservabilityAnalyzer analyze_hessian ----
import numpy as np
np.random.seed(42)
H_test = np.random.randn(12, 12)
H_test = H_test @ H_test.T + 0.1 * np.eye(12)
analysis = ObservabilityAnalyzer.analyze_hessian(H_test, 4)
assert 'condition_number' in analysis, '[TC31] 分析结果含 condition_number FAILED'
assert analysis['condition_number'] > 0, '[TC31] 条件数应为正 FAILED'
assert analysis['gauge_dofs'] == 3, '[TC31] gauge_dofs 应为3 FAILED'

# ---- TC32: ObservabilityAnalyzer generate_random_schur_matrix ----
import numpy as np
np.random.seed(42)
A, Q, T = ObservabilityAnalyzer.generate_random_schur_matrix(6)
assert A.shape == (6, 6), '[TC32] Schur 矩阵形状 FAILED'
assert Q.shape == (6, 6), '[TC32] Q 形状 FAILED'
assert T.shape == (6, 6), '[TC32] T 形状 FAILED'
# Q 应为正交矩阵
assert np.allclose(Q @ Q.T, np.eye(6), atol=1e-8), '[TC32] Q 应为正交 FAILED'

# ---- TC33: CVTKeyframeSampler fit 基本功能 ----
import numpy as np
np.random.seed(42)
points_cvt = np.random.randn(100, 2) * 2.0
cvt = CVTKeyframeSampler(num_generators=5, max_iter=20)
gens, labels = cvt.fit(points_cvt)
assert gens.shape == (5, 2), '[TC33] CVT 生成元形状 FAILED'
assert labels.shape == (100,), '[TC33] CVT 标签形状 FAILED'
assert len(cvt.energies) > 0, '[TC33] CVT 能量记录 FAILED'
# 能量应非增
for i in range(1, len(cvt.energies)):
    assert cvt.energies[i] <= cvt.energies[i-1] + 1e-8, '[TC33] CVT 能量应单调非增 FAILED'

# ---- TC34: OptimalStoppingKeyframeSelector select_keyframes ----
import numpy as np
np.random.seed(42)
gains = np.random.rand(30) * 10.0
selector = OptimalStoppingKeyframeSelector()
indices = selector.select_keyframes(gains)
assert len(indices) >= 1, '[TC34] 至少应选出一个关键帧 FAILED'
assert all(0 <= i < 30 for i in indices), '[TC34] 关键帧索引应在范围内 FAILED'

# ---- TC35: OptimalStoppingKeyframeSelector simulate_strategy 返回合理值 ----
import numpy as np
np.random.seed(42)
sel2 = OptimalStoppingKeyframeSelector()
emp, theo = sel2.simulate_strategy(deck_size=100, trial_num=300)
assert 0.2 < emp < 0.6, '[TC35] 模拟成功率应在合理范围 FAILED'
assert abs(theo - 1.0/np.e) < 1e-6, '[TC35] 理论极限应为 1/e FAILED'

# ---- TC36: InformationGainEstimator compute_fisher_information ----
import numpy as np
pose_info = np.array([0.0, 0.0, 0.0])
landmarks = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
gain = InformationGainEstimator.compute_fisher_information(pose_info, landmarks)
assert gain > 0, '[TC36] Fisher信息增益应为正 FAILED'
assert np.isfinite(gain), '[TC36] Fisher信息增益应为有限值 FAILED'

# ---- TC37: FEMUncertaintyField solve_uncertainty_field 基本求解 ----
import numpy as np
np.random.seed(42)
fem = FEMUncertaintyField(nx=8, ny=8)
u, xg, yg = fem.solve_uncertainty_field(
    domain=((-1.0, 1.0), (-1.0, 1.0)),
    a_func=lambda x, y: 1.0,
    c_func=lambda x, y: 0.1,
    f_func=lambda x, y: 1.0
)
assert u.shape == (8, 8), '[TC37] FEM 解形状 FAILED'
assert np.all(np.isfinite(u)), '[TC37] FEM 解应为有限值 FAILED'
assert np.min(u) >= -1e-10, '[TC37] FEM 解应非负（正源项+Dirichlet边界） FAILED'

# ---- TC38: FEMUncertaintyField sample_field_at_points ----
import numpy as np
np.random.seed(42)
fem2 = FEMUncertaintyField(nx=8, ny=8)
u2, xg2, yg2 = fem2.solve_uncertainty_field(
    domain=((-1.0, 1.0), (-1.0, 1.0)),
    a_func=lambda x, y: 1.0,
    c_func=lambda x, y: 0.1,
    f_func=lambda x, y: 1.0
)
query = np.array([[0.0, 0.0], [0.5, 0.5], [-0.5, -0.3]])
sampled = fem2.sample_field_at_points(u2, xg2, yg2, query)
assert sampled.shape == (3,), '[TC38] 采样值形状 FAILED'
assert np.all(np.isfinite(sampled)), '[TC38] 采样值应为有限值 FAILED'

# ---- TC39: AnnularCovarianceEstimator integrate_annular_covariance ----
import numpy as np
np.random.seed(42)
annular = AnnularCovarianceEstimator(nr=4, nt=8)
def cov_func(x, y):
    return 1.0
integral = annular.integrate_annular_covariance(center=(0.0, 0.0), r1=1.0, r2=2.0, covariance_func=cov_func)
assert integral > 0, '[TC39] 环形积分应为正 FAILED'

# ---- TC40: PermutationOrthogonalityChecker demonstrate_permutation_property ----
import numpy as np
np.random.seed(42)
dot = PermutationOrthogonalityChecker.demonstrate_permutation_property(n=20)
assert abs(dot) < 1e-8, '[TC40] 置换正交性点积应接近0 FAILED'

# ---- TC41: PermutationOrthogonalityChecker check_rotation_orthogonality 已知旋转 ----
import numpy as np
np.random.seed(42)
angle = 0.5
R_true = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
t_true = np.array([1.0, 0.5])
pts_a = np.random.randn(10, 2)
pts_b = pts_a @ R_true.T + t_true
corr = [(i, i) for i in range(10)]
is_valid, R_est, t_est, ortho_err = PermutationOrthogonalityChecker.check_rotation_orthogonality(
    pts_a, pts_b, corr
)
assert is_valid, '[TC41] 正交性检验应通过 FAILED'
assert np.allclose(R_est, R_true, atol=1e-6), '[TC41] 估计的旋转矩阵应接近真值 FAILED'

# ---- TC42: TriangularMeshMap from_point_cloud 基本功能 ----
import numpy as np
np.random.seed(42)
pts_mesh = np.random.rand(30, 2) * 5.0
mesh = TriangularMeshMap()
mesh.from_point_cloud(pts_mesh, max_points=50)
assert mesh.vertices.shape[0] >= 3, '[TC42] 网格顶点数应 >=3 FAILED'
assert mesh.triangles.shape[0] > 0, '[TC42] 三角面片数应 >0 FAILED'

# ---- TC43: TriangularMeshMap compute_mesh_quality ----
min_angle, max_ar = mesh.compute_mesh_quality()
assert 0.0 <= min_angle <= 60.0, '[TC43] 最小角度应在合理范围 FAILED'
assert max_ar >= 1.0, '[TC43] 最大长宽比应 >=1 FAILED'

# ---- TC44: TriangularMeshMap export_to_stl_like ----
verts_out, tris_out, norms_out = mesh.export_to_stl_like()
assert verts_out.shape == mesh.vertices.shape, '[TC44] 导出顶点形状一致 FAILED'
assert tris_out.shape == mesh.triangles.shape, '[TC44] 导出面片形状一致 FAILED'

# ---- TC45: scan_matching_similarity 返回值 ----
import numpy as np
np.random.seed(42)
s1 = np.random.randn(10, 2) * 0.1
s2 = np.random.randn(10, 2) * 0.1
sim = scan_matching_similarity(s1, s2)
assert sim > 0.0, '[TC45] 扫描匹配相似度应为正 FAILED'

# ---- TC46: format_matrix_latex 输出含预期子串 ----
M_test = np.array([[1.0, 2.0], [3.0, 4.0]])
latex_str = format_matrix_latex(M_test, name="A", precision=2)
assert 'begin{bmatrix}' in latex_str, '[TC46] LaTeX应含bmatrix FAILED'

# ---- TC47: DifferentialDriveRobot se2_to_matrix 正确性 ----
import numpy as np
robot_se2 = DifferentialDriveRobot(x=1.0, y=2.0, theta=np.pi/2)
T_se2 = robot_se2.se2_to_matrix()
assert abs(T_se2[0, 2] - 1.0) < 1e-10, '[TC47] SE(2)矩阵 x FAILED'
assert abs(T_se2[1, 2] - 2.0) < 1e-10, '[TC47] SE(2)矩阵 y FAILED'

# ---- TC48: DifferentialDriveRobot matrix_to_se2 静态方法 ----
T_mat = np.array([[0.0, -1.0, 3.0], [1.0, 0.0, 4.0], [0.0, 0.0, 1.0]])
pose_from_mat = DifferentialDriveRobot.matrix_to_se2(T_mat)
assert abs(pose_from_mat[0] - 3.0) < 1e-10, '[TC48] matrix_to_se2 x FAILED'
assert abs(pose_from_mat[2] - np.pi/2) < 1e-6, '[TC48] matrix_to_se2 theta FAILED'

# ---- TC49: 集成测试 - main() 返回预期键 ----
import numpy as np
np.random.seed(42)
results = main()
assert 'ate_odometry' in results, '[TC49] 缺少 ate_odometry FAILED'
assert 'ate_optimized' in results, '[TC49] 缺少 ate_optimized FAILED'
assert 'num_keyframes' in results, '[TC49] 缺少 num_keyframes FAILED'
assert 'num_closures' in results, '[TC49] 缺少 num_closures FAILED'
assert 'condition_number' in results, '[TC49] 缺少 condition_number FAILED'
assert 'mesh_vertices' in results, '[TC49] 缺少 mesh_vertices FAILED'
assert 'mesh_triangles' in results, '[TC49] 缺少 mesh_triangles FAILED'
assert 'fem_max_uncertainty' in results, '[TC49] 缺少 fem_max_uncertainty FAILED'
assert 'sor_relative_error' in results, '[TC49] 缺少 sor_relative_error FAILED'
assert results['ate_odometry'] >= 0, '[TC49] ATE 应为非负 FAILED'
assert results['num_keyframes'] >= 2, '[TC49] 关键帧数应 >=2 FAILED'

# ---- TC50: Lidar2D scan_environment 输出形状 ----
import numpy as np
np.random.seed(42)
obstacles_simple = [{'type': 'circle', 'center': (3.0, 0.0), 'radius': 0.5}]
lidar_test = Lidar2D(num_beams=36, max_range=10.0, fov=np.pi, sigma_range=0.0)
ranges, pts = lidar_test.scan_environment(np.array([0.0, 0.0, 0.0]), obstacles_simple)
assert ranges.shape == (36,), '[TC50] 测距值形状 FAILED'
assert pts.shape == (36, 2), '[TC50] 点云形状 FAILED'

# ---- TC51: generate_synthetic_environment 返回非空列表 ----
env = generate_synthetic_environment()
assert len(env) > 0, '[TC51] 合成环境应非空 FAILED'

# ---- TC52: generate_trajectory_commands 返回正确长度 ----
cmds = generate_trajectory_commands(num_steps=200)
assert len(cmds) == 200, '[TC52] 轨迹命令长度应为200 FAILED'
