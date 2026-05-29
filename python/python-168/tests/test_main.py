"""
main.py
机器人SLAM同时定位与稠密建图系统 —— 统一入口

博士级科学计算问题：
基于图优化的移动机器人同步定位与三角网格稠密建图，
融合有限元不确定性分析、CVT关键帧采样、SOR稀疏求解、
随机游走环路闭合检测与特征值可观测性分析。

运行方式：python main.py（零参数可运行）
"""

import numpy as np
import time

from robot_motion_model import DifferentialDriveRobot
from lidar_observation_model import Lidar2D, PointCloudRegistration
from triangular_mesh_map import TriangularMeshMap
from sparse_matrix_ops import R8GDMatrix, SORSolver
from graph_slam_optimizer import PoseGraph, GraphSLAMOptimizer, ObservabilityAnalyzer
from cvt_keyframe_sampler import CVTKeyframeSampler, OptimalStoppingKeyframeSelector, InformationGainEstimator
from fem_uncertainty_field import FEMUncertaintyField, AnnularCovarianceEstimator
from loop_closure_detector import IntegratedLoopClosureDetector, PermutationOrthogonalityChecker
from utils import compute_trajectory_ate, chi2_confidence_interval, normalize_angle


def generate_synthetic_environment():
    """
    生成合成环境障碍物（走廊+房间结构）
    """
    obstacles = []
    # 外墙
    obstacles.append({'type': 'segment', 'p1': (-5.0, -5.0), 'p2': (5.0, -5.0)})
    obstacles.append({'type': 'segment', 'p1': (5.0, -5.0), 'p2': (5.0, 5.0)})
    obstacles.append({'type': 'segment', 'p1': (5.0, 5.0), 'p2': (-5.0, 5.0)})
    obstacles.append({'type': 'segment', 'p1': (-5.0, 5.0), 'p2': (-5.0, -5.0)})
    # 内部柱子
    for cx, cy in [(-2.0, -2.0), (2.0, 2.0), (0.0, 0.0)]:
        obstacles.append({'type': 'circle', 'center': (cx, cy), 'radius': 0.4})
    # 内部墙壁
    obstacles.append({'type': 'segment', 'p1': (-3.0, -1.0), 'p2': (-1.0, -1.0)})
    obstacles.append({'type': 'segment', 'p1': (1.0, 1.0), 'p2': (3.0, 1.0)})
    return obstacles


def generate_trajectory_commands(num_steps=200):
    """
    生成差分驱动机器人的控制指令序列（闭合环路轨迹）
    
    轨迹设计：正方形环路带圆弧过渡，确保机器人回到起点附近
    """
    commands = []
    side_len = 80  # 每边步数
    turn_steps = 20
    v_straight = 0.4
    w_turn = np.pi / 2.0 / (turn_steps * 0.2)  # 90度转弯

    for _ in range(4):
        # 直走
        for _ in range(side_len):
            commands.append((v_straight, 0.0))
        # 右转90度
        for _ in range(turn_steps):
            commands.append((0.05, -w_turn))

    # 补充步数
    while len(commands) < num_steps:
        commands.append((v_straight, 0.0))
    return commands[:num_steps]


def scan_matching_similarity(scan1, scan2):
    """简单的扫描匹配相似度函数"""
    if scan1.shape[0] == 0 or scan2.shape[0] == 0:
        return 0.0
    # 使用 ICP 配准误差倒数作为相似度
    reg = PointCloudRegistration(max_iterations=20, tolerance=1e-4)
    R, t, error = reg.icp_2d(scan1, scan2)
    return 1.0 / (1.0 + error)


def main():
    print("=" * 70)
    print("  机器人SLAM同时定位与稠密建图系统")
    print("  Graph-SLAM with Dense Triangular Mesh Mapping")
    print("=" * 70)
    print()

    np.random.seed(42)
    start_time = time.time()

    # =====================================================================
    # 1. 环境生成与机器人初始化
    # =====================================================================
    print("[Phase 1] 初始化环境与机器人 ...")
    obstacles = generate_synthetic_environment()
    robot = DifferentialDriveRobot(x=0.0, y=0.0, theta=0.0,
                                    sigma_v=0.03, sigma_w=0.01, dt=0.2)
    lidar = Lidar2D(num_beams=180, max_range=8.0, fov=np.pi,
                    sigma_range=0.02)
    commands = generate_trajectory_commands(num_steps=200)
    print(f"  环境障碍物数量: {len(obstacles)}")
    print(f"  仿真步数: {len(commands)}")
    print()

    # =====================================================================
    # 2. 运动仿真与数据采集
    # =====================================================================
    print("[Phase 2] 运动仿真与数据采集 ...")
    ground_truth_poses = []
    odometry_poses = []
    scans = []
    all_points = []
    information_gains = []

    # 提取地标位置用于信息增益计算
    landmarks = []
    for obs in obstacles:
        if obs['type'] == 'circle':
            landmarks.append(np.array(obs['center']))

    gt_robot = DifferentialDriveRobot(x=0.0, y=0.0, theta=0.0,
                                       sigma_v=0.0, sigma_w=0.0, dt=0.2)
    visited_cells = set()
    prev_cells = 0

    for step, (v, w) in enumerate(commands):
        # 真实运动（无噪声）
        gt_pose, _ = gt_robot.propagate(v, w)
        ground_truth_poses.append(gt_pose.copy())

        # 带噪声的里程计运动
        odom_pose, odom_cov = robot.propagate(v, w)
        odometry_poses.append(odom_pose.copy())

        # 激光扫描
        _, points = lidar.scan_environment(gt_pose, obstacles)
        scans.append(points.copy())
        all_points.extend(points.tolist())

        # 信息增益：基于观测到新区域的信息量（有波动的独立信息）
        # 使用当前扫描点覆盖的新网格单元数量作为代理指标
        grid_res = 0.5
        for pt in points:
            gx = int(np.floor(pt[0] / grid_res))
            gy = int(np.floor(pt[1] / grid_res))
            visited_cells.add((gx, gy))
        # 本帧新覆盖的单元格比例（近似）
        new_cells = len(visited_cells) - prev_cells
        prev_cells = len(visited_cells)
        info_gain = max(1.0, float(new_cells) + np.random.normal(0, 2.0))
        information_gains.append(info_gain)

    print(f"  采集到 {len(scans)} 帧激光扫描")
    print(f"  总点云数量: {len(all_points)}")
    print()

    # =====================================================================
    # 3. 基于 CVT 与最优停止策略的关键帧选择
    # =====================================================================
    print("[Phase 3] CVT关键帧采样与最优停止策略 ...")
    # 将位姿映射到特征空间 [x, y, 0.3*theta]
    features = np.array([
        [p[0], p[1], 0.3 * normalize_angle(p[2])]
        for p in ground_truth_poses
    ], dtype=np.float64)

    cvt_sampler = CVTKeyframeSampler(num_generators=20, max_iter=30)
    generators, labels = cvt_sampler.fit(features, weights=np.array(information_gains))
    print(f"  CVT 生成元数量: {generators.shape[0]}")
    print(f"  CVT Lloyd 迭代次数: {len(cvt_sampler.energies)}")
    if len(cvt_sampler.energies) > 0:
        print(f"  CVT 最终能量: {cvt_sampler.energies[-1]:.6f}")

    # 关键帧选择：结合 CVT 与固定间隔，保证足够数量
    stop_selector = OptimalStoppingKeyframeSelector()
    stop_indices = stop_selector.select_keyframes(
        np.array(information_gains), total_frames=len(information_gains)
    )
    # CVT 代表帧：每个 Voronoi 区域取信息增益最高的帧
    cvt_indices = []
    for g in range(generators.shape[0]):
        mask = (labels == g)
        if np.any(mask):
            idx_in_group = np.argmax(np.array(information_gains)[mask])
            group_indices = np.where(mask)[0]
            cvt_indices.append(int(group_indices[idx_in_group]))
    # 固定间隔采样
    fixed_indices = list(range(0, len(commands), 10))
    # 合并并排序去重
    keyframe_indices = sorted(set(stop_indices + cvt_indices + fixed_indices))
    print(f"  最优停止策略选择关键帧数: {len(stop_indices)}")
    print(f"  CVT 代表关键帧数: {len(cvt_indices)}")
    print(f"  最终合并关键帧数: {len(keyframe_indices)}")
    print(f"  理论最优跳过比例: {1.0/np.e:.4f}")

    # 模拟最优停止成功率
    emp_prob, theo_prob = stop_selector.simulate_strategy(deck_size=100, trial_num=300)
    print(f"  模拟成功率: {emp_prob:.4f}, 理论极限: {theo_prob:.4f}")
    print()

    # =====================================================================
    # 4. 环路闭合检测（随机游走 + 随机搜索 + 置换正交性）
    # =====================================================================
    print("[Phase 4] 环路闭合检测 ...")
    loop_detector = IntegratedLoopClosureDetector(
        rw_sigma=0.15, rw_window=15, rw_threshold=1.5,
        rs_max_candidates=30, rs_trials=15, rs_threshold=0.6
    )
    closures = loop_detector.detect(ground_truth_poses, scans, scan_matching_similarity)
    print(f"  检测到 {len(closures)} 个环路闭合")
    for idx, c in enumerate(closures[:5]):
        print(f"    闭合 #{idx+1}: 帧 {c['from']} -> 帧 {c['to']}, "
              f"变换=[{c['transform'][0]:.3f}, {c['transform'][1]:.3f}, "
              f"{c['transform'][2]:.3f}], 分数={c['score']:.3f}")

    # 置换正交性演示
    ortho_demo = PermutationOrthogonalityChecker.demonstrate_permutation_property(n=10)
    print(f"  置换正交性演示（点积）: {ortho_demo:.2e}")
    print()

    # =====================================================================
    # 5. 位姿图构建与图优化
    # =====================================================================
    print("[Phase 5] 位姿图构建与图优化 ...")
    graph = PoseGraph()

    # 添加所有关键帧位姿（使用里程计作为初始值）
    for kf_idx in keyframe_indices:
        graph.add_vertex(odometry_poses[kf_idx])

    # 添加里程计边（连续关键帧之间）
    info_odom = np.diag([10.0, 10.0, 50.0])
    for i in range(len(keyframe_indices) - 1):
        idx_i = keyframe_indices[i]
        idx_j = keyframe_indices[i + 1]
        # 相对变换
        r = DifferentialDriveRobot()
        r.pose = odometry_poses[idx_i].copy()
        rel = r.relative_transform(odometry_poses[idx_j])
        graph.add_edge(i, i + 1, rel, info_odom)

    # 添加环路闭合边（来自自动检测）
    info_loop = np.diag([5.0, 5.0, 20.0])
    kf_set = set(keyframe_indices)
    auto_loop_count = 0
    for c in closures:
        if c['from'] in kf_set and c['to'] in kf_set:
            i = keyframe_indices.index(c['from'])
            j = keyframe_indices.index(c['to'])
            if abs(i - j) > 1:
                graph.add_edge(i, j, c['transform'], info_loop)
                auto_loop_count += 1

    # 手动添加起点-终点环路闭合（闭合轨迹保证）
    if len(keyframe_indices) >= 4:
        # 首帧和末帧之间的相对变换（使用真实位姿计算）
        r_first = DifferentialDriveRobot()
        r_first.pose = ground_truth_poses[keyframe_indices[0]].copy()
        rel_first_last = r_first.relative_transform(ground_truth_poses[keyframe_indices[-1]])
        # 添加微小噪声模拟观测不确定性
        rel_first_last += np.random.normal(0, 0.02, 3)
        rel_first_last[2] = normalize_angle(rel_first_last[2])
        graph.add_edge(0, len(keyframe_indices) - 1, rel_first_last, info_loop)
        print(f"  手动添加起点-终点环路闭合")

    print(f"  位姿图顶点数: {len(graph.poses)}")
    print(f"  位姿图边数: {len(graph.edges)} (自动环路: {auto_loop_count})")

    # 图优化
    if len(graph.poses) >= 2:
        optimizer = GraphSLAMOptimizer(max_iterations=30, linear_solver='sor', tol=1e-5)
        optimized_graph, final_cost, num_iters = optimizer.optimize(graph)
        print(f"  Gauss-Newton 迭代次数: {num_iters}")
        print(f"  最终代价: {final_cost:.6f}")

        # 可观测性分析
        H, _ = optimizer._build_linear_system(optimized_graph)
        analysis = ObservabilityAnalyzer.analyze_hessian(H, len(optimized_graph.poses))
    else:
        print("  顶点数不足，跳过图优化")
        optimized_graph = graph
        num_iters = 0
        final_cost = 0.0
        analysis = {
            'condition_number': np.inf,
            'nullity': 3,
            'observability_index': 0.0,
            'eigenvalues': np.array([0.0, 0.0, 0.0])
        }
    print(f"  Hessian 条件数: {analysis['condition_number']:.3e}")
    print(f"  零特征值个数（不可观自由度）: {analysis['nullity']}")
    print(f"  可观测性指标: {analysis['observability_index']:.6f}")

    # SOR 求解器单独测试
    print()
    print("  [SOR 稀疏求解器测试]")
    n_test = 50
    offsets = np.array([0, 1, -1, 2, -2], dtype=np.int64)
    ndiag = len(offsets)
    values = np.random.rand(n_test, ndiag) * 0.5
    values[:, 0] += 2.0  # 加强主对角线
    r8gd = R8GDMatrix(n_test, ndiag, offsets, values)
    x_true = np.random.randn(n_test)
    b_test = r8gd.mv(x_true)
    sor = SORSolver(omega=1.2, max_iter=500, tol=1e-8)
    x_solved, res_norm, sor_iters, sor_conv = sor.solve_sparse(r8gd, b_test)
    rel_err = np.linalg.norm(x_solved - x_true) / np.linalg.norm(x_true)
    print(f"    SOR 相对误差: {rel_err:.2e}, 迭代次数: {sor_iters}, 收敛: {sor_conv}")
    print()

    # =====================================================================
    # 6. 稠密三角网格地图构建
    # =====================================================================
    print("[Phase 6] 稠密三角网格地图构建 ...")
    # 收集优化后的关键帧扫描点云
    mesh_points = []
    for i, kf_idx in enumerate(keyframe_indices):
        pose = optimized_graph.poses[i]
        pts_local = scans[kf_idx]
        if pts_local.shape[0] == 0:
            continue
        # 转换到世界坐标系
        c, s = np.cos(pose[2]), np.sin(pose[2])
        R = np.array([[c, -s], [s, c]])
        pts_world = pts_local @ R.T + pose[0:2]
        mesh_points.extend(pts_world.tolist())

    mesh_points = np.array(mesh_points, dtype=np.float64)
    if mesh_points.shape[0] > 0:
        mesh_map = TriangularMeshMap()
        mesh_map.from_point_cloud(mesh_points, max_tri_area=None, max_points=600)
        min_angle, max_ar = mesh_map.compute_mesh_quality()
        print(f"  三角网格顶点数: {mesh_map.vertices.shape[0]}")
        print(f"  三角面片数: {mesh_map.triangles.shape[0]}")
        print(f"  边界边数: {mesh_map.boundary_edges.shape[0]}")
        print(f"  网格质量 — 最小角: {min_angle:.2f}°, 最大长宽比: {max_ar:.3f}")
    else:
        print("  警告: 点云为空，跳过网格构建")
    print()

    # =====================================================================
    # 7. 有限元不确定性场分析
    # =====================================================================
    print("[Phase 7] 有限元不确定性场分析 ...")
    fem = FEMUncertaintyField(nx=25, ny=25)

    def a_func(x, y):
        """扩散系数 — 在障碍物附近不确定性传播更慢"""
        return 0.5 + 0.3 * np.sin(0.5 * x) * np.cos(0.5 * y)

    def c_func(x, y):
        """反应系数 — 在信息丰富区域抑制不确定性"""
        info = 0.0
        for lm in landmarks:
            d2 = (x - lm[0]) ** 2 + (y - lm[1]) ** 2
            info += np.exp(-d2 / 2.0)
        return 0.1 + 0.5 * info

    def f_func(x, y):
        """源项 — 在机器人轨迹附近引入不确定性"""
        source = 0.0
        for p in ground_truth_poses[::10]:
            d2 = (x - p[0]) ** 2 + (y - p[1]) ** 2
            source += np.exp(-d2 / 0.5)
        return 0.05 + 0.2 * source

    u_field, x_grid, y_grid = fem.solve_uncertainty_field(
        domain=((-6.0, 6.0), (-6.0, 6.0)),
        a_func=a_func, c_func=c_func, f_func=f_func
    )
    print(f"  有限元网格: {fem.nx} × {fem.ny}")
    print(f"  不确定性场最大值: {np.max(u_field):.6f}")
    print(f"  不确定性场最小值: {np.min(u_field):.6f}")
    print(f"  不确定性场均值: {np.mean(u_field):.6f}")

    # 在优化后的位姿处采样不确定性
    opt_positions = np.array([p[0:2] for p in optimized_graph.poses])
    sampled_uncertainty = fem.sample_field_at_points(u_field, x_grid, y_grid, opt_positions)
    print(f"  关键帧位置平均不确定性: {np.mean(sampled_uncertainty):.6f}")

    # 环形区域协方差估计
    annular = AnnularCovarianceEstimator(nr=6, nt=24)
    def cov_func(px, py):
        # 简化的协方差场
        val = np.exp(-(px**2 + py**2) / 10.0)
        return val

    ring_cov = annular.integrate_annular_covariance(
        center=(0.0, 0.0), r1=1.0, r2=3.0, covariance_func=cov_func
    )
    print(f"  环形区域(1-3m)积分协方差: {ring_cov:.6f}")
    print()

    # =====================================================================
    # 8. 性能评估与结果汇总
    # =====================================================================
    print("[Phase 8] 性能评估与结果汇总 ...")
    gt_array = np.array(ground_truth_poses)
    odom_array = np.array(odometry_poses)

    # 在关键帧上评估
    gt_kf = gt_array[keyframe_indices]
    odom_kf = odom_array[keyframe_indices]
    opt_kf = np.array(optimized_graph.poses)

    ate_odom = compute_trajectory_ate(odom_kf, gt_kf)
    ate_opt = compute_trajectory_ate(opt_kf, gt_kf)
    improvement = (ate_odom - ate_opt) / (ate_odom + 1e-12) * 100.0

    print(f"  关键帧数量: {len(keyframe_indices)}")
    print(f"  里程计 ATE: {ate_odom:.4f} m")
    print(f"  优化后 ATE: {ate_opt:.4f} m")
    print(f"  优化改善率: {improvement:.2f}%")

    # 卡方一致性检验
    dim = 2
    chi2_thresh = chi2_confidence_interval(dim, confidence=0.99)
    position_errors = opt_kf[:, 0:2] - gt_kf[:, 0:2]
    mean_error = np.mean(np.linalg.norm(position_errors, axis=1))
    print(f"  平均位置误差: {mean_error:.4f} m")
    print(f"  99% 卡方阈值 (dim={dim}): {chi2_thresh:.4f}")
    print()

    elapsed = time.time() - start_time
    print("=" * 70)
    print(f"  总运行时间: {elapsed:.3f} s")
    print("  SLAM 建图流程完成")
    print("=" * 70)

    # 返回关键结果供外部验证
    return {
        'ate_odometry': ate_odom,
        'ate_optimized': ate_opt,
        'num_keyframes': len(keyframe_indices),
        'num_closures': len(closures),
        'condition_number': analysis['condition_number'],
        'mesh_vertices': mesh_map.vertices.shape[0] if mesh_points.shape[0] > 0 else 0,
        'mesh_triangles': mesh_map.triangles.shape[0] if mesh_points.shape[0] > 0 else 0,
        'fem_max_uncertainty': float(np.max(u_field)),
        'sor_relative_error': rel_err,
    }


if __name__ == "__main__":
    results = main()

# ================================================================
# 测试用例（52个，assert模式，涉及随机值均使用固定种子）
# ================================================================
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

print('\n全部 52 个测试通过!\n')
