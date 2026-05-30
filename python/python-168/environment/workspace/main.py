
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
    obstacles = []

    obstacles.append({'type': 'segment', 'p1': (-5.0, -5.0), 'p2': (5.0, -5.0)})
    obstacles.append({'type': 'segment', 'p1': (5.0, -5.0), 'p2': (5.0, 5.0)})
    obstacles.append({'type': 'segment', 'p1': (5.0, 5.0), 'p2': (-5.0, 5.0)})
    obstacles.append({'type': 'segment', 'p1': (-5.0, 5.0), 'p2': (-5.0, -5.0)})

    for cx, cy in [(-2.0, -2.0), (2.0, 2.0), (0.0, 0.0)]:
        obstacles.append({'type': 'circle', 'center': (cx, cy), 'radius': 0.4})

    obstacles.append({'type': 'segment', 'p1': (-3.0, -1.0), 'p2': (-1.0, -1.0)})
    obstacles.append({'type': 'segment', 'p1': (1.0, 1.0), 'p2': (3.0, 1.0)})
    return obstacles


def generate_trajectory_commands(num_steps=200):
    commands = []
    side_len = 80
    turn_steps = 20
    v_straight = 0.4
    w_turn = np.pi / 2.0 / (turn_steps * 0.2)

    for _ in range(4):

        for _ in range(side_len):
            commands.append((v_straight, 0.0))

        for _ in range(turn_steps):
            commands.append((0.05, -w_turn))


    while len(commands) < num_steps:
        commands.append((v_straight, 0.0))
    return commands[:num_steps]


def scan_matching_similarity(scan1, scan2):
    if scan1.shape[0] == 0 or scan2.shape[0] == 0:
        return 0.0

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




    print("[Phase 2] 运动仿真与数据采集 ...")
    ground_truth_poses = []
    odometry_poses = []
    scans = []
    all_points = []
    information_gains = []


    landmarks = []
    for obs in obstacles:
        if obs['type'] == 'circle':
            landmarks.append(np.array(obs['center']))

    gt_robot = DifferentialDriveRobot(x=0.0, y=0.0, theta=0.0,
                                       sigma_v=0.0, sigma_w=0.0, dt=0.2)
    visited_cells = set()
    prev_cells = 0

    for step, (v, w) in enumerate(commands):

        gt_pose, _ = gt_robot.propagate(v, w)
        ground_truth_poses.append(gt_pose.copy())


        odom_pose, odom_cov = robot.propagate(v, w)
        odometry_poses.append(odom_pose.copy())


        _, points = lidar.scan_environment(gt_pose, obstacles)
        scans.append(points.copy())
        all_points.extend(points.tolist())



        grid_res = 0.5
        for pt in points:
            gx = int(np.floor(pt[0] / grid_res))
            gy = int(np.floor(pt[1] / grid_res))
            visited_cells.add((gx, gy))

        new_cells = len(visited_cells) - prev_cells
        prev_cells = len(visited_cells)
        info_gain = max(1.0, float(new_cells) + np.random.normal(0, 2.0))
        information_gains.append(info_gain)

    print(f"  采集到 {len(scans)} 帧激光扫描")
    print(f"  总点云数量: {len(all_points)}")
    print()




    print("[Phase 3] CVT关键帧采样与最优停止策略 ...")

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


    stop_selector = OptimalStoppingKeyframeSelector()
    stop_indices = stop_selector.select_keyframes(
        np.array(information_gains), total_frames=len(information_gains)
    )

    cvt_indices = []
    for g in range(generators.shape[0]):
        mask = (labels == g)
        if np.any(mask):
            idx_in_group = np.argmax(np.array(information_gains)[mask])
            group_indices = np.where(mask)[0]
            cvt_indices.append(int(group_indices[idx_in_group]))

    fixed_indices = list(range(0, len(commands), 10))

    keyframe_indices = sorted(set(stop_indices + cvt_indices + fixed_indices))
    print(f"  最优停止策略选择关键帧数: {len(stop_indices)}")
    print(f"  CVT 代表关键帧数: {len(cvt_indices)}")
    print(f"  最终合并关键帧数: {len(keyframe_indices)}")
    print(f"  理论最优跳过比例: {1.0/np.e:.4f}")


    emp_prob, theo_prob = stop_selector.simulate_strategy(deck_size=100, trial_num=300)
    print(f"  模拟成功率: {emp_prob:.4f}, 理论极限: {theo_prob:.4f}")
    print()




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


    ortho_demo = PermutationOrthogonalityChecker.demonstrate_permutation_property(n=10)
    print(f"  置换正交性演示（点积）: {ortho_demo:.2e}")
    print()




    print("[Phase 5] 位姿图构建与图优化 ...")
    graph = PoseGraph()


    for kf_idx in keyframe_indices:
        graph.add_vertex(odometry_poses[kf_idx])








    raise NotImplementedError("Hole 3: pose graph edge construction not implemented")

    print(f"  位姿图顶点数: {len(graph.poses)}")
    print(f"  位姿图边数: {len(graph.edges)} (自动环路: {auto_loop_count})")


    if len(graph.poses) >= 2:
        optimizer = GraphSLAMOptimizer(max_iterations=30, linear_solver='sor', tol=1e-5)
        optimized_graph, final_cost, num_iters = optimizer.optimize(graph)
        print(f"  Gauss-Newton 迭代次数: {num_iters}")
        print(f"  最终代价: {final_cost:.6f}")


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


    print()
    print("  [SOR 稀疏求解器测试]")
    n_test = 50
    offsets = np.array([0, 1, -1, 2, -2], dtype=np.int64)
    ndiag = len(offsets)
    values = np.random.rand(n_test, ndiag) * 0.5
    values[:, 0] += 2.0
    r8gd = R8GDMatrix(n_test, ndiag, offsets, values)
    x_true = np.random.randn(n_test)
    b_test = r8gd.mv(x_true)
    sor = SORSolver(omega=1.2, max_iter=500, tol=1e-8)
    x_solved, res_norm, sor_iters, sor_conv = sor.solve_sparse(r8gd, b_test)
    rel_err = np.linalg.norm(x_solved - x_true) / np.linalg.norm(x_true)
    print(f"    SOR 相对误差: {rel_err:.2e}, 迭代次数: {sor_iters}, 收敛: {sor_conv}")
    print()




    print("[Phase 6] 稠密三角网格地图构建 ...")

    mesh_points = []
    for i, kf_idx in enumerate(keyframe_indices):
        pose = optimized_graph.poses[i]
        pts_local = scans[kf_idx]
        if pts_local.shape[0] == 0:
            continue

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




    print("[Phase 7] 有限元不确定性场分析 ...")
    fem = FEMUncertaintyField(nx=25, ny=25)

    def a_func(x, y):
        return 0.5 + 0.3 * np.sin(0.5 * x) * np.cos(0.5 * y)

    def c_func(x, y):
        info = 0.0
        for lm in landmarks:
            d2 = (x - lm[0]) ** 2 + (y - lm[1]) ** 2
            info += np.exp(-d2 / 2.0)
        return 0.1 + 0.5 * info

    def f_func(x, y):
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


    opt_positions = np.array([p[0:2] for p in optimized_graph.poses])
    sampled_uncertainty = fem.sample_field_at_points(u_field, x_grid, y_grid, opt_positions)
    print(f"  关键帧位置平均不确定性: {np.mean(sampled_uncertainty):.6f}")


    annular = AnnularCovarianceEstimator(nr=6, nt=24)
    def cov_func(px, py):

        val = np.exp(-(px**2 + py**2) / 10.0)
        return val

    ring_cov = annular.integrate_annular_covariance(
        center=(0.0, 0.0), r1=1.0, r2=3.0, covariance_func=cov_func
    )
    print(f"  环形区域(1-3m)积分协方差: {ring_cov:.6f}")
    print()




    print("[Phase 8] 性能评估与结果汇总 ...")
    gt_array = np.array(ground_truth_poses)
    odom_array = np.array(odometry_poses)


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
