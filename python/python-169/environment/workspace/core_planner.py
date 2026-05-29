"""
核心规划器模块：整合所有子系统
==============================
将以下模块集成为一个完整的机械臂轨迹规划与避障系统:
  - kinematics_dynamics    : 运动学/动力学/ODE/CGNE
  - bernstein_path         : Bézier轨迹生成
  - obstacle_geometry      : 障碍物几何/SDF/棱柱积分
  - configuration_space    : 构型空间采样/单形网格/PDF采样
  - sparse_linear_algebra  : GMRES/ILU/CGNE
  - pseudospectral_control : 伪谱配点/高斯积分
  - roadmap_graph          : PRM/HITS/Profile数据
  - discrete_planning      : Diophantine/精确覆盖/RREF
  - derivative_free_opt    : PRAXIS无导数优化
  - milp_parser            : CPLEX解解析

核心科学问题:
  在 cluttered 3D环境中，7自由度冗余机械臂的实时轨迹规划与动态避障。
  综合使用伪谱法粗规划、PRM图搜索、Bézier曲线光滑化、
  PRAXIS在线优化，实现从起点到目标轮廓跟踪的完整运动。
"""

import numpy as np
from typing import List, Tuple, Dict, Optional

from kinematics_dynamics import (
    ManipulatorKinematics, StiffODEIntegrator, differential_ik_solver,
    manipulator_dynamics_ode
)
from bernstein_path import (
    JointSpaceBezierTrajectory, generate_minimum_jerk_bezier,
    clamp_control_points_to_joint_limits
)
from obstacle_geometry import (
    PolyhedralObstacle, generate_box_obstacle, generate_sphere_obstacle
)
from configuration_space import ConfigurationSampler
from sparse_linear_algebra import SparseSolver
from pseudospectral_control import PseudospectralCollocation
from roadmap_graph import (
    RoadmapGraph, build_prm_roadmap, profile_data, scale_profile_to_workspace
)
from discrete_planning import allocate_control_cycles, workspace_coverage_exact_cover
from derivative_free_opt import PraxisOptimizer, trajectory_cost_function
from milp_parser import generate_example_cplex_xml, parse_milp_trajectory_decision


class ManipulatorMotionPlanner:
    r"""
    7自由度机械臂运动规划器。
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.kin = ManipulatorKinematics()
        self.n_dof = self.kin.n_dof
        self.q_min = -np.pi * np.ones(self.n_dof)
        self.q_max = np.pi * np.ones(self.n_dof)
        self.sampler = ConfigurationSampler(self.q_min, self.q_max, self.n_dof, seed=seed)
        self.obstacles = []
        self.trajectory = None
        self.roadmap = None

    def add_obstacle(self, obs: PolyhedralObstacle):
        self.obstacles.append(obs)

    def _collision_check(self, q: np.ndarray) -> bool:
        """构型空间碰撞检测：检查末端和若干中间连杆位置。"""
        q = np.asarray(q, dtype=float)
        try:
            T_ee = self.kin.forward_kinematics(q)
            p_ee = T_ee[:3, 3]
        except Exception:
            return True
        for obs in self.obstacles:
            if obs.collision_check(p_ee, safety_margin=0.08):
                return True
            # 简化：检查腕部位置（第5连杆）
            if hasattr(self.kin, '_T_list') and len(self.kin._T_list) > 4:
                p_wrist = self.kin._T_list[4][:3, 3]
                if obs.collision_check(p_wrist, safety_margin=0.05):
                    return True
        return False

    def _obstacle_distance_penalty(self, p: np.ndarray) -> float:
        """返回点到所有障碍物的最小有符号距离。"""
        p = np.asarray(p, dtype=float)
        if not self.obstacles:
            return 1.0
        min_dist = min(obs.signed_distance(p) for obs in self.obstacles)
        return min_dist

    def plan_pseudospectral_trajectory(self, q_start: np.ndarray, q_goal: np.ndarray,
                                        t0: float = 0.0, tf: float = 2.0) -> JointSpaceBezierTrajectory:
        r"""
        使用伪谱法进行粗规划：在Legendre节点上求解微分逆运动学，
        然后用Bézier曲线拟合结果。
        """
        q_start = np.asarray(q_start, dtype=float)
        q_goal = np.asarray(q_goal, dtype=float)
        ps = PseudospectralCollocation(n_nodes=12)
        ps.scale_time(t0, tf)
        # 在节点上插值：从 q_start 到 q_goal 的线性插值作为初始猜测
        nodes_t = ps.nodes * ps.scale + (t0 + tf) / 2.0
        state_guess = np.zeros((ps.n_nodes, self.n_dof), dtype=float)
        for i in range(ps.n_nodes):
            alpha = (nodes_t[i] - t0) / (tf - t0 + 1e-14)
            state_guess[i] = q_start + alpha * (q_goal - q_start)

        # 伪谱微分约束：速度应平滑
        # 这里使用简化动力学：\dot{q} = dq_des，其中 dq_des 为最小加加速度插值
        def dynamics(q_vec):
            return (q_goal - q_start) / (tf - t0)

        res = ps.collocation_constraints(state_guess, dynamics)
        # 用残差修正初始猜测（逐维度简单阻尼修正）
        # D_scaled 是 (n_nodes, n_nodes)，state_guess 是 (n_nodes, n_dof)
        # 残差 res = D_scaled @ state_guess - dynamics_val (广播到每列)
        # 修正量: 对每列独立，使用 D_scaled 的伪逆近似
        # TODO: Hole 2 - 实现伪谱配点约束修正
        # 利用 ps.D_scaled 和 CGNE 求解器对每维关节进行修正
        # 目标: 使配点约束残差最小化
        state_refined = state_guess.copy()
        raise NotImplementedError("Hole 2: 请实现伪谱配点约束修正逻辑")
        # 投影到关节限位
        state_refined = np.clip(state_refined, self.q_min, self.q_max)
        # 用Bézier拟合这些点
        # 直接取起点、中间点、终点作为控制点
        P = np.vstack([q_start, state_refined[ps.n_nodes // 2], q_goal])
        # 升阶到5次以获得更平滑轨迹
        P5 = self._degree_elevate(P, 5)
        P5 = clamp_control_points_to_joint_limits(P5, self.q_min, self.q_max)
        return JointSpaceBezierTrajectory(P5, t0, tf)

    def _degree_elevate(self, P: np.ndarray, target_degree: int) -> np.ndarray:
        r"""
        将Bézier曲线从当前次数提升到target_degree。
        使用de Casteljau风格的升阶公式:
          P_i^{(n+1)} = (i/(n+1)) P_{i-1}^{(n)} + (1 - i/(n+1)) P_i^{(n)}
        """
        P = np.array(P, dtype=float)
        n = P.shape[0] - 1
        while n < target_degree:
            n_new = n + 1
            P_new = np.zeros((n_new + 1, P.shape[1]), dtype=float)
            P_new[0] = P[0]
            P_new[n_new] = P[n]
            for i in range(1, n_new):
                P_new[i] = (i / n_new) * P[i - 1] + (1.0 - i / n_new) * P[i]
            P = P_new
            n = n_new
        return P

    def plan_prm_path(self, q_start: np.ndarray, q_goal: np.ndarray) -> List[np.ndarray]:
        r"""
        使用PRM在构型空间中搜索路径。
        """
        q_start = np.asarray(q_start, dtype=float)
        q_goal = np.asarray(q_goal, dtype=float)
        # 混合采样：均匀+高斯
        means = [q_start, q_goal, (q_start + q_goal) / 2.0]
        covs = [0.5 * np.eye(self.n_dof)] * 3
        samples1 = self.sampler.gaussian_mixture_sample(100, means, covs)
        samples2 = self.sampler.uniform_random(100)
        all_samples = np.vstack([samples1, samples2])
        valid = [q for q in all_samples if not self._collision_check(q)]
        if len(valid) < 10:
            # 若碰撞太密集，放宽检测
            valid = all_samples[:50].tolist()
        graph = RoadmapGraph(self.n_dof)
        start_idx = graph.add_node(q_start)
        goal_idx = graph.add_node(q_goal)
        for q in valid:
            graph.add_node(q)
        graph.knn_edges(k=5, radius=2.0)
        path_idx, cost = graph.dijkstra(start_idx, goal_idx)
        if not path_idx:
            # 若PRM失败，返回直线路径
            return [q_start, q_goal]
        return [graph.nodes[i] for i in path_idx]

    def optimize_trajectory_praxis(self, q_start: np.ndarray, q_goal: np.ndarray,
                                    n_control_points: int = 6) -> JointSpaceBezierTrajectory:
        r"""
        使用PRAXIS无导数优化器对轨迹控制点进行优化。
        """
        q_start = np.asarray(q_start, dtype=float)
        q_goal = np.asarray(q_goal, dtype=float)
        # 初始化控制点：线性插值
        P_init = np.zeros((n_control_points, self.n_dof), dtype=float)
        for i in range(n_control_points):
            alpha = i / (n_control_points - 1.0)
            P_init[i] = q_start + alpha * (q_goal - q_start)
        # 展平为优化变量（固定起点和终点）
        x0 = P_init[1:-1].flatten()
        n_inner = x0.size
        if n_inner == 0:
            P_opt = P_init
        else:
            def cost_fn(x):
                P_full = P_init.copy()
                if n_inner > 0:
                    P_full[1:-1] = x.reshape(n_control_points - 2, self.n_dof)
                return trajectory_cost_function(
                    P_full.flatten(),
                    lambda q: self.kin.forward_kinematics(q),
                    lambda p: self._obstacle_distance_penalty(p),
                    q_start, q_goal
                )
            opt = PraxisOptimizer(tol=1e-4, max_iter=200, h0=0.2)
            x_opt, f_opt = opt.minimize(cost_fn, x0)
            P_opt = P_init.copy()
            P_opt[1:-1] = x_opt.reshape(n_control_points - 2, self.n_dof)
        P_opt = clamp_control_points_to_joint_limits(P_opt, self.q_min, self.q_max)
        # 升阶到5次
        P5 = self._degree_elevate(P_opt, 5)
        return JointSpaceBezierTrajectory(P5, 0.0, 2.0)

    def run_full_pipeline(self) -> Dict:
        r"""
        执行完整的规划管线:
          1. 初始化障碍物环境
          2. 设置起点和目标（Profile轮廓跟踪）
          3. PRM粗路径规划
          4. 伪谱法轨迹细化
          5. PRAXIS优化
          6. 离散资源分配（Diophantine）
          7. MILP决策解析
          8. 动力学仿真验证
        """
        # 1. 障碍物
        obs1 = generate_box_obstacle([0.5, 0.0, 0.3], [0.4, 0.4, 0.4], density=2.0)
        obs2 = generate_sphere_obstacle([0.3, 0.4, 0.5], 0.15, n_segments=12, density=1.5)
        self.add_obstacle(obs1)
        self.add_obstacle(obs2)

        # 2. 起点和目标构型
        q_start = np.array([0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.0], dtype=float)
        q_goal = np.array([0.5, 0.3, -0.2, -1.0, 0.3, 0.8, 0.5], dtype=float)
        q_start = np.clip(q_start, self.q_min, self.q_max)
        q_goal = np.clip(q_goal, self.q_min, self.q_max)

        # 3. PRM粗路径
        path_nodes = self.plan_prm_path(q_start, q_goal)
        print(f"[PRM] 找到路径节点数: {len(path_nodes)}")

        # 4. 伪谱法轨迹（对每段路径）
        traj_segments = []
        for i in range(len(path_nodes) - 1):
            seg = self.plan_pseudospectral_trajectory(
                path_nodes[i], path_nodes[i + 1],
                t0=0.0, tf=1.0
            )
            traj_segments.append(seg)

        # 5. PRAXIS优化整条轨迹
        optimized_traj = self.optimize_trajectory_praxis(q_start, q_goal, n_control_points=6)
        self.trajectory = optimized_traj
        print(f"[PRAXIS] 优化完成，轨迹时间区间 [{optimized_traj.t0:.2f}, {optimized_traj.tf:.2f}]")

        # 6. 离散资源分配
        total_cycles = 100
        joint_weights = np.array([5, 5, 4, 4, 3, 3, 2], dtype=int)
        cycle_allocations = allocate_control_cycles(total_cycles, joint_weights, max_solutions=20)
        print(f"[Diophantine] 控制周期分配方案数: {cycle_allocations.shape[0] if cycle_allocations.size > 0 else 0}")

        # 7. MILP决策解析
        xml_sol = generate_example_cplex_xml()
        milp_decision = parse_milp_trajectory_decision(xml_sol)
        print(f"[MILP] 选择走廊: {milp_decision['selected_corridors']}")

        # 8. 动力学仿真验证
        f_dyn = manipulator_dynamics_ode(self.kin)
        y0 = np.concatenate([q_start, np.zeros(self.n_dof)])
        integrator = StiffODEIntegrator(tol=1e-5)
        t_arr, y_arr = integrator.integrate(f_dyn, (0.0, 1.0), y0, h0=0.02)
        print(f"[ODE] 动力学仿真完成，时间步数: {t_arr.size}")

        # 9. 伪谱代价积分
        ps = PseudospectralCollocation(n_nodes=8)
        ps.scale_time(0.0, 1.0)
        vel_vals = np.array([np.linalg.norm(optimized_traj.velocity(t)) for t in ps.nodes * ps.scale + 0.5])
        cost_integral = ps.integrate_cost(vel_vals)
        print(f"[Pseudospectral] 速度代价积分: {cost_integral:.4f}")

        # 10. 可操纵性评估
        manips = []
        for t in np.linspace(optimized_traj.t0, optimized_traj.tf, 20):
            q = optimized_traj.position(t)
            manips.append(self.kin.manipulability_measure(q))
        avg_manip = np.mean(manips)
        min_manip = np.min(manips)
        print(f"[Manipulability] 平均={avg_manip:.4f}, 最小={min_manip:.4f}")

        # 11. Profile轮廓跟踪测试
        profile = profile_data()
        workspace = (np.array([-1.0, -1.0, 0.0]), np.array([1.0, 1.0, 1.0]))
        target_curve = scale_profile_to_workspace(profile, workspace)
        print(f"[Profile] 目标轮廓点数: {target_curve.shape[0]}")

        # 12. 稀疏求解器测试（CGNE微分逆运动学）
        dx_des = np.array([0.1, 0.05, 0.02, 0.0, 0.0, 0.0])
        dq_ik = differential_ik_solver(self.kin, q_start, dx_des)
        print(f"[CGNE] 微分逆运动学解范数: {np.linalg.norm(dq_ik):.4f}")

        return {
            'trajectory': optimized_traj,
            'path_nodes': path_nodes,
            'cycle_allocations': cycle_allocations,
            'milp_decision': milp_decision,
            't_arr': t_arr,
            'y_arr': y_arr,
            'cost_integral': cost_integral,
            'avg_manipulability': avg_manip,
            'min_manipulability': min_manip,
            'target_curve': target_curve,
            'dq_ik': dq_ik,
        }
