
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
        q = np.asarray(q, dtype=float)
        try:
            T_ee = self.kin.forward_kinematics(q)
            p_ee = T_ee[:3, 3]
        except Exception:
            return True
        for obs in self.obstacles:
            if obs.collision_check(p_ee, safety_margin=0.08):
                return True

            if hasattr(self.kin, '_T_list') and len(self.kin._T_list) > 4:
                p_wrist = self.kin._T_list[4][:3, 3]
                if obs.collision_check(p_wrist, safety_margin=0.05):
                    return True
        return False

    def _obstacle_distance_penalty(self, p: np.ndarray) -> float:
        p = np.asarray(p, dtype=float)
        if not self.obstacles:
            return 1.0
        min_dist = min(obs.signed_distance(p) for obs in self.obstacles)
        return min_dist

    def plan_pseudospectral_trajectory(self, q_start: np.ndarray, q_goal: np.ndarray,
                                        t0: float = 0.0, tf: float = 2.0) -> JointSpaceBezierTrajectory:
        q_start = np.asarray(q_start, dtype=float)
        q_goal = np.asarray(q_goal, dtype=float)
        ps = PseudospectralCollocation(n_nodes=12)
        ps.scale_time(t0, tf)

        nodes_t = ps.nodes * ps.scale + (t0 + tf) / 2.0
        state_guess = np.zeros((ps.n_nodes, self.n_dof), dtype=float)
        for i in range(ps.n_nodes):
            alpha = (nodes_t[i] - t0) / (tf - t0 + 1e-14)
            state_guess[i] = q_start + alpha * (q_goal - q_start)



        def dynamics(q_vec):
            return (q_goal - q_start) / (tf - t0)

        res = ps.collocation_constraints(state_guess, dynamics)







        state_refined = state_guess.copy()
        raise NotImplementedError("Hole 2: 请实现伪谱配点约束修正逻辑")

        state_refined = np.clip(state_refined, self.q_min, self.q_max)


        P = np.vstack([q_start, state_refined[ps.n_nodes // 2], q_goal])

        P5 = self._degree_elevate(P, 5)
        P5 = clamp_control_points_to_joint_limits(P5, self.q_min, self.q_max)
        return JointSpaceBezierTrajectory(P5, t0, tf)

    def _degree_elevate(self, P: np.ndarray, target_degree: int) -> np.ndarray:
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
        q_start = np.asarray(q_start, dtype=float)
        q_goal = np.asarray(q_goal, dtype=float)

        means = [q_start, q_goal, (q_start + q_goal) / 2.0]
        covs = [0.5 * np.eye(self.n_dof)] * 3
        samples1 = self.sampler.gaussian_mixture_sample(100, means, covs)
        samples2 = self.sampler.uniform_random(100)
        all_samples = np.vstack([samples1, samples2])
        valid = [q for q in all_samples if not self._collision_check(q)]
        if len(valid) < 10:

            valid = all_samples[:50].tolist()
        graph = RoadmapGraph(self.n_dof)
        start_idx = graph.add_node(q_start)
        goal_idx = graph.add_node(q_goal)
        for q in valid:
            graph.add_node(q)
        graph.knn_edges(k=5, radius=2.0)
        path_idx, cost = graph.dijkstra(start_idx, goal_idx)
        if not path_idx:

            return [q_start, q_goal]
        return [graph.nodes[i] for i in path_idx]

    def optimize_trajectory_praxis(self, q_start: np.ndarray, q_goal: np.ndarray,
                                    n_control_points: int = 6) -> JointSpaceBezierTrajectory:
        q_start = np.asarray(q_start, dtype=float)
        q_goal = np.asarray(q_goal, dtype=float)

        P_init = np.zeros((n_control_points, self.n_dof), dtype=float)
        for i in range(n_control_points):
            alpha = i / (n_control_points - 1.0)
            P_init[i] = q_start + alpha * (q_goal - q_start)

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

        P5 = self._degree_elevate(P_opt, 5)
        return JointSpaceBezierTrajectory(P5, 0.0, 2.0)

    def run_full_pipeline(self) -> Dict:

        obs1 = generate_box_obstacle([0.5, 0.0, 0.3], [0.4, 0.4, 0.4], density=2.0)
        obs2 = generate_sphere_obstacle([0.3, 0.4, 0.5], 0.15, n_segments=12, density=1.5)
        self.add_obstacle(obs1)
        self.add_obstacle(obs2)


        q_start = np.array([0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.0], dtype=float)
        q_goal = np.array([0.5, 0.3, -0.2, -1.0, 0.3, 0.8, 0.5], dtype=float)
        q_start = np.clip(q_start, self.q_min, self.q_max)
        q_goal = np.clip(q_goal, self.q_min, self.q_max)


        path_nodes = self.plan_prm_path(q_start, q_goal)
        print(f"[PRM] 找到路径节点数: {len(path_nodes)}")


        traj_segments = []
        for i in range(len(path_nodes) - 1):
            seg = self.plan_pseudospectral_trajectory(
                path_nodes[i], path_nodes[i + 1],
                t0=0.0, tf=1.0
            )
            traj_segments.append(seg)


        optimized_traj = self.optimize_trajectory_praxis(q_start, q_goal, n_control_points=6)
        self.trajectory = optimized_traj
        print(f"[PRAXIS] 优化完成，轨迹时间区间 [{optimized_traj.t0:.2f}, {optimized_traj.tf:.2f}]")


        total_cycles = 100
        joint_weights = np.array([5, 5, 4, 4, 3, 3, 2], dtype=int)
        cycle_allocations = allocate_control_cycles(total_cycles, joint_weights, max_solutions=20)
        print(f"[Diophantine] 控制周期分配方案数: {cycle_allocations.shape[0] if cycle_allocations.size > 0 else 0}")


        xml_sol = generate_example_cplex_xml()
        milp_decision = parse_milp_trajectory_decision(xml_sol)
        print(f"[MILP] 选择走廊: {milp_decision['selected_corridors']}")


        f_dyn = manipulator_dynamics_ode(self.kin)
        y0 = np.concatenate([q_start, np.zeros(self.n_dof)])
        integrator = StiffODEIntegrator(tol=1e-5)
        t_arr, y_arr = integrator.integrate(f_dyn, (0.0, 1.0), y0, h0=0.02)
        print(f"[ODE] 动力学仿真完成，时间步数: {t_arr.size}")


        ps = PseudospectralCollocation(n_nodes=8)
        ps.scale_time(0.0, 1.0)
        vel_vals = np.array([np.linalg.norm(optimized_traj.velocity(t)) for t in ps.nodes * ps.scale + 0.5])
        cost_integral = ps.integrate_cost(vel_vals)
        print(f"[Pseudospectral] 速度代价积分: {cost_integral:.4f}")


        manips = []
        for t in np.linspace(optimized_traj.t0, optimized_traj.tf, 20):
            q = optimized_traj.position(t)
            manips.append(self.kin.manipulability_measure(q))
        avg_manip = np.mean(manips)
        min_manip = np.min(manips)
        print(f"[Manipulability] 平均={avg_manip:.4f}, 最小={min_manip:.4f}")


        profile = profile_data()
        workspace = (np.array([-1.0, -1.0, 0.0]), np.array([1.0, 1.0, 1.0]))
        target_curve = scale_profile_to_workspace(profile, workspace)
        print(f"[Profile] 目标轮廓点数: {target_curve.shape[0]}")


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
