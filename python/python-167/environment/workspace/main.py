
import numpy as np
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import Timer, check_numerical_singularity, gershgorin_discs
from config_parser import parse_robot_config, extract_link_params
from terrain_model import TriangulatedTerrain, QuadrilateralTerrainPatch, generate_sample_terrain
from robot_kinematics import SerialLegKinematics, JointLimitConstraint, FootContactGeometry
from gait_dynamics import CPGNetwork, TrapezoidalIntegrator, StanceSwingAutomaton, LegDynamics
from numerical_solver import CholeskySolver, BlockTridiagonalSolver, Radix2FFT, VandermondeSolver, MatrixMultiplyBenchmark
from trajectory_planner import TSPBruteForce, PolynomialSwingTrajectory, FootfallPlanner
from stability_optimizer import SupportPolygon, StabilityMargin, SupportGraphCentrality, LinearStabilityConstraint
from chaotic_search import GaitParameterOptimizer


def build_default_robot_config() -> str:
    xml = """<?xml version="1.0"?>
<robot name="hexapod">
  <link name="base_link">
    <inertial>
      <mass value="2.5"/>
      <inertia ixx="0.012" ixy="0.0" ixz="0.0" iyy="0.015" iyz="0.0" izz="0.018"/>
      <origin xyz="0 0 0.05"/>
    </inertial>
  </link>
  <link name="leg1_coxa">
    <inertial>
      <mass value="0.08"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0002"/>
      <origin xyz="0.03 0 0"/>
    </inertial>
  </link>
  <link name="leg1_femur">
    <inertial>
      <mass value="0.12"/>
      <inertia ixx="0.0003" ixy="0.0" ixz="0.0" iyy="0.0002" iyz="0.0" izz="0.0003"/>
      <origin xyz="0 0 0.06"/>
    </inertial>
  </link>
  <link name="leg1_tibia">
    <inertial>
      <mass value="0.10"/>
      <inertia ixx="0.0002" ixy="0.0" ixz="0.0" iyy="0.0002" iyz="0.0" izz="0.0001"/>
      <origin xyz="0 0 0.08"/>
    </inertial>
  </link>
  <joint name="joint1_1" type="revolute">
    <parent link="base_link"/>
    <child link="leg1_coxa"/>
    <origin xyz="0.12 0.08 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.785" upper="0.785"/>
  </joint>
  <joint name="joint1_2" type="revolute">
    <parent link="leg1_coxa"/>
    <child link="leg1_femur"/>
    <origin xyz="0.04 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.524" upper="0.524"/>
  </joint>
  <joint name="joint1_3" type="revolute">
    <parent link="leg1_femur"/>
    <child link="leg1_tibia"/>
    <origin xyz="0 0 0.12"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.047" upper="0.785"/>
  </joint>
</robot>"""
    return xml


def demo_kinematics_and_contact():
    print("\n" + "=" * 60)
    print("[Demo 1] 机器人运动学与接触几何")
    print("=" * 60)


    dh = np.array([
        [0.0,   0.0, 0.04,  np.pi / 2],
        [0.0,   0.0, 0.12,  0.0],
        [0.0,   0.0, 0.12,  0.0],
    ])
    limits = np.array([
        [-np.pi / 4, np.pi / 4],
        [-np.pi / 6, np.pi / 6],
        [-np.pi / 3, np.pi / 2 * 0.75],
    ])

    leg = SerialLegKinematics(dh, limits)
    q_test = np.array([0.1, -0.2, 0.3])
    T_ee, transforms = leg.forward_kinematics(q_test)
    print(f"正向运动学：关节角 q = {q_test}")
    print(f"足端位姿 T_ee = \n{T_ee}")

    J = leg.jacobian(q_test)
    print(f"Jacobian = \n{J}")


    target = np.array([0.15, 0.05, -0.18])
    q_inv = leg.inverse_kinematics_numerical(target, q_test)
    T_check, _ = leg.forward_kinematics(q_inv)
    err_inv = np.linalg.norm(T_check[:3, 3] - target)
    print(f"逆运动学：目标 = {target}")
    print(f"求解关节角 q_inv = {q_inv}")
    print(f"逆解误差 = {err_inv:.6e}")


    jlc = JointLimitConstraint(limits, safety_margin=0.05)
    dist = jlc.geodesic_distance_to_limit(q_inv)
    print(f"关节到限位测地距离 = {dist}")
    grad = jlc.penalty_gradient(q_inv)
    print(f"限位惩罚梯度 = {grad}")


    foot = FootContactGeometry(mu=0.8, contact_radius=0.02)
    f_test = np.array([5.0, 2.0, 20.0])
    n_test = np.array([0.0, 0.0, 1.0])
    residual = foot.friction_cone_residual(f_test, n_test)
    print(f"接触力摩擦锥违反量 = {residual:.6f} (负值=在锥内)")
    tau_contact = foot.contact_moment(f_test, np.array([0.01, 0.0, 0.0]))
    print(f"接触力矩 = {tau_contact}")


    centers, radii = gershgorin_discs(J @ J.T)
    print(f"Jacobian^T·J 的 Gershgorin 圆盘中心 = {centers}")
    print(f"圆盘半径 = {radii}")


def demo_terrain_and_mesh():
    print("\n" + "=" * 60)
    print("[Demo 2] 地形建模与足部接触面")
    print("=" * 60)

    terrain = generate_sample_terrain()
    print(f"示例地形：顶点数 = {len(terrain.vertices)}, 面片数 = {len(terrain.faces)}")
    print(f"地形 AABB = [{terrain.aabb_min}, {terrain.aabb_max}]")


    xq, yq = 0.0, 0.0
    z, normal, face_idx = terrain.query_height(xq, yq)
    print(f"查询点 ({xq}, {yq}) -> 高度 z = {z:.6f}, 法向量 = {normal}, 面片 = {face_idx}")


    quad_nodes = np.array([
        [-0.5, -0.5, 0.0],
        [0.5, -0.5, 0.0],
        [0.5, 0.5, 0.1],
        [-0.5, 0.5, 0.1],
    ])
    quad = QuadrilateralTerrainPatch(quad_nodes)
    p_mid, J_mid = quad.bilinear_interpolate(0.5, 0.5)
    n_mid = quad.normal_at(0.5, 0.5)
    print(f"四边形中心插值点 = {p_mid}")
    print(f"Jacobian = \n{J_mid}")
    print(f"法向量 = {n_mid}")


    tmp_dir = os.path.dirname(os.path.abspath(__file__))
    node_file = os.path.join(tmp_dir, "tmp_terrain.node")
    elem_file = os.path.join(tmp_dir, "tmp_terrain.ele")
    with open(node_file, 'w') as f:
        f.write(f"{len(terrain.vertices)} 3 0 0\n")
        for i, v in enumerate(terrain.vertices):
            f.write(f"{i+1} {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
    with open(elem_file, 'w') as f:
        f.write(f"{len(terrain.faces)} 3 0\n")
        for i, face in enumerate(terrain.faces):
            f.write(f"{i+1} {face[0]+1} {face[1]+1} {face[2]+1}\n")

    terrain2 = TriangulatedTerrain.from_node_element(node_file, elem_file)
    print(f"从文件读取的地形：顶点数 = {len(terrain2.vertices)}, 面片数 = {len(terrain2.faces)}")


    os.remove(node_file)
    os.remove(elem_file)


    stl_file = os.path.join(tmp_dir, "tmp_terrain.stl")
    with open(stl_file, 'w') as f:
        f.write("solid test\n")
        f.write("facet normal 0 0 1\n")
        f.write("  outer loop\n")
        f.write("    vertex 0 0 0\n")
        f.write("    vertex 1 0 0\n")
        f.write("    vertex 0 1 0\n")
        f.write("  endloop\n")
        f.write("endfacet\n")
        f.write("endsolid test\n")
    terrain_stl = TriangulatedTerrain.from_stl_ascii(stl_file)
    print(f"从 STL 读取的地形：顶点数 = {len(terrain_stl.vertices)}, 面片数 = {len(terrain_stl.faces)}")
    os.remove(stl_file)


def demo_gait_dynamics():
    print("\n" + "=" * 60)
    print("[Demo 3] 步态动力学与 CPG 控制")
    print("=" * 60)











    raise NotImplementedError("Hole 3: 请补全 demo_gait_dynamics 的实现")


def demo_numerical_solvers():
    print("\n" + "=" * 60)
    print("[Demo 4] 大规模数值线性代数")
    print("=" * 60)


    A_chol = np.array([
        [4.0, 2.0, 1.0],
        [2.0, 5.0, 2.0],
        [1.0, 2.0, 3.0],
    ], dtype=float)
    chol = CholeskySolver(eps=1e-13)
    L = chol.decompose(A_chol)
    print(f"Cholesky 分解 L = \n{L}")
    print(f"验证 L·L^T = \n{L @ L.T}")
    b_chol = np.array([1.0, 2.0, 3.0])
    x_chol = chol.solve(A_chol, b_chol)
    print(f"Cholesky 求解 x = {x_chol}")
    print(f"残差 ||Ax-b|| = {np.linalg.norm(A_chol @ x_chol - b_chol):.6e}")


    block_solver = BlockTridiagonalSolver(block_size=3)
    N_blocks = 4
    lower = [np.eye(3) * 0.1 for _ in range(N_blocks - 1)]
    diag = [np.diag([2.0 + i * 0.5, 2.0 + i * 0.5, 2.0 + i * 0.5]) for i in range(N_blocks)]
    upper = [np.eye(3) * 0.1 for _ in range(N_blocks - 1)]
    rhs_bt = [np.ones(3) * (i + 1) for i in range(N_blocks)]
    x_bt = block_solver.solve(lower, diag, upper, rhs_bt)
    print(f"块三对角求解结果：")
    for i, xi in enumerate(x_bt):
        print(f"  x_{i} = {xi}")


    fft_solver = Radix2FFT()
    dt = 0.01
    t_signal = np.arange(0, 2.0, dt)
    signal = np.sin(2.0 * np.pi * 5.0 * t_signal) + 0.5 * np.sin(2.0 * np.pi * 12.0 * t_signal)

    N_pow2 = 2 ** int(np.ceil(np.log2(len(signal))))
    signal_padded = np.zeros(N_pow2)
    signal_padded[:len(signal)] = signal
    freqs, psd = fft_solver.power_spectral_density(signal_padded, dt)
    peak_idx = np.argmax(psd[:N_pow2 // 2])
    print(f"FFT 功率谱密度峰值频率 ≈ {freqs[peak_idx]:.2f} Hz")


    vand = VandermondeSolver()
    x_nodes = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b_vand = np.array([2.0, 5.0, 10.0, 17.0, 26.0])
    coeffs, info = vand.solve(x_nodes, b_vand)
    print(f"Vandermonde 求解 info = {info}, 系数 = {coeffs}")
    p_eval = vand.evaluate(x_nodes, coeffs, np.array([2.5]))
    print(f"插值验证 p(2.5) = {p_eval[0]:.6f} (理论 7.25)")


    A_mm = np.random.rand(64, 48)
    B_mm = np.random.rand(48, 32)
    C_mm = MatrixMultiplyBenchmark.multiply(A_mm, B_mm)
    print(f"矩阵乘法 C = A·B 结果形状 = {C_mm.shape}")


def demo_trajectory_planning():
    print("\n" + "=" * 60)
    print("[Demo 5] 足端轨迹规划")
    print("=" * 60)


    candidates = np.array([
        [0.12, 0.08, 0.0],
        [0.15, 0.05, 0.0],
        [0.10, 0.10, 0.0],
        [0.18, 0.09, 0.0],
        [0.14, 0.12, 0.0],
    ])
    tsp = TSPBruteForce()
    best_perm, min_cost, avg_cost, max_cost = tsp.solve(candidates)
    print(f"候选落点 TSP 最优排序 = {best_perm}")
    print(f"最短路径 = {min_cost:.6f}, 平均 = {avg_cost:.6f}, 最长 = {max_cost:.6f}")


    traj = PolynomialSwingTrajectory()
    coeffs = traj.fit_quintic(
        T=0.3,
        p_start=0.0, p_end=0.15,
        v_start=0.0, v_end=0.0,
        a_start=0.0, a_end=0.0
    )
    print(f"5次多项式系数 = {coeffs}")
    p_mid, v_mid, a_mid = traj.evaluate(coeffs, 0.15)
    print(f"t=0.15: p={p_mid:.6f}, v={v_mid:.6f}, a={a_mid:.6f}")


    planner = FootfallPlanner(swing_height=0.05, swing_period=0.3)
    perm, sorted_pts = planner.plan_footholds(candidates)
    print(f"FootfallPlanner 排序结果 = {perm}")
    swing_traj = planner.generate_swing_trajectory(
        p_start=np.array([0.0, 0.0, 0.0]),
        p_end=np.array([0.15, 0.05, 0.0]),
        n_samples=10
    )
    print(f"摆动轨迹形状 = {swing_traj.shape}")
    print(f"轨迹首点 = {swing_traj[0]}, 末点 = {swing_traj[-1]}")


def demo_stability_analysis():
    print("\n" + "=" * 60)
    print("[Demo 6] 稳定性分析与约束优化")
    print("=" * 60)


    foot_positions = np.array([
        [0.15, 0.10],
        [0.18, -0.08],
        [-0.05, 0.12],
        [-0.10, -0.05],
        [0.05, 0.15],
        [0.02, -0.12],
    ])
    support = SupportPolygon(foot_positions)
    print(f"支撑多边形凸包顶点数 = {len(support.hull)}")

    com = np.array([0.05, 0.02])
    inside = support.contains_point(com)
    margin = support.distance_to_boundary(com)
    print(f"COM = {com}, 在支撑多边形内？ {inside}, 边界距离 = {margin:.6f}")


    stab = StabilityMargin(robot_mass=5.0)
    com_3d = np.array([0.05, 0.02, 0.12])
    a_com = np.array([0.1, 0.05, 0.0])
    zmp = stab.zmp_position(com_3d, a_com, np.zeros(3), np.zeros((6, 3)), foot_positions)
    print(f"ZMP 位置 = {zmp}")


    graph = SupportGraphCentrality(n_legs=6)
    stance_state = np.array([1, 1, 0, 1, 0, 1])
    coupling = np.ones((6, 6)) * 0.5 + np.eye(6) * 1.0
    M = graph.build_transition_matrix(stance_state, coupling)
    pr = graph.pagerank(M)
    print(f"支撑图 PageRank 中心性 = {pr}")
    print(f"最稳定腿索引 = {np.argmax(pr)}")


    lp = LinearStabilityConstraint()
    A_cons, b_cons = lp.com_feasible_region(support, margin=0.02)
    print(f"COM 可行域约束矩阵 A 形状 = {A_cons.shape}")
    print(f"约束满足性 A·com ≤ b: {np.all(A_cons @ com <= b_cons)}")


def demo_chaotic_optimization():
    print("\n" + "=" * 60)
    print("[Demo 7] 混沌全局步态参数优化")
    print("=" * 60)


    def fake_stability(params):
        T, stride, h, k, d = params

        return 0.05 + 0.1 * np.sin(T * 3) * np.cos(stride * 10) - 0.01 * d

    def fake_energy(params):
        T, stride, h, k, d = params
        return 0.5 * stride ** 2 + 2.0 * h ** 2 + 0.1 * k + 0.2 * d ** 2

    optimizer = GaitParameterOptimizer()
    x_opt, f_opt = optimizer.optimize(fake_stability, fake_energy)
    print(f"优化后步态参数：")
    print(f"  T_gait       = {x_opt[0]:.4f} s")
    print(f"  stride_length= {x_opt[1]:.4f} m")
    print(f"  swing_height = {x_opt[2]:.4f} m")
    print(f"  coupling_k   = {x_opt[3]:.4f}")
    print(f"  damping      = {x_opt[4]:.4f}")
    print(f"  最优适应度   = {f_opt:.6f}")


def demo_config_parsing():
    print("\n" + "=" * 60)
    print("[Demo 8] 机器人配置 XML 解析")
    print("=" * 60)

    xml = build_default_robot_config()
    config = parse_robot_config(xml)
    links = extract_link_params(config)
    print(f"解析到的 link 数量 = {len(links)}")
    for name, params in links.items():
        print(f"  {name}: mass={params['mass']:.3f}kg, com={params['com']}, "
              f"I=({params['inertia']['ixx']:.4f}, {params['inertia']['iyy']:.4f}, {params['inertia']['izz']:.4f})")


def run_full_pipeline():
    timer = Timer()
    print("\n" + "#" * 60)
    print("# 六足机器人多足步态优化系统 —— 完整演示")
    print("#" * 60)

    demo_config_parsing()
    demo_kinematics_and_contact()
    demo_terrain_and_mesh()
    demo_gait_dynamics()
    demo_numerical_solvers()
    demo_trajectory_planning()
    demo_stability_analysis()
    demo_chaotic_optimization()

    elapsed = timer.elapsed()
    print("\n" + "#" * 60)
    print(f"# 全部演示完成，总耗时 = {elapsed:.4f} 秒")
    print("#" * 60)


    print("\n[全局数值鲁棒性检查]")
    test_mats = [
        np.eye(5),
        np.diag([1.0, 1e-10, 1.0]),
        np.array([[1.0, 2.0], [2.0, 4.0]]),
    ]
    for i, M in enumerate(test_mats):
        is_sing = check_numerical_singularity(M)
        print(f"  测试矩阵 {i+1}: 奇异/近奇异 = {is_sing}")

    print("\n[INFO] 所有模块运行完毕，无报错。")


if __name__ == "__main__":
    np.random.seed(42)
    run_full_pipeline()
