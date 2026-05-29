"""
main.py
六足机器人多足步态优化系统 —— 统一入口。

本项目基于 15 个种子项目的核心算法，围绕"机器人学：多足机器人步态优化"
领域，融合构造了一个博士级科研计算系统。

运行方式：
    python main.py
无需任何参数，程序将自动执行完整的步态优化流程并输出结果。
"""

import numpy as np
import os
import sys

# 确保模块路径正确
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
    """
    构建默认六足机器人 URDF-like XML 配置字符串（供 xml2struct 解析）。
    """
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
    """
    演示模块1：运动学正解、逆解、接触几何与关节限位约束。
    """
    print("\n" + "=" * 60)
    print("[Demo 1] 机器人运动学与接触几何")
    print("=" * 60)

    # DH 参数: [theta_offset, d, a, alpha] (Craig 约定)
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

    # 数值逆运动学
    target = np.array([0.15, 0.05, -0.18])
    q_inv = leg.inverse_kinematics_numerical(target, q_test)
    T_check, _ = leg.forward_kinematics(q_inv)
    err_inv = np.linalg.norm(T_check[:3, 3] - target)
    print(f"逆运动学：目标 = {target}")
    print(f"求解关节角 q_inv = {q_inv}")
    print(f"逆解误差 = {err_inv:.6e}")

    # 关节限位约束
    jlc = JointLimitConstraint(limits, safety_margin=0.05)
    dist = jlc.geodesic_distance_to_limit(q_inv)
    print(f"关节到限位测地距离 = {dist}")
    grad = jlc.penalty_gradient(q_inv)
    print(f"限位惩罚梯度 = {grad}")

    # 接触几何
    foot = FootContactGeometry(mu=0.8, contact_radius=0.02)
    f_test = np.array([5.0, 2.0, 20.0])
    n_test = np.array([0.0, 0.0, 1.0])
    residual = foot.friction_cone_residual(f_test, n_test)
    print(f"接触力摩擦锥违反量 = {residual:.6f} (负值=在锥内)")
    tau_contact = foot.contact_moment(f_test, np.array([0.01, 0.0, 0.0]))
    print(f"接触力矩 = {tau_contact}")

    # Gershgorin 圆盘分析 Jacobian 条件
    centers, radii = gershgorin_discs(J @ J.T)
    print(f"Jacobian^T·J 的 Gershgorin 圆盘中心 = {centers}")
    print(f"圆盘半径 = {radii}")


def demo_terrain_and_mesh():
    """
    演示模块2：地形三角网格、四边形插值、STL 解析。
    """
    print("\n" + "=" * 60)
    print("[Demo 2] 地形建模与足部接触面")
    print("=" * 60)

    terrain = generate_sample_terrain()
    print(f"示例地形：顶点数 = {len(terrain.vertices)}, 面片数 = {len(terrain.faces)}")
    print(f"地形 AABB = [{terrain.aabb_min}, {terrain.aabb_max}]")

    # 查询高度
    xq, yq = 0.0, 0.0
    z, normal, face_idx = terrain.query_height(xq, yq)
    print(f"查询点 ({xq}, {yq}) -> 高度 z = {z:.6f}, 法向量 = {normal}, 面片 = {face_idx}")

    # 四边形地形补丁
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

    # 写入临时三角网格文件并读取（测试 triangle_io 功能）
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

    # 清理临时文件
    os.remove(node_file)
    os.remove(elem_file)

    # STL 解析测试（生成一个简单 ASCII STL 字符串）
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
    """
    演示模块3：CPG 网络、梯形法 ODE 积分、支撑-摆动自动机、单腿动力学。
    """
    print("\n" + "=" * 60)
    print("[Demo 3] 步态动力学与 CPG 控制")
    print("=" * 60)

    # CPG 网络初始化
    cpg = CPGNetwork(n_osc=6, alpha=50.0, mu=1.0, omega=2.0 * np.pi * 1.0, coupling_strength=5.0)
    state0 = np.zeros(12)
    state0[0] = 1.0  # 初始扰动

    # 梯形法积分 CPG
    integrator = TrapezoidalIntegrator(it_max=10)
    t, y = integrator.integrate(cpg.rhs, (0.0, 3.0), state0, n_steps=300)
    phase_final = cpg.extract_phase(y[-1])
    amp_final = cpg.extract_amplitude(y[-1])
    print(f"CPG 积分完成：t ∈ [0, 3], 步数 = 300")
    print(f"最终相位 = {phase_final}")
    print(f"最终振幅 = {amp_final}")

    # 支撑-摆动自动机
    ssa = StanceSwingAutomaton(n_legs=6, stance_min_steps=3)
    states_over_time = []
    for i in range(y.shape[0]):
        phase = cpg.extract_phase(y[i])
        s = ssa.update(phase)
        states_over_time.append(s.copy())
    states = np.array(states_over_time)
    print(f"支撑-摆动状态序列形状 = {states.shape}")
    print(f"最终支撑腿索引 = {np.where(states[-1] == 1)[0]}")

    # 单腿动力学仿真
    M_leg = np.diag([0.08, 0.12, 0.10])
    C_leg = np.diag([0.5, 0.5, 0.5])
    leg_dyn = LegDynamics(M_leg, C_leg, gravity=9.81)
    q_leg = np.array([0.1, -0.1, 0.2])
    dq_leg = np.array([0.5, -0.3, 0.1])
    tau_leg = np.array([0.2, -0.1, 0.05])
    f_contact = np.array([2.0, 1.0, 15.0])
    J_leg = np.array([
        [-0.08, -0.12, -0.12],
        [0.15, 0.05, -0.03],
        [0.0, 0.12, 0.12],
    ])
    ddq = leg_dyn.dynamics(q_leg, dq_leg, tau_leg, f_contact, J_leg)
    print(f"单腿动力学：q̈ = {ddq}")

    # 状态空间 rhs 演示
    state_leg = np.concatenate((q_leg, dq_leg))

    def J_func(q):
        return J_leg  # 简化：常数 Jacobian

    rhs_leg = leg_dyn.state_space_rhs(state_leg, tau_leg, f_contact, J_func)
    print(f"状态空间 rhs = {rhs_leg}")


def demo_numerical_solvers():
    """
    演示模块4：Cholesky 分解、块三对角求解、FFT、Vandermonde、矩阵乘法。
    """
    print("\n" + "=" * 60)
    print("[Demo 4] 大规模数值线性代数")
    print("=" * 60)

    # Cholesky 分解
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

    # 块三对角求解
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

    # FFT 功率谱密度
    fft_solver = Radix2FFT()
    dt = 0.01
    t_signal = np.arange(0, 2.0, dt)
    signal = np.sin(2.0 * np.pi * 5.0 * t_signal) + 0.5 * np.sin(2.0 * np.pi * 12.0 * t_signal)
    # 补零到 2 的幂
    N_pow2 = 2 ** int(np.ceil(np.log2(len(signal))))
    signal_padded = np.zeros(N_pow2)
    signal_padded[:len(signal)] = signal
    freqs, psd = fft_solver.power_spectral_density(signal_padded, dt)
    peak_idx = np.argmax(psd[:N_pow2 // 2])
    print(f"FFT 功率谱密度峰值频率 ≈ {freqs[peak_idx]:.2f} Hz")

    # Vandermonde 求解
    vand = VandermondeSolver()
    x_nodes = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b_vand = np.array([2.0, 5.0, 10.0, 17.0, 26.0])  # 对应 p(x) = x^2 + 1
    coeffs, info = vand.solve(x_nodes, b_vand)
    print(f"Vandermonde 求解 info = {info}, 系数 = {coeffs}")
    p_eval = vand.evaluate(x_nodes, coeffs, np.array([2.5]))
    print(f"插值验证 p(2.5) = {p_eval[0]:.6f} (理论 7.25)")

    # 矩阵乘法基准
    A_mm = np.random.rand(64, 48)
    B_mm = np.random.rand(48, 32)
    C_mm = MatrixMultiplyBenchmark.multiply(A_mm, B_mm)
    print(f"矩阵乘法 C = A·B 结果形状 = {C_mm.shape}")


def demo_trajectory_planning():
    """
    演示模块5：TSP 落点排序、多项式摆动轨迹。
    """
    print("\n" + "=" * 60)
    print("[Demo 5] 足端轨迹规划")
    print("=" * 60)

    # TSP 落点排序
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

    # 多项式摆动轨迹
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

    # 综合足端轨迹规划
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
    """
    演示模块6：支撑多边形、ZMP、支撑图中心性、LP 约束。
    """
    print("\n" + "=" * 60)
    print("[Demo 6] 稳定性分析与约束优化")
    print("=" * 60)

    # 支撑多边形
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

    # ZMP
    stab = StabilityMargin(robot_mass=5.0)
    com_3d = np.array([0.05, 0.02, 0.12])
    a_com = np.array([0.1, 0.05, 0.0])
    zmp = stab.zmp_position(com_3d, a_com, np.zeros(3), np.zeros((6, 3)), foot_positions)
    print(f"ZMP 位置 = {zmp}")

    # 支撑图中心性（PageRank）
    graph = SupportGraphCentrality(n_legs=6)
    stance_state = np.array([1, 1, 0, 1, 0, 1])
    coupling = np.ones((6, 6)) * 0.5 + np.eye(6) * 1.0
    M = graph.build_transition_matrix(stance_state, coupling)
    pr = graph.pagerank(M)
    print(f"支撑图 PageRank 中心性 = {pr}")
    print(f"最稳定腿索引 = {np.argmax(pr)}")

    # LP 约束
    lp = LinearStabilityConstraint()
    A_cons, b_cons = lp.com_feasible_region(support, margin=0.02)
    print(f"COM 可行域约束矩阵 A 形状 = {A_cons.shape}")
    print(f"约束满足性 A·com ≤ b: {np.all(A_cons @ com <= b_cons)}")


def demo_chaotic_optimization():
    """
    演示模块7：混沌全局优化步态参数。
    """
    print("\n" + "=" * 60)
    print("[Demo 7] 混沌全局步态参数优化")
    print("=" * 60)

    # 模拟的稳定性与能量函数（用于演示）
    def fake_stability(params):
        T, stride, h, k, d = params
        # 随机的但确定性的映射
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
    """
    演示模块8：XML 配置解析。
    """
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
    """
    执行完整计算流程，包含计时与数值鲁棒性检查。
    """
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

    # 全局数值检查
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

# ================================================================
# 测试用例（40个，assert模式，涉及随机值均使用固定种子）
# ================================================================
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

print('\n全部 40 个测试通过!\n')
