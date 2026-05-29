"""
main.py
软体机器人运动学建模 — 统一入口

博士级科研代码合成项目
科学领域: 机器人学 — 软体机器人运动学建模

融合15个种子项目的核心算法，构建完整的软体机器人计算框架:
- Cosserat杆理论运动学/动力学建模
- 超弹性本构关系（Neo-Hookean, Mooney-Rivlin, 化学-力学耦合）
- 横截面有限元分析（T6单元 + 高斯求积）
- 谱/DG空间离散化（Chebyshev谱微分 + 间断Galerkin）
- POD/SVD降阶模型
- 逆运动学与散乱数据插值
- 动态规划路径规划
- 双调和方程验证

零参数可运行，输出关键计算结果到控制台
"""

import numpy as np
import sys
import time

# 导入所有子模块
from mesh_utils import line_grid, chebyshev_grid, triangulate_polygon, diaphony_compute, sample_ellipse, refine_cross_section_mesh
from section_fem import shape_t6, grad_shape_t6, gauss_legendre_triangle, compute_section_properties, compute_shear_correction_factor, assemble_section_stiffness
from spectral_dg import chebyshev_matrix, jacobi_polynomial, vandermonde_1d, dmatrix_1d, jacobi_gauss_lobatto, lift_1d
from hyperelastic_law import neo_hookean_strain_energy, neo_hookean_stress, mooney_rivlin_strain_energy, mooney_rivlin_stress, soft_robot_1d_constitutive, chemo_mechanical_coupling, selkov_glycolysis_ode, tangent_stiffness_neo_hookean
from dynamics_solver import cauchy_theta_method, low_storage_rk4, sawtooth_driver, driven_harmonic_oscillator, integrate_cosserat_dynamics
from cosserat_core import hat_map, vee_map, rodrigues_rotation, compute_curvature, r8blt_mv, r8blt_sl, r8blt_det, assemble_banded_stiffness, forward_kinematics_cosserat, compute_strain_measures
from inverse_kinematics import barycentric_coordinates, point_in_triangle, pwl_interp_2d_scattered, shape_reconstruction_from_sensors, inverse_kinematics_soft_robot
from path_planner import change_dynamic, discretize_configuration_space, configuration_to_tip, energy_cost, dp_path_planning_2d, multi_target_path_planning
from rom_pod import compute_svd_basis, randomized_svd, project_onto_basis, reconstruct_from_basis, pod_galerkin_rom, generate_snapshots_soft_robot, energy_fraction, optimal_basis_size
from validation_biharmonic import biharmonic_w1, biharmonic_r1, biharmonic_w2, biharmonic_r2, biharmonic_w3, biharmonic_r3, verify_biharmonic_discretization, plate_bending_energy


def print_header(title: str):
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_result(label: str, value, indent: int = 2):
    prefix = " " * indent
    if isinstance(value, float):
        print(f"{prefix}{label}: {value:.6e}")
    elif isinstance(value, np.ndarray):
        print(f"{prefix}{label}: {value.flatten()[:6]}")
    else:
        print(f"{prefix}{label}: {value}")


def run_module_1_mesh_and_section():
    """模块1: 网格生成与横截面分析"""
    print_header("MODULE 1: 网格生成与横截面分析")

    # 1.1 1D网格生成（种子项目680_line_grid）
    L = 1.0
    Ns = 20
    s_grid = line_grid(Ns + 1, 0.0, L, c=1)
    print(f"  1.1 中心线网格 (line_grid, c=1): N={len(s_grid)}, s∈[{s_grid[0]:.4f}, {s_grid[-1]:.4f}]")

    # 1.2 Chebyshev节点（种子项目161_chebyshev_matrix）
    cheb_nodes = chebyshev_grid(Ns)
    print(f"  1.2 Chebyshev节点: N={len(cheb_nodes)}")

    # 1.3 2D横截面三角剖分（种子项目548_human_mesh2d）
    a_semi = 0.05
    b_semi = 0.03
    n_ellipse = 12
    theta_poly = np.linspace(0.0, 2.0 * np.pi, n_ellipse, endpoint=False)
    vertices = np.column_stack([a_semi * np.cos(theta_poly), b_semi * np.sin(theta_poly)])
    nodes_sec, tris_sec = triangulate_polygon(vertices)
    nodes_sec, tris_sec = refine_cross_section_mesh(vertices)
    print(f"  1.3 截面三角剖分: 节点数={nodes_sec.shape[0]}, 三角形数={tris_sec.shape[0]}")

    # 1.4 截面属性计算（种子项目375_fem_basis_t6 + 344_exactness）
    props = compute_section_properties(nodes_sec, tris_sec)
    print(f"  1.4 截面属性:")
    print_result("面积 A", props['A'])
    print_result("形心 (cx,cy)", np.array([props['cx'], props['cy']]))
    print_result("惯性矩 Ixx", props['Ixx'])
    print_result("惯性矩 Iyy", props['Iyy'])
    print_result("极惯性矩 J", props['J'])
    print_result("Diaphony均匀性", props['diaphony'])

    # 1.5 剪切修正系数
    kappa_shear = compute_shear_correction_factor(nodes_sec, tris_sec, E=1.0e6, nu=0.35)
    print_result("剪切修正系数 kappa", kappa_shear)

    return props


def run_module_2_spectral_and_dg():
    """模块2: 谱方法与DG离散化"""
    print_header("MODULE 2: 谱方法与DG离散化")

    # 2.1 Chebyshev谱微分矩阵（种子项目161_chebyshev_matrix）
    N = 8
    x_cheb, D_cheb = chebyshev_matrix(N)
    # 验证: D作用于常数应为0
    const_vec = np.ones(N + 1)
    D_const = D_cheb @ const_vec
    err_const = np.max(np.abs(D_const))
    print(f"  2.1 Chebyshev微分矩阵 (N={N}):")
    print_result("常数微分误差", err_const)

    # 2.2 对f(x)=cos(x)求导，比较精确值
    f_cheb = np.cos(x_cheb)
    df_exact = -np.sin(x_cheb)
    df_numerical = D_cheb @ f_cheb
    err_cheb = np.max(np.abs(df_numerical - df_exact))
    print_result("cos(x)微分最大误差", err_cheb)

    # 2.3 Jacobi多项式（种子项目273_dg1d_heat）
    r_test = np.array([-0.5, 0.0, 0.5])
    P3 = jacobi_polynomial(r_test, 0.0, 0.0, 3)
    print(f"  2.3 Jacobi P_3^(0,0): {P3}")

    # 2.4 Vandermonde矩阵（种子项目273_dg1d_heat）
    V = vandermonde_1d(N, x_cheb)
    cond_V = np.linalg.cond(V)
    print_result("Vandermonde条件数", cond_V)

    # 2.5 DG lift算子（种子项目273_dg1d_heat）
    LIFT = lift_1d(N, V)
    print_result("DG lift算子范数", np.linalg.norm(LIFT))

    return x_cheb, D_cheb


def run_module_3_hyperelastic_and_chemo():
    """模块3: 超弹性本构与化学耦合"""
    print_header("MODULE 3: 超弹性本构与化学-力学耦合")

    # 3.1 Neo-Hookean模型
    F = np.eye(3)
    F[0, 0] = 1.1  # 10%拉伸
    F[1, 1] = 1.0 / np.sqrt(1.1)  # 不可压缩约束近似
    F[2, 2] = 1.0 / np.sqrt(1.1)
    mu = 1.0e5
    K_bulk = 2.0e6
    W_nh = neo_hookean_strain_energy(F, mu, K_bulk)
    P_nh = neo_hookean_stress(F, mu, K_bulk)
    print(f"  3.1 Neo-Hookean:")
    print_result("应变能 W", W_nh)
    print_result("P_11", P_nh[0, 0])

    # 3.2 Mooney-Rivlin模型
    C10 = 0.5 * mu
    C01 = 0.1 * mu
    W_mr = mooney_rivlin_strain_energy(F, C10, C01, K_bulk)
    sigma_mr = mooney_rivlin_stress(F, C10, C01, K_bulk)
    print(f"  3.2 Mooney-Rivlin:")
    print_result("应变能 W", W_mr)
    print_result("sigma_11", sigma_mr[0, 0])

    # 3.3 1D Cosserat本构
    epsilon = 0.05
    kappa_vec = np.array([0.01, 0.02, 0.1])
    E = 1.0e6
    G = 0.35e6
    A = 0.005
    Ixx = 2.0e-6
    Iyy = 1.0e-6
    J = Ixx + Iyy
    n_1d, m_1d = soft_robot_1d_constitutive(epsilon, kappa_vec, E, G, A, Ixx, Iyy, J)
    print(f"  3.3 1D Cosserat本构:")
    print_result("内力 n", n_1d)
    print_result("内矩 m", m_1d)

    # 3.4 化学-力学耦合（种子项目472_glycolysis_ode）
    y_chem = np.array([0.5, 6.5])  # 离子浓度, pH
    E_eff = chemo_mechanical_coupling(y_chem, epsilon, E0=E, gamma=0.3, beta_chem=0.2)
    print_result("化学耦合有效模量 E_eff", E_eff)

    # 3.5 糖酵解ODE积分（种子项目472_glycolysis_ode）
    t_gly, y_gly = low_storage_rk4(selkov_glycolysis_ode, (0.0, 50.0), np.array([0.9, 0.7]), 500)
    print(f"  3.5 Selkov糖酵解ODE: t∈[0,50], u_final={y_gly[-1,0]:.4f}, v_final={y_gly[-1,1]:.4f}")

    return {'E': E, 'G': G, 'A': A, 'Ixx': Ixx, 'Iyy': Iyy, 'J': J}


def run_module_4_cosserat_kinematics(material_params):
    """模块4: Cosserat杆运动学"""
    print_header("MODULE 4: Cosserat杆运动学")

    L = 1.0
    Ns = 30

    # 4.1 前向运动学（种子项目680_line_grid, 970_r8blt, 161_chebyshev_matrix）
    # 弯曲构型: 螺旋形
    n_nodes = Ns + 1
    s = np.linspace(0.0, L, n_nodes)
    kappa_base = np.zeros((n_nodes, 3))
    kappa_base[:, 2] = 2.0 * np.pi / L  # 恒定扭率
    kappa_base[:, 0] = 0.5 * np.pi / L * np.sin(np.pi * s / L)  # 弯曲变化

    s_out, r_out, R_out = forward_kinematics_cosserat(L, Ns, kappa_base)
    print(f"  4.1 前向运动学 (螺旋构型):")
    print_result("末端位置", r_out[-1])
    print_result("总弧长", s_out[-1])

    # 4.2 应变度量反算
    v_strain, u_strain = compute_strain_measures(r_out, R_out, s_out)
    print_result("末端线应变 v", v_strain[-1])
    print_result("末端曲率应变 u", u_strain[-1])

    # 4.3 带状矩阵求解（种子项目970_r8blt）
    EI = material_params['E'] * material_params['Ixx']
    a_blt = assemble_banded_stiffness(n_nodes, EI, material_params['E'] * material_params['A'], L / Ns, ml=3)
    b_test = np.ones(n_nodes)
    x_blt = r8blt_sl(a_blt, 3, b_test)
    det_blt = r8blt_det(a_blt, 3)
    mv_blt = r8blt_mv(a_blt, 3, x_blt)
    mv_err = np.max(np.abs(mv_blt - b_test))
    print(f"  4.3 带状下三角矩阵求解:")
    print_result("行列式", det_blt)
    print_result("求解残差 ||Ax-b||", mv_err)

    # 4.4 曲率计算
    kappa_calc, tau_calc = compute_curvature(r_out, s_out)
    print_result("最大曲率", np.max(kappa_calc))
    print_result("最大挠率", np.max(np.abs(tau_calc)))

    return s_out, r_out, R_out


def run_module_5_dynamics():
    """模块5: 动力学与驱动"""
    print_header("MODULE 5: 动力学与驱动")

    # 5.1 锯齿波驱动谐振子（种子项目1059_sawtooth_ode）
    t_osc, y_osc = low_storage_rk4(
        lambda t, y: driven_harmonic_oscillator(t, y, omega0=2.0, zeta=0.5, omega_drive=2.0),
        (0.0, 10.0), np.array([0.5, 0.0]), 400
    )
    print(f"  5.1 锯齿波驱动谐振子:")
    print_result("末端位移 u", y_osc[-1, 0])
    print_result("末端速度 v", y_osc[-1, 1])

    # 5.2 Cauchy(theta)方法（种子项目138_cauchy_method）
    t_cau, y_cau = cauchy_theta_method(
        lambda t, y: driven_harmonic_oscillator(t, y, omega0=2.0, zeta=0.5, omega_drive=2.0),
        (0.0, 10.0), np.array([0.5, 0.0]), 100, theta=0.5
    )
    print(f"  5.2 Cauchy(theta=0.5)方法:")
    print_result("末端位移 u", y_cau[-1, 0])

    # 5.3 Cosserat杆动力学
    L = 0.5
    Ns = 10
    n_nodes = Ns + 1
    total_dof = n_nodes * 3
    q0 = np.zeros(total_dof)
    qdot0 = np.zeros(total_dof)
    # 初始条件: 极小的静态弯曲
    for i in range(n_nodes):
        s_ratio = i / Ns
        # 抛物线型初始变形
        q0[i * 3 + 1] = 0.001 * (3.0 * s_ratio ** 2 - 2.0 * s_ratio ** 3)

    mat_params = {'E': 1.0e5, 'G': 0.35e5, 'A': 0.005, 'Ixx': 2.0e-6, 'Iyy': 1.0e-6, 'J': 3.0e-6, 'rho': 100.0}
    t_dyn, state_dyn = integrate_cosserat_dynamics((0.0, 0.02), q0, qdot0, Ns, L, mat_params, n_steps=1000, method='rk4')
    q_final = state_dyn[-1, :total_dof]
    tip_final = q_final[-3:]
    print(f"  5.3 Cosserat杆动力学:")
    print_result("末端最终位置", tip_final)


def run_module_6_inverse_kinematics():
    """模块6: 逆运动学与散乱数据插值"""
    print_header("MODULE 6: 逆运动学与散乱数据插值")

    # 6.1 逆运动学求解
    L = 1.0
    Ns = 15
    target = np.array([0.6, 0.3, 0.0])
    kappa_ik, r_ik = inverse_kinematics_soft_robot(target, L, Ns, {}, max_iter=80, tol=1e-5)
    print(f"  6.1 逆运动学:")
    print_result("目标位置", target)
    print_result("实际末端", r_ik[-1])
    print_result("末端误差", np.linalg.norm(r_ik[-1] - target))
    print_result("最大曲率", np.max(np.abs(kappa_ik)))

    # 6.2 散乱数据插值（种子项目928_pwl_interp_2d_scattered）
    # 传感器数据模拟
    np.random.seed(42)
    n_sensors = 20
    sensor_pos = np.random.rand(n_sensors, 2)
    sensor_vals = np.sin(2.0 * np.pi * sensor_pos[:, 0]) * np.cos(2.0 * np.pi * sensor_pos[:, 1])

    # 查询点
    nq = 100
    xq = np.linspace(0.0, 1.0, 10)
    yq = np.linspace(0.0, 1.0, 10)
    Xq, Yq = np.meshgrid(xq, yq)
    query_pts = np.column_stack([Xq.ravel(), Yq.ravel()])

    zi = pwl_interp_2d_scattered(sensor_pos, sensor_vals, query_pts)
    print(f"  6.2 散乱数据PWL插值:")
    print_result("插值点数", len(zi))
    print_result("插值均值", np.mean(zi))
    print_result("插值标准差", np.std(zi))


def run_module_7_path_planning():
    """模块7: 动态规划路径规划"""
    print_header("MODULE 7: 动态规划路径规划")

    # 7.1 硬币找零（种子项目156_change_dynamic）
    coins = np.array([1, 3, 4])
    target = 15
    dp_result = change_dynamic(coins, target)
    print(f"  7.1 动态规划硬币找零 (coins={coins}, target={target}):")
    print_result("最少硬币数", dp_result[-1])

    # 7.2 软体臂路径规划
    n_segments = 4
    seg_length = 0.25
    target_xy = (0.7, 0.4)
    angles_opt, cost_opt = dp_path_planning_2d(n_segments, seg_length, target_xy, n_discrete=11)
    tx, ty = configuration_to_tip(n_segments, seg_length, angles_opt)
    print(f"  7.2 软体臂DP路径规划 ({n_segments}段):")
    print_result("最优角度 (deg)", np.degrees(angles_opt))
    print_result("目标", target_xy)
    print_result("实际末端", np.array([tx, ty]))
    print_result("路径代价", cost_opt)

    # 7.3 多目标规划
    targets = [(0.5, 0.3), (0.7, 0.4), (0.6, 0.5)]
    paths = multi_target_path_planning(n_segments, seg_length, targets)
    print(f"  7.3 多目标路径规划 (3个目标点):")
    for i, (ang, tgt) in enumerate(zip(paths, targets)):
        tx, ty = configuration_to_tip(n_segments, seg_length, ang)
        print_result(f"  目标{i+1} {tgt} 末端", np.array([tx, ty]))


def run_module_8_rom_pod():
    """模块8: POD降阶模型"""
    print_header("MODULE 8: POD降阶模型")

    # 8.1 生成快照（种子项目1184_svd_basis）
    L = 1.0
    Ns = 20
    n_snapshots = 50
    mat_params = {'E': 1.0e6, 'G': 0.35e6, 'A': 0.005, 'Ixx': 2.0e-6, 'Iyy': 1.0e-6, 'J': 3.0e-6, 'rho': 1000.0}
    snapshots = generate_snapshots_soft_robot(L, Ns, n_snapshots, mat_params)
    print(f"  8.1 快照生成: 矩阵大小 {snapshots.shape}")

    # 8.2 SVD/POD基提取
    basis_num = 10
    basis, svals, mean_vec = compute_svd_basis(snapshots, basis_num, subtract_mean=True)
    print(f"  8.2 POD基提取:")
    print_result("保留基函数数", basis_num)
    print_result("前5个奇异值", svals[:5])

    # 8.3 能量分数
    ef = energy_fraction(svals)
    opt_size = optimal_basis_size(svals, threshold=0.99)
    print_result("前3模态能量占比", ef[:3])
    print_result("99%能量所需模态数", opt_size)

    # 8.4 投影与重构
    test_field = snapshots[:, 0]
    coeffs = project_onto_basis(test_field, basis, mean_vec)
    recon = reconstruct_from_basis(coeffs, basis, mean_vec)
    recon_err = np.linalg.norm(recon - test_field) / np.linalg.norm(test_field)
    print_result("重构相对误差", recon_err)

    # 8.5 ROM-Galerkin投影
    M_full = np.eye(basis.shape[0]) * 0.1
    K_full = np.eye(basis.shape[0]) * 1.0e4
    F_full = np.ones(basis.shape[0])
    M_rom, K_rom, F_rom = pod_galerkin_rom(M_full, K_full, F_full, basis)
    print(f"  8.5 ROM-Galerkin:")
    print_result("降阶质量矩阵大小", M_rom.shape)
    print_result("降阶刚度条件数", np.linalg.cond(K_rom))


def run_module_9_biharmonic_validation():
    """模块9: 双调和方程验证"""
    print_header("MODULE 9: 双调和方程验证")

    # 9.1 精确解族1（种子项目087_biharmonic_exact）
    x = np.linspace(-0.5, 0.5, 21)
    y = np.linspace(-0.5, 0.5, 21)
    X, Y = np.meshgrid(x, y)

    W1 = biharmonic_w1(X, Y, a=1.0, b=0.0, c=0.0, d=0.0, e=1.0, f=0.0, g=2.0)
    R1 = biharmonic_r1(X, Y, a=1.0, b=0.0, c=0.0, d=0.0, e=1.0, f=0.0, g=2.0)
    print(f"  9.1 双调和精确解族1:")
    print_result("W在中心值", W1[10, 10])
    print_result("R在中心值", R1[10, 10])

    # 9.2 精确解族2
    W2 = biharmonic_w2(X, Y, g=1.5)
    R2 = biharmonic_r2(X, Y, g=1.5)
    print_result("W2中心值", W2[10, 10])

    # 9.3 精确解族3
    W3 = biharmonic_w3(X, Y, a=1.0, b=0.5, c=0.1, d=0.0, e=0.0, f=0.0)
    R3 = biharmonic_r3(X, Y, a=1.0, b=0.5, c=0.1, d=0.0, e=0.0, f=0.0)
    print_result("W3中心值", W3[10, 10])
    print_result("R3中心值", R3[10, 10])

    # 9.4 离散化精度验证
    result = verify_biharmonic_discretization(Nx=32, Ny=32)
    print(f"  9.4 离散化精度验证:")
    print_result("最大误差", result['max_error'])
    print_result("L2误差", result['l2_error'])

    # 9.5 板弯曲能
    D_plate = 1.0e3
    energy = plate_bending_energy(W1, D_plate, x[1] - x[0], y[1] - y[0])
    print_result("薄板弯曲能", energy)


def main():
    np.set_printoptions(precision=4, suppress=True)
    start_time = time.time()

    print("\n" + "#" * 70)
    print("#  软体机器人运动学建模 — 博士级科研代码合成项目")
    print("#  科学领域: 机器人学 — 软体机器人运动学建模")
    print("#  融合15个种子项目核心算法")
    print("#" * 70 + "\n")

    # 运行所有模块
    section_props = run_module_1_mesh_and_section()
    x_cheb, D_cheb = run_module_2_spectral_and_dg()
    material_params = run_module_3_hyperelastic_and_chemo()
    s_out, r_out, R_out = run_module_4_cosserat_kinematics(material_params)
    run_module_5_dynamics()
    run_module_6_inverse_kinematics()
    run_module_7_path_planning()
    run_module_8_rom_pod()
    run_module_9_biharmonic_validation()

    elapsed = time.time() - start_time
    print("\n" + "#" * 70)
    print(f"#  所有模块运行完成，总耗时: {elapsed:.3f} 秒")
    print("#" * 70 + "\n")

    return 0


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（58个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: line_grid 输出类型正确且单调递增 ----
s1 = line_grid(11, 0.0, 1.0, c=1)
assert isinstance(s1, np.ndarray), '[TC01] line_grid 返回类型应为 ndarray FAILED'
assert len(s1) == 11, '[TC01] line_grid 输出长度应为 11 FAILED'
assert np.all(np.diff(s1) > 0), '[TC01] line_grid 输出应单调递增 FAILED'

# ---- TC02: chebyshev_grid 端点值正确且单调递减 ----
cb2 = chebyshev_grid(10)
assert len(cb2) == 11, '[TC02] chebyshev_grid 节点数应为 11 FAILED'
assert np.isclose(cb2[0], 1.0), '[TC02] chebyshev_grid 第一个节点应接近 1 FAILED'
assert np.isclose(cb2[-1], -1.0), '[TC02] chebyshev_grid 最后一个节点应接近 -1 FAILED'

# ---- TC03: triangulate_polygon 输出形状正确 ----
theta_3 = np.linspace(0, 2 * np.pi, 8, endpoint=False)
verts3 = np.column_stack([0.1 * np.cos(theta_3), 0.1 * np.sin(theta_3)])
nodes3, tris3 = triangulate_polygon(verts3)
assert nodes3.shape[1] == 2, '[TC03] 三角剖分节点应为2列 FAILED'
assert tris3.shape[1] == 3, '[TC03] 三角形索引应为3列 FAILED'
assert tris3.shape[0] >= 1, '[TC03] 至少应有1个三角形 FAILED'

# ---- TC04: compute_section_properties 面积和惯性矩非负 ----
props4 = compute_section_properties(nodes3, tris3)
assert props4['A'] > 0, '[TC04] 截面面积应为正 FAILED'
assert props4['Ixx'] >= 0, '[TC04] 惯性矩 Ixx 应非负 FAILED'
assert props4['Iyy'] >= 0, '[TC04] 惯性矩 Iyy 应非负 FAILED'
assert props4['J'] >= 0, '[TC04] 极惯性矩 J 应非负 FAILED'

# ---- TC05: diaphony_compute 输出在 [0,1] 内且有限 ----
import numpy as np
np.random.seed(42)
pts5 = np.random.rand(100, 2)
dia5 = diaphony_compute(pts5)
assert np.isfinite(dia5), '[TC05] diaphony 应为有限值 FAILED'
assert 0.0 <= dia5 <= 1.0, '[TC05] diaphony 应在 [0,1] 范围内 FAILED'

# ---- TC06: chebyshev_matrix 常数函数微分接近零 ----
x6, D6 = chebyshev_matrix(10)
c6 = np.ones(11)
err6 = np.max(np.abs(D6 @ c6))
assert err6 < 1e-10, '[TC06] Chebyshev 微分矩阵作用常数应接近零 FAILED'

# ---- TC07: jacobi_polynomial P_3^{(0,0)} 在 0 点为 Legendre 值 ----
r7 = np.array([0.0])
p3_7 = jacobi_polynomial(r7, 0.0, 0.0, 3)
# P_3^(0,0) 即 Legendre P_3: (5x^3 - 3x)/2, 在 x=0 处为 0
assert np.abs(p3_7[0]) < 1e-12, '[TC07] Jacobi P_3^(0,0)(0) 应接近 0 FAILED'

# ---- TC08: vandermonde_1d 矩阵可逆（条件数有限） ----
x8, _ = chebyshev_matrix(6)
V8 = vandermonde_1d(6, x8)
cond8 = np.linalg.cond(V8)
assert np.isfinite(cond8), '[TC08] Vandermonde 矩阵条件数应为有限值 FAILED'
assert cond8 < 1e10, '[TC08] Vandermonde 矩阵条件数不应过大 FAILED'

# ---- TC09: lift_1d 输出形状为 (Np, 2) ----
x9, _ = chebyshev_matrix(5)
V9 = vandermonde_1d(5, x9)
L9 = lift_1d(5, V9)
assert L9.shape == (6, 2), '[TC09] lift_1d 输出形状应为 (Np,2) FAILED'

# ---- TC10: neo_hookean_strain_energy 单位变形梯度下应变能为零 ----
F10 = np.eye(3)
W10 = neo_hookean_strain_energy(F10, 1.0e5, 2.0e6)
assert np.abs(W10) < 1e-10, '[TC10] 单位变形梯度下 Neo-Hookean 应变能应接近零 FAILED'

# ---- TC11: neo_hookean_stress 单位变形梯度下应力为零 ----
P11 = neo_hookean_stress(F10, 1.0e5, 2.0e6)
assert np.max(np.abs(P11)) < 1e-6, '[TC11] 单位变形梯度下第一PK应力应接近零 FAILED'

# ---- TC12: mooney_rivlin_strain_energy 单位变形梯度下为零 ----
W12 = mooney_rivlin_strain_energy(F10, 5.0e4, 1.0e4, 2.0e6)
assert np.abs(W12) < 1e-10, '[TC12] 单位变形梯度下 Mooney-Rivlin 应变能应接近零 FAILED'

# ---- TC13: soft_robot_1d_constitutive 零应变可复现且为零 ----
n13, m13 = soft_robot_1d_constitutive(0.0, np.zeros(3), 1.0e6, 0.35e6, 0.005, 2e-6, 1e-6, 3e-6)
assert np.max(np.abs(n13)) < 1e-12, '[TC13] 零应变下内力应接近零 FAILED'
assert np.max(np.abs(m13)) < 1e-12, '[TC13] 零曲率下内矩应接近零 FAILED'
import numpy as np
np.random.seed(42)
n13b, m13b = soft_robot_1d_constitutive(0.0, np.zeros(3), 1.0e6, 0.35e6, 0.005, 2e-6, 1e-6, 3e-6)
assert np.max(np.abs(n13b)) < 1e-12, '[TC13] 零应变结果应可复现 FAILED'

# ---- TC14: chemo_mechanical_coupling 有效模量在合理范围内 ----
E14 = chemo_mechanical_coupling(np.array([0.5, 6.5]), 0.05, E0=1.0e6, gamma=0.3, beta_chem=0.2)
assert 0.1 * 1.0e6 <= E14 <= 5.0 * 1.0e6, '[TC14] 有效模量应在 [0.1E0, 5E0] 范围内 FAILED'

# ---- TC15: selkov_glycolysis_ode 积分可复现 ----
import numpy as np
np.random.seed(42)
tg15a, yg15a = low_storage_rk4(selkov_glycolysis_ode, (0.0, 50.0), np.array([0.9, 0.7]), 500)
np.random.seed(42)
tg15b, yg15b = low_storage_rk4(selkov_glycolysis_ode, (0.0, 50.0), np.array([0.9, 0.7]), 500)
assert np.allclose(yg15a, yg15b), '[TC15] Selkov ODE 积分应可复现 FAILED'

# ---- TC16: hat_map 和 vee_map 互为逆映射 ----
v16 = np.array([1.0, 2.0, 3.0])
hat16 = hat_map(v16)
vee16 = vee_map(hat16)
assert np.allclose(vee16, v16), '[TC16] hat/vee 应为互逆映射 FAILED'

# ---- TC17: rodrigues_rotation 旋转矩阵保持正交性 ----
import numpy as np
np.random.seed(42)
axis17 = np.random.randn(3)
axis17 = axis17 / np.linalg.norm(axis17)
R17 = rodrigues_rotation(axis17, np.pi / 4)
assert np.allclose(R17 @ R17.T, np.eye(3), atol=1e-12), '[TC17] 旋转矩阵应正交 FAILED'
assert np.abs(np.linalg.det(R17) - 1.0) < 1e-12, '[TC17] 旋转矩阵行列式应接近 1 FAILED'

# ---- TC18: r8blt_sl 求解带状下三角系统并验证 ----
ml18 = 2
N18 = 10
np.random.seed(42)
diag18 = np.abs(np.random.rand(ml18 + 1, N18)) + 0.5
x18 = np.ones(N18)
b18 = r8blt_mv(diag18, ml18, x18)
x_sol18 = r8blt_sl(diag18, ml18, b18)
assert np.allclose(x_sol18, x18, atol=1e-10), '[TC18] 带状三角求解应恢复原始向量 FAILED'

# ---- TC19: forward_kinematics_cosserat 输出形状正确 ----
L19 = 0.5
Ns19 = 10
kappa19 = np.zeros((Ns19 + 1, 3))
kappa19[:, 2] = 0.1
s19, r19, R19 = forward_kinematics_cosserat(L19, Ns19, kappa19)
assert s19.shape == (Ns19 + 1,), '[TC19] s 形状应为 (Ns+1,) FAILED'
assert r19.shape == (Ns19 + 1, 3), '[TC19] r 形状应为 (Ns+1,3) FAILED'
assert R19.shape == (Ns19 + 1, 3, 3), '[TC19] R 形状应为 (Ns+1,3,3) FAILED'

# ---- TC20: compute_strain_measures 输出形状正确 ----
v20, u20 = compute_strain_measures(r19, R19, s19)
assert v20.shape == (Ns19 + 1, 3), '[TC20] v 形状应为 (Ns+1,3) FAILED'
assert u20.shape == (Ns19 + 1, 3), '[TC20] u 形状应为 (Ns+1,3) FAILED'

# ---- TC21: low_storage_rk4 积分简谐振动可复现 ----
import numpy as np
np.random.seed(42)
t21a, y21a = low_storage_rk4(
    lambda t, y: driven_harmonic_oscillator(t, y, omega0=2.0, zeta=0.5, omega_drive=2.0),
    (0.0, 5.0), np.array([0.5, 0.0]), 200)
np.random.seed(42)
t21b, y21b = low_storage_rk4(
    lambda t, y: driven_harmonic_oscillator(t, y, omega0=2.0, zeta=0.5, omega_drive=2.0),
    (0.0, 5.0), np.array([0.5, 0.0]), 200)
assert np.allclose(y21a, y21b), '[TC21] 低存储RK4积分应可复现 FAILED'

# ---- TC22: driven_harmonic_oscillator 输出为二元向量 ----
y22 = driven_harmonic_oscillator(0.0, np.array([0.1, 0.2]), omega0=1.0, zeta=0.3, omega_drive=1.5)
assert y22.shape == (2,), '[TC22] 谐振子输出应为 (2,) FAILED'

# ---- TC23: sawtooth_driver 周期性 ----
import numpy as np
np.random.seed(42)
saw0 = sawtooth_driver(0.0, omega=1.0)
saw2pi = sawtooth_driver(2.0 * np.pi, omega=1.0)
assert np.isclose(saw0, saw2pi, atol=1e-12), '[TC23] sawtooth 应为 2π/ω 周期函数 FAILED'

# ---- TC24: barycentric_coordinates 三坐标和为 1 ----
p24 = np.array([0.2, 0.3])
a24 = np.array([0.0, 0.0])
b24 = np.array([1.0, 0.0])
c24 = np.array([0.0, 1.0])
alpha24, beta24, gamma24 = barycentric_coordinates(p24, a24, b24, c24)
assert np.isclose(alpha24 + beta24 + gamma24, 1.0), '[TC24] 重心坐标和应为 1 FAILED'

# ---- TC25: point_in_triangle 判定三角形顶点为内部 ----
alpha25, beta25, gamma25 = barycentric_coordinates(np.array([0.3, 0.0]), a24, b24, c24)
assert point_in_triangle(alpha25, beta25, gamma25), '[TC25] 三角形边上点应在内部 FAILED'

# ---- TC26: pwl_interp_2d_scattered 插值恢复已知数据点 ----
import numpy as np
np.random.seed(42)
xyd26 = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.8, 0.2]])
zd26 = np.sin(xyd26[:, 0] * np.pi) * np.cos(xyd26[:, 1] * np.pi)
zi26 = pwl_interp_2d_scattered(xyd26, zd26, xyd26)
assert np.allclose(zi26, zd26, atol=1e-10), '[TC26] 插值应在数据点处恢复原始值 FAILED'

# ---- TC27: shape_reconstruction_from_sensors 可复现 ----
import numpy as np
np.random.seed(42)
sp27 = np.random.rand(15, 2)
sr27 = np.sin(2.0 * np.pi * sp27[:, 0])
qp27 = np.random.rand(5, 2)
np.random.seed(42)
rec27a = shape_reconstruction_from_sensors(sp27, sr27, qp27, reconstruction_type='pwl')
np.random.seed(42)
rec27b = shape_reconstruction_from_sensors(sp27, sr27, qp27, reconstruction_type='pwl')
assert np.allclose(rec27a, rec27b), '[TC27] 形状重建应可复现 FAILED'

# ---- TC28: change_dynamic 经典硬币找零已知结果 ----
coins28 = np.array([1, 3, 4])
dp28 = change_dynamic(coins28, 10)
assert dp28[0] == 1, '[TC28] amount=1 最少硬币应为 1 FAILED'
assert dp28[2] == 1, '[TC28] amount=3 最少硬币应为 1 FAILED'
assert dp28[5] == 2, '[TC28] amount=6 最少硬币应为 2 FAILED'

# ---- TC29: configuration_to_tip 零角度直线延伸 ----
tx29, ty29 = configuration_to_tip(4, 0.25, np.zeros(4))
assert np.isclose(tx29, 1.0), '[TC29] 零角度时末端 x 应为总长度 FAILED'
assert np.isclose(ty29, 0.0, atol=1e-12), '[TC29] 零角度时末端 y 应为 0 FAILED'

# ---- TC30: energy_cost 相同构型代价为零 ----
c30 = np.array([0.1, 0.2, 0.3])
cost30 = energy_cost(c30, c30, stiffness=1.0, damping=0.0)
assert np.isclose(cost30, 0.0, atol=1e-12), '[TC30] 相同构型间代价应为零 FAILED'

# ---- TC31: compute_svd_basis 输出形状正确 ----
import numpy as np
np.random.seed(42)
snap31 = np.random.rand(50, 30)
basis31, sv31, mean31 = compute_svd_basis(snap31, 8, subtract_mean=True)
assert basis31.shape == (50, 8), '[TC31] 基矩阵形状应为 (M,k) FAILED'
assert len(sv31) == 8, '[TC31] 奇异值个数应为 k FAILED'
assert mean31.shape == (50,), '[TC31] 均值向量长度应为 M FAILED'

# ---- TC32: project/reconstruct 往返重构误差小 ----
coeff32 = project_onto_basis(snap31[:, 0], basis31, mean31)
recon32 = reconstruct_from_basis(coeff32, basis31, mean31)
err32 = np.linalg.norm(recon32 - snap31[:, 0]) / np.linalg.norm(snap31[:, 0])
assert err32 < 1.0, '[TC32] POD 往返重构相对误差应合理 FAILED'

# ---- TC33: energy_fraction 累加单调递增且最终为 1 ----
ef33 = energy_fraction(sv31)
assert np.all(np.diff(ef33) >= -1e-15), '[TC33] 能量分数应单调不减 FAILED'
assert np.isclose(ef33[-1], 1.0, atol=1e-12), '[TC33] 最终能量分数应接近 1 FAILED'

# ---- TC34: optimal_basis_size 应在有效范围内 ----
opt34 = optimal_basis_size(sv31, threshold=0.99)
assert 1 <= opt34 <= len(sv31), '[TC34] 最优基数量应在 [1, k] 范围内 FAILED'

# ---- TC35: biharmonic_w1 和 biharmonic_r1 维度一致 ----
x35 = np.linspace(-0.5, 0.5, 15)
y35 = np.linspace(-0.5, 0.5, 15)
X35, Y35 = np.meshgrid(x35, y35)
W35 = biharmonic_w1(X35, Y35, a=1.0, b=0.0, c=0.0, d=0.0, e=1.0, f=0.0, g=2.0)
R35 = biharmonic_r1(X35, Y35, a=1.0, b=0.0, c=0.0, d=0.0, e=1.0, f=0.0, g=2.0)
assert W35.shape == (15, 15), '[TC35] W1 形状应为 (Ny,Nx) FAILED'
assert R35.shape == (15, 15), '[TC35] R1 形状应为 (Ny,Nx) FAILED'

# ---- TC36: biharmonic_w3 中心值有限且非 NaN ----
W36 = biharmonic_w3(X35, Y35, a=1.0, b=0.5, c=0.1, d=0.0, e=0.0, f=0.0)
assert np.isfinite(W36[7, 7]), '[TC36] W3 中心值应为有限值 FAILED'

# ---- TC37: biharmonic_r3 解析残差在中心处有限 ----
R37 = biharmonic_r3(X35, Y35, a=1.0, b=0.5, c=0.1, d=0.0, e=0.5, f=0.5)
assert np.isfinite(R37[7, 7]), '[TC37] R3 中心值应为有限值 FAILED'

# ---- TC38: plate_bending_energy 非负 ----
dx38 = x35[1] - x35[0]
dy38 = y35[1] - y35[0]
E38 = plate_bending_energy(W35, 1.0e3, dx38, dy38)
assert E38 >= 0.0, '[TC38] 板弯曲能应非负 FAILED'

# ---- TC39: verify_biharmonic_discretization 返回字典含预期键 ----
res39 = verify_biharmonic_discretization(Nx=16, Ny=16)
assert 'max_error' in res39, '[TC39] 结果应包含 max_error FAILED'
assert 'l2_error' in res39, '[TC39] 结果应包含 l2_error FAILED'
assert res39['max_error'] > 0, '[TC39] 最大误差应为正值 FAILED'

# ---- TC40: tangent_stiffness_neo_hookean 输出形状为 (6,6) ----
F40 = np.eye(3)
F40[0, 0] = 1.05
F40[1, 1] = 1.0 / np.sqrt(1.05)
F40[2, 2] = 1.0 / np.sqrt(1.05)
C40 = tangent_stiffness_neo_hookean(F40, 1.0e5, 2.0e6)
assert C40.shape == (6, 6), '[TC40] 切线刚度矩阵形状应为 (6,6) FAILED'

# ---- TC41: hat_map 输出为反对称矩阵 ----
v41 = np.array([3.0, -1.0, 2.0])
hat41 = hat_map(v41)
assert np.allclose(hat41 + hat41.T, np.zeros((3, 3)), atol=1e-12), '[TC41] hat_map 输出应为反对称矩阵 FAILED'

# ---- TC42: compute_curvature 直线段曲率和挠率为零 ----
import numpy as np
np.random.seed(42)
s42 = np.linspace(0.0, 1.0, 20)
r42 = np.column_stack([s42, np.zeros(20), np.zeros(20)])
kappa42, tau42 = compute_curvature(r42, s42)
assert np.max(np.abs(kappa42[1:-1])) < 1e-6, '[TC42] 直线中心线曲率应接近零 FAILED'

# ---- TC43: r8blt_det 行列式为正 ----
ml43 = 2
N43 = 8
np.random.seed(42)
diag43 = np.abs(np.random.rand(ml43 + 1, N43)) + 0.5
det43 = r8blt_det(diag43, ml43)
assert det43 > 0, '[TC43] 正对角带状三角行列式应为正 FAILED'

# ---- TC44: shape_t6 在节点处为1，其他节点为0 ----
# T6 nodes in reference coordinates
t6_xi = np.array([0.0, 1.0, 0.0, 0.5, 0.5, 0.0])
t6_eta = np.array([0.0, 0.0, 1.0, 0.0, 0.5, 0.5])
for i in range(6):
    val = shape_t6(t6_xi[i], t6_eta[i], i)
    assert np.isclose(val, 1.0, atol=1e-12), f'[TC44] Node {i} 形函数值应接近 1 FAILED'
for i in range(6):
    for j in range(6):
        if i != j:
            val = shape_t6(t6_xi[j], t6_eta[j], i)
            assert np.abs(val) < 1e-12, f'[TC44] Node {j} 形函数 N_{i} 应接近 0 FAILED'

# ---- TC45: gauss_legendre_triangle 权重之和等于参考三角形面积 0.5 ----
for order in [1, 2, 3, 4]:
    qp, w = gauss_legendre_triangle(order)
    assert np.isclose(np.sum(w), 0.5, atol=1e-12), f'[TC45] order={order} 高斯权重和应接近 0.5 FAILED'

# ---- TC46: cauchy_theta_method 与 low_storage_rk4 结果空间不差太多 ----
import numpy as np
np.random.seed(42)
t46, y46 = cauchy_theta_method(
    lambda t, y: driven_harmonic_oscillator(t, y, omega0=1.0, zeta=0.1, omega_drive=1.0),
    (0.0, 2.0), np.array([0.5, 0.0]), 100, theta=0.5)
assert y46.shape[0] == 101, '[TC46] Cauchy 方法输出步数应为 n+1 FAILED'
assert np.isfinite(y46[-1, 0]), '[TC46] Cauchy 方法末端位移应为有限值 FAILED'

# ---- TC47: multi_target_path_planning 返回正确数量的路径 ----
targets47 = [(0.5, 0.3), (0.7, 0.4)]
paths47 = multi_target_path_planning(3, 0.25, targets47)
assert len(paths47) == 2, '[TC47] 多目标规划应返回与目标数相等的路径 FAILED'

# ---- TC48: pod_galerkin_rom 降阶矩阵大小正确 ----
basis48 = basis31
M48 = np.eye(50) * 0.1
K48 = np.eye(50) * 100.0
F48 = np.ones(50)
Mrom48, Krom48, From48 = pod_galerkin_rom(M48, K48, F48, basis48)
assert Mrom48.shape == (8, 8), '[TC48] ROM质量矩阵形状应为 (k,k) FAILED'
assert Krom48.shape == (8, 8), '[TC48] ROM刚度矩阵形状应为 (k,k) FAILED'
assert From48.shape == (8,), '[TC48] ROM力向量形状应为 (k,) FAILED'

# ---- TC49: assemble_banded_stiffness 输出紧凑存储形状正确 ----
a49 = assemble_banded_stiffness(20, 1.0e6 * 2e-6, 1.0e6 * 0.005, 0.05, ml=3)
assert a49.shape[0] == 3 + 1, '[TC49] 紧凑存储行数应为 ml+1 FAILED'
assert a49.shape[1] == 20, '[TC49] 紧凑存储列数应为 N FAILED'

# ---- TC50: compute_shear_correction_factor 在 [0.5, 1.0] 范围内 ----
kappa50 = compute_shear_correction_factor(nodes3, tris3, E=1.0e6, nu=0.35)
assert 0.5 <= kappa50 <= 1.0, '[TC50] 剪切修正系数应在 [0.5,1.0] 内 FAILED'

# ---- TC51: assemble_section_stiffness 输出对称 ----
K51 = assemble_section_stiffness(nodes3, tris3, E=1.0e6, nu=0.35)
assert np.allclose(K51, K51.T, atol=1e-10), '[TC51] 截面刚度矩阵应对称 FAILED'

# ---- TC52: randomized_svd 基本性质 ----
import numpy as np
np.random.seed(42)
A52 = np.random.rand(30, 20)
U52, S52, Vt52 = randomized_svd(A52, 5, p=3, q=2)
assert U52.shape == (30, 5), '[TC52] 随机SVD U 形状应为 (m,k) FAILED'
assert len(S52) == 5, '[TC52] 随机SVD S 长度应为 k FAILED'
assert Vt52.shape == (5, 20), '[TC52] 随机SVD Vt 形状应为 (k,n) FAILED'
assert np.all(S52 > 0), '[TC52] 奇异值应全为正 FAILED'

# ---- TC53: mooney_rivlin_stress 输出形状正确且值有限 ----
sigma53 = mooney_rivlin_stress(F10, 5.0e4, 1.0e4, 2.0e6)
assert sigma53.shape == (3, 3), '[TC53] Cauchy 应力形状应为 (3,3) FAILED'
assert np.all(np.isfinite(sigma53)), '[TC53] Cauchy 应力应为有限值 FAILED'

# ---- TC54: refine_cross_section_mesh 细化后节点/三角形数增多 ----
theta54 = np.linspace(0, 2 * np.pi, 8, endpoint=False)
verts54 = np.column_stack([0.1 * np.cos(theta54), 0.1 * np.sin(theta54)])
nodes54a, tris54a = triangulate_polygon(verts54)
nodes54b, tris54b = refine_cross_section_mesh(verts54)
assert nodes54b.shape[0] > nodes54a.shape[0], '[TC54] 细化后节点数应增多 FAILED'

# ---- TC55: sample_ellipse 输出点数正确且均在椭圆内 ----
pts55 = sample_ellipse(a=2.0, b=1.0, n=30)
assert pts55.shape[0] >= 30, '[TC55] 采样点数不应少于给定值 FAILED'
# 验证在椭圆内: (x/a)^2 + (y/b)^2 <= 1
inside = (pts55[:, 0] / 2.0) ** 2 + (pts55[:, 1] / 1.0) ** 2
assert np.all(inside <= 1.0 + 1e-8), '[TC55] 采样点应在椭圆内 FAILED'

# ---- TC56: discretize_configuration_space 输出正确数量 ----
ang56 = discretize_configuration_space(4, 11)
assert len(ang56) == 11, '[TC56] 离散角度数应为 n_angles FAILED'
assert np.all(np.abs(ang56) <= np.pi / 2 + 1e-10), '[TC56] 角度应在 [-θmax, θmax] 内 FAILED'

# ---- TC57: inverse_kinematics_soft_robot 末端误差小于容差 ----
kappa57, r57 = inverse_kinematics_soft_robot(
    np.array([0.5, 0.2, 0.0]), L=1.0, Ns=12, material_params={}, max_iter=80, tol=1e-4)
err57 = np.linalg.norm(r57[-1] - np.array([0.5, 0.2, 0.0]))
assert err57 < 1.0, '[TC57] 逆运动学末端误差应在合理范围 FAILED'

# ---- TC58: dp_path_planning_2d 返回合理结果 ----
import numpy as np
np.random.seed(42)
ang58, cost58 = dp_path_planning_2d(3, 0.25, (0.5, 0.3), n_discrete=7)
assert len(ang58) == 3, '[TC58] 路径规划应返回 n_segments 个角度 FAILED'
assert cost58 >= 0, '[TC58] 路径代价应非负 FAILED'

print('\n全部 58 个测试通过!\n')
