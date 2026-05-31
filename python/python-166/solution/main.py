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
    sys.exit(main())
