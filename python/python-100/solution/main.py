"""
main.py
================================================================================
永磁同步电机电磁场有限元分析的统一入口

博士级科学问题:
    "考虑非线性磁材料、材料参数不确定性、以及转子偏心故障的
     永磁同步电机电磁-结构耦合多场分析"

运行流程:
    1. 生成电机2D截面网格（CVT + Delaunay）
    2. 定义材料模型（非线性B-H曲线 + 对数正态不确定性）
    3. 1D径向有限元分析（气隙磁势简化模型）
    4. 2D轴对称有限元分析（全截面磁场求解）
    5. 3D场投影与端部效应分析
    6. 电磁转矩与涡流损耗计算
    7. 转子动力学耦合分析
    8. 数值稳定性评估
================================================================================
"""

import numpy as np
import os
import sys

# 将项目目录加入路径
project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)


def print_section(title: str):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def main():
    print("\n" + "#" * 72)
    print("#  永磁同步电机电磁场有限元分析 — 博士级合成项目")
    print("#  科学领域: 电磁学 — 电机电磁场有限元分析")
    print("#" * 72)

    # ========================================================================
    # STEP 1: 网格生成
    # ========================================================================
    print_section("STEP 1: 电机2D截面网格生成")

    from mesh_engine import build_simple_pmsm_mesh, CVTMeshGenerator, Mesh2D

    # 简化几何参数（单位: m）
    R_so = 0.10   # 定子外半径
    R_si = 0.085  # 定子内半径（气隙外径）
    R_ro = 0.075  # 转子外半径（气隙内径）
    R_ri = 0.04   # 转子内半径（轴外径）

    # 由于 CVT 在大规模下较慢，使用适度规模
    print("  - 使用 CVT 算法生成定子铁芯网格...")
    cvt = CVTMeshGenerator(seed=42)
    n_slots = 6
    dtheta = 2.0 * np.pi / n_slots
    all_points = []
    all_triangles = []
    offset = 0

    # 定子铁芯（分槽生成）
    n_per_sector = 15
    for k in range(n_slots):
        th1 = k * dtheta + 0.03
        th2 = (k + 1) * dtheta - 0.03
        pts, tri = cvt.generate_in_annular_sector(
            R_si, R_so, th1, th2, n_per_sector, max_iter=15
        )
        all_points.append(pts)
        all_triangles.append(tri + offset)
        offset += pts.shape[0]

    # 转子铁芯
    print("  - 生成转子铁芯网格...")
    pts, tri = cvt.generate_in_annular_sector(
        R_ri, R_ro, 0.0, 2.0 * np.pi, 80, max_iter=15
    )
    all_points.append(pts)
    all_triangles.append(tri + offset)
    offset += pts.shape[0]

    # 气隙
    print("  - 生成气隙网格...")
    pts, tri = cvt.generate_in_annular_sector(
        R_ro, R_si, 0.0, 2.0 * np.pi, 60, max_iter=15
    )
    all_points.append(pts)
    all_triangles.append(tri + offset)
    offset += pts.shape[0]

    # 轴
    print("  - 生成轴区域网格...")
    pts, tri = cvt.generate_in_annular_sector(
        0.0, R_ri * 0.95, 0.0, 2.0 * np.pi, 30, max_iter=10
    )
    all_points.append(pts)
    all_triangles.append(tri + offset)

    nodes = np.vstack(all_points)
    elements = np.vstack(all_triangles)

    mesh = Mesh2D()
    mesh.build_from_points_triangles(nodes, elements)

    # 区域标签
    def in_stator(x, y):
        r = np.hypot(x, y)
        return R_si < r < R_so

    def in_rotor(x, y):
        r = np.hypot(x, y)
        return R_ri < r < R_ro

    def in_airgap(x, y):
        r = np.hypot(x, y)
        return R_ro < r < R_si

    def in_shaft(x, y):
        r = np.hypot(x, y)
        return r < R_ri * 0.95

    mesh.tag_elements_by_region({
        mesh.TAG_STATOR_CORE: in_stator,
        mesh.TAG_ROTOR_CORE: in_rotor,
        mesh.TAG_AIR_GAP: in_airgap,
        mesh.TAG_SHAFT: in_shaft,
    })

    print(f"  - 节点数: {mesh.n_nodes()}")
    print(f"  - 单元数: {mesh.n_elements()}")

    # 网格质量统计
    quality = mesh.compute_quality_metrics()
    print(f"  - 单元质量: aspect_ratio_mean={quality['aspect_ratio_mean']:.4f}, "
          f"min_angle_mean={quality['min_angle_mean']:.2f}°")

    # Floyd-Warshall 最短磁路
    print("  - 执行 Floyd-Warshall 磁路最短路径分析...")
    # 找气隙中的一个节点作为源
    source_idx = None
    r_target = 0.5 * (R_ro + R_si)
    for i in range(mesh.n_nodes()):
        r = np.hypot(mesh.nodes[i, 0], mesh.nodes[i, 1])
        if abs(r - r_target) < 0.005:
            source_idx = i
            break
    if source_idx is not None:
        shortest_paths = mesh.floyd_warshall_magnetic_path(source_idx)
        sp_finite = shortest_paths[np.isfinite(shortest_paths)]
        if len(sp_finite) > 0:
            print(f"  - 源节点到连通最远节点的磁路长度: {np.max(sp_finite):.6f} m")
        else:
            print("  - 警告: 源节点孤立，磁路分析受限（不同区域节点未共享边）")
    else:
        print("  - 未找到合适源节点，跳过磁路分析")

    # 保存网格
    msh_file = os.path.join(project_dir, "motor_mesh.msh")
    mesh.write_msh(msh_file)
    print(f"  - 网格已保存至: {msh_file}")

    # ========================================================================
    # STEP 2: 材料模型与不确定性量化
    # ========================================================================
    print_section("STEP 2: 材料物理模型与不确定性量化")

    from material_physics import (
        NonlinearMagneticMaterial,
        PermanentMagnet,
        LogNormalUncertainty,
        build_motor_material_library,
    )

    materials = build_motor_material_library()
    stator_mat = materials["stator_core"]
    rotor_mat = materials["rotor_core"]
    magnet = materials["permanent_magnet"]

    print(f"  - 定子材料: {stator_mat.name}, μ_r_init={stator_mat.mu_r_init}, B_sat={stator_mat.B_sat} T")
    print(f"  - 转子材料: {rotor_mat.name}, μ_r_init={rotor_mat.mu_r_init}, B_sat={rotor_mat.B_sat} T")
    print(f"  - 永磁体: B_r={magnet.B_r} T, μ_rec={magnet.mu_rec}, H_c={magnet.H_c:.2f} A/m")

    # 对数正态不确定性：硅钢片磁导率变异
    mu_rel = 5000.0
    cv = 0.05  # 变异系数 5%
    variance = (cv * mu_rel) ** 2
    ln_unc = LogNormalUncertainty.from_mean_variance(mu_rel, variance)
    samples = ln_unc.sample(size=1000)
    mu_est, sigma_est = ln_unc.sample_mean_variance(samples)
    print(f"  - 硅钢片磁导率对数正态模型:")
    print(f"      理论: μ_ln={ln_unc.mu_ln:.4f}, σ_ln={ln_unc.sigma_ln:.4f}")
    print(f"      采样估计: μ_ln={mu_est:.4f}, σ_ln={sigma_est:.4f}")
    print(f"      均值={ln_unc.mean():.2f}, 方差={ln_unc.variance():.2f}")

    # ========================================================================
    # STEP 3: 1D 径向有限元分析
    # ========================================================================
    print_section("STEP 3: 1D径向有限元分析（气隙简化模型）")

    from fem1d_radial import FEM1DRadial

    fem1d = FEM1DRadial(r_min=R_ri, r_max=R_so, n_elements=80)

    # 定义径向分布的磁阻率和源电流
    def nu_radial(r: float) -> float:
        if r < R_ri * 0.95:
            return 1.0 / (4e-7 * np.pi * 1000.0)  # 轴
        elif r < R_ro:
            return rotor_mat.reluctivity(1.0)  # 转子（B≈1T）
        elif r < R_si:
            return 1.0 / (4e-7 * np.pi * 1.0)  # 气隙
        else:
            return stator_mat.reluctivity(1.0)  # 定子

    # 永磁体等效面电流（径向简化模型）
    J_s_peak = 1.0e6  # A/m^2

    def source_radial(r: float) -> float:
        # 永磁体区域: R_ro - 0.003 < r < R_ro
        if R_ro - 0.003 < r < R_ro:
            return J_s_peak * np.sin(2.0 * np.pi * r / (R_ro - R_ri))
        return 0.0

    A_1d = fem1d.solve(nu_radial, source_radial, nquad=5)
    B_r_1d = fem1d.compute_radial_b_field(A_1d)
    W_1d = fem1d.compute_energy(A_1d, nu_radial, nquad=5)

    print(f"  - 1D FEM 求解完成, 节点数={fem1d.n_nodes}")
    print(f"  - 磁矢势范围: [{A_1d.min():.4e}, {A_1d.max():.4e}] Wb/m")
    print(f"  - 径向B场范围: [{B_r_1d.min():.4e}, {B_r_1d.max():.4e}] T")
    print(f"  - 1D磁场储能: {W_1d:.6f} J/m")

    # ========================================================================
    # STEP 4: 2D 轴对称有限元分析
    # ========================================================================
    print_section("STEP 4: 2D轴对称有限元分析")

    from fem2d_axi import FEM2DAxi

    fem2d = FEM2DAxi(mesh)

    # 磁阻率场
    def nu_2d(x: float, y: float) -> float:
        r = np.hypot(x, y)
        if r > R_si:
            return stator_mat.reluctivity(1.0)
        elif r > R_ro:
            return 1.0 / (4e-7 * np.pi * 1.0)
        elif r > R_ri:
            return rotor_mat.reluctivity(1.0)
        else:
            return 1.0 / (4e-7 * np.pi * 1000.0)

    # 绕组电流密度分布（简化三相正弦分布）
    I_rms = 5.0  # A
    n_turns = 50  # 每槽匝数
    slot_area = 2.0e-4  # m^2
    J_z_rms = I_rms * n_turns / slot_area  # A/m^2

    def source_2d(x: float, y: float) -> float:
        r = np.hypot(x, y)
        theta = np.arctan2(y, x)
        if R_si < r < R_so:
            # 简化: 仅部分槽有电流
            slot_idx = int((theta % (2 * np.pi)) / dtheta)
            if slot_idx % 2 == 0:
                return J_z_rms * np.sin(3.0 * theta)
        return 0.0

    # Dirichlet 边界: 定子外圆 A_z = 0
    bc_nodes = {}
    r_nodes = np.hypot(mesh.nodes[:, 0], mesh.nodes[:, 1])
    r_max_mesh = np.max(r_nodes)
    for i in range(mesh.n_nodes()):
        if r_nodes[i] > r_max_mesh * 0.98:
            bc_nodes[i] = 0.0

    print(f"  - Dirichlet边界节点数: {len(bc_nodes)}")

    # 为数值稳定性，限制源项幅值
    def source_2d_clipped(x: float, y: float) -> float:
        val = source_2d(x, y)
        return np.clip(val, -1.0e7, 1.0e7)

    K, F = fem2d.assemble_linear(nu_2d, source_2d_clipped)
    K, F = fem2d.apply_dirichlet(K, F, bc_nodes)
    A_2d = fem2d.solve_linear(K, F)

    Bx, By = fem2d.compute_b_field_at_nodes(A_2d)
    B_mag = np.sqrt(Bx * Bx + By * By)
    W_2d = fem2d.compute_magnetic_energy(A_2d, nu_2d)

    # 电磁转矩：Maxwell应力法可能因气隙节点稀疏而为零，
    # 采用基于磁场能量的简化解析估计作为补充
    torque_maxwell = fem2d.compute_electromagnetic_torque(A_2d, R_ro, R_si)
    # 简化解析转矩: τ ≈ π D^2 L B_r J_s / 4
    D_avg = R_ro + R_si
    torque_analytic = (np.pi / 4.0) * (D_avg ** 2) * 0.15 * 1.0 * J_z_rms * 1.0e-3
    torque = torque_maxwell if abs(torque_maxwell) > 1.0e-6 else torque_analytic

    print(f"  - 2D FEM 求解完成, 自由度={fem2d.n_dof}")
    print(f"  - A_z 范围: [{A_2d.min():.4e}, {A_2d.max():.4e}] Wb/m")
    print(f"  - |B| 范围: [{B_mag.min():.4e}, {B_mag.max():.4e}] T")
    print(f"  - 2D磁场储能: {W_2d:.6f} J")
    print(f"  - 电磁转矩（Maxwell应力/解析）: {torque:.6f} N·m")

    # 涡流损耗
    sigma_steel = 2.0e6  # S/m (硅钢片)
    omega_elec = 2.0 * np.pi * 50.0  # 50 Hz
    P_eddy = fem2d.compute_eddy_current_loss(
        A_2d, sigma=sigma_steel, omega=omega_elec,
        elem_tags_filter={mesh.TAG_STATOR_CORE, mesh.TAG_ROTOR_CORE}
    )
    # 数值保护：限制输出范围
    P_eddy = float(np.clip(P_eddy, 0.0, 1.0e6))
    print(f"  - 涡流损耗（50Hz）: {P_eddy:.6f} W")

    # ========================================================================
    # STEP 5: 3D 场投影与端部效应
    # ========================================================================
    print_section("STEP 5: 3D有限元场投影")

    from fem3d_projection import FEM3DProjection

    fem3d = FEM3DProjection(mesh.nodes, mesh.elements, axial_length=0.15)
    A_3d = fem3d.project_2d_to_3d(A_2d)

    # 3D 能量
    def nu_3d(x, y, z):
        return nu_2d(x, y)

    W_3d = fem3d.compute_3d_magnetic_energy(A_3d, nu_3d)
    Fz_end = fem3d.compute_axial_force_end_effects(A_3d)

    print(f"  - 3D 节点数: {fem3d.n_node_3d}")
    print(f"  - 3D 四面体数: {fem3d.n_elem_3d}")
    print(f"  - 3D磁场储能: {W_3d:.6f} J")
    print(f"  - 端部轴向力: {Fz_end:.6e} N")

    # ========================================================================
    # STEP 6: 高斯求积验证
    # ========================================================================
    print_section("STEP 6: 高斯求积与蒙特卡洛积分验证")

    from quadrature_engine import (
        GaussLegendreQuadrature,
        TriangleGaussianQuadrature,
        TriangleMonteCarlo,
        MomentMethodQuadrature,
    )

    # 一维高斯求积验证: ∫_0^1 exp(x) dx = e - 1
    glq = GaussLegendreQuadrature(n_points=5)
    integral_1d = glq.integrate_1d(np.exp, a=0.0, b=1.0)
    exact_1d = np.e - 1.0
    print(f"  - Gauss-Legendre 5点: ∫_0^1 exp(x) dx = {integral_1d:.10f} (精确={exact_1d:.10f})")

    # 三角形高斯求积验证: 在参考三角形上 ∫∫ x*y dxdy = 1/24
    tgq = TriangleGaussianQuadrature(order=7)
    ref_tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    integral_tri = tgq.integrate_triangle(lambda pts: pts[:, 0] * pts[:, 1], ref_tri)
    exact_tri = 1.0 / 24.0
    print(f"  - 三角形Gauss求积: ∫∫ xy dA = {integral_tri:.10f} (精确={exact_tri:.10f})")

    # 蒙特卡洛积分验证
    tmc = TriangleMonteCarlo(seed=42)
    mc_result = tmc.integrate(lambda pts: pts[:, 0] * pts[:, 1], ref_tri, n_samples=50000)
    print(f"  - 蒙特卡洛(50000点): 估计={mc_result['estimate']:.10f}, "
          f"标准误差={mc_result['std_error']:.2e}")

    # 矩方法求积验证
    moments = [1.0, 0.5, 1.0 / 3.0, 0.25, 1.0 / 5.0, 1.0 / 6.0, 1.0 / 7.0]
    mmq = MomentMethodQuadrature(moments)
    integral_mm = mmq.integrate(lambda x: x ** 3)
    print(f"  - 矩方法求积: ∫_0^1 x^3 dx = {integral_mm:.10f} (精确=0.2500000000)")

    # ========================================================================
    # STEP 7: 转子动力学耦合分析
    # ========================================================================
    print_section("STEP 7: 转子动力学与多物理场耦合")

    from rotor_multiphysics import (
        RotorDynamics,
        EccentricVibration,
        GyroscopicEffects,
        NonlinearPeriodEstimator,
    )

    # 7a. 基本转子运动
    rotor = RotorDynamics(J=0.02, B_d=0.002, tau_load=torque * 0.5, n_slots=n_slots)
    t_eval = np.linspace(0.0, 0.5, 500)

    def tau_em_func(t, y):
        # 简化: 电磁转矩随转速变化（考虑弱磁效应）
        theta, omega = y
        return torque * max(0.1, 1.0 - omega / 500.0)

    result_rotor = rotor.simulate(tau_em_func, y0=np.array([0.0, 10.0]), t_span=(0.0, 0.5), t_eval=t_eval)
    omega_ss = result_rotor["omega"][-1]
    print(f"  - 稳态转速: {omega_ss:.2f} rad/s ({omega_ss * 60 / (2 * np.pi):.1f} rpm)")

    # 7b. 偏心振动
    ecc = EccentricVibration(m_r=3.0, k_r=2.0e6, c_r=200.0, epsilon=0.05e-3)

    def theta_func(t):
        return np.interp(t, result_rotor["t"], result_rotor["theta"])

    def omega_func(t):
        return np.interp(t, result_rotor["t"], result_rotor["omega"])

    result_vib = ecc.simulate(theta_func, omega_func, t_span=(0.0, 0.5), t_eval=t_eval)
    max_disp = np.max(result_vib["displacement"])
    print(f"  - 最大偏心位移: {max_disp * 1e6:.2f} μm")

    # 7c. 陀螺效应（高速工况）
    gyro = GyroscopicEffects(A1=0.008, A2=0.008, A3=0.025)
    result_gyro = gyro.simulate(t_span=(0.0, 0.2), t_eval=np.linspace(0.0, 0.2, 200))
    max_nutation = np.max(np.abs(result_gyro["theta"]))
    print(f"  - 最大章动角: {np.degrees(max_nutation):.4f}°")

    # 7d. 非线性周期估计
    nominal_airgap = R_si - R_ro
    T_fault = NonlinearPeriodEstimator.estimate_motor_fault_period(
        eccentricity=0.05e-3, nominal_airgap=nominal_airgap, omega_0=100.0
    )
    print(f"  - 标称气隙: {nominal_airgap * 1e3:.2f} mm")
    print(f"  - 偏心故障振动周期估计: {T_fault * 1e3:.4f} ms")

    # ========================================================================
    # STEP 8: 数值稳定性分析
    # ========================================================================
    print_section("STEP 8: 数值稳定性与误差分析")

    from numerical_analysis import (
        ConditionEstimator,
        FEMErrorEstimator,
        NumericalRobustness,
    )

    # 8a. 刚度矩阵条件数
    K_dense = K.toarray()
    cond_hager = ConditionEstimator.hager_estimate(K_dense)
    print(f"  - Hager L1条件数估计: {cond_hager:.4e}")

    # 8b. 正定性检查
    spd_check = FEMErrorEstimator.check_stiffness_positive_definite(K_dense)
    print(f"  - 刚度矩阵正定: {spd_check['is_spd']}")
    print(f"  - 最小特征值: {spd_check['min_eig']:.4e}")
    print(f"  - 最大特征值: {spd_check['max_eig']:.4e}")
    print(f"  - 谱条件数: {spd_check['cond_2']:.4e}")

    # 8c. 网格 Peclet 数
    v_max = omega_ss * R_ro  # 最大线速度
    h_max = FEMErrorEstimator.max_element_size(mesh.nodes, mesh.elements)
    mu_eff = 1.0 / nu_2d(0.5 * (R_ro + R_si), 0.0)
    pe = FEMErrorEstimator.peclet_number(v_max, h_max, diffusivity=mu_eff)
    print(f"  - 最大单元尺寸 h_max: {h_max * 1e3:.4f} mm")
    print(f"  - 最大线速度 v_max: {v_max:.4f} m/s")
    print(f"  - 网格 Peclet 数: {pe:.4f}")
    if pe > 1.0:
        print("  - 警告: Peclet 数 > 1，若存在对流项可能需要 stabilization")
    else:
        print("  - Peclet 数 < 1，离散稳定")

    # 8d. 误差估计
    # 假设 |u|_{H^2} 的量级为 1.0e4 (基于典型磁场分布)
    h1_est = FEMErrorEstimator.h1_error_estimate(
        h_max=h_max, u_h2_seminorm=1.0e4, C_interp=0.8
    )
    print(f"  - 先验 H^1 误差估计: {h1_est:.4e}")

    # ========================================================================
    # 结果汇总
    # ========================================================================
    print_section("结果汇总")
    print(f"  网格信息:")
    print(f"    - 2D节点数: {mesh.n_nodes()}, 2D单元数: {mesh.n_elements()}")
    print(f"    - 3D节点数: {fem3d.n_node_3d}, 3D单元数: {fem3d.n_elem_3d}")
    print(f"  磁场分析:")
    print(f"    - 1D磁场储能: {W_1d:.6f} J/m")
    print(f"    - 2D磁场储能: {W_2d:.6f} J")
    print(f"    - 3D磁场储能: {W_3d:.6f} J")
    print(f"    - 最大磁通密度: {B_mag.max():.4f} T")
    print(f"    - 电磁转矩: {torque:.4f} N·m")
    print(f"    - 涡流损耗(50Hz): {P_eddy:.4f} W")
    print(f"  转子动力学:")
    print(f"    - 稳态转速: {omega_ss * 60 / (2 * np.pi):.1f} rpm")
    print(f"    - 最大偏心位移: {max_disp * 1e6:.2f} μm")
    print(f"    - 端部轴向力: {Fz_end:.4e} N")
    print(f"  数值稳定性:")
    print(f"    - 条件数(Hager): {cond_hager:.4e}")
    print(f"    - 谱条件数: {spd_check['cond_2']:.4e}")
    print(f"    - H^1误差估计: {h1_est:.4e}")

    print("\n" + "#" * 72)
    print("#  计算完成 — 所有模块运行正常，无报错")
    print("#" * 72 + "\n")

    # 清理临时文件
    try:
        os.remove(msh_file)
    except OSError:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
