#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py
水声传播抛物方程模型 — 统一入口

本项目为博士级科研代码合成项目，研究领域为：
    声学工程：水声传播抛物方程模型（Parabolic Equation for Underwater Acoustics）

基于 15 个种子项目的核心算法，融合构建一个面向深海复杂环境的三维宽角抛物方程
（Wide-Angle Parabolic Equation, WAPE）传播建模系统。

运行方式：
    python main.py

无需任何命令行参数，程序自动完成从环境建模、网格生成、声源初始化、
PE 步进求解到传播损失分析的全流程，并输出数值结果与诊断信息。
"""

import numpy as np
import os
import sys

# =============================================================================
# 模块导入
# =============================================================================
from environment import OceanEnvironment
from mesh_builder import generate_depth_grid, generate_range_grid, PEMesh
from source_field import build_initial_field, source_power_normalization, disk_uniform_sample
from boundary_conditions import BoundaryConditionHandler
from parabolic_solver import ParabolicSolver
from modal_analysis import NormalModeAnalyzer, ModalConstraintSolver
from scattering_model import VolumeScatteringModel, ReverberationModel, BoxDistanceStatistics, SpatialCorrelation
from propagation_loss import PropagationLoss, ReceiverArray, MultipathStatistics
from utils import triangle_monomial_integral, binomial_coefficient, chebyshev_to_monomial_matrix
from special_functions import sincn_fun, cisi, alnorm


def print_section(title):
    """打印格式化节标题。"""
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def main():
    print("\n")
    print("*" * 72)
    print("*  水声传播宽角抛物方程（WAPE）建模系统")
    print("*  研究领域：声学工程 — 深海复杂环境声传播数值模拟")
    print("*" * 72)

    # ==========================================================================
    # 1. 海洋环境参数设置
    # ==========================================================================
    print_section("1. 海洋环境参数设置")

    env = OceanEnvironment(
        c0=1500.0,
        z_axis=1000.0,
        B=1000.0,
        epsilon=0.0057,
        rho0=1024.0,
        kappa_T=4.6e-10,
        g=9.80665,
        P0=1.01325e5,
        seabed_type="clay",
        seabed_cp=1700.0,
        seabed_cs=800.0,
        seabed_rho=1900.0,
        seabed_loss=0.5,
        depth_max=4000.0,
        frequency=100.0
    )

    # 海底地形参数：平坦 + 高斯山丘
    env.bathymetry_params = {
        'H0': 4000.0,
        'H1': 0.0,
        'beta': 1.0,
        'r0': 0.0,
        'L': 1.0,
        'H2': 600.0,
        'r_c': 25000.0,
        'sigma_r': 8000.0,
    }

    print(f"  声道轴声速 c0 = {env.c0:.2f} m/s")
    print(f"  声道轴深度 z_axis = {env.z_axis:.1f} m")
    print(f"  Munk 尺度 B = {env.B:.1f} m")
    print(f"  Munk 扰动 ε = {env.epsilon:.5f}")
    print(f"  工作频率 f = {env.frequency:.1f} Hz")
    print(f"  参考波数 k0 = {env.k0:.6f} rad/m")
    print(f"  最大水深 = {env.depth_max:.1f} m")
    print(f"  吸收系数 = {env.absorption_db_per_km():.4f} dB/km")

    # 声速剖面采样验证
    z_test = np.linspace(0, env.depth_max, 5)
    c_test = env.sound_speed(z_test)
    print(f"  声速剖面采样 (m/s): {np.round(c_test, 2)}")

    # ==========================================================================
    # 2. 计算网格生成
    # ==========================================================================
    print_section("2. 计算网格生成")

    z_max = env.depth_max
    nz = 201
    stretch_power = 2.5
    z_grid = generate_depth_grid(z_max, nz, stretch_power, z_axis=env.z_axis)

    r_max = 20000.0  # 20 km
    dr = env.c0 / (4.0 * env.frequency)  # 稳定性条件：λ/4（兼顾精度与速度）
    r_grid = generate_range_grid(r_max, dr)

    mesh = PEMesh(r_grid, z_grid, env)

    print(f"  深度网格点数 nz = {nz}")
    print(f"  深度方向首步 dz_min = {np.min(np.diff(z_grid)):.3f} m")
    print(f"  深度方向末步 dz_max = {np.max(np.diff(z_grid)):.3f} m")
    print(f"  水平网格点数 nr = {mesh.nr}")
    print(f"  水平步长 dr = {dr:.3f} m")
    print(f"  最大水平距离 r_max = {r_max/1000:.1f} km")

    quality = mesh.mesh_quality_stats()
    print(f"  三角形单元数 = {quality.get('num_elements', 0)}")
    print(f"  有效节点数 = {quality.get('num_valid_nodes', 0)}")
    print(f"  平均纵横比 = {quality.get('aspect_ratio_mean', 0):.3f}")

    # ==========================================================================
    # 3. 声源初始场生成
    # ==========================================================================
    print_section("3. 声源初始场生成")

    z_s = 200.0  # 声源深度 200 m
    w0 = 8.0     # 束腰半径
    source_type = 'gaussian'

    u0 = build_initial_field(
        z_grid, z_s,
        source_type=source_type,
        k0=env.k0, w0=w0, R_c=np.inf
    )
    u0 = source_power_normalization(u0, z_grid)

    print(f"  声源深度 z_s = {z_s:.1f} m")
    print(f"  束腰半径 w0 = {w0:.2f} m")
    print(f"  源类型 = {source_type}")
    print(f"  初始场能量 = {np.trapezoid(np.abs(u0)**2, z_grid):.6e}")

    # 圆盘声源采样验证（来自 301_disk01_monte_carlo）
    disk_samples = disk_uniform_sample(1000, radius=0.5)
    disk_mean_radius = float(np.mean(np.linalg.norm(disk_samples, axis=1)))
    print(f"  圆盘声源均匀采样验证: 平均半径 = {disk_mean_radius:.4f} m (理论=0.3333)")

    # ==========================================================================
    # 4. 边界条件初始化
    # ==========================================================================
    print_section("4. 边界条件初始化")

    bc_handler = BoundaryConditionHandler(env, mesh)

    # 测试海底导纳迭代（来自 807_nonlin_fixed_point）
    gamma_b = bc_handler.compute_seabed_admittance(theta_grazing=0.1)
    print(f"  海底导纳 γ_b = {gamma_b:.4e}")

    # PML 轮廓测试
    pml_sigma = bc_handler.pml_profile(z_grid)
    print(f"  PML 最大吸收系数 σ_max = {np.max(pml_sigma):.4f}")

    # 海底多边形编码（来自 905_pram 的边界词思想）
    x_poly, y_poly = bc_handler.bathymetry_polygon()
    print(f"  海底边界多边形顶点数 = {len(x_poly)}")

    # ==========================================================================
    # 5. 宽角抛物方程求解
    # ==========================================================================
    print_section("5. 宽角抛物方程（WAPE）求解")

    solver = ParabolicSolver(env, mesh, bc_handler, method='cn_fd')
    print(f"  求解方法: Crank-Nicolson 有限差分")
    print(f"  开始步进求解 ...")

    U = solver.solve(u0, z_s, progress_interval=max(1, mesh.nr // 10))

    e_err = solver.energy_conservation_error()
    print(f"  求解完成。全局能量守恒相对误差 = {e_err:.6e}")

    # ==========================================================================
    # 6. 简正波模态分析
    # ==========================================================================
    print_section("6. 简正波模态分析")

    mode_analyzer = NormalModeAnalyzer(env, z_min=0.0, z_max=env.depth_max, n_cheb=32)
    kr, phi, z_mode = mode_analyzer.solve_eigenproblem(n_modes=10)

    N_wkb = mode_analyzer.estimate_mode_count_wkb()
    N_dio = mode_analyzer.estimate_mode_count_diophantine()

    print(f"  传播模态数（谱求解） = {len(kr)}")
    print(f"  WKB 估计模态数 = {N_wkb}")
    print(f"  Diophantine 约束模态数 = {N_dio}")

    if len(kr) > 0:
        vp = mode_analyzer.modal_phase_velocity(kr)
        print(f"  第1模态水平波数 k_r,0 = {kr[0]:.6f} rad/m")
        print(f"  第1模态相速度 v_p,0 = {vp[0]:.2f} m/s")
        if phi.shape[1] > 0:
            vg0 = mode_analyzer.modal_group_velocity(phi[:, 0], kr[0])
            print(f"  第1模态群速度 v_g,0 ≈ {vg0:.2f} m/s")

    # 模态激励系数
    if phi.shape[1] > 0:
        A_n = mode_analyzer.modal_excitation_coefficients(phi, z_s)
        print(f"  模态激励系数前5个: {np.abs(A_n[:5])}")

    # 离散约束求解验证（来自 899_polyomino_parity）
    solver_dio = ModalConstraintSolver()
    n_solutions = solver_dio.solve_inequality_integer(
        a=np.pi / env.depth_max,
        b=env.k0
    )
    print(f"  Diophantine 不等式解数 = {len(n_solutions)}")

    # ==========================================================================
    # 7. 体积散射与混响分析
    # ==========================================================================
    print_section("7. 体积散射与混响分析")

    scat_model = VolumeScatteringModel(
        sigma0=5e-6, z0=60.0, alpha=0.4, Lambda=150.0,
        particle_radius=0.005, gamma_kappa=0.08, gamma_rho=0.03
    )

    z_scat_test = np.array([10.0, 50.0, 100.0, 500.0, 1000.0])
    sv_db = scat_model.scattering_strength_db(z_scat_test)
    print(f"  体积散射强度 S_v (dB): {np.round(sv_db, 2)}")

    rev_model = ReverberationModel(scat_model, c_water=env.c0)
    RL = rev_model.reverberation_level(
        R=10000.0, tau_pulse=0.01, SL_db=200.0, TL_db=60.0,
        z_scatter=100.0
    )
    print(f"  10 km 处混响级 RL = {RL:.2f} dB")

    # 3D 盒子随机距离统计（来自 113_box_distance）
    box_stats = BoxDistanceStatistics(a=1000.0, b=500.0, c=200.0, seed=42)
    mu_D, sigma_D = box_stats.mean_distance_monte_carlo(n_samples=20000)
    print(f"  散射体积随机距离统计: μ_D = {mu_D:.2f} m, σ_D = {sigma_D:.2f} m")

    # QMC 散射积分验证（来自 498_hammersley）
    def test_integrand(x):
        return np.sin(x[0]) * np.cos(x[1]) * np.exp(-x[2])

    from scattering_model import qmc_scattering_integral
    qmc_val = qmc_scattering_integral(
        test_integrand,
        bounds=[(0, 1), (0, 1), (0, 1)],
        n_samples=2048
    )
    print(f"  QMC 测试积分结果 = {qmc_val:.8f} (理论≈0.1918)")

    # ==========================================================================
    # 8. 传播损失计算与后处理
    # ==========================================================================
    print_section("8. 传播损失计算与后处理")

    pl = PropagationLoss(U, r_grid, z_grid, mesh.seafloor_depth)
    tl_coh = pl.coherent_tl()

    # 深度平均 TL
    tl_dasl = pl.depth_averaged_tl()
    print(f"  深度平均传播损失（首点）= {tl_dasl[0]:.2f} dB")
    print(f"  深度平均传播损失（末点）= {tl_dasl[-1]:.2f} dB")

    # 固定深度接收器
    z_rec = 500.0
    tl_rec = pl.tl_at_receiver(z_rec)
    print(f"  {z_rec:.0f} m 深度接收器 TL（末点）= {tl_rec[-1]:.2f} dB")

    # 接收器阵列（来自 1426_xyzl_display 的 3D 几何）
    z_vla = np.linspace(50.0, 1500.0, 16)
    r_vla = np.full_like(z_vla, 30000.0)
    vla = ReceiverArray(r_vla, z_vla)
    w_dc = vla.dolph_chebyshev_weights(sidelobe_db=-25)
    print(f"  VLA 阵元数 = {vla.n_receivers}")
    print(f"  Dolph-Chebyshev 权值和 = {np.sum(w_dc):.4f}")

    # 收敛区分析
    zones = pl.convergence_zone_analysis(tl_coh, threshold_db=3.0)
    print(f"  检测到的收敛区数量 = {len(zones)}")
    if zones:
        for idx, (i, j, rr, zz, tlval) in enumerate(zones[:3]):
            print(f"    收敛区 {idx+1}: r={rr/1000:.1f} km, z={zz:.0f} m, TL={tlval:.2f} dB")

    # 声影区检测
    shadows = pl.shadow_zone_detection(tl_coh, tl_threshold=85.0)
    print(f"  声影区检测到的距离步数 = {len(shadows)}")

    # 多径统计
    mp_stats = MultipathStatistics(pl, c_water=env.c0)
    tau_spread = mp_stats.delay_spread_estimate(mu_D)
    B_coh = mp_stats.coherence_bandwidth(tau_spread)
    print(f"  估计时延扩展 σ_τ = {tau_spread*1000:.4f} ms")
    print(f"  估计相干带宽 B_c = {B_coh:.2f} Hz")

    # ==========================================================================
    # 9. 空间相关分析
    # ==========================================================================
    print_section("9. 空间相关分析")

    spat_corr = SpatialCorrelation(U, r_grid, z_grid)
    C_r = spat_corr.correlation_1d(axis='r', lag_index=5)
    C_z = spat_corr.correlation_1d(axis='z', lag_index=5)
    print(f"  水平方向平均相关系数（lag=5）= {np.mean(C_r):.4f}")
    print(f"  深度方向平均相关系数（lag=5）= {np.mean(C_z):.4f}")

    # ==========================================================================
    # 10. 数值工具验证（来自种子项目核心算法）
    # ==========================================================================
    print_section("10. 种子项目核心算法验证")

    # 10.1 三角单元积分（来自 1307_triangle_integrals）
    tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    I_tri = triangle_monomial_integral(1, 1, tri)
    print(f"  参考三角形 x·y 积分 = {I_tri:.6f} (理论=1/24≈0.041667)")

    # 10.2 Chebyshev 转换矩阵（来自 894_polynomial_conversion）
    M_cheb = chebyshev_to_monomial_matrix(5)
    print(f"  Chebyshev→Monomial 矩阵行列式 = {np.linalg.det(M_cheb):.6f}")

    # 10.3 二项式系数（来自 044_asa152 的 log-gamma 技巧）
    c_20_10 = binomial_coefficient(20, 10)
    print(f"  C(20,10) = {c_20_10:.1f} (理论=184756)")

    # 10.4 sinc 函数与 Ci/Si（来自 1082_sinc）
    x_sinc = np.array([0.0, 0.5, 1.0, 2.0])
    sn = sincn_fun(x_sinc)
    print(f"  sinc_n(0,0.5,1.0,2.0) = {np.round(sn, 4)}")
    ci_val, si_val = cisi(np.array([1.0, 5.0, 20.0]))
    print(f"  Ci(1,5,20) = {np.round(ci_val, 4)}")
    print(f"  Si(1,5,20) = {np.round(si_val, 4)}")

    # 10.5 正态 CDF（来自 044_asa152 的 alnorm）
    phi_0 = alnorm(0.0)
    phi_2 = alnorm(2.0)
    print(f"  Φ(0) = {phi_0:.6f} (理论=0.5)")
    print(f"  Φ(2) = {phi_2:.6f} (理论≈0.9772)")

    # 10.6 点是否在多边形内（来自 1265_toms112）
    from mesh_builder import point_in_polygon
    x_poly = [0, 10, 10, 0]
    y_poly = [0, 0, 10, 10]
    inside = point_in_polygon(x_poly, y_poly, 5, 5)
    outside = point_in_polygon(x_poly, y_poly, 15, 5)
    print(f"  点(5,5) 在多边形内 = {inside} (理论=True)")
    print(f"  点(15,5) 在多边形内 = {outside} (理论=False)")

    # ==========================================================================
    # 11. 结果汇总
    # ==========================================================================
    print_section("11. 结果汇总")
    print(f"  求解域: 水平 {r_max/1000:.1f} km × 深度 {z_max:.0f} m")
    print(f"  网格规模: {mesh.nr} × {mesh.nz} = {mesh.nr*mesh.nz:,} 节点")
    print(f"  能量守恒误差: {e_err:.6e}")
    print(f"  传播模态数: {len(kr)}")
    print(f"  深度平均末点 TL: {tl_dasl[-1]:.2f} dB")
    print(f"  混响级 (10 km): {RL:.2f} dB")
    print(f"  时延扩展: {tau_spread*1000:.4f} ms")
    print(f"  相干带宽: {B_coh:.2f} Hz")
    print(f"  收敛区数量: {len(zones)}")
    print("\n  >>> 所有计算正常完成，数值结果已输出。 <<<")
    print("*" * 72 + "\n")

    return {
        'U': U,
        'tl_coh': tl_coh,
        'tl_dasl': tl_dasl,
        'kr': kr,
        'phi': phi,
        'zones': zones,
        'energy_error': e_err,
    }


if __name__ == '__main__':
    result = main()

# ================================================================
# 测试用例（25个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: sincn_fun在零点返回1 ----
result = sincn_fun(np.array([0.0]))
assert np.abs(result[0] - 1.0) < 1e-10, '[TC01] sincn_fun在零点返回1 FAILED'

# ---- TC02: sincn_fun在整数点返回0 ----
result = sincn_fun(np.array([1.0, 2.0, -1.0]))
assert np.all(np.abs(result) < 1e-10), '[TC02] sincn_fun在整数点返回0 FAILED'

# ---- TC03: alnorm(0)接近0.5 ----
result = alnorm(0.0)
assert np.abs(result - 0.5) < 1e-4, '[TC03] alnorm(0)接近0.5 FAILED'

# ---- TC04: alnorm(2)接近理论值0.97725 ----
result = alnorm(2.0)
assert np.abs(result - 0.97725) < 1e-4, '[TC04] alnorm(2)接近理论值 FAILED'

# ---- TC05: cisi的Si(1)接近已知值 ----
ci_val, si_val = cisi(np.array([1.0]))
assert np.abs(si_val[0] - 0.946083070367183) < 1e-4, '[TC05] cisi的Si(1) FAILED'

# ---- TC06: binomial_coefficient(20,10)等于184756 ----
result = binomial_coefficient(20, 10)
assert abs(result - 184756.0) < 0.5, '[TC06] binomial_coefficient(20,10) FAILED'

# ---- TC07: chebyshev_to_monomial_matrix形状正确且可逆 ----
M = chebyshev_to_monomial_matrix(5)
assert M.shape == (6, 6), '[TC07] chebyshev_to_monomial_matrix形状 FAILED'
assert abs(np.linalg.det(M)) > 1e-10, '[TC07] chebyshev_to_monomial_matrix行列式 FAILED'

# ---- TC08: point_in_polygon对矩形内点返回True ----
from mesh_builder import point_in_polygon
inside = point_in_polygon([0, 10, 10, 0], [0, 0, 10, 10], 5, 5)
assert inside == True, '[TC08] point_in_polygon对矩形内点 FAILED'

# ---- TC09: point_in_polygon对矩形外点返回False ----
from mesh_builder import point_in_polygon
outside = point_in_polygon([0, 10, 10, 0], [0, 0, 10, 10], 15, 5)
assert outside == False, '[TC09] point_in_polygon对矩形外点 FAILED'

# ---- TC10: generate_depth_grid边界与单调性 ----
z_grid = generate_depth_grid(1000.0, 51, stretch_power=2.0)
assert z_grid[0] == 0.0, '[TC10] generate_depth_grid首点 FAILED'
assert abs(z_grid[-1] - 1000.0) < 1e-6, '[TC10] generate_depth_grid末点 FAILED'
assert np.all(np.diff(z_grid) > 0), '[TC10] generate_depth_grid单调性 FAILED'

# ---- TC11: build_initial_field高斯束峰值在声源深度处 ----
z = np.linspace(0, 100, 101)
u = build_initial_field(z, z_s=50.0, source_type='gaussian', k0=1.0, w0=5.0)
assert np.argmax(np.abs(u)) == 50, '[TC11] build_initial_field峰值位置 FAILED'

# ---- TC12: source_power_normalization后能量为1 ----
z = np.linspace(0, 100, 101)
u = build_initial_field(z, z_s=50.0, source_type='gaussian', k0=1.0, w0=5.0)
u_norm = source_power_normalization(u, z)
power = np.trapezoid(np.abs(u_norm)**2, z)
assert abs(power - 1.0) < 1e-6, '[TC12] source_power_normalization能量 FAILED'

# ---- TC13: disk_uniform_sample所有点在圆盘内 ----
np.random.seed(42)
samples = disk_uniform_sample(1000, radius=1.0, seed=42)
radii = np.linalg.norm(samples, axis=1)
assert np.all(radii <= 1.0 + 1e-10), '[TC13] disk_uniform_sample越界 FAILED'

# ---- TC14: hammersley_sequence值域在0到1之间 ----
from source_field import hammersley_sequence
seq = hammersley_sequence(0, 100, 3)
assert np.all(seq >= 0.0) and np.all(seq <= 1.0), '[TC14] hammersley_sequence值域 FAILED'

# ---- TC15: OceanEnvironment声速最小值不低于1400 ----
env = OceanEnvironment(c0=1500.0, z_axis=1000.0, B=1000.0, epsilon=0.0057, depth_max=4000.0, frequency=100.0)
z_test = np.linspace(0, 4000, 41)
c_vals = env.sound_speed(z_test)
assert np.all(c_vals >= 1400.0), '[TC15] OceanEnvironment声速最小值 FAILED'

# ---- TC16: OceanEnvironment吸收系数非负 ----
env = OceanEnvironment(c0=1500.0, z_axis=1000.0, B=1000.0, epsilon=0.0057, depth_max=4000.0, frequency=100.0)
alpha = env.absorption_db_per_km()
assert alpha >= 0.0, '[TC16] OceanEnvironment吸收系数 FAILED'

# ---- TC17: ModalConstraintSolver正确求解Diophantine不等式 ----
solver = ModalConstraintSolver()
solutions = solver.solve_inequality_integer(a=2.0, b=10.0)
assert solutions == [0, 1, 2, 3, 4], '[TC17] ModalConstraintSolver求解 FAILED'

# ---- TC18: ReceiverArray Dolph-Chebyshev权重和为1 ----
z_vla = np.linspace(50.0, 1500.0, 16)
r_vla = np.full_like(z_vla, 10000.0)
vla = ReceiverArray(r_vla, z_vla)
w = vla.dolph_chebyshev_weights(sidelobe_db=-25)
assert abs(np.sum(w) - 1.0) < 1e-10, '[TC18] Dolph-Chebyshev权重和 FAILED'

# ---- TC19: VolumeScatteringModel散射强度非负 ----
scat = VolumeScatteringModel(sigma0=1e-6, z0=50.0, alpha=0.3, Lambda=100.0)
z_test = np.array([10.0, 50.0, 100.0])
sv = scat.scattering_strength_linear(z_test)
assert np.all(sv >= 0.0), '[TC19] VolumeScatteringModel散射强度 FAILED'

# ---- TC20: triangle_monomial_integral在参考三角形上积分x*y ----
tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
I_tri = triangle_monomial_integral(1, 1, tri)
assert abs(I_tri - 1.0/24.0) < 1e-12, '[TC20] triangle_monomial_integral FAILED'

# ---- TC21: BoundaryConditionHandler海面边界条件将首点置零 ----
env = OceanEnvironment(c0=1500.0, z_axis=1000.0, B=1000.0, epsilon=0.0057, depth_max=4000.0, frequency=100.0)
z_grid = generate_depth_grid(4000.0, 51, stretch_power=2.0)
r_grid = generate_range_grid(1000.0, 100.0)
mesh = PEMesh(r_grid, z_grid, env)
bc = BoundaryConditionHandler(env, mesh)
u = np.ones(51, dtype=np.complex128)
u_out = bc.apply_surface_bc(u)
assert u_out[0] == 0.0, '[TC21] apply_surface_bc首点 FAILED'

# ---- TC22: safe_divide对零除数返回fill_value ----
from utils import safe_divide
result = safe_divide(5.0, 0.0, fill_value=999.0)
assert result == 999.0, '[TC22] safe_divide对零除数返回fill_value FAILED'

# ---- TC23: sincu_fun在零点返回1 ----
from special_functions import sincu_fun
result = sincu_fun(np.array([0.0]))
assert np.abs(result[0] - 1.0) < 1e-10, '[TC23] sincu_fun在零点返回1 FAILED'

# ---- TC24: 三角形数公式正确 ----
from utils import triangle_number
result = triangle_number(10)
assert result == 55, '[TC24] triangle_number(10) FAILED'

# ---- TC25: 集成测试main返回结果包含关键字段 ----
result = main()
assert 'U' in result, '[TC25] main结果缺少U FAILED'
assert 'tl_coh' in result, '[TC25] main结果缺少tl_coh FAILED'
assert 'kr' in result, '[TC25] main结果缺少kr FAILED'

print('\n全部 25 个测试通过!\n')
