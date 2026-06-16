#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多铁性材料磁电耦合模拟: 高阶有限差分与稳定性分析
==================================================
统一入口 (零参数可运行)

项目概述:
    本项目实现 BiFeO3 多铁性材料的 Landau-Ginzburg-Devonshire (LGD)
    自由能泛函的数值模拟. 通过高阶有限差分空间离散和多种时间积分格式
    (FTCS/半隐式/RK4) 求解极化-磁化耦合演化方程.

    集成 15 个种子项目的核心算法:
    - 755_mesh_etoe: 畴结构网格拓扑
    - 972_r8but: 上三角带状矩阵求解
    - 984_r8lt: 下三角矩阵求解
    - 855_pdflib: 概率分布与热噪声
    - 1186_GlacierWeilin: 相图分类树
    - 1136_SonyResearch: 跨场耦合连接器
    - 486_gray_scott: 反应扩散类比
    - 1234_HackBio: 序参量 PCA
    - 1223_gev26: LP 界估计
    - 1306_triangle: BZ 三角采样
    - 1270_PAC: MD 轨迹分析
    - 1147_Hilbert-Smith: 熵迹分析
    - 088_biharmonic: 高阶有限差分
    - 434_fisher: FTCS 时间格式
    - 1058_dndimitri: 多任务代理模型

运行:
    python main.py
"""

import sys
import os
import time
import numpy as np

# 确保模块路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multiferroic_constants import (BiFeO3LGD, BiFeO3Lattice,
                                     SimulationConfig, EPSILON_0, MU_0,
                                     K_BOLTZMANN)
from lgd_free_energy import LGDFreeEnergyFunctional
from high_order_fd import (compact_laplacian_2d, sixth_order_second_derivative,
                            von_neumann_stability_limit, biharmonic_2d,
                            biharmonic_1d_sparse)
from banded_solver import (BandedMatrixUpper, BandedMatrixLower,
                            thomas_algorithm, build_lgd_banded_matrix_1d)
from mesh_domain_topology import DomainMeshTopology
from stochastic_thermal import ThermalNoiseGenerator
from magnetoelectric_connector import MagnetoelectricConnector
from brillouin_zone import BrillouinZoneSampler
from phase_classifier import MultiferroicPhaseClassifier
from magnetoelectric_bounds import MagnetoelectricBounds
from entropy_trace import MagnetoelectricEntropyTrace
from md_trajectory_analysis import MDTrajectoryAnalyzer
from order_parameter_analysis import OrderParameterAnalyzer
from simulation_engine import MultiferroicSimulationEngine


def print_header():
    """打印项目标题."""
    print()
    print("=" * 70)
    print("  PROJECT 285: 多铁性材料磁电耦合模拟")
    print("  高阶有限差分与稳定性分析 (小规模可复现实验)")
    print()
    print("  材料体系: BiFeO₃ (BFO) 钙钛矿多铁性材料")
    print("  理论框架: Landau-Ginzburg-Devonshire 自由能泛函")
    print("  数值方法: 6阶有限差分 + 半隐式时间积分 + ADI 分解")
    print("=" * 70)
    print()


def demo_numerical_methods():
    """
    演示数值方法模块.

    展示高阶有限差分、带状求解器、稳定性分析.
    """
    print("-" * 50)
    print("  1. 数值方法验证")
    print("-" * 50)

    # 1.1 高阶有限差分精度验证
    print("\n  [1.1] 高阶有限差分精度测试")
    n = 64
    L = 1.0
    x = np.linspace(0, L, n, endpoint=False)
    h = L / n

    # 测试函数: f(x) = sin(2πx)
    f_exact = np.sin(2 * np.pi * x)
    df_exact = 2 * np.pi * np.cos(2 * np.pi * x)
    d2f_exact = -(2 * np.pi) ** 2 * np.sin(2 * np.pi * x)

    from high_order_fd import (first_derivative_2nd_order,
                                first_derivative_4th_order,
                                sixth_order_first_derivative,
                                second_derivative_2nd_order,
                                second_derivative_4th_order,
                                sixth_order_second_derivative)

    # 将 1D 转为 2D (ny=1)
    f_2d = f_exact[:, np.newaxis] * np.ones((1, 1))
    f_2d = np.tile(f_exact[:, np.newaxis], (1, 10))

    df2 = first_derivative_2nd_order(f_2d, h, axis=0)
    df4 = first_derivative_4th_order(f_2d, h, axis=0)
    df6 = sixth_order_first_derivative(f_2d, h, axis=0)

    d2f2 = second_derivative_2nd_order(f_2d, h, axis=0)
    d2f4 = second_derivative_4th_order(f_2d, h, axis=0)
    d2f6 = sixth_order_second_derivative(f_2d, h, axis=0)

    df_ref = np.tile(df_exact[:, np.newaxis], (1, 10))
    d2f_ref = np.tile(d2f_exact[:, np.newaxis], (1, 10))

    err_df2 = np.max(np.abs(df2 - df_ref))
    err_df4 = np.max(np.abs(df4 - df_ref))
    err_df6 = np.max(np.abs(df6 - df_ref))
    err_d2f2 = np.max(np.abs(d2f2 - d2f_ref))
    err_d2f4 = np.max(np.abs(d2f4 - d2f_ref))
    err_d2f6 = np.max(np.abs(d2f6 - d2f_ref))

    print(f"    一阶导数误差:")
    print(f"      O(h²)  : {err_df2:.4e}")
    print(f"      O(h⁴)  : {err_df4:.4e}")
    print(f"      O(h⁶)  : {err_df6:.4e}")
    print(f"    二阶导数误差:")
    print(f"      O(h²)  : {err_d2f2:.4e}")
    print(f"      O(h⁴)  : {err_d2f4:.4e}")
    print(f"      O(h⁶)  : {err_d2f6:.4e}")

    # 1.2 带状矩阵求解器
    print("\n  [1.2] 带状矩阵求解器测试")
    n_test = 20
    mu = 2

    # 上三角带状矩阵
    A_upper = BandedMatrixUpper(n_test, mu)
    for i in range(n_test):
        A_upper.data[0, i] = 3.0  # 主对角
        if i < n_test - 1:
            A_upper.data[1, i] = 1.0
        if i < n_test - 2:
            A_upper.data[2, i] = 0.5

    b_test = np.random.randn(n_test)
    x_sol = A_upper.solve(b_test)
    residual = np.linalg.norm(A_upper.matvec(x_sol) - b_test)
    print(f"    上三角求解残差: {residual:.4e}")
    print(f"    行列式: {A_upper.determinant():.4e}")

    # 下三角带状矩阵
    A_lower = BandedMatrixLower(n_test, mu)
    for i in range(n_test):
        A_lower.data[0, i] = 4.0
    A_lower.data[1, :n_test - 1] = -1.0  # 下次对角线

    x_lower = A_lower.solve(b_test)
    residual_lower = np.linalg.norm(A_lower.matvec(x_lower) - b_test)
    print(f"    下三角求解残差: {residual_lower:.4e}")

    # Thomas 算法 (三对角系统)
    n_tri = 20
    a_tri = -np.ones(n_tri)
    a_tri[0] = 0.0  # a[0] unused
    b_tri = 3.0 * np.ones(n_tri)
    c_tri = -np.ones(n_tri)
    c_tri[-1] = 0.0  # c[n-1] unused
    d_tri = np.random.randn(n_tri)
    x_thomas = thomas_algorithm(a_tri, b_tri, c_tri, d_tri)

    # 构造三对角矩阵验证残差
    A_tri = np.diag(b_tri) + np.diag(a_tri[1:], -1) + np.diag(c_tri[:-1], 1)
    thomas_residual = np.linalg.norm(A_tri @ x_thomas - d_tri)
    print(f"    Thomas 算法残差: {thomas_residual:.4e}")

    # 1.3 稳定性分析
    print("\n  [1.3] 稳定性分析")
    lgd = BiFeO3LGD()
    config = SimulationConfig()

    dt_max_2 = von_neumann_stability_limit(
        lgd.G11, lgd.L_P, config.dx, config.dy, fd_order=2
    )
    dt_max_4 = von_neumann_stability_limit(
        lgd.G11, lgd.L_P, config.dx, config.dy, fd_order=4
    )
    dt_max_6 = von_neumann_stability_limit(
        lgd.G11, lgd.L_P, config.dx, config.dy, fd_order=6
    )

    print(f"    最大稳定 dt (O(h²)): {dt_max_2:.4e} s")
    print(f"    最大稳定 dt (O(h⁴)): {dt_max_4:.4e} s")
    print(f"    最大稳定 dt (O(h⁶)): {dt_max_6:.4e} s")
    print(f"    当前 dt: {config.dt:.4e} s")


def demo_free_energy():
    """
    演示 LGD 自由能计算.
    """
    print("\n" + "-" * 50)
    print("  2. LGD 自由能计算")
    print("-" * 50)

    lgd = BiFeO3LGD()
    fe = LGDFreeEnergyFunctional(lgd)

    nx, ny = 16, 16
    dx = dy = 1e-9

    # 测试不同温度的自由能
    T_values = [300, 600, 900, 1200]
    P0 = 0.95  # C/m²

    print(f"\n  温度依赖的 Landau 体能量密度:")
    print(f"  {'T (K)':>8s}  {'α₁(T) (V·m/C)':>16s}  {'f_L (J/m³)':>16s}")

    for T in T_values:
        alpha_T = lgd.alpha1_temperature(T)
        P_test = np.zeros((nx, ny, 3))
        P_test[..., 2] = P0  # 沿 z 方向

        f_L = fe.f_landau(P_test, T)
        f_mean = np.mean(f_L)

        print(f"  {T:8d}  {alpha_T:16.4e}  {f_mean:16.4e}")


def demo_simulation():
    """
    运行主模拟.
    """
    print("\n" + "-" * 50)
    print("  3. 主模拟: 时间演化")
    print("-" * 50)

    config = SimulationConfig()
    config.nx = 32
    config.ny = 32
    config.n_steps = 100
    config.output_interval = 25
    config.dt = 1e-14
    config.temperature = 300.0
    config.scheme = 'FTCS'
    config.fd_order = 4

    engine = MultiferroicSimulationEngine(config)
    engine.initialize_fields(mode='random_ferroelectric')

    t_start = time.time()
    engine.run()
    t_elapsed = time.time() - t_start

    print(f"\n  模拟用时: {t_elapsed:.2f} s")

    return engine


def demo_analysis(engine):
    """
    运行后处理分析.
    """
    print("\n" + "-" * 50)
    print("  4. 后处理分析")
    print("-" * 50)

    results = engine.run_postprocessing()

    # 打印汇总
    summary = engine.generate_summary(results)
    print(summary)

    return results


def demo_additional_modules():
    """
    演示其他独立模块.
    """
    print("\n" + "-" * 50)
    print("  5. 补充模块演示")
    print("-" * 50)

    # 5.1 布里渊区采样
    print("\n  [5.1] 布里渊区三角采样")
    bz = BrillouinZoneSampler('cubic', 3.96e-10, 4)
    print(f"    k 点数: {len(bz.k_points)}")
    print(f"    高对称点: {list(bz.high_sym_points.keys())}")

    # 三角形直方图
    k_2d = bz.k_points[:, :2] if bz.k_points.shape[1] >= 2 else \
        np.column_stack([bz.k_points[:, 0], np.zeros(len(bz.k_points))])
    # 映射到 [0,1]²
    k_min = np.min(k_2d, axis=0)
    k_max = np.max(k_2d, axis=0)
    k_range = k_max - k_min
    k_range[k_range < 1e-30] = 1.0
    k_normalized = (k_2d - k_min) / k_range
    hist, stats = bz.triangle_histogram_2d(k_normalized, 4)
    print(f"    三角直方图均匀性: {stats['uniformity']:.4f}")

    # 5.2 磁电界
    print("\n  [5.2] 磁电系数界")
    mb = MagnetoelectricBounds(chi_e=100, chi_m=0.01)
    bounds = mb.compute_lp_bounds()
    print(f"    CS 界: ±{bounds['cs_bound']:.4e} s/m")

    A, b, names = mb.build_conservation_matrix()
    print(f"    约束矩阵: {A.shape[0]}×{A.shape[1]}")
    print(f"    变量: {names}")

    # 5.3 熵迹
    print("\n  [5.3] 熵迹分析")
    et = MagnetoelectricEntropyTrace(n_modes=64, seed=285)
    t_vals = np.logspace(-1, 1, 15)
    t_out, theta = et.compute_entropy_trace(t_vals)
    print(f"    Θ(t) 范围: [{np.min(theta):.4e}, {np.max(theta):.4e}]")

    # 5.4 MD 分析
    print("\n  [5.4] MD 轨迹分析")
    md = MDTrajectoryAnalyzer()
    traj = md.generate_synthetic_trajectory(n_steps=200, T=300)
    d_stats = md.analyze_distances(traj)
    e_stats = md.analyze_energies(traj)
    print(f"    Fe-O 平均距离: {d_stats['d_FeO_angstrom']['mean']:.3f} Å")
    print(f"    总能涨落: {e_stats['E_total_eV']['fluctuation']:.6f} eV²")

    # 5.5 互信息
    print("\n  [5.5] 互信息计算")
    P_samples = np.random.randn(500)
    M_samples = 0.5 * P_samples + 0.5 * np.random.randn(500)
    I_pm, H_p, H_m, H_pm = et.mutual_information(P_samples, M_samples)
    print(f"    I(P;M) = {I_pm:.4f} nats")
    print(f"    H(P) = {H_p:.4f}, H(M) = {H_m:.4f}, H(P,M) = {H_pm:.4f}")


def main():
    """主函数."""
    print_header()

    # 1. 数值方法验证
    demo_numerical_methods()

    # 2. 自由能计算
    demo_free_energy()

    # 3. 主模拟
    engine = demo_simulation()

    # 4. 后处理分析
    results = demo_analysis(engine)

    # 5. 补充模块
    demo_additional_modules()

    print("\n" + "=" * 70)
    print("  模拟完成. 所有模块运行正常.")
    print("=" * 70)
    print()


if __name__ == '__main__':
    main()
