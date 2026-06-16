#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py
========
统一入口: 空间等离子体波粒相互作用 — 高阶有限差分与稳定性分析

====================================================================
计算等离子体: 空间等离子体波粒相互作用
高阶有限差分与稳定性分析 (小规模可复现实验)
====================================================================

本程序模拟以下物理过程:

1. Landau 阻尼 (线性 Vlasov-Poisson)
   - 初始扰动 Maxwellian 分布
   - 半拉格朗日求解 Vlasov 方程
   - 测量电场能量衰减速率
   - 与理论 Landau 阻尼率比较

2. 双流不稳定性
   - 双束分布函数
   - 观测电场指数增长
   - 测量增长率并与理论比较

3. 高阶有限差分精度验证
   - 2阶、4阶、6阶差分格式
   - 收敛性测试

4. 波-粒子共振分析
   - Quasilinear 扩散系数
   - 粒子轨道追踪 (Henon 映射)
   - 共振区相空间结构

融合种子项目 (15个):
  1004_vmc_pde, 305_dist_plot, 1282_eig_mom, 1422_xyl_display,
  321_dueling_idiots, 517_henon_orbit, 602_jaccard_distance,
  379_fem_to_medit, 886_polygon_integrals, 1142_PINNs,
  980_r8gd, 063_backward_euler, 1362_sparse_grid,
  329_ellipse_distance, 568_i4lib

使用方法:
  python main.py  (零参数运行)
"""

import sys
import os
import numpy as np
import math
import time

# 确保模块路径正确
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grid_integer_lib import (
    i4_choose, i4_factorial, i4_double_factorial,
    i4_bit_hi1, plasma_dispersion_z, IntegerRNG
)
from banded_matrix_ops import (
    BandedMatrix, make_dif2_matrix, make_high_order_laplacian
)
from high_order_fd import (
    fd_weights, FDCoefficientMatrix, FourthOrderDissipationOperator,
    spectral_resolution_analysis, BackwardEulerVlasov
)
from mesh_topology import (
    PlasmaPhaseSpaceMesh, PhaseSpacePartition,
    MeshDataConverter, moment_polygon, moment_normalized,
    dcircle, dpoly, dellipse
)
from wave_particle_resonance import (
    QuasilinearDiffusion, chirikov_overlap_parameter,
    PlasmaHenonMap, BrownianCollisionOperator,
    maxwellian_1d, maxwellian_derivative,
    landau_resonance_condition, cyclotron_frequency
)
from vlasov_evolution import (
    VlasovPoissonSolver, ElectrostaticFieldSolver
)
from stability_analysis import (
    NumericalStabilityAnalyzer, PlasmaDispersionSolver,
    HankelModeDecomposition, TwoStreamInstabilityAnalyzer,
    SparseGridStability
)
from phase_space_diagnostics import (
    PhaseSpaceDiagnostician, density, current_density,
    kinetic_energy, field_energy, gibbs_entropy,
    distribution_support_similarity, jaccard_distance
)
from velocity_space_quadrature import (
    VelocityQuadrature, GaussHermiteQuadrature,
    PolygonMomentCalculator
)
from signed_distance_geom import (
    EllipseSampler, MagnetosphereTopology,
    dist_circle, dist_ellipse, sdf_union, sdf_intersection
)


def print_header():
    """打印程序标题."""
    print("=" * 70)
    print(" 空间等离子体波粒相互作用")
    print(" 高阶有限差分与稳定性分析 (小规模可复现实验)")
    print("=" * 70)
    print()
    print(" 物理模型: 1D 静电 Vlasov-Poisson 系统")
    print(" 数值方法: 半拉格朗日 + 谱方法 Poisson 求解器")
    print(" 分析工具: von Neumann 稳定性 / Hankel 模态分解")
    print()


def run_landau_damping():
    """
    实验 1: Landau 阻尼基准测试

    物理设置:
      f₀(x,v) = (1/(2π))^{1/2} (1 + α cos(kx)) exp(-v²/2)
      α = 0.01 (小扰动), k = 0.5

    理论 Landau 阻尼率:
      γ_L ≈ -0.01154 (for k=0.5, v_th=1)

    观测: 电场能量 |E|² ∝ exp(2γ_L t)
    """
    print("-" * 60)
    print(" 实验 1: Landau 阻尼")
    print("-" * 60)

    # 物理参数
    omega_p = 1.0
    v_th = 1.0
    n0 = 1.0
    alpha = 0.01     # 扰动振幅
    k0 = 0.5         # 扰动波数

    # 数值参数
    nx = 32
    nv = 64
    x_min, x_max = 0.0, 2.0 * math.pi / k0
    v_min, v_max = -6.0 * v_th, 6.0 * v_th

    print(f"  网格: nx={nx}, nv={nv}")
    print(f"  空间: [{x_min:.2f}, {x_max:.2f}], dx={((x_max-x_min)/nx):.4f}")
    print(f"  速度: [{v_min:.2f}, {v_max:.2f}], dv={((v_max-v_min)/nv):.4f}")

    # 理论值
    solver_disp = PlasmaDispersionSolver(omega_p, v_th)
    omega_theory = solver_disp.solve_dispersion(k0)
    gamma_theory = omega_theory.imag
    omega_bg = solver_disp.bohmgross_frequency(k0)
    print(f"\n  理论值:")
    print(f"    Bohm-Gross 频率: ω = {omega_bg:.6f}")
    print(f"    Landau 阻尼率:   γ = {gamma_theory:.6f}")
    print(f"    精确色散关系:    ω = {omega_theory.real:.6f} + {omega_theory.imag:.6f}i")

    # 初始分布函数
    x = np.linspace(x_min + (x_max - x_min) / (2 * nx),
                     x_max - (x_max - x_min) / (2 * nx), nx)
    v = np.linspace(v_min + (v_max - v_min) / (2 * nv),
                     v_max - (v_max - v_min) / (2 * nv), nv)
    xx, vv = np.meshgrid(x, v, indexing='ij')
    f0 = (1.0 / (math.sqrt(2.0 * math.pi) * v_th)
          * (1.0 + alpha * np.cos(k0 * xx))
          * np.exp(-0.5 * (vv / v_th) ** 2))

    # 创建求解器
    solver = VlasovPoissonSolver(nx, nv, x_min, x_max, v_min, v_max)

    # CFL 条件
    dx = (x_max - x_min) / nx
    dv = (v_max - v_min) / nv
    dt = 0.5 * min(dx / (np.max(np.abs(v)) + 1e-10),
                    dv / (alpha * omega_p + 1e-10))
    dt = min(dt, 0.2)

    print(f"\n  时间步长: dt = {dt:.4f}")
    print(f"  CFL 数:   ~{np.max(np.abs(v)) * dt / min(dx, dv):.3f}")

    # 运行模拟
    n_steps = 100
    print(f"\n  运行 {n_steps} 步...")
    t_start = time.time()
    history = solver.run_simulation(f0, dt, n_steps, output_interval=25)
    elapsed = time.time() - t_start
    print(f"  完成! 耗时: {elapsed:.2f}s")

    # 从电场历史提取阻尼率
    E_hist = history.get('field_energy', [])
    t_hist = history.get('time', [])

    if len(E_hist) > 10 and all(e > 1e-30 for e in E_hist):
        log_E = np.log(np.maximum(np.array(E_hist[10:]), 1e-30))
        t_fit = np.array(t_hist[10:])
        if len(t_fit) > 2:
            coeffs = np.polyfit(t_fit, log_E, 1)
            gamma_numerical = coeffs[0] / 2.0
            print(f"\n  数值结果:")
            print(f"    电场衰减速率: γ_num = {gamma_numerical:.6f}")
            print(f"    理论阻尼率:   γ_th  = {gamma_theory:.6f}")
            if abs(gamma_theory) > 1e-10:
                rel_error = abs((gamma_numerical - gamma_theory)
                                / gamma_theory)
                print(f"    相对误差:       {rel_error * 100:.1f}%")
    else:
        print("  (电场数据不足，跳过阻尼率拟合)")

    # 相空间诊断
    diagnostician = PhaseSpaceDiagnostician(x, v)
    E_final = history.get('E_final', np.zeros(nx))
    f_final = history.get('f_final', f0)
    diagnostics = diagnostician.compute_all(f_final, E_final)
    print(f"\n  终态诊断:")
    print(f"    最大密度:     {diagnostics['max_density']:.4e}")
    print(f"    动能:         {diagnostics['kinetic_energy']:.4e}")
    print(f"    电场能:       {diagnostics['field_energy']:.4e}")
    print(f"    Gibbs 熵:     {diagnostics['entropy']:.4e}")
    print(f"    总粒子数:     {diagnostics['total_particles']:.4e}")

    return history


def run_two_stream():
    """
    实验 2: 双流不稳定性

    初始分布:
      f₀(v) = (n_b/2) [G(v-v_b) + G(v+v_b)]
    其中 G 为归一化 Maxwellian.

    不稳定性条件: v_b > v_th
    最大增长率: k_max ≈ ω_p / (√3 v_b)
    """
    print("\n" + "-" * 60)
    print(" 实验 2: 双流不稳定性")
    print("-" * 60)

    v_b = 2.0
    v_th = 0.5
    omega_p = 1.0

    # 理论分析
    ts_analyzer = TwoStreamInstabilityAnalyzer(v_b, v_th, omega_p)
    k_max_theory, gamma_max_theory = ts_analyzer.most_unstable_mode()
    print(f"  束流速度:   v_b = {v_b}")
    print(f"  热速度:     v_th = {v_th}")
    print(f"  最不稳定模: k = {k_max_theory:.4f}, γ = {gamma_max_theory:.4f}")

    # 数值模拟
    nx = 32
    nv = 64
    x_min, x_max = 0.0, 2.0 * math.pi / k_max_theory
    v_min, v_max = -8.0 * v_th - v_b, 8.0 * v_th + v_b

    solver = VlasovPoissonSolver(nx, nv, x_min, x_max, v_min, v_max)

    # 双束分布
    x = np.linspace(x_min + (x_max - x_min) / (2 * nx),
                     x_max - (x_max - x_min) / (2 * nx), nx)
    v = np.linspace(v_min + (v_max - v_min) / (2 * nv),
                     v_max - (v_max - v_min) / (2 * nv), nv)
    xx, vv = np.meshgrid(x, v, indexing='ij')

    f_plus = np.exp(-0.5 * ((vv - v_b) / v_th) ** 2)
    f_minus = np.exp(-0.5 * ((vv + v_b) / v_th) ** 2)
    f0 = 0.5 * (f_plus + f_minus) / (math.sqrt(2.0 * math.pi) * v_th)
    # 添加小扰动触发不稳定性
    f0 *= (1.0 + 0.01 * np.cos(k_max_theory * xx))

    # 时间步长
    dx = (x_max - x_min) / nx
    dv = (v_max - v_min) / nv
    dt = min(0.3 / gamma_max_theory if gamma_max_theory > 0 else 0.1, 0.15)

    n_steps = 80
    print(f"\n  运行 {n_steps} 步 (dt={dt:.4f})...")
    history = solver.run_simulation(f0, dt, n_steps, output_interval=20)

    # 分析增长
    E_hist = history.get('field_energy', [])
    t_hist = history.get('time', [])
    if len(E_hist) > 5:
        # 线性增长阶段拟合
        E_arr = np.array(E_hist)
        positive_mask = E_arr > 1e-20
        if np.sum(positive_mask) > 5:
            log_E = np.log(E_arr[positive_mask])
            t_fit = np.array(t_hist)[positive_mask]
            # 取中间段拟合
            n_fit = len(t_fit)
            i_start = n_fit // 4
            i_end = 3 * n_fit // 4
            if i_end > i_start + 2:
                coeffs = np.polyfit(t_fit[i_start:i_end],
                                     log_E[i_start:i_end], 1)
                gamma_num = coeffs[0] / 2.0
                print(f"\n  数值增长率: γ_num = {gamma_num:.4f}")
                print(f"  理论增长率: γ_th  = {gamma_max_theory:.4f}")

    # 诊断
    x_arr = solver.x
    v_arr = solver.v
    diag = PhaseSpaceDiagnostician(x_arr, v_arr)
    f_final = history.get('f_final', f0)
    E_final = history.get('E_final', np.zeros(nx))
    d = diag.compute_all(f_final, E_final)
    print(f"\n  终态:")
    print(f"    电场能:     {d['field_energy']:.4e}")
    print(f"    最大密度:   {d['max_density']:.4e}")
    print(f"    支撑集 Jaccard: "
          f"{distribution_support_similarity(f0, f_final):.3f}")

    return history


def run_fd_accuracy():
    """
    实验 3: 高阶有限差分精度验证

    对函数 f(x) = sin(kx) 测试 P 阶差分的收敛性.
    理论: 误差 ∝ h^P
    """
    print("\n" + "-" * 60)
    print(" 实验 3: 高阶有限差分精度验证")
    print("-" * 60)

    k_wave = 2.0 * math.pi
    f_exact = lambda x: np.sin(k_wave * x)
    df_exact = lambda x: k_wave * np.cos(k_wave * x)

    for order in [2, 4, 6]:
        n_test = 64
        x = np.linspace(0, 1.0, n_test, endpoint=False)
        h = x[1] - x[0]

        grid = np.arange(-order // 2, order // 2 + 1, dtype=float) * h
        w = fd_weights(grid, 0.0, 1)[:, 1]

        f_vals = f_exact(x)
        df_num = np.zeros(n_test)
        hw = order // 2

        for i in range(hw, n_test - hw):
            for j_idx, j in enumerate(range(-hw, hw + 1)):
                df_num[i] += w[j_idx] * f_vals[i + j] / h

        # 边界用低阶格式
        for i in range(hw):
            df_num[i] = (f_vals[(i + 1) % n_test] - f_vals[i]) / h
        for i in range(n_test - hw, n_test):
            df_num[i] = (f_vals[i] - f_vals[(i - 1) % n_test]) / h

        error = np.sqrt(np.mean((df_num - df_exact(x)) ** 2))
        print(f"  {order}阶差分: L2 误差 = {error:.4e}  (h = {h:.4f})")

    # 收敛率分析
    print("\n  网格收敛性:")
    for order in [2, 4]:
        errors = []
        hs = []
        for n in [16, 32, 64, 128]:
            x = np.linspace(0, 1.0, n, endpoint=False)
            h = x[1] - x[0]
            hs.append(h)

            grid = np.arange(-order // 2, order // 2 + 1, dtype=float) * h
            w = fd_weights(grid, 0.0, 1)[:, 1]
            f_vals = f_exact(x)
            df_num = np.zeros(n)
            hw = order // 2

            for i in range(hw, n - hw):
                for j_idx, j in enumerate(range(-hw, hw + 1)):
                    df_num[i] += w[j_idx] * f_vals[i + j] / h
            for i in range(hw):
                df_num[i] = (f_vals[(i + 1) % n] - f_vals[i]) / h
            for i in range(n - hw, n):
                df_num[i] = (f_vals[i] - f_vals[(i - 1) % n]) / h

            err = np.sqrt(np.mean((df_num - df_exact(x)) ** 2))
            errors.append(err)
            print(f"    P={order}, N={n:4d}: error = {err:.4e}")

        if errors[0] > 1e-30 and errors[-1] > 1e-30:
            rate = (math.log(errors[0]) - math.log(errors[-1])) / \
                   (math.log(hs[-1]) - math.log(hs[0]))
            print(f"    收敛阶: {rate:.2f}  (理论值: {order})")


def run_backward_euler_test():
    """
    实验 4: Backward Euler 隐式时间积分测试

    测试方程: dy/dt = -λy, y(0) = 1
    精确解: y(t) = exp(-λt)
    """
    print("\n" + "-" * 60)
    print(" 实验 4: Backward Euler 隐式时间积分")
    print("-" * 60)

    lam = 5.0
    T = 2.0
    y0_val = 1.0
    n_test = 50

    def rhs(y):
        return -lam * y

    # 不同时间步长
    for n_steps in [20, 50, 100]:
        dt = T / n_steps
        be = BackwardEulerVlasov(1, dt, tol=1e-10, max_iter=30)

        y = np.array([y0_val])
        for _ in range(n_steps):
            y = be.step(y, lambda yv: rhs(yv[0]).reshape(1))

        y_exact = y0_val * math.exp(-lam * T)
        error = abs(y[0] - y_exact)
        print(f"  N={n_steps:4d} (dt={dt:.4f}): "
              f"y_num={y[0]:.6f}, y_exact={y_exact:.6f}, "
              f"error={error:.4e}")


def run_wave_particle_analysis():
    """
    实验 5: 波-粒子共振与混沌轨道分析
    """
    print("\n" + "-" * 60)
    print(" 实验 5: 波-粒子共振与混沌轨道")
    print("-" * 60)

    # Quasilinear 扩散
    v_grid = np.linspace(-5.0, 5.0, 64)
    omega = complex(1.2, -0.05)
    k_wave = 0.5
    E0 = 0.1

    ql = QuasilinearDiffusion(v_grid, k_wave, E0)
    D = ql.diffusion_coefficient(omega)
    print(f"  Quasilinear 扩散:")
    print(f"    max(D) = {np.max(D):.4e}")
    print(f"    共振速度 v_φ = {omega.real / k_wave:.4f}")

    # Henon 映射轨道
    print(f"\n  Henon 型辛映射:")
    for alpha in [0.3, 0.8, 1.2]:
        hmap = PlasmaHenonMap(alpha)
        lyap = hmap.lyapunov_exponent(0.1, 0.1, n_steps=2000)
        bounded = hmap.is_bounded(0.1, 0.1, n_check=500)
        print(f"    α={alpha:.1f}: λ_Lyap = {lyap:.4f}, "
              f"有界 = {bounded}")

    # Chirikov 重叠
    k_arr = np.array([0.3, 0.5, 0.7])
    amp = np.array([0.1, 0.15, 0.1])
    s = chirikov_overlap_parameter(k_arr, amp)
    print(f"\n  Chirikov 重叠参数: s = {s:.4f}")
    print(f"    {'s > 1 → 全局混沌' if s > 1 else 's < 1 → 局部混沌'}")

    # Brownian 碰撞
    nu_coll = 0.05
    coll = BrownianCollisionOperator(v_grid, nu_coll, v_th=1.0)
    f_test = maxwellian_1d(v_grid, 1.0, 1.0) * (1.0 + 0.1 * np.sin(v_grid))
    Cf = coll.apply(f_test)
    f_eq = coll.equilibrium()
    print(f"\n  碰撞算子:")
    print(f"    max|C[f]| = {np.max(np.abs(Cf)):.4e}")
    print(f"    ||f - f_eq|| = {np.sqrt(np.sum((f_test - f_eq) ** 2)):.4e}")


def run_phase_space_diagnostics():
    """
    实验 6: 相空间诊断与几何分析
    """
    print("\n" + "-" * 60)
    print(" 实验 6: 相空间诊断与几何分析")
    print("-" * 60)

    # 速度矩 (多边形积分)
    vertices_x = np.array([0.0, 4.0, 4.0, 0.0])
    vertices_y = np.array([0.0, 0.0, 3.0, 3.0])
    pmc = PolygonMomentCalculator(vertices_x, vertices_y)

    print("  多边形矩积分 (Steger 1996):")
    print(f"    面积:   A = {pmc.area():.4f}")
    cx, cy = pmc.centroid()
    print(f"    质心:   ({cx:.4f}, {cy:.4f})")
    print(f"    ν₂₀ = {pmc.moment(2, 0):.4f}")
    print(f"    ν₀₂ = {pmc.moment(0, 2):.4f}")
    print(f"    ν₁₁ = {pmc.moment(1, 1):.4f}")

    # 椭圆采样
    a_ell, b_ell = 3.0, 2.0
    sampler = EllipseSampler(a_ell, b_ell)
    print(f"\n  椭圆采样 (a={a_ell}, b={b_ell}):")
    print(f"    偏心率: e = {sampler.eccentricity:.4f}")
    mu, var = sampler.distance_stats(500)
    print(f"    平均距离: μ = {mu:.4f}, 方差: σ² = {var:.4f}")
    print(f"    曲率(t=0):   κ = {sampler.curvature(0):.4f}")
    print(f"    曲率(t=π/2): κ = {sampler.curvature(math.pi / 2):.4f}")

    # 磁层拓扑
    mag = MagnetosphereTopology(r_mp=10.0, r_tail=30.0)
    test_pts = np.array([[5.0, 0.0], [15.0, 5.0], [25.0, 0.0],
                          [-20.0, 1.0]])
    print(f"\n  磁层拓扑:")
    d_mp = mag.magnetopause_sdf(test_pts)
    for i in range(len(test_pts)):
        inside = "内部" if d_mp[i] < 0 else "外部"
        print(f"    ({test_pts[i, 0]:5.1f}, {test_pts[i, 1]:5.1f}): "
              f"φ = {d_mp[i]:.2f} → {inside}")

    # Jaccard 距离
    f1 = np.array([1, 2, 3, 4, 5])
    f2 = np.array([3, 4, 5, 6, 7])
    jd = jaccard_distance(f1, f2)
    print(f"\n  Jaccard 距离:")
    print(f"    A = {f1}, B = {f2}")
    print(f"    d_J(A,B) = {jd:.4f}")


def run_stability_analysis():
    """
    实验 7: von Neumann 稳定性与色散分析
    """
    print("\n" + "-" * 60)
    print(" 实验 7: von Neumann 稳定性分析")
    print("-" * 60)

    schemes = ['upwind', 'centered', 'lax_wendroff', 'leapfrog']
    for scheme in schemes:
        analyzer = NumericalStabilityAnalyzer(scheme)
        nu_range, max_G = analyzer.stability_boundary()

        # 找临界 CFL
        stable_mask = max_G <= 1.001
        if np.any(stable_mask):
            nu_crit = nu_range[stable_mask][-1]
        else:
            nu_crit = 0.0

        print(f"  {scheme:14s}: 临界 CFL ≈ {nu_crit:.3f}")

    # 谱分辨率
    print("\n  差分格式谱分辨率:")
    for order in [2, 4, 6]:
        theta, error = spectral_resolution_analysis(order)
        # 找误差 < 1% 的范围
        small_err = theta[error < 0.01]
        if len(small_err) > 0:
            theta_max = small_err[-1]
            print(f"    {order}阶: 1%误差范围 θ < {theta_max:.3f} "
                  f"({theta_max / math.pi * 100:.0f}% Nyquist)")
        else:
            print(f"    {order}阶: 1%误差范围极小")

    # 稀疏网格
    print("\n  稀疏网格求积:")
    for level in [3, 5, 7]:
        nodes, weights = SparseGridStability.gauss_hermite_nodes_weights(level)
        print(f"    level={level}: {len(nodes)} 节点, "
              f"∫exp(-x²)dx ≈ {np.sum(weights):.6f} (精确: √π={math.sqrt(math.pi):.6f})")


def run_matrix_operations():
    """
    实验 8: 带状矩阵运算与特征值分析
    """
    print("\n" + "-" * 60)
    print(" 实验 8: 带状矩阵运算")
    print("-" * 60)

    # 二阶差分矩阵
    n = 16
    L2 = make_dif2_matrix(n)
    x_test = np.sin(math.pi * np.arange(n) / (n - 1))
    Lx = L2.mv(x_test)
    print(f"  二阶差分矩阵 ({n}×{n}):")
    print(f"    对角线数: {L2.ndiag}")
    print(f"    ||L·sin(πx)||∞ = {np.max(np.abs(Lx)):.4e}")

    # 特征值
    eigs = L2.eigenvalues_tridiagonal()
    print(f"    λ_min = {eigs[0]:.4f}, λ_max = {eigs[-1]:.4f}")
    print(f"    谱条件数: {eigs[-1] / max(eigs[0], 1e-30):.2f}")

    # 高阶 Laplacian
    for order in [4, 6]:
        L_p = make_high_order_laplacian(n, order)
        print(f"  {order}阶 Laplacian: ndiag={L_p.ndiag}")

    # 线性系统求解
    b = np.ones(n)
    x_sol = L2.solve(b)
    residual = np.max(np.abs(L2.mv(x_sol) - b))
    print(f"\n  线性系统求解 (Lx = 1):")
    print(f"    残差: {residual:.4e}")

    # 整数库演示
    print(f"\n  整数运算库:")
    print(f"    C(20, 10) = {i4_choose(20, 10)}")
    print(f"    10! = {i4_factorial(10)}")
    print(f"    9!! = {i4_double_factorial(9)}")
    print(f"    bit_hi1(1024) = {i4_bit_hi1(1024)}")


def run_quadrature():
    """
    实验 9: 速度空间求积精度
    """
    print("\n" + "-" * 60)
    print(" 实验 9: 速度空间求积")
    print("-" * 60)

    v_grid = np.linspace(-6.0, 6.0, 64)
    f_maxwell = maxwellian_1d(v_grid, 1.0, 1.0)

    # 梯形 vs Simpson
    for method in ['trapezoidal', 'simpson']:
        quad = VelocityQuadrature(v_grid, method=method)
        M0 = quad.moment(f_maxwell, 0)
        M2 = quad.moment(f_maxwell, 2)
        v_th = quad.thermal_speed(f_maxwell)
        print(f"  {method:12s}: M0={M0:.6f}, M2={M2:.6f}, "
              f"v_th={v_th:.6f}")

    # Gauss-Hermite
    for n_pts in [8, 16, 32]:
        gh = GaussHermiteQuadrature(n_pts, v_th=1.0)
        # ∫ v² exp(-v²) dv = √π / 2
        integrand = gh.v ** 2 * np.exp(gh.v ** 2)  # 抵消权重
        val = gh.integrate(integrand)
        exact = math.sqrt(math.pi) / 2.0
        print(f"  Gauss-Hermite ({n_pts:2d}pts): "
              f"∫v²exp(-v²)dv = {val:.6f} (exact={exact:.6f})")


def main():
    """主函数: 运行全部实验."""
    print_header()

    t_total = time.time()

    # 运行各实验
    try:
        run_landau_damping()
    except Exception as e:
        print(f"  [ERROR] Landau 阻尼实验失败: {e}")

    try:
        run_two_stream()
    except Exception as e:
        print(f"  [ERROR] 双流实验失败: {e}")

    try:
        run_fd_accuracy()
    except Exception as e:
        print(f"  [ERROR] 有限差分精度测试失败: {e}")

    try:
        run_backward_euler_test()
    except Exception as e:
        print(f"  [ERROR] Backward Euler 测试失败: {e}")

    try:
        run_wave_particle_analysis()
    except Exception as e:
        print(f"  [ERROR] 波-粒子分析失败: {e}")

    try:
        run_phase_space_diagnostics()
    except Exception as e:
        print(f"  [ERROR] 相空间诊断失败: {e}")

    try:
        run_stability_analysis()
    except Exception as e:
        print(f"  [ERROR] 稳定性分析失败: {e}")

    try:
        run_matrix_operations()
    except Exception as e:
        print(f"  [ERROR] 矩阵运算失败: {e}")

    try:
        run_quadrature()
    except Exception as e:
        print(f"  [ERROR] 求积测试失败: {e}")

    # 总结
    elapsed_total = time.time() - t_total
    print("\n" + "=" * 70)
    print(" 全部实验完成!")
    print(f" 总耗时: {elapsed_total:.2f}s")
    print("=" * 70)

    # 写入结果摘要
    result_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "simulation_results.txt")
    try:
        with open(result_file, 'w') as fout:
            fout.write("空间等离子体波粒相互作用模拟结果摘要\n")
            fout.write("=" * 50 + "\n\n")
            fout.write(f"运行时间: {elapsed_total:.2f}s\n")
            fout.write("实验列表:\n")
            fout.write("  1. Landau 阻尼基准测试\n")
            fout.write("  2. 双流不稳定性\n")
            fout.write("  3. 高阶有限差分精度验证\n")
            fout.write("  4. Backward Euler 隐式积分\n")
            fout.write("  5. 波-粒子共振与混沌轨道\n")
            fout.write("  6. 相空间诊断与几何分析\n")
            fout.write("  7. von Neumann 稳定性分析\n")
            fout.write("  8. 带状矩阵运算\n")
            fout.write("  9. 速度空间求积\n")
        print(f"\n 结果摘要已写入: {result_file}")
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
