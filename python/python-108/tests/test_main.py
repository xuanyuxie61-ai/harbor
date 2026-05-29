# -*- coding: utf-8 -*-
"""
main.py
光热耦合微腔传感器全耦合仿真平台 —— 统一入口

运行方式
--------
    python main.py

本程序零参数可运行，自动执行以下博士级科学计算流程：
    1. 微环谐振腔几何建模与三角网格生成
    2. 光场 Helmholtz 方程有限差分求解（5点/9点 stencil，带状矩阵 LU）
    3. 热传导方程稳态求解（Robin 边界，光吸收热源）
    4. 光热耦合自洽迭代（Picard + 松弛）
    5. 谐振模式本征值分析（二分法求谐振波长，幂迭代）
    6. 传感响应神经网络代理模型训练与预测
    7. 蒙特卡洛不确定性量化（材料参数随机扰动）
    8. 误差分析与收敛诊断（L1/L2 范数、Richardson 外推）
    9. 高维球面求积与 Vandermonde2D 积分验证

科学问题背景
------------
微环谐振腔（Micro-ring Resonator）是一种高品质因数（Q-factor）光学微腔，
广泛应用于生化传感、温度监测与光通信。当光在微环中谐振时，光吸收产生热量，
导致温度上升，进而通过热光效应改变材料折射率，反过来影响光场分布——此即
光热耦合（opto-thermal coupling）。在高功率或高灵敏度传感场景下，该耦合效应
不可忽略，必须自洽求解光-热-材料三场耦合问题。
"""

import numpy as np
import sys
import os

# 确保当前目录在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geometry_mesh import MicrocavityGeometry, CVV
from quadrature_engine import SphereQuadrature, Vandermonde2DQuadrature, GaussLegendreTensor
from helmholtz_fd import HelmholtzSolver, LaplacianStencils, BandedMatrixSolver, BorderedBandedSolver
from thermal_fd import ThermalSolver
from eigenvalue_solver import ResonanceEigenSolver, BisectionSolver, CollatzPolynomial
from photothermal_coupler import PhotothermalCoupler
from stochastic_uq import MonteCarloUQ, HypersphereSampler, RandomVariateGenerator
from nn_surrogate import SensorResponseSurrogate
from error_norms import ErrorNorms


def step1_geometry_and_mesh():
    """步骤1：微腔几何建模、网格生成、弧长计算与 CVV 结构测试"""
    print("=" * 60)
    print("步骤 1: 微腔几何建模与网格生成")
    print("=" * 60)

    geom = MicrocavityGeometry(
        R_major=10.0e-6,
        r_minor=1.5e-6,
        waveguide_gap=0.3e-6,
        waveguide_width=0.5e-6,
        n_ring=3.47,
        n_clad=1.44,
        n_env=1.00
    )

    # 弧长计算（梯形法则）
    arc_len = geom.arc_length_trapezoidal(0.0, 2 * np.pi, n=2000)
    theory = 2.0 * np.pi * geom.r_minor
    print(f"  微环截面周长 (梯形法, n=2000): {arc_len:.6e} m")
    print(f"  理论值 2πr:                      {theory:.6e} m")
    print(f"  弧长相对误差: {abs(arc_len - theory) / theory:.6e}")

    # 生成截面网格
    nodes, elements, markers = geom.generate_cross_section_mesh(nr=40, n_theta=60)
    print(f"  网格节点数: {nodes.shape[0]}, 三角单元数: {elements.shape[0]}")

    # 网格质量
    quality = geom.compute_mesh_quality(nodes, elements)
    print(f"  最小角: {quality['min_angle_deg']:.2f}°, 总面积: {quality['total_area']:.6e} m²")

    # CVV 数据结构测试（融入 147_cell）
    row_lengths = [3, 5, 4, 6, 2]
    cvv = CVV(row_lengths, dtype=float)
    for i in range(len(row_lengths)):
        cvv.set_row(i, np.arange(row_lengths[i], dtype=float))
    print(f"  CVV 测试: 行数={cvv.size()[0]}, 总元素={cvv.size()[1]}")
    print(f"  CVV 第2行: {cvv.get_row(2)}")

    # FEM I/O 测试（融入 376_fem_io）
    geom.fem_write_nodes(nodes, "nodes.txt")
    geom.fem_write_elements(elements, "elements.txt")
    nodes_read = geom.fem_read_nodes("nodes.txt")
    elements_read = geom.fem_read_elements("elements.txt")
    assert np.allclose(nodes, nodes_read)
    assert np.array_equal(elements, elements_read)
    print("  FEM I/O 读写验证通过")

    # 波导节点
    wg_nodes = geom.generate_waveguide_nodes(n_points=30)
    print(f"  耦合波导节点数: {wg_nodes.shape[0]}")

    return geom, nodes, elements, markers


def step2_quadrature_validation():
    """步骤2：数值积分引擎验证（球面积分 + 2D Vandermonde）"""
    print("\n" + "=" * 60)
    print("步骤 2: 数值积分引擎验证")
    print("=" * 60)

    # 球面精确积分（融入 1119_sphere_integrals）
    sq = SphereQuadrature()
    print(f"  单位球面面积: {sq.sphere01_area():.10f}")

    test_exponents = [(0, 0, 0), (2, 0, 0), (2, 2, 0), (4, 0, 0), (2, 2, 2)]
    for e in test_exponents:
        exact = sq.monomial_integral_exact(e)
        mc, err = sq.monte_carlo_integral(5000, e, seed=42)
        print(f"  单项式{e}: 精确值={exact:.6e}, MC估计={mc:.6e} ± {err:.6e}")

    # 2D Vandermonde 求积（融入 951_quadrature_weights_vandermonde_2d）
    vq = Vandermonde2DQuadrature()
    t = 2
    n_needed = (t + 1) * (t + 2) // 2
    # 在 [0,1]×[0,1] 上构造节点
    xs = np.array([0.1, 0.5, 0.9, 0.2, 0.8, 0.5])
    ys = np.array([0.1, 0.2, 0.1, 0.8, 0.8, 0.5])
    xs, ys = xs[:n_needed], ys[:n_needed]
    w = vq.compute_weights(xs, ys, t, rect_a=0.0, rect_b=1.0, rect_c=0.0, rect_d=1.0)
    # 测试对 f(x,y)=1 的积分
    val_const = vq.integrate(np.ones_like(w), w)
    print(f"  Vandermonde2D 对常数1积分: {val_const:.6f} (期望 1.0)")
    # 测试对 x*y 积分
    val_xy = vq.integrate(xs * ys, w)
    print(f"  Vandermonde2D 对 xy 积分: {val_xy:.6f} (期望 0.25)")

    # 高斯-勒让德张量积积分
    gt = GaussLegendreTensor()
    f_test = lambda x, y: np.sin(np.pi * x) * np.cos(np.pi * y)
    approx = gt.tensor_quad_2d(f_test, 0.0, 1.0, 0.0, 1.0, m=8)
    # 精确值: ∫₀¹ sin(πx) dx · ∫₀¹ cos(πy) dy = (2/π)·0 = 0
    print(f"  Gauss-Legendre 张量积 (sinπx·cosπy): {approx:.6e} (期望 ~0)")


def step3_helmholtz_and_thermal():
    """步骤3：Helmholtz 光场与热传导求解"""
    print("\n" + "=" * 60)
    print("步骤 3: Helmholtz 光场与热传导求解")
    print("=" * 60)

    # 建立计算域：4μm × 4μm，网格 41×41
    L = 4.0e-6
    nx = ny = 41
    k0 = 2.0 * np.pi / (1.55e-6)  # 1550 nm 真空波数

    # 构造环形折射率分布
    R_center = 10.0e-6  # 不匹配 L，但我们只在局部域模拟
    # 重新定义一个局部环中心
    cx, cy = L / 2, L / 2
    r_ring = 1.5e-6
    X, Y = np.meshgrid(np.linspace(0, L, nx), np.linspace(0, L, ny))
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    n_profile = np.where(dist <= r_ring, 3.47, 1.44)

    # Helmholtz 求解（融入 647_laplacian, 973_r8cb, 974_r8cbb）
    helm = HelmholtzSolver(nx, ny, L, L, k0, n_profile, boundary_value=0.0)
    # 使用高斯分布源避免点源奇异性
    source_mask = np.zeros((ny, nx), dtype=complex)
    cx_idx, cy_idx = nx // 2, ny // 2
    for j in range(ny):
        for i in range(nx):
            dx = (i - cx_idx) * helm.hx
            dy = (j - cy_idx) * helm.hy
            r2 = dx**2 + dy**2
            source_mask[j, i] = np.exp(-r2 / (2.0 * (0.3e-6)**2))
    E = helm.solve_for_rhs(source_mask.real) + 1j * helm.solve_for_rhs(source_mask.imag)
    intensity = helm.compute_optical_intensity(E)
    print(f"  光场求解完成: max|E|={np.max(np.abs(E)):.4e}, max I={np.max(intensity):.4e}")

    # 拉普拉斯 stencil 测试（融入 647_laplacian）
    u_test = np.sin(2 * np.pi * X / L) * np.sin(2 * np.pi * Y / L)
    Lu5 = LaplacianStencils.laplacian5_2d(u_test, L / (nx - 1))
    # 理论 Lu = - ( (2π/L)² + (2π/L)² ) · u = -2·(2π/L)²·u
    Lu_theory = -2.0 * (2.0 * np.pi / L) ** 2 * u_test
    err_L2 = np.sqrt(np.mean((Lu5[1:-1, 1:-1] - Lu_theory[1:-1, 1:-1]) ** 2))
    print(f"  5点 Laplacian L² 误差: {err_L2:.4e}")

    # 热传导求解（融入 647_laplacian, 973_r8cb）
    thermal = ThermalSolver(nx, ny, L, L, kappa=1.4e2, h_conv=10.0, T_ambient=300.0)
    Q = thermal.compute_absorbed_heat(intensity, alpha_abs=1.0e-3)
    T = thermal.solve_steady_state(Q, bc_type="robin")
    delta_n = thermal.compute_thermal_lens(T, dn_dT=1.86e-4)
    print(f"  温度场求解完成: max ΔT={np.max(T - 300.0):.4e} K, max Δn={np.max(delta_n):.4e}")

    return helm, thermal, E, T, delta_n


def step4_photothermal_coupling(helm, thermal, E, T, delta_n):
    """步骤4：光热耦合自洽求解"""
    print("\n" + "=" * 60)
    print("步骤 4: 光热耦合自洽迭代")
    print("=" * 60)

    coupler = PhotothermalCoupler(
        helm, thermal,
        dn_dT=1.86e-4, n0=3.47,
        alpha_abs=1.0e-3,
        max_iter=8, tol=1e-6
    )

    source_mask = np.zeros((helm.ny, helm.nx), dtype=bool)
    source_mask[helm.ny // 2, helm.nx // 2] = True
    E_c, T_c, n_c, iters = coupler.self_consistent_solve(source_mask, source_amplitude=1.0 + 0.0j)

    print(f"  自洽迭代次数: {iters}")
    print(f"  收敛后 max|T|: {np.max(T_c):.4f} K")
    print(f"  收敛后 mean|n|: {np.mean(n_c):.6f}")

    # 热源积分（融入 951_quadrature_weights_vandermonde_2d）
    Q_c = coupler.compute_heat_source(E_c)
    P_trap = coupler.integrate_heat_source(Q_c, "trapezoidal")
    P_vand = coupler.integrate_vandermonde_2d(Q_c, total_degree=2)
    print(f"  总吸收功率 (梯形法): {P_trap:.6e} W/m")
    print(f"  总吸收功率 (Vandermonde): {P_vand:.6e} W/m")

    # WGM 球面积分（融入 1119_sphere_integrals）
    P_sphere = coupler.wgm_mode_sphere_integral(E_c, R_sphere=10.0e-6)
    print(f"  等效球面积分功率: {P_sphere:.6e}")

    return coupler, E_c, T_c, n_c


def step5_eigenvalue_and_resonance():
    """步骤5：本征值分析与谐振波长计算"""
    print("\n" + "=" * 60)
    print("步骤 5: 本征值分析与谐振波长")
    print("=" * 60)

    solver = ResonanceEigenSolver(R_major=10.0e-6, n_nominal=3.47)

    # 二分法求谐振波长（融入 094_bisection）
    # 假设 n_eff 有微弱色散：n(λ) = 3.47 - 1e-4*(λ-1550)/1550
    def n_eff_func(lam_nm):
        return 3.47 - 1e-4 * (lam_nm - 1550.0) / 1550.0

    m = 140  # 方位角模式数
    lambda_res, iters = solver.find_resonance_wavelength(m, n_eff_func, 1500.0, 1600.0)
    print(f"  模式 m={m} 的谐振波长: {lambda_res:.4f} nm (二分迭代 {iters} 次)")

    # 自由光谱范围 FSR
    ng = 3.5
    fsr = solver.compute_mode_spacing(m, lambda_res, n_eff_func(lambda_res), ng)
    print(f"  自由光谱范围 FSR: {fsr * 1e9:.4f} nm")

    # 灵敏度
    S = solver.sensitivity_dlambda_dn(m, lambda_res, n_eff_func(lambda_res))
    print(f"  灵敏度 dλ/dn: {S:.4f} nm/RIU")

    # 幂迭代本征值（融入 198_collatz_polynomial 的迭代思想）
    # 构造一个小型测试矩阵
    n_test = 50
    A_test = np.diag(np.linspace(1.0, 10.0, n_test)) + 0.1 * np.random.default_rng(42).random((n_test, n_test))
    lam, vec, it_ev = solver.power_iteration_eigenvalue(A_test, max_iter=500, tol=1e-10)
    print(f"  幂迭代最大本征值: {lam:.6f} (迭代 {it_ev} 次)")

    # Collatz-like 多项式动力学（融入 198_collatz_polynomial）
    poly = CollatzPolynomial(np.array([1, 0, 1, 1]))  # 1 + x² + x³
    seq = poly.sequence(max_steps=20)
    print(f"  Collatz 多项式序列长度: {len(seq)}, 终止次数: {seq[-1].sum() if len(seq[-1])>0 else 0}")

    # 平滑不动点
    fp = CollatzPolynomial.smooth_analog(0.5, max_iter=50)
    print(f"  Collatz 平滑映射不动点 (x₀=0.5): {fp:.6e}")

    return solver, lambda_res


def step6_neural_network_surrogate():
    """步骤6：神经网络代理模型训练与评估"""
    print("\n" + "=" * 60)
    print("步骤 6: 神经网络代理模型")
    print("=" * 60)

    surrogate = SensorResponseSurrogate()
    surrogate.train(n_points=200)

    # 测试预测
    test_cases = [
        (0.0, 1.00),
        (5.0, 1.02),
        (-5.0, 1.03),
        (10.0, 1.01),
    ]
    print("  代理模型预测 (ΔT_env, n_env) → Δλ [pm]:")
    for dt, ne in test_cases:
        pred = surrogate.predict(dt, ne)
        print(f"    ({dt:+.1f} K, {ne:.2f}) → {pred:+.3f} pm")

    # 评估指标
    X_test, Y_test = surrogate.generate_training_data(100)
    metrics = surrogate.nn.evaluate_metrics(X_test, Y_test)
    print(f"  测试集指标: MSE={metrics['mse']:.4f}, RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")

    return surrogate


def step7_stochastic_uq():
    """步骤7：蒙特卡洛不确定性量化"""
    print("\n" + "=" * 60)
    print("步骤 7: 蒙特卡洛不确定性量化")
    print("=" * 60)

    # 超球面采样统计（融入 563_hypersphere_angle）
    hyper = HypersphereSampler(dim=10, seed=42)
    stats = hyper.angle_statistics(n_pairs=2000)
    print(f"  10维超球面角度统计:")
    print(f"    E[|cos θ|] = {stats['mean_abs_cos']:.6f} (理论 {stats['theoretical_mean_abs_cos']:.6f})")
    print(f"    std|cos θ| = {stats['std_abs_cos']:.6f}")

    # 随机变量生成（融入 1012_ranlib）
    rng = RandomVariateGenerator(seed=42)
    samples = {
        "normal": rng.normal(0.0, 1.0, size=1000),
        "gamma": rng.gamma(2.0, 1.0, size=1000),
        "beta": rng.beta(2.0, 5.0, size=1000),
        "exponential": rng.exponential(1.0, size=1000),
    }
    for name, arr in samples.items():
        print(f"  {name}: mean={np.mean(arr):.4f}, std={np.std(arr, ddof=1):.4f}")

    # MC 不确定性传播
    mc = MonteCarloUQ(n_samples=200, seed=42)
    base_params = {
        "R_major": 10.0e-6,
        "r_minor": 1.5e-6,
        "n_ring": 3.47,
        "kappa": 140.0,
        "alpha_abs": 1.0e-3,
    }
    std_params = {
        "R_major": 0.05e-6,
        "r_minor": 0.02e-6,
        "n_ring": 0.01,
        "kappa": 5.0,
        "alpha_abs": 0.1e-3,
    }

    def simple_forward(p):
        # 简化的前向模型：谐振波长
        m = 120
        lambda0 = 1550.0
        n_g = 3.5
        # 近似：λ = 2π·R·n / m
        lam = 2.0 * np.pi * p["R_major"] * p["n_ring"] / m * 1e9
        fsr = lam ** 2 / (2.0 * np.pi * p["R_major"] * n_g) * 1e9
        return {"lambda_res_nm": lam, "fsr_pm": fsr * 1000.0}

    summary = mc.run_mc_propagation(base_params, std_params, simple_forward)
    print(f"  MC 传播: 成功样本 {summary['n_success']}/{summary['n_requested']}")
    for k in ["lambda_res_nm", "fsr_pm"]:
        s = summary[k]
        print(f"    {k}: mean={s['mean']:.4f}, std={s['std']:.4f}, [p5,p95]=[{s['p5']:.4f},{s['p95']:.4f}]")

    return summary


def step8_error_analysis(E, T, delta_n, helm, thermal):
    """步骤8：误差分析与范数计算"""
    print("\n" + "=" * 60)
    print("步骤 8: 误差分析与范数计算")
    print("=" * 60)

    err = ErrorNorms()

    # 对温度场计算 L1, L2 范数（融入 812_norm_l1）
    volumes = np.full_like(T, helm.hx * helm.hy)
    l1_T = err.l1_norm_discrete(T.flatten(), volumes.flatten())
    l2_T = err.l2_norm_discrete(T.flatten(), volumes.flatten())
    linf_T = err.linf_norm(T)
    print(f"  温度场 L¹ 范数: {l1_T:.4e}")
    print(f"  温度场 L² 范数: {l2_T:.4e}")
    print(f"  温度场 L∞ 范数: {linf_T:.4e}")

    # 质量指标
    metrics_T = err.compute_quality_metrics(T)
    print(f"  温度场动态范围: {metrics_T['dynamic_range_db']:.2f} dB")

    # 相对误差（模拟近似解与精确解对比）
    # 用解析近似 T_exact ≈ T + O(h²) 的 Richardson 外推思想
    # 这里仅演示范数计算
    T_approx = T + 1e-3 * np.random.default_rng(7).normal(size=T.shape)
    rel_l2 = err.relative_l2_error(T_approx, T, volumes)
    rel_l1 = err.relative_l1_error(T_approx, T, volumes)
    print(f"  扰动近似解相对 L² 误差: {rel_l2:.6e}")
    print(f"  扰动近似解相对 L¹ 误差: {rel_l1:.6e}")

    # 收敛阶估计（Richardson 外推思想）
    resolutions = np.array([0.1, 0.05, 0.025, 0.0125])
    errors = np.array([0.01, 0.0025, 0.000625, 0.000156])
    orders = err.convergence_order(errors, resolutions)
    print(f"  估计收敛阶: {orders}")

    # Richardson 外推
    extrap = err.richardson_extrapolation(errors[0] * np.ones(1), errors[1] * np.ones(1), p=2.0)
    print(f"  Richardson 外推值: {extrap[0]:.6e}")


def step9_bordered_banded_demo():
    """步骤9：边界带状矩阵求解演示（融入 974_r8cbb）"""
    print("\n" + "=" * 60)
    print("步骤 9: 边界带状矩阵 Schur 补求解演示")
    print("=" * 60)

    n1, n2 = 20, 4
    ml, mu = 2, 2
    # 构造测试带状矩阵 A1
    A1_dense = np.diag(np.ones(n1) * 4.0) + np.diag(np.ones(n1 - 1) * -1.0, 1) + np.diag(np.ones(n1 - 1) * -1.0, -1)
    A1_dense += np.diag(np.ones(n1 - 2) * -0.5, 2) + np.diag(np.ones(n1 - 2) * -0.5, -2)
    A2 = np.random.default_rng(33).random((n1, n2))
    A3 = np.random.default_rng(44).random((n2, n1))
    A4 = np.eye(n2) * 5.0 + np.random.default_rng(55).random((n2, n2)) * 0.1

    # 打包为带状紧凑存储
    lda = ml + mu + 1
    A1_band = np.zeros((lda, n1))
    for j in range(n1):
        for i in range(max(0, j - mu), min(n1, j + ml + 1)):
            A1_band[mu + i - j, j] = A1_dense[i, j]

    bbs = BorderedBandedSolver(n1, n2, ml, mu)
    bbs.set_blocks(A1_band, A2, A3, A4)
    bbs.factorize()

    b = np.random.default_rng(66).random(n1 + n2)
    x = bbs.solve(b)

    # 验证残差
    full_A = np.block([[A1_dense, A2], [A3, A4]])
    residual = full_A @ x - b
    res_norm = np.linalg.norm(residual)
    print(f"  残差范数 ‖Ax-b‖₂: {res_norm:.6e}")
    print(f"  解向量前5个分量: {x[:5]}")


def main():
    print("\n" + "#" * 60)
    print("#  光热耦合微腔传感器全耦合仿真平台")
    print("#  Opto-Thermal Coupled Microcavity Sensor Simulation")
    print("#" * 60 + "\n")

    # 依次执行各步骤
    geom, nodes, elements, markers = step1_geometry_and_mesh()
    step2_quadrature_validation()
    helm, thermal, E, T, delta_n = step3_helmholtz_and_thermal()
    coupler, E_c, T_c, n_c = step4_photothermal_coupling(helm, thermal, E, T, delta_n)
    solver, lambda_res = step5_eigenvalue_and_resonance()
    surrogate = step6_neural_network_surrogate()
    summary = step7_stochastic_uq()
    step8_error_analysis(E_c, T_c, delta_n, helm, thermal)
    step9_bordered_banded_demo()

    print("\n" + "=" * 60)
    print("仿真全部完成。所有模块零参数运行通过。")
    print("=" * 60)

    # 清理临时文件
    for tmp in ["nodes.txt", "elements.txt"]:
        if os.path.exists(tmp):
            os.remove(tmp)


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（40个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: CVV 数据结构基本操作正确性 ----
cvv = CVV([3, 5, 4], dtype=float)
for i, n in enumerate([3, 5, 4]):
    cvv.set_row(i, np.arange(n, dtype=float))
assert cvv.size() == (3, 12), '[TC01] CVV 尺寸不正确 FAILED'
assert np.allclose(cvv.get_row(1), np.array([0., 1., 2., 3., 4.])), '[TC01] CVV 行数据不正确 FAILED'

# ---- TC02: CVV 索引访问正确性 ----
cvv2 = CVV([2, 3], dtype=int)
cvv2.set_row(0, np.array([10, 20]))
cvv2.set_row(1, np.array([30, 40, 50]))
assert cvv2.iget(0, 1) == 20, '[TC02] CVV iget 不正确 FAILED'
assert cvv2.iget(1, 2) == 50, '[TC02] CVV iget 不正确 FAILED'

# ---- TC03: MicrocavityGeometry 弧长解析验证 ----
geom = MicrocavityGeometry(R_major=10.0e-6, r_minor=1.5e-6)
arc_len = geom.arc_length_trapezoidal(0.0, 2 * np.pi, n=2000)
theory = 2.0 * np.pi * geom.r_minor
assert abs(arc_len - theory) / theory < 1e-4, '[TC03] 弧长相对误差过大 FAILED'

# ---- TC04: 网格生成尺寸正确性 ----
nodes, elements, markers = geom.generate_cross_section_mesh(nr=5, n_theta=8)
assert nodes.shape == (40, 2), '[TC04] 节点数不正确 FAILED'
assert elements.shape[1] == 3, '[TC04] 单元不是三角形 FAILED'

# ---- TC05: SphereQuadrature 球面面积 = 4π ----
sq = SphereQuadrature()
A = sq.sphere01_area()
assert abs(A - 4.0 * np.pi) < 1e-12, '[TC05] 球面积与 4π 不匹配 FAILED'

# ---- TC06: SphereQuadrature 奇次单项式积分为 0（对称性） ----
assert abs(sq.monomial_integral_exact((1, 0, 0))) < 1e-14, '[TC06] 奇次单项式积分应为 0 FAILED'

# ---- TC07: SphereQuadrature (0,0,0) 积分 = 4π ----
I_000 = sq.monomial_integral_exact((0, 0, 0))
assert abs(I_000 - 4.0 * np.pi) < 1e-12, '[TC07] (0,0,0) 积分应为 4π FAILED'

# ---- TC08: SphereQuadrature (2,0,0) 精确值验证 ----
I_200 = sq.monomial_integral_exact((2, 0, 0))
assert abs(I_200 - 4.0 * np.pi / 3.0) < 1e-12, '[TC08] (2,0,0) 积分应为 4π/3 FAILED'

# ---- TC09: SphereQuadrature 均匀采样点落在单位球面上 ----
np.random.seed(42)
pts = sq.uniform_sample(100, seed=99)
norms = np.sqrt(np.sum(pts**2, axis=1))
assert np.all(np.abs(norms - 1.0) < 1e-14), '[TC09] 采样点不在单位球面上 FAILED'

# ---- TC10: Vandermonde2D 常数积分 = 矩形面积 ----
vq = Vandermonde2DQuadrature()
t = 2
n_needed = (t + 1) * (t + 2) // 2
xs = np.linspace(0.1, 0.9, n_needed)
ys = np.linspace(0.1, 0.9, n_needed)
w = vq.compute_weights(xs, ys, t, rect_a=0.0, rect_b=1.0, rect_c=0.0, rect_d=1.0)
val = vq.integrate(np.ones(n_needed), w)
assert abs(val - 1.0) < 1e-10, '[TC10] Vandermonde2D 对 1 积分应为 1.0 FAILED'

# ---- TC11: ErrorNorms L¹ 范数——已知数组 ----
err = ErrorNorms()
arr = np.array([1.0, -2.0, 3.0])
l1 = err.l1_norm_discrete(arr)
assert abs(l1 - 6.0) < 1e-12, '[TC11] L1 范数不正确 FAILED'

# ---- TC12: ErrorNorms L² 范数——已知数组 ----
l2 = err.l2_norm_discrete(arr)
assert abs(l2 - np.sqrt(14.0)) < 1e-12, '[TC12] L2 范数不正确 FAILED'

# ---- TC13: ErrorNorms L∞ 范数 ----
linf = err.linf_norm(arr)
assert abs(linf - 3.0) < 1e-12, '[TC13] L∞ 范数不正确 FAILED'

# ---- TC14: ErrorNorms 离散 L¹ 范数（带体积元） ----
vals = np.array([2.0, 3.0])
vols = np.array([0.5, 0.5])
l1_vol = err.l1_norm_discrete(vals, vols)
assert abs(l1_vol - 2.5) < 1e-12, '[TC14] 带体积元 L1 范数不正确 FAILED'

# ---- TC15: ErrorNorms 收敛阶估计——已知二阶 ----
e = np.array([0.01, 0.0025, 0.000625])
h = np.array([0.1, 0.05, 0.025])
orders = err.convergence_order(e, h)
assert np.all(np.abs(orders - 2.0) < 0.1), '[TC15] 收敛阶估计不正确 FAILED'

# ---- TC16: ErrorNorms Richardson 外推——二阶修正 ----
uh = np.array([1.0 + 0.04])
uh2 = np.array([1.0 + 0.01])
extrap = err.richardson_extrapolation(uh, uh2, p=2.0)
assert abs(extrap[0] - 1.0) < 1e-12, '[TC16] Richardson 外推不正确 FAILED'

# ---- TC17: BisectionSolver 求 x²-2=0 的根 ----
bisect = BisectionSolver(max_iter=200, tol=1e-12)
root, iters = bisect.solve(lambda x: x**2 - 2, 1.0, 2.0)
assert abs(root - np.sqrt(2.0)) < 1e-10, '[TC17] 二分法求 √2 不准确 FAILED'
assert iters > 0, '[TC17] 二分迭代次数应为正 FAILED'

# ---- TC18: BisectionSolver 同号区间触发 ValueError ----
try:
    bisect.solve(lambda x: x**2 + 1, 1.0, 2.0)
    assert False, '[TC18] 应触发 ValueError FAILED'
except ValueError:
    pass

# ---- TC19: CollatzPolynomial 次数正确 ----
poly = CollatzPolynomial(np.array([1, 0, 1, 1]))
assert poly.degree() == 3, '[TC19] 多项式次数不正确 FAILED'

# ---- TC20: CollatzPolynomial 序列在最大步数内终止 ----
seq = poly.sequence(max_steps=30)
assert len(seq) <= 30, '[TC20] 序列未在最大步数内终止 FAILED'

# ---- TC21: CollatzPolynomial smooth_analog |x|<1 收敛到 0 ----
fp = CollatzPolynomial.smooth_analog(0.5, max_iter=50)
assert abs(fp) < 1e-10, '[TC21] 平滑映射小初值应收敛到 0 FAILED'

# ---- TC22: ResonanceEigenSolver 灵敏度 dλ/dn 公式 ----
solver = ResonanceEigenSolver(R_major=10.0e-6, n_nominal=3.47)
S = solver.sensitivity_dlambda_dn(m=140, lambda_nm=1550.0, n_eff=3.47)
assert abs(S - 1550.0 / 3.47) < 1e-10, '[TC22] 灵敏度公式不正确 FAILED'

# ---- TC23: 谐振条件残差符号验证 ----
res = solver.resonance_condition(m=140, lambda_nm=1500.0, n_eff=3.47)
assert isinstance(res, float), '[TC23] 谐振条件残差应为浮点数 FAILED'
assert np.isfinite(res), '[TC23] 谐振条件残差应有限 FAILED'

# ---- TC24: 幂迭代最大本征值——对角占优矩阵 ----
np.random.seed(42)
n_test = 20
A_test = np.diag(np.linspace(1.0, 10.0, n_test))
lam, vec, it_ev = solver.power_iteration_eigenvalue(A_test, max_iter=500, tol=1e-10)
assert abs(lam - 10.0) < 1e-6, '[TC24] 幂迭代最大本征值不正确 FAILED'

# ---- TC25: 随机变量生成器固定种子可复现 ----
np.random.seed(42)
rng1 = RandomVariateGenerator(seed=42)
s1 = rng1.normal(0.0, 1.0, size=100)
rng2 = RandomVariateGenerator(seed=42)
s2 = rng2.normal(0.0, 1.0, size=100)
assert np.allclose(s1, s2), '[TC25] 种子 42 下随机数不可复现 FAILED'

# ---- TC26: HypersphereSampler 采样点在超球面上 ----
np.random.seed(42)
hyper = HypersphereSampler(dim=5, seed=42)
pts = hyper.sample(50)
norms = np.linalg.norm(pts, axis=1)
assert np.all(np.abs(norms - 1.0) < 1e-14), '[TC26] 超球面采样点不在球面上 FAILED'

# ---- TC27: HypersphereSampler 角度统计数值范围 ----
np.random.seed(42)
stats = hyper.angle_statistics(n_pairs=200)
assert 0.0 <= stats['mean_abs_cos'] <= 1.0, '[TC27] mean_abs_cos 超出 [0,1] FAILED'
assert stats['mean_abs_cos'] >= 0.0, '[TC27] mean_abs_cos 应为非负 FAILED'

# ---- TC28: BandedMatrixSolver 对角求解残差 ----
bs = BandedMatrixSolver(n=5, ml=0, mu=0)
for i in range(5):
    bs.set_element(i, i, float(i + 1) * 2.0)
bs.factorize_np()
b = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
x = bs.solve(b)
Ax = bs.matvec(x)
assert np.linalg.norm(Ax - b) < 1e-14, '[TC28] 带状对角求解残差过大 FAILED'
assert np.allclose(x, np.ones(5)), '[TC28] 对角求解预期 x=[1,1,1,1,1] FAILED'

# ---- TC29: LaplacianStencils 5点格式解析验证 ----
nx_l = ny_l = 21
L_l = 2.0
x_arr = np.linspace(0, L_l, nx_l)
y_arr = np.linspace(0, L_l, ny_l)
X, Y = np.meshgrid(x_arr, y_arr)
u = np.sin(np.pi * X / L_l) * np.sin(np.pi * Y / L_l)
h_l = L_l / (nx_l - 1)
Lu = LaplacianStencils.laplacian5_2d(u, h_l)
Lu_theory = -2.0 * (np.pi / L_l)**2 * u
interior = Lu[1:-1, 1:-1] - Lu_theory[1:-1, 1:-1]
assert np.sqrt(np.mean(interior**2)) < 0.05, '[TC29] 5点 Laplacian 误差过大 FAILED'

# ---- TC30: LaplacianStencils 9点 torus 对常数为零 ----
u_const = np.ones((10, 10))
Lu9 = LaplacianStencils.laplacian9_torus(u_const, h=0.1)
assert np.max(np.abs(Lu9)) < 1e-14, '[TC30] 9点 torus Laplacian 对常数应输出 0 FAILED'

# ---- TC31: SensorResponseSurrogate 训练后预测输出数值范围 ----
surr = SensorResponseSurrogate()
surr.train(n_points=100)
pred = surr.predict(0.0, 1.00)
assert np.isfinite(pred), '[TC31] 代理模型预测应有限 FAILED'
assert isinstance(pred, float), '[TC31] 预测应为浮点数 FAILED'

# ---- TC32: SensorResponseSurrogate 可复现性（相同种子） ----
surr2 = SensorResponseSurrogate()
surr2.train(n_points=100)
pred2 = surr2.predict(0.0, 1.00)
# 同种子、同训练数据，预测应相同
assert abs(pred - pred2) < 1e-10, '[TC32] 相同种子的训练结果应一致 FAILED'

# ---- TC33: ErrorNorms compute_quality_metrics 动态范围正值 ----
field = np.linspace(0.1, 10.0, 100)
metrics = err.compute_quality_metrics(field)
assert metrics['max'] > metrics['min'], '[TC33] 质量指标 max/min 顺序错误 FAILED'
assert metrics['dynamic_range_db'] > 0, '[TC33] 动态范围应为正值 FAILED'

# ---- TC34: ErrorNorms relative_l2_error 零残差返回 0 ----
A_ref = np.array([1.0, 2.0, 3.0])
rel = err.relative_l2_error(A_ref, A_ref)
assert rel < 1e-14, '[TC34] 零残差相对 L2 误差应为 0 FAILED'

# ---- TC35: LaplacianStencils 3点非均匀1D——已知解析解 ----
x_1d = np.array([0.0, 0.5, 1.2, 2.0])
u_1d = x_1d ** 2
Lu3 = LaplacianStencils.laplacian3_uneven_1d(u_1d, x_1d)
# u'' = 2，内部点应近似
assert abs(Lu3[1] - 2.0) < 1e-10, '[TC35] 3点非均匀 Laplacian 不正确 FAILED'
assert abs(Lu3[2] - 2.0) < 1e-10, '[TC35] 3点非均匀 Laplacian 不正确 FAILED'

# ---- TC36: GaussLegendreTensor 对 f(x,y)=1 积分为面积 ----
gt = GaussLegendreTensor()
f_one = lambda x, y: 1.0
approx_one = gt.tensor_quad_2d(f_one, 0.0, 1.0, 0.0, 1.0, m=4)
assert abs(approx_one - 1.0) < 1e-14, '[TC36] GaussLegendre 对 1 积分应为 1.0 FAILED'

# ---- TC37: MonteCarloUQ 参数扰动键完整性 ----
np.random.seed(42)
mc = MonteCarloUQ(n_samples=5, seed=42)
base = {"R_major": 10.0e-6, "r_minor": 1.5e-6, "n_ring": 3.47}
std = {"R_major": 0.05e-6, "r_minor": 0.02e-6, "n_ring": 0.01}
p = mc.parameter_perturbation(base, std)
assert set(p.keys()) == set(base.keys()), '[TC37] 扰动参数字典键不正确 FAILED'
assert p["n_ring"] > 0, '[TC37] 扰动后折射率应 > 0 FAILED'

# ---- TC38: MonteCarloUQ MC 传播输出结构正确 ----
np.random.seed(42)
mc2 = MonteCarloUQ(n_samples=30, seed=42)
base2 = {"R_major": 10.0e-6, "r_minor": 1.5e-6, "n_ring": 3.47}
std2 = {"R_major": 0.05e-6, "r_minor": 0.02e-6, "n_ring": 0.01}
fwd = lambda p: {"lambda_res_nm": 2.0 * np.pi * p["R_major"] * p["n_ring"] / 120 * 1e9}
summary = mc2.run_mc_propagation(base2, std2, fwd)
assert summary['n_success'] > 0, '[TC38] MC 应有成功样本 FAILED'
assert 'lambda_res_nm' in summary, '[TC38] 输出应包含 lambda_res_nm FAILED'
assert summary['lambda_res_nm']['std'] >= 0, '[TC38] std 应为非负 FAILED'

# ---- TC39: ErrorNorms residual_norm 一致性 ----
residual = np.array([0.1, -0.2, 0.3])
res_norm = err.residual_norm(residual)
l2_direct = np.sqrt(np.sum(residual**2))
assert abs(res_norm - l2_direct) < 1e-14, '[TC39] residual_norm 与直接计算不一致 FAILED'

# ---- TC40: BandedMatrixSolver matvec 对角矩阵乘法 ----
bs2 = BandedMatrixSolver(n=4, ml=0, mu=0)
for i in range(4):
    bs2.set_element(i, i, float(i + 1))
bs2.factorize_np()
x_vec = np.array([2.0, 3.0, 4.0, 5.0])
b_vec = bs2.matvec(x_vec)
assert abs(b_vec[0] - 2.0) < 1e-14, '[TC40] matvec row0 不正确 FAILED'
assert abs(b_vec[1] - 6.0) < 1e-14, '[TC40] matvec row1 不正确 FAILED'
assert abs(b_vec[2] - 12.0) < 1e-14, '[TC40] matvec row2 不正确 FAILED'
assert abs(b_vec[3] - 20.0) < 1e-14, '[TC40] matvec row3 不正确 FAILED'

print('\n全部 40 个测试通过!\n')
