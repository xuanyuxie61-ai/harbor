# -*- coding: utf-8 -*-

import numpy as np
import sys
import os


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


    arc_len = geom.arc_length_trapezoidal(0.0, 2 * np.pi, n=2000)
    theory = 2.0 * np.pi * geom.r_minor
    print(f"  微环截面周长 (梯形法, n=2000): {arc_len:.6e} m")
    print(f"  理论值 2πr:                      {theory:.6e} m")
    print(f"  弧长相对误差: {abs(arc_len - theory) / theory:.6e}")


    nodes, elements, markers = geom.generate_cross_section_mesh(nr=40, n_theta=60)
    print(f"  网格节点数: {nodes.shape[0]}, 三角单元数: {elements.shape[0]}")


    quality = geom.compute_mesh_quality(nodes, elements)
    print(f"  最小角: {quality['min_angle_deg']:.2f}°, 总面积: {quality['total_area']:.6e} m²")


    row_lengths = [3, 5, 4, 6, 2]
    cvv = CVV(row_lengths, dtype=float)
    for i in range(len(row_lengths)):
        cvv.set_row(i, np.arange(row_lengths[i], dtype=float))
    print(f"  CVV 测试: 行数={cvv.size()[0]}, 总元素={cvv.size()[1]}")
    print(f"  CVV 第2行: {cvv.get_row(2)}")


    geom.fem_write_nodes(nodes, "nodes.txt")
    geom.fem_write_elements(elements, "elements.txt")
    nodes_read = geom.fem_read_nodes("nodes.txt")
    elements_read = geom.fem_read_elements("elements.txt")
    assert np.allclose(nodes, nodes_read)
    assert np.array_equal(elements, elements_read)
    print("  FEM I/O 读写验证通过")


    wg_nodes = geom.generate_waveguide_nodes(n_points=30)
    print(f"  耦合波导节点数: {wg_nodes.shape[0]}")

    return geom, nodes, elements, markers


def step2_quadrature_validation():
    print("\n" + "=" * 60)
    print("步骤 2: 数值积分引擎验证")
    print("=" * 60)


    sq = SphereQuadrature()
    print(f"  单位球面面积: {sq.sphere01_area():.10f}")

    test_exponents = [(0, 0, 0), (2, 0, 0), (2, 2, 0), (4, 0, 0), (2, 2, 2)]
    for e in test_exponents:
        exact = sq.monomial_integral_exact(e)
        mc, err = sq.monte_carlo_integral(5000, e, seed=42)
        print(f"  单项式{e}: 精确值={exact:.6e}, MC估计={mc:.6e} ± {err:.6e}")


    vq = Vandermonde2DQuadrature()
    t = 2
    n_needed = (t + 1) * (t + 2) // 2

    xs = np.array([0.1, 0.5, 0.9, 0.2, 0.8, 0.5])
    ys = np.array([0.1, 0.2, 0.1, 0.8, 0.8, 0.5])
    xs, ys = xs[:n_needed], ys[:n_needed]
    w = vq.compute_weights(xs, ys, t, rect_a=0.0, rect_b=1.0, rect_c=0.0, rect_d=1.0)

    val_const = vq.integrate(np.ones_like(w), w)
    print(f"  Vandermonde2D 对常数1积分: {val_const:.6f} (期望 1.0)")

    val_xy = vq.integrate(xs * ys, w)
    print(f"  Vandermonde2D 对 xy 积分: {val_xy:.6f} (期望 0.25)")


    gt = GaussLegendreTensor()
    f_test = lambda x, y: np.sin(np.pi * x) * np.cos(np.pi * y)
    approx = gt.tensor_quad_2d(f_test, 0.0, 1.0, 0.0, 1.0, m=8)

    print(f"  Gauss-Legendre 张量积 (sinπx·cosπy): {approx:.6e} (期望 ~0)")


def step3_helmholtz_and_thermal():
    print("\n" + "=" * 60)
    print("步骤 3: Helmholtz 光场与热传导求解")
    print("=" * 60)


    L = 4.0e-6
    nx = ny = 41
    k0 = 2.0 * np.pi / (1.55e-6)


    R_center = 10.0e-6

    cx, cy = L / 2, L / 2
    r_ring = 1.5e-6
    X, Y = np.meshgrid(np.linspace(0, L, nx), np.linspace(0, L, ny))
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    n_profile = np.where(dist <= r_ring, 3.47, 1.44)


    helm = HelmholtzSolver(nx, ny, L, L, k0, n_profile, boundary_value=0.0)

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


    u_test = np.sin(2 * np.pi * X / L) * np.sin(2 * np.pi * Y / L)
    Lu5 = LaplacianStencils.laplacian5_2d(u_test, L / (nx - 1))

    Lu_theory = -2.0 * (2.0 * np.pi / L) ** 2 * u_test
    err_L2 = np.sqrt(np.mean((Lu5[1:-1, 1:-1] - Lu_theory[1:-1, 1:-1]) ** 2))
    print(f"  5点 Laplacian L² 误差: {err_L2:.4e}")


    thermal = ThermalSolver(nx, ny, L, L, kappa=1.4e2, h_conv=10.0, T_ambient=300.0)
    Q = thermal.compute_absorbed_heat(intensity, alpha_abs=1.0e-3)
    T = thermal.solve_steady_state(Q, bc_type="robin")
    delta_n = thermal.compute_thermal_lens(T, dn_dT=1.86e-4)
    print(f"  温度场求解完成: max ΔT={np.max(T - 300.0):.4e} K, max Δn={np.max(delta_n):.4e}")

    return helm, thermal, E, T, delta_n


def step4_photothermal_coupling(helm, thermal, E, T, delta_n):
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


    Q_c = coupler.compute_heat_source(E_c)
    P_trap = coupler.integrate_heat_source(Q_c, "trapezoidal")
    P_vand = coupler.integrate_vandermonde_2d(Q_c, total_degree=2)
    print(f"  总吸收功率 (梯形法): {P_trap:.6e} W/m")
    print(f"  总吸收功率 (Vandermonde): {P_vand:.6e} W/m")


    P_sphere = coupler.wgm_mode_sphere_integral(E_c, R_sphere=10.0e-6)
    print(f"  等效球面积分功率: {P_sphere:.6e}")

    return coupler, E_c, T_c, n_c


def step5_eigenvalue_and_resonance():
    print("\n" + "=" * 60)
    print("步骤 5: 本征值分析与谐振波长")
    print("=" * 60)

    solver = ResonanceEigenSolver(R_major=10.0e-6, n_nominal=3.47)



    def n_eff_func(lam_nm):
        return 3.47 - 1e-4 * (lam_nm - 1550.0) / 1550.0

    m = 140
    lambda_res, iters = solver.find_resonance_wavelength(m, n_eff_func, 1500.0, 1600.0)
    print(f"  模式 m={m} 的谐振波长: {lambda_res:.4f} nm (二分迭代 {iters} 次)")


    ng = 3.5
    fsr = solver.compute_mode_spacing(m, lambda_res, n_eff_func(lambda_res), ng)
    print(f"  自由光谱范围 FSR: {fsr * 1e9:.4f} nm")


    S = solver.sensitivity_dlambda_dn(m, lambda_res, n_eff_func(lambda_res))
    print(f"  灵敏度 dλ/dn: {S:.4f} nm/RIU")



    n_test = 50
    A_test = np.diag(np.linspace(1.0, 10.0, n_test)) + 0.1 * np.random.default_rng(42).random((n_test, n_test))
    lam, vec, it_ev = solver.power_iteration_eigenvalue(A_test, max_iter=500, tol=1e-10)
    print(f"  幂迭代最大本征值: {lam:.6f} (迭代 {it_ev} 次)")


    poly = CollatzPolynomial(np.array([1, 0, 1, 1]))
    seq = poly.sequence(max_steps=20)
    print(f"  Collatz 多项式序列长度: {len(seq)}, 终止次数: {seq[-1].sum() if len(seq[-1])>0 else 0}")


    fp = CollatzPolynomial.smooth_analog(0.5, max_iter=50)
    print(f"  Collatz 平滑映射不动点 (x₀=0.5): {fp:.6e}")

    return solver, lambda_res


def step6_neural_network_surrogate():
    print("\n" + "=" * 60)
    print("步骤 6: 神经网络代理模型")
    print("=" * 60)

    surrogate = SensorResponseSurrogate()
    surrogate.train(n_points=200)


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


    X_test, Y_test = surrogate.generate_training_data(100)
    metrics = surrogate.nn.evaluate_metrics(X_test, Y_test)
    print(f"  测试集指标: MSE={metrics['mse']:.4f}, RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")

    return surrogate


def step7_stochastic_uq():
    print("\n" + "=" * 60)
    print("步骤 7: 蒙特卡洛不确定性量化")
    print("=" * 60)


    hyper = HypersphereSampler(dim=10, seed=42)
    stats = hyper.angle_statistics(n_pairs=2000)
    print(f"  10维超球面角度统计:")
    print(f"    E[|cos θ|] = {stats['mean_abs_cos']:.6f} (理论 {stats['theoretical_mean_abs_cos']:.6f})")
    print(f"    std|cos θ| = {stats['std_abs_cos']:.6f}")


    rng = RandomVariateGenerator(seed=42)
    samples = {
        "normal": rng.normal(0.0, 1.0, size=1000),
        "gamma": rng.gamma(2.0, 1.0, size=1000),
        "beta": rng.beta(2.0, 5.0, size=1000),
        "exponential": rng.exponential(1.0, size=1000),
    }
    for name, arr in samples.items():
        print(f"  {name}: mean={np.mean(arr):.4f}, std={np.std(arr, ddof=1):.4f}")


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

        m = 120
        lambda0 = 1550.0
        n_g = 3.5

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
    print("\n" + "=" * 60)
    print("步骤 8: 误差分析与范数计算")
    print("=" * 60)

    err = ErrorNorms()


    volumes = np.full_like(T, helm.hx * helm.hy)
    l1_T = err.l1_norm_discrete(T.flatten(), volumes.flatten())
    l2_T = err.l2_norm_discrete(T.flatten(), volumes.flatten())
    linf_T = err.linf_norm(T)
    print(f"  温度场 L¹ 范数: {l1_T:.4e}")
    print(f"  温度场 L² 范数: {l2_T:.4e}")
    print(f"  温度场 L∞ 范数: {linf_T:.4e}")


    metrics_T = err.compute_quality_metrics(T)
    print(f"  温度场动态范围: {metrics_T['dynamic_range_db']:.2f} dB")




    T_approx = T + 1e-3 * np.random.default_rng(7).normal(size=T.shape)
    rel_l2 = err.relative_l2_error(T_approx, T, volumes)
    rel_l1 = err.relative_l1_error(T_approx, T, volumes)
    print(f"  扰动近似解相对 L² 误差: {rel_l2:.6e}")
    print(f"  扰动近似解相对 L¹ 误差: {rel_l1:.6e}")


    resolutions = np.array([0.1, 0.05, 0.025, 0.0125])
    errors = np.array([0.01, 0.0025, 0.000625, 0.000156])
    orders = err.convergence_order(errors, resolutions)
    print(f"  估计收敛阶: {orders}")


    extrap = err.richardson_extrapolation(errors[0] * np.ones(1), errors[1] * np.ones(1), p=2.0)
    print(f"  Richardson 外推值: {extrap[0]:.6e}")


def step9_bordered_banded_demo():
    print("\n" + "=" * 60)
    print("步骤 9: 边界带状矩阵 Schur 补求解演示")
    print("=" * 60)

    n1, n2 = 20, 4
    ml, mu = 2, 2

    A1_dense = np.diag(np.ones(n1) * 4.0) + np.diag(np.ones(n1 - 1) * -1.0, 1) + np.diag(np.ones(n1 - 1) * -1.0, -1)
    A1_dense += np.diag(np.ones(n1 - 2) * -0.5, 2) + np.diag(np.ones(n1 - 2) * -0.5, -2)
    A2 = np.random.default_rng(33).random((n1, n2))
    A3 = np.random.default_rng(44).random((n2, n1))
    A4 = np.eye(n2) * 5.0 + np.random.default_rng(55).random((n2, n2)) * 0.1


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


    for tmp in ["nodes.txt", "elements.txt"]:
        if os.path.exists(tmp):
            os.remove(tmp)


if __name__ == "__main__":
    main()
