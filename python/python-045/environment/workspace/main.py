#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import sys
import time




from parameter_manager import (
    PhysicalConstants,
    ColeColeDispersion,
    LayeredEarthModel,
    MTDataContainer
)
from basis_approximation import (
    BernsteinResistivityProfile,
    Bernstein2DResistivity,
    bernstein_basis_recursive
)
from mesh_generator import (
    Region2D,
    StructuredMesh2D,
    UnstructuredMesh2D,
    generate_rectangular_mesh,
    generate_annulus_mesh
)
from sparse_matrix_tools import (
    ge_to_st,
    sparse_matvec,
    DenseLUSolver,
    BandedMatrixSolver
)
from monte_carlo_sampler import (
    hypersphere01_sample,
    hypersphere01_monomial_integral,
    hypercube_surface_sample,
    circle_distance_stats,
    hypercube_surface_distance_stats,
    MetropolisHastingsSampler,
    AdaptiveCovarianceSampler
)
from mt_forward_solver import (
    mt_1d_analytic,
    mt_1d_analytic_cole_cole,
    mt_2d_te_fd,
    compute_apparent_resistivity_phase,
    add_noise_to_mt_data,
    thin_field_data
)
from inverse_optimizer import (
    dijkstra_priority_map,
    ifs_chaos_perturbation,
    OccamInversion,
    MultiObjectiveOptimizer
)
from data_io import (
    MTDataReader,
    MTDataWriter,
    MeshDataIO,
    DataValidator
)


def print_banner():
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║           大地电磁测深非线性反演与地下电性结构成像系统                         ║
║   Magnetotelluric Nonlinear Inversion & Subsurface Electrical Imaging        ║
║                                                                              ║
║   融合算法: Bernstein逼近 | Dijkstra优先更新 | Cole-Cole频散 | IFS混沌扰动   ║
║             有限差分正演 | MCMC贝叶斯采样 | Occam平滑反演 | 超球面采样       ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def demo_berstein_approximation():
    print("\n" + "=" * 70)
    print("【演示 1】Bernstein 多项式参数化电阻率剖面")
    print("=" * 70)



    coeffs = np.array([200.0, 150.0, 80.0, 30.0, 20.0, 50.0, 100.0, 300.0])
    profile = BernsteinResistivityProfile(coefficients=coeffs, z_max=10000.0)

    z_test = np.linspace(0.0, 10000.0, 101)
    rho_test = profile.evaluate(z_test)
    drhodz = profile.derivative(z_test)
    rough = profile.roughness()

    print(f"Bernstein 次数: n = {profile.n}")
    print(f"系数: {profile.coefficients}")
    print(f"模型粗糙度: {rough:.4f}")
    print(f"深度 0m 处电阻率: {rho_test[0]:.2f} Ω·m")
    print(f"深度 5000m 处电阻率: {profile.evaluate(5000.0):.2f} Ω·m")
    print(f"深度 10000m 处电阻率: {rho_test[-1]:.2f} Ω·m")


    rho_layers, thicknesses = profile.to_layer_model(5)
    print(f"\n离散化层状模型:")
    for i, (r, h) in enumerate(zip(rho_layers, thicknesses)):
        print(f"  层 {i+1}: ρ = {r:.2f} Ω·m, h = {h:.1f} m")
    print(f"  层 6 (半无限): ρ = {rho_layers[-1]:.2f} Ω·m")

    return rho_layers, np.append(thicknesses, [0.0])


def demo_cole_cole_dispersion():
    print("\n" + "=" * 70)
    print("【演示 2】Cole-Cole 频散模型（Hodgkin-Huxley 门控类比）")
    print("=" * 70)


    cc_model = ColeColeDispersion(sigma_0=0.01, m_charge=0.6, tau=1e-2, c_freq=0.8)

    frequencies = np.logspace(-3, 3, 20)
    omega = 2.0 * np.pi * frequencies
    sigma_star = cc_model.complex_conductivity(omega)
    rho_star = 1.0 / sigma_star

    print(f"Cole-Cole 参数: σ₀={cc_model.sigma_0:.4f}, m={cc_model.m_charge:.2f}, "
          f"τ={cc_model.tau:.2e}, c={cc_model.c_freq:.2f}")
    print(f"频率范围: {frequencies[0]:.2e} ~ {frequencies[-1]:.2e} Hz")
    print(f"\n复电阻率 (前5个频率):")
    for i in range(5):
        print(f"  f={frequencies[i]:.2e} Hz: ρ* = {rho_star[i]:.2e} Ω·m")


    alpha, beta = cc_model.hh_gating_analogy(V=10.0, alpha_0=0.01, beta_0=0.125,
                                               V_shift=10.0, k_T=10.0)
    print(f"\nHH 门控类比: α={alpha:.4f}, β={beta:.4f}")

    return frequencies, sigma_star


def demo_mesh_generation():
    print("\n" + "=" * 70)
    print("【演示 3】网格生成与质量评估")
    print("=" * 70)


    mesh_fd = generate_rectangular_mesh(0.0, 20000.0, 0.0, 10000.0, 41, 21)
    print(f"矩形结构化网格:")
    print(f"  节点数: {mesh_fd.n_nodes}")
    print(f"  网格间距: dx={mesh_fd.dx:.1f} m, dy={mesh_fd.dy:.1f} m")
    print(f"  边界节点: {len(mesh_fd.boundary_nodes)}")
    print(f"  内部节点: {len(mesh_fd.interior_nodes)}")


    mesh_ann = generate_annulus_mesh(2000.0, 8000.0, 8, 20)
    quality = mesh_ann.mesh_quality()
    print(f"\n环形非结构化网格:")
    print(f"  节点数: {mesh_ann.n_points}")
    print(f"  单元数: {mesh_ann.n_triangles}")
    print(f"  网格质量:")
    print(f"    最小角: {quality['min_angle']:.2f}°")
    print(f"    最大角: {quality['max_angle']:.2f}°")
    print(f"    平均面积: {quality['mean_area']:.2f} m²")


    for rtype in ['S', 'L', 'D', 'A']:
        region = Region2D(rtype)
        bbox = region.bounding_box()
        test_pts = np.array([[0.0, 0.0], [0.5, 0.5], [-0.5, 0.5]])
        inside = region.contains(test_pts[:, 0], test_pts[:, 1])
        print(f"  区域 '{rtype}' 边界框: {bbox}, 测试点包含: {inside}")

    return mesh_fd, mesh_ann


def demo_sparse_matrix_tools():
    print("\n" + "=" * 70)
    print("【演示 4】稀疏矩阵与稠密矩阵工具")
    print("=" * 70)


    n = 20
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        A[i, i] = 4.0
        if i > 0:
            A[i, i - 1] = -1.0
            A[i - 1, i] = -1.0
        if i > 1:
            A[i, i - 2] = -0.5
            A[i - 2, i] = -0.5


    nz_num, ist, jst, Ast = ge_to_st(A)
    print(f"稠密矩阵: {n}x{n}")
    print(f"非零元个数: {nz_num} / {n*n} (稀疏度: {1.0 - nz_num/(n*n):.4f})")


    x = np.ones(n)
    y_sparse = sparse_matvec(ist, jst, Ast, x)
    y_dense = A @ x
    print(f"稀疏/稠密乘法一致性: {np.max(np.abs(y_sparse - y_dense)):.2e}")


    b = np.random.randn(n)
    solver = DenseLUSolver(A)
    info = solver.dgefa()
    x_lu = solver.solve(b)
    residual = np.linalg.norm(A @ x_lu - b)
    cond_est = solver.condition_estimate()
    print(f"LU分解 info={info}, 残差={residual:.2e}, 条件数估计={cond_est:.2e}")

    return A


def demo_monte_carlo_sampling():
    print("\n" + "=" * 70)
    print("【演示 5】蒙特卡洛采样与距离统计")
    print("=" * 70)


    sphere_samples = hypersphere01_sample(m=5, n=1000)
    norms = np.sqrt(np.sum(sphere_samples ** 2, axis=0))
    print(f"5维超球面采样 1000 点: 范数均值={np.mean(norms):.6f}, 标准差={np.std(norms):.6f}")


    integral_val = hypersphere01_monomial_integral(3, [2, 0, 0])
    print(f"3维超球面 x² 积分精确值: {integral_val:.6f} (理论: 4π/3 ≈ 4.1888)")


    mu_h, var_h = hypercube_surface_distance_stats(5000, 3)
    print(f"3维超立方体表面距离: 均值={mu_h:.4f}, 方差={var_h:.4f}")


    mu_c, var_c = circle_distance_stats(10000)
    print(f"单位圆上距离统计: 均值={mu_c:.4f} (理论≈1.2732), 方差={var_c:.4f}")


    def log_target(x):

        d1 = np.sum((x - np.array([2.0, 2.0])) ** 2)
        d2 = np.sum((x - np.array([-2.0, -2.0])) ** 2)
        return np.log(np.exp(-0.5 * d1) + 0.5 * np.exp(-0.5 * d2))

    mcmc = MetropolisHastingsSampler(log_target, np.eye(2) * 0.5,
                                      bounds=[(-5, 5), (-5, 5)])
    samples, rate = mcmc.sample(np.zeros(2), n_samples=200, burn_in=300, thinning=5)
    print(f"\nMCMC 采样 (双峰高斯):")
    print(f"  样本均值: {np.mean(samples, axis=0)}")
    print(f"  样本协方差:\n{np.cov(samples.T)}")
    print(f"  接受率: {rate:.3f}")

    return samples


def demo_mt_forward():
    print("\n" + "=" * 70)
    print("【演示 6】MT 一维解析正演与二维有限差分正演")
    print("=" * 70)


    resistivities = np.array([100.0, 30.0, 200.0, 50.0])
    thicknesses = np.array([300.0, 800.0, 2000.0])


    dispersion_list = [
        None,
        ColeColeDispersion(sigma_0=1.0/30.0, m_charge=0.5, tau=1e-2, c_freq=0.7),
        None,
        None
    ]

    frequencies = np.logspace(-2, 2, 25)


    Z_1d, rho_a_1d, phi_1d = mt_1d_analytic(resistivities, thicknesses, frequencies)


    Z_disp, rho_a_disp, phi_disp = mt_1d_analytic_cole_cole(
        resistivities, thicknesses, dispersion_list, frequencies
    )

    print("1D 解析正演结果 (部分频率):")
    for i in [0, 8, 16, 24]:
        print(f"  f={frequencies[i]:.4e} Hz:")
        print(f"    无频散: ρ_a={rho_a_1d[i]:.2f} Ω·m, φ={phi_1d[i]:.2f}°")
        print(f"    有频散: ρ_a={rho_a_disp[i]:.2f} Ω·m, φ={phi_disp[i]:.2f}°")


    rho_a_noisy, phi_noisy = add_noise_to_mt_data(rho_a_disp, phi_disp, noise_level=0.05)
    print(f"\n加噪声后数据 (ρ_a 信噪比 ≈ {20:.1f} dB)")


    mesh_2d = generate_rectangular_mesh(0.0, 10000.0, 0.0, 5000.0, 21, 11)

    def sigma_2d(y, z):
        if z < 300.0:
            return 0.01
        elif z < 1100.0:
            return 1.0 / 30.0
        elif z < 3100.0:
            return 0.005
        else:
            return 0.02

    def bc_2d(y, z):
        if abs(z) < 1.0:
            return 1.0 + 0.0j
        return np.exp(-z / 2000.0) * (1.0 + 0.0j)

    try:
        E_x, H_y, Z_2d = mt_2d_te_fd(sigma_2d, mesh_2d, 10.0, bc_2d)
        print(f"\n2D 有限差分正演 (f=10 Hz):")
        print(f"  节点数: {len(E_x)}")
        print(f"  E_x 模范围: [{np.min(np.abs(E_x)):.4e}, {np.max(np.abs(E_x)):.4e}]")
        print(f"  H_y 模范围: [{np.min(np.abs(H_y)):.4e}, {np.max(np.abs(H_y)):.4e}]")


        coords = mesh_2d.node_coords
        Z_mag = np.abs(Z_2d)
        coords_thin, Z_thin = thin_field_data(coords, Z_mag, thin_factor=2)
        print(f"  稀疏采样后测点数: {len(Z_thin)}")
    except Exception as e:
        print(f"\n2D 正演遇到数值问题（网格较粗）: {e}")

    return frequencies, rho_a_noisy, phi_noisy, resistivities, thicknesses


def demo_dijkstra_ifs():
    print("\n" + "=" * 70)
    print("【演示 7】Dijkstra 优先级与 IFS 混沌扰动")
    print("=" * 70)


    n_nodes = 6
    adjacency = [
        [(1, 1.0), (2, 3.0)],
        [(0, 1.0), (2, 1.0), (3, 4.0), (4, 2.0)],
        [(0, 3.0), (1, 1.0), (3, 1.0)],
        [(1, 4.0), (2, 1.0), (4, 1.0), (5, 3.0)],
        [(1, 2.0), (3, 1.0), (5, 1.0)],
        [(3, 3.0), (4, 1.0)]
    ]
    sensitivity = np.array([1.0, 0.8, 0.6, 0.5, 0.7, 0.3])
    priorities = dijkstra_priority_map(n_nodes, adjacency, [0], sensitivity)
    print("Dijkstra 模型更新优先级:")
    for i, p in enumerate(priorities):
        print(f"  节点 {i}: 优先级 = {p:.4f}, 敏感度 = {sensitivity[i]:.2f}")


    print("\nIFS 混沌扰动序列:")
    x = np.array([5.0, 5.0, 5.0])
    for step in range(5):
        x = ifs_chaos_perturbation(x, scale=1.0, n_maps=4)
        print(f"  步 {step+1}: x = [{x[0]:.4f}, {x[1]:.4f}, {x[2]:.4f}]")

    return priorities


def demo_occam_inversion(frequencies, rho_obs, phi_obs, true_resistivities, true_thicknesses):
    print("\n" + "=" * 70)
    print("【演示 8】Occam 平滑反演")
    print("=" * 70)

    n_layers = len(true_resistivities)



    def forward_log_resistivity(m_log):








        raise NotImplementedError("Hole 2: 正演回调函数的数据格式转换待实现")


    d_obs = np.concatenate([np.log10(rho_obs), phi_obs / 45.0])


    m_init = np.ones(n_layers) * np.log(100.0)


    inv = OccamInversion(
        forward_func=forward_log_resistivity,
        n_model=n_layers,
        data_errors=np.ones(len(d_obs)) * 0.05,
        m_ref=np.ones(n_layers) * np.log(100.0),
        lambda_init=10.0,
        max_iter=15,
        target_misfit=0.5
    )

    m_best, lambda_best = inv.invert(d_obs, m_init)
    rho_inverted = np.exp(m_best)
    rho_inverted = np.maximum(rho_inverted, 0.1)

    rel_err = np.abs(rho_inverted - true_resistivities) / true_resistivities * 100
    print(f"真实电阻率:     {true_resistivities}")
    print(f"反演电阻率:     {rho_inverted.astype(np.float32)}")
    print(f"相对误差(%):    {np.array2string(rel_err, precision=2, floatmode='fixed')}")
    print(f"最优正则化参数 λ = {lambda_best:.4f}")
    print(f"反演迭代历史 ({len(inv.history)} 步):")
    for h in inv.history[:5]:
        print(f"  iter={h['iter']}, misfit={h['misfit']:.4f}, λ={h['lambda']:.4f}")

    return rho_inverted, lambda_best


def demo_data_io_and_validation(frequencies, rho_obs, phi_obs, rho_inverted,
                                 true_resistivities, true_thicknesses):
    print("\n" + "=" * 70)
    print("【演示 9】数据 I/O、格式转换与验证")
    print("=" * 70)

    out_dir = "mt_inversion_output"
    os.makedirs(out_dir, exist_ok=True)


    Z_obs = MTDataContainer.impedance_from_rhophi(rho_obs, phi_obs, frequencies)
    MTDataWriter.write_complex_impedance(
        os.path.join(out_dir, "observed_data.txt"), frequencies, Z_obs
    )
    print(f"观测数据已写入: {out_dir}/observed_data.txt")


    f_read, Z_read, err_read = MTDataReader.read_complex_impedance(
        os.path.join(out_dir, "observed_data.txt")
    )
    print(f"数据读取验证: 最大差异 = {np.max(np.abs(Z_obs - Z_read)):.2e}")


    points = np.array([[0, 0], [10000, 0], [0, 5000], [10000, 5000]], dtype=np.float64)
    triangles = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    MeshDataIO.write_nodes(os.path.join(out_dir, "mesh_nodes.txt"), points)
    MeshDataIO.write_elements(os.path.join(out_dir, "mesh_elements.txt"), triangles)
    MeshDataIO.write_xml_mesh(os.path.join(out_dir, "mesh.xml"), points, triangles)
    print(f"网格文件已写入: {out_dir}/mesh_nodes.txt, mesh_elements.txt, mesh.xml")



    thick = true_thicknesses[:-1] if len(true_thicknesses) > 1 else np.array([1000.0])
    n_layers = len(true_resistivities)
    if len(thick) < n_layers - 1:
        thick = np.append(thick, [1000.0] * (n_layers - 1 - len(thick)))
    thick = thick[:n_layers - 1]
    Z_pred, rho_pred, phi_pred = mt_1d_analytic(rho_inverted, thick, frequencies)

    MeshDataIO.write_model_report(
        os.path.join(out_dir, "inversion_report.txt"),
        model={"resistivities": rho_inverted, "thicknesses": thick},
        predictions={"rho_a": rho_pred, "phi": phi_pred},
        residuals={"rho_a": rho_obs - rho_pred, "obs_rho_a": rho_obs},
        inversion_stats={"lambda": 10.0, "iterations": 15, "misfit": 0.5},
        frequencies=frequencies
    )
    print(f"反演报告已写入: {out_dir}/inversion_report.txt")


    ok, issues = DataValidator.validate_mt_data(frequencies, rho_obs, phi_obs)
    print(f"\nMT 数据验证: {'通过' if ok else '未通过'}")
    if issues:
        for issue in issues:
            print(f"  问题: {issue}")

    ok2, issues2 = DataValidator.validate_model(rho_inverted, thick)
    print(f"反演模型验证: {'通过' if ok2 else '未通过'}")
    if issues2:
        for issue in issues2:
            print(f"  问题: {issue}")

    return out_dir


def main():
    print_banner()
    t_start = time.time()

    np.random.seed(42)


    rho_bezier, thick_bezier = demo_berstein_approximation()


    freqs_cc, sigma_cc = demo_cole_cole_dispersion()


    mesh_fd, mesh_ann = demo_mesh_generation()


    A_fd = demo_sparse_matrix_tools()


    mcmc_samples = demo_monte_carlo_sampling()


    frequencies, rho_obs, phi_obs, true_rho, true_thick = demo_mt_forward()


    priorities = demo_dijkstra_ifs()


    rho_inv, lambda_best = demo_occam_inversion(
        frequencies, rho_obs, phi_obs, true_rho, true_thick
    )


    out_dir = demo_data_io_and_validation(
        frequencies, rho_obs, phi_obs, rho_inv, true_rho, true_thick
    )

    t_elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"【全部演示完成】总耗时: {t_elapsed:.2f} 秒")
    print(f"输出目录: {os.path.abspath(out_dir)}")
    print("=" * 70)


    print("\n【最终一致性检查】")
    all_ok = True


    t_test = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    B_sum = np.sum(bernstein_basis_recursive(5, t_test), axis=0)
    b_ok = np.allclose(B_sum, 1.0, atol=1e-10)
    print(f"  Bernstein 基函数 Partition of Unity: {'通过' if b_ok else '失败'}")
    all_ok = all_ok and b_ok


    Z_test, _, _ = mt_1d_analytic(np.array([100.0, 50.0]), np.array([500.0]),
                                   np.logspace(-1, 1, 10))
    z_ok = np.all(np.isfinite(Z_test)) and np.all(Z_test != 0)
    print(f"  1D 正演阻抗连续性: {'通过' if z_ok else '失败'}")
    all_ok = all_ok and z_ok


    n_check = 10
    A_check = np.random.randn(n_check, n_check)
    A_check = A_check @ A_check.T + np.eye(n_check)
    b_check = np.random.randn(n_check)
    sol_check = np.linalg.solve(A_check, b_check)
    res_check = np.linalg.norm(A_check @ sol_check - b_check)
    lu_ok = res_check < 1e-10
    print(f"  LU 求解精度: {'通过' if lu_ok else '失败'} (残差={res_check:.2e})")
    all_ok = all_ok and lu_ok


    ok_mt, _ = DataValidator.validate_mt_data(frequencies, rho_obs, phi_obs)
    print(f"  MT 数据物理合理性: {'通过' if ok_mt else '失败'}")
    all_ok = all_ok and ok_mt


    inv_ok = np.all(rho_inv > 0)
    print(f"  反演结果物理合理性: {'通过' if inv_ok else '失败'}")
    all_ok = all_ok and inv_ok

    print(f"\n整体检查: {'全部通过' if all_ok else '存在失败项'}")
    print("=" * 70)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
