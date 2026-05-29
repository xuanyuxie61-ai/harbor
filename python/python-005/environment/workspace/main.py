#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py
CMB 各向异性综合分析系统 — 统一入口

本程序执行以下博士级科学计算流程：
1. 初始化 ΛCDM 宇宙学参数
2. 求解线性化爱因斯坦-玻尔兹曼方程（多 k 模式）
3. 计算转移函数 T_l(k) 并进行 Chebyshev 谱插值
4. 使用快速高精度求积（Gauss-Legendre / Clenshaw-Curtis / Fejér）
   计算角功率谱 C_l^{TT}
5. 二分法定位声学峰并计算峰间距比
6. 生成球面层级三角网格（HEALPix-like）
7. 计算巡天掩膜几何矩与波束窗函数
8. 模拟双C形前景污染并进行边缘检测
9. 积分卫星刚体姿态动力学与扫描策略
10. 最小二乘拟合宇宙学参数并计算 Fisher 矩阵

运行方式：
    python main.py
（零参数，直接运行）
"""

import numpy as np
import time
import sys

# 导入所有科学计算模块
from utils import (
    spherical_bessel_j_array, gamma_lanczos, binomial,
    robust_divide, clip_to_unit, ensure_positive,
)
from boltzmann_solver import CosmologyParams, BoltzmannSolver
from transfer_function import TransferFunctionComputer, ChebyshevInterpolator
from los_integration import (
    FastQuadrature, los_integral_power_spectrum, compute_sachs_wolfe_integral,
)
from power_spectrum import (
    primordial_power_spectrum, compute_Cl_spectrum,
    find_acoustic_peaks, compute_peak_spacing_ratio,
)
from spherical_mesh import SphericalMesh
from mask_beam import (
    SurveyMask, polygon_moment, polygon_area, polygon_centroid,
    disk_monomial_integral, gaussian_beam_window, beam_convolved_Cl,
    point_in_polygon,
)
from foreground_edges import (
    generate_double_c_foreground, foreground_temperature_profile,
    detect_edges_1d, shepp_logan_2d, gradient_edge_detector_2d,
    compute_residual_rms,
)
from satellite_dynamics import (
    GyroscopeDynamics, generate_scanning_trajectory,
    compute_hit_map, compute_coverage_uniformity,
)
from parameter_fit import (
    solve_least_squares_qr, run_lls_test_suite,
    CosmologyFitter, compute_fisher_matrix, theory_Cl_model,
)


def print_section(title: str):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def main():
    t_start = time.time()
    np.random.seed(42)

    # ===================================================================
    # 1. 宇宙学参数与玻尔兹曼方程求解
    # ===================================================================
    print_section("1. 线性化爱因斯坦-玻尔兹曼方程求解")
    params = CosmologyParams()
    print(f"  Ω_b h^2 = {params.Omega_b:.5f}")
    print(f"  Ω_c h^2 = {params.Omega_c:.5f}")
    print(f"  h       = {params.h:.4f}")

    k_modes = np.logspace(-3, -1, 5)  # 5 个 k 模式
    print(f"  求解 {len(k_modes)} 个 k 模式的微扰演化 ...")
    boltzmann_results = []
    for k in k_modes:
        solver = BoltzmannSolver(params, k_mode=k, n_eta=800, eta_max=8000.0)
        eta, D0, D1, D2, vb = solver.solve()
        T_k = solver.transfer_function_today()
        boltzmann_results.append({
            "k": k, "eta": eta, "Delta0": D0, "Delta1": D1,
            "Delta2": D2, "vb": vb, "T_k": T_k,
        })
        print(f"    k={k:.4f} Mpc^-1  →  T(k)={T_k:.6f}")

    # ===================================================================
    # 2. 转移函数与 Chebyshev 谱插值
    # ===================================================================
    print_section("2. 转移函数 Chebyshev 谱插值")
    lmax_demo = 40
    tf_comp = TransferFunctionComputer(lmax=lmax_demo, k_min=1e-3, k_max=0.1, n_cheb=16)
    tf_comp.precompute_all()
    print(f"  预计算 l=2...{lmax_demo} 的 Chebyshev 插值器（每个 l {tf_comp.n_cheb} 节点）")
    # 演示插值精度
    k_test = 0.05
    for l_test in [10, 20, 30]:
        T_interp = tf_comp.get_transfer(l_test, k_test)
        print(f"    T_{l_test}({k_test}) ≈ {T_interp:.6f}")

    # ===================================================================
    # 3. 角功率谱 C_l 与快速求积
    # ===================================================================
    print_section("3. 角功率谱 C_l 计算（四种快速求积规则）")
    l_values = np.arange(2, lmax_demo + 1)

    def transfer_wrapper(l: int, k: float) -> float:
        return tf_comp.get_transfer(l, k)

    # 使用 Gauss-Legendre 计算 C_l
    Cl_GL = np.zeros(len(l_values))
    for idx, l in enumerate(l_values):
        def Tl(k):
            return transfer_wrapper(l, k)
        Cl_GL[idx] = los_integral_power_spectrum(
            l, Tl, primordial_power_spectrum,
            k_min=1e-3, k_max=0.1, n_quad=32, rule="gauss_legendre"
        )
    print(f"  Gauss-Legendre:  C_2 = {Cl_GL[0]:.6e},  C_{lmax_demo} = {Cl_GL[-1]:.6e}")

    # 对比 Clenshaw-Curtis
    Cl_CC = np.zeros(len(l_values))
    for idx, l in enumerate(l_values):
        def Tl2(k):
            return transfer_wrapper(l, k)
        Cl_CC[idx] = los_integral_power_spectrum(
            l, Tl2, primordial_power_spectrum,
            k_min=1e-3, k_max=0.1, n_quad=32, rule="clenshaw_curtis"
        )
    diff_CC = np.max(np.abs(Cl_GL - Cl_CC))
    print(f"  Clenshaw-Curtis: max|ΔC_l| = {diff_CC:.6e}")

    # ===================================================================
    # 4. 声学峰定位（二分法）
    # ===================================================================
    print_section("4. 声学峰定位与峰间距比")
    peaks = find_acoustic_peaks(l_values, Cl_GL, n_peaks=3)
    print("  检测到的声学峰：")
    for i, (lp, Clp) in enumerate(peaks, 1):
        print(f"    第 {i} 峰: l = {lp},  C_l = {Clp:.6e}")
    R_ratio = compute_peak_spacing_ratio(peaks)
    print(f"  平均峰间距比 R = {R_ratio:.4f}（ΛCDM 理论预期 ~1.55）")

    # ===================================================================
    # 5. 球面三角网格（二十面体细分）
    # ===================================================================
    print_section("5. 球面层级三角网格")
    mesh = SphericalMesh(nsides=2)
    print(f"  顶点数: {mesh.n_vertices}")
    print(f"  面数:   {mesh.n_faces}")
    total_A = mesh.total_area()
    print(f"  总面积: {total_A:.6f} sr  (理论 4π = {4*np.pi:.6f})")
    print(f"  面积相对误差: {abs(total_A - 4*np.pi)/(4*np.pi)*100:.4f}%")
    # 单个面面积统计
    areas = [mesh.face_area(i) for i in range(mesh.n_faces)]
    print(f"  面面积均值: {np.mean(areas):.6e},  标准差: {np.std(areas):.6e}")
    mesh.write_mesh("spherical_mesh")
    print("  网格已输出到 spherical_mesh_*.txt")

    # ===================================================================
    # 6. 掩膜几何与波束分析
    # ===================================================================
    print_section("6. 巡天掩膜几何矩与波束窗函数")
    # 构造一个八边形近似掩膜
    angles = np.linspace(0, 2 * np.pi, 9)[:-1]
    mask_x = 2.0 * np.cos(angles)
    mask_y = 2.0 * np.sin(angles)
    mask = SurveyMask(mask_x, mask_y)
    print(f"  掩膜面积: {mask.area():.4f}")
    cx, cy = mask.centroid()
    print(f"  掩膜质心: ({cx:.4f}, {cy:.4f})")
    e_ellip = mask.ellipticity()
    print(f"  掩膜椭圆率: {e_ellip:.4f}")
    # 几何矩
    mu20 = polygon_moment(len(mask_x), mask_x, mask_y, 2, 0)
    mu02 = polygon_moment(len(mask_x), mask_x, mask_y, 0, 2)
    print(f"  二阶矩 μ20={mu20:.4f}, μ02={mu02:.4f}")
    # 波束窗函数
    fwhm = 7.0  # arcmin
    for l_beam in [100, 500, 1000]:
        Bl = gaussian_beam_window(l_beam, fwhm)
        print(f"  B_{l_beam} (FWHM={fwhm}') = {Bl:.6e}")
    # 圆盘矩积分（波束近似）
    r_beam = np.radians(fwhm / 60.0) / 2.0
    I00 = disk_monomial_integral(r_beam, 0, 0)
    print(f"  等效圆盘面积 (r={r_beam:.4f} rad): {I00:.6e}")

    # ===================================================================
    # 7. 前景污染模拟与边缘检测
    # ===================================================================
    print_section("7. 前景污染模拟与边缘检测")
    x_fg, y_fg, labels_fg = generate_double_c_foreground(n1=200, n2=200)
    print(f"  生成 {len(x_fg)} 个双C形前景样本点")
    print(f"  成分0占比: {np.mean(labels_fg == 0)*100:.1f}%, 成分1占比: {np.mean(labels_fg == 1)*100:.1f}%")
    # 一维温度剖面边缘检测
    theta_1d = np.linspace(-1.0, 1.0, 200)
    T_fg = foreground_temperature_profile(theta_1d, amplitude=100.0, width=0.3)
    edges = detect_edges_1d(T_fg, theta_1d, window=5, threshold=0.8)
    print(f"  一维温度剖面检测到 {len(edges)} 个边缘点")
    # 2D Shepp-Logan 幻影边缘
    nx, ny = 64, 64
    x2d = np.linspace(-1.0, 1.0, nx)
    y2d = np.linspace(-1.0, 1.0, ny)
    X2d, Y2d = np.meshgrid(x2d, y2d)
    phantom = shepp_logan_2d(X2d, Y2d)
    edge_mask = gradient_edge_detector_2d(phantom, threshold=15.0)
    n_edge_pixels = np.sum(edge_mask)
    print(f"  2D Shepp-Logan 幻影边缘像素数: {n_edge_pixels} / {nx*ny}")
    # 残留评估
    cmb_dummy = np.random.randn(ny, nx) * 10.0
    rms_res = compute_residual_rms(cmb_dummy, phantom, np.ones((ny, nx), dtype=bool))
    print(f"  前景残留 RMS: {rms_res:.2f} μK")

    # ===================================================================
    # 8. 卫星扫描动力学
    # ===================================================================
    print_section("8. 卫星刚体姿态与扫描策略")
    gyro = GyroscopeDynamics(A1=1.0, A2=1.0, A3=0.5, m=0.1)
    y0 = np.array([0.0, np.radians(45.0), 0.0, 0.1, 0.05, 2.0])
    t_arr, y_arr = gyro.integrate(y0, (0.0, 10.0), n_steps=500)
    print(f"  积分步数: {len(t_arr)}")
    print(f"  末端欧拉角: ψ={np.degrees(y_arr[-1,0]):.2f}°, "
          f"θ={np.degrees(y_arr[-1,1]):.2f}°, φ={np.degrees(y_arr[-1,2]):.2f}°")
    print(f"  末端角速度: ω=({y_arr[-1,3]:.4f}, {y_arr[-1,4]:.4f}, {y_arr[-1,5]:.4f}) rad/s")
    # 扫描轨迹
    t_scan, th_scan, ph_scan = generate_scanning_trajectory(n_steps=1000)
    hits = compute_hit_map(th_scan, ph_scan, n_theta=18, n_phi=36)
    uniformity = compute_coverage_uniformity(hits)
    print(f"  扫描覆盖均匀性 U = {uniformity:.4f}")

    # ===================================================================
    # 9. 最小二乘测试与参数拟合
    # ===================================================================
    print_section("9. 病态最小二乘测试与宇宙学参数拟合")
    lls_results = run_lls_test_suite()
    for test_name, res in lls_results.items():
        print(f"  [{test_name}]")
        print(f"    cond(A) = {res['cond_A']:.4e}")
        print(f"    QR 残差 = {res['residual_qr']:.4e}")
        print(f"    与精确解误差 = {res['error_vs_exact']:.4e}")

    # 拟合演示
    l_fit = np.arange(2, 41, 2)
    params_true = np.array([2.1, 0.965, 0.0224, 0.120, 0.673])
    Cl_theory = np.array([theory_Cl_model(params_true, np.array([li]))[0] for li in l_fit])
    sigma_noise = Cl_theory * 0.05 + 1e-10
    Cl_data = Cl_theory + np.random.randn(len(l_fit)) * sigma_noise

    fitter = CosmologyFitter(l_fit, Cl_data, sigma_noise,
                             ["A_s", "n_s", "omb", "omc", "h"])
    p0 = np.array([2.0, 0.96, 0.022, 0.12, 0.67])
    p_best, cov = fitter.fit(p0, max_iter=10)
    print(f"  χ^2 拟合结果:")
    for name, val, err in zip(fitter.param_names, p_best, np.sqrt(np.diag(cov))):
        print(f"    {name} = {val:.5f} ± {err:.5f}")
    print(f"  最终 χ^2 / dof = {fitter.chi2(p_best):.4f} / {len(l_fit)-len(p0)}")

    # Fisher 矩阵
    fisher = compute_fisher_matrix(l_fit, params_true, sigma_noise)
    print(f"  Fisher 矩阵条件数: {np.linalg.cond(fisher):.4e}")

    # ===================================================================
    # 10. 综合统计与性能
    # ===================================================================
    print_section("10. 综合统计")
    t_elapsed = time.time() - t_start
    print(f"  总运行时间: {t_elapsed:.3f} s")
    print(f"  所有模块零报错通过")
    print("\n" + "=" * 72)
    print("  CMB 各向异性综合分析系统 — 运行完成")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
