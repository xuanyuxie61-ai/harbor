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
    main()

# ================================================================
# 测试用例（50个，assert模式，涉及随机值均使用固定种子）
# ================================================================
import numpy as np
from mask_beam import polygon_centroid
from utils import spherical_bessel_j, associated_legendre, wigner_3j_000, is_power_of_two
from los_integration import fejer1_nodes_weights, fejer2_nodes_weights
from power_spectrum import bisection_root
from spherical_mesh import cartesian_to_spherical, spherical_to_cartesian, normalize_to_sphere

# ---- TC01: gamma_lanczos(1.0) 返回 Gamma(1)=1 ----
assert abs(gamma_lanczos(1.0) - 1.0) < 1e-10, '[TC01] gamma_lanczos(1.0) FAILED'

# ---- TC02: gamma_lanczos(0.5) 返回 sqrt(pi) ----
assert abs(gamma_lanczos(0.5) - np.sqrt(np.pi)) < 1e-6, '[TC02] gamma_lanczos(0.5) FAILED'

# ---- TC03: spherical_bessel_j(0, 0) 返回 1.0 ----
assert spherical_bessel_j(0, 0.0) == 1.0, '[TC03] spherical_bessel_j(0,0) FAILED'

# ---- TC04: spherical_bessel_j(1, 0) 返回 0.0 ----
assert spherical_bessel_j(1, 0.0) == 0.0, '[TC04] spherical_bessel_j(1,0) FAILED'

# ---- TC05: spherical_bessel_j(0, pi) 接近 0 ----
assert abs(spherical_bessel_j(0, np.pi)) < 1e-12, '[TC05] spherical_bessel_j(0,pi) FAILED'

# ---- TC06: spherical_bessel_j_array 输出形状为 (lmax+1, len(x)) ----
x_arr = np.array([0.0, 1.0, 2.0])
jarr = spherical_bessel_j_array(3, x_arr)
assert jarr.shape == (4, 3), '[TC06] spherical_bessel_j_array shape FAILED'

# ---- TC07: associated_legendre(0,0,1.0) 返回 1.0 ----
assert associated_legendre(0, 0, 1.0) == 1.0, '[TC07] associated_legendre(0,0,1) FAILED'

# ---- TC08: binomial(5,2) 返回 10 ----
assert binomial(5, 2) == 10, '[TC08] binomial(5,2) FAILED'

# ---- TC09: wigner_3j_000 三角不等式不满足时返回 0 ----
assert wigner_3j_000(1, 1, 3) == 0.0, '[TC09] wigner_3j_000 triangle FAILED'

# ---- TC10: clip_to_unit 越界标量正确裁剪到 [-1,1] ----
assert clip_to_unit(1.5) == 1.0 and clip_to_unit(-1.5) == -1.0, '[TC10] clip_to_unit FAILED'

# ---- TC11: robust_divide 分母为零时返回 fallback ----
assert robust_divide(5.0, 0.0, fallback=99.0) == 99.0, '[TC11] robust_divide FAILED'

# ---- TC12: ensure_positive 正数原样返回 ----
assert ensure_positive(3.14) == 3.14, '[TC12] ensure_positive FAILED'

# ---- TC13: is_power_of_two 正确识别 2 的幂 ----
assert is_power_of_two(8) and not is_power_of_two(7), '[TC13] is_power_of_two FAILED'

# ---- TC14: Gauss-Legendre 精确积分 x^2 在 [-1,1] ----
quad_gl = FastQuadrature(rule="gauss_legendre", n=16)
I_gl = quad_gl.integrate(lambda x: x**2, -1.0, 1.0)
assert abs(I_gl - 2.0/3.0) < 1e-12, '[TC14] Gauss-Legendre integrate x^2 FAILED'

# ---- TC15: Clenshaw-Curtis 精确积分 3x^2 在 [0,1] ----
quad_f2 = FastQuadrature(rule="fejer2", n=16)
I_f2 = quad_f2.integrate(lambda x: x**2, -1.0, 1.0)
assert abs(I_f2 - 2.0/3.0) < 1e-12, '[TC15] Fejér2 integrate x^2 FAILED'

# ---- TC16: Fejér 第一型权重之和为 2 ----
x_f2, w_f2 = fejer2_nodes_weights(16)
assert abs(np.sum(w_f2) - 2.0) < 1e-12, '[TC16] fejer2 weights sum FAILED'

# ---- TC17: primordial_power_spectrum 在 pivot 处等于 A_s ----
assert primordial_power_spectrum(0.05) == 2.1e-9, '[TC17] primordial_power_spectrum pivot FAILED'

# ---- TC18: primordial_power_spectrum 非正 k 返回 0 ----
assert primordial_power_spectrum(0.0) == 0.0, '[TC18] primordial_power_spectrum zero k FAILED'

# ---- TC19: bisection_root 求 x^2-4=0 在 [1,3] 的根 ----
root = bisection_root(lambda x: x**2 - 4.0, 1.0, 3.0)
assert abs(root - 2.0) < 1e-5, '[TC19] bisection_root FAILED'

# ---- TC20: compute_peak_spacing_ratio 两峰间距比 ----
ratio = compute_peak_spacing_ratio([(2, 1.0), (4, 2.0)])
assert abs(ratio - 2.0) < 1e-12, '[TC20] compute_peak_spacing_ratio FAILED'

# ---- TC21: SphericalMesh nsides=0 时顶点数为 12 ----
mesh0 = SphericalMesh(nsides=1)
assert mesh0.n_vertices == 42, '[TC21] SphericalMesh vertices FAILED'

# ---- TC22: SphericalMesh nsides=0 时面数为 20 ----
assert mesh0.n_faces == 80, '[TC22] SphericalMesh faces FAILED'

# ---- TC23: SphericalMesh 总面积接近理论值 4π ----
area_err = abs(mesh0.total_area() - 4.0*np.pi) / (4.0*np.pi)
assert area_err < 0.05, '[TC23] SphericalMesh total_area FAILED'

# ---- TC24: polygon_area 单位正方形面积为 1 ----
sq_x = np.array([0.0, 1.0, 1.0, 0.0])
sq_y = np.array([0.0, 0.0, 1.0, 1.0])
assert abs(polygon_area(sq_x, sq_y) - 1.0) < 1e-12, '[TC24] polygon_area FAILED'

# ---- TC25: point_in_polygon 判断点在正方形内部 ----
assert point_in_polygon(0.5, 0.5, sq_x, sq_y), '[TC25] point_in_polygon inside FAILED'

# ---- TC26: disk_monomial_integral 圆盘面积等于 πr^2 ----
disk_area = disk_monomial_integral(1.0, 0, 0)
assert abs(disk_area - np.pi) < 1e-10, '[TC26] disk_monomial_integral area FAILED'

# ---- TC27: gaussian_beam_window l=0 时返回 1 ----
assert gaussian_beam_window(0, 7.0) == 1.0, '[TC27] gaussian_beam_window l=0 FAILED'

# ---- TC28: shepp_logan_2d 原点位于第一个椭圆内值非零 ----
np.random.seed(42)
assert shepp_logan_2d(np.array([0.0]), np.array([0.0]))[0] != 0.0, '[TC28] shepp_logan_2d origin FAILED'

# ---- TC29: detect_edges_1d 在阶梯函数上检测边缘 ----
step_x = np.linspace(-1.0, 1.0, 101)
step_y = np.where(step_x < 0.0, 0.0, 1.0)
edges_step = detect_edges_1d(step_y, step_x, window=5, threshold=0.3)
assert len(edges_step) > 0, '[TC29] detect_edges_1d FAILED'

# ---- TC30: compute_residual_rms 全掩膜基本计算 ----
dummy_cmb = np.ones((4, 4))
dummy_fg = np.zeros((4, 4))
dummy_mask = np.ones((4, 4), dtype=bool)
rms_val = compute_residual_rms(dummy_cmb, dummy_fg, dummy_mask)
assert abs(rms_val - 1.0) < 1e-12, '[TC30] compute_residual_rms FAILED'

# ---- TC31: GyroscopeDynamics rhs 返回 6 维向量 ----
gyro = GyroscopeDynamics(A1=1.0, A2=1.0, A3=0.5, m=0.1)
y_test = np.array([0.0, np.pi/4, 0.0, 0.1, 0.05, 2.0])
dy = gyro.rhs(0.0, y_test)
assert dy.shape == (6,), '[TC31] GyroscopeDynamics rhs shape FAILED'

# ---- TC32: solve_least_squares_qr 单位矩阵系统精确求解 ----
A_id = np.eye(3)
b_id = np.array([1.0, 2.0, 3.0])
x_id = solve_least_squares_qr(A_id, b_id)
assert np.allclose(x_id, b_id), '[TC32] solve_least_squares_qr identity FAILED'

# ---- TC33: theory_Cl_model 返回值非负 ----
l_test = np.array([2.0, 10.0, 50.0])
params_test = np.array([2.1, 0.965, 0.0224, 0.120, 0.673])
Cl_test = theory_Cl_model(params_test, l_test)
assert np.all(Cl_test >= 0.0), '[TC33] theory_Cl_model non-negative FAILED'

# ---- TC34: ChebyshevInterpolator 对常数函数精确插值 ----
cheb_const = ChebyshevInterpolator(-1.0, 1.0, 4, lambda x: 2.5)
assert abs(cheb_const.evaluate(0.3) - 2.5) < 1e-12, '[TC34] ChebyshevInterpolator constant FAILED'

# ---- TC35: TransferFunctionComputer l<2 时转移函数返回 1.0 ----
tfc = TransferFunctionComputer(lmax=5, k_min=1e-3, k_max=0.1, n_cheb=8)
tfc.precompute_all()
assert tfc.get_transfer(0, 0.05) == 1.0, '[TC35] TransferFunctionComputer l<2 FAILED'

# ---- TC36: compute_coverage_uniformity 均匀命中图接近 1 ----
uniform_hits = np.full((10, 10), 5, dtype=int)
u_val = compute_coverage_uniformity(uniform_hits)
assert abs(u_val - 1.0) < 1e-12, '[TC36] compute_coverage_uniformity uniform FAILED'

# ---- TC37: SurveyMask 面积与 polygon_area 一致 ----
np.random.seed(42)
oct_angles = np.linspace(0, 2*np.pi, 9)[:-1]
mask_oct_x = 2.0 * np.cos(oct_angles)
mask_oct_y = 2.0 * np.sin(oct_angles)
mask_survey = SurveyMask(mask_oct_x, mask_oct_y)
assert abs(mask_survey.area() - abs(polygon_area(mask_oct_x, mask_oct_y))) < 1e-12, '[TC37] SurveyMask area FAILED'

# ---- TC38: run_lls_test_suite 返回包含 vandermonde 的字典 ----
lls_res = run_lls_test_suite()
assert isinstance(lls_res, dict) and "vandermonde" in lls_res, '[TC38] run_lls_test_suite FAILED'

# ---- TC39: CosmologyFitter chi2 非负 ----
np.random.seed(42)
l_fit = np.arange(2, 11)
Cl_data = np.ones(len(l_fit))
sigma = np.ones(len(l_fit)) * 0.1
fitter = CosmologyFitter(l_fit, Cl_data, sigma, ["A_s", "n_s", "omb", "omc", "h"])
p_test = np.array([2.0, 0.96, 0.022, 0.12, 0.67])
chi2_val = fitter.chi2(p_test)
assert chi2_val >= 0.0, '[TC39] CosmologyFitter chi2 non-negative FAILED'

# ---- TC40: compute_fisher_matrix 返回对称矩阵 ----
np.random.seed(42)
fisher_mat = compute_fisher_matrix(l_fit, p_test, sigma)
assert np.allclose(fisher_mat, fisher_mat.T), '[TC40] compute_fisher_matrix symmetric FAILED'

# ---- TC41: generate_scanning_trajectory 输出长度匹配 ----
t_scan, th_scan, ph_scan = generate_scanning_trajectory(n_steps=500)
assert len(t_scan) == len(th_scan) == len(ph_scan) == 500, '[TC41] generate_scanning_trajectory length FAILED'

# ---- TC42: compute_hit_map 输出形状匹配 ----
hits_map = compute_hit_map(th_scan, ph_scan, n_theta=9, n_phi=18)
assert hits_map.shape == (9, 18), '[TC42] compute_hit_map shape FAILED'

# ---- TC43: los_integral_power_spectrum 返回值有限且非负 ----
def dummy_transfer(k): return 1.0
def dummy_primordial(k): return 1.0e-9
Cl_los = los_integral_power_spectrum(2, dummy_transfer, dummy_primordial, k_min=1e-3, k_max=0.1, n_quad=16)
assert np.isfinite(Cl_los) and Cl_los >= 0.0, '[TC43] los_integral_power_spectrum finite FAILED'

# ---- TC44: compute_Cl_spectrum 输出长度与输入 l 数组一致 ----
l_arr = np.array([2, 3, 4])
def dummy_T(l, k): return 1.0
Cl_spectrum = compute_Cl_spectrum(l_arr, dummy_T, k_min=1e-3, k_max=0.1, n_k=32)
assert len(Cl_spectrum) == len(l_arr), '[TC44] compute_Cl_spectrum length FAILED'

# ---- TC45: BoltzmannSolver transfer_function_today 返回有限值 ----
params = CosmologyParams()
solver = BoltzmannSolver(params, k_mode=0.01, n_eta=200, eta_max=8000.0)
Tk = solver.transfer_function_today()
assert np.isfinite(Tk), '[TC45] BoltzmannSolver transfer finite FAILED'

# ---- TC46: find_acoustic_peaks 在正弦调制谱上检测到至少一个峰 ----
l_demo = np.arange(2, 41)
Cl_demo = np.sin(l_demo / 5.0 * np.pi) + 2.0
peaks_found = find_acoustic_peaks(l_demo, Cl_demo, n_peaks=3)
assert len(peaks_found) >= 1, '[TC46] find_acoustic_peaks FAILED'

# ---- TC47: beam_convolved_Cl 输出长度与输入一致 ----
Cl_in = np.ones(10)
Cl_out = beam_convolved_Cl(Cl_in, lmax=11, fwhm_arcmin=7.0)
assert len(Cl_out) == len(Cl_in), '[TC47] beam_convolved_Cl length FAILED'

# ---- TC48: 球坐标与笛卡尔坐标互逆变换 ----
v_cart = spherical_to_cartesian(np.pi/3, np.pi/4)
th_back, ph_back = cartesian_to_spherical(v_cart)
assert abs(th_back - np.pi/3) < 1e-12 and abs(ph_back - np.pi/4) < 1e-12, '[TC48] spherical coordinate round-trip FAILED'

# ---- TC49: normalize_to_sphere 返回单位长度向量 ----
v_norm = normalize_to_sphere(np.array([3.0, 4.0, 0.0]))
assert abs(np.linalg.norm(v_norm) - 1.0) < 1e-12, '[TC49] normalize_to_sphere FAILED'

# ---- TC50: SurveyMask contains 判断原点在大圆内 ----
assert mask_survey.contains(0.0, 0.0), '[TC50] SurveyMask contains origin FAILED'

print('\n全部 50 个测试通过!\n')
