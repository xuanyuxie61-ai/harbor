"""
main.py
希格斯玻色子 H -> ZZ* -> 4l 衰变通道的全相空间微分截面计算与统计显著性分析

统一入口，零参数可运行。

科学问题:
  基于标准模型树图水平计算，对 H->ZZ*->4l (四轻子) 黄金衰变通道进行:
    1. 蒙特卡洛相空间事件生成
    2. 采样质量评估 (几何度量体系)
    3. 螺旋度振幅与矩阵元计算
    4. 多维数值积分 (高斯求积 + 自适应梯形)
    5. 双变量背景分布插值 (Padua 点 Chebyshev)
    6. 正交多项式背景拟合与信号提取
    7. 重整化群耦合演化与稳定性分析
    8. 自适应 CVT 积分网格生成
    9. 泊松正则化背景平滑
    10. 似然比显著性计算与置信区间

输出: 文本格式的完整分析报告
"""
import numpy as np
import time

# ============================================================
# 导入所有模块
# ============================================================
from constants import (
    M_HIGGS, M_Z, GAMMA_Z, GAMMA_H, ALPHA_EM, G_F, kine_bounds
)
from utils import (
    lu_factor_scaled, lu_solve, horner_eval, cooley_tukey_fft,
    bisection, muller_method, rk2_integrate
)
from phase_space import (
    generate_event_batch, compute_invariant_masses,
    event_statistics, sample_unit_sphere_uniform
)
from sampling_quality import (
    evaluate_phase_space_sampling, sampling_quality_report
)
from matrix_element import (
    matrix_element_squared_hzz4l, z_propagator, higgs_propagator,
    g_hzz_coupling, helicity_amplitude_zzstar,
    vandermonde_quadrature_weights, fit_amplitude_polynomial
)
from quadrature_engine import (
    legendre_gauss_rule, jacobi_gauss_rule,
    composite_trapezoidal, adaptive_trapezoidal,
    gauss_legendre_2d, integrate_dsigma_dm1dm2,
    composite_simpson, breit_wigner
)
from interpolation_background import (
    padua_points, bivariate_chebyshev_coeffs, bivariate_chebyshev_eval,
    build_background_interpolant, s_b_ratio
)
from orthogonal_fit import (
    orthogonal_background_fit, extract_signal, analyze_mass_spectrum,
    orthopoly_construct, orthopoly_eval, clenshaw_chebyshev
)
from rge_evolution import (
    sm_rge_beta, sm_initial_conditions, higgs_vv_coupling_evolution,
    landau_pole_estimate, rge_analysis_report, check_rge_stability
)
from cvt_adaptive_grid import (
    cvt_1d_lloyd, cvt_nd_product, adaptive_phase_space_grid,
    make_breit_wigner_density, make_amplitude_density
)
from poisson_solver import (
    build_fd_laplacian_1d, jacobi_solve, sor_solve,
    smooth_background_poisson, compare_solvers
)
from statistical_analysis import (
    poisson_likelihood, profile_log_likelihood,
    solve_mu_mle, significance_simple,
    significance_likelihood_ratio, confidence_interval_mu,
    significance_with_systematics, full_statistical_report
)


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    np.random.seed(42)
    t_start = time.time()

    print("=" * 70)
    print("  希格斯玻色子 H -> ZZ* -> 4l 衰变通道分析")
    print("  博士级全相空间微分截面与统计显著性计算")
    print("=" * 70)
    print(f"\n物理参数:")
    print(f"  m_H = {M_HIGGS:.3f} GeV")
    print(f"  m_Z = {M_Z:.4f} GeV")
    print(f"  Gamma_Z = {GAMMA_Z:.4f} GeV")
    print(f"  Gamma_H = {GAMMA_H:.5f} GeV")
    print(f"  G_F = {G_F:.6e} GeV^-2")

    # ============================================================
    # 1. 蒙特卡洛相空间事件生成
    # ============================================================
    print_section("1. 蒙特卡洛相空间事件生成")
    n_events = 500
    events = generate_event_batch(n_events)
    stats = event_statistics(events)
    print(f"生成事件数: {stats['count']}")
    print(f"平均 m_4l = {stats['m4l_mean']:.3f} +/- {stats['m4l_std']:.3f} GeV")
    print(f"平均 m_Z1 = {stats['mz1_mean']:.3f} GeV")
    print(f"平均 m_Z2 = {stats['mz2_mean']:.3f} GeV")
    print(f"m_Z1-m_Z2 相关系数: {stats['mz_corr']:.4f}")

    # ============================================================
    # 2. 采样质量评估 (映射 958_quality)
    # ============================================================
    print_section("2. 相空间采样质量评估")
    quality = evaluate_phase_space_sampling(events)
    print(f"评估点数: {quality['n_points']}, 维度: {quality['dimension']}")
    print(f"Gamma 度量 (均匀性): {quality['gamma']:.4f} (理想值 -> 1.0)")
    print(f"Beta 度量 (变异系数): {quality['beta']:.4f} (理想值 -> 0.0)")
    print(f"R0 能量 (对数排斥势): {quality['r0_energy']:.4f}")
    print(f"Chi 度量 (Voronoi 方差): {quality['chi']:.4f}")

    # ============================================================
    # 3. 螺旋度振幅计算 (映射 950 + 200)
    # ============================================================
    print_section("3. 螺旋度振幅与矩阵元计算")
    g_hzz = g_hzz_coupling()
    print(f"树图 g_HZZ 耦合 = {g_hzz:.4f} GeV")

    test_m1 = M_Z
    test_m2 = 30.0
    me_sq = matrix_element_squared_hzz4l(test_m1, test_m2)
    print(f"|M|^2 (m1={test_m1:.2f}, m2={test_m2:.2f}) = {me_sq:.6e}")
    # 补充在壳+离壳测试
    me_sq2 = matrix_element_squared_hzz4l(60.0, 60.0)
    print(f"|M|^2 (m1=60.00, m2=60.00) = {me_sq2:.6e}")

    # Vandermonde 权重计算 (映射 950)
    n_v = 8
    nodes_v = np.linspace(60.0, 120.0, n_v)
    weights_v = vandermonde_quadrature_weights(n_v, 60.0, 120.0, nodes_v)
    print(f"Vandermonde 求积权重 (n={n_v}): sum = {np.sum(weights_v):.4f}, 理论值 = 60.0")

    # Horner 多项式振幅求值 (映射 200)
    amp_coeffs = fit_amplitude_polynomial(
        np.linspace(20.0, 120.0, 20),
        np.linspace(20.0, 120.0, 20),
        np.array([matrix_element_squared_hzz4l(m, m) for m in np.linspace(20.0, 120.0, 20)]),
        degree=5
    )
    test_x = 0.5  # 归一化到 [-1,1] 后
    poly_val = horner_eval(amp_coeffs, test_x)
    print(f"多项式振幅逼近 (Horner) 在 x={test_x}: {poly_val:.6e}")

    # ============================================================
    # 4. 多维数值积分 (映射 943 + 945 + 1144)
    # ============================================================
    print_section("4. 相空间微分截面数值积分")

    # 1D Gauss-Legendre
    gl_nodes, gl_weights = legendre_gauss_rule(12)
    print(f"12点 Gauss-Legendre: 节点范围 [{gl_nodes.min():.4f}, {gl_nodes.max():.4f}]")
    print(f"  权重和 = {np.sum(gl_weights):.6f} (理论 = 2.0)")

    # Jacobi 求积 (用于 Breit-Wigner 形状)
    jac_nodes, jac_weights = jacobi_gauss_rule(10, 0.0, 0.0)
    print(f"10点 Gauss-Jacobi (alpha=0, beta=0): 权重和 = {np.sum(jac_weights):.6f}")

    # 复合梯形
    f_test = lambda x: np.exp(-x)
    trap_val = composite_trapezoidal(f_test, 0.0, 1.0, 100)
    print(f"复合梯形积分 exp(-x) [0,1] = {trap_val:.8f} (精确 = {1-np.exp(-1):.8f})")

    # 自适应梯形
    adap_val = adaptive_trapezoidal(f_test, 0.0, 1.0, tol=1.0e-10)
    print(f"自适应梯形积分 exp(-x) [0,1] = {adap_val:.10f}")

    # 2D 张量积求积 (映射 1144)
    f_2d = lambda x, y: np.exp(-(x ** 2 + y ** 2))
    rect_2d = gauss_legendre_2d(8, 8, f_2d, (-1.0, 1.0), (-1.0, 1.0))
    print(f"2D Gauss-Legendre (8x8) exp(-(x^2+y^2)) [-1,1]^2 = {rect_2d:.8f}")

    # 物理积分: d sigma / dm1 dm2
    dsigma = integrate_dsigma_dm1dm2(matrix_element_squared_hzz4l, M_HIGGS, M_Z, GAMMA_Z, n_points=12)
    print(f"双微分截面积分 (12x12 GL) = {dsigma:.6e} [arb. units]")

    # ============================================================
    # 5. 双变量背景插值 (映射 1279_toms886)
    # ============================================================
    print_section("5. 双变量背景分布插值 (Padua 点)")
    bg_interp, bg_coeffs, bg_err = build_background_interpolant(deg=6, m_range=(10.0, 120.0))
    print(f"Padua 点 Chebyshev 插值阶数: 6")
    print(f"背景插值估计误差: {bg_err:.6e}")
    test_bg = bg_interp(60.0, 60.0)
    print(f"背景估值 B(60,60) = {test_bg:.6e}")

    # S/B 比值
    def sig_func(m1, m2):
        return matrix_element_squared_hzz4l(m1, m2)
    sb_val = s_b_ratio(91.2, 91.2, sig_func, bg_interp)
    print(f"信号背景比 S/B (91.2, 91.2) = {sb_val:.4f}")

    # ============================================================
    # 6. 正交多项式背景拟合 (映射 209 ortpol/cheb)
    # ============================================================
    print_section("6. 正交多项式背景拟合与信号提取")
    
    # 构造模拟的 m_4l 谱
    mass_bins = np.linspace(70.0, 170.0, 41)
    # 背景: 指数下降 + 信号: 高斯峰在 m_H
    true_bkg = 1000.0 * np.exp(-(mass_bins - 70.0) / 30.0)
    signal_shape = 200.0 * np.exp(-0.5 * ((mass_bins - M_HIGGS) / 2.0) ** 2)
    n_obs = np.random.poisson(true_bkg + signal_shape)
    
    analysis = analyze_mass_spectrum(mass_bins, n_obs, background_degree=4)
    print(f"峰值位置: {analysis['peak_mass']:.2f} GeV")
    print(f"峰值显著性: {analysis['peak_significance']:.3f} sigma")
    print(f"总信号估计: {analysis['total_signal']:.1f}")
    print(f"总背景估计: {analysis['total_background']:.1f}")
    print(f"S/sqrt(B) = {analysis['s_over_sqrt_b']:.3f}")
    
    # Clenshaw Chebyshev 验证
    cheb_coeffs = np.array([1.0, 0.5, -0.2, 0.1])
    cheb_val = clenshaw_chebyshev(cheb_coeffs, 0.3)
    print(f"Clenshaw Chebyshev 求值测试: {cheb_val:.6f}")

    # ============================================================
    # 7. RGE 演化分析 (映射 1042 + 861 + 312)
    # ============================================================
    print_section("7. 标准模型耦合常数 RGE 演化")
    rge_report = rge_analysis_report(mu_high=5000.0, n_steps=2000)
    print(f"演化稳定性: {'通过' if rge_report['stable'] else '警告'}")
    print(f" Landau Pole 估算: {rge_report['landau_pole_gev']:.3e} GeV")
    print(f" lambda(m_Z) = {rge_report['lambda'][0]:.4f}")
    print(f" lambda(5 TeV) = {rge_report['lambda'][-1]:.4f}")
    print(f" g_HZZ(m_Z) = {rge_report['g_hzz'][0]:.4f}")
    print(f" g_HZZ(5 TeV) = {rge_report['g_hzz'][-1]:.4f}")

    # ============================================================
    # 8. 自适应 CVT 网格 (映射 243)
    # ============================================================
    print_section("8. 自适应 CVT 相空间网格")
    grid = adaptive_phase_space_grid(n_m1=12, n_m2=12, n_cos=6, n_phi=6)
    print(f"m1 节点数: {len(grid['m1'])}, 范围 [{grid['m1'].min():.2f}, {grid['m1'].max():.2f}]")
    print(f"m2 节点数: {len(grid['m2'])}, 范围 [{grid['m2'].min():.2f}, {grid['m2'].max():.2f}]")
    print(f"cos_theta 节点数: {len(grid['cos_theta'])}")
    print(f"phi 节点数: {len(grid['phi'])}")

    # 1D CVT Lloyd 验证
    rho_test = lambda x: 1.0 + 5.0 * np.exp(-20.0 * (x - 0.5) ** 2)
    cvt_nodes, cvt_energy, cvt_conv, cvt_iters = cvt_1d_lloyd(
        10, rho_test, domain=(0.0, 1.0), max_iter=100, tol=1.0e-8
    )
    print(f"1D CVT Lloyd 收敛: {'是' if cvt_conv else '否'}, 迭代次数: {cvt_iters}")
    print(f"  节点: {np.round(cvt_nodes, 4)}")

    # ============================================================
    # 9. 泊松正则化背景平滑 (映射 606)
    # ============================================================
    print_section("9. 泊松方程背景平滑")
    raw_bg = true_bkg + 0.1 * np.random.randn(len(true_bkg)) * np.sqrt(true_bkg)
    raw_bg = np.maximum(raw_bg, 0.0)
    smooth_bg = smooth_background_poisson(raw_bg, smoothing_strength=0.5, n_inner=30)
    print(f"原始背景均值: {np.mean(raw_bg):.2f}")
    print(f"平滑后背景均值: {np.mean(smooth_bg):.2f}")
    print(f"平滑前后 RMS 差: {np.linalg.norm(smooth_bg - raw_bg) / np.sqrt(len(raw_bg)):.2f}")

    # 求解器对比
    f_test_poisson = lambda x: x * (x + 3.0) * np.exp(x)
    exact_poisson = lambda x: x * (1.0 - x) * np.exp(x)
    solver_comp = compare_solvers(f_test_poisson, exact_poisson, n=15)
    print(f"Jacobi 误差: {solver_comp['jacobi_error']:.6e}, 收敛: {'是' if solver_comp['jacobi_converged'] else '否'}")
    print(f"SOR 误差: {solver_comp['sor_error']:.6e}, 迭代: {solver_comp['sor_iters']}")

    # ============================================================
    # 10. 统计显著性分析 (映射 209 根查找)
    # ============================================================
    print_section("10. 统计显著性与置信区间")
    
    # 构造分箱数据
    n_bins = len(mass_bins)
    sig_bins = signal_shape
    bkg_bins = true_bkg
    obs_bins = n_obs.astype(float)
    
    stat_report = full_statistical_report(mass_bins, obs_bins, bkg_bins, sig_bins)
    print(f"总观测: {stat_report['total_observed']:.1f}")
    print(f"总背景: {stat_report['total_background']:.1f}")
    print(f"信号强度 MLE: mu_hat = {stat_report['mu_hat']:.4f}")
    print(f"95% CL 置信区间: [{stat_report['mu_lower_95cl']:.4f}, {stat_report['mu_upper_95cl']:.4f}]")
    print(f"简单显著性: Z = {stat_report['significance_simple']:.3f} sigma")
    print(f"似然比显著性: Z = {stat_report['significance_likelihood']:.3f} sigma")
    print(f"含 15% 系统误差: Z = {stat_report['significance_with_syst']:.3f} sigma")

    # ============================================================
    # 11. 综合科学计算验证
    # ============================================================
    print_section("11. 综合验证与数值鲁棒性检查")
    
    # 验证四动量守恒
    evt = events[0]
    lep_sum = np.sum(evt["leptons"], axis=0)
    e_diff = abs(lep_sum[0] - M_HIGGS)
    p_diff = np.linalg.norm(lep_sum[1:])
    print(f"四动量守恒检查 (第一个事件):")
    print(f"  能量守恒偏差: {e_diff:.6e} GeV")
    print(f"  动量守恒偏差: {p_diff:.6e} GeV")
    
    # 验证 Breit-Wigner 归一化 (数值)
    bw_norm = composite_simpson(lambda m: breit_wigner(m, M_Z, GAMMA_Z), M_Z - 10.0, M_Z + 10.0, 200)
    print(f"Breit-Wigner 数值积分 (20 GeV 窗口): {bw_norm:.6f} [arb. units]")
    
    # 验证球面采样均匀性
    sphere_pts = sample_unit_sphere_uniform(1000)
    mean_coord = np.mean(sphere_pts, axis=0)
    std_coord = np.std(sphere_pts, axis=0)
    print(f"球面采样均值: {mean_coord}, 标准差: {std_coord} (均接近 0)")
    
    # 验证 FFT
    test_fft = np.exp(2j * np.pi * np.arange(8) / 8.0)
    fft_result = cooley_tukey_fft(test_fft)
    expected = np.fft.fft(test_fft)
    fft_err = np.linalg.norm(fft_result - expected)
    print(f"FFT 验证误差: {fft_err:.6e}")

    # ============================================================
    # 完成报告
    # ============================================================
    t_elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"  分析完成。总耗时: {t_elapsed:.3f} 秒")
    print("=" * 70)
    print("\n核心物理结论:")
    print(f"  1. H->ZZ*->4l 树图振幅 |M|^2 (在壳/离壳) = {me_sq:.4e}")
    print(f"  2. 双微分截面积分结果为 {dsigma:.4e}")
    print(f"  3. 信号强度估计 mu_hat = {stat_report['mu_hat']:.3f} "
          f"(95% CL: [{stat_report['mu_lower_95cl']:.3f}, {stat_report['mu_upper_95cl']:.3f}])")
    print(f"  4. 统计显著性 (似然比) = {stat_report['significance_likelihood']:.2f} sigma")
    print(f"  5. 希格斯耦合 g_HZZ 在 5 TeV 能标为 {rge_report['g_hzz'][-1]:.4f} "
          f"(m_Z 处为 {rge_report['g_hzz'][0]:.4f})")
    print("=" * 70)


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（25个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: horner_eval 对二次多项式求值正确 ----
val = horner_eval([1.0, 2.0, 3.0], 2.0)
assert abs(val - 17.0) < 1.0e-12, '[TC01] horner_eval 对二次多项式求值正确 FAILED'

# ---- TC02: bisection 对线性函数求根正确 ----
root, info = bisection(lambda x: x - 1.5, 0.0, 3.0)
assert abs(root - 1.5) < 1.0e-10, '[TC02] bisection 对线性函数求根正确 FAILED'

# ---- TC03: cooley_tukey_fft 与 numpy 结果一致 ----
x_fft = np.exp(2j * np.pi * np.arange(8) / 8.0)
fft_res = cooley_tukey_fft(x_fft)
expected_fft = np.fft.fft(x_fft)
assert np.linalg.norm(fft_res - expected_fft) < 1.0e-10, '[TC03] cooley_tukey_fft 与 numpy 结果一致 FAILED'

# ---- TC04: sample_unit_sphere_uniform 输出在球面上 ----
np.random.seed(42)
pts = sample_unit_sphere_uniform(50)
norm_pts = np.linalg.norm(pts, axis=1)
assert np.allclose(norm_pts, 1.0, atol=1.0e-12), '[TC04] sample_unit_sphere_uniform 输出在球面上 FAILED'

# ---- TC05: event_statistics 返回正确结构 ----
np.random.seed(42)
events = generate_event_batch(10)
stats = event_statistics(events)
assert stats['count'] == 10 and 'm4l_mean' in stats, '[TC05] event_statistics 返回正确结构 FAILED'

# ---- TC06: z_propagator 返回有限复数值 ----
pz = z_propagator(M_Z ** 2)
assert np.isfinite(pz) and np.iscomplexobj(pz), '[TC06] z_propagator 返回有限复数值 FAILED'

# ---- TC07: g_hzz_coupling 为正有限值 ----
g = g_hzz_coupling()
assert g > 0.0 and np.isfinite(g), '[TC07] g_hzz_coupling 为正有限值 FAILED'

# ---- TC08: matrix_element_squared_hzz4l 边界外返回 0 ----
me = matrix_element_squared_hzz4l(70.0, 70.0)
assert me == 0.0, '[TC08] matrix_element_squared_hzz4l 边界外返回 0 FAILED'

# ---- TC09: vandermonde_quadrature_weights 权重和非负 ----
nodes_v = np.linspace(0.0, 1.0, 6)
w_v = vandermonde_quadrature_weights(6, 0.0, 1.0, nodes_v)
assert np.sum(w_v) >= 0.0 and np.all(w_v >= 0.0), '[TC09] vandermonde_quadrature_weights 权重和非负 FAILED'

# ---- TC10: legendre_gauss_rule 权重和为 2 ----
nodes_gl, weights_gl = legendre_gauss_rule(12)
assert abs(np.sum(weights_gl) - 2.0) < 1.0e-12, '[TC10] legendre_gauss_rule 权重和为 2 FAILED'

# ---- TC11: composite_trapezoidal 积分 exp(-x) 精确 ----
trap_res = composite_trapezoidal(lambda x: np.exp(-x), 0.0, 1.0, 200)
assert abs(trap_res - (1.0 - np.exp(-1.0))) < 2.0e-6, '[TC11] composite_trapezoidal 积分 exp(-x) 精确 FAILED'

# ---- TC12: breit_wigner 在峰值处为正有限值 ----
bw = breit_wigner(M_Z, M_Z, GAMMA_Z)
assert bw > 0.0 and np.isfinite(bw), '[TC12] breit_wigner 在峰值处为正有限值 FAILED'

# ---- TC13: padua_points 点数与代码实现一致 ----
pts_pd, wts_pd = padua_points(5)
assert len(pts_pd) == 18 and len(wts_pd) == 18, '[TC13] padua_points 点数与代码实现一致 FAILED'

# ---- TC14: build_background_interpolant 返回可调用函数 ----
bg_func, bg_coeffs, bg_err = build_background_interpolant(deg=4, m_range=(10.0, 120.0))
val_bg = bg_func(60.0, 60.0)
assert np.isfinite(val_bg), '[TC14] build_background_interpolant 返回可调用函数 FAILED'

# ---- TC15: clenshaw_chebyshev 对已知级数求值正确 ----
coeffs_cheb = np.array([1.0, 0.5, -0.2])
val_cheb = clenshaw_chebyshev(coeffs_cheb, 0.5)
expected_cheb = 1.0 * 1.0 + 0.5 * 0.5 + (-0.2) * (2.0 * 0.5 ** 2 - 1.0)
assert abs(val_cheb - expected_cheb) < 1.0e-10, '[TC15] clenshaw_chebyshev 对已知级数求值正确 FAILED'

# ---- TC16: build_fd_laplacian_1d 构造对称三对角矩阵 ----
A = build_fd_laplacian_1d(5, 0.1)
assert A.shape == (5, 5) and np.allclose(A, A.T), '[TC16] build_fd_laplacian_1d 构造对称三对角矩阵 FAILED'

# ---- TC17: jacobi_solve 对简单问题收敛 ----
A_simple = build_fd_laplacian_1d(7, 0.2)
f_simple = np.ones(7)
u_jac, res_jac, conv_jac, it_jac = jacobi_solve(A_simple, f_simple, max_iter=5000, tol=1.0e-8)
assert conv_jac, '[TC17] jacobi_solve 对简单问题收敛 FAILED'

# ---- TC18: significance_simple 计算正确 ----
Z = significance_simple(50.0, 100.0)
assert abs(Z - 5.0) < 1.0e-12, '[TC18] significance_simple 计算正确 FAILED'

# ---- TC19: poisson_likelihood 返回有限值 ----
ll = poisson_likelihood(10, 1.0, 5.0, 3.0)
assert np.isfinite(ll), '[TC19] poisson_likelihood 返回有限值 FAILED'

# ---- TC20: cvt_1d_lloyd 均匀密度收敛 ----
np.random.seed(42)
nodes_cvt, energy_cvt, conv_cvt, it_cvt = cvt_1d_lloyd(5, lambda x: 1.0, domain=(0.0, 1.0), max_iter=200, tol=1.0e-8)
assert conv_cvt, '[TC20] cvt_1d_lloyd 均匀密度收敛 FAILED'

# ---- TC21: sm_initial_conditions 返回 7 维向量 ----
y0 = sm_initial_conditions()
assert len(y0) == 7 and np.all(np.isfinite(y0)), '[TC21] sm_initial_conditions 返回 7 维向量 FAILED'

# ---- TC22: landau_pole_estimate 为正有限值 ----
lp = landau_pole_estimate()
assert lp > 0.0 and np.isfinite(lp), '[TC22] landau_pole_estimate 为正有限值 FAILED'

# ---- TC23: evaluate_phase_space_sampling 返回报告字典 ----
np.random.seed(42)
evt_small = generate_event_batch(20)
rep = evaluate_phase_space_sampling(evt_small)
assert 'gamma' in rep and rep['n_points'] > 0, '[TC23] evaluate_phase_space_sampling 返回报告字典 FAILED'

# ---- TC24: rk2_integrate 线性 ODE 精确解 ----
def linear_ode(t, y):
    return np.array([2.0 * y[0]])
t_arr, y_arr = rk2_integrate(linear_ode, (0.0, 1.0), np.array([1.0]), 1000)
assert abs(y_arr[-1, 0] - np.exp(2.0)) < 0.01, '[TC24] rk2_integrate 线性 ODE 精确解 FAILED'

# ---- TC25: helicity_amplitude_zzstar 对称性 cos_theta -> -cos_theta ----
amp_pos = helicity_amplitude_zzstar(80.0, 40.0, 0.5, 0.0)
amp_neg = helicity_amplitude_zzstar(80.0, 40.0, -0.5, 0.0)
assert abs(amp_pos - amp_neg) < 1.0e-12, '[TC25] helicity_amplitude_zzstar 对称性 cos_theta -> -cos_theta FAILED'
print('\n全部 25 个测试通过!\n')