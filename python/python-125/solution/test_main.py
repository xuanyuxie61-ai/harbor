"""
main.py
视网膜神经编码与视觉通路的多尺度计算模型 —— 统一入口

================================================================================
科学问题概述
================================================================================
本项目构建了一个从光感受器层到神经节细胞层的端到端多尺度计算模型，
模拟视觉刺激在视网膜中的处理过程。核心科学问题包括：

1. 视网膜空间结构的离散化与网格质量评估
2. 光感受器光转导级联与光适应动力学
3. 双极细胞感受野的中心-周边拮抗计算
4. 突触间隙神经递质的反应-扩散动力学
5. 神经节细胞的脉冲发放编码与信息效率分析
6. 大规模视网膜神经网络的线性系统求解
7. 视觉刺激空间参数的高维采样与组合探索

================================================================================
数学物理模型
================================================================================

【光转导ODE系统】
    d[PDE*]/dt = α·I_light - β·[PDE*]
    d[cGMP]/dt = α_gc_max / (1+([Ca²⁺]/K_gc)^{n_gc}) - γ·[PDE*]·[cGMP]
    d[Ca²⁺]/dt = -η·I_Ca

【光适应稳态方程】（有限差分Jacobi迭代）
    D·∇²C + S = 0
    C_{i,j}^{new} = (C_{i-1,j}+C_{i+1,j}+C_{i,j-1}+C_{i,j+1}+h²·S/D)/4

【Gray-Scott突触反应-扩散】
    ∂U/∂t = D_u·∇²U - U·V² + F·(1-U)
    ∂V/∂t = D_v·∇²V + U·V² - (F+K)·V

【差异高斯感受野】
    RF(x,y) = A_c·exp(-r²/(2σ_c²)) - A_s·exp(-r²/(2σ_s²))

【非齐次泊松发放】（Thinning方法）
    λ(t) = r_0 + g·max(0, R_bipolar(t))

【CBB网络矩阵求解】（Schur补）
    S = A4 - A3·A1^{-1}·A2
"""

import numpy as np
import time

# 导入各模块
from retina_geometry import (
    generate_hexagonal_photoreceptor_array,
    delaunay_triangulation_2d,
    evaluate_mesh_quality,
    convex_hull_2d,
    hexagon_moment_integral,
)
from photoreceptor import (
    solve_light_adaptation_steady_state,
    integrate_photocurrent_clenshaw_curtis,
    solve_phototransduction_rk4,
)
from bipolar_cell import (
    dog_receptive_field,
    compute_bipolar_response_convolution,
    decompose_rf_with_legendre_basis,
    reconstruct_rf_from_legendre_coeffs,
    gauss_legendre_quadrature,
    legendre_polynomial_value,
)
from synaptic_diffusion import (
    simulate_synaptic_transmission,
    compute_synaptic_efficacy,
)
from neural_encoding import (
    trig_interp_basis,
    trig_interpolate_spike_pattern,
    horner_polynomial_eval,
    analyze_spike_train,
    encoding_efficiency,
)
from spike_generation import (
    generate_inhomogeneous_poisson_spikes,
    simulate_rgc_spike_train,
    sample_neural_parameter_space,
    faure_generate,
)
from network_solver import (
    solve_cbb_system,
    build_retinal_network_matrix,
    band_lu_factorize,
    band_lu_solve,
)
from stimulus_generator import (
    sinusoidal_grating,
    gaussian_blob,
    white_noise_stimulus,
    drifting_grating,
    explore_synaptic_combinations,
    evaluate_tuning_function,
)
from math_utils import (
    simpson_integration,
    trapezoidal_integration,
    spherical_harmonic_y,
    matrix_condition_number_estimate,
    log_gamma_lanczos,
    erf_approx,
    relative_error,
)


def section_header(title: str):
    """打印章节标题。"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_retina_geometry():
    """运行视网膜几何建模模块。"""
    section_header("1. 视网膜几何离散化与网格质量评估")
    
    # 生成六边形感光细胞阵列（基于527_hexagon_integrals）
    radius = 2.0  # 感光细胞半径（微米）
    n_rings = 6
    photoreceptors = generate_hexagonal_photoreceptor_array(radius, n_rings)
    print(f"  生成六边形感光细胞阵列：{photoreceptors.shape[0]} 个细胞")
    
    # Delaunay三角化（基于469_geompack）
    triangles = delaunay_triangulation_2d(photoreceptors)
    print(f"  Delaunay三角化：{triangles.shape[0]} 个三角形")
    
    # 网格质量评估（基于1348_triangulation_quality）
    quality = evaluate_mesh_quality(photoreceptors, triangles)
    print(f"  网格质量指标：")
    print(f"    ALPHA最小值/平均值: {quality['alpha_min']:.4f} / {quality['alpha_ave']:.4f}")
    print(f"    Q最小值/平均值: {quality['q_min']:.4f} / {quality['q_ave']:.4f}")
    print(f"    面积 最小/最大/平均: {quality['area_min']:.4f} / {quality['area_max']:.4f} / {quality['area_ave']:.4f}")
    print(f"    边界边数: {quality['num_boundary_edges']}")
    
    # 凸包计算
    hull = convex_hull_2d(photoreceptors)
    print(f"  凸包顶点数: {len(hull)}")
    
    # 六边形矩计算（基于527_hexagon_integrals）
    # 定义单位六边形顶点
    hex_vertices = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3.0) / 2.0],
        [-0.5, np.sqrt(3.0) / 2.0],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3.0) / 2.0],
        [0.5, -np.sqrt(3.0) / 2.0],
    ], dtype=np.float64)
    
    moment_20 = hexagon_moment_integral(2, 0, hex_vertices)
    moment_02 = hexagon_moment_integral(0, 2, hex_vertices)
    moment_22 = hexagon_moment_integral(2, 2, hex_vertices)
    print(f"  单位六边形矩积分:")
    print(f"    M(2,0) = {moment_20:.6f}")
    print(f"    M(0,2) = {moment_02:.6f}")
    print(f"    M(2,2) = {moment_22:.6f}")
    
    return photoreceptors, triangles, quality


def run_photoreceptor_model():
    """运行光感受器模型。"""
    section_header("2. 光感受器光转导与光适应")
    
    # 光适应稳态方程（基于512_heated_plate）
    nx, ny = 32, 32
    source = np.zeros((nx, ny), dtype=np.float64)
    # 中心光源（模拟光斑刺激）
    cx, cy = nx // 2, ny // 2
    for i in range(nx):
        for j in range(ny):
            dist2 = (i - cx) ** 2 + (j - cy) ** 2
            source[i, j] = 50.0 * np.exp(-dist2 / 50.0)
    
    C_steady, iters, err = solve_light_adaptation_steady_state(
        nx, ny, 100.0, 100.0, 0.0, 100.0, source, epsilon=1e-6, max_iter=10000
    )
    print(f"  光适应稳态求解: 迭代{iters}次, 最终误差{err:.2e}")
    print(f"  稳态浓度范围: [{C_steady.min():.2f}, {C_steady.max():.2f}]")
    
    # Clenshaw-Curtis求积（基于144_cc_project）
    def intensity_profile(x):
        return 100.0 * np.exp(-x ** 2 / 0.5)
    
    photocurrent = integrate_photocurrent_clenshaw_curtis(
        intensity_profile, -1.0, 1.0, n=64
    )
    print(f"  Clenshaw-Curtis光电流积分: {photocurrent:.6f}")
    
    # 光转导ODE求解（RK4）
    def light_pulse(t):
        return 10.0 if 0.05 <= t <= 0.15 else 0.0
    
    y0 = np.array([0.0, 0.5, 0.1], dtype=np.float64)  # [PDE*, cGMP, Ca]
    params = {
        'alpha_pde': 2.0,
        'beta_pde': 5.0,
        'alpha_gc_max': 10.0,
        'K_gc': 0.1,
        'n_gc': 4.0,
        'gamma': 1.0,
        'eta': 1.0,
        'g_max': 1.0,
        'K_cGMP': 0.05,
        'V_m': -40.0,
        'E_Ca': 40.0,
    }
    
    t_arr, y_arr = solve_phototransduction_rk4(
        light_pulse, y0, (0.0, 0.5), 0.0005, params
    )
    print(f"  光转导ODE求解: t∈[0,0.5]s, {len(t_arr)}个时间点")
    print(f"  最终状态: PDE*={y_arr[-1,0]:.4f}, cGMP={y_arr[-1,1]:.4f}, Ca={y_arr[-1,2]:.4f}")
    
    return C_steady, photocurrent, (t_arr, y_arr)


def run_bipolar_cell():
    """运行双极细胞模型。"""
    section_header("3. 双极细胞感受野与Legendre基展开")
    
    # 生成差异高斯感受野
    nx, ny = 64, 64
    x = np.linspace(-1.0, 1.0, nx)
    y = np.linspace(-1.0, 1.0, ny)
    X, Y = np.meshgrid(x, y)
    
    rf = dog_receptive_field(X, Y, A_c=1.0, sigma_c=0.15, A_s=0.5, sigma_s=0.40)
    print(f"  DoG感受野: 中心σ={0.15}, 周边σ={0.40}")
    print(f"  感受野范围: [{rf.min():.4f}, {rf.max():.4f}]")
    
    # Gauss-Legendre求积节点（基于661_legendre_polynomial）
    x_gl, w_gl = gauss_legendre_quadrature(16)
    print(f"  Gauss-Legendre 16点求积: 节点范围[{x_gl.min():.4f}, {x_gl.max():.4f}]")
    
    # Legendre多项式值
    x_test = np.linspace(-1.0, 1.0, 100)
    V_leg = legendre_polynomial_value(8, x_test)
    print(f"  Legendre多项式 P_0到P_8 在100点处求值完成")
    
    # 感受野Legendre基展开
    def rf_profile_1d(xi):
        return np.exp(-xi ** 2 / (2.0 * 0.15 ** 2)) - 0.5 * np.exp(-xi ** 2 / (2.0 * 0.40 ** 2))
    
    coeffs = decompose_rf_with_legendre_basis(rf_profile_1d, max_degree=12, n_quad=32)
    print(f"  Legendre基展开系数 (前6项): {coeffs[:6]}")
    
    # 重构验证
    rf_reconstructed = reconstruct_rf_from_legendre_coeffs(coeffs, x_test)
    rf_exact = np.array([rf_profile_1d(xi) for xi in x_test])
    recon_error = np.max(np.abs(rf_reconstructed - rf_exact))
    print(f"  重构最大误差: {recon_error:.2e}")
    
    # 双极细胞对光栅刺激的响应
    grating = sinusoidal_grating(32, 32, spatial_freq=2.0, orientation=np.pi / 4.0)
    rf_params = {'A_c': 1.0, 'sigma_c': 0.15, 'A_s': 0.5, 'sigma_s': 0.40}
    response = compute_bipolar_response_convolution(grating, rf_params, grid_spacing=0.1)
    print(f"  双极细胞对正弦光栅响应: {response:.6f}")
    
    return rf, coeffs, response


def run_synaptic_diffusion():
    """运行突触反应-扩散模型。"""
    section_header("4. 突触间隙神经递质反应-扩散")
    
    # Gray-Scott模拟（基于486_gray_scott_movie）
    result = simulate_synaptic_transmission(
        nx=64, ny=64, n_steps=5000,
        Du=0.16, Dv=0.08, F=0.035, K=0.060,
        dt=0.5, dx=0.5, dy=0.5,
        initial_condition='localized',
        boundary='periodic'
    )
    
    print(f"  Gray-Scott模拟完成: {result['n_steps']}步")
    print(f"  最终U范围: [{result['final_U'].min():.4f}, {result['final_U'].max():.4f}]")
    print(f"  最终V范围: [{result['final_V'].min():.4f}, {result['final_V'].max():.4f}]")
    
    # 突触传递效能
    efficacy = compute_synaptic_efficacy(result['final_V'], threshold=0.1)
    print(f"  突触传递效能:")
    print(f"    峰值浓度: {efficacy['peak_concentration']:.4f}")
    print(f"    平均浓度: {efficacy['mean_concentration']:.4f}")
    print(f"    激活区域占比: {efficacy['active_fraction']:.4f}")
    print(f"    总效能: {efficacy['total_efficacy']:.4f}")
    
    return result, efficacy


def run_neural_encoding():
    """运行神经编码分析。"""
    section_header("5. 神经节细胞脉冲发放模式分析")
    
    # 三角插值基函数（基于1357_trig_interp_basis）
    x_test = np.linspace(-0.9, 0.9, 200)
    k = 7  # 奇数阶
    basis_vals = trig_interp_basis(x_test, k)
    print(f"  三角插值基 τ_{k}(x) 在200点处求值完成")
    print(f"  基函数范围: [{basis_vals.min():.4f}, {basis_vals.max():.4f}]")
    
    # 三角插值重构脉冲模式
    spike_times = np.array([0.0, 0.25, 0.5, 0.75])
    spike_values = np.array([1.0, 0.8, 1.2, 0.9])
    t_eval = np.linspace(0.0, 1.0, 100)
    interp_pattern = trig_interpolate_spike_pattern(spike_times, spike_values, t_eval, period=1.0)
    print(f"  三角插值重构脉冲模式完成")
    
    # Horner多项式求值（基于200_collocation）
    coeffs = np.array([0.5, 1.2, -0.3, 0.1, -0.02])
    x_horner = np.linspace(0.0, 1.0, 50)
    poly_vals = horner_polynomial_eval(coeffs, x_horner)
    print(f"  Horner多项式求值完成 (次数={len(coeffs)-1})")
    
    # 发放率函数
    rate_func = lambda t: 20.0 + 30.0 * np.sin(2.0 * np.pi * 5.0 * t) ** 2
    
    # 生成非齐次泊松脉冲（基于320_duel_simulation的蒙特卡洛方法）
    spike_times_mc = generate_inhomogeneous_poisson_spikes(
        rate_func, 0.0, 1.0, dt=0.0005, max_rate=60.0, seed=42
    )
    print(f"  非齐次泊松脉冲生成: {len(spike_times_mc)} 个脉冲")
    
    # 脉冲序列分析
    stats = analyze_spike_train(spike_times_mc, 1.0, n_bins=50)
    print(f"  脉冲统计: 平均发放率={stats['mean_rate']:.2f} Hz, CV_ISI={stats['cv_isi']:.4f}")
    
    # 编码效率
    dt_bin = 0.001
    t_bins = np.arange(0.0, 1.0, dt_bin)
    stimulus = np.array([rate_func(t) for t in t_bins])
    spike_binary = np.zeros_like(t_bins)
    for st in spike_times_mc:
        idx = int(st / dt_bin)
        if 0 <= idx < len(spike_binary):
            spike_binary[idx] = 1.0
    
    eff = encoding_efficiency(spike_binary, stimulus, dt_bin)
    print(f"  编码效率: 信息率={eff['info_rate_bits']:.4f} bits, 相关={eff['correlation']:.4f}")
    
    return spike_times_mc, stats, eff


def run_network_solver():
    """运行神经网络求解器。"""
    section_header("6. 大规模视网膜神经网络线性系统求解")
    
    # 构建CBB矩阵（基于974_r8cbb）
    n_local = 50
    n_long_range = 10
    A1_band, A2, A3, A4 = build_retinal_network_matrix(
        n_local, n_long_range, connectivity_radius=3.0
    )
    
    print(f"  CBB矩阵构建完成: N1={n_local}, N2={n_long_range}")
    print(f"  A1带状矩阵形状: {A1_band.shape}")
    print(f"  A2形状: {A2.shape}, A3形状: {A3.shape}, A4形状: {A4.shape}")
    
    # 构造右端向量
    b = np.random.random(n_local + n_long_range)
    
    # Schur补求解
    x = solve_cbb_system(A1_band, A2, A3, A4, b, n_local, n_long_range, ml=3, mu=3)
    print(f"  CBB系统求解完成, 解范数: {np.linalg.norm(x):.4f}")
    
    # 验证残差
    # 重构完整矩阵验证
    A1_full = np.zeros((n_local, n_local))
    for j in range(n_local):
        for i in range(max(0, j - 3), min(n_local, j + 4)):
            row_in_band = 3 + (i - j)
            if 0 <= row_in_band < A1_band.shape[0]:
                A1_full[i, j] = A1_band[row_in_band, j]
    
    A_full = np.block([[A1_full, A2], [A3, A4]])
    residual = np.linalg.norm(A_full @ x - b)
    print(f"  残差范数: {residual:.2e}")
    
    # 条件数估计
    cond_est = matrix_condition_number_estimate(A_full)
    print(f"  矩阵条件数估计: {cond_est:.2e}")
    
    return x, residual, cond_est


def run_stimulus_exploration():
    """运行刺激生成与参数探索。"""
    section_header("7. 视觉刺激生成与高维参数采样")
    
    # 正弦光栅（基于597_iplot的函数表达式概念）
    grating = sinusoidal_grating(32, 32, spatial_freq=3.0, orientation=np.pi / 6.0)
    print(f"  正弦光栅生成: 空间频率=3.0 cpd, 朝向=30°")
    print(f"  光栅强度范围: [{grating.min():.4f}, {grating.max():.4f}]")
    
    # 高斯blob
    blob = gaussian_blob(32, 32, sigma_x=0.2, sigma_y=0.3)
    print(f"  高斯blob生成: σ_x=0.2, σ_y=0.3")
    
    # 漂移光栅
    drift = drifting_grating(16, 16, n_frames=20, spatial_freq=2.0,
                             temporal_freq=4.0, orientation=0.0, dt=0.01)
    print(f"  漂移光栅生成: {drift.shape[0]}帧, 时间频率=4.0 Hz")
    
    # 突触组合探索（基于1273_toms515）
    combos = explore_synaptic_combinations(n_synapses=20, subset_size=5, n_combinations=10)
    print(f"  突触子集组合探索: C(20,5)中选取10个组合")
    print(f"  第一个组合: {combos[0]}")
    
    # Faure准随机参数采样（基于349_faure）
    param_ranges = {
        'spatial_freq': (0.5, 10.0),
        'contrast': (0.1, 1.0),
        'orientation': (0.0, np.pi),
        'temporal_freq': (1.0, 20.0),
    }
    faure_samples = sample_neural_parameter_space(n_samples=20, param_ranges=param_ranges, skip=50)
    print(f"  Faure准随机参数采样: 4维参数空间, 20个样本")
    print(f"  样本空间频率范围: [{faure_samples['spatial_freq'].min():.2f}, {faure_samples['spatial_freq'].max():.2f}]")
    
    return grating, drift, combos, faure_samples


def run_math_verification():
    """运行数学工具验证。"""
    section_header("8. 数学工具与数值验证")
    
    # Simpson积分验证
    def f_test(x):
        return np.sin(np.pi * x) ** 2
    
    exact = 0.5  # ∫_0^1 sin²(πx) dx = 0.5
    approx = simpson_integration(f_test, 0.0, 1.0, n=1000)
    err = relative_error(approx, exact)
    print(f"  Simpson积分验证: ∫sin²(πx)dx ≈ {approx:.8f}, 精确值=0.5, 相对误差={err:.2e}")
    
    # 梯形积分
    x_trap = np.linspace(0.0, 1.0, 1000)
    y_trap = np.sin(np.pi * x_trap) ** 2
    approx_trap = trapezoidal_integration(x_trap, y_trap)
    err_trap = relative_error(approx_trap, exact)
    print(f"  梯形积分验证: 相对误差={err_trap:.2e}")
    
    # 球谐函数
    Y_10 = spherical_harmonic_y(1, 0, np.pi / 2.0, 0.0)
    print(f"  球谐函数 Y_1^0(π/2, 0) = {Y_10.real:.6f} (理论值≈0.4886)")
    
    # Lanczos log-gamma
    lg_5 = log_gamma_lanczos(5.0)
    print(f"  log-Gamma(5) = {lg_5:.6f} (理论值=ln(24)≈3.1781)")
    
    # 误差函数
    erf_1 = erf_approx(1.0)
    print(f"  erf(1.0) ≈ {erf_1:.6f} (理论值≈0.8427)")
    
    return approx, Y_10, lg_5, erf_1


def main():
    """主函数：执行完整的视网膜神经编码多尺度模拟流程。"""
    print("\n" + "#" * 70)
    print("#  视网膜神经编码与视觉通路的多尺度计算模型")
    print("#  Multi-scale Computational Model of Retinal Neural Encoding")
    print("#" * 70)
    print(f"\n  启动时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    t_start = time.time()
    
    # 1. 视网膜几何
    photoreceptors, triangles, quality = run_retina_geometry()
    
    # 2. 光感受器模型
    C_steady, photocurrent, ode_result = run_photoreceptor_model()
    
    # 3. 双极细胞模型
    rf, rf_coeffs, bipolar_response = run_bipolar_cell()
    
    # 4. 突触扩散
    syn_result, efficacy = run_synaptic_diffusion()
    
    # 5. 神经编码
    spike_times, spike_stats, encoding_eff = run_neural_encoding()
    
    # 6. 网络求解
    network_x, residual, cond_est = run_network_solver()
    
    # 7. 刺激探索
    grating, drift, combos, faure_samples = run_stimulus_exploration()
    
    # 8. 数学验证
    math_results = run_math_verification()
    
    t_elapsed = time.time() - t_start
    
    # 总结
    section_header("模拟完成总结")
    print(f"  总运行时间: {t_elapsed:.3f} 秒")
    print(f"  模块执行数: 8/8")
    print(f"  感光细胞数: {photoreceptors.shape[0]}")
    print(f"  三角单元数: {triangles.shape[0]}")
    print(f"  网格ALPHA质量: {quality['alpha_ave']:.4f}")
    print(f"  光电流积分: {photocurrent:.4f}")
    print(f"  突触峰值浓度: {efficacy['peak_concentration']:.4f}")
    print(f"  生成脉冲数: {len(spike_times)}")
    print(f"  脉冲平均发放率: {spike_stats['mean_rate']:.2f} Hz")
    print(f"  编码信息率: {encoding_eff['info_rate_bits']:.4f} bits")
    print(f"  网络残差: {residual:.2e}")
    print(f"  矩阵条件数: {cond_est:.2e}")
    print(f"\n  所有计算模块运行完毕，无报错。")
    print("=" * 70)


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（58个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: simpson_integration analytic validation ∫_0^1 sin²(πx) dx = 0.5 ----
def f_sin2(x):
    return np.sin(np.pi * x) ** 2
result_simpson = simpson_integration(f_sin2, 0.0, 1.0, n=1000)
assert abs(result_simpson - 0.5) < 1e-6, '[TC01] simpson_integration sin²(πx) FAILED'

# ---- TC02: trapezoidal_integration returns finite non-negative for non-negative y ----
x_trap = np.linspace(0.0, 1.0, 100)
y_trap = np.sin(np.pi * x_trap) ** 2
result_trap = trapezoidal_integration(x_trap, y_trap)
assert np.isfinite(result_trap), '[TC02] trapezoidal_integration not finite FAILED'
assert result_trap >= 0.0, '[TC02] trapezoidal_integration negative FAILED'

# ---- TC03: relative_error exact match returns zero ----
assert relative_error(1.5, 1.5) == 0.0, '[TC03] relative_error exact match FAILED'

# ---- TC04: erf_approx(0) ≈ 0 ----
assert abs(erf_approx(0.0)) < 1e-6, '[TC04] erf_approx(0) FAILED'

# ---- TC05: erf_approx(1.0) ≈ 0.8427 ----
erf1 = erf_approx(1.0)
assert abs(erf1 - 0.8427007929) < 0.005, '[TC05] erf_approx(1) FAILED'

# ---- TC06: log_gamma_lanczos(5) ≈ ln(24) ----
lg5 = log_gamma_lanczos(5.0)
assert abs(lg5 - np.log(24.0)) < 1e-6, '[TC06] log_gamma_lanczos(5) FAILED'

# ---- TC07: spherical_harmonic_y Y_0^0 = 1/√(4π) ----
Y00 = spherical_harmonic_y(0, 0, 0.0, 0.0)
expected_Y00 = 1.0 / np.sqrt(4.0 * np.pi)
assert abs(Y00.real - expected_Y00) < 1e-10, '[TC07] spherical_harmonic Y_0^0 FAILED'
assert abs(Y00.imag) < 1e-10, '[TC07] spherical_harmonic imag part FAILED'

# ---- TC08: legendre_polynomial_value P_0(x)=1 for all x ----
x_test = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
V_leg = legendre_polynomial_value(0, x_test)
assert np.allclose(V_leg[:, 0], 1.0), '[TC08] legendre P_0 FAILED'

# ---- TC09: legendre_polynomial_value P_1(x)=x ----
V_leg1 = legendre_polynomial_value(1, x_test)
assert np.allclose(V_leg1[:, 1], x_test), '[TC09] legendre P_1 identity FAILED'

# ---- TC10: dog_receptive_field returns ndarray of correct shape ----
x_rf, y_rf = np.linspace(-1, 1, 32), np.linspace(-1, 1, 32)
X_rf, Y_rf = np.meshgrid(x_rf, y_rf)
rf = dog_receptive_field(X_rf, Y_rf, A_c=1.0, sigma_c=0.15, A_s=0.5, sigma_s=0.40)
assert isinstance(rf, np.ndarray), '[TC10] dog_receptive_field type FAILED'
assert rf.shape == (32, 32), '[TC10] dog_receptive_field shape FAILED'

# ---- TC11: dog_receptive_field center value = A_c - A_s ----
rf_center = dog_receptive_field(np.array([[0.0]]), np.array([[0.0]]), A_c=1.0, sigma_c=0.15, A_s=0.5, sigma_s=0.40)
assert abs(rf_center[0, 0] - 0.5) < 1e-10, '[TC11] dog_receptive_field center FAILED'

# ---- TC12: gauss_legendre_quadrature weights sum to 2 ----
x_gl, w_gl = gauss_legendre_quadrature(16)
assert abs(np.sum(w_gl) - 2.0) < 1e-10, '[TC12] gauss_legendre weights sum FAILED'

# ---- TC13: generate_hexagonal_photoreceptor_array correct count ----
pts_hex = generate_hexagonal_photoreceptor_array(1.0, 2)
assert pts_hex.shape[0] == 1 + 6 + 12, '[TC13] hex array count FAILED'
assert pts_hex.shape[1] == 2, '[TC13] hex array shape FAILED'

# ---- TC14: hexagon_moment_integral symmetry M(2,0) = M(0,2) ----
hex_verts = np.array([
    [1.0, 0.0], [0.5, np.sqrt(3.0)/2.0], [-0.5, np.sqrt(3.0)/2.0],
    [-1.0, 0.0], [-0.5, -np.sqrt(3.0)/2.0], [0.5, -np.sqrt(3.0)/2.0]
])
M20 = hexagon_moment_integral(2, 0, hex_verts)
M02 = hexagon_moment_integral(0, 2, hex_verts)
assert abs(M20 - M02) < 1e-10, '[TC14] hexagon moment symmetry FAILED'

# ---- TC15: convex_hull_2d returns non-empty result ----
hull = convex_hull_2d(pts_hex)
assert len(hull) >= 3, '[TC15] convex_hull_2d result FAILED'

# ---- TC16: evaluate_mesh_quality returns valid quality metrics ----
triangles = delaunay_triangulation_2d(pts_hex)
quality = evaluate_mesh_quality(pts_hex, triangles)
assert isinstance(quality, dict), '[TC16] mesh quality type FAILED'
assert 0.0 <= quality['alpha_ave'] <= 1.0 + 1e-10, '[TC16] mesh quality alpha range FAILED'
assert quality['num_triangles'] > 0, '[TC16] mesh quality num_triangles FAILED'

# ---- TC17: horner_polynomial_eval constant polynomial ----
p_const = horner_polynomial_eval(np.array([3.0]), np.array([0.0, 1.0, -5.0]))
assert np.allclose(p_const, 3.0), '[TC17] horner constant FAILED'

# ---- TC18: trig_interp_basis τ_k(0) = 1 ----
tau_zero = trig_interp_basis(np.array([0.0]), 7)
assert abs(tau_zero[0] - 1.0) < 1e-12, '[TC18] trig_interp_basis at origin FAILED'

# ---- TC19: encoding_efficiency zero spike train gives zero info ----
spikes_zero = np.zeros(1000)
stim_const = np.ones(1000)
eff_zero = encoding_efficiency(spikes_zero, stim_const, 0.001)
assert eff_zero['info_rate_bits'] == 0.0, '[TC19] encoding_efficiency zero FAILED'

# ---- TC20: faure_generate output in [0,1] range ----
faure_pts = faure_generate(3, 20, skip=50)
assert np.all(faure_pts >= 0.0) and np.all(faure_pts <= 1.0), '[TC20] faure_generate range FAILED'
assert faure_pts.shape == (3, 20), '[TC20] faure_generate shape FAILED'

# ---- TC21: generate_inhomogeneous_poisson_spikes seed reproducibility ----
rate_fn = lambda t: 30.0 + 20.0 * np.sin(2.0 * np.pi * 5.0 * t)
spikes1 = generate_inhomogeneous_poisson_spikes(rate_fn, 0.0, 1.0, dt=0.001, max_rate=55.0, seed=42)
spikes2 = generate_inhomogeneous_poisson_spikes(rate_fn, 0.0, 1.0, dt=0.001, max_rate=55.0, seed=42)
assert np.array_equal(spikes1, spikes2), '[TC21] poisson spikes reproducibility FAILED'

# ---- TC22: band_lu_factorize and band_lu_solve on identity-like diagonal matrix ----
ml_test, mu_test, n_test = 2, 2, 10
A_test = np.zeros((ml_test + mu_test + 1, n_test))
A_test[mu_test, :] = 1.0
A_lu, info = band_lu_factorize(A_test.copy(), n_test, ml_test, mu_test)
assert info == 0, '[TC22] band_lu_factorize info FAILED'
b_test = np.ones(n_test)
x_test2 = band_lu_solve(A_lu, b_test, n_test, ml_test, mu_test)
assert np.allclose(x_test2, 1.0), '[TC22] band_lu_solve identity FAILED'

# ---- TC23: build_retinal_network_matrix correct shapes ----
A1b, A2m, A3m, A4m = build_retinal_network_matrix(20, 5, connectivity_radius=2.0)
assert A1b.shape == (5, 20), '[TC23] A1_band shape FAILED'
assert A2m.shape == (20, 5), '[TC23] A2 shape FAILED'
assert A3m.shape == (5, 20), '[TC23] A3 shape FAILED'
assert A4m.shape == (5, 5), '[TC23] A4 shape FAILED'

# ---- TC24: sinusoidal_grating output in [0,1] range ----
grating_test = sinusoidal_grating(32, 32, spatial_freq=3.0, orientation=np.pi/4.0)
assert grating_test.shape == (32, 32), '[TC24] sinusoidal_grating shape FAILED'
assert np.all(grating_test >= 0.0) and np.all(grating_test <= 1.0), '[TC24] sinusoidal_grating range FAILED'

# ---- TC25: gaussian_blob center is maximum ----
blob_test = gaussian_blob(31, 31, sigma_x=0.3, sigma_y=0.3, center_x=0.0, center_y=0.0)
assert blob_test.shape == (31, 31), '[TC25] gaussian_blob shape FAILED'
assert abs(blob_test.max() - 1.0) < 1e-10, '[TC25] gaussian_blob center max FAILED'

# ---- TC26: white_noise_stimulus reproducibility ----
noise1 = white_noise_stimulus(16, 16, seed=42)
noise2 = white_noise_stimulus(16, 16, seed=42)
assert np.array_equal(noise1, noise2), '[TC26] white_noise reproducibility FAILED'

# ---- TC27: integrate_photocurrent_clenshaw_curtis constant function ----
def const_fn(x):
    return 3.0
integral_const = integrate_photocurrent_clenshaw_curtis(const_fn, -1.0, 1.0, n=32)
assert abs(integral_const - 6.0) < 1e-6, '[TC27] clenshaw_curtis constant FAILED'

# ---- TC28: compute_synaptic_efficacy on zero field returns zero total_efficacy ----
zero_v = np.zeros((8, 8))
eff_syn = compute_synaptic_efficacy(zero_v, threshold=0.1)
assert eff_syn['total_efficacy'] == 0.0, '[TC28] synaptic efficacy zero FAILED'
assert eff_syn['active_fraction'] == 0.0, '[TC28] synaptic active_fraction FAILED'

# ---- TC29: solve_light_adaptation_steady_state returns finite values and correct shape ----
src = np.zeros((8, 8))
C_steady, iters, err = solve_light_adaptation_steady_state(
    8, 8, 10.0, 10.0, 10.0, 10.0, src, epsilon=1e-8, max_iter=5000
)
assert C_steady.shape == (8, 8), '[TC29] light adaptation shape FAILED'
assert np.all(np.isfinite(C_steady)), '[TC29] light adaptation finite FAILED'
assert iters > 0, '[TC29] light adaptation iterations FAILED'

# ---- TC30: decompose_rf_with_legendre_basis returns expected number of coefficients ----
def rf_test(x):
    return np.exp(-x**2 / 0.1)
coeffs_rf = decompose_rf_with_legendre_basis(rf_test, max_degree=8, n_quad=32)
assert len(coeffs_rf) == 9, '[TC30] decompose_rf coeff count FAILED'
assert np.all(np.isfinite(coeffs_rf)), '[TC30] decompose_rf finite FAILED'

# ---- TC31: reconstruct_rf_from_legendre_coeffs returns finite values ----
x_rec = np.linspace(-1.0, 1.0, 50)
rf_rec = reconstruct_rf_from_legendre_coeffs(coeffs_rf, x_rec)
assert len(rf_rec) == 50, '[TC31] reconstruct_rf shape FAILED'
assert np.all(np.isfinite(rf_rec)), '[TC31] reconstruct_rf finite FAILED'

# ---- TC32: compute_bipolar_response_convolution returns scalar ----
grating_small = sinusoidal_grating(16, 16, spatial_freq=2.0, orientation=0.0)
rf_params_test = {'A_c': 1.0, 'sigma_c': 0.15, 'A_s': 0.5, 'sigma_s': 0.40}
response_bp = compute_bipolar_response_convolution(grating_small, rf_params_test, grid_spacing=0.1)
assert isinstance(response_bp, float), '[TC32] bipolar response type FAILED'
assert np.isfinite(response_bp), '[TC32] bipolar response finite FAILED'

# ---- TC33: spherical_harmonic Y_1^0(π/2, 0) = 0 (cos(π/2)=0) ----
Y10 = spherical_harmonic_y(1, 0, np.pi/2.0, 0.0)
assert abs(Y10.real) < 1e-10, '[TC33] spherical_harmonic Y_1^0 FAILED'

# ---- TC34: drifting_grating correct shape ----
drift_test = drifting_grating(8, 8, n_frames=10, spatial_freq=2.0, temporal_freq=4.0, orientation=0.0, dt=0.01)
assert drift_test.shape == (10, 8, 8), '[TC34] drifting_grating shape FAILED'

# ---- TC35: matrix_condition_number_estimate on identity returns near 1 ----
I_mat = np.eye(5)
cond_I = matrix_condition_number_estimate(I_mat)
assert 0.5 < cond_I < 2.0, '[TC35] condition number identity FAILED'

# ---- TC36: sample_neural_parameter_space returns dict with correct keys ----
param_ranges_test = {
    'freq': (0.5, 10.0),
    'contrast': (0.1, 1.0),
}
np.random.seed(42)
samples_test = sample_neural_parameter_space(n_samples=10, param_ranges=param_ranges_test, skip=50)
assert 'freq' in samples_test, '[TC36] param space key FAILED'
assert len(samples_test['freq']) == 10, '[TC36] param space length FAILED'

# ---- TC37: explore_synaptic_combinations correct shape ----
combos_test = explore_synaptic_combinations(n_synapses=10, subset_size=3, n_combinations=5)
assert combos_test.shape == (5, 3), '[TC37] explore combos shape FAILED'

# ---- TC38: evaluate_tuning_function returns finite array ----
tune_vals = evaluate_tuning_function('x^2', np.array([0.0, 1.0, 2.0]))
assert np.all(np.isfinite(tune_vals)), '[TC38] evaluate_tuning finite FAILED'

# ---- TC39: simpson_integration returns float scalar ----
result_s_type = simpson_integration(lambda x: x, 0.0, 1.0, n=100)
assert isinstance(result_s_type, float), '[TC39] simpson return type FAILED'

# ---- TC40: trig_interpolate_spike_pattern returns correct length ----
st_times = np.array([0.0, 0.5])
st_vals = np.array([1.0, 0.5])
t_eval_test = np.linspace(0.0, 2.0, 20)
interp_res = trig_interpolate_spike_pattern(st_times, st_vals, t_eval_test, period=2.0)
assert len(interp_res) == 20, '[TC40] trig_interpolate shape FAILED'
assert np.all(np.isfinite(interp_res)), '[TC40] trig_interpolate finite FAILED'

# ---- TC41: analyze_spike_train basic stats for simple spike train ----
simple_spikes = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
stats_test = analyze_spike_train(simple_spikes, 1.0, n_bins=10)
assert stats_test['n_spikes'] == 5, '[TC41] analyze_spike_train count FAILED'
assert stats_test['mean_rate'] == 5.0, '[TC41] analyze_spike_train rate FAILED'

# ---- TC42: simulate_rgc_spike_train seed reproducibility ----
np.random.seed(42)
bp_test = np.random.random(100)
t_test = np.linspace(0.0, 0.1, 100)
spikes_a = simulate_rgc_spike_train(bp_test, t_test, seed=42)
np.random.seed(42)
bp_test2 = np.random.random(100)
spikes_b = simulate_rgc_spike_train(bp_test2, t_test, seed=42)
assert np.array_equal(spikes_a, spikes_b), '[TC42] rgc spike reproducibility FAILED'

# ---- TC43: simulate_synaptic_transmission returns dict with expected keys ----
syn_result = simulate_synaptic_transmission(
    nx=16, ny=16, n_steps=200,
    Du=0.16, Dv=0.08, F=0.035, K=0.060,
    dt=0.5, dx=0.5, dy=0.5,
    initial_condition='localized', boundary='periodic'
)
assert 'final_U' in syn_result, '[TC43] synaptic dict U FAILED'
assert 'final_V' in syn_result, '[TC43] synaptic dict V FAILED'
assert 'n_steps' in syn_result, '[TC43] synaptic dict n_steps FAILED'

# ---- TC44: solve_phototransduction_rk4 integration returns finite values ----
def light_zero(t):
    return 0.0
y0_test = np.array([0.0, 0.5, 0.1])
params_test = {
    'alpha_pde': 2.0, 'beta_pde': 5.0, 'alpha_gc_max': 10.0,
    'K_gc': 0.1, 'n_gc': 4.0, 'gamma': 1.0, 'eta': 1.0,
    'g_max': 1.0, 'K_cGMP': 0.05, 'V_m': -40.0, 'E_Ca': 40.0,
}
t_arr, y_arr = solve_phototransduction_rk4(light_zero, y0_test, (0.0, 0.1), 0.001, params_test)
assert len(t_arr) > 0, '[TC44] rk4 timesteps FAILED'
assert np.all(np.isfinite(y_arr)), '[TC44] rk4 finite FAILED'
assert y_arr.shape[1] == 3, '[TC44] rk4 3 vars FAILED'

# ---- TC45: log_gamma_lanczos returns finite for positive input ----
lg_pos = log_gamma_lanczos(3.5)
assert np.isfinite(lg_pos), '[TC45] log_gamma finite FAILED'
assert lg_pos > 0.0, '[TC45] log_gamma positive FAILED'

# ---- TC46: spherical_harmonic returns complex type ----
Y_22 = spherical_harmonic_y(2, 1, np.pi/3.0, np.pi/4.0)
assert isinstance(Y_22, complex), '[TC46] spherical_harmonic type FAILED'

# ---- TC47: simpson_integration of x^2 from 0 to 1 = 1/3 ----
result_x2 = simpson_integration(lambda x: x**2, 0.0, 1.0, n=1000)
assert abs(result_x2 - 1.0/3.0) < 1e-8, '[TC47] simpson x^2 FAILED'

# ---- TC48: gauss_legendre_quadrature nodes are within [-1, 1] ----
x_gl32, w_gl32 = gauss_legendre_quadrature(32)
assert np.all(x_gl32 >= -1.0) and np.all(x_gl32 <= 1.0), '[TC48] gauss_legendre nodes range FAILED'
assert np.all(w_gl32 > 0.0), '[TC48] gauss_legendre weights positive FAILED'

# ---- TC49: horner_polynomial_eval quadratic x^2 ----
p_quad = horner_polynomial_eval(np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 2.0, 3.0]))
assert np.allclose(p_quad, np.array([0.0, 1.0, 4.0, 9.0])), '[TC49] horner quadratic FAILED'

# ---- TC50: build_retinal_network_matrix A4 is square ----
_, _, _, A4_test = build_retinal_network_matrix(15, 5, connectivity_radius=2.0)
assert A4_test.shape == (5, 5), '[TC50] A4 square FAILED'

# ---- TC51: solve_cbb_system returns vector of correct length ----
np.random.seed(123)
A1b_cbb, A2_cbb, A3_cbb, A4_cbb = build_retinal_network_matrix(10, 3, connectivity_radius=2.0)
b_cbb = np.random.random(13)
x_cbb = solve_cbb_system(A1b_cbb, A2_cbb, A3_cbb, A4_cbb, b_cbb, 10, 3, ml=2, mu=2)
assert len(x_cbb) == 13, '[TC51] solve_cbb length FAILED'
assert np.all(np.isfinite(x_cbb)), '[TC51] solve_cbb finite FAILED'

# ---- TC52: faure_generate different dimensions ----
faure_5d = faure_generate(5, 8, skip=20)
assert faure_5d.shape == (5, 8), '[TC52] faure 5d shape FAILED'
assert np.all(faure_5d >= 0.0) and np.all(faure_5d <= 1.0), '[TC52] faure 5d range FAILED'

# ---- TC53: white_noise_stimulus returns values in [0,1] ----
noise_range = white_noise_stimulus(20, 20, seed=123)
assert np.all(noise_range >= 0.0) and np.all(noise_range <= 1.0), '[TC53] white_noise range FAILED'

# ---- TC54: drifting_grating values in [0,1] ----
drift_range = drifting_grating(8, 8, n_frames=5, spatial_freq=2.0, temporal_freq=4.0, orientation=0.0, dt=0.01)
assert np.all(drift_range >= 0.0) and np.all(drift_range <= 1.0), '[TC54] drifting_grating range FAILED'

# ---- TC55: matrix_condition_number_estimate returns finite ----
A_rand = np.random.random((4, 4)) + 10.0 * np.eye(4)
cond_rand = matrix_condition_number_estimate(A_rand)
assert np.isfinite(cond_rand), '[TC55] condition number finite FAILED'
assert cond_rand > 0.0, '[TC55] condition number positive FAILED'

# ---- TC56: compute_synaptic_efficacy on non-zero field returns positive total_efficacy ----
v_nonzero = np.ones((8, 8)) * 0.5
eff_nonzero = compute_synaptic_efficacy(v_nonzero, threshold=0.1)
assert eff_nonzero['total_efficacy'] > 0.0, '[TC56] synaptic efficacy positive FAILED'
assert eff_nonzero['active_fraction'] > 0.0, '[TC56] synaptic active_fraction positive FAILED'
assert eff_nonzero['peak_concentration'] == 0.5, '[TC56] synaptic peak concentration FAILED'

# ---- TC57: simpson_integration of cos from 0 to π/2 = 1 ----
result_cos = simpson_integration(lambda x: np.cos(x), 0.0, np.pi/2.0, n=1000)
assert abs(result_cos - 1.0) < 1e-7, '[TC57] simpson cos FAILED'

# ---- TC58: erf_approx is odd function: erf(-x) = -erf(x) ----
erf_neg1 = erf_approx(-1.0)
erf_pos1 = erf_approx(1.0)
assert abs(erf_neg1 + erf_pos1) < 1e-6, '[TC58] erf odd symmetry FAILED'

print('\n全部 58 个测试通过!\n')
