
import numpy as np
import time


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
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_retina_geometry():
    section_header("1. 视网膜几何离散化与网格质量评估")
    

    radius = 2.0
    n_rings = 6
    photoreceptors = generate_hexagonal_photoreceptor_array(radius, n_rings)
    print(f"  生成六边形感光细胞阵列：{photoreceptors.shape[0]} 个细胞")
    

    triangles = delaunay_triangulation_2d(photoreceptors)
    print(f"  Delaunay三角化：{triangles.shape[0]} 个三角形")
    

    quality = evaluate_mesh_quality(photoreceptors, triangles)
    print(f"  网格质量指标：")
    print(f"    ALPHA最小值/平均值: {quality['alpha_min']:.4f} / {quality['alpha_ave']:.4f}")
    print(f"    Q最小值/平均值: {quality['q_min']:.4f} / {quality['q_ave']:.4f}")
    print(f"    面积 最小/最大/平均: {quality['area_min']:.4f} / {quality['area_max']:.4f} / {quality['area_ave']:.4f}")
    print(f"    边界边数: {quality['num_boundary_edges']}")
    

    hull = convex_hull_2d(photoreceptors)
    print(f"  凸包顶点数: {len(hull)}")
    


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
    section_header("2. 光感受器光转导与光适应")
    

    nx, ny = 32, 32
    source = np.zeros((nx, ny), dtype=np.float64)

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
    

    def intensity_profile(x):
        return 100.0 * np.exp(-x ** 2 / 0.5)
    
    photocurrent = integrate_photocurrent_clenshaw_curtis(
        intensity_profile, -1.0, 1.0, n=64
    )
    print(f"  Clenshaw-Curtis光电流积分: {photocurrent:.6f}")
    







    raise NotImplementedError("Hole 3: 光转导ODE调用设置与结果提取待实现")
    
    return C_steady, photocurrent, (t_arr, y_arr)


def run_bipolar_cell():
    section_header("3. 双极细胞感受野与Legendre基展开")
    

    nx, ny = 64, 64
    x = np.linspace(-1.0, 1.0, nx)
    y = np.linspace(-1.0, 1.0, ny)
    X, Y = np.meshgrid(x, y)
    
    rf = dog_receptive_field(X, Y, A_c=1.0, sigma_c=0.15, A_s=0.5, sigma_s=0.40)
    print(f"  DoG感受野: 中心σ={0.15}, 周边σ={0.40}")
    print(f"  感受野范围: [{rf.min():.4f}, {rf.max():.4f}]")
    

    x_gl, w_gl = gauss_legendre_quadrature(16)
    print(f"  Gauss-Legendre 16点求积: 节点范围[{x_gl.min():.4f}, {x_gl.max():.4f}]")
    

    x_test = np.linspace(-1.0, 1.0, 100)
    V_leg = legendre_polynomial_value(8, x_test)
    print(f"  Legendre多项式 P_0到P_8 在100点处求值完成")
    

    def rf_profile_1d(xi):
        return np.exp(-xi ** 2 / (2.0 * 0.15 ** 2)) - 0.5 * np.exp(-xi ** 2 / (2.0 * 0.40 ** 2))
    
    coeffs = decompose_rf_with_legendre_basis(rf_profile_1d, max_degree=12, n_quad=32)
    print(f"  Legendre基展开系数 (前6项): {coeffs[:6]}")
    

    rf_reconstructed = reconstruct_rf_from_legendre_coeffs(coeffs, x_test)
    rf_exact = np.array([rf_profile_1d(xi) for xi in x_test])
    recon_error = np.max(np.abs(rf_reconstructed - rf_exact))
    print(f"  重构最大误差: {recon_error:.2e}")
    

    grating = sinusoidal_grating(32, 32, spatial_freq=2.0, orientation=np.pi / 4.0)
    rf_params = {'A_c': 1.0, 'sigma_c': 0.15, 'A_s': 0.5, 'sigma_s': 0.40}
    response = compute_bipolar_response_convolution(grating, rf_params, grid_spacing=0.1)
    print(f"  双极细胞对正弦光栅响应: {response:.6f}")
    
    return rf, coeffs, response


def run_synaptic_diffusion():
    section_header("4. 突触间隙神经递质反应-扩散")
    

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
    

    efficacy = compute_synaptic_efficacy(result['final_V'], threshold=0.1)
    print(f"  突触传递效能:")
    print(f"    峰值浓度: {efficacy['peak_concentration']:.4f}")
    print(f"    平均浓度: {efficacy['mean_concentration']:.4f}")
    print(f"    激活区域占比: {efficacy['active_fraction']:.4f}")
    print(f"    总效能: {efficacy['total_efficacy']:.4f}")
    
    return result, efficacy


def run_neural_encoding():
    section_header("5. 神经节细胞脉冲发放模式分析")
    

    x_test = np.linspace(-0.9, 0.9, 200)
    k = 7
    basis_vals = trig_interp_basis(x_test, k)
    print(f"  三角插值基 τ_{k}(x) 在200点处求值完成")
    print(f"  基函数范围: [{basis_vals.min():.4f}, {basis_vals.max():.4f}]")
    

    spike_times = np.array([0.0, 0.25, 0.5, 0.75])
    spike_values = np.array([1.0, 0.8, 1.2, 0.9])
    t_eval = np.linspace(0.0, 1.0, 100)
    interp_pattern = trig_interpolate_spike_pattern(spike_times, spike_values, t_eval, period=1.0)
    print(f"  三角插值重构脉冲模式完成")
    

    coeffs = np.array([0.5, 1.2, -0.3, 0.1, -0.02])
    x_horner = np.linspace(0.0, 1.0, 50)
    poly_vals = horner_polynomial_eval(coeffs, x_horner)
    print(f"  Horner多项式求值完成 (次数={len(coeffs)-1})")
    

    rate_func = lambda t: 20.0 + 30.0 * np.sin(2.0 * np.pi * 5.0 * t) ** 2
    

    spike_times_mc = generate_inhomogeneous_poisson_spikes(
        rate_func, 0.0, 1.0, dt=0.0005, max_rate=60.0, seed=42
    )
    print(f"  非齐次泊松脉冲生成: {len(spike_times_mc)} 个脉冲")
    

    stats = analyze_spike_train(spike_times_mc, 1.0, n_bins=50)
    print(f"  脉冲统计: 平均发放率={stats['mean_rate']:.2f} Hz, CV_ISI={stats['cv_isi']:.4f}")
    

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
    section_header("6. 大规模视网膜神经网络线性系统求解")
    

    n_local = 50
    n_long_range = 10
    A1_band, A2, A3, A4 = build_retinal_network_matrix(
        n_local, n_long_range, connectivity_radius=3.0
    )
    
    print(f"  CBB矩阵构建完成: N1={n_local}, N2={n_long_range}")
    print(f"  A1带状矩阵形状: {A1_band.shape}")
    print(f"  A2形状: {A2.shape}, A3形状: {A3.shape}, A4形状: {A4.shape}")
    

    b = np.random.random(n_local + n_long_range)
    

    x = solve_cbb_system(A1_band, A2, A3, A4, b, n_local, n_long_range, ml=3, mu=3)
    print(f"  CBB系统求解完成, 解范数: {np.linalg.norm(x):.4f}")
    


    A1_full = np.zeros((n_local, n_local))
    for j in range(n_local):
        for i in range(max(0, j - 3), min(n_local, j + 4)):
            row_in_band = 3 + (i - j)
            if 0 <= row_in_band < A1_band.shape[0]:
                A1_full[i, j] = A1_band[row_in_band, j]
    
    A_full = np.block([[A1_full, A2], [A3, A4]])
    residual = np.linalg.norm(A_full @ x - b)
    print(f"  残差范数: {residual:.2e}")
    

    cond_est = matrix_condition_number_estimate(A_full)
    print(f"  矩阵条件数估计: {cond_est:.2e}")
    
    return x, residual, cond_est


def run_stimulus_exploration():
    section_header("7. 视觉刺激生成与高维参数采样")
    

    grating = sinusoidal_grating(32, 32, spatial_freq=3.0, orientation=np.pi / 6.0)
    print(f"  正弦光栅生成: 空间频率=3.0 cpd, 朝向=30°")
    print(f"  光栅强度范围: [{grating.min():.4f}, {grating.max():.4f}]")
    

    blob = gaussian_blob(32, 32, sigma_x=0.2, sigma_y=0.3)
    print(f"  高斯blob生成: σ_x=0.2, σ_y=0.3")
    

    drift = drifting_grating(16, 16, n_frames=20, spatial_freq=2.0,
                             temporal_freq=4.0, orientation=0.0, dt=0.01)
    print(f"  漂移光栅生成: {drift.shape[0]}帧, 时间频率=4.0 Hz")
    

    combos = explore_synaptic_combinations(n_synapses=20, subset_size=5, n_combinations=10)
    print(f"  突触子集组合探索: C(20,5)中选取10个组合")
    print(f"  第一个组合: {combos[0]}")
    

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
    section_header("8. 数学工具与数值验证")
    

    def f_test(x):
        return np.sin(np.pi * x) ** 2
    
    exact = 0.5
    approx = simpson_integration(f_test, 0.0, 1.0, n=1000)
    err = relative_error(approx, exact)
    print(f"  Simpson积分验证: ∫sin²(πx)dx ≈ {approx:.8f}, 精确值=0.5, 相对误差={err:.2e}")
    

    x_trap = np.linspace(0.0, 1.0, 1000)
    y_trap = np.sin(np.pi * x_trap) ** 2
    approx_trap = trapezoidal_integration(x_trap, y_trap)
    err_trap = relative_error(approx_trap, exact)
    print(f"  梯形积分验证: 相对误差={err_trap:.2e}")
    

    Y_10 = spherical_harmonic_y(1, 0, np.pi / 2.0, 0.0)
    print(f"  球谐函数 Y_1^0(π/2, 0) = {Y_10.real:.6f} (理论值≈0.4886)")
    

    lg_5 = log_gamma_lanczos(5.0)
    print(f"  log-Gamma(5) = {lg_5:.6f} (理论值=ln(24)≈3.1781)")
    

    erf_1 = erf_approx(1.0)
    print(f"  erf(1.0) ≈ {erf_1:.6f} (理论值≈0.8427)")
    
    return approx, Y_10, lg_5, erf_1


def main():
    print("\n" + "#" * 70)
    print("#  视网膜神经编码与视觉通路的多尺度计算模型")
    print("#  Multi-scale Computational Model of Retinal Neural Encoding")
    print("#" * 70)
    print(f"\n  启动时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    t_start = time.time()
    

    photoreceptors, triangles, quality = run_retina_geometry()
    

    C_steady, photocurrent, ode_result = run_photoreceptor_model()
    

    rf, rf_coeffs, bipolar_response = run_bipolar_cell()
    

    syn_result, efficacy = run_synaptic_diffusion()
    

    spike_times, spike_stats, encoding_eff = run_neural_encoding()
    

    network_x, residual, cond_est = run_network_solver()
    

    grating, drift, combos, faure_samples = run_stimulus_exploration()
    

    math_results = run_math_verification()
    
    t_elapsed = time.time() - t_start
    

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
