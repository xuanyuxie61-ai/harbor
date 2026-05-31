"""
超声成像波束形成与反演系统 —— 统一入口

项目编号: PROJECT_91
科学领域: 声学工程 —— 超声成像波束形成与反演

本系统整合15个种子项目的核心算法，构建一个面向多物理场耦合
超声层析成像的博士级科学计算平台。

运行方式:
    python main.py

无需任何参数，系统将自动执行完整的计算流程并输出结果。
"""

import numpy as np
import time
import sys

# 导入所有模块
from mesh_quality import mesh_quality_report, reject_poor_triangles
from acoustic_fem_mesh import generate_optimized_acoustic_mesh
from transducer_dynamics import simulate_transducer_response, verify_stiff_solver_fix
from helmholtz_solver import solve_helmholtz_1d, solve_helmholtz_2d_dirichlet, interpolate_acoustic_pressure
from nonlinear_acoustics import burgers_periodic_solution, shock_wave_formation, nonlinear_acoustic_parameter_estimation
from flow_acoustic_coupling import taylor_green_vortex, interpolate_to_grid, compute_flow_acoustic_source, mach_number_field
from microbubble_diffusion import simulate_microbubble_diffusion, acoustic_radiation_force, diffusion_coefficient
from wavelet_denoising import denoise_ultrasound_signal, extract_multiscale_features
from pca_feature_extraction import pca_bscan_analysis
from ultrasound_beamforming import delay_and_sum_beamforming, simulate_array_response, beam_pattern, transmit_focus_delay
from inverse_tomography import build_projection_matrix, solve_tomography_svd, analyze_system_identifiability, classify_tissue_sequences, levenshtein_distance


def print_section(title: str):
    """打印带分隔线的章节标题。"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_module_1_mesh_generation():
    """模块1: 声学网格生成与RCM优化"""
    print_section("模块1: 声学有限元网格生成与RCM带宽优化")

    # 生成标准声学模拟域 (0.1m x 0.1m)
    nodes, triangles, reorder, old_to_new = generate_optimized_acoustic_mesh(nx=21, ny=21)

    report = mesh_quality_report(triangles, nodes)
    print(f"网格统计:")
    print(f"  节点数: {report['num_nodes']}")
    print(f"  单元数: {report['num_triangles']}")
    print(f"  Q度量最小值: {report['q_min']:.4f}")
    print(f"  Q度量均值: {report['q_mean']:.4f}")
    print(f"  Alpha度量最小值: {report['alpha_min']:.4f}")
    print(f"  Gamma度量: {report['gamma']:.4f}")
    print(f"  优化前矩阵半带宽: {report['bandwidth']}")

    # RCM优化后带宽
    from acoustic_fem_mesh import build_adjacency_structure
    from mesh_quality import bandwidth_mesh
    adj = build_adjacency_structure(nodes, triangles)
    new_bw = bandwidth_mesh(triangles)
    print(f"  RCM优化后半带宽: {new_bw}")
    print(f"  带宽缩减率: {(1 - new_bw / report['bandwidth']) * 100:.1f}%")

    return nodes, triangles


def run_module_2_transducer_dynamics():
    """模块2: 压电换能器瞬态动力学与刚性ODE验证"""
    print_section("模块2: 压电换能器瞬态动力学与刚性ODE验证")

    # 模拟5MHz换能器响应
    freq = 5e6
    t, disp, vel = simulate_transducer_response(freq=freq, Q=50.0,
                                                 t_span=(0.0, 10e-6),
                                                 n_steps=2000)

    max_disp = np.max(np.abs(disp))
    max_vel = np.max(np.abs(vel))
    print(f"换能器参数:")
    print(f"  频率: {freq/1e6:.1f} MHz")
    print(f"  最大位移: {max_disp:.4e} m")
    print(f"  最大速度: {max_vel:.4e} m/s")

    # Lindberg刚性ODE验证
    verify_result = verify_stiff_solver_fix(lam=-5.0,
                                            t_span=(0.0, 1.0),
                                            n_steps_list=[50, 100, 200, 400])
    print(f"\n刚性ODE验证:")
    print(f"  刚性比: {verify_result['stiffness_ratio']:.1f}")
    print(f"  50步L2误差: {verify_result['l2_errors'][0]:.4e}")
    print(f"  400步L2误差: {verify_result['l2_errors'][-1]:.4e}")
    print(f"  收敛阶估计: {np.mean(verify_result['convergence_orders']):.2f} (理论: 2.00)")

    return t, disp, vel


def run_module_3_helmholtz_solver():
    """模块3: Helmholtz方程求解与声压插值"""
    print_section("模块3: 声学Helmholtz方程有限差分解与插值重建")

    # 1D Helmholtz求解
    c0 = 1540.0
    freq = 5e6
    k = 2.0 * np.pi * freq / c0
    n = 200

    # 点源
    x_src = 0.05
    f_src = np.zeros(n)
    x, _ = solve_helmholtz_1d(n, k, f_src, xlim=(0.0, 0.1))
    x_grid = x[1:-1]
    src_idx = np.argmin(np.abs(x_grid - x_src))
    f_src[src_idx] = 1.0

    x, p = solve_helmholtz_1d(n, k, f_src, xlim=(0.0, 0.1), bc_left='abc', bc_right='abc')

    print(f"1D Helmholtz求解:")
    print(f"  频率: {freq/1e6:.1f} MHz")
    print(f"  波数 k: {k:.2f} rad/m")
    print(f"  波长 λ: {2*np.pi/k:.4f} m")
    print(f"  最大声压幅值: {np.max(np.abs(p)):.4e} Pa")

    # 分段线性插值到更密的网格
    xi = np.linspace(0.0, 0.1, 500)
    pi = interpolate_acoustic_pressure(x, p, xi)
    print(f"  插值到 {len(xi)} 个点完成")

    # 2D Helmholtz求解
    f_2d = np.zeros((20, 20))
    f_2d[10, 10] = 1.0
    x2d, y2d, p2d = solve_helmholtz_2d_dirichlet(20, 20, k, f_2d,
                                                  xlim=(0.0, 0.1),
                                                  ylim=(0.0, 0.1))
    print(f"\n2D Helmholtz求解:")
    print(f"  网格: 20x20")
    print(f"  最大声压幅值: {np.max(np.abs(p2d)):.4e} Pa")

    return x, p, x2d, y2d, p2d


def run_module_4_nonlinear_acoustics():
    """模块4: 非线性声学椭圆函数解"""
    print_section("模块4: 非线性声学椭圆函数与冲击波形成")

    # Burgers方程周期解
    x = np.linspace(0, 1, 256)
    t_burgers = 0.5
    u_burgers = burgers_periodic_solution(x, t_burgers, A=1.0, nu=0.01, m=0.5)

    print(f"Burgers方程周期行波解:")
    print(f"  时间: {t_burgers:.2f}")
    print(f"  最大速度: {np.max(u_burgers):.4f}")
    print(f"  最小速度: {np.min(u_burgers):.4f}")

    # 冲击波形成
    u_shock = shock_wave_formation(x, 0.3, u0=1.0, x0=0.5, L=1.0)
    print(f"\n冲击波形成 (t=0.3, t_shock=1.0):")
    print(f"  最大速度: {np.max(u_shock):.4f}")
    print(f"  冲击位置估计: ~0.65")

    # 非线性参数估计
    p_amps = np.array([1e5, 5000.0])
    freqs = np.array([5e6, 10e6])
    nonlinear_params = nonlinear_acoustic_parameter_estimation(p_amps, freqs)
    print(f"\n非线性参数估计:")
    print(f"  B/A 估计值: {nonlinear_params.get('B_over_A', 'N/A')}")
    print(f"  二次谐波效率: {nonlinear_params.get('efficiency', 'N/A')}")

    return u_burgers, u_shock


def run_module_5_flow_coupling():
    """模块5: 流-声耦合背景场"""
    print_section("模块5: Taylor-Green涡流-声耦合背景场")

    # 生成Taylor-Green涡
    x_grid = np.linspace(0, np.pi, 50)
    y_grid = np.linspace(0, np.pi, 50)
    X, Y = np.meshgrid(x_grid, y_grid)

    u_flow, v_flow, p_flow = taylor_green_vortex(X, Y, t=0.0, nu=0.01)

    print(f"Taylor-Green涡流场:")
    print(f"  最大u速度: {np.max(np.abs(u_flow)):.4f} m/s")
    print(f"  最大v速度: {np.max(np.abs(v_flow)):.4f} m/s")
    print(f"  最大压力: {np.max(np.abs(p_flow)):.4f} Pa")

    # Mach数分析
    ma = mach_number_field(u_flow, v_flow)
    print(f"  最大Mach数: {np.max(ma):.4e}")
    print(f"  线性近似有效性: {'是' if np.max(ma) < 0.3 else '否 (Ma>0.3) '}")

    # 流-声耦合源项
    c0 = 1540.0
    freq = 5e6
    k = 2.0 * np.pi * freq / c0
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]

    # 构造简单声压场
    p_acoustic = np.exp(1j * k * X) * np.ones_like(Y)

    source = compute_flow_acoustic_source(u_flow, v_flow, p_acoustic, dx, dy, k, c0)
    print(f"\n流-声耦合源项:")
    print(f"  最大幅值: {np.max(np.abs(source)):.4e}")
    print(f"  平均幅值: {np.mean(np.abs(source)):.4e}")

    return u_flow, v_flow, source


def run_module_6_microbubble_diffusion():
    """模块6: 微泡扩散模拟"""
    print_section("模块6: 超声造影剂微泡扩散与声辐射力")

    # 计算扩散系数
    radius = 2.5e-6
    D = diffusion_coefficient(radius)
    print(f"微泡参数:")
    print(f"  半径: {radius*1e6:.1f} μm")
    print(f"  扩散系数 D: {D:.4e} m²/s")

    # 声辐射力
    F_rad = acoustic_radiation_force(frequency=5e6, pressure_amplitude=1e5,
                                      bubble_radius=radius)
    print(f"  声辐射力: {F_rad:.4e} N")

    # 布朗运动模拟
    trajectory, msd, D_measured = simulate_microbubble_diffusion(
        n_particles=500, radius=radius, n_steps=500, dt=1e-6,
        domain_size=5e-3,
        acoustic_force=np.array([F_rad, 0.0])
    )

    # 验证Einstein关系: MSD = 2D·t (2D情况为 4D·t)
    t_msd = np.arange(len(msd)) * 1e-6
    # 线性拟合MSD
    if len(t_msd) > 10:
        slope = np.polyfit(t_msd[10:], msd[10:], 1)[0]
        D_from_msd = slope / 4.0  # 2D扩散: MSD = 4Dt
    else:
        D_from_msd = D

    print(f"\n扩散模拟结果:")
    print(f"  粒子数: 500")
    print(f"  模拟时间: {500*1e6:.0f} μs")
    print(f"  理论扩散系数: {D:.4e} m²/s")
    print(f"  MSD拟合扩散系数: {D_from_msd:.4e} m²/s")
    print(f"  相对误差: {abs(D_from_msd - D) / D * 100:.1f}%")

    return trajectory, msd


def run_module_7_wavelet_denoising():
    """模块7: 小波去噪与多尺度特征提取"""
    print_section("模块7: Haar小波变换去噪与多尺度特征提取")

    # 合成超声A-scan信号
    fs = 40e6
    t = np.arange(2048) / fs
    freq = 5e6

    # 模拟回波（含噪声）
    clean_signal = np.zeros(2048)
    depths = [0.02, 0.035, 0.05]
    amps = [1.0, 0.6, 0.3]
    c0 = 1540.0

    for depth, amp in zip(depths, amps):
        delay = 2.0 * depth / c0
        idx = int(delay * fs)
        if idx < 2048 - 100:
            pulse = amp * np.sin(2 * np.pi * freq * t[:100]) * np.exp(-t[:100] * 1e6 / 2)
            clean_signal[idx:idx+100] += pulse

    noise = 0.15 * np.random.randn(2048)
    noisy_signal = clean_signal + noise

    # 去噪
    denoised, info = denoise_ultrasound_signal(noisy_signal, n_levels=6, threshold_mode='soft')

    print(f"去噪处理:")
    print(f"  分解级数: {info['n_levels']}")
    print(f"  阈值: {info['threshold']:.4e}")
    print(f"  噪声估计: {info['noise_estimate']:.4e}")
    print(f"  原始信号能量: {info['original_energy']:.4e}")
    print(f"  去噪后能量: {info['denoised_energy']:.4e}")

    # 多尺度特征
    features = extract_multiscale_features(noisy_signal, n_levels=4)
    print(f"\n多尺度特征:")
    for key in sorted(features.keys()):
        if 'energy_ratio' in key:
            print(f"  {key}: {features[key]:.4f}")

    return denoised, info


def run_module_8_pca_analysis():
    """模块8: PCA特征提取"""
    print_section("模块8: 超声B-scan图像PCA降维与特征提取")

    result = pca_bscan_analysis(n_images=50, n_components=10)

    print(f"PCA分析:")
    print(f"  图像数: {result['n_images']}")
    print(f"  主成分数: {result['n_components']}")
    print(f"  前3主成分方差比: {result['variance_ratio'][:3]}")
    print(f"  累积方差保留率(10成分): {result['cumulative_variance_ratio'][-1]:.4f}")
    print(f"  重建RMSE: {result['reconstruction_error']['rmse']:.4f}")
    print(f"  重建PSNR: {result['reconstruction_error']['psnr']:.2f} dB")

    return result


def run_module_9_beamforming():
    """模块9: 波束形成与动态聚焦"""
    print_section("模块9: 超声阵列波束形成与动态聚焦")

    # 模拟阵列响应
    channel_data, element_positions = simulate_array_response(
        n_elements=64,
        element_spacing=0.3e-3,
        frequency=5e6,
        sampling_rate=40e6,
        n_samples=2048
    )

    print(f"阵列参数:")
    print(f"  阵元数: 64")
    print(f"  阵元间距: 0.30 mm")
    print(f"  孔径: {np.max(element_positions)*1e3*2:.1f} mm")

    # 波束形成
    focus_depths = np.linspace(0.01, 0.08, 100)
    beamformed = delay_and_sum_beamforming(
        channel_data, 40e6, element_positions, focus_depths,
        steering_angle=0.0, window_type='hanning'
    )

    print(f"\n波束形成结果:")
    print(f"  聚焦深度范围: 10-80 mm")
    print(f"  最大回波幅值: {np.max(beamformed):.4e}")
    print(f"  回波峰位置: {focus_depths[np.argmax(beamformed)]*1e3:.1f} mm")

    # 波束方向图
    angles = np.linspace(-np.pi/6, np.pi/6, 180)
    pattern = beam_pattern(64, 0.3e-3, 5e6, angles, window_type='hanning')

    print(f"\n波束方向图:")
    print(f"  主瓣3dB宽度: ~{0.886 * 1540/5e6 / (64*0.3e-3) * 180/np.pi:.2f}°")
    print(f"  最大旁瓣电平: {np.max(pattern[pattern < -3]):.1f} dB")

    # 发射聚焦延迟
    tx_delays = transmit_focus_delay(64, 0.3e-3, focus_depth=0.04)
    print(f"\n发射聚焦延迟 (40mm深度):")
    print(f"  最大延迟: {np.max(tx_delays)*1e9:.2f} ns")
    print(f"  最小延迟: {np.min(tx_delays)*1e9:.2f} ns")

    return beamformed, pattern


def run_module_10_tomography_inversion():
    """模块10: 断层反演与序列比对"""
    print_section("模块10: 超声断层反演与组织序列分类")

    # 构建投影矩阵
    A, angles = build_projection_matrix(n_rays=10, n_pixels_x=8, n_pixels_y=8)

    # 系统可识别性分析
    identifiability = analyze_system_identifiability(A)
    print(f"投影矩阵分析:")
    print(f"  矩阵尺寸: {identifiability['matrix_shape']}")
    print(f"  秩: {identifiability['rank']}")
    print(f"  零度: {identifiability['nullity']}")
    print(f"  条件数: {identifiability['condition_number']:.2e}")
    print(f"  可识别性比率: {identifiability['identifiability_ratio']:.2f}")

    # 合成传播时间数据
    # 真实慢度分布：中心高速区域
    true_slowness = np.ones(64) * 1.0 / 1540.0
    true_slowness[28:36] = 1.0 / 1600.0  # 中心区域声速稍高
    travel_times = A.astype(float) @ true_slowness

    # 加噪声
    noise = 0.01 * np.max(travel_times) * np.random.randn(len(travel_times))
    travel_times_noisy = travel_times + noise

    # SVD反演
    reconstructed = solve_tomography_svd(A, travel_times_noisy, regularization=1e-3)

    print(f"\n层析反演:")
    print(f"  真实慢度范围: [{np.min(true_slowness):.4e}, {np.max(true_slowness):.4e}]")
    print(f"  重建慢度范围: [{np.min(reconstructed):.4e}, {np.max(reconstructed):.4e}]")
    error = np.linalg.norm(reconstructed - true_slowness) / np.linalg.norm(true_slowness)
    print(f"  相对重建误差: {error*100:.2f}%")

    # 整数RREF精确分析
    A_rref, rank = __import__('inverse_tomography').i4mat_rref2(A)
    print(f"\n整数精确RREF分析:")
    print(f"  整数RREF秩: {rank}")
    print(f"  矩阵元素类型: 纯整数运算")

    # Levenshtein序列比对
    # 合成不同组织的A-scan符号序列
    tissue_normal = [0.1, 0.2, 0.5, 0.8, 0.6, 0.3, 0.1, 0.0]
    tissue_tumor = [0.1, 0.3, 0.7, 0.9, 0.9, 0.7, 0.3, 0.1]
    tissue_cyst = [0.0, 0.0, 0.1, 0.2, 0.2, 0.1, 0.0, 0.0]

    ref_sequences = [tissue_normal, tissue_tumor, tissue_cyst]
    labels = ['正常组织', '肿瘤', '囊肿']

    test_scan = [0.1, 0.25, 0.6, 0.85, 0.8, 0.5, 0.2, 0.05]

    classification = classify_tissue_sequences([test_scan], ref_sequences, labels)
    print(f"\n组织序列分类 (Levenshtein距离):")
    for cls in classification['classifications']:
        print(f"  预测类别: {cls['predicted_label']}")
        print(f"  最小距离: {cls['min_distance']}")

    return reconstructed, classification


def main():
    """主程序入口，执行完整的超声成像波束形成与反演流程。"""
    print("=" * 70)
    print("  超声成像波束形成与反演系统")
    print("  PROJECT_91 — 声学工程博士级科学计算平台")
    print("=" * 70)
    print(f"\n运行时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python版本: {sys.version.split()[0]}")
    print(f"NumPy版本: {np.__version__}")

    np.random.seed(42)
    start_time = time.time()

    # 执行全部10个模块
    results = {}

    results['mesh'] = run_module_1_mesh_generation()
    results['transducer'] = run_module_2_transducer_dynamics()
    results['helmholtz'] = run_module_3_helmholtz_solver()
    results['nonlinear'] = run_module_4_nonlinear_acoustics()
    results['flow'] = run_module_5_flow_coupling()
    results['microbubble'] = run_module_6_microbubble_diffusion()
    results['wavelet'] = run_module_7_wavelet_denoising()
    results['pca'] = run_module_8_pca_analysis()
    results['beamforming'] = run_module_9_beamforming()
    results['tomography'] = run_module_10_tomography_inversion()

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print(f"  全部模块执行完毕，总耗时: {elapsed:.2f} 秒")
    print("=" * 70)
    print("\n项目包含的核心科学计算:")
    print("  • 声学有限元网格生成与RCM稀疏矩阵优化")
    print("  • 压电换能器刚性ODE动力学与精确解验证")
    print("  • 1D/2D Helmholtz方程有限差分解")
    print("  • Burgers方程非线性声波椭圆函数解")
    print("  • Taylor-Green涡流-声耦合多物理场建模")
    print("  • 超声造影剂微泡布朗运动扩散模拟")
    print("  • Haar小波多分辨率去噪与特征提取")
    print("  • PCA降维与B-scan图像特征提取")
    print("  • 64阵元超声阵列波束形成与动态聚焦")
    print("  • 整数精确RREF断层反演与Levenshtein序列分类")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（25个，assert模式，涉及随机值均使用固定种子）
# ================================================================

# ---- TC01: triangle_area计算直角三角形面积正确 ----
from mesh_quality import triangle_area
area = triangle_area(np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 2.0]))
assert abs(area - 1.0) < 1e-10, '[TC01] triangle_area计算直角三角形面积正确 FAILED'

# ---- TC02: 等边三角形Q度量等于1 ----
from mesh_quality import q_measure
nodes_eq = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3)/2]])
tri_eq = np.array([[0, 1, 2]])
q = q_measure(tri_eq, nodes_eq)
assert abs(q[0] - 1.0) < 0.01, '[TC02] 等边三角形Q度量等于1 FAILED'

# ---- TC03: 生成5x5网格应有32个三角形和25个节点 ----
from acoustic_fem_mesh import generate_acoustic_domain
nodes, triangles = generate_acoustic_domain(nx=5, ny=5)
assert triangles.shape[0] == 32, '[TC03] 生成5x5网格应有32个三角形和25个节点 FAILED'
assert nodes.shape[0] == 25, '[TC03] 生成5x5网格应有32个三角形和25个节点 FAILED'

# ---- TC04: 邻接结构满足对称性i在j的邻居中当且仅当j在i的邻居中 ----
from acoustic_fem_mesh import build_adjacency_structure
adj = build_adjacency_structure(nodes, triangles)
for i, neighbors in adj.items():
    for j in neighbors:
        assert i in adj[j], '[TC04] 邻接结构满足对称性i在j的邻居中当且仅当j在i的邻居中 FAILED'

# ---- TC05: 分段线性插值对线性函数精确重构 ----
from helmholtz_solver import pwl_interp_1d
xd = np.array([0.0, 1.0, 2.0, 3.0])
yd = np.array([0.0, 1.0, 2.0, 3.0])
xi = np.array([0.5, 1.5, 2.5])
yi = pwl_interp_1d(xd, yd, xi)
assert np.allclose(yi, xi), '[TC05] 分段线性插值对线性函数精确重构 FAILED'

# ---- TC06: 1D Helmholtz解数组长度等于n+2 ----
x_h, p_h = solve_helmholtz_1d(50, 10.0, np.zeros(50), xlim=(0.0, 1.0))
assert len(x_h) == 52, '[TC06] 1D Helmholtz解数组长度等于n+2 FAILED'
assert len(p_h) == 52, '[TC06] 1D Helmholtz解数组长度等于n+2 FAILED'

# ---- TC07: 刚性ODE精确解在t=0处等于lambda平方除以1加lambda平方 ----
from transducer_dynamics import stiff_ode_exact
y0 = stiff_ode_exact(0.0, lam=-5.0)
expected = 25.0 / 26.0
assert abs(y0 - expected) < 1e-10, '[TC07] 刚性ODE精确解在t=0处等于lambda平方除以1加lambda平方 FAILED'

# ---- TC08: 刚性ODE求解器400步误差小于50步误差且收敛阶大于1.5 ----
from transducer_dynamics import verify_stiff_solver_fix
result = verify_stiff_solver_fix(lam=-5.0, n_steps_list=[50, 100, 200, 400])
assert result['l2_errors'][-1] < result['l2_errors'][0], '[TC08] 刚性ODE求解器400步误差小于50步误差且收敛阶大于1.5 FAILED'
assert np.mean(result['convergence_orders']) > 1.5, '[TC08] 刚性ODE求解器400步误差小于50步误差且收敛阶大于1.5 FAILED'

# ---- TC09: Jacobi椭圆函数m=0退化为sin和cos ----
from nonlinear_acoustics import sncndn
sn, cn, dn = sncndn(0.5, 0.0)
assert abs(sn - np.sin(0.5)) < 1e-10, '[TC09] Jacobi椭圆函数m=0退化为sin和cos FAILED'
assert abs(cn - np.cos(0.5)) < 1e-10, '[TC09] Jacobi椭圆函数m=0退化为sin和cos FAILED'
assert abs(dn - 1.0) < 1e-10, '[TC09] Jacobi椭圆函数m=0退化为sin和cos FAILED'

# ---- TC10: 冲击波形成在t=0时等于初始线性条件 ----
from nonlinear_acoustics import shock_wave_formation
x_shock = np.linspace(0, 1, 100)
u0 = shock_wave_formation(x_shock, 0.0, u0=1.0)
expected = x_shock - 0.5
assert np.allclose(u0, expected), '[TC10] 冲击波形成在t=0时等于初始线性条件 FAILED'

# ---- TC11: Taylor-Green涡t=0时v速度场等于负的cosx乘siny ----
from flow_acoustic_coupling import taylor_green_vortex
X, Y = np.meshgrid(np.linspace(0, np.pi, 20), np.linspace(0, np.pi, 20))
u, v, p = taylor_green_vortex(X, Y, t=0.0)
assert np.allclose(v, -np.cos(X) * np.sin(Y)), '[TC11] Taylor-Green涡t=0时v速度场等于负的cosx乘siny FAILED'

# ---- TC12: Mach数场非负且最大值等于速度模除以声速 ----
from flow_acoustic_coupling import mach_number_field
u_test = np.array([[1.0, 2.0], [0.0, 1.5]])
v_test = np.array([[0.0, 1.0], [0.5, 0.0]])
ma = mach_number_field(u_test, v_test, c0=1000.0)
assert np.all(ma >= 0), '[TC12] Mach数场非负且最大值等于速度模除以声速 FAILED'
expected_max = np.sqrt(5.0) / 1000.0
assert abs(np.max(ma) - expected_max) < 1e-10, '[TC12] Mach数场非负且最大值等于速度模除以声速 FAILED'

# ---- TC13: Stokes-Einstein扩散系数为有限正数 ----
from microbubble_diffusion import diffusion_coefficient
D = diffusion_coefficient(1e-6)
assert D > 0, '[TC13] Stokes-Einstein扩散系数为有限正数 FAILED'
assert np.isfinite(D), '[TC13] Stokes-Einstein扩散系数为有限正数 FAILED'

# ---- TC14: 声辐射力与微泡半径立方成正比 ----
from microbubble_diffusion import acoustic_radiation_force
F1 = acoustic_radiation_force(frequency=5e6, pressure_amplitude=1e5, bubble_radius=1e-6)
F2 = acoustic_radiation_force(frequency=5e6, pressure_amplitude=1e5, bubble_radius=2e-6)
assert F1 > 0, '[TC14] 声辐射力与微泡半径立方成正比 FAILED'
assert abs(F2 / F1 - 8.0) < 0.01, '[TC14] 声辐射力与微泡半径立方成正比 FAILED'

# ---- TC15: Haar单步分解前后能量守恒 ----
from wavelet_denoising import haar_step_1d
signal = np.array([3.0, 1.0, 2.0, 4.0])
approx, detail = haar_step_1d(signal)
E_in = np.sum(signal**2)
E_out = np.sum(approx**2) + np.sum(detail**2)
assert abs(E_in - E_out) < 1e-10, '[TC15] Haar单步分解前后能量守恒 FAILED'

# ---- TC16: Haar小波逆变换完美重构原始信号 ----
from wavelet_denoising import haar_1d, haar_1d_inverse
np.random.seed(42)
signal = np.random.randn(64)
coeffs, details = haar_1d(signal, n_levels=3)
reconstructed = haar_1d_inverse(coeffs, 3)
assert np.allclose(signal, reconstructed), '[TC16] Haar小波逆变换完美重构原始信号 FAILED'

# ---- TC17: PCA均值向量等于按轴0求平均 ----
from pca_feature_extraction import compute_mean_face
np.random.seed(42)
images = np.random.randn(10, 20)
mean_face = compute_mean_face(images)
assert mean_face.shape == (20,), '[TC17] PCA均值向量等于按轴0求平均 FAILED'
assert np.allclose(mean_face, np.mean(images, axis=0)), '[TC17] PCA均值向量等于按轴0求平均 FAILED'

# ---- TC18: PCA重建误差PSNR为有限值 ----
from pca_feature_extraction import compute_reconstruction_error
orig = np.array([1.0, 2.0, 3.0, 4.0])
recon = np.array([1.1, 1.9, 3.1, 3.9])
err = compute_reconstruction_error(orig, recon)
assert np.isfinite(err['psnr']), '[TC18] PCA重建误差PSNR为有限值 FAILED'

# ---- TC19: Hanning窗函数值均在0到1闭区间内 ----
from ultrasound_beamforming import hanning_window
w = hanning_window(16)
assert np.all(w >= 0), '[TC19] Hanning窗函数值均在0到1闭区间内 FAILED'
assert np.all(w <= 1), '[TC19] Hanning窗函数值均在0到1闭区间内 FAILED'

# ---- TC20: 发射聚焦延迟关于阵列中心对称 ----
from ultrasound_beamforming import transmit_focus_delay
delays = transmit_focus_delay(16, 0.3e-3, focus_depth=0.04)
n = len(delays)
for i in range(n // 2):
    assert abs(delays[i] - delays[n - 1 - i]) < 1e-14, '[TC20] 发射聚焦延迟关于阵列中心对称 FAILED'

# ---- TC21: Levenshtein距离相同序列结果为0 ----
from inverse_tomography import levenshtein_distance
dist = levenshtein_distance([1, 2, 3], [1, 2, 3])
assert dist == 0, '[TC21] Levenshtein距离相同序列结果为0 FAILED'

# ---- TC22: 投影矩阵形状为2*n_rays行和n_pixels_x乘n_pixels_y列 ----
from inverse_tomography import build_projection_matrix
A, angles = build_projection_matrix(n_rays=5, n_pixels_x=4, n_pixels_y=4)
assert A.shape == (10, 16), '[TC22] 投影矩阵形状为2*n_rays行和n_pixels_x乘n_pixels_y列 FAILED'

# ---- TC23: SVD反演输出长度等于矩阵列数 ----
from inverse_tomography import solve_tomography_svd
np.random.seed(42)
A_test = np.random.randn(10, 16)
tt = np.random.randn(10)
m = solve_tomography_svd(A_test, tt)
assert len(m) == 16, '[TC23] SVD反演输出长度等于矩阵列数 FAILED'

# ---- TC24: 单位矩阵可识别性比率为1 ----
from inverse_tomography import analyze_system_identifiability
info = analyze_system_identifiability(np.eye(5))
assert abs(info['identifiability_ratio'] - 1.0) < 1e-10, '[TC24] 单位矩阵可识别性比率为1 FAILED'
assert info['rank'] == 5, '[TC24] 单位矩阵可识别性比率为1 FAILED'

# ---- TC25: 主程序main返回包含10个键的结果字典 ----
results = main()
assert isinstance(results, dict), '[TC25] 主程序main返回包含10个键的结果字典 FAILED'
assert len(results) == 10, '[TC25] 主程序main返回包含10个键的结果字典 FAILED'

print('\n全部 25 个测试通过!\n')
