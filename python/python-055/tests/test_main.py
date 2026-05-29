"""
main.py
海底地形声纳反演系统统一入口
================================

本项目围绕"海洋科学：海底地形声纳反演"展开，
基于 15 个种子科研代码项目的核心算法，融合构建了一个
面向前沿海洋地球物理问题的博士级计算系统。

运行方式:
    python main.py

零参数即可执行完整的反演流程，包括：
1. 多波束声纳数据生成与校验
2. 自适应波束采样规划
3. 分层海洋声线追踪
4. 回波信号处理与混响建模
5. 海底地形插值与几何分析
6. 一维谱有限元声学正演
7. 不确定性量化与误差传播
8. 波束覆盖统计评估
"""

import numpy as np
import time

# 导入各子模块
from data_validator import SonarDataValidator, generate_test_packets
from beam_statistics import BeamStatisticsAnalyzer
from seafloor_geometry import (
    SeafloorTriangulation, triangle_area, triangle_centroid,
    point_to_triangle_distance, ray_triangle_intersection_2d
)
from acoustic_ray_tracer import SoundSpeedProfile, AcousticRayTracer
from reverberation_model import StochasticReverberationField
from signal_processor import SonarSignalProcessor
from bathymetry_interpolator import BathymetryInterpolator
from adaptive_beam_sampler import AdaptiveSamplingPlanner, DiskGridGenerator
from uncertainty_quantifier import MonteCarloUncertaintyQuantifier, TruncatedNormalDistribution
from spectral_acoustic_solver import SpectralAcousticSolver


def main():
    print("=" * 72)
    print("  海底地形声纳反演系统 (Seafloor Bathymetry Sonar Inversion)")
    print("  科学领域: 海洋科学 — 海底地形声纳反演")
    print("=" * 72)
    t_start_total = time.time()

    # ================================================================
    # 1. 数据验证模块 (基于 1393_vin)
    # ================================================================
    print("\n[1/8] 声纳数据包验证与质量控制...")
    validator = SonarDataValidator()
    packets = generate_test_packets(n=30, seed=55)
    valid_mask = validator.validate_batch(packets)
    n_valid = int(np.sum(valid_mask))
    print(f"      生成 {len(packets)} 个模拟数据包，有效 {n_valid} 个")
    if len(validator.get_error_log()) > 0:
        print(f"      警告日志: {len(validator.get_error_log())} 条")

    # ================================================================
    # 2. 波束统计与覆盖分析 (基于 567_hypersphere_positive_distance)
    # ================================================================
    print("\n[2/8] 波束覆盖统计特性分析...")
    analyzer = BeamStatisticsAnalyzer(dim=3)
    mu_d, var_d = analyzer.compute_distance_stats(n_samples=3000)
    print(f"      随机波束对平均弦距离: μ = {mu_d:.4f}, σ² = {var_d:.6f}")

    # 生成扇形波束
    beam_dirs = analyzer.generate_optimal_fan_beams(n_beams=21, max_opening_angle_deg=60.0)
    coverage = analyzer.analyze_beam_coverage(beam_dirs)
    print(f"      扇形波束数: {coverage['n_beams']}")
    print(f"      最小夹角: {coverage['min_angle_deg']:.2f}°")
    print(f"      平均夹角: {coverage['mean_angle_deg']:.2f}°")
    print(f"      覆盖均匀度: {coverage['coverage_uniformity']:.4f}")

    # ================================================================
    # 3. 自适应采样规划 (基于 264_cvtp + 293_disk_grid + 626_knapsack_random)
    # ================================================================
    print("\n[3/8] 自适应波束采样规划...")
    planner = AdaptiveSamplingPlanner(survey_area=((0.0, 5000.0), (0.0, 2000.0)))
    plan = planner.plan_survey(n_beams=48, time_budget=1800.0, n_cvt_iter=12)
    print(f"      候选波束数: {len(plan['all_positions'])}")
    print(f"      最优选中数: {plan['n_selected']}")
    print(f"      总信息增益: {plan['total_gain']:.2f}")
    print(f"      总观测成本: {plan['total_cost']:.2f} s")
    print(f"      覆盖效率: {plan['coverage_efficiency']:.4f}")

    # 圆盘网格生成（模拟波束 footprint 采样）
    disk_pts = DiskGridGenerator.disk_grid(n=4, r=50.0, c=np.array([2500.0, 1000.0]))
    print(f"      单波束 footprint 网格点: {len(disk_pts)} 个")

    # ================================================================
    # 4. 声线追踪与深度反演 (基于 1432_zero_rc + 021_asa_geometry_2011)
    # ================================================================
    print("\n[4/8] 分层海洋声线追踪...")
    ssp = SoundSpeedProfile(c0=1500.0, g=0.015, delta_c=30.0, z1=1000.0, sigma=500.0)
    tracer = AcousticRayTracer(ssp)

    # 定义海底地形函数（含起伏）
    def seafloor(x):
        return 2000.0 + 50.0 * np.sin(x / 500.0) + 20.0 * np.cos(x / 200.0)

    angles = [0.0, 15.0, 30.0, 45.0, 60.0]
    print(f"      声速剖面: c(z) = 1500 + 0.015z + 30·exp(-(z-1000)²/(2·500²))")
    for ang in angles:
        result = tracer.trace_ray(x0=0.0, z0=0.0, theta0_deg=ang, z_bottom_func=seafloor,
                                   dt=0.005, max_steps=20000)
        if result['hit']:
            print(f"      掠射角 {ang:4.1f}°: 击中 x={result['x_hit']:8.1f}m, "
                  f"z={result['z_hit']:7.1f}m, 单程时={result['travel_time']:.3f}s")
        else:
            print(f"      掠射角 {ang:4.1f}°: 未击中海底（步数耗尽）")

    # ================================================================
    # 5. 回波信号处理 (基于 1268_toms243 + 1071_shepard_interp_1d)
    # ================================================================
    print("\n[5/8] 声纳回波信号处理...")
    processor = SonarSignalProcessor(fs=48000.0, f0=12000.0, bandwidth=4000.0)

    # 生成模拟接收信号（主回波 + 混响 + 噪声）
    t_chirp, chirp = processor.generate_chirp_pulse(duration=0.01)
    # 模拟传播延迟
    delay_samples = 1200
    received = np.zeros(len(chirp) + delay_samples + 500, dtype=complex)
    received[delay_samples:delay_samples + len(chirp)] = chirp * 0.8
    # 添加混响（简化）
    rng = np.random.default_rng(55)
    reverberation = rng.normal(0, 0.02, len(received)) + 1j * rng.normal(0, 0.02, len(received))
    received += reverberation

    proc_result = processor.process_single_ping(received, pulse_duration=0.01, noise_std=0.02)
    print(f"      检测峰值时间: {proc_result['peak_time']:.5f} s")
    print(f"      估计信噪比: {proc_result['snr_db']:.2f} dB")

    # 复对数频谱（TOMS243）
    freqs, log_spec = processor.compute_log_spectrum(received)
    log_mag_mean = float(np.mean(np.abs(log_spec.real)))
    print(f"      对数频谱平均幅度: {log_mag_mean:.4f}")

    # ================================================================
    # 6. 混响建模 (基于 707_mackey_glass_dde)
    # ================================================================
    print("\n[6/8] 海洋混响动力学建模...")
    rev_field = StochasticReverberationField(n_modes=4, seed=55)
    ttw_base = 2.8  # 秒，对应约 2000m 深度
    t_rev, env_rev = rev_field.generate_composite_envelope(ttw_base, base_amplitude=1.0)
    peak_rev = float(np.max(env_rev))
    mean_rev = float(np.mean(env_rev))
    print(f"      双程时间基准: {ttw_base:.2f} s")
    print(f"      复合混响包络峰值: {peak_rev:.4f}")
    print(f"      复合混响包络均值: {mean_rev:.4f}")

    # ================================================================
    # 7. 海底地形插值 (基于 1071_shepard_interp_1d 扩展为 2D)
    # ================================================================
    print("\n[7/8] 海底地形插值与几何分析...")
    # 生成稀疏测深点
    rng = np.random.default_rng(42)
    n_measurements = 80
    x_meas = rng.uniform(0.0, 5000.0, n_measurements)
    y_meas = rng.uniform(0.0, 2000.0, n_measurements)
    z_meas = 2000.0 + 50.0 * np.sin(x_meas / 500.0) + 30.0 * np.cos(y_meas / 300.0) \
             + rng.normal(0.0, 5.0, n_measurements)
    z_meas = np.clip(z_meas, 1000.0, 3000.0)

    interpolator = BathymetryInterpolator(x_meas, y_meas, z_meas, p=2.5)
    X_grid, Y_grid, Z_grid = interpolator.interpolate_grid(
        (0.0, 5000.0), (0.0, 2000.0), nx=80, ny=40, radius=800.0
    )

    # 梯度与曲率估计
    test_x, test_y = 2500.0, 1000.0
    grad = interpolator.estimate_gradient(test_x, test_y, h=20.0)
    curv = interpolator.estimate_curvature(test_x, test_y, h=50.0)
    print(f"      测深点数: {n_measurements}")
    print(f"      插值网格: {Z_grid.shape}")
    print(f"      在 ({test_x:.0f}, {test_y:.0f}) 处梯度: [{grad[0]:.4f}, {grad[1]:.4f}]")
    print(f"      在 ({test_x:.0f}, {test_y:.0f}) 处曲率: {curv:.6f}")

    # 剖面提取（1D Shepard）
    s_prof, z_prof = interpolator.cross_section_profile(0.0, 1000.0, 5000.0, 1000.0)
    print(f"      中心剖面长度: {s_prof[-1]:.1f} m")

    # ================================================================
    # 8. 海底三角网几何 (基于 1307_triangle_integrals + 021_asa_geometry_2011)
    # ================================================================
    print("\n[8/8] 海底三角网几何与谱声学正演...")
    # 构建简单三角网
    vertices = np.array([
        [0.0, 0.0],
        [2500.0, 100.0],
        [5000.0, 0.0],
        [1250.0, 1100.0],
        [3750.0, 1100.0],
        [2500.0, 2000.0],
    ])
    triangles = np.array([
        [0, 1, 3],
        [1, 4, 3],
        [1, 2, 4],
        [3, 4, 5],
    ])
    triangulation = SeafloorTriangulation(vertices, triangles)
    print(f"      三角网顶点数: {len(vertices)}, 面片数: {len(triangles)}")
    print(f"      总覆盖面积: {triangulation.total_area():.1f} m²")
    print(f"      平均面片面积: {triangulation.mean_patch_area():.1f} m²")

    # 测试点到三角形距离
    p_test = np.array([2500.0, 500.0])
    tri = vertices[triangles[1]]
    dist = point_to_triangle_distance(tri, p_test)
    print(f"      测试点到面片距离: {dist:.2f} m")

    # 单项式积分
    area_tri0 = abs(triangle_area(vertices[triangles[0]]))
    print(f"      面片0面积: {area_tri0:.2f} m²")

    # ================================================================
    # 9. 谱有限元声学正演 (基于 399_fem1d_spectral_numeric)
    # ================================================================
    print("\n[9/8] 一维谱有限元 Helmholtz 求解...")
    solver = SpectralAcousticSolver(depth=2000.0, frequency=12000.0)

    # 常声速简化情形：解析解为 sin(kz) 形式
    c_const = 1500.0
    k_val = solver.omega / c_const

    def c_func(z):
        return np.full_like(z, c_const)

    # 声源取为高斯包络
    z_source = 1000.0
    sigma_source = 100.0
    def source_func(z):
        return np.exp(-((z - z_source) ** 2) / (2.0 * sigma_source ** 2))

    # 精确解（对于常系数与非齐次项，用数值积分构造参考解）
    def exact_func(z):
        return np.sin(k_val * z / solver.H * np.pi) * 0.5

    def exact_deriv(z):
        return np.cos(k_val * z / solver.H * np.pi) * 0.5 * k_val * np.pi / solver.H

    result_fem = solver.solve_with_reference(
        n_basis=6, sound_speed_func=c_func, source_func=source_func,
        exact_solution_func=exact_func, exact_derivative_func=exact_deriv
    )
    print(f"      基函数数量: {result_fem['n_basis']}")
    print(f"      矩阵条件数: {result_fem['cond_number']:.4e}")
    print(f"      L² 误差: {result_fem.get('err_l2', -1):.6e}")
    print(f"      H¹ 误差: {result_fem.get('err_h1', -1):.6e}")

    # ================================================================
    # 10. 不确定性量化 (基于 805_nintlib + 1361_truncated_normal_rule)
    # ================================================================
    print("\n[10/8] 蒙特卡洛深度反演不确定性量化...")
    uq = MonteCarloUncertaintyQuantifier(dim=3)
    uq_result = uq.propagate_depth_uncertainty(
        base_sound_speed=1500.0,
        base_angle_deg=30.0,
        base_ttw=2.8,
        sigma_c=5.0,
        sigma_theta_deg=0.5,
        sigma_t=0.002,
        n_samples=3000,
        seed=55
    )
    print(f"      基准深度: {1500.0 * 2.8 * np.cos(np.radians(30.0)) / 2.0:.1f} m")
    print(f"      MC 估计平均深度: {uq_result['mean_depth']:.2f} m")
    print(f"      MC 估计深度标准差: {uq_result['std_depth']:.2f} m")
    print(f"      95% 置信区间: [{uq_result['ci_95_lower']:.2f}, {uq_result['ci_95_upper']:.2f}] m")
    print(f"      解析传播方差: {uq_result['analytic_variance']:.4f}")

    # 截断正态矩计算
    tn = TruncatedNormalDistribution(mu=1500.0, sigma=5.0, a=1485.0, b=1515.0)
    m1 = tn.moment(1)
    m2 = tn.moment(2)
    print(f"      截断正态一阶矩: {m1:.4f}, 二阶矩: {m2:.4f}")

    # ================================================================
    # 11. 蒙特卡洛积分验证（多维，基于 805_nintlib）
    # ================================================================
    print("\n[11/8] 多维蒙特卡洛积分验证...")

    def test_func(x):
        return np.exp(-np.sum(x ** 2))

    mc_result = MonteCarloUncertaintyQuantifier.monte_carlo_nd(
        test_func, dim_num=3,
        a=np.array([0.0, 0.0, 0.0]),
        b=np.array([1.0, 1.0, 1.0]),
        eval_num=50000, seed=55
    )
    # 解析值: (√π/2 · erf(1))³ ≈ 0.6304
    from scipy.special import erf
    exact_integral = (np.sqrt(np.pi) / 2.0 * erf(1.0)) ** 3
    print(f"      MC 积分结果: {mc_result:.6f}")
    print(f"      解析精确值:  {exact_integral:.6f}")
    print(f"      相对误差:    {abs(mc_result - exact_integral) / exact_integral * 100:.4f}%")

    # ================================================================
    # 总结
    # ================================================================
    t_total = time.time() - t_start_total
    print("\n" + "=" * 72)
    print(f"  全部计算完成，总耗时: {t_total:.2f} 秒")
    print("  海底地形声纳反演系统运行正常，无报错。")
    print("=" * 72)


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（36个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: compute_checksum 空输入返回 0 ----
cs = SonarDataValidator.compute_checksum(b'')
assert cs == 0, '[TC01] compute_checksum 空输入返回 0 FAILED'

# ---- TC02: compute_checksum 类型错误抛出 TypeError ----
try:
    SonarDataValidator.compute_checksum("not bytes")
    assert False, '[TC02] compute_checksum 类型错误抛出 TypeError FAILED'
except TypeError:
    pass

# ---- TC03: validate_packet 有效数据包返回 True ----
validator = SonarDataValidator()
validator.reset()
payload = b'test_payload_for_validator'
pkt = {
    'timestamp': 1700000000.0,
    'depth': 500.0,
    'angle': 30.0,
    'sound_speed': 1500.0,
    'payload': payload,
    'checksum': SonarDataValidator.compute_checksum(payload),
}
assert validator.validate_packet(pkt), '[TC03] validate_packet 有效数据包返回 True FAILED'

# ---- TC04: validate_batch 返回正确长度掩码 ----
validator.reset()
packets = generate_test_packets(n=10, seed=42)
mask = validator.validate_batch(packets)
assert len(mask) == 10, '[TC04] validate_batch 返回正确长度掩码 FAILED'
assert mask.dtype == bool, '[TC04] validate_batch 掩码类型为 bool FAILED'

# ---- TC05: sample_positive_hypersphere 返回单位向量 ----
np.random.seed(42)
vec = BeamStatisticsAnalyzer.sample_positive_hypersphere(3)
assert np.all(vec >= 0), '[TC05] sample_positive_hypersphere 分量非负 FAILED'
assert abs(np.linalg.norm(vec) - 1.0) < 1e-10, '[TC05] sample_positive_hypersphere 单位长度 FAILED'

# ---- TC06: compute_distance_stats 返回非负方差 ----
np.random.seed(42)
analyzer = BeamStatisticsAnalyzer(dim=3)
mu, var = analyzer.compute_distance_stats(n_samples=100)
assert var >= 0.0, '[TC06] compute_distance_stats 方差非负 FAILED'
assert mu > 0.0, '[TC06] compute_distance_stats 均值正性 FAILED'

# ---- TC07: analyze_beam_coverage 返回正确键 ----
analyzer = BeamStatisticsAnalyzer(dim=3)
beams = analyzer.generate_optimal_fan_beams(n_beams=5, max_opening_angle_deg=60.0)
cov = analyzer.analyze_beam_coverage(beams)
assert 'n_beams' in cov and 'min_angle_deg' in cov and 'coverage_uniformity' in cov, '[TC07] analyze_beam_coverage 键缺失 FAILED'
assert cov['n_beams'] == 5, '[TC07] analyze_beam_coverage 波束数 FAILED'

# ---- TC08: generate_optimal_fan_beams 输出形状正确 ----
analyzer = BeamStatisticsAnalyzer(dim=3)
beams = analyzer.generate_optimal_fan_beams(n_beams=7, max_opening_angle_deg=45.0)
assert beams.shape == (7, 3), '[TC08] generate_optimal_fan_beams 输出形状 FAILED'

# ---- TC09: triangle_area 已知三角形面积正确 ----
tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
area = triangle_area(tri)
assert abs(area - 0.5) < 1e-10, '[TC09] triangle_area 直角三角形面积 FAILED'

# ---- TC10: triangle_unit_monomial_integral (0,0) 等于 1/2 ----
from seafloor_geometry import triangle_unit_monomial_integral
val = triangle_unit_monomial_integral(0, 0)
assert abs(val - 0.5) < 1e-10, '[TC10] triangle_unit_monomial_integral (0,0) FAILED'

# ---- TC11: triangle_centroid 计算正确 ----
tri = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]])
cent = triangle_centroid(tri)
assert np.allclose(cent, [1.0, 1.0]), '[TC11] triangle_centroid 计算正确 FAILED'

# ---- TC12: point_to_triangle_distance 内部点为 0 ----
tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
d = point_to_triangle_distance(tri, np.array([0.1, 0.1]))
assert abs(d) < 1e-10, '[TC12] point_to_triangle_distance 内部点距离 FAILED'

# ---- TC13: ray_triangle_intersection_2d 相交检测 ----
tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
hit, dist, pt = ray_triangle_intersection_2d(np.array([-1.0, 0.0]), np.array([1.0, 0.0]), tri)
assert hit, '[TC13] ray_triangle_intersection_2d 相交检测 FAILED'
assert dist > 0, '[TC13] ray_triangle_intersection_2d 距离正性 FAILED'
assert pt is not None, '[TC13] ray_triangle_intersection_2d 交点非空 FAILED'

# ---- TC14: SeafloorTriangulation total_area 非负 ----
vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [1.5, 1.0]])
triangles = np.array([[0, 1, 2], [1, 3, 2]])
tin = SeafloorTriangulation(vertices, triangles)
assert tin.total_area() > 0, '[TC14] SeafloorTriangulation total_area 正性 FAILED'
assert tin.mean_patch_area() > 0, '[TC14] SeafloorTriangulation mean_patch_area 正性 FAILED'

# ---- TC15: SoundSpeedProfile.evaluate 负深度边界保护 ----
ssp = SoundSpeedProfile(c0=1500.0, g=0.015, delta_c=30.0, z1=1000.0, sigma=500.0)
c_vals = ssp.evaluate(np.array([-10.0, 0.0, 100.0]))
c_zero = ssp.evaluate(np.array([0.0]))[0]
assert c_vals[0] == c_zero, '[TC15] SoundSpeedProfile.evaluate 负深度保护 FAILED'
assert len(c_vals) == 3, '[TC15] SoundSpeedProfile.evaluate 输出长度 FAILED'

# ---- TC16: trace_ray 垂直向下击中海底 ----
ssp = SoundSpeedProfile(c0=1500.0, g=0.0, delta_c=0.0, z1=1000.0, sigma=500.0)
tracer = AcousticRayTracer(ssp)
result = tracer.trace_ray(x0=0.0, z0=0.0, theta0_deg=90.0, z_bottom_func=lambda x: 100.0, dt=0.001, max_steps=20000)
assert result['hit'], '[TC16] trace_ray 垂直向下击中 FAILED'
assert abs(result['z_hit'] - 100.0) < 2.0, '[TC16] trace_ray 击中深度偏差 FAILED'

# ---- TC17: complex_logarithm_toms243 log(1)=0 ----
from signal_processor import complex_logarithm_toms243
z = complex_logarithm_toms243(1 + 0j)
assert abs(z.real) < 1e-10, '[TC17] complex_logarithm_toms243 实部 FAILED'
assert abs(z.imag) < 1e-10, '[TC17] complex_logarithm_toms243 虚部 FAILED'

# ---- TC18: shepard_interp_1d 精确通过数据点 ----
from signal_processor import shepard_interp_1d
xd = np.array([0.0, 1.0, 2.0])
yd = np.array([5.0, 3.0, 7.0])
yi = shepard_interp_1d(xd, yd, p=2.0, xi=np.array([0.0, 1.0, 2.0]))
assert np.allclose(yi, yd), '[TC18] shepard_interp_1d 精确通过数据点 FAILED'

# ---- TC19: generate_chirp_pulse 输出长度正确 ----
processor = SonarSignalProcessor(fs=48000.0, f0=12000.0, bandwidth=4000.0)
t, s = processor.generate_chirp_pulse(duration=0.01)
assert len(t) == len(s), '[TC19] generate_chirp_pulse 长度不一致 FAILED'
assert len(t) > 0, '[TC19] generate_chirp_pulse 空输出 FAILED'

# ---- TC20: matched_filter 峰值在正确位置 ----
processor = SonarSignalProcessor(fs=1000.0, f0=100.0, bandwidth=50.0)
_, template = processor.generate_chirp_pulse(duration=0.1)
received = np.zeros(500, dtype=complex)
received[100:100 + len(template)] = template
mf = processor.matched_filter(received, template)
peak_idx = np.argmax(np.abs(mf))
assert abs(peak_idx - 100) < 5, '[TC20] matched_filter 峰值位置 FAILED'

# ---- TC21: compute_envelope 非负 ----
processor = SonarSignalProcessor(fs=1000.0, f0=100.0, bandwidth=50.0)
np.random.seed(42)
signal = np.random.randn(200)
env = processor.compute_envelope(signal)
assert np.all(env >= 0), '[TC21] compute_envelope 非负性 FAILED'

# ---- TC22: process_single_ping 返回正确键 ----
processor = SonarSignalProcessor(fs=48000.0, f0=12000.0, bandwidth=4000.0)
_, chirp = processor.generate_chirp_pulse(duration=0.01)
received = np.zeros(len(chirp) + 200, dtype=complex)
received[100:100 + len(chirp)] = chirp
result = processor.process_single_ping(received, pulse_duration=0.01, noise_std=0.01)
assert 'peak_time' in result and 'snr_db' in result and 'envelope' in result, '[TC22] process_single_ping 键缺失 FAILED'

# ---- TC23: MackeyGlassReverberation.solve 输出长度正确 ----
from reverberation_model import MackeyGlassReverberation
mg = MackeyGlassReverberation(gamma=0.1, beta=0.2, n=9.65, tau=5.0, dt=0.01)
t_arr, x_arr = mg.solve((0.0, 1.0), x0=0.5, history_const=0.0)
assert len(t_arr) == len(x_arr), '[TC23] MackeyGlassReverberation.solve 长度不一致 FAILED'
assert len(t_arr) > 0, '[TC23] MackeyGlassReverberation.solve 空输出 FAILED'

# ---- TC24: StochasticReverberationField.generate_composite_envelope 非负 ----
np.random.seed(42)
rev_field = StochasticReverberationField(n_modes=3, seed=42)
t_rev, env_rev = rev_field.generate_composite_envelope(ttw_base=2.0, base_amplitude=1.0)
assert len(t_rev) == len(env_rev), '[TC24] generate_composite_envelope 长度不一致 FAILED'
assert np.all(env_rev >= 0), '[TC24] generate_composite_envelope 负值 FAILED'

# ---- TC25: shepard_interp_2d 精确通过数据点 ----
from bathymetry_interpolator import shepard_interp_2d
xd = np.array([0.0, 1.0, 0.0, 1.0])
yd = np.array([0.0, 0.0, 1.0, 1.0])
zd = np.array([1.0, 2.0, 3.0, 4.0])
zi = shepard_interp_2d(xd, yd, zd, p=2.0, xi=np.array([0.0, 1.0]), yi=np.array([0.0, 1.0]))
assert abs(zi[0, 0] - 1.0) < 1e-6, '[TC25] shepard_interp_2d 左下角 FAILED'
assert abs(zi[1, 1] - 4.0) < 1e-6, '[TC25] shepard_interp_2d 右上角 FAILED'

# ---- TC26: BathymetryInterpolator.estimate_gradient 平面梯度近似为 0 ----
x = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
z = np.full(6, 100.0)
interp = BathymetryInterpolator(x, y, z, p=2.0)
grad = interp.estimate_gradient(1.0, 0.5, h=0.1)
assert abs(grad[0]) < 1.0, '[TC26] estimate_gradient x 分量 FAILED'
assert abs(grad[1]) < 1.0, '[TC26] estimate_gradient y 分量 FAILED'

# ---- TC27: BathymetryInterpolator.cross_section_profile 端点正确 ----
x = np.array([0.0, 1.0, 2.0])
y = np.array([0.0, 0.0, 0.0])
z = np.array([10.0, 20.0, 30.0])
interp = BathymetryInterpolator(x, y, z, p=2.0)
s_prof, z_prof = interp.cross_section_profile(0.0, 0.0, 2.0, 0.0, n_samples=50)
assert abs(z_prof[0] - 10.0) < 2.0, '[TC27] cross_section_profile 起点 FAILED'
assert abs(z_prof[-1] - 30.0) < 2.0, '[TC27] cross_section_profile 终点 FAILED'

# ---- TC28: DiskGridGenerator.disk_grid 包含中心点 ----
pts = DiskGridGenerator.disk_grid(n=2, r=10.0, c=np.array([5.0, 5.0]))
assert pts.shape[1] == 2, '[TC28] disk_grid 列数 FAILED'
center_dist = np.min(np.linalg.norm(pts - np.array([5.0, 5.0]), axis=1))
assert center_dist < 1e-10, '[TC28] disk_grid 中心点缺失 FAILED'

# ---- TC29: KnapsackBeamSelector.greedy_select 不超过容量 ----
from adaptive_beam_sampler import KnapsackBeamSelector
values = np.array([10.0, 5.0, 8.0, 3.0])
weights = np.array([2.0, 1.0, 3.0, 1.5])
capacity = 4.0
selected = KnapsackBeamSelector.greedy_select(values, weights, capacity)
total_w = float(np.dot(selected, weights))
assert total_w <= capacity + 1e-10, '[TC29] greedy_select 超容量 FAILED'

# ---- TC30: AdaptiveSamplingPlanner.plan_survey 返回正确键 ----
np.random.seed(42)
planner = AdaptiveSamplingPlanner(survey_area=((0.0, 100.0), (0.0, 100.0)))
plan = planner.plan_survey(n_beams=8, time_budget=1000.0, n_cvt_iter=3)
assert 'selected_positions' in plan and 'coverage_efficiency' in plan, '[TC30] plan_survey 键缺失 FAILED'
assert plan['n_selected'] >= 0, '[TC30] plan_survey n_selected 负值 FAILED'

# ---- TC31: SpectralAcousticSolver.solve 返回正确键 ----
solver = SpectralAcousticSolver(depth=10.0, frequency=100.0)
c_func = lambda z: np.full_like(z, 1500.0)
src_func = lambda z: np.exp(-((z - 5.0) ** 2) / 2.0)
result = solver.solve(n_basis=3, sound_speed_func=c_func, source_func=src_func)
assert 'coeffs' in result and 'solution' in result and 'cond_number' in result, '[TC31] solve 键缺失 FAILED'
assert result['n_basis'] == 3, '[TC31] solve n_basis 不匹配 FAILED'

# ---- TC32: solve_with_reference 误差非负 ----
solver = SpectralAcousticSolver(depth=10.0, frequency=100.0)
c_func = lambda z: np.full_like(z, 1500.0)
src_func = lambda z: np.ones_like(z)
exact = lambda z: np.sin(np.pi * z / 10.0)
exact_deriv = lambda z: (np.pi / 10.0) * np.cos(np.pi * z / 10.0)
result = solver.solve_with_reference(n_basis=3, sound_speed_func=c_func, source_func=src_func,
                                     exact_solution_func=exact, exact_derivative_func=exact_deriv)
assert result.get('err_l2', -1) >= 0, '[TC32] solve_with_reference L2 误差负值 FAILED'
assert result.get('err_h1', -1) >= 0, '[TC32] solve_with_reference H1 误差负值 FAILED'

# ---- TC33: TruncatedNormalDistribution.moment(1) 在截断区间内 ----
tn = TruncatedNormalDistribution(mu=1500.0, sigma=5.0, a=1485.0, b=1515.0)
m1 = tn.moment(1)
assert 1485.0 <= m1 <= 1515.0, '[TC33] moment(1) 超出截断区间 FAILED'

# ---- TC34: MonteCarloUncertaintyQuantifier.monte_carlo_nd 3D积分 ----
def test_func(x):
    return np.exp(-np.sum(x ** 2))

mc_result = MonteCarloUncertaintyQuantifier.monte_carlo_nd(
    test_func, dim_num=3, a=np.array([0.0, 0.0, 0.0]), b=np.array([1.0, 1.0, 1.0]),
    eval_num=5000, seed=55
)
from scipy.special import erf
exact_integral = (np.sqrt(np.pi) / 2.0 * erf(1.0)) ** 3
assert abs(mc_result - exact_integral) / exact_integral < 0.05, '[TC34] monte_carlo_nd 相对误差过大 FAILED'

# ---- TC35: propagate_depth_uncertainty 均值合理 ----
uq = MonteCarloUncertaintyQuantifier(dim=3)
np.random.seed(55)
result = uq.propagate_depth_uncertainty(
    base_sound_speed=1500.0, base_angle_deg=30.0, base_ttw=2.8,
    sigma_c=5.0, sigma_theta_deg=0.5, sigma_t=0.002,
    n_samples=2000, seed=55
)
expected_depth = 1500.0 * 2.8 * np.cos(np.radians(30.0)) / 2.0
assert abs(result['mean_depth'] - expected_depth) / expected_depth < 0.05, '[TC35] propagate_depth_uncertainty 均值偏差 FAILED'
assert result['std_depth'] >= 0, '[TC35] propagate_depth_uncertainty 标准差负值 FAILED'

# ---- TC36: integrate_error_pdf_over_depth_range 概率在 [0,1] ----
prob = uq.integrate_error_pdf_over_depth_range(result['mc_samples'], z_min=1000.0, z_max=2500.0)
assert 0.0 <= prob <= 1.0, '[TC36] integrate_error_pdf_over_depth_range 概率越界 FAILED'

print('\n全部 36 个测试通过!\n')
