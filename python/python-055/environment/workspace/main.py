
import numpy as np
import time


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




    print("\n[1/8] 声纳数据包验证与质量控制...")
    validator = SonarDataValidator()
    packets = generate_test_packets(n=30, seed=55)
    valid_mask = validator.validate_batch(packets)
    n_valid = int(np.sum(valid_mask))
    print(f"      生成 {len(packets)} 个模拟数据包，有效 {n_valid} 个")
    if len(validator.get_error_log()) > 0:
        print(f"      警告日志: {len(validator.get_error_log())} 条")




    print("\n[2/8] 波束覆盖统计特性分析...")
    analyzer = BeamStatisticsAnalyzer(dim=3)
    mu_d, var_d = analyzer.compute_distance_stats(n_samples=3000)
    print(f"      随机波束对平均弦距离: μ = {mu_d:.4f}, σ² = {var_d:.6f}")


    beam_dirs = analyzer.generate_optimal_fan_beams(n_beams=21, max_opening_angle_deg=60.0)
    coverage = analyzer.analyze_beam_coverage(beam_dirs)
    print(f"      扇形波束数: {coverage['n_beams']}")
    print(f"      最小夹角: {coverage['min_angle_deg']:.2f}°")
    print(f"      平均夹角: {coverage['mean_angle_deg']:.2f}°")
    print(f"      覆盖均匀度: {coverage['coverage_uniformity']:.4f}")




    print("\n[3/8] 自适应波束采样规划...")
    planner = AdaptiveSamplingPlanner(survey_area=((0.0, 5000.0), (0.0, 2000.0)))
    plan = planner.plan_survey(n_beams=48, time_budget=1800.0, n_cvt_iter=12)
    print(f"      候选波束数: {len(plan['all_positions'])}")
    print(f"      最优选中数: {plan['n_selected']}")
    print(f"      总信息增益: {plan['total_gain']:.2f}")
    print(f"      总观测成本: {plan['total_cost']:.2f} s")
    print(f"      覆盖效率: {plan['coverage_efficiency']:.4f}")


    disk_pts = DiskGridGenerator.disk_grid(n=4, r=50.0, c=np.array([2500.0, 1000.0]))
    print(f"      单波束 footprint 网格点: {len(disk_pts)} 个")




    print("\n[4/8] 分层海洋声线追踪...")
    ssp = SoundSpeedProfile(c0=1500.0, g=0.015, delta_c=30.0, z1=1000.0, sigma=500.0)
    tracer = AcousticRayTracer(ssp)


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




    print("\n[5/8] 声纳回波信号处理...")
    processor = SonarSignalProcessor(fs=48000.0, f0=12000.0, bandwidth=4000.0)


    t_chirp, chirp = processor.generate_chirp_pulse(duration=0.01)

    delay_samples = 1200
    received = np.zeros(len(chirp) + delay_samples + 500, dtype=complex)
    received[delay_samples:delay_samples + len(chirp)] = chirp * 0.8

    rng = np.random.default_rng(55)
    reverberation = rng.normal(0, 0.02, len(received)) + 1j * rng.normal(0, 0.02, len(received))
    received += reverberation

    proc_result = processor.process_single_ping(received, pulse_duration=0.01, noise_std=0.02)
    print(f"      检测峰值时间: {proc_result['peak_time']:.5f} s")
    print(f"      估计信噪比: {proc_result['snr_db']:.2f} dB")


    freqs, log_spec = processor.compute_log_spectrum(received)
    log_mag_mean = float(np.mean(np.abs(log_spec.real)))
    print(f"      对数频谱平均幅度: {log_mag_mean:.4f}")




    print("\n[6/8] 海洋混响动力学建模...")
    rev_field = StochasticReverberationField(n_modes=4, seed=55)
    ttw_base = 2.8
    t_rev, env_rev = rev_field.generate_composite_envelope(ttw_base, base_amplitude=1.0)
    peak_rev = float(np.max(env_rev))
    mean_rev = float(np.mean(env_rev))
    print(f"      双程时间基准: {ttw_base:.2f} s")
    print(f"      复合混响包络峰值: {peak_rev:.4f}")
    print(f"      复合混响包络均值: {mean_rev:.4f}")




    print("\n[7/8] 海底地形插值与几何分析...")

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


    test_x, test_y = 2500.0, 1000.0
    grad = interpolator.estimate_gradient(test_x, test_y, h=20.0)
    curv = interpolator.estimate_curvature(test_x, test_y, h=50.0)
    print(f"      测深点数: {n_measurements}")
    print(f"      插值网格: {Z_grid.shape}")
    print(f"      在 ({test_x:.0f}, {test_y:.0f}) 处梯度: [{grad[0]:.4f}, {grad[1]:.4f}]")
    print(f"      在 ({test_x:.0f}, {test_y:.0f}) 处曲率: {curv:.6f}")


    s_prof, z_prof = interpolator.cross_section_profile(0.0, 1000.0, 5000.0, 1000.0)
    print(f"      中心剖面长度: {s_prof[-1]:.1f} m")




    print("\n[8/8] 海底三角网几何与谱声学正演...")

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


    p_test = np.array([2500.0, 500.0])
    tri = vertices[triangles[1]]
    dist = point_to_triangle_distance(tri, p_test)
    print(f"      测试点到面片距离: {dist:.2f} m")


    area_tri0 = abs(triangle_area(vertices[triangles[0]]))
    print(f"      面片0面积: {area_tri0:.2f} m²")




    print("\n[9/8] 一维谱有限元 Helmholtz 求解...")
    solver = SpectralAcousticSolver(depth=2000.0, frequency=12000.0)












    raise NotImplementedError("Hole_3: 请实现谱声学正演的输入定义与求解调用")
    print(f"      基函数数量: {result_fem['n_basis']}")
    print(f"      矩阵条件数: {result_fem['cond_number']:.4e}")
    print(f"      L² 误差: {result_fem.get('err_l2', -1):.6e}")
    print(f"      H¹ 误差: {result_fem.get('err_h1', -1):.6e}")




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


    tn = TruncatedNormalDistribution(mu=1500.0, sigma=5.0, a=1485.0, b=1515.0)
    m1 = tn.moment(1)
    m2 = tn.moment(2)
    print(f"      截断正态一阶矩: {m1:.4f}, 二阶矩: {m2:.4f}")




    print("\n[11/8] 多维蒙特卡洛积分验证...")

    def test_func(x):
        return np.exp(-np.sum(x ** 2))

    mc_result = MonteCarloUncertaintyQuantifier.monte_carlo_nd(
        test_func, dim_num=3,
        a=np.array([0.0, 0.0, 0.0]),
        b=np.array([1.0, 1.0, 1.0]),
        eval_num=50000, seed=55
    )

    from scipy.special import erf
    exact_integral = (np.sqrt(np.pi) / 2.0 * erf(1.0)) ** 3
    print(f"      MC 积分结果: {mc_result:.6f}")
    print(f"      解析精确值:  {exact_integral:.6f}")
    print(f"      相对误差:    {abs(mc_result - exact_integral) / exact_integral * 100:.4f}%")




    t_total = time.time() - t_start_total
    print("\n" + "=" * 72)
    print(f"  全部计算完成，总耗时: {t_total:.2f} 秒")
    print("  海底地形声纳反演系统运行正常，无报错。")
    print("=" * 72)


if __name__ == "__main__":
    main()
