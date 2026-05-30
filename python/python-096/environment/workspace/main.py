
import numpy as np
import sys




from array_geometry import (
    generate_planar_array,
    generate_conformal_array,
    distmesh_2d_simple,
    tet_mesh_quality_metrics,
    sphere_llt_grid_line_count,
)
from em_field_core import (
    ArrayFactorCalculator,
    MutualCouplingMatrix,
    near_field_e_field,
    C_LIGHT,
    ETA_0,
)
from stochastic_channel import (
    RandomWalkPhaseNoise,
    sample_spatial_fading,
    bivariate_normal_cdf,
    sidelobe_level_cdf,
)
from beamforming_optimizer import (
    RK12Solver,
    KeplerPerturbedArrayCoupling,
    LangfordBeamPhaseDynamics,
    AdaptiveBeamformerODE,
)
from quadrature_engine import (
    integrate_square_minimal,
    test_wedge_quadrature_exactness,
    wedge01_volume,
    square_minimal_rule,
)
from phase_quantization import (
    DigitalPhaseShifter,
    hamming_distance_matrix_gray,
    hamming_distance_matrix_binary,
    generate_codebook_sequence,
)
from numerical_utils import (
    NewtonInterpolator1D,
    filename_increment,
    safe_inverse_sqrt,
    rotation_matrix_z,
)
from convergence_analysis import (
    cube_distance_pdf_exact,
    cube_distance_stats_monte_carlo,
    histogram_2d_uniformity,
    compute_array_pattern_metric,
    convergence_rate_residual,
)


def print_section(title: str):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def demo_array_geometry():
    print_section("[1] 阵列几何生成与网格质量评估")


    pos_planar = generate_planar_array(nx=8, ny=8, dx=0.5, dy=0.5,
                                        aperture_type='rectangular')
    print(f"  平面阵列单元数: {pos_planar.shape[0]}")


    pos_conformal = generate_conformal_array(r=1.0, lat_num=4, long_num=8)
    line_count = sphere_llt_grid_line_count(lat_num=4, long_num=8)
    print(f"  球面共形阵列单元数: {pos_conformal.shape[0]}, 网格线数: {line_count}")


    def fd_circle(p):
        return np.sqrt(np.sum(p ** 2, axis=1)) - 1.0

    def fh_uniform(p):
        return np.ones(p.shape[0])

    p_nonuniform, t_nonuniform = distmesh_2d_simple(
        fd_circle, fh_uniform, h0=0.25,
        box=np.array([[-1.0, -1.0], [1.0, 1.0]]),
        iteration_max=20, pfix=None
    )
    print(f"  非均匀圆形阵列（DistMesh）单元数: {p_nonuniform.shape[0]}, 三角形数: {t_nonuniform.shape[0]}")


    if t_nonuniform.shape[0] > 0:

        nodes_3d = np.hstack([p_nonuniform, np.zeros((p_nonuniform.shape[0], 1))])

        nodes_3d_top = nodes_3d.copy()
        nodes_3d_top[:, 2] = 0.1
        nodes_all = np.vstack([nodes_3d, nodes_3d_top])
        nt = t_nonuniform.shape[0]
        tetra = np.zeros((nt, 4), dtype=int)
        tetra[:, :3] = t_nonuniform
        tetra[:, 3] = t_nonuniform[:, 0] + p_nonuniform.shape[0]
        quality = tet_mesh_quality_metrics(nodes_all, tetra)
        print("  四面体网格质量统计:")
        for key, val in quality.items():
            print(f"    {key}: min={val['min']:.6f}, mean={val['mean']:.6f}, max={val['max']:.6f}, var={val['var']:.6e}")

    return pos_planar, pos_conformal


def demo_em_field_and_coupling(pos_planar: np.ndarray):
    print_section("[2] 电磁方向图与互耦分析")

    freq = 3.0e9




    raise NotImplementedError("Hole 3: demo_em_field_and_coupling 核心调用待实现")


def demo_stochastic_analysis():
    print_section("[3] 随机信道与统计噪声分析")


    rw = RandomWalkPhaseNoise(step_delta=0.02, seed=42)
    time, x2_ave, x2_max = rw.simulate(step_num=500, walk_num=2000)
    theoretical = rw.theoretical_msd(500)
    print(f"  随机游走 500 步后均方位移: {x2_ave[-1]:.6f}")
    print(f"  理论均方位移: {theoretical:.6f}")
    print(f"  最大位移平方: {x2_max[-1]:.6f}")


    gains = sample_spatial_fading(n_samples=1000, correlation_length=0.3, seed=42)
    print(f"  空间衰落增益均值: {np.mean(gains):.4f}, 标准差: {np.std(gains):.4f}")


    rho = 0.5
    prob = bivariate_normal_cdf(1.0, 1.0, rho)
    print(f"  双变量正态 CDF(1.0, 1.0; rho={rho}) = {prob:.6f}")


    sll_prob = sidelobe_level_cdf(-10.0, n_elements=64)
    print(f"  64 元阵列旁瓣低于 -10 dB 的概率: {sll_prob:.4f}")


def demo_beamforming_optimization():
    print_section("[4] 自适应波束赋形 ODE 优化")




    raise NotImplementedError("Hole 4: demo_beamforming_optimization 核心调用待实现")


def demo_quadrature_and_interpolation():
    print_section("[5] 高精度数值积分与插值验证")



    def f_test(x, y):
        return x ** 2 + y ** 2

    exact = 8.0 / 3.0
    for deg in [3, 5, 7, 10]:
        approx = integrate_square_minimal(f_test, deg=deg)
        err = abs(approx - exact)
        print(f"  正方形最小规则 deg={deg}: 积分值={approx:.10f}, 误差={err:.3e}")



    wedge_points = np.array([
        [0.25, 0.25, 0.0],
        [0.25, 0.25, 0.5],
        [0.25, 0.25, -0.5],
        [0.5, 0.25, 0.0],
        [0.25, 0.5, 0.0],
    ])
    wedge_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    exactness = test_wedge_quadrature_exactness(wedge_points, wedge_weights, degree_max=2)
    print("  楔形体求积精确性检验:")
    for deg, err in exactness.items():
        print(f"    总次数 {deg}: 最大误差 = {err:.6e}")


    xd = np.linspace(-1.0, 1.0, 7)
    yd = np.sin(np.pi * xd)
    interp = NewtonInterpolator1D(xd, yd)
    xi = np.array([0.3, -0.7, 0.99])
    yi = interp.evaluate(xi)
    yi_true = np.sin(np.pi * xi)
    print(f"  Newton 插值误差在测试点: {np.max(np.abs(yi - yi_true)):.3e}")


def demo_phase_quantization():
    print_section("[6] 数字移相器量化与格雷码编码")


    dps = DigitalPhaseShifter(bits=6)
    continuous_phase = np.array([0.0, 0.5, 1.2, 2.5, 3.8, 5.5, 6.1])
    quantized, gray_codes = dps.quantize_and_code(continuous_phase)
    print("  连续相位 -> 量化相位 (rad):")
    for cp, qp in zip(continuous_phase, quantized.flatten()):
        print(f"    {cp:.3f} -> {qp:.3f}")


    n_states = 15
    dg = hamming_distance_matrix_gray(n_states)
    db = hamming_distance_matrix_binary(n_states)

    gray_adj = [int(dg[i, i + 1]) for i in range(n_states)]
    bin_adj = [int(db[i, i + 1]) for i in range(n_states)]
    print(f"  格雷码相邻状态汉明距离: {gray_adj[:8]}...")
    print(f"  二进制相邻状态汉明距离: {bin_adj[:8]}...")
    print(f"  格雷码最大相邻距离: {max(gray_adj)}, 二进制最大相邻距离: {max(bin_adj)}")


    names = generate_codebook_sequence("beam_codebook_001.dat", 5)
    print(f"  码本文件名序列: {names}")


def demo_convergence_and_distance():
    print_section("[7] 收敛分析与距离统计验证")


    mu_est, var_est = cube_distance_stats_monte_carlo(n_samples=20000, seed=42)
    print(f"  立方体距离 MC 估计: 均值={mu_est:.6f}, 方差={var_est:.6f}")
    print(f"  理论参考: E[D]≈0.661707, Var[D]≈0.062223")


    d_test = np.array([0.3, 0.8, 1.2, 1.5])
    pdf_vals = cube_distance_pdf_exact(d_test)
    print(f"  距离 PDF 在 d={d_test}: {pdf_vals}")


    np.random.seed(42)
    sample_points = np.random.randn(2000, 2) * 0.3
    uniformity = histogram_2d_uniformity(sample_points, bins=8)
    print(f"  2D 直方图均匀性: chi2={uniformity['chi2']:.2f}, p={uniformity['p_uniformity']:.4f}, max_dev={uniformity['max_deviation']:.4f}")


    theta_pat = np.linspace(-np.pi / 2, np.pi / 2, 361)
    pattern_db = 20.0 * np.log10(np.maximum(np.sinc(4.0 * np.sin(theta_pat)), 1e-4))
    mainlobe_idx = np.where(np.abs(theta_pat) < 0.15)[0]
    metrics = compute_array_pattern_metric(pattern_db, mainlobe_idx)
    print(f"  方向图指标: 峰值旁瓣={metrics['peak_sidelobe_level_db']:.2f} dB, ISLR={metrics['integrated_sidelobe_ratio_db']:.2f} dB")


    residuals = np.array([1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125])
    rate = convergence_rate_residual(residuals)
    print(f"  残差收敛速率估计: {rate:.4f} (理论 0.5)")


def demo_filename_utils():
    print_section("[8] 工程辅助工具验证")

    fn = "simulation_009.txt"
    for _ in range(3):
        fn = filename_increment(fn)
        print(f"  文件名递增: {fn}")


    Rz = rotation_matrix_z(np.pi / 4)
    R_test = np.array([1.0, 0.0, 0.0])
    R_rot = Rz @ R_test
    print(f"  Z轴旋转45度验证: ({R_rot[0]:.4f}, {R_rot[1]:.4f}, {R_rot[2]:.4f})")


def main():
    print("\n" + "#" * 72)
    print("#  电磁学：天线阵列波束赋形与优化 — 博士级科研代码合成项目")
    print("#  项目编号: 096_synth_project")
    print("#  科学领域: 电磁学 — 大规模相控阵自适应波束赋形")
    print("#" * 72)


    np.random.seed(2024)

    pos_planar, pos_conformal = demo_array_geometry()
    pattern_db, theta = demo_em_field_and_coupling(pos_planar)
    demo_stochastic_analysis()
    demo_beamforming_optimization()
    demo_quadrature_and_interpolation()
    demo_phase_quantization()
    demo_convergence_and_distance()
    demo_filename_utils()

    print("\n" + "#" * 72)
    print("#  仿真完成。所有模块运行正常，无报错。")
    print("#" * 72 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
