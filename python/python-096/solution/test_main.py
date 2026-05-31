"""
main.py
=======
电磁学：天线阵列波束赋形与优化 — 统一入口

博士级科学计算问题：
  面向 5G/6G 大规模 MIMO 系统的自适应数字波束赋形，综合考虑：
    1. 三维共形阵列与非均匀平面阵列的几何生成
    2. 单元间电磁互耦效应（受摄开普勒势场模型）
    3. 射频链路相位噪声的随机游走过程
    4. 数字移相器量化误差（格雷码编码优化）
    5. 高精度数值积分（最小求积规则）验证方向图计算精度
    6. 基于梯度流的自适应相位优化（ODE 演化）
    7. 旁瓣电平的概率统计分析与双变量正态模型
    8. 近场电场分布计算与四面体网格质量评估

运行方式：
    python main.py

输出：
    在标准输出打印完整的仿真结果与性能指标。
"""

import numpy as np
import sys

# ---------------------------------------------------------------------------
# 导入各子模块
# ---------------------------------------------------------------------------
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
    """演示阵列几何生成与网格质量评估。"""
    print_section("[1] 阵列几何生成与网格质量评估")

    # 1.1 平面阵列
    pos_planar = generate_planar_array(nx=8, ny=8, dx=0.5, dy=0.5,
                                        aperture_type='rectangular')
    print(f"  平面阵列单元数: {pos_planar.shape[0]}")

    # 1.2 球面共形阵列（基于 sphere_llt_grid）
    pos_conformal = generate_conformal_array(r=1.0, lat_num=4, long_num=8)
    line_count = sphere_llt_grid_line_count(lat_num=4, long_num=8)
    print(f"  球面共形阵列单元数: {pos_conformal.shape[0]}, 网格线数: {line_count}")

    # 1.3 非均匀平面阵列（基于 distmesh）
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

    # 1.4 四面体网格质量（基于 tet_mesh_quality）
    if t_nonuniform.shape[0] > 0:
        # 将 2D 三角形提升为 3D 四面体（简单构造）
        nodes_3d = np.hstack([p_nonuniform, np.zeros((p_nonuniform.shape[0], 1))])
        # 在 z=0.1 处复制一层
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
    """演示电磁方向图计算与互耦分析。"""
    print_section("[2] 电磁方向图与互耦分析")

    freq = 3.0e9  # 3 GHz
    array = ArrayFactorCalculator(pos_planar, frequency_hz=freq)

    # 2.1 计算方向图
    theta = np.linspace(-np.pi / 2, np.pi / 2, 361)
    phi = np.zeros_like(theta)
    array.apply_steering(theta_s=np.pi / 6, phi_s=0.0)
    pattern_db = array.compute_total_pattern_db(theta, phi, normalize=True)
    peak_idx = np.argmax(pattern_db)
    print(f"  波束指向: {np.degrees(theta[peak_idx]):.2f}°")
    print(f"  主瓣峰值: {pattern_db[peak_idx]:.2f} dB")

    # 2.2 方向性系数
    D = array.directivity(theta_grid=91, phi_grid=180)
    print(f"  方向性系数 D = {D:.2f} ({10.0 * np.log10(D):.2f} dBi)")

    # 2.3 互耦矩阵
    coupling = MutualCouplingMatrix(pos_planar, frequency_hz=freq)
    Z = coupling.build_impedance_matrix()
    print(f"  互耦矩阵条件数: {np.linalg.cond(Z):.2e}")
    print(f"  典型互阻抗 Z_01 = {Z[0, 1]:.3f} Ohm")

    # 2.4 近场电场
    obs_points = np.array([[0.5, 0.0, 0.1], [0.0, 0.5, 0.1], [-0.5, 0.0, 0.1]])
    currents = np.ones(pos_planar.shape[0], dtype=complex)
    E_near = near_field_e_field(pos_planar, currents, obs_points, freq)
    print(f"  近场电场幅度: {[f'{np.linalg.norm(e):.3e} V/m' for e in E_near]}")

    return pattern_db, theta


def demo_stochastic_analysis():
    """演示随机信道与统计噪声分析。"""
    print_section("[3] 随机信道与统计噪声分析")

    # 3.1 随机游走相位噪声
    rw = RandomWalkPhaseNoise(step_delta=0.02, seed=42)
    time, x2_ave, x2_max = rw.simulate(step_num=500, walk_num=2000)
    theoretical = rw.theoretical_msd(500)
    print(f"  随机游走 500 步后均方位移: {x2_ave[-1]:.6f}")
    print(f"  理论均方位移: {theoretical:.6f}")
    print(f"  最大位移平方: {x2_max[-1]:.6f}")

    # 3.2 空间衰落采样
    gains = sample_spatial_fading(n_samples=1000, correlation_length=0.3, seed=42)
    print(f"  空间衰落增益均值: {np.mean(gains):.4f}, 标准差: {np.std(gains):.4f}")

    # 3.3 双变量正态 CDF（Owen T 函数）
    rho = 0.5
    prob = bivariate_normal_cdf(1.0, 1.0, rho)
    print(f"  双变量正态 CDF(1.0, 1.0; rho={rho}) = {prob:.6f}")

    # 3.4 旁瓣电平概率
    sll_prob = sidelobe_level_cdf(-10.0, n_elements=64)
    print(f"  64 元阵列旁瓣低于 -10 dB 的概率: {sll_prob:.4f}")


def demo_beamforming_optimization():
    """演示基于 ODE 的自适应波束赋形优化。"""
    print_section("[4] 自适应波束赋形 ODE 优化")

    # 4.1 受摄开普勒互耦势场
    kepler = KeplerPerturbedArrayCoupling(delta=0.015, e=0.6)
    solver_k = RK12Solver(kepler.derivative)
    t_k, y_k, e_k = solver_k.solve((0.0, 20.0), kepler.y0, 400)
    H0 = kepler.conserved_quantity(kepler.y0)
    H_end = kepler.conserved_quantity(y_k[-1, :])
    print(f"  受摄开普勒 Hamiltonian 初始值: {H0:.6f}")
    print(f"  受摄开普勒 Hamiltonian 终值:   {H_end:.6f}")
    print(f"  能量相对漂移: {abs(H_end - H0) / abs(H0):.6e}")

    # 4.2 Langford 相位动力学
    langford = LangfordBeamPhaseDynamics()
    solver_l = RK12Solver(langford.derivative)
    t_l, y_l, e_l = solver_l.solve((0.0, 50.0), langford.y0, 1000)
    lyap = langford.lyapunov_exponent_estimate((0.0, 50.0), 2500)
    print(f"  Langford 系统终态: ({y_l[-1, 0]:.4f}, {y_l[-1, 1]:.4f}, {y_l[-1, 2]:.4f})")
    print(f"  最大 Lyapunov 指数估计: {lyap:.6f}")

    # 4.3 自适应相位权重梯度流
    bf = AdaptiveBeamformerODE(n_elements=16, steering_angle=np.pi / 6)
    t_bf, phase_hist = bf.evolve(tspan=(0.0, 5.0), n_steps=200)
    final_phase = phase_hist[-1, :]
    print(f"  自适应波束赋形最终相位范围: [{np.min(final_phase):.4f}, {np.max(final_phase):.4f}] rad")


def demo_quadrature_and_interpolation():
    """演示高精度求积与插值。"""
    print_section("[5] 高精度数值积分与插值验证")

    # 5.1 正方形最小规则积分
    # 测试 f(x,y) = x^2 + y^2，精确值 = 8/3
    def f_test(x, y):
        return x ** 2 + y ** 2

    exact = 8.0 / 3.0
    for deg in [3, 5, 7, 10]:
        approx = integrate_square_minimal(f_test, deg=deg)
        err = abs(approx - exact)
        print(f"  正方形最小规则 deg={deg}: 积分值={approx:.10f}, 误差={err:.3e}")

    # 5.2 楔形体求积精确性检验
    # 构造简单的梯形/中点规则节点在楔形体上
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

    # 5.3 Newton 插值验证
    xd = np.linspace(-1.0, 1.0, 7)
    yd = np.sin(np.pi * xd)
    interp = NewtonInterpolator1D(xd, yd)
    xi = np.array([0.3, -0.7, 0.99])
    yi = interp.evaluate(xi)
    yi_true = np.sin(np.pi * xi)
    print(f"  Newton 插值误差在测试点: {np.max(np.abs(yi - yi_true)):.3e}")


def demo_phase_quantization():
    """演示数字移相器量化与格雷码编码。"""
    print_section("[6] 数字移相器量化与格雷码编码")

    # 6.1 6 位移相器量化
    dps = DigitalPhaseShifter(bits=6)
    continuous_phase = np.array([0.0, 0.5, 1.2, 2.5, 3.8, 5.5, 6.1])
    quantized, gray_codes = dps.quantize_and_code(continuous_phase)
    print("  连续相位 -> 量化相位 (rad):")
    for cp, qp in zip(continuous_phase, quantized.flatten()):
        print(f"    {cp:.3f} -> {qp:.3f}")

    # 6.2 汉明距离对比
    n_states = 15
    dg = hamming_distance_matrix_gray(n_states)
    db = hamming_distance_matrix_binary(n_states)
    # 相邻状态汉明距离
    gray_adj = [int(dg[i, i + 1]) for i in range(n_states)]
    bin_adj = [int(db[i, i + 1]) for i in range(n_states)]
    print(f"  格雷码相邻状态汉明距离: {gray_adj[:8]}...")
    print(f"  二进制相邻状态汉明距离: {bin_adj[:8]}...")
    print(f"  格雷码最大相邻距离: {max(gray_adj)}, 二进制最大相邻距离: {max(bin_adj)}")

    # 6.3 码本文件名序列
    names = generate_codebook_sequence("beam_codebook_001.dat", 5)
    print(f"  码本文件名序列: {names}")


def demo_convergence_and_distance():
    """演示收敛分析与距离统计。"""
    print_section("[7] 收敛分析与距离统计验证")

    # 7.1 立方体内距离蒙特卡罗
    mu_est, var_est = cube_distance_stats_monte_carlo(n_samples=20000, seed=42)
    print(f"  立方体距离 MC 估计: 均值={mu_est:.6f}, 方差={var_est:.6f}")
    print(f"  理论参考: E[D]≈0.661707, Var[D]≈0.062223")

    # 7.2 立方体距离 PDF 解析验证
    d_test = np.array([0.3, 0.8, 1.2, 1.5])
    pdf_vals = cube_distance_pdf_exact(d_test)
    print(f"  距离 PDF 在 d={d_test}: {pdf_vals}")

    # 7.3 二维直方图均匀性
    np.random.seed(42)
    sample_points = np.random.randn(2000, 2) * 0.3
    uniformity = histogram_2d_uniformity(sample_points, bins=8)
    print(f"  2D 直方图均匀性: chi2={uniformity['chi2']:.2f}, p={uniformity['p_uniformity']:.4f}, max_dev={uniformity['max_deviation']:.4f}")

    # 7.4 方向图质量指标
    theta_pat = np.linspace(-np.pi / 2, np.pi / 2, 361)
    pattern_db = 20.0 * np.log10(np.maximum(np.sinc(4.0 * np.sin(theta_pat)), 1e-4))
    mainlobe_idx = np.where(np.abs(theta_pat) < 0.15)[0]
    metrics = compute_array_pattern_metric(pattern_db, mainlobe_idx)
    print(f"  方向图指标: 峰值旁瓣={metrics['peak_sidelobe_level_db']:.2f} dB, ISLR={metrics['integrated_sidelobe_ratio_db']:.2f} dB")

    # 7.5 残差收敛速率
    residuals = np.array([1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125])
    rate = convergence_rate_residual(residuals)
    print(f"  残差收敛速率估计: {rate:.4f} (理论 0.5)")


def demo_filename_utils():
    """演示文件名工具。"""
    print_section("[8] 工程辅助工具验证")

    fn = "simulation_009.txt"
    for _ in range(3):
        fn = filename_increment(fn)
        print(f"  文件名递增: {fn}")

    # 旋转矩阵验证
    Rz = rotation_matrix_z(np.pi / 4)
    R_test = np.array([1.0, 0.0, 0.0])
    R_rot = Rz @ R_test
    print(f"  Z轴旋转45度验证: ({R_rot[0]:.4f}, {R_rot[1]:.4f}, {R_rot[2]:.4f})")


def main():
    """主函数：零参数运行，依次执行所有演示模块。"""
    print("\n" + "#" * 72)
    print("#  电磁学：天线阵列波束赋形与优化 — 博士级科研代码合成项目")
    print("#  项目编号: 096_synth_project")
    print("#  科学领域: 电磁学 — 大规模相控阵自适应波束赋形")
    print("#" * 72)

    # 设置随机种子保证可复现
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
    main()
# ---- TC01: generate_conformal_array 球面点半径一致 ----
pos_conf = generate_conformal_array(1.0, 2, 4)
r_dist = np.linalg.norm(pos_conf, axis=1)
assert np.allclose(r_dist, 1.0), '[TC01] generate_conformal_array 球面点半径一致 FAILED'

# ---- TC02: sphere_llt_grid_line_count 公式验证 ----
lines = sphere_llt_grid_line_count(2, 4)
assert lines == 24, '[TC02] sphere_llt_grid_line_count 公式验证 FAILED'

# ---- TC03: generate_planar_array 矩形口径单元数 ----
pos = generate_planar_array(4, 4, 0.5, 0.5, aperture_type='rectangular')
assert pos.shape == (16, 3), '[TC03] generate_planar_array 矩形口径单元数 FAILED'

# ---- TC04: generate_planar_array 圆形口径过滤 ----
pos_c = generate_planar_array(4, 4, 0.5, 0.5, aperture_type='circular')
assert pos_c.shape[0] <= 16 and pos_c.shape[1] == 3, '[TC04] generate_planar_array 圆形口径过滤 FAILED'

# ---- TC05: generate_conformal_array 输出尺寸 ----
pos_conf2 = generate_conformal_array(1.0, 2, 4)
assert pos_conf2.shape == (10, 3), '[TC05] generate_conformal_array 输出尺寸 FAILED'

# ---- TC06: tet_mesh_quality_metrics 返回结构 ----
nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
tetra = np.array([[0, 1, 2, 3]])
q = tet_mesh_quality_metrics(nodes, tetra)
assert 'quality1' in q and 'min' in q['quality1'], '[TC06] tet_mesh_quality_metrics 返回结构 FAILED'

# ---- TC07: ArrayFactorCalculator 方向图输出尺寸 ----
pos = generate_planar_array(4, 4, 0.5, 0.5)
afc = ArrayFactorCalculator(pos, frequency_hz=3e9)
theta = np.linspace(-np.pi / 2, np.pi / 2, 37)
phi = np.zeros_like(theta)
af = afc.compute_array_factor(theta, phi)
assert af.shape == theta.shape, '[TC07] ArrayFactorCalculator 方向图输出尺寸 FAILED'

# ---- TC08: ArrayFactorCalculator 总方向图峰值在波束指向 ----
afc.apply_steering(np.pi / 6, 0.0)
pat_db = afc.compute_total_pattern_db(theta, phi, normalize=True)
assert np.max(pat_db) >= -0.1, '[TC08] ArrayFactorCalculator 总方向图峰值 FAILED'

# ---- TC09: ArrayFactorCalculator 方向性系数为正 ----
D = afc.directivity(theta_grid=31, phi_grid=31)
assert D > 0.0 and np.isfinite(D), '[TC09] ArrayFactorCalculator 方向性系数为正 FAILED'

# ---- TC10: MutualCouplingMatrix 阻抗矩阵对称 ----
mc = MutualCouplingMatrix(pos[:4], frequency_hz=3e9)
Z = mc.build_impedance_matrix()
assert Z.shape == (4, 4) and np.allclose(Z, Z.T), '[TC10] MutualCouplingMatrix 阻抗矩阵对称 FAILED'

# ---- TC11: near_field_e_field 返回形状正确 ----
obs = np.array([[0.5, 0.0, 0.1], [0.0, 0.5, 0.1]])
curr = np.ones(pos.shape[0], dtype=complex)
E = near_field_e_field(pos, curr, obs, 3e9)
assert E.shape == (2, 3), '[TC11] near_field_e_field 返回形状正确 FAILED'

# ---- TC12: bivariate_normal_cdf 返回值在概率区间 ----
p_bound = bivariate_normal_cdf(1.0, 1.0, 0.5)
assert 0.0 <= p_bound <= 1.0, '[TC12] bivariate_normal_cdf 返回值在概率区间 FAILED'

# ---- TC13: bivariate_normal_cdf 独立性分解 ----
p2 = bivariate_normal_cdf(0.0, 0.0, 0.0)
assert abs(p2 - 0.25) < 1e-3, '[TC13] bivariate_normal_cdf 独立性分解 FAILED'

# ---- TC14: RandomWalkPhaseNoise 理论均方位移 ----
rw = RandomWalkPhaseNoise(step_delta=0.02, seed=42)
msd = rw.theoretical_msd(500)
assert abs(msd - 500 * 0.02 ** 2) < 1e-10, '[TC14] RandomWalkPhaseNoise 理论均方位移 FAILED'

# ---- TC15: RandomWalkPhaseNoise 可复现性 ----
np.random.seed(42)
rw1 = RandomWalkPhaseNoise(step_delta=0.02, seed=42)
t1, x1, m1 = rw1.simulate(10, 100)
np.random.seed(42)
rw2 = RandomWalkPhaseNoise(step_delta=0.02, seed=42)
t2, x2, m2 = rw2.simulate(10, 100)
assert np.allclose(x1, x2) and np.allclose(m1, m2), '[TC15] RandomWalkPhaseNoise 可复现性 FAILED'

# ---- TC16: sample_spatial_fading 输出尺寸与范围 ----
np.random.seed(42)
gains = sample_spatial_fading(n_samples=200, correlation_length=0.3, seed=42)
assert gains.shape == (200,) and np.all(gains > 0), '[TC16] sample_spatial_fading 输出尺寸与范围 FAILED'

# ---- TC17: sidelobe_level_cdf 概率范围 ----
prob = sidelobe_level_cdf(-10.0, n_elements=64)
assert 0.0 <= prob <= 1.0, '[TC17] sidelobe_level_cdf 概率范围 FAILED'

# ---- TC18: RK12Solver 常数 ODE 精确保持 ----
def const_ode(t, y):
    return np.zeros_like(y)


solver = RK12Solver(const_ode)
t, y, e = solver.solve((0.0, 1.0), np.array([2.0, 3.0]), 10)
assert np.allclose(y[-1], np.array([2.0, 3.0])), '[TC18] RK12Solver 常数 ODE 精确保持 FAILED'

# ---- TC19: KeplerPerturbedArrayCoupling Hamiltonian 有限 ----
kep = KeplerPerturbedArrayCoupling(delta=0.015, e=0.6)
H = kep.conserved_quantity(kep.y0)
assert np.isfinite(H), '[TC19] KeplerPerturbedArrayCoupling Hamiltonian 有限 FAILED'

# ---- TC20: LangfordBeamPhaseDynamics derivative 输出维度 ----
lang = LangfordBeamPhaseDynamics()
dydt = lang.derivative(0.0, lang.y0)
assert dydt.shape == (3,), '[TC20] LangfordBeamPhaseDynamics derivative 输出维度 FAILED'

# ---- TC21: DigitalPhaseShifter 量化相位周期性 ----
dps = DigitalPhaseShifter(bits=4)
qphase = dps.quantize_phase(np.array([0.0, 2 * np.pi, 4 * np.pi]))
assert np.allclose(qphase, qphase[0]), '[TC21] DigitalPhaseShifter 量化相位周期性 FAILED'

# ---- TC22: hamming_distance_matrix_gray 相邻距离为1 ----
dg = hamming_distance_matrix_gray(7)
adj = [int(dg[i, i + 1]) for i in range(7)]
assert all(v == 1 for v in adj), '[TC22] hamming_distance_matrix_gray 相邻距离为1 FAILED'

# ---- TC23: integrate_square_minimal 常数函数精确 ----
val = integrate_square_minimal(lambda x, y: np.ones_like(x), deg=3)
assert abs(val - 4.0) < 1e-12, '[TC23] integrate_square_minimal 常数函数精确 FAILED'

# ---- TC24: integrate_square_minimal x2+y2 解析验证 ----
val2 = integrate_square_minimal(lambda x, y: x ** 2 + y ** 2, deg=3)
assert abs(val2 - 8.0 / 3.0) < 1e-10, '[TC24] integrate_square_minimal x2+y2 解析验证 FAILED'

# ---- TC25: NewtonInterpolator1D 节点处精确恢复 ----
xd = np.array([0.0, 1.0, 2.0, 3.0])
yd = xd ** 2
interp = NewtonInterpolator1D(xd, yd)
yi = interp.evaluate(xd)
assert np.allclose(yi, yd), '[TC25] NewtonInterpolator1D 节点处精确恢复 FAILED'

# ---- TC26: convergence_rate_residual 几何级数 ----
res = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
rate = convergence_rate_residual(res)
assert abs(rate - 0.5) < 0.05, '[TC26] convergence_rate_residual 几何级数 FAILED'

# ---- TC27: safe_inverse_sqrt 非正输入鲁棒性 ----
result = safe_inverse_sqrt(np.array([-1.0, 0.0, 4.0]))
assert np.all(np.isfinite(result)) and abs(result[2] - 0.5) < 1e-10, '[TC27] safe_inverse_sqrt 非正输入鲁棒性 FAILED'

# ---- TC28: rotation_matrix_z 正交性验证 ----
Rz = rotation_matrix_z(np.pi / 3)
assert np.allclose(Rz @ Rz.T, np.eye(3)), '[TC28] rotation_matrix_z 正交性验证 FAILED'

# ---- TC29: cube_distance_pdf_exact 非负性 ----
dvals = np.array([0.1, 0.5, 1.0, 1.3, 1.6])
pdf_vals = cube_distance_pdf_exact(dvals)
assert np.all(pdf_vals >= 0), '[TC29] cube_distance_pdf_exact 非负性 FAILED'

# ---- TC30: filename_increment 基本递增 ----
fn = filename_increment("sim_009.txt")
assert fn == "sim_010.txt", '[TC30] filename_increment 基本递增 FAILED'
print('\n全部 30 个测试通过!\n')
