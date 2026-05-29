"""
main.py
QGP-Transport: 重离子碰撞夸克胶子等离子体输运与相变动力学全链路模拟

统一入口，零参数可运行。
运行该文件将完成以下计算流程:
1. 核几何建模 (Woods-Saxon + Glauber)
2. 重叠区三角网格生成
3. 2+1维粘性流体力学演化
4. 椭圆积分与部分子能量损失
5. 蒙特卡洛事件采样
6. SVD事件涨落分析
7. 参数优化拟合
8. 格点线性系统求解
9. 统计推断与相变分析
"""

import numpy as np
import sys
import time

from nuclear_geometry import NuclearGeometry
from mesh_generator import MeshGenerator
from hydro_evolution import HydroEvolution
from elliptic_integrals import QGPDispersionRelation, rf_carlson, rc_carlson, rd_carlson, rj_carlson
from random_generator import MiddleSquareHybrid, QGPEventSampler
from monte_carlo_sampler import CombinatorialPhysics, HistogramAnalysis, PartonCascade
from svd_analysis import EventSVDAnalyzer, FlowHarmonicDecomposition
from parameter_optimization import QuadraticOptimizer, QGPParameterFit
from linear_solver import RREFSolver, HammingErrorDetection, LatticeDiracSolver
from statistical_inference import NonCentralTDistribution, QGPStatisticalInference


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    np.seterr(divide='ignore', invalid='ignore')
    start_time = time.time()

    print("\n" + "#" * 70)
    print("# QGP-Transport: 重离子碰撞夸克胶子等离子体全链路模拟")
    print("# 科学领域: 粒子物理 — 重离子碰撞夸克胶子等离子体")
    print("#" * 70)

    # ============================================================
    # 1. 核几何建模
    # ============================================================
    print_section("1. 核几何建模 (Au+Au 碰撞)")
    geo = NuclearGeometry(mass_number_a=197, mass_number_b=197,
                          radius_param=1.12, diffuseness=0.54,
                          nucleon_cross_section=4.2)
    b = 5.0  # 碰撞参数 [fm]
    x_grid = np.linspace(-12.0, 12.0, 60)
    y_grid = np.linspace(-12.0, 12.0, 60)

    n_part, n_coll = geo.compute_npart_ncoll(b, x_grid, y_grid)
    eps2, eps4 = geo.eccentricity(b, x_grid, y_grid)
    print(f"  碰撞参数 b = {b:.2f} fm")
    print(f"  参与者数 N_part = {n_part:.1f}")
    print(f"  二叉碰撞数 N_coll = {n_coll:.1f}")
    print(f"  二阶偏心距 ε₂ = {eps2:.4f}")
    print(f"  四阶偏心距 ε₄ = {eps4:.4f}")
    print(f"  Woods-Saxon R_A = {geo.R_A:.3f} fm")
    print(f"  归一化密度 ρ₀ = {geo.rho0:.4f} nucleons/fm³")

    # 边界字追踪
    bx, by = geo.tortoise_boundary_word(n_segments=32)
    print(f"  核表面边界离散化: {len(bx)} 段")

    # ============================================================
    # 2. 三角网格生成
    # ============================================================
    print_section("2. 重叠区自适应三角网格生成")
    mesh = MeshGenerator(max_area=1.0, min_angle=20.0)

    def density_func(x, y):
        return float(geo.thickness_function(np.array([x]), np.array([y]), 'A')[0])

    points, triangles = mesh.generate_adaptive_mesh(
        (-10.0, 10.0), (-10.0, 10.0), density_func, n_base=20
    )
    areas = mesh.triangle_area(points, triangles)
    quality = mesh.triangle_quality(points, triangles)
    adj_counts = mesh.adjacency_count(len(points), triangles)
    boundary = mesh.boundary_nodes(triangles)

    print(f"  节点数: {len(points)}")
    print(f"  三角形数: {len(triangles)}")
    print(f"  平均面积: {np.mean(areas):.4f} fm²")
    print(f"  平均质量: {np.mean(quality):.4f} (1.0=等边)")
    print(f"  边界节点数: {len(boundary)}")
    print(f"  最大邻接度: {np.max(adj_counts)}")

    # 在网格上积分参与者密度
    rho_part = geo.participant_density_profile(b, x_grid, y_grid)
    # 将网格点映射到粗网格
    integral_part = np.trapezoid(np.trapezoid(rho_part, y_grid, axis=1), x_grid, axis=0)
    print(f"  参与者密度积分: {integral_part:.2f} nucleons")

    # ============================================================
    # 3. 粘性流体力学演化
    # ============================================================
    print_section("3. 2+1维粘性流体力学演化")
    hydro = HydroEvolution(eta_s_over_s=0.08, cs2=1.0/3.0,
                           g_star=47.5, tau0=0.6, tau_f=8.0, dtau=0.1)

    # 初始能量密度分布
    nx, ny = 40, 40
    x_h = np.linspace(-10.0, 10.0, nx)
    y_h = np.linspace(-10.0, 10.0, ny)
    Xh, Yh = np.meshgrid(x_h, y_h, indexing='ij')
    # HOLE 3: Convert participant density profile to initial energy density

    tau_grid, eps_hist, T_hist, s_hist = hydro.evolve_2d(
        x_h, y_h, epsilon0, nx=nx, ny=ny
    )

    # Bjorken 1D解析解对比
    tau_1d, eps_1d, T_1d = hydro.bjorken_1d(15.0)
    print(f"  初始固有时 τ₀ = {hydro.tau0} fm/c")
    print(f"  终止固有时 τ_f = {hydro.tau_f} fm/c")
    print(f"  时间步数: {len(tau_grid)}")
    print(f"  初始峰值温度: {np.max(T_hist[0]):.3f} GeV")
    print(f"  最终峰值温度: {np.max(T_hist[-1]):.3f} GeV")
    print(f"  1D Bjorken初始温度: {T_1d[0]:.3f} GeV")
    print(f"  1D Bjorken终止温度: {T_1d[-1]:.3f} GeV")

    # 冻结面
    T_fo = 0.154
    tau_fo = hydro.freezeout_surface(tau_grid, T_hist, T_freezeout=T_fo)
    valid_fo = tau_fo[tau_fo > 0]
    if len(valid_fo) > 0:
        print(f"  冻结温度 T_fo = {T_fo:.3f} GeV")
        print(f"  平均冻结时间: {np.mean(valid_fo):.2f} fm/c")
        print(f"  冻结时间范围: [{np.min(valid_fo):.2f}, {np.max(valid_fo):.2f}] fm/c")

    # 流速与熵产生
    ux, uy = hydro.flow_velocity(eps_hist, x_h, y_h)
    s_prod = hydro.entropy_production(tau_grid, eps_hist, ux, uy)
    print(f"  累积粘性熵产生: {s_prod[-1]:.4f}")

    # ============================================================
    # 4. 椭圆积分与色散关系
    # ============================================================
    print_section("4. Carlson不完全椭圆积分与QGP色散关系")
    rf_val, ierr = rf_carlson(1.0, 2.0, 3.0)
    rc_val, ierr2 = rc_carlson(1.0, 2.0)
    rd_val, ierr3 = rd_carlson(1.0, 2.0, 3.0)
    rj_val, ierr4 = rj_carlson(1.0, 2.0, 3.0, 4.0)
    print(f"  RF(1,2,3) = {rf_val:.8f}")
    print(f"  RC(1,2)   = {rc_val:.8f}")
    print(f"  RD(1,2,3) = {rd_val:.8f}")
    print(f"  RJ(1,2,3,4) = {rj_val:.8f}")

    disp = QGPDispersionRelation()
    m_g = 0.5  # 热胶子质量 [GeV]
    T_mid = 0.3  # 中间温度
    q_gluon = 10.0
    delta_E = disp.gluon_energy_loss(q_gluon, m_g, T_mid)
    p_broad = disp.parton_momentum_broadening(2.0, 1.5, 5.0)
    rho_dilep = disp.dilepton_spectral_function(1.0, T_mid, 0.105)
    print(f"  胶子能量损失 ΔE(q={q_gluon} GeV) = {delta_E:.4f} GeV")
    print(f"  部分子动量展宽 ⟨p_⊥²⟩ = {p_broad:.4f} GeV²")
    print(f"  双轻子谱函数 ρ(q=1 GeV) = {rho_dilep:.6e}")

    # ============================================================
    # 5. 随机数生成与事件采样
    # ============================================================
    print_section("5. 混合伪随机数生成器与蒙特卡洛事件采样")
    rng = MiddleSquareHybrid(seed=20240503, d=5)
    cycle_len, pre_steps = rng.cycle_length(max_steps=50000)
    print(f"  混合RNG周期长度: {cycle_len}")
    print(f"  进入周期步数: {pre_steps}")

    sampler = QGPEventSampler(rng)
    b_samples = sampler.sample_impact_parameter(b_max=15.0, n_samples=500)
    pt_samples = sampler.sample_thermal_momentum(T=0.3, m=0.14, n_samples=500)
    phi_samples = sampler.sample_azimuthal_angle(v2=0.05, n_samples=500)
    fluct_samples = sampler.sample_fluctuation(mean=100.0, std=15.0, n_samples=500)

    print(f"  碰撞参数采样均值: {np.mean(b_samples):.2f} fm")
    print(f"  热动量采样均值: {np.mean(pt_samples):.3f} GeV")
    print(f"  方位角 v₂ 估计: {np.mean(np.cos(2*phi_samples)):.4f}")
    print(f"  涨落采样均值: {np.mean(fluct_samples):.2f}")

    # ============================================================
    # 6. 组合数学与部分子级联
    # ============================================================
    print_section("6. 组合数学: 部分子多重数与色单态计数")
    comb = CombinatorialPhysics()
    stirling = comb.stirling_numbers_second_kind(10, 10)
    bell = comb.bell_numbers(10)
    print(f"  S(10,5) = {stirling[10,5]}")
    print(f"  B(10) = {bell[10]}")
    cascade = PartonCascade(alpha_s=0.3, q0=1.0)
    print(f"  10个胶子色单态估计: {cascade.color_singlet_combinatorics(10)['total_singlets_estimate']}")

    mult = cascade.multiplicity_distribution(E_init=50.0, n_events=200)
    hist_mult = HistogramAnalysis(mult, n_bins=20, range_limits=(0.0, 20.0))
    print(f"  部分子平均多重数: {hist_mult.mean():.2f}")
    print(f"  多重数方差: {hist_mult.variance():.2f}")
    print(f"  偏度: {hist_mult.skewness():.3f}")

    # ============================================================
    # 7. SVD事件涨落分析
    # ============================================================
    print_section("7. SVD/PCA 事件涨落模式分解")
    # 构造模拟事件集
    n_events = 50
    n_pixels = nx * ny
    event_matrix = np.zeros((n_pixels, n_events))
    for iev in range(n_events):
        # 每个事件有随机涨落
        noise = np.random.normal(0.0, 0.05, (nx, ny))
        event_map = T_hist[-1] * (1.0 + noise)
        event_matrix[:, iev] = event_map.flatten()

    svd = EventSVDAnalyzer(n_components=8)
    svd.fit(event_matrix)
    ev_ratio = svd.explained_variance_ratio()
    cum_var = svd.cumulative_variance()
    print(f"  事件数: {n_events}")
    print(f"  主成分数: {svd.n_components}")
    print(f"  第1主成分方差比: {ev_ratio[0]:.4f}")
    print(f"  前3主成分累积方差: {cum_var[min(2, len(cum_var)-1)]:.4f}")
    print(f"  前5主成分累积方差: {cum_var[min(4, len(cum_var)-1)]:.4f}")

    # 流谐波分析
    flow = FlowHarmonicDecomposition()
    v2_2 = flow.cumulant_v2(phi_samples)
    v4_4 = flow.cumulant_v4(phi_samples[:200])
    print(f"  二阶累积量 v₂{{2}} = {v2_2:.4f}")
    print(f"  四阶累积量 v₄{{4}} = {v4_4:.4f}")
    eps_est = flow.eccentricity_from_flow(v2_2, response_coeff=0.18)
    print(f"  从v₂反推偏心距: ε₂ ≈ {eps_est:.4f}")

    # ============================================================
    # 8. 参数优化拟合
    # ============================================================
    print_section("8. QGP参数优化拟合")
    fitter = QGPParameterFit()

    # 模拟实验数据
    pt_bins = np.linspace(0.5, 5.0, 10)
    v2_exp = 0.05 * np.tanh(pt_bins / 2.0) * (1.0 + np.random.normal(0, 0.02, len(pt_bins)))
    eta_s_fit, chi2_v2 = fitter.fit_eta_over_s(v2_exp, pt_bins)
    cs2_fit, res_pt = fitter.fit_cs2(mean_pt_data=1.2)
    tau0_fit, res_dn = fitter.fit_tau0(dNch_deta_data=1200.0)

    print(f"  拟合 η/s = {eta_s_fit:.4f} (χ² = {chi2_v2:.2f})")
    print(f"  拟合 c_s² = {cs2_fit:.4f} (残差 = {res_pt:.4f})")
    print(f"  拟合 τ₀ = {tau0_fit:.3f} fm/c (残差 = {res_dn:.2f})")

    # 联合拟合
    all_params = fitter.fit_all_parameters(v2_exp, mean_pt=1.2, dNch_deta=1200.0)
    print(f"  联合拟合结果:")
    for k, v in all_params.items():
        print(f"    {k} = {v:.4f}")

    # ============================================================
    # 9. 线性系统求解与误差检测
    # ============================================================
    print_section("9. 格点QCD线性系统求解与Hamming误差检测")
    solver = RREFSolver()

    # 测试线性系统
    A_test = np.array([
        [2.0, 1.0, -1.0],
        [-3.0, -1.0, 2.0],
        [-2.0, 1.0, 2.0]
    ])
    b_test = np.array([8.0, -11.0, -3.0])
    x_sol = solver.solve(A_test, b_test)
    rank_a = solver.rank(A_test)
    det_a = solver.determinant(A_test)
    print(f"  测试矩阵秩: {rank_a}")
    print(f"  测试矩阵行列式: {det_a:.2f}")
    print(f"  解: x = [{x_sol[0]:.4f}, {x_sol[1]:.4f}, {x_sol[2]:.4f}]")

    # Hamming校验
    check = HammingErrorDetection.check_linear_system(A_test, x_sol, b_test)
    print(f"  线性系统校验: {'通过' if check else '失败'}")

    x_red, res_norm = HammingErrorDetection.redundant_solve(A_test, b_test)
    print(f"  冗余求解残差范数: {res_norm:.2e}")

    # Hamming编码测试
    data_bits = np.array([1, 0, 1, 1])
    codeword = HammingErrorDetection.encode(data_bits)
    decoded, corrected = HammingErrorDetection.decode(codeword)
    print(f"  Hamming编码: {data_bits} -> {codeword} -> {decoded}")
    print(f"  纠错能力验证: {'通过' if np.array_equal(decoded, data_bits) else '失败'}")

    # 格点Dirac求解
    dirac = LatticeDiracSolver(mass=0.1, lattice_size=6)
    D_mat = dirac.wilson_dirac_matrix()
    eta_src = np.ones(D_mat.shape[0])
    psi_sol = dirac.solve(eta_src)
    print(f"  Wilson-Dirac矩阵大小: {D_mat.shape}")
    print(f"  Dirac方程解范数: {np.linalg.norm(psi_sol):.4f}")

    # ============================================================
    # 10. 统计推断
    # ============================================================
    print_section("10. 统计推断与相变临界分析")
    inference = QGPStatisticalInference()

    # 中心度显著性
    t_stat, p_val = inference.centrality_significance(
        n_part_observed=350.0, n_part_mean=380.0, n_part_std=25.0, df=20.0
    )
    print(f"  参与者数偏离检验: t = {t_stat:.3f}, p = {p_val:.4f}")

    # v₂显著性
    t_v2, sig_v2 = inference.v2_significance(
        v2_observed=0.05, v2_stat_error=0.003, v2_systematic=0.005
    )
    print(f"  v₂信号显著性: {sig_v2:.2f}σ")

    # 临界温度置信区间
    T_lower, T_upper = inference.critical_temperature_confidence(
        T_measured=0.156, T_error=0.007, confidence=0.95
    )
    print(f"  临界温度 95% CI: [{T_lower:.4f}, {T_upper:.4f}] GeV")

    # 相变临界点搜索
    T_scan = np.linspace(0.12, 0.20, 50)
    # 模拟热 susceptibility (在T_c附近有峰)
    chi_true = 5.0 * np.exp(-((T_scan - 0.155) / 0.01) ** 2) + 0.5
    T_c, chi_max = inference.find_critical_point(T_scan, chi_true)
    print(f"  相变临界温度 T_c = {T_c:.4f} GeV")
    print(f"  最大 susceptibility = {chi_max:.2f}")

    # 非中心t分位数搜索
    t_q, iters = NonCentralTDistribution.quantile_search(
        p=0.95, df=30.0, delta=0.0, a=-5.0, b=5.0
    )
    print(f"  t₀.₉₅(30) = {t_q:.4f} (迭代 {iters} 次)")

    # ============================================================
    # 总结
    # ============================================================
    print_section("计算完成总结")
    elapsed = time.time() - start_time
    print(f"  总计算时间: {elapsed:.2f} 秒")
    print(f"  模拟系统: Au+Au @ √s_NN = 200 GeV")
    print(f"  碰撞参数: b = {b:.1f} fm")
    print(f"  初始峰值温度: {np.max(T_hist[0]):.3f} GeV")
    print(f"  拟合QGP参数:")
    print(f"    η/s = {eta_s_fit:.4f}")
    print(f"    c_s² = {cs2_fit:.4f}")
    print(f"    τ₀ = {tau0_fit:.3f} fm/c")
    print(f"  相变临界温度: T_c ≈ {T_c:.4f} GeV")
    print("=" * 70)
    print("QGP-Transport 模拟流程全部完成。")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
