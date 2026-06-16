"""
main.py — 格点 QCD 强子谱关联函数拟合: 高阶有限差分与稳定性分析
=================================================================
主入口文件, 零参数可运行.

完整流程:
  1. 构建格点几何
  2. 初始化规范场配置 (U(1) 简化版用于小规模实验)
  3. Wilson 梯度流演化
  4. 构造 Dirac 算子
  5. 生成合成关联函数数据
  6. 多指数拟合提取强子质量
  7. 高阶有限差分色散关系分析
  8. 稳定性分析 (t_min 扫描, 态数目扫描, 交叉验证)
  9. Bootstrap/Jackknife 误差分析
 10. 谱函数重构
 11. 有限体积修正
 12. 综合报告

种子项目融合映射:
  [995_r8sm]     → 传播子 Sherman-Morrison 秩-1 修正
  [1158]         → 神经网络辅助质量提取 + 集成预测
  [916]          → 高阶求积规则 → 动量空间积分
  [185]          → 超球邻居枚举 → 格点距离场
  [1208]         → 张量分解 → 多算符关联矩阵
  [1422]         → 坐标/链接 I/O
  [1086]         → 变分法 + 多起始 L-BFGS-B 拟合
  [925]          → 分段线性 → 质量-跳跃参数插值
  [305]          → 距离函数 → 空间关联
  [1378]         → Voronoi → 源构造 (简化为分桶)
  [1065]         → 自助训练 → Bootstrap 增强
  [328]          → 椭圆积分 → Lüscher zeta 函数
  [945]          → 梯形积分 → 谱表示积分
  [913]          → 并行素数 → 并行配置管理
  [121]          → ODE → Wilson 梯度流
"""

import sys
import os
import time
import numpy as np

# 确保模块路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def section_header(title: str, width: int = 72):
    """打印章节标题."""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_result(key: str, value, indent: int = 2):
    """格式化打印结果."""
    prefix = " " * indent
    if isinstance(value, float):
        print(f"{prefix}{key}: {value:.6f}")
    elif isinstance(value, np.ndarray):
        if value.size <= 6:
            print(f"{prefix}{key}: {value}")
        else:
            print(f"{prefix}{key}: array[{value.shape}]")
    elif isinstance(value, (list, tuple)):
        if len(value) <= 6:
            print(f"{prefix}{key}: {value}")
        else:
            print(f"{prefix}{key}: list[{len(value)}]")
    else:
        print(f"{prefix}{key}: {value}")


def main():
    """主函数: 零参数运行完整格点 QCD 分析流程."""
    start_time = time.time()

    print("=" * 72)
    print("  格点 QCD: 强子谱关联函数拟合")
    print("  高阶有限差分与稳定性分析 (小规模可复现实验)")
    print("=" * 72)

    # ==================================================================
    # 1. 格点几何构建 (源自 [185] + [305] + [1422])
    # ==================================================================
    section_header("1. 格点几何构建")
    from lattice_geometry import LatticeGeometry, compute_distance_matrix

    Ls, Lt = 4, 12
    geo = LatticeGeometry(Ls=Ls, Lt=Lt, ndim=4)
    print(f"  格点: Ls={Ls}, Lt={Lt}, V={geo.volume}")
    print(f"  形状: {geo.shape}")

    # 距离矩阵 (源自 [305])
    dist = compute_distance_matrix(geo)
    print(f"  距离矩阵: max={np.max(dist):.3f}, mean={np.mean(dist):.3f}")

    # 超球邻居 (源自 [185])
    origin = np.zeros(4, dtype=np.int64)
    neighbors = geo.sphere_neighbors(origin, radius=1.5)
    print(f"  半径 1.5 内邻居数: {len(neighbors)}")

    # 连通性 I/O (源自 [1422])
    io_path = os.path.join(os.path.dirname(__file__), 'lattice_io.json')
    geo.save_connectivity(io_path)
    print(f"  连通性已保存到: {os.path.basename(io_path)}")

    # ==================================================================
    # 2. 规范场与 Wilson 流 (源自 [121] + [1065])
    # ==================================================================
    section_header("2. 规范场与 Wilson 梯度流")
    from gauge_wilson_flow import (GaugeField, integrate_wilson_flow,
                                    compute_t0_energy)

    gauge = GaugeField(geo, Nc=2, config_type='cold', beta=6.0, seed=236)
    avg_plaq = gauge.average_plaquette()
    print(f"  初始平均 plaquette: {avg_plaq:.6f}")

    # Wilson 流 (源自 [121] 的 ODE 积分)
    t_flow, E_flow = integrate_wilson_flow(gauge, t_max=0.3, dt=0.05,
                                            method='RK3')
    print(f"  Wilson 流: {len(t_flow)} 步, t_max={t_flow[-1]:.2f}")
    if np.any(E_flow > 0):
        t0 = compute_t0_energy(E_flow, t_flow, target=0.01)
        print(f"  t^2 E(t) 范围: [{np.min(t_flow**2*E_flow):.6f}, "
              f"{np.max(t_flow**2*E_flow):.6f}]")
        print(f"  t_0 估计 (target=0.01): {t0:.4f}")
    else:
        print("  冷起始: 能量密度为零 (平凡配置)")

    # ==================================================================
    # 3. Dirac 算子 (源自 [1086] + [995] + [925])
    # ==================================================================
    section_header("3. Wilson Dirac 算子")
    from dirac_operator import WilsonDiracOperator

    kappa = 0.12
    dirac = WilsonDiracOperator(geo, gauge, kappa=kappa, c_sw=0.0)
    print(f"  {dirac}")
    print(f"  kappa_c = {dirac.kappa_c:.4f}, kappa/kappa_c = "
          f"{kappa/dirac.kappa_c:.3f}")

    # 质量-跳跃参数插值 (源自 [925])
    mass_grid = np.array([0.5, 0.3, 0.15, 0.08, 0.03])
    kappa_grid = 1.0 / (2.0 * (mass_grid + 4.0))
    target_mass = 0.2
    kappa_interp = WilsonDiracOperator.interpolate_mass_to_kappa(
        mass_grid, kappa_grid, target_mass)
    print(f"  质量插值: m={target_mass} → kappa={kappa_interp:.5f}")

    # Sherman-Morrison 传播子更新 (源自 [995])
    # 使用小型子格点用于演示
    sc_dim = dirac.spin_color_dim
    V_small = min(geo.volume, 4)  # 小型子格点
    V_total = V_small * sc_dim
    S0 = np.eye(V_total, dtype=np.complex128) * 0.1
    delta_U = 0.01 * np.eye(2, dtype=np.complex128)
    S_updated = dirac.propagator_rank1_update(S0, 0, 0, delta_U)
    diff_norm = np.linalg.norm(S_updated - S0)
    print(f"  Sherman-Morrison 修正: ||delta_S|| = {diff_norm:.6e}")

    # ==================================================================
    # 4. 关联函数构造 (源自 [1086] + [1208])
    # ==================================================================
    section_header("4. 强子关联函数构造")
    from correlator_observable import HadronCorrelator

    # 单算符关联函数
    corr = HadronCorrelator(geo, hadron_type='pion')
    true_masses = [0.4, 0.9, 1.5]
    true_amps = [1.0, 0.3, 0.05]
    corr.generate_synthetic_correlator(true_masses, true_amps,
                                        noise_level=0.01, seed=236)
    print(f"  {corr}")
    print(f"  真实质量: {true_masses}")
    print(f"  C(t=1)={corr.C[1]:.6f}, C(t=6)={corr.C[6]:.6f}")

    # 有效质量
    t_arr, m_eff = corr.effective_mass(method='cosh')
    valid = ~np.isnan(m_eff)
    if np.any(valid):
        plateau = m_eff[3:8]
        plateau_valid = plateau[~np.isnan(plateau)]
        if len(plateau_valid) > 0:
            print(f"  有效质量 plateau (t=3..7): "
                  f"{np.mean(plateau_valid):.4f} ± {np.std(plateau_valid):.4f}")

    # 多算符关联矩阵 (源自 [1208] 的张量分解思想)
    corr_multi = HadronCorrelator(geo, hadron_type='pion')
    corr_multi.generate_multi_op_correlator(n_ops=3, masses=true_masses,
                                             noise_level=0.01, seed=236)
    print(f"  多算符基组: {corr_multi.n_operators} 个算符")

    # 变分分析 (GEVP)
    try:
        eigvals, energies = corr_multi.variational_analysis(t_0=2)
        print(f"  GEVP 本征值: {eigvals[:3]}")
        print(f"  GEVP 能量估计: {energies[:3]}")
    except Exception as e:
        print(f"  GEVP 分析: {e}")

    # 空间关联 (源自 [305])
    r_vals, C_r = corr.spatial_correlator()
    print(f"  空间关联: {len(r_vals)} 个距离壳层")

    # ==================================================================
    # 5. 关联函数拟合 (源自 [1086] + [1158])
    # ==================================================================
    section_header("5. 多指数关联函数拟合")
    from correlator_fitting import CorrelatorFitter

    fitter = CorrelatorFitter(t_arr, corr.C, C_errors=corr.C_errors)
    print(f"  {fitter}")

    # 单态拟合
    res_1 = fitter.fit_single_state(t_min=2, E_init=0.5, n_restarts=5)
    print(f"  单态拟合: E = {res_1['E']:.4f} (真值 {true_masses[0]:.4f}), "
          f"chi2/dof = {res_1['chi2_dof']:.2f}")

    # 双态拟合
    res_2 = fitter.fit_multi_state(n_states=2, t_min=2, n_restarts=5)
    print(f"  双态拟合: E_0 = {res_2['masses'][0]:.4f}, "
          f"E_1 = {res_2['masses'][1]:.4f}")
    print(f"  chi2/dof = {res_2['chi2_dof']:.2f}")

    # 神经网络辅助 (源自 [1158])
    nn_res = fitter.neural_network_mass_extraction()
    print(f"  NN 辅助估计: m = {nn_res['mass_estimate']:.4f}")

    # ==================================================================
    # 6. 高阶有限差分与色散关系 (源自 [916] + [945])
    # ==================================================================
    section_header("6. 高阶有限差分与色散关系")
    from finite_diff_stencil import (improved_lattice_derivative,
                                      dispersion_relation,
                                      symanzik_improvement_coefficients,
                                      compute_discretization_error,
                                      tadpole_improvement_factor)

    # Symanzik 改善系数
    sym_coeffs = symanzik_improvement_coefficients(max_order=4)
    for K, c in sym_coeffs:
        print(f"  K={K}: c = [{', '.join(f'{ci:.6f}' for ci in c)}]")

    # 色散关系
    p_test = np.linspace(0.01, np.pi, 50)
    for K in [1, 2, 3, 4]:
        p_hat_sq = dispersion_relation(p_test, half_width=K)
        p_true_sq = p_test ** 2
        max_err = np.max(np.abs(p_hat_sq - p_true_sq) / p_true_sq)
        print(f"  色散关系 K={K}: 最大相对误差 = {max_err:.2e}")

    # Tadpole 改善
    u0 = tadpole_improvement_factor(max(avg_plaq, 0.01))
    print(f"  Tadpole 因子 u_0 = {u0:.4f}")

    # 离散化误差阶数
    a_values = np.array([0.2, 0.15, 0.1, 0.08])
    masses_sim = np.array([0.42, 0.41, 0.405, 0.402])
    est_order = compute_discretization_error(masses_sim, a_values, 0.4)
    print(f"  离散化误差阶数: p ≈ {est_order:.2f}")

    # ==================================================================
    # 7. 稳定性分析 (源自 [121] + [1065])
    # ==================================================================
    section_header("7. 拟合稳定性分析")
    from stability_analysis import StabilityAnalyzer

    analyzer = StabilityAnalyzer(fitter)

    # t_min 扫描
    tmin_res = analyzer.t_min_scan(t_min_start=1, t_min_end=5,
                                    n_states=1, n_restarts=3)
    print(f"  t_min 扫描: {len(tmin_res['t_min_values'])} 个点")
    print(f"    质量范围: [{np.min(tmin_res['masses']):.4f}, "
          f"{np.max(tmin_res['masses']):.4f}]")
    print(f"    稳定性评分: {tmin_res['stability_score']:.4f}")

    # 态数目扫描
    nstate_res = analyzer.n_state_scan(max_states=3, t_min=2, n_restarts=3)
    print(f"  态数目扫描:")
    for i, (n, m, chi2) in enumerate(zip(
            nstate_res['n_states'],
            nstate_res['ground_masses'],
            nstate_res['chi2_dof'])):
        print(f"    N={n}: m_0={m:.4f}, chi2/dof={chi2:.2f}, "
              f"w_Akaike={nstate_res['akaike_weights'][i]:.3f}")
    print(f"    模型平均质量: {nstate_res['model_averaged_mass']:.4f}")

    # 条件数分析
    cond_res = analyzer.correlation_matrix_analysis()
    print(f"  相关矩阵条件数: {cond_res['condition_number']:.2e}")
    print(f"  条件良好: {cond_res['is_well_conditioned']}")

    # 交叉验证
    cv_res = analyzer.cross_validate_fit(n_folds=3, t_min=2)
    print(f"  交叉验证: 质量 = {cv_res['mean_mass']:.4f} "
          f"± {cv_res['std_mass']:.4f}, CV-MSE = {cv_res['cv_mse']:.2e}")

    # ==================================================================
    # 8. Bootstrap/Jackknife 误差 (源自 [1065])
    # ==================================================================
    section_header("8. Bootstrap/Jackknife 统计误差")
    from bootstrap_errors import BootstrapAnalyzer
    from parallel_config import ParallelConfigManager

    # 生成合成配置集合
    pcm = ParallelConfigManager(n_configs=15, seed_base=236, n_workers=2)
    C_ens = pcm.generate_synthetic_ensemble(
        Lt=Lt, true_mass=true_masses[0], noise_level=0.02,
        autocorrelation=0.3, seed=236)
    print(f"  配置集合: {C_ens.shape[0]} 个配置, {C_ens.shape[1]} 个时间片")

    boot_analyzer = BootstrapAnalyzer(C_ens, t_arr)

    # 自相关
    ac_res = boot_analyzer.autocorrelation_analysis()
    print(f"  积分自相关时间: tau_int = {ac_res['tau_int']:.2f}")
    print(f"  有效统计量: N_eff = {ac_res['N_eff']:.1f}")

    # Bootstrap
    boot_res = boot_analyzer.bootstrap_fit(n_bootstrap=80, t_min=2, seed=42)
    print(f"  Bootstrap: m = {boot_res['mean_mass']:.4f} "
          f"± {boot_res['std_mass']:.4f}")
    print(f"  95% CI: [{boot_res['confidence_interval'][0]:.4f}, "
          f"{boot_res['confidence_interval'][1]:.4f}]")

    # Jackknife
    jk_res = boot_analyzer.jackknife_fit(t_min=2, seed=42)
    print(f"  Jackknife: m = {jk_res['mass_jk']:.4f} "
          f"± {jk_res['sigma_jk']:.4f}")

    # 自训练增强 (源自 [1065])
    st_res = boot_analyzer.self_training_mass_enhancement(
        n_labeled=5, n_rounds=2, confidence_threshold=0.5, seed=42)
    print(f"  自训练: 标记 {st_res['final_labeled_count']}/{C_ens.shape[0]}, "
          f"伪标签 {st_res['pseudo_label_count']}")

    # ==================================================================
    # 9. 动量空间积分 (源自 [916] + [945])
    # ==================================================================
    section_header("9. 动量空间积分")
    from momentum_integration import (brillouin_zone_integral_1d,
                                       lattice_tadpole_integral,
                                       spectral_representation_integral)

    # 1D Brillouin 区积分
    def test_integrand(p):
        return 1.0 / (4.0 * np.sin(p / 2) ** 2 + 0.1)

    for method in ['gauss', 'simpson', 'trapezoid']:
        val = brillouin_zone_integral_1d(test_integrand, n_points=32,
                                          method=method)
        print(f"  Brillouin 积分 ({method}): {val:.6f}")

    # 格点 tadpole 积分
    tad_1d = lattice_tadpole_integral(mass_sq=0.16, n_points=16, ndim=1)
    tad_2d = lattice_tadpole_integral(mass_sq=0.16, n_points=16, ndim=2)
    print(f"  Tadpole 积分 1D: {tad_1d:.6f}")
    print(f"  Tadpole 积分 2D: {tad_2d:.6f}")

    # 谱表示
    omega = np.linspace(0.01, 3.0, 60)
    rho_test = np.exp(-(omega - 0.4) ** 2 / 0.1) + 0.3 * np.exp(
        -(omega - 0.9) ** 2 / 0.2)
    t_spec = np.arange(1, 8, dtype=np.float64)
    C_spec = spectral_representation_integral(rho_test, omega, t_spec)
    print(f"  谱表示: {len(t_spec)} 个时间片, "
          f"C(t=1)={C_spec[0]:.4f}")

    # ==================================================================
    # 10. 谱函数重构 (源自 [945] + [1158])
    # ==================================================================
    section_header("10. 谱函数重构 (最大熵方法)")
    from spectral_analysis import SpectralReconstructor

    # 使用合成的关联函数
    t_fit = np.arange(1, 8, dtype=np.float64)
    C_fit = corr.C[1:8]
    C_err = np.maximum(corr.C_errors[1:8], 1e-6)

    spec = SpectralReconstructor(t_fit, C_fit, C_err,
                                  omega_max=3.0, n_omega=60)
    mem_res = spec.reconstruct_mem(alpha=0.5, n_iterations=100)
    print(f"  MEM 重构: chi2 = {mem_res['chi2']:.2f}, "
          f"S = {mem_res['entropy']:.4f}")

    # 求和规则
    sr = spec.sum_rule_check(mem_res['rho'])
    print(f"  谱函数峰值位置: omega = {sr['peak_position']:.3f}")
    print(f"  零阶矩: {sr['zeroth_moment']:.4f}")
    print(f"  非负性: {sr['positive_definite']}")

    # Alpha 扫描
    alpha_res = spec.alpha_scan(alpha_values=np.logspace(-1, 1, 4),
                                 n_iterations=50)
    print(f"  最优 alpha: {alpha_res['optimal_alpha']:.3f}")

    # ==================================================================
    # 11. 有限体积修正 (源自 [328])
    # ==================================================================
    section_header("11. 有限体积修正与 Lüscher 分析")
    from finite_volume import (carlson_RF, carlson_RD, ellipse_perimeter,
                                luscher_zeta_00, luscher_delta_E,
                                finite_volume_correction,
                                infinite_volume_extrap)

    # Carlson 椭圆积分
    rf_val = carlson_RF(1.0, 2.0, 3.0)
    rd_val = carlson_RD(1.0, 2.0, 3.0)
    print(f"  Carlson RF(1,2,3) = {rf_val.real:.6f}")
    print(f"  Carlson RD(1,2,3) = {rd_val.real:.6f}")

    # 椭圆周长
    peri = ellipse_perimeter(2.0, 1.0)
    print(f"  椭圆周长 (a=2, b=1): {peri:.4f}")

    # Lüscher zeta 函数
    Z00 = luscher_zeta_00(q_sq=0.1, n_max=10)
    print(f"  Lüscher Z_00(1; 0.1) = {Z00:.4f}")

    # 有限体积能移
    dE = luscher_delta_E(L=float(Ls), m=true_masses[0], a0=0.1)
    print(f"  有限体积能移 (L={Ls}): delta_E = {dE:.6f}")

    # 质量修正
    dm = finite_volume_correction(true_masses[0], float(Ls))
    print(f"  有限体积质量修正: delta_m/m = "
          f"{dm/true_masses[0]:.4e}")

    # 无限体积外推
    L_values = np.array([4, 6, 8, 10], dtype=np.float64)
    masses_L = np.array([true_masses[0] +
                          finite_volume_correction(true_masses[0], L)
                          for L in L_values])
    fv_res = infinite_volume_extrap(masses_L, L_values, true_masses[0])
    print(f"  无限体积外推: m_inf = {fv_res['m_inf']:.4f}, "
          f"shift = {fv_res['finite_volume_shift']:.2e}")

    # ==================================================================
    # 12. 并行计算 (源自 [913] + [1065])
    # ==================================================================
    section_header("12. 并行计算工具")

    # 并行素数计数
    n_primes = ParallelConfigManager.count_primes_parallel(1000, n_workers=2)
    print(f"  素数计数 (N=1000): {n_primes}")

    # Hasenbusch 质量
    hb_masses = ParallelConfigManager.hasenbusch_masses(
        n_masses=3, m_light=0.01, m_heavy=1.0)
    print(f"  Hasenbusch 质量: {hb_masses}")

    # 并行 bootstrap
    par_boot = pcm.parallel_bootstrap_fit(C_ens, t_arr,
                                           n_bootstrap=40, t_min=2)
    print(f"  并行 Bootstrap: m = {par_boot['mean_mass']:.4f} "
          f"± {par_boot['std_mass']:.4f}")

    # ==================================================================
    # 综合结果汇总
    # ==================================================================
    section_header("综合结果汇总")
    print(f"  格点: {Ls}^3 x {Lt}, V = {geo.volume}")
    print(f"  真实基态质量: {true_masses[0]:.4f}")
    print(f"  ---")
    print(f"  拟合方法对比:")
    print(f"    单态拟合:     {res_1['E']:.4f}")
    print(f"    双态拟合:     {res_2['masses'][0]:.4f}")
    print(f"    NN 辅助:      {nn_res['mass_estimate']:.4f}")
    print(f"    Bootstrap:    {boot_res['mean_mass']:.4f} "
          f"± {boot_res['std_mass']:.4f}")
    print(f"    Jackknife:    {jk_res['mass_jk']:.4f} "
          f"± {jk_res['sigma_jk']:.4f}")
    print(f"    模型平均:     {nstate_res['model_averaged_mass']:.4f}")
    print(f"    无限体积:     {fv_res['m_inf']:.4f}")
    print(f"  ---")
    print(f"  稳定性: {tmin_res['stability_score']:.3f} "
          f"({'稳定' if tmin_res['stability_score'] < 1.0 else '需关注'})")
    print(f"  条件数: {cond_res['condition_number']:.2e} "
          f"({'良好' if cond_res['is_well_conditioned'] else '病态'})")

    elapsed = time.time() - start_time
    print(f"\n  总计算时间: {elapsed:.2f} 秒")
    print("=" * 72)
    print("  格点 QCD 分析完成")
    print("=" * 72)

    # 清理临时文件
    if os.path.exists(io_path):
        os.remove(io_path)

    return 0


if __name__ == '__main__':
    sys.exit(main())
