"""
main.py — 计算高能物理: 异常事件检测与新物理信号搜索
              高阶有限差分与稳定性分析 (小规模可复现实验)

统一入口, 零参数可运行.

科学问题:
  在1+1维标量场论框架下, 通过高阶有限差分方法求解Klein-Gordon方程,
  分析数值格式的Von Neumann稳定性, 模拟包含BSM (Z'共振) 信号的
  Drell-Yan对撞事件, 经探测器模拟后, 使用Profile Likelihood Ratio、
  核密度异常检测和主动学习参数空间搜索等方法, 发现超出标准模型的新物理信号.

种子项目 (15个):
  270_dfield9     → 方向场/ODE → 时间演化积分器
  982_r8ge_np     → LU分解     → 隐式格式线性求解
  1000_dasayan05  → 随机过程   → 部分子簇射模拟
  095_bisection   → 二分法     → 共振质量搜索
  1097_catniplab  → 吸引子     → 稳定性盆地分析
  867_persistence → 持久统计   → Welford在线统计
  508_hb_to_mm    → 稀疏格式   → 差分模板索引
  034_asa082      → 行列式     → 稳定性矩阵分析
  1143_ALRLDA     → RLDA/AL    → 主动学习参数扫描
  1206_EMIT_SIM   → PDE演化    → 格点场时间步进
  470_gl_fast_rule→ GL正交     → 高精度数值积分
  501_hand_area   → MC面积     → 相空间积分
  002_advection   → 对流PDE    → 差分格式稳定性
  443_fn          → 特殊函数   → 散射截面计算
  1289_BABILong   → 噪声注入   → 探测器噪声建模
"""

import numpy as np
import math
import time
import sys


# =====================================================================
#  工具函数
# =====================================================================

def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def subsection(title):
    print(f"\n  --- {title} ---")


# =====================================================================
#  Phase 0: 物理常数与配置
# =====================================================================

def phase_0_setup():
    section("Phase 0: 物理常数与实验配置")

    constants = {
        'alpha_em': 1.0/137.036,
        'alpha_s_MZ': 0.1179,
        'mZ': 91.1876,
        'GammaZ': 2.4952,
        'mW': 80.379,
        'mH': 125.10,
        'sqrt_s': 13000.0,
    }

    lattice_params = {
        'N_x': 128,
        'h': 0.1,
        'mass': 2.0,
        'lam': 0.5,
        'fd_order': 4,
    }

    bsm_params = {
        'mZp': 500.0,
        'GammaZp': 15.0,
        'gp': 0.3,
    }

    exp_params = {
        'n_events': 200,
        'M_range': (60.0, 120.0),
        'n_bins': 15,
        'seed': 42,
    }

    print(f"  √s = {constants['sqrt_s']/1000:.0f} TeV")
    print(f"  mZ = {constants['mZ']:.2f} GeV, ΓZ = {constants['GammaZ']:.2f} GeV")
    print(f"  格点: N_x={lattice_params['N_x']}, h={lattice_params['h']}, m={lattice_params['mass']}")
    print(f"  BSM: mZ'={bsm_params['mZp']} GeV, g'={bsm_params['gp']}")
    print(f"  事件: {exp_params['n_events']}, 质量窗口: {exp_params['M_range']}")

    return constants, lattice_params, bsm_params, exp_params


# =====================================================================
#  Phase 1: 稳定性分析
# =====================================================================

def phase_1_stability(lattice_params):
    section("Phase 1: 高阶有限差分稳定性分析")

    from lattice_config import LatticeConfig
    from stability_analysis import stability_limit, amplification_factor, cfl_number
    from finite_difference import modified_wavenumber

    N_x = lattice_params['N_x']
    h = lattice_params['h']
    mass = lattice_params['mass']
    fd_order = lattice_params['fd_order']

    # 格点配置
    lat = LatticeConfig(N_x=N_x, h=h, mass=mass, lam=lattice_params['lam'])
    print(f"  {lat.summary()}")

    # 稳定性极限
    dt_max = stability_limit(h, mass, fd_order)
    dt_safe = 0.8 * dt_max
    cfl = cfl_number(h, dt_safe, fd_order)

    print(f"\n  CFL稳定性:")
    print(f"    dt_max = {dt_max:.6f}")
    print(f"    dt_safe (0.8×) = {dt_safe:.6f}")
    print(f"    CFL数 = {cfl:.4f}")

    # 不同阶数对比
    print(f"\n  差分阶数 vs dt_max:")
    for order in [2, 4, 6]:
        dt_o = stability_limit(h, mass, order)
        print(f"    {order}阶: dt_max = {dt_o:.6f}")

    # 色散关系
    k = np.linspace(0.01, np.pi/h, 50)
    k_tilde = modified_wavenumber(k, h, fd_order)
    omega = np.sqrt(k_tilde**2 + mass**2)
    v_g_num = np.gradient(omega, k)
    v_g_exact = k / np.sqrt(k**2 + mass**2)
    v_g_err = np.abs(v_g_num - v_g_exact) / np.maximum(np.abs(v_g_exact), 1e-10)

    print(f"\n  色散关系:")
    print(f"    群速度最大相对误差: {np.max(v_g_err):.4e}")
    print(f"    群速度平均相对误差: {np.mean(v_g_err):.4e}")

    # 放大因子
    G_mag, _ = amplification_factor(k, h, dt_safe, mass, fd_order)
    print(f"\n  放大因子 (dt={dt_safe:.4f}):")
    print(f"    max|G| = {np.max(G_mag):.10f}")
    print(f"    稳定性: {'稳定' if np.max(G_mag) <= 1.0+1e-10 else '不稳定'}")

    return dt_safe


# =====================================================================
#  Phase 2: 格点场演化
# =====================================================================

def phase_2_field_evolution(lattice_params, dt):
    section("Phase 2: 格点场论时间演化")

    from lattice_config import LatticeConfig
    from field_evolution import create_gaussian_packet, compute_energy, leapfrog_evolve

    N_x = lattice_params['N_x']
    h = lattice_params['h']
    mass = lattice_params['mass']
    lam = lattice_params['lam']

    lat = LatticeConfig(N_x=N_x, h=h, mass=mass, lam=lam)

    # 初始条件: 两个相向运动的高斯波包
    x = lat.x
    L = lat.L
    x1, x2 = L * 0.3, L * 0.7
    k0 = 1.0  # 动量
    sigma = 0.5

    phi1 = create_gaussian_packet(x, x1, k0, sigma, 1.0)
    phi2 = create_gaussian_packet(x, x2, -k0, sigma, 1.0)
    phi0 = phi1 + phi2
    phi_dot0 = -k0 * phi1 + k0 * phi2  # 时间导数

    print(f"  初始条件: 两个高斯波包")
    print(f"    位置: x1={x1:.2f}, x2={x2:.2f}")
    print(f"    动量: k0=±{k0}")
    print(f"    宽度: σ={sigma}")
    print(f"    最大|φ| = {np.max(np.abs(phi0)):.4f}")

    # 初始能量
    E0 = compute_energy(phi0, phi_dot0, h, mass, lam, lattice_params['fd_order'])
    print(f"    初始能量 E0 = {E0:.6f}")

    # Leapfrog演化
    N_t = 100
    dt_evo = min(dt, 0.5 * dt)  # 更保守
    print(f"\n  Leapfrog演化: N_t={N_t}, dt={dt_evo:.4f}")

    phi_final, phi_dot_final, E_hist, phi_max_hist = leapfrog_evolve(
        phi0, phi_dot0, N_x, h, dt_evo, N_t, mass, lam,
        fd_order=lattice_params['fd_order']
    )

    E_final = E_hist[-1]
    E_drift = abs(E_final - E0) / max(abs(E0), 1e-10)

    print(f"    最终能量 E_f = {E_final:.6f}")
    print(f"    能量漂移 |ΔE|/E0 = {E_drift:.2e}")
    print(f"    最大|φ| 终值 = {np.max(np.abs(phi_final)):.4f}")
    print(f"    能量守恒: {'良好' if E_drift < 0.01 else '需改善'}")

    return {
        'E_drift': E_drift,
        'phi_final': phi_final,
        'E_history': E_hist,
    }


# =====================================================================
#  Phase 3: 特殊函数验证
# =====================================================================

def phase_3_special_functions():
    section("Phase 3: 特殊函数库验证")

    from special_functions import (gamma_func, log_gamma, erf_func,
                                   breit_wigner, alpha_s_running,
                                   gauss_legendre_nodes_weights,
                                   phase_space_2body)

    # Gamma函数
    print("  Γ(x) 验证:")
    tests = [(0.5, math.sqrt(math.pi)), (1.0, 1.0), (2.0, 1.0)]
    for x, exact in tests:
        g = gamma_func(x)
        err = abs(g - exact) / exact
        print(f"    Γ({x}) = {g:.8f}, 精确 = {exact:.8f}, 误差 = {err:.2e}")

    # erf
    print("\n  erf(x) 验证:")
    for x in [0.0, 0.5, 1.0, 2.0]:
        e = erf_func(x)
        print(f"    erf({x}) = {e:.6f}")

    # Breit-Wigner
    print("\n  Breit-Wigner 共振:")
    mZ, GammaZ = 91.2, 2.5
    for dM in [-10, -5, 0, 5, 10]:
        M = mZ + dM
        bw = breit_wigner(M, mZ, GammaZ)
        print(f"    BW({M:.1f}) = {bw:.6e}")

    # α_s跑动
    print("\n  α_s(Q) 跑动:")
    for Q in [10, 50, 91.2, 200, 1000]:
        a_s = alpha_s_running(Q)
        print(f"    α_s({Q:5.1f} GeV) = {a_s:.6f}")

    # Gauss-Legendre
    print("\n  Gauss-Legendre 正交验证:")
    for n in [4, 8, 16]:
        nodes, weights = gauss_legendre_nodes_weights(n)
        # ∫_{-1}^{1} x^{2n-2} dx = 2/(2n-1)
        exact = 2.0 / (2*n - 1)
        computed = np.sum(weights * nodes**(2*n - 2))
        err = abs(computed - exact) / exact
        print(f"    n={n:2d}: 误差 = {err:.2e}")

    # 相空间
    print("\n  二体相空间:")
    phi2_ee = phase_space_2body(mZ**2, 0.000511, 0.000511)
    phi2_tt = phase_space_2body(mZ**2, 172.76, 172.76)
    print(f"    Φ₂(Z→ee) = {phi2_ee:.6f}")
    print(f"    Φ₂(Z→tt) = {phi2_tt:.6f} (运动学禁止)")


# =====================================================================
#  Phase 4: 事件生成
# =====================================================================

def phase_4_events(constants, bsm_params, exp_params):
    section("Phase 4: 事件生成与探测器模拟")

    from event_generator import EventGenerator
    from detector_sim import DetectorSimulation, compute_data_quality

    n_events = exp_params['n_events']
    M_range = exp_params['M_range']
    seed = exp_params['seed']

    # SM事件
    gen_sm = EventGenerator(constants['sqrt_s'], constants['mZ'],
                            constants['GammaZ'], seed)
    events_sm = gen_sm.generate_drell_yan(n_events, M_range[0], M_range[1])
    print(f"  SM事件: {len(events_sm)}")

    # BSM事件
    gen_bsm = EventGenerator(constants['sqrt_s'], constants['mZ'],
                             constants['GammaZ'], seed)
    events_bsm = gen_bsm.generate_drell_yan(
        n_events, M_range[0], M_range[1],
        bsm_params['mZp'], bsm_params['GammaZp'], bsm_params['gp']
    )
    print(f"  BSM事件: {len(events_bsm)}")

    # 探测器模拟
    det = DetectorSimulation(seed + 1)
    events_reco = det.process_events(events_bsm)

    n_reco = sum(1 for e in events_reco if e.get('reconstructed'))
    n_good = sum(1 for e in events_reco if compute_data_quality(e).get('good_event'))

    print(f"  重建通过: {n_reco}/{len(events_reco)} ({100*n_reco/max(len(events_reco),1):.1f}%)")
    print(f"  好事件: {n_good}/{len(events_reco)} ({100*n_good/max(len(events_reco),1):.1f}%)")

    # 只返回好事件
    good_events = [e for e in events_reco if compute_data_quality(e).get('good_event')]
    return good_events, events_sm


# =====================================================================
#  Phase 5: 异常检测
# =====================================================================

def phase_5_anomaly(events, events_sm, exp_params):
    section("Phase 5: 异常事件检测")

    from anomaly_detector import (WelfordAccumulator, RunningChiSquared,
                                  ProfileLikelihoodTest, KernelAnomalyDetector,
                                  AttentionWeightedScorer, AnomalyDetectionPipeline,
                                  significance_asimov)

    n_bins = exp_params['n_bins']
    M_range = exp_params['M_range']

    if len(events) < 5:
        print("  事件数不足, 跳过")
        return {}

    # Welford在线统计
    print("  在线统计 (Welford):")
    stats = WelfordAccumulator()
    for e in events:
        stats.update(e.get('M', 0))
    s = stats.summary()
    print(f"    N={s['n']}, mean={s['mean']:.2f}, std={s['std']:.2f}")

    # χ²检验
    M_data = np.array([e.get('M', 0) for e in events])
    M_bkg = np.array([e.get('M', 0) for e in events_sm])
    bins = np.linspace(M_range[0], M_range[1], n_bins + 1)
    hist_data, _ = np.histogram(M_data, bins=bins)
    hist_bkg, _ = np.histogram(M_bkg, bins=bins)

    chi2_test = RunningChiSquared(n_bins)
    chi2, ndf, p_chi2 = chi2_test.compute(
        hist_data, hist_bkg, np.maximum(hist_bkg, 1.0))
    print(f"\n  χ²检验: χ²={chi2:.2f}, ndf={ndf}, p={p_chi2:.4f}")

    # Profile Likelihood
    center = n_bins // 2
    signal_template = np.array([math.exp(-0.5*((i-center)/2.0)**2)
                                for i in range(n_bins)])
    signal_template *= max(np.sum(hist_data) - np.sum(hist_bkg), 1.0)
    signal_template = np.maximum(signal_template, 0.01)

    pl_test = ProfileLikelihoodTest(n_bins)
    q0, Z, p_local = pl_test.compute_test_statistic(
        hist_data, signal_template, hist_bkg.astype(float), syst=0.05)
    p_global = min(p_local * n_bins, 1.0)

    print(f"\n  Profile Likelihood:")
    print(f"    q₀ = {q0:.4f}")
    print(f"    Z = {Z:.3f} σ")
    print(f"    p_local = {p_local:.4e}")
    print(f"    p_global = {p_global:.4e}")

    # 核密度异常检测
    features = ['M', 'pT', 'rapidity', 'cos_theta']
    X_data = np.array([[e.get(f, 0) for f in features] for e in events])
    X_bkg = np.array([[e.get(f, 0) for f in features] for e in events_sm])

    kde = KernelAnomalyDetector()
    kde.fit(X_bkg)
    scores = kde.score_batch(X_data)

    print(f"\n  核密度异常检测:")
    print(f"    平均分数: {np.mean(scores):.3f}")
    print(f"    最大分数: {np.max(scores):.3f}")

    top_k = min(5, len(scores))
    top_idx = np.argsort(scores)[-top_k:][::-1]
    print(f"    Top-{top_k}异常事件:")
    for i, idx in enumerate(top_idx):
        print(f"      #{i+1}: score={scores[idx]:.3f}, M={events[idx].get('M',0):.1f}")

    # 注意力加权
    att = AttentionWeightedScorer(len(features))
    att.fit(X_bkg)
    att_scores, att_weights = att.score_batch(X_data)
    print(f"\n  注意力加权:")
    print(f"    权重: {att_weights}")
    print(f"    平均分数: {np.mean(att_scores):.3f}")

    # 综合流水线
    pipeline = AnomalyDetectionPipeline(n_bins, M_range)
    pipeline.fit_background(events_sm)
    results = pipeline.analyze(events)

    print(f"\n  综合结果:")
    print(f"    全局p值: {results['global_pvalue']:.4e}")
    sig = results['significance']
    print(f"    显著性: Z={sig[1]:.3f}σ, p={sig[2]:.3e}")

    return results


# =====================================================================
#  Phase 6: 主动学习
# =====================================================================

def phase_6_active_learning(constants):
    section("Phase 6: 主动学习参数空间探索")

    from active_learning import ActiveLearningExplorer, sherman_morrison_update
    from event_generator import PartonDistribution
    from anomaly_detector import significance_asimov

    def objective(theta):
        M_Zp, g_Zp = theta[0], theta[1]
        width = 0.03 * M_Zp
        pdf = PartonDistribution(Q=M_Zp)
        L = pdf.parton_luminosity(M_Zp, 'u')
        s = g_Zp**2 * L * 500
        b = L * 50 + 5.0
        return significance_asimov(max(s, 0.01), max(b, 0.01))

    param_bounds = [(200.0, 800.0), (0.05, 0.8)]

    explorer = ActiveLearningExplorer(2, param_bounds, objective, seed=42)
    results = explorer.run_exploration(n_initial=6, n_iterations=8, n_candidates=20)

    print(f"  总评估点: {len(results['points'])}")
    print(f"  初始点: {results['n_initial']}")
    print(f"  迭代: {results['n_iterations']}")

    scores = results['scores']
    best_idx = np.argmax(scores)
    best_pt = results['points'][best_idx]
    print(f"\n  最优点:")
    print(f"    MZ' = {best_pt[0]:.1f} GeV, g' = {best_pt[1]:.4f}")
    print(f"    Z = {scores[best_idx]:.3f} σ")

    # Sherman-Morrison验证
    n_test = 4
    rng = np.random.RandomState(42)
    A = rng.randn(n_test, n_test)
    A = A @ A.T + np.eye(n_test) * 2
    A_inv = np.linalg.inv(A)
    u = rng.randn(n_test)
    v = rng.randn(n_test)
    A_inv_sm = sherman_morrison_update(A_inv, u, v)
    A_inv_exact = np.linalg.inv(A + np.outer(u, v))
    err = np.linalg.norm(A_inv_sm - A_inv_exact) / np.linalg.norm(A_inv_exact)
    print(f"\n  Sherman-Morrison验证:")
    print(f"    相对误差: {err:.2e}")

    return results


# =====================================================================
#  Phase 7: 共振搜索
# =====================================================================

def phase_7_resonance_search(constants):
    section("Phase 7: 二分法共振质量搜索")

    from resonance_finder import find_resonance_mass
    from event_generator import PartonDistribution
    from anomaly_detector import significance_asimov

    def compute_Z(M_Zp):
        g = 0.3
        width = 0.03 * M_Zp
        pdf = PartonDistribution(Q=M_Zp)
        L = pdf.parton_luminosity(M_Zp, 'u')
        s = g**2 * L * 500
        b = L * 50 + 5.0
        return significance_asimov(max(s, 0.01), max(b, 0.01))

    target_Z = 3.0
    M_res, info = find_resonance_mass(compute_Z, target_Z, (100, 2000), tol=1.0)

    print(f"  目标: Z = {target_Z}σ")
    print(f"  搜索结果: MZ' = {M_res:.1f} GeV")
    print(f"  收敛: {info.get('converged', False)}")
    if info.get('converged'):
        print(f"  迭代: {info.get('iterations', 0)}")
        print(f"  残差: {info.get('residual', 0):.4f}")

    # 质量扫描
    print(f"\n  质量扫描:")
    M_scan = np.linspace(100, 2000, 20)
    for M in M_scan[::5]:
        Z = compute_Z(M)
        bar = '#' * int(min(Z * 3, 40))
        print(f"    M={M:6.0f} GeV: Z={Z:.3f}σ {bar}")


# =====================================================================
#  Phase 8: Monte Carlo相空间
# =====================================================================

def phase_8_monte_carlo():
    section("Phase 8: Monte Carlo相空间积分")

    from monte_carlo import mc_integrate_2body, mc_integrate_3body, mc_cross_section

    mZ = 91.2
    me, mmu, mt = 0.000511, 0.10566, 172.76

    # 二体
    print("  二体相空间:")
    phi2_ee, _ = mc_integrate_2body(mZ**2, me, me)
    phi2_mumu, _ = mc_integrate_2body(mZ**2, mmu, mmu)
    phi2_tt, _ = mc_integrate_2body(mZ**2, mt, mt)
    print(f"    Φ₂(Z→ee) = {phi2_ee:.6f}")
    print(f"    Φ₂(Z→μμ) = {phi2_mumu:.6f}")
    print(f"    Φ₂(Z→tt) = {phi2_tt:.6f} (禁止)")

    # 三体
    print("\n  三体相空间 (MC):")
    phi3, err3 = mc_integrate_3body(mZ**2, me, me, me, n_samples=10000)
    print(f"    Φ₃(Z→3e) = {phi3:.4f} ± {err3:.4f}")

    # 截面
    print("\n  MC截面积分:")
    sigma, sigma_err = mc_cross_section(13000, 60, 120, n_events=5000)
    print(f"    σ(DY) = {sigma:.4e} ± {sigma_err:.4e}")


# =====================================================================
#  主函数
# =====================================================================

def main():
    start_time = time.time()

    print("=" * 70)
    print("  计算高能物理: 异常事件检测与新物理信号搜索")
    print("  高阶有限差分与稳定性分析 (小规模可复现实验)")
    print("=" * 70)
    print(f"\n  Python {sys.version.split()[0]}, NumPy {np.__version__}")

    # Phase 0
    constants, lattice_params, bsm_params, exp_params = phase_0_setup()

    # Phase 1
    dt = phase_1_stability(lattice_params)

    # Phase 2
    field_results = phase_2_field_evolution(lattice_params, dt)

    # Phase 3
    phase_3_special_functions()

    # Phase 4
    events, events_sm = phase_4_events(constants, bsm_params, exp_params)

    # Phase 5
    anomaly_results = phase_5_anomaly(events, events_sm, exp_params)

    # Phase 6
    al_results = phase_6_active_learning(constants)

    # Phase 7
    phase_7_resonance_search(constants)

    # Phase 8
    phase_8_monte_carlo()

    # 总结
    elapsed = time.time() - start_time
    section("计算完成")
    print(f"\n  总运行时间: {elapsed:.2f} 秒")
    print(f"  能量漂移: {field_results['E_drift']:.2e}")
    print(f"  好事件: {len(events)}")
    if anomaly_results:
        sig = anomaly_results.get('significance', (0, 0, 1))
        print(f"  显著性: Z={sig[1]:.2f}σ")
        print(f"  全局p值: {anomaly_results.get('global_pvalue', 1):.4e}")
    print(f"\n  融合15个种子项目的核心算法完成.")
    print(f"\n{'='*70}")


if __name__ == '__main__':
    main()
