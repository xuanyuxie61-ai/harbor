"""
main.py
=======
统一入口: 电池电极材料离子扩散模拟
高阶有限差分与稳定性分析 (小规模可复现实验)

PROJECT_281 合成项目

融合种子项目:
  - 053_asa266: 统计特殊函数 (正态CDF, digamma, trigamma)
  - 1297_FormalCellular: 形式化验证框架
  - 122_buckling_spring: 参数空间稳定性分析
  - 892_polyiamonds: 组合枚举与邻域操作
  - 164_chebyshev1_exactness: Chebyshev 求积精确性
  - 1064_ajakef_Earthquake_Infrasound: 波形信号分析
  - 536_hilbert_curve_3d: 3D Hilbert 曲线映射
  - 1127_nschawor_eeg-mu-alpha: 频谱分析与特征提取
  - 198_collatz_polynomial: 迭代映射动力学
  - 381_fem_to_triangle: 有限元网格思想
  - 531_hexahedron_jaskowiec_rule: 高阶求积规则
  - 568_i4lib: 整数工具函数库
  - 938_qr_solve: QR 分解与线性求解
  - 1405_web_matrix: 幂法与转移矩阵分析
  - 1081_FranciscoHS_toy-model-cis-code: 物理模型参数化

科学问题:
  NMC-622 正极颗粒中 Li⁺ 离子的非线性扩散模拟。
  使用 4 阶紧致有限差分空间离散 + Crank-Nicolson 隐式时间积分,
  配合自适应步长控制与完整的 von Neumann 稳定性分析。

运行: python main.py
"""

import sys
import math
import numpy as np

# 导入所有模块
from electrode_constants import (
    N_GRID, N_TIME_STEPS, DT_INITIAL, T_REF, C_MAX,
    PARTICLE_RADIUS, D_REF, E_ACT, R_GAS, FARADAY,
    arrhenius_diffusivity, thermal_voltage, dimensionless_groups,
    erf_approx, normal_cdf, digamma_approx, trigamma_approx
)
from thermodynamic_models import (
    redlich_kister_ocv, ocv_derivative, thermodynamic_factor,
    margules_activity_coefficient, concentration_dependent_D,
    chemical_potential, spinodal_boundaries
)
from compact_finite_difference import (
    compact_fd_2nd_derivative_matrix, modified_wavenumber_compact,
    chebyshev_nodes_interval, chebyshev_quadrature_exactness,
    spectral_radius_diffusion_operator
)
from stability_analysis import (
    von_neumann_ftcs, von_neumann_crank_nicolson,
    von_neumann_compact_crank_nicolson,
    stability_boundary_parameter_space,
    cfl_condition, stiffness_ratio, power_method_eigenvalue,
    amplification_matrix, adaptive_timestep
)
from time_integrator import (
    TimeIntegratorState, crank_nicolson_step, compute_new_timestep
)
from boundary_conditions import (
    butler_volmer_current, overpotential,
    apply_boundary_conditions, mass_conservation_check
)
from crystal_lattice import (
    NMCCrystalStructure, PolycrystalGrainNetwork,
    hilbert_3d_h_to_xyz, hilbert_3d_xyz_to_h
)
from combinatorial_enumeration import (
    i4_choose, i4_choose_log, i4_is_prime,
    configuration_entropy, enumerate_lattice_configurations,
    ijk_neighbors, collatz_lattice_step
)
from matrix_eigenvalue import (
    qr_factorization, qr_solve_linear, qr_eigenvalue_iteration,
    markov_transition_analysis, condition_number_estimate
)
from signal_analysis import (
    power_spectral_density, extract_relaxation_times,
    concentration_autocorrelation, multiscale_decomposition
)
from diagnostic_output import (
    concentration_profile_stats, mass_conservation_detailed,
    energy_balance_check, convergence_diagnostics,
    format_diagnostic_report
)
from diffusion_simulation import (
    run_diffusion_simulation, validate_simulation,
    create_radial_grid, initial_concentration_profile
)


def print_header():
    """打印项目头信息。"""
    print("=" * 72)
    print("  PROJECT 281: 电池电极材料离子扩散模拟")
    print("  高阶有限差分与稳定性分析 (小规模可复现实验)")
    print("  博士级计算材料科学合成项目")
    print("=" * 72)
    print()


def section(title):
    """分节标题。"""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def phase1_physical_constants():
    """阶段 1: 物理常数与材料参数验证。"""
    section("阶段 1: 物理常数与 Arrhenius 扩散系数")

    # Arrhenius 扩散系数
    T_values = [273.15, 298.15, 323.15, 348.15, 373.15]
    print(f"\n  {'T [K]':>8s}  {'D [m²/s]':>14s}  {'V_T [mV]':>10s}")
    print(f"  {'─' * 8}  {'─' * 14}  {'─' * 10}")
    for T in T_values:
        D = arrhenius_diffusivity(D_REF, E_ACT, T)
        V_T = thermal_voltage(T) * 1000  # mV
        print(f"  {T:8.2f}  {D:14.4e}  {V_T:10.4f}")

    # 无量纲数群
    dim_groups = dimensionless_groups(T_REF, PARTICLE_RADIUS)
    print(f"\n  无量纲数群 (T={T_REF}K, L={PARTICLE_RADIUS*1e6:.1f}μm):")
    print(f"    Thiele 模数:       {dim_groups['thiele_modulus']:.4e}")
    print(f"    Damköhler 数:      {dim_groups['damkohler_number']:.4e}")

    # 特殊函数验证
    print(f"\n  特殊函数验证:")
    x_test = [0.5, 1.0, 2.0, 5.0]
    print(f"    erf(x) 测试:  ", end="")
    for x in x_test:
        print(f"erf({x})={erf_approx(x):.6f}  ", end="")
    print()

    print(f"    正态 CDF:       Φ(0)={normal_cdf(0):.6f}, Φ(1)={normal_cdf(1):.6f}")
    print(f"    digamma:        ψ(1)={digamma_approx(1):.6f}, ψ(5)={digamma_approx(5):.6f}")
    print(f"    trigamma:       ψ₁(1)={trigamma_approx(1):.6f}, ψ₁(5)={trigamma_approx(5):.6f}")


def phase2_thermodynamics():
    """阶段 2: 热力学模型验证。"""
    section("阶段 2: 热力学模型 — Redlich-Kister OCV 与活度")

    # OCV 曲线
    soc_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print(f"\n  {'SOC':>6s}  {'OCV [V]':>10s}  {'dU/dx [V]':>12s}  {'Θ':>10s}  {'γ':>8s}")
    print(f"  {'─' * 6}  {'─' * 10}  {'─' * 12}  {'─' * 10}  {'─' * 8}")
    for soc in soc_values:
        U = redlich_kister_ocv(soc)
        dUdx = ocv_derivative(soc)
        theta = thermodynamic_factor(soc)
        gamma = margules_activity_coefficient(soc)
        print(f"  {soc:6.2f}  {U:10.4f}  {dUdx:12.4f}  {theta:10.4f}  {gamma:8.4f}")

    # Spinodal 边界
    spinodal_pts = spinodal_boundaries()
    print(f"\n  Spinodal 边界点 (d²U/dx² = 0):")
    if spinodal_pts:
        for i, x_sp in enumerate(spinodal_pts):
            print(f"    x_sp[{i}] = {x_sp:.6f}")
    else:
        print(f"    未检测到 spinodal 边界 (模型在 0-1 范围内稳定)")

    # 化学势
    c_test = [5000.0, 25000.0, 45000.0]
    print(f"\n  化学势 μ(c, T={T_REF}K):")
    for c in c_test:
        mu = chemical_potential(c, T_REF)
        D_eff = concentration_dependent_D(c, T_REF)
        print(f"    c={c:.0f} mol/m³: μ={mu:.4f} J/mol, D_eff={D_eff:.4e} m²/s")


def phase3_numerical_methods():
    """阶段 3: 数值方法验证。"""
    section("阶段 3: 高阶紧致有限差分与 Chebyshev 求积")

    # 紧致格式修正波数
    print(f"\n  紧致格式修正波数分析 (α=0.4):")
    kh_test = np.linspace(0, math.pi, 11)
    kh_prime_sq, kh_exact_sq = modified_wavenumber_compact(kh_test)
    kh_prime_label = "(k'h)²"
    print(f"    {'k*h':>8s}  {kh_prime_label:>12s}  {'(kh)²':>12s}  {'误差':>10s}")
    for i in range(len(kh_test)):
        err = abs(kh_prime_sq[i] - kh_exact_sq[i])
        print(f"    {kh_test[i]:8.4f}  {kh_prime_sq[i]:12.6f}  {kh_exact_sq[i]:12.6f}  {err:10.2e}")

    # Chebyshev 求积精确性
    print(f"\n  Chebyshev 求积规则精确性检验 (n=8):")
    cheb_results = chebyshev_quadrature_exactness(8, degree_max=15)
    for r in cheb_results[:8]:
        status = "✓" if r['is_exact'] else "✗"
        print(f"    degree {r['degree']:2d}: error = {r['absolute_error']:.2e}  {status}")

    # 谱半径
    D_test = D_REF
    N_test = 32
    h_test = PARTICLE_RADIUS / (N_test - 1)
    rho = spectral_radius_diffusion_operator(D_test, N_test, h_test)
    print(f"\n  扩散算子谱半径:")
    print(f"    D = {D_test:.2e}, N = {N_test}, h = {h_test:.2e}")
    print(f"    ρ(L) = {rho:.4e} rad/s")
    print(f"    CFL 限制 Δt_max = {0.9 * 2.0 / rho:.4e} s")


def phase4_stability():
    """阶段 4: 稳定性分析。"""
    section("阶段 4: von Neumann 稳定性分析与参数空间边界")

    # FTCS 稳定性
    print(f"\n  FTCS 格式稳定性:")
    r_ftcs = [0.1, 0.25, 0.4, 0.5, 0.51, 0.75, 1.0]
    ftcs_results = von_neumann_ftcs(r_ftcs)
    print(f"    {'r':>6s}  {'ρ(G)':>10s}  {'稳定':>6s}  {'CFL比':>8s}")
    for r in r_ftcs:
        res = ftcs_results[r]
        status = "是" if res['is_stable'] else "否"
        print(f"    {r:6.2f}  {res['spectral_radius']:10.6f}  {status:>6s}  {res['cfl_ratio']:8.3f}")

    # Crank-Nicolson 稳定性
    print(f"\n  Crank-Nicolson 格式稳定性:")
    r_cn = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
    cn_results = von_neumann_crank_nicolson(r_cn)
    print(f"    {'r':>6s}  {'ρ(G)':>10s}  {'稳定':>6s}  {'振荡比':>8s}")
    for r in r_cn:
        res = cn_results[r]
        status = "是" if res['is_stable'] else "否"
        print(f"    {r:6.1f}  {res['amplification_max']:10.6f}  {status:>6s}  {res['oscillation_fraction']:8.3f}")

    # 紧致 CN
    print(f"\n  紧致 CN 格式稳定性:")
    compact_results = von_neumann_compact_crank_nicolson([0.5, 1.0, 5.0, 10.0])
    for r, res in compact_results.items():
        status = "是" if res['is_stable'] else "否"
        print(f"    r={r:5.1f}: ρ(G)={res['amplification_max']:.6f}, 稳定={status}")

    # 参数空间稳定性边界
    print(f"\n  参数空间稳定性边界 (λ vs μ):")
    Lambda, Mu, stab_map, boundary_pts = stability_boundary_parameter_space(20, 20)
    n_stable = int(np.sum(stab_map))
    n_total = stab_map.size
    print(f"    稳定区域占比: {n_stable}/{n_total} = {n_stable/n_total:.2%}")
    print(f"    边界点数: {len(boundary_pts)}")

    # CFL 条件
    D_typical = concentration_dependent_D(C_MAX * 0.5, T_REF)
    N_test = 64
    h_test = PARTICLE_RADIUS / (N_test - 1)
    for scheme in ['ftcs', 'cn', 'compact_cn']:
        cfl = cfl_condition(D_typical, h_test, scheme=scheme)
        print(f"\n    {scheme.upper()} CFL 条件:")
        print(f"      Δt_max = {cfl['dt_max']:.4e} s")
        print(f"      CFL 数 = {cfl['cfl_number']:.4f}")


def phase5_crystal_structure():
    """阶段 5: 晶体结构与多晶网络。"""
    section("阶段 5: NMC 晶体结构与 Hilbert 曲线多晶网络")

    # 晶体结构
    crystal = NMCCrystalStructure()
    print(f"\n  NMC-622 晶体参数 (R-3m):")
    print(f"    a = {crystal.a * 1e10:.3f} Å")
    print(f"    c = {crystal.c * 1e10:.3f} Å")
    print(f"    V = {crystal.volume * 1e30:.3f} Å³")

    # 各向异性扩散张量
    D_para = 10.0 * D_REF
    D_perp = 0.1 * D_REF
    D_tensor = crystal.diffusion_tensor_anisotropic(D_para, D_perp)
    print(f"\n  各向异性扩散张量 [m²/s]:")
    print(f"    D_∥ = {D_para:.2e}, D_⊥ = {D_perp:.2e}")
    for i in range(3):
        print(f"    [{D_tensor[i,0]:.2e}, {D_tensor[i,1]:.2e}, {D_tensor[i,2]:.2e}]")

    # Hilbert 曲线
    print(f"\n  3D Hilbert 曲线映射 (order=2):")
    print(f"    h → (x, y, z):")
    for h in range(8):
        x, y, z = hilbert_3d_h_to_xyz(h, 2)
        print(f"      h={h}: ({x}, {y}, {z})")

    # 多晶网络
    network = PolycrystalGrainNetwork(n_grains=8, hilbert_order=1)
    network.initialize_random_orientations(seed=42)
    network.compute_grain_diffusivities(D_REF, anisotropy_ratio=50.0)
    network.compute_grain_boundary_diffusivities(0.25 * 1.602e-19, T_REF)
    summary = network.summary()
    print(f"\n  多晶晶粒网络:")
    print(f"    晶粒数: {summary['n_grains']}")
    print(f"    连接数: {summary['n_edges']}")
    print(f"    平均 D_grain = {summary['mean_grain_D']:.4e} m²/s")
    print(f"    平均 D_gb = {summary['mean_gb_D']:.4e} m²/s")

    # 连通性矩阵
    L = network.connectivity_matrix()
    eigs_L = np.linalg.eigvalsh(L)
    print(f"    Laplacian 特征值: {[f'{e:.4f}' for e in sorted(eigs_L)]}")


def phase6_combinatorial():
    """阶段 6: 组合枚举与构型分析。"""
    section("阶段 6: 格点构型枚举与组合分析")

    # 组合数
    print(f"\n  组合数 C(n,k):")
    for n in [6, 10, 15, 20]:
        for k in [n//4, n//2, 3*n//4]:
            c = i4_choose(n, k)
            log_c = i4_choose_log(n, k)
            print(f"    C({n},{k}) = {c}  (ln = {log_c:.4f})")

    # 构型熵
    print(f"\n  构型熵 S/k_B (n_sites=6):")
    for soc in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        S_exact, S_stirling = configuration_entropy(soc, n_sites=6)
        print(f"    x={soc:.1f}: S_exact/k_B={S_exact:.4f}, S_Stirling/k_B={S_stirling:.4f}")

    # 完全枚举 (小系统)
    print(f"\n  完全枚举 (4 格点, 2 Li):")
    configs = enumerate_lattice_configurations(4, 2)
    print(f"    构型数 = {len(configs)} (应为 C(4,2)=6)")
    for i, cfg in enumerate(configs[:6]):
        print(f"      构型 {i}: {cfg}")

    # 素性检验
    print(f"\n  素性检验 (项目568):")
    primes = [n for n in range(2, 30) if i4_is_prime(n)]
    print(f"    2~29 的素数: {primes}")

    # Collatz 格点迭代
    print(f"\n  Collatz 格点迭代:")
    test_config = (1, 0, 1, 1, 0, 0)
    print(f"    初始构型: {test_config}")
    current = test_config
    for step in range(5):
        current = collatz_lattice_step(current)
        print(f"    步骤 {step+1}: {current}")


def phase7_matrix_analysis():
    """阶段 7: 矩阵分解与特征值分析。"""
    section("阶段 7: QR 分解与矩阵特征值分析")

    # QR 分解测试
    np.random.seed(42)
    A_test = np.random.randn(5, 5)
    Q, R = qr_factorization(A_test)
    reconstruction_error = np.linalg.norm(Q @ R - A_test)
    orthogonality_error = np.linalg.norm(Q.T @ Q - np.eye(5))
    print(f"\n  QR 分解 (5×5 随机矩阵):")
    print(f"    重建误差 ||QR - A|| = {reconstruction_error:.2e}")
    print(f"    正交性误差 ||Q^TQ - I|| = {orthogonality_error:.2e}")

    # QR 求解
    b_test = np.random.randn(5)
    x_sol, residual = qr_solve_linear(A_test, b_test)
    print(f"    求解残差 ||Ax - b|| = {residual:.2e}")

    # 特征值
    eigs, _, n_iter = qr_eigenvalue_iteration(A_test)
    eigs_np = np.linalg.eigvals(A_test)
    eigs_sorted = np.sort(np.abs(eigs))[::-1]
    eigs_np_sorted = np.sort(np.abs(eigs_np))[::-1]
    eig_error = np.max(np.abs(eigs_sorted - eigs_np_sorted))
    print(f"\n  QR 特征值迭代 (5×5):")
    print(f"    迭代次数: {n_iter}")
    print(f"    特征值误差 (vs numpy): {eig_error:.2e}")
    print(f"    模最大特征值: {eigs_sorted[0]:.6f}")
    print(f"    模最小特征值: {eigs_sorted[-1]:.6f}")

    # 条件数
    cond = condition_number_estimate(A_test)
    print(f"    条件数估计: {cond:.4e}")

    # 转移矩阵 PageRank
    print(f"\n  Markov 转移矩阵分析:")
    T_test = np.abs(np.random.randn(4, 4))
    T_test = T_test / T_test.sum(axis=0, keepdims=True)  # 列归一化
    stationary, n_power, gap = markov_transition_analysis(T_test)
    print(f"    幂法迭代次数: {n_power}")
    print(f"    谱间隙: {gap:.6f}")
    print(f"    稳态分布: {[f'{s:.4f}' for s in stationary]}")


def phase8_signal_analysis():
    """阶段 8: 信号分析方法。"""
    section("阶段 8: 扩散信号分析与弛豫提取")

    # 合成测试信号 (多指数衰减)
    t = np.linspace(0, 10, 200)
    signal = (0.5 * np.exp(-t / 1.0)
              + 0.3 * np.exp(-t / 3.0)
              + 0.1 * np.exp(-t / 8.0)
              + 0.02 * np.random.randn(len(t)))

    # PSD
    freqs, psd = power_spectral_density(signal, dt=t[1] - t[0])
    print(f"\n  功率谱密度 (合成衰减信号):")
    print(f"    频率采样点数: {len(freqs)}")
    print(f"    最大 PSD 频率: {freqs[np.argmax(psd[1:])+1]:.4f} Hz")

    # 弛豫时间提取
    tau_fit, amp_fit, resid = extract_relaxation_times(signal, t, n_modes=3)
    print(f"\n  弛豫时间提取 (Prony 分析):")
    print(f"    拟合残差: {resid:.6f}")
    for i in range(len(tau_fit)):
        print(f"    模式 {i}: τ = {tau_fit[i]:.4f} s, A = {amp_fit[i]:.6f}")

    # 自相关
    lags, autocorr, T_int = concentration_autocorrelation(signal)
    print(f"\n  自相关分析:")
    print(f"    积分时间尺度: {T_int:.4f} 步")
    print(f"    C(0) = {autocorr[0]:.6f}")
    print(f"    C(10) = {autocorr[10]:.6f}" if len(autocorr) > 10 else "")

    # 多尺度分解
    approx, details = multiscale_decomposition(signal, n_levels=3)
    print(f"\n  多尺度分解:")
    for i, (a, d) in enumerate(zip(approx[1:], details)):
        print(f"    层级 {i+1}: 近似长度={len(a)}, 细节能量={np.sum(d**2):.6f}")


def phase9_full_simulation():
    """阶段 9: 完整仿真运行。"""
    section("阶段 9: 完整扩散仿真运行")

    print(f"\n  运行参数:")
    print(f"    网格点数: {N_GRID}")
    print(f"    最大步数: 200")
    print(f"    温度: {T_REF} K")
    print(f"    颗粒半径: {PARTICLE_RADIUS*1e6:.1f} μm")

    # 运行仿真
    results = run_diffusion_simulation(
        n_grid=N_GRID,
        n_steps=200,
        dt_initial=1.0,
        T=T_REF,
        soc_initial=0.3,
        Phi_s=4.0,
        Phi_e=0.0,
        boundary_mode='constant_current',
        I_app=-1e-5
    )

    # 验证
    validation = validate_simulation(results)
    print(f"\n  仿真验证结果:")
    for check_name, passed in validation['checks'].items():
        status = "通过" if passed else "失败"
        print(f"    {check_name}: {status}")
    print(f"    总结: {validation['n_passed']}/{validation['n_checks']} 项通过")

    # 诊断报告
    report = format_diagnostic_report(
        results['final_stats'],
        results['final_mass'],
        results['final_energy'],
        results['convergence'],
        results.get('eigenvalue_info'),
        results.get('grain_network_summary')
    )
    print(f"\n{report}")

    return results, validation


def phase10_summary(results, validation):
    """阶段 10: 总结。"""
    section("阶段 10: 综合结论")

    print(f"""
  1. 扩散系数:
     - Arrhenius 修正 D(T={T_REF}K) = {arrhenius_diffusivity(D_REF, E_ACT, T_REF):.4e} m²/s
     - 浓度依赖范围: [{concentration_dependent_D(0.1*C_MAX, T_REF):.2e},
                       {concentration_dependent_D(0.9*C_MAX, T_REF):.2e}] m²/s

  2. 热力学:
     - OCV 范围: [{redlich_kister_ocv(0.1):.3f}, {redlich_kister_ocv(0.9):.3f}] V
     - 最大热力学因子: Θ_max = {max(thermodynamic_factor(x) for x in np.linspace(0.05, 0.95, 19)):.4f}
     - Spinodal 边界: {spinodal_boundaries()}

  3. 数值方法:
     - 空间离散: 4 阶紧致有限差分 (Lele 1992)
     - 时间积分: Crank-Nicolson (无条件稳定)
     - 非线性求解: Newton-Raphson + Thomas 算法
     - 自适应步长: 嵌入对误差估计

  4. 稳定性:
     - FTCS CFL 限制: r ≤ 0.5
     - CN 格式: 对所有 r 稳定
     - 紧致 CN: 修正波数下仍无条件稳定

  5. 仿真结果:
     - 最终 SOC 范围: [{results['final_stats']['center_soc']:.4f},
                       {results['final_stats']['surface_soc']:.4f}]
     - 验证状态: {validation['n_passed']}/{validation['n_checks']} 通过
""")


def main():
    """主函数: 按阶段执行所有分析。"""
    print_header()

    try:
        # 阶段 1: 物理常数
        phase1_physical_constants()

        # 阶段 2: 热力学模型
        phase2_thermodynamics()

        # 阶段 3: 数值方法
        phase3_numerical_methods()

        # 阶段 4: 稳定性分析
        phase4_stability()

        # 阶段 5: 晶体结构
        phase5_crystal_structure()

        # 阶段 6: 组合枚举
        phase6_combinatorial()

        # 阶段 7: 矩阵分析
        phase7_matrix_analysis()

        # 阶段 8: 信号分析
        phase8_signal_analysis()

        # 阶段 9: 完整仿真
        results, validation = phase9_full_simulation()

        # 阶段 10: 总结
        phase10_summary(results, validation)

        print("\n" + "=" * 72)
        print("  PROJECT 281 完成 — 所有阶段成功执行")
        print("=" * 72)

    except Exception as e:
        print(f"\n  错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
