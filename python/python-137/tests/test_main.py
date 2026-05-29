# -*- coding: utf-8 -*-
"""
main.py

博士级结晶过程成核与生长动力学合成项目

科学问题：
    化学工程中的结晶过程是分离纯化的核心单元操作。本研究聚焦于
    多组分溶液在程序冷却与混沌混合条件下的晶体尺寸分布 (CSD)
    演化动力学，通过人口平衡方程 (PBE)、贝叶斯参数推断 (DREAM MCMC)
    和高维稀疏网格不确定性量化 (Smolyak UQ) 的耦合计算框架，
    实现结晶动力学参数的反演与工艺稳健性分析。

核心科学方程体系：
1. 人口平衡方程 (PBE):
   ∂f(L,t)/∂t + ∂[G(L,t,σ)·f(L,t)]/∂L = B(σ,t)·δ(L-L_0)

2. 矩方程 (Method of Moments):
   dμ_j/dt = j·∫_0^∞ L^{j-1} G(L,σ) f(L,t) dL + B(σ,t)·L_0^j

3. 经典成核理论 (CNT):
   B = A·exp(-16πγ³v_m² / [3(k_B T ln S)²])

4. 尺寸依赖生长 (ΔL-law):
   G(L,σ,T) = k_g0·exp(-E_g/(RT))·σ^g·(1+αL)^β

5. 质量平衡:
   dc/dt = -3·ρ_c·k_v·∫_0^∞ L² G(L,σ) f(L,t) dL

6. 溶解度 (van't Hoff):
   ln(c_sat) = -ΔH_diss/(RT) + ΔS_diss/R

7. DREAM MCMC 后验:
   p(θ|D) ∝ L(D|θ)·π(θ)

8. 稀疏网格 Smolyak:
   Q_L^{(d)} f = Σ_{|ℓ|_1≤L+d-1} (-1)^{L+d-1-|ℓ|_1}·C(d-1,L+d-1-|ℓ|_1)·⊗_{i=1}^d Q_{ℓ_i}^{(1)} f
"""

import numpy as np
import os
import sys

# 设置随机种子以保证可复现性
np.random.seed(2024)

from cooling_profile import (
    linear_cooling, sawtooth_cooling, optimal_cooling_polynomial,
    solubility_vanthoff, supersaturation
)
from chaotic_mixing import (
    generate_chaotic_mixing_trajectory, map_chen_to_supersaturation_fluctuation
)
from nucleation_model import (
    classical_nucleation_rate, secondary_nucleation_rate,
    total_nucleation_rate, critical_nucleus_radius,
    stochastic_nucleation_events
)
from growth_kinetics import (
    power_law_growth, size_dependent_growth,
    two_step_growth, bcf_spiral_growth, growth_rate_dispersion
)
from population_balance import PopulationBalanceSolver, quadrature_integrate
from csd_analysis import (
    kmeans_1d, discretize_csd_kmeans,
    csd_statistical_moments, diffraction_inversion_feret
)
from special_functions import lambert_w, fresnel_integrals, fraunhofer_diffraction_particle_size
from simplex_sampling import (
    dirichlet_sample_uniform_simplex,
    wedge01_monomial_integral,
    tetrahedron01_monomial_integral,
    composition_space_integral
)
from sparse_grid_uq import sparse_grid_integrate, uncertainty_quantification_crystallization
from mcmc_inference import dream_mcmc, log_prior_lognormal, log_likelihood_gaussian, estimate_parameters_summary
from data_io import (
    r8vec2_write, i4vec_transpose_print, format_moment_vector,
    write_simulation_results, read_simulation_results, index_set_to_string
)


def print_section(title):
    """打印分隔标题。"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_special_functions():
    """演示特殊函数在结晶问题中的应用。"""
    print_section("1. 特殊函数验证与应用")

    # Lambert W 在尺寸依赖生长解析解中的应用
    t_vals = np.linspace(0, 3600, 100)
    alpha = 0.01  # 1/m
    k_g = 1e-7  # m/s
    sigma = 0.5
    from nucleation_model import analytical_size_dependent_growth_law
    L_analytical = analytical_size_dependent_growth_law(t_vals, alpha, -1.0, k_g, sigma)
    print(f"  Lambert W 解析生长解：t=3600s 时 L = {L_analytical[-1]:.4e} m")

    # Fresnel 积分验证
    x_test = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    C, S = fresnel_integrals(x_test)
    print(f"  Fresnel 积分验证：")
    for i in range(len(x_test)):
        print(f"    x={x_test[i]:.1f}: C={C[i]:.6f}, S={S[i]:.6f}")

    # Fraunhofer 衍射颗粒尺寸分析
    theta = np.linspace(0.001, 0.1, 200)
    wavelength = 632.8e-9  # He-Ne 激光
    radius = 25e-6
    intensity = fraunhofer_diffraction_particle_size(radius, wavelength, theta)
    print(f"  Fraunhofer 衍射：颗粒半径 {radius*1e6:.1f} μm，"
          f"峰值光强角度 {theta[np.argmax(intensity)]*1e3:.3f} mrad")

    return t_vals, L_analytical


def demo_cooling_profiles():
    """演示冷却曲线及其对过饱和度的影响。"""
    print_section("2. 程序冷却曲线与过饱和度")

    t_total = 7200.0  # 2 小时
    T0, Tf = 350.0, 300.0
    t = np.linspace(0, t_total, 500)

    # 三种冷却策略
    T_linear = linear_cooling(t, T0, Tf, t_total)
    T_optimal = optimal_cooling_polynomial(t, T0, Tf, t_total, order=3)
    T_sawtooth = sawtooth_cooling(t, T_linear, delta_T=2.0, period=600.0)

    # van't Hoff 参数（以硫酸铵为例）
    H_diss = 25000.0  # J/mol
    S_diss = 70.0  # J/(mol·K)
    c0 = 0.55  # 初始浓度（高于饱和浓度以确保结晶发生）

    # 计算过饱和度
    sigma_linear = supersaturation(c0, T_linear, H_diss, S_diss)
    sigma_optimal = supersaturation(c0, T_optimal, H_diss, S_diss)
    sigma_sawtooth = supersaturation(c0, T_sawtooth, H_diss, S_diss)

    print(f"  线性冷却终点过饱和度: σ = {sigma_linear[-1]:.4f}")
    print(f"  最优冷却终点过饱和度: σ = {sigma_optimal[-1]:.4f}")
    print(f"  锯齿波冷却平均过饱和度: σ_avg = {np.mean(sigma_sawtooth):.4f}")
    print(f"  锯齿波冷却过饱和度标准差: σ_std = {np.std(sigma_sawtooth):.4f}")

    return t, T_linear, T_sawtooth, sigma_linear, H_diss, S_diss, c0


def demo_chaotic_mixing(t_span, sigma_base):
    """演示混沌混合对过饱和度的影响。"""
    print_section("3. 混沌混合与过饱和度涨落")

    sol = generate_chaotic_mixing_trajectory(t_span, y0=[-0.1, 0.5, -0.6],
                                               params={'a': 40.0, 'b': 3.0, 'c': 28.0})
    print(f"  Chen 混沌吸引子积分成功：t ∈ [{sol.t[0]:.1f}, {sol.t[-1]:.1f}] s")

    t_check = np.linspace(t_span[0], t_span[1], 500)
    sigma_local = map_chen_to_supersaturation_fluctuation(t_check, sol, sigma_base,
                                                           scale_T=0.5, scale_c=0.3)

    print(f"  基础过饱和度: σ_base = {sigma_base:.4f}")
    print(f"  局部过饱和度范围: [{np.min(sigma_local):.4f}, {np.max(sigma_local):.4f}]")
    print(f"  局部过饱和度均值: {np.mean(sigma_local):.4f}")

    # 混沌混合增强的成核率
    from chaotic_mixing import mixing_enhanced_nucleation_rate
    B_avg, B_instant = mixing_enhanced_nucleation_rate(sigma_base, t_check, sol,
                                                        B0=1e15, scale_T=0.5, scale_c=0.3)
    B_base = classical_nucleation_rate(sigma_base, 320.0, A_prefactor=1e15)
    print(f"  基础成核率: B_base = {float(B_base):.4e} #/(m³·s)")
    print(f"  混沌增强平均成核率: B_avg = {B_avg:.4e} #/(m³·s)")
    print(f"  增强因子: {B_avg / max(float(B_base), 1e-100):.2f}x")

    return sol, sigma_local


def demo_nucleation_growth():
    """演示成核与生长动力学模型。"""
    print_section("4. 成核与生长动力学模型")

    sigma_range = np.linspace(0.01, 1.0, 50)
    T = 320.0  # K
    MT = 50.0  # kg/m³

    # 初级成核
    B_prim = classical_nucleation_rate(sigma_range, T, A_prefactor=1e20)
    # 二级成核
    B_sec = secondary_nucleation_rate(sigma_range, MT, kb_sec=1e8, b_exp=2.0, j_exp=1.0)
    # 总成核
    B_total = B_prim + B_sec

    print(f"  σ=0.5 时：")
    print(f"    初级成核率: {B_prim[24]:.4e} #/(m³·s)")
    print(f"    二级成核率: {B_sec[24]:.4e} #/(m³·s)")
    print(f"    总成核率: {B_total[24]:.4e} #/(m³·s)")

    # 临界核半径
    r_star = critical_nucleus_radius(0.5, T)
    print(f"    临界核半径: r* = {r_star:.4e} m = {r_star*1e9:.2f} nm")

    # 生长速率模型比较
    L_test = 50e-6  # 50 μm
    G_power = power_law_growth(0.5, T, k_g0=1e-4, E_g=40000.0, g_exp=1.5)
    G_size = size_dependent_growth(L_test, 0.5, T, k_g0=1e-4, E_g=40000.0,
                                    g_exp=1.5, alpha=100.0, beta=-0.3)
    G_twostep = two_step_growth(0.5, T, k_d=1e-5, k_r0=1e-3, E_r=35000.0, g_r=2.0)
    G_bcf = bcf_spiral_growth(0.5, T, A_bcf=1e-4, B_bcf=0.2, E_act=45000.0)

    print(f"  生长速率比较 (σ=0.5, T=320K, L=50μm):")
    print(f"    幂律模型: G = {G_power:.4e} m/s")
    print(f"    尺寸依赖: G = {G_size:.4e} m/s")
    print(f"    两步模型: G = {G_twostep:.4e} m/s")
    print(f"    BCF 模型: G = {G_bcf:.4e} m/s")

    # 生长速率分散
    G_disp = growth_rate_dispersion(G_power, cv=0.15, n_samples=1000)
    print(f"    生长速率分散 (CV=0.15): 均值={np.mean(G_disp):.4e}, 标准差={np.std(G_disp):.4e}")

    return sigma_range, B_total


def demo_population_balance(t, T_linear, H_diss, S_diss, c0):
    """演示人口平衡方程求解。"""
    print_section("5. 人口平衡方程求解 (矩方法)")

    params = {
        'rho_c': 1560.0,      # kg/m³
        'kv': 0.5236,         # 球形体积形状因子
        'L0': 1e-9,           # 临界核尺寸 1 nm
        'k_g0': 5e-5,         # m/s
        'E_g': 42000.0,       # J/mol
        'g_exp': 1.2,         # -
        'alpha': 50.0,        # 1/m
        'beta': -0.2,         # -
        'A_prefactor': 1e18,  # #/(m³·s)
        'kb_sec': 5e7,        # -
        'b_exp': 1.8,         # -
        'j_exp': 0.8,         # -
        'H_diss': H_diss,     # J/mol
        'S_diss': S_diss,     # J/(mol·K)
        'c0': c0,             # -
        'T0': 350.0,          # K
        'V': 0.001,           # m³
        'gamma0': 0.025,      # J/m²
    }

    solver = PopulationBalanceSolver(params)
    T_func = lambda tau: float(linear_cooling(tau, 350.0, 300.0, 7200.0))

    sol = solver.solve((0.0, 7200.0), T_func, method='RK45')
    print(f"  PBE 求解成功：{sol.message}")
    print(f"  积分步数: {sol.nfev}")

    # 提取结果
    t_eval = np.linspace(0, 7200, 100)
    moments_history = []
    c_history = []
    for te in t_eval:
        mu, c = solver.get_moments_at_time(sol, te)
        moments_history.append(mu)
        c_history.append(c)

    moments_history = np.array(moments_history)
    c_history = np.array(c_history)

    # 最终时刻的矩量
    mu_final, c_final = solver.get_moments_at_time(sol, 7200.0)
    print(f"\n  t = 7200 s 时的矩量:")
    print(format_moment_vector(mu_final, [f"μ_{i}" for i in range(6)]))
    print(f"  最终浓度: c = {c_final:.6f} kg/kg")

    # CSD 参数
    csd_params = solver.get_csd_parameters(sol, 7200.0)
    print(f"\n  CSD 参数 (对数正态近似):")
    print(f"    总晶数密度: N = {csd_params['N']:.4e} #/m³")
    print(f"    平均尺寸: L_mean = {csd_params['L_mean']*1e6:.3f} μm")
    print(f"    变异系数: CV = {csd_params['CV']:.4f}")

    return sol, t_eval, moments_history, c_history, csd_params


def demo_csd_analysis(L_grid, f_values, csd_params):
    """演示 CSD 分析工具。"""
    print_section("6. CSD 分析与 K-Means 离散化")

    # K-Means 聚类离散化
    k_classes = 8
    class_sizes, class_counts, boundaries = discretize_csd_kmeans(L_grid, f_values, k_classes)
    print(f"  K-Means 离散化为 {k_classes} 个尺寸类:")
    print(f"    代表尺寸 (μm): {np.array2string(class_sizes * 1e6, precision=2, separator=' ')}")
    print(f"    各类晶数密度 (#/m³): {np.array2string(class_counts, precision=4, separator=' ')}")

    # 统计矩量
    stats = csd_statistical_moments(L_grid, f_values)
    print(f"\n  CSD 统计特征:")
    print(f"    均值: {stats['mean']*1e6:.3f} μm")
    print(f"    标准差: {stats['std']*1e6:.3f} μm")
    print(f"    偏度: {stats['skewness']:.4f}")
    print(f"    峰度: {stats['kurtosis']:.4f}")

    # 衍射反演
    theta = np.linspace(0.001, 0.05, 100)
    wavelength = 632.8e-9
    intensity = fraunhofer_diffraction_particle_size(stats['mean'] / 2.0, wavelength, theta)
    # 添加噪声
    intensity_noisy = intensity + np.random.normal(0, 0.01 * np.max(intensity), len(intensity))
    L_bins, n_L = diffraction_inversion_feret(theta, intensity_noisy, wavelength,
                                               L_min=1e-6, L_max=200e-6, n_bins=50)
    peak_L = L_bins[np.argmax(n_L)]
    print(f"\n  激光衍射粒度反演:")
    print(f"    反演峰值尺寸: {peak_L*1e6:.2f} μm")
    print(f"    真实平均尺寸: {stats['mean']*1e6:.2f} μm")

    return class_sizes, class_counts


def demo_simplex_integration():
    """演示单纯形采样与积分。"""
    print_section("7. 多组分溶解度空间积分")

    # 三组分体系的 Gibbs 自由能曲面积分
    def gibbs_excess_energy(x):
        """
        简化的 Margules 方程 excess Gibbs 自由能：
        G^E/RT = x_1·x_2·A_12 + x_1·x_3·A_13 + x_2·x_3·A_23
        """
        x1, x2 = x[:, 0], x[:, 1]
        x3 = 1.0 - x1 - x2
        x3 = np.where(x3 < 0, 0.0, x3)
        A12, A13, A23 = 2.5, 1.8, 3.0
        ge = x1 * x2 * A12 + x1 * x3 * A13 + x2 * x3 * A23
        return ge

    integral, err = composition_space_integral(gibbs_excess_energy, dim=2, n_samples=50000)
    print(f"  三组分 Margules  excess Gibbs 自由能积分:")
    print(f"    积分值: {integral:.6f}")
    print(f"    Monte Carlo 标准误差: {err:.6f}")

    # 楔形区域精确积分验证
    e_test = [2, 1, 0]
    I_wedge = wedge01_monomial_integral(e_test)
    print(f"\n  楔形区域精确单项式积分:")
    print(f"    ∫_W x² y¹ z⁰ dV = {I_wedge:.8f}")

    # 四面体精确积分验证
    I_tet = tetrahedron01_monomial_integral(e_test)
    print(f"  四面体精确单项式积分:")
    print(f"    ∫_T x² y¹ z⁰ dV = {I_tet:.8f}")

    # 与 Monte Carlo 比较
    rng = np.random.default_rng(42)
    samples = dirichlet_sample_uniform_simplex(100000, 2, rng)
    x1, x2 = samples[:, 0], samples[:, 1]
    mc_vals = (x1 ** 2) * (x2 ** 1)
    mc_estimate = np.mean(mc_vals) / 2.0  # Vol(T_2) = 1/2
    print(f"  Monte Carlo 验证 (100000 样本): {mc_estimate:.8f}")

    return integral


def demo_sparse_grid_uq():
    """演示稀疏网格不确定性量化。"""
    print_section("8. 稀疏网格不确定性量化 (UQ)")

    # 定义一个简化的结晶模型输出函数
    # 输出 = 最终平均尺寸，依赖于 3 个不确定参数
    def crystallization_model(params):
        """
        params = [ln k_g0, E_g/10000, ln k_b]
        """
        k_g0 = np.exp(params[0])
        E_g = params[1] * 10000.0
        k_b = np.exp(params[2])
        # 简化模型：L_mean ∝ k_g0 * exp(-E_g/RT) * k_b^0.1
        T_avg = 325.0
        R = 8.314
        L_mean = k_g0 * np.exp(-E_g / (R * T_avg)) * (k_b ** 0.1) * 1e6
        return L_mean

    # 参数分布
    param_dists = [
        {'mean': -9.0, 'std': 0.5},   # ln k_g0
        {'mean': 4.0, 'std': 0.3},    # E_g/10000
        {'mean': 16.0, 'std': 1.0},   # ln k_b
    ]

    mean_val, variance, std_val = uncertainty_quantification_crystallization(
        crystallization_model, param_dists, level_max=3
    )

    print(f"  三参数稀疏网格 UQ (level_max=3):")
    print(f"    输出期望 (平均尺寸): {mean_val:.4f} μm")
    print(f"    输出方差: {variance:.4f}")
    print(f"    输出标准差: {std_val:.4f} μm")

    # 高维积分测试
    def test_func_nd(x):
        """测试函数：f(x) = exp(-Σ x_i²)"""
        return np.exp(-np.sum(x ** 2, axis=1))

    for dim in [2, 3]:
        result = sparse_grid_integrate(test_func_nd, dim, level_max=3)
        # 精确值：∫_{-1}^1 exp(-x²) dx = √π · erf(1)
        from scipy.special import erf
        exact = (np.sqrt(np.pi) * erf(1.0)) ** dim
        print(f"  {dim}D 稀疏网格测试: 数值={result:.8f}, 精确={exact:.8f}, 误差={abs(result-exact):.2e}")

    return mean_val, std_val


def demo_mcmc_inference(c_history_true, t_eval):
    """演示 DREAM MCMC 参数推断。"""
    print_section("9. DREAM MCMC 贝叶斯参数推断")

    # 模拟实验数据：从真实浓度曲线添加噪声
    sigma_noise = 0.005
    data_noisy = c_history_true + np.random.normal(0, sigma_noise, len(c_history_true))

    # 定义前向模型
    def forward_model(theta):
        """
        theta = [k_g0, E_g, k_b]
        简化的矩模型预测浓度曲线
        """
        k_g0, E_g, k_b = theta
        if k_g0 <= 0 or E_g <= 0 or k_b <= 0:
            return np.full(len(t_eval), 1e10)

        # 简化模型：dc/dt ≈ -C·k_g0·exp(-E_g/RT)·σ^g
        T = 350.0 - 50.0 * t_eval / 7200.0
        sigma = np.maximum((0.45 - 0.35 * np.exp(-25000.0 / (8.314 * T))) / 0.35, 0.0)
        # 简化的浓度衰减模型
        rate = k_g0 * np.exp(-E_g / (8.314 * T)) * (sigma ** 1.2) * k_b * 1e-8
        c_pred = 0.45 * np.exp(-np.cumsum(rate) * (t_eval[1] - t_eval[0]))
        return c_pred

    # 定义对数后验
    mu_ln = np.array([-9.0, np.log(42000.0), np.log(5e7)])
    sigma_ln = np.array([0.5, 0.1, 0.5])

    def log_posterior(theta):
        lp = log_prior_lognormal(theta, mu_ln, sigma_ln)
        if lp == -np.inf:
            return -np.inf
        ll = log_likelihood_gaussian(theta, forward_model, data_noisy, sigma_noise)
        return lp + ll

    # 运行 DREAM
    bounds = np.array([[1e-8, 10000.0, 1e3],
                       [1e-2, 80000.0, 1e10]], dtype=float)

    print(f"  开始 DREAM MCMC 采样 (3 条链, 1500 代)...")
    samples, logpost, R_hat, acc_rate = dream_mcmc(
        log_posterior, n_params=3, n_chains=3, n_generations=1500,
        bounds=bounds, init_scale=0.1, gr_threshold=1.05
    )

    print(f"  DREAM 采样完成:")
    print(f"    接受率: {acc_rate:.4f}")
    print(f"    Gelman-Rubin R̂: {R_hat}")
    print(f"    参数 1 (k_g0) R̂: {R_hat[0]:.4f}")
    print(f"    参数 2 (E_g)  R̂: {R_hat[1]:.4f}")
    print(f"    参数 3 (k_b)  R̂: {R_hat[2]:.4f}")

    # 参数摘要
    summary = estimate_parameters_summary(samples)
    param_names = ['k_g0 (m/s)', 'E_g (J/mol)', 'k_b (-)']
    print(f"\n  后验参数估计:")
    for i, name in enumerate(param_names):
        print(f"    {name}:")
        print(f"      均值: {summary['mean'][i]:.4e}")
        print(f"      中位数: {summary['median'][i]:.4e}")
        print(f"      标准差: {summary['std'][i]:.4e}")
        print(f"      95% CI: [{summary['ci_lower'][i]:.4e}, {summary['ci_upper'][i]:.4e}]")

    return samples, summary


def demo_data_io(sol, t_eval, moments_history, c_history, csd_params, class_sizes):
    """演示数据 I/O 功能。"""
    print_section("10. 数据输入输出与结果管理")

    # 格式化输出矩量
    mu_final = moments_history[-1, :]
    print("  最终矩量格式化输出:")
    print(format_moment_vector(mu_final))

    # 索引集输出
    active_classes = np.where(class_sizes > 1e-7)[0]
    idx_str = index_set_to_string(active_classes, name="ActiveClasses")
    print(f"\n  活跃尺寸类索引集: {idx_str}")

    # 向量对写入
    output_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(output_dir, "simulation_data.txt")
    r8vec2_write(data_file, t_eval, c_history)
    print(f"\n  时间-浓度数据已写入: {data_file}")

    # 结构化 JSON 输出
    json_file = os.path.join(output_dir, "simulation_results.json")
    data_dict = {
        'time': t_eval,
        'concentration': c_history,
        'moments': moments_history,
        'csd_mean_um': csd_params['L_mean'] * 1e6,
        'csd_cv': csd_params['CV'],
    }
    metadata = {
        'project': 'Crystallization Nucleation & Growth Kinetics',
        'domain': 'Chemical Engineering',
        'author': 'PhD-Level Synthesis',
        'n_moments': 6,
        'solver': 'Method of Moments + RK45',
    }
    write_simulation_results(json_file, data_dict, metadata)
    print(f"  结构化结果已写入: {json_file}")

    # 解析验证
    meta_read, data_read = read_simulation_results(json_file)
    print(f"\n  读取验证:")
    print(f"    项目名称: {meta_read.get('project', 'N/A')}")
    print(f"    CSD 平均尺寸: {data_read.get('csd_mean_um', 0):.3f} μm")


def main():
    """
    主程序入口。零参数运行，执行完整的结晶动力学分析流程。
    """
    print("\n" + "#" * 70)
    print("#  博士级科研代码合成项目")
    print("#  领域：化学工程 — 结晶过程成核与生长动力学")
    print("#  项目编号：PROJECT_137")
    print("#" * 70)

    # 1. 特殊函数
    demo_special_functions()

    # 2. 冷却曲线
    t, T_linear, T_sawtooth, sigma_linear, H_diss, S_diss, c0 = demo_cooling_profiles()

    # 3. 混沌混合
    t_span = (0.0, 7200.0)
    sol_chaos, sigma_local = demo_chaotic_mixing(t_span, np.mean(sigma_linear))

    # 4. 成核与生长
    sigma_range, B_total = demo_nucleation_growth()

    # 5. 人口平衡方程
    sol_pbe, t_eval, moments_history, c_history, csd_params = demo_population_balance(
        t, T_linear, H_diss, S_diss, c0
    )

    # 6. CSD 分析
    # 构造近似的 CSD 分布
    L_grid = np.linspace(1e-6, 200e-6, 500)
    mu_ln = csd_params['mu_ln']
    sigma_ln = csd_params['sigma_ln']
    N = csd_params['N']
    f_values = (N / (L_grid * sigma_ln * np.sqrt(2 * np.pi))) * \
               np.exp(-0.5 * ((np.log(L_grid) - mu_ln) / sigma_ln) ** 2)
    f_values = np.where(np.isnan(f_values), 0.0, f_values)
    class_sizes, class_counts = demo_csd_analysis(L_grid, f_values, csd_params)

    # 7. 单纯形积分
    demo_simplex_integration()

    # 8. 稀疏网格 UQ
    demo_sparse_grid_uq()

    # 9. MCMC 推断
    samples, summary = demo_mcmc_inference(c_history, t_eval)

    # 10. 数据 I/O
    demo_data_io(sol_pbe, t_eval, moments_history, c_history, csd_params, class_sizes)

    # 清理可视化相关文件
    print_section("计算完成")
    print("  所有计算模块已成功执行，无报错。")
    print("  结果文件保存在项目目录中。")
    print("\n" + "#" * 70)


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（55个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: linear_cooling - midpoint check ----
t = np.array([3600.0])
T = linear_cooling(t, 350.0, 300.0, 7200.0)
assert abs(T[0] - 325.0) < 1e-6, '[TC01] linear_cooling midpoint FAILED'

# ---- TC02: natural_cooling - initial value equals T0 ----
from cooling_profile import natural_cooling
T = natural_cooling(np.array([0.0]), 350.0, 300.0, 1000.0)
assert abs(T[0] - 350.0) < 1e-6, '[TC02] natural_cooling t=0 FAILED'

# ---- TC03: solubility_vanthoff - positive output ----
c_sat = solubility_vanthoff(np.array([320.0]), 25000.0, 70.0)
assert np.all(c_sat > 0), '[TC03] solubility_vanthoff positive FAILED'

# ---- TC04: supersaturation - zero when c equals c_sat ----
T = np.array([320.0])
c_sat_val = solubility_vanthoff(T, 25000.0, 70.0)
sigma = supersaturation(c_sat_val, T, 25000.0, 70.0)
assert abs(sigma[0]) < 1e-6, '[TC04] supersaturation zero equilibrium FAILED'

# ---- TC05: optimal_cooling_polynomial - monotonic decreasing ----
t = np.linspace(0, 7200, 100)
T_opt = optimal_cooling_polynomial(t, 350.0, 300.0, 7200.0, order=3)
assert np.all(np.diff(T_opt) <= 0), '[TC05] optimal_cooling monotonic FAILED'

# ---- TC06: lambert_w - W(e) = 1.0 ----
w = lambert_w(np.array([np.e]), branch=0)
assert abs(w[0] - 1.0) < 1e-10, '[TC06] lambert_w W(e)=1 FAILED'

# ---- TC07: lambert_w - W(10) is finite and positive ----
w = lambert_w(np.array([10.0]), branch=0)
assert np.isfinite(w[0]) and w[0] > 1.5, '[TC07] lambert_w W(10) FAILED'

# ---- TC08: fresnel_integrals - C(0)=0, S(0)=0 ----
C, S = fresnel_integrals(np.array([0.0]))
assert abs(C[0]) < 1e-12 and abs(S[0]) < 1e-12, '[TC08] fresnel C(0)=S(0)=0 FAILED'

# ---- TC09: fresnel_integrals - large x asymptote near 0.5 ----
C, S = fresnel_integrals(np.array([10.0]))
assert abs(C[0] - 0.5) < 0.06 and abs(S[0] - 0.5) < 0.06, '[TC09] fresnel asymptote FAILED'

# ---- TC10: fraunhofer_diffraction - peak near theta=0 ----
theta = np.linspace(0.001, 0.1, 100)
I = fraunhofer_diffraction_particle_size(25e-6, 632.8e-9, theta)
assert np.argmax(I) == 0, '[TC10] fraunhofer peak at theta~0 FAILED'

# ---- TC11: chen_attractor_rhs - at origin gives zero derivative ----
from chaotic_mixing import chen_attractor_rhs
dstate = chen_attractor_rhs(0.0, np.array([0.0, 0.0, 0.0]))
assert np.allclose(dstate, [0.0, 0.0, 0.0], atol=1e-12), '[TC11] chen attractor origin FAILED'

# ---- TC12: classical_nucleation_rate - increases with supersaturation ----
B1 = classical_nucleation_rate(np.array([0.3]), 320.0, A_prefactor=1e15)
B2 = classical_nucleation_rate(np.array([0.5]), 320.0, A_prefactor=1e15)
assert float(B2) > float(B1), '[TC12] CNT monotonic in sigma FAILED'

# ---- TC13: secondary_nucleation_rate - zero at zero sigma ----
B = secondary_nucleation_rate(np.array([0.0]), 50.0)
assert float(B) == 0.0, '[TC13] secondary nucleation zero FAILED'

# ---- TC14: critical_nucleus_radius - returns positive value ----
r_star = critical_nucleus_radius(0.5, 320.0)
assert float(r_star) > 0, '[TC14] critical radius positive FAILED'

# ---- TC15: power_law_growth - zero at zero sigma ----
G = power_law_growth(0.0, 320.0, 1e-4, 40000.0, 1.5)
assert float(G) == 0.0, '[TC15] power law growth zero FAILED'

# ---- TC16: size_dependent_growth - reduces to power law when alpha=0, beta=0 ----
G_sd = size_dependent_growth(50e-6, 0.5, 320.0, 1e-4, 40000.0, 1.5, 0.0, 0.0)
G_pl = power_law_growth(0.5, 320.0, 1e-4, 40000.0, 1.5)
assert abs(float(G_sd) - float(G_pl)) < 1e-15, '[TC16] size_dependent reduces to power law FAILED'

# ---- TC17: two_step_growth - positive output for valid inputs ----
G = two_step_growth(0.5, 320.0, 1e-5, 1e-3, 35000.0, 2.0)
assert float(G) > 0, '[TC17] two_step positive FAILED'

# ---- TC18: bcf_spiral_growth - positive output for valid inputs ----
G = bcf_spiral_growth(0.5, 320.0, 1e-4, 0.2, 45000.0)
assert float(G) > 0, '[TC18] BCF positive FAILED'

# ---- TC19: newton_cotes_open_weights - returns correct number of nodes/weights ----
from population_balance import newton_cotes_open_weights
x, w = newton_cotes_open_weights(5, -1.0, 1.0)
assert len(x) == 5 and len(w) == 5, '[TC19] NCO weights shape FAILED'
assert np.all(x >= -1.0) and np.all(x <= 1.0), '[TC19] NCO nodes in interval FAILED'

# ---- TC20: quadrature_integrate - runs and returns float without exception ----
I = quadrature_integrate(lambda x: np.ones_like(x), 0.0, 1.0, n=32)
assert isinstance(I, float), '[TC20] quadrature returns float FAILED'

# ---- TC21: kmeans_1d - two cluster separation ----
rng = np.random.default_rng(42)
data = np.concatenate([rng.normal(0, 0.5, 100), rng.normal(5, 0.5, 100)])
centers, labels, inertia = kmeans_1d(data, 2, rng=rng)
assert len(centers) == 2, '[TC21] kmeans two clusters count FAILED'
assert abs(centers[0]) < 1.0, '[TC21] kmeans center near 0 FAILED'

# ---- TC22: dirichlet_sample_uniform_simplex - reproducibility with fixed seed ----
rng1 = np.random.default_rng(42)
samples1 = dirichlet_sample_uniform_simplex(100, 2, rng1)
rng2 = np.random.default_rng(42)
samples2 = dirichlet_sample_uniform_simplex(100, 2, rng2)
assert np.allclose(samples1, samples2), '[TC22] dirichlet reproducibility FAILED'

# ---- TC23: wedge01_monomial_integral - known value for (0,0,0) = volume ----
I = wedge01_monomial_integral(np.array([0, 0, 0]))
# wedge W: x>=0,y>=0,x+y<=1,-1<=z<=1, volume = 1/2 * 2 = 1
assert abs(I - 1.0) < 1e-12, '[TC23] wedge integral (0,0,0) FAILED'

# ---- TC24: tetrahedron01_monomial_integral - known value for (0,0,0) ----
I = tetrahedron01_monomial_integral(np.array([0, 0, 0]))
assert abs(I - 1.0/6.0) < 1e-12, '[TC24] tetrahedron integral (0,0,0) FAILED'

# ---- TC25: sparse_grid_integrate - 2D exp(-x^2-y^2) accuracy ----
def f_2d(x):
    return np.exp(-x[:, 0]**2 - x[:, 1]**2)
result = sparse_grid_integrate(f_2d, 2, level_max=3)
from scipy.special import erf
exact = (np.sqrt(np.pi) * erf(1.0)) ** 2
assert abs(result - exact) < 0.01, '[TC25] sparse grid 2D FAILED'

# ---- TC26: log_prior_lognormal - valid positive params give finite value ----
lp = log_prior_lognormal(np.array([1e-5, 40000.0, 5e7]),
                          np.array([-9.0, np.log(42000.0), np.log(5e7)]),
                          np.array([0.5, 0.1, 0.5]))
assert np.isfinite(lp), '[TC26] log prior finite FAILED'

# ---- TC27: log_prior_lognormal - negative parameter gives -inf ----
lp = log_prior_lognormal(np.array([-1e-5, 40000.0, 5e7]),
                          np.array([-9.0, np.log(42000.0), np.log(5e7)]),
                          np.array([0.5, 0.1, 0.5]))
assert lp == -np.inf, '[TC27] log prior negative FAILED'

# ---- TC28: gelman_rubin_diagnostic - identical chains give R_hat near 1 ----
from mcmc_inference import gelman_rubin_diagnostic
np.random.seed(42)
chain = np.random.randn(1, 500, 3)
chains = np.repeat(chain, 3, axis=0)
R_hat = gelman_rubin_diagnostic(chains)
# For identical chains, R_hat = sqrt((n-1)/n) ≈ 0.999
assert np.all(np.abs(R_hat - 1.0) < 0.002), '[TC28] Gelman-Rubin identical FAILED'

# ---- TC29: format_moment_vector - output contains expected value string ----
mu = np.array([1.0, 2.0, 3.0])
s = format_moment_vector(mu, ['a', 'b', 'c'])
assert '1.000000e+00' in s, '[TC29] format moment vector FAILED'

# ---- TC30: index_set_to_string - correct formatting ----
s = index_set_to_string([0, 3, 1], name="Test")
assert s == "Test = {0, 1, 3}", '[TC30] index set to string FAILED'

# ---- TC31: csd_statistical_moments - Gaussian distribution mean recovery ----
L = np.linspace(1e-6, 200e-6, 1000)
mu_target = 100e-6
sigma_target = 20e-6
f = np.exp(-0.5 * ((L - mu_target) / sigma_target) ** 2)
stats = csd_statistical_moments(L, f)
assert abs(stats['mean'] - mu_target) < 1e-6, '[TC31] CSD moments mean FAILED'

# ---- TC32: lcg_park_miller - deterministic output ----
from nucleation_model import lcg_park_miller
seed1, u1 = lcg_park_miller(42)
seed2, u2 = lcg_park_miller(42)
assert seed1 == seed2 and abs(u1 - u2) < 1e-15, '[TC32] LCG deterministic FAILED'

# ---- TC33: growth_rate_dispersion - reproducibility with fixed seed ----
rng1 = np.random.default_rng(42)
G1 = growth_rate_dispersion(1e-8, cv=0.1, n_samples=100, rng=rng1)
rng2 = np.random.default_rng(42)
G2 = growth_rate_dispersion(1e-8, cv=0.1, n_samples=100, rng=rng2)
assert np.allclose(G1, G2), '[TC33] GRD reproducibility FAILED'

# ---- TC34: parse_solution_vector - basic parsing ----
from data_io import parse_solution_vector
text = "x0 = 1.5\nx1 = 3.14\nx2 = 2.718"
vals = parse_solution_vector(text)
assert len(vals) == 3 and abs(vals[0] - 1.5) < 1e-10, '[TC34] parse solution FAILED'

# ---- TC35: monomial_value - basic evaluation ----
from simplex_sampling import monomial_value
x = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
v = monomial_value(2, 3, np.array([2, 1]), x)
assert abs(v[0] - 2.0) < 1e-12, '[TC35] monomial value FAILED'

# ---- TC36: tetrahedron01_volume - known value 1/6 ----
from simplex_sampling import tetrahedron01_volume
assert abs(tetrahedron01_volume() - 1.0/6.0) < 1e-12, '[TC36] tetrahedron volume FAILED'

# ---- TC37: sawtooth_cooling - periodic average approximates base average ----
t = np.linspace(0, 6000, 1001)
T_base = linear_cooling(t, 350.0, 300.0, 6000.0)
T_saw = sawtooth_cooling(t, T_base, delta_T=5.0, period=100.0)
assert abs(np.mean(T_saw) - np.mean(T_base)) < 0.1, '[TC37] sawtooth mean FAILED'

# ---- TC38: total_nucleation_rate - positive output ----
B = total_nucleation_rate(0.5, 320.0, 50.0)
assert float(B) > 0, '[TC38] total nucleation positive FAILED'

# ---- TC39: write_simulation_results and read_simulation_results roundtrip ----
tmpf = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_test_roundtrip.json')
data = {'a': np.array([1.0, 2.0, 3.0]), 'b': 42.0}
meta = {'test': 'roundtrip'}
write_simulation_results(tmpf, data, meta)
meta_r, data_r = read_simulation_results(tmpf)
assert abs(data_r['a'][1] - 2.0) < 1e-12, '[TC39] write/read roundtrip FAILED'
os.remove(tmpf)

# ---- TC40: r8vec2_write produces readable output with correct header ----
tmpf = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_test_r8vec2.txt')
x = np.array([0.0, 1.0, 2.0])
y = np.array([3.0, 4.0, 5.0])
r8vec2_write(tmpf, x, y)
with open(tmpf, 'r') as f:
    content = f.read()
assert '# Paired vectors, n = 3' in content, '[TC40] r8vec2 write FAILED'
os.remove(tmpf)

# ---- TC41: i4vec_transpose_print - runs without exception ----
i4vec_transpose_print(np.array([1, 2, 3, 4, 5]), title="Test")
assert True, '[TC41] i4vec print no crash FAILED'

# ---- TC42: lambert_w - branch -1 on valid input returns value < -1 ----
w = lambert_w(np.array([-0.3]), branch=-1)
assert np.isfinite(w[0]) and w[0] < -1.0, '[TC42] lambert_w branch -1 FAILED'

# ---- TC43: dirichlet sampling - all samples in simplex ----
rng = np.random.default_rng(100)
samples = dirichlet_sample_uniform_simplex(500, 3, rng)
assert np.all(samples >= 0), '[TC43] dirichlet non-negative FAILED'
assert np.all(np.sum(samples, axis=1) <= 1.0 + 1e-12), '[TC43] dirichlet in simplex FAILED'

# ---- TC44: stochastic_nucleation_events - deterministic with same seed ----
n1, s1 = stochastic_nucleation_events(0.1, 320.0, 1.0, 0.001, seed=12345)
n2, s2 = stochastic_nucleation_events(0.1, 320.0, 1.0, 0.001, seed=12345)
assert n1 == n2, '[TC44] stochastic nucleation deterministic FAILED'

# ---- TC45: analytical_size_dependent_growth_law - beta=-1 produces increasing L ----
from nucleation_model import analytical_size_dependent_growth_law
t = np.linspace(0, 3600, 100)
L_analytical = analytical_size_dependent_growth_law(t, 0.01, -1.0, 1e-7, 0.5)
assert L_analytical[-1] > L_analytical[0], '[TC45] analytical growth increasing FAILED'
assert np.all(np.isfinite(L_analytical)), '[TC45] analytical growth finite FAILED'

# ---- TC46: population balance solver short integration ----
params = {
    'rho_c': 1560.0, 'kv': 0.5236, 'L0': 1e-9, 'k_g0': 5e-5,
    'E_g': 42000.0, 'g_exp': 1.2, 'alpha': 50.0, 'beta': -0.2,
    'A_prefactor': 1e18, 'kb_sec': 5e7, 'b_exp': 1.8, 'j_exp': 0.8,
    'H_diss': 25000.0, 'S_diss': 70.0, 'c0': 0.55, 'T0': 350.0,
    'V': 0.001, 'gamma0': 0.025
}
solver = PopulationBalanceSolver(params)
T_func = lambda tau: float(linear_cooling(tau, 350.0, 300.0, 7200.0))
sol = solver.solve((0.0, 1800.0), T_func, method='RK45')
assert sol.success, '[TC46] PBE solver success FAILED'
mu, c = solver.get_moments_at_time(sol, 1800.0)
assert len(mu) == 6, '[TC46] PBE moments count FAILED'
assert c > 0, '[TC46] PBE concentration positive FAILED'

# ---- TC47: csd_parameters from PBE solver ----
csd_params = solver.get_csd_parameters(sol, 1800.0)
assert csd_params['N'] > 0, '[TC47] CSD params N positive FAILED'
assert csd_params['L_mean'] > 0, '[TC47] CSD params L_mean positive FAILED'

# ---- TC48: generate_chaotic_mixing_trajectory - returns solution with expected fields ----
sol_chaos = generate_chaotic_mixing_trajectory((0.0, 100.0), y0=[-0.1, 0.5, -0.6])
assert sol_chaos.success, '[TC48] chaos trajectory success FAILED'

# ---- TC49: map_chen_to_supersaturation_fluctuation - returns bounded values ----
t_check = np.linspace(0, 100, 200)
sigma_loc = map_chen_to_supersaturation_fluctuation(t_check, sol_chaos, 0.5)
assert np.all(sigma_loc >= 0) and np.all(sigma_loc <= 5.0), '[TC49] chaos fluctuation bounded FAILED'

# ---- TC50: composition_space_integral - returns finite non-negative result ----
def const_func(x):
    return np.ones(x.shape[0])
integral, err = composition_space_integral(const_func, dim=2, n_samples=5000)
assert integral > 0 and np.isfinite(integral), '[TC50] composition integral positive FAILED'
assert err >= 0, '[TC50] composition integral error non-negative FAILED'

# ---- TC51: uncertainty_quantification_crystallization - returns finite output ----
def simple_model(params):
    return params[0] + 2 * params[1] + 3 * params[2]
param_dists = [
    {'mean': 0.0, 'std': 1.0},
    {'mean': 0.0, 'std': 1.0},
    {'mean': 0.0, 'std': 1.0},
]
mean_val, variance, std_val = uncertainty_quantification_crystallization(
    simple_model, param_dists, level_max=2
)
assert np.isfinite(mean_val), '[TC51] UQ mean finite FAILED'
assert variance >= 0, '[TC51] UQ variance non-negative FAILED'

# ---- TC52: discretize_csd_kmeans - returns correct number of classes ----
L_grid = np.linspace(1e-6, 200e-6, 500)
mu_ln = np.log(50e-6)
sigma_ln = 0.3
N = 1e12
f_vals = (N / (L_grid * sigma_ln * np.sqrt(2 * np.pi))) * \
         np.exp(-0.5 * ((np.log(L_grid) - mu_ln) / sigma_ln) ** 2)
f_vals = np.where(np.isnan(f_vals), 0.0, f_vals)
k = 6
class_sizes, class_counts, boundaries = discretize_csd_kmeans(L_grid, f_vals, k)
assert len(class_sizes) == k, '[TC52] discretize CSD class count FAILED'
assert len(boundaries) == k + 1, '[TC52] discretize CSD boundaries FAILED'

# ---- TC53: diffraction_inversion_feret - returns correct output shapes ----
theta = np.linspace(0.001, 0.05, 50)
wavelength = 632.8e-9
radius = 30e-6
I_clean = fraunhofer_diffraction_particle_size(radius, wavelength, theta)
L_bins, n_L = diffraction_inversion_feret(theta, I_clean, wavelength, L_min=1e-6, L_max=200e-6, n_bins=40)
assert len(L_bins) == 40 and len(n_L) == 40, '[TC53] diffraction inversion shape FAILED'

# ---- TC54: index_set_to_string - empty set formatting ----
s = index_set_to_string([], name="Empty")
assert "∅" in s, '[TC54] index set empty FAILED'

# ---- TC55: format_moment_vector - default names ----
mu = np.array([1.0, 2.0])
s = format_moment_vector(mu)
assert "μ_0" in s and "μ_1" in s, '[TC55] format moment default names FAILED'

print('\n全部 55 个测试通过!\n')
