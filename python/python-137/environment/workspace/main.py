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

    # TODO: Demonstrate nucleation and growth kinetics models.
    #
    # Required steps:
    # 1. Create sigma_range = np.linspace(0.01, 1.0, 50), set T=320.0, MT=50.0.
    # 2. Compute primary nucleation rate B_prim via classical_nucleation_rate(sigma_range, T, A_prefactor=1e20).
    # 3. Compute secondary nucleation rate B_sec via secondary_nucleation_rate(sigma_range, MT, ...).
    # 4. B_total = B_prim + B_sec.
    # 5. Print nucleation rates at sigma=0.5 (index 24).
    # 6. Compute and print critical nucleus radius r_star at sigma=0.5, T.
    # 7. Compare growth models at L_test=50e-6, sigma=0.5, T=320K:
    #    - power_law_growth
    #    - size_dependent_growth (alpha=100.0, beta=-0.3)
    #    - two_step_growth (k_d=1e-5, k_r0=1e-3, E_r=35000.0, g_r=2.0)
    #    - bcf_spiral_growth (A_bcf=1e-4, B_bcf=0.2, E_act=45000.0)
    # 8. Compute growth_rate_dispersion(G_power, cv=0.15, n_samples=1000) and print stats.
    # 9. Return sigma_range, B_total.
    raise NotImplementedError("Hole 3: demo_nucleation_growth is not implemented.")


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
