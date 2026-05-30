# -*- coding: utf-8 -*-

import numpy as np
import os
import sys


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
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_special_functions():
    print_section("1. 特殊函数验证与应用")


    t_vals = np.linspace(0, 3600, 100)
    alpha = 0.01
    k_g = 1e-7
    sigma = 0.5
    from nucleation_model import analytical_size_dependent_growth_law
    L_analytical = analytical_size_dependent_growth_law(t_vals, alpha, -1.0, k_g, sigma)
    print(f"  Lambert W 解析生长解：t=3600s 时 L = {L_analytical[-1]:.4e} m")


    x_test = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    C, S = fresnel_integrals(x_test)
    print(f"  Fresnel 积分验证：")
    for i in range(len(x_test)):
        print(f"    x={x_test[i]:.1f}: C={C[i]:.6f}, S={S[i]:.6f}")


    theta = np.linspace(0.001, 0.1, 200)
    wavelength = 632.8e-9
    radius = 25e-6
    intensity = fraunhofer_diffraction_particle_size(radius, wavelength, theta)
    print(f"  Fraunhofer 衍射：颗粒半径 {radius*1e6:.1f} μm，"
          f"峰值光强角度 {theta[np.argmax(intensity)]*1e3:.3f} mrad")

    return t_vals, L_analytical


def demo_cooling_profiles():
    print_section("2. 程序冷却曲线与过饱和度")

    t_total = 7200.0
    T0, Tf = 350.0, 300.0
    t = np.linspace(0, t_total, 500)


    T_linear = linear_cooling(t, T0, Tf, t_total)
    T_optimal = optimal_cooling_polynomial(t, T0, Tf, t_total, order=3)
    T_sawtooth = sawtooth_cooling(t, T_linear, delta_T=2.0, period=600.0)


    H_diss = 25000.0
    S_diss = 70.0
    c0 = 0.55


    sigma_linear = supersaturation(c0, T_linear, H_diss, S_diss)
    sigma_optimal = supersaturation(c0, T_optimal, H_diss, S_diss)
    sigma_sawtooth = supersaturation(c0, T_sawtooth, H_diss, S_diss)

    print(f"  线性冷却终点过饱和度: σ = {sigma_linear[-1]:.4f}")
    print(f"  最优冷却终点过饱和度: σ = {sigma_optimal[-1]:.4f}")
    print(f"  锯齿波冷却平均过饱和度: σ_avg = {np.mean(sigma_sawtooth):.4f}")
    print(f"  锯齿波冷却过饱和度标准差: σ_std = {np.std(sigma_sawtooth):.4f}")

    return t, T_linear, T_sawtooth, sigma_linear, H_diss, S_diss, c0


def demo_chaotic_mixing(t_span, sigma_base):
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


    from chaotic_mixing import mixing_enhanced_nucleation_rate
    B_avg, B_instant = mixing_enhanced_nucleation_rate(sigma_base, t_check, sol,
                                                        B0=1e15, scale_T=0.5, scale_c=0.3)
    B_base = classical_nucleation_rate(sigma_base, 320.0, A_prefactor=1e15)
    print(f"  基础成核率: B_base = {float(B_base):.4e} #/(m³·s)")
    print(f"  混沌增强平均成核率: B_avg = {B_avg:.4e} #/(m³·s)")
    print(f"  增强因子: {B_avg / max(float(B_base), 1e-100):.2f}x")

    return sol, sigma_local


def demo_nucleation_growth():
    print_section("4. 成核与生长动力学模型")

















    raise NotImplementedError("Hole 3: demo_nucleation_growth is not implemented.")


def demo_population_balance(t, T_linear, H_diss, S_diss, c0):
    print_section("5. 人口平衡方程求解 (矩方法)")

    params = {
        'rho_c': 1560.0,
        'kv': 0.5236,
        'L0': 1e-9,
        'k_g0': 5e-5,
        'E_g': 42000.0,
        'g_exp': 1.2,
        'alpha': 50.0,
        'beta': -0.2,
        'A_prefactor': 1e18,
        'kb_sec': 5e7,
        'b_exp': 1.8,
        'j_exp': 0.8,
        'H_diss': H_diss,
        'S_diss': S_diss,
        'c0': c0,
        'T0': 350.0,
        'V': 0.001,
        'gamma0': 0.025,
    }

    solver = PopulationBalanceSolver(params)
    T_func = lambda tau: float(linear_cooling(tau, 350.0, 300.0, 7200.0))

    sol = solver.solve((0.0, 7200.0), T_func, method='RK45')
    print(f"  PBE 求解成功：{sol.message}")
    print(f"  积分步数: {sol.nfev}")


    t_eval = np.linspace(0, 7200, 100)
    moments_history = []
    c_history = []
    for te in t_eval:
        mu, c = solver.get_moments_at_time(sol, te)
        moments_history.append(mu)
        c_history.append(c)

    moments_history = np.array(moments_history)
    c_history = np.array(c_history)


    mu_final, c_final = solver.get_moments_at_time(sol, 7200.0)
    print(f"\n  t = 7200 s 时的矩量:")
    print(format_moment_vector(mu_final, [f"μ_{i}" for i in range(6)]))
    print(f"  最终浓度: c = {c_final:.6f} kg/kg")


    csd_params = solver.get_csd_parameters(sol, 7200.0)
    print(f"\n  CSD 参数 (对数正态近似):")
    print(f"    总晶数密度: N = {csd_params['N']:.4e} #/m³")
    print(f"    平均尺寸: L_mean = {csd_params['L_mean']*1e6:.3f} μm")
    print(f"    变异系数: CV = {csd_params['CV']:.4f}")

    return sol, t_eval, moments_history, c_history, csd_params


def demo_csd_analysis(L_grid, f_values, csd_params):
    print_section("6. CSD 分析与 K-Means 离散化")


    k_classes = 8
    class_sizes, class_counts, boundaries = discretize_csd_kmeans(L_grid, f_values, k_classes)
    print(f"  K-Means 离散化为 {k_classes} 个尺寸类:")
    print(f"    代表尺寸 (μm): {np.array2string(class_sizes * 1e6, precision=2, separator=' ')}")
    print(f"    各类晶数密度 (#/m³): {np.array2string(class_counts, precision=4, separator=' ')}")


    stats = csd_statistical_moments(L_grid, f_values)
    print(f"\n  CSD 统计特征:")
    print(f"    均值: {stats['mean']*1e6:.3f} μm")
    print(f"    标准差: {stats['std']*1e6:.3f} μm")
    print(f"    偏度: {stats['skewness']:.4f}")
    print(f"    峰度: {stats['kurtosis']:.4f}")


    theta = np.linspace(0.001, 0.05, 100)
    wavelength = 632.8e-9
    intensity = fraunhofer_diffraction_particle_size(stats['mean'] / 2.0, wavelength, theta)

    intensity_noisy = intensity + np.random.normal(0, 0.01 * np.max(intensity), len(intensity))
    L_bins, n_L = diffraction_inversion_feret(theta, intensity_noisy, wavelength,
                                               L_min=1e-6, L_max=200e-6, n_bins=50)
    peak_L = L_bins[np.argmax(n_L)]
    print(f"\n  激光衍射粒度反演:")
    print(f"    反演峰值尺寸: {peak_L*1e6:.2f} μm")
    print(f"    真实平均尺寸: {stats['mean']*1e6:.2f} μm")

    return class_sizes, class_counts


def demo_simplex_integration():
    print_section("7. 多组分溶解度空间积分")


    def gibbs_excess_energy(x):
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


    e_test = [2, 1, 0]
    I_wedge = wedge01_monomial_integral(e_test)
    print(f"\n  楔形区域精确单项式积分:")
    print(f"    ∫_W x² y¹ z⁰ dV = {I_wedge:.8f}")


    I_tet = tetrahedron01_monomial_integral(e_test)
    print(f"  四面体精确单项式积分:")
    print(f"    ∫_T x² y¹ z⁰ dV = {I_tet:.8f}")


    rng = np.random.default_rng(42)
    samples = dirichlet_sample_uniform_simplex(100000, 2, rng)
    x1, x2 = samples[:, 0], samples[:, 1]
    mc_vals = (x1 ** 2) * (x2 ** 1)
    mc_estimate = np.mean(mc_vals) / 2.0
    print(f"  Monte Carlo 验证 (100000 样本): {mc_estimate:.8f}")

    return integral


def demo_sparse_grid_uq():
    print_section("8. 稀疏网格不确定性量化 (UQ)")



    def crystallization_model(params):
        k_g0 = np.exp(params[0])
        E_g = params[1] * 10000.0
        k_b = np.exp(params[2])

        T_avg = 325.0
        R = 8.314
        L_mean = k_g0 * np.exp(-E_g / (R * T_avg)) * (k_b ** 0.1) * 1e6
        return L_mean


    param_dists = [
        {'mean': -9.0, 'std': 0.5},
        {'mean': 4.0, 'std': 0.3},
        {'mean': 16.0, 'std': 1.0},
    ]

    mean_val, variance, std_val = uncertainty_quantification_crystallization(
        crystallization_model, param_dists, level_max=3
    )

    print(f"  三参数稀疏网格 UQ (level_max=3):")
    print(f"    输出期望 (平均尺寸): {mean_val:.4f} μm")
    print(f"    输出方差: {variance:.4f}")
    print(f"    输出标准差: {std_val:.4f} μm")


    def test_func_nd(x):
        return np.exp(-np.sum(x ** 2, axis=1))

    for dim in [2, 3]:
        result = sparse_grid_integrate(test_func_nd, dim, level_max=3)

        from scipy.special import erf
        exact = (np.sqrt(np.pi) * erf(1.0)) ** dim
        print(f"  {dim}D 稀疏网格测试: 数值={result:.8f}, 精确={exact:.8f}, 误差={abs(result-exact):.2e}")

    return mean_val, std_val


def demo_mcmc_inference(c_history_true, t_eval):
    print_section("9. DREAM MCMC 贝叶斯参数推断")


    sigma_noise = 0.005
    data_noisy = c_history_true + np.random.normal(0, sigma_noise, len(c_history_true))


    def forward_model(theta):
        k_g0, E_g, k_b = theta
        if k_g0 <= 0 or E_g <= 0 or k_b <= 0:
            return np.full(len(t_eval), 1e10)


        T = 350.0 - 50.0 * t_eval / 7200.0
        sigma = np.maximum((0.45 - 0.35 * np.exp(-25000.0 / (8.314 * T))) / 0.35, 0.0)

        rate = k_g0 * np.exp(-E_g / (8.314 * T)) * (sigma ** 1.2) * k_b * 1e-8
        c_pred = 0.45 * np.exp(-np.cumsum(rate) * (t_eval[1] - t_eval[0]))
        return c_pred


    mu_ln = np.array([-9.0, np.log(42000.0), np.log(5e7)])
    sigma_ln = np.array([0.5, 0.1, 0.5])

    def log_posterior(theta):
        lp = log_prior_lognormal(theta, mu_ln, sigma_ln)
        if lp == -np.inf:
            return -np.inf
        ll = log_likelihood_gaussian(theta, forward_model, data_noisy, sigma_noise)
        return lp + ll


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
    print_section("10. 数据输入输出与结果管理")


    mu_final = moments_history[-1, :]
    print("  最终矩量格式化输出:")
    print(format_moment_vector(mu_final))


    active_classes = np.where(class_sizes > 1e-7)[0]
    idx_str = index_set_to_string(active_classes, name="ActiveClasses")
    print(f"\n  活跃尺寸类索引集: {idx_str}")


    output_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(output_dir, "simulation_data.txt")
    r8vec2_write(data_file, t_eval, c_history)
    print(f"\n  时间-浓度数据已写入: {data_file}")


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


    meta_read, data_read = read_simulation_results(json_file)
    print(f"\n  读取验证:")
    print(f"    项目名称: {meta_read.get('project', 'N/A')}")
    print(f"    CSD 平均尺寸: {data_read.get('csd_mean_um', 0):.3f} μm")


def main():
    print("\n" + "#" * 70)
    print("#  博士级科研代码合成项目")
    print("#  领域：化学工程 — 结晶过程成核与生长动力学")
    print("#  项目编号：PROJECT_137")
    print("#" * 70)


    demo_special_functions()


    t, T_linear, T_sawtooth, sigma_linear, H_diss, S_diss, c0 = demo_cooling_profiles()


    t_span = (0.0, 7200.0)
    sol_chaos, sigma_local = demo_chaotic_mixing(t_span, np.mean(sigma_linear))


    sigma_range, B_total = demo_nucleation_growth()


    sol_pbe, t_eval, moments_history, c_history, csd_params = demo_population_balance(
        t, T_linear, H_diss, S_diss, c0
    )



    L_grid = np.linspace(1e-6, 200e-6, 500)
    mu_ln = csd_params['mu_ln']
    sigma_ln = csd_params['sigma_ln']
    N = csd_params['N']
    f_values = (N / (L_grid * sigma_ln * np.sqrt(2 * np.pi))) * \
               np.exp(-0.5 * ((np.log(L_grid) - mu_ln) / sigma_ln) ** 2)
    f_values = np.where(np.isnan(f_values), 0.0, f_values)
    class_sizes, class_counts = demo_csd_analysis(L_grid, f_values, csd_params)


    demo_simplex_integration()


    demo_sparse_grid_uq()


    samples, summary = demo_mcmc_inference(c_history, t_eval)


    demo_data_io(sol_pbe, t_eval, moments_history, c_history, csd_params, class_sizes)


    print_section("计算完成")
    print("  所有计算模块已成功执行，无报错。")
    print("  结果文件保存在项目目录中。")
    print("\n" + "#" * 70)


if __name__ == "__main__":
    main()
