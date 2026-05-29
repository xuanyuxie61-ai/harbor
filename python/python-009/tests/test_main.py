#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py
系外行星大气光谱反演系统 — 统一入口

科学问题：
    基于透射光谱观测数据，利用贝叶斯框架反演热木星大气参数，
    包括温度-压强剖面、化学丰度分布、云层光学厚度等。

运行方式：
    python main.py

零参数可运行，内置模拟观测数据与完整反演流程。
"""

import numpy as np
import time
import sys
import os

# 导入子模块
from atmospheric_model import AtmosphericProfile, ChemicalEquilibrium, CloudModel
from mesh_generator import AtmosphericMesh, distance_function_sphere_shell, mesh_size_function
from spectral_synthesis import MolecularCrossSection, RayleighScattering, LineProfile
from radiative_transfer import RadiativeTransferSolver
from sphere_quadrature import gauss_legendre_angles, delta_eddington_approximation
from sparse_linear_algebra import CRSMatrix, crs_gmres
from monte_carlo_sampler import MetropolisHastingsSampler, SimplexSampler, PinkNoiseGenerator, NestedSampler
from inversion_solver import LevenbergMarquardt, BroydenSolver, TikhonovRegularization
from data_io import TecDataset, write_spectrum_ascii, read_spectrum_ascii, save_json_metadata


def generate_synthetic_observation():
    """
    生成模拟观测数据（正向模型）。

    物理场景：
        热木星 HD 189733 b 类似行星
        恒星：类太阳恒星
        观测：凌日透射光谱
    """
    print("=" * 70)
    print("STEP 1: 生成模拟观测数据 (正向模型)")
    print("=" * 70)

    # 行星物理参数
    M_jupiter = 1.898e27  # kg
    R_jupiter = 6.9911e7  # m
    M_sun = 1.98847e30  # kg
    AU = 1.496e11  # m

    planet = AtmosphericProfile(
        planet_mass_kg=1.14 * M_jupiter,
        planet_radius_m=1.138 * R_jupiter,
        star_mass_kg=0.82 * M_sun,
        orbital_distance_m=0.031 * AU
    )

    print(f"  行星平衡温度: T_eq = {planet.T_eq:.1f} K")

    # 构造压强网格 (100 层，1e-1 Pa 到 1e7 Pa)
    mesh = AtmosphericMesh(n_layers=60, P_top=1e-1, P_bot=1e7,
                           planet_radius_m=planet.R_p)
    P = mesh.P_center

    # 真实温度剖面 (Guillot 模型)
    T_true = planet.guillot_temperature_profile(
        P, T_int=150.0, T_irr=planet.T_eq * 1.2,
        gamma=16.0 / 3.0, kappa_ir=1e-2, kappa_v1=6e-3, kappa_v2=1e-4
    )

    # 化学丰度
    chem = ChemicalEquilibrium(['H2', 'He', 'H2O', 'CH4', 'CO', 'CO2', 'Na'])
    abundances = {}
    for sp in chem.species:
        abundances[sp] = chem.equilibrium_abundance(sp, T_true, P,
                                                      metallicity=1.0, C_O_ratio=0.54)

    # 平均分子量
    mu_avg = chem.mean_molecular_weight(abundances)

    # 重力
    g = planet.gravity(np.zeros_like(P))

    # 海拔
    z = planet.altitude_from_pressure(P, T_true, mu_avg)

    # 云层
    cloud = CloudModel(P_cloud_top=1e3, P_cloud_base=1e5,
                       cloud_opacity=2.0, particle_radius_m=1e-6)
    cloud_tau = cloud.cloud_optical_depth(P)

    # 波长网格 (0.3 - 5.0 μm)
    wavelength_um = np.linspace(0.3, 5.0, 150)
    wavenumber_cm = 1e4 / wavelength_um  # cm^-1

    # 计算分子截面
    species_active = ['H2O', 'CH4', 'CO', 'CO2', 'Na']
    cross_sections = {}
    for sp in species_active:
        mol = MolecularCrossSection(sp)
        sigma = np.zeros((len(P), len(wavelength_um)), dtype=np.float64)
        for i in range(len(P)):
            sigma[i, :] = mol.compute_cross_section(wavenumber_cm, T_true[i], P[i])
        cross_sections[sp] = sigma

    # 瑞利散射截面
    rayleigh = RayleighScattering.effective_cross_section(wavelength_um)

    # 辐射传输求解
    rt_solver = RadiativeTransferSolver(wavelength_um, planet.R_p)
    tau_cumulative = rt_solver.compute_optical_depth(
        P, T_true, abundances, cross_sections, g,
        rayleigh_cross_section=rayleigh,
        cloud_optical_depth=cloud_tau[:, None] if cloud_tau.ndim == 1 else cloud_tau
    )

    # 透射光谱
    transit_depth = rt_solver.transit_depth_spectrum(P, T_true, tau_cumulative, z)

    # 添加噪声（模拟观测误差 ~50 ppm）
    noise_level = 50.0  # ppm
    noise = np.random.normal(0.0, noise_level, size=len(wavelength_um))
    transit_depth_noisy = transit_depth + noise
    error = np.full_like(transit_depth_noisy, noise_level)

    print(f"  模拟波长范围: {wavelength_um[0]:.2f} - {wavelength_um[-1]:.2f} μm")
    print(f"  数据点数: {len(wavelength_um)}")
    print(f"  平均透射深度: {np.mean(transit_depth):.2f} ppm")
    print(f"  噪声水平: {noise_level:.1f} ppm")

    # 保存数据
    write_spectrum_ascii(wavelength_um, transit_depth_noisy, error, "observed_spectrum.dat")
    print("  观测数据已保存至 observed_spectrum.dat")

    return {
        'wavelength_um': wavelength_um,
        'wavenumber_cm': wavenumber_cm,
        'transit_depth_obs': transit_depth_noisy,
        'transit_depth_true': transit_depth,
        'error': error,
        'P': P,
        'T_true': T_true,
        'z': z,
        'g': g,
        'abundances_true': abundances,
        'mu_avg': mu_avg,
        'cloud': cloud,
        'planet': planet,
        'mesh': mesh,
        'cross_sections_mol': {sp: MolecularCrossSection(sp) for sp in species_active}
    }


def forward_model(params, data_dict):
    """
    正向模型：从参数计算理论透射光谱。

    参数向量:
        params[0]: T_int (K) - 内部温度
        params[1]: log10(metallicity)
        params[2]: C/O ratio
        params[3]: log10(cloud_opacity)
        params[4]: log10(P_cloud_top) (Pa)
        params[5:5+N_layers-3]: T_perturbation (温度剖面扰动，相对于 Guillot 模型)
    """
    wavelength_um = data_dict['wavelength_um']
    wavenumber_cm = data_dict['wavenumber_cm']
    P = data_dict['P']
    g = data_dict['g']
    planet = data_dict['planet']
    mesh = data_dict['mesh']
    cs_mols = data_dict['cross_sections_mol']

    n_layers = len(P)

    # 解析参数
    T_int = params[0]
    metallicity = 10.0**params[1]
    C_O = params[2]
    cloud_opacity = 10.0**params[3]
    P_cloud_top = 10.0**params[4]

    # 温度剖面: Guillot + 扰动
    T_base = planet.guillot_temperature_profile(P, T_int=T_int, T_irr=planet.T_eq * 1.2)
    n_perturb = min(len(params) - 5, n_layers)
    if n_perturb > 0:
        perturb = np.zeros(n_layers)
        perturb[:n_perturb] = params[5:5 + n_perturb]
        # 对扰动进行平滑（融合 diffuse 思想）
        for _ in range(2):
            perturb[1:-1] = 0.5 * perturb[1:-1] + 0.25 * (perturb[:-2] + perturb[2:])
        T = T_base + perturb
        T = np.clip(T, 50.0, 5000.0)
    else:
        T = T_base

    # 化学丰度
    chem = ChemicalEquilibrium(['H2', 'He', 'H2O', 'CH4', 'CO', 'CO2', 'Na'])
    abundances = {}
    for sp in chem.species:
        abundances[sp] = chem.equilibrium_abundance(sp, T, P, metallicity=metallicity, C_O_ratio=C_O)
    mu_avg = chem.mean_molecular_weight(abundances)

    # 海拔
    z = planet.altitude_from_pressure(P, T, mu_avg)

    # 云层
    cloud = CloudModel(P_cloud_top=P_cloud_top, P_cloud_base=P_cloud_top * 100.0,
                       cloud_opacity=cloud_opacity, particle_radius_m=1e-6)
    cloud_tau = cloud.cloud_optical_depth(P)

    # 截面
    cross_sections = {}
    for sp, mol in cs_mols.items():
        sigma = np.zeros((n_layers, len(wavelength_um)), dtype=np.float64)
        for i in range(n_layers):
            sigma[i, :] = mol.compute_cross_section(wavenumber_cm, T[i], P[i])
        cross_sections[sp] = sigma

    # 瑞利散射
    rayleigh = RayleighScattering.effective_cross_section(wavelength_um)

    # 辐射传输
    rt_solver = RadiativeTransferSolver(wavelength_um, planet.R_p)
    tau_cumulative = rt_solver.compute_optical_depth(
        P, T, abundances, cross_sections, g,
        rayleigh_cross_section=rayleigh,
        cloud_optical_depth=cloud_tau[:, None]
    )

    transit_depth = rt_solver.transit_depth_spectrum(P, T, tau_cumulative, z)
    return transit_depth


def inversion_least_squares(data_dict):
    """
    最小二乘反演：使用 Levenberg-Marquardt 优化。
    """
    print("\n" + "=" * 70)
    print("STEP 2: 最小二乘参数反演 (Levenberg-Marquardt)")
    print("=" * 70)

    obs = data_dict['transit_depth_obs']
    err = data_dict['error']
    n_layers = len(data_dict['P'])

    # 参数初始化
    # [T_int, log10(Z), C/O, log10(cloud_op), log10(P_cloud), T_perturb...]
    n_perturb = min(5, n_layers)
    x0 = np.array([100.0, 0.0, 0.54, 0.0, 2.0] + [0.0] * n_perturb)

    def residual_func(x):
        model = forward_model(x, data_dict)
        return (model - obs) / err

    def jacobian_func(x):
        eps = 1e-4
        r0 = residual_func(x)
        J = np.zeros((len(r0), len(x)))
        for j in range(len(x)):
            xj = x.copy()
            h = eps * max(abs(xj[j]), 1.0)
            xj[j] += h
            J[:, j] = (residual_func(xj) - r0) / h
        return J

    lm = LevenbergMarquardt(max_iter=30, tol=1e-4)
    x_opt, iters, final_cost = lm.solve(residual_func, jacobian_func, x0)

    print(f"  迭代次数: {iters}")
    print(f"  最终 χ²: {2 * final_cost:.2f}")
    print(f"  优化参数:")
    print(f"    T_int = {x_opt[0]:.1f} K")
    print(f"    [M/H] = {10**x_opt[1]:.2f} × 太阳")
    print(f"    C/O   = {x_opt[2]:.3f}")
    print(f"    Cloud τ = {10**x_opt[3]:.2f}")
    print(f"    P_cloud = {10**x_opt[4]:.1e} Pa")

    return x_opt


def bayesian_mcmc_analysis(data_dict, x_init):
    """
    贝叶斯 MCMC 不确定性分析。
    """
    print("\n" + "=" * 70)
    print("STEP 3: 贝叶斯 MCMC 不确定性量化")
    print("=" * 70)

    obs = data_dict['transit_depth_obs']
    err = data_dict['error']

    def log_posterior(x):
        # 边界检查
        if x[0] < 10 or x[0] > 1000:
            return -1e300
        if x[1] < -2 or x[1] > 2:
            return -1e300
        if x[2] < 0.1 or x[2] > 2.0:
            return -1e300
        if x[3] < -2 or x[3] > 3:
            return -1e300
        if x[4] < -1 or x[4] > 6:
            return -1e300

        try:
            model = forward_model(x, data_dict)
            chi2 = np.sum(((model - obs) / err) ** 2)
            # 先验约束
            log_prior = -0.5 * ((x[0] - 150.0) / 100.0) ** 2
            log_prior += -0.5 * (x[1] / 1.0) ** 2
            log_prior += -0.5 * ((x[2] - 0.54) / 0.3) ** 2
            return -0.5 * chi2 + log_prior
        except Exception:
            return -1e300

    n_params = len(x_init)
    proposal_cov = np.diag([20.0, 0.1, 0.05, 0.1, 0.2] + [10.0] * (n_params - 5))
    bounds = [(10, 1000), (-2, 2), (0.1, 2.0), (-2, 3), (-1, 6)]
    for _ in range(n_params - 5):
        bounds.append((-200, 200))

    sampler = MetropolisHastingsSampler(log_posterior, proposal_cov, bounds)
    samples, log_probs, acc_rate = sampler.sample(
        x_init, n_samples=500, burn_in=200, thin=2, seed=42
    )

    print(f"  采样数: {len(samples)}")
    print(f"  接受率: {acc_rate:.3f}")

    # 参数统计
    param_names = ['T_int (K)', 'log10([M/H])', 'C/O', 'log10(cloud_τ)', 'log10(P_cloud)']
    print(f"  后验参数估计 (均值 ± 标准差):")
    for i in range(min(5, n_params)):
        mean = np.mean(samples[:, i])
        std = np.std(samples[:, i])
        print(f"    {param_names[i]:20s}: {mean:8.3f} ± {std:.3f}")

    return samples, log_probs


def numerical_diagnostics(data_dict, x_opt):
    """
    数值诊断与稳定性测试。
    """
    print("\n" + "=" * 70)
    print("STEP 4: 数值诊断与模型验证")
    print("=" * 70)

    # 1. 网格质量评估
    mesh = data_dict['mesh']
    nodes_2d, elements_2d = mesh.generate_2d_shell_mesh(n_angular=16)
    metrics = mesh.mesh_quality_metrics(nodes_2d, elements_2d)
    print(f"  2D 壳层网格质量:")
    print(f"    最小角: {metrics['min_angle_deg']:.1f}°")
    print(f"    最大角: {metrics['max_angle_deg']:.1f}°")
    print(f"    平均质量因子: {metrics['mean_quality']:.3f}")
    print(f"    最大纵横比: {metrics['max_aspect_ratio']:.2f}")

    # 2. 球面积分测试
    mu, w_mu, phi, w_phi = gauss_legendre_angles(n_polar=8, n_azimuth=16)
    # 积分常数函数应得 4π
    f_const = np.ones((len(mu), len(phi)))
    from sphere_quadrature import integrate_sphere_function
    I_const = integrate_sphere_function(f_const, w_mu, w_phi)
    print(f"\n  球面积分测试:")
    print(f"    ∫ 1 dΩ = {I_const:.6f}  (理论值: {4*np.pi:.6f})")
    print(f"    相对误差: {abs(I_const - 4*np.pi) / (4*np.pi) * 100:.4f}%")

    # 3. 相函数归一化测试
    from sphere_quadrature import henyey_greenstein_phase_function, compute_scatter_angles
    cos_theta = np.linspace(-1, 1, 200)
    p_hg = henyey_greenstein_phase_function(cos_theta, g=0.5)
    # 数值积分验证
    I_hg = 2 * np.pi * np.trapezoid(p_hg, cos_theta)
    print(f"\n  Henyey-Greenstein 相函数归一化:")
    print(f"    ∫ P(cosΘ) dΩ = {I_hg:.6f}  (理论值: 1.0)")

    # 4. Delta-Eddington 近似验证
    tau_test = np.linspace(0.1, 10.0, 50)
    omega_test = np.full_like(tau_test, 0.8)
    R, T = delta_eddington_approximation(tau_test, omega_test, g=0.5, mu0=0.5)
    conservation = R + T
    print(f"\n  Delta-Eddington 能量守恒:")
    print(f"    max|R + T - 1| = {np.max(np.abs(conservation - 1.0)):.6e}")

    # 5. 稀疏矩阵运算测试
    n_test = 100
    A_dense = np.diag(np.ones(n_test) * 2.0) + np.diag(np.ones(n_test - 1) * (-1.0), 1) + np.diag(np.ones(n_test - 1) * (-1.0), -1)
    crs_A = CRSMatrix.from_dense(A_dense)
    x_true = np.random.randn(n_test)
    b = crs_A.multiply(x_true)
    x_solved, iters, res = crs_gmres(crs_A, b, tol=1e-10, max_iter=200)
    rel_err = np.linalg.norm(x_solved - x_true) / np.linalg.norm(x_true)
    print(f"\n  稀疏线性求解器测试 (GMRES):")
    print(f"    迭代次数: {iters}")
    print(f"    相对残差: {res:.2e}")
    print(f"    解相对误差: {rel_err:.2e}")

    # 6. 单纯形采样测试
    samples_simplex = SimplexSampler.sample_unit_simplex(m=5, n=10000)
    mean_sum = np.mean(np.sum(samples_simplex, axis=1))
    print(f"\n  单纯形均匀采样测试:")
    print(f"    Σ x_i 均值 = {mean_sum:.6f}  (理论值: 1.0)")

    # 7. 模型预测对比
    model_best = forward_model(x_opt, data_dict)
    residuals = (data_dict['transit_depth_obs'] - model_best) / data_dict['error']
    chi2_reduced = np.sum(residuals**2) / (len(residuals) - len(x_opt))
    print(f"\n  反演结果统计:")
    print(f"    还原 χ² / 自由度 = {chi2_reduced:.3f}")
    print(f"    残差 RMS = {np.std(residuals):.3f}")

    return {
        'mesh_quality': metrics,
        'sphere_integral_error': abs(I_const - 4 * np.pi) / (4 * np.pi),
        'hg_normalization_error': abs(I_hg - 1.0),
        'gmres_relative_error': rel_err,
        'chi2_reduced': chi2_reduced
    }


def main():
    """主程序入口。"""
    print("\n" + "#" * 70)
    print("#  系外行星大气光谱反演系统")
    print("#  Exoplanet Atmospheric Spectral Retrieval System")
    print("#" * 70)
    print(f"#  运行时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 70 + "\n")

    np.random.seed(42)
    t_start = time.time()

    # Step 1: 生成模拟观测
    data_dict = generate_synthetic_observation()

    # Step 2: 最小二乘反演
    x_opt = inversion_least_squares(data_dict)

    # Step 3: 贝叶斯 MCMC
    samples, log_probs = bayesian_mcmc_analysis(data_dict, x_opt)

    # Step 4: 数值诊断
    diagnostics = numerical_diagnostics(data_dict, x_opt)

    # Step 5: 保存结果
    print("\n" + "=" * 70)
    print("STEP 5: 结果保存")
    print("=" * 70)

    # 保存优化后的模型光谱
    model_best = forward_model(x_opt, data_dict)
    write_spectrum_ascii(data_dict['wavelength_um'], model_best, None, "retrieved_spectrum.dat")

    # 保存温度剖面对比
    T_retrieved = data_dict['planet'].guillot_temperature_profile(
        data_dict['P'], T_int=x_opt[0], T_irr=data_dict['planet'].T_eq * 1.2
    )
    n_perturb = min(len(x_opt) - 5, len(data_dict['P']))
    if n_perturb > 0:
        perturb = np.zeros(len(data_dict['P']))
        perturb[:n_perturb] = x_opt[5:5 + n_perturb]
        for _ in range(2):
            perturb[1:-1] = 0.5 * perturb[1:-1] + 0.25 * (perturb[:-2] + perturb[2:])
        T_retrieved = T_retrieved + perturb
        T_retrieved = np.clip(T_retrieved, 50.0, 5000.0)

    with open("temperature_profile_comparison.dat", "w") as f:
        f.write("# P(Pa)  T_true(K)  T_retrieved(K)\n")
        for i in range(len(data_dict['P'])):
            f.write(f"{data_dict['P'][i]:.6e} {data_dict['T_true'][i]:.4f} {T_retrieved[i]:.4f}\n")

    # 保存 MCMC 样本
    np.savetxt("mcmc_samples.dat", samples, fmt="%.6f")

    # 保存元数据
    metadata = {
        'project': 'Exoplanet Atmospheric Spectral Retrieval',
        'planet_radius_m': float(data_dict['planet'].R_p),
        'equilibrium_temperature_K': float(data_dict['planet'].T_eq),
        'n_wavelengths': int(len(data_dict['wavelength_um'])),
        'n_layers': int(len(data_dict['P'])),
        'optimized_parameters': {
            'T_int_K': float(x_opt[0]),
            'metallicity': float(10**x_opt[1]),
            'C_O_ratio': float(x_opt[2]),
            'cloud_optical_depth': float(10**x_opt[3]),
            'cloud_top_pressure_Pa': float(10**x_opt[4])
        },
        'diagnostics': {k: float(v) if isinstance(v, (int, float, np.floating)) else str(v)
                        for k, v in diagnostics.items()}
    }
    save_json_metadata(metadata, "retrieval_metadata.json")

    print("  retrieved_spectrum.dat       — 反演模型光谱")
    print("  temperature_profile_comparison.dat — 温度剖面对比")
    print("  mcmc_samples.dat             — MCMC 后验样本")
    print("  retrieval_metadata.json      — 反演元数据")

    t_elapsed = time.time() - t_start
    print("\n" + "#" * 70)
    print(f"#  运行完成，总耗时: {t_elapsed:.2f} 秒")
    print("#" * 70 + "\n")

    return 0


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（30个，assert模式，涉及随机值均使用固定种子）
# ================================================================

from sphere_quadrature import integrate_sphere_function, henyey_greenstein_phase_function, compute_scatter_angles, spherical_to_cartesian, cartesian_to_spherical
from inversion_solver import BisectionRootFinder
from data_io import convert_triangle_to_fem


# ---- TC01: AtmosphericProfile gravity returns finite positive values ----
planet = AtmosphericProfile(planet_mass_kg=1.898e27, planet_radius_m=6.9911e7, star_mass_kg=1.98847e30, orbital_distance_m=1.496e11)
g = planet.gravity(np.array([0.0, 1e6, 1e7]))
assert np.all(g > 0) and np.all(np.isfinite(g)), '[TC01] AtmosphericProfile gravity returns finite positive values FAILED'

# ---- TC02: Guillot temperature profile finite and positive ----
P = np.logspace(1, 5, 10)
T = planet.guillot_temperature_profile(P, T_int=150.0, T_irr=1000.0)
assert np.all(T > 0) and np.all(np.isfinite(T)), '[TC02] Guillot temperature profile finite and positive FAILED'

# ---- TC03: Hydrostatic pressure grid monotonic increasing ----
P_grid = planet.hydrostatic_pressure_grid(10, 1e-1, 1e7)
assert np.all(np.diff(P_grid) > 0), '[TC03] Hydrostatic pressure grid monotonic increasing FAILED'

# ---- TC04: Scale height positive and finite ----
H = planet.scale_height(1000.0, 2.3)
assert H > 0 and np.isfinite(H), '[TC04] Scale height positive and finite FAILED'

# ---- TC05: ChemicalEquilibrium H2 abundance equals 0.85 ----
chem = ChemicalEquilibrium(['H2', 'He', 'H2O'])
vmr_h2 = chem.equilibrium_abundance('H2', np.array([1000.0]), np.array([1e5]))
assert abs(vmr_h2[0] - 0.85) < 1e-10, '[TC05] ChemicalEquilibrium H2 abundance equals 0.85 FAILED'

# ---- TC06: Mean molecular weight calculation in expected range ----
abundances = {'H2': np.array([0.85]), 'He': np.array([0.15])}
mu = chem.mean_molecular_weight(abundances)
assert 2.0 < mu[0] < 3.0, '[TC06] Mean molecular weight calculation in expected range FAILED'

# ---- TC07: Cloud optical depth non-negative ----
cloud = CloudModel(P_cloud_top=1e3, P_cloud_base=1e5, cloud_opacity=2.0)
tau_cloud = cloud.cloud_optical_depth(P_grid)
assert np.all(tau_cloud >= 0), '[TC07] Cloud optical depth non-negative FAILED'

# ---- TC08: 2D shell mesh generation shape correctness ----
mesh = AtmosphericMesh(n_layers=10, P_top=1e-1, P_bot=1e7, planet_radius_m=6.9911e7)
nodes, elements = mesh.generate_2d_shell_mesh(n_angular=8)
assert nodes.shape[1] == 2 and elements.shape[1] == 3, '[TC08] 2D shell mesh generation shape correctness FAILED'

# ---- TC09: Distance function sphere shell sign properties ----
points = np.array([[0.0, 0.0], [5.0, 0.0], [15.0, 0.0]])
d = distance_function_sphere_shell(points, R_inner=5.0, R_outer=10.0)
assert d[0] > 0 and d[1] <= 0 and d[2] > 0, '[TC09] Distance function sphere shell sign properties FAILED'

# ---- TC10: Mesh size function within bounds ----
pts = np.array([[6.9911e7, 0.0], [7.5e7, 0.0]])
h = mesh_size_function(pts, R_p=6.9911e7, h_min=1e3, h_max=1e5)
assert np.all(h >= 1e3) and np.all(h <= 1e5), '[TC10] Mesh size function within bounds FAILED'

# ---- TC11: Voigt profile positive and finite ----
nu = np.linspace(-1e10, 1e10, 1000)
profile = LineProfile.voigt_profile(nu, nu0=0.0, alpha_d=1e9, gamma_l=1e8)
assert np.all(profile >= 0) and np.all(np.isfinite(profile)), '[TC11] Voigt profile positive and finite FAILED'

# ---- TC12: Voigt profile normalization integral near unity ----
nu_fine = np.linspace(-5e10, 5e10, 20001)
profile_norm = LineProfile.voigt_profile(nu_fine, nu0=0.0, alpha_d=1e9, gamma_l=1e8)
I_voigt = np.trapezoid(profile_norm, nu_fine)
assert abs(I_voigt - 1.0) < 0.1, '[TC12] Voigt profile normalization integral near unity FAILED'

# ---- TC13: Molecular cross section non-negative and finite ----
mol = MolecularCrossSection('H2O')
wn = np.linspace(1000, 7000, 100)
sigma = mol.compute_cross_section(wn, T=1000.0, P=1e5)
assert np.all(sigma >= 0) and np.all(np.isfinite(sigma)), '[TC13] Molecular cross section non-negative and finite FAILED'

# ---- TC14: Rayleigh scattering follows lambda^-4 scaling ----
lam = np.array([1.0, 2.0])
sigma_r = RayleighScattering.cross_section_H2(lam)
assert abs(sigma_r[1] / sigma_r[0] - (1.0/2.0)**4) < 1e-10, '[TC14] Rayleigh scattering follows lambda^-4 scaling FAILED'

# ---- TC15: Radiative transfer optical depth monotonic ----
rt = RadiativeTransferSolver(np.linspace(0.3, 5.0, 10), planet_radius_m=6.9911e7)
P_small = np.logspace(3, 5, 5)
T_small = np.full_like(P_small, 1000.0)
abund = {'H2': np.full_like(P_small, 0.85), 'He': np.full_like(P_small, 0.15)}
cs = {'H2': np.zeros((5, 10)), 'He': np.zeros((5, 10))}
g_small = np.full_like(P_small, 10.0)
tau = rt.compute_optical_depth(P_small, T_small, abund, cs, g_small)
assert np.all(np.diff(tau[:, 0]) >= -1e-15), '[TC15] Radiative transfer optical depth monotonic FAILED'

# ---- TC16: Gauss-Legendre weights sum to 2 ----
mu_gl, w_mu_gl, phi_gl, w_phi_gl = gauss_legendre_angles(n_polar=8, n_azimuth=16)
assert abs(np.sum(w_mu_gl) - 2.0) < 1e-14, '[TC16] Gauss-Legendre weights sum to 2 FAILED'

# ---- TC17: Sphere integral of constant function equals 4pi ----
f_const = np.ones((8, 16))
I_sphere = integrate_sphere_function(f_const, w_mu_gl, w_phi_gl)
assert abs(I_sphere - 4.0 * np.pi) < 1e-10, '[TC17] Sphere integral of constant function equals 4pi FAILED'

# ---- TC18: Henyey-Greenstein isotropic for g=0 ----
cos_theta = np.linspace(-1, 1, 100)
p_hg = henyey_greenstein_phase_function(cos_theta, g=0.0)
assert np.allclose(p_hg, 1.0/(4.0*np.pi)), '[TC18] Henyey-Greenstein isotropic for g=0 FAILED'

# ---- TC19: Delta-Eddington energy conservation R+T bounded ----
tau_test = np.linspace(0.1, 5.0, 10)
omega_test = np.full_like(tau_test, 0.8)
R_de, T_de = delta_eddington_approximation(tau_test, omega_test, g=0.5, mu0=0.5)
assert np.all(R_de >= 0) and np.all(T_de >= 0) and np.all(R_de <= 1.0) and np.all(T_de <= 1.0), '[TC19] Delta-Eddington R and T within [0,1] FAILED'

# ---- TC20: CRS matrix from dense and multiply correctness ----
A_dense = np.array([[2.0, 1.0], [1.0, 2.0]])
crs = CRSMatrix.from_dense(A_dense)
x_vec = np.array([1.0, 2.0])
y_vec = crs.multiply(x_vec)
assert np.allclose(y_vec, A_dense @ x_vec), '[TC20] CRS matrix from dense and multiply correctness FAILED'

# ---- TC21: GMRES solver accuracy for tridiagonal system ----
n_test = 20
A_tridiag = np.diag(np.ones(n_test)*2.0) + np.diag(np.ones(n_test-1)*(-1.0), 1) + np.diag(np.ones(n_test-1)*(-1.0), -1)
crs_A = CRSMatrix.from_dense(A_tridiag)
x_true = np.ones(n_test)
b_vec = crs_A.multiply(x_true)
x_sol, iters_gmres, res_gmres = crs_gmres(crs_A, b_vec, tol=1e-10, max_iter=200)
rel_err = np.linalg.norm(x_sol - x_true) / np.linalg.norm(x_true)
assert rel_err < 1e-6, '[TC21] GMRES solver accuracy for tridiagonal system FAILED'

# ---- TC22: Simplex sampler sum to one ----
np.random.seed(42)
samples = SimplexSampler.sample_unit_simplex(m=5, n=1000, seed=42)
sum_samples = np.sum(samples, axis=1)
assert np.allclose(sum_samples, 1.0), '[TC22] Simplex sampler sum to one FAILED'

# ---- TC23: Pink noise zero mean and unit std ----
np.random.seed(42)
png = PinkNoiseGenerator(beta=1.0)
noise = png.generate(n=10000, seed=42)
assert abs(np.mean(noise)) < 0.05 and abs(np.std(noise) - 1.0) < 0.05, '[TC23] Pink noise zero mean and unit std FAILED'

# ---- TC24: Tikhonov first order difference matrix shape ----
L1 = TikhonovRegularization.first_order_difference_matrix(5)
assert L1.shape == (4, 5), '[TC24] Tikhonov first order difference matrix shape FAILED'

# ---- TC25: Bisection root finder solves x^2-4=0 ----
finder = BisectionRootFinder(a=0.0, b=5.0, tol=1e-10)
root = finder.solve(lambda x: x**2 - 4.0)
assert abs(root - 2.0) < 1e-9, '[TC25] Bisection root finder solves x^2-4=0 FAILED'

# ---- TC26: TecDataset variable add and retrieve ----
ds = TecDataset(title="Test", variables=[])
ds.add_variable("X", np.array([1.0, 2.0, 3.0]))
ds.add_variable("Y", np.array([4.0, 5.0, 6.0]))
assert np.allclose(ds.get_variable("X"), np.array([1.0, 2.0, 3.0])), '[TC26] TecDataset variable add and retrieve FAILED'

# ---- TC27: Triangle to FEM 1-based index conversion ----
nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
elems_1based = np.array([[1, 2, 3]])
fem_nodes, fem_elems = convert_triangle_to_fem(nodes, elems_1based)
assert np.allclose(fem_elems, np.array([[0, 1, 2]])), '[TC27] Triangle to FEM 1-based index conversion FAILED'

# ---- TC28: Spectrum ASCII write and read roundtrip ----
wavelength = np.array([1.0, 2.0, 3.0])
flux = np.array([10.0, 20.0, 30.0])
error = np.array([0.1, 0.2, 0.3])
write_spectrum_ascii(wavelength, flux, error, "test_spectrum_tmp.dat")
data_back = read_spectrum_ascii("test_spectrum_tmp.dat")
assert np.allclose(data_back['wavelength'], wavelength) and np.allclose(data_back['flux'], flux), '[TC28] Spectrum ASCII write and read roundtrip FAILED'
import os
os.remove("test_spectrum_tmp.dat")

# ---- TC29: Coordinate conversion roundtrip ----
theta = np.array([0.5, 1.0, 1.5])
phi = np.array([0.0, 1.0, 2.0])
x, y, z = spherical_to_cartesian(theta, phi)
theta_back, phi_back = cartesian_to_spherical(x, y, z)
assert np.allclose(theta, theta_back) and np.allclose(phi, phi_back), '[TC29] Coordinate conversion roundtrip FAILED'

# ---- TC30: Scatter angle same direction equals 1 ----
mu_s = np.array([0.5])
phi_s = np.array([1.0])
cos_scatter = compute_scatter_angles(mu_s, phi_s, mu_s, phi_s)
assert np.allclose(cos_scatter, 1.0), '[TC30] Scatter angle same direction equals 1 FAILED'

print('\n全部 30 个测试通过!\n')
