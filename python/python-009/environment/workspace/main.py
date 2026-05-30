#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
import sys
import os


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
    print("=" * 70)
    print("STEP 1: 生成模拟观测数据 (正向模型)")
    print("=" * 70)


    M_jupiter = 1.898e27
    R_jupiter = 6.9911e7
    M_sun = 1.98847e30
    AU = 1.496e11

    planet = AtmosphericProfile(
        planet_mass_kg=1.14 * M_jupiter,
        planet_radius_m=1.138 * R_jupiter,
        star_mass_kg=0.82 * M_sun,
        orbital_distance_m=0.031 * AU
    )

    print(f"  行星平衡温度: T_eq = {planet.T_eq:.1f} K")


    mesh = AtmosphericMesh(n_layers=60, P_top=1e-1, P_bot=1e7,
                           planet_radius_m=planet.R_p)
    P = mesh.P_center


    T_true = planet.guillot_temperature_profile(
        P, T_int=150.0, T_irr=planet.T_eq * 1.2,
        gamma=16.0 / 3.0, kappa_ir=1e-2, kappa_v1=6e-3, kappa_v2=1e-4
    )


    chem = ChemicalEquilibrium(['H2', 'He', 'H2O', 'CH4', 'CO', 'CO2', 'Na'])
    abundances = {}
    for sp in chem.species:
        abundances[sp] = chem.equilibrium_abundance(sp, T_true, P,
                                                      metallicity=1.0, C_O_ratio=0.54)


    mu_avg = chem.mean_molecular_weight(abundances)


    g = planet.gravity(np.zeros_like(P))


    z = planet.altitude_from_pressure(P, T_true, mu_avg)


    cloud = CloudModel(P_cloud_top=1e3, P_cloud_base=1e5,
                       cloud_opacity=2.0, particle_radius_m=1e-6)
    cloud_tau = cloud.cloud_optical_depth(P)


    wavelength_um = np.linspace(0.3, 5.0, 150)
    wavenumber_cm = 1e4 / wavelength_um


    species_active = ['H2O', 'CH4', 'CO', 'CO2', 'Na']
    cross_sections = {}
    for sp in species_active:
        mol = MolecularCrossSection(sp)
        sigma = np.zeros((len(P), len(wavelength_um)), dtype=np.float64)
        for i in range(len(P)):
            sigma[i, :] = mol.compute_cross_section(wavenumber_cm, T_true[i], P[i])
        cross_sections[sp] = sigma


    rayleigh = RayleighScattering.effective_cross_section(wavelength_um)


    rt_solver = RadiativeTransferSolver(wavelength_um, planet.R_p)
    tau_cumulative = rt_solver.compute_optical_depth(
        P, T_true, abundances, cross_sections, g,
        rayleigh_cross_section=rayleigh,
        cloud_optical_depth=cloud_tau[:, None] if cloud_tau.ndim == 1 else cloud_tau
    )


    transit_depth = rt_solver.transit_depth_spectrum(P, T_true, tau_cumulative, z)


    noise_level = 50.0
    noise = np.random.normal(0.0, noise_level, size=len(wavelength_um))
    transit_depth_noisy = transit_depth + noise
    error = np.full_like(transit_depth_noisy, noise_level)

    print(f"  模拟波长范围: {wavelength_um[0]:.2f} - {wavelength_um[-1]:.2f} μm")
    print(f"  数据点数: {len(wavelength_um)}")
    print(f"  平均透射深度: {np.mean(transit_depth):.2f} ppm")
    print(f"  噪声水平: {noise_level:.1f} ppm")


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
    wavelength_um = data_dict['wavelength_um']
    wavenumber_cm = data_dict['wavenumber_cm']
    P = data_dict['P']
    g = data_dict['g']
    planet = data_dict['planet']
    mesh = data_dict['mesh']
    cs_mols = data_dict['cross_sections_mol']

    n_layers = len(P)


    T_int = params[0]
    metallicity = 10.0**params[1]
    C_O = params[2]
    cloud_opacity = 10.0**params[3]
    P_cloud_top = 10.0**params[4]


    T_base = planet.guillot_temperature_profile(P, T_int=T_int, T_irr=planet.T_eq * 1.2)
    n_perturb = min(len(params) - 5, n_layers)
    if n_perturb > 0:
        perturb = np.zeros(n_layers)
        perturb[:n_perturb] = params[5:5 + n_perturb]

        for _ in range(2):
            perturb[1:-1] = 0.5 * perturb[1:-1] + 0.25 * (perturb[:-2] + perturb[2:])
        T = T_base + perturb
        T = np.clip(T, 50.0, 5000.0)
    else:
        T = T_base


    chem = ChemicalEquilibrium(['H2', 'He', 'H2O', 'CH4', 'CO', 'CO2', 'Na'])
    abundances = {}
    for sp in chem.species:
        abundances[sp] = chem.equilibrium_abundance(sp, T, P, metallicity=metallicity, C_O_ratio=C_O)
    mu_avg = chem.mean_molecular_weight(abundances)









    transit_depth = np.zeros(len(wavelength_um))
    return transit_depth


def inversion_least_squares(data_dict):
    print("\n" + "=" * 70)
    print("STEP 2: 最小二乘参数反演 (Levenberg-Marquardt)")
    print("=" * 70)

    obs = data_dict['transit_depth_obs']
    err = data_dict['error']
    n_layers = len(data_dict['P'])



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
    print("\n" + "=" * 70)
    print("STEP 3: 贝叶斯 MCMC 不确定性量化")
    print("=" * 70)

    obs = data_dict['transit_depth_obs']
    err = data_dict['error']

    def log_posterior(x):

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


    param_names = ['T_int (K)', 'log10([M/H])', 'C/O', 'log10(cloud_τ)', 'log10(P_cloud)']
    print(f"  后验参数估计 (均值 ± 标准差):")
    for i in range(min(5, n_params)):
        mean = np.mean(samples[:, i])
        std = np.std(samples[:, i])
        print(f"    {param_names[i]:20s}: {mean:8.3f} ± {std:.3f}")

    return samples, log_probs


def numerical_diagnostics(data_dict, x_opt):
    print("\n" + "=" * 70)
    print("STEP 4: 数值诊断与模型验证")
    print("=" * 70)


    mesh = data_dict['mesh']
    nodes_2d, elements_2d = mesh.generate_2d_shell_mesh(n_angular=16)
    metrics = mesh.mesh_quality_metrics(nodes_2d, elements_2d)
    print(f"  2D 壳层网格质量:")
    print(f"    最小角: {metrics['min_angle_deg']:.1f}°")
    print(f"    最大角: {metrics['max_angle_deg']:.1f}°")
    print(f"    平均质量因子: {metrics['mean_quality']:.3f}")
    print(f"    最大纵横比: {metrics['max_aspect_ratio']:.2f}")


    mu, w_mu, phi, w_phi = gauss_legendre_angles(n_polar=8, n_azimuth=16)

    f_const = np.ones((len(mu), len(phi)))
    from sphere_quadrature import integrate_sphere_function
    I_const = integrate_sphere_function(f_const, w_mu, w_phi)
    print(f"\n  球面积分测试:")
    print(f"    ∫ 1 dΩ = {I_const:.6f}  (理论值: {4*np.pi:.6f})")
    print(f"    相对误差: {abs(I_const - 4*np.pi) / (4*np.pi) * 100:.4f}%")


    from sphere_quadrature import henyey_greenstein_phase_function, compute_scatter_angles
    cos_theta = np.linspace(-1, 1, 200)
    p_hg = henyey_greenstein_phase_function(cos_theta, g=0.5)

    I_hg = 2 * np.pi * np.trapezoid(p_hg, cos_theta)
    print(f"\n  Henyey-Greenstein 相函数归一化:")
    print(f"    ∫ P(cosΘ) dΩ = {I_hg:.6f}  (理论值: 1.0)")


    tau_test = np.linspace(0.1, 10.0, 50)
    omega_test = np.full_like(tau_test, 0.8)
    R, T = delta_eddington_approximation(tau_test, omega_test, g=0.5, mu0=0.5)
    conservation = R + T
    print(f"\n  Delta-Eddington 能量守恒:")
    print(f"    max|R + T - 1| = {np.max(np.abs(conservation - 1.0)):.6e}")


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


    samples_simplex = SimplexSampler.sample_unit_simplex(m=5, n=10000)
    mean_sum = np.mean(np.sum(samples_simplex, axis=1))
    print(f"\n  单纯形均匀采样测试:")
    print(f"    Σ x_i 均值 = {mean_sum:.6f}  (理论值: 1.0)")


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
    print("\n" + "#" * 70)
    print("#  系外行星大气光谱反演系统")
    print("#  Exoplanet Atmospheric Spectral Retrieval System")
    print("#" * 70)
    print(f"#  运行时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 70 + "\n")

    np.random.seed(42)
    t_start = time.time()


    data_dict = generate_synthetic_observation()


    x_opt = inversion_least_squares(data_dict)


    samples, log_probs = bayesian_mcmc_analysis(data_dict, x_opt)


    diagnostics = numerical_diagnostics(data_dict, x_opt)


    print("\n" + "=" * 70)
    print("STEP 5: 结果保存")
    print("=" * 70)


    model_best = forward_model(x_opt, data_dict)
    write_spectrum_ascii(data_dict['wavelength_um'], model_best, None, "retrieved_spectrum.dat")


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


    np.savetxt("mcmc_samples.dat", samples, fmt="%.6f")


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
    sys.exit(main())
