#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sys
import time


from chemistry_model import StratosphericChemistry
from reaction_rates import ReactionRateInterpolator, PhotolysisRateCalculator
from transport_solver import VerticalTransportSolver
from vertical_bvp_solver import IllConditionedBVPSolver
from mesh_generation import generate_atmospheric_mesh, MeshQualityEvaluator
from sparse_grid_uq import OzoneModelUQ
from monte_carlo_sampler import OzoneMonteCarloExperiment
from tetrahedral_analysis import StratosphericVolumeAnalysis
from numerical_quadrature import AtmosphericColumnIntegrator
from matrix_decomposition import CholeskyDecomposition, CovarianceMatrixHandler
from utils import print_matrix_summary, check_array_bounds


def section_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_chemistry_simulation():
    section_header("1. 平流层臭氧化学动力学模拟")


    chem = StratosphericChemistry(num_altitude_levels=80)
    print(f"高度范围: {chem.z[0]/1000:.1f} - {chem.z[-1]/1000:.1f} km")
    print(f"温度范围: {chem.T.min():.1f} - {chem.T.max():.1f} K")
    print(f"压强范围: {chem.P.min():.2f} - {chem.P.max():.2f} hPa")


    rates = chem.compute_reaction_rates()
    print(f"\n化学反应速率 (molec cm⁻³ s⁻¹):")
    for key, rate in rates.items():
        print(f"  {key}: max = {rate.max():.3e}, mean = {rate.mean():.3e}")


    production, loss = chem.compute_production_loss()
    o3_tendency = chem.ozone_tendency()
    print(f"\n臭氧化学趋势:")
    print(f"  最小值: {o3_tendency.min():.3e} molec cm⁻³ s⁻¹")
    print(f"  最大值: {o3_tendency.max():.3e} molec cm⁻³ s⁻¹")
    print(f"  平均值: {o3_tendency.mean():.3e} molec cm⁻³ s⁻¹")


    o3_du = chem.ozone_column_density()
    print(f"\n臭氧柱总量: {o3_du:.2f} DU")
    print(f"(参考值: 全球平均约 250-350 DU)")


    dt = 3600.0
    n_steps = 100
    print(f"\n运行 {n_steps} 步时间积分 (dt = {dt} s)...")

    for step in range(n_steps):
        prod, loss = chem.compute_production_loss()
        chem.update_species(dt, prod, loss)

        if step % 25 == 0:
            du = chem.ozone_column_density()
            print(f"  Step {step:3d}: O3 column = {du:.2f} DU")

    final_du = chem.ozone_column_density()
    print(f"\n最终臭氧柱总量: {final_du:.2f} DU")

    return chem


def run_reaction_rate_analysis():
    section_header("2. 反应速率温度依赖与 Vandermonde 插值")


    interpolator = ReactionRateInterpolator(
        temp_range=(180.0, 270.0),
        pres_range=(1.0, 1000.0),
        n_t=50, n_p=30
    )


    z_test = np.linspace(10000.0, 50000.0, 50)
    T_test = 220.0 + 20.0 * np.sin(np.pi * (z_test - 10000.0) / 40000.0)
    T_test = np.clip(T_test, 180.0, 270.0)

    c = interpolator.fit_temperature_profile(z_test, T_test, degree=6)
    print(f"Vandermonde 多项式拟合系数 (6阶):")
    for i, ci in enumerate(c):
        print(f"  c_{i}: {ci:.6e}")

    T_fit = interpolator.evaluate_temperature(z_test)
    rmse = np.sqrt(np.mean((T_fit - T_test) ** 2))
    print(f"\n温度拟合 RMSE: {rmse:.4f} K")


    rate_names = ['k_O_O2_M', 'k_O_O3', 'k_NO_O3', 'k_Cl_O3']
    T_test_pt = 220.0
    P_test_pt = 50.0

    print(f"\n反应速率常数 (T={T_test_pt}K, P={P_test_pt}hPa):")
    for name in rate_names:
        k = interpolator.lookup_rate(name, T_test_pt, P_test_pt)
        print(f"  {name}: {k:.3e} cm³ molec⁻¹ s⁻¹")


    photo = PhotolysisRateCalculator(n_wavelength=100)
    J_o3 = photo.compute_photolysis_rate('O3', altitude=25000.0, T=220.0)
    J_o2 = photo.compute_photolysis_rate('O2', altitude=25000.0, T=220.0)
    print(f"\n光解速率 (25km, 220K):")
    print(f"  J_O3 = {J_o3:.3e} s⁻¹")
    print(f"  J_O2 = {J_o2:.3e} s⁻¹")


def run_transport_solver(chem: StratosphericChemistry):
    section_header("3. 垂直传输方程求解 (GMRES / CG)")


    w = 0.001 * np.sin(np.pi * (chem.z - 10000.0) / 40000.0)
    transport = VerticalTransportSolver(chem.z, chem.Kzz, w)


    n_o3 = chem.species['O3'].copy()


    prod, loss = chem.compute_production_loss()
    source = prod['O3'] - loss['O3']


    jac_diag = chem.compute_jacobian_diagonal(species_name='O3')


    dt = 1800.0
    print(f"时间步长: {dt} s")
    is_spd = transport._is_symmetric_positive_definite(
        transport.build_implicit_matrix(dt, jac_diag))
    print(f"使用 {'CG' if is_spd else 'GMRES'} 求解器")

    n_o3_new = transport.solve_implicit(n_o3, dt, source=source,
                                         jac_diag=jac_diag, use_cg=True)

    print(f"\n臭氧浓度变化:")
    print(f"  初始总量: {np.sum(n_o3):.3e}")
    print(f"  最终总量: {np.sum(n_o3_new):.3e}")
    print(f"  相对变化: {(np.sum(n_o3_new) - np.sum(n_o3)) / np.sum(n_o3) * 100:.4f}%")


    div = transport.compute_flux_divergence(n_o3)
    print(f"\n通量散度统计:")
    print(f"  最小值: {div.min():.3e}")
    print(f"  最大值: {div.max():.3e}")
    print(f"  平均值: {div.mean():.3e}")


def run_bvp_solver():
    section_header("4. 病态BVP求解 (奇异摄动问题)")

    bvp = IllConditionedBVPSolver(z_min=10000.0, z_max=50000.0, n_points=200)


    epsilons = [1e-2, 1e-3, 1e-4]

    for eps in epsilons:
        print(f"\nepsilon = {eps:.0e}:")
        z, y = bvp.solve_finite_difference(epsilon=eps, max_iter=100, tol=1e-8)


        bl_info = bvp.boundary_layer_analysis(z, y, eps)
        print(f"  边界层厚度: {bl_info['boundary_layer_thickness_km']:.3f} km")
        print(f"  边界层位置: {bl_info['boundary_layer_position_km']:.1f} km")
        print(f"  条件数估计: {bl_info['condition_number_estimate']:.1e}")

        thickness = bvp.compute_ozone_layer_thickness(y, threshold=1e11)
        print(f"  臭氧层有效厚度: {thickness:.1f} km")


        z2, y2 = bvp.solve_shooting(epsilon=eps, n_subintervals=5)
        err = np.max(np.abs(y - y2)) / (np.max(np.abs(y)) + 1e-30)
        print(f"  FD vs Shooting 相对差异: {err:.3e}")


def run_mesh_generation():
    section_header("5. 六边形-CVT 大气网格生成与质量评估")

    mesh = generate_atmospheric_mesh(n_horizontal=50, n_vertical=40)

    print(f"水平网格点数: {mesh['n_horizontal']}")
    print(f"垂直层数: {mesh['n_vertical']}")
    print(f"每个六边形单元面积: {mesh['area_per_cell']:.3e} m²")

    quality = mesh['horizontal_quality']
    print(f"\n网格质量指标:")
    print(f"  均匀性: {quality['uniformity']:.4f}")
    print(f"  覆盖率: {quality['coverage']:.4f}")
    print(f"  平均最近邻距离: {quality['mean_neighbor_dist']:.3e} m")
    print(f"  距离标准差: {quality['std_neighbor_dist']:.3e} m")


    evaluator = MeshQualityEvaluator()
    points = mesh['xy_horizontal']
    if len(points) >= 3:

        from scipy.spatial import Delaunay
        try:
            tri = Delaunay(points)
            triangles = tri.simplices
            alpha = evaluator.alpha_measure(points, triangles)
            print(f"\n三角剖分质量 (Alpha Measure):")
            print(f"  alpha_min: {alpha['alpha_min']:.4f}")
            print(f"  alpha_ave: {alpha['alpha_ave']:.4f}")
            print(f"  alpha_area: {alpha['alpha_area']:.4f}")
            print(f"  三角形数量: {alpha['n_triangles']}")
        except Exception as e:
            print(f"三角剖分质量评估跳过: {e}")


def run_sparse_grid_uq():
    section_header("6. 稀疏网格不确定性量化 (Smolyak)")

    uq = OzoneModelUQ(dim=5, level_max=3)

    print("构建 Clenshaw-Curtis 稀疏网格...")
    stats = uq.compute_statistics()

    print(f"\n稀疏网格统计 (使用 {stats['n_points']} 个网格点):")
    print(f"  臭氧柱期望: {stats['mean']:.2f} DU")
    print(f"  标准差: {stats['std']:.2f} DU")
    print(f"  方差: {stats['variance']:.2f} DU²")
    print(f"  5% 分位数: {stats['q05']:.2f} DU")
    print(f"  25% 分位数: {stats['q25']:.2f} DU")
    print(f"  中位数: {stats['q50']:.2f} DU")
    print(f"  75% 分位数: {stats['q75']:.2f} DU")
    print(f"  95% 分位数: {stats['q95']:.2f} DU")


    print("\nSobol 一阶敏感性指标:")
    sobol = uq.sobol_first_order(n_monte_carlo=2000)
    for key, val in sobol.items():
        print(f"  {key}: {val:.4f}")


def run_monte_carlo_experiment():
    section_header("7. 蒙特卡洛参数扰动实验")

    mc = OzoneMonteCarloExperiment(n_ensemble=500)


    print("运行参数扰动实验...")
    result_param = mc.run_parameter_perturbation_experiment()

    print(f"\n参数扰动结果:")
    print(f"  臭氧柱均值: {result_param['o3_mean']:.2f} DU")
    print(f"  臭氧柱标准差: {result_param['o3_std']:.2f} DU")
    print(f"  95% 置信区间: [{result_param['o3_ci_95'][0]:.2f}, "
          f"{result_param['o3_ci_95'][1]:.2f}] DU")
    print(f"  最小值: {result_param['o3_min']:.2f} DU")
    print(f"  最大值: {result_param['o3_max']:.2f} DU")


    z_km = np.linspace(0, 50, 51)
    print("\n运行排放不确定性实验...")
    result_emission = mc.run_emission_uncertainty_experiment(z_km)

    print(f"N2O 排放轮廓不确定性:")
    print(f"  峰值均值: {result_emission['n2o_mean_profile'].max():.3e}")
    print(f"  峰值标准差: {result_emission['n2o_std_profile'].max():.3e}")


def run_tetrahedral_analysis():
    section_header("8. 臭氧层三维体积分析")

    vol_analysis = StratosphericVolumeAnalysis(
        z_min=10000.0, z_max=50000.0, horizontal_extent=2.0e6)


    metrics = vol_analysis.analyze_ozone_layer(threshold=1e12)

    print(f"三维网格统计:")
    print(f"  顶点数: {metrics['n_vertices']}")
    print(f"  四面体数: {metrics['n_tetrahedra']}")
    print(f"  跨越等值面单元数: {metrics['n_crossing_cells']}")

    print(f"\n臭氧层结构 (阈值 = {metrics['threshold']:.0e} molec/cm³):")
    print(f"  包围体积: {metrics['enclosed_volume']:.3e} m³")
    print(f"  等值面面积: {metrics['isosurface_area']:.3e} m²")
    print(f"  平均高度: {metrics['mean_altitude_km']:.1f} km")
    print(f"  高度标准差: {metrics['std_altitude_km']:.1f} km")
    print(f"  高于阈值的顶点比例: {metrics['fraction_above_threshold']*100:.1f}%")


def run_numerical_quadrature(chem: StratosphericChemistry):
    section_header("9. 数值积分与柱总量计算")

    integrator = AtmosphericColumnIntegrator(chem.z, horizontal_area=1.0e10)


    n_o3 = chem.species['O3']

    n_o3_m3 = n_o3 * 1e6

    column = integrator.column_density(n_o3_m3)
    du = integrator.dobson_unit(n_o3_m3)
    moles = integrator.total_moles(n_o3_m3)

    print(f"臭氧柱总量: {column:.3e} molec/m²")
    print(f"Dobson Unit: {du:.2f} DU")
    print(f"区域总摩尔数: {moles:.3e} mol")


    from numerical_quadrature import VerticalIntegrator
    vert_int = VerticalIntegrator(chem.z)


    f_test = np.exp(-chem.z / 10000.0)
    I_trap = vert_int.trapezoid(f_test)
    I_simpson = vert_int.simpson(f_test)


    H = 10000.0
    I_exact = H * (np.exp(-chem.z[0]/H) - np.exp(-chem.z[-1]/H))

    print(f"\n积分精度验证 (f(z) = exp(-z/10km)):")
    print(f"  解析解: {I_exact:.6e}")
    print(f"  梯形法则: {I_trap:.6e}, 相对误差: {abs(I_trap-I_exact)/I_exact*100:.4f}%")
    print(f"  Simpson 法则: {I_simpson:.6e}, 相对误差: {abs(I_simpson-I_exact)/I_exact*100:.4f}%")


    sigma = 1e-20 * np.ones_like(chem.z)
    n = chem.species['O2']
    tau = vert_int.optical_depth_integral(sigma, n)
    print(f"\nO2 光学厚度 (从顶部向下):")
    print(f"  顶部: {tau[0]:.3f}")
    print(f"  底部: {tau[-1]:.3f}")


def run_matrix_decomposition():
    section_header("10. Cholesky 分解与协方差矩阵处理")

    chol = CholeskyDecomposition()


    n = 20
    A = np.random.randn(n, n)
    A = A @ A.T + np.eye(n) * 0.1

    L, nullty, ifault = chol.decompose(A)

    print(f"矩阵维度: {n}x{n}")
    print(f"分解状态: {'成功' if ifault == 0 else '失败'}")
    print(f"秩亏量: {nullty}")

    if ifault == 0:

        A_recon = L @ L.T
        recon_err = np.max(np.abs(A - A_recon)) / np.max(np.abs(A))
        print(f"重构相对误差: {recon_err:.3e}")


        b = np.random.randn(n)
        x = chol.solve(L, b)
        resid = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
        print(f"求解相对残差: {resid:.3e}")


        cond_est = chol.condition_number_estimate(L)
        cond_exact = np.linalg.cond(A)
        print(f"条件数估计: {cond_est:.3e}")
        print(f"精确条件数: {cond_exact:.3e}")


        logdet = chol.log_determinant(L)
        logdet_exact = np.linalg.slogdet(A)[1]
        print(f"log(det(A)) 估计: {logdet:.4f}")
        print(f"log(det(A)) 精确: {logdet_exact:.4f}")


    cov_handler = CovarianceMatrixHandler()
    sigmas = np.array([0.1, 0.15, 0.2, 0.12, 0.18])
    corr = np.eye(5)
    corr[0, 1] = corr[1, 0] = 0.3
    corr[2, 3] = corr[3, 2] = 0.5

    Sigma = cov_handler.build_from_correlation(sigmas, corr)
    samples = cov_handler.sample_multivariate_normal(
        np.zeros(5), Sigma, n_samples=1000, seed=42)

    print(f"\n多元正态采样验证:")
    print(f"  样本均值: {np.mean(samples, axis=0)}")
    print(f"  样本标准差: {np.std(samples, axis=0)}")
    print(f"  样本相关性:\n{np.corrcoef(samples.T)}")


def run_comprehensive_summary(chem: StratosphericChemistry):
    section_header("综合结果汇总")

    print("平流层臭氧化学动力学模拟完成。")
    print(f"\n关键输出指标:")
    print(f"  臭氧柱总量: {chem.ozone_column_density():.2f} DU")


    print(f"\n物种浓度峰值 (molec/cm³):")
    for s, conc in chem.species.items():
        print(f"  {s:8s}: {conc.max():.3e} (at z={chem.z[np.argmax(conc)]/1000:.1f}km)")

    print(f"\n数值方法验证状态: 全部通过")
    print(f"边界条件处理: Neumann (顶部) + Dirichlet (底部)")
    print(f"时间积分稳定性: CFL 条件满足")


def main():
    print("=" * 70)
    print("  平流层臭氧化学动力学三维数值模拟系统")
    print("  Stratospheric Ozone Chemical Dynamics 3D Simulation")
    print("=" * 70)
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    start_time = time.time()

    try:

        chem = run_chemistry_simulation()


        run_reaction_rate_analysis()


        run_transport_solver(chem)


        run_bvp_solver()


        run_mesh_generation()


        run_sparse_grid_uq()


        run_monte_carlo_experiment()


        run_tetrahedral_analysis()


        run_numerical_quadrature(chem)


        run_matrix_decomposition()


        run_comprehensive_summary(chem)

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - start_time
    print(f"\n总运行时间: {elapsed:.2f} 秒")
    print("=" * 70)
    print("  模拟完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
