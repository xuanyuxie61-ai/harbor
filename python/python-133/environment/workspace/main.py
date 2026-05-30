#!/usr/bin/env python3

import sys
import numpy as np
import time




from polymerization_kinetics import (
    PolymerizationParameters,
    integrate_polymerization,
    compute_conversion_and_pdi,
    exact_solution_batch,
)
from reaction_diffusion_dg import DG1DReactionDiffusion
from molecular_weight_distribution import (
    flory_schulz_distribution,
    flory_schulz_moments,
    lognormal_mwd_pdf,
    truncated_normal_sample,
    polygon_moment,
    mwd_from_moments,
    compute_pdi_from_moments,
    local_mwd_broadening,
)
from monte_carlo_chain_sampler import (
    ellipse_sample,
    coarse_grained_chain_mc,
    mixing_efficiency_estimate,
    critical_pore_size,
    radius_of_gyration,
)
from uncertainty_quantification import (
    propagate_uncertainty,
    sensitivity_index_sobol,
)
from vandermonde_reconstruction import (
    reconstruct_mwd_curve,
    derivative_mwd_curve,
    monomial_moments_from_coeffs,
)
from nonlinear_solver import (
    newton_krylov_solve,
    gel_effect_diffusion,
    nonlinear_source_reaction,
    nonlinear_source_derivative,
    detq_orthogonal,
)
from cvt_reactor_discretization import optimal_reactor_nodes
from quadrature_validation import validate_quadrature_rule, convergence_order_estimate


def print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_result(label: str, value, unit: str = "") -> None:
    unit_str = f" [{unit}]" if unit else ""
    print(f"  {label:<40s}: {value:.6e}{unit_str}")


def main() -> int:
    np.set_printoptions(precision=4, suppress=True)
    start_time = time.time()

    print("\n" + "#" * 70)
    print("#  化学工程：聚合反应动力学与分子量分布多尺度耦合模拟")
    print("#  Chemical Engineering: Polymerization Kinetics & MWD")
    print("#" * 70)




    print_header("模块 1: 自由基聚合反应动力学矩方程")

    params = PolymerizationParameters(
        kd=2.0e-4,
        ki=5.0e2,
        kp=2.5e3,
        ktc=5.0e6,
        ktd=5.0e6,
        ktr=5.0e-2,
        f=0.55,
        M0=8.5,
        I0=5.0e-3,
        S0=0.0,
        T=343.15,
        tstop=3600.0,
    )




    t_vec, y_mat = None, None
    results = {}


    idx_final = -1
    print_result("单体转化率 X", results['conversion'][idx_final], "")
    print_result("数均聚合度 DP_n", results['DP_n'][idx_final], "")
    print_result("重均聚合度 DP_w", results['DP_w'][idx_final], "")
    print_result("多分散指数 PDI", results['PDI'][idx_final], "")
    print_result("数均分子量 M_n", results['Mn'][idx_final], "g/mol")
    print_result("重均分子量 M_w", results['Mw'][idx_final], "g/mol")


    M_approx = exact_solution_batch(np.array([params.tstop]), params)
    print_result("解析近似 M(t_stop)", M_approx[0], "mol/L")




    print_header("模块 2: 一维 DG 反应-扩散-对流方程")

    dg_solver = DG1DReactionDiffusion(
        N=1, K=10, x_left=0.0, x_right=1.0,
        v=0.0, D_diff=1.0e-3
    )


    u0 = params.M0 * np.sin(np.pi * dg_solver.x)

    def source_func(x_arr, t):

        return 0.1 * np.ones_like(x_arr)

    x_dg, u_dg = dg_solver.solve(u0, final_time=10.0, source_func=source_func)
    print(f"  DG 求解器: {dg_solver.K} 单元, 阶数 {dg_solver.N}")
    print_result("入口浓度", float(u_dg[0, 0]), "mol/L")
    print_result("出口浓度", float(u_dg[0, -1]), "mol/L")
    print_result("最大浓度", float(np.max(u_dg)), "mol/L")
    print_result("最小浓度", float(np.min(u_dg)), "mol/L")




    print_header("模块 3: 分子量分布与截断对数正态模型")

    n_grid = np.arange(1, 2001, dtype=float)
    p_growth = 0.995




    pmf_fs = None
    moments_fs = None
    DP_n_fs, DP_w_fs, PDI_fs = None, None, None

    print_result("Flory-Schulz DP_n", DP_n_fs, "")
    print_result("Flory-Schulz DP_w", DP_w_fs, "")
    print_result("Flory-Schulz PDI", PDI_fs, "")


    mw_grid = np.logspace(1.0, 6.0, 500)
    pdf_ln = lognormal_mwd_pdf(mw_grid, mu_log=8.0, sigma_log=1.5, a=100.0, b=1.0e6)
    print_result("截断对数正态积分", float(np.trapezoid(pdf_ln, mw_grid)), "")



    x_poly = np.array([0.0, 1.0, 1.0, 0.0])
    y_poly = np.array([0.0, 0.0, 1.0, 1.0])
    nu_00 = polygon_moment(4, x_poly, y_poly, 0, 0)
    nu_11 = polygon_moment(4, x_poly, y_poly, 1, 1)
    print_result("多边形零阶矩 ν_00", nu_00, "m²")
    print_result("多边形一阶矩 ν_11", nu_11, "m⁴")


    moments_broad = local_mwd_broadening(moments_fs, velocity_gradient=5.0,
                                         diffusion_coeff=1.0e-5, reaction_rate=0.1)
    print_result("展宽修正后 μ_1", moments_broad[1], "")




    print_header("模块 4: 聚合物链构象蒙特卡洛与混合效率")


    A_ellipsoid = np.array([[4.0, 1.0], [1.0, 2.0]])
    ellipsoid_samples = ellipse_sample(500, A_ellipsoid, r=1.0)
    print_result("椭球采样均值 x", float(np.mean(ellipsoid_samples[0, :])), "")
    print_result("椭球采样均值 y", float(np.mean(ellipsoid_samples[1, :])), "")


    chain_samples = coarse_grained_chain_mc(n_segments=100, n_samples=200,
                                            kuhn_length=0.5)
    rg_mean = np.mean(np.linalg.norm(chain_samples, axis=1))
    print_result("平均末端距", rg_mean, "nm")


    avg_area, efficiency, theoretical = mixing_efficiency_estimate(n_trials=5000)
    print_result("圆盘随机三角形平均面积", avg_area, "")
    print_result("混合效率 η_mix", efficiency, "")
    print_result("理论值", theoretical, "")


    dc = critical_pore_size(chain_samples, porosity=0.4)
    print_result("临界孔隙尺寸", dc, "nm")




    print_header("模块 5: 稀疏网格 Gauss-Hermite 不确定性量化")

    def polymer_model(xi):

        kp_eff = 2500.0 * np.exp(0.1 * xi[0])
        kt_eff = 1.0e7 * np.exp(0.15 * xi[1])
        f_eff = 0.55 + 0.05 * xi[2]
        f_eff = np.clip(f_eff, 0.1, 1.0)

        lam0_ss = np.sqrt(2.0 * f_eff * params.kd * params.I0 / kt_eff)
        p = kp_eff * params.M0 / (kp_eff * params.M0 + kt_eff * lam0_ss + 1.0e-12)
        p = min(p, 0.9999)
        return (1.0 + p) / (1.0 - p)

    uq_results = propagate_uncertainty(
        polymer_model, dim_num=3, level_max=3,
        param_means=np.zeros(3),
        param_stds=np.array([1.0, 1.0, 1.0])
    )

    print_result("PDI 均值", uq_results['mean'], "")
    print_result("PDI 方差", uq_results['variance'], "")
    print_result("PDI 标准差", uq_results['std'], "")
    print_result("PDI 偏度", uq_results['skewness'], "")
    print_result("PDI 峰度", uq_results['kurtosis'], "")


    sobol_idx = sensitivity_index_sobol(
        uq_results['values'], uq_results['weights'],
        uq_results['points'], dim_num=3
    )
    print(f"  Sobol 敏感度 S1 (kp, kt, f)   : {sobol_idx}")




    print_header("模块 6: 分子量分布 Vandermonde 插值重构")


    mw_data = np.array([1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6])
    wf_data = lognormal_mwd_pdf(mw_data, mu_log=8.0, sigma_log=1.2,
                                a=50.0, b=2.0e6)
    M_interp, w_interp, coeffs = reconstruct_mwd_curve(
        mw_data, wf_data, n_interp=300, log_scale=True
    )
    print_result("重构分布峰值", float(np.max(w_interp)), "")
    print_result("重构分布峰值位置", float(M_interp[np.argmax(w_interp)]), "g/mol")


    moments_recon = monomial_moments_from_coeffs(coeffs, scale=1.0, max_moment=2)
    print_result("重构 μ_0", moments_recon[0], "")
    print_result("重构 μ_1", moments_recon[1], "g/mol")




    print_header("模块 7: 凝胶效应非线性反应-扩散 Newton-Krylov 求解")

    n2d = 21
    u0_2d = np.zeros(n2d * n2d)

    for i in range(n2d):
        for j in range(n2d):
            k = i * n2d + j
            x = -1.0 + 2.0 * i / (n2d - 1)
            y = -1.0 + 2.0 * j / (n2d - 1)
            u0_2d[k] = 0.1 * np.exp(-(x ** 2 + y ** 2) / 0.5)

    source_2d = np.ones(n2d * n2d) * 0.5

    u_sol, n_iter, final_res = newton_krylov_solve(
        n=n2d,
        u0=u0_2d,
        diff_func=lambda u: gel_effect_diffusion(u, D0=1.0e-3, beta=2.5),
        reaction_func=lambda u: nonlinear_source_reaction(u, k0=2.0, activation=8.0),
        reaction_deriv=lambda u: nonlinear_source_derivative(u, k0=2.0, activation=8.0),
        source=source_2d,
        xleft=-1.0, xright=1.0,
        tol=1.0e-6, max_iter=20
    )

    print_result("Newton 迭代次数", float(n_iter), "")
    print_result("最终残差范数", final_res, "")
    print_result("最大转化率", float(np.max(u_sol)), "")
    print_result("平均转化率", float(np.mean(u_sol)), "")


    test_mat = np.random.randn(5, 5)
    q, _ = np.linalg.qr(test_mat)
    det_val, ifault = detq_orthogonal(q)
    print_result("QR 正交行列式", det_val, "")
    print(f"  正交性检验 ifault                : {ifault}")




    print_header("模块 8: CVT 反应器最优空间离散化")

    g, energy, motion = optimal_reactor_nodes(n_nodes=16, n_iter=20, n_samples=60)
    print_result("最终 Lloyd 能量", energy[-1], "")
    print_result("最终平均移动", motion[-1], "")
    print(f"  生成元数                         : {g.shape[0]}")
    print(f"  生成元质心范围 x                 : [{g[:,0].min():.4f}, {g[:,0].max():.4f}]")
    print(f"  生成元质心范围 y                 : [{g[:,1].min():.4f}, {g[:,1].max():.4f}]")




    print_header("模块 9: 稀疏网格求积规则精确性验证")


    from numpy.polynomial.hermite import hermgauss
    x_gh, w_gh = hermgauss(5)
    grid_pt = x_gh.reshape(1, -1)
    grid_wt = w_gh

    validation = validate_quadrature_rule(grid_pt, grid_wt, dim_num=1, degree_max=9)
    print(f"  总测试单项式数                   : {validation['total_tests']}")
    print_result("最大误差", validation['max_error'], "")
    for deg, info in validation['errors_by_degree'].items():
        if info['max_error'] > 1.0e-12:
            print(f"  阶数 {deg:2d} 最大误差                : {info['max_error']:.6e}")


    errors_demo = [1.0e-2, 1.0e-4, 1.0e-6]
    npts_demo = [10, 50, 250]
    p_est = convergence_order_estimate(errors_demo, npts_demo)
    print_result("演示收敛阶估计", p_est, "")




    elapsed = time.time() - start_time
    print("\n" + "#" * 70)
    print(f"#  模拟完成，总耗时: {elapsed:.3f} s")
    print("#" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
