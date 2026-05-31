#!/usr/bin/env python3
"""
main.py
========
聚合反应动力学与分子量分布多尺度耦合模拟系统

统一入口，零参数运行。
本程序执行以下完整流程：
  1. 自由基聚合反应动力学 ODE 求解（矩方法）
  2. 一维 DG 反应-扩散方程求解（反应器轴向温度/浓度分布）
  3. 分子量分布计算与截断对数正态建模
  4. 聚合物链构象蒙特卡洛采样与混合效率估计
  5. 参数不确定性量化（稀疏网格 Gauss-Hermite）
  6. 分子量分布 Vandermonde 插值重构
  7. 二维非线性反应-扩散 Newton-Krylov 求解（凝胶效应）
  8. CVT 反应器最优空间离散化
  9. 求积规则精确性验证
"""

import sys
import numpy as np
import time

# ---------------------------------------------------------------------------
# 导入各模块
# ---------------------------------------------------------------------------
from polymerization_kinetics import (
    PolymerizationParameters,
    integrate_polymerization,
    compute_conversion_and_pdi,
    exact_solution_batch,
    polymerization_deriv,
    polymerization_initial_state,
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
    disk_triangle_picking,
)
from uncertainty_quantification import (
    propagate_uncertainty,
    sensitivity_index_sobol,
    sparse_grid_hermite,
)
from vandermonde_reconstruction import (
    reconstruct_mwd_curve,
    derivative_mwd_curve,
    monomial_moments_from_coeffs,
    vandermonde_matrix_1d,
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

    # =====================================================================
    # 模块 1: 聚合反应动力学 ODE 求解 (种子 090, 1037)
    # =====================================================================
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

    t_vec, y_mat = integrate_polymerization(params, n_steps=2000)
    results = compute_conversion_and_pdi(t_vec, y_mat, params)

    # 输出关键结果（取最终时刻）
    idx_final = -1
    print_result("单体转化率 X", results['conversion'][idx_final], "")
    print_result("数均聚合度 DP_n", results['DP_n'][idx_final], "")
    print_result("重均聚合度 DP_w", results['DP_w'][idx_final], "")
    print_result("多分散指数 PDI", results['PDI'][idx_final], "")
    print_result("数均分子量 M_n", results['Mn'][idx_final], "g/mol")
    print_result("重均分子量 M_w", results['Mw'][idx_final], "g/mol")

    # 解析解对比
    M_approx = exact_solution_batch(np.array([params.tstop]), params)
    print_result("解析近似 M(t_stop)", M_approx[0], "mol/L")

    # =====================================================================
    # 模块 2: 一维 DG 反应-扩散求解器 (种子 273, 359)
    # =====================================================================
    print_header("模块 2: 一维 DG 反应-扩散-对流方程")

    dg_solver = DG1DReactionDiffusion(
        N=1, K=10, x_left=0.0, x_right=1.0,
        v=0.0, D_diff=1.0e-3
    )

    # 初始浓度分布：正弦型，满足边界 u=0
    u0 = params.M0 * np.sin(np.pi * dg_solver.x)

    def source_func(x_arr, t):
        # 温和的反应源项
        return 0.1 * np.ones_like(x_arr)

    x_dg, u_dg = dg_solver.solve(u0, final_time=10.0, source_func=source_func)
    print(f"  DG 求解器: {dg_solver.K} 单元, 阶数 {dg_solver.N}")
    print_result("入口浓度", float(u_dg[0, 0]), "mol/L")
    print_result("出口浓度", float(u_dg[0, -1]), "mol/L")
    print_result("最大浓度", float(np.max(u_dg)), "mol/L")
    print_result("最小浓度", float(np.min(u_dg)), "mol/L")

    # =====================================================================
    # 模块 3: 分子量分布建模 (种子 1360, 886)
    # =====================================================================
    print_header("模块 3: 分子量分布与截断对数正态模型")

    n_grid = np.arange(1, 2001, dtype=float)
    p_growth = 0.995
    pmf_fs = flory_schulz_distribution(n_grid, p_growth)
    moments_fs = flory_schulz_moments(p_growth, max_moment=4)
    DP_n_fs, DP_w_fs, PDI_fs = compute_pdi_from_moments(moments_fs)

    print_result("Flory-Schulz DP_n", DP_n_fs, "")
    print_result("Flory-Schulz DP_w", DP_w_fs, "")
    print_result("Flory-Schulz PDI", PDI_fs, "")

    # 截断对数正态分布
    mw_grid = np.logspace(1.0, 6.0, 500)
    pdf_ln = lognormal_mwd_pdf(mw_grid, mu_log=8.0, sigma_log=1.5, a=100.0, b=1.0e6)
    print_result("截断对数正态积分", float(np.trapezoid(pdf_ln, mw_grid)), "")

    # 多边形矩（模拟反应器截面）
    # 单位正方形
    x_poly = np.array([0.0, 1.0, 1.0, 0.0])
    y_poly = np.array([0.0, 0.0, 1.0, 1.0])
    nu_00 = polygon_moment(4, x_poly, y_poly, 0, 0)
    nu_11 = polygon_moment(4, x_poly, y_poly, 1, 1)
    print_result("多边形零阶矩 ν_00", nu_00, "m²")
    print_result("多边形一阶矩 ν_11", nu_11, "m⁴")

    # 流场展宽修正
    moments_broad = local_mwd_broadening(moments_fs, velocity_gradient=5.0,
                                         diffusion_coeff=1.0e-5, reaction_rate=0.1)
    print_result("展宽修正后 μ_1", moments_broad[1], "")

    # =====================================================================
    # 模块 4: 蒙特卡洛链采样与混合效率 (种子 331, 298)
    # =====================================================================
    print_header("模块 4: 聚合物链构象蒙特卡洛与混合效率")

    # 椭球采样模拟受限链构象空间
    A_ellipsoid = np.array([[4.0, 1.0], [1.0, 2.0]])
    ellipsoid_samples = ellipse_sample(500, A_ellipsoid, r=1.0)
    print_result("椭球采样均值 x", float(np.mean(ellipsoid_samples[0, :])), "")
    print_result("椭球采样均值 y", float(np.mean(ellipsoid_samples[1, :])), "")

    # 粗粒化链采样
    chain_samples = coarse_grained_chain_mc(n_segments=100, n_samples=200,
                                            kuhn_length=0.5)
    rg_mean = np.mean(np.linalg.norm(chain_samples, axis=1))
    print_result("平均末端距", rg_mean, "nm")

    # 混合效率
    avg_area, efficiency, theoretical = mixing_efficiency_estimate(n_trials=5000)
    print_result("圆盘随机三角形平均面积", avg_area, "")
    print_result("混合效率 η_mix", efficiency, "")
    print_result("理论值", theoretical, "")

    # 临界孔径
    dc = critical_pore_size(chain_samples, porosity=0.4)
    print_result("临界孔隙尺寸", dc, "nm")

    # =====================================================================
    # 模块 5: 不确定性量化 (种子 1105)
    # =====================================================================
    print_header("模块 5: 稀疏网格 Gauss-Hermite 不确定性量化")

    def polymer_model(xi):
        # 3 维参数不确定性：log(kp), log(kt), f
        kp_eff = 2500.0 * np.exp(0.1 * xi[0])
        kt_eff = 1.0e7 * np.exp(0.15 * xi[1])
        f_eff = 0.55 + 0.05 * xi[2]
        f_eff = np.clip(f_eff, 0.1, 1.0)
        # 简化 PDI 估计：PDI ≈ 1 + p, p ≈ kp[M] / (kp[M] + kt*λ0)
        lam0_ss = np.sqrt(2.0 * f_eff * params.kd * params.I0 / kt_eff)
        p = kp_eff * params.M0 / (kp_eff * params.M0 + kt_eff * lam0_ss + 1.0e-12)
        p = min(p, 0.9999)
        return (1.0 + p) / (1.0 - p)  # DP_w approximation

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

    # Sobol 敏感度
    sobol_idx = sensitivity_index_sobol(
        uq_results['values'], uq_results['weights'],
        uq_results['points'], dim_num=3
    )
    print(f"  Sobol 敏感度 S1 (kp, kt, f)   : {sobol_idx}")

    # =====================================================================
    # 模块 6: Vandermonde 分布重构 (种子 1384)
    # =====================================================================
    print_header("模块 6: 分子量分布 Vandermonde 插值重构")

    # 构造离散 MWD 数据点
    mw_data = np.array([1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6])
    wf_data = lognormal_mwd_pdf(mw_data, mu_log=8.0, sigma_log=1.2,
                                a=50.0, b=2.0e6)
    M_interp, w_interp, coeffs = reconstruct_mwd_curve(
        mw_data, wf_data, n_interp=300, log_scale=True
    )
    print_result("重构分布峰值", float(np.max(w_interp)), "")
    print_result("重构分布峰值位置", float(M_interp[np.argmax(w_interp)]), "g/mol")

    # 由系数计算矩
    moments_recon = monomial_moments_from_coeffs(coeffs, scale=1.0, max_moment=2)
    print_result("重构 μ_0", moments_recon[0], "")
    print_result("重构 μ_1", moments_recon[1], "g/mol")

    # =====================================================================
    # 模块 7: 凝胶效应非线性反应-扩散求解 (种子 871, 034)
    # =====================================================================
    print_header("模块 7: 凝胶效应非线性反应-扩散 Newton-Krylov 求解")

    n2d = 21
    u0_2d = np.zeros(n2d * n2d)
    # 初始猜测：中心高转化率
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

    # 正交性检验
    test_mat = np.random.randn(5, 5)
    q, _ = np.linalg.qr(test_mat)
    det_val, ifault = detq_orthogonal(q)
    print_result("QR 正交行列式", det_val, "")
    print(f"  正交性检验 ifault                : {ifault}")

    # =====================================================================
    # 模块 8: CVT 反应器离散化 (种子 247)
    # =====================================================================
    print_header("模块 8: CVT 反应器最优空间离散化")

    g, energy, motion = optimal_reactor_nodes(n_nodes=16, n_iter=20, n_samples=60)
    print_result("最终 Lloyd 能量", energy[-1], "")
    print_result("最终平均移动", motion[-1], "")
    print(f"  生成元数                         : {g.shape[0]}")
    print(f"  生成元质心范围 x                 : [{g[:,0].min():.4f}, {g[:,0].max():.4f}]")
    print(f"  生成元质心范围 y                 : [{g[:,1].min():.4f}, {g[:,1].max():.4f}]")

    # =====================================================================
    # 模块 9: 求积规则验证 (种子 930, 776)
    # =====================================================================
    print_header("模块 9: 稀疏网格求积规则精确性验证")

    # 构造一维 5 点 Gauss-Hermite 规则进行验证
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

    # 收敛阶估计（模拟不同阶数）
    errors_demo = [1.0e-2, 1.0e-4, 1.0e-6]
    npts_demo = [10, 50, 250]
    p_est = convergence_order_estimate(errors_demo, npts_demo)
    print_result("演示收敛阶估计", p_est, "")

    # =====================================================================
    # 总时间统计
    # =====================================================================
    elapsed = time.time() - start_time
    print("\n" + "#" * 70)
    print(f"#  模拟完成，总耗时: {elapsed:.3f} s")
    print("#" * 70)

    return 0


# ================================================================
# 测试用例（30个，assert模式，涉及随机值均使用固定种子）
# ================================================================

# ---- TC01: PolymerizationParameters 验证规则通过 ----
params_test = PolymerizationParameters(
    kd=1.0e-4, ki=5.0e2, kp=2.5e3, ktc=5.0e6, ktd=5.0e6, ktr=1.0e-1,
    f=0.6, M0=8.0, I0=1.0e-2, S0=0.0, T=333.15, tstop=100.0
)
assert params_test.kd > 0.0, '[TC01] kd 必须为正 FAILED'
assert params_test.f > 0.0 and params_test.f <= 1.0, '[TC01] f 范围错误 FAILED'

# ---- TC02: 有效速率常数温度修正合理 ----
keff = params_test.effective_rate_constants()
assert keff['kp'] > 0.0, '[TC02] 有效 kp 非正 FAILED'
assert keff['ktc'] > 0.0, '[TC02] 有效 ktc 非正 FAILED'

# ---- TC03: 初始状态向量维度正确 ----
y0 = polymerization_initial_state(params_test)
assert y0.shape == (9,), '[TC03] 初始状态向量维度应为9 FAILED'
assert y0[0] == params_test.M0, '[TC03] y0[0] 应为 M0 FAILED'
assert y0[1] == params_test.I0, '[TC03] y0[1] 应为 I0 FAILED'

# ---- TC04: 导数函数返回有限值 ----
dydt = polymerization_deriv(0.0, y0, params_test)
assert dydt.shape == (9,), '[TC04] 导数向量维度应为9 FAILED'
assert np.all(np.isfinite(dydt)), '[TC04] 导数包含非有限值 FAILED'

# ---- TC05: integrate_polymerization 输出形状正确 ----
t_vec, y_mat = integrate_polymerization(params_test, n_steps=200)
assert t_vec.shape[0] == y_mat.shape[0], '[TC05] t 和 y 行数不一致 FAILED'
assert y_mat.shape[1] == 9, '[TC05] y_mat 列数应为9 FAILED'

# ---- TC06: 单体浓度随时间单调不增 ----
M_vals = y_mat[:, 0]
assert np.all(M_vals[1:] <= M_vals[:-1] + 1.0e-10), '[TC06] 单体浓度应单调不增 FAILED'

# ---- TC07: 转化率在 [0,1] 范围 ----
results_poly = compute_conversion_and_pdi(t_vec, y_mat, params_test)
conv = results_poly['conversion']
assert np.all(conv >= 0.0) and np.all(conv <= 1.0), '[TC07] 转化率超出 [0,1] FAILED'

# ---- TC08: PDI >= 1.0 ----
pdi_vals = results_poly['PDI']
valid = pdi_vals[pdi_vals > 1.0e-10]
assert np.all(valid >= 1.0 - 1.0e-8), '[TC08] PDI 应 >= 1.0 FAILED'

# ---- TC09: exact_solution_batch 返回正确形状 ----
t_test = np.array([10.0, 100.0, 500.0])
M_approx = exact_solution_batch(t_test, params_test)
assert M_approx.shape == t_test.shape, '[TC09] 解析近似输出形状错误 FAILED'
assert np.all(M_approx > 0.0), '[TC09] 单体浓度近似应为正 FAILED'

# ---- TC10: Flory-Schulz 分布总和接近1 ----
import numpy as np
n_grid = np.arange(1, 1001, dtype=float)
p_test = 0.995
pmf_fs = flory_schulz_distribution(n_grid, p_test)
pmf_sum = np.sum(pmf_fs)
assert pmf_sum > 0.9 and pmf_sum < 1.1, '[TC10] Flory-Schulz 分布之和应接近1 FAILED'

# ---- TC11: Flory-Schulz 矩解析一致性 ----
moments_fs_test = flory_schulz_moments(p_test, max_moment=4)
DP_n_fs, DP_w_fs, PDI_fs = compute_pdi_from_moments(moments_fs_test)
assert abs(DP_n_fs - 1.0/(1.0-p_test)) < 1.0e-4, '[TC11] DP_n 与解析值不一致 FAILED'
assert abs(PDI_fs - (1.0+p_test)) < 1.0e-4, '[TC11] PDI 与解析值不一致 FAILED'

# ---- TC12: 对数正态 PDF 非负 ----
mw_grid = np.logspace(1.0, 5.0, 200)
pdf_ln = lognormal_mwd_pdf(mw_grid, mu_log=8.0, sigma_log=1.5, a=10.0, b=2.0e5)
assert np.all(pdf_ln >= 0.0), '[TC12] PDF 应为非负 FAILED'

# ---- TC13: 截断正态采样在边界内 ----
np.random.seed(42)
samples_tn = truncated_normal_sample(mu=0.0, sigma=1.0, a=-2.0, b=2.0, size=200)
assert np.all(samples_tn >= -2.0) and np.all(samples_tn <= 2.0), '[TC13] 截断采样超出边界 FAILED'

# ---- TC14: 多边形矩面积验证 (单位正方形) ----
x_poly = np.array([0.0, 1.0, 1.0, 0.0])
y_poly = np.array([0.0, 0.0, 1.0, 1.0])
nu_00 = polygon_moment(4, x_poly, y_poly, 0, 0)
assert abs(nu_00 - 1.0) < 1.0e-10, '[TC14] 正方形面积应为1 FAILED'

# ---- TC15: 椭球采样点均有限且维度正确 ----
A_ell = np.array([[4.0, 1.0], [1.0, 2.0]])
np.random.seed(42)
ell_samples = ellipse_sample(100, A_ell, r=1.0)
assert ell_samples.shape == (2, 100), '[TC15] 椭球采样输出形状错误 FAILED'
assert np.all(np.isfinite(ell_samples)), '[TC15] 椭球采样包含非有限值 FAILED'

# ---- TC16: 粗粒化链采样可复现 ----
np.random.seed(2024)
chain1 = coarse_grained_chain_mc(n_segments=50, n_samples=30, kuhn_length=0.5)
np.random.seed(2024)
chain2 = coarse_grained_chain_mc(n_segments=50, n_samples=30, kuhn_length=0.5)
assert np.allclose(chain1, chain2), '[TC16] 固定种子结果应一致 FAILED'

# ---- TC17: 混合效率估计输出合理 ----
np.random.seed(42)
avg_area, efficiency, theoretical = mixing_efficiency_estimate(n_trials=2000)
assert avg_area > 0.0, '[TC17] 平均面积应为正 FAILED'
assert efficiency > 0.0 and efficiency < 1.0, '[TC17] 混合效率应在 (0,1) FAILED'

# ---- TC18: 临界孔隙尺寸为正 ----
chain_test = np.random.randn(30, 3)
dc = critical_pore_size(chain_test, porosity=0.4)
assert dc > 0.0, '[TC18] 临界孔隙尺寸应为正 FAILED'

# ---- TC19: 稀疏网格节点非空 ----
points_uq, weights_uq = sparse_grid_hermite(dim_num=2, level_max=2)
assert points_uq.shape[1] > 0, '[TC19] 稀疏网格节点数应为正 FAILED'
assert weights_uq.shape[0] == points_uq.shape[1], '[TC19] 权重维度应与节点数一致 FAILED'

# ---- TC20: 不确定性传播返回完整统计量 ----
test_model_tc20 = lambda xi: float(xi[0]**2 + xi[1]**2)
uq_res = propagate_uncertainty(test_model_tc20, dim_num=2, level_max=2,
                               param_means=np.zeros(2), param_stds=np.ones(2))
assert 'mean' in uq_res and 'variance' in uq_res, '[TC20] UQ 缺少统计量 FAILED'
assert uq_res['variance'] >= 0.0, '[TC20] 方差应为非负 FAILED'

# ---- TC21: Vandermonde 矩阵列数正确 ----
x_v = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
V_test, sc = vandermonde_matrix_1d(x_v, 3)
assert V_test.shape == (5, 3), '[TC21] Vandermonde 形状应为 (5,3) FAILED'
assert np.allclose(V_test[:, 0], 1.0), '[TC21] 第一列应为1 FAILED'

# ---- TC22: MWD 重构输出尺寸正确 ----
mw_data = np.array([1e2, 1e3, 1e4, 1e5, 1e6])
wf_data = lognormal_mwd_pdf(mw_data, mu_log=8.0, sigma_log=1.2, a=10.0, b=2.0e6)
M_int, w_int, coeffs = reconstruct_mwd_curve(mw_data, wf_data, n_interp=100, log_scale=True)
assert M_int.shape[0] == 100, '[TC22] 插值输出尺寸错误 FAILED'
assert w_int.shape[0] == 100, '[TC22] 权重输出尺寸错误 FAILED'
assert np.all(w_int >= 0.0), '[TC22] 插值权重应为非负 FAILED'

# ---- TC23: 凝胶效应扩散系数单调递减 ----
c_test = np.linspace(0.0, 0.99, 50)
D_test = gel_effect_diffusion(c_test, D0=1.0e-3, beta=2.5)
assert np.all(D_test[1:] <= D_test[:-1] + 1.0e-15), '[TC23] 扩散系数应单调递减 FAILED'
assert np.all(D_test > 0.0), '[TC23] 扩散系数应为正 FAILED'

# ---- TC24: 非线性源项在 [0,1] 端点为零 ----
c_end = np.array([0.0, 1.0])
R_end = nonlinear_source_reaction(c_end, k0=2.0, activation=8.0)
assert abs(R_end[0]) < 1.0e-14, '[TC24] 源项在 c=0 应为0 FAILED'
assert abs(R_end[1]) < 1.0e-14, '[TC24] 源项在 c=1 应为0 FAILED'

# ---- TC25: QR 正交矩阵行列式绝对值接近1 ----
test_mat = np.random.randn(5, 5)
q_test, _ = np.linalg.qr(test_mat)
det_val, ifault = detq_orthogonal(q_test)
assert abs(abs(det_val) - 1.0) < 1.0e-3, '[TC25] 正交矩阵行列式绝对值应接近1 FAILED'

# ---- TC26: Newton-Krylov 求解收敛 ----
n_small = 5
u0_test = np.ones(n_small * n_small) * 0.01
src_test = np.ones(n_small * n_small) * 0.5
u_sol, n_iter, final_res = newton_krylov_solve(
    n=n_small, u0=u0_test,
    diff_func=lambda u: gel_effect_diffusion(u, D0=1.0e-3, beta=2.5),
    reaction_func=lambda u: nonlinear_source_reaction(u, k0=2.0, activation=8.0),
    reaction_deriv=lambda u: nonlinear_source_derivative(u, k0=2.0, activation=8.0),
    source=src_test, xleft=-1.0, xright=1.0, tol=1.0e-6, max_iter=30
)
assert final_res < 1.0e-3, '[TC26] Newton 求解应收敛 FAILED'
assert n_iter <= 30, '[TC26] 迭代次数超出最大限制 FAILED'

# ---- TC27: 最佳反应器节点收敛 ----
g, energy, motion = optimal_reactor_nodes(n_nodes=8, n_iter=15, n_samples=40)
assert g.shape[0] == 8, '[TC27] 生成元数量应为8 FAILED'
assert energy[-1] >= 0.0, '[TC27] 能量应为非负 FAILED'
assert motion[-1] < motion[0] or motion[-1] <= 1.0e-10, '[TC27] 移动应趋于收敛 FAILED'

# ---- TC28: 求积规则验证覆盖全部单项式 ----
from numpy.polynomial.hermite import hermgauss
x_gh, w_gh = hermgauss(3)
grid_pt = x_gh.reshape(1, -1)
grid_wt = w_gh
validation = validate_quadrature_rule(grid_pt, grid_wt, dim_num=1, degree_max=4)
assert validation['total_tests'] > 0, '[TC28] 应测试至少一个单项式 FAILED'
assert validation['max_error'] < 1.0e-10, '[TC28] 精确求积规则误差应为0 FAILED'

# ---- TC29: 收敛阶估计为正 ----
demo_errors = [1.0e-2, 1.0e-4, 1.0e-6]
demo_npts = [10, 50, 250]
p_est = convergence_order_estimate(demo_errors, demo_npts)
assert p_est > 0.0, '[TC29] 收敛阶应为正 FAILED'

# ---- TC30: main() 零参数运行返回0 ----
ret = main()
assert ret == 0, '[TC30] main() 应返回0 FAILED'

print('\n全部 30 个测试通过!\n')
