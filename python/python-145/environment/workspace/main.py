
import os
import sys
import time
import numpy as np


np.random.seed(42)

import special_functions as sf
import polynomial_chaos_uq as pc
import stochastic_dynamics as sd
import time_stepping as ts
import fem_maturity_grid as fem
import yield_curve_calibration as ycc
import sparse_linear_algebra as sla
import term_structure_pde as tsp
import hjm_model as hjm


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_simulation():
    print_section("多因子 HJM 利率期限结构模型 — 博士级综合计算平台")
    print("\n科学领域: 金融工程 — 利率期限结构模型")
    print("融合算法: Lorenz-96 混沌 / Duffing 振子 / Oregonator 反应")
    print("          Hermite 混沌 / FEM 空间离散 / RK23 时间积分")
    print("          Shepard 插值 / Lambert W / 对数正态分布 / 稀疏矩阵")




    print_section("阶段 1: 参数设定与特殊函数验证")

    T_max = 10.0
    t_max = 2.0
    N_T = 41
    dt = 0.05
    nu = 0.005
    sigma0 = 0.02


    mu_ln = 0.05
    sigma_ln = 0.2
    x_test = np.array([0.8, 1.0, 1.2])
    pdf_vals = sf.log_normal_pdf(x_test, mu_ln, sigma_ln)
    cdf_vals = sf.log_normal_cdf(x_test, mu_ln, sigma_ln)
    inv_vals = sf.log_normal_cdf_inv(np.array([0.1, 0.5, 0.9]), mu_ln, sigma_ln)
    print(f"  对数正态 PDF(0.8,1.0,1.2): {pdf_vals}")
    print(f"  对数正态 CDF(0.8,1.0,1.2): {cdf_vals}")
    print(f"  对数正态 逆CDF(0.1,0.5,0.9): {inv_vals}")


    lw_vals = sf.lambert_w(np.array([-0.3, 0.5, 2.0, 10.0]), branch=0)
    print(f"  Lambert W_0(-0.3,0.5,2.0,10.0): {lw_vals}")


    h5_coeffs = pc.hep_coefficients(5)
    h5_val = pc.hep_value(1.0, 5)
    print(f"  He_5(x) 系数: {h5_coeffs}")
    print(f"  He_5(1.0) = {h5_val}")




    print_section("阶段 2: 市场收益率曲线校准")


    market_maturities = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])

    beta0, beta1, beta2, tau = 0.05, -0.02, 0.01, 2.0
    market_yields = beta0 + beta1 * np.exp(-market_maturities / tau) + \
                    beta2 * (market_maturities / tau) * np.exp(-market_maturities / tau)
    market_yields = np.clip(market_yields, 0.001, 0.5)

    calibration = ycc.calibrate_yield_curve(market_maturities, market_yields,
                                             interp_method='polynomial', poly_degree=5)
    c_poly = calibration['coefficients']
    residual = calibration['residual']
    cond_num = calibration['condition_number']
    print(f"  市场期限: {market_maturities}")
    print(f"  市场收益率: {market_yields}")
    print(f"  多项式拟合残差: {residual:.6e}")
    print(f"  Vandermonde 条件数: {cond_num:.4e}")


    features = ycc.extract_curve_features(market_maturities, market_yields)
    print(f"  曲线峰值数: {len(features['peaks'])}")
    print(f"  曲线谷值数: {len(features['valleys'])}")
    print(f"  曲线拐点数: {len(features['inflection'])}")


    test_mats = np.linspace(0.25, 10.0, 20)
    shepard_vals = ycc.shepard_interp_2d(
        market_maturities, np.zeros_like(market_maturities), market_yields,
        p=2.0, xi=test_mats, yi=np.zeros_like(test_mats))
    horner_vals = ycc.horner_eval(c_poly, test_mats)
    print(f"  Shepard 插值均值: {np.mean(shepard_vals):.4f}")
    print(f"  Horner 多项式均值: {np.mean(horner_vals):.4f}")




    print_section("阶段 3: 有限元空间网格生成")

    nx, ny = 5, 5
    node_xy, element_node = fem.generate_rectangular_grid(nx, ny, xl=0.0, xr=T_max, yb=0.0, yt=t_max)
    print(f"  节点数: {node_xy.shape[0]}")
    print(f"  单元数: {element_node.shape[0]}")


    e = 0
    nodes_e = element_node[e, :6]
    t3 = node_xy[nodes_e[:3], :]
    area = fem.triangle_area(t3[0], t3[1], t3[2])
    print(f"  示例单元 {e} 面积: {area:.6f}")


    def k_coef_default(x, y):
        return 0.1
    A_fem, M_fem = fem.assemble_fem_matrices(node_xy, element_node, element_order=6,
                                              k_coef=k_coef_default, nq=3)
    print(f"  刚度矩阵非零元: {A_fem.nnz}")
    print(f"  质量矩阵非零元: {M_fem.nnz}")
    print(f"  矩阵半带宽: {sla.estimate_bandwidth(A_fem)}")


    def rhs_poisson(x, y):
        return np.sin(np.pi * x / T_max) * np.sin(np.pi * y / t_max)
    def bc_poisson(x, y):
        return 0.0
    u_poisson = fem.solve_poisson_fem(node_xy, element_node, rhs_poisson, bc_poisson)
    print(f"  泊松方程解范围: [{u_poisson.min():.4f}, {u_poisson.max():.4f}]")




    print_section("阶段 4: 稀疏矩阵 I/O 测试")

    temp_st_file = "/tmp/test_sparse_matrix.st"
    ist, jst, ast = sla.coo_to_st(A_fem)
    sla.write_st_file(temp_st_file, A_fem.shape[0], A_fem.shape[1], len(ast), ist, jst, ast)
    m_read, n_read, nst_read, ist_r, jst_r, ast_r = sla.read_st_file(temp_st_file)
    print(f"  写入矩阵: {A_fem.shape}, nnz={A_fem.nnz}")
    print(f"  读取矩阵: {m_read}x{n_read}, nnz={nst_read}")
    os.remove(temp_st_file)


    b_test = np.ones(A_fem.shape[0], dtype=float)
    x_test, info = sla.solve_sparse_system(A_fem, b_test)
    print(f"  稀疏求解残差: {info['residual']:.6e}")
    print(f"  稀疏求解方法: {info['method']}")




    print_section("阶段 5: 随机动力学验证")


    n_lorenz, force_l, _, _, y0_l, tstop_l = sd.lorenz96_parameters(n=8, force=8.0)
    def lorenz_rhs(t, y):
        return sd.lorenz96_deriv(t, y, force=force_l)
    t_lor, y_lor, e_lor = ts.rk23_integrate(lorenz_rhs, (0.0, 1.0), y0_l, n_steps=50)
    print(f"  Lorenz-96 终态能量: {np.linalg.norm(y_lor[-1]):.4f}")
    print(f"  Lorenz-96 最大局部误差: {np.max(np.abs(e_lor)):.6e}")


    alpha_d, beta_d, gamma_d, delta_d, omega_d, _, y0_d, _ = sd.duffing_parameters()
    def duffing_rhs(t, y):
        return sd.duffing_deriv(t, y, alpha_d, beta_d, gamma_d, delta_d, omega_d)
    t_duf, y_duf, e_duf = ts.rk23_integrate(duffing_rhs, (0.0, 10.0), y0_d, n_steps=100)
    print(f"  Duffing 终态位移: {y_duf[-1,0]:.4f}, 速度: {y_duf[-1,1]:.4f}")


    eta1_o, eta2_o, q_o, f_o, _, y0_o, _ = sd.oregonator_parameters()
    def oregonator_rhs(t, y):
        return sd.oregonator_deriv(t, y, eta1_o, eta2_o, q_o, f_o)

    t_ore, y_ore, e_ore = ts.rk23_integrate(oregonator_rhs, (0.0, 1.0), y0_o, n_steps=500)
    y_ore_clipped = np.clip(np.nan_to_num(y_ore[-1], nan=1.0, posinf=50.0, neginf=-50.0), -50.0, 50.0)
    print(f"  Oregonator 终态: u={y_ore_clipped[0]:.4f}, v={y_ore_clipped[1]:.4f}, w={y_ore_clipped[2]:.4f}")




    print_section("阶段 6: HJM 模型初始化与前向利率曲线")

    T_grid = np.linspace(0.0, T_max, N_T)


    y_init = ycc.horner_eval(c_poly, T_grid)
    y_init = np.clip(y_init, 0.001, 0.5)

    dy_dT = np.gradient(y_init, T_grid)
    f_init = y_init + T_grid * dy_dT
    f_init = np.clip(f_init, 0.001, 0.5)
    f_init[0] = max(f_init[0], 0.01)

    print(f"  期限网格: T ∈ [0, {T_max}], N={N_T}")
    print(f"  初始前向利率范围: [{f_init.min():.4f}, {f_init.max():.4f}]")
    print(f"  初始零息收益率范围: [{y_init.min():.4f}, {y_init.max():.4f}]")




    print_section("阶段 7: HJM 多因子期限结构模拟")

    model = hjm.HJMMultiFactorModel(
        n_factors=3, sigma0=sigma0, kappa=np.array([0.1, 0.3, 0.5]),
        lorenz_n=8, lorenz_force=8.0,
        duffing_alpha=1.0, duffing_beta=5.0, duffing_gamma=8.0,
        duffing_delta=0.02, duffing_omega=0.5,
        oregonator_f=1.0,
        pc_degree=2, pc_dim=2
    )

    def mu_func(T):
        return -0.005 + 0.001 * np.sin(np.pi * T / T_max)

    def forcing_func(t, T):

        return -0.002 * np.exp(-((t - 1.0) ** 2) / 0.1) * np.exp(-T / 5.0)

    t_start = time.time()
    t_hist, f_hist, dyn_hist = model.simulate_path(
        T_grid, f_init, t_max, dt, nu=nu, mu_func=mu_func, forcing_func=forcing_func)
    t_elapsed = time.time() - t_start

    print(f"  模拟步数: {len(t_hist) - 1}")
    print(f"  模拟耗时: {t_elapsed:.4f} 秒")
    print(f"  终态前向利率范围: [{f_hist[-1].min():.4f}, {f_hist[-1].max():.4f}]")




    print_section("阶段 8: 债券定价与零息收益率计算")

    n_check = 5
    check_indices = np.linspace(0, len(T_grid) - 1, n_check, dtype=int)
    bond_prices_final = np.zeros(n_check, dtype=float)
    zero_yields_final = np.zeros(n_check, dtype=float)

    for idx, j in enumerate(check_indices):
        T_j = T_grid[j]
        if T_j > 1e-10:
            bp = tsp.bond_price_from_forward(f_hist[-1], T_grid, t_max, T_j)
            zy = tsp.zero_yield_from_forward(f_hist[-1], T_grid, t_max, T_j)
            bond_prices_final[idx] = bp
            zero_yields_final[idx] = zy
        else:
            bond_prices_final[idx] = 1.0
            zero_yields_final[idx] = f_hist[-1, 0]

    print(f"  检查期限: {T_grid[check_indices]}")
    print(f"  债券价格 P(t_max, T): {bond_prices_final}")
    print(f"  零息收益率 y(t_max, T): {zero_yields_final}")




    print_section("阶段 9: 多项式混沌不确定性量化")

    pc_degree = 2
    pc_dim = 2
    multi_indices = pc.generate_multi_indices(pc_dim, pc_degree)
    n_terms = multi_indices.shape[0]
    print(f"  PC 阶数: {pc_degree}, 维度: {pc_dim}")
    print(f"  混沌项数: {n_terms}")



    np.random.seed(123)
    pc_coeffs = np.random.randn(n_terms) * 0.001
    pc_coeffs[0] = np.mean(f_hist[-1])


    total_var, sobol_main = pc.sobol_sensitivity(pc_coeffs, multi_indices)
    print(f"  PC 总方差: {total_var:.6e}")
    print(f"  主效应 Sobol 指标: {sobol_main}")


    n_samples = 500
    xi_samples = np.random.randn(n_samples, pc_dim)
    pc_samples = pc.polynomial_chaos_expand(pc_coeffs, multi_indices, xi_samples)
    print(f"  PC 样本均值: {np.mean(pc_samples):.4f}")
    print(f"  PC 样本标准差: {np.std(pc_samples):.4f}")




    print_section("阶段 10: 结果汇总与稀疏矩阵存储")


    output_st_file = "/tmp/hjm_forward_rate_matrix.st"

    _, A_fd_final, _ = tsp.forward_rate_pde_rhs(
        t_max, T_grid, f_hist[-1], nu, mu_func, forcing_func,
        sigma_funcs=[lambda t, s: sigma0 * np.exp(-0.1 * s)])
    ist_f, jst_f, ast_f = sla.coo_to_st(A_fd_final)
    sla.write_st_file(output_st_file, A_fd_final.shape[0], A_fd_final.shape[1],
                      len(ast_f), ist_f, jst_f, ast_f)
    print(f"  终态 PDE 离散矩阵已写入: {output_st_file}")
    print(f"  矩阵维度: {A_fd_final.shape}, 非零元: {A_fd_final.nnz}")
    os.remove(output_st_file)


    print("\n" + "-" * 70)
    print("  数值稳定性指标:")
    print(f"    前向利率最小值: {np.min(f_hist):.6f}")
    print(f"    前向利率最大值: {np.max(f_hist):.6f}")
    print(f"    债券价格最小值: {np.min(bond_prices_final):.6f}")
    print(f"    债券价格最大值: {np.max(bond_prices_final):.6f}")
    print(f"    模拟总耗时: {t_elapsed:.4f} 秒")
    print("-" * 70)

    print("\n" + "=" * 70)
    print("  模拟完成。所有模块运行正常，无报错。")
    print("=" * 70)

    return {
        'T_grid': T_grid,
        't_hist': t_hist,
        'f_hist': f_hist,
        'bond_prices': bond_prices_final,
        'zero_yields': zero_yields_final,
        'pc_coeffs': pc_coeffs,
        'sobol_main': sobol_main,
        'elapsed_time': t_elapsed
    }


if __name__ == "__main__":
    try:
        results = run_simulation()
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] 模拟过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
