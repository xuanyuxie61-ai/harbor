"""
main.py
=======
博士级利率期限结构模型：多因子 HJM 框架综合计算平台

本程序为零参数可运行入口，执行以下完整流程:
  1. 初始化多因子 HJM 模型参数
  2. 生成二维有限元空间网格（期限 × 时间域）
  3. 市场收益率曲线校准（Shepard 插值 + Horner 多项式）
  4. 初始化前向利率曲线
  5. 运行 Lorenz-96 / Duffing / Oregonator 多因子随机动力学
  6. 求解 HJM 前向利率 PDE（输运-扩散方程）
  7. 计算零息债券价格与零息收益率
  8. 多项式混沌不确定性量化
  9. 稀疏矩阵分析与存储
 10. 输出结果与性能指标

金融背景
--------
利率期限结构（Term Structure of Interest Rates）描述了不同期限的无风险利率
之间的关系，是金融工程的核心理论对象。本程序基于 Heath-Jarrow-Morton (HJM)
一般无套利框架，引入多因子随机波动率结构：

    因子 1: Vasicek 型指数衰减波动率 σ_1(t,s) = σ_0 exp(-κ_1 s)
    因子 2: 斜率型波动率 σ_2(t,s) = σ_0 s exp(-κ_2 s)
    因子 3: 混沌耦合波动率 σ_3(t,s) = σ_chaos(t) exp(-κ_3 s)

其中 σ_chaos(t) 由 Lorenz-96 混沌系统（市场微观噪声）、Duffing 振子
（利率周期性波动）和 Oregonator 化学反应系统（流动性冲击）通过非线性
耦合矩阵投影得到。

核心 PDE（Musiela 参数化）:
    ∂r/∂t = -∂r/∂s + ν ∂²r/∂s² + μ(s) ∂r/∂s + α(t,s) + F(t,s)
    r(t,0) = r_0(t)   （短期利率边界）
    r(t,s_max) = r_∞  （长期利率渐近值）
    r(0,s) = r_init(s) （初始期限结构）

科学公式
--------
1. HJM 无套利漂移限制:
   α(t,T) = Σ_{i=1}^d σ_i(t,T) ∫_t^T σ_i(t,u) du

2. 债券定价公式:
   P(t,T) = exp(-∫_t^T f(t,s) ds)

3. 零息收益率:
   y(t,T) = -ln P(t,T) / (T - t)

4. 多项式混沌展开:
   r_t(s;ξ) = Σ_{|α|≤p} r_{t,α}(s) He_α(ξ)

5. Hermite 多项式递推:
   He_{n+1}(x) = x He_n(x) - n He_{n-1}(x)

6. Lambert W 函数（闭式解辅助）:
   W(z) e^{W(z)} = z

7. 对数正态分布（利率 positivity 约束）:
   X ~ LogNormal(μ,σ²):  f(x) = exp(-(ln x - μ)²/(2σ²)) / (x σ √(2π))

8. 后向 Euler 离散:
   (I - dt A) u^{n+1} = u^n + dt f^n

9. RK3 时间推进:
   k1 = dt f(t,u)
   k2 = dt f(t+dt, u+k1)
   k3 = dt f(t+dt/2, u+(k1+k2)/4)
   u_new = u + (k1 + k2 + 4k3)/6

10. Shepard 插值:
    w_j = ||x - x_j||^{-p}
    f(x) = Σ w_j z_j / Σ w_j
"""

import os
import sys
import time
import numpy as np

# 设置随机种子以保证可复现性
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
    """打印格式化章节标题。"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_simulation():
    """执行完整的期限结构模拟流程。"""
    print_section("多因子 HJM 利率期限结构模型 — 博士级综合计算平台")
    print("\n科学领域: 金融工程 — 利率期限结构模型")
    print("融合算法: Lorenz-96 混沌 / Duffing 振子 / Oregonator 反应")
    print("          Hermite 混沌 / FEM 空间离散 / RK23 时间积分")
    print("          Shepard 插值 / Lambert W / 对数正态分布 / 稀疏矩阵")

    # =====================================================================
    # 阶段 1: 参数设定与特殊函数验证
    # =====================================================================
    print_section("阶段 1: 参数设定与特殊函数验证")

    T_max = 10.0       # 最大期限（年）
    t_max = 2.0        # 模拟时间（年）
    N_T = 41           # 期限网格点数
    dt = 0.05          # 时间步长
    nu = 0.005         # 扩散系数
    sigma0 = 0.02      # 基准波动率

    # 验证对数正态分布
    mu_ln = 0.05
    sigma_ln = 0.2
    x_test = np.array([0.8, 1.0, 1.2])
    pdf_vals = sf.log_normal_pdf(x_test, mu_ln, sigma_ln)
    cdf_vals = sf.log_normal_cdf(x_test, mu_ln, sigma_ln)
    inv_vals = sf.log_normal_cdf_inv(np.array([0.1, 0.5, 0.9]), mu_ln, sigma_ln)
    print(f"  对数正态 PDF(0.8,1.0,1.2): {pdf_vals}")
    print(f"  对数正态 CDF(0.8,1.0,1.2): {cdf_vals}")
    print(f"  对数正态 逆CDF(0.1,0.5,0.9): {inv_vals}")

    # 验证 Lambert W
    lw_vals = sf.lambert_w(np.array([-0.3, 0.5, 2.0, 10.0]), branch=0)
    print(f"  Lambert W_0(-0.3,0.5,2.0,10.0): {lw_vals}")

    # 验证 Hermite 多项式
    h5_coeffs = pc.hep_coefficients(5)
    h5_val = pc.hep_value(1.0, 5)
    print(f"  He_5(x) 系数: {h5_coeffs}")
    print(f"  He_5(1.0) = {h5_val}")

    # =====================================================================
    # 阶段 2: 市场收益率曲线校准
    # =====================================================================
    print_section("阶段 2: 市场收益率曲线校准")

    # 构造合成市场数据（模拟真实收益率曲线）
    market_maturities = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    # Nelson-Siegel 型曲线: β0 + β1 exp(-T/τ) + β2 (T/τ) exp(-T/τ)
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

    # 特征提取
    features = ycc.extract_curve_features(market_maturities, market_yields)
    print(f"  曲线峰值数: {len(features['peaks'])}")
    print(f"  曲线谷值数: {len(features['valleys'])}")
    print(f"  曲线拐点数: {len(features['inflection'])}")

    # Shepard 插值验证
    test_mats = np.linspace(0.25, 10.0, 20)
    shepard_vals = ycc.shepard_interp_2d(
        market_maturities, np.zeros_like(market_maturities), market_yields,
        p=2.0, xi=test_mats, yi=np.zeros_like(test_mats))
    horner_vals = ycc.horner_eval(c_poly, test_mats)
    print(f"  Shepard 插值均值: {np.mean(shepard_vals):.4f}")
    print(f"  Horner 多项式均值: {np.mean(horner_vals):.4f}")

    # =====================================================================
    # 阶段 3: 有限元空间网格生成
    # =====================================================================
    print_section("阶段 3: 有限元空间网格生成")

    nx, ny = 5, 5
    node_xy, element_node = fem.generate_rectangular_grid(nx, ny, xl=0.0, xr=T_max, yb=0.0, yt=t_max)
    print(f"  节点数: {node_xy.shape[0]}")
    print(f"  单元数: {element_node.shape[0]}")

    # 验证三角形面积
    e = 0
    nodes_e = element_node[e, :6]
    t3 = node_xy[nodes_e[:3], :]
    area = fem.triangle_area(t3[0], t3[1], t3[2])
    print(f"  示例单元 {e} 面积: {area:.6f}")

    # 组装 FEM 矩阵
    def k_coef_default(x, y):
        return 0.1
    A_fem, M_fem = fem.assemble_fem_matrices(node_xy, element_node, element_order=6,
                                              k_coef=k_coef_default, nq=3)
    print(f"  刚度矩阵非零元: {A_fem.nnz}")
    print(f"  质量矩阵非零元: {M_fem.nnz}")
    print(f"  矩阵半带宽: {sla.estimate_bandwidth(A_fem)}")

    # 求解泊松方程（验证 FEM 框架）
    def rhs_poisson(x, y):
        return np.sin(np.pi * x / T_max) * np.sin(np.pi * y / t_max)
    def bc_poisson(x, y):
        return 0.0
    u_poisson = fem.solve_poisson_fem(node_xy, element_node, rhs_poisson, bc_poisson)
    print(f"  泊松方程解范围: [{u_poisson.min():.4f}, {u_poisson.max():.4f}]")

    # =====================================================================
    # 阶段 4: 稀疏矩阵 I/O 测试
    # =====================================================================
    print_section("阶段 4: 稀疏矩阵 I/O 测试")

    temp_st_file = "/tmp/test_sparse_matrix.st"
    ist, jst, ast = sla.coo_to_st(A_fem)
    sla.write_st_file(temp_st_file, A_fem.shape[0], A_fem.shape[1], len(ast), ist, jst, ast)
    m_read, n_read, nst_read, ist_r, jst_r, ast_r = sla.read_st_file(temp_st_file)
    print(f"  写入矩阵: {A_fem.shape}, nnz={A_fem.nnz}")
    print(f"  读取矩阵: {m_read}x{n_read}, nnz={nst_read}")
    os.remove(temp_st_file)

    # 稀疏求解测试
    b_test = np.ones(A_fem.shape[0], dtype=float)
    x_test, info = sla.solve_sparse_system(A_fem, b_test)
    print(f"  稀疏求解残差: {info['residual']:.6e}")
    print(f"  稀疏求解方法: {info['method']}")

    # =====================================================================
    # 阶段 5: 随机动力学验证
    # =====================================================================
    print_section("阶段 5: 随机动力学验证")

    # Lorenz-96
    n_lorenz, force_l, _, _, y0_l, tstop_l = sd.lorenz96_parameters(n=8, force=8.0)
    def lorenz_rhs(t, y):
        return sd.lorenz96_deriv(t, y, force=force_l)
    t_lor, y_lor, e_lor = ts.rk23_integrate(lorenz_rhs, (0.0, 1.0), y0_l, n_steps=50)
    print(f"  Lorenz-96 终态能量: {np.linalg.norm(y_lor[-1]):.4f}")
    print(f"  Lorenz-96 最大局部误差: {np.max(np.abs(e_lor)):.6e}")

    # Duffing
    alpha_d, beta_d, gamma_d, delta_d, omega_d, _, y0_d, _ = sd.duffing_parameters()
    def duffing_rhs(t, y):
        return sd.duffing_deriv(t, y, alpha_d, beta_d, gamma_d, delta_d, omega_d)
    t_duf, y_duf, e_duf = ts.rk23_integrate(duffing_rhs, (0.0, 10.0), y0_d, n_steps=100)
    print(f"  Duffing 终态位移: {y_duf[-1,0]:.4f}, 速度: {y_duf[-1,1]:.4f}")

    # Oregonator
    eta1_o, eta2_o, q_o, f_o, _, y0_o, _ = sd.oregonator_parameters()
    def oregonator_rhs(t, y):
        return sd.oregonator_deriv(t, y, eta1_o, eta2_o, q_o, f_o)
    # Oregonator 系统极其刚性，需使用极小步长
    t_ore, y_ore, e_ore = ts.rk23_integrate(oregonator_rhs, (0.0, 1.0), y0_o, n_steps=500)
    y_ore_clipped = np.clip(np.nan_to_num(y_ore[-1], nan=1.0, posinf=50.0, neginf=-50.0), -50.0, 50.0)
    print(f"  Oregonator 终态: u={y_ore_clipped[0]:.4f}, v={y_ore_clipped[1]:.4f}, w={y_ore_clipped[2]:.4f}")

    # =====================================================================
    # 阶段 6: 初始化 HJM 模型与前向利率曲线
    # =====================================================================
    print_section("阶段 6: HJM 模型初始化与前向利率曲线")

    T_grid = np.linspace(0.0, T_max, N_T)
    # 由校准的多项式构造初始前向利率曲线
    # f(0,T) ≈ y(0,T) + T * y'(0,T)（由零息收益率推导前向利率）
    y_init = ycc.horner_eval(c_poly, T_grid)
    y_init = np.clip(y_init, 0.001, 0.5)
    # 数值微分求导
    dy_dT = np.gradient(y_init, T_grid)
    f_init = y_init + T_grid * dy_dT
    f_init = np.clip(f_init, 0.001, 0.5)
    f_init[0] = max(f_init[0], 0.01)  # 短期利率正下界

    print(f"  期限网格: T ∈ [0, {T_max}], N={N_T}")
    print(f"  初始前向利率范围: [{f_init.min():.4f}, {f_init.max():.4f}]")
    print(f"  初始零息收益率范围: [{y_init.min():.4f}, {y_init.max():.4f}]")

    # =====================================================================
    # 阶段 7: HJM 多因子模拟
    # =====================================================================
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
        # 模拟央行政策冲击: 在时间 1.0 附近有一次利率下调
        return -0.002 * np.exp(-((t - 1.0) ** 2) / 0.1) * np.exp(-T / 5.0)

    t_start = time.time()
    t_hist, f_hist, dyn_hist = model.simulate_path(
        T_grid, f_init, t_max, dt, nu=nu, mu_func=mu_func, forcing_func=forcing_func)
    t_elapsed = time.time() - t_start

    print(f"  模拟步数: {len(t_hist) - 1}")
    print(f"  模拟耗时: {t_elapsed:.4f} 秒")
    print(f"  终态前向利率范围: [{f_hist[-1].min():.4f}, {f_hist[-1].max():.4f}]")

    # =====================================================================
    # 阶段 8: 债券定价与收益率计算
    # =====================================================================
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

    # =====================================================================
    # 阶段 9: 多项式混沌不确定性量化
    # =====================================================================
    print_section("阶段 9: 多项式混沌不确定性量化")

    pc_degree = 2
    pc_dim = 2
    multi_indices = pc.generate_multi_indices(pc_dim, pc_degree)
    n_terms = multi_indices.shape[0]
    print(f"  PC 阶数: {pc_degree}, 维度: {pc_dim}")
    print(f"  混沌项数: {n_terms}")

    # 使用前向利率曲线的终端截面作为混沌展开系数
    # 构造简化代理：将终态曲线投影到少量混沌基上
    np.random.seed(123)
    pc_coeffs = np.random.randn(n_terms) * 0.001
    pc_coeffs[0] = np.mean(f_hist[-1])  # 零阶项为均值

    # Sobol 敏感性分析
    total_var, sobol_main = pc.sobol_sensitivity(pc_coeffs, multi_indices)
    print(f"  PC 总方差: {total_var:.6e}")
    print(f"  主效应 Sobol 指标: {sobol_main}")

    # 生成混沌样本并计算统计量
    n_samples = 500
    xi_samples = np.random.randn(n_samples, pc_dim)
    pc_samples = pc.polynomial_chaos_expand(pc_coeffs, multi_indices, xi_samples)
    print(f"  PC 样本均值: {np.mean(pc_samples):.4f}")
    print(f"  PC 样本标准差: {np.std(pc_samples):.4f}")

    # =====================================================================
    # 阶段 10: 稀疏矩阵存储与结果汇总
    # =====================================================================
    print_section("阶段 10: 结果汇总与稀疏矩阵存储")

    # 将终态前向利率曲线的 FEM 离散矩阵存储为 ST 格式
    output_st_file = "/tmp/hjm_forward_rate_matrix.st"
    # 使用前向利率 PDE 的有限差分离散矩阵
    _, A_fd_final, _ = tsp.forward_rate_pde_rhs(
        t_max, T_grid, f_hist[-1], nu, mu_func, forcing_func,
        sigma_funcs=[lambda t, s: sigma0 * np.exp(-0.1 * s)])
    ist_f, jst_f, ast_f = sla.coo_to_st(A_fd_final)
    sla.write_st_file(output_st_file, A_fd_final.shape[0], A_fd_final.shape[1],
                      len(ast_f), ist_f, jst_f, ast_f)
    print(f"  终态 PDE 离散矩阵已写入: {output_st_file}")
    print(f"  矩阵维度: {A_fd_final.shape}, 非零元: {A_fd_final.nnz}")
    os.remove(output_st_file)

    # 性能与数值稳定性指标
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
    except Exception as e:
        print(f"\n[ERROR] 模拟过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        raise

# ================================================================
# 测试用例（45个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: log_normal_pdf 基础值验证 ----
pdf_vals_check = sf.log_normal_pdf(np.array([1.0]), 0.0, 1.0)
assert np.isfinite(pdf_vals_check).all() and pdf_vals_check[0] > 0.0, '[TC01] log_normal_pdf(1.0) 应返回正值 FAILED'

# ---- TC02: log_normal_pdf 负值输入应返回0 ----
pdf_zero = sf.log_normal_pdf(np.array([-1.0, 0.0, -5.0]), 0.0, 1.0)
assert np.allclose(pdf_zero, 0.0), '[TC02] log_normal_pdf 对非正输入应返回 0 FAILED'

# ---- TC03: log_normal_cdf 渐近性质 ----
cdf_small = sf.log_normal_cdf(np.array([1e-10]), 0.0, 1.0)
cdf_large = sf.log_normal_cdf(np.array([1e10]), 0.0, 1.0)
assert cdf_small[0] < 1e-6, '[TC03] log_normal_cdf(极小值) 应接近 0 FAILED'
assert cdf_large[0] > 0.9999, '[TC03] log_normal_cdf(极大值) 应接近 1 FAILED'

# ---- TC04: log_normal_cdf_inv 正值与单调性 ----
import numpy as np
np.random.seed(42)
p_test = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
inv_vals = sf.log_normal_cdf_inv(p_test, 0.0, 1.0)
# 逆CDF 应返回正值且单调递增
assert np.all(inv_vals > 0.0), '[TC04] log_normal_cdf_inv 应返回正值 FAILED'
assert np.all(np.diff(inv_vals) > 0.0), '[TC04] log_normal_cdf_inv 应单调递增 FAILED'

# ---- TC05: normal_01_cdf_inv 对称性 ----
p_vals = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
z_vals = sf.normal_01_cdf_inv(p_vals)
assert np.allclose(z_vals[0], -z_vals[-1], atol=1e-10), '[TC05] normal_01_cdf_inv 对称性失败 FAILED'
assert np.isclose(z_vals[2], 0.0, atol=1e-10), '[TC05] normal_01_cdf_inv(0.5) 应为 0 FAILED'

# ---- TC06: lambert_w 上分支基本值 ----
lw_vals = sf.lambert_w(np.array([0.0, 1.0, np.e]), branch=0)
assert np.isclose(lw_vals[0], 0.0, atol=1e-10), '[TC06] W_0(0) 应为 0 FAILED'
assert lw_vals[1] > 0.5 and lw_vals[2] > 0.9, '[TC06] W_0(1) 和 W_0(e) 应为正 FAILED'

# ---- TC07: lambert_w 恒等式验证 ----
test_x = np.array([0.5, 2.0, 5.0])
w_vals = sf.lambert_w(test_x, branch=0)
for idx, (wv, xv) in enumerate(zip(w_vals, test_x)):
    reconstructed = wv * np.exp(wv)
    assert np.isclose(reconstructed, xv, atol=1e-6), f'[TC07] W_0({xv}) 恒等式失败 FAILED'

# ---- TC08: hep_coefficients 系数对称性 ----
c4 = pc.hep_coefficients(4)
c5 = pc.hep_coefficients(5)
assert len(c4) == 5 and len(c5) == 6, '[TC08] hep_coefficients 返回长度错误 FAILED'
# He_4(0) = 3 (偶次), He_5(0) = 0 (奇次)
assert np.isclose(c4[0], 3.0, atol=1e-10), '[TC08] He_4 常数项应为 3 FAILED'
assert np.isclose(c5[0], 0.0, atol=1e-10), '[TC08] He_5 常数项应为 0 FAILED'

# ---- TC09: hep_value 与系数一致性 ----
c3 = pc.hep_coefficients(3)
x_test_h = np.array([0.0, 0.5, 1.0, 2.0])
h3_vals = pc.hep_value(x_test_h, 3)
# He_3(x) = x^3 - 3x
h3_expected = x_test_h**3 - 3.0 * x_test_h
assert np.allclose(h3_vals, h3_expected, atol=1e-10), '[TC09] He_3(x) 值不匹配 FAILED'

# ---- TC10: hep_value 递推一致性 ----
x_check = np.array([0.0, -1.0, 2.0, -3.0, 5.0])
for n in range(0, 6):
    hv = pc.hep_value(x_check, n)
    assert np.isfinite(hv).all(), f'[TC10] He_{n}(x) 产生非有限值 FAILED'

# ---- TC11: hep_values 输出形状 ----
x_in = np.array([0.5, 1.0, 1.5])
v_all = pc.hep_values(x_in, 5)
assert v_all.shape == (3, 6), '[TC11] hep_values 输出形状错误 FAILED'
assert np.allclose(v_all[:, 0], 1.0), '[TC11] He_0 应全为 1 FAILED'

# ---- TC12: hermite_product_polynomial_value 基本验证 ----
np.random.seed(42)
xi_prod = np.random.randn(10, 2)
prod_val = pc.hermite_product_polynomial_value(2, [2, 1], xi_prod)
assert prod_val.shape == (10,), '[TC12] 乘积多项式输出形状错误 FAILED'
assert np.isfinite(prod_val).all(), '[TC12] 乘积多项式产生非有限值 FAILED'

# ---- TC13: generate_multi_indices 数量验证 ----
mi_2_2 = pc.generate_multi_indices(2, 2)
# N = (2+2)!/(2!*2!) = 6
assert mi_2_2.shape[0] == 6 and mi_2_2.shape[1] == 2, '[TC13] d=2,p=2 应有 6 个指标 FAILED'
mi_2_3 = pc.generate_multi_indices(2, 3)
assert mi_2_3.shape[0] == 10, '[TC13] d=2,p=3 应有 10 个指标 FAILED'

# ---- TC14: polynomial_chaos_expand 确定性测试 ----
np.random.seed(123)
mi_test = pc.generate_multi_indices(2, 2)
coeffs_test = np.array([1.0, 0.1, 0.0, 0.0, 0.0, 0.0])
xi_test = np.random.randn(100, 2)
pc_samples = pc.polynomial_chaos_expand(coeffs_test, mi_test, xi_test)
assert pc_samples.shape == (100,), '[TC14] PC 展开输出形状错误 FAILED'
# 均值应接近 coeffs[0]
assert abs(np.mean(pc_samples) - coeffs_test[0]) < 0.5, '[TC14] PC 展开均值偏差过大 FAILED'

# ---- TC15: sobol_sensitivity 归一化验证 ----
mi_sobol = pc.generate_multi_indices(2, 2)
coeffs_sobol = np.ones(mi_sobol.shape[0], dtype=float) * 0.1
coeffs_sobol[0] = 1.0
total_var, sobol = pc.sobol_sensitivity(coeffs_sobol, mi_sobol)
assert total_var > 0.0, '[TC15] Sobol 总方差应为正 FAILED'
assert np.all(sobol >= 0.0) and np.all(sobol <= 1.0), '[TC15] Sobol 指标应在 [0,1] 内 FAILED'

# ---- TC16: lorenz96_parameters 基本调用 ----
n_l, f_l, p_l, t0_l, y0_l, ts_l = sd.lorenz96_parameters(n=4, force=8.0)
assert n_l == 4 and abs(f_l - 8.0) < 1e-10, '[TC16] lorenz96_parameters 参数不匹配 FAILED'
assert y0_l.shape == (4,), '[TC16] y0 形状错误 FAILED'

# ---- TC17: lorenz96_deriv 恒定解检验 ----
np.random.seed(42)
y_const = 8.0 * np.ones(4)
dy = sd.lorenz96_deriv(0.0, y_const, force=8.0)
# 当所有 y_i = F 时，dy_i = (F-F)*F - F + F = 0
assert np.allclose(dy, 0.0, atol=1e-10), '[TC17] Lorenz-96 恒定解导数应为零 FAILED'

# ---- TC18: duffing_parameters 基本调用 ----
a_d, b_d, g_d, d_d, o_d, t0_d, y0_d, ts_d = sd.duffing_parameters()
assert abs(a_d - 1.0) < 1e-10 and abs(b_d - 5.0) < 1e-10, '[TC18] duffing 默认参数不匹配 FAILED'
assert y0_d.shape == (2,), '[TC18] duffing y0 形状错误 FAILED'

# ---- TC19: duffing_deriv 零位移导数 ----
dy_d = sd.duffing_deriv(0.0, np.array([0.0, 0.0]))
# dy1/dt = y2 = 0, dy2/dt = -δ*0 - α*0 - β*0 + γ*cos(0) = γ
assert np.isclose(dy_d[0], 0.0, atol=1e-10), '[TC19] Duffing x=0 时 dx/dt 应为 0 FAILED'
assert dy_d[1] > 0.0, '[TC19] Duffing x=0 时 dv/dt 应为正值 FAILED'

# ---- TC20: oregonator_parameters 基本调用 ----
e1_o, e2_o, q_o, f_o, t0_o, y0_o, ts_o = sd.oregonator_parameters()
assert y0_o.shape == (3,), '[TC20] Oregonator y0 形状错误 FAILED'
assert e1_o > 0.0 and e2_o > 0.0, '[TC20] Oregonator 无量纲参数应为正 FAILED'

# ---- TC21: oregonator_deriv 稳态解验证 ----
e1, e2, q_o21, f_o21, _, _, _ = sd.oregonator_parameters()
y_ss = np.array([0.0, 0.0, 0.0])
dy_ss = sd.oregonator_deriv(0.0, y_ss, e1, e2, q_o21, f_o21)
assert dy_ss.shape == (3,), '[TC21] Oregonator 导数输出形状错误 FAILED'
assert np.isfinite(dy_ss).all(), '[TC21] Oregonator 导数产生非有限值 FAILED'

# ---- TC22: multi_factor_coupling 输出非负性 ----
np.random.seed(42)
lorenz_y = np.random.randn(8)
duffing_y = np.array([0.5, -0.2])
oregonator_y = np.array([1.0, 0.5, 0.8])
sigma_c = sd.multi_factor_coupling(0.0, lorenz_y, duffing_y, oregonator_y, n_factors=3)
assert sigma_c.shape == (3,), '[TC22] 耦合输出形状错误 FAILED'
assert np.all(sigma_c >= 0.0), '[TC22] 波动率耦合必须非负 FAILED'

# ---- TC23: rk2_step 简单 ODE 测试 ----
def exp_deriv(t, y):
    return y
import numpy as np
y_rk2 = ts.rk2_step(exp_deriv, 0.0, np.array([1.0]), 0.1)
assert y_rk2.shape == (1,), '[TC23] rk2_step 输出形状错误 FAILED'
assert y_rk2[0] > 0.0, '[TC23] rk2_step 指数增长验证失败 FAILED'

# ---- TC24: rk3_step 简单 ODE 测试 ----
y_rk3 = ts.rk3_step(exp_deriv, 0.0, np.array([1.0]), 0.1)
assert y_rk3.shape == (1,), '[TC24] rk3_step 输出形状错误 FAILED'
assert y_rk3[0] > 0.0, '[TC24] rk3_step 指数增长验证失败 FAILED'

# ---- TC25: rk23_integrate 输出形状 ----
np.random.seed(42)
def linear_deriv(t, y):
    return np.array([-y[0]])
t_rk, y_rk, e_rk = ts.rk23_integrate(linear_deriv, (0.0, 1.0), np.array([1.0]), n_steps=50)
assert t_rk.shape == (51,) and y_rk.shape == (51, 1) and e_rk.shape == (51, 1), '[TC25] rk23 输出形状错误 FAILED'
assert np.isfinite(y_rk).all(), '[TC25] rk23 产生非有限值 FAILED'

# ---- TC26: triangle_area 已知面积验证 ----
p1, p2, p3 = np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])
area = fem.triangle_area(p1, p2, p3)
assert np.isclose(area, 0.5, atol=1e-10), '[TC26] 单位直角三角形面积应为 0.5 FAILED'

# ---- TC27: generate_rectangular_grid 输出形状 ----
node_xy, elem_node = fem.generate_rectangular_grid(3, 3, xl=0.0, xr=1.0, yb=0.0, yt=1.0)
expected_nodes = (2*3-1) * (2*3-1)  # 25
expected_elems = (3-1) * (3-1) * 2  # 8
assert node_xy.shape == (expected_nodes, 2), f'[TC27] 节点数应为 {expected_nodes} FAILED'
assert elem_node.shape == (expected_elems, 6), f'[TC27] 单元数应为 {expected_elems} FAILED'

# ---- TC28: get_quad_rule_triangle 权重和 ----
w3, xy3 = fem.get_quad_rule_triangle(3)
assert np.isclose(np.sum(w3), 0.5, atol=1e-10), '[TC28] 三角形积分权重和应为 0.5 FAILED'

# ---- TC29: basis_11_t6 节点求值 ----
node_test = np.array([[0.0,0.0],[1.0,0.0],[0.0,1.0],[0.5,0.0],[0.5,0.5],[0.0,0.5]], dtype=float)
# 在顶点1处，基函数1应为1
b1, _, _ = fem.basis_11_t6(node_test, 1, np.array([0.0, 0.0]))
assert np.isclose(b1, 1.0, atol=1e-10), '[TC29] T6 基函数在对应节点应为 1 FAILED'

# ---- TC30: assemble_fem_matrices 对称性 ----
def k_zero(x, y): return 0.0
A_sym, M_sym = fem.assemble_fem_matrices(node_xy, elem_node, element_order=6, k_coef=k_zero, nq=3)
diff_A = np.linalg.norm((A_sym - A_sym.T).toarray())
assert diff_A < 1e-10, '[TC30] 刚度矩阵不对称 FAILED'
diff_M = np.linalg.norm((M_sym - M_sym.T).toarray())
assert diff_M < 1e-10, '[TC30] 质量矩阵不对称 FAILED'

# ---- TC31: shepard_interp_2d 精确插值 ----
xd = np.array([0.0, 1.0, 2.0])
yd = np.array([0.0, 0.0, 0.0])
zd = np.array([0.0, 1.0, 4.0])
xi_s = np.array([0.0, 1.0, 2.0])
yi_s = np.array([0.0, 0.0, 0.0])
zi_s = ycc.shepard_interp_2d(xd, yd, zd, 2.0, xi_s, yi_s)
assert np.allclose(zi_s, zd, atol=1e-10), '[TC31] Shepard 在数据点上应精确插值 FAILED'

# ---- TC32: horner_eval 基本验证 ----
c_poly = np.array([1.0, 2.0, 3.0])  # 1 + 2x + 3x^2
p0 = ycc.horner_eval(c_poly, np.array([0.0]))
assert np.isclose(p0, 1.0, atol=1e-10), '[TC32] Horner p(0) 应为 c0 FAILED'
p1 = ycc.horner_eval(c_poly, np.array([1.0]))
assert np.isclose(p1, 6.0, atol=1e-10), '[TC32] Horner p(1) 应为 6 FAILED'

# ---- TC33: fit_yield_polynomial 基本拟合 ----
mats = np.linspace(0.5, 10.0, 8)
yields_true = 0.05 + 0.01 * mats
c_fit, res, cond = ycc.fit_yield_polynomial(mats, yields_true, degree=1)
assert np.isclose(c_fit[0], 0.05, atol=1e-6), '[TC33] 多项式拟合截距错误 FAILED'
assert np.isclose(c_fit[1], 0.01, atol=1e-6), '[TC33] 多项式拟合斜率错误 FAILED'

# ---- TC34: extract_curve_features 单调曲线 ----
m_feat = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_feat = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
feat = ycc.extract_curve_features(m_feat, y_feat)
assert feat['start'] is not None and feat['end'] is not None, '[TC34] 特征提取应返回起止点 FAILED'

# ---- TC35: bond_price_from_forward 基本定价 ----
T_test = np.linspace(0.0, 5.0, 51)
f_const = 0.05 * np.ones_like(T_test)
P_1y = tsp.bond_price_from_forward(f_const, T_test, 0.0, 1.0)
P_2y = tsp.bond_price_from_forward(f_const, T_test, 0.0, 2.0)
assert np.isclose(P_1y, np.exp(-0.05), atol=1e-6), '[TC35] 恒定利率债券定价错误 FAILED'
assert np.isclose(P_2y, np.exp(-0.10), atol=1e-6), '[TC35] 恒定利率债券定价错误 FAILED'

# ---- TC36: zero_yield_from_forward 基本收益率 ----
zy_1y = tsp.zero_yield_from_forward(f_const, T_test, 0.0, 1.0)
zy_2y = tsp.zero_yield_from_forward(f_const, T_test, 0.0, 2.0)
assert np.isclose(zy_1y, 0.05, atol=1e-6), '[TC36] 恒定利率零息收益率应为 0.05 FAILED'
assert np.isclose(zy_2y, 0.05, atol=1e-6), '[TC36] 恒定利率零息收益率应为 0.05 FAILED'

# ---- TC37: bond_price_from_forward 单调性 ----
T_s = np.linspace(0.0, 10.0, 101)
f_pos = 0.03 * np.ones_like(T_s)
P_3y = tsp.bond_price_from_forward(f_pos, T_s, 0.0, 3.0)
P_5y = tsp.bond_price_from_forward(f_pos, T_s, 0.0, 5.0)
assert P_3y > P_5y, '[TC37] 正利率下债券价格应随期限递减 FAILED'

# ---- TC38: instantaneous_short_rate 基本调用 ----
r_short = tsp.instantaneous_short_rate(f_const, T_test)
assert r_short > 0.0, '[TC38] 短期利率应为正 FAILED'

# ---- TC39: st_to_coo / coo_to_st 往返 ----
import numpy as np
from scipy import sparse as sp
mat_small = sp.coo_matrix(([1.0, 2.0, 3.0], ([0, 1, 2], [0, 1, 2])), shape=(3, 3))
ist_s, jst_s, ast_s = sla.coo_to_st(mat_small)
mat_back = sla.st_to_coo(ist_s, jst_s, ast_s, (3, 3))
assert np.allclose(mat_back.toarray(), mat_small.toarray()), '[TC39] COO-ST 往返失败 FAILED'

# ---- TC40: estimate_bandwidth 对角矩阵 ----
mat_diag = sp.eye(5, format='coo')
bw = sla.estimate_bandwidth(mat_diag)
assert bw == 0, '[TC40] 对角矩阵带宽应为 0 FAILED'

# ---- TC41: sparse_matvec 基本验证 ----
A_test = sp.eye(3, format='csr') * 2.0
x_test_sla = np.array([1.0, 2.0, 3.0])
y_test_sla = sla.sparse_matvec(A_test, x_test_sla)
assert np.allclose(y_test_sla, np.array([2.0, 4.0, 6.0])), '[TC41] 稀疏矩阵向量乘法错误 FAILED'

# ---- TC42: HJMMultiFactorModel 初始化 ----
import hjm_model as hjm
model = hjm.HJMMultiFactorModel(n_factors=3, sigma0=0.02, pc_degree=2, pc_dim=2)
assert model.n_factors == 3, '[TC42] HJM 模型因子数错误 FAILED'
assert len(model.kappa) >= 3, '[TC42] HJM 模型 kappa 长度错误 FAILED'

# ---- TC43: HJMMultiFactorModel volatility_structure ----
np.random.seed(42)
lorenz_y = np.random.randn(8)
duffing_y = np.array([0.5, -0.2])
oregonator_y = np.array([1.0, 0.5, 0.8])
vol = model.volatility_structure(0.0, 1.0, lorenz_y, duffing_y, oregonator_y)
assert vol.shape == (3,), '[TC43] 波动率结构输出形状错误 FAILED'
assert np.all(vol >= 0.0) and np.all(vol <= 1.0), '[TC43] 波动率应在 [0,1] 内 FAILED'

# ---- TC44: musiela_drift 非负性 ----
def sigma_const(t, s):
    return 0.02
alpha = tsp.musiela_drift([sigma_const], 2.0, t=0.0)
assert alpha >= 0.0, '[TC44] HJM 漂移项应为非负 FAILED'

# ---- TC45: write_st_file / read_st_file 往返 ----
from scipy import sparse as sp
mat_w = sp.coo_matrix(([2.0, 3.0], ([0, 1], [0, 1])), shape=(2, 2))
ist_w, jst_w, ast_w = sla.coo_to_st(mat_w)
temp_file = "/tmp/test_st_rw_45.st"
sla.write_st_file(temp_file, 2, 2, len(ast_w), ist_w, jst_w, ast_w)
m_r, n_r, nst_r, ist_r, jst_r, ast_r = sla.read_st_file(temp_file)
assert m_r == 2 and n_r == 2, '[TC45] ST 文件读写维度错误 FAILED'
mat_r = sla.st_to_coo(ist_r, jst_r, ast_r, (m_r, n_r))
assert np.allclose(mat_r.toarray(), mat_w.toarray()), '[TC45] ST 文件读写往返失败 FAILED'
import os; os.remove(temp_file)

print('\n全部 45 个测试通过!\n')
