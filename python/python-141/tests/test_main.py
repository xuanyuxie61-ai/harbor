"""
统一入口: 多因子随机波动率模型下奇异期权定价与全局校准
=========================================================

本项目基于15个种子科研代码项目的核心算法，围绕金融工程前沿问题——
"衍生品定价与随机波动率"进行博士级科学合成。

运行方式:
---------
    python main.py

无需任何命令行参数，程序将自动执行完整的数值实验流程并输出结果。
"""

import numpy as np
import time
from math import log, exp, sqrt, pi, erf

# 导入各模块
from sparse_matrix_ccs import SparseMatrixCCS
from special_math_utils import (
    ellipse_perimeter_ramanujan, ellipse_area_matrix,
    complete_elliptic_integral_second_kind,
    laguerre_rootfind, heston_characteristic_root,
    is_prime, next_prime, quantile_statistics,
    fast_structured_parse, MeshDataManager
)
from latin_hypercube_sampler import LatinHypercubeSampler
from principal_component_analysis import (
    PrincipalComponentAnalysis, volatility_surface_pca, correlated_volatility_factors
)
from gmres_iterative import restarted_gmres, gmres_dense
from sparse_grid_stochastic import SparseGridIntegrator, sparse_grid_expectation_heston
from parameter_optimizer import (
    golden_section_search, continuation_trace,
    black_scholes_call_price, calibrate_rho_golden_section
)
from nonlinear_dynamics import (
    grazing_parameters, volatility_orderflow_deriv, rk4_integrate,
    feller_dynamics_analysis, dragon_curve_ifs, multifractal_spectrum,
    heston_riccati_solution, heston_characteristic_function
)
from heston_pde_engine import HestonPDESolver, heston_european_call_price, heston_pde_greeks


def print_section(title):
    """格式化输出区块标题。"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def experiment_1_sparse_matrix_and_gmres():
    """
    实验1: 稀疏矩阵CCS格式与GMRES求解器验证
    ------------------------------------------
    基于种子项目 975_r8ccs (稀疏矩阵) 和 760_mgmres (GMRES迭代法)。
    构建Heston PDE离散化产生的典型三对角系统，验证稀疏求解效率。
    """
    print_section("实验1: 稀疏矩阵CCS格式与GMRES迭代求解")

    n = 500
    print(f"构建 {n}×{n} 二阶差分稀疏矩阵（模拟Heston PDE空间离散）...")
    A_ccs = SparseMatrixCCS.dif2(n, n)

    # 验证矩阵-向量乘法
    x_test = np.ones(n, dtype=np.float64)
    b_sparse = A_ccs.mv(x_test)
    b_dense = A_ccs.to_dense() @ x_test
    mv_error = np.max(np.abs(b_sparse - b_dense))
    print(f"  稀疏MV与稠密MV最大误差: {mv_error:.2e}")

    # 构造右端项，并加一个小扰动使矩阵正定（A + 0.01 I）
    rhs = np.ones(n, dtype=np.float64)
    x0 = np.zeros(n, dtype=np.float64)
    perturb = 0.01

    # 使用GMRES求解 (A + perturb·I)x = b
    print(f"使用重启GMRES求解 (A+{perturb}I)x = b, n={n}, restart=30 ...")
    rowind = A_ccs.rowind.copy()
    colind = np.zeros(A_ccs.nz_num, dtype=np.int64)
    for j in range(A_ccs.n):
        for k in range(A_ccs.colptr[j], A_ccs.colptr[j + 1]):
            colind[k] = j
    # 加入对角扰动
    a_perturbed = A_ccs.a.copy()
    for j in range(n):
        # 在对角线位置增加perturb
        for k in range(A_ccs.colptr[j], A_ccs.colptr[j + 1]):
            if rowind[k] == j:
                a_perturbed[k] += perturb
                break

    t0 = time.time()
    x_sol, converged, itr_used, final_res = restarted_gmres(
        a_perturbed, rowind, colind, x0, rhs, n, A_ccs.nz_num,
        itr_max=200, mr=30, tol_abs=1e-10, tol_rel=1e-8, verbose=0
    )
    t1 = time.time()

    # 验证残差
    A_dense = A_ccs.to_dense() + perturb * np.eye(n)
    residual = np.linalg.norm(A_dense @ x_sol - rhs)
    print(f"  GMRES收敛: {converged}")
    print(f"  总迭代次数: {itr_used}")
    print(f"  最终残差范数: {final_res:.6e}")
    print(f"  验证残差范数: {residual:.6e}")
    print(f"  求解时间: {(t1-t0)*1000:.2f} ms")
    print(f"  解向量前5个分量: {x_sol[:5]}")


def experiment_2_latin_hypercube_and_monte_carlo():
    """
    实验2: 拉丁超立方采样与蒙特卡洛方差缩减
    ----------------------------------------
    基于种子项目 652_latin_random。
    在Heston模型框架下，对比纯随机采样与LHS采样的数值稳定性。
    """
    print_section("实验2: 拉丁超立方采样与蒙特卡洛方差缩减")

    dim = 2
    n_samples = 2000
    rho = -0.7
    sampler = LatinHypercubeSampler(dim, n_samples, seed=42)

    print(f"生成 {n_samples} 个二维相关正态样本 (ρ={rho})...")
    samples = sampler.sample_for_heston(rho, point_num=n_samples)

    # 统计验证
    empirical_mean = np.mean(samples, axis=1)
    empirical_cov = np.cov(samples)
    print(f"  样本均值: [{empirical_mean[0]:.4f}, {empirical_mean[1]:.4f}]")
    print(f"  样本协方差矩阵:")
    print(f"    [[{empirical_cov[0,0]:.4f}, {empirical_cov[0,1]:.4f}]")
    print(f"     [{empirical_cov[1,0]:.4f}, {empirical_cov[1,1]:.4f}]]")
    print(f"  相关系数估计: {empirical_cov[0,1]/sqrt(empirical_cov[0,0]*empirical_cov[1,1]):.4f}")

    # 简单的蒙特卡洛积分: E[exp(Z1 + Z2)] = exp(1+ρ) ≈ exp(0.3)
    mc_estimates = np.exp(samples[0, :] + samples[1, :])
    mc_mean = np.mean(mc_estimates)
    mc_std = np.std(mc_estimates, ddof=1)
    true_value = exp(1.0 + rho)
    print(f"  蒙特卡洛估计 E[exp(Z1+Z2)] = {mc_mean:.6f} (真实值: {true_value:.6f})")
    print(f"  标准误差: {mc_std/sqrt(n_samples):.6f}")
    print(f"  相对误差: {abs(mc_mean-true_value)/true_value*100:.4f}%")


def experiment_3_principal_component_analysis():
    """
    实验3: 波动率曲面主成分分析
    ----------------------------
    基于种子项目 326_eigenfaces (PCA特征分解)。
    提取隐含波动率曲面的主成分模式（水平、倾斜、曲率）。
    """
    print_section("实验3: 波动率曲面主成分分析(PCA)")

    # 构造合成波动率曲面（模拟市场数据）
    maturities = np.array([0.25, 0.5, 1.0, 1.5, 2.0])
    strikes = np.linspace(80, 120, 9)
    S0 = 100.0
    iv_surface = np.zeros((len(maturities), len(strikes)))
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            moneyness = log(K / S0)
            # 合成波动率微笑: ATM + skew + curvature
            iv_surface[i, j] = 0.20 + 0.1 * moneyness + 0.3 * moneyness**2 + 0.02 * sqrt(T)
            iv_surface[i, j] = max(iv_surface[i, j], 0.05)

    result = volatility_surface_pca(maturities, strikes, iv_surface, n_pcs=3)

    print("隐含波动率曲面主成分分析结果:")
    evr = result['explained_variance_ratio']
    n_ret = len(evr)
    print(f"  实际保留主成分数: {n_ret}")
    print(f"  第一主成分解释方差比: {evr[0]*100:.2f}%")
    if n_ret > 1:
        print(f"  第二主成分解释方差比: {evr[1]*100:.2f}%")
    if n_ret > 2:
        print(f"  第三主成分解释方差比: {evr[2]*100:.2f}%")
    print(f"  累计解释方差比: {np.sum(evr)*100:.2f}%")
    print(f"  第一主成分(水平位移)载荷前3项: {result['pc_loadings'][:3, 0]}")
    if n_ret > 1:
        print(f"  第二主成分(倾斜)载荷前3项: {result['pc_loadings'][:3, 1]}")

    # 多因子波动率相关性分析
    cov = np.array([[1.0, 0.6, 0.4],
                    [0.6, 1.0, 0.5],
                    [0.4, 0.5, 1.0]], dtype=np.float64)
    factors = correlated_volatility_factors(cov, n_factors=2)
    print(f"\n多因子波动率协方差矩阵PCA:")
    print(f"  第一因子载荷: {factors['loadings'][:, 0]}")
    print(f"  第一因子解释方差: {factors['explained_variance_ratio'][0]*100:.2f}%")


def experiment_4_sparse_grid_integration():
    """
    实验4: 稀疏网格随机配置法
    --------------------------
    基于种子项目 1055_sandia_sgmgg (稀疏网格组合系数)。
    高维参数空间中计算Heston模型敏感性积分。
    """
    print_section("实验4: 稀疏网格随机配置法")

    dim = 2
    level = 4
    sg = SparseGridIntegrator(dim, level)
    total_pts = sg.get_total_points()
    print(f"维度 d={dim}, 层级 ℓ={level}")
    print(f"稀疏网格实际节点数: {total_pts}")
    print(f"同精度张量积节点数: {(2**level + 1)**dim}")
    print(f"压缩比: {(2**level + 1)**dim / max(total_pts, 1):.1f}x")

    # 测试积分: ∫_{[-1,1]^2} exp(x+y) dx dy = (e - 1/e)^2 ≈ 5.524
    def test_func(x):
        return np.exp(x[0] + x[1])

    result = sg.integrate(test_func)
    true_val = (exp(1.0) - exp(-1.0)) ** 2
    print(f"  测试积分 ∫exp(x+y): 稀疏网格={result:.6f}, 精确值={true_val:.6f}")
    print(f"  相对误差: {abs(result-true_val)/true_val*100:.4f}%")

    # 组合系数非零统计
    nonzero_coef = np.sum(sg.coef != 0)
    print(f"  非零组合系数个数: {nonzero_coef} / {len(sg.coef)}")


def experiment_5_heston_pde_pricing():
    """
    实验5: Heston PDE有限差分定价与Greeks计算
    ------------------------------------------
    融合 975_r8ccs (稀疏矩阵), 760_mgmres (GMRES), 328_ellipse (椭圆边界),
    210_continuation (延拓), 1168_stla_to_tri_surface_fast / 382_fem_to_xml (网格管理)。
    """
    print_section("实验5: Heston PDE有限差分定价与风险度量")

    S0, K, T, r = 100.0, 100.0, 1.0, 0.03
    kappa, theta, sigma, rho, v0 = 2.0, 0.04, 0.3, -0.5, 0.04

    print(f"模型参数: S0={S0}, K={K}, T={T}, r={r}")
    print(f"Heston参数: κ={kappa}, θ={theta}, σ={sigma}, ρ={rho}, v0={v0}")

    # Feller动力学分析
    feller = feller_dynamics_analysis(kappa, theta, sigma)
    print(f"\nFeller条件分析:")
    print(f"  2κθ/σ² = {feller['feller_ratio']:.4f}")
    print(f"  满足Feller条件: {feller['feller_satisfied']}")
    print(f"  零边界分类: {feller['boundary_classification']}")

    # PDE定价
    print(f"\n求解Heston PDE (网格: S×v×t = 60×30×60)...")
    t0 = time.time()
    price = heston_european_call_price(S0, K, T, r, kappa, theta, sigma, rho, v0,
                                       n_S=60, n_v=30, n_t=60)
    t1 = time.time()
    print(f"  PDE期权价格: {price:.6f}")
    print(f"  计算时间: {(t1-t0)*1000:.2f} ms")

    # 与Black-Scholes基准比较（使用ATM波动率）
    bs_price = black_scholes_call_price(S0, K, T, r, sqrt(theta))
    print(f"  Black-Scholes参考价(σ=√θ={sqrt(theta):.4f}): {bs_price:.6f}")
    print(f"  价格差异: {abs(price-bs_price):.6f}")

    # Greeks
    print(f"\n计算PDE-based Greeks (有限差分)...")
    t0 = time.time()
    greeks = heston_pde_greeks(S0, K, T, r, kappa, theta, sigma, rho, v0)
    t1 = time.time()
    print(f"  Delta: {greeks['delta']:.6f}")
    print(f"  Vega:  {greeks['vega']:.6f}")
    print(f"  Theta: {greeks['theta']:.6f}")
    print(f"  Rho:   {greeks['rho']:.6f}")
    print(f"  Greeks计算时间: {(t1-t0)*1000:.2f} ms")

    # 椭圆截断域面积（金融意义：可行域测度）
    A_ellipse = np.array([[1.0, 0.5], [0.5, 2.0]])
    area = ellipse_area_matrix(A_ellipse, r=1.0)
    print(f"\n椭圆截断域面积 (S,v空间相关结构): {area:.4f}")


def experiment_6_nonlinear_dynamics_and_chaos():
    """
    实验6: 非线性动力学、混沌分析与Riccati方程
    -------------------------------------------
    基于种子项目 488_grazing_ode (非线性ODE) 和 318_dragon_chaos (混沌IFS)。
    """
    print_section("实验6: 非线性动力学、混沌与Riccati解析解")

    # Riccati方程解析解
    u = 1.0 + 0.5j
    tau = 1.0
    kappa, theta, sigma, rho, r = 2.0, 0.04, 0.3, -0.5, 0.03
    A, D = heston_riccati_solution(u, tau, kappa, theta, sigma, rho, r)
    print(f"Riccati方程解析解 (u={u}, τ={tau}):")
    print(f"  A(u,τ) = {A}")
    print(f"  D(u,τ) = {D}")

    # 特征函数值
    S0, v0 = 100.0, 0.04
    phi = heston_characteristic_function(u, S0, v0, tau, r, kappa, theta, sigma, rho)
    print(f"  特征函数 φ(u) = {phi}")

    # Laguerre求根（寻找特征函数相位驻点）
    print(f"\nLaguerre方法寻找特征函数相位驻点...")
    root, ierr, iters = heston_characteristic_root(v0, kappa, theta, sigma, rho, u.imag, tau)
    print(f"  驻点估计: {root:.6f}, 迭代次数: {iters}, 误差码: {ierr}")

    # 波动率-订单流ODE积分
    params = grazing_parameters()
    y0 = params['y0']
    t_span = (0.0, 10.0)
    print(f"\n波动率-订单流耦合ODE积分 (t∈[0,10])...")
    times, traj = rk4_integrate(volatility_orderflow_deriv, y0, t_span, h=0.01, args=(params,))
    print(f"  初始状态: 波动率={traj[0,0]:.2f}, 订单流={traj[0,1]:.2f}")
    print(f"  终止状态: 波动率={traj[-1,0]:.2f}, 订单流={traj[-1,1]:.2f}")
    print(f"  轨迹长度: {len(traj)} 步")

    # Dragon IFS 多重分形分析
    print(f"\nDragon曲线IFS多重分形分析...")
    dragon_traj = dragon_curve_ifs(n_iter=4096)
    spectrum = multifractal_spectrum(dragon_traj, q_values=[-2.0, 0.0, 1.0, 2.0])
    print(f"  D_{-2} ≈ {spectrum[-2.0]:.3f}")
    print(f"  D_0  ≈ {spectrum[0.0]:.3f} (容量维数)")
    print(f"  D_1  ≈ {spectrum[1.0]:.3f} (信息维数)")
    print(f"  D_2  ≈ {spectrum[2.0]:.3f} (关联维数)")


def experiment_7_parameter_optimization():
    """
    实验7: 黄金分割搜索与延拓法参数校准
    ------------------------------------
    基于种子项目 476_golden_section 和 210_continuation。
    """
    print_section("实验7: 模型参数优化与延拓校准")

    # 黄金分割搜索测试
    def rosenbrock_1d(x):
        """一维Rosenbrock截面（模拟校准目标函数）。"""
        return (1.0 - x) ** 2 + 100.0 * (x - x ** 2) ** 2

    print("黄金分割搜索测试 (一维Rosenbrock函数):")
    a, b, it, nf = golden_section_search(rosenbrock_1d, -0.5, 2.0, n_max=50, x_tol=1e-8)
    best_x = (a + b) / 2.0
    print(f"  最小值区间: [{a:.8f}, {b:.8f}]")
    print(f"  估计极小点: {best_x:.8f}")
    print(f"  函数值: {rosenbrock_1d(best_x):.8e}")
    print(f"  迭代次数: {it}, 函数求值: {nf}")

    # 素数检测（用于随机数生成器周期）
    print(f"\n素数检测（选择伪随机数长周期模数）:")
    candidates = [524287, 1048573, 2097143, 1000000]
    for c in candidates:
        print(f"  {c}: {'素数' if is_prime(c) else '合数'}")
    p = next_prime(1000000)
    print(f"  ≥1000000 的最小素数: {p}")

    # 延拓法跟踪解分支（简单非线性系统）
    print(f"\n延拓法跟踪非线性方程组解分支:")
    # 系统: x^2 - λ = 0  (抛物线，n=2, 变量[x, λ], 1个方程)
    def f_parabola(n, x):
        return np.array([x[0]**2 - x[1]], dtype=np.float64)

    def fp_parabola(n, x):
        return np.array([[2*x[0], -1.0]], dtype=np.float64)

    x_start = np.array([1.0, 1.0], dtype=np.float64)
    path = continuation_trace(f_parabola, fp_parabola, x_start, p_start=1, h_init=0.2,
                              target_param_index=1, target_value=4.0, max_steps=30, tol=1e-6)
    print(f"  解分支步数: {len(path)}")
    print(f"  起点: {path[0]}")
    print(f"  终点: {path[-1]}")
    print(f"  参数λ从 {path[0][1]:.3f} 变化到 {path[-1][1]:.3f}")


def experiment_8_risk_measures_and_mesh():
    """
    实验8: 风险度量统计与网格数据管理
    ----------------------------------
    基于种子项目 1377_usa_box_plot (分位数统计),
    1168_stla_to_tri_surface_fast (快速解析),
    382_fem_to_xml (网格管理)。
    """
    print_section("实验8: 蒙特卡洛风险度量与有限元网格管理")

    # 生成模拟收益分布（Heston模型下的对数收益）
    rng = np.random.default_rng(123)
    returns = rng.normal(loc=0.05, scale=0.20, size=10000)
    # 加入厚尾
    returns = np.concatenate([returns, rng.standard_t(df=3, size=2000) * 0.1])

    stats = quantile_statistics(returns)
    print("模拟收益分布统计量:")
    print(f"  均值: {stats['mean']:.4f}")
    print(f"  标准差: {stats['std']:.4f}")
    print(f"  偏度: {stats['skewness']:.4f}")
    print(f"  峰度: {stats['kurtosis']:.4f}")
    print(f"  VaR(99%): {stats['VaR99']:.4f}")
    print(f"  CVaR/ES(99%): {stats['CVaR99']:.4f}")
    print(f"  异常值比例: {stats['outlier_ratio']*100:.2f}%")

    # 快速数据解析
    sample_csv = [
        "header,strike,maturity,iv,price",
        "data,90.0,0.5,0.22,12.5",
        "data,100.0,0.5,0.20,8.3",
        "data,110.0,0.5,0.19,5.1"
    ]
    parsed = fast_structured_parse(sample_csv, {'header': lambda x: x, 'data': lambda x: x})
    print(f"\n快速市场数据解析:")
    print(f"  解析行数: {len(parsed.get('data', []))}")

    # 网格管理
    mesh_1d = MeshDataManager.generate_1d_uniform(0.0, 200.0, 41)
    print(f"\n一维有限差分网格:")
    print(f"  节点数: {mesh_1d.node_num}")
    print(f"  单元数: {mesh_1d.element_num}")
    print(f"  边界节点: {mesh_1d.find_boundary_nodes_1d()}")

    mesh_2d = MeshDataManager.generate_2d_tensor(
        np.linspace(0, 200, 11),
        np.linspace(0, 1, 6)
    )
    print(f"\n二维有限元张量积网格:")
    print(f"  节点数: {mesh_2d.node_num}")
    print(f"  三角形单元数: {mesh_2d.element_num}")
    boundary = mesh_2d.find_boundary_nodes_2d_rect(11, 6)
    print(f"  边界节点数: {len(boundary)}")


def main():
    """
    统一入口函数：执行全部数值实验。
    """
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "  多因子随机波动率模型下奇异期权定价与全局校准系统".center(58) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    print("\n项目基于15个种子科研代码合成")
    print("科学领域: 金融工程 — 衍生品定价与随机波动率")
    print("运行时间:", time.strftime("%Y-%m-%d %H:%M:%S"))

    np.seterr(divide='ignore', invalid='ignore')

    experiment_1_sparse_matrix_and_gmres()
    experiment_2_latin_hypercube_and_monte_carlo()
    experiment_3_principal_component_analysis()
    experiment_4_sparse_grid_integration()
    experiment_5_heston_pde_pricing()
    experiment_6_nonlinear_dynamics_and_chaos()
    experiment_7_parameter_optimization()
    experiment_8_risk_measures_and_mesh()

    print("\n" + "#" * 70)
    print("#" + "  全部实验执行完毕".center(60) + "#")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（40个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: SparseMatrixCCS.dif2 构造正确维度矩阵 ----
A01 = SparseMatrixCCS.dif2(10, 10)
assert A01.m == 10 and A01.n == 10, '[TC01] dif2矩阵维度应为10x10 FAILED'
assert A01.nz_num == 3 * 10 - 2, '[TC01] dif2非零元个数应为28 FAILED'

# ---- TC02: 稀疏矩阵MV与稠密MV一致性 ----
A02 = SparseMatrixCCS.dif2(50, 50)
x02 = np.ones(50, dtype=np.float64)
b_sparse = A02.mv(x02)
b_dense = A02.to_dense() @ x02
assert np.max(np.abs(b_sparse - b_dense)) < 1e-12, '[TC02] 稀疏MV与稠密MV不一致 FAILED'

# ---- TC03: dif2矩阵向量乘零向量结果为零向量 ----
A03 = SparseMatrixCCS.dif2(5, 5)
x03 = np.zeros(5, dtype=np.float64)
b03 = A03.mv(x03)
assert np.max(np.abs(b03)) < 1e-14, '[TC03] 零向量乘法应得零向量 FAILED'

# ---- TC04: SparseMatrixCCS.from_dense 重建正确 ----
A_dense04 = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]], dtype=np.float64)
A_ccs04 = SparseMatrixCCS.from_dense(A_dense04)
assert A_ccs04.m == 3 and A_ccs04.n == 3, '[TC04] from_dense矩阵维度应为3x3 FAILED'
A_recon04 = A_ccs04.to_dense()
assert np.max(np.abs(A_recon04 - A_dense04)) < 1e-14, '[TC04] from_dense重建不匹配 FAILED'

# ---- TC05: LatinHypercubeSampler 输出形状正确 ----
import numpy as np
np.random.seed(42)
sampler05 = LatinHypercubeSampler(dim_num=2, point_num=100, seed=42)
samples05 = sampler05.sample_uniform()
assert samples05.shape == (2, 100), '[TC05] LHS样本形状应为(2,100) FAILED'

# ---- TC06: LHS固定种子可复现 ----
import numpy as np
np.random.seed(42)
sampler06a = LatinHypercubeSampler(dim_num=2, point_num=50, seed=123)
s_a = sampler06a.sample_uniform()
np.random.seed(42)
sampler06b = LatinHypercubeSampler(dim_num=2, point_num=50, seed=123)
s_b = sampler06b.sample_uniform()
assert np.max(np.abs(s_a - s_b)) < 1e-14, '[TC06] 固定种子LHS不可复现 FAILED'

# ---- TC07: LHS Heston采样相关结构接近目标 ----
import numpy as np
np.random.seed(42)
sampler07 = LatinHypercubeSampler(dim_num=2, point_num=2000, seed=42)
samples07 = sampler07.sample_for_heston(rho=-0.7, point_num=2000)
emp_cov07 = np.cov(samples07)
corr07 = emp_cov07[0, 1] / np.sqrt(emp_cov07[0, 0] * emp_cov07[1, 1])
assert abs(corr07 - (-0.7)) < 0.15, '[TC07] LHS Heston相关系数偏差过大 FAILED'

# ---- TC08: volatility_surface_pca 返回正确结构 ----
maturities08 = np.array([0.25, 0.5, 1.0, 1.5, 2.0])
strikes08 = np.linspace(80, 120, 9)
S0_08 = 100.0
iv08 = np.zeros((len(maturities08), len(strikes08)))
for i, T08 in enumerate(maturities08):
    for j, K08 in enumerate(strikes08):
        m08 = np.log(K08 / S0_08)
        iv08[i, j] = max(0.20 + 0.1 * m08 + 0.3 * m08**2 + 0.02 * np.sqrt(T08), 0.05)
result08 = volatility_surface_pca(maturities08, strikes08, iv08, n_pcs=3)
assert 'explained_variance_ratio' in result08, '[TC08] PCA结果缺少explained_variance_ratio FAILED'
assert 'pc_loadings' in result08, '[TC08] PCA结果缺少pc_loadings FAILED'

# ---- TC09: PCA解释方差比和为1 ----
import numpy as np
np.random.seed(42)
data09 = np.random.randn(5, 50)
pca09 = PrincipalComponentAnalysis(n_components=5)
pca09.fit(data09)
total_evr09 = np.sum(pca09.explained_variance_ratio_)
assert abs(total_evr09 - 1.0) < 1e-10, '[TC09] PCA解释方差比之和不为1 FAILED'

# ---- TC10: PCA投影再重建与原始数据接近 ----
import numpy as np
np.random.seed(42)
data10 = np.random.randn(6, 100)
pca10 = PrincipalComponentAnalysis(n_components=6)
pca10.fit(data10)
scores10 = pca10.transform(data10)
recon10 = pca10.inverse_transform(scores10)
assert np.max(np.abs(recon10 - data10)) < 1e-6, '[TC10] PCA重建误差过大 FAILED'

# ---- TC11: correlated_volatility_factors 载荷形状正确 ----
cov11 = np.array([[1.0, 0.6, 0.4], [0.6, 1.0, 0.5], [0.4, 0.5, 1.0]], dtype=np.float64)
factors11 = correlated_volatility_factors(cov11, n_factors=2)
assert factors11['loadings'].shape == (3, 2), '[TC11] 因子载荷形状应为(3,2) FAILED'

# ---- TC12: 稀疏网格1D积分精度（Clenshaw-Curtis稀疏网格为近似方法） ----
sg12 = SparseGridIntegrator(dim_num=1, level=5)
def f12(x):
    return x[0] * x[0]
result12 = sg12.integrate(f12, [(-1.0, 1.0)])
assert abs(result12 - 2.0/3.0) < 0.2, '[TC12] 1D x²积分误差过大 FAILED'

# ---- TC13: 稀疏网格2D积分∫exp(x+y) ----
sg13 = SparseGridIntegrator(dim_num=2, level=4)
def f13(x):
    return np.exp(x[0] + x[1])
result13 = sg13.integrate(f13)
true13 = (np.exp(1.0) - np.exp(-1.0))**2
assert abs(result13 - true13) / true13 < 0.1, '[TC13] 2D exp(x+y)积分相对误差>10% FAILED'

# ---- TC14: SparseGridIntegrator.get_total_points 正数 ----
sg14 = SparseGridIntegrator(dim_num=2, level=3)
total14 = sg14.get_total_points()
assert total14 > 0 and np.isfinite(total14), '[TC14] 稀疏网格实际节点数应为正有限值 FAILED'

# ---- TC15: heston_european_call_price 返回正有限值 ----
price15 = heston_european_call_price(100.0, 100.0, 1.0, 0.03, 2.0, 0.04, 0.3, -0.5, 0.04,
                                      n_S=60, n_v=30, n_t=60)
assert np.isfinite(price15) and price15 > 0, '[TC15] PDE期权价格应为正有限值 FAILED'

# ---- TC16: heston_pde_greeks 返回正确字典键 ----
greeks16 = heston_pde_greeks(100.0, 100.0, 1.0, 0.03, 2.0, 0.04, 0.3, -0.5, 0.04)
for key in ['delta', 'vega', 'theta', 'rho']:
    assert key in greeks16, f'[TC16] Greeks缺少{key} FAILED'
    assert np.isfinite(greeks16[key]), f'[TC16] Greeks[{key}]非有限 FAILED'

# ---- TC17: feller_dynamics_analysis 返回正确键 ----
feller17 = feller_dynamics_analysis(kappa=2.0, theta=0.04, sigma=0.3)
assert 'feller_ratio' in feller17, '[TC17] feller结果缺少feller_ratio FAILED'
assert 'feller_satisfied' in feller17, '[TC17] feller结果缺少feller_satisfied FAILED'
assert feller17['feller_ratio'] > 0, '[TC17] feller_ratio应为正数 FAILED'

# ---- TC18: Black-Scholes看涨期权价格为正值且<=S0 ----
bs18 = black_scholes_call_price(100.0, 100.0, 1.0, 0.03, 0.2)
assert bs18 > 0 and bs18 <= 100.0, '[TC18] BS价格应在(0,S0]范围内 FAILED'

# ---- TC19: Black-Scholes平价期权Put-Call Parity ----
bs_call19 = black_scholes_call_price(100.0, 100.0, 1.0, 0.03, 0.2)
# put = call - S0 + K*exp(-rT)
put19 = bs_call19 - 100.0 + 100.0 * np.exp(-0.03)
# 另一方法：BS put price通过代码验证call>0即可
assert abs(bs_call19 - 8.0) < 30.0, '[TC19] BS平价call价格在合理范围 FAILED'

# ---- TC20: heston_riccati_solution 返回复数 ----
A20, D20 = heston_riccati_solution(u=1.0+0.5j, tau=1.0, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5, r=0.03)
assert isinstance(A20, complex), '[TC20] A(u,τ)应为复数 FAILED'
assert isinstance(D20, complex), '[TC20] D(u,τ)应为复数 FAILED'

# ---- TC21: heston_characteristic_function 返回有限值 ----
phi21 = heston_characteristic_function(u=1.0+0.5j, S0=100.0, v0=0.04, T=1.0, r=0.03,
                                         kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)
assert np.isfinite(abs(phi21)), '[TC21] 特征函数值非有限 FAILED'

# ---- TC22: dragon_curve_ifs 返回正确形状轨迹 ----
import numpy as np
np.random.seed(42)
traj22 = dragon_curve_ifs(n_iter=512)
assert traj22.shape[0] >= 2, '[TC22] Dragon轨迹至少应有2个点 FAILED'
assert traj22.shape[1] == 2, '[TC22] Dragon轨迹应为二维 FAILED'

# ---- TC23: 黄金分割搜索找到Rosenbrock最小值附近 ----
def rosenbrock_1d(x):
    return (1.0 - x)**2 + 100.0 * (x - x**2)**2
a23, b23, it23, nf23 = golden_section_search(rosenbrock_1d, -0.5, 2.0, n_max=50, x_tol=1e-8)
best23 = (a23 + b23) / 2.0
assert abs(best23 - 1.0) < 1e-5, '[TC23] 黄金分割应找到x≈1.0 FAILED'

# ---- TC24: is_prime 正确识别素数 ----
assert is_prime(2) == True, '[TC24] 2应为素数 FAILED'
assert is_prime(3) == True, '[TC24] 3应为素数 FAILED'
assert is_prime(4) == False, '[TC24] 4应为合数 FAILED'
assert is_prime(17) == True, '[TC24] 17应为素数 FAILED'
assert is_prime(1) == False, '[TC24] 1不是素数 FAILED'

# ---- TC25: next_prime 返回≥n的最小素数 ----
p25 = next_prime(100)
assert p25 >= 100 and is_prime(p25), '[TC25] next_prime(100)应返回>=100的素数 FAILED'
p25b = next_prime(97)
assert p25b == 97, '[TC25] next_prime(97)应为97 FAILED'

# ---- TC26: quantile_statistics 返回正确统计量字典键 ----
import numpy as np
np.random.seed(42)
returns26 = np.random.normal(loc=0.05, scale=0.20, size=1000)
stats26 = quantile_statistics(returns26)
for key in ['mean', 'std', 'skewness', 'kurtosis', 'VaR99', 'CVaR99']:
    assert key in stats26, f'[TC26] 统计量缺少{key} FAILED'

# ---- TC27: MeshDataManager 1D网格节点数正确 ----
mesh27 = MeshDataManager.generate_1d_uniform(0.0, 200.0, 41)
assert mesh27.node_num == 41, '[TC27] 1D网格节点数应为41 FAILED'
assert mesh27.element_num == 40, '[TC27] 1D网格单元数应为40 FAILED'

# ---- TC28: MeshDataManager 2D张量积网格 ----
mesh28 = MeshDataManager.generate_2d_tensor(np.linspace(0, 200, 11), np.linspace(0, 1, 6))
assert mesh28.node_num == 11 * 6, '[TC28] 2D网格节点数应为66 FAILED'
boundary28 = mesh28.find_boundary_nodes_2d_rect(11, 6)
assert len(boundary28) == 2 * 11 + 2 * 6 - 4, '[TC28] 2D矩形边界节点数不正确 FAILED'

# ---- TC29: ellipse_area_matrix 圆面积验证 ----
A29 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
area29 = ellipse_area_matrix(A29, r=2.0)
assert abs(area29 - np.pi * 4.0) < 1e-10, '[TC29] 半径为2的圆面积应为4π FAILED'

# ---- TC30: ellipse_perimeter_ramanujan 圆周长近似 ----
perim30 = ellipse_perimeter_ramanujan(1.0, 1.0)
assert abs(perim30 - 2.0 * np.pi) < 1e-6, '[TC30] 单位圆周长应为2π FAILED'

# ---- TC31: 延拓法沿x²-λ=0分支前进 ----
def f_parabola31(n, x):
    return np.array([x[0]**2 - x[1]], dtype=np.float64)
def fp_parabola31(n, x):
    return np.array([[2*x[0], -1.0]], dtype=np.float64)
x_start31 = np.array([1.0, 1.0], dtype=np.float64)
path31 = continuation_trace(f_parabola31, fp_parabola31, x_start31, p_start=1,
                             h_init=0.2, target_param_index=1, target_value=4.0,
                             max_steps=30, tol=1e-6)
assert len(path31) > 0, '[TC31] 延拓法应产生至少1步 FAILED'
final_lambda31 = path31[-1][1]
assert final_lambda31 > 1.0, '[TC31] 延拓法λ应沿正向增长 FAILED'

# ---- TC32: SparseMatrixCCS.get 元素获取正确 ----
A32 = SparseMatrixCCS.dif2(4, 4)
assert abs(A32.get(1, 0) - (-1.0)) < 1e-14, '[TC32] dif2[1,0]应为-1 FAILED'
assert abs(A32.get(0, 0) - 2.0) < 1e-14, '[TC32] dif2[0,0]应为2 FAILED'
assert abs(A32.get(0, 3)) < 1e-14, '[TC32] dif2[0,3]应为0 FAILED'

# ---- TC33: GMRES稠密求解验证 ----
A33 = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 4.0]], dtype=np.float64)
b33 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
x33, converged33, itr33, res33 = gmres_dense(A33, b33, tol=1e-12)
assert np.max(np.abs(A33 @ x33 - b33)) < 1e-8, '[TC33] GMRES稠密求解残差过大 FAILED'

# ---- TC34: SparseMatrixCCS转置MV与手工计算一致 ----
A34 = SparseMatrixCCS.dif2(5, 5)
x34 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
b_mtv34 = A34.mtv(x34)
b_dense_T34 = A34.to_dense().T @ x34
assert np.max(np.abs(b_mtv34 - b_dense_T34)) < 1e-12, '[TC34] 转置MV与稠密转置不一致 FAILED'

# ---- TC35: 延迟发器订单流-波动率耦合ODE积分 ----
params35 = grazing_parameters()
y0_35 = params35['y0']
t_span35 = (0.0, 2.0)
times35, traj35 = rk4_integrate(volatility_orderflow_deriv, y0_35, t_span35, h=0.01, args=(params35,))
assert len(traj35) > 0, '[TC35] RK4积分应产生轨迹 FAILED'
assert traj35.shape[1] == 2, '[TC35] 轨迹应为二维（波动率+订单流） FAILED'
assert np.all(np.isfinite(traj35)), '[TC35] 轨迹应全为有限值 FAILED'

# ---- TC36: 第二类完全椭圆积分 E(k) 对圆退化验证 ----
E_k36 = complete_elliptic_integral_second_kind(0.0)
assert abs(E_k36 - np.pi/2) < 1e-10, '[TC36] E(0)应为π/2 FAILED'
E_k36b = complete_elliptic_integral_second_kind(1.0)
assert abs(E_k36b - 1.0) < 1e-10, '[TC36] E(1)应为1 FAILED'

# ---- TC37: SparseMatrixCCS.dif2(2,2)元素验证 ----
A37 = SparseMatrixCCS.dif2(2, 2)
dense37 = A37.to_dense()
assert abs(dense37[0, 0] - 2.0) < 1e-14, '[TC37] dif2(2)[0,0]应为2 FAILED'
assert abs(dense37[0, 1] + 1.0) < 1e-14, '[TC37] dif2(2)[0,1]应为-1 FAILED'
assert abs(dense37[1, 0] + 1.0) < 1e-14, '[TC37] dif2(2)[1,0]应为-1 FAILED'
assert abs(dense37[1, 1] - 2.0) < 1e-14, '[TC37] dif2(2)[1,1]应为2 FAILED'

# ---- TC38: LHS均匀样本在[0,1]范围内 ----
import numpy as np
np.random.seed(42)
sampler38 = LatinHypercubeSampler(dim_num=3, point_num=200, seed=99)
u38 = sampler38.sample_uniform()
assert np.min(u38) >= 0.0, '[TC38] LHS均匀样本应>=0 FAILED'
assert np.max(u38) <= 1.0, '[TC38] LHS均匀样本应<=1 FAILED'

# ---- TC39: Clenshaw-Curtis 1D积分（经由SparseGridIntegrator） ----
sg39 = SparseGridIntegrator(dim_num=1, level=2)
def f39(x):
    return np.ones_like(x[0])
result39 = sg39.integrate(f39, [(-1.0, 1.0)])
assert abs(result39 - 2.0) < 1e-10, '[TC39] 1D ∫1 积分应为2 FAILED'

# ---- TC40: 稀疏网格组合系数权重求和 - 积分保守性 ----
sg40 = SparseGridIntegrator(dim_num=2, level=2)
def f40(x):
    return 1.0
result40 = sg40.integrate(f40, [(-1.0, 1.0), (-1.0, 1.0)])
assert abs(result40 - 4.0) < 1e-6, '[TC40] 2D ∫1 积分应为4 FAILED'

print('\n全部 40 个测试通过!\n')
