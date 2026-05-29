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

    # TODO: 实现Heston PDE定价实验的核心调用链
    # 需要完成:
    #   1. 调用 heston_european_call_price 计算PDE欧式看涨期权价格
    #      参数: S0=100, K=100, T=1.0, r=0.03, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5, v0=0.04
    #      网格: n_S=60, n_v=30, n_t=60
    #   2. 调用 black_scholes_call_price 计算Black-Scholes基准价格（sigma=sqrt(theta)）
    #   3. 计算并输出价格差异 abs(price - bs_price)
    #   4. 调用 heston_pde_greeks 计算PDE-based Greeks (delta, vega, theta, rho)
    #   5. 计时并格式化输出所有结果
    raise NotImplementedError("Hole_3: 需要实现experiment_5的PDE定价调用链与结果展示")

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
