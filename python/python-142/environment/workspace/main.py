"""
main.py
信用风险违约相关性建模 —— 统一入口

项目标题: 空间-时序多主体信用风险违约相关性 PDE-Copula-网络混合框架
(Spatial-Temporal Multi-Name Credit Risk Default Correlation Modeling
 via PDE-Copula-Network Hybrid Framework)

运行方式: python main.py (零参数)
"""

import numpy as np
import sys
import os

# 将当前目录加入路径，确保模块导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from factor_orthogonalization import (
    orthogonalize_credit_factors,
    factor_covariance_from_loadings,
    modified_gram_schmidt
)
from lattice_enumeration import (
    portfolio_weight_grid,
    correlation_simplex_grid
)
from linear_solver_cgne import (
    cg_ne_solve,
    cg_ne_solve_with_regularization,
    helmert_matrix
)
from interpolation_surfaces import (
    interpolate_credit_curve,
    clenshaw_curtis_nodes,
    arc_length_parameterize
)
from spherical_correlation_grid import (
    build_regional_default_correlation,
    sphere_llt_grid_points,
    spherical_voronoi_areas,
    voronoi_neighbor_adjacency
)
from default_contagion_ode import (
    simulate_default_contagion,
    network_cascade_intensity
)
from structural_default_bvp import (
    structural_default_probability_density,
    default_probability_from_structural,
    solve_ill_bvp_fd
)
from copula_quadrature import (
    integrate_standard_triangle,
    cauchy_principal_value,
    gaussian_copula_bivariate_integral
)
from portfolio_knapsack import (
    credit_portfolio_optimization,
    generate_credit_portfolio_data
)
from utils import (
    normal_cdf,
    nearest_correlation_matrix,
    cholesky_with_pivot
)


def run_factor_orthogonalization_module():
    """
    模块 1: 宏观经济因子正交化
    基于 480_gram_schmidt
    """
    print("=" * 60)
    print("模块 1: 宏观经济因子正交化 (Gram-Schmidt)")
    print("=" * 60)
    np.random.seed(42)
    n_assets = 30
    n_factors = 8

    # 生成具有多重共线的原始因子载荷
    base = np.random.randn(n_assets, 3)
    raw_loadings = np.hstack([base + 0.05 * np.random.randn(n_assets, 3) for _ in range(3)])
    raw_loadings = raw_loadings[:, :n_factors]
    raw_loadings = raw_loadings / (np.linalg.norm(raw_loadings, axis=1, keepdims=True) + 1e-12) * 0.7

    orth_loadings = orthogonalize_credit_factors(raw_loadings, method="mgs")
    corr_matrix = factor_covariance_from_loadings(orth_loadings, np.ones(orth_loadings.shape[1]))

    eigvals = np.linalg.eigvalsh(corr_matrix)
    print(f"原始载荷维度: {raw_loadings.shape}")
    print(f"正交化后秩: {orth_loadings.shape[1]}")
    print(f"相关性矩阵最小特征值: {eigvals.min():.6f}")
    print(f"相关性矩阵条件数: {np.linalg.cond(corr_matrix):.2f}")
    print("模块 1 完成.\n")
    return orth_loadings, corr_matrix


def run_lattice_enumeration_module():
    """
    模块 2: 单形格点枚举与权重网格生成
    基于 054_asa299
    """
    print("=" * 60)
    print("模块 2: 单形格点枚举 (Simplex Lattice Enumeration)")
    print("=" * 60)
    n_assets = 4
    n_grid = 6

    weights = portfolio_weight_grid(n_assets, n_grid)
    print(f"生成 {weights.shape[0]} 个权重配置 (n={n_assets}, t={n_grid})")
    print(f"示例权重 (前 3 个):")
    for i in range(min(3, weights.shape[0])):
        print(f"  w_{i+1} = {weights[i]}")
    print(f"权重和检查: max_dev={np.max(np.abs(weights.sum(axis=1) - 1.0)):.2e}")

    # 相关性特征值网格
    ev_grids = correlation_simplex_grid(3, 4)
    print(f"生成 {len(ev_grids)} 个特征值配置用于相关性矩阵敏感性分析")
    print("模块 2 完成.\n")
    return weights


def run_linear_solver_module():
    """
    模块 3: 共轭梯度法求解法方程 (隐含相关性校准)
    基于 151_cg_ne
    """
    print("=" * 60)
    print("模块 3: CG-NE 隐含相关性校准")
    print("=" * 60)
    np.random.seed(123)
    n_obs = 50
    n_params = 10

    # 模拟 Jacobi 矩阵 (市场价格对相关性参数的敏感性)
    A = np.random.randn(n_obs, n_params)
    x_true = np.random.randn(n_params)
    b = A @ x_true + 0.01 * np.random.randn(n_obs)

    x_sol, iters, res = cg_ne_solve(A, b, tol=1e-10)
    print(f"CG-NE 迭代次数: {iters}")
    print(f"残差范数: {res:.2e}")
    print(f"解误差: {np.linalg.norm(x_sol - x_true):.2e}")

    # 正则化版本 (病态情形)
    x_reg, iters_reg, res_reg = cg_ne_solve_with_regularization(A, b, lam=1e-4, tol=1e-10)
    print(f"正则化 CG-NE 迭代次数: {iters_reg}")
    print(f"正则化解误差: {np.linalg.norm(x_reg - x_true):.2e}")
    print("模块 3 完成.\n")
    return x_sol


def run_interpolation_module():
    """
    模块 4: 信用曲线插值
    基于 590_interp
    """
    print("=" * 60)
    print("模块 4: 信用曲线期限结构插值")
    print("=" * 60)
    maturities = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    pd_curve = np.array([0.005, 0.012, 0.025, 0.038, 0.055, 0.072, 0.095])

    eval_maturities = np.linspace(0.5, 10.0, 20)
    pd_interp = interpolate_credit_curve(maturities, pd_curve, eval_maturities, method="linear")

    print(f"观测期限: {maturities}")
    print(f"观测 PD:  {pd_curve}")
    print(f"插值后 5 年期 PD: {pd_interp[10]:.4f}")
    print(f"插值后 10 年期 PD: {pd_interp[-1]:.4f}")

    # Clenshaw-Curtis 节点 (用于快速积分变换)
    cc_nodes = clenshaw_curtis_nodes(16, a=0.0, b=10.0)
    print(f"CC 节点范围: [{cc_nodes.min():.2f}, {cc_nodes.max():.2f}]")
    print("模块 4 完成.\n")
    return pd_interp


def run_spherical_correlation_module():
    """
    模块 5: 球面 Voronoi 区域违约相关性
    基于 1123_sphere_llt_grid, 1131_sphere_voronoi, 1397_voronoi_neighbors, 725_matlab_map
    """
    print("=" * 60)
    print("模块 5: 球面 Voronoi 区域违约相关性建模")
    print("=" * 60)
    xyz, areas, adj, corr = build_regional_default_correlation(n_regions=14, base_correlation=0.35)
    n_reg = xyz.shape[0]
    print(f"实际区域数: {n_reg}")
    print(f"区域面积权重范围: [{areas.min():.4f}, {areas.max():.4f}]")
    print(f"邻接关系数: {np.sum(adj)//2}")
    print(f"相关性矩阵特征值范围: [{np.linalg.eigvalsh(corr).min():.4f}, {np.linalg.eigvalsh(corr).max():.4f}]")

    # 使用 Cholesky 分解生成相关随机变量
    L = cholesky_with_pivot(corr)
    z = np.random.randn(n_reg)
    correlated_defaults = L @ z
    print(f"生成相关违约冲击样本均值: {correlated_defaults.mean():.4f}, 标准差: {correlated_defaults.std():.4f}")
    print("模块 5 完成.\n")
    return xyz, areas, adj, corr


def run_contagion_ode_module():
    """
    模块 6: 违约传染 ODE 动态模拟
    基于 838_oregonator_ode, 1259_theta_method
    """
    print("=" * 60)
    print("模块 6: 违约传染 ODE 动态模拟 (Oregonator-Theta)")
    print("=" * 60)
    t, y = simulate_default_contagion(
        initial_default_rate=0.02,
        initial_pressure=0.05,
        initial_buffer=0.6,
        t_max=5.0,
        n_steps=400,
        theta=0.5,
        params={"eta1": 0.02, "eta2": 0.05, "q": 0.02, "f": 0.8}
    )
    print(f"模拟时间范围: [0, {t[-1]:.2f}] 年")
    print(f"初始违约强度 u(0): {y[0,0]:.4f}")
    print(f"终期违约强度 u(T): {y[-1,0]:.4f}")
    print(f"终期传染压力 v(T): {y[-1,1]:.4f}")
    print(f"终期缓冲水平 w(T): {y[-1,2]:.4f}")

    # 网络级联效应
    adj = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])
    local = np.array([0.05, 0.12, 0.08, 0.15])
    net_intensity = network_cascade_intensity(adj, local, 0.15)
    print(f"局部违约强度: {local}")
    print(f"网络增强强度: {net_intensity}")
    print("模块 6 完成.\n")
    return t, y


def run_structural_bvp_module():
    """
    模块 7: 结构模型病态 BVP 求解
    基于 572_ill_bvp, 359_fd1d_display
    """
    print("=" * 60)
    print("模块 7: 结构模型病态 BVP 与违约概率密度")
    print("=" * 60)
    # 求解病态 BVP
    x_bvp, y_bvp = solve_ill_bvp_fd(epsilon=0.005, n_nodes=150)
    print(f"BVP 网格点数: {len(x_bvp)}")
    print(f"左边界 y(-1) = {y_bvp[0]:.4f} (目标 2.0)")
    print(f"右边界 y(1)  = {y_bvp[-1]:.4f} (目标 1.0)")

    # 资产价值稳态密度
    v_grid = np.linspace(25, 150, 80)
    pdf = structural_default_probability_density(
        v_grid, mu=0.05, sigma=0.25, v_min=20.0, v_max=160.0, default_barrier=30.0
    )
    print(f"资产价值密度积分: {np.trapezoid(pdf, v_grid):.4f}")

    # 解析违约概率对比
    pd_5y = default_probability_from_structural(100.0, 0.05, 0.25, 30.0, 5.0)
    pd_10y = default_probability_from_structural(100.0, 0.05, 0.25, 30.0, 10.0)
    print(f"Merton 模型 5 年 PD: {pd_5y:.4f}")
    print(f"Merton 模型 10 年 PD: {pd_10y:.4f}")
    print("模块 7 完成.\n")
    return pdf


def run_copula_quadrature_module():
    """
    模块 8: Copula 积分与 Cauchy 主值
    基于 1311_triangle_lyness_rule, 139_cauchy_principal_value
    """
    print("=" * 60)
    print("模块 8: Copula 积分与 Cauchy 主值积分")
    print("=" * 60)
    # 三角形积分测试
    f_area = lambda x, y: 1.0
    tri_area = integrate_standard_triangle(f_area, rule=7)
    print(f"标准三角形面积 (应为 0.5): {tri_area:.8f}")

    # Cauchy 主值
    cpv_val = cauchy_principal_value(lambda t: t**2, -1.0, 2.0, 0.5, n=64)
    print(f"CPV int_{{-1}}^2 t^2/(t-0.5) dt = {cpv_val:.6f}")

    # 二元高斯 Copula 积分
    try:
        cop_val = gaussian_copula_bivariate_integral(0.3, 0.7, 0.5, n_quad=32)
        print(f"高斯 Copula C(0.3, 0.7; rho=0.5) = {cop_val:.6f}")
    except Exception as e:
        print(f"Copula 积分计算跳过: {e}")
    print("模块 8 完成.\n")
    return tri_area


def run_portfolio_optimization_module():
    """
    模块 9: 信用组合优化 (0/1 背包)
    基于 628_knapsack_values
    """
    print("=" * 60)
    print("模块 9: 信用组合优化 (0/1 Knapsack)")
    print("=" * 60)
    data = generate_credit_portfolio_data(n_assets=20, seed=42)
    sel, metrics = credit_portfolio_optimization(
        data["expected_returns"],
        data["capital_charges"],
        data["pd_values"],
        data["lgd_values"],
        data["ead_values"],
        total_capital=800.0,
        var_limit=3000.0,
        resolution=800
    )
    n_selected = np.sum(sel)
    print(f"选中资产数: {n_selected} / {len(sel)}")
    print(f"组合总收益: {metrics['total_return']:.2f}")
    print(f"组合资本占用: {metrics['total_capital']:.2f} (上限 800)")
    print(f"组合风险暴露: {metrics['total_risk']:.2f} (上限 3000)")
    print(f"风险调整资本收益率 RAROC: {metrics['raroc']:.4f}")
    print("模块 9 完成.\n")
    return sel, metrics


def run_integrated_credit_risk_analysis():
    """
    综合模块: 将各子模块整合为完整的信用风险分析流程
    """
    print("=" * 60)
    print("综合模块: 整合信用风险违约相关性分析")
    print("=" * 60)

    # 步骤 1: 构建区域相关性结构
    xyz, areas, adj, corr_regional = build_regional_default_correlation(n_regions=10, base_correlation=0.3)

    # 步骤 2: 生成正交因子载荷
    n_assets = 20
    raw_loadings = np.random.randn(n_assets, 5)
    orth_loadings = orthogonalize_credit_factors(raw_loadings, method="mgs")

    # 步骤 3: 结合区域相关性与因子相关性
    n_reg = corr_regional.shape[0]
    # 每个资产映射到一个区域
    asset_region = np.arange(n_assets) % n_reg
    factor_corr = factor_covariance_from_loadings(orth_loadings, np.ones(orth_loadings.shape[1]))

    # TODO(Hole_3): 将因子相关性与空间相关性融合为总相关性矩阵
    # 科学背景: 在综合信用风险模型中，总违约相关性由两部分组成:
    #   (1) 宏观经济因子驱动的系统性相关性 factor_corr (来自 factor_orthogonalization.py)
    #   (2) 地理区域空间相关性 corr_regional (来自 spherical_correlation_grid.py)
    # 融合公式:
    #   total_corr = alpha * factor_corr + (1-alpha) * I
    #   对 i != j: total_corr[i,j] += beta * corr_regional[asset_region[i], asset_region[j]]
    # 其中 alpha = 0.6, beta = 0.2 为混合权重。
    # 融合后必须使用 nearest_correlation_matrix 确保矩阵半正定、对角线为 1。
    # 注意: factor_corr 维度为 n_assets x n_assets, corr_regional 维度为 n_reg x n_reg,
    # 每个资产 i 映射到区域 asset_region[i] = i % n_reg。
    # PLACEHOLDER: 请根据上述融合公式实现正确计算，并确保调用 nearest_correlation_matrix
    total_corr = np.eye(n_assets)  # 临时占位，会导致后续分析结果不正确

    # 步骤 4: 计算组合联合违约概率 (高斯 Copula)
    pd_vec = np.random.uniform(0.02, 0.12, n_assets)
    u_vec = normal_cdf(-1.5 * np.ones(n_assets))  # 简化的映射

    # 步骤 5: 网络传染增强
    local_intensity = pd_vec[asset_region[:n_reg]]
    net_intensity = network_cascade_intensity(adj, local_intensity, 0.1)

    # 步骤 6: 组合优化
    data = generate_credit_portfolio_data(n_assets=n_assets, seed=99)
    sel, metrics = credit_portfolio_optimization(
        data["expected_returns"],
        data["capital_charges"],
        data["pd_values"],
        data["lgd_values"],
        data["ead_values"],
        total_capital=600.0,
        var_limit=2500.0,
        resolution=600
    )

    # 步骤 7: 结构模型补充
    pd_structural = default_probability_from_structural(100.0, 0.05, 0.2, 30.0, 5.0)

    print(f"资产数量: {n_assets}")
    print(f"正交因子数: {orth_loadings.shape[1]}")
    print(f"总相关性矩阵条件数: {np.linalg.cond(total_corr):.2f}")
    print(f"平均网络增强违约强度: {net_intensity.mean():.4f}")
    print(f"选中资产数: {np.sum(sel)}")
    print(f"结构模型 5 年 PD: {pd_structural:.4f}")
    print("综合模块完成.\n")


def main():
    """
    主入口函数: 顺序执行所有模块并输出结果摘要
    """
    print("\n" + "=" * 60)
    print("  信用风险违约相关性建模 —— 博士级科研代码合成项目")
    print("  Spatial-Temporal Default Correlation Modeling")
    print("=" * 60 + "\n")

    # 执行各独立模块
    run_factor_orthogonalization_module()
    run_lattice_enumeration_module()
    run_linear_solver_module()
    run_interpolation_module()
    run_spherical_correlation_module()
    run_contagion_ode_module()
    run_structural_bvp_module()
    run_copula_quadrature_module()
    run_portfolio_optimization_module()

    # 执行整合分析
    run_integrated_credit_risk_analysis()

    print("=" * 60)
    print("  所有模块执行完毕，程序正常结束。")
    print("=" * 60)


if __name__ == "__main__":
    main()
