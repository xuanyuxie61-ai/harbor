"""
main.py
金融工程博士级合成项目统一入口。

项目主题：高维非线性风险平价投资组合优化
——基于流形学习、谱方法与随机PDE的风险度量

运行方式：
    python main.py

无需任何命令行参数，程序将自动生成模拟数据并完成完整分析流程。
"""

import numpy as np
import sys

# 导入各模块
from chebyshev_pricing import (
    chebyshev_grid,
    chebyshev_diff_matrix,
    spectral_var_cvar,
    circle01_monomial_integral,
)
from spherical_embedding import (
    correlation_to_spherical_embedding,
    circle01_sample_random,
    spherical_diversity_index,
    angular_distance_matrix,
)
from network_risk import (
    build_asset_digraph,
    pagerank_systemic_risk,
    delaunay_similarity_triangulation,
    stochastic_risk_diffusion,
    network_risk_contribution,
)
from monte_carlo_simulator import (
    simulate_returns_mc,
    bootstrap_risk_analysis,
    tournament_risk_simulation,
    high_dim_sphere_sampling,
)
from portfolio_optimizer import (
    markowitz_min_variance,
    risk_parity_weights,
    herfindahl_risk_concentration,
    effective_number_of_bets,
    risk_parity_with_budget_constraints,
)
from dynamics_model import (
    coupled_market_dynamics,
    trapezoidal_sde_solver,
    simulate_contagion,
    trapezoidal_ode_solver,
)
from simplex_search import (
    simplex_lattice_points,
    simplex_volume,
    covariance_simplex_volume,
    tet_quality_indicator_from_cov,
    lattice_portfolio_search,
    mesh_base_one,
)
from utils import (
    caesar_perturb,
    matrix_interpolation_upsample,
    polygonal_convex_hull,
    distance_to_position_mds,
    r8mat_condition_number,
)


def generate_synthetic_data(n_assets: int = 8, n_days: int = 252,
                            seed: int = 42) -> dict:
    """
    生成合成资产收益率数据。

    模型：
    - 预期年化收益率 μ_i ~ U(0.02, 0.15)
    - 年化波动率 σ_i ~ U(0.10, 0.35)
    - 相关性矩阵：基于单因子模型生成，再添加随机扰动
        C = β β^T + diag(1 - β^2) + ε，
      其中 β_i ~ U(0.3, 0.7)。
    """
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.02, 0.15, n_assets)
    sigma = rng.uniform(0.10, 0.35, n_assets)
    beta = rng.uniform(0.3, 0.7, n_assets)
    corr = np.outer(beta, beta) + np.diag(1.0 - beta ** 2)
    # 对称化并确保正定性
    corr = 0.5 * (corr + corr.T)
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, 0.05)
    corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
    # 归一化对角线为1
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1.0, 1.0)

    # 生成日度收益率
    cov = np.outer(sigma, sigma) * corr
    cov = cov / 252.0
    returns = rng.multivariate_normal(mu / 252.0, cov, size=n_days)
    return {
        "returns": returns,
        "mu": mu,
        "sigma": sigma,
        "corr": corr,
        "cov": cov * 252.0,
        "n_assets": n_assets,
        "n_days": n_days,
    }


def main():
    print("=" * 70)
    print("金融工程博士级合成项目：高维风险平价投资组合优化")
    print("Project 144: Portfolio Optimization and Risk Parity")
    print("=" * 70)

    # ================================================================
    # 1. 数据生成
    # ================================================================
    print("\n[阶段 1] 生成合成资产数据 ...")
    data = generate_synthetic_data(n_assets=8, n_days=252, seed=42)
    returns = data["returns"]
    mu = data["mu"]
    sigma = data["sigma"]
    corr = data["corr"]
    cov = data["cov"]
    n = data["n_assets"]
    print(f"  资产数量: {n}")
    print(f"  历史样本天数: {data['n_days']}")
    print(f"  年化收益率范围: [{mu.min():.4f}, {mu.max():.4f}]")
    print(f"  年化波动率范围: [{sigma.min():.4f}, {sigma.max():.4f}]")

    # ================================================================
    # 2. Chebyshev 谱方法风险度量
    # ================================================================
    print("\n[阶段 2] Chebyshev 谱方法计算高精度风险度量 ...")
    port_returns = np.mean(returns, axis=1)  # 等权重组合日收益
    risk_metrics = spectral_var_cvar(port_returns, alpha=0.05, n_cheb=64)
    print(f"  等权重组合 VaR(5%):  {risk_metrics['VaR']:.6f}")
    print(f"  等权重组合 CVaR(5%): {risk_metrics['CVaR']:.6f}")
    print(f"  样本均值: {risk_metrics['mean']:.6f}")
    print(f"  样本标准差: {risk_metrics['std']:.6f}")
    # 单位圆积分验证
    integral_val = circle01_monomial_integral(np.array([2, 2]))
    print(f"  单位圆积分验证 (x^2 y^2): {integral_val:.6f}")

    # ================================================================
    # 3. 球面嵌入与几何分散度
    # ================================================================
    print("\n[阶段 3] 资产相关性球面嵌入与几何分散度分析 ...")
    xyz_embed = correlation_to_spherical_embedding(corr, r=1.0, random_seed=42)
    div_index = spherical_diversity_index(xyz_embed)
    ang_dist = angular_distance_matrix(xyz_embed)
    print(f"  球面几何分散度指数: {div_index:.4f} (理论最大={2.0:.4f})")
    print(f"  平均角度距离: {np.mean(ang_dist):.4f} rad")
    # 高维球面采样（用于蒙特卡洛积分）
    sphere_samples = high_dim_sphere_sampling(1000, n)
    print(f"  高维球面采样验证 (1000点, 维度{n}): 完成")

    # ================================================================
    # 4. 网络分析与风险传播
    # ================================================================
    print("\n[阶段 4] 资产网络构建与系统性风险分析 ...")
    adj = build_asset_digraph(n, threshold=0.3, corr=corr)
    pagerank_scores = pagerank_systemic_risk(adj, damping=0.85)
    print(f"  PageRank 系统性风险得分:")
    for i in range(n):
        print(f"    资产 {i+1}: {pagerank_scores[i]:.4f}")

    # Delaunay 三角剖分（基于二维MDS嵌入）
    mds_2d = distance_to_position_mds(1.0 - corr, dim=2)
    delaunay_adj = delaunay_similarity_triangulation(mds_2d)
    n_edges = int(np.sum(delaunay_adj > 0) // 2)
    print(f"  Delaunay 相似性网络边数: {n_edges}")

    # 随机热方程风险扩散
    initial_risk = np.zeros(20 * 20)
    # 将 PageRank 最高的资产映射为风险源
    top_asset = int(np.argmax(pagerank_scores))
    source_idx = top_asset % 20 + 20 * (top_asset // 20 % 20)
    initial_risk[source_idx] = 1.0
    risk_field = stochastic_risk_diffusion(
        adj, initial_risk, omega=np.array([1.0, 0.5, 0.1, 2.0]), nx=20, ny=20
    )
    print(f"  风险扩散场最大响应: {np.max(risk_field):.4f}")
    print(f"  风险扩散场平均响应: {np.mean(risk_field):.4f}")

    # 网络风险贡献
    net_rc = network_risk_contribution(adj, returns)
    print(f"  网络风险贡献度 (归一化):")
    for i in range(n):
        print(f"    资产 {i+1}: {net_rc[i] / np.sum(net_rc):.4f}")

    # ================================================================
    # 5. 蒙特卡洛模拟与Bootstrap
    # ================================================================
    print("\n[阶段 5] 蒙特卡洛路径模拟与Bootstrap风险分析 ...")
    mc_paths = simulate_returns_mc(mu, sigma, corr, T=252, n_paths=3000)
    mc_final = np.sum(mc_paths[:, -1, :], axis=1)  # 组合累积收益
    mc_var = np.percentile(mc_final, 5)
    mc_cvar = np.mean(mc_final[mc_final <= mc_var])
    print(f"  MC 模拟组合 VaR(5%):  {mc_var:.4f}")
    print(f"  MC 模拟组合 CVaR(5%): {mc_cvar:.4f}")

    boot_result = bootstrap_risk_analysis(returns, n_bootstrap=2000)
    print(f"  Bootstrap VaR 均值: {boot_result['VaR_mean']:.6f}")
    print(f"  Bootstrap VaR 95% CI: [{boot_result['VaR_ci'][0]:.6f}, {boot_result['VaR_ci'][1]:.6f}]")
    print(f"  Bootstrap CVaR 均值: {boot_result['CVaR_mean']:.6f}")

    # 锦标赛风险模拟
    strengths = pagerank_scores + 0.1  # 确保正数
    tour_probs = tournament_risk_simulation(strengths, n_games=5000)
    print(f"  锦标赛相对表现概率:")
    for i in range(n):
        print(f"    资产 {i+1}: {tour_probs[i]:.4f}")

    # ================================================================
    # 6. 投资组合优化
    # ================================================================
    print("\n[阶段 6] 投资组合优化 ...")
    min_var = markowitz_min_variance(cov, mu=mu)
    print(f"  最小方差组合风险: {min_var['risk']:.4f}")
    print(f"  最小方差组合预期收益: {min_var['expected_return']:.4f}")
    print(f"  最小方差组合权重:")
    for i in range(n):
        print(f"    资产 {i+1}: {min_var['weights'][i]:.4f}")

    # === HOLE 2 BEGIN ===
    # TODO: 调用风险平价算法并处理返回结果
    # 需完成以下任务：
    #   1) 调用 risk_parity_weights(cov, max_iter=2000, tol=1e-10) 得到 rp
    #   2) 输出 rp['risk'], rp['diversification_ratio'], rp['iterations'],
    #      rp['weights'], rp['risk_contributions']
    #   3) 计算 h_index = herfindahl_risk_concentration(rp['risk_contributions'])
    #   4) 计算 enb = effective_number_of_bets(rp['risk_contributions'])
    #   5) 输出 Herfindahl 风险集中度与 ENB
    #   6) 调用 risk_parity_with_budget_constraints(cov, lower=np.full(n, 0.05),
    #                                                upper=np.full(n, 0.30)) 得到 rp_constrained
    #   7) 输出 rp_constrained['risk'] 与 rp_constrained['weights']
    # 注意：rp 字典键名必须与 portfolio_optimizer.py 中定义的一致
    # === HOLE 2 END ===

    # ================================================================
    # 7. 耦合动力学与传染模拟
    # ================================================================
    print("\n[阶段 7] 资产耦合动力学与风险传染模拟 ...")
    # === HOLE 4 BEGIN ===
    # TODO: 调用耦合动力学模块进行风险传染模拟与确定性ODE验证
    # 需完成以下任务：
    #   1) 调用 simulate_contagion(n_assets=n, T=2.0, dt=0.01, k1=2.0,
    #                               gamma=0.5, sigma_noise=0.3) 得到 t_dyn, y_dyn, max_dev
    #   2) 输出模拟时长与各资产最大偏离幅度
    #   3) 构造 K2_small = [[0.0, 1.0], [1.0, 0.0]]，y0_ode = [0.5, 0.0, -0.3, 0.0]
    #   4) 调用 trapezoidal_ode_solver，漂移项为 lambda t, y: coupled_market_dynamics(...)
    #      参数：k1=1.0, K2=K2_small, gamma=0.3, m=np.ones(n_ode)，n_ode=2
    #      时间区间 (0.0, 10.0)，n_steps=500
    #   5) 计算能量 energy = 0.5 * (y_ode[:, 1]^2 + y_ode[:, 3]^2)
    #   6) 输出双弹簧ODE能量变化（初值→终值）
    # 注意：coupled_market_dynamics 在 dynamics_model.py 中定义，其内部实现已被挖空（Hole 3）
    # === HOLE 4 END ===

    # ================================================================
    # 8. 单纯形格点搜索与质量评估
    # ================================================================
    print("\n[阶段 8] 单纯形格点搜索与协方差矩阵质量评估 ...")
    lattice_result = lattice_portfolio_search(n, t=10, Sigma=cov, mu=mu)
    print(f"  格点搜索评估点数: {lattice_result['n_points_evaluated']}")
    print(f"  格点最优组合风险: {lattice_result['optimal_risk']:.4f}")
    print(f"  格点最优组合夏普比率: {lattice_result['optimal_sharpe']:.4f}")
    print(f"  格点最优权重:")
    for i in range(n):
        print(f"    资产 {i+1}: {lattice_result['optimal_weights'][i]:.4f}")

    cov_vol = covariance_simplex_volume(cov)
    print(f"  协方差矩阵并行多面体体积: {cov_vol:.6f}")

    tet_quality = tet_quality_indicator_from_cov(cov)
    print(f"  协方差子矩阵条件数: {tet_quality['condition_number']:.4f}")
    print(f"  协方差质量评分: {tet_quality['quality_score']:.6f}")

    # mesh_base_one 验证
    test_elements = np.array([[0, 1, 2], [1, 2, 3]])
    fixed = mesh_base_one(test_elements, node_num=4)
    print(f"  网格索引修正验证: {fixed.tolist()}")

    # ================================================================
    # 9. 工具函数验证
    # ================================================================
    print("\n[阶段 9] 工具函数验证 ...")
    perturbed = caesar_perturb(returns[:10, 0], k=3)
    print(f"  Caesar扰动前后均值差: {abs(np.mean(perturbed) - np.mean(returns[:10, 0])):.6f}")

    up = matrix_interpolation_upsample(corr[:4, :4], factor=2)
    print(f"  矩阵上采样 (4×4 → 8×8): 完成, 新形状={up.shape}")

    hull = polygonal_convex_hull(np.column_stack([mu, sigma]))
    print(f"  收益-风险凸包顶点数: {len(hull['vertices'])}")
    print(f"  凸包体积: {hull['volume']:.6f}")

    cond = r8mat_condition_number(cov)
    print(f"  协方差矩阵条件数: {cond:.4f}")

    # ================================================================
    # 10. 汇总输出
    # ================================================================
    print("\n" + "=" * 70)
    print("[总结] 风险平价投资组合优化结果")
    print("=" * 70)
    eq_weights = np.ones(n) / n
    eq_risk = np.sqrt(eq_weights @ cov @ eq_weights)
    print(f"组合类型          | 风险(σ) | 预期收益 | 分散度(DR) | 集中度(H)")
    print("-" * 70)
    print(f"等权重            | {eq_risk:>7.4f} | {np.mean(mu):>8.4f} | {'N/A':>10} | {'N/A':>8}")
    print(f"最小方差          | {min_var['risk']:>7.4f} | {min_var['expected_return']:>8.4f} | {'N/A':>10} | {'N/A':>8}")
    print(f"风险平价          | {rp['risk']:>7.4f} | {'N/A':>8} | {rp['diversification_ratio']:>10.4f} | {h_index:>8.4f}")
    print(f"约束风险平价      | {rp_constrained['risk']:>7.4f} | {'N/A':>8} | {'N/A':>10} | {'N/A':>8}")
    print(f"格点最优          | {lattice_result['optimal_risk']:>7.4f} | {'N/A':>8} | {'N/A':>10} | {'N/A':>8}")
    print("=" * 70)
    print("程序正常结束。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
