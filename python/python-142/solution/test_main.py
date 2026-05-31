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

    # 总相关性 = 因子相关性 + 区域空间相关性 (混合)
    total_corr = 0.6 * factor_corr + 0.4 * np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            ri, rj = asset_region[i], asset_region[j]
            spatial_boost = corr_regional[ri, rj] * 0.2
            total_corr[i, j] += spatial_boost
            total_corr[j, i] += spatial_boost
    total_corr = nearest_correlation_matrix(total_corr)

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

# ================================================================
# 测试用例（30个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: gauss_legendre_nodes_weights 节点数与权重和 ----
from utils import gauss_legendre_nodes_weights
x, w = gauss_legendre_nodes_weights(10)
assert len(x) == 10 and len(w) == 10, '[TC01] 节点/权重长度错误 FAILED'
assert abs(np.sum(w) - 2.0) < 1e-12, '[TC01] 权重和不等于 2 FAILED'
assert np.all(x >= -1.0) and np.all(x <= 1.0), '[TC01] 节点不在 [-1,1] 内 FAILED'

# ---- TC02: logistic_transform 区间映射 ----
from utils import logistic_transform
z = np.array([-100.0, -5.0, 0.0, 5.0, 100.0])
y = logistic_transform(z, a=0.0, b=1.0)
assert np.all(y >= 0.0) and np.all(y <= 1.0), '[TC02] 映射越界 FAILED'
assert y[2] > 0.49 and y[2] < 0.51, '[TC02] z=0 未映射到 0.5 FAILED'

# ---- TC03: safe_sqrt 负数返回 0 ----
from utils import safe_sqrt
x = np.array([-5.0, -0.01, 0.0, 4.0, 9.0])
y = safe_sqrt(x)
assert abs(y[0] - 0.0) < 1e-15, '[TC03] 负数未返回 0 FAILED'
assert abs(y[2] - 0.0) < 1e-15, '[TC03] 零输入错误 FAILED'
assert abs(y[3] - 2.0) < 1e-15, '[TC03] sqrt(4) 应为 2 FAILED'
assert abs(y[4] - 3.0) < 1e-15, '[TC03] sqrt(9) 应为 3 FAILED'

# ---- TC04: normal_cdf(0) ≈ 0.5 ----
assert abs(normal_cdf(np.array([0.0]))[0] - 0.5) < 1e-10, '[TC04] normal_cdf(0) 偏离 0.5 FAILED'
x_test = np.array([-3.0, 0.0, 3.0])
cdf_vals = normal_cdf(x_test)
assert cdf_vals[0] < 0.01, '[TC04] CDF(-3) 应接近 0 FAILED'
assert cdf_vals[2] > 0.99, '[TC04] CDF(3) 应接近 1 FAILED'

# ---- TC05: is_positive_definite 单位阵 ----
from utils import is_positive_definite
I = np.eye(5)
assert is_positive_definite(I), '[TC05] 单位阵应为正定 FAILED'
M_sing = np.ones((4, 4))
assert not is_positive_definite(M_sing), '[TC05] 奇异阵应判定为非正定 FAILED'

# ---- TC06: nearest_correlation_matrix 对角线全为 1 ----
np.random.seed(42)
A_bad = np.random.randn(6, 6)
A_bad = A_bad @ A_bad.T
A_bad = A_bad / np.max(np.abs(A_bad))
R = nearest_correlation_matrix(A_bad)
assert R.shape == (6, 6), '[TC06] 输出维度错误 FAILED'
assert np.allclose(np.diag(R), 1.0, atol=1e-6), '[TC06] 对角线不全为 1 FAILED'
eigvals_R = np.linalg.eigvalsh(R)
assert np.all(eigvals_R > -1e-8), '[TC06] 输出非半正定 FAILED'

# ---- TC07: tridiagonal_solve 对角系统精确解 ----
from utils import tridiagonal_solve
n_tri = 5
b_diag = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
a_sub = np.array([0.0, 0.0, 0.0, 0.0])
c_sup = np.array([0.0, 0.0, 0.0, 0.0])
x_exact = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
d_rhs = b_diag * x_exact  # 纯对角: b_i * x_i = d_i
x_tri = tridiagonal_solve(a_sub, b_diag, c_sup, d_rhs)
assert np.allclose(x_tri, x_exact, atol=1e-12), '[TC07] 对角系统求解错误 FAILED'

# ---- TC08: modified_gram_schmidt Q 列正交归一 ----
np.random.seed(42)
A_test = np.random.randn(8, 5)
Q, R, rank = modified_gram_schmidt(A_test)
assert Q.shape[1] == rank, '[TC08] Q 列数与秩不匹配 FAILED'
assert rank <= 5, '[TC08] 秩超过列数 FAILED'
QtQ = Q.T @ Q
assert np.allclose(QtQ, np.eye(rank), atol=1e-10), '[TC08] Q 列非正交 FAILED'

# ---- TC09: orthogonalize_credit_factors 载荷行范式 ≤ 1 ----
np.random.seed(42)
raw = np.random.randn(20, 6)
orth = orthogonalize_credit_factors(raw, method="mgs")
row_norms = np.sqrt(np.sum(orth**2, axis=1))
assert np.all(row_norms <= 1.0 + 1e-10), '[TC09] 载荷行范式超过 1 FAILED'
assert orth.shape[0] == 20, '[TC09] 输出行数错误 FAILED'

# ---- TC10: portfolio_weight_grid 权重和为 1 ----
np.random.seed(42)
weights = portfolio_weight_grid(3, 5)
assert weights.ndim == 2 and weights.shape[1] == 3, '[TC10] 权重矩阵维度错误 FAILED'
assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-12), '[TC10] 权重和不等于 1 FAILED'
assert np.all(weights >= -1e-12), '[TC10] 权重存在负值 FAILED'

# ---- TC11: correlation_simplex_grid 特征值和等于 n_factors ----
n_f = 3
e_grids = correlation_simplex_grid(n_f, 4)
assert len(e_grids) > 0, '[TC11] 特征值网格为空 FAILED'
for ev in e_grids:
    assert abs(np.sum(ev) - n_f) < 1e-10, '[TC11] 特征值和不等于 n_factors FAILED'

# ---- TC12: cg_ne_solve 用 Helmert 矩阵精确恢复 ----
np.random.seed(42)
n_h = 15
H = helmert_matrix(n_h)
x_true = np.random.randn(n_h)
b_h = H @ x_true
x_sol, iters, res = cg_ne_solve(H, b_h, tol=1e-12)
err = np.linalg.norm(x_sol - x_true)
assert err < 1e-8, '[TC12] CG-NE 解误差过大 FAILED'
assert iters > 0, '[TC12] CG-NE 未迭代 FAILED'

# ---- TC13: clenshaw_curtis_nodes 范围正确 ----
cc = clenshaw_curtis_nodes(16, a=0.0, b=10.0)
assert len(cc) == 16, '[TC13] Clenshaw-Curtis 节点数错误 FAILED'
assert np.all(cc >= 0.0) and np.all(cc <= 10.0), '[TC13] CC 节点越界 FAILED'

# ---- TC14: interpolate_credit_curve 节点精确恢复 ----
t_data = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
p_data = np.array([0.005, 0.012, 0.025, 0.038, 0.055, 0.072, 0.095])
p_check = interpolate_credit_curve(t_data, p_data, t_data, method="linear")
assert np.allclose(p_check, p_data, atol=1e-12), '[TC14] 线性插值节点恢复失败 FAILED'

# ---- TC15: sphere_llt_grid_points 点数正确 ----
lat, lon = 3, 4
xyz_pts = sphere_llt_grid_points(1.0, np.zeros(3), lat, lon)
assert xyz_pts.shape[0] == 2 + lat * lon, '[TC15] LLT 网格点数错误 FAILED'
assert xyz_pts.shape[1] == 3, '[TC15] LLT 网格坐标维度错误 FAILED'

# ---- TC16: spherical_voronoi_areas 面积非负 ----
areas_v, faces_v = spherical_voronoi_areas(xyz_pts)
assert np.all(areas_v >= -1e-12), '[TC16] Voronoi 面积存在负值 FAILED'
assert len(areas_v) == xyz_pts.shape[0], '[TC16] 面积数量与点数不匹配 FAILED'

# ---- TC17: simulate_default_contagion 状态在 [0,1] 内 ----
np.random.seed(42)
t_ode, y_ode = simulate_default_contagion(
    initial_default_rate=0.05,
    initial_pressure=0.1,
    initial_buffer=0.5,
    t_max=2.0,
    n_steps=200,
    theta=0.5
)
assert len(t_ode) == 201, '[TC17] 时间步数错误 FAILED'
assert np.all(y_ode[:, 0] >= -1e-12) and np.all(y_ode[:, 0] <= 1.0 + 1e-12), '[TC17] 违约强度越界 FAILED'
assert np.all(y_ode[:, 1] >= -1e-12) and np.all(y_ode[:, 1] <= 1.0 + 1e-12), '[TC17] 传染压力越界 FAILED'
assert np.all(y_ode[:, 2] >= -1e-12) and np.all(y_ode[:, 2] <= 1.0 + 1e-12), '[TC17] 缓冲水平越界 FAILED'

# ---- TC18: network_cascade_intensity 网络效应增强 ----
adj_test = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
local_test = np.array([0.1, 0.2, 0.15])
net_test = network_cascade_intensity(adj_test, local_test, 0.1)
assert np.all(net_test >= local_test - 1e-12), '[TC18] 网络效应未增强 FAILED'
assert np.all(net_test <= 1.0 + 1e-12), '[TC18] 网络增强强度越界 FAILED'

# ---- TC19: solve_ill_bvp_fd 边界条件满足 ----
x_bvp, y_bvp = solve_ill_bvp_fd(epsilon=0.01, n_nodes=80, bc_left=2.0, bc_right=1.0)
assert np.isclose(y_bvp[0], 2.0, atol=1e-6), '[TC19] 左边界条件不满足 FAILED'
assert np.isclose(y_bvp[-1], 1.0, atol=1e-6), '[TC19] 右边界条件不满足 FAILED'
assert len(x_bvp) == 80, '[TC19] BVP 节点数错误 FAILED'

# ---- TC20: default_probability_from_structural 在 [0,1] 内 ----
pd_s1 = default_probability_from_structural(100.0, 0.05, 0.2, 30.0, 5.0)
pd_s2 = default_probability_from_structural(100.0, 0.05, 0.2, 30.0, 10.0)
pd_s3 = default_probability_from_structural(30.0, 0.05, 0.2, 100.0, 5.0)
assert 0.0 <= pd_s1 <= 1.0, '[TC20] PD_5y 越界 FAILED'
assert 0.0 <= pd_s2 <= 1.0, '[TC20] PD_10y 越界 FAILED'
assert 0.0 <= pd_s3 <= 1.0, '[TC20] PD_v0<barrier 越界 FAILED'
assert pd_s1 < pd_s2, '[TC20] 5年PD应小于10年PD FAILED'

# ---- TC21: integrate_standard_triangle 常数函数面积 ----
f_const = lambda x, y: 1.0
tri_area = integrate_standard_triangle(f_const, rule=7)
assert abs(tri_area - 0.5) < 1e-8, '[TC21] 标准三角形面积不为 0.5 FAILED'

# ---- TC22: cauchy_principal_value 已知解析值 ----
f_cpv = lambda t: t**2
cpv_val = cauchy_principal_value(f_cpv, -1.0, 2.0, 0.5, n=64)
# 解析: int_{-1}^2 t^2/(t-0.5) = int_{-1}^2 (t+0.5 + 0.25/(t-0.5)) dt
# = [t^2/2 + 0.5t]_{-1}^2 + 0.25*ln|(2-0.5)/(-1-0.5)|
# = (2+1-0.5+0.5) + 0.25*ln(1.5/1.5) = 3 + 0 = 3
expected_cpv = 3.0
assert abs(cpv_val - expected_cpv) < 1e-6, '[TC22] CPV 积分值错误 FAILED'

# ---- TC23: knapsack_01_dp 简单已知解 ----
from portfolio_knapsack import knapsack_01_dp
vals = np.array([60, 100, 120], dtype=float)
wts = np.array([10, 20, 30], dtype=int)
cap = 50
max_val, sel = knapsack_01_dp(vals, wts, cap)
assert abs(max_val - 220.0) < 1e-10, '[TC23] 背包最优值应为 220 FAILED'
assert np.sum(wts * sel) <= cap, '[TC23] 背包重量超限 FAILED'

# ---- TC24: generate_credit_portfolio_data 数据结构正确 ----
np.random.seed(42)
data = generate_credit_portfolio_data(n_assets=10, seed=42)
assert "expected_returns" in data, '[TC24] 缺少 expected_returns FAILED'
assert "capital_charges" in data, '[TC24] 缺少 capital_charges FAILED'
assert "pd_values" in data, '[TC24] 缺少 pd_values FAILED'
assert len(data["expected_returns"]) == 10, '[TC24] 资产数量错误 FAILED'
assert np.all(data["pd_values"] >= 0) and np.all(data["pd_values"] <= 1), '[TC24] PD 越界 FAILED'

# ---- TC25: factor_covariance_from_loadings 对角线为 1 ----
np.random.seed(42)
B_test = np.random.randn(15, 4)
B_test = B_test / (np.linalg.norm(B_test, axis=1, keepdims=True) + 1e-12) * 0.6
corr_mat = factor_covariance_from_loadings(B_test, np.ones(4))
assert np.allclose(np.diag(corr_mat), 1.0, atol=1e-6), '[TC25] 协方差矩阵对角线不为 1 FAILED'
assert corr_mat.shape == (15, 15), '[TC25] 协方差矩阵维度错误 FAILED'

# ---- TC26: cholesky_with_pivot 重构验证 ----
np.random.seed(42)
C_test = np.random.randn(6, 6)
C_test = C_test @ C_test.T
C_test = C_test / np.max(np.abs(C_test))
np.fill_diagonal(C_test, 1.0)
L_chol = cholesky_with_pivot(C_test)
assert L_chol is not None, '[TC26] Cholesky 分解返回 None FAILED'
recon = L_chol @ L_chol.T
assert np.allclose(recon, C_test, atol=1e-8), '[TC26] Cholesky 重构失败 FAILED'

# ---- TC27: helmert_matrix 正交性 ----
H8 = helmert_matrix(8)
assert H8.shape == (8, 8), '[TC27] Helmert 矩阵维度错误 FAILED'
HtH = H8 @ H8.T
assert np.allclose(HtH, np.eye(8), atol=1e-10), '[TC27] Helmert 矩阵非正交 FAILED'

# ---- TC28: arc_length_parameterize 弧长参数 ----
pts = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=float)
t_param, s = arc_length_parameterize(pts)
assert len(t_param) == 4, '[TC28] 参数长度错误 FAILED'
assert np.isclose(t_param[0], 0.0), '[TC28] 起点参数不为 0 FAILED'
assert np.isclose(t_param[-1], 1.0), '[TC28] 终点参数不为 1 FAILED'
assert np.all(np.diff(t_param) >= -1e-12), '[TC28] 参数非单调 FAILED'
assert np.all(np.diff(s) >= -1e-12), '[TC28] 累积距离非单调 FAILED'

# ---- TC29: build_regional_default_correlation 输出结构 ----
xyz_r, areas_r, adj_r, corr_r = build_regional_default_correlation(n_regions=8, base_correlation=0.3)
n_r = xyz_r.shape[0]
assert corr_r.shape == (n_r, n_r), '[TC29] 相关性矩阵维度错误 FAILED'
assert np.allclose(np.diag(corr_r), 1.0, atol=1e-5), '[TC29] 相关性矩阵对角线不为 1 FAILED'
assert adj_r.shape == (n_r, n_r), '[TC29] 邻接矩阵维度错误 FAILED'
assert areas_r.shape == (n_r,), '[TC29] 面积向量维度错误 FAILED'
assert abs(np.sum(areas_r) - 1.0) < 1e-10, '[TC29] 面积权重和不等于 1 FAILED'

# ---- TC30: credit_portfolio_optimization 返回有效选择 ----
np.random.seed(42)
d30 = generate_credit_portfolio_data(n_assets=12, seed=42)
sel30, met30 = credit_portfolio_optimization(
    d30["expected_returns"],
    d30["capital_charges"],
    d30["pd_values"],
    d30["lgd_values"],
    d30["ead_values"],
    total_capital=400.0,
    var_limit=1500.0,
    resolution=400
)
assert np.any(sel30), '[TC30] 未选择任何资产 FAILED'
assert met30["total_capital"] <= 400.0 * 1.01, '[TC30] 资本约束违反 FAILED'
assert met30["total_risk"] <= 1500.0 * 1.01, '[TC30] 风险约束违反 FAILED'
assert met30["total_return"] > 0, '[TC30] 总收益非正 FAILED'

print('\n全部 30 个测试通过!\n')
