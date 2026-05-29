#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thermodynamic-Geometric Collaborative Filtering (TGCF)
=====================================================

主入口文件：零参数运行，执行完整的前沿博士级推荐系统协同过滤流程。

科学问题背景
------------
将推荐系统建模为热力学-几何耦合系统：
- 用户与物品视为高维潜空间中的粒子
- 评分视为粒子间的能量交换
- 用户偏好演化服从准周期动力学（quasiperiodic dynamics）
- 信息扩散服从流形上的热传导方程
- 预测不确定性通过 Laguerre 广义多项式混沌（gPC）量化
- 潜空间通过有限元（FEM）离散化实现缺失值插值
- 几何采样、CORDIC 快速计算、截断正态分布、Hilbert 曲线 LSH、
  Levenshtein 文本相似性、三角剖分边界检测等多学科方法协同工作

运行方式
--------
    python main.py
"""

import numpy as np
import time

# 导入所有科学计算模块
from quasiperiodic_dynamics import QuasiperiodicPreferenceDynamics
from laguerre_chaos import LaguerrePolynomialChaos
from cordic_engine import CordicEngine
from fem_embedding import FemEmbeddingInterpolator
from heat_diffusion import HeatDiffusionRecommender
from gaussian_solver import RobustGaussianSolver
from geometric_sampler import GeometricSampler
from hilbert_hashing import HilbertLSH
from string_similarity import StringSimilarityEngine
from truncated_distribution import TruncatedNormalRatingModel
from triangulation_boundary import TriangulationBoundaryDetector
from aggregate_stats import AggregateStatistics


def generate_synthetic_data(n_users=80, n_items=60, density=0.15, seed=42):
    """
    生成具有物理结构的合成评分数据矩阵 R ∈ [1,5]^{n_users × n_items}
    
    物理模型：
        R_{ui} = μ + b_u + b_i + p_u^T q_i + ε_{ui}
    
    其中：
        μ    : 全局平均评分
        b_u  : 用户偏置（user bias），建模为截断正态随机变量
        b_i  : 物品偏置（item bias）
        p_u  : 用户潜向量，采样自高维球面 S^{d-1}
        q_i  : 物品潜向量
        ε_ui : 热噪声，服从 N(0, σ²)
    
    边界处理：
        - 密度控制在 [0.05, 0.50] 区间
        - 评分严格截断在 [1.0, 5.0]
    """
    rng = np.random.RandomState(seed)
    density = np.clip(density, 0.05, 0.50)
    
    d = 8  # 潜空间维度
    mu_global = 3.0
    sigma_bias = 0.6
    sigma_noise = 0.4
    
    # 用户潜向量：球面均匀采样（von Mises-Fisher 分布的均匀特例）
    P = rng.randn(n_users, d)
    P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-12
    
    # 物品潜向量
    Q = rng.randn(n_items, d)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12
    
    # 偏置项
    b_u = rng.normal(0.0, sigma_bias, n_users)
    b_i = rng.normal(0.0, sigma_bias * 0.5, n_items)
    
    # 完整评分矩阵
    R_full = mu_global + b_u[:, None] + b_i[None, :] + P @ Q.T * 2.0
    R_full += rng.normal(0.0, sigma_noise, (n_users, n_items))
    R_full = np.clip(R_full, 1.0, 5.0)
    
    # 随机mask生成观测矩阵
    mask = rng.rand(n_users, n_items) < density
    R_obs = np.where(mask, R_full, np.nan)
    
    # 物品元数据（用于字符串相似性冷启动）
    vocab = ["量子", "热力学", "有限元", "混沌", "扩散", "熵", "流形", "特征",
             "张量", "核函数", "谱分析", "拓扑", "测度", "泛函", "变分",
             "辛几何", "李群", "纤维丛", "同调", "上同调"]
    item_metadata = []
    for i in range(n_items):
        n_words = rng.randint(1, 4)
        words = [vocab[rng.randint(len(vocab))] for _ in range(n_words)]
        item_metadata.append("".join(words))
    
    return R_obs, R_full, P, Q, item_metadata


def main():
    print("=" * 70)
    print("Thermodynamic-Geometric Collaborative Filtering (TGCF)")
    print("前沿博士级推荐系统协同过滤科学计算平台")
    print("=" * 70)
    
    t_start = time.time()
    
    # ================================
    # 1. 数据生成
    # ================================
    print("\n[阶段 1] 生成具有物理结构的合成评分数据...")
    R_obs, R_full, P_true, Q_true, item_metadata = generate_synthetic_data()
    n_users, n_items = R_obs.shape
    observed = ~np.isnan(R_obs)
    print(f"  用户数量: {n_users}, 物品数量: {n_items}")
    print(f"  观测密度: {observed.mean()*100:.2f}%")
    
    # ================================
    # 2. 截断正态分布建模评分边界
    # ================================
    print("\n[阶段 2] 截断正态分布建模有界评分...")
    # 公式: X ~ TN(μ, σ², a, b)
    # f(x) = φ((x-μ)/σ) / [σ(Φ((b-μ)/σ) - Φ((a-μ)/σ))],  x ∈ [a,b]
    trunc_model = TruncatedNormalRatingModel(mu=3.0, sigma=1.2, a=1.0, b=5.0)
    observed_ratings = R_obs[observed]
    trunc_mean = trunc_model.mean()
    trunc_var = trunc_model.variance()
    print(f"  截断正态均值 E[X] = {trunc_mean:.4f}")
    print(f"  截断正态方差 Var[X] = {trunc_var:.4f}")
    samples = trunc_model.sample(1000)
    print(f"  蒙特卡洛样本均值: {samples.mean():.4f}, 样本方差: {samples.var():.4f}")
    
    # ================================
    # 3. 热扩散方程驱动的信息传播
    # ================================
    print("\n[阶段 3] 基于 2D 热传导方程的评分信息扩散...")
    # 物理模型:
    #   ∂u/∂t - α Δu + k(x,y,t) u = f(x,y,t)  在 Ω 内
    #   u = g(x,y,t)                          在 ∂Ω 上
    # 采用后向 Euler 离散:
    #   (I + α Δt L) u^{n+1} = u^n + Δt f^{n+1}
    heat = HeatDiffusionRecommender(alpha=0.05, n_steps=8)
    R_diffused = heat.diffuse(R_obs)
    
    # ================================
    # 4. 有限元潜空间插值 (FEM)
    # ================================
    print("\n[阶段 4] 3D 有限元潜空间插值...")
    # 使用四面体有限元基函数 φ_i 在潜空间中进行插值:
    #   ũ(x) = Σ_{i=1}^4 u_i φ_i(x)
    # 其中 φ_i 为体积坐标（barycentric coordinates）
    fem = FemEmbeddingInterpolator(latent_dim=6)
    R_fem = fem.interpolate(R_diffused)
    
    # ================================
    # 5. 准周期动力学：时间演化
    # ================================
    print("\n[阶段 5] 准周期 ODE 偏好动力学演化...")
    # 四阶准周期系统:
    #   d⁴y/dt⁴ + (π² + 1) d²y/dt² + π² y = 0
    # 精确解: y(t) = cos(t) + cos(π t)
    qpd = QuasiperiodicPreferenceDynamics()
    t_eval = np.linspace(0, 10, 50)
    y_exact = qpd.exact_solution(t_eval)
    y_ode = qpd.integrate_ode(t_eval)
    err_ode = np.linalg.norm(y_exact - y_ode, axis=1).mean()
    print(f"  准周期 ODE 数值解与精确解平均 L2 误差: {err_ode:.2e}")
    # 将动力学用于时间演化因子
    temporal_factor = 1.0 + 0.05 * y_ode[-1, 0]
    R_temporal = R_fem * temporal_factor
    R_temporal = np.clip(R_temporal, 1.0, 5.0)
    
    # ================================
    # 6. CORDIC 快速引擎计算相似度
    # ================================
    print("\n[阶段 6] CORDIC 算法快速近似计算...")
    # CORDIC 旋转模式:
    #   [x_{i+1}]   [1      -σ_i 2^{-i}] [x_i]
    #   [y_{i+1}] = [σ_i 2^{-i}   1    ] [y_i]
    #   z_{i+1} = z_i - σ_i arctan(2^{-i})
    # 其中 σ_i = sign(z_i)
    cordic = CordicEngine(n_iter=24)
    angle = np.pi / 6.0
    cos_c, sin_c = cordic.cossin(angle)
    print(f"  CORDIC cos(π/6) = {cos_c:.10f}, sin(π/6) = {sin_c:.10f}")
    print(f"  真值           = {np.cos(angle):.10f},      = {np.sin(angle):.10f}")
    
    # 使用 CORDIC 计算用户-物品相似度核
    # K(u,i) = exp( -||p_u - q_i||² / (2σ²) )
    # 通过 CORDIC exp 近似加速
    sigma_kernel = 1.5
    similarity_kernel = cordic.compute_similarity_kernel(P_true, Q_true, sigma_kernel)
    
    # ================================
    # 7. Laguerre 广义多项式混沌量化不确定性
    # ================================
    print("\n[阶段 7] Laguerre 广义多项式混沌 (gPC) 不确定性量化...")
    # Laguerre 多项式递推:
    #   L_0(x) = 1
    #   L_1(x) = 1 - x
    #   (n+1) L_{n+1}(x) = (2n+1-x) L_n(x) - n L_{n-1}(x)
    # 
    # 指数加权内积:
    #   T_{ij} = ∫_0^∞ exp(βx) L_i(x) L_j(x) exp(-x) dx
    #          ≈ Σ_k w_k exp(β x_k) L_i(x_k) L_j(x_k)
    lpc = LaguerrePolynomialChaos(max_degree=5)
    beta = 0.3
    exp_table = lpc.exponential_product_table(beta)
    print(f"  Laguerre 指数加权积矩阵范数: {np.linalg.norm(exp_table):.4f}")
    # 不确定性传播: Var[ŷ] ≈ Σ_{k>0} c_k² ⟨L_k, L_k⟩_w
    uncertainty = lpc.propagate_uncertainty(observed_ratings, beta)
    print(f"  评分预测不确定性 (gPC): {uncertainty:.4f}")
    
    # ================================
    # 8. Hilbert 曲线 LSH 快速近邻搜索
    # ================================
    print("\n[阶段 8] 3D Hilbert 空间填充曲线 LSH...")
    # Hilbert 曲线映射: H = xyz_to_h(x,y,z,r)
    # 保持局部性: 欧氏空间中邻近的点在 H 索引中也邻近
    hilbert = HilbertLSH(order=4)
    user_hashes = hilbert.hash_vectors(P_true[:, :3])
    item_hashes = hilbert.hash_vectors(Q_true[:, :3])
    nn_pairs = hilbert.approximate_nn(user_hashes, item_hashes, top_k=5)
    print(f"  Hilbert LSH 近邻对数量: {len(nn_pairs)}")
    
    # ================================
    # 9. Levenshtein 字符串相似性（冷启动）
    # ================================
    print("\n[阶段 9] Levenshtein 编辑距离冷启动处理...")
    # Levenshtein 递推:
    #   d[i,j] = min( d[i-1,j] + 1,
    #                 d[i,j-1] + 1,
    #                 d[i-1,j-1] + cost )
    #   cost = 0 if s[i]=t[j] else 1
    string_sim = StringSimilarityEngine()
    # 为冷启动物品计算元数据相似度矩阵
    cold_item_sim = string_sim.compute_similarity_matrix(item_metadata[:10])
    print(f"  冷启动物品元数据相似度矩阵范数: {np.linalg.norm(cold_item_sim):.4f}")
    
    # ================================
    # 10. 高斯消元求解矩阵分解系统
    # ================================
    print("\n[阶段 10] 鲁棒高斯消元求解隐因子系统...")
    # 矩阵分解优化问题:
    #   min_{P,Q} Σ_{(u,i)∈Ω} (R_{ui} - p_u^T q_i)² + λ(||P||²_F + ||Q||²_F)
    # 固定 Q 时，对每个 u 有线性系统:
    #   (Σ_i q_i q_i^T + λI) p_u = Σ_i R_{ui} q_i
    solver = RobustGaussianSolver()
    # 构造小规模测试系统（使用良态矩阵）
    rng_test = np.random.RandomState(123)
    M = rng_test.randn(20, 20)
    A_test = M.T @ M + 2.0 * np.eye(20)  # 对称正定，条件数良好
    b_test = np.ones(20)
    x_lu = solver.solve_plu(A_test, b_test)
    x_ge = solver.solve_gauss(A_test, b_test)
    x_np = np.linalg.solve(A_test, b_test)
    print(f"  PLU 与 Gauss 消元解差异: {np.linalg.norm(x_lu - x_ge):.2e}")
    print(f"  与 NumPy 参考解差异: {np.linalg.norm(x_lu - x_np):.2e}")
    det_A = solver.determinant(A_test)
    print(f"  测试矩阵行列式: {det_A:.4e}")
    
    # ================================
    # 11. 三角剖分边界检测（社区发现）
    # ================================
    print("\n[阶段 11] 三角剖分边界检测识别利基社区...")
    # Delaunay 三角剖分边界边检测:
    #   - 内部边出现两次（正/反向各一次）
    #   - 边界边仅出现一次
    boundary_det = TriangulationBoundaryDetector()
    # 使用用户潜向量前二维构建三角剖分
    boundary_nodes = boundary_det.detect_boundary(P_true[:, :2])
    print(f"  检测到的边界节点数: {len(boundary_nodes)} / {n_users}")
    
    # ================================
    # 12. 几何采样与蒙特卡洛积分
    # ================================
    print("\n[阶段 12] 几何蒙特卡洛采样与积分...")
    # 单位圆上的单项式积分:
    #   I = ∮_{S¹} x^{e1} y^{e2} ds
    # 若 e1 或 e2 为奇数，I = 0
    # 否则 I = 2 Γ((e1+1)/2) Γ((e2+1)/2) / Γ((e1+e2+2)/2)
    geom = GeometricSampler(n_samples=5000)
    # 采样单位圆上的点用于用户嵌入正则化
    circle_samples = geom.sample_unit_circle(500)
    monomial_integral = geom.circle_monomial_integral([2, 2])
    print(f"  圆上 x²y² 的精确积分值: {monomial_integral:.6f}")
    mc_estimate = geom.monte_carlo_circle_integral(lambda x,y: (x**2)*(y**2), 10000)
    print(f"  蒙特卡洛估计 (N=10000): {mc_estimate:.6f}")
    
    # 正象限圆上的距离统计（多样性度量）
    mu_dist, var_dist = geom.positive_circle_distance_stats(2000)
    print(f"  正象限圆上随机点距离均值: {mu_dist:.4f}, 方差: {var_dist:.4f}")
    
    # 四面体-超平面相交计算置信区域
    tetra = np.array([[0.0, 0.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])
    plane_p = np.array([0.2, 0.2, 0.2])
    plane_n = np.array([1.0, 1.0, 1.0])
    n_int, pts_int = geom.plane_tetrahedron_intersect(plane_p, plane_n, tetra)
    print(f"  平面与四面体相交顶点数: {n_int}")
    if n_int >= 3:
        area_int = geom.quadrilateral_area_3d(pts_int[:n_int])
        print(f"  相交截面面积: {area_int:.4f}")
    
    # ================================
    # 13. 聚合统计评估
    # ================================
    print("\n[阶段 13] 聚合统计与系统评估...")
    # 用户分组统计
    agg = AggregateStatistics()
    # 基于潜向量聚类分组
    groups = agg.group_users_by_embedding(P_true, n_clusters=4)
    group_stats = agg.compute_group_statistics(R_temporal, groups)
    print(f"  用户分组数: {len(group_stats)}")
    for gid, stats in group_stats.items():
        print(f"    组 {gid}: 均值={stats['mean']:.3f}, 标准差={stats['std']:.3f}, "
              f"最小值={stats['min']:.3f}, 最大值={stats['max']:.3f}")
    
    # ================================
    # 14. 最终预测与误差评估
    # ================================
    print("\n[阶段 14] 最终预测合成与精度评估...")
    # 综合模型:
    #   Ŷ = w_heat·Y_heat + w_fem·Y_fem + w_temporal·Y_temporal + w_kernel·K
    # 权重通过最小二乘确定
    # TODO(Hole 3): 实现最终预测合成
    # 综合模型: Ŷ = w_heat·Y_heat + w_fem·Y_fem + w_temporal·Y_temporal + w_kernel·K
    # 可用变量:
    #   - R_diffused  : 热扩散后的评分矩阵 (n_users, n_items)
    #   - R_fem       : FEM 插值后的评分矩阵 (n_users, n_items)
    #   - R_temporal  : 准周期动力学调制后的评分矩阵 (n_users, n_items)
    #   - similarity_kernel : CORDIC 计算的相似度核 (n_users, n_items)
    # 注意: similarity_kernel 的值域约为 [0, 1]，需要映射到评分区间 [1, 5]
    # 权重应通过最小二乘或经验确定，总和为 1
    # 结果需要 clip 到 [1.0, 5.0]
    R_pred = R_fem  # 占位，需要替换为正确的加权融合
    R_pred = np.clip(R_pred, 1.0, 5.0)
    
    # 仅在观测位置评估
    mae = np.nanmean(np.abs(R_pred[observed] - R_full[observed]))
    rmse = np.sqrt(np.nanmean((R_pred[observed] - R_full[observed])**2))
    
    # 测试集（观测中随机取 20% 作为测试）
    rng = np.random.RandomState(2024)
    test_mask = observed & (rng.rand(n_users, n_items) < 0.20)
    if test_mask.sum() > 0:
        mae_test = np.nanmean(np.abs(R_pred[test_mask] - R_full[test_mask]))
        rmse_test = np.sqrt(np.nanmean((R_pred[test_mask] - R_full[test_mask])**2))
    else:
        mae_test = mae
        rmse_test = rmse
    
    print(f"  训练集 MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    print(f"  测试集 MAE: {mae_test:.4f}, RMSE: {rmse_test:.4f}")
    
    # 不确定性校准
    # 预测区间: [ŷ - z_{α/2}·σ, ŷ + z_{α/2}·σ]
    z_alpha = 1.96
    sigma_pred = np.sqrt(uncertainty)
    coverage = np.mean(
        (R_full[test_mask] >= R_pred[test_mask] - z_alpha * sigma_pred) &
        (R_full[test_mask] <= R_pred[test_mask] + z_alpha * sigma_pred)
    )
    print(f"  95% 预测区间覆盖率: {coverage*100:.1f}%")
    
    t_elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"TGCF 全流程执行完成，耗时: {t_elapsed:.3f} 秒")
    print("=" * 70)


if __name__ == "__main__":
    main()
