#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time


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
    rng = np.random.RandomState(seed)
    density = np.clip(density, 0.05, 0.50)
    
    d = 8
    mu_global = 3.0
    sigma_bias = 0.6
    sigma_noise = 0.4
    

    P = rng.randn(n_users, d)
    P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-12
    

    Q = rng.randn(n_items, d)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12
    

    b_u = rng.normal(0.0, sigma_bias, n_users)
    b_i = rng.normal(0.0, sigma_bias * 0.5, n_items)
    

    R_full = mu_global + b_u[:, None] + b_i[None, :] + P @ Q.T * 2.0
    R_full += rng.normal(0.0, sigma_noise, (n_users, n_items))
    R_full = np.clip(R_full, 1.0, 5.0)
    

    mask = rng.rand(n_users, n_items) < density
    R_obs = np.where(mask, R_full, np.nan)
    

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
    



    print("\n[阶段 1] 生成具有物理结构的合成评分数据...")
    R_obs, R_full, P_true, Q_true, item_metadata = generate_synthetic_data()
    n_users, n_items = R_obs.shape
    observed = ~np.isnan(R_obs)
    print(f"  用户数量: {n_users}, 物品数量: {n_items}")
    print(f"  观测密度: {observed.mean()*100:.2f}%")
    



    print("\n[阶段 2] 截断正态分布建模有界评分...")


    trunc_model = TruncatedNormalRatingModel(mu=3.0, sigma=1.2, a=1.0, b=5.0)
    observed_ratings = R_obs[observed]
    trunc_mean = trunc_model.mean()
    trunc_var = trunc_model.variance()
    print(f"  截断正态均值 E[X] = {trunc_mean:.4f}")
    print(f"  截断正态方差 Var[X] = {trunc_var:.4f}")
    samples = trunc_model.sample(1000)
    print(f"  蒙特卡洛样本均值: {samples.mean():.4f}, 样本方差: {samples.var():.4f}")
    



    print("\n[阶段 3] 基于 2D 热传导方程的评分信息扩散...")





    heat = HeatDiffusionRecommender(alpha=0.05, n_steps=8)
    R_diffused = heat.diffuse(R_obs)
    



    print("\n[阶段 4] 3D 有限元潜空间插值...")



    fem = FemEmbeddingInterpolator(latent_dim=6)
    R_fem = fem.interpolate(R_diffused)
    



    print("\n[阶段 5] 准周期 ODE 偏好动力学演化...")



    qpd = QuasiperiodicPreferenceDynamics()
    t_eval = np.linspace(0, 10, 50)
    y_exact = qpd.exact_solution(t_eval)
    y_ode = qpd.integrate_ode(t_eval)
    err_ode = np.linalg.norm(y_exact - y_ode, axis=1).mean()
    print(f"  准周期 ODE 数值解与精确解平均 L2 误差: {err_ode:.2e}")

    temporal_factor = 1.0 + 0.05 * y_ode[-1, 0]
    R_temporal = R_fem * temporal_factor
    R_temporal = np.clip(R_temporal, 1.0, 5.0)
    



    print("\n[阶段 6] CORDIC 算法快速近似计算...")





    cordic = CordicEngine(n_iter=24)
    angle = np.pi / 6.0
    cos_c, sin_c = cordic.cossin(angle)
    print(f"  CORDIC cos(π/6) = {cos_c:.10f}, sin(π/6) = {sin_c:.10f}")
    print(f"  真值           = {np.cos(angle):.10f},      = {np.sin(angle):.10f}")
    



    sigma_kernel = 1.5
    similarity_kernel = cordic.compute_similarity_kernel(P_true, Q_true, sigma_kernel)
    



    print("\n[阶段 7] Laguerre 广义多项式混沌 (gPC) 不确定性量化...")








    lpc = LaguerrePolynomialChaos(max_degree=5)
    beta = 0.3
    exp_table = lpc.exponential_product_table(beta)
    print(f"  Laguerre 指数加权积矩阵范数: {np.linalg.norm(exp_table):.4f}")

    uncertainty = lpc.propagate_uncertainty(observed_ratings, beta)
    print(f"  评分预测不确定性 (gPC): {uncertainty:.4f}")
    



    print("\n[阶段 8] 3D Hilbert 空间填充曲线 LSH...")


    hilbert = HilbertLSH(order=4)
    user_hashes = hilbert.hash_vectors(P_true[:, :3])
    item_hashes = hilbert.hash_vectors(Q_true[:, :3])
    nn_pairs = hilbert.approximate_nn(user_hashes, item_hashes, top_k=5)
    print(f"  Hilbert LSH 近邻对数量: {len(nn_pairs)}")
    



    print("\n[阶段 9] Levenshtein 编辑距离冷启动处理...")





    string_sim = StringSimilarityEngine()

    cold_item_sim = string_sim.compute_similarity_matrix(item_metadata[:10])
    print(f"  冷启动物品元数据相似度矩阵范数: {np.linalg.norm(cold_item_sim):.4f}")
    



    print("\n[阶段 10] 鲁棒高斯消元求解隐因子系统...")




    solver = RobustGaussianSolver()

    rng_test = np.random.RandomState(123)
    M = rng_test.randn(20, 20)
    A_test = M.T @ M + 2.0 * np.eye(20)
    b_test = np.ones(20)
    x_lu = solver.solve_plu(A_test, b_test)
    x_ge = solver.solve_gauss(A_test, b_test)
    x_np = np.linalg.solve(A_test, b_test)
    print(f"  PLU 与 Gauss 消元解差异: {np.linalg.norm(x_lu - x_ge):.2e}")
    print(f"  与 NumPy 参考解差异: {np.linalg.norm(x_lu - x_np):.2e}")
    det_A = solver.determinant(A_test)
    print(f"  测试矩阵行列式: {det_A:.4e}")
    



    print("\n[阶段 11] 三角剖分边界检测识别利基社区...")



    boundary_det = TriangulationBoundaryDetector()

    boundary_nodes = boundary_det.detect_boundary(P_true[:, :2])
    print(f"  检测到的边界节点数: {len(boundary_nodes)} / {n_users}")
    



    print("\n[阶段 12] 几何蒙特卡洛采样与积分...")




    geom = GeometricSampler(n_samples=5000)

    circle_samples = geom.sample_unit_circle(500)
    monomial_integral = geom.circle_monomial_integral([2, 2])
    print(f"  圆上 x²y² 的精确积分值: {monomial_integral:.6f}")
    mc_estimate = geom.monte_carlo_circle_integral(lambda x,y: (x**2)*(y**2), 10000)
    print(f"  蒙特卡洛估计 (N=10000): {mc_estimate:.6f}")
    

    mu_dist, var_dist = geom.positive_circle_distance_stats(2000)
    print(f"  正象限圆上随机点距离均值: {mu_dist:.4f}, 方差: {var_dist:.4f}")
    

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
    



    print("\n[阶段 13] 聚合统计与系统评估...")

    agg = AggregateStatistics()

    groups = agg.group_users_by_embedding(P_true, n_clusters=4)
    group_stats = agg.compute_group_statistics(R_temporal, groups)
    print(f"  用户分组数: {len(group_stats)}")
    for gid, stats in group_stats.items():
        print(f"    组 {gid}: 均值={stats['mean']:.3f}, 标准差={stats['std']:.3f}, "
              f"最小值={stats['min']:.3f}, 最大值={stats['max']:.3f}")
    



    print("\n[阶段 14] 最终预测合成与精度评估...")













    R_pred = R_fem
    R_pred = np.clip(R_pred, 1.0, 5.0)
    

    mae = np.nanmean(np.abs(R_pred[observed] - R_full[observed]))
    rmse = np.sqrt(np.nanmean((R_pred[observed] - R_full[observed])**2))
    

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
