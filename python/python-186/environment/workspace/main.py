"""
main.py
社交网络传播动力学博士级合成项目入口

科学问题:
    基于复杂网络拓扑、空间异质性与延迟反馈的多模态耦合传播动力学模型

    本项目融合15个种子项目的核心算法，构建一个完整的
    社交网络信息-流行病耦合传播模拟与参数校准系统。
"""

import os
import sys
import numpy as np

# 添加项目目录到路径
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

from network_topology import (
    construct_social_network, floyd_warshall, network_efficiency,
    betweenness_centrality, power_method_eigenvector, clustering_coefficient,
    degree_distribution
)
from spatial_mesh import (
    generate_2d_triangular_mesh, refine_mesh_midpoint, fem_sample_on_mesh,
    spatial_diffusion_operator
)
from sparse_algebra import SparseCCS
from epidemic_dynamics import (
    seaihr_ode_rhs, coupled_info_epidemic_rhs, rk4_integrate,
    dde_rk4_integrate, compute_reproduction_number
)
from propagation_front import propagation_front_simulation
from parameter_calibration import (
    StreamingStatistics, bisection_root_finder, calibrate_beta_target,
    maximum_likelihood_estimation, akaike_information_criterion
)
from stochastic_contact import (
    circle_positive_distance_monte_carlo, geometric_contact_rate,
    magic4_test_matrix, percolation_threshold_estimate
)
from data_io import (
    generate_helical_point_cloud, normalize_features,
    write_xyz_data, read_xyz_data, compute_pca_features
)
from utils import check_numerical_stability, entropy, kl_divergence


def main():
    print("=" * 70)
    print("  社交网络传播动力学: 多模态耦合传播模拟系统")
    print("  Social Network Propagation Dynamics: Multi-modal Coupled Model")
    print("=" * 70)

    # ==================================================================
    # 阶段1: 社交网络拓扑构建与分析
    # ==================================================================
    print("\n[阶段1] 社交网络拓扑构建与分析")
    print("-" * 50)

    n_nodes = 100
    adj = construct_social_network(n_nodes, community_structure=True, seed=42)

    # 稀疏矩阵表示
    sparse_adj = SparseCCS.from_dense(adj)

    # 全源最短路径 (Floyd-Warshall)
    dist = floyd_warshall(adj)

    # 网络效率
    eff = network_efficiency(dist)
    print(f"  网络全局效率: {eff:.4f}")

    # 介数中心性
    bc = betweenness_centrality(adj)
    print(f"  介数中心性范围: [{bc.min():.4f}, {bc.max():.4f}]")

    # PageRank-like中心性 (幂法)
    lambda_max, pr_vec = power_method_eigenvector(adj)
    print(f"  主特征值 (PageRank): {lambda_max:.4f}")
    print(f"  中心性最高节点: {np.argmax(pr_vec)} (得分: {pr_vec.max():.4f})")

    # 聚类系数
    cc = clustering_coefficient(adj)
    print(f"  平均聚类系数: {cc.mean():.4f}")

    # 度分布
    degrees, pk = degree_distribution(adj)
    print(f"  平均度: {degrees.mean():.2f}")
    print(f"  度分布熵: {entropy(pk):.4f}")

    # 图拉普拉斯稀疏矩阵
    laplacian = SparseCCS.network_laplacian(adj)
    lambda_lap, _ = laplacian.power_iteration_sparse()
    print(f"  图拉普拉斯谱半径: {lambda_lap:.4f}")

    # ==================================================================
    # 阶段2: 空间异质性建模
    # ==================================================================
    print("\n[阶段2] 空间异质性建模 (有限元方法)")
    print("-" * 50)

    # 生成2D三角网格
    nodes, elements = generate_2d_triangular_mesh(8, 8, 0.0, 1.0, 0.0, 1.0)
    print(f"  初始网格: {nodes.shape[0]} 节点, {elements.shape[0]} 单元")

    # 网格细化 (1次中点细分)
    nodes_ref, elements_ref = refine_mesh_midpoint(nodes, elements)
    print(f"  细化后网格: {nodes_ref.shape[0]} 节点, {elements_ref.shape[0]} 单元")

    # 在节点上定义空间传播风险场
    # 风险场: 高斯型空间异质性
    risk_field = np.exp(-((nodes_ref[:, 0] - 0.5)**2 + (nodes_ref[:, 1] - 0.5)**2) / 0.1)

    # FEM采样
    sample_pts = np.random.rand(50, 2)
    sampled_risk = fem_sample_on_mesh(nodes_ref, elements_ref, risk_field, sample_pts)
    print(f"  空间风险场采样均值: {sampled_risk.mean():.4f}")

    # 空间扩散算子
    K_stiff = spatial_diffusion_operator(nodes_ref, elements_ref)
    print(f"  刚度矩阵条件数估计: {np.linalg.cond(K_stiff):.2e}")

    # ==================================================================
    # 阶段3: 随机接触模型
    # ==================================================================
    print("\n[阶段3] 随机接触模型与几何概率")
    print("-" * 50)

    # 几何概率: 单位圆上两点平均距离
    mean_dist, var_dist = circle_positive_distance_monte_carlo(n_samples=5000)
    print(f"  单位圆正象限平均距离: {mean_dist:.4f} (方差: {var_dist:.4f})")

    # 几何接触率
    contact_rates = geometric_contact_rate(n_nodes, activity_distribution='power_law')
    print(f"  平均接触率: {contact_rates.mean():.4f}")

    # 渗流阈值估计
    p_c = percolation_threshold_estimate(50, n_realizations=20)
    print(f"  渗流阈值估计: {p_c:.4f}")

    # 幻方测试矩阵
    magic_matrix = magic4_test_matrix(8)
    magic_sum = np.sum(magic_matrix[0, :])
    print(f"  8阶幻方幻和: {magic_sum} (理论值: {8*(64+1)//2})")

    # ==================================================================
    # 阶段4: 流行病-信息耦合动力学
    # ==================================================================
    print("\n[阶段4] 流行病-信息耦合传播动力学")
    print("-" * 50)

    # 基础参数
    N_pop = 100000.0
    params = {
        'N': N_pop,
        'beta_0': 0.8,
        'eta_A': 0.5,
        'sigma': 1.0 / 5.2,       # 潜伏期 ~5.2天
        'p_sym': 0.7,
        'gamma_A': 1.0 / 7.0,     # 无症状恢复 ~7天
        'gamma_I': 1.0 / 10.0,    # 有症状恢复 ~10天
        'alpha_H': 0.05,          # 住院率
        'gamma_H': 1.0 / 14.0,    # 住院恢复 ~14天
        'mu': 0.02,               # 住院死亡率
        'omega': 1.0 / 180.0,     # 免疫丧失 ~180天
        'tau': 7.0,               # 信息延迟
        'k_beta': 0.3,
        'beta_info': 2.0,
        'n_info': 9.65,
        'gamma_info': 1.0
    }

    # 初始条件
    I0 = 100.0
    E0 = 200.0
    A0 = 50.0
    S0 = N_pop - I0 - E0 - A0
    H0 = 10.0
    R0_pop = 0.0
    D0 = 0.0
    I_info0 = 0.0

    y0 = np.array([S0, E0, A0, I0, H0, R0_pop, D0, I_info0], dtype=np.float64)

    # 历史函数
    def history_func(t):
        return y0.copy()

    # DDE积分
    t_span = (0.0, 120.0)
    dt = 0.5

    t_hist, y_hist = dde_rk4_integrate(
        coupled_info_epidemic_rhs, y0, t_span, dt, params, history_func
    )

    final_state = y_hist[-1, :]
    print(f"  模拟时长: {t_span[1]} 天")
    print(f"  最终状态 (S,E,A,I,H,R,D,Info):")
    print(f"    S={final_state[0]:.1f}, E={final_state[1]:.1f}, A={final_state[2]:.1f}")
    print(f"    I={final_state[3]:.1f}, H={final_state[4]:.1f}, R={final_state[5]:.1f}")
    print(f"    D={final_state[6]:.1f}, I_info={final_state[7]:.4f}")

    # 计算R0
    params['beta'] = params['beta_0']
    r0 = compute_reproduction_number(params)
    print(f"  基本再生数 R_0: {r0:.4f}")

    # ==================================================================
    # 阶段5: 传播前沿高分辨率模拟 (DG方法)
    # ==================================================================
    print("\n[阶段5] 传播前沿高分辨率模拟 (间断Galerkin)")
    print("-" * 50)

    x_dg, u_dg, t_dg = propagation_front_simulation(
        initial_spread=0.15, diffusion_coeff=0.005, final_time=1.5
    )
    print(f"  DG网格: {x_dg.shape[0]} 节点 x {x_dg.shape[1]} 单元")
    print(f"  最终时间步: {t_dg[-1]:.4f}")
    print(f"  解范围: [{u_dg.min():.4f}, {u_dg.max():.4f}]")

    # ==================================================================
    # 阶段6: 参数校准
    # ==================================================================
    print("\n[阶段6] 参数校准与统计推断")
    print("-" * 50)

    # 流式统计
    stats = StreamingStatistics()
    for val in y_hist[:, 3]:  # I compartment
        stats.update(float(val))
    s = stats.get_stats()
    print(f"  感染人数统计: mean={s['mean']:.1f}, std={s['std']:.1f}")

    # 二分法校准 beta
    target_r0 = 2.5
    beta_cal = calibrate_beta_target(target_r0, params, None, tol=1e-5)
    print(f"  目标 R_0={target_r0} -> 校准 beta={beta_cal:.6f}")

    # 验证
    params_verify = params.copy()
    params_verify['beta'] = beta_cal
    r0_verify = compute_reproduction_number(params_verify)
    print(f"  验证 R_0={r0_verify:.4f}")

    # ==================================================================
    # 阶段7: 多维数据I/O与特征分析
    # ==================================================================
    print("\n[阶段7] 多维数据I/O与特征分析")
    print("-" * 50)

    # 生成螺旋点云
    point_cloud = generate_helical_point_cloud(n_points=200)
    print(f"  生成点云: {point_cloud.shape[0]} 点")

    # 写入/读取XYZ
    tmp_file = os.path.join(project_dir, "temp_pointcloud.xyz")
    write_xyz_data(tmp_file, point_cloud)
    pc_read = read_xyz_data(tmp_file)
    print(f"  XYZ I/O 验证: 写入{point_cloud.shape[0]}点, 读取{pc_read.shape[0]}点")

    # 特征归一化
    norm_pc, norm_params = normalize_features(point_cloud, method='zscore')
    print(f"  标准化后均值: {norm_pc.mean(axis=0)}")
    print(f"  标准化后方差: {np.var(norm_pc, axis=0)}")

    # PCA
    projected, ratio = compute_pca_features(point_cloud, n_components=2)
    print(f"  PCA前2主成分方差比: [{ratio[0]:.4f}, {ratio[1]:.4f}]")

    # 清理临时文件
    if os.path.exists(tmp_file):
        os.remove(tmp_file)

    # ==================================================================
    # 阶段8: 综合指标与稳定性检查
    # ==================================================================
    print("\n[阶段8] 综合指标与数值稳定性检查")
    print("-" * 50)

    checks = [
        ("邻接矩阵", adj),
        ("距离矩阵", dist),
        ("风险场", risk_field),
        ("DG解", u_dg),
        ("感染历史", y_hist[:, 3])
    ]

    all_stable = True
    for name, arr in checks:
        stable = check_numerical_stability(arr, name=name)
        all_stable = all_stable and stable

    if all_stable:
        print("  所有数值检查通过 ✓")
    else:
        print("  部分数值检查未通过 ✗")

    print("\n" + "=" * 70)
    print("  模拟完成。所有模块运行正常。")
    print("=" * 70)


if __name__ == "__main__":
    main()
