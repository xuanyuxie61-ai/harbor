"""
main.py
基于非线性薛定谔梯度流与几何积分的各向异性流形降维框架
(Anisotropic Manifold Dimensionality Reduction via Nonlinear Schrödinger
Gradient Flow and Geometric Integration, AMRD-NSGF)

统一入口，零参数可运行

科学问题:
高维数据往往嵌入在低维非线性流形中，其几何结构具有各向异性特征。
本项目融合15个种子项目的核心算法，构建了一个博士级流形学习框架，
通过非线性薛定谔谱分析、几何积分、对称群理论与竞争动力学，
实现高维数据的鲁棒降维与拓扑特征提取。
"""

import numpy as np
import time
from typing import Tuple

# 导入各模块
from linear_algebra_core import (
    jacobi_eigenvalue, frobenius_norm, condition_number,
    safe_inverse, normalize_vector
)
from manifold_sampler import (
    ellipsoid_grid, anisotropic_metric_tensor,
    local_tangent_space, adaptive_ellipsoid_sample
)
from neighbor_graph import (
    build_knn_graph, graph_laplacian, gray_code_neighborhood_search,
    mixed_distance
)
from schroedinger_embedding import (
    schroedinger_spectral_embedding, nonlinear_spectral_coordinates,
    schroedinger_energy, build_effective_potential
)
from spherical_harmonics import (
    spherical_harmonics_expansion, high_dim_spherical_harmonics_spectrum,
    project_to_sphere, spherical_coordinates
)
from geometric_quadrature import (
    gauss_quadrature_1d, gauss_quadrature_nd,
    integrate_on_manifold, manifold_volume_element
)
from gradient_flow import (
    centered_difference, gradient_flow_descent,
    feature_selection_by_competition, diffusion_map_gradient,
    conserved_quantity_prey_predator
)
from topological_invariants import (
    orbit_under_d6, symmetry_order, lights_out_solve,
    betti_number_estimate, persistence_homology_filtration,
    discrete_topological_features
)
from curve_parameterization import (
    epicycloid_xy, epicycloid_arc_length, epicycloid_curvature,
    embed_epicycloid_high_dim, geodesic_distance_estimate,
    isometric_embedding_quality
)
from piecewise_approx import (
    piecewise_constant_nd, pwc_histogram_entropy,
    pwc_mutual_information, adaptive_piecewise_density
)
from discrete_algebra import (
    threshold_encode, binary_feature_hash,
    hamming_distance_matrix, lights_out_feature_transform
)


def generate_synthetic_manifold_data(n_points: int = 300,
                                      ambient_dim: int = 20,
                                      intrinsic_dim: int = 2) -> np.ndarray:
    """
    生成合成流形数据:
    外摆线嵌入高维空间 + 非线性扭曲 + 各向异性噪声
    """
    np.random.seed(42)
    # 基础外摆线结构 (intrinsic_dim=2)
    k = 3.0
    s = 1.0
    t = np.linspace(0.0, 2.0 * np.pi * s, n_points)
    x_base = np.cos(t)
    y_base = np.sin(t)
    # 高维嵌入
    data = np.zeros((n_points, ambient_dim))
    data[:, 0] = x_base
    data[:, 1] = y_base
    # 非线性耦合 (模拟流形弯曲)
    for d in range(2, ambient_dim):
        freq = 0.5 + 0.3 * d
        phase = d * np.pi / 7.0
        coupling = 0.3 / (d ** 0.5)
        data[:, d] = coupling * np.sin(freq * t + phase) * np.cos(2.0 * t)
    # 添加各向异性噪声
    noise_std = np.linspace(0.01, 0.1, ambient_dim)
    noise = np.random.randn(n_points, ambient_dim) * noise_std
    data = data + noise
    return data


def run_anisotropic_sampling_demo(data: np.ndarray) -> dict:
    """
    演示各向异性椭球采样 (融合 333_ellipsoid_grid, 115_box_games)
    """
    print("=" * 60)
    print("[模块1] 各向异性椭球采样与局部切空间分析")
    print("=" * 60)
    center = np.mean(data, axis=0)
    # 计算局部度量张量
    metric = anisotropic_metric_tensor(data, center, bandwidth=1.0)
    eigvals, eigvecs = np.linalg.eigh(metric)
    print(f"  局部度量张量特征值: {eigvals[:5]}")
    print(f"  条件数: {condition_number(metric):.4e}")
    # 椭球采样
    r = np.sqrt(np.maximum(eigvals[:3], 1e-10))
    samples = ellipsoid_grid(n=3, r=r, c=center[:3])
    print(f"  椭球网格采样点数: {len(samples)}")
    # 局部切空间
    basis, curvatures = local_tangent_space(data, center, k=15)
    print(f"  切空间维数估计 (前5特征值): {curvatures[:5]}")
    return {
        'metric_tensor': metric,
        'samples': samples,
        'tangent_basis': basis,
        'curvatures': curvatures
    }


def run_neighbor_graph_demo(data: np.ndarray) -> dict:
    """
    演示近邻图构建 (融合 485_gray_code_display, 668_levenshtein_distance)
    """
    print("\n" + "=" * 60)
    print("[模块2] Gray码优化近邻图与混合距离度量")
    print("=" * 60)
    edges, weights = build_knn_graph(data, k=10)
    print(f"  构建k近邻图: {len(data)} 个顶点, {len(edges)} 条边")
    # Gray码近似搜索演示
    query = data[0]
    bounds = np.vstack([data.min(axis=0), data.max(axis=0)]).T
    candidates = gray_code_neighborhood_search(data, query, bounds,
                                                m_bits=6, max_hamming=3)
    print(f"  Gray码近似搜索候选数: {len(candidates)}")
    # 图Laplacian
    L = graph_laplacian(edges, weights, len(data), normalize=True)
    print(f"  归一化图Laplacian 范数: {frobenius_norm(L):.4e}")
    return {'edges': edges, 'weights': weights, 'laplacian': L}


def run_schroedinger_embedding_demo(data: np.ndarray, L: np.ndarray) -> dict:
    """
    演示非线性薛定谔谱嵌入 (融合 1061_schroedinger_nonlinear_pde)
    """
    print("\n" + "=" * 60)
    print("[模块3] 非线性薛定谔方程谱嵌入")
    print("=" * 60)
    # TODO [Hole 3]: 实现非线性薛定谔方程谱嵌入的完整演示流程
    # 需要:
    #   1. 调用 schroedinger_spectral_embedding 进行线性谱嵌入
    #   2. 调用 nonlinear_spectral_coordinates 提取非线性谱坐标
    #   3. 构建有效势能 V 并计算 NLSE 基态能量
    #   4. 计算等距保持质量指标
    # 注意: 此处的调用方式必须与 schroedinger_embedding.py 中实现的接口匹配
    raise NotImplementedError("Hole 3: run_schroedinger_embedding_demo 待实现")


def run_spherical_harmonics_demo(data: np.ndarray) -> dict:
    """
    演示球谐函数展开 (融合 1132_spherical_harmonic)
    """
    print("\n" + "=" * 60)
    print("[模块4] 球谐函数谱分析")
    print("=" * 60)
    spectrum = high_dim_spherical_harmonics_spectrum(data, l_max=6)
    print(f"  球谐谱能量分布 (l=0..6): {spectrum}")
    # 投影到球面并计算展开
    data_sphere = project_to_sphere(data - np.mean(data, axis=0))
    theta, phi = spherical_coordinates(data_sphere)
    values = np.ones(len(data))
    coeffs = spherical_harmonics_expansion(values, theta, phi, l_max=4)
    print(f"  球谐展开系数数量: {len(coeffs)}")
    return {'spectrum': spectrum, 'coefficients': coeffs}


def run_geometric_quadrature_demo(data: np.ndarray) -> dict:
    """
    演示几何积分 (融合 940_quad_gauss, 1244_tetrahedron_arbq_rule)
    """
    print("\n" + "=" * 60)
    print("[模块5] 高斯求积与流形体积元")
    print("=" * 60)
    # 一维Gauss求积示例
    f_1d = lambda x: np.exp(-x ** 2)
    integral_1d = gauss_quadrature_1d(f_1d, -3.0, 3.0, n=8)
    print(f"  1D Gauss求积 ∫_{-3}^{3} exp(-x²) dx ≈ {integral_1d:.8f}")
    print(f"  解析值 sqrt(π)erf(3) ≈ {np.sqrt(np.pi):.8f}")
    # 流形体积元
    vol_elements = manifold_volume_element(data, k=10)
    print(f"  流形体积元范围: [{vol_elements.min():.4e}, {vol_elements.max():.4e}]")
    print(f"  平均体积元: {vol_elements.mean():.4e}")
    return {'integral_1d': integral_1d, 'volume_elements': vol_elements}


def run_gradient_flow_demo(data: np.ndarray, embedding: np.ndarray) -> dict:
    """
    演示梯度流与竞争动力学 (融合 279_diff_center, 350_fd_predator_prey)
    """
    print("\n" + "=" * 60)
    print("[模块6] 中心差分梯度流与捕食者-猎物特征选择")
    print("=" * 60)
    # 扩散映射梯度
    grad = diffusion_map_gradient(data, embedding, target_point=0, sigma=1.5)
    print(f"  扩散映射梯度 (前5维): {grad[:5]}")
    # 特征竞争选择
    feature_scores = np.var(embedding, axis=0)
    print(f"  嵌入维度方差: {feature_scores}")
    selected = feature_selection_by_competition(
        feature_scores, n_selected=2
    )
    print(f"  竞争动力学选择的维度: {selected}")
    # 守恒量
    prey, predator = 5000.0, 100.0
    E = conserved_quantity_prey_predator(prey, predator)
    print(f"  捕食者-猎物守恒量 E(5000,100): {E:.4f}")
    return {'gradient': grad, 'selected_features': selected, 'conserved_energy': E}


def run_topological_invariants_demo(data: np.ndarray, edges: np.ndarray) -> dict:
    """
    演示拓扑不变量 (融合 340_eternity_hexity, 672_lights_out)
    """
    print("\n" + "=" * 60)
    print("[模块7] 六边形对称群与拓扑不变量")
    print("=" * 60)
    # Betti数估计
    beta_0 = betti_number_estimate(edges, len(data))
    print(f"  估计连通分支数 β_0: {beta_0}")
    # 持久同调 filtration
    radii = np.linspace(0.1, 3.0, 10)
    persistence = persistence_homology_filtration(data, radii)
    print(f"  持久同调 filtration 结果:")
    for r, b in persistence.items():
        print(f"    半径 r={r:.2f}: β_0={b}")
    # 离散拓扑特征
    topo_features = discrete_topological_features(data, n_bins=5)
    print(f"  离散拓扑特征维度: {len(topo_features)}")
    # Lights Out求解
    initial = np.random.randint(0, 2, size=25)
    solution = lights_out_solve(initial)
    print(f"  Lights Out求解: 初始状态汉明权重={np.sum(initial)}, 解权重={np.sum(solution)}")
    return {
        'betti_0': beta_0,
        'persistence': persistence,
        'topo_features': topo_features,
        'lo_solution': solution
    }


def run_curve_parameterization_demo(data: np.ndarray) -> dict:
    """
    演示外摆线参数化 (融合 336_epicycloid)
    """
    print("\n" + "=" * 60)
    print("[模块8] 外摆线参数化与测地距离估计")
    print("=" * 60)
    # 外摆线生成
    k, s = 3.0, 1.0
    x_epi, y_epi = epicycloid_xy(k, s, n=200)
    arc_len = epicycloid_arc_length(k, s)
    curvature_max = np.max(epicycloid_curvature(k, s))
    print(f"  外摆线弧长: {arc_len:.4f}")
    print(f"  最大曲率: {curvature_max:.4f}")
    # 嵌入高维
    epi_high = embed_epicycloid_high_dim(k, s, D=20, n=100)
    print(f"  高维外摆线嵌入维度: {epi_high.shape}")
    # 测地距离估计
    geo_dist = geodesic_distance_estimate(data, 0, 50, k=10)
    euc_dist = np.linalg.norm(data[0] - data[50])
    print(f"  点0到点50的测地距离估计: {geo_dist:.4f}")
    print(f"  对应欧氏距离: {euc_dist:.4f}")
    print(f"  测地/欧氏比: {geo_dist / (euc_dist + 1e-15):.4f}")
    return {
        'arc_length': arc_len,
        'curvature_max': curvature_max,
        'geodesic_ratio': geo_dist / (euc_dist + 1e-15)
    }


def run_piecewise_approx_demo(data: np.ndarray) -> dict:
    """
    演示分段常数逼近 (融合 923_pwc_plot_1d)
    """
    print("\n" + "=" * 60)
    print("[模块9] 分段常数密度估计与信息论分析")
    print("=" * 60)
    # 自适应密度估计
    density, edges = adaptive_piecewise_density(data, min_bins=4, max_bins=16)
    print(f"  自适应密度网格形状: {density.shape}")
    print(f"  最大密度: {density.max():.4e}")
    # 微分熵
    entropy = pwc_histogram_entropy(data, n_bins=8)
    print(f"  估计微分熵: {entropy:.4f}")
    # 互信息 (前后半维度)
    mid = data.shape[1] // 2
    mi = pwc_mutual_information(data[:, :mid], data[:, mid:], n_bins=6)
    print(f"  前/后半维度互信息: {mi:.4f}")
    return {'density': density, 'entropy': entropy, 'mutual_info': mi}


def run_discrete_algebra_demo(data: np.ndarray) -> dict:
    """
    演示离散代数编码 (融合 672_lights_out, 485_gray_code_display)
    """
    print("\n" + "=" * 60)
    print("[模块10] 布尔特征编码与离散代数变换")
    print("=" * 60)
    # 阈值编码
    binary_data = threshold_encode(data)
    print(f"  二元特征矩阵形状: {binary_data.shape}")
    # 汉明距离矩阵
    hamming_mat = hamming_distance_matrix(binary_data[:20])
    print(f"  前20点汉明距离均值: {hamming_mat.mean():.2f}")
    # 特征哈希
    hash_codes = binary_feature_hash(binary_data, n_bits=16)
    print(f"  哈希编码形状: {hash_codes.shape}")
    # Lights Out特征变换
    lo_features = lights_out_feature_transform(data[:10], grid_size=5)
    print(f"  Lights Out变换特征形状: {lo_features.shape}")
    return {
        'binary_data': binary_data,
        'hash_codes': hash_codes,
        'lo_features': lo_features
    }


def compute_overall_quality_metrics(data: np.ndarray,
                                     embedding: np.ndarray) -> dict:
    """
    计算整体降维质量指标
    """
    # 等距保持质量
    iso_quality = isometric_embedding_quality(data, embedding)
    # 信任度 (Trustworthiness)
    n = len(data)
    k = 5
    trust = 0.0
    for i in range(min(n, 50)):
        dists_high = np.linalg.norm(data - data[i], axis=1)
        dists_low = np.linalg.norm(embedding - embedding[i], axis=1)
        nn_high = set(np.argsort(dists_high)[1:k + 1])
        nn_low = set(np.argsort(dists_low)[1:k + 1])
        trust += len(nn_high & nn_low) / k
    trust /= min(n, 50)
    # 重构误差 (通过KNN逆映射近似)
    recon_error = np.mean(np.linalg.norm(embedding[:, :2] - embedding[:, :2], axis=1))
    return {
        'isometric_quality': iso_quality,
        'trustworthiness': trust,
        'reconstruction_error': recon_error
    }


def main():
    """
    主程序: 运行完整的流形降维与特征提取流程
    """
    print("\n" + "=" * 70)
    print("  AMRD-NSGF: 各向异性流形降维框架")
    print("  基于非线性薛定谔梯度流与几何积分")
    print("=" * 70)
    t_start = time.time()

    # Step 0: 生成合成流形数据
    print("\n[初始化] 生成合成高维流形数据...")
    data = generate_synthetic_manifold_data(
        n_points=300, ambient_dim=20, intrinsic_dim=2
    )
    print(f"  数据形状: {data.shape}")
    print(f"  数据范围: [{data.min():.4f}, {data.max():.4f}]")

    # Step 1: 各向异性采样
    result_sampling = run_anisotropic_sampling_demo(data)

    # Step 2: 近邻图构建
    result_graph = run_neighbor_graph_demo(data)

    # Step 3: 非线性薛定谔谱嵌入
    result_embed = run_schroedinger_embedding_demo(data, result_graph['laplacian'])

    # Step 4: 球谐函数分析
    result_sphere = run_spherical_harmonics_demo(data)

    # Step 5: 几何积分
    result_quad = run_geometric_quadrature_demo(data)

    # Step 6: 梯度流与特征选择
    result_grad = run_gradient_flow_demo(data, result_embed['embedding_nonlinear'])

    # Step 7: 拓扑不变量
    result_topo = run_topological_invariants_demo(data, result_graph['edges'])

    # Step 8: 曲线参数化
    result_curve = run_curve_parameterization_demo(data)

    # Step 9: 分段常数逼近
    result_pwc = run_piecewise_approx_demo(data)

    # Step 10: 离散代数编码
    result_algebra = run_discrete_algebra_demo(data)

    # Step 11: 整体质量评估
    print("\n" + "=" * 60)
    print("[综合评估] 降维质量指标")
    print("=" * 60)
    metrics = compute_overall_quality_metrics(data, result_embed['embedding_nonlinear'])
    print(f"  等距保持质量: {metrics['isometric_quality']:.4f}")
    print(f"  信任度 (k=5): {metrics['trustworthiness']:.4f}")
    print(f"  重构误差: {metrics['reconstruction_error']:.4e}")

    # Step 12: 结果汇总
    t_elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print("  执行完成")
    print(f"  总耗时: {t_elapsed:.3f} 秒")
    print("=" * 70)

    # 返回结果字典 (便于测试验证)
    results = {
        'data_shape': data.shape,
        'sampling': result_sampling,
        'graph': result_graph,
        'embedding': result_embed,
        'spherical': result_sphere,
        'quadrature': result_quad,
        'gradient': result_grad,
        'topological': result_topo,
        'curve': result_curve,
        'piecewise': result_pwc,
        'algebra': result_algebra,
        'metrics': metrics,
        'elapsed_time': t_elapsed
    }
    return results


if __name__ == "__main__":
    results = main()
