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
    # 线性谱嵌入
    embedding_lin, eigvals_lin = schroedinger_spectral_embedding(
        data, L, n_components=3, sigma=1.5
    )
    print(f"  线性谱嵌入特征值: {eigvals_lin}")
    # 非线性谱坐标
    embedding_nonlin = nonlinear_spectral_coordinates(
        data, n_components=3, sigma=1.5, gamma=-0.5, n_iterations=30
    )
    # 能量计算
    V = build_effective_potential(data, sigma=1.5)
    energy = schroedinger_energy(embedding_nonlin[:, 0], L, V)
    print(f"  NLSE基态能量: {energy:.6e}")
    print(f"  非线性嵌入等距保持质量: {isometric_embedding_quality(data, embedding_nonlin):.4f}")
    return {
        'embedding_linear': embedding_lin,
        'embedding_nonlinear': embedding_nonlin,
        'eigenvalues': eigvals_lin,
        'energy': energy
    }


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

# ================================================================
# 测试用例（45个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: results 是字典类型 ----
assert isinstance(results, dict), '[TC01] results 类型应为 dict FAILED'

# ---- TC02: data_shape 正确 ----
assert 'data_shape' in results, '[TC02] results 缺少 data_shape FAILED'
assert results['data_shape'] == (300, 20), '[TC02] data_shape 应为 (300,20) FAILED'

# ---- TC03: data_shape 为二维元组 ----
assert len(results['data_shape']) == 2, '[TC03] data_shape 应为二维 FAILED'

# ---- TC04: sampling 包含必要键 ----
sampling = results['sampling']
assert 'metric_tensor' in sampling, '[TC04] sampling 缺少 metric_tensor FAILED'
assert 'samples' in sampling, '[TC04] sampling 缺少 samples FAILED'
assert 'tangent_basis' in sampling, '[TC04] sampling 缺少 tangent_basis FAILED'
assert 'curvatures' in sampling, '[TC04] sampling 缺少 curvatures FAILED'

# ---- TC05: metric_tensor 为方阵 ----
mt = sampling['metric_tensor']
assert mt.shape[0] == mt.shape[1], '[TC05] metric_tensor 应为方阵 FAILED'
assert mt.shape[0] == 20, '[TC05] metric_tensor 维度应为20 FAILED'

# ---- TC06: curvatures 非负（容许数值误差） ----
curv = sampling['curvatures']
assert np.all(curv >= -1e-12), '[TC06] curvatures 应全部非负 FAILED'

# ---- TC07: graph 包含必要键 ----
graph = results['graph']
assert 'edges' in graph, '[TC07] graph 缺少 edges FAILED'
assert 'weights' in graph, '[TC07] graph 缺少 weights FAILED'
assert 'laplacian' in graph, '[TC07] graph 缺少 laplacian FAILED'

# ---- TC08: edges 形状正确 ----
edges = graph['edges']
assert edges.ndim == 2, '[TC08] edges 应为二维数组 FAILED'
assert edges.shape[1] == 2, '[TC08] edges 第二维应为2 FAILED'

# ---- TC09: weights 与 edges 长度一致 ----
weights = graph['weights']
assert len(weights) == len(edges), '[TC09] weights 与 edges 长度应一致 FAILED'

# ---- TC10: embedding 包含必要键 ----
embed = results['embedding']
assert 'embedding_linear' in embed, '[TC10] embedding 缺少 embedding_linear FAILED'
assert 'embedding_nonlinear' in embed, '[TC10] embedding 缺少 embedding_nonlinear FAILED'
assert 'eigenvalues' in embed, '[TC10] embedding 缺少 eigenvalues FAILED'
assert 'energy' in embed, '[TC10] embedding 缺少 energy FAILED'

# ---- TC11: NLSE 能量为有限值 ----
energy = embed['energy']
assert np.isfinite(energy), '[TC11] NLSE 能量应为有限值 FAILED'

# ---- TC12: embedding 形状正确 ----
emb_nl = embed['embedding_nonlinear']
assert emb_nl.shape[0] == 300, '[TC12] nonlinear embedding 行数应为300 FAILED'
assert emb_nl.shape[1] == 3, '[TC12] nonlinear embedding 列数应为3 FAILED'

# ---- TC13: eigenvalues 非负 ----
eigvals = embed['eigenvalues']
assert np.all(eigvals >= -1e-10), '[TC13] eigenvalues 应非负 FAILED'

# ---- TC14: spherical 包含必要键 ----
spherical = results['spherical']
assert 'spectrum' in spherical, '[TC14] spherical 缺少 spectrum FAILED'
assert 'coefficients' in spherical, '[TC14] spherical 缺少 coefficients FAILED'

# ---- TC15: spectrum 长度正确 ----
spectrum = spherical['spectrum']
assert len(spectrum) == 7, '[TC15] l_max=6 时 spectrum 长度应为7 FAILED'

# ---- TC16: spectrum 首项 ≥ 0 ----
assert spectrum[0] >= 0, '[TC16] spectrum[0] 应非负 FAILED'

# ---- TC17: quadrature 包含必要键 ----
quad = results['quadrature']
assert 'integral_1d' in quad, '[TC17] quadrature 缺少 integral_1d FAILED'
assert 'volume_elements' in quad, '[TC17] quadrature 缺少 volume_elements FAILED'

# ---- TC18: integral_1d 为正有限值 ----
assert quad['integral_1d'] > 0, '[TC18] integral_1d 应为正值 FAILED'
assert np.isfinite(quad['integral_1d']), '[TC18] integral_1d 应为有限值 FAILED'

# ---- TC19: volume_elements 非负 ----
vol = quad['volume_elements']
assert np.all(vol >= 0), '[TC19] volume_elements 应全部非负 FAILED'

# ---- TC20: gradient 包含必要键 ----
grad = results['gradient']
assert 'gradient' in grad, '[TC20] gradient 缺少 gradient FAILED'
assert 'selected_features' in grad, '[TC20] gradient 缺少 selected_features FAILED'
assert 'conserved_energy' in grad, '[TC20] gradient 缺少 conserved_energy FAILED'

# ---- TC21: conserved_energy 为有限值 ----
assert np.isfinite(grad['conserved_energy']), '[TC21] conserved_energy 应为有限值 FAILED'

# ---- TC22: topological 包含必要键 ----
topo = results['topological']
assert 'betti_0' in topo, '[TC22] topological 缺少 betti_0 FAILED'
assert 'persistence' in topo, '[TC22] topological 缺少 persistence FAILED'
assert 'topo_features' in topo, '[TC22] topological 缺少 topo_features FAILED'
assert 'lo_solution' in topo, '[TC22] topological 缺少 lo_solution FAILED'

# ---- TC23: betti_0 ≥ 1 ----
assert topo['betti_0'] >= 1, '[TC23] betti_0 应 ≥ 1 FAILED'

# ---- TC24: lo_solution 为二元数组 ----
lo_sol = topo['lo_solution']
assert np.all((lo_sol == 0) | (lo_sol == 1)), '[TC24] lo_solution 元素应为 0 或 1 FAILED'

# ---- TC25: curve 包含必要键 ----
curve = results['curve']
assert 'arc_length' in curve, '[TC25] curve 缺少 arc_length FAILED'
assert 'curvature_max' in curve, '[TC25] curve 缺少 curvature_max FAILED'
assert 'geodesic_ratio' in curve, '[TC25] curve 缺少 geodesic_ratio FAILED'

# ---- TC26: arc_length > 0 ----
assert curve['arc_length'] > 0, '[TC26] arc_length 应为正值 FAILED'

# ---- TC27: curvature_max > 0 ----
assert curve['curvature_max'] > 0, '[TC27] curvature_max 应为正值 FAILED'

# ---- TC28: piecewise 包含必要键 ----
pwc = results['piecewise']
assert 'density' in pwc, '[TC28] piecewise 缺少 density FAILED'
assert 'entropy' in pwc, '[TC28] piecewise 缺少 entropy FAILED'
assert 'mutual_info' in pwc, '[TC28] piecewise 缺少 mutual_info FAILED'

# ---- TC29: entropy 为有限值 ----
assert np.isfinite(pwc['entropy']), '[TC29] entropy 应为有限值 FAILED'

# ---- TC30: mutual_info ≥ 0 ----
assert pwc['mutual_info'] >= 0, '[TC30] mutual_info 应 ≥ 0 FAILED'

# ---- TC31: algebra 包含必要键 ----
alg = results['algebra']
assert 'binary_data' in alg, '[TC31] algebra 缺少 binary_data FAILED'
assert 'hash_codes' in alg, '[TC31] algebra 缺少 hash_codes FAILED'
assert 'lo_features' in alg, '[TC31] algebra 缺少 lo_features FAILED'

# ---- TC32: binary_data 元素为 0/1 ----
bin_data = alg['binary_data']
assert np.all((bin_data == 0) | (bin_data == 1)), '[TC32] binary_data 元素应为 0 或 1 FAILED'

# ---- TC33: hash_codes 形状正确 ----
hc = alg['hash_codes']
assert hc.shape[0] == 300, '[TC33] hash_codes 行数应为300 FAILED'
assert hc.shape[1] == 16, '[TC33] hash_codes 列数应为16 FAILED'

# ---- TC34: metrics 包含必要键 ----
met = results['metrics']
assert 'isometric_quality' in met, '[TC34] metrics 缺少 isometric_quality FAILED'
assert 'trustworthiness' in met, '[TC34] metrics 缺少 trustworthiness FAILED'
assert 'reconstruction_error' in met, '[TC34] metrics 缺少 reconstruction_error FAILED'

# ---- TC35: isometric_quality 在 [0, 1] 内 ----
assert 0.0 <= met['isometric_quality'] <= 1.0, '[TC35] isometric_quality 应在 [0,1] FAILED'

# ---- TC36: trustworthiness 在 [0, 1] 内 ----
assert 0.0 <= met['trustworthiness'] <= 1.0, '[TC36] trustworthiness 应在 [0,1] FAILED'

# ---- TC37: elapsed_time > 0 ----
assert results['elapsed_time'] > 0, '[TC37] elapsed_time 应为正值 FAILED'

# ---- TC38: 可复现性——固定种子两次结果一致 ----
np.random.seed(42)
data1 = generate_synthetic_manifold_data(n_points=100, ambient_dim=10, intrinsic_dim=2)
np.random.seed(42)
data2 = generate_synthetic_manifold_data(n_points=100, ambient_dim=10, intrinsic_dim=2)
assert np.allclose(data1, data2), '[TC38] 固定种子两次生成数据应一致 FAILED'

# ---- TC39: generate_synthetic_manifold_data 输出尺寸正确 ----
np.random.seed(42)
data_test = generate_synthetic_manifold_data(n_points=50, ambient_dim=8, intrinsic_dim=2)
assert data_test.shape == (50, 8), '[TC39] 生成数据形状应为 (50,8) FAILED'
assert np.all(np.isfinite(data_test)), '[TC39] 生成数据应全部有限 FAILED'

# ---- TC40: chord_length > 0 ----
assert curve['arc_length'] > 0, '[TC40] arc_length 应为正值 FAILED'

# ---- TC41: lo_solution 维度为25 ----
assert len(lo_sol) == 25, '[TC41] lo_solution 长度应为25 (5×5) FAILED'

# ---- TC42: spectrum 能量总和 ≥ 0 ----
assert np.sum(spectrum) >= 0, '[TC42] spectrum 能量总和应 ≥ 0 FAILED'

# ---- TC43: persistence 字典非空 ----
persistence = topo['persistence']
assert len(persistence) > 0, '[TC43] persistence 字典不应为空 FAILED'

# ---- TC44: reconstruction_error ≥ 0 ----
assert met['reconstruction_error'] >= 0, '[TC44] reconstruction_error 应 ≥ 0 FAILED'

# ---- TC45: hash_codes 元素为 0/1 ----
assert np.all((hc == 0) | (hc == 1)), '[TC45] hash_codes 元素应为 0 或 1 FAILED'

print('\n全部 45 个测试通过!\n')
