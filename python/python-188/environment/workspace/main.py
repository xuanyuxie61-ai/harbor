"""
基于多尺度物理-化学动力学约束的语义嵌入空间分析与优化系统
=====================================================================

统一入口文件。零参数可运行，执行完整的语义嵌入分析流程。

本系统融合15个种子项目的核心算法，围绕"数据科学：自然语言处理语义嵌入"
领域，构建了一个前沿博士级科学计算平台。

运行方式:
    python main.py
"""

import sys
import numpy as np

# 导入所有模块
from embedding_bases import SemanticEmbeddingBases
from fem_projection import FEM2DSemanticProjection
from heat_diffusion import SemanticHeatDiffusion
from reaction_dynamics import SemanticReactionDynamics, MultiConceptReactionNetwork
from period_analysis import VanDerPolSemanticOscillator, PredatorPreySemanticCycle
from space_quantization import SemanticSpaceQuantization
from path_optimizer import SemanticPathOptimizer, generate_semantic_distance_matrix
from subset_selector import SemanticSubsetSelector
from fractal_analysis import MandelbrotSemanticBoundary, IFSSemanticTransformer
from structured_decomposition import SemanticStructuredDecomposition
from sampling_init import SemanticSpaceSampler
from numerical_verification import SteinerbergerVerifier
from robust_utils import RobustNumericUtils


def section(title: str):
    """打印章节标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_embedding_bases_analysis():
    """运行语义嵌入正交基分析"""
    section("1. 语义嵌入正交基分析 (Helmholtz/Bessel)")

    bases = SemanticEmbeddingBases(radius=1.0, max_mode_m=4, max_mode_n=3)
    print(f"正交基总数: {bases.num_bases}")

    r_grid = np.linspace(0.01, 1.0, 25)
    theta_grid = np.linspace(0, 2 * np.pi, 50)
    R, T = np.meshgrid(r_grid, theta_grid)
    r_flat = R.flatten()
    theta_flat = T.flatten()

    # 构造高斯型语义密度场
    semantic_field = np.exp(-((r_flat - 0.5) ** 2 + (np.sin(theta_flat)) ** 2) / 0.1)
    coeffs = bases.project_semantic_vector(semantic_field, r_flat, theta_flat)
    reconstructed = bases.reconstruct_semantic_field(coeffs, r_flat, theta_flat)
    error = np.linalg.norm(semantic_field - reconstructed) / np.linalg.norm(semantic_field)

    print(f"谱系数 L2 范数: {np.linalg.norm(coeffs):.6f}")
    print(f"重构相对误差: {error:.6e}")

    gram = bases.basis_orthogonality_check(r_flat, theta_flat)
    off_diag_max = np.max(np.abs(gram - np.eye(bases.num_bases)))
    print(f"正交性偏差: {off_diag_max:.6e}")

    return {'bases': bases, 'coeffs': coeffs, 'error': error}


def run_fem_projection():
    """运行有限元语义密度投影"""
    section("2. 语义密度有限元L2投影")

    fem = FEM2DSemanticProjection(xl=0.0, xr=1.0, yb=0.0, yt=1.0, nx=9, ny=9)
    print(f"网格节点数: {fem.node_num}")
    print(f"三角形单元数: {fem.element_num}")

    def semantic_density(x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y) + x

    U = fem.project(semantic_density)
    errors = fem.compute_l2_error(U, semantic_density)

    print(f"投影解范围: [{U.min():.6f}, {U.max():.6f}]")
    print(f"L2 相对误差: {errors['relative_error']:.6e}")
    print(f"|U| = {errors['u_norm']:.6e}")
    print(f"|W| = {errors['w_norm']:.6e}")
    print(f"|U-W| = {errors['uw_norm']:.6e}")

    return {'fem': fem, 'solution': U, 'errors': errors}


def run_heat_diffusion():
    """运行语义热扩散模拟"""
    section("3. 语义信息稳态热扩散")

    # TODO [Hole 3a]: 调用热扩散求解器并验证守恒律
    # 1. 创建 SemanticHeatDiffusion 实例
    # 2. 定义 conductivity(x) 和 source(x) 函数
    # 3. 调用 diffusion.solve() 获取稳态解 U
    # 4. 调用 diffusion.compute_flux() 计算热流
    # 5. 验证能量守恒: 积分(source) ≈ flux[-1] - flux[0]
    # 6. 打印结果并返回字典
    raise NotImplementedError("Hole 3a: 热扩散模块调用尚未实现")


def run_reaction_dynamics():
    """运行语义反应动力学"""
    section("4. 语义双向反应动力学")

    # TODO [Hole 3b]: 调用反应动力学求解器并验证精确解
    # 1. 创建 SemanticReactionDynamics 实例
    # 2. 数值求解 reaction.solve_numerical()
    # 3. 获取精确解 reaction.exact_solution() 并与数值解对比
    # 4. 验证守恒量 w1+w2 = w10+w20
    # 5. 计算并打印稳态和弛豫时间
    # 6. 构造多概念反应网络并求解
    # 7. 返回结果字典
    raise NotImplementedError("Hole 3b: 反应动力学模块调用尚未实现")


def run_period_analysis():
    """运行周期分析"""
    section("5. 语义系统非线性振荡器周期分析")

    print("Van der Pol 振荡器:")
    for mu in [0.5, 1.0, 2.0]:
        vdp = VanDerPolSemanticOscillator(mu=mu)
        p_est = vdp.period_estimate()
        p_num = vdp.measure_period_numerical(t_span=(0.0, 100.0))
        print(f"  mu={mu}: Urabe={p_est:.4f}, 数值={p_num:.4f}")

    pp = PredatorPreySemanticCycle(alpha=1.0, beta=0.1, gamma=1.5, delta=0.075,
                                   u0=10.0, v0=5.0)
    p_est = pp.period_estimate()
    p_num = pp.measure_period_numerical(t_span=(0.0, 100.0))
    print(f"\n捕食者-猎物周期:")
    print(f"  Shih估计={p_est:.4f}, 数值测量={p_num:.4f}")
    print(f"  守恒能量 E={pp.conserved_energy():.6f}")

    return {'vdp': vdp, 'pp': pp}


def run_space_quantization():
    """运行CVT空间量化"""
    section("6. 语义空间CVT量化")

    cvt = SemanticSpaceQuantization(n_generators=10, max_iter=200, tol=1e-12)
    result = cvt.quantize(init_mode='random', seed=42)

    uniform = np.linspace(0.0, 1.0, cvt.n + 2)[1:-1]
    test_points = np.linspace(0.0, 1.0, 1000)
    error_cvt = cvt.quantization_error(result['generators'], test_points)
    error_uniform = cvt.quantization_error(uniform, test_points)

    print(f"迭代次数: {result['iterations']}")
    print(f"最终能量: {result['final_energy']:.6e}")
    print(f"CVT量化误差: {error_cvt:.6e}")
    print(f"均匀量化误差: {error_uniform:.6e}")
    print(f"改进比例: {(error_uniform - error_cvt) / error_uniform * 100:.2f}%")

    # 二维CVT
    cvt2 = SemanticSpaceQuantization(n_generators=6, max_iter=200, tol=1e-12)
    res2d = cvt2.quantize_2d(n_generators_x=5, n_generators_y=5, init_mode='random', seed=42)
    print(f"\n二维CVT生成元数: {len(res2d['generators_2d'])}")

    return {'cvt': cvt, 'result': result}


def run_path_optimizer():
    """运行TSP路径优化"""
    section("7. 语义嵌入TSP路径优化")

    n = 10
    D = generate_semantic_distance_matrix(n, dim=8, seed=42)
    optimizer = SemanticPathOptimizer(D, seed=42)

    random_path = np.arange(n)
    random_cost = optimizer.path_cost(random_path)
    result = optimizer.multi_start_optimize(num_starts=5, max_variations=3000)

    print(f"语义节点数: {n}")
    print(f"顺序路径代价: {random_cost:.4f}")
    print(f"优化后代价: {result['cost']:.4f}")
    print(f"改进: {(random_cost - result['cost']) / random_cost * 100:.2f}%")
    print(f"最优路径: {result['path']}")

    return {'optimizer': optimizer, 'result': result}


def run_subset_selector():
    """运行特征子集选择"""
    section("8. 语义特征子集选择")

    selector = SemanticSubsetSelector(max_brute_force_n=20)

    # 小规模精确解
    weights = np.array([1, 2, 4, 8, 16, 32])
    target = 22.0
    result = selector.brute_force_search(weights, target)
    print(f"子集和问题: weights={weights}, target={target}")
    print(f"选择: {result['choose']}, 子集和={result['subset_sum']}, 精确={result['found_exact']}")

    # 语义嵌入特征选择
    rng = np.random.default_rng(42)
    embedding = rng.standard_normal(16) * np.exp(-np.arange(16) / 5.0)
    result2 = selector.feature_selection_for_embedding(embedding, target_info_ratio=0.8)
    selected_dims = np.where(result2['choose'] == 1)[0]
    total_info = np.sum(np.abs(embedding))
    selected_info = np.sum(np.abs(embedding[selected_dims]))

    print(f"\n嵌入维度: {len(embedding)}")
    print(f"总信息量: {total_info:.4f}")
    print(f"选择维度数: {len(selected_dims)} / {len(embedding)}")
    print(f"保留信息比: {selected_info / total_info * 100:.2f}%")

    return {'selector': selector, 'result': result2}


def run_fractal_analysis():
    """运行分形分析"""
    section("9. 语义嵌入分形与混沌分析")

    mbd = MandelbrotSemanticBoundary(escape_radius=2.0)
    result = mbd.compute_region(x_min=-1.0, x_max=-0.6, y_min=0.0, y_max=0.4,
                                nx=41, ny=41, max_iter=30)
    print(f"逃逸时间矩阵: {result.shape}")
    print(f"属于Mandelbrot集的点数: {np.sum(result >= 30)}")
    print(f"逃逸点比例: {np.sum(result < 30) / result.size * 100:.2f}%")

    # 分形维数
    D = mbd.estimate_fractal_dimension(
        x_min=-1.0, x_max=0.5, y_min=-1.0, y_max=1.0,
        resolutions=[32, 64, 128]
    )
    print(f"估计分形维数: {D:.4f}")

    # IFS
    ifs = IFSSemanticTransformer(seed=42)
    rng = np.random.default_rng(42)
    embedding = rng.standard_normal(10)
    points = ifs.transform_embedding(embedding, num_iterations=500)
    lyap = ifs.lyapunov_exponent(num_iterations=5000)

    print(f"\nIFS变换点数: {len(points)}")
    print(f"Lyapunov指数: {lyap:.6f}")
    print(f"混沌行为: {'是' if lyap > 0 else '否'}")

    return {'mbd': mbd, 'ifs': ifs}


def run_structured_decomposition():
    """运行结构化分解"""
    section("10. 语义嵌入维度结构化分解")

    decomp = SemanticStructuredDecomposition()

    common_dims = [128, 256, 512, 768]
    for dim in common_dims:
        f1, f2 = decomp.fermat_factor(dim)
        primes = decomp.prime_factors(dim)
        shape_info = decomp.optimal_tensor_shape(dim, max_rank=4)
        print(f"dim={dim:4d}: Fermat=({f1}, {f2}), 质因数={primes}, 张量形状={shape_info['shape']}")

    rng = np.random.default_rng(42)
    embedding = rng.standard_normal(768)
    result = decomp.hierarchical_decomposition(embedding, target_shape=(8, 8, 12))
    print(f"\n768维嵌入分层分解:")
    print(f"  张量形状: {result['tensor_shape']}")
    for me in result['mode_energies']:
        print(f"  模态 {me['mode']}: 主导能量比={me['dominant_energy_ratio']:.4f}")

    return {'decomp': decomp, 'result': result}


def run_sampling_init():
    """运行采样初始化"""
    section("11. 语义空间拉丁超立方采样")

    sampler = SemanticSpaceSampler(seed=42)
    samples = sampler.latin_hypercube_sampling(n_samples=50, n_dims=5)
    lhs_disc = sampler.discrepancy(samples)

    rng = np.random.default_rng(42)
    random_samples = rng.random((50, 5))
    random_disc = sampler.discrepancy(random_samples)

    print(f"LHS 样本范围: [{samples.min():.4f}, {samples.max():.4f}]")
    print(f"LHS 星差异度: {lhs_disc:.6f}")
    print(f"随机采样差异度: {random_disc:.6f}")
    print(f"LHS 改进: {(random_disc - lhs_disc) / random_disc * 100:.2f}%")

    # 超球面方向
    directions = sampler.uniform_direction_sampling(100, 8)
    norms = np.linalg.norm(directions, axis=1)
    print(f"方向向量范数均值: {norms.mean():.6f} (应为1.0)")

    return {'sampler': sampler, 'samples': samples}


def run_numerical_verification():
    """运行数值精度验证"""
    section("12. Steinerberger数值精度验证")

    verifier = SteinerbergerVerifier()
    results = verifier.verify_integration(n_values=[5, 10, 20, 50])

    for r in results['results']:
        print(f"n={r['n']:3d}: 精确={r['exact']:.8f}, "
              f"Simpson误差={r['simpson_error']:.2e}, "
              f"Quad误差={r['quad_error']:.2e}")

    # 语义嵌入积分测试
    rng = np.random.default_rng(42)
    embedding = rng.standard_normal(8)

    def semantic_weight_func(emb, x):
        n = min(len(emb), 10)
        basis = np.sin(np.pi * np.arange(1, n + 1) * x)
        return float(np.dot(emb[:n], basis))

    test_result = verifier.semantic_embedding_integral_test(embedding, semantic_weight_func)
    print(f"\n语义嵌入积分压力测试:")
    print(f"  绝对偏差: {test_result['difference']:.2e}")
    print(f"  相对偏差: {test_result['relative_error']:.2e}")

    return {'verifier': verifier, 'results': results}


def run_robust_utils():
    """运行鲁棒性工具验证"""
    section("13. 数值鲁棒性工具验证")

    utils = RobustNumericUtils()

    # 边界测试
    print(f"safe_divide(1.0, 0.0) = {utils.safe_divide(1.0, 0.0)}")
    print(f"safe_log(0.0) = {utils.safe_log(0.0)}")
    print(f"safe_sqrt(-1.0) = {utils.safe_sqrt(-1.0)}")

    e1 = np.array([1.0, 0.0, 1.0])
    e2 = np.array([0.0, 1.0, 0.0])
    e3 = np.array([0.0, 0.0, 0.0])
    print(f"\n语义相似度:")
    print(f"  sim([1,0,1], [0,1,0]) = {utils.semantic_similarity_safe(e1, e2):.6f}")
    print(f"  sim([1,0,1], [1,0,1]) = {utils.semantic_similarity_safe(e1, e1):.6f}")
    print(f"  sim([0,0,0], [1,0,1]) = {utils.semantic_similarity_safe(e3, e1):.6f}")

    # 病态矩阵
    A = np.array([[1.0, 1.0], [1.0, 1.0000001]])
    b = np.array([2.0, 2.0])
    x = utils.solve_linear_system_safe(A, b)
    print(f"\n病态矩阵求解:")
    print(f"  解: {x}")
    print(f"  残差: {np.linalg.norm(A @ x - b):.2e}")

    return {'utils': utils}


def run_integrated_analysis():
    """运行综合集成分析"""
    section("14. 综合集成分析")

    # 1. 生成语义嵌入空间的采样点
    sampler = SemanticSpaceSampler(seed=42)
    samples = sampler.latin_hypercube_sampling(n_samples=20, n_dims=3)
    directions = sampler.uniform_direction_sampling(20, 3)

    # 2. 构建距离矩阵并优化路径
    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((10, 8))
    from path_optimizer import generate_semantic_distance_matrix
    D = generate_semantic_distance_matrix(10, dim=8, seed=42)
    optimizer = SemanticPathOptimizer(D, seed=42)
    path_result = optimizer.multi_start_optimize(num_starts=3, max_variations=2000)

    # 3. 对路径上的嵌入进行CVT量化
    cvt = SemanticSpaceQuantization(n_generators=5, max_iter=100, tol=1e-10)
    cvt_result = cvt.quantize(init_mode='uniform', seed=42)

    # 4. 特征选择
    selector = SemanticSubsetSelector(max_brute_force_n=16)
    embedding = rng.standard_normal(16) * np.exp(-np.arange(16) / 4.0)
    subset_result = selector.feature_selection_for_embedding(embedding, target_info_ratio=0.75)

    # 5. 结构化分解
    decomp = SemanticStructuredDecomposition()
    tensor_result = decomp.hierarchical_decomposition(embedding[:128], target_shape=(4, 4, 8))

    # 6. 数值验证
    verifier = SteinerbergerVerifier()
    int_result = verifier.verify_integration(n_values=[10, 20])

    # 7. 鲁棒性验证
    utils = RobustNumericUtils()
    sim_mat = utils.batch_semantic_similarity(embeddings)

    print(f"生成语义采样点: {samples.shape}")
    print(f"语义路径优化代价: {path_result['cost']:.4f}")
    print(f"CVT量化误差: {cvt_result['final_energy']:.6e}")
    print(f"特征选择保留维度: {np.sum(subset_result['choose'])} / {len(subset_result['choose'])}")
    print(f"张量分解形状: {tensor_result['tensor_shape']}")
    print(f"数值积分验证通过: {all(r['quad_error'] < 1e-6 for r in int_result['results'])}")
    print(f"批量相似度矩阵对称性偏差: {np.max(np.abs(sim_mat - sim_mat.T)):.2e}")

    return {
        'samples': samples,
        'path_result': path_result,
        'cvt_result': cvt_result,
        'subset_result': subset_result,
        'tensor_result': tensor_result,
        'sim_mat': sim_mat
    }


def main():
    """
    主入口函数。
    
    依次执行所有模块的分析与验证。
    """
    print("=" * 70)
    print("  基于多尺度物理-化学动力学约束的语义嵌入空间分析与优化系统")
    print("  Physics-Informed Semantic Embedding Space Analysis & Optimization")
    print("=" * 70)
    print("\n  科学领域: 数据科学 - 自然语言处理语义嵌入")
    print("  项目编号: PROJECT_188")
    print("  语言: Python 3")

    results = {}

    try:
        results['embedding_bases'] = run_embedding_bases_analysis()
        results['fem_projection'] = run_fem_projection()
        results['heat_diffusion'] = run_heat_diffusion()
        results['reaction_dynamics'] = run_reaction_dynamics()
        results['period_analysis'] = run_period_analysis()
        results['space_quantization'] = run_space_quantization()
        results['path_optimizer'] = run_path_optimizer()
        results['subset_selector'] = run_subset_selector()
        results['fractal_analysis'] = run_fractal_analysis()
        results['structured_decomposition'] = run_structured_decomposition()
        results['sampling_init'] = run_sampling_init()
        results['numerical_verification'] = run_numerical_verification()
        results['robust_utils'] = run_robust_utils()
        results['integrated'] = run_integrated_analysis()

        print("\n" + "=" * 70)
        print("  所有模块执行完毕，系统运行成功!")
        print("=" * 70)

    except Exception as e:
        print(f"\n[ERROR] 执行过程中出现错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    return results


if __name__ == "__main__":
    main()
