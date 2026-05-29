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

    diffusion = SemanticHeatDiffusion(n=41, a=0.0, b=1.0)
    ua, ub = 1.0, 0.0

    def conductivity(x):
        return 1.0 + 0.5 * np.sin(np.pi * x) ** 2

    def source(x):
        return 2.0 * np.exp(-((x - 0.3) ** 2) / 0.01)

    U = diffusion.solve(ua, ub, conductivity, source)
    flux = diffusion.compute_flux(U, conductivity)

    total_source = np.trapezoid([source(xi) for xi in diffusion.x], diffusion.x)
    net_flux = flux[-1] - flux[0]

    print(f"稳态解范围: [{U.min():.6f}, {U.max():.6f}]")
    print(f"边界热流: 左={flux[0]:.6f}, 右={flux[-1]:.6f}")
    print(f"总热源: {total_source:.6f}")
    print(f"净热流: {net_flux:.6f}")
    print(f"守恒偏差: {abs(total_source - net_flux) / (abs(total_source) + 1e-15):.6e}")

    return {'diffusion': diffusion, 'solution': U, 'flux': flux}


def run_reaction_dynamics():
    """运行语义反应动力学"""
    section("4. 语义双向反应动力学")

    reaction = SemanticReactionDynamics(k1=0.3, k2=0.1, w10=1.0, w20=0.0,
                                        t0=0.0, tstop=50.0)
    t_num, w1_num, w2_num = reaction.solve_numerical(num_points=200)
    w1_exact, w2_exact = reaction.exact_solution(t_num)

    error1 = np.max(np.abs(w1_num - w1_exact))
    error2 = np.max(np.abs(w2_num - w2_exact))
    conserved = reaction.conserved_quantity(w1_num, w2_num)
    conserved_exact = reaction.w10 + reaction.w20

    print(f"k1={reaction.k1}, k2={reaction.k2}")
    print(f"数值-精确解最大偏差: w1={error1:.6e}, w2={error2:.6e}")
    print(f"守恒量偏差: {np.max(np.abs(conserved - conserved_exact)):.6e}")
    print(f"稳态: w1_eq={reaction.equilibrium()[0]:.6f}, w2_eq={reaction.equilibrium()[1]:.6f}")
    print(f"弛豫时间: {reaction.relaxation_time():.6f}")

    # 多概念网络
    n = 4
    K = np.zeros((n, n))
    for i in range(n):
        j_next = (i + 1) % n
        K[j_next, i] = 0.2
        K[i, i] -= 0.2
        j_prev = (i - 1) % n
        K[j_prev, i] = 0.1
        K[i, i] -= 0.1

    network = MultiConceptReactionNetwork(n, K, np.array([1.0, 0.0, 0.0, 0.0]))
    t_net, y_net = network.solve((0.0, 50.0), num_points=200)
    print(f"\n多概念网络稳态: {network.equilibrium_state()}")

    return {'reaction': reaction, 'network': network}


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
# ---- TC01: SemanticEmbeddingBases正交基数量为正整数 ----
bases_tc01 = SemanticEmbeddingBases(radius=1.0, max_mode_m=4, max_mode_n=3)
assert bases_tc01.num_bases > 0, '[TC01] num_bases should be positive FAILED'
assert isinstance(bases_tc01.num_bases, int), '[TC01] num_bases should be int FAILED'

# ---- TC02: 语义场重构误差为有限值 ----
import numpy as np
np.random.seed(42)
bases_tc02 = SemanticEmbeddingBases(radius=1.0, max_mode_m=3, max_mode_n=2)
r_grid = np.linspace(0.01, 1.0, 20)
theta_grid = np.linspace(0, 2 * np.pi, 30)
R_m, T_m = np.meshgrid(r_grid, theta_grid)
r_flat = R_m.flatten()
theta_flat = T_m.flatten()
semantic_field = np.exp(-((r_flat - 0.5) ** 2 + (np.sin(theta_flat)) ** 2) / 0.1)
coeffs = bases_tc02.project_semantic_vector(semantic_field, r_flat, theta_flat)
reconstructed = bases_tc02.reconstruct_semantic_field(coeffs, r_flat, theta_flat)
error = np.linalg.norm(semantic_field - reconstructed) / np.linalg.norm(semantic_field)
assert np.isfinite(error), '[TC02] reconstruction error should be finite FAILED'
assert error >= 0.0, '[TC02] reconstruction error should be non-negative FAILED'
assert error < 1.0, '[TC02] reconstruction error should be < 1.0 FAILED'

# ---- TC03: 正交基Gram矩阵对角元接近1 ----
import numpy as np
bases_tc03 = SemanticEmbeddingBases(radius=1.0, max_mode_m=3, max_mode_n=2)
r_grid2 = np.linspace(0.01, 1.0, 15)
theta_grid2 = np.linspace(0, 2 * np.pi, 30)
R2, T2 = np.meshgrid(r_grid2, theta_grid2)
r_flat2 = R2.flatten()
theta_flat2 = T2.flatten()
gram = bases_tc03.basis_orthogonality_check(r_flat2, theta_flat2)
assert gram.shape == (bases_tc03.num_bases, bases_tc03.num_bases), '[TC03] Gram matrix shape wrong FAILED'
assert np.all(np.abs(np.diag(gram) - 1.0) < 0.1), '[TC03] Gram diagonal should be close to 1 FAILED'

# ---- TC04: FEM投影解形状与节点数一致 ----
import numpy as np
fem_tc04 = FEM2DSemanticProjection(xl=0.0, xr=1.0, yb=0.0, yt=1.0, nx=9, ny=9)
def sem_density(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y) + x
U = fem_tc04.project(sem_density)
assert len(U) == fem_tc04.node_num, '[TC04] FEM solution length should equal node_num FAILED'
assert np.all(np.isfinite(U)), '[TC04] FEM solution should be all finite FAILED'

# ---- TC05: FEM L2误差为非负有限值 ----
import numpy as np
fem_tc05 = FEM2DSemanticProjection(xl=0.0, xr=1.0, yb=0.0, yt=1.0, nx=9, ny=9)
def sem_density2(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y) + x
U2 = fem_tc05.project(sem_density2)
errors = fem_tc05.compute_l2_error(U2, sem_density2)
assert errors['relative_error'] >= 0.0, '[TC05] relative_error should be non-negative FAILED'
assert np.isfinite(errors['relative_error']), '[TC05] relative_error should be finite FAILED'

# ---- TC06: 热扩散解长度正确 ----
import numpy as np
diff_tc06 = SemanticHeatDiffusion(n=41, a=0.0, b=1.0)
def cond(x):
    return 1.0 + 0.5 * np.sin(np.pi * x) ** 2
def src(x):
    return 2.0 * np.exp(-((x - 0.3) ** 2) / 0.01)
U_diff = diff_tc06.solve(1.0, 0.0, cond, src)
assert len(U_diff) == 41, '[TC06] solution length should be 41 FAILED'
assert np.all(np.isfinite(U_diff)), '[TC06] solution should be finite FAILED'

# ---- TC07: 热扩散守恒性（总热源≈净热流） ----
import numpy as np
diff_tc07 = SemanticHeatDiffusion(n=41, a=0.0, b=1.0)
def cond2(x):
    return 1.0 + 0.5 * np.sin(np.pi * x) ** 2
def src2(x):
    return 2.0 * np.exp(-((x - 0.3) ** 2) / 0.01)
U2_diff = diff_tc07.solve(1.0, 0.0, cond2, src2)
flux = diff_tc07.compute_flux(U2_diff, cond2)
total_source = np.trapezoid([src2(xi) for xi in diff_tc07.x], diff_tc07.x)
net_flux = flux[-1] - flux[0]
conservation_error = abs(total_source - net_flux) / (abs(total_source) + 1e-15)
assert conservation_error < 0.05, '[TC07] heat conservation error too large FAILED'

# ---- TC08: 反应动力学精确解守恒 ----
import numpy as np
rxn_tc08 = SemanticReactionDynamics(k1=0.3, k2=0.1, w10=1.0, w20=0.0, t0=0.0, tstop=50.0)
t_test = np.linspace(0.0, 50.0, 100)
w1_ex, w2_ex = rxn_tc08.exact_solution(t_test)
conserved = rxn_tc08.conserved_quantity(w1_ex, w2_ex)
assert np.all(np.abs(conserved - conserved[0]) < 1e-12), '[TC08] exact solution should conserve w1+w2 FAILED'

# ---- TC09: 数值解与精确解偏差可控 ----
import numpy as np
rxn_tc09 = SemanticReactionDynamics(k1=0.3, k2=0.1, w10=1.0, w20=0.0, t0=0.0, tstop=50.0)
t_num, w1_num, w2_num = rxn_tc09.solve_numerical(num_points=200)
w1_exact, w2_exact = rxn_tc09.exact_solution(t_num)
error1 = np.max(np.abs(w1_num - w1_exact))
error2 = np.max(np.abs(w2_num - w2_exact))
assert error1 < 1e-6, '[TC09] w1 numerical error too large FAILED'
assert error2 < 1e-6, '[TC09] w2 numerical error too large FAILED'

# ---- TC10: 稳态平衡值满足w1_eq + w2_eq = w10 + w20 ----
rxn_tc10 = SemanticReactionDynamics(k1=0.3, k2=0.1, w10=1.0, w20=0.0)
w1_eq, w2_eq = rxn_tc10.equilibrium()
assert abs(w1_eq + w2_eq - (rxn_tc10.w10 + rxn_tc10.w20)) < 1e-12, '[TC10] equilibrium must conserve total mass FAILED'

# ---- TC11: Van der Pol周期估计为正有限值 ----
vdp_tc11 = VanDerPolSemanticOscillator(mu=1.0)
p_est = vdp_tc11.period_estimate()
assert p_est > 0.0, '[TC11] Van der Pol period estimate should be positive FAILED'
assert np.isfinite(p_est), '[TC11] Van der Pol period estimate should be finite FAILED'

# ---- TC12: 捕食者-猎物守恒能量为有限值 ----
pp_tc12 = PredatorPreySemanticCycle(alpha=1.0, beta=0.1, gamma=1.5, delta=0.075, u0=10.0, v0=5.0)
E = pp_tc12.conserved_energy()
assert np.isfinite(E), '[TC12] conserved energy should be finite FAILED'

# ---- TC13: CVT能量单调不增 ----
import numpy as np
np.random.seed(42)
cvt_tc13 = SemanticSpaceQuantization(n_generators=10, max_iter=200, tol=1e-12)
result_tc13 = cvt_tc13.quantize(init_mode='random', seed=42)
energies = result_tc13['energy_history']
is_decreasing = all(energies[i] >= energies[i + 1] - 1e-14 for i in range(len(energies) - 1))
assert is_decreasing, '[TC13] CVT energy should be monotonically non-increasing FAILED'

# ---- TC14: CVT量化误差小于均匀量化误差 ----
import numpy as np
np.random.seed(42)
cvt_tc14 = SemanticSpaceQuantization(n_generators=10, max_iter=200, tol=1e-12)
result_tc14 = cvt_tc14.quantize(init_mode='random', seed=42)
uniform = np.linspace(0.0, 1.0, cvt_tc14.n + 2)[1:-1]
test_pts = np.linspace(0.0, 1.0, 1000)
error_cvt = cvt_tc14.quantization_error(result_tc14['generators'], test_pts)
error_uniform = cvt_tc14.quantization_error(uniform, test_pts)
assert error_cvt <= error_uniform, '[TC14] CVT error should not exceed uniform error FAILED'

# ---- TC15: TSP优化后代价不大于顺序路径代价 ----
import numpy as np
np.random.seed(42)
D_tc15 = generate_semantic_distance_matrix(10, dim=8, seed=42)
opt_tc15 = SemanticPathOptimizer(D_tc15, seed=42)
random_path = np.arange(10)
random_cost = opt_tc15.path_cost(random_path)
result_tc15 = opt_tc15.multi_start_optimize(num_starts=5, max_variations=3000)
assert result_tc15['cost'] <= random_cost, '[TC15] optimized cost should not exceed sequential cost FAILED'

# ---- TC16: 子集和暴力搜索精确解 ----
import numpy as np
sel_tc16 = SemanticSubsetSelector(max_brute_force_n=20)
weights = np.array([1, 2, 4, 8, 16, 32])
target = 22.0
result_tc16 = sel_tc16.brute_force_search(weights, target)
assert result_tc16['found_exact'], '[TC16] should find exact solution for 2+4+16=22 FAILED'
assert abs(result_tc16['subset_sum'] - target) < 1e-8, '[TC16] subset sum should match target FAILED'

# ---- TC17: Mandelbrot逃逸时间矩阵形状正确 ----
import numpy as np
mbd_tc17 = MandelbrotSemanticBoundary(escape_radius=2.0)
result_tc17 = mbd_tc17.compute_region(x_min=-1.0, x_max=-0.6, y_min=0.0, y_max=0.4, nx=41, ny=41, max_iter=30)
assert result_tc17.shape == (41, 41), '[TC17] Mandelbrot result shape should be (41, 41) FAILED'
assert np.all(result_tc17 >= 0), '[TC17] all escape times should be non-negative FAILED'
assert np.all(result_tc17 <= 30), '[TC17] all escape times should not exceed max_iter FAILED'

# ---- TC18: IFS Lyapunov指数为有限值 ----
import numpy as np
np.random.seed(42)
ifs_tc18 = IFSSemanticTransformer(seed=42)
lyap = ifs_tc18.lyapunov_exponent(num_iterations=5000)
assert np.isfinite(lyap), '[TC18] Lyapunov exponent should be finite FAILED'

# ---- TC19: Fermat因式分解验证 ----
decomp_tc19 = SemanticStructuredDecomposition()
f1, f2 = decomp_tc19.fermat_factor(91)
assert f1 * f2 == 91, '[TC19] Fermat: 91 = f1 * f2 FAILED'

# ---- TC20: LHS样本在[0,1]范围内 ----
import numpy as np
np.random.seed(42)
sampler_tc20 = SemanticSpaceSampler(seed=42)
samples = sampler_tc20.latin_hypercube_sampling(n_samples=50, n_dims=5)
assert samples.shape == (50, 5), '[TC20] LHS sample shape should be (50, 5) FAILED'
assert np.all(samples >= 0.0), '[TC20] LHS samples should be >= 0 FAILED'
assert np.all(samples <= 1.0), '[TC20] LHS samples should be <= 1 FAILED'

# ---- TC21: Steinerberger精确积分公式 ----
verif_tc21 = SteinerbergerVerifier()
exact_n5 = verif_tc21.exact_integral(5)
H5 = float(np.sum(1.0 / np.arange(1, 6)))
expected = (2.0 / np.pi) * H5
assert abs(exact_n5 - expected) < 1e-14, '[TC21] exact integral formula FAILED'

# ---- TC22: safe_divide除零返回默认值 ----
utils_tc22 = RobustNumericUtils()
assert utils_tc22.safe_divide(1.0, 0.0) == 0.0, '[TC22] safe_divide(1, 0) should return default 0.0 FAILED'
assert utils_tc22.safe_divide(6.0, 3.0) == 2.0, '[TC22] safe_divide(6, 3) should be 2.0 FAILED'
assert utils_tc22.safe_log(0.0) == -700.0, '[TC22] safe_log(0) should return default -700.0 FAILED'
assert utils_tc22.safe_sqrt(-1.0) == 0.0, '[TC22] safe_sqrt(-1) should return default 0.0 FAILED'

# ---- TC23: 语义相似度对称且自身相似度为1 ----
import numpy as np
utils_tc23 = RobustNumericUtils()
e1 = np.array([1.0, 0.0, 0.0])
e2 = np.array([0.0, 1.0, 0.0])
sim12 = utils_tc23.semantic_similarity_safe(e1, e2)
sim21 = utils_tc23.semantic_similarity_safe(e2, e1)
sim11 = utils_tc23.semantic_similarity_safe(e1, e1)
assert abs(sim12 - sim21) < 1e-14, '[TC23] cosine similarity should be symmetric FAILED'
assert abs(sim11 - 1.0) < 1e-14, '[TC23] self similarity should be 1.0 FAILED'

# ---- TC24: 零向量语义相似度为0 ----
import numpy as np
utils_tc24 = RobustNumericUtils()
e_zero = np.array([0.0, 0.0, 0.0])
e_norm = np.array([1.0, 2.0, 3.0])
sim_zero = utils_tc24.semantic_similarity_safe(e_zero, e_norm)
assert sim_zero == 0.0, '[TC24] similarity with zero vector should be 0 FAILED'

# ---- TC25: 可复现性：固定种子两次CVT结果相同 ----
import numpy as np
np.random.seed(42)
cvt_tc25a = SemanticSpaceQuantization(n_generators=6, max_iter=100, tol=1e-10)
res_a = cvt_tc25a.quantize(init_mode='random', seed=123)
np.random.seed(42)
cvt_tc25b = SemanticSpaceQuantization(n_generators=6, max_iter=100, tol=1e-10)
res_b = cvt_tc25b.quantize(init_mode='random', seed=123)
assert np.allclose(res_a['generators'], res_b['generators']), '[TC25] CVT results should be reproducible with same seed FAILED'

# ---- TC26: 病态矩阵安全求解不崩溃 ----
import numpy as np
utils_tc26 = RobustNumericUtils()
A = np.array([[1.0, 1.0], [1.0, 1.0000001]])
b = np.array([2.0, 2.0])
x = utils_tc26.solve_linear_system_safe(A, b)
assert len(x) == 2, '[TC26] solution should have length 2 FAILED'
assert np.all(np.isfinite(x)), '[TC26] solution should be finite FAILED'

# ---- TC27: 质因数分解正确性 ----
decomp_tc27 = SemanticStructuredDecomposition()
primes_12 = decomp_tc27.prime_factors(12)
assert sorted(primes_12) == [2, 2, 3], '[TC27] prime factors of 12 should be [2,2,3] FAILED'
primes_768 = decomp_tc27.prime_factors(768)
prod_768 = 1
for p in primes_768:
    prod_768 *= p
assert prod_768 == 768, '[TC27] product of prime factors should equal 768 FAILED'

# ---- TC28: LHS方向采样范数接近1 ----
import numpy as np
np.random.seed(42)
sampler_tc28 = SemanticSpaceSampler(seed=42)
directions = sampler_tc28.uniform_direction_sampling(200, 8)
norms = np.linalg.norm(directions, axis=1)
assert np.all(np.abs(norms - 1.0) < 1e-12), '[TC28] direction vector norms should be 1.0 FAILED'

# ---- TC29: 多概念网络稳态守恒 ----
import numpy as np
n = 4
K = np.zeros((n, n))
for i in range(n):
    j_next = (i + 1) % n
    K[j_next, i] = 0.2
    K[i, i] -= 0.2
    j_prev = (i - 1) % n
    K[j_prev, i] = 0.1
    K[i, i] -= 0.1
y0 = np.array([1.0, 0.0, 0.0, 0.0])
network_tc29 = MultiConceptReactionNetwork(n, K, y0)
y_eq = network_tc29.equilibrium_state()
assert abs(np.sum(y_eq) - np.sum(y0)) < 1e-10, '[TC29] equilibrium state should conserve total mass FAILED'

# ---- TC30: 弛豫时间为正值 ----
rxn_tc30 = SemanticReactionDynamics(k1=0.3, k2=0.1)
tau = rxn_tc30.relaxation_time()
assert tau > 0.0, '[TC30] relaxation time should be positive FAILED'
assert np.isfinite(tau), '[TC30] relaxation time should be finite FAILED'

print('\n全部 30 个测试通过!\n')
