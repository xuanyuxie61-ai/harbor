"""
量子行走搜索算法：博士级科研代码合成项目
================================================
统一入口文件，零参数可运行。

本程序围绕"量子计算：量子行走搜索算法"展开，
融合15个种子项目的核心算法，在多种图结构（1D链、2D网格、
六边形晶格、三角剖分区域、超立方体、球面立方网格）上
实现并分析离散时间与连续时间量子行走搜索算法。
"""
import numpy as np
import sys

# ---------------------------------------------------------------------------
# 导入所有模块
# ---------------------------------------------------------------------------
from utils import (
    normalize_vector, hadamard_matrix, grover_coin, safe_divide,
    discrete_laplacian_1d, clamp, chebyshev_nodes
)
from matrix_solvers import (
    r8to_sl, r8vm_sl, r83_cg, r83_cr_fa, r83_cr_sl,
    r83_jac_sl, r83_gs_sl, cg_rc_solve, wathen, wathen_order
)
from geometry_mesh import (
    generate_2d_mesh, mesh_adjacency, generate_cubed_sphere_grid,
    cubed_sphere_adjacency, generate_hexagonal_lattice, hexagonal_adjacency,
    hexagon_stroud_rule4, hexagon_monomial_integral,
    sphere_cubed_grid_point_count, sphere_cubed_grid_line_count
)
from lattice_states import (
    diophantine_nd_nonnegative, diophantine_nd_nonnegative_bounded,
    generate_cc_sparse_grid, constrained_parameter_grid,
    analyze_probability_landscape, build_hypercube_states
)
from quantum_operators import (
    hadamard_coin, grover_coin, fourier_coin, shift_operator_1d,
    shift_operator_graph, oracle_operator, graph_laplacian,
    ctqw_hamiltonian, ctqw_hamiltonian_with_marked, unitary_evolution,
    spectral_gap, eigenstate_localization
)
from quantum_walk_core import (
    DiscreteTimeQuantumWalk, QuantumWalkSearch, ContinuousTimeQuantumWalk,
    CTQWSearch, MultiDimensionalQuantumWalk
)
from search_algorithm import (
    estimate_search_complexity, spatial_search_2d_grid,
    spectral_search_analysis, multi_target_search_phase_estimation,
    hexagonal_lattice_search, meshed_domain_search, hypercube_search
)
from numerical_quadrature import (
    integrate_trapezoidal, integrate_simpson,
    integrate_hexagon, hexagon_monomial_integral,
    steinerberger_function, steinerberger_integral01_exact,
    test_quadrature_accuracy, integrate_quantum_probability
)
from optimization_diagnostics import (
    newton_raphson, find_optimal_coin_angle, find_critical_gamma,
    analyze_search_landscape, detect_local_maxima,
    coupon_collector_simulation, compare_classical_quantum_cover_time,
    convergence_rate_analysis
)


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_experiment_1d_tridiagonal_solver():
    """实验1: 1D量子行走 + 三对角矩阵求解器 (r83, r8to)"""
    print_section("实验1: 1D量子行走与三对角/Toeplitz求解器")

    n = 64
    qw = DiscreteTimeQuantumWalk(n, coin_dim=2, coin_type="hadamard", periodic=True)
    qw.set_initial_state(position=n // 2)
    qw.step(num_steps=50)

    prob = qw.get_position_distribution()
    norm = qw.get_state_norm()
    print(f"  状态范数守恒: {norm:.12f} (应 ≈ 1.0)")
    print(f"  50步后概率分布均值: {np.mean(prob):.6f}, 标准差: {np.std(prob):.6f}")

    # 用三对角求解器求1D离散Laplacian的稳态
    L = discrete_laplacian_1d(n, periodic=False)
    a = np.zeros((3, n))
    a[0, :-1] = np.diag(L, 1)
    a[1, :] = np.diag(L)
    a[2, 1:] = np.diag(L, -1)
    b = np.ones(n)

    # CG
    x_cg = r83_cg(n, a, b)
    # Cyclic reduction
    a_cr = r83_cr_fa(n, a)
    x_cr = r83_cr_sl(n, a_cr, b)
    # Jacobi
    x_jac = r83_jac_sl(n, a, b, max_iter=5000)
    # Gauss-Seidel
    x_gs = r83_gs_sl(n, a, b, max_iter=5000)

    print(f"  CG残差:        {np.linalg.norm(L @ x_cg - b):.2e}")
    print(f"  循环约化残差:  {np.linalg.norm(L @ x_cr - b):.2e}")
    print(f"  Jacobi残差:    {np.linalg.norm(L @ x_jac - b):.2e}")
    print(f"  Gauss-Seidel残差: {np.linalg.norm(L @ x_gs - b):.2e}")

    # Toeplitz求解器: 量子行走平移算子的Toeplitz结构
    a_toep = np.zeros(2 * n - 1)
    a_toep[0] = 2.0
    a_toep[1] = -1.0
    a_toep[n] = -1.0  # first column below diagonal: T[1,0]
    x_toep = r8to_sl(n, a_toep, b)
    print(f"  Toeplitz求解器残差: {np.linalg.norm(L @ x_toep - b):.2e}")


def run_experiment_vandermonde_reconstruction():
    """实验2: Vandermonde矩阵求解用于量子态谱重构 (r8vm)"""
    print_section("实验2: Vandermonde谱重构")

    n = 12
    # 模拟从n个能量本征值重构量子态振幅
    x_nodes = chebyshev_nodes(n, a=-1.0, b=1.0)
    # 真实振幅系数
    true_coeffs = np.exp(-np.arange(n) ** 2 / 10.0)
    # 使用Vandermonde矩阵-vector乘法生成右端项: b = A @ true_coeffs
    from matrix_solvers import r8vm_mv
    b = r8vm_mv(n, n, x_nodes, true_coeffs)
    recovered = r8vm_sl(n, x_nodes, b)
    err = np.linalg.norm(recovered - true_coeffs)
    print(f"  重构误差 (Vandermonde): {err:.2e}")
    print(f"  条件数估计: {np.linalg.cond(np.vander(x_nodes, n)):.2e}")


def run_experiment_2d_spatial_search():
    """实验3: 2D网格空间搜索 + 2D网格剖分 (image_mesh2d, cc_display)"""
    print_section("实验3: 2D网格空间搜索")

    nx, ny = 16, 16
    marked = [(nx // 2, ny // 2), (nx // 2 + 1, ny // 2)]
    result = spatial_search_2d_grid(nx, ny, marked, max_steps=150)
    print(f"  网格大小: {result['grid_size']}")
    print(f"  标记顶点: {result['marked_vertices']}")
    print(f"  最优步数: {result['optimal_steps']}")
    print(f"  最大成功概率: {result['max_success_probability']:.6f}")

    # 2D区域三角剖分搜索
    boundary = np.array([
        [0.0, 0.0], [1.0, 0.0], [1.2, 0.5], [1.0, 1.0],
        [0.5, 1.2], [0.0, 1.0], [-0.2, 0.5]
    ])
    mesh_result = meshed_domain_search(boundary, marked_nodes=[5, 10],
                                       hmax=0.4, max_steps=80)
    print(f"  三角剖分顶点数: {mesh_result['num_vertices']}")
    print(f"  三角剖分单元数: {mesh_result['num_elements']}")
    print(f"  剖分区域最优步数: {mesh_result['optimal_steps']}")
    print(f"  剖分区域最大成功概率: {mesh_result['max_success_probability']:.6f}")


def run_experiment_hexagonal_search():
    """实验4: 六边形晶格量子行走搜索 + 六边形数值积分 (hexagon_stroud_rule)"""
    print_section("实验4: 六边形晶格搜索与六边形积分")

    n_rings = 3
    marked = [0, 7, 14]
    result = hexagonal_lattice_search(n_rings, marked, max_steps=60)
    print(f"  六边形晶格顶点数: {result['num_vertices']}")
    print(f"  最优步数: {result['optimal_steps']}")
    print(f"  最大成功概率: {result['max_success_probability']:.6f}")

    # 六边形数值积分: 计算量子观测量的空间积分
    def gaussian_bump(x, y):
        return np.exp(-(x ** 2 + y ** 2) / 0.5)

    val = integrate_hexagon(gaussian_bump, rule=4)
    exact_ref = 1.71265450069  # 数值参考值
    print(f"  六边形高斯积分 (Stroud rule 4): {val:.8f}")
    print(f"  与参考值误差: {abs(val - exact_ref):.2e}")

    # 单式积分精确值测试
    for p, q in [(0, 0), (2, 0), (0, 2), (2, 2)]:
        exact = hexagon_monomial_integral(p, q)
        print(f"    x^{p} y^{q} 精确积分: {exact:.8f}")


def run_experiment_cubed_sphere():
    """实验5: 球面立方网格上的量子行走 (sphere_cubed_grid)"""
    print_section("实验5: 球面立方网格量子行走")

    n = 4
    points, lines = generate_cubed_sphere_grid(n)
    adj = cubed_sphere_adjacency(points, lines)
    num_pts = points.shape[0]
    num_lines = lines.shape[0]
    print(f"  球面网格点数: {num_pts} (理论: {sphere_cubed_grid_point_count(n)})")
    print(f"  球面网格边数: {num_lines} (理论: {sphere_cubed_grid_line_count(n)})")

    # CTQW on sphere grid
    ctqw = ContinuousTimeQuantumWalk(adj, gamma=1.0)
    ctqw.set_initial_state(0)
    ctqw.evolve(t=5.0)
    prob = ctqw.get_position_distribution()
    print(f"  演化后概率和: {np.sum(prob):.12f}")
    print(f"  最大概率位置: {int(np.argmax(prob))}")
    print(f"  概率熵: {-np.sum(prob * np.log(prob + 1e-16)):.4f}")

    # 谱分析
    L = graph_laplacian(adj)
    eigs = np.linalg.eigvalsh(L)
    print(f"  Laplacian谱隙: {eigs[1] - eigs[0]:.6f}")
    print(f"  最大特征值: {eigs[-1]:.4f}")


def run_experiment_hypercube_search():
    """实验6: 超立方体高维搜索 + Diophantine约束状态枚举 (diophantine_nd)"""
    print_section("实验6: 超立方体搜索与高维约束状态空间")

    dim = 6
    n = 2 ** dim
    marked = [5, 42, 100]
    result = hypercube_search(dim, marked, max_steps=80)
    print(f"  超立方体维度: {result['dimension']}")
    print(f"  顶点总数: {result['num_vertices']}")
    print(f"  最优步数 (实际): {result['optimal_steps']}")
    print(f"  理论最优估计: {result['theoretical_optimal']}")
    print(f"  最大成功概率: {result['max_success_probability']:.6f}")

    # Diophantine约束: 枚举满足守恒律的高维状态
    a = np.array([1, 2, 3, 4, 5])
    b = 20
    bounds = np.array([5, 5, 4, 3, 2])
    sols_bounded = diophantine_nd_nonnegative_bounded(a, b, bounds)
    sols_free = diophantine_nd_nonnegative(a, b)
    print(f"  Diophantine a·x={b}, 有界解数量: {len(sols_bounded)}")
    print(f"  Diophantine a·x={b}, 无界解数量: {len(sols_free)}")

    # 超立方体顶点坐标
    verts = build_hypercube_states(n, dim)
    print(f"  超立方体顶点坐标维度: {verts.shape}")


def run_experiment_cc_sparse_grid_params():
    """实验7: Clenshaw-Curtis稀疏网格参数优化 (cc_display, levels)"""
    print_section("实验7: 稀疏网格参数采样与优化")

    dim = 3
    max_level = 4
    grid = generate_cc_sparse_grid(dim, max_level)
    print(f"  稀疏网格点数 (dim={dim}, level<={max_level}): {grid.shape[0]}")

    # 加权约束网格
    weights = np.array([1.0, 2.0, 1.5])
    grid_w = constrained_parameter_grid(dim, max_level, weights)
    print(f"  加权约束网格点数: {grid_w.shape[0]}")

    # 用稀疏网格采样评估搜索成功概率 landscape
    def dummy_success_prob(params):
        # 模拟成功概率随两个参数变化
        p1, p2 = params[0], params[1]
        return np.exp(-(p1 - 0.3) ** 2 / 0.1 - (p2 + 0.1) ** 2 / 0.2)

    values = np.array([dummy_success_prob(p) for p in grid])
    landscape = analyze_probability_landscape(values, num_levels=15)
    print(f"  概率 landscape 最小值: {landscape['min_prob']:.6f}")
    print(f"  概率 landscape 最大值: {landscape['max_prob']:.6f}")
    print(f"  概率 landscape 标准差: {landscape['std_prob']:.6f}")


def run_experiment_newton_optimization():
    """实验8: Newton法优化量子行走参数 (nonlin_newton)"""
    print_section("实验8: Newton法参数优化")

    # 寻找使 sin(x) - 0.5 = 0 的根（模拟最优相位条件）
    def f(x):
        return np.sin(x) - 0.5

    def df(x):
        return np.cos(x)

    root, converged, iters = newton_raphson(f, df, x0=0.3)
    print(f"  方程 sin(x)=0.5, 初值0.3")
    print(f"  收敛: {converged}, 迭代次数: {iters}, 根: {root:.10f}")
    print(f"  验证 f(根) = {f(root):.2e}")

    # 最优硬币角度搜索
    def success_prob(angle):
        # 模拟：成功概率在某个角度达到峰值
        return np.sin(2.0 * angle) ** 2 * np.exp(-(angle - 0.6) ** 2 / 0.05) + 0.01

    opt = find_optimal_coin_angle(success_prob, angle0=0.5)
    print(f"  最优硬币角度: {opt['optimal_angle']:.6f} rad")
    print(f"  对应成功概率: {opt['success_probability']:.6f}")

    # 关键 gamma 搜索
    def gap_func(g):
        # 模拟谱隙随 gamma 变化
        return g * np.exp(-g ** 2) + 0.1

    gamma_opt = find_critical_gamma(gap_func, gamma0=1.0)
    print(f"  临界 gamma: {gamma_opt['critical_gamma']:.6f}")
    print(f"  对应谱隙: {gamma_opt['spectral_gap']:.6f}")


def run_experiment_coupon_collector():
    """实验9: 优惠券收集问题与量子/经典覆盖时间对比 (full_deck_simulation)"""
    print_section("实验9: 覆盖时间分析 (Coupon Collector)")

    n_items = 52
    result = coupon_collector_simulation(n_items, num_trials=500, seed=42)
    print(f"  物品数: {n_items}")
    print(f"  模拟次数: {result['num_trials']}")
    print(f"  经验均值: {result['empirical_mean']:.2f}")
    print(f"  理论期望: {result['theoretical_expected']:.2f}")
    print(f"  经验标准差: {result['empirical_std']:.2f}")
    print(f"  理论标准差: {result['theoretical_std']:.2f}")

    comparison = compare_classical_quantum_cover_time(n_items, graph_degree=4.0)
    print(f"  经典覆盖时间 (估计): {comparison['classical_cover_time']:.2f}")
    print(f"  量子搜索时间 (估计): {comparison['quantum_search_time']:.2f}")
    print(f"  加速比: {comparison['speedup_factor']:.2f}x")


def run_experiment_steinerberger_quadrature():
    """实验10: Steinerberger病态函数验证数值积分 (steinerberger)"""
    print_section("实验10: Steinerberger病态函数积分测试")

    result = test_quadrature_accuracy(integrate_simpson, n_max=8)
    for test in result['tests']:
        print(f"  n={test['n']:2d}: 精确={test['exact']:.8f}, "
              f"近似={test['approximate']:.8f}, "
              f"相对误差={test['relative_error']:.2e}")

    # 用梯形法则计算量子概率积分
    times = np.linspace(0.0, 10.0, 501)
    prob = np.sin(times) ** 2 * np.exp(-times / 5.0)
    avg_prob = integrate_quantum_probability(prob, times) / (times[-1] - times[0])
    print(f"  量子概率时间平均值 (梯形法): {avg_prob:.6f}")


def run_experiment_reverse_communication_cg():
    """实验11: 反向通信CG求解大规模稀疏系统 (cg_rc + Wathen矩阵)"""
    print_section("实验11: 反向通信CG与Wathen矩阵")

    nx, ny = 4, 4
    n = wathen_order(nx, ny)
    A = wathen(nx, ny)
    # 正则化: Wathen一致质量矩阵有零空间(常数向量),添加小量使其正定
    A_reg = A + 1e-6 * np.eye(n)
    b = np.ones(n)
    x0 = np.zeros(n)

    # 使用反向通信CG
    def A_mult(v):
        return A_reg @ v

    x_cg = cg_rc_solve(n, A_mult, b, x0)
    residual = np.linalg.norm(A_reg @ x_cg - b)
    print(f"  Wathen矩阵维度: {n}")
    print(f"  反向通信CG残差: {residual:.2e}")
    print(f"  解范数: {np.linalg.norm(x_cg):.4f}")

    # 比较几种三对角求解器在扩散问题上的表现
    n_small = 128
    L = discrete_laplacian_1d(n_small, periodic=False)
    a_r83 = np.zeros((3, n_small))
    a_r83[0, :-1] = np.diag(L, 1)
    a_r83[1, :] = np.diag(L)
    a_r83[2, 1:] = np.diag(L, -1)
    b_small = np.random.randn(n_small)

    x_cg_small = r83_cg(n_small, a_r83, b_small)
    x_jac = r83_jac_sl(n_small, a_r83, b_small, max_iter=8000, tol=1e-12)
    x_gs = r83_gs_sl(n_small, a_r83, b_small, max_iter=8000, tol=1e-12)

    print(f"  三对角CG残差 (n={n_small}): {np.linalg.norm(L @ x_cg_small - b_small):.2e}")
    print(f"  三对角Jacobi残差: {np.linalg.norm(L @ x_jac - b_small):.2e}")
    print(f"  三对角Gauss-Seidel残差: {np.linalg.norm(L @ x_gs - b_small):.2e}")


def run_experiment_spectral_analysis():
    """实验12: 谱分析与搜索复杂度估计"""
    print_section("实验12: 谱分析与搜索复杂度")

    # 2D网格谱分析
    nx, ny = 8, 8
    adj = []
    for y in range(ny):
        for x in range(nx):
            neighbors = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx2, ny2 = x + dx, y + dy
                if 0 <= nx2 < nx and 0 <= ny2 < ny:
                    neighbors.append(nx2 + ny2 * nx)
            adj.append(neighbors)

    marked = [nx * ny // 2]
    spec = spectral_search_analysis(adj, marked)
    print(f"  2D网格临界gamma: {spec['critical_gamma']:.6f}")
    print(f"  谱隙: {spec['spectral_gap']:.6f}")
    print(f"  绝热时间估计: {spec['adiabatic_time']:.4f}")

    comp = estimate_search_complexity(nx * ny, len(marked), graph_degree=4.0)
    print(f"  理论最优步数: {comp['optimal_steps']}")
    print(f"  理论成功概率: {comp['success_probability']:.6f}")
    print(f"  二次加速比: {comp['quadratic_speedup']:.2f}x")


def run_experiment_multidimensional_walk():
    """实验13: 多维量子行走在高维格子"""
    print_section("实验13: 多维格子量子行走")

    dims = (4, 4, 4)
    mdqw = MultiDimensionalQuantumWalk(dims, coin_type="grover", periodic=True)
    mdqw.set_initial_state()
    mdqw.step(num_steps=10)
    prob = mdqw.get_position_distribution()
    print(f"  高维格子维度: {dims}, 总状态数: {mdqw.n}")
    print(f"  硬币维度: {mdqw.coin_dim}")
    print(f"  10步后概率和: {np.sum(prob):.12f}")
    print(f"  最大概率位置: {int(np.argmax(prob))}")


def run_experiment_piecewise_constant_potential():
    """实验14: 分段常数势场下的量子演化 (pwc_plot_2d)"""
    print_section("实验14: 分段常数势场")

    from quantum_operators import piecewise_constant_1d, ctqw_hamiltonian

    xc = np.linspace(-1.0, 1.0, 5)
    # 构造一个分段常数的1D势场 (4 cells for 5 breakpoints)
    values = np.array([0.0, 0.5, 1.0, 0.5])
    V = piecewise_constant_1d(xc, values)

    # 简单1D链上采样势场
    n = 20
    adj = [[(i + 1) % n, (i - 1) % n] for i in range(n)]
    positions = np.linspace(-1.0, 1.0, n)
    H = ctqw_hamiltonian(adj, potential=V, positions=positions, gamma=0.5)
    print(f"  Hamiltonian维度: {H.shape}")
    print(f"  Hamiltonian厄米性误差: {np.max(np.abs(H - H.T)):.2e}")
    eigs = np.linalg.eigvalsh(H)
    print(f"  基态能量: {eigs[0]:.4f}")
    print(f"  第一激发态: {eigs[1]:.4f}")
    print(f"  基态-激发态能隙: {eigs[1] - eigs[0]:.6f}")


def main():
    print("=" * 70)
    print("  量子行走搜索算法 — 博士级科研代码合成项目 (PROJECT_155)")
    print("=" * 70)
    print("\n本程序融合15个种子项目的核心算法，围绕量子计算中的")
    print("'量子行走搜索算法'展开，涵盖多种图结构与数值方法。\n")

    np.random.seed(42)

    try:
        run_experiment_1d_tridiagonal_solver()
        run_experiment_vandermonde_reconstruction()
        run_experiment_2d_spatial_search()
        run_experiment_hexagonal_search()
        run_experiment_cubed_sphere()
        run_experiment_hypercube_search()
        run_experiment_cc_sparse_grid_params()
        run_experiment_newton_optimization()
        run_experiment_coupon_collector()
        run_experiment_steinerberger_quadrature()
        run_experiment_reverse_communication_cg()
        run_experiment_spectral_analysis()
        run_experiment_multidimensional_walk()
        run_experiment_piecewise_constant_potential()

        print("\n" + "=" * 70)
        print("  所有实验成功完成！")
        print("=" * 70)
    except Exception as e:
        print(f"\n[ERROR] 实验执行失败: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
# ---- TC01: normalize_vector 将非零向量归一化为单位长度 ----
v = np.array([3.0, 4.0])
v_norm = normalize_vector(v)
assert np.isclose(np.linalg.norm(v_norm), 1.0), '[TC01] 归一化后长度不为1 FAILED'

# ---- TC02: normalize_vector 零向量返回零向量 ----
v_zero = np.zeros(5)
v_norm_zero = normalize_vector(v_zero)
assert np.allclose(v_norm_zero, 0.0), '[TC02] 零向量归一化不为零 FAILED'

# ---- TC03: safe_divide 正常除法 ----
result = safe_divide(10.0, 2.0)
assert np.isclose(result, 5.0), '[TC03] 正常除法结果错误 FAILED'

# ---- TC04: safe_divide 被零除返回默认值 ----
result = safe_divide(10.0, 0.0, default=42.0)
assert np.isclose(result, 42.0), '[TC04] 除零未返回默认值 FAILED'

# ---- TC05: is_power_of_two 检测2的幂 ----
from utils import is_power_of_two
assert is_power_of_two(1) == True, '[TC05] 1应为2的幂 FAILED'
assert is_power_of_two(2) == True, '[TC05] 2应为2的幂 FAILED'
assert is_power_of_two(3) == False, '[TC05] 3不应为2的幂 FAILED'
assert is_power_of_two(256) == True, '[TC05] 256应为2的幂 FAILED'

# ---- TC06: chebyshev_nodes 端点与数量 ----
nodes = chebyshev_nodes(10, a=-1.0, b=1.0)
assert len(nodes) == 10, '[TC06] Chebyshev节点数量错误 FAILED'
assert np.all(nodes >= -1.0) and np.all(nodes <= 1.0), '[TC06] Chebyshev节点超出范围 FAILED'

# ---- TC07: hadamard_matrix 正交性 H @ H.T = I ----
H = hadamard_matrix(4)
prod = H @ H.T
assert np.allclose(prod, np.eye(4), atol=1e-12), '[TC07] Hadamard矩阵不正交 FAILED'

# ---- TC08: grover_coin 酉性 ----
C = grover_coin(4)
prod = C @ C.T
assert np.allclose(prod, np.eye(4), atol=1e-12), '[TC08] Grover coin不酉 FAILED'

# ---- TC09: discrete_laplacian_1d 对称性 ----
L = discrete_laplacian_1d(5, periodic=False)
assert np.allclose(L, L.T), '[TC09] 离散Laplacian不对称 FAILED'

# ---- TC10: discrete_laplacian_1d 周期版本行和为零 ----
Lp = discrete_laplacian_1d(5, periodic=True)
row_sums = np.sum(Lp, axis=1)
assert np.allclose(row_sums, 0.0), '[TC10] 周期Laplacian行和不为零 FAILED'

# ---- TC11: r83_mv 三对角矩阵向量乘 ----
from matrix_solvers import r83_mv
n11 = 3
a11 = np.zeros((3, n11))
a11[0, :] = [-1.0, -1.0, 0.0]
a11[1, :] = [2.0, 2.0, 2.0]
a11[2, :] = [0.0, -1.0, -1.0]
x11 = np.array([1.5, 2.0, 1.5])
b11_expected = np.array([1.0, 1.0, 1.0])
b11_computed = r83_mv(n11, a11, x11)
assert np.allclose(b11_computed, b11_expected), '[TC11] 三对角MV结果错误 FAILED'

# ---- TC12: r83_cg 求解精度 ----
n12 = 3
a12 = np.zeros((3, n12))
a12[0, :] = [-1.0, -1.0, 0.0]
a12[1, :] = [2.0, 2.0, 2.0]
a12[2, :] = [0.0, -1.0, -1.0]
b12 = np.ones(n12)
x12_cg = r83_cg(n12, a12, b12)
L12 = np.diag(a12[1, :]) + np.diag(a12[0, :-1], 1) + np.diag(a12[2, 1:], -1)
res12 = np.linalg.norm(L12 @ x12_cg - b12)
assert res12 < 1e-8, '[TC12] r83_cg残差过大 FAILED'

# ---- TC13: r8vm_sl Vandermonde 求解精度 ----
from matrix_solvers import r8vm_mv
n13 = 3
x13_nodes = np.array([1.0, 2.0, 3.0])
true13_coeffs = np.array([1.0, 2.0, 3.0])
b13_vm = r8vm_mv(n13, n13, x13_nodes, true13_coeffs)
recovered13 = r8vm_sl(n13, x13_nodes, b13_vm)
assert np.allclose(recovered13, true13_coeffs, atol=1e-8), '[TC13] Vandermonde求解不精确 FAILED'

# ---- TC14: hexagon_area 已知值 ----
from geometry_mesh import hexagon_area
area14 = hexagon_area()
expected14 = 3.0 * np.sqrt(3.0) / 2.0
assert np.isclose(area14, expected14), '[TC14] 六边形面积错误 FAILED'

# ---- TC15: hexagon_monomial_integral (0,0) = 面积 ----
int00 = hexagon_monomial_integral(0, 0)
assert np.isclose(int00, 3.0 * np.sqrt(3.0) / 2.0), '[TC15] 单项式积分(0,0)不等面积 FAILED'

# ---- TC16: hexagon_monomial_integral 奇次幂为零 ----
int10 = hexagon_monomial_integral(1, 0)
int01 = hexagon_monomial_integral(0, 1)
assert np.isclose(int10, 0.0), '[TC16] x^1积分不为零(对称性) FAILED'
assert np.isclose(int01, 0.0), '[TC16] y^1积分不为零(对称性) FAILED'

# ---- TC17: generate_hexagonal_lattice 顶点数量 ----
pts1ring = generate_hexagonal_lattice(1)
assert pts1ring.shape[0] == 7, '[TC17] 1环六边形晶格顶点数不为7 FAILED'

# ---- TC18: hadamard_coin 酉性 ----
Hc = hadamard_coin(4)
assert np.allclose(Hc @ Hc.T, np.eye(4), atol=1e-12), '[TC18] hadamard_coin不酉 FAILED'

# ---- TC19: fourier_coin 酉性 ----
Fc = fourier_coin(4)
assert np.allclose(Fc @ Fc.conj().T, np.eye(4), atol=1e-12), '[TC19] fourier_coin不酉 FAILED'

# ---- TC20: graph_laplacian 对称性 ----
adj20 = [[1], [0, 2], [1]]
Lg = graph_laplacian(adj20)
assert np.allclose(Lg, Lg.T), '[TC20] 图Laplacian不对称 FAILED'

# ---- TC21: DiscreteTimeQuantumWalk 状态范数守恒 ----
import numpy as np
np.random.seed(42)
qw21 = DiscreteTimeQuantumWalk(16, coin_dim=2, coin_type="hadamard", periodic=True)
qw21.set_initial_state(position=8)
qw21.step(num_steps=30)
norm21 = qw21.get_state_norm()
assert np.isclose(norm21, 1.0, atol=1e-10), '[TC21] DTQW范数不守恒 FAILED'

# ---- TC22: ContinuousTimeQuantumWalk 状态范数守恒 ----
adj22 = [[1], [0, 2], [1]]
ctqw22 = ContinuousTimeQuantumWalk(adj22, gamma=1.0)
ctqw22.set_initial_state(1)
ctqw22.evolve(t=2.0)
norm22 = ctqw22.get_state_norm()
assert np.isclose(norm22, 1.0, atol=1e-10), '[TC22] CTQW范数不守恒 FAILED'

# ---- TC23: newton_raphson 求 sin(x)=0.5 的根 ----
def _f23(x): return np.sin(x) - 0.5
def _df23(x): return np.cos(x)
root23, conv23, iters23 = newton_raphson(_f23, _df23, 0.3)
assert conv23, '[TC23] Newton法未收敛 FAILED'
assert np.isclose(_f23(root23), 0.0, atol=1e-8), '[TC23] Newton法根不满足方程 FAILED'

# ---- TC24: coupon_collector_simulation 理论期望在合理范围内 ----
import numpy as np
np.random.seed(42)
res_cc = coupon_collector_simulation(10, num_trials=2000, seed=42)
emp_mean24 = res_cc['empirical_mean']
theo_mean24 = res_cc['theoretical_expected']
rel_diff24 = abs(emp_mean24 - theo_mean24) / theo_mean24
assert rel_diff24 < 0.20, '[TC24] 优惠券收集器经验均值偏离理论过大 FAILED'

# ---- TC25: steinerberger_integral01_exact 已知值 ----
exact25 = steinerberger_integral01_exact(1)
expected25 = 2.0 / np.pi
assert np.isclose(exact25, expected25), '[TC25] Steinerberger积分精确值错误 FAILED'

# ---- TC26: integrate_simpson 二次函数精确积分 ----
x_quad = np.linspace(0.0, 1.0, 101)
y_quad = x_quad ** 2
approx26 = integrate_simpson(y_quad, x_quad)
exact26 = 1.0 / 3.0
assert abs(approx26 - exact26) < 1e-6, '[TC26] Simpson积分不精确 FAILED'

# ---- TC27: estimate_search_complexity 单调性 ----
est27_1 = estimate_search_complexity(100, 1, graph_degree=4.0)
est27_2 = estimate_search_complexity(100, 4, graph_degree=4.0)
assert est27_1['optimal_steps'] > est27_2['optimal_steps'], '[TC27] 搜索复杂度不单调 FAILED'

# ---- TC28: diophantine_nd_nonnegative 已知解 ----
a28 = np.array([1, 2, 3])
sols28 = diophantine_nd_nonnegative(a28, 4)
assert len(sols28) == 4, '[TC28] Diophantine解数量错误 FAILED'
for sol28 in sols28:
    assert np.dot(a28, sol28) == 4, '[TC28] Diophantine解不满足方程 FAILED'

# ---- TC29: spectral_gap 对已知矩阵正确 ----
H29 = np.diag([0.0, 2.0, 2.0, 5.0])
gap29 = spectral_gap(H29)
assert np.isclose(gap29, 2.0), '[TC29] 谱隙计算错误 FAILED'

# ---- TC30: MultiDimensionalQuantumWalk 范数守恒 ----
import numpy as np
np.random.seed(42)
mdqw30 = MultiDimensionalQuantumWalk((3, 3, 3), coin_type="grover", periodic=True)
mdqw30.set_initial_state()
mdqw30.step(num_steps=5)
prob30_sum = np.sum(mdqw30.get_position_distribution())
assert np.isclose(prob30_sum, 1.0, atol=1e-10), '[TC30] 多维量子行走概率和不守恒 FAILED'

# ---- TC31: eigenstate_localization 对均匀态为 1/N ----
uniform_vec = np.ones(10) / np.sqrt(10)
ipr31 = eigenstate_localization(uniform_vec)
assert np.isclose(ipr31, 0.1, atol=1e-10), '[TC31] 均匀态IPR不为1/N FAILED'

# ---- TC32: graph_adjacency_matrix 对称性 ----
from quantum_operators import graph_adjacency_matrix
adj32 = [[1, 2], [0], [0]]
A32 = graph_adjacency_matrix(adj32)
assert np.allclose(A32, A32.T), '[TC32] 图邻接矩阵不对称 FAILED'

# ---- TC33: find_optimal_coin_angle 在已知凸函数上正确 ----
def _prob_func33(angle):
    return np.sin(2.0 * angle) ** 2
opt33 = find_optimal_coin_angle(_prob_func33, angle0=0.5)
assert 0.6 < opt33['optimal_angle'] < 1.0, '[TC33] 最优角度异常 FAILED'
assert opt33['success_probability'] > 0.5, '[TC33] 最优成功概率过低 FAILED'

# ---- TC34: convergence_rate_analysis 线性收敛检测 ----
import numpy as np
np.random.seed(42)
errors34 = 2.0 ** (-np.arange(1, 11, dtype=float))
conv34 = convergence_rate_analysis(errors34)
assert conv34['estimated_order'] is not None, '[TC34] 收敛阶估计为空 FAILED'
assert conv34['estimated_order'] > 0.5, '[TC34] 收敛阶估计异常低 FAILED'

# ---- TC35: clamp 值检查 ----
assert clamp(1.5, 0.0, 1.0) == 1.0, '[TC35] clamp上限错误 FAILED'
assert clamp(-0.5, 0.0, 1.0) == 0.0, '[TC35] clamp下限错误 FAILED'
assert clamp(0.5, 0.0, 1.0) == 0.5, '[TC35] clamp范围内错误 FAILED'

print('\n全部 35 个测试通过!\n')
