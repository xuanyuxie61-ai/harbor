import numpy as np
import sys




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
    print_section("实验1: 1D量子行走与三对角/Toeplitz求解器")

    n = 64
    qw = DiscreteTimeQuantumWalk(n, coin_dim=2, coin_type="hadamard", periodic=True)
    qw.set_initial_state(position=n // 2)
    qw.step(num_steps=50)

    prob = qw.get_position_distribution()
    norm = qw.get_state_norm()
    print(f"  状态范数守恒: {norm:.12f} (应 ≈ 1.0)")
    print(f"  50步后概率分布均值: {np.mean(prob):.6f}, 标准差: {np.std(prob):.6f}")


    L = discrete_laplacian_1d(n, periodic=False)
    a = np.zeros((3, n))
    a[0, :-1] = np.diag(L, 1)
    a[1, :] = np.diag(L)
    a[2, 1:] = np.diag(L, -1)
    b = np.ones(n)


    x_cg = r83_cg(n, a, b)

    a_cr = r83_cr_fa(n, a)
    x_cr = r83_cr_sl(n, a_cr, b)

    x_jac = r83_jac_sl(n, a, b, max_iter=5000)

    x_gs = r83_gs_sl(n, a, b, max_iter=5000)

    print(f"  CG残差:        {np.linalg.norm(L @ x_cg - b):.2e}")
    print(f"  循环约化残差:  {np.linalg.norm(L @ x_cr - b):.2e}")
    print(f"  Jacobi残差:    {np.linalg.norm(L @ x_jac - b):.2e}")
    print(f"  Gauss-Seidel残差: {np.linalg.norm(L @ x_gs - b):.2e}")


    a_toep = np.zeros(2 * n - 1)
    a_toep[0] = 2.0
    a_toep[1] = -1.0
    a_toep[n] = -1.0
    x_toep = r8to_sl(n, a_toep, b)
    print(f"  Toeplitz求解器残差: {np.linalg.norm(L @ x_toep - b):.2e}")


def run_experiment_vandermonde_reconstruction():
    print_section("实验2: Vandermonde谱重构")

    n = 12

    x_nodes = chebyshev_nodes(n, a=-1.0, b=1.0)

    true_coeffs = np.exp(-np.arange(n) ** 2 / 10.0)

    from matrix_solvers import r8vm_mv
    b = r8vm_mv(n, n, x_nodes, true_coeffs)
    recovered = r8vm_sl(n, x_nodes, b)
    err = np.linalg.norm(recovered - true_coeffs)
    print(f"  重构误差 (Vandermonde): {err:.2e}")
    print(f"  条件数估计: {np.linalg.cond(np.vander(x_nodes, n)):.2e}")


def run_experiment_2d_spatial_search():
    print_section("实验3: 2D网格空间搜索")

    nx, ny = 16, 16
    marked = [(nx // 2, ny // 2), (nx // 2 + 1, ny // 2)]
    result = spatial_search_2d_grid(nx, ny, marked, max_steps=150)
    print(f"  网格大小: {result['grid_size']}")
    print(f"  标记顶点: {result['marked_vertices']}")
    print(f"  最优步数: {result['optimal_steps']}")
    print(f"  最大成功概率: {result['max_success_probability']:.6f}")


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
    print_section("实验4: 六边形晶格搜索与六边形积分")

    n_rings = 3
    marked = [0, 7, 14]
    result = hexagonal_lattice_search(n_rings, marked, max_steps=60)
    print(f"  六边形晶格顶点数: {result['num_vertices']}")
    print(f"  最优步数: {result['optimal_steps']}")
    print(f"  最大成功概率: {result['max_success_probability']:.6f}")


    def gaussian_bump(x, y):
        return np.exp(-(x ** 2 + y ** 2) / 0.5)

    val = integrate_hexagon(gaussian_bump, rule=4)
    exact_ref = 1.71265450069
    print(f"  六边形高斯积分 (Stroud rule 4): {val:.8f}")
    print(f"  与参考值误差: {abs(val - exact_ref):.2e}")


    for p, q in [(0, 0), (2, 0), (0, 2), (2, 2)]:
        exact = hexagon_monomial_integral(p, q)
        print(f"    x^{p} y^{q} 精确积分: {exact:.8f}")


def run_experiment_cubed_sphere():
    print_section("实验5: 球面立方网格量子行走")

    n = 4
    points, lines = generate_cubed_sphere_grid(n)
    adj = cubed_sphere_adjacency(points, lines)
    num_pts = points.shape[0]
    num_lines = lines.shape[0]
    print(f"  球面网格点数: {num_pts} (理论: {sphere_cubed_grid_point_count(n)})")
    print(f"  球面网格边数: {num_lines} (理论: {sphere_cubed_grid_line_count(n)})")


    ctqw = ContinuousTimeQuantumWalk(adj, gamma=1.0)
    ctqw.set_initial_state(0)
    ctqw.evolve(t=5.0)
    prob = ctqw.get_position_distribution()
    print(f"  演化后概率和: {np.sum(prob):.12f}")
    print(f"  最大概率位置: {int(np.argmax(prob))}")
    print(f"  概率熵: {-np.sum(prob * np.log(prob + 1e-16)):.4f}")


    L = graph_laplacian(adj)
    eigs = np.linalg.eigvalsh(L)
    print(f"  Laplacian谱隙: {eigs[1] - eigs[0]:.6f}")
    print(f"  最大特征值: {eigs[-1]:.4f}")


def run_experiment_hypercube_search():
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


    a = np.array([1, 2, 3, 4, 5])
    b = 20
    bounds = np.array([5, 5, 4, 3, 2])
    sols_bounded = diophantine_nd_nonnegative_bounded(a, b, bounds)
    sols_free = diophantine_nd_nonnegative(a, b)
    print(f"  Diophantine a·x={b}, 有界解数量: {len(sols_bounded)}")
    print(f"  Diophantine a·x={b}, 无界解数量: {len(sols_free)}")


    verts = build_hypercube_states(n, dim)
    print(f"  超立方体顶点坐标维度: {verts.shape}")


def run_experiment_cc_sparse_grid_params():
    print_section("实验7: 稀疏网格参数采样与优化")

    dim = 3
    max_level = 4
    grid = generate_cc_sparse_grid(dim, max_level)
    print(f"  稀疏网格点数 (dim={dim}, level<={max_level}): {grid.shape[0]}")


    weights = np.array([1.0, 2.0, 1.5])
    grid_w = constrained_parameter_grid(dim, max_level, weights)
    print(f"  加权约束网格点数: {grid_w.shape[0]}")


    def dummy_success_prob(params):

        p1, p2 = params[0], params[1]
        return np.exp(-(p1 - 0.3) ** 2 / 0.1 - (p2 + 0.1) ** 2 / 0.2)

    values = np.array([dummy_success_prob(p) for p in grid])
    landscape = analyze_probability_landscape(values, num_levels=15)
    print(f"  概率 landscape 最小值: {landscape['min_prob']:.6f}")
    print(f"  概率 landscape 最大值: {landscape['max_prob']:.6f}")
    print(f"  概率 landscape 标准差: {landscape['std_prob']:.6f}")


def run_experiment_newton_optimization():
    print_section("实验8: Newton法参数优化")


    def f(x):
        return np.sin(x) - 0.5

    def df(x):
        return np.cos(x)

    root, converged, iters = newton_raphson(f, df, x0=0.3)
    print(f"  方程 sin(x)=0.5, 初值0.3")
    print(f"  收敛: {converged}, 迭代次数: {iters}, 根: {root:.10f}")
    print(f"  验证 f(根) = {f(root):.2e}")


    def success_prob(angle):

        return np.sin(2.0 * angle) ** 2 * np.exp(-(angle - 0.6) ** 2 / 0.05) + 0.01

    opt = find_optimal_coin_angle(success_prob, angle0=0.5)
    print(f"  最优硬币角度: {opt['optimal_angle']:.6f} rad")
    print(f"  对应成功概率: {opt['success_probability']:.6f}")


    def gap_func(g):

        return g * np.exp(-g ** 2) + 0.1

    gamma_opt = find_critical_gamma(gap_func, gamma0=1.0)
    print(f"  临界 gamma: {gamma_opt['critical_gamma']:.6f}")
    print(f"  对应谱隙: {gamma_opt['spectral_gap']:.6f}")


def run_experiment_coupon_collector():
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
    print_section("实验10: Steinerberger病态函数积分测试")

    result = test_quadrature_accuracy(integrate_simpson, n_max=8)
    for test in result['tests']:
        print(f"  n={test['n']:2d}: 精确={test['exact']:.8f}, "
              f"近似={test['approximate']:.8f}, "
              f"相对误差={test['relative_error']:.2e}")


    times = np.linspace(0.0, 10.0, 501)
    prob = np.sin(times) ** 2 * np.exp(-times / 5.0)
    avg_prob = integrate_quantum_probability(prob, times) / (times[-1] - times[0])
    print(f"  量子概率时间平均值 (梯形法): {avg_prob:.6f}")


def run_experiment_reverse_communication_cg():
    print_section("实验11: 反向通信CG与Wathen矩阵")

    nx, ny = 4, 4
    n = wathen_order(nx, ny)
    A = wathen(nx, ny)

    A_reg = A + 1e-6 * np.eye(n)
    b = np.ones(n)
    x0 = np.zeros(n)


    def A_mult(v):
        return A_reg @ v

    x_cg = cg_rc_solve(n, A_mult, b, x0)
    residual = np.linalg.norm(A_reg @ x_cg - b)
    print(f"  Wathen矩阵维度: {n}")
    print(f"  反向通信CG残差: {residual:.2e}")
    print(f"  解范数: {np.linalg.norm(x_cg):.4f}")


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
    print_section("实验12: 谱分析与搜索复杂度")


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
    print_section("实验14: 分段常数势场")

    from quantum_operators import piecewise_constant_1d, ctqw_hamiltonian

    xc = np.linspace(-1.0, 1.0, 5)

    values = np.array([0.0, 0.5, 1.0, 0.5])
    V = piecewise_constant_1d(xc, values)


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
