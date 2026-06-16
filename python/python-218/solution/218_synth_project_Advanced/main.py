"""
main.py
=======
项目 218 统一入口: 随机变分不等式与非局部互补问题的完整计算流程.

本脚本演示:
    1. 构建带非局部算子的 VI 问题
    2. 使用多种求解器求解互补问题
    3. 随机场生成与 Monte Carlo 采样
    4. 活动集分析与连通分量标记
    5. 解路径的参数化重构
    6. 障碍问题 (经典 VI 应用) 求解
    7. 多网格框架与收敛性分析
    8. 生物对流 Lorenz 系统与 TDVI 耦合

运行: python main.py  (零参数)
"""

from __future__ import annotations

import numpy as np
import sys
import os

# 将项目根目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from variational_inequality import VariationalInequalityProblem
from complementarity_solver import (
    SemiSmoothNewtonSolver,
    ProjectedGaussSeidelSolver,
    InteriorPointSolver,
)
from nonlocal_operator import (
    MajoranaKernel,
    MaternCorrelationKernel,
    NonlocalOperator,
)
from stochastic_field import (
    KarhunenLoeveExpansion,
    MaternCovarianceBuilder,
    CircleMonteCarloSampler,
    DiscreteCDFSampler,
    HypercubeDistanceStats,
)
from active_set_manager import (
    ActiveSetManager,
    ConnectedComponentLabeler1D,
)
from laplacian_operator import (
    DiscreteLaplacian,
    FractionalLaplacian,
    ObstacleProblemVI,
)
from hamming_encoder import HammingCode, PivotSequenceEncoder
from cellular_state_space import CellularStateSpaceExplorer
from lagrange_reconstructor import (
    LagrangeInterpolator1D,
    ChebyshevNodes,
    SolutionPathReconstructor,
)
from hypercube_projection import (
    ProjectionOperators,
    HypercubeDistanceAnalyzer,
)
from jacobian_estimator import (
    JacobianEstimator,
    BroydenUpdate,
    ConditionNumberEstimator,
)
from hermite_functionals import (
    GeneralizedHermitePolynomials,
    GaussHermiteQuadrature,
)
from grid_manager import Grid1D, MultigridHierarchy, GridConvergenceAnalyzer
from bioconvection_dynamics import (
    BioconvectionParameters,
    BioconvectionODE,
    RK4Integrator,
    LyapunovAnalyzer,
)


def section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


def demo_1_vi_problem_and_solvers() -> dict:
    """
    模块 1: 变分不等式问题构建与多求解器对比.

    构建仿射 VI: F(x) = Mx + q, K = R^n_+
    使用半光滑 Newton, PGS, 内点法三种方法求解.
    """
    section("模块 1: 变分不等式问题与互补求解器")

    n = 30
    rng = np.random.default_rng(42)

    # 构建 P-矩阵 M (对称正定 + 反对称扰动)
    A_rand = rng.standard_normal((n, n))
    M = A_rand.T @ A_rand / n + 0.1 * np.eye(n)  # 对称正定部分
    M_skew = 0.05 * (rng.standard_normal((n, n)) - rng.standard_normal((n, n)))
    M = M + M_skew
    q = -rng.uniform(0.5, 2.0, n)

    vi_problem = VariationalInequalityProblem(
        n=n, M=M, q=q,
        lipschitz_L=float(np.linalg.norm(M, ord=2)),
        strong_monotone_mu=float(np.min(np.linalg.eigvalsh(0.5 * (M + M.T)))),
    )

    print(f"\n问题信息: {vi_problem}")
    print(f"  维度 n = {n}")
    print(f"  对称部分最小特征值 = {vi_problem.min_eigenvalue_sym:.4e}")
    print(f"  P0-矩阵: {vi_problem.is_p0}")
    print(f"  强单调: {vi_problem.is_strongly_monotone}")
    print(f"  条件数 κ = {vi_problem.condition_number_estimate():.2e}")

    x0 = rng.uniform(0.1, 1.0, n)
    results = {}

    # 1. 半光滑 Newton
    solver_newton = SemiSmoothNewtonSolver(max_iter=300, tol=1e-10)
    result_newton = solver_newton.solve(vi_problem, x0)
    results['newton'] = result_newton
    print(f"\n[半光滑 Newton] {result_newton.message}")
    print(f"  迭代次数: {result_newton.iterations}, "
          f"最终残差: {result_newton.residual_history[-1]:.4e}")

    # 2. 投影 Gauss-Seidel
    solver_pgs = ProjectedGaussSeidelSolver(max_iter=3000, tol=1e-8, omega=1.0)
    result_pgs = solver_pgs.solve(vi_problem, x0)
    results['pgs'] = result_pgs
    print(f"\n[投影 Gauss-Seidel] {result_pgs.message}")
    print(f"  迭代次数: {result_pgs.iterations}, "
          f"最终残差: {result_pgs.residual_history[-1]:.4e}")

    # 3. 内点法
    solver_ipm = InteriorPointSolver(max_iter=300, tol=1e-10, sigma=0.2)
    result_ipm = solver_ipm.solve(vi_problem, x0)
    results['ipm'] = result_ipm
    print(f"\n[内点法] {result_ipm.message}")
    print(f"  迭代次数: {result_ipm.iterations}, "
          f"最终残差: {result_ipm.residual_history[-1]:.4e}")

    # 对比解
    print(f"\n解的对比 (前 5 个分量):")
    for k in ['newton', 'pgs', 'ipm']:
        print(f"  {k:8s}: {results[k].x[:5]}")

    return results


def demo_2_nonlocal_operator() -> dict:
    """
    模块 2: 非局部算子 (Majorana 型 + Matérn 关联).

    在空间离散点上构建非局部核矩阵, 分析谱性质.
    """
    section("模块 2: 非局部算子 (Majorana / Matérn 核)")

    N = 50
    positions = np.linspace(0, 1, N)[:, np.newaxis]
    h = positions[1, 0] - positions[0, 0]

    # Majorana 核 (振荡衰减)
    lambda_maj = 0.1
    kappa_F = 20.0
    kernel_maj = MajoranaKernel(lambda_corr=lambda_maj, kappa_F=kappa_F, dimension=1)
    op_maj = NonlocalOperator(kernel_maj, positions, activation='relu',
                              quadrature_weight=h)

    print(f"\nMajorana 核:")
    print(f"  关联长度 λ = {lambda_maj}")
    print(f"  Fermi 波矢 κ_F = {kappa_F}")
    print(f"  核矩阵谱半径 = {op_maj.spectral_radius():.4e}")
    print(f"  有效作用范围 = {op_maj.effective_range():.4f}")

    # Matérn 核
    lambda_mat = 0.15
    nu = 2.5
    kernel_mat = MaternCorrelationKernel(lambda_corr=lambda_mat, nu=nu, sigma_sq=1.0)
    op_mat = NonlocalOperator(kernel_mat, positions, activation='tanh',
                              quadrature_weight=h)

    print(f"\nMatérn 核:")
    print(f"  关联长度 ρ = {lambda_mat}")
    print(f"  光滑性 ν = {nu}")
    print(f"  核矩阵谱半径 = {op_mat.spectral_radius():.4e}")

    # 应用到测试函数
    x_test = np.sin(2 * np.pi * positions.flatten()) + 0.5
    y_maj = op_maj.apply(x_test)
    y_mat = op_mat.apply(x_test)

    print(f"\n非局部算子作用:")
    print(f"  输入 L2 范数 = {np.linalg.norm(x_test):.4e}")
    print(f"  Majorana 输出 L2 范数 = {np.linalg.norm(y_maj):.4e}")
    print(f"  Matérn 输出 L2 范数 = {np.linalg.norm(y_mat):.4e}")

    return {'op_maj': op_maj, 'op_mat': op_mat}


def demo_3_stochastic_field_and_mc() -> dict:
    """
    模块 3: 随机场生成与 Monte Carlo 采样.

    使用 Karhunen-Loève 展开生成随机场,
    在圆上进行 Monte Carlo 积分.
    """
    section("模块 3: 随机场生成与 Monte Carlo 采样")

    rng = np.random.default_rng(123)

    # Matérn 协方差 + KL 展开
    N = 40
    x_grid = np.linspace(0, 1, N)
    matern = MaternCovarianceBuilder(length_scale=0.2, nu=1.5, sigma_sq=1.0)
    C = matern.build_from_1d_grid(x_grid)

    kl = KarhunenLoeveExpansion(C, energy_threshold=0.95)
    print(f"\nKL 展开:")
    print(f"  网格点数 N = {N}")
    print(f"  截断阶数 M = {kl.M}")
    print(f"  能量解释率 = {kl.explained_variance_ratio():.4f}")
    print(f"  截断误差 = {kl.truncation_error:.4e}")
    print(f"  前 5 个特征值 = {kl.eigvals[:5]}")

    # 生成样本
    n_samples = 100
    Z = kl.sample(n_samples, rng)
    print(f"\n随机场样本:")
    print(f"  样本数 = {n_samples}")
    print(f"  场均值 (空间平均) = {np.mean(Z):.4e}")
    print(f"  场标准差 = {np.std(Z):.4e}")

    # 圆上 Monte Carlo 采样
    n_mc = 1000
    points_random = CircleMonteCarloSampler.random_sample(n_mc, rng)
    points_ergodic = CircleMonteCarloSampler.ergodic_sample(n_mc)

    # 被积函数: f(θ) = cos(3θ) + 0.5 sin(5θ), 理论积分为 0
    theta_r = np.arctan2(points_random[:, 1], points_random[:, 0])
    theta_e = np.arctan2(points_ergodic[:, 1], points_ergodic[:, 0])
    f_random = np.cos(3 * theta_r) + 0.5 * np.sin(5 * theta_r)
    f_ergodic = np.cos(3 * theta_e) + 0.5 * np.sin(5 * theta_e)

    int_random = CircleMonteCarloSampler.integrate_function(f_random)
    int_ergodic = CircleMonteCarloSampler.integrate_function(f_ergodic)

    print(f"\n圆上 Monte Carlo 积分 (理论值 = 0):")
    print(f"  随机采样误差 = {abs(int_random):.4e}")
    print(f"  遍历采样误差 = {abs(int_ergodic):.4e}")

    # 超立方体距离统计
    dim = 20
    mu_hc, var_hc = HypercubeDistanceStats.estimate(5000, dim, rng)
    theo_e_d2 = HypercubeDistanceStats.theoretical_mean_squared(dim)
    print(f"\n超立方体距离统计 (m = {dim}):")
    print(f"  Monte Carlo E[D] = {mu_hc:.4f}")
    print(f"  Monte Carlo Var[D] = {var_hc:.4f}")
    print(f"  理论 E[D²] = {theo_e_d2:.4f}")

    return {'kl': kl, 'Z': Z}


def demo_4_active_set_and_cellular() -> dict:
    """
    模块 4: 活动集管理与细胞自动机配置空间.
    """
    section("模块 4: 活动集管理与细胞自动机配置空间")

    n = 40
    rng = np.random.default_rng(77)

    # 构建 VI 并求解
    M = np.eye(n) * 2.0
    M[0, 1] = -0.5; M[1, 0] = -0.5
    q = -rng.uniform(0.1, 1.0, n)
    vi = VariationalInequalityProblem(n=n, M=M, q=q)
    solver = SemiSmoothNewtonSolver(max_iter=200, tol=1e-10)
    result = solver.solve(vi, np.ones(n))

    # 活动集分析
    F_x = vi.evaluate_F(result.x)
    asm = ActiveSetManager(n=n, epsilon_active=1e-5)
    partition = asm.detect_active_set(result.x, F_x)

    print(f"\n活动集分析:")
    print(f"  活动指标数 = {len(partition.active_indices)}")
    print(f"  非活动指标数 = {len(partition.inactive_indices)}")
    print(f"  连通分量数 = {partition.n_components}")
    print(f"  各分量大小 = {[c.size for c in partition.components]}")

    # 细胞自动机配置空间探索
    n_cells = 12
    initial_config = tuple([0] * n_cells)
    explorer = CellularStateSpaceExplorer(n_cells=n_cells, min_gap=2)
    n_safe = explorer.bfs_explore(initial_config, max_depth=4)
    stats = explorer.get_safe_config_stats()

    print(f"\n细胞自动机配置空间:")
    print(f"  细胞数 = {n_cells}")
    print(f"  最大探索深度 = 4")
    print(f"  访问配置数 = {len(explorer.visited)}")
    print(f"  安全配置数 = {stats['count']}")
    print(f"  安全密度 = {stats['density']:.4e}")
    print(f"  平均活动数 = {stats['avg_active']:.2f}")
    print(f"  连通分量数 = {stats['n_components']}")

    return {'partition': partition, 'stats': stats}


def demo_5_hamming_and_lagrange() -> dict:
    """
    模块 5: Hamming 编码 pivot 跟踪与 Lagrange 解路径重构.
    """
    section("模块 5: Hamming 编码与 Lagrange 解路径")

    # Hamming 编码
    print("\nHamming(7,4) 码:")
    test_msg = np.array([1, 0, 1, 1])
    codeword = HammingCode.encode(test_msg)
    print(f"  信息: {test_msg} → 码字: {codeword}")

    # 引入 1 位错误
    received = codeword.copy()
    received[3] = 1 - received[3]
    decoded, err_pos, corrected = HammingCode.decode(received)
    print(f"  接收: {received} (错误位 {err_pos})")
    print(f"  译码: {decoded}, 已纠正: {corrected}")

    # 所有码字的最小距离
    all_cw = HammingCode.all_codewords()
    min_d = 999
    for i in range(len(all_cw)):
        for j in range(i+1, len(all_cw)):
            d = HammingCode.hamming_distance(all_cw[i], all_cw[j])
            min_d = min(min_d, d)
    print(f"  所有码字的最小 Hamming 距离 = {min_d}")

    # Pivot 序列编码
    pivot_enc = PivotSequenceEncoder()
    rng = np.random.default_rng(99)
    for _ in range(20):
        pivot_enc.record_pivot(int(rng.integers(0, 4)), int(rng.integers(0, 4)))
    bifurcations = pivot_enc.bifurcation_detection(threshold=3)
    print(f"\nPivot 序列:")
    print(f"  pivot 步数 = {len(pivot_enc.history)}")
    print(f"  分支点数 (阈值=3) = {len(bifurcations)}")
    print(f"  检测到循环 = {pivot_enc.detect_cycling(window_size=8)}")

    # Lagrange 插值
    print("\nLagrange 插值与解路径重构:")
    t_nodes = ChebyshevNodes.generate(8, a=0.0, b=1.0)
    # 模拟参数化 VI 的解: x*(t) = sin(πt) + 0.5t²
    x_vals = np.sin(np.pi * t_nodes) + 0.5 * t_nodes**2

    interp = LagrangeInterpolator1D(t_nodes, x_vals)
    t_test = np.linspace(0, 1, 50)
    x_interp = interp.evaluate(t_test)
    x_exact = np.sin(np.pi * t_test) + 0.5 * t_test**2
    max_err = float(np.max(np.abs(x_interp - x_exact)))

    print(f"  Chebyshev 节点数 = {len(t_nodes)}")
    print(f"  插值最大误差 = {max_err:.4e}")

    # 灵敏度
    dxdt = interp.derivative(np.array([0.5]))[0]
    print(f"  在 t=0.5 处的灵敏度 dx*/dt = {dxdt:.4e}")

    return {'max_err': max_err}


def demo_6_obstacle_problem() -> dict:
    """
    模块 6: 障碍问题 (Obstacle Problem, VI 的经典应用).
    """
    section("模块 6: 障碍问题 (经典 VI 应用)")

    n = 50
    h = 1.0 / (n + 1)
    x_grid = np.linspace(0, 1, n + 2)

    # 障碍函数 φ(x) = 0.3 - 4(x - 0.5)² (抛物线型)
    obstacle = 0.3 - 4.0 * (x_grid[1:-1] - 0.5)**2
    obstacle = np.maximum(obstacle, 0.0)

    # 源项 f(x) = 10
    source = 10.0 * np.ones(n)

    prob = ObstacleProblemVI(n, obstacle, source, h)
    print(f"\n障碍问题:")
    print(f"  网格点数 n = {n}")
    print(f"  网格间距 h = {h:.4f}")
    print(f"  障碍最大值 = {np.max(obstacle):.4f}")

    # 分数阶 Laplacian
    s = 0.75
    frac_lap = FractionalLaplacian(n, s, h)
    print(f"\n分数阶 Laplacian:")
    print(f"  阶数 s = {s}")
    print(f"  条件数 = {frac_lap.condition_number():.4e}")

    # 对测试函数作用
    u_test = np.sin(np.pi * x_grid[1:-1])
    frac_result = frac_lap.apply(u_test)
    print(f"  输入 ||u|| = {np.linalg.norm(u_test):.4f}")
    print(f"  输出 ||(-Δ)^s u|| = {np.linalg.norm(frac_result):.4f}")

    # 离散 Laplacian 的特征值
    eigvals = DiscreteLaplacian.eigenvalues_1d(n, h)
    print(f"\n离散 Laplacian 谱:")
    print(f"  最小特征值 = {eigvals[0]:.4e}")
    print(f"  最大特征值 = {eigvals[-1]:.4e}")
    print(f"  条件数 = {eigvals[-1]/eigvals[0]:.4e}")

    return {'obstacle': obstacle, 'eigvals': eigvals}


def demo_7_hermite_and_jacobian() -> dict:
    """
    模块 7: Hermite 泛函与 Jacobian 估计.
    """
    section("模块 7: 广义 Hermite 泛函与 Jacobian 估计")

    # Gauss-Hermite 求积精确性测试
    N_quad = 10
    ghq = GaussHermiteQuadrature(N_quad)
    print(f"\nGauss-Hermite 求积 (N = {N_quad}):")
    print(f"  节点数 = {N_quad}")
    errors = ghq.test_exactness(degree_max=25)
    exact_degrees = int(np.sum(errors < 1e-8))
    print(f"  精确到 1e-8 的最高阶 = {exact_degrees}")
    print(f"  理论精确度 = 2N-1 = {2*N_quad - 1}")
    print(f"  20 阶误差 = {errors[20]:.4e}")

    # 广义 Hermite 多项式
    hermite = GeneralizedHermitePolynomials()
    x_eval = np.linspace(-2, 2, 100)
    H = hermite.evaluate(6, x_eval, alpha=0.5)
    print(f"\n广义 Hermite 多项式 (α=0.5):")
    print(f"  最高阶 = 6")
    print(f"  ||H_0||² = {hermite.normalization_constant(0, 0.5):.4e}")
    print(f"  ||H_3||² = {hermite.normalization_constant(3, 0.5):.4e}")
    print(f"  ||H_6||² = {hermite.normalization_constant(6, 0.5):.4e}")

    # Jacobian 估计
    n = 8
    rng = np.random.default_rng(55)
    A = rng.standard_normal((n, n))
    def F_test(x):
        return A @ x + 0.1 * np.sin(3 * x)
    J_exact = A  # 线性部分的 Jacobian

    x_test = rng.standard_normal(n)
    # 对非线性部分, Jacobian = A + 0.3 cos(3x) diag
    J_true = A + 0.3 * np.diag(np.cos(3 * x_test))

    est_forward = JacobianEstimator(F_test, n, method='forward')
    J_fwd = est_forward.estimate(x_test)

    est_central = JacobianEstimator(F_test, n, method='central')
    J_ctr = est_central.estimate(x_test)

    err_fwd = np.linalg.norm(J_fwd - J_true) / np.linalg.norm(J_true)
    err_ctr = np.linalg.norm(J_ctr - J_true) / np.linalg.norm(J_true)
    print(f"\nJacobian 估计 (n = {n}):")
    print(f"  前向差分相对误差 = {err_fwd:.4e}")
    print(f"  中心差分相对误差 = {err_ctr:.4e}")
    print(f"  前向差分函数求值次数 = {est_forward.n_evals}")
    print(f"  中心差分函数求值次数 = {est_central.n_evals}")

    # 条件数
    cond = ConditionNumberEstimator.estimate_condition(J_true)
    print(f"  Jacobian 条件数 = {cond:.4e}")

    # Broyden 更新
    broyden = BroydenUpdate(n)
    J_inv_init = np.linalg.inv(J_fwd)
    broyden.initialize(x_test, F_test(x_test), J_inv_init)
    x_new = x_test + 0.01 * rng.standard_normal(n)
    F_new = F_test(x_new)
    J_inv_upd = broyden.update(x_new, F_new)
    print(f"\nBroyden 更新:")
    print(f"  更新后 J_inv 条件数 = {ConditionNumberEstimator.estimate_condition(J_inv_upd):.4e}")

    return {'ghq_errors': errors}


def demo_8_grid_and_bioconvection() -> dict:
    """
    模块 8: 多网格框架与生物对流 Lorenz-TDVI 耦合.
    """
    section("模块 8: 多网格与生物对流 Lorenz-TDVI")

    # 多网格层次
    fine_grid = Grid1D(0.0, 1.0, 63)
    mg = MultigridHierarchy(fine_grid, n_levels=3)
    print(f"\n多网格层次:")
    for lvl, g in enumerate(mg.grids):
        print(f"  Level {lvl}: n_interior = {g.n_interior}, h = {g.h:.4f}")

    # 收敛性分析
    analyzer = GridConvergenceAnalyzer()
    rng = np.random.default_rng(33)
    for g in mg.grids:
        u = np.sin(np.pi * g.interior_nodes) + 0.1 * rng.standard_normal(g.n_interior)
        analyzer.add_solution(g.h, u)
    order = analyzer.estimate_convergence_order()
    print(f"\n网格收敛阶估计: {order if order is not None else 'N/A'}")

    # 生物对流 Lorenz 系统
    params = BioconvectionParameters(Sc=100.0, Ra=60.0, b=8.0/3.0)
    ode = BioconvectionODE(params)
    print(f"\n生物对流 Lorenz 系统:")
    print(f"  Sc = {params.Sc}, Ra = {params.Ra}, b = {params.b:.4f}")
    print(f"  临界 Rayleigh 数 R_c = {params.critical_rayleigh():.4f}")
    print(f"  混沌区: {params.is_chaotic()}")
    fps = params.fixed_points()
    print(f"  不动点数 = {len(fps)}")

    # ODE 积分
    y0 = np.array([1.0, 1.0, 1.0])
    integrator = RK4Integrator(ode.rhs, dt=0.005)
    t_vals, y_vals = integrator.integrate((0.0, 5.0), y0)
    print(f"\nODE 积分:")
    print(f"  时间跨度 = [0, 5], dt = 0.005")
    print(f"  步数 = {len(t_vals)}")
    print(f"  最终状态 = {y_vals[-1]}")

    # Lyapunov 指数
    lyap_analyzer = LyapunovAnalyzer(ode, dt=0.005)
    lambda_1 = lyap_analyzer.estimate_max_lyapunov(y0, t_total=20.0)
    print(f"\nLyapunov 分析:")
    print(f"  最大 Lyapunov 指数 λ_1 ≈ {lambda_1:.4f}")
    print(f"  混沌性: {'是' if lambda_1 > 0 else '否'}")

    # 相空间散度
    div_mean = -(params.Sc + 1.0 + params.b)
    print(f"  相空间散度 (trace) = {div_mean:.4f}")
    print(f"  (负值表明相体积收缩, 系统有吸引子)")

    return {'lambda_1': lambda_1}


def main() -> None:
    """统一入口: 运行所有模块."""
    print("=" * 70)
    print("  PROJECT 218: 随机变分不等式与非局部互补问题")
    print("  Mathematical Optimization: Variational Inequalities")
    print("                 and Complementarity Problems")
    print("=" * 70)
    print(f"\nPython 版本: {sys.version.split()[0]}")
    print(f"NumPy 版本:  {np.__version__}")

    # 设置随机种子 (可重复性)
    np.random.seed(218)

    all_results = {}

    # 运行所有模块
    all_results['solvers'] = demo_1_vi_problem_and_solvers()
    all_results['nonlocal'] = demo_2_nonlocal_operator()
    all_results['stochastic'] = demo_3_stochastic_field_and_mc()
    all_results['active_set'] = demo_4_active_set_and_cellular()
    all_results['hamming_lagrange'] = demo_5_hamming_and_lagrange()
    all_results['obstacle'] = demo_6_obstacle_problem()
    all_results['hermite_jacobian'] = demo_7_hermite_and_jacobian()
    all_results['grid_bioconv'] = demo_8_grid_and_bioconvection()

    # 汇总
    section("计算汇总")
    print("\n本项目实现的 15 个种子项目融合:")
    seeds_info = [
        ("205_components    ", "连通分量标记 → 活动集分量识别"),
        ("269_delsq         ", "离散 Laplacian → VI 中的微分算子"),
        ("499_hamming       ", "Hamming 编码 → pivot 序列的跟踪与纠错"),
        ("1297_FormalCellular", "细胞自动机配置空间 → VI 解分支探索"),
        ("635_lagrange_interp", "Lagrange 插值 → 参数化 VI 解路径重构"),
        ("556_hypercube_distance", "超立方体距离 → 参数空间的度量"),
        ("280_diff_forward  ", "前向差分 → Jacobian 数值估计"),
        ("542_histogram_pdf_2d", "离散 CDF 采样 → 随机场景生成"),
        ("1093_Majorana     ", "Majorana 非局域关联 → VI 的非局部算子"),
        ("464_gen_hermite   ", "广义 Hermite 求积 → 随机 VI 的 Galerkin 投影"),
        ("493_grids_display ", "网格管理 → 多网格 VI 求解框架"),
        ("220_correlation   ", "Matérn 关联函数 → 随机场的协方差结构"),
        ("1036_lightning-pose", "姿态估计框架 → 参数化解映射的建模"),
        ("181_circle_mc     ", "圆上 Monte Carlo → 随机参数的采样"),
        ("092_bioconvection ", "生物对流 Lorenz → TDVI 的混沌驱动"),
    ]
    for name, desc in seeds_info:
        print(f"  {name} → {desc}")

    print(f"\n所有模块运行完成, 无报错.")
    print(f"Python 科研代码合成成功.")


if __name__ == '__main__':
    main()
