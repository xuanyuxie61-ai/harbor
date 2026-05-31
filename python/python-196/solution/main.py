"""
main.py
异构HPC集群热-电耦合模拟任务调度系统 — 统一入口

运行方式:
    python main.py

零参数可运行，自动完成以下流程：
    1) 构建异构平台（CPU/GPU/FPGA）
    2) 生成随机任务负载集
    3) 训练性能代理模型（Chebyshev + 最小二乘）
    4) 运行蒙特卡洛不确定性量化（椭圆/超球体/超立方体采样）
    5) 求解2D FEM热场（Poisson方程）
    6) 网格自适应细化与变换
    7) 执行贪心调度与整数规划任务映射
    8) 计算数值积分（Vandermonde + 金字塔积分）
    9) 输出调度指标与误差分析
"""

import numpy as np
import time

from utils import (
    hypersphere_surface_area, hypersphere_volume,
    check_positive_definite_symmetric, rref_matrix
)
from mesh_transform import (
    rotation_matrix_2d, dilation_matrix_2d,
    transform_mesh, polygon_surface_quality,
    adaptive_refinement_markers, refine_marked_elements
)
from task_workload_model import (
    alnorm_cdf, log_normal_pdf, log_normal_sample,
    log_normal_mean, log_normal_variance,
    generate_task_set
)
from quadrature_integrator import (
    vandermonde_quadrature_weights,
    pyramid_monomial_integral, pyramid_volume,
    composite_quadrature_2d, estimate_quadrature_error
)
from monte_carlo_uq import (
    uniform_in_sphere01_map, ellipse_sample, ellipse_area,
    hypersphere01_monomial_integral,
    hypersphere_monte_carlo_integral,
    hypercube_distance_stats,
    antithetic_variates_integral
)
from performance_surrogate import (
    cheby_nodes, divided_differences, newton_interp_eval,
    least_squares_fit, poly_value,
    PerformanceSurrogate
)
from fem_thermal_solver import (
    build_rectangular_mesh,
    fem2d_poisson_solve,
    extract_gradient_at_nodes
)
from heterogeneous_platform import (
    Processor, HeterogeneousPlatform
)
from scheduler_engine import (
    greedy_partition_load_balance,
    solve_task_mapping_ilp,
    reversi_greedy_move,
    schedule_tasks_greedy,
    local_search_improvement
)


def demo_thermal_fem():
    """
    演示2D FEM热求解。
    精确解: u(x,y) = sin(pi x) sin(pi y) + x
    源项:   f(x,y) = 2 pi^2 sin(pi x) sin(pi y)
    """
    def exact(x, y):
        u = np.sin(np.pi * x) * np.sin(np.pi * y) + x
        dudx = np.pi * np.cos(np.pi * x) * np.sin(np.pi * y) + 1.0
        dudy = np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)
        return u, dudx, dudy

    def source(x, y):
        return 2.0 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y)

    nx, ny = 17, 17
    u, nodes, elems, el2, eh1 = fem2d_poisson_solve(
        nx, ny, source, exact,
        xl=0.0, xr=1.0, yb=0.0, yt=1.0,
        conductivity=1.0
    )

    # 网格质量评估
    quality, qmin, qmean = polygon_surface_quality(nodes, elems)
    # 自适应细化
    grad = extract_gradient_at_nodes(u, nodes, elems)
    markers = adaptive_refinement_markers(nodes, elems, grad, threshold_ratio=0.3)
    new_nodes, new_elems = refine_marked_elements(nodes, elems, markers)

    print("=" * 60)
    print("[1] FEM Thermal Solver")
    print(f"    Mesh: {nx}x{ny} nodes, {elems.shape[1]} elements")
    print(f"    L2 error = {el2:.6e}")
    print(f"    H1 error = {eh1:.6e}")
    print(f"    Mesh quality (min/mean) = {qmin:.4f} / {qmean:.4f}")
    print(f"    Adaptive refinement: {np.sum(markers)} elements marked")
    print(f"    Refined mesh: {new_nodes.shape[1]} nodes, {new_elems.shape[1]} elements")
    return u, nodes, elems


def demo_monte_carlo_uq():
    """
    演示蒙特卡洛不确定性量化。
    """
    rng = np.random.default_rng(42)
    print("=" * 60)
    print("[2] Monte Carlo Uncertainty Quantification")

    # 2D椭圆采样（材料参数不确定性）
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    r = 1.0
    samples_ellipse = ellipse_sample(1000, A, r, rng=rng)
    area_est = ellipse_area(A, r)
    print(f"    Ellipse area (analytic) = {area_est:.6f}")

    # 超球体表面积
    for dim in [2, 3, 4, 5]:
        area = hypersphere_surface_area(dim)
        vol = hypersphere_volume(dim)
        print(f"    Hypersphere S_{dim} = {area:.6f}, V_{dim} = {vol:.6f}")

    # 超立方体距离统计（任务特征相似度）
    mu_d, var_d = hypercube_distance_stats(5, 5000, rng=rng)
    print(f"    Hypercube distance stats (dim=5): mu={mu_d:.4f}, var={var_d:.6f}")

    # 超球面积分
    def ones_func(x):
        return 1.0
    val, err = hypersphere_monte_carlo_integral(3, 2000, ones_func, rng=rng)
    print(f"    Sphere integral (dim=3, N=2000): {val:.4f} ± {err:.4f}")

    # 对偶变量方差缩减
    def test_func(x):
        return np.sum(x ** 2)
    val_a, err_a = antithetic_variates_integral(3, 1000, test_func, rng=rng)
    print(f"    Antithetic variates (dim=3): {val_a:.4f} ± {err_a:.4f}")


def demo_surrogate_model():
    """
    演示性能代理模型。
    """
    print("=" * 60)
    print("[3] Performance Surrogate Models")

    def true_perf(x):
        # 模拟执行时间随计算强度的非线性变化
        return 1.0 + 0.5 * np.sin(3.0 * x) + 0.3 * x ** 2

    # Chebyshev代理
    surr_cheb = PerformanceSurrogate(model_type='chebyshev')
    surr_cheb.train((0.0, 1.0), true_perf, n_nodes=12)
    print(f"    Chebyshev surrogate maxerr = {surr_cheb.maxerr:.6e}")

    # 最小二乘代理
    surr_lsq = PerformanceSurrogate(model_type='least_squares')
    surr_lsq.train((0.0, 1.0), true_perf, n_nodes=20, m_poly=8)
    print(f"    Least-squares surrogate residual = {surr_lsq.residual:.6e}")

    # 预测对比
    x_test = np.array([0.1, 0.33, 0.5, 0.77, 0.9])
    y_true = true_perf(x_test)
    y_cheb = surr_cheb.predict(x_test)
    y_lsq = surr_lsq.predict(x_test)
    print(f"    Test predictions:")
    for i in range(len(x_test)):
        print(f"      x={x_test[i]:.2f}: true={y_true[i]:.4f}, "
              f"cheb={y_cheb[i]:.4f}, lsq={y_lsq[i]:.4f}")
    return surr_cheb, surr_lsq


def demo_task_workload():
    """
    演示任务负载建模。
    """
    print("=" * 60)
    print("[4] Task Workload Stochastic Modeling")

    tasks = generate_task_set(n_tasks=20, seed=196)
    rng = np.random.default_rng(196)

    # 对数正态采样
    mu, sigma = 2.0, 0.5
    samples = [log_normal_sample(mu, sigma, rng=rng) for _ in range(1000)]
    print(f"    LogNormal({mu},{sigma}) samples: mean={np.mean(samples):.2f}, "
          f"std={np.std(samples):.2f}")
    print(f"    Theoretical: mean={log_normal_mean(mu,sigma):.2f}, "
          f"var={log_normal_variance(mu,sigma):.2f}")

    # 可靠性计算
    task = tasks[0]
    rel = task.reliability_probability(allocated_time=task.deadline)
    print(f"    Task 0 reliability at deadline: {rel:.4f}")

    # 正态CDF
    z_vals = [-3.0, -1.0, 0.0, 1.0, 3.0]
    print(f"    Normal CDF (AS 66):")
    for z in z_vals:
        print(f"      Phi({z:+.1f}) = {alnorm_cdf(z, upper=False):.6f}")
    return tasks


def demo_quadrature():
    """
    演示数值积分。
    """
    print("=" * 60)
    print("[5] Numerical Quadrature")

    # Vandermonde求积权重
    n = 5
    x_nodes = np.linspace(0.0, 1.0, n)
    w = vandermonde_quadrature_weights(n, 0.0, 1.0, x_nodes)
    integral_est = np.sum(w * np.exp(x_nodes))
    integral_true = np.exp(1.0) - 1.0
    print(f"    Vandermonde quadrature (N={n}): exp(x) integral = {integral_est:.8f}, "
          f"error = {abs(integral_est - integral_true):.2e}")

    # 金字塔积分
    for exp_z in range(0, 5):
        val = pyramid_monomial_integral([0, 0, exp_z])
        print(f"    Pyramid integral z^{exp_z}: {val:.6f}")

    # 复合2D积分
    func = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)
    val_2d = composite_quadrature_2d(func, 0.0, 1.0, 0.0, 1.0, 8, 8)
    true_2d = 4.0 / (np.pi ** 2)
    print(f"    Composite 2D quadrature: {val_2d:.8f}, true={true_2d:.8f}, "
          f"err={abs(val_2d-true_2d):.2e}")

    err_est = estimate_quadrature_error(func, 0.0, 1.0, 0.0, 1.0, 4, 8)
    print(f"    Richardson error estimate: {err_est:.2e}")


def demo_platform_and_scheduler(tasks, surrogate):
    """
    演示异构平台与任务调度。
    """
    print("=" * 60)
    print("[6] Heterogeneous Platform & Task Scheduling")

    platform = HeterogeneousPlatform(ambient_temp=300.0)
    platform.build_default_platform()

    print(f"    Platform: {len(platform.processors)} processors")
    for p in platform.processors:
        print(f"      {p.proc_type}-{p.proc_id}: {p.peak_gflops:.0f} GFLOPS, "
              f"{p.memory_bw_gb_s:.0f} GB/s")

    # 拓扑变换
    platform.rotate_topology(15.0)
    platform.scale_topology(1.2, 0.9)
    print(f"    Topology transformed (rotated 15°, scaled 1.2x0.9)")

    # 贪心调度
    schedule, metrics = schedule_tasks_greedy(
        tasks, platform, surrogate=surrogate,
        alpha_makespan=0.6, alpha_energy=0.3, alpha_reliability=0.1
    )
    print(f"    Schedule metrics:")
    for k, v in metrics.items():
        print(f"      {k} = {v:.6f}")

    # 局部搜索改进
    schedule = local_search_improvement(tasks, platform, schedule, metrics, max_iter=20)
    print(f"    Local search completed.")

    # RREF分析示例（源自 pariomino）
    M = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]], dtype=float)
    rref_m, det = rref_matrix(M)
    print(f"    RREF demo: det(original) ~ {np.linalg.det(M):.4f}, "
          f"pseudo-det from RREF = {det:.4f}")

    # 博弈抢占策略演示
    board = np.zeros((8, 8), dtype=int)
    move_vals = np.random.rand(8, 8)
    i, j = reversi_greedy_move(board, 1, move_vals)
    print(f"    Reversi greedy move selected: ({i},{j}) with value={move_vals[i,j]:.4f}")

    return schedule, platform


def main():
    print("=" * 60)
    print("Heterogeneous HPC Task Scheduling for Thermal-Electrical")
    print("Coupled Simulation — Integrated Scientific Computing System")
    print("=" * 60)
    t0 = time.time()

    # 1. FEM热求解
    u, nodes, elems = demo_thermal_fem()

    # 2. 蒙特卡洛UQ
    demo_monte_carlo_uq()

    # 3. 代理模型训练
    surr_cheb, surr_lsq = demo_surrogate_model()

    # 4. 任务负载生成
    tasks = demo_task_workload()

    # 5. 数值积分
    demo_quadrature()

    # 6. 平台与调度
    schedule, platform = demo_platform_and_scheduler(tasks, surr_cheb)

    t1 = time.time()
    print("=" * 60)
    print(f"Total execution time: {t1 - t0:.3f} seconds")
    print("All modules completed successfully.")
    print("=" * 60)


if __name__ == '__main__':
    main()
