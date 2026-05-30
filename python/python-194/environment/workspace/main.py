#!/usr/bin/env python3

import numpy as np
import sys




from sparse_matrix import BandedSPDMatrix, banded_cholesky_solve, residual_norm
from mesh_partition import (
    compute_cvt, compute_subdomain_boundaries, subdomain_overlap_masks,
    extract_interface_nodes, metric_anisotropic, metric_boundary_layer,
    anisotropic_distance
)
from fekete_quadrature import fekete_line_rule, fekete_triangle_nodes_weights, gll_nodes_weights
from stokes_manufactured import evaluate_solution, compute_discrete_residual
from schwarz_solver import (
    additive_schwarz_iteration, partition_domain_1d,
    solve_local_schur_complement, build_local_stokes_matrix,
    diophantine_partition
)
from time_integrator import (
    adaptive_time_stepping, transient_stokes_step,
    velocity_verlet_step, semi_implicit_euler_step
)
from geometry_utils import (
    hypersphere_positive_distance_stats, compute_partition_quality,
    plane_tetrahedron_intersect, polygon_area_3d,
    interface_matching_score, partial_digest_reconstruct
)
from optimization_utils import (
    golden_section_search, backtracking_line_search,
    optimize_subdomain_overlap, power_iteration_estimate
)
from performance_bench import (
    linpack_benchmark_dense, banded_benchmark,
    subdomain_solver_benchmark, parallel_efficiency_estimate,
    report_benchmark
)


def demo_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_partitioning():
    demo_section("模块 1: 各向异性 CVT 域分解与负载均衡")

    np.random.seed(42)
    n_subdomains = 4
    domain = (0.0, 1.0, 0.0, 1.0)


    generators = compute_cvt(
        n_subdomains=n_subdomains,
        n_iterations=20,
        n_samples=8000,
        metric_func=lambda x: metric_anisotropic(x, alpha=8.0),
        domain=domain,
        tol=1e-4
    )
    print(f"  生成 {n_subdomains} 个子区域中心:")
    for i, g in enumerate(generators):
        print(f"    Subdomain {i+1}: ({g[0]:.4f}, {g[1]:.4f})")


    element_counts = diophantine_partition(100, n_subdomains)
    print(f"  Diophantine 单元分配: {element_counts}, 总和 = {sum(element_counts)}")


    mean_d, var_d = hypersphere_positive_distance_stats(m=5, n_samples=2000)
    print(f"  超球面距离统计 (m=5): 均值={mean_d:.4f}, 方差={var_d:.4f}")


    volumes = np.array(element_counts, dtype=float)
    Q = compute_partition_quality(volumes)
    print(f"  负载均衡质量指标 Q = {Q:.4f} (1.0 为完美均衡)")

    return generators


def run_fekete_quadrature():
    demo_section("模块 2: 谱元 Fekete (GLL) 高阶求积")

    print("  一维 GLL 节点与权重 (参考区间 [0,1]):")
    for p in [2, 4, 6]:
        x, w = fekete_line_rule(0.0, 1.0, p)
        sum_w = np.sum(w)

        exact_moment = 1.0 / (p + 1)
        approx_moment = np.sum(w * x ** p)
        print(f"    p={p}: nodes={x.shape[0]}, sum(w)={sum_w:.6f}, "
              f"int(x^{p}) error={abs(approx_moment - exact_moment):.2e}")

    print("  二维三角形 Fekete 近似节点:")
    pts, wts = fekete_triangle_nodes_weights(p=4)
    print(f"    p=4: {pts.shape[0]} 个节点, 权重和={np.sum(wts):.6f}")


def run_stokes_manufactured():
    demo_section("模块 3: Stokes 制造解验证")

    x = np.linspace(0.0, 1.0, 11)
    y = np.linspace(0.0, 1.0, 11)
    X, Y = np.meshgrid(x, y)

    for sol_type in ["polynomial", "trigonometric"]:
        u, v, p, fx, fy, h = evaluate_solution(X, Y, sol_type=sol_type, nu=1.0)
        max_div = float(np.max(np.abs(h)))
        print(f"  解类型: {sol_type:15s} | max|div(u)| = {max_div:.2e}")


    n_test = 50
    x_rand = np.random.rand(n_test)
    y_rand = np.random.rand(n_test)
    u_ex, v_ex, p_ex, fx, fy, h = evaluate_solution(x_rand, y_rand, sol_type="polynomial", nu=1.0)

    u_h = u_ex + 1e-4 * np.random.randn(n_test)
    v_h = v_ex + 1e-4 * np.random.randn(n_test)
    p_h = p_ex + 1e-4 * np.random.randn(n_test)
    res = compute_discrete_residual(u_h, v_h, p_h, x_rand, y_rand, nu=1.0, sol_type="polynomial")
    print(f"  模拟数值解离散残差 (含 1e-4 噪声): {res:.4e}")


def run_schwarz_solver():
    demo_section("模块 4: 重叠 Schwarz 域分解求解器")

    N = 128
    dx = 1.0 / N
    n_subdomains = 4
    overlap = 4


    subdomains = partition_domain_1d(N, n_subdomains, overlap)
    print(f"  全局节点数: {N}, 子区域数: {n_subdomains}, 重叠: {overlap}")
    print(f"  子区域划分: {subdomains}")


    x_grid = np.linspace(0.0, 1.0, N)
    u_ex, v_ex, p_ex, fx, fy, h = evaluate_solution(x_grid, np.zeros_like(x_grid),
                                                      sol_type="polynomial", nu=1.0)
    f_global = fx.copy()
    g_global = h.copy()

    u_sol, p_sol, res_norm = additive_schwarz_iteration(
        global_n=N,
        subdomains=subdomains,
        overlap=overlap,
        f_global=f_global,
        g_global=g_global,
        dx=dx,
        nu=1.0,
        max_iter=80,
        tol=1e-6
    )
    print(f"  Schwarz 迭代收敛残差: {res_norm:.4e}")
    print(f"  解的 L2 范数: {np.linalg.norm(u_sol):.4f}")


    err = np.linalg.norm(u_sol - u_ex) / max(1.0, np.linalg.norm(u_ex))
    print(f"  相对于制造解的 L2 误差: {err:.4e}")

    return u_sol, p_sol


def run_time_integration():
    demo_section("模块 5: 自适应时间步进与瞬态 Stokes")

    N = 64
    dx = 1.0 / N
    nu = 1.0
    x_grid = np.linspace(0.0, 1.0, N)


    A_band = BandedSPDMatrix(N, 1)
    coeff = nu / (dx ** 2)
    for i in range(N):
        A_band.set(i, i, 2.0 * coeff)
        if i > 0:
            A_band.set(i, i - 1, -coeff)


    u0, _, _, _, _, _ = evaluate_solution(x_grid, np.zeros_like(x_grid),
                                           sol_type="polynomial", nu=nu)


    def rhs_func(t, u):
        _, _, _, fx, _, _ = evaluate_solution(x_grid, np.zeros_like(x_grid),
                                              sol_type="polynomial", nu=nu)
        return fx

    t_hist, u_hist, dt_hist = adaptive_time_stepping(
        u0=u0,
        t_span=(0.0, 0.5),
        dt_init=0.01,
        rhs_func=rhs_func,
        A_band=A_band,
        tol=1e-4,
        gamma_max=3.0,
        dt_min=1e-6,
        dt_max=0.05,
        max_steps=2000
    )
    print(f"  时间积分完成: t={t_hist[-1]:.4f}, 总步数={len(t_hist)-1}")
    print(f"  初始 dt={dt_hist[0]:.4e}, 最终 dt={dt_hist[-1]:.4e}")
    print(f"  解在 t=0.5 的 L2 范数: {np.linalg.norm(u_hist[-1]):.4f}")


    def accel_func(u):
        return -A_band.to_dense() @ u + rhs_func(0.0, u)

    u_vv = u0.copy()
    v_vv = np.zeros(N, dtype=float)
    dt_vv = 0.001
    for _ in range(100):
        u_vv, v_vv, _ = velocity_verlet_step(u_vv, v_vv, accel_func, dt_vv)
    print(f"  Velocity-Verlet (100步, dt={dt_vv}): 位置范数={np.linalg.norm(u_vv):.4f}")

















    raise NotImplementedError("Hole 3: run_time_integration 中瞬态 Stokes 演示需要补全")


def run_geometry_optimization():
    demo_section("模块 6: 几何界面提取与优化")


    tetra = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=float)
    plane_pt = np.array([0.2, 0.2, 0.2])
    plane_n = np.array([1.0, 1.0, 1.0])
    poly = plane_tetrahedron_intersect(plane_pt, plane_n, tetra)
    if poly is not None:
        area = polygon_area_3d(poly)
        print(f"  平面-四面体交截面面积: {area:.6f} (期望 ~0.28)")
    else:
        print("  平面-四面体无交")


    nodes_i = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    nodes_j = np.array([[0.0, 1e-8], [0.5, 1e-7], [1.0, 2e-8]])
    score = interface_matching_score(nodes_i, nodes_j, tol=1e-6)
    print(f"  界面节点匹配得分: {score:.4f}")


    def overlap_residual(alpha):

        return (alpha - 0.15) ** 2 + 0.01 * alpha

    alpha_opt, f_opt = golden_section_search(overlap_residual, 0.0, 0.5, tol=1e-5)
    print(f"  最优重叠率 (黄金分割搜索): alpha={alpha_opt:.6f}, 残差={f_opt:.6e}")


    def quad_obj(x):
        return float(x[0] ** 2 + 2.0 * x[1] ** 2)

    def quad_grad(x):
        return np.array([2.0 * x[0], 4.0 * x[1]], dtype=float)

    x0 = np.array([1.0, 1.0], dtype=float)
    p = -quad_grad(x0)
    alpha_ls = backtracking_line_search(quad_obj, quad_grad, x0, p)
    print(f"  回溯线搜索步长: alpha={alpha_ls:.6f}")


    n = 32
    A_test = BandedSPDMatrix(n, 1)
    for i in range(n):
        A_test.set(i, i, 2.0)
        if i > 0:
            A_test.set(i, i - 1, -1.0)

    def matvec(v):
        return A_test.to_dense() @ v

    lam_max = power_iteration_estimate(matvec, n, max_iter=40)
    print(f"  幂迭代最大特征值估计 (n=32 DIF2): {lam_max:.4f} (理论 2+2cos(pi/33)~3.98)")


    true_points = np.array([0.0, 0.2, 0.5, 0.9, 1.0])
    dists = []
    for i in range(len(true_points)):
        for j in range(i + 1, len(true_points)):
            dists.append(abs(true_points[i] - true_points[j]))
    reconstructed = partial_digest_reconstruct(dists, max_coord=1.0)
    if reconstructed is not None:
        print(f"  PDP 重建点集: {reconstructed}")
    else:
        print("  PDP 重建失败")


def run_performance_benchmark():
    demo_section("模块 7: 子区域求解器性能基准")


    t_dense, m_dense, r_dense = linpack_benchmark_dense(n=256)
    print(f"  稠密求解 (n=256): 时间={t_dense:.4e}s, MFLOPS={m_dense:.2f}, 归一化残差={r_dense:.4e}")


    sizes = [64, 128, 256, 512]
    bench_results = subdomain_solver_benchmark(sizes, bandwidth=3, n_runs=3)
    report_benchmark(bench_results)


    for n_sub in [2, 4, 8, 16]:
        eff = parallel_efficiency_estimate(serial_time=1.0, n_subdomains=n_sub, comm_fraction=0.08)
        print(f"  Amdahl 并行效率估计 (p={n_sub:2d}, s=0.08): {eff:.4f}")


def main():
    print("\n" + "#" * 70)
    print("#  高性能计算：域分解并行有限元求解器 (PROJECT_194)")
    print("#" * 70)
    print("\n  科学问题：基于重叠 Schwarz 域分解的高阶谱元离散")
    print("           求解二维不可压缩瞬态 Stokes 方程")
    print("  核心方法：")
    print("    - 各向异性 CVT 域分解 + Diophantine 整数负载平衡")
    print("    - 谱元 Fekete (GLL) 高阶求积")
    print("    - 重叠 Schwarz 预处理迭代")
    print("    - 自适应时间步进 + 爆破检测")
    print("    - 制造解验证收敛性")

    try:
        run_partitioning()
        run_fekete_quadrature()
        run_stokes_manufactured()
        run_schwarz_solver()
        run_time_integration()
        run_geometry_optimization()
        run_performance_benchmark()

        print("\n" + "#" * 70)
        print("#  所有模块执行成功，程序正常结束。")
        print("#" * 70 + "\n")
        return 0

    except Exception as e:
        print(f"\n  [ERROR] 程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
