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

# ================================================================
# 测试用例（56个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: hypersphere_surface_area dim=1 returns 2.0 ----
s1 = hypersphere_surface_area(1)
assert abs(s1 - 2.0) < 1e-12, '[TC01] dim=1 surface area should be 2.0 FAILED'

# ---- TC02: hypersphere_surface_area dim=2 equals 2*pi ----
s2 = hypersphere_surface_area(2)
assert abs(s2 - 2.0 * np.pi) < 1e-12, '[TC02] dim=2 surface area should be 2*pi FAILED'

# ---- TC03: hypersphere_surface_area dim=3 equals 4*pi ----
s3 = hypersphere_surface_area(3)
assert abs(s3 - 4.0 * np.pi) < 1e-12, '[TC03] dim=3 surface area should be 4*pi FAILED'

# ---- TC04: hypersphere_volume dim=2 equals pi ----
v2 = hypersphere_volume(2)
assert abs(v2 - np.pi) < 1e-12, '[TC04] dim=2 volume should be pi FAILED'

# ---- TC05: hypersphere_volume consistency V_m = S_m / m ----
for d in [2, 3, 4, 5]:
    sd = hypersphere_surface_area(d)
    vd = hypersphere_volume(d)
    assert abs(vd - sd / d) < 1e-12, f'[TC05] V_{d} = S_{d}/{d} FAILED'

# ---- TC06: check_positive_definite_symmetric on identity matrix ----
I2 = np.eye(2)
assert check_positive_definite_symmetric(I2), '[TC06] Identity should be SPD FAILED'

# ---- TC07: check_positive_definite_symmetric rejects non-symmetric matrix ----
ns = np.array([[1, 2], [3, 4]], dtype=float)
assert not check_positive_definite_symmetric(ns), '[TC07] Non-symmetric should be rejected FAILED'

# ---- TC08: rotation_matrix_2d orthogonality (R^T R = I) ----
R = rotation_matrix_2d(np.pi / 3.0)
assert np.allclose(R @ R.T, np.eye(2), atol=1e-14), '[TC08] Rotation matrix should be orthogonal FAILED'

# ---- TC09: rotation_matrix_2d determinant equals 1 ----
assert abs(np.linalg.det(R) - 1.0) < 1e-14, '[TC09] Rotation matrix det should be 1 FAILED'

# ---- TC10: dilation_matrix_2d returns correct scaling ----
D = dilation_matrix_2d(2.0, 3.0)
assert D[0, 0] == 2.0 and D[1, 1] == 3.0 and D[0, 1] == 0.0 and D[1, 0] == 0.0, '[TC10] Dilation matrix FAILED'

# ---- TC11: affine_transform_2d identity ----
import numpy as np
from mesh_transform import affine_transform_2d
pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
transformed = affine_transform_2d(pts)
assert np.allclose(transformed, pts, atol=1e-14), '[TC11] Identity affine transform FAILED'

# ---- TC12: alnorm_cdf at 0 returns 0.5 ----
assert abs(alnorm_cdf(0.0) - 0.5) < 1e-14, '[TC12] Phi(0) should be 0.5 FAILED'

# ---- TC13: alnorm_cdf symmetry Phi(-x) = 1 - Phi(x) ----
for x in [0.5, 1.0, 2.0, 3.0]:
    assert abs(alnorm_cdf(-x) - (1.0 - alnorm_cdf(x))) < 1e-14, f'[TC13] Phi(-{x}) symmetry FAILED'

# ---- TC14: alnorm_cdf upper tail ----
p_lower = alnorm_cdf(1.0, upper=False)
p_upper = alnorm_cdf(1.0, upper=True)
assert abs(p_lower + p_upper - 1.0) < 1e-14, '[TC14] Upper + lower tails should sum to 1 FAILED'

# ---- TC15: log_normal_pdf positive at x>0 ----
pdf_val = log_normal_pdf(np.array([1.0]), 0.0, 1.0)
assert pdf_val[0] > 0, '[TC15] LogNormal PDF at x=1 should be positive FAILED'

# ---- TC16: log_normal_pdf zero at x<=0 ----
pdf_zero = log_normal_pdf(np.array([0.0]), 0.0, 1.0)
assert pdf_zero[0] == 0.0, '[TC16] LogNormal PDF at x=0 should be 0 FAILED'

# ---- TC17: log_normal_mean formula ----
mu, sigma = 2.0, 0.5
expected_mean = np.exp(mu + 0.5 * sigma ** 2)
assert abs(log_normal_mean(mu, sigma) - expected_mean) < 1e-12, '[TC17] LogNormal mean formula FAILED'

# ---- TC18: log_normal_variance formula ----
expected_var = (np.exp(sigma ** 2) - 1.0) * np.exp(2.0 * mu + sigma ** 2)
assert abs(log_normal_variance(mu, sigma) - expected_var) < 1e-12, '[TC18] LogNormal variance formula FAILED'

# ---- TC19: generate_task_set returns correct count with seed reproducibility ----
import numpy as np
tasks_a = generate_task_set(n_tasks=10, seed=42)
tasks_b = generate_task_set(n_tasks=10, seed=42)
assert len(tasks_a) == 10, '[TC19] Task set should have 10 tasks FAILED'
assert len(tasks_b) == 10, '[TC19] Task set B should have 10 tasks FAILED'
for i in range(10):
    assert abs(tasks_a[i].base_flops - tasks_b[i].base_flops) < 1e-9, f'[TC19] Task {i} reproducibility FAILED'

# ---- TC20: binomial_coeff C(5,2)=10 ----
from utils import binomial_coeff
assert binomial_coeff(5, 2) == 10, '[TC20] C(5,2) should be 10 FAILED'

# ---- TC21: binomial_coeff C(n,0)=1, C(n,n)=1 ----
assert binomial_coeff(10, 0) == 1, '[TC21] C(10,0) should be 1 FAILED'
assert binomial_coeff(10, 10) == 1, '[TC21] C(10,10) should be 1 FAILED'

# ---- TC22: vandermonde_quadrature_weights exact for polynomial degree < n ----
import numpy as np
n_q = 4
x_q = np.linspace(0.0, 1.0, n_q)
w_q = vandermonde_quadrature_weights(n_q, 0.0, 1.0, x_q)
# Test exact integration of x^k for k=0..n-1
for k in range(n_q):
    est = np.sum(w_q * (x_q ** k))
    true = 1.0 / (k + 1)
    assert abs(est - true) < 1e-12, f'[TC22] Vandermonde quadrature for x^{k} FAILED'

# ---- TC23: pyramid_monomial_integral zero for odd exponents ----
val_odd = pyramid_monomial_integral([1, 0, 0])
assert val_odd == 0.0, '[TC23] Odd exponent x integral should be 0 FAILED'

# ---- TC24: pyramid_monomial_integral for z^0 gives volume ----
vol_pyr = pyramid_monomial_integral([0, 0, 0])
assert abs(vol_pyr - pyramid_volume()) < 1e-12, '[TC24] Integral of 1 over pyramid should be volume FAILED'

# ---- TC25: pyramid_volume equals 4/3 ----
assert abs(pyramid_volume() - 4.0 / 3.0) < 1e-12, '[TC25] Pyramid volume should be 4/3 FAILED'

# ---- TC26: composite_quadrature_2d sin product integral ----
import numpy as np
f_sin = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)
val_2d = composite_quadrature_2d(f_sin, 0.0, 1.0, 0.0, 1.0, 8, 8)
true_2d_ref = 4.0 / (np.pi ** 2)
assert abs(val_2d - true_2d_ref) < 1e-5, '[TC26] Composite 2D sin integral FAILED'

# ---- TC27: estimate_quadrature_error returns finite non-negative ----
err_est = estimate_quadrature_error(f_sin, 0.0, 1.0, 0.0, 1.0, 4, 8)
assert err_est >= 0.0, '[TC27] Quadrature error estimate should be non-negative FAILED'
assert np.isfinite(err_est), '[TC27] Quadrature error estimate should be finite FAILED'

# ---- TC28: ellipse_area known case ----
A_test = np.array([[2.0, 0.0], [0.0, 2.0]])
r_test = 2.0
area_test = ellipse_area(A_test, r_test)
expected_area = np.pi * r_test * r_test / np.sqrt(np.linalg.det(A_test))
assert abs(area_test - expected_area) < 1e-12, '[TC28] Ellipse area FAILED'

# ---- TC29: ellipse_sample points lie within ellipse ----
import numpy as np
np.random.seed(42)
A_ell = np.array([[4.0, 1.0], [1.0, 3.0]])
samples_ell = ellipse_sample(500, A_ell, 1.5, rng=np.random.default_rng(42))
for i in range(500):
    xv = samples_ell[:, i]
    qf = xv @ A_ell @ xv
    assert qf <= 1.5 ** 2 + 1e-10, f'[TC29] Point {i} outside ellipse FAILED'

# ---- TC30: hypersphere01_monomial_integral zero for odd exponent ----
val_odd_sph = hypersphere01_monomial_integral(3, [1, 0, 0])
assert val_odd_sph == 0.0, '[TC30] Odd monomial on sphere should integrate to 0 FAILED'

# ---- TC31: hypersphere_monte_carlo_integral of constant 1 gives area (reproducible) ----
import numpy as np
np.random.seed(42)
def ones_fn(x):
    return 1.0
val_sph, err_sph = hypersphere_monte_carlo_integral(3, 3000, ones_fn, rng=np.random.default_rng(42))
area_s3 = hypersphere_surface_area(3)
assert abs(val_sph - area_s3) < 0.15, f'[TC31] Sphere MC integral of 1 should be ~{area_s3:.4f} FAILED'

# ---- TC32: hypercube_distance_stats reproducibility ----
import numpy as np
np.random.seed(42)
mu1, var1 = hypercube_distance_stats(4, 2000, rng=np.random.default_rng(42))
np.random.seed(42)
mu2, var2 = hypercube_distance_stats(4, 2000, rng=np.random.default_rng(42))
assert abs(mu1 - mu2) < 1e-14, '[TC32] Hypercube distance stats mu reproducibility FAILED'
assert abs(var1 - var2) < 1e-14, '[TC32] Hypercube distance stats var reproducibility FAILED'

# ---- TC33: cheby_nodes lie within [a,b] ----
xn = cheby_nodes(-1.0, 1.0, 50)
assert np.all(xn >= -1.0) and np.all(xn <= 1.0), '[TC33] Cheby nodes should be within [-1,1] FAILED'

# ---- TC34: cheby_nodes endpoints for n=2 ----
xn2 = cheby_nodes(-1.0, 1.0, 2)
assert abs(max(xn2) - min(xn2)) > 0.0, '[TC34] Cheby nodes for n=2 should be distinct FAILED'

# ---- TC35: divided_differences + newton_interp_eval reproduce data points ----
import numpy as np
xd_t = np.array([0.0, 0.5, 1.0])
yd_t = np.array([1.0, 2.0, 5.0])
dd_t = divided_differences(xd_t, yd_t)
y_back = newton_interp_eval(xd_t, dd_t, xd_t)
assert np.allclose(y_back, yd_t, atol=1e-12), '[TC35] Newton interpolation should reproduce data FAILED'

# ---- TC36: PerformanceSurrogate chebyshev train and predict ----
import numpy as np
np.random.seed(42)
def f_test(x):
    return np.sin(2.0 * np.pi * x)
surr = PerformanceSurrogate(model_type='chebyshev')
surr.train((0.0, 1.0), f_test, n_nodes=15)
x_pred = np.array([0.25, 0.75])
y_pred = surr.predict(x_pred)
for i in range(len(x_pred)):
    assert np.isfinite(y_pred[i]), f'[TC36] Chebyshev surrogate prediction at x={x_pred[i]} FAILED'

# ---- TC37: PerformanceSurrogate least_squares train and predict ----
surr_lsq = PerformanceSurrogate(model_type='least_squares')
surr_lsq.train((0.0, 1.0), f_test, n_nodes=20, m_poly=8)
y_lsq_pred = surr_lsq.predict(x_pred)
for i in range(len(x_pred)):
    assert np.isfinite(y_lsq_pred[i]), f'[TC37] LSQ surrogate prediction at x={x_pred[i]} FAILED'

# ---- TC38: build_rectangular_mesh correct node count ----
nodes_m, elems_m = build_rectangular_mesh(5, 7)
assert nodes_m.shape == (2, 35), f'[TC38] Mesh should have 5*7=35 nodes, got {nodes_m.shape} FAILED'
assert elems_m.shape[1] == 2 * 4 * 6, f'[TC38] Mesh should have 2*4*6=48 elements FAILED'

# ---- TC39: polygon_surface_quality equilateral triangle quality ~1 ----
import numpy as np
eq_nodes = np.array([[0.0, 1.0, 0.5], [0.0, 0.0, np.sqrt(3.0)/2.0]])
eq_elems = np.array([[1, 2, 3]]).T
q, qmin, qmean = polygon_surface_quality(eq_nodes, eq_elems)
assert abs(q[0] - 1.0) < 1e-10, f'[TC39] Equilateral triangle quality should be 1, got {q[0]} FAILED'

# ---- TC40: Processor effective_performance decreases with utilization ----
proc_test = Processor(0, 'CPU', peak_gflops=100.0, memory_bw_gb_s=10.0,
                      power_idle_w=10.0, power_peak_w=100.0,
                      thermal_resistance_k_w=0.5, position_xy=[0.0, 0.0])
proc_test.utilization = 0.0
perf0 = proc_test.effective_performance()
proc_test.utilization = 1.0
perf1 = proc_test.effective_performance()
assert perf1 < perf0, '[TC40] Effective performance should decrease with utilization FAILED'

# ---- TC41: HeterogeneousPlatform build_default_platform has 4 processors ----
platform_test = HeterogeneousPlatform(ambient_temp=300.0)
platform_test.build_default_platform()
assert len(platform_test.processors) == 4, f'[TC41] Default platform should have 4 processors FAILED'
types = [p.proc_type for p in platform_test.processors]
assert types.count('CPU') == 2, '[TC41] Should have 2 CPUs FAILED'
assert types.count('GPU') == 1, '[TC41] Should have 1 GPU FAILED'
assert types.count('FPGA') == 1, '[TC41] Should have 1 FPGA FAILED'

# ---- TC42: HeterogeneousPlatform comm_matrix symmetric ----
cm = platform_test.comm_latency_matrix
assert np.allclose(cm, cm.T, atol=1e-14), '[TC42] Communication matrix should be symmetric FAILED'

# ---- TC43: greedy_partition_load_balance balanced output ----
weights = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
assign_p, loads = greedy_partition_load_balance(weights, 2)
assert len(assign_p) == 5, '[TC43] Assignment should have 5 entries FAILED'
assert len(loads) == 2, '[TC43] Should have 2 bins FAILED'
assert abs(sum(loads) - sum(weights)) < 1e-12, '[TC43] Total load should be preserved FAILED'

# ---- TC44: reversi_greedy_move corner preference ----
import numpy as np
np.random.seed(42)
board = np.zeros((8, 8), dtype=int)
board[0, 0] = 2  # occupy one corner
move_vals = np.random.rand(8, 8)
i_m, j_m = reversi_greedy_move(board, 1, move_vals)
assert i_m == 0 and j_m == 7, f'[TC44] Should pick corner (0,7), got ({i_m},{j_m}) FAILED'

# ---- TC45: solve_task_mapping_ilp assigns each task exactly once ----
import numpy as np
np.random.seed(42)
cost_mat = np.random.rand(6, 3)
assign_ilp, cost_ilp = solve_task_mapping_ilp(6, 3, cost_mat, max_solutions=5)
assert assign_ilp is not None, '[TC45] ILP should return an assignment FAILED'
assert len(set(assign_ilp)) <= 3, '[TC45] Should use at most 3 processors FAILED'
for i in range(6):
    assert 0 <= assign_ilp[i] < 3, f'[TC45] Task {i} assignment out of range FAILED'

# ---- TC46: rref_matrix on identity returns identity ----
import numpy as np
I3 = np.eye(3, dtype=float)
rref_I, det_I = rref_matrix(I3)
assert np.allclose(rref_I, I3, atol=1e-12), '[TC46] RREF of identity should be identity FAILED'

# ---- TC47: rref_matrix on 2x2 singular matrix produces zero row ----
M_sing = np.array([[1.0, 2.0], [2.0, 4.0]])
rref_sing, det_sing = rref_matrix(M_sing)
assert np.allclose(rref_sing[1, :], 0.0, atol=1e-12) or np.allclose(rref_sing[0, :], 0.0, atol=1e-12), '[TC47] RREF of singular matrix should have a zero row FAILED'

# ---- TC48: fem2d_poisson_solve small mesh returns finite solution ----
import numpy as np
def src_small(x, y):
    return 2.0 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y)
def ex_small(x, y):
    u = np.sin(np.pi * x) * np.sin(np.pi * y) + x
    dudx = np.pi * np.cos(np.pi * x) * np.sin(np.pi * y) + 1.0
    dudy = np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)
    return u, dudx, dudy
u_fem, nodes_fem, elems_fem, el2_fem, eh1_fem = fem2d_poisson_solve(
    5, 5, src_small, ex_small, xl=0.0, xr=1.0, yb=0.0, yt=1.0, conductivity=1.0
)
assert np.all(np.isfinite(u_fem)), '[TC48] FEM solution should be finite FAILED'
assert el2_fem >= 0.0, '[TC48] L2 error should be non-negative FAILED'
assert eh1_fem >= 0.0, '[TC48] H1 error should be non-negative FAILED'
assert u_fem.shape[0] == 25, f'[TC48] Solution should have 25 nodes FAILED'

# ---- TC49: extract_gradient_at_nodes returns finite gradients ----
grad = extract_gradient_at_nodes(u_fem, nodes_fem, elems_fem)
assert grad.shape == (2, 25), f'[TC49] Gradient shape should be (2, 25), got {grad.shape} FAILED'
assert np.all(np.isfinite(grad)), '[TC49] Gradient should be finite FAILED'

# ---- TC50: adaptive_refinement_markers returns correct shape ----
markers = adaptive_refinement_markers(nodes_fem, elems_fem, grad, threshold_ratio=0.3)
assert markers.shape[0] == elems_fem.shape[1], '[TC50] Marker count should match element count FAILED'
assert np.sum(markers) > 0, '[TC50] At least one element should be marked FAILED'

# ---- TC51: refine_marked_elements increases node count ----
new_nodes, new_elems = refine_marked_elements(nodes_fem, elems_fem, markers)
assert new_nodes.shape[1] > nodes_fem.shape[1], '[TC51] Refinement should increase node count FAILED'
assert new_elems.shape[1] > elems_fem.shape[1], '[TC51] Refinement should increase element count FAILED'

# ---- TC52: schedule_tasks_greedy produces valid schedule (integration) ----
import numpy as np
np.random.seed(196)
tasks_sched = generate_task_set(n_tasks=6, seed=196)
plat_sched = HeterogeneousPlatform(ambient_temp=300.0)
plat_sched.build_default_platform()
surr_sched = PerformanceSurrogate(model_type='chebyshev')
def dummy_perf(x):
    return 1.0 + 0.2 * np.sin(3.0 * x)
surr_sched.train((0.0, 1.0), dummy_perf, n_nodes=8)
sched, metrics_sched = schedule_tasks_greedy(
    tasks_sched, plat_sched, surrogate=surr_sched,
    alpha_makespan=0.6, alpha_energy=0.3, alpha_reliability=0.1
)
assert 'makespan' in metrics_sched, '[TC52] Metrics should contain makespan FAILED'
assert metrics_sched['makespan'] > 0, '[TC52] Makespan should be positive FAILED'
total_assigned = sum(len(v) for v in sched.values())
assert total_assigned == len(tasks_sched), f'[TC52] All {len(tasks_sched)} tasks should be scheduled FAILED'

# ---- TC53: antithetic_variates_integral bounded result (reproducible) ----
import numpy as np
np.random.seed(42)
def sq_sum(x):
    return np.sum(x ** 2)
res_av, err_av = antithetic_variates_integral(3, 500, sq_sum, rng=np.random.default_rng(42))
assert np.isfinite(res_av), '[TC53] Antithetic variates result should be finite FAILED'
assert err_av >= 0, '[TC53] Antithetic variates error should be non-negative FAILED'

# ---- TC54: uniform_in_sphere01_map all points within unit sphere ----
import numpy as np
np.random.seed(42)
pts_sph = uniform_in_sphere01_map(3, 100, rng=np.random.default_rng(42))
norms = np.linalg.norm(pts_sph, axis=0)
assert np.all(norms <= 1.0 + 1e-12), '[TC54] All points should be within unit sphere FAILED'

# ---- TC55: least_squares_fit residual non-negative ----
import numpy as np
np.random.seed(42)
xd_fit = np.linspace(0.0, 1.0, 10)
yd_fit = 2.0 * xd_fit + 1.0 + 0.01 * np.random.randn(10)
c_fit, res_fit = least_squares_fit(xd_fit, yd_fit, 3)
assert res_fit >= 0.0, '[TC55] LSQ residual should be non-negative FAILED'
assert c_fit.shape == (3,), f'[TC55] Coefficient shape should be (3,), got {c_fit.shape} FAILED'

# ---- TC56: poly_value evaluates correctly at endpoints ----
y0 = poly_value(c_fit, np.array([xd_fit[0]]))
yN = poly_value(c_fit, np.array([xd_fit[-1]]))
assert np.isfinite(y0[0]) and np.isfinite(yN[0]), '[TC56] Poly value should be finite at endpoints FAILED'

print('\n全部 56 个测试通过!\n')
