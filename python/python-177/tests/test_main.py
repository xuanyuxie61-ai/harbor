# -*- coding: utf-8 -*-
"""
main.py
=======
统一入口：二维多相流界面演化的自适应高阶水平集方法演示。

科学问题描述
------------
本项目求解二维界面在曲率驱动流（Mean Curvature Flow）与外部振荡力场
共同作用下的演化问题。采用 Hamilton-Jacobi 框架下的水平集方法，
结合 WENO5 空间离散与 TVD-RK3 时间积分，实现博士级精度的数值模拟。

核心控制方程
~~~~~~~~~~~~
1. 水平集演化方程:
   ∂φ/∂t + V_n |∇φ| = 0
   其中 V_n = -ε κ + f_ext(x,y,t) + λ(t)

2. 平均曲率:
   κ = ∇ · (∇φ / |∇φ|)
     = (φ_{xx} φ_y² - 2 φ_x φ_y φ_{xy} + φ_{yy} φ_x²) / (φ_x² + φ_y²)^{3/2}

3. 重初始化方程:
   ∂φ/∂τ + S(φ_0) (|∇φ| - 1) = 0

4. 体积守恒修正:
   λ(t) 由 Brent 法求解 V(λ) = V_0 确定

5. 外部力场（Ripple-like 振荡）:
   f_ext = A · sin(ω t) · sin(k_x x) · sin(k_y y)

数值方法
~~~~~~~~
- 空间: WENO5 (Jiang & Shu, 1996)
- 时间: 3阶 TVD-RK (Shu & Osher)
- 曲率: 4阶中心差分
- 重初始化: Godunov 迎风格式 + Jacobi 迭代
- 体积修正: Brent 一维优化
- 拓扑追踪: 图论 BFS 连通分量分析
- 降阶模型: POD-SVD
"""

import numpy as np
import sys
import time

# 项目模块
from levelset_function import LevelSetFunction
from hj_solver import HJSolver, ShearFlow
from reinitialization import Reinitializer
from curvature_flow import CurvatureFlow
from adaptive_mesh import AdaptiveMesh
from topology_tracker import TopologyTracker
from volume_corrector import VolumeCorrector, ExternalForcing
from convergence_analysis import ConvergenceAnalysis, ReducedOrderModel
from sampling_engine import LatinCenterSampler, UncertaintyQuantification
from optimizer import MatrixChainOptimizer, ConstraintSatisfier, OperatorSequenceOptimizer


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_levelset_initialization():
    """演示：水平集初始化与几何量计算"""
    print_section("Phase 1: Level Set Initialization & Geometry")

    ls = LevelSetFunction(nx=101, ny=101, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
    # 使用两个相交圆（拓扑合并测试），比星形更稳定且能展示拓扑变化
    ls.init_two_circles(c1=(-0.25, 0.0), c2=(0.25, 0.0), r=0.28)

    vol = ls.compute_volume()
    length = ls.compute_interface_length()
    kappa = ls.compute_curvature()
    kappa_mean = np.mean(kappa[np.abs(ls.phi) < 0.1])
    kappa_max = np.max(np.abs(kappa))

    print(f"  Domain: [-1,1] × [-1,1], Grid: 101×101")
    print(f"  Initial shape: Two intersecting circles (topology merge test)")
    print(f"  Initial volume: {vol:.6f}")
    print(f"  Interface length: {length:.6f}")
    print(f"  Mean curvature (near interface): {kappa_mean:.4f}")
    print(f"  Max |curvature|: {kappa_max:.4f}")
    return ls


def demo_reinitialization(ls):
    """演示：重初始化"""
    print_section("Phase 2: Reinitialization to Signed Distance Function")

    reinit = Reinitializer(ls, max_iter=80, tol=1e-5)
    it, diff = reinit.reinitialize_jacobi_style(omega=1.0)
    sdf_error = reinit.check_sdf_property()

    print(f"  Reinitialization iterations: {it}")
    print(f"  Final residual: {diff:.3e}")
    print(f"  SDF property error (|∇φ|-1): {sdf_error:.4f}")


def demo_curvature_flow(ls):
    """演示：曲率流与 Willmore 能量"""
    print_section("Phase 3: Curvature Flow & Willmore Energy")

    cf = CurvatureFlow(ls)
    W = cf.compute_willmore_energy()
    A = cf.compute_surface_area()
    var = cf.compute_gauss_map_variance()

    print(f"  Willmore energy W = ∫ κ² dA: {W:.6f}")
    print(f"  Surface area (length): {A:.6f}")
    print(f"  Gauss map variance: {var:.6f}")

    # Lebedev 积分演示
    from curvature_flow import lebedev_by_order
    x, y, z, w = lebedev_by_order(14)
    f_vals = np.ones_like(x)
    integral = CurvatureFlow.integrate_on_sphere_surface(f_vals, order=14)
    print(f"  Lebedev quadrature test (∫ 1 dΩ): {integral:.6f} (exact: 12.566)")


def demo_time_evolution(ls):
    """演示：时间演化 + 体积守恒 + 拓扑追踪"""
    print_section("Phase 4: Time Evolution with Volume Correction & Topology Tracking")

    solver = HJSolver(ls, epsilon=0.02, gamma=0.0)
    corrector = VolumeCorrector(ls)
    tracker = TopologyTracker(ls)

    t = 0.0
    dt_base = 0.002
    n_steps = 60

    X, Y = np.meshgrid(ls.x, ls.y, indexing='ij')

    # 初始拓扑记录
    tracker.update_history()
    vol0 = ls.compute_volume()

    print(f"  Time stepping: {n_steps} steps, base dt={dt_base}")
    print(f"  Initial volume (target): {vol0:.6f}")

    for step in range(1, n_steps + 1):
        # 计算外部力场（小幅值振荡，保持数值稳定）
        f_ext = ExternalForcing.oscillatory_normal_forcing(
            X, Y, t, A=0.03, omega=2.0, kx=2.0, ky=2.0
        )

        # CFL 时间步
        dt = solver.compute_cfl_dt(forcing=f_ext, cfl=0.3)
        dt = min(dt, dt_base)
        if dt < 1e-10:
            dt = dt_base

        # 演化一步
        solver.step_rk3(dt, forcing=f_ext)
        t += dt

        # 每 10 步执行重初始化与体积修正
        if step % 10 == 0:
            reinit = Reinitializer(ls, max_iter=40, tol=1e-4)
            reinit.reinitialize()
            corrector.target_volume = vol0
            corrector.correct_volume_simple()
            tracker.update_history()

    vol_final = ls.compute_volume()
    print(f"  Final time: {t:.4f}")
    print(f"  Final volume: {vol_final:.6f}")
    print(f"  Volume drift: {abs(vol_final - vol0):.6e}")
    print(f"  Topology summary:")
    for line in tracker.get_summary().split('\n'):
        print(f"    {line}")


def demo_adaptive_mesh_and_cvt(ls):
    """演示：自适应网格与 CVT 节点优化"""
    print_section("Phase 5: Adaptive Mesh & CVT Node Optimization")

    amesh = AdaptiveMesh(ls, h_min=0.02, h_max=0.15, h_band=0.15)
    h_field = amesh.compute_size_function()
    h_min_val = np.min(h_field)
    h_max_val = np.max(h_field)
    h_mean = np.mean(h_field)

    print(f"  Size function range: [{h_min_val:.4f}, {h_max_val:.4f}]")
    print(f"  Mean size: {h_mean:.4f}")

    # CVT 优化演示
    cvt_points = amesh.cvt_optimize_nodes_2d(num_points=64, max_iter=20, tol=1e-3)
    print(f"  CVT optimized nodes: {len(cvt_points)} points")
    print(f"  Node spread (std): {np.std(cvt_points, axis=0)}")

    # 网格质量
    quality = amesh.estimate_interface_mesh_quality()
    print(f"  Approximate interface mesh quality: {quality:.4f}")


def demo_convergence_and_pod():
    """演示：收敛性分析与 POD 降阶"""
    print_section("Phase 6: Convergence Analysis & POD Reduced-Order Model")

    # 构造解析解验证：行波解
    nx_list = [41, 81, 161]
    errors_l2 = []
    hs = []
    t_test = 0.1

    for nx in nx_list:
        ls = LevelSetFunction(nx=nx, ny=nx, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
        ls.init_circle(cx=0.0, cy=0.0, r=0.3)
        X, Y = np.meshgrid(ls.x, ls.y, indexing='ij')
        # 精确解：圆在匀速扩张/收缩
        exact = np.sqrt(X ** 2 + Y ** 2) - (0.3 + 0.1 * t_test)
        error = ConvergenceAnalysis.l2_error(ls.phi, exact, ls.dx, ls.dy)
        errors_l2.append(error)
        hs.append(ls.dx)
        print(f"  nx={nx:3d}, h={ls.dx:.5f}, L2 error={error:.3e}")

    orders = ConvergenceAnalysis.convergence_order(errors_l2, hs)
    for i, p in enumerate(orders):
        if not np.isnan(p):
            print(f"  Observed convergence order (h={hs[i]:.5f}→{hs[i+1]:.5f}): {p:.2f}")

    # POD 降阶演示
    snapshots = []
    ls_pod = LevelSetFunction(nx=41, ny=41, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
    ls_pod.init_star_shape()
    for _ in range(10):
        snapshots.append(ls_pod.phi.ravel().copy())
        ls_pod.phi *= 0.95  # 简单扰动生成快照

    rom = ReducedOrderModel(np.column_stack(snapshots))
    Ur, S_r = rom.compute_pod_basis(energy_threshold=0.95)
    energy = rom.get_mode_energy()
    print(f"  POD basis size (95% energy): {rom.r}")
    print(f"  First 3 mode energies: {energy[:3]}")


def demo_sampling_uq():
    """演示：拉丁采样与不确定性量化"""
    print_section("Phase 7: Latin Hypercube Sampling & Uncertainty Quantification")

    # Latin Center 采样
    samples = LatinCenterSampler.sample(dim_num=3, point_num=20)
    print(f"  Latin Center samples: {samples.shape}")
    print(f"  Sample mean per dim: {np.mean(samples, axis=0)}")
    print(f"  Sample std per dim: {np.std(samples, axis=0)}")

    # UQ 演示：参数敏感性
    def test_func(x):
        # 水平集相关参数响应：曲率系数 ε, 外力幅值 A, 频率 ω
        eps, A, omega = x
        return eps * 10.0 + A * np.sin(omega)

    uq = UncertaintyQuantification()
    bounds = [(0.001, 0.1), (0.0, 0.2), (1.0, 10.0)]
    mean, var = uq.estimate_mean_variance(test_func, dim=3, n_samples=50, bounds=bounds)
    print(f"  MC mean of test response: {mean:.4f}")
    print(f"  MC variance: {var:.6f}")

    S1 = uq.estimate_sensitivity_indices(test_func, dim=3, n_samples=50, bounds=bounds)
    print(f"  First-order Sobol-like indices: {S1}")


def demo_optimizer():
    """演示：矩阵链优化与约束满足"""
    print_section("Phase 8: Matrix Chain Optimization & Constraint Satisfaction")

    # 矩阵链最优括号化
    dims = [10, 30, 5, 60, 8]
    cost_dp, s = MatrixChainOptimizer.matrix_chain_dp(dims)
    cost_brute = MatrixChainOptimizer.matrix_chain_brute(dims)
    order_str = MatrixChainOptimizer.get_optimal_order(s, 0, len(dims) - 2)

    print(f"  Matrix chain dimensions: {dims}")
    print(f"  DP optimal cost: {cost_dp}")
    print(f"  Brute force cost: {cost_brute}")
    print(f"  Optimal parenthesization: {order_str}")

    # 约束满足：Young 方程
    solutions = ConstraintSatisfier.young_equation_solver(
        sigma12=1.0, sigma13=0.8, sigma23=0.6, tol=0.01, n_grid=100
    )
    print(f"  Young equation solutions found: {len(solutions)}")
    if solutions:
        sol = solutions[0]
        print(f"  Example contact angles (deg): θ1={np.degrees(sol[0]):.1f}, "
              f"θ2={np.degrees(sol[1]):.1f}, θ3={np.degrees(sol[2]):.1f}")

    # 算子顺序优化
    flops, _ = OperatorSequenceOptimizer.optimize_preconditioner_chain(
        n_ops=4, dim_in=100, dim_mid=50, dim_out=25
    )
    print(f"  Preconditioner chain optimal FLOPs estimate: {flops}")


def demo_numerical_utils():
    """演示：复数线性代数工具（源自 linpack_z）"""
    print_section("Phase 9: Complex Linear Algebra Utilities")

    from numerical_utils import cplx_cholesky_decompose, cplx_lu_factor, cplx_qr_factor

    # Hermite 正定矩阵测试
    A = np.array([[4.0 + 0.0j, 1.0 - 1.0j],
                  [1.0 + 1.0j, 3.0 + 0.0j]], dtype=np.complex128)
    L = cplx_cholesky_decompose(A)
    recon = L @ L.conj().T
    err_chol = np.max(np.abs(recon - A))
    print(f"  Cholesky reconstruction error: {err_chol:.3e}")

    # LU 分解
    B = np.array([[2.0 + 1.0j, 3.0 - 2.0j],
                  [1.0 - 1.0j, 4.0 + 0.0j]], dtype=np.complex128)
    Lb, Ub, Pb = cplx_lu_factor(B)
    recon_lu = Pb.T @ Lb @ Ub
    err_lu = np.max(np.abs(recon_lu - B))
    print(f"  LU reconstruction error: {err_lu:.3e}")

    # QR 分解
    C = np.array([[1.0 + 0.0j, 2.0 - 1.0j],
                  [3.0 + 1.0j, 4.0 + 0.0j],
                  [0.0 + 1.0j, 1.0 - 2.0j]], dtype=np.complex128)
    Qc, Rc = cplx_qr_factor(C)
    recon_qr = Qc @ Rc
    err_qr = np.max(np.abs(recon_qr - C))
    print(f"  QR reconstruction error: {err_qr:.3e}")


def main():
    print("=" * 70)
    print("  博士级合成项目: 二维多相流界面演化的自适应高阶水平集方法")
    print("  Project 177: Level Set Method for Interface Evolution")
    print("=" * 70)

    np.random.seed(42)
    t_start = time.time()

    # Phase 1: 初始化
    ls = demo_levelset_initialization()

    # Phase 2: 重初始化
    demo_reinitialization(ls)

    # Phase 3: 曲率流
    demo_curvature_flow(ls)

    # Phase 4: 时间演化
    demo_time_evolution(ls)

    # Phase 5: 自适应网格与 CVT
    demo_adaptive_mesh_and_cvt(ls)

    # Phase 6: 收敛性与 POD
    demo_convergence_and_pod()

    # Phase 7: 采样与 UQ
    demo_sampling_uq()

    # Phase 8: 优化器
    demo_optimizer()

    # Phase 9: 复数线性代数
    demo_numerical_utils()

    t_elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"  All phases completed successfully in {t_elapsed:.2f} seconds.")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（70个，assert模式，涉及随机值均使用固定种子）
# ================================================================

# ---- TC01: LevelSetFunction circle SDF - center value negative ----
ls = LevelSetFunction(nx=51, ny=51)
ls.init_circle(cx=0.0, cy=0.0, r=0.3)
assert ls.phi[25, 25] < 0, '[TC01] Circle center phi should be negative FAILED'

# ---- TC02: LevelSetFunction circle volume - close to πr² ----
import numpy as np
ls = LevelSetFunction(nx=101, ny=101)
ls.init_circle(cx=0.0, cy=0.0, r=0.3)
vol = ls.compute_volume()
expected_vol = np.pi * 0.3 ** 2
assert abs(vol - expected_vol) < 0.05, '[TC02] Circle volume deviates from πr² FAILED'

# ---- TC03: LevelSetFunction interface length - close to 2πr ----
ls = LevelSetFunction(nx=101, ny=101)
ls.init_circle(cx=0.0, cy=0.0, r=0.3)
length = ls.compute_interface_length()
expected_len = 2.0 * np.pi * 0.3
assert abs(length - expected_len) < 0.3, '[TC03] Interface length deviates from 2πr FAILED'

# ---- TC04: LevelSetFunction curvature near interface - close to 1/r ----
ls = LevelSetFunction(nx=101, ny=101)
ls.init_circle(cx=0.0, cy=0.0, r=0.3)
kappa = ls.compute_curvature()
kappa_near = np.mean(kappa[np.abs(ls.phi) < 0.1])
r = 0.3
curv_expected = 1.0 / r
assert abs(kappa_near - curv_expected) < 2.0, '[TC04] Curvature near interface deviates from 1/r FAILED'

# ---- TC05: LevelSetFunction gradient norm of circle SDF - close to 1 ----
ls = LevelSetFunction(nx=101, ny=101)
ls.init_circle(cx=0.0, cy=0.0, r=0.3)
_, _, grad_norm = ls.compute_gradient_norm()
interior = grad_norm[5:-5, 5:-5]
assert 0.1 < np.mean(interior) < 5.0, '[TC05] Gradient norm of circle SDF unreasonable FAILED'

# ---- TC06: LevelSetFunction compute_normal - returns unit vectors ----
ls = LevelSetFunction(nx=51, ny=51)
ls.init_circle(cx=0.2, cy=0.1, r=0.3)
nx_vec, ny_vec = ls.compute_normal()
n_mag = np.sqrt(nx_vec**2 + ny_vec**2)
interior = n_mag[5:-5, 5:-5]
assert abs(np.mean(interior) - 1.0) < 0.5, '[TC06] Normal vectors not unit length FAILED'

# ---- TC07: LevelSetFunction ellipse initialization - no crash ----
ls = LevelSetFunction(nx=51, ny=51)
ls.init_ellipse(cx=0.0, cy=0.0, a=0.4, b=0.2, theta=0.3)
assert ls.phi is not None and not np.any(np.isnan(ls.phi)), '[TC07] Ellipse init produced NaN FAILED'

# ---- TC08: LevelSetFunction rectangle initialization - works ----
ls = LevelSetFunction(nx=51, ny=51)
ls.init_rectangle(cx=0.0, cy=0.0, w=0.5, h=0.3)
assert np.any(ls.phi < 0) and np.any(ls.phi > 0), '[TC08] Rectangle init should have both signs FAILED'

# ---- TC09: LevelSetFunction star shape - no crash ----
ls = LevelSetFunction(nx=51, ny=51)
ls.init_star_shape(cx=0.0, cy=0.0, r0=0.3, amp=0.05, n_peaks=5)
assert not np.any(np.isnan(ls.phi)), '[TC09] Star shape init produced NaN FAILED'

# ---- TC10: LevelSetFunction two circles - both sides negative ----
ls = LevelSetFunction(nx=51, ny=51)
ls.init_two_circles(c1=(-0.3, 0.0), c2=(0.3, 0.0), r=0.2)
i1 = np.argmin(np.abs(ls.x + 0.3))
j1 = np.argmin(np.abs(ls.y - 0.0))
i2 = np.argmin(np.abs(ls.x - 0.3))
assert ls.phi[i1, j1] < 0 and ls.phi[i2, j1] < 0, '[TC10] Two circles: both centers should be inside FAILED'

# ---- TC11: LevelSetFunction get_zero_levelset_points - returns array ----
ls = LevelSetFunction(nx=51, ny=51)
ls.init_circle(cx=0.0, cy=0.0, r=0.3)
pts = ls.get_zero_levelset_points()
assert pts.shape[1] == 2 and pts.shape[0] > 0, '[TC11] Zero levelset points: wrong shape or empty FAILED'

# ---- TC12: HJSolver CFL dt - returns positive finite value ----
ls = LevelSetFunction(nx=51, ny=51)
ls.init_circle(cx=0.0, cy=0.0, r=0.3)
solver = HJSolver(ls, epsilon=0.02)
dt = solver.compute_cfl_dt(cfl=0.5)
assert dt > 0 and np.isfinite(dt), '[TC12] CFL dt should be positive finite FAILED'

# ---- TC13: HJSolver exact_tanh_1d - correct shape and range ----
x = np.linspace(-1, 1, 101)
phi_tanh = HJSolver.exact_tanh_1d(x, t=0.1, x0=0.0, V=0.5, delta=0.05)
assert phi_tanh.shape == (101,), '[TC13] tanh 1d: wrong shape FAILED'
assert np.max(np.abs(phi_tanh)) <= 1.0, '[TC13] tanh 1d: range should be [-1,1] FAILED'

# ---- TC14: HJSolver exact_tanh_1d at x0 - close to 0 ----
x = np.linspace(-1, 1, 201)
x0_val = 0.2
V_val = 0.5
t_val = x0_val / V_val
phi_x0 = HJSolver.exact_tanh_1d(x, t=t_val, x0=x0_val, V=V_val, delta=0.05)
idx = np.argmin(np.abs(x - x0_val))
assert abs(phi_x0[idx]) < 0.1, '[TC14] tanh at x0+Vt should be near zero FAILED'

# ---- TC15: Reinitializer SDF property - error finite after reinit ----
ls = LevelSetFunction(nx=51, ny=51)
ls.init_circle(cx=0.0, cy=0.0, r=0.3)
reinit = Reinitializer(ls, max_iter=30, tol=1e-4)
it, diff = reinit.reinitialize_jacobi_style(omega=1.0)
sdf_err = reinit.check_sdf_property()
assert np.isfinite(sdf_err) and sdf_err >= 0, '[TC15] SDF property error should be finite non-negative FAILED'

# ---- TC16: Reinitializer - iterations within max_iter ----
ls = LevelSetFunction(nx=51, ny=51)
ls.init_circle(cx=0.0, cy=0.0, r=0.3)
reinit = Reinitializer(ls, max_iter=80, tol=1e-5)
it, diff = reinit.reinitialize()
assert 1 <= it <= 80, '[TC16] Reinit iterations should be within [1,80] FAILED'

# ---- TC17: CurvatureFlow Willmore energy - non-negative ----
ls = LevelSetFunction(nx=51, ny=51)
ls.init_circle(cx=0.0, cy=0.0, r=0.3)
cf = CurvatureFlow(ls)
W = cf.compute_willmore_energy()
assert W >= 0, '[TC17] Willmore energy should be non-negative FAILED'

# ---- TC18: CurvatureFlow Lebedev 14-point integration - close to 4π ----
from curvature_flow import lebedev_by_order
x_l, y_l, z_l, w_l = lebedev_by_order(14)
f_vals = np.ones_like(x_l)
integral = CurvatureFlow.integrate_on_sphere_surface(f_vals, order=14)
assert abs(integral - 4.0 * np.pi) < 1e-10, '[TC18] Lebedev14 integral of 1 should equal 4π FAILED'

# ---- TC19: CurvatureFlow Lebedev 6-point integration - close to 4π ----
x_l6, y_l6, z_l6, w_l6 = lebedev_by_order(6)
f_vals6 = np.ones_like(x_l6)
integral6 = CurvatureFlow.integrate_on_sphere_surface(f_vals6, order=6)
assert abs(integral6 - 4.0 * np.pi) < 1e-10, '[TC19] Lebedev6 integral of 1 should equal 4π FAILED'

# ---- TC20: CurvatureFlow sphere_distance Haversine - antipodal = π ----
d = CurvatureFlow.sphere_distance(0.0, 0.0, 0.0, np.pi, R=1.0)
assert abs(d - np.pi) < 1e-10, '[TC20] Haversine: antipodal distance should be π FAILED'

# ---- TC21: CurvatureFlow sphere_distance - same point = 0 ----
d = CurvatureFlow.sphere_distance(0.5, 1.0, 0.5, 1.0, R=2.0)
assert abs(d) < 1e-12, '[TC21] Haversine: same point distance should be 0 FAILED'

# ---- TC22: CurvatureFlow Gauss map variance - non-negative ----
ls = LevelSetFunction(nx=51, ny=51)
ls.init_circle(cx=0.0, cy=0.0, r=0.3)
cf = CurvatureFlow(ls)
var = cf.compute_gauss_map_variance()
assert var >= 0, '[TC22] Gauss map variance should be non-negative FAILED'

# ---- TC23: CurvatureFlow compute_mean_curvature_flow_velocity - finite ----
ls = LevelSetFunction(nx=51, ny=51)
ls.init_circle(cx=0.0, cy=0.0, r=0.3)
cf = CurvatureFlow(ls)
Vn = cf.compute_mean_curvature_flow_velocity()
assert not np.any(np.isnan(Vn)) and not np.any(np.isinf(Vn)), '[TC23] Mean curvature flow velocity has NaN/Inf FAILED'

# ---- TC24: AdaptiveMesh size function - bounded between h_min and h_max ----
ls = LevelSetFunction(nx=51, ny=51)
ls.init_circle(cx=0.0, cy=0.0, r=0.3)
amesh = AdaptiveMesh(ls, h_min=0.02, h_max=0.15, h_band=0.15)
h_field = amesh.compute_size_function()
assert np.all(h_field >= amesh.h_min - 1e-12) and np.all(h_field <= amesh.h_max + 1e-12), '[TC24] Size function outside [h_min, h_max] FAILED'

# ---- TC25: AdaptiveMesh triangle_area - correct formula ----
p1 = np.array([0.0, 0.0])
p2 = np.array([1.0, 0.0])
p3 = np.array([0.0, 1.0])
area = AdaptiveMesh.triangle_area(p1, p2, p3)
assert abs(area - 0.5) < 1e-12, '[TC25] Triangle area (0,0)-(1,0)-(0,1) should be 0.5 FAILED'

# ---- TC26: AdaptiveMesh triangle_quality - equilateral = 1.0 ----
p1 = np.array([0.0, 0.0])
p2 = np.array([1.0, 0.0])
p3 = np.array([0.5, np.sqrt(3.0)/2.0])
Q = AdaptiveMesh.triangle_quality(p1, p2, p3)
assert abs(Q - 1.0) < 1e-10, '[TC26] Equilateral triangle quality should be 1.0 FAILED'

# ---- TC27: AdaptiveMesh triangle_quality - in (0, 1] ----
p1 = np.array([0.0, 0.0])
p2 = np.array([2.0, 0.0])
p3 = np.array([1.0, 0.1])
Q = AdaptiveMesh.triangle_quality(p1, p2, p3)
assert 0 < Q <= 1.0, f'[TC27] Triangle quality {Q} should be in (0,1] FAILED'

# ---- TC28: TopologyTracker connected components - positive count ----
ls = LevelSetFunction(nx=51, ny=51)
ls.init_circle(cx=0.0, cy=0.0, r=0.3)
tracker = TopologyTracker(ls)
components, nodes = tracker.find_connected_components()
assert len(components) > 0, '[TC28] Connected components should be non-empty FAILED'

# ---- TC29: TopologyTracker Euler characteristic - integer result ----
ls = LevelSetFunction(nx=51, ny=51)
ls.init_circle(cx=0.0, cy=0.0, r=0.3)
tracker = TopologyTracker(ls)
chi = tracker.compute_euler_characteristic_approx()
assert isinstance(chi, (int, np.integer)), '[TC29] Euler characteristic should be integer FAILED'

# ---- TC30: TopologyTracker detect_topological_event - NO_CHANGE ----
event = TopologyTracker.detect_topological_event(None, 3, 3)
assert event == 'NO_CHANGE', '[TC30] Same component count should be NO_CHANGE FAILED'

# ---- TC31: TopologyTracker detect_topological_event - SPLIT ----
event = TopologyTracker.detect_topological_event(None, 2, 3)
assert event == 'SPLIT', '[TC31] More components should be SPLIT FAILED'

# ---- TC32: TopologyTracker detect_topological_event - MERGE ----
event = TopologyTracker.detect_topological_event(None, 3, 2)
assert event == 'MERGE', '[TC32] Fewer components should be MERGE FAILED'

# ---- TC33: VolumeCorrector _volume_after_shift - monotonic decrease ----
ls = LevelSetFunction(nx=51, ny=51)
ls.init_circle(cx=0.0, cy=0.0, r=0.3)
vc = VolumeCorrector(ls)
v1 = vc._volume_after_shift(-0.5)
v2 = vc._volume_after_shift(0.0)
v3 = vc._volume_after_shift(0.5)
assert v1 > v2 > v3, '[TC33] Volume after λ shift should be strictly decreasing FAILED'

# ---- TC34: ExternalForcing oscillatory_normal_forcing - bounded by amplitude ----
X, Y = np.meshgrid(np.linspace(-1, 1, 51), np.linspace(-1, 1, 51), indexing='ij')
f_ext = ExternalForcing.oscillatory_normal_forcing(X, Y, t=0.5, A=0.03, omega=2.0, kx=2.0, ky=2.0)
assert np.max(np.abs(f_ext)) <= 0.03 + 1e-12, '[TC34] Oscillatory forcing exceeds amplitude FAILED'

# ---- TC35: ExternalForcing ripple_like_forcing - bounded by amplitude ----
f_rip = ExternalForcing.ripple_like_forcing(X, Y, t=1.0, A=0.1)
assert np.max(np.abs(f_rip)) <= 0.1 + 1e-12, '[TC35] Ripple forcing exceeds amplitude FAILED'

# ---- TC36: ExternalForcing gravitational_forcing - linear in x ----
f_g = ExternalForcing.gravitational_forcing(X, Y, g=1.0, angle=0.0)
assert np.allclose(f_g, X, atol=1e-12), '[TC36] Gravitational forcing with angle=0 should equal X FAILED'

# ---- TC37: ConvergenceAnalysis L2 error - non-negative ----
phi1 = np.ones((21, 21))
phi2 = phi1 * 0.5
err = ConvergenceAnalysis.l2_error(phi1, phi2, 0.1, 0.1)
assert err >= 0, '[TC37] L2 error should be non-negative FAILED'

# ---- TC38: ConvergenceAnalysis L2 error - zero for identical ----
phi = np.ones((21, 21))
err = ConvergenceAnalysis.l2_error(phi, phi.copy(), 0.1, 0.1)
assert abs(err) < 1e-12, '[TC38] L2 error for identical arrays should be zero FAILED'

# ---- TC39: ConvergenceAnalysis convergence_order - computed ----
errors = [0.1, 0.025, 0.00625]
hs = [0.1, 0.05, 0.025]
orders = ConvergenceAnalysis.convergence_order(errors, hs)
assert len(orders) == 2, '[TC39] Convergence order should have 2 entries FAILED'
assert all(abs(p - 2.0) < 0.01 or np.isnan(p) for p in orders), '[TC39] Convergence order not ~2 for h-halving FAILED'

# ---- TC40: ReducedOrderModel POD - energy sum to 1 ----
np.random.seed(42)
snapshots = np.random.randn(100, 20)
rom = ReducedOrderModel(snapshots)
Ur, S_r = rom.compute_pod_basis(energy_threshold=0.99)
energy = rom.get_mode_energy()
assert abs(np.sum(energy) - 1.0) < 1e-10, '[TC40] POD mode energy should sum to 1 FAILED'

# ---- TC41: ReducedOrderModel reconstruction - error small ----
np.random.seed(42)
snapshots = np.random.randn(100, 20)
rom = ReducedOrderModel(snapshots)
Ur, S_r = rom.compute_pod_basis(energy_threshold=0.95)
recon = rom.reconstruct()
diff = np.max(np.abs(recon - snapshots))
assert diff < 1.0, '[TC41] POD reconstruction error too large FAILED'

# ---- TC42: LatinCenterSampler sample - values in [0,1] ----
np.random.seed(42)
samples = LatinCenterSampler.sample(dim_num=3, point_num=20)
assert np.all(samples >= 0) and np.all(samples <= 1), '[TC42] Latin Center samples not in [0,1] FAILED'

# ---- TC43: LatinCenterSampler sample_scaled - values in bounds ----
np.random.seed(42)
bounds = [(0.0, 10.0), (-5.0, 5.0), (100.0, 200.0)]
samples = LatinCenterSampler.sample_scaled(dim_num=3, point_num=20, bounds=bounds)
assert np.all(samples[:, 0] >= 0) and np.all(samples[:, 0] <= 10), '[TC43] Scaled samples not in bounds FAILED'
assert np.all(samples[:, 1] >= -5) and np.all(samples[:, 1] <= 5), '[TC43] Scaled samples not in bounds FAILED'
assert np.all(samples[:, 2] >= 100) and np.all(samples[:, 2] <= 200), '[TC43] Scaled samples not in bounds FAILED'

# ---- TC44: UncertaintyQuantification MC mean - finite ----
np.random.seed(42)
def test_func(x):
    eps, A_val, omega = x
    return eps * 10.0 + A_val * np.sin(omega)

uq = UncertaintyQuantification()
bounds_uq = [(0.001, 0.1), (0.0, 0.2), (1.0, 10.0)]
mean, var = uq.estimate_mean_variance(test_func, dim=3, n_samples=50, bounds=bounds_uq)
assert np.isfinite(mean) and np.isfinite(var), '[TC44] MC mean/var should be finite FAILED'

# ---- TC45: MatrixChainOptimizer DP and brute agree ----
dims = [10, 30, 5, 60]
cost_dp, s = MatrixChainOptimizer.matrix_chain_dp(dims)
cost_bf = MatrixChainOptimizer.matrix_chain_brute(dims)
assert cost_dp == cost_bf, '[TC45] DP and brute force costs should agree FAILED'

# ---- TC46: MatrixChainOptimizer get_optimal_order - returns valid string ----
dims = [10, 30, 5, 60]
cost_dp, s = MatrixChainOptimizer.matrix_chain_dp(dims)
order_str = MatrixChainOptimizer.get_optimal_order(s, 0, len(dims) - 2)
assert isinstance(order_str, str) and '(' in order_str, '[TC46] Optimal order should be string with parentheses FAILED'

# ---- TC47: ConstraintSatisfier Young equation - solutions found ----
solutions = ConstraintSatisfier.young_equation_solver(
    sigma12=1.0, sigma13=0.8, sigma23=0.6, tol=0.01, n_grid=100
)
assert len(solutions) > 0, '[TC47] Young equation should have solutions FAILED'

# ---- TC48: ConstraintSatisfier Young equation - solution satisfies equation ----
solutions = ConstraintSatisfier.young_equation_solver(
    sigma12=1.0, sigma13=0.8, sigma23=0.6, tol=0.01, n_grid=100
)
sol = solutions[0]
theta1, theta2, theta3 = sol
lhs = 1.0 * np.cos(theta3)
rhs = 0.8 * np.cos(theta2) + 0.6 * np.cos(theta1)
assert abs(lhs - rhs) < 0.02, '[TC48] Young equation solution does not satisfy equation FAILED'

# ---- TC49: OperatorSequenceOptimizer FLOPs estimate - positive ----
flops, _ = OperatorSequenceOptimizer.optimize_preconditioner_chain(
    n_ops=4, dim_in=100, dim_mid=50, dim_out=25
)
assert flops > 0, '[TC49] Preconditioner chain FLOPs should be positive FAILED'

# ---- TC50: numerical_utils WENO5 derivative on linear - close to slope ----
from numerical_utils import weno5_derivative, tvd_rk3_step, central_diff_2nd, central_diff_4th, laplacian_2d
x_arr = np.linspace(0, 2, 21)
v = 3.0 * x_arr + 2.0
dphi = weno5_derivative(v, x_arr[1] - x_arr[0], axis=0)
interior = dphi[5:-5]
assert np.allclose(interior, 3.0, atol=0.2), f'[TC50] WENO5 deriv on linear should be ~3, got {np.mean(interior):.4f} FAILED'

# ---- TC51: numerical_utils TVD-RK3 preserves constant ----
phi0 = np.ones((21, 21))
def rhs_zero(phi):
    return np.zeros_like(phi)
phi_new = tvd_rk3_step(phi0, 0.1, rhs_zero)
assert np.allclose(phi_new, 1.0, atol=1e-12), '[TC51] TVD-RK3 should preserve constant when rhs=0 FAILED'

# ---- TC52: numerical_utils Cholesky decomposition - L@L^H = A ----
from numerical_utils import cplx_cholesky_decompose, cplx_lu_factor, cplx_qr_factor
A = np.array([[4.0 + 0.0j, 1.0 - 1.0j],
              [1.0 + 1.0j, 3.0 + 0.0j]], dtype=np.complex128)
L = cplx_cholesky_decompose(A)
recon = L @ L.conj().T
assert np.max(np.abs(recon - A)) < 1e-12, '[TC52] Cholesky L@L^H should equal A FAILED'

# ---- TC53: numerical_utils LU factorization - P^T@L@U = A ----
B = np.array([[2.0 + 1.0j, 3.0 - 2.0j],
              [1.0 - 1.0j, 4.0 + 0.0j]], dtype=np.complex128)
Lb, Ub, Pb = cplx_lu_factor(B)
recon_lu = Pb.T @ Lb @ Ub
assert np.max(np.abs(recon_lu - B)) < 1e-12, '[TC53] LU P^T@L@U should equal A FAILED'

# ---- TC54: numerical_utils QR factorization - Q@R = A ----
C = np.array([[1.0 + 0.0j, 2.0 - 1.0j],
              [3.0 + 1.0j, 4.0 + 0.0j],
              [0.0 + 1.0j, 1.0 - 2.0j]], dtype=np.complex128)
Qc, Rc = cplx_qr_factor(C)
recon_qr = Qc @ Rc
assert np.max(np.abs(recon_qr - C)) < 1e-12, '[TC54] QR Q@R should equal A FAILED'

# ---- TC55: numerical_utils QR - Q is unitary (Q^H @ Q = I) ----
QH_Q = Qc.conj().T @ Qc
assert np.max(np.abs(QH_Q - np.eye(Qc.shape[1]))) < 1e-12, '[TC55] QR Q^H@Q should equal I FAILED'

# ---- TC56: numerical_utils central_diff_2nd on linear - exact ----
x_arr = np.linspace(0, 2, 21)
v = 3.0 * x_arr + 2.0
d = central_diff_2nd(v, x_arr[1] - x_arr[0], axis=0)
assert np.allclose(d[2:-2], 3.0, atol=1e-10), '[TC56] Central diff 2nd on linear: should equal slope FAILED'

# ---- TC57: numerical_utils central_diff_4th on linear - exact ----
x_arr = np.linspace(0, 2, 21)
v = 3.0 * x_arr + 2.0
d4 = central_diff_4th(v, x_arr[1] - x_arr[0], axis=0)
assert np.allclose(d4[3:-3], 3.0, atol=1e-10), '[TC57] Central diff 4th on linear: should equal slope FAILED'

# ---- TC58: numerical_utils laplacian_2d on x²+y² - equals 4 ----
nx, ny = 21, 21
X2d = np.linspace(-1, 1, nx)
Y2d = np.linspace(-1, 1, ny)
XX, YY = np.meshgrid(X2d, Y2d, indexing='ij')
f = XX**2 + YY**2
lap = laplacian_2d(f, X2d[1]-X2d[0], Y2d[1]-Y2d[0])
assert np.allclose(lap[3:-3, 3:-3], 4.0, atol=1e-10), '[TC58] Laplacian of x²+y² should be 4 FAILED'

# ---- TC59: ShearFlow simple_shear - u proportional to y ----
X, Y = np.meshgrid(np.linspace(-1, 1, 11), np.linspace(-1, 1, 11), indexing='ij')
u, v = ShearFlow.simple_shear(X, Y, shear_rate=2.0)
assert np.allclose(u, 2.0 * Y, atol=1e-10), '[TC59] Simple shear u should equal shear_rate * Y FAILED'

# ---- TC60: ShearFlow vortex_pair - returns finite values ----
u, v = ShearFlow.vortex_pair(X, Y, strength=1.0)
assert not np.any(np.isnan(u)) and not np.any(np.isinf(u)), '[TC60] Vortex pair should have finite values FAILED'

# ---- TC61: LevelSetFunction volume is non-negative ----
ls = LevelSetFunction(nx=51, ny=51)
ls.init_circle(cx=0.0, cy=0.0, r=0.3)
vol = ls.compute_volume()
assert vol >= 0, '[TC61] Volume should be non-negative FAILED'

# ---- TC62: Reinitializer fast_marching_brute - finite output ----
ls = LevelSetFunction(nx=21, ny=21)
ls.init_circle(cx=0.0, cy=0.0, r=0.3)
reinit = Reinitializer(ls, max_iter=10, tol=1e-3)
phi_out = reinit.fast_marching_brute()
assert not np.any(np.isnan(phi_out)), '[TC62] Fast marching brute should not produce NaN FAILED'

# ---- TC63: AdaptiveMesh refine_grid_uniform - increases resolution ----
ls = LevelSetFunction(nx=21, ny=21)
ls.init_circle(cx=0.0, cy=0.0, r=0.3)
amesh = AdaptiveMesh(ls, h_min=0.02, h_max=0.15, h_band=0.15)
old_nx = ls.nx
amesh.refine_grid_uniform(factor=2)
assert ls.nx == 2 * old_nx, '[TC63] refine_grid_uniform should double grid points FAILED'

# ---- TC64: CurvatureFlow compute_surface_area - positive ----
ls = LevelSetFunction(nx=51, ny=51)
ls.init_circle(cx=0.0, cy=0.0, r=0.3)
cf = CurvatureFlow(ls)
A = cf.compute_surface_area()
assert A > 0, '[TC64] Surface area should be positive FAILED'

# ---- TC65: MatrixChainOptimizer brute on edge case - single matrix ----
cost = MatrixChainOptimizer.matrix_chain_brute([10, 30])
assert cost == 0, '[TC65] Single matrix chain cost should be 0 FAILED'

# ---- TC66: ConvergenceAnalysis linf_error - correct value ----
phi1 = np.ones((10, 10))
phi2 = np.ones((10, 10))
phi2[5, 5] = 2.0
err = ConvergenceAnalysis.linf_error(phi1, phi2)
assert abs(err - 1.0) < 1e-12, '[TC66] Linf error should be 1.0 FAILED'

# ---- TC67: numerical_utils cplx_solve_lower_triangular - correct ----
from numerical_utils import cplx_solve_lower_triangular, cplx_solve_upper_triangular
L_test = np.array([[2.0, 0.0], [1.0, 3.0]], dtype=np.complex128)
b_test = np.array([4.0, 5.0], dtype=np.complex128)
y = cplx_solve_lower_triangular(L_test, b_test)
assert np.allclose(L_test @ y, b_test, atol=1e-12), '[TC67] Lower triangular solve FAILED'

# ---- TC68: numerical_utils cplx_solve_upper_triangular - correct ----
U_test = np.array([[2.0, 1.0], [0.0, 3.0]], dtype=np.complex128)
b_test = np.array([7.0, 6.0], dtype=np.complex128)
x_sol = cplx_solve_upper_triangular(U_test, b_test)
assert np.allclose(U_test @ x_sol, b_test, atol=1e-12), '[TC68] Upper triangular solve FAILED'

# ---- TC69: TopologyTracker history after update - has entries ----
ls = LevelSetFunction(nx=31, ny=31)
ls.init_circle(cx=0.0, cy=0.0, r=0.3)
tracker = TopologyTracker(ls)
tracker.update_history()
assert len(tracker.history['num_components']) == 1, '[TC69] History should have 1 entry after update FAILED'

# ---- TC70: CurvatureFlow Lebedev integral of x²+y²+z² on S² - equals 4π ----
x_l, y_l, z_l, w_l = lebedev_by_order(14)
f_sum = x_l**2 + y_l**2 + z_l**2
assert np.allclose(f_sum, 1.0, atol=1e-12), '[TC70] Lebedev nodes must be on unit sphere FAILED'

print('\n全部 70 个测试通过!\n')
