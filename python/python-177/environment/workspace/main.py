# -*- coding: utf-8 -*-

import numpy as np
import sys
import time


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
    print_section("Phase 1: Level Set Initialization & Geometry")

    ls = LevelSetFunction(nx=101, ny=101, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))

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
    print_section("Phase 2: Reinitialization to Signed Distance Function")

    reinit = Reinitializer(ls, max_iter=80, tol=1e-5)
    it, diff = reinit.reinitialize_jacobi_style(omega=1.0)
    sdf_error = reinit.check_sdf_property()

    print(f"  Reinitialization iterations: {it}")
    print(f"  Final residual: {diff:.3e}")
    print(f"  SDF property error (|∇φ|-1): {sdf_error:.4f}")


def demo_curvature_flow(ls):
    print_section("Phase 3: Curvature Flow & Willmore Energy")

    cf = CurvatureFlow(ls)
    W = cf.compute_willmore_energy()
    A = cf.compute_surface_area()
    var = cf.compute_gauss_map_variance()

    print(f"  Willmore energy W = ∫ κ² dA: {W:.6f}")
    print(f"  Surface area (length): {A:.6f}")
    print(f"  Gauss map variance: {var:.6f}")


    from curvature_flow import lebedev_by_order
    x, y, z, w = lebedev_by_order(14)
    f_vals = np.ones_like(x)
    integral = CurvatureFlow.integrate_on_sphere_surface(f_vals, order=14)
    print(f"  Lebedev quadrature test (∫ 1 dΩ): {integral:.6f} (exact: 12.566)")


def demo_time_evolution(ls):
    print_section("Phase 4: Time Evolution with Volume Correction & Topology Tracking")

    solver = HJSolver(ls, epsilon=0.02, gamma=0.0)
    corrector = VolumeCorrector(ls)
    tracker = TopologyTracker(ls)

    t = 0.0
    dt_base = 0.002
    n_steps = 60

    X, Y = np.meshgrid(ls.x, ls.y, indexing='ij')


    tracker.update_history()
    vol0 = ls.compute_volume()

    print(f"  Time stepping: {n_steps} steps, base dt={dt_base}")
    print(f"  Initial volume (target): {vol0:.6f}")

    for step in range(1, n_steps + 1):

        f_ext = ExternalForcing.oscillatory_normal_forcing(
            X, Y, t, A=0.03, omega=2.0, kx=2.0, ky=2.0
        )


        dt = solver.compute_cfl_dt(forcing=f_ext, cfl=0.3)
        dt = min(dt, dt_base)
        if dt < 1e-10:
            dt = dt_base


        solver.step_rk3(dt, forcing=f_ext)
        t += dt


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
    print_section("Phase 5: Adaptive Mesh & CVT Node Optimization")

    amesh = AdaptiveMesh(ls, h_min=0.02, h_max=0.15, h_band=0.15)
    h_field = amesh.compute_size_function()
    h_min_val = np.min(h_field)
    h_max_val = np.max(h_field)
    h_mean = np.mean(h_field)

    print(f"  Size function range: [{h_min_val:.4f}, {h_max_val:.4f}]")
    print(f"  Mean size: {h_mean:.4f}")


    cvt_points = amesh.cvt_optimize_nodes_2d(num_points=64, max_iter=20, tol=1e-3)
    print(f"  CVT optimized nodes: {len(cvt_points)} points")
    print(f"  Node spread (std): {np.std(cvt_points, axis=0)}")


    quality = amesh.estimate_interface_mesh_quality()
    print(f"  Approximate interface mesh quality: {quality:.4f}")


def demo_convergence_and_pod():
    print_section("Phase 6: Convergence Analysis & POD Reduced-Order Model")


    nx_list = [41, 81, 161]
    errors_l2 = []
    hs = []
    t_test = 0.1

    for nx in nx_list:
        ls = LevelSetFunction(nx=nx, ny=nx, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
        ls.init_circle(cx=0.0, cy=0.0, r=0.3)
        X, Y = np.meshgrid(ls.x, ls.y, indexing='ij')

        exact = np.sqrt(X ** 2 + Y ** 2) - (0.3 + 0.1 * t_test)
        error = ConvergenceAnalysis.l2_error(ls.phi, exact, ls.dx, ls.dy)
        errors_l2.append(error)
        hs.append(ls.dx)
        print(f"  nx={nx:3d}, h={ls.dx:.5f}, L2 error={error:.3e}")

    orders = ConvergenceAnalysis.convergence_order(errors_l2, hs)
    for i, p in enumerate(orders):
        if not np.isnan(p):
            print(f"  Observed convergence order (h={hs[i]:.5f}→{hs[i+1]:.5f}): {p:.2f}")


    snapshots = []
    ls_pod = LevelSetFunction(nx=41, ny=41, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
    ls_pod.init_star_shape()
    for _ in range(10):
        snapshots.append(ls_pod.phi.ravel().copy())
        ls_pod.phi *= 0.95

    rom = ReducedOrderModel(np.column_stack(snapshots))
    Ur, S_r = rom.compute_pod_basis(energy_threshold=0.95)
    energy = rom.get_mode_energy()
    print(f"  POD basis size (95% energy): {rom.r}")
    print(f"  First 3 mode energies: {energy[:3]}")


def demo_sampling_uq():
    print_section("Phase 7: Latin Hypercube Sampling & Uncertainty Quantification")


    samples = LatinCenterSampler.sample(dim_num=3, point_num=20)
    print(f"  Latin Center samples: {samples.shape}")
    print(f"  Sample mean per dim: {np.mean(samples, axis=0)}")
    print(f"  Sample std per dim: {np.std(samples, axis=0)}")


    def test_func(x):

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
    print_section("Phase 8: Matrix Chain Optimization & Constraint Satisfaction")


    dims = [10, 30, 5, 60, 8]
    cost_dp, s = MatrixChainOptimizer.matrix_chain_dp(dims)
    cost_brute = MatrixChainOptimizer.matrix_chain_brute(dims)
    order_str = MatrixChainOptimizer.get_optimal_order(s, 0, len(dims) - 2)

    print(f"  Matrix chain dimensions: {dims}")
    print(f"  DP optimal cost: {cost_dp}")
    print(f"  Brute force cost: {cost_brute}")
    print(f"  Optimal parenthesization: {order_str}")


    solutions = ConstraintSatisfier.young_equation_solver(
        sigma12=1.0, sigma13=0.8, sigma23=0.6, tol=0.01, n_grid=100
    )
    print(f"  Young equation solutions found: {len(solutions)}")
    if solutions:
        sol = solutions[0]
        print(f"  Example contact angles (deg): θ1={np.degrees(sol[0]):.1f}, "
              f"θ2={np.degrees(sol[1]):.1f}, θ3={np.degrees(sol[2]):.1f}")


    flops, _ = OperatorSequenceOptimizer.optimize_preconditioner_chain(
        n_ops=4, dim_in=100, dim_mid=50, dim_out=25
    )
    print(f"  Preconditioner chain optimal FLOPs estimate: {flops}")


def demo_numerical_utils():
    print_section("Phase 9: Complex Linear Algebra Utilities")

    from numerical_utils import cplx_cholesky_decompose, cplx_lu_factor, cplx_qr_factor


    A = np.array([[4.0 + 0.0j, 1.0 - 1.0j],
                  [1.0 + 1.0j, 3.0 + 0.0j]], dtype=np.complex128)
    L = cplx_cholesky_decompose(A)
    recon = L @ L.conj().T
    err_chol = np.max(np.abs(recon - A))
    print(f"  Cholesky reconstruction error: {err_chol:.3e}")


    B = np.array([[2.0 + 1.0j, 3.0 - 2.0j],
                  [1.0 - 1.0j, 4.0 + 0.0j]], dtype=np.complex128)
    Lb, Ub, Pb = cplx_lu_factor(B)
    recon_lu = Pb.T @ Lb @ Ub
    err_lu = np.max(np.abs(recon_lu - B))
    print(f"  LU reconstruction error: {err_lu:.3e}")


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


    ls = demo_levelset_initialization()


    demo_reinitialization(ls)


    demo_curvature_flow(ls)


    demo_time_evolution(ls)


    demo_adaptive_mesh_and_cvt(ls)


    demo_convergence_and_pod()


    demo_sampling_uq()


    demo_optimizer()


    demo_numerical_utils()

    t_elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"  All phases completed successfully in {t_elapsed:.2f} seconds.")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
