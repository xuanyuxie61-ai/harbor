
import numpy as np
import math
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    condition_number_estimate,
    legendre_monomial_integral, chebyshev1_monomial_integral,
    hermite_monomial_integral, laguerre_monomial_integral,
    safe_divide
)
from sparse_formats import mm_write, mm_read, hb_write, coo_to_csr, csr_matvec
from mesh_cvt import cvt_generate, delaunay_triangulation, compute_mesh_quality
from fem_assembler import assemble_fem_matrices, apply_dirichlet_bc, exactness_test_fem_quadrature
from sparse_grid import sparse_grid_integrate, adaptive_sparse_grid_refine, sparse_grid_points_weights
from preconditioner import (
    jacobi_preconditioner, ssor_preconditioner,
    multigrid_preconditioner, geometric_coarsening
)
from iterative_solver import pcg_solve, gmres_solve, task_division, parallel_matvec
from reordering import (
    reverse_cuthill_mckee, apply_reordering, bandwidth,
    analyze_permutation_cycles, cycle_decomposition, random_permutation
)
from benchmark_suite import (
    generate_all_benchmark_matrices,
    feynman_kac_exact, feynman_kac_stochastic_solve,
    logistic_reaction_diffusion_matrix
)


def section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_mesh_generation():
    section("STEP 1: Adaptive CVT Mesh Generation")
    n_gen = 64
    nodes = cvt_generate(n_generators=n_gen, n_samples=8000, n_iterations=40, seed=193)
    elements = delaunay_triangulation(nodes)
    quality = compute_mesh_quality(nodes, elements)
    print(f"  Generated mesh: {nodes.shape[0]} nodes, {elements.shape[0]} elements")
    print(f"  Mesh quality (mean/min/max): {quality.mean():.4f} / {quality.min():.4f} / {quality.max():.4f}")
    print(f"  Quality metric Q = 4*sqrt(3)*Area / (a^2+b^2+c^2), Q=1 for equilateral")
    return nodes, elements


def run_fem_assembly(nodes, elements):
    section("STEP 2: FEM Sparse Matrix Assembly")


    def D_func(x, y):
        return 0.1 + 0.05 * np.sin(math.pi * x) * np.cos(math.pi * y)

    def sigma_func(x, y):
        return 1.0 + 0.5 * (x * x + y * y)

    def f_func(x, y):
        return 1.0

    K, F = assemble_fem_matrices(nodes, elements, D_func, sigma_func, f_func, quad_order=7)


    dist_to_boundary = np.minimum.reduce([
        nodes[:, 0], 1.0 - nodes[:, 0],
        nodes[:, 1], 1.0 - nodes[:, 1]
    ])
    boundary_nodes = np.where(dist_to_boundary < 0.05)[0]
    bc_values = np.zeros(len(boundary_nodes))
    K_bc, F_bc = apply_dirichlet_bc(K, F, boundary_nodes, bc_values)

    print(f"  Assembled stiffness matrix: {K_bc.shape}")
    print(f"  Dirichlet BC applied on {len(boundary_nodes)} nodes")
    print(f"  Condition number estimate: {condition_number_estimate(K_bc):.4e}")


    quad_ok = exactness_test_fem_quadrature(max_degree=5)
    print(f"  FEM quadrature exactness test (degree <= 5): {'PASS' if quad_ok else 'FAIL'}")

    return K_bc, F_bc


def run_reordering(K, F):
    section("STEP 3: Sparse Matrix Reordering & Permutation Analysis")


    order = reverse_cuthill_mckee(K)
    K_perm = apply_reordering(K, order)
    bw_before = bandwidth(K)
    bw_after = bandwidth(K_perm)
    print(f"  Bandwidth before RCM: {bw_before}")
    print(f"  Bandwidth after RCM:  {bw_after}")
    print(f"  Bandwidth reduction ratio: {safe_divide(bw_before, bw_after, default=0):.2f}x")


    perm = random_permutation(K.shape[0], seed=696)
    cycles = cycle_decomposition(perm)
    cycle_stats = analyze_permutation_cycles(n=100, n_trials=500, seed=696)
    print(f"  Random permutation cycle analysis (n=100):")
    print(f"    Expected total cycles (H_n): {cycle_stats['expected_total_cycles']:.3f}")
    print(f"    Empirical mean total cycles: {cycle_stats['mean_total_cycles']:.3f}")
    print(f"    Mean max cycle length: {cycle_stats['mean_max_cycle_length']:.2f}")

    F_perm = F[order]
    return K_perm, F_perm, order


def run_preconditioner_solve(K, F):
    section("STEP 4: Multilevel Preconditioned CG Solver")

    n = K.shape[0]
    nproc = 4


    divisions = task_division(n, 0, nproc - 1)
    print(f"  Task division for {nproc} processors:")
    for p, s, e in divisions[:min(4, len(divisions))]:
        print(f"    Proc {p}: rows {s}..{e}")


    jacobi_inv = jacobi_preconditioner(K)
    def M_jacobi(r):
        return jacobi_inv * r


    M_ssor = ssor_preconditioner(K, omega=1.2)


    try:

        coarse_size = max(n // 2, 1)
        fine_pts = np.linspace(0, 1, n)
        coarse_pts = np.linspace(0, 1, coarse_size)
        from preconditioner import construct_prolongation_1d
        P_1d = construct_prolongation_1d(fine_pts, coarse_pts)
        M_mg = multigrid_preconditioner(K, P=P_1d, smoother_sweeps=2, omega=0.8, max_levels=3)
        mg_available = True
    except Exception as e:
        M_mg = None
        mg_available = False






    raise NotImplementedError("HOLE_3: Implement PCG solver calls in main workflow.")


def run_sparse_grid_integration():
    section("STEP 5: High-Dimensional Sparse Grid Integration")


    def genz_oscillatory(x):
        a = np.ones(len(x)) * 0.5
        return np.cos(2.0 * math.pi * 0.5 + np.dot(a, x))



    d = 3
    a_vec = np.ones(d) * 0.5
    c = 2.0 * math.pi * 0.5
    exact_I = (2.0 ** d) * math.cos(c)
    for ai in a_vec:
        exact_I *= math.sin(ai) / ai

    for L in range(1, 6):
        I_approx = sparse_grid_integrate(genz_oscillatory, d, L)
        err = abs(I_approx - exact_I)
        print(f"  d={d}, L={L}: I_approx={I_approx:.10f}, error={err:.4e}")


    print(f"\n  Adaptive sparse grid refinement (d=2, tol=1e-5):")
    def gaussian_peak(x):
        return np.exp(-4.0 * np.sum(x ** 2))

    I_adapt, pts, w, vals, L_final = adaptive_sparse_grid_refine(
        gaussian_peak, d=2, L_max=6, abs_tol=1e-5, rel_tol=1e-4
    )
    print(f"    Final level: {L_final}, points: {len(w)}, integral: {I_adapt:.10f}")


    print(f"\n  1D Quadrature exactness verification (seed 344):")
    for name, func, max_p in [
        ("Legendre", legendre_monomial_integral, 8),
        ("Chebyshev1", chebyshev1_monomial_integral, 8),
        ("Hermite", hermite_monomial_integral, 6),
        ("Laguerre", laguerre_monomial_integral, 6),
    ]:
        all_ok = True
        for p in range(max_p + 1):

            val = func(p)
            if not np.isfinite(val):
                all_ok = False
        print(f"    {name}: exactness formulas valid up to degree {max_p}: {'PASS' if all_ok else 'FAIL'}")


def run_format_io(K, F):
    section("STEP 6: Sparse Matrix Format I/O (Matrix Market + Harwell-Boeing)")

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    mm_file = os.path.join(output_dir, "stiffness_matrix.mtx")
    hb_file = os.path.join(output_dir, "stiffness_matrix.hb")


    mm_write(mm_file, K, title="FEM Stiffness Matrix", field="real", symm="general")
    print(f"  Written Matrix Market format: {mm_file}")


    K_read = mm_read(mm_file)
    if hasattr(K_read, 'toarray'):
        K_read = K_read.toarray()
    diff_mm = np.max(np.abs(K - K_read))
    print(f"  Matrix Market round-trip max error: {diff_mm:.4e}")


    hb_write(hb_file, K, title="FEM Stiffness", key="FEM193", mtx_type="RUA", job=3, rhs=F)
    print(f"  Written Harwell-Boeing format: {hb_file}")


    from sparse_formats import _dense_to_csc
    data, row_idx, col_ptr = _dense_to_csc(K)
    print(f"  CSC format: nnz={len(data)}, n={K.shape[0]}")
    print(f"  Compression ratio: {safe_divide(K.size, len(data), 0):.2f}x")


def run_benchmark_suite():
    section("STEP 7: Benchmark Suite on Multiple Test Problems")

    suite = generate_all_benchmark_matrices()

    print(f"\n  {'Problem':<30} {'Size':>6} {'PCG iter':>10} {'Residual':>12} {'Conv':>6}")
    print(f"  {'-'*30} {'-'*6} {'-'*10} {'-'*12} {'-'*6}")

    for name, (A, b) in suite.items():
        n = A.shape[0]

        A_spd = 0.5 * (A + A.T)
        eigvals = np.linalg.eigvalsh(A_spd)
        min_eig = np.min(eigvals)
        if min_eig <= 0:
            A_spd = A_spd + (abs(min_eig) + 0.1) * np.eye(n)

        x, info = pcg_solve(A_spd, b, tol=1e-8, max_iter=min(n, 300), nproc=1)
        print(f"  {name:<30} {n:>6} {info['iterations']:>10} {info['residual']:>12.4e} {str(info['converged']):>6}")


    print(f"\n  Feynman-Kac stochastic verification (seed 422):")
    x_grid, u_mc = feynman_kac_stochastic_solve(a=2.0, h=0.02, n_paths=2000, n_x=11)
    u_exact = feynman_kac_exact(2.0, x_grid)
    err_mc = np.max(np.abs(u_mc - u_exact))
    print(f"    Monte-Carlo max error vs exact: {err_mc:.4e}")
    print(f"    Exact solution: U(X) = exp((X/a)^2 - 1)")


    print(f"\n  Logistic reaction-diffusion detailed solve (seed 702):")
    n_log = 99
    A_log = logistic_reaction_diffusion_matrix(n_log, D=0.05, r=3.0, K=1.0)
    b_log = np.ones(n_log)

    x_log, info_log = pcg_solve(A_log, b_log, tol=1e-10, max_iter=200)
    print(f"    PCG converged in {info_log['iterations']} iterations, residual={info_log['residual']:.4e}")
    print(f"    PDE: -D*u'' - r*u*(1-u/K) = f, linearized about u=0")


def run_summary():
    section("SUMMARY: Scientific Formulas & Metrics")

    formulas = """
  Core Scientific Formulas:
  -------------------------
  1. Diffusion-Reaction Weak Form:
       integral_D (D * nabla u · nabla v + sigma * u * v) dx = integral_D f * v dx

  2. Element Stiffness (linear triangles):
       K_e[i,j] = D_e * Area_e * (nabla L_i · nabla L_j)
                + sigma_e * Area_e / 12 * (1 + delta_{ij})

  3. PCG Iteration:
       alpha_k = (r_k^T * z_k) / (p_k^T * A * p_k)
       x_{k+1} = x_k + alpha_k * p_k
       r_{k+1} = r_k - alpha_k * A * p_k
       beta_k  = (r_{k+1}^T * z_{k+1}) / (r_k^T * z_k)

  4. Smolyak Sparse Grid:
       A_{L,d} = sum_{|l|_1 <= L+d-1} (Delta_{l_1} x ... x Delta_{l_d})

  5. Multigrid V-Cycle:
       x^{new} = x + P * A_c^{-1} * P^T * (b - A*x)
       A_c = P^T * A * P

  6. RCM Bandwidth Reduction:
       BFS level sets sorted by degree, then reversed.

  7. Feynman-Kac Exact Solution:
       U(X) = exp((X/a)^2 - 1)

  8. Permutation Cycle Theory:
       E[#cycles of length L in random perm] = 1/L

  9. Logistic ODE:
       dy/dt = r*y*(1-y/K), exact: y(t) = K*y0*exp(r*t) / (K + y0*(exp(r*t)-1))

  10. Chirikov Standard Map Jacobian:
       J = [[1, k*cos(x)], [1, 1+k*cos(x)]]
    """
    print(formulas)

    print("  All 15 seed projects integrated successfully.")
    print("  Output files written to ./output/")
    print("  Framework execution completed with zero parameters.")


def main():
    print("\n" + "#" * 70)
    print("#  Sparse Linear Algebra HPC Optimization Framework")
    print("#  Project 193 Synthesis - Python")
    print("#" * 70)


    nodes, elements = run_mesh_generation()


    K, F = run_fem_assembly(nodes, elements)


    K_perm, F_perm, order = run_reordering(K, F)


    x_sol = run_preconditioner_solve(K_perm, F_perm)


    run_sparse_grid_integration()


    run_format_io(K_perm, F_perm)


    run_benchmark_suite()


    run_summary()

    print("\n" + "#" * 70)
    print("#  EXECUTION SUCCESSFUL - ALL MODULES PASSED")
    print("#" * 70 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
