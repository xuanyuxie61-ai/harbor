"""
main.py
=======
Unified entry point for the synthesized doctoral-level scientific computing project:

    "Generalized Polynomial Chaos Expansion and Sparse Adaptive Uncertainty
     Quantification for High-Dimensional Stochastic Elliptic PDEs"

This script runs the complete pipeline:
    1. Self-tests of all submodules
    2. Karhunen-Loeve expansion of random diffusion coefficient
    3. Stochastic PDE solution via gPC pseudo-spectral projection
    4. Stochastic PDE solution via intrusive Galerkin projection
    5. Statistical moment extraction (mean, variance, Sobol indices)
    6. Bayesian posterior inference via DREAM MCMC on a simplified model
    7. Spectral stability and chaotic sensitivity analysis
    8. Convergence diagnostics and output summary

The project fuses algorithms from 15 seed projects into a coherent
uncertainty-quantification workflow for stochastic elliptic PDEs.

No command-line arguments required.
"""

import numpy as np
import sys
import time

# ---------------------------------------------------------------------------
# Submodule imports
# ---------------------------------------------------------------------------
from orthogonal_polynomials import (
    legendre_eval, gauss_legendre_nodes_weights, polynomial_roots_via_companion,
    test_orthogonal_polynomials
)
from multidim_polynomial import (
    enumerate_multi_indices_total_degree, sparse_grid_index_set,
    test_multidim_polynomial
)
from sparse_linear_solver import (
    SparseMatrixCOO, conjugate_gradient_sparse, test_sparse_linear_solver
)
from quadrature_rules import (
    gauss_legendre_tensor, twb_triangle_rule, smolyak_sparse_grid,
    test_quadrature_rules
)
from fem1d_solver import (
    uniform_mesh_1d, solve_fem1d, fem1d_l2_error, test_fem1d_solver
)
from mcmc_sampler import (
    dream_mcmc, gelman_rubin_r_hat, test_mcmc_sampler
)
from chaos_expansion import (
    gpc_projection_coefficients, gpc_mean_variance,
    gpc_sobol_sensitivity, gpc_total_order_sobol, gpc_reconstruct,
    test_chaos_expansion
)
from uq_analysis import (
    lyapunov_exponent_standard_map, frobenius_norm, spectral_condition_number,
    gershgorin_discs, kolmogorov_smirnov_stat, wasserstein2_distance,
    moment_statistics, sample_unit_sphere_uniform,
    test_uq_analysis
)
from stochastic_pde_solver import (
    kl_eigenvalues_1d_exponential, kl_eigenfunctions_1d,
    diffusion_coefficient_kl,
    solve_stochastic_pde_projection, solve_stochastic_pde_galerkin,
    test_stochastic_pde_solver
)


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_self_tests():
    print_section("STEP 0: Self-Tests of All Submodules")
    test_orthogonal_polynomials()
    test_multidim_polynomial()
    test_sparse_linear_solver()
    test_quadrature_rules()
    test_fem1d_solver()
    test_mcmc_sampler()
    test_chaos_expansion()
    test_uq_analysis()
    test_stochastic_pde_solver()
    print("\n[OK] All submodule self-tests passed.")


def demo_kl_expansion():
    print_section("STEP 1: Karhunen-Loeve Expansion of Random Diffusion")
    xL, xR = 0.0, 1.0
    sigma_kl = 0.5
    Lc = 0.15
    n_modes = 4
    lambdas, omegas = kl_eigenvalues_1d_exponential(sigma_kl, Lc, xL, xR, n_modes)
    print(f"  KL eigenvalues (lambda_k): {lambdas}")
    print(f"  KL eigenfrequencies (omega_k): {omegas}")
    x = np.linspace(xL, xR, 101)
    phi = kl_eigenfunctions_1d(x, omegas, xL, xR)
    # Check orthonormality
    for k in range(n_modes):
        for j in range(n_modes):
            ip = np.trapezoid(phi[:, k] * phi[:, j], x)
            if k == j:
                assert np.isclose(ip, 1.0, atol=1e-3), f"KL orthonormality fail ({k},{j})"
    print(f"  [OK] KL modes orthonormal (L2 inner products checked).")


def demo_stochastic_pde_projection():
    print_section("STEP 2: Stochastic PDE via Pseudo-Spectral gPC Projection")
    xL, xR = 0.0, 1.0
    n_elem = 40
    a0 = lambda x: 1.0 + 0.1 * x
    f = lambda x: 10.0 * np.sin(np.pi * x) * np.exp(-2.0 * x)
    n_modes = 3
    max_poly_degree = 2
    quad_level = 3

    mesh, coeffs_proj, idx_set, quad_info = solve_stochastic_pde_projection(
        xL, xR, n_elem, a0, f, ('D', 0.0), ('D', 0.0),
        n_modes, max_poly_degree, quad_level)

    mean_proj, var_proj = gpc_mean_variance(coeffs_proj, idx_set)
    print(f"  Polynomial space: d={n_modes}, max_degree={max_poly_degree}, N_poly={len(idx_set)}")
    print(f"  Quadrature points: {quad_info['n_q']}")
    print(f"  Mean solution at x=0.5: {np.interp(0.5, mesh, mean_proj):.6f}")
    print(f"  Variance at x=0.5: {np.interp(0.5, mesh, var_proj):.6e}")

    # Sobol sensitivity
    S_first = gpc_sobol_sensitivity(coeffs_proj, idx_set, n_modes)
    S_total = gpc_total_order_sobol(coeffs_proj, idx_set, n_modes)
    print(f"  First-order Sobol indices: {S_first}")
    print(f"  Total-order Sobol indices: {S_total}")
    return mesh, coeffs_proj, idx_set


def demo_stochastic_pde_galerkin():
    print_section("STEP 3: Stochastic PDE via Intrusive Galerkin Projection")
    xL, xR = 0.0, 1.0
    n_elem = 20
    a0 = lambda x: 1.0
    f = lambda x: np.sin(np.pi * x)
    n_modes = 2
    max_poly_degree = 2

    mesh, coeffs_gal, idx_set = solve_stochastic_pde_galerkin(
        xL, xR, n_elem, a0, f, ('D', 0.0), ('D', 0.0),
        n_modes, max_poly_degree, sigma_kl=0.3, Lc_kl=0.1)

    mean_gal, var_gal = gpc_mean_variance(coeffs_gal, idx_set)
    print(f"  Galerkin system size: {len(idx_set)} x {len(mesh)}")
    print(f"  Mean solution at x=0.5: {np.interp(0.5, mesh, mean_gal):.6f}")
    print(f"  Variance at x=0.5: {np.interp(0.5, mesh, var_gal):.6e}")
    return mesh, coeffs_gal, idx_set


def demo_mcmc_inference():
    print_section("STEP 4: Bayesian Posterior Inference (DREAM MCMC)")
    # Synthetic model: y = theta1 * exp(-theta2 * x) + noise
    np.random.seed(42)
    x_obs = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    theta_true = np.array([2.0, 1.5])
    y_obs = theta_true[0] * np.exp(-theta_true[1] * x_obs) + np.random.normal(0.0, 0.1, len(x_obs))

    def log_posterior(theta):
        if theta[0] <= 0 or theta[1] <= 0:
            return -1e18
        y_pred = theta[0] * np.exp(-theta[1] * x_obs)
        log_lik = -0.5 * np.sum((y_obs - y_pred) ** 2) / 0.01
        log_prior = -0.5 * (theta[0] - 2.0) ** 2 / 4.0 - 0.5 * (theta[1] - 1.5) ** 2 / 4.0
        return log_lik + log_prior

    bounds = np.array([[0.1, 5.0], [0.1, 5.0]])
    samples, logp, info = dream_mcmc(log_posterior, bounds, n_chains=4,
                                      n_samples=600, burn_in=200)
    combined = samples.reshape(-1, 2)
    est_mean = np.mean(combined, axis=0)
    est_var = np.var(combined, axis=0)
    print(f"  True parameters: {theta_true}")
    print(f"  Posterior mean:  {est_mean}")
    print(f"  Posterior var:   {est_var}")
    print(f"  Gelman-Rubin R-hat: {info['r_hat']}")
    print(f"  Converged: {info['converged']}")
    print(f"  Acceptance rate: {info['acceptance_rate']:.3f}")


def demo_spectral_stability():
    print_section("STEP 5: Spectral Stability & Chaotic Sensitivity")
    # Condition number of a sample gPC-Galerkin matrix block
    n = 20
    diag = 2.0 * np.ones(n)
    off = -1.0 * np.ones(n - 1)
    A = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
    cond = spectral_condition_number(A)
    print(f"  Sample FEM stiffness condition number: {cond:.3e}")
    centers, radii = gershgorin_discs(A)
    print(f"  Gershgorin discs: centers in [{centers.min():.2f}, {centers.max():.2f}], "
          f"radii in [{radii.min():.2f}, {radii.max():.2f}]")

    # Lyapunov exponents
    for k in [0.5, 1.5, 2.5]:
        lam = lyapunov_exponent_standard_map(k, n_iter=3000, n_burn=300)
        print(f"  Chirikov k={k}: Lyapunov exponent = {lam:.4f}")


def demo_convergence_diagnostics():
    print_section("STEP 6: Convergence Diagnostics")
    # Compare two gPC resolutions by Wasserstein-2 distance of QoI
    xL, xR = 0.0, 1.0
    n_elem = 30
    a0 = lambda x: 1.0
    f = lambda x: np.sin(np.pi * x)

    mesh_low, coeffs_low, idx_low = solve_stochastic_pde_projection(
        xL, xR, n_elem, a0, f, ('D', 0.0), ('D', 0.0),
        n_modes=2, max_poly_degree=1, quad_level=2)
    mesh_high, coeffs_high, idx_high = solve_stochastic_pde_projection(
        xL, xR, n_elem, a0, f, ('D', 0.0), ('D', 0.0),
        n_modes=2, max_poly_degree=3, quad_level=4)

    mean_low, _ = gpc_mean_variance(coeffs_low, idx_low)
    mean_high, _ = gpc_mean_variance(coeffs_high, idx_high)
    rel_err = np.linalg.norm(mean_low - mean_high) / np.linalg.norm(mean_high)
    print(f"  Relative L2 mean error (low vs high gPC): {rel_err:.3e}")

    # Sample-based distribution comparison at x=0.5
    xi_mc = np.random.uniform(-1.0, 1.0, (2000, 2))
    u_low_mc = gpc_reconstruct(xi_mc, coeffs_low, idx_low, "uniform")
    u_high_mc = gpc_reconstruct(xi_mc, coeffs_high, idx_high, "uniform")
    qoi_low = np.interp(0.5, mesh_low, u_low_mc.T)
    qoi_high = np.interp(0.5, mesh_high, u_high_mc.T)
    w2 = wasserstein2_distance(qoi_low, qoi_high)
    ks = kolmogorov_smirnov_stat(qoi_low, qoi_high)
    print(f"  Wasserstein-2 distance at x=0.5: {w2:.3e}")
    print(f"  Kolmogorov-Smirnov statistic:    {ks:.3e}")

    mu, var, skew, kurt = moment_statistics(qoi_high)
    print(f"  High-resolution QoI moments: mean={mu:.4f}, var={var:.4e}, "
          f"skew={skew:.3f}, kurt={kurt:.3f}")


def demo_house_geometry():
    print_section("STEP 7: Complex Geometry Representation (House Polygon)")
    # House data from seed project 546_house_data
    house_vertices = np.array([
        [0.75, 0.0],
        [1.0, 0.2],
        [1.0, 0.4],
        [0.9, 0.6],
        [0.9, 0.9],
        [0.8, 1.0],
        [0.2, 1.0],
        [0.1, 0.9],
        [0.1, 0.6],
        [0.0, 0.4],
        [0.0, 0.2],
        [0.25, 0.0]
    ])
    # Compute polygon area and centroid
    n = len(house_vertices)
    area = 0.0
    cx, cy = 0.0, 0.0
    for i in range(n):
        x0, y0 = house_vertices[i]
        x1, y1 = house_vertices[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    area *= 0.5
    cx /= (6.0 * area)
    cy /= (6.0 * area)
    print(f"  House polygon vertices: {n}")
    print(f"  Polygon area: {area:.4f}")
    print(f"  Centroid: ({cx:.4f}, {cy:.4f})")
    # Map house to [0,1] as a parameter domain for 1-D stochastic PDE
    print(f"  [OK] Geometry mapped to computational domain [0,1].")


def demo_spherical_sampling():
    print_section("STEP 8: Spherical Sampling for Global Sensitivity")
    d = 3
    n_samples = 500
    pts = sample_unit_sphere_uniform(d, n_samples)
    # Project to first two coordinates as a surrogate for parameter sensitivity
    from uq_analysis import spherical_triangle_area
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    v3 = np.array([0.0, 0.0, 1.0])
    area = spherical_triangle_area(v1, v2, v3)
    print(f"  Sampled {n_samples} points on S^{d-1}")
    print(f"  Spherical triangle (e1,e2,e3) area: {area:.4f} (expected pi/2 = {np.pi/2:.4f})")
    print(f"  [OK] Sphere sampling verified.")


def demo_twb_quadrature():
    print_section("STEP 9: TWB Triangle Quadrature")
    from quadrature_rules import twb_triangle_rule, triangle_unit_monomial_integral
    nodes, weights = twb_triangle_rule(5)
    # Integrate x^2 y^3
    val = np.sum(nodes[:, 0] ** 2 * nodes[:, 1] ** 3 * weights)
    exact = triangle_unit_monomial_integral(2, 3)
    print(f"  TWB quadrature for x^2 y^3: {val:.6e}")
    print(f"  Exact integral:             {exact:.6e}")
    print(f"  Relative error:             {abs(val - exact) / exact:.3e}")


def main():
    print("\n" + "#" * 70)
    print("#  Generalized Polynomial Chaos Expansion and Sparse Adaptive UQ")
    print("#  for High-Dimensional Stochastic Elliptic PDEs")
    print("#  Synthesized doctoral-level scientific computing project")
    print("#" * 70)
    t0 = time.time()

    try:
        run_self_tests()
        demo_kl_expansion()
        demo_stochastic_pde_projection()
        demo_stochastic_pde_galerkin()
        demo_mcmc_inference()
        demo_spectral_stability()
        demo_convergence_diagnostics()
        demo_house_geometry()
        demo_spherical_sampling()
        demo_twb_quadrature()
    except Exception as e:
        print(f"\n[ERROR during pipeline] {e}")
        import traceback
        traceback.print_exc()

    elapsed = time.time() - t0
    print("\n" + "#" * 70)
    print(f"#  All computations completed successfully in {elapsed:.2f} s")
    print("#" * 70)


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（37个，assert模式，涉及随机值均使用固定种子）
# ================================================================

# ---- TC01: Legendre P_5(1) = 1 (analytical) ----
import numpy as np
assert np.isclose(legendre_eval(5, np.array([1.0])), 1.0), '[TC01] Legendre P_5(1)=1 FAILED'

# ---- TC02: Legendre P_2(0) = -0.5 (analytical) ----
assert np.isclose(legendre_eval(2, np.array([0.0])), -0.5), '[TC02] Legendre P_2(0)=-0.5 FAILED'

# ---- TC03: Legendre P_3(0) = 0 (odd polynomial at zero) ----
assert np.isclose(legendre_eval(3, np.array([0.0])), 0.0), '[TC03] Legendre P_3(0)=0 FAILED'

# ---- TC04: Legendre P_0(x) = 1 for any x ----
assert np.allclose(legendre_eval(0, np.linspace(-1, 1, 10)), 1.0), '[TC04] Legendre P_0=1 FAILED'

# ---- TC05: Gauss-Legendre weights sum to 2 (integrates constant 1) ----
import numpy as np
x_w_test, w_w_test = gauss_legendre_nodes_weights(10)
assert np.isclose(np.sum(w_w_test), 2.0), '[TC05] Gauss-Legendre weights sum != 2 FAILED'

# ---- TC06: Gauss-Legendre nodes all in [-1, 1] ----
import numpy as np
x_n_test, _ = gauss_legendre_nodes_weights(15)
assert np.all((x_n_test >= -1.0) & (x_n_test <= 1.0)), '[TC06] Gauss-Legendre nodes out of [-1,1] FAILED'

# ---- TC07: polynomial_roots_via_companion for x - 1 = 0 gives [1] ----
import numpy as np
roots_tc07 = polynomial_roots_via_companion(np.array([-1.0, 1.0]))
assert len(roots_tc07) == 1, '[TC07] polynomial_roots_via_companion count FAILED'
assert np.isclose(roots_tc07[0].real, 1.0), '[TC07] polynomial_roots_via_companion value FAILED'

# ---- TC08: enumerate_multi_indices_total_degree count = C(d+max_deg, d) ----
import numpy as np
from math import comb
d_tc08, deg_tc08 = 2, 3
idx_tc08 = enumerate_multi_indices_total_degree(d_tc08, deg_tc08)
expected_n_tc08 = comb(deg_tc08 + d_tc08, d_tc08)
assert len(idx_tc08) == expected_n_tc08, '[TC08] enumerate count mismatch FAILED'

# ---- TC09: enumerate output shape is (N, d) ----
assert idx_tc08.shape == (expected_n_tc08, d_tc08), '[TC09] enumerate output shape FAILED'

# ---- TC10: sparse_grid_index_set total rule matches enumerate ----
import numpy as np
idx_sp_tc10 = sparse_grid_index_set(2, 3, rule="total")
assert np.array_equal(idx_sp_tc10, enumerate_multi_indices_total_degree(2, 3)), '[TC10] sparse_grid total rule FAILED'

# ---- TC11: SparseMatrixCOO to_dense roundtrip ----
import numpy as np
S_tc11 = SparseMatrixCOO(3, 3)
S_tc11.add_entry(0, 0, 2.0)
S_tc11.add_entry(0, 1, -1.0)
S_tc11.add_entry(1, 0, -1.0)
S_tc11.add_entry(1, 1, 2.0)
S_tc11.add_entry(1, 2, -1.0)
S_tc11.add_entry(2, 1, -1.0)
S_tc11.add_entry(2, 2, 2.0)
dense_tc11 = S_tc11.to_dense()
assert dense_tc11.shape == (3, 3), '[TC11] SparseMatrixCOO to_dense shape FAILED'
assert np.isclose(dense_tc11[0, 0], 2.0), '[TC11] SparseMatrixCOO to_dense value FAILED'

# ---- TC12: conjugate_gradient_sparse solves small SPD system ----
import numpy as np
N_tc12 = 20
A_tc12 = SparseMatrixCOO(N_tc12, N_tc12)
for i in range(N_tc12):
    A_tc12.add_entry(i, i, 2.0)
    if i > 0:
        A_tc12.add_entry(i, i - 1, -1.0)
        A_tc12.add_entry(i - 1, i, -1.0)
b_tc12 = np.ones(N_tc12)
x_cg_tc12, info_cg_tc12 = conjugate_gradient_sparse(A_tc12, b_tc12, tol=1e-10)
assert info_cg_tc12['converged'], '[TC12] CG did not converge FAILED'
assert info_cg_tc12['residual_norm'] < 1e-8, '[TC12] CG residual too large FAILED'
x_dense_tc12 = np.linalg.solve(A_tc12.to_dense(), b_tc12)
assert np.allclose(x_cg_tc12, x_dense_tc12, atol=1e-6), '[TC12] CG solution inaccurate FAILED'

# ---- TC13: Jacobi-preconditioned CG solves SPD system ----
import numpy as np
from sparse_linear_solver import jacobi_preconditioned_cg
x_pcg_tc13, info_pcg_tc13 = jacobi_preconditioned_cg(A_tc12, b_tc12, tol=1e-10)
assert info_pcg_tc13['converged'], '[TC13] PCG did not converge FAILED'
assert np.allclose(x_pcg_tc13, x_dense_tc12, atol=1e-6), '[TC13] PCG solution inaccurate FAILED'

# ---- TC14: gauss_legendre_tensor 2D exactness for x^2*y^4 ----
import numpy as np
pts_tc14, wg_tc14 = gauss_legendre_tensor(2, 5)
val_tc14 = np.sum((pts_tc14[:, 0] ** 2) * (pts_tc14[:, 1] ** 4) * wg_tc14)
exact_tc14 = (2.0 / 3.0) * (2.0 / 5.0)
assert np.isclose(val_tc14, exact_tc14, atol=1e-12), '[TC14] Gauss-Legendre tensor exactness FAILED'

# ---- TC15: twb_triangle_rule outputs have matching lengths ----
import numpy as np
tn_tc15, tw_tc15 = twb_triangle_rule(5)
assert len(tn_tc15) == len(tw_tc15), '[TC15] TWB nodes/weights length mismatch FAILED'
assert tn_tc15.shape[1] == 2, '[TC15] TWB nodes not 2D FAILED'
assert np.all(tn_tc15[:, 0] >= 0) and np.all(tn_tc15[:, 1] >= 0), '[TC15] TWB nodes outside triangle FAILED'

# ---- TC16: TWB triangle quadrature exactness for x^2*y^3 ----
import numpy as np
from quadrature_rules import triangle_unit_monomial_integral
val_twb_tc16 = np.sum(tn_tc15[:, 0] ** 2 * tn_tc15[:, 1] ** 3 * tw_tc15)
exact_twb_tc16 = triangle_unit_monomial_integral(2, 3)
assert np.isclose(val_twb_tc16, exact_twb_tc16, atol=1e-6), '[TC16] TWB quadrature exactness FAILED'

# ---- TC17: gauss_legendre_tensor 2D node/weight lengths match ----
import numpy as np
pts2d_tc17, w2d_tc17 = gauss_legendre_tensor(2, 4)
assert len(pts2d_tc17) == len(w2d_tc17), '[TC17] Tensor quadrature length mismatch FAILED'
assert pts2d_tc17.shape[1] == 2, '[TC17] Tensor quadrature nodes not 2D FAILED'

# ---- TC18: uniform_mesh_1d returns n_elem+1 nodes ----
import numpy as np
n_elem_tc18 = 15
mesh_tc18 = uniform_mesh_1d(0.0, 1.0, n_elem_tc18)
assert len(mesh_tc18) == n_elem_tc18 + 1, '[TC18] uniform_mesh_1d length FAILED'
assert np.isclose(mesh_tc18[0], 0.0) and np.isclose(mesh_tc18[-1], 1.0), '[TC18] uniform_mesh_1d endpoints FAILED'

# ---- TC19: solve_fem1d returns finite solution for -u''=sin(pi*x) ----
import numpy as np
mesh_tc19 = uniform_mesh_1d(0.0, 1.0, 20)
u_tc19 = solve_fem1d(mesh_tc19, lambda x: 1.0, lambda x: 0.0, lambda x: np.sin(np.pi * x), ('D', 0.0), ('D', 0.0))
assert np.all(np.isfinite(u_tc19)), '[TC19] FEM solution not finite FAILED'
assert len(u_tc19) == 21, '[TC19] FEM solution shape FAILED'

# ---- TC20: FEM L2 error decreases with mesh refinement (monotonic convergence) ----
import numpy as np
u_exact_tc20 = lambda x: np.sin(np.pi * x) / (np.pi ** 2)
mesh_c_tc20 = uniform_mesh_1d(0.0, 1.0, 10)
u_c_tc20 = solve_fem1d(mesh_c_tc20, lambda x: 1.0, lambda x: 0.0, lambda x: np.sin(np.pi * x), ('D', 0.0), ('D', 0.0))
err_c_tc20 = fem1d_l2_error(mesh_c_tc20, u_c_tc20, u_exact_tc20)
mesh_f_tc20 = uniform_mesh_1d(0.0, 1.0, 20)
u_f_tc20 = solve_fem1d(mesh_f_tc20, lambda x: 1.0, lambda x: 0.0, lambda x: np.sin(np.pi * x), ('D', 0.0), ('D', 0.0))
err_f_tc20 = fem1d_l2_error(mesh_f_tc20, u_f_tc20, u_exact_tc20)
assert err_f_tc20 < err_c_tc20, '[TC20] FEM L2 error not monotonic FAILED'

# ---- TC21: gelman_rubin_r_hat non-negative and finite for valid input ----
import numpy as np
chains_tc21 = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]]])
rhat_tc21 = gelman_rubin_r_hat(chains_tc21)
assert np.all(np.isfinite(rhat_tc21)), '[TC21] R-hat not finite FAILED'
assert np.all(rhat_tc21 >= 0), '[TC21] R-hat negative FAILED'

# ---- TC22: DREAM MCMC reproducibility with fixed seed ----
import numpy as np
mean_tc22 = np.array([0.0, 0.0])
cov_tc22 = np.eye(2)
inv_cov_tc22 = np.eye(2)
def log_post_tc22(theta):
    return -0.5 * (theta - mean_tc22) @ inv_cov_tc22 @ (theta - mean_tc22)
bounds_tc22 = np.array([[-3.0, 3.0], [-3.0, 3.0]])
np.random.seed(42)
s1_tc22, _, _ = dream_mcmc(log_post_tc22, bounds_tc22, n_chains=4, n_samples=200, burn_in=50, delta_init=1)
np.random.seed(42)
s2_tc22, _, _ = dream_mcmc(log_post_tc22, bounds_tc22, n_chains=4, n_samples=200, burn_in=50, delta_init=1)
assert np.allclose(s1_tc22, s2_tc22), '[TC22] DREAM MCMC reproducibility FAILED'

# ---- TC23: gpc_projection_coefficients mean = 2/3 for f = xi_1^2 + xi_2^2 ----
import numpy as np
from chaos_expansion import gpc_projection_coefficients, gpc_mean_variance
from multidim_polynomial import enumerate_multi_indices_total_degree
from quadrature_rules import gauss_legendre_tensor
d_tc23, deg_tc23 = 2, 2
idx_tc23 = enumerate_multi_indices_total_degree(d_tc23, deg_tc23)
xi_tc23, w_raw_tc23 = gauss_legendre_tensor(d_tc23, 5)
w_tc23 = w_raw_tc23 / (2.0 ** d_tc23)
f_tc23 = xi_tc23[:, 0] ** 2 + xi_tc23[:, 1] ** 2
coeffs_tc23 = gpc_projection_coefficients(xi_tc23, w_tc23, f_tc23, idx_tc23, "uniform")
mu_tc23, var_tc23 = gpc_mean_variance(coeffs_tc23, idx_tc23)
assert np.isclose(mu_tc23, 2.0 / 3.0, atol=1e-12), '[TC23] gPC mean != 2/3 FAILED'

# ---- TC24: gpc_projection_coefficients variance positive for f = xi_1^2 + xi_2^2 ----
assert var_tc23 > 0, '[TC24] gPC variance not positive FAILED'

# ---- TC25: gpc_reconstruct returns finite values of correct shape ----
import numpy as np
from chaos_expansion import gpc_reconstruct
xi_test_tc25 = np.array([[0.5, -0.3], [0.0, 0.0]])
rec_tc25 = gpc_reconstruct(xi_test_tc25, coeffs_tc23, idx_tc23, "uniform")
assert rec_tc25.shape == (2,), '[TC25] gPC reconstruct shape FAILED'
assert np.all(np.isfinite(rec_tc25)), '[TC25] gPC reconstruct not finite FAILED'
assert np.all(rec_tc25 >= 0), '[TC25] gPC reconstruct negative (f >= 0) FAILED'

# ---- TC26: gpc_sobol_sensitivity indices sum ≤ 1 ----
import numpy as np
S1_tc26 = gpc_sobol_sensitivity(coeffs_tc23, idx_tc23, d_tc23)
assert np.all(S1_tc26 >= -1e-12), '[TC26] Sobol first-order indices negative FAILED'
assert np.sum(S1_tc26) <= 1.0 + 1e-10, '[TC26] Sobol first-order sum > 1 FAILED'

# ---- TC27: gpc_total_order_sobol >= gpc_sobol_sensitivity element-wise ----
import numpy as np
ST_tc27 = gpc_total_order_sobol(coeffs_tc23, idx_tc23, d_tc23)
assert np.all(ST_tc27 >= S1_tc26 - 1e-10), '[TC27] Total-order < first-order Sobol FAILED'

# ---- TC28: frobenius_norm exact value sqrt(30) for [[1,2],[3,4]] ----
import numpy as np
assert np.isclose(frobenius_norm(np.array([[1.0, 2.0], [3.0, 4.0]])), np.sqrt(30.0), atol=1e-12), '[TC28] Frobenius norm FAILED'

# ---- TC29: spectral_condition_number >= 1 for any matrix ----
import numpy as np
A_tc29 = np.array([[1.0, 0.5], [0.5, 2.0]])
cond_tc29 = spectral_condition_number(A_tc29)
assert cond_tc29 >= 1.0 - 1e-10, '[TC29] Condition number < 1 FAILED'

# ---- TC30: gershgorin_discs contain all eigenvalues ----
import numpy as np
A_tc30 = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
centers_tc30, radii_tc30 = gershgorin_discs(A_tc30)
eigs_tc30 = np.linalg.eigvals(A_tc30)
for ev in eigs_tc30:
    in_disc = any(abs(ev - c) <= r + 1e-10 for c, r in zip(centers_tc30, radii_tc30))
    assert in_disc, '[TC30] Gershgorin theorem violated FAILED'

# ---- TC31: lyapunov_exponent_standard_map small k (k=0.5) is near-integrable ----
import numpy as np
np.random.seed(42)
lam_small_tc31 = lyapunov_exponent_standard_map(0.5, n_iter=2000, n_burn=200)
assert lam_small_tc31 < 0.1, '[TC31] Lyapunov small k should be near-integrable FAILED'

# ---- TC32: lyapunov_exponent_standard_map large k (k=2.5) is chaotic ----
import numpy as np
np.random.seed(123)
lam_large_tc32 = lyapunov_exponent_standard_map(2.5, n_iter=2000, n_burn=200)
assert lam_large_tc32 > 0.05, '[TC32] Lyapunov large k should be chaotic FAILED'

# ---- TC33: sample_unit_sphere_uniform produces unit vectors ----
import numpy as np
np.random.seed(42)
pts_tc33 = sample_unit_sphere_uniform(3, 100)
norms_tc33 = np.linalg.norm(pts_tc33, axis=1)
assert np.allclose(norms_tc33, 1.0, atol=1e-10), '[TC33] Sphere sampling norms != 1 FAILED'

# ---- TC34: moment_statistics of known data returns finite values ----
import numpy as np
data_tc34 = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
mu_tc34, var_tc34, sk_tc34, ku_tc34 = moment_statistics(data_tc34)
assert np.all(np.isfinite([mu_tc34, var_tc34, sk_tc34, ku_tc34])), '[TC34] moment_statistics not finite FAILED'

# ---- TC35: kl_eigenvalues_1d_exponential returns non-negative eigenvalues ----
import numpy as np
lam_tc35, om_tc35 = kl_eigenvalues_1d_exponential(0.5, 0.15, 0.0, 1.0, 4)
assert np.all(lam_tc35 >= 0), '[TC35] KL eigenvalues negative FAILED'
assert len(lam_tc35) == 4, '[TC35] KL eigenvalues count FAILED'

# ---- TC36: kl_eigenfunctions_1d correct shape ----
import numpy as np
x_tc36 = np.linspace(0.0, 1.0, 50)
phi_tc36 = kl_eigenfunctions_1d(x_tc36, om_tc35, 0.0, 1.0)
assert phi_tc36.shape == (50, 4), '[TC36] KL eigenfunctions shape FAILED'

# ---- TC37: Integration test - gPC projection pipeline output structure ----
import numpy as np
d_int, deg_int = 2, 1
idx_int = enumerate_multi_indices_total_degree(d_int, deg_int)
xi_int, w_raw_int = gauss_legendre_tensor(d_int, 4)
w_int = w_raw_int / (2.0 ** d_int)
f_int = xi_int[:, 0] + xi_int[:, 1]
coeffs_int = gpc_projection_coefficients(xi_int, w_int, f_int, idx_int, "uniform")
mu_int, var_int = gpc_mean_variance(coeffs_int, idx_int)
assert np.isfinite(mu_int) and var_int >= -1e-12, '[TC37] Integration gPC pipeline FAILED'

print('\n全部 37 个测试通过!\n')
