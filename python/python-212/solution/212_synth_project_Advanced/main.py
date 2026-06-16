"""
main.py
=======
Unified entry point for the KKT-VPDE solver.

This script solves the following PDE-constrained optimal control problem:

    min  J(y, u) = 0.5 ||y - y_d||_M^2 + 0.5 alpha ||u||_M^2 + 0.5 (u - u_b)^T B^{-1} (u - u_b)
    s.t. -nu Laplacian(y) + R(y; T1, T2) = u + f    in Omega = [0,1]^2
         y = 0                                         on dOmega
         u_a <= u(x) <= u_b                            a.e. in Omega
         integral_Omega u dx <= E_max

using a primal-dual active-set strategy for the inequality constraints,
with the KKT saddle-point system solved at each iteration via PLU
factorisation with iterative refinement.

The code is organised into modules that reflect the mathematical structure:

  physics_models       -> physical parameters and Bloch relaxation
  pde_operator         -> discretised PDE operator
  cost_functional      -> objective functional and gradients
  matrix_kernels       -> Hilbert, covariance, block KKT assembly
  linear_algebra       -> PLU factorisation, CG, iterative refinement
  kkt_system           -> KKT system solve and residual
  active_set           -> primal-dual active-set strategy
  integral_constraints -> quadrature rules and integral constraint
  chebyshev_accelerator-> Chebyshev convergence acceleration
  variational_assimilation -> 4D-Var data assimilation framework
  line_search          -> Brent's method with reverse communication
  nelder_mead_subsolver-> Nelder-Mead simplex for derivative-free subsolves
  bayesian_precond     -> HTP Bayesian preconditioner for KKT system

Run with zero arguments:
    python main.py

Scientific output:
  - KKT residual history
  - Active-set evolution
  - Optimal control and state statistics
  - Quadrature exactness diagnostics
  - Condition-number estimates
"""

from __future__ import annotations
import sys
import time
import numpy as np

# Project modules
import physics_models as pm
import pde_operator as pde
import cost_functional as cost
import matrix_kernels as mk
import linear_algebra as la
import kkt_system as kkt
import active_set as aset
import integral_constraints as ic
import chebyshev_accelerator as cheb
import variational_assimilation as va
import line_search as ls
import nelder_mead_subsolver as nm
import bayesian_precond as bp


# ---------------------------------------------------------------------------
def print_banner():
    banner = r"""
====================================================================
 KKT-VPDE: Primal-Dual Active-Set Solver for KKT Systems in
           PDE-Constrained Optimal Control with Mixed Constraints
====================================================================
 Scientific domain : Mathematical Optimisation / KKT Conditions
 PDE constraint    : Bloch-Torrey-type reaction-diffusion
 Inequality constr : Box bounds + integral energy budget
 Solver            : Semismooth Newton / primal-dual active set
====================================================================
"""
    print(banner)


# ---------------------------------------------------------------------------
def run_quadrature_diagnostics(N: int):
    """Run quadrature exactness tests and print diagnostics."""
    print("\n[DIAG] Quadrature exactness tests")
    print("-" * 60)

    # 1. Newton-Cotes open rule
    for nco in [3, 5, 7]:
        x, w = ic.line_nco_rule(nco, 0.0, 1.0)
        # Test: integral of x^k from 0 to 1 = 1/(k+1)
        max_err = 0.0
        for k in range(min(nco, 6)):
            exact = 1.0 / (k + 1)
            approx = float(np.sum(w * x ** k))
            max_err = max(max_err, abs(approx - exact))
        print(f"  Newton-Cotes open (n={nco}): max monomial error = {max_err:.3e}")

    # 2. Gauss-Hermite
    for ngh in [4, 8, 12]:
        nodes, weights = ic.gauss_hermite_nodes_weights(ngh)
        # Test: integral of x^(2k) exp(-x^2) = (2k-1)!! sqrt(pi) / 2^k
        max_err = 0.0
        for k in range(min(ngh, 5)):
            deg = 2 * k
            exact = np.prod([2 * j - 1 for j in range(1, k + 1)]) * np.sqrt(np.pi) / (2 ** k) if k > 0 else np.sqrt(np.pi)
            approx = float(np.sum(weights * nodes ** deg))
            max_err = max(max_err, abs(approx - exact))
        print(f"  Gauss-Hermite (n={ngh}): max moment error = {max_err:.3e}")

    # 3. Triangle quadrature
    tri_results = ic.triangle_exactness_test(degree_max=4)
    for deg, err in tri_results:
        print(f"  Triangle quadrature (deg={deg}): max monomial error = {err:.3e}")

    print("-" * 60)


# ---------------------------------------------------------------------------
def run_conditioning_diagnostics(N: int, params: pm.PhysicalParameters):
    """Print conditioning diagnostics for the PDE operator."""
    print("\n[DIAG] PDE operator conditioning")
    print("-" * 60)
    A = pde.build_pde_operator(N, params, y_lin=None)
    cond_est = la.condition_estimate(A)
    print(f"  N = {N}, n2 = {N*N}")
    print(f"  nu = {params.nu:.4e}, kappa_1 = {params.kappa_1:.4e}, kappa_2 = {params.kappa_2:.4e}")
    print(f"  Peclet = {params.peclet:.4e}, Damkohler = {params.damkohler:.4e}")
    print(f"  cond_1(A) estimate = {cond_est:.4e}")

    # Hilbert matrix conditioning
    for nh in [4, 6, 8]:
        H = mk.hilbert_matrix(nh)
        Hinv = mk.hilbert_inverse(nh)
        err = np.linalg.norm(H @ Hinv - np.eye(nh), np.inf)
        print(f"  Hilbert({nh}): ||H H^{{-1}} - I||_inf = {err:.3e}")
    print("-" * 60)


# ---------------------------------------------------------------------------
def main():
    print_banner()

    # ===================================================================
    # 1. Problem setup
    # ===================================================================
    N = 10  # Grid points per dimension (n2 = 100 interior nodes)
    params = pm.setup_parameters(
        nu=1.0e-1,
        T1=1.0,
        T2=0.3,
        M0=1.0,
        u_lower=0.0,
        u_upper=3.5,
        E_max=1.5,
        alpha=1.0e-2,
        B_scale=5.0e-1,
        y_max=4.0,
    )
    print(f"[SETUP] Grid: N={N}, n2={N*N}, h={1.0/(N+1):.4f}")
    print(f"[SETUP] Physical: nu={params.nu}, T1={params.T1}, T2={params.T2}")
    print(f"[SETUP] Control bounds: [{params.u_lower}, {params.u_upper}]")
    print(f"[SETUP] Energy budget: E_max={params.E_max}")
    print(f"[SETUP] Regularisation: alpha={params.alpha}, B_scale={params.B_scale}")

    # ===================================================================
    # 2. Grid and PDE operator
    # ===================================================================
    x_int, y_int, h, XX, YY = pde.create_grid(N)
    A = pde.build_pde_operator(N, params, y_lin=None)  # Linear PDE
    M_diag = pde.lumped_mass_matrix(N)
    n2 = N * N

    print(f"[PDE] Operator A assembled: {A.shape}, symmetric={np.allclose(A, A.T, atol=1e-10)}")

    # ===================================================================
    # 3. Target state from double-C data (project 314)
    # ===================================================================
    y_d_clean, y_class = va.generate_double_c_target(XX, YY, seed=42)
    # Scale up the target to make the problem harder (forces larger controls)
    y_d_clean = 3.0 * y_d_clean
    # Add small noise to simulate observations
    rng = np.random.default_rng(123)
    y_obs_full = y_d_clean + 0.05 * rng.standard_normal(n2)
    # Use clean as target (denoise via assimilation)
    y_d = y_d_clean.copy()
    print(f"[DATA] Target state generated from double-C geometry (scaled)")
    print(f"[DATA] y_d range: [{y_d.min():.4f}, {y_d.max():.4f}], ||y_d|| = {np.linalg.norm(y_d):.4f}")

    # ===================================================================
    # 4. Background control and covariances (4D-Var style)
    # ===================================================================
    u_b = va.build_background_control(N, params)
    B_cov = mk.build_background_covariance(n2, params.B_scale)
    # Regularise B_cov for inversion
    B_cov += 1.0e-4 * np.eye(n2)
    B_cov_inv = la.invert_matrix(B_cov)
    # Symmetrise
    B_cov_inv = 0.5 * (B_cov_inv + B_cov_inv.T)

    R_cov = mk.build_observation_covariance(10, obs_var=1.0e-2)
    print(f"[4DVAR] Background control u_b: range [{u_b.min():.4f}, {u_b.max():.4f}]")
    print(f"[4DVAR] B_cov condition estimate: {la.condition_estimate(B_cov):.4e}")

    # ===================================================================
    # 5. Integral constraint quadrature weights
    # ===================================================================
    c_vec = ic.quadrature_weight_vector_2d(N)
    print(f"[QUAD] Integral constraint weights: sum = {np.sum(c_vec):.6f} (exact = 1.0)")

    # ===================================================================
    # 6. Cost functional
    # ===================================================================
    J = cost.CostFunctional(
        N=N,
        alpha=params.alpha,
        beta=0.0,  # No L1 for now (handled by active set)
        B_cov_inv=B_cov_inv,
        u_background=u_b,
        y_d=y_d,
    )

    # ===================================================================
    # 7. Initial guess (Nelder-Mead style: use background control)
    # ===================================================================
    u = u_b.copy()
    # Project onto feasible set
    u = aset.project_box(u, params)
    ctu = float(np.dot(c_vec, u))
    if ctu > params.E_max:
        u, _ = aset.project_integral(u, c_vec, params.E_max)
        u = aset.project_box(u, params)

    print(f"[INIT] Initial control: range [{u.min():.4f}, {u.max():.4f}]")
    print(f"[INIT] Initial integral: c^T u = {np.dot(c_vec, u):.4f} (budget = {params.E_max})")

    # ===================================================================
    # 8. Main active-set iteration
    # ===================================================================
    print("\n" + "=" * 70)
    print(" KKT ACTIVE-SET ITERATION")
    print("=" * 70)

    max_iter = 30
    tol_kkt = 1.0e-6
    active_lower = np.zeros(n2, dtype=bool)
    active_upper = np.zeros(n2, dtype=bool)

    cheb_accel = cheb.ChebyshevAccelerator(alpha=0.0, beta=0.9)
    u_prev = None

    header = f"{'Iter':>4s} | {'J':>12s} | {'||res||':>10s} | {'|A_low|':>7s} | {'|A_up|':>7s} | {'lam':>8s} | {'c^T u':>8s} | {'change':>6s}"
    print(header)
    print("-" * len(header))

    t_start = time.time()

    for it in range(max_iter):
        # (a) Compute gradient of Lagrangian w.r.t. u (excluding box multipliers)
        y = la.plu_solve_safe(A, u + np.zeros(n2))  # f = 0 for now
        p_adj = la.plu_solve_safe(A.T, M_diag * (y_d - y))
        grad_u = J.gradient_u(u) - p_adj

        # (b) Identify active sets
        active_lower, active_upper, inactive, changed = aset.update_active_sets(
            u, grad_u, active_lower, active_upper, params
        )

        # (c) Build H_reg = alpha * M + B^{-1}
        H_reg = J.hessian_u_block()

        # (d) Solve the KKT system
        f_vec = np.zeros(n2)  # homogeneous source
        try:
            y, p_adj, u_new, lam = kkt.solve_kkt_system(
                A=A,
                M_diag=M_diag,
                H_reg=H_reg,
                c_vec=c_vec,
                f_vec=f_vec,
                y_d=y_d,
                u_background=u_b,
                B_cov_inv=B_cov_inv,
                E_max=params.E_max,
                inactive=inactive,
                active_lower=active_lower,
                active_upper=active_upper,
                params=params,
            )
        except Exception as e:
            print(f"  [WARN] KKT solve failed at iter {it}: {e}")
            # Fallback: gradient step with projection
            step = 0.01
            u_trial = u - step * grad_u
            u_new = aset.project_box(u_trial, params)
            ctu = float(np.dot(c_vec, u_new))
            if ctu > params.E_max:
                u_new, lam = aset.project_integral(u_new, c_vec, params.E_max)
                u_new = aset.project_box(u_new, params)
            else:
                lam = 0.0
            y = la.plu_solve_safe(A, u_new + f_vec)
            p_adj = la.plu_solve_safe(A.T, M_diag * (y_d - y))

        # (e) Chebyshev acceleration (damped)
        if u_prev is not None:
            u_acc = cheb_accel.accelerate_simple(u_new, u, u_prev)
            u_acc = aset.project_box(u_acc, params)
        else:
            u_acc = u_new.copy()

        # (f) Compute KKT residual
        eta, theta = aset.compute_box_multipliers(grad_u, active_lower, active_upper)
        res = kkt.compute_kkt_residual(
            y=y,
            p=p_adj,
            u=u_acc,
            lam=lam,
            A=A,
            M_diag=M_diag,
            H_reg=H_reg,
            c_vec=c_vec,
            f_vec=f_vec,
            y_d=y_d,
            u_background=u_b,
            B_cov_inv=B_cov_inv,
            E_max=params.E_max,
            params=params,
            eta=eta,
            theta=theta,
        )

        # (g) Evaluate cost
        J_val = J.evaluate(y, u_acc)
        ctu_val = float(np.dot(c_vec, u_acc))

        # (h) Print iteration info
        chg = np.linalg.norm(u_acc - u) / max(np.linalg.norm(u), 1.0e-14)
        print(
            f"{it:4d} | {J_val:12.6f} | {res['total']:10.3e} | "
            f"{np.sum(active_lower):7d} | {np.sum(active_upper):7d} | "
            f"{lam:8.4f} | {ctu_val:8.4f} | {chg:6.3f}"
        )

        # (i) Check convergence
        u_prev = u.copy()
        u = u_acc.copy()

        if res["total"] < tol_kkt:
            print(f"\n[CONV] Converged at iteration {it} with KKT residual {res['total']:.3e}")
            break
    else:
        print(f"\n[WARN] Maximum iterations ({max_iter}) reached")

    t_elapsed = time.time() - t_start
    print(f"[TIME] Elapsed: {t_elapsed:.2f} s")

    # ===================================================================
    # 9. Final results
    # ===================================================================
    print("\n" + "=" * 70)
    print(" FINAL RESULTS")
    print("=" * 70)

    # Final KKT residual
    y_final = la.plu_solve_safe(A, u + np.zeros(n2))
    p_final = la.plu_solve_safe(A.T, M_diag * (y_d - y_final))
    grad_final = J.gradient_u(u) - p_final
    eta_final, theta_final = aset.compute_box_multipliers(
        grad_final,
        active_lower,
        active_upper,
    )
    res_final = kkt.compute_kkt_residual(
        y=y_final,
        p=p_final,
        u=u,
        lam=lam,
        A=A,
        M_diag=M_diag,
        H_reg=H_reg,
        c_vec=c_vec,
        f_vec=np.zeros(n2),
        y_d=y_d,
        u_background=u_b,
        B_cov_inv=B_cov_inv,
        E_max=params.E_max,
        params=params,
        eta=eta_final,
        theta=theta_final,
    )

    print(f"  Final cost J = {J.evaluate(y_final, u):.8f}")
    print(f"    Tracking term  = {J.tracking_term(y_final):.8f}")
    print(f"    Tikhonov term  = {J.tikhonov_term(u):.8f}")
    print(f"    Background term= {J.background_term(u):.8f}")
    print(f"  KKT residual (total) = {res_final['total']:.6e}")
    print(f"    primal_pde     = {res_final['primal_pde']:.6e}")
    print(f"    primal_adj     = {res_final['primal_adj']:.6e}")
    print(f"    stationarity   = {res_final['stationarity']:.6e}")
    print(f"    complementarity= {res_final['complementarity']:.6e}")
    print(f"    integral_viol  = {res_final['integral_viol']:.6e}")
    print(f"    dual_feas      = {res_final['dual_feas']:.6e}")

    # Control statistics
    print(f"\n  Control u:")
    print(f"    range = [{u.min():.6f}, {u.max():.6f}]")
    print(f"    mean  = {u.mean():.6f}")
    print(f"    ||u|| = {np.linalg.norm(u):.6f}")
    print(f"    integral c^T u = {np.dot(c_vec, u):.6f} (budget = {params.E_max})")

    # Active set summary
    print(f"\n  Active set:")
    print(f"    |A_lower| = {np.sum(active_lower)}")
    print(f"    |A_upper| = {np.sum(active_upper)}")
    print(f"    |Inactive| = {np.sum(~(active_lower | active_upper))}")

    # Strict complementarity
    sc = aset.check_strict_complementarity(u, eta_final, theta_final, params)
    print(f"    Strict complementarity: {sc['strict']}")
    print(f"    Violations: {sc['violations']}")
    print(f"    Min non-zero multiplier: {sc['min_multiplier']:.6e}")

    # State statistics
    print(f"\n  State y:")
    print(f"    range = [{y_final.min():.6f}, {y_final.max():.6f}]")
    print(f"    ||y - y_d||_M = {np.sqrt(2.0 * J.tracking_term(y_final)):.6e}")

    # ===================================================================
    # 10. Run diagnostics
    # ===================================================================
    run_quadrature_diagnostics(N)
    run_conditioning_diagnostics(N, params)

    # ===================================================================
    # 11. Chebyshev series demonstration
    # ===================================================================
    print("\n[DIAG] Chebyshev series evaluation")
    print("-" * 60)
    # Approximate f(x) = exp(x) on [-1, 1] with Chebyshev series
    n_cheb = 10
    cheb_nodes = np.cos(np.pi * (np.arange(n_cheb) + 0.5) / n_cheb)
    f_vals = np.exp(cheb_nodes)
    coef = cheb.chebyshev_coefficients(f_vals)
    max_err = 0.0
    for x_test in np.linspace(-1.0, 1.0, 50):
        approx = cheb.chebyshev_series_eval(x_test, coef)
        exact = np.exp(x_test)
        max_err = max(max_err, abs(approx - exact))
    print(f"  Chebyshev approx of exp(x) with {n_cheb} terms: max error = {max_err:.3e}")

    # Derivative test
    max_err_d = 0.0
    for x_test in np.linspace(-0.9, 0.9, 50):
        approx_d = cheb.chebyshev_series_derivative(x_test, coef)
        exact_d = np.exp(x_test)
        max_err_d = max(max_err_d, abs(approx_d - exact_d))
    print(f"  Chebyshev derivative of exp(x): max error = {max_err_d:.3e}")

    # ===================================================================
    # 12. Gamma function test (ASA-314 style)
    # ===================================================================
    print("\n[DIAG] Gamma function (Stirling approximation)")
    print("-" * 60)
    for x_test in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        approx_g = params.gamma_function_stirling(x_test)
        import math as _math
        try:
            exact_g = _math.gamma(x_test)
        except (OverflowError, ValueError):
            exact_g = float("inf")
        if exact_g < 1.0e300:
            rel_err = abs(approx_g - exact_g) / max(abs(exact_g), 1.0e-30)
            print(f"  Gamma({x_test:5.1f}): approx={approx_g:.6e}, exact={exact_g:.6e}, rel_err={rel_err:.3e}")
        else:
            print(f"  Gamma({x_test:5.1f}): approx={approx_g:.6e}")

    # ===================================================================
    # 13. Modular multiplication matrix (quantum-inspired, project 1040)
    # ===================================================================
    print("\n[DIAG] Quantum-inspired structured matrices")
    print("-" * 60)
    for a_test, N_test in [(2, 7), (3, 11), (5, 13)]:
        U = mk.modular_multiplication_matrix(a_test, N_test)
        # Check unitarity (orthogonality for real permutation matrices)
        orth_err = np.linalg.norm(U.T @ U - np.eye(U.shape[0]), np.inf)
        print(f"  Modular mult (a={a_test}, N={N_test}): ||U^T U - I||_inf = {orth_err:.3e}")

    # ===================================================================
    # 14. Brent line search diagnostic (project 695)
    # ===================================================================
    print("\n[DIAG] Brent line search (reverse communication)")
    print("-" * 60)
    # Line-search the KKT merit function along the direction d = u - u_b
    d_ls = u - u_b
    def merit(alpha):
        u_trial = u_b + alpha * d_ls
        u_trial = aset.project_box(u_trial, params)
        y_trial = la.plu_solve_safe(A, u_trial)
        return J.evaluate(y_trial, u_trial)
    # Find bracket: search in [0, 2]
    x_opt, f_opt, n_eval = ls.brent_line_search(merit, 0.0, 2.0)
    print(f"  Minimization of alpha |-> J(u_b + alpha d) on [0, 2]:")
    print(f"    alpha_opt = {x_opt:.6f}")
    print(f"    J_opt     = {f_opt:.6e}")
    print(f"    n_eval    = {n_eval}")

    # Self-test on a quadratic
    x_test, f_test, n_test = ls.brent_line_search(
        lambda x: (x - 1.7) ** 2, 0.0, 3.0)
    print(f"  Sanity check on (x-1.7)^2 on [0,3]:")
    print(f"    x_opt = {x_test:.10f} (exact 1.7, err = {abs(x_test-1.7):.2e})")

    # ===================================================================
    # 15. Nelder-Mead simplex diagnostic (project 797)
    # ===================================================================
    print("\n[DIAG] Nelder-Mead simplex (derivative-free subsolver)")
    print("-" * 60)
    # Solve a small 2D KKT subproblem: min ||u - u_target||^2 s.t. bounds
    u_target = 0.5 * (params.u_lower + params.u_upper) * np.ones(2) + 0.3
    def sub_f(z):
        return float(np.sum((z - u_target) ** 2))
    z0 = np.array([params.u_lower + 0.1, params.u_upper - 0.1])
    z_opt, f_z_opt, n_z, conv_z = nm.nelder_mead_box(
        sub_f, z0,
        x_lower=np.array([params.u_lower, params.u_lower]),
        x_upper=np.array([params.u_upper, params.u_upper]),
        tol=1.0e-10, max_feval=2000)
    print(f"  2D box-constrained quadratic subsolve:")
    print(f"    target    = {u_target}")
    print(f"    x_opt     = {z_opt}")
    print(f"    ||err||   = {np.linalg.norm(z_opt - u_target):.2e}")
    print(f"    f_opt     = {f_z_opt:.2e}")
    print(f"    n_eval    = {n_z}, converged = {conv_z}")

    # Rosenbrock self-test
    def rosen(z):
        return (1.0 - z[0]) ** 2 + 100.0 * (z[1] - z[0] ** 2) ** 2
    z0_r = np.array([-1.0, 1.0])
    z_r, f_r, n_r, conv_r = nm.nelder_mead(rosen, z0_r,
                                             tol=1.0e-10, max_feval=8000)
    print(f"  Rosenbrock (derivative-free):")
    print(f"    x_opt     = {z_r}  (err = {np.linalg.norm(z_r-[1,1]):.2e})")
    print(f"    f_opt     = {f_r:.2e}")
    print(f"    n_eval    = {n_r}, converged = {conv_r}")

    # ===================================================================
    # 16. Bayesian / HTP preconditioner diagnostic (project 1247)
    # ===================================================================
    print("\n[DIAG] Bayesian HTP preconditioner")
    print("-" * 60)
    # Build reduced Hessian approximation H_red = A^T M A + alpha I + B_inv
    # for a small subsample (first min(n2, 40) degrees of freedom)
    n_sub = min(n2, 40)
    A_sub = A[:n_sub, :n_sub]
    M_sub = np.diag(M_diag[:n_sub])
    H_sub = A_sub.T @ M_sub @ A_sub + params.alpha * np.eye(n_sub)
    H_sub += B_cov_inv[:n_sub, :n_sub]
    H_sub = 0.5 * (H_sub + H_sub.T)

    P_htp = bp.HTPPreconditioner.build_from_matrix(H_sub, rank=10)
    print(f"  Reduced KKT Hessian H ({n_sub}x{n_sub}), rank=10:")
    print(f"    cond(H)          = {np.linalg.cond(H_sub):.4e}")
    print(f"    cond(P)          = {P_htp.condition_estimate():.4e}")

    # PCG solve
    rng_pcg = np.random.default_rng(99)
    b_pcg = rng_pcg.standard_normal(n_sub)
    x_pcg, pcg_info = bp.pcg_kkt_solve(
        H_sub, b_pcg, P_htp, tol=1.0e-10, max_iter=300)
    x_exact = np.linalg.solve(H_sub, b_pcg)
    print(f"  PCG solve on reduced KKT system:")
    print(f"    converged      : {pcg_info['converged']}")
    print(f"    iterations     : {pcg_info['iterations']}")
    print(f"    residual       : {pcg_info['residual_norm']:.2e}")
    print(f"    ||x_pcg - x_ex||: {np.linalg.norm(x_pcg - x_exact):.2e}")

    # Variational posterior diagnostic
    H_obs_sub = rng_pcg.standard_normal((8, n_sub))
    R_inv_sub = 1.0 * np.eye(8)
    post_diag = bp.variational_posterior_diagnostic(
        H=H_sub - B_cov_inv[:n_sub, :n_sub],  # data-misfit Hessian
        B_inv=B_cov_inv[:n_sub, :n_sub],
        R_inv=R_inv_sub,
        u_b=u_b[:n_sub],
        y_obs=rng_pcg.standard_normal(8),
        H_obs=H_obs_sub,
    )
    print(f"  Variational posterior diagnostic (n_sub={n_sub}):")
    for k, v in post_diag.items():
        print(f"    {k:30s} = {v:.4e}")

    # ===================================================================
    # Done
    # ===================================================================
    print("\n" + "=" * 70)
    print(" KKT-VPDE solver completed successfully.")
    print("=" * 70)

    return 0


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
