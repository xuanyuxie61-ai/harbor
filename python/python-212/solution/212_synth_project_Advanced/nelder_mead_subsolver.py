"""
nelder_mead_subsolver.py
========================
Nelder-Mead simplex algorithm for derivative-free subproblem minimization.

Mathematical background
-----------------------
Given f : R^n -> R, the Nelder-Mead method maintains a simplex S of n+1
vertices {x_0, ..., x_n} with corresponding function values {f_0, ..., f_n}.
At each iteration, assuming the vertices are ordered so that

    f_0 <= f_1 <= ... <= f_n,

the algorithm performs:

  1. Reflection:
     x_r = (1 + rho) * x_bar - rho * x_n
     where x_bar = (1/n) * sum_{i=0}^{n-1} x_i is the centroid of all but
     the worst vertex. If f_0 <= f_r <= f_{n-1}, replace x_n by x_r.

  2. Expansion (if f_r < f_0):
     x_e = (1 + rho*xi) * x_bar - rho*xi * x_n
     If f_e < f_r, replace x_n by x_e; else replace x_n by x_r.

  3. Outside contraction (if f_{n-1} <= f_r < f_n):
     x_c = (1 + rho*gam) * x_bar - rho*gam * x_n
     If f_c <= f_r, replace x_n by x_c; else shrink.

  4. Inside contraction (if f_r >= f_n):
     x_c = (1 - gam) * x_bar + gam * x_n
     If f_c < f_n, replace x_n by x_c; else shrink.

  5. Shrink:
     x_i = sig * x_i + (1 - sig) * x_0  for i = 1, ..., n

Standard parameter choices (Nelder-Mead 1965):
    rho = 1   (reflection)
    xi  = 2   (expansion)
    gam = 1/2 (contraction)
    sig = 1/2 (shrink)

Convergence:
    The simplex collapses toward a minimizer when
        max_i f_i - min_i f_i < tolerance.
Nelder-Mead does not guarantee convergence in general, but works well in
practice for smooth, low-dimensional problems.

KKT role
--------
In the primal-dual active-set framework, Nelder-Mead is used as a robust
fallback subsolver for the reduced problem when:
  - The gradient of the merit function is unreliable (e.g., near a kink
    where the active set changes), or
  - The Hessian is ill-conditioned and the Newton step is unsafe.

Specifically, after the active set A has stabilized, the reduced KKT system
is solved by a projected Newton method. If the Newton direction fails the
Armijo condition or the merit function increases, we fall back to
Nelder-Mead on the reduced subspace {u_I} (inactive components only),
minimizing the penalized objective

    phi(u_I) = J(u_I, u_A) + (penalty for constraint violation).

This combination of fast local Newton and robust derivative-free global
search is a standard pattern in composite optimization.

References
----------
  - Nelder, J.A., Mead, R., "A simplex method for function minimization",
    Computer Journal, 7(4), 1965, pp. 308-313.
  - Borggaard, J., "nelder_mead" (MATLAB, Virginia Tech).
"""

from __future__ import annotations
import numpy as np
from typing import Callable, Tuple


# ---------------------------------------------------------------------------
# Nelder-Mead parameters (standard choices)
# ---------------------------------------------------------------------------
RHO = 1.0    # reflection
XI  = 2.0    # expansion (must be > max(rho, 1))
GAM = 0.5    # contraction (0 < gam < 1)
SIG = 0.5    # shrink (0 < sig < 1)


# ---------------------------------------------------------------------------
# Core Nelder-Mead driver
# ---------------------------------------------------------------------------
def nelder_mead(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    tol: float = 1.0e-6,
    max_feval: int = 2000,
    initial_simplex: np.ndarray = None,
) -> Tuple[np.ndarray, float, int, bool]:
    """Minimize f(x) using the Nelder-Mead simplex method.

    Parameters
    ----------
    f              : callable, the objective f : R^n -> R
    x0             : (n,) initial guess (used as the first vertex)
    tol            : float, convergence tolerance on the simplex diameter
    max_feval      : int, maximum function evaluations
    initial_simplex: (n+1, n) optional initial simplex; if None, we build one
                     by perturbing x0 along coordinate axes

    Returns
    -------
    x_opt   : (n,) the best vertex found
    f_opt   : float, the minimum value
    n_feval : int, number of function evaluations used
    converged : bool, True if tolerance was met before max_feval
    """
    x0 = np.asarray(x0, dtype=float).ravel()
    n = x0.size

    # Build initial simplex if not provided
    if initial_simplex is None:
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0
        for i in range(n):
            step = max(1.0e-3, 0.05 * abs(x0[i])) if x0[i] != 0.0 else 1.0e-3
            vertex = x0.copy()
            vertex[i] += step
            simplex[i + 1] = vertex
    else:
        simplex = np.asarray(initial_simplex, dtype=float)
        if simplex.shape != (n + 1, n):
            raise ValueError(
                f"initial_simplex shape {simplex.shape} != ({n+1}, {n})")

    # Evaluate initial simplex
    fvals = np.array([f(simplex[i]) for i in range(n + 1)])
    n_feval = n + 1

    # Sort vertices by function value
    order = np.argsort(fvals)
    simplex = simplex[order]
    fvals = fvals[order]

    converged = False
    while n_feval < max_feval:
        # Check convergence: range of function values
        if fvals[-1] - fvals[0] < tol:
            converged = True
            break

        # Centroid of all but the worst vertex
        x_bar = np.mean(simplex[:-1], axis=0)

        # --- 1. Reflection ---
        x_r = (1.0 + RHO) * x_bar - RHO * simplex[-1]
        f_r = f(x_r)
        n_feval += 1

        if fvals[0] <= f_r < fvals[-2]:
            # acceptable reflection
            simplex[-1] = x_r
            fvals[-1] = f_r
        elif f_r < fvals[0]:
            # --- 2. Expansion ---
            x_e = (1.0 + RHO * XI) * x_bar - RHO * XI * simplex[-1]
            f_e = f(x_e)
            n_feval += 1
            if f_e < f_r:
                simplex[-1] = x_e
                fvals[-1] = f_e
            else:
                simplex[-1] = x_r
                fvals[-1] = f_r
        elif fvals[-2] <= f_r < fvals[-1]:
            # --- 3. Outside contraction ---
            x_c = (1.0 + RHO * GAM) * x_bar - RHO * GAM * simplex[-1]
            f_c = f(x_c)
            n_feval += 1
            if f_c <= f_r:
                simplex[-1] = x_c
                fvals[-1] = f_c
            else:
                simplex, fvals, n_feval = _shrink(
                    simplex, fvals, f, n_feval)
        else:
            # --- 4. Inside contraction (f_r >= fvals[-1]) ---
            x_c = (1.0 - GAM) * x_bar + GAM * simplex[-1]
            f_c = f(x_c)
            n_feval += 1
            if f_c < fvals[-1]:
                simplex[-1] = x_c
                fvals[-1] = f_c
            else:
                simplex, fvals, n_feval = _shrink(
                    simplex, fvals, f, n_feval)

        # Re-sort
        order = np.argsort(fvals)
        simplex = simplex[order]
        fvals = fvals[order]

    return simplex[0], fvals[0], n_feval, converged


def _shrink(
    simplex: np.ndarray,
    fvals: np.ndarray,
    f: Callable,
    n_feval: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Shrink the simplex toward the best vertex x_0."""
    n = simplex.shape[1]
    x0 = simplex[0].copy()
    for i in range(1, n + 1):
        simplex[i] = SIG * simplex[i] + (1.0 - SIG) * x0
        fvals[i] = f(simplex[i])
        n_feval += 1
    return simplex, fvals, n_feval


# ---------------------------------------------------------------------------
# Box-constrained variant for KKT subsolves
# ---------------------------------------------------------------------------
def nelder_mead_box(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    x_lower: np.ndarray,
    x_upper: np.ndarray,
    penalty: float = 1.0e6,
    tol: float = 1.0e-6,
    max_feval: int = 2000,
) -> Tuple[np.ndarray, float, int, bool]:
    """Nelder-Mead with box constraints via penalty.

    Constraints: x_lower <= x <= x_upper  (componentwise)

    The penalty method adds a large cost when any component violates its
    bounds, allowing the unconstrained simplex method to handle bounded
    problems. For tight enforcement, use a large penalty.

    KKT context:
        This is used to solve the reduced KKT subproblem on the inactive set,
        where the active constraints (bounds) are fixed. The box penalty
        ensures iterates remain in the feasible region of the original
        inequality-constrained problem.
    """
    x_lower = np.asarray(x_lower, dtype=float)
    x_upper = np.asarray(x_upper, dtype=float)

    def penalized(x):
        val = f(x)
        viol = np.maximum(x_lower - x, 0.0) + np.maximum(x - x_upper, 0.0)
        return val + penalty * np.sum(viol ** 2)

    return nelder_mead(penalized, x0, tol=tol, max_feval=max_feval)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("[nelder_mead] Self-test")
    print("-" * 60)

    # Test 1: Rosenbrock function f(x,y) = (1-x)^2 + 100(y-x^2)^2
    def rosenbrock(z):
        x, y = z
        return (1.0 - x) ** 2 + 100.0 * (y - x ** 2) ** 2

    z0 = np.array([-1.0, 1.0])
    z_opt, f_opt, nf, conv = nelder_mead(rosenbrock, z0,
                                          tol=1.0e-10, max_feval=10000)
    print(f"  Rosenbrock: f(1,1) = 0")
    print(f"    x_opt   = {z_opt}  (err = {np.linalg.norm(z_opt-[1,1]):.2e})")
    print(f"    f_opt   = {f_opt:.2e}")
    print(f"    n_eval  = {nf}, converged = {conv}")

    # Test 2: quadratic with minimum at origin
    def quadratic(z):
        return np.sum(z ** 2)

    z0_2 = np.array([3.0, -2.0, 1.5])
    z_opt2, f_opt2, nf2, conv2 = nelder_mead(quadratic, z0_2,
                                              tol=1.0e-10, max_feval=5000)
    print(f"  Quadratic in R^3: f(0,0,0) = 0")
    print(f"    ||x_opt|| = {np.linalg.norm(z_opt2):.2e}")
    print(f"    f_opt     = {f_opt2:.2e}")
    print(f"    n_eval    = {nf2}, converged = {conv2}")

    # Test 3: box-constrained
    def box_f(z):
        return (z[0] - 3.0) ** 2 + (z[1] - 3.0) ** 2

    z0_3 = np.array([1.0, 1.0])
    lb = np.array([0.0, 0.0])
    ub = np.array([2.0, 2.0])
    z_opt3, f_opt3, nf3, conv3 = nelder_mead_box(
        box_f, z0_3, lb, ub, tol=1.0e-10)
    print(f"  Box-constrained quadratic, min at (2,2):")
    print(f"    x_opt   = {z_opt3}  (exact [2,2])")
    print(f"    f_opt   = {f_opt3:.2e}")
    print(f"    n_eval  = {nf3}, converged = {conv3}")
