"""
Nonlinear Solver Module
=======================
Provides robust nonlinear equation solvers for climate equilibrium
and stability analysis.

Incorporates:
- Continuation method (from 1203_test_con) for tracing solution branches
- Reverse-communication root finder (from 1044_roots_rc)
- Newton-Maehly polynomial root finder (from 801_newton_maehly)

Scientific Background:
----------------------
Climate systems exhibit multiple stable equilibria (ice-free, ice-age,
snowball Earth). Finding and tracing these equilibria requires:

1. Continuation methods for parameter-dependent problems:
   F(u, lambda) = 0
   where lambda is orbital forcing parameter.

2. Stability analysis via eigenvalues of Jacobian:
   det(J - mu*I) = 0
   where J = dF/du.

3. Bifurcation detection where eigenvalues cross imaginary axis.
"""

import numpy as np


def newton_maehly(coeffs, max_iter=100, tol=1e-12):
    """
    Newton-Maehly algorithm for finding all roots of a polynomial.
    From 801_newton_maehly.

    For polynomial P(z) = c[0] + c[1]*z + ... + c[d]*z^d,
    simultaneously finds all d roots using deflation-like correction.

    Formula:
    z_i^{new} = z_i - P(z_i) / (P'(z_i) - P(z_i) * sum_{j!=i} 1/(z_i - z_j))

    Parameters
    ----------
    coeffs : array_like
        Complex polynomial coefficients [c0, c1, ..., cd].
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    roots : ndarray
        Complex roots.
    """
    c = np.array(coeffs, dtype=complex)
    d = len(c) - 1
    if d <= 0:
        return np.array([])

    # Cauchy bound for root magnitudes
    radius = 1.0 + np.max(np.abs(c[:-1] / (c[-1] + 1e-20)))
    radius = min(radius, 100.0)

    # Initial guess: roots of unity scaled by radius
    theta = np.linspace(0, 2.0 * np.pi, d, endpoint=False)
    roots = radius * np.exp(1j * theta)

    for iteration in range(max_iter):
        roots_old = roots.copy()

        for i in range(d):
            pz, dpz = poly_and_derivative(c, roots[i])
            s = 0.0
            for j in range(d):
                if j != i:
                    diff = roots[i] - roots[j]
                    if abs(diff) > 1e-15:
                        s += 1.0 / diff
            denom = dpz - pz * s
            if abs(denom) < 1e-15:
                denom = 1e-15
            roots[i] = roots[i] - pz / denom

        # Convergence check
        max_change = np.max(np.abs(roots - roots_old))
        max_poly = np.max(np.abs([poly_and_derivative(c, z)[0] for z in roots]))

        if max_change < tol and max_poly < tol * 10:
            return roots

    return roots


def poly_and_derivative(c, z):
    """
    Evaluate polynomial and its derivative at z using Horner's method.
    From 801_newton_maehly.

    P(z) = c[0] + c[1]*z + ... + c[d]*z^d
    P'(z) = c[1] + 2*c[2]*z + ... + d*c[d]*z^{d-1}

    Parameters
    ----------
    c : array_like
        Coefficients.
    z : complex
        Evaluation point.

    Returns
    -------
    pz, dpz : complex
        Polynomial and derivative values.
    """
    c = np.array(c, dtype=complex)
    d = len(c) - 1
    if d < 0:
        return 0.0, 0.0

    p = c[d]
    dp = 0.0
    for k in range(d - 1, -1, -1):
        dp = dp * z + p
        p = p * z + c[k]

    return p, dp


def roots_rc(n, x, fx, q):
    """
    Reverse-communication nonlinear equation solver.
    From 1044_roots_rc.

    Solves F(x) = 0 for n equations without storing the Jacobian.
    Uses secant-like updates with a Q matrix storing past iterates.

    Parameters
    ----------
    n : int
        Number of equations.
    x : ndarray
        Current iterate.
    fx : ndarray
        Function value at x.
    q : ndarray
        State matrix, shape (2*n+2, n+2).
        q[2*n, 0] should be 0.0 for first call.

    Returns
    -------
    xnew : ndarray
        Next point to evaluate.
    ferr : float
        Function error sum(|fx|).
    q : ndarray
        Updated state matrix.
    converged : bool
        Whether converged.
    """
    x = np.asarray(x, dtype=float).flatten()
    fx = np.asarray(fx, dtype=float).flatten()
    q = np.array(q, dtype=float)

    ferr = np.sum(np.abs(fx))

    # Check for first call
    if abs(q[2 * n, 0]) < 1e-15:
        # Initialize
        for i in range(n):
            for j in range(n + 1):
                q[i, j] = 0.0
                q[i + n, j] = 0.0
            q[i, i] = 100.0
            q[i + n, i] = 1.0

        q[2 * n, 0:n] = 1.0e30
        q[2 * n + 1, 0:n] = n

        for i in range(n):
            q[i + n, n] = x[i]

        q[0:n, n] = fx
        q[2 * n, n] = ferr
        q[2 * n + 1, n] = 0.0

        # Initial perturbation for next evaluation
        xnew = x.copy()
        xnew[0] += 0.01
        return xnew, ferr, q, False

    # Not first call - update Q matrix
    jsus = 0
    for i in range(1, n + 1):
        if 2 * n <= q[2 * n + 1, i]:
            q[2 * n, i] = 1.0e30
        if q[2 * n + 1, jsus] < (n + 3) / 2:
            jsus = i
        if (n + 3) / 2 <= q[2 * n + 1, i] and q[2 * n, jsus] < q[2 * n, i]:
            jsus = i

    for i in range(n):
        q[i + n, jsus] = x[i]
        q[i, jsus] = fx[i]

    q[2 * n, jsus] = ferr
    q[2 * n + 1, jsus] = 0.0

    # Find best column
    jsma = 0
    for j in range(n + 1):
        if q[2 * n, j] < q[2 * n, jsma]:
            jsma = j

    # Swap best to last column
    if jsma != n:
        for i in range(2 * n + 2):
            q[i, jsma], q[i, n] = q[i, n], q[i, jsma]

    # Simple secant update for next iterate
    xnew = x.copy()
    if ferr < 1e-8:
        return xnew, ferr, q, True

    # Use pseudo-Newton step with approximate Jacobian
    for i in range(n):
        # Simple damping
        xnew[i] = x[i] * 0.9 + 0.1 * q[i + n, n]

    return xnew, ferr, q, False


def continuation_solver(fun, x0, lambda0, lambda_target, dlambda=0.01,
                         max_steps=1000, tol=1e-6):
    """
    Pseudo-arclength continuation for parameter-dependent problems.
    From 1203_test_con.

    Traces solution branch of F(x, lambda) = 0 from lambda0 to lambda_target.

    Parameters
    ----------
    fun : callable
        Function F(x, lambda) returning ndarray.
    x0 : ndarray
        Initial solution at lambda0.
    lambda0 : float
        Initial parameter.
    lambda_target : float
        Target parameter.
    dlambda : float
        Initial parameter step.
    max_steps : int
        Maximum continuation steps.
    tol : float
        Newton tolerance.

    Returns
    -------
    lambdas : list
        Parameter values.
    solutions : list
        Solution vectors.
    """
    x = np.array(x0, dtype=float)
    lam = lambda0
    lambdas = [lam]
    solutions = [x.copy()]

    direction = 1.0 if lambda_target > lambda0 else -1.0
    step = 0

    while direction * (lam - lambda_target) < 0 and step < max_steps:
        step += 1
        # Predictor: Euler step along tangent
        f_val = fun(x, lam)
        n = len(x)

        # Approximate tangent by finite difference
        dlam_test = dlambda * 0.01
        f_plus = fun(x, lam + dlam_test)
        dx_dlam = (x - x)  # placeholder
        # Simple forward difference for dx/dlambda
        try:
            # Use small perturbation
            dx_dlam_approx = np.zeros(n)
            for i in range(n):
                x_pert = x.copy()
                h_pert = max(abs(x[i]) * 1e-6, 1e-8)
                x_pert[i] += h_pert
                f_pert = fun(x_pert, lam)
                df_dx = (f_pert - f_val) / h_pert
                dx_dlam_approx[i] = -np.sum(df_dx * (f_plus - f_val)) / (dlam_test * max(np.sum(df_dx ** 2), 1e-20))
        except Exception:
            dx_dlam_approx = np.zeros(n)

        # Predictor step
        x_pred = x + dx_dlam_approx * dlambda
        lam_pred = lam + direction * dlambda

        # Corrector: Newton iteration
        x_corr = x_pred.copy()
        for newton_iter in range(20):
            f_corr = fun(x_corr, lam_pred)
            if np.linalg.norm(f_corr) < tol:
                break

            # Finite difference Jacobian
            J = np.zeros((n, n))
            for i in range(n):
                h_pert = max(abs(x_corr[i]) * 1e-6, 1e-8)
                x_pert = x_corr.copy()
                x_pert[i] += h_pert
                J[:, i] = (fun(x_pert, lam_pred) - f_corr) / h_pert

            try:
                delta = np.linalg.solve(J, -f_corr)
                x_corr = x_corr + delta
                x_corr = np.clip(x_corr, -1e6, 1e6)
            except np.linalg.LinAlgError:
                # Reduce step and try again
                dlambda *= 0.5
                break
        else:
            # Newton failed, reduce step
            dlambda *= 0.5
            if abs(dlambda) < 1e-8:
                break
            continue

        # Successful step
        x = x_corr
        lam = lam_pred
        lambdas.append(lam)
        solutions.append(x.copy())

        # Adaptive step size
        dlambda = min(abs(dlambda) * 1.1, abs(lambda_target - lambda0) * 0.1)
        dlambda *= direction

    return lambdas, solutions


def find_equilibrium_temperature_ebm(insolation, albedo_func, olr_func,
                                      T_guess=280.0, tol=1e-4, max_iter=100):
    """
    Find equilibrium temperature for zero-dimensional EBM.
    Solve: (1 - alpha(T)) * Q = A + B * T

    Parameters
    ----------
    insolation : float
        Mean insolation in W/m^2.
    albedo_func : callable
        albedo(T) -> float.
    olr_func : callable
        olr(T) -> float.
    T_guess : float
        Initial temperature guess.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    float
        Equilibrium temperature in K.
    """
    # TODO: Implement Newton iteration to find equilibrium temperature.
    # The energy balance equation is: (1 - alpha(T)) * Q = OLR(T)
    # Use the provided albedo_func and olr_func callbacks.
    # Must handle the case where the Jacobian becomes singular.
    # Must clip temperature to [200, 350] K.
    raise NotImplementedError("Hole_3: Equilibrium temperature solver is not implemented.")


def stability_eigenvalues(J):
    """
    Compute eigenvalues of Jacobian for stability analysis.
    Uses characteristic polynomial + Newton-Maehly root finding.

    Parameters
    ----------
    J : ndarray
        Jacobian matrix.

    Returns
    -------
    eigenvalues : ndarray
        Complex eigenvalues.
    stability : str
        'stable', 'unstable', or 'oscillatory'.
    """
    n = J.shape[0]
    if n <= 2:
        # Direct computation for small systems
        eigenvalues = np.linalg.eigvals(J)
    else:
        # Compute characteristic polynomial coefficients
        # det(lambda*I - J) = 0
        coeffs = np.poly(J)
        # Find roots using Newton-Maehly
        eigenvalues = newton_maehly(coeffs)

    max_real = np.max(np.real(eigenvalues))
    if max_real < -1e-6:
        stability = 'stable'
    elif max_real > 1e-6:
        stability = 'unstable'
    else:
        # Check for imaginary parts
        if np.max(np.abs(np.imag(eigenvalues))) > 1e-6:
            stability = 'oscillatory'
        else:
            stability = 'marginally_stable'

    return eigenvalues, stability
