"""
Nonlinear solvers for the implicit algebraic equations in membrane permeation.

Adapted from:
  - broyden.m (Broyden's quasi-Newton method)
  - Additional trust-region and line-search safeguards.
"""

import numpy as np


def broyden_solve(f, x0, atol=1e-10, rtol=1e-8, maxit=100, maxdim=20):
    """
    Broyden's method for solving f(x) = 0.

    Parameters:
        f: callable, returns residual vector.
        x0: initial guess.
        atol, rtol: stopping tolerances.
        maxit: maximum nonlinear iterations.
        maxdim: maximum Broyden updates before restart.

    Returns:
        x: solution estimate.
        ierr: 0 if converged, 1 otherwise.
        history: list of residual norms.
    """
    x = np.asarray(x0, dtype=float).copy()
    n = x.size

    fc = f(x)
    fnrm = np.linalg.norm(fc) / max(np.sqrt(n), 1.0)
    stop_tol = atol + rtol * fnrm

    stp = np.zeros((n, maxdim), dtype=float)
    stp_nrm = np.zeros(maxdim, dtype=float)
    stp[:, 0] = -fc
    stp_nrm[0] = float(np.dot(stp[:, 0], stp[:, 0]))

    nbroy = 0
    itc = 0
    history = [fnrm]

    while itc < maxit:
        nbroy += 1
        fnrmo = fnrm
        itc += 1
        x = x + stp[:, nbroy - 1]
        fc = f(x)
        fnrm = np.linalg.norm(fc) / max(np.sqrt(n), 1.0)
        history.append(fnrm)

        if fnrm <= stop_tol:
            return x, 0, history

        if fnrmo <= fnrm:
            return x, 1, history

        if nbroy < maxdim:
            z = -fc.copy()
            if nbroy > 1:
                for kbr in range(nbroy - 1):
                    z = z + stp[:, kbr + 1] * (np.dot(stp[:, kbr], z) / stp_nrm[kbr])
            zz = np.dot(stp[:, nbroy - 1], z) / stp_nrm[nbroy - 1]
            stp[:, nbroy] = z / (1.0 - zz)
            stp_nrm[nbroy] = float(np.dot(stp[:, nbroy], stp[:, nbroy]))
        else:
            stp[:, 0] = -fc
            stp_nrm[0] = float(np.dot(stp[:, 0], stp[:, 0]))
            nbroy = 0

    return x, 1, history


def newton_raphson_solve(f, x0, df=None, atol=1e-10, rtol=1e-8, maxit=50):
    """
    Classical Newton-Raphson with finite-difference Jacobian fallback.
    """
    x = np.asarray(x0, dtype=float).copy()
    n = x.size
    eps_fd = np.sqrt(np.finfo(float).eps)

    for itc in range(maxit):
        r = f(x)
        fnrm = np.linalg.norm(r) / max(np.sqrt(n), 1.0)
        if fnrm <= atol + rtol * fnrm:
            return x, 0

        if df is not None:
            J = df(x)
        else:
            J = np.zeros((n, n), dtype=float)
            for j in range(n):
                xj = x.copy()
                h = eps_fd * max(abs(xj[j]), 1.0)
                xj[j] += h
                J[:, j] = (f(xj) - r) / h

        try:
            dx = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:
            return x, 1

        # Simple line search
        alpha = 1.0
        for _ in range(10):
            x_new = x + alpha * dx
            r_new = f(x_new)
            if np.linalg.norm(r_new) < np.linalg.norm(r):
                x = x_new
                break
            alpha *= 0.5
        else:
            x = x + alpha * dx

    return x, 1


def solve_permeation_nonlinear(p_feed_co2, p_feed_ch4, P_co2, P_ch4,
                                p_perm, membrane_thickness,
                                p_perm_co2_guess, p_perm_ch4_guess,
                                T=308.15):
    """
    Solve the nonlinear algebraic system for permeate-side partial pressures
    and stage cut under the assumption of ideal mixing on the permeate side.

    For a binary mixture with ideal mixing, the permeate composition satisfies:
        y_co2 / y_ch4 = (P_co2/P_ch4) * (p_feed_co2 - p_perm_co2) / (p_feed_ch4 - p_perm_ch4)
    where y_i = p_perm_i / p_perm_total.  This yields a scalar nonlinear equation
    for p_perm_co2, solved here by robust bisection + Broyden refinement.
    """
    L = membrane_thickness
    alpha = P_co2 / P_ch4 if P_ch4 > 0 else 1.0
    p_total = p_perm

    def scalar_residual(ppco2):
        ppch4 = p_total - ppco2
        if ppco2 <= 0 or ppch4 <= 0 or ppco2 >= p_total:
            return 1e30
        lhs = ppco2 / ppch4
        rhs = alpha * (p_feed_co2 - ppco2) / max(p_feed_ch4 - ppch4, 1e-30)
        return lhs - rhs

    # Bisection bracketing
    a = 1e-6
    b = p_total - 1e-6
    fa = scalar_residual(a)
    fb = scalar_residual(b)
    # Ensure sign change
    if fa * fb > 0:
        # Fallback: return feed ratio scaled guess
        ppco2 = p_total * p_feed_co2 / (p_feed_co2 + p_feed_ch4)
        ppch4 = p_total - ppco2
        # theta from total flux
        J_total = P_co2 / L * (p_feed_co2 - ppco2) + P_ch4 / L * (p_feed_ch4 - ppch4)
        RT = 8.314 * T
        theta = J_total * RT / max(p_total, 1e-30)
        theta = np.clip(theta, 0.0, 0.999)
        return np.array([ppco2, ppch4, theta]), 1

    for _ in range(100):
        c = 0.5 * (a + b)
        fc = scalar_residual(c)
        if abs(fc) < 1e-12:
            a = b = c
            break
        if fa * fc <= 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    ppco2 = 0.5 * (a + b)
    ppch4 = p_total - ppco2

    # Stage cut from total flux
    J_total = P_co2 / L * (p_feed_co2 - ppco2) + P_ch4 / L * (p_feed_ch4 - ppch4)
    RT = 8.314 * T
    theta = J_total * RT / max(p_total, 1e-30)
    theta = float(np.clip(theta, 0.0, 0.999))

    return np.array([ppco2, ppch4, theta]), 0
