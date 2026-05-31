"""
Nonlinear Root-Finding for Band Degeneracies
=============================================
Implements robust numerical methods to locate band-crossing (Dirac) points
and solve the self-consistency equations that arise in the moiré
Hamiltonian.

Scientific Background
---------------------
A band crossing occurs when two eigenvalues of the Hamiltonian become
degenerate at some k-point:

    E_n(k) = E_{n+1}(k) .

Near a Dirac point k_D the dispersion is linear:

    E_±(k) = E_D ± ħ v_F |k − k_D| + O(|k − k_D|²) .

To find k_D numerically we define the scalar function

    f(k) = |E_{n+1}(k) − E_n(k)|²

and minimize it to zero.  This is a nonlinear root-finding problem in
2D (or higher if other parameters are varied).

Two algorithms are provided:
  1. Reverse-communication multi-step secant method (adapted from
     Gaston Gonnet's roots_rc) for solving systems of nonlinear equations.
  2. Laguerre's method for polynomial-like local minima, which uses
     first and second derivatives to achieve super-linear convergence.
"""

import numpy as np
from typing import Tuple, List, Optional, Callable


def roots_rc_step(
    n: int,
    x: np.ndarray,
    fx: np.ndarray,
    q: Optional[np.ndarray] = None,
    damp: float = 0.99,
) -> Tuple[np.ndarray, float, Optional[np.ndarray], str]:
    """
    Single step of the reverse-communication multi-step secant method
    for solving F(x) = 0 in ℝⁿ.

    The algorithm maintains a history matrix Q of size (2n+2) × (n+2)
    that records previous iterates and function values.  At each step
    it solves the small linear system

        Q[0:n, 0:n] · Δ = Q[0:n, n]

    and updates x_new = x_old − Δ, with optional damping if the step
    is too large.

    Parameters
    ----------
    n : int
        Dimension of the system.
    x : np.ndarray of shape (n,)
        Current iterate.
    fx : np.ndarray of shape (n,)
        Function value F(x).
    q : np.ndarray or None
        History matrix.  If None, initialize.
    damp : float
        Damping coefficient (0 < damp < 1).

    Returns
    -------
    xnew : np.ndarray of shape (n,)
        Next point at which to evaluate F.
    ferr : float
        Estimated function error norm.
    qnew : np.ndarray or None
        Updated history matrix.
    status : str
        "continue" or "converged".
    """
    x = np.asarray(x, dtype=float)
    fx = np.asarray(fx, dtype=float)
    if x.shape != (n,) or fx.shape != (n,):
        raise ValueError("x and fx must be 1D arrays of length n.")

    if q is None:
        # Initialize Q matrix
        q = np.zeros((2 * n + 2, n + 2))
        q[0:n, n] = x.copy()
        q[n:2 * n, n] = fx.copy()
        q[0:n, n + 1] = x.copy()
        q[n:2 * n, n + 1] = fx.copy()
        # First step: simple perturbation
        xnew = x + 0.01 * (1.0 + np.abs(x))
        ferr = np.linalg.norm(fx)
        return xnew, ferr, q, "continue"

    qnew = q.copy()
    # Store current (x, fx) in the next column
    col = int(qnew[2 * n, 0]) if qnew.shape[0] > 2 * n else 0
    col = col % (n + 1)
    qnew[0:n, col] = x
    qnew[n:2 * n, col] = fx
    qnew[2 * n, 0] = float((col + 1) % (n + 1))

    # Build the small linear system from the last n columns
    # Simplified: use the most recent n distinct points
    cols = []
    for c in range(n + 1):
        if c != col:
            cols.append(c)
    cols = cols[-n:]

    A = np.zeros((n, n))
    b = np.zeros(n)
    for i in range(n):
        b[i] = -fx[i]
        for j in range(n):
            # Finite-difference Jacobian approximation
            dx_j = qnew[0:n, cols[j]] - x
            df_j = qnew[n:2 * n, cols[j]] - fx
            denom = np.dot(dx_j, dx_j)
            if denom > 1e-14:
                A[i, j] = df_j[i] * dx_j[j] / denom
            else:
                A[i, j] = 1.0 if i == j else 0.0

    # Solve for step
    try:
        delta = np.linalg.solve(A + 1e-10 * np.eye(n), b)
    except np.linalg.LinAlgError:
        delta = b  # fallback

    xnew = x + delta
    # Damping if step is large
    step_norm = np.linalg.norm(delta)
    x_norm = np.linalg.norm(x)
    if step_norm > 2.0 * x_norm and x_norm > 1e-10:
        xnew = x + damp * delta

    ferr = np.linalg.norm(fx)
    if ferr < 1e-12:
        return xnew, ferr, qnew, "converged"

    return xnew, ferr, qnew, "continue"


def laguerre_minimize_2d(
    func: Callable[[np.ndarray], float],
    x0: np.ndarray,
    grad_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    hess_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> Tuple[np.ndarray, float, int]:
    """
    Minimize a scalar function f(x) in 2D using a Laguerre-inspired
    iterative scheme.

    The classical Laguerre root-finder for a polynomial p(z) of degree d
    reads

        z_{new} = z − (d·p) / (p′ ± sqrt((d−1)² p′² − d(d−1) p p″))

    We adapt this to optimization by applying it along the gradient
direction.  At each step we compute the directional derivatives
    g = ∇f,  H = ∇²f, and update

        x_{new} = x − η g

    where the step size η is chosen from the Laguerre formula applied
    to the 1D restriction f(x − t g):

        η = (gᵀg) / (gᵀHg + sqrt((gᵀHg)² − (gᵀg)(gᵀH²g))) .

    Parameters
    ----------
    func : callable
        f(x) → float.
    x0 : np.ndarray of shape (2,)
        Initial guess.
    grad_func : callable, optional
        ∇f(x).  If None, computed by finite differences.
    hess_func : callable, optional
        ∇²f(x).  If None, computed by finite differences.
    tol : float
        Convergence tolerance on |∇f|.
    max_iter : int

    Returns
    -------
    x_min : np.ndarray
    f_min : float
    iters : int
    """
    x = np.asarray(x0, dtype=float)
    if x.shape != (2,):
        raise ValueError("x0 must be a 2D vector.")

    def finite_diff_grad(x_):
        h = 1e-6
        g = np.zeros(2)
        for d in range(2):
            xp = x_.copy()
            xm = x_.copy()
            xp[d] += h
            xm[d] -= h
            g[d] = (func(xp) - func(xm)) / (2.0 * h)
        return g

    def finite_diff_hess(x_):
        h = 1e-5
        H = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                xpp = x_.copy()
                xpm = x_.copy()
                xmp = x_.copy()
                xmm = x_.copy()
                xpp[i] += h; xpp[j] += h
                xpm[i] += h; xpm[j] -= h
                xmp[i] -= h; xmp[j] += h
                xmm[i] -= h; xmm[j] -= h
                H[i, j] = (func(xpp) - func(xpm) - func(xmp) + func(xmm)) / (4.0 * h * h)
        return 0.5 * (H + H.T)

    if grad_func is None:
        grad_func = finite_diff_grad
    if hess_func is None:
        hess_func = finite_diff_hess

    for it in range(max_iter):
        g = grad_func(x)
        grad_norm = np.linalg.norm(g)
        if grad_norm < tol:
            return x, func(x), it + 1

        H = hess_func(x)
        # Laguerre-inspired step along gradient direction
        gHg = np.dot(g, H @ g)
        gH2g = np.dot(H @ g, H @ g)
        g2 = np.dot(g, g)

        discriminant = gHg ** 2 - g2 * gH2g
        if discriminant < 0.0:
            discriminant = 0.0
        denom = gHg + np.sqrt(discriminant)
        if abs(denom) < 1e-14:
            denom = 1e-14
        eta = g2 / denom

        x_new = x - eta * g
        # Line search safeguard
        f_old = func(x)
        f_new = func(x_new)
        if f_new > f_old:
            # Backtrack
            for _ in range(10):
                eta *= 0.5
                x_new = x - eta * g
                f_new = func(x_new)
                if f_new < f_old:
                    break
        x = x_new

    return x, func(x), max_iter


def find_band_crossing_2d(
    band_energies_func: Callable[[np.ndarray], np.ndarray],
    n_band: int,
    k0: np.ndarray,
    search_radius: float = 0.1,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, float, int]:
    """
    Locate a 2D band crossing (Dirac point) between bands n_band and
    n_band+1 by minimizing the squared gap.

    Parameters
    ----------
    band_energies_func : callable
        Function k → array of band energies (sorted).
    n_band : int
        Index of the lower band.
    k0 : np.ndarray of shape (2,)
        Initial guess for the crossing location.
    search_radius : float
        Trust region radius in nm^{-1}.
    tol : float
        Tolerance on the gap.

    Returns
    -------
    k_cross : np.ndarray
        Location of the crossing.
    gap : float
        Gap value at k_cross.
    iters : int
        Number of iterations.
    """
    def gap_squared(k):
        energies = band_energies_func(k)
        e_sorted = np.sort(energies)
        if n_band + 1 >= len(e_sorted):
            return 1e6
        delta = e_sorted[n_band + 1] - e_sorted[n_band]
        # Add a penalty for leaving the search region
        penalty = 0.0
        if np.linalg.norm(k - k0) > search_radius:
            penalty = 1e3 * (np.linalg.norm(k - k0) - search_radius) ** 2
        return float(delta ** 2 + penalty)

    k_cross, f_min, iters = laguerre_minimize_2d(
        gap_squared, k0, tol=tol, max_iter=200
    )
    gap = np.sqrt(max(f_min, 0.0))
    return k_cross, gap, iters


def solve_self_consistent_moire(
    fixed_point_func: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 100,
    alpha_mix: float = 0.5,
) -> Tuple[np.ndarray, int, float]:
    """
    Solve the fixed-point equation x = f(x) using Anderson mixing /
    simple mixing.

    Iteration:
        x_{new} = α f(x) + (1−α) x_old

    Parameters
    ----------
    fixed_point_func : callable
        f(x) → x_new.
    x0 : np.ndarray
        Initial guess.
    tol : float
    max_iter : int
    alpha_mix : float
        Mixing parameter.

    Returns
    -------
    x : np.ndarray
        Converged solution.
    iters : int
    residual : float
    """
    x = np.asarray(x0, dtype=float)
    for it in range(max_iter):
        fx = fixed_point_func(x)
        x_new = alpha_mix * fx + (1.0 - alpha_mix) * x
        diff = np.linalg.norm(x_new - x)
        x = x_new
        if diff < tol:
            return x, it + 1, diff
    return x, max_iter, diff
