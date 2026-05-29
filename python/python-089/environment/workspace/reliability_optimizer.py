"""
Structural Reliability Analysis and Optimization Module
========================================================
Based on projects:
  - 834_opt_golden (golden section search)
  - 052_asa245 (gamma/log-gamma special functions)

Implements First-Order Reliability Method (FORM) and Second-Order
Reliability Method (SORM) with golden-section optimization for finding
the design point (most probable point of failure).

Key formulas:
- Limit state function: g(x) = 0 defines failure surface
- Standard normal transformation: u = Phi^{-1}(F_X(x))
- Reliability index: beta = min ||u|| subject to g(x(u)) = 0
- Failure probability: P_f ≈ Phi(-beta)  (FORM)
- SORM correction: P_f ≈ Phi(-beta) * prod_i (1 + beta*kappa_i)^{-1/2}
  where kappa_i are principal curvatures at design point.

- Gamma function for chi-square and gamma distributions:
  Gamma(z) = integral_0^inf t^{z-1} e^{-t} dt
  log Gamma via Lanczos approximation for numerical stability.
"""

import numpy as np


# ============================================================================
# Gamma and Special Functions (Project 052_asa245)
# ============================================================================

def lngamma(z):
    """
    Compute log(Gamma(z)) using Lanczos approximation.
    
    For z > 0:
    log Gamma(z) ≈ (z-0.5)*log(z+6.5) - (z+6.5) + log(C(z))
    where C(z) = sqrt(2*pi) * [a0 + sum_j a_j / (z + j)]
    
    Coefficients from Lanczos (1964) with g=7, n=9.
    
    Parameters
    ----------
    z : float or ndarray
        Argument (must be positive).
    
    Returns
    -------
    value : float or ndarray
        log(Gamma(z)).
    ier : int
        0 if success, 1 if z <= 0.
    """
    z = np.asarray(z, dtype=float)
    
    a = np.array([
        0.9999999999995183,
        676.5203681218835,
        -1259.139216722289,
        771.3234287757674,
        -176.6150291498386,
        12.50734324009056,
        -0.1385710331296526,
        0.9934937113930748e-05,
        0.1659470187408462e-06
    ])
    lnsqrt2pi = 0.9189385332046727
    
    if np.any(z <= 0):
        return np.full_like(z, 0.0), 1
    
    ier = 0
    value = np.zeros_like(z)
    tmp = z + 7.0
    
    for j in range(8, 0, -1):
        value = value + a[j] / tmp
        tmp = tmp - 1.0
    
    value = value + a[0]
    value = np.log(value) + lnsqrt2pi - (z + 6.5) + (z - 0.5) * np.log(z + 6.5)
    
    if np.isscalar(z):
        return float(value), ier
    return value, ier


def gamma_function(z):
    """
    Compute Gamma(z) = exp(log Gamma(z)).
    
    Parameters
    ----------
    z : float or ndarray
        Positive argument.
    
    Returns
    -------
    value : float or ndarray
    """
    lg, ier = lngamma(z)
    if ier != 0:
        return np.full_like(np.asarray(z), np.nan)
    return np.exp(lg)


def chi2_pdf(x, k):
    """
    Chi-square probability density function.
    
    f(x; k) = x^{k/2 - 1} * exp(-x/2) / (2^{k/2} * Gamma(k/2))
    
    Parameters
    ----------
    x : float or ndarray
    k : int
        Degrees of freedom.
    
    Returns
    -------
    pdf : float or ndarray
    """
    x_arr = np.asarray(x, dtype=float)
    x_safe = np.maximum(x_arr, 0.0)
    half_k = k / 2.0
    lg, ier = lngamma(half_k)
    if ier != 0:
        return np.zeros_like(x_safe)
    denom = (2.0 ** half_k) * np.exp(lg)
    pdf = (x_safe ** (half_k - 1.0)) * np.exp(-x_safe / 2.0) / denom
    if isinstance(x, np.ndarray):
        pdf[x_safe == 0] = 0.0
    elif x_safe == 0:
        pdf = 0.0
    return pdf


def rayleigh_pdf(x, sigma):
    """
    Rayleigh PDF for amplitude of narrow-band Gaussian process.
    
    f(x; sigma) = (x / sigma^2) * exp(-x^2 / (2*sigma^2))
    
    Parameters
    ----------
    x : float or ndarray
    sigma : float
        Scale parameter.
    
    Returns
    -------
    pdf : float or ndarray
    """
    x = np.asarray(x, dtype=float)
    x = np.maximum(x, 0.0)
    return (x / (sigma ** 2)) * np.exp(-x ** 2 / (2.0 * sigma ** 2))


def normal_cdf(x):
    """
    Standard normal cumulative distribution function.
    
    Phi(x) = 0.5 * [1 + erf(x / sqrt(2))]
    
    Parameters
    ----------
    x : float or ndarray
    
    Returns
    -------
    cdf : float or ndarray
    """
    from math import erf
    x_arr = np.asarray(x, dtype=float)
    
    # Vectorized erf
    try:
        result = 0.5 * (1.0 + np.vectorize(erf)(x_arr / np.sqrt(2.0)))
    except Exception:
        # Manual approximation for large values
        result = np.zeros_like(x_arr)
        for i, val in enumerate(x_arr.flat):
            try:
                result.flat[i] = 0.5 * (1.0 + erf(val / np.sqrt(2.0)))
            except OverflowError:
                result.flat[i] = 1.0 if val > 0 else 0.0
    return result


# ============================================================================
# Golden Section Optimization (Project 834_opt_golden)
# ============================================================================

def golden_section_search(f, a, b, n_iter=50, x_tol=1e-8):
    """
    Golden section search for minimizer of unimodal function f on [a, b].
    
    Golden ratio: phi = (sqrt(5) - 1) / 2 ≈ 0.618
    
    Algorithm:
    1. Set x1 = b - phi*(b-a), x2 = a + phi*(b-a)
    2. Evaluate f1 = f(x1), f2 = f(x2)
    3. If f1 < f2: b = x2, x2 = x1, f2 = f1, compute new x1
       Else:       a = x1, x1 = x2, f1 = f2, compute new x2
    4. Repeat until |b-a| < tol
    
    Parameters
    ----------
    f : callable
        Unimodal function.
    a, b : float
        Initial bracket.
    n_iter : int
        Maximum iterations.
    x_tol : float
        Interval tolerance.
    
    Returns
    -------
    x_min : float
        Estimated minimizer.
    f_min : float
        Function value at minimum.
    it : int
        Number of iterations performed.
    """
    phi = (np.sqrt(5.0) - 1.0) / 2.0  # ~0.618
    
    x1 = b - phi * (b - a)
    x2 = a + phi * (b - a)
    f1 = f(x1)
    f2 = f(x2)
    
    for it in range(1, n_iter + 1):
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = b - phi * (b - a)
            f1 = f(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + phi * (b - a)
            f2 = f(x2)
        
        if abs(b - a) < x_tol:
            break
    
    x_min = (a + b) / 2.0
    f_min = f(x_min)
    return x_min, f_min, it


def line_search_armijo(f, df, x0, direction, alpha_init=1.0, c=1e-4, rho=0.5):
    """
    Backtracking line search satisfying Armijo condition.
    
    f(x + alpha*d) <= f(x) + c*alpha*df(x)^T*d
    
    Parameters
    ----------
    f : callable
    df : callable
        Gradient.
    x0 : ndarray
    direction : ndarray
    alpha_init : float
    c : float
    rho : float
    
    Returns
    -------
    alpha : float
    """
    alpha = alpha_init
    f0 = f(x0)
    df0 = df(x0)
    slope = np.dot(df0, direction)
    
    for _ in range(20):
        if f(x0 + alpha * direction) <= f0 + c * alpha * slope:
            return alpha
        alpha *= rho
    
    return alpha


# ============================================================================
# FORM/SORM Reliability Methods
# ============================================================================

def form_reliability(g_func, dg_du, u0=None, dim=2, max_iter=50, tol=1e-6):
    """
    First-Order Reliability Method (FORM).
    
    Iteratively find the design point u* minimizing ||u|| subject to g(u)=0.
    
    HL-RF (Hasofer-Lind Rackwitz-Fiessler) algorithm:
    1. Initialize u
    2. Compute g(u) and grad_g(u)
    3. Update: u_{k+1} = [grad_g^T u_k - g(u_k)] / ||grad_g||^2 * grad_g
    4. Converge when ||u_{k+1} - u_k|| < tol
    
    Parameters
    ----------
    g_func : callable
        Limit state function g(u).
    dg_du : callable
        Gradient of g with respect to u.
    u0 : ndarray, optional
        Initial guess.
    dim : int
        Dimension of standard normal space.
    max_iter : int
    tol : float
    
    Returns
    -------
    beta : float
        Reliability index.
    u_star : ndarray
        Design point.
    P_f_form : float
        FORM failure probability.
    """
    if u0 is None:
        u = np.zeros(dim)
    else:
        u = u0.copy()
    
    for it in range(max_iter):
        g_val = g_func(u)
        grad_g = dg_du(u)
        grad_norm_sq = np.dot(grad_g, grad_g)
        
        if grad_norm_sq < 1e-20:
            break
        
        # HL-RF update
        alpha = -grad_g / np.sqrt(grad_norm_sq)
        beta_k = (np.dot(grad_g, u) - g_val) / np.sqrt(grad_norm_sq)
        u_new = beta_k * alpha
        
        if np.linalg.norm(u_new - u) < tol:
            u = u_new
            break
        
        u = u_new
    
    beta = np.linalg.norm(u)
    u_star = u
    P_f_form = 1.0 - normal_cdf(beta)
    
    return beta, u_star, float(P_f_form)


def form_with_golden_search(g_func, dim=2, beta_max=5.0, n_directions=36):
    """
    Alternative FORM using golden section search along radial directions.
    
    For each direction theta, find beta such that g(beta*theta) = 0
    using golden section search on |g(beta*theta)|.
    
    Parameters
    ----------
    g_func : callable
    dim : int
    beta_max : float
    n_directions : int
        Number of angular directions to search.
    
    Returns
    -------
    beta : float
    u_star : ndarray
    P_f : float
    """
    best_beta = beta_max
    best_u = None
    
    # Search over angular directions in standard normal space
    if dim == 2:
        angles = np.linspace(0, 2 * np.pi, n_directions, endpoint=False)
        for theta in angles:
            direction = np.array([np.cos(theta), np.sin(theta)])
            
            def g_along_beta(b):
                return abs(g_func(b * direction))
            
            try:
                beta_opt, g_opt, _ = golden_section_search(
                    g_along_beta, 0.0, beta_max, n_iter=40, x_tol=1e-6
                )
                if g_opt < 1e-3 and beta_opt < best_beta:
                    best_beta = beta_opt
                    best_u = beta_opt * direction
            except Exception:
                continue
    else:
        # For higher dimensions, use random directions
        np.random.seed(42)
        for _ in range(n_directions):
            direction = np.random.randn(dim)
            direction = direction / np.linalg.norm(direction)
            
            def g_along_beta(b):
                return abs(g_func(b * direction))
            
            try:
                beta_opt, g_opt, _ = golden_section_search(
                    g_along_beta, 0.0, beta_max, n_iter=40, x_tol=1e-6
                )
                if g_opt < 1e-3 and beta_opt < best_beta:
                    best_beta = beta_opt
                    best_u = beta_opt * direction
            except Exception:
                continue
    
    if best_u is None:
        best_u = np.zeros(dim)
        best_beta = 0.0
    
    P_f = 1.0 - normal_cdf(best_beta)
    return best_beta, best_u, float(P_f)


def sorm_reliability(g_func, dg_du, d2g_du2, u_star, beta):
    """
    Second-Order Reliability Method (SORM) correction.
    
    Breitung's formula:
    P_f_sorm ≈ Phi(-beta) * prod_{i=1}^{n-1} (1 + beta * kappa_i)^{-1/2}
    
    where kappa_i are principal curvatures of the failure surface at u*.
    
    Parameters
    ----------
    g_func : callable
    dg_du : callable
    d2g_du2 : callable
        Hessian matrix of g.
    u_star : ndarray
        Design point.
    beta : float
        Reliability index.
    
    Returns
    -------
    P_f_sorm : float
    curvatures : ndarray
        Principal curvatures.
    """
    grad = dg_du(u_star)
    hess = d2g_du2(u_star)
    grad_norm = np.linalg.norm(grad)
    
    if grad_norm < 1e-14 or beta < 1e-10:
        return 1.0 - normal_cdf(beta), np.array([])
    
    # Normalized gradient (outward normal)
    n_vec = grad / grad_norm
    
    # Projection matrix onto tangent plane
    dim = len(u_star)
    P = np.eye(dim) - np.outer(n_vec, n_vec)
    
    # Scaled Hessian on tangent plane
    H_tilde = P @ hess @ P / grad_norm
    
    # Eigenvalues of H_tilde restricted to tangent plane
    # (one eigenvalue will be ~0 corresponding to normal direction)
    eigvals = np.linalg.eigvalsh(H_tilde)
    eigvals = np.sort(eigvals)
    
    # Principal curvatures (exclude the zero associated with normal)
    curvatures = eigvals[1:] if len(eigvals) > 1 else eigvals
    
    # Breitung correction
    correction = 1.0
    for kappa in curvatures:
        factor = 1.0 + beta * kappa
        if factor > 1e-10:
            correction /= np.sqrt(factor)
    
    P_f_sorm = (1.0 - normal_cdf(beta)) * correction
    P_f_sorm = np.clip(P_f_sorm, 0.0, 1.0)
    
    return float(P_f_sorm), curvatures
