"""
utils.py

Mathematical utility functions including orthogonal polynomials,
quadrature rules, special functions, and interpolation kernels.

This module synthesizes algorithms from:
    - 081_besselzero: Bessel function zeros via Halley's method
    - 641_laguerre_polynomial: Laguerre orthogonal polynomials
    - 463_gegenbauer_rule: Gegenbauer quadrature construction
    - 466_gen_laguerre_exactness: Generalized Laguerre integrals
    - 638_lagrange_nd: N-dimensional Lagrange interpolation
    - 1373_uniform: Pseudorandom number generators
"""

import numpy as np
from scipy.special import gamma, factorial, spherical_jn, eval_legendre
from scipy.special import roots_legendre, roots_laguerre, roots_jacobi


# =============================================================================
# Bessel Function Zeros (from 081_besselzero)
# =============================================================================

def bessel_zero_j(n, k, kind=1, tol=1e-12, max_iter=100):
    """
    Compute the k-th positive zero of the Bessel function J_n(x) or Y_n(x)
    using Halley's method with least-squares initial guesses.
    
    The Bessel function J_n(x) satisfies:
        x^2 * y'' + x * y' + (x^2 - n^2) * y = 0
    
    Halley's iteration:
        x_{new} = x - f(x) / f'(x) * [ 1 - f(x)*f''(x) / (2*f'(x)^2) ]^{-1}
    
    For Bessel functions:
        J_n'(x) = (n/x) * J_n(x) - J_{n+1}(x)
        J_n''(x) = (n^2/x^2 - 1) * J_n(x) - (1/x) * J_{n+1}(x)
    
    Args:
        n: order (can be float)
        k: index of zero (1, 2, 3, ...)
        kind: 1 for J_n, 2 for Y_n
        tol: convergence tolerance
        max_iter: maximum iterations
        
    Returns:
        float: the k-th positive zero
    """
    from scipy.special import jv, yv
    
    n = abs(n)
    
    # Initial guess using asymptotic formulas
    if k == 1:
        # Empirical fit for first zero
        if kind == 1:
            x0 = 0.4116 + 0.99999 * n + 0.6980 * (n + 1) ** 0.3353 + 1.0698 * (n + 1) ** 0.3397
        else:
            x0 = 0.0795 + 0.999998 * n + 0.8904 * (n + 1) ** 0.3354 + 0.0271 * (n + 1) ** 0.3087
    elif k == 2:
        if kind == 1:
            x0 = 1.93395 + 1.00008 * n - 0.80572 * (n + 1) ** 0.4562 + 3.38765 * (n + 1) ** 0.3884
        else:
            x0 = 1.04503 + 1.00002 * n - 0.43792 * (n + 1) ** 0.4348 + 2.70113 * (n + 1) ** 0.3662
    elif k == 3:
        if kind == 1:
            x0 = 5.40771 + 1.00094 * n + 2.66926 * (n + 1) ** 0.4297 - 0.17493 * (n + 1) ** 0.6335
        else:
            x0 = 3.72778 + 1.00035 * n + 2.68567 * (n + 1) ** 0.3982 - 0.11298 * (n + 1) ** 0.6048
    else:
        # For k >= 4, use uniform spacing approximation
        # First compute zeros 2 and 3
        x2 = bessel_zero_j(n, 2, kind, tol, max_iter)
        x3 = bessel_zero_j(n, 3, kind, tol, max_iter)
        spacing = x3 - x2
        x0 = x3 + (k - 3) * spacing
    
    # Halley's method
    x = x0
    for _ in range(max_iter):
        if kind == 1:
            fx = jv(n, x)
            fpx = jv(n - 1, x) - n / x * jv(n, x) if x > 1e-10 else 0.0
            # Simplified derivative using recurrence
            if x > 1e-10:
                fpx = 0.5 * (jv(n - 1, x) - jv(n + 1, x))
            else:
                fpx = 0.0
        else:
            fx = yv(n, x)
            if x > 1e-10:
                fpx = 0.5 * (yv(n - 1, x) - yv(n + 1, x))
            else:
                fpx = 0.0
        
        if abs(fpx) < 1e-30:
            break
            
        dx = fx / fpx
        x_new = x - dx
        
        if abs(dx) < tol * max(abs(x), 1.0):
            return x_new
        x = x_new
    
    return x


# =============================================================================
# Laguerre Polynomials (from 641_laguerre_polynomial)
# =============================================================================

def laguerre_polynomial(m, n_max, x):
    """
    Evaluate Laguerre polynomials L_n(x) for n = 0, 1, ..., n_max.
    
    The Laguerre polynomials satisfy the orthogonality relation:
        integral_0^inf exp(-x) * L_n(x) * L_m(x) dx = delta_{nm}
    
    Three-term recurrence:
        L_0(x) = 1
        L_1(x) = 1 - x
        n * L_n(x) = (2n - 1 - x) * L_{n-1}(x) - (n - 1) * L_{n-2}(x)
    
    Rodrigues formula:
        L_n(x) = (1/n!) * exp(x) * (d/dx)^n [ x^n * exp(-x) ]
              = sum_{k=0}^n (-1)^k * C(n,k) * x^k / k!
    
    Args:
        m: number of evaluation points
        n_max: maximum polynomial degree
        x: array of shape (m,) or (m,1) with evaluation points
        
    Returns:
        v: array of shape (m, n_max+1) with v[:,j] = L_j(x)
    """
    x = np.asarray(x).reshape(-1)
    m = len(x)
    
    if n_max < 0:
        return np.zeros((m, 0))
    
    v = np.zeros((m, n_max + 1))
    v[:, 0] = 1.0
    
    if n_max == 0:
        return v
    
    v[:, 1] = 1.0 - x
    
    for j in range(2, n_max + 1):
        v[:, j] = ((2.0 * j - 1.0 - x) * v[:, j - 1] - (j - 1.0) * v[:, j - 2]) / j
    
    return v


def generalized_laguerre_integral(expon, alpha):
    """
    Evaluate the generalized Laguerre integral:
        I = integral_0^inf x^n * x^alpha * exp(-x) dx
          = Gamma(alpha + n + 1)
    
    This is the exact value used for quadrature exactness testing.
    
    Args:
        expon: exponent n (non-negative integer)
        alpha: weight exponent (alpha > -1)
        
    Returns:
        float: exact integral value
    """
    if alpha <= -1.0:
        raise ValueError("alpha must be > -1 for generalized Laguerre")
    return gamma(alpha + expon + 1)


# =============================================================================
# Gegenbauer (Jacobi) Quadrature (from 463_gegenbauer_rule)
# =============================================================================

def gegenbauer_rule(order, alpha, a, b):
    """
    Generate a Gauss-Gegenbauer quadrature rule for:
        integral_a^b [(x-a)(b-x)]^alpha f(x) dx
        approx = sum_i w_i * f(x_i)
    
    The Gegenbauer polynomials C_n^{(lambda)}(x) are Jacobi polynomials
    with parameters (lambda - 1/2, lambda - 1/2).
    Here lambda = alpha + 1/2.
    
    The quadrature nodes are the zeros of C_n^{(lambda)}(x).
    
    Args:
        order: number of quadrature points
        alpha: exponent parameter (alpha > -1)
        a, b: interval endpoints
        
    Returns:
        x, w: nodes and weights
    """
    if alpha <= -1.0:
        raise ValueError("alpha must be > -1")
    if a >= b:
        raise ValueError("require a < b")
    if order < 1:
        raise ValueError("order must be >= 1")
    
    # Map to standard interval [-1, 1]
    # The weight is (1-x^2)^alpha on [-1, 1]
    # Corresponding Jacobi parameters: beta = alpha, alpha_param = alpha
    jacobi_alpha = alpha
    jacobi_beta = alpha
    
    # Get Jacobi roots and weights on [-1, 1]
    xi, wi = roots_jacobi(order, jacobi_alpha, jacobi_beta)
    
    # Transform to [a, b]
    x = 0.5 * (b - a) * xi + 0.5 * (a + b)
    w = wi * ((b - a) / 2.0) ** (2.0 * alpha + 1.0)
    
    return x, w


# =============================================================================
# N-Dimensional Lagrange Interpolation (from 638_lagrange_nd)
# =============================================================================

def lagrange_basis_1d(x_nodes, x_eval):
    """
    Compute 1D Lagrange basis functions:
        l_i(x) = prod_{j != i} (x - x_j) / (x_i - x_j)
    
    Args:
        x_nodes: 1D array of n nodes (must be distinct)
        x_eval: scalar or array of evaluation points
        
    Returns:
        L: array of shape (len(x_eval), n) with basis values
    """
    x_nodes = np.asarray(x_nodes)
    x_eval = np.atleast_1d(x_eval)
    n = len(x_nodes)
    
    if len(np.unique(x_nodes)) < n:
        raise ValueError("Nodes must be distinct")
    
    L = np.ones((len(x_eval), n))
    for i in range(n):
        for j in range(n):
            if i != j:
                denom = x_nodes[i] - x_nodes[j]
                if abs(denom) < 1e-14:
                    denom = 1e-14  # numerical safeguard
                L[:, i] *= (x_eval - x_nodes[j]) / denom
    
    return L


def lagrange_interp_nd(grid_nodes, values, x_eval):
    """
    N-dimensional tensor-product Lagrange interpolation.
    
    Given function values on a tensor-product grid:
        f_{i1,i2,...,id} = f(x_{1,i1}, x_{2,i2}, ..., x_{d,id})
    
    The interpolant is:
        P(x) = sum_{i1} ... sum_{id} f_{i1,...,id} 
               * l_{i1}^{(1)}(x_1) * ... * l_{id}^{(d)}(x_d)
    
    Args:
        grid_nodes: list of d arrays, each with 1D nodes
        values: array of shape (n1, n2, ..., nd) with function values
        x_eval: array of shape (d,) or (m, d) with evaluation points
        
    Returns:
        interpolated values
    """
    d = len(grid_nodes)
    values = np.asarray(values)
    x_eval = np.atleast_2d(x_eval)
    m = x_eval.shape[0]
    
    if x_eval.shape[1] != d:
        raise ValueError("x_eval must have d columns")
    
    # Compute 1D basis functions for each dimension
    basis_1d = []
    for dim in range(d):
        L = lagrange_basis_1d(grid_nodes[dim], x_eval[:, dim])
        basis_1d.append(L)
    
    # Tensor product summation
    result = np.zeros(m)
    
    # For efficiency with small dimensions, use explicit loops
    # For larger dimensions, one would use einsum
    if d == 1:
        result = basis_1d[0] @ values.flatten()
    elif d == 2:
        n1, n2 = values.shape
        for i in range(n1):
            for j in range(n2):
                result += values[i, j] * basis_1d[0][:, i] * basis_1d[1][:, j]
    elif d == 3:
        n1, n2, n3 = values.shape
        for i in range(n1):
            for j in range(n2):
                for k in range(n3):
                    result += values[i, j, k] * basis_1d[0][:, i] * basis_1d[1][:, j] * basis_1d[2][:, k]
    else:
        # Fallback: use flattened indexing
        flat_vals = values.flatten()
        indices = np.array(np.unravel_index(np.arange(len(flat_vals)), values.shape)).T
        for idx, val in enumerate(flat_vals):
            contrib = np.ones(m)
            for dim in range(d):
                contrib *= basis_1d[dim][:, indices[idx, dim]]
            result += val * contrib
    
    return result


# =============================================================================
# Random Number Generators (from 1373_uniform)
# =============================================================================

def r8vec_uniform_01(n, rng=None):
    """
    Generate n uniform random numbers in [0, 1).
    
    Args:
        n: number of samples
        rng: numpy random Generator (optional)
        
    Returns:
        array of shape (n,)
    """
    if rng is None:
        rng = np.random.default_rng()
    return rng.random(n)


def r8vec_normal_01(n, rng=None):
    """
    Generate n standard normal random numbers N(0, 1).
    
    The probability density function is:
        phi(x) = (1/sqrt(2*pi)) * exp(-x^2/2)
    
    Args:
        n: number of samples
        rng: numpy random Generator (optional)
        
    Returns:
        array of shape (n,)
    """
    if rng is None:
        rng = np.random.default_rng()
    return rng.standard_normal(n)


# =============================================================================
# Sparse Grid Helpers (from 1104_sparse_grid_gl)
# =============================================================================

def comp_next(n, k, a, more, h, t):
    """
    Compute the next composition of N into K parts.
    
    A composition of N into K parts is a sequence (a_1, ..., a_K) of
    non-negative integers such that a_1 + ... + a_K = N.
    
    Implementation based on Nijenhuis-Wilf algorithm.
    
    Args:
        n, k: composition parameters
        a: current composition (array-like)
        more, h, t: state variables
        
    Returns:
        a, more, h, t: updated values
    """
    a = np.asarray(a, dtype=int)
    
    if not more:
        a[:] = 0
        a[0] = n
        more = True
        h = 0
        t = n
        return a, more, h, t
    
    if 1 < t:
        h = 0
    
    h = h + 1
    t = a[h - 1]
    a[h - 1] = 0
    a[0] = t - 1
    a[h] = a[h] + 1
    
    more = (a[k - 1] != n)
    
    return a, more, h, t


def level_to_order_open(dim_num, level_1d):
    """
    Convert 1D levels to orders for open Gauss-Legendre rules.
    
    For sparse grids, the order sequence is: 1, 3, 7, 15, 31, 63, 127, ...
    which corresponds to levels 0, 1, 2, 3, 4, 5, 6, ...
    
    Args:
        dim_num: spatial dimension
        level_1d: array of levels for each dimension
        
    Returns:
        order_1d: array of orders
    """
    level_1d = np.asarray(level_1d, dtype=int)
    order_1d = np.zeros(dim_num, dtype=int)
    
    for i in range(dim_num):
        if level_1d[i] < 0:
            order_1d[i] = 1
        else:
            order_1d[i] = 2 ** (level_1d[i] + 1) - 1
    
    return order_1d
