"""
polynomial_chaos.py -- Polynomial Chaos Expansion with Gegenbauer Basis
========================================================================
Implements generalized Polynomial Chaos (gPC) using Gegenbauer (ultraspherical)
polynomials as the orthogonal basis for the stochastic expansion.

The Gegenbauer polynomials C_n^{(alpha)}(x) satisfy the orthogonality:
    integral_{-1}^{1} C_m^{(alpha)}(x) C_n^{(alpha)}(x) (1-x^2)^{alpha-1/2} dx
        = h_n delta_{mn}
where h_n = pi * 2^{1-2alpha} * Gamma(n+2alpha) / (n! * (n+alpha) * [Gamma(alpha)]^2)

Special cases:
    alpha = 0.5  -> Legendre polynomials (uniform weight)
    alpha = 1.0  -> Chebyshev polynomials of the second kind

Seed references:
  - 462_gegenbauer_polynomial: recurrence, Gauss-Gegenbauer quadrature,
    hypergeometric function, basis conversion matrices
  - 804_nint_exactness_mixed: exactness testing framework

Scientific context:
  In stochastic Galerkin methods, the uncertain solution u(x,t,xi) is expanded as:
      u(x,t,xi) = sum_{k=0}^{P} u_k(x,t) Phi_k(xi)
  where Phi_k are multivariate Gegenbauer polynomials and u_k are deterministic
  coefficients to be solved for. The triple-product tensor
      C_{ijk} = E[Phi_i Phi_j Phi_k]
  appears in the stochastic Galerkin projection of nonlinear terms.
"""
import numpy as np
from scipy.special import gamma as gamma_func
from typing import Tuple


# ---------------------------------------------------------------------------
# Gegenbauer polynomial evaluation via three-term recurrence
# ---------------------------------------------------------------------------
def gegenbauer_value(max_order: int, alpha: float, x: np.ndarray) -> np.ndarray:
    """
    Evaluate Gegenbauer polynomials C_0^{(alpha)}, ..., C_{max_order}^{(alpha)}
    at points x using the three-term recurrence:

        C_0^{(alpha)}(x) = 1
        C_1^{(alpha)}(x) = 2*alpha*x
        n * C_n^{(alpha)}(x) = 2(n+alpha-1)*x*C_{n-1}^{(alpha)}(x)
                                - (n+2*alpha-2)*C_{n-2}^{(alpha)}(x)

    Parameters
    ----------
    max_order : int
        Maximum polynomial order (>= 0).
    alpha : float
        Gegenbauer parameter (must be > -0.5, alpha != 0).
    x : ndarray
        Evaluation points.

    Returns
    -------
    C : ndarray, shape (max_order+1, len(x))
        C[k, j] = C_k^{(alpha)}(x[j]).
    """
    if max_order < 0:
        raise ValueError("max_order must be >= 0")
    if alpha <= -0.5 or abs(alpha) < 1e-14:
        raise ValueError("alpha must be > -0.5 and != 0")
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    n_pts = len(x)
    C = np.zeros((max_order + 1, n_pts), dtype=np.float64)
    C[0, :] = 1.0
    if max_order >= 1:
        C[1, :] = 2.0 * alpha * x
    for n in range(2, max_order + 1):
        a_n = 2.0 * (n + alpha - 1.0) / n
        b_n = (n + 2.0 * alpha - 2.0) / n
        C[n, :] = a_n * x * C[n - 1, :] - b_n * C[n - 2, :]
    return C


def gegenbauer_single(n: int, alpha: float, x: np.ndarray) -> np.ndarray:
    """Evaluate only C_n^{(alpha)}(x) without storing all lower orders."""
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2.0 * alpha * x
    c_prev2 = np.ones_like(x)
    c_prev1 = 2.0 * alpha * x
    for k in range(2, n + 1):
        a_k = 2.0 * (k + alpha - 1.0) / k
        b_k = (k + 2.0 * alpha - 2.0) / k
        c_curr = a_k * x * c_prev1 - b_k * c_prev2
        c_prev2 = c_prev1
        c_prev1 = c_curr
    return c_prev1


# ---------------------------------------------------------------------------
# Derivative of Gegenbauer polynomials
# ---------------------------------------------------------------------------
def gegenbauer_derivative(n: int, alpha: float, x: np.ndarray) -> np.ndarray:
    """
    Compute d/dx C_n^{(alpha)}(x) using the identity:
        d/dx C_n^{(alpha)}(x) = 2*alpha * C_{n-1}^{(alpha+1)}(x)
    for n >= 1, and 0 for n = 0.
    """
    if n == 0:
        return np.zeros_like(np.atleast_1d(x))
    return 2.0 * alpha * gegenbauer_single(n - 1, alpha + 1.0, x)


# ---------------------------------------------------------------------------
# Gauss-Gegenbauer quadrature via Newton iteration on roots
# ---------------------------------------------------------------------------
def gauss_gegenbauer(n: int, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Gauss-Gegenbauer quadrature nodes and weights using the
    Stroud-Secrest method (Newton iteration on roots of C_n^{(alpha)}).

    The quadrature rule satisfies:
        integral_{-1}^{1} f(x) (1-x^2)^{alpha-1/2} dx
            ~= sum_{i=1}^{n} w_i f(x_i)
    and is exact for polynomials of degree <= 2n - 1.

    Parameters
    ----------
    n : int
        Number of quadrature points (>= 1).
    alpha : float
        Gegenbauer parameter (> -0.5, != 0).

    Returns
    -------
    nodes : ndarray, shape (n,)
        Quadrature abscissas in (-1, 1), sorted ascending.
    weights : ndarray, shape (n,)
        Corresponding quadrature weights (positive).
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if alpha <= -0.5 or abs(alpha) < 1e-14:
        raise ValueError("alpha must be > -0.5 and != 0")

    if n == 1:
        return np.array([0.0]), np.array([_gegenbauer_weight_total(alpha)])

    # Initial guesses from asymptotic formula (Abramowitz & Stegun 22.16.10)
    nodes = np.zeros(n)
    for i in range(n):
        # Chebyshev-type initial guess refined by Gegenbauer correction
        theta = np.pi * (4 * (i + 1) - 1) / (4 * n + 2 * alpha + 1)
        nodes[i] = np.cos(theta + (2 * alpha - 1) * np.cos(theta) /
                          (2 * (n + alpha) ** 2))

    # Newton iteration to refine roots
    for _ in range(100):
        C_vals = gegenbauer_value(n, alpha, nodes)
        Cn = C_vals[n, np.arange(n)]
        Cn_minus1 = C_vals[n - 1, np.arange(n)]
        # Derivative: C_n'(x) = [n*C_{n-1}*x*(2*alpha+n-1) - (n+2*alpha-1)*C_{n-1}] / (x^2-1)
        # Simplified via: C_n'(x_i) at root x_i:
        #   C_n'(x_i) = 2*alpha * C_{n-1}^{(alpha+1)}(x_i)
        Cn_prime = 2.0 * alpha * gegenbauer_single(n - 1, alpha + 1.0, nodes)
        # Safeguard against zero derivative
        Cn_prime = np.where(np.abs(Cn_prime) < 1e-30, 1e-30, Cn_prime)
        delta = Cn / Cn_prime
        nodes -= delta
        # Keep nodes in (-1, 1)
        nodes = np.clip(nodes, -1.0 + 1e-14, 1.0 - 1e-14)
        if np.max(np.abs(delta)) < 1e-15:
            break

    nodes = np.sort(nodes)

    # Compute weights: w_i = integral / (C_n'(x_i) * C_{n-1}^{(alpha)}(x_i))
    # Using the formula: w_i = 2^{1-2alpha} * pi * Gamma(n+2alpha) /
    #                               (n! * Gamma(alpha)^2) * 1/(C_n'(x_i)*C_{n-1}(x_i))
    # Simplified: use the Christoffel-Darboux formula
    C_vals_final = gegenbauer_value(n, alpha, nodes)
    Cn_prime_final = 2.0 * alpha * gegenbauer_single(n - 1, alpha + 1.0, nodes)
    Cn_minus1_final = C_vals_final[n - 1, np.arange(n)]

    # Weight formula from Golub-Welsch: w_i = mu_0 / (p_n'(x_i) * p_{n-1}(x_i))
    # where mu_0 = integral_{-1}^{1} (1-x^2)^{alpha-1/2} dx
    mu_0 = _gegenbauer_weight_total(alpha)
    weights = mu_0 / (Cn_prime_final * Cn_minus1_final * n)
    # Alternative: compute from the normalization
    # Ensure positivity
    weights = np.abs(weights)
    # Normalize weights to sum to mu_0
    weights *= mu_0 / np.sum(weights)

    return nodes, weights


def _gegenbauer_weight_total(alpha: float) -> float:
    """
    Total weight integral: mu_0 = integral_{-1}^{1} (1-x^2)^{alpha-1/2} dx
        = sqrt(pi) * Gamma(alpha + 0.5) / Gamma(alpha + 1)
    """
    return np.sqrt(np.pi) * gamma_func(alpha + 0.5) / gamma_func(alpha + 1.0)


# ---------------------------------------------------------------------------
# Jacobi matrix for Golub-Welsch eigenvalue approach
# ---------------------------------------------------------------------------
def gegenbauer_jacobi_matrix(n: int, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct the symmetric tridiagonal Jacobi matrix J for Gegenbauer
    polynomials. The eigenvalues of J are the Gauss-Gegenbauer nodes,
    and the weights are derived from the first components of eigenvectors.

    The recurrence x*C_n = a_n*C_{n+1} + b_n*C_n + c_n*C_{n-1} gives:
        a_n = n / (2*(n+alpha))
        b_n = 0  (symmetric weight)
        c_n = (n+2*alpha-1) / (2*(n+alpha-1))

    The Jacobi matrix has:
        diagonal:     d_i = b_i = 0
        subdiagonal:  e_i = sqrt(c_{i+1} / a_i) = sqrt((i+1)*(i+2*alpha) / (4*(i+alpha)*(i+alpha+1)))

    Parameters
    ----------
    n : int
        Size of the Jacobi matrix.
    alpha : float
        Gegenbauer parameter.

    Returns
    -------
    d : ndarray, shape (n,)
        Diagonal entries (all zero for symmetric weight).
    e : ndarray, shape (n-1,)
        Subdiagonal entries.
    """
    d = np.zeros(n)
    e = np.zeros(max(n - 1, 0))
    for i in range(n - 1):
        k = i + 1
        e[i] = np.sqrt(k * (k + 2.0 * alpha) /
                       (4.0 * (k + alpha) * (k + alpha - 1.0) + 1e-30))
    return d, e


# ---------------------------------------------------------------------------
# Triple-product tensor for stochastic Galerkin
# ---------------------------------------------------------------------------
def gegenbauer_triple_product(p: int, alpha: float) -> np.ndarray:
    """
    Compute the triple-product tensor C_{ijk} for i,j,k in {0,...,p}:
        C_{ijk} = integral_{-1}^{1} C_i C_j C_k (1-x^2)^{alpha-1/2} dx
                  / h_k
    where h_k = integral C_k^2 (1-x^2)^{alpha-1/2} dx.

    This tensor appears in the stochastic Galerkin formulation when
    projecting products of uncertain quantities:
        E[Phi_i Phi_j Phi_k] = C_{ijk}

    Uses Gauss-Gegenbauer quadrature with sufficient points for exactness.

    Parameters
    ----------
    p : int
        Maximum polynomial order.
    alpha : float
        Gegenbauer parameter.

    Returns
    -------
    C : ndarray, shape (p+1, p+1, p+1)
        Triple-product tensor.
    """
    # Need quadrature exact for degree 3p, so use ceil(3p/2)+1 points
    nq = max(int(np.ceil(1.5 * p)) + 2, p + 2)
    nodes, weights = gauss_gegenbauer(nq, alpha)
    w = (1.0 - nodes ** 2) ** (alpha - 0.5)
    C_vals = gegenbauer_value(p, alpha, nodes)  # shape (p+1, nq)

    # Normalization: h_k = integral C_k^2 w(x) dx
    h = np.zeros(p + 1)
    for k in range(p + 1):
        h[k] = np.sum(weights * w * C_vals[k, :] ** 2)
    h = np.where(np.abs(h) < 1e-30, 1e-30, h)

    tensor = np.zeros((p + 1, p + 1, p + 1))
    for i in range(p + 1):
        for j in range(p + 1):
            # Vectorized over k
            integrand = C_vals[i, :] * C_vals[j, :] * C_vals  # (p+1, nq)
            tensor[i, j, :] = (weights * w) @ integrand.T / h

    return tensor


# ---------------------------------------------------------------------------
# Multivariate tensor-product basis
# ---------------------------------------------------------------------------
def multi_index_set(dim: int, order: int) -> np.ndarray:
    """
    Generate the multi-index set for total-order polynomial chaos:
        {alpha in N_0^dim : |alpha| <= order}

    Parameters
    ----------
    dim : int
        Number of random dimensions.
    order : int
        Maximum total order.

    Returns
    -------
    indices : ndarray, shape (P, dim)
        Multi-indices where P = C(dim+order, order).
    """
    if dim == 1:
        return np.arange(order + 1).reshape(-1, 1)
    indices = []
    _generate_multi_indices(dim, order, [], indices)
    return np.array(indices, dtype=int)


def _generate_multi_indices(dim: int, remaining: int, current: list, result: list):
    """Recursive helper to enumerate multi-indices."""
    if dim == 1:
        result.append(current + [remaining])
        return
    for k in range(remaining + 1):
        _generate_multi_indices(dim - 1, remaining - k, current + [k], result)


def basis_size(dim: int, order: int) -> int:
    """Number of terms in total-order PC expansion: C(dim+order, order)."""
    from math import comb
    return comb(dim + order, order)


# ---------------------------------------------------------------------------
# Hypergeometric function 2F1 (for Gegenbauer integral computation)
# ---------------------------------------------------------------------------
def hypergeometric_2f1(a: float, b: float, c: float, x: float) -> complex:
    """
    Evaluate the Gauss hypergeometric function 2F1(a,b;c;x) via series:
        2F1(a,b;c;x) = sum_{n=0}^{inf} (a)_n (b)_n / (c)_n * x^n / n!
    where (q)_n = q(q+1)...(q+n-1) is the Pochhammer symbol.

    Converges for |x| < 1. Uses Euler transformation for x near 1:
        2F1(a,b;c;x) = (1-x)^{c-a-b} 2F1(c-a,c-b;c;x)
    """
    if abs(x) < 1e-15:
        return complex(1.0)
    if abs(x) >= 1.0:
        # Use analytic continuation for |x| >= 1
        if abs(x - 1.0) < 1e-10:
            # 2F1(a,b;c;1) = Gamma(c)*Gamma(c-a-b) / (Gamma(c-a)*Gamma(c-b))
            if c - a - b > 0:
                from scipy.special import gamma as G
                return complex(G(c) * G(c - a - b) / (G(c - a) * G(c - b)))
            else:
                return complex(np.inf)
        # Pfaff transformation: 2F1(a,b;c;x) = (1-x)^{-a} 2F1(a,c-b;c;x/(x-1))
        x_t = x / (x - 1.0)
        if abs(x_t) < 1.0:
            prefactor = (1.0 - x) ** (-a)
            return complex(prefactor) * hypergeometric_2f1(a, c - b, c, x_t)
        return complex(np.nan)

    # Direct series summation
    total = 1.0 + 0.0j
    term = 1.0 + 0.0j
    for n in range(1, 500):
        term *= (a + n - 1) * (b + n - 1) / ((c + n - 1) * n) * x
        total += term
        if abs(term) < 1e-15 * abs(total):
            break
    return total


# ---------------------------------------------------------------------------
# Gegenbauer monomial integral (for exactness testing)
# ---------------------------------------------------------------------------
def gegenbauer_monomial_integral(exponent: int, alpha: float) -> float:
    """
    Compute integral_{-1}^{1} x^n (1-x^2)^{alpha-1/2} dx analytically:
        = 0                           if n is odd
        = Gamma((n+1)/2) Gamma(alpha+1/2) / (Gamma((n+2)/2) Gamma(alpha+1))  ... (not quite)

    Actually: = B((n+1)/2, alpha+1/2) if n even, 0 if n odd
    where B is the Beta function.

    More precisely:
        integral_{-1}^{1} x^{2m} (1-x^2)^{alpha-1/2} dx
            = Gamma(m + 1/2) Gamma(alpha + 1/2) / Gamma(m + alpha + 1)
    """
    if exponent % 2 == 1:
        return 0.0
    m = exponent // 2
    return gamma_func(m + 0.5) * gamma_func(alpha + 0.5) / gamma_func(m + alpha + 1.0)
