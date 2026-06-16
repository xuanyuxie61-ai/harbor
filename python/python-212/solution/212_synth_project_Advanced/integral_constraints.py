"""
integral_constraints.py
=======================
Quadrature-based evaluation of integral inequality constraints in the KKT system.

The integral constraint has the form

    integral_Omega  u(x) dx  <=  E_max

On the discrete grid, this becomes

    sum_i  w_i u_i  <=  E_max

where w_i are quadrature weights.  We provide three quadrature families:

  1. Composite Newton-Cotes open rule    (from 685_line_nco_rule)
  2. Gauss-Hermite rule on (-inf, +inf)  (from 464_gen_hermite_exactness)
  3. Triangle quadrature on simplices    (from 1302_triangle_exactness)

KKT role
--------
The quadrature weight vector  c = (w_1, ..., w_{N^2})  appears as the
(1,4) and (4,1) blocks of the KKT saddle-point matrix, and the multiplier
lambda >= 0 enforces complementarity  lambda * (c^T u - E_max) = 0.
"""

from __future__ import annotations
import math
import numpy as np


# ---------------------------------------------------------------------------
# Newton-Cotes open rule  (from 685)
# ---------------------------------------------------------------------------

def line_nco_rule(n: int, a: float = 0.0, b: float = 1.0):
    """Compute a Newton-Cotes *open* quadrature rule on [a, b] of order n.

    The abscissas do NOT include the endpoints a and b.

    x_k = ((n - k) * a + (k + 1) * b) / (n + 1),   k = 0, ..., n-1

    Weights are obtained by integrating the Lagrange basis polynomials:

        w_k = integral_a^b  L_k(x) dx

    where  L_k(x) = prod_{j != k} (x - x_j) / (x_k - x_j).

    Returns
    -------
    x : (n,) abscissas
    w : (n,) weights
    """
    if n < 1:
        raise ValueError("line_nco_rule: n must be >= 1")

    x = np.zeros(n, dtype=np.float64)
    for k in range(n):
        x[k] = ((n - k) * a + (k + 1) * b) / (n + 1)

    w = np.zeros(n, dtype=np.float64)
    for k in range(n):
        # Build Lagrange basis polynomial which is 1 at x[k] and 0 at x[j], j != k.
        # L_k(x) = prod_{j != k} (x - x[j]) / (x[k] - x[j])
        # We integrate L_k analytically via the monomial representation.
        # Coefficients of L_k in the monomial basis:  c_m x^m
        denom = 1.0
        for j in range(n):
            if j != k:
                denom *= (x[k] - x[j])
        if abs(denom) < 1.0e-30:
            w[k] = 0.0
            continue

        # Build polynomial numerator:  prod_{j != k} (x - x[j])
        # Start with poly = [1.0]  (constant polynomial)
        poly = np.array([1.0], dtype=np.float64)
        for j in range(n):
            if j != k:
                # Multiply by (x - x[j])
                poly = np.convolve(poly, np.array([-x[j], 1.0]))

        # Integrate  poly from a to b
        integral = 0.0
        for m, c_m in enumerate(poly):
            # integral of c_m * x^m from a to b = c_m * (b^{m+1} - a^{m+1}) / (m+1)
            integral += c_m * (b ** (m + 1) - a ** (m + 1)) / (m + 1)
        w[k] = integral / denom

    return x, w


def composite_nco_2d(N: int, a: float = 0.0, b: float = 1.0):
    """Composite Newton-Cotes open rule on [a, b]^2 with N sub-intervals.

    Returns the 1-D weight vector of length N (for tensor product use).
    """
    x, w = line_nco_rule(N, a, b)
    return x, w


def quadrature_weight_vector_2d(N: int) -> np.ndarray:
    """Return the N^2 weight vector  c  for the integral constraint

        integral_{[0,1]^2}  u(x, y) dx dy  =  sum_{i,j}  w_i w_j u_{ij}

    using the tensor product of 1-D Newton-Cotes open rules of order N.
    The weight vector is arranged in lexicographic order matching the PDE
    discretisation.
    """
    _, w1d = line_nco_rule(N, 0.0, 1.0)
    # Tensor product weights
    w2d = np.outer(w1d, w1d).ravel()
    return w2d


# ---------------------------------------------------------------------------
# Gauss-Hermite quadrature  (from 464)
# ---------------------------------------------------------------------------

def gauss_hermite_nodes_weights(n: int):
    """Compute n-point Gauss-Hermite quadrature nodes and weights.

    The rule integrates  integral_{-inf}^{+inf}  exp(-x^2) f(x) dx
    exactly for polynomials of degree <= 2n-1.

    We use the Golub-Welsch algorithm: nodes are eigenvalues of the
    symmetric tridiagonal Jacobi matrix

        J = diag(sqrt(1), sqrt(2), ..., sqrt(n-1))  (off-diagonal)

    and weights are  w_i = pi * v_{i,0}^2  where v_{i,0} is the first
    component of the i-th eigenvector.
    """
    if n < 1:
        raise ValueError("gauss_hermite_nodes_weights: n must be >= 1")
    if n == 1:
        return np.array([0.0]), np.array([math.sqrt(math.pi)])

    # Jacobi matrix (symmetric tridiagonal)
    beta = np.sqrt(np.arange(1, n, dtype=np.float64))
    J = np.diag(beta, -1) + np.diag(beta, 1)

    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(J)

    nodes = eigvals
    weights = math.sqrt(math.pi) * eigvecs[0, :] ** 2
    return nodes, weights


def generalized_hermite_rule(n: int, alpha: float = 0.0):
    """Generalized Gauss-Hermite rule for weight function

        w(x) = |x|^alpha * exp(-x^2)    on (-inf, +inf)

    For alpha = 0, this reduces to standard Gauss-Hermite.
    The recurrence coefficients for the monic orthogonal polynomials are:

        a_k = 0            (symmetric weight)
        b_0 = Gamma((alpha+1)/2)
        b_k = k/2          if k even
        b_k = (k + alpha)/2  if k odd
    """
    if n < 1:
        raise ValueError("generalized_hermite_rule: n must be >= 1")

    # Recurrence coefficients for monic polynomials
    # p_{k+1}(x) = x p_k(x) - b_k p_{k-1}(x)
    b = np.zeros(n, dtype=np.float64)
    # b_0 = integral |x|^alpha exp(-x^2) dx = Gamma((alpha+1)/2)
    b[0] = math.gamma((alpha + 1.0) / 2.0)
    for k in range(1, n):
        if k % 2 == 0:
            b[k] = k / 2.0
        else:
            b[k] = (k + alpha) / 2.0

    # Jacobi matrix
    sqrt_b = np.sqrt(b[1:])
    J = np.diag(sqrt_b, -1) + np.diag(sqrt_b, 1)
    eigvals, eigvecs = np.linalg.eigh(J)
    nodes = eigvals
    weights = b[0] * eigvecs[0, :] ** 2
    return nodes, weights


# ---------------------------------------------------------------------------
# Triangle quadrature  (from 1302)
# ---------------------------------------------------------------------------

def triangle_quadrature_rule(degree: int = 2):
    """Return a quadrature rule for the unit triangle  T = {(x,y): x>=0, y>=0, x+y<=1}.

    The rule integrates polynomials up to the given degree exactly.

    Degree 1:  midpoint rule (1 point, weight 1/2)
    Degree 2:  3-point rule (vertices, weight 1/6 each)
    Degree 3:  4-point rule (centroid + 3 edge midpoints)
    Degree 4:  6-point rule

    Returns
    -------
    pts : (n_pts, 2) array of (x, y) coordinates
    wts : (n_pts,) array of weights
    """
    if degree <= 1:
        # Midpoint rule:  1 point at centroid, weight = area = 1/2
        pts = np.array([[1.0 / 3.0, 1.0 / 3.0]])
        wts = np.array([0.5])
    elif degree == 2:
        # 3-point rule at vertices, weight 1/6 each
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        wts = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    elif degree == 3:
        # 4-point rule: centroid + vertices scaled
        a1 = 1.0 / 3.0
        a2 = 1.0 / 5.0
        b2 = 3.0 / 5.0
        pts = np.array([
            [a1, a1],
            [a2, a2],
            [a2, b2],
            [b2, a2],
        ])
        wts = np.array([-27.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0])
    else:
        # 6-point rule for degree >= 4
        a1 = 0.0915762135098
        a2 = 0.44594849091596
        w1 = 0.10995174365532 / 2.0
        w2 = 0.22338158967801 / 2.0
        pts = np.array([
            [a1, a1],
            [a1, 1.0 - 2.0 * a1],
            [1.0 - 2.0 * a1, a1],
            [a2, a2],
            [a2, 1.0 - 2.0 * a2],
            [1.0 - 2.0 * a2, a2],
        ])
        wts = np.array([w1, w1, w1, w2, w2, w2])
    return pts, wts


def triangle_exactness_test(degree_max: int = 5):
    """Test polynomial exactness of the triangle quadrature rule.

    For monomial  x^a y^b  on the unit triangle, the exact integral is

        integral_T  x^a y^b dx dy  =  a! b! / (a + b + 2)!

    We compare this to the quadrature approximation for all (a, b) with
    a + b <= degree_max.
    """
    results = []
    for deg in range(degree_max + 1):
        pts, wts = triangle_quadrature_rule(degree=deg)
        max_err = 0.0
        for a_exp in range(deg + 2):
            for b_exp in range(deg + 2 - a_exp):
                # Exact integral
                exact = (math.factorial(a_exp) * math.factorial(b_exp)
                         / math.factorial(a_exp + b_exp + 2))
                # Quadrature approximation
                approx = 0.0
                for k in range(len(wts)):
                    approx += wts[k] * (pts[k, 0] ** a_exp) * (pts[k, 1] ** b_exp)
                err = abs(approx - exact)
                max_err = max(max_err, err)
        results.append((deg, max_err))
    return results


# ---------------------------------------------------------------------------
# Integral constraint evaluation
# ---------------------------------------------------------------------------

def evaluate_integral_constraint(u: np.ndarray, c_vec: np.ndarray) -> float:
    """Evaluate  c^T u  (the discrete integral of u)."""
    return float(np.dot(c_vec, u))


def integral_constraint_violation(u: np.ndarray, c_vec: np.ndarray, E_max: float) -> float:
    """Return  max(0, c^T u - E_max)  (violation of the integral constraint)."""
    val = evaluate_integral_constraint(u, c_vec)
    return max(0.0, val - E_max)
