"""
numerical_utils.py - Core Numerical Foundations for Surrogate-Based UQ

This module provides the numerical building blocks for polynomial chaos
surrogate construction and uncertainty quantification. It integrates:
  - Gauss-Legendre quadrature via implicit QL algorithm (from annulus_rule)
  - Polynomial arithmetic over GF(2) for combinatorial indexing (from collatz_polynomial)
  - Forward Euler ODE integration (from exm)
  - Convergence assessment utilities (from control_bio)
  - Iterative smoothing operators (from polygon_average)

Mathematical Framework
----------------------
1. Gauss-Legendre Quadrature [1]:
   Nodes x_i and weights w_i are eigenvalues/eigenvectors of the Jacobi matrix:
     J = tridiag(beta_i, alpha_i, beta_i)
   where alpha_i = 0, beta_i = i / sqrt(4*i^2 - 1) for Legendre polynomials.

2. Implicit QL Algorithm [2]:
   Diagonalizes symmetric tridiagonal matrix T = Q*D*Q^T via:
     - Shifted QR steps with Wilkinson shift
     - Accumulated rotations for eigenvector computation

3. GF(2) Polynomial Arithmetic [3]:
   Polynomials P(x) = sum c_k x^k over GF(2) with c_k in {0,1}:
     - Addition: XOR of coefficient vectors
     - Multiplication: convolution mod 2
     - Division: polynomial long division mod 2

4. Iterative Polygon Smoothing [4]:
   Vertex averaging: p_i^{k+1} = (p_i^k + p_{i+1}^k) / 2
   Convergence to ellipse under repeated application (eigenvalue analysis).

References
----------
[1] Golub & Welsch, "Calculation of Gauss quadrature rules", Math. Comp., 1969.
[2] EISPACK imtqlx routine; Kautsky & Elhay, "IQPACK", 1989.
[3] Lidl & Niederreiter, "Finite Fields", Cambridge Univ. Press, 1997.
[4] Symons et al., "Polygon averaging and convergence", Amer. Math. Monthly, 2006.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, List
import math
import sys


# ---------------------------------------------------------------------------
# Gauss-Legendre quadrature via Elhay-Kautsky / implicit QL algorithm
# ---------------------------------------------------------------------------

def imtqlx(n: int, d: NDArray, e: NDArray, z: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Implicit QL algorithm for symmetric tridiagonal matrices.

    Diagonalizes T = tridiag(e, d, e) by accumulating Givens rotations.
    Modified from EISPACK imtqlx: also applies rotations to vector z,
    giving eigenvector components in z.

    Parameters
    ----------
    n : int
        Matrix dimension.
    d : ndarray(n,)
        Diagonal elements (modified in-place to eigenvalues).
    e : ndarray(n,)
        Sub-diagonal elements (e[0] is ignored; e[1..n-1] are sub-diag).
    z : ndarray(n,)
        Initial vector; on output contains first components of eigenvectors.

    Returns
    -------
    d : ndarray(n,)
        Eigenvalues in ascending order.
    e : ndarray(n,)
        Sub-diagonal (destroyed).
    z : ndarray(n,)
        First row of eigenvector matrix (quadrature weights before scaling).

    Algorithm
    ---------
    Wilkinson shift: sigma = d[n-1] - sign(p)*sqrt(p^2 + e[n-1]^2)
    where p is the (1,1) element of the trailing 2x2 shift matrix.
    Givens rotation to chase bulge; convergence when |e[m]| < eps*(|d[m]|+|d[m+1]|).
    """
    d = d.copy()
    e = e.copy()
    z = z.copy()

    if n <= 0:
        return d, e, z
    if n == 1:
        return d, e, z

    eps = np.finfo(float).eps
    e[n - 1] = 0.0

    for l in range(n):
        j = 0
        while True:
            # Find small sub-diagonal element
            m = l
            while m < n - 1:
                test = abs(d[m]) + abs(d[m + 1])
                if test == 0.0:
                    test = 1.0
                if abs(e[m]) <= eps * test:
                    break
                m += 1

            if m == l:
                break

            if j >= 100:
                raise RuntimeError(f"imtqlx: QL iteration failed to converge for l={l}, j={j}")

            j += 1

            # Wilkinson shift
            g = (d[l + 1] - d[l]) / (2.0 * e[l])
            r = math.sqrt(g * g + 1.0)
            sign_g = 1.0 if g >= 0 else -1.0
            g = d[m] - d[l] + e[l] / (g + sign_g * r)

            s = 1.0
            c = 1.0
            p = 0.0

            for i in range(m - 1, l - 1, -1):
                f = s * e[i]
                b = c * e[i]

                if abs(f) >= abs(g):
                    c = g / f
                    r = math.sqrt(c * c + 1.0)
                    e[i + 1] = f * r
                    s = 1.0 / r
                    c = c * s
                else:
                    s = f / g
                    r = math.sqrt(s * s + 1.0)
                    e[i + 1] = g * r
                    c = 1.0 / r
                    s = s * c

                g = d[i + 1] - p
                r = (d[i] - g) * s + 2.0 * c * b
                p = s * r
                d[i + 1] = g + p
                g = c * r - b

                # Accumulate rotation into z
                zz = z[i + 1]
                z[i + 1] = s * z[i] + c * zz
                z[i] = c * z[i] - s * zz

            d[l] -= p
            e[l] = g
            if m < n - 1:
                e[m] = 0.0

    # Sort eigenvalues and corresponding z values
    idx = np.argsort(d)
    d = d[idx]
    z = z[idx]

    return d, e, z


def legendre_ek_compute(n: int) -> Tuple[NDArray, NDArray]:
    """
    Compute Gauss-Legendre quadrature rule of order n on [-1, 1].

    Uses Elhay-Kautsky method: construct Jacobi matrix for Legendre
    polynomials and diagonalize via implicit QL.

    The Jacobi matrix for monic Legendre polynomials satisfies the
    three-term recurrence:
      P_{k+1}(x) = (x - alpha_k) P_k(x) - beta_k^2 P_{k-1}(x)
    with alpha_k = 0, beta_k = k / sqrt(4k^2 - 1).

    Parameters
    ----------
    n : int
        Number of quadrature points (order of the rule).

    Returns
    -------
    x : ndarray(n,)
        Quadrature nodes in [-1, 1].
    w : ndarray(n,)
        Quadrature weights, summing to 2 (length of interval).

    Mathematical Details
    --------------------
    The Golub-Welsch algorithm identifies nodes as eigenvalues of J
    and weights as w_i = 2 * v_{i,1}^2 where v_{i,1} is the first
    component of the i-th normalized eigenvector.
    """
    if n <= 0:
        return np.array([]), np.array([])
    if n == 1:
        return np.array([0.0]), np.array([2.0])

    # Jacobi matrix diagonal and sub-diagonal
    d = np.zeros(n)  # alpha_k = 0 for Legendre
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = i / math.sqrt(4.0 * i * i - 1.0)

    # Initial z vector: z[0] = sqrt(weight_0) = sqrt(2)
    z = np.zeros(n)
    z[0] = math.sqrt(2.0)

    # Diagonalize
    d, e, z = imtqlx(n, d, e, z)

    # Nodes and weights
    x = d
    w = z * z  # w_i = 2 * v_{i,1}^2 (z already contains sqrt(2)*v_{i,1})

    return x, w


def rule_adjust(a: float, b: float, c: float, d: float,
                norder: int, x: NDArray, w: NDArray) -> Tuple[NDArray, NDArray]:
    """
    Affine mapping of a quadrature rule from [c, d] to [a, b].

    Given rule (x, w) on [c, d], map to [a, b] via:
      x_new = a + (b - a) * (x - c) / (d - c)
      w_new = w * (b - a) / (d - c)

    This preserves the integral:
      integral_a^b f(x) dx = sum_i w_new_i * f(x_new_i)
    """
    scale = (b - a) / (d - c)
    x_new = a + (x - c) * scale
    w_new = w * scale
    return x_new, w_new


# ---------------------------------------------------------------------------
# GF(2) Polynomial Arithmetic (for combinatorial multi-index operations)
# ---------------------------------------------------------------------------

def gf2_poly_degree(p: NDArray) -> int:
    """
    Degree of polynomial over GF(2).

    Parameters
    ----------
    p : ndarray
        Coefficient vector [c_0, c_1, ..., c_n] with c_i in {0,1}.
        p[i] is coefficient of x^i.

    Returns
    -------
    int
        Highest index with nonzero coefficient, or -1 for zero polynomial.
    """
    for i in range(len(p) - 1, -1, -1):
        if p[i] != 0:
            return i
    return -1


def gf2_poly_add(p1: NDArray, p2: NDArray) -> NDArray:
    """Addition of two GF(2) polynomials (XOR of coefficient vectors)."""
    n = max(len(p1), len(p2))
    result = np.zeros(n, dtype=int)
    for i in range(len(p1)):
        result[i] ^= p1[i]
    for i in range(len(p2)):
        result[i] ^= p2[i]
    return result


def gf2_poly_mul(p1: NDArray, p2: NDArray) -> NDArray:
    """Multiplication of two GF(2) polynomials (convolution mod 2)."""
    if len(p1) == 0 or len(p2) == 0:
        return np.array([], dtype=int)
    n = len(p1) + len(p2) - 1
    result = np.zeros(n, dtype=int)
    for i in range(len(p1)):
        if p1[i]:
            for j in range(len(p2)):
                result[i + j] ^= p2[j]
    return result


def collatz_polynomial_step(p: NDArray) -> NDArray:
    """
    One step of the Collatz polynomial iteration over GF(2).

    If p is divisible by x (constant term = 0): divide by x.
    Otherwise: compute P*(x+1) + 1 over GF(2).

    This generates orbits in the space of GF(2) polynomials,
    analogous to the 3n+1 problem for integers. The orbit
    structure is used here for combinatorial indexing of
    multi-index sets in sparse polynomial chaos expansions.

    Parameters
    ----------
    p : ndarray
        Coefficient vector in GF(2).

    Returns
    -------
    ndarray
        Next polynomial in the Collatz orbit.
    """
    if len(p) == 0:
        return np.array([], dtype=int)

    if p[0] == 0:
        # Divisible by x: shift right (divide by x)
        if len(p) <= 1:
            return np.array([0], dtype=int)
        return p[1:].copy()
    else:
        # P*(x+1) + 1: multiply by (x+1) = [1,1], then add 1 = [1]
        xp1 = np.array([1, 1], dtype=int)
        prod = gf2_poly_mul(p, xp1)
        result = gf2_poly_add(prod, np.array([1], dtype=int))
        return result


# ---------------------------------------------------------------------------
# Multi-index set generation for polynomial chaos
# ---------------------------------------------------------------------------

def total_order_multi_index(d: int, p: int) -> NDArray:
    """
    Generate multi-index set for total-order polynomial chaos.

    A = {alpha in N_0^d : |alpha| <= p}
    where |alpha| = alpha_1 + alpha_2 + ... + alpha_d

    Uses graded lexicographic ordering (from ubvec GRLEX):
    first sorted by total degree |alpha|, then lexicographically.

    The cardinality is C(d+p, p) = (d+p)! / (d! * p!)

    Parameters
    ----------
    d : int
        Number of dimensions (random inputs).
    p : int
        Maximum total degree.

    Returns
    -------
    indices : ndarray(m, d)
        Multi-index set where m = C(d+p, p).

    Algorithm
    ---------
    Recursive enumeration: for each total degree k=0,1,...,p,
    enumerate all compositions of k into d non-negative parts
    using the stars-and-bars combinatorial construction.
    """
    if d <= 0 or p < 0:
        return np.zeros((0, max(d, 1)), dtype=int)

    indices = []

    def enumerate_degree(remaining_dim: int, remaining_degree: int,
                         current: List[int]):
        """Enumerate all compositions of remaining_degree into remaining_dim parts."""
        if remaining_dim == 1:
            indices.append(current + [remaining_degree])
            return
        for k in range(remaining_degree + 1):
            enumerate_degree(remaining_dim - 1, remaining_degree - k,
                             current + [k])

    for k in range(p + 1):
        enumerate_degree(d, k, [])

    return np.array(indices, dtype=int)


def hyperbolic_cross_multi_index(d: int, p: int, q: float = 0.5) -> NDArray:
    """
    Generate hyperbolic cross multi-index set for sparse PCE.

    A_q = {alpha in N_0^d : prod_{i=1}^d (1 + alpha_i)^q <= 2^p}

    For q < 1, this is a strict subset of the total-order set,
    dramatically reducing cardinality for high dimensions while
    preserving approximation of smooth functions.

    The weighted norm ||alpha||_{HC,q} = prod(1+alpha_i)^q captures
    the anisotropic decay of polynomial chaos coefficients for
    functions with bounded mixed derivatives.

    Parameters
    ----------
    d : int
        Number of dimensions.
    p : int
        Level parameter (controls cardinality).
    q : float
        Norm parameter (0 < q <= 1). Default 0.5 for standard HC.

    Returns
    -------
    indices : ndarray(m, d)
        Hyperbolic cross multi-index set.
    """
    if d <= 0 or p < 0:
        return np.zeros((0, max(d, 1)), dtype=int)

    threshold = 2.0 ** p

    indices = []
    # Enumerate via recursive search with pruning
    def search(dim: int, current: List[int], current_weight: float):
        if dim == d:
            indices.append(current[:])
            return
        # Maximum possible value for this dimension
        max_val = int((threshold / current_weight) ** (1.0 / q)) if current_weight > 0 else int(threshold ** (1.0 / q))
        max_val = min(max_val, 2 * p + 1)  # safety bound
        for k in range(max_val + 1):
            new_weight = current_weight * (1.0 + k) ** q
            if new_weight <= threshold:
                search(dim + 1, current + [k], new_weight)

    search(0, [], 1.0)

    if len(indices) == 0:
        return np.zeros((0, d), dtype=int)

    # Sort by total degree
    arr = np.array(indices, dtype=int)
    total_deg = arr.sum(axis=1)
    arr = arr[np.argsort(total_deg)]
    return arr


# ---------------------------------------------------------------------------
# ODE Integration (Forward Euler with adaptive step)
# ---------------------------------------------------------------------------

def forward_euler(f, y0: NDArray, t_span: Tuple[float, float],
                  dt: float = 0.001, rtol: float = 1e-6) -> Tuple[NDArray, NDArray]:
    """
    Forward Euler ODE integration with basic adaptive stepping.

    Solves dy/dt = f(t, y) with y(t0) = y0.

    For reaction-diffusion systems, the stiffness ratio dictates
    time step constraints:
      dt < 2 / |lambda_max|
    where lambda_max is the most negative eigenvalue of the
    Jacobian (diffusion term dominates: lambda ~ -D/h^2).

    Parameters
    ----------
    f : callable
        RHS function f(t, y) -> dy/dt.
    y0 : ndarray
        Initial condition.
    t_span : (float, float)
        (t_start, t_end).
    dt : float
        Initial time step.
    rtol : float
        Relative tolerance for adaptive stepping.

    Returns
    -------
    t : ndarray(nt,)
        Time points.
    y : ndarray(nt, ny)
        Solution at each time point.
    """
    t0, tf = t_span
    y = np.array(y0, dtype=float)
    t = t0
    trajectory = [y.copy()]
    times = [t]

    max_steps = 1000000
    step_count = 0
    current_dt = dt

    while t < tf and step_count < max_steps:
        step_count += 1

        # Clamp dt to not overshoot
        if t + current_dt > tf:
            current_dt = tf - t

        # Forward Euler step
        dydt = f(t, y)
        y_new = y + current_dt * dydt

        # Basic error estimation: compare with half-step
        half_dt = current_dt / 2.0
        y_half = y + half_dt * dydt
        dydt_half = f(t + half_dt, y_half)
        y_half2 = y_half + half_dt * dydt_half

        err = np.linalg.norm(y_new - y_half2)
        y_norm = np.linalg.norm(y_half2)

        if y_norm > 1e-14:
            rel_err = err / y_norm
        else:
            rel_err = err

        if rel_err > rtol * 10 and current_dt > 1e-12:
            # Error too large, reduce step
            current_dt *= 0.5
            continue

        # Accept step
        y = y_half2  # Use more accurate half-step result
        t += current_dt
        times.append(t)
        trajectory.append(y.copy())

        # Try to increase step
        if rel_err < rtol * 0.1:
            current_dt = min(current_dt * 1.5, (tf - t0) / 10.0)

    return np.array(times), np.array(trajectory)


# ---------------------------------------------------------------------------
# Iterative Smoothing Operator (from polygon averaging)
# ---------------------------------------------------------------------------

def iterative_smoothing_1d(values: NDArray, n_iterations: int = 5,
                           alpha: float = 0.5) -> NDArray:
    """
    Iterative neighbor-averaging smoothing for 1D data.

    Generalizes polygon averaging to 1D signals:
      v_i^{k+1} = (1 - alpha) * v_i^k + alpha * (v_{i-1}^k + v_{i+1}^k) / 2

    For alpha = 0.5, this is the standard polygon averaging step.
    Convergence analysis: eigenvalues of the smoothing operator are
      lambda_k = 1 - alpha * (1 - cos(2*pi*k/N))
    so high-frequency modes decay fastest.

    Parameters
    ----------
    values : ndarray(n,)
        Input signal.
    n_iterations : int
        Number of smoothing passes.
    alpha : float
        Smoothing parameter in (0, 1].

    Returns
    -------
    ndarray(n,)
        Smoothed signal.
    """
    n = len(values)
    if n < 3:
        return values.copy()

    v = values.copy().astype(float)
    for _ in range(n_iterations):
        v_new = v.copy()
        for i in range(1, n - 1):
            v_new[i] = (1.0 - alpha) * v[i] + alpha * 0.5 * (v[i - 1] + v[i + 1])
        # Neumann boundary conditions
        v_new[0] = (1.0 - alpha) * v[0] + alpha * v[1]
        v_new[-1] = (1.0 - alpha) * v[-1] + alpha * v[-2]
        v = v_new

    return v


# ---------------------------------------------------------------------------
# Convergence assessment (from control_bio)
# ---------------------------------------------------------------------------

def convergence_ratio(x_new: NDArray, x_old: NDArray,
                      current_min: float = 1.0) -> float:
    """
    Compute convergence ratio using L1-norm relative change.

    ratio = ||x_new - x_old||_1 / (||x_old||_1 + eps)

    Used in forward-backward sweep convergence testing.
    The minimum ratio over all iterations tracks monotone convergence.

    Parameters
    ----------
    x_new : ndarray
        Current iterate.
    x_old : ndarray
        Previous iterate.
    current_min : float
        Running minimum ratio.

    Returns
    -------
    float
        Updated minimum convergence ratio.
    """
    denom = np.linalg.norm(x_old, 1) + 1e-14
    ratio = np.linalg.norm(x_new - x_old, 1) / denom
    return min(current_min, ratio)


# ---------------------------------------------------------------------------
# Special functions for UQ
# ---------------------------------------------------------------------------

def gamma_ln(z: float) -> float:
    """
    Log-gamma function via Stirling's approximation with correction terms.

    ln(Gamma(z)) = (z - 0.5)*ln(z) - z + 0.5*ln(2*pi)
                   + 1/(12*z) - 1/(360*z^3) + 1/(1260*z^5)

    Accurate for z > 0. For z <= 0, uses reflection formula:
      Gamma(z)*Gamma(1-z) = pi / sin(pi*z)
    """
    if z <= 0:
        if z == int(z):
            return float('inf')  # poles at non-positive integers
        # Reflection formula
        return math.log(math.pi / abs(math.sin(math.pi * z))) - gamma_ln(1.0 - z)

    if z < 0.5:
        return math.log(math.pi / abs(math.sin(math.pi * z))) - gamma_ln(1.0 - z)

    # Stirling series
    result = (z - 0.5) * math.log(z) - z + 0.5 * math.log(2.0 * math.pi)
    z2 = z * z
    result += 1.0 / (12.0 * z)
    result -= 1.0 / (360.0 * z * z2)
    result += 1.0 / (1260.0 * z * z2 * z2)
    return result


def beta_function(a: float, b: float) -> float:
    """
    Beta function B(a, b) = Gamma(a)*Gamma(b) / Gamma(a+b).

    Computed in log-space for numerical stability:
      ln B(a,b) = ln Gamma(a) + ln Gamma(b) - ln Gamma(a+b)
    """
    if a <= 0 or b <= 0:
        return float('nan')
    log_b = gamma_ln(a) + gamma_ln(b) - gamma_ln(a + b)
    return math.exp(log_b)


def multinomial_coefficient(alpha: NDArray) -> float:
    """
    Multinomial coefficient: (|alpha|)! / (alpha_1! * alpha_2! * ... * alpha_d!)

    Used in PCE normalization and tensor product weight computation.
    """
    n = int(np.sum(alpha))
    log_coeff = gamma_ln(n + 1.0)
    for a in alpha:
        log_coeff -= gamma_ln(int(a) + 1.0)
    return math.exp(log_coeff)


# ---------------------------------------------------------------------------
# Bounding and conditioning utilities
# ---------------------------------------------------------------------------

def safe_divide(a: float, b: float, default: float = 0.0,
                eps: float = 1e-14) -> float:
    """Safe division with near-zero denominator protection."""
    if abs(b) < eps:
        return default if abs(a) < eps else (float('inf') if a > 0 else float('-inf'))
    return a / b


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi] range."""
    return max(lo, min(hi, value))


def condition_number_estimate(A: NDArray) -> float:
    """
    Estimate condition number of matrix A.
    Uses ratio of max to min singular values.
    """
    try:
        s = np.linalg.svd(A, compute_uv=False)
        if s[-1] < 1e-15:
            return float('inf')
        return float(s[0] / s[-1])
    except np.linalg.LinAlgError:
        return float('inf')
