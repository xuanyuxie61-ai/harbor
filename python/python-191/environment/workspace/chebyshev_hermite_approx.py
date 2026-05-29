"""
chebyshev_hermite_approx.py

Chebyshev Proxy Rootfinder and Hermite Interpolant for Matrix Element Approximation.

Scientific Background:
----------------------
1. Chebyshev Proxy Rootfinder (CPR):
   For a smooth function f on [a,b], we expand in Chebyshev polynomials:
   
       f(x) approx p_N(x) = sum_{j=0}^{N} c_j T_j(xi)
   
   where xi = 2*(x-a)/(b-a) - 1 maps [a,b] to [-1,1],
   and T_j(xi) = cos(j * arccos(xi)) are Chebyshev polynomials of the first kind.
   
   The coefficients are computed by collocation at Chebyshev nodes:
       xi_k = cos(k*pi/N),  k = 0,...,N
   
   The roots of p_N are found via the Chebyshev companion matrix C:
       C_{1,2} = 1
       C_{j,j-1} = 1/2,  C_{j,j+1} = 1/2   for j = 2,...,N-1
       C_{N,1:N} = -a_0/(2*a_N), ..., -a_{N-1}/(2*a_N) + 1/2
   
   The eigenvalues of C give the roots in [-1,1].

2. Hermite Interpolant:
   Given data (x_i, y_i, y'_i) for i=1,...,n, the Hermite interpolant H(x)
   satisfies H(x_i) = y_i and H'(x_i) = y'_i.
   
   Using divided differences with repeated nodes:
       z_{2i-1} = z_{2i} = x_i
       d_{2i-1} = y_i
       d_{2i} = y'_i
   
   The Newton form is:
       H(x) = d_0 + d_1*(x-z_0) + d_2*(x-z_0)*(x-z_1) + ...

3. Application to Matrix Elements:
   When matrix elements A_{ij} = K(x_i, x_j) involve expensive kernel evaluations,
   Chebyshev/Hermite interpolation provides O(N) or O(N^2) approximations.
"""

import numpy as np
from typing import Tuple, Callable, Optional


def chebyshev_nodes(a: float, b: float, N: int) -> np.ndarray:
    """
    Compute Chebyshev nodes of the second kind on [a, b]:
    
        x_k = (a+b)/2 + (b-a)/2 * cos(k*pi/N),  k = 0,...,N
    
    Args:
        a, b: interval endpoints
        N: degree of polynomial (N+1 nodes)
    
    Returns:
        x: array of shape (N+1,)
    """
    if a >= b:
        raise ValueError(f"Require a < b, got a={a}, b={b}")
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")
    k = np.arange(N + 1)
    xi = np.cos(k * np.pi / N)
    x = 0.5 * (b - a) * xi + 0.5 * (b + a)
    return x


def chebyshev_coefficients(f_vals: np.ndarray) -> np.ndarray:
    """
    Compute Chebyshev coefficients from function values at Chebyshev nodes.
    
    Using the discrete Chebyshev transform:
        c_j = (2/N) * sum_{k=0}^{N}'' f(x_k) * cos(j*k*pi/N) / (p_k)
    where p_0 = p_N = 2, p_k = 1 otherwise, and sum'' means halve endpoints.
    
    Args:
        f_vals: function values at N+1 Chebyshev nodes
    
    Returns:
        c: Chebyshev coefficients of length N+1
    """
    N = len(f_vals) - 1
    if N < 1:
        return f_vals.copy()
    
    k = np.arange(N + 1)
    pj = np.ones(N + 1)
    pj[0] = 2.0
    pj[N] = 2.0
    
    # Compute coefficients via DCT-like summation
    c = np.zeros(N + 1)
    for j in range(N + 1):
        c[j] = (2.0 / N) * np.sum(
            f_vals * np.cos(j * k * np.pi / N) / pj
        )
    return c


def chebyshev_companion_matrix(c: np.ndarray, epscutoff: float = 1e-13) -> np.ndarray:
    """
    Build the Chebyshev companion matrix for rootfinding.
    
    After truncating negligible trailing coefficients, form:
        A[0,1] = 1
        A[j,j-1] = 0.5, A[j,j+1] = 0.5  for j=1,...,Nt-2
        A[Nt-1,:Nt] = -c[0:Nt] / (2*c[Nt])
        A[Nt-1,Nt-2] += 0.5
    
    Args:
        c: Chebyshev coefficients
        epscutoff: truncation tolerance
    
    Returns:
        A: companion matrix of shape (Nt, Nt)
    """
    N = len(c) - 1
    if N < 1:
        raise ValueError("Need at least degree 1 polynomial")
    
    # Truncate tail
    cmax = np.max(np.abs(c))
    Nt = N
    tailnorm = 0.0
    for k in range(N, 0, -1):
        tailnorm += np.abs(c[k])
        if tailnorm < epscutoff * cmax and Nt > 1:
            Nt -= 1
        else:
            break
    
    if Nt < 1:
        Nt = 1
    
    A = np.zeros((Nt, Nt))
    if Nt > 1:
        A[0, 1] = 1.0
        for j in range(1, Nt - 1):
            A[j, j - 1] = 0.5
            A[j, j + 1] = 0.5
        
        denom = 2.0 * c[Nt]
        if abs(denom) < 1e-300:
            denom = 1e-300
        A[Nt - 1, :Nt] = -c[:Nt] / denom
        A[Nt - 1, Nt - 2] += 0.5
    
    return A


def cpr_roots(
    f: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    N: int = 64,
    tau: float = 1e-8,
    sigma: float = 1e-6
) -> Tuple[np.ndarray, float]:
    """
    Chebyshev Proxy Rootfinder.
    
    Finds real roots of f(x)=0 on [a,b] using Chebyshev interpolation
    followed by companion matrix eigenvalue computation.
    
    Mathematical guarantee:
        For analytic f in a Bernstein ellipse, the Chebyshev interpolant
        converges exponentially: ||f - p_N||_inf <= C * rho^{-N}.
    
    Args:
        f: function handle (vectorized)
        a, b: interval endpoints
        N: degree of Chebyshev expansion
        tau: imaginary-part tolerance for real root acceptance
        sigma: Chebyshev-interval extension tolerance
    
    Returns:
        roots: real roots in [a,b]
        Einter: interstitial interpolation residual estimate
    """
    if a >= b:
        raise ValueError("Require a < b")
    if N < 2:
        raise ValueError("N must be >= 2")
    
    t = np.arange(N + 1) * np.pi / N
    xi = np.cos(t)
    x = 0.5 * (b - a) * xi + 0.5 * (b + a)
    
    try:
        fa = f(x)
    except Exception:
        fa = np.array([f(xi_val) for xi_val in x])
    fa = np.asarray(fa, dtype=np.float64)
    
    # Chebyshev coefficients
    c = chebyshev_coefficients(fa)
    
    # Companion matrix
    A = chebyshev_companion_matrix(c)
    
    # Eigenvalues
    all_roots = np.linalg.eigvals(A)
    
    # Filter real roots in [-1-sigma, 1+sigma]
    roots_list = []
    for ev in all_roots:
        if np.abs(ev.imag) < tau * max(1.0, np.abs(ev.real)):
            if np.abs(ev.real) <= 1.0 + sigma:
                root_physical = 0.5 * (b - a) * ev.real + 0.5 * (b + a)
                if a - sigma * (b - a) <= root_physical <= b + sigma * (b - a):
                    roots_list.append(root_physical)
    
    roots = np.sort(np.array(roots_list))
    
    # Interstitial residual
    tinter = t[:N] + 0.5 / N
    xiint = np.cos(tinter)
    xinter = 0.5 * (b - a) * xiint + 0.5 * (b + a)
    try:
        fainter = f(xinter)
    except Exception:
        fainter = np.array([f(xi_val) for xi_val in xinter])
    
    # Evaluate interpolant at interstitial points
    taall = np.concatenate([t, tinter])
    faall = np.concatenate([fa, fainter])
    
    fpoly = np.zeros(len(taall))
    for j in range(len(c)):
        fpoly += c[j] * np.cos(j * taall)
    
    Einter = np.max(np.abs(faall[N + 1:] - fpoly[N + 1:]))
    
    return roots, Einter


def hermite_divided_differences(
    x: np.ndarray,
    y: np.ndarray,
    yp: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build divided difference table for Hermite interpolation.
    
    Given nodes x_i with function values y_i and derivatives yp_i,
    construct the table for the unique polynomial H of degree <= 2n-1
    satisfying H(x_i) = y_i and H'(x_i) = yp_i.
    
    The extended node vector is:
        z = [x_0, x_0, x_1, x_1, ..., x_{n-1}, x_{n-1}]
    
    Args:
        x: distinct nodes, shape (n,)
        y: function values, shape (n,)
        yp: derivative values, shape (n,)
    
    Returns:
        z: extended nodes, shape (2n,)
        d: divided differences (last entry is leading coefficient)
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    yp = np.asarray(yp).ravel()
    
    n = len(x)
    if len(y) != n or len(yp) != n:
        raise ValueError("x, y, yp must have same length")
    if n == 0:
        return np.array([]), np.array([])
    
    # Check distinctness
    if len(np.unique(x)) != n:
        raise ValueError("Hermite nodes must be distinct")
    
    nd = 2 * n
    z = np.zeros(nd)
    z[0::2] = x
    z[1::2] = x
    
    d = np.zeros(nd)
    d[0] = y[0]
    d[2::2] = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    d[1::2] = yp
    
    # TODO HOLE 2: Compute higher-order divided differences for Hermite interpolation.
    # Given extended nodes z (length 2n) and base differences d,
    # compute the full divided difference table for the unique polynomial
    # H of degree <= 2n-1 satisfying H(x_i) = y_i and H'(x_i) = yp_i.
    # The algorithm iterates i from 2 to 2n-1 and j from 2n-1 down to i,
    # with d[j] = (d[j] - d[j-1]) / (z[j] - z[j-i]) for distinct nodes.
    raise NotImplementedError("Hole 2: hermite divided differences not implemented")


def hermite_evaluate(
    z: np.ndarray,
    d: np.ndarray,
    x_eval: np.ndarray
) -> np.ndarray:
    """
    Evaluate Hermite interpolant using Newton form.
    
        H(x) = d_0 + d_1*(x-z_0) + d_2*(x-z_0)*(x-z_1) + ...
    
    Args:
        z: extended nodes
        d: divided differences
        x_eval: evaluation points
    
    Returns:
        H(x_eval)
    """
    x_eval = np.asarray(x_eval)
    nd = len(d)
    if nd == 0:
        return np.zeros_like(x_eval)
    
    result = d[-1] * np.ones_like(x_eval, dtype=np.float64)
    for i in range(nd - 2, -1, -1):
        result = result * (x_eval - z[i]) + d[i]
    return result


def approximate_matrix_element(
    kernel: Callable[[float, float], float],
    xi: float,
    xj: float,
    order: int = 8,
    method: str = "chebyshev"
) -> float:
    """
    Approximate a matrix element K(xi, xj) using high-order interpolation.
    
    For diagonal-dominant kernels, we interpolate along the radial direction
    r = |xi - xj| using Chebyshev or Hermite methods.
    
    Args:
        kernel: K(x, y) function
        xi, xj: evaluation points
        order: interpolation order
        method: 'chebyshev' or 'hermite'
    
    Returns:
        approximate kernel value
    """
    r = abs(xi - xj)
    eps = 1e-12
    
    if method == "chebyshev":
        a = max(0.0, r - 0.5)
        b = r + 0.5
        nodes = chebyshev_nodes(a, b, order)
        vals = np.array([kernel(xi, xj + (t - r)) for t in nodes])
        # Barycentric interpolation at point r
        c = chebyshev_coefficients(vals)
        # Evaluate at r (mapped to xi=-1..1)
        xi_mapped = 2.0 * (r - a) / (b - a) - 1.0
        if abs(xi_mapped) > 1.0 + eps:
            xi_mapped = np.clip(xi_mapped, -1.0, 1.0)
        result = np.sum(c * np.cos(np.arange(len(c)) * np.arccos(np.clip(xi_mapped, -1.0, 1.0))))
        return result
    
    elif method == "hermite":
        # Use Hermite interpolation with central difference for derivative
        h = 0.01
        nodes = np.linspace(max(0.0, r - 0.5), r + 0.5, min(order, 4))
        vals = np.array([kernel(xi, xj + (t - r)) for t in nodes])
        dvals = np.array([
            (kernel(xi, xj + (t - r) + h) - kernel(xi, xj + (t - r) - h)) / (2 * h)
            for t in nodes
        ])
        z, d = hermite_divided_differences(nodes, vals, dvals)
        return float(hermite_evaluate(z, d, np.array([r]))[0])
    
    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    # Test CPR on a simple function
    f = lambda x: np.cos(3 * x)
    roots, err = cpr_roots(f, 0.0, 2.0, N=32)
    print("CPR roots:", roots)
    print("Interstitial error:", err)
    
    # Test Hermite
    x = np.array([0.0, 1.0, 2.0])
    y = np.sin(x)
    yp = np.cos(x)
    z, d = hermite_divided_differences(x, y, yp)
    print("Hermite at 0.5:", hermite_evaluate(z, d, np.array([0.5])))
