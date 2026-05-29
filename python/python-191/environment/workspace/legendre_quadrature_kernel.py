"""
legendre_quadrature_kernel.py

Fast Gauss-Legendre Quadrature and Disk Integration for Kernel Matrix Construction.

Scientific Background:
----------------------
1. Gauss-Legendre Quadrature:
   For integral I(f) = integral_{-1}^{1} f(x) dx,
   the n-point Gauss-Legendre rule is exact for polynomials up to degree 2n-1:
   
       Q_n(f) = sum_{i=1}^{n} w_i * f(x_i)
   
   where x_i are roots of P_n(x) (Legendre polynomial of degree n),
   and weights w_i = 2 / [(1 - x_i^2) * (P_n'(x_i))^2].
   
   Legendre polynomial recurrence:
       P_0(x) = 1
       P_1(x) = x
       (k+1) P_{k+1}(x) = (2k+1) x P_k(x) - k P_{k-1}(x)

2. Glaser-Liu-Rokhlin (GLR) Fast Algorithm:
   Computes roots and weights in O(n) time using Taylor expansions
   and local Newton iterations starting from asymptotic approximations:
   
       theta_k = pi * (4k - 1) / (4n + 2)
       x_k approx [1 - (n-1)/(8n^3) - 1/(384n^4)*(39 - 28/sin^2(theta_k))] * cos(theta_k)

3. Disk Integration:
   For the unit disk D = {(x,y): x^2 + y^2 <= 1},
   
       integral_D f(x,y) dA = pi * sum_{j=1}^{NT} sum_{i=1}^{NR} w_i * f(r_i*cos(t_j), r_i*sin(t_j))
   
   where r_i = sqrt((xi_i + 1)/2) maps Legendre nodes to [0,1],
   and t_j = 2*pi*(j-1)/NT are uniform angles.

4. Application to Kernel Matrices:
   Matrix elements of integral operators often involve:
   K_{ij} = integral_D K(x_i, y; x_j, y) dy or similar convolutions.
   High-order quadrature ensures spectral accuracy for smooth kernels.
"""

import numpy as np
from typing import Tuple


def _legendre_poly_and_deriv(n: int, x: float) -> Tuple[float, float]:
    """
    Evaluate P_n(x) and P_n'(x) using recurrence relations.
    
    Recurrence:
        P_0 = 1, P_1 = x
        (k+1) P_{k+1} = (2k+1) x P_k - k P_{k-1}
    
    Derivative recurrence:
        (1 - x^2) P_k' = k (P_{k-1} - x P_k)
    """
    if n == 0:
        return 1.0, 0.0
    if n == 1:
        return x, 1.0
    
    pm2 = 1.0
    pm1 = x
    for k in range(1, n):
        p = ((2 * k + 1) * x * pm1 - k * pm2) / (k + 1)
        pm2 = pm1
        pm1 = p
    
    # Derivative
    if abs(x) >= 1.0 - 1e-15:
        # Use alternative formula near endpoints
        dp = n * (pm1 - x * pm2) / (1e-15 if abs(1 - x * x) < 1e-15 else 1 - x * x)
    else:
        dp = n * (pm2 - x * pm1) / (1 - x * x)
    
    return pm1, dp


def legendre_compute_glr(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Gauss-Legendre nodes and weights using the GLR fast algorithm.
    
    For n points, returns x in [-1,1] and weights w summing to 2.
    
    Args:
        n: number of quadrature points (n >= 1)
    
    Returns:
        x: nodes, shape (n,)
        w: weights, shape (n,)
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if n == 1:
        return np.array([0.0]), np.array([2.0])
    
    x = np.zeros(n)
    w = np.zeros(n)
    
    # Number of symmetric pairs to compute
    m = n // 2
    
    for k in range(1, m + 1):
        # Asymptotic initial guess
        theta = np.pi * (4 * k - 1) / (4 * n + 2)
        xk = (1.0 - (n - 1) / (8.0 * n ** 3) -
              1.0 / (384.0 * n ** 4) * (39.0 - 28.0 / (np.sin(theta) ** 2))) * np.cos(theta)
        
        # Newton iteration for root of P_n
        for _ in range(20):
            p, dp = _legendre_poly_and_deriv(n, xk)
            if abs(dp) < 1e-300:
                break
            dx = p / dp
            xk -= dx
            if abs(dx) < 1e-15:
                break
        
        idx = k - 1
        x[idx] = -xk
        x[n - 1 - idx] = xk
        
        # Weight computation
        _, dp = _legendre_poly_and_deriv(n, xk)
        w_val = 2.0 / ((1.0 - xk * xk) * dp * dp)
        w[idx] = w_val
        w[n - 1 - idx] = w_val
    
    # Handle middle point for odd n
    if n % 2 == 1:
        mid = m
        x[mid] = 0.0
        _, dp = _legendre_poly_and_deriv(n, 0.0)
        if abs(dp) < 1e-300:
            dp = 1e-300
        w[mid] = 2.0 / (dp * dp)
    
    # Normalize weights to sum to 2
    w_sum = np.sum(w)
    if w_sum > 0:
        w *= 2.0 / w_sum
    
    return x, w


def rescale_quadrature(
    x: np.ndarray,
    w: np.ndarray,
    a: float,
    b: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rescale Gauss-Legendre rule from [-1,1] to [a,b].
    
    Mapping: t = 0.5 * ((x+1)*b - (x-1)*a)
    Jacobian: dt = 0.5 * (b-a) dx
    
    Args:
        x: nodes in [-1,1]
        w: weights for [-1,1]
        a, b: new interval endpoints
    
    Returns:
        t: nodes in [a,b]
        w_scaled: weights for [a,b]
    """
    if a >= b:
        raise ValueError("Require a < b")
    t = 0.5 * ((x + 1.0) * b - (x - 1.0) * a)
    w_scaled = 0.5 * (b - a) * w
    return t, w_scaled


def disk01_rule(nr: int, nt: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct a quadrature rule for the unit disk.
    
    Uses Gauss-Legendre in the radial direction and uniform angles.
    
    The integral formula:
        I(f) = integral_D f(x,y) dA
             = pi * sum_{j=1}^{nt} sum_{i=1}^{nr} w_i * f(r_i*cos(t_j), r_i*sin(t_j))
    
    where r_i = sqrt(xr_i) with xr_i Legendre nodes on [0,1],
    and w_i are corresponding weights divided by nt.
    
    Args:
        nr: number of radial points
        nt: number of angular points
    
    Returns:
        w: weights for disk rule, shape (nr,)
        r: radial nodes, shape (nr,)
        t: angular nodes, shape (nt,)
    """
    if nr < 1 or nt < 1:
        raise ValueError("nr and nt must be >= 1")
    
    xr, wr = legendre_compute_glr(nr)
    # Shift to [0,1]
    xr = (xr + 1.0) / 2.0
    wr = wr / 2.0
    
    # TODO(Hole_1): Implement the radial mapping and angular nodes for unit disk quadrature.
    # Scientific knowledge required:
    #   - For polar coordinates on unit disk: dA = r dr dθ
    #   - Mapping from Gauss-Legendre nodes xr in [0,1] to radius: r = sqrt(xr)
    #     (Jacobian: dr = du / (2*sqrt(u)), which cancels the r factor in dA)
    #   - Angular nodes: t_j = 2*pi*j/nt for j=0,...,nt-1
    #   - Weights: each radial weight wr_i is divided by nt (number of angles)
    # Hint: the correct radial mapping ensures uniform area measure on the disk.
    raise NotImplementedError("Hole_1: disk01_rule radial mapping and weights not implemented")
    return w, r, t


def integrate_disk_kernel(
    kernel: callable,
    nr: int = 16,
    nt: int = 32
) -> float:
    """
    Integrate a rotationally symmetric kernel over the unit disk.
    
    For K(x,y) = K(r) depending only on radius:
        I = integral_D K(r) dA = 2*pi * integral_0^1 K(r) * r dr
    
    For general K(x,y), full 2D quadrature is applied.
    
    Args:
        kernel: function K(x, y) -> float
        nr, nt: quadrature resolution
    
    Returns:
        approximate integral value
    """
    w, r, t = disk01_rule(nr, nt)
    total = 0.0
    for it in range(nt):
        cos_t = np.cos(t[it])
        sin_t = np.sin(t[it])
        for ir in range(nr):
            x = r[ir] * cos_t
            y = r[ir] * sin_t
            total += w[ir] * kernel(x, y)
    return np.pi * total


def construct_kernel_matrix_1d(
    nodes: np.ndarray,
    kernel_func: callable,
    quadrature_order: int = 16
) -> np.ndarray:
    """
    Construct a kernel matrix K_{ij} = integral_{-1}^{1} phi_i(x) phi_j(x) w(x) dx
    using Gauss-Legendre quadrature.
    
    For the standard L2 inner product on [-1,1]:
        K_{ij} = sum_{q=1}^{n_q} w_q * phi_i(x_q) * phi_j(x_q)
    
    Args:
        nodes: basis function centers, shape (n,)
        kernel_func: basis function evaluator phi(x, center)
        quadrature_order: GL quadrature order
    
    Returns:
        K: kernel matrix of shape (n, n)
    """
    n = len(nodes)
    if n == 0:
        return np.zeros((0, 0))
    
    xq, wq = legendre_compute_glr(quadrature_order)
    
    # Evaluate basis functions at quadrature points
    Phi = np.zeros((n, quadrature_order))
    for i in range(n):
        Phi[i, :] = kernel_func(xq, nodes[i])
    
    # K = Phi * diag(wq) * Phi^T
    K = Phi @ np.diag(wq) @ Phi.T
    return K


def construct_kernel_matrix_2d_disk(
    nodes: np.ndarray,
    kernel_func: callable,
    nr: int = 16,
    nt: int = 32
) -> np.ndarray:
    """
    Construct a 2D kernel matrix using disk quadrature.
    
    K_{ij} = integral_D phi_i(x,y) phi_j(x,y) dA
    
    Args:
        nodes: node coordinates, shape (n, 2)
        kernel_func: phi(x, y, cx, cy) -> float
        nr, nt: disk quadrature resolution
    
    Returns:
        K: matrix of shape (n, n)
    """
    n = nodes.shape[0]
    if n == 0:
        return np.zeros((0, 0))
    
    w, r, t = disk01_rule(nr, nt)
    
    # Generate all quadrature points on disk
    nq = nr * nt
    xq = np.zeros(nq)
    yq = np.zeros(nq)
    wq = np.zeros(nq)
    idx = 0
    for it in range(nt):
        for ir in range(nr):
            xq[idx] = r[ir] * np.cos(t[it])
            yq[idx] = r[ir] * np.sin(t[it])
            wq[idx] = w[ir]
            idx += 1
    
    Phi = np.zeros((n, nq))
    for i in range(n):
        Phi[i, :] = kernel_func(xq, yq, nodes[i, 0], nodes[i, 1])
    
    K = np.pi * Phi @ np.diag(wq) @ Phi.T
    return K


if __name__ == "__main__":
    # Test GL quadrature on a polynomial
    x, w = legendre_compute_glr(8)
    # Integral of x^6 from -1 to 1 = 2/7
    val = np.sum(w * x ** 6)
    print("GL test (should be ~0.2857):", val)
    
    # Test disk integration
    k = lambda x, y: x * x + y * y
    val = integrate_disk_kernel(k, nr=16, nt=32)
    print("Disk integral (should be ~pi/2):", val)
