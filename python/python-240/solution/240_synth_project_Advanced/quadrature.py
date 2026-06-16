"""
数值积分模块：对称求积法则
Symmetric quadrature rules for spatial and angular integration.

Algorithms sourced from:
  - 916_prism_jaskowiec_rule (prism quadrature for 3D integration)
  - 1316_triangle_symq_rule (symmetric triangle quadrature up to degree 50)
"""
import numpy as np


def gauss_legendre_1d(n_points):
    """
    Gauss-Legendre quadrature points and weights on [-1, 1].

    Parameters
    ----------
    n_points : int
        Number of quadrature points (1 to 5)

    Returns
    -------
    points : ndarray
    weights : ndarray
    """
    if n_points == 1:
        return np.array([0.0]), np.array([2.0])
    elif n_points == 2:
        return np.array([-0.5773502692, 0.5773502692]), np.array([1.0, 1.0])
    elif n_points == 3:
        return np.array([-0.7745966692, 0.0, 0.7745966692]), np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])
    elif n_points == 4:
        pts = np.array([-0.8611363116, -0.3399810436, 0.3399810436, 0.8611363116])
        wts = np.array([0.3478548451, 0.6521451549, 0.6521451549, 0.3478548451])
        return pts, wts
    elif n_points == 5:
        pts = np.array([-0.9061798459, -0.5384693101, 0.0, 0.5384693101, 0.9061798459])
        wts = np.array([0.2369268850, 0.4786286705, 0.5688888889, 0.4786286705, 0.2369268850])
        return pts, wts
    else:
        raise ValueError(f"Unsupported n_points={n_points}")


def gauss_legendre_2d(n_points_1d):
    """
    2D Gauss-Legendre quadrature on [-1, 1]² (tensor product).

    Parameters
    ----------
    n_points_1d : int
        Points per dimension

    Returns
    -------
    x, y : ndarray
        Quadrature point coordinates
    weights : ndarray
        Product weights
    """
    pts_1d, wts_1d = gauss_legendre_1d(n_points_1d)
    x, y = np.meshgrid(pts_1d, pts_1d, indexing='ij')
    wx, wy = np.meshgrid(wts_1d, wts_1d, indexing='ij')
    weights = wx * wy
    return x.flatten(), y.flatten(), weights.flatten()


def triangle_quadrature_symmetric(order):
    """
    Symmetric quadrature rules for reference triangle (0,0)-(1,0)-(0,1).

    From 1316_triangle_symq_rule: order k rule is exact for polynomials
    of degree ≤ k.

    Parameters
    ----------
    order : int
        Quadrature order (1, 2, 3, 4, 5)

    Returns
    -------
    xi, eta : ndarray
        Barycentric-like coordinates
    weights : ndarray
        Quadrature weights (sum to 0.5 = triangle area)
    """
    if order == 1:
        # 1-point centroid rule
        xi = np.array([1.0/3.0])
        eta = np.array([1.0/3.0])
        wts = np.array([0.5])
    elif order == 2:
        # 3-point rule (degree 2)
        xi = np.array([1.0/6.0, 2.0/3.0, 1.0/6.0])
        eta = np.array([1.0/6.0, 1.0/6.0, 2.0/3.0])
        wts = np.array([1.0/6.0, 1.0/6.0, 1.0/6.0])
    elif order == 3:
        # 4-point rule (degree 3)
        xi = np.array([1.0/3.0, 0.6, 0.2, 0.2])
        eta = np.array([1.0/3.0, 0.2, 0.6, 0.2])
        wts = np.array([-27.0/96.0, 25.0/96.0, 25.0/96.0, 25.0/96.0])
    elif order == 4:
        # 6-point rule (degree 4)
        a1 = 0.445948490915965
        a2 = 0.091576213509771
        w1 = 0.111690794839005
        w2 = 0.054975871827661
        xi = np.array([a1, 1-2*a1, a1, a2, 1-2*a2, a2])
        eta = np.array([a1, a1, 1-2*a1, a2, a2, 1-2*a2])
        wts = np.array([w1, w1, w1, w2, w2, w2])
    elif order == 5:
        # 7-point rule (degree 5)
        a1 = 0.470142064105115
        a2 = 0.101286507323456
        w0 = 0.1125
        w1 = 0.066197076394253
        w2 = 0.062969590272414
        xi = np.array([1.0/3.0, a1, 1-2*a1, a1, a2, 1-2*a2, a2])
        eta = np.array([1.0/3.0, a1, a1, 1-2*a1, a2, a2, 1-2*a2])
        wts = np.array([w0, w1, w1, w1, w2, w2, w2])
    else:
        raise ValueError(f"Unsupported order {order}")

    return xi, eta, wts


def prism_quadrature_jaskowiec(order_triangle, order_1d):
    """
    Prism quadrature by Jaskowiec rule (from 916_prism_jaskowiec_rule).
    Tensor product of triangle rule and 1D Gauss rule for prism elements.

    Used for 3D spacetime integration (transverse × longitudinal).

    Parameters
    ----------
    order_triangle : int
        Triangle quadrature order
    order_1d : int
        1D Gauss-Legendre order

    Returns
    -------
    xi_tri, eta_tri : ndarray (n_tri,)
        Triangle coordinates
    z_1d : ndarray (n_1d,)
        1D points
    weights_tri, weights_1d : ndarray
        Respective weights
    """
    xi_tri, eta_tri, wts_tri = triangle_quadrature_symmetric(order_triangle)
    z_1d, wts_1d = gauss_legendre_1d(order_1d)

    return xi_tri, eta_tri, z_1d, wts_tri, wts_1d


def integrate_over_triangle(f, tri_vertices, order=3):
    """
    Integrate function f over a triangle using symmetric quadrature.

    Parameters
    ----------
    f : callable
        Function f(x, y) to integrate
    tri_vertices : tuple of (x, y) pairs
        Triangle vertices
    order : int
        Quadrature order

    Returns
    -------
    float
        Integral value
    """
    (x1, y1), (x2, y2), (x3, y3) = tri_vertices

    # Jacobian: 2 * area
    J = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

    xi, eta, wts = triangle_quadrature_symmetric(order)

    # Map reference triangle to physical triangle
    # x = (1-ξ-η) x1 + ξ x2 + η x3
    # y = (1-ξ-η) y1 + ξ y2 + η y3
    integral = 0.0
    for i in range(len(xi)):
        x_phys = (1 - xi[i] - eta[i]) * x1 + xi[i] * x2 + eta[i] * x3
        y_phys = (1 - xi[i] - eta[i]) * y1 + xi[i] * y2 + eta[i] * y3
        integral += wts[i] * f(x_phys, y_phys)

    return integral * J


def integrate_over_domain(f, x_grid, y_grid, dx, dy):
    """
    Integrate function over rectangular domain using trapezoidal rule.

    Parameters
    ----------
    f : ndarray (nx, ny)
        Function values on grid
    x_grid, y_grid : ndarray
        Grid coordinates
    dx, dy : float
        Grid spacing

    Returns
    -------
    float
        Integral value
    """
    # 2D trapezoidal rule
    integral = np.sum(f) * dx * dy
    # Boundary corrections (half weight)
    integral -= 0.5 * (np.sum(f[0, :]) + np.sum(f[-1, :])) * dx * dy
    integral -= 0.5 * (np.sum(f[:, 0]) + np.sum(f[:, -1])) * dx * dy
    # Corner corrections
    integral += 0.25 * (f[0, 0] + f[0, -1] + f[-1, 0] + f[-1, -1]) * dx * dy
    return integral


def angular_integration_periodic(f_phi, n_phi=72):
    """
    Integrate periodic function over [0, 2π] using trapezoidal rule.
    Exact for trigonometric polynomials up to degree n_phi/2.

    Parameters
    ----------
    f_phi : ndarray (n_phi,)
        Function values at uniform φ points
    n_phi : int
        Number of points

    Returns
    -------
    float
        Integral
    """
    dphi = 2.0 * np.pi / n_phi
    return np.sum(f_phi) * dphi


def monomial_integral_triangle(p, q):
    """
    Exact integral of monomial x^p y^q over reference triangle:
        ∫∫ x^p y^q dx dy = p! q! / (p + q + 2)!

    From 1316_triangle_unit_monomial_integral.

    Parameters
    ----------
    p, q : int
        Monomial powers

    Returns
    -------
    float
        Exact integral
    """
    from math import factorial
    return factorial(p) * factorial(q) / factorial(p + q + 2)


def verify_quadrature_accuracy(order, n_tests=10):
    """
    Verify that symmetric quadrature of given order integrates
    polynomials of degree ≤ order exactly.

    Parameters
    ----------
    order : int
        Quadrature order to test
    n_tests : int
        Number of random polynomials to test

    Returns
    -------
    max_error : float
        Maximum integration error
    """
    rng = np.random.default_rng(42)
    max_err = 0.0

    for _ in range(n_tests):
        # Random polynomial of degree = order
        coeffs = rng.uniform(-1, 1, (order + 1, order + 1))

        def f(x, y):
            val = 0.0
            for p in range(order + 1):
                for q in range(order + 1 - p):
                    val += coeffs[p, q] * x**p * y**q
            return val

        # Exact integral (sum of monomial integrals)
        exact = 0.0
        for p in range(order + 1):
            for q in range(order + 1 - p):
                exact += coeffs[p, q] * monomial_integral_triangle(p, q)

        # Quadrature approximation
        approx = integrate_over_triangle(f, ((0, 0), (1, 0), (0, 1)), order=order)

        err = abs(exact - approx)
        max_err = max(max_err, err)

    return max_err
