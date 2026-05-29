"""
quadrature_rules.py
===================
High-Dimensional Gaussian Quadrature for Chemical Source Term Integration.

Based on seed project 936 (pyramid_rule):
- Legendre-Gauss quadrature for spatial dimensions
- Jacobi-Gauss quadrature for composition space (mixture fraction)
- Tensor-product pyramid rules for 3D reaction rate integration

Scientific Context:
-------------------
In turbulent combustion DNS, the mean reaction rate at a point cannot be
computed from mean quantities alone due to the strong nonlinearity of
Arrhenius kinetics. The Reynolds-averaged or filtered reaction rate requires
integration over the joint probability density function (PDF):

  ω̄_k = ∭ ω̇_k(Y, T, P) * P(Y, T, P) dY dT dP

For laminar flamelet libraries, the scalar dissipation rate χ and mixture
fraction Z define the flame structure. Integrals over the composition space
pyramid (a 3D simplex for three-scalar mixing) require specialized quadrature.

This module provides:
1. Legendre-Gauss quadrature for [-1,1] with weight w(x)=1
2. Jacobi-Gauss quadrature for [-1,1] with weight w(x)=(1-x)^α(1+x)^β
3. Pyramid tensor-product rules for 3D integration
4. Application to chemical source term averaging
"""

import numpy as np
from math import gamma as math_gamma


def legendre_gauss_nodes_weights(n):
    """
    Compute n-point Legendre-Gauss quadrature nodes and weights on [-1,1].
    Based on seed 936 (legendre_compute).

    The integral:  ∫_{-1}^{1} f(x) dx ≈ Σ_i w_i f(x_i)

    Algorithm: Golub-Welsch eigenvalue method on symmetric tridiagonal
    Jacobi matrix with zero diagonal and off-diagonals:
        β_j = j / sqrt(4j² - 1)
    """
    if n < 1:
        raise ValueError("Quadrature order must be >= 1")
    if n == 1:
        return np.array([0.0]), np.array([2.0])

    # Jacobi matrix for Legendre polynomials
    j = np.arange(1.0, n)
    beta = j / np.sqrt(4.0 * j**2 - 1.0)
    J = np.diag(beta, 1) + np.diag(beta, -1)

    eigvals, eigvecs = np.linalg.eigh(J)
    x = eigvals
    w = 2.0 * eigvecs[0, :]**2
    return x, w


def jacobi_gauss_nodes_weights(n, alpha, beta_param):
    """
    Compute n-point Jacobi-Gauss quadrature on [-1,1] with weight
    w(x) = (1-x)^α (1+x)^β.

    Based on seed 936 (jacobi_compute).

    The integral:  ∫_{-1}^{1} (1-x)^α (1+x)^β f(x) dx ≈ Σ_i w_i f(x_i)

    Jacobi matrix elements:
        a_i = (β² - α²) / ((2i+α+β)(2i+α+β+2))
        b_i = 2/(2i+α+β) * sqrt(i(i+α+β)(i+α)(i+β) / ((2i+α+β-1)(2i+α+β+1)))
    """
    if n < 1:
        raise ValueError("Quadrature order must be >= 1")
    if alpha <= -1.0 or beta_param <= -1.0:
        raise ValueError("Alpha and beta must be > -1")

    if n == 1:
        x0 = (beta_param - alpha) / (alpha + beta_param + 2.0)
        # Weight from analytic formula
        w0 = (2.0**(alpha + beta_param + 1.0)
              * math_gamma(alpha + 1.0) * math_gamma(beta_param + 1.0)
              / math_gamma(alpha + beta_param + 2.0))
        return np.array([x0]), np.array([w0])

    # Diagonal and off-diagonal of Jacobi matrix
    i = np.arange(1.0, n)
    ab = (beta_param**2 - alpha**2) / ((2.0 * i + alpha + beta_param)
                                        * (2.0 * i + alpha + beta_param + 2.0))
    bb = np.zeros(n - 1)
    for idx in range(len(i)):
        ii = i[idx]
        num = 4.0 * ii * (ii + alpha + beta_param) * (ii + alpha) * (ii + beta_param)
        den = ((2.0 * ii + alpha + beta_param)**2
               * (2.0 * ii + alpha + beta_param - 1.0)
               * (2.0 * ii + alpha + beta_param + 1.0))
        bb[idx] = np.sqrt(num / den)

    J = np.diag(ab) + np.diag(bb, 1) + np.diag(bb, -1)
    eigvals, eigvecs = np.linalg.eigh(J)
    x = eigvals

    # Weights from first eigenvector component
    mu0 = (2.0**(alpha + beta_param + 1.0)
           * math_gamma(alpha + 1.0) * math_gamma(beta_param + 1.0)
           / math_gamma(alpha + beta_param + 2.0))
    w = mu0 * eigvecs[0, :]**2
    return x, w


def pyramid_tensor_product_quadrature(legendre_order, jacobi_order):
    """
    Construct a 3D quadrature rule for the unit pyramid:
        -(1-z) ≤ x ≤ 1-z
        -(1-z) ≤ y ≤ 1-z
                 0 ≤ z ≤ 1

    Based on seed 936 (pyramid_handle).

    The rule is a tensor product of:
      - Legendre-Gauss of order legendre_order for x and y (mapped to [-1,1])
      - Jacobi-Gauss with α=2, β=0 of order jacobi_order for z

    Volume of unit pyramid = 4/3.

    Returns
    -------
    xpts : ndarray, shape (3, npts)
        Quadrature nodes (x, y, z).
    w : ndarray, shape (npts,)
        Quadrature weights (sum to 1 for unit-weight integration).
    """
    leg_x, leg_w = legendre_gauss_nodes_weights(legendre_order)
    jac_x, jac_w = jacobi_gauss_nodes_weights(jacobi_order, 2.0, 0.0)

    npts = legendre_order * legendre_order * jacobi_order
    xpts = np.zeros((3, npts))
    w = np.zeros(npts)

    volume = 4.0 / 3.0
    l = 0
    for k in range(jacobi_order):
        zk = (jac_x[k] + 1.0) / 2.0
        wk = jac_w[k] / 2.0
        for j in range(legendre_order):
            xj = leg_x[j]
            wj = leg_w[j]
            for i in range(legendre_order):
                xi = leg_x[i]
                wi = leg_w[i]
                xpts[0, l] = xi * (1.0 - zk)
                xpts[1, l] = xj * (1.0 - zk)
                xpts[2, l] = zk
                w[l] = wi * wj * wk / (4.0 * volume)
                l += 1

    return xpts, w


def integrate_reaction_rate_over_pdf(omega_func, pdf_func, legendre_order=4,
                                     jacobi_order=4, pdf_params=None):
    """
    Integrate a reaction rate ω(Y,T,P) weighted by a PDF P(Y,T,P)
    over the composition space using pyramid quadrature.

    Parameters
    ----------
    omega_func : callable
        Function omega(Y, T, P) returning reaction rate.
    pdf_func : callable
        Function pdf(Y, T, P, params) returning PDF value.
    legendre_order, jacobi_order : int
        Quadrature orders.
    pdf_params : dict or None
        Parameters for the PDF function.

    Returns
    -------
    mean_rate : float
        Integrated mean reaction rate.
    """
    xpts, w = pyramid_tensor_product_quadrature(legendre_order, jacobi_order)

    # Map pyramid coordinates to physical composition space
    # x → Y_H2 ∈ [0, 1-z]
    # y → Y_O2 ∈ [0, 1-z]
    # z → T/T_max (normalized temperature)
    # This is a simplified mapping for demonstration.

    mean_rate = 0.0
    for i in range(len(w)):
        Y_h2 = max(0.0, min(1.0, (xpts[0, i] + 1.0) / 2.0))
        Y_o2 = max(0.0, min(1.0, (xpts[1, i] + 1.0) / 2.0))
        T_norm = max(0.0, min(1.0, xpts[2, i]))
        T = 300.0 + T_norm * 2200.0  # 300K to 2500K

        # Normalize mass fractions
        total = Y_h2 + Y_o2
        if total > 0:
            Y_h2 = Y_h2 / total * 0.3  # max 30% H2 in mixture
            Y_o2 = Y_o2 / total * 0.7  # balance O2/N2
        else:
            Y_h2 = 0.0
            Y_o2 = 0.233
        Y_h2o = 0.0
        Y_n2 = 1.0 - Y_h2 - Y_o2 - Y_h2o
        Y = np.array([Y_h2, Y_o2, Y_h2o, Y_n2])
        Y = Y / Y.sum()

        omega_val = omega_func(Y, T)
        pdf_val = pdf_func(Y, T, 101325.0, pdf_params)
        mean_rate += w[i] * omega_val * pdf_val

    return mean_rate


def gauss_legendre_2d_tensor(nx, ny, ax, bx, ay, by):
    """
    2D tensor-product Gauss-Legendre quadrature on rectangle [ax,bx]×[ay,by].

    Returns
    -------
    x : ndarray, shape (nx*ny,)
    y : ndarray, shape (nx*ny,)
    w : ndarray, shape (nx*ny,)
    """
    x1d, wx = legendre_gauss_nodes_weights(nx)
    y1d, wy = legendre_gauss_nodes_weights(ny)

    # Map from [-1,1] to [a,b]
    x1d = 0.5 * (bx - ax) * x1d + 0.5 * (bx + ax)
    y1d = 0.5 * (by - ay) * y1d + 0.5 * (by + ay)
    wx = 0.5 * (bx - ax) * wx
    wy = 0.5 * (by - ay) * wy

    npts = nx * ny
    x = np.zeros(npts)
    y = np.zeros(npts)
    w = np.zeros(npts)
    idx = 0
    for j in range(ny):
        for i in range(nx):
            x[idx] = x1d[i]
            y[idx] = y1d[j]
            w[idx] = wx[i] * wy[j]
            idx += 1
    return x, y, w
