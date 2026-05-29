"""
quadrature_rules.py
===================
Numerical quadrature rules for composite finite-element integration.

Incorporates core algorithms from:
- 950_quadrature_weights_vandermonde : Vandermonde-based quadrature weight
  computation for exact integration of monomials.
- 519_hermite_exactness : Gauss-Hermite quadrature for probabilistic
  integration over unbounded domains (stochastic fiber strength).

Scientific role:
    Provides Gaussian quadrature for element stiffness integration and
    Hermite quadrature for stochastic analysis of composite strength.
    The quadrature rules ensure exact integration of polynomial strain
    fields up to a specified degree.

Key formulas:
-----------
1. Gauss-Legendre quadrature on [a, b]:
   Integral_a^b f(x) dx = sum_{i=1}^n w_i f(x_i)
   where x_i are roots of P_n(x) and w_i = 2 / [(1-x_i^2) (P'_n(x_i))^2].

2. Quadrature weights via Vandermonde:
   Solve V^T * w = rhs, where rhs_i = (b^i - a^i) / i.

3. Gauss-Hermite quadrature (physicist's weight):
   Integral_{-inf}^{+inf} f(x) exp(-x^2) dx = sum_{i=1}^n w_i f(x_i)
   Exact for polynomials up to degree 2n-1.

4. Hermite monomial exact integral:
   H(n) = (n-1)!! * sqrt(pi) / 2^{n/2}   for n even
   H(n) = 0                               for n odd

5. Stochastic strength integration:
   For fiber strength X_T ~ N(mu, sigma^2), the probability of failure
   under stress sigma_11 is:
   P_f = Phi( (sigma_11 - mu) / sigma )
   where Phi is the standard normal CDF.
"""

import numpy as np


def quadrature_weights_vandermonde(n, a, b, x):
    """
    Compute quadrature weights by solving the Vandermonde system.

    Given n distinct nodes x_i in [a, b], find weights w_i such that
    the quadrature rule is exact for monomials x^{k}, k=0..n-1.

    Parameters
    ----------
    n : int
        Number of points.
    a, b : float
        Interval endpoints.
    x : ndarray, shape (n,)
        Quadrature nodes (must be distinct).

    Returns
    -------
    w : ndarray, shape (n,)
        Quadrature weights.
    """
    x = np.asarray(x, dtype=float).flatten()
    if len(x) != n:
        raise ValueError("x must have length n.")
    if np.any(np.diff(np.sort(x)) < 1e-12):
        raise ValueError("Quadrature nodes must be distinct.")

    V = np.zeros((n, n))
    V[0, :] = 1.0
    for i in range(1, n):
        V[i, :] = V[i - 1, :] * x

    rhs = np.zeros(n)
    for i in range(n):
        power = i + 1
        rhs[i] = (b ** power - a ** power) / power

    # Solve V^T w = rhs
    w = np.linalg.solve(V.T, rhs)
    return w


def gauss_legendre_nodes_weights(n):
    """
    Compute Gauss-Legendre quadrature nodes and weights on [-1, 1].

    Uses the Golub-Welsch algorithm via the symmetric tridiagonal
    Jacobi matrix for monic Legendre polynomials.

    Parameters
    ----------
    n : int
        Number of points.

    Returns
    -------
    x, w : ndarray
        Nodes and weights.
    """
    if n < 1:
        raise ValueError("n must be >= 1.")

    # Jacobi matrix for monic Legendre: alpha_i = 0, beta_i = i/sqrt(4i^2-1)
    if n == 1:
        return np.array([0.0]), np.array([2.0])

    i = np.arange(1.0, n)
    beta = i / np.sqrt(4.0 * i * i - 1.0)
    J = np.diag(beta, 1) + np.diag(beta, -1)
    eigvals, eigvecs = np.linalg.eigh(J)
    x = eigvals
    w = 2.0 * (eigvecs[0, :] ** 2)
    return x, w


def factorial2(n):
    """Double factorial n!! = n*(n-2)*...*2 or 1."""
    if n < 0:
        return 1.0
    result = 1.0
    while n > 1:
        result *= n
        n -= 2
    return result


def hermite_monomial_integral(n, option=1):
    """
    Exact integral of x^n * exp(-x^2) over (-inf, +inf).

    Parameters
    ----------
    n : int
        Monomial degree.
    option : int
        1: physicist weight exp(-x^2)
        2: probabilist weight exp(-x^2/2)
        3: normalized physicist
        4: normalized probabilist

    Returns
    -------
    value : float
        Exact integral value.
    """
    if n < 0:
        return -np.inf
    if n % 2 == 1:
        return 0.0

    if option in (0, 1):
        value = factorial2(n - 1) * np.sqrt(np.pi) / (2.0 ** (n / 2.0))
    elif option == 2:
        value = factorial2(n - 1) * np.sqrt(2.0 * np.pi)
    elif option == 3:
        value = factorial2(n - 1) / (2.0 ** (n / 2.0))
    elif option == 4:
        value = factorial2(n - 1)
    else:
        raise ValueError("Invalid option.")
    return value


def gauss_hermite_nodes_weights(n, option=1):
    """
    Compute Gauss-Hermite quadrature nodes and weights.

    For physicist's weight exp(-x^2):
        x_i are eigenvalues of symmetric tridiagonal Jacobi matrix
        with alpha_i = 0, beta_i = sqrt(i/2).

    Returns
    -------
    x, w : ndarray
    """
    if n < 1:
        raise ValueError("n must be >= 1.")

    # Build symmetric tridiagonal Jacobi matrix
    alpha = np.zeros(n)
    beta = np.zeros(n - 1)
    for i in range(n - 1):
        beta[i] = np.sqrt((i + 1.0) / 2.0)

    J = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)
    eigvals, eigvecs = np.linalg.eigh(J)
    x = eigvals

    # Weights from first component of eigenvectors
    w = np.sqrt(np.pi) * (eigvecs[0, :] ** 2)

    if option == 2:
        x *= np.sqrt(2.0)
        w *= np.sqrt(2.0)
    elif option == 3:
        w /= np.sqrt(np.pi)
    elif option == 4:
        x *= np.sqrt(2.0)
        w /= np.sqrt(2.0 * np.pi)
        w *= np.sqrt(2.0)

    return x, w


def stochastic_fiber_failure_probability(sigma_11, mu_strength, sigma_strength,
                                         n_hermite=16):
    """
    Compute probability of fiber failure using Gauss-Hermite quadrature
    for the stochastic integral.

    Model: Fiber strength X_T ~ Normal(mu_strength, sigma_strength^2).
    The failure probability is:
        P_f = P(X_T < sigma_11)
          = 1/sqrt(2*pi) * integral exp(-z^2/2) * H(sigma_11 - mu - sigma*z) dz

    where H is the Heaviside step function.

    Parameters
    ----------
    sigma_11 : float
        Applied axial stress (Pa).
    mu_strength : float
        Mean fiber strength (Pa).
    sigma_strength : float
        Standard deviation of fiber strength (Pa).
    n_hermite : int
        Number of Hermite quadrature points.

    Returns
    -------
    P_f : float
        Failure probability in [0, 1].
    """
    if sigma_strength <= 0:
        return 1.0 if sigma_11 >= mu_strength else 0.0

    x, w = gauss_hermite_nodes_weights(n_hermite, option=4)
    # Transform: z = mu + sigma * sqrt(2) * xi, weight includes exp(-xi^2)
    # For option 4: integral f(xi) exp(-xi^2)/sqrt(pi) dxi
    z = mu_strength + sigma_strength * np.sqrt(2.0) * x
    indicator = (z <= sigma_11).astype(float)
    P_f = np.sum(w * indicator) / np.sqrt(np.pi)
    return np.clip(P_f, 0.0, 1.0)


def integrate_strain_energy(element_strain, C_matrix, quad_order=4):
    """
    Integrate element strain energy density using Gauss-Legendre quadrature.

    U = 0.5 * integral epsilon^T C epsilon dV
      ~ 0.5 * sum_i w_i * (epsilon_i^T C epsilon_i) * J_i

    Parameters
    ----------
    element_strain : callable or ndarray
        Strain field evaluated at quadrature points.
    C_matrix : ndarray (3, 3)
        Stiffness matrix.
    quad_order : int
        Quadrature order.

    Returns
    -------
    energy : float
        Integrated strain energy.
    """
    xi, wi = gauss_legendre_nodes_weights(quad_order)
    energy = 0.0
    for i in range(quad_order):
        for j in range(quad_order):
            # Tensor product quadrature on reference element [-1,1]^2
            w_ij = wi[i] * wi[j]
            if callable(element_strain):
                eps = element_strain(xi[i], xi[j])
            else:
                eps = element_strain[i * quad_order + j]
            eps = np.asarray(eps, dtype=float).flatten()[:3]
            energy += w_ij * 0.5 * eps @ C_matrix @ eps
    return energy
