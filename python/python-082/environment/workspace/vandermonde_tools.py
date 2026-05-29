"""
vandermonde_tools.py
====================
Vandermonde matrix utilities and Discontinuous Galerkin (DG) spectral operators.

Incorporates core algorithms from:
- 1385_vandermonde_interp_2d : 2D Vandermonde interpolation matrix for
  polynomial reconstruction of displacement fields over damaged elements.
- 274_dg1d_maxwell : 1D DG operators (Jacobi polynomials, differentiation
  matrices, lift operators) adapted for high-order strain field approximation
  across composite ply interfaces.

Scientific role:
    Provides high-order polynomial interpolation and spectral differentiation
    for strain/stress fields in composite damage analysis. The DG framework
    allows discontinuities in displacement gradients across ply boundaries,
    which is essential for capturing delamination and matrix cracking.

Key formulas:
-----------
1. 1D Vandermonde matrix on Legendre-Gauss-Lobatto nodes:
   V_{ij} = P_{j-1}(r_i),  i=1..N+1, j=1..N+1
   where P_n are normalized Legendre polynomials.

2. Differentiation matrix:
   D_r = V_r * V^{-1}
   where (V_r)_{ij} = dP_{j-1}/dr (r_i)

3. 2D Vandermonde interpolation (total degree M):
   For n data points (x_i, y_i) and polynomial
   p(x,y) = sum_{s=0}^{M} sum_{ex+ey=s} c_{ex,ey} x^{ex} y^{ey}
   the Vandermonde matrix is:
   A_{i,j} = x_i^{ex} * y_i^{ey}

4. Jacobi polynomial recurrence (orthonormal):
   P_0(x) = 1/sqrt(gamma0)
   P_1(x) = ((alpha+beta+2)*x/2 + (alpha-beta)/2) / sqrt(gamma1)
   a_n P_{n+1} = (x - b_n) P_n - a_{n-1} P_{n-1}

5. Legendre-Gauss-Lobatto nodes (collocation points):
   Roots of (1-x^2) P'_N(x) = 0, used for spectral element discretization.
"""

import numpy as np


def jacobi_polynomial(x, alpha, beta, N):
    """
    Evaluate normalized Jacobi polynomial P_N^{(alpha,beta)}(x).

    Parameters
    ----------
    x : array_like
        Evaluation points.
    alpha, beta : float
        Jacobi parameters (>-1).
    N : int
        Polynomial degree.

    Returns
    -------
    P : ndarray
        Values of P_N at x.
    """
    x = np.asarray(x, dtype=float)
    if alpha <= -1.0 or beta <= -1.0:
        raise ValueError("Jacobi parameters must be > -1.")
    if N < 0:
        return np.zeros_like(x)

    # Gamma function via scipy if available, else use math.gamma
    try:
        from math import gamma as _gamma
    except ImportError:
        def _gamma(z):
            return np.exp(_gammaln(z))

    def _gammaln(z):
        # Lanczos approximation for log-gamma
        if z < 0.5:
            return np.log(np.pi) - np.log(np.sin(np.pi * z)) - _gammaln(1.0 - z)
        p = [76.18009172947146, -86.50532032941677,
             24.01409824083091, -1.231739572450155,
             0.1208650973866179e-2, -0.5395239384953e-5]
        y = z
        x_lanczos = y + 5.5
        x_lanczos -= (y + 0.5) * np.log(x_lanczos)
        ser = 1.000000000190015
        for i, p_i in enumerate(p):
            y += 1.0
            ser += p_i / y
        return -x_lanczos + np.log(2.5066282746310005 * ser / z)

    gamma0 = (2.0 ** (alpha + beta + 1.0) / (alpha + beta + 1.0)
              * _gamma(alpha + 1.0) * _gamma(beta + 1.0)
              / _gamma(alpha + beta + 1.0))

    PL = np.zeros((N + 1, x.size))
    PL[0, :] = 1.0 / np.sqrt(gamma0)
    if N == 0:
        return PL[0, :]

    gamma1 = (alpha + 1.0) * (beta + 1.0) / (alpha + beta + 3.0) * gamma0
    PL[1, :] = (((alpha + beta + 2.0) * x / 2.0 + (alpha - beta) / 2.0)
                / np.sqrt(gamma1))
    if N == 1:
        return PL[1, :]

    a_old = (2.0 / (2.0 + alpha + beta)
             * np.sqrt((alpha + 1.0) * (beta + 1.0)
                       / (alpha + beta + 3.0)))

    for i in range(1, N):
        h1 = 2.0 * i + alpha + beta
        a_new = (2.0 / (h1 + 2.0)
                 * np.sqrt((i + 1.0) * (i + 1.0 + alpha + beta)
                           * (i + 1.0 + alpha) * (i + 1.0 + beta)
                           / (h1 + 1.0) / (h1 + 3.0)))
        b_new = - (alpha ** 2 - beta ** 2) / h1 / (h1 + 2.0)
        PL[i + 1, :] = (1.0 / a_new
                        * (-a_old * PL[i - 1, :]
                           + (x - b_new) * PL[i, :]))
        a_old = a_new

    return PL[N, :]


def grad_jacobi_polynomial(x, alpha, beta, N):
    """
    Derivative of Jacobi polynomial: d/dx P_N^{(alpha,beta)}(x).
    Uses the identity: dP_N/dx = sqrt(N*(N+alpha+beta+1)) * P_{N-1}^{(alpha+1,beta+1)}
    """
    if N == 0:
        return np.zeros_like(np.asarray(x, dtype=float))
    x = np.asarray(x, dtype=float)
    coeff = np.sqrt(N * (N + alpha + beta + 1.0))
    return coeff * jacobi_polynomial(x, alpha + 1.0, beta + 1.0, N - 1)


def vandermonde_1d(N, r):
    """
    Initialize 1D Vandermonde matrix V_{ij} = P_{j-1}(r_i).

    Parameters
    ----------
    N : int
        Polynomial order.
    r : ndarray
        Node coordinates in reference element [-1, 1].

    Returns
    -------
    V : ndarray, shape (len(r), N+1)
    """
    r = np.asarray(r, dtype=float)
    V = np.zeros((len(r), N + 1))
    for j in range(N + 1):
        V[:, j] = jacobi_polynomial(r, 0.0, 0.0, j)
    return V


def grad_vandermonde_1d(N, r):
    """
    Initialize gradient of Vandermonde matrix (V_r)_{ij} = dP_{j-1}/dr(r_i).
    """
    r = np.asarray(r, dtype=float)
    Vr = np.zeros((len(r), N + 1))
    for j in range(N + 1):
        Vr[:, j] = grad_jacobi_polynomial(r, 0.0, 0.0, j)
    return Vr


def differentiation_matrix_1d(N, r, V):
    """
    Compute 1D differentiation matrix D_r = V_r * V^{-1}.

    Parameters
    ----------
    N : int
        Order.
    r : ndarray
        Nodes.
    V : ndarray
        Vandermonde matrix.

    Returns
    -------
    Dr : ndarray
        Differentiation matrix.
    """
    Vr = grad_vandermonde_1d(N, r)
    # Solve V^T * Dr^T = Vr^T  =>  Dr = Vr * inv(V)
    # Use least squares for stability if V is near-singular
    try:
        Dr = Vr @ np.linalg.inv(V)
    except np.linalg.LinAlgError:
        Dr = Vr @ np.linalg.pinv(V)
    return Dr


def jacobi_gauss_lobatto(alpha, beta, N):
    """
    Compute Legendre-Gauss-Lobatto nodes (alpha=beta=0).
    These are the collocation points for spectral elements:
    x_0 = -1, x_N = 1, and the interior nodes are roots of P'_{N}(x).
    """
    if N == 0:
        return np.array([-1.0, 1.0])
    if N == 1:
        return np.array([-1.0, 0.0, 1.0])

    # Interior nodes: eigenvalues of Jacobi matrix with modified last row
    # For LGL: use Newton iteration on P_{N-1}^{(1,1)}(x)
    from math import cos, pi
    x = np.zeros(N + 1)
    x[0] = -1.0
    x[N] = 1.0

    # Initial guess from Chebyshev nodes
    for i in range(1, N):
        x[i] = -cos(pi * i / N)

    # Newton iteration for P'_{N}(x) = 0
    eps = 1e-14
    for _ in range(100):
        P = jacobi_polynomial(x[1:N], 1.0, 1.0, N - 1)
        dP = grad_jacobi_polynomial(x[1:N], 1.0, 1.0, N - 1)
        dx = -P / (dP + eps)
        x[1:N] += dx
        if np.max(np.abs(dx)) < eps:
            break

    return x


def vandermonde_interp_2d_matrix(n, m, x, y):
    """
    Compute 2D Vandermonde interpolation matrix for total degree M.

    Given n = T(M+1) data points (x_i, y_i), construct the matrix A such that
    A * c = z, where c are the polynomial coefficients and z are the data values.

    The polynomial basis is ordered lexicographically by total degree s:
    p(x,y) = sum_{s=0}^{M} sum_{ex=s}^{0} c_{ex, s-ex} x^{ex} y^{s-ex}

    Parameters
    ----------
    n : int
        Number of data points. Must equal T(M+1) = (M+1)(M+2)/2.
    m : int
        Total polynomial degree.
    x, y : ndarray, shape (n,)
        Data locations.

    Returns
    -------
    A : ndarray, shape (n, n)
        Vandermonde matrix.
    """
    x = np.asarray(x, dtype=float).flatten()
    y = np.asarray(y, dtype=float).flatten()
    if len(x) != n or len(y) != n:
        raise ValueError("x and y must have length n.")

    tmp1 = (m + 1) * (m + 2) // 2
    if n != tmp1:
        raise ValueError(f"For interpolation, need n = T(M+1) = {tmp1}, got n={n}")

    A = np.zeros((n, n))
    j = 0
    for s in range(m + 1):
        for ex in range(s, -1, -1):
            ey = s - ex
            A[:, j] = (x ** ex) * (y ** ey)
            j += 1
    return A


def polynomial_value_2d(n, c, m, x, y):
    """
    Evaluate 2D polynomial with coefficients c at points (x, y).

    Parameters
    ----------
    n : int
        Number of coefficients (= T(M+1)).
    c : ndarray, shape (n,)
        Coefficients in the same order as vandermonde_interp_2d_matrix.
    m : int
        Degree.
    x, y : float or ndarray
        Evaluation points.

    Returns
    -------
    value : ndarray
        Polynomial values.
    """
    c = np.asarray(c, dtype=float).flatten()
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    if len(c) != n:
        raise ValueError("Length of c must equal n.")

    value = np.zeros_like(x, dtype=float)
    j = 0
    for s in range(m + 1):
        for ex in range(s, -1, -1):
            ey = s - ex
            value += c[j] * (x ** ex) * (y ** ey)
            j += 1
    return value
