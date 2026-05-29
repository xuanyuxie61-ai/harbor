# -*- coding: utf-8 -*-
"""
radial_solver.py
================
Spectral-element / collocation solver for the radial Schrödinger equation
in spherical nuclei.

This module fuses the **1-D spectral FEM** philosophy of
*fem1d_spectral_numeric* with the **Vandermonde-based quadrature**
mechanism of *quadrature_weights_vandermonde*.

Physical model
--------------
The radial Schrödinger equation for orbital angular momentum :math:`l` is

.. math::
    -\frac{\hbar^2}{2m}\frac{d^2 u}{dr^2}
    + \left[V(r) + \frac{\hbar^2}{2m}\frac{l(l+1)}{r^2}\right] u
    = E\,u \;,

where :math:`u(r)=rR(r)` is the reduced radial wave function.

Discretisation
--------------
On a finite interval :math:`[0, r_{\max}]` we introduce :math:`N`
collocation points :math:`\{x_i\}_{i=1}^{N}` (Gauss-Lobatto-Legendre
nodes mapped to the physical domain).  The unknown solution is expanded
in Lagrange cardinal polynomials :math:`\phi_j(x)`:

.. math::
    u(x) \approx \sum_{j=1}^{N} c_j\,\phi_j(x) \;.

The stiffness matrix :math:`A` and mass matrix :math:`B` are

.. math::
    A_{ij} = \int_0^{r_{\max}} \!
    \left[\frac{\hbar^2}{2m}\phi_i'(x)\phi_j'(x)
    + V_{\text{eff}}(x)\,\phi_i(x)\phi_j(x)\right] dx \;,

.. math::
    B_{ij} = \int_0^{r_{\max}} \phi_i(x)\phi_j(x)\,dx \;.

The generalised eigenvalue problem :math:`A c = E B c` is solved
for the :math:`N-2` interior degrees of freedom after imposing the
Dirichlet boundary condition :math:`u(0)=u(r_{\max})=0`.
"""

import numpy as np
from scipy.special import legendre
from constants import hbar2_over_2m


def gauss_lobatto_nodes(N, a=-1.0, b=1.0):
    r"""
    Gauss-Lobatto-Legendre quadrature nodes on :math:`[a,b]`.

    The :math:`N` nodes on :math:`[-1,1]` are
    :math:`x_0=-1, x_{N-1}=1` and the remaining :math:`N-2` roots of
    :math:`P'_{N-1}(x)`, where :math:`P_{N-1}` is the Legendre polynomial.

    Parameters
    ----------
    N : int
        Number of nodes (>= 2).
    a, b : float
        Physical interval endpoints.

    Returns
    -------
    x : ndarray, shape (N,)
        Nodes in :math:`[a,b]`.
    """
    if N < 2:
        raise ValueError("gauss_lobatto_nodes requires N >= 2.")
    # Legendre polynomial of degree N-1
    P = legendre(N - 1)
    # Derivative roots on (-1,1)
    dp = np.polyder(P)
    inner = np.roots(dp)
    # Keep only real roots inside (-1,1)
    inner = np.real(inner[np.isreal(inner)])
    inner = inner[(inner > -1.0) & (inner < 1.0)]
    inner = np.sort(inner)
    # Should have N-2 roots; if not, fall back to Chebyshev
    if inner.size < N - 2:
        inner = -np.cos(np.pi * np.arange(1, N - 1) / (N - 1))
    x = np.concatenate([[-1.0], inner[:N - 2], [1.0]])
    x = np.sort(x)
    # Map to [a,b]
    x = 0.5 * (b - a) * x + 0.5 * (b + a)
    return x


def gauss_lobatto_weights(N):
    r"""
    Gauss-Lobatto-Legendre quadrature weights on :math:`[-1,1]`.

    The weights are

    .. math::
        w_i = \frac{2}{N(N-1)\,[P_{N-1}(x_i)]^2}\;,
        \qquad x_i = \pm 1 \;,

    with the same formula valid for interior nodes.

    Parameters
    ----------
    N : int
        Number of nodes.

    Returns
    -------
    w : ndarray, shape (N,)
        Quadrature weights.
    """
    x = gauss_lobatto_nodes(N, -1.0, 1.0)
    P = legendre(N - 1)
    pval = np.polyval(P, x)
    w = 2.0 / (N * (N - 1) * (pval ** 2))
    # End-point correction for numerical stability
    w[0] = 2.0 / (N * (N - 1))
    w[-1] = 2.0 / (N * (N - 1))
    return w


def vandermonde_quadrature_weights(x, a, b):
    r"""
    Compute quadrature weights by solving a Vandermonde system.

    Given nodes :math:`\{x_i\}_{i=1}^{N}` on :math:`[a,b]`, find weights
    :math:`\{w_i\}` such that

    .. math::
        \sum_{i=1}^{N} w_i\,x_i^{k-1}
        = \int_a^b x^{k-1}\,dx
        = \frac{b^{k} - a^{k}}{k}\;,
        \qquad k = 1,\dots,N \;.

    This yields the linear system :math:`V^{\!T} w = r` with

    .. math::
        V_{ki} = x_i^{k-1}\;,\qquad
        r_k = \frac{b^{k} - a^{k}}{k}\;.

    Parameters
    ----------
    x : ndarray, shape (N,)
        Quadrature nodes.
    a, b : float
        Interval endpoints.

    Returns
    -------
    w : ndarray, shape (N,)
        Quadrature weights.
    """
    N = x.size
    V = np.vander(x, N, increasing=True).T  # V_{k,i} = x_i^{k-1}
    rhs = np.empty(N)
    for k in range(1, N + 1):
        rhs[k - 1] = (b ** k - a ** k) / k
    # Solve V^T w = rhs  =>  w = (V^T)^{-1} rhs
    w = np.linalg.solve(V.T, rhs)
    return w


def lagrange_derivative_matrix(x):
    r"""
    Spectral differentiation matrix :math:`D` for nodes :math:`\{x_i\}`.

    .. math::
        D_{ij} = \begin{cases}
        \displaystyle\frac{\lambda_i}{\lambda_j}\frac{1}{x_i-x_j}, & i\neq j \\[8pt]
        -\displaystyle\sum_{k\neq i} D_{ik}, & i=j
        \end{cases}

    with :math:`\lambda_i = \prod_{k\neq i}(x_i-x_k)^{-1}`.

    Parameters
    ----------
    x : ndarray, shape (N,)
        Distinct nodes.

    Returns
    -------
    D : ndarray, shape (N, N)
        Differentiation matrix.
    """
    N = x.size
    D = np.zeros((N, N))
    # Barycentric weights
    lam = np.ones(N)
    for i in range(N):
        for k in range(N):
            if k != i:
                lam[i] *= 1.0 / (x[i] - x[k])
    for i in range(N):
        for j in range(N):
            if i != j:
                D[i, j] = (lam[i] / lam[j]) / (x[i] - x[j])
    # Diagonal by sum rule (negative sum of off-diagonals)
    for i in range(N):
        D[i, i] = -np.sum(D[i, :])
    return D


def solve_radial_schroedinger(rmax, N, l, V_func, n_eig=5,
                              method='gll', mass_nucleon=None):
    r"""
    Solve the radial Schrödinger equation on :math:`[0, r_{\max}]`.

    Uses a **symmetric finite-difference** discretisation (Numerov-like
    accuracy) on a uniform mesh, which is numerically stable for nuclear
    potentials with centrifugal barriers.

    The discretised Hamiltonian on a uniform grid :math:`r_i = i h` is

    .. math::
        H_{ii}     &= \frac{2\hbar^2}{2m h^2} + V_{\text{eff}}(r_i) \;,
        \quad i = 1,\dots,N-2 \\
        H_{i,i+1}  &= H_{i+1,i} = -\frac{\hbar^2}{2m h^2} \;.

    Dirichlet boundary conditions :math:`u(0)=u(r_{\max})=0` are imposed
    by omitting the boundary points from the matrix.

    Parameters
    ----------
    rmax : float
        Maximum radius in fm.
    N : int
        Number of mesh points.
    l : int
        Orbital angular momentum.
    V_func : callable
        Potential function :math:`V(r)` in MeV.
    n_eig : int
        Number of lowest eigenvalues/vectors to return.
    method : {'gll', 'vandermonde'}
        Ignored in the current stable implementation; kept for API compatibility.
    mass_nucleon : float, optional
        Nucleon mass in MeV/c².

    Returns
    -------
    energies : ndarray, shape (n_eig,)
        Bound-state eigenvalues in MeV.
    wavefunctions : ndarray, shape (N, n_eig)
        Radial wave functions :math:`u(r)` at the mesh points.
    r : ndarray, shape (N,)
        Uniform radial mesh.
    """
    if rmax <= 0:
        raise ValueError("rmax must be positive.")
    if N < 4:
        raise ValueError("N must be at least 4.")
    if l < 0:
        raise ValueError("l must be non-negative.")

    # Uniform mesh on [0, rmax]
    r = np.linspace(0.0, rmax, N)
    h = r[1] - r[0]

    if mass_nucleon is None:
        h2m = hbar2_over_2m()
    else:
        h2m = hbar2_over_2m(mass_nucleon)

    # HOLE 1: Implement the effective-potential construction and Hamiltonian
    # matrix assembly for the radial Schrödinger equation.
    # Steps:
    #   1. Compute the effective potential V_eff(r) = V_func(r) + centrifugal,
    #      where centrifugal = h2m * l * (l + 1) / r^2  (protect r=0).
    #   2. Build a symmetric tridiagonal Hamiltonian for interior points
    #      i = 1 .. N-2 (Dirichlet boundaries u(0)=u(rmax)=0 are omitted):
    #        main_diag[i]  = 2*coeff + V_eff[i+1]
    #        off_diag[i]   = -coeff
    #      with coeff = h2m / h^2.
    #   3. Solve the eigenvalue problem H c = E c.
    #   4. Re-insert boundary zeros into the eigenvectors.
    #   5. Normalise each wavefunction so that ∫|u|^2 dr = 1.
    #   6. Return the lowest n_eig bound states (E < 0), or lowest continuum
    #      states if no bound states exist.
    raise NotImplementedError("HOLE 1: radial Schrödinger solver core is not implemented.")

    # Re-insert boundary zeros
    wavefunctions = np.zeros((N, n_int))
    wavefunctions[1:N - 1, :] = eigvecs

    # Normalise wavefunctions:  ∫|u|^2 dr = 1
    for k in range(n_int):
        norm = np.sqrt(np.trapz(wavefunctions[:, k] ** 2, r))
        if norm > 0:
            wavefunctions[:, k] /= norm

    # Return only bound states (E < 0) up to n_eig
    bound_mask = eigvals < 0.0
    bound_vals = eigvals[bound_mask]
    bound_wf = wavefunctions[:, bound_mask]

    n_return = min(n_eig, bound_vals.size)
    if n_return == 0:
        # If no bound states, return lowest continuum states
        n_return = min(n_eig, eigvals.size)
        return eigvals[:n_return], wavefunctions[:, :n_return], r

    return bound_vals[:n_return], bound_wf[:, :n_return], r


def radial_matrix_element(r, u1, u2, operator=None):
    r"""
    Compute radial matrix element
    :math:`\langle u_1 | \hat{O} | u_2 \rangle = \int_0^{\infty} u_1(r)
    \,O(r)\,u_2(r)\,dr`.

    Parameters
    ----------
    r : ndarray
        Radial mesh.
    u1, u2 : ndarray
        Radial wave functions (same length as r).
    operator : callable or None
        Operator :math:`O(r)`.  If None, unity (overlap).

    Returns
    -------
    me : float
        Matrix element.
    """
    if operator is None:
        integrand = u1 * u2
    else:
        integrand = u1 * operator(r) * u2
    return np.trapz(integrand, r)


def kinetic_energy_matrix_element(r, u1, u2, l, mass_nucleon=None):
    r"""
    Kinetic-energy matrix element in the radial basis.

    .. math::
        T_{ij} = \int_0^{\infty} u_i(r)
        \left[-\frac{\hbar^2}{2m}\frac{d^2}{dr^2}
        + \frac{\hbar^2 l(l+1)}{2m r^2}\right] u_j(r)\,dr

    Parameters
    ----------
    r : ndarray
        Radial mesh.
    u1, u2 : ndarray
        Wave functions.
    l : int
        Orbital angular momentum.
    mass_nucleon : float, optional

    Returns
    -------
    T : float
        Kinetic matrix element in MeV.
    """
    h2m = hbar2_over_2m(mass_nucleon)
    # Second derivative via finite differences (5-point stencil)
    d2u2 = np.zeros_like(u2)
    h = np.diff(r)
    # Variable spacing: use uniform approx if nearly uniform
    h_avg = np.mean(h)
    if np.max(np.abs(h - h_avg)) < 0.1 * h_avg:
        # Uniform 5-point stencil
        h2 = h_avg ** 2
        d2u2[2:-2] = (-u2[:-4] + 16.0 * u2[1:-3]
                      - 30.0 * u2[2:-2]
                      + 16.0 * u2[3:-1] - u2[4:]) / (12.0 * h2)
        # Boundaries: 3-point
        d2u2[0] = (2.0 * u2[0] - 5.0 * u2[1] + 4.0 * u2[2] - u2[3]) / h2
        d2u2[1] = (u2[0] - 2.0 * u2[1] + u2[2]) / h2
        d2u2[-2] = (u2[-3] - 2.0 * u2[-2] + u2[-1]) / h2
        d2u2[-1] = (2.0 * u2[-1] - 5.0 * u2[-2]
                    + 4.0 * u2[-3] - u2[-4]) / h2
    else:
        # Non-uniform: simple 3-point stencil
        for i in range(1, r.size - 1):
            hp = r[i + 1] - r[i]
            hm = r[i] - r[i - 1]
            d2u2[i] = 2.0 * (hm * u2[i + 1] - (hp + hm) * u2[i]
                             + hp * u2[i - 1]) / (hp * hm * (hp + hm))
        d2u2[0] = d2u2[1]
        d2u2[-1] = d2u2[-2]

    r_safe = np.where(r < 1e-6, 1e-6, r)
    centrifugal = h2m * l * (l + 1) / (r_safe ** 2)
    integrand = u1 * (-h2m * d2u2 + centrifugal * u2)
    return np.trapz(integrand, r)
