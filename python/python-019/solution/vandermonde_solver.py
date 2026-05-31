"""
vandermonde_solver.py
---------------------
Spectral interpolation and Vandermonde-system solvers for non-Hermitian
eigenvalue problems.

Adapted from seed project 1004_r8vm (Vandermonde matrix utilities).

Scientific Background
=====================
Given eigenvalues {E_j} of a non-Hermitian Hamiltonian sampled at
N parameter points {λ_j}, we often need to reconstruct the characteristic
polynomial or interpolate the energy band structure:

    p(E) = det(H(λ) - E I) = Π_{j=1}^{N} (E - E_j(λ)).

A Vandermonde system arises when fitting a polynomial to spectral data:

    V a = f,

where V_{ij} = λ_i^{j-1} and f_i = E(λ_i). Solving for coefficient
vector a gives the interpolating polynomial.

Vandermonde matrices are notoriously ill-conditioned. For complex nodes
{λ_j} lying near the unit circle or clustered around exceptional points,
the condition number grows exponentially with N. We implement the
Björck-Pereyra algorithm, which is O(N^2) and backward stable for
real, ordered nodes.

For non-Hermitian spectral interpolation near exceptional points, we
also provide a Lagrange interpolant based on barycentric coordinates,
which avoids explicit Vandermonde inversion.
"""

import numpy as np


def vandermonde_solve_bjorck_pereyra(nodes, rhs):
    """
    Solve the Vandermonde linear system V a = rhs using the
    Björck-Pereyra algorithm.

    V_{ij} = nodes_i^{j-1},  i,j = 0..N-1.

    Parameters
    ----------
    nodes : ndarray, shape (N,)
        Distinct nodes (real or complex).
    rhs : ndarray, shape (N,) or (N, M)
        Right-hand side(s).

    Returns
    -------
    a : ndarray
        Coefficient vector(s).
    info : int
        0 = success, 1 = singular (repeated nodes).
    """
    nodes = np.asarray(nodes).ravel()
    rhs = np.asarray(rhs)
    N = nodes.size

    if rhs.ndim == 1:
        rhs = rhs.reshape(-1, 1)
    if rhs.shape[0] != N:
        raise ValueError("rhs must have length equal to number of nodes.")

    # Check for repeated nodes
    for j in range(N - 1):
        for i in range(j + 1, N):
            if abs(nodes[i] - nodes[j]) < 1e-14:
                return np.zeros_like(rhs), 1

    x = rhs.astype(complex).copy()

    # Forward elimination
    for k in range(N - 1):
        for i in range(N - 1, k, -1):
            x[i, :] = x[i, :] - nodes[k] * x[i - 1, :]

    # Back substitution
    for k in range(N - 1, -1, -1):
        if k < N - 1:
            for i in range(k + 1, N):
                x[i, :] = x[i, :] / (nodes[i] - nodes[i - k - 1])
        for i in range(k, N - 1):
            x[i, :] = x[i, :] - x[i + 1, :]

    if rhs.shape[1] == 1:
        x = x.ravel()
    return x, 0


def vandermonde_determinant(nodes):
    """
    Compute det(V) = Π_{0≤j<i≤N-1} (x_i - x_j).
    """
    N = nodes.size
    det_val = 1.0 + 0.0j
    for j in range(N):
        for i in range(j + 1, N):
            det_val *= (nodes[i] - nodes[j])
    return det_val


def barycentric_lagrange_interpolate(nodes, values, z):
    """
    Barycentric Lagrange interpolation at points z.

    For nodes {x_j} and values {f_j}, the interpolant is

        p(z) = Σ_j (w_j f_j / (z - x_j)) / Σ_j (w_j / (z - x_j)),

    with barycentric weights w_j = 1 / Π_{k≠j} (x_j - x_k).

    Parameters
    ----------
    nodes : ndarray, shape (N,)
    values : ndarray, shape (N,)
    z : ndarray or float
        Evaluation points.

    Returns
    -------
    pz : ndarray
        Interpolated values.
    """
    nodes = np.asarray(nodes)
    values = np.asarray(values)
    z = np.asarray(z)
    N = nodes.size

    # Compute barycentric weights
    w = np.ones(N, dtype=complex)
    for j in range(N):
        for k in range(N):
            if k != j:
                w[j] /= (nodes[j] - nodes[k])

    z_flat = z.ravel()
    pz = np.zeros_like(z_flat, dtype=complex)

    for idx, zz in enumerate(z_flat):
        # Check if zz coincides with any node
        exact = np.isclose(zz, nodes)
        if np.any(exact):
            pz[idx] = values[np.argmax(exact)]
            continue

        num = np.sum(w * values / (zz - nodes))
        den = np.sum(w / (zz - nodes))
        if abs(den) < 1e-30:
            pz[idx] = np.nan
        else:
            pz[idx] = num / den

    return pz.reshape(z.shape)


def interpolate_energy_band(param_points, energy_points, eval_points):
    """
    Interpolate an energy band E(λ) from sampled data using the
    stable barycentric Lagrange formula.

    Parameters
    ----------
    param_points : ndarray
        Parameter values λ_j.
    energy_points : ndarray
        Corresponding energies E_j.
    eval_points : ndarray
        Points at which to evaluate the interpolant.

    Returns
    -------
    interpolated : ndarray
    """
    return barycentric_lagrange_interpolate(param_points, energy_points, eval_points)


def characteristic_polynomial_from_roots(roots):
    """
    Construct the monic characteristic polynomial coefficients from
    its roots using Vieta's formulas.

    p(E) = E^N + c_{N-1} E^{N-1} + ... + c_0
         = Π_{j=1}^{N} (E - E_j)

    Parameters
    ----------
    roots : ndarray
        Eigenvalues E_j.

    Returns
    -------
    coeffs : ndarray
        Coefficients [c_N, c_{N-1}, ..., c_0] with c_N = 1.
    """
    roots = np.asarray(roots)
    coeffs = [1.0]
    for r in roots:
        coeffs = np.convolve(coeffs, [1.0, -r])
    return coeffs
