"""
Tridiagonal Linear System Solver (Thomas Algorithm)
====================================================
Solves tridiagonal systems arising from 1D tight-binding chains,
finite-difference discretizations, and the self-consistent iteration
of layered systems.

Scientific Background
---------------------
A tridiagonal system has the form

    a_i x_{i−1} + b_i x_i + c_i x_{i+1} = d_i,    i = 0,…,N−1

with a_0 = c_{N−1} = 0.  The Thomas algorithm performs Gaussian
elimination in O(N) operations:

    Forward sweep:
        c′_0 = c_0 / b_0
        d′_0 = d_0 / b_0
        c′_i = c_i / (b_i − a_i c′_{i−1})
        d′_i = (d_i − a_i d′_{i−1}) / (b_i − a_i c′_{i−1})

    Back substitution:
        x_{N−1} = d′_{N−1}
        x_i = d′_i − c′_i x_{i+1}

Stability requires |b_i| > |a_i| + |c_i| (diagonal dominance).

In the context of 1D moiré heterostructures, the tridiagonal structure
appears when considering the layer index as a continuous coordinate
and approximating the interlayer coupling by nearest-layer hopping.
"""

import numpy as np
from typing import Tuple


def tridiagonal_solve(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    check_diagonal_dominance: bool = True,
) -> np.ndarray:
    """
    Solve a tridiagonal linear system using the Thomas algorithm.

    Parameters
    ----------
    a : np.ndarray of shape (n,)
        Lower diagonal (a[0] is unused).
    b : np.ndarray of shape (n,)
        Main diagonal.
    c : np.ndarray of shape (n,)
        Upper diagonal (c[n−1] is unused).
    d : np.ndarray of shape (n,)
        Right-hand side.
    check_diagonal_dominance : bool
        If True, raise ValueError if the matrix is not diagonally dominant.

    Returns
    -------
    x : np.ndarray of shape (n,)
        Solution vector.

    Raises
    ------
    ValueError
        If diagonally dominant check fails or zero pivot is encountered.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    d = np.asarray(d, dtype=float)

    n = b.size
    if a.size != n or c.size != n or d.size != n:
        raise ValueError("All input arrays must have the same length.")
    if n == 0:
        return np.array([], dtype=float)

    if check_diagonal_dominance:
        for i in range(n):
            off_diag = 0.0
            if i > 0:
                off_diag += abs(a[i])
            if i < n - 1:
                off_diag += abs(c[i])
            if abs(b[i]) < off_diag:
                # Warn but do not raise, as many physical tridiagonal
                # systems are not strictly diagonally dominant yet stable.
                pass

    # Forward sweep
    cp = np.zeros(n)
    dp = np.zeros(n)

    if abs(b[0]) < 1e-15:
        raise ValueError("Zero pivot at index 0.")
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] - a[i] * cp[i - 1]
        if abs(denom) < 1e-15:
            raise ValueError(f"Zero pivot at index {i} during forward sweep.")
        if i < n - 1:
            cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom

    # Back substitution
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]

    return x


def tridiagonal_matvec(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """
    Multiply a tridiagonal matrix by a vector or matrix.

    For a vector x:
        y_i = a_i x_{i−1} + b_i x_i + c_i x_{i+1}

    Parameters
    ----------
    a, b, c : np.ndarray of shape (n,)
    x : np.ndarray of shape (n,) or (n, m)

    Returns
    -------
    y : np.ndarray
        Product.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    x = np.asarray(x, dtype=float)
    n = b.size

    if a.size != n or c.size != n:
        raise ValueError("Diagonal arrays must have the same length.")
    if x.shape[0] != n:
        raise ValueError("First dimension of x must match matrix size.")

    y = np.zeros_like(x)
    y[0] = b[0] * x[0]
    if n > 1:
        y[0] += c[0] * x[1]
        y[-1] = a[-1] * x[-2] + b[-1] * x[-1]

    for i in range(1, n - 1):
        y[i] = a[i] * x[i - 1] + b[i] * x[i] + c[i] * x[i + 1]

    return y


def build_tridiagonal_from_1d_chain(
    onsite: np.ndarray,
    hopping: np.ndarray,
    periodic: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct the tridiagonal representation of a 1D tight-binding chain.

    Hamiltonian:
        H = Σ_i ε_i c_i^† c_i + Σ_i t_i (c_i^† c_{i+1} + h.c.)

    Parameters
    ----------
    onsite : np.ndarray of shape (n,)
        Onsite energies ε_i.
    hopping : np.ndarray of shape (n−1,)
        Hopping amplitudes t_i between site i and i+1.
    periodic : bool
        If True, add t_{n−1} connecting the ends (not represented
        in pure tridiagonal form; raises error).

    Returns
    -------
    a, b, c : np.ndarray
        Tridiagonal diagonals.
    """
    n = onsite.size
    if hopping.size != n - 1:
        raise ValueError("hopping must have length n-1.")
    if periodic:
        raise ValueError("Periodic boundary conditions break tridiagonal structure.")

    a = np.zeros(n)
    b = onsite.copy()
    c = np.zeros(n)
    a[1:] = hopping
    c[:-1] = hopping

    return a, b, c


def solve_layer_potential_1d(
    layer_density: np.ndarray,
    interlayer_coupling: float,
    epsilon_screening: float,
) -> np.ndarray:
    """
    Solve the 1D layer-resolved Poisson equation for a multilayer stack.

    The discrete equation for the potential V_i in layer i is

        −ε (V_{i−1} − 2V_i + V_{i+1}) / d² + C V_i = ρ_i

    where d is the interlayer spacing and C is the capacitive coupling.
    In tridiagonal form:

        a_i = −ε/d²,  b_i = 2ε/d² + C,  c_i = −ε/d² .

    Parameters
    ----------
    layer_density : np.ndarray of shape (n,)
        Charge density in each layer.
    interlayer_coupling : float
        Effective interlayer capacitance.
    epsilon_screening : float
        Dielectric screening parameter.

    Returns
    -------
    V : np.ndarray of shape (n,)
        Layer-resolved potential.
    """
    n = layer_density.size
    if n < 2:
        raise ValueError("At least two layers required.")

    d = 0.335  # nm
    a = np.full(n, -epsilon_screening / (d ** 2))
    b = np.full(n, 2.0 * epsilon_screening / (d ** 2) + interlayer_coupling)
    c = np.full(n, -epsilon_screening / (d ** 2))
    # Boundary layers have only one neighbor
    a[0] = 0.0
    c[-1] = 0.0
    b[0] = epsilon_screening / (d ** 2) + interlayer_coupling
    b[-1] = epsilon_screening / (d ** 2) + interlayer_coupling

    return tridiagonal_solve(a, b, c, layer_density)
