"""
protein_diffusion.py
Smoluchowski dynamics and nonlinear diffusion for repair protein search.

Derived from: 818_normal_ode + 901_porous_medium_exact

Models the diffusive search of DNA repair proteins along the DNA contour
and in the surrounding nucleoplasm. The search process combines:

  1. 1D sliding along DNA (described by Smoluchowski equation on a line)
  2. 3D volume diffusion with nonlinear crowding (porous-medium type)

Core equations:

  Smoluchowski (1D sliding):
      dP/dt = D_s * d^2P/dx^2 - (D_s/(k_B*T)) * d/dx(F(x)*P)
  where F(x) = -dU/dx is the DNA-protein interaction force,
  D_s is the 1D sliding diffusion coefficient.

  Porous-medium (3D crowding):
      dC/dt = nabla^2(C^m) + S(x,t)
  where m > 1 accounts for excluded-volume effects in the crowded
  nucleoplasm, and S(x,t) is a source term representing protein
  recruitment to damage foci.

  Normal-mode ODE (probability relaxation):
      dP/dt = -D * k^2 * P
  exact solution: P(t) = P(0) * exp(-D*k^2*t)
"""

import numpy as np


def smoluchowski_1d_sliding_step(P, U, D_s, kT, dt, dx, boundary="reflecting"):
    """
    Advance the probability density P(x,t) of a repair protein sliding
    along DNA by one time step using the Smoluchowski equation.

    Discretization (Crank-Nicolson for diffusion, explicit for drift):
        P_i^{n+1} = P_i^n + dt*D_s/dx^2 * (P_{i+1} - 2P_i + P_{i-1})
                    - dt/(2*dx*kT) * [F_{i+1/2}*(P_{i+1}+P_i) - F_{i-1/2}*(P_i+P_{i-1})]

    Parameters
    ----------
    P : ndarray, shape (N,)
        Probability density at current time.
    U : ndarray, shape (N,)
        Potential energy landscape along DNA (kT units).
    D_s : float
        1D sliding diffusion coefficient (nm^2/us).
    kT : float
        Thermal energy (k_B*T in consistent units).
    dt : float
        Time step.
    dx : float
        Spatial grid spacing.
    boundary : str
        'reflecting' or 'absorbing'.

    Returns
    -------
    P_new : ndarray, shape (N,)
    """
    N = len(P)
    if N < 3:
        return P.copy()

    P = np.asarray(P, dtype=float)
    U = np.asarray(U, dtype=float)

    # Force at half-grid points: F = -dU/dx
    F_half = np.zeros(N - 1)
    for i in range(N - 1):
        dU = U[i + 1] - U[i]
        F_half[i] = -dU / dx

    P_new = np.zeros(N)
    alpha = D_s * dt / (dx * dx)

    for i in range(N):
        # Diffusion term
        if i == 0:
            P_ip1 = P[i + 1]
            P_im1 = P[i + 1] if boundary == "reflecting" else 0.0
        elif i == N - 1:
            P_ip1 = P[i - 1] if boundary == "reflecting" else 0.0
            P_im1 = P[i - 1]
        else:
            P_ip1 = P[i + 1]
            P_im1 = P[i - 1]

        diff = alpha * (P_ip1 - 2.0 * P[i] + P_im1)

        # Drift term (upwind-like)
        drift = 0.0
        if i < N - 1:
            drift -= (dt / (2.0 * dx * kT)) * F_half[i] * (P[i + 1] + P[i])
        if i > 0:
            drift += (dt / (2.0 * dx * kT)) * F_half[i - 1] * (P[i] + P[i - 1])

        P_new[i] = P[i] + diff + drift

    # Normalize and clip negative probabilities (numerical robustness)
    P_new = np.maximum(P_new, 0.0)
    total = np.sum(P_new)
    if total > 0:
        P_new /= total
    else:
        P_new = np.ones(N) / N

    return P_new


def porous_medium_step_1d(C, m, dt, dx, source=None, boundary="neumann"):
    """
    Advance the concentration C(x,t) under the porous-medium equation
    with nonlinear diffusion coefficient D(C) = m * C^{m-1}:

        dC/dt = d^2(C^m)/dx^2 + S(x,t)

    Discretized via finite differences:
        C_i^{n+1} = C_i^n + dt/dx^2 * [(C_{i+1}^m - 2*C_i^m + C_{i-1}^m)] + dt*S_i

    Parameters
    ----------
    C : ndarray, shape (N,)
        Concentration.
    m : float
        Porous-medium exponent (m > 1 for crowding).
    dt : float
        Time step.
    dx : float
        Grid spacing.
    source : ndarray, shape (N,), optional
        Source term.
    boundary : str
        'neumann' or 'dirichlet'.

    Returns
    -------
    C_new : ndarray, shape (N,)
    """
    N = len(C)
    if N < 3:
        return C.copy()

    C = np.maximum(np.asarray(C, dtype=float), 0.0)
    Cm = C ** m

    C_new = np.zeros(N)
    coeff = dt / (dx * dx)

    for i in range(N):
        if i == 0:
            Cm_im1 = Cm[1] if boundary == "neumann" else 0.0
            Cm_ip1 = Cm[1]
        elif i == N - 1:
            Cm_im1 = Cm[N - 2]
            Cm_ip1 = Cm[N - 2] if boundary == "neumann" else 0.0
        else:
            Cm_im1 = Cm[i - 1]
            Cm_ip1 = Cm[i + 1]

        laplace = Cm_ip1 - 2.0 * Cm[i] + Cm_im1
        C_new[i] = C[i] + coeff * laplace

    if source is not None:
        C_new += dt * np.asarray(source)

    # Robustness: clamp negative concentrations
    C_new = np.maximum(C_new, 0.0)
    return C_new


def porous_medium_barenblatt_solution(x, t, m, C0=1.0, delta=0.01):
    """
    Evaluate the Barenblatt self-similar solution to the porous-medium equation.

    For 1D:
        C(x,t) = (t+delta)^{-beta} * max(0, A - gamma*(x/(t+delta)^{beta})^2)^{alpha}

    where:
        alpha = 1/(m-1)
        beta  = 1/(m+1)
        gamma = (m-1)/(2*m*(m+1))

    Parameters
    ----------
    x : ndarray
        Positions.
    t : float
        Time.
    m : float
        Exponent.
    C0 : float
        Amplitude parameter.
    delta : float
        Time offset.

    Returns
    -------
    C : ndarray
    """
    alpha = 1.0 / (m - 1.0)
    beta = 1.0 / (m + 1.0)
    gamma = (m - 1.0) / (2.0 * m * (m + 1.0))

    bot = (t + delta) ** beta
    A = C0
    factor = A - gamma * (x / bot) ** 2
    C = np.zeros_like(x, dtype=float)
    mask = factor > 0
    # Safe power computation only on positive factors
    with np.errstate(invalid="ignore"):
        C[mask] = (t + delta) ** (-beta) * factor[mask] ** alpha
    return C


def normal_mode_relaxation(P0, D, k, t):
    """
    Exact solution for probability relaxation in a harmonic potential:

        dP/dt = -D * k^2 * P
        P(t) = P0 * exp(-D * k^2 * t)

    Parameters
    ----------
    P0 : float
        Initial probability.
    D : float
        Diffusion coefficient.
    k : float
        Wave number (inverse length).
    t : float or ndarray
        Time.

    Returns
    -------
    P : float or ndarray
    """
    return P0 * np.exp(-D * k * k * t)


def sliding_search_time(dna_length, D_s, target_size):
    """
    Theoretical mean first-passage time for a protein sliding along
    DNA to find a target of size 'target_size' on a DNA of length L.

    For 1D diffusion on a line with reflecting boundaries:
        <T> = L^2 / (3 * D_s)   for uniformly distributed target
    For a target of size a << L:
        <T> ~ L^2 / (2 * D_s) * (1 - a/L)^2

    Parameters
    ----------
    dna_length : float
        DNA contour length.
    D_s : float
        Sliding diffusion coefficient.
    target_size : float
        Size of the damage site.

    Returns
    -------
    mean_time : float
        Mean first-passage time.
    """
    if dna_length <= 0 or D_s <= 0:
        return float("inf")
    a = target_size / dna_length
    if a >= 1.0:
        return 0.0
    mean_time = (dna_length ** 2) / (2.0 * D_s) * (1.0 - a) ** 2
    return mean_time
