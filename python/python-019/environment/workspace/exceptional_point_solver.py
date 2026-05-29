"""
exceptional_point_solver.py
---------------------------
Find exceptional points in non-Hermitian Hamiltonians.

Adapted from seed project 1430_zero_laguerre (Laguerre root-finding).

Scientific Background
=====================
Exceptional points (EPs) are singularities in the complex parameter plane
where two or more eigenvalues and their corresponding eigenvectors coalesce.
For a 2×2 Hamiltonian H(λ), the eigenvalues are

    E_±(λ) = (Tr H ± √(Δ(λ))) / 2,

with discriminant Δ(λ) = (Tr H)^2 - 4 det H.

An EP of order 2 occurs when Δ(λ_EP) = 0 and ∂_λ Δ(λ_EP) ≠ 0.
Higher-order EPs require higher-order zeros of Δ.

The Laguerre method is a cubically convergent root-finding algorithm
well-suited for complex polynomials:

    z_{n+1} = z_n - (m / (G ± √((m-1)(m H - G^2)))) * f(z_n)/f'(z_n)

where m is the polynomial degree, G = f'(z)/f(z), H = G^2 - f''(z)/f(z).
"""

import numpy as np
from hamiltonian_builder import (
    build_pt_symmetric_hamiltonian_1d,
    build_pt_symmetric_hamiltonian_2d,
    build_nonhermitian_ssh_hamiltonian,
    discriminant_2x2,
)


def laguerre_root_find(f, x0, degree, abserr=1e-12, kmax=100):
    """
    Laguerre root-finding method for complex or real functions.

    Parameters
    ----------
    f : callable
        f(x, ider) returns function value (ider=0), first derivative (ider=1),
        or second derivative (ider=2).
    x0 : complex or float
        Initial guess.
    degree : int
        Polynomial degree (must be >= 2).
    abserr : float
        Convergence tolerance.
    kmax : int
        Maximum iterations.

    Returns
    -------
    x : complex or float
        Estimated root.
    ierror : int
        0 if successful, 2 if max iterations exceeded, 3 if denominator vanishes.
    k : int
        Number of iterations performed.
    """
    if degree < 2:
        raise ValueError("degree must be at least 2.")
    x = x0
    ierror = 0
    k = 0
    beta = 1.0 / (degree - 1)

    while True:
        fx = f(x, 0)
        if abs(fx) <= abserr:
            break
        k += 1
        if k > kmax:
            ierror = 2
            return x, ierror, k

        dfx = f(x, 1)
        d2fx = f(x, 2)

        z = dfx ** 2 - (beta + 1.0) * fx * d2fx
        # Ensure the sqrt argument has non-negative real part for stability
        z = complex(z)
        if z.real < 0:
            # If negative real, we can still take principal sqrt
            pass
        bot = beta * dfx + np.sqrt(z)

        if abs(bot) < 1e-30:
            ierror = 3
            return x, ierror, k

        dx = - (beta + 1.0) * fx / bot
        x = x + dx

    return x, ierror, k


def find_exceptional_points_1d(t=1.0, m=0.5, gamma=0.3, k_guess_grid=64):
    """
    Find exceptional points in the 1D PT-symmetric Hamiltonian by
    scanning the complex k-plane and applying Laguerre refinement.

    The discriminant Δ(k) for H(k) = (m + t cos k) σ_z + t sin k σ_y + iγ σ_x
    is a function of k. EPs occur at Δ(k) = 0.

    Parameters
    ----------
    t, m, gamma : float
        Hamiltonian parameters.
    k_guess_grid : int
        Number of initial guesses on a grid in the complex plane.

    Returns
    -------
    ep_list : list of complex
        List of unique exceptional points found.
    """
    # Define discriminant function and its derivatives via finite differences
    def delta_func(k, ider=0):
        eps = 1e-8
        if ider == 0:
            H = build_pt_symmetric_hamiltonian_1d(k, t, m, gamma)
            return discriminant_2x2(H)
        elif ider == 1:
            Hp = build_pt_symmetric_hamiltonian_1d(k + eps, t, m, gamma)
            Hm = build_pt_symmetric_hamiltonian_1d(k - eps, t, m, gamma)
            return (discriminant_2x2(Hp) - discriminant_2x2(Hm)) / (2.0 * eps)
        elif ider == 2:
            Hp = build_pt_symmetric_hamiltonian_1d(k + eps, t, m, gamma)
            H0 = build_pt_symmetric_hamiltonian_1d(k, t, m, gamma)
            Hm = build_pt_symmetric_hamiltonian_1d(k - eps, t, m, gamma)
            return (discriminant_2x2(Hp) - 2.0 * discriminant_2x2(H0) + discriminant_2x2(Hm)) / (eps ** 2)
        else:
            raise ValueError("ider must be 0, 1, or 2.")

    # Scan complex plane grid for initial guesses
    re_vals = np.linspace(-np.pi, np.pi, k_guess_grid)
    im_vals = np.linspace(-2.0, 2.0, k_guess_grid // 2)

    roots_found = []
    tol_merge = 1e-6

    for re_k in re_vals:
        for im_k in im_vals:
            k0 = re_k + 1j * im_k
            root, ierr, _ = laguerre_root_find(delta_func, k0, degree=4, abserr=1e-14, kmax=80)
            if ierr == 0:
                # Check if root is new
                is_new = True
                for existing in roots_found:
                    if abs(root - existing) < tol_merge:
                        is_new = False
                        break
                if is_new:
                    roots_found.append(root)

    return roots_found


def find_exceptional_points_ssh(t1=1.0, t2=0.5, gamma=0.2, k_guess_grid=48):
    """
    Find exceptional points in the non-Hermitian SSH model.

    Parameters
    ----------
    t1, t2, gamma : float
        SSH parameters.
    k_guess_grid : int
        Grid density for initial guesses.

    Returns
    -------
    ep_list : list of complex
    """
    def delta_func(k, ider=0):
        # TODO: Implement the discriminant function and its finite-difference derivatives
        # for the non-Hermitian SSH model, then apply Laguerre root-finding.
        raise NotImplementedError("SSH discriminant function is missing.")

    re_vals = np.linspace(-np.pi, np.pi, k_guess_grid)
    im_vals = np.linspace(-1.5, 1.5, k_guess_grid // 2)
    roots_found = []
    tol_merge = 1e-6

    for re_k in re_vals:
        for im_k in im_vals:
            k0 = re_k + 1j * im_k
            root, ierr, _ = laguerre_root_find(delta_func, k0, degree=4, abserr=1e-14, kmax=80)
            if ierr == 0:
                is_new = True
                for existing in roots_found:
                    if abs(root - existing) < tol_merge:
                        is_new = False
                        break
                if is_new:
                    roots_found.append(root)

    return roots_found


def local_exceptional_point_order(H, param, dH_dparam, eps=1e-8):
    """
    Estimate the order of an exceptional point by checking the vanishing
    of successive derivatives of the discriminant.

    For a 2×2 Hamiltonian depending on parameter λ, near λ_EP we have
    Δ(λ) ≈ C (λ - λ_EP)^n, where n is the EP order.

    Parameters
    ----------
    H : ndarray, shape (2, 2)
        Hamiltonian at the EP.
    param : float
        Parameter value at the EP.
    dH_dparam : callable
        Function dH_dparam(p) returning ∂H/∂λ at parameter p.
    eps : float
        Finite-difference step.

    Returns
    -------
    order : int
        Estimated EP order (1 = no EP, >=2 = EP).
    """
    from hamiltonian_builder import discriminant_2x2

    dH = dH_dparam(param)
    # First derivative of discriminant
    delta0 = discriminant_2x2(H)
    Hp = H + eps * dH
    Hm = H - eps * dH
    delta1 = (discriminant_2x2(Hp) - discriminant_2x2(Hm)) / (2.0 * eps)

    if abs(delta0) > 1e-8:
        return 1  # Not an EP
    if abs(delta1) > 1e-6:
        return 2  # Generic second-order EP
    # Higher order would require more derivatives
    return 3
