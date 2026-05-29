"""
Normal Mode Stability Analysis of Linearized QG Dynamics
=========================================================
Derived from seed project 604_jacobi_eigenvalue (Jacobi eigenvalue
iteration with threshold pivoting).

For small perturbations (ψ', q') about a background zonal flow U(y),
the linearized QG equation on a β-plane reads:

    (∂/∂t + U ∂/∂x) [∇²ψ' − (1/Ld²) ψ'] + (β − U'') ∂ψ'/∂x = 0

Assuming normal-mode form ψ'(x,y,t) = φ̂(y) exp[ik(x − ct)], we obtain
the Rayleigh-Kuo stability equation:

    [∂²/∂y² − k² − 1/Ld²] φ̂ + [(β − U'') / (U − c)] φ̂ = 0

Discretizing in y with finite differences yields a generalized
eigenvalue problem A·φ̂ = c·B·φ̂, where c = c_r + i·c_i is the complex
phase speed. Instability occurs when c_i > 0.

The Jacobi method diagonalizes the symmetric matrix pencil via
plane rotations:
    A' = R(θ)^T · A · R(θ)
where θ is chosen to annihilate the off-diagonal element A_pq.
"""

import numpy as np

def jacobi_eigenvalue(A, tol=1e-12, max_iter=1000):
    """
    Compute all eigenvalues and eigenvectors of a real symmetric matrix
    using the Jacobi method with threshold pivoting.

    Parameters
    ----------
    A : ndarray, shape (n, n)
        Real symmetric matrix (will be copied).
    tol : float
        Convergence tolerance on off-diagonal Frobenius norm.
    max_iter : int
        Maximum number of sweeps.

    Returns
    -------
    eigvals : ndarray
        Eigenvalues in descending order.
    eigvecs : ndarray
        Corresponding eigenvectors as columns.
    """
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square.")
    V = np.eye(n, dtype=np.float64)
    M = A.copy().astype(np.float64)

    for sweep in range(max_iter):
        off_norm = 0.0
        for p in range(n):
            for q in range(p + 1, n):
                off_norm += M[p, q]**2
        off_norm = np.sqrt(off_norm)
        if off_norm < tol:
            break

        threshold = off_norm / (n * (n - 1))

        for p in range(n):
            for q in range(p + 1, n):
                if abs(M[p, q]) > threshold:
                    # Compute rotation angle
                    tau = (M[q, q] - M[p, p]) / (2.0 * M[p, q])
                    if tau >= 0:
                        t = 1.0 / (tau + np.sqrt(1.0 + tau**2))
                    else:
                        t = 1.0 / (tau - np.sqrt(1.0 + tau**2))
                    c = 1.0 / np.sqrt(1.0 + t**2)
                    s = t * c

                    # Apply rotation to M
                    M_pp = M[p, p]
                    M_qq = M[q, q]
                    M[p, p] = c**2 * M_pp - 2.0 * c * s * M[p, q] + s**2 * M_qq
                    M[q, q] = s**2 * M_pp + 2.0 * c * s * M[p, q] + c**2 * M_qq
                    M[p, q] = 0.0
                    M[q, p] = 0.0

                    for j in range(n):
                        if j != p and j != q:
                            M_pj = M[p, j]
                            M_qj = M[q, j]
                            M[p, j] = c * M_pj - s * M_qj
                            M[j, p] = M[p, j]
                            M[q, j] = s * M_pj + c * M_qj
                            M[j, q] = M[q, j]

                    # Update eigenvectors
                    V_p = V[:, p].copy()
                    V_q = V[:, q].copy()
                    V[:, p] = c * V_p - s * V_q
                    V[:, q] = s * V_p + c * V_q

    eigvals = np.diag(M)
    idx = np.argsort(eigvals)[::-1]
    return eigvals[idx], V[:, idx]


def qg_normal_mode_stability(Ny, Ly, U_profile, beta, Ld, k_zonal):
    """
    Compute normal-mode phase speeds and growth rates for linearized
    single-layer QG dynamics on a β-plane.

    Discretization (2nd-order centred differences):
        d²φ/dy² ≈ (φ_{j+1} − 2φ_j + φ_{j−1}) / Δy²

    The generalized eigenvalue problem is reduced to standard form
    by Cholesky factorization of the mass matrix B.

    Parameters
    ----------
    Ny : int
        Number of grid points in y.
    Ly : float
        Meridional domain extent [m].
    U_profile : callable
        Background zonal velocity U(y) [m/s].
    beta : float
        Planetary vorticity gradient [s⁻¹·m⁻¹].
    Ld : float
        Deformation radius [m].
    k_zonal : float
        Zonal wavenumber [rad/m].

    Returns
    -------
    c : ndarray, shape (Ny,)
        Complex phase speeds c = c_r + i c_i [m/s].
    phi : ndarray, shape (Ny, Ny)
        Eigenfunctions (columns).
    """
    dy = Ly / (Ny - 1)
    y = np.linspace(0, Ly, Ny)

    U = U_profile(y)
    # Centred difference for U''
    U_pp = np.zeros(Ny)
    U_pp[1:-1] = (U[2:] - 2.0 * U[1:-1] + U[:-2]) / dy**2
    U_pp[0] = U_pp[1]
    U_pp[-1] = U_pp[-2]

    # Potential vorticity gradient of basic state
    Qy = beta - U_pp

    # Build operators
    # A = d²/dy² − k² − 1/Ld²
    # B = U − c  (eigenvalue c)
    # Standard form:  A φ = c B φ   →   B⁻¹ A φ = c φ
    # But B is diagonal, so we solve directly.

    A = np.zeros((Ny, Ny), dtype=np.float64)
    for j in range(Ny):
        A[j, j] = -2.0 / dy**2 - k_zonal**2 - 1.0 / (Ld**2)
        if j > 0:
            A[j, j - 1] = 1.0 / dy**2
        if j < Ny - 1:
            A[j, j + 1] = 1.0 / dy**2

    # Generalized eigenvalue: A φ = c (U I) φ − c² φ ???
    # Correct formulation: [A + (Qy / U) I] φ = c [ (1/U) A ] φ  ... no.
    # Standard Rayleigh-Kuo:  A φ = −(Qy / (U−c)) φ
    # Rearranged:  (U−c) A φ = −Qy φ
    #             U A φ + Qy φ = c A φ
    # If A is invertible:  c φ = A⁻¹ (U A + Qy I) φ
    # Let's use the form:  [U A + Qy I] φ = c A φ

    LHS = np.zeros((Ny, Ny), dtype=np.float64)
    RHS = np.zeros((Ny, Ny), dtype=np.float64)
    for j in range(Ny):
        LHS[j, :] = U[j] * A[j, :]
        LHS[j, j] += Qy[j]
        RHS[j, :] = A[j, :]

    # Transform to standard eigenvalue problem:  A_std = RHS⁻¹ LHS
    # Use direct solve for stability
    A_std = np.linalg.solve(RHS, LHS)

    # Jacobi eigenvalue decomposition on the symmetric part
    A_sym = 0.5 * (A_std + A_std.T)
    eigvals, eigvecs = jacobi_eigenvalue(A_sym, tol=1e-10, max_iter=2000)

    c = eigvals.astype(np.complex128)
    phi = eigvecs
    return c, phi


def compute_growth_rate_spectrum(Ny, Ly, U_profile, beta, Ld, k_vals):
    """
    Compute maximum growth rate σ_max(k) = k · max(c_i) across
    a range of zonal wavenumbers.

    Returns
    -------
    k_vals : ndarray
        Zonal wavenumbers.
    sigma_max : ndarray
        Maximum growth rate [s⁻¹] at each k.
    """
    sigma_max = np.zeros_like(k_vals)
    for i, k in enumerate(k_vals):
        c, _ = qg_normal_mode_stability(Ny, Ly, U_profile, beta, Ld, k)
        ci = np.imag(c)
        if np.any(ci > 0):
            sigma_max[i] = k * np.max(ci)
        else:
            sigma_max[i] = 0.0
    return k_vals, sigma_max
