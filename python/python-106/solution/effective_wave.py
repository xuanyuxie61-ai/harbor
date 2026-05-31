"""
effective_wave.py
=================
1D effective-medium wave equation solver for plasmonic waveguides.

For a 1D metal-insulator-metal (MIM) waveguide with effective permittivity
ε_eff(x), the transverse-magnetic (TM) mode obeys the Helmholtz-like equation:

    d²H_y/dx² + k₀² ε_eff(x) H_y = β² H_y

where k₀ = ω/c and β is the propagation constant.  Discretizing on a uniform
grid x_i with spacing h gives the finite-difference stencil:

    (H_{i-1} − 2 H_i + H_{i+1}) / h² + k₀² ε_i H_i = β² H_i

Rearranging, this is a generalized eigenvalue problem A H = β² H with a
trilidiagonal matrix A.  For propagation-loss analysis we solve:

    A H = λ H,   λ = β²

The matrix A in tridiagonal form (for interior points) is:
    sub-diag:   1/h²
    diag:      −2/h² + k₀² ε_i
    super-diag: 1/h²

We use the conjugate-gradient method adapted from the r83s_cg seed for
solving the linear system (A − λI) H = source when a specific β is sought.

For lossy metals, ε_eff is complex, making A non-Hermitian.  We restrict
the present solver to real-symmetric cases (lossless approximation) or
use the real part of ε_eff.
"""

import numpy as np


def r83s_matvec(n, a, x):
    """
    Multiply a symmetric tridiagonal scalar matrix (R83S format) by a vector.

    Matrix structure (constant sub, diag, super diagonals):
        [ a[1]  a[0]    0      0   ... ]
        [ a[2]  a[1]   a[0]    0   ... ]
        [  0    a[2]   a[1]  a[0] ... ]
        ...

    Parameters
    ----------
    n : int
    a : ndarray, shape (3,)
        [sub, diag, super].
    x : ndarray, shape (n,)

    Returns
    -------
    b : ndarray, shape (n,)
    """
    b = np.zeros(n, dtype=float)
    for j in range(n):
        i_start = max(0, j - 1)
        i_end = min(n, j + 2)
        for i in range(i_start, i_end):
            b[i] += a[i - j + 1] * x[j]
    return b


def conjugate_gradient_r83s(n, a, b, x0, tol=1e-12, max_iter=None):
    """
    Solve A x = b for a symmetric positive-definite R83S tridiagonal matrix
    using the conjugate gradient method.

    Parameters
    ----------
    n : int
    a : ndarray, shape (3,)
    b : ndarray, shape (n,)
    x0 : ndarray, shape (n,)
    tol : float
    max_iter : int or None

    Returns
    -------
    x : ndarray
    """
    if max_iter is None:
        max_iter = n
    x = x0.astype(float).copy()
    b = b.astype(float).copy()

    ap = r83s_matvec(n, a, x)
    r = b - ap
    p = r.copy()

    for _ in range(max_iter):
        ap = r83s_matvec(n, a, p)
        pap = np.dot(p, ap)
        pr = np.dot(p, r)
        if abs(pap) < 1e-30:
            break
        alpha = pr / pap
        x += alpha * p
        r -= alpha * ap
        rap = np.dot(r, ap)
        beta = -rap / pap
        p = r + beta * p
        if np.linalg.norm(r) < tol:
            break

    return x


def build_1d_waveguide_matrix(epsilon_eff, k0, h, boundary='PEC'):
    """
    Build the finite-difference matrix for the 1D waveguide equation.

    Parameters
    ----------
    epsilon_eff : ndarray, shape (N,)
        Effective permittivity at each grid point.
    k0 : float
        Free-space wave number.
    h : float
        Grid spacing.
    boundary : str
        'PEC' (perfect electric conductor) or 'PML' (perfectly matched layer).

    Returns
    -------
    A : ndarray, shape (N, N)
        Full matrix for eigenvalue analysis.
    a_r83s : ndarray, shape (3,)
        Tridiagonal coefficients [sub, diag, super] for interior points.
    """
    N = epsilon_eff.size
    if N < 3:
        raise ValueError("At least 3 grid points required.")
    if h <= 0:
        raise ValueError("Grid spacing h must be positive.")

    inv_h2 = 1.0 / (h ** 2)
    diag = -2.0 * inv_h2 + (k0 ** 2) * np.real(epsilon_eff)

    A = np.zeros((N, N))
    for i in range(N):
        A[i, i] = diag[i]
        if i > 0:
            A[i, i - 1] = inv_h2
        if i < N - 1:
            A[i, i + 1] = inv_h2

    if boundary == 'PEC':
        A[0, 0] += inv_h2
        A[N - 1, N - 1] += inv_h2
    elif boundary == 'PML':
        # Simple PML: add imaginary conductivity at boundaries
        sigma = np.zeros(N)
        pml_width = max(1, N // 10)
        for i in range(pml_width):
            sigma[i] = (i + 1) ** 2 * 0.1
            sigma[N - 1 - i] = (i + 1) ** 2 * 0.1
        A += np.diag(1j * k0 * sigma)
    else:
        raise ValueError("Unknown boundary condition.")

    a_r83s = np.array([inv_h2, np.mean(diag), inv_h2])
    return A, a_r83s


def solve_waveguide_modes(epsilon_eff, k0, h, num_modes=5, boundary='PEC'):
    """
    Compute the propagation constants β and mode profiles H_y(x) for a
    1D plasmonic waveguide.

    Parameters
    ----------
    epsilon_eff : ndarray
    k0 : float
    h : float
    num_modes : int
    boundary : str

    Returns
    -------
    betas : ndarray, shape (num_modes,)
        Propagation constants.
    modes : ndarray, shape (N, num_modes)
        Mode profiles (normalized).
    """
    A, _ = build_1d_waveguide_matrix(epsilon_eff, k0, h, boundary)
    N = A.shape[0]
    if num_modes > N:
        num_modes = N

    # For Hermitian matrices, use numpy.linalg.eigh
    if np.allclose(A, A.T.conj()):
        eigvals, eigvecs = np.linalg.eigh(A)
    else:
        eigvals, eigvecs = np.linalg.eig(A)

    # Sort by real part descending (guided modes have largest β²)
    idx = np.argsort(-np.real(eigvals))
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    betas = np.sqrt(eigvals[:num_modes])
    modes = eigvecs[:, :num_modes]

    # Normalize modes
    for m in range(num_modes):
        norm = np.sqrt(np.trapezoid(np.abs(modes[:, m]) ** 2, dx=h))
        if norm > 0:
            modes[:, m] /= norm

    return betas, modes


def effective_permittivity_mim_waveguide(epsilon_metal, epsilon_dielectric,
                                         width_metal, width_dielectric, wavelength):
    """
    Compute the effective permittivity of a metal-insulator-metal (MIM)
    waveguide using the parallel-plate approximation:

        ε_eff = (w_m ε_m + w_d ε_d) / (w_m + w_d)

    and the penetration-depth correction:

        ε_eff^{(TM)} ≈ ε_d  ( 1 + 2 (ε_d / |ε_m|) (λ / w_d) )

    Parameters
    ----------
    epsilon_metal : complex
    epsilon_dielectric : float
    width_metal, width_dielectric : float
        Layer widths (m).
    wavelength : float
        Vacuum wavelength (m).

    Returns
    -------
    epsilon_eff : complex
    """
    if width_dielectric <= 0:
        raise ValueError("Dielectric width must be positive.")
    ratio = width_dielectric / wavelength
    correction = 1.0 + 2.0 * (epsilon_dielectric / (abs(epsilon_metal) + 1e-20)) / ratio
    return epsilon_dielectric * correction
