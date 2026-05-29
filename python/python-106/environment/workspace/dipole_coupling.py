"""
dipole_coupling.py
==================
Coupled dipole model (CDM) for plasmonic nanoparticle assemblies.

Each nanoparticle j is represented as a point dipole with polarizability
α_j(ω).  The self-consistent dipole moment under incident field E_inc is:

    p_j = α_j [ E_inc(r_j) + Σ_{k≠j} G(r_j, r_k) p_k ]

where G(r_j, r_k) is the free-space dyadic Green's tensor:

    G(r) = (k² / ε₀) (e^{ikr} / 4πr) [
              (1 + i/(kr) − 1/(kr)²) I
            − (1 + 3i/(kr) − 3/(kr)²) r̂ ⊗ r̂ ]

with k = n_medium ω / c.

In matrix form, defining the interaction matrix A_{jj'} and incident vector
b_j = E_inc(r_j), the linear system is:

    (I/α − G) p = E_inc

or   A p = b.

The spectral properties of A determine the collective plasmon modes.
Connectivity between particles is analyzed via a directed graph where an
arc i→j exists if the near-field coupling |G_{ij}| exceeds a threshold.
"""

import numpy as np


def dyadic_green_tensor(r_vec, k, eps0=8.854187817e-12):
    """
    Compute the free-space dyadic Green's tensor G(r) for a single distance vector.

    Parameters
    ----------
    r_vec : ndarray, shape (3,)
        Displacement vector r (m).
    k : float
        Wave number in medium (rad/m).
    eps0 : float
        Vacuum permittivity (F/m).

    Returns
    -------
    G : ndarray, shape (3, 3)
        Dyadic Green's tensor.
    """
    r = np.linalg.norm(r_vec)
    if r < 1e-18:
        # Self-term regularization: use Clausius-Mossotti local field
        return np.zeros((3, 3), dtype=complex)

    kr = k * r
    if abs(kr) < 1e-12:
        # Static limit
        prefactor = 1.0 / (4.0 * np.pi * eps0 * (r ** 3))
        rr = np.outer(r_vec, r_vec) / (r ** 2)
        G = prefactor * (3.0 * rr - np.eye(3))
        return G

    prefactor = (k ** 2) / eps0 * np.exp(1j * kr) / (4.0 * np.pi * r)
    rr = np.outer(r_vec, r_vec) / (r ** 2)

    term1 = (1.0 + 1j / kr - 1.0 / (kr ** 2)) * np.eye(3)
    term2 = (1.0 + 3.0j / kr - 3.0 / (kr ** 2)) * rr
    G = prefactor * (term1 - term2)
    return G


def build_coupling_matrix(positions, polarizabilities, omega, eps_medium=1.0):
    """
    Build the full coupled-dipole interaction matrix A = diag(1/α) − G.

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Nanoparticle positions (m).
    polarizabilities : ndarray, shape (N,)
        Complex polarizability α_j(ω) (C·m²/V = F·m²).
    omega : float
        Angular frequency (rad/s).
    eps_medium : float
        Medium permittivity.

    Returns
    -------
    A : ndarray, shape (3N, 3N)
        Complex interaction matrix.
    """
    c = 2.99792458e8
    N = positions.shape[0]
    if N == 0:
        raise ValueError("At least one particle required.")
    if positions.shape[1] != 3:
        raise ValueError("Positions must be N×3.")

    k_medium = omega * np.sqrt(eps_medium) / c
    A = np.zeros((3 * N, 3 * N), dtype=complex)

    for j in range(N):
        # Self-term: inverse polarizability on block diagonal
        alpha_j = polarizabilities[j]
        if abs(alpha_j) < 1e-40:
            alpha_j = 1e-40
        inv_alpha = 1.0 / alpha_j
        A[3 * j:3 * j + 3, 3 * j:3 * j + 3] = inv_alpha * np.eye(3)

        for k_ in range(j + 1, N):
            r_vec = positions[j] - positions[k_]
            Gjk = dyadic_green_tensor(r_vec, k_medium)
            A[3 * j:3 * j + 3, 3 * k_:3 * k_ + 3] = -Gjk
            A[3 * k_:3 * k_ + 3, 3 * j:3 * j + 3] = -Gjk.T

    return A


def incident_plane_wave(positions, E0, kvec, pol):
    """
    Compute incident plane-wave electric field at particle positions.

    E_inc(r) = E₀ ε̂ exp(i k·r)

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
    E0 : float
        Field amplitude (V/m).
    kvec : ndarray, shape (3,)
        Wave vector.
    pol : ndarray, shape (3,)
        Polarization unit vector.

    Returns
    -------
    b : ndarray, shape (3N,)
        Flattened incident field vector.
    """
    N = positions.shape[0]
    b = np.zeros(3 * N, dtype=complex)
    for j in range(N):
        phase = np.exp(1j * np.dot(kvec, positions[j]))
        b[3 * j:3 * j + 3] = E0 * pol * phase
    return b


def solve_dipole_moments(A, b, tol=1e-12, max_iter=5000):
    """
    Solve A p = b using BiCGSTAB for sparse complex linear systems.
    Falls back to dense numpy.linalg.solve for small systems.

    Parameters
    ----------
    A : ndarray
    b : ndarray
    tol : float
    max_iter : int

    Returns
    -------
    p : ndarray
    """
    N = b.size
    if N <= 60:
        return np.linalg.solve(A, b)

    try:
        from scipy.sparse.linalg import bicgstab
        from scipy.sparse import csr_matrix
        p, info = bicgstab(csr_matrix(A), b, tol=tol, maxiter=max_iter)
        if info != 0:
            p = np.linalg.solve(A, b)
    except Exception:
        p = np.linalg.solve(A, b)
    return p


def build_coupling_graph(positions, omega, eps_medium=1.0, threshold=1.0e30):
    """
    Build a directed-graph adjacency list from near-field coupling strengths.
    An arc i → j exists if |G_{ij}| > threshold.

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
    omega : float
    eps_medium : float
    threshold : float
        Coupling-strength threshold (SI units).

    Returns
    -------
    adjacency : list of list of int
        adjacency[i] contains all j such that i→j exists.
    arc_list : list of tuple
        Flat list of (i, j) arcs.
    """
    c = 2.99792458e8
    N = positions.shape[0]
    k = omega * np.sqrt(eps_medium) / c
    adjacency = [[] for _ in range(N)]
    arc_list = []

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            r_vec = positions[i] - positions[j]
            G = dyadic_green_tensor(r_vec, k)
            strength = np.linalg.norm(G)
            if strength > threshold:
                adjacency[i].append(j)
                arc_list.append((i, j))

    return adjacency, arc_list


def polarizability_clausius_mossotti(eps_particle, eps_medium, volume):
    """
    Clausius-Mossotti polarizability of a spherical nanoparticle:

        α = 3 ε₀ V (ε_p − ε_m) / (ε_p + 2 ε_m)

    Parameters
    ----------
    eps_particle : complex
    eps_medium : float
    volume : float
        Particle volume (m³).

    Returns
    -------
    alpha : complex
    """
    # TODO: Implement the Clausius-Mossotti polarizability of a spherical nanoparticle.
    # Formula:  α = 3 ε₀ V (ε_p − ε_m) / (ε_p + 2 ε_m)
    # Handle the case where the denominator is near zero.
    raise NotImplementedError("Hole 2: Clausius-Mossotti polarizability formula is missing.")
