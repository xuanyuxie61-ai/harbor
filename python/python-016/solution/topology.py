"""
Topological Invariant Calculation for Moiré Heterostructures
============================================================
Computes Z2 topological invariants and parity-based indices for the
moiré bands of twisted bilayer graphene, using techniques adapted from
mod-2 linear algebra.

Scientific Background
---------------------
Twisted bilayer graphene near the magic angle realizes a nearly flat
band with non-trivial topology.  The relevant topological invariant for
time-reversal-invariant 2D systems is the Z2 index (Kane-Mele), which
is defined modulo 2.

For inversion-symmetric systems the Z2 index can be computed from the
parity eigenvalues at time-reversal-invariant momenta (TRIM):

    δ(K_i) = Π_{occupied} ξ_{2m}(K_i)

    (−1)^ν = Π_{i=1}^{4} δ(K_i)

where K_i are the four TRIM in the 2D BZ: Γ, M, and the two X points.

More generally, for a set of Bloch bands the Berry curvature is

    Ω_n(k) = ∇_k × ⟨u_{nk}| i ∇_k |u_{nk}⟩

and the Chern number is

    C_n = (1/2π) ∫_{BZ} Ω_n(k) d²k .

For TBG the first moiré valence band carries Chern number C = ±1 in
certain regions of the phase diagram (when C2T symmetry is broken by
substrate alignment).

The Fu-Kane formula for the Z2 invariant uses the Pfaffian of the
overlap matrix of time-reversal partners:

    (−1)^ν = Π_{K_i} sgn[ Pf( ⟨u_m(K_i)| Θ |u_n(K_i)⟩ ) ]

where Θ is the time-reversal operator.

We approximate the Berry curvature by the discretized formula on a
k-mesh:

    Ω_n(k) ≈ (1/A_tri) Im[ ln( U_1 U_2 U_3 ) ]

where U_j = ⟨u_{n,k_j} | u_{n,k_{j+1}}⟩ are link variables around a
plaquette, and A_tri is the plaquette area.
"""

import numpy as np
from typing import Tuple, List


def berry_curvature_discrete(
    k_tri: np.ndarray,
    u_states: np.ndarray,
) -> float:
    """
    Compute the Berry curvature of a single band from three k-points
    forming a triangular plaquette.

    Using the discretized formula (Fukui-Hatsugai-Suzuki method):

        U_{12} = ⟨u_1 | u_2⟩ / |⟨u_1 | u_2⟩|
        U_{23} = ⟨u_2 | u_3⟩ / |⟨u_2 | u_3⟩|
        U_{31} = ⟨u_3 | u_1⟩ / |⟨u_3 | u_1⟩|

        F = Im[ ln( U_{12} U_{23} U_{31} ) ]

    Parameters
    ----------
    k_tri : np.ndarray of shape (3, 2)
        k-points of the triangle.
    u_states : np.ndarray of shape (3, N)
        Bloch wavefunctions at the three k-points.

    Returns
    -------
    float
        Berry flux through the triangle.
    """
    U_link = np.zeros(3, dtype=complex)
    for e in range(3):
        i = e
        j = (e + 1) % 3
        overlap = np.vdot(u_states[i], u_states[j])
        if abs(overlap) < 1e-14:
            overlap = 1e-14
        U_link[e] = overlap / abs(overlap)

    product = U_link[0] * U_link[1] * U_link[2]
    flux = np.imag(np.log(product))
    return float(flux)


def chern_number_from_mesh(
    kpoints: np.ndarray,
    band_energies: np.ndarray,
    band_vectors: np.ndarray,
    band_index: int,
) -> float:
    """
    Compute the Chern number of a single band from a triangular mesh
    of k-points.

        C = (1/2π) Σ_triangles F_tri

    Parameters
    ----------
    kpoints : np.ndarray of shape (N, 2)
    band_energies : np.ndarray of shape (N, n_bands)
    band_vectors : np.ndarray of shape (N, n_bands, n_orbitals)
    band_index : int

    Returns
    -------
    float
        Chern number (should be close to an integer).
    """
    from scipy.spatial import Delaunay

    try:
        tri = Delaunay(kpoints)
    except Exception:
        return 0.0

    total_flux = 0.0
    for simplex in tri.simplices:
        k_tri = kpoints[simplex]
        u_states = band_vectors[simplex, band_index, :]
        flux = berry_curvature_discrete(k_tri, u_states)
        total_flux += flux

    chern = total_flux / (2.0 * np.pi)
    return float(chern)


def parity_eigenvalues_at_trim(
    H_trim: np.ndarray,
    inversion_center: np.ndarray,
) -> np.ndarray:
    """
    Compute the parity eigenvalues ξ_n = ±1 of the occupied bands at a
    TRIM point for an inversion-symmetric Hamiltonian.

    The inversion operator I acts as

        I ψ(r) = ψ(−r)

    In the tight-binding basis, the matrix representation is a permutation
    matrix P with P_{ij} = δ_{i,π(j)} where π maps each orbital to its
    inverted partner.

    For simplicity we approximate P as the identity (valid when each basis
    orbital is its own inversion partner) and compute the eigenvalues of H.
    In a more complete treatment one would construct the proper permutation.

    Parameters
    ----------
    H_trim : np.ndarray of shape (N, N)
        Hamiltonian at the TRIM point.
    inversion_center : np.ndarray of shape (2,)
        Inversion center in real space.

    Returns
    -------
    np.ndarray of shape (N,)
        Parity eigenvalues (±1) approximated from the sign of eigenvalues
        of the symmetrized inversion operator.
    """
    N = H_trim.shape[0]
    # Approximation: construct a simple inversion matrix that pairs
    # orbitals i ↔ N−1−i (valid for symmetric arrangements)
    P = np.zeros((N, N))
    for i in range(N):
        j = N - 1 - i
        P[i, j] = 1.0

    # The parity eigenvalues come from diagonalizing P·H (commuting with H)
    # For inversion-symmetric H, [P, H] = 0, so they share eigenvectors.
    # We compute the expectation value of P in each eigenstate of H.
    energies, vectors = np.linalg.eigh(H_trim)
    parities = np.zeros(N)
    for n in range(N):
        psi = vectors[:, n]
        val = np.real(np.vdot(psi, P @ psi))
        # Round to nearest ±1
        parities[n] = 1.0 if val > 0 else -1.0

    return parities


def z2_index_from_parity(
    trim_hamiltonians: List[np.ndarray],
    trim_names: List[str],
    n_occupied: int,
) -> int:
    """
    Compute the Z2 topological index from parity eigenvalues at the
    four time-reversal-invariant momenta (TRIM) using the Fu-Kane formula.

        δ_i = Π_{n=1}^{n_occupied} ξ_{2n}(K_i)
        (−1)^ν = Π_{i=1}^{4} δ_i

    Parameters
    ----------
    trim_hamiltonians : list of np.ndarray
        Hamiltonians at the four TRIM points.
    trim_names : list of str
        Names of the TRIM points.
    n_occupied : int
        Number of occupied bands (counting spin degeneracy).

    Returns
    -------
    int
        Z2 index: 0 (trivial) or 1 (non-trivial).
    """
    if len(trim_hamiltonians) != 4:
        raise ValueError("Exactly four TRIM points are required for 2D Z2.")

    delta_product = 1
    for i, H in enumerate(trim_hamiltonians):
        parities = parity_eigenvalues_at_trim(H, np.zeros(2))
        # Take product over the occupied bands
        # For spin-degenerate systems, use every second band (pairs)
        occ_parities = parities[:n_occupied]
        delta_i = int(np.prod(occ_parities))
        delta_product *= delta_i

    z2 = 0 if delta_product == 1 else 1
    return z2


def mod2_matrix_rank(
    M: np.ndarray,
    tol: float = 1e-10,
) -> int:
    """
    Compute the rank of a matrix over the field F_2 (modulo 2).

    This is used in topological classification where parities and
    winding numbers are defined modulo 2.

    Algorithm: Gaussian elimination over F_2 using bit operations.
    For small matrices we use standard Gaussian elimination with rounding.

    Parameters
    ----------
    M : np.ndarray of shape (m, n)
    tol : float

    Returns
    -------
    int
        Rank over F_2.
    """
    M = np.asarray(M, dtype=float).copy()
    m, n = M.shape
    rank = 0
    for col in range(n):
        # Find pivot
        pivot = -1
        for row in range(rank, m):
            if abs(M[row, col]) > tol:
                pivot = row
                break
        if pivot == -1:
            continue
        # Swap rows
        M[[rank, pivot]] = M[[pivot, rank]]
        # Eliminate
        for row in range(m):
            if row != rank and abs(M[row, col]) > tol:
                # Over F_2: subtraction = addition
                M[row] = np.abs(M[row] - M[rank])
                M[row] = (M[row] > tol).astype(float)
        rank += 1
        if rank >= m:
            break
    return rank


def lights_out_matrix_moire(
    nx: int,
    ny: int,
) -> np.ndarray:
    """
    Construct the 25×25-style Lights Out matrix for a 2D grid of
    nx × ny moiré stacking sites, where pressing a button toggles
    the stacking state of the site and its nearest neighbors.

    In the context of topology, this matrix represents the action of
    local stacking rearrangements on the global parity configuration.

    Parameters
    ----------
    nx, ny : int
        Grid dimensions.

    Returns
    -------
    np.ndarray of shape (nx*ny, nx*ny)
        Lights Out matrix over F_2 (entries are 0 or 1).
    """
    N = nx * ny
    L = np.zeros((N, N), dtype=int)
    for ix in range(nx):
        for iy in range(ny):
            idx = ix * ny + iy
            L[idx, idx] = 1
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                jx = ix + dx
                jy = iy + dy
                if 0 <= jx < nx and 0 <= jy < ny:
                    jdx = jx * ny + jy
                    L[idx, jdx] = 1
    return L


def moire_wilson_loop(
    k_path: np.ndarray,
    H_builder_at_k: callable,
    band_index: int,
) -> complex:
    """
    Compute the Wilson loop (Berry phase) along a closed path in
    k-space for a given band.

        W = Π_{j} ⟨u_{k_j} | u_{k_{j+1}}⟩

    For a closed path the phase φ = −Im[ln W] gives the Berry phase.
    Quantized values (0 or π) indicate trivial or non-trivial topology.

    Parameters
    ----------
    k_path : np.ndarray of shape (M, 2)
        Closed k-path (first and last points should be identical or
        implicitly connected).
    H_builder_at_k : callable
        Function k → (energies, vectors).
    band_index : int

    Returns
    -------
    complex
        Wilson loop eigenvalue.
    """
    M = k_path.shape[0]
    if M < 2:
        return 1.0 + 0.0j

    W = 1.0 + 0.0j
    for j in range(M):
        k1 = k_path[j]
        k2 = k_path[(j + 1) % M]
        _, vecs1 = H_builder_at_k(k1)
        _, vecs2 = H_builder_at_k(k2)
        u1 = vecs1[:, band_index]
        u2 = vecs2[:, band_index]
        overlap = np.vdot(u1, u2)
        W *= overlap

    return W
