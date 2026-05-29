"""
Band-Structure Solver for Twisted Bilayer Graphene
===================================================
Diagonalizes the tight-binding Hamiltonian and extracts the electronic
band structure E_n(k) along high-symmetry paths in the moiré Brillouin
zone (MBZ).  Implements direct diagonalization, degeneracy detection,
and Fermi-level determination.

Scientific Background
---------------------
For a periodic crystal, the Bloch theorem states

    ψ_{nk}(r) = e^{i k·r} u_{nk}(r)

with u_{nk}(r) periodic in the lattice.  The Schrödinger equation
reduces to diagonalizing the Hamiltonian in the basis of localized
orbitals at each k-point:

    Σ_m H_{nm}(k) c_{m}^{(ν)}(k) = E_ν(k) c_{n}^{(ν)}(k)

where

    H_{nm}(k) = Σ_R e^{i k·R} H_{n,m}(R)

and R runs over all lattice vectors.  In a supercell approximation
with periodic boundary conditions, k is restricted to the Γ-point
(0,0) if the supercell is sufficiently large.  For smaller supercells
one must sample k-points in the reduced Brillouin zone.

The density of states (per spin) is

    D(E) = Σ_{n,k} δ(E − E_n(k))

and the filling factor at temperature T is

    ν = ∫_{-∞}^{E_F} D(E) dE .

At charge neutrality (ν = 0) the Fermi level sits at the Dirac point.
"""

import numpy as np
from typing import Tuple, List, Optional


def diagonalize_hamiltonian(
    H: np.ndarray, hermitian_check: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Diagonalize a real symmetric (or Hermitian) Hamiltonian matrix.

    Returns eigenvalues (sorted ascending) and corresponding eigenvectors.

    Parameters
    ----------
    H : np.ndarray of shape (N, N)
    hermitian_check : bool
        If True, verify that H is Hermitian within tolerance.

    Returns
    -------
    energies : np.ndarray of shape (N,)
    vectors : np.ndarray of shape (N, N)
        Column j is the eigenvector for energies[j].

    Raises
    ------
    ValueError
        If H is not square or not Hermitian (when checked).
    """
    H = np.asarray(H, dtype=complex)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("Hamiltonian must be a square matrix.")
    N = H.shape[0]
    if hermitian_check:
        diff = np.max(np.abs(H - H.conj().T))
        if diff > 1e-10:
            raise ValueError(f"Hamiltonian is not Hermitian (max diff {diff}).")
    energies, vectors = np.linalg.eigh(H)
    # Ensure real eigenvalues for Hermitian matrix
    energies = np.real(energies)
    return energies, vectors


def high_symmetry_path_mbz(
    theta_deg: float,
    n_points_per_segment: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate k-points along the high-symmetry path Γ → M → K → Γ
    in the moiré Brillouin zone (MBZ) for twisted bilayer graphene.

    The moiré reciprocal lattice vectors are b1, b2 with magnitude

        |b| = 8π / (√3 L_M)

    where L_M = a / (2 sin(θ/2)).

    The high-symmetry points in the hexagonal MBZ are:

        Γ = (0, 0)
        M = (1/2, 0)  in reduced coordinates (b1, b2 basis)
        K = (1/3, 1/3)

    Parameters
    ----------
    theta_deg : float
        Twist angle in degrees.
    n_points_per_segment : int
        Number of points along each segment (excluding duplicate endpoints).

    Returns
    -------
    kpoints : np.ndarray of shape (M, 2)
        Cartesian k-points in nm^{-1}.
    labels : np.ndarray of shape (M,)
        Path labels for plotting (not used when visualization is removed).
    """
    from tight_binding import moire_lattice_constant

    L_M = moire_lattice_constant(theta_deg)
    # Hexagonal reciprocal lattice: b1, b2
    b_mag = 4.0 * np.pi / (np.sqrt(3.0) * L_M)
    b1 = b_mag * np.array([1.0, 0.0])
    b2 = b_mag * np.array([0.5, np.sqrt(3.0) * 0.5])

    # High-symmetry points in reduced coordinates
    Gamma_red = np.array([0.0, 0.0])
    M_red = np.array([0.5, 0.0])
    K_red = np.array([1.0 / 3.0, 1.0 / 3.0])

    points_red = [Gamma_red, M_red, K_red, Gamma_red]
    labels_segments = ["Gamma", "M", "K", "Gamma"]

    kpoints_list = []
    labels_list = []

    for seg in range(len(points_red) - 1):
        p_start = points_red[seg]
        p_end = points_red[seg + 1]
        for t in np.linspace(0.0, 1.0, n_points_per_segment, endpoint=False):
            p_red = (1.0 - t) * p_start + t * p_end
            p_cart = p_red[0] * b1 + p_red[1] * b2
            kpoints_list.append(p_cart)
            labels_list.append(labels_segments[seg])
    # Add final point
    kpoints_list.append(points_red[-1][0] * b1 + points_red[-1][1] * b2)
    labels_list.append(labels_segments[-1])

    return np.array(kpoints_list), np.array(labels_list)


def compute_band_structure_along_path(
    H_builder_func,
    theta_deg: float,
    n_points: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the full band structure E_n(k) along the high-symmetry path
    by constructing the k-dependent Hamiltonian at each point.

    For a supercell model, the k-dependence enters via Bloch phases:

        H_{nm}(k) = H_{nm}^{(0)} · exp[i k·(r_n − r_m)]

    where H^{(0)} is the real-space tight-binding matrix.

    Parameters
    ----------
    H_builder_func : callable
        Function that returns (H0, positions, layer_index).
    theta_deg : float
        Twist angle.
    n_points : int
        Points per segment.

    Returns
    -------
    kpoints : np.ndarray of shape (M, 2)
    energies : np.ndarray of shape (M, N)
        energies[i, n] = E_n(k_i).
    """
    kpoints, _ = high_symmetry_path_mbz(theta_deg, n_points)
    H0, positions, layer_index = H_builder_func()
    N = H0.shape[0]
    M = kpoints.shape[0]
    energies = np.zeros((M, N))

    # TODO: Hole 2 - implement k-dependent Hamiltonian construction and diagonalization
    # Scientific background: Bloch theorem
    #   H_{nm}(k) = H_{nm}^{(0)} · exp[i k·(r_n − r_m)]
    # Hermitianize Hk, then diagonalize to obtain E_n(k).
    raise NotImplementedError("Hole 2: implement k-dependent Hamiltonian via Bloch phases")

    return kpoints, energies


def find_fermi_level(
    energies: np.ndarray,
    num_electrons: Optional[int] = None,
    temperature: float = 0.0,
) -> float:
    """
    Determine the Fermi level E_F for a given set of single-particle
    energies.  At T = 0 this is the highest occupied energy level.

    For a spin-degenerate system with N_e electrons and N_s states,
    each state can hold two electrons (spin up/down).  Thus the
    Fermi level is the energy of the (N_e/2)-th level when sorted.

    Parameters
    ----------
    energies : np.ndarray
        Single-particle energies. Can be 1D (n_bands,) for a single k-point
        or 2D (n_k, n_bands) for multiple k-points.
    num_electrons : int, optional
        Total number of electrons.  If None, assume half-filling
        (one electron per orbital, counting spin implicitly).
    temperature : float
        Temperature in K (not used for T=0).

    Returns
    -------
    float
        Fermi energy in eV.
    """
    energies = np.asarray(energies)
    if energies.ndim == 2:
        n_k, n_bands = energies.shape
        # Use a single k-point's energies to determine Fermi level
        # (all k-points in the BZ are equivalent for a periodic system)
        all_e = np.sort(energies[0])
        n_states = n_bands
    else:
        all_e = np.sort(np.ravel(energies))
        n_states = all_e.size

    if num_electrons is None:
        num_electrons = n_states  # half-filling
    # Each state holds 2 electrons (spin)
    occupied_index = num_electrons // 2
    occupied_index = max(0, min(occupied_index, n_states - 1))
    return float(all_e[occupied_index])


def band_gap_at_fermi_level(
    energies: np.ndarray,
    fermi_level: float,
    tolerance: float = 1e-6,
) -> Tuple[float, int, int]:
    """
    Compute the band gap at the Fermi level.

    gap = min_{E_n > E_F} E_n − max_{E_n ≤ E_F} E_n

    Parameters
    ----------
    energies : np.ndarray
        All eigenvalues. Can be 1D or 2D.
    fermi_level : float
    tolerance : float
        Numerical tolerance for degeneracy.

    Returns
    -------
    gap : float
    highest_occupied : int
        Number of bands below or at E_F (per k-point).
    lowest_unoccupied : int
        Number of bands above E_F (per k-point).
    """
    energies = np.asarray(energies)
    if energies.ndim == 2:
        # Use first k-point for gap analysis
        e = np.sort(energies[0])
    else:
        e = np.sort(np.ravel(energies))

    occupied = e[e <= fermi_level + tolerance]
    unoccupied = e[e > fermi_level + tolerance]
    if occupied.size == 0 or unoccupied.size == 0:
        return 0.0, occupied.size, unoccupied.size
    gap = float(np.min(unoccupied) - np.max(occupied))
    return gap, occupied.size, unoccupied.size


def locate_dirac_points(
    kpoints: np.ndarray,
    energies: np.ndarray,
    n_bands_around_gap: int = 4,
    degeneracy_tol: float = 1e-4,
) -> List[Tuple[int, int, float]]:
    """
    Search for Dirac-like points (band touching or near-touching) along
    the k-path by looking for minimal energy gaps between consecutive
    bands near the charge-neutrality point.

    A Dirac cone satisfies E_+(k) − E_-(k) ≈ ħ v_F |k − k_D|.

    Parameters
    ----------
    kpoints : np.ndarray of shape (M, 2)
    energies : np.ndarray of shape (M, N)
    n_bands_around_gap : int
        Number of bands above and below the gap center to inspect.
    degeneracy_tol : float
        Tolerance for declaring a near-degeneracy.

    Returns
    -------
    list of tuples (ik, band_index, gap_value)
        Locations where a small gap is detected.
    """
    M, N = energies.shape
    mid_band = N // 2
    band_min = max(0, mid_band - n_bands_around_gap)
    band_max = min(N, mid_band + n_bands_around_gap)

    dirac_points = []
    for ik in range(M):
        e_sorted = np.sort(energies[ik])
        for b in range(band_min, band_max - 1):
            gap = e_sorted[b + 1] - e_sorted[b]
            if gap < degeneracy_tol:
                dirac_points.append((ik, b, float(gap)))

    return dirac_points


def group_velocity(
    kpoint: np.ndarray,
    H_builder_func,
    dk: float = 1e-4,
) -> np.ndarray:
    """
    Compute the electron group velocity at a given k-point via finite
    differences:

        v_{n,α} = (1/ħ) ∂E_n / ∂k_α

    Parameters
    ----------
    kpoint : np.ndarray of shape (2,)
    H_builder_func : callable
        Returns (H0, positions, layer_index).
    dk : float
        Finite-difference step in nm^{-1}.

    Returns
    -------
    np.ndarray of shape (N, 2)
        Group velocities for each band in eV·nm (ħ = 1 in these units).
    """
    H0, positions, layer_index = H_builder_func()
    N = H0.shape[0]

    def energy_at_k(k):
        Hk = np.zeros((N, N), dtype=complex)
        for i in range(N):
            for j in range(N):
                phase = np.exp(1j * np.dot(k, positions[i, :2] - positions[j, :2]))
                Hk[i, j] = H0[i, j] * phase
        Hk = 0.5 * (Hk + Hk.conj().T)
        evals = np.linalg.eigvalsh(Hk)
        return np.real(evals)

    e0 = energy_at_k(kpoint)
    v = np.zeros((N, 2))
    for alpha in range(2):
        kp = kpoint.copy()
        kp[alpha] += dk
        km = kpoint.copy()
        km[alpha] -= dk
        ep = energy_at_k(kp)
        em = energy_at_k(km)
        v[:, alpha] = (ep - em) / (2.0 * dk)

    return v
