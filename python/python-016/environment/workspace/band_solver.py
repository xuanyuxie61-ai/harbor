
import numpy as np
from typing import Tuple, List, Optional


def diagonalize_hamiltonian(
    H: np.ndarray, hermitian_check: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    H = np.asarray(H, dtype=complex)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("Hamiltonian must be a square matrix.")
    N = H.shape[0]
    if hermitian_check:
        diff = np.max(np.abs(H - H.conj().T))
        if diff > 1e-10:
            raise ValueError(f"Hamiltonian is not Hermitian (max diff {diff}).")
    energies, vectors = np.linalg.eigh(H)

    energies = np.real(energies)
    return energies, vectors


def high_symmetry_path_mbz(
    theta_deg: float,
    n_points_per_segment: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    from tight_binding import moire_lattice_constant

    L_M = moire_lattice_constant(theta_deg)

    b_mag = 4.0 * np.pi / (np.sqrt(3.0) * L_M)
    b1 = b_mag * np.array([1.0, 0.0])
    b2 = b_mag * np.array([0.5, np.sqrt(3.0) * 0.5])


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

    kpoints_list.append(points_red[-1][0] * b1 + points_red[-1][1] * b2)
    labels_list.append(labels_segments[-1])

    return np.array(kpoints_list), np.array(labels_list)


def compute_band_structure_along_path(
    H_builder_func,
    theta_deg: float,
    n_points: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    kpoints, _ = high_symmetry_path_mbz(theta_deg, n_points)
    H0, positions, layer_index = H_builder_func()
    N = H0.shape[0]
    M = kpoints.shape[0]
    energies = np.zeros((M, N))





    raise NotImplementedError("Hole 2: implement k-dependent Hamiltonian via Bloch phases")

    return kpoints, energies


def find_fermi_level(
    energies: np.ndarray,
    num_electrons: Optional[int] = None,
    temperature: float = 0.0,
) -> float:
    energies = np.asarray(energies)
    if energies.ndim == 2:
        n_k, n_bands = energies.shape


        all_e = np.sort(energies[0])
        n_states = n_bands
    else:
        all_e = np.sort(np.ravel(energies))
        n_states = all_e.size

    if num_electrons is None:
        num_electrons = n_states

    occupied_index = num_electrons // 2
    occupied_index = max(0, min(occupied_index, n_states - 1))
    return float(all_e[occupied_index])


def band_gap_at_fermi_level(
    energies: np.ndarray,
    fermi_level: float,
    tolerance: float = 1e-6,
) -> Tuple[float, int, int]:
    energies = np.asarray(energies)
    if energies.ndim == 2:

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
