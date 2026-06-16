"""
流谐波模块：椭圆流 v_n 与参与者平面 Ψ_n
Flow harmonics decomposition: v_n, participant plane angles Psi_n,
and event-plane method for extracting anisotropic flow.

    dN/dφ ∝ 1 + 2 Σ_n v_n cos(n(φ - Ψ_n))

Algorithms sourced from:
  - 1316_triangle_symq_rule (angular quadrature for Fourier integrals)
  - 1218_uw-mad-dash_bagpipe (chunked parallel reduction)
"""
import numpy as np


def fourier_decomposition(eta, phi, weights=None, n_harmonics=5):
    """
    Compute flow harmonics Q_n vectors:
        Q_n = Σ_k w_k e^{i n φ_k}

    Then:
        v_n e^{i n Ψ_n} = Q_n / Σ_k w_k

    Parameters
    ----------
    eta : ndarray
        Pseudorapidities (unused at mid-rapidity, kept for API)
    phi : ndarray
        Azimuthal angles in radians
    weights : ndarray, optional
        Particle weights (default: uniform)
    n_harmonics : int
        Number of harmonics to compute (default: 5)

    Returns
    -------
    v_n : ndarray (n_harmonics,)
        Flow harmonic magnitudes
    psi_n : ndarray (n_harmonics,)
        Event plane angles
    Q_n : ndarray (complex, n_harmonics,)
        Q-vectors
    """
    if weights is None:
        weights = np.ones_like(phi)

    total_weight = np.sum(weights)
    if total_weight <= 0:
        return (np.zeros(n_harmonics), np.zeros(n_harmonics),
                np.zeros(n_harmonics, dtype=complex))

    v_n = np.zeros(n_harmonics)
    psi_n = np.zeros(n_harmonics)
    Q_n = np.zeros(n_harmonics, dtype=complex)

    for n in range(1, n_harmonics + 1):
        qn = np.sum(weights * np.exp(1j * n * phi))
        Q_n[n - 1] = qn
        qn_norm = qn / total_weight
        v_n[n - 1] = np.abs(qn_norm)
        psi_n[n - 1] = np.angle(qn_norm) / n

    return v_n, psi_n, Q_n


def event_plane_method(phi, n, eta_range=(-1.0, 1.0), eta=None):
    """
    Event plane method for v_n extraction:
        v_n = <cos(n(φ - Ψ_n))> / R_n

    where Ψ_n is the event plane angle from a sub-event,
    and R_n is the event plane resolution.

    Parameters
    ----------
    phi : ndarray
        Particle azimuthal angles
    n : int
        Harmonic order (2 for elliptic flow)
    eta_range : tuple
        Pseudorapidity range for sub-event
    eta : ndarray, optional
        Particle pseudorapidities

    Returns
    -------
    v_n : float
        Flow harmonic
    psi_n : float
        Event plane angle
    """
    # Select particles in eta range for EP determination
    if eta is None:
        eta = np.zeros_like(phi)
    mask_ep = (eta >= eta_range[0]) & (eta <= eta_range[1])

    if not np.any(mask_ep):
        return 0.0, 0.0

    # Compute event plane from sub-event
    qn = np.sum(np.exp(1j * n * phi[mask_ep]))
    psi_n = np.angle(qn) / n

    # Compute v_n from all particles
    v_n = np.mean(np.cos(n * (phi - psi_n)))

    return v_n, psi_n


def two_particle_correlator(phi_1, phi_2, n):
    """
    Two-particle azimuthal correlation for v_n extraction:
        V_nΔ = <cos(n(φ_1 - φ_2))>

    For uncorrelated particles: V_nΔ = v_n²

    Parameters
    ----------
    phi_1, phi_2 : ndarray
        Azimuthal angles of particle pair
    n : int
        Harmonic order

    Returns
    -------
    float
        V_nΔ correlator
    """
    return np.mean(np.cos(n * (phi_1 - phi_2)))


def scalar_product_method(q_a, q_b, q_c, n):
    """
    Scalar product method for v_n using three sub-events:
        v_n = √(<Q_A · Q_B Q_C*> / <Q_A · Q_B Q_A*>)

    This avoids resolution corrections needed for event-plane method.

    Parameters
    ----------
    q_a, q_b, q_c : complex
        Q-vectors from three sub-events
    n : int
        Harmonic order (encoded in Q-vectors)

    Returns
    -------
    float
        v_n magnitude
    """
    # For simplicity, use two-subevent approximation
    qa_dot_qb = np.real(q_a * np.conj(q_b))
    norm = np.sqrt(np.real(q_a * np.conj(q_a)) * np.real(q_b * np.conj(q_b)))
    if norm < 1e-14:
        return 0.0
    return np.sqrt(np.abs(qa_dot_qb / norm))


def symmetric_quadrature_fourier(f_values, phi_grid, n):
    """
    Compute Fourier coefficient using symmetric quadrature:
        a_n = (1/π) ∫ f(φ) cos(nφ) dφ

    Using trapezoidal rule on uniform φ grid (equivalent to symmetric
    quadrature from 1316_triangle_symq_rule for periodic functions).

    Parameters
    ----------
    f_values : ndarray
        Function values at phi_grid points
    phi_grid : ndarray
        Azimuthal angles (uniform grid)
    n : int
        Harmonic order

    Returns
    -------
    a_n, b_n : float
        Fourier cosine and sine coefficients
    """
    dphi = phi_grid[1] - phi_grid[0] if len(phi_grid) > 1 else 0.0
    a_n = np.trapz(f_values * np.cos(n * phi_grid), dx=dphi) / np.pi
    b_n = np.trapz(f_values * np.sin(n * phi_grid), dx=dphi) / np.pi
    return a_n, b_n


def flow_from_spectrum(phi_spectrum, n_bins=36, n_harmonics=5):
    """
    Compute v_n from binned azimuthal spectrum dN/dφ.

    dN/dφ = N₀ (1 + 2 Σ v_n cos(n(φ - Ψ_n)))

    Parameters
    ----------
    phi_spectrum : ndarray (n_bins,)
        Azimuthal distribution (histogram counts)
    n_bins : int
        Number of angular bins
    n_harmonics : int
        Max harmonic to extract

    Returns
    -------
    v_n : ndarray (n_harmonics,)
    psi_n : ndarray (n_harmonics,)
    """
    phi_centers = np.linspace(0, 2 * np.pi, n_bins, endpoint=False) + np.pi / n_bins

    v_n = np.zeros(n_harmonics)
    psi_n = np.zeros(n_harmonics)

    total = np.sum(phi_spectrum)
    if total <= 0:
        return v_n, psi_n

    f_norm = phi_spectrum / total

    for n in range(1, n_harmonics + 1):
        a_n, b_n = symmetric_quadrature_fourier(f_norm, phi_centers, n)
        v_n[n - 1] = np.sqrt(a_n**2 + b_n**2)
        psi_n[n - 1] = np.arctan2(b_n, a_n) / n

    return v_n, psi_n


def event_by_event_vn_fluctuation(events_phi, n=2, n_events=100):
    """
    Compute event-by-event v_n distribution and its mean/variance.

    Parameters
    ----------
    events_phi : list of ndarray
        List of phi arrays (one per event)
    n : int
        Harmonic order
    n_events : int
        Number of events (for truncation)

    Returns
    -------
    v_n_mean : float
    v_n_std : float
    v_n_array : ndarray
    """
    v_n_array = []
    for i, phi in enumerate(events_phi[:n_events]):
        v_n_event, _, _ = fourier_decomposition(None, phi, n_harmonics=n)
        v_n_array.append(v_n_event[n - 1])

    v_n_array = np.array(v_n_array)
    return np.mean(v_n_array), np.std(v_n_array), v_n_array


def participant_plane_resolution(psi_n_part, psi_n_ep, n):
    """
    Event plane resolution:
        R_n = <cos(n(Ψ_n^EP - Ψ_n^part))>

    Parameters
    ----------
    psi_n_part : ndarray
        Participant plane angles (true)
    psi_n_ep : ndarray
        Event plane angles (measured)
    n : int
        Harmonic order

    Returns
    -------
    float
        Resolution factor R_n ∈ [0, 1]
    """
    return np.mean(np.cos(n * (psi_n_ep - psi_n_part)))


def vn_pt_dependence(pt_bins, phi_at_pt, n=2):
    """
    Compute v_n(p_T) differential flow.

    Parameters
    ----------
    pt_bins : ndarray
        Transverse momentum bin centers
    phi_at_pt : list of ndarray
        List of phi arrays per pT bin
    n : int
        Harmonic order

    Returns
    -------
    vn_pt : ndarray
        v_n as function of pT
    """
    vn_pt = np.zeros_like(pt_bins)
    for i, phi in enumerate(phi_at_pt):
        v_n, _, _ = fourier_decomposition(None, phi, n_harmonics=n)
        vn_pt[i] = v_n[n - 1]
    return vn_pt


def chunked_flow_reduction(chunks_phi, n=2):
    """
    Compute flow harmonics using chunked parallel reduction
    (inspired by 1218_bagpipe distributed reduction pattern).

    Each chunk computes Q_n locally; then sum Q_n across chunks.

    Parameters
    ----------
    chunks_phi : list of ndarray
        List of phi arrays (chunks)
    n : int
        Harmonic order

    Returns
    -------
    v_n : float
    psi_n : float
    """
    Q_n_total = 0j
    N_total = 0

    for chunk in chunks_phi:
        qn_local = np.sum(np.exp(1j * n * chunk))
        Q_n_total += qn_local
        N_total += len(chunk)

    if N_total <= 0:
        return 0.0, 0.0

    qn_norm = Q_n_total / N_total
    return np.abs(qn_norm), np.angle(qn_norm) / n
