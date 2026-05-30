
import numpy as np
from utils_numerical import safe_divide


def snapshot_pod(A: np.ndarray, num_modes: int = None, energy_threshold: float = 0.99) -> dict:
    M, N = A.shape

    if M == 0 or N == 0:
        return {
            'modes': np.zeros((M, 1)),
            'eigenvalues': np.zeros(1),
            'energy_fraction': np.zeros(1),
            'cum_energy': np.zeros(1),
            'num_modes': 0,
            ' Psi': np.zeros(M)
        }


    Psi = np.mean(A, axis=1)
    A_centered = A - Psi[:, None]


    L = A_centered.T @ A_centered


    L += 1e-12 * np.eye(N) * np.trace(L) / N


    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
    except np.linalg.LinAlgError:

        U, S, Vt = np.linalg.svd(A_centered, full_matrices=False)
        eigenvalues = S ** 2
        eigenvectors = Vt.T


    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]


    eigenvalues = np.maximum(eigenvalues, 0.0) / max(N - 1, 1)


    tol = 1e-10 * np.max(eigenvalues) if np.max(eigenvalues) > 0 else 1e-14
    good_mask = eigenvalues > tol
    num_good = int(np.sum(good_mask))

    eigenvalues = eigenvalues[good_mask]
    eigenvectors = eigenvectors[:, good_mask]


    modes = A_centered @ eigenvectors


    for k in range(modes.shape[1]):
        norm = np.linalg.norm(modes[:, k])
        if norm > 1e-14:
            modes[:, k] /= norm


    total_energy = np.sum(eigenvalues)
    energy_fraction = safe_divide(eigenvalues, total_energy)
    cum_energy = np.cumsum(energy_fraction)


    if num_modes is None:
        num_modes = int(np.searchsorted(cum_energy, energy_threshold) + 1)
    num_modes = min(num_modes, num_good)

    modes = modes[:, :num_modes]
    eigenvalues = eigenvalues[:num_modes]
    energy_fraction = energy_fraction[:num_modes]
    cum_energy = cum_energy[:num_modes]

    return {
        'modes': modes,
        'eigenvalues': eigenvalues,
        'energy_fraction': energy_fraction,
        'cum_energy': cum_energy,
        'num_modes': num_modes,
        'Psi': Psi,
        'A_centered': A_centered
    }


def reconstruct_from_pod(pod_result: dict, coefficients: np.ndarray = None) -> np.ndarray:
    Psi = pod_result['Psi']
    modes = pod_result['modes']
    A_centered = pod_result['A_centered']

    if coefficients is None:
        coefficients = modes.T @ A_centered

    reconstruction = Psi[:, None] + modes @ coefficients
    return reconstruction


def compute_turbulent_kinetic_energy(u_snapshots: np.ndarray, v_snapshots: np.ndarray) -> dict:
    M, N = u_snapshots.shape


    u_mean = np.mean(u_snapshots, axis=1)
    v_mean = np.mean(v_snapshots, axis=1)


    up = u_snapshots - u_mean[:, None]
    vp = v_snapshots - v_mean[:, None]


    tke = 0.5 * (np.mean(up ** 2, axis=1) + np.mean(vp ** 2, axis=1))


    R_uv = np.mean(up * vp, axis=1)
    R_uu = np.mean(up ** 2, axis=1)
    R_vv = np.mean(vp ** 2, axis=1)


    A = np.vstack([up, vp])
    pod = snapshot_pod(A, num_modes=min(20, N, 2 * M))

    return {
        'tke': tke,
        'R_uu': R_uu,
        'R_vv': R_vv,
        'R_uv': R_uv,
        'u_mean': u_mean,
        'v_mean': v_mean,
        'pod': pod
    }


def compute_pod_galerkin_coefficients(pod_modes: np.ndarray, snapshots: np.ndarray) -> np.ndarray:
    Psi = np.mean(snapshots, axis=1)
    A_centered = snapshots - Psi[:, None]
    coefficients = pod_modes.T @ A_centered
    return coefficients


def compute_modal_dynamics(pod_result: dict, dt: float) -> dict:
    A_centered = pod_result['A_centered']
    modes = pod_result['modes']
    coeffs = modes.T @ A_centered
    num_modes = coeffs.shape[0]


    autocorr = []
    for k in range(num_modes):
        c = coeffs[k, :]
        c_norm = c - np.mean(c)
        if np.std(c_norm) < 1e-14:
            autocorr.append(np.zeros(len(c)))
            continue
        corr = np.correlate(c_norm, c_norm, mode='full')
        corr = corr[len(corr) // 2:]
        corr /= corr[0] if corr[0] > 0 else 1.0
        autocorr.append(corr)


    frequencies = []
    for k in range(num_modes):
        c = coeffs[k, :]
        fft_vals = np.abs(np.fft.rfft(c))
        freqs = np.fft.rfftfreq(len(c), d=dt)
        if len(freqs) > 1:
            peak_idx = np.argmax(fft_vals[1:]) + 1
            frequencies.append(float(freqs[peak_idx]))
        else:
            frequencies.append(0.0)

    return {
        'coefficients': coeffs,
        'autocorrelation': autocorr,
        'dominant_frequencies': frequencies
    }
