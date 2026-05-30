
import numpy as np
from numerics_core import givens_rotation, safe_divide
from typing import Tuple, Optional, List






def compute_radial_energy_spectrum(psih: np.ndarray, kx: np.ndarray, ky: np.ndarray,
                                   Ld: float = 1.0, n_bins: int = None) -> Tuple[np.ndarray, np.ndarray]:
    Ny, Nx = psih.shape
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX ** 2 + KY ** 2
    K = np.sqrt(K2)

    energy_density = 0.5 * (K2 + 1.0 / (Ld ** 2)) * np.abs(psih) ** 2

    if n_bins is None:
        n_bins = max(Nx, Ny) // 2

    k_max = max(np.max(np.abs(kx)), np.max(np.abs(ky)))
    dk = k_max / n_bins
    k_bins = np.arange(n_bins) * dk + 0.5 * dk
    E = np.zeros(n_bins)
    count = np.zeros(n_bins)

    k_flat = K.flatten()
    e_flat = energy_density.flatten()
    for i in range(n_bins):
        k_low = i * dk
        k_high = (i + 1) * dk
        mask = (k_flat >= k_low) & (k_flat < k_high)
        if np.any(mask):
            E[i] = np.sum(e_flat[mask])
            count[i] = np.sum(mask)

    E = safe_divide(E, count)
    return k_bins, E


def compute_energy_flux(psih: np.ndarray, q_h: np.ndarray,
                        kx: np.ndarray, ky: np.ndarray,
                        Ld: float = 1.0, n_bins: int = None) -> Tuple[np.ndarray, np.ndarray]:
    Ny_r, Nx_r = psih.shape

    Nx_phys = 2 * (Nx_r - 1)
    Ny_phys = Ny_r
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX ** 2 + KY ** 2
    K = np.sqrt(K2)


    dpsi_dx_h = 1j * KX * psih
    dpsi_dy_h = 1j * KY * psih
    dq_dx_h = 1j * KX * q_h
    dq_dy_h = 1j * KY * q_h

    dpsi_dx = np.fft.irfft2(dpsi_dx_h, s=(Ny_phys, Nx_phys))
    dpsi_dy = np.fft.irfft2(dpsi_dy_h, s=(Ny_phys, Nx_phys))
    dq_dx = np.fft.irfft2(dq_dx_h, s=(Ny_phys, Nx_phys))
    dq_dy = np.fft.irfft2(dq_dy_h, s=(Ny_phys, Nx_phys))

    jac_phys = dpsi_dx * dq_dy - dpsi_dy * dq_dx
    jac_h = np.fft.rfft2(jac_phys)


    transfer = np.real(np.conj(psih) * jac_h)

    if n_bins is None:
        n_bins = max(Nx_phys, Ny_phys) // 2

    k_max = max(np.max(np.abs(kx)), np.max(np.abs(ky)))
    dk = k_max / n_bins
    k_bins = np.arange(n_bins) * dk + 0.5 * dk

    k_flat = K.flatten()
    t_flat = transfer.flatten()


    Pi = np.zeros(n_bins)
    for i in range(n_bins):
        k_cut = k_bins[i]
        mask = k_flat <= k_cut
        if np.any(mask):
            Pi[i] = -np.sum(t_flat[mask])

    return k_bins, Pi


def compute_enstrophy_spectrum(q_h: np.ndarray, kx: np.ndarray, ky: np.ndarray,
                               n_bins: int = None) -> Tuple[np.ndarray, np.ndarray]:
    Ny_r, Nx_r = q_h.shape
    Nx_phys = 2 * (Nx_r - 1)
    Ny_phys = Ny_r
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX ** 2 + KY ** 2)
    z_density = 0.5 * np.abs(q_h) ** 2

    if n_bins is None:
        n_bins = max(Nx_phys, Ny_phys) // 2
    k_max = max(np.max(np.abs(kx)), np.max(np.abs(ky)))
    dk = k_max / n_bins
    k_bins = np.arange(n_bins) * dk + 0.5 * dk
    Z = np.zeros(n_bins)
    count = np.zeros(n_bins)

    k_flat = K.flatten()
    z_flat = z_density.flatten()
    for i in range(n_bins):
        k_low = i * dk
        k_high = (i + 1) * dk
        mask = (k_flat >= k_low) & (k_flat < k_high)
        if np.any(mask):
            Z[i] = np.sum(z_flat[mask])
            count[i] = np.sum(mask)

    Z = safe_divide(Z, count)
    return k_bins, Z






def energy_transfer_path_count(max_scale: int, allowed_steps: List[int] = None) -> np.ndarray:
    if max_scale < 0:
        raise ValueError("max_scale must be non-negative")
    if allowed_steps is None:

        allowed_steps = [1, 2, 3]
    allowed_steps = [s for s in allowed_steps if s > 0]
    if not allowed_steps:
        raise ValueError("allowed_steps must contain positive integers")

    dp = np.zeros(max_scale + 1, dtype=np.int64)
    dp[0] = 1
    for s in range(1, max_scale + 1):
        total = 0
        for step in allowed_steps:
            if s - step >= 0:
                total += dp[s - step]
        dp[s] = total
    return dp


def cascade_path_entropy(dp: np.ndarray) -> float:
    p = dp.astype(float) / np.sum(dp)
    p = p[p > 0]
    return -np.sum(p * np.log(p))






def jacobi_eigenvalue(A: np.ndarray, max_iter: int = 1000, tol: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    A = np.asarray(A, dtype=float).copy()
    n = A.shape[0]
    if n == 0:
        return np.array([]), np.array([])
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")

    V = np.eye(n)

    for it in range(max_iter):

        max_val = 0.0
        p, q = 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                if abs(A[i, j]) > max_val:
                    max_val = abs(A[i, j])
                    p, q = i, j

        if max_val < tol:
            break


        if A[p, p] == A[q, q]:
            theta = np.pi / 4.0
        else:
            tau = (A[q, q] - A[p, p]) / (2.0 * A[p, q])
            if tau >= 0:
                t = 1.0 / (tau + np.sqrt(1.0 + tau ** 2))
            else:
                t = -1.0 / (-tau + np.sqrt(1.0 + tau ** 2))
            c = 1.0 / np.sqrt(1.0 + t ** 2)
            s = t * c


        app = A[p, p]
        aqq = A[q, q]
        apq = A[p, q]

        A[p, p] = c * c * app - 2.0 * c * s * apq + s * s * aqq
        A[q, q] = s * s * app + 2.0 * c * s * apq + c * c * aqq
        A[p, q] = 0.0
        A[q, p] = 0.0

        for i in range(n):
            if i != p and i != q:
                aip = A[i, p]
                aiq = A[i, q]
                A[i, p] = c * aip - s * aiq
                A[p, i] = A[i, p]
                A[i, q] = s * aip + c * aiq
                A[q, i] = A[i, q]

            vip = V[i, p]
            viq = V[i, q]
            V[i, p] = c * vip - s * viq
            V[i, q] = s * vip + c * viq

    eigvals = np.diag(A)

    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = V[:, idx]
    return eigvals, eigvecs


def vertical_mode_decomposition(N2_profile: np.ndarray, dz: float,
                                f0: float = 1e-4, n_modes: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nz = len(N2_profile)
    if nz < 3:
        raise ValueError("N2_profile must have at least 3 points")
    N2 = np.asarray(N2_profile, dtype=float)
    if np.any(N2 < 0):
        raise ValueError("N^2 must be non-negative")





    L = np.zeros((nz, nz))
    for i in range(1, nz - 1):
        L[i, i - 1] = 1.0 / dz ** 2
        L[i, i] = -2.0 / dz ** 2
        L[i, i + 1] = 1.0 / dz ** 2

    L[0, 0] = -1.0 / dz ** 2
    L[0, 1] = 1.0 / dz ** 2
    L[-1, -2] = 1.0 / dz ** 2
    L[-1, -1] = -1.0 / dz ** 2





    N2_safe = np.where(N2 < 1e-15, 1e-15, N2)
    N_sqrt = np.sqrt(N2_safe)
    N_inv_sqrt = 1.0 / N_sqrt

    A = -np.diag(N_inv_sqrt) @ L @ np.diag(N_inv_sqrt)

    A = 0.5 * (A + A.T)

    eigvals, eigvecs = jacobi_eigenvalue(A, max_iter=5000, tol=1e-14)


    phi_raw = np.diag(N_inv_sqrt) @ eigvecs



    positive_mask = eigvals > 1e-15
    eigvals = eigvals[positive_mask]
    phi_raw = phi_raw[:, positive_mask]

    c_n = np.sqrt(1.0 / eigvals)
    Ld_n = c_n / abs(f0)


    n_available = phi_raw.shape[1]
    if n_modes is None or n_modes > n_available:
        n_modes = n_available

    phi_n = np.zeros((nz, n_modes))
    for m in range(n_modes):
        norm = np.sqrt(np.trapezoid(phi_raw[:, m] ** 2, dx=dz))
        if norm > 1e-15:
            phi_n[:, m] = phi_raw[:, m] / norm
        else:
            phi_n[:, m] = phi_raw[:, m]

    return c_n[:n_modes], phi_n, Ld_n[:n_modes]






class SpectralBudgetAnalyzer:

    def __init__(self, k_bins: np.ndarray):
        self.k_bins = np.asarray(k_bins)
        self.n_bins = len(k_bins)
        self.E_history = []
        self.Pi_history = []
        self.Z_history = []
        self.t_history = []

    def record(self, t: float, E: np.ndarray, Pi: np.ndarray, Z: np.ndarray):
        self.t_history.append(t)
        self.E_history.append(E.copy())
        self.Pi_history.append(Pi.copy())
        self.Z_history.append(Z.copy())

    def compute_time_averaged_spectrum(self, t_start: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        t_arr = np.array(self.t_history)
        idx = t_arr >= t_start
        if not np.any(idx):
            return self.k_bins, np.zeros(self.n_bins), np.zeros(self.n_bins)
        E_avg = np.mean(np.array(self.E_history)[idx], axis=0)
        Pi_avg = np.mean(np.array(self.Pi_history)[idx], axis=0)
        Z_avg = np.mean(np.array(self.Z_history)[idx], axis=0)
        return self.k_bins, E_avg, Pi_avg

    def compute_cascade_direction(self, E: np.ndarray, Pi: np.ndarray) -> dict:

        zero_crossings = []
        for i in range(len(Pi) - 1):
            if Pi[i] * Pi[i + 1] < 0:
                zero_crossings.append(self.k_bins[i])


        k_safe = np.where(self.k_bins < 1e-10, 1e-10, self.k_bins)
        mask = (E > 1e-15) & (k_safe > 1e-10)
        if np.sum(mask) > 3:
            logk = np.log(k_safe[mask])
            logE = np.log(E[mask])
            alpha = np.polyfit(logk, logE, 1)[0]
        else:
            alpha = np.nan

        return {
            "zero_crossings": zero_crossings,
            "power_law_slope": float(alpha),
            "inverse_cascade_indicator": float(np.mean(Pi[:len(Pi)//2])),
            "forward_cascade_indicator": float(np.mean(Pi[len(Pi)//2:]))
        }


if __name__ == "__main__":

    A = np.array([[4.0, 1.0, 0.0],
                  [1.0, 3.0, 1.0],
                  [0.0, 1.0, 2.0]])
    vals, vecs = jacobi_eigenvalue(A)
    print("Eigenvalues:", vals)
    print("Residual:", np.max(np.abs(A @ vecs - vecs @ np.diag(vals))))


    nz = 50
    z = np.linspace(0, -1000, nz)
    N2 = 1e-5 * np.exp(z / 200)
    c, phi, Ld = vertical_mode_decomposition(N2, abs(z[1]-z[0]), f0=1e-4, n_modes=5)
    print("Vertical modes c_n:", c)
    print("Rossby radii Ld_n (km):", Ld / 1e3)


    dp = energy_transfer_path_count(20)
    print("Path counts (first 10):", dp[:10])
    print("Cascade entropy:", cascade_path_entropy(dp))
