
import numpy as np


def compute_biorthogonal_eigenvectors(H):
    if H.shape[0] != H.shape[1]:
        raise ValueError("H must be a square matrix.")


    E, right = np.linalg.eig(H)

    E_left, left_dag = np.linalg.eig(H.T)


    left = left_dag.T


    N = H.shape[0]
    for n in range(N):
        overlap = np.vdot(left[n, :], right[:, n])
        if abs(overlap) < 1e-30:
            raise RuntimeError(f"Zero biorthogonal overlap for eigenvalue {n}.")
        left[n, :] = left[n, :] / np.conj(overlap)

    return E, right, left


def berry_connection_1d(H_func, k, dk=1e-5):
    E0, right0, left0 = compute_biorthogonal_eigenvectors(H_func(k))
    Ep, rightp, leftp = compute_biorthogonal_eigenvectors(H_func(k + dk))
    Em, rightm, leftm = compute_biorthogonal_eigenvectors(H_func(k - dk))



    n = np.argmin(E0.real)

    d_right = (rightp[:, n] - rightm[:, n]) / (2.0 * dk)
    A = 1j * np.vdot(left0[n, :], d_right)
    return A


def berry_curvature_2d(H_func, kx, ky, dk=1e-5):

    def get_eig(kx_, ky_):
        E, right, left = compute_biorthogonal_eigenvectors(H_func(kx_, ky_))
        n = np.argmin(E.real)
        return right[:, n], left[n, :]

    rp_kx, lp_kx = get_eig(kx + dk, ky)
    rm_kx, lm_kx = get_eig(kx - dk, ky)
    rp_ky, lp_ky = get_eig(kx, ky + dk)
    rm_ky, lm_ky = get_eig(kx, ky - dk)
    r_pp, l_pp = get_eig(kx + dk, ky + dk)
    r_pm, l_pm = get_eig(kx + dk, ky - dk)
    r_mp, l_mp = get_eig(kx - dk, ky + dk)
    r_mm, l_mm = get_eig(kx - dk, ky - dk)


    d2_r_dkxdky = (r_pp - r_pm - r_mp + r_mm) / (4.0 * dk * dk)


    d_r_dkx = (rp_kx - rm_kx) / (2.0 * dk)
    d_r_dky = (rp_ky - rm_ky) / (2.0 * dk)
    d_l_dkx = (lp_kx - lm_kx) / (2.0 * dk)
    d_l_dky = (lp_ky - lm_ky) / (2.0 * dk)


    r0, l0 = get_eig(kx, ky)


    Omega = 1j * (
        np.vdot(d_l_dkx, d_r_dky) - np.vdot(d_l_dky, d_r_dkx)
    )
    return Omega


def zak_phase_1d(H_func, k_points=401, a=1.0):
    k_vals = np.linspace(-np.pi / a, np.pi / a, k_points)
    A_vals = np.array([berry_connection_1d(H_func, k) for k in k_vals])
    gamma_zak = np.trapz(A_vals, k_vals)
    return gamma_zak


def chern_number_2d(H_func, kx_points=81, ky_points=81, dk=None):
    kx_vals = np.linspace(-np.pi, np.pi, kx_points)
    ky_vals = np.linspace(-np.pi, np.pi, ky_points)
    if dk is None:
        dk = (kx_vals[1] - kx_vals[0]) * 0.1

    Omega_grid = np.zeros((kx_points, ky_points), dtype=complex)
    for i, kx in enumerate(kx_vals):
        for j, ky in enumerate(ky_vals):
            try:
                Omega_grid[i, j] = berry_curvature_2d(H_func, kx, ky, dk=dk)
            except Exception:
                Omega_grid[i, j] = 0.0

    C = np.trapz(np.trapz(Omega_grid, ky_vals, axis=1), kx_vals, axis=0) / (2.0 * np.pi)
    return C


def winding_number_complex_energy(H_func, k_points=401):
    k_vals = np.linspace(-np.pi, np.pi, k_points)
    E_vals = np.zeros(k_points, dtype=complex)
    for i, k in enumerate(k_vals):
        H = H_func(k)
        E, _, _ = compute_biorthogonal_eigenvectors(H)
        E_vals[i] = E[np.argmin(E.real)]

    dE = np.gradient(E_vals, k_vals)
    integrand = dE / E_vals
    W = np.trapz(integrand, k_vals) / (2.0j * np.pi)
    return W.real
