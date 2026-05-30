
import numpy as np
from scipy.special import spherical_jn, spherical_yn, kv, iv


def spherical_bessel_j0(x):
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x)
    mask = np.abs(x) > 1e-12
    result[mask] = np.sin(x[mask]) / x[mask]
    result[~mask] = 1.0 - x[~mask] ** 2 / 6.0
    return result


def spherical_bessel_y0(x):
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x)
    mask = np.abs(x) > 1e-12
    result[mask] = -np.cos(x[mask]) / x[mask]

    result[~mask] = -1.0 / (x[~mask] + 1e-30)
    return result


def modified_bessel_half(x, kind='I'):
    x = np.asarray(x, dtype=float)
    x_safe = np.where(np.abs(x) < 1e-15, 1e-15, np.abs(x))
    if kind == 'I':
        return np.sqrt(2.0 / (np.pi * x_safe)) * np.sinh(x_safe)
    else:
        return np.sqrt(np.pi / (2.0 * x_safe)) * np.exp(-x_safe)


def neutron_diffusion_solution(r, R_star, D, sigma_a, S0):
    r = np.asarray(r, dtype=float)
    L = np.sqrt(D / sigma_a)
    if L <= 0 or R_star <= 0:
        raise ValueError("扩散长度和半径必须为正")


    x = r / L
    x_max = R_star / L


    phi_h = np.zeros_like(x)
    mask = x > 1e-12
    phi_h[mask] = np.sinh(x[mask]) / x[mask]
    phi_h[~mask] = 1.0 + x[~mask] ** 2 / 6.0


    n = len(r)
    if n < 3:
        return np.zeros_like(r)


    dr = r[1] - r[0]
    if not np.allclose(np.diff(r), dr, rtol=1e-5):

        r_uniform = np.linspace(r[0], r[-1], n)

        phi_uniform = neutron_diffusion_solution(r_uniform, R_star, D, sigma_a, S0)
        return np.interp(r, r_uniform, phi_uniform)

    main_diag = np.full(n, -2.0 / (dr ** 2) - 1.0 / (L ** 2))
    lower_diag = 1.0 / (dr ** 2) - 1.0 / (r[1:] * dr)
    upper_diag = 1.0 / (dr ** 2) + 1.0 / (r[:-1] * dr)




    A = np.diag(main_diag) + np.diag(upper_diag, k=1) + np.diag(lower_diag, k=-1)
    A[0, 0] = -2.0 / (dr ** 2) - 1.0 / (L ** 2)
    A[0, 1] = 2.0 / (dr ** 2)
    A[-1, :] = 0.0
    A[-1, -1] = 1.0

    rhs = -S0 * (1.0 - r / R_star) / D
    rhs[-1] = 0.0

    try:
        phi = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:

        phi = phi_h * 0.0


    phi = np.maximum(phi, 0.0)
    return phi


def neutron_capture_rate_profile(r, phi, n_n, sigma_capture):
    phi = np.asarray(phi, dtype=float)
    n_n = np.asarray(n_n, dtype=float)
    if n_n.ndim == 0:
        n_n = np.full_like(phi, n_n)
    rate = n_n * sigma_capture * phi
    return rate


def test_neutron_transport():
    x = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
    j0 = spherical_bessel_j0(x)
    y0 = spherical_bessel_y0(x)
    print(f"[neutron_transport] j0(1.0) = {spherical_bessel_j0(1.0):.6f}, exact = {np.sin(1.0):.6f}")


    r = np.linspace(1e3, 1e6, 500)
    R_star = 1e6
    D = 1e5
    sigma_a = 1e-3
    S0 = 1e20
    phi = neutron_diffusion_solution(r, R_star, D, sigma_a, S0)
    print(f"[neutron_transport] Neutron flux at center: {phi[0]:.3e} cm^{-2}s^{-1}")
    print(f"[neutron_transport] Neutron flux at surface: {phi[-1]:.3e} cm^{-2}s^{-1}")
    assert phi[0] > phi[-1], "Flux should decrease outward"


if __name__ == "__main__":
    test_neutron_transport()
