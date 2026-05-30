
import numpy as np


def joukowsky_transform(z):
    z = np.asarray(z, dtype=complex)

    w = np.zeros_like(z, dtype=complex)
    mask = np.abs(z) > 1e-15
    w[mask] = 0.5 * (z[mask] + 1.0 / z[mask])

    return w


def joukowsky_inverse(w, branch='+'):
    w = np.asarray(w, dtype=complex)
    discriminant = w ** 2 - 1.0
    sqrt_disc = np.sqrt(discriminant)
    if branch == '+':
        z = w + sqrt_disc
    else:
        z = w - sqrt_disc
    return z


def map_accretion_streamline(radius_ratio, theta, offset=0.1):
    theta = np.asarray(theta, dtype=float)

    z = radius_ratio * np.exp(1j * theta) + offset
    w = joukowsky_transform(z)
    return np.real(w), np.imag(w)


def temperature_field_conformal(rho, phi, T_core, T_inf):
    rho = np.asarray(rho, dtype=float)
    rho_inner = 1.0
    rho_outer = 10.0

    rho = np.clip(rho, rho_inner + 1e-12, rho_outer - 1e-12)
    T = T_core + (T_inf - T_core) * np.log(rho / rho_inner) / np.log(rho_outer / rho_inner)
    return T


def test_conformal_mapping():
    theta = np.linspace(0, 2 * np.pi, 100)
    z = 1.1 * np.exp(1j * theta)
    w = joukowsky_transform(z)
    z_rec = joukowsky_inverse(w, branch='+')
    err = np.max(np.abs(z_rec - z))
    print(f"[conformal_mapping] Inverse transform max error = {err:.3e}")
    assert err < 1e-10, "Joukowsky inverse inaccurate"


    w_r, w_i = map_accretion_streamline(1.2, theta, offset=0.15)
    print(f"[conformal_mapping] Accretion streamline mapped to {len(w_r)} points")


if __name__ == "__main__":
    test_conformal_mapping()
