
import numpy as np
from spectral_expansion import spectral_expand_reaction_rate, spectral_evaluate_reaction_rate



K_BOLTZMANN = 1.380649e-16
HBAR = 1.054571817e-27
AMU = 1.66053906660e-24
C_LIGHT = 2.99792458e10
N_AVOGADRO = 6.02214076e23


def reduced_mass(m1, m2):
    return m1 * m2 / (m1 + m2)


def neutron_capture_rate_mackeown(Z, A, T9, S_n, level_density_param=10.0):
    T = T9 * 1e9
    kT_MeV = K_BOLTZMANN * T * 6.241509e5





    mu = reduced_mass(AMU, AMU)
    thermal_wavelength = HBAR / np.sqrt(2.0 * np.pi * mu * K_BOLTZMANN * T)




    v_T = np.sqrt(2.0 * K_BOLTZMANN * T / AMU)
    de_broglie = HBAR / (AMU * v_T)


    g_factor = (A + 1.0) / A


    gamma_ratio = 0.1 * (S_n / kT_MeV) ** 2


    E_star = S_n + kT_MeV
    level_spacing = np.exp(-2.0 * np.sqrt(level_density_param * E_star))


    sigma_cap = 2.0 * np.pi * de_broglie ** 2 * g_factor * gamma_ratio / level_spacing
    rate = v_T * sigma_cap


    if np.isnan(rate) or np.isinf(rate) or rate < 0:
        rate = 1e-30
    return rate


def photodisintegration_rate(Z, A, T9, S_n, capture_rate):
    T = T9 * 1e9
    kT_MeV = K_BOLTZMANN * T * 6.241509e5






    raise NotImplementedError("Hole 1: photodisintegration_rate 核心公式待实现")


def beta_decay_rate(T_half):
    if T_half <= 0 or np.isnan(T_half) or np.isinf(T_half):
        return 1e-30
    return np.log(2.0) / T_half


def alpha_decay_rate(Z, A, Q_alpha):
    if Q_alpha <= 0.1:
        return 1e-30

    a_gn = -25.0
    b_gn = 1.5
    log_rate = a_gn - b_gn * Z / np.sqrt(Q_alpha)
    rate = 10.0 ** log_rate
    if np.isnan(rate) or rate < 0:
        rate = 1e-30
    return rate


def fission_rate(Z, A, n_n_density, T9):
    if Z < 90 or A < 230:
        return 0.0

    sigma_f = 1e-24 * max(0.0, (A - 220) / 20.0)
    rate = n_n_density * sigma_f
    return max(rate, 0.0)


def build_reaction_rate_table(nuclides, T9_range, S_n_table, T_half_table):
    rates = {
        'capture': {},
        'photodis': {},
        'beta': {},
        'alpha': {},
        'fission': {}
    }
    for z, n, a in nuclides:
        key = (z, a)
        S_n = S_n_table.get(key, 8.0)
        T_half = T_half_table.get(key, 1.0)

        cap_rates = []
        phot_rates = []
        for T9 in T9_range:
            cr = neutron_capture_rate_mackeown(z, a, T9, S_n)
            cap_rates.append(cr)
            pr = photodisintegration_rate(z, a, T9, S_n, cr)
            phot_rates.append(pr)

        rates['capture'][key] = np.array(cap_rates)
        rates['photodis'][key] = np.array(phot_rates)
        rates['beta'][key] = beta_decay_rate(T_half)
        rates['alpha'][key] = alpha_decay_rate(z, a, 5.0)
        rates['fission'][key] = 0.0

    return rates


def test_reaction_rates():
    T9 = 1.5
    cr = neutron_capture_rate_mackeown(26, 56, T9, 8.0)
    pr = photodisintegration_rate(26, 56, T9, 8.0, cr)
    br = beta_decay_rate(1.0)
    ar = alpha_decay_rate(92, 238, 4.5)
    fr = fission_rate(92, 238, 1e30, T9)
    print(f"[reaction_rates] n-capture rate = {cr:.3e} cm^3/s")
    print(f"[reaction_rates] photodis rate = {pr:.3e} s^{-1}")
    print(f"[reaction_rates] beta decay rate = {br:.3e} s^{-1}")
    print(f"[reaction_rates] alpha decay rate = {ar:.3e} s^{-1}")
    print(f"[reaction_rates] fission rate = {fr:.3e} s^{-1}")


if __name__ == "__main__":
    test_reaction_rates()
