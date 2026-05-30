import numpy as np

R = 8.314


def arrhenius_with_deactivation(value_25, T, E_a, dS, H_d):
    T = float(T)
    if T <= 0:
        return 0.0
    term1 = np.exp(E_a * (T - 298.15) / (298.15 * R * T))
    term2_num = 1.0 + np.exp((298.15 * dS - H_d) / (298.15 * R))
    term2_den = 1.0 + np.exp((T * dS - H_d) / (R * T))
    return value_25 * term1 * term2_num / max(term2_den, 1e-14)


def electron_transport_rate(i_abs, j_max, alpha_e=0.425, theta=0.7):
    a = theta
    b = -(i_abs * alpha_e + j_max)
    c = i_abs * alpha_e * j_max
    disc = b ** 2 - 4.0 * a * c
    if disc < 0:
        disc = 0.0
    j = (-b - np.sqrt(disc)) / (2.0 * a)
    return max(j, 0.0)


def farquhar_photosynthesis(ci, oi, t_k, i_abs,
                            vcmax_25=80.0, jmax_25=136.0,
                            rd_25=1.2, kc_25=404.9, ko_25=278.4,
                            gamma_star_25=36.9,
                            Ea_vcmax=65330.0, Ea_jmax=43540.0,
                            Ea_rd=46390.0, Ea_kc=79430.0,
                            Ea_ko=36380.0, Ea_gamma=37830.0,
                            dS_vcmax=485.0, Hd_vcmax=150000.0,
                            dS_jmax=495.0, Hd_jmax=152000.0):

    vcmax = arrhenius_with_deactivation(vcmax_25, t_k, Ea_vcmax, dS_vcmax, Hd_vcmax)
    jmax = arrhenius_with_deactivation(jmax_25, t_k, Ea_jmax, dS_jmax, Hd_jmax)
    rd = arrhenius_with_deactivation(rd_25, t_k, Ea_rd, 490.0, 150000.0)
    kc = arrhenius_with_deactivation(kc_25, t_k, Ea_kc, 650.0, 150000.0)
    ko = arrhenius_with_deactivation(ko_25, t_k, Ea_ko, 650.0, 150000.0)
    gamma_star = arrhenius_with_deactivation(gamma_star_25, t_k, Ea_gamma, 650.0, 150000.0)







    raise NotImplementedError("Hole 1: 请补全 FvCB 光合模型核心公式")


def temperature_sensitivity_centered(ci, oi, t_k, i_abs, h=0.5, **kwargs):
    an_plus, _, _, _, _ = farquhar_photosynthesis(ci, oi, t_k + h, i_abs, **kwargs)
    an_minus, _, _, _, _ = farquhar_photosynthesis(ci, oi, t_k - h, i_abs, **kwargs)
    d_adt = (an_plus - an_minus) / (2.0 * h)
    return d_adt


def canopy_photosynthesis_integrated(z_levels, lai_profile, radiation_profile,
                                     temperature_profile, ci, oi,
                                     vcmax_25=80.0, jmax_25=136.0, **kwargs):
    dz = np.diff(z_levels, prepend=0.0)
    dz = np.maximum(dz, 1e-6)
    a_total = 0.0
    for i in range(len(z_levels)):
        lai = lai_profile[i]
        i_abs = radiation_profile[i] * 0.85
        t_k = temperature_profile[i] + 273.15
        an, _, _, _, _ = farquhar_photosynthesis(ci, oi, t_k, i_abs,
                                                  vcmax_25=vcmax_25,
                                                  jmax_25=jmax_25, **kwargs)
        a_total += max(an, 0.0) * lai * dz[i]
    return a_total
