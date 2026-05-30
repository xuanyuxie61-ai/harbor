# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import gammaln


HBARC = 197.3269804
M_NUCLEON = 939.0
HBAR2_2M = HBARC**2 / (2.0 * M_NUCLEON)



SKYRME_PARAMS = {
    't0': -1800.0,
    't1': 481.0,
    't2': -549.0,
    't3': 13674.0,
    'x0': 0.356,
    'x1': -0.511,
    'x2': -1.0,
    'x3': 1.326,
    'alpha': 1.0 / 6.0,
    'W0': 120.0,
}


def alnorm(x, upper):
    from math import erfc, sqrt
    if upper:
        if x < 0.0:
            return 1.0 - alnorm(-x, True)
        return 0.5 * erfc(x / sqrt(2.0))
    else:
        if x < 0.0:
            return 1.0 - alnorm(-x, True)
        return 1.0 - 0.5 * erfc(x / sqrt(2.0))


def betain(x, a, b, albeta):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    ps = a
    cx = 1.0 - x
    if b * x <= 1.0 and x <= 0.95:
        term = 1.0
        ai = 1.0
        value = 1.0
        while True:
            term = term * (b - ai) * x / ai
            value = value + term
            if abs(term) <= 1e-15:
                break
            ai = ai + 1.0
        return value * np.exp(a * np.log(x) - albeta) / a

    return 0.5


def tnc(t, df, delta):
    alnrpi = 0.5723649429247001
    errmax = 1.0e-10
    itrmax = 100
    r2pi = 0.7978845608028654

    value = 0.0
    ifault = 0

    if df <= 0.0:
        ifault = 2
        return value, ifault

    tt = t
    del_param = delta
    negdel = 0

    if t < 0.0:
        negdel = 1
        tt = -tt
        del_param = -del_param

    en = 1.0
    x = t * t / (t * t + df)

    if x <= 0.0:
        ifault = 0
        value = value + alnorm(del_param, 1)
        if negdel:
            value = 1.0 - value
        return value, ifault

    lam = del_param * del_param
    p = 0.5 * np.exp(-0.5 * lam)
    q = r2pi * p * del_param
    s = 0.5 - p
    a = 0.5
    b = 0.5 * df
    rxb = (1.0 - x)**b
    albeta = alnrpi + gammaln(b) - gammaln(a + b)
    xodd = betain(x, a, b, albeta)
    godd = 2.0 * rxb * np.exp(a * np.log(x) - albeta)
    xeven = 1.0 - rxb
    geven = b * x * rxb
    value = p * xodd + q * xeven

    for _ in range(itrmax):
        a = a + 1.0
        xodd = xodd - godd
        xeven = xeven - geven
        if a + b - 1.0 <= 0 or a <= 0:
            break
        godd = godd * x * (a + b - 1.0) / a
        geven = geven * x * (a + b - 0.5) / (a + 0.5)
        p = p * lam / (2.0 * en)
        q = q * lam / (2.0 * en + 1.0)
        s = s - p
        en = en + 1.0
        value = value + p * xodd + q * xeven
        errbd = 2.0 * s * (xodd - godd)
        if errbd <= errmax:
            ifault = 0
            break
        if itrmax < en:
            ifault = 1
            break

    value = value + alnorm(del_param, 1)
    if negdel:
        value = 1.0 - value
    return value, ifault


def skyrme_energy_density(rho_n, rho_p, params=None):
    if params is None:
        params = SKYRME_PARAMS

    rho = rho_n + rho_p
    if rho <= 0.0:
        return 0.0, 0.0


    k_fn = (3.0 * np.pi**2 * rho_n)**(1.0/3.0)
    k_fp = (3.0 * np.pi**2 * rho_p)**(1.0/3.0)


    tau_n = (3.0/5.0) * (3.0 * np.pi**2)**(2.0/3.0) * rho_n**(5.0/3.0)
    tau_p = (3.0/5.0) * (3.0 * np.pi**2)**(2.0/3.0) * rho_p**(5.0/3.0)
    tau = tau_n + tau_p

    t0 = params['t0']
    t3 = params['t3']
    x0 = params['x0']
    x3 = params['x3']
    alpha = params['alpha']


    rho2 = rho * rho
    rho_alpha = rho**alpha



    e2 = 0.5 * t0 * ((1.0 + 0.5 * x0) * rho2 - (x0 + 0.5) * (rho_n**2 + rho_p**2))

    e3 = t3 / 24.0 * ((1.0 + 0.5 * x3) * rho_alpha * rho2
                      - (x3 + 0.5) * rho_alpha * (rho_n**2 + rho_p**2))


    e_kin = HBAR2_2M * tau

    energy_density = e_kin + e2 + e3



    de2_drho = t0 * ((1.0 + 0.5 * x0) * rho - (x0 + 0.5) * rho)
    de3_drho = t3 / 24.0 * ((1.0 + 0.5 * x3) * (alpha + 2.0) * rho_alpha * rho
                            - (x3 + 0.5) * (alpha + 2.0) * rho_alpha * rho)
    dekin_drho = HBAR2_2M * (3.0 * np.pi**2)**(2.0/3.0) * rho**(2.0/3.0)

    dH_drho = dekin_drho + de2_drho + de3_drho
    pressure = rho * dH_drho - energy_density

    return energy_density, pressure


def nuclear_matter_properties(rho, x_p=0.5, params=None):
    if rho <= 0.0:
        raise ValueError("密度必须大于0")
    if x_p < 0.0 or x_p > 1.0:
        raise ValueError("质子分数必须在[0,1]区间内")

    rho_n = rho * (1.0 - x_p)
    rho_p = rho * x_p

    e_dens, press = skyrme_energy_density(rho_n, rho_p, params)
    e_per_nucleon = e_dens / rho


    e_dens_pure_n, _ = skyrme_energy_density(rho, 0.0, params)
    e_dens_sym, _ = skyrme_energy_density(rho / 2.0, rho / 2.0, params)
    e_sym = e_dens_pure_n / rho - e_dens_sym / rho


    dr = 1e-4 * rho
    e1, _ = skyrme_energy_density(rho * (1.0 - dr) * (1.0 - x_p),
                                   rho * (1.0 - dr) * x_p, params)
    e2, _ = skyrme_energy_density(rho * (1.0 + dr) * (1.0 - x_p),
                                   rho * (1.0 + dr) * x_p, params)
    e0 = e_dens
    d2e = (e1 / (rho * (1.0 - dr)) - 2.0 * e0 / rho
           + e2 / (rho * (1.0 + dr))) / (dr * rho)**2
    K = 9.0 * rho**2 * d2e

    return {
        'energy_per_nucleon': e_per_nucleon,
        'pressure': press,
        'symmetry_energy': e_sym,
        'incompressibility': K,
        'density': rho,
        'proton_fraction': x_p
    }


def parameter_uncertainty_t_stat(estimates, true_value, std_errors, confidence=0.95):
    n = len(estimates)
    if n == 0:
        return 0.0, None, None

    df = max(n - 1, 1)
    t_vals = [(e - true_value) / se for e, se in zip(estimates, std_errors)]


    delta = (np.mean(estimates) - true_value) / (np.std(estimates, ddof=1) / np.sqrt(n) + 1e-15)


    alpha = 1.0 - confidence

    from scipy.stats import t as t_dist
    t_crit = t_dist.ppf(1.0 - alpha / 2.0, df)


    covered = 0
    for t_val in t_vals:
        p_val, _ = tnc(abs(t_val), df, abs(delta))
        if p_val >= alpha / 2.0:
            covered += 1

    coverage = covered / n
    margin = t_crit * np.std(estimates, ddof=1) / np.sqrt(n)
    ci_lower = np.mean(estimates) - margin
    ci_upper = np.mean(estimates) + margin

    return coverage, ci_lower, ci_upper


if __name__ == '__main__':

    rho = 0.16
    props = nuclear_matter_properties(rho, x_p=0.5)
    print(f"饱和密度: {rho} fm^-3")
    print(f"能量/核子: {props['energy_per_nucleon']:.2f} MeV")
    print(f"对称能: {props['symmetry_energy']:.2f} MeV")
    print(f"不可压缩系数: {props['incompressibility']:.2f} MeV")
