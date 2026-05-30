
import numpy as np



ECC_AMP = np.array([
    0.01860798, 0.01627522, 0.01300660, 0.00988829, 0.00336700,
    0.00333076, 0.00235400, 0.00140015, 0.00100700, 0.00085700
])
ECC_FREQ = np.array([
    0.01137420, 0.01300660, 0.01363260, 0.01486780, 0.01513080,
    0.01529980, 0.01546340, 0.01571620, 0.01586320, 0.01602460
])
ECC_PHASE = np.array([
    1.75464, 1.54336, 5.70929, 4.57677, 3.25909,
    5.68770, 2.43468, 3.27189, 4.32389, 5.25102
])


OBL_AMP = np.array([
    -0.02460765, -0.00225853, -0.00124802, -0.00085904, -0.00066601,
    -0.00055916, -0.00047169, -0.00033916, -0.00031762, -0.00026127
])
OBL_FREQ = np.array([
    0.02673800, 0.02733000, 0.02762800, 0.02791600, 0.02824600,
    0.02857800, 0.02887400, 0.02913600, 0.02936200, 0.02963200
])
OBL_PHASE = np.array([
    2.68040, 5.39300, 5.28800, 5.65600, 2.61800,
    3.43700, 2.83100, 0.58600, 5.70900, 1.42300
])


PREC_AMP = np.array([
    0.01919900, -0.01745800, 0.01480300, 0.01076600, 0.00926100,
    0.00784900, 0.00586800, 0.00578300, 0.00528600, 0.00493900
])
PREC_FREQ = np.array([
    0.02994600, 0.02777400, 0.02763300, 0.02748500, 0.02684700,
    0.02676500, 0.02668200, 0.02656800, 0.02640600, 0.02631100
])
PREC_PHASE = np.array([
    4.20732, 3.24587, 5.78354, 2.97566, 0.81878,
    3.59989, 5.85739, 5.98854, 1.34608, 5.68475
])

EPSILON_0 = 0.409092804
S0 = 1361.0


def chebyshev_evaluate(x, coef):
    nc = len(coef)
    if nc == 0:
        return 0.0
    if nc == 1:
        return coef[0]


    x = max(-1.0, min(1.0, x))

    b0 = coef[nc - 1]
    b1 = 0.0
    b2 = 0.0
    x2 = 2.0 * x

    for i in range(nc - 2, -1, -1):
        b2 = b1
        b1 = b0
        b0 = coef[i] - b2 + x2 * b1

    return 0.5 * (b0 - b2)


def chebyshev_fit(fvals, n):
    m = len(fvals)
    if m < n:
        n = m

    k = np.arange(m)
    t = np.cos(np.pi * (k + 0.5) / m)

    coef = np.zeros(n)
    for j in range(n):
        coef[j] = (2.0 / m) * np.sum(fvals * np.cos(j * np.arccos(t)))
    coef[0] *= 0.5
    return coef


def compute_eccentricity(time_kyr):
    is_scalar = np.isscalar(time_kyr)
    time_kyr = np.atleast_1d(np.asarray(time_kyr, dtype=float))
    M = np.sum(ECC_AMP[:, None] * np.sin(ECC_FREQ[:, None] * time_kyr[None, :] + ECC_PHASE[:, None]), axis=0)
    N = np.sum(ECC_AMP[:, None] * np.cos(ECC_FREQ[:, None] * time_kyr[None, :] + ECC_PHASE[:, None]), axis=0)
    e = np.sqrt(M ** 2 + N ** 2)

    e = np.clip(e, 0.0, 0.07)
    return float(e[0]) if is_scalar else e


def compute_obliquity(time_kyr):
    is_scalar = np.isscalar(time_kyr)
    time_kyr = np.atleast_1d(np.asarray(time_kyr, dtype=float))
    delta = np.sum(OBL_AMP[:, None] * np.sin(OBL_FREQ[:, None] * time_kyr[None, :] + OBL_PHASE[:, None]), axis=0)
    eps = EPSILON_0 + delta
    eps = np.clip(eps, 0.38, 0.45)
    return float(eps[0]) if is_scalar else eps


def compute_precession(time_kyr):
    is_scalar = np.isscalar(time_kyr)
    time_kyr = np.atleast_1d(np.asarray(time_kyr, dtype=float))
    prec = np.sum(PREC_AMP[:, None] * np.sin(PREC_FREQ[:, None] * time_kyr[None, :] + PREC_PHASE[:, None]), axis=0)
    prec = np.clip(prec, -0.07, 0.07)
    return float(prec[0]) if is_scalar else prec


def compute_orbital_elements(time_kyr):
    return compute_eccentricity(time_kyr), compute_obliquity(time_kyr), compute_precession(time_kyr)


def solar_longitude(day_of_year, ecc, prec):
    day_of_year = np.asarray(day_of_year, dtype=float)

    M = 2.0 * np.pi * day_of_year / 365.25

    E = M.copy() if isinstance(M, np.ndarray) else M
    for _ in range(5):
        E = M + ecc * np.sin(E)

    nu = 2.0 * np.arctan(np.sqrt((1.0 + ecc) / (1.0 - ecc)) * np.tan(E / 2.0))

    sin_omega = prec / max(ecc, 1e-12)
    sin_omega = np.clip(sin_omega, -1.0, 1.0)
    omega = np.arcsin(sin_omega)
    lambda_s = nu + omega
    lambda_s = np.mod(lambda_s, 2.0 * np.pi)
    return float(lambda_s) if np.isscalar(day_of_year) or day_of_year.ndim == 0 else lambda_s


def declination(day_of_year, ecc, eps, prec):
    lam = solar_longitude(day_of_year, ecc, prec)
    sin_delta = np.sin(eps) * np.sin(lam)
    sin_delta = np.clip(sin_delta, -1.0, 1.0)
    delta = np.arcsin(sin_delta)
    return delta


def daily_insolation(latitude_deg, day_of_year, ecc, eps, prec):
    phi = np.deg2rad(latitude_deg)
    delta = declination(day_of_year, ecc, eps, prec)


    tan_phi = np.tan(phi)
    tan_delta = np.tan(delta)
    cos_h0 = -tan_phi * tan_delta
    cos_h0 = np.clip(cos_h0, -1.0, 1.0)
    h0 = np.arccos(cos_h0)



    M = 2.0 * np.pi * day_of_year / 365.25
    rho2 = (1.0 - ecc ** 2) / ((1.0 + ecc * np.cos(M)) ** 2)
    rho2 = np.clip(rho2, 0.8, 1.3)

    insol = (S0 / np.pi) * rho2 * (h0 * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(h0))
    insol = np.clip(insol, 0.0, S0 * 1.5)
    return float(insol) if np.isscalar(latitude_deg) and np.isscalar(day_of_year) else insol


def annual_mean_insolation(latitude_deg, ecc, eps, prec, ndays=365):
    days = np.linspace(0, 365.25, ndays)
    insol = daily_insolation(latitude_deg, days, ecc, eps, prec)
    return np.trapezoid(insol, days) / 365.25


def compute_insolation_map(latitudes, time_kyr, ndays=73):
    nlat = len(latitudes)
    nt = len(time_kyr)
    insol_map = np.zeros((nlat, nt))

    for j in range(nt):
        e, eps, prec = compute_orbital_elements(time_kyr[j])
        for i in range(nlat):
            insol_map[i, j] = annual_mean_insolation(latitudes[i], e, eps, prec, ndays)

    return insol_map


def runge_function(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + 25.0 * x * x)


def runge_derivative(x):
    x = np.asarray(x, dtype=float)
    return -50.0 * x / ((1.0 + 25.0 * x * x) ** 2)


def test_interpolation_accuracy():
    n = 20

    k = np.arange(n)
    t_cheb = np.cos(np.pi * (k + 0.5) / n)
    f_cheb = runge_function(t_cheb)
    coef = chebyshev_fit(f_cheb, n)


    t_eq = np.linspace(-1, 1, n)
    f_eq = runge_function(t_eq)


    x_test = np.linspace(-0.99, 0.99, 200)
    f_true = runge_function(x_test)


    f_interp_cheb = np.array([chebyshev_evaluate(x, coef) for x in x_test])
    err_cheb = np.max(np.abs(f_interp_cheb - f_true))


    f_interp_eq = np.zeros_like(x_test)
    for i, x in enumerate(x_test):
        w = np.ones(n)
        for j in range(n):
            for k_idx in range(n):
                if k_idx != j:
                    w[j] *= (t_eq[j] - t_eq[k_idx])
            w[j] = 1.0 / w[j]
        num = np.sum(w * f_eq / (x - t_eq + 1e-15))
        den = np.sum(w / (x - t_eq + 1e-15))
        f_interp_eq[i] = num / den

    err_eq = np.max(np.abs(f_interp_eq - f_true))

    return err_cheb, err_eq
