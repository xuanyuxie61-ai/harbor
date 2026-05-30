
import numpy as np



_DEFAULTS = {
    "c": np.sqrt(3.0) / 15.0,
    "delta": 1.0 / 75.0,
    "m": 3.0,
    "t0": 0.0,
    "tstop": 4.0,
}


def porous_medium_parameters(c_user=None, delta_user=None, m_user=None,
                             t0_user=None, tstop_user=None):
    params = dict(_DEFAULTS)
    if c_user is not None:
        params["c"] = float(c_user)
    if delta_user is not None:
        params["delta"] = float(delta_user)
    if m_user is not None:
        params["m"] = float(m_user)
    if t0_user is not None:
        params["t0"] = float(t0_user)
    if tstop_user is not None:
        params["tstop"] = float(tstop_user)
    return params


def porous_medium_exact(x, t, params=None):
    if params is None:
        params = porous_medium_parameters()

    c = params["c"]
    delta = params["delta"]
    m = params["m"]

    alpha = 1.0 / (m - 1.0)
    beta = 1.0 / (m + 1.0)
    gamma = (m - 1.0) / (2.0 * m * (m + 1.0))

    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)

    scalar_input = (x.ndim == 0 and t.ndim == 0)
    x = np.atleast_1d(x)
    t = np.atleast_1d(t)


    x, t = np.broadcast_arrays(x, t)

    bot = (t + delta) ** beta
    factor = c - gamma * (x / bot) ** 2


    u = np.zeros_like(factor)
    ut = np.zeros_like(factor)
    ux = np.zeros_like(factor)
    uxx = np.zeros_like(factor)

    mask = factor > 0.0
    if np.any(mask):
        f = factor[mask]
        u[mask] = (t[mask] + delta) ** (-beta) * f ** alpha

        ut[mask] = (2.0 * alpha * beta * gamma
                    * (t[mask] + delta) ** (-1.0 - 3.0 * beta)
                    * x[mask] ** 2 * f ** (alpha - 1.0)
                    - beta * (t[mask] + delta) ** (-1.0 - beta) * f ** alpha)

        ux[mask] = (-2.0 * alpha * gamma
                    * (t[mask] + delta) ** (-3.0 * beta)
                    * x[mask] * f ** (alpha - 1.0))

        uxx[mask] = (4.0 * (alpha - 1.0) * alpha * gamma ** 2
                     * (t[mask] + delta) ** (-5.0 * beta)
                     * x[mask] ** 2 * f ** (alpha - 2.0)
                     - 2.0 * alpha * gamma
                     * (t[mask] + delta) ** (-3.0 * beta)
                     * f ** (alpha - 1.0))

    if scalar_input:
        return u.item(), ut.item(), ux.item(), uxx.item()
    return u, ut, ux, uxx


def blast_wave_energy_density_profile(r_cm, t_s, E_iso=1e53,
                                      n_ism=1.0, gamma_ad=4.0 / 3.0):
    m_p = 1.6726219e-24
    c = 2.99792458e10
    Gamma_0 = 300.0

    r_dec = ((3.0 * E_iso) / (4.0 * np.pi * n_ism * m_p * c ** 2 * Gamma_0 ** 2)) ** (1.0 / 3.0)
    t_dec = r_dec / (2.0 * Gamma_0 ** 2 * c)


    if t_dec <= 0.0:
        t_dec = 1.0
    if r_dec <= 0.0:
        r_dec = 1.0

    xi = r_cm / r_dec
    tau = t_s / t_dec


    m_pme = (gamma_ad + 1.0) / (gamma_ad - 1.0)
    params = porous_medium_parameters(m_user=m_pme)

    u, _, _, _ = porous_medium_exact(xi, tau, params=params)

    scale = E_iso / (4.0 * np.pi * r_dec ** 3)
    eps = scale * u
    eps = np.clip(eps, 0.0, None)
    return eps
