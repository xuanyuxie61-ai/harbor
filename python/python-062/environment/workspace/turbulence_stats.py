
import numpy as np


def plane_average(field):
    return np.mean(np.mean(field, axis=0), axis=0)


def volume_average(field):
    return np.mean(field)


def compute_tke(u, v, w):

    up = u - volume_average(u)
    vp = v - volume_average(v)
    wp = w - volume_average(w)

    tke = 0.5 * (up**2 + vp**2 + wp**2)
    tke_mean = volume_average(tke)
    return tke, tke_mean


def compute_reynolds_stresses(u, v, w):
    up = u - volume_average(u)
    vp = v - volume_average(v)
    wp = w - volume_average(w)

    R = {
        'uu': volume_average(up * up),
        'vv': volume_average(vp * vp),
        'ww': volume_average(wp * wp),
        'uv': volume_average(up * vp),
        'uw': volume_average(up * wp),
        'vw': volume_average(vp * wp),
    }
    return R


def compute_heat_flux(u, v, w, theta):
    up = u - volume_average(u)
    vp = v - volume_average(v)
    wp = w - volume_average(w)
    tp = theta - volume_average(theta)

    qx = volume_average(up * tp)
    qy = volume_average(vp * tp)
    qz = volume_average(wp * tp)
    return qx, qy, qz


def longitudinal_structure_function(u, axis=0, max_lag=None):
    if max_lag is None:
        max_lag = u.shape[axis] // 2

    D_ll = np.zeros(max_lag, dtype=np.float64)
    counts = np.zeros(max_lag, dtype=np.int64)


    for lag in range(1, max_lag + 1):
        slc1 = [slice(None)] * 3
        slc2 = [slice(None)] * 3
        slc1[axis] = slice(lag, None)
        slc2[axis] = slice(None, -lag)

        diff = u[tuple(slc1)] - u[tuple(slc2)]
        D_ll[lag - 1] = np.sum(diff**2)
        counts[lag - 1] = diff.size

    r = np.arange(1, max_lag + 1)
    D_ll = D_ll / np.maximum(counts, 1)

    return r, D_ll


def monte_carlo_turbulence_stat(samples, dim=3, eval_num=10000):

    total = 0.0
    for _ in range(eval_num):
        x = np.random.rand(dim)
        total += samples(dim, x)

    volume = 1.0
    result = total * volume / eval_num
    return result


def compute_kolmogorov_scales(epsilon, nu):
    eta = (nu**3 / epsilon) ** 0.25
    tau_eta = (nu / epsilon) ** 0.5
    u_eta = (nu * epsilon) ** 0.25


    Re_lambda = (15.0 * epsilon * (nu / epsilon) ** 0.5 / nu) ** 0.5

    return eta, tau_eta, u_eta, Re_lambda
