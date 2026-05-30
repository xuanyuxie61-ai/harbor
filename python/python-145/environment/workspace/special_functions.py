
import numpy as np


def log_normal_pdf(x, mu, sigma):
    if sigma <= 0.0:
        raise ValueError("log_normal_pdf: sigma 必须大于 0")
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x, dtype=float)
    mask = x > 0.0
    if np.any(mask):
        result[mask] = np.exp(-0.5 * ((np.log(x[mask]) - mu) / sigma) ** 2) / (
            sigma * x[mask] * np.sqrt(2.0 * np.pi)
        )
    return result


def log_normal_cdf(x, mu, sigma):
    if sigma <= 0.0:
        raise ValueError("log_normal_cdf: sigma 必须大于 0")
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x, dtype=float)
    mask = x > 0.0
    if np.any(mask):
        z = (np.log(x[mask]) - mu) / sigma
        result[mask] = 0.5 * (1.0 + _erf_approx(z))
    return result


def log_normal_cdf_inv(cdf, mu, sigma):
    if sigma <= 0.0:
        raise ValueError("log_normal_cdf_inv: sigma 必须大于 0")
    cdf = np.asarray(cdf, dtype=float)
    if np.any(cdf < 0.0) or np.any(cdf > 1.0):
        raise ValueError("log_normal_cdf_inv: cdf 必须在 [0, 1] 区间内")
    logx = normal_01_cdf_inv(cdf)
    return np.exp(mu + sigma * logx)


def log_normal_sample(mu, sigma, size=None):
    if sigma <= 0.0:
        raise ValueError("log_normal_sample: sigma 必须大于 0")
    u = np.random.uniform(0.0, 1.0, size=size)

    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    return log_normal_cdf_inv(u, mu, sigma)


def normal_01_cdf_inv(p):
    p = np.asarray(p, dtype=float)
    if np.any(p <= 0.0) or np.any(p >= 1.0):
        raise ValueError("normal_01_cdf_inv: p 必须在 (0, 1) 开区间内")

    a = np.array([
        3.3871328727963666080, 1.3314166789178437745e2,
        1.9715909503065514427e3, 1.3731693765509461125e4,
        4.5921953931549871457e4, 6.7265770927008700853e4,
        3.3430575583588128105e4, 2.5090809287301226727e3
    ])
    b = np.array([
        1.0, 4.2313330701600911252e1,
        6.8718700749205790830e2, 5.3941960214247511077e3,
        2.1213794301586595867e4, 3.9307895800092710610e4,
        2.8729085735721942674e4, 5.2264952788528545610e3
    ])
    c = np.array([
        1.42343711074968357734, 4.63033784615654529590,
        5.76949722146069140550, 3.64784832476320460504,
        1.27045825245236838258, 2.41780725177450611770e-1,
        2.27238449892691845833e-2, 7.74545014278341407640e-4
    ])
    d = np.array([
        1.0, 2.05319162663775882187,
        1.67638483018380384940, 6.89767334985100004550e-1,
        1.48103976427480074590e-1, 1.51986665636164571966e-2,
        5.47593808499534494600e-4, 1.05075007164441684324e-9
    ])
    e = np.array([
        6.65790464350110377720, 5.46378491116411436990,
        1.78482653991729133580, 2.96560571828504891230e-1,
        2.65321895265761230930e-2, 1.24266094738807843860e-3,
        2.71155556874348757815e-5, 2.01033439929228813265e-7
    ])
    f = np.array([
        1.0, 5.99832206555887937690e-1,
        1.36929880922735805310e-1, 1.48753612908506148525e-2,
        7.86869131145613259100e-4, 1.84631831751005468180e-5,
        1.42151175831644588870e-7, 2.04426310338993978564e-15
    ])

    const1 = 0.180625
    const2 = 1.6
    split1 = 0.425
    split2 = 5.0

    q = p - 0.5
    abs_q = np.abs(q)
    value = np.zeros_like(p, dtype=float)


    mask_center = abs_q <= split1
    if np.any(mask_center):
        r = const1 - q[mask_center] * q[mask_center]
        value[mask_center] = q[mask_center] * _poly_eval(7, a, r) / _poly_eval(7, b, r)


    mask_tail = ~mask_center
    if np.any(mask_tail):
        r = np.where(q[mask_tail] < 0.0, p[mask_tail], 1.0 - p[mask_tail])
        r = np.clip(r, 1e-300, None)
        r = np.sqrt(-np.log(r))

        mask_mid = mask_tail & (r <= split2)
        mask_far = mask_tail & (r > split2)

        if np.any(mask_mid):
            r_mid = r[mask_mid] - const2
            value[mask_mid] = _poly_eval(7, c, r_mid) / _poly_eval(7, d, r_mid)
            neg = q[mask_mid] < 0.0
            value[mask_mid] = np.where(neg, -value[mask_mid], value[mask_mid])

        if np.any(mask_far):
            r_far = r[mask_far] - split2
            val_far = _poly_eval(7, e, r_far) / _poly_eval(7, f, r_far)
            neg = q[mask_far] < 0.0
            val_far = np.where(neg, -val_far, val_far)
            value[mask_far] = val_far

    return value


def _poly_eval(n, c, x):
    x = np.asarray(x, dtype=float)
    val = np.full_like(x, c[n], dtype=float)
    for i in range(n - 1, -1, -1):
        val = val * x + c[i]
    return val


def _erf_approx(z):
    z = np.asarray(z, dtype=float)
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p_erf = 0.3275911

    sign_z = np.sign(z)
    z_abs = np.abs(z)
    t = 1.0 / (1.0 + p_erf * z_abs)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-z_abs * z_abs)
    return sign_z * y


def lambert_w(x, branch=0, offset_mode=0):
    x = np.asarray(x, dtype=float)
    W = np.full_like(x, np.nan, dtype=float)

    niter = 1
    em = -np.exp(-1.0)
    em9 = -np.exp(-9.0)
    c13 = 1.0 / 3.0
    c23 = 2.0 * c13
    em2 = 2.0 / em
    d12 = -em2
    tb = 0.5 ** 52
    tb2 = np.sqrt(tb)
    x0 = tb ** (1.0 / 6.0) * 0.5
    x1 = (1.0 - 17.0 * tb ** (2.0 / 7.0)) * em
    an3 = 8.0 / 3.0
    an4 = 135.0 / 83.0
    an5 = 166.0 / 39.0
    an6 = 3167.0 / 3549.0
    s2 = np.sqrt(2.0)
    s21 = 2.0 * s2 - 3.0
    s22 = 4.0 - 3.0 * s2
    s23 = s2 - 2.0

    branch_arr = np.broadcast_to(np.asarray(branch), x.shape)

    for idx in np.ndindex(x.shape):
        xi = x[idx]
        nb = int(branch_arr[idx])

        if offset_mode == 1:
            delx = xi
            if delx < 0.0:
                continue
            xx = xi + em
        else:
            if xi < em:
                continue
            elif np.isclose(xi, em, atol=1e-15):
                W[idx] = -1.0
                continue
            xx = xi
            delx = xx - em

        if nb == 0:

            if np.abs(xx) <= x0:
                W[idx] = xx / (1.0 + xx / (1.0 + xx / (2.0 + xx / (0.6 + 0.34 * xx))))
                continue
            elif xx <= x1:
                reta = np.sqrt(d12 * delx)
                W[idx] = reta / (1.0 + reta / (3.0 + reta / (reta / (an4 + reta / (reta * an6 + an5)) + an3))) - 1.0
                continue
            elif xx <= 20.0:
                reta = s2 * np.sqrt(1.0 - xx / em)
                an2 = 4.612634277343749 * np.sqrt(np.sqrt(reta + 1.09556884765625))
                W[idx] = reta / (1.0 + reta / (3.0 + (s21 * an2 + s22) * reta / (s23 * (an2 + reta)))) - 1.0
            else:
                zl = np.log(xx)
                W[idx] = np.log(xx / np.log(xx / zl ** np.exp(-1.124491989777808 / (0.4225028202459761 + zl))))
        else:

            if xx >= 0.0:
                continue
            elif xx <= x1:
                reta = np.sqrt(d12 * delx)
                W[idx] = reta / (reta / (3.0 + reta / (reta / (an4 + reta / (reta * an6 - an5)) - an3)) - 1.0) - 1.0
                continue
            elif xx <= em9:
                zl = np.log(-xx)
                t = -1.0 - zl
                ts = np.sqrt(t)
                W[idx] = zl - (2.0 * ts) / (s2 + (c13 - t / (270.0 + ts * 127.0471381349219)) * ts)
            else:
                zl = np.log(-xx)
                eta = 2.0 - em2 * xx
                W[idx] = np.log(xx / np.log(-xx / ((1.0 - 0.5043921323068457 * (zl + 1.0)) * (np.sqrt(eta) + eta / 3.0) + 1.0)))


        wv = W[idx]
        if not np.isnan(wv) and wv != 0.0:
            zn = np.log(xx / wv) - wv
            temp = 1.0 + wv
            temp2 = temp + c23 * zn
            temp2 = 2.0 * temp * temp2
            w_new = wv * (1.0 + (zn / temp) * (temp2 - zn) / (temp2 - 2.0 * zn))
            W[idx] = w_new

    return W
