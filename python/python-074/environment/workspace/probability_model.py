
import numpy as np
from scipy.special import gammaln, beta as beta_func


def alogam(x):
    if x <= 0.0:
        return 0.0, 1

    y = x
    if x < 7.0:
        f = 1.0
        z = y
        while z < 7.0:
            f *= z
            z += 1.0
        y = z
        f = -np.log(f)
    else:
        f = 0.0

    z = 1.0 / (y * y)
    value = (
        f
        + (y - 0.5) * np.log(y)
        - y
        + 0.918938533204673
        + (((-0.000595238095238 * z + 0.000793650793651) * z
            - 0.002777777777778) * z + 0.083333333333333) / y
    )
    return value, 0


def log_beta(p, q):
    if p <= 0.0 or q <= 0.0:
        raise ValueError("log_beta: p 和 q 必须为正。")
    lp, _ = alogam(p)
    lq, _ = alogam(q)
    lpq, _ = alogam(p + q)
    return lp + lq - lpq


def incomplete_beta_series(x, p, q, max_iter=1000, tol=1e-14):
    if x < 0.0 or x > 1.0:
        raise ValueError("incomplete_beta_series: x 必须在 [0,1] 内。")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0


    if x > 0.5:
        return 1.0 - incomplete_beta_series(1.0 - x, q, p, max_iter, tol)




    log_b = log_beta(p, q)
    prefactor = np.exp(p * np.log(x) - log_b - np.log(p))

    coeff = 1.0
    sum_series = 1.0
    for n in range(1, max_iter):
        coeff *= (p + n - 1.0) * (1.0 - q + n - 1.0) / ((p + n) * n) * x
        sum_series += coeff
        if abs(coeff) < tol * abs(sum_series):
            break

    return prefactor * sum_series


def incomplete_beta(x, p, q):
    try:
        return incomplete_beta_series(x, p, q)
    except Exception:
        from scipy.special import betainc
        return betainc(p, q, x)


def phase_cdf(theta, p, q, period=2.0 * np.pi):
    if theta < 0.0:
        return 0.0
    if theta > period:
        return 1.0
    x = theta / period
    return incomplete_beta(x, p, q)


def phase_pdf(theta, p, q, period=2.0 * np.pi):
    if theta < 0.0 or theta > period:
        return 0.0
    x = theta / period
    log_b = log_beta(p, q)
    log_f = (p - 1.0) * np.log(x) + (q - 1.0) * np.log(1.0 - x) - log_b - np.log(period)
    return np.exp(log_f)


def estimate_vortex_shedding_frequency(lift_history, dt):
    n = len(lift_history)
    if n < 10:
        return 0.0, 0.0


    signal = lift_history - np.mean(lift_history)


    f_signal = np.fft.fft(signal, n=2 * n)
    autocorr = np.fft.ifft(f_signal * np.conj(f_signal)).real
    autocorr = autocorr[:n]
    if abs(autocorr[0]) > 1e-15:
        autocorr /= autocorr[0]
    else:
        autocorr[:] = 0.0


    peak_idx = None
    for i in range(2, n // 2):
        if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1]:
            if autocorr[i] > 0.1:
                peak_idx = i
                break

    if peak_idx is None or peak_idx == 0:

        freqs = np.fft.rfftfreq(n, d=dt)
        fft_vals = np.abs(np.fft.rfft(signal))

        fft_vals[0] = 0.0
        peak_f_idx = np.argmax(fft_vals)
        f_est = freqs[peak_f_idx]
    else:
        f_est = 1.0 / (peak_idx * dt)

    return f_est, f_est


def fit_beta_parameters(phase_samples):
    x = np.asarray(phase_samples) / (2.0 * np.pi)
    x = np.clip(x, 1e-8, 1.0 - 1e-8)

    mean_x = np.mean(x)
    var_x = np.var(x)

    if var_x < 1e-12:

        p = q = 1000.0
        return p, q

    factor = mean_x * (1.0 - mean_x) / var_x - 1.0
    p = mean_x * factor
    q = (1.0 - mean_x) * factor

    p = max(p, 0.1)
    q = max(q, 0.1)
    return p, q


def log_normal_cl_cdf(cl_amp, mu_ln, sigma_ln):
    from scipy.stats import norm
    if cl_amp <= 0:
        return 0.0
    z = (np.log(cl_amp) - mu_ln) / sigma_ln
    return norm.cdf(z)
