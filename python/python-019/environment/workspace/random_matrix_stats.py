
import numpy as np
from scipy.special import gammaln


def log_beta(a, b):
    return gammaln(a) + gammaln(b) - gammaln(a + b)


def incomplete_beta(x, p, q):
    acu = 1e-14
    value = x
    ifault = 0

    if p <= 0.0 or q <= 0.0:
        ifault = 1
        return value, ifault
    if x < 0.0 or x > 1.0:
        ifault = 2
        return value, ifault
    if x == 0.0 or x == 1.0:
        return value, ifault

    psq = p + q
    cx = 1.0 - x

    if p < psq * x:
        xx = cx
        cx = x
        pp = q
        qq = p
        indx = 1
    else:
        xx = x
        pp = p
        qq = q
        indx = 0

    term = 1.0
    ai = 1.0
    value = 1.0
    ns = int(np.floor(qq + cx * psq))

    rx = xx / cx
    temp = qq - ai
    if ns == 0:
        rx = xx

    while True:
        term = term * temp * rx / (pp + ai)
        value = value + term
        temp = abs(term)

        if temp <= acu and temp <= acu * value:
            beta_log = log_beta(pp, qq)
            value = value * np.exp(pp * np.log(xx) + (qq - 1.0) * np.log(cx) - beta_log) / pp
            if indx:
                value = 1.0 - value
            break

        ai += 1.0
        ns -= 1
        if ns >= 0:
            temp = qq - ai
            if ns == 0:
                rx = xx
        else:
            temp = psq
            psq += 1.0

    return value, ifault


def level_spacing_ratios(eigenvalues, sort_by='real'):
    if sort_by == 'real':
        idx = np.argsort(eigenvalues.real)
    elif sort_by == 'imag':
        idx = np.argsort(eigenvalues.imag)
    else:
        idx = np.argsort(np.abs(eigenvalues))

    E_sorted = eigenvalues[idx]
    spacings = np.abs(np.diff(E_sorted))
    if len(spacings) < 2:
        return spacings, np.array([])

    ratios = np.minimum(spacings[:-1], spacings[1:]) / np.maximum(spacings[:-1], spacings[1:])
    return spacings, ratios


def wigner_poisson_mixture_cdf(s, alpha):
    wigner_cdf = 1.0 - np.exp(-np.pi * s ** 2 / 4.0)
    poisson_cdf = 1.0 - np.exp(-s)
    return alpha * wigner_cdf + (1.0 - alpha) * poisson_cdf


def generate_ginibre_spectrum(N, seed=42):
    rng = np.random.default_rng(seed)
    M = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))) / np.sqrt(2.0 * N)
    return np.linalg.eigvals(M)


def analyze_spacing_statistics(eigenvalues, num_bins=20):
    spacings, ratios = level_spacing_ratios(eigenvalues, sort_by='real')
    if len(spacings) == 0:
        return None
    mean_s = spacings.mean()
    if mean_s < 1e-15:
        return None
    s_norm = spacings / mean_s

    hist, bin_edges = np.histogram(s_norm, bins=num_bins, range=(0.0, 4.0), density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])


    best_alpha = 0.0
    best_chi2 = 1e308
    for alpha in np.linspace(0.0, 1.0, 21):
        pdf = (
            alpha * (np.pi * bin_centers / 2.0) * np.exp(-np.pi * bin_centers ** 2 / 4.0)
            + (1.0 - alpha) * np.exp(-bin_centers)
        )
        chi2 = np.sum((hist - pdf) ** 2 / (pdf + 1e-10))
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_alpha = alpha

    return {
        'bin_centers': bin_centers,
        'histogram': hist,
        'alpha_fit': best_alpha,
        'mean_spacing': mean_s,
        'ratios': ratios,
    }
