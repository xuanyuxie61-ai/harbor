import numpy as np
from constants import TINY, M_HIGGS, GAMMA_H
from utils import bisection, muller_method, lu_factor_scaled, lu_solve, safe_divide




def poisson_likelihood(n_obs, mu, s, b):
    lam = mu * s + b
    if lam < TINY:
        lam = TINY

    log_fact = 0.0
    if n_obs > 0:
        log_fact = n_obs * np.log(n_obs) - n_obs + 0.5 * np.log(2.0 * np.pi * max(n_obs, 1.0))
    return n_obs * np.log(lam) - lam - log_fact


def profile_log_likelihood(n_obs_list, s_list, b_list, mu):
    total = 0.0
    for n, s, b in zip(n_obs_list, s_list, b_list):
        total += poisson_likelihood(n, mu, s, b)
    return total





def solve_mu_mle(n_obs_list, s_list, b_list, mu_min=0.0, mu_max=10.0):
    def dlnL_dmu(mu):
        total = 0.0
        for n, s, b in zip(n_obs_list, s_list, b_list):
            lam = mu * s + b
            if lam < TINY:
                lam = TINY
            total += (n - lam) * s / lam
        return total
    

    f_min = dlnL_dmu(mu_min)
    f_max = dlnL_dmu(mu_max)
    
    if f_min * f_max > 0:

        if abs(f_min) < abs(f_max):
            return mu_min, {"status": "boundary", "reason": "derivative_same_sign"}
        return mu_max, {"status": "boundary", "reason": "derivative_same_sign"}
    
    mu_hat, info = bisection(dlnL_dmu, mu_min, mu_max, tol=1.0e-10, max_iter=200)
    if mu_hat is None:

        mu_hat, info = muller_method(dlnL_dmu, mu_min, (mu_min + mu_max) / 2.0, mu_max, tol=1.0e-10)
    
    if mu_hat is None or mu_hat < 0:
        mu_hat = 0.0
    
    return mu_hat, info





def significance_simple(signal, background):
    if background <= 0:
        return 0.0
    return signal / np.sqrt(background)


def significance_likelihood_ratio(n_obs_list, s_list, b_list, mu_test=0.0):
    mu_hat, _ = solve_mu_mle(n_obs_list, s_list, b_list)
    mu_hat = max(mu_hat, 0.0)
    
    lnL_test = profile_log_likelihood(n_obs_list, s_list, b_list, mu_test)
    lnL_hat = profile_log_likelihood(n_obs_list, s_list, b_list, mu_hat)
    
    q = -2.0 * (lnL_test - lnL_hat)
    q = max(q, 0.0)
    
    Z = np.sqrt(q)
    return Z, q, mu_hat





def confidence_interval_mu(n_obs_list, s_list, b_list, cl=0.95):
    mu_hat, _ = solve_mu_mle(n_obs_list, s_list, b_list)
    lnL_max = profile_log_likelihood(n_obs_list, s_list, b_list, mu_hat)
    

    delta = 0.5 * 3.841
    
    def target(mu):
        return profile_log_likelihood(n_obs_list, s_list, b_list, mu) - lnL_max + delta
    

    mu_lower = 0.0
    if target(0.0) > 0:

        try:
            mu_lower, _ = bisection(target, 0.0, mu_hat, tol=1.0e-8)
            if mu_lower is None:
                mu_lower = 0.0
        except Exception:
            mu_lower = 0.0
    

    mu_upper = mu_hat

    scale = 2.0
    while scale < 1.0e6:
        mu_test = mu_hat + scale * max(mu_hat, 0.1)
        if target(mu_test) < 0:
            try:
                mu_upper, _ = bisection(target, mu_hat, mu_test, tol=1.0e-8)
                if mu_upper is None:
                    mu_upper = mu_test
            except Exception:
                mu_upper = mu_test
            break
        scale *= 2.0
    else:
        mu_upper = mu_hat + 1.0e6
    
    return mu_lower, mu_upper





def covariance_matrix_from_systematics(syst_errors):
    n = len(syst_errors)
    cov = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cov[i, j] = syst_errors[i] * syst_errors[j]
        cov[i, i] += syst_errors[i] ** 2
    return cov


def significance_with_systematics(signal, background, syst_bkg_frac=0.1):
    if background <= 0:
        return 0.0
    denom = np.sqrt(background + (syst_bkg_frac * background) ** 2)
    return signal / denom





def full_statistical_report(mass_bins, n_obs, n_bkg, n_sig_expected):
    n_obs = np.asarray(n_obs, dtype=float)
    n_bkg = np.asarray(n_bkg, dtype=float)
    n_sig = np.asarray(n_sig_expected, dtype=float)
    

    total_obs = np.sum(n_obs)
    total_bkg = np.sum(n_bkg)
    total_sig = np.sum(n_sig)
    

    simple_Z = significance_simple(total_obs - total_bkg, total_bkg)
    

    lr_Z, q_mu, mu_hat = significance_likelihood_ratio(n_obs.tolist(), n_sig.tolist(), n_bkg.tolist())
    

    mu_lo, mu_hi = confidence_interval_mu(n_obs.tolist(), n_sig.tolist(), n_bkg.tolist())
    

    Z_syst = significance_with_systematics(total_obs - total_bkg, total_bkg, syst_bkg_frac=0.15)
    
    return {
        "total_observed": total_obs,
        "total_background": total_bkg,
        "total_signal_expected": total_sig,
        "mu_hat": mu_hat,
        "mu_lower_95cl": mu_lo,
        "mu_upper_95cl": mu_hi,
        "significance_simple": simple_Z,
        "significance_likelihood": lr_Z,
        "significance_with_syst": Z_syst,
        "test_statistic_q0": q_mu,
        "mass_bins": mass_bins,
        "n_observed": n_obs,
        "n_background": n_bkg,
        "n_signal": n_sig,
    }
