"""
假设检验模块：统计显著性分析
Statistical hypothesis testing for flow harmonics significance.

Algorithms sourced from:
  - 1194_jake-soloff_principal-agent-hypothesis-testing (statistical tests)
"""
import numpy as np
from scipy import stats


def ttest_v2_nonzero(v2_samples, alpha=0.05):
    """
    One-sample t-test: H0: v2 = 0 vs H1: v2 > 0

    From 1194: test whether measured elliptic flow is statistically
    different from zero.

    Parameters
    ----------
    v2_samples : ndarray
        Event-by-event v2 values
    alpha : float
        Significance level

    Returns
    -------
    dict with:
        't_stat' : float
        'p_value' : float
        'reject_null' : bool
        'mean_v2' : float
        'std_v2' : float
    """
    n = len(v2_samples)
    if n < 2:
        return {'t_stat': 0.0, 'p_value': 1.0, 'reject_null': False,
                'mean_v2': 0.0, 'std_v2': 0.0}

    mean_v2 = np.mean(v2_samples)
    std_v2 = np.std(v2_samples, ddof=1)

    if std_v2 < 1e-14:
        t_stat = 0.0
        p_value = 1.0
    else:
        t_stat = mean_v2 / (std_v2 / np.sqrt(n))
        # One-sided test
        p_value = 1.0 - stats.t.cdf(t_stat, df=n-1)

    return {
        't_stat': t_stat,
        'p_value': p_value,
        'reject_null': p_value < alpha,
        'mean_v2': mean_v2,
        'std_v2': std_v2,
        'n_events': n,
    }


def ftest_variance_ratio(v2_group1, v2_group2, alpha=0.05):
    """
    F-test for equality of variances between centrality classes.

    H0: σ1² = σ2²
    H1: σ1² ≠ σ2²

    Parameters
    ----------
    v2_group1, v2_group2 : ndarray
        v2 samples from two groups
    alpha : float
        Significance level

    Returns
    -------
    dict
        Test results
    """
    var1 = np.var(v2_group1, ddof=1)
    var2 = np.var(v2_group2, ddof=1)

    if var2 < 1e-14:
        f_stat = np.inf
        p_value = 0.0
    else:
        f_stat = var1 / var2
        df1 = len(v2_group1) - 1
        df2 = len(v2_group2) - 1
        p_value = 2.0 * min(stats.f.cdf(f_stat, df1, df2),
                           1.0 - stats.f.cdf(f_stat, df1, df2))

    return {
        'f_stat': f_stat,
        'p_value': p_value,
        'reject_null': p_value < alpha,
        'var1': var1,
        'var2': var2,
    }


def kolmogorov_smirnov_test(v2_data, v2_model):
    """
    KS test: are data and model from same distribution?

    Parameters
    ----------
    v2_data : ndarray
    v2_model : ndarray

    Returns
    -------
    dict
    """
    ks_stat, p_value = stats.ks_2samp(v2_data, v2_model)
    return {
        'ks_stat': ks_stat,
        'p_value': p_value,
        'reject_null': p_value < 0.05,
    }


def chi_squared_test(observed, expected, dof=None):
    """
    Chi-squared goodness-of-fit test.

    Parameters
    ----------
    observed : ndarray
    expected : ndarray
    dof : int, optional
        Degrees of freedom

    Returns
    -------
    dict
    """
    if dof is None:
        dof = len(observed) - 1

    chi2 = np.sum((observed - expected)**2 / np.maximum(expected, 1e-10))
    p_value = 1.0 - stats.chi2.cdf(chi2, dof)

    return {
        'chi2': chi2,
        'p_value': p_value,
        'reject_null': p_value < 0.05,
        'dof': dof,
    }


def confidence_interval(v2_samples, confidence=0.95):
    """
    Compute confidence interval for mean v2.

    Parameters
    ----------
    v2_samples : ndarray
    confidence : float

    Returns
    -------
    dict
    """
    n = len(v2_samples)
    mean = np.mean(v2_samples)
    sem = stats.sem(v2_samples)
    ci = stats.t.interval(confidence, df=n-1, loc=mean, scale=sem)

    return {
        'mean': mean,
        'ci_lower': ci[0],
        'ci_upper': ci[1],
        'confidence': confidence,
    }


def anova_vn_by_centrality(vn_by_centrality):
    """
    One-way ANOVA: are v_n means different across centrality classes?

    Parameters
    ----------
    vn_by_centrality : list of ndarray
        v_n samples per centrality bin

    Returns
    -------
    dict
    """
    if len(vn_by_centrality) < 2:
        return {'f_stat': 0.0, 'p_value': 1.0, 'reject_null': False}

    f_stat, p_value = stats.f_oneway(*vn_by_centrality)
    return {
        'f_stat': f_stat,
        'p_value': p_value,
        'reject_null': p_value < 0.05,
        'n_groups': len(vn_by_centrality),
    }


def bootstrap_confidence_interval(v2_samples, n_bootstrap=1000, confidence=0.95):
    """
    Bootstrap confidence interval for v2.

    Parameters
    ----------
    v2_samples : ndarray
    n_bootstrap : int
    confidence : float

    Returns
    -------
    dict
    """
    rng = np.random.default_rng(42)
    n = len(v2_samples)
    bootstrap_means = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        sample = rng.choice(v2_samples, size=n, replace=True)
        bootstrap_means[i] = np.mean(sample)

    alpha = 1.0 - confidence
    ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return {
        'mean': np.mean(v2_samples),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'bootstrap_std': np.std(bootstrap_means),
    }
