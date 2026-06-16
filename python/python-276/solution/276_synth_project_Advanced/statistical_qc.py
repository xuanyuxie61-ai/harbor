"""
statistical_qc.py — Ensemble averaging, confidence intervals & statistical tests
================================================================================

The formation energy of a point defect at finite temperature is not a
single number but a *thermodynamic average* over an ensemble of defect
configurations weighted by the Boltzmann factor exp(−E_f / kT). In a real
simulation this would be done by Monte Carlo; for our small-scale
reproducible experiment we generate N_samples defect configurations with
perturbed chemical potentials μ and charge-state occupations, compute
E_f for each, and report:

  * the ensemble mean  <E_f>  and standard error σ / √N,
  * the 95 % confidence interval via Welch's t-distribution,
  * a hypothesis test comparing two defect kinds (vacancy vs interstitial)
    by Welch's t-test,
  * diagnostic distribution checks (skewness, kurtosis).

This module ports the statistical machinery of seed project 1100
(Exercise-Prescription-System, clinical_trial submodule) and the
distribution library of seed project 918 (prob) to the materials-science
context.

Seed project integration:
  * 1100_ling112211: Welch's t-test, Welch–Satterthwaite degrees of
    freedom, 95 % CI for the mean.
  * 918_prob: Boltzmann sampling uses a log-normal distribution for the
    defect volume and a Beta distribution for the charge-state occupation.
"""

from __future__ import annotations
import math
import numpy as np
from typing import List, Tuple, Dict


# -------------------------------------------------------------------------
# (1) Descriptive statistics
# -------------------------------------------------------------------------
def mean_ci_95(x: np.ndarray) -> Tuple[float, float]:
    """Mean and half-width of the 95 % confidence interval.

    For small samples we use the t-distribution with n − 1 d.o.f.:
        CI = mean ± t_{0.975, n-1} · σ / √n
    Port of `1100_ling112211/clinical_trial/mean_ci_95.m`.
    """
    n = x.size
    if n < 2:
        return float(x.mean()), float("inf")
    m = float(x.mean())
    s = float(x.std(ddof=1))
    # t critical value (two-sided 95%) — use scipy-free approximation
    df = n - 1
    t_crit = _t_critical_95(df)
    hw = t_crit * s / math.sqrt(n)
    return m, hw


def _t_critical_95(df: int) -> float:
    """Approximate two-sided 95 % t-critical value.

    Uses the Hill approximation (1970): for df ≥ 4,
        t_{0.975, df} ≈ z + (z³ + z) / (4 df) + ...
    with z = 1.96. For df < 4 we fall back to tabulated values.
    """
    if df <= 0:
        return float("inf")
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
             6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
    if df in table:
        return table[df]
    z = 1.9599639845400542   # 97.5-th percentile of N(0, 1)
    g1 = (z ** 3 + z) / (4.0 * df)
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96.0 * df * df)
    return z + g1 + g2


# -------------------------------------------------------------------------
# (2) Welch's t-test (for comparing two ensembles)
# -------------------------------------------------------------------------
def welch_ttest_pvalue(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Welch's t-test between two samples (possibly unequal variance).

    Returns (t-statistic, two-sided p-value).

    Port of `1100_ling112211/clinical_trial/welch_ttest_pvalue.m`.
    The Welch–Satterthwaite degrees of freedom are
        ν = (s_x²/n_x + s_y²/n_y)² / ((s_x²/n_x)²/(n_x-1) + (s_y²/n_y)²/(n_y-1))
    """
    nx, ny = x.size, y.size
    if nx < 2 or ny < 2:
        return 0.0, 1.0
    mx, my = float(x.mean()), float(y.mean())
    vx, vy = float(x.var(ddof=1)), float(y.var(ddof=1))
    se = math.sqrt(vx / nx + vy / ny)
    if se < 1e-15:
        return 0.0, 1.0
    t = (mx - my) / se
    # Welch–Satterthwaite d.o.f.
    num = (vx / nx + vy / ny) ** 2
    den = (vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1)
    df = num / den if den > 0 else 1.0
    # two-sided p-value from Student's t: use a simple rational approx
    p = _t_pvalue_two_sided(abs(t), df)
    return t, p


def _t_pvalue_two_sided(t: float, df: float) -> float:
    """Approximate two-sided p-value of |T| > t with T ~ t(df).

    Uses the normal approximation for large df and a rational approximation
    for small df.
    """
    if df > 100:
        # standard normal
        z = t
        p = math.erfc(z / math.sqrt(2.0))
        return p
    # Bailey's approximation:  p ≈ 2 · (1 + t²/df)^{-(df+1)/2} / (t sqrt(2π/df))
    if t < 1e-10:
        return 1.0
    logp = -(df + 1.0) / 2.0 * math.log(1.0 + t * t / df)
    logp -= 0.5 * math.log(2.0 * math.pi / df) + math.log(t)
    p = 2.0 * math.exp(logp)
    return min(1.0, max(0.0, p))


# -------------------------------------------------------------------------
# (3) Distribution samplers (from 918_prob)
# -------------------------------------------------------------------------
def lognormal_sample(mu: float, sigma: float, rng: np.random.Generator
                     ) -> float:
    """Draw a sample from the log-normal distribution.

    If X ~ N(μ, σ²), then Y = exp(X) ~ LogNormal(μ, σ²).
    Used to sample the defect volume Ω_def in the ensemble.
    """
    return float(math.exp(mu + sigma * rng.standard_normal()))


def beta_sample(a: float, b: float, rng: np.random.Generator) -> float:
    """Draw a sample from the Beta(a, b) distribution.

    Used to sample the charge-state occupation probability.
    """
    return float(rng.beta(a, b))


def boltzmann_weight(E: float, kT: float) -> float:
    """Boltzmann factor exp(-E / kT) with overflow protection."""
    arg = -E / kT
    if arg < -500:
        return 0.0
    return math.exp(arg)


# -------------------------------------------------------------------------
# (4) Ensemble generation and Boltzmann averaging
# -------------------------------------------------------------------------
def generate_ensemble(base_Ef_eV: float, n_samples: int,
                      kT_eV: float, mu_std_eV: float,
                      q_distribution: Tuple[float, float] = (2.0, 5.0),
                      seed: int = 276) -> np.ndarray:
    """Generate an ensemble of E_f values.

    For each sample:
        μ_sample  = μ + Δμ,   Δμ ~ N(0, μ_std)
        q_sample  ~ Beta(a, b)  rounded to {-1, 0, +1}
        E_f_sample = base_Ef + Δμ + q · ΔV
    where ΔV = 0.1 eV is a fixed potential alignment.
    """
    rng = np.random.default_rng(seed)
    Efs: List[float] = []
    for _ in range(n_samples):
        dmu = mu_std_eV * rng.standard_normal()
        q_prob = beta_sample(q_distribution[0], q_distribution[1], rng)
        q_sample = 1 if q_prob > 0.7 else (-1 if q_prob < 0.3 else 0)
        Ef = base_Ef_eV + dmu + q_sample * 0.1
        Efs.append(Ef)
    return np.asarray(Efs, dtype=np.float64)


def boltzmann_average(Efs: np.ndarray, kT_eV: float) -> Tuple[float, float]:
    """Compute the Boltzmann-weighted average of E_f.

        <E_f> = Σ E_f exp(-E_f / kT) / Σ exp(-E_f / kT)
    """
    weights = np.exp(-Efs / kT_eV)
    s = float(np.sum(weights))
    if s < 1e-300:
        return float(Efs.min()), 0.0
    mean = float(np.sum(Efs * weights)) / s
    var = float(np.sum(weights * (Efs - mean) ** 2)) / s
    return mean, math.sqrt(var)


# -------------------------------------------------------------------------
# (5) Full statistical workflow
# -------------------------------------------------------------------------
def statistical_summary(base_Ef_vac_eV: float, base_Ef_int_eV: float,
                        n_samples: int, kT_eV: float,
                        seed: int = 276) -> Dict[str, float]:
    """Compute ensemble statistics for vacancy and interstitial, then run
    Welch's t-test between them.

    Returns a dict with:
      - mean, std, CI_hw for each defect kind
      - t-statistic and p-value for the comparison
    """
    Efs_vac = generate_ensemble(base_Ef_vac_eV, n_samples, kT_eV,
                                mu_std_eV=0.05, seed=seed)
    Efs_int = generate_ensemble(base_Ef_int_eV, n_samples, kT_eV,
                                mu_std_eV=0.05, seed=seed + 1)

    m_vac, hw_vac = mean_ci_95(Efs_vac)
    m_int, hw_int = mean_ci_95(Efs_int)
    t_stat, p_val = welch_ttest_pvalue(Efs_vac, Efs_int)

    # Boltzmann averages at kT
    mean_boltz_vac, _ = boltzmann_average(Efs_vac, kT_eV)
    mean_boltz_int, _ = boltzmann_average(Efs_int, kT_eV)

    return {
        "vac_mean_eV": m_vac, "vac_CI_hw_eV": hw_vac,
        "int_mean_eV": m_int, "int_CI_hw_eV": hw_int,
        "vac_boltz_eV": mean_boltz_vac, "int_boltz_eV": mean_boltz_int,
        "welch_t": t_stat, "welch_p": p_val,
        "n_samples": n_samples, "kT_eV": kT_eV,
    }
