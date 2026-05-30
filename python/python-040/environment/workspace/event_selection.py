#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from typing import Tuple, Dict, List, Optional


def reconstruct_invariant_mass(
    pt1: float,
    eta1: float,
    phi1: float,
    pt2: float,
    eta2: float,
    phi2: float,
    mass1: float = 0.000511,
    mass2: float = 0.000511
) -> float:
    px1 = pt1 * np.cos(phi1)
    py1 = pt1 * np.sin(phi1)
    pz1 = pt1 * np.sinh(eta1)
    e1 = np.sqrt(pt1 ** 2 * np.cosh(eta1) ** 2 + mass1 ** 2)

    px2 = pt2 * np.cos(phi2)
    py2 = pt2 * np.sin(phi2)
    pz2 = pt2 * np.sinh(eta2)
    e2 = np.sqrt(pt2 ** 2 * np.cosh(eta2) ** 2 + mass2 ** 2)

    m2 = (e1 + e2) ** 2 - (px1 + px2) ** 2 - (py1 + py2) ** 2 - (pz1 + pz2) ** 2
    m2 = max(m2, 0.0)
    return np.sqrt(m2)


def generate_drell_yan_background(
    n_events: int,
    mass_range: Tuple[float, float],
    seed: Optional[int] = None
) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)

    m_min, m_max = mass_range
    u = np.random.uniform(0.0, 1.0, n_events)


    a = m_min ** (-2)
    b = m_max ** (-2)
    masses = np.sqrt(1.0 / (b + u * (a - b)))


    sigma_rel = 0.02
    noise = np.random.normal(0.0, sigma_rel * masses)
    masses += noise
    masses = np.maximum(masses, m_min)

    return masses


def generate_signal_events(
    n_events: int,
    zp_mass: float,
    zp_width: float,
    mass_range: Tuple[float, float],
    seed: Optional[int] = None
) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)

    m_min, m_max = mass_range
    masses = []
    max_attempts = n_events * 1000
    attempts = 0


    m_peak = zp_mass
    peak_height = 1.0 / ((m_peak ** 2 - zp_mass ** 2) ** 2 + zp_mass ** 2 * zp_width ** 2)
    if peak_height < 1e-30:
        peak_height = 1e-30

    while len(masses) < n_events and attempts < max_attempts:
        m = np.random.uniform(m_min, m_max)
        bw = 1.0 / ((m ** 2 - zp_mass ** 2) ** 2 + zp_mass ** 2 * zp_width ** 2)
        u = np.random.uniform(0.0, peak_height)
        if u <= bw:

            sigma_m = 0.015 * m
            m_smeared = m + np.random.normal(0.0, sigma_m)
            if m_min <= m_smeared <= m_max:
                masses.append(m_smeared)
        attempts += 1

    return np.array(masses)


def cl_s_limit(
    n_observed: int,
    n_background: float,
    n_signal_hypothesis: float,
    confidence_level: float = 0.95
) -> bool:
    from math import exp, factorial

    def poisson_cdf(k: int, mu: float) -> float:
        if mu <= 0.0:
            return 1.0 if k >= 0 else 0.0
        cdf = 0.0
        for n in range(k + 1):

            log_p = n * np.log(mu) - mu
            for i in range(1, n + 1):
                log_p -= np.log(i)
            cdf += np.exp(log_p)
        return min(cdf, 1.0)

    cl_b = poisson_cdf(n_observed, n_background)
    cl_sb = poisson_cdf(n_observed, n_background + n_signal_hypothesis)

    if cl_b < 1e-10:
        cl_s = 0.0
    else:
        cl_s = cl_sb / cl_b

    return cl_s < (1.0 - confidence_level)


def run_full_analysis(
    zp_model_params: Dict,
    luminosity_fb: float = 3000.0,
    n_bins: int = 50
) -> Dict:
    m_zp = zp_model_params['mass']
    gamma_zp = zp_model_params['total_width']
    gq = zp_model_params.get('gq_coupling', 0.1)


    mass_min = max(m_zp - 15.0 * gamma_zp, 50.0)
    mass_max = m_zp + 15.0 * gamma_zp
    mass_range = (mass_min, mass_max)







    sqrt_s = 13000.0
    s = sqrt_s ** 2
    sigma_zp = None
    br_ee = None


    n_sig_expected = None



    sigma_dy = 10.0 * (1000.0 / m_zp) ** 3
    n_bkg_expected = sigma_dy * luminosity_fb * 1000.0 * 0.5 * br_ee


    n_sig = int(np.round(n_sig_expected))
    n_bkg = int(np.round(n_bkg_expected))

    sig_masses = generate_signal_events(max(n_sig, 10), m_zp, gamma_zp, mass_range, seed=42)
    bkg_masses = generate_drell_yan_background(max(n_bkg, 100), mass_range, seed=43)


    bins = np.linspace(mass_min, mass_max, n_bins + 1)
    sig_counts, _ = np.histogram(sig_masses, bins=bins)
    bkg_counts, _ = np.histogram(bkg_masses, bins=bins)
    obs_counts = sig_counts + bkg_counts

    bin_centers = (bins[:-1] + bins[1:]) / 2.0


    try:
        from signal_processing import resonance_peak_finder
    except ImportError:
        from .signal_processing import resonance_peak_finder
    peak_mass, peak_height, peak_sig = resonance_peak_finder(
        bin_centers, obs_counts, window_width=3.0 * gamma_zp
    )


    n_obs_total = int(np.sum(obs_counts))
    n_bkg_total = max(float(np.sum(bkg_counts)), 1.0)
    sigma_95 = 1.96 * np.sqrt(n_bkg_total) / (luminosity_fb * 1000.0 * 0.5 * br_ee)


    try:
        from parameter_scan import discovery_potential
    except ImportError:
        from .parameter_scan import discovery_potential
    z_values = discovery_potential(
        np.array([sigma_zp]),
        np.array([sigma_dy]),
        np.array([luminosity_fb]),
        np.array([0.05])
    )
    discovery_z = z_values[0]

    return {
        'zp_mass': m_zp,
        'zp_width': gamma_zp,
        'signal_yield_expected': n_sig_expected,
        'background_yield_expected': n_bkg_expected,
        'peak_mass': peak_mass,
        'peak_height': peak_height,
        'peak_significance': peak_sig,
        'exclusion_limit_sigma_95_pb': sigma_95,
        'discovery_potential_z': discovery_z,
        'bins': bin_centers,
        'signal_histogram': sig_counts,
        'background_histogram': bkg_counts,
        'observed_histogram': obs_counts,
    }


def format_physics_summary(results: Dict) -> str:
    lines = [
        "=" * 70,
        "  LHC BSM Z' → ℓ⁺ℓ⁻ 信号分析结果汇总",
        "=" * 70,
        f"  Z' 质量:           {results['zp_mass']:.2f} GeV",
        f"  Z' 总宽度:         {results['zp_width']:.4f} GeV",
        f"  预期信号产额:      {results['signal_yield_expected']:.2f}",
        f"  预期背景产额:      {results['background_yield_expected']:.2f}",
        "-" * 70,
        "  共振峰搜索结果:",
        f"    峰位质量:        {results['peak_mass']:.2f} GeV",
        f"    峰高度:          {results['peak_height']:.2f}",
        f"    局部显著性:      {results['peak_significance']:.3f} σ",
        "-" * 70,
        "  统计推断:",
        f"    95% CL 截面上限: {results['exclusion_limit_sigma_95_pb']:.6f} pb",
        f"    发现潜力 (Z):    {results['discovery_potential_z']:.3f} σ",
        "=" * 70,
    ]
    return "\n".join(lines)
