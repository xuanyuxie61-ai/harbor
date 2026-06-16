"""
Gauge orbit analysis and configuration distance metrics.

Adapted from:
  - 192_closest_point_brute (brute-force closest-point search)
  - 866_permutation_distance (permutation-based distance metrics)

Key formulas:
    Landau gauge functional:
        F_U[Omega] = sum_{x,mu} Re Tr Omega(x) U_mu(x) Omega^dag(x+mu)
    Ulam distance: d_Ulam(pi, sigma) = N - LIS(pi^{-1} sigma)
"""

import numpy as np
from constants import LatticeParams
from lattice_geometry import LatticeGeometry, SiteIndex
from gauge_field import GaugeField, _project_to_su2


def gauge_field_frobenius_distance(gf1: GaugeField, gf2: GaugeField) -> float:
    diff = gf1._U - gf2._U
    return float(np.sqrt(np.sum(np.abs(diff) ** 2)))


def find_closest_gauge_transform_brute(gf: GaugeField, target_gf: GaugeField,
                                         n_random: int = 50, seed: int = 0) -> dict:
    if n_random < 1:
        raise ValueError(f"n_random must be >= 1, got {n_random}")
    rng = np.random.default_rng(seed)
    best_dist = float('inf')
    for _ in range(n_random):
        omega = _random_gauge_transform(gf.p, rng)
        gf_trial = gf.copy()
        gf_trial.gauge_transform(omega)
        d = gauge_field_frobenius_distance(gf_trial, target_gf)
        if d < best_dist:
            best_dist = d
    return {'min_distance': best_dist}


def _random_gauge_transform(params, rng):
    from constants import su2_random
    shape = params.shape
    omega = np.empty(shape + (2, 2), dtype=complex)
    for idx in np.ndindex(*shape):
        omega[idx] = su2_random(rng)
    return omega


def landau_gauge_fix(gf: GaugeField, n_iter: int = 100,
                      omega_relax: float = 1.7) -> dict:
    if n_iter < 1:
        raise ValueError(f"n_iter must be >= 1, got {n_iter}")
    if not (1.0 < omega_relax < 2.0):
        raise ValueError(f"omega_relax must be in (1, 2), got {omega_relax}")
    gf_gf = gf.copy()
    geo = gf_gf.geo
    functional_history = []
    theta_history = []
    for it in range(n_iter):
        max_theta = 0.0
        for s in geo.all_sites():
            Delta = np.zeros((2, 2), dtype=complex)
            for mu in range(4):
                U_mu = gf_gf.link(s, mu)
                s_m = geo.neighbor(s, mu, False)
                U_mu_m_dag = gf_gf.link(s_m, mu).conj().T
                Delta += (U_mu - U_mu_m_dag)
            Delta_anti = 0.5 * (Delta - Delta.conj().T)
            tr = np.trace(Delta_anti)
            Delta_anti -= 0.5 * tr * np.eye(2, dtype=complex)
            theta = float(np.sqrt(np.sum(np.abs(Delta_anti) ** 2)))
            max_theta = max(max_theta, theta)
            if theta > 1e-14:
                arg = -1j * omega_relax * Delta_anti
                Om = _su2_exp_approx(arg)
                for mu2 in range(4):
                    U = gf_gf.link(s, mu2)
                    gf_gf.set_link(s, mu2, _project_to_su2(Om @ U))
                for mu2 in range(4):
                    s_m = geo.neighbor(s, mu2, False)
                    U_m = gf_gf.link(s_m, mu2)
                    gf_gf.set_link(s_m, mu2, _project_to_su2(U_m @ Om.conj().T))
        F = _landau_functional(gf_gf)
        functional_history.append(F)
        theta_history.append(max_theta)
        if max_theta < 1e-8:
            return {
                'gauge_field': gf_gf, 'functional': F, 'theta': max_theta,
                'converged': True, 'n_iterations': it + 1,
            }
    return {
        'gauge_field': gf_gf,
        'functional': functional_history[-1] if functional_history else 0.0,
        'theta': theta_history[-1] if theta_history else float('inf'),
        'converged': False, 'n_iterations': n_iter,
    }


def _landau_functional(gf):
    total = 0.0
    for s in gf.geo.all_sites():
        for mu in range(4):
            U = gf.link(s, mu)
            total += (U[0, 0] + U[1, 1]).real
    return total


def _su2_exp_approx(A):
    I = np.eye(2, dtype=complex)
    exp_A = I + A + 0.5 * A @ A
    return _project_to_su2(exp_A)


def permutation_distance_between_configs(gf1, gf2) -> dict:
    plaq1 = _plaquette_trace_vector(gf1)
    plaq2 = _plaquette_trace_vector(gf2)
    if len(plaq1) != len(plaq2):
        raise ValueError("Configurations have different numbers of plaquettes")
    rank1 = _argsort_to_permutation(plaq1)
    rank2 = _argsort_to_permutation(plaq2)
    lis_len = _longest_increasing_subsequence_length(rank1, rank2)
    ulam = len(rank1) - lis_len
    kt = _kendall_tau_distance(rank1, rank2)
    n = len(rank1)
    d_sq = np.sum((rank1.astype(float) - rank2.astype(float)) ** 2)
    spearman = 1.0 - 6.0 * d_sq / (n * (n ** 2 - 1) + 1e-30)
    return {
        'ulam_distance': int(ulam),
        'kendall_tau': int(kt),
        'spearman_rho': float(spearman),
        'n_plaquettes': n,
    }


def _plaquette_trace_vector(gf):
    values = []
    geo = gf.geo
    for s in geo.all_sites():
        for mu in range(4):
            for nu in range(mu + 1, 4):
                s_mu = geo.neighbor(s, mu, True)
                s_nu = geo.neighbor(s, nu, True)
                U_p = (gf.link(s, mu) @ gf.link(s_mu, nu)
                       @ gf.link(s_nu, mu).conj().T @ gf.link(s, nu).conj().T)
                tr = 0.5 * (U_p[0, 0].real + U_p[1, 1].real)
                values.append(tr)
    return np.array(values)


def _argsort_to_permutation(arr):
    return np.argsort(np.argsort(arr))


def _longest_increasing_subsequence_length(pi, sigma):
    n = len(pi)
    pi_inv = np.argsort(pi)
    composed = pi_inv[sigma]
    import bisect
    tails = []
    for x in composed:
        pos = bisect.bisect_left(tails, x)
        if pos == len(tails):
            tails.append(x)
        else:
            tails[pos] = x
    return len(tails)


def _kendall_tau_distance(pi, sigma):
    n = len(pi)
    pi_inv = np.argsort(pi)
    composed = pi_inv[sigma]
    inversions = 0
    for i in range(n):
        for j in range(i + 1, n):
            if composed[i] > composed[j]:
                inversions += 1
    return inversions
