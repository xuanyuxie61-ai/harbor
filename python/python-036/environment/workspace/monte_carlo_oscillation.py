
import numpy as np
from constants import (
    THETA_12, THETA_23, THETA_13, DELTA_CP,
    DELTA_M2_21, DELTA_M2_31, DELTA_M2_31_IH
)
from pmns_matrix import build_pmns_matrix
from neutrino_hamiltonian import build_vacuum_hamiltonian, solve_hamiltonian_eigen


def sample_unit_square_2d(n_samples, seed=None):
    rng = np.random.default_rng(seed)
    return rng.random((n_samples, 2))


def sample_unit_cube_3d(n_samples, seed=None):
    rng = np.random.default_rng(seed)
    return rng.random((n_samples, 3))


def monomial_value(m, e, x):
    value = 1.0
    for i in range(m):
        if e[i] < 0:
            raise ValueError("Exponents must be non-negative")
        if e[i] == 0:
            continue
        value *= x[i] ** e[i]
    return value


def cube_monomial_integral(e):
    m = len(e)
    integral = 1.0
    for i in range(m):
        if e[i] < 0:
            raise ValueError("Exponents must be non-negative")
        integral /= (e[i] + 1)
    return integral


def monte_carlo_oscillation_probability(
        energy_range_gev, baseline_range_km,
        n_samples=50000, hierarchy='normal',
        param_uncertainties=None, seed=None
):
    rng = np.random.default_rng(seed)
    E_min, E_max = energy_range_gev
    L_min, L_max = baseline_range_km

    if E_min <= 0 or L_min < 0:
        raise ValueError("Energy and baseline ranges must be positive")


    E_samples = rng.uniform(E_min, E_max, n_samples)
    L_samples = rng.uniform(L_min, L_max, n_samples)


    if param_uncertainties is None:
        param_uncertainties = {}

    t12_samples = rng.normal(
        THETA_12, param_uncertainties.get('theta12', 0.0), n_samples
    )
    t23_samples = rng.normal(
        THETA_23, param_uncertainties.get('theta23', 0.0), n_samples
    )
    t13_samples = rng.normal(
        THETA_13, param_uncertainties.get('theta13', 0.0), n_samples
    )
    dcp_samples = rng.normal(
        DELTA_CP, param_uncertainties.get('delta_cp', 0.0), n_samples
    )


    t12_samples = np.clip(t12_samples, 0.01, np.pi / 2 - 0.01)
    t23_samples = np.clip(t23_samples, 0.01, np.pi / 2 - 0.01)
    t13_samples = np.clip(t13_samples, 0.01, np.pi / 2 - 0.01)

    dm31 = DELTA_M2_31 if hierarchy == 'normal' else DELTA_M2_31_IH

    P_ee = np.zeros(n_samples, dtype=np.float64)
    P_em = np.zeros(n_samples, dtype=np.float64)
    P_et = np.zeros(n_samples, dtype=np.float64)

    for i in range(n_samples):
        E = E_samples[i]
        L = L_samples[i]

        U = build_pmns_matrix(
            t12_samples[i], t23_samples[i], t13_samples[i], dcp_samples[i]
        )
        M2 = np.diag([0.0, DELTA_M2_21, dm31])


        H = (1.0 / (2.0 * E * 1e9)) * (U @ M2 @ U.conj().T)



        L_ev_inv = L * 5.067730889e9


        eigenvalues, eigenvectors = np.linalg.eigh(H)
        D = np.diag(np.exp(-1j * eigenvalues * L_ev_inv))
        U_prop = eigenvectors @ D @ eigenvectors.conj().T


        psi0 = np.array([1.0, 0.0, 0.0], dtype=np.complex128)
        psi_L = U_prop @ psi0

        P_ee[i] = abs(psi_L[0]) ** 2
        P_em[i] = abs(psi_L[1]) ** 2
        P_et[i] = abs(psi_L[2]) ** 2

    return {
        'P_ee_mean': float(np.mean(P_ee)),
        'P_ee_std': float(np.std(P_ee)),
        'P_em_mean': float(np.mean(P_em)),
        'P_em_std': float(np.std(P_em)),
        'P_et_mean': float(np.mean(P_et)),
        'P_et_std': float(np.std(P_et)),
        'E_samples': E_samples,
        'L_samples': L_samples,
        'P_ee': P_ee,
        'P_em': P_em,
        'P_et': P_et
    }


def mc_hierarchy_significance(
        energy_gev, baseline_km,
        n_samples=20000, sigma_dm31=0.03e-3, seed=None
):
    rng = np.random.default_rng(seed)


    dm31_nh = rng.normal(DELTA_M2_31, sigma_dm31, n_samples)
    dm31_ih = rng.normal(DELTA_M2_31_IH, sigma_dm31, n_samples)


    nh_correct = np.sum(dm31_nh > 0) / n_samples
    ih_correct = np.sum(dm31_ih < 0) / n_samples





    return {
        'nh_correct_rate': float(nh_correct),
        'ih_correct_rate': float(ih_correct),
        'nh_confidence_sigma': float(
            np.sqrt(2.0) * abs(DELTA_M2_31) / sigma_dm31
        ),
        'ih_confidence_sigma': float(
            np.sqrt(2.0) * abs(DELTA_M2_31_IH) / sigma_dm31
        )
    }


def mc_integrate_oscillation_over_spectrum(
        energy_spectrum, weights, baseline_km,
        n_samples_per_bin=1000, hierarchy='normal', seed=None
):
    rng = np.random.default_rng(seed)
    n_bins = len(energy_spectrum)

    if len(weights) != n_bins:
        raise ValueError("energy_spectrum and weights must have same length")
    if np.sum(weights) <= 0:
        raise ValueError("weights must sum to positive value")

    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / np.sum(weights)

    dm31 = DELTA_M2_31 if hierarchy == 'normal' else DELTA_M2_31_IH
    U = build_pmns_matrix()
    M2 = np.diag([0.0, DELTA_M2_21, dm31])

    total_P_ee = 0.0
    total_P_em = 0.0
    total_P_et = 0.0

    for b in range(n_bins):
        E0 = energy_spectrum[b]
        if E0 <= 0:
            continue
        w = weights[b]


        dE = 0.05 * E0
        E_samples = rng.uniform(E0 - dE, E0 + dE, n_samples_per_bin)
        E_samples = np.clip(E_samples, 0.001, None)

        for E in E_samples:
            H = (1.0 / (2.0 * E * 1e9)) * (U @ M2 @ U.conj().T)
            L_ev_inv = baseline_km * 5.067730889e9

            eigenvalues, eigenvectors = np.linalg.eigh(H)
            D = np.diag(np.exp(-1j * eigenvalues * L_ev_inv))
            U_prop = eigenvectors @ D @ eigenvectors.conj().T

            psi0 = np.array([1.0, 0.0, 0.0], dtype=np.complex128)
            psi_L = U_prop @ psi0

            total_P_ee += w * abs(psi_L[0]) ** 2 / n_samples_per_bin
            total_P_em += w * abs(psi_L[1]) ** 2 / n_samples_per_bin
            total_P_et += w * abs(psi_L[2]) ** 2 / n_samples_per_bin

    return {
        'P_ee_avg': float(total_P_ee),
        'P_em_avg': float(total_P_em),
        'P_et_avg': float(total_P_et),
        'sum_prob': float(total_P_ee + total_P_em + total_P_et)
    }
