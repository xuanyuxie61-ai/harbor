
import numpy as np
from typing import Optional, Tuple


def b3_compliance_function(
    t: float, t_prime: float,
    q1: float, q2: float, q3: float, q4: float,
    lambda_: float = 1.0
) -> float:
    if t <= t_prime:
        return q1

    dt = t - t_prime
    tau = t_prime



    Q = 1.0 - np.exp(-lambda_ * np.sqrt(dt / (tau + 1.0)))


    log_term = np.log1p(dt ** 0.3)


    flow_term = np.log(t / tau) if t / tau > 1.0 else 0.0

    J = q1 + q2 * Q + q3 * log_term + q4 * flow_term
    return max(J, q1)


def b3_creep_coefficient(
    t: float, t_prime: float, E28: float,
    q1: float, q2: float, q3: float, q4: float
) -> float:

    E_tprime = E28 * np.sqrt(t_prime / 28.0) if t_prime < 28.0 else E28
    J = b3_compliance_function(t, t_prime, q1, q2, q3, q4)
    phi = E_tprime * J - 1.0
    return max(phi, 0.0)


def mc2010_creep_coefficient(
    t: float, t0: float,
    fcm: float, RH: float, h0: float,
    cement_type: str = "N"
) -> float:
    if t <= t0:
        return 0.0

    fcm0 = 10.0


    phi_RH = 1.0 + (1.0 - RH / 100.0) / (0.1 * (h0 ** (1.0 / 3.0)))


    beta_fcm = 5.3 / np.sqrt(fcm / fcm0)


    beta_t0 = 1.0 / (0.1 + t0 ** 0.20)


    phi_0 = phi_RH * beta_fcm * beta_t0


    alpha_factor = {"N": 1.0, "R": 1.25, "SL": 0.85}.get(cement_type, 1.0)


    beta_H = min(1.5 * h0 + 250.0 * alpha_factor, 1500.0 * alpha_factor)


    dt = t - t0
    beta_c = (dt / (beta_H + dt)) ** 0.3

    phi = phi_0 * beta_c
    return max(phi, 0.0)


def mc2010_shrinkage_strain(
    t: float, ts: float,
    fcm: float, RH: float, h0: float,
    cement_type: str = "N"
) -> float:
    if t <= ts:
        return 0.0

    alpha_as = {"N": 800e-6, "R": 700e-6, "SL": 900e-6}.get(cement_type, 800e-6)


    eps_cas = alpha_as * ((0.1 * fcm) / (6.0 + 0.1 * fcm)) ** 2.5


    beta_RH = 1.0 - (RH / 100.0) ** 3


    dt = t - ts
    beta_s = np.sqrt(dt / (350.0 * (h0 / 100.0) ** 2 + dt))

    eps_cs = eps_cas * beta_RH * beta_s
    return eps_cs


def aging_elastic_modulus(
    t: float, E28: float, s: float = 0.25
) -> float:
    if t <= 0:
        t = 0.01
    ratio = t / (4.0 + 0.85 * t)
    return E28 * (ratio ** s)


def kelvin_chain_compliance(
    t: float, t_prime: float,
    E0: float, E_k: np.ndarray, eta_k: np.ndarray
) -> float:
    if t <= t_prime:
        return 1.0 / E0

    dt = t - t_prime
    J = 1.0 / E0
    for Ek, etak in zip(E_k, eta_k):
        if Ek > 0 and etak > 0:
            tau_k = etak / Ek
            J += (1.0 / Ek) * (1.0 - np.exp(-dt / tau_k))
    return J


def maxwell_chain_relaxation(
    t: float, t_prime: float,
    E0: float, E_i: np.ndarray, eta_i: np.ndarray
) -> float:
    if t <= t_prime:

        return E0 + np.sum(E_i)

    dt = t - t_prime
    R = E0
    for Ei, etai in zip(E_i, eta_i):
        if Ei > 0 and etai > 0:
            tau_i = etai / Ei
            R += Ei * np.exp(-dt / tau_i)
    return R


def complex_modulus_maxwell(
    omega: np.ndarray, E0: float, E_i: np.ndarray, eta_i: np.ndarray
) -> np.ndarray:
    omega = np.asarray(omega)
    E_star = np.full_like(omega, E0, dtype=complex)

    for Ei, etai in zip(E_i, eta_i):
        if Ei > 0 and etai > 0:
            tau_i = etai / Ei
            iw_tau = 1j * omega * tau_i
            E_star += Ei * iw_tau / (1.0 + iw_tau)

    return E_star


def degree_of_hydration(
    t: float, T: float, alpha_inf: float = 0.85,
    tau_h: float = 24.0, beta_h: float = 0.7
) -> float:
    T_ref = 293.15
    Ea_R = 4000.0


    t_eq = t * np.exp(Ea_R * (1.0 / T_ref - 1.0 / T))

    alpha = alpha_inf * np.exp(-(tau_h / t_eq) ** beta_h)
    return min(alpha, alpha_inf)


def mature_compressive_strength(
    t: float, T: float, fcm28: float
) -> float:
    alpha = degree_of_hydration(t, T)
    alpha28 = degree_of_hydration(28.0, T)
    if alpha28 < 1e-10:
        return fcm28
    n = 1.0
    return fcm28 * (alpha / alpha28) ** n


def equivalent_age_linear(
    t: float, T_history: np.ndarray, dt: float,
    Ea: float = 33500.0, R_gas: float = 8.314
) -> float:
    T_ref = 293.15
    t_eq = 0.0
    for T in T_history:
        beta_T = np.exp((Ea / R_gas) * (1.0 / T_ref - 1.0 / T))
        t_eq += dt * beta_T
    return t_eq


def stress_strain_creep_integral(
    time_points: np.ndarray, strain_history: np.ndarray,
    E28: float, phi_func
) -> np.ndarray:
    n = len(time_points)
    stress = np.zeros(n)

    for k in range(n):
        sigma_k = 0.0
        for i in range(k + 1):
            if i == 0:
                d_eps = strain_history[i]
            else:
                d_eps = strain_history[i] - strain_history[i - 1]

            t_k = time_points[k]
            t_i = time_points[i]
            phi = phi_func(t_k, t_i)
            E_ti = aging_elastic_modulus(t_i, E28)
            R = E_ti / (1.0 + phi) if phi >= 0 else E_ti
            sigma_k += R * d_eps

        stress[k] = sigma_k

    return stress


def effective_creep_modulus(
    t: float, t0: float, E28: float, phi: float
) -> float:




    pass
