
import numpy as np
from typing import Tuple


def neo_hookean_strain_energy(F: np.ndarray, mu: float, K_bulk: float) -> float:
    if F.shape != (3, 3):
        raise ValueError("F must be 3x3")

    C = F.T @ F
    I1 = np.trace(C)
    J = np.linalg.det(F)
    if abs(J) < 1e-14:
        J = 1e-14

    lnJ = np.log(abs(J))
    W = 0.5 * mu * (I1 - 3.0) - mu * lnJ + 0.5 * K_bulk * lnJ ** 2
    return W


def neo_hookean_stress(F: np.ndarray, mu: float, K_bulk: float) -> np.ndarray:
    if F.shape != (3, 3):
        raise ValueError("F must be 3x3")

    J = np.linalg.det(F)
    if abs(J) < 1e-14:
        J = np.sign(J + 1e-14) * 1e-14

    F_inv_T = np.linalg.inv(F).T
    P = mu * (F - F_inv_T) + K_bulk * np.log(abs(J)) * F_inv_T
    return P


def mooney_rivlin_strain_energy(F: np.ndarray, C10: float, C01: float, K_bulk: float) -> float:
    if F.shape != (3, 3):
        raise ValueError("F must be 3x3")

    C = F.T @ F
    I1 = np.trace(C)
    I2 = 0.5 * (I1 ** 2 - np.trace(C @ C))
    J = np.linalg.det(F)
    if abs(J) < 1e-14:
        J = 1e-14

    lnJ = np.log(abs(J))
    W = C10 * (I1 - 3.0) + C01 * (I2 - 3.0) + 0.5 * K_bulk * lnJ ** 2
    return W


def mooney_rivlin_stress(F: np.ndarray, C10: float, C01: float, K_bulk: float) -> np.ndarray:
    if F.shape != (3, 3):
        raise ValueError("F must be 3x3")

    J = np.linalg.det(F)
    if abs(J) < 1e-14:
        J = np.sign(J + 1e-14) * 1e-14

    B = F @ F.T
    I1 = np.trace(B)

    sigma = (2.0 / J) * (C10 + C01 * I1) * B - (2.0 * C01 / J) * (B @ B)
    sigma += (K_bulk * np.log(abs(J)) / J) * np.eye(3)
    return sigma


def soft_robot_1d_constitutive(epsilon: float, kappa: np.ndarray,
                               E: float, G: float, A: float,
                               Ixx: float, Iyy: float, J: float) -> Tuple[np.ndarray, np.ndarray]:

    N_axial = E * A * epsilon

    V1 = G * A * kappa[0] * 0.0
    V2 = G * A * kappa[1] * 0.0


    M1 = E * Ixx * kappa[0]
    M2 = E * Iyy * kappa[1]
    M3 = G * J * kappa[2]

    n = np.array([V1, V2, N_axial])
    m = np.array([M1, M2, M3])
    return n, m


def chemo_mechanical_coupling(y_chem: np.ndarray, epsilon: float,
                              E0: float, gamma: float, beta_chem: float) -> float:
    c_ion = y_chem[0]
    pH = y_chem[1]


    pH_factor = np.exp(-beta_chem * abs(pH - 7.0))

    ion_factor = 1.0 + gamma * c_ion

    E_eff = E0 * ion_factor * pH_factor

    E_eff = max(E0 * 0.1, min(E0 * 5.0, E_eff))
    return E_eff


def selkov_glycolysis_ode(t: float, y: np.ndarray, a: float = 0.08, b: float = 0.6) -> np.ndarray:
    u, v = y
    dudt = -u + a * v + u * u * v
    dvdt = b - a * v - u * u * v
    return np.array([dudt, dvdt])


def tangent_stiffness_neo_hookean(F: np.ndarray, mu: float, K_bulk: float) -> np.ndarray:
    if F.shape != (3, 3):
        raise ValueError("F must be 3x3")

    J = np.linalg.det(F)
    if abs(J) < 1e-14:
        J = 1e-14

    F_inv = np.linalg.inv(F)
    lnJ = np.log(abs(J))


    C = np.zeros((6, 6))

    for i in range(3):
        C[i, i] = mu + K_bulk
    for i in range(3, 6):
        C[i, i] = 0.5 * mu


    vol_term = K_bulk * lnJ - mu
    for i in range(3):
        for j in range(3):
            C[i, j] += vol_term

    return C
