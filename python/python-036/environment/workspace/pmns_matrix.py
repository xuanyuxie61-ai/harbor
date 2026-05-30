
import numpy as np
from constants import (
    THETA_12, THETA_23, THETA_13, DELTA_CP,
    DELTA_M2_21, DELTA_M2_31, DELTA_M2_31_IH
)


def rotation_12(theta12):
    c = np.cos(theta12)
    s = np.sin(theta12)
    R = np.array([
        [ c,  s,  0.0],
        [-s,  c,  0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.complex128)
    return R


def rotation_13(theta13, delta=0.0):
    c = np.cos(theta13)
    s = np.sin(theta13)
    exp_delta = np.exp(1j * delta)
    R = np.array([
        [ c,           0.0,  s / exp_delta],
        [ 0.0,         1.0,  0.0          ],
        [-s * exp_delta, 0.0,  c           ]
    ], dtype=np.complex128)
    return R


def rotation_23(theta23):
    c = np.cos(theta23)
    s = np.sin(theta23)
    R = np.array([
        [1.0, 0.0, 0.0],
        [0.0,  c,  s ],
        [0.0, -s,  c ]
    ], dtype=np.complex128)
    return R


def build_pmns_matrix(theta12=None, theta23=None, theta13=None, delta_cp=None):
    t12 = THETA_12 if theta12 is None else float(theta12)
    t23 = THETA_23 if theta23 is None else float(theta23)
    t13 = THETA_13 if theta13 is None else float(theta13)
    dcp = DELTA_CP if delta_cp is None else float(delta_cp)


    eps = 1e-12
    if t12 < eps or t12 > np.pi / 2 - eps:
        raise ValueError(f"theta12 must be in (0, π/2), got {t12}")
    if t23 < eps or t23 > np.pi / 2 - eps:
        raise ValueError(f"theta23 must be in (0, π/2), got {t23}")
    if t13 < eps or t13 > np.pi / 2 - eps:
        raise ValueError(f"theta13 must be in (0, π/2), got {t13}")






    raise NotImplementedError("HOLE 1: build_pmns_matrix 核心构造尚未实现")


def build_mass_matrix(delta_m2_21=None, delta_m2_31=None, hierarchy='normal'):
    dm21 = DELTA_M2_21 if delta_m2_21 is None else float(delta_m2_21)

    if delta_m2_31 is None:
        if hierarchy.lower() == 'normal':
            dm31 = DELTA_M2_31
        elif hierarchy.lower() == 'inverted':
            dm31 = DELTA_M2_31_IH
        else:
            raise ValueError("hierarchy must be 'normal' or 'inverted'")
    else:
        dm31 = float(delta_m2_31)


    if dm21 <= 0:
        raise ValueError(f"delta_m2_21 must be positive, got {dm21}")

    M2 = np.diag([0.0, dm21, dm31])
    return M2


def check_unitarity(U, tol=1e-10):
    identity = np.eye(3, dtype=np.complex128)
    udag = U.conj().T

    err1 = np.max(np.abs(U @ udag - identity))
    err2 = np.max(np.abs(udag @ U - identity))

    row_sums = np.sum(np.abs(U) ** 2, axis=1)
    col_sums = np.sum(np.abs(U) ** 2, axis=0)
    err3 = np.max(np.abs(row_sums - 1.0))
    err4 = np.max(np.abs(col_sums - 1.0))

    max_error = max(err1, err2, err3, err4)
    return max_error < tol, max_error


def pmns_to_mass_basis(U, flavor_state):
    flavor_state = np.asarray(flavor_state, dtype=np.complex128)
    if flavor_state.shape != (3,):
        raise ValueError("flavor_state must be a 3-element vector")
    return U.conj().T @ flavor_state


def mass_to_flavor_basis(U, mass_state):
    mass_state = np.asarray(mass_state, dtype=np.complex128)
    if mass_state.shape != (3,):
        raise ValueError("mass_state must be a 3-element vector")
    return U @ mass_state


def get_initial_flavor_state(flavor='electron'):
    flavor_map = {
        'electron': 0,
        'muon': 1,
        'tau': 2,
        'e': 0,
        'mu': 1,
        'tau': 2
    }
    idx = flavor_map.get(flavor.lower(), 0)
    psi = np.zeros(3, dtype=np.complex128)
    psi[idx] = 1.0
    return psi


def jarkslog_invariant(U):
    j = np.imag(U[0, 0] * U[1, 1] * np.conj(U[0, 1]) * np.conj(U[1, 0]))
    return float(j)
