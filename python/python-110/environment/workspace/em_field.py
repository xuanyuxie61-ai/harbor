
import numpy as np
from typing import Tuple, Dict
from utils import validate_array_1d, validate_array_2d
from mesh_generator import reference_to_physical_q4, quadrilateral_area



C_LIGHT = 2.99792458e8
MU0 = 4.0 * np.pi * 1e-7
EPS0 = 8.854187817e-12


def gaussian_mode_profile(
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
    y0: float,
    w0: float,
    amplitude: float = 1.0,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r2 = (x - x0) ** 2 + (y - y0) ** 2
    E = amplitude * np.exp(-r2 / (w0 ** 2))
    return E


def lorentzian_cavity_mode(
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
    y0: float,
    R_cavity: float,
    n_eff: float = 3.5,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    E = 1.0 / (1.0 + (r / R_cavity) ** 4)
    return E


def effective_mode_volume_2d(
    nodes: np.ndarray,
    elements: np.ndarray,
    E_field: np.ndarray,
    epsilon_r: np.ndarray,
) -> float:
    nodes = validate_array_2d(nodes, "nodes")
    elements = validate_array_2d(elements, "elements")
    E_field = validate_array_1d(E_field, "E_field")
    epsilon_r = validate_array_1d(epsilon_r, "epsilon_r")
    if elements.shape[0] != 4:
        raise ValueError("Only Q4 elements supported")
    n_elem = elements.shape[1]
    if E_field.size != nodes.shape[1]:
        raise ValueError("E_field length must match number of nodes")
    if epsilon_r.size != n_elem and epsilon_r.size != nodes.shape[1]:
        raise ValueError("epsilon_r length mismatch")

    from mesh_generator import quadrilateral_area
    total = 0.0
    max_val = 0.0
    for e in range(n_elem):
        q4 = np.zeros((2, 4), dtype=float)
        for k in range(4):
            q4[:, k] = nodes[:, elements[k, e]]
        area = quadrilateral_area(q4)

        E_center = 0.0
        for k in range(4):
            E_center += E_field[elements[k, e]]
        E_center /= 4.0
        if epsilon_r.size == n_elem:
            eps = epsilon_r[e]
        else:
            eps = 0.0
            for k in range(4):
                eps += epsilon_r[elements[k, e]]
            eps /= 4.0
        val = eps * (E_center ** 2)
        total += val * area
        if val > max_val:
            max_val = val
    if max_val < 1e-20:
        max_val = 1e-20
    return total / max_val


def purcell_factor(
    Q: float,
    V_eff: float,
    wavelength: float,
    n_eff: float = 3.5,
) -> float:
    if Q <= 0 or V_eff <= 0 or wavelength <= 0:
        raise ValueError("Q, V_eff, and wavelength must be positive")


    raise NotImplementedError("Hole 4: 请实现 purcell_factor 函数体")


def interpolate_field_on_mesh(
    nodes: np.ndarray,
    E_nodes: np.ndarray,
    query_points: np.ndarray,
) -> np.ndarray:
    nodes = validate_array_2d(nodes, "nodes")
    E_nodes = validate_array_1d(E_nodes, "E_nodes")
    query_points = validate_array_2d(query_points, "query_points")
    if nodes.shape[0] != 2 or query_points.shape[0] != 2:
        raise ValueError("Coordinates must be 2D")
    n_query = query_points.shape[1]
    n_nodes = nodes.shape[1]
    E_query = np.zeros(n_query, dtype=float)
    p = 2.0
    for q in range(n_query):
        xq, yq = query_points[0, q], query_points[1, q]
        dist2 = (nodes[0, :] - xq) ** 2 + (nodes[1, :] - yq) ** 2
        dist2 = np.where(dist2 < 1e-20, 1e-20, dist2)
        w = 1.0 / (dist2 ** (p / 2.0))
        E_query[q] = np.sum(w * E_nodes) / np.sum(w)
    return E_query


def fem_mode_solver_1d(
    x: np.ndarray,
    epsilon_profile: np.ndarray,
    target_wavelength: float,
) -> Dict[str, np.ndarray]:
    x = validate_array_1d(x, "x")
    epsilon_profile = validate_array_1d(epsilon_profile, "epsilon_profile")
    if x.size != epsilon_profile.size:
        raise ValueError("x and epsilon_profile must have same size")
    n = x.size
    dx = float(x[1] - x[0])
    if abs(dx) < 1e-20:
        raise ValueError("Grid spacing too small")
    k0 = 2.0 * np.pi / target_wavelength


    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        A[i, i] = -2.0 / (dx ** 2) + k0 ** 2 * epsilon_profile[i]
        if i > 0:
            A[i, i - 1] = 1.0 / (dx ** 2)
        if i < n - 1:
            A[i, i + 1] = 1.0 / (dx ** 2)

    A[0, :] = 0.0
    A[0, 0] = 1.0
    A[n - 1, :] = 0.0
    A[n - 1, n - 1] = 1.0

    eigvals, eigvecs = np.linalg.eigh(A)

    idx = np.argsort(-eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    return {
        "beta2": eigvals,
        "beta": np.sqrt(np.maximum(eigvals, 0.0)),
        "mode_profiles": eigvecs,
        "x": x,
    }


def spontaneous_emission_rate(
    dipole_moment: float,
    omega: float,
    local_density_of_states: float,
) -> float:
    if omega <= 0 or local_density_of_states < 0:
        raise ValueError("omega must be positive and LDOS non-negative")
    gamma = (
        (omega ** 3) * (dipole_moment ** 2) * local_density_of_states
        / (3.0 * np.pi * EPS0 * (1.054571817e-34) * (C_LIGHT ** 3))
    )
    return gamma
