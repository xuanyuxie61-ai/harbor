
import numpy as np
from typing import Tuple, Dict
from utils import (
    validate_array_1d,
    validate_array_2d,
    build_sparse_hamiltonian_indices,
    spmatvec,
    tridiagonal_solve,
)



H_BAR = 1.054571817e-34
M_ELECTRON = 9.10938356e-31
EV_TO_J = 1.602176634e-19


def effective_mass(material: str = "InAs") -> float:
    table = {
        "InAs": 0.023,
        "GaAs": 0.067,
        "InP": 0.077,
    }
    return table.get(material, 0.067)


def spherical_confinement_potential(r: np.ndarray, R_dot: float, V0: float) -> np.ndarray:
    r = validate_array_1d(r, "r")
    V = np.where(r <= R_dot, 0.0, V0)
    return V


def stark_field_potential(x: np.ndarray, F_field: float) -> np.ndarray:
    x = validate_array_1d(x, "x")
    e_charge = 1.602176634e-19
    return -e_charge * F_field * x


def harmonic_confinement_potential(r: np.ndarray, hw: float) -> np.ndarray:
    r = validate_array_1d(r, "r")
    if hw <= 0:
        raise ValueError("hw must be positive")
    V = 0.5 * hw * (r ** 2)
    return V


def coulomb_potential_1d(x: np.ndarray, eps_r: float = 12.9) -> np.ndarray:
    x = validate_array_1d(x, "x")
    eps0 = 8.854187817e-12
    e_charge = 1.602176634e-19
    a_B_eff = 30.0e-9
    abs_x = np.abs(x)
    V = - (e_charge ** 2) / (4.0 * np.pi * eps0 * eps_r * (abs_x + a_B_eff))
    return V


def build_kinetic_hamiltonian_1d(
    x_grid: np.ndarray, m_star_ratio: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_grid = validate_array_1d(x_grid, "x_grid")
    n = x_grid.size
    if n < 3:
        raise ValueError("Grid must have at least 3 points")
    dx = float(x_grid[1] - x_grid[0])
    if abs(dx) < 1e-20:
        raise ValueError("Grid spacing dx is too small or non-uniform")
    m_star = m_star_ratio * M_ELECTRON
    coeff = (H_BAR ** 2) / (2.0 * m_star * (dx ** 2))
    rows, cols, data = build_sparse_hamiltonian_indices(n)

    data = data * coeff


    return rows, cols, data


def add_potential_to_hamiltonian(
    rows: np.ndarray,
    cols: np.ndarray,
    data: np.ndarray,
    V: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    V = validate_array_1d(V, "V")
    n = V.size
    new_rows = list(rows)
    new_cols = list(cols)
    new_data = list(data)
    for i in range(n):
        new_rows.append(i)
        new_cols.append(i)
        new_data.append(V[i])
    return (
        np.array(new_rows, dtype=int),
        np.array(new_cols, dtype=int),
        np.array(new_data, dtype=float),
    )


def sparse_to_dense(rows: np.ndarray, cols: np.ndarray, data: np.ndarray, n: int) -> np.ndarray:
    A = np.zeros((n, n), dtype=float)
    for r, c, d in zip(rows, cols, data):
        if 0 <= r < n and 0 <= c < n:
            A[r, c] += d
    return A


def solve_eigenvalues_1d(
    x_grid: np.ndarray,
    m_star_ratio: float,
    potential_type: str = "spherical",
    **potential_params,
) -> Dict[str, np.ndarray]:
    x_grid = validate_array_1d(x_grid, "x_grid")
    n = x_grid.size
    rows, cols, data = build_kinetic_hamiltonian_1d(x_grid, m_star_ratio)

    if potential_type == "spherical":
        R_dot = potential_params.get("R_dot", 5.0e-9)
        V0 = potential_params.get("V0", 0.5 * EV_TO_J)
        V = spherical_confinement_potential(x_grid, R_dot, V0)
    elif potential_type == "harmonic":
        hw = potential_params.get("hw", 0.05 * EV_TO_J)
        V = harmonic_confinement_potential(x_grid, hw)
    else:
        V = np.zeros_like(x_grid)

    rows, cols, data = add_potential_to_hamiltonian(rows, cols, data, V)
    H_dense = sparse_to_dense(rows, cols, data, n)


    H_dense = 0.5 * (H_dense + H_dense.T)


    H_dense = np.real(H_dense)


    eigvals, eigvecs = np.linalg.eigh(H_dense)


    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    return {
        "energies_J": eigvals,
        "energies_eV": eigvals / EV_TO_J,
        "wavefunctions": eigvecs,
        "x_grid": x_grid,
    }


def dipole_matrix_element_1d(
    psi_a: np.ndarray, psi_b: np.ndarray, x_grid: np.ndarray
) -> float:
    psi_a = validate_array_1d(psi_a, "psi_a")
    psi_b = validate_array_1d(psi_b, "psi_b")
    x_grid = validate_array_1d(x_grid, "x_grid")
    if not (psi_a.size == psi_b.size == x_grid.size):
        raise ValueError("Array sizes must match")
    dx = x_grid[1] - x_grid[0]
    integrand = psi_a * x_grid * psi_b
    d_ab = np.trapezoid(integrand, x_grid)

    e_charge = 1.602176634e-19
    return e_charge * d_ab


def exciton_binding_energy_1d(
    x_grid: np.ndarray,
    psi_e: np.ndarray,
    psi_h: np.ndarray,
    eps_r: float = 12.9,
) -> float:
    x_grid = validate_array_1d(x_grid, "x_grid")
    psi_e = validate_array_1d(psi_e, "psi_e")
    psi_h = validate_array_1d(psi_h, "psi_h")
    n = x_grid.size
    dx = float(x_grid[1] - x_grid[0])
    rho_e = np.abs(psi_e) ** 2
    rho_h = np.abs(psi_h) ** 2
    E_bind = 0.0
    for i in range(n):
        for j in range(n):
            dx_ij = x_grid[i] - x_grid[j]
            Vc = coulomb_potential_1d(np.array([dx_ij]), eps_r)[0]
            E_bind += rho_e[i] * rho_h[j] * Vc * (dx ** 2)
    return -E_bind


def reduced_mass(m_e_star: float, m_h_star: float) -> float:
    if m_e_star <= 0 or m_h_star <= 0:
        raise ValueError("Effective masses must be positive")
    return (m_e_star * m_h_star) / (m_e_star + m_h_star)
