
import numpy as np
from typing import Tuple, Callable, Optional
from utils import EPS_MACHINE, rms_norm


def basis_phi(x: float, xL: float, xR: float, derivative: bool = False) -> float:
    h = xR - xL
    if abs(h) < EPS_MACHINE:
        return 0.0
    if derivative:
        return -1.0 / h
    return (xR - x) / h


def basis_psi(x: float, xL: float, xR: float, derivative: bool = False) -> float:
    h = xR - xL
    if abs(h) < EPS_MACHINE:
        return 0.0
    if derivative:
        return 1.0 / h
    return (x - xL) / h


def assemble_tridiagonal_system(
    nodes: np.ndarray,
    A_func: Callable[[float], float],
    B_func: Callable[[float], float],
    F_func: Callable[[float], float],
    m_left: float,
    m_right: float,
    bc_left: str = "dirichlet",
    bc_right: str = "dirichlet",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = nodes.size - 1
    N = nodes.size
    a_diag = np.zeros(N, dtype=float)
    a_left = np.zeros(N, dtype=float)
    a_right = np.zeros(N, dtype=float)
    rhs = np.zeros(N, dtype=float)

    for ie in range(n):
        xL = nodes[ie]
        xR = nodes[ie + 1]
        h = xR - xL
        if h <= 0.0:
            continue

        xq1 = 0.5 * (xL + xR) - h / (2.0 * np.sqrt(3.0))
        xq2 = 0.5 * (xL + xR) + h / (2.0 * np.sqrt(3.0))
        wq = 0.5

        for iq, xq in enumerate([xq1, xq2]):









            raise NotImplementedError("Hole_3: 请实现 assemble_tridiagonal_system 中的单元刚度矩阵组装")


    if bc_left == "dirichlet":
        a_diag[0] = 1.0
        a_right[0] = 0.0
        rhs[0] = m_left
        a_left[0] = 0.0
    else:
        rhs[0] += m_left

    if bc_right == "dirichlet":
        a_diag[-1] = 1.0
        a_left[-1] = 0.0
        rhs[-1] = m_right
        a_right[-1] = 0.0
    else:
        rhs[-1] += m_right

    return a_left, a_diag, a_right, rhs


def solve_tridiagonal(
    a_left: np.ndarray, a_diag: np.ndarray, a_right: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
    n = rhs.size
    c_prime = np.zeros(n - 1, dtype=float)
    d_prime = np.zeros(n, dtype=float)
    c_prime[0] = a_right[0] / a_diag[0]
    d_prime[0] = rhs[0] / a_diag[0]

    for i in range(1, n - 1):
        denom = a_diag[i] - a_left[i] * c_prime[i - 1]
        if abs(denom) < EPS_MACHINE:
            denom = EPS_MACHINE
        c_prime[i] = a_right[i] / denom
        d_prime[i] = (rhs[i] - a_left[i] * d_prime[i - 1]) / denom

    denom = a_diag[n - 1] - a_left[n - 1] * c_prime[n - 2]
    if abs(denom) < EPS_MACHINE:
        denom = EPS_MACHINE
    d_prime[n - 1] = (rhs[n - 1] - a_left[n - 1] * d_prime[n - 2]) / denom

    x = np.zeros(n, dtype=float)
    x[n - 1] = d_prime[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]
    return x


def refine_mesh_locally(
    nodes: np.ndarray,
    solution: np.ndarray,
    error_threshold: float = 0.01,
    max_nodes: int = 200,
) -> np.ndarray:
    n = nodes.size
    if n >= max_nodes:
        return nodes
    new_nodes = [nodes[0]]
    current_n = n
    for i in range(n - 1):
        h = nodes[i + 1] - nodes[i]

        if i == 0 or i == n - 2:
            m_pp = 0.0
        else:
            m_pp = (solution[i + 2] - 2.0 * solution[i + 1] + solution[i]) / (h * h)
        eta = abs(h * h * m_pp)
        if eta > error_threshold and current_n < max_nodes:
            mid = 0.5 * (nodes[i] + nodes[i + 1])
            new_nodes.append(mid)
            new_nodes.append(nodes[i + 1])
            current_n += 1
        else:
            new_nodes.append(nodes[i + 1])
    return np.array(new_nodes)


def adaptive_fem_order_parameter(
    A_func: Callable[[float], float],
    B_func: Callable[[float], float],
    F_func: Callable[[float], float],
    m_left: float = 0.0,
    m_right: float = 1.0,
    n_initial: int = 8,
    max_refinements: int = 6,
    error_threshold: float = 0.005,
    max_nodes: int = 300,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    nodes = np.linspace(0.0, 1.0, n_initial + 1)
    history = []

    for step in range(max_refinements):
        aL, aD, aR, rhs = assemble_tridiagonal_system(
            nodes, A_func, B_func, F_func, m_left, m_right
        )
        sol = solve_tridiagonal(aL, aD, aR, rhs)

        n_nodes = nodes.size
        errors = np.zeros(n_nodes - 1)
        energy_density = np.zeros(n_nodes - 1)
        for ie in range(n_nodes - 1):
            h = nodes[ie + 1] - nodes[ie]
            if ie > 0 and ie < n_nodes - 2:
                m_pp = (sol[ie + 1] - 2.0 * sol[ie] + sol[ie - 1]) / ((nodes[ie] - nodes[ie - 1]) ** 2)
            else:
                m_pp = 0.0
            errors[ie] = abs(h * h * m_pp)

            mid = 0.5 * (nodes[ie] + nodes[ie + 1])
            mp = (sol[ie + 1] - sol[ie]) / h
            energy_density[ie] = 0.5 * A_func(mid) * mp * mp + 0.5 * B_func(mid) * sol[ie] ** 2 - F_func(mid) * sol[ie]

        max_err = float(np.max(errors))
        history.append({"step": step, "n_nodes": n_nodes, "max_error": max_err})
        if max_err < error_threshold or n_nodes >= max_nodes:
            break
        nodes = refine_mesh_locally(nodes, sol, error_threshold, max_nodes)



    aL, aD, aR, rhs = assemble_tridiagonal_system(
        nodes, A_func, B_func, F_func, m_left, m_right
    )
    solution = solve_tridiagonal(aL, aD, aR, rhs)
    n_nodes = nodes.size
    energy_density = np.zeros(n_nodes - 1)
    for ie in range(n_nodes - 1):
        h = nodes[ie + 1] - nodes[ie]
        mid = 0.5 * (nodes[ie] + nodes[ie + 1])
        mp = (solution[ie + 1] - solution[ie]) / h
        energy_density[ie] = 0.5 * A_func(mid) * mp * mp + 0.5 * B_func(mid) * solution[ie] ** 2 - F_func(mid) * solution[ie]

    return nodes, solution, energy_density, history
