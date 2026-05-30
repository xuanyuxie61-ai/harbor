
import numpy as np
from typing import Tuple, Optional




EPSILON_MACHINE = np.finfo(float).eps
MAX_ITERATIONS = 10000
TOLERANCE_DEFAULT = 1e-10


ELEMENTARY_CHARGE = 4.80320427e-10
ELECTRON_MASS = 9.10938356e-28
BOLTZMANN_CONSTANT = 1.380649e-16


def safe_divide(a: np.ndarray, b: np.ndarray, fallback: float = 0.0) -> np.ndarray:
    b = np.asarray(b, dtype=float)
    a = np.asarray(a, dtype=float)
    result = np.empty_like(a, dtype=float)
    mask = np.abs(b) > EPSILON_MACHINE * 100.0
    result[mask] = a[mask] / b[mask]
    result[~mask] = fallback
    return result


def check_bounds(x: np.ndarray, x_min: float, x_max: float,
                 name: str = "variable") -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x_clipped = np.clip(x, x_min, x_max)
    violations = np.sum((x < x_min) | (x > x_max))
    if violations > 0:
        print(f"[WARNING] {name}: {violations} values out of bounds [{x_min}, {x_max}], clipped.")
    return x_clipped


def compute_triangle_area(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    p1, p2, p3 = np.asarray(p1), np.asarray(p2), np.asarray(p3)
    area = 0.5 * ((p2[0] - p1[0]) * (p3[1] - p1[1])
                  - (p3[0] - p1[0]) * (p2[1] - p1[1]))
    if abs(area) < 1e-14:
        print("[WARNING] Triangle area near zero; degenerate element detected.")
        return 0.0
    return area


def reference_to_physical_q4(q4: np.ndarray, rs: np.ndarray) -> np.ndarray:
    q4 = np.asarray(q4, dtype=float)
    rs = np.asarray(rs, dtype=float)
    n = rs.shape[0]
    r = rs[:, 0]
    s = rs[:, 1]


    r = check_bounds(r, 0.0, 1.0, "r")
    s = check_bounds(s, 0.0, 1.0, "s")

    psi = np.zeros((4, n))
    psi[0, :] = (1.0 - r) * (1.0 - s)
    psi[1, :] = r * (1.0 - s)
    psi[2, :] = r * s
    psi[3, :] = (1.0 - r) * s

    xy = q4.T @ psi
    return xy.T


def mesh_base_one(element_node: np.ndarray, node_num: int) -> np.ndarray:
    element_node = np.asarray(element_node, dtype=int)
    node_min = element_node.min()
    node_max = element_node.max()

    if node_min == 0 and node_max == node_num - 1:
        print("[INFO] Detected 0-based indexing; converting to 1-based.")
        return element_node + 1
    elif node_min == 1 and node_max == node_num:
        return element_node
    else:
        raise ValueError(
            f"Mesh indexing inconsistent: node_min={node_min}, node_max={node_max}, node_num={node_num}"
        )


def gauss_seidel_sweep(n: int, rhs: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, float]:
    x = np.asarray(x, dtype=float).copy()
    rhs = np.asarray(rhs, dtype=float)
    if x.size < n or rhs.size < n:
        raise ValueError("Input vectors too short for given n.")

    x_new = x.copy()
    for i in range(1, n - 1):
        x_new[i] = 0.5 * (rhs[i] + x_new[i - 1] + x[i + 1])

    d = np.max(np.abs(x_new - x))
    return x_new, d


def restrict_coarse_to_fine(n_coarse: int, u_coarse: np.ndarray, n_fine: int,
                            u_fine: np.ndarray) -> np.ndarray:
    u_coarse = np.asarray(u_coarse, dtype=float)
    u_fine = np.asarray(u_fine, dtype=float).copy()
    expected_fine = 2 * (n_coarse - 1) + 1
    if n_fine != expected_fine:
        raise ValueError(f"Fine grid size mismatch: expected {expected_fine}, got {n_fine}")
    if u_fine.size < n_fine or u_coarse.size < n_coarse:
        raise ValueError("Solution arrays too short.")

    for j in range(n_coarse - 1):
        u_fine[2 * j] = u_coarse[j]
        u_fine[2 * j + 1] = 0.5 * (u_coarse[j] + u_coarse[j + 1])
    u_fine[n_fine - 1] = u_coarse[n_coarse - 1]
    return u_fine


def restrict_fine_to_coarse(n_fine: int, u_fine: np.ndarray, rhs_fine: np.ndarray,
                            n_coarse: int) -> Tuple[np.ndarray, np.ndarray]:
    u_fine = np.asarray(u_fine, dtype=float)
    rhs_fine = np.asarray(rhs_fine, dtype=float)
    expected_coarse = (n_fine - 1) // 2 + 1
    if n_coarse != expected_coarse:
        raise ValueError(f"Coarse grid size mismatch: expected {expected_coarse}, got {n_coarse}")
    if (n_fine - 1) % 2 != 0:
        raise ValueError("n_fine must be odd for standard restriction.")

    u_coarse = np.zeros(n_coarse)
    rhs_coarse = np.zeros(n_coarse)


    u_coarse[0] = u_fine[0]
    rhs_coarse[0] = rhs_fine[0]
    u_coarse[n_coarse - 1] = u_fine[n_fine - 1]
    rhs_coarse[n_coarse - 1] = rhs_fine[n_fine - 1]

    for j in range(1, n_coarse - 1):
        fine_idx = 2 * j
        rhs_coarse[j] = (
            0.25 * rhs_fine[fine_idx - 1]
            + 0.5 * rhs_fine[fine_idx]
            + 0.25 * rhs_fine[fine_idx + 1]
        )
        u_coarse[j] = u_fine[fine_idx]

    return u_coarse, rhs_coarse


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0
