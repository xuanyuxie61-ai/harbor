
import numpy as np
from typing import Tuple, List
from itertools import combinations_with_replacement


def line_unit_o05() -> Tuple[np.ndarray, np.ndarray]:
    x = np.array([
        -0.9061798459386640,
        -0.5384693101056831,
         0.0,
         0.5384693101056831,
         0.9061798459386640
    ])
    w = np.array([
        0.2369268850561891,
        0.4786286704993665,
        0.5688888888888889,
        0.4786286704993665,
        0.2369268850561891
    ])
    return x, w


def line_unit_o03() -> Tuple[np.ndarray, np.ndarray]:
    x = np.array([-np.sqrt(3.0/5.0), 0.0, np.sqrt(3.0/5.0)])
    w = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])
    return x, w


def transform_interval(xi: np.ndarray, a: float, b: float) -> np.ndarray:
    return 0.5 * (b - a) * xi + 0.5 * (a + b)


def cube_rule(ax: float, bx: float, ay: float, by: float, az: float, bz: float,
              order_1d: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    if order_1d == 3:
        xi, wi = line_unit_o03()
    elif order_1d == 5:
        xi, wi = line_unit_o05()
    else:
        raise ValueError("Only order_1d = 3 or 5 is supported")
    
    x_nodes = transform_interval(xi, ax, bx)
    y_nodes = transform_interval(xi, ay, by)
    z_nodes = transform_interval(xi, az, bz)
    
    volume = (bx - ax) * (by - ay) * (bz - az)
    scale = volume / 8.0
    

    n = order_1d
    nodes = np.zeros((n ** 3, 3))
    weights = np.zeros(n ** 3)
    idx = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                nodes[idx, 0] = x_nodes[i]
                nodes[idx, 1] = y_nodes[j]
                nodes[idx, 2] = z_nodes[k]
                weights[idx] = wi[i] * wi[j] * wi[k] * scale
                idx += 1
    return nodes, weights


def cube_monomial_integral(ax: float, bx: float, ay: float, by: float,
                           az: float, bz: float, alpha: int, beta: int, gamma_exp: int) -> float:
    if alpha < 0 or beta < 0 or gamma_exp < 0:
        raise ValueError("Exponents must be non-negative")
    
    def power_integral(a, b, p):
        if p == -1:
            return np.log(b / a) if a * b > 0 else 0.0
        return (b ** (p + 1) - a ** (p + 1)) / (p + 1)
    
    Ix = power_integral(ax, bx, alpha)
    Iy = power_integral(ay, by, beta)
    Iz = power_integral(az, bz, gamma_exp)
    return float(Ix * Iy * Iz)


def integrate_partition_function_subdomain(coords_min: np.ndarray, coords_max: np.ndarray,
                                           potential_func: callable,
                                           kT: float = 1.0,
                                           order_1d: int = 5) -> float:
    ax, ay, az = coords_min
    bx, by, bz = coords_max
    nodes, weights = cube_rule(ax, bx, ay, by, az, bz, order_1d)
    energies = potential_func(nodes)
    boltzmann = np.exp(-energies / kT)
    Z = float(np.sum(boltzmann * weights))
    return Z


def comp_next(n: int, k: int) -> List[Tuple[int, ...]]:
    if k == 1:
        return [(n,)]
    result = []
    for i in range(n + 1):
        for tail in comp_next(n - i, k - 1):
            result.append((i,) + tail)
    return result


def test_cube_rule_precision(ax: float, bx: float, ay: float, by: float,
                             az: float, bz: float, max_degree: int = 4) -> dict:
    errors = {}
    for order_1d in [3, 5]:
        for total_deg in range(max_degree + 1):
            for comp in comp_next(total_deg, 3):
                alpha, beta, gamma_exp = comp
                nodes, weights = cube_rule(ax, bx, ay, by, az, bz, order_1d)
                fvals = (nodes[:, 0] ** alpha) * (nodes[:, 1] ** beta) * (nodes[:, 2] ** gamma_exp)
                num_int = np.sum(fvals * weights)
                ana_int = cube_monomial_integral(ax, bx, ay, by, az, bz, alpha, beta, gamma_exp)
                key = (order_1d, alpha, beta, gamma_exp)
                errors[key] = abs(num_int - ana_int)
    return errors
