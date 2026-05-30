
import numpy as np
from typing import Tuple, Callable


def local_basis_1d(order: int, node_x: np.ndarray, x: float) -> np.ndarray:
    node_x = np.asarray(node_x, dtype=float).flatten()
    phi = np.ones(order)
    for i in range(order):
        for j in range(order):
            if i != j:
                if abs(node_x[i] - node_x[j]) < 1e-14:
                    raise ValueError("local_basis_1d: 节点坐标重复")
                phi[i] *= (x - node_x[j]) / (node_x[i] - node_x[j])
    return phi


def local_basis_prime_1d(order: int, node_x: np.ndarray, x: float) -> np.ndarray:
    node_x = np.asarray(node_x, dtype=float).flatten()
    dphi = np.zeros(order)
    for i in range(order):
        for k in range(order):
            if i != k:
                prod = 1.0
                for j in range(order):
                    if j != i and j != k:
                        prod *= (x - node_x[j]) / (node_x[i] - node_x[j])
                dphi[i] += prod / (node_x[i] - node_x[k])
    return dphi


def legendre_com(n: int) -> Tuple[np.ndarray, np.ndarray]:
    if n < 1:
        raise ValueError("legendre_com: n 必须大于 0")

    xtab = np.zeros(n)
    weight = np.zeros(n)
    e1 = n * (n + 1)
    m = (n + 1) // 2

    for i in range(1, m + 1):
        mp1mi = m + 1 - i
        t = np.pi * (4.0 * i - 1.0) / (4.0 * n + 2.0)
        x0 = np.cos(t) * (1.0 - (1.0 - 1.0 / n) / (8.0 * n * n))


        for _ in range(10):
            pkm1 = 1.0
            pk = x0
            for k in range(2, n + 1):
                pkp1 = 2.0 * x0 * pk - pkm1 - (x0 * pk - pkm1) / k
                pkm1 = pk
                pk = pkp1
            d1 = n * (pkm1 - x0 * pk)
            dpn = d1 / (1.0 - x0 * x0)
            dx = pk / dpn
            x0 = x0 - dx
            if abs(dx) < 1e-14:
                break

        xtab[mp1mi - 1] = x0

        fx = d1
        weight[mp1mi - 1] = 2.0 * (1.0 - x0 * x0) / (fx * fx)

    if n % 2 == 1:
        xtab[m - 1] = 0.0


    for i in range(1, m + 1):
        xtab[n - i] = -xtab[i - 1]
        weight[n - i] = weight[i - 1]

    return xtab, weight


def assemble_beam_system(
    n_elements: int,
    length: float,
    E: float,
    I: float,
    rho: float,
    A: float,
    q_func: Callable[[np.ndarray], np.ndarray],
    order: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_nodes = n_elements * order + 1
    x_nodes = np.linspace(0.0, length, n_nodes)
    h = length / n_elements

    K = np.zeros((n_nodes, n_nodes))
    M = np.zeros((n_nodes, n_nodes))
    F = np.zeros(n_nodes)

    xtab, wgt = legendre_com(4)

    for e in range(n_elements):
        x_e = x_nodes[e * order : (e + 1) * order + 1]
        x_mid = 0.5 * (x_e[0] + x_e[-1])
        jac = 0.5 * (x_e[-1] - x_e[0])

        Ke = np.zeros((order + 1, order + 1))
        Me = np.zeros((order + 1, order + 1))
        Fe = np.zeros(order + 1)

        for iq in range(len(xtab)):
            xi = xtab[iq]
            x_phys = x_mid + jac * xi
            w = wgt[iq] * jac

            phi = local_basis_1d(order + 1, x_e, x_phys)
            dphi = local_basis_prime_1d(order + 1, x_e, x_phys)

            for i in range(order + 1):
                for j in range(order + 1):

                    Ke[i, j] += E * I * dphi[i] * dphi[j] * w / (h * h)
                    Me[i, j] += rho * A * phi[i] * phi[j] * w
                Fe[i] += q_func(np.array([x_phys]))[0] * phi[i] * w


        for i in range(order + 1):
            gi = e * order + i
            for j in range(order + 1):
                gj = e * order + j
                K[gi, gj] += Ke[i, j]
                M[gi, gj] += Me[i, j]
            F[gi] += Fe[i]

    return K, M, F


def solve_beam_static(
    n_elements: int = 20,
    length: float = 30.0,
    E: float = 2.1e11,
    I: float = 0.5,
    rho: float = 7850.0,
    A: float = 2.0,
    drag_force: float = 5.0e4,
) -> Tuple[np.ndarray, np.ndarray]:
    def q_func(x_arr: np.ndarray) -> np.ndarray:

        x = np.asarray(x_arr)
        q0 = drag_force / length
        return q0 * (1.0 - x / length)

    K, M, F = assemble_beam_system(n_elements, length, E, I, rho, A, q_func)
    n_nodes = n_elements * 2 + 1


    K_reduced = K[1:n_nodes, 1:n_nodes]
    F_reduced = F[1:n_nodes]


    cond_num = np.linalg.cond(K_reduced)
    if cond_num > 1e14:

        K_reduced += 1e-8 * np.eye(n_nodes - 1) * np.max(np.abs(K_reduced))

    w_reduced = np.linalg.solve(K_reduced, F_reduced)
    w = np.concatenate(([0.0], w_reduced))
    x = np.linspace(0.0, length, n_nodes)
    return x, w


def compute_mooring_tension(
    anchor_distance: float = 200.0,
    water_depth: float = 40.0,
    line_density: float = 50.0,
    horizontal_force: float = 1.0e6,
) -> float:
    g = 9.81
    a = horizontal_force / (line_density * g)

    y_mid = a * (np.cosh(anchor_distance / (2.0 * a)) - 1.0)
    if y_mid > water_depth:

        slope = water_depth / (0.5 * anchor_distance)
        T_max = horizontal_force * np.sqrt(1.0 + slope ** 2)
    else:
        T_max = horizontal_force * np.cosh(anchor_distance / (2.0 * a))
    return float(T_max)
