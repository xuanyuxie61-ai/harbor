
import numpy as np
import math
from typing import List, Tuple





def polygon_grid_points(n: int, nv: int, v: np.ndarray, ng: int) -> np.ndarray:
    if n < 1:
        raise ValueError("n must be at least 1.")
    if nv < 3:
        raise ValueError("A polygon must have at least 3 vertices.")
    if ng < 1:
        raise ValueError("ng must be positive.")

    xg = np.zeros((ng, 2))
    p = 0


    vc = np.array([np.sum(v[:, 0]) / nv, np.sum(v[:, 1]) / nv])
    xg[p, :] = vc
    p += 1


    for l in range(nv):
        lp1 = (l + 1) % nv
        for i in range(1, n + 1):
            for j in range(0, n - i + 1):
                if p >= ng:
                    return xg
                k = n - i - j
                xg[p, :] = (i * v[l, :] + j * v[lp1, :] + k * vc) / n
                p += 1

    return xg


def polygon_grid_count(n: int, nv: int) -> int:
    if n < 1 or nv < 3:
        raise ValueError("Invalid parameters.")
    return 1 + nv * n * (n + 1) // 2





def r8mat_solve(n: int, nrhs: int, a: np.ndarray) -> Tuple[np.ndarray, int]:
    a = np.array(a, dtype=float)
    info = 0

    for j in range(n):

        ipivot = j
        apivot = abs(a[j, j])
        for i in range(j + 1, n):
            if abs(a[i, j]) > apivot:
                apivot = abs(a[i, j])
                ipivot = i

        if apivot < 1e-30:
            info = j + 1
            return a, info


        if ipivot != j:
            a[[j, ipivot], :] = a[[ipivot, j], :]


        pivot = a[j, j]
        a[j, :] /= pivot


        for i in range(n):
            if i != j:
                factor = a[i, j]
                a[i, :] -= factor * a[j, :]

    return a, info


def solve_lattice_elasticity(nodes: np.ndarray, elements: List[Tuple[int, ...]],
                             boundary_nodes: List[int],
                             external_force: np.ndarray) -> np.ndarray:
    n_nodes = nodes.shape[0]
    ndof = 2 * n_nodes

    K = np.zeros((ndof, ndof))
    F = np.zeros(ndof)

    for i in range(n_nodes):
        F[2 * i] = external_force[i, 0]
        F[2 * i + 1] = external_force[i, 1]


    E_mod = 1.0
    nu = 0.3
    D_mat = (E_mod / (1.0 - nu**2)) * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, (1.0 - nu) / 2.0]
    ])

    for elem in elements:
        if len(elem) == 3:

            i, j, k = elem
            xi, yi = nodes[i]
            xj, yj = nodes[j]
            xk, yk = nodes[k]

            area = 0.5 * abs((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))
            if area < 1e-14:
                continue


            b1 = yj - yk
            b2 = yk - yi
            b3 = yi - yj
            c1 = xk - xj
            c2 = xi - xk
            c3 = xj - xi

            B = (1.0 / (2.0 * area)) * np.array([
                [b1, 0.0, b2, 0.0, b3, 0.0],
                [0.0, c1, 0.0, c2, 0.0, c3],
                [c1, b1, c2, b2, c3, b3]
            ])

            Ke = area * B.T @ D_mat @ B

            local_dofs = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1, 2 * k, 2 * k + 1]
            for ii in range(6):
                for jj in range(6):
                    K[local_dofs[ii], local_dofs[jj]] += Ke[ii, jj]


    for bn in boundary_nodes:
        for d in range(2):
            dof = 2 * bn + d
            K[dof, :] = 0.0
            K[:, dof] = 0.0
            K[dof, dof] = 1.0
            F[dof] = 0.0


    aug = np.hstack([K, F.reshape(-1, 1)])
    sol, info = r8mat_solve(ndof, 1, aug)

    if info != 0:
        raise RuntimeError(f"Linear system is singular at step {info}.")

    displacement = np.zeros((n_nodes, 2))
    for i in range(n_nodes):
        displacement[i, 0] = sol[ndof - 1, 2 * i]
        displacement[i, 1] = sol[ndof - 1, 2 * i + 1]

    return displacement





def generate_crust_lattice_hexagonal(
    lattice_constant: float,
    n_layers: int
) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    if lattice_constant <= 0.0 or n_layers < 1:
        raise ValueError("Invalid lattice parameters.")

    nodes_list = []

    nodes_list.append([0.0, 0.0])


    for layer in range(1, n_layers + 1):
        for k in range(6 * layer):
            angle = 2.0 * math.pi * k / (6.0 * layer)
            r = layer * lattice_constant
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            nodes_list.append([x, y])

    nodes = np.array(nodes_list)


    elements = []
    n_total = len(nodes_list)


    for k in range(6):
        n0 = 0
        n1 = 1 + k
        n2 = 1 + (k + 1) % 6
        elements.append((n0, n1, n2))


    for layer in range(1, n_layers):
        start_inner = 1 + 3 * layer * (layer - 1)
        n_inner = 6 * layer
        start_outer = start_inner + n_inner
        n_outer = 6 * (layer + 1)

        for k in range(n_inner):
            i1 = start_inner + k
            i2 = start_inner + (k + 1) % n_inner

            ratio = n_outer / n_inner
            o1 = start_outer + int(k * ratio) % n_outer
            o2 = start_outer + int((k + 1) * ratio) % n_outer
            elements.append((i1, i2, o1))
            elements.append((i2, o2, o1))

    return nodes, elements


def compute_crust_shear_modulus(nodes: np.ndarray, elements: List[Tuple[int, int, int]],
                                young_modulus: float = 1.0e35) -> float:
    if len(elements) == 0:
        return 0.0

    total_area = 0.0
    for elem in elements:
        i, j, k = elem
        xi, yi = nodes[i]
        xj, yj = nodes[j]
        xk, yk = nodes[k]
        area = 0.5 * abs((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))
        total_area += area

    nu = 0.3
    shear = young_modulus / (2.0 * (1.0 + nu))
    return shear
