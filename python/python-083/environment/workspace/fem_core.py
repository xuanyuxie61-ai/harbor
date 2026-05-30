
import numpy as np
from typing import Tuple, List, Optional






def build_vtoe(element_node: np.ndarray, n_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    n_elements, n_local = element_node.shape

    degree = np.zeros(n_nodes, dtype=np.int32)
    for e in range(n_elements):
        for k in range(n_local):
            v = element_node[e, k]
            if 0 <= v < n_nodes:
                degree[v] += 1

    vtoe_ptr = np.zeros(n_nodes + 1, dtype=np.int32)
    vtoe_ptr[1:] = np.cumsum(degree)

    temp_ptr = vtoe_ptr[:-1].copy()
    vtoe = np.empty(vtoe_ptr[-1], dtype=np.int32)

    for e in range(n_elements):
        for k in range(n_local):
            v = element_node[e, k]
            if 0 <= v < n_nodes:
                idx = temp_ptr[v]
                vtoe[idx] = e
                temp_ptr[v] += 1

    return vtoe_ptr, vtoe


def triangulation_t3_to_t4(node_xy: np.ndarray, element_node: np.ndarray):
    n_elements = element_node.shape[0]
    n_nodes_orig = node_xy.shape[0]

    new_xy = np.zeros((n_elements, 2), dtype=np.float64)
    for e in range(n_elements):
        v0, v1, v2 = element_node[e, :3]
        new_xy[e, 0] = (node_xy[v0, 0] + node_xy[v1, 0] + node_xy[v2, 0]) / 3.0
        new_xy[e, 1] = (node_xy[v0, 1] + node_xy[v1, 1] + node_xy[v2, 1]) / 3.0

    node_xy4 = np.vstack([node_xy, new_xy])
    element_node4 = np.zeros((n_elements, 4), dtype=np.int32)
    element_node4[:, :3] = element_node[:, :3]
    element_node4[:, 3] = np.arange(n_nodes_orig, n_nodes_orig + n_elements)
    return node_xy4, element_node4






def grlex_rank(mono: Tuple[int, ...]) -> int:
    d = len(mono)
    t = sum(mono)

    rank = _nchoosek(t + d, d) - 1

    s = t
    for i in range(d):
        for j in range(mono[i]):
            rank += _nchoosek(s - j + d - i - 1, d - i - 1)
        s -= mono[i]
    return rank


def _nchoosek(n: int, k: int) -> int:
    if k < 0 or k > n or n < 0:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    res = 1
    for i in range(1, k + 1):
        res = res * (n - k + i) // i
    return res


def shape_function_t3(xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
    return np.array([1.0 - xi - eta, xi, eta], dtype=np.float64)


def shape_function_t4(xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
    L0 = 1.0 - xi - eta
    L1 = xi
    L2 = eta
    Lb = 27.0 * L0 * L1 * L2
    N = np.array([L0 - Lb / 3.0, L1 - Lb / 3.0, L2 - Lb / 3.0, Lb], dtype=np.float64)
    return N


def dshape_t3(xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
    return np.array([[-1.0, 1.0, 0.0],
                     [-1.0, 0.0, 1.0]], dtype=np.float64).T


def dshape_t4(xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
    L0 = 1.0 - xi - eta
    L1 = xi
    L2 = eta

    dLb_dxi = 27.0 * ((-1.0) * L1 * L2 + L0 * 1.0 * L2 + L0 * L1 * 0.0)
    dLb_deta = 27.0 * ((-1.0) * L1 * L2 + L0 * 0.0 * L2 + L0 * L1 * 1.0)
    dN = np.zeros((4, 2), dtype=np.float64)
    dN[0, 0] = -1.0 - dLb_dxi / 3.0
    dN[0, 1] = -1.0 - dLb_deta / 3.0
    dN[1, 0] = 1.0 - dLb_dxi / 3.0
    dN[1, 1] = 0.0 - dLb_deta / 3.0
    dN[2, 0] = 0.0 - dLb_dxi / 3.0
    dN[2, 1] = 1.0 - dLb_deta / 3.0
    dN[3, 0] = dLb_dxi
    dN[3, 1] = dLb_deta
    return dN






def gauss_triangle(order: int = 3):
    if order == 1:
        pts = np.array([[1.0/3.0, 1.0/3.0]], dtype=np.float64)
        wts = np.array([1.0], dtype=np.float64)
    elif order == 3:
        pts = np.array([[2.0/3.0, 1.0/6.0],
                        [1.0/6.0, 2.0/3.0],
                        [1.0/6.0, 1.0/6.0]], dtype=np.float64)
        wts = np.array([1.0/3.0, 1.0/3.0, 1.0/3.0], dtype=np.float64)
    elif order == 4:
        a = 0.816847572980459
        b = 0.091576213509771
        c = 0.108103018168070
        d = 0.445948490915965
        w1 = 0.109951743655322
        w2 = 0.223381589678011
        pts = np.array([[a, b], [b, a], [b, b],
                        [d, c], [c, d], [d, d]], dtype=np.float64)
        wts = np.array([w1, w1, w1, w2, w2, w2], dtype=np.float64)
    elif order == 7:

        a1, a2 = 0.101286507323456, 0.797426985353087
        b1, b2 = 0.470142064105115, 0.059715871789770
        c = 0.333333333333333
        w1 = 0.125939180544827
        w2 = 0.132394152788506
        w3 = 0.225000000000000
        pts = np.array([[a1, a2], [a2, a1], [a1, a1],
                        [b1, b2], [b2, b1], [b1, b1],
                        [c, c]], dtype=np.float64)
        wts = np.array([w1, w1, w1, w2, w2, w2, w3], dtype=np.float64)
    else:
        raise ValueError(f"Unsupported Gauss order {order}")

    wts = wts * 0.5
    return pts, wts






def elastic_d_matrix(E: float, nu: float, plane_stress: bool = True) -> np.ndarray:
    D = np.zeros((3, 3), dtype=np.float64)



    raise NotImplementedError("Hole 1: elastic_d_matrix 未实现")
    return D






def compute_element_stiffness_t3(node_xy_e: np.ndarray, E: float, nu: float,
                                  plane_stress: bool = True) -> np.ndarray:
    D = elastic_d_matrix(E, nu, plane_stress)
    pts, wts = gauss_triangle(order=3)
    Ke = np.zeros((6, 6), dtype=np.float64)

    dN_dxi = dshape_t3(0.0, 0.0)
    J = dN_dxi.T @ node_xy_e
    detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
    if abs(detJ) < 1e-14:
        raise ValueError("Degenerate triangle element detected (detJ ~ 0).")
    invJ = np.linalg.inv(J)
    dN_dx = dN_dxi @ invJ

    for q in range(len(wts)):
        B = np.zeros((3, 6), dtype=np.float64)
        for i in range(3):
            B[0, 2*i]   = dN_dx[i, 0]
            B[1, 2*i+1] = dN_dx[i, 1]
            B[2, 2*i]   = dN_dx[i, 1]
            B[2, 2*i+1] = dN_dx[i, 0]
        Ke += B.T @ D @ B * wts[q] * abs(detJ)
    return Ke


def compute_element_stiffness_t4(node_xy_e: np.ndarray, E: float, nu: float,
                                  plane_stress: bool = True) -> np.ndarray:
    D = elastic_d_matrix(E, nu, plane_stress)
    pts, wts = gauss_triangle(order=4)
    Ke = np.zeros((8, 8), dtype=np.float64)
    for q in range(len(wts)):
        xi, eta = pts[q, 0], pts[q, 1]
        dN_dxi = dshape_t4(xi, eta)
        J = dN_dxi.T @ node_xy_e
        detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if abs(detJ) < 1e-14:
            raise ValueError("Degenerate T4 element detected.")
        invJ = np.linalg.inv(J)
        dN_dx = dN_dxi @ invJ
        B = np.zeros((3, 8), dtype=np.float64)
        for i in range(4):
            B[0, 2*i]   = dN_dx[i, 0]
            B[1, 2*i+1] = dN_dx[i, 1]
            B[2, 2*i]   = dN_dx[i, 1]
            B[2, 2*i+1] = dN_dx[i, 0]
        Ke += B.T @ D @ B * wts[q] * abs(detJ)
    return Ke






def assemble_global_stiffness(node_xy: np.ndarray, element_node: np.ndarray,
                               E: float, nu: float, plane_stress: bool = True,
                               element_type: str = "T3") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_nodes = node_xy.shape[0]
    n_elements = element_node.shape[0]
    n_local = element_node.shape[1]
    dof_per_node = 2
    n_edof = n_local * dof_per_node

    rows = []
    cols = []
    vals = []

    for e in range(n_elements):
        enodes = element_node[e, :]
        node_xy_e = node_xy[enodes, :]
        if element_type.upper() == "T3":
            Ke = compute_element_stiffness_t3(node_xy_e, E, nu, plane_stress)
        elif element_type.upper() == "T4":
            Ke = compute_element_stiffness_t4(node_xy_e, E, nu, plane_stress)
        else:
            raise ValueError(f"Unknown element type: {element_type}")

        edof = np.zeros(n_edof, dtype=np.int32)
        for i in range(n_local):
            edof[2*i]   = enodes[i] * 2
            edof[2*i+1] = enodes[i] * 2 + 1

        for i in range(n_edof):
            for j in range(n_edof):
                rows.append(edof[i])
                cols.append(edof[j])
                vals.append(Ke[i, j])

    return np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32), np.array(vals, dtype=np.float64)






def apply_dirichlet_bcs(K_dense: np.ndarray, F: np.ndarray, bc_nodes: np.ndarray,
                         bc_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    K_mod = K_dense.copy()
    F_mod = F.copy()
    penalty = 1e12 * np.max(np.abs(K_dense))
    if penalty == 0.0:
        penalty = 1e12

    for i, dof in enumerate(bc_nodes):
        val = bc_values[i]
        K_mod[dof, :] = 0.0
        K_mod[:, dof] = 0.0
        K_mod[dof, dof] = penalty
        F_mod[dof] = val * penalty
    return K_mod, F_mod


def solve_fem_system(node_xy: np.ndarray, element_node: np.ndarray,
                      E: float, nu: float, F_ext: np.ndarray,
                      bc_nodes: np.ndarray, bc_values: np.ndarray,
                      plane_stress: bool = True, element_type: str = "T3") -> np.ndarray:
    n_dof = node_xy.shape[0] * 2
    rows, cols, vals = assemble_global_stiffness(
        node_xy, element_node, E, nu, plane_stress, element_type)

    K = np.zeros((n_dof, n_dof), dtype=np.float64)
    for r, c, v in zip(rows, cols, vals):
        K[r, c] += v

    K_mod, F_mod = apply_dirichlet_bcs(K, F_ext, bc_nodes, bc_values)


    cond_est = np.linalg.cond(K_mod)
    if cond_est > 1e16:

        K_mod += np.eye(n_dof) * 1e-8 * np.max(np.abs(K_mod))

    U = np.linalg.solve(K_mod, F_mod)
    return U






def compute_element_stress(node_xy: np.ndarray, element_node: np.ndarray,
                            U: np.ndarray, E: float, nu: float,
                            plane_stress: bool = True, element_type: str = "T3") -> np.ndarray:
    n_elements = element_node.shape[0]
    stress = np.zeros((n_elements, 3), dtype=np.float64)
    D = elastic_d_matrix(E, nu, plane_stress)

    for e in range(n_elements):
        enodes = element_node[e, :]
        node_xy_e = node_xy[enodes, :]
        n_local = len(enodes)
        if element_type.upper() == "T3":
            dN_dxi = dshape_t3(0.0, 0.0)
        else:
            dN_dxi = dshape_t4(1.0/3.0, 1.0/3.0)
        J = dN_dxi.T @ node_xy_e
        detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if abs(detJ) < 1e-14:
            continue
        invJ = np.linalg.inv(J)
        dN_dx = dN_dxi @ invJ
        B = np.zeros((3, n_local * 2), dtype=np.float64)
        for i in range(n_local):
            B[0, 2*i]   = dN_dx[i, 0]
            B[1, 2*i+1] = dN_dx[i, 1]
            B[2, 2*i]   = dN_dx[i, 1]
            B[2, 2*i+1] = dN_dx[i, 0]
        edof = np.zeros(n_local * 2, dtype=np.int32)
        for i in range(n_local):
            edof[2*i]   = enodes[i] * 2
            edof[2*i+1] = enodes[i] * 2 + 1
        u_e = U[edof]
        eps = B @ u_e
        sigma = D @ eps
        stress[e, :] = sigma
    return stress






def generate_rectangular_mesh(lx: float, ly: float, nx: int, ny: int):
    x = np.linspace(0.0, lx, nx + 1)
    y = np.linspace(0.0, ly, ny + 1)
    X, Y = np.meshgrid(x, y)
    node_xy = np.column_stack([X.ravel(), Y.ravel()])

    n_nodes_row = nx + 1
    elements = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * n_nodes_row + i
            n1 = n0 + 1
            n2 = n0 + n_nodes_row
            n3 = n2 + 1

            elements.append([n0, n1, n2])

            elements.append([n1, n3, n2])
    element_node = np.array(elements, dtype=np.int32)
    return node_xy, element_node
