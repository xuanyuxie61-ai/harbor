
import numpy as np
from typing import Callable, Optional, Tuple, List






def tetrahedron_volume(verts: np.ndarray) -> float:
    if verts.shape != (4, 3):
        raise ValueError("verts必须为(4,3)数组")
    M = np.zeros((4, 4))
    M[:, :3] = verts
    M[:, 3] = 1.0
    vol = np.abs(np.linalg.det(M)) / 6.0
    return float(max(vol, 1e-15))


def compute_tet_quality(verts: np.ndarray) -> float:
    vol = tetrahedron_volume(verts)
    if vol < 1e-14:
        return 0.0


    edges = []
    for i in range(4):
        for j in range(i + 1, 4):
            edges.append(np.sum((verts[i] - verts[j]) ** 2))
    edges = np.array(edges)




    quality = 12.0 * (3.0 * vol ** 2) ** (1.0 / 3.0) / np.sum(edges)
    return float(np.clip(quality, 0.0, 1.0))


def regular_tetrahedral_mesh(
    bounds: Tuple[np.ndarray, np.ndarray],
    n_per_dim: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    xmin, xmax = bounds
    dims = 3


    grids = [np.linspace(xmin[d], xmax[d], n_per_dim) for d in range(dims)]
    node_list = []
    for i in range(n_per_dim):
        for j in range(n_per_dim):
            for k in range(n_per_dim):
                node_list.append([grids[0][i], grids[1][j], grids[2][k]])
    nodes = np.array(node_list)

    def node_index(i, j, k):
        return i * n_per_dim * n_per_dim + j * n_per_dim + k

    tets = []
    for i in range(n_per_dim - 1):
        for j in range(n_per_dim - 1):
            for k in range(n_per_dim - 1):

                n000 = node_index(i, j, k)
                n100 = node_index(i + 1, j, k)
                n010 = node_index(i, j + 1, k)
                n110 = node_index(i + 1, j + 1, k)
                n001 = node_index(i, j, k + 1)
                n101 = node_index(i + 1, j, k + 1)
                n011 = node_index(i, j + 1, k + 1)
                n111 = node_index(i + 1, j + 1, k + 1)


                tets.append([n000, n100, n110, n111])
                tets.append([n000, n100, n101, n111])
                tets.append([n000, n001, n101, n111])
                tets.append([n000, n001, n011, n111])
                tets.append([n000, n010, n110, n111])
                tets.append([n000, n010, n011, n111])

    tets = np.array(tets, dtype=int)
    return nodes, tets






def assemble_fem_matrices(
    nodes: np.ndarray,
    tets: np.ndarray,
    drift_fn: Callable[[np.ndarray], np.ndarray],
    diffusion_fn: Callable[[np.ndarray], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    n_nodes = nodes.shape[0]
    M = np.zeros((n_nodes, n_nodes))
    A = np.zeros((n_nodes, n_nodes))

    for tet in tets:
        verts = nodes[tet, :]
        vol = tetrahedron_volume(verts)
        if vol < 1e-14:
            continue



        D = np.zeros((3, 3))
        for d in range(3):
            D[:, d] = verts[d + 1, :] - verts[0, :]

        try:
            D_inv = np.linalg.inv(D)
        except np.linalg.LinAlgError:
            continue




        grads = np.zeros((4, 3))
        grads[0, :] = -np.sum(D_inv, axis=0)
        grads[1:4, :] = D_inv


        centroid = np.mean(verts, axis=0)
        f_cent = np.atleast_1d(drift_fn(centroid))
        sigma_cent = np.atleast_1d(diffusion_fn(centroid))
        if sigma_cent.ndim < 2:
            sigma_cent = np.diag(sigma_cent)
        eps_mat = 0.5 * (sigma_cent @ sigma_cent.T)
        eps_trace = np.trace(eps_mat)


        for i_local in range(4):
            for j_local in range(4):
                gi = grads[i_local, :]
                gj = grads[j_local, :]
                ii = tet[i_local]
                jj = tet[j_local]


                if i_local == j_local:
                    M[ii, jj] += vol / 4.0
                else:
                    M[ii, jj] += vol / 20.0


                stiff = eps_trace * vol * np.dot(gi, gj)



                conv = -vol / 4.0 * np.dot(f_cent, gi)
                if i_local == j_local:
                    conv *= 1.0

                A[ii, jj] += stiff + conv

    return M, A






def bicg_solver(
    A_mat: np.ndarray,
    b_vec: np.ndarray,
    x0: Optional[np.ndarray] = None,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, float, int, int]:
    n = len(b_vec)
    if x0 is None:
        x = np.zeros(n)
    else:
        x = np.array(x0, dtype=float)

    bnrm = np.linalg.norm(b_vec)
    if bnrm == 0.0:
        bnrm = 1.0

    r = b_vec - A_mat @ x
    error = np.linalg.norm(r) / bnrm
    if error < tol:
        return x, error, 0, 0

    r_tld = r.copy()
    rho_old = 1.0
    p = np.zeros(n)
    p_tld = np.zeros(n)

    flag = 0

    for it in range(1, max_iter + 1):
        z = r.copy()
        z_tld = r_tld.copy()
        rho = np.dot(z, r_tld)

        if abs(rho) < 1e-30:
            flag = -1
            break

        if it == 1:
            p = z.copy()
            p_tld = z_tld.copy()
        else:
            beta = rho / rho_old
            p = z + beta * p
            p_tld = z_tld + beta * p_tld

        q = A_mat @ p
        q_tld = A_mat.T @ p_tld
        denom = np.dot(p_tld, q)

        if abs(denom) < 1e-30:
            flag = -1
            break

        alpha = rho / denom
        x = x + alpha * p
        r = r - alpha * q
        r_tld = r_tld - alpha * q_tld

        error = np.linalg.norm(r) / bnrm
        if error <= tol:
            flag = 0
            break

        rho_old = rho
    else:
        flag = 1

    return x, error, it, flag






def solve_hjb_backward(
    nodes: np.ndarray,
    tets: np.ndarray,
    drift_fn: Callable[[np.ndarray], np.ndarray],
    diffusion_fn: Callable[[np.ndarray], np.ndarray],
    terminal_cost_fn: Callable[[np.ndarray], np.ndarray],
    running_cost_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    tspan: Tuple[float, float],
    n_time: int,
    control_candidates: Optional[List[np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    t0, tstop = tspan
    dt = (tstop - t0) / n_time
    t_grid = np.linspace(t0, tstop, n_time + 1)

    n_nodes = nodes.shape[0]


    M, A_base = assemble_fem_matrices(nodes, tets, drift_fn, diffusion_fn)


    V_next = terminal_cost_fn(nodes)
    if len(V_next) != n_nodes:
        raise ValueError("终端代价函数输出维度与节点数不匹配")

    V_history = np.zeros((n_time + 1, n_nodes))
    V_history[-1, :] = V_next


    if control_candidates is None:
        control_candidates = [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([-1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, -1.0, 0.0]),
        ]

    I_mat = np.eye(n_nodes)

    for n in range(n_time - 1, -1, -1):

        rhs = M @ V_next


        best_cost = np.full(n_nodes, np.inf)
        for u_cand in control_candidates:
            cost = running_cost_fn(nodes, u_cand)
            best_cost = np.minimum(best_cost, cost)

        rhs += dt * best_cost


        LHS = M + dt * A_base

        LHS += 1e-10 * I_mat


        V_current, err, iters, flag = bicg_solver(LHS, rhs, x0=V_next, max_iter=2000, tol=1e-10)

        if flag != 0:

            try:
                V_current = np.linalg.solve(LHS, rhs)
            except np.linalg.LinAlgError:
                V_current = V_next.copy()

        V_history[n, :] = V_current
        V_next = V_current.copy()

    return V_history, t_grid
