
import numpy as np


def shepard_interp_nd(m, xd, zd, p, xi):
    nd = xd.shape[0]
    ni = xi.shape[0]

    if xd.shape[1] != m or xi.shape[1] != m:
        raise ValueError("shepard_interp_nd: 维度不匹配")

    zi = np.zeros(ni)

    for i in range(ni):
        if p == 0.0:
            w = np.ones(nd) / nd
        else:
            w = np.zeros(nd)
            exact_match = -1

            for j in range(nd):
                dist = np.linalg.norm(xi[i] - xd[j])
                if dist < 1e-14:
                    exact_match = j
                    break
                w[j] = dist

            if exact_match >= 0:
                w = np.zeros(nd)
                w[exact_match] = 1.0
            else:
                w = 1.0 / (w ** p)
                s = np.sum(w)
                if s > 1e-14:
                    w = w / s
                else:
                    w = np.ones(nd) / nd

        zi[i] = np.dot(w, zd)

    return zi


def prolongation_shepard(coarse_nodes, coarse_solution, fine_nodes, p=2.0):
    if len(coarse_nodes) == 0 or len(fine_nodes) == 0:
        raise ValueError("prolongation_shepard: 空网格")

    m = coarse_nodes.shape[1]
    fine_solution = shepard_interp_nd(m, coarse_nodes, coarse_solution, p, fine_nodes)
    return fine_solution


def restriction_integral(coarse_nodes, coarse_triangles, fine_nodes, fine_solution):
    n_coarse = len(coarse_nodes)
    coarse_solution = np.zeros(n_coarse)

    for i in range(n_coarse):

        dists = np.linalg.norm(fine_nodes - coarse_nodes[i], axis=1)

        sigma = np.mean(dists[dists > 1e-14]) if np.any(dists > 1e-14) else 1.0
        weights = np.exp(-(dists ** 2) / (2 * sigma ** 2))
        weights_sum = np.sum(weights)
        if weights_sum > 1e-14:
            coarse_solution[i] = np.sum(weights * fine_solution) / weights_sum
        else:
            coarse_solution[i] = fine_solution[np.argmin(dists)]

    return coarse_solution


def solution_transfer_between_meshes(
    old_nodes, old_triangles, old_solution,
    new_nodes, new_triangles, transfer_type='shepard'
):
    n_new = len(new_nodes)

    if transfer_type == 'shepard':
        new_solution = prolongation_shepard(old_nodes, old_solution, new_nodes, p=2.0)
    elif transfer_type == 'integral':
        new_solution = restriction_integral(old_nodes, old_triangles, new_nodes, old_solution)

        if len(new_nodes) > len(old_nodes):
            new_solution = prolongation_shepard(old_nodes, old_solution, new_nodes, p=2.0)
    else:
        raise ValueError(f"solution_transfer_between_meshes: 未知的 transfer_type={transfer_type}")

    return new_solution
