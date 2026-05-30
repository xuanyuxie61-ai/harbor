
import numpy as np
from typing import Tuple, Optional






def sparse_mv(rows: np.ndarray, cols: np.ndarray, vals: np.ndarray,
              x: np.ndarray, n: int) -> np.ndarray:
    y = np.zeros(n, dtype=np.float64)
    for r, c, v in zip(rows, cols, vals):
        y[r] += v * x[c]
    return y


def sparse_sym_cg(rows: np.ndarray, cols: np.ndarray, vals: np.ndarray,
                   b: np.ndarray, n: int, tol: float = 1e-10,
                   max_iter: Optional[int] = None) -> np.ndarray:
    if max_iter is None:
        max_iter = n
    x = np.zeros(n, dtype=np.float64)
    r = b - sparse_mv(rows, cols, vals, x, n)
    p = r.copy()
    rs_old = np.dot(r, r)
    norm_b = np.linalg.norm(b)
    if norm_b < 1e-14:
        norm_b = 1.0

    for _ in range(max_iter):
        Ap = sparse_mv(rows, cols, vals, p, n)
        alpha = rs_old / (np.dot(p, Ap) + 1e-20)
        x += alpha * p
        r -= alpha * Ap
        rs_new = np.dot(r, r)
        if np.sqrt(rs_new) / norm_b < tol:
            break
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new
    return x






def density_filter(element_centers: np.ndarray, rho: np.ndarray,
                   r_min: float) -> np.ndarray:
    n_elements = len(rho)

    dx = element_centers[:, 0:1] - element_centers[:, 0].reshape(1, -1)
    dy = element_centers[:, 1:2] - element_centers[:, 1].reshape(1, -1)
    dist = np.sqrt(dx**2 + dy**2)
    H = np.maximum(0.0, r_min - dist)
    sum_H = np.sum(H, axis=1)
    rho_tilde = np.where(sum_H > 1e-14, (H @ rho) / sum_H, rho)
    return rho_tilde


def heaviside_projection(rho_tilde: np.ndarray, beta: float,
                         eta: float = 0.5) -> np.ndarray:
    num = np.tanh(beta * eta) + np.tanh(beta * (rho_tilde - eta))
    den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    return num / den






def simp_interpolation(rho: np.ndarray, E0: float, E_min: float = 1e-9,
                        p: float = 3.0) -> np.ndarray:
    return E_min + np.power(rho, p) * (E0 - E_min)


def simp_derivative(rho: np.ndarray, E0: float, E_min: float = 1e-9,
                     p: float = 3.0) -> np.ndarray:
    return p * np.power(rho, p - 1.0) * (E0 - E_min)






def compute_compliance_sensitivity(element_node: np.ndarray, node_xy: np.ndarray,
                                    U: np.ndarray, E_e: np.ndarray,
                                    dE_drho: np.ndarray, nu: float,
                                    plane_stress: bool = True) -> np.ndarray:
    n_elements = element_node.shape[0]
    sens = np.zeros(n_elements, dtype=np.float64)
    from fem_core import elastic_d_matrix
    D0 = elastic_d_matrix(1.0, nu, plane_stress)




    raise NotImplementedError("Hole 2: compute_compliance_sensitivity 未实现")
    return sens






def oc_update(rho: np.ndarray, sens: np.ndarray, volfrac: float,
              move: float = 0.2, eta_oc: float = 0.5) -> np.ndarray:
    n_elements = len(rho)
    rho_min = 1e-3
    rho_new = np.zeros_like(rho)

    l1, l2 = 0.0, 1e6
    while (l2 - l1) / (l2 + l1 + 1e-10) > 1e-4:
        lmid = 0.5 * (l1 + l2)


        Be = np.zeros(n_elements, dtype=np.float64)
        for e in range(n_elements):
            if abs(sens[e]) < 1e-20:
                Be[e] = 1.0
            else:
                Be[e] = (-sens[e]) / lmid

                if Be[e] <= 0:
                    Be[e] = rho_min

        for e in range(n_elements):

            factor = Be[e] ** eta_oc
            rnew = rho[e] * factor

            rnew = max(rho_min, max(rho[e] - move, min(1.0, min(rho[e] + move, rnew))))
            rho_new[e] = rnew

        if np.mean(rho_new) > volfrac:
            l1 = lmid
        else:
            l2 = lmid
    return rho_new






def simp_topology_optimization(node_xy: np.ndarray, element_node: np.ndarray,
                                F_ext: np.ndarray, bc_nodes: np.ndarray,
                                bc_values: np.ndarray, E0: float, nu: float,
                                volfrac: float, n_iter: int = 100,
                                r_min: float = 1.5, plane_stress: bool = True,
                                use_filter: bool = True,
                                use_projection: bool = False) -> Tuple[np.ndarray, np.ndarray, list, list]:
    n_elements = element_node.shape[0]
    n_dof = node_xy.shape[0] * 2


    rho = np.full(n_elements, volfrac, dtype=np.float64)
    rho_min = 1e-3
    rho = np.maximum(rho, rho_min)


    element_centers = np.zeros((n_elements, 2), dtype=np.float64)
    for e in range(n_elements):
        enodes = element_node[e, :]
        element_centers[e] = np.mean(node_xy[enodes, :], axis=0)

    history_compliance = []
    history_vol = []

    for it in range(n_iter):

        if use_filter:
            rho_f = density_filter(element_centers, rho, r_min)
        else:
            rho_f = rho.copy()
        if use_projection:
            beta_proj = min(64.0, 1.0 + 0.5 * it)
            rho_f = heaviside_projection(rho_f, beta_proj)


        E_e = simp_interpolation(rho_f, E0)
        dE = simp_derivative(rho_f, E0)


        from fem_core import assemble_global_stiffness
        rows, cols, vals = assemble_global_stiffness(
            node_xy, element_node, 1.0, nu, plane_stress, "T3")

        vals_scaled = np.zeros_like(vals)

        idx = 0
        n_local = 3
        n_edof = n_local * 2
        for e in range(n_elements):
            scale = E_e[e]
            for _ in range(n_edof * n_edof):
                vals_scaled[idx] = vals[idx] * scale
                idx += 1


        K = np.zeros((n_dof, n_dof), dtype=np.float64)
        for r, c, v in zip(rows, cols, vals_scaled):
            K[r, c] += v

        K_mod, F_mod = apply_dirichlet_bcs(K, F_ext, bc_nodes, bc_values)
        cond_est = np.linalg.cond(K_mod)
        if cond_est > 1e16:
            K_mod += np.eye(n_dof) * 1e-8 * np.max(np.abs(K_mod))


        U = np.linalg.solve(K_mod, F_mod)


        compliance = np.dot(F_ext, U)
        history_compliance.append(compliance)
        history_vol.append(np.mean(rho_f))


        sens = compute_compliance_sensitivity(
            element_node, node_xy, U, E_e, dE, nu, plane_stress)


        if use_filter:
            dx = element_centers[:, 0:1] - element_centers[:, 0].reshape(1, -1)
            dy = element_centers[:, 1:2] - element_centers[:, 1].reshape(1, -1)
            dist = np.sqrt(dx**2 + dy**2)
            H = np.maximum(0.0, r_min - dist)
            sum_H = np.sum(H, axis=1)
            sum_H_safe = np.maximum(1e-14, sum_H)


            sens_f = H.T @ (sens / sum_H_safe)
            sens = sens_f


        rho = oc_update(rho, sens, volfrac)

    return rho, U, history_compliance, history_vol


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
