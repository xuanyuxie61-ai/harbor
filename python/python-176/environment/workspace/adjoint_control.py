
import numpy as np


def solve_state_forward(nodes, elements, boundary_nodes, boundary_edges,
                        M, A, B, y0, q_seq, f_fn, nu, c, T, n_time):
    n_nodes = nodes.shape[0]
    n_bnd = len(boundary_nodes)
    dt = T / n_time
    y_seq = np.zeros((n_time + 1, n_nodes), dtype=float)
    y_seq[0] = y0.copy()


    bnd_map = {int(b): i for i, b in enumerate(boundary_nodes)}

    for n in range(n_time):
        t_np1 = (n + 1) * dt

        F = np.zeros(n_nodes, dtype=float)
        for e in elements:
            i, j, k = e
            p1, p2, p3 = nodes[i], nodes[j], nodes[k]
            area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
            if area < 1.0e-15:
                continue
            xc = (p1[0] + p2[0] + p3[0]) / 3.0
            yc = (p1[1] + p2[1] + p3[1]) / 3.0
            fv = f_fn(xc, yc, t_np1)
            F[i] += area * fv / 3.0
            F[j] += area * fv / 3.0
            F[k] += area * fv / 3.0


        q_full = np.zeros(n_nodes, dtype=float)
        for b_idx, glob_idx in enumerate(boundary_nodes):
            q_full[glob_idx] = q_seq[n + 1, b_idx]
        control_term = B @ q_full







        raise NotImplementedError("Hole_2: 请实现状态方程的隐式 Euler 时间步进")


def solve_adjoint_backward(nodes, elements, boundary_nodes, boundary_edges,
                           M, A, B, y_seq, yd_seq, nu, c, T, n_time):
    n_nodes = nodes.shape[0]
    dt = T / n_time
    p_seq = np.zeros((n_time + 1, n_nodes), dtype=float)
    p_seq[-1] = 0.0

    for n in range(n_time - 1, -1, -1):






        raise NotImplementedError("Hole_3: 请实现伴随方程的后向隐式 Euler 时间步进")


def compute_objective(nodes, elements, boundary_nodes, boundary_edges,
                      M, B, L_bd, y_seq, yd_seq, q_seq, alpha, beta, T, n_time):
    n_nodes = nodes.shape[0]
    dt = T / n_time
    J = 0.0

    for n in range(n_time + 1):
        dy = y_seq[n] - yd_seq[n]

        J += 0.5 * dt * np.dot(dy, M @ dy)


        q_full = np.zeros(n_nodes, dtype=float)
        for b_idx, glob_idx in enumerate(boundary_nodes):
            q_full[glob_idx] = q_seq[n, b_idx]
        J += 0.5 * alpha * dt * np.dot(q_full, B @ q_full)


        if beta > 0.0 and L_bd is not None:
            J += 0.5 * beta * dt * np.dot(q_seq[n], L_bd @ q_seq[n])

    return J


def compute_gradient(nodes, elements, boundary_nodes, boundary_edges,
                     M, B, L_bd, p_seq, q_seq, alpha, beta, T, n_time):
    n_bnd = len(boundary_nodes)
    dt = T / n_time
    grad = np.zeros((n_time + 1, n_bnd), dtype=float)

    for n in range(n_time + 1):
        q_full = np.zeros(nodes.shape[0], dtype=float)
        for b_idx, glob_idx in enumerate(boundary_nodes):
            q_full[glob_idx] = q_seq[n, b_idx]


        g_full = alpha * (B @ q_full) + (B @ p_seq[n])


        for b_idx, glob_idx in enumerate(boundary_nodes):
            grad[n, b_idx] = g_full[glob_idx]

        if beta > 0.0 and L_bd is not None:
            grad[n] += beta * (L_bd @ q_seq[n])

    return grad


def build_boundary_laplacian_1d(boundary_nodes, nodes):
    n_bnd = len(boundary_nodes)
    L = np.zeros((n_bnd, n_bnd), dtype=float)

    if n_bnd < 2:
        return L


    coords = nodes[boundary_nodes]


    angles = np.arctan2(coords[:, 1], coords[:, 0])
    order = np.argsort(angles)
    sorted_indices = [boundary_nodes[o] for o in order]


    for i in range(n_bnd):
        i1 = order[i]
        i2 = order[(i + 1) % n_bnd]
        p1 = coords[i1]
        p2 = coords[i2]
        h = np.linalg.norm(p2 - p1)
        if h < 1.0e-15:
            h = 1.0
        inv_h = 1.0 / h
        L[i1, i1] += inv_h
        L[i1, i2] -= inv_h
        L[i2, i1] -= inv_h
        L[i2, i2] += inv_h

    return L


def armijo_line_search(nodes, elements, boundary_nodes, boundary_edges,
                       M, A, B, L_bd, y0, yd_seq, f_fn, grad, q_seq,
                       alpha, beta, nu, c, T, n_time,
                       eta_init=1.0, c_armijo=1.0e-4, rho=0.5, max_iter=10):
    J0 = compute_objective(nodes, elements, boundary_nodes, boundary_edges,
                           M, B, L_bd, solve_state_forward(nodes, elements, boundary_nodes, boundary_edges,
                                                           M, A, B, y0, q_seq, f_fn, nu, c, T, n_time),
                           yd_seq, q_seq, alpha, beta, T, n_time)
    grad_norm2 = np.sum(grad ** 2)

    eta = eta_init
    for _ in range(max_iter):
        q_new = q_seq - eta * grad

        q_new = np.clip(q_new, -100.0, 100.0)
        y_new = solve_state_forward(nodes, elements, boundary_nodes, boundary_edges,
                                    M, A, B, y0, q_new, f_fn, nu, c, T, n_time)
        J_new = compute_objective(nodes, elements, boundary_nodes, boundary_edges,
                                  M, B, L_bd, y_new, yd_seq, q_new, alpha, beta, T, n_time)
        if J_new <= J0 - c_armijo * eta * grad_norm2:
            return eta, q_new, y_new, J_new
        eta *= rho

    return eta, q_seq, solve_state_forward(nodes, elements, boundary_nodes, boundary_edges,
                                           M, A, B, y0, q_seq, f_fn, nu, c, T, n_time), J0


def optimize_control(nodes, elements, boundary_nodes, boundary_edges,
                     M, A, B, y0, yd_seq, f_fn,
                     alpha=1.0e-3, beta=1.0e-5, nu=0.1, c=1.0,
                     T=1.0, n_time=20, max_iter=30, tol=1.0e-6):
    n_bnd = len(boundary_nodes)
    L_bd = build_boundary_laplacian_1d(boundary_nodes, nodes)


    q_seq = np.zeros((n_time + 1, n_bnd), dtype=float)

    history = []
    for k in range(max_iter):

        y_seq = solve_state_forward(nodes, elements, boundary_nodes, boundary_edges,
                                    M, A, B, y0, q_seq, f_fn, nu, c, T, n_time)


        p_seq = solve_adjoint_backward(nodes, elements, boundary_nodes, boundary_edges,
                                       M, A, B, y_seq, yd_seq, nu, c, T, n_time)


        grad = compute_gradient(nodes, elements, boundary_nodes, boundary_edges,
                                M, B, L_bd, p_seq, q_seq, alpha, beta, T, n_time)


        J_val = compute_objective(nodes, elements, boundary_nodes, boundary_edges,
                                  M, B, L_bd, y_seq, yd_seq, q_seq, alpha, beta, T, n_time)
        history.append(J_val)

        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            print(f"  优化收敛于迭代 {k}, 梯度范数 = {grad_norm:.6e}, J = {J_val:.6e}")
            break


        eta, q_seq, y_seq, J_new = armijo_line_search(
            nodes, elements, boundary_nodes, boundary_edges,
            M, A, B, L_bd, y0, yd_seq, f_fn, grad, q_seq,
            alpha, beta, nu, c, T, n_time
        )

        if k % 5 == 0:
            print(f"  迭代 {k}: J = {J_val:.6e}, ‖∇J‖ = {grad_norm:.6e}, η = {eta:.4e}")


    y_seq = solve_state_forward(nodes, elements, boundary_nodes, boundary_edges,
                                M, A, B, y0, q_seq, f_fn, nu, c, T, n_time)
    p_seq = solve_adjoint_backward(nodes, elements, boundary_nodes, boundary_edges,
                                   M, A, B, y_seq, yd_seq, nu, c, T, n_time)

    return q_seq, y_seq, p_seq, history
