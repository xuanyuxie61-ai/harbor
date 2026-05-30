
import numpy as np


def chebyshev_nodes(a, b, n):
    if n <= 0:
        raise ValueError("chebyshev_nodes: n 必须为正整数")
    if n == 1:
        return np.array([(a + b) / 2.0])

    k = np.arange(n)
    theta = (2 * k + 1) * np.pi / (2 * n)
    x_std = np.cos(theta)
    x = 0.5 * (a + b) + 0.5 * (b - a) * x_std
    return x


def divided_differences(x, y):
    n = len(x)
    dd = y.copy()
    for i in range(1, n):
        for j in range(n - 1, i - 1, -1):
            dd[j] = (dd[j] - dd[j - 1]) / (x[j] - x[j - i])
    return dd


def newton_interpolation(x_nodes, dd, x_eval):
    n = len(dd)
    y_eval = dd[-1] * np.ones_like(x_eval)
    for i in range(n - 2, -1, -1):
        y_eval = dd[i] + (x_eval - x_nodes[i]) * y_eval
    return y_eval


def chebyshev_approximate_1d(f, a, b, n, n_eval=1001):
    x_cheb = chebyshev_nodes(a, b, n)
    y_cheb = f(x_cheb)
    dd = divided_differences(x_cheb, y_cheb)


    x_eval = np.linspace(a, b, n_eval)
    y_interp = newton_interpolation(x_cheb, dd, x_eval)
    y_exact = f(x_eval)

    max_error = np.max(np.abs(y_interp - y_exact))
    return max_error, dd, x_cheb


def compute_element_error_indicator(
    nodes, triangles, solution, element_idx,
    cheb_degree=4
):
    tri = triangles[element_idx]
    p1, p2, p3 = nodes[tri[0]], nodes[tri[1]], nodes[tri[2]]
    u1, u2, u3 = solution[tri[0]], solution[tri[1]], solution[tri[2]]


    area = 0.5 * abs(
        (p2[0] - p1[0]) * (p3[1] - p1[1]) -
        (p3[0] - p1[0]) * (p2[1] - p1[1])
    )


    e1 = np.linalg.norm(p2 - p1)
    e2 = np.linalg.norm(p3 - p2)
    e3 = np.linalg.norm(p1 - p3)
    h_T = max(e1, e2, e3)

    if area < 1e-14 or h_T < 1e-14:
        return 0.0


    edges = [(p1, p2, u1, u2), (p2, p3, u2, u3), (p3, p1, u3, u1)]
    edge_errors = []

    for pa, pb, ua, ub in edges:
        edge_len = np.linalg.norm(pb - pa)
        if edge_len < 1e-14:
            continue


        def f_edge(t):
            return ua + t * (ub - ua)


        t_cheb = chebyshev_nodes(0.0, 1.0, cheb_degree)
        y_cheb = f_edge(t_cheb)
        dd = divided_differences(t_cheb, y_cheb)


        t_eval = np.linspace(0, 1, 51)
        y_linear = f_edge(t_eval)
        y_cheb_interp = newton_interpolation(t_cheb, dd, t_eval)

        edge_error = np.max(np.abs(y_linear - y_cheb_interp))
        edge_errors.append(edge_error)


    if len(edge_errors) > 0:
        eta = h_T * max(edge_errors)
    else:
        eta = 0.0


    grad_est = max(abs(u2 - u1), abs(u3 - u2), abs(u1 - u3)) / h_T
    eta += h_T ** 2 * grad_est

    return eta


def compute_all_error_indicators(nodes, triangles, solution, cheb_degree=4):
    n_tri = len(triangles)
    errors = np.zeros(n_tri)

    for t in range(n_tri):
        errors[t] = compute_element_error_indicator(
            nodes, triangles, solution, t, cheb_degree
        )

    total_error = np.sqrt(np.sum(errors ** 2))
    return errors, total_error


def compute_gradient_recovery_error(nodes, triangles, solution):
    n_nodes = len(nodes)
    n_tri = len(triangles)


    elem_grads = np.zeros((n_tri, 2))
    elem_areas = np.zeros(n_tri)

    for t in range(n_tri):
        tri = triangles[t]
        p1, p2, p3 = nodes[tri[0]], nodes[tri[1]], nodes[tri[2]]
        u1, u2, u3 = solution[tri[0]], solution[tri[1]], solution[tri[2]]

        area = 0.5 * (
            (p2[0] - p1[0]) * (p3[1] - p1[1]) -
            (p3[0] - p1[0]) * (p2[1] - p1[1])
        )
        elem_areas[t] = abs(area)

        if abs(area) > 1e-14:

            dx = np.array([p2[0] - p1[0], p3[0] - p1[0]])
            dy = np.array([p2[1] - p1[1], p3[1] - p1[1]])
            du = np.array([u2 - u1, u3 - u1])


            A_mat = np.array([[dx[0], dx[1]], [dy[0], dy[1]]])
            if abs(np.linalg.det(A_mat)) > 1e-14:
                grad = np.linalg.solve(A_mat.T, du)
                elem_grads[t] = grad


    node_grads = np.zeros((n_nodes, 2))
    node_areas = np.zeros(n_nodes)

    for t in range(n_tri):
        tri = triangles[t]
        for i in range(3):
            node_grads[tri[i]] += elem_grads[t] * elem_areas[t]
            node_areas[tri[i]] += elem_areas[t]

    for i in range(n_nodes):
        if node_areas[i] > 1e-14:
            node_grads[i] /= node_areas[i]


    errors = np.zeros(n_tri)
    for t in range(n_tri):
        tri = triangles[t]
        recovered_grad = np.mean(node_grads[tri], axis=0)
        diff = elem_grads[t] - recovered_grad
        errors[t] = np.sqrt(np.sum(diff ** 2)) * np.sqrt(elem_areas[t])

    total_error = np.sqrt(np.sum(errors ** 2))
    return errors, total_error
