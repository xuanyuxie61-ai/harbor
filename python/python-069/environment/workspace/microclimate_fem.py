import numpy as np
from utils import bandwidth, basis_11_t6, reference_to_physical_t3, triangle_area_2d, banded_solve


def basis_11_t3(t3, i, p):
    x1, y1 = t3[0, 0], t3[1, 0]
    x2, y2 = t3[0, 1], t3[1, 1]
    x3, y3 = t3[0, 2], t3[1, 2]
    area2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    area2 = max(abs(area2), 1e-14) * np.sign(area2) if area2 != 0 else 1e-14

    if i == 0:
        bi = ((x2 - x3) * (p[1] - y3) - (y2 - y3) * (p[0] - x3)) / area2
        dbidx = (y2 - y3) / area2
        dbidy = (x3 - x2) / area2
    elif i == 1:
        bi = ((x3 - x1) * (p[1] - y1) - (y3 - y1) * (p[0] - x1)) / area2
        dbidx = (y3 - y1) / area2
        dbidy = (x1 - x3) / area2
    elif i == 2:
        bi = ((x1 - x2) * (p[1] - y2) - (y1 - y2) * (p[0] - x2)) / area2
        dbidx = (y1 - y2) / area2
        dbidy = (x2 - x1) / area2
    else:
        raise ValueError("T3 basis index i must be 0,1,2")
    return bi, dbidx, dbidy


def tetens_vapor_pressure(t_celsius):
    t = np.asarray(t_celsius, dtype=float)
    t = np.clip(t, -50.0, 60.0)
    arg = 17.27 * t / (t + 237.3)
    arg = np.clip(arg, -50.0, 50.0)
    return 0.6108 * np.exp(arg)


def latent_heat_flux(t_celsius, gs, ea, patm=101.325):
    es = tetens_vapor_pressure(t_celsius)
    lambda_v = 2.45e6
    mw = 18.015e-3
    E = gs * (es - ea) / patm
    return lambda_v * mw * E


def quad_rule_t3(n=7):
    if n == 3:
        w = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        p = np.array([[1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
                      [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0]])
        return w, p
    elif n == 7:
        a = 0.797426985353087
        b = 0.101286507323456
        c = 0.059715871789770
        d = 0.470142064105115
        w = np.array([0.225000000000000,
                      0.125939180544827, 0.125939180544827, 0.125939180544827,
                      0.132394152788506, 0.132394152788506, 0.132394152788506])
        p = np.array([[1.0 / 3.0, a, b, b, c, d, d],
                      [1.0 / 3.0, b, a, b, d, c, d]])
        return w, p
    else:

        return quad_rule_t3(3)


def assemble_microclimate_fem(node_xy, element_node, node_boundary, alpha,
                               q_rad, q_latent_func, dt):
    node_num = node_xy.shape[0]
    element_num = element_node.shape[1]
    element_order = element_node.shape[0]
    quad_num = 7
    ib = bandwidth(element_order, element_num, element_node)

    a = np.zeros((3 * ib + 1, node_num), dtype=float)
    f = np.zeros(node_num, dtype=float)
    quad_w, quad_xy = quad_rule_t3(quad_num)

    for element in range(element_num):
        t3 = node_xy[element_node[0:3, element], :].T
        if element_order == 6:
            t6 = node_xy[element_node[:, element], :].T
        phys_xy = reference_to_physical_t3(t3, quad_num, quad_xy)
        area = triangle_area_2d(t3)
        w = area * quad_w

        for quad in range(quad_num):
            xq, yq = phys_xy[0, quad], phys_xy[1, quad]

            qr = q_rad[xq, yq] if callable(q_rad) else float(q_rad)

            ql = q_latent_func(xq, yq)

            for test in range(element_order):
                i = element_node[test, element]
                if element_order == 3:
                    bi, dbidx, dbidy = basis_11_t3(t3, test, phys_xy[:, quad])
                else:
                    bi, dbidx, dbidy = basis_11_t6(t6, test, phys_xy[:, quad])
                f[i] += w[quad] * (qr - ql) * bi
                for basis in range(element_order):
                    j = element_node[basis, element]
                    if element_order == 3:
                        bj, dbjdx, dbjdy = basis_11_t3(t3, basis, phys_xy[:, quad])
                    else:
                        bj, dbjdx, dbjdy = basis_11_t6(t6, basis, phys_xy[:, quad])

                    mass = bi * bj
                    stiff = dbidx * dbjdx + dbidy * dbjdy
                    row = i - j + 2 * ib
                    a[row, j] += w[quad] * (mass + dt * alpha * stiff)
    return a, f, ib


def solve_microclimate(node_xy, element_node, node_boundary,
                       alpha_diff, dt, n_steps, t_initial, t_ambient,
                       q_rad_profile, gs_profile, ea, patm=101.325):
    node_num = node_xy.shape[0]
    T = np.full(node_num, t_initial, dtype=float)
    results = [T.copy()]

    def q_latent(x, y):

        return 0.0

    for step in range(n_steps):
        q_rad = q_rad_profile[step] if step < len(q_rad_profile) else q_rad_profile[-1]

        avg_T = np.mean(T)
        ql_val = latent_heat_flux(avg_T, np.mean(gs_profile), ea, patm)

        def ql_func(x, y):
            return ql_val

        a, rhs, ib = assemble_microclimate_fem(node_xy, element_node,
                                                node_boundary, alpha_diff,
                                                q_rad, ql_func, dt)

        for node in range(node_num):
            if node_boundary[node]:
                col_low = max(node - ib, 0)
                col_high = min(node + ib, node_num - 1)
                for col in range(col_low, col_high + 1):
                    a[node - col + 2 * ib, col] = 0.0
                a[2 * ib, node] = 1.0
                rhs[node] = t_ambient

        T_new = banded_solve(a, rhs, ib)
        T_new = np.clip(T_new, -20.0, 60.0)
        T = T_new
        results.append(T.copy())
    return results
