
import numpy as np
from mesh_transform import polygon_surface_quality


def build_rectangular_mesh(nx, ny, xl=0.0, xr=1.0, yb=0.0, yt=1.0):
    node_num = nx * ny
    node_xy = np.zeros((2, node_num), dtype=float)
    k = 0
    for j in range(1, ny + 1):
        for i in range(1, nx + 1):
            node_xy[0, k] = ((nx - i) * xl + (i - 1) * xr) / (nx - 1)
            node_xy[1, k] = ((ny - j) * yb + (j - 1) * yt) / (ny - 1)
            k += 1

    element_num = 2 * (nx - 1) * (ny - 1)



    element_node = np.zeros((3, element_num), dtype=int)
    raise NotImplementedError("Hole_2: build_rectangular_mesh 的 element_node 索引生成待实现")

    return node_xy, element_node


def fem2d_poisson_solve(nx, ny, source_func, exact_func=None,
                        xl=0.0, xr=1.0, yb=0.0, yt=1.0,
                        conductivity=1.0):
    node_xy, element_node = build_rectangular_mesh(nx, ny, xl, xr, yb, yt)
    node_num = node_xy.shape[1]
    element_num = element_node.shape[1]


    A = np.zeros((node_num, node_num), dtype=float)
    F = np.zeros(node_num, dtype=float)








    raise NotImplementedError("Hole_1: FEM 刚度矩阵组装循环待实现")


    if exact_func is not None:
        k = 0
        for j in range(1, ny + 1):
            for i in range(1, nx + 1):
                if i == 1 or i == nx or j == 1 or j == ny:
                    u_bc, _, _ = exact_func(node_xy[0, k], node_xy[1, k])
                    A[k, :] = 0.0
                    A[k, k] = 1.0
                    F[k] = u_bc
                k += 1


    u = np.linalg.solve(A, F)


    el2 = 0.0
    eh1 = 0.0
    if exact_func is not None:
        for e in range(element_num):
            i1, i2, i3 = element_node[:, e] - 1
            x1, y1 = node_xy[:, i1]
            x2, y2 = node_xy[:, i2]
            x3, y3 = node_xy[:, i3]
            area = 0.5 * abs(
                x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
            )
            if area < 1e-14:
                continue

            for q1 in range(3):
                q2 = (q1 + 1) % 3
                nq1 = element_node[q1, e] - 1
                nq2 = element_node[q2, e] - 1
                xq = 0.5 * (node_xy[0, nq1] + node_xy[0, nq2])
                yq = 0.5 * (node_xy[1, nq1] + node_xy[1, nq2])
                wq = 1.0 / 3.0

                uh = 0.0
                dudxh = 0.0
                dudyh = 0.0
                for tj1 in range(3):
                    tj2 = (tj1 + 1) % 3
                    tj3 = (tj1 + 2) % 3
                    ntj1 = element_node[tj1, e] - 1
                    ntj2 = element_node[tj2, e] - 1
                    ntj3 = element_node[tj3, e] - 1

                    qj = 0.5 * (
                        (node_xy[0, ntj3] - node_xy[0, ntj2]) * (yq - node_xy[1, ntj2])
                        - (node_xy[1, ntj3] - node_xy[1, ntj2]) * (xq - node_xy[0, ntj2])
                    ) / area
                    dqjdx = -0.5 * (node_xy[1, ntj3] - node_xy[1, ntj2]) / area
                    dqjdy = 0.5 * (node_xy[0, ntj3] - node_xy[0, ntj2]) / area

                    uh += u[ntj1] * qj
                    dudxh += u[ntj1] * dqjdx
                    dudyh += u[ntj1] * dqjdy

                u_ex, dudx_ex, dudy_ex = exact_func(xq, yq)
                el2 += (uh - u_ex) ** 2 * area
                eh1 += ((dudxh - dudx_ex) ** 2 + (dudyh - dudy_ex) ** 2) * area

        el2 = np.sqrt(el2)
        eh1 = np.sqrt(eh1)

    return u, node_xy, element_node, float(el2), float(eh1)


def extract_gradient_at_nodes(u, node_xy, element_node):
    node_num = node_xy.shape[1]
    grad = np.zeros((2, node_num), dtype=float)
    weight = np.zeros(node_num, dtype=float)
    element_num = element_node.shape[1]

    for e in range(element_num):
        i1, i2, i3 = element_node[:, e] - 1
        x1, y1 = node_xy[:, i1]
        x2, y2 = node_xy[:, i2]
        x3, y3 = node_xy[:, i3]
        area = 0.5 * abs(
            x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
        )
        if area < 1e-14:
            continue


        dudx = (u[i1] * (y2 - y3) + u[i2] * (y3 - y1) + u[i3] * (y1 - y2)) / (2.0 * area)
        dudy = (u[i1] * (x3 - x2) + u[i2] * (x1 - x3) + u[i3] * (x2 - x1)) / (2.0 * area)

        for idx in [i1, i2, i3]:
            grad[0, idx] += dudx * area
            grad[1, idx] += dudy * area
            weight[idx] += area

    mask = weight > 0
    grad[:, mask] = grad[:, mask] / weight[mask]
    return grad
