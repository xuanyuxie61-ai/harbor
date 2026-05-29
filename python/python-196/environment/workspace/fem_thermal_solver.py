"""
fem_thermal_solver.py
2D热-电耦合有限元求解器

包含：
- 矩形区域上线性三角形单元的FEM组装与求解（源自 fem2d_poisson_rectangle_linear）
- 多边形表面网格节点/单元拓扑处理（源自 polygonal_surface_display）
- L2与H1误差估计

科学背景：
芯片热管理中的稳态热传导方程（Poisson方程）：
    -k * nabla^2 T(x,y) = Q(x,y),   (x,y) in Omega
    T(x,y) = T_b(x,y),              (x,y) on partial Omega

其中:
    k: 导热系数 [W/(m·K)]
    T: 温度场 [K]
    Q: 体积热源密度 [W/m^3]

FEM弱形式:
    寻找 T_h in V_h 使得
    k * integral_Omega grad(T_h) . grad(v_h) dOmega = integral_Omega Q v_h dOmega,  forall v_h in V_h^0

离散后得到线性系统: A T = F

矩阵元:
    A_{ij} = sum_e area_e * (dphi_i/dx * dphi_j/dx + dphi_i/dy * dphi_j/dy)
    F_i    = sum_e area_e * w_q * Q(x_q,y_q) * phi_i(x_q,y_q)
"""

import numpy as np
from mesh_transform import polygon_surface_quality


def build_rectangular_mesh(nx, ny, xl=0.0, xr=1.0, yb=0.0, yt=1.0):
    """
    构建矩形区域上的均匀节点和三角形单元网格。
    源自 fem2d_poisson_rectangle_linear 的网格生成部分。

    节点编号（从1开始）:
        K = i + (j-1)*nx,  i=1..nx, j=1..ny

    单元编号（从1开始）:
        每个矩形划分为2个三角形:
          左下三角: (i,j) -> (i+1,j) -> (i,j+1)
          右上三角: (i+1,j+1) -> (i,j+1) -> (i+1,j)
    """
    node_num = nx * ny
    node_xy = np.zeros((2, node_num), dtype=float)
    k = 0
    for j in range(1, ny + 1):
        for i in range(1, nx + 1):
            node_xy[0, k] = ((nx - i) * xl + (i - 1) * xr) / (nx - 1)
            node_xy[1, k] = ((ny - j) * yb + (j - 1) * yt) / (ny - 1)
            k += 1

    element_num = 2 * (nx - 1) * (ny - 1)
    # TODO Hole_2: 完成矩形网格的三角形单元节点索引生成。
    # 注意：需要确定使用 0-based 还是 1-based 索引，并确保与 fem2d_poisson_solve 和
    # mesh_transform.py 中的索引处理保持一致。
    element_node = np.zeros((3, element_num), dtype=int)
    raise NotImplementedError("Hole_2: build_rectangular_mesh 的 element_node 索引生成待实现")

    return node_xy, element_node


def fem2d_poisson_solve(nx, ny, source_func, exact_func=None,
                        xl=0.0, xr=1.0, yb=0.0, yt=1.0,
                        conductivity=1.0):
    """
    使用线性三角形单元求解2D Poisson方程。
    源自 fem2d_poisson_rectangle_linear。

    方程: -conductivity * nabla^2 u = source_func(x,y)
    边界: u = exact_func(x,y) on boundary

    参数:
        nx, ny: int, 网格点数
        source_func: callable, (x,y) -> float
        exact_func: callable, (x,y) -> (u, dudx, dudy)
        conductivity: float, 导热系数

    返回:
        u: ndarray, shape (node_num,), 数值解
        node_xy: ndarray, 节点坐标
        element_node: ndarray, 单元拓扑
        el2: float, L2误差
        eh1: float, H1半范数误差
    """
    node_xy, element_node = build_rectangular_mesh(nx, ny, xl, xr, yb, yt)
    node_num = node_xy.shape[1]
    element_num = element_node.shape[1]

    # 组装
    A = np.zeros((node_num, node_num), dtype=float)
    F = np.zeros(node_num, dtype=float)

    # TODO Hole_1: 完成 FEM 刚度矩阵和右端向量的单元组装。
    # 关键知识点：
    #   1) 线性三角形单元的面积计算与形函数梯度
    #   2) 中边点求积规则 (wq = 1/3)
    #   3) 单元刚度矩阵: A_e[i,j] += area * wq * conductivity * (dphi_i/dx * dphi_j/dx + dphi_i/dy * dphi_j/dy)
    #   4) 右端项: F_e[i] += area * wq * source(xq,yq) * phi_i(xq,yq)
    #   5) 注意 element_node 的索引基制 (0-based 或 1-based) 必须与 Hole_2 保持一致
    raise NotImplementedError("Hole_1: FEM 刚度矩阵组装循环待实现")

    # 边界条件
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

    # 求解
    u = np.linalg.solve(A, F)

    # 误差估计
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
    """
    从FEM解中提取节点梯度（用于网格自适应细化）。
    使用单元梯度面积加权平均到节点。
    """
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

        # 单元常数梯度
        dudx = (u[i1] * (y2 - y3) + u[i2] * (y3 - y1) + u[i3] * (y1 - y2)) / (2.0 * area)
        dudy = (u[i1] * (x3 - x2) + u[i2] * (x1 - x3) + u[i3] * (x2 - x1)) / (2.0 * area)

        for idx in [i1, i2, i3]:
            grad[0, idx] += dudx * area
            grad[1, idx] += dudy * area
            weight[idx] += area

    mask = weight > 0
    grad[:, mask] = grad[:, mask] / weight[mask]
    return grad
