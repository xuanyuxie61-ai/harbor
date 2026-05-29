"""
fem_pressure_wave.py
二维有限元法求解气泡崩溃产生的压力波传播

核心物理模型:
1. 声学波动方程（弱形式）:
   ∫_Ω (∂²p/∂t²) v dΩ + c² ∫_Ω ∇p · ∇v dΩ = c² ∫_{∂Ω} (∂p/∂n) v dS

2. 时间离散（隐式 Euler）:
   (M + c² Δt² K) p^{n+1} = 2M p^n - M p^{n-1} + Δt² f^{n+1}
   其中 M 为质量矩阵, K 为刚度矩阵。

3. 气泡边界条件:
   在气泡壁 r = R(t) 处，压力由 Rayleigh-Plesset 方程给出:
   p(r=R) = p_g - 2σ/R - 4μ(dR/dt)/R

映射来源:
- 410_fem2d_predator_prey_fast: 2D FEM 框架 → 压力波传播 FEM
- 475_gmsh_to_fem: 网格数据读取与转换 → 自定义网格生成
"""

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve, gmres, splu
from scipy.spatial import Delaunay


def generate_square_mesh(a, b, h):
    """
    生成正方形区域 [a,b]×[a,b] 的三角形网格。
    对应 475_gmsh_to_fem 的网格数据读取功能，但改为程序化生成。

    返回:
        nodes: N_nodes x 2 的节点坐标
        elements: N_elements x 3 的三角形单元（节点索引）
    """
    x = np.arange(a, b + h, h)
    y = np.arange(a, b + h, h)
    X, Y = np.meshgrid(x, y)
    points = np.vstack([X.ravel(), Y.ravel()]).T

    tri = Delaunay(points)
    elements = tri.simplices

    # 去除退化单元
    valid = []
    for elem in elements:
        p0, p1, p2 = points[elem]
        area = 0.5 * abs((p1[0]-p0[0])*(p2[1]-p0[1]) - (p2[0]-p0[0])*(p1[1]-p0[1]))
        if area > 1e-15:
            valid.append(elem)
    elements = np.array(valid, dtype=int)

    return points, elements


def fem_matrices_2d(nodes, elements, c_sound, rho):
    """
    组装二维 FEM 质量矩阵 M 和刚度矩阵 K。

    对三角形单元，局部质量矩阵（一致质量）:
      M_e = (Area/12) * [[2,1,1],[1,2,1],[1,1,2]]
    局部刚度矩阵:
      K_e = Area * B^T B,  B = [∂N_i/∂x, ∂N_i/∂y]
    """
    n_nodes = len(nodes)
    M_data = []
    M_row = []
    M_col = []
    K_data = []
    K_row = []
    K_col = []

    for elem in elements:
        idx = elem
        x = nodes[idx, 0]
        y = nodes[idx, 1]

        # 三角形面积
        area = 0.5 * abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
        if area < 1e-15:
            continue

        # 形函数梯度
        b_coeff = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]])
        c_coeff = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])

        # 局部质量矩阵（一致质量）
        Me = (area / 12.0) * np.array([[2.0, 1.0, 1.0],
                                       [1.0, 2.0, 1.0],
                                       [1.0, 1.0, 2.0]])
        # 局部刚度矩阵
        Be = np.vstack([b_coeff, c_coeff]) / (2.0 * area)
        Ke = area * (Be.T @ Be)

        for i in range(3):
            for j in range(3):
                M_row.append(idx[i])
                M_col.append(idx[j])
                M_data.append(Me[i, j])
                K_row.append(idx[i])
                K_col.append(idx[j])
                K_data.append(Ke[i, j])

    M = csr_matrix((M_data, (M_row, M_col)), shape=(n_nodes, n_nodes))
    K = csr_matrix((K_data, (K_row, K_col)), shape=(n_nodes, n_nodes))
    return M, K


def apply_boundary_conditions(M, K, nodes, bubble_center, bubble_radius, p_wall, dt, c_sound):
    """
    施加气泡壁 Dirichlet 边界条件和外边界辐射边界条件。

    参数:
        bubble_center: 气泡中心 [2]
        bubble_radius: 当前气泡半径
        p_wall: 气泡壁压力
    """
    n_nodes = len(nodes)
    bc_nodes = []

    for i in range(n_nodes):
        dist = np.linalg.norm(nodes[i] - bubble_center)
        # 气泡壁附近的节点标记为 Dirichlet
        if dist <= bubble_radius * 1.1:
            bc_nodes.append(i)
        # 外边界（域边界）应用近似吸收边界

    bc_nodes = list(set(bc_nodes))

    # 修改矩阵
    M_mod = M.copy()
    K_mod = K.copy()

    for i in bc_nodes:
        # 将第 i 行除对角线外清零
        row_start = M_mod.indptr[i]
        row_end = M_mod.indptr[i + 1]
        M_mod.data[row_start:row_end] = 0.0
        M_mod.data[row_start] = 1.0  # 假设 CSR 中对角线在行首（不一定，这里简化）

        row_start = K_mod.indptr[i]
        row_end = K_mod.indptr[i + 1]
        K_mod.data[row_start:row_end] = 0.0

    # 更稳妥的方式：转换为密集矩阵修改（小网格时可行）
    if n_nodes <= 2000:
        M_mod = M.toarray()
        K_mod = K.toarray()
        for i in bc_nodes:
            M_mod[i, :] = 0.0
            M_mod[:, i] = 0.0
            M_mod[i, i] = 1.0
            K_mod[i, :] = 0.0
            K_mod[:, i] = 0.0
            K_mod[i, i] = 1.0
        M_mod = csr_matrix(M_mod)
        K_mod = csr_matrix(K_mod)

    return M_mod, K_mod, bc_nodes


def solve_pressure_wave_fem(nodes, elements, c_sound, rho, t_span, dt,
                            bubble_center, bubble_radius_func, p_wall_func,
                            p_init=0.0):
    """
    使用 FEM 求解压力波传播。

    时间离散（Newmark-β 方法，β=1/4, γ=1/2）:
      M * a^{n+1} + c² K * p^{n+1} = f^{n+1}
      v^{n+1} = v^n + Δt/2 * (a^n + a^{n+1})
      p^{n+1} = p^n + Δt * v^n + (Δt²/4) * (a^n + a^{n+1})

    参数:
        bubble_radius_func: 函数 R(t)
        p_wall_func: 函数 p_wall(t)
    返回:
        p_history: 每个时间步的压力场
    """
    n_nodes = len(nodes)
    M, K = fem_matrices_2d(nodes, elements, c_sound, rho)

    # 使用集中质量矩阵以提高稳定性
    M_lumped = diags(np.array(M.sum(axis=1)).ravel())

    n_steps = int((t_span[1] - t_span[0]) / dt)
    p = np.full(n_nodes, p_init)
    v = np.zeros(n_nodes)
    a = np.zeros(n_nodes)

    p_history = [p.copy()]

    # 预分解矩阵
    A_mat = M_lumped + (dt**2 / 4.0) * c_sound**2 * K
    try:
        lu = splu(A_mat.tocsc())
    except RuntimeError:
        lu = None

    # TODO: 实现 Newmark-β 时间推进循环
    pass

    return np.array(p_history)


def acoustic_energy_fem(p, v, nodes, elements, rho, c_sound):
    """
    计算 FEM 网格上的声能密度:
    E = (1/2) * (∫_Ω p²/(ρc²) dΩ + ∫_Ω ρ v² dΩ)
    """
    n_nodes = len(nodes)
    E = 0.0
    for elem in elements:
        x = nodes[elem, 0]
        y = nodes[elem, 1]
        area = 0.5 * abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
        if area < 1e-15:
            continue
        # 单元内取节点平均值
        p_avg = np.mean(p[elem])
        v_avg = np.mean(v[elem]) if hasattr(v, '__getitem__') else 0.0
        E += area * (0.5 * p_avg**2 / (rho * c_sound**2) + 0.5 * rho * v_avg**2)
    return E


def find_boundary_edges(nodes, elements):
    """
    提取网格边界边。
    对应 410_fem2d_predator_prey_fast 中的 boundedges。
    """
    edges = {}
    for elem in elements:
        e = [(elem[0], elem[1]), (elem[1], elem[2]), (elem[2], elem[0])]
        for edge in e:
            e_sorted = tuple(sorted(edge))
            if e_sorted in edges:
                edges[e_sorted] += 1
            else:
                edges[e_sorted] = 1
    boundary = [e for e, count in edges.items() if count == 1]
    return np.array(boundary)


def pressure_gradient_at_nodes(p, nodes, elements):
    """
    计算节点压力梯度（最小二乘恢复）。
    """
    n_nodes = len(nodes)
    grad_p = np.zeros((n_nodes, 2))
    count = np.zeros(n_nodes)

    for elem in elements:
        x = nodes[elem, 0]
        y = nodes[elem, 1]
        area = 0.5 * abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
        if area < 1e-15:
            continue

        b_coeff = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]]) / (2.0 * area)
        c_coeff = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]]) / (2.0 * area)

        grad_elem = np.array([np.dot(b_coeff, p[elem]), np.dot(c_coeff, p[elem])])
        for i in range(3):
            grad_p[elem[i]] += grad_elem
            count[elem[i]] += 1

    for i in range(n_nodes):
        if count[i] > 0:
            grad_p[i] /= count[i]

    return grad_p
