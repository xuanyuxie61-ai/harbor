"""
sparse_fem.py

稀疏矩阵组装与有限元离散模块

本模块融合以下种子项目的核心算法：
  - 1156_st_to_ge: 稀疏三元组（ST）格式到稠密矩阵（GE）格式的转换

科学背景：
  有限元方法（FEM）离散肿瘤生长 PDE 时，刚度矩阵 K 和质量矩阵 M
  通常是大型稀疏矩阵。稀疏三元组（Coordinate/COO）格式存储为：

    IST(K), JST(K), AST(K)   for K = 1 .. NST

  其中 (IST(K), JST(K)) 为非零元行列索引，AST(K) 为数值。
  同一位置可能出现多次（如不同单元对同一自由度的贡献），
  GE 格式需将同位置的值累加：

    A_{i,j} = sum_{k: IST(k)=i, JST(k)=j} AST(k)

  本模块实现从 COO 到稠密格式的转换，以及 FEM 刚度矩阵的组装。
"""

import numpy as np
from typing import Tuple


def st_to_ge(nst: int, ist: np.ndarray, jst: np.ndarray,
             ast: np.ndarray) -> np.ndarray:
    """
    将稀疏三元组（ST）格式转换为稠密一般（GE）格式。

    参数:
        nst: 非零元个数
        ist: (nst,) 行索引（1-based）
        jst: (nst,) 列索引（1-based）
        ast: (nst,) 数值

    返回:
        Age: (m, n) 稠密矩阵，m = max(ist), n = max(jst)
    """
    if nst < 0:
        raise ValueError("st_to_ge: nst >= 0")
    ist = np.asarray(ist, dtype=int)
    jst = np.asarray(jst, dtype=int)
    ast = np.asarray(ast, dtype=float)

    if ist.shape[0] < nst or jst.shape[0] < nst or ast.shape[0] < nst:
        raise ValueError("st_to_ge: 输入数组长度不足")

    m = int(np.max(ist)) if nst > 0 else 0
    n = int(np.max(jst)) if nst > 0 else 0
    Age = np.zeros((m, n))

    for k in range(nst):
        i = ist[k] - 1  # 转为 0-based
        j = jst[k] - 1
        if 0 <= i < m and 0 <= j < n:
            Age[i, j] += ast[k]

    return Age


def assemble_fem_stiffness_2d(
    nodes: np.ndarray, triangles: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    组装二维线性有限元刚度矩阵（Poisson 方程）。

    对三角形单元 e，其节点为 (a,b,c)，局部刚度矩阵为：
        K_e = (1 / (4 * |T_e|)) * [ ... ]
    其中 |T_e| 为单元面积，矩阵元素由边向量内积决定：
        K_{ij}^{(e)} = (1 / (4*A_e)) * ( (x_j - x_k)*(x_i - x_k) + (y_j - y_k)*(y_i - y_k) )
      其中 (i,j,k) 为 (a,b,c) 的轮换。

    参数:
        nodes: (N, 2) 节点坐标
        triangles: (T, 3) 三角形节点索引（0-based）

    返回:
        ist, jst, ast: COO 格式数组
        nst: 非零元个数
    """
    N = nodes.shape[0]
    T = triangles.shape[0]

    ist_list = []
    jst_list = []
    ast_list = []

    for t in range(T):
        a, b, c = triangles[t, :]
        xa, ya = nodes[a]
        xb, yb = nodes[b]
        xc, yc = nodes[c]

        # 计算面积（带符号）
        area = 0.5 * ((xb - xa) * (yc - ya) - (xc - xa) * (yb - ya))
        area_abs = abs(area)
        if area_abs < 1e-15:
            continue

        # 局部刚度矩阵组装
        # 使用公式: K_loc[i,j] = (1/(4A)) * dot(v_j_perp, v_i_perp)
        # 其中 v_a = (yb-yc, xc-xb), v_b = (yc-ya, xa-xc), v_c = (ya-yb, xb-xa)
        vx = np.array([yb - yc, yc - ya, ya - yb])
        vy = np.array([xc - xb, xa - xc, xb - xa])

        for i_loc in range(3):
            for j_loc in range(3):
                val = (vx[i_loc] * vx[j_loc] + vy[i_loc] * vy[j_loc]) / (4.0 * area_abs)
                ist_list.append(int([a, b, c][i_loc]) + 1)  # 1-based
                jst_list.append(int([a, b, c][j_loc]) + 1)
                ast_list.append(val)

    ist = np.array(ist_list, dtype=int)
    jst = np.array(jst_list, dtype=int)
    ast = np.array(ast_list, dtype=float)
    nst = len(ast_list)

    return ist, jst, ast, nst


def apply_dirichlet_bc(
    A: np.ndarray, b: np.ndarray, bc_nodes: np.ndarray, bc_values: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对稠密矩阵 A 和右端项 b 施加 Dirichlet 边界条件。

    方法: 对边界节点 i，将 A[i,:] 清零，A[i,i]=1，b[i]=g_i。

    参数:
        A: (N, N) 稠密矩阵
        b: (N,) 右端项
        bc_nodes: (M,) 边界节点索引（0-based）
        bc_values: (M,) 边界值

    返回:
        A_mod, b_mod
    """
    A_mod = A.copy()
    b_mod = b.copy()

    for idx, node in enumerate(bc_nodes):
        if 0 <= node < A_mod.shape[0]:
            A_mod[node, :] = 0.0
            A_mod[:, node] = 0.0
            A_mod[node, node] = 1.0
            b_mod[node] = bc_values[idx]

    return A_mod, b_mod


def sparse_matrix_vector_product(
    ist: np.ndarray, jst: np.ndarray, ast: np.ndarray,
    nst: int, x: np.ndarray, n_rows: int
) -> np.ndarray:
    """
    稀疏三元组格式的矩阵-向量乘法 y = A * x。

    参数:
        ist, jst, ast: COO 数组（1-based 索引）
        nst: 非零元个数
        x: (n_cols,) 向量
        n_rows: 结果向量长度

    返回:
        y: (n_rows,) 结果
    """
    y = np.zeros(n_rows)
    for k in range(nst):
        i = ist[k] - 1
        j = jst[k] - 1
        if 0 <= i < n_rows and 0 <= j < x.shape[0]:
            y[i] += ast[k] * x[j]
    return y


def compute_fem_l2_error(
    u_h: np.ndarray, u_exact: np.ndarray,
    nodes: np.ndarray, triangles: np.ndarray
) -> float:
    """
    计算有限元解的 L2 误差（基于单元质心近似）。

    ||u_h - u_exact||_{L2}^2 = sum_e A_e * |u_h(x_c) - u_exact(x_c)|^2
    """
    error_sq = 0.0
    for t in range(triangles.shape[0]):
        a, b, c = triangles[t, :]
        xa, ya = nodes[a]
        xb, yb = nodes[b]
        xc, yc = nodes[c]
        area = 0.5 * abs((xb - xa) * (yc - ya) - (xc - xa) * (yb - ya))
        if area < 1e-15:
            continue

        # 质心处的近似解（线性插值）
        u_centroid = (u_h[a] + u_h[b] + u_h[c]) / 3.0
        x_c = (xa + xb + xc) / 3.0
        y_c = (ya + yb + yc) / 3.0
        u_ex = u_exact(x_c, y_c)
        error_sq += area * (u_centroid - u_ex) ** 2

    return np.sqrt(error_sq)
