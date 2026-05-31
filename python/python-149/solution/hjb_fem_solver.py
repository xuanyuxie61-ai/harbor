"""
hjb_fem_solver.py
Hamilton-Jacobi-Bellman方程有限元求解器

融合种子项目:
  - 1230_tet_mesh: 四面体网格剖分与质量评估
  - 085_bicg: 双共轭梯度法求解大型稀疏线性系统
  - 738_matrix_assemble_parfor: 大规模矩阵组装思想

科学背景:
  对于受控的随机神经动力学系统:
      dX = f(X,u) dt + σ(X) dW

  其值函数 V(x,t) 满足后向HJB方程:

      ∂V/∂t + min_u { L(x,u) + ∇V·f(x,u) + 0.5 Tr[σσ^T ∇²V] } = 0

  终端条件: V(x,T) = Φ(x)

  采用有限元法(FEM)在空间方向离散，隐式欧拉在时间方向离散:

      (M + Δt A) V^{n} = M V^{n+1} + Δt b^n

  其中 M 为质量矩阵，A 为刚度矩阵与对流矩阵的聚合，b 为源项。
"""

import numpy as np
from typing import Callable, Optional, Tuple, List


# ============================================================================
# 四面体几何工具（融合tet_mesh思想）
# ============================================================================

def tetrahedron_volume(verts: np.ndarray) -> float:
    """
    计算四面体体积。

        V = |det([v1-v0, v2-v0, v3-v0])| / 6

    Parameters
    ----------
    verts : ndarray, shape (4, 3)
        四个顶点坐标

    Returns
    -------
    vol : float
        体积（边界保护：确保非负）
    """
    if verts.shape != (4, 3):
        raise ValueError("verts必须为(4,3)数组")
    M = np.zeros((4, 4))
    M[:, :3] = verts
    M[:, 3] = 1.0
    vol = np.abs(np.linalg.det(M)) / 6.0
    return float(max(vol, 1e-15))


def compute_tet_quality(verts: np.ndarray) -> float:
    """
    四面体质量度量（内切球半径与外接球半径之比）:

        ρ = 3 * r_in / R_out

    对于正四面体 ρ = 1，退化四面体 ρ → 0。
    """
    vol = tetrahedron_volume(verts)
    if vol < 1e-14:
        return 0.0

    # 计算六条边长的平方
    edges = []
    for i in range(4):
        for j in range(i + 1, 4):
            edges.append(np.sum((verts[i] - verts[j]) ** 2))
    edges = np.array(edges)

    # 面积向量之和的近似（使用Cayley-Menger行列式简化计算）
    # 简化为体积与边长关系
    # 质量度量近似: ρ ≈ c * V^{2/3} / sum(edges^2)
    quality = 12.0 * (3.0 * vol ** 2) ** (1.0 / 3.0) / np.sum(edges)
    return float(np.clip(quality, 0.0, 1.0))


def regular_tetrahedral_mesh(
    bounds: Tuple[np.ndarray, np.ndarray],
    n_per_dim: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    在三维状态空间（或增广状态空间）中生成规则四面体网格。

    由于实际计算限制，这里生成三维立方体区域的规则四面体剖分，
    每个立方体单元剖分为6个四面体（Kuhn剖分）。

    Parameters
    ----------
    bounds : (xmin, xmax)
        xmin, xmax 为长度为3的数组
    n_per_dim : int
        每维节点数

    Returns
    -------
    nodes : ndarray, shape (n_nodes, 3)
    tets : ndarray, shape (n_tets, 4)
        四面体单元索引（从0开始）
    """
    xmin, xmax = bounds
    dims = 3

    # 生成均匀网格节点
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
                # 立方体8个顶点索引
                n000 = node_index(i, j, k)
                n100 = node_index(i + 1, j, k)
                n010 = node_index(i, j + 1, k)
                n110 = node_index(i + 1, j + 1, k)
                n001 = node_index(i, j, k + 1)
                n101 = node_index(i + 1, j, k + 1)
                n011 = node_index(i, j + 1, k + 1)
                n111 = node_index(i + 1, j + 1, k + 1)

                # Kuhn剖分：6个四面体
                tets.append([n000, n100, n110, n111])
                tets.append([n000, n100, n101, n111])
                tets.append([n000, n001, n101, n111])
                tets.append([n000, n001, n011, n111])
                tets.append([n000, n010, n110, n111])
                tets.append([n000, n010, n011, n111])

    tets = np.array(tets, dtype=int)
    return nodes, tets


# ============================================================================
# 有限元矩阵组装（融合matrix_assemble思想）
# ============================================================================

def assemble_fem_matrices(
    nodes: np.ndarray,
    tets: np.ndarray,
    drift_fn: Callable[[np.ndarray], np.ndarray],
    diffusion_fn: Callable[[np.ndarray], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    组装有限元质量矩阵 M 和聚合刚度/对流矩阵 A。

    对于线性元 (P1)，在四面体单元 T 上：

        M_{ij} = ∫_T φ_i φ_j dx
        A_{ij} = ∫_T [ ε ∇φ_i·∇φ_j - (f·∇φ_i) φ_j ] dx

    其中 ε = 0.5 * Tr(σσ^T) 为有效扩散系数。

    Parameters
    ----------
    nodes : ndarray, shape (n_nodes, 3)
    tets : ndarray, shape (n_tets, 4)
    drift_fn : callable
        漂移向量场 f(x) → ndarray(3,)
    diffusion_fn : callable
        扩散矩阵 σ(x) → ndarray(3,3)

    Returns
    -------
    M : ndarray, shape (n_nodes, n_nodes)
    A : ndarray, shape (n_nodes, n_nodes)
    """
    n_nodes = nodes.shape[0]
    M = np.zeros((n_nodes, n_nodes))
    A = np.zeros((n_nodes, n_nodes))

    for tet in tets:
        verts = nodes[tet, :]
        vol = tetrahedron_volume(verts)
        if vol < 1e-14:
            continue

        # 形函数梯度（常数，在P1元上）
        # 构造矩阵 [v1-v0, v2-v0, v3-v0]
        D = np.zeros((3, 3))
        for d in range(3):
            D[:, d] = verts[d + 1, :] - verts[0, :]

        try:
            D_inv = np.linalg.inv(D)
        except np.linalg.LinAlgError:
            continue

        # 四个形函数的梯度
        # ∇φ_0 = -(D_inv)^T * [1,1,1]
        # ∇φ_i = (D_inv)^T * e_i, i=1,2,3
        grads = np.zeros((4, 3))
        grads[0, :] = -np.sum(D_inv, axis=0)
        grads[1:4, :] = D_inv

        # 单元重心处的漂移与扩散
        centroid = np.mean(verts, axis=0)
        f_cent = np.atleast_1d(drift_fn(centroid))
        sigma_cent = np.atleast_1d(diffusion_fn(centroid))
        if sigma_cent.ndim < 2:
            sigma_cent = np.diag(sigma_cent)
        eps_mat = 0.5 * (sigma_cent @ sigma_cent.T)
        eps_trace = np.trace(eps_mat)

        # 单元矩阵组装
        for i_local in range(4):
            for j_local in range(4):
                gi = grads[i_local, :]
                gj = grads[j_local, :]
                ii = tet[i_local]
                jj = tet[j_local]

                # 质量矩阵 (P1元 lumping 近似: M_ii = vol/4)
                if i_local == j_local:
                    M[ii, jj] += vol / 4.0
                else:
                    M[ii, jj] += vol / 20.0  # 非对角

                # 刚度项: ε ∫ ∇φ_i·∇φ_j dx = ε * vol * (gi·gj)
                stiff = eps_trace * vol * np.dot(gi, gj)

                # 对流项: -∫ (f·∇φ_i) φ_j dx
                # 近似: -vol/4 * (f·gi)  (对j取平均)
                conv = -vol / 4.0 * np.dot(f_cent, gi)
                if i_local == j_local:
                    conv *= 1.0  # lumping

                A[ii, jj] += stiff + conv

    return M, A


# ============================================================================
# BiCG求解器（融合bicg思想）
# ============================================================================

def bicg_solver(
    A_mat: np.ndarray,
    b_vec: np.ndarray,
    x0: Optional[np.ndarray] = None,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, float, int, int]:
    """
    双共轭梯度法 (BiCG) 求解 Ax = b。

    BiCG算法（针对一般非对称矩阵）:
        r_0 = b - A x_0
        r̃_0 = r_0
        for k = 1,2,...
            ρ_k = (r̃_{k-1}, z_{k-1})
            if ρ_k == 0: break
            β = ρ_k / ρ_{k-1}
            p_k = z_{k-1} + β p_{k-1}
            p̃_k = z̃_{k-1} + β p̃_{k-1}
            α = ρ_k / (p̃_k, A p_k)
            x_k = x_{k-1} + α p_k
            r_k = r_{k-1} - α A p_k
            r̃_k = r̃_{k-1} - α A^T p̃_k
            检查收敛

    Returns
    -------
    x : ndarray
        解向量
    error : float
        相对残差 ||b-Ax||/||b||
    it : int
        迭代次数
    flag : int
        0=收敛, 1=达到最大迭代, -1= breakdown
    """
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


# ============================================================================
# HJB时间步进求解器
# ============================================================================

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
    """
    通过有限元+隐式欧拉后向求解HJB方程。

    值函数 V(x,t) 在节点上离散为 V_i^n ≈ V(nodes[i], t_n)。

    对于每个时间层 n（从T逆向到0）：
        (M + Δt A) V^n = M V^{n+1} + Δt * min_u [ L(x,u) + ... ]

    这里对控制进行离散搜索（假设控制候选集已知）。

    Parameters
    ----------
    control_candidates : list of ndarray or None
        控制候选向量列表

    Returns
    -------
    V_history : ndarray, shape (n_time+1, n_nodes)
        各时间层的值函数
    t_grid : ndarray
        时间网格
    """
    t0, tstop = tspan
    dt = (tstop - t0) / n_time
    t_grid = np.linspace(t0, tstop, n_time + 1)

    n_nodes = nodes.shape[0]

    # 组装空间矩阵
    M, A_base = assemble_fem_matrices(nodes, tets, drift_fn, diffusion_fn)

    # 终端条件
    V_next = terminal_cost_fn(nodes)
    if len(V_next) != n_nodes:
        raise ValueError("终端代价函数输出维度与节点数不匹配")

    V_history = np.zeros((n_time + 1, n_nodes))
    V_history[-1, :] = V_next

    # 默认控制候选
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
        # 右端项: M V^{n+1} + dt * min_u running_cost
        rhs = M @ V_next

        # 对每个控制候选计算运行代价并取最小
        best_cost = np.full(n_nodes, np.inf)
        for u_cand in control_candidates:
            cost = running_cost_fn(nodes, u_cand)
            best_cost = np.minimum(best_cost, cost)

        rhs += dt * best_cost

        # 左侧系统矩阵 (M + dt * A)
        LHS = M + dt * A_base
        # 添加小量正则化防止奇异
        LHS += 1e-10 * I_mat

        # 用BiCG求解
        V_current, err, iters, flag = bicg_solver(LHS, rhs, x0=V_next, max_iter=2000, tol=1e-10)

        if flag != 0:
            # 若BiCG失败，回退到直接求解（小规模时）
            try:
                V_current = np.linalg.solve(LHS, rhs)
            except np.linalg.LinAlgError:
                V_current = V_next.copy()

        V_history[n, :] = V_current
        V_next = V_current.copy()

    return V_history, t_grid
