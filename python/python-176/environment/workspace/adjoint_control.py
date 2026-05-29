"""
adjoint_control.py
================================================================================
PDE 约束最优控制伴随方程方法核心模块

本模块是整個合成项目的算法核心，负责：
  1. 状态方程（前向热/反应-扩散方程）的时间推进
  2. 伴随方程（后向线性化方程）的时间推进
  3. 目标泛函 J(q) 及其梯度 ∇J(q) 的计算
  4. 梯度下降优化循环

科学背景
--------
考虑二维椭圆域 Ω 上的非定常反应-扩散方程约束最优控制问题：

状态方程（前向 PDE）：
    ∂y/∂t − ν Δy + c y³ = f(x,t)          in Ω × (0,T)
    ν ∂y/∂n = q(s,t)                      on ∂Ω × (0,T)
    y(x,0) = y_0(x)                       in Ω

目标泛函（跟踪型）：
    J(q) = ½∫_0^T ∫_Ω (y − y_d)² dx dt
         + (α/2)∫_0^T ∫_{∂Ω} q² ds dt
         + (β/2)∫_0^T ∫_{∂Ω} |∂_s q|² ds dt

其中 y_d 是期望状态，α > 0 是控制代价系数，β ≥ 0 是控制光滑性系数。

伴随方程（后向 PDE）：
    −∂p/∂t − ν Δp + 3c y² p = y − y_d     in Ω × (0,T)
    ν ∂p/∂n = 0                           on ∂Ω × (0,T)
    p(x,T) = 0                            in Ω

梯度公式（通过 Lagrange 乘子法推导）：
    构造 Lagrangian
    L = J + ∫_0^T ∫_Ω p (∂y/∂t − νΔy + c y³ − f) dx dt
    对 q 求变分得：
    δJ = ∫_0^T ∫_{∂Ω} (α q + p|_{∂Ω}) δq ds dt
    因此梯度为：
    ∇J(q) = α q + p|_{∂Ω}  （在边界上）

    若加入光滑项 β |∂_s q|²，则梯度增加项 −β ∂_s² q。

优化算法
--------
使用带 Armijo 线搜索的梯度下降法：
  q^{k+1} = q^k − η_k ∇J(q^k)
  其中 η_k 通过 Armijo 条件确定：
  J(q^k − η ∇J) ≤ J(q^k) − c η ‖∇J‖²，c ∈ (0,1)

关键公式
--------
1. 半离散状态方程（隐式欧拉，线性化非线性项）：
   (M + Δt ν A + Δt c M diag((y^n)²)) y^{n+1}
       = M y^n + Δt F^{n+1} + Δt B q^{n+1}

2. 半离散伴随方程（隐式欧拉，后向）：
   (M + Δt ν A + Δt 3c M diag((y^{n+1})²)) p^n
       = M p^{n+1} + Δt M (y^{n+1} − y_d^{n+1})

3. 离散目标泛函：
   J ≈ Σ_n Δt [ ½ (y^n − y_d^n)^T M (y^n − y_d^n)
              + (α/2) (q^n)^T B q^n
              + (β/2) (q^n)^T L_{bd} q^n ]
   其中 L_{bd} 是边界上的 Laplace 矩阵（一维离散）。

4. 离散梯度（在边界节点上）：
   (∇J)_i = α (B q)_i + (B p)_i + β (L_{bd} q)_i
"""

import numpy as np


def solve_state_forward(nodes, elements, boundary_nodes, boundary_edges,
                        M, A, B, y0, q_seq, f_fn, nu, c, T, n_time):
    """
    使用前向隐式欧拉求解状态方程。
    非线性项 c y³ 采用线性化：c y²_{old} · y_{new}（Picard 型）。

    参数
    ----
    nodes, elements, boundary_nodes, boundary_edges : 网格数据
    M, A, B : FEM 矩阵
    y0      : 初始状态（节点值）
    q_seq   : (n_time+1, n_boundary) 控制序列
    f_fn    : 源项函数 f(x,y,t)
    nu, c   : 扩散系数、反应系数
    T       : 终止时间
    n_time  : 时间步数

    返回
    ----
    y_seq : (n_time+1, n_nodes) 状态轨迹
    """
    n_nodes = nodes.shape[0]
    n_bnd = len(boundary_nodes)
    dt = T / n_time
    y_seq = np.zeros((n_time + 1, n_nodes), dtype=float)
    y_seq[0] = y0.copy()

    # 构建从全局边界节点索引到局部 q 索引的映射
    bnd_map = {int(b): i for i, b in enumerate(boundary_nodes)}

    for n in range(n_time):
        t_np1 = (n + 1) * dt
        # 右端项
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

        # 控制项：B 作用于扩展的 q（全节点向量，边界外为零）
        q_full = np.zeros(n_nodes, dtype=float)
        for b_idx, glob_idx in enumerate(boundary_nodes):
            q_full[glob_idx] = q_seq[n + 1, b_idx]
        control_term = B @ q_full

        # TODO(Hole_2): 实现状态方程的隐式 Euler 时间步进。
        # 科学知识点：
        #   - 非线性项 c*y³ 的 Picard 型线性化：c*(y_old)² * y_new
        #   - LHS = M + dt*nu*A + dt*c*diag(y_old²)*M_lumped
        #   - rhs = M @ y_old + dt * F + dt * control_term
        #   - 使用 np.linalg.solve(LHS, rhs) 求解线性系统
        raise NotImplementedError("Hole_2: 请实现状态方程的隐式 Euler 时间步进")


def solve_adjoint_backward(nodes, elements, boundary_nodes, boundary_edges,
                           M, A, B, y_seq, yd_seq, nu, c, T, n_time):
    """
    使用后向隐式欧拉求解伴随方程。
    注意伴随方程是后向时间积分，从 p(T) = 0 开始。
    """
    n_nodes = nodes.shape[0]
    dt = T / n_time
    p_seq = np.zeros((n_time + 1, n_nodes), dtype=float)
    p_seq[-1] = 0.0  # p(T) = 0

    for n in range(n_time - 1, -1, -1):
        # TODO(Hole_3): 实现伴随方程的后向隐式 Euler 时间步进。
        # 科学知识点：
        #   - 伴随方程是后向时间积分：从 p(T)=0 开始，逐步回推
        #   - 右端项 rhs = M @ p^{n+1} + dt * M @ (y^{n+1} - y_d^{n+1})
        #   - LHS = M + dt*nu*A + dt*3c*diag((y^{n+1})²)*M_lumped
        #   - 使用 np.linalg.solve(LHS, rhs) 求解
        raise NotImplementedError("Hole_3: 请实现伴随方程的后向隐式 Euler 时间步进")


def compute_objective(nodes, elements, boundary_nodes, boundary_edges,
                      M, B, L_bd, y_seq, yd_seq, q_seq, alpha, beta, T, n_time):
    """
    计算离散目标泛函 J(q)。
    """
    n_nodes = nodes.shape[0]
    dt = T / n_time
    J = 0.0

    for n in range(n_time + 1):
        dy = y_seq[n] - yd_seq[n]
        # 状态偏差项
        J += 0.5 * dt * np.dot(dy, M @ dy)

        # 控制代价项
        q_full = np.zeros(n_nodes, dtype=float)
        for b_idx, glob_idx in enumerate(boundary_nodes):
            q_full[glob_idx] = q_seq[n, b_idx]
        J += 0.5 * alpha * dt * np.dot(q_full, B @ q_full)

        # 控制光滑项
        if beta > 0.0 and L_bd is not None:
            J += 0.5 * beta * dt * np.dot(q_seq[n], L_bd @ q_seq[n])

    return J


def compute_gradient(nodes, elements, boundary_nodes, boundary_edges,
                     M, B, L_bd, p_seq, q_seq, alpha, beta, T, n_time):
    """
    计算离散梯度 ∇J(q) 在边界节点上的值。
    梯度维度与 q_seq 相同：(n_time+1, n_boundary)
    """
    n_bnd = len(boundary_nodes)
    dt = T / n_time
    grad = np.zeros((n_time + 1, n_bnd), dtype=float)

    for n in range(n_time + 1):
        q_full = np.zeros(nodes.shape[0], dtype=float)
        for b_idx, glob_idx in enumerate(boundary_nodes):
            q_full[glob_idx] = q_seq[n, b_idx]

        # 梯度 = α B q + B p + β L_bd q
        g_full = alpha * (B @ q_full) + (B @ p_seq[n])

        # 提取边界分量
        for b_idx, glob_idx in enumerate(boundary_nodes):
            grad[n, b_idx] = g_full[glob_idx]

        if beta > 0.0 and L_bd is not None:
            grad[n] += beta * (L_bd @ q_seq[n])

    return grad


def build_boundary_laplacian_1d(boundary_nodes, nodes):
    """
    构建边界上的一维 Laplace 矩阵（用于控制光滑性惩罚）。
    假设边界节点按顺序排列（近似）。如果无序，则基于最近邻连接。
    """
    n_bnd = len(boundary_nodes)
    L = np.zeros((n_bnd, n_bnd), dtype=float)

    if n_bnd < 2:
        return L

    # 获取边界节点坐标
    coords = nodes[boundary_nodes]

    # 按角度排序（假设是凸边界）
    angles = np.arctan2(coords[:, 1], coords[:, 0])
    order = np.argsort(angles)
    sorted_indices = [boundary_nodes[o] for o in order]

    # 构建环形 1D Laplace
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
    """
    Armijo 线搜索：寻找步长 η 使得
    J(q − η grad) ≤ J(q) − c η ‖grad‖²
    """
    J0 = compute_objective(nodes, elements, boundary_nodes, boundary_edges,
                           M, B, L_bd, solve_state_forward(nodes, elements, boundary_nodes, boundary_edges,
                                                           M, A, B, y0, q_seq, f_fn, nu, c, T, n_time),
                           yd_seq, q_seq, alpha, beta, T, n_time)
    grad_norm2 = np.sum(grad ** 2)

    eta = eta_init
    for _ in range(max_iter):
        q_new = q_seq - eta * grad
        # 边界处理：裁剪到合理范围
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
    """
    主优化循环：使用伴随方程方法求解最优边界控制。

    返回
    ----
    q_opt   : 最优控制序列
    y_opt   : 最优状态轨迹
    p_opt   : 伴随轨迹
    history : 目标泛函历史
    """
    n_bnd = len(boundary_nodes)
    L_bd = build_boundary_laplacian_1d(boundary_nodes, nodes)

    # 初始化控制为零
    q_seq = np.zeros((n_time + 1, n_bnd), dtype=float)

    history = []
    for k in range(max_iter):
        # 1) 求解状态方程
        y_seq = solve_state_forward(nodes, elements, boundary_nodes, boundary_edges,
                                    M, A, B, y0, q_seq, f_fn, nu, c, T, n_time)

        # 2) 求解伴随方程
        p_seq = solve_adjoint_backward(nodes, elements, boundary_nodes, boundary_edges,
                                       M, A, B, y_seq, yd_seq, nu, c, T, n_time)

        # 3) 计算梯度
        grad = compute_gradient(nodes, elements, boundary_nodes, boundary_edges,
                                M, B, L_bd, p_seq, q_seq, alpha, beta, T, n_time)

        # 4) 计算目标泛函
        J_val = compute_objective(nodes, elements, boundary_nodes, boundary_edges,
                                  M, B, L_bd, y_seq, yd_seq, q_seq, alpha, beta, T, n_time)
        history.append(J_val)

        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            print(f"  优化收敛于迭代 {k}, 梯度范数 = {grad_norm:.6e}, J = {J_val:.6e}")
            break

        # 5) 线搜索
        eta, q_seq, y_seq, J_new = armijo_line_search(
            nodes, elements, boundary_nodes, boundary_edges,
            M, A, B, L_bd, y0, yd_seq, f_fn, grad, q_seq,
            alpha, beta, nu, c, T, n_time
        )

        if k % 5 == 0:
            print(f"  迭代 {k}: J = {J_val:.6e}, ‖∇J‖ = {grad_norm:.6e}, η = {eta:.4e}")

    # 最终计算
    y_seq = solve_state_forward(nodes, elements, boundary_nodes, boundary_edges,
                                M, A, B, y0, q_seq, f_fn, nu, c, T, n_time)
    p_seq = solve_adjoint_backward(nodes, elements, boundary_nodes, boundary_edges,
                                   M, A, B, y_seq, yd_seq, nu, c, T, n_time)

    return q_seq, y_seq, p_seq, history
