# -*- coding: utf-8 -*-
"""
sparse_solver.py
稀疏矩阵处理与求解模块

融合来源:
- 1156_st_to_ge: 稀疏矩阵格式转换 (ST 到 GE)

功能:
- COO/CSR 格式稀疏矩阵的组装与转换
- 使用共轭梯度法 (CG) 和 BiCGSTAB 求解大型稀疏线性系统
- 压力 Poisson 方程的求解（SIMPLE 算法中关键步骤）

数学背景:
  在有限元/谱元离散中，刚度矩阵 K 具有高度稀疏性。
  对于压力 Poisson 方程:
    nabla^2 p = f
  离散后得到线性系统:
    A * p = b
  其中 A 为对称正定稀疏矩阵，适合使用 CG 法求解。

  CG 算法:
    给定初始猜测 x_0，设 r_0 = b - A*x_0, p_0 = r_0
    for k = 0, 1, 2, ...:
      alpha_k = (r_k^T * r_k) / (p_k^T * A * p_k)
      x_{k+1} = x_k + alpha_k * p_k
      r_{k+1} = r_k - alpha_k * A * p_k
      beta_k = (r_{k+1}^T * r_{k+1}) / (r_k^T * r_k)
      p_{k+1} = r_{k+1} + beta_k * p_k
"""

import numpy as np


def st_to_ge(nst, ist, jst, Ast):
    """
    将 ST（稀疏三元组）格式矩阵转换为稠密 GE 格式。
    融合自 1156_st_to_ge。

    参数:
      nst: 非零元个数
      ist, jst: 行索引和列索引数组
      Ast: 非零元值数组

    返回:
      Age: 稠密矩阵
    """
    ist = np.asarray(ist, dtype=int)
    jst = np.asarray(jst, dtype=int)
    Ast = np.asarray(Ast, dtype=float)

    m = int(np.max(ist)) if len(ist) > 0 else 0
    n = int(np.max(jst)) if len(jst) > 0 else 0
    Age = np.zeros((m, n), dtype=float)

    for kst in range(nst):
        i = ist[kst] - 1  # 转为 0-based
        j = jst[kst] - 1
        if 0 <= i < m and 0 <= j < n:
            Age[i, j] += Ast[kst]

    return Age


def assemble_sparse_st(connections, values, n_nodes):
    """
    从单元连接关系组装全局稀疏矩阵（ST 格式）。

    参数:
      connections: 单元-节点连接列表，每个元素为局部刚度矩阵和全局节点索引
      values: 局部矩阵值列表
      n_nodes: 全局节点数

    返回:
      A: 稠密全局矩阵（小规模时使用）
    """
    A = np.zeros((n_nodes, n_nodes), dtype=float)
    for local_K, global_dof in zip(values, connections):
        n_loc = len(global_dof)
        for i in range(n_loc):
            for j in range(n_loc):
                gi = global_dof[i]
                gj = global_dof[j]
                if 0 <= gi < n_nodes and 0 <= gj < n_nodes:
                    A[gi, gj] += local_K[i, j]
    return A


def conjugate_gradient(A, b, x0=None, tol=1e-10, max_iter=1000):
    """
    共轭梯度法求解 Ax = b。

    数学推导:
      最小化二次泛函:
        J(x) = 0.5 * x^T * A * x - b^T * x
      其梯度为:
        grad J = A*x - b = -r
      CG 在 Krylov 子空间中寻找最优解。

    参数:
      A: (n, n) 对称正定矩阵
      b: (n,) 右端项
      x0: 初始猜测
      tol: 残差容差
      max_iter: 最大迭代次数

    返回:
      x: 解向量
      info: 迭代信息字典
    """
    n = len(b)
    b = np.asarray(b, dtype=float)
    A = np.asarray(A, dtype=float)

    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).copy()

    r = b - A @ x
    p = r.copy()
    rs_old = float(np.dot(r, r))

    if rs_old < tol * tol:
        return x, {"iter": 0, "residual": np.sqrt(rs_old)}

    for k in range(max_iter):
        Ap = A @ p
        alpha = rs_old / (np.dot(p, Ap) + 1e-15)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = float(np.dot(r, r))

        if np.sqrt(rs_new) < tol:
            return x, {"iter": k + 1, "residual": np.sqrt(rs_new)}

        beta = rs_new / (rs_old + 1e-15)
        p = r + beta * p
        rs_old = rs_new

    return x, {"iter": max_iter, "residual": np.sqrt(rs_old), "converged": False}


def bicgstab(A, b, x0=None, tol=1e-10, max_iter=1000):
    """
    BiCGSTAB 算法求解非对称线性系统 Ax = b。

    数学模型:
      对于非对称矩阵（如对流-扩散问题），CG 不再适用。
      BiCGSTAB 是 BiCG 的稳定化版本，通过最小残差法思想
      减少 BiCG 的振荡行为。

    参数:
      A: (n, n) 非奇异矩阵
      b: (n,) 右端项
      x0: 初始猜测
      tol: 容差
      max_iter: 最大迭代次数

    返回:
      x: 解向量
      info: 迭代信息
    """
    n = len(b)
    b = np.asarray(b, dtype=float)
    A = np.asarray(A, dtype=float)

    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).copy()

    r = b - A @ x
    r0 = r.copy()
    rho_old = 1.0
    alpha = 1.0
    omega = 1.0
    p = np.zeros(n, dtype=float)
    v = np.zeros(n, dtype=float)

    for k in range(max_iter):
        rho = np.dot(r0, r)
        if abs(rho) < 1e-15:
            return x, {"iter": k, "residual": np.linalg.norm(r), "converged": False}

        beta = (rho / (rho_old + 1e-15)) * (alpha / (omega + 1e-15))
        p = r + beta * (p - omega * v)
        v = A @ p
        alpha = rho / (np.dot(r0, v) + 1e-15)
        s = r - alpha * v

        if np.linalg.norm(s) < tol:
            x = x + alpha * p
            return x, {"iter": k + 1, "residual": np.linalg.norm(s), "converged": True}

        t = A @ s
        omega = np.dot(t, s) / (np.dot(t, t) + 1e-15)
        x = x + alpha * p + omega * s
        r = s - omega * t
        rho_old = rho

        if np.linalg.norm(r) < tol:
            return x, {"iter": k + 1, "residual": np.linalg.norm(r), "converged": True}

    return x, {"iter": max_iter, "residual": np.linalg.norm(r), "converged": False}


def solve_pressure_poisson(p, div_u, dx, dy, dz, tol=1e-8, max_iter=500):
    """
    求解压力 Poisson 方程: nabla^2 p = div(u) / dt。

    数学模型:
      在 SIMPLE 算法中，压力修正方程为:
        nabla^2 p' = (1/dt) * div(u*)
      使用七点差分格式离散 Laplacian:
        (p_{i+1,j,k} - 2p_{i,j,k} + p_{i-1,j,k}) / dx^2
      + (p_{i,j+1,k} - 2p_{i,j,k} + p_{i,j-1,k}) / dy^2
      + (p_{i,j,k+1} - 2p_{i,j,k} + p_{i,j,k-1}) / dz^2 = div(u)

    参数:
      p: 初始压力场
      div_u: 速度散度
      dx, dy, dz: 网格间距
      tol, max_iter: CG 求解参数

    返回:
      p: 修正后的压力场
    """
    nx, ny, nz = p.shape

    # 构造系数矩阵（使用迭代法避免显式构造大矩阵）
    def apply_laplacian(phi):
        result = np.zeros_like(phi)
        result[1:-1, 1:-1, 1:-1] = (
            (phi[2:, 1:-1, 1:-1] - 2 * phi[1:-1, 1:-1, 1:-1] + phi[:-2, 1:-1, 1:-1]) / dx ** 2
            + (phi[1:-1, 2:, 1:-1] - 2 * phi[1:-1, 1:-1, 1:-1] + phi[1:-1, :-2, 1:-1]) / dy ** 2
            + (phi[1:-1, 1:-1, 2:] - 2 * phi[1:-1, 1:-1, 1:-1] + phi[1:-1, 1:-1, :-2]) / dz ** 2
        )
        return result

    # 展平为向量进行 CG 迭代
    b = div_u.flatten()
    x0 = p.flatten()

    # 使用带预处理的迭代法（简化为 Jacobi 预处理）
    n = len(b)
    x = x0.copy()
    r = b - apply_laplacian(x.reshape(nx, ny, nz)).flatten()

    # 对角预处理
    diag_val = -2.0 * (1.0 / dx ** 2 + 1.0 / dy ** 2 + 1.0 / dz ** 2)
    if abs(diag_val) < 1e-15:
        diag_val = -1.0

    z = r / diag_val
    p_vec = z.copy()
    rz_old = np.dot(r, z)

    for k in range(max_iter):
        Ap = apply_laplacian(p_vec.reshape(nx, ny, nz)).flatten()
        alpha = rz_old / (np.dot(p_vec, Ap) + 1e-15)
        x = x + alpha * p_vec
        r = r - alpha * Ap

        if np.linalg.norm(r) < tol:
            break

        z = r / diag_val
        rz_new = np.dot(r, z)
        beta = rz_new / (rz_old + 1e-15)
        p_vec = z + beta * p_vec
        rz_old = rz_new

    return x.reshape(nx, ny, nz)
