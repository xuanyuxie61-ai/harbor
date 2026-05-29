"""
matrix_exponential_int.py
=========================
矩阵指数计算与时间积分模块（融合 741_matrix_exponential）

功能：
- 使用 scaling-and-squaring + Padé 近似计算矩阵指数 exp(A)
- 将矩阵指数应用于PCE-Galerkin系统的时间精确积分
- 基于Moler & Van Loan (2003) 的算法

数学公式：
- exp(A) = (exp(A/2^s))^{2^s}
- Padé (q,q) 近似: R_{qq}(A) = D_{qq}(A)^{-1} N_{qq}(A)
  N_{qq}(A) = Σ_{k=0}^{q} c_k A^k,  D_{qq}(A) = Σ_{k=0}^{q} (-1)^k c_k A^k
  c_k = (2q-k)! q! / [(2q)! k! (q-k)!]
- 对于PCE系统 dU/dt = -A U，精确解: U(t) = exp(-A t) U(0)
"""

import numpy as np


def matrix_exponential_pade(A, q=6):
    """
    使用scaling-and-squaring和Padé近似计算矩阵指数 exp(A)。
    基于 r8mat_expm1.m 的核心思想，移植到Python/NumPy。
    
    参数:
        A: (n,n) 实方阵
        q: Padé近似的阶数（默认6）
    
    返回:
        E: exp(A) 的近似
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    
    # Scaling
    inf_norm = np.linalg.norm(A, ord=np.inf)
    if inf_norm < 1e-15:
        return np.eye(n)
    
    s = max(0, int(np.ceil(np.log2(inf_norm))))
    s = max(0, s + 1)
    A_scaled = A / (2.0 ** s)
    
    # Padé近似
    I = np.eye(n)
    c = 0.5
    E = I + c * A_scaled
    D = I - c * A_scaled
    X = A_scaled.copy()
    
    p = True
    for k in range(2, q + 1):
        c = c * (q - k + 1) / (k * (2 * q - k + 1))
        X = A_scaled @ X
        cX = c * X
        E = E + cX
        if p:
            D = D + cX
        else:
            D = D - cX
        p = not p
    
    # 解 D E' = E
    E = np.linalg.solve(D, E)
    
    # Squaring
    for _ in range(s):
        E = E @ E
    
    return E


def pce_matrix_exponential_step(A_pce, u, dt):
    """
    对PCE-Galerkin系统使用矩阵指数进行精确时间推进。
    u(t+dt) = exp(-A_pce * dt) @ u
    
    参数:
        A_pce: (N_pce, N_pce) PCE耦合矩阵
        u: (N_pce,) 当前PCE系数
        dt: 时间步长
    
    返回:
        u_new: 推进后的系数
    """
    # HOLE 2: 需要实现PCE系统的矩阵指数时间推进
    # 核心知识：
    #   - PCE系统的ODE为 dU/dt = -A_pce @ U
    #   - 精确解为 U(t+dt) = exp(-A_pce * dt) @ U(t)
    #   - 使用 matrix_exponential_pade 计算矩阵指数
    # 注意：矩阵指数的输入应为 -A_pce * dt
    raise NotImplementedError("HOLE 2: pce_matrix_exponential_step 待修复")


def pce_matrix_exponential_integrate(A_pce, u0, tf, nt):
    """
    使用矩阵指数对PCE系统做全时间积分。
    适用于中小规模PCE系统（degree <= 10）。
    
    返回:
        t: 时间数组
        U: (nt+1, n_pce) 系数历史
    """
    u = np.asarray(u0, dtype=float).copy()
    dt = tf / nt
    n_pce = len(u)
    U = np.zeros((nt + 1, n_pce))
    U[0] = u
    
    for i in range(nt):
        u = pce_matrix_exponential_step(A_pce, u, dt)
        U[i + 1] = u
    
    t = np.linspace(0, tf, nt + 1)
    return t, U
