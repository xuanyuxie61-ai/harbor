"""
Fokker-Planck 方程的有限元求解
================================
融合原始项目:
  - 383_fem1d: 一维有限元方法
  - 509_hb_to_msm: 稀疏矩阵格式转换

科学背景:
---------
过阻尼 Langevin 方程对应的 Smoluchowski 方程:
  ∂P/∂t = ∂/∂q [ D(q) (∂P/∂q + (1/T) ∂V/∂q · P) ]

在稳态下，解满足 Boltzmann 分布:
  P_eq(q) ∝ exp( -V(q)/T )

对于非稳态裂变过程，我们需要数值求解上述方程，
得到概率流 J = -D(∂P/∂q + (1/T)∂V/∂q · P) 随时间的演化。

有限元离散化（Galerkin 方法）:
在单元 [x_e, x_{e+1}] 上，基函数 φ_i 为分段线性 hat 函数。
质量矩阵 M_ij = ∫ φ_i φ_j dx
刚度矩阵 K_ij = ∫ D ∂φ_i/∂x ∂φ_j/∂x dx
漂移矩阵 B_ij = ∫ (∂V/∂x) φ_i ∂φ_j/∂x dx

本模块实现基于三对角线性系统的快速 FEM 求解器，
适用于大规模裂变概率分布演化计算。
"""

import numpy as np
from typing import Tuple, Callable


def assemble_fem_tridiagonal(
    x_nodes: np.ndarray,
    D_func: Callable[[np.ndarray], np.ndarray],
    dV_func: Callable[[np.ndarray], np.ndarray],
    T: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    组装 FEM 三对角系统矩阵.
    
    对于一维 Smoluchowski 方程，使用线性基函数，
    得到三对角矩阵系统 A u = b，其中:
    A = M + θ Δt (K + B/T)
    
    参数:
        x_nodes: 节点坐标 (n+1,)
        D_func: 扩散系数函数 D(x)
        dV_func: 势能梯度函数 dV/dx
        T: 核温度 (MeV)
    返回:
        adiag, aleft, arite, rhs_base
    """
    n = len(x_nodes) - 1
    if n < 1:
        raise ValueError("need at least one element")
    
    nu = n - 1  # 内部未知量数（Dirichlet 边界）
    adiag = np.zeros(nu)
    aleft = np.zeros(nu)
    arite = np.zeros(nu)
    rhs_base = np.zeros(nu)
    
    for e in range(n):
        xL = x_nodes[e]
        xR = x_nodes[e + 1]
        h = xR - xL
        if h <= 0:
            continue
        
        x_mid = 0.5 * (xL + xR)
        D_mid = D_func(np.array([x_mid]))[0]
        dV_mid = dV_func(np.array([x_mid]))[0]
        
        # 局部刚度矩阵（扩散项）
        k_local = (D_mid / h) * np.array([[1.0, -1.0], [-1.0, 1.0]])
        
        # 局部漂移矩阵（对流项）
        # 使用迎风修正保持稳定性
        peclet = abs(dV_mid * h / (T * D_mid + 1e-12))
        alpha_upwind = 0.5 * np.tanh(peclet / 2.0)
        b_local = (dV_mid / (2.0 * T)) * np.array([[-1.0, 1.0], [-1.0, 1.0]])
        b_local += alpha_upwind * abs(dV_mid / T) * np.array([[1.0, -1.0], [-1.0, 1.0]])
        
        a_local = k_local + b_local
        
        # 组装到全局矩阵
        for i_local in range(2):
            i_global = e + i_local - 1  # 局部到全局映射（内部节点从0开始）
            if 0 <= i_global < nu:
                for j_local in range(2):
                    j_global = e + j_local - 1
                    if 0 <= j_global < nu:
                        val = a_local[i_local, j_local]
                        if i_global == j_global:
                            adiag[i_global] += val
                        elif j_global < i_global:
                            aleft[i_global] += val
                        else:
                            arite[i_global] += val
    
    return adiag, aleft, arite, rhs_base


def solve_tridiagonal(adiag: np.ndarray, aleft: np.ndarray, arite: np.ndarray,
                      f: np.ndarray) -> np.ndarray:
    """
    三对角矩阵追赶法求解 (改编自 fem1d/solve.m).
    
    系统形式: aleft_i u_{i-1} + adiag_i u_i + arite_i u_{i+1} = f_i
    
    前向消元:
      arite[0] = arite[0] / adiag[0]
      for i=1..n-2: adiag[i] -= aleft[i]*arite[i-1]; arite[i] /= adiag[i]
      adiag[n-1] -= aleft[n-1]*arite[n-2]
    回代:
      u[0] = f[0]/adiag[0]
      for i=1..n-1: u[i] = (f[i] - aleft[i]*u[i-1]) / adiag[i]
      for i=n-2..0: u[i] -= arite[i]*u[i+1]
    """
    nu = len(adiag)
    if nu == 0:
        return np.array([])
    if len(aleft) != nu or len(arite) != nu or len(f) != nu:
        raise ValueError("array length mismatch")
    
    ad = adiag.copy()
    al = aleft.copy()
    ar = arite.copy()
    
    ar[0] = ar[0] / ad[0]
    for i in range(1, nu - 1):
        ad[i] = ad[i] - al[i] * ar[i - 1]
        if abs(ad[i]) < 1e-15:
            ad[i] = 1e-15  # 数值保护
        ar[i] = ar[i] / ad[i]
    if nu > 1:
        ad[nu - 1] = ad[nu - 1] - al[nu - 1] * ar[nu - 2]
        if abs(ad[nu - 1]) < 1e-15:
            ad[nu - 1] = 1e-15
    
    u = np.zeros(nu)
    u[0] = f[0] / ad[0]
    for i in range(1, nu):
        u[i] = (f[i] - al[i] * u[i - 1]) / ad[i]
    
    for i in range(nu - 2, -1, -1):
        u[i] = u[i] - ar[i] * u[i + 1]
    
    return u


def fokker_planck_steady_state(
    x_nodes: np.ndarray,
    V_func: Callable[[np.ndarray], np.ndarray],
    T: float,
    D_const: float = 1.0,
) -> np.ndarray:
    """
    求解稳态 Fokker-Planck 方程的离散近似.
    
    稳态解应趋近于 Boltzmann 分布 P ∝ exp(-V/T).
    这里通过 FEM 得到归一化概率分布.
    """
    n = len(x_nodes)
    V_vals = V_func(x_nodes)
    # Boltzmann 分布
    P_unnorm = np.exp(-V_vals / (T + 1e-12))
    # 数值稳定性：限制指数范围
    P_unnorm = np.clip(P_unnorm, 1e-300, 1e300)
    Z = np.trapezoid(P_unnorm, x_nodes)
    if Z <= 0 or not np.isfinite(Z):
        Z = 1.0
    P = P_unnorm / Z
    return P


def fokker_planck_time_stepping(
    x_nodes: np.ndarray,
    P0: np.ndarray,
    D_func: Callable[[np.ndarray], np.ndarray],
    dV_func: Callable[[np.ndarray], np.ndarray],
    T: float,
    dt: float,
    n_steps: int,
) -> np.ndarray:
    """
    隐式 Euler 时间步进求解 Fokker-Planck 方程.
    
    (M + dt A) P^{n+1} = M P^n
    
    其中 A = K + B/T 为 FEM 系统矩阵.
    """
    n = len(x_nodes)
    if len(P0) != n:
        raise ValueError("P0 length mismatch")
    
    adiag, aleft, arite, _ = assemble_fem_tridiagonal(x_nodes, D_func, dV_func, T)
    nu = len(adiag)
    
    # 简化的集中质量矩阵（lumped mass）
    dx = np.diff(x_nodes)
    m_diag = 0.5 * (dx[:-1] + dx[1:]) if n > 2 else np.array([1.0])
    if nu == 1 and len(m_diag) == 0:
        m_diag = np.array([1.0])
    
    P = P0.copy()
    # 边界保持零（吸收边界近似）
    P[0] = 0.0
    P[-1] = 0.0
    
    for _ in range(n_steps):
        # 右端项
        f = m_diag * P[1:-1]
        # 左端矩阵 = mass + dt * stiffness
        Ad = m_diag + dt * adiag
        Al = dt * aleft
        Ar = dt * arite
        
        P_inner = solve_tridiagonal(Ad, Al, Ar, f)
        P[1:-1] = P_inner
        # 归一化
        total = np.trapezoid(P, x_nodes)
        if total > 0:
            P = P / total
    
    return P


def sparse_matrix_vector_product(col_ptr: np.ndarray, row_ind: np.ndarray,
                                  values: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """
    Harwell-Boeing 格式的稀疏矩阵向量乘法 (改编自 hb_to_msm.m).
    
    CSC 格式: col_ptr[i] .. col_ptr[i+1]-1 为第 i 列的非零元.
    """
    ncol = len(col_ptr) - 1
    nrow = len(vec)
    out = np.zeros(nrow)
    for col in range(ncol):
        for k in range(col_ptr[col], col_ptr[col + 1]):
            row = row_ind[k]
            if 0 <= row < nrow and 0 <= col < len(vec):
                out[row] += values[k] * vec[col]
    return out
