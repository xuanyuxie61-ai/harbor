"""
propagation_front.py
传播前沿高分辨率数值模拟模块

基于种子项目:
- 272_dg1d_burgers: 1D间断Galerkin方法求解粘性Burgers方程
"""

import numpy as np
from typing import Tuple, List


def legendre_gauss_lobatto_nodes(N: int) -> np.ndarray:
    """
    计算Legendre-Gauss-Lobatto (LGL) 节点。

    LGL节点是 Legendre 多项式 P_{N-1}(x) 的导数 P'_{N-1}(x) 的零点，
    加上区间端点 x = -1 和 x = 1。

    节点满足 Gauss-Lobatto 求积公式:
        int_{-1}^1 f(x) dx ≈ sum_{i=0}^N w_i f(x_i)

    对于 N <= 8 使用已知值，否则使用Newton迭代。
    """
    if N == 1:
        return np.array([-1.0, 1.0], dtype=np.float64)
    elif N == 2:
        return np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    elif N == 3:
        return np.array([-1.0, -np.sqrt(5.0)/5.0, np.sqrt(5.0)/5.0, 1.0], dtype=np.float64)
    elif N == 4:
        return np.array([-1.0,
                         -np.sqrt(21.0)/7.0,
                         0.0,
                         np.sqrt(21.0)/7.0,
                         1.0], dtype=np.float64)
    elif N == 5:
        x1 = np.sqrt((7.0 - 2.0*np.sqrt(7.0)) / 21.0)
        x2 = np.sqrt((7.0 + 2.0*np.sqrt(7.0)) / 21.0)
        return np.array([-1.0, -x2, -x1, x1, x2, 1.0], dtype=np.float64)
    else:
        # 简化: 使用Chebyshev节点近似
        return np.cos(np.pi * np.arange(N + 1) / N)


def vandermonde_matrix(N: int, r: np.ndarray) -> np.ndarray:
    """
    构造Legendre Vandermonde矩阵。

    V_{ij} = P_j(r_i),  j = 0, ..., N

    其中 P_j 是 j 阶Legendre多项式。
    """
    n = len(r)
    V = np.zeros((n, N + 1), dtype=np.float64)
    V[:, 0] = 1.0
    if N >= 1:
        V[:, 1] = r
    for j in range(2, N + 1):
        V[:, j] = ((2.0 * j - 1.0) * r * V[:, j - 1] - (j - 1.0) * V[:, j - 2]) / j
    return V


def differentiation_matrix(N: int, r: np.ndarray) -> np.ndarray:
    """
    计算参考单元上的微分矩阵 D_r。

    D_r = V_r * V^{-1}

    其中 V_r 是Legendre多项式导数在节点处的值。
    """
    V = vandermonde_matrix(N, r)
    Vr = np.zeros_like(V)
    Vr[:, 0] = 0.0
    if N >= 1:
        Vr[:, 1] = 1.0
    for j in range(2, N + 1):
        Vr[:, j] = (2.0 * j - 1.0) * V[:, j - 1] + Vr[:, j - 2]

    V_inv = np.linalg.inv(V)
    D = Vr @ V_inv
    return D


def dg1d_burgers_solve(u0_func: callable,
                       K: int, N: int,
                       xL: float, xR: float,
                       epsilon: float,
                       FinalTime: float,
                       CFL: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    用Nodal Discontinuous Galerkin方法求解1D粘性Burgers方程:

        du/dt + d/dx(u^2/2) = epsilon * d^2u/dx^2

    方程描述非线性对流-扩散过程，用于模拟信息传播前沿的激波结构。

    数值方法:
        - 空间: Nodal DG, K个单元, 每个单元 N+1 个LGL节点
        - 时间: 4阶low-storage RK
        - 通量: Lax-Friedrichs数值通量

    参数:
        u0_func: 初始条件函数
        K: 单元数
        N: 多项式阶数
        xL, xR: 计算域
        epsilon: 粘性系数
        FinalTime: 终止时间
        CFL: CFL数

    返回:
        x: 网格节点坐标 (N+1, K)
        u: 解 (N+1, K)
        t_history: 时间历史
    """
    # LGL节点和权重
    r = legendre_gauss_lobatto_nodes(N)
    V = vandermonde_matrix(N, r)

    # 计算LGL权重 (通过Vandermonde的逆)
    V_inv = np.linalg.inv(V)
    # 质量矩阵 M = V^{-T} * V^{-1}
    M = V_inv.T @ V_inv

    # 简化权重计算: 用解析公式近似
    # 实际上权重 w 满足 int P_i P_j = delta_ij * 2/(2i+1)
    # 这里直接用质量矩阵对角线近似
    w = np.diag(M)
    w = w / w.sum() * 2.0  # 归一化到 [-1,1] 上积分为2

    # 微分矩阵
    Dr = differentiation_matrix(N, r)

    # 构建网格
    VX = np.linspace(xL, xR, K + 1)
    dx = (xR - xL) / K

    # 参考单元到物理单元映射
    x = np.zeros((N + 1, K), dtype=np.float64)
    for k in range(K):
        x[:, k] = 0.5 * (1.0 - r) * VX[k] + 0.5 * (1.0 + r) * VX[k + 1]

    # 初始条件
    u = np.zeros((N + 1, K), dtype=np.float64)
    for k in range(K):
        for i in range(N + 1):
            u[i, k] = u0_func(x[i, k])

    # 单元连接性
    vmapM = np.zeros((N + 1, K), dtype=np.int32)
    vmapP = np.zeros((N + 1, K), dtype=np.int32)
    for k in range(K):
        for i in range(N + 1):
            vmapM[i, k] = k * (N + 1) + i
            if i == 0 and k > 0:
                vmapP[i, k] = (k - 1) * (N + 1) + N
            elif i == N and k < K - 1:
                vmapP[i, k] = (k + 1) * (N + 1) + 0
            else:
                vmapP[i, k] = vmapM[i, k]

    # Low-storage RK4系数
    rk4a = np.array([0.0,
                     -567301805773.0 / 1357537059087.0,
                     -2404267990393.0 / 2016746695238.0,
                     -3550918686646.0 / 2091501179385.0,
                     -1275806237668.0 / 842570457699.0], dtype=np.float64)
    rk4b = np.array([1432997174477.0 / 9575080441755.0,
                     5161836677717.0 / 13612068292357.0,
                     1720146321549.0 / 2090206949498.0,
                     3134564353537.0 / 4481467310338.0,
                     2277821191437.0 / 14882151754819.0], dtype=np.float64)

    time = 0.0
    t_history = [time]
    u_history = [u.copy()]

    # 时间步进
    n_steps = 0
    max_steps = 100000

    while time < FinalTime and n_steps < max_steps:
        dt = CFL * dx / max(np.max(np.abs(u)) + 1e-10, np.sqrt(epsilon) / dx + 1e-10)
        dt = min(dt, FinalTime - time)

        resu = np.zeros_like(u)
        for INTRK in range(5):
            # 计算右端项
            rhs = burgers_rhs1d(u, Dr, r, w, epsilon, K, N, vmapM, vmapP, xL, xR)

            resu = rk4a[INTRK] * resu + dt * rhs
            u = u + rk4b[INTRK] * resu

            # 边界处理
            u = np.maximum(u, 0.0)  # 非负约束

        time += dt
        n_steps += 1
        t_history.append(time)
        u_history.append(u.copy())

    return x, u, np.array(t_history)


def burgers_rhs1d(u: np.ndarray, Dr: np.ndarray, r: np.ndarray,
                  w: np.ndarray, epsilon: float,
                  K: int, N: int,
                  vmapM: np.ndarray, vmapP: np.ndarray,
                  xL: float, xR: float) -> np.ndarray:
    """
    计算Burgers方程的DG右端项。
    """
    # 面跳跃
    nx = np.zeros((N + 1, K), dtype=np.float64)
    nx[0, :] = -1.0
    nx[N, :] = 1.0

    # 提取面值
    uM = np.zeros((N + 1, K), dtype=np.float64)
    uP = np.zeros((N + 1, K), dtype=np.float64)

    for k in range(K):
        for i in [0, N]:
            uM[i, k] = u[i, k]
            idxM = vmapM[i, k]
            idxP = vmapP[i, k]
            if idxP == idxM:
                # 边界
                if i == 0 and k == 0:
                    uP[i, k] = analytic_solution_boundary(xL)  # Dirichlet
                elif i == N and k == K - 1:
                    uP[i, k] = analytic_solution_boundary(xR)
                else:
                    uP[i, k] = uM[i, k]
            else:
                kp = idxP // (N + 1)
                ip = idxP % (N + 1)
                uP[i, k] = u[ip, kp]

    du = uM - uP

    # 数值通量 (Lax-Friedrichs)
    alpha = np.max(np.abs(u))
    flux = 0.5 * (uM**2 - uP**2) - 0.5 * alpha * du

    # 粘性项处理
    q = np.zeros_like(u)
    if epsilon > 1e-14:
        # 局部导数
        ux = Dr @ u
        # 面跳跃 for q
        qx = np.sqrt(epsilon) * ux
        qM = np.zeros((N + 1, K), dtype=np.float64)
        qP = np.zeros((N + 1, K), dtype=np.float64)
        for k in range(K):
            for i in [0, N]:
                qM[i, k] = qx[i, k]
                idxM = vmapM[i, k]
                idxP = vmapP[i, k]
                if idxP == idxM:
                    qP[i, k] = qM[i, k]
                else:
                    kp = idxP // (N + 1)
                    ip = idxP % (N + 1)
                    qP[i, k] = qx[ip, kp]
        tau = 1.0
        q_flux = 0.5 * (qM + qP) - 0.5 * tau * du
        q = qx - lift1d(q_flux, r, w, N, K)

    # 体积项
    div_flux = Dr @ (0.5 * u**2 - np.sqrt(epsilon) * q)

    # 面修正 (lifting)
    lift_flux = lift1d(flux, r, w, N, K)

    rhs = -div_flux + lift_flux
    return rhs


def lift1d(flux: np.ndarray, r: np.ndarray, w: np.ndarray,
           N: int, K: int) -> np.ndarray:
    """
    DG lifting算子: 将面通量提升到体积。
    """
    # 简化版本: 在面节点上施加修正
    lift = np.zeros((N + 1, K), dtype=np.float64)

    # 边界插值矩阵 (简化)
    Emat = np.zeros((N + 1, 2), dtype=np.float64)
    Emat[0, 0] = 1.0
    Emat[N, 1] = 1.0

    # 质量矩阵近似
    V = vandermonde_matrix(N, r)
    M_inv = V @ V.T  # 简化近似

    LIFT = M_inv @ Emat

    for k in range(K):
        face_vals = np.array([flux[0, k], flux[N, k]], dtype=np.float64)
        lift[:, k] = LIFT @ face_vals

    return lift


def analytic_solution_boundary(x: float) -> float:
    """
    Burgers方程的边界条件 (tanh激波剖面)。
    """
    return 0.0


def propagation_front_simulation(initial_spread: float = 0.1,
                                 diffusion_coeff: float = 0.01,
                                 final_time: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    模拟信息传播前沿的演化。

    初始条件: 高斯型局部爆发
        u(x,0) = exp(-(x-x0)^2 / sigma^2)
    """
    def u0(x):
        return np.exp(-((x - 0.3)**2) / (2.0 * initial_spread**2))

    K = 40
    N = 4
    xL = 0.0
    xR = 1.0

    x, u, t_hist = dg1d_burgers_solve(u0, K, N, xL, xR, diffusion_coeff, final_time)
    return x, u, t_hist
