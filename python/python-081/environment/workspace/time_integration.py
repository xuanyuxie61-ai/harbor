"""
结构动力学时间积分模块
======================
基于种子项目:
  - 833_ode_trapezoidal: 隐式梯形法ODE求解器
  - 1059_sawtooth_ode: 锯齿波驱动谐振子

科学背景:
  在大变形结构动力学中，半离散化后的运动方程为:
      M ü + C u̇ + R(u) = F_ext(t)
  其中 M 为质量矩阵，C 为阻尼矩阵，R(u) 为非线性内力。

  本模块实现:
  1. Newmark-β法 (隐式，无条件稳定，当 β=1/4, γ=1/2 时等价于
     平均加速度法，具有二阶精度)
  2. 隐式梯形法 (Trapezoidal rule，用于对比)
  3. 锯齿波周期冲击载荷生成 (模拟机械冲击、往复载荷)

关键公式:
  Newmark-β预测-校正格式:
    u_{n+1} = u_n + Δt v_n + (Δt^2/2)[(1-2β)a_n + 2β a_{n+1}]
    v_{n+1} = v_n + Δt [(1-γ)a_n + γ a_{n+1}]

  有效刚度矩阵:
    K_eff = K_T + (1/(β Δt^2)) M + (γ/(β Δt)) C

  有效残差:
    R_eff = F_ext(t_{n+1}) - R(u_{n+1}) - M a_{n+1} - C v_{n+1}

  隐式梯形法(一阶系统):
    y_{n+1} = y_n + (Δt/2)[f(t_n, y_n) + f(t_{n+1}, y_{n+1})]
"""

import numpy as np
from typing import Tuple, Optional, Callable


def compute_lumped_mass_matrix(nodes: np.ndarray, elements: np.ndarray,
                                density: float = 1000.0) -> np.ndarray:
    """
    计算集中质量矩阵(对角化)。
    对每个单元计算质量 ρV，平均分配到4个节点。

    参数:
        nodes: (N, 3) 节点坐标
        elements: (E, 4) 单元连接表
        density: 材料密度 (kg/m^3)

    返回:
        M: (3N, 3N) 对角质量矩阵(稀疏存储为对角向量)
    """
    n_nodes = nodes.shape[0]
    mass = np.zeros(n_nodes, dtype=np.float64)

    for e in elements:
        x0, x1, x2, x3 = nodes[e[0]], nodes[e[1]], nodes[e[2]], nodes[e[3]]
        mat = np.vstack([x1 - x0, x2 - x0, x3 - x0])
        vol = abs(np.linalg.det(mat)) / 6.0
        m_e = density * vol
        for n in e:
            mass[n] += m_e / 4.0

    M_diag = np.zeros(3 * n_nodes, dtype=np.float64)
    for i in range(n_nodes):
        M_diag[3 * i:3 * i + 3] = mass[i]
    return M_diag


def compute_rayleigh_damping(M_diag: np.ndarray, K: np.ndarray,
                              alpha_ray: float = 0.0,
                              beta_ray: float = 0.0) -> np.ndarray:
    """
    Rayleigh阻尼矩阵: C = α M + β K
    由于 K 可能很大，这里返回对角近似 C_diag = α M_diag + β diag(K)。

    参数:
        M_diag: (3N,) 质量矩阵对角线
        K: (3N, 3N) 刚度矩阵
        alpha_ray: 质量比例阻尼系数
        beta_ray: 刚度比例阻尼系数

    返回:
        C_diag: (3N,) 阻尼矩阵对角近似
    """
    C_diag = alpha_ray * M_diag + beta_ray * np.diag(K)
    # 保证非负
    C_diag = np.maximum(C_diag, 0.0)
    return C_diag


def sawtooth_wave(t: float, period: float = 1.0,
                  amplitude: float = 1.0) -> float:
    """
    生成周期锯齿波: s(t) = A * (2 * frac(t/T) - 1)
    其中 frac 为小数部分。范围 [-A, A]。

    参数:
        t: 时间
        period: 周期 T
        amplitude: 振幅 A

    返回:
        s: 锯齿波值
    """
    if period <= 0:
        raise ValueError("周期必须为正")
    frac = (t / period) - np.floor(t / period)
    return amplitude * (2.0 * frac - 1.0)


def newmark_beta_step(u_n: np.ndarray, v_n: np.ndarray, a_n: np.ndarray,
                       dt: float, M_diag: np.ndarray, C_diag: np.ndarray,
                       compute_residual: Callable,
                       compute_stiffness: Callable,
                       beta_newmark: float = 0.25,
                       gamma_newmark: float = 0.5,
                       tol: float = 1e-8,
                       max_iter: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    执行一个Newmark-β时间步。

    参数:
        u_n, v_n, a_n: 当前位移、速度、加速度
        dt: 时间步长
        M_diag, C_diag: 对角质量/阻尼矩阵
        compute_residual: 函数(u) -> (R, F_ext) 计算残差和外部力
        compute_stiffness: 函数(u) -> K_T 计算切线刚度
        beta_newmark, gamma_newmark: Newmark参数

    返回:
        u_new, v_new, a_new, converged
    """
    n_dof = u_n.shape[0]
    # 预测
    u_pred = u_n + dt * v_n + (dt ** 2 / 2.0) * (1.0 - 2.0 * beta_newmark) * a_n
    v_pred = v_n + dt * (1.0 - gamma_newmark) * a_n

    # 初始猜测
    u_new = u_pred.copy()
    a_new = np.zeros(n_dof, dtype=np.float64)

    converged = False
    for it in range(max_iter):
        # 由 u_new 反推加速度
        a_new = (u_new - u_pred) / (beta_newmark * dt ** 2)
        v_new = v_pred + gamma_newmark * dt * a_new

        R_int, F_ext = compute_residual(u_new)
        # 动力学残差: R_dyn = F_ext - R_int - M a - C v
        R_dyn = F_ext - R_int - M_diag * a_new - C_diag * v_new

        if np.linalg.norm(R_dyn) < tol * (np.linalg.norm(F_ext) + 1.0):
            converged = True
            break

        K_T = compute_stiffness(u_new)
        # 有效刚度
        K_eff = K_T + (1.0 / (beta_newmark * dt ** 2)) * np.diag(M_diag) \
                + (gamma_newmark / (beta_newmark * dt)) * np.diag(C_diag)

        try:
            du = np.linalg.solve(K_eff, R_dyn)
        except np.linalg.LinAlgError:
            # 边界处理: 若奇异则使用伪逆
            du = np.linalg.lstsq(K_eff, R_dyn, rcond=None)[0]
        u_new += du

    if not converged:
        # 边界处理: 若不收敛，回退到预测值
        u_new = u_pred.copy()
        a_new = np.zeros(n_dof, dtype=np.float64)
        v_new = v_pred.copy()

    return u_new, v_new, a_new, converged


def trapezoidal_step(y_n: np.ndarray, t_n: float, dt: float,
                      f: Callable[[float, np.ndarray], np.ndarray],
                      max_inner_iter: int = 10,
                      tol: float = 1e-10) -> np.ndarray:
    """
    隐式梯形法单步推进:
      y_{n+1} = y_n + (dt/2) * [f(t_n, y_n) + f(t_{n+1}, y_{n+1})]

    对隐式方程使用不动点迭代求解。

    参数:
        y_n: 当前状态向量
        t_n: 当前时间
        dt: 时间步长
        f: 右端函数 f(t, y)
        max_inner_iter: 内层不动点迭代次数
        tol: 收敛容差

    返回:
        y_next: 下一时刻状态
    """
    t_next = t_n + dt
    f_n = f(t_n, y_n)
    y_next = y_n + dt * f_n  # 显式Euler作为初值

    for _ in range(max_inner_iter):
        f_next = f(t_next, y_next)
        y_new = y_n + 0.5 * dt * (f_n + f_next)
        if np.linalg.norm(y_new - y_next) < tol:
            y_next = y_new
            break
        y_next = y_new

    return y_next
