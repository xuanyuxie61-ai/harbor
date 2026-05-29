"""
nonlinear_solver.py
===================
非线性反应-扩散方程的牛顿迭代求解器。

基于种子项目 125_burgers_steady_viscous 重构：
- 原项目使用牛顿法求解稳态粘性 Burgers 方程；
- 该方程形式为：u u' = ν u''，兼具非线性对流与扩散项。

在本系统中，将其核心牛顿迭代框架迁移到催化剂颗粒内的
耦合非线性反应-扩散方程组：

    残差向量 F(U) = [F_C; F_T]
    雅可比矩阵 J = [∂F_C/∂C, ∂F_C/∂T;
                    ∂F_T/∂C, ∂F_T/∂T]

    牛顿步：J ΔU = -F(U)
           U^{k+1} = U^{k} + ω ΔU

其中阻尼因子 ω ∈ (0,1] 用于保证收敛。

同时包含一维有限差分离散的非线性 BVP 求解，
以及伪瞬态延续（pseudo-transient continuation）策略，
用于处理强非线性导致的收敛困难。
"""

import numpy as np
from linear_solvers import solve_sparse_system, solve_tridiagonal


class NonlinearSolverError(Exception):
    """非线性求解异常。"""
    pass


def newton_solve_burgers_style(u_init, residual_func, jacobian_func,
                                max_iter=50, tol=1e-10, damping=1.0,
                                min_damping=0.1):
    """
    基于 burgers_steady_viscous 的牛顿迭代求解器。

    求解非线性方程组 F(u) = 0。

    算法流程：
        1. 初始化 u = u_init
        2. 计算残差 F(u) 与雅可比 J(u)
        3. 解线性系统 J Δu = -F
        4. 线搜索确定阻尼因子 ω
        5. 更新 u = u + ω Δu
        6. 重复直到 ||F|| < tol 或达到最大迭代次数

    Parameters
    ----------
    u_init : ndarray
        初始猜测。
    residual_func : callable
        f(u) -> ndarray，计算残差。
    jacobian_func : callable
        f(u) -> ndarray (n, n)，计算雅可比矩阵。
    max_iter : int
    tol : float
    damping : float
        初始阻尼因子。
    min_damping : float
        最小阻尼因子。

    Returns
    -------
    u : ndarray
        收敛解。
    info : dict
        包含迭代次数、最终残差、收敛标志。
    """
    u = np.asarray(u_init, dtype=float).copy()
    n = u.size

    for it in range(max_iter):
        F = residual_func(u)
        f_norm = np.linalg.norm(F, np.inf)

        if f_norm < tol:
            return u, {"converged": True, "iter": it, "resid": f_norm}

        J = jacobian_func(u)

        # 解牛顿步
        try:
            du = solve_sparse_system(J, -F)
        except Exception as exc:
            # 退化为阻尼最小二乘或伪逆
            du = np.linalg.lstsq(J, -F, rcond=None)[0]

        du_norm = np.linalg.norm(du, np.inf)
        u_norm = np.linalg.norm(u, np.inf)
        if du_norm < tol * (u_norm + 1.0):
            return u, {"converged": True, "iter": it, "resid": f_norm}

        # 线搜索
        omega = damping
        while omega >= min_damping:
            u_new = u + omega * du
            F_new = residual_func(u_new)
            f_new_norm = np.linalg.norm(F_new, np.inf)
            if f_new_norm < f_norm:
                u = u_new
                break
            omega *= 0.5
        else:
            # 线搜索失败，接受最小步长
            u = u + min_damping * du

    return u, {"converged": False, "iter": max_iter, "resid": f_norm}


def solve_coupled_diffusion_reaction_newton(r_nodes, D_e, lambda_eff,
                                             kinetics_model, particle_model,
                                             max_iter=50, tol=1e-8):
    """
    块 Gauss-Seidel / 阻尼牛顿法求解耦合的浓度-温度扩散-反应方程组。

    方程组（球形颗粒）：
        D_e (C'' + 2/r C') - R(C, T) = 0
        λ_eff (T'' + 2/r T') + (-ΔH) R(C, T) = 0

    采用块迭代策略（比全牛顿法更鲁棒）：
        1. 固定 T，用有限体积法求解 C（线性化迭代）
        2. 用更新的 C 计算反应热，求解温度方程（线性）
        3. 重复直到收敛

    Parameters
    ----------
    r_nodes : ndarray
        径向节点。
    D_e : float
        有效扩散系数 [m²/s]。
    lambda_eff : float
        有效导热系数 [W/(m·K)]。
    kinetics_model : object
        动力学模型（rate 方法）。
    particle_model : object
        颗粒模型（包含边界条件参数）。

    Returns
    -------
    C : ndarray
    T : ndarray
    info : dict
    """
    n = r_nodes.size
    if n < 3:
        raise NonlinearSolverError("节点数至少为 3")

    C_surf = particle_model.C_surface_A
    T_surf = particle_model.T_surface

    # 初始猜测
    C = np.linspace(C_surf * 0.85, C_surf, n)
    T = np.linspace(T_surf + 20.0, T_surf, n)
    C[-1] = C_surf
    T[-1] = T_surf

    for it in range(max_iter):
        # TODO: Hole 2 — 实现耦合浓度-温度方程的块 Gauss-Seidel 迭代
        # 要求：
        # 1. 固定 T，对 C 方程构造三对角系统并求解：
        #      D_e (C'' + 2/r C') - R(C, T) = 0
        #    使用与 Hole 1 一致的有限体积离散（球坐标守恒形式）。
        #    r=0 边界：a_diag[0] = 6*D_e/dr0², c_sup[0] = -6*D_e/dr0²
        #    r=R 边界：C(R) = C_surf（Dirichlet）
        #    rhs_C[i] = -kinetics_model.rate(Ci, C_surface_B, Ti)
        #    求解后：C_new = solve_tridiagonal(...)，并 clip 到 [0, C_surf*1.01]
        # 2. 固定 C_new，对 T 方程构造三对角系统并求解：
        #      λ_eff (T'' + 2/r T') + (-ΔH) R(C, T) = 0
        #    使用 λ_eff 代替 D_e 的相同离散格式。
        #    rhs_T[i] = R_local * (-heat_of_reaction)
        #    r=0 边界：a_diag_T[0] = 6*λ_eff/dr0², c_sup_T[0] = -6*λ_eff/dr0²
        #    r=R 边界：T(R) = T_surf
        #    求解后：T_new = solve_tridiagonal(...)，并 clip 到 [T_surf-50, T_surf+200]
        # 3. 注意：C 和 T 的 r=0 边界处理必须与 Hole 1 的数学原理一致
        raise NotImplementedError("Hole 2: 请实现耦合 C-T 方程的块迭代离散与求解")

        # ---- 收敛检查 ----
        change_C = np.linalg.norm(C_new - C) / max(np.linalg.norm(C), 1e-12)
        change_T = np.linalg.norm(T_new - T) / max(np.linalg.norm(T), 1e-12)
        C = 0.6 * C_new + 0.4 * C
        T = 0.6 * T_new + 0.4 * T

        if max(change_C, change_T) < tol:
            return C, T, {"converged": True, "iter": it + 1, "resid": max(change_C, change_T)}

    return C, T, {"converged": False, "iter": max_iter, "resid": max(change_C, change_T)}


def pseudo_transient_continuation(r_nodes, D_e, lambda_eff,
                                   kinetics_model, particle_model,
                                   dt_init=1e-6, dt_max=1.0, t_final=100.0):
    """
    伪瞬态延续法（Pseudo-Transient Continuation）。

    将稳态问题转化为瞬态问题：
        ∂C/∂t = D_e ∇²C - R(C,T)
        ∂T/∂t = λ_eff ∇²T + (-ΔH)R(C,T)

    使用大时间步长推进到稳态，适合强非线性问题。

    Parameters
    ----------
    dt_init : float
        初始时间步。
    dt_max : float
        最大时间步。
    t_final : float
        最大伪时间。

    Returns
    -------
    C, T : ndarray
    info : dict
    """
    n = r_nodes.size
    C = np.ones(n) * particle_model.C_surface_A * 0.5
    T = np.ones(n) * particle_model.T_surface
    C[-1] = particle_model.C_surface_A
    T[-1] = particle_model.T_surface

    t = 0.0
    dt = dt_init
    step = 0

    while t < t_final and step < 10000:
        # 构造隐式欧拉系统（简化：只对角项隐式）
        a_diag = np.ones(n) / dt
        b_sub = np.zeros(n - 1)
        c_sup = np.zeros(n - 1)
        rhs_C = C / dt
        rhs_T = T / dt

        for i in range(1, n - 1):
            rm = r_nodes[i]
            dr_p = r_nodes[i + 1] - r_nodes[i]
            dr_m = r_nodes[i] - r_nodes[i - 1]
            r_plus = 0.5 * (r_nodes[i] + r_nodes[i + 1])
            r_minus = 0.5 * (r_nodes[i] + r_nodes[i - 1])
            vol = rm ** 2 * 0.5 * (dr_p + dr_m)

            a_diag[i] += (r_plus ** 2 * D_e / dr_p + r_minus ** 2 * D_e / dr_m) / vol / dt * 0.0  # 显式处理
            # 实际使用显式欧拉简化
            Ci = max(float(C[i]), 0.0)
            Ti = max(float(T[i]), 200.0)
            R_local = kinetics_model.rate(Ci, particle_model.C_surface_B, Ti)
            rhs_C[i] += -R_local
            rhs_T[i] += R_local * (-particle_model.heat_of_reaction) / (1000.0)  # 热容简化

        # 边界
        a_diag[0] = 1.0
        rhs_C[0] = C[1]  # 近似对称
        rhs_T[0] = T[1]
        a_diag[-1] = 1.0
        rhs_C[-1] = particle_model.C_surface_A
        rhs_T[-1] = particle_model.T_surface

        # 简化的显式更新
        C_new = solve_tridiagonal(a_diag, b_sub, c_sup, rhs_C)
        T_new = solve_tridiagonal(a_diag, b_sub, c_sup, rhs_T)

        change = np.linalg.norm(C_new - C) / max(np.linalg.norm(C), 1e-12)
        C = C_new
        T = T_new
        t += dt
        step += 1

        dt = min(dt * 1.1, dt_max)
        if change < 1e-8:
            break

    return C, T, {"steps": step, "time": t, "change": change}
