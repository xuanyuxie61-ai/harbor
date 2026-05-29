"""
dynamic_integrator.py
博士级大变形非线性有限元分析 — 动态时间积分模块

融合原项目:
  - 833_ode_trapezoidal: 梯形法隐式 ODE 求解
  - 1059_sawtooth_ode: 锯齿波周期性荷载驱动

核心数学:
  1. 半离散化运动方程:
     M a_{n+1} + C v_{n+1} + R(u_{n+1}) = F_{ext}(t_{n+1})
     其中:
       M: 一致质量矩阵
       C: Rayleigh 阻尼矩阵 C = α_M M + β_K K_T
       R(u): 内力向量（非线性函数）
       F_ext: 外力向量
       u, v, a: 位移、速度、加速度

  2. Newmark-β 方法（梯形法特例 β=1/4, γ=1/2）:
     u_{n+1} = u_n + Δt v_n + (Δt^2/2)[(1-2β)a_n + 2β a_{n+1}]
     v_{n+1} = v_n + Δt [(1-γ)a_n + γ a_{n+1}]

     当 β=1/4, γ=1/2 时，等价于平均加速度法（无条件稳定）

  3. 隐式梯形法则（等价于 Newmark 平均加速度法）:
     将运动方程在 [t_n, t_{n+1}] 上积分，采用梯形近似:
       v_{n+1} = v_n + (Δt/2)(a_n + a_{n+1})
       u_{n+1} = u_n + (Δt/2)(v_n + v_{n+1})

     联立得:
       a_{n+1} = (4/(Δt^2))(u_{n+1} - u_n) - (4/Δt)v_n - a_n
       v_{n+1} = (2/Δt)(u_{n+1} - u_n) - v_n

     代入运动方程得到关于 u_{n+1} 的非线性方程:
       R̃(u_{n+1}) = M a_{n+1}(u) + C v_{n+1}(u) + R(u) - F_{ext}(t_{n+1}) = 0

     使用 Newton-Raphson 迭代:
       K̃_T Δu = -R̃
       其中有效刚度:
         K̃_T = (4/Δt^2)M + (2/Δt)C + K_T(u_{n+1}^{(i)})

  4. 锯齿波周期性荷载:
     f_sawtooth(t; T, A) = (2A/T) * (t mod T) - A
     用于模拟循环加载（如地震、机械振动）
"""

import numpy as np


class DynamicIntegratorError(Exception):
    pass


def sawtooth_wave(t, period, amplitude):
    """
    锯齿波函数

    源自原项目 1059_sawtooth_ode (sawtooth_driver)

    数学:
      s(t) = (2A/T) * (t mod T) - A
      范围: [-A, A]
    """
    t_mod = t % period
    return (2.0 * amplitude / period) * t_mod - amplitude


def compute_rayleigh_damping(M, K_T, alpha_m, beta_k):
    """
    计算 Rayleigh 阻尼矩阵

    数学:
      C = α_M * M + β_K * K_T
    """
    return alpha_m * M + beta_k * K_T


def newmark_predictor(u_n, v_n, a_n, dt, beta=0.25, gamma=0.5):
    """
    Newmark 预测步

    数学:
      u_pred = u_n + dt v_n + (dt^2/2)(1-2β) a_n
      v_pred = v_n + dt (1-γ) a_n
    """
    u_pred = u_n + dt * v_n + 0.5 * dt ** 2 * (1.0 - 2.0 * beta) * a_n
    v_pred = v_n + dt * (1.0 - gamma) * a_n
    return u_pred, v_pred


def newmark_corrector(u_pred, v_pred, du, dt, beta=0.25, gamma=0.5):
    """
    Newmark 修正步

    数学:
      u_{n+1} = u_pred + β dt^2 a_{n+1}
      这里简化为: u_{n+1} = u_pred + du
      a_{n+1} = du / (β dt^2)
      v_{n+1} = v_pred + γ dt a_{n+1}
    """
    u_new = u_pred + du
    a_new = du / (beta * dt ** 2)
    v_new = v_pred + gamma * dt * a_new
    return u_new, v_new, a_new


def trapezoidal_step(u_n, v_n, a_n, dt, M, C, compute_internal_force,
                     compute_tangent_stiffness, F_ext,
                     tol=1e-8, max_iter=20):
    """
    梯形法（平均加速度法）单步时间积分

    源自原项目 833_ode_trapezoidal (ode_trapezoidal)

    数学:
      有效刚度: K̃ = (4/Δt^2)M + (2/Δt)C + K_T
      初始预测: u^(0) = u_n + Δt v_n + (Δt^2/2) a_n
      迭代 i:
        R = M a(u^(i)) + C v(u^(i)) + R_int(u^(i)) - F_ext
        K̃^(i) = (4/Δt^2)M + (2/Δt)C + K_T(u^(i))
        Δu = solve(K̃^(i), -R)
        u^(i+1) = u^(i) + Δu

    输入:
        u_n, v_n, a_n: 当前时刻的位移、速度、加速度
        dt: 时间步长
        M, C: 质量矩阵、阻尼矩阵
        compute_internal_force: 函数，输入 u，返回内力向量 R_int
        compute_tangent_stiffness: 函数，输入 u，返回切线刚度 K_T
        F_ext: 当前时刻外力向量
    输出:
        u_new, v_new, a_new: 新时刻的位移、速度、加速度
        converged: 是否收敛
    """
    n_dof = len(u_n)
    coeff_a = 4.0 / (dt ** 2)
    coeff_v = 2.0 / dt

    # 预测步
    u = u_n + dt * v_n + 0.5 * dt ** 2 * a_n

    for it in range(max_iter):
        R_int, K_T = compute_internal_force(u), compute_tangent_stiffness(u)

        # 由 u 反推 a, v
        a = coeff_a * (u - u_n) - coeff_v * v_n - a_n
        v = coeff_v * (u - u_n) - v_n

        # 残差
        residual = M @ a + C @ v + R_int - F_ext
        res_norm = np.linalg.norm(residual)

        if res_norm < tol:
            return u, v, a, True

        # 有效刚度
        K_eff = coeff_a * M + coeff_v * C + K_T

        # 求解修正量
        try:
            du = np.linalg.solve(K_eff, -residual)
        except np.linalg.LinAlgError:
            # 奇异时加阻尼
            K_eff += 1e-6 * np.eye(n_dof) * np.max(np.abs(K_eff))
            du = np.linalg.solve(K_eff, -residual)

        u = u + du

    # 未收敛，返回最后结果
    a = coeff_a * (u - u_n) - coeff_v * v_n - a_n
    v = coeff_v * (u - u_n) - v_n
    return u, v, a, False


def dynamic_analysis_trapezoidal(u0, v0, a0, t_span, n_steps,
                                  M, C_func, compute_internal_force,
                                  compute_tangent_stiffness, F_ext_func,
                                  tol=1e-8, max_iter=20):
    """
    完整的动态分析（梯形法）

    输入:
        u0, v0, a0: 初始位移、速度、加速度
        t_span: (t_start, t_end)
        n_steps: 时间步数
        M: 质量矩阵
        C_func: 函数，输入 K_T，返回阻尼矩阵 C
        compute_internal_force: 函数，输入 u，返回 R_int
        compute_tangent_stiffness: 函数，输入 u，返回 K_T
        F_ext_func: 函数，输入 t，返回外力向量
    输出:
        t_array: (n_steps+1,) 时间序列
        u_history: (n_steps+1, n_dof) 位移历史
        v_history: (n_steps+1, n_dof) 速度历史
        a_history: (n_steps+1, n_dof) 加速度历史
    """
    t_start, t_end = t_span
    dt = (t_end - t_start) / n_steps
    n_dof = len(u0)

    t_array = np.linspace(t_start, t_end, n_steps + 1)
    u_hist = np.zeros((n_steps + 1, n_dof))
    v_hist = np.zeros((n_steps + 1, n_dof))
    a_hist = np.zeros((n_steps + 1, n_dof))

    u_hist[0] = u0
    v_hist[0] = v0
    a_hist[0] = a0

    u = u0.copy()
    v = v0.copy()
    a = a0.copy()

    for i in range(n_steps):
        t_new = t_array[i + 1]
        F_ext = F_ext_func(t_new)
        K_T = compute_tangent_stiffness(u)
        C = C_func(K_T)

        u, v, a, converged = trapezoidal_step(
            u, v, a, dt, M, C,
            compute_internal_force, compute_tangent_stiffness, F_ext,
            tol=tol, max_iter=max_iter
        )

        u_hist[i + 1] = u
        v_hist[i + 1] = v
        a_hist[i + 1] = a

    return t_array, u_hist, v_hist, a_hist
