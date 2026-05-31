"""
ode_integration.py
================================================================================
常微分方程时间积分与敏感性分析模块

本模块融合以下种子项目的核心算法：
  - 825_ode_euler     : 显式前向 Euler 方法
  - 1064_sensitive_ode: 敏感依赖 ODE 系统（y'' = y）的解析解与导数函数

科学背景
--------
在最优控制伴随方程方法中，状态方程是前向 ODE/PDE，伴随方程是后向 ODE/PDE。
时间积分器的稳定性、精度与敏感性分析能力至关重要。

对于半离散系统
    M dy/dt + A y + N(y) = F + B q
前向 Euler 给出：M (y^{n+1} − y^n)/Δt + A y^n + N(y^n) = F^n + B q^n
即 y^{n+1} = y^n + Δt M^{-1} [F^n + B q^n − A y^n − N(y^n)]

伴随方程（后向）的离散需要与状态方程使用相容的格式，以保证
离散梯度与连续梯度的一致性（optimize-then-discretize vs 
discretize-then-optimize 的兼容性问题）。

敏感依赖性验证
--------------
种子项目 1064_sensitive_ode 给出的系统 y₁' = y₂, y₂' = y₁（即 y'' = y）
是双曲型方程的特征方程，其解由 exp(t) 与 exp(−t) 线性组合。
该系统的 Lyapunov 指数 λ = 1，意味着微小扰动将以 e^{λt} 增长。
在最优控制中，这种敏感依赖要求伴随方程的后向积分必须足够稳定，
否则数值误差会在逆向传播中被放大。

关键公式
--------
1. 显式 Euler（前向）:
   y_{n+1} = y_n + h · f(t_n, y_n)

2. 隐式 Euler（后向）:
   y_{n+1} = y_n + h · f(t_{n+1}, y_{n+1})
   通常需要 Newton 迭代求解非线性方程。

3. 梯形法则（Crank-Nicolson）:
   y_{n+1} = y_n + h/2 · [f(t_n, y_n) + f(t_{n+1}, y_{n+1})]
   二阶精度，A-稳定。

4. 敏感 ODE 解析解:
   y(t) = C₁ e^t + C₂ e^{-t}
   其中 C₁, C₂ 由初始条件确定。
   对于 y(0) = 1+ε, y'(0) = −1，有
   C₁ = ε/2, C₂ = 1 + ε/2
   因此 y(t) = (ε/2) e^t + (1 + ε/2) e^{-t}
"""

import numpy as np


def explicit_euler(f, y0, t_span, n_steps):
    """
    显式前向 Euler 方法求解 ODE: y' = f(t, y)。

    参数
    ----
    f       : 右端函数，接受 (t, y) 返回导数数组
    y0      : 初始条件
    t_span  : (t0, t1)
    n_steps : 时间步数

    返回
    ----
    t_array : 时间数组 (n_steps+1,)
    y_array : 解数组 (n_steps+1, len(y0))
    """
    t0, t1 = t_span
    h = (t1 - t0) / n_steps
    dim = len(np.atleast_1d(y0))
    t_array = np.linspace(t0, t1, n_steps + 1)
    y_array = np.zeros((n_steps + 1, dim), dtype=float)
    y_array[0] = y0

    for n in range(n_steps):
        y_array[n + 1] = y_array[n] + h * f(t_array[n], y_array[n])

    return t_array, y_array


def implicit_euler_linear(M, A, b_fn, y0, t_span, n_steps):
    """
    隐式 Euler 求解线性 ODE 系统：
        M dy/dt + A y = b(t)
    离散格式：(M + h A) y^{n+1} = M y^n + h b^{n+1}

    参数
    ----
    M      : 质量矩阵（稠密 numpy 数组）
    A      : 刚度矩阵
    b_fn   : 右端项函数 b(t)
    y0     : 初始条件
    t_span : (t0, t1)
    n_steps: 时间步数

    返回
    ----
    t_array, y_array
    """
    t0, t1 = t_span
    h = (t1 - t0) / n_steps
    dim = len(np.atleast_1d(y0))
    t_array = np.linspace(t0, t1, n_steps + 1)
    y_array = np.zeros((n_steps + 1, dim), dtype=float)
    y_array[0] = y0

    LHS = M + h * A
    try:
        # 尝试直接求逆/求解
        import scipy.linalg as la
        for n in range(n_steps):
            rhs = M @ y_array[n] + h * b_fn(t_array[n + 1])
            y_array[n + 1] = la.solve(LHS, rhs, assume_a='pos')
    except Exception:
        # 回退到 numpy
        for n in range(n_steps):
            rhs = M @ y_array[n] + h * b_fn(t_array[n + 1])
            y_array[n + 1] = np.linalg.solve(LHS, rhs)

    return t_array, y_array


def crank_nicolson_linear(M, A, b_fn, y0, t_span, n_steps):
    """
    Crank-Nicolson（梯形法则）求解线性 ODE 系统：
        M dy/dt + A y = b(t)
    离散格式：(M + h/2 A) y^{n+1} = (M − h/2 A) y^n + h/2 (b^n + b^{n+1})

    稳定性：A-稳定，二阶精度。
    """
    t0, t1 = t_span
    h = (t1 - t0) / n_steps
    dim = len(np.atleast_1d(y0))
    t_array = np.linspace(t0, t1, n_steps + 1)
    y_array = np.zeros((n_steps + 1, dim), dtype=float)
    y_array[0] = y0

    LHS = M + 0.5 * h * A
    RHS_mat = M - 0.5 * h * A
    for n in range(n_steps):
        rhs = RHS_mat @ y_array[n] + 0.5 * h * (b_fn(t_array[n]) + b_fn(t_array[n + 1]))
        y_array[n + 1] = np.linalg.solve(LHS, rhs)

    return t_array, y_array


def trapezoid_integrate(f_vals, h):
    """
    复合梯形法则数值积分。
    ∫_a^b f(x) dx ≈ h/2 [f_0 + 2 Σ_{i=1}^{n-1} f_i + f_n]
    二阶精度，对光滑函数误差 O(h²)。

    参数
    ----
    f_vals : 等距节点上的函数值数组 (n+1,)
    h      : 步长
    """
    n = len(f_vals) - 1
    if n < 1:
        return 0.0
    val = 0.5 * f_vals[0] + 0.5 * f_vals[-1] + np.sum(f_vals[1:-1])
    return h * val


def sensitive_ode_rhs(t, y):
    """
    敏感依赖 ODE 的右端函数：
        y1' = y2
        y2' = y1
    即 y'' = y，具有 Lyapunov 指数 λ = ±1。
    """
    y = np.atleast_1d(y)
    return np.array([y[1], y[0]], dtype=float)


def sensitive_ode_exact(t, epsilon=0.0):
    """
    敏感依赖 ODE 的解析解。
    初始条件：y(0) = 1 + ε, y'(0) = −1
    解析解：y(t) = (ε/2) e^t + (1 + ε/2) e^{-t}
    """
    t = np.atleast_1d(t)
    c1 = 0.5 * epsilon
    c2 = 1.0 + 0.5 * epsilon
    y1 = c1 * np.exp(t) + c2 * np.exp(-t)
    y2 = c1 * np.exp(t) - c2 * np.exp(-t)
    return np.column_stack((y1, y2))


def verify_adjoint_consistency(y_state, p_adjoint, M, dt):
    """
    验证状态方程与伴随方程离散格式的相容性。
    对线性问题，离散伴随应与连续伴随相容。
    计算量：⟨M y^{n+1}, p^n⟩ − ⟨M y^n, p^{n+1}⟩ 应在理想情况下趋于零。

    这里我们计算一个简化的相容性指标：
    I = Σ_n [p^{n+1}^T M (y^{n+1} − y^n) + y^n^T M (p^n − p^{n+1})]
      = p^N^T M y^N − p^0^T M y^0   (理论上)
    """
    n_steps = len(y_state) - 1
    lhs = 0.0
    for n in range(n_steps):
        dy = y_state[n + 1] - y_state[n]
        dp = p_adjoint[n] - p_adjoint[n + 1]
        lhs += np.dot(p_adjoint[n + 1], M @ dy) + np.dot(y_state[n], M @ dp)

    rhs = np.dot(p_adjoint[-1], M @ y_state[-1]) - np.dot(p_adjoint[0], M @ y_state[0])
    return abs(lhs - rhs)
