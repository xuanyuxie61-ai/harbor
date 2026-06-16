"""
time_delay_stability.py — 时滞反馈系统的稳定性与 Floquet 分析
============================================================
本模块分析顶夸克质量拟合迭代算法的稳定性,
借鉴时滞动力系统的边界碰撞分岔和 Floquet 理论。

数学基础:
  质量拟合可建模为离散迭代系统:
    m_{n+1} = F(m_n, m_{n-1}, ..., α)
  其中 α 是系统参数。

  线性化稳定性:
    δm_{n+1} = J δm_n
  其中 J = ∂F/∂m 是 Jacobian 矩阵。
  稳定性条件: |λ_max(J)| < 1 (谱半径)。

  对于含时滞的迭代:
    m_{n+1} = F(m_n, m_{n-d})
  状态扩展为 (m_n, m_{n-1}, ..., m_{n-d+1}),
  Jacobian 为 (d+1)×(d+1) 矩阵。

  Floquet 乘子:
    对于周期-1 轨道 m*, 乘子 μ 满足:
    det(J - μI) = 0
  |μ| < 1: 稳定; |μ| = 1: 分岔点; |μ| > 1: 不稳定

  边界碰撞分岔:
    当迭代函数 F 有分段线性不连续性时,
    边界碰撞分岔发生在轨道碰到不连续边界时。
    条件: m* = m_boundary, 且 |λ_left × λ_right| 跨越 1

  映射种子项目:
    - 1085_PierceRyan: 边界碰撞分岔与 Floquet 分析
    - 006_anishchenko_ode: 非线性 ODE 与 Heaviside 开关
"""

import numpy as np


def floquet_multipliers_1d(iteration_func, m_fixed, h=1e-6):
    """
    计算一维离散迭代的 Floquet 乘子。

    对于 m_{n+1} = F(m_n), 在不动点 m*:
      μ = F'(m*)

    使用中心差分:
      F'(m*) ≈ (F(m*+h) - F(m*-h)) / (2h)

    稳定性: |μ| < 1 稳定, |μ| > 1 不稳定

    参数:
        iteration_func: 迭代函数 F(m)
        m_fixed: 不动点 m*
        h: 差分步长

    返回:
        (mu, is_stable)
    """
    fp = iteration_func(m_fixed + h)
    fm = iteration_func(m_fixed - h)
    mu = (fp - fm) / (2.0 * h)

    is_stable = abs(mu) < 1.0

    return mu, is_stable


def floquet_multipliers_nd(iteration_func, x_fixed, h=1e-6):
    """
    计算 N 维离散迭代的 Floquet 乘子。

    对于 x_{n+1} = F(x_n), 在不动点 x*:
      J_{ij} = ∂F_i/∂x_j
      μ_k = eigenvalues(J)

    参数:
        iteration_func: 向量迭代函数 F: R^n → R^n
        x_fixed: 不动点
        h: 差分步长

    返回:
        (multipliers, is_stable, J)
    """
    x_fixed = np.asarray(x_fixed, dtype=float)
    n = len(x_fixed)

    J = np.zeros((n, n))
    F0 = np.asarray(iteration_func(x_fixed))

    for j in range(n):
        x_p = x_fixed.copy()
        x_m = x_fixed.copy()
        x_p[j] += h
        x_m[j] -= h
        J[:, j] = (np.asarray(iteration_func(x_p)) -
                   np.asarray(iteration_func(x_m))) / (2.0 * h)

    eigenvalues = np.linalg.eigvals(J)
    multipliers = eigenvalues
    is_stable = np.all(np.abs(multipliers) < 1.0)

    return multipliers, is_stable, J


def border_collision_detection(iteration_func, boundary_func, m_range,
                               n_points=1000):
    """
    检测边界碰撞分岔点。

    边界碰撞发生在:
      m* = F(m*) 且 m* = boundary

    算法:
    1. 扫描参数空间, 找到不动点 m*(α)
    2. 检测 m*(α) 何时穿过边界
    3. 在穿越点计算左右 Jacobian

    映射种子项目:
      - 1085_PierceRyan: 边界碰撞检测

    参数:
        iteration_func: F(m, alpha) 对固定 alpha
        boundary_func: boundary(alpha) 边界函数
        m_range: (m_min, m_max)
        n_points: 扫描点数

    返回:
        collision_points: 碰撞参数值列表
        stability_info: 稳定性信息
    """
    m_values = np.linspace(m_range[0], m_range[1], n_points)

    # 计算不动点残差
    residuals = np.zeros(n_points)
    for i, m in enumerate(m_values):
        Fm = iteration_func(m)
        residuals[i] = Fm - m

    # 检测符号变化 (不动点)
    crossings = []
    for i in range(n_points - 1):
        if residuals[i] * residuals[i + 1] < 0:
            # 二分法精化
            a, b = m_values[i], m_values[i + 1]
            for _ in range(50):
                mid = (a + b) / 2.0
                if (iteration_func(a) - a) * (iteration_func(mid) - mid) < 0:
                    b = mid
                else:
                    a = mid
            m_star = (a + b) / 2.0
            crossings.append(m_star)

    # 对每个不动点检查边界碰撞
    collision_points = []
    stability_info = []

    for m_star in crossings:
        boundary_val = boundary_func(m_star)
        if abs(m_star - boundary_val) < 0.01 * abs(m_range[1] - m_range[0]):
            collision_points.append(m_star)

            # 计算 Floquet 乘子
            mu, stable = floquet_multipliers_1d(iteration_func, m_star)
            stability_info.append({
                'm_star': m_star,
                'multiplier': mu,
                'is_stable': stable
            })

    return collision_points, stability_info


def convergence_monitor(iterates, tol=1e-8, max_iter=1000):
    """
    监测迭代序列的收敛性 (基于 Anishchenko 型开关动力学)。

    检测收敛模式:
    1. 单调收敛: |x_{n+1} - x_n| 单调递减
    2. 振荡收敛: 符号交替但幅度递减
    3. 周期振荡: 进入极限环
    4. 混沌: 对初始条件敏感

    映射种子项目:
      - 006_anishchenko_ode: Heaviside 开关 + 非线性动力学

    参数:
        iterates: 迭代值序列
        tol: 收敛阈值
        max_iter: 最大迭代数

    返回:
        convergence_info: 收敛状态字典
    """
    n = len(iterates)
    if n < 3:
        return {'status': 'insufficient_data', 'converged': False}

    # 连续差
    diffs = np.diff(iterates)
    abs_diffs = np.abs(diffs)

    # 收敛检查
    final_diff = abs_diffs[-1] if len(abs_diffs) > 0 else float('inf')
    converged = final_diff < tol

    # 收敛模式检测
    if n >= 10:
        recent_diffs = abs_diffs[-10:]

        # 单调递减?
        is_monotone = np.all(np.diff(recent_diffs) < 0)

        # 振荡? (符号交替)
        sign_changes = np.sum(np.diff(np.sign(diffs[-10:])) != 0)
        is_oscillating = sign_changes >= 7

        # 周期性检测 (自相关)
        if n >= 20:
            last_20 = iterates[-20:]
            mean_val = np.mean(last_20)
            centered = last_20 - mean_val
            var = np.sum(centered**2)
            if var > 0:
                autocorr = np.correlate(centered, centered, mode='full')
                autocorr = autocorr[len(autocorr) // 2:] / var
                # 寻找第一个非零峰
                period = None
                for lag in range(2, 10):
                    if lag < len(autocorr) and autocorr[lag] > 0.5:
                        period = lag
                        break
            else:
                period = None
        else:
            period = None

        if is_monotone:
            mode = 'monotone'
        elif is_oscillating:
            mode = 'oscillating'
        elif period is not None:
            mode = f'periodic_{period}'
        else:
            mode = 'irregular'
    else:
        mode = 'unknown'

    return {
        'status': mode,
        'converged': converged,
        'final_residual': final_diff,
        'n_iterations': n,
    }


def iterative_mass_solver(likelihood_func, m_init, m_range,
                          learning_rate=0.1, max_iter=200, tol=1e-6):
    """
    基于迭代的顶夸克质量求解器 (含稳定性监测)。

    迭代方案:
      m_{n+1} = m_n + η × ∂lnL/∂m

    等价于梯度上升求最大似然:
      m* = argmax_m ln L(m)

    含自适应学习率和时滞反馈:
      m_{n+1} = m_n + η_n × (∂lnL/∂m|_n + α × (m_n - m_{n-1}))

    映射种子项目:
      - 006_anishchenko_ode: 非线性迭代 + 开关
      - 322_duffing_ode: 非线性振荡器收敛

    参数:
        likelihood_func: ln L(m) 函数
        m_init: 初始猜测
        m_range: (m_min, m_max) 物理范围
        learning_rate: 学习率 η
        max_iter: 最大迭代数
        tol: 收敛阈值

    返回:
        result: 求解结果字典
    """
    m = float(m_init)
    m_prev = m
    h = 1e-4  # 数值导数步长

    iterates = [m]
    converged = False

    for step in range(max_iter):
        # 数值导数 ∂lnL/∂m
        lp = likelihood_func(m + h)
        lm = likelihood_func(m - h)
        grad = (lp - lm) / (2.0 * h)

        # 时滞反馈项
        delay_term = 0.1 * (m - m_prev)

        # 更新
        m_new = m + learning_rate * (grad + delay_term)

        # 边界约束
        m_new = np.clip(m_new, m_range[0], m_range[1])

        # 监测收敛
        if abs(m_new - m) < tol:
            converged = True
            iterates.append(m_new)
            break

        m_prev = m
        m = m_new
        iterates.append(m)

        # 自适应学习率 (基于梯度变化)
        if step > 5:
            recent_grads = []
            for k in range(max(0, step - 3), step):
                mk = iterates[k]
                gp = likelihood_func(mk + h)
                gm = likelihood_func(mk - h)
                recent_grads.append((gp - gm) / (2.0 * h))
            grad_var = np.var(recent_grads) if len(recent_grads) > 1 else 1.0
            learning_rate = min(0.1, 0.01 / max(grad_var, 1e-10))

    # 收敛分析
    conv_info = convergence_monitor(iterates, tol=tol)

    # Floquet 稳定性
    def iter_map(m_val):
        lp = likelihood_func(m_val + h)
        lm = likelihood_func(m_val - h)
        grad = (lp - lm) / (2.0 * h)
        return m_val + learning_rate * grad

    mu, stable = floquet_multipliers_1d(iter_map, m)

    return {
        'm_opt': m,
        'converged': converged,
        'n_iterations': len(iterates) - 1,
        'iterates': iterates,
        'convergence_info': conv_info,
        'floquet_multiplier': mu,
        'is_stable': stable,
        'final_likelihood': likelihood_func(m),
    }
