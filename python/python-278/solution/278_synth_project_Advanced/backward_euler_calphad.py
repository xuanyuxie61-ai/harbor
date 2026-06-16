"""
backward_euler_calphad.py
=========================
适用于 CALPHAD 相场演化的后向 Euler (隐式) 时间积分器。

基于 Picard 迭代的隐式后向 Euler 方法:
  c^{n+1} = c^n + dt * F(c^{n+1})

  其中 F(c) = M * ∇² [d²G/dc² * (c - c0) - κ * ∇²c]
             (Cahn-Hilliard 方程的半隐式线性化)

求解策略:
  1. 在每一时间步进行 Picard 迭代:
     c^{(k+1)} = c^n + dt * F(c^{(k)})
  2. 或者使用 Newton 迭代加速收敛
  3. 自适应时间步长控制

该方法源自 seed project 064_backward_euler_fixed,
其中使用固定点迭代而非直接非线性求解器来处理隐式步。
"""

import numpy as np
from gibbs_energy_calphad import second_derivative_G
from high_order_fd import (
    compact_second_derivative,
    compact_fourth_derivative,
)
from calphad_fec_constants import (
    BE_MAX_PICARD,
    BE_TOL,
    CH_MOBILITY_0,
    CH_KAPPA,
    CH_LX,
    CH_NX,
    CH_DT,
)


def mobility_function(c, T, M0):
    """
    浓度依赖的迁移率函数 (Cahn-Hilliard 方程):

    M(c) = M_0 * c * (1 - c) * exp(-Q/(R*T))

    其中 Q 为扩散激活能, c(1-c) 保证在无溶质/纯溶质时
    迁移率为零 (物理约束)。

    Parameters
    ----------
    c : np.ndarray
        浓度场
    T : float
        温度 (K)
    M0 : float
        基准迁移率 (m²/(J·s))

    Returns
    -------
    np.ndarray
        M(c) 浓度依赖迁移率
    """
    c = np.clip(c, 1.0e-15, 1.0 - 1.0e-15)
    Q_act = 140000.0  # Fe-C 扩散激活能 J/mol
    R = 8.3145
    # 迁移率: M(c) = M0 * c*(1-c) * exp(-Q/RT)
    M = M0 * c * (1.0 - c) * np.exp(-Q_act / (R * T))
    return np.maximum(M, 1.0e-30)  # 保证正定性


def cahn_hilliard_rhs(c, T, h, M0, kappa, d2G_dc2_array):
    """
    计算 Cahn-Hilliard 方程的右端项:

    dc/dt = M * ∇² [d²G/dc² * (c - c_ref) - κ * ∇²c]
          = M * ∇²μ

    其中化学势 μ = dG/dc - κ*∇²c

    Parameters
    ----------
    c : np.ndarray
        浓度场 c(x)
    T : float
        温度 (K)
    h : float
        空间步长
    M0 : float
        基准迁移率
    kappa : float
        梯度能系数
    d2G_dc2_array : np.ndarray
        各网格点处的 d²G/dc² 值

    Returns
    -------
    np.ndarray
        dc/dt (右端项)
    """
    N = len(c)
    # 化学势: mu = d2G * delta_c - kappa * nabla^2 c
    # 其中 delta_c = c - c_mean (相对参考浓度的偏差)
    c_mean = np.mean(c)
    delta_c = c - c_mean

    # 使用紧致差分计算 ∇²c
    d2c_dx2 = compact_second_derivative(c, h)

    # 化学势 (逐点计算)
    mu = d2G_dc2_array * delta_c - kappa * d2c_dx2

    # 迁移率
    M = mobility_function(c, T, M0)

    # ∇²(M * mu): 先计算 M*mu, 再取 ∇²
    M_mu = M * mu
    laplacian_M_mu = compact_second_derivative(M_mu, h)

    return laplacian_M_mu


def backward_euler_picard(c0, T, dt, n_steps, h, M0, kappa,
                          phase, c_ref=None):
    """
    后向 Euler 方法配合 Picard 迭代求解 Cahn-Hilliard 方程:

    c^{n+1} = c^n + dt * F(c^{n+1})

    使用 Picard 迭代:
    For k = 0, 1, 2, ...
        c^{(k+1)} = c^n + dt * F(c^{(k)})
    Until ||c^{(k+1)} - c^{(k)}|| < tol

    Parameters
    ----------
    c0 : np.ndarray
        初始浓度场
    T : float
        温度 (K)
    dt : float
        时间步长 (s)
    n_steps : int
        时间步数
    h : float
        空间步长
    M0 : float
        基准迁移率
    kappa : float
        梯度能系数
    phase : str
        相名称 (用于计算 d²G/dc²)
    c_ref : float, optional
        参考浓度 (默认取平均值)

    Returns
    -------
    dict
        {
            'c_final': np.ndarray,     # 最终浓度场
            'c_history': list,         # 每 50 步的浓度快照
            'energy_history': list,    # 自由能演化
            'converged': bool,         # 是否收敛
            'n_iter_total': int        # 总 Picard 迭代次数
        }
    """
    N = len(c0)
    c = c0.copy()
    if c_ref is None:
        c_ref = np.mean(c)

    c_history = [c.copy()]
    energy_history = []
    n_iter_total = 0
    converged = True

    for step in range(n_steps):
        # 计算当前浓度处的 d²G/dc²
        d2G_dc2 = second_derivative_G(c, T, phase)

        # Picard 迭代
        c_old = c.copy()
        picard_converged = False

        for picard_iter in range(BE_MAX_PICARD):
            # 显式估计右端项
            rhs = cahn_hilliard_rhs(c, T, h, M0, kappa, d2G_dc2)

            # 后向 Euler 更新
            c_new = c_old + dt * rhs

            # 物理约束: 浓度在 [0, 1] 范围内
            c_new = np.clip(c_new, 1.0e-12, 1.0 - 1.0e-12)

            # 守恒约束: 保持总质量不变
            c_new = c_new - (np.mean(c_new) - np.mean(c0))
            c_new = np.clip(c_new, 1.0e-12, 1.0 - 1.0e-12)

            # 收敛检查
            diff_norm = np.max(np.abs(c_new - c))
            if diff_norm < BE_TOL:
                picard_converged = True
                break

            c = c_new
            n_iter_total += 1

        if not picard_converged:
            converged = False

        c = c_new if picard_converged else c

        # 计算系统总自由能
        from gibbs_energy_calphad import gibbs_substitutional
        G_local = gibbs_substitutional(c, T, phase)
        # 加上梯度能项
        grad_c = compact_second_derivative(c, h)  # 近似 ∇²c
        E_grad = 0.5 * kappa * np.sum(grad_c * c) * h
        E_total = np.sum(G_local) * h + E_grad
        energy_history.append(E_total)

        # 记录快照
        if (step + 1) % max(1, n_steps // 10) == 0 or step == 0:
            c_history.append(c.copy())

    return {
        'c_final': c,
        'c_history': c_history,
        'energy_history': energy_history,
        'converged': converged,
        'n_iter_total': n_iter_total,
    }


def backward_euler_adaptive(c0, T, dt_init, t_total, h, M0, kappa,
                            phase, safety=0.9, dt_min=1e-12, dt_max=1e-2):
    """
    自适应时间步长的后向 Euler 方法。

    时间步长控制策略:
    1. 估计局部截断误差: err = ||c_BE(dt) - c_BE(dt/2)||
    2. 如果 err < tol: 接受步, dt_new = dt * (tol/err)^(1/2)
    3. 如果 err >= tol: 拒绝步, dt_new = dt * (tol/err)^(1/2)

    Parameters
    ----------
    c0 : np.ndarray
        初始浓度场
    T : float
        温度 (K)
    dt_init : float
        初始时间步长
    t_total : float
        总模拟时间
    h : float
        空间步长
    M0 : float
        基准迁移率
    kappa : float
        梯度能系数
    phase : str
        相名称
    safety : float
        安全因子 (0 < safety < 1)
    dt_min : float
        最小允许时间步长
    dt_max : float
        最大允许时间步长

    Returns
    -------
    dict
        同 backward_euler_picard, 外加 dt_history
    """
    c = c0.copy()
    t = 0.0
    dt = dt_init
    tol = 1.0e-6

    c_history = [c.copy()]
    energy_history = []
    dt_history = [dt]
    n_steps = 0
    n_iter_total = 0
    converged = True

    while t < t_total and n_steps < 5000:
        # 确保最后一步不超调
        if t + dt > t_total:
            dt = t_total - t

        # 半步结果
        result_half = backward_euler_picard(
            c, T, dt / 2, 1, h, M0, kappa, phase
        )
        c_half = result_half['c_final']

        # 全步结果
        result_full = backward_euler_picard(
            c, T, dt, 1, h, M0, kappa, phase
        )
        c_full = result_full['c_final']
        n_iter_total += result_full['n_iter_total']

        # 误差估计
        err = np.max(np.abs(c_full - c_half))
        err = max(err, 1.0e-30)

        if err <= tol or dt <= dt_min * 1.01:
            # 接受步
            c = c_full
            t += dt
            n_steps += 1

            # 计算自由能
            from gibbs_energy_calphad import gibbs_substitutional
            G_local = gibbs_substitutional(c, T, phase)
            energy_history.append(float(np.sum(G_local) * h))

            if n_steps % max(1, int(t_total / dt_init / 20)) == 0:
                c_history.append(c.copy())

        # 自适应步长更新
        dt_new = safety * dt * (tol / err) ** 0.5
        dt = np.clip(dt_new, dt_min, dt_max)
        dt_history.append(dt)

    return {
        'c_final': c,
        'c_history': c_history,
        'energy_history': energy_history,
        'converged': converged,
        'n_iter_total': n_iter_total,
        'dt_history': dt_history,
        'n_steps': n_steps,
    }
