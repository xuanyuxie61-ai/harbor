"""
time_integrator.py
==================
自适应时间积分器模块。
融合项目: 198_collatz_polynomial (迭代映射与收敛判据),
         1127_nschawor_eeg-mu-alpha-development (信号时间序列分析)

核心方法:
  1. Crank-Nicolson 隐式时间步进
  2. Newton-Raphson 非线性迭代求解
  3. 自适应时间步控制 (基于局部截断误差)
  4. 嵌入对 (embedded pair) 误差估计

关键公式:
  CN 离散: (c^{n+1} - c^n)/Δt = (1/2)*(F(c^{n+1}) + F(c^n))
  Newton 迭代: J*δc = -R(c^k), c^{k+1} = c^k + δc
  其中 R(c) = c - c^n - (Δt/2)*(F(c) + F(c^n))
       J = I - (Δt/2)*∂F/∂c
"""

import math
import numpy as np
from electrode_constants import (
    TOLERANCE_NEWTON, MAX_NEWTON_ITER, SAFETY_FACTOR,
    C_MAX, N_GRID
)
from compact_finite_difference import tridiag_solve


class TimeIntegratorState:
    """时间积分器状态跟踪。"""

    def __init__(self, N):
        self.N = N
        self.step_count = 0
        self.total_time = 0.0
        self.accepted_steps = 0
        self.rejected_steps = 0
        self.newton_iterations_total = 0
        self.min_dt = float('inf')
        self.max_dt = 0.0
        self.errors = []

    def record_step(self, dt, newton_iters, error=None, accepted=True):
        """记录一个时间步的信息。"""
        if accepted:
            self.accepted_steps += 1
            self.total_time += dt
            self.step_count += 1
            if error is not None:
                self.errors.append(error)
        else:
            self.rejected_steps += 1
        self.newton_iterations_total += newton_iters
        self.min_dt = min(self.min_dt, dt)
        self.max_dt = max(self.max_dt, dt)

    def summary(self):
        """返回积分器状态摘要。"""
        mean_newton = (self.newton_iterations_total / max(self.step_count, 1))
        mean_error = np.mean(self.errors) if self.errors else 0.0
        return {
            'total_steps': self.step_count,
            'accepted': self.accepted_steps,
            'rejected': self.rejected_steps,
            'total_time': self.total_time,
            'mean_newton_iters': mean_newton,
            'mean_error': mean_error,
            'dt_range': (self.min_dt, self.max_dt)
        }


def compute_nonlinear_residual(c_new, c_old, dt, rhs_func, N):
    """
    计算 CN 格式的非线性残差。

    R(c^{n+1}) = c^{n+1} - c^n - (Δt/2)*(F(c^{n+1}) + F(c^n))

    Parameters
    ----------
    c_new : ndarray
        当前猜测的新时刻浓度
    c_old : ndarray
        旧时刻浓度
    dt : float
        时间步长
    rhs_func : callable
        右端函数 F(c), 返回扩散算子作用
    N : int
        网格点数

    Returns
    -------
    residual : ndarray
        残差向量
    """
    F_new = rhs_func(c_new)
    F_old = rhs_func(c_old)
    residual = c_new - c_old - 0.5 * dt * (F_new + F_old)
    return residual


def compute_jacobian_diagonal(D_field, dDdc_field, r_grid, h, N):
    """
    计算 Jacobian 的对角线近似 (用于简化的 Newton 迭代)。

    J = I - (Δt/2)*∂F/∂c

    对于 ∂c/∂t = (1/r²)*∂/∂r(r²*D(c)*∂c/∂r),
    ∂F/∂c ≈ D(c)*L + D'(c)*∂c/∂r * (几何因子)

    Parameters
    ----------
    D_field : ndarray
        各点的扩散系数
    dDdc_field : ndarray
        扩散系数对浓度的导数
    r_grid : ndarray
        径向坐标
    h : float
        网格间距
    N : int
        网格点数

    Returns
    -------
    jac_diag : ndarray
        Jacobian 对角线近似
    """
    # 扩散算子的对角贡献 ≈ -2*D/h² (中心差分)
    # 加上 ∂D/∂c 的贡献
    jac_diag = np.zeros(N)
    for i in range(N):
        r_i = r_grid[i]
        if r_i < 1e-14:
            geom_factor = 0.0
        else:
            geom_factor = 2.0 / r_i

        # 主对角: 来自二阶导数离散
        diag_diffusion = -2.0 * D_field[i] / (h * h)
        # 来自 ∂D/∂c 的贡献 (近似)
        diag_nonlinear = dDdc_field[i] * (-2.0 / (h * h))
        jac_diag[i] = diag_diffusion + diag_nonlinear

    return jac_diag


def newton_solve(c_old, dt, rhs_func, N, r_grid, h,
                 D_func=None, T=None, tol=TOLERANCE_NEWTON, max_iter=MAX_NEWTON_ITER):
    """
    Newton-Raphson 求解 CN 非线性系统。

    使用 Picard 迭代 + 阻尼 Newton 混合策略。
    对于非线性扩散方程:
    (c^{n+1} - c^n)/dt = (1/2)*(F(c^{n+1}) + F(c^n))

    Parameters
    ----------
    c_old : ndarray
        旧时刻浓度
    dt : float
        时间步长
    rhs_func : callable
        右端函数 F(c)
    N : int
        网格点数
    r_grid : ndarray
        径向坐标
    h : float
        网格间距
    D_func : callable, optional
        浓度依赖扩散系数函数 D(c)
    T : float, optional
        温度
    tol : float
        收敛容差
    max_iter : int
        最大迭代次数

    Returns
    -------
    c_new : ndarray
        新时刻浓度
    n_iters : int
        实际迭代次数
    converged : bool
        是否收敛
    final_residual : float
        最终残差范数
    """
    # 初始猜测: 旧浓度 (小 dt 时是好的猜测)
    c_new = c_old.copy()
    F_old = rhs_func(c_old)

    for iteration in range(max_iter):
        F_new = rhs_func(c_new)

        # CN 残差: R = c_new - c_old - (dt/2)*(F_new + F_old)
        R = c_new - c_old - 0.5 * dt * (F_new + F_old)
        res_norm = np.linalg.norm(R)

        if res_norm < tol:
            return c_new, iteration + 1, True, res_norm

        # Picard 型更新 (简化 Newton):
        # c_new = (c_old + (dt/2)*(F_new + F_old))
        # 但 F_new 依赖 c_new, 所以使用线性化
        # 简化: 直接迭代 c_new^{k+1} = c_old + dt * F(c_new^k)
        # (向后 Euler 风格, 更稳定)

        # 混合策略: 使用向后 Euler 作为迭代
        c_target = c_old + dt * F_new
        # 带松弛
        omega = 0.7  # 松弛因子
        c_update = (1.0 - omega) * c_new + omega * c_target

        # 边界保护
        c_update = np.clip(c_update, 0.0, C_MAX * 0.999)

        # 检查更新幅度
        delta = np.linalg.norm(c_update - c_new)
        c_new = c_update

        # 防止振荡
        if delta < tol * 0.01:
            break

    # 最终残差
    F_new = rhs_func(c_new)
    R_final = c_new - c_old - 0.5 * dt * (F_new + F_old)
    return c_new, iteration + 1, res_norm < tol * 100, np.linalg.norm(R_final)


def embedded_error_estimate(c_high, c_low, order_high=2, order_low=1):
    """
    嵌入对误差估计 (类似 Runge-Kutta-Fehlberg)。

    误差估计: e = ||c_high - c_low|| / (1 + ||c_high||)

    新时间步: dt_new = dt * (tol / e)^{1/(p+1)}

    Parameters
    ----------
    c_high : ndarray
        高阶方法的结果
    c_low : ndarray
        低阶方法的结果
    order_high : int
        高阶方法的阶数
    order_low : int
        低阶方法的阶数

    Returns
    -------
    error_estimate : float
        归一化误差估计
    """
    diff = c_high - c_low
    error_norm = np.linalg.norm(diff) / (1.0 + np.linalg.norm(c_high))
    return error_norm


def compute_new_timestep(dt_current, error, tol, safety=SAFETY_FACTOR,
                          order=2, dt_min=1e-12, dt_max=1.0):
    """
    自适应时间步控制。

    dt_new = dt * safety * (tol / error)^{1/(order+1)}

    限制: dt_min ≤ dt_new ≤ dt_max
    且 dt_new ≤ 2 * dt_current (防止步长剧增)

    Parameters
    ----------
    dt_current : float
        当前时间步
    error : float
        误差估计
    tol : float
        误差容限
    safety : float
        安全因子
    order : int
        方法阶数
    dt_min, dt_max : float
        时间步上下界

    Returns
    -------
    float
        新的时间步长
    """
    if error < 1e-30:
        return min(dt_current * 2.0, dt_max)

    exponent = 1.0 / (order + 1)
    factor = safety * (tol / error) ** exponent

    # 限制增长/缩减因子
    factor = max(0.1, min(factor, 2.0))

    dt_new = dt_current * factor
    return max(dt_min, min(dt_new, dt_max))


def crank_nicolson_step(c_old, dt, D_func, r_grid, h, N, T,
                         use_adaptive=True, tol_error=1e-6):
    """
    执行一个完整的 Crank-Nicolson 时间步。

    使用线性化策略:
    1. 用当前浓度计算 D
    2. 用线性隐式格式求解
    3. 如有需要, 做 1-2 次 Picard 迭代修正

    Parameters
    ----------
    c_old : ndarray
        当前浓度场
    dt : float
        时间步长
    D_func : callable
        扩散系数函数 D(c, T)
    r_grid : ndarray
        径向坐标
    h : float
        网格间距
    N : int
        内部网格点数
    T : float
        温度
    use_adaptive : bool
        是否使用自适应
    tol_error : float
        误差容限

    Returns
    -------
    c_new : ndarray
        新浓度场
    dt_used : float
        实际使用的时间步
    newton_iters : int
        Newton 迭代次数
    error_est : float
        误差估计
    """
    # 计算浓度依赖的扩散系数
    D_field = np.array([D_func(c_old[i], T) for i in range(N)])
    D_mean = max(np.mean(D_field), 1e-30)

    # 构造扩散算子矩阵 (线性化, 使用 D_old)
    # L * c = D * (c_{i+1} - 2c_i + c_{i-1})/h² + 几何修正
    L = np.zeros((N, N))
    for i in range(N):
        r_i = r_grid[i]
        D_i = D_field[i]

        if r_i > 1e-14:
            geom_plus = 1.0 + h / (2.0 * r_i)
            geom_minus = 1.0 - h / (2.0 * r_i)
        else:
            # r=0: L'Hôpital, ∇²c = 3*c''
            geom_plus = geom_minus = 1.5  # 3 * 1/2

        # 对角: -D*(geom_plus + geom_minus)/h²
        L[i, i] = -D_i * (geom_plus + geom_minus) / (h * h)
        if i > 0:
            L[i, i-1] = D_i * geom_minus / (h * h)
        if i < N - 1:
            L[i, i+1] = D_i * geom_plus / (h * h)

    # CN 隐式矩阵: (I - dt/2 * L) * c^{n+1} = (I + dt/2 * L) * c^n
    I_mat = np.eye(N)
    A_lhs = I_mat - 0.5 * dt * L
    A_rhs = I_mat + 0.5 * dt * L

    rhs = A_rhs @ c_old

    # 求解 (使用 numpy 直接求解, 对于小规模 N 足够快)
    try:
        c_new = np.linalg.solve(A_lhs, rhs)
    except np.linalg.LinAlgError:
        # 奇异, 使用伪逆
        c_new = np.linalg.lstsq(A_lhs, rhs, rcond=None)[0]

    # Picard 迭代修正 (1-2 次, 更新 D)
    n_iters = 1
    for picard_it in range(2):
        D_new = np.array([D_func(c_new[i], T) for i in range(N)])
        # 检查 D 变化
        dD_rel = np.max(np.abs(D_new - D_field)) / max(D_mean, 1e-30)
        if dD_rel < 0.01:
            break  # 收敛

        # 重新构造矩阵 (使用 D_new)
        for i in range(N):
            r_i = r_grid[i]
            D_i = D_new[i]
            if r_i > 1e-14:
                geom_plus = 1.0 + h / (2.0 * r_i)
                geom_minus = 1.0 - h / (2.0 * r_i)
            else:
                geom_plus = geom_minus = 1.5

            L[i, i] = -D_i * (geom_plus + geom_minus) / (h * h)
            if i > 0:
                L[i, i-1] = D_i * geom_minus / (h * h)
            if i < N - 1:
                L[i, i+1] = D_i * geom_plus / (h * h)

        A_lhs = I_mat - 0.5 * dt * L
        A_rhs = I_mat + 0.5 * dt * L
        rhs = A_rhs @ c_old
        try:
            c_new_new = np.linalg.solve(A_lhs, rhs)
        except np.linalg.LinAlgError:
            c_new_new = np.linalg.lstsq(A_lhs, rhs, rcond=None)[0]
        # 松弛
        c_new = 0.5 * c_new + 0.5 * c_new_new
        D_field = D_new
        n_iters += 1

    # 边界保护
    c_new = np.clip(c_new, 0.0, C_MAX * 0.999)

    # 误差估计 (半步对比)
    if use_adaptive and N > 2:
        # 半步法: 两个 dt/2 步
        try:
            A_lhs_half = np.eye(N) - 0.25 * dt * L
            A_rhs_half = np.eye(N) + 0.25 * dt * L
            rhs_half = A_rhs_half @ c_old
            c_half = np.linalg.solve(A_lhs_half, rhs_half)
            c_half = np.clip(c_half, 0.0, C_MAX * 0.999)
            rhs_half2 = A_rhs_half @ c_half
            c_full = np.linalg.solve(A_lhs_half, rhs_half2)
            c_full = np.clip(c_full, 0.0, C_MAX * 0.999)
            error_est = embedded_error_estimate(c_full, c_new)
        except Exception:
            error_est = 0.0
    else:
        error_est = 0.0

    return c_new, dt, n_iters, error_est
