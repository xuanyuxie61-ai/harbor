"""
 inverse_problem.py
 
 融合种子项目:
   - 216_control_bio_homework: 最优控制、RK4 时间积分、伴随状态法思想
 
 科学应用:
   全波形反演（Full Waveform Inversion, FWI）是一个 PDE 约束优化问题:
     min_m  J(m) = 0.5 * ||d_obs - F(m)||^2_2
     s.t.   A(m) u = f
   
   其中 m 为模型参数（如速度），u 为波场，F(m) 为正演算子，d_obs 为观测数据。
   
   使用伴随状态法（Adjoint State Method）计算梯度:
     nabla_m J = - (partial A / partial m)^* [lambda]
   其中伴随场 lambda 满足:
     A^*(m) lambda = (F(m) - d_obs)
   
   这是最优控制理论在地球物理中的直接应用，将模型参数视为"控制变量"，
   波动方程视为"状态方程"，数据残差视为"代价函数"。
"""

import numpy as np
from wave_propagation import rk4_integrate
from helmholtz_solver import gmres_solve


def compute_misfit(d_obs, d_calc):
    """
    计算数据残差（L2 范数 misfit）。
    
    目标泛函:
      J = 0.5 * sum_i ||d_obs_i - d_calc_i||^2
    
    Parameters
    ----------
    d_obs, d_calc : ndarray
        观测数据和计算数据。
    
    Returns
    -------
    misfit : float
        残差值。
    residual : ndarray
        残差向量 d_calc - d_obs。
    """
    residual = d_calc - d_obs
    misfit = 0.5 * np.sum(residual ** 2)
    return misfit, residual


def compute_gradient_fd(objective_func, m, h=1e-4):
    """
    使用有限差分法计算目标函数梯度（仅用于验证）。
    
    公式:
      grad_i = (J(m + h*e_i) - J(m - h*e_i)) / (2h)
    
    Parameters
    ----------
    objective_func : callable
        目标函数 J(m)。
    m : ndarray
        模型参数。
    h : float
        差分步长。
    
    Returns
    -------
    grad : ndarray
        梯度向量。
    """
    n = len(m)
    grad = np.zeros(n)
    j0 = objective_func(m)
    for i in range(n):
        m_plus = m.copy()
        m_minus = m.copy()
        m_plus[i] += h
        m_minus[i] -= h
        grad[i] = (objective_func(m_plus) - objective_func(m_minus)) / (2.0 * h)
    return grad


def adjoint_state_gradient_1d(m, d_obs, dx, dt, nt, source_pos, source_fn,
                               boundary='absorbing'):
    """
    使用伴随状态法计算一维 FWI 的梯度。
    
    正演方程（状态方程）:
      d^2u/dt^2 = c^2(x) * d^2u/dx^2 + f(t)*delta(x - x_s)
    
    伴随方程:
      d^2lambda/dt^2 = c^2(x) * d^2lambda/dx^2 + (u - d_obs)*delta(x - x_r)
      终端条件: lambda(T) = 0, dlambda/dt(T) = 0
    
    梯度公式:
      grad(x) = -2/c^3(x) * integral_0^T (d^2u/dx^2) * lambda dt
    
    注意：伴随方程需要反向时间积分。
    
    Parameters
    ----------
    m : ndarray, shape (nx,)
        当前速度模型 c(x)。
    d_obs : ndarray, shape (nt+1, nx)
        观测波场（或仅在接收器位置的记录）。
    dx, dt : float
        空间和时间步长。
    nt : int
        时间步数。
    source_pos : int
        震源位置。
    source_fn : callable
        震源时间函数。
    boundary : str
        边界条件。
    
    Returns
    -------
    grad : ndarray, shape (nx,)
        速度模型梯度。
    misfit : float
        当前 misfit。
    """
    nx = len(m)
    c = np.asarray(m, dtype=float)
    
    # 正演模拟
    from wave_propagation import seismic_wave_rk4_1d
    u_hist, t = seismic_wave_rk4_1d(nx, dx, nt, dt, c, source_fn, source_pos,
                                     boundary=boundary)
    
    # 计算 misfit（使用全波场作为观测，简化处理）
    misfit, residual = compute_misfit(d_obs, u_hist)
    
    # 伴随方程反向积分（简化为使用相同的 RK4 但时间反向）
    # 将二阶系统转化为一阶并反向积分
    def adjoint_deriv(tau, y):
        # tau 为反向时间: tau = T - t
        # 状态: y = [lambda, dlambda/dtau]
        lam = y[:nx]
        vlam = y[nx:]
        dlam_dtau = vlam.copy()
        dvlam_dtau = np.zeros(nx)
        t_forward = nt * dt - tau
        t_idx = min(int(t_forward / dt), nt)
        # TODO: 实现伴随方程的空间二阶导数离散和边界条件
        # 科学知识：伴随方程 d²λ/dt² = c²(x)·d²λ/dx² 的空间离散
        # 关键约束：此处的离散格式必须与 wave_propagation.py 中正演方程的
        # 空间离散格式保持一致（算子自伴性），否则 FWI 梯度计算将产生错误。
        # 要求：
        #   1. 内部点使用与正演一致的中心差分
        #   2. 边界条件类型与正演一致（absorbing / reflecting）
        pass
        # 残差作为伴随源
        if 0 <= t_idx <= nt:
            dvlam_dtau += residual[t_idx, :]
        return np.concatenate([dlam_dtau, dvlam_dtau])
    
    y0 = np.zeros(2 * nx)
    tau, y_adj = rk4_integrate(adjoint_deriv, (0.0, nt * dt), y0, nt)
    # 反转时间顺序
    lambda_field = y_adj[::-1, :nx]
    
    # 计算梯度: grad = -2/c^3 * integral(uxx * lambda) dt
    grad = np.zeros(nx)
    for ti in range(nt + 1):
        uxx = np.zeros(nx)
        u_t = u_hist[ti, :]
        for i in range(1, nx - 1):
            uxx[i] = (u_t[i - 1] - 2 * u_t[i] + u_t[i + 1]) / dx ** 2
        grad += uxx * lambda_field[ti, :]
    grad = -2.0 * grad / (c ** 3) * dt
    
    # 平滑梯度以提高数值稳定性
    grad_smooth = grad.copy()
    for i in range(1, nx - 1):
        grad_smooth[i] = 0.25 * grad[i - 1] + 0.5 * grad[i] + 0.25 * grad[i + 1]
    grad_smooth[0] = grad[0]
    grad_smooth[-1] = grad[-1]
    
    return grad_smooth, misfit


def fwi_gradient_descent_1d(m_init, d_obs, dx, dt, nt, source_pos, source_fn,
                             n_iter=20, step_length=1e6, boundary='absorbing',
                             verbose=True):
    """
    一维全波形反演的梯度下降法。
    
    优化算法:
      m^{k+1} = m^k - alpha * grad J(m^k)
    
    其中步长 alpha 采用简单的线性递减策略:
      alpha_k = step_length / (1 + 0.1 * k)
    
    正则化:
      J_reg = J + 0.5 * beta * ||L m||^2
    其中 L 为离散 Laplacian，beta 为正则化参数。
    
    Parameters
    ----------
    m_init : ndarray
        初始速度模型。
    d_obs : ndarray, shape (nt+1, nx)
        观测数据。
    dx, dt : float
        网格参数。
    nt : int
        时间步数。
    source_pos : int
        震源位置。
    source_fn : callable
        震源时间函数。
    n_iter : int
        反演迭代次数。
    step_length : float
        初始步长。
    boundary : str
        边界条件。
    verbose : bool
        是否打印迭代信息。
    
    Returns
    -------
    m_history : list
        每步的模型。
    misfit_history : list
        每步的 misfit。
    """
    m = m_init.copy()
    m_history = [m.copy()]
    misfit_history = []
    beta_reg = 1e-4  # 正则化参数
    
    for k in range(n_iter):
        grad, misfit = adjoint_state_gradient_1d(
            m, d_obs, dx, dt, nt, source_pos, source_fn, boundary=boundary
        )
        
        # Tikhonov 正则化梯度
        reg_term = np.zeros_like(m)
        for i in range(1, len(m) - 1):
            reg_term[i] = -2 * m[i] + m[i - 1] + m[i + 1]
        reg_term[0] = 0.0
        reg_term[-1] = 0.0
        grad += beta_reg * reg_term / dx ** 2
        
        alpha = step_length / (1.0 + 0.1 * k)
        m = m - alpha * grad
        # 速度约束
        m = np.clip(m, 1000.0, 8000.0)
        
        m_history.append(m.copy())
        misfit_history.append(misfit)
        
        if verbose and k % 5 == 0:
            print(f"  FWI Iteration {k}: misfit = {misfit:.6e}, alpha = {alpha:.3e}")
    
    return m_history, misfit_history


def tomography_traveltime_1d(m, dx, source_pos, receiver_positions):
    """
    一维地震层析成像的旅行时计算。
    
    射线理论旅行时:
      T(x_r) = integral_{x_s}^{x_r} 1/c(x) dx
    
    对于一维水平层状介质，使用梯形法则数值积分:
      T = sum_i dx / c_i
    
    Parameters
    ----------
    m : ndarray, shape (nx,)
        速度模型。
    dx : float
        网格间距。
    source_pos : int
        震源索引。
    receiver_positions : list of int
        接收器索引列表。
    
    Returns
    -------
    traveltimes : ndarray
        各接收器的旅行时。
    """
    c = np.asarray(m, dtype=float)
    traveltimes = np.zeros(len(receiver_positions))
    for ir, rec_pos in enumerate(receiver_positions):
        i1 = min(source_pos, rec_pos)
        i2 = max(source_pos, rec_pos)
        if i1 == i2:
            traveltimes[ir] = 0.0
        else:
            traveltimes[ir] = np.sum(dx / c[i1:i2])
    return traveltimes
