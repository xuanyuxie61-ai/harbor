"""
ode_midpoint_phasefield.py
===========================
中点法 (Theta-Method) 隐式 ODE 求解器,
用于 CALPHAD 相场动力学方程的时间积分。

种子项目 828_ode_midpoint:
  中点法: y_{n+1} = y_n + h * f(t + h/2, (y_n + y_{n+1})/2)
  使用不动点迭代求解隐式方程。

映射到 CALPHAD:
  将 Allen-Cahn 方程 (有序参量演化):
    d(eta)/dt = -L * dF/d(eta)

  其中 eta 为有序参量 (长程序参量), F 为自由能泛函。

  对于 Fe-C 体系 BCC→FCC 转变:
    eta = 0: 完全 BCC (α-Fe)
    eta = 1: 完全 FCC (γ-Fe)
    0 < eta < 1: 混合态

  自由能:
    F(eta) = (1-phi(eta)) * G_BCC(x,T) + phi(eta) * G_FCC(x,T)
           + W * g(eta)
    其中 phi(eta) = eta^3 * (6*eta^2 - 15*eta + 10) (平滑插值)
          g(eta) = eta^2 * (1-eta)^2 (双阱势)
          W: 势垒高度

  中点法求解:
    eta_{n+1} = eta_n + dt * f(eta_mid)
    eta_mid = (eta_n + eta_{n+1}) / 2
    使用 Picard 迭代求 eta_mid
"""

import numpy as np
from calphad_fec_constants import R_GAS


def interpolation_phi(eta):
    """
    平滑插值函数 (五阶多项式):
        phi(eta) = eta^3 * (6*eta^2 - 15*eta + 10)

    满足:
        phi(0) = 0, phi(1) = 1
        phi'(0) = phi'(1) = 0
        phi''(0) = phi''(1) = 0

    Parameters
    ----------
    eta : float or np.ndarray
        有序参量

    Returns
    -------
    float or np.ndarray
    """
    eta = np.clip(eta, 0.0, 1.0)
    return eta ** 3 * (6.0 * eta ** 2 - 15.0 * eta + 10.0)


def double_well_g(eta):
    """
    双阱势函数:
        g(eta) = eta^2 * (1 - eta)^2

    极小值在 eta=0 和 eta=1, 极大值在 eta=0.5。

    Parameters
    ----------
    eta : float or np.ndarray

    Returns
    -------
    float or np.ndarray
    """
    return eta ** 2 * (1.0 - eta) ** 2


def dg_deta(eta):
    """
    双阱势的导数:
        g'(eta) = 2*eta*(1-eta)*(1-2*eta)
    """
    return 2.0 * eta * (1.0 - eta) * (1.0 - 2.0 * eta)


def dphi_deta(eta):
    """
    平滑插值的导数:
        phi'(eta) = 30 * eta^2 * (1-eta)^2
    """
    eta = np.clip(eta, 0.0, 1.0)
    return 30.0 * eta ** 2 * (1.0 - eta) ** 2


def allen_cahn_rhs(eta, x_C, T, L_kinetic, W_barrier, phase_a, phase_b):
    """
    Allen-Cahn 方程右端项:

    d(eta)/dt = -L * [dphi/deta * (G_FCC - G_BCC) + W * g'(eta)]

    Parameters
    ----------
    eta : float
        有序参量
    x_C : float
        碳摩尔分数
    T : float
        温度 (K)
    L_kinetic : float
        动力学系数
    W_barrier : float
        势垒高度 (J/mol)
    phase_a : str
        alpha 相 (eta=0)
    phase_b : str
        beta 相 (eta=1)

    Returns
    -------
    float
        d(eta)/dt
    """
    from gibbs_energy_calphad import gibbs_substitutional

    G_a = gibbs_substitutional(x_C, T, phase_a)
    G_b = gibbs_substitutional(x_C, T, phase_b)

    dF_deta = dphi_deta(eta) * (G_b - G_a) + W_barrier * dg_deta(eta)

    return -L_kinetic * dF_deta


def ode_midpoint_implicit(f, y0, t_span, n_steps, max_picard=30, tol=1e-10):
    """
    隐式中点法 ODE 求解器。

    y_{n+1} = y_n + h * f(t_n + h/2, (y_n + y_{n+1})/2)

    使用 Picard 迭代求解 y_{n+1}:
    1. 初始猜测: y_{n+1}^{(0)} = y_n
    2. y_mid^{(k)} = (y_n + y_{n+1}^{(k)}) / 2
    3. y_{n+1}^{(k+1)} = y_n + h * f(t_mid, y_mid^{(k)})
    4. 检查收敛

    Parameters
    ----------
    f : callable
        右端项函数 f(t, y)
    y0 : float or np.ndarray
        初始值
    t_span : tuple
        (t_start, t_end)
    n_steps : int
        步数
    max_picard : int
        Picard 最大迭代次数
    tol : float
        收敛容差

    Returns
    -------
    dict
        {
            't': np.ndarray,
            'y': np.ndarray,
            'converged': bool,
            'n_picard_total': int,
        }
    """
    t0, tf = t_span
    h = (tf - t0) / n_steps
    t = np.linspace(t0, tf, n_steps + 1)

    is_scalar = np.isscalar(y0)
    if is_scalar:
        y_arr = np.zeros(n_steps + 1)
        y_arr[0] = y0
    else:
        y_arr = np.zeros((n_steps + 1, len(y0)))
        y_arr[0] = y0

    converged = True
    n_picard_total = 0

    for n in range(n_steps):
        y_n = y_arr[n]
        y_next = y_n.copy() if not is_scalar else y_n

        for picard_iter in range(max_picard):
            if is_scalar:
                y_mid = 0.5 * (y_n + y_next)
                f_mid = f(t[n] + h / 2, y_mid)
                y_new = y_n + h * f_mid
            else:
                y_mid = 0.5 * (y_n + y_next)
                f_mid = f(t[n] + h / 2.0, y_mid)
                y_new = y_n + h * f_mid

            # 收敛检查
            if is_scalar:
                diff = abs(y_new - y_next)
            else:
                diff = np.max(np.abs(y_new - y_next))

            y_next = y_new
            n_picard_total += 1

            if diff < tol:
                break

        if diff >= tol:
            converged = False

        y_arr[n + 1] = y_next

    return {
        't': t,
        'y': y_arr,
        'converged': converged,
        'n_picard_total': n_picard_total,
    }


def phase_transformation_trajectory(x_C, T, L_kinetic=1e-3,
                                    W_barrier=5000.0,
                                    n_steps=200,
                                    phase_a='BCC', phase_b='FCC'):
    """
    计算 BCC→FCC 相变的有序参量演化轨迹。

    Parameters
    ----------
    x_C : float
        碳摩尔分数
    T : float
        温度 (K)
    L_kinetic : float
        动力学系数
    W_barrier : float
        势垒高度 (J/mol)
    n_steps : int
        时间步数
    phase_a : str
        初始相
    phase_b : str
        终态相

    Returns
    -------
    dict
        中点法求解结果
    """
    # 初始: 完全 alpha 相
    eta0 = 0.01

    def rhs(t, eta):
        return allen_cahn_rhs(eta, x_C, T, L_kinetic, W_barrier,
                              phase_a, phase_b)

    t_total = 1.0 / L_kinetic  # 特征时间

    return ode_midpoint_implicit(rhs, eta0, (0.0, t_total), n_steps)
