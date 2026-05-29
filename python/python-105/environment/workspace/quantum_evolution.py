r"""
quantum_evolution.py
====================
SPDC 量子态耦合演化求解器 —— 融合原项目 1041_robertson_ode (刚性 ODE 系统)
与 063_backward_euler (隐式后向欧拉时间积分)。

物理模型
--------
在相互作用绘景中，Type-II SPDC 过程的三模耦合方程可写为

.. math::
    \frac{d a_s}{d t} &= -\frac{\gamma_s}{2} a_s + \kappa(t) a_i^\dagger + f_s(t) \\
    \frac{d a_i}{d t} &= -\frac{\gamma_i}{2} a_i + \kappa(t) a_s^\dagger + f_i(t) \\
    \frac{d a_p}{d t} &= -\frac{\gamma_p}{2} a_p - \kappa(t) a_s a_i + f_p(t)

其中 :math:`a_{s,i,p}` 为信号、闲置、泵浦模式的湮灭算符，
:math:`\kappa(t)` 为时变耦合强度，:math:`\gamma` 为腔/波导损耗率，
:math:`f` 为噪声算符（热噪声或真空涨落）。

对于平均场（c-number 近似），定义态向量
:math:`y = [\langle a_s\rangle, \langle a_i\rangle, \langle a_p\rangle]^T`，
方程形如 Robertson 型刚性系统：

.. math::
    \frac{dy_1}{dt} &= -\frac{\gamma_s}{2} y_1 + \kappa^* y_2^* y_3 + f_1 \\
    \frac{dy_2}{dt} &= -\frac{\gamma_i}{2} y_2 + \kappa^* y_1^* y_3 + f_2 \\
    \frac{dy_3}{dt} &= -\frac{\gamma_p}{2} y_3 - \kappa y_1 y_2 + f_3

当 :math:`\gamma_s \sim \gamma_i \ll |\kappa| \ll \gamma_p` 时，系统呈现刚性，
显式积分需要极小时步。本模块采用 **后向欧拉 + Newton-Raphson** 隐式迭代：

.. math::
    y_{n+1} = y_n + h \, f(t_{n+1}, y_{n+1})

残差

.. math::
    R(y_{n+1}) = y_{n+1} - y_n - h f(t_{n+1}, y_{n+1}) = 0

Newton 迭代：

.. math::
    J_R \, \delta y = -R, \quad y_{n+1}^{(k+1)} = y_{n+1}^{(k)} + \delta y

其中 Jacobian :math:`J_R = I - h J_f`。
"""

import numpy as np
from linear_solver import gauss_elimination_partial_pivot


def spdc_derivative(t: float, y: np.ndarray,
                    gamma: np.ndarray, kappa_func: callable,
                    f_noise: callable) -> np.ndarray:
    r"""
    SPDC 平均场刚性 ODE 右端项。

    参数
    ----
    t : float
        时间。
    y : np.ndarray, shape (3,)
        [y_s, y_i, y_p]，复振幅。
    gamma : np.ndarray, shape (3,)
        损耗率 [:math:`\gamma_s, \gamma_i, \gamma_p`]。
    kappa_func : callable(t) -> complex
        时变耦合系数。
    f_noise : callable(t) -> np.ndarray, shape (3,)
        噪声驱动项。

    返回
    ----
    dydt : np.ndarray, shape (3,)
    """
    y = np.asarray(y, dtype=np.complex128)
    if y.shape != (3,):
        raise ValueError("y 必须为长度 3 的向量。")
    if np.any(gamma < 0.0):
        raise ValueError("损耗率 gamma 必须非负。")

    kappa = kappa_func(t)
    f = f_noise(t)

    dydt = np.zeros(3, dtype=np.complex128)
    dydt[0] = -0.5 * gamma[0] * y[0] + np.conj(kappa) * np.conj(y[1]) * y[2] + f[0]
    dydt[1] = -0.5 * gamma[1] * y[1] + np.conj(kappa) * np.conj(y[0]) * y[2] + f[1]
    dydt[2] = -0.5 * gamma[2] * y[2] - kappa * y[0] * y[1] + f[2]
    return dydt


def spdc_jacobian(t: float, y: np.ndarray,
                  gamma: np.ndarray, kappa_func: callable) -> np.ndarray:
    r"""
    右端项关于 y 的 Jacobian :math:`J_f = \partial f / \partial y`（3x3 复矩阵）。

    由于 y 为复变量，在 Newton 迭代中将其拆为实部/虚部，Jacobian 扩展为 6x6。
    """
    kappa = kappa_func(t)
    J = np.zeros((3, 3), dtype=np.complex128)
    J[0, 0] = -0.5 * gamma[0]
    J[0, 1] = np.conj(kappa) * np.conj(y[2])
    J[0, 2] = np.conj(kappa) * np.conj(y[1])

    J[1, 0] = np.conj(kappa) * np.conj(y[2])
    J[1, 1] = -0.5 * gamma[1]
    J[1, 2] = np.conj(kappa) * np.conj(y[0])

    J[2, 0] = -kappa * y[1]
    J[2, 1] = -kappa * y[0]
    J[2, 2] = -0.5 * gamma[2]
    return J


def backward_euler_spdc(y0: np.ndarray, t_span: tuple, n_steps: int,
                        gamma: np.ndarray, kappa_func: callable,
                        f_noise: callable,
                        newton_tol: float = 1e-10,
                        max_newton: int = 20) -> tuple:
    """
    后向欧拉隐式积分 SPDC 刚性 ODE。

    参数
    ----
    y0 : np.ndarray, shape (3,)
        初始复振幅。
    t_span : tuple (t0, tf)
    n_steps : int
        时间步数，> 0。
    gamma : np.ndarray, shape (3,)
    kappa_func, f_noise : callable
    newton_tol : float
        Newton 迭代残差阈值。
    max_newton : int
        每步最大 Newton 迭代次数。

    返回
    ----
    t : np.ndarray, shape (n_steps+1,)
    y : np.ndarray, shape (n_steps+1, 3)
        复振幅历史。
    """
    y0 = np.asarray(y0, dtype=np.complex128)
    if y0.shape != (3,):
        raise ValueError("y0 必须为长度 3 的向量。")
    if n_steps <= 0:
        raise ValueError("n_steps 必须为正。")

    t0, tf = t_span
    h = (tf - t0) / n_steps
    if h <= 0.0:
        raise ValueError("t_span 必须满足 tf > t0。")

    t = np.linspace(t0, tf, n_steps + 1)
    y = np.zeros((n_steps + 1, 3), dtype=np.complex128)
    y[0, :] = y0

    for n in range(n_steps):
        tp = t[n + 1]
        y_old = y[n, :]
        yp = y_old + h * spdc_derivative(t[n], y_old, gamma, kappa_func, f_noise)

        # Newton-Raphson for implicit equation
        for _ in range(max_newton):
            f_tp = spdc_derivative(tp, yp, gamma, kappa_func, f_noise)
            Jf = spdc_jacobian(tp, yp, gamma, kappa_func)
            R = yp - y_old - h * f_tp

            # Build 6x6 real Jacobian of residual: I - h * Jf
            J_R = np.eye(6, dtype=np.float64)
            J_R[:3, :3] -= h * Jf.real
            J_R[:3, 3:] += h * Jf.imag
            J_R[3:, :3] -= h * Jf.imag
            J_R[3:, 3:] -= h * Jf.real

            R_real = np.hstack([R.real, R.imag])
            try:
                delta = gauss_elimination_partial_pivot(J_R, -R_real)
            except ValueError:
                # Fallback to least-squares if singular
                delta = np.linalg.lstsq(J_R, -R_real, rcond=None)[0]

            yp += delta[:3] + 1j * delta[3:]
            if np.linalg.norm(R) < newton_tol:
                break

        y[n + 1, :] = yp

    return t, y


def robertson_like_conservation(y: np.ndarray) -> np.ndarray:
    """
    计算类似 Robertson 系统的守恒量（光子数守恒检查）。

    .. math::
        C(t) = |y_s|^2 + |y_i|^2 + |y_p|^2

    参数
    ----
    y : np.ndarray, shape (n, 3)

    返回
    ----
    C : np.ndarray, shape (n,)
    """
    return np.sum(np.abs(y) ** 2, axis=1)
