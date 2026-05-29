"""
neural_mass_dynamics.py
神经质量模型动力学系统

融合种子项目:
  - 863_pendulum_ode_period: ODE参数管理与周期分析思想
  - 059_autocatalytic_ode: 自催化反应动力学 → 神经元兴奋-抑制耦合

科学背景:
  采用扩展的Wilson-Cowan神经质量模型描述神经元群体的平均场动力学:

    τ_e dE/dt = -E + S_e(a_ee E - a_ei I + P + u(t)) + ξ_e(t)
    τ_i dI/dt = -I + S_i(a_ie E - a_ii I + Q)      + ξ_i(t)

  其中 E(t), I(t) 分别为兴奋性与抑制性神经元群体的归一化活动率，
  S(x) = 1/(1+exp(-kx+θ)) 为S型激活函数，
  u(t) 为外部控制输入（如经颅直流刺激tDCS），
  ξ(t) 代表突触噪声的Wiener过程增量。

  控制目标: 通过最优控制策略 u*(t) 使系统从初始状态驱动到目标theta振荡模式。
"""

import numpy as np
from typing import Tuple, Callable, Optional

# ============================================================================
# 全局默认参数（类似pendulum_parameters的persistent机制）
# ============================================================================
_DEFAULT_PARAMS = {
    "tau_e": 4.0,      # 兴奋性时间常数 (ms)
    "tau_i": 12.0,     # 抑制性时间常数 (ms)
    "a_ee": 16.0,      # 兴奋→兴奋耦合强度
    "a_ei": 12.0,      # 抑制→兴奋耦合强度
    "a_ie": 15.0,      # 兴奋→抑制耦合强度
    "a_ii": 3.0,       # 抑制→抑制耦合强度
    "k_e": 1.3,        # 兴奋性Sigmoid增益
    "k_i": 2.0,        # 抑制性Sigmoid增益
    "theta_e": 4.0,    # 兴奋性Sigmoid阈值
    "theta_i": 3.7,    # 抑制性Sigmoid阈值
    "P": 1.0,          # 兴奋性外部输入基线
    "Q": 0.0,          # 抑制性外部输入基线
    "sigma_e": 0.05,   # 兴奋性噪声强度
    "sigma_i": 0.03,   # 抑制性噪声强度
    "t0": 0.0,
    "tstop": 200.0,
    "y0": np.array([0.1, 0.05]),
}


def get_neural_parameters(
    tau_e: Optional[float] = None,
    tau_i: Optional[float] = None,
    a_ee: Optional[float] = None,
    a_ei: Optional[float] = None,
    a_ie: Optional[float] = None,
    a_ii: Optional[float] = None,
    k_e: Optional[float] = None,
    k_i: Optional[float] = None,
    theta_e: Optional[float] = None,
    theta_i: Optional[float] = None,
    P: Optional[float] = None,
    Q: Optional[float] = None,
    sigma_e: Optional[float] = None,
    sigma_i: Optional[float] = None,
    t0: Optional[float] = None,
    tstop: Optional[float] = None,
    y0: Optional[np.ndarray] = None,
) -> Tuple:
    """
    返回Wilson-Cowan神经质量模型参数。
    若提供参数则更新默认值（类似MATLAB persistent机制）。
    """
    global _DEFAULT_PARAMS
    # 更新提供的参数
    local_vars = locals()
    for key in _DEFAULT_PARAMS:
        if local_vars[key] is not None:
            _DEFAULT_PARAMS[key] = local_vars[key]

    return (
        _DEFAULT_PARAMS["tau_e"],
        _DEFAULT_PARAMS["tau_i"],
        _DEFAULT_PARAMS["a_ee"],
        _DEFAULT_PARAMS["a_ei"],
        _DEFAULT_PARAMS["a_ie"],
        _DEFAULT_PARAMS["a_ii"],
        _DEFAULT_PARAMS["k_e"],
        _DEFAULT_PARAMS["k_i"],
        _DEFAULT_PARAMS["theta_e"],
        _DEFAULT_PARAMS["theta_i"],
        _DEFAULT_PARAMS["P"],
        _DEFAULT_PARAMS["Q"],
        _DEFAULT_PARAMS["sigma_e"],
        _DEFAULT_PARAMS["sigma_i"],
        _DEFAULT_PARAMS["t0"],
        _DEFAULT_PARAMS["tstop"],
        _DEFAULT_PARAMS["y0"],
    )


def sigmoid_activation(x: float, k: float, theta: float) -> float:
    """
    S型激活函数（神经元发放率响应函数）:
        S(x) = 1 / (1 + exp(-k*(x - theta)))

    具有边界保护: 当 |x| 极大时避免溢出。
    """
    if not np.isfinite(x):
        return 0.0
    # 防止指数溢出
    arg = -k * (x - theta)
    if arg > 700.0:
        return 0.0
    if arg < -700.0:
        return 1.0
    return 1.0 / (1.0 + np.exp(arg))


# 向量化版本
_sigmoid_vec = np.vectorize(sigmoid_activation, otypes=[float])


def neural_mass_deriv(
    t: float,
    y: np.ndarray,
    control_fn: Optional[Callable[[float, np.ndarray], float]] = None,
) -> np.ndarray:
    """
    Wilson-Cowan神经质量模型右侧导数:

        dE/dt = (1/τ_e) * [ -E + S_e(a_ee E - a_ei I + P + u(t)) ]
        dI/dt = (1/τ_i) * [ -I + S_i(a_ie E - a_ii I + Q) ]

    Parameters
    ----------
    t : float
        当前时间 (ms)
    y : np.ndarray, shape (2,)
        状态向量 [E, I]
    control_fn : callable or None
        控制输入函数 u(t, y)

    Returns
    -------
    dydt : np.ndarray, shape (2,)
        时间导数 [dE/dt, dI/dt]
    """
    (
        tau_e, tau_i,
        a_ee, a_ei, a_ie, a_ii,
        k_e, k_i, theta_e, theta_i,
        P, Q,
        sigma_e, sigma_i,
        t0, tstop, y0,
    ) = get_neural_parameters()

    E, I = y[0], y[1]

    # 控制输入（边界保护）
    u = 0.0
    if control_fn is not None:
        try:
            u = float(control_fn(t, y))
        except Exception:
            u = 0.0
    # 限制控制输入幅值，模拟生理约束（|u| ≤ 5 mA/cm^2）
    u = np.clip(u, -5.0, 5.0)

    # 兴奋性输入
    exc_input = a_ee * E - a_ei * I + P + u
    # 抑制性输入
    inh_input = a_ie * E - a_ii * I + Q

    # 激活函数
    Se = sigmoid_activation(exc_input, k_e, theta_e)
    Si = sigmoid_activation(inh_input, k_i, theta_i)

    # 边界保护：活动率必须在 [0,1] 内
    Se = np.clip(Se, 0.0, 1.0)
    Si = np.clip(Si, 0.0, 1.0)

    # TODO: Hole 1 — 补充Wilson-Cowan神经质量模型的右侧导数计算
    # 提示: dE/dt = (1/τ_e) * [ -E + S_e(a_ee E - a_ei I + P + u) ]
    #       dI/dt = (1/τ_i) * [ -I + S_i(a_ie E - a_ii I + Q) ]
    pass


def neural_mass_jacobian(
    y: np.ndarray,
    control_fn: Optional[Callable[[float, np.ndarray], float]] = None,
) -> np.ndarray:
    """
    计算神经质量模型关于状态 y 的Jacobian矩阵 J = ∂f/∂y:

        J = [ ∂f1/∂E   ∂f1/∂I ]
            [ ∂f2/∂E   ∂f2/∂I ]

    其中 f1 = dE/dt, f2 = dI/dt。

    ∂Se/∂x = k_e * Se * (1 - Se)
    ∂Si/∂x = k_i * Si * (1 - Si)
    """
    (
        tau_e, tau_i,
        a_ee, a_ei, a_ie, a_ii,
        k_e, k_i, theta_e, theta_i,
        P, Q,
        sigma_e, sigma_i,
        t0, tstop, y0,
    ) = get_neural_parameters()

    E, I = y[0], y[1]

    u = 0.0
    if control_fn is not None:
        try:
            u = float(control_fn(0.0, y))
        except Exception:
            u = 0.0
    u = np.clip(u, -5.0, 5.0)

    exc_input = a_ee * E - a_ei * I + P + u
    inh_input = a_ie * E - a_ii * I + Q

    Se = sigmoid_activation(exc_input, k_e, theta_e)
    Si = sigmoid_activation(inh_input, k_i, theta_i)

    dSe = k_e * Se * (1.0 - Se)
    dSi = k_i * Si * (1.0 - Si)

    J = np.zeros((2, 2))
    J[0, 0] = (-1.0 + a_ee * dSe) / tau_e
    J[0, 1] = (-a_ei * dSe) / tau_e
    J[1, 0] = (a_ie * dSi) / tau_i
    J[1, 1] = (-1.0 - a_ii * dSi) / tau_i

    return J


def neural_oscillation_period(
    linearized: bool = True,
) -> float:
    """
    估计神经质量模型在平衡点附近的振荡周期。

    若 linearized=True，在零状态附近线性化并计算特征值:
        λ = α ± iω  →  T = 2π/|ω|

    Returns
    -------
    period : float
        估计周期 (ms)
    """
    (
        tau_e, tau_i,
        a_ee, a_ei, a_ie, a_ii,
        k_e, k_i, theta_e, theta_i,
        P, Q,
        sigma_e, sigma_i,
        t0, tstop, y0,
    ) = get_neural_parameters()

    if linearized:
        # 在 E≈0, I≈0 处线性化
        S0e = sigmoid_activation(P, k_e, theta_e)
        S0i = sigmoid_activation(Q, k_i, theta_i)
        dSe0 = k_e * S0e * (1.0 - S0e)
        dSi0 = k_i * S0i * (1.0 - S0i)

        A = np.array([
            [(-1.0 + a_ee * dSe0) / tau_e, (-a_ei * dSe0) / tau_e],
            [(a_ie * dSi0) / tau_i, (-1.0 - a_ii * dSi0) / tau_i],
        ])

        eigenvalues = np.linalg.eigvals(A)
        imag_parts = np.imag(eigenvalues)
        omega = np.max(np.abs(imag_parts))
        if omega < 1e-12:
            # 无振荡，返回一个很大的周期值
            return 1e6
        period = 2.0 * np.pi / omega
    else:
        # 非线性数值积分求周期（简化处理）
        period = 2.0 * np.pi * np.sqrt(tau_e * tau_i)

    return float(period)


def compute_running_cost(
    y: np.ndarray,
    u: float,
    y_target: np.ndarray,
    Q_mat: Optional[np.ndarray] = None,
    R_scalar: float = 1.0,
) -> float:
    """
    计算LQR型运行代价:

        L(y,u) = (y - y_target)^T Q (y - y_target) + R u^2

    Parameters
    ----------
    y : np.ndarray
        当前状态
    u : float
        控制输入
    y_target : np.ndarray
        目标状态
    Q_mat : np.ndarray or None
        状态权重矩阵 (2x2)
    R_scalar : float
        控制权重

    Returns
    -------
    cost : float
        标量代价值
    """
    if Q_mat is None:
        Q_mat = np.eye(len(y))
    dy = y - y_target
    state_cost = float(dy @ Q_mat @ dy)
    control_cost = R_scalar * u * u
    return state_cost + control_cost


def compute_terminal_cost(
    y: np.ndarray,
    y_target: np.ndarray,
    P_mat: Optional[np.ndarray] = None,
) -> float:
    """
    终端代价:

        Φ(y) = (y - y_target)^T P (y - y_target)
    """
    if P_mat is None:
        P_mat = np.eye(len(y))
    dy = y - y_target
    return float(dy @ P_mat @ dy)
