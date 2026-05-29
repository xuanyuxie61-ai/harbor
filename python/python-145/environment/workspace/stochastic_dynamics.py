"""
stochastic_dynamics.py
======================
博士级多因子随机动力学：利率期限结构的非线性扰动与混沌驱动

本模块将三个经典动力系统（Lorenz-96、Duffing、Oregonator）
映射为利率期限结构模型的多因子随机驱动源：

  1. Lorenz-96 系统：高维混沌过程，模拟市场微观结构中的高维噪声耦合。
     方程:
       dy_i/dt = (y_{i+1} - y_{i-2}) * y_{i-1} - y_i + F,  i = 1,...,N
     其中指标循环 wrapping，F 为外部强迫项。
     在金融意义下，F 对应央行的长期政策利率锚定水平。

  2. Duffing 振子：非线性受迫振荡，描述利率周期性波动与制度切换。
     方程:
       x'' + δ x' + α x + β x³ = γ cos(ω t)
     改写为一阶系统:
       y1' = y2
       y2' = -δ y2 - α y1 - β y1³ + γ cos(ω t)
     其中 β > 0 对应硬弹簧（利率上限约束），β < 0 对应软弹簧（利率下限约束）。

  3. Oregonator 系统：化学反应振荡器，模拟市场流动性冲击的传播与衰减。
     方程（无量纲化）:
       du/dt = (q v - u v + u (1 - u)) / η1
       dv/dt = (-q v - u v + f w) / η2
       dw/dt = u - w
     其中 u, v, w 分别对应：市场流动性水平、交易摩擦强度、订单簿深度。
     该系统的极限环行为对应流动性危机的周期性爆发。

综合多因子模型:
    将上述三个系统的输出通过非线性耦合矩阵投影到前向利率的波动率空间，
    形成多因子 HJM 框架下的随机驱动项。
"""

import numpy as np


# ---------------------------------------------------------------------------
# Lorenz-96 混沌系统
# ---------------------------------------------------------------------------

def lorenz96_parameters(n=4, force=8.0, perturb=0.001, t0=0.0, y0=None, tstop=30.0):
    """
    Lorenz-96 系统参数。

    Parameters
    ----------
    n : int
        系统维度，默认 4（小规模用于快速演示）。
    force : float
        外部强迫项 F，对应央行政策利率锚。
    perturb : float
        初始条件扰动幅度。
    t0 : float
        初始时间。
    y0 : np.ndarray or None
        初始条件；None 时使用 force 附近的随机扰动。
    tstop : float
        终止时间。

    Returns
    -------
    tuple
        (n, force, perturb, t0, y0, tstop)
    """
    if n < 3:
        raise ValueError("lorenz96_parameters: n 必须至少为 3")
    if y0 is None:
        s = perturb * np.random.randn(n)
        y0 = force * np.ones(n) + s
    y0 = np.asarray(y0, dtype=float)
    if y0.shape[0] != n:
        raise ValueError("lorenz96_parameters: y0 长度必须与 n 一致")
    return n, force, perturb, t0, y0, tstop


def lorenz96_deriv(t, y, force=8.0):
    """
    Lorenz-96 右端项。

    公式:
        dydt[i] = (y[i+1] - y[i-2]) * y[i-1] - y[i] + force
    指标循环: y[-1] = y[n-1], y[-2] = y[n-2], y[n] = y[0], y[n+1] = y[1]。

    Parameters
    ----------
    t : float
        当前时间（不显含）。
    y : np.ndarray, shape (n,)
        状态向量。
    force : float
        外部强迫项。

    Returns
    -------
    np.ndarray, shape (n,)
        时间导数。
    """
    y = np.asarray(y, dtype=float)
    n = y.shape[0]
    if n < 3:
        raise ValueError("lorenz96_deriv: y 长度必须至少为 3")

    i = np.arange(n)
    ip1 = np.roll(i, -1)
    im1 = np.roll(i, 1)
    im2 = np.roll(i, 2)

    dydt = (y[ip1] - y[im2]) * y[im1] - y[i] + force
    return dydt


# ---------------------------------------------------------------------------
# Duffing 振子
# ---------------------------------------------------------------------------

def duffing_parameters(alpha=1.0, beta=5.0, gamma=8.0, delta=0.02,
                       omega=0.5, t0=0.0, y0=None, tstop=100.0):
    """
    Duffing 振子参数。

    Parameters
    ----------
    alpha : float
        线性刚度系数。
    beta : float
        非线性刚度系数；beta > 0 为硬弹簧，beta < 0 为软弹簧。
    gamma : float
        外部激励幅值。
    delta : float
        阻尼系数。
    omega : float
        外部激励频率。
    t0, tstop : float
        时间区间。
    y0 : np.ndarray or None
        初始条件 [x, x']；None 时为 [1.0, 0.0]。

    Returns
    -------
    tuple
        (alpha, beta, gamma, delta, omega, t0, y0, tstop)
    """
    if y0 is None:
        y0 = np.array([1.0, 0.0], dtype=float)
    y0 = np.asarray(y0, dtype=float)
    if y0.shape[0] != 2:
        raise ValueError("duffing_parameters: y0 必须为二维向量 [x, x']")
    return alpha, beta, gamma, delta, omega, t0, y0, tstop


def duffing_deriv(t, y, alpha=1.0, beta=5.0, gamma=8.0, delta=0.02, omega=0.5):
    """
    Duffing 振子右端项。

    方程组:
        y1' = y2
        y2' = -δ y2 - α y1 - β y1³ + γ cos(ω t)

    Parameters
    ----------
    t : float
        当前时间。
    y : np.ndarray, shape (2,)
        状态向量 [x, x']。
    alpha, beta, gamma, delta, omega : float
        模型参数。

    Returns
    -------
    np.ndarray, shape (2,)
        时间导数。
    """
    y = np.asarray(y, dtype=float)
    if y.shape[0] != 2:
        raise ValueError("duffing_deriv: y 必须为二维向量")
    y1, y2 = y[0], y[1]
    dy1dt = y2
    dy2dt = -delta * y2 - alpha * y1 - beta * (y1 ** 3) + gamma * np.cos(omega * t)
    return np.array([dy1dt, dy2dt], dtype=float)


# ---------------------------------------------------------------------------
# Oregonator 化学反应系统（流动性冲击模型）
# ---------------------------------------------------------------------------

def oregonator_parameters(eta1=None, eta2=None, q=None, f=1.0,
                          t0=0.0, y0=None, tstop=25.0):
    """
    Oregonator 系统参数，基于 Field-Körös-Noyes 化学反应机理。

    原始动力学常数:
        k2 = 2.4e6, k3 = 1.28, k4 = 3.0e3, k5 = 33.6
        a = 0.06, b = 0.02, kc = 1.0

    无量纲参数推导:
        η1 = kc * b / (k5 * a)
        η2 = 2 * kc * k4 * b / (k2 * k5 * a)
        q  = 2 * k3 * k4 / (k2 * k5)

    Parameters
    ----------
    eta1, eta2, q : float or None
        无量纲参数；None 时使用上述化学反应机理默认值。
    f : float
        化学计量系数，控制振荡模式。
    t0, tstop : float
        时间区间。
    y0 : np.ndarray or None
        初始条件 [u, v, w]；None 时为 [1.0, 1.0, 1.0]。

    Returns
    -------
    tuple
        (eta1, eta2, q, f, t0, y0, tstop)
    """
    a = 0.06
    b = 0.02
    k2 = 2.4e6
    k3 = 1.28
    k4 = 3.0e3
    k5 = 33.6
    kc = 1.0

    if eta1 is None:
        eta1 = kc * b / (k5 * a)
    if eta2 is None:
        eta2 = 2.0 * kc * k4 * b / (k2 * k5 * a)
    if q is None:
        q = 2.0 * k3 * k4 / (k2 * k5)
    if y0 is None:
        y0 = np.array([1.0, 1.0, 1.0], dtype=float)
    y0 = np.asarray(y0, dtype=float)
    if y0.shape[0] != 3:
        raise ValueError("oregonator_parameters: y0 必须为三维向量")
    return eta1, eta2, q, f, t0, y0, tstop


def oregonator_deriv(t, y, eta1, eta2, q, f):
    """
    Oregonator 右端项。

    方程:
        du/dt = (q v - u v + u (1 - u)) / η1
        dv/dt = (-q v - u v + f w) / η2
        dw/dt = u - w

    Parameters
    ----------
    t : float
        当前时间（不显含）。
    y : np.ndarray, shape (3,)
        状态向量 [u, v, w]。
    eta1, eta2, q, f : float
        模型参数。

    Returns
    -------
    np.ndarray, shape (3,)
        时间导数。
    """
    y = np.asarray(y, dtype=float)
    if y.shape[0] != 3:
        raise ValueError("oregonator_deriv: y 必须为三维向量")
    u, v, w = y[0], y[1], y[2]
    # 数值鲁棒性：截断极端值防止溢出
    u = np.clip(u, -100.0, 100.0)
    v = np.clip(v, -100.0, 100.0)
    w = np.clip(w, -100.0, 100.0)
    dudt = (q * v - u * v + u * (1.0 - u)) / eta1
    dvdt = (-q * v - u * v + f * w) / eta2
    dwdt = u - w
    return np.array([dudt, dvdt, dwdt], dtype=float)


# ---------------------------------------------------------------------------
# 多因子耦合投影
# ---------------------------------------------------------------------------

def multi_factor_coupling(t, lorenz_state, duffing_state, oregonator_state,
                          n_factors=3, coupling_matrix=None):
    """
    将三个动力学系统的状态投影到 HJM 多因子波动率空间。

    投影公式:
        σ_k(t) = Σ_{j=1}^{3} C_{k,j} * ξ_j(t),  k = 1,...,n_factors
    其中 ξ_1 = ||lorenz_state||_2 / sqrt(N) 为归一化混沌强度，
          ξ_2 = duffing_state[0] 为位移项（利率周期位置），
          ξ_3 = oregonator_state[0] 为流动性水平。

    Parameters
    ----------
    t : float
        当前时间。
    lorenz_state : np.ndarray
        Lorenz-96 状态向量。
    duffing_state : np.ndarray, shape (2,)
        Duffing 状态向量。
    oregonator_state : np.ndarray, shape (3,)
        Oregonator 状态向量。
    n_factors : int
        输出因子数。
    coupling_matrix : np.ndarray or None
        耦合矩阵，shape (n_factors, 3)；None 时使用默认随机正交耦合。

    Returns
    -------
    np.ndarray, shape (n_factors,)
        投影后的多因子波动率驱动项。
    """
    lorenz_state = np.asarray(lorenz_state, dtype=float)
    duffing_state = np.asarray(duffing_state, dtype=float)
    oregonator_state = np.asarray(oregonator_state, dtype=float)

    # TODO HOLE_3: 实现多因子耦合投影。
    # 将三个动力学系统的状态归一化并投影到 HJM 波动率空间：
    #   xi1 = ||lorenz_state||_2 / sqrt(N)   （归一化混沌强度）
    #   xi2 = duffing_state[0]                （位移项，利率周期位置）
    #   xi3 = oregonator_state[0]             （流动性水平）
    # 若 coupling_matrix 为 None，使用默认 3x3 耦合矩阵。
    # 然后计算 sigma = coupling_matrix @ xi，并保证非负性（波动率必须非负）。
    # 注意：coupling_matrix 的形状必须为 (n_factors, 3)。
    raise NotImplementedError("HOLE_3: 多因子耦合投影尚未实现")
