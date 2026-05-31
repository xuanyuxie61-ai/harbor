"""
particle_drift.py
带电粒子在托卡马克磁场中的引导中心漂移运动。

核心物理模型：
  刚体欧拉方程与引导中心运动方程在数学结构上具有深刻的类比性。
  刚体自由旋转的 Euler 方程为：

      dω₁/dt = (1/I₃ - 1/I₂) ω₂ ω₃
      dω₂/dt = (1/I₁ - 1/I₃) ω₁ ω₃
      dω₃/dt = (1/I₂ - 1/I₁) ω₁ ω₂

  引导中心在弯曲磁场中的漂移速度可写为：

      v_∇B = (μ / (q B)) (B × ∇B) / B
      v_κ   = (m v_∥² / (q B)) (B × κ) / B

  其中 μ = m v_⊥² / (2B) 为磁矩，κ = (b̂ · ∇) b̂ 为磁场曲率。

  在简化模型中，我们将引导中心三个速度分量 (v_R, v_Z, v_∥) 的动力学
  类比为刚体 Euler 方程，非线性耦合项来自磁场不均匀性与曲率效应：

      dv_R/dt  =  α₁ v_Z v_∥  +  S_R(t)
      dv_Z/dt  =  α₂ v_R v_∥  +  S_Z(t)
      dv_∥/dt  =  α₃ v_R v_Z  +  S_∥(t)

  其中 α_i = (1/I_j - 1/I_k) 为类比系数，S(t) 为外部电场/碰撞源项。

数值方法：
  采用二阶 Runge-Kutta (RK2) 时间积分，保持能量守恒性质（类比刚体
  旋转的角动量守恒）。
"""

import numpy as np
from parameters import get_drift_params


def drift_derivative(t, y, i1, i2, i3, E_field=None, collision_freq=0.0):
    """
    引导中心漂移速度导数（类比刚体 Euler 方程）。

    参数
    ------
    t : float
        时间 [s]。
    y : ndarray, shape (3,)
        [v_R, v_Z, v_∥] 引导中心速度分量 [m/s]。
    i1, i2, i3 : float
        类比惯性矩参数。
    E_field : callable, optional
        外部电场函数 E(t) -> ndarray(3)。
    collision_freq : float
        碰撞频率 [Hz]。

    返回
    ------
    dydt : ndarray, shape (3,)
        速度时间导数。
    """
    y = np.asarray(y, dtype=float)
    if y.shape != (3,):
        raise ValueError("状态向量必须为 3 维")

    vR, vZ, vpar = y

    # 非线性耦合项（类比刚体 Euler 方程）
    dvR_dt = (1.0 / i3 - 1.0 / i2) * vZ * vpar
    dvZ_dt = (1.0 / i1 - 1.0 / i3) * vR * vpar
    dvpar_dt = (1.0 / i2 - 1.0 / i1) * vR * vZ

    # 外部电场驱动（Lorentz 力类比）
    if E_field is not None:
        E = np.asarray(E_field(t), dtype=float)
        if E.shape != (3,):
            raise ValueError("电场向量必须为 3 维")
        q_over_m = 1.0e8  # 简化有效荷质比 [C/kg]
        dvR_dt += q_over_m * E[0]
        dvZ_dt += q_over_m * E[1]
        dvpar_dt += q_over_m * E[2]

    # 碰撞阻尼（简化 Fokker-Planck 摩擦项）
    dvR_dt -= collision_freq * vR
    dvZ_dt -= collision_freq * vZ
    dvpar_dt -= collision_freq * vpar

    return np.array([dvR_dt, dvZ_dt, dvpar_dt], dtype=float)


def rk2_integrate(dydt_func, t_span, y0, n_steps=10000):
    """
    二阶 Runge-Kutta 时间积分器。

    算法
    ----
    对于方程 dy/dt = f(t, y)：
        k1 = h f(t_n, y_n)
        k2 = h f(t_n + h, y_n + k1)
        y_{n+1} = y_n + (k1 + k2) / 2

    参数
    ------
    dydt_func : callable
        导数函数 f(t, y) -> ndarray。
    t_span : tuple (t0, tstop)
    y0 : ndarray
        初始条件。
    n_steps : int
        积分步数。

    返回
    ------
    t_arr : ndarray
    y_arr : ndarray, shape (n_steps+1, len(y0))
    """
    t0, tstop = t_span
    h = (tstop - t0) / n_steps
    if h <= 0:
        raise ValueError("时间步长必须为正")

    dim = len(np.asarray(y0))
    t_arr = np.zeros(n_steps + 1)
    y_arr = np.zeros((n_steps + 1, dim))
    t_arr[0] = t0
    y_arr[0, :] = np.asarray(y0, dtype=float)

    for n in range(n_steps):
        tn = t_arr[n]
        yn = y_arr[n, :]
        k1 = h * dydt_func(tn, yn)
        k2 = h * dydt_func(tn + h, yn + k1)
        y_arr[n + 1, :] = yn + 0.5 * (k1 + k2)
        t_arr[n + 1] = tn + h

    return t_arr, y_arr


def simulate_guiding_center(drift_params=None, n_steps=5000):
    """
    模拟单个带电粒子在托卡马克中的引导中心运动。

    参数
    ------
    drift_params : dict or None
        漂移参数字典，默认调用 get_drift_params()。
    n_steps : int
        RK2 积分步数。

    返回
    ------
    t_arr : ndarray
        时间数组 [s]。
    y_arr : ndarray, shape (n_steps+1, 3)
        [v_R, v_Z, v_∥] 轨迹。
    energy : ndarray
        归一化动能历史。
    """
    if drift_params is None:
        drift_params = get_drift_params()

    i1 = drift_params["i1"]
    i2 = drift_params["i2"]
    i3 = drift_params["i3"]
    t0 = drift_params["t0"]
    y0 = drift_params["y0"]
    tstop = drift_params["tstop"]

    # 构造简谐电场 E(t) = E0 cos(ω t)
    omega = 2.0 * np.pi * 1.0e3  # 1 kHz 扰动
    E0 = np.array([1.0e-4, 0.5e-4, 0.2e-4])

    def E_field(t):
        return E0 * np.cos(omega * t)

    # 碰撞频率（电子-离子碰撞，简化）
    nu_ei = 1.0e2  # [Hz]

    def deriv(t, y):
        return drift_derivative(t, y, i1, i2, i3, E_field=E_field, collision_freq=nu_ei)

    t_arr, y_arr = rk2_integrate(deriv, (t0, tstop), y0, n_steps=n_steps)

    # 计算归一化动能（类比刚体旋转动能 T = 0.5 Σ I_i ω_i²）
    energy = 0.5 * (i1 * y_arr[:, 0] ** 2 +
                    i2 * y_arr[:, 1] ** 2 +
                    i3 * y_arr[:, 2] ** 2)

    return t_arr, y_arr, energy


def compute_magnetic_moment(v_perp, B):
    """
    计算磁矩 μ = m v_⊥² / (2B)。

    参数
    ------
    v_perp : float or ndarray
        垂直速度 [m/s]。
    B : float or ndarray
        磁场强度 [T]。

    返回
    ------
    mu : float or ndarray
        磁矩 [J/T]。
    """
    m_eff = 3.34e-27  # 氘核质量 [kg]
    B = np.asarray(B)
    B_safe = np.where(np.abs(B) < 1e-15, 1e-15, B)
    return 0.5 * m_eff * np.asarray(v_perp) ** 2 / B_safe


def compute_adiabatic_invariant(y_arr, B_arr):
    """
    检查磁矩绝热不变性（J = ∮ p_∥ dℓ 近似）。

    参数
    ------
    y_arr : ndarray
        速度轨迹。
    B_arr : ndarray
        对应位置的磁场强度。

    返回
    ------
    mu_arr : ndarray
        磁矩历史。
    mu_relative_std : float
        磁矩相对标准差（衡量绝热不变性保持程度）。
    """
    v_perp_sq = y_arr[:, 0] ** 2 + y_arr[:, 1] ** 2
    mu_arr = compute_magnetic_moment(np.sqrt(v_perp_sq), B_arr)
    if np.mean(mu_arr) > 1e-30:
        mu_relative_std = np.std(mu_arr) / np.mean(mu_arr)
    else:
        mu_relative_std = 0.0
    return mu_arr, mu_relative_std
