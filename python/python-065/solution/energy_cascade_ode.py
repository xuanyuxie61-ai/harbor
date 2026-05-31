"""
energy_cascade_ode.py

基于 437_flame_ode 核心算法的大气能量级串 ODE 模块。

原项目 flame_ode 描述了一个火焰半径的 stiff ODE：
    y' = y^2 - y^3 = y^2 (1 - y)
其精确解涉及 Lambert W 函数：
    y(t) = 1 / ( W(A * exp(A - t)) + 1 ),  A = (1/δ - 1)

在本气候归因框架中，我们将该模型类比为大气能量级串：
- y 表示极端天气系统的归一化能量强度
- y^2 项代表来自大尺度系统的能量输入（面积比例）
- y^3 项代表由于湍流耗散导致的能量损失（体积比例）
- 该 ODE 描述了极端事件从发展到饱和的动力学过程

核心科学公式：
- 能量级串方程（归一化形式）：
    dE/dt = α E^2 - β E^3
    归一化后：dy/dτ = y^2 - y^3,  y = E/E_eq, τ = α t
- Lambert W 函数定义：
    W(z) * exp(W(z)) = z
- 精确解：
    y(τ) = 1 / ( W( A * exp(A - τ) ) + 1 )
    其中 A = (1/y_0 - 1)
- 有限时间爆破：当 y_0 > 0 时，dy/dτ > 0 直到 y → 1
"""

import numpy as np
from scipy.special import lambertw


def lambert_w_approx(x, branch=0):
    """
    Lambert W 函数的近似计算（基于 437_lambert_w 的核心思想）。

    使用 scipy.special.lambertw 作为主要实现，但包含边界处理。
    """
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x, dtype=np.complex128)

    # 对于实数 x >= -1/e，主分支 W_0 为实数
    mask_real = (x >= -1.0 / np.e) & (x < np.inf)
    if np.any(mask_real):
        result[mask_real] = lambertw(x[mask_real], k=branch)

    # 对于 x < -1/e，返回复数（物理上不应出现）
    mask_complex = x < -1.0 / np.e
    if np.any(mask_complex):
        result[mask_complex] = lambertw(x[mask_complex], k=branch)

    return result


def flame_deriv(t, y):
    """
    火焰/能量级串 ODE 的右端项（基于 437_flame_deriv）。

    dy/dt = y^2 * (1 - y)
    """
    y = np.asarray(y)
    return y ** 2 * (1.0 - y)


def energy_cascade_exact(t, y0=0.01):
    """
    能量级串方程的精确解（基于 437_flame_exact）。

    Parameters
    ----------
    t : ndarray
        时间（归一化）。
    y0 : float
        初始能量强度（0 < y0 < 1）。

    Returns
    -------
    y : ndarray
        精确解。
    """
    t = np.asarray(t, dtype=np.float64)
    if y0 <= 0.0 or y0 >= 1.0:
        raise ValueError("y0 必须在 (0, 1) 之间")

    a = (1.0 - y0) / y0
    y = np.zeros_like(t)
    for i in range(t.size):
        arg = a * np.exp(a - t.flat[i])
        if arg < -1.0 / np.e:
            # 超出 Lambert W 定义域，使用数值解法
            y.flat[i] = 0.0
        else:
            w_val = lambertw(arg, k=0)
            y.flat[i] = 1.0 / (np.real(w_val) + 1.0)
    return y


def solve_energy_cascade_rk4(t_span, y0, n_steps=1000):
    """
    使用 RK4 数值求解能量级串方程。

    方程：dy/dt = y^2 - y^3
    """
    t0, tf = t_span
    h = (tf - t0) / n_steps
    t = np.linspace(t0, tf, n_steps + 1)
    y = np.zeros(n_steps + 1)
    y[0] = y0

    for i in range(n_steps):
        k1 = flame_deriv(t[i], y[i])
        k2 = flame_deriv(t[i] + 0.5 * h, y[i] + 0.5 * h * k1)
        k3 = flame_deriv(t[i] + 0.5 * h, y[i] + 0.5 * h * k2)
        k4 = flame_deriv(t[i] + h, y[i] + h * k3)
        y[i + 1] = y[i] + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # 边界处理
        if y[i + 1] > 1.0:
            y[i + 1] = 1.0
        elif y[i + 1] < 0.0:
            y[i + 1] = 0.0

    return t, y


def energy_saturation_time(y0, epsilon=0.99):
    """
    计算能量达到饱和（y = epsilon）所需的时间。

    精确公式：
        τ_sat = A - W( A * exp(A) * (1/ε - 1) )
        其中 A = (1/y_0 - 1)
    """
    if y0 <= 0.0 or y0 >= 1.0:
        return np.inf
    a = (1.0 - y0) / y0
    target = a * np.exp(a) * (1.0 / epsilon - 1.0)
    if target < -1.0 / np.e:
        return np.inf
    w_val = lambertw(target, k=0)
    tau_sat = a - np.real(w_val)
    return float(tau_sat)


def atmospheric_energy_model(intensity, tau, delta=0.001):
    """
    大气极端事件能量级串模型。

    模拟极端天气系统从初始扰动 δ 增长到平衡强度 1 的过程。
    该动力学对应于：
        - 早期阶段（y << 1）：dy/dt ≈ y^2，超指数增长
        - 晚期阶段（y → 1）：dy/dt → 0，渐近饱和

    Parameters
    ----------
    intensity : float
        当前能量强度（归一化）。
    tau : float
        归一化时间。
    delta : float
        初始扰动幅度。

    Returns
    -------
    energy : float
        t = tau 时刻的能量强度。
    """
    if intensity <= 0:
        intensity = delta
    t_exact = np.array([tau])
    y_exact = energy_cascade_exact(t_exact, y0=intensity)
    return float(y_exact[0])


def test_energy_cascade():
    t = np.linspace(0, 2, 100)
    y0 = 0.01
    y_exact = energy_cascade_exact(t, y0)
    t_num, y_num = solve_energy_cascade_rk4((0, 2), y0, n_steps=500)
    # 插值到同一时间点比较
    y_interp = np.interp(t, t_num, y_num)
    max_err = np.max(np.abs(y_exact - y_interp))
    assert max_err < 0.05  # RK4 在 stiff 问题上精度有限

    tau_sat = energy_saturation_time(y0)
    assert tau_sat > 0
    print("energy_cascade_ode 自测试通过")


if __name__ == "__main__":
    test_energy_cascade()
