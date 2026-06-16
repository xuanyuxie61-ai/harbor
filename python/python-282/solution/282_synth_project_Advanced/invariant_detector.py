"""
invariant_detector.py
=====================

SEI 演化无量纲不变量检测与稳态"ringdown"分析模块。

融合种子项目：
    1256_AXVIAM_gw-ringdown-invariant：
        引力波环down不变量 Xi = |dh/dt| / (|d^2h/dt^2| + eps)
        → 类比 SEI 演化的无量纲稳态不变量

核心思想：
    将 SEI 厚度生长速率类比引力波环down信号：
        Ξ(t) = |dL/dt| / (|d^2L/dt^2| + ε)
    在 SEI 达到准稳态时，Ξ 趋于常值 plateau，
    定义无量纲不变量：
        Ξ̃ = τ_char / Ξ_plateau
    其中 τ_char = L_SEI^2 / D_LI_SEI 为特征扩散时间。

作者: DA-Synthesis
"""

import math
try:
    from . import sei_parameters as P
except ImportError:
    import sei_parameters as P


def numerical_derivative(y, dt):
    """
    中心差分数值一阶导数。

    Parameters
    ----------
    y : list[float]
        时间序列。
    dt : float
        时间步长。

    Returns
    -------
    dy : list[float]
        导数序列（边界使用一阶差分）。
    """
    n = len(y)
    dy = [0.0] * n
    if n < 2:
        return dy
    dy[0] = (y[1] - y[0]) / dt
    dy[n - 1] = (y[n - 1] - y[n - 2]) / dt
    for i in range(1, n - 1):
        dy[i] = (y[i + 1] - y[i - 1]) / (2.0 * dt)
    return dy


def numerical_second_derivative(y, dt):
    """
    中心差分二阶导数。

    Parameters
    ----------
    y : list[float]
        时间序列。
    dt : float
        时间步长。

    Returns
    -------
    ddy : list[float]
        二阶导数序列。
    """
    n = len(y)
    ddy = [0.0] * n
    if n < 3:
        return ddy
    for i in range(1, n - 1):
        ddy[i] = (y[i + 1] - 2.0 * y[i] + y[i - 1]) / (dt * dt)
    return ddy


def compute_sei_invariant(thickness_history, dt):
    """
    计算 SEI 演化的无量纲不变量 Ξ(t)（类比引力波 ringdown 不变量）。

    数学定义：
        Ξ(t) = |dL/dt| / (|d^2L/dt^2| + ε)

    其中 ε 为防止除零的小量。

    Parameters
    ----------
    thickness_history : list[float]
        SEI 厚度时间序列 [m]。
    dt : float
        时间步长 [s]。

    Returns
    -------
    xi_t : list[float]
        无量纲不变量时间序列。
    """
    eps = 1.0e-30
    dL_dt = numerical_derivative(thickness_history, dt)
    d2L_dt2 = numerical_second_derivative(thickness_history, dt)

    xi_t = []
    for i in range(len(thickness_history)):
        numerator = abs(dL_dt[i])
        denominator = abs(d2L_dt2[i]) + eps
        xi_t.append(numerator / denominator)

    return xi_t


def detect_plateau(xi_t, window_size=5, tolerance=0.05):
    """
    检测 Ξ(t) 的 plateau 区间（稳态标志）。

    算法：滑动窗口内方差小于阈值则认为达到 plateau。

    Parameters
    ----------
    xi_t : list[float]
        不变量时间序列。
    window_size : int
        滑动窗口大小。
    tolerance : float
        相对方差容限。

    Returns
    -------
    plateau_start : int
        plateau 起始索引（-1 表示未找到）。
    plateau_value : float
        plateau 平均值。
    plateau_length : int
        plateau 长度。
    """
    n = len(xi_t)
    if n < window_size:
        return -1, 0.0, 0

    best_start = -1
    best_length = 0
    best_mean = 0.0

    for start in range(n - window_size + 1):
        window = xi_t[start:start + window_size]
        mean_val = sum(window) / len(window)
        if abs(mean_val) < 1.0e-30:
            continue
        variance = sum((w - mean_val) ** 2 for w in window) / len(window)
        rel_var = math.sqrt(variance) / abs(mean_val)

        if rel_var < tolerance:
            # 延伸查找完整 plateau
            end = start + window_size
            while end < n:
                ext_window = xi_t[start:end + 1]
                ext_mean = sum(ext_window) / len(ext_window)
                if abs(ext_mean) < 1.0e-30:
                    break
                ext_var = sum((w - ext_mean) ** 2 for w in ext_window) / len(ext_window)
                if math.sqrt(ext_var) / abs(ext_mean) > tolerance * 2:
                    break
                end += 1

            length = end - start
            if length > best_length:
                best_length = length
                best_start = start
                best_mean = sum(xi_t[start:end]) / length

    return best_start, best_mean, best_length


def compute_dimensionless_invariant(plateau_value, characteristic_time):
    """
    计算最终无量纲不变量 Ξ̃。

    数学形式：
        Ξ̃ = characteristic_time / Ξ_plateau

    Parameters
    ----------
    plateau_value : float
        Ξ 的 plateau 值 [s]。
    characteristic_time : float
        特征时间 [s]。

    Returns
    -------
    float
        无量纲不变量 Ξ̃。
    """
    if abs(plateau_value) < 1.0e-30:
        return 0.0
    return characteristic_time / plateau_value


def sei_thickness_from_concentration(c_profile, dx, molar_volume, porosity=None):
    """
    从浓度场估算 SEI 等效厚度。

    简化模型：
        L_SEI = integral (1 - c(x)/c_max) dx * porosity_correction

    Parameters
    ----------
    c_profile : list[float]
        浓度分布。
    dx : float
        空间步长。
    molar_volume : float
        SEI 产物摩尔体积 [m^3/mol]。
    porosity : list[float] or None
        局部孔隙率。

    Returns
    -------
    float
        SEI 等效厚度 [m]。
    """
    n = len(c_profile)
    c_max = max(c_profile) if max(c_profile) > 0 else 1.0
    total = 0.0
    for i in range(n):
        phi_i = porosity[i] if porosity is not None else 1.0
        total += (1.0 - c_profile[i] / c_max) * dx * phi_i
    return total


def run_invariant_analysis_demo():
    """
    演示不变量分析（使用合成的 SEI 厚度衰减信号）。

    Returns
    -------
    dict
        分析结果。
    """
    # 合成指数衰减 + 振荡的 SEI 厚度信号（类比 ringdown）
    dt_test = P.DT_EXPLICIT
    n_test = 200
    tau_decay = 20.0 * dt_test
    omega_osc = 2.0 * math.pi / (10.0 * dt_test)

    thickness = []
    for i in range(n_test):
        t = i * dt_test
        # 指数衰减趋向稳态 + 小振荡
        L = P.THICKNESS_SEI_INIT * (1.0 + 0.3 * math.exp(-t / tau_decay)
                                      * math.cos(omega_osc * t))
        thickness.append(L)

    xi_t = compute_sei_invariant(thickness, dt_test)
    p_start, p_value, p_length = detect_plateau(xi_t, window_size=10)

    tau_char = P.L_DOMAIN ** 2 / P.D_LI_SEI
    xi_tilde = compute_dimensionless_invariant(p_value, tau_char) if p_value > 0 else 0.0

    return {
        "thickness_history": thickness,
        "xi_t": xi_t,
        "plateau_start": p_start,
        "plateau_value": p_value,
        "plateau_length": p_length,
        "xi_tilde": xi_tilde,
        "tau_characteristic": tau_char,
    }


if __name__ == "__main__":
    result = run_invariant_analysis_demo()
    print(f"[invariant_detector] SEI 演化不变量分析")
    print(f"  特征时间 τ = {result['tau_characteristic']:.6e} s")
    print(f"  Plateau 起始: {result['plateau_start']}")
    print(f"  Plateau 值 Ξ = {result['plateau_value']:.6e} s")
    print(f"  无量纲不变量 Ξ̃ = {result['xi_tilde']:.6e}")
