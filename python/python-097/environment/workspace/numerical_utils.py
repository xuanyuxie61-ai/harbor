"""
numerical_utils.py

数值工具与辅助算法模块。
融合is_prime的素数判定、sphere_distance的球面蒙特卡洛采样、
以及rigid_body_ode的守恒量监测思想，
为电磁仿真提供底层数值支持。

核心内容:
---------
1. 素数生成与判定: 用于选择避免数值共振的网格参数
2. 球面随机采样: 用于天线辐射方向图的蒙特卡洛验证
3. 能量守恒监测: 基于刚体ODE的守恒量思想
4. 数值误差估计: L2范数、RMS误差
"""

import numpy as np


# ============================================================
# 素数工具（基于is_prime项目）
# ============================================================

def is_prime(n):
    """
    判定整数n是否为素数。
    基于is_prime1.m的核心算法，优化为大数场景。

    Parameters
    ----------
    n : int
        待判定整数

    Returns
    -------
    bool
    """
    if not isinstance(n, int):
        n = int(n)
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    # 只需检查到sqrt(n)
    limit = int(np.sqrt(n)) + 1
    for i in range(3, limit, 2):
        if n % i == 0:
            return False
    return True


def next_prime(n):
    """找到大于n的最小素数。"""
    candidate = n + 1
    while True:
        if is_prime(candidate):
            return candidate
        candidate += 1


def generate_prime_grid_steps(nx_target, ny_target, nz_target):
    """
    生成基于素数的网格步数，以避免周期性数值共振。

    在FDTD中，如果网格步数与波长比为简单有理数，
    可能导致数值色散误差积累。使用素数步数可破坏这种周期性。

    Parameters
    ----------
    nx_target, ny_target, nz_target : int
        目标网格数

    Returns
    -------
    nx, ny, nz : int
        最接近目标的素数网格数
    """
    nx = next_prime(nx_target - 1)
    ny = next_prime(ny_target - 1)
    nz = next_prime(nz_target - 1)
    return nx, ny, nz


# ============================================================
# 球面采样与距离统计（基于sphere_distance项目）
# ============================================================

def sphere_unit_sample(n_samples=1):
    """
    在单位球面上均匀采样。
    基于sphere_unit_sample.m的核心算法（高斯分布归一化法）。

    Parameters
    ----------
    n_samples : int
        采样点数

    Returns
    -------
    samples : ndarray, shape (n_samples, 3)
    """
    samples = np.random.randn(n_samples, 3)
    norms = np.linalg.norm(samples, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1e-15, norms)
    samples = samples / norms
    return samples


def sphere_distance_stats(n_samples=10000):
    """
    统计球面上随机点对之间的距离分布。
    基于sphere_distance_stats.m的核心算法。

    理论值: 平均距离 = 4/3 ≈ 1.3333
           方差 = 64/(45π) - 16/9 + ...

    Parameters
    ----------
    n_samples : int
        采样点对数

    Returns
    -------
    dict
        {'mean', 'variance', 'distances'}
    """
    p = sphere_unit_sample(n_samples)
    q = sphere_unit_sample(n_samples)
    distances = np.linalg.norm(p - q, axis=1)

    mean_dist = np.mean(distances)
    if n_samples > 1:
        variance = np.sum((distances - mean_dist) ** 2) / (n_samples - 1)
    else:
        variance = 0.0

    return {
        'mean': mean_dist,
        'variance': variance,
        'distances': distances,
    }


def monte_carlo_radiation_pattern(e_field_func, n_samples=5000):
    """
    使用蒙特卡洛方法计算天线辐射方向图。

    Parameters
    ----------
    e_field_func : callable
        给定方向(θ,φ)返回|E|²的函数
    n_samples : int
        方向采样数

    Returns
    -------
    dict
        {'directions', 'power_density', 'total_radiated_power'}
    """
    # 在球面上均匀采样方向
    directions = sphere_unit_sample(n_samples)

    # 转换为球坐标
    theta = np.arccos(np.clip(directions[:, 2], -1.0, 1.0))
    phi = np.arctan2(directions[:, 1], directions[:, 0])

    power = np.array([e_field_func(t, p) for t, p in zip(theta, phi)])

    # 总辐射功率（球面积分）
    d_omega = 4.0 * np.pi / n_samples
    total_power = np.sum(power) * d_omega

    return {
        'theta': theta,
        'phi': phi,
        'power_density': power,
        'total_radiated_power': total_power,
    }


# ============================================================
# 守恒量监测（基于rigid_body_ode项目）
# ============================================================

def check_energy_conservation(energy_history, time_history, power_loss_history, tol=1e-3):
    """
    检验电磁能量守恒。

    能量守恒方程:
    dW_em/dt + P_loss + P_rad = 0

    离散形式:
    (W_{n+1} - W_n)/Δt + P_loss,n ≈ 0

    Parameters
    ----------
    energy_history : list or ndarray
        能量历史
    time_history : list or ndarray
        时间历史
    power_loss_history : list or ndarray
        损耗功率历史
    tol : float
        相对容差

    Returns
    -------
    dict
        {'conserved', 'max_relative_error', 'energy_drift'}
    """
    W = np.asarray(energy_history, dtype=np.float64)
    t = np.asarray(time_history, dtype=np.float64)
    P = np.asarray(power_loss_history, dtype=np.float64)

    # 清理NaN/Inf
    valid = np.isfinite(W) & np.isfinite(t)
    if np.any(valid):
        W = W[valid]
        t = t[valid]
    valid_p = np.isfinite(P)
    if np.any(valid_p):
        P = P[valid_p]

    if len(W) < 2 or len(P) < 1:
        return {'conserved': True, 'max_relative_error': 0.0, 'energy_drift': 0.0}

    # 计算能量变化率
    dt = np.diff(t)
    valid_dt = dt > 1e-30
    if not np.any(valid_dt):
        return {'conserved': True, 'max_relative_error': 0.0, 'energy_drift': 0.0}

    dW_dt = np.diff(W) / dt
    dW_dt = np.where(np.isfinite(dW_dt), dW_dt, 0.0)

    # 取对应时间点的损耗功率
    n_min = min(len(P), len(dW_dt))
    P_avg = 0.5 * (P[:n_min-1] + P[1:n_min]) if len(P) == len(W) else P[:n_min]
    P_avg = np.where(np.isfinite(P_avg), P_avg, 0.0)

    # 守恒检验: dW/dt + P ≈ 0
    residual = dW_dt[:len(P_avg)] + P_avg
    W_avg = 0.5 * (W[:len(P_avg)] + W[1:len(P_avg)+1])
    W_avg = np.where(np.abs(W_avg) < 1e-30, 1e-30, W_avg)

    relative_error = np.abs(residual / W_avg)
    relative_error = np.where(np.isfinite(relative_error), relative_error, 0.0)
    max_error = float(np.max(relative_error)) if len(relative_error) > 0 else 0.0
    energy_drift = float((W[-1] - W[0]) / W[0]) if abs(W[0]) > 1e-30 else 0.0

    return {
        'conserved': max_error < tol,
        'max_relative_error': max_error,
        'energy_drift': energy_drift,
    }


def rigid_body_like_conserved_quantity(field_momenta, inertia_tensor):
    """
    类比刚体ODE的守恒量，计算电磁场的"角动量"型守恒量。

    在腔体电磁场中，可定义伪角动量:
    L_pseudo = Σ r × (εE × μH)

    Parameters
    ----------
    field_momenta : ndarray, shape (n, 3)
        场动量向量
    inertia_tensor : ndarray, shape (3, 3)
        惯量张量（类比腔体几何参数）

    Returns
    -------
    conserved : float
        守恒量 (L · I^{-1} · L)
    """
    # 计算角动量
    L = np.sum(field_momenta, axis=0)

    # 类似刚体的动能: T = ½ L · I^{-1} · L
    try:
        I_inv = np.linalg.inv(inertia_tensor)
        conserved = 0.5 * np.dot(L, I_inv @ L)
    except np.linalg.LinAlgError:
        conserved = np.dot(L, L)

    return conserved


# ============================================================
# 误差估计
# ============================================================

def rms_error(u_numeric, u_exact):
    """
    计算均方根误差。
    基于rms.m的核心算法。

    RMS = sqrt( mean( (u_numeric - u_exact)² ) )
    """
    diff = u_numeric - u_exact
    return np.sqrt(np.mean(diff ** 2))


def relative_l2_error(u_numeric, u_exact, eps=1e-30):
    """相对L2误差。"""
    diff_norm = np.linalg.norm(u_numeric - u_exact)
    exact_norm = np.linalg.norm(u_exact)
    if exact_norm < eps:
        return diff_norm
    return diff_norm / exact_norm


def convergence_rate(errors, resolutions):
    """
    估计数值方法的收敛阶。

    假设 error ∝ h^p，则 p = log(error2/error1) / log(h2/h1)

    Parameters
    ----------
    errors : list
        不同分辨率下的误差
    resolutions : list
        对应的分辨率（如网格数或步长）

    Returns
    -------
    rates : list
        收敛阶估计
    """
    errors = np.asarray(errors)
    resolutions = np.asarray(resolutions)

    rates = []
    for i in range(len(errors) - 1):
        if errors[i] > 1e-30 and errors[i + 1] > 1e-30 and resolutions[i] != resolutions[i + 1]:
            p = np.log(errors[i + 1] / errors[i]) / np.log(resolutions[i + 1] / resolutions[i])
            rates.append(p)

    return rates
