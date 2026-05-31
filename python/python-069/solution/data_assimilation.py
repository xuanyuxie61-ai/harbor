"""
数据同化与噪声模块：基于 image_noise 思想，
为涡度相关观测通量数据添加观测噪声，模拟测量误差。

核心公式：
  观测噪声模型：
      F_obs = F_true + epsilon
      epsilon ~ N(0, sigma_F^2)
      sigma_F = alpha * |F_true| + beta

  随机误差（盐椒噪声模拟仪器异常）：
      以概率 p_spike，观测值替换为随机异常值

  数据同化权重（卡尔曼增益简化版）：
      K = P_f / (P_f + R)
      x_a = x_f + K * (y_obs - H * x_f)
"""
import numpy as np


def add_gaussian_noise(data, alpha=0.05, beta=0.5):
    """
    添加异方差高斯噪声。
    data: 真实通量值 (umol/m^2/s)
    alpha: 比例系数
    beta: 基线噪声 (umol/m^2/s)
    """
    data = np.asarray(data, dtype=float)
    sigma = alpha * np.abs(data) + beta
    noise = np.random.normal(0.0, sigma)
    return data + noise


def add_spike_noise(data, level=0.02, magnitude=10.0):
    """
    添加盐椒型异常噪声（模拟仪器故障）。
    level: 异常点比例
    magnitude: 异常幅度
    """
    data = np.asarray(data, dtype=float)
    mask = np.random.rand(*data.shape) < level
    noise = np.random.normal(0.0, magnitude, size=data.shape)
    result = data.copy()
    result[mask] += noise[mask]
    return result


def simple_kalman_update(x_forecast, P_forecast, y_obs, H, R):
    """
    简化卡尔曼更新。
    x_forecast: 预测状态
    P_forecast: 预测误差方差
    y_obs: 观测值
    H: 观测算子（标量）
    R: 观测误差方差
    返回: x_analysis, P_analysis
    """
    x_f = float(x_forecast)
    P_f = float(P_forecast)
    y = float(y_obs)
    h = float(H)
    r = float(R)
    S = h ** 2 * P_f + r
    if abs(S) < 1e-14:
        S = 1e-14
    K = h * P_f / S
    x_a = x_f + K * (y - h * x_f)
    P_a = (1.0 - K * h) * P_f
    return x_a, max(P_a, 0.0)


def ensemble_assimilation(ensemble, observations, obs_variance,
                          inflation_factor=1.02):
    """
    集合卡尔曼滤波（EnKF）简化版。
    ensemble: (n_members,) 集合成员
    observations: (n_obs,) 观测值
    obs_variance: 观测误差方差
    返回: 更新后的集合
    """
    ens = np.asarray(ensemble, dtype=float)
    n = len(ens)
    x_mean = np.mean(ens)
    P = np.var(ens) * inflation_factor
    # 逐个观测更新
    for y in observations:
        for i in range(n):
            ens[i], _ = simple_kalman_update(ens[i], P, y, 1.0, obs_variance)
        x_mean = np.mean(ens)
        P = np.var(ens)
    return ens
