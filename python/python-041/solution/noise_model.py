"""
 noise_model.py
 
 融合种子项目:
   - 839_ornstein_uhlenbeck: Ornstein-Uhlenbeck 随机过程、Euler-Maruyama 方法
 
 科学应用:
   地震数据中的背景噪声通常具有时间相关性和均值回归特性。
   Ornstein-Uhlenbeck (OU) 过程是描述此类噪声的理想数学模型:
     dX(t) = theta * (mu - X(t)) dt + sigma dW(t)
   其中 theta 为回归速率，mu 为长期均值，sigma 为波动强度，W(t) 为标准布朗运动。
   
   在地震勘探中，OU 过程可用于:
   1. 模拟检波器的环境噪声（风噪、微震噪声）
   2. 表征地下介质的随机非均匀性（随机场）
   3. 为全波形反演提供统计上真实的数据协方差结构
"""

import numpy as np


def ornstein_uhlenbeck_euler(theta, mu, sigma, x0, tmax, n, rng=None):
    """
    使用 Euler 方法离散求解 Ornstein-Uhlenbeck SDE。
    
    离散格式:
      X_{j+1} = X_j + dt * theta * (mu - X_j) + sigma * dW_j
    其中 dW_j ~ N(0, dt)。
    
    解析解（用于验证）:
      X(t) = mu + (x0 - mu) * exp(-theta * t) + sigma * integral_0^t exp(-theta*(t-s)) dW(s)
    其稳态分布为 N(mu, sigma^2 / (2*theta))。
    
    Parameters
    ----------
    theta : float
        均值回归速率，theta > 0。
    mu : float
        长期均值。
    sigma : float
        波动强度，sigma >= 0。
    x0 : float
        初始值。
    tmax : float
        终止时间。
    n : int
        时间步数。
    rng : numpy.random.Generator, optional
    
    Returns
    -------
    t : ndarray, shape (n+1,)
        时间网格。
    x : ndarray, shape (n+1,)
        OU 过程路径。
    """
    if theta <= 0:
        raise ValueError("theta must be positive")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if n <= 0:
        raise ValueError("n must be positive")
    if rng is None:
        rng = np.random.default_rng()
    dt = tmax / n
    t = np.linspace(0.0, tmax, n + 1)
    x = np.zeros(n + 1)
    x[0] = x0
    # 布朗运动增量 dW ~ N(0, dt)
    dw = np.sqrt(dt) * rng.standard_normal(n)
    for j in range(n):
        x[j + 1] = x[j] + dt * theta * (mu - x[j]) + sigma * dw[j]
    return t, x


def ornstein_uhlenbeck_euler_maruyama(theta, mu, sigma, x0, tmax, n, r, rng=None):
    """
    使用 Euler-Maruyama 多尺度方法求解 OU SDE。
    
    多尺度离散格式:
      dt_large = tmax / n
      dt_small = dt_large / r
      X_{j+1} = X_j + dt_large * theta * (mu - X_j) + sigma * sum_{l=1}^r dW_{j,l}
    其中 dW_{j,l} ~ N(0, dt_small)。
    
    此方法在地震噪声建模中具有优势：可以用大步长处理确定性漂移项，
    用小步长精确采样布朗运动的高频分量。
    
    Parameters
    ----------
    theta : float
        均值回归速率。
    mu : float
        长期均值。
    sigma : float
        波动强度。
    x0 : float
        初始值。
    tmax : float
        终止时间。
    n : int
        大尺度时间步数。
    r : int
        每个大尺度步中的小尺度步数。
    rng : numpy.random.Generator, optional
    
    Returns
    -------
    t : ndarray, shape (n+1,)
        时间网格。
    x : ndarray, shape (n+1,)
        OU 过程路径。
    """
    if theta <= 0 or sigma < 0 or n <= 0 or r <= 0:
        raise ValueError("Invalid parameters")
    if rng is None:
        rng = np.random.default_rng()
    dt_large = tmax / n
    dt_small = dt_large / r
    t = np.linspace(0.0, tmax, n + 1)
    x = np.zeros(n + 1)
    x[0] = x0
    for j in range(n):
        dw = np.sqrt(dt_small) * rng.standard_normal(r)
        x[j + 1] = x[j] + dt_large * theta * (mu - x[j]) + sigma * np.sum(dw)
    return t, x


def generate_seismic_noise(n_traces, n_samples, dt, theta=5.0, sigma=0.05, rng=None):
    """
    生成模拟地震记录噪声。
    
    每道数据的噪声为独立的 OU 过程，参数 theta 和 sigma 可以随道变化
    以模拟空间变化的噪声环境。
    
    噪声模型:
      n_i(t_j) = OU_i(theta_i, mu=0, sigma_i, x0=0, tmax=n_samples*dt)
    
    Parameters
    ----------
    n_traces : int
        地震道数。
    n_samples : int
        每道采样点数。
    dt : float
        采样间隔（秒）。
    theta : float
        噪声回归速率。
    sigma : float
        噪声强度。
    rng : numpy.random.Generator, optional
    
    Returns
    -------
    noise : ndarray, shape (n_traces, n_samples)
        噪声矩阵。
    """
    if rng is None:
        rng = np.random.default_rng()
    tmax = n_samples * dt
    noise = np.zeros((n_traces, n_samples))
    for i in range(n_traces):
        # 每道略有不同的参数以增加真实性
        th = theta * (1.0 + 0.1 * rng.standard_normal())
        sg = sigma * (1.0 + 0.1 * rng.standard_normal())
        _, x = ornstein_uhlenbeck_euler(th, 0.0, sg, 0.0, tmax, n_samples, rng=rng)
        # x 长度为 n_samples + 1，取前 n_samples 或后 n_samples 个点
        if len(x) > n_samples:
            x = x[:n_samples]
        elif len(x) < n_samples:
            # 填充最后一个值
            x = np.concatenate([x, np.full(n_samples - len(x), x[-1])])
        noise[i, :] = x
    return noise


def generate_random_velocity_perturbation(nx, ny, theta=2.0, sigma=0.1, dx=1.0, rng=None):
    """
    生成二维空间 OU 随机场作为速度模型的随机扰动。
    
    将一维 OU 过程扩展到二维，通过独立 OU 过程的叠加:
      delta_v(x,y) = sum_k c_k * OU_k(theta, 0, sigma)
    
    这种随机场具有指数型协方差结构:
      C(r) = sigma^2 * exp(-theta * |r|)
    
    为了避免数值发散，使用稳态采样：直接从稳态分布 N(0, sigma^2/(2*theta))
    采样并做时间演化较短步数的 OU 过程，使其保持有界。
    
    Parameters
    ----------
    nx, ny : int
        网格尺寸。
    theta : float
        空间相关长度倒数。
    sigma : float
        扰动幅度。
    dx : float
        网格间距。
    rng : numpy.random.Generator, optional
    
    Returns
    -------
    perturbation : ndarray, shape (ny, nx)
        速度扰动场。
    """
    if rng is None:
        rng = np.random.default_rng()
    # 使用稳态分布初始化，然后做少量 OU 步以保持相关性
    std = sigma / np.sqrt(2.0 * theta)
    perturbation = rng.normal(0.0, std, size=(ny, nx))
    # 做几次确定性平滑以模拟空间相关（高斯滤波近似）
    for _ in range(3):
        smoothed = perturbation.copy()
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                smoothed[j, i] = 0.25 * (
                    perturbation[j - 1, i] + perturbation[j + 1, i] +
                    perturbation[j, i - 1] + perturbation[j, i + 1]
                )
        perturbation = 0.7 * perturbation + 0.3 * smoothed
    return perturbation
