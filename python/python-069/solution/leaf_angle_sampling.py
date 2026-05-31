"""
叶片角度分布采样模块：基于 simplex_monte_carlo 思想，
在叶片倾角-方位角分布的单纯形上进行蒙特卡洛采样，
计算有效投影面积 G 函数。

核心公式：
  叶片角度空间可视为三维概率单纯形：
      S = { (p1,p2,p3) | p_i >= 0, sum(p_i) = 1 }
  其中 p1,p2,p3 分别代表水平叶、倾斜叶、直立叶的比例。

  单位单纯形上均匀采样：
      x = -ln(u1) / sum(-ln(ui))
  然后通过仿射变换映射到一般单纯形。

  有效投影面积 G（Ross 公式）：
      G(theta_s) = integral_0^{2pi} integral_0^{pi/2}
          |cos(theta_l)cos(theta_s) + sin(theta_l)sin(theta_s)cos(phi_l-phi_s)|
          * f(theta_l, phi_l) sin(theta_l) d(theta_l) d(phi_l)
"""
import numpy as np


def simplex_unit_sample(m, n):
    """
    在单位 m 维单纯形上均匀采样 n 个点。
    返回: (m, n)
    """
    e = -np.log(np.random.rand(m + 1, n))
    s = np.sum(e, axis=0, keepdims=True)
    x = e[:m, :] / s
    return x


def simplex_general_sample(m, n, t):
    """
    在一般 m 维单纯形上采样 n 个点。
    t: (m, m+1) 单纯形顶点
    返回: (m, n)
    """
    x1 = simplex_unit_sample(m, n)
    x = t[:, :m] @ x1 + t[:, m:m + 1] @ (1.0 - np.sum(x1, axis=0, keepdims=True))
    return x


def simplex_unit_volume(m):
    """m 维单位单纯形的体积 = 1/m!"""
    vol = 1.0
    for i in range(2, m + 1):
        vol /= float(i)
    return vol


def leaf_angle_monte_carlo(n_samples, theta_s, phi_s=0.0,
                           theta_l_mean=np.pi / 4, sigma_theta=np.pi / 6):
    """
    用蒙特卡洛在叶片角度分布的单纯形上估算 G(theta_s)。
    n_samples: 采样数
    theta_s: 太阳天顶角
    phi_s: 太阳方位角
    theta_l_mean: 叶片倾角均值
    sigma_theta: 叶片倾角标准差
    返回: G 值估计
    """
    # 叶片倾角 theta_l 和方位角 phi_l 的分布
    theta_l = np.random.normal(theta_l_mean, sigma_theta, n_samples)
    theta_l = np.clip(theta_l, 0.01, np.pi / 2 - 0.01)
    phi_l = np.random.uniform(0.0, 2.0 * np.pi, n_samples)

    # 计算 cos(xi)，xi 为叶片法线与太阳光线夹角
    cos_xi = (np.cos(theta_l) * np.cos(theta_s)
              + np.sin(theta_l) * np.sin(theta_s) * np.cos(phi_l - phi_s))

    # 概率密度 f(theta_l, phi_l) 的归一化：
    # 假设 phi_l 均匀，theta_l 正态截断
    f_phi = 1.0 / (2.0 * np.pi)
    # 简化的 theta_l 密度（忽略截断的精确归一化，用拒绝采样近似）
    f_theta = np.exp(-0.5 * ((theta_l - theta_l_mean) / sigma_theta) ** 2)
    f_theta /= (sigma_theta * np.sqrt(2.0 * np.pi))
    # 雅可比 sin(theta_l)
    jacobian = np.sin(theta_l)

    integrand = np.abs(cos_xi) * f_theta * f_phi * jacobian
    # 积分域：theta_l in [0, pi/2], phi_l in [0, 2pi]
    volume = (np.pi / 2.0) * (2.0 * np.pi)
    g_estimate = volume * np.mean(integrand)
    return g_estimate


def g_function_table(theta_s_range, n_samples=20000,
                     theta_l_mean=np.pi / 4, sigma_theta=np.pi / 6):
    """
    预计算不同太阳天顶角下的 G 函数表。
    返回: (theta_s_values, g_values)
    """
    theta_s_vals = np.asarray(theta_s_range, dtype=float)
    g_vals = np.array([leaf_angle_monte_carlo(n_samples, ts,
                                               theta_l_mean=theta_l_mean,
                                               sigma_theta=sigma_theta)
                       for ts in theta_s_vals])
    return theta_s_vals, g_vals
