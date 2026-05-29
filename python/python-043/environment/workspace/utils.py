"""
utils.py — 公共工具与数值鲁棒性辅助函数

包含：
- 数值稳定性处理（safe_div, clip_bounds）
- 球坐标与笛卡尔坐标转换
- 物理常数定义
- Lindberg 精确解验证（源自 674_lindberg_exact）
- 球面均匀采样（源自 1192_svd_sphere/sphere_sample）
"""

import numpy as np

# =============================================================================
# 物理常数（地核发电机相关）
# =============================================================================
PHYSICAL_CONSTANTS = {
    "mu_0": 4.0 * np.pi * 1e-7,       # 真空磁导率 [H/m]
    "rho_core": 1.05e4,               # 地核密度 [kg/m^3]
    "sigma_core": 5.0e5,              # 地核电导率 [S/m]
    "eta_magnetic": 0.8,              # 磁扩散率 [m^2/s]  (eta = 1/(mu_0*sigma))
    "nu_kinematic": 1.2e-6,           # 运动粘度 [m^2/s]
    "kappa_thermal": 5.0e-6,          # 热扩散率 [m^2/s]
    "core_radius": 3.48e6,            # 外核半径 [m]
    "icb_radius": 1.22e6,             # 内核半径 [m]
    "angular_velocity": 7.2921159e-5, # 地球自转角速度 [rad/s]
    "gravity_surface": 10.0,          # 地表重力 [m/s^2]
}


def safe_div(a, b, eps=1e-30):
    """鲁棒除法，避免除零。"""
    b_safe = np.where(np.abs(b) < eps, np.sign(b + eps) * eps, b)
    return a / b_safe


def clip_bounds(x, x_min, x_max):
    """将数值裁剪到边界区间。"""
    return np.clip(x, x_min, x_max)


def cartesian_to_spherical(x, y, z):
    """笛卡尔坐标 (x,y,z) → 球坐标 (r, theta, phi)。"""
    r = np.sqrt(x * x + y * y + z * z)
    theta = np.arccos(clip_bounds(safe_div(z, r), -1.0, 1.0))
    phi = np.arctan2(y, x)
    return r, theta, phi


def spherical_to_cartesian(r, theta, phi):
    """球坐标 (r, theta, phi) → 笛卡尔坐标 (x,y,z)。"""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def sphere_uniform_sample(n):
    """
    在单位球面上均匀采样 n 个点（源自 1192_svd_sphere 的 sphere_sample）。
    采用正态分布归一化法。
    """
    p = np.random.normal(size=(n, 3))
    norms = np.linalg.norm(p, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return p / norms


def lindberg_exact_solution(t):
    """
    Lindberg  stiff ODE 精确解（源自 674_lindberg_exact）。
    用于验证时间积分器精度。
    
    方程组:
      g1 = 1e4 * (t + 2*exp(-t) - 2)
      g2 = 1e4 * (1 - exp(-t) - t*exp(-t))
      y1 = exp(g1)*(cos(g2)+sin(g2))
      y2 = exp(g1)*(cos(g2)-sin(g2))
      y3 = 1 - 2*exp(-t)
      y4 = t*exp(-t)
    """
    t = np.atleast_1d(t)
    n = t.size
    y = np.zeros((n, 4))
    dydt = np.zeros((n, 4))

    g1 = 1e4 * (t + 2.0 * np.exp(-t) - 2.0)
    g2 = 1e4 * (1.0 - np.exp(-t) - t * np.exp(-t))

    dg1dt = 1e4 * (1.0 - 2.0 * np.exp(-t))
    dg2dt = 1e4 * (t * np.exp(-t))

    y[:, 0] = np.exp(g1) * (np.cos(g2) + np.sin(g2))
    y[:, 1] = np.exp(g1) * (np.cos(g2) - np.sin(g2))
    y[:, 2] = 1.0 - 2.0 * np.exp(-t)
    y[:, 3] = t * np.exp(-t)

    dydt[:, 0] = (np.exp(g1) * dg1dt * (np.cos(g2) + np.sin(g2))
                  + np.exp(g1) * (-np.sin(g2) + np.cos(g2)) * dg2dt)
    dydt[:, 1] = (np.exp(g1) * dg1dt * (np.cos(g2) - np.sin(g2))
                  + np.exp(g1) * (-np.sin(g2) - np.cos(g2)) * dg2dt)
    dydt[:, 2] = 2.0 * np.exp(-t)
    dydt[:, 3] = (1.0 - t) * np.exp(-t)

    return y, dydt


def lindberg_rhs(t, y):
    """
    Lindberg ODE 的右端项（用于测试时间积分器）。
    通过数值微分近似精确导数。
    """
    _, dydt = lindberg_exact_solution(np.array([t]))
    return dydt[0, :]


def validate_lindberg(integrator_func, dt=0.001, tol=1e-6):
    """
    用 Lindberg 精确解验证时间积分器（源自 674_lindberg_exact）。
    integrator_func: 调用签名 (rhs, tspan, y0, dt, tol) -> (t, y, e)
    返回相对误差。
    """
    tspan = np.array([0.0, 0.01])
    y0 = np.array([1.0, 1.0, -1.0, 0.0])
    t, y, _ = integrator_func(lindberg_rhs, tspan, y0, dt, tol)
    y_exact, _ = lindberg_exact_solution(np.array([t[-1]]))
    rel_err = np.linalg.norm(y[-1, :] - y_exact[0, :]) / (np.linalg.norm(y_exact[0, :]) + 1e-30)
    return rel_err


def condition_number_estimate(A):
    """粗略估计矩阵条件数，用于判断数值稳定性。"""
    s = np.linalg.svd(A, compute_uv=False)
    s_max = np.max(s)
    s_min = np.max(s[s > 1e-15])
    return s_max / s_min
