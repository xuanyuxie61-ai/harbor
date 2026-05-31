"""
环境响应插值模块：基于 test_interp_1d，
提供温度、VPD、土壤水分等环境因子对光合速率响应曲线的插值。

核心公式：
  温度响应曲线（非对称高斯型）：
      f_T(T) = exp( -((T - T_opt) / T_sigma)^2 ) * (1 + alpha * (T - T_opt))

  VPD 响应曲线（指数型）：
      f_D(D) = exp(-beta * D)

  土壤水分响应曲线（Sigmoid 型）：
      f_theta(theta) = 1 / (1 + exp(-k * (theta - theta_50)))
"""
import numpy as np


def temperature_response(t_celsius, t_opt=25.0, t_sigma=12.0, alpha_asym=0.02):
    """
    非对称高斯温度响应函数。
    返回: [0,1] 范围内的响应系数
    """
    dt = t_celsius - t_opt
    f = np.exp(-(dt / t_sigma) ** 2) * max(1.0 + alpha_asym * dt, 0.1)
    return float(np.clip(f, 0.0, 1.0))


def vpd_response(vpd, beta=0.05):
    """
    VPD 响应函数 (kPa)。
    """
    return float(np.exp(-beta * max(vpd, 0.0)))


def soil_moisture_response(theta, theta_50=0.25, k=20.0):
    """
    土壤水分响应函数 (m^3/m^3)。
    """
    return float(1.0 / (1.0 + np.exp(-k * (theta - theta_50))))


def piecewise_linear_interp(x_vals, y_vals, xq):
    """
    分段线性插值（模拟 p00_f 系列的 1D 插值思想）。
    x_vals: 已知节点（已排序）
    y_vals: 已知函数值
    xq: 查询点（标量或数组）
    """
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    xq = np.asarray(xq, dtype=float)
    scalar = xq.ndim == 0
    xq = np.atleast_1d(xq)
    result = np.zeros_like(xq, dtype=float)
    for i in range(len(xq)):
        xi = xq[i]
        if xi <= x[0]:
            result[i] = y[0]
        elif xi >= x[-1]:
            result[i] = y[-1]
        else:
            idx = np.searchsorted(x, xi) - 1
            idx = max(0, min(idx, len(x) - 2))
            dx = x[idx + 1] - x[idx]
            if abs(dx) < 1e-14:
                result[i] = y[idx]
            else:
                t = (xi - x[idx]) / dx
                result[i] = y[idx] * (1.0 - t) + y[idx + 1] * t
    return float(result[0]) if scalar else result


def build_response_tables():
    """
    构建标准环境响应查找表。
    返回: dict of (x_vals, y_vals)
    """
    t_range = np.array([-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0])
    t_resp = np.array([0.0, 0.05, 0.15, 0.35, 0.60, 0.85, 1.0, 0.90, 0.65, 0.30, 0.05])

    vpd_range = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
    vpd_resp = np.array([1.0, 0.95, 0.85, 0.70, 0.55, 0.40, 0.25, 0.10, 0.05])

    sm_range = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50])
    sm_resp = np.array([0.05, 0.10, 0.20, 0.40, 0.60, 0.75, 0.85, 0.92, 1.0])

    return {
        'temperature': (t_range, t_resp),
        'vpd': (vpd_range, vpd_resp),
        'soil_moisture': (sm_range, sm_resp)
    }


def compute_environmental_factor(t_c, vpd, theta, tables):
    """
    综合计算环境限制因子（三个响应的乘积）。
    """
    ft = piecewise_linear_interp(tables['temperature'][0], tables['temperature'][1], t_c)
    fd = piecewise_linear_interp(tables['vpd'][0], tables['vpd'][1], vpd)
    fsm = piecewise_linear_interp(tables['soil_moisture'][0], tables['soil_moisture'][1], theta)
    return ft * fd * fsm
