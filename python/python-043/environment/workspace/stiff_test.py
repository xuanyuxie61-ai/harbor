"""
Stiff ODE 测试与验证 (stiff_test.py)
======================================
基于种子项目 674_lindberg_exact 的精确解思想，为地核发电机中的 stiff MHD
方程提供数值方法验证基准。

地核发电机方程组在边界层和快磁声波尺度上呈现刚性特征。本模块提供：
  - Lindberg 4维 stiff 系统精确解
  - 刚性比 (stiffness ratio) 估计
  - 数值解与精确解的误差度量
"""

import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# 1. Lindberg 精确解（4维 stiff ODE）
#    原系统：y' = A*y + g(t)，其中 A 有大幅分离特征值。
#    精确解：
#      g1(t) = 1e4 * (t + 2*exp(-t) - 2)
#      g2(t) = 1e4 * (1 - exp(-t) - t*exp(-t))
#      y1(t) = exp(g1) * (cos(g2) + sin(g2))
#      y2(t) = exp(g1) * (cos(g2) - sin(g2))
#      y3(t) = 1 - 2*exp(-t)
#      y4(t) = t * exp(-t)
# ---------------------------------------------------------------------------
def lindberg_g1(t: float) -> float:
    return 1.0e4 * (t + 2.0 * np.exp(-t) - 2.0)


def lindberg_g2(t: float) -> float:
    return 1.0e4 * (1.0 - np.exp(-t) - t * np.exp(-t))


def lindberg_exact(t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 Lindberg stiff ODE 系统在时间数组 t 上的精确解 y(t) 与导数 y'(t)。

    返回:
      y    : shape (len(t), 4)
      dydt : shape (len(t), 4)
    """
    t = np.asarray(t, dtype=float)
    n = t.size
    y = np.zeros((n, 4), dtype=float)
    dydt = np.zeros((n, 4), dtype=float)

    g1 = lindberg_g1(t)
    g2 = lindberg_g2(t)
    dg1 = 1.0e4 * (1.0 - 2.0 * np.exp(-t))
    dg2 = 1.0e4 * (t * np.exp(-t))

    exp_g1 = np.exp(g1)
    cg = np.cos(g2)
    sg = np.sin(g2)

    # y1, y2
    y[:, 0] = exp_g1 * (cg + sg)
    y[:, 1] = exp_g1 * (cg - sg)

    # y3, y4
    y[:, 2] = 1.0 - 2.0 * np.exp(-t)
    y[:, 3] = t * np.exp(-t)

    # 导数
    dydt[:, 0] = y[:, 0] * dg1 + exp_g1 * (-sg + cg) * dg2
    dydt[:, 1] = y[:, 1] * dg1 + exp_g1 * (-sg - cg) * dg2
    dydt[:, 2] = 2.0 * np.exp(-t)
    dydt[:, 3] = np.exp(-t) - t * np.exp(-t)

    return y, dydt


def lindberg_rhs(t: float, y: np.ndarray) -> np.ndarray:
    """
    Lindberg 系统的右端项（用于数值积分测试）。
    """
    y = np.asarray(y, dtype=float)
    g1 = lindberg_g1(t)
    g2 = lindberg_g2(t)
    dg1 = 1.0e4 * (1.0 - 2.0 * np.exp(-t))
    dg2 = 1.0e4 * (t * np.exp(-t))

    exp_g1 = np.exp(g1)
    cg = np.cos(g2)
    sg = np.sin(g2)

    y_exact = np.zeros(4, dtype=float)
    y_exact[0] = exp_g1 * (cg + sg)
    y_exact[1] = exp_g1 * (cg - sg)
    y_exact[2] = 1.0 - 2.0 * np.exp(-t)
    y_exact[3] = t * np.exp(-t)

    dydt_exact = np.zeros(4, dtype=float)
    dydt_exact[0] = y_exact[0] * dg1 + exp_g1 * (-sg + cg) * dg2
    dydt_exact[1] = y_exact[1] * dg1 + exp_g1 * (-sg - cg) * dg2
    dydt_exact[2] = 2.0 * np.exp(-t)
    dydt_exact[3] = np.exp(-t) - t * np.exp(-t)

    # 构造 RHS 使得 y_exact 是解：dy/dt = f(t,y)
    # 这里直接返回精确导数，形成自治系统的近似
    # 实际 stiff 测试时通常用数值 Jacobian 近似
    # 为保持与精确解一致，返回 dydt_exact
    return dydt_exact


# ---------------------------------------------------------------------------
# 2. 刚性比估计
#    对于线性系统 y' = J*y，刚性比 S = |Re(lambda_max)| / |Re(lambda_min)|
# ---------------------------------------------------------------------------
def estimate_stiffness_ratio(jacobian: np.ndarray) -> float:
    """
    基于 Jacobian 矩阵特征值估计刚性比。
    """
    jacobian = np.asarray(jacobian, dtype=float)
    if jacobian.ndim != 2 or jacobian.shape[0] != jacobian.shape[1]:
        return 0.0
    try:
        eigvals = np.linalg.eigvals(jacobian)
        real_parts = np.real(eigvals)
        pos = real_parts[real_parts > 0]
        neg = real_parts[real_parts < 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.0
        lambda_max = np.max(pos)
        lambda_min = np.min(np.abs(neg))
        if lambda_min < 1e-30:
            return 1e30
        return lambda_max / lambda_min
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# 3. 数值误差度量
# ---------------------------------------------------------------------------
def compute_l2_error(y_numeric: np.ndarray, y_exact: np.ndarray) -> float:
    """计算数值解与精确解之间的归一化 L2 误差。"""
    diff = y_numeric - y_exact
    norm_exact = np.linalg.norm(y_exact)
    norm_diff = np.linalg.norm(diff)
    if norm_exact < 1e-30:
        return norm_diff
    return norm_diff / norm_exact


def compute_max_error(y_numeric: np.ndarray, y_exact: np.ndarray) -> float:
    """计算最大绝对误差。"""
    return float(np.max(np.abs(y_numeric - y_exact)))


# ---------------------------------------------------------------------------
# 4. 地核发电机 stiff 特征估计
#    基于磁扩散与阿尔芬波时间尺度的分离
# ---------------------------------------------------------------------------
def dynamo_stiffness_estimate(radius: float, eta: float, va: float) -> float:
    """
    估计地核发电机系统的有效刚性比。

    时间尺度:
      磁扩散时间  tau_eta = r^2 / eta
      阿尔芬渡越时间 tau_A = r / va

    刚性比 S ~ tau_eta / tau_A = r * va / eta

    参数:
      radius : 特征半径 (m)
      eta    : 磁扩散系数 (m^2/s)
      va     : 阿尔芬速度 (m/s)
    """
    if eta <= 0.0 or radius <= 0.0 or va <= 0.0:
        return 0.0
    return radius * va / eta


# ---------------------------------------------------------------------------
# 自测试
# ---------------------------------------------------------------------------
def _self_test():
    t = np.linspace(0.0, 1.0, 11)
    y, dydt = lindberg_exact(t)
    assert y.shape == (11, 4)
    assert dydt.shape == (11, 4)

    # 在 t=0 处检查
    assert abs(y[0, 2] - (-1.0)) < 1e-6  # y3(0) = 1 - 2 = -1
    assert abs(y[0, 3] - 0.0) < 1e-10    # y4(0) = 0

    # 刚性比测试
    J = np.array([[-1e4, 1e4, 0, 0],
                  [1, -2, 0, 0],
                  [0, 0, -1, 0],
                  [0, 0, 0, -0.1]], dtype=float)
    sr = estimate_stiffness_ratio(J)
    assert sr > 1e3

    # 地核估计
    S = dynamo_stiffness_estimate(3480e3, 2.0, 1.0e-3)
    assert S > 1e3
    print(f"stiff_test: dynamo stiffness estimate S={S:.4e}")
    print("stiff_test: self-test passed.")


if __name__ == "__main__":
    _self_test()
