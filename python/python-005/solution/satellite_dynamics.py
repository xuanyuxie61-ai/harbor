# -*- coding: utf-8 -*-
"""
satellite_dynamics.py
CMB 卫星扫描策略刚体姿态动力学

核心物理：
    卫星绕地球-日 L2 点的刚体姿态由欧拉角 (ψ, θ, φ) 描述。
    运动方程（欧拉-拉格朗日）：
        dψ/dt = (ω₁ sinφ + ω₂ cosφ) / sinθ
        dθ/dt = ω₁ cosφ − ω₂ sinφ
        dφ/dt = ω₃ − cosθ · dψ/dt
        dω₁/dt = [(A₂−A₃) ω₂ ω₃ + M₁] / A₁
        dω₂/dt = [(A₃−A₁) ω₃ ω₁ + M₂] / A₂
        dω₃/dt = [(A₁−A₂) ω₁ ω₂ + M₃] / A₃
    其中 A_i 为主转动惯量，M_i 为外力矩（引力梯度 + 控制力矩）。

    本模块还包含受摄动 n-body 轨道演化（简化版 L2 点轨道），
    融合种子项目 495_gyroscope_ode（陀螺姿态 ODE）与
    345_exm/orbits（N-body 轨道积分）。
"""

import numpy as np
from typing import Tuple
from utils import robust_divide, clip_to_unit


# ---------------------------------------------------------------------------
# 刚体陀螺姿态 ODE（来自 gyroscope_ode）
# ---------------------------------------------------------------------------
class GyroscopeDynamics:
    """阻尼陀螺的欧拉角 + 角速度演化。"""

    def __init__(self, A1: float = 1.0, A2: float = 1.0, A3: float = 0.5,
                 m: float = 1.0):
        """
        Parameters
        ----------
        A1, A2, A3 : float
            主转动惯量。
        m : float
            等效外力矩幅值（重力梯度近似）。
        """
        self.A1 = A1
        self.A2 = A2
        self.A3 = A3
        self.m = m

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        返回 dy/dt，y = [ψ, θ, φ, ω1, ω2, ω3]。
        包含奇点保护（sinθ ≈ 0 时正则化）。
        """
        psi, theta, phi, w1, w2, w3 = y
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        sin_p = np.sin(phi)
        cos_p = np.cos(phi)
        # 防止 sinθ 过小的正则化
        sin_t_reg = sin_t if abs(sin_t) > 1e-8 else np.copysign(1e-8, sin_t)

        # 欧拉角运动学
        dpsi = (w1 * sin_p + w2 * cos_p) / sin_t_reg
        dtheta = w1 * cos_p - w2 * sin_p
        dphi = w3 - cos_t * dpsi

        # 外力矩（简化重力梯度）
        M1 = -self.m * self.A1 * sin_t * cos_p
        M2 = self.m * self.A2 * sin_t * sin_p
        M3 = 0.0

        # 欧拉动力学方程
        dw1 = ((self.A2 - self.A3) * w2 * w3 + M1) / self.A1
        dw2 = ((self.A3 - self.A1) * w3 * w1 + M2) / self.A2
        dw3 = ((self.A1 - self.A2) * w1 * w2 + M3) / self.A3

        return np.array([dpsi, dtheta, dphi, dw1, dw2, dw3])

    def rk4_step(self, t: float, y: np.ndarray, dt: float) -> np.ndarray:
        """经典四阶 Runge-Kutta 单步。"""
        k1 = self.rhs(t, y)
        k2 = self.rhs(t + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = self.rhs(t + 0.5 * dt, y + 0.5 * dt * k2)
        k4 = self.rhs(t + dt, y + dt * k3)
        return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def integrate(self, y0: np.ndarray, t_span: Tuple[float, float],
                  n_steps: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        从 t_span[0] 到 t_span[1] 积分，返回 (t_array, y_array)。
        """
        t0, t1 = t_span
        dt = (t1 - t0) / n_steps
        t_arr = np.linspace(t0, t1, n_steps + 1)
        y_arr = np.zeros((n_steps + 1, len(y0)))
        y_arr[0] = y0
        for i in range(n_steps):
            y_arr[i + 1] = self.rk4_step(t_arr[i], y_arr[i], dt)
        return t_arr, y_arr


# ---------------------------------------------------------------------------
# 卫星扫描策略生成
# ---------------------------------------------------------------------------
def generate_scanning_trajectory(n_steps: int = 2000,
                                  spin_period_min: float = 60.0,
                                  precession_period_min: float = 192.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成 Planck-like 扫描轨迹。
    卫星自转轴以固定倾角 θ 绕日-卫星连线进动，同时卫星绕自转轴高速旋转。
    返回单位球面上的指向 (theta_point, phi_point) 数组。
    """
    t = np.linspace(0.0, 1.0, n_steps)  # 归一化时间
    omega_spin = 2.0 * np.pi * n_steps / spin_period_min  # 归一化角频率
    omega_prec = 2.0 * np.pi * n_steps / precession_period_min

    theta_ax = np.radians(45.0)  # 自转轴与黄道面法线夹角
    phi_ax = omega_prec * t       # 进动角

    # 卫星绕自转轴的旋转角
    phi_spin = omega_spin * t

    # 卫星 boresight 方向（在卫星本体坐标系中为 z 轴，与自转轴成固定角）
    boresight_tilt = np.radians(85.0)

    # 球坐标变换到笛卡尔，再旋转
    theta_p = np.zeros(n_steps)
    phi_p = np.zeros(n_steps)

    for i in range(n_steps):
        # 本体坐标系中的 boresight
        bx = np.sin(boresight_tilt) * np.cos(phi_spin[i])
        by = np.sin(boresight_tilt) * np.sin(phi_spin[i])
        bz = np.cos(boresight_tilt)
        # 绕 y 轴旋转 theta_ax，再绕 z 轴旋转 phi_ax
        ct = np.cos(theta_ax)
        st = np.sin(theta_ax)
        cp = np.cos(phi_ax[i])
        sp = np.sin(phi_ax[i])
        # Rz(phi_ax) @ Ry(theta_ax) @ [bx, by, bz]
        x = cp * (ct * bx + st * bz) - sp * by
        y = sp * (ct * bx + st * bz) + cp * by
        z = -st * bx + ct * bz
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        if r < 1e-12:
            theta_p[i] = 0.0
            phi_p[i] = 0.0
        else:
            theta_p[i] = np.arccos(clip_to_unit(z / r))
            phi_p[i] = np.arctan2(y, x)
            if phi_p[i] < 0:
                phi_p[i] += 2.0 * np.pi

    return t, theta_p, phi_p


# ---------------------------------------------------------------------------
# 覆盖均匀性评估
# ---------------------------------------------------------------------------
def compute_hit_map(theta: np.ndarray, phi: np.ndarray,
                    n_theta: int = 36, n_phi: int = 72) -> np.ndarray:
    """
    将扫描轨迹离散化为球面像素命中计数图（简易等经纬网格）。
    """
    hits = np.zeros((n_theta, n_phi), dtype=int)
    dtheta = np.pi / n_theta
    dphi = 2.0 * np.pi / n_phi
    for th, ph in zip(theta, phi):
        it = min(int(th / dtheta), n_theta - 1)
        ip = min(int(ph / dphi), n_phi - 1)
        hits[it, ip] += 1
    return hits


def compute_coverage_uniformity(hits: np.ndarray) -> float:
    """
    计算扫描覆盖均匀性：
        U = 1 - σ_hits / mean_hits
    U → 1 表示完全均匀，U → 0 表示不均匀。
    """
    mean_h = np.mean(hits)
    if mean_h < 1e-12:
        return 0.0
    std_h = np.std(hits)
    return max(0.0, 1.0 - std_h / mean_h)
