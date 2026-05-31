"""
机械臂运动学与动力学模块
========================================
基于种子项目:
  - 1283_tough_ode   : 刚性ODE积分器思想，用于多体动力学时间推进
  - 151_cg_ne        : CGNE（共轭梯度法求解正规方程）用于微分逆运动学

核心数学模型:
  1. 改进DH参数法齐次变换:
     ^{i-1}T_i = Rot(z,θ_i)·Trans(z,d_i)·Trans(x,a_i)·Rot(x,α_i)

  2. 空间速度传播（一阶微分运动学）:
     v_i = v_{i-1} + ω_{i-1} × r_i + \dot{q}_i z_i   (平移)
     ω_i = ω_{i-1} + \dot{q}_i z_i                   (旋转)

  3. 几何雅可比矩阵 J(q) ∈ ℝ^{6×n}:
     [ v_E ]   [ J_v ]            [ z_i × (p_E - p_i) ]   (转动关节)
     [ ω_E ] = [ J_ω ] \dot{q},  J_i = [        z_i       ]

  4. 欧拉-拉格朗日动力学:
     τ = M(q)\ddot{q} + C(q,\dot{q})\dot{q} + g(q)
     其中 M(q) = Σ [ m_i J_{v_i}^T J_{v_i} + J_{ω_i}^T I_i J_{ω_i} ]

  5. CGNE 求解 J Δq = Δx（不直接计算 J^T J）:
     r_0 = b - A x_0
     z_0 = A^T r_0
     d_0 = z_0
     α_k = (z_k^T z_k) / ((A d_k)^T (A d_k))
     x_{k+1} = x_k + α_k d_k
     r_{k+1} = r_k - α_k A d_k
     z_{k+1} = A^T r_{k+1}
     β_k = (z_{k+1}^T z_{k+1}) / (z_k^T z_k)
     d_{k+1} = z_{k+1} + β_k d_k
"""

import numpy as np
from typing import Tuple, Optional

# ---------------------------------------------------------------------------
# DH 参数与正向运动学
# ---------------------------------------------------------------------------

class ManipulatorKinematics:
    """
    7自由度冗余机械臂的正向/微分运动学。
    采用改进DH参数（Modified DH convention）。
    """

    def __init__(self):
        # 7-DOF 冗余机械臂的MDH参数 (a_i, α_i, d_i, θ_i_offset)
        # 基于典型的7轴协作机械臂结构（如KUKA iiwa / Franka Emika Panda）
        self.n_dof = 7
        self.mdh = np.array([
            [0.0,       0.0,            0.333,      0.0      ],  # joint 1
            [0.0,      -np.pi/2,        0.0,        0.0      ],  # joint 2
            [0.0,       np.pi/2,        0.316,      0.0      ],  # joint 3
            [0.0825,    np.pi/2,        0.0,        0.0      ],  # joint 4
            [-0.0825,  -np.pi/2,        0.384,      0.0      ],  # joint 5
            [0.0,       np.pi/2,        0.0,        0.0      ],  # joint 6
            [0.088,     np.pi/2,        0.107,      0.0      ],  # joint 7
        ], dtype=float)
        # 连杆质心位置（相对连杆坐标系，简化模型）
        self.link_mass = np.array([2.5, 2.5, 2.0, 2.0, 1.5, 1.5, 1.2], dtype=float)
        self.link_inertia = [
            np.diag([0.01, 0.01, 0.01]),
            np.diag([0.01, 0.01, 0.01]),
            np.diag([0.008, 0.008, 0.008]),
            np.diag([0.008, 0.008, 0.008]),
            np.diag([0.005, 0.005, 0.005]),
            np.diag([0.005, 0.005, 0.005]),
            np.diag([0.003, 0.003, 0.003]),
        ]

    def _mdh_transform(self, a: float, alpha: float, d: float, theta: float) -> np.ndarray:
        """
        改进DH参数齐次变换矩阵:
        | cosθ  -sinθ   0    a |
        | sinθ·cosα  cosθ·cosα  -sinα  -d·sinα |
        | sinθ·sinα  cosθ·sinα   cosα   d·cosα |
        | 0      0      0    1 |
        """
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        T = np.array([
            [ct,     -st,     0.0,   a],
            [st*ca,  ct*ca,  -sa,  -d*sa],
            [st*sa,  ct*sa,   ca,   d*ca],
            [0.0,    0.0,    0.0,   1.0]
        ], dtype=float)
        return T

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """
        正向运动学：计算末端位姿 T_0^7 ∈ SE(3)。
        同时保存所有连杆变换矩阵用于雅可比计算。
        """
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.size != self.n_dof:
            raise ValueError(f"关节角维度必须为 {self.n_dof}, 得到 {q.size}")
        # 边界限幅
        q = np.clip(q, -np.pi, np.pi)

        T = np.eye(4, dtype=float)
        self._T_list = [T.copy()]
        for i in range(self.n_dof):
            a, alpha, d, theta0 = self.mdh[i]
            theta = theta0 + q[i]
            T_i = self._mdh_transform(a, alpha, d, theta)
            T = T @ T_i
            self._T_list.append(T.copy())
        return T

    def geometric_jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        计算几何雅可比矩阵 J(q) ∈ ℝ^{6×n}。
        前3行为线速度部分 J_v，后3行为角速度部分 J_ω。
        """
        T_ee = self.forward_kinematics(q)
        p_ee = T_ee[:3, 3]
        J = np.zeros((6, self.n_dof), dtype=float)
        z0 = np.array([0.0, 0.0, 1.0])
        for i in range(self.n_dof):
            T_i = self._T_list[i]
            p_i = T_i[:3, 3]
            z_i = T_i[:3, 2]
            J[:3, i] = np.cross(z_i, p_ee - p_i)
            J[3:, i] = z_i
        return J

    def manipulability_measure(self, q: np.ndarray) -> float:
        """
        可操纵性度量（Yoshikawa）:
        w(q) = sqrt(det(J_v J_v^T))
        若机械臂接近奇异位形，该值趋近于0。
        """
        J = self.geometric_jacobian(q)
        Jv = J[:3, :]
        try:
            m = np.sqrt(np.linalg.det(Jv @ Jv.T))
        except np.linalg.LinAlgError:
            m = 0.0
        if not np.isfinite(m):
            m = 0.0
        return float(m)


# ---------------------------------------------------------------------------
# 刚性ODE积分器（基于1283_tough_ode的思想）
# ---------------------------------------------------------------------------

class StiffODEIntegrator:
    r"""
    针对机械臂刚体动力学的半隐式中点法（Singly Diagonally Implicit Runge-Kutta, SDIRK）。
    用于积分形如  \dot{y} = f(t, y)  的 stiff ODE。

    采用2阶SDIRK格式（单对角隐式龙格-库塔）:
      y_{n+1} = y_n + h·b_1·k_1 + h·b_2·k_2
      k_1 = f(t_n + c_1 h, y_n + h·a_{11}·k_1)
      k_2 = f(t_n + c_2 h, y_n + h·a_{21}·k_1 + h·a_{22}·k_2)

    系数（L-稳定2阶SDIRK）:
      γ = 1 - 1/√2 ≈ 0.2929
      A = [[γ, 0], [1-γ, γ]]
      b = [1-γ, γ]
      c = [γ, 1]
    """

    def __init__(self, tol: float = 1e-6, max_iter: int = 20):
        self.tol = tol
        self.max_iter = max_iter
        self.gamma = 1.0 - 1.0 / np.sqrt(2.0)

    def _newton_solve(self, f, t, y, h, gamma, k_guess):
        """Newton迭代求解隐式阶段方程 k = f(t, y + h·γ·k)。"""
        k = k_guess.copy()
        for _ in range(self.max_iter):
            y_stage = y + h * gamma * k
            f_val = f(t, y_stage)
            # 近似Jacobian为有限差分（标量情形推广到向量）
            eps = np.sqrt(np.finfo(float).eps) * (np.linalg.norm(y_stage) + 1.0)
            if eps < 1e-14:
                eps = 1e-14
            # 使用对角近似: (I - hγJ)^{-1} ≈ I / (1 - hγ·diag(J))
            # 这里简化为阻尼Newton修正
            residual = k - f_val
            if np.linalg.norm(residual) < self.tol:
                break
            # 简单标量阻尼
            damp = 1.0 / (1.0 + h * gamma * 0.1)
            k = k - damp * residual
            # 边界截断
            k = np.clip(k, -1e6, 1e6)
        return k

    def integrate(self, f, t_span: Tuple[float, float], y0: np.ndarray,
                  h0: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        从 t0 到 tf 积分 ODE，使用自适应步长。
        返回 (t_array, y_array)。
        """
        t0, tf = t_span
        y0 = np.asarray(y0, dtype=float).reshape(-1)
        if h0 is None:
            h0 = (tf - t0) / 100.0
        h = max(min(h0, tf - t0), 1e-12)
        t = t0
        y = y0.copy()
        t_out = [t]
        y_out = [y.copy()]
        gamma = self.gamma

        while t < tf - 1e-14:
            if t + h > tf:
                h = tf - t
            # 第一阶段
            k1 = self._newton_solve(f, t, y, h, gamma, f(t, y))
            # 第二阶段
            y_stage2 = y + h * (1.0 - gamma) * k1
            k2 = self._newton_solve(f, t + h, y_stage2, h, gamma, f(t + h, y_stage2))
            # 更新
            y_new = y + h * ((1.0 - gamma) * k1 + gamma * k2)
            # 简单误差估计（嵌入式方法）
            y_embed = y + h * 0.5 * (k1 + k2)
            err_est = np.linalg.norm(y_new - y_embed) + 1e-20
            # 步长控制
            fac = 0.9 * (self.tol / err_est) ** 0.5
            fac = max(0.2, min(5.0, fac))
            if err_est > self.tol and h > 1e-12:
                h *= fac
                continue
            t += h
            y = y_new
            t_out.append(t)
            y_out.append(y.copy())
            h *= fac
            h = max(min(h, tf - t), 1e-12)
        return np.array(t_out), np.array(y_out)


# ---------------------------------------------------------------------------
# CGNE 求解器（基于151_cg_ne）
# ---------------------------------------------------------------------------

def cgne_solve(A: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray] = None,
               max_iter: int = 500, tol: float = 1e-10) -> np.ndarray:
    r"""
    共轭梯度法求解正规方程 (CGNE / CGNR)。
    求解线性系统 A x = b，其中 A 可能非对称或矩形。
    隐式求解 A^T A x = A^T b，但不显式构造 A^T A。

    输入:
        A : (m, n) 矩阵
        b : (m,)   右端项
        x0: (n,)   初始猜测
    输出:
        x : (n,)   近似解
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    m, n = A.shape
    if b.size != m:
        raise ValueError(f"b维度{b.size}与A行数{m}不匹配")
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).reshape(-1).copy()
        if x.size != n:
            raise ValueError(f"x0维度{x.size}与A列数{n}不匹配")

    r = b - A @ x
    z = A.T @ r
    d = z.copy()
    ztz_old = float(z @ z)
    if ztz_old < tol * tol:
        return x

    for k in range(max_iter):
        Ad = A @ d
        denom = float(Ad @ Ad)
        if abs(denom) < np.finfo(float).eps:
            break
        alpha = ztz_old / denom
        x = x + alpha * d
        r = r - alpha * Ad
        z = A.T @ r
        ztz_new = float(z @ z)
        if np.sqrt(ztz_new) < tol:
            break
        beta = ztz_new / ztz_old
        d = z + beta * d
        ztz_old = ztz_new
        # 数值溢出保护
        if not np.isfinite(x).all():
            x = np.zeros(n, dtype=float)
            break
    return x


def differential_ik_solver(kin: ManipulatorKinematics, q: np.ndarray,
                           dx_des: np.ndarray, damping: float = 0.01) -> np.ndarray:
    r"""
    微分逆运动学：求解 J(q) \dot{q} ≈ v_des，使用阻尼CGNE。
    通过增广矩阵处理阻尼最小二乘:
        [   J    ]          [ v_des ]
        [ λ·I  ] \dot{q} ≈ [   0   ]
    """
    q = np.asarray(q, dtype=float).reshape(-1)
    dx_des = np.asarray(dx_des, dtype=float).reshape(-1)
    J = kin.geometric_jacobian(q)
    m, n = J.shape
    # 增广
    A_aug = np.vstack([J, damping * np.eye(n)])
    b_aug = np.concatenate([dx_des, np.zeros(n)])
    dq = cgne_solve(A_aug, b_aug, max_iter=300, tol=1e-8)
    # 边界限幅
    max_dq = 2.0  # rad/s
    norm_dq = np.linalg.norm(dq)
    if norm_dq > max_dq:
        dq = dq * (max_dq / norm_dq)
    return dq


# ---------------------------------------------------------------------------
# 动力学模型（简化版，用于 stiff ODE 演示）
# ---------------------------------------------------------------------------

def manipulator_dynamics_ode(kin: ManipulatorKinematics):
    r"""
    返回一个函数 f(t, y)，描述机械臂状态演化:
      y = [q; \dot{q}]   (状态: 14维)
      \dot{q} = \dot{q}
      \ddot{q} = M^{-1}(q) [τ - C(q,\dot{q})\dot{q} - g(q)]
    此处采用简化模型：使用质量矩阵 M 的对角近似 + Coriolis/centrifugal 项。
    """
    n = kin.n_dof
    # 简化：M 为对角矩阵（各关节解耦近似）
    M_diag = np.array([2.5, 2.5, 2.0, 1.8, 1.2, 1.0, 0.8], dtype=float)

    def f(t: float, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.size != 2 * n:
            raise ValueError(f"状态向量维度应为{2*n}")
        q = y[:n]
        dq = y[n:]
        # 限幅
        q = np.clip(q, -np.pi, np.pi)
        dq = np.clip(dq, -5.0, 5.0)
        # 简化重力项
        g_vec = np.array([0.0, -5.0*np.sin(q[1]), -3.0*np.sin(q[2]),
                          -2.0*np.sin(q[3]), -1.5*np.sin(q[4]),
                          -1.0*np.sin(q[5]), -0.8*np.sin(q[6])], dtype=float)
        # 简化Coriolis项（对角阻尼）
        Cdq = 0.1 * dq * np.abs(dq)
        # 控制力矩（PD控制跟踪零速度）
        tau = -2.0 * dq - 0.5 * q
        ddq = (tau - Cdq - g_vec) / M_diag
        ddq = np.clip(ddq, -20.0, 20.0)
        return np.concatenate([dq, ddq])
    return f
