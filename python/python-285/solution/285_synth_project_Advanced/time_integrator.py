"""
时间积分器模块
===============
对应种子项目: 434_fisher_pde_ftcs (FTCS 显式格式 → LGD 时间推进)

物理背景:
    极化和磁化的时间演化方程:
        ∂P/∂t = -L_P · δF/δP + ξ_P      (LGD 方程, 过阻尼动力学)
        ∂M/∂t = -γ(M × H_eff) - αγ(M × (M × H_eff))/|M| + ξ_M  (LLG 方程)

    时间积分格式:
    1. FTCS (Forward Time Centered Space):
        P^{n+1} = P^n + dt · RHS(P^n)
        简单但条件稳定: dt ≤ h²/(4LG)

    2. 半隐式 (Crank-Nicolson / θ-方法):
        P^{n+1} - θ·dt·L·G·∇²P^{n+1} = P^n + (1-θ)·dt·L·G·∇²P^n + ...
        θ=0.5: Crank-Nicolson (二阶, 无耗散)
        θ=1: 全隐式 (一阶, 强耗散, 无条件稳定)

    3. RK4 (四阶 Runge-Kutta):
        k1 = f(t_n, y_n)
        k2 = f(t_n + h/2, y_n + h·k1/2)
        k3 = f(t_n + h/2, y_n + h·k2/2)
        k4 = f(t_n + h, y_n + h·k3)
        y_{n+1} = y_n + (h/6)(k1 + 2k2 + 2k3 + k4)

LLG 方程特殊处理:
    - 需保持 |M| = const (磁化守恒)
    - 使用 Heun 方法或隐式中点法保持约束

核心公式:
    CFL 条件 (FTCS, 扩散型):
        dt ≤ h² / (2·d·D)
    其中 d 为空间维数, D 为扩散系数.

    von Neumann 稳定性:
        增长因子 g(k) = 1 - 4·D·dt/h²·sin²(kh/2)
        |g| ≤ 1 → dt ≤ h²/(2D)
"""

import numpy as np
from high_order_fd import compact_laplacian_2d
from banded_solver import build_lgd_banded_matrix_1d
from scipy.sparse import linalg as splinalg


class TimeIntegrator:
    """
    LGD-LLG 耦合系统的时间积分器.

    支持多种格式: FTCS, 半隐式, RK4.
    """

    def __init__(self, config, lgd_params, free_energy, connector):
        """
        参数:
            config: SimulationConfig 实例
            lgd_params: BiFeO3LGD 实例
            free_energy: LGDFreeEnergyFunctional 实例
            connector: MagnetoelectricConnector 实例
        """
        self.config = config
        self.params = lgd_params
        self.free_energy = free_energy
        self.connector = connector

    # ============================================================
    # LGD 方程右端项
    # ============================================================

    def rhs_P(self, P, M, T, noise_P=None):
        """
        LGD 方程的右端项.

        ∂P/∂t = L_P · H_eff^P + noise

        参数:
            P: 极化场, shape (nx, ny, 3)
            M: 磁化场, shape (nx, ny, 3)
            T: 温度
            noise_P: 热噪声 (可选)

        返回:
            dPdt: 极化变化率, shape (nx, ny, 3)
        """
        H_eff = self.free_energy.effective_field_P(
            P, M, T, self.config.dx, self.config.dy, self.config.fd_order
        )

        # 连接器修正
        delta_P = self.connector.connector_M_to_P(P, M)

        dPdt = self.params.L_P * (H_eff + delta_P)

        if noise_P is not None:
            dPdt += noise_P

        return dPdt

    # ============================================================
    # LLG 方程右端项
    # ============================================================

    def rhs_M(self, P, M, T, noise_M=None):
        """
        LLG 方程的右端项.

        ∂M/∂t = -γ(M × H_eff) - αγ(M × (M × H_eff))/|M|

        使用 LLG 的 Landau-Lifshitz 形式 (等价但更易数值处理):
            ∂M/∂t = -γ'/(1+α²) · (M × H_eff + α(M × (M × H_eff))/|M|)

        参数:
            P: 极化场
            M: 磁化场
            T: 温度
            noise_M: 热噪声

        返回:
            dMdt: 磁化变化率
        """
        H_eff = self.free_energy.effective_field_M(P, M, T)

        # 连接器修正
        delta_M = self.connector.connector_P_to_M(P, M)
        H_total = H_eff + delta_M

        alpha = self.params.alpha_Gilbert
        gamma = 1.759e11  # rad/(s·T), 电子旋磁比

        # 有效旋磁比
        gamma_prime = gamma / (1.0 + alpha ** 2)

        M_norm = np.linalg.norm(M, axis=2, keepdims=True)
        M_norm = np.maximum(M_norm, 1e-30)

        # M × H_eff
        M_cross_H = np.cross(M, H_total)

        # M × (M × H_eff) / |M|
        M_cross_MH = np.cross(M, M_cross_H) / M_norm

        dMdt = -gamma_prime * (M_cross_H + alpha * M_cross_MH)

        if noise_M is not None:
            dMdt += noise_M

        return dMdt

    # ============================================================
    # FTCS 格式 (对应种子项目 434)
    # ============================================================

    def step_ftcs(self, P, M, T, noise_P=None, noise_M=None):
        """
        FTCS (Forward Time Centered Space) 一步.

        P^{n+1} = P^n + dt · rhs_P(P^n, M^n)
        M^{n+1} = M^n + dt · rhs_M(P^n, M^n)

        对应种子项目 434 的 FTCS 格式.

        稳定性: dt ≤ dx²/(4·L_P·G_eff)

        参数:
            P, M: 当前场
            T: 温度
            noise_P, noise_M: 噪声

        返回:
            P_new, M_new: 更新后的场
        """
        dt = self.config.dt

        dPdt = self.rhs_P(P, M, T, noise_P)
        dMdt = self.rhs_M(P, M, T, noise_M)

        P_new = P + dt * dPdt
        M_new = M + dt * dMdt

        # 磁化守恒约束: 投影回球面
        M_new = self._project_M_constraint(M_new)

        return P_new, M_new

    # ============================================================
    # 半隐式格式
    # ============================================================

    def step_semi_implicit(self, P, M, T, theta=0.5,
                           noise_P=None, noise_M=None):
        """
        半隐式 (θ-方法) 一步.

        将梯度能隐式处理, 非线性部分显式处理:

        (I - θ·dt·L·G·∇²)P^{n+1} = P^n + dt·(非线性项 + 噪声)

        通过 ADI 分解为两个 1D 问题.

        参数:
            P, M: 当前场
            T: 温度
            theta: 隐式参数 (0.5=CN, 1=全隐式)
            noise_P, noise_M: 噪声

        返回:
            P_new, M_new
        """
        dt = self.config.dt
        dx, dy = self.config.dx, self.config.dy
        G = self.params.G11
        L = self.params.L_P
        nx, ny = P.shape[:2]

        # 显式部分: 非线性 + 磁电耦合 + 噪声
        H_eff = self.free_energy.effective_field_P(
            P, M, T, dx, dy, self.config.fd_order
        )
        delta_P = self.connector.connector_M_to_P(P, M)

        rhs_explicit = L * (H_eff + delta_P)
        if noise_P is not None:
            rhs_explicit += noise_P

        # 隐式部分: 逐分量解 2D ADI
        P_new = np.zeros_like(P)
        for comp in range(3):
            P_new[..., comp] = self._adi_step_2d(
                P[..., comp], rhs_explicit[..., comp],
                G, L, dt, dx, dy, theta, self.config.fd_order
            )

        # 磁化: 显式 (LLG 非线性强, 不适合隐式)
        dMdt = self.rhs_M(P, M, T, noise_M)
        M_new = M + dt * dMdt
        M_new = self._project_M_constraint(M_new)

        return P_new, M_new

    def _adi_step_2d(self, field, rhs, G, L, dt, dx, dy, theta, fd_order):
        """
        2D ADI 单步.

        Peaceman-Rachford:
            (I - ½θ·dt·L·G·D_x²) u* = (I + ½(1-θ)·dt·L·G·D_y²) u^n
                                       + dt·rhs
            (I - ½θ·dt·L·G·D_y²) u^{n+1} = u*
        """
        nx, ny = field.shape

        # 构建 1D 带状矩阵
        A_x = build_lgd_banded_matrix_1d(
            nx, dx, G, 0.5 * theta * dt * L, 1.0, fd_order
        )
        A_y = build_lgd_banded_matrix_1d(
            ny, dy, G, 0.5 * theta * dt * L, 1.0, fd_order
        )

        # x 方向
        rhs_x = field + dt * rhs
        u_star = np.zeros_like(field)
        for j in range(ny):
            u_star[:, j] = splinalg.spsolve(A_x, rhs_x[:, j])

        # y 方向
        u_new = np.zeros_like(field)
        for i in range(nx):
            u_new[i, :] = splinalg.spsolve(A_y, u_star[i, :])

        return u_new

    # ============================================================
    # RK4 格式
    # ============================================================

    def step_rk4(self, P, M, T, noise_P=None, noise_M=None):
        """
        四阶 Runge-Kutta 一步.

        k1 = f(t_n, y_n)
        k2 = f(t_n + dt/2, y_n + dt·k1/2)
        k3 = f(t_n + dt/2, y_n + dt·k2/2)
        k4 = f(t_n + dt, y_n + dt·k3)
        y_{n+1} = y_n + (dt/6)(k1 + 2k2 + 2k3 + k4)

        注意: RK4 对扩散方程的稳定性约束比 FTCS 更严格.

        参数:
            P, M: 当前场
            T: 温度
            noise_P, noise_M: 噪声 (仅在第一步使用)

        返回:
            P_new, M_new
        """
        dt = self.config.dt

        # k1
        k1_P = self.rhs_P(P, M, T, noise_P)
        k1_M = self.rhs_M(P, M, T, noise_M)

        # k2
        P2 = P + 0.5 * dt * k1_P
        M2 = self._project_M_constraint(M + 0.5 * dt * k1_M)
        k2_P = self.rhs_P(P2, M2, T)
        k2_M = self.rhs_M(P2, M2, T)

        # k3
        P3 = P + 0.5 * dt * k2_P
        M3 = self._project_M_constraint(M + 0.5 * dt * k2_M)
        k3_P = self.rhs_P(P3, M3, T)
        k3_M = self.rhs_M(P3, M3, T)

        # k4
        P4 = P + dt * k3_P
        M4 = self._project_M_constraint(M + dt * k3_M)
        k4_P = self.rhs_P(P4, M4, T)
        k4_M = self.rhs_M(P4, M4, T)

        # 组合
        P_new = P + (dt / 6.0) * (k1_P + 2 * k2_P + 2 * k3_P + k4_P)
        M_new = M + (dt / 6.0) * (k1_M + 2 * k2_M + 2 * k3_M + k4_M)
        M_new = self._project_M_constraint(M_new)

        return P_new, M_new

    # ============================================================
    # 磁化约束投影
    # ============================================================

    def _project_M_constraint(self, M, M_sat=None):
        """
        将磁化投影到 |M| = M_s 球面.

        LLG 保持 |M| 不变, 但数值误差可能导致漂移.
        每步投影以维持约束.

        参数:
            M: 磁化场
            M_sat: 饱和磁化 (默认使用参数值)

        返回:
            M_proj: 投影后的磁化场
        """
        if M_sat is None:
            M_sat = self.params.M_saturation

        M_norm = np.linalg.norm(M, axis=2, keepdims=True)
        M_norm = np.maximum(M_norm, 1e-30)

        M_proj = M * (M_sat / M_norm)

        return M_proj

    # ============================================================
    # 自适应时间步长
    # ============================================================

    def estimate_optimal_dt(self, P, M, T, safety_factor=0.4):
        """
        估计最优时间步长.

        基于当前场的梯度和有效场强度:
            dt_opt = safety / max(|rhs_P|) * min(P_scale)

        参数:
            P, M: 当前场
            T: 温度
            safety_factor: 安全因子

        返回:
            dt_opt: 推荐时间步长
        """
        dPdt = self.rhs_P(P, M, T)
        max_rate = np.max(np.abs(dPdt))

        if max_rate < 1e-30:
            return 1e-12

        P_scale = np.max(np.abs(P))
        if P_scale < 1e-30:
            P_scale = 0.01

        dt_opt = safety_factor * P_scale / max_rate

        # 限制范围
        dt_opt = max(dt_opt, 1e-16)
        dt_opt = min(dt_opt, 1e-11)

        return dt_opt

    # ============================================================
    # 能量守恒监控
    # ============================================================

    def energy_conservation_error(self, P_old, M_old, P_new, M_new, T):
        """
        计算一步的能量变化 (监控用).

        ΔF = F(P_new, M_new) - F(P_old, M_old)

        对于保守系统 (无噪声, 无阻尼), ΔF = 0.
        对于耗散系统 (LGD + LLG), ΔF ≤ 0.

        参数:
            P_old, M_old: 旧场
            P_new, M_new: 新场
            T: 温度

        返回:
            delta_F: 能量变化
            relative_error: 相对误差
        """
        F_old = self.free_energy.total_free_energy(
            P_old, M_old, T, self.config.dx, self.config.dy,
            self.config.fd_order
        )
        F_new = self.free_energy.total_free_energy(
            P_new, M_new, T, self.config.dx, self.config.dy,
            self.config.fd_order
        )

        delta_F = F_new - F_old
        F_ref = max(abs(F_old), 1e-30)
        relative_error = abs(delta_F) / F_ref

        return delta_F, relative_error
