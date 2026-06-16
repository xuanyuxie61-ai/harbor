"""
radial_schrodinger.py
===================================================================
径向薛定谔方程高阶有限差分解法模块

映射种子项目:
  - 353_fd1d_advection_ftcs: FTCS 有限差分 → 径向方程的 FTCS 时间推进 (不稳定性演示)
  - 1374_unstable_ode: 不稳定ODE → 库仑-核势场中的不稳定数值行为分析
  - 1069_andrew-cr_discrete_flow_models: 离散流模型 → 波函数离散层传播

核心物理方程:
  径向薛定谔方程 (约化径向波函数 u_l = r * R_l):
    -hbar^2/(2m) * d^2u_l/dr^2 + [V(r) + hbar^2*l*(l+1)/(2m*r^2)] * u_l = E * u_l

  无量纲化形式:
    d^2u_l/dr^2 + [k^2 - 2m/hbar^2 * V(r) - l*(l+1)/r^2] * u_l = 0

  令 K^2(r) = k^2 - 2m/hbar^2 * V(r) - l*(l+1)/r^2
    => d^2u/dr^2 = -K^2(r) * u

有限差分格式:
  二阶中心差分 (Numerov 前驱):
    u_{n+1} = 2*u_n - u_{n-1} + h^2 * f_n
  四阶 Numerov 方法:
    (1 + h^2/12 * k_{n+1}^2) * u_{n+1} = 2*(1 - 5h^2/12 * k_n^2) * u_n
                                        - (1 + h^2/12 * k_{n-1}^2) * u_{n-1}
  FTCS (一阶时间, 中心空间, 教学用不稳定格式):
    u^{n+1}_j = 2*u^n_j - u^{n-1}_j + c^2*dt^2/dx^2 * (u^n_{j+1} - 2*u^n_j + u^n_{j-1})
===================================================================
"""

import numpy as np
from typing import Tuple, Optional, Dict
from optical_potential import OpticalPotential, OpticalPotentialParams


# ---------- 物理常数 ----------
HBAR_C = 197.3269804        # hbar*c [MeV·fm]


class RadialSchrodingerSolver:
    """
    径向薛定谔方程求解器

    支持多种有限差分格式:
    1. Numerov (4阶) — 主算法, 用于高精度相移计算
    2. 二阶中心差分 — 基准对比
    3. FTCS — 用于演示不稳定性 (映射自 353_fd1d_advection_ftcs)

    边界条件:
      原点: u(0) = 0 (正则性条件)
      外边界: u(R_max) 由渐近形式确定
    """

    def __init__(
        self,
        potential: OpticalPotential,
        r_max: float = 30.0,
        n_points: int = 2000,
        r_start: float = 0.01
    ):
        """
        参数:
            potential: 光学势对象
            r_max: 积分外边界 [fm]
            n_points: 网格点数
            r_start: 积分起始点 (避开原点奇点) [fm]
        """
        self.pot = potential
        self.r_max = r_max
        self.n_points = n_points
        self.r_start = r_start
        self.r = np.linspace(r_start, r_max, n_points)
        self.dr = self.r[1] - self.r[0]

        # 入射波数
        self.k = potential.p.k_wavevector
        self.E = potential.p.energy
        self.m_red = potential.p.mass_reduced

    def _local_wavevector_squared(self, V: np.ndarray, l: int) -> np.ndarray:
        """
        局部波数平方:
            K^2(r) = k^2 - (2m/hbar^2) * V(r) - l*(l+1)/r^2

        其中 k^2 = 2mE/hbar^2
        """
        kappa = 2.0 * self.m_red / HBAR_C ** 2
        centrifugal = l * (l + 1) / self.r ** 2
        K2 = self.k ** 2 - kappa * V - centrifugal
        return K2

    def solve_numerov(
        self, l: int, V: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Numerov 方法求解径向方程 (4阶精度):

        定义:
            k_n^2 = K^2(r_n) = k^2 - 2m/hbar^2 * V(r_n) - l(l+1)/r_n^2
            g_n = 1 + h^2/12 * k_n^2   (注意符号约定)

        Numerov 递推:
            g_{n+1} * u_{n+1} = 2*(1 - 5*h^2/12 * k_n^2) * u_n
                               - g_{n-1} * u_{n-1}

        边界条件:
            u(r_0) = 0
            u(r_1) = r_1^(l+1) (小r渐近行为)

        返回:
            (r, u): 网格点和约化径向波函数
        """
        n = self.n_points
        h = self.dr
        h2 = h * h
        h2_12 = h2 / 12.0

        # 计算局部波数平方
        K2 = self._local_wavevector_squared(V.real, l)  # Numerov 用实部

        # Numerov 系数
        # g_n = 1 + h^2/12 * K^2_n
        g = 1.0 + h2_12 * K2
        # f_n = 2*(1 - 5*h^2/12 * K^2_n)
        f = 2.0 * (1.0 - 5.0 * h2_12 * K2)

        u = np.zeros(n, dtype=np.complex128)

        # 边界条件
        u[0] = 0.0 + 0.0j
        if n > 1:
            # 小 r 行为: u ~ r^(l+1)
            u[1] = (self.r[1] ** (l + 1)) * (1.0 + 0.0j)

        # Numerov 向前递推
        for i in range(1, n - 1):
            if abs(g[i + 1]) < 1e-15:
                # 数值保护: 防止除零
                g[i + 1] = 1e-15

            u[i + 1] = (f[i] * u[i] - g[i - 1] * u[i - 1]) / g[i + 1]

            # 数值溢出保护 (对数重新标度)
            if abs(u[i + 1]) > 1e100:
                scale = 1e-100
                u[:i + 2] *= scale

        return self.r.copy(), u

    def solve_second_order_cd(
        self, l: int, V: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        二阶中心差分求解 (作为 Numerov 的基准对比):

        d^2u/dr^2 ≈ (u_{n+1} - 2*u_n + u_{n-1}) / h^2 = -K^2_n * u_n

        => u_{n+1} = 2*u_n - u_{n-1} - h^2 * K^2_n * u_n
                   = (2 - h^2 * K^2_n) * u_n - u_{n-1}
        """
        n = self.n_points
        h = self.dr
        h2 = h * h

        K2 = self._local_wavevector_squared(V.real, l)

        u = np.zeros(n, dtype=np.complex128)
        u[0] = 0.0 + 0.0j
        if n > 1:
            u[1] = self.r[1] ** (l + 1) * (1.0 + 0.0j)

        for i in range(1, n - 1):
            u[i + 1] = (2.0 - h2 * K2[i]) * u[i] - u[i - 1]

            if abs(u[i + 1]) > 1e100:
                scale = 1e-100
                u[:i + 2] *= scale

        return self.r.copy(), u

    def solve_ftcs_time_evolution(
        self, l: int, V: np.ndarray,
        n_steps: int = 500,
        dt: float = 0.001,
        psi_init: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        FTCS (Forward Time Centered Space) 时间推进求解
        (映射自 353_fd1d_advection_ftcs — 演示不稳定性)

        含时薛定谔方程:
            i*hbar * dpsi/dt = H * psi
            dpsi/dt = -(i/hbar) * H * psi

        FTCS 离散化:
            psi^{n+1}_j = psi^n_j - i*dt/hbar * [
                -hbar^2/(2m*h^2) * (psi^n_{j+1} - 2*psi^n_j + psi^n_{j-1})
                + V_j * psi^n_j
            ]

        定义 Courant 数:
            C = hbar * dt / (2 * m * h^2)

        FTCS 无条件不稳定 (冯·诺伊曼分析):
            增长因子 |G| = sqrt(1 + 4*C^2*sin^2(kh/2)) >= 1

        参数:
            l: 角动量量子数
            V: 势能数组
            n_steps: 时间步数
            dt: 时间步长 [fm/c]
            psi_init: 初始波函数

        返回:
            (r, psi_final, norm_history): 最终波函数和范数历史
        """
        n = self.n_points
        h = self.dr
        h2 = h * h

        # Courant 数
        C = HBAR_C * dt / (2.0 * self.m_red * h2)

        # 离心势
        centrifugal = l * (l + 1) / self.r ** 2

        # 初始波函数
        if psi_init is None:
            # 高斯波包
            r0 = self.r_max * 0.3
            sigma = self.r_max * 0.08
            k0 = self.k
            psi = np.exp(-(self.r - r0) ** 2 / (2 * sigma ** 2)) * \
                  np.exp(1j * k0 * self.r)
            psi *= self.r ** (l + 1)  # 满足原点边界条件
        else:
            psi = psi_init.copy()

        # FTCS 时间推进
        norm_history = np.zeros(n_steps)

        for step in range(n_steps):
            psi_new = np.zeros(n, dtype=np.complex128)

            # 内部点 FTCS 更新
            for j in range(1, n - 1):
                # 动能项 (中心差分)
                d2psi = (psi[j + 1] - 2.0 * psi[j] + psi[j - 1]) / h2
                # 有效势
                V_eff = V[j] + HBAR_C ** 2 / (2.0 * self.m_red) * centrifugal[j]
                # FTCS 更新
                psi_new[j] = psi[j] - 1j * dt / HBAR_C * (
                    -HBAR_C ** 2 / (2.0 * self.m_red) * d2psi + V_eff * psi[j])

            # 边界条件
            psi_new[0] = 0.0
            psi_new[-1] = psi[-2]  # 近似透射边界

            psi = psi_new
            norm_history[step] = np.sqrt(np.sum(np.abs(psi) ** 2) * h)

        return self.r.copy(), psi, norm_history

    def solve_discrete_flow_propagation(
        self, l: int, V: np.ndarray,
        n_layers: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        离散流传播法求解径向方程
        (映射自 1069_andrew-cr_discrete_flow_models)

        将径向区域分成 N 层, 每层内势视为常数,
        通过传递矩阵逐层传播波函数:

        第 j 层:
            V_j = V(r_j) (常数近似, 最近邻插值)
            kappa_j^2 = 2m/hbar^2 * (E - V_j) - l*(l+1)/r_j^2

            若 kappa_j^2 > 0 (振荡区):
                T_j = [[cos(k_j*d), sin(k_j*d)/k_j],
                       [-k_j*sin(k_j*d), cos(k_j*d)]]
            若 kappa_j^2 < 0 (隧穿区):
                q_j = sqrt(-kappa_j^2)
                T_j = [[cosh(q_j*d), sinh(q_j*d)/q_j],
                       [q_j*sinh(q_j*d), cosh(q_j*d)]]

        总传递矩阵: T = T_N * T_{N-1} * ... * T_1

        参数:
            l: 角动量量子数
            V: 势能数组
            n_layers: 分层数

        返回:
            (r_layers, u_layers, transfer_matrix)
        """
        # 等间距分层
        r_layers = np.linspace(self.r_start, self.r_max, n_layers + 1)
        d = r_layers[1] - r_layers[0]

        kappa = 2.0 * self.m_red / HBAR_C ** 2

        # 总传递矩阵 (初始为单位矩阵)
        T_total = np.eye(2, dtype=np.complex128)

        # 存储每层的波函数
        u_vals = np.zeros(n_layers + 1, dtype=np.complex128)
        dup_vals = np.zeros(n_layers + 1, dtype=np.complex128)

        # 初始条件
        u_vals[0] = 0.0 + 0.0j
        dup_vals[0] = 1.0 + 0.0j  # u'(r_0) = 1 (归一化任意)

        for j in range(n_layers):
            r_mid = 0.5 * (r_layers[j] + r_layers[j + 1])

            # 最近邻插值获取势能值 (映射自 792_nearest_interp_1d)
            idx = np.argmin(np.abs(self.r - r_mid))
            V_j = V[idx]

            # 局部波数
            centrifugal = l * (l + 1) / r_mid ** 2 if r_mid > 1e-10 else 0.0
            k2_j = kappa * (self.E - V_j) - centrifugal

            # 层传递矩阵
            T_j = np.eye(2, dtype=np.complex128)
            if k2_j.real > 0:
                kj = np.sqrt(complex(k2_j))
                kj_d = kj * d
                T_j[0, 0] = np.cos(kj_d)
                T_j[0, 1] = np.sin(kj_d) / kj if abs(kj) > 1e-15 else d
                T_j[1, 0] = -kj * np.sin(kj_d)
                T_j[1, 1] = np.cos(kj_d)
            else:
                qj = np.sqrt(complex(-k2_j))
                qj_d = qj * d
                # 双曲函数保护
                qj_d = min(abs(qj_d), 50.0) * np.sign(qj_d + 1e-30)
                T_j[0, 0] = np.cosh(qj_d)
                T_j[0, 1] = np.sinh(qj_d) / qj if abs(qj) > 1e-15 else d
                T_j[1, 0] = qj * np.sinh(qj_d)
                T_j[1, 1] = np.cosh(qj_d)

            T_total = T_j @ T_total

        return r_layers, u_vals, T_total

    def get_solver_info(self) -> Dict:
        """返回求解器参数信息"""
        return {
            'r_max': self.r_max,
            'n_points': self.n_points,
            'dr': self.dr,
            'r_start': self.r_start,
            'wave_number_k': self.k,
            'energy': self.E,
            'reduced_mass': self.m_red,
        }
