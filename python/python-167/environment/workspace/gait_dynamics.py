"""
gait_dynamics.py
多足机器人步态动力学与中枢模式发生器（CPG）模块。
融入种子项目：
  - 1288_trapezoidal_fixed（梯形法固定点迭代 ODE 积分）
  - 671_life（Conway 生命游戏 → 映射为离散布尔足部支撑状态机）

科学背景：
多足机器人步态可建模为耦合非线性振荡器网络：
    τ·q̇_i + q_i = Σ_j w_{ij}·φ(q_j) + u_i(t) + ξ_i
其中 φ 为激活函数，u_i 为高位指令，ξ_i 为地形反馈。
同时，每条腿的支撑/摆动（stance/swing）相位可用离散细胞自动机近似，
实现混合连续-离散动力学系统。
"""

import numpy as np
from typing import Callable, Tuple, List
from utils import clip_to_bounds, robust_sqrt


class CPGNetwork:
    """
    耦合 Hopf 振荡器网络作为 CPG 模型。

    单个 Hopf 振荡器（极限环）：
        ẋ = α·(μ - r^2)·x - ω·y
        ẏ = α·(μ - r^2)·y + ω·x
    其中 r^2 = x^2 + y^2，μ > 0 控制振幅，ω 控制频率，α 控制收敛速度。

    对 N 个振荡器，加入耦合项：
        ẋ_i = α·(μ - r_i^2)·x_i - ω_i·y_i + Σ_j c_{ij}·(x_j - x_i)
        ẏ_i = α·(μ - r_i^2)·y_i + ω_i·x_i + Σ_j c_{ij}·(y_j - y_i)
    """

    def __init__(self, n_osc: int = 6, alpha: float = 50.0, mu: float = 1.0,
                 omega: float = 2.0 * np.pi * 1.0, coupling_strength: float = 5.0):
        self.n = n_osc
        self.alpha = alpha
        self.mu = mu
        self.omega = omega
        # 三足交替步态（tripod gait）相位耦合矩阵
        # 腿 0,2,4 与腿 1,3,5 反相（相位差 π）
        self.C = np.zeros((n_osc, n_osc))
        for i in range(n_osc):
            for j in range(n_osc):
                if i != j:
                    # 同组腿强同步，异组腿反相耦合
                    same_group = (i % 2) == (j % 2)
                    self.C[i, j] = coupling_strength if same_group else -coupling_strength * 0.5

    def rhs(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        状态 state 为 2N 维向量：[x0, x1, ..., x_{N-1}, y0, y1, ..., y_{N-1}]

        TODO: 实现耦合 Hopf 振荡器的右侧表达式。
        单个 Hopf 极限环：
            ẋ = α·(μ - r²)·x - ω·y
            ẏ = α·(μ - r²)·y + ω·x
        其中 r² = x² + y²。
        对 N 个振荡器，加入耦合项：
            ẋ_i = α·(μ - r_i²)·x_i - ω·y_i + Σ_j c_{ij}·(x_j - x_i)
            ẏ_i = α·(μ - r_i²)·y_i + ω·x_i + Σ_j c_{ij}·(y_j - y_i)
        返回 np.concatenate((dx, dy))。
        """
        raise NotImplementedError("Hole 1: 请补全 CPGNetwork.rhs 的实现")

    def extract_phase(self, state: np.ndarray) -> np.ndarray:
        """
        从状态提取相位角 φ_i = atan2(y_i, x_i)。
        """
        x = state[:self.n]
        y = state[self.n:]
        return np.arctan2(y, x)

    def extract_amplitude(self, state: np.ndarray) -> np.ndarray:
        return np.sqrt(state[:self.n] ** 2 + state[self.n:] ** 2)


class TrapezoidalIntegrator:
    """
    源自 trapezoidal_fixed.m 的固定点迭代梯形法。

    数学原理：
    对 ODE  ẏ = f(t, y)，梯形离散格式为
        y_{n+1} = y_n + h/2 · [ f(t_n, y_n) + f(t_{n+1}, y_{n+1}) ]
    由于 y_{n+1} 出现在等式两边，采用固定点迭代：
        y_{n+1}^{(k+1)} = y_n + h/2 · [ f(t_n, y_n) + f(t_{n+1}, y_{n+1}^{(k)}) ]
    迭代至收敛（默认 10 次），局部截断误差 O(h^3)，全局误差 O(h^2)。
    """

    def __init__(self, it_max: int = 10):
        self.it_max = it_max

    def integrate(self, f: Callable, tspan: Tuple[float, float], y0: np.ndarray,
                  n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        返回时间数组 t (n_steps+1,) 与解数组 y (n_steps+1, dim)。
        """
        t0, tf = tspan
        h = (tf - t0) / n_steps
        dim = y0.size
        t = np.linspace(t0, tf, n_steps + 1)
        y = np.zeros((n_steps + 1, dim))
        y[0] = y0.flatten()
        for i in range(n_steps):
            tn = t[i]
            yn = y[i]
            f_tn = f(tn, yn)
            # TODO: 实现梯形法固定点迭代。
            # 数学原理：
            #   y_{n+1} = y_n + h/2 · [ f(t_n, y_n) + f(t_{n+1}, y_{n+1}) ]
            # 由于 y_{n+1} 出现在等式两边，采用固定点迭代：
            #   y_{n+1}^{(k+1)} = y_n + h/2 · [ f(t_n, y_n) + f(t_{n+1}, y_{n+1}^{(k)}) ]
            # 先用显式 Euler 预测初值，再迭代 self.it_max 次至收敛。
            # 局部截断误差 O(h³)，全局误差 O(h²)。
            raise NotImplementedError("Hole 2: 请补全 TrapezoidalIntegrator.integrate 的梯形固定点迭代")
        return t, y


class StanceSwingAutomaton:
    """
    源自 life_update.m 的细胞自动机思想，映射到多足机器人支撑-摆动相位。

    每条腿 i 在离散时间 t 处于状态 s_i(t) ∈ {0, 1}：
        0 = 摆动 (swing)
        1 = 支撑 (stance)

    状态转移规则（基于 CPG 相位与相邻腿状态）：
    1. 若腿 i 的 CPG 相位 φ_i ∈ [-π/2, π/2]，则倾向支撑；否则倾向摆动。
    2. 若相邻腿（考虑拓扑邻接）多数处于摆动，则腿 i 强制支撑（稳定性约束）。
    3. 支撑持续时间存在最小阈值 T_stance_min，防止过早抬腿。

    该规则与生命游戏的局部邻居交互思想同构，但物理意义完全不同。
    """

    def __init__(self, n_legs: int = 6, stance_min_steps: int = 3,
                 phase_stance_center: float = 0.0, phase_stance_width: float = np.pi / 2):
        self.n = n_legs
        self.stance_min = stance_min_steps
        self.phi_c = phase_stance_center
        self.phi_w = phase_stance_width
        # 记录每条腿的连续支撑步数
        self.stance_counter = np.zeros(n_legs, dtype=int)
        # 邻接拓扑：六足机器人通常呈环形排列
        self.neighbors = {i: [(i - 1) % n_legs, (i + 1) % n_legs] for i in range(n_legs)}

    def update(self, phase: np.ndarray) -> np.ndarray:
        """
        根据当前 CPG 相位更新支撑/摆动状态。
        返回新的状态向量 s_new ∈ {0,1}^n。
        """
        s_new = np.zeros(self.n, dtype=int)
        for i in range(self.n):
            phi = phase[i]
            # 规则1：相位决定基础倾向
            in_stance_window = abs(((phi - self.phi_c + np.pi) % (2 * np.pi)) - np.pi) <= self.phi_w
            desired = 1 if in_stance_window else 0

            # 规则2：邻居摆动多数时强制支撑
            neighbor_swing_count = sum(1 for j in self.neighbors[i] if self.stance_counter[j] == 0)
            if neighbor_swing_count >= len(self.neighbors[i]):
                desired = 1

            # 规则3：最小支撑时间约束
            if self.stance_counter[i] > 0 and self.stance_counter[i] < self.stance_min and desired == 0:
                desired = 1

            s_new[i] = desired
            if s_new[i] == 1:
                self.stance_counter[i] += 1
            else:
                self.stance_counter[i] = 0
        return s_new

    def reset(self):
        self.stance_counter = np.zeros(self.n, dtype=int)


class LegDynamics:
    """
    单腿动力学：简化质量-弹簧-阻尼模型。

    连续动力学方程（支撑相）：
        M·q̈ + C(q, q̇)·q̇ + G(q) = τ + J^T·f_c
    其中 M 为惯性矩阵，C 为科氏力与离心力矩阵，G 为重力项，
    τ 为关节力矩，f_c 为足端接触力，J 为足端 Jacobian。

    本模块将其降阶为一阶 ODE 系统：
        [ q̇ ]   [       q̇        ]
        [   ] = [ M^{-1}·(τ + J^T f_c - C·q̇ - G) ]
        [ q̈ ]   [                              ]
    """

    def __init__(self, mass_matrix: np.ndarray, damping: np.ndarray, gravity: float = 9.81):
        self.M = np.asarray(mass_matrix, dtype=float)
        self.C_mat = np.asarray(damping, dtype=float)
        self.g = gravity
        self.M_inv = np.linalg.inv(self.M)

    def dynamics(self, q: np.ndarray, dq: np.ndarray, tau: np.ndarray,
                 f_contact: np.ndarray, J: np.ndarray) -> np.ndarray:
        """
        计算 q̈ = M^{-1} · (τ + J^T·f_c - C·dq - G(q))。
        为简化，G(q) 取为重力在关节空间的投影 g·J^T·e_z。
        """
        G = J.T @ np.array([0.0, 0.0, self.g])
        rhs = tau + J.T @ f_contact - self.C_mat @ dq - G
        ddq = self.M_inv @ rhs
        return ddq

    def state_space_rhs(self, state: np.ndarray, tau: np.ndarray,
                        f_contact: np.ndarray, J_func: Callable) -> np.ndarray:
        """
        一阶状态空间 rhs，state = [q; dq]。
        J_func: 给定 q 返回 Jacobian 的函数句柄。
        """
        n = state.size // 2
        q = state[:n]
        dq = state[n:]
        J = J_func(q)
        ddq = self.dynamics(q, dq, tau, f_contact, J)
        return np.concatenate((dq, ddq))
