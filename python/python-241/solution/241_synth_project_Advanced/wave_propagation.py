"""
wave_propagation.py
===================================================================
核散射波函数离散层传播与匹配模块

映射种子项目:
  - 1069_andrew-cr_discrete_flow_models: 离散流去噪 → 波函数逐层传播 (传递矩阵)
  - 792_nearest_interp_1d: 最近邻插值 → 势在层界面的插值

核心物理:
  传递矩阵法 (Transfer Matrix Method):

  将径向区域 [r_0, R_max] 分成 N 层,
  每层内势视为常数 V_j = V(r_j) (最近邻插值).

  第 j 层传递矩阵:
    振荡区 (E > V_j):
        T_j = [[cos(q_j*d), sin(q_j*d)/q_j],
               [-q_j*sin(q_j*d), cos(q_j*d)]]
        q_j = sqrt(2m/hbar^2 * (E - V_j) - l(l+1)/r_j^2)

    隧穿区 (E < V_j):
        T_j = [[cosh(p_j*d), sinh(p_j*d)/p_j],
               [p_j*sinh(p_j*d), cosh(p_j*d)]]
        p_j = sqrt(2m/hbar^2 * (V_j - E) + l(l+1)/r_j^2)

  总传递矩阵:
    M = T_N * T_{N-1} * ... * T_1

  从 M 提取 S 矩阵:
    S = -(M_21 - i*k*M_11 + i*k*M_22 + k^2*M_12) / ...

  波函数匹配 (渐近区):
    u(r) -> A*(cos(delta)*j_l(kr) - sin(delta)*n_l(kr))  (r -> inf)
===================================================================
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from optical_potential import OpticalPotential


# ---------- 物理常数 ----------
HBAR_C = 197.3269804


class DiscreteLayerPropagator:
    """
    离散层传递矩阵传播器
    (映射自 discrete_flow_models 的逐层去噪/传播思想)

    核心思想:
    1. 将连续势离散化为 N 个常势层
    2. 每层用解析解 (平面波/指数衰减) 精确传播
    3. 层间通过连续性条件匹配
    4. 总传递矩阵 = 各层矩阵之积

    优势:
    - 避免数值积分累积误差
    - 自然处理隧穿区
    - 适合光学势 (复数势)
    """

    def __init__(
        self,
        potential: OpticalPotential,
        n_layers: int = 200,
        r_max: float = 30.0,
        r_start: float = 0.1,
    ):
        self.pot = potential
        self.n_layers = n_layers
        self.r_max = r_max
        self.r_start = r_start

        # 层边界
        self.r_boundaries = np.linspace(r_start, r_max, n_layers + 1)
        self.layer_width = (r_max - r_start) / n_layers

        # 波数
        self.k = potential.p.k_wavevector
        self.E = potential.p.energy
        self.m_red = potential.p.mass_reduced
        self.kappa = 2.0 * self.m_red / HBAR_C ** 2

    def _get_potential_at_layer(
        self, r_mid: float, V_grid: np.ndarray, r_grid: np.ndarray
    ) -> complex:
        """
        最近邻插值获取层中点势能
        (映射自 792_nearest_interp_1d)

        对 r_mid, 找到 r_grid 中最近的网格点, 返回对应势值
        """
        idx = np.argmin(np.abs(r_grid - r_mid))
        return V_grid[idx]

    def single_layer_matrix(
        self, V_layer: complex, r_mid: float, l: int, d: float
    ) -> np.ndarray:
        """
        计算单层的传递矩阵

        局部波数:
            q^2 = 2m/hbar^2 * (E - V) - l*(l+1)/r^2

        振荡区 (q^2 > 0, 实数 q):
            T = [[cos(qd), sin(qd)/q],
                 [-q*sin(qd), cos(qd)]]

        隧穿区 (q^2 < 0, 虚数 q = i*p):
            T = [[cosh(pd), sinh(pd)/p],
                 [p*sinh(pd), cosh(pd)]]

        参数:
            V_layer: 层内势能 (复数)
            r_mid: 层中心径向位置
            l: 角动量量子数
            d: 层厚度

        返回:
            2x2 传递矩阵
        """
        # 局部波数平方 (含离心项)
        centrifugal = l * (l + 1) / r_mid ** 2 if r_mid > 1e-10 else 0.0
        q2 = self.kappa * (self.E - V_layer) - centrifugal

        T = np.eye(2, dtype=complex)

        if abs(q2) < 1e-15:
            # 近零波数: 自由传播近似
            T[0, 1] = d
            return T

        q = np.sqrt(complex(q2))

        if q2.real > 0 and abs(q2.imag) < abs(q2.real):
            # 振荡区
            qd = q * d
            # 数值保护
            if abs(qd) > 100:
                # 大相位: 使用三角恒等式避免精度损失
                qd_mod = qd.real % (2 * np.pi) + 1j * qd.imag
                T[0, 0] = np.cos(qd_mod)
                T[0, 1] = np.sin(qd_mod) / q if abs(q) > 1e-15 else d
                T[1, 0] = -q * np.sin(qd_mod)
                T[1, 1] = np.cos(qd_mod)
            else:
                T[0, 0] = np.cos(qd)
                T[0, 1] = np.sin(qd) / q
                T[1, 0] = -q * np.sin(qd)
                T[1, 1] = np.cos(qd)
        else:
            # 隧穿区 / 复数波数
            p = np.sqrt(complex(-q2))
            pd = p * d

            # 数值保护: 限制双曲函数增长
            pd_real = min(abs(pd.real), 50.0)
            pd = pd_real * (pd / (abs(pd) + 1e-30)) if abs(pd) > 0 else 0.0

            if abs(p) > 1e-15:
                T[0, 0] = np.cosh(pd)
                T[0, 1] = np.sinh(pd) / p
                T[1, 0] = p * np.sinh(pd)
                T[1, 1] = np.cosh(pd)
            else:
                T[0, 1] = d

        return T

    def propagate(
        self,
        V_grid: np.ndarray,
        r_grid: np.ndarray,
        l: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行完整的离散层传播

        步骤:
        1. 在每层中点获取势能值 (最近邻插值)
        2. 计算单层传递矩阵
        3. 累乘得到总传递矩阵 M
        4. 从 M 提取 S 矩阵和相移

        参数:
            V_grid: 势能数组 (在 r_grid 上)
            r_grid: 径向网格
            l: 角动量量子数

        返回:
            (S_matrix, total_transfer_matrix)
        """
        # 总传递矩阵 (初始为单位矩阵)
        M_total = np.eye(2, dtype=complex)

        for j in range(self.n_layers):
            r_mid = 0.5 * (self.r_boundaries[j] + self.r_boundaries[j + 1])
            V_j = self._get_potential_at_layer(r_mid, V_grid, r_grid)

            T_j = self.single_layer_matrix(V_j, r_mid, l, self.layer_width)

            # 累乘 (注意顺序: 从内到外)
            M_total = T_j @ M_total

            # 数值稳定化: 定期重标度
            norm = np.max(np.abs(M_total))
            if norm > 1e50:
                M_total /= norm

        # 从 M 提取 S 矩阵
        S = self._extract_s_from_M(M_total)

        return S, M_total

    def _extract_s_from_M(self, M: np.ndarray) -> complex:
        """
        从传递矩阵提取 S 矩阵:

        渐近匹配:
            在 r = R_max 处, 波函数为:
                u(R) = M_11 * u_0 + M_12 * u'_0
                u'(R) = M_21 * u_0 + M_22 * u'_0

            匹配 outgoing/incoming 波:
                u(R) ~ A*(exp(-ikR) - S*exp(ikR))

        S = -(M_22*k - i*M_21 + i*k^2*M_12 - k*M_11) /
             (M_22*k + i*M_21 + i*k^2*M_12 + k*M_11)

        (简化形式, 对 u(0)=0, u'(0)=1 的初始条件)
        """
        k = self.k
        M11, M12 = M[0, 0], M[0, 1]
        M21, M22 = M[1, 0], M[1, 1]

        #  outgoing/incoming 分量
        #  u(R) = A*exp(-ikR) + B*exp(ikR)
        #  u'(R) = -ik*A*exp(-ikR) + ik*B*exp(ikR)
        #  S = B/A = (u' + ik*u) / (u' - ik*u) * exp(-2ikR)  (在 R_max)

        # 从 M 矩阵: u(R) = M11*u_0 + M12*u'_0 = M12 (因 u_0=0, u'_0=1)
        #            u'(R) = M22
        u_R = M12
        dup_R = M22

        # 对数导数
        if abs(u_R) < 1e-300:
            return complex(1.0, 0.0)  # 无散射

        L = dup_R / u_R

        # S = (L - ik) / (L + ik) * exp(-2ikR)
        # (在 R_max 处, 渐近匹配)
        num = L - 1j * k
        den = L + 1j * k

        if abs(den) < 1e-300:
            return complex(-1.0, 0.0)

        S = num / den
        # 相因子修正
        S *= np.exp(-2j * k * self.r_max)

        return S

    def compute_wavefunction_profile(
        self,
        V_grid: np.ndarray,
        r_grid: np.ndarray,
        l: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算径向波函数剖面 (逐层传播)

        从原点向外传播, 记录每层边界的 u 和 u'

        返回:
            (r_boundaries, u_profile)
        """
        # 初始条件
        state = np.array([0.0, 1.0], dtype=complex)  # [u, u'] at r_start

        u_profile = np.zeros(self.n_layers + 1, dtype=complex)
        u_profile[0] = state[0]

        for j in range(self.n_layers):
            r_mid = 0.5 * (self.r_boundaries[j] + self.r_boundaries[j + 1])
            V_j = self._get_potential_at_layer(r_mid, V_grid, r_grid)

            T_j = self.single_layer_matrix(V_j, r_mid, l, self.layer_width)
            state = T_j @ state

            u_profile[j + 1] = state[0]

            # 重标度保护
            if abs(state[0]) > 1e50:
                state /= abs(state[0])
                u_profile[:j + 2] /= abs(state[0])

        return self.r_boundaries.copy(), u_profile

    def transmission_coefficient(self, l: int = 0) -> float:
        """
        透射系数 (对反应截面的贡献):
            T_l = 1 - |S_l|^2

        T_l = 0: 纯弹性 (透明核)
        T_l = 1: 完全吸收 (黑核)
        """
        # 需要先从 propagate 获取 S
        return 0.0  # 占位, 由外部调用设置
