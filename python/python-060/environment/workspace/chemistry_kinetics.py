# -*- coding: utf-8 -*-
"""
chemistry_kinetics.py
平流层臭氧化学动力学模块。

核心反应网络（Chapman 机制 + 催化循环）：

  R1:  O2 + hν → O + O              (J_O2)
  R2:  O + O2 + M → O3 + M          (k2)
  R3:  O3 + hν → O2 + O(1D)         (J_O3)
  R4:  O + O3 → 2 O2                (k4)
  R5:  NO + O3 → NO2 + O2           (k5)
  R6:  NO2 + O → NO + O2            (k6)
  R7:  Cl + O3 → ClO + O2           (k7)
  R8:  ClO + O → Cl + O2            (k8)
  R9:  OH + O3 → HO2 + O2           (k9)
 R10:  HO2 + O → OH + O2            (k10)

物种向量：
  c = [O, O(1D), O3, NO, NO2, Cl, ClO, OH, HO2, O2, N2, M]^T

连续性方程：

    \frac{\partial c_i}{\partial t} = P_i - L_i c_i + S_i^{trans}

其中 P_i 为生产项，L_i 为损失频率，S_i^{trans} 为输运源汇。

融合来源：
  - 572_ill_bvp: 边界值问题的 ODE 离散化思想
"""

import numpy as np
from utils import clip_positive, safe_divide

# 阿伦尼乌斯参数（简化但基于真实大气化学）
# 单位：cm^3 molecule^{-1} s^{-1}（双分子），cm^6 molecule^{-2} s^{-1}（三体）
ARR2 = {
    'O_O3':     {'A': 8.0e-12, 'Ea': 2060.0},   # O + O3 -> 2O2
    'NO_O3':    {'A': 1.8e-12, 'Ea': 1370.0},   # NO + O3 -> NO2 + O2
    'NO2_O':    {'A': 9.3e-12, 'Ea': -120.0},   # NO2 + O -> NO + O2
    'Cl_O3':    {'A': 2.3e-11, 'Ea': 200.0},    # Cl + O3 -> ClO + O2
    'ClO_O':    {'A': 2.8e-11, 'Ea': -260.0},   # ClO + O -> Cl + O2
    'OH_O3':    {'A': 1.7e-12, 'Ea': 940.0},    # OH + O3 -> HO2 + O2
    'HO2_O':    {'A': 2.9e-11, 'Ea': -200.0},   # HO2 + O -> OH + O2
}

# 三体反应 O + O2 + M -> O3 + M
ARR3 = {'O_O2_M': {'A': 6.0e-34, 'Ea': -1450.0, 'n': 2.4}}

R_GAS = 1.987  # 气体常数 [cal mol^{-1} K^{-1}]


def arrhenius_rate(A, Ea, T):
    r"""
    阿伦尼乌斯速率常数：

        k(T) = A \exp\!\left(-\frac{E_a}{R T}\right)

    Parameters
    ----------
    A : float
        指前因子。
    Ea : float
        活化能 [cal/mol 或 kcal/mol，与 R 单位匹配]。
    T : float
        温度 [K]。

    Returns
    -------
    k : float
    """
    T = clip_positive(T, 100.0)
    return A * np.exp(-Ea / (R_GAS * T))


def three_body_rate(A, Ea, n, T, M):
    r"""
    三体反应速率常数（简化 Troe 表达式）：

        k_0(T) = A \left(\frac{T}{300}\right)^{-n} \exp\!\left(-\frac{E_a}{R T}\right)
        k = k_0(T) \cdot M

    Parameters
    ----------
    A, Ea, n : float
    T : float
    M : float
        第三体数密度 [molecules cm^{-3}]。

    Returns
    -------
    k : float
    """
    T = clip_positive(T, 100.0)
    k0 = A * (T / 300.0) ** (-n) * np.exp(-Ea / (R_GAS * T))
    return k0 * M


class StratosphericChemistry:
    r"""
    平流层臭氧化学动力学求解器。

    求解 stiff ODE 系统：

        \frac{d\mathbf{c}}{dt} = \mathbf{f}(\mathbf{c}, t)

    采用半隐式 Rosenbrock 方法（SDIRK 简化）处理刚性。
    """

    # 物种索引
    IDX_O = 0
    IDX_O1D = 1
    IDX_O3 = 2
    IDX_NO = 3
    IDX_NO2 = 4
    IDX_Cl = 5
    IDX_ClO = 6
    IDX_OH = 7
    IDX_HO2 = 8
    IDX_O2 = 9
    IDX_N2 = 10
    IDX_M = 11
    N_SPECIES = 12

    def __init__(self, T_k=220.0, M_cm3=2.5e19):
        self.T = T_k
        self.M = M_cm3
        self._compute_rate_constants()

    def _compute_rate_constants(self):
        r"""
        预计算各反应速率常数。
        """
        self.k = {}
        for key, param in ARR2.items():
            self.k[key] = arrhenius_rate(param['A'], param['Ea'], self.T)
        for key, param in ARR3.items():
            self.k[key] = three_body_rate(param['A'], param['Ea'], param['n'], self.T, self.M)

    def set_photolysis_rates(self, J_o2, J_o3):
        r"""
        设置光解速率常数。

        Parameters
        ----------
        J_o2, J_o3 : float
            单位 s^{-1}。
        """
        self.J_o2 = float(clip_positive(J_o2))
        self.J_o3 = float(clip_positive(J_o3))

    def set_temperature(self, T_k):
        r"""更新温度并重算速率常数。"""
        self.T = T_k
        self._compute_rate_constants()

    def production_loss(self, c, J_o2=None, J_o3=None):
        r"""
        计算各物种的生产项 P 与损失频率 L。

        Parameters
        ----------
        c : ndarray, shape (N_SPECIES,)
            物种浓度 [molecules cm^{-3}]。
        J_o2, J_o3 : float, optional

        Returns
        -------
        P : ndarray
            生产项 [molecules cm^{-3} s^{-1}]。
        L : ndarray
            损失频率 [s^{-1}]。
        """
        c = np.asarray(c, dtype=float)
        if c.size < self.N_SPECIES:
            c = np.pad(c, (0, self.N_SPECIES - c.size), constant_values=0.0)
        c = np.maximum(c, 1e-30)

        if J_o2 is None:
            J_o2 = getattr(self, 'J_o2', 1e-11)
        if J_o3 is None:
            J_o3 = getattr(self, 'J_o3', 1e-3)

        k = self.k
        O, O1D, O3 = c[self.IDX_O], c[self.IDX_O1D], c[self.IDX_O3]
        NO, NO2 = c[self.IDX_NO], c[self.IDX_NO2]
        Cl, ClO = c[self.IDX_Cl], c[self.IDX_ClO]
        OH, HO2 = c[self.IDX_OH], c[self.IDX_HO2]
        O2 = c[self.IDX_O2]
        M = c[self.IDX_M]

        P = np.zeros(self.N_SPECIES)
        L = np.zeros(self.N_SPECIES)

        # R1: O2 + hν -> 2O
        P[self.IDX_O] += 2.0 * J_o2 * O2
        L[self.IDX_O2] += J_o2

        # R2: O + O2 + M -> O3 + M
        P[self.IDX_O3] += k['O_O2_M'] * O * O2
        L[self.IDX_O] += k['O_O2_M'] * O2
        L[self.IDX_O2] += k['O_O2_M'] * O

        # R3: O3 + hν -> O2 + O(1D)
        P[self.IDX_O1D] += J_o3 * O3
        P[self.IDX_O2] += J_o3 * O3
        L[self.IDX_O3] += J_o3

        # R4: O + O3 -> 2O2
        rx = k['O_O3'] * O * O3
        P[self.IDX_O2] += 2.0 * rx
        L[self.IDX_O] += k['O_O3'] * O3
        L[self.IDX_O3] += k['O_O3'] * O

        # R5: NO + O3 -> NO2 + O2
        rx = k['NO_O3'] * NO * O3
        P[self.IDX_NO2] += rx
        P[self.IDX_O2] += rx
        L[self.IDX_NO] += k['NO_O3'] * O3
        L[self.IDX_O3] += k['NO_O3'] * NO

        # R6: NO2 + O -> NO + O2
        rx = k['NO2_O'] * NO2 * O
        P[self.IDX_NO] += rx
        P[self.IDX_O2] += rx
        L[self.IDX_NO2] += k['NO2_O'] * O
        L[self.IDX_O] += k['NO2_O'] * NO2

        # R7: Cl + O3 -> ClO + O2
        rx = k['Cl_O3'] * Cl * O3
        P[self.IDX_ClO] += rx
        P[self.IDX_O2] += rx
        L[self.IDX_Cl] += k['Cl_O3'] * O3
        L[self.IDX_O3] += k['Cl_O3'] * Cl

        # R8: ClO + O -> Cl + O2
        rx = k['ClO_O'] * ClO * O
        P[self.IDX_Cl] += rx
        P[self.IDX_O2] += rx
        L[self.IDX_ClO] += k['ClO_O'] * O
        L[self.IDX_O] += k['ClO_O'] * ClO

        # R9: OH + O3 -> HO2 + O2
        rx = k['OH_O3'] * OH * O3
        P[self.IDX_HO2] += rx
        P[self.IDX_O2] += rx
        L[self.IDX_OH] += k['OH_O3'] * O3
        L[self.IDX_O3] += k['OH_O3'] * OH

        # R10: HO2 + O -> OH + O2
        rx = k['HO2_O'] * HO2 * O
        P[self.IDX_OH] += rx
        P[self.IDX_O2] += rx
        L[self.IDX_HO2] += k['HO2_O'] * O
        L[self.IDX_O] += k['HO2_O'] * HO2

        # O(1D) 快速猝灭: O(1D) + M -> O + M
        k_quench = 2.9e-11 * np.exp(-67.0 / self.T) if self.T > 0 else 2.9e-11
        P[self.IDX_O] += k_quench * O1D * M
        L[self.IDX_O1D] += k_quench * M

        return P, L

    def rhs(self, c, t=0.0, transport_source=None):
        r"""
        计算 ODE 右端项 f(c) = P - L·c + S_trans。

        Parameters
        ----------
        c : ndarray
        t : float
        transport_source : ndarray, optional
            输运源汇项 [molecules cm^{-3} s^{-1}]。

        Returns
        -------
        f : ndarray
        """
        P, L = self.production_loss(c)
        f = P - L * c[:self.N_SPECIES]
        if transport_source is not None:
            s = np.asarray(transport_source, dtype=float)
            if s.size < self.N_SPECIES:
                s = np.pad(s, (0, self.N_SPECIES - s.size))
            f += s[:self.N_SPECIES]
        return f

    def jacobian_approx(self, c, J_o2=None, J_o3=None):
        r"""
        近似 Jacobian 矩阵 J_{ij} = ∂f_i/∂c_j（对角近似，用于半隐式积分）。

        对角元近似为：

            J_{ii} \approx -L_i

        Parameters
        ----------
        c : ndarray

        Returns
        -------
        J_diag : ndarray
            Jacobian 对角元。
        """
        _, L = self.production_loss(c, J_o2, J_o3)
        return -L

    def step_rosenbrock(self, c, dt, transport_source=None):
        r"""
        单步一阶 Rosenbrock 方法（线性隐式欧拉）：

            (I - dt J) \Delta c = dt f(c^n)
            c^{n+1} = c^n + \Delta c

        这里使用对角近似 J = diag(J_diag)。

        Parameters
        ----------
        c : ndarray
        dt : float
            时间步长 [s]。
        transport_source : ndarray, optional

        Returns
        -------
        c_new : ndarray
        """
        # TODO: 实现对角隐式 Rosenbrock 步进
        # 需要计算 rhs(c) 和 jacobian_approx(c)，然后求解 (I - dt*J_diag)*dc = dt*f
        raise NotImplementedError("Hole 1: 请实现对角隐式 Rosenbrock 时间步进")

    def integrate(self, c0, t_span, dt_max=60.0, transport_source=None):
        r"""
        时间积分直到 t_span。

        Parameters
        ----------
        c0 : ndarray
        t_span : float
            总积分时间 [s]。
        dt_max : float
            最大时间步长。
        transport_source : ndarray or callable
            若 callable，则 signature 为 S(t)。

        Returns
        -------
        c : ndarray
            最终浓度。
        t_history : list
        c_history : list
        """
        c = np.asarray(c0, dtype=float).copy()
        t = 0.0
        t_history = [0.0]
        c_history = [c.copy()]

        while t < t_span:
            dt = dt_max
            if t + dt > t_span:
                dt = t_span - t

            if callable(transport_source):
                S = transport_source(t)
            else:
                S = transport_source

            # 对角隐式 Rosenbrock 具有 A-稳定性，允许大步长跨越刚性。
            # 此处使用简化的局部误差估计进行步长控制。
            c_new = self.step_rosenbrock(c, dt, S)
            c_half1 = self.step_rosenbrock(c, dt * 0.5, S)
            c_half2 = self.step_rosenbrock(c_half1, dt * 0.5, S)
            err = np.linalg.norm(c_new - c_half2) / (np.linalg.norm(c_new) + 1e-20)
            if err < 0.2:
                c = c_half2
                t += dt
                t_history.append(t)
                c_history.append(c.copy())
            else:
                dt_max = max(dt * 0.5, 1.0)

        return c, t_history, c_history
