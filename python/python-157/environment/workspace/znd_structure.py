"""
znd_structure.py
ZND (Zel'dovich-von Neumann-Döring) 爆轰结构一维求解器
融合来源：861_pendulum_nonlinear_ode（ODE 右端项结构）
           315_double_well_ode（双阱势能思想 → 化学反应势阱）
"""
import numpy as np
from combustion_utils import (
    check_positive, check_interval,
    arrhenius_rate, sound_speed_from_prho,
    R_UNIVERSAL, DEFAULT_GAMMA, DEFAULT_Q,
    DEFAULT_E_A, DEFAULT_A_PRE, DEFAULT_RHO_0,
    DEFAULT_P_0, DEFAULT_T_0, DEFAULT_W_MOL
)


class ZNDSolver:
    r"""
    ZND 模型求解器。

    控制方程（在随爆轰波传播的坐标系 ξ = x - D*t 中）:
        mass:     rho * (u - D) = const = m_dot
        momentum: p + m_dot * (u - D) = const
        energy:   h + 0.5 * (u - D)^2 = const
        reaction: dλ/dξ = -A/D * exp(-Ea/(R*T)) * (1-λ)^n

    其中 h = e + p/rho = gamma/(gamma-1) * p/rho + (1-λ)*Q 为总比焓。
    """

    def __init__(self, gamma=DEFAULT_GAMMA, Q=DEFAULT_Q,
                 A=DEFAULT_A_PRE, Ea=DEFAULT_E_A,
                 n_order=1.0, rho0=DEFAULT_RHO_0,
                 p0=DEFAULT_P_0, T0=DEFAULT_T_0,
                 W_mol=DEFAULT_W_MOL):
        self.gamma = gamma
        self.Q = Q
        self.A = A
        self.Ea = Ea
        self.n_order = n_order
        self.rho0 = rho0
        self.p0 = p0
        self.T0 = T0
        self.W_mol = W_mol
        self.R_specific = R_UNIVERSAL / W_mol

    def cj_velocity(self):
        r"""
        计算 CJ 爆轰速度:
            D_CJ^2 = 2*(gamma^2 - 1)*Q + gamma*p0/rho0
        """
        a0_sq = self.gamma * self.p0 / self.rho0
        D_cj_sq = 2.0 * (self.gamma ** 2 - 1.0) * self.Q + a0_sq
        if D_cj_sq <= 0.0:
            raise ValueError("CJ velocity squared non-positive")
        return np.sqrt(D_cj_sq)

    def von_neumann_state(self, D=None):
        r"""
        计算 Von Neumann 尖峰状态（激波后的状态，尚未反应）。
        使用 Rankine-Hugoniot 关系:
            M = D / a0
            p_vN/p0 = 1 + 2*gamma/(gamma+1) * (M^2 - 1)
            rho_vN/rho0 = (gamma+1)*M^2 / ((gamma-1)*M^2 + 2)
        """
        if D is None:
            D = self.cj_velocity()
        check_positive(D, "D")
        a0 = np.sqrt(self.gamma * self.p0 / self.rho0)
        M = D / a0
        p_ratio = 1.0 + 2.0 * self.gamma / (self.gamma + 1.0) * (M * M - 1.0)
        rho_ratio = (self.gamma + 1.0) * M * M / ((self.gamma - 1.0) * M * M + 2.0)
        p_vn = self.p0 * p_ratio
        rho_vn = self.rho0 * rho_ratio
        T_vn = p_vn / (rho_vn * self.R_specific)
        u_vn = D * (1.0 - self.rho0 / rho_vn)
        return rho_vn, p_vn, T_vn, u_vn

    def _rhs(self, y, D):
        r"""
        ZND ODE 右端项:
            y = [rho, u, p, lambda]^T
        在波坐标系 ξ 中的演化。

        质量守恒:   rho * (u - D) = rho_vN * (u_vN - D)
        动量守恒:   p + rho*(u-D)^2 = p_vN + rho_vN*(u_vN-D)^2
        能量守恒:   gamma/(gamma-1) * p/rho + 0.5*(u-D)^2 + (1-λ)*Q = const
        反应方程:   dλ/dξ = -A/D * exp(-Ea/(R*T)) * (1-λ)^n
        """
        rho, u, p, lam = y
        if rho <= 0.0 or p <= 0.0:
            return np.zeros(4)
        if lam < 0.0:
            lam = 0.0
        if lam > 1.0:
            lam = 1.0

        T = p / (rho * self.R_specific)
        if T <= 0.0:
            T = 1.0e-6

        # TODO: 实现 ZND ODE 右端项的核心计算
        # 1. 计算反应进度空间导数 dλ/dξ = -A/D * exp(-Ea/(R*T)) * (1-λ)^n
        # 2. 利用质量、动量、能量守恒的微分关系，推导 drho/dξ, du/dξ, dp/dξ
        # 提示: 使用 m_dot = rho_vN * (u_vN - D) 和 v_rel = m_dot / rho
        # 注意与 combustion_utils.arrhenius_rate 及 reaction_kinetics.chemical_source_term 的协同一致性
        raise NotImplementedError("Hole_3: 请实现 ZND ODE 右端项的守恒关系推导")

    def solve(self, D=None, ximax=1.0e-2, npts=2000):
        r"""
        使用改进 Euler 法（Heun 法）积分 ZND 结构。
        返回:
            xi: 空间坐标数组 [m]
            sol: [rho, u, p, lambda] 的解矩阵
        """
        if D is None:
            D = self.cj_velocity()
        check_positive(D, "D")
        check_positive(ximax, "ximax")
        check_positive(npts, "npts")

        rho_vn, p_vn, T_vn, u_vn = self.von_neumann_state(D)
        self.rho_vn = rho_vn
        self.u_vn = u_vn

        # 初始条件
        y = np.array([rho_vn, u_vn, p_vn, 0.0])
        xi = np.linspace(0.0, ximax, npts)
        sol = np.zeros((npts, 4))
        sol[0] = y

        for i in range(1, npts):
            dx = xi[i] - xi[i - 1]
            k1 = self._rhs(y, D)
            y2 = y + dx * k1
            # 边界截断
            y2[0] = max(y2[0], 1.0e-6)
            y2[2] = max(y2[2], 1.0e-6)
            y2[3] = max(0.0, min(1.0, y2[3]))
            k2 = self._rhs(y2, D)
            y = y + 0.5 * dx * (k1 + k2)
            y[0] = max(y[0], 1.0e-6)
            y[2] = max(y[2], 1.0e-6)
            y[3] = max(0.0, min(1.0, y[3]))
            sol[i] = y

        return xi, sol

    def induction_length(self, xi, sol, threshold=0.95):
        r"""
        计算诱导区长度:
            反应进度达到 threshold（默认 95%）时的位置。
        """
        lambda_profile = sol[:, 3]
        for i in range(len(xi)):
            if lambda_profile[i] >= threshold:
                return xi[i]
        return xi[-1]

    def half_reaction_length(self, xi, sol):
        r"""
        计算半反应长度:
            lambda = 0.5 时的位置。
        """
        return self.induction_length(xi, sol, threshold=0.5)
