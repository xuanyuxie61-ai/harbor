"""
reaction_kinetics.py
化学反应动力学与热力学状态模块
融合来源：861_pendulum_nonlinear_ode（参数管理与ODE RHS 结构）
           315_double_well_ode（势能/能量函数思想）
"""
import numpy as np
from combustion_utils import (
    arrhenius_rate, znd_progress_variable_derivative,
    temperature_from_energy, sound_speed_from_prho,
    check_positive, check_nonnegative, check_interval,
    R_UNIVERSAL, DEFAULT_GAMMA, DEFAULT_Q, DEFAULT_E_A,
    DEFAULT_A_PRE, DEFAULT_RHO_0, DEFAULT_P_0, DEFAULT_T_0,
    DEFAULT_W_MOL
)


class ReactiveState:
    r"""
    反应气体状态向量:
        U = [rho, rho*u, rho*v, E, rho*lambda]^T
    其中:
        rho     : 密度 [kg/m^3]
        u, v    : x, y 方向速度 [m/s]
        E       : 总比能 [J/m^3], E = rho*e + 0.5*rho*(u^2+v^2)
        lambda  : 反应进度变量 [0, 1]
    """

    def __init__(self, rho, u, v, e, lambda_var):
        check_positive(rho, "rho")
        check_nonnegative(lambda_var, "lambda")
        if lambda_var > 1.0:
            lambda_var = 1.0
        self.rho = float(rho)
        self.u = float(u)
        self.v = float(v)
        self.e = float(e)
        self.lambda_var = float(lambda_var)

    def to_conservative(self):
        r"""
        转换为守恒量向量:
            U1 = rho
            U2 = rho*u
            U3 = rho*v
            U4 = E = rho*e + 0.5*rho*(u^2+v^2)
            U5 = rho*lambda
        """
        U = np.zeros(5)
        U[0] = self.rho
        U[1] = self.rho * self.u
        U[2] = self.rho * self.v
        U[3] = self.rho * self.e + 0.5 * self.rho * (self.u ** 2 + self.v ** 2)
        U[4] = self.rho * self.lambda_var
        return U

    @classmethod
    def from_conservative(cls, U, gamma=DEFAULT_GAMMA, Q=DEFAULT_Q):
        r"""
        由守恒量恢复原始状态。
        """
        rho = U[0]
        if rho <= 0.0:
            rho = 1.0e-12
        u = U[1] / rho
        v = U[2] / rho
        E = U[3]
        e = E / rho - 0.5 * (u ** 2 + v ** 2)
        lambda_var = U[4] / rho
        lambda_var = max(0.0, min(1.0, lambda_var))
        return cls(rho, u, v, e, lambda_var)

    def pressure(self, gamma=DEFAULT_GAMMA, Q=DEFAULT_Q, W_mol=DEFAULT_W_MOL):
        r"""
        理想气体状态方程（反应流体）:
            p = (gamma - 1) * rho * (e - (1 - λ)*Q)
        注意: e 包含化学能，需要扣除化学势能项得到热能。
        """
        thermal_e = self.e - (1.0 - self.lambda_var) * Q
        p = (gamma - 1.0) * self.rho * thermal_e
        if p < 0.0:
            p = 1.0e-6
        return p

    def temperature(self, gamma=DEFAULT_GAMMA, Q=DEFAULT_Q, W_mol=DEFAULT_W_MOL):
        r"""
        由状态方程计算温度:
            T = p / (rho * R_specific),  R_specific = R / W_mol
        """
        p = self.pressure(gamma, Q, W_mol)
        R_specific = R_UNIVERSAL / W_mol
        T = p / (self.rho * R_specific)
        if T <= 0.0:
            T = 1.0e-6
        return T


def chemical_source_term(state, gamma=DEFAULT_GAMMA, Q=DEFAULT_Q,
                         A=DEFAULT_A_PRE, Ea=DEFAULT_E_A,
                         n_order=1.0, W_mol=DEFAULT_W_MOL):
    r"""
    化学反应源项:
        omega = [0, 0, 0, Q * rho * dλ/dt, rho * dλ/dt]^T
    其中 dλ/dt = -A * exp(-Ea/(R*T)) * (1-λ)^n
    返回 dU/dt 的源项向量（仅化学部分）。
    """
    # TODO: 构造化学反应源项向量 omega
    # 需要调用 znd_progress_variable_derivative 计算 dλ/dt
    # 注意 dλ/dt 为负值，omega[3] 为释热项，omega[4] 为反应进度源项
    raise NotImplementedError("Hole_2: 请实现化学反应源项 omega 的构造")


def euler_flux_x(state, gamma=DEFAULT_GAMMA, Q=DEFAULT_Q, W_mol=DEFAULT_W_MOL):
    r"""
    x 方向 Euler 通量:
        F = [rho*u, rho*u^2 + p, rho*u*v, (E+p)*u, rho*lambda*u]^T
    """
    p = state.pressure(gamma, Q, W_mol)
    F = np.zeros(5)
    F[0] = state.rho * state.u
    F[1] = state.rho * state.u ** 2 + p
    F[2] = state.rho * state.u * state.v
    F[3] = (state.rho * state.e + 0.5 * state.rho * (state.u ** 2 + state.v ** 2) + p) * state.u
    F[4] = state.rho * state.lambda_var * state.u
    return F


def euler_flux_y(state, gamma=DEFAULT_GAMMA, Q=DEFAULT_Q, W_mol=DEFAULT_W_MOL):
    r"""
    y 方向 Euler 通量:
        G = [rho*v, rho*u*v, rho*v^2 + p, (E+p)*v, rho*lambda*v]^T
    """
    p = state.pressure(gamma, Q, W_mol)
    G = np.zeros(5)
    G[0] = state.rho * state.v
    G[1] = state.rho * state.u * state.v
    G[2] = state.rho * state.v ** 2 + p
    G[3] = (state.rho * state.e + 0.5 * state.rho * (state.u ** 2 + state.v ** 2) + p) * state.v
    G[4] = state.rho * state.lambda_var * state.v
    return G


def reactive_euler_rhs(U, dx, dy, gamma=DEFAULT_GAMMA, Q=DEFAULT_Q,
                       A=DEFAULT_A_PRE, Ea=DEFAULT_E_A,
                       n_order=1.0, W_mol=DEFAULT_W_MOL):
    r"""
    2D 反应 Euler 方程的右端项（空间半离散）:
        dU/dt = -dF/dx - dG/dy + omega
    使用一阶迎风格量（用于概念验证，可升级至高阶）。
    输入 U: [nx, ny, 5] 的守恒量场。
    返回 dUdt: [nx, ny, 5] 的时间导数场。
    """
    nx, ny, nvar = U.shape
    if nvar != 5:
        raise ValueError("U must have shape [nx, ny, 5]")
    dUdt = np.zeros_like(U)

    for i in range(nx):
        for j in range(ny):
            state = ReactiveState.from_conservative(U[i, j], gamma, Q)
            omega = chemical_source_term(state, gamma, Q, A, Ea, n_order, W_mol)
            dUdt[i, j] += omega

    # x 方向空间离散（一阶迎风，内点）
    for i in range(1, nx):
        for j in range(ny):
            stateL = ReactiveState.from_conservative(U[i - 1, j], gamma, Q)
            stateR = ReactiveState.from_conservative(U[i, j], gamma, Q)
            FL = euler_flux_x(stateL, gamma, Q, W_mol)
            FR = euler_flux_x(stateR, gamma, Q, W_mol)
            # Lax-Friedrichs 型数值通量
            aL = sound_speed_from_prho(stateL.pressure(gamma, Q, W_mol), stateL.rho, gamma)
            aR = sound_speed_from_prho(stateR.pressure(gamma, Q, W_mol), stateR.rho, gamma)
            alpha = max(abs(stateL.u) + aL, abs(stateR.u) + aR, 1.0e-12)
            F_num = 0.5 * (FL + FR) - 0.5 * alpha * (U[i, j] - U[i - 1, j])
            if i > 0:
                dUdt[i - 1, j] -= F_num / dx
            if i < nx:
                dUdt[i, j] += F_num / dx

    # y 方向空间离散
    for i in range(nx):
        for j in range(1, ny):
            stateL = ReactiveState.from_conservative(U[i, j - 1], gamma, Q)
            stateR = ReactiveState.from_conservative(U[i, j], gamma, Q)
            GL = euler_flux_y(stateL, gamma, Q, W_mol)
            GR = euler_flux_y(stateR, gamma, Q, W_mol)
            aL = sound_speed_from_prho(stateL.pressure(gamma, Q, W_mol), stateL.rho, gamma)
            aR = sound_speed_from_prho(stateR.pressure(gamma, Q, W_mol), stateR.rho, gamma)
            alpha = max(abs(stateL.v) + aL, abs(stateR.v) + aR, 1.0e-12)
            G_num = 0.5 * (GL + GR) - 0.5 * alpha * (U[i, j] - U[i, j - 1])
            if j > 0:
                dUdt[i, j - 1] -= G_num / dy
            if j < ny:
                dUdt[i, j] += G_num / dy

    return dUdt
