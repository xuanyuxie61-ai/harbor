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
        U = np.zeros(5)
        U[0] = self.rho
        U[1] = self.rho * self.u
        U[2] = self.rho * self.v
        U[3] = self.rho * self.e + 0.5 * self.rho * (self.u ** 2 + self.v ** 2)
        U[4] = self.rho * self.lambda_var
        return U

    @classmethod
    def from_conservative(cls, U, gamma=DEFAULT_GAMMA, Q=DEFAULT_Q):
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
        thermal_e = self.e - (1.0 - self.lambda_var) * Q
        p = (gamma - 1.0) * self.rho * thermal_e
        if p < 0.0:
            p = 1.0e-6
        return p

    def temperature(self, gamma=DEFAULT_GAMMA, Q=DEFAULT_Q, W_mol=DEFAULT_W_MOL):
        p = self.pressure(gamma, Q, W_mol)
        R_specific = R_UNIVERSAL / W_mol
        T = p / (self.rho * R_specific)
        if T <= 0.0:
            T = 1.0e-6
        return T


def chemical_source_term(state, gamma=DEFAULT_GAMMA, Q=DEFAULT_Q,
                         A=DEFAULT_A_PRE, Ea=DEFAULT_E_A,
                         n_order=1.0, W_mol=DEFAULT_W_MOL):



    raise NotImplementedError("Hole_2: 请实现化学反应源项 omega 的构造")


def euler_flux_x(state, gamma=DEFAULT_GAMMA, Q=DEFAULT_Q, W_mol=DEFAULT_W_MOL):
    p = state.pressure(gamma, Q, W_mol)
    F = np.zeros(5)
    F[0] = state.rho * state.u
    F[1] = state.rho * state.u ** 2 + p
    F[2] = state.rho * state.u * state.v
    F[3] = (state.rho * state.e + 0.5 * state.rho * (state.u ** 2 + state.v ** 2) + p) * state.u
    F[4] = state.rho * state.lambda_var * state.u
    return F


def euler_flux_y(state, gamma=DEFAULT_GAMMA, Q=DEFAULT_Q, W_mol=DEFAULT_W_MOL):
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
    nx, ny, nvar = U.shape
    if nvar != 5:
        raise ValueError("U must have shape [nx, ny, 5]")
    dUdt = np.zeros_like(U)

    for i in range(nx):
        for j in range(ny):
            state = ReactiveState.from_conservative(U[i, j], gamma, Q)
            omega = chemical_source_term(state, gamma, Q, A, Ea, n_order, W_mol)
            dUdt[i, j] += omega


    for i in range(1, nx):
        for j in range(ny):
            stateL = ReactiveState.from_conservative(U[i - 1, j], gamma, Q)
            stateR = ReactiveState.from_conservative(U[i, j], gamma, Q)
            FL = euler_flux_x(stateL, gamma, Q, W_mol)
            FR = euler_flux_x(stateR, gamma, Q, W_mol)

            aL = sound_speed_from_prho(stateL.pressure(gamma, Q, W_mol), stateL.rho, gamma)
            aR = sound_speed_from_prho(stateR.pressure(gamma, Q, W_mol), stateR.rho, gamma)
            alpha = max(abs(stateL.u) + aL, abs(stateR.u) + aR, 1.0e-12)
            F_num = 0.5 * (FL + FR) - 0.5 * alpha * (U[i, j] - U[i - 1, j])
            if i > 0:
                dUdt[i - 1, j] -= F_num / dx
            if i < nx:
                dUdt[i, j] += F_num / dx


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
