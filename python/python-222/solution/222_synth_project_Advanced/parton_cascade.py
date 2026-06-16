# -*- coding: utf-8 -*-
"""
parton_cascade.py
=================

Parton shower 演化与守恒律 ODE 系统。

融合种子项目:
    - 208_conservation_ode: 守恒 ODE (pendulum, predator-prey, rigid-body)
    - 1124_rspence821505_Variational-Data-Consistent-Assimilation: 变分数据同化

物理模型:
    Parton shower 的演化可表述为 coupled ODE 系统:
        d f_i(x, t) / dt = sum_j int_x^1 dz/(z) P_{ij}(z) f_j(x/z, t)
                          - f_i(x, t) int dz P_{ji}(z)

    其中 t = ln(Q^2/mu^2) 为演化"时间", x 为动量分数。

    能量-动量守恒要求:
        sum_i int_0^1 dx x f_i(x, t) = const  (对任意 t)

    本模块将 shower 演化转化为守恒 ODE, 并引入 4D-Var 变分同化
    优化初始 parton 分布。
"""

from __future__ import annotations
import math
from typing import List, Tuple, Callable, Dict
import constants as C
from splitting_kernels import P_qq_lo, P_qg_lo, P_gq_lo, P_gg_lo, gauss_legendre


# ======================================================================
# Parton 分布向量
# ======================================================================
class PartonState:
    """
    离散化的 parton 分布状态:
        f = [f_g(x_1), ..., f_g(x_N), f_q(x_1), ..., f_q(x_N)]
    """
    def __init__(self, x_grid: List[float], fg: List[float], fq: List[float]):
        self.x = list(x_grid)
        self.fg = list(fg)
        self.fq = list(fq)
        self.n = len(x_grid)

    def copy(self) -> 'PartonState':
        return PartonState(list(self.x), list(self.fg), list(self.fq))

    def energy_momentum(self) -> float:
        """
        能量-动量求和规则:
            M_1 = sum_i int_0^1 dx x f_i(x)
        使用 Gauss-Legendre 求积。
        """
        n = 64
        nodes, weights = gauss_legendre(n)
        mom = 0.0
        for i in range(n):
            xi = nodes[i]
            wg = _interp(self.x, self.fg, xi)
            wq = _interp(self.x, self.fq, xi)
            mom += weights[i] * xi * (wg + wq)
        return mom

    def parton_number(self) -> float:
        """parton 数 (非守恒量, 随演化增长)"""
        n = 64
        nodes, weights = gauss_legendre(n)
        num = 0.0
        for i in range(n):
            xi = nodes[i]
            wg = _interp(self.x, self.fg, xi)
            wq = _interp(self.x, self.fq, xi)
            num += weights[i] * (wg + wq)
        return num


def _interp(x_grid: List[float], f: List[float], x: float) -> float:
    """线性插值, 外推为 0"""
    if x <= x_grid[0] or x >= x_grid[-1]:
        return 0.0
    for i in range(len(x_grid) - 1):
        if x_grid[i] <= x <= x_grid[i+1]:
            t = (x - x_grid[i]) / (x_grid[i+1] - x_grid[i])
            return f[i] + t * (f[i+1] - f[i])
    return 0.0


# ======================================================================
# Parton Shower 守恒 ODE (来自 208_conservation_ode 框架)
# ======================================================================
def shower_rhs(state: PartonState, alpha_s: float) -> Tuple[List[float], List[float]]:
    """
    计算 d f_i / dt 的右端项 (DGLAP 演化方程离散化)。

    守恒性类比 208_conservation_ode/pendulum_conserved:
        H = sum_i int dx x f_i(x) = const
    此守恒量由分裂核的 first moment 为零保证。
    """
    n = state.n
    dfg = [0.0] * n
    dfq = [0.0] * n
    nq = 32  # 积分点数
    nodes, weights = gauss_legendre(nq)

    for i, xi in enumerate(state.x):
        # 胶子演化: g -> gg + sum_q q\bar{q}
        sg = 0.0
        for k in range(nq):
            z = nodes[k]
            if z < C.Z_CUT or z > 1.0 - C.Z_CUT:
                continue
            x_over_z = xi / z
            if x_over_z > 1.0 or x_over_z < state.x[0]:
                continue
            gg = _interp(state.x, state.fg, x_over_z)
            qq = _interp(state.x, state.fq, x_over_z)
            sg += weights[k] * (P_gg_lo(z) * gg / z + P_gq_lo(z) * qq / z)

        # virtual 减去
        virtual_g = 0.0
        for k in range(nq):
            z = nodes[k]
            if z < C.Z_CUT or z > 1.0 - C.Z_CUT:
                continue
            virtual_g += weights[k] * P_gg_lo(z)
        sg -= state.fg[i] * virtual_g

        dfg[i] = alpha_s / (2.0 * C.PI) * sg

        # 夸克演化: q -> qg
        sq = 0.0
        for k in range(nq):
            z = nodes[k]
            if z < C.Z_CUT or z > 1.0 - C.Z_CUT:
                continue
            x_over_z = xi / z
            if x_over_z > 1.0 or x_over_z < state.x[0]:
                continue
            qq = _interp(state.x, state.fq, x_over_z)
            gg = _interp(state.x, state.fg, x_over_z)
            sq += weights[k] * (P_qq_lo(z) * qq / z + P_qg_lo(z) * gg / z)

        virtual_q = 0.0
        for k in range(nq):
            z = nodes[k]
            if z < C.Z_CUT or z > 1.0 - C.Z_CUT:
                continue
            virtual_q += weights[k] * P_qq_lo(z)
        sq -= state.fq[i] * virtual_q

        dfq[i] = alpha_s / (2.0 * C.PI) * sq

    return dfg, dfq


def shower_conserved_quantity(state: PartonState) -> float:
    """
    守恒量: 能量-动量求和规则。
    对应 208_conservation_ode 中 pendulum_conserved 的角色。
    """
    return state.energy_momentum()


# ======================================================================
# RK4 时间积分
# ======================================================================
def evolve_shower_rk4(state0: PartonState, dt: float, n_steps: int,
                       alpha_s_func: Callable[[float], float] = None) -> List[PartonState]:
    """
    四阶 Runge-Kutta 演化 shower ODE。
    返回轨迹 [state_0, state_1, ..., state_{n_steps}]。
    """
    if alpha_s_func is None:
        alpha_s_func = lambda t: C.safe_alpha_s(2.0 * math.exp(-t / 2.0))

    trajectory = [state0.copy()]
    state = state0.copy()
    t = 0.0
    m0 = state0.energy_momentum()  # 初始动量矩 (守恒目标)

    for step in range(n_steps):
        als = alpha_s_func(t)

        # RK4
        k1_fg, k1_fq = shower_rhs(state, als)
        s2 = _add_scaled(state, k1_fg, k1_fq, 0.5 * dt)
        k2_fg, k2_fq = shower_rhs(s2, als)
        s3 = _add_scaled(state, k2_fg, k2_fq, 0.5 * dt)
        k3_fg, k3_fq = shower_rhs(s3, als)
        s4 = _add_scaled(state, k3_fg, k3_fq, dt)
        k4_fg, k4_fq = shower_rhs(s4, als)

        for i in range(state.n):
            state.fg[i] += (dt / 6.0) * (k1_fg[i] + 2*k2_fg[i] + 2*k3_fg[i] + k4_fg[i])
            state.fq[i] += (dt / 6.0) * (k1_fq[i] + 2*k2_fq[i] + 2*k3_fq[i] + k4_fq[i])
            # 非负截断
            state.fg[i] = max(0.0, state.fg[i])
            state.fq[i] = max(0.0, state.fq[i])

        # 强制动量守恒 (重整化)
        _enforce_momentum_conservation(state, m0)

        t += dt
        trajectory.append(state.copy())

    return trajectory


def _enforce_momentum_conservation(state: 'PartonState', target_momentum: float) -> None:
    """
    重整化 parton 分布以强制动量守恒。
    这是 parton shower MC 中的标准步骤。
    """
    current = state.energy_momentum()
    if current <= 1e-15 or target_momentum <= 1e-15:
        return
    scale = target_momentum / current
    for i in range(state.n):
        state.fg[i] *= scale
        state.fq[i] *= scale


def _add_scaled(s: PartonState, dfg: List[float], dfq: List[float],
                eps: float) -> PartonState:
    """辅助: s + eps * df"""
    new_fg = [s.fg[i] + eps * dfg[i] for i in range(s.n)]
    new_fq = [s.fq[i] + eps * dfq[i] for i in range(s.n)]
    for i in range(s.n):
        new_fg[i] = max(0.0, new_fg[i])
        new_fq[i] = max(0.0, new_fq[i])
    return PartonState(s.x, new_fg, new_fq)


# ======================================================================
# 变分数据同化 (来自 1124_Variational-Data-Consistent-Assimilation)
# ======================================================================
def background_cost(state: PartonState, background: PartonState,
                    B_inv_diag: List[float]) -> float:
    """
    背景项 J_b:
        J_b = 0.5 * (f - f_b)^T B^{-1} (f - f_b)

    对应 1124/cost_functions.py 中的 background_loss。
    """
    j = 0.0
    for i in range(state.n):
        dg = state.fg[i] - background.fg[i]
        dq = state.fq[i] - background.fq[i]
        j += 0.5 * B_inv_diag[i] * (dg * dg + dq * dq)
    return j


def observation_cost(state: PartonState, obs_x: List[float],
                     obs_val: List[float], R_inv: float) -> float:
    """
    观测项 J_o:
        J_o = 0.5 * sum_k (1/R) * (f(x_k) - y_k)^2

    对应 1124/cost_functions.py 中的 observation_loss。
    """
    j = 0.0
    for k, xk in enumerate(obs_x):
        fg = _interp(state.x, state.fg, xk)
        fq = _interp(state.x, state.fq, xk)
        diff = (fg + fq) - obs_val[k]
        j += 0.5 * R_inv * diff * diff
    return j


def total_cost(state: PartonState, background: PartonState,
               B_inv_diag: List[float],
               obs_x: List[float], obs_val: List[float],
               R_inv: float, gamma: float = 1.0) -> float:
    """
    总代价函数:
        J = J_b + gamma * J_o
    """
    return background_cost(state, background, B_inv_diag) + gamma * observation_cost(state, obs_x, obs_val, R_inv)


def variational_assimilation_step(state0: PartonState,
                                   background: PartonState,
                                   obs_x: List[float], obs_val: List[float],
                                   B_inv_diag: List[float], R_inv: float,
                                   dt: float, n_steps: int,
                                   learning_rate: float = 0.01) -> PartonState:
    """
    简化版 4D-Var: 梯度下降优化初始条件,
    使演化末端状态与观测数据的代价最小。
    """
    best = state0.copy()
    best_cost = float('inf')

    for iteration in range(50):
        traj = evolve_shower_rk4(state0, dt, n_steps)
        final = traj[-1]
        cost = total_cost(final, background, B_inv_diag, obs_x, obs_val, R_inv)

        if cost < best_cost:
            best_cost = cost
            best = state0.copy()

        # 简单有限差分梯度
        for i in range(state0.n):
            eps = 1e-4
            state0_p = state0.copy()
            state0_p.fg[i] += eps
            traj_p = evolve_shower_rk4(state0_p, dt, min(n_steps, 3))
            cost_p = total_cost(traj_p[-1], background, B_inv_diag, obs_x, obs_val, R_inv)

            grad_g = (cost_p - cost) / eps
            state0.fg[i] -= learning_rate * grad_g
            state0.fg[i] = max(0.0, state0.fg[i])

    return best


# ======================================================================
# 初始 parton 分布构造
# ======================================================================
def initial_pdf_valence(x: float, Q2: float = 10.0) -> float:
    """
    简化的价夸克分布 (CTEQ 型参数化):
        x f_v(x) ~ A x^a (1-x)^b
    满足 baryon number 求和规则 int_0^1 f_v dx = 1。
    """
    A = 3.0  # 归一化 (两个价夸克: u + d = 3)
    a = 0.5
    b = 3.0
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return A * x ** (a - 1.0) * (1.0 - x) ** b


def initial_pdf_gluon(x: float, Q2: float = 10.0) -> float:
    """
    简化胶子分布:
        x g(x) ~ A_g x^{-lambda} (1-x)^n
    """
    A_g = 5.0
    lam = 0.3
    n = 5.0
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return A_g * x ** (-lam - 1.0) * (1.0 - x) ** n


def make_initial_state(nx: int = 16, Q2: float = 10.0) -> PartonState:
    """
    构造初始 parton 分布状态 (均匀 log-x 网格)。
    """
    # 对数均匀网格: x in [1e-3, 1 - 1e-3]
    x_grid = []
    for i in range(nx):
        xi = 1e-3 + (1.0 - 2e-3) * i / (nx - 1)
        x_grid.append(xi)

    fg = [initial_pdf_gluon(xi, Q2) for xi in x_grid]
    fq = [initial_pdf_valence(xi, Q2) for xi in x_grid]
    return PartonState(x_grid, fg, fq)


# ======================================================================
# 自测
# ======================================================================
if __name__ == "__main__":
    print("=== Parton Shower Conservation Test ===")
    s0 = make_initial_state(12, 10.0)
    m0 = shower_conserved_quantity(s0)
    print(f"  Initial momentum sum: {m0:.6f}")

    traj = evolve_shower_rk4(s0, 0.05, 5)
    m_final = shower_conserved_quantity(traj[-1])
    print(f"  Final   momentum sum: {m_final:.6f}")
    print(f"  Conservation violation: {abs(m_final - m0):.2e}")

    for k, s in enumerate(traj):
        n_partons = s.parton_number()
        print(f"  Step {k}: <x> = {shower_conserved_quantity(s):.6f}, "
              f"N = {n_partons:.4f}")
