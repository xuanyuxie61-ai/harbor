"""
dynamics_evolution.py

基于 md_fast (745_md_fast)、reaction_twoway_ode (1018_reaction_twoway_ode)、
sawtooth_ode (1059_sawtooth_ode) 的实时动力学演化模块。

本模块模拟强关联电子系统在周期驱动下的非平衡动力学:
1. 周期驱动 Hubbard 模型: H(t) = H_0 + V(t)
   V(t) = A sin(Ω t) Σ_i (-1)^i n_i   (交变电场)
2. 或 sawtooth 型驱动: V(t) = A · sawtooth(Ω t) Σ_i r_i · n_i
3. 使用 Velocity-Verlet 型积分传播密度矩阵

同时包含两能级反应动力学模型，描述 doublon-holon 对的产生-湮灭过程。
"""

import numpy as np
from scipy.linalg import expm
from typing import Callable, Tuple


# ---------------------------------------------------------------------------
# md_fast: Velocity-Verlet 型密度矩阵传播
# ---------------------------------------------------------------------------

def velocity_verlet_propagator(H0: np.ndarray, dt: float, mass: float = 1.0) -> np.ndarray:
    """
    将 Velocity-Verlet 思想映射到量子演化:
        U(dt) = exp(-i H0 dt) ≈ exp(-i H0 dt/2) · exp(-i H0 dt/2)
    
    这里采用 Trotter 分解实现高阶传播子。
    """
    if H0.shape[0] != H0.shape[1]:
        raise ValueError("H0 必须是方阵")
    if dt <= 0:
        raise ValueError("dt > 0 required")
    # 二阶 Trotter
    return expm(-1j * H0 * dt)


def driven_hubbard_evolution(H0: np.ndarray, drive_func: Callable, times: np.ndarray,
                              nsites: int) -> np.ndarray:
    """
    周期驱动 Hubbard 模型的实时演化。
    
    H(t) = H_0 + V(t)
    在每个时间步使用短时分段常数近似:
        U(t_{n+1}, t_n) ≈ exp(-i H(t_n) Δt)
    
    参数:
        H0: 时无关部分哈密顿量
        drive_func: V(t) 的矩阵值函数
        times: 时间网格
        nsites: 格点数
    
    返回:
        rho_t: 密度矩阵随时间演化，形状 (nt, dim, dim)
    """
    dim = H0.shape[0]
    nt = len(times)
    if nt < 2:
        raise ValueError("times 长度必须 >= 2")
    rho = np.zeros((nt, dim, dim), dtype=np.complex128)
    # 初始态: 基态 (T=0) 或热态
    evals, evecs = np.linalg.eigh(H0)
    gs = evecs[:, 0]
    rho[0] = np.outer(gs, gs.conj())
    for n in range(nt - 1):
        dt = times[n + 1] - times[n]
        t_mid = times[n] + dt * 0.5
        Vt = drive_func(t_mid)
        Ht = H0 + Vt
        U = expm(-1j * Ht * dt)
        rho[n + 1] = U @ rho[n] @ U.conj().T
    return rho


def sawtooth_wave(t: float, omega: float) -> float:
    """
    周期为 T = 2π/ω 的标准 sawtooth 波:
        f(t) = (t mod T) / T - 0.5
    返回值范围 [-0.5, 0.5)。
    """
    if omega <= 0:
        raise ValueError("omega > 0 required")
    T = 2.0 * np.pi / omega
    val = (t % T) / T - 0.5
    return val


def sawtooth_drive_matrix(nsites: int, amplitude: float, omega: float, times: np.ndarray) -> list:
    """
    构造 sawtooth 驱动势能矩阵序列:
        V_{ij}(t) = A · sawtooth(ω t) · (-1)^i · δ_{ij}
    """
    V_list = []
    for t in times:
        V = np.diag([amplitude * sawtooth_wave(t, omega) * ((-1) ** i) for i in range(nsites)])
        V_list.append(V)
    return V_list


# ---------------------------------------------------------------------------
# reaction_twoway_ode: 两能级反应动力学
# ---------------------------------------------------------------------------

def reaction_twoway_ode_rhs(y: np.ndarray, k1: float, k2: float) -> np.ndarray:
    """
    两能级反应动力学:
        dw1/dt = -k1 w1 + k2 w2
        dw2/dt = +k1 w1 - k2 w2
    
    物理意义: w1 = doublon 密度, w2 = holon 密度
    k1 = doublon 衰变速率, k2 = doublon-holon 对产生速率。
    """
    w1, w2 = y
    if w1 < 0 or w2 < 0:
        # 边界保护
        w1 = max(w1, 0.0)
        w2 = max(w2, 0.0)
    dw1dt = -k1 * w1 + k2 * w2
    dw2dt = +k1 * w1 - k2 * w2
    return np.array([dw1dt, dw2dt])


def solve_reaction_twoway(k1: float, k2: float, w0: np.ndarray, t_span: Tuple[float, float],
                           nt: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    用 RK4 求解两能级反应 ODE。
    """
    if k1 < 0 or k2 < 0:
        raise ValueError("k1, k2 >= 0 required")
    if nt < 2:
        raise ValueError("nt >= 2 required")
    t = np.linspace(t_span[0], t_span[1], nt)
    dt = t[1] - t[0]
    y = np.zeros((nt, 2))
    y[0] = w0
    for i in range(nt - 1):
        k1_rk = reaction_twoway_ode_rhs(y[i], k1, k2)
        k2_rk = reaction_twoway_ode_rhs(y[i] + 0.5 * dt * k1_rk, k1, k2)
        k3_rk = reaction_twoway_ode_rhs(y[i] + 0.5 * dt * k2_rk, k1, k2)
        k4_rk = reaction_twoway_ode_rhs(y[i] + dt * k3_rk, k1, k2)
        y[i + 1] = y[i] + (dt / 6.0) * (k1_rk + 2 * k2_rk + 2 * k3_rk + k4_rk)
        # 非负约束
        y[i + 1] = np.maximum(y[i + 1], 0.0)
    return t, y


def doublon_dynamics_hubbard(U: float, t_hop: float, beta: float, t_max: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    模拟 Hubbard 模型中的 doublon 动力学:
        - k1 ∝ t_hop^2 / U   ( doublon 隧穿衰减 )
        - k2 ∝ exp(-β U)     ( 热激发产生 doublon-holon 对 )
    """
    if U <= 0:
        raise ValueError("U > 0 required")
    k1 = t_hop ** 2 / U
    k2 = np.exp(-beta * U) * t_hop
    w0 = np.array([0.1, 0.1])
    return solve_reaction_twoway(k1, k2, w0, (0.0, t_max), nt=500)


def energy_evolution(rho_t: np.ndarray, H0: np.ndarray) -> np.ndarray:
    """计算密度矩阵随时间的能量期望值 Tr[ρ(t) H0]。"""
    nt = len(rho_t)
    E = np.zeros(nt)
    for i in range(nt):
        E[i] = float(np.trace(rho_t[i] @ H0).real)
    return E


if __name__ == "__main__":
    t, y = doublon_dynamics_hubbard(U=4.0, t_hop=1.0, beta=2.0, t_max=10.0)
    print(f"Doublon final: w1={y[-1,0]:.4f}, w2={y[-1,1]:.4f}")
