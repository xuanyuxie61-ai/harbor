"""
chiral_dynamics.py
==================
手征有效场论中的非线性动力学：Goldstone 玻色子与手征对称性自发破缺。

原项目映射：
  - 1386_vanderpol_ode：Van der Pol 非线性振子（用于手征场的阻尼振荡）
  - 091_biochemical_nonlinear_ode：反应网络与化学计量矩阵（用于夸克-介子耦合系统）

物理背景
--------
在 QCD 中，手征对称性 SU(N_f)_L × SU(N_f)_R 自发破缺为对角子群 SU(N_f)_V，
产生 N_f^2 - 1 个近无质量的 Goldstone 玻色子（π, K, η）。
低能有效理论为手征微扰论（ChPT），其拉氏量为：

    L_2 = (f_0^2 / 4) Tr[ ∂_μ U † ∂^μ U ] + (f_0^2 B_0 / 2) Tr[ M (U + U†) ]

其中 U = exp( i Σ_a π_a τ_a / f_0 ) ∈ SU(2)，M = diag(m_u, m_d)。

Gell-Mann–Oakes–Renner (GMOR) 关系：

    m_π^2 f_π^2 = (m_u + m_d) B_0 f_0^2

对于大 N_c 极限，介子场的行为可用非线性振子描述。
Van der Pol 振子：

    d²u/dt² - μ (1 - u²) du/dt + u = 0

可类比为手征场在非平衡态下的阻尼振荡，其中 μ 控制非线性增益/耗散。

夸克-介子耦合系统可视为生化反应网络：

    q + q̄  ⇌  π  (产生/湮灭)
    π + π  →  ρ  (散射)

用化学计量矩阵 S 和速率向量 r 描述：
    dC/dt = S · r(C)
"""

import numpy as np
from scipy.integrate import solve_ivp


def chiral_lagrangian_deriv(t, y, mu_chiral: float = 2.0,
                            f0: float = 0.092, B0: float = 2.7e3,
                            mq: float = 0.0035):
    """
    手征场的 Van der Pol 型非线性动力学。

    将 π 场展开为 u(t) = π(t) / f_0，则运动方程为：

        du/dt = v
        dv/dt = μ (1 - u²) v - ω_0² u - γ v

    其中 ω_0² = m_π² = 2 B_0 m_q / f_0²，
    μ 为非线性系数，γ 为耗散。

    Parameters
    ----------
    t : float
        时间。
    y : np.ndarray
        状态向量 [u, v]。
    mu_chiral : float
        非线性系数（类比 Van der Pol μ）。
    f0 : float
        裸衰变常数（GeV）。
    B0 : float
        夸克凝聚参数（MeV）。
    mq : float
        平均夸克质量（GeV）。

    Returns
    -------
    dydt : np.ndarray
        时间导数。
    """
    u, v = y
    # GMOR 关系给出的 π 质量平方
    mpi_sq = 2.0 * B0 * mq / (f0 ** 2)
    omega0_sq = mpi_sq
    gamma = 0.5  # 耗散

    dudt = v
    dvdt = mu_chiral * (1.0 - u ** 2) * v - omega0_sq * u - gamma * v
    return np.array([dudt, dvdt])


def solve_chiral_oscillator(y0: np.ndarray, t_span: tuple,
                            mu_chiral: float = 2.0) -> tuple:
    """
    求解手征振子的时间演化。

    Returns
    -------
    t : np.ndarray
        时间网格。
    y : np.ndarray
        解轨迹。
    """
    sol = solve_ivp(
        lambda t, y: chiral_lagrangian_deriv(t, y, mu_chiral),
        t_span, y0, method='RK45', rtol=1e-9, atol=1e-12,
        dense_output=True
    )
    t = np.linspace(t_span[0], t_span[1], 500)
    y = sol.sol(t)
    return t, y


class QuarkMesonReactionNetwork:
    """
    夸克-介子耦合反应网络。

    物种：q（夸克）、q̄（反夸克）、π（π介子）、ρ（ρ介子）
    反应：
        R1: q + q̄ → π        (rmax1 * [q][q̄] / (Kq + [q]))
        R2: π → q + q̄        (e1 * [π])
        R3: π + π → ρ        (rmax2 * [π]² / (Kpi + [π]))
    """

    def __init__(self, a: float = 1.0, b: float = 0.5,
                 kc: float = 0.1, kn: float = 0.05,
                 rmax1: float = 1.0, rmax2: float = 0.3,
                 e1: float = 0.2, e2: float = 0.1):
        self.a = a
        self.b = b
        self.kc = kc
        self.kn = kn
        self.rmax1 = rmax1
        self.rmax2 = rmax2
        self.e1 = e1
        self.e2 = e2

        # 化学计量矩阵 S（4 物种 × 3 反应）
        self.S = np.array([
            [-a,  a,  0.0],   # q
            [-b,  b,  0.0],   # qbar
            [ 1.0, -1.0, -1.0],  # pi
            [ 0.0,  0.0,  1.0],  # rho
        ])

    def rates(self, conc: np.ndarray) -> np.ndarray:
        """
        计算反应速率向量 r。

        conc = [q, qbar, pi, rho]
        """
        q, qbar, pi, rho = conc
        q = max(q, 1e-12)
        qbar = max(qbar, 1e-12)
        pi = max(pi, 1e-12)

        r1 = self.rmax1 * q * qbar / (self.kc + q)
        r2 = self.e1 * pi
        r3 = self.rmax2 * pi ** 2 / (self.kn + pi)

        return np.array([r1, r2, r3])

    def deriv(self, t: float, conc: np.ndarray) -> np.ndarray:
        """
        dC/dt = S · r(C)
        """
        r = self.rates(conc)
        return self.S @ r

    def solve(self, c0: np.ndarray, t_span: tuple, n_points: int = 200) -> tuple:
        """
        积分反应网络到稳态。
        """
        sol = solve_ivp(
            self.deriv, t_span, c0, method='BDF',
            rtol=1e-8, atol=1e-10, dense_output=True
        )
        t = np.linspace(t_span[0], t_span[1], n_points)
        c = sol.sol(t)
        return t, c


def chiral_condensate_from_reaction(t: np.ndarray, c: np.ndarray,
                                    f0: float = 0.092) -> np.ndarray:
    """
    从反应网络浓度计算手征凝聚 ⟨q̄q⟩。

    近似：⟨q̄q⟩ ∝ (n_q + n_q̄) / 2 - n_π / f_0²
    """
    q = c[0, :]
    qbar = c[1, :]
    pi = c[2, :]
    sigma = 0.5 * (q + qbar) - pi / (f0 ** 2)
    return sigma


def pion_decay_constant_from_dynamics(mpi: float, B0: float = 2.7e3,
                                      mq: float = 0.0035) -> float:
    """
    由 GMOR 关系计算 π 介子衰变常数：

        f_π = sqrt( (m_u + m_d) B_0 / m_π² )

    Parameters
    ----------
    mpi : float
        π 介子质量（MeV）。
    B0 : float
        凝聚参数（MeV）。
    mq : float
        夸克质量（GeV → 转换为 MeV）。

    Returns
    -------
    fpi : float
        衰变常数（MeV）。
    """
    mq_mev = mq * 1e3
    fpi = np.sqrt((2.0 * mq_mev * B0) / (mpi ** 2 + 1e-10))
    return fpi
