"""
reaction_rg.py
==============
耦合常数的重整化群（RG）流方程：从弱耦合到强耦合的连续演化。

原项目映射：091_biochemical_nonlinear_ode（化学计量矩阵方法用于 RG 流）

物理背景
--------
在格点 QCD 中，裸耦合 g_0 与重整化耦合 g(μ) 的关系由 β 函数决定：

    μ dg/dμ = β(g) = -β_0 g³ / (4π)² - β_1 g⁵ / (4π)⁴ - ...

其中对于 SU(3) 与 N_f 味夸克：

    β_0 = 11 - (2/3) N_f
    β_1 = 102 - (38/3) N_f

在格点上，常用的 RG 方案包括：
1. Schrödinger functional scheme
2. Wilson loop scheme
3. Lattice perturbation theory

本模块将 β 函数的连续流方程与格点上的离散跳跃结合，
模拟耦合常数随能标 μ 的演化。同时引入“反应网络”视角：
将不同能标区的耦合视为相互转化的“物种”，
用化学计量矩阵描述能标跨越时的耦合重分配。

核心公式
--------
Lattice β 函数（两圈近似）：

    a d g_0 / da = -g_0³ [ β_0 / (4π)² + β_1 g_0² / (4π)⁴ ]

或等价地，对 α_s = g_0² / (4π)：

    μ dα_s / dμ = -2 β_0 α_s² / (4π) - 2 β_1 α_s³ / (4π)²

跑动质量：

    μ d m_q / dμ = - γ_m α_s / π * m_q

其中 γ_m 为质量反常维度。
"""

import numpy as np
from scipy.integrate import solve_ivp


def beta_function_su3(g: float, nf: int = 2) -> float:
    """
    SU(3) 纯规范 + N_f 夸克的 β 函数（两圈）。

    β(g) = -β_0 g³ / (16π²) - β_1 g⁵ / (256π⁴)

    Parameters
    ----------
    g : float
        耦合常数。
    nf : int
        夸克味数。

    Returns
    -------
    beta : float
        β(g) 值。
    """
    beta0 = 11.0 - (2.0 / 3.0) * nf
    beta1 = 102.0 - (38.0 / 3.0) * nf
    beta = -beta0 * g ** 3 / (16.0 * np.pi ** 2)
    beta -= beta1 * g ** 5 / (256.0 * np.pi ** 4)
    return beta


def alpha_s_running(mu: np.ndarray, lambda_qcd: float = 0.3,
                    nf: int = 2) -> np.ndarray:
    """
    计算跑动耦合 α_s(μ)（单圈解析解）。

    α_s(μ) = 4π / [ β_0 log(μ² / Λ_QCD²) ]

    Parameters
    ----------
    mu : np.ndarray
        能标（GeV）。
    lambda_qcd : float
        Λ_QCD（GeV）。
    nf : int
        活跃夸克味数。

    Returns
    -------
    alpha : np.ndarray
        α_s 值。
    """
    beta0 = 11.0 - (2.0 / 3.0) * nf
    log_term = np.log((mu ** 2) / (lambda_qcd ** 2))
    log_term = np.where(log_term < 1e-3, 1e-3, log_term)
    alpha = 4.0 * np.pi / (beta0 * log_term)
    return alpha


def rg_flow_equations(t: float, y: np.ndarray, nf: int = 2) -> np.ndarray:
    """
    耦合常数与夸克质量的 RG 流方程组。

    y = [g, m_q]
    dy/dt = [ β(g), -γ_m(g) m_q ]

    其中 t = log(μ / μ_0)。

    Parameters
    ----------
    t : float
        log 能标。
    y : np.ndarray
        [g, m_q]。
    nf : int
        夸克味数。

    Returns
    -------
    dydt : np.ndarray
        流方程右端项。
    """
    g, mq = y
    g = max(g, 1e-6)
    mq = max(mq, 1e-9)

    dgdt = beta_function_su3(g, nf)

    # 质量反常维度（一圈）
    gamma_m = 12.0 / (16.0 * np.pi ** 2)
    dmqdt = -gamma_m * g ** 2 * mq

    return np.array([dgdt, dmqdt])


def solve_rg_flow(g0: float, mq0: float, t_span: tuple,
                  nf: int = 2, n_points: int = 200) -> tuple:
    """
    数值积分 RG 流方程。

    Parameters
    ----------
    g0 : float
        初始耦合（参考能标处）。
    mq0 : float
        初始夸克质量。
    t_span : tuple
        (log(μ_min/μ0), log(μ_max/μ0))。
    nf : int
        夸克味数。
    n_points : int
        输出点数。

    Returns
    -------
    t : np.ndarray
        Log 能标。
    y : np.ndarray
        解 [g(t), m_q(t)]。
    """
    sol = solve_ivp(
        lambda t, y: rg_flow_equations(t, y, nf),
        t_span, [g0, mq0], method='RK45',
        rtol=1e-9, atol=1e-12, dense_output=True
    )
    t = np.linspace(t_span[0], t_span[1], n_points)
    y = sol.sol(t)
    return t, y


def lattice_coupling_from_beta(beta_lat: float, nf: int = 2) -> float:
    """
    从格点 β = 6/g_0² 提取裸耦合 g_0。

    g_0 = sqrt(6 / β)
    """
    if beta_lat <= 0:
        raise ValueError("beta must be positive")
    return np.sqrt(6.0 / beta_lat)


def beta_from_lattice_spacing(a_fm: float, beta0: float = 11.0 - 4.0 / 3.0) -> float:
    """
    利用渐近标度关系估计格点 β：

    a Λ = R(β) exp( -4π² β / (11 N_c) )

    简化关系（SU(2) 纯规范）：
        a ≈ exp( -6π² β / 11 ) / Λ
    """
    lambda_inv_fm = 1.0 / 0.5  # Λ ≈ 0.5 fm^{-1} (粗略估计)
    a_target = a_fm
    # 反解 β
    beta_est = (11.0 / (6.0 * np.pi ** 2)) * np.log(1.0 / (a_target * lambda_inv_fm))
    return beta_est


def rg_step_matrix(n_scales: int = 5) -> np.ndarray:
    """
    构造“反应网络”视角的 RG 步进矩阵。

    将不同能标 μ_i 上的耦合 g_i 视为物种，
    RG 演化视为物种间的线性转化：

        dg_i/dt = Σ_j R_{ij} g_j

    其中 R 为近似的上三角矩阵（高能标耦合影响低能标）。
    """
    R = np.zeros((n_scales, n_scales))
    for i in range(n_scales):
        R[i, i] = -0.5  # 自衰减
        if i > 0:
            R[i, i - 1] = 0.3  # 高能标向低能标流动
    return R


def coupled_rg_reaction_network(g0_vec: np.ndarray, t_span: tuple,
                                n_points: int = 200) -> tuple:
    """
    多能标耦合的耦合 RG 反应网络。
    """
    n = len(g0_vec)
    R = rg_step_matrix(n)

    def deriv(t, g):
        return R @ g

    sol = solve_ivp(deriv, t_span, g0_vec, method='BDF',
                    rtol=1e-8, atol=1e-10, dense_output=True)
    t = np.linspace(t_span[0], t_span[1], n_points)
    g = sol.sol(t)
    return t, g
