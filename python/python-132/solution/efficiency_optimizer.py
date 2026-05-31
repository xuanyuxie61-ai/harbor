"""
efficiency_optimizer.py
=======================
精馏塔能效优化模块。

本模块融合费马因数分解思想（源自项目 420_fermat_factor），
用于理论塔板数与回流比的耦合优化。

科学背景
--------
精馏塔的总成本由操作成本（能耗）与资本成本（设备）组成：

    C_total = C_op + C_cap

操作成本与再沸器热负荷 Q_R 成正比：
    C_op = c_steam * Q_R * t_operation

资本成本与塔板数 N 和塔径 D 相关：
    C_cap = a + b * N^0.8 * D^1.5

回流比 R 与塔板数 N 满足 Gilliland 关联：
    Y = 1 - exp[ (1 + 54.4 X) (X - 1) / (11 + 117.2 X) * sqrt(X) ]

其中：
    X = (R - R_min) / (R + 1)
    Y = (N - N_min) / (N + 1)

费马因数分解思想在此转化为：将总成本函数视为一个"大整数"，
寻找最优的 (N, R) 组合，使得它们"接近"（即成本最小），
类似于寻找 N = A² - B² 中接近的因数。

具体映射：
    设总成本函数 C(N, R) 为待分解的"数"，
    我们寻找整数 N 和实数 R，使得 C(N,R) 最小。
    使用迭代搜索，从 sqrt(C) 附近开始，逐步调整 N 和 R。
"""

import numpy as np
from utils import ensure_positive, clip_with_warning


# ---------------------------------------------------------------------------
# 费马分解思想用于塔板数优化（源自项目 420_fermat_factor）
# ---------------------------------------------------------------------------

def fermat_optimize_trays(N_min, R_min, target_cost_func, N_max=200, tol=1e-4):
    """
    使用类费马分解思想优化理论塔板数 N 与回流比 R。

    核心思想：在 (N, R) 平面上，从接近 sqrt(target) 的区域开始搜索，
    寻找使总成本最小的组合，类似于费马分解从 sqrt(n) 开始寻找因数。

    Parameters
    ----------
    N_min : int
        最小理论塔板数。
    R_min : float
        最小回流比。
    target_cost_func : callable
        总成本函数 C(N, R)。
    N_max : int
        最大搜索塔板数。
    tol : float
        收敛容差。

    Returns
    -------
    N_opt : int
        最优塔板数。
    R_opt : float
        最优回流比。
    C_min : float
        最小成本。
    history : list
        搜索历史。
    """
    N_min = max(int(N_min), 2)
    N_max = max(int(N_max), N_min + 1)
    R_min = max(float(R_min), 1.01)

    # 从 sqrt(N_max * R_range) 附近开始搜索
    N_start = int(np.sqrt(N_max * N_min))
    N_start = clip_with_warning(N_start, N_min, N_max, "N_start")

    C_min = float('inf')
    N_opt = N_min
    R_opt = R_min
    history = []

    for N in range(N_min, N_max + 1):
        # 对于每个 N，搜索最优 R
        # 使用类费马步长：从接近 sqrt(N) 的 R 开始
        R_start = max(R_min, np.sqrt(N) * 0.1)
        R_end = max(R_start + 5.0, R_min * 10.0)
        R_grid = np.linspace(R_start, R_end, 50)

        for R in R_grid:
            C = target_cost_func(N, R)
            history.append((N, R, C))
            if C < C_min:
                C_min = C
                N_opt = N
                R_opt = R

    return N_opt, R_opt, C_min, history


# ---------------------------------------------------------------------------
# Gilliland 关联
# ---------------------------------------------------------------------------

def gilliland_correlation(R, R_min, N, N_min):
    """
    Gilliland 关联用于估算理论塔板数与回流比的关系。

    X = (R - R_min) / (R + 1)
    Y = (N - N_min) / (N + 1)

    Y = 1 - exp[ (1 + 54.4 X)(X - 1) / (11 + 117.2 X) * sqrt(X) ]

    Parameters
    ----------
    R : float
        实际回流比。
    R_min : float
        最小回流比。
    N : int
        实际理论塔板数。
    N_min : int
        最小理论塔板数。

    Returns
    -------
    residual : float
        Gilliland 残差（越接近0越一致）。
    """
    R = max(R, 1.001)
    R_min = max(R_min, 1.0001)
    N = max(int(N), 2)
    N_min = max(int(N_min), 1)

    X = (R - R_min) / (R + 1.0)
    X = max(X, 1e-6)

    Y_calc = 1.0 - np.exp(
        (1.0 + 54.4 * X) * (X - 1.0) / (11.0 + 117.2 * X) * np.sqrt(X)
    )
    Y_actual = (N - N_min) / (N + 1.0)

    return Y_actual - Y_calc


def estimate_N_from_R(R, R_min, N_min):
    """
    由回流比 R 估算所需理论塔板数 N（Gilliland 反解）。
    """
    R = max(R, 1.001)
    R_min = max(R_min, 1.0001)
    N_min = max(int(N_min), 1)

    X = (R - R_min) / (R + 1.0)
    X = max(X, 1e-6)

    Y = 1.0 - np.exp(
        (1.0 + 54.4 * X) * (X - 1.0) / (11.0 + 117.2 * X) * np.sqrt(X)
    )

    N = int(np.ceil((Y * (N_min + 1.0) + N_min) / (1.0 - Y)))
    return max(N, N_min + 1)


# ---------------------------------------------------------------------------
# 成本模型
# ---------------------------------------------------------------------------

def reboiler_duty(R, D, q_cond, lambda_vap, feed_rate, z_F, x_D, x_B):
    """
    计算再沸器热负荷 [W]。

    简化模型：
        V = (R + 1) D
        Q_R = V * λ_vap + q_cond

    Parameters
    ----------
    R : float
        回流比。
    D : float
        馏出液流量 [mol/s]。
    q_cond : float
        冷凝器热负荷 [W]。
    lambda_vap : float
        汽化潜热 [J/mol]。
    feed_rate : float
        进料流量 [mol/s]。
    z_F, x_D, x_B : float
        进料、馏出液、釜液轻组分摩尔分数。

    Returns
    -------
    Q_R : float
        再沸器热负荷 [W]。
    """
    R = max(R, 0.0)
    D = max(D, 1e-12)
    V = (R + 1.0) * D
    Q_R = V * lambda_vap + q_cond
    return Q_R


def total_cost_model(N, R, D, q_cond, lambda_vap, feed_rate,
                     z_F, x_D, x_B, c_steam, t_op, a_cap, b_cap, column_diameter):
    """
    计算年度总成本 [CNY/year]。

    C_total = C_op + C_cap
    C_op = c_steam * Q_R * t_op / (3.6e6)  [kg steam/s -> CNY]
    C_cap = (a_cap + b_cap * N**0.8 * column_diameter**1.5) / t_op * amortization
    """
    Q_R = reboiler_duty(R, D, q_cond, lambda_vap, feed_rate, z_F, x_D, x_B)

    # 操作成本 [CNY/year]，假设蒸汽焓值 2.8 MJ/kg
    C_op = c_steam * (Q_R / 2.8e6) * t_op

    # 资本成本 [CNY/year]，假设10年摊销
    C_cap = (a_cap + b_cap * (N ** 0.8) * (column_diameter ** 1.5)) / 10.0

    return C_op + C_cap


def optimize_distillation_cost(N_min, R_min, D, q_cond, lambda_vap,
                                feed_rate, z_F, x_D, x_B,
                                c_steam, t_op, a_cap, b_cap, column_diameter):
    """
    优化精馏塔年度总成本，返回最优 N 和 R。
    """
    def cost_func(N, R):
        return total_cost_model(
            int(N), R, D, q_cond, lambda_vap, feed_rate,
            z_F, x_D, x_B, c_steam, t_op, a_cap, b_cap, column_diameter
        )

    N_opt, R_opt, C_min, history = fermat_optimize_trays(
        N_min, R_min, cost_func, N_max=max(N_min + 100, 200)
    )

    return N_opt, R_opt, C_min, history
