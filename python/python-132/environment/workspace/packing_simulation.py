"""
packing_simulation.py
=====================
填料塔随机堆积模拟模块。

本模块基于线段随机填充/停车问题（源自项目 682_line_lines_packing），
模拟填料在塔截面内的随机堆积，计算填料效率因子与压降。

科学背景
--------
填料塔中，随机填料（如Raschig环、Pall环、规整填料）的堆积密度直接影响
传质效率与压降。将填料投影到一维轴向，可类比为线段随机填充问题：

在长度 L 的区间内随机放置长度为 l 的线段，Renyi 停车常数约为 0.7476，
表示最大随机堆积密度。

填料效率因子：
    η = η_0 * (1 - ε) / ε

其中 ε 为空隙率，η_0 为基准效率。

Ergun 方程用于估算压降：
    ΔP/L = 150 (1-ε)² μ u / (ε³ d_p²) + 1.75 (1-ε) ρ u² / (ε³ d_p)

其中：
    ε : 空隙率
    μ : 粘度 [Pa s]
    u : 表观气速 [m/s]
    d_p : 填料等效直径 [m]
    ρ : 气相密度 [kg/m³]
"""

import numpy as np
from utils import ensure_positive


# ---------------------------------------------------------------------------
# 线段随机填充（源自项目 682_line_lines_packing）
# ---------------------------------------------------------------------------

def line_packing_simulation(x_min, x_max, seg_width, max_attempts=100000):
    """
    在区间 [x_min, x_max] 内随机放置不重叠的等长线段。

    Parameters
    ----------
    x_min, x_max : float
        区间端点 [m]。
    seg_width : float
        线段长度 [m]。
    max_attempts : int
        最大尝试次数。

    Returns
    -------
    n_parked : int
        成功放置的线段数。
    density_obs : float
        观察到的堆积密度。
    density_max : float
        理论最大堆积密度。
    positions : ndarray
        线段中心位置。
    """
    seg_rad = seg_width / 2.0
    available_length = x_max - x_min

    if available_length <= seg_width:
        return 0, 0.0, 0.0, np.array([])

    positions = []
    n_parked = 0
    latest_success = 0

    for attempt in range(1, max_attempts + 1):
        r = np.random.rand()
        pos = x_min + seg_rad + r * (available_length - 2.0 * seg_rad)

        if len(positions) == 0:
            min_dist = 2.0 * seg_rad + 1e-12
        else:
            min_dist = np.min(np.abs(np.array(positions) - pos))

        if min_dist >= 2.0 * seg_rad:
            positions.append(pos)
            n_parked += 1
            latest_success = attempt
        elif attempt - latest_success > 10000:
            break

    positions = np.array(positions, dtype=float)
    density_max = 1.0 / (2.0 * seg_rad)
    density_obs = n_parked / available_length if available_length > 0 else 0.0

    return n_parked, density_obs, density_max, positions


# ---------------------------------------------------------------------------
# 填料塔堆积模型
# ---------------------------------------------------------------------------

def packing_void_fraction(n_packing, packing_diameter, column_diameter,
                          packing_height, packing_shape_factor=1.0):
    """
    计算填料塔空隙率。

    简化模型：
        ε = 1 - n_packing * V_packing / V_column
        V_packing ≈ π/4 * d_p² * h_p * φ_s

    Parameters
    ----------
    n_packing : int
        填料数量。
    packing_diameter : float
        填料直径 [m]。
    column_diameter : float
        塔径 [m]。
    packing_height : float
        填料层高度 [m]。
    packing_shape_factor : float
        形状因子。

    Returns
    -------
    epsilon : float
        空隙率。
    """
    V_column = np.pi / 4.0 * (column_diameter ** 2) * packing_height
    V_single = np.pi / 4.0 * (packing_diameter ** 2) * packing_diameter * packing_shape_factor
    V_packing_total = n_packing * V_single

    epsilon = 1.0 - V_packing_total / ensure_positive(V_column, name="V_column")
    epsilon = float(np.clip(epsilon, 0.2, 0.98))
    return epsilon


def packing_efficiency_factor(epsilon, epsilon_ref=0.9, eta_ref=1.0):
    """
    填料效率因子。

    η = η_ref * (1 - ε) / (1 - ε_ref) * f(ε)

    Parameters
    ----------
    epsilon : float
        空隙率。
    epsilon_ref : float
        参考空隙率。
    eta_ref : float
        参考效率。

    Returns
    -------
    eta : float
        效率因子。
    """
    epsilon = np.clip(epsilon, 0.2, 0.98)
    epsilon_ref = max(epsilon_ref, 0.2)
    eta = eta_ref * (1.0 - epsilon) / (1.0 - epsilon_ref)
    eta = eta * np.exp(-2.0 * (epsilon - 0.7) ** 2)
    return float(np.clip(eta, 0.1, 2.0))


def ergun_pressure_drop(epsilon, mu, u, rho, d_p, L_packing):
    """
    Ergun 方程计算填料层压降 [Pa]。

    ΔP/L = 150 (1-ε)² μ u / (ε³ d_p²) + 1.75 (1-ε) ρ u² / (ε³ d_p)

    Parameters
    ----------
    epsilon : float
        空隙率。
    mu : float
        粘度 [Pa s]。
    u : float
        表观气速 [m/s]。
    rho : float
        密度 [kg/m³]。
    d_p : float
        填料等效直径 [m]。
    L_packing : float
        填料层高度 [m]。

    Returns
    -------
    dP : float
        压降 [Pa]。
    """
    epsilon = max(epsilon, 0.2)
    mu = max(mu, 1e-6)
    u = max(u, 0.0)
    rho = max(rho, 0.01)
    d_p = max(d_p, 1e-6)
    L_packing = max(L_packing, 1e-6)

    term1 = 150.0 * ((1.0 - epsilon) ** 2) * mu * u / (epsilon ** 3 * d_p ** 2)
    term2 = 1.75 * (1.0 - epsilon) * rho * (u ** 2) / (epsilon ** 3 * d_p)

    dP = (term1 + term2) * L_packing
    return dP


def simulate_random_packing_column(column_diameter, packing_height,
                                    packing_diameter, packing_shape_factor,
                                    mu, u, rho, n_runs=10):
    """
    多次模拟随机堆积，统计平均空隙率与压降。

    Returns
    -------
    results : dict
        包含平均空隙率、效率因子、压降及统计信息。
    """
    epsilons = []
    etas = []
    dPs = []
    densities = []

    # 将填料投影到一维：塔截面周长方向
    circumference = np.pi * column_diameter

    for _ in range(n_runs):
        n_parked, density_obs, density_max, positions = line_packing_simulation(
            0.0, circumference, packing_diameter
        )
        densities.append(density_obs)

        # 估算填料数量（按周长方向密度推算体积）
        area_fraction = density_obs * packing_diameter
        n_packing_est = int(area_fraction * (column_diameter / packing_diameter) ** 2
                            * (packing_height / packing_diameter))
        n_packing_est = max(n_packing_est, 1)

        eps = packing_void_fraction(n_packing_est, packing_diameter,
                                     column_diameter, packing_height,
                                     packing_shape_factor)
        epsilons.append(eps)

        eta = packing_efficiency_factor(eps)
        etas.append(eta)

        dP = ergun_pressure_drop(eps, mu, u, rho, packing_diameter, packing_height)
        dPs.append(dP)

    results = {
        "epsilon_mean": float(np.mean(epsilons)),
        "epsilon_std": float(np.std(epsilons)),
        "eta_mean": float(np.mean(etas)),
        "eta_std": float(np.std(etas)),
        "dP_mean": float(np.mean(dPs)),
        "dP_std": float(np.std(dPs)),
        "density_mean": float(np.mean(densities)),
        "n_runs": n_runs
    }
    return results
