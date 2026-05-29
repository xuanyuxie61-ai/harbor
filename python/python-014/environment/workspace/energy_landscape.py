"""
energy_landscape.py
===================
自旋玻璃能量景观的局部极小搜索与元稳定性分析模块。
融合来源：local_min（Brent 方法：黄金分割 + 抛物线插值）。

物理背景：
    自旋玻璃的能量景观具有大量局部极小（meta-stable states），
    寻找基态是 NP-hard 问题。本模块提供一维剖面上的精确线搜索，
    结合蒙特卡洛退火框架进行全局优化。

核心公式：
    对于给定的自旋构型 {S_i}，沿旋转方向 n̂_i 旋转角度 θ 的能量变化：
        E(θ) = E_0 + Σ_i H_i · (R_i(θ) S_i - S_i) + (1/2) Σ_{ij} J_{ij} (R_i S_i)·(R_j S_j) - ...
    单自旋翻转近似下：
        ΔE_i = 2 S_i · H_i，其中 H_i = Σ_j J_{ij} S_j 为有效场。
"""

import numpy as np
from typing import Tuple, Callable
from utils import EPS_MACHINE, EPS_SQRT
from spin_quaternion import axis_angle_to_q, q_rotate_vector


def local_min_brent(
    f: Callable[[float], float],
    a: float,
    b: float,
    epsi: float = EPS_SQRT,
    t: float = 1e-10,
    max_calls: int = 500,
) -> Tuple[float, float, int]:
    """
    Brent 方法求一元函数在区间 [a, b] 上的局部极小值。
    融合来源：local_min（黄金分割 + 连续抛物线插值）。

    算法保证：若 f 为 δ-单峰且 δ < tol，则误差 < 3*tol，
    其中 tol = epsi * |x| + t。

    收敛阶：对于二阶连续可微函数，通常超线性收敛，阶约 1.3247。

    参数
    ----
    f : callable
        目标函数。
    a, b : float
        搜索区间端点。
    epsi : float
        相对误差容限。
    t : float
        绝对误差容限。
    max_calls : int
        最大函数求值次数。

    返回
    ----
    x : float
        极小点估计。
    fx : float
        极小值。
    calls : int
        实际函数调用次数。
    """
    c_ratio = 0.5 * (3.0 - np.sqrt(5.0))  # 黄金分割比例的平方，≈0.381966
    sa, sb = a, b
    x = sa + c_ratio * (b - a)
    w, v = x, x
    e = 0.0
    fx = f(x)
    calls = 1
    fw, fv = fx, fx

    while calls < max_calls:
        m = 0.5 * (sa + sb)
        tol = epsi * abs(x) + t
        t2 = 2.0 * tol
        if abs(x - m) <= t2 - 0.5 * (sb - sa):
            break

        r_val = 0.0
        q_val = 0.0
        p_val = 0.0
        if tol < abs(e):
            r_val = (x - w) * (fx - fv)
            q_val = (x - v) * (fx - fw)
            p_val = (x - v) * q_val - (x - w) * r_val
            q_val = 2.0 * (q_val - r_val)
            if 0.0 < q_val:
                p_val = -p_val
            q_val = abs(q_val)
            r_val = e
            e = d  # type: ignore # d will be defined below on first use

        # 首次迭代时 d 未定义，需特殊处理
        if calls == 1:
            if x < m:
                e = sb - x
            else:
                e = sa - x
            d = c_ratio * e
        else:
            if (
                abs(p_val) < abs(0.5 * q_val * r_val)
                and q_val * (sa - x) < p_val < q_val * (sb - x)
            ):
                # 抛物线步
                d = p_val / q_val
                u = x + d
                if (u - sa) < t2 or (sb - u) < t2:
                    if x < m:
                        d = tol
                    else:
                        d = -tol
            else:
                # 黄金分割步
                if x < m:
                    e = sb - x
                else:
                    e = sa - x
                d = c_ratio * e

        if tol <= abs(d):
            u = x + d
        elif 0.0 < d:
            u = x + tol
        else:
            u = x - tol

        fu = f(u)
        calls += 1

        if fu <= fx:
            if u < x:
                sb = x
            else:
                sa = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u < x:
                sa = u
            else:
                sb = u
            if fu <= fw or abs(w - x) < EPS_MACHINE:
                v, w = w, u
                fv, fw = fw, fu
            elif fu <= fv or abs(v - x) < EPS_MACHINE or abs(v - w) < EPS_MACHINE:
                v = u
                fv = fu

    return x, fx, calls


def line_search_spin_rotation(
    J: np.ndarray,
    spins: np.ndarray,
    site_idx: int,
    axis: np.ndarray,
    a: float = -np.pi,
    b: float = np.pi,
) -> Tuple[float, np.ndarray, float]:
    """
    对单个自旋绕给定轴进行一维线搜索，寻找能量极小角度。

    参数
    ----
    J : np.ndarray
        交换耦合矩阵。
    spins : np.ndarray, shape (N, 3)
        当前自旋构型。
    site_idx : int
        要旋转的格点索引。
    axis : np.ndarray
        旋转轴（三维矢量，自动归一化）。
    a, b : float
        角度搜索区间。

    返回
    ----
    theta_opt : float
        最优旋转角。
    spin_new : np.ndarray
        旋转后的自旋矢量。
    e_min : float
        能量极小值。
    """
    axis = np.array(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < EPS_MACHINE:
        axis = np.array([0.0, 0.0, 1.0])
    else:
        axis = axis / norm

    N = spins.shape[0]

    def energy_at_angle(theta: float) -> float:
        q = axis_angle_to_q(axis, theta)
        s_rot = q_rotate_vector(q, spins[site_idx])
        new_spins = spins.copy()
        new_spins[site_idx] = s_rot
        # TODO: Hole_2 — 计算旋转后构型的总能量（交换能 + 单轴各向异性能）
        # 交换能公式：E_ex = 0.5 * Σ_{i,j} J_{ij} S_i · S_j
        # 各向异性能公式：E_anis = K * Σ_i S_{i,z}^2，此处 K = 0.05
        raise NotImplementedError("Hole_2: 请实现 energy_at_angle 中的能量计算")

    theta_opt, e_min, _ = local_min_brent(energy_at_angle, a, b)
    q_opt = axis_angle_to_q(axis, theta_opt)
    spin_new = q_rotate_vector(q_opt, spins[site_idx])
    return theta_opt, spin_new, e_min


def greedy_relaxation(
    J: np.ndarray,
    spins: np.ndarray,
    n_sweeps: int = 10,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, float, list]:
    """
    贪心松弛：逐格点进行线搜索能量最小化，模拟零温淬火过程。

    返回
    ----
    spins_final : np.ndarray
        松弛后的构型。
    e_final : float
        最终能量。
    history : list
        每轮能量列表。
    """
    N = spins.shape[0]
    spins = spins.copy()
    history = []
    e_old = float("inf")
    for sweep in range(n_sweeps):
        for i in range(N):
            # 有效场方向作为旋转轴的候选
            H_i = J[i, :] @ spins
            H_norm = np.linalg.norm(H_i)
            if H_norm < EPS_MACHINE:
                continue
            axis = H_i / H_norm
            _, spins[i], _ = line_search_spin_rotation(J, spins, i, axis)
        # 计算总能量
        H = J @ spins
        e_new = 0.5 * np.sum(spins * H) + 0.05 * np.sum(spins[:, 2] ** 2)
        history.append(float(e_new))
        if abs(e_old - e_new) < tol:
            break
        e_old = e_new
    return spins, float(e_new), history


def simulated_annealing_spin_glass(
    J: np.ndarray,
    spins_init: np.ndarray,
    T_init: float = 2.0,
    T_final: float = 1e-4,
    cooling_rate: float = 0.995,
    steps_per_T: int = 100,
    seed: int = 42,
) -> Tuple[np.ndarray, float, list]:
    """
    模拟退火优化自旋玻璃基态。
    采用 Metropolis 准则接受能量上升的旋转操作。

    转移概率：
        P(accept) = exp(-ΔE / k_B T)，此处取 k_B = 1。

    参数
    ----
    J : np.ndarray
        耦合矩阵。
    spins_init : np.ndarray
        初始自旋构型。
    T_init, T_final : float
        初始与最终温度。
    cooling_rate : float
        每步降温系数。
    steps_per_T : int
        每个温度下的蒙特卡洛步数。

    返回
    ----
    spins_best : np.ndarray
        历史最优构型。
    e_best : float
        历史最优能量。
    history : list
        能量演化记录。
    """
    np.random.seed(seed)
    spins = spins_init.copy()
    N = spins.shape[0]
    H = J @ spins
    e_current = 0.5 * np.sum(spins * H) + 0.05 * np.sum(spins[:, 2] ** 2)
    spins_best = spins.copy()
    e_best = e_current
    history = [e_current]
    T = T_init

    while T > T_final:
        for _ in range(steps_per_T):
            i = np.random.randint(N)
            # 随机旋转轴与角度
            axis = np.random.randn(3)
            axis = axis / (np.linalg.norm(axis) + EPS_MACHINE)
            theta = np.random.uniform(-0.5, 0.5)
            q = axis_angle_to_q(axis, theta)
            s_old = spins[i].copy()
            s_new = q_rotate_vector(q, s_old)
            spins[i] = s_new
            H_new = J @ spins
            e_new = 0.5 * np.sum(spins * H_new) + 0.05 * np.sum(spins[:, 2] ** 2)
            delta_e = e_new - e_current
            if delta_e < 0.0 or np.random.rand() < np.exp(-delta_e / T):
                e_current = e_new
                if e_current < e_best:
                    e_best = e_current
                    spins_best = spins.copy()
            else:
                spins[i] = s_old
        history.append(e_current)
        T *= cooling_rate

    return spins_best, e_best, history
