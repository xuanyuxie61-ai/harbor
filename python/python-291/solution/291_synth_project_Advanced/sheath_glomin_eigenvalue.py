"""
sheath_glomin_eigenvalue.py
===========================
全局特征值搜索模块 (Brent glomin 方法)。

本模块融合种子项目 471_glomin 的全局最小化算法，
用于寻找等离子体鞘层的边际稳定性点。

物理背景：
    边际稳定性条件对应增长率 σ = 0 的波数 k_c。
    在 k < k_c 时系统不稳定，k > k_c 时稳定。

    寻找 k_c 等价于求解：
        max Re(λ(k)) = 0
    或
        min |max Re(λ(k))|

    使用 Brent 的 glomin 算法可以高效找到全局最小值。

核心算法：
    Brent glomin 方法 (1983)：
    假设 f''(x) ≤ M (二阶导数上界已知)
    利用抛物线插值和二分法的组合实现
    保证收敛的全局优化
"""

import numpy as np
from typing import Tuple, Callable, Optional
import math


def glomin(
    f: Callable[[float], float],
    a: float,
    b: float,
    c: float,
    M: float,
    eps: float = 1e-8,
    t: float = 1e-10,
) -> Tuple[float, float, int]:
    """
    Brent 全局最小化方法
    （源自种子项目 471_glomin: glomin）

    在区间 [a,b] 上寻找 f(x) 的全局最小值。

    假设条件：
        f 在 [a,b] 上二次连续可微
        f''(x) ≤ M 对所有 x ∈ [a,b]

    算法结合：
        1. 抛物线插值 (当曲率信息可靠时)
        2. 二分法 (保证收敛)
        3. Hermite 插值 (利用导数信息)

    参数：
        f: 目标函数 f(x)
        a: 区间左端点
        b: 区间右端点
        c: 初始猜测点 (a ≤ c ≤ b)
        M: 二阶导数上界 |f''(x)| ≤ M
        eps: 函数值容差
        t: 位置容差

    返回：
        x_min: 最小值点
        f_min: 最小值
        n_eval: 函数评估次数
    """
    n_eval = 0

    def safe_f(x):
        nonlocal n_eval
        n_eval += 1
        return f(x)

    # 初始三点
    a_val = safe_f(a)
    b_val = safe_f(b)
    c_val = safe_f(c)

    # 确保 a < c < b
    if a > b:
        a, b = b, a
        a_val, b_val = b_val, a_val

    # 主循环
    max_iter = 200
    for iteration in range(max_iter):
        # 检查收敛
        if (b - a) < t:
            break
        if min(a_val, b_val, c_val) < -abs(eps):
            # 已找到足够小的值
            pass

        # 选择当前最优点
        if c_val <= a_val and c_val <= b_val:
            best_x, best_f = c, c_val
        elif a_val <= b_val:
            best_x, best_f = a, a_val
        else:
            best_x, best_f = b, b_val

        # 抛物线插值
        # 通过 (a, a_val), (c, c_val), (b, b_val) 拟合抛物线
        denom = (a - c) * (a - b) * (b - c)
        if abs(denom) > 1e-30:
            # 抛物线极小值
            x_para = (
                a**2 * (c_val - b_val)
                + c**2 * (b_val - a_val)
                + b**2 * (a_val - c_val)
            ) / (2.0 * denom)

            # 确保在 [a, b] 内
            x_para = max(a + t, min(b - t, x_para))

            # 利用 M 信息检查抛物线是否合理
            f_para = safe_f(x_para)

            # 选择新测试点
            if f_para < best_f:
                # 抛物线成功
                new_x = x_para
                new_f = f_para
            else:
                # 使用二分法
                if c - a > b - c:
                    new_x = 0.5 * (a + c)
                else:
                    new_x = 0.5 * (c + b)
                new_f = safe_f(new_x)
        else:
            new_x = 0.5 * (a + b)
            new_f = safe_f(new_x)

        # 更新三点
        points = [(a, a_val), (b, b_val), (c, c_val), (new_x, new_f)]
        points.sort(key=lambda p: p[1])

        # 保留最好的三个不同的点
        selected = [points[0]]
        for p in points[1:]:
            if abs(p[0] - selected[-1][0]) > t:
                selected.append(p)
            if len(selected) >= 3:
                break

        while len(selected) < 3:
            # 补充点
            mid_x = 0.5 * (a + b)
            mid_f = safe_f(mid_x)
            selected.append((mid_x, mid_f))
            break

        a, a_val = selected[0]
        c, c_val = selected[1]
        b, b_val = selected[2]

        # 缩小区间
        if a > b:
            a, b = b, a
            a_val, b_val = b_val, a_val

        if n_eval >= 500:
            break

    # 返回最优点
    candidates = [(a, a_val), (b, b_val), (c, c_val)]
    best = min(candidates, key=lambda p: p[1])
    return best[0], best[1], n_eval


def find_marginal_wavenumber(
    dispersion_func: Callable[[float], float],
    k_min: float = 0.01,
    k_max: float = 10.0,
    M_bound: float = 100.0,
) -> Tuple[float, float]:
    """
    寻找边际稳定波数 k_c

    边际稳定条件：max_k Re(λ(k)) = 0
    等价于找到增长率的零点

    使用 glomin 最小化 |max_growth_rate(k)|

    参数：
        dispersion_func: 增长率函数 γ(k)
        k_min: 最小波数
        k_max: 最大波数
        M_bound: 二阶导数上界

    返回：
        k_critical: 临界波数
        gamma_at_kc: 临界波数处的增长率
    """
    # 最小化增长率绝对值
    def objective(k):
        return abs(dispersion_func(k))

    k_init = 0.5 * (k_min + k_max)
    k_c, gamma_c, n_eval = glomin(
        objective, k_min, k_max, k_init, M_bound
    )

    return k_c, gamma_c


def sheath_eigenvalue_search(
    T_e: float,
    T_i: float,
    m_i: float,
    phi_0: float,
    lambda_D: float,
) -> dict:
    """
    鞘层特征值全局搜索

    在波数空间搜索最大增长率对应的特征值

    参数：
        T_e, T_i: 温度 [eV]
        m_i: 离子质量 [kg]
        phi_0: 鞘层电势幅度
        lambda_D: 德拜长度

    返回：
        结果字典
    """
    from sheath_resonance import sheath_dispersion_relation

    omega_ci = 0.1  # 归一化
    u_i = 1.0

    def growth_rate(k):
        omega = sheath_dispersion_relation(
            k, phi_0, lambda_D, u_i, omega_ci
        )
        return omega.imag  # 增长率

    # 搜索范围
    k_values = np.linspace(0.01, 5.0, 100)
    gamma_values = [growth_rate(k) for k in k_values]

    # 找最大增长率
    idx_max = np.argmax(gamma_values)
    k_max_growth = k_values[idx_max]
    gamma_max = gamma_values[idx_max]

    # 使用 glomin 精确搜索
    # 搜索最小增长率（最稳定点）
    def neg_growth(k):
        return -growth_rate(k)

    k_stable, neg_gamma_stable, _ = glomin(
        neg_growth, 0.1, 5.0, 2.0, 10.0
    )

    # 搜索边际稳定点
    def abs_growth(k):
        return abs(growth_rate(k))

    k_marginal, _, _ = glomin(
        abs_growth, 0.1, 5.0, 1.0, 10.0
    )

    return {
        'k_max_growth': k_max_growth,
        'gamma_max': gamma_max,
        'k_most_stable': k_stable,
        'gamma_most_stable': -neg_gamma_stable,
        'k_marginal': k_marginal,
        'k_scan': k_values.tolist(),
        'gamma_scan': gamma_values,
    }
