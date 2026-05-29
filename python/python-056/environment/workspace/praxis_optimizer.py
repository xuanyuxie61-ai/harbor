"""
praxis_optimizer.py
================================================================================
无梯度多维优化模块 (来源于 907_praxis 项目)
================================================================================
本模块实现 Brent 的 PRAXIS 算法（主轴法），用于潮汐能提取系统
的多参数优化。该算法不需要梯度信息，适用于目标函数包含数值
模拟（CFD/FEA）黑箱的情况。在涡轮阵列布局、叶片攻角调度、
系泊系统预张力等优化问题中发挥核心作用。

核心公式:
    二次近似:
        Q(x') = F(x) + ½ (x'-x)^T A (x'-x)
        A = V^{-T} D V^{-1}

    其中 V 为搜索方向矩阵，D 为二阶差分对角阵。

    线性搜索 (黄金分割/抛物线插值):
        沿方向 v 最小化 φ(s) = F(x + s·v)

    SVD 更新主轴方向:
        V, D = minfit(V)  (奇异值分解)
"""

import numpy as np
from typing import Callable, Tuple


def flin(
    n: int,
    jsearch: int,
    l: float,
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    v: np.ndarray,
) -> float:
    """
    沿搜索方向计算函数值。

    参数:
        n: 维度
        jsearch: 搜索方向索引
        l: 步长
        f: 目标函数
        x: 当前点
        v: 方向矩阵

    返回:
        f(x + l * v[:, jsearch])
    """
    x_try = x.copy()
    if 1 <= jsearch <= n:
        x_try = x_try + l * v[:, jsearch - 1]
    elif jsearch == 0:
        x_try = x_try + l * x_try
    return f(x_try)


def minny(
    n: int,
    jsearch: int,
    nits: int,
    d2: float,
    x1: float,
    x2: float,
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    v: np.ndarray,
    h: float,
) -> Tuple[float, float, float, np.ndarray]:
    """
    沿给定方向进行一维极小化。

    参数:
        n, jsearch, nits, d2, x1, x2, f, x, v, h

    返回:
        (d2, lds, fx, x_new)
    """
    small = np.finfo(float).eps ** 2
    m2 = np.sqrt(np.finfo(float).eps)

    def _eval(l: float) -> float:
        return flin(n, jsearch, l, f, x, v)

    # 初始区间
    a = min(x1, x2)
    b = max(x1, x2)
    fa = _eval(a)
    fb = _eval(b)

    if fa > fb:
        a, b = b, a
        fa, fb = fb, fa

    c = b
    fc = fb
    tol = m2 * abs(b - a)

    for _ in range(nits * 10):
        mid = 0.5 * (a + c)
        if abs(c - a) < tol:
            break
        # 抛物线插值
        denom = 2.0 * (fa - 2.0 * fb + fc)
        if abs(denom) > small:
            t = 0.5 * (a + c) + 0.5 * (fa - fc) * (c - a) / denom
            if a < t < c:
                ft = _eval(t)
                if ft < fb:
                    b, fb = t, ft
                    continue
        # 黄金分割步
        if b - a > c - b:
            t = b - 0.381966 * (b - a)
        else:
            t = b + 0.381966 * (c - b)
        ft = _eval(t)
        if ft < fb:
            a, fa = b, fb
            b, fb = t, ft
        elif ft < fc:
            c, fc = t, ft
        else:
            if t < b:
                a, fa = t, ft
            else:
                c, fc = t, ft

    lds = b
    x_new = x.copy()
    if 1 <= jsearch <= n:
        x_new = x_new + lds * v[:, jsearch - 1]
    return d2, lds, fb, x_new


def svsort(n: int, d: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    对奇异值和奇异向量按降序排序。

    参数:
        n: 维度
        d: 奇异值数组
        v: 右奇异向量矩阵

    返回:
        (d_sorted, v_sorted)
    """
    idx = np.argsort(-np.abs(d))
    return d[idx], v[:, idx]


def praxis(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    tol: float = 1e-6,
    h0: float = 1.0,
    max_iter: int = 500,
) -> Tuple[float, np.ndarray, int]:
    """
    PRAXIS 无梯度优化算法。

    参数:
        f: 目标函数 F(x)
        x0: 初始猜测
        tol: 收敛容差
        h0: 初始步长
        max_iter: 最大函数评估次数

    返回:
        (fmin, xmin, nfev)
    """
    x = np.asarray(x0, dtype=float).copy()
    n = x.size
    machep = np.finfo(float).eps
    small = machep * machep
    m2 = np.sqrt(machep)
    t = small + abs(tol)
    h = max(h0, 100.0 * t)

    v = np.eye(n)
    d = np.zeros(n)
    nf = 1
    fx = f(x)

    ldt = h
    ktm = 1
    kt = 0

    while nf < max_iter:
        sf = d[0]
        d[0] = 0.0

        # 沿第一方向搜索
        x_prev = x.copy()
        d2, s, fx_new, x = minny(n, 1, 2, d[0], 0.0, 0.0, f, x, v, h)
        d[0] = d2

        # 内循环: 沿非共轭方向搜索
        for k in range(2, n + 1):
            y = x.copy()
            sf = fx_new

            for k2 in range(k, n + 1):
                d2, s, fx_new, x = minny(n, k2, 2, d[k2 - 1], 0.0, 0.0, f, x, v, h)
                d[k2 - 1] = d2

            # 沿共轭方向搜索
            for k2 in range(1, k):
                d2, s, fx_new, x = minny(n, k2, 2, d[k2 - 1], 0.0, 0.0, f, x, v, h)
                d[k2 - 1] = d2

            # 更新搜索方向
            lds = np.linalg.norm(x - y)
            if lds > small:
                v[:, k - 1] = (x - y) / lds
                d2, lds, fx_new, x = minny(n, k, 4, 0.0, lds, 0.0, f, x, v, h)
                d[k - 1] = d2

        ldt = 0.01 * ldt
        ldt = max(ldt, lds)
        t2 = m2 * np.linalg.norm(x) + t
        if 0.5 * t2 < ldt:
            kt = -1
        kt += 1
        if kt > ktm:
            break

        # SVD 更新方向
        vt = v.T
        try:
            u_mat, s_vals, _ = np.linalg.svd(vt, full_matrices=False)
            d = 1.0 / (s_vals + small)
            d, v = svsort(n, d, u_mat)
        except np.linalg.LinAlgError:
            break

        # 检查收敛
        if np.linalg.norm(x - x_prev) < tol * (1.0 + np.linalg.norm(x)):
            break

    return fx_new, x, nf


def optimize_turbine_array(
    n_turbines: int = 5,
    domain_size: float = 500.0,
    min_spacing: float = 50.0,
) -> Tuple[np.ndarray, float]:
    """
    优化潮汐涡轮阵列布局以最大化总功率输出。

    目标函数:
        minimize  -Σ_i P_i(x_i, y_i)
        subject to  ||(x_i,y_i) - (x_j,y_j)|| ≥ d_min

    其中功率使用 Jensen 尾流模型估算:
        P_i = ½ ρ A C_p [U_∞ (1 - Σ_j C_T (D/(D+k·d_{ij}))²)]³

    参数:
        n_turbines: 涡轮数量
        domain_size: 域大小 (m)
        min_spacing: 最小间距 (m)

    返回:
        (positions, total_power)
    """
    rho = 1025.0
    A = 20.0
    cp = 16.0 / 27.0
    Ct = 0.8
    D = np.sqrt(4.0 * A / np.pi)
    k_wake = 0.05
    U_inf = 2.5

    def objective(x_flat: np.ndarray) -> float:
        if x_flat.size != 2 * n_turbines:
            return 1e10
        pos = x_flat.reshape((n_turbines, 2))
        # 惩罚项: 间距约束违反
        penalty = 0.0
        for i in range(n_turbines):
            for j in range(i + 1, n_turbines):
                dist = np.linalg.norm(pos[i] - pos[j])
                if dist < min_spacing:
                    penalty += 1e6 * (min_spacing - dist) ** 2

        # Jensen 尾流模型
        speeds = np.full(n_turbines, U_inf)
        for i in range(n_turbines):
            deficit = 0.0
            for j in range(n_turbines):
                if j == i:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dx > 0:  # i 在 j 下游
                    wake_diam = D + 2.0 * k_wake * dist
                    if abs(dy) < 0.5 * wake_diam:
                        deficit += Ct * (D / wake_diam) ** 2
            speeds[i] = max(U_inf * (1.0 - deficit), 0.1)

        # TODO(Hole_1): 根据 Jensen 尾流模型计算各涡轮功率并返回负总功率
        # 物理公式: P_i = 0.5 * rho * A * C_p * U_i^3
        # 其中 U_i = U_inf * (1.0 - deficit) 为考虑尾流影响后的有效流速
        # 请补全 power 的计算逻辑
        power = np.zeros(n_turbines)
        return 0.0

    # 初始布局: 均匀分布
    x0 = np.random.rand(2 * n_turbines) * domain_size
    fmin, xmin, _ = praxis(objective, x0, tol=1e-4, h0=domain_size * 0.1, max_iter=2000)
    positions = xmin.reshape((n_turbines, 2))

    # 投影到域内
    positions = np.clip(positions, 0.0, domain_size)
    total_power = -objective(xmin)
    # TODO(Hole_1): 确认返回值顺序与 main.py 中的解包一致
    return total_power, positions
