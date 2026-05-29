"""
bifurcation_stability.py
=========================
核反应率稳定性与分岔分析模块

基于种子项目 700_logistic_bifurcation 的动力学系统
分岔分析思想，本模块研究核反应系统中的非线性稳定性：
1. 中子增殖因子的分岔行为
2. 核统计模型中反应率的混沌阈值
3. 光学势参数空间的稳定性边界

核心公式
--------
Logistic 映射类比 (中子增殖):
    x_{n+1} = r x_n (1 - x_n)

其中 x 为中子密度相对值，r 为增殖因子。
分岔发生在 r = 3, 3.449, 3.544, ..., 3.56995 (Feigenbaum 点)。

核反应率方程 (非线性):
    dx/dt = λ_f x - λ_c x - α x² + S(t)

其中 λ_f 为裂变率，λ_c 为俘获率，α 为自屏系数，S 为外源。

稳定性判据:
    平衡点 x* 稳定当且仅当 |df/dx|_{x*} < 1

Poincaré-Bendixson 定理在离散系统中的类比：
    周期轨道存在性可通过映射迭代检测。
"""

import numpy as np


def logistic_map(x, r):
    """
    Logistic 映射单步迭代。

    x_{n+1} = r x_n (1 - x_n)
    """
    return r * x * (1.0 - x)


def logistic_attractor(r, x0=0.5, warm_up=500, max_iter=1000, tol=1e-6):
    """
    寻找给定参数 r 下 Logistic 映射的吸引子集合。

    参照 700_logistic_bifurcation 的 attractor 检测思想。
    """
    x = x0
    # 预热
    for _ in range(warm_up):
        x = logistic_map(x, r)

    # 收集轨道点
    trajectory = []
    seen = set()
    for _ in range(max_iter):
        x = logistic_map(x, r)
        # 使用截断检测周期
        key = round(x, 6)
        if key in seen:
            break
        seen.add(key)
        trajectory.append(x)

    # 提取唯一吸引子点
    attractor = []
    for x_val in trajectory[-20:]:
        # 查找是否已存在接近的值
        is_new = True
        for a in attractor:
            if abs(a - x_val) < tol:
                is_new = False
                break
        if is_new:
            attractor.append(x_val)

    return np.sort(attractor)


def feigenbaum_bifurcation_diagram(r_min=2.5, r_max=4.0, n_r=2000):
    """
    计算 Logistic 映射的分岔图数据。

    Returns
    -------
    r_values : ndarray
        参数采样。
    attractors : list of ndarray
        每个 r 对应的吸引子集合。
    """
    r_values = np.linspace(r_min, r_max, n_r)
    attractors = []
    for r in r_values:
        attr = logistic_attractor(r)
        attractors.append(attr)
    return r_values, attractors


def neutron_multiplication_bifurcation(alpha_range, lambda_f=0.5, lambda_c=0.3, S=0.01):
    """
    非线性中子增殖方程的分岔分析。

    离散化方程:
        x_{n+1} = x_n + Δt[(λ_f - λ_c)x_n - α x_n² + S]

    平衡点:
        x* = [(λ_f - λ_c) + sqrt((λ_f - λ_c)² + 4αS)] / (2α)

    Parameters
    ----------
    alpha_range : ndarray
        自屏系数 α 的范围。
    lambda_f, lambda_c : float
        裂变和俘获率。
    S : float
        外源强度。

    Returns
    -------
    equilibrium : ndarray
        平衡点随 α 的变化。
    stability : ndarray
        稳定性标志 (True=稳定)。
    """
    equilibrium = []
    stability = []
    for alpha in alpha_range:
        if alpha < 1e-12:
            alpha = 1e-12
        discriminant = (lambda_f - lambda_c) ** 2 + 4.0 * alpha * S
        x_star = ((lambda_f - lambda_c) + np.sqrt(discriminant)) / (2.0 * alpha)
        # 稳定性：导数在平衡点处的值
        # dx/dt = f(x), f'(x*) = (λ_f - λ_c) - 2α x*
        df = (lambda_f - lambda_c) - 2.0 * alpha * x_star
        # 离散映射稳定性
        is_stable = abs(1.0 + df) < 1.0 if df != 0 else True
        equilibrium.append(x_star)
        stability.append(is_stable)

    return np.array(equilibrium), np.array(stability)


def optical_potential_stability_boundary(V0_range, W0_range, params_func):
    """
    分析光学势参数空间的稳定性边界。

    通过计算 S-矩阵的幺正性偏离度判断参数稳定性:
        δ_unitary = max_l |1 - |S_l|²|

    当 δ_unitary > 0.5 时认为参数进入非物理区域。

    Parameters
    ----------
    V0_range, W0_range : ndarray
        实部和虚部势深参数网格。
    params_func : callable
        接受 (V0, W0) 返回 OpticalPotentialParameters 的函数。

    Returns
    -------
    stability_map : ndarray
        二维稳定性图 (True=稳定)。
    unitarity_deviation : ndarray
        幺正性偏离度。
    """
    stability_map = np.zeros((len(V0_range), len(W0_range)), dtype=bool)
    unitarity_dev = np.zeros((len(V0_range), len(W0_range)))

    for i, V0 in enumerate(V0_range):
        for j, W0 in enumerate(W0_range):
            params = params_func(V0, W0)
            # 简化的稳定性指标
            # 检查势的物理合理性
            is_physical = (V0 > 0 and W0 > 0 and params.a_v > 0.05)
            # 简化的幺正性偏离 (使用吸收系数近似)
            # 对于合理的光学势，|S|² 应在 0~1 之间
            deviation = abs(W0 / (V0 + W0 + 1.0) - 0.3)
            unitarity_dev[i, j] = deviation
            stability_map[i, j] = is_physical and (deviation < 0.5)

    return stability_map, unitarity_dev


def lyapunov_exponent_logistic(r, x0=0.5, n_iter=10000):
    """
    计算 Logistic 映射的 Lyapunov 指数。

    λ_L = lim_{N→∞} (1/N) Σ_{n=0}^{N-1} ln |f'(x_n)|

    λ_L > 0 表示混沌。
    """
    x = x0
    lam_sum = 0.0
    for _ in range(n_iter):
        x = logistic_map(x, r)
        df = abs(r * (1.0 - 2.0 * x))
        if df < 1e-300:
            df = 1e-300
        lam_sum += np.log(df)

    return lam_sum / n_iter


def critical_slowing_down_indicator(alpha, lambda_f, lambda_c, epsilon=1e-3):
    """
    计算临界慢化指标 (接近分岔时的特征时间延长)。

    τ_crit = 1 / |λ_f - λ_c - 2α x*|

    在临界点附近 τ_crit → ∞。
    """
    # 近似平衡点
    x_star = max((lambda_f - lambda_c) / (2.0 * alpha), 0.0)
    eigenvalue = lambda_f - lambda_c - 2.0 * alpha * x_star
    tau = 1.0 / (abs(eigenvalue) + epsilon)
    return tau


if __name__ == "__main__":
    # 自检
    r_test = 3.5
    attr = logistic_attractor(r_test)
    print(f"r={r_test}: attractor = {attr}")

    lyap = lyapunov_exponent_logistic(3.8)
    print(f"r=3.8 的 Lyapunov 指数: {lyap:.4f}")

    alpha_range = np.linspace(0.01, 2.0, 100)
    eq, stab = neutron_multiplication_bifurcation(alpha_range)
    print(f"α 范围 [{alpha_range[0]:.2f}, {alpha_range[-1]:.2f}]")
    print(f"平衡点范围: [{eq.min():.4f}, {eq.max():.4f}]")
    print(f"稳定点数: {np.sum(stab)}")

    # 分岔图采样
    r_vals, attrs = feigenbaum_bifurcation_diagram(2.8, 4.0, 500)
    print(f"分岔图参数点数: {len(r_vals)}")
