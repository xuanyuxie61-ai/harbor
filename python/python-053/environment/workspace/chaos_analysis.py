"""
chaos_analysis.py
=================
基于 levy_dragon_chaos (670_levy_dragon_chaos) 与 cross_chaos (227_cross_chaos) 的
迭代函数系统 (IFS) 框架，分析海气耦合系统的非线性动力学特征，
包括吸引子维数、Lyapunov 指数与分岔行为。

科学背景
--------
海气耦合系统是一个典型的非线性耗散系统。根据 Timmermann et al. (2003) 的理论，
ENSO 在参数空间的某些区域可以表现出混沌行为（不规则振荡），
而在另一些区域则是周期性或准周期性的。

本模块通过：
1. IFS 方法生成海气耦合吸引子的分形近似；
2. 计算 Lyapunov 指数量化系统对初值的敏感依赖性；
3. 进行分岔分析，识别周期倍增通往混沌的路径。

核心公式
--------
1. 海气耦合系统的 Poincaré 映射（离散化）：
   
   对 recharge-discharge oscillator 在 T_E = 0 的截面上取 Poincaré 截面，
   得到一维映射：h_{n+1} = f(h_n)

2. Lyapunov 指数（一维映射）：
   
   λ = lim_{N→∞} (1/N) * Σ_{n=0}^{N-1} ln |f'(h_n)|

   λ > 0 表示混沌，λ < 0 表示稳定周期轨道。

3. 关联维数 (Grassberger-Procaccia)：
   
   C(r) = (2 / (N(N-1))) * Σ_{i<j} Θ(r - ||x_i - x_j||)
   
   D_2 = lim_{r→0} d(ln C(r)) / d(ln r)

4. IFS 吸引子（类比 levy_dragon）：
   对于线性映射 A 和平移 b，迭代：
   x_{n+1} = A * x_n + b
   
   在海气耦合中，A 可视为线性化 Jacobian，b 为外部强迫。

5. Cross 混沌吸引子的多尺度映射：
   x_{n+1} = A_k * x_n + b_k,  k 以概率 p_k 选择
   
   用于模拟 ENSO 在多种海气状态间的随机切换。
"""

import numpy as np
from typing import Tuple, List, Optional


def lyapunov_exponent_1d(f: callable, df: callable, x0: float,
                         n_iter: int = 10000, n_transient: int = 1000) -> float:
    """
    计算一维映射的 Lyapunov 指数。

    公式：
    λ = (1/N) * Σ_{n=0}^{N-1} ln |f'(x_n)|

    参数
    ----
    f : callable
        映射函数 f(x)。
    df : callable
        导数函数 f'(x)。
    x0 : float
        初始点。
    n_iter : int
        迭代次数。
    n_transient : int
        暂态丢弃次数。

    返回
    ----
    lyap : float
        Lyapunov 指数。
    """
    x = float(x0)
    # 暂态
    for _ in range(n_transient):
        x = f(x)
        if not np.isfinite(x):
            x = x0

    lyap_sum = 0.0
    count = 0
    for _ in range(n_iter):
        x = f(x)
        dfx = df(x)
        if abs(dfx) < 1e-15:
            dfx = 1e-15
        if np.isfinite(dfx) and dfx != 0.0:
            lyap_sum += np.log(abs(dfx))
            count += 1
        if not np.isfinite(x):
            x = x0

    if count == 0:
        return 0.0
    return lyap_sum / count


def correlation_dimension(trajectory: np.ndarray,
                          r_min: float = 1e-3,
                          r_max: float = 1.0,
                          n_r: int = 50) -> float:
    """
    使用 Grassberger-Procaccia 算法计算关联维数 D2。

    公式：
    C(r) = (2/(N(N-1))) * Σ_{i<j} Θ(r - ||x_i - x_j||)
    D2 = d(ln C)/d(ln r) 在适中 r 区间的斜率

    参数
    ----
    trajectory : np.ndarray, shape (N, dim)
        相空间轨迹。
    r_min, r_max : float
        距离范围。
    n_r : int
        r 的采样点数。

    返回
    ----
    d2 : float
        关联维数估计。
    """
    trajectory = np.atleast_2d(trajectory).T
    if trajectory.ndim != 2:
        raise ValueError("trajectory must be 2D array")

    N = trajectory.shape[0]
    if N < 100:
        return 0.0

    # 子采样以控制计算量
    max_samples = 2000
    if N > max_samples:
        idx = np.random.choice(N, max_samples, replace=False)
        traj = trajectory[idx]
        N = max_samples
    else:
        traj = trajectory

    r_vals = np.logspace(np.log10(r_min), np.log10(r_max), n_r)
    c_vals = np.zeros(n_r)

    for i_r, r in enumerate(r_vals):
        count = 0
        for i in range(N):
            dists = np.linalg.norm(traj[i + 1:] - traj[i], axis=1)
            count += np.sum(dists < r)
        c_vals[i_r] = 2.0 * count / (N * (N - 1))

    # 在 C(r) 适中的区域拟合斜率
    valid = (c_vals > 1e-4) & (c_vals < 0.5)
    if np.sum(valid) < 5:
        valid = c_vals > 0

    if np.sum(valid) < 3:
        return 0.0

    log_r = np.log(r_vals[valid])
    log_c = np.log(c_vals[valid])

    # 线性回归
    A = np.vstack([log_r, np.ones_like(log_r)]).T
    slope, _ = np.linalg.lstsq(A, log_c, rcond=None)[0]
    return float(slope)


def levy_dragon_ifs(n_iter: int = 10000) -> np.ndarray:
    """
    生成 Levy Dragon 分形（IFS 方法）。

    映射：
    A0 = [[0.5, 0.5], [-0.5, 0.5]],  b0 = [0.5, 0.5]
    A1 = [[0.5, -0.5], [0.5, 0.5]],  b1 = [-0.5, 0.5]

    以概率 0.5 选择每个映射。

    在海气耦合的类比中，A0, A1 可视为不同 ENSO 相位下的线性化动力学，
    b0, b1 为对应相位的平均状态偏移。
    """
    A0 = np.array([[0.5, 0.5], [-0.5, 0.5]])
    A1 = np.array([[0.5, -0.5], [0.5, 0.5]])
    b0 = np.array([0.5, 0.5])
    b1 = np.array([-0.5, 0.5])

    x = np.random.rand(2)
    points = np.zeros((n_iter, 2))

    for i in range(n_iter):
        if np.random.rand() < 0.5:
            x = A0 @ x + b0
        else:
            x = A1 @ x + b1
        points[i] = x

    return points


def cross_chaos_ifs(n_iter: int = 10000) -> np.ndarray:
    """
    生成 Cross 混沌吸引子（五映射 IFS）。

    映射：
    A = (1/3) * I
    b = [(1/3,0), (0,1/3), (1/3,1/3), (2/3,1/3), (1/3,2/3)]

    以等概率选择 5 个平移。

    用于模拟 ENSO 在五种海气状态（强 El Niño, 弱 El Niño, 中性,
    弱 La Niña, 强 La Niña）之间的随机游走。
    """
    A = np.array([[1.0 / 3.0, 0.0],
                  [0.0, 1.0 / 3.0]])
    b = np.array([
        [1.0 / 3.0, 0.0],
        [0.0, 1.0 / 3.0],
        [1.0 / 3.0, 1.0 / 3.0],
        [2.0 / 3.0, 1.0 / 3.0],
        [1.0 / 3.0, 2.0 / 3.0]
    ]).T

    x = np.random.rand(2)
    points = np.zeros((n_iter, 2))

    for i in range(n_iter):
        j = np.random.randint(0, 5)
        x = A @ x + b[:, j]
        points[i] = x

    return points


def enso_poincare_map(h_n: float,
                      r: float = 0.25,
                      alpha: float = 0.5,
                      R: float = 1.0,
                      epsilon: float = 0.3,
                      gamma: float = 0.4) -> float:
    """
    构造 ENSO recharge-discharge oscillator 的 Poincaré 映射近似。

    在 T_E = 0 的截面上，h_W 的演化近似为逻辑斯蒂型映射：

    h_{n+1} = μ * h_n * (1 - h_n / K)

    其中 μ = 1 + Δt * (γ - Rα/r), K = r / (Rα) * (γ - Rα/r) / ε

    参数
    ----
    h_n : float
        当前 WWV 异常值。
    r, alpha, R, epsilon, gamma : float
        RDO 模型参数。

    返回
    ----
    h_next : float
        下一个 Poincaré 截面值。
    """
    dt = 0.01
    mu = 1.0 + dt * (gamma - R * alpha / r)
    K = (r / (R * alpha)) * (gamma - R * alpha / r) / epsilon if epsilon > 0 else 1e10

    if K <= 0:
        return h_n * np.exp(-dt * r)

    h_next = mu * h_n * (1.0 - h_n / K)
    return h_next


def enso_lyapunov_exponent(r: float = 0.25,
                           alpha: float = 0.5,
                           R: float = 1.0,
                           epsilon: float = 0.3,
                           gamma: float = 0.4,
                           n_iter: int = 5000) -> float:
    """
    计算 ENSO Poincaré 映射的 Lyapunov 指数。

    参数
    ----
    r, alpha, R, epsilon, gamma : float
        RDO 参数。
    n_iter : int
        迭代次数。

    返回
    ----
    lyap : float
        Lyapunov 指数。
    """
    dt = 0.01
    mu = 1.0 + dt * (gamma - R * alpha / r)
    K = (r / (R * alpha)) * (gamma - R * alpha / r) / epsilon if epsilon > 0 else 1e10

    def f(h):
        if K <= 0:
            return h * np.exp(-dt * r)
        return mu * h * (1.0 - h / K)

    def df(h):
        if K <= 0:
            return np.exp(-dt * r)
        return mu * (1.0 - 2.0 * h / K)

    return lyapunov_exponent_1d(f, df, 0.1, n_iter=n_iter)


def bifurcation_diagram(param_name: str,
                        param_range: np.ndarray,
                        r: float = 0.25,
                        alpha: float = 0.5,
                        R: float = 1.0,
                        epsilon: float = 0.3,
                        gamma: float = 0.4) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    生成 ENSO 模型的分岔图。

    参数
    ----
    param_name : str
        扫描参数名（"gamma", "R", "epsilon" 等）。
    param_range : np.ndarray
        参数取值范围。
    r, alpha, R, epsilon, gamma : float
        基准参数。

    返回
    ----
    params : np.ndarray
        参数值。
    attractors : List[np.ndarray]
        每个参数值对应的吸引子点集。
    """
    attractors = []
    params = []

    for p in param_range:
        kw = {"r": r, "alpha": alpha, "R": R, "epsilon": epsilon, "gamma": gamma}
        kw[param_name] = p

        dt = 0.01
        mu = 1.0 + dt * (kw["gamma"] - kw["R"] * kw["alpha"] / kw["r"])
        K = (kw["r"] / (kw["R"] * kw["alpha"])) * (kw["gamma"] - kw["R"] * kw["alpha"] / kw["r"]) / kw["epsilon"] \
            if kw["epsilon"] > 0 else 1e10

        if K <= 0:
            attractors.append(np.array([0.0]))
            params.append(p)
            continue

        # 迭代至吸引子
        h = 0.1
        for _ in range(2000):
            h = mu * h * (1.0 - h / K)

        # 收集吸引子点
        points = []
        for _ in range(100):
            h = mu * h * (1.0 - h / K)
            points.append(h)

        attractors.append(np.array(points))
        params.append(p)

    return np.array(params), attractors
