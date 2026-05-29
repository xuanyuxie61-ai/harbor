"""
uncertainty_quantification.py
=============================
不确定性量化与可靠性分析模块。
整合自：
  - 556_hypercube_distance：高维超立方体随机采样与距离统计
  - 348_fair_dice_simulation：逆变换采样（离散/连续分布转换）

物理背景：
  增材制造工艺存在大量不确定性来源：材料参数波动、工艺参数偏差、
  几何误差等。蒙特卡洛方法通过大量随机采样评估这些不确定性对
  结构性能（柔度、强度、寿命）的影响。

核心方法：
  1. 高维设计空间均匀采样（hypercube_distance 思想）
  2. 逆变换采样：将均匀随机变量转换为目标分布（fair_dice 思想）
  3. Monte Carlo 估计：均值、方差、失效概率
  4. Sobol 灵敏度指标（基于方差分解）
"""

import numpy as np
from typing import Callable, Tuple, List, Optional


# =============================================================================
# 1. 高维超立方体采样 (hypercube_distance 思想)
# =============================================================================

def sample_hypercube_uniform(d: int, n_samples: int,
                              bounds: Optional[np.ndarray] = None,
                              seed: Optional[int] = None) -> np.ndarray:
    """
    在 d 维单位超立方体 [0,1]^d 内生成 n_samples 个均匀随机点。

    如果提供 bounds（shape (d, 2)），则映射到 [bounds[:,0], bounds[:,1]]。
    """
    if seed is not None:
        np.random.seed(seed)
    samples = np.random.rand(n_samples, d)
    if bounds is not None:
        samples = bounds[:, 0] + samples * (bounds[:, 1] - bounds[:, 0])
    return samples


def hypercube_distance_statistics(samples: np.ndarray) -> dict:
    """
    统计高维空间中样本点对距离分布。
    随着维度增加，随机点之间的欧氏距离趋于集中（距离集中现象）。

    返回均值、方差、最小/最大距离。
    """
    n = samples.shape[0]
    if n < 2:
        return {"mean": 0.0, "var": 0.0, "min": 0.0, "max": 0.0}
    # 计算所有无序点对的距离
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(samples[i] - samples[j])
            dists.append(d)
    dists = np.array(dists)
    return {
        "mean": float(np.mean(dists)),
        "var": float(np.var(dists)),
        "std": float(np.std(dists)),
        "min": float(np.min(dists)),
        "max": float(np.max(dists)),
        "median": float(np.median(dists)),
    }


# =============================================================================
# 2. 逆变换采样 (fair_dice 思想推广到连续分布)
# =============================================================================

def inverse_transform_sampling(cdf_func: Callable[[float], float],
                                n_samples: int,
                                xmin: float = -3.0,
                                xmax: float = 3.0,
                                n_grid: int = 1000,
                                seed: Optional[int] = None) -> np.ndarray:
    """
    逆变换采样：给定 CDF，生成服从该分布的随机样本。

    算法：
        1. 生成 U ~ Uniform(0,1)
        2. 求解 X = CDF^{-1}(U)  （通过查表+线性插值）

    Parameters
    ----------
    cdf_func : callable
        累积分布函数 CDF(x)。
    n_samples : int
        采样数量。
    xmin, xmax : float
        CDF 反函数求解区间。
    """
    if seed is not None:
        np.random.seed(seed)
    # 在网格上预计算 CDF
    x_grid = np.linspace(xmin, xmax, n_grid)
    cdf_grid = np.array([cdf_func(x) for x in x_grid])
    # 确保单调性
    cdf_grid = np.maximum.accumulate(cdf_grid)
    # 归一化到 [0,1]
    cdf_min, cdf_max = cdf_grid[0], cdf_grid[-1]
    if cdf_max > cdf_min:
        cdf_grid = (cdf_grid - cdf_min) / (cdf_max - cdf_min)
    else:
        cdf_grid = np.linspace(0.0, 1.0, n_grid)

    U = np.random.rand(n_samples)
    # 线性插值求反函数
    samples = np.interp(U, cdf_grid, x_grid)
    return samples


def gaussian_cdf_approx(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """
    高斯分布的近似 CDF（误差函数）。
        Φ(x; μ, σ) = 0.5 · [1 + erf((x-μ)/(σ·√2))]
    """
    from math import erf
    return 0.5 * (1.0 + erf((x - mu) / (sigma * np.sqrt(2.0))))


def sample_gaussian(mu: float, sigma: float, n_samples: int,
                     seed: Optional[int] = None) -> np.ndarray:
    """
    使用 Box-Muller / 逆变换采样生成高斯随机变量。
    """
    if seed is not None:
        np.random.seed(seed)
    # 使用 numpy 内置方法（等同于逆变换采样）
    return np.random.normal(mu, sigma, n_samples)


def sample_discrete_distribution(pmf: np.ndarray, n_samples: int,
                                  seed: Optional[int] = None) -> np.ndarray:
    """
    离散分布逆变换采样（fair_dice 思想的直接推广）。

    Parameters
    ----------
    pmf : ndarray
        概率质量函数，自动归一化。
    """
    if seed is not None:
        np.random.seed(seed)
    pmf = np.array(pmf, dtype=np.float64)
    pmf = pmf / np.sum(pmf)
    cdf = np.cumsum(pmf)
    U = np.random.rand(n_samples)
    samples = np.searchsorted(cdf, U)
    return samples


# =============================================================================
# 3. Monte Carlo 可靠性分析
# =============================================================================

def monte_carlo_reliability(performance_func: Callable[[np.ndarray], float],
                            input_sampler: Callable[[], np.ndarray],
                            n_samples: int,
                            threshold: float = 0.0) -> dict:
    """
    蒙特卡洛估计失效概率。

    定义极限状态函数 g(X)，失效域 F = {X | g(X) ≤ threshold}。
    失效概率估计：
        P_f ≈ (1/N) · Σ_{i=1}^N I[g(X_i) ≤ threshold]

    方差：Var(P_f) = P_f(1-P_f)/N

    Returns
    -------
    dict: 包含 pf_estimate, std_error, cov, samples 等。
    """
    failures = 0
    g_vals = []
    for _ in range(n_samples):
        X = input_sampler()
        g = performance_func(X)
        g_vals.append(g)
        if g <= threshold:
            failures += 1

    pf = failures / n_samples
    std_err = np.sqrt(pf * (1.0 - pf) / n_samples)
    cov = std_err / (pf + 1e-14)  # 变异系数

    g_vals = np.array(g_vals)
    return {
        "pf_estimate": pf,
        "std_error": std_err,
        "cov": cov,
        "mean_g": float(np.mean(g_vals)),
        "std_g": float(np.std(g_vals)),
        "min_g": float(np.min(g_vals)),
        "max_g": float(np.max(g_vals)),
        "n_samples": n_samples,
    }


def latin_hypercube_sampling(d: int, n_samples: int,
                              bounds: Optional[np.ndarray] = None,
                              seed: Optional[int] = None) -> np.ndarray:
    """
    Latin Hypercube Sampling (LHS)：比纯随机采样更均匀地覆盖设计空间。

    算法：
        将每维分成 n_samples 个等概率区间，
        在每个区间随机采样，并通过随机排列保证投影均匀性。
    """
    if seed is not None:
        np.random.seed(seed)
    samples = np.zeros((n_samples, d), dtype=np.float64)
    for dim in range(d):
        # 随机排列区间
        perm = np.random.permutation(n_samples)
        # 在每个区间内均匀随机
        u = (perm + np.random.rand(n_samples)) / n_samples
        samples[:, dim] = u
    if bounds is not None:
        samples = bounds[:, 0] + samples * (bounds[:, 1] - bounds[:, 0])
    return samples


# =============================================================================
# 4. Sobol 灵敏度分析（一阶指标）
# =============================================================================

def sobol_first_order_indices(func: Callable[[np.ndarray], float],
                               d: int, n_samples: int = 1024,
                               bounds: Optional[np.ndarray] = None) -> np.ndarray:
    """
    基于 Saltelli 采样方案的一阶 Sobol 灵敏度指数。

    方差分解：
        Var(Y) = Σ_i V_i + Σ_{i<j} V_{ij} + ...
        S_i = V_i / Var(Y)

    其中 V_i = Var_{X_i}(E_{X_{~i}}[Y | X_i])

    这里采用简化的蒙特卡洛估计：
        S_i ≈ [ (1/N) Σ_j f(B_j)(f(A_j^i) - f(A_j)) ] / Var(f(A))
    A, B 为两组独立 LHS 样本，A^i 为 A 的第 i 维替换为 B 的第 i 维。
    """
    A = latin_hypercube_sampling(d, n_samples, bounds, seed=42)
    B = latin_hypercube_sampling(d, n_samples, bounds, seed=43)

    f_A = np.array([func(a) for a in A])
    f_B = np.array([func(b) for b in B])
    var_y = np.var(f_A)
    if var_y < 1e-14:
        return np.zeros(d)

    S1 = np.zeros(d)
    for i in range(d):
        A_Bi = A.copy()
        A_Bi[:, i] = B[:, i]
        f_ABi = np.array([func(x) for x in A_Bi])
        # Jansen 估计器
        S1[i] = np.mean(f_B * (f_ABi - f_A)) / var_y
        # 截断到合理范围
        S1[i] = max(0.0, min(1.0, S1[i]))
    return S1


# =============================================================================
# 5. 增材制造参数不确定性模型
# =============================================================================

def generate_am_process_parameters(n_samples: int,
                                    seed: Optional[int] = None) -> np.ndarray:
    """
    生成增材制造工艺参数的不确定性样本。
    参数（归一化到 [0,1]）：
        [激光功率波动, 扫描速度波动, 层厚波动, 预热温度波动, 粉末粒径波动]
    假设各参数服从截断高斯或均匀分布。
    """
    d = 5
    samples = latin_hypercube_sampling(d, n_samples, seed=seed)
    return samples


def parameter_to_physical(sample: np.ndarray) -> dict:
    """
    将归一化样本映射到实际物理参数。
    """
    return {
        "laser_power_var": 0.9 + 0.2 * sample[0],       # 标称值的 ±10%
        "scan_speed_var": 0.9 + 0.2 * sample[1],
        "layer_thickness_var": 0.85 + 0.3 * sample[2],
        "preheat_temp_var": 0.95 + 0.1 * sample[3],
        "powder_size_var": 0.8 + 0.4 * sample[4],
    }
