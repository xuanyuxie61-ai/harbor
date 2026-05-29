"""
monte_carlo_ensemble.py

基于 683_line_monte_carlo 和 1177_subset_distance 核心算法的
蒙特卡洛集合模拟与距离分析模块。

原项目 line_monte_carlo 提供了线段上的蒙特卡洛采样和单项式积分；
subset_distance 提供了子集 Hamming 距离及其统计量。

在本气候归因框架中，该模块用于：
1. 气候模型集合的随机采样与扰动生成
2. 集合成员间的 Hamming 距离分析，量化不同初始条件/参数设置下
   极端事件归因结果的一致性
3. 通过蒙特卡洛积分估计归因概率的置信区间

核心公式：
- 线段上均匀随机采样：x ~ U([0, 1])
- 蒙特卡洛积分：∫_0^1 f(x) dx ≈ (1/N) Σ f(x_k)
- Hamming 距离：d_H(A, B) = |A Δ B| = Σ_i |1_A(i) - 1_B(i)|
- 子集距离统计量：
    μ = E[d_H],  σ^2 = Var(d_H)
- 归因概率的蒙特卡洛估计：
    P_attribution = #{ensemble members with event > threshold} / N_total
"""

import numpy as np


def line01_sample_random(n, seed=None):
    """
    在单位线段 [0,1] 上均匀随机采样（基于 683_line01_sample_random）。
    """
    rng = np.random.default_rng(seed)
    return rng.random(n)


def line01_monomial_integral(e):
    """
    单项式 x^e 在 [0,1] 上的精确积分（基于 683_line01_monomial_integral）。

    公式：∫_0^1 x^e dx = 1 / (e + 1)，e ≠ -1
    """
    if e == -1:
        raise ValueError("e = -1 不允许")
    return 1.0 / (e + 1)


def monte_carlo_line_integral(fun, n, seed=None):
    """
    线段 [0,1] 上的蒙特卡洛积分。
    """
    x = line01_sample_random(n, seed)
    fx = fun(x)
    return float(np.mean(fx))


def subset_distance_hamming(t1, t2):
    """
    两个子集的 Hamming 距离（基于 1177_subset_distance_hamming）。

    Parameters
    ----------
    t1, t2 : ndarray, shape (m,)
        0/1 指示向量。
    """
    t1 = np.asarray(t1)
    t2 = np.asarray(t2)
    if t1.shape != t2.shape:
        raise ValueError("t1 和 t2 形状必须相同")
    return int(np.sum(t1 != t2))


def subset_sample(m, rng=None):
    """
    随机生成一个 m 元集合的子集（基于 1177_subset_sample）。
    """
    if rng is None:
        rng = np.random.default_rng()
    return (rng.random(m) > 0.5).astype(np.int64)


def subset_distance_stats(m, n_samples=1000, seed=None):
    """
    估计子集距离的统计量（基于 1177_subset_distance_stats）。

    Returns
    -------
    mu, var : float
        均值和方差。
    """
    rng = np.random.default_rng(seed)
    distances = []
    for _ in range(n_samples):
        s1 = subset_sample(m, rng)
        s2 = subset_sample(m, rng)
        distances.append(subset_distance_hamming(s1, s2))
    distances = np.array(distances, dtype=np.float64)
    mu = float(np.mean(distances))
    if n_samples > 1:
        var = float(np.sum((distances - mu) ** 2) / (n_samples - 1))
    else:
        var = 0.0
    return mu, var


def generate_ensemble_perturbations(base_field, n_members, perturbation_scale=0.1,
                                     seed=None):
    """
    生成气候场集合扰动。

    Parameters
    ----------
    base_field : ndarray
        基础气候场。
    n_members : int
        集合成员数。
    perturbation_scale : float
        扰动幅度（相对于场标准差）。

    Returns
    -------
    ensemble : ndarray, shape (n_members, *base_field.shape)
    """
    rng = np.random.default_rng(seed)
    std = np.std(base_field)
    if std < 1e-14:
        std = 1.0
    ensemble = np.zeros((n_members,) + base_field.shape, dtype=np.float64)
    for i in range(n_members):
        noise = rng.normal(0.0, perturbation_scale * std, base_field.shape)
        ensemble[i] = base_field + noise
    return ensemble


def ensemble_attribution_distance(ensemble_binary, threshold_ratio=0.5):
    """
    计算集合成员间极端事件空间模式的平均 Hamming 距离。

    Parameters
    ----------
    ensemble_binary : ndarray, shape (n_members, m, n)
        二值极端事件场。
    threshold_ratio : float
        判定为"显著归因"的集合比例阈值。

    Returns
    -------
    mean_dist : float
        平均 pairwise Hamming 距离。
    consensus_mask : ndarray
        集合共识掩膜（超过 threshold_ratio 成员同意的格点）。
    """
    n_members = ensemble_binary.shape[0]
    flat = ensemble_binary.reshape(n_members, -1)
    m = flat.shape[1]

    # pairwise Hamming 距离
    total_dist = 0
    count = 0
    for i in range(n_members):
        for j in range(i + 1, n_members):
            total_dist += subset_distance_hamming(flat[i], flat[j])
            count += 1

    mean_dist = total_dist / count if count > 0 else 0.0

    # 共识掩膜
    consensus = np.mean(flat, axis=0)
    consensus_mask = (consensus >= threshold_ratio).astype(np.int64)

    return mean_dist, consensus_mask.reshape(ensemble_binary.shape[1:])


def test_monte_carlo():
    val = monte_carlo_line_integral(lambda x: x ** 2, 10000, seed=42)
    exact = line01_monomial_integral(2)
    assert abs(val - exact) < 0.05
    mu, var = subset_distance_stats(10, 500, seed=42)
    assert mu >= 0
    print("monte_carlo_ensemble 自测试通过")


if __name__ == "__main__":
    test_monte_carlo()
