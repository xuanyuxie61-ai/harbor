"""
monte_carlo_averaging.py — 蒙特卡洛无序平均与拉丁超立方采样
==============================================================

本模块实现拓扑绝缘体边界态性质的蒙特卡洛无序平均:

1. **标准蒙特卡洛**: 随机采样无序构型, 计算系综平均
2. **拉丁超立方采样 (LHS)**: 参数空间的分层采样
3. **最优停止理论**: 自适应采样数目的确定

物理问题
--------
无序拓扑绝缘体的系综平均性质:
    ⟨G⟩ = (1/N_samples) Σ_i G[V_i]

其中 G[V_i] 是第 i 个无序构型 V_i 下的电导。

收敛判据: 标准误差 σ/√N < tol

来源映射
--------
- 533_high_card_parfor: 蒙特卡洛模拟框架
- 652_latin_random: 拉丁超立方采样
- 1373_uniform: 线性同余发生器 (可重复随机序列)
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional


class LinearCongruentialGenerator:
    """
    线性同余发生器 (LCG), 用于可重复的随机序列。

    X_{n+1} = (a × X_n + c) mod m

    标准参数 (Numerical Recipes):
        a = 1664525, c = 1013904223, m = 2^32

    借鉴 1373_uniform 的 LCG 实现。
    """

    def __init__(self, seed: int = 42,
                 a: int = 1664525, c: int = 1013904223,
                 m: int = 2**32):
        self.a = a
        self.c = c
        self.m = m
        self.state = seed % m
        self.initial_state = self.state

    def next_int(self) -> int:
        self.state = (self.a * self.state + self.c) % self.m
        return self.state

    def next_float(self) -> float:
        return self.next_int() / self.m

    def next_normal(self) -> float:
        """Box-Muller 变换生成标准正态随机数。"""
        u1 = max(self.next_float(), 1e-15)
        u2 = self.next_float()
        return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)

    def next_array(self, size: int) -> np.ndarray:
        return np.array([self.next_float() for _ in range(size)])

    def normal_array(self, size: int) -> np.ndarray:
        return np.array([self.next_normal() for _ in range(size)])

    def reset(self):
        self.state = self.initial_state


def latin_hypercube_sample(dim: int, n_samples: int,
                           rng: np.random.RandomState = None
                           ) -> np.ndarray:
    """
    拉丁超立方采样 (Latin Hypercube Sampling, LHS)。

    在 d 维空间中生成 n 个样本, 每维被等分为 n 个区间,
    每个区间中恰好有一个样本。

    算法:
    对每维 j = 1, ..., d:
        1. 生成随机排列 π  of {1, ..., n}
        2. 在第 π(i) 个区间中均匀随机取点:
           x_{i,j} = (π(i) - 1 + U(0,1)) / n

    LHS 保证每维的边际分布均匀, 比简单随机采样更高效。

    借鉴 652_latin_random 的实现。

    Parameters
    ----------
    dim : int
        维度
    n_samples : int
        样本数
    rng : RandomState

    Returns
    -------
    samples : ndarray, shape (n_samples, dim)
        在 [0, 1]^dim 中的样本
    """
    if rng is None:
        rng = np.random.RandomState(42)

    samples = np.zeros((n_samples, dim))
    for d in range(dim):
        perm = rng.permutation(n_samples)
        for i in range(n_samples):
            samples[i, d] = (perm[i] + rng.rand()) / n_samples

    return samples


def lhs_to_physical(samples: np.ndarray,
                    param_ranges: Dict[str, Tuple[float, float]]
                    ) -> List[Dict]:
    """
    将 LHS 样本 [0,1]^d 映射到物理参数空间。

    Parameters
    ----------
    samples : ndarray, shape (N, d)
    param_ranges : dict
        参数名 → (min, max) 的映射

    Returns
    -------
    configs : list of dict
    """
    N, d = samples.shape
    param_names = list(param_ranges.keys())

    if d != len(param_names):
        raise ValueError(f"维度不匹配: 样本 {d} 维, 参数 {len(param_names)} 个")

    configs = []
    for i in range(N):
        config = {}
        for j, name in enumerate(param_names):
            lo, hi = param_ranges[name]
            config[name] = lo + samples[i, j] * (hi - lo)
        configs.append(config)

    return configs


def monte_carlo_disorder_average(compute_observable: Callable,
                                 disorder_params: Dict,
                                 n_samples: int = 50,
                                 seed: int = 42,
                                 use_lhs: bool = False,
                                 lhs_params: Dict = None) -> Dict:
    """
    蒙特卡洛无序平均的统一框架。

    对 n_samples 个无序构型计算可观测量, 返回系综平均和统计量。

    自适应停止 (基于最优停止理论的 1/e 法则):
        如果连续 k 个样本的平均值变化 < tol, 提前停止。

    Parameters
    ----------
    compute_observable : callable
        输入 disorder config dict, 返回标量可观测量
    disorder_params : dict
        无序参数 (W, xi, corr_type 等)
    n_samples : int
        最大样本数
    seed : int
    use_lhs : bool
        是否使用 LHS (对参数扫描)
    lhs_params : dict
        LHS 参数范围

    Returns
    -------
    result : dict
    """
    rng = np.random.RandomState(seed)
    lcg = LinearCongruentialGenerator(seed)

    values = []
    running_mean = []
    running_std = []

    early_stop_threshold = 5  # 连续 k 步变化 < tol 则停止
    tol = 1e-4

    for n in range(n_samples):
        # 生成无序构型
        if use_lhs and lhs_params is not None:
            # LHS 模式: 使用预生成的样本
            lhs_samples = latin_hypercube_sample(
                len(lhs_params), n_samples, rng
            )
            configs = lhs_to_physical(lhs_samples, lhs_params)
            config = configs[n]
            # 将 LHS 参数作为额外参数
            W = config.get('W', disorder_params.get('W', 0.1))
            xi = config.get('xi', disorder_params.get('xi', 1.0))
        else:
            W = disorder_params.get('W', 0.1)
            xi = disorder_params.get('xi', 1.0)

        # 使用 LCG 种子确保可重复性
        disorder_seed = lcg.next_int() % (2**31)

        obs_value = compute_observable(
            W=W, xi=xi,
            seed=disorder_seed,
            **{k: v for k, v in disorder_params.items()
               if k not in ('W', 'xi')}
        )

        values.append(obs_value)
        values_arr = np.array(values)

        running_mean.append(float(np.mean(values_arr)))
        running_std.append(float(np.std(values_arr)) if len(values) > 1 else 0.0)

        # 自适应停止检查
        if n >= early_stop_threshold:
            recent_means = running_mean[-early_stop_threshold:]
            max_variation = max(recent_means) - min(recent_means)
            if max_variation < tol:
                break

    values_arr = np.array(values)

    return {
        'values': values,
        'mean': float(np.mean(values_arr)),
        'std': float(np.std(values_arr)),
        'std_error': float(np.std(values_arr) / np.sqrt(len(values_arr))),
        'min': float(np.min(values_arr)),
        'max': float(np.max(values_arr)),
        'n_samples': len(values),
        'n_max': n_samples,
        'running_mean': running_mean,
        'running_std': running_std,
        'seed': seed,
        'method': 'LHS' if use_lhs else 'standard_MC'
    }


def optimal_stopping_sample_size(sigma: float, tol: float,
                                 confidence: float = 0.95) -> int:
    """
    基于最优停止理论估计所需样本数。

    要达到标准误差 < tol (置信度 95%):
        N > (z_{α/2} × σ / tol)²

    其中 z_{0.025} = 1.96。

    借鉴 533_high_card_parfor 的最优停止策略。

    Parameters
    ----------
    sigma : float
        估计的标准差
    tol : float
        目标标准误差
    confidence : float

    Returns
    -------
    N : int
    """
    z = 1.96 if confidence >= 0.95 else 1.645
    N = int(np.ceil((z * sigma / tol) ** 2))
    return max(N, 10)


def convergence_analysis_report(mc_result: Dict) -> str:
    """生成蒙特卡洛收敛分析报告。"""
    lines = []
    lines.append("=" * 60)
    lines.append("蒙特卡洛无序平均收敛分析报告")
    lines.append("=" * 60)
    lines.append(f"采样方法: {mc_result['method']}")
    lines.append(f"样本数: {mc_result['n_samples']} / {mc_result['n_max']}")
    lines.append(f"随机种子: {mc_result['seed']}")

    lines.append(f"\n统计结果:")
    lines.append(f"  系综平均 ⟨O⟩ = {mc_result['mean']:.8f}")
    lines.append(f"  标准差 σ = {mc_result['std']:.8f}")
    lines.append(f"  标准误差 σ/√N = {mc_result['std_error']:.8f}")
    lines.append(f"  最小值 = {mc_result['min']:.8f}")
    lines.append(f"  最大值 = {mc_result['max']:.8f}")

    # 收敛分析
    rm = mc_result['running_mean']
    if len(rm) > 5:
        lines.append(f"\n收敛历史:")
        indices = [0, len(rm)//4, len(rm)//2, 3*len(rm)//4, len(rm)-1]
        for idx in indices:
            lines.append(f"  N={idx+1:4d}: ⟨O⟩ = {rm[idx]:.8f}")

        # 收敛速率
        if len(rm) > 1:
            final_err = abs(rm[-1] - rm[-2])
            lines.append(f"\n  最终步变化: {final_err:.6e}")
            lines.append(f"  收敛: {'是' if final_err < 1e-4 else '否'}")

    # 最优停止建议
    if mc_result['std'] > 0:
        N_opt = optimal_stopping_sample_size(mc_result['std'], 1e-4)
        lines.append(f"\n  达到 σ/√N < 1e-4 所需样本数: {N_opt}")

    return "\n".join(lines)


def parameter_scan_mc(compute_observable: Callable,
                      base_params: Dict,
                      scan_param: str,
                      scan_values: np.ndarray,
                      n_samples_per: int = 20,
                      seed: int = 42) -> Dict:
    """
    沿一个参数方向扫描, 每个点进行蒙特卡洛平均。

    用于生成相图, 研究参数依赖。

    Parameters
    ----------
    compute_observable : callable
    base_params : dict
    scan_param : str
    scan_values : ndarray
    n_samples_per : int
    seed : int

    Returns
    -------
    result : dict
    """
    means = []
    stds = []
    std_errors = []

    for val in scan_values:
        params = base_params.copy()
        params[scan_param] = val

        mc = monte_carlo_disorder_average(
            compute_observable, params,
            n_samples=n_samples_per,
            seed=seed
        )
        means.append(mc['mean'])
        stds.append(mc['std'])
        std_errors.append(mc['std_error'])

    return {
        'scan_param': scan_param,
        'scan_values': scan_values,
        'means': np.array(means),
        'stds': np.array(stds),
        'std_errors': np.array(std_errors),
        'n_samples_per': n_samples_per
    }
