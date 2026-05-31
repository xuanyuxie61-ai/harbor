"""
state_sampler.py
================================================================================
自旋构型采样器：Gray 码枚举、重要性采样与马尔可夫链蒙特卡洛。

融合来源：
  - 622_knapsack_01_brute（Gray 码子集枚举）
  - 585_image_sample（坐标/状态采样框架）
  - 779_monty_hall_simulation（概率决策与条件概率模拟）

物理背景：
  量子退火的状态空间维度随自旋数指数增长（2^N）。对于中等规模系统，
  精确枚举（Gray 码）不可行，需借助采样技术估算配分函数与基态性质。

  1) Gray 码枚举：相邻状态仅差 1 位，可用于精确计算小系统（N≤20）的
     配分函数 Z = Σ_s exp(-β E(s))。

  2) Metropolis-Hastings 马尔可夫链：对大规模系统，通过局部自旋翻转
     构造遍历马尔可夫链，收敛到 Boltzmann 分布 π(s) ∝ exp(-β E(s))。

  3) 多副本模拟（Monty Hall 条件概率思想）：利用已知信息（揭示的局部
     能量景观）指导采样策略，类似于 Monty Hall 问题中的信息更新。
"""

import numpy as np
from typing import Tuple, List, Callable, Optional


def gray_code_sequence(n: int) -> List[np.ndarray]:
    """
    生成 n 位 Gray 码序列，返回长度为 2^n 的二进制向量列表。

    Gray 码递归构造：
        G(1) = [0, 1]
        G(n) = [0 || G(n-1), 1 || reverse(G(n-1))]
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return [np.array([], dtype=int)]
    if n > 20:
        raise ValueError("Gray code enumeration too large for n > 20")
    seq = []
    for g in range(2 ** n):
        gray = g ^ (g >> 1)
        bits = np.array([(gray >> i) & 1 for i in range(n)], dtype=int)
        seq.append(bits)
    return seq


def enumerate_all_energies(energy_func: Callable, n_spins: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 Gray 码枚举计算所有 2^n 个自旋构型的能量。

    返回：
        configs: shape (2^n, n_spins)，元素为 +/-1
        energies: shape (2^n,)
    """
    seq = gray_code_sequence(n_spins)
    configs = np.array([2 * bits - 1 for bits in seq], dtype=int)
    energies = np.array([energy_func(c) for c in configs], dtype=float)
    return configs, energies


def exact_partition_function(energies: np.ndarray, beta: float) -> float:
    """
    精确配分函数：
        Z(β) = Σ_i exp(-β E_i)

    数值稳定性：使用 log-sum-exp 技巧。
    """
    if beta < 0:
        raise ValueError("beta must be non-negative")
    e_min = energies.min()
    shifted = -beta * (energies - e_min)
    # 裁剪防止上溢
    shifted = np.clip(shifted, -700, 700)
    log_z = -beta * e_min + np.log(np.sum(np.exp(shifted)))
    return float(np.exp(log_z))


def exact_thermal_average(energies: np.ndarray, beta: float,
                          observables: np.ndarray) -> float:
    """
    精确计算热力学期望值：
        ⟨O⟩ = Σ_i O_i exp(-β E_i) / Z(β)
    """
    if observables.shape != energies.shape:
        raise ValueError("observables shape must match energies")
    e_min = energies.min()
    shifted = -beta * (energies - e_min)
    shifted = np.clip(shifted, -700, 700)
    weights = np.exp(shifted)
    z = weights.sum()
    if z < 1e-300:
        return 0.0
    return float(np.dot(weights, observables) / z)


class MetropolisSampler:
    """
    Metropolis-Hastings 马尔可夫链蒙特卡洛采样器。
    """

    def __init__(self, n_spins: int, energy_func: Callable,
                 beta: float = 1.0, seed: int = 154):
        if n_spins <= 0:
            raise ValueError("n_spins must be positive")
        self.n_spins = n_spins
        self.energy_func = energy_func
        self.beta = float(beta)
        self.rng = np.random.default_rng(seed)
        self.state = 2 * self.rng.integers(0, 2, size=n_spins) - 1
        self.e_curr = self.energy_func(self.state)

    def sweep(self) -> Tuple[np.ndarray, float]:
        """
        执行一次完整 sweep（对每个自旋尝试翻转一次）。
        """
        for i in range(self.n_spins):
            self.state[i] *= -1
            e_new = self.energy_func(self.state)
            delta = e_new - self.e_curr
            if delta <= 0:
                accept = True
            else:
                prob = np.exp(-self.beta * delta)
                prob = min(prob, 1.0)
                accept = self.rng.random() < prob
            if accept:
                self.e_curr = e_new
            else:
                self.state[i] *= -1
        return self.state.copy(), self.e_curr

    def sample(self, n_sweeps: int, burn_in: int = 100,
               thinning: int = 10) -> dict:
        """
        采集样本序列，包含退火（burn-in）与稀释（thinning）。
        """
        # burn-in
        for _ in range(burn_in):
            self.sweep()
        states = []
        energies = []
        for k in range(n_sweeps):
            self.sweep()
            if k % thinning == 0:
                states.append(self.state.copy())
                energies.append(self.e_curr)
        return {
            "states": np.array(states),
            "energies": np.array(energies),
        }


class ParallelTemperingSampler:
    """
    并行回火（Replica Exchange Monte Carlo）采样器。

    物理思想：在多个温度 β_1 > β_2 > ... > β_M 上并行运行 MCMC，
    以一定概率交换相邻温度的构型，帮助低温副本跨越能量势垒。
    """

    def __init__(self, n_spins: int, energy_func: Callable,
                 betas: np.ndarray, seed: int = 154):
        if n_spins <= 0:
            raise ValueError("n_spins must be positive")
        if len(betas) < 2:
            raise ValueError("need at least 2 temperatures for parallel tempering")
        self.n_spins = n_spins
        self.energy_func = energy_func
        self.betas = np.array(betas, dtype=float)
        self.n_replicas = len(betas)
        self.rng = np.random.default_rng(seed)
        self.states = np.array([
            2 * self.rng.integers(0, 2, size=n_spins) - 1
            for _ in range(self.n_replicas)
        ])
        self.energies = np.array([energy_func(s) for s in self.states])

    def replica_exchange_step(self) -> None:
        """
        尝试相邻副本间的交换。

        交换接受概率（详细平衡条件）：
            P_acc = min(1, exp( (β_i - β_j)(E_i - E_j) ))
        """
        for i in range(self.n_replicas - 1):
            j = i + 1
            delta = (self.betas[i] - self.betas[j]) * (self.energies[i] - self.energies[j])
            if delta >= 0 or self.rng.random() < np.exp(min(delta, 0.0)):
                # 交换
                self.states[[i, j], :] = self.states[[j, i], :]
                self.energies[[i, j]] = self.energies[[j, i]]

    def local_update_step(self) -> None:
        """对每个副本执行一次 Metropolis sweep。"""
        for r in range(self.n_replicas):
            sampler = MetropolisSampler(self.n_spins, self.energy_func,
                                        self.betas[r], seed=self.rng.integers(0, 2 ** 31))
            sampler.state = self.states[r].copy()
            sampler.e_curr = self.energies[r]
            sampler.sweep()
            self.states[r] = sampler.state
            self.energies[r] = sampler.e_curr

    def sample(self, n_steps: int, exchange_freq: int = 5) -> dict:
        """
        采集样本。
        """
        all_states = []
        all_energies = []
        for step in range(n_steps):
            self.local_update_step()
            if step % exchange_freq == 0:
                self.replica_exchange_step()
            all_states.append(self.states.copy())
            all_energies.append(self.energies.copy())
        return {
            "states": np.array(all_states),
            "energies": np.array(all_energies),
            "betas": self.betas.copy(),
        }


class ConditionalProbabilitySampler:
    """
    条件概率采样器（Monty Hall 信息更新思想的推广）。

    在量子退火中，当我们观测到一部分自旋的取值时，可以更新剩余自旋
    的条件分布，并据此进行有偏采样。
    """

    def __init__(self, n_spins: int, energy_func: Callable,
                 seed: int = 154):
        self.n_spins = n_spins
        self.energy_func = energy_func
        self.rng = np.random.default_rng(seed)

    def sample_given_partial(self, fixed_spins: dict, n_samples: int = 100,
                             beta: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        在固定部分自旋的条件下采样其余自旋。

        fixed_spins: dict {index: value (+1 or -1)}
        """
        free_indices = [i for i in range(self.n_spins) if i not in fixed_spins]
        n_free = len(free_indices)
        samples = []
        energies = []
        for _ in range(n_samples):
            s = np.zeros(self.n_spins, dtype=int)
            for idx, val in fixed_spins.items():
                s[idx] = val
            # 对自由自旋随机初始化
            s[free_indices] = 2 * self.rng.integers(0, 2, size=n_free) - 1
            # 局部 Metropolis 松弛
            for _ in range(50):
                i = free_indices[self.rng.integers(0, n_free)]
                s[i] *= -1
                e_new = self.energy_func(s)
                s[i] *= -1
                e_old = self.energy_func(s)
                delta = e_new - e_old
                if delta <= 0 or self.rng.random() < np.exp(-beta * delta):
                    s[i] *= -1
            samples.append(s.copy())
            energies.append(self.energy_func(s))
        return np.array(samples), np.array(energies)
