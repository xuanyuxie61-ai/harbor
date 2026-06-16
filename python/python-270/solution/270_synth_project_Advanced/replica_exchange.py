"""
replica_exchange.py
===================

Replica Exchange (Parallel Tempering) Monte Carlo。

物理背景
--------
自旋玻璃在低温下具有极其复杂的能量景观 (rugged energy landscape),
标准 Metropolis MC 会被困在局部极小中, 导致严重的临界慢化
(critical slowing down)。

Replica Exchange (Parallel Tempering) 通过并行模拟多个温度副本
来增强采样:

1. 在温度 T_1 < T_2 < ... < T_R 上并行运行 R 个副本
2. 周期性地在相邻温度 (T_i, T_{i+1}) 之间尝试交换:
       (S^{(i)}, T_i) <-> (S^{(i+1)}, T_{i+1})
3. 交换接受概率:
       p_accept = min(1, exp(Delta))
   其中:
       Delta = (beta_i - beta_{i+1}) * (E_{i+1} - E_i)

最优温度分配
------------
几何级数: T_k = T_min * (T_max / T_min)^{(k-1)/(R-1)}
或基于等接受率准则:
    Delta_beta_k * sqrt(C_V(T_k)) * N = const

交换率
------
最优交换率约 20-30%, 对应:
    |Delta_beta| * sqrt(Var(E)) ≈ 1

本模块核心算法来源于 seed project:
- 264_cvtp (cvtp_iteration, cvtp_find_closest): CVT 迭代
  (映射为温度空间的优化分配)
- 246_cvt_1d_sampling: Lloyd 算法
  (映射为温度副本的自适应调节)
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from spin_lattice_geometry import CubicLattice3D, total_energy
from monte_carlo_core import MetropolisMC


# =====================================================================
#  温度分配
# =====================================================================

def geometric_temperature_ladder(T_min: float, T_max: float,
                                 R: int) -> np.ndarray:
    """
    几何级数温度阶梯:
        T_k = T_min * (T_max / T_min)^{(k-1)/(R-1)},  k = 1, ..., R

    参数
    ----
    T_min : float
        最低温度
    T_max : float
        最高温度
    R : int
        副本数

    返回
    ----
    temperatures : ndarray (R,)
    """
    if T_min <= 0:
        raise ValueError(f"T_min 必须 > 0, 当前 T_min={T_min}")
    if T_max <= T_min:
        raise ValueError(f"T_max 必须 > T_min")
    if R < 2:
        raise ValueError(f"至少需要 R=2 个副本")

    ratio = (T_max / T_min) ** (1.0 / (R - 1))
    temperatures = T_min * ratio ** np.arange(R)
    return temperatures


def optimal_temperature_ladder(T_min: float, T_max: float,
                               R: int, N: int,
                               J_var: float = 1.0,
                               z: int = 6) -> np.ndarray:
    """
    基于等接受率的最优温度分配。

    原理: 在平均场近似下, 能量方差:
        Var(E) ≈ N * z * J_var^2 * beta^2 / 2

    等接受率条件:
        (beta_k - beta_{k+1}) * sqrt(Var(E_k)) = const

    这给出 beta 的非均匀间隔:
        Delta_beta_k ∝ 1 / sqrt(Var(E_k))

    参数
    ----
    T_min : float
    T_max : float
    R : int
    N : int
        格点数
    J_var : float
    z : int

    返回
    ----
    temperatures : ndarray (R,)
    """
    betas = np.linspace(1.0 / T_max, 1.0 / T_min, R)

    # 迭代优化
    for _ in range(10):
        var_E = np.array([
            N * z * J_var ** 2 * b ** 2 / 2.0
            for b in betas
        ])
        sigma_E = np.sqrt(var_E)

        # 等接受率条件: Delta_beta_k * sigma_E_k = const
        total_integral = betas[-1] - betas[0]
        weights = 1.0 / (sigma_E + 1e-15)
        weights /= np.sum(weights)
        new_betas = betas[0] + total_integral * np.cumsum(weights)
        new_betas[-1] = betas[-1]
        betas = new_betas

    temperatures = 1.0 / betas
    return temperatures


# =====================================================================
#  Replica Exchange MC
# =====================================================================

class ReplicaExchangeMC:
    """
    Parallel Tempering Monte Carlo。

    参数
    ----
    lattice : CubicLattice3D
    couplings : dict
    temperatures : ndarray (R,)
    n_sweeps_per_exchange : int
        每 n 个 MC sweep 尝试一次交换
    rng : Generator
    """

    def __init__(self, lattice: CubicLattice3D,
                 couplings: Dict[Tuple[int, int], float],
                 temperatures: np.ndarray,
                 n_sweeps_per_exchange: int = 10,
                 rng: Optional[np.random.Generator] = None):
        self.lattice = lattice
        self.couplings = couplings
        self.temperatures = np.sort(temperatures)
        self.R = len(temperatures)
        self.n_sweeps_per_exchange = n_sweeps_per_exchange
        self.rng = rng if rng is not None else np.random.default_rng()

        if self.R < 2:
            raise ValueError("至少需要 2 个副本")

        self.betas = 1.0 / self.temperatures

        # 初始化副本
        self.replicas = []
        self.mcs = []
        for r in range(self.R):
            s = self.rng.choice([-1.0, +1.0], size=lattice.N).reshape(
                (lattice.L,) * 3
            )
            self.replicas.append(s)
            self.mcs.append(MetropolisMC(lattice, self.betas[r], rng=self.rng))

        # 统计
        self.n_exchange_proposed = 0
        self.n_exchange_accepted = 0
        self.exchange_history = []

    def exchange_acceptance_probability(self, r1: int, r2: int) -> float:
        """
        计算相邻副本 r1, r2 的交换接受概率:
            Delta = (beta_{r1} - beta_{r2}) * (E_{r2} - E_{r1})
            p = min(1, exp(Delta))
        """
        E1 = total_energy(self.replicas[r1], self.couplings, self.lattice)
        E2 = total_energy(self.replicas[r2], self.couplings, self.lattice)

        delta_beta = self.betas[r1] - self.betas[r2]
        delta_E = E2 - E1
        Delta = delta_beta * delta_E

        # 数值保护
        if Delta >= 0:
            return 1.0
        elif Delta < -500:
            return 0.0
        else:
            return float(np.exp(Delta))

    def attempt_exchange(self, r1: int, r2: int) -> bool:
        """
        尝试交换副本 r1 和 r2 的自旋配置。
        """
        p = self.exchange_acceptance_probability(r1, r2)
        self.n_exchange_proposed += 1

        if self.rng.random() < p:
            # 交换配置
            self.replicas[r1], self.replicas[r2] = (
                self.replicas[r2].copy(), self.replicas[r1].copy()
            )
            self.n_exchange_accepted += 1
            return True
        return False

    def sweep_with_exchanges(self) -> Dict[str, float]:
        """
        执行一轮 MC sweep + 交换尝试。
        """
        # 每个副本执行 n_sweeps_per_exchange 个 MC sweep
        energies = np.zeros(self.R)
        for r in range(self.R):
            for _ in range(self.n_sweeps_per_exchange):
                self.replicas[r] = self.mcs[r].single_sweep(
                    self.replicas[r], self.couplings
                )
            energies[r] = total_energy(
                self.replicas[r], self.couplings, self.lattice
            ) / self.lattice.N

        # 随机选择交换方向 (偶数对或奇数对)
        if self.rng.random() < 0.5:
            pairs = [(r, r + 1) for r in range(0, self.R - 1, 2)]
        else:
            pairs = [(r, r + 1) for r in range(1, self.R - 1, 2)]

        exchange_results = []
        for (r1, r2) in pairs:
            accepted = self.attempt_exchange(r1, r2)
            exchange_results.append((r1, r2, accepted))

        self.exchange_history.append(exchange_results)

        return {
            "energies": energies,
            "n_exchanges_proposed": len(pairs),
            "n_exchanges_accepted": sum(1 for _, _, a in exchange_results if a),
        }

    def exchange_ratio(self) -> float:
        """总交换接受率"""
        if self.n_exchange_proposed == 0:
            return 0.0
        return self.n_exchange_accepted / self.n_exchange_proposed

    def run(self, n_rounds: int) -> Dict[str, np.ndarray]:
        """
        运行 n_rounds 轮 replica exchange。

        返回
        ----
        results : dict
            每个温度的能量时间序列、交换统计
        """
        energy_series = {r: np.zeros(n_rounds) for r in range(self.R)}
        exchange_rate_series = np.zeros(n_rounds)

        for round_idx in range(n_rounds):
            result = self.sweep_with_exchanges()
            for r in range(self.R):
                energy_series[r][round_idx] = result["energies"][r]
            exchange_rate_series[round_idx] = (
                result["n_exchanges_accepted"] /
                max(1, result["n_exchanges_proposed"])
            )

        return {
            "energies": energy_series,
            "temperatures": self.temperatures.copy(),
            "betas": self.betas.copy(),
            "exchange_rate_series": exchange_rate_series,
            "overall_exchange_ratio": self.exchange_ratio(),
        }


# =====================================================================
#  基于 CVT 的温度空间优化
# =====================================================================

def cvt_temperature_optimization(T_min: float, T_max: float,
                                  R: int, N: int,
                                  J_var: float = 1.0,
                                  z: int = 6,
                                  n_iterations: int = 50
                                  ) -> np.ndarray:
    """
    使用 Lloyd 算法 (CVT 迭代) 优化温度分配。

    将温度空间 [1/T_max, 1/T_min] 分割为 R 个 Voronoi 区域,
    使得每个区域的 "权重" (基于能量方差) 相等。

    权重函数:
        rho(beta) = sqrt(Var(E(beta))) ∝ beta * sqrt(N * z * J_var^2 / 2)

    参数
    ----
    T_min, T_max : float
    R : int
    N : int
    J_var : float
    z : int
    n_iterations : int

    返回
    ----
    temperatures : ndarray (R,)
    """
    beta_min = 1.0 / T_max
    beta_max = 1.0 / T_min

    # 初始: 均匀分布在 beta 空间
    betas = np.linspace(beta_min, beta_max, R)

    for _ in range(n_iterations):
        # 计算每个 beta 处的权重
        weights = np.array([
            b * np.sqrt(N * z * J_var ** 2 / 2.0)
            for b in betas
        ])

        # Lloyd 迭代: 将每个生成器移动到其 Voronoi 区域的质心
        new_betas = np.zeros(R)
        for k in range(R):
            # Voronoi 边界
            if k == 0:
                left = beta_min
            else:
                left = 0.5 * (betas[k - 1] + betas[k])
            if k == R - 1:
                right = beta_max
            else:
                right = 0.5 * (betas[k] + betas[k + 1])

            # 加权质心
            n_sample = 100
            beta_samples = np.linspace(left, right, n_sample)
            w_samples = beta_samples * np.sqrt(N * z * J_var ** 2 / 2.0)
            new_betas[k] = np.sum(beta_samples * w_samples) / np.sum(w_samples)

        betas = np.sort(new_betas)

    return 1.0 / betas
