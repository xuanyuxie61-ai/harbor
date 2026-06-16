"""
monte_carlo_core.py
===================

Metropolis Monte Carlo 核心算法与可观测量的测量。

物理背景
--------
对于 Ising 自旋玻璃, Metropolis-Hastings 算法:

1. 随机选取一个自旋 S_i
2. 计算能量变化 Delta E:
       Delta E = 2 * S_i * sum_{j in nbr(i)} J_{ij} * S_j
3. 以概率 p_accept 翻转 S_i:
       p_accept = min(1, exp(-beta * Delta E))
   等价地: 若 Delta E <= 0, 一定接受; 否则以 exp(-beta*Delta E) 概率接受。

一个 Monte Carlo Sweep (MCS) = N 次单自旋尝试翻转

可观测量的测量
--------------
1. 能量密度:
       e = <H> / N
2. 磁化强度:
       m = <|sum_i S_i|> / N
3. 自旋玻璃序参量 (Edwards-Anderson):
       q_EA = <(1/N) * sum_i <S_i>_T^2>
   在两次独立副本 (replica) 中:
       q = (1/N) * sum_i S_i^{(1)} * S_i^{(2)}
4. Binder 累积量:
       g = (1/2) * (3 - <q^4> / <q^2>^2)
   在 T_c 处, g 对不同 L 的曲线交叉
5. 自旋玻璃磁化率:
       chi_SG = N * <q^2>
6. 自旋-自旋关联函数:
       C(r) = <S_0 * S_r>
7. 比热:
       C_V = beta^2 * (<H^2> - <H>^2) / N

本模块核心算法来源于 seed project:
- 041_asa136 (kmns, optra, qtran): K-means 聚类算法
  (映射为对 replica overlap 分布的聚类, 识别纯态)
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from spin_lattice_geometry import CubicLattice3D, total_energy, magnetization


# =====================================================================
#  Metropolis Monte Carlo
# =====================================================================

class MetropolisMC:
    """
    Ising 自旋玻璃的 Metropolis Monte Carlo 模拟。

    参数
    ----
    lattice : CubicLattice3D
    beta : float
        逆温度 beta = 1/T
    rng : numpy.random.Generator or None
    """

    def __init__(self, lattice: CubicLattice3D,
                 beta: float,
                 rng: Optional[np.random.Generator] = None):
        if beta < 0:
            raise ValueError(f"beta 不能为负: beta={beta}")
        self.lattice = lattice
        self.beta = beta
        self.rng = rng if rng is not None else np.random.default_rng()
        self.N = lattice.N
        self.n_accepted = 0
        self.n_proposed = 0

    def acceptance_probability(self, delta_E: float) -> float:
        """
        Metropolis 接受概率:
            p = min(1, exp(-beta * Delta E))
        """
        if delta_E <= 0:
            return 1.0
        x = -self.beta * delta_E
        if x < -500:
            return 0.0
        return float(np.exp(x))

    def delta_energy(self, spins: np.ndarray,
                     couplings: Dict[Tuple[int, int], float],
                     site: int) -> float:
        """
        翻转 S_i 的能量变化:
            Delta E = E(S_i -> -S_i) - E(S_i)
                    = 2 * S_i * sum_{j in nbr(i)} J_{ij} * S_j
        """
        s_i = spins.flat[site]
        h_eff = 0.0
        for nbr in self.lattice.neighbor_table[site]:
            if nbr < 0:
                continue
            key = (min(site, nbr), max(site, nbr))
            J = couplings.get(key, 0.0)
            h_eff += J * spins.flat[nbr]
        return 2.0 * s_i * h_eff

    def single_sweep(self, spins: np.ndarray,
                     couplings: Dict[Tuple[int, int], float]) -> np.ndarray:
        """
        执行一个 Monte Carlo Sweep (N 次尝试翻转)。
        使用随机序列选择翻转位点 (random sequential update)。
        """
        spins = spins.copy()
        sites = self.rng.permutation(self.N)

        for site in sites:
            dE = self.delta_energy(spins, couplings, site)
            p = self.acceptance_probability(dE)
            if self.rng.random() < p:
                spins.flat[site] *= -1
                self.n_accepted += 1
            self.n_proposed += 1

        return spins

    def acceptance_ratio(self) -> float:
        """当前接受率"""
        if self.n_proposed == 0:
            return 0.0
        return self.n_accepted / self.n_proposed

    def reset_counters(self):
        """重置接受/提议计数器"""
        self.n_accepted = 0
        self.n_proposed = 0


# =====================================================================
#  可观测量的测量
# =====================================================================

def energy_density(spins: np.ndarray,
                   couplings: Dict[Tuple[int, int], float],
                   lattice: CubicLattice3D) -> float:
    """
    能量密度 e = H / N
    """
    return total_energy(spins, couplings, lattice) / lattice.N


def magnetization_density_obs(spins: np.ndarray) -> float:
    """
    磁化强度密度 |m| = |sum S_i| / N
    """
    return abs(magnetization(spins)) / spins.size


def overlap(spins1: np.ndarray, spins2: np.ndarray) -> float:
    """
    两个自旋配置的 overlap (序参量):
        q = (1/N) * sum_i S_i^{(1)} * S_i^{(2)}

    这是自旋玻璃理论中的核心量 (Parisi 序参量)。
    """
    if spins1.shape != spins2.shape:
        raise ValueError("两个配置的形状必须相同")
    return float(np.mean(spins1 * spins2))


def overlap_squared(spins1: np.ndarray, spins2: np.ndarray) -> float:
    """q^2"""
    q = overlap(spins1, spins2)
    return q * q


def binder_cumulant(q_samples: np.ndarray) -> float:
    """
    Binder 累积量:
        g = (1/2) * (3 - <q^4> / <q^2>^2)

    在 T_c 处, g 对不同 L 的曲线交叉于 universal value g* ≈ 0.63
    (3D Ising universality class)

    参数
    ----
    q_samples : ndarray
        多个 MCS 下测量的 q 值序列
    """
    if len(q_samples) < 2:
        return 0.0
    q2 = np.mean(q_samples ** 2)
    q4 = np.mean(q_samples ** 4)
    if q2 < 1e-15:
        return 0.0
    return 0.5 * (3.0 - q4 / (q2 ** 2))


def chiq_susceptibility(q_samples: np.ndarray, N: int) -> float:
    """
    自旋玻璃磁化率:
        chi_SG = N * <q^2>
    """
    return N * np.mean(q_samples ** 2)


def specific_heat(energy_samples: np.ndarray, beta: float, N: int) -> float:
    """
    比热:
        C_V = beta^2 * (<H^2> - <H>^2) / N
            = beta^2 * Var(H) / N
    """
    if len(energy_samples) < 2:
        return 0.0
    var_H = np.var(energy_samples, ddof=1)
    return beta ** 2 * var_H / N


def autocorrelation_function(series: np.ndarray,
                             max_lag: Optional[int] = None) -> np.ndarray:
    """
    自相关函数:
        C(t) = <A(0) * A(t)> - <A>^2
              / (<A^2> - <A>^2)

    参数
    ----
    series : ndarray (T,)
        时间序列
    max_lag : int or None
        最大滞后, 默认为 len(series)//4

    返回
    ----
    acf : ndarray (max_lag+1,)
        归一化自相关函数
    """
    T = len(series)
    if max_lag is None:
        max_lag = T // 4
    max_lag = min(max_lag, T - 1)

    mean_A = np.mean(series)
    var_A = np.var(series)
    if var_A < 1e-15:
        return np.zeros(max_lag + 1)

    acf = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag == 0:
            acf[lag] = 1.0
        else:
            c = np.mean((series[:T - lag] - mean_A) * (series[lag:] - mean_A))
            acf[lag] = c / var_A
    return acf


def integrated_autocorrelation_time(acf: np.ndarray) -> float:
    """
    积分自相关时间:
        tau_int = 1/2 + sum_{t=1}^{max_lag} C(t)

    有效独立样本数: N_eff = N_samples / (2 * tau_int)
    """
    return 0.5 + np.sum(acf[1:])


# =====================================================================
#  K-means 聚类分析 (纯态识别)
# =====================================================================

def kmeans_overlap_clustering(q_matrix: np.ndarray,
                              n_clusters: int = 2,
                              max_iter: int = 100,
                              rng: Optional[np.random.Generator] = None
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对 replica overlap 矩阵进行 K-means 聚类,
    识别自旋玻璃的不同纯态 (pure states)。

    在 Parisi 平均场理论中, 自旋玻璃相有无穷多个纯态,
    其 overlap 分布 P(q) 是非平凡的。

    参数
    ----
    q_matrix : ndarray (n_replicas, n_replicas)
        所有 replica 之间的 overlap 矩阵
    n_clusters : int
        聚类数
    max_iter : int
        最大迭代
    rng : Generator

    返回
    ----
    centers : ndarray (n_clusters, n_replicas)
        聚类中心
    labels : ndarray (n_replicas,)
        每个 replica 的标签
    wss : ndarray (n_clusters,)
        每个簇的 within-cluster sum of squares

    本算法来源于 seed project 041_asa136 (kmns):
        Algorithm AS 136: A K-Means Clustering Algorithm
        Hartigan & Wong, Applied Statistics, 1979
    """
    if rng is None:
        rng = np.random.default_rng()

    n = q_matrix.shape[0]
    if n_clusters > n:
        n_clusters = n

    # 用 overlap 矩阵的行作为特征向量
    features = q_matrix.copy()

    # 初始化: 随机选择 n_clusters 个不重复的样本作为初始中心
    init_indices = rng.choice(n, size=n_clusters, replace=False)
    centers = features[init_indices].copy()

    labels = np.zeros(n, dtype=np.int64)
    wss = np.zeros(n_clusters, dtype=np.float64)

    for iteration in range(max_iter):
        # 分配: 每个点归入最近的中心
        new_labels = np.zeros(n, dtype=np.int64)
        for i in range(n):
            dists = np.array([np.sum((features[i] - centers[k]) ** 2)
                              for k in range(n_clusters)])
            new_labels[i] = np.argmin(dists)

        # 检查收敛
        if np.all(new_labels == labels) and iteration > 0:
            break
        labels = new_labels

        # 更新中心
        for k in range(n_clusters):
            members = features[labels == k]
            if len(members) > 0:
                centers[k] = np.mean(members, axis=0)

    # 计算 WSS
    for k in range(n_clusters):
        members = features[labels == k]
        if len(members) > 0:
            wss[k] = np.sum((members - centers[k]) ** 2)

    return centers, labels, wss


# =====================================================================
#  完整 MC 模拟运行器
# =====================================================================

def run_mc_simulation(lattice: CubicLattice3D,
                      couplings: Dict[Tuple[int, int], float],
                      beta: float,
                      n_sweeps: int = 500,
                      n_equil: int = 100,
                      n_replicas: int = 2,
                      seed: int = 42
                      ) -> Dict[str, np.ndarray]:
    """
    运行完整的 Monte Carlo 模拟, 测量所有可观测量。

    参数
    ----
    lattice : CubicLattice3D
    couplings : dict
        耦合常数
    beta : float
        逆温度
    n_sweeps : int
        测量 sweep 数
    n_equil : int
        平衡化 sweep 数
    n_replicas : int
        副本数 (用于计算 overlap)
    seed : int
        随机种子

    返回
    ----
    results : dict
        包含所有可观测量的时间序列
    """
    rng = np.random.default_rng(seed)
    mc = MetropolisMC(lattice, beta, rng=rng)

    # 初始化 n_replicas 个独立副本 (随机自旋配置)
    replicas = []
    for _ in range(n_replicas):
        s = rng.choice([-1.0, +1.0], size=lattice.N).reshape((lattice.L,) * 3)
        replicas.append(s)

    # 平衡化
    for sweep_idx in range(n_equil):
        for r in range(n_replicas):
            replicas[r] = mc.single_sweep(replicas[r], couplings)

    mc.reset_counters()

    # 测量
    energy_series = np.zeros(n_sweeps)
    mag_series = np.zeros(n_sweeps)
    q_series = np.zeros(n_sweeps)
    q2_series = np.zeros(n_sweeps)

    for sweep_idx in range(n_sweeps):
        for r in range(n_replicas):
            replicas[r] = mc.single_sweep(replicas[r], couplings)

        # 测量能量和磁化 (第一个副本)
        E = total_energy(replicas[0], couplings, lattice)
        energy_series[sweep_idx] = E / lattice.N
        mag_series[sweep_idx] = abs(magnetization(replicas[0])) / lattice.N

        # 测量 overlap (副本 0 和副本 1)
        if n_replicas >= 2:
            q = overlap(replicas[0], replicas[1])
            q_series[sweep_idx] = q
            q2_series[sweep_idx] = q * q

    # 计算统计量
    results = {
        "energy_density": energy_series,
        "magnetization_density": mag_series,
        "overlap": q_series,
        "overlap_squared": q2_series,
        "beta": beta,
        "n_sweeps": n_sweeps,
        "acceptance_ratio": mc.acceptance_ratio(),
        "mean_energy": np.mean(energy_series),
        "var_energy": np.var(energy_series),
        "mean_mag": np.mean(mag_series),
        "mean_q": np.mean(q_series),
        "mean_q2": np.mean(q2_series),
        "binder_cumulant": binder_cumulant(q_series),
        "chi_SG": chiq_susceptibility(q_series, lattice.N),
        "C_V": specific_heat(energy_series * lattice.N, beta, lattice.N),
    }

    # 自相关分析
    if n_sweeps > 10:
        acf_E = autocorrelation_function(energy_series)
        results["autocorr_energy"] = acf_E
        results["tau_int_energy"] = integrated_autocorrelation_time(acf_E)

        if n_replicas >= 2:
            acf_q = autocorrelation_function(q_series)
            results["autocorr_q"] = acf_q
            results["tau_int_q"] = integrated_autocorrelation_time(acf_q)

    return results
