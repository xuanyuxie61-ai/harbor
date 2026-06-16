"""
umbrella_sampling.py
====================

Umbrella Sampling 与 WHAM 分析。

物理背景
--------
自旋玻璃的 overlap 序参量 q 的分布 P(q) 包含丰富的物理信息:

- 在顺磁相 (T > T_c): P(q) = delta(q) (单峰)
- 在自旋玻璃相 (T < T_c): P(q) 有非平凡结构
  * 平均场 (SK 模型): P(q) 在 [q_EA, 0] 上连续分布
  * 有限维 (d=3): P(q) 的形状仍有争议 (droplet vs RSB)

直接 MC 采样 P(q) 的困难: 在 P(q) 很小的区域 (如 q=0 附近),
系统很少被访问, 统计很差。

Umbrella Sampling 解决方案
---------------------------
在多个 q 窗口中添加偏置势:
    H_bias = H + (kappa/2) * (q - q_0)^2

其中 kappa 是弹性常数, q_0 是窗口的中心。

每个窗口的偏置采样给出:
    P_bias(q) ∝ P(q) * exp(-beta * (kappa/2) * (q - q_0)^2)

WHAM (Weighted Histogram Analysis Method)
------------------------------------------
合并 R 个窗口的数据:
    P(q) = sum_{k=1}^{R} N_k(q) / sum_{j=1}^{R} n_j * exp(-beta * W_j(q))

其中:
    W_j(q) = (kappa_j / 2) * (q - q_0_j)^2 - F_j
    F_j = -beta^{-1} * ln integral P(q) * exp(-beta * W_j(q)) dq

F_j 通过自洽迭代求解。

本模块核心算法来源于 seed project:
- 1155_anhkiet120206-lgtm (EEIO Analysis): 输入输出分析中的
  约束优化方法
  (映射为 WHAM 的自洽方程求解)
- 1287_keb721_NucleiMorphology (umbrella_sampling): 伞形采样
  (直接映射为 q-space 的伞形采样)
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from spin_lattice_geometry import CubicLattice3D, total_energy
from monte_carlo_core import MetropolisMC, overlap


# =====================================================================
#  Umbrella Sampling MC
# =====================================================================

class UmbrellaSampler:
    """
    在 overlap q 空间上进行 Umbrella Sampling。

    偏置势:
        V_bias(q) = (kappa / 2) * (q - q_0)^2

    参数
    ----
    lattice : CubicLattice3D
    couplings : dict
    beta : float
    q0 : float
        偏置势的中心
    kappa : float
        弹性常数 (控制偏置强度)
    reference_spins : ndarray
        参考副本 (用于计算 q)
    rng : Generator
    """

    def __init__(self, lattice: CubicLattice3D,
                 couplings: Dict[Tuple[int, int], float],
                 beta: float,
                 q0: float,
                 kappa: float,
                 reference_spins: np.ndarray,
                 rng: Optional[np.random.Generator] = None):
        self.lattice = lattice
        self.couplings = couplings
        self.beta = beta
        self.q0 = q0
        self.kappa = kappa
        self.reference_spins = reference_spins
        self.rng = rng if rng is not None else np.random.default_rng()

    def bias_energy(self, spins: np.ndarray) -> float:
        """
        偏置势:
            V_bias = (kappa / 2) * (q - q0)^2
        """
        q = overlap(spins, self.reference_spins)
        return 0.5 * self.kappa * (q - self.q0) ** 2

    def delta_bias_energy(self, spins: np.ndarray,
                          site: int, old_spin: float) -> float:
        """
        翻转 S_i 后的偏置势变化:
            Delta V_bias = (kappa/2) * [(q_new - q0)^2 - (q_old - q0)^2]
        """
        q_old = overlap(spins, self.reference_spins)
        # 翻转后的新 overlap
        delta_q = -2.0 * old_spin * self.reference_spins.flat[site] / self.lattice.N
        q_new = q_old + delta_q
        return 0.5 * self.kappa * ((q_new - self.q0) ** 2 - (q_old - self.q0) ** 2)

    def single_sweep(self, spins: np.ndarray) -> np.ndarray:
        """
        一个带偏置的 MC sweep。
        """
        spins = spins.copy()
        N = self.lattice.N
        sites = self.rng.permutation(N)

        for site in sites:
            # 原始哈密顿量的 Delta E
            s_i = spins.flat[site]
            h_eff = 0.0
            for nbr in self.lattice.neighbor_table[site]:
                if nbr < 0:
                    continue
                key = (min(site, nbr), max(site, nbr))
                J = self.couplings.get(key, 0.0)
                h_eff += J * spins.flat[nbr]
            delta_E_phys = 2.0 * s_i * h_eff

            # 偏置势的 Delta V
            delta_V_bias = self.delta_bias_energy(spins, site, s_i)

            # 总 Delta
            delta_total = delta_E_phys + delta_V_bias

            # Metropolis 判据
            if delta_total <= 0:
                accept = True
            else:
                x = -self.beta * delta_total
                accept = self.rng.random() < np.exp(x) if x > -500 else False

            if accept:
                spins.flat[site] *= -1

        return spins

    def run(self, spins_init: np.ndarray,
            n_sweeps: int, n_equil: int = 50
            ) -> Dict[str, np.ndarray]:
        """
        运行 umbrella sampling 模拟。
        """
        spins = spins_init.copy()

        # 平衡化
        for _ in range(n_equil):
            spins = self.single_sweep(spins)

        # 采样
        q_series = np.zeros(n_sweeps)
        E_series = np.zeros(n_sweeps)
        V_bias_series = np.zeros(n_sweeps)

        for step in range(n_sweeps):
            spins = self.single_sweep(spins)
            q_series[step] = overlap(spins, self.reference_spins)
            E_series[step] = total_energy(spins, self.couplings, self.lattice) / self.lattice.N
            V_bias_series[step] = self.bias_energy(spins)

        return {
            "q_samples": q_series,
            "energy_samples": E_series,
            "bias_energy_samples": V_bias_series,
            "q0": self.q0,
            "kappa": self.kappa,
        }


# =====================================================================
#  WHAM 分析
# =====================================================================

class WHAMAnalyzer:
    """
    Weighted Histogram Analysis Method (WHAM).

    合并 R 个 umbrella 窗口的数据, 获得无偏的 P(q)。

    参数
    ----
    windows : list of dict
        每个窗口的采样结果
    q_bins : ndarray
        q 的直方图 bins
    beta : float
    tolerance : float
        自洽迭代收敛阈值
    max_iter : int
    """

    def __init__(self, windows: List[Dict],
                 q_bins: np.ndarray,
                 beta: float,
                 tolerance: float = 1e-8,
                 max_iter: int = 1000):
        self.windows = windows
        self.q_bins = q_bins
        self.beta = beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.R = len(windows)

    def compute_histograms(self) -> List[np.ndarray]:
        """
        计算每个窗口的 q 直方图。
        """
        histograms = []
        for w in self.windows:
            hist, _ = np.histogram(w["q_samples"], bins=self.q_bins)
            histograms.append(hist.astype(np.float64))
        return histograms

    def bias_potential(self, window_idx: int, q_val: float) -> float:
        """
        第 window_idx 个窗口在 q_val 处的偏置势:
            W_j(q) = (kappa_j / 2) * (q - q0_j)^2
        """
        w = self.windows[window_idx]
        return 0.5 * w["kappa"] * (q_val - w["q0"]) ** 2

    def solve(self) -> Dict[str, np.ndarray]:
        """
        自洽求解 WHAM 方程:

        P(q) = sum_k N_k(q) / sum_j n_j * exp(-beta * (W_j(q) - F_j))

        F_j = -beta^{-1} * ln integral P(q) * exp(-beta * W_j(q)) dq

        返回
        ----
        result : dict
            "q_centers": q bins 的中心
            "P_q": 无偏的 P(q)
            "F": 自由能 F_j
            "free_energy": F(q) = -beta^{-1} * ln P(q)
        """
        histograms = self.compute_histograms()
        n_samples = np.array([len(w["q_samples"]) for w in self.windows])
        q_centers = 0.5 * (self.q_bins[:-1] + self.q_bins[1:])
        n_bins = len(q_centers)

        # 初始化 F_j = 0
        F = np.zeros(self.R)

        for iteration in range(self.max_iter):
            # 计算 P(q)
            numerator = np.sum(histograms, axis=0)
            denominator = np.zeros(n_bins)

            for j in range(self.R):
                W_j = np.array([self.bias_potential(j, q) for q in q_centers])
                denominator += n_samples[j] * np.exp(-self.beta * (W_j - F[j]))

            denominator = np.maximum(denominator, 1e-300)
            P_q = numerator / denominator
            P_q = np.maximum(P_q, 1e-300)  # 避免 log(0)

            # 归一化
            dq = self.q_bins[1] - self.q_bins[0]
            P_q /= (np.sum(P_q) * dq)

            # 更新 F_j
            F_new = np.zeros(self.R)
            for j in range(self.R):
                W_j = np.array([self.bias_potential(j, q) for q in q_centers])
                integrand = P_q * np.exp(-self.beta * W_j)
                integral = np.sum(integrand) * dq
                if integral > 1e-300:
                    F_new[j] = -np.log(integral) / self.beta
                else:
                    F_new[j] = F[j]

            # 检查收敛
            dF = np.max(np.abs(F_new - F))
            F = F_new
            if dF < self.tolerance:
                break

        # 自由能 F(q)
        free_energy = -np.log(P_q) / self.beta
        free_energy -= np.min(free_energy)  # 归一化使最小值为 0

        return {
            "q_centers": q_centers,
            "P_q": P_q,
            "F_windows": F,
            "free_energy": free_energy,
            "n_iterations": iteration + 1,
            "converged": dF < self.tolerance,
        }


# =====================================================================
#  多窗口 Umbrella Sampling 运行器
# =====================================================================

def run_umbrella_sampling_windows(lattice: CubicLattice3D,
                                  couplings: Dict[Tuple[int, int], float],
                                  beta: float,
                                  q0_list: np.ndarray,
                                  kappa: float = 50.0,
                                  n_sweeps: int = 200,
                                  n_equil: int = 50,
                                  seed: int = 42
                                  ) -> List[Dict]:
    """
    在多个 q 窗口上运行 umbrella sampling。
    """
    rng = np.random.default_rng(seed)

    # 生成参考副本
    ref_spins = rng.choice([-1.0, +1.0], size=lattice.N).reshape(
        (lattice.L,) * 3
    )

    windows = []
    for q0 in q0_list:
        sampler = UmbrellaSampler(
            lattice, couplings, beta, q0, kappa, ref_spins, rng=rng
        )
        # 初始配置: 与参考副本 overlap 接近 q0
        spins_init = ref_spins.copy()
        if q0 < 0.5:
            # 随机翻转一些自旋以降低 overlap
            n_flip = int((1.0 - q0) * lattice.N / 2)
            flip_sites = rng.choice(lattice.N, size=min(n_flip, lattice.N), replace=False)
            for site in flip_sites:
                spins_init.flat[site] *= -1

        result = sampler.run(spins_init, n_sweeps, n_equil)
        windows.append(result)

    return windows
