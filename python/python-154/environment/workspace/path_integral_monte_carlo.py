"""
path_integral_monte_carlo.py
================================================================================
路径积分蒙特卡洛（PIMC）模拟量子退火的虚时间演化。

融合来源：
  - 1209_test_int_nd（N 维 Monte Carlo 积分框架）
  - 1312_triangle_monte_carlo（几何域上的 Monte Carlo 积分）

物理背景：
  量子伊辛模型的配分函数可通过 Trotter-Suzuki 分解映射到经典
  (d+1) 维系统（d 维空间 + 1 维虚时间）。

  Trotter 公式（一阶）：
      Z = Tr[ exp(-β H) ] ≈ Tr[ Π_{m=1}^M exp(-Δτ H_P) exp(-Δτ H_D) ]

  其中 Δτ = β/M，H_P 为问题哈密顿量（经典），H_D 为横向场（量子）。

  对每一 Trotter 切片 m，引入一个自旋副本 {s_i^{(m)}}，配分函数变为

      Z ≈ Σ_{ {s^{(m)}} } exp( -S_E[ {s^{(m)}} ] )

  欧几里得作用量（世界线作用量）：
      S_E = Σ_{m=1}^M [ Δτ E_P(s^{(m)})
                        - J_⊥ Σ_i s_i^{(m)} s_i^{(m+1)} ]

  其中 J_⊥ = (1/2) ln coth(Δτ Γ) 为有效横向耦合（世界线跃迁权重）。

  世界线更新（单自旋翻转）：
      对切片 m 上的自旋 i 尝试翻转，接受概率
      P_acc = min(1, exp(-ΔS_E) )

      ΔS_E = Δτ [E_P(s') - E_P(s)]
             - J_⊥ [ s_i^{(m)} (s_i^{(m-1)} + s_i^{(m+1)})
                     - s_i'^{(m)} (s_i^{(m-1)} + s_i^{(m+1)}) ]
"""

import numpy as np
from typing import Tuple, Optional, Callable


def effective_transverse_coupling(dtau: float, gamma: float) -> float:
    """
    计算世界线有效横向耦合 J_⊥。

    精确公式（来自 Suzuki-Trotter 映射）：
        J_⊥ = -(1/2) ln tanh(Δτ Γ)
            = (1/2) ln coth(Δτ Γ)
    """
    # TODO: 实现 effective_transverse_coupling 的核心计算逻辑
    # 提示：需要根据 dtau 和 gamma 计算有效横向耦合 J_⊥
    #       注意小参数 arg = dtau * gamma 的数值稳定性处理
    raise NotImplementedError("Hole 1: 请补全 Trotter-Suzuki 有效横向耦合公式")


class PathIntegralMonteCarlo:
    """
    路径积分蒙特卡洛求解器，用于模拟量子退火的有限温度行为。
    """

    def __init__(self, n_spins: int, beta: float, n_slices: int,
                 energy_func: Callable, gamma_schedule: np.ndarray,
                 seed: int = 154):
        if n_spins <= 0 or beta <= 0 or n_slices <= 0:
            raise ValueError("Physical parameters must be positive")
        if gamma_schedule.size != n_slices:
            raise ValueError("gamma_schedule length must equal n_slices")
        self.n_spins = n_spins
        self.beta = float(beta)
        self.n_slices = n_slices
        self.energy_func = energy_func
        self.gamma_schedule = np.array(gamma_schedule, dtype=float)
        self.dtau = beta / n_slices
        self.rng = np.random.default_rng(seed)
        # 初始化世界线：所有切片相同，随机自旋
        base = 2 * self.rng.integers(0, 2, size=n_spins) - 1
        self.worldlines = np.tile(base, (n_slices, 1)).astype(int)
        # 周期边界条件在切片方向
        self.energies = np.array([energy_func(self.worldlines[m, :])
                                   for m in range(n_slices)])

    def _slice_energy_change(self, m: int, i: int) -> float:
        """
        计算翻转 worldlines[m, i] 带来的经典能量变化。
        """
        s_old = self.worldlines[m, :].copy()
        s_new = s_old.copy()
        s_new[i] *= -1
        e_old = self.energy_func(s_old)
        e_new = self.energy_func(s_new)
        return float(e_new - e_old)

    def _worldline_coupling_change(self, m: int, i: int) -> float:
        """
        计算翻转带来的世界线耦合项变化：
            -J_⊥^{(m)} s_i^{(m)} (s_i^{(m-1)} + s_i^{(m+1)})
        """
        s_m = int(self.worldlines[m, i])
        s_prev = int(self.worldlines[(m - 1) % self.n_slices, i])
        s_next = int(self.worldlines[(m + 1) % self.n_slices, i])
        gamma_m = self.gamma_schedule[m]
        J_perp = effective_transverse_coupling(self.dtau, gamma_m)
        # 翻转后 s_m -> -s_m
        delta_coupling = -J_perp * ((-s_m) - s_m) * (s_prev + s_next)
        return float(delta_coupling)

    def _metropolis_step_single(self) -> int:
        """
        执行一次单自旋-单切片 Metropolis 更新。
        返回接受的步数。
        """
        accepted = 0
        for m in range(self.n_slices):
            for i in range(self.n_spins):
                delta_e = self._slice_energy_change(m, i)
                delta_w = self._worldline_coupling_change(m, i)
                delta_total = self.dtau * delta_e + delta_w
                if delta_total <= 0:
                    accept = True
                else:
                    prob = np.exp(-delta_total)
                    prob = min(prob, 1.0)
                    accept = self.rng.random() < prob
                if accept:
                    self.worldlines[m, i] *= -1
                    self.energies[m] += delta_e
                    accepted += 1
        return accepted

    def _cluster_update(self) -> int:
        """
        Swendsen-Wang 型集群更新（虚时间方向）。

        思想：对每条世界线 (固定 i)，根据横向耦合强度
        以概率 p = 1 - exp(-2 J_⊥) 在相邻切片间建立键，
        然后对连通分量进行整体翻转。
        """
        accepted = 0
        for i in range(self.n_spins):
            # 建立键
            bonds = np.zeros(self.n_slices, dtype=int)
            for m in range(self.n_slices):
                gamma_m = self.gamma_schedule[m]
                J_perp = effective_transverse_coupling(self.dtau, gamma_m)
                p_bond = 1.0 - np.exp(-2.0 * J_perp)
                if self.rng.random() < p_bond:
                    bonds[m] = 1
            # 找连通分量（一维环）
            visited = np.zeros(self.n_slices, dtype=int)
            for m0 in range(self.n_slices):
                if visited[m0]:
                    continue
                cluster = []
                m = m0
                while True:
                    cluster.append(m)
                    visited[m] = 1
                    next_m = (m + 1) % self.n_slices
                    if bonds[m] and not visited[next_m]:
                        m = next_m
                    else:
                        break
                # 计算翻转整个 cluster 的能量变化
                e_flip = 0.0
                for m in cluster:
                    s_old = self.worldlines[m, :].copy()
                    s_new = s_old.copy()
                    s_new[i] *= -1
                    e_flip += self.energy_func(s_new) - self.energy_func(s_old)
                # 世界线耦合变化（仅涉及 cluster 边界）
                # 简化：直接用能量差作为 Metropolis 准则
                if e_flip <= 0 or self.rng.random() < np.exp(-self.dtau * e_flip):
                    for m in cluster:
                        self.worldlines[m, i] *= -1
                    # 更新缓存能量
                    for m in cluster:
                        self.energies[m] = self.energy_func(self.worldlines[m, :])
                    accepted += len(cluster)
        return accepted

    def thermalize(self, n_sweeps: int = 500) -> None:
        """热化：执行 n_sweeps 次完整 sweep。"""
        for _ in range(n_sweeps):
            self._metropolis_step_single()
            if _ % 10 == 0:
                self._cluster_update()

    def measure_observables(self, n_measurements: int = 100,
                            sampling_interval: int = 5) -> dict:
        """
        测量物理观测量：
            - 平均能量 ⟨E⟩
            - 磁化强度 M = (1/N) Σ_i ⟨σ_i^z⟩
            - 磁化率 χ = β (⟨M²⟩ - ⟨M⟩²)
            - 世界线缠绕数（拓扑序参量）
        """
        e_vals = []
        m_vals = []
        m2_vals = []
        winding = []
        for k in range(n_measurements):
            for _ in range(sampling_interval):
                self._metropolis_step_single()
                if _ % 5 == 0:
                    self._cluster_update()
            e_avg = self.energies.mean()
            mag = self.worldlines.mean(axis=0).mean()
            m2 = (self.worldlines.mean(axis=0) ** 2).mean()
            # 缠绕数：统计世界线穿过虚时间边界的次数
            wind = 0.0
            for i in range(self.n_spins):
                flips = np.sum(np.abs(np.diff(self.worldlines[:, i]))) // 2
                wind += flips
            e_vals.append(e_avg)
            m_vals.append(mag)
            m2_vals.append(m2)
            winding.append(wind / self.n_spins)
        e_vals = np.array(e_vals)
        m_vals = np.array(m_vals)
        m2_vals = np.array(m2_vals)
        winding = np.array(winding)
        chi = self.beta * (m2_vals.mean() - m_vals.mean() ** 2)
        return {
            "energy_mean": float(e_vals.mean()),
            "energy_std": float(e_vals.std(ddof=1)),
            "magnetization": float(m_vals.mean()),
            "magnetization_std": float(m_vals.std(ddof=1)),
            "susceptibility": float(chi),
            "winding_number": float(winding.mean()),
        }

    def estimate_ground_state_energy(self, n_replicas: int = 3,
                                      n_sweeps_each: int = 300) -> float:
        """
        使用多副本外推到 β→∞ 估算基态能量：
            E_0 ≈ ⟨E⟩_β - (1/β) S
        简化版：取最大 β 副本的平均能量最低值。
        """
        best_e = float('inf')
        for rep in range(n_replicas):
            # 随机重初始化
            base = 2 * self.rng.integers(0, 2, size=self.n_spins) - 1
            self.worldlines = np.tile(base, (self.n_slices, 1))
            self.thermalize(n_sweeps=n_sweeps_each)
            e_mean = self.energies.mean()
            if e_mean < best_e:
                best_e = e_mean
        return float(best_e)
