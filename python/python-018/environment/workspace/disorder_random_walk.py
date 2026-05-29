"""
disorder_random_walk.py

基于种子项目 1010_random_walk_2d_simulation（2D随机游走）,
模拟马约拉纳费米子在含有随机杂质的Kitaev链中的Andreev反射
与空间扩散过程。

物理模型：
    在无序Kitaev链中，化学势受到随机扰动：
        μ_i = μ_0 + W * ω_i
    其中W为无序强度，ω_i ∈ [-1, 1]为均匀随机变量。

    马约拉纳零能模的波函数在实空间中的扩散可类比为
    随机游走过程，但受到超导能隙的约束。Andreev反射
    概率由以下公式给出：
        P_A = |Δ|^2 / (|Δ|^2 + E^2)
    对于零能模，P_A → 1，意味着完全Andreev反射。
"""

import numpy as np
from typing import Tuple, List, Optional


class MajoranaDisorderRandomWalk:
    """
    马约拉纳费米子在无序拓扑超导体中的随机游走模拟器。

    将2D随机游走映射到一维Kitaev链的空间扩散：
        - x方向：实空间格点位置
        - y方向：Andreev反射引起的相位积累

    每一步游走代表一个相干散射事件，包括：
        1. 正常散射（透射或反射）
        2. Andreev反射（电子↔空穴转换）
    """

    def __init__(self, n_sites: int, disorder_strength: float,
                 delta: float, t: float = 1.0, mu0: float = 0.0,
                 rng_seed: Optional[int] = None):
        """
        初始化无序随机游走模拟器。

        Args:
            n_sites: 格点数
            disorder_strength: 无序强度 W
            delta: 超导配对势
            t: 跃迁强度
            mu0: 平均化学势
            rng_seed: 随机数种子
        """
        self.n = n_sites
        self.W = disorder_strength
        self.delta = delta
        self.t = t
        self.mu0 = mu0

        self.rng = np.random.RandomState(rng_seed)
        self._generate_disorder_potential()

    def _generate_disorder_potential(self) -> None:
        """
        生成随机无序势场。

         disorder[i] = W * (2*rand() - 1)
        使得每个格点的化学势为 μ_i = μ0 + disorder[i]
        """
        self.disorder = self.W * (2.0 * self.rng.rand(self.n) - 1.0)
        self.mu_profile = self.mu0 + self.disorder

    def _andreev_reflection_probability(self, energy: float) -> float:
        """
        计算Andreev反射概率。

        对于能量为E的准粒子入射到正常-超导界面：
            P_A(E) = |Δ|^2 / (|E|^2 + |Δ|^2)

        在Blonder-Tinkham-Klapwijk (BTK) 理论框架下，
        零偏压时（E=0）的Andreev反射概率为1，对应
        量子化电导 G = 2e^2/h。
        """
        if abs(self.delta) < 1e-15:
            return 0.0
        return (self.delta ** 2) / (energy ** 2 + self.delta ** 2 + 1e-15)

    def _localization_length_estimate(self, site: int) -> float:
        """
        估计局域化长度。

        在弱无序极限下，Kitaev链中马约拉纳零能模的
        局域化长度近似为：
            ξ ≈ v_F / E_gap ≈ 2t / |Δ|

        当无序存在时，有效局域化长度减小：
            ξ_eff ≈ ξ / (1 + (W/Δ)^2)
        """
        xi0 = 2.0 * abs(self.t) / (abs(self.delta) + 1e-15)
        reduction = 1.0 + (self.W / (abs(self.delta) + 1e-15)) ** 2
        return xi0 / reduction

    def simulate_andreev_random_walk(self, num_steps: int,
                                     num_walks: int,
                                     energy: float = 0.0) -> Tuple[np.ndarray,
                                                                    np.ndarray]:
        """
        模拟马约拉纳费米子的Andreev反射随机游走。

        算法（基于2D随机游走的映射）：
            状态由 (position, phase) 描述
            每个时间步：
                - 以概率 P_A 发生Andreev反射（相位反转）
                - 以概率 1-P_A 发生正常散射（位置变化±1）

        Args:
            num_steps: 每个walker的时间步数
            num_walks: walker数量
            energy: 准粒子能量

        Returns:
            d2_ave: 平均位移平方 <r^2>(t)
            d2_max: 最大位移平方 max(r^2)(t)
        """
        if num_steps < 1 or num_walks < 1:
            raise ValueError("步数和walker数必须为正")

        p_andreev = self._andreev_reflection_probability(energy)
        p_normal = 1.0 - p_andreev

        d2_ave = np.zeros(num_steps + 1)
        d2_max = np.zeros(num_steps + 1)

        for walk in range(num_walks):
            x = 0.0  # 实空间位置
            y = 0.0  # 相位积累（类比2D游走的y坐标）

            for step in range(1, num_steps + 1):
                r = self.rng.rand()
                if r < p_andreev:
                    # Andreev反射：相位翻转，位置基本不变
                    y = -y + np.pi
                    # 小幅度位置扰动（相干长度尺度）
                    xi = self._localization_length_estimate(int(abs(x)) % self.n)
                    x += self.rng.normal(0.0, xi * 0.1)
                else:
                    # 正常散射：位置变化±1（格点跳跃）
                    dx = 1.0 if self.rng.rand() < 0.5 else -1.0
                    x += dx
                    y += self.rng.normal(0.0, 0.1)

                d2 = x * x + y * y
                d2_ave[step] += d2
                d2_max[step] = max(d2_max[step], d2)

        d2_ave /= num_walks

        return d2_ave, d2_max

    def compute_participation_ratio(self, wavefunction: np.ndarray) -> float:
        """
        计算波函数的反参与率（Inverse Participation Ratio, IPR）。

        IPR用于表征波函数的空间局域化程度：
            IPR = Σ_i |ψ_i|^4 / (Σ_i |ψ_i|^2)^2

        对于扩展态，IPR ~ 1/N；
        对于完全局域态，IPR ~ 1。

        在拓扑超导体中，马约拉纳零能模的IPR随着无序
        增强而增大，表明波函数从边界向内部收缩。
        """
        if wavefunction is None or len(wavefunction) == 0:
            return 0.0

        wf = np.asarray(wavefunction, dtype=np.float64)
        norm_sq = np.sum(wf * wf)
        if norm_sq < 1e-15:
            return 0.0

        ipr = np.sum(wf ** 4) / (norm_sq ** 2)
        return float(ipr)

    def disorder_averaged_correlation(self, num_realizations: int,
                                       max_distance: int) -> np.ndarray:
        """
        计算无序平均的空间关联函数。

        对于两个格点i和j，马约拉纳算符的关联函数为：
            C(r) = <γ_i γ_j> = δ_{ij} + G_{ij}
        其中G_{ij}为格点格林函数。

        在拓扑相中，关联函数呈指数衰减：
            C(r) ~ exp(-r/ξ) * cos(k_F * r)
        其中ξ为相干长度，k_F为费米波矢。
        """
        if max_distance < 0 or max_distance >= self.n:
            max_distance = self.n // 2

        corr = np.zeros(max_distance + 1)
        counts = np.zeros(max_distance + 1)

        for _ in range(num_realizations):
            self._generate_disorder_potential()

            # 简化的关联计算：基于无序势的自关联
            for r in range(max_distance + 1):
                for i in range(self.n - r):
                    j = i + r
                    val = self.mu_profile[i] * self.mu_profile[j]
                    corr[r] += val
                    counts[r] += 1

        # 避免除零
        mask = counts > 0
        corr[mask] /= counts[mask]
        # 归一化
        if abs(corr[0]) > 1e-15:
            corr /= corr[0]

        return corr

    def localization_length_scaling(self, w_vals: np.ndarray,
                                     num_realizations: int = 50) -> np.ndarray:
        """
        研究无序强度W与局域化长度的标度关系。

        在Anderson局域化理论中，一维系统的局域化长度
        与无序强度满足：
            ξ(W) ~ ξ_0 * (Δ/W)^2  (强无序极限)

        通过数值模拟提取有效局域化长度。
        """
        xi_eff = np.zeros_like(w_vals)
        original_W = self.W

        for idx, w in enumerate(w_vals):
            if w < 1e-10:
                xi_eff[idx] = float(self.n)
                continue

            self.W = w
            iprs = []
            for _ in range(num_realizations):
                self._generate_disorder_potential()
                # 用 disorder profile 的方差反比估计局域化长度
                var = np.var(self.mu_profile)
                if var > 1e-15:
                    iprs.append(1.0 / var)

            if iprs:
                xi_eff[idx] = np.mean(iprs)

        self.W = original_W
        return xi_eff


def demo():
    """演示无序随机游走模拟。"""
    walker = MajoranaDisorderRandomWalk(
        n_sites=50, disorder_strength=0.5, delta=0.8, t=1.0, mu0=0.0,
        rng_seed=42
    )
    d2_ave, d2_max = walker.simulate_andreev_random_walk(
        num_steps=100, num_walks=500, energy=0.0
    )
    print("Andreev Random Walk <r^2>(t=100):", d2_ave[-1])
    print("Max r^2(t=100):", d2_max[-1])

    corr = walker.disorder_averaged_correlation(
        num_realizations=20, max_distance=20
    )
    print("Disorder correlation C(r=0):", corr[0])
    print("Disorder correlation C(r=5):", corr[5])


if __name__ == "__main__":
    from typing import Optional
    demo()
