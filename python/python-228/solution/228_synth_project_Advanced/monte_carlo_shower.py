"""
monte_carlo_shower.py — 蒙特卡罗簇射采样器
============================================

融合种子项目:
  - 1410_wedge_monte_carlo : 蒙特卡罗积分 (楔形域采样)
  - 318_dragon_chaos       : 随机迭代 (Chaos game)

本模块实现电磁簇射的蒙特卡罗模拟:

1. 相互作用长度采样:
   t_int = -X0 * ln(xi)  (xi ~ Uniform(0,1))
   — 指数分布的自由程

2. Bremsstrahlung 能量分配 (Bethe-Heitler):
   P(k, k') dk' = (1/k') * [1 + (1-k'/k)^2 - 2/3*(1-k'/k)] dk'
   — 通过反转 CDF 采样次级光子能量

3. 对产生能量分配:
   P(k, E+) dE+ = [E+^2 + E-^2 + 2/3*E+*E-] / k^3 dE+
   其中 E- = k - E+

4. 多重散射角 (Highland 公式):
   theta_0 = (13.6 MeV / (beta*p*c)) * z * sqrt(x/X0) * [1 + 0.038*ln(x/X0)]

5. Molière 散射分布采样:
   theta ~ (chi_c / 2) * sqrt(-2*ln(xi))

物理约束:
  能量守恒: E_parent = E_daughter1 + E_daughter2
  动量守恒: 在超相对论极限下自动满足
  粒子数: N_max ~ E0/Ec (簇射中最大粒子数)
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from material_properties import MaterialSpec


@dataclass
class MCParticle:
    """蒙特卡罗粒子"""
    particle_type: str    # "electron", "positron", "photon"
    energy_MeV: float     # 能量 [MeV]
    depth_X0: float       # 当前深度 [辐射长度]
    x_cm: float           # 横向位置 x [cm]
    y_cm: float           # 横向位置 y [cm]
    theta: float          # 极角 [rad]
    phi: float            # 方位角 [rad]
    weight: float = 1.0   # 统计权重
    generation: int = 0   # 代次 (0 = 初始粒子)


@dataclass
class MCShowerResult:
    """蒙特卡罗簇射结果"""
    energy_deposition: List[Tuple[float, float]]  # (depth_X0, dE [MeV])
    lateral_distribution: List[Tuple[float, float]]  # (r_cm, dE)
    particle_count_vs_depth: List[Tuple[float, int]]
    total_deposited: float
    total_particles: int
    max_generation: int
    shower_max_X0: float
    leakage_fraction: float


class MonteCarloShowerSimulator:
    """电磁簇射蒙特卡罗模拟"""

    def __init__(
        self,
        material: MaterialSpec,
        seed: int = 42,
        max_particles: int = 5000,
        max_depth_X0: float = 30.0,
        energy_threshold_MeV: float = 0.1,
    ):
        self.mat = material
        self.rng = random.Random(seed)
        self.max_particles = max_particles
        self.max_depth = max_depth_X0
        self.E_threshold = energy_threshold_MeV

    def simulate(
        self,
        E0_MeV: float,
        particle_type: str = "electron",
        n_events: int = 10,
    ) -> MCShowerResult:
        """
        模拟电磁簇射

        算法:
          1. 创建初始粒子
          2. 对每个粒子:
             a. 采样自由程 t = -X0 * ln(xi)
             b. 根据粒子类型选择作用类型
             c. 生成次级粒子
             d. 记录能量沉积
          3. 重复直到所有粒子低于阈值或超出探测器
        """
        all_depositions = []
        all_lateral = []
        all_counts = []
        total_deposited = 0.0
        max_gen = 0
        total_leaked = 0.0

        for iev in range(n_events):
            # 重置 RNG 种子以保证可重复性 (每个事件)
            event_rng = random.Random(self.rng.randint(0, 2**31))

            # 初始粒子
            initial = MCParticle(
                particle_type=particle_type,
                energy_MeV=E0_MeV,
                depth_X0=0.0,
                x_cm=0.0,
                y_cm=0.0,
                theta=0.0,
                phi=0.0,
            )

            active_particles = [initial]
            event_depositions = []
            event_lateral = []
            event_max_gen = 0

            step = 0
            while active_particles and step < 200:
                step += 1
                next_particles = []

                for p in active_particles:
                    if p.energy_MeV < self.E_threshold:
                        # 能量沉积在当前位置
                        event_depositions.append((p.depth_X0, p.energy_MeV * p.weight))
                        total_deposited += p.energy_MeV * p.weight
                        continue

                    if p.depth_X0 > self.max_depth:
                        total_leaked += p.energy_MeV * p.weight
                        continue

                    # 采样自由程
                    xi = event_rng.random()
                    free_path_X0 = -math.log(max(xi, 1e-30))

                    # 推进到作用点
                    p.depth_X0 += free_path_X0

                    if p.depth_X0 > self.max_depth:
                        total_leaked += p.energy_MeV * p.weight
                        continue

                    # 多重散射
                    theta_scatter = self._sample_scattering_angle(
                        p.energy_MeV, free_path_X0, event_rng,
                    )
                    p.theta = math.sqrt(p.theta ** 2 + theta_scatter ** 2)

                    # 横向位移
                    dx = free_path_X0 * self.mat.X0_cm * math.sin(p.theta) * math.cos(p.phi)
                    dy = free_path_X0 * self.mat.X0_cm * math.sin(p.theta) * math.sin(p.phi)
                    p.x_cm += dx
                    p.y_cm += dy
                    r_cm = math.sqrt(p.x_cm ** 2 + p.y_cm ** 2)

                    # 作用类型
                    if p.particle_type == "photon":
                        # 光子: 对产生 (80%) 或 Compton (20%, 简化忽略)
                        if event_rng.random() < 0.8:
                            secondaries = self._pair_production(p, event_rng)
                        else:
                            # Compton 简化: 光子损失部分能量
                            frac = 0.3 + 0.4 * event_rng.random()
                            e_dep = p.energy_MeV * frac * p.weight
                            event_depositions.append((p.depth_X0, e_dep))
                            total_deposited += e_dep
                            p.energy_MeV *= (1.0 - frac)
                            secondaries = [p] if p.energy_MeV > self.E_threshold else []
                    else:
                        # 电子/正电子: bremsstrahlung + 电离
                        e_ion_loss = self._ionization_step(p.energy_MeV, free_path_X0)
                        e_brem = self._sample_bremsstrahlung(p.energy_MeV, event_rng)

                        # 电离能量沉积
                        event_depositions.append((p.depth_X0, e_ion_loss * p.weight))
                        total_deposited += e_ion_loss * p.weight
                        event_lateral.append((r_cm, e_ion_loss * p.weight))

                        # Bremsstrahlung 光子
                        secondaries = []
                        if e_brem > self.E_threshold and len(next_particles) + len(active_particles) < self.max_particles:
                            photon = MCParticle(
                                particle_type="photon",
                                energy_MeV=e_brem,
                                depth_X0=p.depth_X0,
                                x_cm=p.x_cm,
                                y_cm=p.y_cm,
                                theta=p.theta * 0.5,  # 小角近似
                                phi=event_rng.uniform(0, 2 * math.pi),
                                weight=p.weight,
                                generation=p.generation + 1,
                            )
                            secondaries.append(photon)

                        # 继续传播的电子
                        p.energy_MeV -= (e_ion_loss + e_brem)
                        if p.energy_MeV > self.E_threshold:
                            secondaries.append(p)

                    for s in secondaries:
                        if s.generation > event_max_gen:
                            event_max_gen = s.generation
                        next_particles.append(s)

                active_particles = next_particles

            # 处理剩余粒子
            for p in active_particles:
                if p.energy_MeV > self.E_threshold:
                    total_leaked += p.energy_MeV * p.weight

            all_depositions.extend(event_depositions)
            all_lateral.extend(event_lateral)
            max_gen = max(max_gen, event_max_gen)

        # 归一化
        n_avg = max(n_events, 1)
        total_deposited /= n_avg
        total_leaked /= n_avg

        # 找到 shower max
        depth_bins = {}
        for d, e in all_depositions:
            bin_idx = int(d * 10)  # 0.1 X0 分辨率
            depth_bins[bin_idx] = depth_bins.get(bin_idx, 0.0) + e / n_avg
        shower_max = 0.0
        max_dep = 0.0
        for bin_idx, dep in depth_bins.items():
            if dep > max_dep:
                max_dep = dep
                shower_max = (bin_idx + 0.5) / 10.0

        total_init = E0_MeV
        leakage = total_leaked / max(total_init, 1e-30)

        return MCShowerResult(
            energy_deposition=all_depositions,
            lateral_distribution=all_lateral,
            particle_count_vs_depth=[],
            total_deposited=total_deposited,
            total_particles=len(all_depositions),
            max_generation=max_gen,
            shower_max_X0=shower_max,
            leakage_fraction=leakage,
        )

    def _sample_scattering_angle(
        self, E_MeV: float, step_X0: float, rng: random.Random,
    ) -> float:
        """
        Highland 公式采样散射角

        theta_0 = (13.6 MeV / (beta*p)) * sqrt(step/X0) * [1 + 0.038*ln(step/X0)]

        采样: theta = theta_0 * G(0,1) (高斯近似)
        """
        m_e = 0.511  # MeV
        gamma = 1.0 + E_MeV / m_e
        beta = math.sqrt(1.0 - 1.0 / (gamma * gamma))
        p = beta * gamma * m_e  # MeV/c

        if p < 1e-10:
            return 0.0

        log_x = math.log(max(step_X0, 1e-10))
        theta_0 = (13.6 / p) * math.sqrt(max(step_X0, 0.0)) * (1.0 + 0.038 * log_x)
        theta_0 = max(theta_0, 0.0)

        # 高斯采样 (Box-Muller)
        u1 = max(rng.random(), 1e-30)
        u2 = rng.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

        return abs(theta_0 * z)

    def _sample_bremsstrahlung(
        self, E_MeV: float, rng: random.Random,
    ) -> float:
        """
        Bremsstrahlung 光子能量采样

        Bethe-Heitler 谱:
          dP/dk = (1/k) * [1 + (1-k/E)^2 - 2/3*(1-k/E)]

        采样策略:
          1. 以 1/k 分布采样 (主要项)
          2. 以修正因子接受/拒绝
        """
        if E_MeV < self.E_threshold:
            return 0.0

        # 采样 k ~ 1/k (对数均匀)
        k_min = max(self.E_threshold, self.mat.Ec_MeV * 0.01)
        k_max = E_MeV * 0.95  # 保留部分能量给电子

        if k_max <= k_min:
            return 0.0

        # 对数均匀采样
        log_k = math.log(k_min) + rng.random() * (math.log(k_max) - math.log(k_min))
        k = math.exp(log_k)

        # 接受/拒绝 (Bethe-Heitler 修正)
        ratio = k / E_MeV
        acceptance = 1.0 + (1.0 - ratio) ** 2 - (2.0 / 3.0) * (1.0 - ratio)
        acceptance /= (4.0 / 3.0)  # 归一化到最大值

        if rng.random() < acceptance:
            return k
        else:
            return 0.0

    def _pair_production(
        self, photon: MCParticle, rng: random.Random,
    ) -> List[MCParticle]:
        """
        对产生: gamma → e+ + e-

        能量分配:
          dP/dE+ ∝ [E+^2 + E-^2 + 2/3*E+*E-] / k^3
          E+ + E- = k (光子能量)

        采样: 均匀分配 + 接受/拒绝
        """
        k = photon.energy_MeV
        if k < 2.0 * 0.511:  # 阈值 2*m_e*c^2
            return []

        # 采样 E+
        E_plus = rng.uniform(0.511, k - 0.511)
        E_minus = k - E_plus

        # 接受/拒绝
        max_prob = k * k / 3.0
        prob = (E_plus ** 2 + E_minus ** 2 + (2.0 / 3.0) * E_plus * E_minus) / max(k * k, 1e-30)
        prob /= max_prob if max_prob > 0 else 1.0

        if rng.random() > prob:
            E_plus = k / 2.0
            E_minus = k / 2.0

        # 生成 e+ 和 e-
        e_plus = MCParticle(
            particle_type="positron",
            energy_MeV=E_plus,
            depth_X0=photon.depth_X0,
            x_cm=photon.x_cm,
            y_cm=photon.y_cm,
            theta=photon.theta * 0.3,
            phi=rng.uniform(0, 2 * math.pi),
            weight=photon.weight,
            generation=photon.generation + 1,
        )
        e_minus = MCParticle(
            particle_type="electron",
            energy_MeV=E_minus,
            depth_X0=photon.depth_X0,
            x_cm=photon.x_cm,
            y_cm=photon.y_cm,
            theta=photon.theta * 0.3,
            phi=rng.uniform(0, 2 * math.pi),
            weight=photon.weight,
            generation=photon.generation + 1,
        )

        return [e_plus, e_minus]

    def _ionization_step(self, E_MeV: float, step_X0: float) -> float:
        """电离连续能量损失"""
        return self.mat.ionization_loss_rate(E_MeV) * step_X0 * self.mat.X0_cm
