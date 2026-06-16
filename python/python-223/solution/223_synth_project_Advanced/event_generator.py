"""
event_generator.py — 部分子事件生成器与喷注 constituent 模拟

融合种子项目:
  - 1161_AmandaRosa_paper-code-artifact : 模拟框架 (framework/subscriber 模式)
  - 208_conservation_ode : ODE 守恒量系统 (用于部分子簇射演化)

物理背景:
  高能对撞中, 部分子 (quark, gluon) 经历级联辐射 (parton shower),
  最终强子化为可观测的粒子. 这一过程由 DGLAP 演化方程描述:

    ∂f_i(x, Q²)/∂ln(Q²) = Σ_j ∫_x^1 (dz/z) P_{ij}(z) f_j(x/z, Q²)

  其中 P_{ij}(z) 为 splitting function:
    P_{qq}(z) = C_F · (1+z²)/(1-z)        (quark → quark + gluon)
    P_{gq}(z) = C_F · (1+(1-z)²)/z        (quark → gluon + quark)
    P_{qg}(z) = T_R · (z²+(1-z)²)          (gluon → quark + antiquark)
    P_{gg}(z) = 2C_A · (z/(1-z) + (1-z)/z + z(1-z))

  颜色因子:
    C_F = (N_c²-1)/(2N_c) = 4/3   (SU(3))
    C_A = N_c = 3
    T_R = 1/2

  耦合常数跑动:
    α_s(Q²) = 12π / ((33-2n_f) ln(Q²/Λ²_QCD))
"""

import numpy as np
import math
from typing import List, Tuple, Optional
from jet_fourvector import FourVector, random_fourvector


# ─────────────────────────────────────────────────────────────────────────────
# QCD 耦合常数与 splitting functions
# ─────────────────────────────────────────────────────────────────────────────

# 物理常数
N_C = 3          # 颜色数
N_F = 5          # 活跃味数 (对 LHC 能标)
C_F = (N_C ** 2 - 1) / (2 * N_C)  # = 4/3
C_A = N_C        # = 3
T_R = 0.5
LAMBDA_QCD = 0.250  # Λ_QCD (GeV), 对 n_f=5

def alpha_s(Q: float) -> float:
    """QCD 跑动耦合常数 (1-loop).

    α_s(Q²) = 12π / ((33-2n_f) · ln(Q²/Λ²))

    对 Q < Λ_QCD, 使用冻结值 α_s(Q_min) 以避免 Landau pole.
    """
    Q_min = LAMBDA_QCD * 1.1  # 截断
    Q_eff = max(Q, Q_min)
    beta0 = (33 - 2 * N_F) / (12 * math.pi)
    alpha = 1.0 / (beta0 * math.log(Q_eff ** 2 / LAMBDA_QCD ** 2))
    return max(alpha, 0.01)  # 下限截断


def P_qq(z: float) -> float:
    """Altarelli-Parisi splitting function P_{qq}(z).

    P_{qq}(z) = C_F · (1+z²)/(1-z)  + virtual correction

    对 z → 1 有 soft singularity, 需要正则化.
    """
    if z <= 0 or z >= 1:
        return 0.0
    # + prescription 正则化 (简化): 截断 z_max = 1-ε
    eps = 1e-6
    z_eff = min(z, 1 - eps)
    return C_F * (1 + z_eff ** 2) / (1 - z_eff)


def P_gq(z: float) -> float:
    """P_{gq}(z) = C_F · (1+(1-z)²)/z."""
    if z <= 0 or z >= 1:
        return 0.0
    return C_F * (1 + (1 - z) ** 2) / z


def P_qg(z: float) -> float:
    """P_{qg}(z) = T_R · (z²+(1-z)²)."""
    if z <= 0 or z >= 1:
        return 0.0
    return T_R * (z ** 2 + (1 - z) ** 2)


def P_gg(z: float) -> float:
    """P_{gg}(z) = 2C_A · [z/(1-z) + (1-z)/z + z(1-z)]."""
    if z <= 0 or z >= 1:
        return 0.0
    eps = 1e-6
    z_eff = min(max(z, eps), 1 - eps)
    return 2 * C_A * (z_eff / (1 - z_eff) + (1 - z_eff) / z_eff +
                      z_eff * (1 - z_eff))


# ─────────────────────────────────────────────────────────────────────────────
# 部分子簇射模拟器 (简化版)
# ─────────────────────────────────────────────────────────────────────────────

class PartonShower:
    """简化版部分子簇射模拟器.

    使用 angular-ordered shower:
      每次分裂: 母部分子 → 两个子部分子
      分裂概率: dP ∝ α_s/(2π) · P(z) · dz · dQ²/Q²
      演化从 Q_max (喷注能标) 到 Q_min (强子化能标 ~1 GeV)
    """

    def __init__(self, Q_max: float = 500.0,
                 Q_min: float = 1.0,
                 seed: int = 42):
        self.Q_max = Q_max
        self.Q_min = Q_min
        self.rng = np.random.default_rng(seed)
        self.history: List[dict] = []
        self.n_splittings = 0

    def generate_shower(self, initial_parton: FourVector,
                        parton_type: str = 'quark',
                        max_depth: int = 10) -> List[FourVector]:
        """生成部分子簇射.

        Args:
            initial_parton: 初始部分子四动量
            parton_type: 'quark' 或 'gluon'
            max_depth: 最大分裂深度

        Returns:
            最终部分子列表 (将强子化为可观测粒子)
        """
        # 递归分裂
        final_partons = self._split_recursive(
            initial_parton, parton_type,
            self.Q_max, 0, max_depth
        )
        return final_partons

    def _split_recursive(self, parton: FourVector,
                         ptype: str, Q: float,
                         depth: int, max_depth: int) -> List[FourVector]:
        """递归分裂.

        在 each 步骤:
        1. 计算分裂概率 ΔP
        2. 如果 ΔP > random, 执行分裂
        3. 否则停止 ( Sudakov form factor )

        Sudakov form factor:
          Δ(Q₁, Q₂) = exp(-∫_{Q₁²}^{Q₂²} dQ'²/Q'² · ∫dz · α_s/(2π) P(z))
        """
        if depth >= max_depth or Q <= self.Q_min:
            return [parton]

        # 分裂概率 (简化估计)
        alpha = alpha_s(Q)
        if ptype == 'quark':
            P_max = P_qq(0.5)  # z=0.5 处最大值
        else:
            P_max = P_gg(0.5)

        # Sudakov 概率
        dQ = Q - self.Q_min
        delta_P = alpha / (2 * math.pi) * P_max * math.log(Q ** 2 / max(self.Q_min ** 2, 1e-10))
        delta_P = min(delta_P, 0.9)  # 截断

        # Monte Carlo 决策
        if self.rng.uniform() > delta_P:
            return [parton]  # 不分裂

        # 采样 z (能量份额)
        z = self.rng.beta(2.0, 2.0)  # 简化: Beta 分布近似
        z = max(0.05, min(0.95, z))

        # 分裂角度
        theta = math.sqrt(max(0.0, 2 * (Q ** 2 - self.Q_min ** 2))) / Q
        theta = min(theta, math.pi / 4)

        # 生成两个子部分子
        child1, child2 = self._split_kinematics(parton, z, theta, ptype)

        self.n_splittings += 1
        self.history.append({
            'depth': depth,
            'z': z,
            'theta': theta,
            'Q': Q,
            'parent_type': ptype,
            'parent_pT': parton.pT
        })

        # 递归分裂
        Q_new = Q * math.sqrt(z * (1 - z))
        if Q_new <= self.Q_min:
            return [child1, child2]

        child1_type = ptype  # 简化: quark 辐射后仍为 quark
        child2_type = 'gluon'

        result1 = self._split_recursive(child1, child1_type, Q_new,
                                        depth + 1, max_depth)
        result2 = self._split_recursive(child2, child2_type, Q_new,
                                        depth + 1, max_depth)

        return result1 + result2

    def _split_kinematics(self, parent: FourVector,
                          z: float, theta: float,
                          ptype: str) -> Tuple[FourVector, FourVector]:
        """从母部分子生成两个子部分子的运动学.

        在母部分子静止系中:
          E₁ = z · E_parent
          E₂ = (1-z) · E_parent
          夹角 = θ

        然后 boost 回实验室系.
        """
        # 简化: 在横向平面内分裂
        phi_parent = parent.phi
        phi1 = phi_parent + theta / 2
        phi2 = phi_parent - theta / 2

        E1 = z * parent.E
        E2 = (1 - z) * parent.E

        # 保持 η 近似相同 (准直极限)
        eta = parent.eta
        pT1 = E1 / math.cosh(eta)
        pT2 = E2 / math.cosh(eta)

        px1 = pT1 * math.cos(phi1)
        py1 = pT1 * math.sin(phi1)
        pz1 = pT1 * math.sinh(eta)

        px2 = pT2 * math.cos(phi2)
        py2 = pT2 * math.sin(phi2)
        pz2 = pT2 * math.sinh(eta)

        child1 = FourVector(E1, px1, py1, pz1)
        child2 = FourVector(E2, px2, py2, pz2)

        return child1, child2


# ─────────────────────────────────────────────────────────────────────────────
# 事件生成框架 (源自 1161 framework/subscriber 模式)
# ─────────────────────────────────────────────────────────────────────────────

class EventGenerator:
    """事件生成器框架.

    采用 framework/subscriber 模式 (源自 1161):
    - 多个物理过程可以注册为 subscriber
    - 事件生成按顺序触发各 subscriber
    - 支持事件选择和加权

    物理过程:
    1. q q̄ → g g  (双喷注)
    2. g g → g g  (胶子融合)
    3. q g → q g  (Compton 散射)
    """

    def __init__(self, sqrt_s: float = 13000.0, seed: int = 42):
        self.sqrt_s = sqrt_s
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.subscribers: List[callable] = []
        self.events: List[List[FourVector]] = []

    def register(self, subscriber: callable):
        """注册一个事件生成 subscriber."""
        self.subscribers.append(subscriber)

    def generate_event(self, process: str = 'dijet',
                       n_initial: int = 2) -> List[FourVector]:
        """生成一个物理事件.

        Args:
            process: 'dijet', 'multijet', 'shower'
            n_initial: 初始部分子数

        Returns:
            粒子四动量列表
        """
        if process == 'dijet':
            event = self._generate_dijet()
        elif process == 'multijet':
            event = self._generate_multijet()
        elif process == 'shower':
            event = self._generate_showered_event()
        else:
            event = self._generate_dijet()

        # 通知 subscribers
        for sub in self.subscribers:
            event = sub(event)

        self.events.append(event)
        return event

    def _generate_dijet(self) -> List[FourVector]:
        """生成双喷注事件 (2→2 散射).

        q q̄ → q q̄ 的运动学:
          t = -s/2 · (1 - cos θ*)
          dσ/dt ∝ α_s²/s² · (s²+u²)/t²  (leading order)

        在质心系:
          p₁ = (E, 0, 0, E)
          p₂ = (E, 0, 0, -E)
          p₃ = (E, E sin θ*, 0, E cos θ*)
          p₄ = (E, -E sin θ*, 0, -E cos θ*)
        """
        E_beam = self.sqrt_s / 2

        # 散射角 (从 LO 矩阵元采样)
        # dσ/dcos θ* ∝ (1+cos²θ*)/(1-cos θ*)²  → 向前峰值
        cos_theta = 1 - 2 * self.rng.uniform(0.1, 0.9)
        sin_theta = math.sqrt(max(0, 1 - cos_theta ** 2))

        # 部分子分布权重 (简化)
        x1 = self.rng.uniform(0.01, 0.5)
        x2 = self.sqrt_s * x1 / (self.sqrt_s * (1 - x1 + 0.01))
        x2 = min(max(x2, 0.01), 0.5)
        E_parton = math.sqrt(x1 * x2) * self.sqrt_s / 2

        # 四个部分子
        p1 = FourVector(x1 * E_beam, 0, 0, x1 * E_beam)
        p2 = FourVector(x2 * E_beam, 0, 0, -x2 * E_beam)
        p3 = FourVector(E_parton,
                        E_parton * sin_theta, 0,
                        E_parton * cos_theta)
        p4 = FourVector(E_parton,
                        -E_parton * sin_theta, 0,
                        -E_parton * cos_theta)

        return [p1, p2, p3, p4]

    def _generate_multijet(self, n_jets: int = 6) -> List[FourVector]:
        """生成多喷注事件."""
        particles = []
        for _ in range(n_jets):
            p = random_fourvector(
                pT_range=(20.0, 200.0),
                eta_range=(-2.5, 2.5),
                rng=self.rng
            )
            particles.append(p)

        # pT 平衡
        sum_px = sum(p.px for p in particles)
        sum_py = sum(p.py for p in particles)
        for i in range(len(particles)):
            particles[i] = FourVector(
                particles[i].E,
                particles[i].px - sum_px / n_jets,
                particles[i].py - sum_py / n_jets,
                particles[i].pz
            )
        return particles

    def _generate_showered_event(self, n_partons: int = 2) -> List[FourVector]:
        """生成经过部分子簇射的事件."""
        # 先生成硬散射
        base_event = self._generate_multijet(n_partons)

        # 对每个部分子进行簇射
        shower = PartonShower(Q_max=200.0, Q_min=1.0, seed=self.seed)
        all_particles = []
        for p in base_event:
            showered = shower.generate_shower(p, max_depth=5)
            all_particles.extend(showered)

        return all_particles


# ─────────────────────────────────────────────────────────────────────────────
# 事件过滤器
# ─────────────────────────────────────────────────────────────────────────────

def pT_filter(event: List[FourVector],
              pT_min: float = 20.0) -> List[FourVector]:
    """pT 截断过滤器."""
    return [p for p in event if p.pT >= pT_min]


def eta_filter(event: List[FourVector],
               eta_max: float = 2.5) -> List[FourVector]:
    """η 接受度过滤器."""
    return [p for p in event if abs(p.eta) <= eta_max]


def isolation_filter(event: List[FourVector],
                     delta_R_min: float = 0.4) -> List[FourVector]:
    """隔离过滤器: 移除距离太近的粒子.

    对每对粒子, 如果 ΔR < delta_R_min, 保留 pT 较大者.
    """
    if len(event) <= 1:
        return event

    filtered = list(event)
    changed = True
    while changed:
        changed = False
        new_filtered = []
        removed = set()
        for i in range(len(filtered)):
            if i in removed:
                continue
            keep = True
            for j in range(i + 1, len(filtered)):
                if j in removed:
                    continue
                dr = filtered[i].delta_R(filtered[j])
                if dr < delta_R_min:
                    # 保留 pT 较大的
                    if filtered[i].pT < filtered[j].pT:
                        removed.add(i)
                    else:
                        removed.add(j)
                    changed = True
            if i not in removed:
                new_filtered.append(filtered[i])
        filtered = new_filtered

    return filtered
