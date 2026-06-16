"""
jet_fourvector.py — 四维动量 Lorentz 向量运算模块

融合种子项目:
  - 964_r83p : 三维点集的代数运算、行列式、随机采样
  - 961_r8_scale : 浮点数邻域 (next/previous) 的精确控制
  - 849_partition_brute : 集合划分用于动量分区优化

核心物理对象: 粒子四维动量 p^μ = (E, p_x, p_y, p_z)
Lorentz 度规: g_{μν} = diag(+1, -1, -1, -1) (mostly-minus 约定)

关键不变量:
  p² = E² - |p⃗|² = m²   (质量壳条件)
  s = (p₁ + p₂)²          (Mandelstam 变量)
  ΔR² = (Δη)² + (Δφ)²     (η-φ 空间角距离)
  rapidity:  y = 0.5 ln[(E+p_z)/(E-p_z)]
  pseudorapidity: η = -ln[tan(θ/2)]
  transverse momentum: p_T = √(p_x² + p_y²)
"""

import numpy as np
import math
import itertools

# ─────────────────────────────────────────────────────────────────────────────
# 浮点数邻域操作 (源自 961_r8_scale)
# 在 jet substructure 的数值稳定性分析中, 需要精确控制 pT 阈值的浮点邻域
# ─────────────────────────────────────────────────────────────────────────────

def r8_next(r8: float) -> float:
    """返回大于 r8 的最小浮点数 (IEEE-754 上邻域)."""
    if r8 == 0.0:
        return np.finfo(float).tiny
    elif r8 == -np.finfo(float).tiny:
        return 0.0
    elif r8 == np.finfo(float).max:
        return float('inf')
    elif r8 == float('inf'):
        return float('inf')
    elif r8 > 0.0:
        return np.nextafter(r8, float('inf'))
    else:
        return np.nextafter(r8, float('-inf'))


def r8_previous(r8: float) -> float:
    """返回小于 r8 的最大浮点数 (IEEE-754 下邻域)."""
    if r8 == 0.0:
        return -np.finfo(float).tiny
    elif r8 == np.finfo(float).tiny:
        return 0.0
    elif r8 == np.finfo(float).max:
        return np.nextafter(float('inf'), 0.0)
    elif r8 == -float('inf'):
        return -float('inf')
    elif r8 > 0.0:
        return np.nextafter(r8, float('-inf'))
    else:
        return np.nextafter(r8, float('inf'))


def r8_bracket(r8: float, n_eps: int = 1) -> list:
    """返回 r8 的浮点邻域括号: [r8-n*eps, r8, r8+n*eps].
    用于 pT 阈值扫描时的边界鲁棒性测试.
    """
    lo, hi = r8, r8
    for _ in range(abs(n_eps)):
        if n_eps > 0:
            hi = r8_next(hi)
            lo = r8_previous(lo)
        else:
            hi = r8_previous(hi)
            lo = r8_next(lo)
    return [lo, r8, hi]


# ─────────────────────────────────────────────────────────────────────────────
# 四维动量类
# ─────────────────────────────────────────────────────────────────────────────

class FourVector:
    """Lorentz 四维动量 p^μ = (E, px, py, pz).

    度规符号: (+, -, -, -).
    所有粒子默认满足质量壳条件 E² = |p⃗|² + m².
    """

    __slots__ = ('E', 'px', 'py', 'pz', 'm')

    def __init__(self, E: float, px: float, py: float, pz: float,
                 mass: float = 0.0):
        self.E = float(E)
        self.px = float(px)
        self.py = float(py)
        self.pz = float(pz)
        # 质量壳约束: m² = E² - p⃗²  (允许微小偏离)
        m2 = self.E ** 2 - (self.px ** 2 + self.py ** 2 + self.pz ** 2)
        self.m = math.sqrt(max(m2, 0.0)) if mass == 0.0 else float(mass)

    # ── 基本代数 (源自 964_r83p 的三维点集运算推广) ──

    def __add__(self, other: 'FourVector') -> 'FourVector':
        return FourVector(
            self.E + other.E,
            self.px + other.px,
            self.py + other.py,
            self.pz + other.pz
        )

    def __sub__(self, other: 'FourVector') -> 'FourVector':
        return FourVector(
            self.E - other.E,
            self.px - other.px,
            self.py - other.py,
            self.pz - other.pz
        )

    def __mul__(self, scalar: float) -> 'FourVector':
        s = float(scalar)
        return FourVector(self.E * s, self.px * s, self.py * s, self.pz * s)

    def __rmul__(self, scalar: float) -> 'FourVector':
        return self.__mul__(scalar)

    def __repr__(self) -> str:
        return (f"FourVector(E={self.E:.6f}, px={self.px:.6f}, "
                f"py={self.py:.6f}, pz={self.pz:.6f}, m={self.m:.6f})")

    # ── Lorentz 不变量 ──

    def invariant_mass_sq(self) -> float:
        """p² = E² - |p⃗|²."""
        return self.E ** 2 - (self.px ** 2 + self.py ** 2 + self.pz ** 2)

    def invariant_mass(self) -> float:
        m2 = self.invariant_mass_sq()
        return math.sqrt(max(m2, 0.0))

    def dot(self, other: 'FourVector') -> float:
        """Lorentz 内积: p₁·p₂ = E₁E₂ - p⃗₁·p⃗₂."""
        return (self.E * other.E - self.px * other.px
                - self.py * other.py - self.pz * other.pz)

    # ── 运动学变量 ──

    @property
    def pT(self) -> float:
        """横动量 p_T = √(p_x² + p_y²)."""
        return math.sqrt(self.px ** 2 + self.py ** 2)

    @property
    def pT_sq(self) -> float:
        return self.px ** 2 + self.py ** 2

    @property
    def phi(self) -> float:
        """方位角 φ ∈ (-π, π]."""
        return math.atan2(self.py, self.px)

    @property
    def theta(self) -> float:
        """极角 θ ∈ [0, π]."""
        p_mag = math.sqrt(self.px ** 2 + self.py ** 2 + self.pz ** 2)
        if p_mag < 1e-300:
            return 0.0
        cos_theta = max(-1.0, min(1.0, self.pz / p_mag))
        return math.acos(cos_theta)

    @property
    def eta(self) -> float:
        """赝快度 η = -ln[tan(θ/2)]."""
        theta_val = self.theta
        if theta_val < 1e-15:
            return 1e10
        if abs(math.pi - theta_val) < 1e-15:
            return -1e10
        return -math.log(math.tan(theta_val / 2.0))

    @property
    def rapidity(self) -> float:
        """快度 y = 0.5 * ln[(E+p_z)/(E-p_z)]."""
        denom_plus = self.E + self.pz
        denom_minus = self.E - self.pz
        if denom_minus <= 0.0:
            return 1e10
        if denom_plus <= 0.0:
            return -1e10
        return 0.5 * math.log(denom_plus / denom_minus)

    @property
    def p3_mag(self) -> float:
        """三维动量模 |p⃗|."""
        return math.sqrt(self.px ** 2 + self.py ** 2 + self.pz ** 2)

    def delta_R(self, other: 'FourVector') -> float:
        """ΔR = √((Δη)² + (Δφ)²)  — 喷注聚类核心距离度量."""
        deta = self.eta - other.eta
        dphi = self.phi - other.phi
        # 将 Δφ 折叠到 [-π, π]
        while dphi > math.pi:
            dphi -= 2.0 * math.pi
        while dphi < -math.pi:
            dphi += 2.0 * math.pi
        return math.sqrt(deta ** 2 + dphi ** 2)

    def delta_R_sq(self, other: 'FourVector') -> float:
        deta = self.eta - other.eta
        dphi = self.phi - other.phi
        while dphi > math.pi:
            dphi -= 2.0 * math.pi
        while dphi < -math.pi:
            dphi += 2.0 * math.pi
        return deta ** 2 + dphi ** 2


# ─────────────────────────────────────────────────────────────────────────────
# 集合划分与动量分区 (源自 849_partition_brute)
# ─────────────────────────────────────────────────────────────────────────────

def partition_momentum_brute(particles: list, R: float = 0.4) -> dict:
    """暴力搜索最优二分区划分, 使两个子集的总横动量差异最小.

    这是 NP-hard 划分问题在高能物理中的对应: 将喷注 constituents 划分为
    两个子喷注, 使得 pT 失衡最小化.

    物理动机: 识别双喷注子结构中, 需要枚举所有 2^n 种划分以找到
    全局最优 (对小规模 N ≤ 20 可行).

    Args:
        particles: FourVector 列表
        R: 喷注半径参数 (用于约束条件)

    Returns:
        dict: {'partition': 0/1 标签数组, 'discrepancy': pT 差异,
               'cost': 目标泛函值}
    """
    n = len(particles)
    if n == 0:
        return {'partition': [], 'discrepancy': 0.0, 'cost': 0.0}
    if n > 20:
        # 超过暴力搜索阈值, 返回贪心近似
        return _partition_greedy(particles)

    pT_list = [p.pT for p in particles]
    total_pT = sum(pT_list)
    target = total_pT / 2.0

    best_cost = float('inf')
    best_label = [0] * n

    # 枚举 2^n 种划分 (源自 849_partition_brute 的核心循环)
    for mask in range(1 << n):
        sum0, sum1 = 0.0, 0.0
        labels = []
        for i in range(n):
            if (mask >> i) & 1:
                sum1 += pT_list[i]
                labels.append(1)
            else:
                sum0 += pT_list[i]
                labels.append(0)
        disc = abs(sum0 - sum1)
        cost = disc / total_pT if total_pT > 1e-15 else 0.0
        if cost < best_cost:
            best_cost = cost
            best_label = labels

    return {
        'partition': best_label,
        'discrepancy': abs(
            sum(pT_list[i] for i in range(n) if best_label[i] == 0) -
            sum(pT_list[i] for i in range(n) if best_label[i] == 1)
        ),
        'cost': best_cost
    }


def _partition_greedy(particles: list) -> dict:
    """贪心近似 (LPT 调度): 按 pT 降序分配给当前和较小的子集."""
    pT_list = [p.pT for p in particles]
    order = sorted(range(len(pT_list)), key=lambda i: -pT_list[i])
    labels = [0] * len(particles)
    sum0, sum1 = 0.0, 0.0
    for i in order:
        if sum0 <= sum1:
            sum0 += pT_list[i]
        else:
            sum1 += pT_list[i]
            labels[i] = 1
    total = sum(pT_list)
    disc = abs(sum0 - sum1)
    return {
        'partition': labels,
        'discrepancy': disc,
        'cost': disc / total if total > 1e-15 else 0.0
    }


# ─────────────────────────────────────────────────────────────────────────────
# Mandelstam 变量与 2→2 散射
# ─────────────────────────────────────────────────────────────────────────────

def mandelstam_s(p1: FourVector, p2: FourVector) -> float:
    """Mandelstam s = (p₁ + p₂)²  — 质心系能量平方."""
    total = p1 + p2
    return total.invariant_mass_sq()


def mandelstam_t(p1: FourVector, p3: FourVector) -> float:
    """Mandelstam t = (p₁ - p₃)²  — 动量转移平方."""
    diff = p1 - p3
    return diff.invariant_mass_sq()


def mandelstam_u(p1: FourVector, p4: FourVector) -> float:
    """Mandelstam u = (p₁ - p₄)²."""
    diff = p1 - p4
    return diff.invariant_mass_sq()


def mandelstam_check(s: float, t: float, u: float,
                     m1: float, m2: float, m3: float, m4: float) -> float:
    """验证 Mandelstam 恒等式: s + t + u = m₁² + m₂² + m₃² + m₄².
    返回偏差 |s+t+u - Σm²|."""
    return abs(s + t + u - (m1 ** 2 + m2 ** 2 + m3 ** 2 + m4 ** 2))


# ─────────────────────────────────────────────────────────────────────────────
# 随机动量采样 (源自 964_r83p 的 r83p_random / r83p_fa)
# ─────────────────────────────────────────────────────────────────────────────

def random_fourvector(pT_range: tuple = (20.0, 500.0),
                      eta_range: tuple = (-2.5, 2.5),
                      mass: float = 0.0,
                      rng: np.random.Generator = None) -> FourVector:
    """生成一个随机物理四维动量.

    采样策略:
      p_T ~ Uniform(pT_range)
      η   ~ Uniform(eta_range)
      φ   ~ Uniform(-π, π)

    从 (pT, η, φ) 重建:
      p_x = p_T cos(φ)
      p_y = p_T sin(φ)
      θ = 2 arctan(e^{-η})
      p_z = p_T / tan(θ) = p_T sinh(η)
      E = √(p_T² cosh²η + m²)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    pT = rng.uniform(pT_range[0], pT_range[1])
    eta = rng.uniform(eta_range[0], eta_range[1])
    phi = rng.uniform(-math.pi, math.pi)

    px = pT * math.cos(phi)
    py = pT * math.sin(phi)
    pz = pT * math.sinh(eta)
    # E = √(pT² cosh²η + m²) = pT cosh(η) (对无质量粒子)
    E = math.sqrt(pT ** 2 * math.cosh(eta) ** 2 + mass ** 2)

    return FourVector(E, px, py, pz, mass=mass)


def generate_multijet_event(n_particles: int = 8,
                            sqrt_s: float = 13000.0,
                            seed: int = 42) -> list:
    """生成一个多喷注事件.

    在质心系中生成 n 个粒子, 总动量为零 (横向),
    总能量等于 √s 的一个子份额.

    Args:
        n_particles: 粒子数
        sqrt_s: 质心系能量 (GeV)
        seed: 随机种子

    Returns:
        FourVector 列表
    """
    rng = np.random.default_rng(seed)
    # 每个喷注的能量份额: E_jet ~ √s / n * (1 + 微扰)
    particles = []
    total_E = 0.0

    for i in range(n_particles):
        # 能量从指数分布采样 (模拟 QCD 能谱)
        E_jet = rng.exponential(sqrt_s / (3.0 * n_particles))
        E_jet = max(E_jet, 1.0)  # 最小能量截断
        eta = rng.uniform(-2.5, 2.5)
        phi = rng.uniform(-math.pi, math.pi)
        px = E_jet * math.cos(phi) / math.cosh(eta)
        py = E_jet * math.sin(phi) / math.cosh(eta)
        pz = E_jet * math.tanh(eta)
        total_E += E_jet
        particles.append(FourVector(E_jet, px, py, pz))

    # 横向动量平衡修正: 使 Σp⃗_T ≈ 0
    sum_px = sum(p.px for p in particles)
    sum_py = sum(p.py for p in particles)
    correction_px = -sum_px / n_particles
    correction_py = -sum_py / n_particles
    balanced = []
    for p in particles:
        new_px = p.px + correction_px
        new_py = p.py + correction_py
        new_E = math.sqrt(new_px ** 2 + new_py ** 2 + p.pz ** 2 + p.m ** 2)
        balanced.append(FourVector(new_E, new_px, new_py, p.pz))

    return balanced
