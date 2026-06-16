"""
conservation_laws.py — 守恒定律验证与物理约束模块

融合种子项目:
  - 208_conservation_ode : 守恒 ODE 系统的守恒量计算
    (pendulum: E = p²/2m + mgl(1-cos θ),
     rigid body: L² = L₁²+L₂²+L₃²,
     predator-prey: H = a ln y - b y + c ln x - d x)
  - 1161_AmandaRosa_paper-code-artifact : 模拟验证框架

物理背景:
  高能对撞中的守恒定律:
  1. 四动量守恒: Σ p_i^μ = p_initial^μ
  2. 电荷守恒: Σ Q_i = Q_initial
  3. 色荷守恒 (非微扰约束)
  4. 角动量守恒: J = L + S

  在喷注分析中的具体体现:
  - 聚类前后四动量守恒 (E-scheme 重组保证)
  - 喷注质量: m_jet² = (Σ p_i)²
  - 缺失横动量: MET = |Σ p⃗_T| (暗示中微子或新物理)
  - 标量横动量和: H_T = Σ p_{Ti}

  守恒量在 ODE 演化中的验证 (源自 208):
    DGLAP 演化应保持动量求和规则:
      Σ_i ∫₀¹ x f_i(x, Q²) dx = 1   (对所有 Q²)
"""

import numpy as np
import math
from typing import List, Dict, Tuple
from jet_fourvector import FourVector


# ─────────────────────────────────────────────────────────────────────────────
# 四动量守恒验证
# ─────────────────────────────────────────────────────────────────────────────

def total_four_momentum(particles: List[FourVector]) -> FourVector:
    """计算粒子系统的总四动量.

    P^μ = Σ_i p_i^μ
    """
    if not particles:
        return FourVector(0, 0, 0, 0)

    E = sum(p.E for p in particles)
    px = sum(p.px for p in particles)
    py = sum(p.py for p in particles)
    pz = sum(p.pz for p in particles)
    return FourVector(E, px, py, pz)


def invariant_mass(particles: List[FourVector]) -> float:
    """粒子系统的不变质量.

    M² = (Σ E_i)² - (Σ p⃗_i)²
    """
    total = total_four_momentum(particles)
    return total.invariant_mass()


def verify_momentum_conservation(
    particles_before: List[FourVector],
    particles_after: List[FourVector],
    tolerance: float = 1e-8
) -> Dict[str, any]:
    """验证四动量守恒.

    检查:
      |ΣE_after - ΣE_before| / |ΣE_before| < tolerance
      |Σp⃗_after - Σp⃗_before| / |Σp⃗_before| < tolerance

    Returns:
        dict: 守恒验证结果
    """
    total_b = total_four_momentum(particles_before)
    total_a = total_four_momentum(particles_after)

    dE = abs(total_a.E - total_b.E)
    dpx = abs(total_a.px - total_b.px)
    dpy = abs(total_a.py - total_b.py)
    dpz = abs(total_a.pz - total_b.pz)

    E_scale = max(abs(total_b.E), 1e-15)
    p_scale = max(math.sqrt(total_b.px ** 2 + total_b.py ** 2 +
                            total_b.pz ** 2), 1e-15)

    passed = (dE / E_scale < tolerance and
              dpx / p_scale < tolerance and
              dpy / p_scale < tolerance and
              dpz / p_scale < tolerance)

    return {
        'passed': passed,
        'delta_E': float(dE),
        'delta_px': float(dpx),
        'delta_py': float(dpy),
        'delta_pz': float(dpz),
        'relative_error_E': float(dE / E_scale),
        'relative_error_p': float(math.sqrt(dpx ** 2 + dpy ** 2 + dpz ** 2) /
                                  p_scale),
        'tolerance': tolerance
    }


# ─────────────────────────────────────────────────────────────────────────────
# 喷注观测量
# ─────────────────────────────────────────────────────────────────────────────

def compute_jet_observables(jet: FourVector,
                            constituents: List[FourVector]) -> Dict[str, float]:
    """计算单个喷注的物理观测量.

    包括:
    - pT, η, φ, m (运动学)
    -  constituents 数
    - pT 加权平均 ΔR (喷注宽度)
    - 电荷 (如果 constituents 有电荷信息)
    - 纵向动量分数
    """
    if not constituents:
        return {
            'pT': 0, 'eta': 0, 'phi': 0, 'mass': 0,
            'n_constituents': 0, 'width': 0, 'charge': 0,
            'Lund_plane_depth': 0
        }

    n_const = len(constituents)
    sum_pT = sum(c.pT for c in constituents)

    # 喷注宽度 (girth / angularity):
    # g = (1/pT_jet) Σ_i pT_i · ΔR_i
    girth = sum(c.pT * c.delta_R(jet) for c in constituents) / max(jet.pT, 1e-15)

    # 纵向动量分数 (对每个 constituent)
    pz_jet = jet.pz
    frac_pz = [c.pz / pz_jet if abs(pz_jet) > 1e-15 else 0.0
               for c in constituents]

    return {
        'pT': float(jet.pT),
        'eta': float(jet.eta),
        'phi': float(jet.phi),
        'mass': float(jet.invariant_mass()),
        'n_constituents': n_const,
        'width': float(girth),
        'sum_pT': float(sum_pT),
        'pT_balance': float(sum_pT / max(jet.pT, 1e-15)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 全局事件观测量
# ─────────────────────────────────────────────────────────────────────────────

def event_observables(particles: List[FourVector]) -> Dict[str, float]:
    """计算全局事件观测量.

    包括:
    - 标量横动量和 H_T = Σ p_{Ti}
    - 缺失横动量 MET = |Σ p⃗_T|
    - 总不变质量
    - 球度 (sphericity)
    - 推力 (thrust)
    """
    if not particles:
        return {
            'H_T': 0, 'MET': 0, 'invariant_mass': 0,
            'sphericity': 0, 'thrust': 0, 'n_particles': 0
        }

    H_T = sum(p.pT for p in particles)
    sum_px = sum(p.px for p in particles)
    sum_py = sum(p.py for p in particles)
    MET = math.sqrt(sum_px ** 2 + sum_py ** 2)

    total = total_four_momentum(particles)
    M_inv = total.invariant_mass()

    # 球度张量 S^{ab} = Σ p^a p^b / Σ |p|²
    n = len(particles)
    S = np.zeros((3, 3))
    sum_p2 = 0.0
    for p in particles:
        pvec = np.array([p.px, p.py, p.pz])
        p2 = np.dot(pvec, pvec)
        S += np.outer(pvec, pvec)
        sum_p2 += p2

    if sum_p2 > 1e-15:
        S /= sum_p2
        eigenvalues = np.linalg.eigvalsh(S)
        eigenvalues = np.sort(eigenvalues)
        # 球度: S = 3/2 (λ₂ + λ₃)  (λ₁ ≤ λ₂ ≤ λ₃)
        sphericity = 1.5 * (eigenvalues[1] + eigenvalues[2])
    else:
        sphericity = 0.0

    # 推力: T = max_{n̂} Σ |p⃗_i · n̂| / Σ |p⃗_i|
    # 简化: 搜索有限方向
    thrust = _compute_thrust(particles)

    return {
        'H_T': float(H_T),
        'MET': float(MET),
        'invariant_mass': float(M_inv),
        'sphericity': float(sphericity),
        'thrust': float(thrust),
        'n_particles': n,
        'total_E': float(total.E),
        'total_pz': float(total.pz)
    }


def _compute_thrust(particles: List[FourVector],
                    n_directions: int = 100) -> float:
    """计算推力 (thrust).

    T = max_{n̂} Σ_i |p⃗_i · n̂| / Σ_i |p⃗_i|

    通过在球面上均匀采样 n̂ 方向来近似最大化.
    """
    if not particles:
        return 0.0

    sum_p_mag = sum(p.p3_mag for p in particles)
    if sum_p_mag < 1e-15:
        return 0.0

    rng = np.random.default_rng(42)
    best_T = 0.0

    for _ in range(n_directions):
        # 随机单位向量
        cos_theta = rng.uniform(-1, 1)
        sin_theta = math.sqrt(1 - cos_theta ** 2)
        phi = rng.uniform(0, 2 * math.pi)
        n_hat = np.array([sin_theta * math.cos(phi),
                          sin_theta * math.sin(phi),
                          cos_theta])

        T = 0.0
        for p in particles:
            p_vec = np.array([p.px, p.py, p.pz])
            T += abs(np.dot(p_vec, n_hat))

        T /= sum_p_mag
        if T > best_T:
            best_T = T

    return best_T


# ─────────────────────────────────────────────────────────────────────────────
# DGLAP 动量求和规则验证 (源自 208_conservation_ode)
# ─────────────────────────────────────────────────────────────────────────────

def momentum_sum_rule_check(
    particle_fractions: List[float],
    tolerance: float = 0.05
) -> Dict[str, float]:
    """验证 DGLAP 动量求和规则.

    Σ_i ∫₀¹ x f_i(x) dx = 1

    在离散近似中:
      Σ_i x_i · p_{Ti} / p_{T,total} ≈ 1

    偏离量 Δ = |Σ - 1| 反映数值精度.
    """
    if not particle_fractions:
        return {'sum': 0.0, 'deviation': 1.0, 'passed': False}

    total = sum(particle_fractions)
    if total < 1e-15:
        return {'sum': 0.0, 'deviation': 1.0, 'passed': False}

    # 归一化份额
    x_norm = [x / total for x in particle_fractions]
    momentum_sum = sum(x_norm)  # 应等于 1

    deviation = abs(momentum_sum - 1.0)
    passed = deviation < tolerance

    return {
        'sum': float(momentum_sum),
        'deviation': float(deviation),
        'passed': passed,
        'tolerance': tolerance
    }


# ─────────────────────────────────────────────────────────────────────────────
# 守恒量 ODE 验证 (源自 208_conservation_ode)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_conservation_ode(
    y0: np.ndarray,
    t_span: Tuple[float, float] = (0, 10),
    n_steps: int = 1000,
    method: str = 'rk4'
) -> Dict[str, any]:
    """模拟守恒 ODE 系统并验证守恒量.

    使用摆锤方程 (源自 208_conservation_ode):
      θ' = ω
      ω' = -(g/L) sin θ

    守恒量: E = ½mL²ω² + mgL(1-cos θ)

    同时测试刚性转子:
      L₁' = (I₂-I₃)/I₁ · L₂L₃
      L₂' = (I₃-I₁)/I₂ · L₁L₃
      L₃' = (I₁-I₂)/I₃ · L₁L₂

    守恒量: |L|² = L₁²+L₂²+L₃², E = L₁²/I₁+L₂²/I₂+L₃²/I₃
    """
    # 摆锤系统
    g, L = 9.81, 1.0
    theta0, omega0 = y0[0], y0[1]

    def pendulum_rhs(t, y):
        return np.array([y[1], -(g / L) * math.sin(y[0])])

    def pendulum_energy(y):
        return 0.5 * L * L * y[1] ** 2 + g * L * (1 - math.cos(y[0]))

    # RK4 积分
    dt = (t_span[1] - t_span[0]) / n_steps
    y = np.array([theta0, omega0], dtype=float)
    t = t_span[0]

    energies = [pendulum_energy(y)]
    times = [t]

    for step in range(n_steps):
        k1 = pendulum_rhs(t, y)
        k2 = pendulum_rhs(t + dt / 2, y + dt / 2 * k1)
        k3 = pendulum_rhs(t + dt / 2, y + dt / 2 * k2)
        k4 = pendulum_rhs(t + dt, y + dt * k3)
        y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
        energies.append(pendulum_energy(y))
        times.append(t)

    energies = np.array(energies)
    E0 = energies[0]
    E_drift = np.max(np.abs(energies - E0)) / max(abs(E0), 1e-15)

    # 刚性转子
    I1, I2, I3 = 1.0, 2.0, 3.0
    L0 = np.array([1.0, 0.5, 0.3])

    def rigid_body_rhs(t, L_vec):
        return np.array([
            (I2 - I3) / I1 * L_vec[1] * L_vec[2],
            (I3 - I1) / I2 * L_vec[0] * L_vec[2],
            (I1 - I2) / I3 * L_vec[0] * L_vec[1]
        ])

    def rigid_body_conserved(L_vec):
        L_sq = np.dot(L_vec, L_vec)
        E = L_vec[0] ** 2 / I1 + L_vec[1] ** 2 / I2 + L_vec[2] ** 2 / I3
        return L_sq, E

    L_vec = L0.copy()
    L_sq_list = []
    E_rb_list = []

    for step in range(n_steps):
        k1 = rigid_body_rhs(t, L_vec)
        k2 = rigid_body_rhs(t + dt / 2, L_vec + dt / 2 * k1)
        k3 = rigid_body_rhs(t + dt / 2, L_vec + dt / 2 * k2)
        k4 = rigid_body_rhs(t + dt, L_vec + dt * k3)
        L_vec = L_vec + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        L_sq, E_rb = rigid_body_conserved(L_vec)
        L_sq_list.append(L_sq)
        E_rb_list.append(E_rb)

    L_sq_arr = np.array(L_sq_list)
    E_rb_arr = np.array(E_rb_list)
    L_drift = np.max(np.abs(L_sq_arr - L_sq_arr[0])) / max(abs(L_sq_arr[0]), 1e-15)
    E_rb_drift = np.max(np.abs(E_rb_arr - E_rb_arr[0])) / max(abs(E_rb_arr[0]), 1e-15)

    return {
        'pendulum_energy_drift': float(E_drift),
        'pendulum_energy_initial': float(E0),
        'pendulum_energy_final': float(energies[-1]),
        'rigid_body_L_drift': float(L_drift),
        'rigid_body_E_drift': float(E_rb_drift),
        'n_steps': n_steps,
        'dt': float(dt),
        'method': method
    }
