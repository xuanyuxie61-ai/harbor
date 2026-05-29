"""
damage_mechanics.py
复合材料渐进损伤演化模型：纤维损伤、基体损伤与界面脱粘。
原项目映射：
  - 511_heartbeat_ode 的非线性ODE结构用于疲劳损伤累积率方程
  - 1387_vanderpol_ode_period 的极限环概念用于循环载荷下的稳态损伤演化
  - 020_artery_pde 的波传播参数管理思想用于损伤诱发的应力波衰减
科学背景：
  基于连续损伤力学（CDM）和疲劳损伤理论，建立如下演化方程：
  纤维损伤率：
    d(d_f)/dN = (σ_1 / σ_f0)^m_f / (1 - d_f)^{k_f}
  基体损伤率：
    d(d_m)/dN = (σ_2 / σ_m0)^m_m / (1 - d_m)^{k_m}
  剪切损伤率：
    d(d_s)/dN = (τ_12 / τ_s0)^m_s / (1 - d_s)^{k_s}
  其中 N 为循环次数，σ_i 为应力分量，σ_i0 为疲劳强度系数，m_i、k_i 为材料常数。
  损伤阈值条件（Hashin准则）：
    纤维拉伸：F_ft = (σ_1 / X_T)^2 + (τ_12 / S)^2 ≥ 1
    纤维压缩：F_fc = (σ_1 / X_C)^2 ≥ 1
    基体拉伸：F_mt = (σ_2 / Y_T)^2 + (τ_12 / S)^2 ≥ 1
    基体压缩：F_mc = (σ_2 / (2 S_T))^2 + [(Y_C / (2 S_T))^2 - 1] σ_2/Y_C + (τ_12/S)^2 ≥ 1
"""

import numpy as np
from scipy.integrate import odeint


class DamageParameters:
    """复合材料损伤演化参数容器。"""

    def __init__(self):
        # Hashin 强度准则参数 (MPa)
        self.X_T = 2500.0   # 纤维方向拉伸强度
        self.X_C = 2000.0   # 纤维方向压缩强度
        self.Y_T = 80.0     # 横向拉伸强度
        self.Y_C = 200.0    # 横向压缩强度
        self.S = 120.0      # 面内剪切强度
        self.S_T = 50.0     # 横向剪切强度

        # 疲劳损伤参数
        self.sigma_f0 = 3500.0  # 纤维疲劳强度系数 (MPa)
        self.sigma_m0 = 120.0   # 基体疲劳强度系数 (MPa)
        self.tau_s0 = 180.0     # 剪切疲劳强度系数 (MPa)
        self.m_f = 8.0          # 纤维疲劳指数
        self.m_m = 6.0          # 基体疲劳指数
        self.m_s = 7.0          # 剪切疲劳指数
        self.k_f = 2.5          # 纤维损伤耦合系数
        self.k_m = 2.0          # 基体损伤耦合系数
        self.k_s = 2.2          # 剪切损伤耦合系数

        # 界面脱粘参数（类似heartbeat的sharp response参数）
        self.epsilon_debond = 0.001  # 界面脱粘敏感系数
        self.gamma_debond = 0.45     # 临界能量释放率比例
        self.tau_interface = 60.0    # 界面剪切强度

        # 损伤阈值
        self.d_threshold = 0.99


class DamageState:
    """损伤状态向量 [d_f, d_m, d_s, d_i]。"""

    def __init__(self, d_f=0.0, d_m=0.0, d_s=0.0, d_i=0.0):
        self.d_f = float(np.clip(d_f, 0.0, 0.99))
        self.d_m = float(np.clip(d_m, 0.0, 0.99))
        self.d_s = float(np.clip(d_s, 0.0, 0.99))
        self.d_i = float(np.clip(d_i, 0.0, 0.99))  # 界面脱粘损伤

    def to_array(self):
        return np.array([self.d_f, self.d_m, self.d_s, self.d_i])

    @classmethod
    def from_array(cls, arr):
        return cls(arr[0], arr[1], arr[2], arr[3])

    def is_failed(self):
        return any(v > 0.95 for v in [self.d_f, self.d_m, self.d_s, self.d_i])


def hashin_failure_criteria(stress, params):
    """
    Hashin 失效准则评估。
    stress: [σ_1, σ_2, τ_12] (MPa)
    返回: {mode: factor}，factor >= 1 表示该模式失效。
    """
    sigma1, sigma2, tau12 = stress
    results = {}

    # 纤维拉伸
    if sigma1 >= 0:
        results['fiber_tension'] = (sigma1 / params.X_T) ** 2 + (tau12 / params.S) ** 2
    # 纤维压缩
    else:
        results['fiber_compression'] = (abs(sigma1) / params.X_C) ** 2

    # 基体拉伸
    if sigma2 >= 0:
        results['matrix_tension'] = (sigma2 / params.Y_T) ** 2 + (tau12 / params.S) ** 2
    # 基体压缩
    else:
        term1 = (sigma2 / (2.0 * params.S_T)) ** 2
        term2 = ((params.Y_C / (2.0 * params.S_T)) ** 2 - 1.0) * sigma2 / params.Y_C
        term3 = (tau12 / params.S) ** 2
        results['matrix_compression'] = term1 + term2 + term3

    return results


def damage_evolution_ode(y, N, stress_amplitude, params):
    """
    疲劳损伤演化ODE系统。
    受 511_heartbeat_ode 的非线性ODE结构启发，但用于疲劳损伤累积。
    y = [d_f, d_m, d_s, d_i]
    N: 循环次数（自变量，类似时间）
    stress_amplitude: 循环应力幅值 [σ_1a, σ_2a, τ_12a]
    方程形式类比 heartbeat_deriv 的立方非线性：
      dd_f/dN = (|σ_1a|/σ_f0)^m_f / (1 - d_f)^{k_f} * H(F_ft - 1)
      dd_m/dN = (|σ_2a|/σ_m0)^m_m / (1 - d_m)^{k_m} * H(F_mt - 1)
      dd_s/dN = (|τ_12a|/τ_s0)^m_s / (1 - d_s)^{k_s} * H(F_s - 1)
      dd_i/dN = -1/ε * (d_i^3 - a * d_i + δ_interface)  （类比heartbeat）
    其中 H(x) 为Heaviside阶跃函数，δ_interface 为界面剪应变驱动项。
    """
    d_f, d_m, d_s, d_i = y
    sigma1a, sigma2a, tau12a = stress_amplitude

    # === HOLE 3 ===
    # TODO: Implement fatigue damage evolution ODE system.
    # Given current damage state y = [d_f, d_m, d_s, d_i] and stress amplitude
    # [sigma1a, sigma2a, tau12a], compute the damage evolution rates.
    #
    # Key scientific formulas:
    #   1. Use Hashin failure criteria to determine if damage accumulates
    #      (Heaviside step: H = 1 if criterion >= 1, else 0).
    #   2. Fiber damage rate: dd_f/dN = (|sigma1a|/sigma_f0)^m_f / (1-d_f)^k_f * H_ft
    #   3. Matrix damage rate: dd_m/dN = (|sigma2a|/sigma_m0)^m_m / (1-d_m)^k_m * H_mt
    #   4. Shear damage rate: dd_s/dN = (|tau12a|/tau_s0)^m_s / (1-d_s)^k_s * H_s
    #   5. Interface debonding (heartbeat-like cubic nonlinearity):
    #      dd_i/dN = -1/epsilon * (d_i^3 - a_debond*d_i + interface_drive)
    #      where interface_drive = |tau12a|/tau_interface - gamma_debond
    #   6. Apply numerical clipping: dd_f, dd_m, dd_s >= 0 and capped at d=0.99
    #
    # The resulting damage variables [d_f, d_m, d_s] feed into the degraded
    # stiffness calculation in material_model.py and stiffness_assembly.py.
    raise NotImplementedError("Hole 3: damage_evolution_ode core computation needs implementation.")


def integrate_damage_cycles(initial_damage, stress_history, params, num_cycles):
    """
    积分损伤演化方程 over N cycles。
    stress_history: 每循环的应力幅值序列，shape=(num_cycles, 3)
    返回: damage_states array shape=(num_cycles+1, 4)
    """
    y0 = initial_damage.to_array()
    states = [y0.copy()]

    # 分步积分，每10个循环作为一步（效率与精度平衡）
    step = max(1, num_cycles // 100)
    N_current = 0
    y_current = y0.copy()

    while N_current < num_cycles:
        N_step = min(step, num_cycles - N_current)
        # 取当前阶段的平均应力幅
        idx_start = min(N_current, len(stress_history) - 1)
        idx_end = min(N_current + N_step, len(stress_history))
        if idx_end <= idx_start:
            avg_stress = stress_history[-1]
        else:
            avg_stress = np.mean(stress_history[idx_start:idx_end], axis=0)

        # ODE积分
        N_span = np.linspace(0, float(N_step), max(5, N_step // 2 + 1))
        sol = odeint(damage_evolution_ode, y_current, N_span,
                     args=(avg_stress, params), rtol=1e-6, atol=1e-9)
        y_current = sol[-1].copy()
        # 裁剪到物理范围
        y_current = np.clip(y_current, 0.0, 0.99)
        states.append(y_current.copy())
        N_current += N_step

    return np.array(states)


def compute_damage_dissipation_energy(damage_states, material, params):
    """
    计算损伤耗散能：
      W_d = ∫_0^{d_f} Y_f dd_f + ∫_0^{d_m} Y_m dd_m + ∫_0^{d_s} Y_s dd_s
    其中能量释放率 Y_i = -∂Ψ/∂d_i，Ψ 为弹性应变能密度。
    简化模型：
      Y_f ≈ (σ_1^2) / (2 E_1 (1 - d_f)^2)
      Y_m ≈ (σ_2^2) / (2 E_2 (1 - d_m)^2)
      Y_s ≈ (τ_12^2) / (2 G_12 (1 - d_s)^2)
    """
    n = len(damage_states)
    if n < 2:
        return 0.0

    W_d = 0.0
    for i in range(1, n):
        d_prev = damage_states[i - 1]
        d_curr = damage_states[i]
        dd = d_curr - d_prev
        d_mid = 0.5 * (d_prev + d_curr)

        # 简化的能量释放率（假设单位应力幅）
        denom_f = max(2.0 * material.E1 * (1.0 - d_mid[0]) ** 2, 1e-12)
        denom_m = max(2.0 * material.E2 * (1.0 - d_mid[1]) ** 2, 1e-12)
        denom_s = max(2.0 * material.G12 * (1.0 - d_mid[2]) ** 2, 1e-12)

        Y_f = 1.0 / denom_f
        Y_m = 1.0 / denom_m
        Y_s = 1.0 / denom_s

        W_d += Y_f * dd[0] + Y_m * dd[1] + Y_s * dd[2]

    return W_d


def estimate_damage_period(stress_amplitude, params):
    """
    估计达到完全损伤所需的循环次数（类比vanderpol_period）。
    使用渐进分析近似：
      N_f ≈ (1 - d_0)^{k_f + 1} / [(k_f + 1) * (σ_a/σ_f0)^{m_f}]
    """
    sigma1a, sigma2a, tau12a = stress_amplitude
    d0 = 0.0

    N_f = np.inf
    if abs(sigma1a) > 0:
        rate_f = (abs(sigma1a) / params.sigma_f0) ** params.m_f
        if rate_f > 0:
            N_f = min(N_f, (1.0 - d0) ** (params.k_f + 1.0) / ((params.k_f + 1.0) * rate_f))

    if abs(sigma2a) > 0:
        rate_m = (abs(sigma2a) / params.sigma_m0) ** params.m_m
        if rate_m > 0:
            N_f = min(N_f, (1.0 - d0) ** (params.k_m + 1.0) / ((params.k_m + 1.0) * rate_m))

    if abs(tau12a) > 0:
        rate_s = (abs(tau12a) / params.tau_s0) ** params.m_s
        if rate_s > 0:
            N_f = min(N_f, (1.0 - d0) ** (params.k_s + 1.0) / ((params.k_s + 1.0) * rate_s))

    return N_f if N_f < np.inf else 1e6
