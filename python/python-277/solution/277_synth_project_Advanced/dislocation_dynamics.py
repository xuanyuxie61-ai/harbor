# -*- coding: utf-8 -*-
"""
dislocation_dynamics.py
=======================
位错动力学演化与双稳态分析模块

本模块融合以下种子项目算法:
- 1107_Gelens-Lab_cellcyclemodules: 双稳态分析 → 位错运动的双稳态相变
- 1215_Timothysit_riskychoice-paper-code: 风险选择模型 → 位错脱钉扎的随机跃迁

核心物理:
---------
位错动力学方程 (Overdamped运动):
B v = F_total = F_applied + F_back + F_image + F_thermal

其中:
- v = dx/dt: 位错速度
- B: 拖曳系数
- F_applied = τ b: 外应力驱动力
- F_back: 弹性back-stress (来自其他位错)
- F_image: 镜像力
- F_thermal: 热涨落力

双稳态行为:
在特定应力范围内，位错可以处于两种稳定状态:
1. 锁定态 (locked): 被Peierls势垒捕获
2. 滑动态 (gliding): 在势垒间快速运动

这种双稳态导致塑性变形的不连续性 (jerky flow)
"""

import math
from physical_constants import PI, DEFAULT_MATERIAL, BOLTZMANN_CONSTANT


class BistableDislocationDynamics:
    """
    双稳态位错动力学模型

    受Gelens Lab细胞周期模块的启发，将双稳态分析方法
    应用于位错在Peierls势垒中的运动。

    双稳态条件:
    dF/du = 0 有3个实数解 (2个稳定 + 1个不稳定)

    对于位错运动:
    F(u) = τ b - dγ/du - κ u

    其中 κ 是弹性back-stress刚度

    分岔分析:
    随 τ 变化，系统经历鞍结分岔 (saddle-node bifurcation):
    - τ < τ_lower: 只有锁定态
    - τ_lower < τ < τ_upper: 双稳态
    - τ > τ_upper: 只有滑动态
    """

    def __init__(self, material=None, kappa_back=1e-3):
        """
        初始化双稳态动力学

        Args:
            material: 材料参数
            kappa_back: back-stress刚度 (N/m²)
        """
        self.mat = material or DEFAULT_MATERIAL
        self.kappa = kappa_back

    def total_force(self, u, tau):
        """
        计算位错受到的总力

        F(u) = τ b - (2πγ_amp/b) sin(2πu/b) - κ u

        各项物理含义:
        - τ b: 外应力驱动力
        - -(2πγ_amp/b) sin(2πu/b): Peierls恢复力
        - -κ u: 弹性back-stress

        Args:
            u: 位错位置 (m)
            tau: 施加剪应力 (Pa)

        Returns:
            float: 总力 (Pa·m = N/m, 每单位位错长度)
        """
        b = self.mat.b_magnitude
        gamma_amp = self.mat.gamma_amplitude

        # 外应力项
        F_applied = tau * b

        # Peierls恢复力
        F_peierls = -(2.0 * PI * gamma_amp / b) * math.sin(2.0 * PI * u / b)

        # Back-stress
        F_back = -self.kappa * u

        return F_applied + F_peierls + F_back

    def force_derivative(self, u, tau):
        """
        计算力的导数 dF/du

        dF/du = -(2π/b)² γ_amp cos(2πu/b) - κ

        用于稳定性分析:
        dF/du < 0: 稳定
        dF/du > 0: 不稳定

        Args:
            u: 位错位置
            tau: 施加应力

        Returns:
            float: dF/du (N/m²)
        """
        b = self.mat.b_magnitude
        gamma_amp = self.mat.gamma_amplitude

        dF_du = -(2.0 * PI / b)**2 * gamma_amp * math.cos(2.0 * PI * u / b) - self.kappa
        return dF_du

    def find_fixed_points(self, tau, u_range=None, n_points=1000):
        """
        寻找固定点 (F(u) = 0)

        使用二分法在u范围内搜索所有零点

        Args:
            tau: 施加应力
            u_range: 搜索范围 (u_min, u_max)
            n_points: 离散点数

        Returns:
            list: 固定点信息 [{'u': ..., 'F_prime': ..., 'stable': ...}, ...]
        """
        if u_range is None:
            b = self.mat.b_magnitude
            u_range = (-2.0 * b, 2.0 * b)

        u_min, u_max = u_range
        du = (u_max - u_min) / n_points

        fixed_points = []
        u_prev = u_min
        F_prev = self.total_force(u_prev, tau)

        for i in range(1, n_points + 1):
            u_curr = u_min + i * du
            F_curr = self.total_force(u_curr, tau)

            # 检测符号变化 (零点)
            if F_prev * F_curr < 0:
                # 二分法精确定位
                u_lo, u_hi = u_prev, u_curr
                for _ in range(50):
                    u_mid = (u_lo + u_hi) / 2.0
                    F_mid = self.total_force(u_mid, tau)
                    if F_mid * self.total_force(u_lo, tau) < 0:
                        u_hi = u_mid
                    else:
                        u_lo = u_mid

                u_fp = (u_lo + u_hi) / 2.0
                F_prime = self.force_derivative(u_fp, tau)

                fixed_points.append({
                    'u': u_fp,
                    'F_prime': F_prime,
                    'stable': F_prime < 0,
                    'type': 'stable' if F_prime < 0 else 'unstable',
                })

            u_prev = u_curr
            F_prev = F_curr

        return fixed_points

    def bifurcation_analysis(self, tau_range, n_tau=50):
        """
        分岔分析: 固定点随应力的变化

        追踪稳定和不稳定固定点随 τ 的变化

        Args:
            tau_range: 应力范围 (tau_min, tau_max)
            n_tau: 应力离散点数

        Returns:
            dict: 分岔图数据
        """
        tau_min, tau_max = tau_range
        dtau = (tau_max - tau_min) / n_tau

        stable_branches = []
        unstable_branches = []

        for i in range(n_tau + 1):
            tau = tau_min + i * dtau
            fps = self.find_fixed_points(tau)

            for fp in fps:
                entry = {'tau': tau, 'u': fp['u']}
                if fp['stable']:
                    stable_branches.append(entry)
                else:
                    unstable_branches.append(entry)

        # 检测分岔点 (稳定/不稳定数量变化)
        bifurcation_points = []
        prev_n_stable = None
        for i in range(n_tau + 1):
            tau = tau_min + i * dtau
            fps = self.find_fixed_points(tau)
            n_stable = sum(1 for fp in fps if fp['stable'])
            if prev_n_stable is not None and n_stable != prev_n_stable:
                bifurcation_points.append({
                    'tau': tau,
                    'n_stable_before': prev_n_stable,
                    'n_stable_after': n_stable,
                })
            prev_n_stable = n_stable

        return {
            'stable_branches': stable_branches,
            'unstable_branches': unstable_branches,
            'bifurcation_points': bifurcation_points,
            'n_tau': n_tau,
        }

    def potential_energy(self, u, tau):
        """
        计算位错势能

        V(u) = -τ b u + γ_amp (1 - cos(2πu/b)) + (κ/2) u²

        势能极小值对应稳定固定点
        势能极大值对应不稳定固定点

        Args:
            u: 位错位置
            tau: 施加应力

        Returns:
            float: 势能 (J/m, 每单位位错长度)
        """
        b = self.mat.b_magnitude
        gamma_amp = self.mat.gamma_amplitude

        V = (-tau * b * u +
             gamma_amp * (1.0 - math.cos(2.0 * PI * u / b)) +
             0.5 * self.kappa * u**2)
        return V

    def energy_barrier(self, tau):
        """
        计算相邻稳定点之间的能垒

        ΔE = V(u_unstable) - V(u_stable)

        能垒高度决定热激活率

        Args:
            tau: 施加应力

        Returns:
            dict: 能垒信息
        """
        fps = self.find_fixed_points(tau)

        stable = [fp for fp in fps if fp['stable']]
        unstable = [fp for fp in fps if not fp['stable']]

        if len(stable) < 2 or len(unstable) < 1:
            return {
                'barrier_height': 0.0,
                'has_barrier': False,
                'n_stable': len(stable),
            }

        # 找最高能垒
        max_barrier = 0.0
        for u_fp in unstable:
            V_unstable = self.potential_energy(u_fp['u'], tau)
            for u_s in stable:
                V_stable = self.potential_energy(u_s['u'], tau)
                barrier = V_unstable - V_stable
                if barrier > max_barrier:
                    max_barrier = barrier

        return {
            'barrier_height': max_barrier,
            'has_barrier': max_barrier > 0,
            'n_stable': len(stable),
            'n_unstable': len(unstable),
        }


class StochasticDislocationTransition:
    """
    随机位错跃迁模型

    受风险选择模型 (risky choice) 启发，将位错在势阱间的
    跃迁建模为随机过程。

    位错在双稳态势能中:
    - 在锁定态 (左阱): 等待热激活
    - 跃迁到滑动态 (右阱): 越过能垒
    - 可能回跳到锁定态: 反向跃迁

    跃迁率 (Kramers理论):
    k_{L→R} = (ω_L ω_S / (2π B)) exp(-ΔE_{L→R} / (k_B T))
    k_{R→L} = (ω_R ω_S / (2π B)) exp(-ΔE_{R→L} / (k_B T))

    其中 ω_L, ω_R 是势阱频率, ω_S 是势垒频率
    """

    def __init__(self, dynamics, temperature=300.0, seed=None):
        """
        初始化随机跃迁模型

        Args:
            dynamics: 双稳态动力学对象
            temperature: 温度 (K)
            seed: 随机种子
        """
        self.dynamics = dynamics
        self.temperature = temperature
        self.mat = dynamics.mat
        self.kT = BOLTZMANN_CONSTANT * temperature

        import random
        self.rng = random.Random(seed)

    def well_curvature(self, u_fp, tau):
        """
        计算势阱曲率 (角频率平方)

        ω² = V''(u_fp) / m_eff

        V''(u) = (2π/b)² γ_amp cos(2πu/b) + κ

        对于位错, m_eff ≈ ρ b² (线质量密度)

        Args:
            u_fp: 固定点位置
            tau: 施加应力

        Returns:
            float: ω² (1/s²)
        """
        b = self.mat.b_magnitude
        gamma_amp = self.mat.gamma_amplitude

        V_double_prime = ((2.0*PI/b)**2 * gamma_amp * math.cos(2.0*PI*u_fp/b)
                          + self.dynamics.kappa)

        # 有效质量密度 (每单位位错长度)
        m_eff = self.mat.rho_mass * b**2

        omega_sq = abs(V_double_prime) / m_eff
        return omega_sq

    def kramers_rate(self, u_well, u_saddle, tau):
        """
        计算Kramers跃迁率

        k = (ω_well / (2π)) * (ω_saddle / (2π)) * (2π / (B/m_eff))
            * exp(-ΔE / (k_B T))

        简化为:
        k = (ω_well * ω_saddle / (2π B_eff)) * exp(-ΔE / (k_B T))

        Args:
            u_well: 势阱位置
            u_saddle: 势垒位置
            tau: 施加应力

        Returns:
            float: 跃迁率 (1/s)
        """
        omega_well_sq = self.well_curvature(u_well, tau)
        omega_saddle_sq = self.well_curvature(u_saddle, tau)

        omega_well = math.sqrt(max(omega_well_sq, 0.0))
        omega_saddle = math.sqrt(max(omega_saddle_sq, 0.0))

        # 能垒
        dE = self.dynamics.potential_energy(u_saddle, tau) - \
             self.dynamics.potential_energy(u_well, tau)

        if dE <= 0.0 or self.kT <= 0.0:
            return 0.0

        B = self.mat.drag_coefficient(self.temperature)
        m_eff = self.mat.rho_mass * self.mat.b_magnitude**2

        # Kramers率 (中间摩擦极限)
        k_rate = (omega_well * omega_saddle / (2.0 * PI * B / m_eff)) * \
                 math.exp(-dE / self.kT)

        return k_rate

    def simulate_trajectory(self, tau, t_total, dt):
        """
        模拟位错随机跃迁轨迹

        Gillespie算法 ( kinetic Monte Carlo ):
        1. 计算各通道的跃迁率 a_i
        2. 总率 a_0 = Σ a_i
        3. 等待时间 τ_wait = -ln(U) / a_0
        4. 选择通道: Σ_{j<i} a_j < U' a_0 ≤ Σ_{j≤i} a_j
        5. 执行跃迁, 重复

        Args:
            tau: 施加应力
            t_total: 总模拟时间 (s)
            dt: 输出时间间隔 (s)

        Returns:
            dict: 轨迹数据
        """
        fps = self.dynamics.find_fixed_points(tau)
        stable = [fp for fp in fps if fp['stable']]
        unstable = [fp for fp in fps if not fp['stable']]

        if len(stable) < 2:
            # 单稳态: 无跃迁
            return {
                'time': [0.0],
                'state': [0],
                'n_transitions': 0,
            }

        # 计算跃迁率
        n_wells = len(stable)
        rates = [[0.0]*n_wells for _ in range(n_wells)]

        for i in range(n_wells):
            for j in range(n_wells):
                if i != j and unstable:
                    # 使用最近的势垒
                    u_saddle = unstable[min(j, len(unstable)-1)]['u']
                    rates[i][j] = self.kramers_rate(
                        stable[i]['u'], u_saddle, tau
                    )

        # Gillespie模拟
        t = 0.0
        state = 0  # 起始在第一个势阱
        time_list = [0.0]
        state_list = [state]
        n_transitions = 0

        while t < t_total:
            # 总跃迁率
            total_rate = sum(rates[state][j] for j in range(n_wells) if j != state)

            if total_rate < 1e-30:
                break

            # 等待时间
            u_wait = self.rng.random()
            if u_wait < 1e-300:
                u_wait = 1e-300
            dt_wait = -math.log(u_wait) / total_rate

            t += dt_wait
            if t > t_total:
                break

            # 选择目标态
            u_select = self.rng.random() * total_rate
            cumsum = 0.0
            new_state = state
            for j in range(n_wells):
                if j != state:
                    cumsum += rates[state][j]
                    if cumsum >= u_select:
                        new_state = j
                        break

            state = new_state
            n_transitions += 1

            # 记录
            while len(time_list) < int(t / dt) + 1:
                time_list.append(len(time_list) * dt)
                state_list.append(state)

        return {
            'time': time_list,
            'state': state_list,
            'n_transitions': n_transitions,
            'rates': rates,
            'n_wells': n_wells,
        }
