# -*- coding: utf-8 -*-
"""
stability_analysis.py
=====================
有限差分格式稳定性分析与位错周期检测模块

本模块融合以下种子项目算法:
- 267_cycle_floyd: Floyd周期检测 → 位错周期性振荡检测
- 104_boundary_locus: 边界轨迹法 → 稳定性边界参数化

核心物理:
---------
位错运动的有限差分离散:
u_i^{n+1} = u_i^n + (Δt/B) * [σ_app - σ_back(u^n) - σ_PN(u^n)]

von Neumann稳定性分析:
设 u_j^n = g^n e^{i k j Δx}，代入线性化方程:
g(k) = 1 + Δt * H(k) / B

稳定性条件: |g(k)| ≤ 1 对所有 k

对于扩散型方程 ∂u/∂t = D ∂²u/∂x²:
CFL条件: Δt ≤ Δx² / (2D)

对于含色散项的方程:
Δt ≤ min(Δx²/(2D), Δx⁴/(12α))

Floyd周期检测:
检测位错在循环加载下是否进入极限环 (稳定周期振荡)
"""

import math
from physical_constants import PI, DEFAULT_MATERIAL


# ============================================================================
# Floyd周期检测 — 来自 267_cycle_floyd
# 用于位错周期性振荡的识别
# ============================================================================

class FloydCycleDetector:
    """
    Floyd龟兔赛跑周期检测算法

    用于检测位错在循环载荷下的运动是否收敛到周期轨道。

    算法原理:
    - 乌龟以步长1移动: x_turtle = f(x_turtle)
    - 兔子以步长2移动: x_rabbit = f(f(x_rabbit))
    - 当 x_turtle == x_rabbit 时检测到周期
    - 然后从起点和相遇点分别以步长1移动，再次相遇点即为周期起点

    在位错物理中的应用:
    - 检测Frank-Read源的周期性激活
    - 检测位错在Peierls势垒中的锁定-解锁振荡
    - 检测循环塑性中的位错结构自组织

    时间复杂度: O(μ + λ), 空间复杂度: O(1)
    其中 μ 是循环前的尾巴长度，λ 是周期长度
    """

    def __init__(self, tolerance=1e-8, max_iterations=10000):
        """
        初始化Floyd检测器

        Args:
            tolerance: 收敛容差
            max_iterations: 最大迭代次数
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def detect_cycle(self, f, x0):
        """
        检测序列是否包含周期

        Phase 1: 龟兔相遇
        turtle = f(x0)
        rabbit = f(f(x0))
        while turtle ≠ rabbit:
            turtle = f(turtle)
            rabbit = f(f(rabbit))

        Phase 2: 找周期起点
        tortoise = x0
        while tortoise ≠ rabbit:
            tortoise = f(tortoise)
            rabbit = f(rabbit)
        mu = 起点到周期起点距离

        Phase 3: 计算周期长度
        rabbit = f(tortoise)
        lambda = 1
        while tortoise ≠ rabbit:
            rabbit = f(rabbit)
            lambda += 1

        Args:
            f: 映射函数 f(x) → x
            x0: 初始值

        Returns:
            dict: 周期检测结果
        """
        def seq_equal(a, b):
            """判断序列值是否相等 (考虑容差)"""
            if isinstance(a, (list, tuple)):
                return all(abs(ai - bi) < self.tolerance for ai, bi in zip(a, b))
            return abs(a - b) < self.tolerance

        # Phase 1: 龟兔相遇
        turtle = f(x0)
        rabbit = f(f(x0))

        n_meet = 0
        while not seq_equal(turtle, rabbit):
            turtle = f(turtle)
            rabbit = f(f(rabbit))
            n_meet += 1
            if n_meet > self.max_iterations:
                return {
                    'has_cycle': False,
                    'mu': None,
                    'period': None,
                    'cycle_start': None,
                    'convergence_iterations': n_meet,
                }

        # Phase 2: 找周期起点 (mu)
        mu = 0
        tortoise = x0
        while not seq_equal(tortoise, rabbit):
            tortoise = f(tortoise)
            rabbit = f(rabbit)
            mu += 1

        # Phase 3: 计算周期长度 (lambda)
        period = 1
        rabbit = f(tortoise)
        while not seq_equal(tortoise, rabbit):
            rabbit = f(rabbit)
            period += 1
            if period > self.max_iterations:
                break

        return {
            'has_cycle': True,
            'mu': mu,
            'period': period,
            'cycle_start': tortoise if isinstance(tortoise, (int, float)) else list(tortoise),
            'convergence_iterations': n_meet,
        }

    def detect_cycle_nd(self, f, x0):
        """
        多维状态空间的Floyd检测

        用于位错位置+速度相空间的周期检测

        Args:
            f: 映射函数 f(x_vec) → x_vec
            x0: 初始状态矢量

        Returns:
            dict: 周期检测结果 (同detect_cycle)
        """
        return self.detect_cycle(f, x0)


# ============================================================================
# von Neumann稳定性分析
# ============================================================================

class VonNeumannStability:
    """
    有限差分格式的von Neumann稳定性分析

    对于线性化位错运动方程:
    B ∂u/∂t = μ_eff ∂²u/∂x² - α ∂⁴u/∂x⁴ - κ u

    其中:
    - μ_eff = μb²/(4π(1-ν)): 有效弹性扩散系数
    - α: 过正化系数 (梯度能量)
    - κ = d²γ/du²: Peierls势垒曲率

    代入 Fourier模式 u = g^n exp(ikjΔx):
    g(k) = 1 + Δt/B * [μ_eff D₂(k) - α D₄(k) - κ]

    其中 D₂(k) 是2阶差分算子的符号:
    D₂(k) = (2cos(kΔx) - 2)/Δx² = -4sin²(kΔx/2)/Δx²

    D₄(k) 是4阶差分算子的符号:
    D₄(k) = (2cos(2kΔx) - 8cos(kΔx) + 6)/Δx⁴

    稳定性条件: |g(k)| ≤ 1 对所有 k ∈ [-π/Δx, π/Δx]
    """

    def __init__(self, material=None, fd_order=4):
        """
        初始化稳定性分析器

        Args:
            material: 材料参数
            fd_order: 有限差分精度阶数
        """
        self.mat = material or DEFAULT_MATERIAL
        self.fd_order = fd_order

    def amplification_factor(self, k, dx, dt, mu_eff, alpha, kappa):
        """
        计算放大因子 g(k)

        g(k) = 1 + Δt/B * [μ_eff D₂(k) - α D₄(k) - κ]

        Args:
            k: 波数 (1/m)
            dx: 空间步长
            dt: 时间步长
            mu_eff: 有效弹性系数
            alpha: 过正化系数
            kappa: Peierls势垒曲率

        Returns:
            complex: 放大因子 g(k)
        """
        B = self.mat.drag_coefficient(300.0)  # 拖曳系数

        # 2阶差分符号
        D2 = -4.0 * math.sin(k * dx / 2.0)**2 / dx**2

        # 4阶差分符号
        D4 = (2.0 * math.cos(2.0 * k * dx) - 8.0 * math.cos(k * dx) + 6.0) / dx**4

        # 放大因子
        exponent = dt / B * (mu_eff * D2 - alpha * D4 - kappa)

        # g = 1 + exponent (显式Euler)
        g = 1.0 + exponent

        return complex(g, 0.0)

    def amplification_factor_rk4(self, k, dx, dt, mu_eff, alpha, kappa):
        """
        计算RK4格式的放大因子

        对于 du/dt = L(u)，RK4的放大因子:
        g = 1 + z + z²/2 + z³/6 + z⁴/24

        其中 z = Δt * H(k) 是空间算子的符号

        RK4稳定性域: |g| ≤ 1 要求 z 在复平面特定区域内

        Args:
            k: 波数
            dx: 空间步长
            dt: 时间步长
            mu_eff: 有效弹性系数
            alpha: 过正化系数
            kappa: Peierls势垒曲率

        Returns:
            complex: RK4放大因子
        """
        B = self.mat.drag_coefficient(300.0)

        D2 = -4.0 * math.sin(k * dx / 2.0)**2 / dx**2
        D4 = (2.0 * math.cos(2.0 * k * dx) - 8.0 * math.cos(k * dx) + 6.0) / dx**4

        z = dt / B * (mu_eff * D2 - alpha * D4 - kappa)

        # RK4放大因子
        g = 1.0 + z + z**2/2.0 + z**3/6.0 + z**4/24.0
        return complex(g, 0.0)

    def stability_boundary(self, dx_range, mu_eff, alpha, kappa, scheme='euler'):
        """
        计算稳定性边界 Δt_max(dx)

        对所有 k ∈ [0, π/dx]，求使 |g(k)| ≤ 1 的最大 Δt

        对于Euler格式 + 扩散项:
        Δt_max = 2B * dx² / (μ_eff * 4) = B dx² / (2 μ_eff)

        对于含4阶项:
        Δt_max = min(B dx²/(2μ_eff), B dx⁴/(8α))

        Args:
            dx_range: 空间步长范围 (m)
            mu_eff: 有效弹性系数
            alpha: 过正化系数
            kappa: Peierls势垒曲率
            scheme: 时间格式 ('euler' 或 'rk4')

        Returns:
            dict: 稳定性边界信息
        """
        B = self.mat.drag_coefficient(300.0)
        n_k = 200  # 波数离散点数

        results = []

        for dx in dx_range:
            max_dt = float('inf')

            # 扫描所有波数
            for i_k in range(n_k + 1):
                k = PI * i_k / (dx * n_k)  # k ∈ [0, π/dx]

                # 二分搜索最大 dt
                dt_lo, dt_hi = 0.0, 1e-10
                for _ in range(50):
                    dt_mid = (dt_lo + dt_hi) / 2.0
                    if scheme == 'euler':
                        g = self.amplification_factor(k, dx, dt_mid, mu_eff, alpha, kappa)
                    else:
                        g = self.amplification_factor_rk4(k, dx, dt_mid, mu_eff, alpha, kappa)

                    if abs(g) <= 1.0:
                        dt_lo = dt_mid
                    else:
                        dt_hi = dt_mid

                max_dt = min(max_dt, dt_lo)

            results.append({
                'dx': dx,
                'dt_max': max_dt,
                'cfl_number': max_dt * mu_eff / (B * dx**2) if dx > 0 else 0.0,
            })

        return results

    def cfl_condition(self, dx, mu_eff, alpha=0.0, scheme='euler'):
        """
        计算CFL条件 (解析估算)

        对于扩散方程: Δt ≤ dx²/(2D)
        对于4阶方程: Δt ≤ dx⁴/(12α)

        Args:
            dx: 空间步长
            mu_eff: 有效弹性系数
            alpha: 过正化系数
            scheme: 时间格式

        Returns:
            dict: CFL条件信息
        """
        B = self.mat.drag_coefficient(300.0)

        # 2阶项限制
        if abs(mu_eff) > 1e-30:
            dt_diffusion = B * dx**2 / (2.0 * abs(mu_eff))
        else:
            dt_diffusion = float('inf')

        # 4阶项限制
        if abs(alpha) > 1e-30:
            dt_dispersion = B * dx**4 / (12.0 * abs(alpha))
        else:
            dt_dispersion = float('inf')

        dt_max = min(dt_diffusion, dt_dispersion)

        # RK4格式可以放宽约2.8倍
        if scheme == 'rk4':
            dt_max *= 2.8

        return {
            'dt_diffusion_limit': dt_diffusion,
            'dt_dispersion_limit': dt_dispersion,
            'dt_max': dt_max,
            'cfl_diffusion': mu_eff * dt_max / (B * dx**2) if dx > 0 else 0.0,
        }


# ============================================================================
# 边界轨迹稳定性分析 — 融合 104_boundary_locus
# ============================================================================

class StabilityLocusAnalysis:
    """
    稳定性边界轨迹分析

    在参数空间 (μ_eff, α, κ) 中参数化稳定性边界:
    |g(k_critical)| = 1

    这产生一个参数化曲面 (稳定性域边界)

    对于2参数情况 (Δt, dx):
    稳定域 = {(Δt, dx) : max_k |g(k; Δt, dx)| ≤ 1}
    """

    def __init__(self, material=None):
        self.mat = material or DEFAULT_MATERIAL

    def compute_locus_points(self, k_range, dx, mu_eff, alpha, kappa):
        """
        计算稳定性边界轨迹上的点

        对于每个 k，求满足 |g(k)| = 1 的 Δt 值

        Args:
            k_range: 波数列表
            dx: 空间步长
            mu_eff: 有效弹性系数
            alpha: 过正化系数
            kappa: Peierls势垒曲率

        Returns:
            list: 边界轨迹点 [(k, dt_crit), ...]
        """
        B = self.mat.drag_coefficient(300.0)
        locus = []

        for k in k_range:
            D2 = -4.0 * math.sin(k * dx / 2.0)**2 / dx**2
            D4 = (2.0 * math.cos(2.0 * k * dx) - 8.0 * math.cos(k * dx) + 6.0) / dx**4

            H_k = mu_eff * D2 - alpha * D4 - kappa

            if abs(H_k) < 1e-30:
                continue

            # Euler格式: g = 1 + dt*H/B = ±1
            # dt_crit = -2B/H (使 g = -1)
            if H_k < 0:
                dt_crit = -2.0 * B / H_k
                locus.append({
                    'k': k,
                    'dt_critical': dt_crit,
                    'H_k': H_k,
                    'D2': D2,
                    'D4': D4,
                })

        return locus

    def stability_map(self, mu_eff_range, alpha_values, dx, kappa):
        """
        生成二维稳定性图 (μ_eff vs α)

        对于每个 (μ_eff, α) 对，检查最危险波数的稳定性

        Args:
            mu_eff_range: μ_eff 值列表
            alpha_values: α 值列表
            dx: 空间步长
            kappa: Peierls势垒曲率

        Returns:
            list: 稳定性图数据 [(mu_eff, alpha, stable, dt_max), ...]
        """
        B = self.mat.drag_coefficient(300.0)
        n_k = 100
        dt_base = 1e-14  # 测试用时间步长

        stability_data = []

        for mu_eff in mu_eff_range:
            for alpha in alpha_values:
                stable = True
                max_amplification = 0.0

                for i_k in range(1, n_k):
                    k = PI * i_k / (dx * n_k)
                    g = self._compute_g(k, dx, dt_base, mu_eff, alpha, kappa)
                    amp = abs(g)
                    if amp > max_amplification:
                        max_amplification = amp
                    if amp > 1.0 + 1e-10:
                        stable = False
                        break

                stability_data.append({
                    'mu_eff': mu_eff,
                    'alpha': alpha,
                    'stable': stable,
                    'max_amplification': max_amplification,
                })

        return stability_data

    def _compute_g(self, k, dx, dt, mu_eff, alpha, kappa):
        """计算放大因子"""
        B = self.mat.drag_coefficient(300.0)
        D2 = -4.0 * math.sin(k * dx / 2.0)**2 / dx**2
        D4 = (2.0 * math.cos(2.0 * k * dx) - 8.0 * math.cos(k * dx) + 6.0) / dx**4
        z = dt / B * (mu_eff * D2 - alpha * D4 - kappa)
        return 1.0 + z


if __name__ == '__main__':
    print("=" * 70)
    print("稳定性分析与周期检测验证")
    print("=" * 70)

    # 测试Floyd周期检测
    print("\nFloyd周期检测:")
    detector = FloydCycleDetector(tolerance=1e-6)

    # 测试1: 已知周期序列 (周期3)
    def f_cycle(x):
        return (x * 7 + 3) % 10
    result = detector.detect_cycle(f_cycle, 2)
    print(f"  周期序列: has_cycle={result['has_cycle']}, "
          f"period={result['period']}, mu={result['mu']}")

    # 测试2: 位错振荡模拟
    def dislock_oscillation(theta):
        # 简化模型: θ_{n+1} = θ_n + ωΔt - A sin(θ_n)
        omega_dt = 0.3
        A = 0.2
        new_theta = theta + omega_dt - A * math.sin(theta)
        return new_theta % (2 * PI)

    result2 = detector.detect_cycle(dislock_oscillation, 1.0)
    print(f"  位错振荡: has_cycle={result2['has_cycle']}, "
          f"period={result2['period']}")

    # 测试稳定性分析
    print("\nCFL条件:")
    vn = VonNeumannStability()
    al = DEFAULT_MATERIAL

    mu_eff = al.mu * al.b_magnitude**2 / (4.0 * PI * (1.0 - al.nu))
    alpha = mu_eff * al.zeta_screw**2 / 10.0
    kappa = 2.0 * PI**2 * al.gamma_usf / al.b_magnitude**2

    for dx_factor in [1, 5, 10, 20]:
        dx = al.a_lattice * dx_factor
        cfl = vn.cfl_condition(dx, mu_eff, alpha)
        print(f"  dx={dx_factor}a₀: dt_max={cfl['dt_max']:.4e} s, "
              f"CFL={cfl['cfl_diffusion']:.4f}")

    # 测试稳定性边界轨迹
    print("\n稳定性边界轨迹:")
    locus = StabilityLocusAnalysis()
    k_values = [PI * i / (10 * al.a_lattice * 10) for i in range(1, 50)]
    dx_test = al.a_lattice * 10
    pts = locus.compute_locus_points(k_values, dx_test, mu_eff, alpha, kappa)
    print(f"  边界点数: {len(pts)}")
    if pts:
        dt_min = min(p['dt_critical'] for p in pts)
        dt_max = max(p['dt_critical'] for p in pts)
        print(f"  Δt范围: [{dt_min:.4e}, {dt_max:.4e}] s")
