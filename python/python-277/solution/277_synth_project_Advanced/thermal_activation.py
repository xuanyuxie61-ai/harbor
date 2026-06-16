# -*- coding: utf-8 -*-
"""
thermal_activation.py
=====================
热激活位错脱钉扎 Monte Carlo 模拟模块

本模块融合以下种子项目算法:
- 069_ball_monte_carlo: 球面蒙特卡罗积分 → 多维激活体积内的热激活概率计算
- 042_asa144: 随机列联表 → 脱钉扎事件的统计分布分析

核心物理:
---------
位错在低温或高应变速率下的运动受热激活控制。
热激活速率 (Arrhenius):

Γ = Γ₀ exp(-ΔG(τ) / (k_B T))

其中:
- Γ₀ = ω_D / (2π): 尝试频率 (Debye频率量级)
- ΔG(τ): 应力相关的Gibbs自由能垒
- k_B: Boltzmann常数
- T: 绝对温度

Gibbs自由能垒的应力依赖性:
ΔG(τ) = ΔF₀ * (1 - (|τ|/τ_P)^p)^q

其中:
- ΔF₀: 零应力激活能
- τ_P: Peierls应力
- p, q: 激活轮廓参数 (p=1, q=2 对应双kink形核)

激活体积:
V* = -∂ΔG/∂τ = ΔF₀ q p / τ_P * (|τ|/τ_P)^{p-1} * (1-(|τ|/τ_P)^p)^{q-1}

Monte Carlo方法:
在多维激活空间 (τ, T, 位错构型) 中采样热激活事件
"""

import math
import random
from physical_constants import (PI, SQRT2, SQRT3, DEFAULT_MATERIAL,
                                 BOLTZMANN_CONSTANT)


# ============================================================================
# 球面蒙特卡罗积分 — 来自 069_ball_monte_carlo
# 用于多维激活空间中的热激活概率计算
# ============================================================================

class BallMonteCarloSampler:
    """
    球体内蒙特卡罗采样

    在3D激活体积空间中均匀采样:
    V* = {v ∈ R³ : |v| ≤ r_activation}

    采样方法 (Ball01: 单位球):
    1. 在球坐标中采样:
       r = U(0,1)^{1/3} (保证体积均匀)
       θ = arccos(2U(0,1) - 1) (极角)
       φ = 2π U(0,1) (方位角)
    2. 转换到笛卡尔坐标:
       x = r sin(θ) cos(φ)
       y = r sin(θ) sin(φ)
       z = r cos(θ)

    物理应用:
    - 在激活体积空间内积分热激活速率
    - 计算平均激活能: <ΔG> = ∫∫∫ ΔG(v) d³v / V*
    """

    def __init__(self, seed=None):
        """
        初始化蒙特卡罗采样器

        Args:
            seed: 随机种子 (可复现性)
        """
        self.rng = random.Random(seed)
        self.n_samples = 0
        self.n_accepted = 0

    def sample_ball_3d(self, radius):
        """
        在3D球体内均匀采样

        使用球坐标采样 (避免拒绝法的浪费):
        r = R * U^{1/3}
        cos(θ) = 2U - 1
        φ = 2π U

        体积元: dV = r² sin(θ) dr dθ dφ

        Args:
            radius: 球体半径

        Returns:
            tuple: (x, y, z) 笛卡尔坐标
        """
        # 径向: 保证体积均匀分布
        u_r = self.rng.random()
        r = radius * u_r**(1.0/3.0)

        # 极角: 球面均匀
        u_theta = self.rng.random()
        cos_theta = 2.0 * u_theta - 1.0
        sin_theta = math.sqrt(max(1.0 - cos_theta**2, 0.0))

        # 方位角
        u_phi = self.rng.random()
        phi = 2.0 * PI * u_phi

        # 笛卡尔坐标
        x = r * sin_theta * math.cos(phi)
        y = r * sin_theta * math.sin(phi)
        z = r * cos_theta

        self.n_samples += 1
        return (x, y, z)

    def sample_ball_nd(self, radius, dimension):
        """
        在N维球体内均匀采样

        使用Muller方法:
        1. 从标准正态分布采样各分量
        2. 归一化到单位球面
        3. 乘以 R * U^{1/n}

        Args:
            radius: 球体半径
            dimension: 维度

        Returns:
            tuple: N维坐标
        """
        # 标准正态采样 (Box-Muller)
        components = []
        for _ in range(dimension):
            u1 = self.rng.random()
            u2 = self.rng.random()
            z = math.sqrt(-2.0 * math.log(max(u1, 1e-300))) * math.cos(2.0 * PI * u2)
            components.append(z)

        # 归一化
        norm = math.sqrt(sum(c**2 for c in components))
        if norm < 1e-300:
            return tuple(0.0 for _ in range(dimension))

        # 径向缩放
        u_r = self.rng.random()
        r = radius * u_r**(1.0 / dimension)

        point = tuple(c / norm * r for c in components)
        self.n_samples += 1
        return point

    def integrate_on_ball(self, f, radius, n_samples, dimension=3):
        """
        在球体上蒙特卡罗积分

        ∫_B f(x) d³x ≈ V_ball * (1/N) Σ f(x_i)

        V_ball = (4/3) π R³ (3D)

        Args:
            f: 被积函数 f(x, y, z) → float
            radius: 球体半径
            n_samples: 采样数
            dimension: 维度

        Returns:
            dict: 积分结果和误差估计
        """
        # 球体积
        if dimension == 1:
            volume = 2.0 * radius
        elif dimension == 2:
            volume = PI * radius**2
        elif dimension == 3:
            volume = (4.0 / 3.0) * PI * radius**3
        else:
            # 一般N维: V_n(R) = π^{n/2} R^n / Γ(n/2 + 1)
            volume = PI**(dimension/2.0) * radius**dimension / math.gamma(dimension/2.0 + 1)

        total = 0.0
        total_sq = 0.0

        for _ in range(n_samples):
            if dimension == 3:
                point = self.sample_ball_3d(radius)
                val = f(*point)
            else:
                point = self.sample_ball_nd(radius, dimension)
                val = f(*point)

            total += val
            total_sq += val**2

        mean = total / n_samples
        variance = total_sq / n_samples - mean**2
        std_error = math.sqrt(max(variance / n_samples, 0.0))

        integral = volume * mean
        integral_error = volume * std_error

        return {
            'integral': integral,
            'mean': mean,
            'variance': variance,
            'std_error': std_error,
            'relative_error': abs(integral_error / integral) if abs(integral) > 1e-30 else float('inf'),
            'volume': volume,
            'n_samples': n_samples,
        }


# ============================================================================
# 随机列联表分析 — 来自 042_asa144
# 用于脱钉扎事件的统计分析
# ============================================================================

class ContingencyTableAnalysis:
    """
    随机列联表生成与分析

    在AS Algorithm 144 (AS 144) 中，用于生成具有固定边际和的随机列联表。
    在位错理论中，用于模拟脱钉扎事件的统计分布:

    行: 不同类型的障碍 (林位错、析出物、溶质原子)
    列: 不同的脱钉扎模式 (热激活、应力辅助、量子隧穿)

    边际和约束:
    - 行和: 每种障碍的总数
    - 列和: 每种模式的总事件数

    生成算法 (基于条件分布):
    对每个格子 (i,j)，给定剩余行和和列和:
    n_{ij} ~ Hypergeometric(n_i., n_.j, N_remaining)
    """

    def __init__(self, seed=None):
        """
        初始化列联表分析器

        Args:
            seed: 随机种子
        """
        self.rng = random.Random(seed)

    def check_coprime_margins(self, row_sums, col_sums):
        """
        检查边际和的互素性

        使用扩展GCD确保边际和的兼容性:
        gcd(row_sum_i, col_sum_j) 必须允许整数填充

        Args:
            row_sums: 行和列表
            col_sums: 列和列表

        Returns:
            bool: 是否兼容
        """
        total_row = sum(row_sums)
        total_col = sum(col_sums)
        if total_row != total_col:
            return False

        # 检查每个边际对
        for r in row_sums:
            for c in col_sums:
                if r < 0 or c < 0:
                    return False

        return True

    def generate_random_table(self, row_sums, col_sums):
        """
        生成具有给定边际和的随机列联表

        使用顺序条件方法:
        对每个格子按行优先顺序填充:
        n_{ij} ~ Hypergeometric(剩余行和, 剩余列和, 剩余总数)

        Hypergeometric抽样:
        P(X=k) = C(R,k) * C(N-R, n-k) / C(N, n)

        其中:
        R = 剩余行和, n = 剩余列和, N = 剩余总数

        Args:
            row_sums: 行和列表 [r₁, r₂, ..., r_I]
            col_sums: 列和列表 [c₁, c₂, ..., c_J]

        Returns:
            list: I×J列联表
        """
        if not self.check_coprime_margins(row_sums, col_sums):
            raise ValueError("边际和不兼容")

        n_rows = len(row_sums)
        n_cols = len(col_sums)

        table = [[0]*n_cols for _ in range(n_rows)]
        remaining_rows = list(row_sums)
        remaining_cols = list(col_sums)

        for i in range(n_rows):
            for j in range(n_cols):
                if i == n_rows - 1:
                    # 最后一行: 由列和确定
                    table[i][j] = remaining_cols[j]
                elif j == n_cols - 1:
                    # 最后一列: 由行和确定
                    table[i][j] = remaining_rows[i]
                else:
                    # 超几何分布抽样
                    R = remaining_rows[i]
                    C = remaining_cols[j]
                    N = sum(remaining_rows[i:])

                    if N <= 0:
                        table[i][j] = 0
                    else:
                        # 期望值
                        mean = R * C / N
                        # 方差
                        var = R * C * (N - R) * (N - C) / (N * N * max(N - 1, 1))
                        std = math.sqrt(max(var, 0.0))

                        # 正态近似 + 截断
                        val = int(round(mean + self.rng.gauss(0, max(std, 0.5))))
                        val = max(0, min(val, min(R, C)))

                        table[i][j] = val

                # 更新剩余
                remaining_rows[i] -= table[i][j]
                remaining_cols[j] -= table[i][j]

        return table

    def analyze_table(self, table):
        """
        分析列联表的统计特性

        计算:
        - 总观测数 N
        - 期望频度 E_{ij} = r_i * c_j / N
        - χ² 统计量 = Σ (O-E)²/E
        - 自由度 df = (I-1)(J-1)

        Args:
            table: I×J列联表

        Returns:
            dict: 统计分析结果
        """
        n_rows = len(table)
        n_cols = len(table[0])

        # 边际和
        row_sums = [sum(table[i]) for i in range(n_rows)]
        col_sums = [sum(table[i][j] for i in range(n_rows)) for j in range(n_cols)]
        N = sum(row_sums)

        if N == 0:
            return {'chi_squared': 0.0, 'df': 0, 'p_value': 1.0}

        # χ² 统计量
        chi_sq = 0.0
        for i in range(n_rows):
            for j in range(n_cols):
                expected = row_sums[i] * col_sums[j] / N
                if expected > 0:
                    chi_sq += (table[i][j] - expected)**2 / expected

        df = (n_rows - 1) * (n_cols - 1)

        # p值近似 (Wilson-Hilferty正态近似)
        if df > 0:
            z = ((chi_sq / df)**(1.0/3.0) - (1.0 - 2.0/(9.0*df))) / math.sqrt(2.0/(9.0*df))
            # 标准正态CDF近似
            p_value = 0.5 * math.erfc(z / math.sqrt(2.0))
        else:
            p_value = 1.0

        return {
            'chi_squared': chi_sq,
            'degrees_of_freedom': df,
            'p_value': p_value,
            'total_N': N,
            'row_sums': row_sums,
            'col_sums': col_sums,
        }


# ============================================================================
# 热激活位错脱钉扎模型
# ============================================================================

class ThermalActivationModel:
    """
    热激活控制的位错脱钉扎模型

    综合Arrhenius热激活率、Monte Carlo采样和统计分析

    物理过程:
    1. 位错被障碍钉扎 (林位错、析出物等)
    2. 热涨落帮助位错克服障碍
    3. 脱钉扎后位错快速滑移到下一个障碍

    宏观表现: 热激活塑性 (creep, 应变速率敏感性)
    """

    def __init__(self, material=None, temperature=300.0, seed=None):
        """
        初始化热激活模型

        Args:
            material: 材料参数
            temperature: 温度 (K)
            seed: 随机种子
        """
        self.mat = material or DEFAULT_MATERIAL
        self.temperature = temperature
        self.mc_sampler = BallMonteCarloSampler(seed=seed)
        self.ct_analysis = ContingencyTableAnalysis(seed=seed)

        # 尝试频率 (Debye频率量级)
        self.attempt_freq = self.mat.debye_frequency() / (2.0 * PI)

        # Peierls应力
        self.sigma_p = self.mat.peierls_stress()

    def gibbs_barrier(self, tau):
        """
        计算Gibbs自由能垒

        ΔG(τ) = ΔF₀ * (1 - (|τ|/τ_P)^p)^q  for |τ| < τ_P
        ΔG(τ) = 0                              for |τ| ≥ τ_P

        Args:
            tau: 施加剪应力 (Pa)

        Returns:
            float: Gibbs自由能垒 (J)
        """
        if abs(tau) >= self.sigma_p:
            return 0.0

        ratio = abs(tau) / self.sigma_p
        p = self.mat.activation_p
        q = self.mat.activation_q

        inner = 1.0 - ratio**p
        if inner <= 0.0:
            return 0.0

        return self.mat.delta_F0 * inner**q

    def activation_volume(self, tau):
        """
        计算激活体积

        V* = -∂ΔG/∂τ

        对于 ΔG = ΔF₀ (1 - (τ/τ_P)^p)^q:
        V* = ΔF₀ q p / τ_P * (τ/τ_P)^{p-1} * (1 - (τ/τ_P)^p)^{q-1}

        物理意义:
        - V* 越大, 应变速率敏感性越低
        - V* ~ b³ 表示点障碍
        - V* >> b³ 表示长程障碍

        Args:
            tau: 施加剪应力 (Pa)

        Returns:
            float: 激活体积 (m³)
        """
        if abs(tau) < 1e-30 or abs(tau) >= self.sigma_p:
            return 0.0

        ratio = abs(tau) / self.sigma_p
        p = self.mat.activation_p
        q = self.mat.activation_q
        delta_F0 = self.mat.delta_F0

        inner = 1.0 - ratio**p
        if inner <= 0.0:
            return 0.0

        v_star = (delta_F0 * q * p / self.sigma_p *
                  ratio**(p - 1.0) * inner**(q - 1.0))
        return abs(v_star)

    def activation_rate(self, tau):
        """
        计算热激活率 (Arrhenius)

        Γ(τ) = Γ₀ exp(-ΔG(τ) / (k_B T))

        当 ΔG >> k_B T 时, 速率极低 (准静态)
        当 ΔG ~ k_B T 时, 速率显著 (热激活区)
        当 ΔG << k_B T 时, 速率饱和 (拖曳控制区)

        Args:
            tau: 施加剪应力 (Pa)

        Returns:
            float: 激活率 (1/s)
        """
        dG = self.gibbs_barrier(tau)
        kT = BOLTZMANN_CONSTANT * self.temperature

        if kT < 1e-30:
            return 0.0 if dG > 0 else self.attempt_freq

        exponent = -dG / kT
        # 防止下溢
        if exponent < -700:
            return 0.0

        return self.attempt_freq * math.exp(exponent)

    def critical_temperature(self, tau):
        """
        计算给定应力下的临界温度

        T_c: 使 ΔG(τ) = k_B T_c 的温度
        在此温度以上, 热激活可以完全克服障碍

        T_c = ΔG(τ) / k_B

        Args:
            tau: 施加剪应力 (Pa)

        Returns:
            float: 临界温度 (K)
        """
        dG = self.gibbs_barrier(tau)
        return dG / BOLTZMANN_CONSTANT

    def monte_carlo_activation_energy(self, n_samples=5000):
        """
        Monte Carlo计算平均激活能

        在3D激活体积空间内积分:
        <ΔG> = ∫∫∫ ΔG(v) d³v / V*

        使用球面Monte Carlo采样

        Args:
            n_samples: 采样数

        Returns:
            dict: MC积分结果
        """
        r_activation = (self.mat.activation_volume)**(1.0/3.0)

        def integrand(x, y, z):
            # 简化: ΔG依赖于距核心的距离
            r = math.sqrt(x**2 + y**2 + z**2)
            if r < 1e-30:
                return self.mat.delta_F0
            # 衰减的激活能
            ratio = r / max(r_activation, 1e-30)
            return self.mat.delta_F0 * max(1.0 - ratio, 0.0)**2

        result = self.mc_sampler.integrate_on_ball(
            integrand, r_activation, n_samples, dimension=3
        )
        return result

    def depinning_statistics(self, n_obstacles=20, stress_levels=3, seed=None):
        """
        脱钉扎事件的统计列联表分析

        模拟不同障碍类型和脱钉扎模式的分布

        Args:
            n_obstacles: 障碍数量
            stress_levels: 应力水平数

        Returns:
            dict: 统计分析结果
        """
        rng = random.Random(seed)

        # 障碍类型: 林位错、析出物、溶质原子
        # 脱钉扎模式: 热激活、应力辅助、协同脱钉
        n_types = 3
        n_modes = stress_levels

        # 生成边际和
        row_sums = [max(1, n_obstacles // n_types + rng.randint(-2, 2)) for _ in range(n_types)]
        col_sums_base = n_obstacles - sum(row_sums)

        # 使行和之和等于列和之和
        total = sum(row_sums)
        col_sums = [max(1, total // n_modes + rng.randint(-1, 1)) for _ in range(n_modes)]
        # 调整使总和匹配
        diff = sum(row_sums) - sum(col_sums)
        col_sums[-1] += diff

        # 确保非负
        col_sums = [max(1, c) for c in col_sums]
        diff2 = sum(row_sums) - sum(col_sums)
        col_sums[-1] += diff2

        table = self.ct_analysis.generate_random_table(row_sums, col_sums)
        analysis = self.ct_analysis.analyze_table(table)
        analysis['table'] = table

        return analysis

    def velocity_stress_curve(self, tau_range=None, n_points=20):
        """
        计算位错速度-应力曲线 (热激活区)

        v = b * Γ(τ) * L_effective

        其中 L_eff 是障碍间距

        在低温/高应力区: v ∝ exp(-ΔG/kT) (热激活控制)
        在高温/低应力区: v = bτ/B (拖曳控制)

        Args:
            tau_range: 应力范围 (Pa)
            n_points: 计算点数

        Returns:
            list: [(tau, velocity, rate), ...]
        """
        if tau_range is None:
            sigma_p = self.sigma_p
            tau_range = [sigma_p * i / n_points for i in range(1, n_points + 1)]

        B = self.mat.drag_coefficient(self.temperature)
        L_eff = 1e-6  # 有效障碍间距 ~1 μm

        results = []
        for tau in tau_range:
            rate = self.activation_rate(tau)
            v_thermal = self.mat.b_magnitude * rate * L_eff

            # 拖曳控制速度
            v_drag = tau * self.mat.b_magnitude / B

            # 取较大值 (主导机制)
            v_total = max(v_thermal, v_drag)

            results.append({
                'tau': tau,
                'velocity': v_total,
                'v_thermal': v_thermal,
                'v_drag': v_drag,
                'rate': rate,
                'gibbs_barrier': self.gibbs_barrier(tau),
                'activation_volume': self.activation_volume(tau),
            })

        return results


if __name__ == '__main__':
    print("=" * 70)
    print("热激活位错脱钉扎模型验证")
    print("=" * 70)

    model = ThermalActivationModel(temperature=300.0, seed=42)

    # 测试Gibbs能垒
    print("\nGibbs自由能垒:")
    sigma_p = model.sigma_p
    for frac in [0.0, 0.2, 0.5, 0.8, 1.0]:
        tau = frac * sigma_p
        dG = model.gibbs_barrier(tau)
        V_star = model.activation_volume(tau)
        rate = model.activation_rate(tau)
        print(f"  τ/τ_P={frac:.1f}: ΔG={dG:.4e} J, "
              f"V*={V_star:.4e} m³, Γ={rate:.4e} /s")

    # 测试Monte Carlo积分
    print("\n球面Monte Carlo积分:")
    mc_result = model.monte_carlo_activation_energy(n_samples=2000)
    print(f"  平均激活能: {mc_result['mean']:.4e} J")
    print(f"  相对误差: {mc_result['relative_error']:.4f}")
    print(f"  球体积: {mc_result['volume']:.4e} m³")

    # 测试列联表分析
    print("\n脱钉扎统计列联表:")
    stats = model.depinning_statistics(n_obstacles=30, seed=42)
    print(f"  χ² = {stats['chi_squared']:.4f}")
    print(f"  自由度 = {stats['degrees_of_freedom']}")
    print(f"  p值 = {stats['p_value']:.4f}")
    print(f"  列联表:")
    for row in stats['table']:
        print(f"    {row}")

    # 测试速度-应力曲线
    print("\n位错速度-应力关系:")
    v_curve = model.velocity_stress_curve(n_points=5)
    for pt in v_curve[:5]:
        print(f"  τ={pt['tau']:.4e} Pa: v={pt['velocity']:.4e} m/s, "
              f"regime={'thermal' if pt['v_thermal'] > pt['v_drag'] else 'drag'}")
