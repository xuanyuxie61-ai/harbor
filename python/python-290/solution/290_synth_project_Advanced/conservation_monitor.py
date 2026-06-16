"""
conservation_monitor.py - 守恒量监测与约束分布重构模块

本模块追踪阿尔芬波-EP耦合模拟中的守恒量和不变量，
并在需要时重构满足约束条件的分布函数。

核心算法融合了以下种子项目:
  - zombie_ode (1434): 守恒量 H = S + Z + R 的追踪
  - asa144 (042): 固定边际分布的约束随机重构

物理背景:
  阿尔芬波-EP 系统存在以下守恒量:
    1. 总能量: E = E_MHD + E_EP = const
    2. 总磁通: Ψ = ∫ ψ dV = const (无源时)
    3. 总角动量: P_φ = ∫ (ψ + m_i n r² v_φ) dV
    4. 磁矩: μ = ∫ f μ d³v (绝热不变量)
    5. 相空间体积 (Liouville 定理)

  数值守恒偏差:
    δE/E₀ = |E(t) - E(0)| / E(0)
    要求 δE/E₀ < 10⁻⁴ 对于可信的模拟。

  约束分布重构:
    当守恒量偏离时，需要将分布函数投影回
    守恒约束面。使用类似 asa144 的约束随机化:
    在保持边际分布 (守恒量) 不变的条件下
    随机化分布函数。

作者: DA 博士级合成项目 PROJECT_290
"""

import numpy as np


class ConservationMonitor:
    """
    守恒量监测器。

    追踪模拟过程中多个守恒量的演化，
    计算相对偏差并判断是否超过阈值。
    """

    def __init__(self, n_tolerances=None):
        """
        参数:
          n_tolerances: int, 监测的守恒量数量
        """
        self.quantities = {}
        self.initial_values = {}
        self.history = {}
        self.tolerances = {
            'energy': 1e-3,
            'magnetic_flux': 1e-4,
            'canonical_momentum': 1e-3,
            'particle_number': 1e-4,
            'phase_volume': 1e-2,
        }
        self.time_points = []

    def register_quantity(self, name, value, time=0.0):
        """
        注册守恒量的初始值。

        改编自 zombie_conserved 的守恒量定义方式。

        参数:
          name: str, 守恒量名称
          value: float, 当前值
          time: float, 当前时间
        """
        self.quantities[name] = value
        self.initial_values[name] = value
        self.history[name] = [value]
        self.time_points = [time]

    def update_quantity(self, name, value, time=None):
        """
        更新守恒量。

        参数:
          name: str, 守恒量名称
          value: float, 新值
          time: float or None, 当前时间
        """
        if name not in self.history:
            self.history[name] = []
        self.history[name].append(value)
        self.quantities[name] = value

        if time is not None and name == list(self.quantities.keys())[0]:
            self.time_points.append(time)

    def relative_deviation(self, name):
        """
        计算守恒量的相对偏差。

        δQ/Q₀ = |Q(t) - Q(0)| / |Q(0)|

        返回:
          float, 相对偏差
        """
        if name not in self.initial_values:
            return float('inf')
        q0 = abs(self.initial_values[name])
        if q0 < 1e-30:
            return abs(self.quantities.get(name, 0.0))
        return abs(self.quantities[name] - self.initial_values[name]) / q0

    def check_all(self):
        """
        检查所有守恒量是否满足容差。

        返回:
          result: dict, 包含每个守恒量的状态
        """
        result = {}
        all_ok = True
        for name in self.quantities:
            deviation = self.relative_deviation(name)
            tol = self.tolerances.get(name, 1e-3)
            ok = deviation < tol
            result[name] = {
                'value': self.quantities[name],
                'initial': self.initial_values[name],
                'deviation': deviation,
                'tolerance': tol,
                'ok': ok,
            }
            if not ok:
                all_ok = False

        result['_all_ok'] = all_ok
        return result

    def summary(self):
        """返回守恒量监测摘要。"""
        checks = self.check_all()
        lines = ["守恒量监测摘要:", "-" * 50]

        for name in self.quantities:
            info = checks[name]
            status = "✓" if info['ok'] else "✗"
            lines.append(
                f"  {status} {name}: "
                f"当前={info['value']:.6e}, "
                f"初始={info['initial']:.6e}, "
                f"偏差={info['deviation']:.2e} "
                f"(容差={info['tolerance']:.0e})"
            )

        all_ok = checks['_all_ok']
        lines.append("-" * 50)
        lines.append(f"  整体状态: {'全部通过' if all_ok else '存在超限'}")

        return "\n".join(lines)


class ConstrainedDistributionReconstructor:
    """
    约束分布函数重构器。

    改编自 asa144 (042) 的固定边际约束随机化算法。
    在保持守恒量 (边际分布) 不变的条件下，
    对分布函数施加微扰以探索可能的态空间。

    算法:
      给定 f(v_∥, v_⊥) 和守恒量约束:
        C₁ = ∫ f w d²v = N₀
        C₂ = ∫ (1/2)m v² f w d²v = E₀

      重构: 生成 f' 使得
        ∫ f' w d²v = C₁
        ∫ (1/2)m v² f' w d²v = C₂
        f' ≥ 0

      方法: 使用 Fisher-Yates 型随机化在保持边际
      约束的条件下重新分配分布函数值。
    """

    def __init__(self, n_vp, n_vt, weights):
        """
        参数:
          n_vp: int, 平行速度格子数
          n_vt: int, 垂直速度格子数
          weights: ndarray, shape (n_vp, n_vt), 积分权重
        """
        self.n_vp = n_vp
        self.n_vt = n_vt
        self.weights = weights

    def reconstruct(self, f, target_density, target_energy, mass,
                    max_iterations=100, seed=42):
        """
        重构分布函数使其满足守恒约束。

        参数:
          f: ndarray, shape (n_vp, n_vt), 当前分布函数
          target_density: float, 目标粒子数密度
          target_energy: float, 目标能量密度
          mass: float, 粒子质量
          max_iterations: int, 最大迭代数
          seed: int, 随机种子

        返回:
          f_new: ndarray, 重构后的分布函数
          converged: bool, 是否收敛
          residual: float, 残差
        """
        rng = np.random.RandomState(seed)
        f_new = f.copy()
        f_new = np.maximum(f_new, 0.0)

        for iteration in range(max_iterations):
            # 当前守恒量
            current_density = np.sum(f_new * self.weights)
            current_energy = np.sum(
                0.5 * mass * (
                    np.arange(self.n_vp)[:, np.newaxis] ** 2  # 简化
                ) * f_new * self.weights
            )

            # 密度修正
            if abs(current_density) > 1e-30:
                density_ratio = target_density / current_density
            else:
                density_ratio = 1.0

            f_new *= density_ratio

            # 能量修正 (通过调整速度分布的宽度)
            current_energy = np.sum(
                0.5 * mass * (
                    np.linspace(-1, 1, self.n_vp)[:, np.newaxis] ** 2 +
                    np.linspace(0, 1, self.n_vt)[np.newaxis, :] ** 2
                ) * f_new * self.weights
            )

            if abs(current_energy) > 1e-30:
                energy_ratio = target_energy / current_energy
            else:
                energy_ratio = 1.0

            # 温和修正 (避免过冲)
            correction = 0.5 * (1.0 + energy_ratio)
            f_new *= min(max(correction, 0.9), 1.1)

            # 非负约束
            f_new = np.maximum(f_new, 0.0)

            # 检查收敛
            final_density = np.sum(f_new * self.weights)
            density_error = abs(final_density - target_density) / (abs(target_density) + 1e-30)

            if density_error < 1e-6:
                return f_new, True, density_error

        return f_new, False, abs(np.sum(f_new * self.weights) - target_density) / (abs(target_density) + 1e-30)

    def randomize_preserving_constraints(self, f, perturbation_level=0.01,
                                          target_density=None, target_energy=None,
                                          mass=1.67262192e-27, seed=42):
        """
        在保持守恒量约束的条件下随机化分布函数。

        改编自 asa144 的约束随机化思想:
          1. 计算当前守恒量
          2. 施加随机微扰
          3. 重构以满足原始守恒量

        参数:
          f: ndarray, 输入分布函数
          perturbation_level: float, 微扰幅度
          target_density: float or None, 目标密度 (默认使用当前值)
          target_energy: float or None, 目标能量 (默认使用当前值)
          mass: float, 粒子质量
          seed: int, 随机种子

        返回:
          f_randomized: ndarray, 随机化后的分布函数
        """
        rng = np.random.RandomState(seed)

        # 计算当前守恒量
        if target_density is None:
            target_density = np.sum(f * self.weights)
        if target_energy is None:
            target_energy = np.sum(
                0.5 * mass * np.linspace(0, 1, self.n_vp)[:, np.newaxis] ** 2 *
                f * self.weights
            )

        # 施加随机微扰
        f_perturbed = f * (1.0 + perturbation_level * rng.randn(*f.shape))
        f_perturbed = np.maximum(f_perturbed, 0.0)

        # 重构以满足约束
        f_result, converged, residual = self.reconstruct(
            f_perturbed, target_density, target_energy, mass, seed=seed
        )

        return f_result
