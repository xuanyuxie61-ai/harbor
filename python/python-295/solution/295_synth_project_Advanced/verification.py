#!/usr/bin/env python3
"""
verification.py
===============
数值验证与形式化验证模块，融合种子项目:
  - [1297] FormalCellular: 形式化验证, 配置空间覆盖分析, CDF 分析
  - [619] kepler_perturbed_ode: 哈密顿守恒检验

验证方法:
  1. Method of Manufactured Solutions (MMS)
     对离散算子验证收敛阶数
  2. 守恒律检验
     质量, 动量, 能量守恒
  3. 哈密顿量守恒 (辛积分)
  4. 形式化配置空间覆盖
     验证参数空间的覆盖率和鲁棒性
"""

import numpy as np


class ManufacturedSolutionVerifier:
    """
    人工解方法 (MMS) 验证器。
    通过构造已知精确解, 验证数值方法的收敛阶。

    步骤:
      1. 选择精确解 u_exact(x,t)
      2. 代入控制方程得到源项 f(x,t)
      3. 在不同分辨率下数值求解
      4. 计算误差 ||u_num - u_exact||
      5. 验证误差收敛率
    """

    @staticmethod
    def heat_equation_mms(N_list, t_final=0.1, alpha=1.0):
        """
        热方程 MMS 验证:
          ∂u/∂t = α ∂²u/∂x² + f(x,t)
        选择:
          u_exact(x,t) = sin(πx) exp(-α π² t) + sin(2πx) exp(-4α π² t)
        源项:
          f(x,t) = 0 (因为 u_exact 是齐次方程的解)
        """
        from high_order_fd import HighOrderFD

        errors = []
        dx_list = []

        for N in N_list:
            dx = 1.0 / (N - 1)
            x = np.linspace(0, 1, N)

            # 精确解
            u_exact = np.sin(np.pi * x) * np.exp(-alpha * np.pi ** 2 * t_final) + \
                      np.sin(2 * np.pi * x) * np.exp(-4 * alpha * np.pi ** 2 * t_final)

            # 数值解 (简单 FTCS 显式)
            u = np.sin(np.pi * x) + np.sin(2 * np.pi * x)  # t=0
            dt = 0.4 * dx ** 2 / alpha  # 稳定性条件
            n_steps = int(t_final / dt)
            dt = t_final / n_steps

            for _ in range(n_steps):
                d2u = HighOrderFD.second_derivative_central(u, dx, order=2)
                u = u + dt * alpha * d2u
                u[0] = 0.0
                u[-1] = 0.0

            error = np.sqrt(np.mean((u - u_exact) ** 2))
            errors.append(error)
            dx_list.append(dx)

        # 计算收敛率
        rates = []
        for i in range(1, len(errors)):
            if errors[i] > 1e-30 and errors[i - 1] > 1e-30:
                rate = np.log(errors[i - 1] / errors[i]) / np.log(dx_list[i - 1] / dx_list[i])
                rates.append(rate)
            else:
                rates.append(0.0)

        return {
            'N': N_list,
            'dx': dx_list,
            'errors': errors,
            'rates': rates,
            'expected_order': 2  # FTCS 二阶
        }

    @staticmethod
    def advection_mms(N_list, t_final=1.0, c=1.0):
        """
        对流方程 MMS 验证:
          ∂u/∂t + c ∂u/∂x = 0
        精确解: u(x,t) = sin(2π(x - ct))
        周期边界条件, 域 [0, 1]
        """
        from high_order_fd import HighOrderFD

        errors = []
        dx_list = []

        for N in N_list:
            dx = 1.0 / N
            x = np.linspace(0, 1 - dx, N)

            u = np.sin(2 * np.pi * x)
            u_exact = np.sin(2 * np.pi * (x - c * t_final))

            dt = 0.5 * dx / abs(c)
            n_steps = int(t_final / dt)
            dt = t_final / n_steps

            for _ in range(n_steps):
                # 一阶迎风格式
                if c > 0:
                    u_new = u - c * dt / dx * (u - np.roll(u, 1))
                else:
                    u_new = u - c * dt / dx * (np.roll(u, -1) - u)
                u = u_new

            error = np.sqrt(np.mean((u - u_exact) ** 2))
            errors.append(error)
            dx_list.append(dx)

        rates = []
        for i in range(1, len(errors)):
            if errors[i] > 1e-30 and errors[i - 1] > 1e-30:
                rate = np.log(errors[i - 1] / errors[i]) / np.log(dx_list[i - 1] / dx_list[i])
                rates.append(rate)

        return {'N': N_list, 'dx': dx_list, 'errors': errors, 'rates': rates}


class ConservationChecker:
    """
    守恒律检验器。
    验证数值解是否满足质量、动量、能量守恒。
    """

    @staticmethod
    def check_mass_conservation(rho, u, dr, dz, r, expected_mass):
        """
        质量守恒:
          M = ∫∫ ρ r dr dz (轴对称)
        相对误差: |M_num - M_exact| / M_exact
        """
        # 2D 积分 (梯形法则)
        integrand = rho * r
        mass = np.trapz(np.trapz(integrand, dz, axis=1), dr)
        rel_error = abs(mass - expected_mass) / max(abs(expected_mass), 1e-30)
        return {'mass': mass, 'expected': expected_mass, 'rel_error': rel_error}

    @staticmethod
    def check_momentum_conservation(rho, vr, vz, dr, dz, r, expected_mom_r, expected_mom_z):
        """
        动量守恒:
          P_r = ∫∫ ρ v_r r dr dz
          P_z = ∫∫ ρ v_z r dr dz
        """
        mom_r = np.trapz(np.trapz(rho * vr * r, dz, axis=1), dr)
        mom_z = np.trapz(np.trapz(rho * vz * r, dz, axis=1), dr)
        rel_error_r = abs(mom_r - expected_mom_r) / max(abs(expected_mom_r), 1e-30)
        rel_error_z = abs(mom_z - expected_mom_z) / max(abs(expected_mom_z), 1e-30)
        return {
            'mom_r': mom_r, 'mom_z': mom_z,
            'rel_error_r': rel_error_r, 'rel_error_z': rel_error_z
        }

    @staticmethod
    def check_energy_conservation(rho, e_int, vr, vz, dr, dz, r, expected_energy):
        """
        能量守恒:
          E = ∫∫ ρ (e_int + ½(v_r² + v_z²)) r dr dz
        """
        ke = 0.5 * (vr ** 2 + vz ** 2)
        total_e = rho * (e_int + ke) * r
        energy = np.trapz(np.trapz(total_e, dz, axis=1), dr)
        rel_error = abs(energy - expected_energy) / max(abs(expected_energy), 1e-30)
        return {'energy': energy, 'expected': expected_energy, 'rel_error': rel_error}


class HamiltonianConservationChecker:
    """
    哈密顿量守恒检验 (融合 [619] kepler_perturbed_ode)。
    用于评估辛积分器的长期保能性。

    对哈密顿系统 H(q, p) = T(p) + V(q):
      辛积分器: |H(t) - H(0)| 有界 (不长期漂移)
      非辛积分器: H 可能长期漂移
    """

    @staticmethod
    def kepler_hamiltonian(q, p):
        """
        开普勒问题哈密顿量 (融合 [619]):
          H = ½(p₁² + p₂²) - 1/√(q₁² + q₂²)
        """
        r = np.sqrt(q[0] ** 2 + q[1] ** 2)
        return 0.5 * (p[0] ** 2 + p[1] ** 2) - 1.0 / max(r, 1e-30)

    @staticmethod
    def perturbed_kepler_hamiltonian(q, p, delta=0.005):
        """
        受扰开普勒哈密顿量 (融合 [619]):
          H = ½(p₁² + p₂²) - 1/r - δ/(2r³)
        """
        r = np.sqrt(q[0] ** 2 + q[1] ** 2)
        return (0.5 * (p[0] ** 2 + p[1] ** 2) -
                1.0 / max(r, 1e-30) -
                delta / (2.0 * max(r, 1e-30) ** 3))

    @staticmethod
    def check_energy_drift(H_history, method_name=''):
        """
        检查能量漂移。
        返回:
            dict: {
                'H_initial': 初始哈密顿量,
                'H_final': 最终哈密顿量,
                'max_drift': 最大漂移,
                'rms_drift': RMS 漂移,
                'bounded': 是否有界
            }
        """
        H = np.asarray(H_history)
        H0 = H[0]
        drift = H - H0
        max_drift = np.max(np.abs(drift))
        rms_drift = np.sqrt(np.mean(drift ** 2))

        # 判断是否有界: 检查是否存在长期趋势
        n = len(drift)
        if n < 10:
            bounded = True
        else:
            # 线性拟合漂移
            t = np.arange(n, dtype=np.float64)
            slope = np.polyfit(t, drift, 1)[0]
            bounded = abs(slope) < 1e-6 * abs(H0)

        return {
            'H_initial': H0,
            'H_final': H[-1],
            'max_drift': max_drift,
            'rms_drift': rms_drift,
            'bounded': bounded,
            'relative_max_drift': max_drift / max(abs(H0), 1e-30),
            'method': method_name
        }


class ConfigurationSpaceAnalyzer:
    """
    形式化配置空间分析 (融合 [1297] FormalCellular)。
    分析参数空间的覆盖率和鲁棒性。

    方法:
      1. 在参数空间中均匀采样
      2. 对每个采样点运行模拟
      3. 分析结果的覆盖度 CDF
      4. 识别临界区域和失败模式
    """

    @staticmethod
    def sample_parameter_space(param_ranges, n_samples=100, seed=42):
        """
        参数空间均匀采样。

        参数:
            param_ranges: dict {param_name: (min, max)}
            n_samples: 采样数
        返回:
            samples: list of dicts
        """
        rng = np.random.default_rng(seed)
        samples = []
        for _ in range(n_samples):
            sample = {}
            for name, (lo, hi) in param_ranges.items():
                sample[name] = rng.uniform(lo, hi)
            samples.append(sample)
        return samples

    @staticmethod
    def compute_coverage_cdf(values, n_bins=50):
        """
        计算经验 CDF (累积分布函数)。
        融合 [1297] 中的 CDF 分析。
        """
        values = np.sort(np.asarray(values))
        n = len(values)
        cdf = np.arange(1, n + 1) / n
        return values, cdf

    @staticmethod
    def robustness_analysis(results, threshold):
        """
        鲁棒性分析: 计算满足阈值的参数区域比例。
        """
        results = np.asarray(results)
        n_pass = np.sum(results <= threshold)
        coverage = n_pass / len(results)
        return {
            'n_total': len(results),
            'n_pass': n_pass,
            'coverage': coverage,
            'threshold': threshold
        }


def run_full_verification():
    """
    运行完整验证流程。
    """
    print("=" * 60)
    print("  数值验证报告")
    print("=" * 60)

    # 1. 热方程 MMS
    print("\n[1] 热方程 MMS 验证:")
    mms = ManufacturedSolutionVerifier()
    result = mms.heat_equation_mms([11, 21, 41, 81])
    for i, N in enumerate(result['N']):
        err = result['errors'][i]
        rate = result['rates'][i - 1] if i > 0 else 0
        print(f"  N={N:3d}: error = {err:.4e}, rate = {rate:.2f}")

    # 2. 哈密顿量守恒
    print("\n[2] 哈密顿量守恒 (受扰开普勒):")
    from time_integrator import StormerVerlet
    # V(r) = -1/r - δ/(2r³)
    delta = 0.005

    def grad_V(q):
        r = np.sqrt(q[0] ** 2 + q[1] ** 2)
        r3 = max(r ** 3, 1e-30)
        r5 = max(r ** 5, 1e-30)
        # ∇V = q/r³ + 3δ q/(2r⁵)
        return q / r3 + 3 * delta * q / (2 * r5)

    sv = StormerVerlet(grad_V, mass=1.0, dt=0.01)
    q = np.array([1.0, 0.0])
    p = np.array([0.0, 1.0])
    state = (q, p)
    H_history = []

    for _ in range(1000):
        H = HamiltonianConservationChecker.perturbed_kepler_hamiltonian(state[0], state[1], delta)
        H_history.append(H)
        state, _ = sv.step(0, state)

    checker = HamiltonianConservationChecker()
    energy_report = checker.check_energy_drift(H_history, 'Stormer-Verlet')
    print(f"  H(0) = {energy_report['H_initial']:.8f}")
    print(f"  H(end) = {energy_report['H_final']:.8f}")
    print(f"  Max drift = {energy_report['max_drift']:.4e}")
    print(f"  Bounded = {energy_report['bounded']}")

    # 3. 配置空间覆盖
    print("\n[3] 配置空间覆盖分析:")
    param_ranges = {
        'drive_asymmetry': (0.001, 0.05),
        'roughness': (1e-6, 1e-4),
        'velocity': (2e7, 5e7),
    }
    csa = ConfigurationSpaceAnalyzer()
    samples = csa.sample_parameter_space(param_ranges, n_samples=50)
    print(f"  生成 {len(samples)} 个参数空间采样点")

    return {'mms': result, 'energy': energy_report, 'n_samples': len(samples)}
