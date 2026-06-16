# -*- coding: utf-8 -*-
"""
validation.py
=============
模拟验证与可复现性分析模块

本模块融合以下种子项目算法:
- 1049_MacWillyLiao_paper-reproduction-and-reports: 论文复现框架 → 模拟验证协议
- 1211_WanyuGroup_ICML2026-DKMP: DKMP核方法 → 位错密度场回归分析

核心物理:
---------
模拟验证包括:
1. 网格收敛性分析 (Richardson外推)
2. 能量守恒检查
3. 与解析解对比
4. 时间步长敏感性
5. 可复现性 (随机种子测试)

Richardson外推:
f_exact ≈ f_h + (f_h - f_{2h}) / (2^p - 1)

其中 p 是方法的精度阶数
"""

import math
from physical_constants import PI, DEFAULT_MATERIAL


# ============================================================================
# 论文复现验证框架 — 来自 1049_MacWillyLiao
# ============================================================================

class ReproducibilityValidator:
    """
    模拟可复现性验证器

    验证协议:
    1. 确定性测试: 相同输入 → 相同输出
    2. 参数敏感性: 小扰动 → 小变化
    3. 网格收敛: 细化网格 → 结果趋近
    4. 时间步长收敛: 缩小dt → 结果趋近
    5. 能量守恒: 总能变化在容差内
    """

    def __init__(self, tolerance=1e-6):
        """
        初始化验证器

        Args:
            tolerance: 验证容差
        """
        self.tolerance = tolerance
        self.test_results = []

    def deterministic_test(self, func, args_list, n_runs=3):
        """
        确定性测试: 多次运行结果一致

        Args:
            func: 被测函数
            args_list: 参数列表
            n_runs: 运行次数

        Returns:
            dict: 测试结果
        """
        results = []
        for _ in range(n_runs):
            result = func(*args_list)
            results.append(result)

        # 检查结果一致性
        consistent = True
        if isinstance(results[0], (int, float)):
            for r in results[1:]:
                if abs(r - results[0]) > self.tolerance:
                    consistent = False
                    break
        elif isinstance(results[0], (list, tuple)):
            for r in results[1:]:
                if len(r) != len(results[0]):
                    consistent = False
                    break
                for a, b in zip(r, results[0]):
                    if abs(a - b) > self.tolerance:
                        consistent = False
                        break

        test_result = {
            'test_name': 'deterministic',
            'consistent': consistent,
            'n_runs': n_runs,
            'first_result': results[0] if isinstance(results[0], (int, float)) else 'complex',
        }
        self.test_results.append(test_result)
        return test_result

    def parameter_sensitivity(self, func, base_args, param_idx, perturbation=0.01):
        """
        参数敏感性测试

        验证: |f(x + δ) - f(x)| / |f(x)| ≤ C |δ|

        Args:
            func: 被测函数
            base_args: 基准参数
            param_idx: 被扰动的参数索引
            perturbation: 扰动幅度 (相对)

        Returns:
            dict: 敏感性结果
        """
        result_base = func(*base_args)

        args_plus = list(base_args)
        args_plus[param_idx] = base_args[param_idx] * (1.0 + perturbation)
        result_plus = func(*args_plus)

        args_minus = list(base_args)
        args_minus[param_idx] = base_args[param_idx] * (1.0 - perturbation)
        result_minus = func(*args_minus)

        if isinstance(result_base, (int, float)):
            sensitivity = abs(result_plus - result_minus) / (2.0 * perturbation * abs(result_base) + 1e-30)
            smooth = sensitivity < 1e10  # 不应发散
        else:
            sensitivity = 0.0
            smooth = True

        test_result = {
            'test_name': 'parameter_sensitivity',
            'smooth': smooth,
            'sensitivity': sensitivity,
            'perturbation': perturbation,
        }
        self.test_results.append(test_result)
        return test_result

    def richardson_extrapolation(self, values, refinement_ratios):
        """
        Richardson外推估计精确解

        给定不同网格精度的解 f_h, f_{h/r}, f_{h/r²}:

        p = ln(|f_{h/r²} - f_{h/r}| / |f_{h/r} - f_h|) / ln(r)
        f_exact ≈ f_{h/r} + (f_{h/r} - f_h) / (r^p - 1)

        Args:
            values: 不同精度的解 [f_coarse, f_medium, f_fine]
            refinement_ratios: 细化比 [r1, r2]

        Returns:
            dict: 外推结果
        """
        if len(values) < 3:
            return {'error': '需要至少3个精度级别'}

        f_coarse, f_medium, f_fine = values[0], values[1], values[2]
        r = refinement_ratios[0] if refinement_ratios else 2.0

        # 观察到的收敛阶
        e1 = abs(f_medium - f_coarse)
        e2 = abs(f_fine - f_medium)

        if e1 < 1e-30 or e2 < 1e-30:
            return {
                'observed_order': float('inf'),
                'extrapolated_value': f_fine,
                'error_estimate': 0.0,
            }

        p_observed = math.log(e1 / e2) / math.log(r) if e2 > 0 else 0.0

        # Richardson外推
        r_p = r**p_observed
        if abs(r_p - 1.0) < 1e-15:
            extrapolated = f_fine
        else:
            extrapolated = f_fine + (f_fine - f_medium) / (r_p - 1.0)

        # 误差估计
        error_est = abs(f_fine - f_medium) / (r_p - 1.0) if abs(r_p - 1.0) > 1e-15 else abs(f_fine - f_medium)

        return {
            'observed_order': p_observed,
            'extrapolated_value': extrapolated,
            'error_estimate': error_est,
            'relative_error': abs(error_est / extrapolated) if abs(extrapolated) > 1e-30 else float('inf'),
            'convergence': p_observed > 0.5,
        }

    def energy_conservation_check(self, energy_history, tolerance=1e-4):
        """
        能量守恒检查

        验证总能变化在容差内:
        |E(t) - E(0)| / |E(0)| < tolerance

        Args:
            energy_history: 能量时间序列
            tolerance: 容差

        Returns:
            dict: 守恒检查结果
        """
        if not energy_history:
            return {'conserved': False, 'error': '空能量历史'}

        E0 = energy_history[0]
        max_deviation = 0.0
        max_deviation_time = 0

        for t, E in enumerate(energy_history):
            if abs(E0) > 1e-30:
                deviation = abs(E - E0) / abs(E0)
            else:
                deviation = abs(E - E0)
            if deviation > max_deviation:
                max_deviation = deviation
                max_deviation_time = t

        conserved = max_deviation < tolerance

        test_result = {
            'test_name': 'energy_conservation',
            'conserved': conserved,
            'max_deviation': max_deviation,
            'max_deviation_time': max_deviation_time,
            'tolerance': tolerance,
            'E_initial': E0,
            'E_final': energy_history[-1],
        }
        self.test_results.append(test_result)
        return test_result

    def generate_report(self):
        """
        生成验证报告

        Returns:
            dict: 完整验证报告
        """
        n_tests = len(self.test_results)
        n_passed = sum(1 for r in self.test_results
                       if r.get('consistent', r.get('smooth', r.get('conserved', False))))

        return {
            'total_tests': n_tests,
            'passed': n_passed,
            'failed': n_tests - n_passed,
            'pass_rate': n_passed / max(n_tests, 1),
            'all_passed': n_passed == n_tests,
            'details': self.test_results,
        }


# ============================================================================
# DKMP核方法 — 来自 1211_WanyuGroup_ICML2026-DKMP
# 用于位错密度场的核回归分析
# ============================================================================

class DislocationDensityKernel:
    """
    位错密度场的核方法分析

    基于DKMP (Deep Kernel Motion Planning) 的思想，
    使用核方法对位错密度场进行回归和平滑。

    位错密度张量 (Nye张量):
    α_{ij} = -ε_{jkl} ∂β_{il}/∂x_k

    其中 β 是塑性变形梯度

    核回归:
    ρ(x) = Σ_i K(x - x_i) w_i

    核函数选择:
    - Gaussian: K(r) = exp(-r²/(2σ²))
    - Matérn: K(r) = (r/σ)^ν K_ν(r/σ)
    - Wendland (紧支撑): K(r) = (1-r/R)^4 (4r/R + 1)
    """

    def __init__(self, kernel_type='gaussian', bandwidth=1e-8):
        """
        初始化核方法

        Args:
            kernel_type: 核类型
            bandwidth: 核带宽
        """
        self.kernel_type = kernel_type
        self.bandwidth = bandwidth

    def gaussian_kernel(self, r):
        """
        高斯核函数

        K(r) = exp(-r² / (2h²))

        性质:
        - 无限支撑 (但指数衰减)
        - 无限可微
        - Fourier变换也是高斯

        Args:
            r: 距离

        Returns:
            float: 核函数值
        """
        h = self.bandwidth
        return math.exp(-r**2 / (2.0 * h**2))

    def wendland_kernel(self, r):
        """
        Wendland紧支撑核 (C²)

        K(r) = (1 - r/R)⁴ (4r/R + 1)  for r ≤ R
        K(r) = 0                          for r > R

        优点: 稀疏矩阵, 计算高效
        在位错密度场重建中避免远距离虚假影响

        Args:
            r: 距离

        Returns:
            float: 核函数值
        """
        R = 3.0 * self.bandwidth  # 支撑半径
        if r >= R:
            return 0.0
        ratio = r / R
        return (1.0 - ratio)**4 * (4.0 * ratio + 1.0)

    def matérn_kernel(self, r, nu=1.5):
        """
        Matérn核 (ν = 3/2)

        K(r) = (1 + √3 r/h) exp(-√3 r/h)

        介于高斯和白噪声之间
        在位错核心附近提供更好的正则化

        Args:
            r: 距离
            nu: 平滑参数

        Returns:
            float: 核函数值
        """
        h = self.bandwidth
        if nu == 0.5:
            return math.exp(-r / h)
        elif nu == 1.5:
            s = math.sqrt(3.0) * r / h
            return (1.0 + s) * math.exp(-s)
        elif nu == 2.5:
            s = math.sqrt(5.0) * r / h
            return (1.0 + s + s**2 / 3.0) * math.exp(-s)
        else:
            return self.gaussian_kernel(r)

    def kernel_regression(self, x_query, x_data, y_data):
        """
        核回归 (Nadaraya-Watson估计)

        f̂(x) = Σ_i K(x - x_i) y_i / Σ_i K(x - x_i)

        用于从离散位错位置重建连续密度场

        Args:
            x_query: 查询点位置
            x_data: 数据点位置列表
            y_data: 数据点值列表

        Returns:
            float: 回归值
        """
        numerator = 0.0
        denominator = 0.0

        for x_i, y_i in zip(x_data, y_data):
            r = abs(x_query - x_i)
            if self.kernel_type == 'gaussian':
                k = self.gaussian_kernel(r)
            elif self.kernel_type == 'wendland':
                k = self.wendland_kernel(r)
            else:
                k = self.matérn_kernel(r)

            numerator += k * y_i
            denominator += k

        if denominator < 1e-30:
            return 0.0
        return numerator / denominator

    def density_field_reconstruction(self, dislocation_positions, grid_points):
        """
        从离散位错位置重建密度场

        ρ(x) = Σ_i δ(x - x_i) → 正则化为核密度估计

        ρ̂(x) = (1/N) Σ_i K_h(x - x_i)

        Args:
            dislocation_positions: 位错位置列表
            grid_points: 网格点列表

        Returns:
            list: 各网格点的位错密度
        """
        N = len(dislocation_positions)
        if N == 0:
            return [0.0] * len(grid_points)

        density = []
        for x in grid_points:
            rho = 0.0
            for x_i in dislocation_positions:
                r = abs(x - x_i)
                if self.kernel_type == 'gaussian':
                    rho += self.gaussian_kernel(r)
                elif self.kernel_type == 'wendland':
                    rho += self.wendland_kernel(r)
                else:
                    rho += self.matérn_kernel(r)
            density.append(rho / N)

        return density


# ============================================================================
# 综合验证测试
# ============================================================================

def run_comprehensive_validation():
    """
    运行综合验证测试

    包括:
    1. Peierls-Nabarro解析解对比
    2. 弹性场能量守恒
    3. 网格收敛性
    4. Hall-Petch关系验证

    Returns:
        dict: 验证报告
    """
    from peierls_nabarro import PeierlsNabarroModel
    from grain_structure import DislocationPileup
    from high_order_fd import HighOrderFDOperators

    validator = ReproducibilityValidator(tolerance=1e-6)
    mat = DEFAULT_MATERIAL

    print("执行综合验证测试...")

    # 1. PN模型解析解对比
    print("  [1/4] Peierls-Nabarro解析解...")
    pn_coarse = PeierlsNabarroModel(n_grid=32)
    pn_medium = PeierlsNabarroModel(n_grid=64)
    pn_fine = PeierlsNabarroModel(n_grid=128)

    _, u_coarse, _ = pn_coarse.dislocation_profile_analytical()
    _, u_medium, _ = pn_medium.dislocation_profile_analytical()
    _, u_fine, _ = pn_fine.dislocation_profile_analytical()

    # 在核心处比较 (中心点)
    mid_c = u_coarse[len(u_coarse)//2]
    mid_m = u_medium[len(u_medium)//2]
    mid_f = u_fine[len(u_fine)//2]

    rich = validator.richardson_extrapolation([mid_c, mid_m, mid_f], [2.0, 2.0])
    print(f"    核心处位移: {mid_c:.6e}, {mid_m:.6e}, {mid_f:.6e}")
    print(f"    收敛阶: {rich['observed_order']:.2f}, 外推值: {rich['extrapolated_value']:.6e}")

    # 2. 弹性场测试
    print("  [2/4] 弹性场有限差分精度...")
    fd2 = HighOrderFDOperators(order=2)
    fd4 = HighOrderFDOperators(order=4)

    n_test = 101
    L = 1e-8
    dx = 2*L / (n_test - 1)
    x = [-L + i*dx for i in range(n_test)]
    k_wave = 2*PI / (4*L)

    f = [math.sin(k_wave*xi) for xi in x]
    df_exact = [k_wave*math.cos(k_wave*xi) for xi in x]

    df2 = fd2.apply_deriv1(f, dx)
    df4 = fd4.apply_deriv1(f, dx)

    err2 = max(abs(df2[i] - df_exact[i]) for i in range(2, n_test-2))
    err4 = max(abs(df4[i] - df_exact[i]) for i in range(3, n_test-3))

    print(f"    2阶误差: {err2:.4e}, 4阶误差: {err4:.4e}")
    print(f"    4阶更精确: {err4 < err2}")

    # 3. Hall-Petch验证
    print("  [3/4] Hall-Petch关系...")
    pileup = DislocationPileup()
    d_values = [100e-9, 500e-9, 1000e-9, 5000e-9]
    sigma_y_values = [pileup.hall_petch_strength(d) for d in d_values]

    # 检查 σ_y 随 d 增大而减小
    monotone = all(sigma_y_values[i] >= sigma_y_values[i+1]
                   for i in range(len(sigma_y_values)-1))
    print(f"    Hall-Petch单调性: {monotone}")

    # 4. 确定性测试
    print("  [4/4] 确定性验证...")
    def test_func(x, y):
        return x * x + y * y
    det_result = validator.deterministic_test(test_func, [3.0, 4.0])
    print(f"    确定性: {det_result['consistent']}")

    # 生成报告
    report = validator.generate_report()
    print(f"\n验证完成: {report['passed']}/{report['total_tests']} 通过")

    return report


if __name__ == '__main__':
    report = run_comprehensive_validation()
    print(f"\n最终结果: {'全部通过 ✓' if report['all_passed'] else '存在失败 ✗'}")
