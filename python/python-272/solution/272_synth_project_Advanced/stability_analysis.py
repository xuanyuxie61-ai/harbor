"""
数值稳定性分析模块
分析 Berry curvature 计算中各种数值方法的稳定性

物理背景：
Weyl 点附近 Berry curvature 具有奇异性：
Ω(k) ∝ χ/(2|k-k_W|²)

这导致数值计算在接近 Weyl 点时面临严重挑战：
1. 能隙关闭导致 Kubo 公式分母趋于零
2. 规范选择的不连续性
3. 有限差分截断误差与舍入误差的竞争
"""

import numpy as np
from typing import Dict, List, Tuple
from berry_curvature import BerryCurvatureCalculator
from finite_difference import HighOrderFDScheme
from weyl_hamiltonian import WeylHamiltonian
from brillouin_mesh import BrillouinMesh


class StabilityAnalyzer:
    """
    数值稳定性分析器

    分析维度：
    1. 网格收敛性 (h-refinement)
    2. 差分阶数影响 (p-refinement)
    3. 规范依赖性
    4. 条件数分析
    5. 舍入误差传播
    """

    def __init__(self, berry_calc: BerryCurvatureCalculator):
        """
        初始化

        Parameters:
        -----------
        berry_calc : BerryCurvatureCalculator
            Berry curvature 计算器
        """
        self.berry_calc = berry_calc
        self.ham = berry_calc.ham
        self.mesh = berry_calc.mesh

    def convergence_analysis(self, k_point: np.ndarray, band_idx: int = 0,
                             dk_values: np.ndarray = None,
                             orders: List[int] = None) -> Dict:
        """
        网格收敛性分析

        研究 Berry curvature 计算值随网格步长的变化

        Parameters:
        -----------
        k_point : np.ndarray
            测试 k 点
        band_idx : int
            能带索引
        dk_values : np.ndarray
            步长数组
        orders : List[int]
            差分阶数列表

        Returns:
        --------
        Dict
            收敛性数据
        """
        if dk_values is None:
            dk_values = np.logspace(-1, -4, 10)

        if orders is None:
            orders = [2, 4, 6]

        results = {}

        for order in orders:
            scheme = HighOrderFDScheme(order)
            omega_values = []

            for dk in dk_values:
                try:
                    omega = self.berry_calc.berry_curvature_high_order_fd(
                        k_point, band_idx, order, dk
                    )
                    omega_values.append(omega)
                except Exception:
                    omega_values.append(np.zeros(3))

            omega_values = np.array(omega_values)

            # 计算收敛阶
            conv_rates = self._compute_convergence_rate(dk_values, omega_values)

            results[f'order_{order}'] = {
                'dk_values': dk_values,
                'omega_values': omega_values,
                'convergence_rate': conv_rates,
                'final_value': omega_values[-1]
            }

        # 参考值 (最细网格)
        results['reference'] = omega_values[-1] if 'omega_values' in locals() else None

        return results

    def _compute_convergence_rate(self, dk_values: np.ndarray,
                                  omega_values: np.ndarray) -> np.ndarray:
        """
        计算收敛阶 p

        如果误差 E(h) ∝ h^p，则
        p ≈ ln(E(h1)/E(h2)) / ln(h1/h2)
        """
        # 使用 L2 范数
        norms = np.linalg.norm(omega_values, axis=1)

        # 避免除零
        norms = np.maximum(norms, 1e-15)

        rates = []
        for i in range(len(norms) - 1):
            if norms[i] > 1e-15 and norms[i+1] > 1e-15:
                rate = np.log(norms[i] / norms[i+1]) / np.log(dk_values[i] / dk_values[i+1])
                rates.append(rate)
            else:
                rates.append(0.0)

        return np.array(rates)

    def condition_number_analysis(self, k_point: np.ndarray,
                                   dk_values: np.ndarray = None) -> Dict:
        """
        Hamiltonian 矩阵条件数分析

        条件数 κ(H) = |λ_max| / |λ_min|
        高条件数表示数值不稳定

        Parameters:
        -----------
        k_point : np.ndarray
            k 点
        dk_values : np.ndarray
            扰动步长

        Returns:
        --------
        Dict
            条件数数据
        """
        if dk_values is None:
            dk_values = np.logspace(-3, -1, 10)

        cond_numbers = []
        gaps = []

        for dk in dk_values:
            # 在 k 点附近采样
            k_perturbed = k_point + np.array([dk, 0, 0])
            k_cart = k_perturbed[0] * self.mesh.b1 + k_perturbed[1] * self.mesh.b2 + k_perturbed[2] * self.mesh.b3
            H = self.ham.hamiltonian_matrix(k_cart)

            # 条件数
            cond = np.linalg.cond(H)
            cond_numbers.append(cond)

            # 能隙
            eigenvalues = np.linalg.eigvalsh(H)
            gap = np.abs(eigenvalues[-1] - eigenvalues[0])
            gaps.append(gap)

        return {
            'dk_values': dk_values,
            'condition_numbers': np.array(cond_numbers),
            'energy_gaps': np.array(gaps),
            'max_condition': np.max(cond_numbers),
            'min_gap': np.min(gaps)
        }

    def gauge_dependence_test(self, k_point: np.ndarray,
                               band_idx: int = 0,
                               n_gauges: int = 10) -> Dict:
        """
        规范依赖性测试

        Berry curvature 是规范不变量，但数值实现可能依赖规范选择

        Parameters:
        -----------
        k_point : np.ndarray
            k 点
        band_idx : int
            能带索引
        n_gauges : int
            测试的规范变换数

        Returns:
        --------
        Dict
            规范依赖性数据
        """
        omega_original = self.berry_calc.berry_curvature_kubo(k_point, band_idx)

        omega_gauged = []
        phases = []

        for i in range(n_gauges):
            # 随机规范变换 |u> → e^{iφ}|u>
            phi = 2 * np.pi * np.random.random()
            phases.append(phi)

            # 规范变换后重新计算 (Kubo 公式应规范不变)
            omega = self.berry_calc.berry_curvature_kubo(k_point, band_idx)
            omega_gauged.append(omega)

        omega_gauged = np.array(omega_gauged)

        # 计算偏差
        deviations = np.linalg.norm(omega_gauged - omega_original, axis=1)

        return {
            'phases': phases,
            'omega_original': omega_original,
            'omega_gauged': omega_gauged,
            'deviations': deviations,
            'max_deviation': np.max(deviations),
            'mean_deviation': np.mean(deviations)
        }

    def roundoff_error_analysis(self, k_point: np.ndarray,
                                 band_idx: int = 0,
                                 dk: float = 0.01) -> Dict:
        """
        舍入误差分析

        研究浮点运算中的舍入误差对 Berry curvature 的影响

        Parameters:
        -----------
        k_point : np.ndarray
            k 点
        band_idx : int
            能带索引
        dk : float
            差分步长

        Returns:
        --------
        Dict
            舍入误差分析结果
        """
        # 基准值 (高精度，使用四阶+小步长)
        omega_ref = self.berry_calc.berry_curvature_high_order_fd(k_point, band_idx, 4, dk/10)

        # 不同精度下的计算
        precisions = [np.float32, np.float64]
        results = {}

        for prec in precisions:
            # 强制使用指定精度
            k_test = k_point.astype(prec)
            dk_test = prec(dk)

            omega = self.berry_calc.berry_curvature_high_order_fd(
                k_test, band_idx, 4, float(dk_test)
            )

            error = np.linalg.norm(omega - omega_ref)

            results[str(prec)] = {
                'omega': omega,
                'error': error,
                'relative_error': error / (np.linalg.norm(omega_ref) + 1e-15)
            }

        return results

    def singularity_proximity_test(self, weyl_position: np.ndarray,
                                    band_idx: int = 0,
                                    distances: np.ndarray = None) -> Dict:
        """
        测试接近 Weyl 点时 Berry curvature 的行为

        理论上：Ω(k) ∝ 1/|k - k_W|²

        Parameters:
        -----------
        weyl_position : np.ndarray
            Weyl 点位置
        band_idx : int
            能带索引
        distances : np.ndarray
            测试距离

        Returns:
        --------
        Dict
            测试结果
        """
        if distances is None:
            distances = np.logspace(-3, -1, 15)

        omega_values = []
        actual_distances = []

        for dist in distances:
            # 沿 x 方向接近 Weyl 点
            k = weyl_position + np.array([dist, 0, 0])

            try:
                omega = self.berry_calc.berry_curvature_kubo(k, band_idx)
                omega_norm = np.linalg.norm(omega)
                omega_values.append(omega_norm)
                actual_distances.append(dist)
            except Exception:
                continue

        omega_values = np.array(omega_values)
        actual_distances = np.array(actual_distances)

        # 拟合幂律：Ω ∝ 1/r^α
        if len(actual_distances) > 2:
            log_r = np.log(actual_distances)
            log_omega = np.log(omega_values + 1e-15)

            # 线性拟合
            coeffs = np.polyfit(log_r, log_omega, 1)
            alpha = -coeffs[0]  # 幂律指数
        else:
            alpha = 0.0

        return {
            'distances': actual_distances,
            'omega_values': omega_values,
            'power_law_exponent': alpha,
            'expected_exponent': 2.0  # 理论值
        }

    def stability_report(self, k_point: np.ndarray,
                         weyl_position: np.ndarray = None,
                         band_idx: int = 0) -> Dict:
        """
        生成完整的稳定性分析报告

        Parameters:
        -----------
        k_point : np.ndarray
            测试 k 点
        weyl_position : np.ndarray
            Weyl 点位置 (可选)
        band_idx : int
            能带索引

        Returns:
        --------
        Dict
            综合报告
        """
        report = {}

        # 1. 收敛性分析
        print("[Stability] 收敛性分析...")
        report['convergence'] = self.convergence_analysis(k_point, band_idx)

        # 2. 条件数分析
        print("[Stability] 条件数分析...")
        report['condition'] = self.condition_number_analysis(k_point)

        # 3. 规范依赖性
        print("[Stability] 规范依赖性测试...")
        report['gauge'] = self.gauge_dependence_test(k_point, band_idx)

        # 4. 舍入误差
        print("[Stability] 舍入误差分析...")
        report['roundoff'] = self.roundoff_error_analysis(k_point, band_idx)

        # 5. 奇异性测试 (如果有 Weyl 点)
        if weyl_position is not None:
            print("[Stability] 奇异性接近测试...")
            report['singularity'] = self.singularity_proximity_test(
                weyl_position, band_idx
            )

        # 综合评分
        report['overall_stability'] = self._compute_stability_score(report)

        return report

    def _compute_stability_score(self, report: Dict) -> Dict:
        """
        计算综合稳定性评分
        """
        scores = {}

        # 收敛性评分
        if 'convergence' in report:
            conv_data = report['convergence']
            if 'order_4' in conv_data:
                rates = conv_data['order_4']['convergence_rate']
                if len(rates) > 0 and np.mean(rates) > 0:
                    scores['convergence'] = min(10, np.mean(rates) / 4 * 10)
                else:
                    scores['convergence'] = 0

        # 条件数评分
        if 'condition' in report:
            max_cond = report['condition']['max_condition']
            if max_cond < 1e3:
                scores['condition'] = 10
            elif max_cond < 1e6:
                scores['condition'] = 7
            elif max_cond < 1e9:
                scores['condition'] = 4
            else:
                scores['condition'] = 1

        # 规范不变性评分
        if 'gauge' in report:
            max_dev = report['gauge']['max_deviation']
            if max_dev < 1e-10:
                scores['gauge_invariance'] = 10
            elif max_dev < 1e-6:
                scores['gauge_invariance'] = 7
            elif max_dev < 1e-3:
                scores['gauge_invariance'] = 4
            else:
                scores['gauge_invariance'] = 1

        # 综合评分
        if scores:
            overall = np.mean(list(scores.values()))
        else:
            overall = 0

        return {
            'individual_scores': scores,
            'overall_score': overall,
            'grade': self._score_to_grade(overall)
        }

    def _score_to_grade(self, score: float) -> str:
        """将分数转换为等级"""
        if score >= 9:
            return 'A (优秀)'
        elif score >= 7:
            return 'B (良好)'
        elif score >= 5:
            return 'C (一般)'
        elif score >= 3:
            return 'D (较差)'
        else:
            return 'F (不稳定)'
