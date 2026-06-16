"""
统计分析模块
对 Berry curvature 计算结果进行统计分析和误差评估

数学背景：
Berry curvature 场的统计特性：
- 均值 <Ω> = (1/V_BZ) ∫∫∫ Ω(k) d³k
- 方差 Var(Ω) = <|Ω - <Ω>|²>
- 关联函数 C(r) = <Ω(k)·Ω(k+r)>

在 Weyl 点附近，Berry curvature 具有长尾分布：
P(Ω) ∝ Ω^{-5/2} (对于三维 Weyl 半金属)
"""

import numpy as np
from typing import Dict, List, Tuple


class StatisticalAnalyzer:
    """
    Berry curvature 统计分析器
    """

    def __init__(self):
        """初始化"""
        pass

    def compute_statistics(self, omega_data: np.ndarray) -> Dict:
        """
        计算 Berry curvature 场的基本统计量

        Parameters:
        -----------
        omega_data : np.ndarray
            Berry curvature 数据 (N,) 或 (N, 3)

        Returns:
        --------
        Dict
            统计量字典
        """
        if omega_data.ndim == 1:
            data = omega_data
        else:
            data = np.linalg.norm(omega_data, axis=1)

        stats = {
            'mean': np.mean(data),
            'std': np.std(data),
            'median': np.median(data),
            'min': np.min(data),
            'max': np.max(data),
            'skewness': self._skewness(data),
            'kurtosis': self._kurtosis(data),
            'iqr': np.percentile(data, 75) - np.percentile(data, 25)
        }

        return stats

    def _skewness(self, data: np.ndarray) -> float:
        """计算偏度"""
        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-15:
            return 0.0
        return np.mean(((data - mean) / std)**3)

    def _kurtosis(self, data: np.ndarray) -> float:
        """计算峰度 (超额峰度)"""
        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-15:
            return 0.0
        return np.mean(((data - mean) / std)**4) - 3.0

    def correlation_function(self, omega_field: np.ndarray,
                             k_points: np.ndarray,
                             max_distance: float = 0.5,
                             n_bins: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算 Berry curvature 空间关联函数

        C(r) = <Ω(k)·Ω(k+r)> - <Ω>²

        Parameters:
        -----------
        omega_field : np.ndarray
            Berry curvature 场 (N, 3)
        k_points : np.ndarray
            k 点坐标 (N, 3)
        max_distance : float
            最大关联距离
        n_bins : int
            距离分箱数

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            distances, correlation
        """
        n_points = len(k_points)

        # 计算平均
        omega_mean = np.mean(omega_field, axis=0)
        omega_fluct = omega_field - omega_mean

        # 距离分箱
        distances = np.linspace(0, max_distance, n_bins + 1)
        corr = np.zeros(n_bins)
        counts = np.zeros(n_bins)

        for i in range(n_points):
            for j in range(i + 1, n_points):
                # 计算距离 (考虑周期性)
                dk = k_points[j] - k_points[i]
                dk = dk - np.round(dk)  # 周期化
                dist = np.linalg.norm(dk)

                if dist < max_distance:
                    # 找到对应 bin
                    bin_idx = int(dist / max_distance * n_bins)
                    bin_idx = min(bin_idx, n_bins - 1)

                    # 关联
                    dot_prod = np.dot(omega_fluct[i], omega_fluct[j])
                    corr[bin_idx] += dot_prod
                    counts[bin_idx] += 1

        # 归一化
        mask = counts > 0
        corr[mask] /= counts[mask]

        # 距离中心
        dist_centers = (distances[:-1] + distances[1:]) / 2

        return dist_centers, corr

    def histogram_analysis(self, omega_data: np.ndarray,
                           n_bins: int = 50,
                           fit_power_law: bool = True) -> Dict:
        """
        Berry curvature 分布直方图分析

        Parameters:
        -----------
        omega_data : np.ndarray
            Berry curvature 数据
        n_bins : int
            分箱数
        fit_power_law : bool
            是否拟合幂律分布

        Returns:
        --------
        Dict
            直方图和拟合结果
        """
        if omega_data.ndim > 1:
            data = np.linalg.norm(omega_data, axis=1)
        else:
            data = omega_data

        # 直方图
        hist, bin_edges = np.histogram(data, bins=n_bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        result = {
            'histogram': hist,
            'bin_centers': bin_centers,
            'bin_edges': bin_edges
        }

        # 幂律拟合 P(Ω) ∝ Ω^{-α}
        if fit_power_law and len(data) > 10:
            # 选择正的部分
            mask = bin_centers > 0
            if np.sum(mask) > 3:
                log_x = np.log(bin_centers[mask])
                log_y = np.log(hist[mask] + 1e-15)

                # 线性拟合
                coeffs = np.polyfit(log_x, log_y, 1)
                alpha = -coeffs[0]

                result['power_law_exponent'] = alpha
                result['power_law_fit'] = np.exp(coeffs[1]) * bin_centers**(-alpha)

        return result

    def error_estimation(self, omega_fine: np.ndarray,
                         omega_coarse: np.ndarray) -> Dict:
        """
        误差估计 (网格细化法)

        通过比较粗细网格的结果估计数值误差

        Parameters:
        -----------
        omega_fine : np.ndarray
            细网格结果
        omega_coarse : np.ndarray
            粗网格结果

        Returns:
        --------
        Dict
            误差估计
        """
        # 插值粗网格到细网格 (简化处理)
        if len(omega_coarse) != len(omega_fine):
            # 假设可以简单比较
            n_min = min(len(omega_fine), len(omega_coarse))
            omega_fine = omega_fine[:n_min]
            omega_coarse = omega_coarse[:n_min]

        abs_error = np.abs(omega_fine - omega_coarse)
        rel_error = abs_error / (np.abs(omega_fine) + 1e-15)

        return {
            'max_abs_error': np.max(abs_error),
            'mean_abs_error': np.mean(abs_error),
            'max_rel_error': np.max(rel_error),
            'mean_rel_error': np.mean(rel_error),
            'rmse': np.sqrt(np.mean((omega_fine - omega_coarse)**2))
        }

    def bootstrap_confidence_interval(self, data: np.ndarray,
                                      statistic: str = 'mean',
                                      n_bootstrap: int = 1000,
                                      confidence: float = 0.95) -> Tuple[float, float, float]:
        """
        Bootstrap 置信区间估计

        Parameters:
        -----------
        data : np.ndarray
            原始数据
        statistic : str
            统计量 ('mean', 'median', 'std')
        n_bootstrap : int
            Bootstrap 采样数
        confidence : float
            置信水平

        Returns:
        --------
        Tuple[float, float, float]
            点估计, 下界, 上界
        """
        n = len(data)
        bootstrap_stats = []

        stat_func = {'mean': np.mean, 'median': np.median, 'std': np.std}[statistic]

        for _ in range(n_bootstrap):
            sample = np.random.choice(data, n, replace=True)
            bootstrap_stats.append(stat_func(sample))

        bootstrap_stats = np.array(bootstrap_stats)

        # 点估计
        point_estimate = stat_func(data)

        # 置信区间
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

        return point_estimate, lower, upper

    def topological_statistical_analysis(self, chern_numbers: np.ndarray,
                                         parameter_values: np.ndarray) -> Dict:
        """
        拓扑相变的统计分析

        分析 Chern 数随参数变化的统计特性

        Parameters:
        -----------
        chern_numbers : np.ndarray
            Chern 数序列
        parameter_values : np.ndarray
            参数值

        Returns:
        --------
        Dict
            拓扑统计分析结果
        """
        # 找到相变点 (Chern 数跳变)
        transitions = []
        for i in range(len(chern_numbers) - 1):
            if chern_numbers[i] != chern_numbers[i + 1]:
                transitions.append({
                    'parameter': (parameter_values[i] + parameter_values[i+1]) / 2,
                    'delta_chern': chern_numbers[i + 1] - chern_numbers[i]
                })

        # 相分布
        unique_chern, counts = np.unique(chern_numbers, return_counts=True)

        return {
            'transitions': transitions,
            'n_transitions': len(transitions),
            'phase_distribution': dict(zip(unique_chern.astype(int), counts)),
            'parameter_range': (np.min(parameter_values), np.max(parameter_values)),
            'mean_phase': np.mean(chern_numbers),
            'phase_variance': np.var(chern_numbers)
        }

    def generate_report(self, omega_data: np.ndarray,
                       k_points: np.ndarray = None) -> Dict:
        """
        生成完整的统计分析报告

        Parameters:
        -----------
        omega_data : np.ndarray
            Berry curvature 数据
        k_points : np.ndarray
            k 点坐标 (可选)

        Returns:
        --------
        Dict
            完整报告
        """
        report = {}

        # 基本统计
        print("[Statistics] 计算基本统计量...")
        report['basic_stats'] = self.compute_statistics(omega_data)

        # 直方图分析
        print("[Statistics] 直方图分析...")
        report['histogram'] = self.histogram_analysis(omega_data)

        # 关联函数 (如果有 k 点且尺寸匹配)
        if k_points is not None and len(k_points) == len(omega_data):
            print("[Statistics] 计算关联函数...")
            if omega_data.ndim == 1:
                omega_3d = np.column_stack([omega_data, np.zeros_like(omega_data), np.zeros_like(omega_data)])
            else:
                omega_3d = omega_data
            dist, corr = self.correlation_function(omega_3d, k_points)
            report['correlation'] = {'distances': dist, 'values': corr}
        elif k_points is not None:
            print(f"[Statistics] 跳过关联函数计算 (k 点数 {len(k_points)} 与数据点数 {len(omega_data)} 不匹配)")

        # Bootstrap 置信区间
        print("[Statistics] Bootstrap 置信区间...")
        if omega_data.ndim > 1:
            data_for_bootstrap = np.linalg.norm(omega_data, axis=1)
        else:
            data_for_bootstrap = omega_data
        pe, lb, ub = self.bootstrap_confidence_interval(data_for_bootstrap)
        report['bootstrap'] = {
            'point_estimate': pe,
            'confidence_interval': (lb, ub),
            'confidence_level': 0.95
        }

        return report
