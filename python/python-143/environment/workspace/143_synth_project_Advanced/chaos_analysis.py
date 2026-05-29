"""
chaos_analysis.py
=================
市场微观结构的混沌特征与分形维数分析

本模块基于以下种子项目融合:
- 1075_sierpinski_triangle_chaos: 迭代函数系统(IFS)与混沌映射 → 价格路径的混沌检测与分形维数

核心数学模型:
--------------
1.  迭代函数系统 (IFS) 与分形维数:
    设 {f_1, f_2, ..., f_n} 为压缩映射族, 满足:
        |f_i(x) - f_i(y)| ≤ s_i |x - y|,   s_i ∈ [0,1)
    对价格路径 {P_t}, 构造仿射变换:
        f_i(x) = A_i x + b_i
    其中 A_i = diag(0.5, 0.5) 为缩放矩阵, b_i 为平移向量.
    IFS 吸引子的 Hausdorff 维数 d_H 满足:
        Σ_i s_i^{d_H} = 1
    对均匀缩放 s_i = s:
        d_H = -ln(n) / ln(s)
    例如 Sierpinski 三角形: n=3, s=0.5 → d_H = ln(3)/ln(2) ≈ 1.585

2.  价格路径的盒维数 (Box-Counting Dimension):
    将价格路径覆盖在边长为 ε 的网格上, 计数覆盖盒数 N(ε).
    盒维数:
        d_B = -lim_{ε→0} ln(N(ε)) / ln(ε)
    实际计算采用双对数回归:
        ln(N) ≈ -d_B ln(ε) + C
    对布朗运动路径, d_B = 1.5 (一维).
    若 d_B > 1.5, 表明价格路径比随机游走更"粗糙", 可能存在混沌成分.

3.  最大 Lyapunov 指数:
    衡量相邻轨道的指数分离速率:
        λ_max = lim_{t→∞} lim_{d(0)→0} (1/t) ln( |d(t)| / |d(0)| )
    其中 d(t) 为两条相邻路径的距离.
    λ_max > 0 表明系统具有对初始条件的敏感依赖性 (混沌特征).
    对金融时间序列, 采用 Wolf 算法或 Rosenstein 算法估计.

4.  递归定量分析 (RQA):
    构造相空间嵌入后的递归矩阵:
        R_{ij} = Θ(ε - ||x_i - x_j||)
    其中 Θ 为 Heaviside 阶跃函数, ε 为阈值.
    递归率 (RR):
        RR = (1/N²) Σ_{i,j} R_{ij}
    确定性 (DET):
        DET = Σ_{l=l_min} l P(l) / Σ_{l=1} l P(l)
    其中 P(l) 为对角线长度为 l 的线段分布.

5.  Hurst 指数:
    用于检测长期记忆性:
        E[R(n)/S(n)] = C n^H
    其中 R(n) 为极差, S(n) 为标准差.
    H > 0.5: 持久性 (趋势延续)
    H = 0.5: 随机游走
    H < 0.5: 反持久性 (均值回归)
"""

import numpy as np
from typing import Tuple, Optional


class FractalDimension:
    """
    分形维数计算.
    """

    @staticmethod
    def box_counting_dimension(x: np.ndarray, y: np.ndarray,
                                epsilons: Optional[np.ndarray] = None) -> float:
        """
        盒计数维数.

        Parameters
        ----------
        x, y : np.ndarray
            路径坐标.
        epsilons : np.ndarray, optional
            盒尺寸序列.

        Returns
        -------
        dim : float
            估计的盒维数.
        """
        if len(x) != len(y) or len(x) < 10:
            return 1.0

        if epsilons is None:
            # 生成对数均匀分布的 eps
            log_eps = np.linspace(-3, 0, 20)
            epsilons = 10.0 ** log_eps

        counts = []
        valid_eps = []

        for eps in epsilons:
            if eps <= 0:
                continue
            # 计算覆盖路径所需的最小盒数
            x_bins = np.floor((x - np.min(x)) / eps).astype(int)
            y_bins = np.floor((y - np.min(y)) / eps).astype(int)
            boxes = set(zip(x_bins, y_bins))
            n_boxes = len(boxes)
            if n_boxes > 1:
                counts.append(n_boxes)
                valid_eps.append(eps)

        if len(counts) < 3:
            return 1.0

        counts = np.array(counts, dtype=float)
        valid_eps = np.array(valid_eps, dtype=float)

        # 双对数线性回归
        log_n = np.log(counts)
        log_eps = np.log(1.0 / valid_eps)

        # 最小二乘
        A = np.vstack([log_eps, np.ones(len(log_eps))]).T
        slope, intercept = np.linalg.lstsq(A, log_n, rcond=None)[0]
        return float(slope)


class LyapunovEstimator:
    """
    最大 Lyapunov 指数估计.
    """

    @staticmethod
    def rosenstein_algorithm(data: np.ndarray,
                              embed_dim: int = 5,
                              tau: int = 1,
                              max_steps: int = 50) -> float:
        """
        Rosenstein 算法估计最大 Lyapunov 指数.

        算法步骤:
        1.  相空间重构: x_i = [data_i, data_{i+τ}, ..., data_{i+(m-1)τ}]
        2.  对每个点 x_i, 找到最近邻 x_j (j ≠ i)
        3.  追踪距离 d_i(k) = ||x_{i+k} - x_{j+k}||
        4.  计算平均对数距离: y(k) = (1/q) Σ_i ln(d_i(k))
        5.  对 y(k) 线性拟合, 斜率即为 λ_max
        """
        n = len(data)
        if n < embed_dim * tau + max_steps:
            return 0.0

        # 相空间重构
        embedded = []
        for i in range(n - (embed_dim - 1) * tau):
            point = [data[i + j * tau] for j in range(embed_dim)]
            embedded.append(point)
        embedded = np.array(embedded)
        m = len(embedded)

        if m < 10:
            return 0.0

        # 找到每个点的最近邻
        nearest_dist = np.full(m, np.inf)
        nearest_idx = np.full(m, -1, dtype=int)

        for i in range(m):
            dists = np.linalg.norm(embedded - embedded[i], axis=1)
            dists[i] = np.inf
            nearest_idx[i] = np.argmin(dists)
            nearest_dist[i] = dists[nearest_idx[i]]

        # 追踪距离演化
        k_vals = []
        y_vals = []

        for k in range(1, min(max_steps, m)):
            divergences = []
            for i in range(m - k):
                j = nearest_idx[i]
                if j + k < m:
                    d_new = np.linalg.norm(embedded[i + k] - embedded[j + k])
                    if d_new > 0 and nearest_dist[i] > 0:
                        divergences.append(np.log(d_new))

            if len(divergences) > 5:
                k_vals.append(k)
                y_vals.append(np.mean(divergences))

        if len(k_vals) < 5:
            return 0.0

        # 线性拟合
        k_arr = np.array(k_vals, dtype=float)
        y_arr = np.array(y_vals, dtype=float)

        # 只用前 1/3 点估计 (避免饱和)
        cutoff = len(k_arr) // 3
        if cutoff < 3:
            cutoff = len(k_arr)

        A = np.vstack([k_arr[:cutoff], np.ones(cutoff)]).T
        slope, _ = np.linalg.lstsq(A, y_arr[:cutoff], rcond=None)[0]

        # 归一化到单位时间
        dt = 1.0
        return float(slope / dt)


class HurstExponent:
    """
    Hurst 指数计算.
    """

    @staticmethod
    def rescaled_range(data: np.ndarray,
                        max_lag: Optional[int] = None) -> float:
        """
        R/S 分析估计 Hurst 指数.

        对时间序列 {X_t}, 计算不同时间尺度 n 上的 R(n)/S(n):
            Y_k = Σ_{t=1}^k (X_t - μ_n)
            R(n) = max_{1≤k≤n} Y_k - min_{1≤k≤n} Y_k
            S(n) = sqrt( (1/n) Σ_{t=1}^n (X_t - μ_n)² )
            E[R/S] ~ C n^H
        """
        n = len(data)
        if n < 10:
            return 0.5

        if max_lag is None:
            max_lag = n // 4

        lags = []
        rs_values = []

        for lag in range(10, max_lag, max(1, max_lag // 20)):
            # 将数据分块
            n_chunks = n // lag
            if n_chunks < 2:
                continue

            rs_chunks = []
            for i in range(n_chunks):
                chunk = data[i * lag:(i + 1) * lag]
                mean_c = np.mean(chunk)
                dev_cum = np.cumsum(chunk - mean_c)
                R = np.max(dev_cum) - np.min(dev_cum)
                S = np.std(chunk)
                if S > 1e-12:
                    rs_chunks.append(R / S)

            if len(rs_chunks) > 0:
                lags.append(lag)
                rs_values.append(np.mean(rs_chunks))

        if len(lags) < 3:
            return 0.5

        log_lags = np.log(lags)
        log_rs = np.log(rs_values)
        A = np.vstack([log_lags, np.ones(len(log_lags))]).T
        H, _ = np.linalg.lstsq(A, log_rs, rcond=None)[0]
        return float(np.clip(H, 0.0, 1.0))


class ChaosAnalyzer:
    """
    综合混沌分析器.
    """

    def __init__(self):
        pass

    def analyze_price_path(self, prices: np.ndarray) -> dict:
        """
        对价格路径进行全面的混沌与分形分析.

        Returns
        -------
        metrics : dict
            包含盒维数、Lyapunov指数、Hurst指数等指标.
        """
        if len(prices) < 50:
            return {
                'box_dimension': 1.0,
                'lyapunov_max': 0.0,
                'hurst': 0.5,
                'returns_autocorr': 0.0,
            }

        # 价格路径作为 (t, price) 曲线
        t = np.arange(len(prices))
        # 归一化
        prices_norm = (prices - np.min(prices)) / (np.max(prices) - np.min(prices) + 1e-12)
        t_norm = t / len(prices)

        d_box = FractalDimension.box_counting_dimension(t_norm, prices_norm)

        # 对数收益率
        returns = np.diff(np.log(prices + 1e-12))
        lambda_max = LyapunovEstimator.rosenstein_algorithm(returns)
        H = HurstExponent.rescaled_range(returns)

        # 收益率自相关 (一阶)
        if len(returns) > 1 and np.std(returns) > 1e-12:
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            if np.isnan(autocorr):
                autocorr = 0.0
        else:
            autocorr = 0.0

        return {
            'box_dimension': d_box,
            'lyapunov_max': lambda_max,
            'hurst': H,
            'returns_autocorr': autocorr,
        }

    def regime_classification(self, metrics: dict) -> str:
        """
        基于混沌指标对市场 regime 分类.
        """
        H = metrics.get('hurst', 0.5)
        lam = metrics.get('lyapunov_max', 0.0)
        d_box = metrics.get('box_dimension', 1.0)

        if lam > 0.01 and d_box > 1.3:
            if H > 0.55:
                return "混沌趋势 (Chaotic Trending)"
            elif H < 0.45:
                return "混沌均值回归 (Chaotic Mean-Reverting)"
            else:
                return "弱混沌 (Weak Chaos)"
        else:
            if H > 0.55:
                return "随机趋势 (Random Trending)"
            elif H < 0.45:
                return "随机均值回归 (Random Mean-Reverting)"
            else:
                return "有效市场 (Efficient Market)"
