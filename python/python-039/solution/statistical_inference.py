"""
statistical_inference.py
重离子碰撞数据的统计推断与假设检验

基于种子项目:
- 051_asa243: 非中心t分布累积概率函数
- 095_bisection_integer: 整数二分查找 → 统计量分位数搜索

物理应用:
1. 碰撞中心度的统计分类
2. 流系数的显著性检验
3. 相变临界温度的置信区间估计
4. 系统误差传播的统计建模

数学模型:
1. 非中心t分布: T ~ t_{df, δ}
   用于检验存在系统涨落时的信号显著性
2. 二分查找: 快速定位统计量的临界值
"""

import numpy as np
from typing import Tuple, Optional
from scipy.special import betainc, gammaln


class NonCentralTDistribution:
    """
    非中心t分布: 用于QGP涨落信号的统计推断。

    概率密度:
    f(t; ν, δ) = Σ_{j=0}^∞ [e^{-δ²/2} (δ²/2)^j / j!]
                 · Γ((ν+1+2j)/2) / [√(πν) Γ((ν+2j)/2)]
                 · (ν/(ν+t²))^{(ν+1+2j)/2} (t/√ν)^{2j+1}
    """

    @staticmethod
    def cdf(t: float, df: float, delta: float = 0.0,
            max_iter: int = 200, tol: float = 1e-12) -> Tuple[float, int]:
        """
        计算非中心t分布的累积分布函数。

        使用Lenth算法 (AS 243) 的简化实现。

        Parameters
        ----------
        t : float
            分位点
        df : float
            自由度
        delta : float
            非中心参数
        max_iter : int
            最大迭代次数
        tol : float
            收敛容差

        Returns
        -------
        prob : float
            P(T ≤ t)
        ifault : int
            0=成功
        """
        if df <= 0.0:
            return 0.0, 1

        ifault = 0
        t = float(t)
        df = float(df)
        delta = float(delta)

        # 特殊情况
        if df > 1e6:
            # 近似为正态分布
            from scipy.stats import norm
            prob = norm.cdf(t - delta)
            return prob, ifault

        # 使用不完全beta函数的级数表示
        # P(T ≤ t) = Φ(-δ) + 混合beta函数项
        x = df / (df + t ** 2)
        if t < 0.0:
            sign_t = -1.0
        else:
            sign_t = 1.0

        # 简化实现: 使用scipy的ncx2和正态近似
        try:
            from scipy.stats import nct
            prob = nct.cdf(t, df, delta)
        except Exception:
            # fallback: 正态近似
            from scipy.stats import norm
            approx_mean = delta * np.sqrt(df / 2.0) * np.exp(
                gammaln((df - 1) / 2.0) - gammaln(df / 2.0)
            ) if df > 1 else delta
            approx_var = df * (1.0 + delta ** 2) / (df - 2.0) if df > 2 else 1.0
            prob = norm.cdf((t - approx_mean) / np.sqrt(approx_var))

        return prob, ifault

    @staticmethod
    def quantile_search(p: float, df: float, delta: float = 0.0,
                        a: float = -20.0, b: float = 20.0,
                        tol: float = 1e-6) -> Tuple[float, int]:
        """
        使用二分查找搜索非中心t分布的分位数。

        Parameters
        ----------
        p : float
            目标概率
        df : float
            自由度
        delta : float
            非中心参数
        a, b : float
            搜索区间
        tol : float
            容差

        Returns
        -------
        t_p : float
            分位数
        iters : int
            迭代次数
        """
        if p <= 0.0:
            return a, 0
        if p >= 1.0:
            return b, 0

        fa = NonCentralTDistribution.cdf(a, df, delta)[0]
        fb = NonCentralTDistribution.cdf(b, df, delta)[0]

        # 检查区间有效性
        if fa > p or fb < p:
            # 扩展区间
            while fa > p:
                a *= 2.0
                fa = NonCentralTDistribution.cdf(a, df, delta)[0]
            while fb < p:
                b *= 2.0
                fb = NonCentralTDistribution.cdf(b, df, delta)[0]

        for it in range(1, 200):
            c = (a + b) / 2.0
            fc = NonCentralTDistribution.cdf(c, df, delta)[0]

            if abs(fc - p) < tol or (b - a) < tol:
                return c, it

            if (fa - p) * (fc - p) < 0.0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc

        return (a + b) / 2.0, 200


class BisectionInteger:
    """
    整数域上的二分查找。
    """

    @staticmethod
    def find_root(f, a: int, b: int, max_iter: int = 100) -> Tuple[int, int]:
        """
        在整数区间[a,b]上查找f(c) = 0的根。

        Parameters
        ----------
        f : Callable
            整数函数
        a, b : int
            搜索区间
        max_iter : int
            最大迭代次数

        Returns
        -------
        c : int
            找到的根
        iters : int
            迭代次数
        """
        fa = f(a)
        fb = f(b)

        if fa == 0:
            return a, 0
        if fb == 0:
            return b, 0

        if fa * fb > 0:
            # 同号，返回最近零点的点
            if abs(fa) < abs(fb):
                return a, 0
            else:
                return b, 0

        for it in range(1, max_iter + 1):
            c = (a + b) // 2
            if c == a or c == b:
                return c, it

            fc = f(c)
            if fc == 0:
                return c, it

            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc

        return (a + b) // 2, max_iter


class QGPStatisticalInference:
    """
    QGP实验数据的统计推断。
    """

    def __init__(self):
        pass

    def centrality_significance(self, n_part_observed: float,
                                n_part_mean: float,
                                n_part_std: float,
                                df: float = 10.0) -> Tuple[float, float]:
        """
        检验观测到的参与者数是否显著偏离均值。

        使用非中心t统计量:
        t = (N_part - μ) / (σ/√df)

        Parameters
        ----------
        n_part_observed : float
            观测参与者数
        n_part_mean : float
            平均参与者数
        n_part_std : float
            标准差
        df : float
            自由度

        Returns
        -------
        t_stat : float
            t统计量
        p_value : float
            双尾p值
        """
        if n_part_std < 1e-15:
            return 0.0, 1.0
        t_stat = (n_part_observed - n_part_mean) / (n_part_std / np.sqrt(df))
        delta = 0.0  # 零假设下非中心参数为0
        prob, _ = NonCentralTDistribution.cdf(abs(t_stat), df, delta)
        p_value = 2.0 * (1.0 - prob)
        return float(t_stat), float(p_value)

    def v2_significance(self, v2_observed: float,
                        v2_stat_error: float,
                        v2_systematic: float = 0.0) -> Tuple[float, float]:
        """
        检验椭圆流信号的非零显著性。

        考虑系统误差时，使用非中心t分布:
        t = v₂ / √(σ_stat² + σ_sys²)

        Parameters
        ----------
        v2_observed : float
            观测v₂
        v2_stat_error : float
            统计误差
        v2_systematic : float
            系统误差

        Returns
        -------
        t_stat : float
            t统计量
        significance : float
            显著性 (σ)
        """
        total_err = np.sqrt(v2_stat_error ** 2 + v2_systematic ** 2)
        if total_err < 1e-15:
            return 0.0, 0.0
        t_stat = v2_observed / total_err
        # 使用大自由度近似
        from scipy.stats import norm
        p_value = 2.0 * (1.0 - norm.cdf(abs(t_stat)))
        significance = abs(t_stat)
        return float(t_stat), float(significance)

    def critical_temperature_confidence(self, T_measured: float,
                                        T_error: float,
                                        confidence: float = 0.95) -> Tuple[float, float]:
        """
        计算相变临界温度的置信区间。

        CI = [T̂ - t_{α/2}·σ, T̂ + t_{α/2}·σ]

        Parameters
        ----------
        T_measured : float
            测量温度 [GeV]
        T_error : float
            测量误差 [GeV]
        confidence : float
            置信水平

        Returns
        -------
        T_lower : float
            下限
        T_upper : float
            上限
        """
        if T_error < 0.0:
            T_error = abs(T_error)
        alpha = 1.0 - confidence
        df = 30.0  # 假设自由度

        # 查找t分位数
        t_lower, _ = NonCentralTDistribution.quantile_search(
            alpha / 2.0, df, delta=0.0
        )
        t_upper, _ = NonCentralTDistribution.quantile_search(
            1.0 - alpha / 2.0, df, delta=0.0
        )

        margin = t_upper * T_error
        T_lower = T_measured - margin
        T_upper = T_measured + margin
        return float(T_lower), float(T_upper)

    def find_critical_point(self, temperatures: np.ndarray,
                            susceptibility: np.ndarray) -> Tuple[float, float]:
        """
        使用二分查找定位相变临界点 (热 susceptibility 峰值)。

        Parameters
        ----------
        temperatures : np.ndarray
            温度网格
        susceptibility : np.ndarray
            对应的热 susceptibility

        Returns
        -------
        T_c : float
            临界温度
        chi_max : float
            最大susceptibility
        """
        idx_max = np.argmax(susceptibility)
        chi_max = susceptibility[idx_max]
        T_c = temperatures[idx_max]

        # 使用插值精化
        if 0 < idx_max < len(temperatures) - 1:
            # 二次插值
            T0, T1, T2 = temperatures[idx_max - 1:idx_max + 2]
            C0, C1, C2 = susceptibility[idx_max - 1:idx_max + 2]
            # 拟合抛物线
            A = np.array([
                [T0 ** 2, T0, 1.0],
                [T1 ** 2, T1, 1.0],
                [T2 ** 2, T2, 1.0]
            ])
            try:
                coeffs = np.linalg.solve(A, [C0, C1, C2])
                a, b, c = coeffs
                if abs(a) > 1e-15:
                    T_vertex = -b / (2.0 * a)
                    if T0 <= T_vertex <= T2:
                        T_c = T_vertex
                        chi_max = a * T_vertex ** 2 + b * T_vertex + c
            except np.linalg.LinAlgError:
                pass

        return float(T_c), float(chi_max)
